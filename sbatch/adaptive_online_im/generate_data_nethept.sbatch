#!/bin/bash
#SBATCH --job-name=gen_aoim_nethept
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Usage: sbatch sbatch/adaptive_online_im/generate_data_nethept.sbatch

set -euo pipefail
cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p results logs

TASK="${TASK:-adaptive_online_im}"
DATASET=nethept
RUN="${RUN:-default}"
RUN_JOBID="${RUN_JOBID:-1}"
[ "$RUN_JOBID" = "1" ] && [ -n "${SLURM_JOB_ID:-}" ] && RUN="${RUN}_${SLURM_JOB_ID}"
OUT="results/$TASK/$DATASET/$RUN/data"

# THE graph for this task: Han et al. (PVLDB 2018) and Lei et al. (KDD 2015) both
# report adaptive/online IM on NetHEPT, and it is byte-comparable to ours
# (research/adaptive_online_im.md §6.1). 15,229 nodes / 62,752 arcs.
#
# ONE real graph, so episode diversity comes from rollouts rather than graph
# count: 60 rollouts x 2 models x 4 selectors = 480 episodes. Budget the wall
# clock — 30 MC marginals on a 15K-node graph is the expensive part.

export OUT

python -m data.generate_wm_data \
    --dataset nethept \
    --models IC LT --prob-model weighted --budget-pct-range 1 20 \
    --algorithms random degree pagerank betweenness \
    --rollouts 60 --horizon 10 --inject-p 0.4 --cf-prob 0.4 --cf-branches 2 \
    --action-ops add_node remove_node \
    --weight-lo 0.0 --weight-hi 1.0 \
    --mc-marginals 30 \
    --split 0.7 0.15 0.15 --seed 42 \
    `# SINGLE REAL GRAPH: a graph-disjoint split is impossible here, so this
     # dataset is split PER EPISODE and its test metrics are in-graph, not
     # held-out-graph. Do not pool them with a graph_disjoint synthetic run.
     # The leakage-free use of a real graph is as a TRANSFER TEST set for a
     # model trained on synthetic families.` \
    --split-mode episode_random \
    --out-dir "$OUT"

python -m data.validate_wm_data --dir "$OUT"

echo "[gen_data] done -> $OUT"
echo "[gen_data] train on it with: sbatch sbatch/adaptive_online_im/train.sbatch sage IC $OUT"
