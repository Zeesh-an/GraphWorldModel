#!/bin/bash
#SBATCH --job-name=gen_aoim_plc
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Usage: sbatch sbatch/adaptive_online_im/generate_data_powerlaw_cluster.sbatch

set -euo pipefail
cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p results logs

TASK="${TASK:-adaptive_online_im}"
DATASET=powerlaw_cluster
RUN="${RUN:-default}"
RUN_JOBID="${RUN_JOBID:-1}"
[ "$RUN_JOBID" = "1" ] && [ -n "${SLURM_JOB_ID:-}" ] && RUN="${RUN}_${SLURM_JOB_ID}"
OUT="results/$TASK/$DATASET/$RUN/data"

# RL4IM's family and the only synthetic one in this literature with a matching
# published protocol: BA growth + a triangle-closing step, avg degree ~2m,
# triangle probability 0.05 (research/adaptive_online_im.md §6.3b). RL4IM quotes
# avg degree 3, which this generator cannot hit at integer m; --plc-m 2 is the
# nearest above and the discrepancy is recorded in §11.
#
# 200 nodes matches RL4IM's default |V|. 40 graphs x 20 rollouts = 800 episodes
# per (model, algorithm).
#
# NOTE the data is IDENTICAL to what influence_maximization generates — adaptive
# IM changes the evaluation loop, not the transitions (§9.2). A checkpoint
# trained on an IM run of the same graph family is reusable here via
# --wm-results-json; this script exists so the results tree matches the task.

export OUT

python -m data.generate_wm_data \
    --dataset powerlaw_cluster --num-graphs 40 --syn-nodes 200 \
    --plc-m 2 --plc-p 0.05 \
    --models IC LT --prob-model weighted --budget-pct-range 1 20 \
    --algorithms random degree pagerank betweenness \
    --rollouts 20 --horizon 10 --inject-p 0.4 --cf-prob 0.4 --cf-branches 2 \
    --action-ops add_node remove_node \
    --weight-lo 0.0 --weight-hi 1.0 \
    --mc-marginals 30 \
    --split 0.7 0.15 0.15 --seed 42 --split-mode graph_disjoint \
    --out-dir "$OUT"

python -m data.validate_wm_data --dir "$OUT"

echo "[gen_data] done -> $OUT"
echo "[gen_data] train on it with: sbatch sbatch/adaptive_online_im/train.sbatch sage IC $OUT"
