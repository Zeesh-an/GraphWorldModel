#!/bin/bash
#SBATCH --job-name=gen_in_netscience
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Usage: sbatch sbatch/influence_maximization/generate_data_netscience.sbatch

set -euo pipefail
cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p results logs

# Results land in results/<task>/<dataset>/<run>_<jobid>/data, the same way logs land in
# logs/%x_%j.out, so two generation jobs can never half-overwrite each other's
# transitions. Set RUN_JOBID=0 for a stable
# results/<task>/<dataset>/<run>/data — which is what
# the train/agent scripts downstream expect to be told about.
TASK="${TASK:-influence_maximization}"
DATASET=netscience
RUN="${RUN:-default}"
RUN_JOBID="${RUN_JOBID:-1}"
[ "$RUN_JOBID" = "1" ] && [ -n "${SLURM_JOB_ID:-}" ] && RUN="${RUN}_${SLURM_JOB_ID}"
OUT="results/$TASK/$DATASET/$RUN/data"

# NetScience coauthorship network (1,589 nodes, 2,742 edges, heavily fragmented:
# 128 isolates + many small components — budget allocation across components is
# the decision that matters). Auto-downloads on first use. ONE real graph, so
# episode diversity comes from rollouts: 100 x 2 models x 4 algos = 800 episodes.

# k ~ U(1%, 20% of N) per episode -> 16..318 seeds here. One checkpoint covers
# the whole budget sweep instead of only the k it was generated at.

# Node ops only, matching the BA recipe.
# All-five alternative: --action-ops add_node remove_node add_edge remove_edge set_edge_weight

export OUT

python -m data.generate_wm_data \
    --dataset netscience \
    --models IC LT --prob-model weighted --budget-pct-range 1 20 \
    --algorithms random degree pagerank betweenness \
    --rollouts 100 --horizon 10 --inject-p 0.4 --cf-prob 0.4 --cf-branches 2 \
    --action-ops add_node remove_node \
    --weight-lo 0.0 --weight-hi 1.0 \
    --mc-marginals 30 \
    --split 0.7 0.15 0.15 --seed 42 \
    --out-dir "$OUT"

python -m data.validate_wm_data --dir "$OUT"

python -c "import json, os; print('action_ops:', json.load(open(os.environ['OUT'] + '/metadata.json'))['config']['action_ops'])"

echo "[gen_data] done -> $OUT"
echo "[gen_data] train on it with: sbatch sbatch/wm_train.sbatch sage IC $OUT"
