#!/bin/bash
#SBATCH --job-name=gen_in_sbm
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Usage: sbatch sbatch/influence_maximization/generate_data_sbm.sbatch

set -euo pipefail
# Repo root, derived from this script's own location so the file works in
# any checkout rather than one person's home directory. Set REPO_ROOT to
# override, e.g. when submitting a copy staged outside the tree.
cd "${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Activate the project virtualenv, unless one is already active or VENV
# points somewhere else. Skipped silently when neither exists, so the
# script still runs under a module-loaded or conda interpreter.
VENV="${VENV:-venv}"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
fi
mkdir -p results logs

# Results land in results/<task>/<dataset>/<run>_<jobid>/data, the same way logs land in
# logs/%x_%j.out, so two generation jobs can never half-overwrite each other's
# transitions. Set RUN_JOBID=0 for a stable
# results/<task>/<dataset>/<run>/data, which is what
# the train/agent scripts downstream expect to be told about.
TASK="${TASK:-influence_maximization}"
DATASET=sbm
RUN="${RUN:-default}"
RUN_JOBID="${RUN_JOBID:-1}"
[ "$RUN_JOBID" = "1" ] && [ -n "${SLURM_JOB_ID:-}" ] && RUN="${RUN}_${SLURM_JOB_ID}"
OUT="results/$TASK/$DATASET/$RUN/data"

# SBM-100 with 40 distinct graphs, 4 planted communities each (25 nodes per block,
# p_in 0.15 / p_out 0.01). Same scale/recipe as the BA run so the only variable is
# community structure: within-block degrees are near-uniform, so degree carries
# little seed signal and budget allocation across blocks becomes the real problem.

# Node ops only, matching the BA recipe.
# All-five alternative: --action-ops add_node remove_node add_edge remove_edge set_edge_weight

# k ~ U(1%, 20% of N) per episode -> 1..20 seeds here. One checkpoint covers the
# whole budget sweep instead of only the k it was generated at.

export OUT

python -m data.generate_wm_data \
    --dataset sbm --num-graphs 40 --syn-nodes 100 \
    --sbm-blocks 4 --sbm-p-in 0.15 --sbm-p-out 0.01 \
    --models IC LT --prob-model weighted --budget-pct-range 1 20 \
    --algorithms random degree pagerank betweenness \
    --rollouts 20 --horizon 10 --inject-p 0.4 --cf-prob 0.4 --cf-branches 2 \
    --action-ops add_node remove_node \
    --weight-lo 0.0 --weight-hi 1.0 \
    --mc-marginals 30 \
    --split 0.7 0.15 0.15 --seed 42 --split-mode graph_disjoint \
    --out-dir "$OUT"

python -m data.validate_wm_data --dir "$OUT"

python -c "import json, os; print('action_ops:', json.load(open(os.environ['OUT'] + '/metadata.json'))['config']['action_ops'])"

echo "[gen_data] done -> $OUT"
echo "[gen_data] train on it with: sbatch sbatch/wm_train.sbatch sage IC $OUT"
