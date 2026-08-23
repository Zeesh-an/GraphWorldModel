#!/bin/bash
#SBATCH --job-name=gen_in_nethept
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Usage: sbatch sbatch/influence_maximization/generate_data_nethept.sbatch

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
# results/<task>/<dataset>/<run>/data — which is what
# the train/agent scripts downstream expect to be told about.
TASK="${TASK:-influence_maximization}"
DATASET=nethept
RUN="${RUN:-default}"
RUN_JOBID="${RUN_JOBID:-1}"
[ "$RUN_JOBID" = "1" ] && [ -n "${SLURM_JOB_ID:-}" ] && RUN="${RUN}_${SLURM_JOB_ID}"
OUT="results/$TASK/$DATASET/$RUN/data"

# NetHEPT citation network (15,229 nodes, 62,752 directed edges) — the standard
# IM scale benchmark. Auto-downloads on first use. ONE real graph: 50 rollouts
# x 2 models x 3 algos = 300 episodes (fewer rollouts than the small graphs —
# each 15K-node episode with 30 marginal draws per step is expensive).
#
# NO betweenness: exact Brandes is O(V*E) ~ 10^9 in pure networkx at this size
# (hours). random/degree/pagerank still span diverse seed sets.

# k ~ U(1%, 20% of N) per episode -> 152..3,046 seeds here. One checkpoint covers
# the whole budget sweep instead of only the k it was generated at. The high end
# is the expensive one: 3K seeds saturate this graph fast, so most of the cost is
# the 30 marginal draws per step, not the horizon.

# Node ops only, matching the BA recipe.
# All-five alternative: --action-ops add_node remove_node add_edge remove_edge set_edge_weight

export OUT

python -m data.generate_wm_data \
    --dataset nethept \
    --models IC LT --prob-model weighted --budget-pct-range 1 20 \
    --algorithms random degree pagerank \
    --rollouts 50 --horizon 10 --inject-p 0.4 --cf-prob 0.4 --cf-branches 2 \
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

python -c "import json, os; print('action_ops:', json.load(open(os.environ['OUT'] + '/metadata.json'))['config']['action_ops'])"

echo "[gen_data] done -> $OUT"
echo "[gen_data] train on it with: sbatch sbatch/wm_train.sbatch sage IC $OUT"
