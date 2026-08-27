#!/bin/bash
#SBATCH --job-name=gen_in_jazz
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Usage: sbatch sbatch/influence_maximization/generate_data_jazz.sbatch

set -euo pipefail
# Repo root, derived from this script's own location so the file works in
# any checkout rather than one person's home directory. Set REPO_ROOT to
# override, e.g. when submitting a copy staged outside the tree.
# Under SLURM this file executes as a spool copy, so BASH_SOURCE cannot locate
# the checkout from inside a job; the directory it was submitted from is the
# repo root when launched as documented, and REPO_ROOT overrides either.
if [ -z "${REPO_ROOT:-}" ]; then
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        REPO_ROOT="$SLURM_SUBMIT_DIR"
    else
        REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    fi
fi
cd "$REPO_ROOT"

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
DATASET=jazz
RUN="${RUN:-default}"
RUN_JOBID="${RUN_JOBID:-1}"
[ "$RUN_JOBID" = "1" ] && [ -n "${SLURM_JOB_ID:-}" ] && RUN="${RUN}_${SLURM_JOB_ID}"
OUT="results/$TASK/$DATASET/$RUN/data"

# Jazz collaboration network (198 nodes, 2,742 edges, real community structure).
# Auto-downloads on first use. ONE real graph, so episode diversity comes from
# rollouts instead of graph count: 100 rollouts x 2 models x 4 algos = 800 episodes
# (BA/SBM get 6,400 from 40 graphs x 20 rollouts).

# Node ops only, matching the BA recipe.
# All-five alternative: --action-ops add_node remove_node add_edge remove_edge set_edge_weight

# k ~ U(1%, 20% of N) per episode -> 2..40 seeds here. One checkpoint covers the
# whole budget sweep instead of only the k it was generated at.

export OUT

python -m data.generate_wm_data \
    --dataset jazz \
    --models IC LT --prob-model weighted --budget-pct-range 1 20 \
    --algorithms random degree pagerank betweenness \
    --rollouts 100 --horizon 10 --inject-p 0.4 --cf-prob 0.4 --cf-branches 2 \
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
