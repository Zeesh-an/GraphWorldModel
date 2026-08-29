#!/bin/bash
#SBATCH --job-name=gen_aoim_plc
#SBATCH --output=logs/%x_%j.out
# (logs/ must exist before `sbatch`: Slurm opens the .out file before the job runs)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Usage: sbatch sbatch/adaptive_online_im/generate_data_powerlaw_cluster.sbatch

set -euo pipefail
# Repo root: the checkout this file lives in, whenever this file is the real
# one. Under a Slurm spool copy BASH_SOURCE points outside any checkout, and
# only then is the submit directory trusted; an `srun --pty` / `salloc` shell
# also carries SLURM_JOB_ID, so SLURM_SUBMIT_DIR is never taken first. Set
# REPO_ROOT to override. Same rule as sbatch/pipeline.sbatch.
if [ -z "${REPO_ROOT:-}" ]; then
    checkout="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd || true)"
    if [ -n "$checkout" ] && [ -f "$checkout/pipeline/run.py" ]; then
        REPO_ROOT="$checkout"
    elif [ -n "${SLURM_JOB_ID:-}" ] && [ -f "${SLURM_SUBMIT_DIR:-/nonexistent}/pipeline/run.py" ]; then
        REPO_ROOT="$SLURM_SUBMIT_DIR"
    else
        echo "[repo] cannot locate the checkout: ${BASH_SOURCE[0]} is not inside one and" \
             "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset} has no pipeline/run.py; set REPO_ROOT" >&2
        exit 1
    fi
fi
export REPO_ROOT
cd "$REPO_ROOT"
[ -f pipeline/run.py ] || { echo "[repo] REPO_ROOT=$REPO_ROOT has no pipeline/run.py; set REPO_ROOT to the checkout" >&2; exit 1; }
[ -w . ] || { echo "[repo] REPO_ROOT=$REPO_ROOT is not writable, and results/ and logs/ live here" >&2; exit 1; }
echo "[repo] $REPO_ROOT"

# Activate the project virtualenv, unless one is already active or VENV
# points somewhere else. Skipped silently when neither exists, so the
# script still runs under a module-loaded or conda interpreter.
VENV="${VENV:-venv}"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
fi
mkdir -p results logs

TASK="${TASK:-adaptive_online_im}"
DATASET=powerlaw_cluster
RUN="${RUN:-default}"
RUN_JOBID="${RUN_JOBID:-1}"
[ "$RUN_JOBID" = "1" ] && [ -n "${SLURM_JOB_ID:-}" ] && RUN="${RUN}_${SLURM_JOB_ID}"
OUT="results/$TASK/$DATASET/$RUN/data"

# RL4IM's family and the only synthetic one in this literature with a matching
# published protocol: BA growth + a triangle-closing step, triangle probability
# 0.05 (research/adaptive_online_im.md §6.3b). m=3 and p=0.05 are RL4IM's OWN
# basic_env.yaml values, read from the repo; its paper says "average degree 3",
# which no integer m produces, so the code wins. See §11.
#
# 200 nodes matches RL4IM's default |V|. 40 graphs x 20 rollouts = 800 episodes
# per (model, algorithm).
#
# NOTE the data is IDENTICAL to what influence_maximization generates: adaptive
# IM changes the evaluation loop, not the transitions (§9.2). A checkpoint
# trained on an IM run of the same graph family is reusable here via
# --wm-results-json; this script exists so the results tree matches the task.

export OUT

python -m data.generate_wm_data \
    --dataset powerlaw_cluster --num-graphs 40 --syn-nodes 200 \
    --plc-m 3 --plc-p 0.05 \
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
