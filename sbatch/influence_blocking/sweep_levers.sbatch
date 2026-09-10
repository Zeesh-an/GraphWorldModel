#!/bin/bash
#SBATCH --job-name=ib_levers
#SBATCH --output=logs/%x_%j.out
# (logs/ must exist before `sbatch`: Slurm opens the .out file before the job runs)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00
# The ladder trains f_theta inline in this allocation (pipeline.sbatch does
# not resubmit from inside a job), so the sweep needs the GPU itself
#SBATCH --gres=gpu:1

# The four intervention levers of research/influence_blocking.md §1.1, back to
# back on one graph, each into its own run directory.
#
# Usage: sbatch sbatch/influence_blocking/sweep_levers.sbatch
#        DATASET=facebook sbatch sbatch/influence_blocking/sweep_levers.sbatch
#
# This is the experiment §7 says does not exist. The seeding line of this
# literature (Budak, CLDAG, RPS, NIE) and the structural line (Kimura, Xie,
# SandIMIN, DiffIM) share no dataset outside the SNAP staples and NO PAPER IN
# EITHER LINE REPORTS THE OTHER'S METRIC, so a single evaluator scoring
# add_node, remove_node, remove_edge and set_edge_weight interventions on the
# same graph, against the same rumour, under one metric is a comparison nobody
# has published.
#
# Every lever faces the SAME S_N (deterministic in SEED) and the same tie-break,
# so the only thing varying down the sweep is what the budget buys.

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
VENV="${VENV:-venv}"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
fi
mkdir -p results logs
# The matrix's synthetic rows are 10,000 nodes (final_results_plan.md section 2);
# the real graphs ignore this
export SYN_NODES="${SYN_NODES:-10000}"

DATASET="${DATASET:-email_eu_core}"
LEVERS="${LEVERS:-counter_seed node_block edge_block weight_block}"

for lever in $LEVERS; do
    echo "=== $lever ==="
    TASK=influence_blocking \
    DATASET="$DATASET" \
    RUN="lever_$lever" \
    RUN_JOBID=0 \
    GEN_MODELS=IC \
    BLOCKING_LEVER="$lever" \
    BUDGETS="${BUDGETS:-10 20 30 40 50}" \
        ./sbatch/pipeline.sbatch || echo "[ib_levers] $lever FAILED, continuing"
done

echo "[ib_levers] done -> results/influence_blocking/$DATASET/lever_*"
