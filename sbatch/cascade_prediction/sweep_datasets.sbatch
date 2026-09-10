#!/bin/bash
#SBATCH --job-name=cp_datasets
#SBATCH --output=logs/%x_%j.out
# (logs/ must exist before `sbatch`: Slurm opens the .out file before the job runs)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00
# The ladder trains f_theta inline in this allocation (pipeline.sbatch does
# not resubmit from inside a job), so the sweep needs the GPU itself
#SBATCH --gres=gpu:1

# Every REAL cascade corpus we load, in ascending order of what a loss would mean.
#
# Usage: sbatch sbatch/cascade_prediction/sweep_datasets.sbatch
#
# research/cascade_prediction.md 6.5 ranks these, and the order below is that
# ranking rather than alphabetical:
#
#   1. taoke        auto-downloads, small, and the only corpus here with a REAL
#                   logged transmission edge. Warning: 6.5 lists it under "not
#                   recommended" (too new, single-paper) and it is FRONT-LOADED,
#                   85% of its items launch within 14 minutes, so it runs under
#                   the RANDOM split, which is the only protocol it supports. Smoke
#                   test, not a result.
#   2. digg_cascades auto-downloads, real social graph, real vote times. Its windows
#                   are OURS rather than published (6.2 classifies Digg as a
#                   microscopic corpus with no clock), so its row is self-contained.
#   3. casflow_aps  THE ONE TO DO (6.5): no rate limits, no registration, and 3y/5y
#                   windows long enough that one replayed step is a whole year. Needs
#                   the manual Drive bundle; the loader prints the steps.
#   4. casflow_weibo the corpus carrying almost every published number (Weibo-A)
#   5. casflow_twitter the canonical host is 404 and this mirror is the only route
#   6. memetracker  1.3 GB per month and OUR extraction rather than Topo-LSTM's,
#                   so it runs last and its row says so
#
# 9.9: do not chase the leaderboard. CasFlow is a 2M-parameter model tuned for this
# one task; the value of every row here is diagnostic (9.1), not competitive.

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

run_dataset () {
    local dataset="$1"; shift
    echo "=== $dataset ==="
    env TASK=cascade_prediction DATASET="$dataset" RUN=default RUN_JOBID=0 \
        GEN_MODELS="IC" DIFFUSION_MODEL=IC "$@" \
        ./sbatch/pipeline.sbatch || echo "[cp_datasets] $dataset FAILED, continuing"
}

# final_results_plan.md section 4.8, in rank order. CP_MAX_NODES deletes
# adopters from every cascade and the size filter is re-applied after it, so a
# capped corpus keeps fewer cascades than the uncapped one and is comparable
# only to itself (§8.4): one cap per corpus, stated in the caption. casflow_aps
# runs uncapped at 128G first (its August run needed a cap to fit in 64G during
# the agent stage). taoke needs CP_MIN_SIZE=3 for a leak-free split, and its
# file has to be obtained by hand (the only source returned 404 on 2026-09-06).
run_dataset casflow_weibo   CP_SPLIT=chronological CP_MAX_NODES=50000 MEM=128G
run_dataset casflow_aps     CP_SPLIT=chronological MEM=128G
run_dataset taoke           CP_SPLIT=chronological CP_MIN_SIZE=3
run_dataset casflow_twitter CP_SPLIT=chronological CP_MAX_NODES=50000 MEM=128G
run_dataset digg_cascades   CP_SPLIT=chronological

echo "=== done: results/cascade_prediction/<corpus>/default/report.md ==="
