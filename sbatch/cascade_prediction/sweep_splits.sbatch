#!/bin/bash
#SBATCH --job-name=cp_splits
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

# THE HEADLINE EXPERIMENT of this task, and the one thing here worth running first.
#
# Usage: sbatch sbatch/cascade_prediction/sweep_splits.sbatch
#        DATASET=casflow_aps sbatch sbatch/cascade_prediction/sweep_splits.sbatch
#
# research/cascade_prediction.md 8.3 is the most transferable finding in that file:
# the field's standard 70/15/15 RANDOM-over-cascades split leaks the future.
# Cascades overlap in wall-clock time, so a training cascade's PREDICTION window can
# sit inside a test cascade's OBSERVATION window, and the model learns "there was a
# burst around time T": a global temporal shortcut unavailable at deployment. Under
# CasTemp's leak-free fix:
#
#   * CasFlow and CasDO fall BELOW a plain MLP on the diagnostic scenarios
#   * CasDO, a 2024 TKDE paper, becomes the WORST method on all four corpora
#   * the whole field's APS band moves from 1.19-2.11 to 2.28-4.82
#
# That is not a small correction, and it has not been independently replicated (11).
# These two runs replicate it in OUR harness on OUR corpus with OUR arms, which is
# the cheapest second source anyone can produce. Everything except the split is held
# fixed, so the difference between the two report tables IS the leak.
#
# 6 of this file also applies: our own transitions split by EPISODE rather than by
# time, which is structurally the same choice CasTemp is criticising. It is harmless
# on NDlib episodes (they share no clock) and stops being harmless the moment a real
# log is replayed, which is why `chronological` is the default here.

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

DATASET="${DATASET:-casflow_weibo}"

run_split () {
    local label="$1"; shift
    echo "=== $label ==="
    env TASK=cascade_prediction DATASET="$DATASET" RUN="$label" RUN_JOBID=0 \
        GEN_MODELS="IC" DIFFUSION_MODEL=IC "$@" \
        ./sbatch/pipeline.sbatch || echo "[cp_splits] $label FAILED, continuing"
}

# 1. The leak-free protocol. This is the number to report.
run_split split_chronological CP_SPLIT=chronological

# 2. The field's own protocol, which every published table in 5.1 and 5.2 uses.
#    Its numbers are the ones comparable to those tables, and NOT to run 1.
run_split split_random        CP_SPLIT=random

# 3-4. The same pair on the SECOND observation window. 8.2: every macroscopic paper
#      reports two windows so a reader can see whether an edge survives more
#      observation, and a single-window result is not publishable here. The values
#      are per corpus; unset takes the first published one, so this names the second
#      explicitly. Weibo 3600 = 1 h, Twitter 172800 = 2 d, APS 1826 = 5 y.
case "$DATASET" in
  casflow_weibo|weibo_cascades) SECOND=3600 ;;
  casflow_twitter)              SECOND=172800 ;;
  casflow_aps|aps)              SECOND=1826 ;;
  *)                            SECOND="" ;;
esac

if [ -n "$SECOND" ]; then
    run_split window2_chronological CP_SPLIT=chronological CP_OBSERVATION="$SECOND"
    run_split window2_random        CP_SPLIT=random        CP_OBSERVATION="$SECOND"
fi

echo "=== done: results/cascade_prediction/$DATASET/{split_chronological,split_random,...} ==="
