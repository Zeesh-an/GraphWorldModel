#!/bin/bash
#SBATCH --job-name=cp_conditions
#SBATCH --output=logs/%x_%j.out
# (logs/ must exist before `sbatch`: Slurm opens the .out file before the job runs)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=96:00:00

# The six-condition ladder, plus the two published repos that are wired.
#
# Usage: sbatch sbatch/cascade_prediction/sweep_conditions.sbatch
#        DATASET=casflow_aps sbatch sbatch/cascade_prediction/sweep_conditions.sbatch
#
# research/cascade_prediction.md 2.1 warns that this task's ACTION space is empty,
# `a_t` is NULL at every step, none of the five ops fire, and there is no planning
# regret because there is no decision to plan over. It then concludes that
# conditions 3-6 "collapse into one arm". **That is half right and the half that is
# wrong is why this script exists.** What varies down that ladder here is not the
# intervention but the FORWARD MODEL the predictor may call:
#
#   3 @native       no forward model at all. Feature-driven and point-process
#                   predictors only, and 3.1 records that feature models sometimes
#                   BEAT deep ones on these corpora, so this arm may win outright
#   4 @monte_carlo  NDlib rollouts, at steps x samples x mc_runs real episodes per
#                   forecast_marginals call. Whether it completes is itself a finding
#   5 @oracle       the analytic IC form, which is NOT the true dynamics here, and
#                   that is the entire point (9.1)
#   6 @world_model  one batched matmul per kernel evaluation
#
# --mc-agreement adds the number 9.1 is actually about: the MODELLING error of each arm's
# own forward model with no program in the loop. The gap between that column and the
# arm's own error is what the SEARCH bought; the LEVEL is how far an
# Independent-Cascade-shaped kernel is from a real adoption process. Expect to lose
# (9.2) and read the modelling column rather than the ranking.

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

# The wired published repos ride along as condition 7. Both are TensorFlow/PyTorch
# and both need `python -m baselines.setup_baselines --only <name>` first; a missing
# install is SKIPPED with a marker rather than failing the sweep.
env TASK=cascade_prediction DATASET="$DATASET" RUN=conditions RUN_JOBID=0 \
    GEN_MODELS="IC" DIFFUSION_MODEL=IC \
    CP_SPLIT=chronological \
    BASELINES="all-classical external:casflow external:ctcp" \
    ./sbatch/pipeline.sbatch

echo "=== done: results/cascade_prediction/$DATASET/conditions/report.md ==="
