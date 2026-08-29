#!/bin/bash
#SBATCH --job-name=epi_strength
#SBATCH --output=logs/%x_%j.out
# (logs/ must exist before `sbatch`: Slurm opens the .out file before the job runs)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=96:00:00
# The ladder trains f_theta inline in this allocation (pipeline.sbatch does
# not resubmit from inside a job), so the sweep needs the GPU itself
#SBATCH --gres=gpu:1

# The normalized virus strength `s = lambda_1 * beta / gamma` swept, which is the
# x-axis NetShield's own Fig 1 reports against and the ONLY honest way to present
# a result in this literature.
#
# Usage: sbatch sbatch/epidemic_control/sweep_virus_strength.sbatch
#        DATASET=primary_school sbatch sbatch/epidemic_control/sweep_virus_strength.sbatch
#
# research/epidemic_control.md §8.2 trap 2 is the reason this sweep exists rather
# than a single (beta, gamma) pair: "beta and gamma are free parameters and nobody
# standardizes them. NetShield reports against a normalized virus strength swept on
# the x-axis, which is the honest way; most others fix one pair and do not justify
# it. Any table that fixes beta is comparable only to itself."
#
# The sweep is over GAMMA at fixed beta, because gamma is the rate that decides
# whether the outbreak takes off at all: at gamma = 1.0 the process IS Independent
# Cascade (a node transmits once and is spent, which
# `coding_agent/check_epidemic_control.py` verifies against NDlib), and lowering it
# lengthens the infectious period and pushes `s` up. The lowest values put every
# arm in the regime §5.1 describes for Portland, where the epidemic wins almost
# everywhere and even the best policy leaves 3.7% healthy, which is the regime
# §5.1 says a world model has room in, and the highest put every arm in the regime
# where nothing was going to spread anyway and every method ties.

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
source venv/bin/activate
mkdir -p results logs

DATASET="${DATASET:-hospital_lh10}"
GAMMAS="${GAMMAS:-0.15 0.3 0.5 0.7 1.0}"

for gamma in $GAMMAS; do
    echo "=== gamma=$gamma ==="
    TASK=epidemic_control \
    DATASET="$DATASET" \
    RUN="gamma_${gamma}" \
    RUN_JOBID=0 \
    COMPARE=1 \
    GEN_MODELS=SIR \
    DIFFUSION_MODEL=SIR \
    EPI_GAMMA="$gamma" \
    EPI_BETA="${EPI_BETA:-1.0}" \
    BUDGET_PCTS="${BUDGET_PCTS:-5 10 20}" \
        ./sbatch/pipeline.sbatch || echo "[epi_strength] gamma=$gamma FAILED, continuing"
done

echo "[epi_strength] done -> results/epidemic_control/$DATASET/gamma_*"
