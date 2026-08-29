#!/bin/bash
#SBATCH --job-name=epi_dynamics
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

# The three compartmental models, each with its own dataset, checkpoint and run
# directory. This is the axis research/epidemic_control.md §9.1 puts in order:
# "(a) SIR only: it is the smallest departure. (b) The CompartmentTransitionHead.
# (c) SIS second: it breaks 'ever-infected' entirely and needs the endemic-
# prevalence metric. (d) SEIR last, or never."
#
# Usage: sbatch sbatch/epidemic_control/sweep_dynamics.sbatch
#        DATASET=primary_school DYNAMICS="SIR SIS" sbatch ...
#
# Read the three runs against each other on ONE column and it is the wrong one.
# §8.2 trap 6: SIS HAS NO TERMINAL STATE, so its attack rate is a cumulative
# incidence that grows with the horizon and is not comparable to SIR's final size.
# The comparable column across all three is `epi_endemic_prevalence` in
# summary.csv, and for SIR and SEIR it is near zero by construction. The report
# prints that warning itself on an SIS run.
#
# SEIR's extra latent compartment adds no new mechanism: §9.1 says so outright,
# so it is here for completeness rather than as a result to lead with.

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
DYNAMICS="${DYNAMICS:-SIR SIS SEIR}"

for model in $DYNAMICS; do
    echo "=== $model ==="
    TASK=epidemic_control \
    DATASET="$DATASET" \
    RUN="dyn_$(echo "$model" | tr "A-Z" "a-z")" \
    RUN_JOBID=0 \
    COMPARE=1 \
    GEN_MODELS="$model" \
    DIFFUSION_MODEL="$model" \
    EPI_GAMMA="${EPI_GAMMA:-0.3}" \
    BUDGET_PCTS="${BUDGET_PCTS:-1 5 10 20}" \
        ./sbatch/pipeline.sbatch || echo "[epi_dynamics] $model FAILED, continuing"
done

echo "[epi_dynamics] done -> results/epidemic_control/$DATASET/dyn_*"
