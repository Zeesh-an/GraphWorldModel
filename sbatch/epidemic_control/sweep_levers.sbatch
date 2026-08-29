#!/bin/bash
#SBATCH --job-name=epi_levers
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

# The four intervention levers of research/epidemic_control.md §2.5, back to back
# on one graph, each into its own run directory.
#
# Usage: sbatch sbatch/epidemic_control/sweep_levers.sbatch
#        DATASET=primary_school sbatch sbatch/epidemic_control/sweep_levers.sbatch
#
# What this measures that no published table does: the immunization literature's
# three lines each own ONE lever and none of them reports another's. The physics
# and spectral lines vaccinate NODES (Pastor-Satorras, Cohen, NetShield, DAVA); the
# Van Mieghem / NetMelt / Kimura line cuts ARCS; the fractional line (Preciado,
# Fractional Immunization, DURLECA) reduces contact CONTINUOUSLY. One evaluator
# scoring all four against the same outbreak on the same graph under one metric is
# a comparison that does not currently exist.
#
# `vaccinate` vs `quarantine` is the sharpest pair and the cheapest to read: the
# SAME node set is chosen under both, and they differ only in whether the dosed
# node leaves the attack rate. §8.2 trap 7 is that papers reporting "nodes saved"
# against "no intervention" and against "random vaccination" differ by a large
# constant for exactly this reason.
#
# Every lever faces the SAME outbreak (deterministic in SEED) and the same rates,
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
source venv/bin/activate
mkdir -p results logs

DATASET="${DATASET:-hospital_lh10}"
LEVERS="${LEVERS:-vaccinate quarantine edge_cut contact_reduce}"

for lever in $LEVERS; do
    echo "=== $lever ==="
    TASK=epidemic_control \
    DATASET="$DATASET" \
    RUN="lever_$lever" \
    RUN_JOBID=0 \
    COMPARE=1 \
    GEN_MODELS=SIR \
    DIFFUSION_MODEL=SIR \
    EPI_LEVER="$lever" \
    BUDGET_PCTS="${BUDGET_PCTS:-1 5 10 20}" \
        ./sbatch/pipeline.sbatch || echo "[epi_levers] $lever FAILED, continuing"
done

echo "[epi_levers] done -> results/epidemic_control/$DATASET/lever_*"
