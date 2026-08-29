#!/bin/bash
#SBATCH --job-name=sl_gen_jazz
#SBATCH --output=logs/%x_%j.out
# (logs/ must exist before `sbatch`: Slurm opens the .out file before the job runs)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00

# Labelled source-localization episodes on Jazz (198 / 2,742).
#
# Usage: sbatch sbatch/source_localization/generate_data_jazz.sbatch
#
# Jazz is the single cleanest comparison target in this literature: FOUR papers
# report it at 198 / 2,742 to the digit: IVGD Table 2, SL-VAE Table 5, GraphSL
# Table 1, SIDSL Table 5 (research/source_localization.md §6.1), and it is small
# enough that a full six-condition ladder finishes in hours.
#
# --task supplies the whole data shape from the registry, so no --gen-action-ops
# flag appears here: a recover task generates DIFFUSION-ONLY episodes, with no
# injected actions at any t > 0. That is not a tuning choice. An episode carrying
# a mid-cascade injection has an observation its t=0 seed set did not produce, so
# its (x, y) label would be a lie (§2.1), and the registry asserts it.
#
# INJECT_P and the counterfactual forks are inert for the same reason (empty
# action ops means sample_injection always returns NULL and counterfactual_actions
# produces nothing), so they are left at their defaults rather than zeroed: the
# registry is the single place that decides.
#
# BUDGET_PCT_RANGE is the SOURCE fraction band, which is the one generation knob
# this task genuinely cares about. 1-20% brackets SL-VAE's 10%-of-nodes
# convention (§8.4 trap 1) on both sides, so --sl-budget-mode sweep can later cut
# a 5 / 10 / 20% source-fraction axis out of one dataset.

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

TASK=source_localization \
DATASET=jazz \
RUN="${RUN:-default}" \
RUN_JOBID=0 \
START_STAGE=data END_STAGE=data \
GEN_MODELS="${GEN_MODELS:-IC LT}" \
GEN_ALGORITHMS="${GEN_ALGORITHMS:-random degree pagerank betweenness}" \
ROLLOUTS="${ROLLOUTS:-60}" \
MC_MARGINALS="${MC_MARGINALS:-30}" \
GEN_HORIZON="${GEN_HORIZON:-10}" \
BUDGET_PCT_RANGE="${BUDGET_PCT_RANGE:-1 20}" \
    ./sbatch/pipeline.sbatch

echo "[sl_gen] done -> results/source_localization/jazz/${RUN:-default}/data"
