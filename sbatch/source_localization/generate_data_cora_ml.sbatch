#!/bin/bash
#SBATCH --job-name=sl_gen_cora_ml
#SBATCH --output=logs/%x_%j.out
# (logs/ must exist before `sbatch`: Slurm opens the .out file before the job runs)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Labelled source-localization episodes on Cora-ML (2,810 / 7,981).
#
# Usage: sbatch sbatch/source_localization/generate_data_cora_ml.sbatch
#
# One of the three graphs where every paper in this literature agrees to the
# digit: IVGD, SL-VAE and GraphSL all report 2,810 / 7,981, and our loader
# already applies graph2gauss `standardize()` so it reproduces that exactly
# (research/source_localization.md §6.4). The other two are Jazz and Power Grid.
#
# betweenness is dropped from the selectors: exact Brandes is O(V*E) in pure
# networkx and costs hours at 2.8K nodes for a spine that only has to be diverse.

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
DATASET=cora_ml \
RUN="${RUN:-default}" \
RUN_JOBID=0 \
START_STAGE=data END_STAGE=data \
GEN_MODELS="${GEN_MODELS:-IC LT}" \
GEN_ALGORITHMS="${GEN_ALGORITHMS:-random degree pagerank}" \
ROLLOUTS="${ROLLOUTS:-40}" \
MC_MARGINALS="${MC_MARGINALS:-30}" \
GEN_HORIZON="${GEN_HORIZON:-10}" \
BUDGET_PCT_RANGE="${BUDGET_PCT_RANGE:-1 20}" \
    ./sbatch/pipeline.sbatch

echo "[sl_gen] done -> results/source_localization/cora_ml/${RUN:-default}/data"
