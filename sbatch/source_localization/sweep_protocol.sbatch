#!/bin/bash
#SBATCH --job-name=sl_protocol
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

# Every protocol axis research/source_localization.md §8.5.1 asks for, on ONE
# graph, each into its own results/source_localization/<dataset>/<run>/.
#
# Usage: sbatch sbatch/source_localization/sweep_protocol.sbatch
#        DATASET=jazz sbatch sbatch/source_localization/sweep_protocol.sbatch
#
# The dataset sweep answers "does this work". This one answers "does the number
# mean what the table says it means", which is where an invalid comparison
# actually gets published. Four runs:
#
#   marginal      the default. Our y is averaged over --mc-marginals draws, which
#                 is strictly MORE informative than the single binary realization
#                 SL-VAE observes, so this column is OPTIMISTIC against §5.1
#   binary        the same everything, on a binarized single draw. THIS is the
#                 column comparable to the published tables (§2.9 risk 5), and
#                 reporting only the first one is the version a reviewer will
#                 correctly disbelieve
#   fraction      the source-fraction axis: k fixed by the sweep at 5 / 10 / 20%
#                 of N, with the episode pool filtered to match, which tests
#                 whether a program is budget-brittle
#   dynamics      selected under IC, evaluated under LT. The strongest of the
#                 optional axes: a program that only works under IC has encoded
#                 IC-specific structure rather than an inference rule
#
# The GRAPH axis: the headline, and the one no per-instance method can enter,
# is not here because it needs a finished run to transfer FROM. Run the dataset
# sweep first, then per (source, target) pair:
#
#   TASK=source_localization DATASET=cora_ml RUN=transfer_from_jazz \
#   ARMS=none BASELINES=none \
#   SL_TRANSFER_FROM=results/source_localization/jazz/default/agent/pct10/evolve_free@world_model.json \
#       ./sbatch/pipeline.sbatch

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

DATASET="${DATASET:-jazz}"

run_axis () {
    local label="$1"; shift
    echo "=== $label ==="
    env TASK=source_localization DATASET="$DATASET" RUN="$label" RUN_JOBID=0 \
        COMPARE=1 GEN_MODELS="IC LT" "$@" \
        ./sbatch/pipeline.sbatch || echo "[sl_protocol] $label FAILED, continuing"
}

# 1. The default observation: continuous MC marginals
run_axis observation_marginal SL_OBSERVATION=marginal DIFFUSION_MODEL=IC

# 2. The comparable one: a binarized single draw
run_axis observation_binary SL_OBSERVATION=binary DIFFUSION_MODEL=IC

# 3. Source fraction. `sweep` mode forces the pipeline's k on every instance AND
#    filters the pool to episodes near it, so a row labelled "5% of N" really is
#    about 5%-of-N sources rather than whatever the generator drew
run_axis source_fraction \
    SL_BUDGET_MODE=sweep BUDGET_PCTS="5 10 20" \
    SL_SOURCE_TOLERANCE=0.5 DIFFUSION_MODEL=IC

# 4. Dynamics transfer. Each axis is its own run, so its data stage generates
#    both dynamics again and only the LT file is read
run_axis dynamics_lt DIFFUSION_MODEL=LT

echo "[sl_protocol] done -> results/source_localization/$DATASET/"
