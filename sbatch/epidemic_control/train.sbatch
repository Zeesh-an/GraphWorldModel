#!/bin/bash
#SBATCH --job-name=epi_train
#SBATCH --output=logs/%x_%j.out
# (logs/ must exist before `sbatch`: Slurm opens the .out file before the job runs)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Fit the CompartmentTransitionHead on an already-generated compartmental dataset.
#
# Usage: sbatch sbatch/epidemic_control/train.sbatch
#        DATASET=primary_school DIFFUSION_MODEL=SIS sbatch sbatch/epidemic_control/train.sbatch
#
# `structured` is the repo default. HEAD=structured_residual is only expressible
# because we wrote our own stepper: it anchors `q` on the arc's own `beta_uv`, and
# research/epidemic_control.md §2.2 records that NDlib's SIR/SIS/SEIR carry NO
# per-edge parameter at all, so under them that head cannot exist.
#
# The three rates are read back OFF the dataset's metadata.json rather than passed
# here (`train_wm.resolve_epidemic` does it) because a head whose gamma disagrees
# with the simulator that made the targets is fit against a transition that never
# happened, and nothing about the loss curve would say so.
#
# The number to read first is `ens_prevalence_bias` in the results JSON, not
# `ens_count_bias`. The attack set is monotone and therefore forgiving; the
# prevalence is the NON-monotone quantity, and a head that cannot shrink `I` shows
# up there first (as a positive bias that grows with `t`) while the attack-set
# bias still looks fine.

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
RUN="${RUN:-default}"
DIFFUSION_MODEL="${DIFFUSION_MODEL:-SIR}"
DATA_DIR="results/epidemic_control/$DATASET/$RUN/data"

python -m world_model.train_wm \
    --data-dir "$DATA_DIR" \
    --diffusion-model "$DIFFUSION_MODEL" \
    --model "${WM_MODEL:-sage}" \
    --head "${HEAD:-structured}" \
    --remove-semantics blocked \
    --hidden-dim "${HIDDEN_DIM:-128}" \
    --n-layers "${N_LAYERS:-3}" \
    --epochs "${EPOCHS:-400}" \
    --patience "${PATIENCE:-50}" \
    --lr "${LR:-1e-3}" \
    --batch-size "${BATCH_SIZE:-32}" \
    --pos-weight off \
    --seed "${SEED:-42}" \
    --device "${DEVICE:-cuda}" \
    --plan-demo --plan-graphs "${PLAN_GRAPHS:-5}"

echo "[epi_train] done -> results/epidemic_control/$DATASET/$RUN/world_model"
