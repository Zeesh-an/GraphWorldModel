#!/bin/bash
#SBATCH --job-name=epi_train
#SBATCH --output=logs/%x_%j.out
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
# `--head structured_residual` is the default here and it is only expressible
# because we wrote our own stepper: it anchors `q` on the arc's own `beta_uv`, and
# research/epidemic_control.md §2.2 records that NDlib's SIR/SIS/SEIR carry NO
# per-edge parameter at all, so under them this head cannot exist.
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
cd /home/rrishab/GraphWorldModel/
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
    --head "${HEAD:-structured_residual}" \
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
