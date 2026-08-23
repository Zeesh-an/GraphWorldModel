#!/bin/bash
#SBATCH --job-name=cnd_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Train one world model on already-generated containment data. For the whole
# experiment use sbatch/pipeline.sbatch; this is for retraining a backbone in
# place.
#
# Usage: sbatch sbatch/critical_node_detection/train.sbatch <model> <dm> <data_dir>
#        sbatch sbatch/critical_node_detection/train.sbatch sage IC \
#            results/critical_node_detection/ppi_yeast/default/data
#
# --remove-semantics blocked is NOT optional and is not defaulted here: the head's
# T_exo is fixed at training time, and `train_wm` refuses a value that disagrees
# with the dataset's own metadata.json. A checkpoint trained under `spent` counts
# every blocker as infected and biases the containment number by +k
# (research/critical_node_detection.md §2.3).
#
# ⚠️ GCNII is the backbone to reach for on `power_grid` and the road networks:
# §9.4 predicts a 3-6 layer receptive field cannot see across a diameter-46 mesh,
# and 16 layers is the only configuration here that can. Expect it to lose anyway,
# and report that.

set -euo pipefail
# Repo root, derived from this script's own location so the file works in
# any checkout rather than one person's home directory. Set REPO_ROOT to
# override, e.g. when submitting a copy staged outside the tree.
cd "${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Activate the project virtualenv, unless one is already active or VENV
# points somewhere else. Skipped silently when neither exists, so the
# script still runs under a module-loaded or conda interpreter.
VENV="${VENV:-venv}"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
fi
mkdir -p results logs

MODEL="${1:-sage}"
DM="${2:-IC}"
DATA="${3:-${DATA:-results/critical_node_detection/ppi_yeast/default/data}}"

CKPT=()
[ -n "${CKPT_DIR:-}" ] && CKPT=(--ckpt-dir "$CKPT_DIR")

case "$MODEL" in
  gt)    EXTRA="--n-layers 3 --n-heads 4 --ffn-dim 128" ;;
  gat)   EXTRA="--n-layers 3 --n-heads 4" ;;
  gcnii) EXTRA="--n-layers 16 --gcnii-alpha 0.1 --gcnii-lamda 0.5" ;;
  *)     EXTRA="--n-layers 3" ;; # gcn, sage
esac

# structured_residual is IC-only (it anchors q on logit(w)); LT gets the plain
# structured head, whose LTThresholdHead needs no remove_semantics branch
HEAD="structured_residual"
[ "$DM" = "LT" ] && HEAD="structured"

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU-ONLY'))"

python -m world_model.train_wm \
    --data-dir "$DATA" --diffusion-model "$DM" \
    --model "$MODEL" --head "$HEAD" --remove-semantics blocked \
    --hidden-dim 128 $EXTRA \
    --epochs 400 --lr 1e-3 --weight-decay 5e-4 --batch-size 32 \
    --pos-weight off --patience 50 --seed 42 \
    --device cuda --plan-demo "${CKPT[@]}"

echo "[cnd_train] done: model=$MODEL dm=$DM head=$HEAD data=$DATA"
echo "[cnd_train] checkpoint -> ${CKPT_DIR:-$(dirname "$DATA")/world_model}/wm_${MODEL}_${DM}.pt"
