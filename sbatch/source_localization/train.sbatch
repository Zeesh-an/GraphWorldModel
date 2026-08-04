#!/bin/bash
#SBATCH --job-name=sl_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Train one world model on already-generated source-localization data. For the
# whole experiment use sbatch/pipeline.sbatch; this is for retraining a backbone
# in place.
#
# Usage: sbatch sbatch/source_localization/train.sbatch <model> <dm> <data_dir>
#        sbatch sbatch/source_localization/train.sbatch sage IC \
#            results/source_localization/jazz/default/data
#
# `--remove-semantics spent` is the default and is right here: this task emits no
# remove_node at all, so the semantics never fires — but the flag is stated rather
# than omitted because `train_wm` compares it against the dataset's own
# metadata.json, and a silent mismatch is the failure that check exists for.
#
# TWO THINGS TO WATCH, and they are the reason this task is a good diagnostic for
# the world model rather than only a consumer of it
# (research/source_localization.md §2.9 risk 4, §9.4):
#
#   ens_count_bias   a rollout that SATURATES destroys the information the inverse
#                    problem needs, so a positive bias here shows up as a collapsed
#                    F1 downstream while one-step delta_f1 — dominated by unchanged
#                    nodes — stays comfortable. That asymmetry is the point: source
#                    localization fails LOUDLY where the existing metrics do not.
#
#   the head         the structured heads are what make the model invertible in
#                    practice. ICTransmissionHead composes p_new from per-edge
#                    transmission and cannot saturate by construction, and it is
#                    also what makes `wm_sl.soft_rollout` differentiable in the
#                    relaxed source vector with no custom backward code. A
#                    `--head linear` checkpoint will train fine and invert badly.

set -euo pipefail
cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p results logs

MODEL="${1:-sage}"
DM="${2:-IC}"
DATA="${3:-${DATA:-results/source_localization/jazz/default/data}}"

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
    --model "$MODEL" --head "$HEAD" --remove-semantics spent \
    --hidden-dim 128 $EXTRA \
    --epochs 400 --lr 1e-3 --weight-decay 5e-4 --batch-size 32 \
    --pos-weight off --patience 50 --seed 42 \
    --device cuda --plan-demo "${CKPT[@]}"

echo "[sl_train] done: model=$MODEL dm=$DM head=$HEAD data=$DATA"
echo "[sl_train] checkpoint -> ${CKPT_DIR:-$(dirname "$DATA")/world_model}/wm_${MODEL}_${DM}.pt"
