#!/bin/bash
#SBATCH --job-name=cr_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Train one world model on already-generated cascade-reconstruction data. For the
# whole experiment use sbatch/pipeline.sbatch; this is for retraining a backbone
# in place.
#
# Usage: sbatch sbatch/cascade_reconstruction/train.sbatch <model> <dm> <data_dir>
#        sbatch sbatch/cascade_reconstruction/train.sbatch sage IC \
#            results/cascade_reconstruction/jazz/default/data
#
# THE HEAD IS NOT A PREFERENCE HERE, it is the prerequisite
# (research/cascade_reconstruction.md §2.1, §9 item 6). A `linear` head's
# free-running rollout SATURATES to the whole graph, which makes any smoothing
# objective degenerate: the model believes every hidden node was infected, and
# every reconstruction metric collapses to "predict everything". The
# self-terminating ICTransmissionHead is what makes trajectory likelihood a
# meaningful score at all: `count_bias` near zero is the number to check before
# reading any decode as a result.
#
# The second thing to watch is the same one source localization watches, and for a
# related reason: `ens_count_bias` positive means the rollout runs away, and a
# runaway kernel scores every proposed history as plausible, so an MCMC decoder
# accepts everything and its acceptance rate goes to 1. That shows up in arm A's
# `mcmc_acceptance_rate` long before it shows up in one-step delta_f1.

set -euo pipefail
cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p results logs

MODEL="${1:-sage}"
DM="${2:-IC}"
DATA="${3:-${DATA:-results/cascade_reconstruction/jazz/default/data}}"

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

echo "[cr_train] done: model=$MODEL dm=$DM head=$HEAD data=$DATA"
echo "[cr_train] checkpoint -> ${CKPT_DIR:-$(dirname "$DATA")/world_model}/wm_${MODEL}_${DM}.pt"
