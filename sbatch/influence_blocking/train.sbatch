#!/bin/bash
#SBATCH --job-name=ib_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Train one COMPETITIVE world model on already-generated blocking data. For the
# whole experiment use sbatch/pipeline.sbatch; this is for retraining a backbone
# in place.
#
# Usage: sbatch sbatch/influence_blocking/train.sbatch <model> <dm> <data_dir>
#        sbatch sbatch/influence_blocking/train.sbatch sage IC \
#            results/influence_blocking/email_eu_core/default/data
#
# Nothing here passes --competitive, --tie-break or --positive-prob, and that is
# deliberate: train_wm reads all three back from the dataset's own metadata.json.
# A head whose tie-break disagrees with the simulator that produced the targets is
# fit against a transition that never happened, and nothing in the loss curve
# would say so (research/influence_blocking.md §8.4).
#
# --remove-semantics blocked is likewise not optional: `spent` counts each blocked
# node as infected and biases every prevented-influence number by exactly +k.
#
# Warning: expect IC to carry this task. Under CLT every node draws TWO hidden
# thresholds (He et al. SDM'12 §3), so the LT head pays the single-cascade
# partial-observability tax twice and its one-step ceiling drops again (§2.2).

set -euo pipefail
cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p results logs

MODEL="${1:-sage}"
DM="${2:-IC}"
DATA="${3:-${DATA:-results/influence_blocking/email_eu_core/default/data}}"

CKPT=()
[ -n "${CKPT_DIR:-}" ] && CKPT=(--ckpt-dir "$CKPT_DIR")

case "$MODEL" in
  gt)    EXTRA="--n-layers 3 --n-heads 4 --ffn-dim 128" ;;
  gat)   EXTRA="--n-layers 3 --n-heads 4" ;;
  gcnii) EXTRA="--n-layers 16 --gcnii-alpha 0.1 --gcnii-lamda 0.5" ;;
  *)     EXTRA="--n-layers 3" ;; # gcn, sage
esac

# structured_residual is IC-only (it anchors q on logit(w)); CLT carries no
# per-edge transmission probability to anchor on, so LT gets the plain head
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

echo "[ib_train] done: model=$MODEL dm=$DM head=$HEAD data=$DATA"
echo "[ib_train] checkpoint -> ${CKPT_DIR:-$(dirname "$DATA")/world_model}/wm_${MODEL}_${DM}.pt"
