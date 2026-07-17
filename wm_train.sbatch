#!/bin/bash
#SBATCH --job-name=wm_train
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Usage: sbatch wm_train.sbatch <model> <diffusion_model> <data_dir>
# Usage: sbatch wm_train.sbatch sage IC data/output/ba20_marg_structured

set -euo pipefail
cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p world_model/checkpoints logs

# model           = gcn | sage | gt | gat | gcnii  (default sage)
# diffusion_model = IC | LT                        (default IC)
# data_dir        = a data/output/<...> directory  (default data/output/ba_all)
MODEL="${1:-sage}"
DM="${2:-IC}"
DATA="${3:-data/output/ba_all}"

# Backbone-specific architecture flags
case "$MODEL" in
  gt)    EXTRA="--n-layers 3 --n-heads 4 --ffn-dim 128" ;;
  gat)   EXTRA="--n-layers 3 --n-heads 4" ;;
  gcnii) EXTRA="--n-layers 8 --gcnii-alpha 0.1 --gcnii-lamda 0.5" ;;
  *)     EXTRA="--n-layers 3" ;; # gcn, sage
esac

# Confirm the job actually got a GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU-ONLY'))"

python -m world_model.train_wm \
    --data-dir "$DATA" --diffusion-model "$DM" \
    --model "$MODEL" --head structured --hidden-dim 64 $EXTRA \
    --epochs 400 --lr 1e-3 --weight-decay 5e-4 --batch-size 32 \
    --pos-weight off --patience 50 --seed 42 \
    --device cuda --plan-demo \
    --ckpt-dir world_model/checkpoints \
    --results "world_model/checkpoints/$(basename "$DATA")_${MODEL}_${DM}.json"

echo "[train] done: model=$MODEL dm=$DM data=$DATA"
