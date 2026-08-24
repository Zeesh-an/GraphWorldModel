#!/bin/bash
#SBATCH --job-name=wm_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Train one world model on already-generated data. For the whole experiment use
# sbatch/pipeline.sbatch; this is for retraining a single backbone in place.
#
# Usage: sbatch sbatch/adaptive_online_im/train.sbatch <model> <diffusion_model> <data_dir>
#        sbatch sbatch/adaptive_online_im/train.sbatch sage IC \
#            results/adaptive_online_im/powerlaw_cluster/default/data
#
# The checkpoint and results JSON land beside the data they were trained on, in
# results/<task>/<dataset>/<run>/world_model/, NOT in a per-job directory.
# That coupling is
# deliberate: every downstream reader (eval_planning, eval_rollout_ensemble, the
# agent stage) locates a checkpoint from its data dir, and the filenames already
# carry the backbone (wm_<model>_<dm>.pt), so five backbones on one dataset do
# not collide. Pass CKPT_DIR to override, e.g. per-job checkpoints:
#   CKPT_DIR=results/influence_maximization/ba/default/world_model_$SLURM_JOB_ID \
#       sbatch sbatch/adaptive_online_im/train.sbatch

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

# model           = gcn | sage | gt | gat | gcnii   (default sage)
# diffusion_model = IC | LT                         (default IC)
# data_dir        = any run's data/ directory
#                   (default results/adaptive_online_im/powerlaw_cluster/default/data)
MODEL="${1:-sage}"
DM="${2:-IC}"
DATA="${3:-${DATA:-results/adaptive_online_im/powerlaw_cluster/default/data}}"

CKPT=()
[ -n "${CKPT_DIR:-}" ] && CKPT=(--ckpt-dir "$CKPT_DIR")

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
    --model "$MODEL" --head structured_residual --hidden-dim 64 $EXTRA \
    --epochs 400 --lr 1e-3 --weight-decay 5e-4 --batch-size 32 \
    --pos-weight off --patience 50 --seed 42 \
    --device cuda --plan-demo "${CKPT[@]}"

echo "[train] done: model=$MODEL dm=$DM data=$DATA"
echo "[train] checkpoint -> ${CKPT_DIR:-$(dirname "$DATA")/world_model}/wm_${MODEL}_${DM}.pt"
