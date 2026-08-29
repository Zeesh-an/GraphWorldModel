#!/bin/bash
#SBATCH --job-name=wm_train
#SBATCH --output=logs/%x_%j.out
# (logs/ must exist before `sbatch`: Slurm opens the .out file before the job runs)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Train one world model on already-generated data. For the whole experiment use
# sbatch/pipeline.sbatch; this is for retraining a single backbone in place.
#
# Usage: sbatch sbatch/influence_maximization/train.sbatch <model> <diffusion_model> <data_dir>
#        sbatch sbatch/influence_maximization/train.sbatch sage IC \
#            results/influence_maximization/sbm/default/data
#
# The checkpoint and results JSON land beside the data they were trained on, in
# results/<task>/<dataset>/<run>/world_model/, NOT in a per-job directory.
# That coupling is
# deliberate: every downstream reader (eval_planning, eval_rollout_ensemble, the
# agent stage) locates a checkpoint from its data dir, and the filenames already
# carry the backbone (wm_<model>_<dm>.pt), so five backbones on one dataset do
# not collide. Pass CKPT_DIR to override, e.g. per-job checkpoints:
#   CKPT_DIR=results/influence_maximization/ba/default/world_model_$SLURM_JOB_ID \
#       sbatch sbatch/influence_maximization/train.sbatch

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
#                   (default results/influence_maximization/ba/default/data)
MODEL="${1:-sage}"
DM="${2:-IC}"
DATA="${3:-${DATA:-results/influence_maximization/ba/default/data}}"

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
    --model "$MODEL" --head "${HEAD:-structured}" --hidden-dim 64 $EXTRA \
    --epochs 400 --lr 1e-3 --weight-decay 5e-4 --batch-size 32 \
    --pos-weight off --patience 50 --seed 42 \
    --device cuda --plan-demo ${CKPT[@]+"${CKPT[@]}"}

echo "[train] done: model=$MODEL dm=$DM data=$DATA"
echo "[train] checkpoint -> ${CKPT_DIR:-$(dirname "$DATA")/world_model}/wm_${MODEL}_${DM}.pt"
