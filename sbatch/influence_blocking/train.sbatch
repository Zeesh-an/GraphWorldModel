#!/bin/bash
#SBATCH --job-name=ib_train
#SBATCH --output=logs/%x_%j.out
# (logs/ must exist before `sbatch`: Slurm opens the .out file before the job runs)
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

# structured is the repo default. HEAD=structured_residual anchors q on
# logit(w) and is IC-only: CLT / LT carry no per-edge probability to anchor on
HEAD="${HEAD:-structured}"

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU-ONLY'))"

python -m world_model.train_wm \
    --data-dir "$DATA" --diffusion-model "$DM" \
    --model "$MODEL" --head "$HEAD" --remove-semantics blocked \
    --hidden-dim 128 $EXTRA \
    --epochs 400 --lr 1e-3 --weight-decay 5e-4 --batch-size 32 \
    --pos-weight off --patience 50 --seed 42 \
    --device cuda --plan-demo ${CKPT[@]+"${CKPT[@]}"}

echo "[ib_train] done: model=$MODEL dm=$DM head=$HEAD data=$DATA"
echo "[ib_train] checkpoint -> ${CKPT_DIR:-$(dirname "$DATA")/world_model}/wm_${MODEL}_${DM}.pt"
