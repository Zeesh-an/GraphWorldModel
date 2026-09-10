#!/usr/bin/env bash
# Experiment 1 — the controlled action-conditioning comparison.
#
# Four arms x N seeds x {IC, LT}, everything else held fixed: the same dataset,
# the same graph-disjoint split, the same epochs/patience/batch/lr, the same
# rollout horizon and the same evaluation blocks. The ONLY variable is where the
# action reaches the learned transition.
#
#   none           the current model: action = three input columns of X
#   message_blind  + a message MLP with its action inputs zeroed (CAPACITY control)
#   global         + a global action embedding z_a in the message
#   message        + a per-edge action relevance r_uv as well (the variant)
#
# The four arms of one (dynamics, seed) group run concurrently, so the group
# takes one run's wall clock. Threads are split so the four together stay under
# the core count.
#
#   scripts/run_action_conditioning_sweep.sh experiments/ba24/data experiments/ba24/wm "42 43 44" "IC LT"

set -euo pipefail

DATA_DIR="${1:-experiments/ba24/data}"
OUT_DIR="${2:-experiments/ba24/wm}"
SEEDS="${3:-42 43 44}"
DYNAMICS="${4:-IC LT}"
PYTHON="${PYTHON:-/opt/anaconda3/bin/python3}"
EPOCHS="${EPOCHS:-120}"
PATIENCE="${PATIENCE:-25}"
THREADS="${THREADS:-3}"
# Extra train_wm flags shared by all four arms of a sweep, and a tag naming the
# setting they define. The IC-with-true-w setting cannot discriminate the arms —
# the structured head is handed q = w, so the encoder has almost nothing left to
# learn — which is why the informative settings are `--hide-edge-weights` (q must
# be inferred from structure) and LT (thresholds are hidden by construction).
EXTRA_FLAGS="${EXTRA_FLAGS:-}"
TAG="${TAG:-}"

ARMS="none message_blind global message"

mkdir -p "$OUT_DIR" "$OUT_DIR/logs"

for dynamics in $DYNAMICS; do
  for seed in $SEEDS; do
    echo "=== $dynamics seed=$seed ==="
    for arm in $ARMS; do
      tag="${dynamics}${TAG}_${arm}_s${seed}"
      if [ -f "$OUT_DIR/$tag.json" ]; then
        echo "  [skip] $tag already done"
        continue
      fi
      OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
      "$PYTHON" -m world_model.train_wm \
        --data-dir "$DATA_DIR" \
        --diffusion-model "$dynamics" \
        --model sage --head structured --pos-weight off \
        --hidden-dim 64 --n-layers 3 \
        --epochs "$EPOCHS" --batch-size 32 --patience "$PATIENCE" \
        --lr 1e-3 --weight-decay 5e-4 --dropout 0.1 \
        --seed "$seed" --device cpu \
        --action-conditioning "$arm" \
        $EXTRA_FLAGS \
        --plan-demo --plan-graphs 5 \
        --ckpt-dir "$OUT_DIR/ckpt_$tag" \
        --results "$OUT_DIR/$tag.json" \
        > "$OUT_DIR/logs/$tag.log" 2>&1 &
    done
    wait
    echo "=== $dynamics seed=$seed done ==="
  done
done

echo "all runs complete -> $OUT_DIR"
