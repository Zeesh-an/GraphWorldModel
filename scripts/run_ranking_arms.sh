#!/usr/bin/env bash
# The `Downstream` column of Experiment 1's table, on the metric the project
# already uses: `scripts/eval_ranking.py` — preference accuracy, Kendall tau,
# Precision@K, calls-to-first-win and trusted-call reduction over a pool of real
# IM algorithms scored against NDlib on held-out graphs.
#
#   scripts/run_ranking_arms.sh experiments/ba24/wm experiments/ba24/data IC 42
#
# Runs the four arms of one (setting, seed) group sequentially — eval_ranking is
# dominated by its own Monte Carlo, so running four at once just splits the
# cores it already wants.

set -euo pipefail

WM_DIR="${1:-experiments/ba24/wm}"
DATA_DIR="${2:-experiments/ba24/data}"
SETTING="${3:-IC}"
SEED="${4:-42}"
PYTHON="${PYTHON:-/opt/anaconda3/bin/python3}"
GRAPHS="${GRAPHS:-6}"
MC_RUNS="${MC_RUNS:-64}"
SAMPLES="${SAMPLES:-20}"
THREADS="${THREADS:-4}"

OUT_DIR="$WM_DIR/ranking"
mkdir -p "$OUT_DIR"

for arm in none message_blind global message; do
  tag="${SETTING}_${arm}_s${SEED}"

  if [ ! -f "$WM_DIR/$tag.json" ]; then
    echo "[skip] $tag not trained"
    continue
  fi

  if [ -f "$OUT_DIR/$tag.json" ]; then
    echo "[skip] $tag ranking already done"
    continue
  fi

  echo "=== ranking $tag ==="
  OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
  "$PYTHON" -m scripts.eval_ranking \
    --results "$WM_DIR/$tag.json" \
    --data-dir "$DATA_DIR" \
    --n-graphs "$GRAPHS" \
    --mc-runs "$MC_RUNS" \
    --n-samples "$SAMPLES" \
    --seed 0 \
    --out "$OUT_DIR/$tag.json"
done

echo "ranking complete -> $OUT_DIR"
