#!/bin/bash
#SBATCH --job-name=cp_datasets
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00

# Every REAL cascade corpus we load, in ascending order of what a loss would mean.
#
# Usage: sbatch sbatch/cascade_prediction/sweep_datasets.sbatch
#
# research/cascade_prediction.md 6.5 ranks these, and the order below is that
# ranking rather than alphabetical:
#
#   1. taoke        auto-downloads, small, and the only corpus here with a REAL
#                   logged transmission edge. Warning: 6.5 lists it under "not
#                   recommended" (too new, single-paper) and it is FRONT-LOADED —
#                   85% of its items launch within 14 minutes — so it runs under
#                   the RANDOM split, which is the only protocol it supports. Smoke
#                   test, not a result.
#   2. digg_cascades auto-downloads, real social graph, real vote times. Its windows
#                   are OURS rather than published (6.2 classifies Digg as a
#                   microscopic corpus with no clock), so its row is self-contained.
#   3. casflow_aps  THE ONE TO DO (6.5): no rate limits, no registration, and 3y/5y
#                   windows long enough that one replayed step is a whole year. Needs
#                   the manual Drive bundle; the loader prints the steps.
#   4. casflow_weibo the corpus carrying almost every published number (Weibo-A)
#   5. casflow_twitter the canonical host is 404 and this mirror is the only route
#   6. memetracker  1.3 GB per month and OUR extraction rather than Topo-LSTM's,
#                   so it runs last and its row says so
#
# 9.9: do not chase the leaderboard. CasFlow is a 2M-parameter model tuned for this
# one task; the value of every row here is diagnostic (9.1), not competitive.

set -euo pipefail
cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p results logs

run_dataset () {
    local dataset="$1"; shift
    echo "=== $dataset ==="
    env TASK=cascade_prediction DATASET="$dataset" RUN=default RUN_JOBID=0 \
        COMPARE=1 GEN_MODELS="IC" DIFFUSION_MODEL=IC "$@" \
        ./sbatch/pipeline.sbatch || echo "[cp_datasets] $dataset FAILED, continuing"
}

# Front-loaded: a leak-free split is impossible on it, and the pipeline says so
# with the numbers rather than failing silently
run_dataset taoke           CP_SPLIT=random CP_MAX_CASCADES=2000

run_dataset digg_cascades   CP_SPLIT=chronological
run_dataset casflow_aps     CP_SPLIT=chronological
run_dataset casflow_weibo   CP_SPLIT=chronological CP_MAX_NODES=50000
run_dataset casflow_twitter CP_SPLIT=chronological CP_MAX_NODES=50000
run_dataset memetracker     CP_SPLIT=chronological CP_MAX_NODES=20000

echo "=== done: results/cascade_prediction/<corpus>/default/report.md ==="
