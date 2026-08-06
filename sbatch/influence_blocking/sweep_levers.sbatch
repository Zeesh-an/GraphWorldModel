#!/bin/bash
#SBATCH --job-name=ib_levers
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00

# The four intervention levers of research/influence_blocking.md §1.1, back to
# back on one graph, each into its own run directory.
#
# Usage: sbatch sbatch/influence_blocking/sweep_levers.sbatch
#        DATASET=facebook sbatch sbatch/influence_blocking/sweep_levers.sbatch
#
# This is the experiment §7 says does not exist. The seeding line of this
# literature (Budak, CLDAG, RPS, NIE) and the structural line (Kimura, Xie,
# SandIMIN, DiffIM) share no dataset outside the SNAP staples and NO PAPER IN
# EITHER LINE REPORTS THE OTHER'S METRIC, so a single evaluator scoring
# add_node, remove_node, remove_edge and set_edge_weight interventions on the
# same graph, against the same rumour, under one metric is a comparison nobody
# has published.
#
# Every lever faces the SAME S_N (deterministic in SEED) and the same tie-break,
# so the only thing varying down the sweep is what the budget buys.

set -euo pipefail
cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p results logs

DATASET="${DATASET:-email_eu_core}"
LEVERS="${LEVERS:-counter_seed node_block edge_block weight_block}"

for lever in $LEVERS; do
    echo "=== $lever ==="
    TASK=influence_blocking \
    DATASET="$DATASET" \
    RUN="lever_$lever" \
    RUN_JOBID=0 \
    COMPARE=1 \
    GEN_MODELS=IC \
    BLOCKING_LEVER="$lever" \
    BUDGETS="${BUDGETS:-10 20 30 40 50}" \
        ./sbatch/pipeline.sbatch || echo "[ib_levers] $lever FAILED, continuing"
done

echo "[ib_levers] done -> results/influence_blocking/$DATASET/lever_*"
