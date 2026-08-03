#!/bin/bash
#SBATCH --job-name=cnd_gen_power_grid
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Containment transitions on the US power grid (4,941 / 6,594).
#
# Usage: sbatch sbatch/critical_node_detection/generate_data_power_grid.sbatch
#
# This is the single best-matched benchmark in research/critical_node_detection.md
# §7 — CoreHD, BPD and NIRM all publish absolute numbers on this exact 4,941/6,594
# file — and §9.4 predicts we LOSE on it: diameter 46 exhausts a 3-6 layer
# receptive field, and MIND's Table 5 puts FINDER at 161.7 on `eu-powergrid`
# against EI's 80.6. Run it first and report the loss as predicted-and-confirmed
# rather than discovering it late.
#
# --task supplies remove_node generation and `blocked` semantics from the
# registry, so no --gen-action-ops or --remove-semantics flag appears here.
# betweenness is dropped from the selectors: exact Brandes is O(V*E) in pure
# networkx and takes hours at 5K nodes.

set -euo pipefail
cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p results logs

TASK=critical_node_detection \
DATASET=power_grid \
RUN="${RUN:-default}" \
RUN_JOBID=0 \
START_STAGE=data END_STAGE=data \
GEN_MODELS="IC LT" \
GEN_ALGORITHMS="random degree pagerank" \
ROLLOUTS="${ROLLOUTS:-30}" \
MC_MARGINALS="${MC_MARGINALS:-30}" \
INJECT_P="${INJECT_P:-0.6}" \
BUDGET_PCT_RANGE="1 20" \
    ./sbatch/pipeline.sbatch

echo "[cnd_gen] done -> results/critical_node_detection/power_grid/${RUN:-default}/data"
