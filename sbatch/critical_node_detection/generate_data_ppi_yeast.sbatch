#!/bin/bash
#SBATCH --job-name=cnd_gen_ppi_yeast
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Containment transitions on the Yeast PPI giant component (2,224 / 6,609).
#
# Usage: sbatch sbatch/critical_node_detection/generate_data_ppi_yeast.sbatch
#
# The most widely shared benchmark in this literature: CoreHD, BPD, GND, FINDER,
# NIRM, DCRS, SPR and BPHD all report it (research/critical_node_detection.md §7)
#, and unlike `power_grid` it is a graph class message passing handles well
# (mean degree 5.9, small diameter). If the world model is going to win anywhere,
# it is here, which makes this the natural companion run to the power-grid one.
#
# Warning: Three graphs go by "PPI" (§6.4). This loader is the 2,224-node GCC that GND
# publishes; CoreHD's Table I quotes the raw 2,361/6,646 network.

set -euo pipefail
cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p results logs

TASK=critical_node_detection \
DATASET=ppi_yeast \
RUN="${RUN:-default}" \
RUN_JOBID=0 \
START_STAGE=data END_STAGE=data \
GEN_MODELS="IC LT" \
GEN_ALGORITHMS="random degree pagerank betweenness" \
ROLLOUTS="${ROLLOUTS:-30}" \
MC_MARGINALS="${MC_MARGINALS:-30}" \
INJECT_P="${INJECT_P:-0.6}" \
BUDGET_PCT_RANGE="1 20" \
    ./sbatch/pipeline.sbatch

echo "[cnd_gen] done -> results/critical_node_detection/ppi_yeast/${RUN:-default}/data"
