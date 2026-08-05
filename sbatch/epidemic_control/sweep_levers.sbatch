#!/bin/bash
#SBATCH --job-name=epi_levers
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00

# The four intervention levers of research/epidemic_control.md §2.5, back to back
# on one graph, each into its own run directory.
#
# Usage: sbatch sbatch/epidemic_control/sweep_levers.sbatch
#        DATASET=primary_school sbatch sbatch/epidemic_control/sweep_levers.sbatch
#
# What this measures that no published table does: the immunization literature's
# three lines each own ONE lever and none of them reports another's. The physics
# and spectral lines vaccinate NODES (Pastor-Satorras, Cohen, NetShield, DAVA); the
# Van Mieghem / NetMelt / Kimura line cuts ARCS; the fractional line (Preciado,
# Fractional Immunization, DURLECA) reduces contact CONTINUOUSLY. One evaluator
# scoring all four against the same outbreak on the same graph under one metric is
# a comparison that does not currently exist.
#
# `vaccinate` vs `quarantine` is the sharpest pair and the cheapest to read: the
# SAME node set is chosen under both, and they differ only in whether the dosed
# node leaves the attack rate. §8.2 trap 7 is that papers reporting "nodes saved"
# against "no intervention" and against "random vaccination" differ by a large
# constant for exactly this reason.
#
# Every lever faces the SAME outbreak (deterministic in SEED) and the same rates,
# so the only thing varying down the sweep is what the budget buys.

set -euo pipefail
cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p results logs

DATASET="${DATASET:-hospital_lh10}"
LEVERS="${LEVERS:-vaccinate quarantine edge_cut contact_reduce}"

for lever in $LEVERS; do
    echo "=== $lever ==="
    TASK=epidemic_control \
    DATASET="$DATASET" \
    RUN="lever_$lever" \
    RUN_JOBID=0 \
    COMPARE=1 \
    GEN_MODELS=SIR \
    DIFFUSION_MODEL=SIR \
    EPI_LEVER="$lever" \
    BUDGET_PCTS="${BUDGET_PCTS:-1 5 10 20}" \
        ./sbatch/pipeline.sbatch || echo "[epi_levers] $lever FAILED, continuing"
done

echo "[epi_levers] done -> results/epidemic_control/$DATASET/lever_*"
