#!/bin/bash
#SBATCH --job-name=cr_datasets
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00

# The full condition ladder across every graph with a published comparison, in
# ascending order of how much a loss on it would mean.
#
# Usage: sbatch sbatch/cascade_reconstruction/sweep_datasets.sbatch
#
# ORDER IS DELIBERATE. research/cascade_reconstruction.md §8.2 trap 4 and §2.11
# risk 5 name ONE graph specifically: on `ca_grqc` (assortativity 0.164) a plain
# Personalized PageRank baseline BEATS principled tree sampling, and loses
# elsewhere. That is the row where a weak result is most informative, so it runs
# early rather than being buried: "if the search cannot beat PageRank there, the
# result is not real".
#
#   jazz          198 nodes. The fastest place to converge a number at all
#   infectious    410 / 2,765. Xiao ICDM'18's smallest row, assortativity 0.0121
#   email_univ    1,133 / 5,451. Xiao ICDM'18, assortativity -0.0007: the
#                 assortativity-NEUTRAL control for ca_grqc
#   uci_students  1,266 / 6,451. Xiao ICDM'18 + Rozenshtein KDD'16
#   citeseer      DIPT's third graph. Warning: our count is the graph2gauss LCC and
#                 NOT the 3,327 / 4,732 that circulates: see the loader
#   cora_ml       2,810 / 7,981. DIPT's headline row, and one where our graph is
#                 known to match theirs byte for byte
#   ca_grqc       5,242 / 14,484. THE PAGERANK ROW. Run it, read it, and if the
#                 search does not clear personalized_pagerank here, say so
#   power_grid    4,941 / 6,594. DIPT's best path precision (0.680) and the
#                 sparsest graph in the set
#   ca_hepth      8,638 / 24,806. Xiao SDM'18's arxiv-hep-th
#   oregon2       11,461 / 32,730. DITTO's own graph, and the ONLY place
#                 external:ditto can be run against its own published row
#   rt_pol        18,470 / 48,053. DITTO's Pol, where BOTH supervised imputers
#                 OOM in its own table: the scale reference
#
# `--cr-setting final_snapshot` is what makes the DITTO comparison legitimate:
# DITTO conditions on the terminal state and nothing else whatever the sweep runs,
# so under any other setting it is answering a strictly harder instance than every
# other arm. The setting sweep is sbatch/cascade_reconstruction/sweep_protocol.sbatch.

set -euo pipefail
cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p results logs

SETTING="${CR_SETTING:-partial_times}"

for DATASET in jazz infectious email_univ uci_students citeseer cora_ml ca_grqc \
               power_grid ca_hepth oregon2 rt_pol; do
    echo "=== $DATASET ($SETTING) ==="
    env TASK=cascade_reconstruction DATASET="$DATASET" RUN="${RUN:-default}" \
        RUN_JOBID=0 COMPARE=1 CR_SETTING="$SETTING" \
        ./sbatch/pipeline.sbatch \
        || echo "[cr_datasets] $DATASET FAILED, continuing"
done

echo "[cr_datasets] done -> results/cascade_reconstruction/"
