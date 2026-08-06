#!/bin/bash
#SBATCH --job-name=sl_sweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00

# The six-condition ladder plus arm A on every graph this task has a published
# comparison for, back to back, each into results/source_localization/<name>/.
#
# Usage: sbatch sbatch/source_localization/sweep_datasets.sbatch
#        DATASETS="jazz karate" sbatch sbatch/source_localization/sweep_datasets.sbatch
#
# The order runs from the graphs whose published counts match ours byte-for-byte
# to the ones where they do not, so a comparability caveat never arrives as a
# surprise (research/source_localization.md §6.1, §6.4, §11):
#
#   karate       34 / 78, in four papers; the smallest full sweep there is, and
#                the graph the 2026 GNN benchmark shows a 72.9% top-5 ceiling on
#   dolphins     62 / 159, in IVGD, GraphSL and the GNN benchmark. THE graph this
#                literature uses that no other task file needed
#   jazz         198 / 2,742: four papers agree to the digit. (key)
#   netscience   1,589 / 2,742: comparable to IVGD, GraphSL and SIDSL, and NOT
#                to SL-VAE, whose 1,565 / 13,532 is a different graph that is
#                undocumented and not publicly downloadable (§6.4.1). Say so in
#                any table that carries this row.
#   cora_ml      2,810 / 7,981: three papers agree; our loader already applies
#                graph2gauss standardize()
#   power_grid   4,941 / 6,594: four papers agree. The largest matched graph,
#                and a diameter-46 mesh, so it is also where a k-layer encoder's
#                receptive field runs out
#   deezer       47,538 / 222,887: IVGD's SCALABILITY column and nothing else,
#                so there is no F1 row to compare against. It is here for the
#                COST axis of §2.3.2, which needs a size the others cannot give.
#
# Warning: NO published number in this literature is under IC or LT. SL-VAE uses SI/SIR,
# SL-Diff / SIDSL / DDMSL use real cascades, the GNN benchmark uses SIR. Our
# graphs match byte-for-byte and our DYNAMICS do not, so every number this sweep
# produces is self-contained (§11). That is the biggest comparability gap in the
# file, and it is bigger than any graph-version disagreement.

set -euo pipefail
cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p results logs

DATASETS="${DATASETS:-karate dolphins jazz netscience cora_ml power_grid}"

for dataset in $DATASETS; do
    echo "=== $dataset ==="
    # exact Brandes is O(V*E) in pure networkx; drop it past a few thousand nodes
    case "$dataset" in
      karate|dolphins|jazz|netscience) selectors="random degree pagerank betweenness" ;;
      *)                               selectors="random degree pagerank" ;;
    esac
    # karate is a synthetic family here (nx.karate_club_graph), so it takes a
    # graph POPULATION rather than a single instance
    case "$dataset" in
      karate) graphs=1 ;;
      *)      graphs=1 ;;
    esac

    TASK=source_localization \
    DATASET="$dataset" \
    RUN="${RUN:-default}" \
    RUN_JOBID=0 \
    COMPARE=1 \
    NUM_GRAPHS="$graphs" \
    GEN_MODELS=IC \
    GEN_ALGORITHMS="$selectors" \
    SL_INSTANCES="${SL_INSTANCES:-20}" \
    SL_OBSERVATION="${SL_OBSERVATION:-marginal}" \
        ./sbatch/pipeline.sbatch || echo "[sl_sweep] $dataset FAILED, continuing"
done

echo "[sl_sweep] done -> results/source_localization/"
