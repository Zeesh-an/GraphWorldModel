#!/bin/bash
#SBATCH --job-name=ib_datasets
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00

# The six-condition ladder on every graph this task has a published comparison
# for, back to back, each into results/influence_blocking/<name>/.
#
# Usage: sbatch sbatch/influence_blocking/sweep_datasets.sbatch
#        DATASETS="email_eu_core facebook" sbatch sbatch/influence_blocking/sweep_datasets.sbatch
#
# The order runs from the best-matched comparison to the weakest:
#
#   email_eu_core   the ONLY graph reported by SandIMIN, both Xie papers and NIE
#                   (§6.1) — the one place both halves of this literature meet
#   facebook        SandIMIN, both Xie papers AND StratLearner; the graph where
#                   §5.6 records proximity COLLAPSING to 0.170 after scoring
#                   0.770 elsewhere, so it is where the strong baseline breaks
#   wiki_vote       SandIMIN and both Xie papers; directed, dense
#   netscience      TC-AIBM's `net-sci`, but see §6.3 — theirs is 1,461 nodes to
#                   our 1,589 with an IDENTICAL edge count, almost certainly the
#                   same graph with degree-0 nodes dropped, NOT confirmed
#   nethept         CLDAG and RPS. RPS's Table 2 reports 15,229 / 62,752 / avg
#                   4.1, digit for digit our loader's output — the cleanest match
#                   in this literature, per §6.3
#   netphy          CLDAG's largest
#
# Budgets are ABSOLUTE (§8.2): percent-of-N speaks to no blocking paper at all.

set -euo pipefail
cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p results logs

DATASETS="${DATASETS:-email_eu_core facebook wiki_vote netscience nethept netphy}"
BUDGETS="${BUDGETS:-10 20 30 40 50}"

for dataset in $DATASETS; do
    echo "=== $dataset ==="
    # betweenness attacker selection is O(V*E) exact Brandes; drop it past a few thousand
    case "$dataset" in
      email_eu_core|netscience) selectors="random degree pagerank betweenness" ;;
      *)                        selectors="random degree pagerank" ;;
    esac

    TASK=influence_blocking \
    DATASET="$dataset" \
    RUN="${RUN:-default}" \
    RUN_JOBID=0 \
    COMPARE=1 \
    GEN_MODELS=IC \
    NEGATIVE_SELECTORS="$selectors" \
    BUDGETS="$BUDGETS" \
        ./sbatch/pipeline.sbatch || echo "[ib_datasets] $dataset FAILED, continuing"
done

echo "[ib_datasets] done -> results/influence_blocking/"
