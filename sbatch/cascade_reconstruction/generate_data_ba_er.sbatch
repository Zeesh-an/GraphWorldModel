#!/bin/bash
#SBATCH --job-name=cr_gen_ba_er
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# DITTO's own two synthetic graphs, at DITTO's own parameters: the two rows of
# research/cascade_reconstruction.md §5.1 that are reachable without a loader.
#
# Usage: sbatch sbatch/cascade_reconstruction/generate_data_ba_er.sbatch
#
# From DITTO's Appendix D.1 and its `inc/data.py`, both [verified]:
#   BA   n = 1,000, attachment m = 4    -> 3,984 undirected edges
#   ER   n = 1,000, p = 0.008           -> 3,987 undirected edges
#   both T = 10, 5% of nodes as sources
#
# Our IC is not DITTO's SI, and that is the ONE thing every number produced here
# has to say out loud: §11 records that no published cascade-reconstruction result
# exists under IC or LT, so the graphs match byte for byte and the dynamics do
# not. The comparison is "same graph, stated different dynamics", which is worth
# more than nothing and less than a reproduction.
#
# --mc-marginals is raised to 200 here rather than left at 30 (§2.11 risk 3):
# Farajtabar uses 400-500 samples and Xiao 1,000 sampled trees, and 30 is fine for
# a TRAINING target but reads as noise in a masked-episode ground truth. This is
# the generation stage, so it costs wall clock once and is paid back by every arm.

set -euo pipefail
cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p results logs

ROLLOUTS="${ROLLOUTS:-60}"
MC_MARGINALS="${MC_MARGINALS:-200}"

for SPEC in "ba --ba-m 4" "er --er-p 0.008"; do
    set -- $SPEC
    DATASET="$1"; shift
    echo "=== $DATASET ($* ) ==="

    # --inject-p 0 and no --action-ops: a recover task's episodes are
    # diffusion-only, so T_exo = identity and the whole observable history is
    # caused by the t=0 seed commit alone. --trace-parents is set by the registry
    # when the pipeline drives this, and explicitly here because this is standalone.
    python -m data.generate_wm_data \
        --task cascade_reconstruction --dataset "$DATASET" \
        --num-graphs 1 --syn-nodes 1000 "$@" \
        --models IC LT --rollouts "$ROLLOUTS" --horizon 10 \
        --budget-pct-range 5 5 \
        --algorithms random degree pagerank betweenness \
        --inject-p 0 --cf-prob 0 --action-ops \
        --trace-parents --mc-marginals "$MC_MARGINALS" --seed 42 \
    `# SINGLE GRAPH per dataset: a graph-disjoint split is impossible here, so` \
    `# episodes are split PER EPISODE and test metrics are in-graph` \
    --split-mode episode_random \
        --out-dir "results/cascade_reconstruction/$DATASET/default/data"
done

echo "[cr_gen] done -> results/cascade_reconstruction/{ba,er}/default/data"
