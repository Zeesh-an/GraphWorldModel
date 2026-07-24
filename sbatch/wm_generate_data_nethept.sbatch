#!/bin/bash
#SBATCH --job-name=wm_generate_data_nethept
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Usage: sbatch sbatch/wm_generate_data_nethept.sbatch

set -euo pipefail
cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p results logs

# NetHEPT citation network (15,229 nodes, 62,752 directed edges) — the standard
# IM scale benchmark. Auto-downloads on first use. ONE real graph: 50 rollouts
# x 2 models x 3 algos = 300 episodes (fewer rollouts than the small graphs —
# each 15K-node episode with 30 marginal draws per step is expensive).
#
# NO betweenness: exact Brandes is O(V*E) ~ 10^9 in pure networkx at this size
# (hours). random/degree/pagerank still span diverse seed sets.

# k ~ U(1%, 20% of N) per episode -> 152..3,046 seeds here. One checkpoint covers
# the whole budget sweep instead of only the k it was generated at. The high end
# is the expensive one: 3K seeds saturate this graph fast, so most of the cost is
# the 30 marginal draws per step, not the horizon.

# Node ops only, matching the BA recipe.
# All-five alternative: --action-ops add_node remove_node add_edge remove_edge set_edge_weight

python -m data.generate_wm_data \
    --dataset nethept \
    --models IC LT --prob-model weighted --budget-pct-range 1 20 \
    --algorithms random degree pagerank \
    --rollouts 50 --horizon 10 --inject-p 0.4 --cf-prob 0.4 --cf-branches 2 \
    --action-ops add_node remove_node \
    --weight-lo 0.0 --weight-hi 1.0 \
    --mc-marginals 30 \
    --split 0.7 0.15 0.15 --seed 42 \
    --out-dir results/nethept/data

python -m data.validate_wm_data --dir results/nethept/data

python -c "import json; print('action_ops:', json.load(open('results/nethept/data/metadata.json'))['config']['action_ops'])"

echo "[gen_data] done -> results/nethept/data"
