#!/bin/bash
#SBATCH --job-name=wm_generate_data_ba
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Usage: sbatch sbatch/wm_generate_data_ba.sbatch

set -euo pipefail
cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p results logs

# BA-100 with 40 distinct graphs (index folded into the seed -> ba_n100_m3_s42..s61),
# add_node and remove_node action ops, IC + LT, four cheap seed selectors

# random/degree/pagerank/betweenness already span a wide range of seed sets, and the
# 40-graph diversity is what makes the one-step metrics trustworthy here.

# Fewer rollouts per graph (10) but many graphs = 40 x 2 models x 4 algos x 20 = 6400 episodes

# --action-ops add_node remove_node add_edge remove_edge set_edge_weight

# k ~ U(1%, 20% of N) per episode -> 1..20 seeds here. One checkpoint covers the
# whole budget sweep instead of only the k it was generated at.

python -m data.generate_wm_data \
    --dataset ba --num-graphs 40 --syn-nodes 100 --ba-m 3 \
    --models IC LT --prob-model weighted --budget-pct-range 1 20 \
    --algorithms random degree pagerank betweenness \
    --rollouts 20 --horizon 10 --inject-p 0.4 --cf-prob 0.4 --cf-branches 2 \
    --action-ops add_node remove_node \
    --weight-lo 0.0 --weight-hi 1.0 \
    --mc-marginals 30 \
    --split 0.7 0.15 0.15 --seed 42 \
    --out-dir results/ba40/data

python -m data.validate_wm_data --dir results/ba40/data

python -c "import json; print('action_ops:', json.load(open('results/ba40/data/metadata.json'))['config']['action_ops'])"

echo "[gen_data] done -> results/ba40/data"