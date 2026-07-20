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
mkdir -p data/output logs

# BA-100 with 40 distinct graphs (index folded into the seed -> ba_n100_m3_s42..s61),
# add_node and remove_node action ops, IC + LT, four cheap seed selectors

# random/degree/pagerank/betweenness already span a wide range of seed sets, and the
# 40-graph diversity is what makes the one-step metrics trustworthy here.

# Fewer rollouts per graph (10) but many graphs = 40 x 2 models x 4 algos x 20 = 6400 episodes

# --action-ops add_node remove_node add_edge remove_edge set_edge_weight

python -m data.generate_wm_data \
    --dataset ba --num-graphs 40 --syn-nodes 100 --ba-m 3 \
    --models IC LT --prob-model weighted --budget 5 \
    --algorithms random degree pagerank betweenness \
    --rollouts 20 --horizon 10 --inject-p 0.4 --cf-prob 0.4 --cf-branches 2 \
    --action-ops add_node remove_node \
    --weight-lo 0.0 --weight-hi 1.0 \
    --mc-marginals 30 \
    --split 0.7 0.15 0.15 --seed 42 \
    --out-dir data/output/ba40_marg_structured

python -m data.validate_wm_data --dir data/output/ba40_marg_structured

python -c "import json; print('action_ops:', json.load(open('data/output/ba40_marg_structured/metadata.json'))['config']['action_ops'])"

echo "[gen_data] done -> data/output/ba40_marg_structured"