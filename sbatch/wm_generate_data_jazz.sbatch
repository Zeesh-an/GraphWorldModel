#!/bin/bash
#SBATCH --job-name=wm_generate_data_jazz
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Usage: sbatch sbatch/wm_generate_data_jazz.sbatch

set -euo pipefail
cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p data/output logs

# Jazz collaboration network (198 nodes, 2,742 edges, real community structure).
# Auto-downloads on first use. ONE real graph, so episode diversity comes from
# rollouts instead of graph count: 100 rollouts x 2 models x 4 algos = 800 episodes
# (BA/SBM get 6,400 from 40 graphs x 20 rollouts).

# Node ops only, matching the BA recipe.
# All-five alternative: --action-ops add_node remove_node add_edge remove_edge set_edge_weight

# k ~ U(1%, 20% of N) per episode -> 2..40 seeds here. One checkpoint covers the
# whole budget sweep instead of only the k it was generated at.

python -m data.generate_wm_data \
    --dataset jazz \
    --models IC LT --prob-model weighted --budget-pct-range 1 20 \
    --algorithms random degree pagerank betweenness \
    --rollouts 100 --horizon 10 --inject-p 0.4 --cf-prob 0.4 --cf-branches 2 \
    --action-ops add_node remove_node \
    --weight-lo 0.0 --weight-hi 1.0 \
    --mc-marginals 30 \
    --split 0.7 0.15 0.15 --seed 42 \
    --out-dir data/output/jazz_marg_structured

python -m data.validate_wm_data --dir data/output/jazz_marg_structured

python -c "import json; print('action_ops:', json.load(open('data/output/jazz_marg_structured/metadata.json'))['config']['action_ops'])"

echo "[gen_data] done -> data/output/jazz_marg_structured"
