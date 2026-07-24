#!/bin/bash
#SBATCH --job-name=pipeline_ba
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Whole experiment for BA-100: data -> world model -> every arm at every budget
# -> plots -> results/ba40/report.md
#
# Usage: sbatch sbatch/pipeline_ba.sbatch [start_stage] [end_stage]
#        sbatch sbatch/pipeline_ba.sbatch            # everything
#        sbatch sbatch/pipeline_ba.sbatch agent      # agent onwards (data+WM reused)
#        sbatch sbatch/pipeline_ba.sbatch plots report

set -euo pipefail
cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p results logs

START="${1:-data}"
END="${2:-report}"

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 40 distinct graphs x 2 dynamics x 4 selectors x 20 rollouts = 6,400 episodes.
# Node ops only; k drawn per episode from 1-20% of N so one checkpoint serves
# the whole budget sweep below.
python -m pipeline.run \
    --dataset ba --tag ba40 \
    --num-graphs 40 --syn-nodes 100 --ba-m 3 \
    --gen-models IC LT --gen-algorithms random degree pagerank betweenness \
    --gen-action-ops add_node remove_node \
    --budget-pct-range 1 20 --rollouts 20 --gen-horizon 10 --mc-marginals 30 \
    --wm-model sage --head structured_residual --hidden-dim 64 --n-layers 3 \
    --epochs 400 --batch-size 32 --pos-weight off --patience 50 \
    --evaluator world_model --diffusion-model IC \
    --arms baseline:degree_discount baseline:celf_pp routing one_shot_free evolve_scored \
    --budget-pcts 1 5 10 20 \
    --llm-model claude-sonnet-5 --outer-iters 5 \
    --horizon 10 --n-samples 50 --mc-runs 200 --compare \
    --allowed-ops add_node remove_node \
    --device cuda --seed 42 \
    --start-stage "$START" --end-stage "$END"

echo "[pipeline] done: results/ba40/report.md"
