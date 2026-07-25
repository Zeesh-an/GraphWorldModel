#!/bin/bash
#SBATCH --job-name=pipeline_sbm
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Whole experiment for SBM-100 (4 planted communities). Same scale as the BA run,
# so the only variable is community structure: within-block degrees are near-uniform,
# which kills the degree heuristic and makes budget allocation across blocks the
# real problem.
#
# Usage: sbatch sbatch/pipeline_sbm.sbatch [start_stage] [end_stage]

set -euo pipefail
cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p results logs

START="${1:-data}"
END="${2:-report}"

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

python -m pipeline.run \
    --dataset sbm --tag sbm40 \
    --num-graphs 40 --syn-nodes 100 \
    --sbm-blocks 4 --sbm-p-in 0.15 --sbm-p-out 0.01 \
    --gen-models IC LT --gen-algorithms random degree pagerank betweenness \
    --gen-action-ops add_node remove_node \
    --budget-pct-range 1 20 --rollouts 20 --gen-horizon 10 --mc-marginals 30 \
    --wm-model sage --head structured_residual --hidden-dim 64 --n-layers 3 \
    --epochs 400 --batch-size 32 --pos-weight off --patience 50 \
    --diffusion-model IC \
    --baselines high_degree degree_discount pagerank_seeds celf_pp imm random_seeds \
    --arms routing one_shot_free@native one_shot_free@monte_carlo \
           one_shot_free@oracle one_shot_free@world_model \
    --budget-pcts 1 5 10 20 \
    --llm-model claude-sonnet-5 --outer-iters 5 \
    --horizon 10 --n-samples 50 --mc-runs 200 --native-mc-runs 1 --compare \
    --allowed-ops add_node remove_node \
    --device cuda --seed 42 \
    --start-stage "$START" --end-stage "$END"

echo "[pipeline] done: results/sbm40/report.md"
