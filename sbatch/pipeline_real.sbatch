#!/bin/bash
#SBATCH --job-name=pipeline_real
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Whole experiment for ONE real graph. Episode diversity comes from rollouts
# rather than graph count, so --rollouts is high and --num-graphs stays 1.
#
# Usage: sbatch sbatch/pipeline_real.sbatch <dataset> [start_stage] [end_stage]
#        sbatch sbatch/pipeline_real.sbatch jazz
#        sbatch sbatch/pipeline_real.sbatch netscience agent
#        sbatch sbatch/pipeline_real.sbatch nethept

set -euo pipefail
cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p results logs

DATASET="${1:-jazz}"
START="${2:-data}"
END="${3:-report}"

# Exact Brandes betweenness is O(V*E) in pure networkx — hours on 15K+ nodes,
# so the big graphs drop it and lean on random/degree/pagerank instead.
case "$DATASET" in
  nethept|twitter|digg|youtube|weibo)
    SELECTORS="random degree pagerank"
    ROLLOUTS=50
    ;;
  *)
    SELECTORS="random degree pagerank betweenness"
    ROLLOUTS=100
    ;;
esac

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

python -m pipeline.run \
    --dataset "$DATASET" --tag "$DATASET" \
    --gen-models IC LT --gen-algorithms $SELECTORS \
    --gen-action-ops add_node remove_node \
    --budget-pct-range 1 20 --rollouts "$ROLLOUTS" --gen-horizon 10 --mc-marginals 30 \
    --wm-model sage --head structured_residual --hidden-dim 128 --n-layers 3 \
    --epochs 400 --batch-size 32 --pos-weight off --patience 50 \
    --evaluator world_model --diffusion-model IC \
    --arms baseline:degree_discount baseline:celf_pp routing one_shot_free evolve_scored \
    --budget-pcts 1 5 10 20 \
    --llm-model claude-sonnet-5 --outer-iters 5 \
    --horizon 10 --n-samples 50 --mc-runs 200 --compare \
    --allowed-ops add_node remove_node \
    --device cuda --seed 42 \
    --start-stage "$START" --end-stage "$END"

echo "[pipeline] done: results/$DATASET/report.md"
