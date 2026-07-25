#!/bin/bash
#SBATCH --job-name=pipeline_baselines
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# The full six-condition baseline table for one dataset, in one report.
#
#   1 Pure GA                   6 classical algorithms, no LLM
#   2 GA routing                one LLM call picks from the pool
#   3 Native coding agent       LLM + real executions only  (@native)
#   4 Agent + MC simulation     LLM + averaged simulator     (@monte_carlo)
#   5 Agent + oracle dynamics   LLM + true dynamics          (@oracle)
#   6 Ours: agent + learned GWM LLM + f_theta                (@world_model)
#
# Conditions 3-6 hold the method fixed at one_shot_free, so the ONLY thing that
# varies down the ladder is the inner-loop evaluator. --compare is mandatory
# here: it replays every winner on the same ground-truth referee, which is what
# makes rows from different evaluators comparable at all.
#
# Usage: sbatch sbatch/pipeline_baselines.sbatch <dataset> [tag] [start_stage] [end_stage]
#        sbatch sbatch/pipeline_baselines.sbatch sbm sbm40
#        sbatch sbatch/pipeline_baselines.sbatch netscience netscience agent

set -euo pipefail
cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p results logs

DATASET="${1:-sbm}"
TAG="${2:-$DATASET}"
START="${3:-data}"
END="${4:-report}"

# Synthetic families get a graph population; a real dataset is a single graph
case "$DATASET" in
  er|ba|ws|sbm|karate) GRAPHS="--num-graphs 40 --syn-nodes 100" ;;
  *)                   GRAPHS="" ;;
esac

case "$DATASET" in
  nethept|twitter|digg|youtube|weibo) SELECTORS="random degree pagerank"; ROLLOUTS=50 ;;
  er|ba|ws|sbm|karate)                SELECTORS="random degree pagerank betweenness"; ROLLOUTS=20 ;;
  *)                                  SELECTORS="random degree pagerank betweenness"; ROLLOUTS=100 ;;
esac

python -c "import torch; print('CUDA:', torch.cuda.is_available())"

python -m pipeline.run \
    --dataset "$DATASET" --tag "$TAG" $GRAPHS \
    --gen-models IC LT --gen-algorithms $SELECTORS \
    --gen-action-ops add_node remove_node \
    --budget-pct-range 1 20 --rollouts "$ROLLOUTS" --gen-horizon 10 --mc-marginals 30 \
    --wm-model sage --head structured_residual --hidden-dim 64 --n-layers 3 \
    --epochs 400 --batch-size 32 --pos-weight off --patience 50 \
    --diffusion-model IC \
    --baselines high_degree degree_discount pagerank_seeds celf_pp imm random_seeds \
    --arms routing \
           one_shot_free@native \
           one_shot_free@monte_carlo \
           one_shot_free@oracle \
           one_shot_free@world_model \
    --budget-pcts 1 5 10 20 \
    --llm-model claude-sonnet-5 --outer-iters 5 \
    --horizon 10 --n-samples 50 --mc-runs 200 --native-mc-runs 1 --compare \
    --allowed-ops add_node remove_node \
    --device cuda --seed 42 \
    --start-stage "$START" --end-stage "$END"

echo "[pipeline] done: results/$TAG/report.md"
