#!/bin/bash
#SBATCH --job-name=pipeline_oracle
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Outer-loop development against the ground-truth dynamics ceiling: conditions 1-5
# only, so no world model is trained and the train stage is skipped automatically
# (no GPU needed). Use this to iterate on the coding agent itself; add
# one_shot_free@world_model to --arms once the arms are settled.
#
# Usage: sbatch sbatch/pipeline_oracle.sbatch <dataset> [tag] [start_stage] [end_stage]
#        sbatch sbatch/pipeline_oracle.sbatch sbm sbm40
#        sbatch sbatch/pipeline_oracle.sbatch netscience netscience agent

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

python -m pipeline.run \
    --dataset "$DATASET" --tag "$TAG" $GRAPHS \
    --gen-models IC --gen-algorithms random degree pagerank \
    --gen-action-ops add_node remove_node \
    --budget-pct-range 1 20 --rollouts 20 --gen-horizon 10 --mc-marginals 30 \
    --diffusion-model IC \
    --baselines high_degree degree_discount pagerank_seeds celf_pp imm random_seeds \
    --arms routing one_shot_free@native one_shot_free@monte_carlo one_shot_free@oracle \
    --budget-pcts 1 5 10 20 \
    --llm-model claude-sonnet-5 --outer-iters 5 \
    --horizon 10 --n-samples 50 --mc-runs 200 --native-mc-runs 1 --compare \
    --allowed-ops add_node remove_node \
    --device cpu --seed 42 \
    --start-stage "$START" --end-stage "$END"

echo "[pipeline] done: results/$TAG/report.md"
