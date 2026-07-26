#!/bin/bash
#SBATCH --job-name=wm_generate_data_sbm
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Usage: sbatch sbatch/wm_generate_data_sbm.sbatch

set -euo pipefail
cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p results logs

# Results land in results/<tag>_<jobid>/data, the same way logs land in
# logs/%x_%j.out, so two generation jobs can never half-overwrite each other's
# transitions. Set TAG_JOBID=0 for a stable results/<tag>/data — which is what
# the train/agent scripts downstream expect to be told about.
TAG="${TAG:-sbm40}"
TAG_JOBID="${TAG_JOBID:-1}"
[ "$TAG_JOBID" = "1" ] && [ -n "${SLURM_JOB_ID:-}" ] && TAG="${TAG}_${SLURM_JOB_ID}"
OUT="results/$TAG/data"

# SBM-100 with 40 distinct graphs, 4 planted communities each (25 nodes per block,
# p_in 0.15 / p_out 0.01). Same scale/recipe as the BA run so the only variable is
# community structure — within-block degrees are near-uniform, so degree carries
# little seed signal and budget allocation across blocks becomes the real problem.

# Node ops only, matching the BA recipe.
# All-five alternative: --action-ops add_node remove_node add_edge remove_edge set_edge_weight

# k ~ U(1%, 20% of N) per episode -> 1..20 seeds here. One checkpoint covers the
# whole budget sweep instead of only the k it was generated at.

export OUT

python -m data.generate_wm_data \
    --dataset sbm --num-graphs 40 --syn-nodes 100 \
    --sbm-blocks 4 --sbm-p-in 0.15 --sbm-p-out 0.01 \
    --models IC LT --prob-model weighted --budget-pct-range 1 20 \
    --algorithms random degree pagerank betweenness \
    --rollouts 20 --horizon 10 --inject-p 0.4 --cf-prob 0.4 --cf-branches 2 \
    --action-ops add_node remove_node \
    --weight-lo 0.0 --weight-hi 1.0 \
    --mc-marginals 30 \
    --split 0.7 0.15 0.15 --seed 42 \
    --out-dir "$OUT"

python -m data.validate_wm_data --dir "$OUT"

python -c "import json, os; print('action_ops:', json.load(open(os.environ['OUT'] + '/metadata.json'))['config']['action_ops'])"

echo "[gen_data] done -> $OUT"
echo "[gen_data] train on it with: sbatch sbatch/wm_train.sbatch sage IC $OUT"
