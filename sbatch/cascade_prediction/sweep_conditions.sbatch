#!/bin/bash
#SBATCH --job-name=cp_conditions
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=96:00:00

# The six-condition ladder, plus the two published repos that are wired.
#
# Usage: sbatch sbatch/cascade_prediction/sweep_conditions.sbatch
#        DATASET=casflow_aps sbatch sbatch/cascade_prediction/sweep_conditions.sbatch
#
# research/cascade_prediction.md 2.1 warns that this task's ACTION space is empty,
# `a_t` is NULL at every step, none of the five ops fire, and there is no planning
# regret because there is no decision to plan over. It then concludes that
# conditions 3-6 "collapse into one arm". **That is half right and the half that is
# wrong is why this script exists.** What varies down that ladder here is not the
# intervention but the FORWARD MODEL the predictor may call:
#
#   3 @native       no forward model at all. Feature-driven and point-process
#                   predictors only, and 3.1 records that feature models sometimes
#                   BEAT deep ones on these corpora, so this arm may win outright
#   4 @monte_carlo  NDlib rollouts, at steps x samples x mc_runs real episodes per
#                   forecast_marginals call. Whether it completes is itself a finding
#   5 @oracle       the analytic IC form, which is NOT the true dynamics here, and
#                   that is the entire point (9.1)
#   6 @world_model  one batched matmul per kernel evaluation
#
# --compare adds the number 9.1 is actually about: the MODELLING error of each arm's
# own forward model with no program in the loop. The gap between that column and the
# arm's own error is what the SEARCH bought; the LEVEL is how far an
# Independent-Cascade-shaped kernel is from a real adoption process. Expect to lose
# (9.2) and read the modelling column rather than the ranking.

set -euo pipefail
cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p results logs

DATASET="${DATASET:-casflow_weibo}"

# The wired published repos ride along as condition 7. Both are TensorFlow/PyTorch
# and both need `python -m baselines.setup_baselines --only <name>` first; a missing
# install is SKIPPED with a marker rather than failing the sweep.
env TASK=cascade_prediction DATASET="$DATASET" RUN=conditions RUN_JOBID=0 \
    COMPARE=1 GEN_MODELS="IC" DIFFUSION_MODEL=IC \
    CP_SPLIT=chronological \
    BASELINES="all-classical external:casflow external:ctcp" \
    ./sbatch/pipeline.sbatch

echo "=== done: results/cascade_prediction/$DATASET/conditions/report.md ==="
