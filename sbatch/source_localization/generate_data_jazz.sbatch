#!/bin/bash
#SBATCH --job-name=sl_gen_jazz
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00

# Labelled source-localization episodes on Jazz (198 / 2,742).
#
# Usage: sbatch sbatch/source_localization/generate_data_jazz.sbatch
#
# Jazz is the single cleanest comparison target in this literature: FOUR papers
# report it at 198 / 2,742 to the digit: IVGD Table 2, SL-VAE Table 5, GraphSL
# Table 1, SIDSL Table 5 (research/source_localization.md §6.1), and it is small
# enough that a full six-condition ladder finishes in hours.
#
# --task supplies the whole data shape from the registry, so no --gen-action-ops
# flag appears here: a recover task generates DIFFUSION-ONLY episodes, with no
# injected actions at any t > 0. That is not a tuning choice. An episode carrying
# a mid-cascade injection has an observation its t=0 seed set did not produce, so
# its (x, y) label would be a lie (§2.1), and the registry asserts it.
#
# INJECT_P and the counterfactual forks are inert for the same reason (empty
# action ops means sample_injection always returns NULL and counterfactual_actions
# produces nothing), so they are left at their defaults rather than zeroed: the
# registry is the single place that decides.
#
# BUDGET_PCT_RANGE is the SOURCE fraction band, which is the one generation knob
# this task genuinely cares about. 1-20% brackets SL-VAE's 10%-of-nodes
# convention (§8.4 trap 1) on both sides, so --sl-budget-mode sweep can later cut
# a 5 / 10 / 20% source-fraction axis out of one dataset.

set -euo pipefail
cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p results logs

TASK=source_localization \
DATASET=jazz \
RUN="${RUN:-default}" \
RUN_JOBID=0 \
START_STAGE=data END_STAGE=data \
GEN_MODELS="${GEN_MODELS:-IC LT}" \
GEN_ALGORITHMS="${GEN_ALGORITHMS:-random degree pagerank betweenness}" \
ROLLOUTS="${ROLLOUTS:-60}" \
MC_MARGINALS="${MC_MARGINALS:-30}" \
GEN_HORIZON="${GEN_HORIZON:-10}" \
BUDGET_PCT_RANGE="${BUDGET_PCT_RANGE:-1 20}" \
    ./sbatch/pipeline.sbatch

echo "[sl_gen] done -> results/source_localization/jazz/${RUN:-default}/data"
