#!/bin/bash
#SBATCH --job-name=aoim_sweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

# The four runnable branches of research/adaptive_online_im.md §1, back to back
# on one dataset, each into its own `run` so they sit side by side under
# results/adaptive_online_im/<dataset>/.
#
# Usage: DATASET=powerlaw_cluster sbatch sbatch/adaptive_online_im/sweep_branches.sbatch
#
# Branch (c), continuous-time, is absent on purpose: it needs a new simulator, a
# new state and a new head (§2.4c), and the doc's own recommendation is to defer
# it. Cumulative regret is absent for the reason recorded in §11.

set -euo pipefail
# Repo root, derived from this script's own location so the file works in
# any checkout rather than one person's home directory. Set REPO_ROOT to
# override, e.g. when submitting a copy staged outside the tree.
# Under SLURM this file executes as a spool copy, so BASH_SOURCE cannot locate
# the checkout from inside a job; the directory it was submitted from is the
# repo root when launched as documented, and REPO_ROOT overrides either.
if [ -z "${REPO_ROOT:-}" ]; then
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        REPO_ROOT="$SLURM_SUBMIT_DIR"
    else
        REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    fi
fi
cd "$REPO_ROOT"

# Activate the project virtualenv, unless one is already active or VENV
# points somewhere else. Skipped silently when neither exists, so the
# script still runs under a module-loaded or conda interpreter.
VENV="${VENV:-venv}"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
fi
mkdir -p results logs

DATASET="${DATASET:-powerlaw_cluster}"
BUDGET_PCTS="${BUDGET_PCTS:-1 5 10 20}"
# Reuse one checkpoint across every branch: the transitions are identical, only
# the evaluation loop differs (§9.2), so retraining per branch would burn GPU
# hours to relearn the same function.
WM="${WM:-results/adaptive_online_im/$DATASET/default/world_model/sage_IC.json}"

common=(TASK=adaptive_online_im DATASET="$DATASET" RUN_JOBID=0 COMPARE=1)

# (a) Adaptive IM, k-sweep: r fixed, k varies with the budget. Han et al.'s
#     k-setting. This is the headline run and the one the gap table comes from.
env "${common[@]}" RUN=adaptive_r4 ROUNDS=4 \
    BUDGET_PCTS="$BUDGET_PCTS" ./sbatch/pipeline.sbatch

# (a) Adaptive IM, b-sweep: b fixed, r derived. Han et al.'s b-setting, and the
#     axis their own AdaptIM-1 runs out of memory on at small b.
env "${common[@]}" RUN=adaptive_b2 PER_ROUND_BUDGET=2 \
    WM_RESULTS_JSON="$WM" SKIP_STAGES=data,train \
    BUDGET_PCTS="$BUDGET_PCTS" ./sbatch/pipeline.sbatch

# (a) Myopic feedback: the policy sees only the current wave. The one axis the
#     theory literature actually cares about (§1.1), and the setting where
#     adaptive submodularity fails.
env "${common[@]}" RUN=adaptive_myopic ROUNDS=4 FEEDBACK_MODEL=myopic \
    WM_RESULTS_JSON="$WM" SKIP_STAGES=data,train \
    BUDGET_PCTS="$BUDGET_PCTS" ./sbatch/pipeline.sbatch

# (d) Dynamic / streaming: the graph rewires under the policy while it seeds.
#     Applied to every arm, so a static plan's loss here is the measurement.
env "${common[@]}" RUN=streaming ROUNDS=4 EDIT_RATE=0.02 \
    WM_RESULTS_JSON="$WM" SKIP_STAGES=data,train \
    BUDGET_PCTS="$BUDGET_PCTS" ./sbatch/pipeline.sbatch

# (e) Multi-round: 3 separate campaigns of k seeds, scored on their union.
env "${common[@]}" RUN=multi_round ROUNDS=4 CAMPAIGNS=3 \
    WM_RESULTS_JSON="$WM" SKIP_STAGES=data,train \
    BUDGET_PCTS="$BUDGET_PCTS" ./sbatch/pipeline.sbatch

# (b) Online / bandit information state: the world model is trained on ones
#     instead of the true p(u->v), so it must infer transmission from structure.
#     Needs its own checkpoint and --head structured (structured_residual
#     anchors on logit(w) and cannot be masked).
env "${common[@]}" RUN=w_hidden ROUNDS=4 HIDE_EDGE_WEIGHTS=1 HEAD=structured \
    BUDGET_PCTS="$BUDGET_PCTS" ./sbatch/pipeline.sbatch

echo "[aoim_sweep] done -> results/adaptive_online_im/$DATASET/"
