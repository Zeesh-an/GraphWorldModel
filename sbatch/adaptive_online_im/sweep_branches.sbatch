#!/bin/bash
#SBATCH --job-name=aoim_sweep
#SBATCH --output=logs/%x_%j.out
# (logs/ must exist before `sbatch`: Slurm opens the .out file before the job runs)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
# The ladder trains f_theta inline in this allocation (pipeline.sbatch does
# not resubmit from inside a job), so the sweep needs the GPU itself
#SBATCH --gres=gpu:1

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
# Repo root: the checkout this file lives in, whenever this file is the real
# one. Under a Slurm spool copy BASH_SOURCE points outside any checkout, and
# only then is the submit directory trusted; an `srun --pty` / `salloc` shell
# also carries SLURM_JOB_ID, so SLURM_SUBMIT_DIR is never taken first. Set
# REPO_ROOT to override. Same rule as sbatch/pipeline.sbatch.
if [ -z "${REPO_ROOT:-}" ]; then
    checkout="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd || true)"
    if [ -n "$checkout" ] && [ -f "$checkout/pipeline/run.py" ]; then
        REPO_ROOT="$checkout"
    elif [ -n "${SLURM_JOB_ID:-}" ] && [ -f "${SLURM_SUBMIT_DIR:-/nonexistent}/pipeline/run.py" ]; then
        REPO_ROOT="$SLURM_SUBMIT_DIR"
    else
        echo "[repo] cannot locate the checkout: ${BASH_SOURCE[0]} is not inside one and" \
             "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset} has no pipeline/run.py; set REPO_ROOT" >&2
        exit 1
    fi
fi
export REPO_ROOT
cd "$REPO_ROOT"
[ -f pipeline/run.py ] || { echo "[repo] REPO_ROOT=$REPO_ROOT has no pipeline/run.py; set REPO_ROOT to the checkout" >&2; exit 1; }
[ -w . ] || { echo "[repo] REPO_ROOT=$REPO_ROOT is not writable, and results/ and logs/ live here" >&2; exit 1; }
echo "[repo] $REPO_ROOT"

# Activate the project virtualenv, unless one is already active or VENV
# points somewhere else. Skipped silently when neither exists, so the
# script still runs under a module-loaded or conda interpreter.
VENV="${VENV:-venv}"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
fi
mkdir -p results logs
# The matrix's synthetic rows are 10,000 nodes (final_results_plan.md section 2);
# the real graphs ignore this
export SYN_NODES="${SYN_NODES:-10000}"

DATASET="${DATASET:-netscience}"
BUDGET_PCTS="${BUDGET_PCTS:-1 5 10 20}"
common=(TASK=adaptive_online_im DATASET="$DATASET" RUN_JOBID=0)

# Reuse (a)'s transitions and checkpoint across every later branch: the
# transitions are identical, only the evaluation loop differs (§9.2), so
# regenerating or retraining per branch would relearn the same function. Layout
# is per run, so the parts are linked into each branch's run; the pipeline sees
# them on disk and skips those stages. FORCE=1 on a linked run rewrites (a)'s.
link_run () {
    local run="$1"; shift
    mkdir -p "results/adaptive_online_im/$DATASET/$run"
    for part in "$@"; do
        [ -e "results/adaptive_online_im/$DATASET/$run/$part" ] \
            || ln -s "$REPO_ROOT/results/adaptive_online_im/$DATASET/adaptive_r4/$part" \
                     "results/adaptive_online_im/$DATASET/$run/$part"
    done
}

# (a) Adaptive IM, k-sweep: r fixed, k varies with the budget. Han et al.'s
#     k-setting. This is the headline run and the one the gap table comes from.
env "${common[@]}" RUN=adaptive_r4 ROUNDS=4 \
    BUDGET_PCTS="$BUDGET_PCTS" ./sbatch/pipeline.sbatch

# (a) Adaptive IM, b-sweep: b fixed, r derived. Han et al.'s b-setting, and the
#     axis their own AdaptIM-1 runs out of memory on at small b.
link_run adaptive_b2 data world_model
env "${common[@]}" RUN=adaptive_b2 PER_ROUND_BUDGET=2 \
    SKIP_STAGES="data train" \
    BUDGET_PCTS="$BUDGET_PCTS" ./sbatch/pipeline.sbatch

# (a) Myopic feedback: the policy sees only the current wave. The one axis the
#     theory literature actually cares about (§1.1), and the setting where
#     adaptive submodularity fails.
link_run adaptive_myopic data world_model
env "${common[@]}" RUN=adaptive_myopic ROUNDS=4 FEEDBACK_MODEL=myopic \
    SKIP_STAGES="data train" \
    BUDGET_PCTS="$BUDGET_PCTS" ./sbatch/pipeline.sbatch

# (d) Dynamic / streaming: the graph rewires under the policy while it seeds.
#     Applied to every arm, so a static plan's loss here is the measurement.
link_run streaming data world_model
env "${common[@]}" RUN=streaming ROUNDS=4 EDIT_RATE=0.02 \
    SKIP_STAGES="data train" \
    BUDGET_PCTS="$BUDGET_PCTS" ./sbatch/pipeline.sbatch

# (e) Multi-round: 3 separate campaigns of k seeds, scored on their union.
link_run multi_round data world_model
env "${common[@]}" RUN=multi_round ROUNDS=4 CAMPAIGNS=3 \
    SKIP_STAGES="data train" \
    BUDGET_PCTS="$BUDGET_PCTS" ./sbatch/pipeline.sbatch

# (b) Online / bandit information state: the world model is trained on ones
#     instead of the true p(u->v), so it must infer transmission from structure.
#     Needs its own checkpoint and --head structured, the default
#     (structured_residual anchors on logit(w) and cannot be masked).
link_run w_hidden data
env "${common[@]}" RUN=w_hidden ROUNDS=4 HIDE_EDGE_WEIGHTS=1 HEAD=structured SKIP_STAGES=data \
    BUDGET_PCTS="$BUDGET_PCTS" ./sbatch/pipeline.sbatch

echo "[aoim_sweep] done -> results/adaptive_online_im/$DATASET/"
