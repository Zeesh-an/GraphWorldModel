#!/bin/bash
#SBATCH --job-name=cnd_sweep
#SBATCH --output=logs/%x_%j.out
# (logs/ must exist before `sbatch`: Slurm opens the .out file before the job runs)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00
# The ladder trains f_theta inline in this allocation (pipeline.sbatch does
# not resubmit from inside a job), so the sweep needs the GPU itself
#SBATCH --gres=gpu:1

# The whole six-condition ladder on every dataset this task has a published
# comparison for, back to back, each into results/critical_node_detection/<name>/.
#
# Usage: sbatch sbatch/critical_node_detection/sweep_datasets.sbatch
#        DATASETS="ppi_yeast crime" sbatch sbatch/critical_node_detection/sweep_datasets.sbatch
#
# The order is deliberate: it runs from the graph classes where a learned model
# should do well to the one where research/critical_node_detection.md §9.4
# predicts it fails, so a bad power-grid number arrives as a confirmed prediction
# rather than as the first result anyone sees:
#
#   crime, corruption   tiny (754 / 309), reported by GND, FINDER, NIRM and SPR;
#                       cheapest full sweep, and where an agent arm has room to
#                       actually search
#   ppi_yeast           the most widely shared benchmark in the literature (§7)
#   hamsterster         dense social, the graph class degree heuristics own
#   intnet1             hub-dominated AS graph: hardest place to beat HDA
#   usair97, openflights transport, moderate diameter
#   road_eu, power_grid MESH graphs, diameter 46. §9.4's predicted loss.
#
# Every arm faces the same outbreak (deterministic in SEED), and the structural
# columns land in each run's report.md beside the spread column, which is what
# lets a row be placed against the published dismantling tables at all (§8.3).
#
# The outbreak is 10% of N by registry default, and that is load-bearing: the
# planner is told where it is, so a budget at or above the outbreak's one-hop
# ring is trivial for any outbreak-aware arm (§8.2 trap 10). On hamsterster and
# intnet1 (dense, hub-dominated) even 10% can leave the ring under the 20%
# budget; the report's ring note names any budget that cleared it, and
# `frontier_removal` in the default pool is the row an agent arm has to beat.

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

DATASETS="${DATASETS:-crime corruption ppi_yeast hamsterster intnet1 usair97 openflights road_eu power_grid}"
BUDGET_PCTS="${BUDGET_PCTS:-1 5 10 20}"

for dataset in $DATASETS; do
    echo "=== $dataset ==="
    # betweenness selection is O(V*E) exact Brandes; drop it past a few thousand
    case "$dataset" in
      crime|corruption|usair97|road_eu) selectors="random degree pagerank betweenness" ;;
      *)                                selectors="random degree pagerank" ;;
    esac

    TASK=critical_node_detection \
    DATASET="$dataset" \
    RUN="${RUN:-default}" \
    RUN_JOBID=0 \
    COMPARE=1 \
    GEN_MODELS=IC \
    GEN_ALGORITHMS="$selectors" \
    BUDGET_PCTS="$BUDGET_PCTS" \
        ./sbatch/pipeline.sbatch || echo "[cnd_sweep] $dataset FAILED, continuing"
done

echo "[cnd_sweep] done -> results/critical_node_detection/"
