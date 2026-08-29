#!/bin/bash
#SBATCH --job-name=cnd_gen_power_grid
#SBATCH --output=logs/%x_%j.out
# (logs/ must exist before `sbatch`: Slurm opens the .out file before the job runs)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Containment transitions on the US power grid (4,941 / 6,594).
#
# Usage: sbatch sbatch/critical_node_detection/generate_data_power_grid.sbatch
#
# This is the single best-matched benchmark in research/critical_node_detection.md
# §7: CoreHD, BPD and NIRM all publish absolute numbers on this exact 4,941/6,594
# file, and §9.4 predicts we LOSE on it: diameter 46 exhausts a 3-6 layer
# receptive field, and MIND's Table 5 puts FINDER at 161.7 on `eu-powergrid`
# against EI's 80.6. Run it first and report the loss as predicted-and-confirmed
# rather than discovering it late.
#
# --task supplies remove_node generation and `blocked` semantics from the
# registry, so no --gen-action-ops or --remove-semantics flag appears here.
# betweenness is dropped from the selectors: exact Brandes is O(V*E) in pure
# networkx and takes hours at 5K nodes.

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

# SINGLE REAL GRAPH: no graph-disjoint split exists for one graph, so this
# dataset is split per episode and its test metrics are in-graph. Do not pool
# them with a graph_disjoint synthetic run. (A comment cannot sit inside the
# assignment prefix below: a backtick word there turns SPLIT_MODE=... into the
# command name.)
TASK=critical_node_detection \
DATASET=power_grid \
RUN="${RUN:-default}" \
RUN_JOBID=0 \
START_STAGE=data END_STAGE=data \
GEN_MODELS="IC LT" \
GEN_ALGORITHMS="random degree pagerank" \
ROLLOUTS="${ROLLOUTS:-30}" \
MC_MARGINALS="${MC_MARGINALS:-30}" \
INJECT_P="${INJECT_P:-0.6}" \
BUDGET_PCT_RANGE="1 20" \
SPLIT_MODE=episode_random \
    ./sbatch/pipeline.sbatch

echo "[cnd_gen] done -> results/critical_node_detection/power_grid/${RUN:-default}/data"
