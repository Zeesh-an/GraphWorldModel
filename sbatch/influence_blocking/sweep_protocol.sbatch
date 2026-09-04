#!/bin/bash
#SBATCH --job-name=ib_protocol
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

# The three protocol axes this literature reports and mostly does not state:
# the TIE-BREAK (§8.4), the ATTACKER model (§8.3), and Budak's DETECTION DELAY.
#
# Usage: sbatch sbatch/influence_blocking/sweep_protocol.sbatch
#        AXES="tie_break" sbatch sbatch/influence_blocking/sweep_protocol.sbatch
#
# Why each one is here rather than a default nobody looks at:
#
#   tie_break   §8.4: the three rules differ ONLY in how a same-step arrival
#               resolves, which is a one-line change in a simulator and a large
#               change in the reported numbers, and most papers never say which
#               they used. TC-AIBM's three-rule comparison is the exception, and
#               it is published as bar charts with no numeric table (§11), so
#               this sweep produces numbers that do not currently exist.
#
#   attacker    §8.3: every blocking result is conditioned on how S_N was chosen
#               and the papers disagree: SandIMIN draws randomly from the top
#               200, TC-AIBM reports Degree / IMM / PageRank, CLDAG sweeps
#               largest-degree and random. Our spine selectors already cover it.
#
#   delay       Budak's r, and CLDAG's "first mover has a clear advantage" in
#               experimental form. The rumour runs r steps unopposed before the
#               blocker may act; this is what makes the task non-trivial.
#
# The tie-break axis REGENERATES data per value, because the rule is baked into
# the targets. The other two are read only by the agent stage, so they start
# there on SOURCE_RUN's transitions and checkpoint (default: the `default` run
# the generate script and the dataset sweep write), linked into each axis run
# because Layout is per run. FORCE=1 on a linked run rewrites SOURCE_RUN's copy.

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
source venv/bin/activate
mkdir -p results logs

DATASET="${DATASET:-email_eu_core}"
AXES="${AXES:-tie_break attacker delay}"
SOURCE_RUN="${SOURCE_RUN:-default}"

link_run () {
    local src="results/influence_blocking/$DATASET/$SOURCE_RUN"
    local dst="results/influence_blocking/$DATASET/$1"
    [ -f "$src/data/metadata.json" ] \
        || { echo "[ib_protocol] $src/data has no metadata.json: generate into RUN=$SOURCE_RUN first" >&2; return 1; }
    mkdir -p "$dst"
    for part in data world_model; do
        if [ -e "$src/$part" ] && [ ! -e "$dst/$part" ]; then
            ln -s "$REPO_ROOT/$src/$part" "$dst/$part"
        fi
    done
}

for axis in $AXES; do
  case "$axis" in
    tie_break)
      for rule in negative positive fixed; do
        echo "=== tie_break=$rule ==="
        TASK=influence_blocking DATASET="$DATASET" RUN="tie_$rule" RUN_JOBID=0 \
        GEN_MODELS=IC TIE_BREAK="$rule" BUDGETS="10 20 30 40 50" \
            ./sbatch/pipeline.sbatch || echo "[ib_protocol] tie_$rule FAILED"
      done ;;
    attacker)
      for selector in random degree pagerank betweenness; do
        echo "=== attacker=$selector ==="
        link_run "attacker_$selector"
        TASK=influence_blocking DATASET="$DATASET" RUN="attacker_$selector" RUN_JOBID=0 \
        GEN_MODELS=IC OUTBREAK_SELECTOR="$selector" \
        BUDGETS="10 20 30 40 50" START_STAGE=agent \
            ./sbatch/pipeline.sbatch || echo "[ib_protocol] attacker_$selector FAILED"
      done ;;
    delay)
      for r in 0 1 2 4; do
        echo "=== detection_delay=$r ==="
        link_run "delay_$r"
        TASK=influence_blocking DATASET="$DATASET" RUN="delay_$r" RUN_JOBID=0 \
        GEN_MODELS=IC DETECTION_DELAY="$r" \
        BUDGETS="10 20 30 40 50" START_STAGE=agent \
            ./sbatch/pipeline.sbatch || echo "[ib_protocol] delay_$r FAILED"
      done ;;
  esac
done

echo "[ib_protocol] done -> results/influence_blocking/$DATASET/"
