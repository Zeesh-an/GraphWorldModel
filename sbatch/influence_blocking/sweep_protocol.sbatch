#!/bin/bash
#SBATCH --job-name=ib_protocol
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00

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
# the targets. The other two do not.

set -euo pipefail
cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p results logs

DATASET="${DATASET:-email_eu_core}"
AXES="${AXES:-tie_break attacker delay}"

for axis in $AXES; do
  case "$axis" in
    tie_break)
      for rule in negative positive fixed; do
        echo "=== tie_break=$rule ==="
        TASK=influence_blocking DATASET="$DATASET" RUN="tie_$rule" RUN_JOBID=0 \
        COMPARE=1 GEN_MODELS=IC TIE_BREAK="$rule" BUDGETS="10 20 30 40 50" \
            ./sbatch/pipeline.sbatch || echo "[ib_protocol] tie_$rule FAILED"
      done ;;
    attacker)
      for selector in random degree pagerank betweenness; do
        echo "=== attacker=$selector ==="
        TASK=influence_blocking DATASET="$DATASET" RUN="attacker_$selector" RUN_JOBID=0 \
        COMPARE=1 GEN_MODELS=IC OUTBREAK_SELECTOR="$selector" \
        BUDGETS="10 20 30 40 50" START_STAGE=agent \
            ./sbatch/pipeline.sbatch || echo "[ib_protocol] attacker_$selector FAILED"
      done ;;
    delay)
      for r in 0 1 2 4; do
        echo "=== detection_delay=$r ==="
        TASK=influence_blocking DATASET="$DATASET" RUN="delay_$r" RUN_JOBID=0 \
        COMPARE=1 GEN_MODELS=IC DETECTION_DELAY="$r" \
        BUDGETS="10 20 30 40 50" START_STAGE=agent \
            ./sbatch/pipeline.sbatch || echo "[ib_protocol] delay_$r FAILED"
      done ;;
  esac
done

echo "[ib_protocol] done -> results/influence_blocking/$DATASET/"
