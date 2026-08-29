#!/bin/bash
#SBATCH --job-name=cr_protocol
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

# The four SETTINGS of research/cascade_reconstruction.md §2.7 and the observation
# sweep of §8.3, on ONE graph, each into its own
# results/cascade_reconstruction/<dataset>/<run>/.
#
# Usage: sbatch sbatch/cascade_reconstruction/sweep_protocol.sbatch
#        DATASET=ca_grqc sbatch sbatch/cascade_reconstruction/sweep_protocol.sbatch
#
# §8.3 is emphatic that these are four separate PROTOCOLS rather than four knobs:
# a decoder selected under final_snapshot is solving a different problem from one
# selected under partial_times, so the rows are reported separately and NEVER
# pooled. Each gets its own run directory for exactly that reason.
#
#   partial_times    (c-i) a subsample of the infected set WITH activation times.
#                    Xiao SDM'18's OrderedSteinerTree input exactly, and the
#                    easiest of the four: the observed times constrain the tree
#   partial_nodes    (a) the same subsample with times WITHHELD. The
#                    Xiao/Sadikov/NetFill regime: the ordering has to be inferred
#                    before any parent can be
#   final_snapshot   (c-ii) the terminal state alone. DITTO's DASH formulation,
#                    the hardest published one, and THE ONE TO LEAD WITH: it is
#                    where a learned kernel should beat a mean-field beta-hat, and
#                    it is the only setting the wired `external:ditto` arm can
#                    answer at all
#   hidden_nodes     (d) partial_times plus nodes deleted from the GRAPH. Untested
#                    territory per §2.7, which is why it runs last
#
# The observation-rate sweep is the fifth axis and the one §8.2 trap 1 warns
# about: this literature uses the symbol sigma for both the report rate and its
# complement, so the runs are labelled with the direction spelled out.

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

DATASET="${DATASET:-jazz}"

run_axis () {
    local label="$1"; shift
    echo "=== $label ==="
    env TASK=cascade_reconstruction DATASET="$DATASET" RUN="$label" RUN_JOBID=0 \
        COMPARE=1 GEN_MODELS="IC LT" "$@" \
        ./sbatch/pipeline.sbatch || echo "[cr_protocol] $label FAILED, continuing"
}

# 1-4. The four settings, at the default 30% report rate
run_axis setting_partial_times  CR_SETTING=partial_times  DIFFUSION_MODEL=IC
run_axis setting_partial_nodes  CR_SETTING=partial_nodes  DIFFUSION_MODEL=IC
run_axis setting_final_snapshot CR_SETTING=final_snapshot DIFFUSION_MODEL=IC
run_axis setting_hidden_nodes   CR_SETTING=hidden_nodes   CR_HIDDEN_RATE=0.1 \
    DIFFUSION_MODEL=IC

# 5-7. The observation-rate axis. `reported_10` means each infected node was
#      REPORTED with probability 0.1, i.e. 90% of the cascade is unobserved: the
#      direction is in the label because reading it backwards is the documented
#      hazard, and Sadikov's own finding is that correction beats doing nothing
#      only below a threshold (§8.2 trap 2), so the crossover is the result
run_axis reported_10 CR_SETTING=partial_times CR_OBSERVATION_RATE=0.1 DIFFUSION_MODEL=IC
run_axis reported_50 CR_SETTING=partial_times CR_OBSERVATION_RATE=0.5 DIFFUSION_MODEL=IC
run_axis reported_90 CR_SETTING=partial_times CR_OBSERVATION_RATE=0.9 DIFFUSION_MODEL=IC

# 8. The reward axis itself. §8.3: "a reader cannot interpret a program-search
#    result without knowing what the search was told to want", so lambda is swept
#    rather than asserted. lambda = 0 is the NODE-ONLY protocol §2.6 says makes a
#    search discard the tree: it is here as the control that demonstrates that,
#    not as a setting to report a headline from
run_axis tree_weight_00 CR_SETTING=partial_times CR_TREE_WEIGHT=0.0 DIFFUSION_MODEL=IC
run_axis tree_weight_09 CR_SETTING=partial_times CR_TREE_WEIGHT=0.9 DIFFUSION_MODEL=IC

# 9. Dynamics. LT has NO transmission edge: activation is a threshold crossing
#    over the whole active neighbourhood, so its ground truth is a parent SET and
#    its path precision is structurally easier than IC's. The two are never
#    compared (§2.6); this row exists so the difference is measured rather than
#    assumed
#    No @oracle arm under LT: the analytic oracle is IC-only (thresholds are drawn
#    per episode and never stored), and the pipeline refuses the pairing up front,
#    so the MC arm is the LT ceiling here
run_axis dynamics_lt CR_SETTING=partial_times DIFFUSION_MODEL=LT \
    ARMS="routing evolve_free@native evolve_free@monte_carlo evolve_free@world_model decode_free@world_model"

echo "[cr_protocol] done -> results/cascade_reconstruction/$DATASET/"
