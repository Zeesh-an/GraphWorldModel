#!/bin/bash
#SBATCH --job-name=epi_datasets
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=168:00:00

# The full ladder across every matched graph, ordered so a loss arrives where it
# means the most.
#
# Usage: sbatch sbatch/epidemic_control/sweep_datasets.sbatch
#        DATASETS="hospital_lh10 primary_school" sbatch ...
#
# The order is the argument. research/epidemic_control.md §6.6 ranks what to add
# and §7 records what each graph is comparable to:
#
#   1-3  The SOCIOPATTERNS traces: the empirical networks this literature is about,
#        and the ones where community structure is real rather than planted. NO
#        PUBLISHED BASELINE exists on any of them (§7), so they are a realism
#        result, not a comparison. `hospital_lh10` is 75 nodes and runs in minutes;
#        `primary_school` has 11 ground-truth classes; `malawi_village` is the
#        SPARSEST at avg degree 8.1, which §8.3 names as exactly where
#        `acquaintance_immunization` most embarrasses methods that read the graph.
#   4    `football`: near-regular degree and 12 known conferences, the control
#        that shows what happens when the heavy tail the degree heuristic feeds on
#        is gone.
#   5-7  The SPECTRAL LINE's benchmarks, with published `lambda_1` per graph
#        (§5.2): Oregon-1 58.72, Oregon-2 70.74, and both reproduce to the decimal.
#        These are where our eigendrop column can be read next to somebody else's.
#   8    `ca_grqc`: RLGN's own graph at 5,242 / 14,496, the single most comparable
#        published table (§5.1). Warning: its budget convention is PER STEP and
#        ours is one-shot, which §8.2 trap 3 records as non-comparable: cite its
#        numbers as context, not as a run.
#   9    `deezer_ro`: RLGN's GEMSEC-RO column, loaded at its own true edge count
#        because that paper's Table S4 pairs Romania's nodes with Hungary's edges.
#   10   `power_grid`: the mesh graph §9.4's prediction turns on. MIND's Table 5
#        puts Explosive Immunization at 80.6 on `eu-powergrid` against FINDER's
#        161.7, so a physics heuristic beating a learned method by 2x here is the
#        limitation we predict and should CONFIRM rather than discover. Last on
#        purpose.

set -euo pipefail
cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p results logs

DATASETS="${DATASETS:-hospital_lh10 primary_school malawi_village football oregon1 oregon2_010331 brightkite ca_grqc deezer_ro power_grid}"

for dataset in $DATASETS; do
    echo "=== $dataset ==="
    TASK=epidemic_control \
    DATASET="$dataset" \
    RUN="${RUN:-default}" \
    RUN_JOBID=0 \
    COMPARE=1 \
    GEN_MODELS=SIR \
    DIFFUSION_MODEL=SIR \
    BUDGET_PCTS="${BUDGET_PCTS:-1 5 10 20}" \
        ./sbatch/pipeline.sbatch || echo "[epi_datasets] $dataset FAILED, continuing"
done

echo "[epi_datasets] done -> results/epidemic_control/*/${RUN:-default}"
