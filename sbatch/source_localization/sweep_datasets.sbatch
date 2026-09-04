#!/bin/bash
#SBATCH --job-name=sl_sweep
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

# The six-condition ladder plus arm A on every graph this task has a published
# comparison for, back to back, each into results/source_localization/<name>/.
#
# Usage: sbatch sbatch/source_localization/sweep_datasets.sbatch
#        DATASETS="jazz karate" sbatch sbatch/source_localization/sweep_datasets.sbatch
#
# The order runs from the graphs whose published counts match ours byte-for-byte
# to the ones where they do not, so a comparability caveat never arrives as a
# surprise (research/source_localization.md §6.1, §6.4, §11):
#
#   karate       34 / 78, in four papers; the smallest full sweep there is, and
#                the graph the 2026 GNN benchmark shows a 72.9% top-5 ceiling on
#   dolphins     62 / 159, in IVGD, GraphSL and the GNN benchmark. THE graph this
#                literature uses that no other task file needed
#   jazz         198 / 2,742: four papers agree to the digit. (key)
#   netscience   1,589 / 2,742: comparable to IVGD, GraphSL and SIDSL, and NOT
#                to SL-VAE, whose 1,565 / 13,532 is a different graph that is
#                undocumented and not publicly downloadable (§6.4.1). Say so in
#                any table that carries this row.
#   cora_ml      2,810 / 7,981: three papers agree; our loader already applies
#                graph2gauss standardize()
#   power_grid   4,941 / 6,594: four papers agree. The largest matched graph,
#                and a diameter-46 mesh, so it is also where a k-layer encoder's
#                receptive field runs out
#   deezer       47,538 / 222,887: IVGD's SCALABILITY column and nothing else,
#                so there is no F1 row to compare against. It is here for the
#                COST axis of §2.3.2, which needs a size the others cannot give.
#
# Warning: NO published number in this literature is under IC or LT. SL-VAE uses SI/SIR,
# SL-Diff / SIDSL / DDMSL use real cascades, the GNN benchmark uses SIR. Our
# graphs match byte-for-byte and our DYNAMICS do not, so every number this sweep
# produces is self-contained (§11). That is the biggest comparability gap in the
# file, and it is bigger than any graph-version disagreement.

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

DATASETS="${DATASETS:-karate dolphins jazz netscience cora_ml power_grid}"

for dataset in $DATASETS; do
    echo "=== $dataset ==="
    # exact Brandes is O(V*E) in pure networkx; drop it past a few thousand nodes
    case "$dataset" in
      karate|dolphins|jazz|netscience) selectors="random degree pagerank betweenness" ;;
      *)                               selectors="random degree pagerank" ;;
    esac
    # karate is a synthetic family here (nx.karate_club_graph), so
    # pipeline.sbatch would default it to a 40-graph population: one instance
    graphs=1

    TASK=source_localization \
    DATASET="$dataset" \
    RUN="${RUN:-default}" \
    RUN_JOBID=0 \
    \
    NUM_GRAPHS="$graphs" \
    GEN_MODELS=IC \
    GEN_ALGORITHMS="$selectors" \
    SL_INSTANCES="${SL_INSTANCES:-20}" \
    SL_OBSERVATION="${SL_OBSERVATION:-marginal}" \
        ./sbatch/pipeline.sbatch || echo "[sl_sweep] $dataset FAILED, continuing"
done

echo "[sl_sweep] done -> results/source_localization/"
