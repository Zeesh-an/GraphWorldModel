#!/bin/bash
#SBATCH --job-name=ib_gen_email
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Two-cascade transitions on email-Eu-core: THE graph to generate first for this
# task. It is the one benchmark reported by SandIMIN, both Xie papers AND NIE
# (research/influence_blocking.md §6.1), so it is the single place our table can be
# put next to both halves of a literature whose two halves share no metric (§7).
#
# Usage: sbatch sbatch/influence_blocking/generate_data_email_eu_core.sbatch
#
# |S_N| = 1% of N is 10 nodes, which is SandIMIN's own default misinformation seed
# count. §5.4 is why it stays small: at |S_N| = 1000 on NetHEPT even 1000 blockers
# remove 17% of the negative spread, so a large rumour puts every method in a
# regime where nothing works and every arm ties at "barely anything".
#
# All four levers are injected so ONE checkpoint serves every lever: the head has
# to have seen a counter-seed, a deletion, a cut and a reweight to predict any of
# them. `none` is in the blocker list on purpose: those episodes are the unopposed
# sigma(S_N, empty) reference every prevented-influence number divides by.

set -euo pipefail
cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p results logs

python -m data.generate_wm_data \
    --task influence_blocking --dataset email_eu_core --run "${RUN:-default}" \
    --competitive --remove-semantics blocked \
    --tie-break "${TIE_BREAK:-auto}" --positive-prob "${POSITIVE_PROB:-shared}" \
    --negative-pct "${NEGATIVE_PCT:-1.0}" \
    --negative-selectors random degree pagerank \
    --blocker-selectors none random proximity degree pagerank \
    --models IC LT --prob-model weighted \
    --action-ops add_node remove_node remove_edge set_edge_weight \
    --rollouts "${ROLLOUTS:-20}" --horizon 10 \
    --inject-p 0.4 --cf-prob 0.4 --cf-branches 2 \
    --mc-marginals 30 --budget-pct-range 1 20 --seed 42

echo "[ib_gen] done -> results/influence_blocking/email_eu_core/${RUN:-default}/data"
