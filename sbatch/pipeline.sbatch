#!/bin/bash
#SBATCH --job-name=gwm
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#
# (the four directives above are only the floor for a direct
#  `sbatch sbatch/pipeline.sbatch`; the submit path below overrides all of
#  them per dataset, and adds --gres only when the run actually needs one)
#
# ============================================================================
# THE pipeline script. data -> train -> agent -> plots -> report, for any
# dataset, any subset of stages, any subset of conditions.
#
# Run it directly and it queues ITSELF with the right allocation:
#
#     DATASET=ba ./sbatch/pipeline.sbatch
#
# Every `python -m pipeline.run` flag is an environment variable. Ordinary
# shell quoting, so a list value is just a quoted string:
#
#     DATASET=sbm ARMS="evolve_free@oracle routing" ./sbatch/pipeline.sbatch
#
# GPU IS AUTOMATIC. Before submitting, the script asks the pipeline whether
# this exact set of flags ever puts a tensor on a device, and requests a GPU
# only if it does:
#
#     GPU     training f_theta, any @world_model arm, and the learned
#             external repos (deepim, touplegdd, glie, lense, him, gcomb,
#             iminfector, moeim, piano)
#     CPU     data generation, plots, the report, the classical pool, the
#             routing/native/monte_carlo/oracle arms, and the C++ repos
#             (opim, ssa, subsim, imm, tim, oim)
#
# So an oracle-only or baselines-only sweep queues CPU-only and starts sooner,
# and DEVICE follows the same decision. See it without submitting anything:
#
#     DATASET=ba DRY_RUN=1 ./sbatch/pipeline.sbatch
#
# Override it if you need to: GRES=gpu:2, or GRES=none to force CPU.
#
# RESULTS GO IN results/<task>/<dataset>/<run>_<jobid>/, the same way logs go to
# logs/%x_%j.out, so two jobs on the same dataset can never overwrite each
# other. The job id is printed on the first line of the .out, along with the
# exact command to resume that run. Set RUN_JOBID=0 for a stable
# results/<task>/<dataset>/<run>/ that every rerun appends to, which is what
# you want when you are deliberately resuming or chaining.
#
# TASK selects the graph task and is the first results level. Only
# influence_maximization is runnable today; `python -c "from pipeline.tasks
# import tasks; print(sorted(tasks))"` lists them, and research/<task>.md is
# the literature review for each.
#
# Rules for the env vars:
#   * list-valued vars are space-separated inside quotes:  GEN_MODELS="IC LT"
#   * the literal string `none` empties a list flag:       BASELINES=none
#   * booleans are 1 / 0:                                  COMPARE=1  CREDIT=0
#   * anything unset falls back to a per-dataset default (see DEFAULTS below)
#
# Slurm-side knobs, all optional: JOB_NAME, GRES, MEM, TIME, CPUS, PARTITION,
# and SBATCH_ARGS for anything else to pass through verbatim.
#
# ============================================================================
# TEMPLATES  (replace <dataset>; every one is copy-pasteable)
# ============================================================================
#
# --- 1. FULL PIPELINE -------------------------------------------------------
# All five stages, all seven conditions, four budgets. The headline run.
# Requests a GPU (it trains f_theta).
#
#   DATASET=ba ./sbatch/pipeline.sbatch
#
#   DATASET=nethept MEM=64G TIME=48:00:00 ./sbatch/pipeline.sbatch
#
#
# --- 2. ORACLE ONLY ---------------------------------------------------------
# Condition 5 alone: the agent sees the true dynamics. No world model, so the
# train stage drops itself and this queues CPU-only. The ceiling, and the
# fastest way to iterate on the agent itself.
#
#   DATASET=sbm ARMS=evolve_free@oracle BASELINES=none \
#       ./sbatch/pipeline.sbatch
#
#
# --- 3. MC + ORACLE + WM ----------------------------------------------------
# The evaluator ablation proper: conditions 4, 5, 6 with the method held fixed,
# so the ONLY thing that varies is the inner-loop feedback. GPU.
#
#   DATASET=sbm \
#   ARMS="evolve_free@monte_carlo evolve_free@oracle evolve_free@world_model" \
#       ./sbatch/pipeline.sbatch
#
#
# --- 4. WORLD MODEL ONLY ----------------------------------------------------
# Just our method (condition 6), nothing to compare against. GPU.
#
#   DATASET=ba ARMS=evolve_free@world_model BASELINES=none \
#       ./sbatch/pipeline.sbatch
#
#
# --- 5. MONTE CARLO ONLY ----------------------------------------------------
# Condition 4 alone: averaged simulator rollouts as feedback. CPU-only.
# MC_RUNS sets both its cost and its noise.
#
#   DATASET=sbm ARMS=evolve_free@monte_carlo BASELINES=none \
#   MC_RUNS=200 ./sbatch/pipeline.sbatch
#
#
# --- 5b. NATIVE AGENT ONLY --------------------------------------------------
# Condition 3: real executions only, NATIVE_MC_RUNS episode(s) per candidate.
# Keep it at 1 for the honestly model-free baseline: the whole point of this
# arm is that every number the agent sees costs a real episode. CPU-only.
#
#   DATASET=sbm ARMS=evolve_free@native BASELINES=none \
#   NATIVE_MC_RUNS=1 ./sbatch/pipeline.sbatch
#
#
# --- 6. BASELINES ONLY ------------------------------------------------------
# No LLM at all: the classical pool (condition 1) plus every installed
# published repo (condition 7). ARMS=none removes conditions 2-6.
# CPU-only unless a learned repo is in the set.
#
#   DATASET=netscience ARMS=none BASELINES=all ./sbatch/pipeline.sbatch
#
#   DATASET=netscience ARMS=none BASELINES=all-classical ./sbatch/pipeline.sbatch
#   DATASET=netscience ARMS=none BASELINES=all-external  ./sbatch/pipeline.sbatch
#
# Install the external repos first (once per machine):
#   python -m baselines.setup_baselines --all
#   python -m baselines.setup_baselines --list
#
#
# --- 7. CERTAIN BASELINES ONLY ----------------------------------------------
# Any mix of library algorithms and external:<name> repos.
#
#   DATASET=netscience ARMS=none \
#   BASELINES="celf_pp imm degree_discount external:opim external:ssa" \
#       ./sbatch/pipeline.sbatch
#
# Library algorithms (condition 1): any of:
#   adaptive_greedy betweenness_seeds celf celf_local_search celf_pp
#   closeness_seeds cofim collective_influence community_celf community_im
#   community_ris degree_discount degree_ris_refine eigenvector_seeds
#   filtered_ris genetic_algorithm high_degree hill_climbing imm irie
#   kshell_seeds ldag mia_pmia pagerank_greedy pagerank_seeds random_seeds
#   ris_basic simulated_annealing skim sp1m ssa static_greedy tim
#   vanilla_greedy voterank weighted_degree
#
# External repos (condition 7): external:<name> for any of:
#   learned (GPU):  moeim touplegdd deepim gcomb iminfector glie lense him piano
#   C++ (CPU):      opim ssa subsim imm tim oim
#
#
# --- 8. NO BASELINES --------------------------------------------------------
# Conditions 2-6 only, nothing classical. For when the baseline numbers for
# this dataset are already on disk and only the agent side changed.
#
#   DATASET=ba BASELINES=none ./sbatch/pipeline.sbatch
#
#
# --- 9. SKIP WM TRAINING, USE ORACLE ----------------------------------------
# Two ways, and they mean different things:
#
#   (a) no world-model arm at all -> train removes itself, queues CPU-only:
#   DATASET=sbm \
#   ARMS="routing evolve_free@native evolve_free@monte_carlo evolve_free@oracle" \
#       ./sbatch/pipeline.sbatch
#
#   (b) keep a @world_model arm but reuse a checkpoint trained earlier:
#   DATASET=ba SKIP_STAGES=train ./sbatch/pipeline.sbatch
#   (b) still asks for a GPU, because the arm rolls f_theta out. It fails
#   loudly if the run's world_model/ has no checkpoint: it reuses, it
#   does not silently fall back to the oracle.
#
#
# --- 10. FROM CERTAIN STAGES ONLY -------------------------------------------
# Stages, in order: data train agent plots report
#
#   START_STAGE=agent                 agent onwards, reuse data + checkpoint
#   START_STAGE=plots                 redo the reporting half only
#   START_STAGE=data END_STAGE=data   generate transitions and stop
#   START_STAGE=train END_STAGE=train retrain the world model and stop
#   START_STAGE=agent END_STAGE=agent run the sweep, no plots or report
#
#   DATASET=ba START_STAGE=agent ./sbatch/pipeline.sbatch
#
# Data generation alone (CPU-only: this is what the old wm_generate_data_*
# scripts did):
#
#   DATASET=nethept START_STAGE=data END_STAGE=data MEM=64G \
#       ./sbatch/pipeline.sbatch
#
# Training alone on already-generated data (what wm_train.sbatch did):
#
#   DATASET=ba START_STAGE=train END_STAGE=train WM_MODEL=sage \
#       ./sbatch/pipeline.sbatch
#
# Finished stages are detected on disk and reused, so re-running the whole
# thing after a crash costs nothing for the parts that completed. FORCE=1
# recomputes the selected stages regardless.
#
#
# --- 11. PLOTS ONLY ---------------------------------------------------------
# Regenerate every figure from results already on disk. Seconds, CPU-only, no
# LLM calls. PLOTS then REPORT is the usual pair.
#
#   DATASET=ba START_STAGE=plots END_STAGE=plots ./sbatch/pipeline.sbatch
#   DATASET=ba START_STAGE=plots ./sbatch/pipeline.sbatch
#
# Fast enough to just run on the login node instead:
#   python -m pipeline.run --dataset ba --start-stage plots
#
#
# --- 12. CODING AGENT ONLY, ALL LLM KNOBS -----------------------------------
# START_STAGE=agent END_STAGE=agent reuses the data and the checkpoint and
# only re-runs the sweep. Every agent-side parameter:
#
#   DATASET=sbm START_STAGE=agent END_STAGE=agent FORCE=1 \
#   ARMS=evolve_scored@world_model \
#   LLM_MODEL=gpt-5.6-terra TEMPERATURE=0.7 OUTER_ITERS=20 \
#   BUDGET_PCTS="1 5 10 20" HORIZON=10 N_SAMPLES=50 MC_RUNS=200 \
#   ALLOWED_OPS="add_node remove_node" COMPARE=1 CREDIT=0 \
#       ./sbatch/pipeline.sbatch
#
#   ARMS is where method and strategy mode live: <method>_<mode>@<evaluator>
#     method  = one_shot | per_step | windowed | evolve
#     mode    = free   (writes a whole Strategy program, may call algorithms.*)
#             | scored (writes only score()/schedule() inside the fixed harness)
#     evaluator = native | monte_carlo | oracle | world_model
#   so evolve_scored@world_model is population search over scoring functions
#   judged by our learned model. Plus the two loop-free conditions:
#     routing              condition 2, one LLM call picks a library algorithm
#     baseline:<algorithm> condition 1, no LLM
#
#   LLM_MODEL       gateway model name (gpt-5.6-terra, gpt-5.6-sol, ...)
#   TEMPERATURE     sampling temperature; unset = provider default, 0.0 = greedy
#   OUTER_ITERS     refinement iterations per agent arm (conditions 1-2 force 1)
#   WINDOWS         window count, `windowed` method only
#   MC_RUNS         referee replay runs, and the @monte_carlo arm's own runs
#   NATIVE_MC_RUNS  real episodes per candidate for a @native arm
#   N_SAMPLES       rollout samples for @world_model and @oracle
#   HORIZON         agent rollout horizon
#   ALLOWED_OPS     ops the strategies may emit (add_node remove_node
#                   add_edge remove_edge set_edge_weight). UNSET = the task
#                   registry's own set, which is the right answer for every
#                   task: add_node remove_node for influence maximization,
#                   remove_node for critical node detection
#   OUTBREAK_PCT    critical node detection: size of the exogenous outbreak
#                   every arm must contain, as a % of N. Unset = registry (1.0)
#   OUTBREAK_SELECTOR  how its sources are picked: random (default) degree
#                   pagerank betweenness kshell. Deterministic in SEED, so
#                   every arm in a sweep faces the same outbreak. On influence
#                   blocking these two are |S_N| and the ATTACKER MODEL, which
#                   is that task's second experimental axis
#   EPI_LEVER       epidemic control: which of the four interventions the
#                   budget buys: vaccinate (immune, uncounted), quarantine
#                   (isolated but counted), edge_cut, contact_reduce. Sets the
#                   budgeted op and the permitted op together.
#   EPI_BETA        per-contact transmission as a multiple of the arc's own
#                   probability; EPI_GAMMA the rate of leaving I (recovery under
#                   SIR/SEIR, return-to-susceptible under SIS). Both land in
#                   metadata.json: §8.2 trap 2 records that a table which fixes
#                   them without stating them is comparable only to itself.
#   BLOCKING_LEVER  influence blocking: which of the four published
#                   interventions the budget buys: counter_seed (default),
#                   node_block, edge_block, weight_block
#   TIE_BREAK       influence blocking: which cascade wins a node both reach on
#                   the same step. auto (default) = each dynamics' own founding
#                   paper. Crosses generation, training AND evaluation, so set
#                   it once or not at all
#   POSITIVE_PROB   influence blocking: the blocker's per-edge probability.
#                   shared (default) = COICM; a float = MCICM, and 1.0 is
#                   Budak's high-effectiveness property
#   DETECTION_DELAY influence blocking: Budak's r, the steps the rumour runs
#                   before the blocker may act (default 0)
#   NEGATIVE_SELECTORS  influence blocking, data stage: how each episode's S_N
#                   is chosen (default: random degree pagerank)
#   BLOCKER_SELECTORS   ...and how its t=0 blocker set is; `none` leaves the
#                   rumour unopposed and is the prevented-influence reference
#   SL_SELECT_SPLIT source localization: episodes the outer loop's reward is
#                   computed on (default: train)
#   SL_EVAL_SPLIT   ...and the HELD-OUT episodes the winning program is re-run
#                   on unmodified, which is the number every table reports.
#                   Selecting and reporting on the same episodes proves nothing
#                   about amortization (default: test)
#   SL_INSTANCES    labelled episodes per split; every candidate program pays
#                   this many executions (default: 20)
#   SL_OBSERVATION  marginal (the MC-averaged P(infected), strictly MORE
#                   informative than the published protocol's) or binary (a
#                   single realized draw, the comparable column)
#   SL_BUDGET_MODE  episode = each instance's own source count, the published
#                   given-k convention; sweep = the pipeline's k, with the pool
#                   filtered to that source fraction
#   SL_PRIOR        arm A only: none = SL-VAE (a), vae = the full method
#   SL_STEPS        arm A only: Adam steps per instance
#   SL_TRANSFER_FROM  run ANOTHER run's winning program here, unmodified: the
#                   graph axis of the amortization claim
#
# CASCADE RECONSTRUCTION (--task cascade_reconstruction). Every one is inert for
# every other task, exactly like the SL block above.
#   CR_SETTING      which of the four observation regimes is masked:
#                   partial_times (a subsample of the infected set WITH
#                   activation times: the ordered-Steiner regime),
#                   partial_nodes (the same subsample, times withheld),
#                   final_snapshot (the terminal state only: DITTO's DASH, the
#                   hardest published formulation), or hidden_nodes (nodes
#                   DELETED from the graph, not merely unobserved). These are
#                   four separate protocols and their rows are never pooled
#   CR_OBSERVATION_RATE  probability an infected node IS REPORTED. Stated in that
#                   direction because this literature uses the symbol sigma for
#                   both the report rate and its complement
#   CR_HIDDEN_RATE  fraction of non-source nodes deleted under hidden_nodes
#   CR_INSTANCES    masked cascades per split; every candidate decoder pays this
#                   many executions
#   CR_SELECT_SPLIT / CR_EVAL_SPLIT  the two disjoint pools, same reason as SL's
#   CP_SPLIT        cascade prediction: `chronological` (leak-free, the default)
#                   or `random` (the field's own, which LEAKS). The gap between
#                   the two IS the experiment (research/cascade_prediction.md 8.3)
#   CP_OBSERVATION  t_o in the CORPUS's own time unit; unset takes the corpus's
#                   own first published window. 8.2 pairs TWO per corpus and a
#                   single-window result is not publishable in this literature
#   CP_HORIZON      t_p, same units; unset takes the corpus's own
#   CP_METRIC       which error the reward is; all of them MINIMIZE
#   CR_TREE_WEIGHT  lambda in lambda * PathPrecision + (1 - lambda) * EventF1.
#                   At least 0.5 is a REQUIREMENT: recovering the node set is
#                   nearly free, so a search rewarded mostly on Event F1
#                   converges on decoders that never attempt the tree
#   CR_MCMC_PROPOSALS / CR_MCMC_BURN_IN  arm A only (decode_*)
#   EVALUATOR       fallback for arms written without an @evaluator suffix
#   BUDGET_PCTS     seed budgets as % of N, one run per value. UNSET = the task
#                   registry's own sweep, which is a single 10% point for source
#                   localization because k there is a property of the instance
#   BUDGETS         absolute k, overrides BUDGET_PCTS entirely
#   COMPARE         1 = replay every winner on the shared ground-truth referee.
#                   Keep this at 1. Without it the table is not comparable
#                   across conditions, because each arm's own reward comes
#                   from its own evaluator.
#   CREDIT          1 = per-action counterfactual credit in the feedback.
#                   Costs one extra rollout per action.
#   ALLOW_MC_ALGOS  1 = let generated scripts call the per-candidate-simulation
#                   algorithms (celf, vanilla_greedy, ...). Default 0: they
#                   exceed 60s per call and their episodes never reach
#                   real_env_episodes. Baselines and routing are unaffected.
#   STRATEGY_TIMEOUT wall-clock cap (s) on one generated plan_horizon()/act()
#                   call; an overrun becomes a repair turn instead of hanging
#                   the sweep. Default 300, 0 disables.
#   LLM_PRICE_IN / LLM_PRICE_OUT
#                   USD per 1M prompt / completion tokens, for the cost column.
#                   Tokens are always counted; cost stays empty unless both are
#                   set (the gateway bills nothing per token).
#
#   A killed agent arm leaves <arm>.ckpt.json beside its result and the next run
#   of the same job resumes that search from where it stopped. FORCE=1 deletes
#   the checkpoint first, so a forced re-run is genuinely fresh.
#   GRAPH_ID        evaluate on a specific graph in the store
#   BASELINE_TIMEOUT seconds before an external repo subprocess is killed
#
# Comparing two LLMs on the same data, as two jobs into two runs:
#
#   DATASET=sbm RUN=sonnet LLM_MODEL=gpt-5.6-terra ./sbatch/pipeline.sbatch
#   DATASET=sbm RUN=gpt    LLM_MODEL=gpt-5.6-sol     ./sbatch/pipeline.sbatch
#
#
# --- 13. ALL WORLD-MODEL TRAINING KNOBS -------------------------------------
#
#   DATASET=ba RUN=gt START_STAGE=train END_STAGE=train FORCE=1 \
#   WM_MODEL=gt HEAD=structured_residual HIDDEN_DIM=128 N_LAYERS=3 \
#   N_HEADS=4 FFN_DIM=256 DROPOUT=0.1 \
#   EPOCHS=400 LR=1e-3 WEIGHT_DECAY=5e-4 BATCH_SIZE=32 \
#   POS_WEIGHT=off PATIENCE=50 PLAN_GRAPHS=5 NO_PLAN_DEMO=0 \
#       ./sbatch/pipeline.sbatch
#
#   WM_MODEL     gcn | sage | gt | gat | gcnii
#   HEAD         linear | structured | structured_residual
#                structured/structured_residual REQUIRE POS_WEIGHT=off,
#                pos_weight inflates the per-edge q globally and collapses
#                one-step accuracy.
#   HIDDEN_DIM   64 for <1K nodes, 128 for 1K-5K, 256 for >10K
#   N_LAYERS     3 for gcn/sage/gt/gat; 8-16 for gcnii, which is built for depth
#   N_HEADS      gat and gt only
#   FFN_DIM      gt only
#   GCNII_ALPHA  gcnii initial-residual strength, gcnii only
#   GCNII_LAMDA  gcnii identity-mapping decay, gcnii only
#   POS_WEIGHT   auto | off
#   NO_PLAN_DEMO 1 skips the multi-graph planning demo after training
#
# Backbone sweep, five jobs into five runs sharing one dataset: generate the
# data once, then each job reuses it:
#
#   DATASET=ba START_STAGE=data END_STAGE=data ./sbatch/pipeline.sbatch
#   for M in gcn sage gat gt gcnii; do
#     DATASET=ba WM_MODEL=$M START_STAGE=train JOB_NAME=gwm_$M \
#         ./sbatch/pipeline.sbatch
#   done
#   (all five share results/influence_maximization/ba/default/data; checkpoints
#    are named per backbone, so they do not collide)
#
#
# --- 14. ALL DATA-GENERATION KNOBS ------------------------------------------
#
#   DATASET=sbm RUN=dense START_STAGE=data END_STAGE=data \
#   NUM_GRAPHS=40 SYN_NODES=200 SBM_BLOCKS=4 SBM_P_IN=0.2 SBM_P_OUT=0.005 \
#   GEN_MODELS="IC LT" GEN_ALGORITHMS="random degree pagerank betweenness celf" \
#   GEN_ACTION_OPS="add_node remove_node" PROB_MODEL=weighted \
#   BUDGET_PCT_RANGE="1 20" ROLLOUTS=20 GEN_HORIZON=10 \
#   INJECT_P=0.4 CF_PROB=0.4 CF_BRANCHES=2 MC_MARGINALS=30 SPLIT="0.7 0.15 0.15" \
#       ./sbatch/pipeline.sbatch
#
#   NUM_GRAPHS       synthetic only; a real dataset is one graph, and passing
#                    anything but 1 for it is a hard error
#   SYN_NODES        nodes per synthetic graph
#   ER_P BA_M WS_K WS_P SBM_BLOCKS SBM_P_IN SBM_P_OUT   family-specific
#   GEN_MODELS       IC, LT, or both: one transition file per dynamics
#   GEN_ALGORITHMS   spine seed selectors: random degree pagerank betweenness
#                    celf local_search. betweenness is exact Brandes, O(V*E)
#                    in pure networkx, so it is dropped by default on the big
#                    graphs (hours at 15K+ nodes).
#   GEN_ACTION_OPS   ops injected during generation. UNSET = the task
#                    registry's own set (add_node remove_node for influence
#                    maximization, remove_node for critical node detection);
#                    set it explicitly for diffusion-only data or to add the
#                    three edge ops for the full action space.
#   PROB_MODEL       weighted (p = 1/in_degree) | uniform (UNIFORM_P)
#   BUDGET_PCT_RANGE per-episode k drawn from this band, which is what lets ONE
#                    checkpoint serve the whole budget sweep. It must cover
#                    every BUDGET_PCTS value you intend to evaluate at.
#   ROLLOUTS         episodes per (graph, dynamics, selector)
#   INJECT_P         probability an intermediate step injects an action
#   SPLIT_MODE       graph_disjoint | episode_random; unset resolves per dataset
#                    (disjoint for synthetic families with >= 3 graphs)
#   CF_PROB          probability a step spawns counterfactual forks: the
#                    signal that forces action-conditioning
#   CF_BRANCHES      counterfactual branches per fork
#   MC_MARGINALS     draws per step for the soft targets. Must be >= 1 or
#                    training fails; 30 is the calibrated default.
#   SPLIT            train/val/test fractions
#
#
# --- 15. LT INSTEAD OF IC ---------------------------------------------------
# DIFFUSION_MODEL picks the dynamics used downstream of generation, selecting
# both the transition file and the structured head. Generate both dynamics up
# front and either can be run later without regenerating.
#
#   DATASET=ba RUN=lt DIFFUSION_MODEL=LT ./sbatch/pipeline.sbatch
#
#
# --- 16. ABSOLUTE BUDGETS ---------------------------------------------------
# BUDGETS overrides BUDGET_PCTS, for matching a published table's k exactly.
#
#   DATASET=nethept BUDGETS="10 20 30 40 50" ./sbatch/pipeline.sbatch
#
#
# --- 17. RESUME AFTER A CRASH -----------------------------------------------
# Results live in results/<task>/<dataset>/<run>_<jobid>/, so resuming means
# naming the tree the dead job created: a bare resubmit would start a FRESH
# one. The dead job's .out prints the line to use, verbatim:
#
#   [sbatch] resume this run with: RUN=default_918273 RUN_JOBID=0 .../pipeline.sbatch
#
#   DATASET=ba RUN=918273 RUN_JOBID=0 ./sbatch/pipeline.sbatch
#
# Every finished stage, every finished (budget, arm) result, and every skipped
# external baseline is on disk and gets reused; only the missing work runs.
# Check what is done first:
#
#   R=results/influence_maximization/ba/default
#   cat $R/pipeline.json                       # stage status + timings
#   column -s, -t $R/summary.csv | less -S     # every finished arm
#
# For a long-lived experiment you keep adding to, drop the job id entirely and
# always resume into the same tree:
#
#   DATASET=ba RUN_JOBID=0 ./sbatch/pipeline.sbatch
#
# To redo one stage from scratch instead of resuming it:
#
#   DATASET=ba RUN=918273 RUN_JOBID=0 \
#   START_STAGE=agent END_STAGE=agent FORCE=1 ./sbatch/pipeline.sbatch
#
#
# --- 18. DRY RUN ------------------------------------------------------------
# Print the resolved sbatch line and the full pipeline command, submit
# nothing. Always do this before a long run.
#
#   DATASET=ba DRY_RUN=1 ./sbatch/pipeline.sbatch
#
# To also validate the flags against the real parser and see the arm plan
# without executing a stage, on the login node:
#
#   python -m pipeline.run --dataset ba \
#       --skip-stages data train agent plots report
#
#
# --- 19. SMOKE TEST ---------------------------------------------------------
# Prove the whole path end to end in a couple of minutes on a tiny graph
# before burning an allocation. Login node, no sbatch.
#
#   python -m pipeline.run --dataset ba --run smoke \
#       --num-graphs 2 --syn-nodes 60 --rollouts 2 --mc-marginals 3 \
#       --epochs 3 --gen-models IC \
#       --baselines high_degree --arms evolve_free@oracle \
#       --budget-pcts 10 --outer-iters 1 --mc-runs 20 --compare
#
# ============================================================================

set -euo pipefail

# Resolve before the cd, so `./sbatch/pipeline.sbatch` still resubmits the
# right file when it is launched from somewhere other than the repo root
script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# Repo root, derived from this script's own location so the file works in
# any checkout rather than one person's home directory. Set REPO_ROOT to
# override, e.g. when submitting a copy staged outside the tree.
cd "${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Activate the project virtualenv, unless one is already active or VENV
# points somewhere else. Skipped silently when neither exists, so the
# script still runs under a module-loaded or conda interpreter.
VENV="${VENV:-venv}"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
fi
mkdir -p results logs

# ---------------------------------------------------------------------------
# DEFAULTS
# ---------------------------------------------------------------------------
DATASET="${DATASET:?DATASET is required, e.g. DATASET=ba ./sbatch/pipeline.sbatch}"
TASK="${TASK:-influence_maximization}"
RUN="${RUN:-default}"
# Append _<jobid> to the results directory, the same way logs get _%j
RUN_JOBID="${RUN_JOBID:-1}"

# Adaptive / online IM knobs; empty means "not this branch"
ROUNDS="${ROUNDS:-}"
PER_ROUND_BUDGET="${PER_ROUND_BUDGET:-}"
ROUND_GAP="${ROUND_GAP:-}"
FEEDBACK_MODEL="${FEEDBACK_MODEL:-}"
EDIT_RATE="${EDIT_RATE:-}"
CAMPAIGNS="${CAMPAIGNS:-}"
HIDE_EDGE_WEIGHTS="${HIDE_EDGE_WEIGHTS:-0}"

# Critical node detection: the exogenous outbreak every arm contains.
# Empty means the task registry's own value.
OUTBREAK_PCT="${OUTBREAK_PCT:-}"
OUTBREAK_SELECTOR="${OUTBREAK_SELECTOR:-}"

# Influence blocking. Empty means the pipeline default; the two selector lists
# only reach the data stage, the other four cross all three stages.
BLOCKING_LEVER="${BLOCKING_LEVER:-}"
TIE_BREAK="${TIE_BREAK:-}"
POSITIVE_PROB="${POSITIVE_PROB:-}"
DETECTION_DELAY="${DETECTION_DELAY:-}"
NEGATIVE_SELECTORS="${NEGATIVE_SELECTORS:-}"
BLOCKER_SELECTORS="${BLOCKER_SELECTORS:-}"

# Source localization; empty means the pipeline default. The two SPLITS are the
# load-bearing pair: the outer loop's reward is computed on SL_SELECT_SPLIT and
# the winning program is re-run unmodified on SL_EVAL_SPLIT, which is the number
# every table reports (research/source_localization.md §8.5.1).
SL_SELECT_SPLIT="${SL_SELECT_SPLIT:-}"
SL_EVAL_SPLIT="${SL_EVAL_SPLIT:-}"
SL_INSTANCES="${SL_INSTANCES:-}"
SL_OBSERVATION="${SL_OBSERVATION:-}"
SL_BUDGET_MODE="${SL_BUDGET_MODE:-}"
SL_SOURCE_TOLERANCE="${SL_SOURCE_TOLERANCE:-}"
SL_PRIOR="${SL_PRIOR:-}"
SL_STEPS="${SL_STEPS:-}"
SL_LR="${SL_LR:-}"
SL_CARDINALITY_WEIGHT="${SL_CARDINALITY_WEIGHT:-}"
SL_PRIOR_WEIGHT="${SL_PRIOR_WEIGHT:-}"
SL_PRIOR_EPOCHS="${SL_PRIOR_EPOCHS:-}"
SL_TRANSFER_FROM="${SL_TRANSFER_FROM:-}"

# Cascade reconstruction. `CR_SETTING` is the load-bearing one: the four settings
# are four PROTOCOLS rather than four knobs, and a decoder selected under
# final_snapshot is solving a different problem from one selected under
# partial_times (research/cascade_reconstruction.md §8.3).
CR_SETTING="${CR_SETTING:-}"
CR_OBSERVATION_RATE="${CR_OBSERVATION_RATE:-}"
CR_HIDDEN_RATE="${CR_HIDDEN_RATE:-}"
CR_INSTANCES="${CR_INSTANCES:-}"
CR_SELECT_SPLIT="${CR_SELECT_SPLIT:-}"
CR_EVAL_SPLIT="${CR_EVAL_SPLIT:-}"
CR_TREE_WEIGHT="${CR_TREE_WEIGHT:-}"

# Cascade prediction. `CP_SPLIT` is the load-bearing one: research/
# cascade_prediction.md §8.3 shows the field's own random-over-cascades split LEAKS
# the future, and that fixing it moved a decade of published numbers: two 2021-24
# SOTA models fell below a plain MLP. `chronological` is the default and the leaky
# protocol is available on purpose, because the GAP between them is the experiment.
# CP_OBSERVATION / CP_HORIZON are in the CORPUS's own time unit (seconds for the
# social corpora, DAYS for APS) and default to that corpus's own published windows.
CP_OBSERVATION="${CP_OBSERVATION:-}"
CP_HORIZON="${CP_HORIZON:-}"
CP_STEP="${CP_STEP:-}"
CP_MIN_SIZE="${CP_MIN_SIZE:-}"
CP_TRUNCATE="${CP_TRUNCATE:-}"
CP_SPLIT="${CP_SPLIT:-}"
CP_TARGET="${CP_TARGET:-}"
CP_GRAPH="${CP_GRAPH:-}"
CP_MAX_CASCADES="${CP_MAX_CASCADES:-}"
CP_MAX_NODES="${CP_MAX_NODES:-}"
CP_SELECT_SPLIT="${CP_SELECT_SPLIT:-}"
CP_EVAL_SPLIT="${CP_EVAL_SPLIT:-}"
CP_INSTANCES="${CP_INSTANCES:-}"
CP_OBSERVATION_STEPS="${CP_OBSERVATION_STEPS:-}"
CP_METRIC="${CP_METRIC:-}"
CP_FORECAST_SAMPLES="${CP_FORECAST_SAMPLES:-}"

# Epidemic control. `EPI_LEVER` is the load-bearing one: the four interventions of
# research/epidemic_control.md §2.5 are four different problems, and vaccinate and
# quarantine differ in the DOSE ACCOUNTING rather than in what is chosen (§8.2 trap
# 7's "recovered is not removed"). EPI_BETA / EPI_GAMMA cross all three stages and
# land in metadata.json, because §8.2 trap 2 records that a table which fixes them
# without stating them is comparable only to itself. Note GEN_MODELS and
# DIFFUSION_MODEL must both be one of SIR / SIS / SEIR here: the pipeline defaults
# them from the task registry, so leaving both unset is the SIR setting.
EPI_LEVER="${EPI_LEVER:-}"
CONTACT_REDUCTION="${CONTACT_REDUCTION:-}"
EPI_BETA="${EPI_BETA:-}"
EPI_GAMMA="${EPI_GAMMA:-}"
EPI_ALPHA="${EPI_ALPHA:-}"
EPI_BURN_IN="${EPI_BURN_IN:-}"
OUTBREAK_SELECTORS="${OUTBREAK_SELECTORS:-}"
IMMUNIZER_SELECTORS="${IMMUNIZER_SELECTORS:-}"
CR_MCMC_PROPOSALS="${CR_MCMC_PROPOSALS:-}"
CR_MCMC_BURN_IN="${CR_MCMC_BURN_IN:-}"

# Dataset class drives the settings that genuinely cannot share a default:
# graph population, seed selectors, encoder width, and memory.
case "$DATASET" in
  er|ba|ws|sbm|powerlaw_cluster|kronecker|karate)
    class_graphs=40 ; class_nodes=100 ; class_rollouts=20
    class_selectors="random degree pagerank betweenness"
    class_hidden=64 ; class_mem=16G ; class_time=24:00:00
    ;;
  nethept|twitter|digg|youtube|weibo)
    # Exact Brandes betweenness is O(V*E) in pure networkx: hours at this
    # size, so these lean on random/degree/pagerank instead
    class_graphs=1 ; class_nodes=100 ; class_rollouts=50
    class_selectors="random degree pagerank"
    class_hidden=256 ; class_mem=64G ; class_time=48:00:00
    ;;
  *)
    # One real graph: episode diversity comes from rollouts, not graph count
    class_graphs=1 ; class_nodes=100 ; class_rollouts=100
    class_selectors="random degree pagerank betweenness"
    class_hidden=128 ; class_mem=32G ; class_time=24:00:00
    ;;
esac

# Driver
START_STAGE="${START_STAGE:-data}"
END_STAGE="${END_STAGE:-report}"
SKIP_STAGES="${SKIP_STAGES:-none}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
SEED="${SEED:-42}"
FORCE="${FORCE:-0}"

# Data stage
NUM_GRAPHS="${NUM_GRAPHS:-$class_graphs}"
SYN_NODES="${SYN_NODES:-$class_nodes}"
ER_P="${ER_P:-0.05}"
BA_M="${BA_M:-3}"
WS_K="${WS_K:-6}"
WS_P="${WS_P:-0.1}"
SBM_BLOCKS="${SBM_BLOCKS:-4}"
SBM_P_IN="${SBM_P_IN:-0.15}"
SBM_P_OUT="${SBM_P_OUT:-0.01}"
# Left EMPTY for a compartmental task so pipeline.run.resolve_dynamics defaults
# both this and DIFFUSION_MODEL from the registry; IC/LT and SIR/SIS/SEIR produce
# different state layouts and cannot share a dataset
if [ "$TASK" = "epidemic_control" ]; then
  GEN_MODELS="${GEN_MODELS:-SIR}"
else
  GEN_MODELS="${GEN_MODELS:-IC LT}"
fi
GEN_ALGORITHMS="${GEN_ALGORITHMS:-$class_selectors}"
# Unset = the TASK REGISTRY decides, which is what makes --task
# critical_node_detection generate remove_node transitions and let its
# planner emit remove_node without any flag here. Set it to override.
GEN_ACTION_OPS="${GEN_ACTION_OPS:-}"
PROB_MODEL="${PROB_MODEL:-weighted}"
UNIFORM_P="${UNIFORM_P:-0.1}"
BUDGET_PCT_RANGE="${BUDGET_PCT_RANGE:-1 20}"
ROLLOUTS="${ROLLOUTS:-$class_rollouts}"
GEN_HORIZON="${GEN_HORIZON:-10}"
INJECT_P="${INJECT_P:-0.4}"
CF_PROB="${CF_PROB:-0.4}"
CF_BRANCHES="${CF_BRANCHES:-2}"
MC_MARGINALS="${MC_MARGINALS:-30}"
SPLIT="${SPLIT:-0.7 0.15 0.15}"
# Unset lets pipeline.run resolve it per dataset: graph_disjoint for a
# synthetic family with >= 3 graphs, episode_random for a single real graph
# (which cannot be split disjointly at all; its test metrics are then
# in-graph, not held-out-graph). Set explicitly to force either.
SPLIT_MODE="${SPLIT_MODE:-}"

# Train stage
WM_MODEL="${WM_MODEL:-sage}"
HEAD="${HEAD:-structured_residual}"
HIDDEN_DIM="${HIDDEN_DIM:-$class_hidden}"
N_LAYERS="${N_LAYERS:-3}"
N_HEADS="${N_HEADS:-4}"
FFN_DIM="${FFN_DIM:-128}"
GCNII_ALPHA="${GCNII_ALPHA:-0.1}"
GCNII_LAMDA="${GCNII_LAMDA:-0.5}"
DROPOUT="${DROPOUT:-0.1}"
EPOCHS="${EPOCHS:-400}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
POS_WEIGHT="${POS_WEIGHT:-off}"
PATIENCE="${PATIENCE:-50}"
PLAN_GRAPHS="${PLAN_GRAPHS:-5}"
NO_PLAN_DEMO="${NO_PLAN_DEMO:-0}"
# k-seed full-horizon planning (0 = off), off-policy rollout tests, and the
# action feature encoding. See world_model/README.md for what each one buys.
PLAN_BUDGET_K="${PLAN_BUDGET_K:-0}"
PLAN_BUDGET_GRAPHS="${PLAN_BUDGET_GRAPHS:-3}"
PLAN_BUDGET_HORIZON="${PLAN_BUDGET_HORIZON:-20}"
OOD_POLICIES="${OOD_POLICIES:-}"
ACTION_ENCODING="${ACTION_ENCODING:-basic}"

# Agent stage
# Unset means "let pipeline.tasks decide": influence_maximization gets the
# static IM classics, adaptive_online_im gets AdaptGreedy/EPIC and the per-round
# heuristics. Hardcoding a default here would shadow that and silently run
# static baselines on an adaptive task. "none" still means no baselines at all.
BASELINES="${BASELINES:-}"
BASELINE_TIMEOUT="${BASELINE_TIMEOUT:-3600}"
# Mirrors pipeline.conditions.default_arms. evolve, not one_shot: both refine a
# program against the same feedback for the same LLM calls, but evolve edits the
# population best each generation while one_shot edits the latest attempt.
# Same: unset lets the task registry choose, which for adaptive_online_im pairs
# every adaptive arm with its matched non-adaptive control
ARMS="${ARMS:-}"
NATIVE_MC_RUNS="${NATIVE_MC_RUNS:-1}"
# Empty means the task registry's own sweep. Source localization declares a
# SINGLE 10%-of-N point because k there is a property of the instance rather than
# a choice, and hardcoding the four-point ladder here would silently turn that
# into four runs of one experiment.
BUDGET_PCTS="${BUDGET_PCTS:-}"
BUDGETS="${BUDGETS:-}"
EVALUATOR="${EVALUATOR:-oracle}"
LLM_MODEL="${LLM_MODEL:-gpt-5.6-terra}"
TEMPERATURE="${TEMPERATURE:-}"
if [ "$TASK" = "epidemic_control" ]; then
  DIFFUSION_MODEL="${DIFFUSION_MODEL:-SIR}"
else
  DIFFUSION_MODEL="${DIFFUSION_MODEL:-IC}"
fi
HORIZON="${HORIZON:-10}"
OUTER_ITERS="${OUTER_ITERS:-20}"
WINDOWS="${WINDOWS:-3}"
MC_RUNS="${MC_RUNS:-200}"
N_SAMPLES="${N_SAMPLES:-50}"
# Unset = the task registry decides; see GEN_ACTION_OPS above
ALLOWED_OPS="${ALLOWED_OPS:-}"
COMPARE="${COMPARE:-1}"
CREDIT="${CREDIT:-0}"
ALLOW_MC_ALGOS="${ALLOW_MC_ALGOS:-0}"
STRATEGY_TIMEOUT="${STRATEGY_TIMEOUT:-}"
LLM_PRICE_IN="${LLM_PRICE_IN:-}"
LLM_PRICE_OUT="${LLM_PRICE_OUT:-}"
GRAPH_ID="${GRAPH_ID:-}"

# ---------------------------------------------------------------------------
# ASSEMBLE
# ---------------------------------------------------------------------------
# Unquoted expansion is deliberate for the list-valued vars: each word becomes
# its own argv entry, which is what nargs="+" wants.
flags=(
  # --task belongs here rather than beside --run below: the pre-flight
  # --print-resources probe runs BEFORE the submit, and without it that probe
  # answers for the DEFAULT task, which is a wrong gpu/stage decision on every
  # other task and an outright crash when the baselines are task-registered
  --task "$TASK"
  --dataset "$DATASET"
  --results-root "$RESULTS_ROOT" --seed "$SEED"
  --start-stage "$START_STAGE" --end-stage "$END_STAGE"

  --num-graphs "$NUM_GRAPHS" --syn-nodes "$SYN_NODES"
  --er-p "$ER_P" --ba-m "$BA_M" --ws-k "$WS_K" --ws-p "$WS_P"
  --sbm-blocks "$SBM_BLOCKS" --sbm-p-in "$SBM_P_IN" --sbm-p-out "$SBM_P_OUT"
  --gen-models $GEN_MODELS
  --gen-algorithms $GEN_ALGORITHMS
  --prob-model "$PROB_MODEL" --uniform-p "$UNIFORM_P"
  --budget-pct-range $BUDGET_PCT_RANGE
  --rollouts "$ROLLOUTS" --gen-horizon "$GEN_HORIZON"
  --inject-p "$INJECT_P" --cf-prob "$CF_PROB" --cf-branches "$CF_BRANCHES"
  --mc-marginals "$MC_MARGINALS" --split $SPLIT

  --wm-model "$WM_MODEL" --head "$HEAD"
  --hidden-dim "$HIDDEN_DIM" --n-layers "$N_LAYERS"
  --n-heads "$N_HEADS" --ffn-dim "$FFN_DIM"
  --gcnii-alpha "$GCNII_ALPHA" --gcnii-lamda "$GCNII_LAMDA"
  --dropout "$DROPOUT" --epochs "$EPOCHS" --lr "$LR"
  --weight-decay "$WEIGHT_DECAY" --batch-size "$BATCH_SIZE"
  --pos-weight "$POS_WEIGHT" --patience "$PATIENCE" --plan-graphs "$PLAN_GRAPHS"
  --plan-budget-k "$PLAN_BUDGET_K" --plan-budget-graphs "$PLAN_BUDGET_GRAPHS"
  --plan-budget-horizon "$PLAN_BUDGET_HORIZON"
  --action-encoding "$ACTION_ENCODING"

  --baseline-timeout "$BASELINE_TIMEOUT" --native-mc-runs "$NATIVE_MC_RUNS"
  --evaluator "$EVALUATOR" --llm-model "$LLM_MODEL"
  --diffusion-model "$DIFFUSION_MODEL" --horizon "$HORIZON"
  --outer-iters "$OUTER_ITERS" --windows "$WINDOWS"
  --mc-runs "$MC_RUNS" --n-samples "$N_SAMPLES"
)

# Op lists are omitted when unset so the task registry supplies them; passing
# the influence-maximization pair unconditionally would make every task look
# like influence maximization to the generator and the planner
[ -n "$GEN_ACTION_OPS" ] && flags+=(--gen-action-ops $GEN_ACTION_OPS)
[ -n "$ALLOWED_OPS" ]    && flags+=(--allowed-ops $ALLOWED_OPS)

# `none` means the empty list: argparse's nargs="*" takes zero values when the
# next token is another flag, which is exactly how you ask for no baselines
# Three states each: unset (omit the flag, registry decides), "none" (bare flag,
# empty list), or an explicit list
if [ "$BASELINES" = "none" ]; then flags+=(--baselines)
elif [ -n "$BASELINES" ]; then flags+=(--baselines $BASELINES); fi
if [ "$ARMS" = "none" ]; then flags+=(--arms)
elif [ -n "$ARMS" ]; then flags+=(--arms $ARMS); fi
[ "$SKIP_STAGES" = "none" ] || flags+=(--skip-stages $SKIP_STAGES)

# BUDGETS is absolute k and wins outright; otherwise budgets are percentages, and
# an empty BUDGET_PCTS omits the flag so the task registry supplies the sweep
if [ -n "$BUDGETS" ]; then
  flags+=(--budgets $BUDGETS)
elif [ -n "$BUDGET_PCTS" ]; then
  flags+=(--budget-pcts $BUDGET_PCTS)
fi

[ -n "$TEMPERATURE" ]     && flags+=(--temperature "$TEMPERATURE")
[ -n "$GRAPH_ID" ]        && flags+=(--graph-id "$GRAPH_ID")
[ "$COMPARE" = "1" ]      && flags+=(--compare)
[ "$CREDIT" = "1" ]       && flags+=(--credit)
[ "$ALLOW_MC_ALGOS" = "1" ] && flags+=(--allow-mc-algorithms)
[ -n "$STRATEGY_TIMEOUT" ] && flags+=(--strategy-timeout "$STRATEGY_TIMEOUT")
[ -n "$LLM_PRICE_IN" ]  && flags+=(--llm-price-in "$LLM_PRICE_IN")
[ -n "$LLM_PRICE_OUT" ] && flags+=(--llm-price-out "$LLM_PRICE_OUT")
[ "$FORCE" = "1" ]        && flags+=(--force)
[ "$NO_PLAN_DEMO" = "1" ] && flags+=(--no-plan-demo)
# Adaptive IM (research/adaptive_online_im.md). Read only by `adaptive` arms and
# by the branches below, so leaving them unset is exactly the static IM setting.
[ -n "$ROUNDS" ]            && flags+=(--rounds "$ROUNDS")
[ -n "$PER_ROUND_BUDGET" ]  && flags+=(--per-round-budget "$PER_ROUND_BUDGET")
[ -n "$ROUND_GAP" ]         && flags+=(--round-gap "$ROUND_GAP")
[ -n "$FEEDBACK_MODEL" ]    && flags+=(--feedback-model "$FEEDBACK_MODEL")
[ -n "$EDIT_RATE" ]         && flags+=(--edit-rate "$EDIT_RATE")
[ -n "$CAMPAIGNS" ]         && flags+=(--campaigns "$CAMPAIGNS")
[ -n "$OUTBREAK_PCT" ]      && flags+=(--outbreak-pct "$OUTBREAK_PCT")
[ -n "$OUTBREAK_SELECTOR" ] && flags+=(--outbreak-selector "$OUTBREAK_SELECTOR")
# Epidemic control (research/epidemic_control.md). Inert unless the task registry
# marks the task epidemic, so leaving them unset is exactly every other task.
[ -n "$EPI_LEVER" ]           && flags+=(--epi-lever "$EPI_LEVER")
[ -n "$CONTACT_REDUCTION" ]   && flags+=(--contact-reduction "$CONTACT_REDUCTION")
[ -n "$EPI_BETA" ]            && flags+=(--epi-beta "$EPI_BETA")
[ -n "$EPI_GAMMA" ]           && flags+=(--epi-gamma "$EPI_GAMMA")
[ -n "$EPI_ALPHA" ]           && flags+=(--epi-alpha "$EPI_ALPHA")
[ -n "$EPI_BURN_IN" ]         && flags+=(--epi-burn-in "$EPI_BURN_IN")
[ -n "$OUTBREAK_SELECTORS" ]  && flags+=(--outbreak-selectors $OUTBREAK_SELECTORS)
[ -n "$IMMUNIZER_SELECTORS" ] && flags+=(--immunizer-selectors $IMMUNIZER_SELECTORS)

[ -n "$BLOCKING_LEVER" ]     && flags+=(--blocking-lever "$BLOCKING_LEVER")
[ -n "$TIE_BREAK" ]          && flags+=(--tie-break "$TIE_BREAK")
[ -n "$POSITIVE_PROB" ]      && flags+=(--positive-prob "$POSITIVE_PROB")
[ -n "$DETECTION_DELAY" ]    && flags+=(--detection-delay "$DETECTION_DELAY")
[ -n "$NEGATIVE_SELECTORS" ] && flags+=(--negative-selectors $NEGATIVE_SELECTORS)
[ -n "$BLOCKER_SELECTORS" ]  && flags+=(--blocker-selectors $BLOCKER_SELECTORS)

[ -n "$SL_SELECT_SPLIT" ]      && flags+=(--sl-select-split "$SL_SELECT_SPLIT")
[ -n "$SL_EVAL_SPLIT" ]        && flags+=(--sl-eval-split "$SL_EVAL_SPLIT")
[ -n "$SL_INSTANCES" ]         && flags+=(--sl-instances "$SL_INSTANCES")
[ -n "$SL_OBSERVATION" ]       && flags+=(--sl-observation "$SL_OBSERVATION")
[ -n "$SL_BUDGET_MODE" ]       && flags+=(--sl-budget-mode "$SL_BUDGET_MODE")
[ -n "$SL_SOURCE_TOLERANCE" ]  && flags+=(--sl-source-tolerance "$SL_SOURCE_TOLERANCE")
[ -n "$SL_PRIOR" ]             && flags+=(--sl-prior "$SL_PRIOR")
[ -n "$SL_STEPS" ]             && flags+=(--sl-steps "$SL_STEPS")
[ -n "$SL_LR" ]                && flags+=(--sl-lr "$SL_LR")
[ -n "$SL_CARDINALITY_WEIGHT" ] && flags+=(--sl-cardinality-weight "$SL_CARDINALITY_WEIGHT")
[ -n "$SL_PRIOR_WEIGHT" ]      && flags+=(--sl-prior-weight "$SL_PRIOR_WEIGHT")
[ -n "$SL_PRIOR_EPOCHS" ]      && flags+=(--sl-prior-epochs "$SL_PRIOR_EPOCHS")
[ -n "$SL_TRANSFER_FROM" ]     && flags+=(--sl-transfer-from "$SL_TRANSFER_FROM")
[ -n "$CR_SETTING" ]           && flags+=(--cr-setting "$CR_SETTING")
[ -n "$CR_OBSERVATION_RATE" ]  && flags+=(--cr-observation-rate "$CR_OBSERVATION_RATE")
[ -n "$CR_HIDDEN_RATE" ]       && flags+=(--cr-hidden-rate "$CR_HIDDEN_RATE")
[ -n "$CR_INSTANCES" ]         && flags+=(--cr-instances "$CR_INSTANCES")
[ -n "$CR_SELECT_SPLIT" ]      && flags+=(--cr-select-split "$CR_SELECT_SPLIT")
[ -n "$CR_EVAL_SPLIT" ]        && flags+=(--cr-eval-split "$CR_EVAL_SPLIT")
[ -n "$CR_TREE_WEIGHT" ]       && flags+=(--cr-tree-weight "$CR_TREE_WEIGHT")
[ -n "$CR_MCMC_PROPOSALS" ]    && flags+=(--cr-mcmc-proposals "$CR_MCMC_PROPOSALS")
[ -n "$CR_MCMC_BURN_IN" ]      && flags+=(--cr-mcmc-burn-in "$CR_MCMC_BURN_IN")
[ -n "$CP_OBSERVATION" ]       && flags+=(--cp-observation "$CP_OBSERVATION")
[ -n "$CP_HORIZON" ]           && flags+=(--cp-horizon "$CP_HORIZON")
[ -n "$CP_STEP" ]              && flags+=(--cp-step "$CP_STEP")
[ -n "$CP_MIN_SIZE" ]          && flags+=(--cp-min-size "$CP_MIN_SIZE")
[ -n "$CP_TRUNCATE" ]          && flags+=(--cp-truncate "$CP_TRUNCATE")
[ -n "$CP_SPLIT" ]             && flags+=(--cp-split "$CP_SPLIT")
[ -n "$CP_TARGET" ]            && flags+=(--cp-target "$CP_TARGET")
[ -n "$CP_GRAPH" ]             && flags+=(--cp-graph "$CP_GRAPH")
[ -n "$CP_MAX_CASCADES" ]      && flags+=(--cp-max-cascades "$CP_MAX_CASCADES")
[ -n "$CP_MAX_NODES" ]         && flags+=(--cp-max-nodes "$CP_MAX_NODES")
[ -n "$CP_SELECT_SPLIT" ]      && flags+=(--cp-select-split "$CP_SELECT_SPLIT")
[ -n "$CP_EVAL_SPLIT" ]        && flags+=(--cp-eval-split "$CP_EVAL_SPLIT")
[ -n "$CP_INSTANCES" ]         && flags+=(--cp-instances "$CP_INSTANCES")
[ -n "$CP_OBSERVATION_STEPS" ] && flags+=(--cp-observation-steps "$CP_OBSERVATION_STEPS")
[ -n "$CP_METRIC" ]            && flags+=(--cp-metric "$CP_METRIC")
[ -n "$CP_FORECAST_SAMPLES" ]  && flags+=(--cp-forecast-samples "$CP_FORECAST_SAMPLES")
[ "$HIDE_EDGE_WEIGHTS" = "1" ] && flags+=(--hide-edge-weights)
# Space-separated list, e.g. OOD_POLICIES="degree_seed null"; unquoted on purpose
# so the shell splits it into one argument per policy
# shellcheck disable=SC2206
[ -n "$OOD_POLICIES" ]      && flags+=(--ood-policies $OOD_POLICIES)
[ -n "$SPLIT_MODE" ]        && flags+=(--split-mode "$SPLIT_MODE")

# ---------------------------------------------------------------------------
# SUBMIT: outside a job, queue ourselves with the allocation these flags need
# ---------------------------------------------------------------------------
# Slurm resolves #SBATCH directives at submission, so the GPU decision cannot
# be made from inside the job. Asking the pipeline keeps the answer in one
# place: it already knows the arm grammar and which repos are learned.
if [ -z "${SLURM_JOB_ID:-}" ]; then
  resources="$(python -m pipeline.run "${flags[@]}" --print-resources)"
  echo "$resources" | grep -v '^gpu='
  wants_gpu="$(echo "$resources" | sed -n 's/^gpu=//p')"

  if [ -n "${GRES:-}" ]; then
    gres="$GRES"
  elif [ "$wants_gpu" = "1" ]; then
    gres="gpu:1"
  else
    gres="none"
  fi

  submit=(
    --job-name="${JOB_NAME:-gwm_${DATASET}_${RUN}}"
    --cpus-per-task="${CPUS:-4}"
    --mem="${MEM:-$class_mem}"
    --time="${TIME:-$class_time}"
  )
  [ "$gres" != "none" ] && submit+=(--gres="$gres")
  [ -n "${PARTITION:-}" ] && submit+=(--partition="$PARTITION")
  [ -n "${SBATCH_ARGS:-}" ] && submit+=($SBATCH_ARGS)

  # Follow the ALLOCATION, not the workload: GRES=none on a training run means
  # "train on CPU", and telling torch cuda there would just crash on the node
  export DEVICE="${DEVICE:-$([ "$gres" = "none" ] && echo cpu || echo cuda)}"

  echo "[submit] sbatch ${submit[*]} --export=ALL $script_path"
  echo "[submit] DEVICE=$DEVICE"

  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[submit] python -m pipeline.run ${flags[*]} --device $DEVICE"
    echo "[submit] DRY_RUN=1, nothing submitted"
    exit 0
  fi

  exec sbatch "${submit[@]}" --export=ALL "$script_path"
fi

# ---------------------------------------------------------------------------
# RUN: inside the job
# ---------------------------------------------------------------------------
# Set by the submit path above. Left unset when this script was submitted
# directly with `sbatch ... pipeline.sbatch`, in which case follow whatever
# the allocation actually granted.
DEVICE="${DEVICE:-$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")}"
flags+=(--device "$DEVICE")

# The job id only exists in here, which is why --run is appended now rather
# than in the array above. One job -> one results tree, exactly like one job ->
# one .out file, so concurrent jobs on the same dataset cannot overwrite each
# other. RESUMING A JOB MEANS PASSING ITS RUN BACK: RUN=default_12345, or set
# RUN_JOBID=0 to get a stable run label that every rerun appends to.
run_label="$RUN"
[ "$RUN_JOBID" = "1" ] && [ -n "${SLURM_JOB_ID:-}" ] && run_label="${RUN}_${SLURM_JOB_ID}"
flags+=(--run "$run_label")

echo "[sbatch] job=${SLURM_JOB_NAME:-local} id=${SLURM_JOB_ID:-none} host=$(hostname)"
echo "[sbatch] task=$TASK dataset=$DATASET run=$run_label device=$DEVICE stages=$START_STAGE..$END_STAGE"
echo "[sbatch] resume this run with: RUN=$run_label RUN_JOBID=0 $script_path"
python -c "import torch; print('[sbatch] CUDA:', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU-ONLY'))"
echo "[sbatch] python -m pipeline.run ${flags[*]}"

python -m pipeline.run "${flags[@]}"

echo "[sbatch] done -> $RESULTS_ROOT/$TASK/$DATASET/$run_label/report.md"
