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
#     DATASET=ba TAG=ba40 ./sbatch/pipeline.sbatch
#
# Every `python -m pipeline.run` flag is an environment variable. Ordinary
# shell quoting, so a list value is just a quoted string:
#
#     DATASET=sbm ARMS="one_shot_free@oracle routing" ./sbatch/pipeline.sbatch
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
#     DATASET=ba TAG=ba40 DRY_RUN=1 ./sbatch/pipeline.sbatch
#
# Override it if you need to: GRES=gpu:2, or GRES=none to force CPU.
#
# RESULTS GO IN results/<tag>_<jobid>/, the same way logs go to logs/%x_%j.out,
# so two jobs on the same dataset can never overwrite each other. The job id is
# printed on the first line of the .out, along with the exact command to resume
# that run. Set TAG_JOBID=0 for a stable results/<tag>/ that every rerun appends
# to — which is what you want when you are deliberately resuming or chaining.
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
# TEMPLATES  (replace <dataset> / <tag>; every one is copy-pasteable)
# ============================================================================
#
# --- 1. FULL PIPELINE -------------------------------------------------------
# All five stages, all seven conditions, four budgets. The headline run.
# Requests a GPU (it trains f_theta).
#
#   DATASET=ba TAG=ba40 ./sbatch/pipeline.sbatch
#
#   DATASET=nethept MEM=64G TIME=48:00:00 ./sbatch/pipeline.sbatch
#
#
# --- 2. ORACLE ONLY ---------------------------------------------------------
# Condition 5 alone: the agent sees the true dynamics. No world model, so the
# train stage drops itself and this queues CPU-only. The ceiling, and the
# fastest way to iterate on the agent itself.
#
#   DATASET=sbm TAG=sbm40 ARMS=one_shot_free@oracle BASELINES=none \
#       ./sbatch/pipeline.sbatch
#
#
# --- 3. MC + ORACLE + WM ----------------------------------------------------
# The evaluator ablation proper: conditions 4, 5, 6 with the method held fixed,
# so the ONLY thing that varies is the inner-loop feedback. GPU.
#
#   DATASET=sbm TAG=sbm40 \
#   ARMS="one_shot_free@monte_carlo one_shot_free@oracle one_shot_free@world_model" \
#       ./sbatch/pipeline.sbatch
#
#
# --- 4. WORLD MODEL ONLY ----------------------------------------------------
# Just our method (condition 6), nothing to compare against. GPU.
#
#   DATASET=ba TAG=ba40 ARMS=one_shot_free@world_model BASELINES=none \
#       ./sbatch/pipeline.sbatch
#
#
# --- 5. MONTE CARLO ONLY ----------------------------------------------------
# Condition 4 alone: averaged simulator rollouts as feedback. CPU-only.
# MC_RUNS sets both its cost and its noise.
#
#   DATASET=sbm TAG=sbm40 ARMS=one_shot_free@monte_carlo BASELINES=none \
#   MC_RUNS=200 ./sbatch/pipeline.sbatch
#
#
# --- 5b. NATIVE AGENT ONLY --------------------------------------------------
# Condition 3: real executions only, NATIVE_MC_RUNS episode(s) per candidate.
# Keep it at 1 for the honestly model-free baseline — the whole point of this
# arm is that every number the agent sees costs a real episode. CPU-only.
#
#   DATASET=sbm TAG=sbm40 ARMS=one_shot_free@native BASELINES=none \
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
# Library algorithms (condition 1) — any of:
#   adaptive_greedy betweenness_seeds celf celf_local_search celf_pp
#   closeness_seeds cofim collective_influence community_celf community_im
#   community_ris degree_discount degree_ris_refine eigenvector_seeds
#   filtered_ris genetic_algorithm high_degree hill_climbing imm irie
#   kshell_seeds ldag mia_pmia pagerank_greedy pagerank_seeds random_seeds
#   ris_basic simulated_annealing skim sp1m ssa static_greedy tim
#   vanilla_greedy voterank weighted_degree
#
# External repos (condition 7) — external:<name> for any of:
#   learned (GPU):  moeim touplegdd deepim gcomb iminfector glie lense him piano
#   C++ (CPU):      opim ssa subsim imm tim oim
#
#
# --- 8. NO BASELINES --------------------------------------------------------
# Conditions 2-6 only, nothing classical. For when the baseline numbers for
# this tag are already on disk and only the agent side changed.
#
#   DATASET=ba TAG=ba40 BASELINES=none ./sbatch/pipeline.sbatch
#
#
# --- 9. SKIP WM TRAINING, USE ORACLE ----------------------------------------
# Two ways, and they mean different things:
#
#   (a) no world-model arm at all -> train removes itself, queues CPU-only:
#   DATASET=sbm TAG=sbm40 \
#   ARMS="routing one_shot_free@native one_shot_free@monte_carlo one_shot_free@oracle" \
#       ./sbatch/pipeline.sbatch
#
#   (b) keep a @world_model arm but reuse a checkpoint trained earlier:
#   DATASET=ba TAG=ba40 SKIP_STAGES=train ./sbatch/pipeline.sbatch
#   (b) still asks for a GPU, because the arm rolls f_theta out. It fails
#   loudly if results/<tag>/world_model/ has no checkpoint — it reuses, it
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
#   DATASET=ba TAG=ba40 START_STAGE=agent ./sbatch/pipeline.sbatch
#
# Data generation alone (CPU-only — this is what the old wm_generate_data_*
# scripts did):
#
#   DATASET=nethept START_STAGE=data END_STAGE=data MEM=64G \
#       ./sbatch/pipeline.sbatch
#
# Training alone on already-generated data (what wm_train.sbatch did):
#
#   DATASET=ba TAG=ba40 START_STAGE=train END_STAGE=train WM_MODEL=sage \
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
#   DATASET=ba TAG=ba40 START_STAGE=plots END_STAGE=plots ./sbatch/pipeline.sbatch
#   DATASET=ba TAG=ba40 START_STAGE=plots ./sbatch/pipeline.sbatch
#
# Fast enough to just run on the login node instead:
#   python -m pipeline.run --dataset ba --tag ba40 --start-stage plots
#
#
# --- 12. CODING AGENT ONLY, ALL LLM KNOBS -----------------------------------
# START_STAGE=agent END_STAGE=agent reuses the data and the checkpoint and
# only re-runs the sweep. Every agent-side parameter:
#
#   DATASET=sbm TAG=sbm40 START_STAGE=agent END_STAGE=agent FORCE=1 \
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
#                   add_edge remove_edge set_edge_weight)
#   EVALUATOR       fallback for arms written without an @evaluator suffix
#   BUDGET_PCTS     seed budgets as % of N, one run per value
#   BUDGETS         absolute k, overrides BUDGET_PCTS entirely
#   COMPARE         1 = replay every winner on the shared ground-truth referee.
#                   Keep this at 1. Without it the table is not comparable
#                   across conditions, because each arm's own reward comes
#                   from its own evaluator.
#   CREDIT          1 = per-action counterfactual credit in the feedback.
#                   Costs one extra rollout per action.
#   GRAPH_ID        evaluate on a specific graph in the store
#   BASELINE_TIMEOUT seconds before an external repo subprocess is killed
#
# Comparing two LLMs on the same data, as two jobs into two tags:
#
#   DATASET=sbm TAG=sbm40_sonnet LLM_MODEL=gpt-5.6-terra ./sbatch/pipeline.sbatch
#   DATASET=sbm TAG=sbm40_gpt    LLM_MODEL=gpt-5.6-sol     ./sbatch/pipeline.sbatch
#
#
# --- 13. ALL WORLD-MODEL TRAINING KNOBS -------------------------------------
#
#   DATASET=ba TAG=ba40_gt START_STAGE=train END_STAGE=train FORCE=1 \
#   WM_MODEL=gt HEAD=structured_residual HIDDEN_DIM=128 N_LAYERS=3 \
#   N_HEADS=4 FFN_DIM=256 DROPOUT=0.1 \
#   EPOCHS=400 LR=1e-3 WEIGHT_DECAY=5e-4 BATCH_SIZE=32 \
#   POS_WEIGHT=off PATIENCE=50 PLAN_GRAPHS=5 NO_PLAN_DEMO=0 \
#       ./sbatch/pipeline.sbatch
#
#   WM_MODEL     gcn | sage | gt | gat | gcnii
#   HEAD         linear | structured | structured_residual
#                structured/structured_residual REQUIRE POS_WEIGHT=off —
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
# Backbone sweep, five jobs into five tags sharing one dataset — generate the
# data once, then each job reuses it:
#
#   DATASET=ba TAG=ba40 START_STAGE=data END_STAGE=data ./sbatch/pipeline.sbatch
#   for M in gcn sage gat gt gcnii; do
#     DATASET=ba TAG=ba40 WM_MODEL=$M START_STAGE=train JOB_NAME=gwm_$M \
#         ./sbatch/pipeline.sbatch
#   done
#   (all five share results/ba40/data; checkpoints are named per backbone, so
#    they do not collide)
#
#
# --- 14. ALL DATA-GENERATION KNOBS ------------------------------------------
#
#   DATASET=sbm TAG=sbm_dense START_STAGE=data END_STAGE=data \
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
#   GEN_MODELS       IC, LT, or both — one transition file per dynamics
#   GEN_ALGORITHMS   spine seed selectors: random degree pagerank betweenness
#                    celf local_search. betweenness is exact Brandes, O(V*E)
#                    in pure networkx, so it is dropped by default on the big
#                    graphs (hours at 15K+ nodes).
#   GEN_ACTION_OPS   ops injected during generation. Omitting all of them
#                    gives diffusion-only data; node ops are the default; add
#                    the three edge ops for the full action space.
#   PROB_MODEL       weighted (p = 1/in_degree) | uniform (UNIFORM_P)
#   BUDGET_PCT_RANGE per-episode k drawn from this band, which is what lets ONE
#                    checkpoint serve the whole budget sweep. It must cover
#                    every BUDGET_PCTS value you intend to evaluate at.
#   ROLLOUTS         episodes per (graph, dynamics, selector)
#   INJECT_P         probability an intermediate step injects an action
#   CF_PROB          probability a step spawns counterfactual forks — the
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
#   DATASET=ba TAG=ba40_lt DIFFUSION_MODEL=LT ./sbatch/pipeline.sbatch
#
#
# --- 16. ABSOLUTE BUDGETS ---------------------------------------------------
# BUDGETS overrides BUDGET_PCTS, for matching a published table's k exactly.
#
#   DATASET=nethept BUDGETS="10 20 30 40 50" ./sbatch/pipeline.sbatch
#
#
# --- 17. RESUME AFTER A CRASH -----------------------------------------------
# Results live in results/<tag>_<jobid>/, so resuming means naming the tree the
# dead job created — a bare resubmit would start a FRESH one. The dead job's
# .out prints the line to use, verbatim:
#
#   [sbatch] resume this run with: TAG=ba40_918273 TAG_JOBID=0 .../pipeline.sbatch
#
#   DATASET=ba TAG=ba40_918273 TAG_JOBID=0 ./sbatch/pipeline.sbatch
#
# Every finished stage, every finished (budget, arm) result, and every skipped
# external baseline is on disk and gets reused; only the missing work runs.
# Check what is done first:
#
#   cat results/<tag>/pipeline.json                       # stage status + timings
#   column -s, -t results/<tag>/summary.csv | less -S     # every finished arm
#
# For a long-lived experiment you keep adding to, drop the job id entirely and
# always resume into the same tree:
#
#   DATASET=ba TAG=ba40 TAG_JOBID=0 ./sbatch/pipeline.sbatch
#
# To redo one stage from scratch instead of resuming it:
#
#   DATASET=ba TAG=ba40_918273 TAG_JOBID=0 \
#   START_STAGE=agent END_STAGE=agent FORCE=1 ./sbatch/pipeline.sbatch
#
#
# --- 18. DRY RUN ------------------------------------------------------------
# Print the resolved sbatch line and the full pipeline command, submit
# nothing. Always do this before a long run.
#
#   DATASET=ba TAG=ba40 DRY_RUN=1 ./sbatch/pipeline.sbatch
#
# To also validate the flags against the real parser and see the arm plan
# without executing a stage, on the login node:
#
#   python -m pipeline.run --dataset ba --tag ba40 \
#       --skip-stages data train agent plots report
#
#
# --- 19. SMOKE TEST ---------------------------------------------------------
# Prove the whole path end to end in a couple of minutes on a tiny graph
# before burning an allocation. Login node, no sbatch.
#
#   python -m pipeline.run --dataset ba --tag smoke \
#       --num-graphs 2 --syn-nodes 60 --rollouts 2 --mc-marginals 3 \
#       --epochs 3 --gen-models IC \
#       --baselines high_degree --arms one_shot_free@oracle \
#       --budget-pcts 10 --outer-iters 1 --mc-runs 20 --compare
#
# ============================================================================

set -euo pipefail

# Resolve before the cd, so `./sbatch/pipeline.sbatch` still resubmits the
# right file when it is launched from somewhere other than the repo root
script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

cd /home/rrishab/GraphWorldModel/
source venv/bin/activate
mkdir -p results logs

# ---------------------------------------------------------------------------
# DEFAULTS
# ---------------------------------------------------------------------------
DATASET="${DATASET:?DATASET is required, e.g. DATASET=ba ./sbatch/pipeline.sbatch}"
TAG="${TAG:-$DATASET}"
# Append _<jobid> to the results directory, the same way logs get _%j
TAG_JOBID="${TAG_JOBID:-1}"

# Dataset class drives the settings that genuinely cannot share a default:
# graph population, seed selectors, encoder width, and memory.
case "$DATASET" in
  er|ba|ws|sbm|karate)
    class_graphs=40 ; class_nodes=100 ; class_rollouts=20
    class_selectors="random degree pagerank betweenness"
    class_hidden=64 ; class_mem=16G ; class_time=24:00:00
    ;;
  nethept|twitter|digg|youtube|weibo)
    # Exact Brandes betweenness is O(V*E) in pure networkx — hours at this
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
GEN_MODELS="${GEN_MODELS:-IC LT}"
GEN_ALGORITHMS="${GEN_ALGORITHMS:-$class_selectors}"
GEN_ACTION_OPS="${GEN_ACTION_OPS:-add_node remove_node}"
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

# Agent stage
BASELINES="${BASELINES:-high_degree degree_discount pagerank_seeds celf_pp imm random_seeds}"
BASELINE_TIMEOUT="${BASELINE_TIMEOUT:-3600}"
ARMS="${ARMS:-routing one_shot_free@native one_shot_free@monte_carlo one_shot_free@oracle one_shot_free@world_model}"
NATIVE_MC_RUNS="${NATIVE_MC_RUNS:-1}"
BUDGET_PCTS="${BUDGET_PCTS:-1 5 10 20}"
BUDGETS="${BUDGETS:-}"
EVALUATOR="${EVALUATOR:-oracle}"
LLM_MODEL="${LLM_MODEL:-gpt-5.6-terra}"
TEMPERATURE="${TEMPERATURE:-}"
DIFFUSION_MODEL="${DIFFUSION_MODEL:-IC}"
HORIZON="${HORIZON:-10}"
OUTER_ITERS="${OUTER_ITERS:-5}"
WINDOWS="${WINDOWS:-3}"
MC_RUNS="${MC_RUNS:-200}"
N_SAMPLES="${N_SAMPLES:-50}"
ALLOWED_OPS="${ALLOWED_OPS:-add_node remove_node}"
COMPARE="${COMPARE:-1}"
CREDIT="${CREDIT:-0}"
GRAPH_ID="${GRAPH_ID:-}"

# ---------------------------------------------------------------------------
# ASSEMBLE
# ---------------------------------------------------------------------------
# Unquoted expansion is deliberate for the list-valued vars: each word becomes
# its own argv entry, which is what nargs="+" wants.
flags=(
  --dataset "$DATASET"
  --results-root "$RESULTS_ROOT" --seed "$SEED"
  --start-stage "$START_STAGE" --end-stage "$END_STAGE"

  --num-graphs "$NUM_GRAPHS" --syn-nodes "$SYN_NODES"
  --er-p "$ER_P" --ba-m "$BA_M" --ws-k "$WS_K" --ws-p "$WS_P"
  --sbm-blocks "$SBM_BLOCKS" --sbm-p-in "$SBM_P_IN" --sbm-p-out "$SBM_P_OUT"
  --gen-models $GEN_MODELS
  --gen-algorithms $GEN_ALGORITHMS
  --gen-action-ops $GEN_ACTION_OPS
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

  --baseline-timeout "$BASELINE_TIMEOUT" --native-mc-runs "$NATIVE_MC_RUNS"
  --evaluator "$EVALUATOR" --llm-model "$LLM_MODEL"
  --diffusion-model "$DIFFUSION_MODEL" --horizon "$HORIZON"
  --outer-iters "$OUTER_ITERS" --windows "$WINDOWS"
  --mc-runs "$MC_RUNS" --n-samples "$N_SAMPLES"
  --allowed-ops $ALLOWED_OPS
)

# `none` means the empty list: argparse's nargs="*" takes zero values when the
# next token is another flag, which is exactly how you ask for no baselines
[ "$BASELINES" = "none" ]   && flags+=(--baselines)   || flags+=(--baselines $BASELINES)
[ "$ARMS" = "none" ]        && flags+=(--arms)        || flags+=(--arms $ARMS)
[ "$SKIP_STAGES" = "none" ] || flags+=(--skip-stages $SKIP_STAGES)

# BUDGETS is absolute k and wins outright; otherwise budgets are percentages
if [ -n "$BUDGETS" ]; then
  flags+=(--budgets $BUDGETS)
else
  flags+=(--budget-pcts $BUDGET_PCTS)
fi

[ -n "$TEMPERATURE" ]     && flags+=(--temperature "$TEMPERATURE")
[ -n "$GRAPH_ID" ]        && flags+=(--graph-id "$GRAPH_ID")
[ "$COMPARE" = "1" ]      && flags+=(--compare)
[ "$CREDIT" = "1" ]       && flags+=(--credit)
[ "$FORCE" = "1" ]        && flags+=(--force)
[ "$NO_PLAN_DEMO" = "1" ] && flags+=(--no-plan-demo)

# ---------------------------------------------------------------------------
# SUBMIT — outside a job, queue ourselves with the allocation these flags need
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
    --job-name="${JOB_NAME:-gwm_$TAG}"
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
# RUN — inside the job
# ---------------------------------------------------------------------------
# Set by the submit path above. Left unset when this script was submitted
# directly with `sbatch ... pipeline.sbatch`, in which case follow whatever
# the allocation actually granted.
DEVICE="${DEVICE:-$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")}"
flags+=(--device "$DEVICE")

# The job id only exists in here, which is why --tag is appended now rather
# than in the array above. One job -> one results tree, exactly like one job ->
# one .out file, so concurrent jobs on the same dataset cannot overwrite each
# other. RESUMING A JOB MEANS PASSING ITS TAG BACK: TAG=ba40_12345, or set
# TAG_JOBID=0 to get a stable tag that every rerun appends to.
run_tag="$TAG"
[ "$TAG_JOBID" = "1" ] && [ -n "${SLURM_JOB_ID:-}" ] && run_tag="${TAG}_${SLURM_JOB_ID}"
flags+=(--tag "$run_tag")

echo "[sbatch] job=${SLURM_JOB_NAME:-local} id=${SLURM_JOB_ID:-none} host=$(hostname)"
echo "[sbatch] dataset=$DATASET tag=$run_tag device=$DEVICE stages=$START_STAGE..$END_STAGE"
echo "[sbatch] resume this run with: TAG=$run_tag TAG_JOBID=0 $script_path"
python -c "import torch; print('[sbatch] CUDA:', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU-ONLY'))"
echo "[sbatch] python -m pipeline.run ${flags[*]}"

python -m pipeline.run "${flags[@]}"

echo "[sbatch] done -> $RESULTS_ROOT/$run_tag/report.md"
