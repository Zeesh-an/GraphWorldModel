# Helpers for section 9 of final_results_plan.md (ablations and extra experiments).
# Source this from the repo root, once per shell: source sbatch/ablation_helpers.sh
# It only defines functions; they submit through ./sbatch/pipeline.sbatch.

# Host 1: influence maximization on netscience under IC. Later assignments win, so a call overrides any knob.
im_abl() {
  env TASK=influence_maximization DATASET=netscience RUN_JOBID=0 \
    DIFFUSION_MODEL=IC GEN_MODELS=IC BASELINES=none ARMS=none \
    LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
    WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
    ROLLOUTS=100 MC_MARGINALS=30 \
    HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
    EVALUATOR=oracle BUDGET_PCTS="10" \
    HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
    MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
    FEEDBACK=default ACTION_CONDITIONING=message \
    CPUS=4 GRES=none MEM=31G TIME=24:00:00 \
    "$@" ./sbatch/pipeline.sbatch
}

# Host 2: critical node detection on power_grid under IC
cnd_abl() {
  env TASK=critical_node_detection DATASET=power_grid RUN_JOBID=0 \
    DIFFUSION_MODEL=IC GEN_MODELS=IC BASELINES=none ARMS=none \
    LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
    WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
    ROLLOUTS=100 MC_MARGINALS=30 \
    HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
    EVALUATOR=oracle BUDGET_PCTS="10" OUTBREAK_PCT=10 OUTBREAK_SELECTOR=random \
    HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
    MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
    FEEDBACK=default ACTION_CONDITIONING=message \
    CPUS=4 GRES=none MEM=31G TIME=24:00:00 \
    "$@" ./sbatch/pipeline.sbatch
}

# copy_run <task> <dataset> <new_run> [data]: copies data/ and world_model/ out of final_ic,
# or data/ alone when the fourth argument is `data` (the runs that retrain)
copy_run() {
  local src="results/$1/$2/final_ic" dst="results/$1/$2/$3"
  [ -e "$dst" ] && { echo "refusing to overwrite $dst"; return 1; }
  mkdir -p "$dst"
  cp -r "$src/data" "$dst/"
  [ "${4:-both}" = "data" ] || cp -r "$src/world_model" "$dst/"
  # The copied JSONs name the run they came from; point them at the copy so it stands alone
  grep -rl "$2/final_ic" "$dst/data/metadata.json" "$dst"/world_model/*.json 2>/dev/null \
    | xargs -r perl -pi -e "s#\Q$2/final_ic\E#$2/$3#g"
}
