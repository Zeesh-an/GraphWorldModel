# Test run: every baseline plus the oracle coding agent, all 8 tasks

SLURM, one `sbatch/pipeline.sbatch` submission per task. Each runs **condition 1** (our classical baselines), **condition 7** (every wired published repo, classical and learned), and **condition 5** (`evolve_free@oracle`, the coding agent against the analytic oracle).

**No world model anywhere.** `SKIP_STAGES=train` is safe because `needs_world_model` returns `False` for every arm here, so no checkpoint is built or loaded. That is what keeps these cheap.

`COMPARE=1` on every submission. Without it each arm reports its own evaluator's number and the table is not a comparison; `--compare` replays every arm through one ground-truth Monte Carlo referee, which is the only column that may be read across rows.

---

## 0. Setup, once

```bash
source venv/bin/activate
python -m baselines.setup_baselines --all      # clones, patches, builds, per-baseline venvs
python -m baselines.setup_baselines --list     # confirm what is ready
```

**55 of the 111 registered repos are wired** and appear below. The rest are `blocked` with a recorded reason (no public code, Python 2, wrong input object, wrong output type) and are not runnable by anyone. `setup_baselines --all` will report failures for repos whose upstream has drifted; those are safe to drop from a `BASELINES` string without changing anything else.

### Setup status: 52 of 55

Five rounds against real cluster runs took this from 13 failures to 3, all of them fixes to OUR harness rather than to the repos. The three that remain are environment-bound and are **already removed from the `BASELINES` lists below**, so the commands run clean as written.

| Still failing | Why | To enable it |
| --- | --- | --- |
| `finder`, `finder_epi` | `tensorflow-gpu==1.14.0`'s newest wheel is cp37m, and CPython 3.7 is end-of-life and absent from uv's managed downloads, so the venv falls back to 3.8 and the pin can never resolve. Relaxing it is not a route: this is TF1 with custom Cython extensions, so the `compat.v1` shim that carries `coupledgnn` and `cascn` does not carry it | install a system 3.7 (pyenv, conda, module), then add `external:finder` back to the CND list and `external:finder_epi` to the EC list. The adapters are written and run the moment one exists |
| `decycler` | Boost is not installed on the node | `module load boost`, or `apt install libboost-program-options-dev`, then add `external:decycler` back to the CND list |

**What was actually wrong, all of it in our own setup code.** Recorded because each one was silent rather than loud, and four of the five would have quietly produced a wrong or missing baseline rather than an error.

| Bug | Effect |
| --- | --- |
| the patch idempotency test asked "is `new` in the file?" | for a pin-relaxing patch the replacement is a SUBSTRING of the original, so it matched the unpatched text and skipped forever. **5 patches across `ccgl` and `casflow` had never once applied**, while printing `patch already applied` every run. Now the test depends on the patch shape |
| `spec.python` was passed to nothing, and a wrong-version venv was never rebuilt | ~110 specs carried a pin that did nothing, and once a venv existed at the wrong version no re-run could fix it. Now `uv venv --python <ver>` plus a version check that rebuilds |
| `torch-scatter` / `torch-sparse` built in isolation | their `setup.py` imports torch. Now pre-installed with `--no-build-isolation`, under `FORCE_ONLY_CPU=1` because the node's CUDA 12.9 disagreed with the wheel's 13.0. The scan reads requirements files as well as `pip_packages`, and normalizes names per PEP 503 (glie spells it `torch_sparse`) |
| `explosive_immunization` relinked stale objects, and its makefile ignores `LIBS` | `-fcommon` never reached the compile step and `-lm` never reached the link. Now `make clean` first, with `-lm` inside `CFLAGS` |
| the ABI hint scanned the whole error for `cp3(\d)` | it read "You require CPython 3.8" and reported the interpreter you already have as the fix, and parsed `cp310` as minor version 1. Now parses only the published-tag list with `\d+`, and says so plainly when the pin is already correct |

Two more traps worth knowing:

- `rl4im`'s `torch==1.7.0` has no arm64 wheels at all, so it installs on the cluster and not on a Mac.
- `adaptiveim` is C++ and needs `make -f Makefile_expepic`. Its `rdtsc()` is x86 inline asm, patched to `std::chrono` at setup.

---

## 1. The gateway blocks the agent arm, not the baselines

Checked 2026-08-12:

| Family | State |
| --- | --- |
| all 7 `gpt-*` models | `429 model_cooldown`, reset ~146 h (about 2026-08-18) |
| all 7 `claude-*` models | `503 auth_unavailable: no auth available (providers=claude)` |

Until one family answers, set `ARMS=none` and the submissions below run baselines only. Everything else is unaffected. When the gateway returns, re-submit the same line with the `ARMS` value shown and `START_STAGE=agent`; the `data/` stage is already on disk and only the agent arm runs.

If a different model is live, set `LLM_MODEL=<name>`.

---

## 2. Dataset per task

Chosen to finish in reasonable time while still being a graph the task's own literature publishes on.

| Task | Dataset | Size | Why |
| --- | --- | --- | --- |
| `influence_maximization` | `netscience` | 1,589 / 2,742 | your existing IM run's graph, so this is directly comparable |
| `adaptive_online_im` | `netscience` | 1,589 / 2,742 | same graph, so the adaptivity gap divides against the IM row |
| `critical_node_detection` | `power_grid` | 4,941 / 6,594 | byte-identical to the file CoreHD, BPD and NIRM all report. §9.4 predicts we LOSE here, which is the informative case |
| `source_localization` | `jazz` | 198 / 2,742 | SL-VAE's own graph; reward is exact F1, so no evaluator noise |
| `influence_blocking` | `email_eu_core` | 1,005 / 24,929 | SandIMIN Table 5 |
| `cascade_reconstruction` | `infectious` | 410 / 2,765 | Xiao ICDM'18 Table I, digit for digit |
| `epidemic_control` | `primary_school` | 242 | SocioPatterns, 11 ground-truth classes, dense enough to test the DAVA/NetShield reversal |
| `cascade_prediction` | `casflow_aps` | corpus | CasFlow's own bundle, this literature's de-facto benchmark |

---

## 3. The eight submissions

```bash
# ---------------------------------------------------------------- 1. influence maximization
TASK=influence_maximization \
DATASET=netscience \
RUN=testrun \
RUN_JOBID=0 \
SKIP_STAGES=train \
BASELINES="high_degree degree_discount pagerank_seeds celf_pp imm voterank random_seeds \
external:opim external:ssa external:subsim \
external:touplegdd external:deepim external:moeim external:glie" \
ARMS="evolve_free@oracle" \
EVALUATOR=oracle \
BUDGET_PCTS="1 5 10 20" \
HORIZON=10 MC_RUNS=200 OUTER_ITERS=10 N_SAMPLES=50 \
COMPARE=1 FORCE=1 BASELINE_TIMEOUT=21600 \
GRES=gpu:1 MEM=64G TIME=48:00:00 \
./sbatch/pipeline.sbatch

# ---------------------------------------------------------------- 2. adaptive online IM
# ROUNDS=3 is what makes it adaptive. static_split and imm are the matched non-adaptive
# controls the adaptivity gap divides by, so do not drop them.
# adapt_greedy is deliberately ABSENT: see the cost note in section 5.
TASK=adaptive_online_im \
DATASET=netscience \
RUN=testrun \
RUN_JOBID=0 \
SKIP_STAGES=train \
BASELINES="adapt_epic adapt_degree_discount adapt_degree adapt_pagerank adapt_random \
static_split celf_pp imm \
external:adaptiveim external:rl4im" \
ARMS="adaptive_free@oracle evolve_free@oracle" \
EVALUATOR=oracle \
ROUNDS=3 ROUND_GAP=1 FEEDBACK_MODEL=full_adoption \
BUDGET_PCTS="1 5 10 20" \
HORIZON=10 MC_RUNS=200 OUTER_ITERS=10 N_SAMPLES=50 \
COMPARE=1 FORCE=1 BASELINE_TIMEOUT=21600 \
GRES=gpu:1 MEM=64G TIME=48:00:00 \
./sbatch/pipeline.sbatch

# ---------------------------------------------------------------- 3. critical node detection
# remove_semantics=blocked comes from the registry and is asserted; do not override it.
TASK=critical_node_detection \
DATASET=power_grid \
RUN=testrun \
RUN_JOBID=0 \
SKIP_STAGES=train \
BASELINES="adaptive_degree iterative_betweenness collective_influence_r corehd corehd_r \
bpd_r decycling explosive_immunization gnd gndr netshield kshell_removal \
degree_removal betweenness_removal acquaintance_immunization random_removal \
external:gdm external:mind external:nirm external:dcrs external:selinda \
external:gnd external:collective_influence \
external:explosive_immunization external:dismantling_review" \
ARMS="evolve_free@oracle" \
EVALUATOR=oracle \
BUDGET_PCTS="1 5 10 20" \
HORIZON=10 MC_RUNS=200 OUTER_ITERS=10 N_SAMPLES=50 \
COMPARE=1 FORCE=1 BASELINE_TIMEOUT=21600 \
GRES=gpu:1 MEM=64G TIME=48:00:00 \
./sbatch/pipeline.sbatch

# ---------------------------------------------------------------- 4. source localization
# k is a property of the INSTANCE, so the budget sweep is a single point by design.
# SL_OBSERVATION=binary is the comparable column: our default `marginal` is strictly
# more informative than the single realization the literature observes.
TASK=source_localization \
DATASET=jazz \
RUN=testrun \
RUN_JOBID=0 \
SKIP_STAGES=train \
SL_OBSERVATION=binary SL_BUDGET_MODE=episode \
BASELINES="lpsi netsleuth ojc jordan_center rumor_centrality dynamic_age \
effective_distance dmp_localize infected_degree infected_betweenness \
infected_closeness infected_eigenvector random_sources \
external:graphsl_lpsi external:graphsl_netsleuth external:graphsl_ojc \
external:graphsl_gcnsi external:graphsl_ivgd external:graphsl_slvae \
external:cosasi_jordan external:cosasi_netsleuth external:cosasi_lisn \
external:cosasi_rumor_centrality" \
ARMS="evolve_free@oracle gradient_free@oracle" \
EVALUATOR=oracle \
HORIZON=10 MC_RUNS=200 OUTER_ITERS=10 N_SAMPLES=50 \
COMPARE=1 FORCE=1 BASELINE_TIMEOUT=21600 \
GRES=gpu:1 MEM=64G TIME=48:00:00 \
./sbatch/pipeline.sbatch

# ---------------------------------------------------------------- 5. influence blocking
# BUDGETS is ABSOLUTE k, not a percentage: percent-of-N is used by NOBODY in this
# literature, while k in {10..50} is the shared convention.
TASK=influence_blocking \
DATASET=email_eu_core \
RUN=testrun \
RUN_JOBID=0 \
SKIP_STAGES=train \
BLOCKING_LEVER=counter_seed TIE_BREAK=auto \
BASELINES="proximity multi_hop_proximity rps cmia_o cldag forward_blocking \
reverse_blocking degree_blocking pagerank_blocking betweenness_blocking random_blocking \
external:sandimin external:imin_joc external:diffim" \
ARMS="evolve_free@oracle" \
EVALUATOR=oracle \
BUDGETS="10 20 30 40 50" \
HORIZON=10 MC_RUNS=200 OUTER_ITERS=10 N_SAMPLES=50 \
COMPARE=1 FORCE=1 BASELINE_TIMEOUT=21600 \
GRES=gpu:1 MEM=64G TIME=48:00:00 \
./sbatch/pipeline.sbatch

# ---------------------------------------------------------------- 6. cascade reconstruction
# --trace-parents is switched on by the registry. observed_only is the reward-gaming
# control (precision 1.0 by construction) and must stay in the pool.
TASK=cascade_reconstruction \
DATASET=infectious \
RUN=testrun \
RUN_JOBID=0 \
SKIP_STAGES=train \
CR_SETTING=final_snapshot CR_OBSERVATION_RATE=0.5 CR_TREE_WEIGHT=0.6 \
BASELINES="delayed_bfs ordered_steiner_closure greedy_ordered steiner_tree tree_sampling \
personalized_pagerank consistent_tree_wpct consistent_tree_wbct cult dhrec cri netfill \
jordan_backward observed_only one_hop random_reconstruction \
external:ditto external:ditto_dhrec external:ditto_cri \
external:grin external:spin external:deep_demixing" \
ARMS="decode_free@oracle evolve_free@oracle" \
EVALUATOR=oracle \
HORIZON=10 MC_RUNS=200 OUTER_ITERS=10 N_SAMPLES=50 \
COMPARE=1 FORCE=1 BASELINE_TIMEOUT=21600 \
GRES=gpu:1 MEM=64G TIME=48:00:00 \
./sbatch/pipeline.sbatch

# ---------------------------------------------------------------- 7. epidemic control
# SIR/SIS/SEIR come from the registry. beta and gamma cross all three stages and are
# read back by the head, so set them here rather than per-stage.
TASK=epidemic_control \
DATASET=primary_school \
RUN=testrun \
RUN_JOBID=0 \
SKIP_STAGES=train \
DIFFUSION_MODEL=SIR GEN_MODELS=SIR EPI_LEVER=vaccinate EPI_GAMMA=0.3 \
BASELINES="degree_immunization adaptive_degree_immunization acquaintance_immunization \
netshield netshield_plus dava dava_fast frontier_immunization greedy_walk \
eigenvector_immunization kshell_immunization betweenness_immunization random_immunization \
external:netimm_netshield external:netimm_dava external:netimm_dava_fast \
external:netimm_netshape external:netimm_degree external:netimm_random \
external:gdm_epi external:collective_influence_epi \
external:explosive_immunization_epi external:dismantling_review_epi" \
ARMS="evolve_free@oracle" \
EVALUATOR=oracle \
BUDGET_PCTS="1 5 10 20" \
HORIZON=15 MC_RUNS=200 OUTER_ITERS=10 N_SAMPLES=50 \
COMPARE=1 FORCE=1 BASELINE_TIMEOUT=21600 \
GRES=gpu:1 MEM=64G TIME=48:00:00 \
./sbatch/pipeline.sbatch

# ---------------------------------------------------------------- 8. cascade prediction
# No simulator: real logs replayed. CP_SPLIT=chronological is the leak-free protocol
# and the headline experiment; run CP_SPLIT=random afterwards as the A/B.
TASK=cascade_prediction \
DATASET=casflow_aps \
RUN=testrun \
RUN_JOBID=0 \
SKIP_STAGES=train \
CP_SPLIT=chronological CP_METRIC=msle \
BASELINES="szabo_huberman feature_linear feature_gbt weng_communities \
seismic hawkes hawkes_hybrid rpp hip branching_factor \
neighborhood_size degree_scaled reachability persistence mean_size random_prediction \
external:casflow external:ccgl external:ctcp external:cascn external:coupledgnn" \
ARMS="evolve_free@oracle" \
EVALUATOR=oracle \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=50 \
COMPARE=1 FORCE=1 BASELINE_TIMEOUT=21600 \
GRES=gpu:1 MEM=64G TIME=48:00:00 \
./sbatch/pipeline.sbatch
```

**If you run `cascade_prediction` on `taoke` instead**, add `CP_MIN_SIZE=3`. At the shipped `10` the surviving pool publishes 85% of itself within 785 s of the first, so no `--cp-horizon` can both clear that gap and exceed the 3600 s observation window, and the generator raises rather than leaking.

---

## 4. Single agent arms, outside the pipeline

The same shape as your existing IM commands, for iterating on one arm without re-running a whole task. `--data-dir` points at the `data/` the submissions above already produced.

```bash
# netscience, evolve/free, oracle
python -m coding_agent.run \
    --model gpt-5.6-terra \
    --task influence_maximization \
    --data-dir results/influence_maximization/netscience/testrun/data \
    --method evolve --strategy-mode free \
    --evaluator oracle --budget-pct 5 --horizon 10 \
    --allowed-ops add_node \
    --outer-iters 10 --n-samples 50 \
    --mc-runs 200 --compare \
    --out-json results/influence_maximization/netscience/testrun/agent/pct5/evolve_free@oracle.json

# netscience, evolve/scored, oracle
python -m coding_agent.run \
    --model gpt-5.6-terra \
    --task influence_maximization \
    --data-dir results/influence_maximization/netscience/testrun/data \
    --method evolve --strategy-mode scored \
    --evaluator oracle --budget-pct 5 --horizon 10 \
    --allowed-ops add_node \
    --outer-iters 10 --n-samples 50 \
    --mc-runs 200 --compare \
    --out-json results/influence_maximization/netscience/testrun/agent/pct5/evolve_scored@oracle.json

# power_grid containment, evolve/free, oracle  (the budget buys remove_node)
python -m coding_agent.run \
    --model gpt-5.6-terra \
    --task critical_node_detection \
    --data-dir results/critical_node_detection/power_grid/testrun/data \
    --method evolve --strategy-mode free \
    --evaluator oracle --budget-pct 5 --horizon 10 \
    --allowed-ops remove_node \
    --outer-iters 10 --n-samples 50 \
    --mc-runs 200 --compare \
    --out-json results/critical_node_detection/power_grid/testrun/agent/pct5/evolve_free@oracle.json
```

---

## 5. Two cost traps

**`adapt_greedy` costs about 37 minutes for one row** and is left out of submission 2 on purpose. One round is 14.7 s on a 198-node graph (40 candidates re-scored against 8 simulations each), and the policy is re-invoked inside *every* referee episode, so at `MC_RUNS=200` it is `14.7 s x rounds x 200`. That cost IS the finding adaptive IM exists to publish, so run it deliberately:

```bash
TASK=adaptive_online_im DATASET=netscience RUN=adaptgreedy RUN_JOBID=0 \
SKIP_STAGES=train BASELINES="adapt_greedy adapt_epic static_split" ARMS=none \
ROUNDS=3 BUDGET_PCTS="5" HORIZON=10 MC_RUNS=20 COMPARE=1 FORCE=1 \
GRES=gpu:1 MEM=64G TIME=12:00:00 ./sbatch/pipeline.sbatch
```

**`BASELINE_TIMEOUT=21600` (6 h) is per external repo.** The learned ones train before they select: `finder`, `gdm`, `mind`, `rl4im`, `deepim` and `casflow` all fit a model first. With 11 externals on `critical_node_detection` the worst case is long, which is why `TIME=48:00:00`.

---

## 6. What to read, and what would mean a bug

| Task | Column | Row to beat | A result that means something is wrong |
| --- | --- | --- | --- |
| `influence_maximization` | `spread_ground_truth` | `imm`, and `external:opim` | every arm within noise of `high_degree`: the graph is degree-trivial |
| `adaptive_online_im` | the adaptivity gap table, and `evaluator_seconds` | `static_split` at matched `k` | a gap far above 1.0. Theory caps the myopic gap at 4 and non-adaptive greedy is provably no worse, so a large spread win is a bug. **The claim is cost** |
| `critical_node_detection` | `spread_ground_truth`, plus the structural table | `adaptive_degree` (HDA) | `degree_rank_spearman` near 0.762: the arm re-derived the degree heuristic, MIND's finding about GDM |
| `source_localization` | `f1`, and `generalization_gap` | `lpsi` / `external:graphsl_lpsi` | selection F1 far above held-out F1: the program memorized episodes |
| `influence_blocking` | `prevented_influence` | `proximity`, `external:sandimin` | `cldag` or `cmia_o` near `random_blocking`: that was a real bug, fixed 2026-08-12, pinned by `check_mia_scores_do_not_collapse_onto_the_periphery` |
| `cascade_reconstruction` | tree-weighted reward, `path_precision` vs `event_f1` | `delayed_bfs`, `external:ditto` | reward near `trivial_decoder_reward`, or `path_recall` under half |
| `epidemic_control` | attack rate, `eigendrop_vs_attack.png` | `degree_immunization`, `external:netimm_dava` | our `netshield` and `dava` disagreeing with `netimm_*`: they produce byte-identical node sets, so a difference is an adapter bug |
| `cascade_prediction` | `msle`, with `n_failed` beside it | `szabo_huberman`; `mean_size` is the real floor | any arm below `trivial_predictor_error` |

Every arm also prints `evaluator_calls` / `evaluator_seconds` and `real_env_episodes`. At `@oracle` the last should be small; the comparison that matters later is against `@monte_carlo`.
