# Coding-Agent Outer Loop

An LLM coding agent that **designs Influence-Maximization algorithms as executable Python programs**, where the designed algorithm emits graph interventions across **all** timesteps `t₀ … t_T` (not just a seed set at `t₀`) and is scored by rolling its actions through a learned Graph World Model (fast inner loop) with the NDlib Monte-Carlo simulator as ground truth.

---

## 1. Problem formulation: multi-timestep algorithms

### The reframing

Mark an algorithm's output as `R` (its _result_). Every classical IM/SL algorithm (PageRank seeding, greedy/CELF, RIS, source localization) produces only **`R_{t₀}`**: one solution at time zero, after which the dynamics simply run. That is the **degenerate special case** of what this subsystem targets:

```
classical:   R_{t₀}                          (seed set, then hands off to dynamics)
this work:   R_{t₀}, R_{t₁}, R_{t₂}, …, R_{t_T}   (an intervention at every timestep)
```

Equivalently, in optimization terms: the classical formulation confines the decision variables to `t₀`; here the problem is **augmented with decision variables at `t₁, t₂, …`**. Even adding only `t₁` changes the problem class, a human can hand-design PageRank, but not a "super-PageRank" that anticipates and schedules interventions across time. A coding agent can attempt it, and classical methods fall out as the `R_{t₀}`-only special case (an all-at-`t₀` plan with empty later bags, exactly what the `--baseline` mode constructs).

**Scope note:** the horizon does not need to be long for this to be a real contribution. Two or three intervention timesteps (`t₀ + t₁`, or `t₀ + t₁ + t₂`) already constitute the augmented problem; 100-step plans are neither needed nor affordable.

### Formal objects

- **State** `S_t = (infected_t, frontier_t)`: ever-activated nodes and the currently spreading wave (sorted node-id lists; `data.wm_simulator.State`).
- **Action bag** `A_t = [ActionOp, …]`: zero or more operations applied at step `t`. `ActionOp(op, target, destination=None, weight=None)` is the **unified action representation**: an explicit _type_ dimension (`op`) and _target_ dimensions (`target`, `destination`, `weight`), so one formulation covers seed injection, node removal, and structural edits across task types instead of one encoding per problem. The same value type is used by the data generator, the trained world model, and every strategy here (`types.ActionOp` is the simulator's class, re-exported, there is exactly one definition in the project).
- **Plan** `R_{t₀…T} = [A_0, A_1, …, A_T]`: what Method 1 strategies emit (`horizon + 1` bags; bag `t` is applied at timestep `t`).
- **Transition** `S_{t+1} ~ P(· | S_t, A_t, G_t)`: one simulator/world-model step: the deterministic action effect (T_exo) followed by one stochastic diffusion step (T_endo). Edge ops mutate `G_t` itself.
- **Reward** `J = E[ |infected_{T+1}| ]`: expected final spread, estimated by an ensemble (`mc_runs` NDlib rollouts or `n_samples` world-model rollouts).
- **Budget** `b`: at most `b` actions of the task's `budget_op` across the whole plan (`add_node` for seeding, `remove_node` for containment and vaccination, an edge op for the edge levers); every other op is uncharged.

### What the feedback optimizes

The refinement loop adjusts **the algorithm (the skill), never the LLM's weights**. The outer loop searches over _programs_: propose an algorithm → roll its actions through the environment → feed the rollout diagnostics back as a revision prompt → the agent rewrites the program. The world model's role is to make each candidate evaluation cheap enough that this program-level search is affordable.

The feedback per iteration (built by `methods/base.py::summarize` + `prompts.py::build_feedback_prompt`) contains:

- **reward ± SE**: ensemble-mean final spread with its noise floor;
- **paired delta with a noise verdict** (`methods/base.py::paired_delta`), the signed change against the incumbent, plus the 2σ band on that comparison and an explicit ruling: `INSIDE THE NOISE, this change did nothing measurable`. This matters more than it sounds. On a hub-dominated graph the entire algorithmic spread between `high_degree`, `degree_discount` and RIS can be ~4 nodes while the ensemble SE at `--n-samples 50` is ~4 nodes: without the band the agent reads sampling luck as a result and chases it;
- **the seed set it actually chose**: every seed with its total degree and community id. Improving a selection you cannot see is guesswork;
- **infected + frontier curves** and, when the cascade dies before the horizon, the death step ("actions scheduled after t=X did nothing");
- **unreached-nodes report**: per-node `P(infected)` across the ensemble (`Trajectory.final_marginals`, computed by every env), ranked by **estimated residual gain** rather than degree: a high-degree node the cascade never reaches is usually unreachable, whereas a high-gain one is worth a seed;
- **residual-gain hints**: the highest-value nodes _not_ seeded, scored by reverse-reachable-set coverage not already covered by the seed set, in units of expected extra nodes. This is the marginal-gain signal CELF would compute, at RIS cost, and it never touches the metered evaluator. IC only (RR sets do not describe LT), skipped above `rr_max_nodes`, and the cover index is cached on the `GraphInfo` so it is built once per run, not once per turn;
- **community coverage**: seeds per community against community size and the mean reach inside it, plus a count of communities that got no seed at all;
- **adjacent-seed-pairs report**: seeds with overlapping neighborhoods (likely redundant budget);
- **baseline leaderboard**: one rollout per algorithm in `methods/base.py::anchor_algorithms` (`high_degree`, `degree_discount`, `pagerank_seeds`, `imm`, `random_seeds`; RIS members dropped under LT) in the same env at the same budget and horizon, sorted, included in every prompt. The bar is the top row, not one arbitrary classical algorithm. Under an MC evaluator these rollouts are charged to the arm like any other, so the list is deliberately short, trim `anchor_algorithms` if the episode cost matters more than the target;
- **reference diff**: the _strongest_ baseline's per-node marginals diffed against the agent's, naming the specific nodes it reaches that this strategy misses (and vice versa) plus the net expected-spread gap. Costs no extra rollouts: both marginal vectors already exist;
- **the incumbent script itself**: see below;
- optional **per-action counterfactual credit** (`--credit`) and **error tracebacks** on failed scripts.

Feedback is deliberately descriptive, never prescriptive: the environments report where the cascade went and what each action bought, but never which node to pick, the algorithm design stays with the agent.

### The edit target is the incumbent, not the last attempt

Every feedback turn shows two programs: the one that just ran, and (when they differ) the **best-scoring script so far**, labelled as the one to edit. This is the difference between a hill climb and a random walk. Editing the latest attempt means a regression becomes the base for every iteration after it, so a single bad sample derails the rest of the search; `evolve` has always done this correctly (its parent is the population best), and `one_shot` now does too.

### Every generation is scored on a fresh realization, paired

A sampled evaluator (`@monte_carlo`, `@oracle`, `@world_model`) used to roll out every candidate on the environment's one fixed base seed. That is common random numbers taken too far: the loop sees the diagnostics of one realization and tunes to its noise. Measured on `epidemic_control/primary_school` at `k=48`, the search reported 40.6, the same program re-scored 67-76 on three other seeds, and the ground-truth referee said 63.7; the gap was 12-30 nodes at every budget, and on SIR over 16 steps with 50 samples it reversed the ranking against `frontier_immunization`. `methods.base.evaluate_strategy` now takes a `seed`, `evolve` and `one_shot` draw one per generation (`seed + generation + 1`; `one_shot` counts successful evaluations rather than turns, so a repair turn does not burn a realization), and `methods.base.rescore` re-runs the **incumbent** on that same seed before the two are compared, so a comparison is paired within a generation and never across. The incumbent's plan is cached on the strategy (`_plan`), so a re-score costs one rollout, not a second `plan_horizon()`. `rescore` is a no-op for the recover and forecast families, whose rewards have no realization to overfit and whose evaluation is the expensive half. A refinement that does not change the plan now reads as exactly zero improvement rather than as whatever the new draw happened to say.

### Checkpoint and resume

`one_shot` and `evolve` write `<arm>.ckpt.json` beside the result they will become, after **every** turn: evaluations and repairs alike. A file there means "this arm did not finish". The next run of the same arm picks it up and continues from the generation it stopped at: population, best-so-far (its trajectory serialized, not re-evaluated), counters, the full conversation thread, the transcript, and the anchor leaderboard, so a resume re-pays for neither the baseline rollouts nor the LLM calls it already made.

A checkpoint is only reused when its **fingerprint** matches the run about to start: method, mode, evaluator, model, temperature, budget, horizon, `outer_iters`, allowed ops, mc_runs, n_samples, seed, graph size, the checkpoint path, and every task-defining setting that is not in the results path (task, remove semantics, outbreak, lever, tie-break, compartmental rates, round schedule, edit rate, campaigns, the inverse/decoder/forecast pools). Resuming a search under a different budget or model would silently splice two experiments together, which is worse than losing the work, so a mismatch prints what differed and starts fresh. `--force` (pipeline) and `--no-resume` (standalone) delete the checkpoint first, because "redo" must mean redo. A finished run deletes its own checkpoint, so a leftover file always means an interruption.

`layout.result_globs()` excludes the suffix, so a checkpoint is never mistaken for a result by the report or the summary table. `per_step` and `windowed` have no refinement loop and nothing to resume; canned arms are one deterministic rollout and are not checkpointed either.

### Failed scripts do not consume the search budget

A script that fails to parse, import, build, or run teaches the next turn something, but it is not an evaluation. `--outer-iters` counts **evaluations**; failures draw on a separate `max_repairs` budget (default 3) and the loop only gives up once that is exhausted. Previously a run with `--outer-iters 5` and two bad scripts silently became a 3-round search. That is `one_shot`'s rule. Under `evolve` a failed generation is a generation: it records the error, increments stagnation, feeds the error text into the next prompt, and the method raises (and the pipeline marks the arm skipped) only if every generation fails.

---

## 2. Architecture

```
             ┌────────────────────── outer loop (methods/) ──────────────────────┐
             │                                                                    │
  LLM (agent.py) ──► Python Strategy script ──► executor.py ──► ActionFn(state,t) │
             ▲            (exec + validate)                          │            │
             │                                                        ▼            │
             │                                    Environment.rollout (envs/)      │
             │                                    WM (fast) / NDlib MC (truth)     │
             │                                                        │            │
             └──────── reward + summary + credit + errors ◄───────────┘            │
                                                                                   │
             └─────────────────────────────────────────────────────────────────────┘
```

| File          | Responsibility                                                                                                                                                                                           |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `types.py`    | `GraphInfo` (read-only graph view: `edge_index (2,E)`, `ic_probs (E,)`, lazy adjacency/degree caches), `TaskSpec` (budget/horizon/dynamics/`allowed_ops`), `Trajectory`, `Strategy` protocol, `ActionFn` |
| `agent.py`    | `LLMProvider` protocol, `GatewayProvider` (lab gateway), `extract_code_block`, `CodingAgent`                                                                                                             |
| `prompts.py`  | system prompts per method (output contract + reply skeleton), user prompt (task/graph/API), feedback prompt                                                                                              |
| `executor.py` | `build_strategy` (exec in controlled namespace), `call_strategy` (runtime-error conversion), `validate_actions`, `StrategyError`                                                                         |
| `methods/`    | `base.py` (evaluate, paired re-score, acceptance, anchors, feedback), `one_shot.py`, `per_step.py`, `windowed.py`, `evolve.py` (also `adaptive`): all implement `OuterLoopMethod.optimize(agent, env, task, graph) → (Strategy, Trajectory)` |
| `envs/`       | `MonteCarloEnvironment` (NDlib ground truth), `WorldModelEnvironment` (trained GNN simulator, or the `structured_oracle` head), `MultiRoundEnvironment` (wraps either for `--campaigns`) behind one `rollout(action_fn, horizon, budget, seed)` interface |
| `tools/`      | `primitives.py` (23 classical-IM building blocks, 16 of them advertised to the model), `algorithms.py` (36 named algorithms), the per-task pools (`adaptive_`, `dismantling_`, `blocking_`, `immunization_`, `localization_`, `reconstruction_`, `prediction_algorithms.py`), `library_api.py` (auto-generated prompt reference) |
| `credit.py`   | per-action counterfactual credit (batched on the world-model env), solo-cascade stagnation timing, learned-kernel bottlenecks                                                                                                                                            |
| `run.py`      | experiment driver: config, provider selection, baselines, compare replay, results JSON                                                                                                                   |

The single coupling point between methods and environments is the **`ActionFn` closure**: `(State, timestep) → list[ActionOp]`. Environments never know whether actions come from a fixed plan, a live policy, or an LLM call, which is exactly what makes the world model a drop-in replacement for the simulator.

---

## 3. The methods (`--method`)

The first three methods are two extremes plus a middle point on a cost/adaptivity spectrum; `evolve` (the pipeline default) and `adaptive` are the population searches described after them. They differ **only** in who produces the `t > 0` actions and how often the LLM is invoked; the environment contract is identical.

One later addition sits outside that spectrum: `adaptive` (method 5) is the same population search over a per-round policy.

### Method 1: `one_shot`: one super-algorithm emits the whole horizon

The agent designs a single algorithm whose `plan_horizon(graph, budget, horizon)` returns the entire plan `R_{t₀…T}` up front. The environment rolls the whole plan; the final reward (plus a trajectory summary, plus optional per-action credit, plus the full traceback if the script failed) is appended to the next prompt, and the agent **revises the algorithm**: up to `--outer-iters` times, keeping the best-by-reward candidate.

- Inner execution is an **open-loop plan** (committed before seeing the realized stochastic trajectory); the outer skill-refinement loop is closed.
- LLM cost: `outer_iters` calls total. Rollout cost per candidate is one ensemble.
- Implementation detail: `plan_horizon` is invoked through `call_strategy`, so a runtime crash _inside generated code_ becomes a `StrategyError` whose traceback is fed back as a repair turn rather than killing the run. Every bag is validated (op legality vs `allowed_ops`, node range, per-bag and whole-plan seed budget).
- This is the highest-novelty method (the "super-algorithm") and the feasibility probe: can a program anticipate the full horizon?

### Method 2: `per_step`: re-prompt the agent at every timestep

The LLM **is** the policy: at each timestep the current state (full infected + frontier lists) is appended to the prompt, the agent writes a fresh script, its `act(state, graph, timestep)` is executed for that step only.

- Maximally adaptive: every action conditions on the realized state.
- **Prohibitively expensive**: the LLM is called once per _(ensemble sample × timestep)_. With `n_samples=20` and horizon 10 that is up to ~200 coding-agent calls per evaluation. Treat this method as the adaptivity **upper bound / cost reference**, not the practical operating point. Every call is logged (`[per_step] LLM call k (t=…, |infected|=…)`) because that count is the cost.
- The budget is **per episode, not per bag**. Without that the policy could legally emit `budget` seeds at every one of `horizon+1` timesteps and play the same nominal k as a `one_shot` arm with 11× the seeds. The counter resets at `t=0`, which is exactly right under a sequential env (`MonteCarloEnvironment`); under a lockstep ensemble env every sample's `t=0` also resets, so from `t>0` the cap is shared across samples: stricter than per-sample, never looser. Each turn's prompt states the remaining budget and which nodes are already seeded.
- Caveats: there is no repair loop (a `StrategyError` mid-rollout aborts the method), and the archived `script` in the results JSON is the _last_ generated script (one exists per call).

### Method 3: `windowed`: one online algorithm, re-applied per time window

The agent designs the algorithm **once**; the timeline is split into `--windows` stages (`window_length = max(1, (horizon+1) // windows)`), and the algorithm's `act(state, graph, window_index)` is consulted only at window boundaries: solving a fresh sub-problem on the current state each window (classical algorithms may be reused per window; the prompt says so explicitly). Budget applies **per window call**.

> This is the one method whose effective budget is not `--budget`. The multiplier is the number of boundaries, not `--windows`: at horizon 10 with `--windows 3` the window length rounds to 3 and `t=0,3,6,9` all fire, so a nominal `k` plays `4k` seeds. The results JSON records the real number as `effective_budget` so the sweep table is not silently read as an equal-budget comparison against `one_shot`, check that column before ranking a windowed arm.

- One LLM call, closed-loop at window granularity: cheaper than Method 2, more reactive than Method 1.
- The simplest method to stand up ("apply the old IM algorithm, staged") and it reframes the objective from a single terminal target into per-window objectives with state rolled forward between stages.

### Method 4: `evolve`: population search over algorithm edits (EvoX-lite)

Every generation after the first applies ONE operator to the population. Six operators, drawn with weights by a budget-aware schedule (`methods/evolve.py::choose_operator`): the exploit side edits the population best (`refine`: one targeted change; `parameters`: numeric constants only; `simplify`: remove what the diagnostics do not justify, must be shorter), the explore side makes a new mechanism (`crossover`: the best with a rank-weighted partner; `synthesize`: one strategy from the best three; `from_scratch`: a new program around an idea chosen by an idea search, several mechanisms proposed and judged against the library and the population in one call). The exploration mass is MCTS-AHD's schedule, `0.8 x (remaining generations / total)` plus `0.15` per stalled generation, so restructuring happens early and refinement late. A candidate replaces the incumbent only when its paired delta clears the larger standard error of the two rollouts (`methods/base.py::accepts`); a `simplify` child may tie if it is shorter, which is the parsimony pressure toward algorithms a reader can follow. Every prompt carries an attempts table (every generation: operator, reward, delta, accepted, mechanism, hint) and a population table (reward, compute, size, mechanism), plus a running memory of design rules: after each scored generation one short reflection call on the (worse, better) pair returns a hint under twenty words and the memory revised under fifty (ReEvo's two reflections in one call). Every script opens with `# MECHANISM: <one sentence>`, which is what the tables, the idea judge and the write-up read. The thread keeps a diff against the parent plus a one-line verdict per attempt rather than the whole script (`Conversation.compact_last`), so the window (now ten exchanges) holds most of a search. Stagnation still counts, and at `stagnation_patience` it forces an explore operator, as restructure used to fire.

### Method 5, `adaptive`: the same population search over a per-round policy

For **adaptive influence maximization** (`--task adaptive_online_im`). Structurally identical to `evolve` (same population, same six operators, same checkpoints), differing in exactly one thing: what the generated program is, and therefore how it is evaluated. `methods/base.py::evaluate_strategy` is the single fork.

|                | `evolve`                                              | `adaptive`                                                        |
| -------------- | ----------------------------------------------------- | ------------------------------------------------------------------ |
| the program is | `plan_horizon(graph, budget, horizon)`, decided up front | `act(state, graph, timestep)`, called once per round               |
| the budget is  | `k` seeds, all placed in the plan                     | `k` seeds split into `r` batches summing to `k`                    |
| what it sees   | the graph                                             | the graph **and** the state the previous batch's diffusion produced |

- **`--rounds r`** splits `k` into `r` near-equal batches (Han et al.'s k-sweep). **`--per-round-budget b`** instead fixes `b` and derives `r = ⌈k/b⌉` (their b-sweep). Batches always sum to exactly `k`, and the cap is enforced per round, so no cross-call counter is needed, which matters because `MonteCarloEnvironment` loops (episode, timestep) while `WorldModelEnvironment` loops (timestep, sample), and any accumulated state would mean different things under the two.
- **`--round-gap g`** puts `g` timesteps of diffusion between rounds. A schedule whose last batch would land past `--horizon` is **rejected**, not clipped: silently dropping a batch would spend less than `k` and report it as `k`.
- **`--feedback-model {full_adoption,myopic}`** decides what `act()` reads. `myopic` blanks `state.infected`, leaving only the wave activated since the last step. That has a consequence worth knowing: a myopic policy cannot tell that a candidate is already active, so it cannot filter one out. Erroring would make the arm unrunnable, and letting the op through would be worse (`add_node` writes NDlib status `1` over status `2`, re-arming a spent IC spreader), so the seed is **dropped** and the slot is spent for nothing. Under `full_adoption` the same proposal raises, because the policy was handed the set it failed to filter. Blind re-seeding costing budget *is* the price of the weaker observation.
- **One policy copy per possible world.** A generated `act()` routinely keeps state across calls ("what have I already chosen"). `WorldModelEnvironment` queries the policy once per ensemble member per round, interleaved, so one shared object saw 50 different histories as one: sample 1's second batch excluded sample 0's picks, and by sample 49 every member had committed a different seed set. Netscience at `k=318`: 506 in the loop, 724 on the referee. `State.sample` (never serialized) names the member, and `rounds.adaptive_action_fn` routes each one to its own shallow copy of the policy (data attributes deep-copied, the bound primitives shared). `check_adaptive.a_stateful_policy_gets_one_copy_per_sample` fails on the old code.
- **Adaptive arms are capped at `n_samples = 50`** (`pipeline.run.adaptive_n_samples`) whatever `--n-samples` says: `act()` runs once per ensemble member per round, so the policy's own compute scales linearly with the sample count, where every other arm's evaluation is one batched forward pass. At the ladder default of 200 an uncapped adaptive evaluation would take half an hour.
- **The adaptivity gap needs both arms.** `spread(adaptive) / spread(non-adaptive)` at matched `k` is only meaningful with a control, so the task registry's `default_arms` pair every `adaptive_<mode>@E` with `evolve_<mode>@E` for all four evaluators, and `pipeline/conditions.py::adaptivity_gaps` divides them on the shared ground-truth replay. Do not expect a gap above 1: theory caps the myopic gap at 4 and proves non-adaptive greedy is no worse across all graphs. **The claim is cost**: `evaluator_seconds` for both sides sits in the same table.

Results carry `rounds`, `round_batches`, `round_gap`, `feedback_model` and `round_spreads` (spread after each round), because the adaptive-IM literature splits three ways on the budget convention and a spread number is not comparable without `(k, b, r)` together.

### Baselines for `adaptive` (condition 1)

`--baselines` normally names a member of `tools/algorithms.py`, every one of which returns a static seed set. Those are the *control* side of the adaptivity gap, not adaptive baselines. `tools/adaptive_algorithms.py` holds the per-round policies, signature `(state, graph, batch, dynamics) -> seeds`:

| name | what it is |
| --- | --- |
| `adapt_greedy` | **AdaptGreedy** (Golovin & Krause 2011; Han et al. PVLDB 2018): greedy marginal gain re-estimated each round against the realized state |
| `adapt_epic` | **EPIC** (Han et al. 2018): AdaptGreedy with RIS per batch; RR sets the active set already covers are dropped first |
| `adapt_degree_discount` | DegreeDiscount over susceptibles, discounted by what the cascade already reached |
| `adapt_degree`, `adapt_pagerank`, `adapt_random` | the per-round heuristic floor |
| `static_split` | one static ranking dealt a batch per round: the same-machinery control, so the timing penalty is separated from the adaptivity benefit |

`parse_arm` marks these `method="adaptive"` from the name alone, so `--baselines adapt_greedy` runs down the round path and lands in the adaptivity-gap table on the adaptive side. They are also in the executor namespace and in an adaptive prompt's API reference, so a generated policy can call or extend one instead of reinventing AdaptGreedy.

**Budget the greedy one.** `adapt_greedy` costs `batch x candidates x mc_runs` simulations per round. Measured on an 80-node graph at `k=8, r=3`: 70.5 evaluator-seconds against `adapt_epic`'s 0.3. That is the published reason EPIC exists, and the axis the world model is meant to flatten.

### Containment: `critical_node_detection` (condition 1 and the sign flip)

The one task family where the outer loop MINIMIZES. Everything else is shared (same simulator, same features, same heads, same six conditions) and four things differ:

1. **The sign.** `pipeline.tasks.Task.objective` is the single source of it. `coding_agent.types.improves` / `best_by` / `rank_by` route every "is this better" in `evolve`, `one_shot`, `baseline_anchor`, `paired_delta`, the report, the plots and the summary, so a maximize-shaped harness cannot silently name the worst arm the winner. `paired_delta` also flips the verdict text, or the model would be coached to undo every improvement it makes.
2. **The outbreak.** The planner does not start the cascade. `coding_agent/containment.py` selects `--outbreak-pct` of `N` by `--outbreak-selector`, deterministically in `--seed`, and injects it at `t=0` through an action-fn wrapper shaped exactly like `stream.wrap`. Applied **after** validation, because the sources' `add_node` ops are exogenous and `--allowed-ops remove_node` would reject them. Every path that builds an `ActionFn` goes through `methods.base.wrap_exogenous`, so an anchor or a `per_step` arm cannot accidentally face no outbreak and post an unbeatable zero.
3. **The budget buys removals.** `validate_actions` counts `task.budget_op`, and the planner emits a **bare** `remove_node`; the harness expands it into the deletion bag. Both structured heads document that a blocked node's edges are gone from `edge_index` and rely on it, so the expansion is required rather than tidy. Letting the planner emit the edge ops itself would be an unbudgeted second intervention, so it is rejected with a message that says so.
4. **Outbreak sources cannot be removed.** Deleting patient zero ends the outbreak instead of containing it: that is `source_localization`, a different task. Without the rule a uniform random removal set beats every dismantler whenever it happens to include a source, which is what the first end-to-end run actually produced.
5. **Generated programs are offline; the plan oracle is gone.** `self.score_plan(plan)` used to bind the arm's own evaluator on the generated program for the three exogenous-cascade tasks. It was removed outright on 2026-09-04: an algorithm the coding agent writes may not call the world model or any evaluator at runtime, so what it returns has to be computable from the graph, the edge probabilities and the outbreak alone. The arm's evaluator still scores the plan and writes the feedback and the probe answers, which is where conditions 3-6 differ.
6. **The planner is TOLD the outbreak, and the budget must sit below its ring.** This makes the task reactive containment rather than the literature's blind dismantling: every published dismantler returns the same set for any outbreak, the agent reads the sources off its prompt. A budget at or above the outbreak's one-hop ring `|N_1(S) \ S|` is then trivial for any outbreak-aware arm (delete the ring, the cascade cannot leave `S`, the row reads exactly `|S|`). The first power_grid sweep at a 1% outbreak had the ring at 129 nodes and three of four budgets above it, and the agent's 49.00 against 98-160 for every dismantler was that asymmetry, not a method. Three things follow: the default `outbreak_pct` is **10%** so the 1/5/10/20% sweep sits under the ring on the sparse benchmarks; `frontier_removal` (the ring, highest degree first, the same rule as `frontier_immunization`) is in the pool and the task defaults as the outbreak-aware control the agent has to beat; and every result carries `outbreak_ring` / `ring_fits`, which the report turns into a note naming the trivial budgets.

`coding_agent/check_containment.py` asserts all six on graphs where the answer is known by hand. Run it after touching any of them.

### Baselines for `critical_node_detection` (condition 1)

`tools/dismantling_algorithms.py`, signature `(graph, budget, diffusion_model, **kw) -> list[int]`, returning a **removal** set:

| name | what it is |
| --- | --- |
| `adaptive_degree` | **HDA**: remove the highest-degree node, recompute, repeat. **The row that has to be beaten**, MIND's Table 5 puts it at 119.9 against FINDER's 115.0 across 47 networks |
| `iterative_betweenness`, `approx_iterative_betweenness` | **BI / ABI** (Wandelt et al. 2018), best in 70-80% of their cases and almost never reported by a learned paper |
| `collective_influence_removal`, `collective_influence_r` | **CI** (Morone & Makse, Nature 2015): optimal percolation, adaptive removal. The paper's CI is the `_r` one, adaptive removal **plus** greedy reinsertion |
| `corehd`, `corehd_r` | **CoreHD** (Zdeborová et al. 2016): 2-core + highest degree |
| `bpd`, `bpd_r` | **BPD** (Mugisha & Zhou, PRE 2016): belief propagation over the feedback-vertex-set spin model at `x = 12`, decimated, then tree breaking. The inference version of what CoreHD does with degree alone |
| `decycling`, `decycling_r` | the same two stages with a GREEDY stage 1: the control that isolates what BPD's message passing buys |
| `articulation_removal` | cut vertices, ranked by the split they cause |
| `explosive_immunization` | **EI** (Clusella et al. 2016): inverse Achlioptas construction |
| `gnd`, `gndr`, `egnd` | **GND** (Ren et al. PNAS 2019): spectral bisection + weighted vertex cover, recursed. `egnd` sweeps the Fiedler split point and keeps the best cut |
| `netshield` | **NetShield** (Tong et al. 2010): greedy eigenvalue drop, the only member whose objective is epidemic rather than structural |
| `kshell_removal`, `pagerank_removal`, `degree_removal`, `betweenness_removal` | one-pass centrality controls |
| `frontier_removal` | the outbreak's one-hop ring, highest degree first: the only member that conditions on WHERE the outbreak is, and the control every agent arm has to beat. At a budget above the ring it contains everything |
| `acquaintance_immunization` | Cohen et al. 2003: pick a random node, immunize a random neighbour. The zero-knowledge floor |
| `random_removal` | the trivial floor |
| `greedy_blocking` | CELF with the sign flipped: add the node whose deletion most reduces the SIMULATED spread. The honest strong bar, and unaffordable, which is the whole argument for `f_θ` |

`greedy_blocking` is in `mc_dismantling_algorithms` and blocked from generated scripts unless `--allow-mc-algorithms`, exactly as `celf` is and for the same reason: it simulates once per candidate per pick, so its episodes never reach `MonteCarloEnvironment.episodes_used`.

**On the `_r` variants.** §8.2 trap 2: `X` and `X+R` are different methods routinely cited under one name, so both are registered. The published reinsertion pass is defined only once the graph is dismantled below `threshold * N`, which a fixed budget rarely reaches, under that bar every `_r` variant was a bit-for-bit copy of its base (measured on BA-500 at k=15%: giant component 21, bar 5). `_reinsert` therefore holds the ACHIEVED giant component when the absolute bar is out of reach, and `_reinsert_and_refill` puts the freed budget back through tree breaking. That is the paper's rule wherever the paper's rule applies, and the same idea extended to the budgets we run; the docstrings say so.

**Min-Sum is deliberately absent.** It was implemented from the published equations and measured to be wrong: erratic in its own parameters, and worse than the greedy `decycling` it should improve on. `abraunst/decycler` is the registered external route. The measurements are in [`research/critical_node_detection.md`](../research/critical_node_detection.md) §11 so the next attempt does not start from zero.

**Expect the ranking to disagree with the structural columns.** On the smoke SBM, `articulation_removal` was the best dismantler by giant-component drop and the *worst* by contained spread. That is [`research/critical_node_detection.md`](../research/critical_node_detection.md) §5.8 reproducing itself, not a bug.

### Inversion: `source_localization` (a different contract, not a different sign)

The one task family where the outer loop writes an **inference algorithm** instead of choosing an intervention, and the only place in this package where nothing is emitted at all. Given a graph and an OBSERVED diffusion state `y`, recover the seed set `x` that produced it, scored by a label-free consistency reward with F1 against the truth reported after the search. Six things differ:

1. **The contract is `localize`, not `plan_horizon`.** `def localize(self, graph, observation, budget) -> list[int]`, plus an optional `source_scores(graph, observation) -> np.ndarray`. `methods.base.evaluate_strategy` dispatches on `TaskSpec.recovers` and calls `localization.evaluate_localizer`, which runs the program once per labelled episode instead of rolling anything out. The contract is chosen by the TASK, not the method: `evolve` is still a population search, and what the population contains is a localizer.
2. **The forward oracle is one name with four bindings, on the HARNESS side.** `localization.ForwardOracle` resolves to: a raiser under `@native` (no forward model, by design), NDlib under `@monte_carlo`, the exact batched simulator under `@oracle`, and `f_θ` under `@world_model`. It computes the consistency reward and the residual feedback, and that is what makes conditions 3-6 an ablation on ONE variable: the program is byte-identical across them and OFFLINE (since 2026-09-04 it is not bound on generated strategies at all; only a canned library baseline that needs a kernel, such as `resim_greedy`, receives it). Every scoring rollout routes through `environment.rollout`, so it lands in the cost accounting.
3. **The reward is label-free consistency.** For every selection episode the harness rolls the recovered set forward through the arm's own evaluator (the same metered rollout the program's oracle uses, counted apart from it) and scores minus the mean squared error against the observed state: 0 is perfect, higher is better, and it is computable at deployment where no label exists. That is the same shape as an intervention task's reward, so conditions 3-6 are a ladder in exactly the IM sense (one real episode, Monte Carlo, the analytic form, `f_θ`). Diffusion is many-to-one, so a set that reproduces `y` need not be the true set; that identifiability is measured rather than assumed.
4. **The true sources are read only after the search.** `localization_label_metrics` computes F1, precision, recall and AUC against the stored sources for the winner on both splits, after the closing write-up, and they are reported beside the reward and never fed to it; `f1_generalization_gap` rides beside `generalization_gap`. The feedback the agent sees is the residual: the nodes its sources over-explain and under-explain when rolled forward. the referee re-measures the reward on its own simulator (`referee_reward`), exactly as a spread is replayed.
5. **Two disjoint episode pools.** The search optimizes on `--sl-select-split` and the winner is re-run unmodified on `--sl-eval-split`; the second is what every table reports and `generalization_gap` is the difference. Selecting and reporting on the same episodes cannot distinguish an algorithm from a memorized set of cascades. `--sl-transfer-from <another run's results JSON>` runs a program selected on a different GRAPH, which is the headline comparison and one no per-instance method can enter.
6. **`k` is a property of the instance.** The localizer is told how many sources to name, so `--sl-budget-mode episode` makes `PR = RE = F1` by construction and the task's budget sweep is a single point. `--sl-budget-mode sweep` forces the pipeline's `k` and filters the pool to that source fraction; an empty band raises rather than silently widening.

`coding_agent/check_source_localization.py` asserts all six on graphs where the answer is known by hand, plus the metric definitions, the label extraction and that no label reaches the loop before the post-search merge. Run it after touching any of them.

### Baselines for `source_localization` (condition 1)

`tools/localization_algorithms.py`, signature `(graph, observation, budget, **kw) -> list[int]`, returning a **source** set. Each name also has a paired scorer in `localization_scorers` returning a per-node vector: F1 scores the SET, AUC scores the RANKING, and for the sequential members those are deliberately not top-k of one another.

| name | what it is |
| --- | --- |
| `lpsi` | **LPSI** (Wang et al. AAAI 2017): label the infected `+1` and the uninfected `−1`, propagate to convergence, take the LOCAL MAXIMA. **The row that has to be beaten**, SIDSL's Table 1 puts it at F1 0.544 on Digg against SL-VAE's 0.479 and DDMSL's 0.517, with no learning in it at all |
| `netsleuth` | **NETSLEUTH** (Prakash et al. ICDM 2012): eigenvector of the infected subgraph's submatrix Laplacian, seeds picked by deflation |
| `ojc` | **OJC** (Zhu et al. AAAI 2017): cover the observed infected nodes with balls, then take the Jordan centre of the cover. Built for partial observation, so its candidates include unobserved neighbours |
| `jordan_center` | the eccentricity minimizer of the infected subgraph (Zhu & Ying, ToN 2016): one centre per infected COMPONENT, since components cannot share a source |
| `rumor_centrality` | **Shah & Zaman (2010)**, the paper that founded the field: count spreading orders, in logs, over a BFS tree per component. SINGLE-source, see the note below |
| `dynamic_age` | Fioriti & Chinnici (2012): drop in the infected subgraph's leading eigenvalue when `v` is removed |
| `effective_distance` | Brockmann & Helbing (Science 2013): in `d(u→v) = 1 − log p(u→v)` a contagion becomes a circular wave, so the origin is where every infected node sits at a similar effective distance |
| `dmp_localize` | **DMP** (Lokhov et al. 2014): deterministic mean-field forward per hypothesis, greedy MAP over the observation's likelihood. Sequential, because a second source is only worth adding where the first one's cascade fails to explain `y` |
| `infected_degree`, `infected_betweenness`, `infected_closeness`, `infected_eigenvector` | the **Comin, Costa** suite restricted to the infected subgraph: the cheap-heuristic floor everything is compared against |
| `random_sources` | the floor |
| `resim_greedy` | greedy minimization of `‖y − f(x̂)‖²`. The forward-model-using classical baseline |

`resim_greedy` is in `mc_localization_algorithms` and blocked from generated scripts unless `--allow-mc-algorithms`, exactly as `celf` and `greedy_blocking` are: called without a `predict` argument it falls back to a PRIVATE NDlib estimator whose episodes never reach `MonteCarloEnvironment.episodes_used`. Nothing is lost by blocking it: a generated program is offline and never calls an oracle.

**Know what LPSI actually promises**, because assuming more is how a wrong implementation passes a test. It names local maxima of a converged label field, which is a *centre*-of-the-infected-region estimator: on a path `0-1-2` with all three infected it names node 1, even though the cascade started at an end. That is not a bug and it is not fixable inside LPSI, it is §1's ill-posedness showing up on a seven-node graph, and it is why §2.9 risk 3 says the achievable ceiling is uncharacterized. What LPSI does guarantee is that it never names a node the observation says was uninfected, and both properties are asserted in the self-check.

**`rumor_centrality` and `jordan_center` are single-source estimators** evaluated under a multi-source protocol, so they score near zero at `k = 10%` of `N` **by construction**: a property of the protocol, not of the method (§8.2: the two literatures never mix their numbers). They are registered because they founded the field and because a `--budgets 1` run makes them admissible.

### Decoding: `cascade_reconstruction` (the same family, a bigger object)

The second `recover` task, and the one that splits the family the way `influence_blocking` splits containment: both invert something and neither emits an action, but a localizer names a SET of `|V|` candidates while a decoder produces a coherent `T × |V|` trajectory. Five things differ from the localization path above, and `TaskSpec.decodes` routes all of them.

1. **The contract is `reconstruct`.** `def reconstruct(self, graph, observation, horizon) -> dict[int, tuple[int, int | None]]`, mapping each node believed infected to `(activation timestep, inferred parent)` with `parent = None` marking a source and uninfected nodes simply absent. `methods.base.evaluate_strategy` checks `decodes` before `recovers` and calls `reconstruction.evaluate_reconstructor`. **The recovered sources fall out for free** (the subset with no parent IS the seed-set estimate) so source localization is the projection of this task rather than a sibling of it, and `source_f1` is reported on every row without extra machinery.

2. **The primitive is the raw KERNEL, not a seed-set forward pass, and it lives in the harness.** `reconstruction.StepOracle` returns `P(v activates next step)` for an ARBITRARY proposed state, which a rollout cannot give: NDlib only ever advances forward from what it holds, so the sampling binding goes through a new `Simulator.set_state` that writes a hypothesis the simulator never visited. `transition_logprob` derives from the same call rather than being a second oracle. The harness scores the decoded history with it; the generated decoder is offline (since 2026-09-04) and works from `graph.ic_probs`, the reports and their times, so an analytic one-step IC rule inside the program is the natural way to run a likelihood-driven local search.

3. **The inner loop runs two orders of magnitude hotter.** ~10⁴ kernel calls per instance against ~10², so `kernel_calls_per_instance` is a first-class column and 4-vs-6 is the sharpest cost comparison in the repo. "Arm 4 does not complete at a useful proposal count" is a reportable finding here, not a missing row.

4. **The reward is the history's likelihood under the arm's own kernel, label-free.** `reconstruction.score_history` sums a `1/N` source prior per declared source (without it, declaring every report a source explains anything without asserting a transmission) and the kernel's log-probability of every transition the decoded history asserts (plus one terminal step asserting nothing more activated), divides by the node count, and subtracts the fraction of the observation the history contradicts: reported nodes dropped, reported times moved, nodes named that a final snapshot says stayed clean. An implausible parent is exactly what the likelihood punishes, which is how §2.6's warning (the node half is nearly free, so a search scored on it stops attempting the tree) is answered without a label. Path precision, event F1, timing NRMSE and the tree-weighted `--cr-tree-weight` score against the stored history are computed by `reconstruction_label_metrics` for the winner on both splits AFTER the search and reported beside the reward, never fed to it. Every results JSON carries `trivial_decoder_reward` and `trivial_decoder_tree_score` (everyone reachable, parents by BFS): a reward a trivial decoder can reach is a wrong reward. **PathPrecision is a precision**, so naming three edges and getting them right scores 1.0 on it; `path_recall`, `jaccard` and `n_tree_edges` are reported beside it.

5. **It needed a simulator change nothing else does.** NDlib emits no transmission edge at all, so `--trace-parents` swaps in `TracedICModel` / `TracedThresholdModel` and every record gains a `parents` field. Under IC that is one parent and it is *the first successful `u` in node order*, which is a real and documented bias; under LT there is no transmission edge at all and the value is the whole active in-neighbourhood, which makes LT's path precision structurally easier and means the two dynamics are never compared. `load_cascades` RAISES on a dataset generated without it rather than silently falling back to the node half.

`--cr-setting` is a protocol rather than a knob and its four values are four experiments whose rows are never pooled: `partial_times`, `partial_nodes`, `final_snapshot` (DITTO's DASH) and `hidden_nodes`. `--cr-observation-rate` is the probability a node **is reported**, spelled out because two papers in this literature use `σ` for opposite quantities. `coding_agent/check_cascade_reconstruction.py` asserts all of it in 26 checks, including that the traced models match the untraced ones in distribution, a traced model that changed the dynamics would invalidate every episode.

### Compartmental: `epidemic_control` (the family the heads were not built for)

The one runnable task whose **dynamics are not monotone**: nodes recover and stop transmitting, and under SIS become susceptible again. Everything else in this repo keeps the assumption `ICTransmissionHead` rests on, so `research/epidemic_control.md` §2.6 calls this the task that tests whether "structured head" generalizes past IC.

The contract is `plan_horizon` as usual (this is an intervention task, not an inverse one) and what changes is **what a unit of budget buys**. `--epi-lever` picks one of §2.5's four interventions and sets `budget_op` and `allowed_ops` together:

| lever | op | what the harness does with a bare `remove_node` | papers |
| --- | --- | --- | --- |
| `vaccinate` | `remove_node` | expands to the node PLUS its incident arcs: immune, uncounted, never infectable | Pastor-Satorras & Vespignani PRE'02, Cohen PRL'03, NetShield ICDM'10, DAVA SDM'14 |
| `quarantine` | `remove_node` | expands to the incident arcs ALONE: isolated, but still in the graph and still counted | Holme PRE'02, RLGN ICML'21 |
| `edge_cut` | `remove_edge` | nothing; the arc is the intervention | Kimura TKDD'09, Van Mieghem PRE'11, NetMelt CIKM'12 |
| `contact_reduce` | `set_edge_weight` | nothing; `--contact-reduction r` scales the arc to `r * beta_uv` | Fractional Immunization SDM'13, Preciado TCNS'14, DURLECA KDD'20 |

`vaccinate` versus `quarantine` is the sharpest pair and the cheapest to read: the SAME node set is chosen under both and they differ only in whether the dosed node leaves the attack rate. That is §8.2 trap 7's "recovered is not removed" made into a lever, and it is why papers reporting "nodes saved" against "no intervention" and against "random vaccination" differ by a large constant. `contact_reduce` is expressible at all only because we wrote our own stepper, NDlib's compartmental models carry no per-arc parameter.

Two rows of §2.5 are deliberately out of scope with a stated reason. **Contact tracing is not an action**: it changes the OBSERVATION, not the graph or the state, and belongs in a POMDP observation model we do not have. **Quarantine with a duration** is not expressible either: a release timer is hidden state and the compartment head is Markov in `(state, action)`.

The reported block is prevented infections plus the outbreak's SHAPE (peak prevalence, time to peak, AUC, endemic prevalence) because §8.2 trap 4 is that a good policy flattens rather than eliminates, so a terminal-state number alone can rank two policies backwards. The eigendrop rides along as CONTEXT and never as the score: §8.2 trap 1 is that a method can win it and lose the attack rate, which is exactly what `netshield` and `dava` do to each other depending on the graph. `coding_agent/check_epidemic_control.py` asserts all of it in 78 checks, including that the oracle head reproduces the simulator's own one-step marginals under all three dynamics.

### Baselines for `epidemic_control` (condition 1)

`tools/immunization_algorithms.py`, signature `(graph, budget, diffusion_model, **kw) -> list[int] | list[tuple]`, returning node ids under the two node levers and `(u, v)` arcs under the two edge ones. Twenty-three members across all three lines of `research/epidemic_control.md` §3, because they optimize **different objectives** and a table with only one line cannot see its own blind spot.

| name | line | what it is |
| --- | --- | --- |
| `degree_immunization` | physics | **Pastor-Satorras & Vespignani PRE'02**: top-k degree, computed once. **The row that has to be beaten**, RLGN's own Table 2 has Degree and Eigenvector tying to within 0.1 on two of five graphs |
| `adaptive_degree_immunization` | physics | **Holme PRE'02**'s `RD` arm: recompute after every dose. Not interconvertible with the static version |
| `acquaintance_immunization` | physics | **Cohen PRL'03**: pick a random node, dose a random NEIGHBOUR. No global information at all, and §8.3 names it as the row that most embarrasses learned methods on sparse graphs |
| `eigenvector_immunization` | physics | NetShield's `Eigs` row: the top-k that NetShield's collective selection is supposed to beat |
| `pagerank_immunization` / `betweenness_immunization` / `kshell_immunization` | physics | the centrality controls; k-shell is Kitsak's "influential spreaders are in the core, not the hubs" |
| `random_immunization` | physics | **not a throwaway floor**: the GAP between it and degree is the 2002 result that founded this field |
| `netshield` | spectral | **Tong ICDM'10**: greedy on the submodular Shield-value. Its approximation to the true eigendrop is 0.977-1.000 in its own Table 3 |
| `netshield_plus` | spectral | **TKDE'15**: recompute the eigenvector every `b` doses. A separate row because `X` and `X+` get cited under one name |
| `greedy_walk` | spectral | **Saha SDM'15** (SRMN): closed `k`-walks through a node, recomputed. Reimplemented from prose, the authors' code is MATLAB |
| `preciado_allocation` | spectral | **Preciado TCNS'14**'s geometric program, discretized to a top-k. Labelled a discretization, not the method |
| `netmelt` | spectral (edge) | **Tong CIKM'12**: score arc `(i, j)` by `u(i)·u(j)`. The canonical edge baseline |
| `product_degree` / `eigen_score` | spectral (edge) | **Van Mieghem PRE'11**'s two heuristics. Algebraically the same rule under two names, which is a fact worth reading off a table |
| `greedy_walk_edge` | spectral (edge) | GreedyWalk's SRME variant, adaptive |
| `edge_betweenness_cut` | spectral (edge) | the bridge control |
| `dava` / `dava_fast` | data-aware | **Zhang & Prakash SDM'14**: merge the observed infected set into a superseed, build a dominator tree, cut the nodes all paths must pass through. **The row this task is positioned against** (§9.4) |
| `frontier_immunization` / `frontier_edge_cut` | data-aware | dose the susceptible boundary of the observed outbreak. The control DAVA has to beat to have contributed anything |
| `mc_greedy_immunization` | simulation | greedy on the simulated attack rate. NP-hard and NOT submodular here, so no `(1 - 1/e)` bound: a strong heuristic, not a ceiling with a proof |
| `random_edge_cut` |   | the edge levers' floor |

`mc_greedy_immunization` is blocked from generated scripts by default for the same reason `celf` and `greedy_blocking` are: it re-simulates on a private simulator and bypasses the metered evaluator.

### Baselines for `cascade_reconstruction` (condition 1)

`tools/reconstruction_algorithms.py`, signature `(graph, observation, horizon, **kw) -> dict[int, tuple[int, int | None]]`, returning a whole **trajectory**.

| name | what it is |
| --- | --- |
| `delayed_bfs` | **Xiao SDM'18**: attach terminals in increasing observed time along the cheapest path from the tree so far, delaying the interior nodes to land between the endpoints. `O(m + k log k)`. **The row that has to be beaten**, no learning in it, node precision above 0.8 in its own paper, and inside the agent's expressible space |
| `ordered_steiner_closure` | Xiao's `closure`: metric closure over the terminals, MST, expand. `O(√k)` |
| `greedy_ordered` | Xiao's `greedy`: nearest-first attachment rather than earliest-first |
| `steiner_tree` | plain minimum Steiner tree, ignoring the observed ORDER: the control the three above beat on order accuracy |
| `tree_sampling` | **Xiao ICDM'18**: sample Steiner trees, read off per-node marginals. The only classical method here that outputs calibrated probabilities |
| `personalized_pagerank` | the assortativity trap. Xiao ICDM'18 found a plain random walker BEATS tree sampling on `grqc` (assortativity 0.164) and loses elsewhere: **run it on `ca_grqc` specifically**; if the search cannot clear it there the result is not real |
| `consistent_tree_wpct` / `consistent_tree_wbct` | **Zong ICDM'12**'s pair. WPCT constrains every node on a rooted path, WBCT only bounds the endpoint, and the gap between them is why both are here |
| `cult` | **Rozenshtein KDD'16**'s `α`-TempSteinerTree: a forest with `α` trading tree cost against root count. The only method that assumes NO propagation model, and therefore the honest ceiling for "what can you do without a kernel" |
| `dhrec` | **Sefer & Kingsford ICDM'14**: prize-collecting dominating-set vertex cover, greedily. DITTO's own MLE baseline |
| `cri` | **Chen TNSE'16**: cluster the infected subgraph, reverse-infect from each centre. DITTO's second |
| `netfill` | **Sundareisan SDM'15**: fill in a missing node when enough of its neighbourhood is infected, to convergence |
| `jordan_backward` | greedy forward decode from the Jordan centres: the cheap first cut |
| `observed_only` | **Rozenshtein's `Reports`**: report what you saw, infer nothing. Node precision 1.0 by construction, and the concrete answer to "is this reward gameable" |
| `one_hop` | Rozenshtein's second control: the reports plus their one-hop neighbourhood |
| `random_reconstruction` | the floor |
| `mcmc_decode` / `forward_backward` | the kernel-using pair: Metropolis-Hastings over histories, and forward-filter/backward-sample smoothing |

Every decoder shares one parent rule (`finalize`), deliberately: most published methods output a node set and a time rather than a tree, so a Path Precision difference between two rows is a difference in their **times** rather than in a tree-building trick one of them happens to have. A consequence worth knowing before reading the column: `order_accuracy` is 1.0 for the whole library pool by construction, because `finalize` only ever attaches an earlier-activating parent. It is a validity check on SYNTHESIS, not a quality measure across the classical pool.

`mcmc_decode` and `forward_backward` are in `mc_reconstruction_algorithms` and blocked from generated scripts unless `--allow-mc-algorithms`, for the `celf` reason plus one more: they cost `proposals × horizon` kernel evaluations per instance, and a generated program already HAS the metered kernel, writing the search around it itself is the only way that cost lands in the arm's own `kernel_calls`.

### Streaming graphs and multi-round campaigns

Two more §1 branches of the adaptive-IM literature, both of which apply to **every** arm rather than only the adaptive ones. An arm whose graph moved compared against one whose graph did not, or a union compared against a single campaign, measures the setting instead of the method.

- **`--edit-rate f`** (dynamic/streaming): a deterministic schedule of exogenous edge insertions and deletions, `f · |E|` of each per timestep. `coding_agent/stream.py`. The edits ride in the action bag, so neither environment changed: the MC env hands the bag to `Simulator.advance` and the WM env routes it through `apply_edge_ops`. They are appended *after* the policy's bag is validated, because they are exogenous and `--allowed-ops add_node` would otherwise reject them. An **adaptive** policy is handed `stream.graph_at(t)`, the graph as it now stands; a **static** plan was decided at `t=0` and cannot react. That gap is the point, not a bug. The schedule is a pure function of `(graph, seed, timestep)` because the two environments iterate in opposite orders and anything stateful would give them different histories.
- **`--campaigns r`** (multi-round): `r` separate diffusions of `budget` seeds, scored on the union. `MultiRoundEnvironment` wraps any evaluator, so it composes with all four. `infected` carries the union into the next campaign and `frontier` stays that campaign's own wave. The union is `P(v never activated) = ∏_i (1 − p_i(v))` over the per-campaign marginals: **exact** for a fixed schedule, not an approximation. the referee is wrapped too, or it would score a single campaign against the arm's union.

**`spread_curve`** lands in every result regardless: the ensemble-mean `E|infected|` after each timestep, padded to `horizon + 2` by holding the final value. That padding is exact rather than smoothing, because a rollout only breaks when the frontier **and** the bag are empty, which is a fixed point of monotone IC/LT. It is what makes `σ(S, T)` readable at any `T`; `infected_counts` is the representative sample and stops when its cascade died.

### `--strategy-mode`: what the agent is allowed to write

Orthogonal to the method choice (supported for `one_shot`, `evolve` and `adaptive`):

- **`free`** (default): the agent writes a whole `Strategy` program with the full library callable. Observed failure mode: portfolio composition, the agent runs many library algorithms and picks the winner, editing nothing.
- **`scored`**: the agent may only fill in the internals of an algorithm. A fixed harness (`ScoredStrategy.plan_horizon`, `types.py`) greedily picks the argmax of `score(node, selected, graph)` until budget is spent, then calls `schedule(seeds, graph, horizon)` (default: all at t=0). The executor rejects any override of `plan_horizon`; the exec namespace contains **no `algorithms` module** and no simulation primitives (`mc_simulate_spread`, `compute_marginal_gain`, `build_simulator` are hidden, otherwise the agent re-derives CELF instead of inventing structural scoring logic). The library appears in the prompt as an _ideas menu_ only, any borrowed idea must be written out inside `score()`, where it can be mutated. Canned/`--baseline` scripts always run in free mode regardless of the flag.

The intended experiment grid: `one_shot × scored` tests whether edit-level generation works at all; `evolve × scored` is the full population search over algorithm internals; `free` arms remain as the composition comparison.

### The trade-off spectrum

|                              | Method 1 (`one_shot`)                       | Method 3 (`windowed`)                   | Method 2 (`per_step`)                    | Method 4 (`evolve`)                            |
| ---------------------------- | ------------------------------------------- | --------------------------------------- | ---------------------------------------- | ---------------------------------------------- |
| LLM/agent calls              | `outer_iters` (refinement only)             | 1 (designed once)                       | one per sample × timestep                | `outer_iters` (population edits)               |
| Adaptivity to realized state | none (open-loop plan)                       | per-window                              | per-step                                 | none (open-loop plan)                          |
| Cost                         | low                                         | low, medium                              | very high                                | medium (many cheap WM rollouts)                |
| Novelty                      | highest (super-algorithm across time)       | moderate (classical algos, staged)      | adaptivity bound, not a practical method | highest with `scored` (evolves algo internals) |
| Loop type                    | open-loop plan + closed skill refinement    | closed-loop policy (window granularity) | closed-loop policy (LLM as policy)       | population evolution (parent + operator)       |
| Failure handling             | full repair loop (errors → revision prompt) | fail-fast                               | fail-fast                                | error fed to next generation                   |

---

## 4. The action space

Five operations, shared with the data generator and the world model (`data.wm_simulator.valid_action_ops`):

| op                | fields                         | IC effect                                          | LT effect                              |
| ----------------- | ------------------------------ | -------------------------------------------------- | -------------------------------------- |
| `add_node`        | `target`                       | activate node (transmits during this step)         | activate node                          |
| `remove_node`     | `target`                       | `--remove-semantics` decides, see below            | `--remove-semantics` decides           |
| `add_edge`        | `target→destination`, `weight` | add arc with transmission prob `weight`            | add edge structurally (weight ignored) |
| `remove_edge`     | `target→destination`           | remove arc                                         | remove edge                            |
| `set_edge_weight` | `target→destination`, `weight` | set transmission prob                              | no-op                                  |

**`--remove-semantics`** picks what `remove_node` does: `spent` (the default: stays counted, stops spreading, keeps its edges) or `blocked` (deleted from the graph, uncounted, cannot transmit or be infected, edges gone). The full table is in [`data/README.md`](../data/README.md). It is not just a simulator setting: `build_system_prompt` appends the matching rule to the system prompt, so the agent plans against the semantics it will actually be scored under, and with `--evaluator world_model` the run **refuses to start** unless it matches the checkpoint's, since the head's `T_exo` was fixed at training time. The rule is only appended when `remove_node` is in `--allowed-ops`.

**Budget semantics (and a known sharp edge):** `validate_actions` charges only `task.budget_op` against the budget (`add_node` for a seeding task, `remove_node` for a containment one), edge ops are otherwise **free**. Left unconstrained, capable models reliably discover the degenerate exploit: boost every frontier out-edge to ~1.0 and convert stochastic IC into deterministic percolation (observed: `mc_reward` 99.97/100). Two controls exist today:

- `--allowed-ops add_node remove_node`: restricts the ops a strategy may emit; enforced in the prompt (`any other op is REJECTED`) _and_ by `validate_actions`, whose rejection message feeds the repair loop.
- A per-op cost/budget model is the planned fix for making edge ops a fair, non-degenerate part of the game.

**Rules vs. coverage are independent knobs:** `--allowed-ops` sets the _game rules_ for the agent; the world model's training data sets its _coverage_. Train the evaluator on the superset of ops (supersets are safe; subsets bite, see §7).

---

## 5. The agent layer: from chat reply to validated program

### What the LLM sees (the complete context)

The system prompt (per method) carries: the role; the output contract (exactly one fenced ```python block containing optional imports and one `Strategy` subclass); the **import policy** (below); the injected-names documentation; the action rules including what `remove_node` actually does; the method's interface (`plan_horizon` vs `act`); a **horizon note** chosen from `allowed_ops` (below); and two **worked exemplars** at the level the search should start from. The user prompt carries: task and objective strings, `diffusion_model`, `budget` (absolute and as % of nodes), `horizon`, `allowed_ops`, the **graph profile** (below), the full auto-generated **library API reference** (every algorithm and primitive signature with its first docstring line, docstrings in `tools/` are literally prompt text), the **full source of three library algorithms**, and the closing instruction. Refinement turns append: previous reward, the paired delta with its noise verdict, trajectory summary, the **reference diff** (below), the incumbent script, the error traceback (repair path), and the credit report (`--credit`).

#### Imports (`executor.allowed_imports`)

Generated scripts **may** import `numpy`, `networkx`, `scipy`, `math`, `random`, `statistics`, `heapq`, `bisect`, `collections`, `itertools`, `functools`, and nothing else. `executor.check_script_imports` enforces this on the AST before the script is compiled, and also rejects `__import__`, `eval`, `exec`, `compile`, `open`, and `input` by name, without those the whitelist would be trivially bypassable and the check would be theatre. A rejection is a `StrategyError`, so it lands as a repair turn naming the allowed set.

The prompt previously forbade imports outright while the exec namespace exposed full `__builtins__`, so `import numpy` already worked and the model simply never tried: it wrote pure-Python loops over the adjacency next to a numpy install. Vectorized scoring is what makes a large RIS `theta` affordable inside `--strategy-timeout`, which is the difference between a crippled RIS strategy and a competitive one. _This is a research scaffold, not a security sandbox._

#### Exemplars and library source

The reply skeleton used to be a one-line `high_degree` call with "do not return this unchanged": an anchor so low the model would submit a variation of it and spend its iterations tuning a constant. The system prompt now carries two full strategies (`prompts.one_shot_exemplars`): a community-aware discount, and a lazy-greedy max-coverage over reverse-reachable sets with the staleness-stamped heap that makes a large `theta` affordable. The user prompt additionally carries the complete source of `degree_discount`, `cofim`, and `degree_ris_refine` (`library_api.sourced_algorithms`), one per idiom the library uses. Signatures alone leave the model guessing at how a seed set is actually built here.

#### The horizon note (`allowed_ops`-dependent)

Under `add_node`-only IC the cascade is progressive and monotone, so a seed at `t>0` has strictly fewer steps to spread than the same seed at `t=0` and every schedule is dominated by "all seeds at `t=0`". The old prompt spent its most prominent paragraph pushing the model to "schedule interventions across t₀…t_T": a flat direction it would burn iterations exploring. `build_system_prompt` now takes the `TaskSpec` and picks: `temporal_scheduling_note` when the task allows edge ops (where timing genuinely matters), `seed_timing_note` otherwise, which says plainly to put every seed in element 0 and spend the effort on *which* nodes.

The LLM never sees raw topology. By design, the _generated code_ gets the real graph at runtime (`GraphInfo` with full `edge_index`, `ic_probs`, neighbor/degree accessors); the LLM's job is to write an algorithm that inspects the graph programmatically, not to reason over an adjacency list in context. Serializing edges into the prompt would ask the model to eyeball what its own program computes exactly, would let it hardcode node ids for one instance instead of writing an algorithm that generalizes, and would not fit past a few thousand edges anyway.

#### The graph profile (`tools/graph_profile.py`)

Aggregate structure instead of topology: what a human expert uses to choose an approach, computed once and cached on the `GraphInfo`:

- **degree distribution**: mean/median/max/p90/p99, plus `cv` and `max/mean`. Measured on 100-node families: `ba` cv=0.95, `sbm` 0.46, `er` 0.40, `ws` 0.11, so the `heavy_tail_cv = 0.7` cutoff separates hub-dominated graphs cleanly.
- **components**: count, largest-component share, isolate count. On netscience that is 396 components with the largest holding 23.9% and 128 isolates, which is *the* strategic fact about that graph and was previously invisible.
- **k-core**, **clustering**, **degree assortativity**, **density**, and **reciprocity** (directed only).
- **communities**: label propagation count, modularity, largest sizes. Above modularity 0.3 the profile says outright that budget should be allocated across communities rather than by global score.
- **IC transmission**: edge-probability stats plus per-node expected out-transmission. Note the *mean* is pinned to exactly 1.0 by the weighted-cascade construction (`p = 1/in_degree` makes every in-sum 1), so it carries no information; the profile reports the **median** and the fraction of supercritical nodes instead.

Above `heavy_stats_max_arcs` (200k arcs) clustering switches to a sampled estimate and assortativity/communities are skipped, and the profile *says so* rather than silently omitting them. netscience profiles in 0.04s.

#### The reference diff (`methods/base.py::reference_diff`)

`baseline_anchor` rolls out every algorithm in `anchor_algorithms` in the same environment and returns the sorted table plus the **best** one's trajectory, so its per-node marginals can be diffed against the agent's. **Zero extra rollouts beyond the leaderboard**: both vectors are already paid for. Refinement turns get:

```
REFERENCE SCORES: classical baselines run on THIS graph, under THIS evaluator,
at the same budget and horizon. Beating the top row is the bar:
  degree_discount        47.50 (±4.13 SE)
  imm                    45.67 (±3.10 SE)
  pagerank_seeds         45.58 (±3.32 SE)
  high_degree            44.75 (±3.71 SE)
  random_seeds           36.08 (±3.59 SE)

REFERENCE DIFF (your cascade vs degree_discount's: the strongest baseline on
this graph: same evaluator, per-node P(infected)). Listed nodes are DECISIVE
flips only (one side >= 50%, the other < 10%); the net line below sums every
node, so it is larger:
  it reaches 3 nodes you miss: top by degree: 13(d=8, ref P=0.57 vs yours 0.07), …
  net expected spread vs the reference: -15.34 nodes
```

The double threshold keeps threshold-straddling nodes out of the list. In `evolve` the diff is appended to each population record's summary, so it travels with a candidate whenever it is shown as parent or inspiration. Canned arms (classical baselines, routing) skip the anchor entirely: they never read a prompt, so the rollout would only burn real episodes.

### Extraction, execution, validation

1. `extract_code_block` collects all fenced blocks and prefers the first one containing `class ` (models pad replies with prose snippets in extra fences), falling back to the first fence, then the raw text.
2. `check_script_imports` parses the AST and rejects any import outside `allowed_imports`, plus the builtins that would defeat that check.
3. `build_strategy` compiles and `exec`s the script in a **controlled namespace** containing exactly the names the prompts advertise: `ActionOp`, `State`, `GraphInfo`, `Strategy`, `primitives`, `cascade_features`, and every library pool (`algorithms`, `adaptive_algorithms`, `dismantling_algorithms`, `blocking_algorithms`, `immunization_algorithms`, `localization_algorithms` + `localization_scorers`, `reconstruction_algorithms`, `prediction_algorithms`) with the MC-heavy members bound to raisers. Candidate classes are discovered by attribute (`plan_horizon`/`act`/`localize`/`reconstruct`/`predict`), excluding the injected base _by identity_ (a script may legally name its class `Strategy`). The instantiated strategy gets `source_script` attached so results JSONs archive the exact code that earned the reward. _This is a research scaffold, not a security sandbox._
4. `call_strategy` wraps every invocation of generated code: any exception becomes a `StrategyError` carrying the exception type and full traceback, and the call runs under a **wall-clock cap** (`--strategy-timeout`, default 300s, `0` disables). An overrun becomes a repair turn that names the limit and says what to do about it, instead of hanging the whole sweep on one O(N²) scan. The cap is a SIGALRM, so it lands between bytecodes: a call stuck inside a long numpy/networkx C call overruns until that call returns.
5. `validate_actions` checks each bag: op ∈ `allowed_ops`, node ids in range, **no node seeded twice** (a duplicate `add_node` spends two units of budget on one node and would otherwise pass silently as budget the strategy never used), and seeds within budget. `validate_plan` additionally enforces the total across the plan and rejects re-seeding a node at a later timestep; `per_step` enforces the same across the episode.

`StrategyError` is the failure currency of the subsystem: it marks "the generated program is wrong" (recoverable, becomes a revision prompt), as opposed to every other exception, which is a bug in _this_ codebase and crashes loudly.

### Providers

`GatewayProvider` speaks the OpenAI chat API against the lab gateway, routing the bearer token by model family (`claude-*` vs `gpt-*`; tokens + `GATEWAY_BASE_URL` live in the gitignored `.env`, loaded by `run.py`). Two operational quirks it absorbs: the gateway's Claude account **drops system messages**, so system+user are folded into a single user turn (verified harmless for gpt models); and retries are owned locally (3 attempts with printed warnings) with SDK retries disabled. `--temperature 0.0` gives greedy decoding for reproducible evals on models that accept one (GPT-6 Astra does not, so the provider drops a requested temperature with a warning; `--reasoning-effort`, default `high`, is the knob it takes instead), keep the provider default (sampling) for refinement runs, where cross-iteration diversity is what the search feeds on. Any object with `complete(messages) → str` (a full chat thread, system turn first) satisfies `LLMProvider`; tests and `--baseline` inject canned providers.

---

## 6. Environments: the inner loop

Both environments implement `rollout(action_fn, horizon, budget, seed) → Trajectory` with identical semantics: start from the empty state (or, on a competitive task, from the rumour `S_N` already committed); at each timestep `t ∈ [0, horizon]` ask `action_fn(state, t)` for a bag, advance one step, record; stop early when the cascade is dead (empty frontier) _and_ the strategy is idle (empty bag). Reward is the **ensemble mean** of final infected counts; `cost.reward_se` (sample std / √n) self-reports the noise; the recorded `states`/`actions` are the **first sample's** trajectory (the representative), whose last count is overwritten by the ensemble mean, which is why `counts` can end in a non-integer.

### `MonteCarloEnvironment`: ground truth

`mc_runs` independent NDlib simulators (fresh child-seeded simulator per run), exact IC/LT stepwise dynamics with full action support. This is the reference the oracle referee is checked against under `--mc-agreement`, and the evaluator of the `@monte_carlo` ladder arm.

### `WorldModelEnvironment`: the learned simulator

Loads a trained `WorldModel` from the checkpoint a `train_wm.py` results JSON names (`ckpt_dir / wm_<backbone>_<dynamics>.pt`; keep per-dataset `ckpt_dir`s, the filename does not encode the dataset). The checkpoint is self-describing (`world_model/checkpoint.py`): its own spec decides the architecture, dynamics, remove semantics, competitive or compartmental layout, hidden edge weights and action encoding, and the JSON's `"config"` block is consulted only for a pre-v2 bare state dict. Rollout mechanics:

- **Lockstep block-diagonal batching**: all `n_samples` rollouts advance together; each timestep is ONE forward pass over the disjoint union of the samples' graphs (adjacency normalization is per-component, so this is exact, the same trick as the training collate). ~20× fewer forwards than per-sample loops.
- **Copy-on-write edge state**: the graph tensor is built once per rollout and rebuilt only when an edge op actually fires; each sample owns its own adjacency after mutating it.
- **Chunked forward passes**: samples are advanced in chunks of at most `default_max_block_arcs` (16M) arcs per forward pass, so 200 samples of a 4M-arc graph cost 50 passes per timestep instead of one pass over an 800M-arc block; on a graph the size of netscience one chunk holds every sample and the cost stays one pass per timestep.
- **Coupled sampling**: each step draws the _new infections_ once from the frontier marginal, then derives both channels, `frontier := new wave`, `infected` accumulates through the action semantics (adds join; IC removes stay counted; LT removes return to susceptible). Independent per-channel Bernoullis (the previous scheme) create impossible states ("ghost spreaders" with frontier=1, infected=0) that systematically inflate free-running rollouts (measured at ~+1 count bias via the oracle head).
- Per-sample early termination with an active mask; per-sample states are honored for `per_step` semantics (each sample's `action_fn` call sees that sample's own state).

### Trusting the evaluator (read before believing `reward`)

- **`wm_reeval_minus_referee` is the trust meter, not `arm_minus_referee`.** The referee replay always runs, but an arm's `reward` is a maximum over generations at one rollout seed, so it carries selection optimism (the winner's curse) plus that seed's luck. For an agent arm under `@world_model` or `@oracle` the winner's bags are therefore rolled out three more times on the arm's own evaluator at fresh seeds (`seed + 1 .. seed + 3`), and the mean of those minus the referee is the fidelity number. The WM's validated band on in-distribution plans is roughly ±2.
- **OOD is about action _intensity_, not just op type.** A WM trained with ~1 injected op per step scores single interventions well and mass interventions arbitrarily badly (observed error growing +2.4 → −5.7 → −39.7 with edge-boost count). Match the training action distribution to what the agent is allowed to emit.
- **One-step metrics cannot certify an evaluator.** A per-wave transmission optimism of ~1% is invisible to teacher-forced `delta_f1`/Brier and compounds to +2 nodes over a 10-step rollout. Gate world-model checkpoints on `rollout.ens_count_bias` (target |bias| ≲ 1), and prefer the `structured_residual` head for IC when training data lacks edge-weight diversity (it anchors per-edge transmission on the true probability and learns only a residual correction; zero correction equals the validated oracle).
- Ensemble sizes are a noise knob: comparisons between strategies need `mc_runs` such that the gap exceeds ~2×√(se₁² + se₂²); the `--mc-runs` default (200) gives SE ≈ 0.6 on BA-100-scale spreads.

---

## 7. Counterfactual credit (`--credit`)

`credit.py` converts the scalar reward into causal, per-action feedback: the executed plan and its `k` leave-one-out ablations (each with one budgeted action deleted; a stream's exogenous edge edits are never ablated) are scored on the **same environment at the base seed**, and `delta = base_reward − ablated_reward` is the spread that action is responsible for (`~0` = wasted budget). It runs only under `@oracle` and `@world_model`, where the batched environment scores all `k + 1` plans in one call of up to 512 sample blocks; under `@monte_carlo` it would cost `(k + 1) × mc_runs` real episodes, and the recover and forecast families emit no actions to credit. Two things ride along at no extra rollouts: each action's **solo cascade** (the one-bag plan of that action alone, with its final size and the step at which it stopped growing) and the **frontier bottleneck**, the unreached nodes the model's own kernel says one more wave would catch. With `--credit` (the default):

- refinement prompts (`one_shot` and `evolve`) include the per-action report: the agent sees "your `set_edge_weight(7→9)` contributed +0.00" instead of a bare scalar, which is precisely the signal that kills wasted actions.
- The results JSON gains `credit_base_reward` + one `{t, op, target, delta}` entry per action.
- For state-dependent strategies (`per_step`/`windowed`) the recorded bags are replayed as a fixed plan, so credit is an approximation there (the live policy would have reacted to the ablated cascade).

Cost: one batched call per scored candidate. The batched path stripes samples across plans, so two ablations do not share realization streams and the residual noise is the ensemble standard error the loop already lives with.

---

## 8. The tools library

The strategy's callable surface: advertised to the LLM verbatim via `library_api.build_api_reference()` (signatures via `inspect`, summaries from the first docstring line).

**Primitives** (`tools/primitives.py`, 16 advertised): degree/out-degree/weighted- degree scoring, top-k selection, power-iteration PageRank, eigenvector/closeness centrality, `mc_simulate_spread` (the honest NDlib spread estimator), marginal gain, reverse-reachable-set sampling + max-coverage selection (`batch_reverse_sample`, `ris_select`), label-propagation communities + largest-remainder budget allocation, RIS sample-size heuristic, live-edge sampling + reachability, truncated path-product influence scores.

**Named algorithms** (`tools/algorithms.py`, 36, uniform signature `(graph, budget, diffusion_model, **kw) → list[int]`):

| Family        | Functions                                                                     |
| ------------- | ----------------------------------------------------------------------------- |
| Degree        | `high_degree`, `weighted_degree`, `degree_discount`                           |
| Centrality    | `pagerank_seeds`, `eigenvector_seeds`, `closeness_seeds`                      |
| Greedy (MC)   | `vanilla_greedy`, `celf`, `celf_pp`, `adaptive_greedy`                        |
| RIS           | `ris_basic`, `tim`, `imm`, `ssa`, `filtered_ris`                              |
| Path          | `sp1m`, `mia_pmia`, `ldag`                                                    |
| Sketch        | `static_greedy`, `skim`                                                       |
| Community     | `community_im`, `cofim`, `community_ris`                                      |
| Metaheuristic | `simulated_annealing`, `hill_climbing`, `genetic_algorithm`                   |
| Hybrid        | `pagerank_greedy`, `degree_ris_refine`, `celf_local_search`, `community_celf` |
| Other         | `betweenness_seeds`, `kshell_seeds`, `voterank`, `collective_influence`, `irie`, `random_seeds` |

Heavy algorithms (MIA/PMIA, LDAG, SKIM, StaticGreedy, SSA, IMM, CELF++, Adaptive Greedy) are faithful-but-simplified, noted in their docstrings. All MC-based estimators use the real NDlib simulator: the honest classical cost that the world model exists to undercut in the outer loop.

### What the agent may NOT call (`executor.mc_blocked_algorithms`)

Eleven of the thirty-six are hidden from **generated scripts** in every method, unless `--allow-mc-algorithms`:

```
vanilla_greedy  celf  celf_pp  celf_local_search  community_celf  pagerank_greedy
adaptive_greedy  hill_climbing  static_greedy  genetic_algorithm  simulated_annealing
```

Each one estimates spread by simulating the cascade once per candidate node per pick. Measured on BA-1589 (≈ netscience) at k=79 = 5% of N:

| kept | s | blocked | s |
| ---- | --- | ------- | --- |
| `imm` | 0.55 | `vanilla_greedy` | >120 |
| `tim` | 0.48 | `celf`, `celf_pp` | >60 |
| `skim` | 0.50 | `adaptive_greedy`, `hill_climbing` | >60 |
| `ris_basic`, `filtered_ris` | 0.08 | `static_greedy` | >60 |
| `betweenness_seeds` | 2.68 | `pagerank_greedy`, `community_celf` | >60 |
| `degree_discount`, `voterank`, … | <0.15 | `celf_local_search` | >60 |
| | | `genetic_algorithm` | 18.05 |
| | | `simulated_annealing` | 17.21 |

Nothing in the library sits between 2.68s and 17.21s, so the cut needs no arbitrary threshold. Note `static_greedy` calls **no** MC primitive: it is blocked because its per-pick snapshot-reachability scan has the same shape and cost. `genetic_algorithm` / `simulated_annealing` are bounded (a fixed population × generations budget, not a full-N scan) but still call `mc_simulate_spread`, so they are blocked on the honesty ground below.

**The honesty ground matters more than the speed.** These run on a private `Simulator` built inside `primitives`, so `MonteCarloEnvironment.episodes_used` never sees them. Measured: one `one_shot` iteration whose script called `celf(mc_runs=20)` simulated **7,462** NDlib episodes and reported `real_env_episodes: 2`. That field is the sample-efficiency axis the whole condition ladder is read on, and conditions 3 (`@native`, "real executions only") and 6 ("no real episodes") both depend on it meaning what it says.

There is also a cleanliness argument: `celf_pp` is a condition-1 baseline, so letting an agent arm call it makes condition 6 partly *be* condition 1 plus scheduling. Blocking it forces the agent to beat CELF++ rather than invoke it.

**Baselines and routing are exempt.** `--baseline celf_pp` (condition 1) and a routing pick (condition 2) run through the same canned-script path, and *are* that algorithm: blocking would delete the arm rather than speed it up, and their cost is honestly attributed. `run_experiment` computes `effective_allow_mc = config.allow_mc_algorithms or canned_script is not None`.

A blocked name stays bound to a raiser rather than vanishing, so calling one produces a legible repair turn naming the alternatives instead of an `AttributeError` traceback. The prompt lists the blocked set up front (`build_api_reference(exclude=…)`) so the first iteration is not spent discovering it.

---

## 9. Running experiments

> **For a full sweep, use the pipeline instead.** `python -m pipeline.run` runs every baseline condition at every budget, resumes what it already finished, and writes the plots and `results/<task>/<dataset>/<run>/report.md`. One arm at one budget is one file at `results/<task>/<dataset>/<run>/agent/<budget>/<arm>.json`: exactly what the commands below produce, so the two are interchangeable:
>
> ```bash python -m pipeline.run --dataset ba --budget-pcts 1 5 10 20 \
>     --outer-iters 5 ```
>
> That default expands to the six-condition taxonomy: `--baselines` names the classical pool (condition 1), and `--arms` names `routing` (2) plus `evolve_free@{native,monte_carlo,oracle,world_model}` (3-6). An arm spec is `baseline:<algorithm>`, `routing`, or `<method>_<mode>[@<evaluator>]`, where the evaluator suffix is what makes conditions 3-6 differ in exactly one thing. See `pipeline/conditions.py` for the grammar.
>
> The shared referee is what makes a multi-condition table valid: each arm's own reward comes from its own evaluator, so only the referee replay (`referee_reward`, the exact oracle simulator at 1,000 samples by default) is comparable across rows; `--mc-agreement` adds the NDlib check.

```bash
# LLM run: node-ops game, WM evaluator, MC compare, credit feedback
python -m coding_agent.run --data-dir results/influence_maximization/ba/default/data \
    --wm-results-json results/influence_maximization/ba/default/world_model/sage_IC.json \
    --method one_shot --evaluator world_model --budget 5 --horizon 10 \
    --allowed-ops add_node remove_node \
    --model gpt-6-astra --outer-iters 3 --credit \
    --out-json results/influence_maximization/ba/default/agent/pct5/run.json

# Classical baseline through the identical pipeline (no LLM, no .env needed)
python -m coding_agent.run --data-dir results/influence_maximization/ba/default/data \
    --wm-results-json results/influence_maximization/ba/default/world_model/sage_IC.json \
    --method one_shot --evaluator world_model --budget 5 --horizon 10 \
    --baseline celf --outer-iters 1 --out-json results/influence_maximization/ba/default/agent/pct5/baseline_celf.json

# GA routing: one LLM call picks a library algorithm (no code synthesis),
# then it runs through the identical --baseline canned path
python -m coding_agent.run --data-dir results/influence_maximization/ba/default/data \
    --model gpt-6-astra \
    --wm-results-json results/influence_maximization/ba/default/world_model/sage_IC.json \
    --method one_shot --evaluator world_model --budget 5 --horizon 10 \
    --routing --outer-iters 1 --out-json results/influence_maximization/ba/default/agent/pct5/routing.json

# Oracle-dynamics ceiling: true IC transitions, no checkpoint / --wm-results-json
python -m coding_agent.run --data-dir results/influence_maximization/ba/default/data \
    --model gpt-6-astra \
    --method one_shot --evaluator oracle --budget 5 --horizon 10 \
    --outer-iters 5 --out-json results/influence_maximization/ba/default/agent/pct5/oracle_run.json

# Native coding agent: one real execution per candidate, episodes counted
#   ... --evaluator monte_carlo --mc-runs 1

# Reproducible eval: greedy decoding
#   ... --temperature 0.0
```

Key flags: `--method {one_shot,per_step,windowed,evolve,adaptive}` · `--strategy-mode {free,scored}` (scored = agent fills `score()`/`schedule()` hooks in the fixed `ScoredStrategy` harness; no `algorithms.*`; one_shot/evolve/adaptive only) · `--evaluator {world_model,monte_carlo,oracle}` (`oracle` = the exact dynamics via the `structured_oracle` head, no checkpoint: `q = p(u→v)` under IC, the closed-form threshold hazard under LT, the simulator's own rates under SIR/SIS/SEIR and both campaigns' probabilities under competition; the model-based ceiling and the default referee; `monte_carlo --mc-runs 1` = the native-agent condition, one real execution per candidate) · `--model` (gateway name; default `gpt-6-astra`) · `--temperature` (omit = provider default; `0.0` = greedy) · `--allowed-ops` · `--baseline <algorithm>` (synthesizes the all-at-`t₀` special-case plan) · `--routing` (GA-routing baseline: the LLM selects one pool algorithm from a name+summary menu, adaptive _selection_ without synthesis; the pick is recorded as `model: routing:<algo>` and the raw reply as `routing_reply`) · `--budget` / `--budget-pct` (percent of `num_nodes`, overrides `--budget`, same resolution rule as data generation) / `--horizon` / `--windows` / `--outer-iters` · `--mc-runs` (MC ensemble / compare size; default 200) · `--n-samples` (WM and oracle ensemble; default 200) · `--graph-id` (default: first graph in the store) · `--referee` (with `--mc-agreement` for the NDlib replay) · `--credit` · `--out-json`.

### Results JSON schema

| Key                                                                | Meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `method`, `evaluator`, `model`                                     | provenance; `model` is the gateway name, `baseline:<algo>`, `routing:<algo>`, or `canned`                                                                                                                                                                                                                                                                                                                                                                      |
| `graph`                                                            | `{graph_id, num_nodes, num_edges, directed}`; `num_edges` counts directed arcs (edge_index columns), matching the prompt stats                                                                                                                                                                                                                                                                                                                                 |
| `budget`, `budget_pct`                                             | the seed budget the run was constrained to, absolute and as % of `num_nodes`                                                                                                                                                                                                                                                                                                                                                                                   |
| `effective_budget`                                                 | seeds actually committable per episode. Equals `budget` for every method except `windowed`, whose budget is per window call by design: read this column before ranking a windowed arm against the others                                                                                                                                                                                                                                                      |
| `reward`, `spread_pct`                                             | ensemble-mean final spread under the inner-loop evaluator, absolute and as % of `num_nodes`                                                                                                                                                                                                                                                                                                                                                                    |
| `summary`                                                          | one-line trajectory summary (final spread, steps, per-step counts)                                                                                                                                                                                                                                                                                                                                                                                             |
| `script`                                                           | the exact source of the winning strategy (per_step: last generated script)                                                                                                                                                                                                                                                                                                                                                                                     |
| `explanation`                                                      | the agent's own plain-English write-up of the search and the winning script, in markdown (§ headings below). `null` for baseline/routing arms, which synthesized nothing                                                                                                                                                                                                                                                                                       |
| `cost`                                                             | `{n_samples\|mc_runs, env, reward_se, rollout_seconds}` for the winning trajectory                                                                                                                                                                                                                                                                                                                                                                             |
| `timeline`                                                         | per-timestep log of the representative rollout: bag applied at `t` + post-step `infected`/`frontier` lists and counts; may be shorter than horizon (early termination)                                                                                                                                                                                                                                                                                         |
| `real_env_episodes`                                                | cumulative real-environment episodes consumed by inner-loop feedback (0 for `world_model`/`oracle`; the referee and agreement replays are excluded)                                                                                                                                                                                                                                                                                                                |
| `evaluator_calls`, `evaluator_seconds`, `forward_passes`           | the rest of the inner-loop cost, on the same accounting basis: how many times the search queried its evaluator, the wall clock spent inside it, and the model-based unit of work (0 for `monte_carlo`). **`evaluator_seconds` is the axis the six-condition cost claim is read on**: `elapsed_seconds` is dominated by LLM latency, and `cost.rollout_seconds` is only the final rollout. All four counters are captured before the `--credit` and `--referee` (with `--mc-agreement` for the NDlib replay) replays, which query the same environment for post-hoc analysis and would otherwise charge the search for work it did not do |
| `llm_transcript`                                                   | every turn verbatim (`{turn, kind, prompt, reply}`) with the model's prose intact. The code extractor keeps only the fenced block, but the prose around it is where the model says what it was trying to do: irrecoverable afterwards, and the first thing worth reading when a run goes wrong. Empty for canned arms                                                                                                                                        |
| `llm_usage`                                                        | `{calls, prompt_tokens, completion_tokens, total_tokens, cost_usd}` summed over every provider the run created, the routing call included. `cost_usd` is `null` unless `--llm-price-in`/`--llm-price-out` were given: the lab gateway fronts Pro subscriptions and bills nothing per token, so there is no rate to assume. Token counts are exact either way, and also land in `summary.csv`                                                                  |
| `seed`                                                             | base rollout seed for the run (`--seed`); each rollout additionally records the seed it actually used in its own `cost.seed`, and `mc_seed` / `wm_reeval_seeds` record the referee and re-evaluation seeds                                                                                                                                                                                                                                                     |
| `history`                                                          | per-outer-iteration `{iteration, reward, best, script, parent_iteration, plan_seconds, rollout_seconds}` (evolve also logs `operator`; failed iterations carry `reward: null`, `error`, and their `repair` index). `script` is the attempt's full program and `parent_iteration` the iteration whose script it edited (`null` for one written from scratch), so the starting, intermediate and final algorithms of a run are all readable from the JSON. Empty for baseline/routing arms. Drives the convergence plot. `plan_seconds` is the generated algorithm's own compute, with free-mode composition scripts it dominates wall clock, and it is the only way to tell a slow-but-good strategy from a fast-but-lucky one |
| `selection_reward`, `generalization_gap`, `spread_curve`, `probes`, `memory` | the search's own reward before the held-out re-run (recover and forecast tasks) and the gap to the held-out one; the ensemble-mean infected count per timestep padded to `horizon + 2`; every probe the model asked with its answer, rollouts and seconds; and `evolve`'s running design-rule memory |
| `referee`, `referee_samples`                                       | which referee replayed the winner (`oracle` by default, or `monte_carlo`) and at how many samples (`--referee-samples`, 1,000 by default), independent of the inner loop's `--mc-runs`                                                                                                                                                                                                                                                                                           |
| `arm`, `arm_spec`, `condition`, `condition_name`, `budget_label`   | added when the run came from `pipeline.run`: which arm, which of the six baseline conditions, and which point of the budget sweep                                                                                                                                                                                                                                                                                                                              |
| `credit_base_reward`, `credit`                                     | with `--credit`: paired-ablation base reward + per-action deltas                                                                                                                                                                                                                                                                                                                                                                                               |
| `referee_reward`, `referee_spread_pct`, `referee_reward_se`, `referee_rollout_seconds` | the referee's replay of the winning strategy, the headline number (absolute + % of `num_nodes`). The replay re-runs the exact **actions** the winner took, not the strategy, a script that samples (RIS with a live seed, a randomized local search) returns a different seed set on a second call, so re-planning would referee a strategy that never ran                                                                                                        |
| `arm_minus_referee`, `wm_reeval_mean`, `wm_reeval_seeds`, `wm_reeval_minus_referee` | the arm's own reward minus the referee (it carries the winner's curse, so it is not the fidelity number) and, for an agent arm under `@world_model` or `@oracle`, the winner's bags re-rolled at three fresh seeds on the arm's own evaluator: `wm_reeval_minus_referee` is the evaluator-fidelity number |
| `mc_reward`, `mc_reward_se`, `mc_rollout_seconds`, `mc_agreement_runs`, `referee_minus_mc` | with `--mc-agreement`: the same winner on NDlib Monte Carlo, the independent check on the oracle referee and the per-rollout timing                                                                                                                                                                                                                                                                                                                                                                                                    |
| `elapsed_seconds`                                                  | whole experiment including LLM calls                                                                                                                                                                                                                                                                                                                                                                                                                           |

### The `explanation` write-up

After the refinement loop ends, one extra turn goes out **on the same conversation thread** (`Conversation.ask`: prose, no code extraction), so the model is describing the scripts it can still see rather than reconstructing them from the winner alone. `build_explanation_prompt` echoes the winning script anyway, because the winner is the max over iterations and is often *not* the last turn, without the echo the model narrates the wrong algorithm. The thread is also trimmed to the last `history_exchanges` turns, so the model cannot see most of its earlier attempts; the prompt therefore renders every iteration from `history` as a unified diff against the script it edited (the first script in full), capped at `max_diff_lines` per iteration, and tells the model to read each change off its diff. Before this, every iteration outside the window came back as "no longer visible".

The reply is printed to stdout and stored as `explanation`, in markdown under seven fixed headings:

```
## Summary                        what the final algorithm is, what it scored
## Iteration log                  one ### per iteration: goal, what changed in the method, did it work
## From first attempt to winner   the starting algorithm, the intermediate step that moved the reward most, start-to-finish difference and reward move
## Closest classical algorithm    the nearest published/library method (from the opening turn's baseline table), what is shared, exactly how it differs
## How the final algorithm works  method-level walkthrough: what each stage buys, library algorithms called, own ideas marked [new]; no code, no libraries, no bookkeeping
## What is new and why it wins    per [new] step: what the agent figured out, why the classical method cannot produce it, which iteration and reward move it accounts for; plus the graph property exploited and the diagnostic showing it
## Limitations                    where it fails, what to try next
```

Cost: one LLM call per agent arm per budget point. Canned arms (`--baseline`, `--routing`, external repos) skip it, `_CannedProvider` answers every prompt with its script, and a library algorithm has nothing to explain.

Timing semantics: `cost.rollout_seconds` vs `mc_rollout_seconds` is the WM-vs-MC speed comparison on the same strategy (normalize by ensemble size: `/n_samples` vs `/mc_runs`). For `per_step`, rollout time includes LLM latency, use `one_shot`/`windowed` for clean simulator timing.

---

## 9b. Feedback tiers (`--feedback`)

What a refinement generation is TOLD after its candidate is scored, as a
controlled variable rather than a fixed format. `legacy` is the default and is
this document's existing `summarize()` output, unchanged.

| tier     | blocks                                          | extra world-model rollouts / turn |
| -------- | ----------------------------------------------- | --------------------------------- |
| `f0`     | scalar reward                                    | 0                                 |
| `f1`     | + per-seed drop attribution `V(S) - V(S\{v})`   | k                                 |
| `f2`     | + regional coverage `sum_{v in R} p_v`           | k                                 |
| `f3`     | + seed overlap, bridge coverage, stagnation      | 2k                                |
| `legacy` | the existing feedback (default)                  | 0                                 |

The blocks come from `coding_agent/diagnostics.py`, which also backs the
agent-initiated `drop` / `swap` / `region` probes. Two properties matter for any
comparison across tiers, and both are asserted in `tests/test_diagnostics.py`:

- **no tier spends a trusted-simulator episode.** Every block is the graph, the
  bound evaluator, or arithmetic on its output, and `ProbeCost` reports which by
  reading the environment's own meters — the same probe on an `@monte_carlo` arm
  correctly reports `simulator_episodes`.
- **`legacy` is not a rung of the ladder.** It carries community reach and
  adjacent-seed hints (richer than `f2` in places) and it reads
  `graph.ic_probs` through its reverse-reachable residual gains, so it is not a
  clean "world-model feedback only" arm. Use f0-f3 for feedback-content claims.

A third property matters for the tasks that MINIMIZE (containment, influence
blocking, immunization): every reported difference is oriented by `task.sense`,
so **positive always means "this helps the objective"**. A raw
`V(S) - V(S\{v})` reads backwards under containment — the number that means "the
removal worked" is negative — and unoriented feedback would coach the agent to
undo its own improvements. `PlanDiagnostics(..., sense=task.sense)`; the report
header also states the direction outright.

Per-generation diagnostic blocks, with their costs, land in the results JSON
under `diagnostics`. Background: `research/action_conditioning_and_feedback.md`.

---

## 10. Practical notes

- **BA graphs are degree-trivial.** On BA-100, degree ≈ CELF ≈ any portfolio; a strong agent will _find the ceiling_ (≈ the classical baselines), not beat it. Separation comes from community-structured graphs (WS/SBM/real), from temporal scheduling, and (once budgeted) from edge ops that no classical baseline uses.
- `per_step` cost scales as active-samples × timesteps LLM calls per evaluation; budget accordingly or use `--evaluator monte_carlo --mc-runs <small>` while prototyping it.
- `--outer-iters 1` disables the repair loop; ≥3 recommended for any live model (the first script is frequently imperfect, and the traceback-as-feedback path is what fixes it).
- The `.env` at the repo root provides `GATEWAY_BASE_URL`, `CLAUDE_GATEWAY_TOKEN`, `CHATGPT_GATEWAY_TOKEN` (gitignored; `chmod 600`).
- Old results JSONs archive scripts written against the former `Action` alias; the exec namespace now exposes `ActionOp`: re-generated scripts are unaffected, but replaying archived pre-rename scripts verbatim would fail.

---

## Forecasting: the fourth contract (`--task cascade_prediction`)

Three contracts above this one produce a NODE SET of some kind: a plan, a source set, a trajectory. This one produces a **number**, and it is the only task here whose reward MINIMIZES for a reason unrelated to containment.

```python
def predict(self, graph, observation, horizon) -> float | None
```

The popularity a REAL logged cascade will have reached by `horizon`, given its first `observation.observed_steps` timesteps. `None` DECLINES, which is a legitimate answer rather than an error: `research/cascade_prediction.md` §8.4 records that generative models refuse to score supercritical cascades (SEISMIC produced no prediction for 1,022 of ~20K News cascades at five minutes) and that papers reporting the mean over scoreable cascades alone "silently favour the model that gives up more often". Declines land in `n_failed`, never in the error, and `call_strategy(..., allow_none=True)` is the one place in this repo where a `None` return is not a missing implementation.

**The one new primitive is the harness's forward model, `prediction.ForecastOracle`**, with `expected_popularity` derived from the same call rather than being a second oracle. Its four bindings ARE conditions 3-6, exactly as the forward oracle and the kernel are for the two inverse tasks, and here that is the ONLY thing that varies down the ladder, because §2.1's `a_t = NULL` empties the action space entirely. The generated predictor is offline (since 2026-09-04): the forward model runs only in the harness, for the modelling-error check under `--mc-agreement` and for the feedback, and a canned kernel-using baseline such as `mc_forward` still receives it.

Pass `observation.frontier` rather than the whole adopter set: someone who adopted five steps ago has already had their chance to spread, and seeding a forward model from all of them over-predicts badly.

**`@native` may win outright.** §3.1 records CasFlow's own ablation finding that feature models "in some cases even beat deep learning models", and §5.1 puts Feature-based at MSLE 1.9881 on APS-3y against CasFlow's 1.4370: closer than a decade of architecture would suggest. That arm gets `cascade_features(graph, observation)`, which is Cheng et al.'s five classes minus content (our corpora carry no text), and the single most predictive quantity in this literature is in it: `rate_second_half`, the adoption rate in the SECOND HALF of the observation window, at 0.73 accuracy against 0.65 for the best structural feature.

Scored mode gives the agent `growth_factor(features, graph, observation)` (a multiplier on the observed popularity) and that is the tightest of the four scored harnesses: Szabo & Huberman's founding result is that `log P(t_p)` is near-linear in `log P(t_o)`, i.e. that the whole problem IS a multiplier, so the constrained space is exactly the one this literature's feature line occupies rather than a subset of it.

Two floors travel with every result and both are stronger than they sound. Under a LOG-space error the geometric mean of the training sizes is the minimizer over instance-blind rules (`trivial_predictor_error`), and predicting the already-observed count unchanged is right whenever a cascade is finished, which most are (`persistence_error`). An arm that does not clear both has learned the corpus's size distribution rather than anything about the instance. `coding_agent/check_cascade_prediction.py` asserts that and seven other properties.
