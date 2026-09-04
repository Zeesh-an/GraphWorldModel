# Coding Agent Methodology

This document is the complete technical description of the coding-agent side of GraphWorldModel: what the agent is asked to do, what it is shown, how its programs are executed and scored, how the search over programs proceeds, what feedback it receives from the world model and from the harness, what is measured afterwards, and which parts are taken from published methods. Every statement below was checked against the code as of 2026-09-04. Where a number is a default it is named as such; where a behaviour was chosen for a documented reason the reason is stated. Citations are collected in Section 14 and referenced inline as [n].

Section map: 1 problem statement and notation; 2 the ladder of conditions; 3 the pipeline around the agent; 4 the four evaluators; 5 the generated program (contract, sandbox, validation, the offline rule); 6 the context the model is given; 7 the search over programs; 8 feedback; 9 what happens after the search; 10 the per-task harnesses and rewards; 11 the adaptive, streaming and multi-round branches; 12 condition 9, the published discovery systems; 13 provenance table; 14 references; 15 known caveats.

---

## 1. Problem statement and notation

### 1.1 The object being optimized

A graph $G = (V, E)$ with $N = |V|$ nodes is stored as an arc list `edge_index` of shape $(2, E)$ with one transmission probability $p_{uv} \in [0, 1]$ per arc (`ic_probs`). An undirected graph is stored with both orientations of every edge. A diffusion process runs on $G$ for a horizon of $H$ timesteps. Its state at time $t$ is a tuple of node sets. For the single-cascade tasks it is $s_t = (I_t, F_t)$ where $I_t$ is the set of ever-activated nodes and $F_t \subseteq I_t$ the nodes activated at step $t$ (the frontier). The two-cascade task doubles the pair, $s_t = (I^-_t, F^-_t, I^+_t, F^+_t)$, with the negative cascade first. The compartmental task carries four exclusive compartments $(S_t, E_t, I_t, R_t)$ together with the ever-infected set.

An action bag $a_t$ is a list of operations `ActionOp(op, target, destination=None, weight=None)` with `op` one of `add_node`, `remove_node`, `add_edge`, `remove_edge`, `set_edge_weight`. The transition factorizes as

$$s_{t+1} = T_{\text{endo}}\bigl(T_{\text{exo}}(s_t, a_t)\bigr),$$

where $T_{\text{exo}}$ is the deterministic effect of the bag and $T_{\text{endo}}$ is one stochastic (IC, SIR, SIS, SEIR) or threshold-deterministic (LT) diffusion step. The world model $f_\theta(G, s_t, a_t) \to s_{t+1}$ learns both, and the coding agent uses $f_\theta$ as one of four interchangeable evaluators of the same programs.

### 1.2 Four problem families

The task registry (`pipeline/tasks.py`) holds eight runnable tasks in four families, distinguished by the `objective` field and by four boolean structural flags. The `sense` property is what every comparison actually reads, and it differs from the objective for two families:

| Family | `objective` | `sense` | Tasks | What the program returns |
|---|---|---|---|---|
| maximize | `maximize` | maximize | `influence_maximization`, `adaptive_online_im` | a seed plan (or a per-round seeding policy) |
| minimize | `minimize` | minimize | `critical_node_detection`, `influence_blocking`, `epidemic_control` | a removal, blocking or dosing plan |
| recover | `recover` | maximize (the reward is a consistency, an F1 is reported) | `source_localization`, `cascade_reconstruction` | a source set, or a full trajectory |
| forecast | `forecast` | minimize (the reward is an error) | `cascade_prediction` | a number per cascade |

The four structural flags each mark a difference the sign cannot express: `competitive` (two cascades, own simulator, 8-channel state), `epidemic` (non-monotone dynamics, four compartments, 9-channel state), `reconstructs` (the answer is a trajectory with parents, which needs traced data), `observational` (the transitions are replayed from a real log and no simulator runs). The runtime mirror of the registry is `TaskSpec` (`coding_agent/types.py`), whose predicates `contains`, `blocks`, `immunizes`, `recovers`, `decodes`, `forecasts`, `observes` route every branch in the prompts, the harness, the report and the plots. `contains` is `sense == minimize and not forecasts`; without the guard every reader would describe a prediction arm as an outbreak it was shrinking.

Thirteen import-time assertions in the registry enforce the invariants the families rely on, among them: a containment task must use `blocked` remove semantics (under `spent` each removed node stays counted and biases every containment number by exactly $+k$); a recover task must generate diffusion-only episodes (`default_gen_action_ops=()`), so that the $t=0$ seed commit and the final state form a well-defined $(x, y)$ pair; a competitive or epidemic task must have a non-zero outbreak; a task cannot be both competitive and epidemic; a forecast task is exempt from the `budget_op in allowed_ops` check because it spends no budget.

### 1.3 Reward and its noise

For every intervention task the reward of a program $\pi$ under evaluator $\mathcal{E}$ on realization seed $\xi$ is the ensemble mean of the final infected count over $n$ sampled rollouts,

$$\hat R_{\mathcal{E}}(\pi; \xi) = \frac{1}{n} \sum_{s=1}^{n} \bigl|I^{(s)}_{\text{end}}\bigr|, \qquad \widehat{\text{se}} = \frac{\sigma_{n-1}\bigl(\{|I^{(s)}_{\text{end}}|\}\bigr)}{\sqrt{n}},$$

where $n$ is `mc_runs` (NDlib episodes) under the Monte Carlo evaluator and `n_samples` (sampled world-model rollouts) under the oracle and world-model evaluators. Both environments compute exactly these two formulas (`envs/world_model_env.py`, `envs/monte_carlo_env.py`). For the two-cascade task $|I_{\text{end}}|$ is the rumour's final size; for the compartmental task it is the attack set (ever infected). The two inverse families and the forecast family have their own rewards, all label-free and all defined in Section 10. Each reward carries a standard error in `Trajectory.cost["reward_se"]`, and that number drives both the acceptance rule of the search (Section 7.6) and the noise band the model is shown (Section 8.2).

### 1.4 The design question

The methodology is a two-axis experiment. The first axis is who designs the algorithm: a fixed classical pool (condition 1), an LLM choosing from that pool (condition 2), an LLM writing programs (conditions 3 to 6), a published repository (condition 7), or a published LLM discovery system running its own loop (condition 9). The second axis is what inner-loop feedback the designer gets: one real episode per candidate with no forward model (native), a Monte Carlo simulator (NDlib), the exact vectorized oracle of the same dynamics, or the learned world model $f_\theta$. Conditions 3 to 6 hold the search method fixed and vary only the evaluator, so a difference between two of those rows is a difference between two evaluators and nothing else.

---

## 2. The ladder of conditions

### 2.1 Condition numbers and arm grammar

`pipeline/conditions.py` defines the vocabulary. Evaluators are `native`, `monte_carlo`, `oracle`, `world_model`. Methods are `one_shot`, `per_step`, `windowed`, `evolve`, `adaptive`. Modes are `free` and `scored`. There is deliberately no condition 8.

| Condition | Name | Arm spec grammar | Method | Evaluator |
|---|---|---|---|---|
| 1 | Pure GA | `baseline:<algorithm>[@<evaluator>]` | `one_shot` (or `adaptive` for a per-round policy) | `oracle` unless given |
| 2 | GA routing | `routing[@<evaluator>]` | `one_shot` | `oracle` unless given |
| 3 | Native coding agent | `<method>_<mode>@native` | as named | one real NDlib episode per candidate, no forward model |
| 4 | Agent + MC simulation | `<method>_<mode>@monte_carlo` | as named | NDlib, `mc_runs` episodes |
| 5 | Agent + oracle dynamics | `<method>_<mode>@oracle` | as named | exact batched simulator |
| 6 | Ours: agent + learned GWM | `<method>_<mode>@world_model` | as named | $f_\theta$ |
| 7 | Published baseline (external repo) | `external:<name>[@<evaluator>]` | `one_shot` (or `adaptive` if the repo is rounds-aware) | `oracle` unless given |
| 9 | Published LLM algorithm discovery | `discovery:<name>[@<evaluator>]` | `one_shot` | `oracle` unless given |

`parse_arm` resolves the spec in this order: an explicit `@evaluator` outside the valid set raises; `discovery:` and `external:` prefixes are read first; then `baseline:` and the literal `routing`; anything else is split with `rpartition("_")` into method and mode and must name a valid pair. For conditions 1, 2, 7 and 9 the evaluator only decides how a single result is scored, so it defaults to `selection_evaluator = oracle`, which is also the referee (Section 9.4); their one evaluation therefore is the row's final number. For conditions 3 to 6 the evaluator is part of the arm name and of the condition number. A baseline whose name is an adaptive per-round policy is recorded with `method="adaptive"` so the adaptivity-gap table divides it by a static control rather than treating it as one; an external repository flagged `rounds_aware` takes the same path because its canned script has the shape of a per-round policy.

`Arm.is_agent` is true exactly for conditions 3 to 6 and drives three decisions in the pipeline: the number of outer iterations (1 for every non-agent arm), the sample count of the single evaluation (the referee's count for non-agent arms), and whether the arm fans out over `--llm-models` (conditions 3 to 6, 2 and 9 do; 1 and 7 run once regardless).

### 2.2 Native is Monte Carlo with one episode and no forward model

`native` is not an environment. `resolve_evaluator` rewrites it to `monte_carlo` with `native_mc_runs` episodes per candidate (default 1), and separately sets `native_arm=True` on the experiment config. `native_arm` is what makes condition 3 what it is: `TaskSpec.forward_model = not native_arm`, so the harness scores candidates on one real noisy episode, refuses probes, and binds the raising `unavailable_*` oracle stubs on canned baselines (Section 5.7). The flag is included in the checkpoint fingerprint because it is not recoverable from the evaluator name, which reads `monte_carlo` for both conditions 3 and 4.

### 2.3 Default ladder and default pool

`default_arms = ("routing", "evolve_free@native", "evolve_free@monte_carlo", "evolve_free@oracle", "evolve_free@world_model")`. The method is `evolve` rather than `one_shot` because `evolve` edits the population best each generation while `one_shot` edits the latest attempt, so `one_shot` compounds a regression instead of rejecting it. The IM default baseline pool is `high_degree`, `degree_discount`, `pagerank_seeds`, `celf_pp`, `imm`, `random_seeds` (one per family: heuristic, discount, centrality, greedy, sketch, floor); every other task overrides it in the registry, and the two lever tasks resolve their pool per lever. `adaptive_online_im` overrides the arms with paired `evolve_free@X` and `adaptive_free@X` for each evaluator so the adaptivity gap always has its denominator at the same budget on the same evaluator.

### 2.4 Cross-condition comparability

Each arm's `reward` is measured by its own evaluator: one noisy episode for native, a model estimate for the world model. The only column that may be read across rows is `referee_reward`, the shared oracle replay of the executed action bags at 1,000 samples (Section 9.4). `ground_truth_reward(result)` returns it when present and falls back to `reward` otherwise, and every plot and table that ranks arms reads through it. Seven algorithm pools are checked at import time for name collisions (only `netshield` and `acquaintance_immunization` legitimately exist in two pools and are disambiguated by task).

---

## 3. The pipeline around the agent

### 3.1 Stages

`python -m pipeline.run --task <task> --dataset <name>` drives five stages, `data`, `train`, `agent`, `plots`, `report`, and resumes finished work. `train` is dropped automatically when no arm uses `@world_model` or when `--wm-results-json` points at an existing checkpoint. Everything lands in `results/<task>/<dataset>/<run>/`, with agent results at `agent/<budget>/<arm>.json` and external or discovery results at `baselines/<budget>/<arm>.json`; `pipeline/layout.py` is the only place these paths are joined.

### 3.2 Budget points and the arm loop

Budgets are a list of `(label, budget_pct, budget)` points. `--budgets` gives absolute $k$ values; otherwise a task's `default_budgets` (influence blocking ships $k \in \{10, 20, 30, 40, 50\}$, the convention of that literature) takes precedence over percentages; otherwise `--budget-pcts` or the task's `default_budget_pcts` (the repo-wide default is $1, 5, 10, 20$ percent of $N$; the inverse and forecast tasks ship a single 10 percent point because $k$ is a property of the instance or unused). The agent stage loops budgets outside and arms inside with one progress bar. An existing result JSON is reused unless `--force`; a failed arm writes a `.skipped.json` marker with the reason so a resume does not retry it, and a later success retracts the marker.

### 3.3 From pipeline config to experiment config

For each (budget, arm) the pipeline builds one `ExperimentConfig` (`coding_agent/run.py`). The load-bearing derivations:

- `outer_iters = 1` for every non-agent arm and for a transfer arm (`--sl-transfer-from`), because those have no refinement loop.
- `n_samples = referee_samples` for a non-agent arm (its single evaluation doubles as its referee replay), `min(n_samples, 50)` for an adaptive arm (its per-round policy runs once per ensemble member, so evaluation cost is linear in the sample count), and `n_samples` (default 200) otherwise. 200 is the ladder default because 50 samples put the reward standard error (2.5 to 12 nodes) above the deltas the late search iterations decide between.
- `native_arm = (arm.evaluator == native)`; `model = arm.llm_model or config.llm_model`.
- Every task-family block (adaptive rounds, outbreak, blocking lever, epidemic rates, SL/CR/CP splits) is forwarded always, so one code path serves every family.

Before any arm runs, `verify_gateway_model` lists the gateway's served models and raises if the configured one (default `gpt-6-astra`) is absent, so a run does not spend hours on data and training before its first LLM call fails.

### 3.4 What `run_experiment` resolves before the search

In order: the budget (`budget_pct` of $N$ overrides an absolute $k$, the same rule data generation uses); the executor timeout (`--strategy-timeout`, default 300 s, module-level); the outbreak, derived only from `(graph, size, selector, seed)` so both simulator and arm face the same one; the environment (Section 4) with the outbreak as `negative_seeds`; the multi-round wrapper if `--campaigns > 1`; the instance pools for the inverse and forecast tasks (two disjoint splits each); the lever, which sets `budget_op` and `allowed_ops` together so an arm can never be budgeted for one op and permitted another; and finally the `TaskSpec` that every downstream reader consumes.

For the outbreak, `outbreak_pct` comes from the registry unless overridden (10 percent for critical node detection, 1 percent for influence blocking where it is $|S_N|$, 1 percent for epidemic control, 0 for the seeding tasks), and `select_outbreak` picks `max(1, round(N * pct / 100))` nodes by the selector (`random` by default, deterministic in the seed; `degree`, `pagerank`, `betweenness`, `kshell` ignore the seed entirely).

### 3.5 Routing, transfer and canned scripts

Condition 2 is one two-message LLM call (a routing system prompt per family and a routing user prompt that carries the task header, the task blocks and the graph profile but no library reference) whose reply must name exactly one algorithm from the task's menu, parsed by exact match and then by whole-word regex over menu names sorted longest first; ambiguity raises. The pick then runs through the identical canned path as `--baseline`.

Every non-agent arm is a canned script: a `Strategy` subclass written by the harness around one library call. There are eight shapes (adaptive per-round policy with the round schedule inlined, predictor, reconstructor, localizer with both `localize` and `source_scores`, blocker through `blocking_plan`, immunizer through `immunization_plan`, dismantler through `removal_plan`, static seed set), each gated by pool membership and, for the two lever tasks, by whether the member's output shape matches the lever. A canned script always runs in free mode with the expensive simulation-based members re-exposed, because the declared baseline is that algorithm and blocking it would delete the arm rather than speed it up. The provider is `_CannedProvider`, which answers every prompt with the script and marks `canned = True`; that flag is the single gate on the evaluator bindings (Section 5.7).

---

## 4. The four evaluators

Every evaluator implements `rollout(action_fn, horizon, budget, seed=None) -> Trajectory` where `action_fn(state, timestep) -> list[ActionOp]`, and every cost counter is read off the environment after the search: `episodes_used` (real NDlib episodes), `rollout_calls`, `evaluator_seconds`, `forward_passes`.

### 4.1 Monte Carlo (NDlib)

`MonteCarloEnvironment` runs `mc_runs` (default 200) independent episodes per call. The call's seed spawns one child seed per episode from its own generator, so replaying a seed reproduces the whole ensemble. Each episode builds a fresh NDlib simulator [7] (IC or LT with per-edge thresholds), or the competitive simulator with the rumour already committed, or the compartmental stepper, and steps `for t in 0..H`: `bag = action_fn(state tagged with sample=run, t)`, `state = simulator.advance(bag)`, stopping early when the frontier (both frontiers under competition, plus the exposed set under SEIR) and the bag are both empty. `episodes_used += mc_runs` once per call and once per `step_marginals` call. `evaluator_seconds` grows with `mc_runs` times rounds here and stays flat under the world model, which is the cost claim of the whole ladder.

### 4.2 The world model

`WorldModelEnvironment` wraps a trained checkpoint (loaded from the results JSON, which names the checkpoint; the checkpoint's own spec decides dynamics, remove semantics, competitive or epidemic layout, hidden edge weights and action encoding). One rollout advances `n_samples` sampled states in lockstep:

1. All samples' graphs are stacked as one block-diagonal `GraphInput` (edge indices offset by `sample * N`), so one forward pass evaluates every sample. Edge state is copy-on-write per sample: an edge op in a sample's bag rewrites that sample's arc dictionary and the block is rebuilt.
2. Per timestep, each active sample's `action_fn` is called with its own `State` (tagged with `sample=s`); features are built per sample from the post-action graph (the degree column is cached by the identity of the edge array with a re-check, because two edge ops can free and re-allocate an array at the same address).
3. One forward pass yields probabilities of shape $(n, N, c)$ with $c = 2$ (single cascade), 4 (competitive) or 5 (compartmental).
4. One uniform draw of shape $(n, N)$ against the frontier column decides the new infections for all samples at once (including dead ones, which keeps the random stream aligned as samples die). Drawing the frontier and deriving the infected set from it is deliberate: independent per-channel draws produce ghost spreaders (`frontier=1, infected=0`) that inflate free-running rollouts.
5. Under `spent` IC semantics a removed node stays counted; under LT or `blocked` it leaves. The competitive and compartmental branches delegate to two coupled samplers (Section 4.5).
6. A sample deactivates when $t > 0$ and its frontier(s) and bag are empty; the call ends when every sample is inactive. Dead samples keep appending their frozen count so per-sample curves stay equal-length, which is what lets credit read stagnation timing without extra rollouts.

`episodes_used` stays 0 by definition; `forward_passes` counts one per timestep per rollout plus one per `step_marginals` call. The returned trajectory carries sample 0 as the representative path, `final_marginals` (per-node activation frequency over samples), `spread_curve` (ensemble mean count per step, padded by holding the last value, which is exact because an empty frontier with an empty bag is a fixed point of monotone dynamics), and `prevalence_curve` for the compartmental task (padded with zeros, because a dead epidemic's prevalence is zero).

`step_marginals(state)` returns $P(v \text{ activates at } t{+}1 \mid s_t)$ from one forward pass against the base edges; it is the transition kernel the reconstruction harness scores histories with (Section 10.2), and it is what makes arm 6 pay one batched product where arm 4 pays `mc_runs` real episodes per kernel evaluation.

### 4.3 The oracle

`WorldModelEnvironment.oracle(...)` builds the same environment around a throwaway one-layer GCN with the `structured_oracle` head, whose per-edge transmission $q_{uv}$ is pinned to the true $p_{uv}$ (IC), to the closed-form threshold hazard (LT), to the simulator's own $(\beta w_{uv}, \gamma, \alpha)$ (compartmental), and to the true probabilities of both campaigns composed through the tie-break (competitive). The encoder output never reaches the head, so no checkpoint exists and no learning happens. The oracle is therefore the same simulator as NDlib in distribution but vectorized and batched; the self-checks measure it against thousands of NDlib draws for every head. It plays two roles: the ceiling condition 5 (an agent with perfect dynamics), and the referee every final number is read on (Section 9.4). Verified on the smoke runs: an IM winner scored 33.34 ± 0.17 on the oracle against 33.30 ± 1.15 on NDlib.

### 4.4 Native

Condition 3 is the Monte Carlo environment with `mc_runs = 1` (one real episode per candidate) and `forward_model = False`. The program is identical to the other three arms; what it loses is any forward model in the harness: no probes, no residual or likelihood feedback computed with a kernel, and the raising stubs on canned baselines. The one-episode reward is noisy by design, and the paired-seed protocol of the search (Section 7.5) is what keeps the search from tuning to that noise.

### 4.5 The coupled samplers

Both non-single-cascade branches sample from the head's marginals with one uniform per node so the sampled state is one the simulator can reach. Competitive: `total = p_neg + p_pos` on free nodes; `activates = u < total`; `goes_negative = activates and u < p_neg`. A blocker seed cannot flip a rumour node; a removed node leaves both cascades. Compartmental: $S$ is derived as $1 - E - I - R$, the four are renormalized (the head's composition passes through a sigmoid and a clamp, so the sum is 1 only up to floating point), one uniform picks the bucket by cumulative sum, a dosed node is forced to $S$, `exposed` is masked to empty except under SEIR and `recovered` under SIS, and the ever-infected set advances from the sample (a node counts as newly infected exactly when it was susceptible before the step and is not after), which is the only definition that survives SIS sending a node back to $S$.

### 4.6 Common random numbers and the rollout seed

`--seed` (default 42) is the base seed of every rollout in a run, shared across candidates on purpose [20], and every rollout records the seed it actually used in `cost["seed"]`. The search loops override it per generation (Section 7.5); the referee, credit and probes use the base seed; the world-model re-evaluation uses `seed + 1 .. seed + 3` to break the sharing on purpose (Section 9.6).

### 4.7 Two wrappers that compose with any evaluator

`MultiRoundEnvironment` (`--campaigns r`) runs $r$ genuinely fresh inner rollouts with seeds `seed + i` (LT re-draws its thresholds, which is correct because they are per-episode), hands the policy the realized union so far merged into `infected` so campaign 3 can tell itself from campaign 1 without a counter, and scores the union exactly by

$$R = \sum_{v \in V} \Bigl(1 - \prod_{i=1}^{r} \bigl(1 - p_i(v)\bigr)\Bigr),$$

with $p_i(v)$ the per-node marginals the inner environment already returns. It forwards the four cost counters rather than shadowing them and reports `reward_se = 0.0` deliberately, because the standard error of one campaign is not the standard error of the union. `GraphEditStream` (`--edit-rate`) is a deterministic schedule of exogenous edge edits per timestep, a pure function of (base graph, seed, timestep), delivered through the action bag so that the Monte Carlo (episode, timestep) and world-model (timestep, sample) loop orders see the same $A_t$ (Section 11.2).

---

## 5. The generated program

### 5.1 The contract

A generated script is one Python module defining exactly one class that subclasses `Strategy`, a `typing.Protocol` (`coding_agent/types.py`) with five entry points. A script implements only the one its task needs, and the executor checks that the required one is present:

| Entry point | Signature | Task family |
|---|---|---|
| `plan_horizon` | `(graph, budget, horizon) -> list[list[ActionOp]]`, one bag per timestep $0..H$ | intervention tasks under `one_shot` and `evolve` |
| `act` | `(state, graph, timestep) -> list[ActionOp]` | `per_step`, `windowed`, and the adaptive per-round policy |
| `localize` | `(graph, observation, budget) -> list[int]`, plus optional `source_scores(graph, observation) -> np.ndarray` | source localization |
| `reconstruct` | `(graph, observation, horizon) -> dict[int, tuple[int, int \| None]]` | cascade reconstruction |
| `predict` | `(graph, observation, horizon) -> float \| None` | cascade prediction |

The program sees the graph as a read-only `GraphInfo` (`num_nodes`, `edge_index`, `ic_probs`, `directed`, lazy `out_neighbors`, `in_neighbors`, total `degree`), never a simulator handle. Because `Strategy` is a Protocol whose bodies are `...`, `hasattr` cannot tell a written method from an inherited stub; every harness therefore uses an identity test, `getattr(type(strategy), name) is getattr(Strategy, name)`, to decide whether the program wrote the method. Without it an optional `source_scores` would become a required one that fails with "returned None", and a missing `predict` would read as a decline on every cascade.

### 5.2 Scored mode

`--strategy-mode scored` swaps the contract for `ScoredStrategy`, a fixed harness in which the search edits only a scoring rule. `plan_horizon` is a fixed greedy loop: repeatedly pick the unselected, unprotected node with the highest `score(node, selected, graph)` (ties break to the lowest id) until the budget is spent, then `schedule(seeds, graph, horizon)` places the picks (default: everything at $t=0$ under the task's budget op). The inverse and forecast tasks get the same treatment: `localize` is the same greedy over `source_score(node, graph, observation, selected)` (default: the Comin and Costa infected-degree count [27], $-\infty$ for an uninfected node), `reconstruct` is a fixed ordered-Steiner Dijkstra out of estimated roots over `edge_cost(u, v, p, graph, observation)` (default $-\log p$, so a most-likely path is a cheapest path), and `predict` returns `max(P(t_o), P(t_o) \cdot \text{growth\_factor}(\text{features}))` (default factor 2.0, Cheng et al.'s doubling constant [31]). Overriding a fixed harness method is rejected by the executor, because that would turn scored mode back into free mode. Scored mode hides the callable library entirely and the three simulation primitives, so the model must invent structural scoring logic rather than compose CELF.

### 5.3 The sandbox

`coding_agent/executor.py` builds the strategy in four steps. First an AST pass: a `SyntaxError` and any use of `__import__`, `eval`, `exec`, `compile`, `open` or `input` are rejected by name, and every import's top-level package must be one of `numpy`, `networkx`, `scipy`, `math`, `random`, `statistics`, `heapq`, `bisect`, `collections`, `itertools`, `functools`. Second, the script is `exec`'d into a namespace that contains exactly the names the prompts advertise: `ActionOp`, `State`, `GraphInfo`, `Strategy`, the eight algorithm pools as namespaces (`algorithms`, `adaptive_algorithms`, `dismantling_algorithms`, `blocking_algorithms` with its lever map, `immunization_algorithms` with its lever map, `localization_algorithms`, `localization_scorers`, `reconstruction_algorithms`, `prediction_algorithms`), `cascade_features`, the whole `primitives` module, and the three plan helpers `containment`, `blocking`, `epidemic`. All pools are present under every task so the namespace never diverges from the prompt. Third, the last class defined in the script that exposes one of the five entry points wins. Fourth, `strategy.source_script` is stamped (archived verbatim in every results JSON) and `strategy.canned` is set from the caller (Section 5.7).

Every call into the program goes through `call_strategy`, which arms a `SIGALRM` wall-clock limit of `strategy_timeout_seconds` (default 300; 0 disables; only on the main thread of a Unix process, so a call stuck inside a long C routine overruns until it returns). A timeout, a `None` return (except from `predict`, which may decline) and any exception become a `StrategyError` whose message is written to be fed back as a repair turn: the timeout text names the usual causes (a scan over all nodes inside a per-pick loop, an oversized sample), the `None` text names the wrong-entry-point diagnosis.

### 5.4 Blocked members and why they stay visible

Eleven IM algorithms (`vanilla_greedy`, `celf`, `celf_pp`, `celf_local_search`, `community_celf`, `pagerank_greedy`, `adaptive_greedy`, `hill_climbing`, `static_greedy`, `genetic_algorithm`, `simulated_annealing`) and one simulation-based member of each other pool (`greedy_blocking`, `greedy_prevention`, `mc_greedy_immunization`, `resim_greedy`, `mcmc_decode` and `forward_backward`, `mc_forward`) are blocked from generated code unless `--allow-mc-algorithms`. Two reasons are recorded in the executor. Time: on BA-1589 at $k = 79$ all of them exceed 60 s per call and `vanilla_greedy` exceeds 120 s, while everything still available runs under 2.7 s. Honesty: their episodes run on a private NDlib simulator invisible to `episodes_used`; one `one_shot` iteration calling `celf(mc_runs=20)` simulated 7,462 episodes and reported `real_env_episodes = 2`, and that field is the sample-efficiency axis the ladder is read on. Blocked names remain in the namespace as raisers so that calling one is a legible repair turn rather than an `AttributeError`; each raiser names the blocklist, the reason, and legal alternatives, and each blocklist is validated against its pool at import time so a rename cannot silently unblock a member.

### 5.5 Validation

A plan is validated before it is rolled out, in two layers. `validate_actions` checks one bag: the op is allowed for the task (with a targeted hint when an edge op is emitted under a `remove_node` budget, since the harness expands deletions itself), node ids are in range, a `remove_node` on a protected outbreak source is rejected with the source list and the reason (deleting patient zero ends the outbreak rather than containing it, which is a different task, and the published immunization protocol vaccinates first and then infects a non-immunized node), edge ops carry an in-range destination, a `set_edge_weight` must name an existing arc and may only lower it (DiffIM's relaxation $\tilde p_{uv} = p_{uv} \cdot \tilde r$, $\tilde r \in [0, 1]$; raising an arc would be an unbudgeted boost dressed as a block), a duplicate budgeted key inside a bag is rejected, and the bag's budgeted units must not exceed the budget. Only the task's `budget_op` counts toward the budget: a containment plan's `remove_edge` ops are the mechanics of a node deletion and charging them would make $k$ mean $\deg(v)$ different things per node. A node op is keyed by its target, an edge op by the whole arc, so two arcs out of one hub are two interventions. `validate_plan` then adds what a single bag cannot see: a budgeted key repeated across timesteps is rejected, and the whole-plan total must not exceed the budget. A blocking task's sources are not protected (the rumour is already spreading, so cutting a source is a legitimate if late move); every other outbreak is.

The inverse tasks validate their own objects rather than reusing the action validator. A source set must be integers in range, without duplicates, and at most `budget` long (extra names would buy recall for free). A reconstruction must be a dict of `(time, parent)` pairs with times in $[0, H]$, `parent = None` exactly when `time = 0`, every parent an actual in-neighbour (otherwise path precision would be scored against edges the graph lacks), no hidden node named as node or parent, and at least one node. A prediction may be `None` (decline), must be finite, at least the observed popularity (nobody un-adopts) and at most $N$.

### 5.6 Exogenous wrappers, applied after validation

`evaluate_strategy` wraps the validated `action_fn` in exactly one cascade wrapper and then the edit stream. `Outbreak.wrap` (containment) prepends the outbreak's `add_node` bag at $t=0$ and expands every bare `remove_node` into its deletion bag (`remove_node(v)` plus a `remove_edge` in both orientations for every incident arc; without it the structured head would zero the node's infected channel and its still-present in-edges would re-infect it). `NegativeCascade.wrap` (blocking) seeds nothing, because the rumour is committed by the competitive simulator's `reset` and an `add_node` bag would start the positive cascade; it holds anything emitted before the detection delay $r$ and delivers it at $t = r$ by re-querying the policy for steps $0..r$ (deferring rather than dropping, since every library blocker commits its whole bag at $t=0$ and dropping scored every such arm at the unopposed spread), and expands removals. `Immunization.wrap` (epidemic) seeds the index cases and expands a dose into node plus arcs under `vaccinate` or arcs alone under `quarantine`. Seeds go first in the bag so a later removal overwrites an earlier seed mechanically; whether that is allowed is the validator's separate answer, and it is no. Wrapping after validation is deliberate: the outbreak's ops would be rejected under the task's allowed ops and must not count against the budget.

### 5.7 The offline rule

A generated program never receives a handle on the evaluator. `attach_context` always binds `strategy.outbreak` and `strategy.budget_op`, and binds `predict_marginals`, `step_marginals` with `transition_logprob`, and `forecast_marginals` with `expected_popularity` only when `strategy.canned` is true, which holds for library, routed and external baselines built by the harness and for nothing else. The same gate is applied inside the three inverse harnesses. This is a deliberate rule dated 2026-09-04, not a gap: conditions 3 to 6 differ in the forward model the harness scores with, and the arm's evaluator reaches a generated program only through the reward and the feedback the harness computes. A generated program may re-simulate analytically over `graph.ic_probs` (a mean-field IC pass); it cannot call a simulator. Under the native condition the canned bindings are raising stubs that name the condition and list legal alternatives, so a library member that needs a kernel fails with a repair-shaped message rather than a traceback.

### 5.8 Plan caching and re-scoring

`evaluate_strategy` caches the validated plan on the strategy as `_plan`, because a plan is a function of `(graph, budget, horizon)` and never of the evaluator seed. Re-scoring the incumbent on a fresh seed (Section 7.5) therefore pays only a rollout, never a second `plan_horizon()`. The same cache means a strategy whose `plan_horizon` is nondeterministic is frozen after its first evaluation.

---

## 6. The context the model is given

### 6.1 The conversation

A `Conversation` holds a system turn and an archive. `send(user_text)` calls the model on the current window, extracts the code block (the first fenced block that defines a class, else the first fence, else the stripped reply), and stores the assistant turn as only the re-fenced script; the verbatim reply, including the prose around the code, is kept in `transcript` and written into the results JSON, since that prose is where the model says what it was trying to do. `ask` is the prose turn used for the write-up and appends nothing to the working thread. `compact_last` replaces the last assistant turn with a diff-plus-verdict (Section 7.10). The window is the system turn, the opening task turn, and the last 10 exchanges (`default_history_exchanges = 10`, raised from 6 once assistant turns became diffs). At 20 outer iterations the middle is dropped by design: the population best and its diagnostics ride in every prompt, so an old exchange carries nothing the search still needs.

### 6.2 The gateway

`GatewayProvider` fronts the lab gateway through the OpenAI chat-completions client with the client's own retries disabled and three retries owned here (each failure prints a warning; an empty completion counts as a failure). The gateway silently drops the system role, so `fold_system` folds it into the first user turn. Sampling arguments are chosen per model family: a temperature is dropped with a warning for `gpt-6*` (which accepts none), `reasoning_effort` (default `high`) is sent for every non-Claude model, and the bearer token is chosen by model family (`CLAUDE_GATEWAY_TOKEN` for `claude*`, else `CHATGPT_GATEWAY_TOKEN`). Prompt and completion tokens are counted per provider and merged across providers (routing spins up its own); dollar cost is null unless both `--llm-price-in/out` are given.

### 6.3 The system prompt

`build_system_prompt(method, strategy_mode, task)` assembles, in order: the family rules, the method body, the exemplars, the remove-semantics note, and a horizon note. The contract is decided by the task, not the method: `evolve` on an intervention task resolves to the `one_shot` body (`plan_horizon`), on an inverse task to `localize` or `reconstruct`, on the forecast task to `predict`.

The family rules open with a brief per family that states the objective in one sentence, whether lower or higher is better, and the published traps of that literature in plain words. The seeding brief says the cascade starts from the seeds and nowhere else. The containment brief says removing an outbreak source is rejected, that unreached blockers do nothing, that hubs are what `adaptive_degree` already does, and that structural dismantling and cascade containment rank nodes differently and sometimes oppositely. The four blocking briefs (one per lever) say that arriving first is the entire game, that high degree is a trap and the published finding is that the degree heuristic cannot be used for influence blocking at all [38], that proximity is the strong cheap baseline, and that for the node lever the trivial neighbour-degree heuristic beats two principled VLDB algorithms in a fifth of their own table's cells. The four epidemic briefs say that dosing an index case is rejected, that the dynamics are not monotone (recovery, burn-out, peak), that NetShield [35] and DAVA [36] disagree by graph, that plain top-degree is a 2002 heuristic and the row to beat, and that acquaintance immunization uses no global information. The localization brief states the consistency reward and that the true sources are never shown or used to score, warns about ill-posedness and that sources rarely cluster, and names LPSI [24] as the bar. The reconstruction brief writes the reward as a formula, explains the $1/N$ source prior (each declared source costs about $\log N$), and tells the model to spend effort on parents and times because thirteen years of published work say the node set is nearly free. The prediction brief writes MSLE out, says lower is better unlike every other task, says the dynamics are not independent cascade, that temporal features dominate, that the classical bar is one line ($\log P(t_p) = \alpha \log P(t_o) + \beta$ [32]), that the branching ratio is the mechanism, and that declining is allowed and counted, not penalized.

After the brief come the shared rules: the output format (one fenced block whose first line must be `# MECHANISM: <one sentence>`, one class, nothing else at module level), the import whitelist, the available names, and the action rules per family (budget accounting, what is rejected). The remove-semantics note states what `remove_node` does under `spent` or `blocked` and is omitted for tasks that cannot emit the op. The horizon note is one of nine (seed, containment, blocking, epidemic, adaptive, temporal scheduling when an edge op is allowed, localization, reconstruction, prediction), each saying where in time the intervention belongs and why.

The method bodies describe the outer-loop contract: `plan_horizon` returns $H+1$ bags; `act` is called once per timestep, per window, or per round; the adaptive body says a static set dealt in slices is the non-adaptive strategy wearing a costume, that theory says it is hard to beat on spread alone, and states the feedback-model split (re-seeding an active node is rejected under `full_adoption` and silently dropped under `myopic`). The inverse and forecast bodies say the program is called once per instance and must be an algorithm, not a fit to one cascade.

Two worked exemplars close the system prompt for every free-mode family, chosen so that beating both is the stated bar: for IM a community-discount greedy and a lazy-greedy RIS coverage; for containment an outbreak-exposure cut and a reachable-set HDA; per blocking lever a race-ahead arrival heuristic and a narrowed proximity, a live-edge cut-point counter and the neighbour-degree heuristic, a carried-traffic arc score (and its `set_edge_weight` twin); for epidemic an outbreak-dominator score and a reachable NetShield; for localization a vectorized LPSI with per-community maxima and a mean-field re-simulation refinement; for reconstruction a likelihood tree and a delayed-BFS start refined by analytic likelihood; for prediction a temporal Szabo and Huberman correction and a branching-ratio blend. `per_step`, `windowed` and `adaptive` get a reply-shape snippet instead of exemplars.

### 6.4 The opening user turn

`build_user_prompt` emits, in order: the task line with the objective sentence for the family; `diffusion_model`; `budget` with its percent of $N$ and the budget unit (seeds, removals, doses, counter-seeds, arc cuts, or unused); `horizon`; the allowed ops (or "none: this task emits no actions"); then up to five conditional blocks; then the graph profile; then the library reference; then the closing instruction.

The conditional blocks: the outbreak block (the source list up to 60 ids, labelled rumour seeds, index cases or outbreak sources, marked as fixed and available as `self.outbreak`; for the epidemic task the dynamics line with $\beta$, $\gamma$, $\alpha$ and the mean infectious period; for blocking the tie-break rule in words and the detection delay); the round block (the exact timesteps and batch sizes at which `act` is called, and what the policy observes under the feedback model); the observation block (source localization: episode counts, source and infected ranges, and whether the observation is a binarized draw or a Monte Carlo marginal); the mask block (reconstruction: which of the four protocols is in force and what it implies, plus the instance ranges and the reward restated); the cascade block (prediction: observed and final size ranges, the growth ratio distribution, and the two floors, the geometric mean and persistence, which any algorithm that ties with has not used the instance).

The graph profile (`tools/graph_profile.py`) is aggregate structure only and never an adjacency list, so the model cannot hardcode node ids: node and arc counts, directedness, the total-degree distribution (mean, median, max, p90, p99, coefficient of variation, and a heavy-tail verdict at CV above 0.7, a threshold measured on BA 0.95, SBM 0.46, ER 0.40, WS 0.11), density, components with the largest's share and a warning when budget cannot cross components, the maximal k-core, reciprocity for directed graphs, and, when the graph has at most 200,000 arcs, average clustering, degree assortativity, and label-propagation communities with modularity and the five largest sizes (a note recommends allocating across communities when modularity exceeds 0.3); above the cap clustering is sampled with 2,000 trials and the rest is skipped. The IC transmission lines give the edge-probability range and the median expected out-transmission per node with the fraction at or above 1.0, which says whether the typical node replaces itself.

The library reference (`tools/library_api.py`) lists every callable member of the task's pool by introspected signature and first docstring line, names the blocked members with the reason, prints the 16 advertised primitives, and for IM additionally prints three full library sources (`degree_discount`, `cofim`, `degree_ris_refine`) as worked idioms. Each pool's reference names the row that has to be beaten and why: `adaptive_degree` for containment; `proximity` for blocking (with `degree_blocking` listed as the published failure mode); `degree_immunization`, `netshield`, `dava` and `acquaintance_immunization` for epidemic control; `lpsi` for localization, with a second block of per-node scorers for `source_scores`; `delayed_bfs` for reconstruction, with `personalized_pagerank` listed because a plain random walker beats tree sampling on assortative graphs; `szabo_huberman` and `mean_size` for prediction, with the 21-key `cascade_features` dictionary and the note that the reshare rate in the second half of the window is the single best published feature. The blocking and immunization references are filtered by lever, because an edge selector shown to a counter-seeding arm returns arcs and the resulting program would be rejected. In scored mode the reference is replaced by a menu of ideas that are not callable.

The two search methods append two more things to this turn: the anchor leaderboard (Section 8.4) and the probe contract (Section 8.7). `per_step` and `windowed` get neither.

---

## 7. The search over programs

### 7.1 The methods

Five outer-loop methods implement `optimize(agent, environment, task, graph) -> (Strategy, Trajectory)`. Two have a refinement loop, an anchor leaderboard, paired seeds, an acceptance rule, credit, probes and checkpoints: `evolve` (the default, and `adaptive`, which is `EvolveSearch` with a different label and a per-round contract) and `one_shot`. Two are single-rollout arms with no incumbent, no seed control and nothing to resume: `per_step` and `windowed`. Routing (condition 2) is not a method but a single call that picks a canned script. The rest of this section describes `evolve` in full and the others by their differences.

### 7.2 Startup

The system prompt is built for the method, mode and task (Section 6.3). Unless a checkpoint carries it, the anchor leaderboard is computed by rolling every classical baseline of the task's anchor list through the arm's own evaluator at the same budget, horizon, outbreak, edit stream and removal expansion (Section 8.4); its text is appended to the opening turn together with the probe contract, and the best anchor's trajectory is kept for the reference diff. The opening turn is sent once and then rides in every window. A checkpoint (`<result>.ckpt.json`) is honoured only if its fingerprint matches every setting that changes what the reward measures (method, mode, evaluator, model, temperature, dynamics, budget, horizon, iterations, ops, sample counts, seed, graph size, the native flag, the select splits and every task-family knob); a mismatch prints the differing keys and starts fresh. On resume the thread, population, incumbent, stagnation, history, memory, probe log and anchor are restored, so a killed sweep never pays for a finished generation again.

### 7.3 The six operators

Each generation applies one operator to produce one candidate. Three exploit the population best (the parent); three explore.

| Operator | Kind | Instruction to the model (abridged) | What is shown |
|---|---|---|---|
| `refine` | exploit | a small, targeted improvement to the parent: adjust one term, fix one weakness the diagnostics expose | parent script and diagnostics, plus up to 2 alternative scripts from the population |
| `parameters` | exploit | change only numeric constants (weights, thresholds, sample counts, radii); same functions, same control flow | parent only |
| `simplify` | exploit | remove components the diagnostics do not justify; the child must be shorter and score within the noise band, and that counts as success | parent only |
| `crossover` | explore | combine the parent's mechanism with a partner's; the child must contain an identifiable part of each | parent and partner, both with diagnostics |
| `synthesize` | explore | read every strategy shown and write one new strategy taking the best-supported idea from each; must differ from every one shown and from every library algorithm | the top 3 of the population with diagnostics, plus the chosen idea |
| `from_scratch` | explore | do not edit any program shown; implement the chosen idea as a new strategy; must not re-implement a library algorithm or a mechanism already in the population | the chosen idea only |

The operator families are EoH's [9] (its `i1`, `e1`, `e2`, `m1`, `m2`, `m3` are initialization, two crossovers, and three mutations that respectively restructure, re-tune parameters and simplify), MCTS-AHD's five-operator set [11], and LLaMEA's mutation prompts [12]; `simplify` with a shorter-and-not-worse acceptance is the parsimony pressure LLaMEA and OpenEvolve [13] describe. The static weights are `refine 3`, `parameters 1`, `simplify 1`, `crossover 2`, `synthesize 1`, `from_scratch 1`.

### 7.4 Exploration schedule and operator choice

Let $g$ be the generation index, $G$ the total (`--outer-iters`, default 20), and $\text{stag}_g$ the number of consecutive generations without an accepted candidate. The exploration weight is

$$\lambda_g = \min\Bigl(1,\; 0.8 \cdot \frac{\max(G - g, 0)}{G} + 0.15 \cdot \text{stag}_g\Bigr),$$

and is forced to 1 when $\text{stag}_g \ge$ `stagnation_patience` (default 2). This is MCTS-AHD's budget-aware exploration constant, $\lambda_0 \times$ the remaining evaluation fraction, with a stagnation bonus added. The operator is drawn from a categorical distribution with unnormalized weight $w_o \lambda_g$ for an explore operator and $w_o (1 - \lambda_g)$ for an exploit operator; `crossover` and `synthesize` are removed when the population has fewer than two members; if every weight is zero, `refine` is used. The generator is `np.random.default_rng([seed, g])`, so the operator sequence is reproducible per run. The first generation is always the seed turn (the opening prompt), and a generation whose population is empty re-seeds with the last error appended.

### 7.5 Parent, partner and inspirations

The population is every accepted-or-not scored candidate as a record `(iteration, script, mechanism, reward, plan_seconds, summary)`. Every generation ranks it best-first under the task's sense; the parent is the best, never the last attempt. For `crossover` the partner is drawn from the rest with rank-weighted probability $\propto 1/(\text{rank} + 2)$, EoH's selection rule [9]. For `refine` the top two others are shown as alternatives (scripts only). For `synthesize` the top three are shown with diagnostics.

### 7.6 Idea search before code

For `from_scratch` and `synthesize` a separate one-shot call precedes the code call. Its prompt asks for three distinct mechanisms, shows the task's library menu (one line per algorithm, marked as already available to everyone so re-implementing one is not new), the mechanism line and reward of every population member, the search memory and the attempts table, and asks for JSON with the ideas, a 0 to 10 novelty score per idea (10 = shares no mechanism with the library or the population), the chosen index and a one-sentence reason. The chosen idea is spliced into the code prompt. Only `ideas` and `chosen` are read; the scores and reason are for the model's own deliberation. This is DeepEvolve's plan-then-implement split [15] and a lightweight form of novelty search [17].

### 7.7 The generation prompt

Every generation after the seed sends one user turn assembled by `build_evolve_prompt`, in this order: the opening line; the search memory (omitted when empty); the attempts table; the population table; the verdict on the model's own last edit (so a candidate that did not become the parent still gets a verdict); the programs the operator calls for, each with its diagnostics; the last error if the previous attempt failed; the operator instruction; and the reminder that the first line inside the code block must be `# MECHANISM: <one sentence>`. Pending probe answers from the previous generation are appended after that. The attempts table (OpenEvolve's previous-attempts table [13] with ReEvo's hints [10]) has one row per generation with iteration, operator, reward, delta against the incumbent, accepted or not, the mechanism sentence and the reflection hint; when longer than 24 rows it keeps the first and the last 23. The population table lists every surviving candidate with reward, plan seconds, code lines and mechanism, sorted best-first.

### 7.8 Paired evaluation

Each generation $g$ uses realization seed $\xi_g = \text{seed} + g + 1$. The candidate is scored on $\xi_g$, and the incumbent is re-scored on the same $\xi_g$ (`rescore`, which pays only a rollout because the plan is cached), so every comparison is paired within a generation and never across generations, the common-random-numbers protocol [20]. This replaced scoring the whole search on one fixed realization, which let the search overfit that realization by up to 30 nodes: on `epidemic_control/primary_school` the search reported 40.6 at $k = 48$ on seed 42, the same program re-scored 67 to 76 on seeds 43 to 45, and the referee said 63.7. `rescore` is a no-op for the inverse and forecast families, whose rewards have no realization to overfit and whose evaluation is the expensive half.

### 7.9 Acceptance

With candidate trajectory $\tau$ and incumbent $\tau^*$ on the same seed, $\Delta = R(\tau) - R(\tau^*)$ and the band $b = \max(\text{se}(\tau), \text{se}(\tau^*))$. The candidate is accepted when it improves on the incumbent by more than $b$ in the task's sense, or, for `simplify` only, when its source is strictly shorter and the incumbent does not improve on it by more than $b$ (a shorter tie wins). Both trajectories were measured on the same realization, so the band is that comparison's own noise rather than an arbitrary epsilon. Note that the prose the model reads reports a different number, the two-sigma band $2\sqrt{\text{se}(\tau)^2 + \text{se}(\tau^*)^2}$ on the difference (Section 8.2), while the attempts table records the acceptance band.

### 7.10 Reflection and memory

After every scored non-seed generation whose candidate differs from the incumbent, one extra call plays ReEvo's reflection [10]: the reviewer system prompt, the worse and the better program of the pair (ordered by the acceptance decision) with their diagnostics truncated to 1,500 characters, and the prior memory; it must answer with JSON holding a hint (at most 20 words: what made the better one better, as a design rule) and a revised memory (at most 50 words: the prior memory revised with this hint, keeping only rules the evidence still supports). The hint lands in the attempts table and the history; the memory replaces the running memory, is checkpointed, is shown at the top of every subsequent prompt, and is written into the results JSON. This is ReEvo's short-term and long-term reflection collapsed into one call, with Reflexion's verbal memory [18] as the running store.

### 7.11 Stagnation, history and thread compaction

An accepted candidate resets the stagnation counter and becomes the incumbent. A rejected exploit candidate increments it; a rejected explore candidate resets it, because an explore move opens a fresh refinement window even without improvement. Every generation appends a history record (iteration, reward, incumbent reward, operator, mechanism, delta, band, accepted, hint, script, parent iteration, plan and rollout seconds), which is what the write-up (Section 9.2) renders as diffs. The assistant turn just sent is then replaced in the working thread by a one-line verdict (operator, reward, delta, band, accepted or rejected, hint) followed by the mechanism line and a unified diff of the script against its parent (two lines of context, truncated at 200 lines; the full script only when there is no parent), about a tenth of the tokens; the verbatim reply survives in the transcript. Finally the pending probes are answered against this generation's executed bags and the checkpoint is written.

### 7.12 Failure handling

A candidate that fails to build, times out, raises, or fails validation records a history row with the error, increments stagnation, compacts its turn with a `FAILED` verdict, answers any probes against the incumbent's bags, checkpoints and moves on; the next prompt carries the error text. Unlike `one_shot`, a failed generation is a generation. If every generation fails the method raises and the pipeline records the arm as skipped rather than killing the sweep.

### 7.13 The loop, as pseudocode

```
population, best, memory, stagnation = [], None, "", 0
for g in 0..G-1:
    if population is empty: operator = seed; user = opening turn (+ last error)
    else:
        operator = choose(rng(seed, g), lambda_g, |population|, stagnation)
        parent = best of population; partner / inspirations / everyone per operator
        idea = idea_search(...) if operator in {from_scratch, synthesize}
        user = evolve_prompt(operator, parent, ..., memory, attempts, population, idea)
    user += pending probe answers
    script = conversation.send(user); probes = parse_probes(reply)
    try:
        strategy = build_strategy(script)                      # sandbox, contract check
        tau = evaluate(strategy, seed + g + 1)                 # validate + wrap + rollout
        best = rescore(best, seed + g + 1)                     # paired incumbent
    except StrategyError as e: record failure; stagnation += 1; continue
    summary = summarize(tau) + paired_delta + reference_diff + credit
    population.append(record(script, tau, summary))
    accepted = accepts(tau, best.tau, sense, operator, script, best.script)
    hint, memory = reflect(worse, better, memory)
    if accepted: best = (strategy, tau); stagnation = 0
    elif operator is explore: stagnation = 0
    else: stagnation += 1
    history.append(...); compact_last(diff + verdict); answer probes; checkpoint
return best
```

### 7.14 `one_shot`

The single-thread refinement loop. It runs `while evaluations < outer_iters` with two counters, so a failed script (a repair turn) does not consume an evaluation; repairs have their own budget of 3 (0 for a canned script, whose repair would re-send the identical script). The realization seed is `seed + evaluations + 1`. The feedback turn shows the reward, the paired delta against the previous best, the trajectory summary, the reference diff, the credit report, the script that produced the result, and, when a better script exists, the incumbent script labelled as the one to edit ("the attempt that just ran is shown so you can see what did not work: do not build on it"); this echo is deliberately redundant so it survives window trimming. Acceptance uses the same rule as `evolve` with the operator fixed to `refine`, so the simplify clause never fires. Its history records `script` and `parent_iteration = best_iteration`, so its write-up renders diffs too.

### 7.15 `per_step` and `windowed`

`per_step` runs one episode with one LLM call per (ensemble sample, timestep): at $t = 0$ the thread is reset and the opening prompt re-sent with the current state; at $t > 0$ the turn carries the state, the remaining budget and the already-seeded set; each turn rebuilds a strategy and calls `act(state, graph, t)`. Budget is enforced per episode through a seeded set, since otherwise a policy could legally emit `budget` seeds at every one of $H + 1$ timesteps. `windowed` designs one online algorithm in a single turn and the harness consults `act` at window boundaries ($\lfloor (H+1)/\text{windows} \rfloor$ steps apart, default 3 windows), validating each call against the per-window budget; its `effective_budget` is `budget × boundaries` and is written into the results so the sweep table is not read as equal-budget. Neither uses a seed override, an edit stream, an anchor, credit, probes or a checkpoint.

### 7.16 `adaptive`

`adaptive` is `evolve` with a per-round `act` contract. The dispatch happens in `evaluate_strategy` off `TaskSpec.rounds`: the harness builds the round batches and an `action_fn` that calls `act` only at scheduled timesteps and returns an empty bag between them, so the cascade runs and the next round has something new to observe. Section 11.1 describes the round machinery.

---

## 8. Feedback

Every scored candidate produces a feedback bundle that is stored in its population record (so it reaches any later prompt that shows that candidate as parent, partner or inspiration) and, under `one_shot`, is sent directly in the next turn. The bundle is the trajectory summary, the paired delta, the reference diff and, when enabled, the credit report; probe answers arrive one generation later. Nothing in the bundle is computed from a label.

### 8.1 The trajectory summary

For an intervention task `summarize` prints, in order: the reward with its standard error under the family's name (`final_spread`, or `final_infected (LOWER IS BETTER)`, or `final_rumour_size (LOWER IS BETTER)`), the number of steps and the per-step infected counts; the per-step frontier counts; the counter-cascade counts under blocking; and, when the last frontier is empty, the step at which the cascade died with the note that actions scheduled after it did nothing. Then one of three blocks.

Seeding block: the chosen seeds with degree and community id (up to 40); adjacent seed pairs, flagged as overlapping neighbourhoods and likely redundant budget; the unreached nodes (ensemble $P(\text{infected}) < 0.10$) as a count and a list of up to 20 ordered by estimated residual gain (a high-degree node the cascade never reaches is usually unreachable, whereas a high-gain one is worth a seed), each with degree and gain; community coverage (seeds per community, size, mean reach, for the eight largest communities, and how many communities with more than one node got no seed at all); and the ten highest-value unseeded nodes by residual gain, or the statement that no unseeded node has meaningful residual gain.

Containment block: the fixed outbreak sources with degrees; the removed nodes in removal order; the removal distance from the outbreak (how many removals sit at 1, 2, 3 hops and how many farther, with the note that those can only help if the cascade reaches that far); and the nodes the cascade still reaches ($P \ge 0.50$), top by degree.

Blocking block: the rumour seeds; the lever picks in order; under counter-seeding the reachability of the picks from the rumour (how many sit within 3 hops of $S_N$ and how many are adjacent to a source, with the note that a seed the rumour never reaches saves nobody and one it reaches first saves nobody either, because prevented influence counts only nodes that would otherwise have been infected); and the nodes the rumour still reaches.

The residual gain behind the seeding block is CELF's marginal gain at RIS cost [2, 5]: for IC on graphs up to 200,000 nodes, $\theta = \min(20{,}000, 10N)$ reverse-reachable sets are sampled once per graph (seed 0) and cached; with $C(S)$ the set of RR sets covered by the seeds, the gain of an unseeded $v$ is $|C(\{v\}) \setminus C(S)| \cdot N / \theta$. It never touches the metered evaluator. Communities are label-propagation communities cached on the graph. The three non-intervention families have their own summaries, described with their rewards in Section 10.

### 8.2 The paired delta

`paired_delta` prints the change against the incumbent on the same realization, the direction ("more is better" or "fewer is better"), the two-sigma band $2\sqrt{\text{se}_c^2 + \text{se}_i^2}$, and one of three verdicts: inside the noise ("this change did nothing measurable, so do not read anything into its sign"), a real improvement ("keep what caused it"), or a real regression ("undo what caused it"). Two decimals for node counts, four for F1, the reconstruction score and prediction errors, where a 0.02 move is large.

### 8.3 The reference diff

When an anchor exists, `reference_diff` compares the candidate's per-node ensemble marginals with the best anchor's on the same evaluator and lists only decisive flips: nodes the reference reaches at $P \ge 0.50$ that the candidate leaves under $0.10$, and the reverse, each sorted by degree with up to ten entries carrying degree and both probabilities, followed by the net expected difference summed over every node. The labels flip with the sense (under minimize, "it fails to protect n nodes you save" and "you lose n nodes it protects"). The gap between the two thresholds keeps threshold-straddlers out of the lists.

### 8.4 The anchor leaderboard

Before the first generation the harness rolls a short list of classical baselines through the arm's own evaluator and prints `REFERENCE SCORES: ... run on THIS graph, under THIS evaluator, at the same budget and horizon. Beating the top row is the bar` (or "LOWER is better here"), one row per baseline with reward and standard error. The lists are deliberately short because under `@monte_carlo` those episodes are charged to the arm: IM `high_degree`, `degree_discount`, `pagerank_seeds`, `imm` (IC only, since RIS does not describe LT), `random_seeds`; adaptive IM adds `adapt_epic`, `adapt_degree_discount`, `static_split` (AdaptGreedy is omitted for cost); containment `adaptive_degree`, `corehd`, `collective_influence_removal`, `netshield`, `random_removal`; epidemic control per lever (`degree_immunization`, `netshield`, `dava`, `acquaintance_immunization`, `random_immunization` for the node levers; `netmelt`, `product_degree`, `frontier_edge_cut`, `random_edge_cut` for the arc levers) plus the `no_immunization` floor; blocking per lever (`proximity`, `reverse_blocking`, `rps`, `degree_blocking`, `random_blocking` for counter-seeding; `imin_lhga`, `imin_lsbm`, `advanced_greedy`, `degree_blocking`, `random_blocking` for node blocking; `kimura_link_blocking`, `out_edge_blocking`, `random_edge_blocking` for the arc levers) plus the `no_blocking` floor with prevented influence per row; localization `lpsi`, `netsleuth`, `jordan_center`, `dmp_localize`, `infected_degree`, `random_sources` with over- and under-explained counts; reconstruction `delayed_bfs`, `personalized_pagerank`, `consistent_tree_wpct`, `cult`, `dhrec`, `observed_only`, `random_reconstruction`; prediction `szabo_huberman`, `feature_linear`, `seismic`, `hawkes`, `branching_factor`, `persistence`, `mean_size` with MALE, MAPE, PCC and declines per row and a header saying the score is an error. Every anchor goes through `evaluate_strategy`, so it faces the identical outbreak, edit stream and removal expansion; an anchor rolled out bare would face no outbreak on a containment task and post an unbeatable zero. The best anchor's name and trajectory are kept for the reference diff, and the anchor is checkpointed so a resume does not re-pay for it.

### 8.5 Per-action credit

Under `@oracle` and `@world_model` (never `@monte_carlo`, where it would cost $(k+1) \times$ `mc_runs` real episodes; never for the inverse and forecast families, which emit no actions) `credit_feedback` computes a leave-one-out delta per budgeted action: the executed plan and its $k$ ablations, each with one action removed (only actions whose op is in the task's allowed ops are ablated, so a stream's exogenous edge edits are never credited to the policy), are scored in one batched call of up to 512 sample blocks, and the report prints `delta = spread lost if that single action is removed; ~0 = wasted budget` per action. `augment_solo` adds each action's solo cascade (the one-bag plan of that action alone; a blocked removal travels with its incident edge ops) with its final size and the step at which it stopped growing (one past the last step whose count moved by at least 0.5), read off the per-block curves at no extra rollouts. `frontier_bottlenecks` then asks the model's own kernel one question: with the whole reached region ($P \ge 0.50$) placed on the frontier, which unreached nodes ($P < 0.10$) would be caught in one more wave; the top eight with $P \ge 0.05$ are printed as the front the intervention is closest to leaking through (containment) or the frontier the cascade dies before crossing (seeding). Credit runs at the base seed. The batched path stripes samples across plans inside one call, so two plans do not share realization streams there; the residual noise is the ensemble standard error the loop already lives with.

### 8.6 Probes

Beside its code block the model may add one fenced `probes` block with at most six what-if questions about the plan its script produces this iteration; the answers arrive with the next iteration's feedback and every probe is metered evaluator work. Three ops: `drop(node)` returns the plan reward and the reward without that node's actions (its marginal contribution, two rollouts); `swap(a, b)` returns the reward if `b` replaces `a` (two rollouts); `region(nodes)` returns the expected probability mass the plan captures inside a node set of at most 300 (one rollout). Batched environments answer drop and swap in one batched call. Invalid JSON, a wrong shape, an unknown op or a failing probe produce a note rather than an error, and more than six are truncated with a note. Under the native condition the contract is never shown and a hallucinated block is answered with "probes are unavailable under the native condition: no forward model may be consulted during this search". The probed plan is the bags the last candidate actually executed, or the incumbent's when the generation failed; every answer is logged with its rollouts and seconds and written into the results JSON.

### 8.7 Repair turns

A `StrategyError` message is the feedback. The executor's messages are written for that purpose: the import whitelist and the pre-bound names for an illegal import; the blocklist, the reason and legal alternatives for a blocked member; the usual causes for a timeout; the wrong-entry-point diagnosis for a `None` return; the sources and the reason for a protected removal; the contract text for a missing entry point; and, for the round-based policy, the exact rule that was broken (re-seeding an active node under full-adoption feedback, over-committing a round).

### 8.8 What the model is never shown

The true sources of a localization instance, the true history of a reconstruction instance, and the logged final size of a prediction cascade never enter a prompt, a reward or a selection decision; the label metrics are computed on the winner after the search (Section 9.3). The model is never shown an adjacency list or the referee's numbers, and it never sees the held-out split.

---

## 9. After the search

### 9.1 The winner and the held-out re-run

`optimize` returns the incumbent, the best over generations, not the last turn. For the inverse and forecast families the unmodified winner is then re-run on the disjoint evaluation split (`--sl-eval-split`, `--cr-eval-split`, `--cp-eval-split`, default `test` against a `train` selection split); the held-out number overwrites `reward`, the search's number is kept as `selection_reward`, and `generalization_gap = reward_heldout - reward_selection` exposes memorization (for prediction a positive gap means worse on unseen cascades). A fitted predictor keeps `fit_examples` pinned to the selection pool on the held-out run, so it may calibrate on the cascades the search saw and never on the ones it is scored against. Condition 9's fitness scorer reads `selection_reward` and never the held-out split.

### 9.2 The write-up

For every non-canned arm the thread is asked one prose question (`build_explanation_prompt`) whose reply is stored as `explanation` and printed under an `ALGORITHM WRITE-UP` banner. The prompt echoes the winning script (the winner is the max over iterations, not the last turn, and the thread is windowed) with its rollout diagnostics, then every iteration with the exact change it made rendered as a unified diff against the script it edited (from the history's `script` and `parent_iteration`, two context lines, capped at 120 diff lines; a from-scratch script is given in full; an unchanged resubmission is said so), each labelled with its reward and the incumbent re-scored on the same realization, its operator and its parent, plus a trajectory line (first scored attempt and its reward; the winner's reward). It then demands exactly seven headings in order: Summary; Iteration log (read the change off the diff, be specific, never invent a change); From first attempt to winner; Closest classical algorithm; How the final algorithm works (name every library algorithm called, prefix the model's own ideas with `[new]`, skip bookkeeping, no libraries, no quoted code, no variable names); What is new and why it wins; Limitations. The three middle headings answer the three questions the lab's reviewer asked of the first write-ups.

### 9.3 Label metrics, computed only now

Source localization: precision, recall, F1, accuracy and AUC of the winner's sets against the stored sources on both splits, with AUC from `source_scores` when the program wrote one and otherwise from the rank of the returned list (which ties every un-nominated node; which rule was used is recorded as `auc_source`), plus `f1_generalization_gap`. Cascade reconstruction: node and event precision, recall and F1, Matthews correlation, timing MAE and NRMSE, source precision and recall, path precision and recall, Jaccard, order accuracy, and the reported `tree_score = λ·PathPrecision + (1-λ)·EventF1` with $\lambda = 0.6$, plus `tree_score_generalization_gap`; and `trivial_decoder_reward`, the score of "everyone reachable from the reports by BFS" under both the reward and the tree score, a required check that a trivial decoder scores badly. Cascade prediction: the full 22-metric block (Section 10.7) and the two floors, `trivial_predictor_error` (the geometric mean of the training sizes, the exact minimizer of MSLE over instance-blind constants) and `persistence_error`.

### 9.4 The referee replay

Every arm's final number is replayed on one shared referee, the exact oracle simulator at `--referee-samples` (default 1,000; 200 left a standard error above the gaps the smokes compared), built by `_build_referee` from the arm's config with the evaluator swapped and the sample counts raised, wrapped in the multi-round environment when campaigns exceed one. Three branches:

- Intervention tasks replay the executed action bags (`trajectory.actions`), never a re-planned set: a sampling script returns a different set on a second call, and a `per_step` re-plan would fire a fresh LLM call per timestep. A canned arm whose own evaluator is the referee at no fewer samples reuses its evaluation as the replay (the rule the pipeline sets up for every non-agent arm). Written: `referee_reward`, `referee_reward_se`, `referee_seed`, `referee_spread_pct`, `referee_rollout_seconds`, `arm_minus_referee`. Blocking and epidemic control re-measure both terms of their prevented quantity on the referee (`referee_unopposed_spread`, `referee_prevented_influence`; `referee_unprotected_attack_rate`, `referee_prevented_infections`, `referee_prevalence_curve` and its curve metrics), because a difference of two evaluators' numbers is not a quantity.
- Localization and reconstruction re-measure the reward under the referee's own oracle: `referee_reward` is the mean consistency of the stored sets re-simulated on the referee (with `resim_error_true_sources` beside it, since on an ill-posed problem a recovered set can reproduce $y$ better than the truth did), or the mean log-likelihood of the stored histories under the referee's kernel plus a re-simulation of the recovered roots.
- Prediction has no replay: its reward is already exact, and `referee_reward` is the held-out reward. With `--mc-agreement` it computes instead the modelling-error check of Section 10.7.

### 9.5 Monte Carlo agreement

`--mc-agreement` (optional, meant for the critical datasets) replays the same bags on an NDlib environment with the same dynamics (`--mc-agreement-runs`, default `mc_runs`) and writes `mc_reward`, `mc_reward_se`, `mc_rollout_seconds`, `mc_seed` and `referee_minus_mc`, the independent check on the oracle and the per-rollout timing row the efficiency claim is read on. For the inverse tasks the same referee measurement is repeated on NDlib.

### 9.6 The world-model re-evaluation

For a non-canned arm under `@world_model` or `@oracle` the winner's bags are rolled out three more times on the arm's own evaluator at seeds `seed + 1`, `seed + 2`, `seed + 3`, and `wm_reeval_mean - referee_reward` is written as `wm_reeval_minus_referee`. The arm's `reward` is a max over iterations at one rollout seed, so it carries selection optimism (the winner's curse) plus that seed's luck; evaluator fidelity is judged by `wm_reeval_minus_referee`, not by `arm_minus_referee`.

### 9.7 The results JSON

One JSON per arm and budget. The base block: task, method, evaluator, model (or `baseline:<name>`, `routing:<name>`, `canned`, `external:<name>`, `discovery:<name>`), graph statistics, budget and percent, `effective_budget`, reward and percent of $N$, summary, script, explanation, the full LLM transcript, token usage, cost block (with the rollout seed), base seed, history (every generation), probes with `probe_calls` and `probe_seconds`, `final_marginals` (dropped above 200,000 nodes), `real_env_episodes`, `evaluator_calls`, `evaluator_seconds`, `forward_passes`, the timeline (bag and resulting state per timestep of the representative sample), `spread_curve`, `spread_at_horizon`, `selection_reward`, `objective` (the sense), and the running evolve `memory`. The four cost fields are read before credit and the referee run, so post-hoc analysis is never charged to the search. Then a family block (prediction protocol and metrics; reconstruction setting, rate, weight and kernel counts; localization mode, budget mode, AUC source, forward and scoring calls; blocking model, tie-break, attacker and prevented influence; epidemic rates, curves and spectral context; containment outbreak, ring, `ring_fits` and structural functionals), conditional extras (streaming, multi-round, adaptive schedule and per-round spreads, the routing reply), the credit block, the referee block, the agreement block, the re-evaluation block, elapsed seconds and the arm stamp. The checkpoint is deleted on success, so a checkpoint on disk always means "did not finish".

### 9.8 Report and plots

The report and the figures read only `ground_truth_reward`. Beside the algorithm-discovery figures (budget against spread, condition comparison, evaluator fidelity, cascade, runtime, convergence) every task adds its own: dismantling curve and structural-versus-spread for containment; prevented influence and blocking ratio for blocking; epidemic curve and eigendrop-versus-attack for epidemic control; localization metrics, cost and generalization gap; reconstruction metrics and tree-versus-node; prediction metrics, popularity scatter and cost. The report's world-model section covers provenance, every evaluation block and in-loop evaluator fidelity.

---

## 10. The task harnesses and their rewards

### 10.1 Influence maximization and adaptive IM

Reward: the ensemble-mean final infected count of a plan committed at $t = 0$ (Section 1.3). The program returns $H + 1$ bags; the seeding timing note tells it to put every seed in element 0. Every result carries `spread_curve`, so $\sigma(S, T)$ is readable at any $T \le H$. The adaptive variant keeps every part of this and changes only the evaluation loop (Section 11.1).

### 10.2 Critical node detection (reactive containment)

An exogenous outbreak of $\max(1, \lfloor 0.10 N \rceil)$ sources (default selector `random`, deterministic in the seed) is injected at $t = 0$ by the wrapper; the budget buys bare `remove_node` deletions that the harness expands into node plus incident arcs after validation. Reward: the outbreak's final infected count under the removals, lower is better. Removing a source is rejected (Section 5.5). Budget counts distinct `remove_node` targets. The one-hop ring $|N_1(S) \setminus S|$ is recorded as `outbreak_ring` with `ring_fits = budget >= ring`, because at a budget at or above the ring every outbreak-aware arm deletes the ring and scores exactly $|S|$ (the first `power_grid` sweep had three of four budgets there, at a 1 percent outbreak whose ring was 129 nodes, which is why the default outbreak is 10 percent). `frontier_removal`, the ring highest-degree first, is in the default pool as the control every outbreak-aware arm has to beat: the first sweep's agent arm scored exactly the source count against 98 to 160 for every published dismantler, and its whole program was this one-hop ring, so without that row the gap reads as a method and is in fact the information asymmetry. Library dismantlers meet the no-deleting-the-source rule in `removal_plan`, which drops sources and duplicates and tops the set back up by degree. The connectivity functionals of the dismantling literature (pairwise connectivity, largest component, component count, Schneider's $R$, ANC, $\rho$ at the 1 percent threshold, and `degree_rank_spearman`, the correlation of the removal order with degree that says whether an arm re-derived the degree heuristic) are computed exactly on the executed removal set and printed as context, never as the score, because the same centralities rank in opposite orders under a spreading and a connectivity objective.

### 10.3 Influence blocking

A rumour $S_N$ of $\max(1, \lfloor 0.01 N \rceil)$ nodes is committed by the competitive simulator's `reset` and is already spreading; `--blocking-lever` chooses what one unit of budget buys, and sets `budget_op` and `allowed_ops` together: `counter_seed` (`add_node` seeds the positive cascade; Budak et al. [38a], CLDAG [38b], RPS [38c]), `node_block` (`remove_node`; SandIMIN and the Xie line), `edge_block` (`remove_edge`; Kimura et al. [38d]), `weight_block` (`set_edge_weight` to 0.0, the $\tilde r = 0$ end of DiffIM's continuous relaxation). Reward: the rumour's remaining final size, lower is better. The reported quantity is prevented influence $\sigma(S_N, \emptyset) - \sigma(S_N \mid \text{blockers})$, with the unopposed term measured by one empty-plan rollout on the same evaluator (charged to the arm), so a blocker protecting nodes the rumour never reaches scores exactly zero; `prevented_pct_of_unopposed` and the informative axis `budget_ratio = |S_P| / |S_N|` are written beside it. The tie-break (`negative`, `positive` or `fixed` dominance; `auto` resolves to positive dominance under IC, Budak's convention, and negative dominance under LT, the competitive-LT convention) and the detection delay $r$ are stated in the prompt and recorded with the result. The weight lever's validator cap is the arc's own probability. `proximity_ring` (BFS from $S_N$ by out-neighbours, highest degree within a hop) and `exposure_scores` (damped propagation from the sources, 4 hops at decay 0.5) are shared building blocks used by the proximity family, the diagnostics and the anchors, because the most common way a blocker wastes budget is protecting nodes the cascade never reaches.

### 10.4 Epidemic control

An outbreak of $\max(1, \lfloor 0.01 N \rceil)$ index cases is injected at $t = 0$ by the wrapper into an SIR, SIS or SEIR process on our own per-arc stepper (`data/wm_epidemic.py`) with `--epi-beta` (arc multiplier, default 1.0), `--epi-gamma` (recovery, default 0.3; 1.0 under SIR reproduces IC exactly) and `--epi-alpha` (SEIR incubation, default 0.5). `--epi-lever` maps to ops with one wrinkle, two levers both spend `remove_node`: `vaccinate` expands a dose into node plus arcs (immune, uncounted; NetShield, DAVA, Pastor-Satorras and Vespignani, Cohen), `quarantine` into arcs alone (isolated, still in the graph, still counted; the "recovered is not removed" trap made into a lever), `edge_cut` is `remove_edge`, `contact_reduce` is `set_edge_weight` to `--contact-reduction` times the arc (default 0.0, so it is directly comparable to `edge_cut` at the same $k$). Contact tracing and quarantine with a duration are out of scope with stated reasons (the first changes the observation, the second is hidden state a Markov head cannot represent). Reward: the attack rate, the number of nodes ever infected, lower is better; under SIS, which has no terminal state, the report reads endemic prevalence instead. Prevented infections are measured against an empty plan that is wrapped (the outbreak is injected by the wrapper, so an unwrapped empty plan would start no epidemic and silently make every prevented number negative). The curve metrics (peak prevalence, time to peak, area under the infectious curve, endemic prevalence after a burn-in fraction of 0.5) are computed from the prevalence curve because a good policy flattens rather than eliminates and a terminal number alone can rank two policies backwards. The eigendrop $\lambda_1^{\text{intact}} - \lambda_1$ is computed for the node levers and printed as context, never as the score, since a method can win on eigendrop and lose on final size. `dava` and `netshield` in our library produce byte-identical node sets to the authors' repository.

### 10.5 Source localization

Instances are $(G, y, x)$ triples read straight off existing data: the $t = 0$ action bag of an episode is the source set $x$ and the last record's next state is the observation $y$ (a Monte Carlo marginal under `--sl-observation marginal`, strictly more informative than the literature's single binary draw, or one realization under `binary`, which is the comparable column). Two disjoint pools of `--sl-instances` (default 20) episodes are subsampled deterministically in the seed (episodes are written selector-major, so a prefix would be all one selector). Under `--sl-budget-mode episode` (the default, SL-VAE's given-$k$ convention) the program is handed the instance's own source count, which makes precision, recall and F1 equal by construction; `sweep` forces the pipeline's $k$ and filters the pool to source counts within a tolerance band of it, raising rather than widening when the band is empty.

The program implements `localize(graph, observation, budget) -> list[int]` and optionally `source_scores`. The reward is label-free consistency: for every instance the harness's `ForwardOracle` rolls the returned set forward through the arm's own evaluator (the exact seed commit the generator writes at $t = 0$, followed by $H$ empty bags; one metered rollout counted in `scoring_calls`) and scores

$$\text{consistency}_i = -\frac{1}{N} \sum_{v \in V} \bigl(\hat y_i(v) - y_i(v)\bigr)^2, \qquad R = \frac{1}{|\mathcal{I}|} \sum_i \text{consistency}_i,$$

higher is better and 0 is perfect. Diffusion is many-to-one, so a set that reproduces $y$ need not be the true set; that identifiability is measured after the search (Section 9.3), never assumed. The feedback (`summarize_localization`) is the residual: the reward line, the mean numbers of sources named and of nodes over-explained (residual at least $+0.25$: the cascade reaches them, the observation says clean) and under-explained (at most $-0.25$), the harness scoring-call count with the statement that the program is offline and makes none, the worst and best episodes with their named sources and their over- and under-explained nodes (node, residual, degree), and one diagnosis: overshoot ("prefer nodes whose forward reach stays inside the observed region: peripheral members of it, one per component") or undershoot ("name one node per unexplained component before refining the rest"). The `@native` binding for canned library members is a raiser that names the condition and lists label-propagation, infected-subgraph centralities, per-component centres and separation constraints as legal alternatives. `--sl-transfer-from` runs another run's winning script as a canned strategy, the graph-transfer axis no per-instance method can enter.

### 10.6 Cascade reconstruction

Instances are masked cascades from traced data (`--trace-parents`, which swaps NDlib's models for subclasses that record which $u$ infected $v$; under LT the truth is a parent set and path precision is structurally easier than under IC, so the two are never compared). `--cr-setting` selects one of four protocols whose rows are never pooled: `partial_times` (a subsample of the infected set with activation times, the default), `partial_nodes` (the same subsample with times withheld), `final_snapshot` (the terminal state alone, the hardest published formulation), `hidden_nodes` (nodes deleted from the adjacency, never sources, at `--cr-hidden-rate` 0.1). `--cr-observation-rate` (default 0.3) is the probability a node is reported; an empty report set is repaired by forcing one random infected node, since a degenerate instance is not a hard one. Each episode's mask uses its own stream `rng([seed, index])`, so it is stable when the instance limit changes; a dataset without parents raises unless `--cr-tree-weight 0`.

The program implements `reconstruct(graph, observation, horizon) -> {node: (time, parent)}` with `parent = None` marking a source. The reward is the log-probability of the decoded history under the generative model, per node, minus the fraction of the observation it contradicts:

$$R_i = \frac{1}{\max(1, N)} \Bigl[\, n_{\text{src}} \log \tfrac{1}{N'} + (N - n_{\text{src}}) \log\bigl(1 - \tfrac{1}{N'}\bigr) + \sum_{t} \Bigl( \sum_{v \in \text{gained}_t} \log m_t(v) + \sum_{v \in \text{missed}_t} \log\bigl(1 - m_t(v)\bigr) \Bigr) \Bigr] - 1.0 \cdot \bigl(1 - \text{consistency}_i\bigr),$$

with $N' = \max(2, N)$, $m_t = $ the arm's transition kernel (`StepOracle`, one `step_marginals` call per asserted step, counted in `scoring_kernel_calls`) evaluated at the implied state $(I_t, F_t)$, $\text{gained}_t$ the susceptible nodes the history says activate at $t+1$, $\text{missed}_t$ the susceptible nodes it says do not, plus one terminal step asserting nothing more activated when the horizon allows it. The $1/N$ source prior is what stops "every reported node is a source at $t = 0$" from outscoring every real decoder (each declared source costs about $\log N$; measured on the first smoke run). Consistency is $1 - (\text{missing} + \text{mistimed} + \text{outside}) / \text{constraints}$, where missing are reported nodes the decode dropped, mistimed are reported times it moved, outside are decoded nodes a snapshot says stayed clean, and constraints count the reported nodes, the reported times, and (with a snapshot) the decoded nodes. Both parts are computable at deployment.

Feedback (`summarize_reconstruction`) prints the reward line with its two parts, the per-cascade counts of decoded nodes, sources, dropped reports, moved times and outside nodes, the scoring-kernel count, then the worst and best cascades with the eight least probable transmissions asserted (probability, child, parent, time), the eight activations the kernel expected at $P \ge 0.5$ that the decode left out, and the contradicted reports; and up to three diagnoses: arguing with the data (dropping or moving a report costs more than any likelihood gain), near-impossible transmissions (the parent was not active at $t - 1$ or the arc almost never transmits), and inferring nothing beyond the reports. Every library decoder shares one parent rule (`finalize`: clamp times, backfill an incoherent node's most likely in-neighbour at $t - 1$, attach each non-source to the highest-probability earlier in-neighbour preferring $t - 1$), so a path-precision difference between rows is a difference in times, and `order_accuracy` reads 1.0 for the whole library pool by construction.

### 10.7 Cascade prediction

The only task that runs no simulator: a real corpus is replayed into the same transition format by `data/wm_cascades.py`, elapsed time is binned by `--cp-step`, and `load_forecasts` raises on a simulated dataset. The program sees a `CascadeObservation` (the whole observed adoption history as `{node: timestep}`, the last observed wave as `frontier`, `observed_steps`, `horizon`, the root, the publish time, `popularity = |adopters|`) and `self.fit_examples`, the selection pool it may fit constants on, and implements `predict(graph, observation, horizon) -> float | None`. A prediction must be at least the observed popularity and at most $N$; `None` or a non-finite value declines the cascade, which is counted in `n_failed` and never scored (SEISMIC declines supercritical cascades; reporting the mean over scoreable cascades alone silently favours the model that gives up more often), and declining every cascade is rejected.

The reward is the configured error, `--cp-metric` (default `msle`), on the `--cp-target` quantity (`increment`, CasFlow's label, or `total`), lower is better and `inf` when non-finite:

$$\text{MSLE} = \frac{1}{n} \sum_{i} \bigl(\log_2 \max(\hat P_i, 1) - \log_2 \max(P_i, 1)\bigr)^2.$$

Because three independent choices hide inside the name (log base, total versus increment, smoothing offset) and published tables mix them, `popularity_metrics` computes every variant this repo can report: `msle`, `msle_offset`, `msle_natural`, `msle_increment`, `male`, `male_increment`, `mape` (CasFlow's code form), `mape_casft`, `mrse`, `mrse_median`, `wroperc` (relative error at least 0.5), the median, 75th and 95th percentile of the absolute percentage error (SEISMIC's quantiles, because the mean is outlier-dominated), `pcc`, `r2`, `coverage` (top 10 percent hit rate), `decline_rate`, and `doubling_accuracy` (Cheng et al.'s balanced framing). The standard error is computed from per-cascade errors, not from the predictions (the earlier bug put the whole search inside the noise band on any real corpus). Feedback (`summarize_prediction`) prints the error line with declines, the log-space and raw-count errors, the rank and tail metrics, the kernel-call count, the direction (how many cascades were over-predicted, the mean $\log_2$ residual, and, when its magnitude exceeds 0.5, the diagnosis that a constant multiplicative bias is the cheapest fix because MSLE is symmetric in log space), a note that the cascades are real and whatever produced them is not independent cascade, and the best and worst cascades. The `--mc-agreement` modelling-error check rolls the arm's own forward model forward from each observed prefix with no program in the loop (`ForecastOracle`, 2 samples) and reports `model_msle`, the falsification number for the IC assumption. `--cp-split chronological` (default) is the leak-free protocol; `random` reproduces the field's leaky 70/15/15 split on purpose.

---

## 11. Adaptive, streaming and multi-round branches

### 11.1 Adaptive IM: rounds

`--rounds r` (default 4) or `--per-round-budget b` (Han et al.'s $b$-sweep [22], which wins when both are given and yields $r = \lceil k / b \rceil$ with the last batch trimmed so the sum is exactly $k$) splits the budget into batches `[base + 1] * extra + [base] * (r - extra)` with `base, extra = divmod(k, r)` and `r = min(r, k)`. The batches land at timesteps `0, gap, 2·gap, ...` (`--round-gap`, default 1); a schedule whose last batch falls past the horizon raises rather than clipping, because seeds scheduled past the horizon are silently never spent. The budget is therefore enforced structurally per round, with no cross-call counter, since any cumulative state would mean different things under the Monte Carlo (episode, timestep) and world-model (timestep, sample) loop orders. `--feedback-model` decides what the policy observes at a round boundary: `full_adoption` hands it the realized state, `myopic` blanks `infected` and hands it the current wave only, the information state where adaptive submodularity fails [23]. Re-seeding an already active node is rejected under `full_adoption` (the policy was handed the set and could have filtered it) and silently dropped with the slot spent under `myopic` (the price of the weaker observation). The world-model environment queries the policy once per ensemble member per round, interleaved, so each sample gets its own shallow clone of the policy with non-callable attributes deep-copied (a single object once saw 50 histories as one). The anchor `static_split` deals one static ranking (degree discount) in batches, stateless across rounds, so the adaptivity gap `spread(adaptive) / spread(non-adaptive)` at matched $k$ has a denominator computed under the identical round machinery; theory caps the myopic gap at 4 and non-adaptive greedy is no worse across all graphs [23], so the claim is cost (`evaluator_seconds`), not spread. Every adaptive result records the schedule and the per-round spreads.

### 11.2 Dynamic graphs: the edit stream

`--edit-rate` applies a deterministic schedule of exogenous edge edits per timestep, a fraction of $|E|$, generated from (base graph, seed, timestep) and delivered through the action bag by `GraphEditStream.wrap`, so no environment changed and both loop orders see the same $A_t$. An adaptive policy is handed `stream.graph_at(t)` and can react; a static plan cannot, which is the experiment. Credit never ablates a stream edit, because those ops are not the policy's.

### 11.3 Multi-round campaigns and hidden edge weights

`--campaigns r` wraps any evaluator in `MultiRoundEnvironment` (Section 4.7) for both the arm and the referee, so a union is never compared with a single spread. `--hide-edge-weights` trains the world model on ones instead of the true $p_{uv}$, the online or bandit information state; the environment must be told the same flag or a masked model would be rolled out against the true probabilities, the one place masking once silently did not apply.

---

## 12. Condition 9: published LLM algorithm-discovery systems

Nine systems run as `discovery:<name>` (alias `all-discovery`): OpenEvolve [13], CodeEvolve [14], LLaMEA [12], EoH [9], ReEvo [10], MCTS-AHD [11], LLM4AD's FunSearch [8, 16] and HillClimb [16], and DeepEvolve [15]. The rule is condition 7's rule applied to a program instead of a set: each system runs its own loop, prompts and published defaults inside its own virtual environment, never imports this package, and is handed one problem per run. The artefact that crosses the boundary is a program.

The problem is one `Contract` per task (function name, signature, description, a degree-heuristic initial program, and the reward's name): `select_seeds(graph, k)`, `select_removals(graph, k, outbreak)`, `select_blockers(graph, k, rumour, lever)`, `select_doses(graph, k, outbreak, lever)`, `localize(graph, observation, k)`, `reconstruct(graph, observation, horizon)`, `predict(graph, observation)`; adaptive IM reuses the IM contract (the program commits $k$ seeds at $t = 0$, the non-adaptive side of the gap). The task statement every system receives is identical: the description, the instance line (node and arc counts, directedness, dynamics, budget, horizon, lever and the task-specific protocol line), the fitness paragraph ("non-negative and higher is better ... scored by a fixed Monte Carlo simulator; it never changes between candidates"), and the rules (the import whitelist, no file, network or subprocess access, the contract function last in the file, the time limit, "there is no simulator you can query").

Fitness is one subprocess, `baselines/score_program.py`, run under our interpreter with the arm's own `ExperimentConfig` (same outbreak, lever, splits, seed) pinned to one canned pass on the selection split with credit, agreement and checkpoints off. It wraps the candidate through `program_script`, which converts the graph to a networkx graph with arc weights, resolves `<name>_v<k>` versions (ReEvo and MCTS-AHD rename the evolved function), de-duplicates and clamps the returned picks, and produces a `Strategy` whose plan puts everything at $t = 0$, so the discovered code runs under the identical executor, validation and referee as generated code and the world model is unreachable by construction. `oriented_fitness` maps every reward to a non-negative higher-is-better number with 0 as the floor for a failed candidate: `max(0, 1 + reward)` for localization, `exp(min(0, reward))` for reconstruction, `max(0, reward)` for a maximizing task, `1 / (1 + error)` for prediction, `max(0, N - reward)` (nodes saved) for a minimizing spread task; the raw reward travels beside it under its own name. Failed or timed-out candidates score 0 rather than raising. Secrets never enter the context file or the argv: the launcher copies environment variables by name.

Published defaults are the point, so sample budgets differ from ours by 5 to 50 times: OpenEvolve 100 iterations of SEARCH/REPLACE evolution over a MAP-Elites island archive [17]; CodeEvolve 50 epochs on 3 islands with meta-prompting and a CVT-MAP-Elites grid; LLaMEA a $(5 + 5)$ evolution strategy over 100 evaluations; EoH population 5 over 20 generations with its five operators (it minimizes, so it receives `-fitness`); ReEvo `max_fe` 100 with population 10 from an initial 30, with its client patched so the initial population is 30 requests rather than one `n=30` call, which the gateway collapses to one choice; MCTS-AHD `max_fe` 1,000 with UCT, progressive widening and thought alignment; FunSearch and HillClimb 20 samples with LLM4AD's sampler and sandbox (DeepMind released the database and no sampler); DeepEvolve OpenEvolve's database behind an agents research loop with hosted web search, every agent role on the run's model. Two more repositories are registered as blocked with verified reasons (LLM4AD_Next, GS4CO). `external.info` carries each framework's own bookkeeping; `DISCOVERY_SMOKE=1` shrinks every budget to a few samples for the adapter self-check only; `python -m baselines.check_discovery [--live <name>]` is the runnable check.

---

## 13. Provenance: what is taken from where

| Component | Where it lives | Source or inspiration |
|---|---|---|
| Diffusion models IC and LT, seed-set objective $\sigma(S)$ | `data/wm_simulator.py`, NDlib | Kempe, Kleinberg and Tardos [1]; Rossetti et al. [7] |
| Lazy greedy, CELF and CELF++ in the library; residual-gain feedback as CELF's marginal gain at RIS cost | `tools/algorithms.py`, `methods/base.py` | Leskovec et al. [2]; Goyal et al. [3]; Borgs et al. [5] |
| Degree discount, IMM, reverse-reachable sampling primitives | `tools/algorithms.py`, `tools/primitives.py` | Chen et al. [4]; Tang et al. [6]; Borgs et al. [5] |
| Per-round adaptive policies (AdaptGreedy, EPIC), the $b$-sweep, the adaptivity gap and the myopic feedback model | `rounds.py`, `tools/adaptive_algorithms.py` | Han et al. [22]; Golovin and Krause, Peng and Chen, Chen and Peng [23] |
| The learned world model as the environment inside an agent loop | `envs/world_model_env.py` | Ha and Schmidhuber [21] |
| Common random numbers, paired re-scoring of the incumbent | `methods/evolve.py`, `methods/base.py` | Glasserman and Yao [20] |
| Population of programs with LLM crossover and mutation operators; rank-weighted partner selection | `methods/evolve.py`, `prompts.py` | EoH [9]; LLaMEA [12] |
| Budget-aware exploration constant; five-operator set with thought alignment | `methods/evolve.py` | MCTS-AHD [11] |
| Previous-attempts table; diff-based edits; parsimony pressure; island archive in condition 9 | `prompts.py`, `methods/evolve.py` | OpenEvolve and AlphaEvolve [13]; MAP-Elites [17] |
| Short- and long-term reflection collapsed into one call; hints in the attempts table; running verbal memory | `methods/evolve.py`, `prompts.py` | ReEvo [10]; Reflexion [18]; Self-Refine [19] |
| Idea search before code, novelty judged against the library and the population | `prompts.py` | DeepEvolve [15]; novelty search [17] |
| Program-space search with a fixed sandboxed evaluator; score-signature clustering and islands in condition 9 | `executor.py`, `baselines/discovery.py` | FunSearch [8]; LLM4AD [16] |
| Consistency reward for source localization; LPSI as the bar; per-node scorers for a real AUC | `localization.py`, `tools/localization_algorithms.py` | LPSI [24]; NETSLEUTH [25]; SL-VAE [26]; Comin and Costa [27] |
| Ordered-Steiner and delayed-BFS decoders; tree sampling; the DASH final-snapshot setting; timing NRMSE and the tree metrics | `reconstruction.py`, `tools/reconstruction_algorithms.py`, `world_model/wm_metrics.py` | Xiao et al. [28]; Rozenshtein et al. [29]; DITTO [30] |
| Cascade features, the doubling constant and the second-half reshare rate; Szabo and Huberman's log-linear rule; SEISMIC's declines and quantiles; CasFlow's MSLE form; the leak-free chronological split | `prediction.py`, `tools/prediction_algorithms.py`, `data/wm_cascades.py` | Cheng et al. [31]; Szabo and Huberman [32]; Zhao et al. [34]; CasFlow [33]; CasTemp (see `research/cascade_prediction.md`) |
| NetShield, DAVA, degree and acquaintance immunization; the eigendrop as context | `tools/immunization_algorithms.py`, `epidemic.py` | Tong et al. [35]; Zhang and Prakash [36]; Pastor-Satorras and Vespignani, Cohen et al. [37] |
| The four blocking levers, prevented influence, the tie-break conventions, the detection delay, the weight relaxation | `blocking.py`, `data/wm_competitive.py` | Budak et al., He et al., Tong et al., Kimura et al., SandIMIN, DiffIM [38] |
| Dismantling baselines and the connectivity functionals printed as context; `degree_rank_spearman` | `tools/dismantling_algorithms.py`, `world_model/wm_metrics.py` | Morone and Makse, Zdeborová et al., Braunstein et al., Ren et al., Clusella et al., Fan et al., Grassia et al., Tian et al. [39]; Schneider et al. [41] |
| Backbones behind $f_\theta$ | `world_model/model/*.py` | GCN, GraphSAGE, GATv2, Graph Transformer, GCNII [40] |

Everything not listed (the structured heads, the offline rule, the four-binding harness oracles, the label-free rewards for the two inverse tasks, the referee protocol, the anchor leaderboard, the reference diff, the residual-gain and community feedback, per-action credit with solo cascades and kernel bottlenecks, probes, thread compaction, the write-up with diffs, the scored-mode harnesses, the deletion-bag expansion, the ring control, the paired-seed protocol, and the exact union for multi-round campaigns) is this repository's own design.

---

## 14. References

1. Kempe, D., Kleinberg, J., Tardos, É. Maximizing the spread of influence through a social network. KDD 2003.
2. Leskovec, J., Krause, A., Guestrin, C., Faloutsos, C., VanBriesen, J., Glance, N. Cost-effective outbreak detection in networks (CELF). KDD 2007.
3. Goyal, A., Lu, W., Lakshmanan, L. V. S. CELF++: optimizing the greedy algorithm for influence maximization in social networks. WWW 2011 companion.
4. Chen, W., Wang, Y., Yang, S. Efficient influence maximization in social networks (DegreeDiscount). KDD 2009.
5. Borgs, C., Brautbar, M., Chayes, J., Lucier, B. Maximizing social influence in nearly optimal time (reverse reachable sets). SODA 2014. arXiv 1212.0884.
6. Tang, Y., Shi, Y., Xiao, X. Influence maximization in near-linear time: a martingale approach (IMM). SIGMOD 2015.
7. Rossetti, G., Milli, L., Rinzivillo, S., Sîrbu, A., Pedreschi, D., Giannotti, F. NDlib: a Python library to model and analyze diffusion processes over complex networks. International Journal of Data Science and Analytics, 2018. https://github.com/GiulioRossetti/ndlib
8. Romera-Paredes, B., Barekatain, M., Novikov, A., et al. Mathematical discoveries from program search with large language models (FunSearch). Nature 625, 2024.
9. Liu, F., Tong, X., Yuan, M., Lin, X., Luo, F., Wang, Z., Lu, Z., Zhang, Q. Evolution of Heuristics: towards efficient automatic algorithm design using large language models (EoH). ICML 2024. arXiv 2401.02051.
10. Ye, H., Wang, J., Cao, Z., Berto, F., Hua, C., Kim, H., Park, J., Song, G. ReEvo: large language models as hyper-heuristics with reflective evolution. NeurIPS 2024. arXiv 2402.01145.
11. Zheng, Z., Xie, Z., Wang, Z., Hooi, B. Monte Carlo tree search for comprehensive exploration in LLM-based automatic heuristic design (MCTS-AHD). ICML 2025. arXiv 2501.08603.
12. van Stein, N., Bäck, T. LLaMEA: a large language model evolutionary algorithm for automatically generating metaheuristics. IEEE Transactions on Evolutionary Computation, 2025. arXiv 2405.20132.
13. Novikov, A., et al. AlphaEvolve: a coding agent for scientific and algorithmic discovery. Google DeepMind, 2025. Sharma, A. OpenEvolve: an open-source evolutionary coding agent. https://github.com/codelion/openevolve, 2025.
14. CodeEvolve: an open-source evolutionary coding agent for algorithm discovery and optimization. arXiv 2510.14150, 2025.
15. DeepEvolve: combining deep research with algorithm evolution. arXiv 2510.06056, 2025.
16. Liu, F., et al. LLM4AD: a platform for algorithm design with large language models. arXiv 2412.17287, 2024.
17. Lehman, J., Stanley, K. O. Abandoning objectives: evolution through the search for novelty alone. Evolutionary Computation 19(2), 2011. Mouret, J.-B., Clune, J. Illuminating search spaces by mapping elites. arXiv 1504.04909, 2015.
18. Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., Yao, S. Reflexion: language agents with verbal reinforcement learning. NeurIPS 2023. arXiv 2303.11366.
19. Madaan, A., et al. Self-Refine: iterative refinement with self-feedback. NeurIPS 2023. arXiv 2303.17651.
20. Glasserman, P., Yao, D. D. Some guidelines and guarantees for common random numbers. Management Science 38(6), 1992.
21. Ha, D., Schmidhuber, J. World models. arXiv 1803.10122, 2018.
22. Han, K., Huang, K., Xiao, X., Tang, J., Sun, A., Tang, X. Efficient algorithms for adaptive influence maximization (AdaptGreedy, EPIC). PVLDB 11(9), 2018.
23. Golovin, D., Krause, A. Adaptive submodularity: theory and applications in active learning and stochastic optimization. JAIR 42, 2011. Peng, B., Chen, W. Adaptive influence maximization with myopic feedback. NeurIPS 2019. Chen, W., Peng, B. On adaptivity gaps of influence maximization under the independent cascade model with full-adoption feedback. ISAAC 2019.
24. Wang, Z., Wang, C., Pei, J., Ye, X. Multiple source detection without knowing the underlying propagation model (LPSI). AAAI 2017.
25. Prakash, B. A., Vreeken, J., Faloutsos, C. Spotting culprits in epidemics: how many and which ones? (NETSLEUTH). ICDM 2012.
26. Ling, C., Jiang, J., Wang, J., Liang, Z. Source localization of graph diffusion via variational autoencoders for graph inverse problems (SL-VAE). KDD 2022.
27. Comin, C. H., da Fontoura Costa, L. Identifying the starting point of a spreading process in complex networks. Physical Review E 84, 056105, 2011.
28. Xiao, H., Rozenshtein, P., Tatti, N., Gionis, A. Reconstructing a cascade from temporal observations. SDM 2018. Xiao, H., Aslay, C., Gionis, A. Robust cascade reconstruction by Steiner tree sampling. ICDM 2018.
29. Rozenshtein, P., Gionis, A., Prakash, B. A., Vreeken, J. Reconstructing an epidemic over time. KDD 2016.
30. Qiu, R., Wang, D., Ying, L., Poor, H. V., Zhang, Y., Tong, H. Reconstructing graph diffusion history from a single snapshot (DITTO). KDD 2023.
31. Cheng, J., Adamic, L., Dow, P. A., Kleinberg, J., Leskovec, J. Can cascades be predicted? WWW 2014.
32. Szabo, G., Huberman, B. A. Predicting the popularity of online content. Communications of the ACM 53(8), 2010. arXiv 0811.0405.
33. Xu, X., Zhou, F., Zhang, K., Liu, S., Trajcevski, G. CasFlow: exploring hierarchical structures and propagation uncertainty for cascade prediction. IEEE TKDE, 2021.
34. Zhao, Q., Erdogdu, M. A., He, H. Y., Rajaraman, A., Leskovec, J. SEISMIC: a self-exciting point process model for predicting tweet popularity. KDD 2015.
35. Tong, H., Prakash, B. A., Tsourakakis, C., Eliassi-Rad, T., Faloutsos, C., Chau, D. H. On the vulnerability of large graphs (NetShield). ICDM 2010.
36. Zhang, Y., Prakash, B. A. DAVA: distributing vaccines over networks under prior information. SDM 2014.
37. Pastor-Satorras, R., Vespignani, A. Immunization of complex networks. Physical Review E 65, 036104, 2002. Cohen, R., Havlin, S., ben-Avraham, D. Efficient immunization strategies for computer networks and populations. Physical Review Letters 91, 247901, 2003.
38. Budak, C., Agrawal, D., El Abbadi, A. Limiting the spread of misinformation in social networks. WWW 2011. He, X., Song, G., Chen, W., Jiang, Q. Influence blocking maximization in social networks under the competitive linear threshold model (CLDAG). SDM 2012. Tong, G., Wu, W., Du, D.-Z. Randomized rumor blocking in social networks (RPS). INFOCOM 2017. Kimura, M., Saito, K., Motoda, H. Blocking links to minimize contamination spread in a social network. ACM TKDD 3(2), 2009. Wang, et al. SandIMIN. PVLDB 2024, arXiv 2405.12871. Tong, G. StratLearner: learning a strategy for misinformation prevention in social networks. NeurIPS 2020. DiffIM. AAAI 2025.
39. Morone, F., Makse, H. A. Influence maximization in complex networks through optimal percolation (CI). Nature 524, 2015. Zdeborová, L., Zhang, P., Zhou, H.-J. Fast and simple decycling and dismantling of networks (CoreHD). Scientific Reports 6, 2016. Braunstein, A., Dall'Asta, L., Semerjian, G., Zdeborová, L. Network dismantling (Min-Sum). PNAS 113(44), 2016. Ren, X.-L., Gleinig, N., Helbing, D., Antulov-Fantulin, N. Generalized network dismantling (GND). PNAS 116(14), 2019. Clusella, P., Grassberger, P., Pérez-Reche, F. J., Politi, A. Immunization and targeted destruction of networks using explosive percolation (EI). Physical Review Letters 117, 208301, 2016. Fan, C., Zeng, L., Sun, Y., Liu, Y.-Y. Finding key players in complex networks through deep reinforcement learning (FINDER). Nature Machine Intelligence 2, 2020. Grassia, M., De Domenico, M., Mangioni, G. Machine learning dismantling and early-warning signals of disintegration in complex systems (GDM). Nature Communications 12, 2021. Tian, H., et al. MIND. AAAI 2026.
40. Kipf, T. N., Welling, M. Semi-supervised classification with graph convolutional networks. ICLR 2017. Hamilton, W. L., Ying, R., Leskovec, J. Inductive representation learning on large graphs (GraphSAGE). NeurIPS 2017. Brody, S., Alon, U., Yahav, E. How attentive are graph attention networks? (GATv2). ICLR 2022. Chen, M., Wei, Z., Huang, Z., Ding, B., Li, Y. Simple and deep graph convolutional networks (GCNII). ICML 2020.
41. Schneider, C. M., Moreira, A. A., Andrade, J. S., Havlin, S., Herrmann, H. J. Mitigation of malicious attacks on networks. PNAS 108(10), 2011.

The per-task literature reviews under `research/` carry the full bibliographies, URLs and published result tables these entries were checked against.

---

## 15. Known caveats and asymmetries

These are true of the code as written and worth knowing before reading a number.

- Two different noise bands exist. The prose the model reads reports the two-sigma band on the difference, $2\sqrt{\text{se}_c^2 + \text{se}_i^2}$; the acceptance rule and the attempts table use $\max(\text{se}_c, \text{se}_i)$.
- The realization seed counts turns under `evolve` (`seed + iteration + 1`, failures included) and successful evaluations under `one_shot` (`seed + evaluations + 1`), so a repair turn does not burn a realization there.
- The plan cache freezes a nondeterministic `plan_horizon` after its first evaluation; re-scoring reuses the cached plan by design.
- The world-model environment records `cost["n_samples"]` as the constructor value even when a batched call overrides `num_samples`; `step_marginals` accepts a seed it never uses and always reads the base edges, so it does not see edge ops applied during a rollout.
- Credit, probes and the referee run at the base seed; the batched credit path stripes samples across plans, so two ablated plans do not share realization streams.
- `ForwardOracle` (localization) does not count its empty-seed shortcut in `forward_calls`, while `ForecastOracle` (prediction) counts every call; the two cost columns are not identical units.
- `max_listed_episodes` in `prompts.py` is a dead constant: the episode blocks report ranges rather than lists.
- The Monte Carlo agreement environment for the two inverse tasks is built without an outbreak or competitive configuration; those tasks have neither, so nothing is lost, but the code path differs from the intervention branch.
- Until 2026-09-04 the prediction harness bound `expected_popularity` on generated strategies as a closure over a missing oracle, which would have raised a `TypeError` rather than a `StrategyError` if a generated predictor had called it; it is now bound only on canned baselines, like its two siblings.
- The running design-rule memory of `evolve` was checkpointed but not written to the results JSON until 2026-09-04; it is now the `memory` key.
- On the six-generation smoke run recorded the same day (`results/influence_maximization/ba/smoke_evolve`), no candidate cleared the acceptance band, so the population held one member throughout and `crossover` and `synthesize` were never eligible; all five operator draws after the seed were `refine` or `from_scratch`, as the schedule predicts for a population of one.
