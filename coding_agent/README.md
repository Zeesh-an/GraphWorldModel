# Coding-Agent Outer Loop

An LLM coding agent that **designs Influence-Maximization algorithms as executable Python programs**, where the designed algorithm emits graph interventions across **all** timesteps `t₀ … t_T` — not just a seed set at `t₀` — and is scored by rolling its actions through a learned Graph World Model (fast inner loop) with the NDlib Monte-Carlo simulator as ground truth.

---

## 1. Problem formulation: multi-timestep algorithms

### The reframing

Mark an algorithm's output as `R` (its _result_). Every classical IM/SL algorithm — PageRank seeding, greedy/CELF, RIS, source localization — produces only **`R_{t₀}`**: one solution at time zero, after which the dynamics simply run. That is the **degenerate special case** of what this subsystem targets:

```
classical:   R_{t₀}                          (seed set, then hands off to dynamics)
this work:   R_{t₀}, R_{t₁}, R_{t₂}, …, R_{t_T}   (an intervention at every timestep)
```

Equivalently, in optimization terms: the classical formulation confines the decision variables to `t₀`; here the problem is **augmented with decision variables at `t₁, t₂, …`**. Even adding only `t₁` changes the problem class — a human can hand-design PageRank, but not a "super-PageRank" that anticipates and schedules interventions across time. A coding agent can attempt it, and classical methods fall out as the `R_{t₀}`-only special case (an all-at-`t₀` plan with empty later bags — exactly what the `--baseline` mode constructs).

**Scope note:** the horizon does not need to be long for this to be a real contribution. Two or three intervention timesteps (`t₀ + t₁`, or `t₀ + t₁ + t₂`) already constitute the augmented problem; 100-step plans are neither needed nor affordable.

### Formal objects

- **State** `S_t = (infected_t, frontier_t)` — ever-activated nodes and the currently spreading wave (sorted node-id lists; `data.wm_simulator.State`).
- **Action bag** `A_t = [ActionOp, …]` — zero or more operations applied at step `t`. `ActionOp(op, target, destination=None, weight=None)` is the **unified action representation**: an explicit _type_ dimension (`op`) and _target_ dimensions (`target`, `destination`, `weight`), so one formulation covers seed injection, node removal, and structural edits across task types instead of one encoding per problem. The same value type is used by the data generator, the trained world model, and every strategy here (`types.ActionOp` is the simulator's class, re-exported — there is exactly one definition in the project).
- **Plan** `R_{t₀…T} = [A_0, A_1, …, A_T]` — what Method 1 strategies emit (`horizon + 1` bags; bag `t` is applied at timestep `t`).
- **Transition** `S_{t+1} ~ P(· | S_t, A_t, G_t)` — one simulator/world-model step: the deterministic action effect (T_exo) followed by one stochastic diffusion step (T_endo). Edge ops mutate `G_t` itself.
- **Reward** `J = E[ |infected_{T+1}| ]` — expected final spread, estimated by an ensemble (`mc_runs` NDlib rollouts or `n_samples` world-model rollouts).
- **Budget** `b` — at most `b` `add_node` actions across the whole plan.

### What the feedback optimizes

The refinement loop adjusts **the algorithm (the skill), never the LLM's weights**. The outer loop searches over _programs_: propose an algorithm → roll its actions through the environment → feed the rollout diagnostics back as a revision prompt → the agent rewrites the program. The world model's role is to make each candidate evaluation cheap enough that this program-level search is affordable.

The feedback per iteration (built by `methods/base.py::summarize` + `prompts.py::build_feedback_prompt`) contains:

- **reward ± SE** — ensemble-mean final spread with its noise floor;
- **paired delta with a noise verdict** (`methods/base.py::paired_delta`) — the signed change against the incumbent, plus the 2σ band on that comparison and an explicit ruling: `INSIDE THE NOISE — this change did nothing measurable`. This matters more than it sounds. On a hub-dominated graph the entire algorithmic spread between `high_degree`, `degree_discount` and RIS can be ~4 nodes while the ensemble SE at `--n-samples 50` is ~4 nodes: without the band the agent reads sampling luck as a result and chases it;
- **the seed set it actually chose** — every seed with its total degree and community id. Improving a selection you cannot see is guesswork;
- **infected + frontier curves** and, when the cascade dies before the horizon, the death step ("actions scheduled after t=X did nothing");
- **unreached-nodes report** — per-node `P(infected)` across the ensemble (`Trajectory.final_marginals`, computed by every env), ranked by **estimated residual gain** rather than degree: a high-degree node the cascade never reaches is usually unreachable, whereas a high-gain one is worth a seed;
- **residual-gain hints** — the highest-value nodes _not_ seeded, scored by reverse-reachable-set coverage not already covered by the seed set, in units of expected extra nodes. This is the marginal-gain signal CELF would compute, at RIS cost, and it never touches the metered evaluator. IC only (RR sets do not describe LT), skipped above `rr_max_nodes`, and the cover index is cached on the `GraphInfo` so it is built once per run, not once per turn;
- **community coverage** — seeds per community against community size and the mean reach inside it, plus a count of communities that got no seed at all;
- **adjacent-seed-pairs report** — seeds with overlapping neighborhoods (likely redundant budget);
- **baseline leaderboard** — one rollout per algorithm in `methods/base.py::anchor_algorithms` (`high_degree`, `degree_discount`, `pagerank_seeds`, `imm`, `random_seeds`; RIS members dropped under LT) in the same env at the same budget and horizon, sorted, included in every prompt. The bar is the top row, not one arbitrary classical algorithm. Under an MC evaluator these rollouts are charged to the arm like any other, so the list is deliberately short — trim `anchor_algorithms` if the episode cost matters more than the target;
- **reference diff** — the _strongest_ baseline's per-node marginals diffed against the agent's, naming the specific nodes it reaches that this strategy misses (and vice versa) plus the net expected-spread gap. Costs no extra rollouts: both marginal vectors already exist;
- **the incumbent script itself** — see below;
- optional **per-action counterfactual credit** (`--credit`) and **error tracebacks** on failed scripts.

Feedback is deliberately descriptive, never prescriptive: the environments report where the cascade went and what each action bought, but never which node to pick — the algorithm design stays with the agent.

### The edit target is the incumbent, not the last attempt

Every feedback turn shows two programs: the one that just ran, and — when they differ — the **best-scoring script so far**, labelled as the one to edit. This is the difference between a hill climb and a random walk. Editing the latest attempt means a regression becomes the base for every iteration after it, so a single bad sample derails the rest of the search; `evolve` has always done this correctly (its parent is the population best), and `one_shot` now does too.

### Checkpoint and resume

`one_shot` and `evolve` write `<arm>.ckpt.json` beside the result they will become, after **every** turn — evaluations and repairs alike. A file there means "this arm did not finish". The next run of the same arm picks it up and continues from the generation it stopped at: population, best-so-far (its trajectory serialized, not re-evaluated), counters, the full conversation thread, the transcript, and the anchor leaderboard, so a resume re-pays for neither the baseline rollouts nor the LLM calls it already made.

A checkpoint is only reused when its **fingerprint** matches the run about to start — method, mode, evaluator, model, temperature, budget, horizon, `outer_iters`, allowed ops, mc_runs, n_samples, seed, and graph size. Resuming a search under a different budget or model would silently splice two experiments together, which is worse than losing the work, so a mismatch prints what differed and starts fresh. `--force` (pipeline) and `--no-resume` (standalone) delete the checkpoint first, because "redo" must mean redo. A finished run deletes its own checkpoint, so a leftover file always means an interruption.

`layout.result_globs()` excludes the suffix, so a checkpoint is never mistaken for a result by the report or the summary table. `per_step` and `windowed` have no refinement loop and nothing to resume; canned arms are one deterministic rollout and are not checkpointed either.

### Failed scripts do not consume the search budget

A script that fails to parse, import, build, or run teaches the next turn something, but it is not an evaluation. `--outer-iters` counts **evaluations**; failures draw on a separate `max_repairs` budget (default 3) and the loop only gives up once that is exhausted. Previously a run with `--outer-iters 5` and two bad scripts silently became a 3-round search.

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
| `methods/`    | `one_shot.py`, `per_step.py`, `windowed.py`, `evolve.py` — all implement `OuterLoopMethod.optimize(agent, env, task, graph) → (Strategy, Trajectory)`                                                    |
| `envs/`       | `MonteCarloEnvironment` (NDlib ground truth), `WorldModelEnvironment` (trained GNN simulator) behind one `rollout(action_fn, horizon, budget, seed)` interface                                           |
| `tools/`      | `primitives.py` (16 classical-IM building blocks), `algorithms.py` (30 named algorithms), `library_api.py` (auto-generated prompt reference)                                                             |
| `credit.py`   | per-action counterfactual credit via paired ablation rollouts                                                                                                                                            |
| `run.py`      | experiment driver: config, provider selection, baselines, compare replay, results JSON                                                                                                                   |

The single coupling point between methods and environments is the **`ActionFn` closure**: `(State, timestep) → list[ActionOp]`. Environments never know whether actions come from a fixed plan, a live policy, or an LLM call — which is exactly what makes the world model a drop-in replacement for the simulator.

---

## 3. The three methods (`--method`)

The methods are two extremes plus a middle point on a cost/adaptivity spectrum. They differ **only** in who produces the `t > 0` actions and how often the LLM is invoked; the environment contract is identical.

### Method 1 — `one_shot`: one super-algorithm emits the whole horizon

The agent designs a single algorithm whose `plan_horizon(graph, budget, horizon)` returns the entire plan `R_{t₀…T}` up front. The environment rolls the whole plan; the final reward (plus a trajectory summary, plus optional per-action credit, plus the full traceback if the script failed) is appended to the next prompt, and the agent **revises the algorithm** — up to `--outer-iters` times, keeping the best-by-reward candidate.

- Inner execution is an **open-loop plan** (committed before seeing the realized stochastic trajectory); the outer skill-refinement loop is closed.
- LLM cost: `outer_iters` calls total. Rollout cost per candidate is one ensemble.
- Implementation detail: `plan_horizon` is invoked through `call_strategy`, so a runtime crash _inside generated code_ becomes a `StrategyError` whose traceback is fed back as a repair turn rather than killing the run. Every bag is validated (op legality vs `allowed_ops`, node range, per-bag and whole-plan seed budget).
- This is the highest-novelty method (the "super-algorithm") and the feasibility probe: can a program anticipate the full horizon?

### Method 2 — `per_step`: re-prompt the agent at every timestep

The LLM **is** the policy: at each timestep the current state (full infected + frontier lists) is appended to the prompt, the agent writes a fresh script, its `act(state, graph, timestep)` is executed for that step only.

- Maximally adaptive — every action conditions on the realized state.
- **Prohibitively expensive**: the LLM is called once per _(ensemble sample × timestep)_. With `n_samples=20` and horizon 10 that is up to ~200 coding-agent calls per evaluation. Treat this method as the adaptivity **upper bound / cost reference**, not the practical operating point. Every call is logged (`[per_step] LLM call k (t=…, |infected|=…)`) because that count is the cost.
- The budget is **per episode, not per bag**. Without that the policy could legally emit `budget` seeds at every one of `horizon+1` timesteps and play the same nominal k as a `one_shot` arm with 11× the seeds. The counter resets at `t=0`, which is exactly right under a sequential env (`MonteCarloEnvironment`); under a lockstep ensemble env every sample's `t=0` also resets, so from `t>0` the cap is shared across samples — stricter than per-sample, never looser. Each turn's prompt states the remaining budget and which nodes are already seeded.
- Caveats: there is no repair loop (a `StrategyError` mid-rollout aborts the method), and the archived `script` in the results JSON is the _last_ generated script (one exists per call).

### Method 3 — `windowed`: one online algorithm, re-applied per time window

The agent designs the algorithm **once**; the timeline is split into `--windows` stages (`window_length = max(1, (horizon+1) // windows)`), and the algorithm's `act(state, graph, window_index)` is consulted only at window boundaries — solving a fresh sub-problem on the current state each window (classical algorithms may be reused per window; the prompt says so explicitly). Budget applies **per window call**.

> This is the one method whose effective budget is not `--budget`. The multiplier is the number of boundaries, not `--windows`: at horizon 10 with `--windows 3` the window length rounds to 3 and `t=0,3,6,9` all fire, so a nominal `k` plays `4k` seeds. The results JSON records the real number as `effective_budget` so the sweep table is not silently read as an equal-budget comparison against `one_shot` — check that column before ranking a windowed arm.

- One LLM call, closed-loop at window granularity: cheaper than Method 2, more reactive than Method 1.
- The simplest method to stand up — "apply the old IM algorithm, staged" — and it reframes the objective from a single terminal target into per-window objectives with state rolled forward between stages.

### Method 4 — `evolve`: population search over algorithm edits (EvoX-lite)

Every generation after the first is an **edit of a parent** from a population of all past candidates, never a fresh program. Each iteration: pick the population best as parent, attach up to 2 high-reward alternatives as inspiration, apply a **variation operator** — `refine` (small targeted change: tune a weight, adjust one term) or `restructure` (redesign the core idea, same contract) — and evaluate the result into the population. Operator choice is stagnation-driven: `stagnation_patience` (2) non-improving iterations force a `restructure`, which then opens a fresh refinement window. `--outer-iters` is the total generation count (10+ recommended). Failed scripts feed the error into the next generation prompt, like one_shot's repair turn.

### `--strategy-mode`: what the agent is allowed to write

Orthogonal to the method choice (supported for `one_shot` and `evolve`):

- **`free`** (default): the agent writes a whole `Strategy` program with the full library callable. Observed failure mode: portfolio composition — the agent runs many library algorithms and picks the winner, editing nothing.
- **`scored`**: the agent may only fill in the internals of an algorithm. A fixed harness (`ScoredStrategy.plan_horizon`, `types.py`) greedily picks the argmax of `score(node, selected, graph)` until budget is spent, then calls `schedule(seeds, graph, horizon)` (default: all at t=0). The executor rejects any override of `plan_horizon`; the exec namespace contains **no `algorithms` module** and no simulation primitives (`mc_simulate_spread`, `compute_marginal_gain`, `build_simulator` are hidden — otherwise the agent re-derives CELF instead of inventing structural scoring logic). The library appears in the prompt as an _ideas menu_ only — any borrowed idea must be written out inside `score()`, where it can be mutated. Canned/`--baseline` scripts always run in free mode regardless of the flag.

The intended experiment grid: `one_shot × scored` tests whether edit-level generation works at all; `evolve × scored` is the full population search over algorithm internals; `free` arms remain as the composition comparison.

### The trade-off spectrum

|                              | Method 1 (`one_shot`)                       | Method 3 (`windowed`)                   | Method 2 (`per_step`)                    | Method 4 (`evolve`)                            |
| ---------------------------- | ------------------------------------------- | --------------------------------------- | ---------------------------------------- | ---------------------------------------------- |
| LLM/agent calls              | `outer_iters` (refinement only)             | 1 (designed once)                       | one per sample × timestep                | `outer_iters` (population edits)               |
| Adaptivity to realized state | none (open-loop plan)                       | per-window                              | per-step                                 | none (open-loop plan)                          |
| Cost                         | low                                         | low–medium                              | very high                                | medium (many cheap WM rollouts)                |
| Novelty                      | highest (super-algorithm across time)       | moderate (classical algos, staged)      | adaptivity bound, not a practical method | highest with `scored` (evolves algo internals) |
| Loop type                    | open-loop plan + closed skill refinement    | closed-loop policy (window granularity) | closed-loop policy (LLM as policy)       | population evolution (parent + operator)       |
| Failure handling             | full repair loop (errors → revision prompt) | fail-fast                               | fail-fast                                | error fed to next generation                   |

---

## 4. The action space

Five operations, shared with the data generator and the world model (`data.wm_simulator.valid_action_ops`):

| op                | fields                         | IC effect                                          | LT effect                              |
| ----------------- | ------------------------------ | -------------------------------------------------- | -------------------------------------- |
| `add_node`        | `target`                       | activate node (transmits during this step)         | activate node                          |
| `remove_node`     | `target`                       | leaves the frontier, **stays counted** as infected | back to Susceptible (can re-activate)  |
| `add_edge`        | `target→destination`, `weight` | add arc with transmission prob `weight`            | add edge structurally (weight ignored) |
| `remove_edge`     | `target→destination`           | remove arc                                         | remove edge                            |
| `set_edge_weight` | `target→destination`, `weight` | set transmission prob                              | no-op                                  |

**Budget semantics (and a known sharp edge):** `validate_actions` charges only `add_node` against the budget — edge ops are currently **free**. Left unconstrained, capable models reliably discover the degenerate exploit: boost every frontier out-edge to ~1.0 and convert stochastic IC into deterministic percolation (observed: `mc_reward` 99.97/100). Two controls exist today:

- `--allowed-ops add_node remove_node` — restricts the ops a strategy may emit; enforced in the prompt (`any other op is REJECTED`) _and_ by `validate_actions`, whose rejection message feeds the repair loop.
- A per-op cost/budget model is the planned fix for making edge ops a fair, non-degenerate part of the game.

**Rules vs. coverage are independent knobs:** `--allowed-ops` sets the _game rules_ for the agent; the world model's training data sets its _coverage_. Train the evaluator on the superset of ops (supersets are safe; subsets bite — see §7).

---

## 5. The agent layer: from chat reply to validated program

### What the LLM sees (the complete context)

The system prompt (per method) carries: the role; the output contract (exactly one fenced ```python block containing optional imports and one `Strategy` subclass); the **import policy** (below); the injected-names documentation; the action rules including what `remove_node` actually does; the method's interface (`plan_horizon` vs `act`); a **horizon note** chosen from `allowed_ops` (below); and two **worked exemplars** at the level the search should start from. The user prompt carries: task and objective strings, `diffusion_model`, `budget` (absolute and as % of nodes), `horizon`, `allowed_ops`, the **graph profile** (below), the full auto-generated **library API reference** (every algorithm and primitive signature with its first docstring line — docstrings in `tools/` are literally prompt text), the **full source of three library algorithms**, and the closing instruction. Refinement turns append: previous reward, the paired delta with its noise verdict, trajectory summary, the **reference diff** (below), the incumbent script, the error traceback (repair path), and the credit report (`--credit`).

#### Imports (`executor.allowed_imports`)

Generated scripts **may** import `numpy`, `networkx`, `scipy`, `math`, `random`, `statistics`, `heapq`, `bisect`, `collections`, `itertools`, `functools`, and nothing else. `executor.check_script_imports` enforces this on the AST before the script is compiled, and also rejects `__import__`, `eval`, `exec`, `compile`, `open`, and `input` by name — without those the whitelist would be trivially bypassable and the check would be theatre. A rejection is a `StrategyError`, so it lands as a repair turn naming the allowed set.

The prompt previously forbade imports outright while the exec namespace exposed full `__builtins__`, so `import numpy` already worked and the model simply never tried: it wrote pure-Python loops over the adjacency next to a numpy install. Vectorized scoring is what makes a large RIS `theta` affordable inside `--strategy-timeout`, which is the difference between a crippled RIS strategy and a competitive one. _This is a research scaffold, not a security sandbox._

#### Exemplars and library source

The reply skeleton used to be a one-line `high_degree` call with "do not return this unchanged" — an anchor so low the model would submit a variation of it and spend its iterations tuning a constant. The system prompt now carries two full strategies (`prompts.one_shot_exemplars`): a community-aware discount, and a lazy-greedy max-coverage over reverse-reachable sets with the staleness-stamped heap that makes a large `theta` affordable. The user prompt additionally carries the complete source of `degree_discount`, `cofim`, and `degree_ris_refine` (`library_api.sourced_algorithms`) — one per idiom the library uses. Signatures alone leave the model guessing at how a seed set is actually built here.

#### The horizon note (`allowed_ops`-dependent)

Under `add_node`-only IC the cascade is progressive and monotone, so a seed at `t>0` has strictly fewer steps to spread than the same seed at `t=0` and every schedule is dominated by "all seeds at `t=0`". The old prompt spent its most prominent paragraph pushing the model to "schedule interventions across t₀…t_T" — a flat direction it would burn iterations exploring. `build_system_prompt` now takes the `TaskSpec` and picks: `temporal_scheduling_note` when the task allows edge ops (where timing genuinely matters), `seed_timing_note` otherwise, which says plainly to put every seed in element 0 and spend the effort on *which* nodes.

The LLM never sees raw topology. By design, the _generated code_ gets the real graph at runtime (`GraphInfo` with full `edge_index`, `ic_probs`, neighbor/degree accessors); the LLM's job is to write an algorithm that inspects the graph programmatically, not to reason over an adjacency list in context. Serializing edges into the prompt would ask the model to eyeball what its own program computes exactly, would let it hardcode node ids for one instance instead of writing an algorithm that generalizes, and would not fit past a few thousand edges anyway.

#### The graph profile (`tools/graph_profile.py`)

Aggregate structure instead of topology — what a human expert uses to choose an approach, computed once and cached on the `GraphInfo`:

- **degree distribution** — mean/median/max/p90/p99, plus `cv` and `max/mean`. Measured on 100-node families: `ba` cv=0.95, `sbm` 0.46, `er` 0.40, `ws` 0.11, so the `heavy_tail_cv = 0.7` cutoff separates hub-dominated graphs cleanly.
- **components** — count, largest-component share, isolate count. On netscience that is 396 components with the largest holding 23.9% and 128 isolates, which is *the* strategic fact about that graph and was previously invisible.
- **k-core**, **clustering**, **degree assortativity**, **density**, and **reciprocity** (directed only).
- **communities** — label propagation count, modularity, largest sizes. Above modularity 0.3 the profile says outright that budget should be allocated across communities rather than by global score.
- **IC transmission** — edge-probability stats plus per-node expected out-transmission. Note the *mean* is pinned to exactly 1.0 by the weighted-cascade construction (`p = 1/in_degree` makes every in-sum 1), so it carries no information; the profile reports the **median** and the fraction of supercritical nodes instead.

Above `heavy_stats_max_arcs` (200k arcs) clustering switches to a sampled estimate and assortativity/communities are skipped — and the profile *says so* rather than silently omitting them. netscience profiles in 0.04s.

#### The reference diff (`methods/base.py::reference_diff`)

`baseline_anchor` rolls out every algorithm in `anchor_algorithms` in the same environment and returns the sorted table plus the **best** one's trajectory, so its per-node marginals can be diffed against the agent's. **Zero extra rollouts beyond the leaderboard** — both vectors are already paid for. Refinement turns get:

```
REFERENCE SCORES — classical baselines run on THIS graph, under THIS evaluator,
at the same budget and horizon. Beating the top row is the bar:
  degree_discount        47.50 (±4.13 SE)
  imm                    45.67 (±3.10 SE)
  pagerank_seeds         45.58 (±3.32 SE)
  high_degree            44.75 (±3.71 SE)
  random_seeds           36.08 (±3.59 SE)

REFERENCE DIFF (your cascade vs degree_discount's — the strongest baseline on
this graph — same evaluator, per-node P(infected)). Listed nodes are DECISIVE
flips only (one side >= 50%, the other < 10%); the net line below sums every
node, so it is larger:
  it reaches 3 nodes you miss — top by degree: 13(d=8, ref P=0.57 vs yours 0.07), …
  net expected spread vs the reference: -15.34 nodes
```

The double threshold keeps threshold-straddling nodes out of the list. In `evolve` the diff is appended to each population record's summary, so it travels with a candidate whenever it is shown as parent or inspiration. Canned arms (classical baselines, routing) skip the anchor entirely — they never read a prompt, so the rollout would only burn real episodes.

### Extraction, execution, validation

1. `extract_code_block` collects all fenced blocks and prefers the first one containing `class ` (models pad replies with prose snippets in extra fences), falling back to the first fence, then the raw text.
2. `check_script_imports` parses the AST and rejects any import outside `allowed_imports`, plus the builtins that would defeat that check.
3. `build_strategy` compiles and `exec`s the script in a **controlled namespace** containing exactly the names the prompts advertise: `ActionOp`, `State`, `GraphInfo`, `Strategy`, `algorithms`, `primitives`. Candidate classes are discovered by attribute (`plan_horizon`/`act`), excluding the injected base _by identity_ (a script may legally name its class `Strategy`). The instantiated strategy gets `source_script` attached so results JSONs archive the exact code that earned the reward. _This is a research scaffold, not a security sandbox._
4. `call_strategy` wraps every invocation of generated code: any exception becomes a `StrategyError` carrying the exception type and full traceback, and the call runs under a **wall-clock cap** (`--strategy-timeout`, default 300s, `0` disables). An overrun becomes a repair turn that names the limit and says what to do about it, instead of hanging the whole sweep on one O(N²) scan. The cap is a SIGALRM, so it lands between bytecodes: a call stuck inside a long numpy/networkx C call overruns until that call returns.
5. `validate_actions` checks each bag: op ∈ `allowed_ops`, node ids in range, **no node seeded twice** (a duplicate `add_node` spends two units of budget on one node and would otherwise pass silently as budget the strategy never used), and seeds within budget. `validate_plan` additionally enforces the total across the plan and rejects re-seeding a node at a later timestep; `per_step` enforces the same across the episode.

`StrategyError` is the failure currency of the subsystem: it marks "the generated program is wrong" (recoverable — becomes a revision prompt), as opposed to every other exception, which is a bug in _this_ codebase and crashes loudly.

### Providers

`GatewayProvider` speaks the OpenAI chat API against the lab gateway, routing the bearer token by model family (`claude-*` vs `gpt-*`; tokens + `GATEWAY_BASE_URL` live in the gitignored `.env`, loaded by `run.py`). Two operational quirks it absorbs: the gateway's Claude account **drops system messages**, so system+user are folded into a single user turn (verified harmless for gpt models); and retries are owned locally (3 attempts with printed warnings) with SDK retries disabled. `--temperature 0.0` gives greedy decoding for reproducible evals — keep the provider default (sampling) for refinement runs, where cross-iteration diversity is what the search feeds on. Any object with `complete(system, user) → str` satisfies `LLMProvider`; tests and `--baseline` inject canned providers.

---

## 6. Environments: the inner loop

Both environments implement `rollout(action_fn, horizon, budget, seed) → Trajectory` with identical semantics: start from the empty state; at each timestep `t ∈ [0, horizon]` ask `action_fn(state, t)` for a bag, advance one step, record; stop early when the cascade is dead (empty frontier) _and_ the strategy is idle (empty bag). Reward is the **ensemble mean** of final infected counts; `cost.reward_se` (sample std / √n) self-reports the noise; the recorded `states`/`actions` are the **first sample's** trajectory (the representative), whose last count is overwritten by the ensemble mean — which is why `counts` can end in a non-integer.

### `MonteCarloEnvironment` — ground truth

`mc_runs` independent NDlib simulators (fresh child-seeded simulator per run), exact IC/LT stepwise dynamics with full action support. This is the baseline the world model is measured against, and what `--compare` replays the winning strategy on.

### `WorldModelEnvironment` — the learned simulator

Loads a trained `WorldModel` from a `train_wm.py` results JSON: the `"config"` block reconstructs the exact architecture, and the checkpoint path is derived as `ckpt_dir / wm_<backbone>_<dynamics>.pt` (keep per-dataset `ckpt_dir`s — the filename does not encode the dataset). Rollout mechanics:

- **Lockstep block-diagonal batching**: all `n_samples` rollouts advance together; each timestep is ONE forward pass over the disjoint union of the samples' graphs (adjacency normalization is per-component, so this is exact — the same trick as the training collate). ~20× fewer forwards than per-sample loops.
- **Copy-on-write edge state**: the graph tensor is built once per rollout and rebuilt only when an edge op actually fires; each sample owns its own adjacency after mutating it.
- **Coupled sampling**: each step draws the _new infections_ once from the frontier marginal, then derives both channels — `frontier := new wave`, `infected` accumulates through the action semantics (adds join; IC removes stay counted; LT removes return to susceptible). Independent per-channel Bernoullis (the previous scheme) create impossible states — "ghost spreaders" with frontier=1, infected=0 — that systematically inflate free-running rollouts (measured at ~+1 count bias via the oracle head).
- Per-sample early termination with an active mask; per-sample states are honored for `per_step` semantics (each sample's `action_fn` call sees that sample's own state).

### Trusting the evaluator (read before believing `reward`)

- **`wm_minus_mc` is the trust meter.** Always run `--compare` while iterating; the WM's validated band on in-distribution plans is roughly ±2 (≈ its own `reward_se` at 20 samples).
- **OOD is about action _intensity_, not just op type.** A WM trained with ~1 injected op per step scores single interventions well and mass interventions arbitrarily badly (observed error growing +2.4 → −5.7 → −39.7 with edge-boost count). Match the training action distribution to what the agent is allowed to emit.
- **One-step metrics cannot certify an evaluator.** A per-wave transmission optimism of ~1% is invisible to teacher-forced `delta_f1`/Brier and compounds to +2 nodes over a 10-step rollout. Gate world-model checkpoints on `rollout.ens_count_bias` (target |bias| ≲ 1), and prefer the `structured_residual` head for IC when training data lacks edge-weight diversity (it anchors per-edge transmission on the true probability and learns only a residual correction; zero correction equals the validated oracle).
- Ensemble sizes are a noise knob: comparisons between strategies need `mc_runs` such that the gap exceeds ~2×√(se₁² + se₂²); the `--mc-runs` default (200) gives SE ≈ 0.6 on BA-100-scale spreads.

---

## 7. Counterfactual credit (`--credit`)

`credit.py` converts the scalar reward into causal, per-action feedback: for each action in the plan, re-roll the **same environment with the same seed** with that single action deleted; `delta = base_reward − ablated_reward` is the spread that action is responsible for (`~0` = wasted budget). The shared seed makes the comparison paired (common-random-numbers variance reduction). With `--credit`:

- `one_shot` refinement prompts include the per-action report — the agent sees "your `set_edge_weight(7→9)` contributed +0.00" instead of a bare scalar, which is precisely the signal that kills wasted actions.
- The results JSON gains `credit_base_reward` + one `{t, op, target, delta}` entry per action.
- For state-dependent strategies (`per_step`/`windowed`) the recorded bags are replayed as a fixed plan, so credit is an approximation there (the live policy would have reacted to the ablated cascade).

Cost: one extra ensemble rollout per action in the plan.

---

## 8. The tools library

The strategy's callable surface — advertised to the LLM verbatim via `library_api.build_api_reference()` (signatures via `inspect`, summaries from the first docstring line).

**Primitives** (`tools/primitives.py`, 16 advertised): degree/out-degree/weighted- degree scoring, top-k selection, power-iteration PageRank, eigenvector/closeness centrality, `mc_simulate_spread` (the honest NDlib spread estimator), marginal gain, reverse-reachable-set sampling + max-coverage selection (`batch_reverse_sample`, `ris_select`), label-propagation communities + largest-remainder budget allocation, RIS sample-size heuristic, live-edge sampling + reachability, truncated path-product influence scores.

**Named algorithms** (`tools/algorithms.py`, 30, uniform signature `(graph, budget, diffusion_model, **kw) → list[int]`):

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

Heavy algorithms (MIA/PMIA, LDAG, SKIM, StaticGreedy, SSA, IMM, CELF++, Adaptive Greedy) are faithful-but-simplified, noted in their docstrings. All MC-based estimators use the real NDlib simulator — the honest classical cost that the world model exists to undercut in the outer loop.

### What the agent may NOT call (`executor.mc_blocked_algorithms`)

Eleven of the thirty are hidden from **generated scripts** in every method, unless `--allow-mc-algorithms`:

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

Nothing in the library sits between 2.68s and 17.21s, so the cut needs no arbitrary threshold. Note `static_greedy` calls **no** MC primitive — it is blocked because its per-pick snapshot-reachability scan has the same shape and cost. `genetic_algorithm` / `simulated_annealing` are bounded (a fixed population × generations budget, not a full-N scan) but still call `mc_simulate_spread`, so they are blocked on the honesty ground below.

**The honesty ground matters more than the speed.** These run on a private `Simulator` built inside `primitives`, so `MonteCarloEnvironment.episodes_used` never sees them. Measured: one `one_shot` iteration whose script called `celf(mc_runs=20)` simulated **7,462** NDlib episodes and reported `real_env_episodes: 2`. That field is the sample-efficiency axis the whole condition ladder is read on, and conditions 3 (`@native`, "real executions only") and 6 ("no real episodes") both depend on it meaning what it says.

There is also a cleanliness argument: `celf_pp` is in `default_baselines`, so letting an agent arm call it makes condition 6 partly *be* condition 1 plus scheduling. Blocking it forces the agent to beat CELF++ rather than invoke it.

**Baselines and routing are exempt.** `--baseline celf_pp` (condition 1) and a routing pick (condition 2) run through the same canned-script path, and *are* that algorithm — blocking would delete the arm rather than speed it up, and their cost is honestly attributed. `run_experiment` computes `effective_allow_mc = config.allow_mc_algorithms or canned_script is not None`.

A blocked name stays bound to a raiser rather than vanishing, so calling one produces a legible repair turn naming the alternatives instead of an `AttributeError` traceback. The prompt lists the blocked set up front (`build_api_reference(exclude=…)`) so the first iteration is not spent discovering it.

---

## 9. Running experiments

> **For a full sweep, use the pipeline instead.** `python -m pipeline.run` runs every baseline condition at every budget, resumes what it already finished, and writes the plots and `results/<task>/<dataset>/<run>/report.md`. One arm at one budget is one file at `results/<task>/<dataset>/<run>/agent/<budget>/<arm>.json` — exactly what the commands below produce, so the two are interchangeable:
>
> ```bash python -m pipeline.run --dataset ba --budget-pcts 1 5 10 20 \
>     --outer-iters 5 --compare ```
>
> That default expands to the six-condition taxonomy: `--baselines` names the classical pool (condition 1), and `--arms` names `routing` (2) plus `one_shot_free@{native,monte_carlo,oracle,world_model}` (3–6). An arm spec is `baseline:<algorithm>`, `routing`, or `<method>_<mode>[@<evaluator>]`, where the evaluator suffix is what makes conditions 3–6 differ in exactly one thing. See `pipeline/conditions.py` for the grammar.
>
> `--compare` is what makes a multi-condition table valid: each arm's own reward comes from its own evaluator, so only the shared ground-truth MC replay is comparable across rows.

```bash
# LLM run: node-ops game, WM evaluator, MC compare, credit feedback
python -m coding_agent.run --data-dir results/influence_maximization/ba/default/data \
    --wm-results-json results/influence_maximization/ba/default/world_model/sage_IC.json \
    --method one_shot --evaluator world_model --budget 5 --horizon 10 --compare \
    --allowed-ops add_node remove_node \
    --model gpt-5.6-terra --outer-iters 3 --credit \
    --out-json results/influence_maximization/ba/default/agent/pct5/run.json

# Classical baseline through the identical pipeline (no LLM, no .env needed)
python -m coding_agent.run --data-dir results/influence_maximization/ba/default/data \
    --wm-results-json results/influence_maximization/ba/default/world_model/sage_IC.json \
    --method one_shot --evaluator world_model --budget 5 --horizon 10 --compare \
    --baseline celf --outer-iters 1 --out-json results/influence_maximization/ba/default/agent/pct5/baseline_celf.json

# GA routing: one LLM call picks a library algorithm (no code synthesis),
# then it runs through the identical --baseline canned path
python -m coding_agent.run --data-dir results/influence_maximization/ba/default/data \
    --model gpt-5.6-terra \
    --wm-results-json results/influence_maximization/ba/default/world_model/sage_IC.json \
    --method one_shot --evaluator world_model --budget 5 --horizon 10 --compare \
    --routing --outer-iters 1 --out-json results/influence_maximization/ba/default/agent/pct5/routing.json

# Oracle-dynamics ceiling: true IC transitions, no checkpoint / --wm-results-json
python -m coding_agent.run --data-dir results/influence_maximization/ba/default/data \
    --model gpt-5.6-terra \
    --method one_shot --evaluator oracle --budget 5 --horizon 10 --compare \
    --outer-iters 5 --out-json results/influence_maximization/ba/default/agent/pct5/oracle_run.json

# Native coding agent: one real execution per candidate, episodes counted
#   ... --evaluator monte_carlo --mc-runs 1

# Reproducible eval: greedy decoding
#   ... --temperature 0.0
```

Key flags: `--method {one_shot,per_step,windowed,evolve}` · `--strategy-mode {free,scored}` (scored = agent fills `score()`/`schedule()` hooks in the fixed `ScoredStrategy` harness; no `algorithms.*`; one_shot/evolve only) · `--evaluator {world_model,monte_carlo,oracle}` (`oracle` = true IC dynamics via the `structured_oracle` head — no checkpoint, IC-only, the model-based ceiling; `monte_carlo --mc-runs 1` = the native-agent condition, one real execution per candidate) · `--model` (gateway name; default `gpt-5.6-terra`) · `--temperature` (omit = provider default; `0.0` = greedy) · `--allowed-ops` · `--baseline <algorithm>` (synthesizes the all-at-`t₀` special-case plan) · `--routing` (GA-routing baseline: the LLM selects one pool algorithm from a name+summary menu — adaptive _selection_ without synthesis; the pick is recorded as `model: routing:<algo>` and the raw reply as `routing_reply`) · `--budget` / `--budget-pct` (percent of `num_nodes`, overrides `--budget` — same resolution rule as data generation) / `--horizon` / `--windows` / `--outer-iters` · `--mc-runs` (MC ensemble / compare size; default 200) · `--n-samples` (WM ensemble; default 20) · `--graph-id` (default: first graph in the store) · `--compare` · `--credit` · `--out-json`.

### Results JSON schema

| Key                                                                | Meaning                                                                                                                                                                |
| ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `method`, `evaluator`, `model`                                     | provenance; `model` is the gateway name, `baseline:<algo>`, `routing:<algo>`, or `canned`                                                                              |
| `graph`                                                            | `{graph_id, num_nodes, num_edges, directed}`; `num_edges` counts directed arcs (edge_index columns), matching the prompt stats                                         |
| `budget`, `budget_pct`                                             | the seed budget the run was constrained to, absolute and as % of `num_nodes`                                                                                           |
| `effective_budget`                                                 | seeds actually committable per episode. Equals `budget` for every method except `windowed`, whose budget is per window call by design — read this column before ranking a windowed arm against the others |
| `reward`, `spread_pct`                                             | ensemble-mean final spread under the inner-loop evaluator, absolute and as % of `num_nodes`                                                                            |
| `summary`                                                          | one-line trajectory summary (final spread, steps, per-step counts)                                                                                                     |
| `script`                                                           | the exact source of the winning strategy (per_step: last generated script)                                                                                             |
| `explanation`                                                      | the agent's own plain-English write-up of the search and the winning script, in markdown (§ headings below). `null` for baseline/routing arms, which synthesized nothing |
| `cost`                                                             | `{n_samples                                                                                                                                                            | mc_runs, env, reward_se, rollout_seconds}` for the winning trajectory |
| `timeline`                                                         | per-timestep log of the representative rollout: bag applied at `t` + post-step `infected`/`frontier` lists and counts; may be shorter than horizon (early termination) |
| `real_env_episodes`                                                | cumulative real-environment episodes consumed by inner-loop feedback (0 for `world_model`/`oracle`; the `--compare` referee replay is excluded)                        |
| `llm_transcript`                                                   | every turn verbatim — `{turn, kind, prompt, reply}` — with the model's prose intact. The code extractor keeps only the fenced block, but the prose around it is where the model says what it was trying to do: irrecoverable afterwards, and the first thing worth reading when a run goes wrong. Empty for canned arms |
| `llm_usage`                                                        | `{calls, prompt_tokens, completion_tokens, total_tokens, cost_usd}` summed over every provider the run created, the routing call included. `cost_usd` is `null` unless `--llm-price-in`/`--llm-price-out` were given — the lab gateway fronts Pro subscriptions and bills nothing per token, so there is no rate to assume. Token counts are exact either way, and also land in `summary.csv` |
| `seed`                                                             | base rollout seed for the run (`--seed`); each rollout additionally records the seed it actually used in its own `cost.seed`, and `mc_seed` / `wm_reeval_seeds` record the referee and re-evaluation seeds |
| `history`                                                          | per-outer-iteration `{iteration, reward, best, plan_seconds, rollout_seconds}` (evolve also logs `operator`; failed iterations carry `reward: null`, `error`, and their `repair` index). Empty for baseline/routing arms. Drives the convergence plot. `plan_seconds` is the generated algorithm's own compute — with free-mode composition scripts it dominates wall clock, and it is the only way to tell a slow-but-good strategy from a fast-but-lucky one |
| `referee_mc_runs`                                                  | with `--compare`: runs used by the ground-truth replay (`--referee-mc-runs`, else `--mc-runs`) — stays high even when a native arm's inner loop ran at `--mc-runs 1`   |
| `arm`, `arm_spec`, `condition`, `condition_name`, `budget_label`   | added when the run came from `pipeline.run`: which arm, which of the six baseline conditions, and which point of the budget sweep                                       |
| `credit_base_reward`, `credit`                                     | with `--credit`: paired-ablation base reward + per-action deltas                                                                                                       |
| `mc_reward`, `mc_spread_pct`, `mc_reward_se`, `mc_rollout_seconds` | with `--compare`: ground-truth replay of the winning strategy (absolute + % of `num_nodes`). The replay re-runs the exact **actions** the winner took, not the strategy — a script that samples (RIS with a live seed, a randomized local search) returns a different seed set on a second call, so re-planning would referee a strategy that never ran |
| `wm_minus_mc`                                                      | evaluator fidelity on this exact strategy — the trust meter                                                                                                            |
| `elapsed_seconds`                                                  | whole experiment including LLM calls                                                                                                                                   |

### The `explanation` write-up

After the refinement loop ends, one extra turn goes out **on the same conversation thread** (`Conversation.ask` — prose, no code extraction), so the model is describing the scripts it can still see rather than reconstructing them from the winner alone. `build_explanation_prompt` echoes the winning script anyway, because the winner is the max over iterations and is often *not* the last turn — without the echo the model narrates the wrong algorithm. It also passes the full `history`, which survives even when a long run trims the middle of the thread, and the prompt tells the model to say "no longer visible" rather than invent an iteration it cannot see.

The reply is printed to stdout and stored as `explanation`, in markdown under five fixed headings:

```
## Summary                        what the final algorithm is, what it scored
## Iteration log                  one ### per iteration: goal, actual code change, did it work
## How the final algorithm works  numbered walkthrough in execution order
## Why it beats the baseline      structural property exploited + supporting diagnostic
## Limitations                    where it fails, what to try next
```

Cost: one LLM call per agent arm per budget point. Canned arms (`--baseline`, `--routing`, external repos) skip it — `_CannedProvider` answers every prompt with its script, and a library algorithm has nothing to explain.

Timing semantics: `cost.rollout_seconds` vs `mc_rollout_seconds` is the WM-vs-MC speed comparison on the same strategy (normalize by ensemble size: `/n_samples` vs `/mc_runs`). For `per_step`, rollout time includes LLM latency — use `one_shot`/`windowed` for clean simulator timing.

---

## 10. Practical notes

- **BA graphs are degree-trivial.** On BA-100, degree ≈ CELF ≈ any portfolio; a strong agent will _find the ceiling_ (≈ the classical baselines), not beat it. Separation comes from community-structured graphs (WS/SBM/real), from temporal scheduling, and — once budgeted — from edge ops that no classical baseline uses.
- `per_step` cost scales as active-samples × timesteps LLM calls per evaluation; budget accordingly or use `--evaluator monte_carlo --mc-runs <small>` while prototyping it.
- `--outer-iters 1` disables the repair loop; ≥3 recommended for any live model (the first script is frequently imperfect, and the traceback-as-feedback path is what fixes it).
- The `.env` at the repo root provides `GATEWAY_BASE_URL`, `CLAUDE_GATEWAY_TOKEN`, `CHATGPT_GATEWAY_TOKEN` (gitignored; `chmod 600`).
- Old results JSONs archive scripts written against the former `Action` alias; the exec namespace now exposes `ActionOp` — re-generated scripts are unaffected, but replaying archived pre-rename scripts verbatim would fail.
