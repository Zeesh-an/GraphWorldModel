# Action-Conditioned World-Model Data (Influence Maximization)

Generates `(G, s_t, a_t, s_{t+1}, R)` transition data for training a graph **world model** on diffusion dynamics under node- and edge-level interventions. The world model learns the one-step transition `f(G, s_t, a_t) → s_{t+1}`; this package produces the supervised `(input, target)` pairs it is trained on.

- **Graph** `G` — a directed/undirected graph with per-edge IC transmission probabilities (`ic_probs`) and LT weights (`lt_weights`), saved once per graph.
- **State** `s_t = (infected, frontier)` — `infected` = every node ever activated; `frontier` = the nodes that activated _this step_ (IC: the status-1 spreaders; LT: the fresh wave that flipped this iteration).
- **Action** `a_t` — a bag of node/edge ops: `add_node`, `remove_node`, `add_edge`, `remove_edge`, `set_edge_weight` (empty bag = `NULL`). Which ops are used is set by `--action-ops`; omitting it generates diffusion-only data.
- **Next state** `s_{t+1}` — the realized next state, **plus** soft Monte-Carlo marginals `P(infected)` / `P(frontier)` per node (`--mc-marginals`), which are the actual training targets.
- **Reward** `R` — spread gain (Δ activated-node count this step).

Backbone simulator: **NDlib** IC/LT driven one step at a time, with mid-rollout `status` mutation (node ops) and live edge mutation (edge ops). Per-episode rollouts come from classical IM "spine" seed selectors plus random and counterfactual action injection (same-state / different-action coverage). See the design spec: `docs/superpowers/specs/2026-06-09-action-conditioned-wm-im-data-gen-design.md`.

---

## Quick start

> Generation is stage 1 of `python -m pipeline.run`, which also trains the world model, runs every agent arm, plots, and writes a report. Use the commands below when you want data generation on its own.

**Where things go.** Generated datasets land in `results/<task>/<dataset>/<run>/data/` (`--out-dir`, defaulting to `results/<dataset>/data`). Raw downloads land in `data/raw/<dataset>/` and are shared across every run — both are gitignored.

```bash
source .venv/bin/activate

# NOTE: run as modules from the repo root (python -m ...); the file-path form
# breaks the repo-root imports. Omitting --algorithms defaults to ALL SIX spine
# selectors — celf and local_search are MC-greedy and turn a minutes-long
# generation into hours. Pass the four analytic selectors for fast generation.

# Diffusion-only (no actions) — the default when --action-ops is omitted
python -m data.generate_wm_data --dataset ba --num-graphs 1

# Node action interventions
python -m data.generate_wm_data --dataset ba --num-graphs 1 \
    --action-ops add_node remove_node

# Edge action interventions
python -m data.generate_wm_data --dataset ba --num-graphs 1 \
    --action-ops add_edge remove_edge set_edge_weight

# Multi-graph synthetic set (trustworthy ranking) — 20 BA graphs, both dynamics,
# all five ops, four cheap selectors (the ba20_marg_structured recipe)
python -m data.generate_wm_data --dataset ba --num-graphs 20 \
    --action-ops add_node remove_node add_edge remove_edge set_edge_weight --models IC LT \
    --algorithms random degree pagerank betweenness \
    --out-dir results/influence_maximization/ba/default/data

# Real dataset (downloads on first use)
python -m data.generate_wm_data --dataset jazz --action-ops add_node remove_node add_edge remove_edge set_edge_weight

# Tiny end-to-end check
python -m data.generate_wm_data --smoke --out-dir /tmp/wm_smoke

# Validate a produced dataset (gate checks)
python -m data.validate_wm_data --dir results/influence_maximization/ba/default/data
```

---

## The generation pipeline, end to end

`generate_wm_data.py::run_generation` is the orchestrator. The flow:

```
for each graph G in the dataset:                 # _iter_bundles
    save G once to graphs/<graph_id>.npz         # GraphStore.save
    for model in {IC, LT}:                        # --models
        for algorithm in spine selectors:        # --algorithms
            for rollout in range(--rollouts):     # independent episodes
                split = assign train/val/test     # --split, RNG draw
                k = resolve seed budget           # _resolve_budget, per episode
                run one episode -> write transitions   # _episode_transitions
write graphs_index.json + metadata.json
```

### 1. Build the graph (`wm_graphs.py`)

`make_synthetic_bundle` / `make_real_bundle` produce a `GraphBundle`:

| Field         | Shape / type     | Meaning                                                      |
| ------------- | ---------------- | ------------------------------------------------------------ |
| `nx_graph`    | networkx graph   | the structure (directed for citations, undirected otherwise) |
| `edge_index`  | `(2, E)` int32   | `[src, dst]` arcs in COO form                                |
| `ic_probs`    | `(E,)` float32   | per-edge IC transmission probability `p(u→v)`                |
| `lt_weights`  | `(E,)` float32   | per-edge LT influence weight (`= ic_probs`)                  |
| `node_feats`  | `(N, F)` float32 | node features; synthetic graphs use `log1p(degree)` (`F=1`)  |
| `node_labels` | `(N,)` int32     | class labels (real datasets) or zeros                        |
| `ic_prob_map` | `dict[(u,v)→p]`  | edge→prob map the simulator configures NDlib with            |

Edge probabilities come from `graph_utils.build_edge_index`: by default the **weighted cascade** model `p(u→v) = 1 / in_degree(v)` (high-in-degree nodes are harder to activate per-edge). `--prob-model uniform` replaces this with a constant `--uniform-p` on every edge. LT weights are a copy of the IC probs (they already satisfy the LT requirement that incoming weights sum to ≤ 1 per node).

### 2. Pick the t=0 seed set (`wm_actions.py::select_seeds`)

Each episode commits a seed set chosen by one of the six **spine algorithms** (`SPINE_ALGORITHMS`). These span the cheap-but-weak to expensive-but-strong range so the dataset covers a spectrum of seed qualities:

| Algorithm      | How it picks k seeds                                                                |
| -------------- | ----------------------------------------------------------------------------------- |
| `random`       | k distinct uniform-random nodes                                                     |
| `degree`       | top-k by (out-)degree                                                               |
| `pagerank`     | top-k by PageRank (random-walk importance)                                          |
| `betweenness`  | top-k by betweenness centrality (nodes on many shortest paths)                      |
| `celf`         | greedy marginal-gain: add the node that most increases MC-estimated spread, ×k      |
| `local_search` | start from `degree`, then 1-swap seeds while estimated spread improves (≤ 3 rounds) |

`celf` and `local_search` call `estimate_spread`, an NDlib Monte-Carlo spread oracle (seed → diffuse to horizon, average final infected count over `mc_runs`).

### 3. Roll the episode forward (`generate_wm_data.py::_episode_transitions`)

A fresh `Simulator` is reset for the chosen dynamics, then stepped over `--horizon + 1` timesteps:

- **t = 0** the action _is_ the seed commit: `add_node` ops for every seed node.
- **t > 0** an action is sampled by `sample_injection`: with probability `1 - --inject-p` it is `NULL` (pure diffusion); otherwise one op is drawn uniformly from `--action-ops` with a random valid target on the live graph.
- The cascade is advanced one step (`Simulator.advance_marginal`) and the `(s_t, a_t, s_{t+1}, R)` record is written to the **main** branch.
- The loop stops early once the cascade is dead (no frontier) and no action is pending.

### 4. Counterfactual forks (same state, different action)

At intermediate steps, with probability `--cf-prob`, the simulator is snapshotted and `--cf-branches` alternative action bags (drawn by `counterfactual_actions`) are each applied from the _same_ `s_t`. Under `--remove-semantics blocked` a removal fork is a full node-deletion bag, so it mutates the graph as well as the status; `Simulator.restore()` rewinds status and the blocked set but not the graph, so the generator pairs every fork with `Simulator.revert_edges(bag)`, which re-adds the stripped arcs at their original probabilities. Each fork is written as a `cf_i` branch. This gives the trainer matched `(s_t, a, s_{t+1})` vs `(s_t, a', s'_{t+1})` pairs — the only signal that forces the model to be _action-conditioned_ rather than state-autoregressive, and the basis of the **action-sensitivity** eval metric.

### 5. Monte-Carlo soft marginals (`Simulator.advance_marginal`)

The single realized `s_{t+1}` is one Bernoulli draw from a stochastic process (for IC). Training on that single draw caps one-step accuracy at the label noise floor. Instead, `advance_marginal`:

1. applies the action once (the exogenous transition `T_exo` is deterministic),
2. snapshots the post-action status,
3. runs `--mc-marginals` independent diffusion draws (restoring status between draws), and
4. averages each node's activation frequency into a probability.

The result is the **true one-step marginal** `P(node infected at t+1)` and `P(node in frontier at t+1)`, stored sparsely as `{node: prob}`. IC uses all `--mc-marginals` draws (stochastic); LT uses a single draw (deterministic given its hidden thresholds). These soft marginals are the world model's regression targets and are **required** by the training pipeline (`--mc-marginals >= 1`, default 30).

---

## Output (`output/<dataset>/`)

| File                                            | Contents                                                               |
| ----------------------------------------------- | ---------------------------------------------------------------------- |
| `graphs/<graph_id>.npz`                         | one graph: `edge_index, ic_probs, lt_weights, node_feats, node_labels` |
| `graphs_index.json`                             | `graph_id` → metadata (type, directed, n_nodes, n_edges, …)            |
| `transitions_<IC\|LT>_<train\|val\|test>.jsonl` | one transition record per line                                         |
| `metadata.json`                                 | full generation config + episode count                                 |

Each transition row (JSONL):

```json
{
  "graph_id": "ba_n100_m3_s0", "diffusion_model": "IC",
  "episode_id": "ba_n100_m3_s0|IC|pagerank|k5|r2", "algorithm": "pagerank",
  "branch": "main", "t": 3,
  "state":      {"infected": [...], "frontier": [...], "infected_count": 6, "frontier_count": 2},
  "action":     [{"op": "add_node", "target": 17}],
  "next_state": {"infected": [...], "frontier": [...], "infected_count": 9, "frontier_count": 2},
  "reward": 3.0,
  "next_marginal_infected": {"4": 1.0, "9": 0.83, "12": 0.4, ...},
  "next_marginal_frontier": {"12": 0.4, "20": 0.27, ...}
}
```

- `branch` is `"main"` for the executed trajectory or `"cf_i"` for counterfactual forks (same `t` / `state`, different `action`).
- An empty `action` list is `NULL`.
- Edge ops carry the second endpoint as `"destination"` and, for `add_edge` / `set_edge_weight`, a `"weight"` (the IC transmission probability), e.g. `{"op": "set_edge_weight", "target": 4, "destination": 11, "weight": 0.62}`.
- `next_marginal_infected` / `next_marginal_frontier` are sparse `{node: prob}` maps (string keys, rounded to 6 dp) — the soft MC targets. They are present whenever `--mc-marginals >= 1` (the default).

### The same rows read as labelled `(x, y)` pairs (`--task source_localization`)

Source localization needs no new simulator, no new action op and no regeneration run: the labels are already on disk, and `world_model/wm_data.py::load_episode_endpoints` regroups them by `episode_id`. Two rules, and getting either backwards produces a plausible, silent, wrong label:

- **the source set `x` is the `t = 0`, `branch = "main"` record's `action`** — the seed commit, as a bag of `add_node` ops. Not its `state`, which is empty by construction.
- **the observation `y` is the LAST main record's `next_state`** (binarized) or its `next_marginal_infected` (continuous). Not its `action`, which is `NULL` by then.

Warning: The continuous form is the marginal of the **last step**, conditioned on the realized trajectory up to it — not `P(infected | x)` for the whole cascade. Every node infected earlier reads exactly `1.0` and only the final wave is fractional. It is still a continuous `y ∈ [0,1]^{|V|}`, which is the input type SL-VAE assumes, and it is strictly **more** informative than the single binary draw that literature observes, which is why both forms are kept and only the binarized one is comparable to a published table.

`branch = "cf_i"` rows are **skipped**: a counterfactual fork changes the action mid-episode, so its terminal state was not produced by the `t = 0` seed set alone. For the same reason the task registry pins `default_gen_action_ops = ()` for any `recover` task and asserts it — an episode carrying a mid-cascade injection has an observation its seed set did not cause, and its `(x, y)` pair would be a lie. With empty action ops `sample_injection` always returns `NULL` and `counterfactual_actions` produces nothing, so `--inject-p` and `--cf-prob` are inert rather than needing to be zeroed.

### Compartmental episodes (`--gen-models SIR|SIS|SEIR`)

`--task epidemic_control` swaps the NDlib simulator for `data/wm_epidemic.py`, and the reason is `research/epidemic_control.md` §2.2 rather than taste: NDlib's `SIRModel` / `SISModel` / `SEIRModel` all declare an EMPTY edge-parameter dict and compare a draw against one scalar `beta` per neighbour, so there is no per-arc transmission probability. That deletes `set_edge_weight` (the whole graded contact-reduction branch of that literature), degenerates `GraphInput.edge_weight` to ones, and makes `structured_residual` — an anchor on `logit(w)` — inexpressible. Our stepper is four lines of transition rule and recovers all three.

An episode's `algorithm` is a PAIR here, `<outbreak selector>+<immunizer selector>`, exactly as a competitive episode's is: the outbreak model is a second experimental axis a seeding task does not have, and the `none` immunizer leaves the outbreak unprotected and supplies the reference every prevented-infections number divides by. The t=0 bag carries the index cases as `add_node` AND the doses as full deletion bags, so the head's `T_exo` sees a vaccinated node's arcs already gone from `edge_index`.

Records gain four marginals — `next_marginal_incidence`, `_exposed`, `_infectious`, `_recovered` — and `next_marginal_infected` stays the EVER-infected one, which is what lets every existing reader keep working. `state` gains `exposed` and `recovered`, both omitted when empty so an IC/LT JSONL is byte-identical. Three rates land in `metadata.json` (`epi_beta`, `epi_gamma`, `epi_alpha`) and `train_wm` reads them back rather than taking a flag: §8.2 trap 2 records that beta and gamma are free parameters nobody standardizes, so a table that fixes them without stating them is comparable only to itself, and a head whose gamma disagrees with the simulator that made its targets is fit against a transition that never happened.

`--epi-gamma 1.0` under SIR **is Independent Cascade** — a node transmits once and is then spent — which `coding_agent/check_epidemic_control.py` measures against NDlib rather than asserting.

### ...and as whole HISTORIES (`--task cascade_reconstruction`)

Cascade reconstruction reads the same rows one level up: not the two ends of an episode but every step in between, regrouped by `world_model/wm_data.py::load_episode_trajectories`. The activation time of a node is the `t` at which it first appears in a main record's `next_state.frontier`, which needs no extra bookkeeping — the `frontier` channel has been on disk all along.

What it DOES need is a field nothing else reads. **`--trace-parents` records who infected whom**, and it exists because NDlib does not produce that at all: `IndependentCascadesModel.iteration` sets `actual_status[v] = 1` on a successful coin flip without recording the responsible `u`. `data/wm_simulator.py` therefore ships `TracedICModel` and `TracedThresholdModel`, built by re-classing a base instance rather than by subclassing (NDlib's own `__init__` calls `super(self.__class__, self)`, which recurses forever from a subclass), and every record then carries:

```json
"parents": {"16": [45], "28": [35], "47": [46]}
```

Three rules, and each one is load-bearing:

- **an EMPTY list marks a source.** An `add_node` is an injection, not a transmission, so the seeds of the `t = 0` bag are recorded with no cause at all — that is what makes `parent = None` mean "source" in the decoder's own contract.
- **under IC there is exactly one parent, and it is biased.** NDlib iterates spreaders in NODE ORDER and skips any `v` already flipped this step, so the recorded parent is *the first successful `u` in node order* rather than a uniformly random one among the successes. Real, documentable, and stated wherever a tree number is.
- **under LT the value is a SET.** Activation there is a threshold crossing over the whole active in-neighbourhood, so there is no single transmitting edge; the whole active neighbourhood is recorded and a predicted parent counts as correct if it is a member. That makes LT's path precision structurally easier than IC's, and the two are never compared.

Tracing is off by default because it costs an append per successful flip and only one task scores it. `pipeline.run` turns it on from the task registry, so `--task cascade_reconstruction` gets it without a flag; `data.generate_wm_data --task cascade_reconstruction` does the same for a standalone invocation. A dataset generated without it makes `PathPrecision` unscoreable, and `coding_agent/reconstruction.py::load_cascades` **raises** rather than silently falling back to the node half — which is the exact failure the tree-weighted reward exists to prevent.

### ...and REPLAYED FROM A REAL LOG (`--task cascade_prediction`)

Everything above describes transitions a simulator produced. Cascade prediction is the one task whose transitions did not come from one at all: `data/wm_cascades.py` replays a **real observed cascade corpus** into the identical JSONL, so `wm_data.py`, the feature builder, the heads and the training loop need no branch — and `run_generation` DELEGATES to it rather than branching, because there is no seed selector, no injection, no counterfactual fork and no Monte-Carlo marginal on that path at all.

Four differences, each forced rather than chosen (`research/cascade_prediction.md` §2.1, §2.4, §8.3):

- **The action is `NULL` at every step.** Nothing intervenes; the cascade is only watched. The `t = 0` record still carries the root as an `add_node` bag — that is how every reader here recovers an episode's sources — and there is no injection at any later step and no fork, because a log has nothing to fork on.
- **The targets are HARD.** Our soft targets are `P(infected)` averaged over `--mc-marginals` re-runs, and a real cascade **happened once**, so `next_marginal_infected` is the realized 0/1 indicator. §2.4 calls this the main technical risk of the whole exercise and §11 records that how much it costs our one-step `delta_f1` is unestablished by anything in that literature — they never had soft targets to lose.
- **Elapsed time is BINNED.** The corpora publish seconds (or DAYS, for APS) since publication and our simulator is discrete-time. `--cp-step` is the bin width, derived from the OBSERVATION window rather than the horizon: these corpora observe a tiny fraction of their horizons (Weibo 0.5 h of 24 h), so binning the horizon uniformly would leave the prefix with one step and delete the wave series a predictor reads.
- **A replayed episode stops only when NOTHING LATER is non-empty.** The simulator breaks on an empty frontier because that is a fixed point of a monotone cascade; a real log is not a Markov process and routinely goes quiet for a bin and resumes, so the simulator's rule would truncate every cascade at its first lull.

`metadata.json` gains an `observed` block — corpus, time unit, both windows, the step, the split protocol, the participant filter, the truncation, the cascade counts and `hard_targets: true` — and that block is the ONLY thing that says `transitions_IC_train.jsonl` was not simulated. `world_model.wm_data` reads it through `data.wm_cascades.dataset_is_observed`, and `coding_agent.prediction.load_forecasts` **raises** on a simulated dataset rather than scoring it, because §2.2's whole argument collapses if the "real cascade" is an NDlib rollout.

**The dynamics label names the KERNEL, not the source.** A replayed corpus is written under `--diffusion-model IC` and fits the IC head, and that is the experiment rather than a mislabelling: §2.2 records that the IC composition rule is a modelling commitment rather than a learned fact, so fitting it to real retweets is exactly the falsification test. Only ONE dynamics is written per replay — the transitions are identical whatever kernel label they carry, so writing both IC and LT would double the file for no second experiment.

**A cascade corpus loader exposes three functions rather than two** (`data/datasets/cascade_common.py`): the usual `download_<name>()` and `load_<name>(path)`, plus `load_<name>_cascades(path) -> list[Cascade]`. Both halves come from one parse so a node id means the same thing in each, and the graph is built FROM the observed propagation paths for every corpus but Digg, which publishes a real friendship network and uses it. Eight are registered in `wm_graphs.cascade_corpora`: `casflow_weibo` / `casflow_twitter` / `casflow_aps` (one manual Drive bundle, three corpora, and the canonical five-field line format every repo in this literature reads), `weibo_cascades` and `aps` (the raw publisher routes, both manual), and `digg_cascades`, `memetracker` and `taoke` (auto-downloading).

---

## The 5 action ops (identical set for every dataset)

The action space is unified across tasks: every action is `(op, target, [destination], [weight])`. `target` is the node (or edge source `u`); `destination` is the edge sink `v`; `weight` is the IC transmission probability for the edge.

| op                | fields in the record                | IC effect                                    | LT effect                              |
| ----------------- | ----------------------------------- | -------------------------------------------- | -------------------------------------- |
| `add_node`        | `target=v`                          | status→1 (node becomes a spreader next step) | status→1                               |
| `remove_node`     | `target=v`                          | depends on `--remove-semantics`, see below   | depends on `--remove-semantics`        |
| `add_edge`        | `target=u, destination=v, weight=w` | add arc u→v, set transmission p=`w`          | add edge structurally (weight ignored) |
| `remove_edge`     | `target=u, destination=v`           | remove arc u→v                               | remove edge structurally               |
| `set_edge_weight` | `target=u, destination=v, weight=w` | set arc u→v transmission p=`w`               | **no-op** (LT ignores edge weights)    |

### `--remove-semantics`: what `remove_node` means

Two readings, and they are different problems. The value is recorded in `metadata.json`, and `train_wm.py` refuses to train a head whose `T_exo` disagrees with it.

| | `spent` (default) | `blocked` |
| --- | --- | --- |
| reading | the spreader is used up | the node is deleted from the graph |
| IC status | `2` (Removed) | `2`, plus tracked in `Simulator.blocked` |
| LT status | `0` (Susceptible) | `0`, plus tracked in `Simulator.blocked` |
| counted as infected? | **yes** under IC, no under LT | **no**, under both |
| can transmit? | no | no |
| can (re-)activate? | no under IC, **yes** under LT | no, under both |
| edges removed? | no | **yes** (the bag carries them) |
| right for | influence maximization | containment: critical node detection, influence blocking, immunization |

Under `spent`, pre-emptively removing `k` susceptible nodes under IC inflates the measured final spread by exactly `+k`, because `active_nodes()` counts NDlib status `2`. That is correct for "this spreader has already spent its shot" and wrong for any minimize-the-spread objective.

Under `blocked`, `wm_actions.delete_node_bag()` expands one deletion into `remove_node(v)` plus a `remove_edge` per incident arc, so the whole existing pipeline handles it unchanged: `build_features` marks `CH_EDGE`, `reconstruct_episode_adjacency` replays the removals, and both rollout paths call `apply_edge_ops`. No sixth op. `Simulator._enforce_blocked()` additionally holds the node down after each step, so a bare `remove_node` from a hand-written strategy is still correct.

Two consequences worth knowing before reading a `blocked` dataset:

- Injected removals target **susceptible** nodes (containment blocks ahead of the cascade), not active ones as `spent` does.
- Counterfactual forks DO carry removals, as full deletion bags, and the generator reverts their edge deletions afterwards (see step 4). Skipping the fork instead — which is what this used to do, on the argument that `restore()` cannot rewind the graph — left a containment dataset with no two actions from the same state, so `action_sensitivity` read exactly **0.0**. `data/check_remove_semantics.py::revert_edges_undoes_a_deletion_fork` is the guard.

The three data "settings" are just which ops you pass to `--action-ops`:

- **Setting 1 (diffusion-only):** omit `--action-ops`
- **Setting 2 (node):** `--action-ops add_node remove_node`
- **Setting 3 (edge):** `--action-ops add_edge remove_edge set_edge_weight`
- **Containment (critical node detection):** `--action-ops remove_node --remove-semantics blocked`. Through `pipeline.run --task critical_node_detection` neither flag is needed: `pipeline/tasks.py` supplies both, so the ops the data teaches and the ops the planner may emit come from one registry entry and cannot disagree.

How these become model inputs is documented in [`world_model/README.md`](../world_model/README.md): node ops set the `act_add` / `act_remove` input channels, and edge ops set the `act_edge_endpoint` channel and mutate the per-episode adjacency.

---

## `--competitive`: two cascades (influence blocking)

`--competitive` swaps the NDlib `Simulator` for [`data/wm_competitive.py`](wm_competitive.py)`::CompetitiveSimulator` and generates **two-cascade** episodes. Through `pipeline.run --task influence_blocking` the flag is supplied by the registry, so nothing has to be passed by hand.

What changes:

- **A rumour `S_N` is committed at `t=0` by `reset()`, not by an action.** `--negative-pct` sizes it (1% of `N` by default) and `--negative-selectors` chooses it — the spine selectors again, because the published attacker models are exactly `random`, `degree`, `pagerank` and IMM. `s_0` is therefore **not empty**, unlike every other task: the rumour moved first, which is the premise.
- **`add_node` seeds the POSITIVE cascade.** The blocker only ever helps itself; there is no action that seeds the rumour. `--blocker-selectors` picks each episode's `t=0` blocker set, and `none` is on that list on purpose — those episodes are the unopposed `σ(S_N, ∅)` reference every prevented-influence number divides by.
- **Four soft targets per step instead of two.** `next_marginal_pos_infected` / `next_marginal_pos_frontier` join the existing pair, and every record carries the episode's `negative_seeds`.
- **`State` gains `pos_infected` / `pos_frontier`.** They are omitted from the JSONL when empty, so a single-cascade dataset is byte-identical to what it was.

Two dynamics parameters are recorded in `metadata.json` under `competitive`, because both move the published numbers and most papers state neither:

| Flag | Meaning |
| ---- | ------- |
| `--tie-break` | which cascade wins a node both reach on the **same step**: `negative` / `positive` / `fixed` dominance. `auto` (default) resolves per dynamics to that dynamics' own founding paper — positive under IC (Budak), negative under LT (He et al.'s CLT) |
| `--positive-prob` | `shared` (default) is **COICM**, one probability per edge independent of information type; a float is **MCICM**, and `1.0` is Budak's high-effectiveness property |

`train_wm.py` reads all three (`competitive`, `tie_break`, `positive_prob`) back out of `metadata.json` rather than taking them as flags: a head whose tie-break disagrees with the simulator that produced its targets is fit against a transition that never happened, and nothing in the loss curve would say so.

```bash
python -m data.generate_wm_data --task influence_blocking --dataset email_eu_core \
    --competitive --remove-semantics blocked --negative-pct 1.0 \
    --negative-selectors random degree pagerank \
    --blocker-selectors none random proximity degree \
    --action-ops add_node remove_node remove_edge set_edge_weight --models IC
```

---

## Diffusion dynamics

| Model  | Determinism                    | Per-step rule                                                                                                                                                                                                                      |
| ------ | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **IC** | stochastic                     | each newly infected `u` gets one try to infect each out-neighbor `v` with prob `p(u→v)`; success → `v` infected. Monotone (no node ever de-activates). Status: 0 Susceptible, 1 Infected (spreader, this step), 2 Removed (spent). |
| **LT** | deterministic given thresholds | `v` activates when the summed weight of its active in-neighbors ≥ its threshold `θ_v`. Thresholds are drawn `U(0,1)` per node **per episode and not stored**, so from a state-only view LT looks stochastic.                       |

This determinism difference is why IC averages over `--mc-marginals` draws while LT uses a single draw, and why the world model uses a different structured head per dynamics (see `world_model/README.md`).

---

## Customizable probabilities / knobs

| Flag                              | Default         | What it controls                                                                                           |
| --------------------------------- | --------------- | ---------------------------------------------------------------------------------------------------------- |
| `--dataset`                       | `cora_ml`       | real (`jazz, email_eu_core, netscience, cora_ml, facebook, power_grid, ca_grqc, wiki_vote, lastfm_asia, nethept, netphy, epinions, twitter, digg, youtube, orkut, livejournal, weibo`) or synthetic (`er, ba, ws, sbm, powerlaw_cluster, kronecker, karate`); the choices list is derived from `wm_graphs.real_directed`, so adding a loader adds a choice. `weibo` needs a manual AMiner download (see `data/datasets/weibo.py`). Full catalogue: `research/influence_maximization.md` §6 |
| `--plc-m` / `--plc-p`             | `2` / `0.05`    | `powerlaw_cluster`: edges per new node and the triangle-closing probability. RL4IM quotes average degree 3, which this generator (avg ≈ `2m`) cannot hit exactly at integer `m`; see `research/adaptive_online_im.md` §6.3 |
| `--kron-variant`                  | `core_periphery`| `kronecker` seed matrix: `core_periphery`, `random`, or `hierarchical` (ConTinEst's three). Generated at the next power of two and induced down to `--syn-nodes`; `O(N²)` sampling, guarded at 20K |
| `--num-graphs`                    | `1`             | number of synthetic graph instances (folded into the seed)                                                 |
| `--syn-nodes`                     | `100`           | nodes per synthetic graph                                                                                  |
| `--models`                        | `IC LT`         | which dynamics to generate transitions for                                                                 |
| `--prob-model {weighted,uniform}` | `weighted`      | IC prob: `weighted` = 1/in_degree(v); `uniform` = constant `--uniform-p`                                   |
| `--uniform-p`                     | `0.1`           | the constant IC prob (and LT weight) when `--prob-model uniform`                                           |
| `--budget-pct-range LO HI`        | `1 20`          | **default** — draw k ~ U(LO%, HI% of N) per episode, so one WM covers a whole budget sweep                  |
| `--no-budget-range`               | `False`         | disable the range and use the two flags below instead                                                      |
| `--budget` / `--budget-pct`       | `5` / `None`    | fixed seed-set size k for the whole run; only consulted with `--no-budget-range` (pct overrides absolute)   |
| `--algorithms`                    | all 6 spine     | which seed selectors to roll out                                                                           |
| `--rollouts` / `--horizon`        | `10` / `10`     | episodes per (graph, model, algo) / max timesteps                                                          |
| `--inject-p`                      | `0.3`           | P(an intermediate step injects an action at all vs NULL)                                                   |
| `--action-ops`                    | `[]`            | which ops to inject; empty = diffusion-only                                                                |
| `--weight-lo` / `--weight-hi`     | `0.0` / `1.0`   | range for `add_edge`/`set_edge_weight` weights, sampled `U(lo, hi)`                                        |
| `--cf-prob`                       | `0.2`           | P(spawn counterfactual forks at a step)                                                                    |
| `--cf-branches`                   | `2`             | # alternate-action branches per fork                                                                       |
| `--mc-marginals`                  | `30`            | MC draws per step to estimate soft next-step marginal targets (1 = single-draw binary target)              |
| `--split`                         | `0.7 0.15 0.15` | train / val / test episode split probabilities                                                             |
| `--er-p`                          | `0.05`          | ER edge probability G(n, p) — structural                                                                   |
| `--ba-m`                          | `3`             | BA attachment count (not a prob)                                                                           |
| `--ws-k` / `--ws-p`               | `6` / `0.1`     | WS ring degree / rewire probability (structural)                                                           |
| `--sbm-blocks`                    | `4`             | SBM community count (block sizes split `--syn-nodes` evenly)                                               |
| `--sbm-p-in` / `--sbm-p-out`      | `0.15` / `0.01` | SBM within-block / cross-block edge probability                                                            |
| `--seed`                          | `42`            | master RNG (graph + selection + injection + sim all derive from it)                                        |

> LT node thresholds are drawn `U(0, 1)` per node internally — no CLI flag, and they are not stored (the world model must learn the threshold _marginal_).

---

## Validation (`validate_wm_data.py`)

`python -m data.validate_wm_data --dir <output_dir>` runs post-hoc gate checks on a produced dataset and prints a JSON summary:

| Check                        | What it confirms                                                                                                             |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| reward spread                | rewards are not collapsed to a constant (mean/std/min/max)                                                                   |
| `main_monotone_ratio`        | main-branch `infected_count` is non-decreasing step to step (IC sanity)                                                      |
| `action_sensitivity_pairs`   | groups where same `(graph, episode, t, state)` + different action → different `next_state` (counterfactual coverage is real) |
| `per_algorithm_final_spread` | mean terminal spread per spine algorithm (eyeball `random < degree/pagerank < celf/local_search`)                            |

---

## Modules

| Module                | Responsibility                                                                                                    |
| --------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `wm_graphs.py`        | graph providers (real + synthetic) + edge probabilities → `GraphBundle`                                           |
| `wm_simulator.py`     | `State` / `ActionOp` types + NDlib stepwise IC/LT sim with action injection, `advance_marginal`, snapshot/restore |
| `wm_actions.py`       | spine seed selectors + MC spread oracle + injection schedule + counterfactual candidates                          |
| `wm_cascades.py`      | **CP**: replay a REAL logged corpus as `(s_t, NULL, s_{t+1})` — the leak-free chronological split, the binning, hard targets, the `observed` metadata block |
| `graph_utils.py`      | adjacency → `edge_index` + IC/LT edge probabilities (shared)                                                      |
| `generate_wm_data.py` | graph store + JSONL transition writer + CLI orchestrator                                                          |
| `validate_wm_data.py` | post-hoc gate-check harness                                                                                       |
| `datasets/`           | per-dataset download + load helpers (lazy-imported by `wm_graphs.py`)                                             |
| `datasets/cascade_common.py` | the CASCADE-corpus contract: `Cascade`, CasFlow's canonical line format, the published filters, the graph built from observed paths |
| `old/`                | **legacy** diffusion-only CND/IM/SL generators (archived; superseded by `generate_wm_data.py`)                    |
