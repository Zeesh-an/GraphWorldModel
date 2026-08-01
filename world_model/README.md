# Action-Conditioned Graph World Model

This package trains and evaluates the world model

```
f_θ(G, s_t, a_t) → s_{t+1}
```

an autoregressive, discrete-time transition model of graph diffusion under interventions. It is trained **teacher-forced one-step** on the `(G, s_t, a_t, s_{t+1})` transitions produced by `data/generate_wm_data.py` (see [`data/README.md`](../data/README.md)), and is **plug-and-play across five GNN backbones** (`gcn`, `sage`, `gt`, `gat`, `gcnii`).

The transition factorizes as `s_{t+1} = T_endo(T_exo(s_t, a_t))`:

- **`T_exo`** — the deterministic, immediate effect of the action (seed a node, drop a node from the frontier, add/remove/reweight an edge).
- **`T_endo`** — the (possibly stochastic) diffusion step that follows.

The model learns both. The headline use case is as a fast, differentiable **simulator**: roll it forward to predict cascades, or score one-step interventions for planning — without running the expensive classical simulator at inference time.

---

## Pipeline at a glance

```
JSONL transitions ──▶ TransitionDataset ──▶ collate (block-diagonal batch)
   (results/<task>/<dataset>/<run>/data)        │                       │
                          ▼                       ▼
                   build_features            build_graph_input
                   X:(N,6), y_inf,y_fr       adj_norm, edge_index, edge_weight
                          │                       │
                          └──────────┬────────────┘
                                     ▼
                            WorldModel.forward
                       encoder(X, graph) ─▶ h:(N,H)
                              head(h, X, graph) ─▶ logits:(N,2)
                                     │
                          BCEWithLogits(·, y_inf) + BCEWithLogits(·, y_fr)
```

Train: `train_wm.py`. Model + heads: `wm_model.py`. Data/feature plumbing: `wm_data.py`. Metrics: `wm_metrics.py`. Evaluation (one-step, rollout, planning): `wm_eval.py`. Standalone re-eval CLIs: `eval_planning.py`, `eval_rollout_ensemble.py`, `eval_structured_oracle.py`.

---

## The model input `X` — shape `(N, 6)`

Every transition is turned into a per-node feature matrix `X` with one row per node and **6 channels** (`wm_data.py::build_features`, `IN_CHANNELS = 6`). All channels describe the situation at time `t`, _before_ diffusion:

| col | name (`CH_*`) | represents                                                                 | type             | set from                          |
| --- | ------------- | -------------------------------------------------------------------------- | ---------------- | --------------------------------- |
| 0   | `CH_INFECTED` | node is infected (ever-activated) at time `t`                              | binary 0/1       | `record["state"]["infected"]`     |
| 1   | `CH_FRONTIER` | node is in the current spreading wave (frontier) at time `t`               | binary 0/1       | `record["state"]["frontier"]`     |
| 2   | `CH_DEGREE`   | `log1p(total degree)` of the node in the time-`t` graph `A_t`              | continuous float | counts in `edge_index` (in + out) |
| 3   | `CH_ADD`      | node is the target of an `add_node` op this step (just seeded)             | binary 0/1       | the action bag                    |
| 4   | `CH_REMOVE`   | node is the target of a `remove_node` op this step                         | binary 0/1       | the action bag                    |
| 5   | `CH_EDGE`     | node is an endpoint of an edge op (`add/remove/set_edge_weight`) this step | binary 0/1       | both `target` and `destination`   |

Channels 0–1 are the **state** `s_t`. Channel 2 is **structure**. Channels 3–5 are the **action** `a_t` projected onto nodes — this is what makes the model _action-conditioned_. Edge ops additionally change the graph itself (next section), so channel 5 flags the touched nodes while the adjacency carries the actual structural change.

### Targets — `y_inf`, `y_fr`, each shape `(N,)`

The regression targets are the **soft Monte-Carlo marginals** from data generation: `y_inf[v] = P(v infected at t+1)` and `y_fr[v] = P(v in frontier at t+1)`. They are read from `next_marginal_infected` / `next_marginal_frontier` and scattered into dense `(N,)` vectors. `build_features` **requires** these soft targets and raises `KeyError` if a record lacks them (regenerate with `--mc-marginals >= 1`). Training on the marginal — not a single Bernoulli draw — is what lifted one-step `delta_f1` from ~0.52 to ~0.83 on IC.

---

## Graph structure into the model — `GraphInput`

`wm_data.py::build_graph_input` turns the per-transition edge list into a `GraphInput` with three views the backbones consume:

| field         | shape / type    | meaning                                                                                                                                                                      |
| ------------- | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `adj_norm`    | sparse `(N, N)` | symmetrically renormalized adjacency `D^{-1/2}(A+I)D^{-1/2}` (with self-loops); row = `dst`, col = `src`, so a matmul aggregates **from in-neighbors**. Used by GCN / GCNII. |
| `edge_index`  | `(2, E)` long   | `[src, dst]` arcs (no self-loops added here). Used by SAGE / GAT / GT.                                                                                                       |
| `edge_weight` | `(E,)` float    | per-edge weight: **IC** keeps `p(u→v)`; **LT** uses all-ones (LT ignores edge weights, relying on structure + hidden thresholds).                                            |

The normalization uses the **weighted** in-degree, so IC transmission probabilities flow through the propagation, not just connectivity.

### Per-episode adjacency reconstruction (edge actions)

For diffusion-only / node-action data, every step reuses the graph's base edges. For **edge-action** episodes the graph changes over time, so `wm_data.py::reconstruct_episode_adjacency` replays the edge ops to build the correct `A_t` for each `(t, branch)`:

- **main** transitions see the **post-action** graph at their step (cumulative edge ops through `t` inclusive);
- **counterfactual** (`cf_i`) transitions branch from the **pre-step** graph (cumulative edge ops through `t-1`).

This guarantees the model is always shown the adjacency that actually produced the recorded `s_{t+1}`.

### Batching

`collate_transitions` stacks `B` transitions into one **disjoint block-diagonal** graph: node ids are offset per sample, `X`/`y` are concatenated, and a single `GraphInput` is built over the union. Because the blocks are disconnected, message passing never crosses sample boundaries — it is exactly equivalent to running each graph independently, but in one batched forward pass.

---

## The backbones (encoders) — `X (N,6) → h (N, H)`

All five live in `world_model/model/` and share the encoder interface `forward(X, graph) → (N, hidden_dim)`. Each is a stack of pre-norm residual blocks ending in a `LayerNorm`. (Each file also contains a legacy `*ForwardModel` class from the old seed→outcome pipeline; the world model uses only the `*Encoder` classes.) The encoder is selected by `--model` via the `BACKBONES` registry in `wm_model.py`.

| backbone                                       | `--model` | what it does                                                                                                                                                                                                      | graph view used             |
| ---------------------------------------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| **GCN** (`gcn.py`)                             | `gcn`     | Kipf-Welling: `h = sparse.mm(adj_norm, h)` then `Linear → GELU → Dropout`, pre-norm residual.                                                                                                                     | `adj_norm`                  |
| **GraphSAGE** (`graphsage.py`)                 | `sage`    | mean aggregator, concat variant: `h_neigh = weighted_mean(h[src])` (self-loops stripped), `h = Linear(concat(h, h_neigh))` (2H→H) → GELU → Dropout, residual. Self enters via the concat, neighbors via the mean. | `edge_index`, `edge_weight` |
| **GATv2** (`gat.py`)                           | `gat`     | dynamic attention `e_uv = aᵀ·LeakyReLU(W_l h_v + W_r h_u)`, multi-head scatter-softmax over incoming edges; `edge_weight` enters as a `log`-bias on the scores; self-loops added so each node attends to itself.  | `edge_index`, `edge_weight` |
| **Graph Transformer** (`graph_transformer.py`) | `gt`      | scaled dot-product attention `Q_v·K_u/√d_k` per edge, scatter-softmax, position-wise FFN, pre-norm residual blocks; `edge_weight` as `log`-bias; self-loops added.                                                | `edge_index`, `edge_weight` |
| **GCNII** (`gcnii.py`)                         | `gcnii`   | initial-residual + identity mapping: `support = (1-α)·adj·h + α·h⁰`, `h = ReLU((1-β_l)·support + β_l·W·support)`, `β_l = log(λ/l + 1)`. Built to go deep without oversmoothing.                                   | `adj_norm`                  |

Shared knobs: `--hidden-dim` (`H`), `--n-layers`, `--dropout`; attention models take `--n-heads` (and GT `--ffn-dim`); GCNII takes `--gcnii-alpha`, `--gcnii-lamda`. There is **no positional encoding** in the world-model encoders — positional/degree information enters through `CH_DEGREE` instead. Recommended sizes scale with graph size (hidden 64 → 128 → 256 from <1K to >10K nodes; GCNII goes deeper, 8 → 16 layers).

---

## The output heads — `h (N,H) → logits (N,2)`

`wm_model.py` defines three heads, selected by `--head` (and the dynamics). All return **logits** `(N, 2)` = `[next_infected_logit, next_frontier_logit]`, so training (`BCEWithLogits`) and eval (`sigmoid`) are head-agnostic.

### `linear` (free head)

`nn.Linear(H, 2)` — each node embedding maps directly to two logits. Maximally flexible, but on a free-running rollout it learns "more active → more spread" with no structural cap and **saturates** to the whole graph on its own out-of-distribution states. Fine for one-step metrics; unfaithful as a simulator.

### `structured` IC — `ICTransmissionHead`

Predict the **mechanism**, not the state. The head:

1. applies `T_exo` from the action channels: `infected = clamp(CH_INFECTED + CH_ADD)`, `frontier = clamp(CH_FRONTIER + CH_ADD)·(1 − CH_REMOVE)`. Under `--remove-semantics blocked` the first term also picks up `·(1 − CH_REMOVE)`, because a blocked node has left the graph and stops counting (see [`data/README.md`](../data/README.md));
2. predicts a per-edge transmission propensity `q(u→v) = sigmoid(MLP([h_u, h_v, w_uv]))`;
3. gates by an active source `t_uv = q_uv · frontier_u` and derives the IC infection form `p_new(v) = 1 − ∏_{u→v} (1 − t_uv)` (via a stable log-sum-exp scatter);
4. composes the next state monotonically: `y_inf = infected + (1 − infected)·p_new`, `y_fr = (1 − infected)·p_new`.

Locality + the `frontier_u` gate make this **structurally unable to saturate**: a susceptible node with no active in-neighbor has `p_new = 0`, so the cascade self-terminates. This is the fix that took IC rollout `count_bias` from ~+49 (linear) to ~−0.6 while preserving one-step `delta_f1 ≈ 0.83`. **Train it with `--pos-weight off`** — `pos_weight` globally inflates `q` and collapses one-step accuracy; the structural form already prevents the all-zeros degenerate.

### `structured` LT — `LTThresholdHead`

LT thresholds are hidden and re-drawn per episode, so the head predicts the activation _probability_ as a learned monotone function of the active-neighbor fraction:

- `f_v = (active in-neighbor weight) / (total in-neighbor weight)`,
- `p_new(v) = [f_v > 0]·sigmoid(τ·(f_v − θ̂_v))`, with per-node `θ̂_v = sigmoid(Linear(h_v))` and learned sharpness `τ = softplus(param)`,
- `T_exo`: `add_node → active`, `remove_node → susceptible`. One expression covers both remove semantics: under `spent` the node resets to status 0 and may re-activate through its intact edges; under `blocked` those edges are gone from `edge_index`, so `f_v = 0` and the gate holds it down. Unlike the IC head, this one needs no `remove_semantics` branch.

The `f_v > 0` gate gives the same self-terminating bound as IC. One-step metrics are looser than IC by design (the best a state-only model can do is the threshold marginal `P(activate | f_v)`), but the rollout is faithful.

### `structured_residual` (IC only, `--head structured_residual`)

`ICTransmissionHead(residual=True)`: `q = sigmoid(logit(w) + MLP([h_u, h_v, w]))`. The MLP learns a residual correction on the true IC transmission prob, so zero correction reproduces the oracle exactly. Use when the training data lacks edge-weight diversity (e.g. node-op-only action sets): the plain `structured` head's `w → q` mapping is then unanchored and drifts optimistic in free-running rollouts (count_bias ≈ +2) despite equal one-step metrics.

### `structured_oracle` (validation only)

`ICTransmissionHead(oracle=True)`: skips the MLP and sets `q = edge_weight` (the true IC transmission prob). No learning — it validates that the IC structural form itself is correct (expected `count_bias ≈ 0`) before trusting a learned head. IC-only (LT thresholds aren't stored). Exposed via `eval_structured_oracle.py`, not `train_wm.py`.

`WorldModel.forward` runs `h = encoder(X, graph)`, then `head(h)` for `linear` or `head(h, X, graph)` for the structured heads.

---

## Training — `train_wm.py`

**Teacher-forced one-step.** Each batch is a set of true `(s_t, a_t)` inputs; the model predicts the next-state marginals and is scored against the MC targets:

```
logits = model(X, graph)                                  # (N, 2)
loss   = BCEWithLogits(logits[:,0], y_inf)                # next-infected
       + BCEWithLogits(logits[:,1], y_fr)                 # next-frontier
```

- **Class imbalance.** Diffusion changes are sparse (few new infections per step), so `--pos-weight auto` up-weights the positive class (ratio clamped to `[1, 50]`) to stop a collapse to all-zeros. Use `--pos-weight off` for the structured heads (see above).
- **Optimizer.** Adam, `--lr` (1e-3), `--weight-decay` (5e-4), `--batch-size`.
- **Model selection.** Validate every epoch with `evaluate_one_step`; checkpoint the best **val `delta_f1`**; early-stop after `--patience` epochs without improvement.
- **Final report.** Reload the best checkpoint and write a results JSON with four blocks: `history` (per-epoch train loss + val `delta_f1`), `test` (one-step), `rollout` (stochastic ensemble), and (with `--plan-demo`) `planning` (multi-graph regret). Checkpoint saved to `wm_<model>_<diffusion>.pt`.

**Output paths.** `--ckpt-dir` defaults to a sibling of the data directory — `results/<task>/<dataset>/<run>/data` puts the checkpoint and results JSON in `results/<task>/<dataset>/<run>/world_model/`. `--results` defaults to `<ckpt-dir>/<model>_<diffusion>.json`. Pass either explicitly to override.

Example:

```bash
python -m world_model.train_wm \
    --data-dir results/influence_maximization/ba/default/data --diffusion-model IC \
    --model sage --head structured --pos-weight off \
    --hidden-dim 64 --n-layers 3 --epochs 400 --batch-size 32 --patience 50 \
    --seed 42 --device cuda --plan-demo --plan-graphs 5
# -> results/influence_maximization/ba/default/world_model/{wm_sage_IC.pt, sage_IC.json}
```

`train_world_model(TrainConfig(...))` is the importable form — it is what `pipeline/run.py` calls for the train stage, and it returns the same results dict it writes to disk.

---

## Evaluation — `wm_eval.py`

Three families of metrics, all written into the results JSON. Full definitions and worked SAGE numbers are in [`checkpoints/RESULTS.md`](checkpoints/RESULTS.md).

### 1. One-step (`evaluate_one_step`) — block `test`

Teacher-forced, threshold 0.5; soft targets are thresholded at 0.5 for the binary metrics, kept raw for Brier.

| metric                              | meaning                                                                                                                              |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `infected_acc` / `frontier_acc`     | per-node accuracy (dominated by unchanged nodes — high is easy)                                                                      |
| `new_infection_f1`                  | F1 on **newly infected** nodes (restricted to nodes susceptible at `t`)                                                              |
| `delta_f1`                          | F1 on nodes whose state **changed** `t → t+1` — the early-stop / headline metric                                                     |
| `add_seed_success`                  | fraction of `add_node` targets the model predicts as infected (should be 1.0)                                                        |
| `remove_frontier_success`           | fraction of `remove_node` targets predicted as not-frontier. Under `--remove-semantics blocked` a structured head zeroes those nodes in `T_exo`, so this reads ~1.0 by construction and stops being informative |
| `action_sensitivity`                | mean # of distinct outputs across counterfactual actions at the same state (>0 ⇒ the model reacts to the action, not just the state) |
| `brier_infected` / `brier_frontier` | MSE of predicted prob vs the soft marginal — calibration, lower better                                                               |
| `persistence`                       | the "predict next = current" baseline; its `delta_f1`/`new_infection_f1` are 0 by construction                                       |

### 2. Free-running rollout (`rollout_ensemble`) — block `rollout`

Treats the model as a **stochastic simulator**: for each test episode it rolls `n_samples` trajectories, at each step **sampling** the next state from the predicted marginals (not thresholding) and re-applying the recorded action, then compares the model's marginal/count distribution to the true NDlib simulator's MC trajectory under the same actions.

| metric                                           | meaning                                                                                        |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| `ens_marg_mae`                                   | mean \|model marginal − true marginal\| over nodes and steps (lower better; IC oracle ≈ 0.091) |
| `ens_count_w1`                                   | mean per-step Wasserstein-1 between model and true infected-count distributions                |
| `ens_count_bias`                                 | mean per-step `E[model count] − E[true count]`; ≈0 unbiased, ≫0 = saturation                   |
| `ens_final_count_model` / `ens_final_count_true` | mean final infected counts                                                                     |

Most meaningful for **IC** (genuinely stochastic). For LT the true re-run draws fresh hidden thresholds, so the comparison is indicative rather than exact.

### 3. Planning regret (`planning_regret_multi`) — block `planning`

Uses the world model to _choose_ an intervention. At sampled states it scores candidate `add_node` actions by predicted one-step spread, picks the argmax, and measures **regret** = `oracle_spread − true_spread(chosen)` (true spread via MC on the real simulator). Averaged over `--plan-graphs` graphs (each with its own seed offset) for resolution + a cross-graph std.

| metric               | meaning                                                             |
| -------------------- | ------------------------------------------------------------------- |
| `plan_regret_model`  | regret of the model's choice (lower better; 0 = always optimal)     |
| `plan_regret_degree` | regret of picking the highest-degree candidate (heuristic baseline) |
| `plan_regret_random` | regret of a random candidate (floor)                                |
| `*_std`              | cross-graph standard deviation (error bar)                          |

A useful planner must beat `random` decisively and at least match `degree`.

### Standalone re-eval CLIs (no retraining)

These rebuild a model from a results JSON's `config`, reload its `.pt` checkpoint, recompute one block, and write it back — for when only the metric changed, not the weights:

| script                      | recomputes                                                                 |
| --------------------------- | -------------------------------------------------------------------------- |
| `eval_planning.py`          | multi-graph planning regret                                                |
| `eval_rollout_ensemble.py`  | stochastic ensemble rollout                                                |
| `eval_structured_oracle.py` | the IC oracle rollout (q = true edge prob) — validates the structural form |

---

## Files

| File                        | Responsibility                                                                                                                                   |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `train_wm.py`               | training loop, early stopping, results JSON                                                                                                      |
| `wm_model.py`               | `WorldModel`, backbone registry, `ICTransmissionHead` / `LTThresholdHead` / linear head                                                          |
| `wm_data.py`                | feature builder (`X`, channels), `GraphInput`, per-episode adjacency, dataset + collate                                                          |
| `wm_metrics.py`             | F1 / accuracy / Brier / persistence primitives                                                                                                   |
| `wm_eval.py`                | one-step eval, ensemble rollout, planning regret, simulator rebuild                                                                              |
| `eval_planning.py`          | recompute planning regret on a checkpoint                                                                                                        |
| `eval_rollout_ensemble.py`  | recompute ensemble rollout on a checkpoint                                                                                                       |
| `eval_structured_oracle.py` | IC structural-form oracle check                                                                                                                  |
| `model/*.py`                | the five backbone encoders + `model_utils.py` (each file also retains an unused legacy `*ForwardModel` class from the old seed→outcome pipeline) |
| `checkpoints/`              | trained `.pt` weights, per-run results JSONs, and `RESULTS.md`                                                                                   |
