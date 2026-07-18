# Graph World Model

A learned, action-conditioned **simulator of graph diffusion dynamics**. Given a
graph `G`, a diffusion state `s_t`, and an intervention `a_t`, the model predicts
the next state:

```
f_θ(G, s_t, a_t) → s_{t+1}
```

It is trained on `(G, s_t, a_t, s_{t+1})` transitions harvested from a real
diffusion simulator, and at inference replaces that expensive simulator with a
fast, differentiable forward pass. The long-term goal (not yet built) is to use
the world model as the **predictive environment inside a coding-agent loop** for
graph-algorithm design — roll out candidate algorithms cheaply instead of
executing them.

The current focus is **Influence Maximization (IM)** dynamics under two diffusion
models — **Independent Cascade (IC)** and **Linear Threshold (LT)** — with node-
and edge-level interventions.

> **Two-layer docs.** This README is the overview. The deep technical references
> are [`data/README.md`](data/README.md) (data generation) and
> [`world_model/README.md`](world_model/README.md) (features, models, training,
> evaluation). Worked results are in
> [`world_model/checkpoints/RESULTS.md`](world_model/checkpoints/RESULTS.md).

---

## The pipeline, end to end

```
1. GENERATE          data/generate_wm_data.py
   real simulator (NDlib IC/LT) rolled out under spine seed selectors +
   action injection  ──▶  JSONL transitions + soft Monte-Carlo targets

2. BUILD FEATURES    world_model/wm_data.py
   each transition  ──▶  X:(N,6) node features + GraphInput + soft targets y

3. TRAIN             world_model/train_wm.py
   encoder backbone + dynamics-matched head, teacher-forced one-step,
   BCE(next_infected) + BCE(next_frontier)

4. EVALUATE          world_model/wm_eval.py
   one-step accuracy · free-running rollout fidelity · planning regret
```

### 1 — Data generation (`data/`)

A `GraphBundle` is built for each graph (structure + per-edge IC probabilities
`p(u→v) = 1/in_degree(v)` by default + node features). For each `(graph,
dynamics, seed-algorithm, rollout)` an **episode** is simulated step by step with
NDlib:

- **t = 0** commits a seed set chosen by one of six classical "spine" selectors
  (`random, degree, pagerank, betweenness, celf, local_search`).
- **t > 0** optionally injects an action (probability `--inject-p`), else `NULL`.
- each step is advanced and recorded as `(s_t, a_t, s_{t+1}, reward)`.
- **counterfactual forks** re-apply *different* actions from the same `s_t` (same
  state, different action) — the signal that forces action-conditioning.
- **Monte-Carlo marginals**: each step is re-run `--mc-marginals` times (default
  30) to estimate the *true* one-step probability `P(infected)` / `P(frontier)`
  per node — these soft marginals are the training targets.

Full details, the JSONL schema, and every flag: [`data/README.md`](data/README.md).

### 2 — Feature construction (`world_model/wm_data.py`)

Each transition becomes a per-node feature matrix and a graph view.

**`X` — shape `(N, 6)`**, one row per node, six channels describing time `t`
before diffusion:

| col | channel | represents | type |
| --- | ------- | ---------- | ---- |
| 0 | `infected`    | node ever-activated at `t` (state)              | binary |
| 1 | `frontier`    | node in the current spreading wave at `t` (state)| binary |
| 2 | `degree`      | `log1p(total degree)` in the time-`t` graph (structure) | float |
| 3 | `act_add`     | target of an `add_node` op this step (action)   | binary |
| 4 | `act_remove`  | target of a `remove_node` op this step (action) | binary |
| 5 | `act_edge`    | endpoint of an edge op this step (action)       | binary |

Channels 0–1 are the **state**, 2 is **structure**, 3–5 are the **action**
projected onto nodes. **Targets** `y_inf, y_fr` (each `(N,)`) are the soft MC
marginals `P(node infected/frontier at t+1)`.

**Graph structure** is passed as a `GraphInput`: a symmetrically renormalized
sparse adjacency `D^{-1/2}(A+I)D^{-1/2}` (used by GCN/GCNII) plus `edge_index` and
`edge_weight` (used by SAGE/GAT/GT). IC keeps the transmission probability as the
edge weight; LT uses all-ones. Edge actions are replayed per episode so each step
sees the adjacency that actually produced its `s_{t+1}`.

### 3 — Model (`world_model/wm_model.py`)

A **backbone encoder** maps `X (N,6) → h (N, hidden)`, then a **head** maps
`h → logits (N,2) = [next_infected, next_frontier]`.

Five plug-and-play backbones (`--model`):

| backbone | `--model` | mechanism | graph view |
| -------- | --------- | --------- | ---------- |
| GCN | `gcn` | `Â X W`, pre-norm residual | `adj_norm` |
| GraphSAGE | `sage` | `concat(self, weighted-mean(neighbors))` | `edge_index` + `edge_weight` |
| GATv2 | `gat` | dynamic attention `aᵀLeakyReLU(W_l h_v + W_r h_u)`, multi-head | `edge_index` + `edge_weight` |
| Graph Transformer | `gt` | scaled dot-product attention + FFN, pre-norm | `edge_index` + `edge_weight` |
| GCNII | `gcnii` | initial residual + identity mapping (deep) | `adj_norm` |

Three heads (`--head`):

- **`linear`** — `Linear(hidden, 2)`; flexible but saturates the whole graph on a
  free-running rollout (no structural cap).
- **`structured` (IC)** — `ICTransmissionHead`: predict a per-edge transmission
  `q(u→v)` and derive the IC infection form `p_new(v) = 1 − ∏(1 − q·frontier_u)`.
  Locality makes the cascade **self-terminating** — it structurally cannot
  saturate. Train with `--pos-weight off`.
- **`structured` (LT)** — `LTThresholdHead`: predict activation as a learned
  monotone function of the active-neighbor fraction `f_v`, gated by `f_v > 0`.

The structured heads are the fix that turned a runaway rollout (final count ~99
of 100) into a faithful one (final count within ~1 node of truth). See
[`world_model/README.md`](world_model/README.md) for the math.

### 4 — Training & evaluation (`world_model/train_wm.py`, `wm_eval.py`)

Teacher-forced one-step: `loss = BCEWithLogits(·, y_inf) + BCEWithLogits(·,
y_fr)`. Adam, early-stop on validation `delta_f1`. Evaluation reports three
families:

- **one-step** — `delta_f1` / `new_infection_f1`, calibration (`brier`),
  action-specific success, action-sensitivity, vs a persistence baseline;
- **free-running rollout** — sampled-ensemble marginal MAE, count Wasserstein-1,
  and **count bias** (≈0 = no saturation), vs the true MC trajectory;
- **planning regret** — use the model to pick a one-step intervention and measure
  spread lost vs the oracle, against degree and random baselines.

---

## The action space (5 ops, unified across tasks)

Every action is `(op, target, [destination], [weight])`. `target` is the node (or
edge source `u`); `destination` is the edge sink `v`; `weight` is the IC
transmission probability for the edge.

| op | record fields | IC effect | LT effect |
| -- | ------------- | --------- | --------- |
| `add_node`        | `target=v`                          | activate `v` (spreader next step)        | activate `v`                       |
| `remove_node`     | `target=v`                          | mark Removed/spent (stays counted)       | back to Susceptible (can re-activate) |
| `add_edge`        | `target=u, destination=v, weight=w` | add arc `u→v` with transmission `w`      | add edge structurally (weight ignored) |
| `remove_edge`     | `target=u, destination=v`           | remove arc `u→v`                         | remove edge structurally           |
| `set_edge_weight` | `target=u, destination=v, weight=w` | set arc `u→v` transmission `w`           | no-op (LT ignores edge weights)    |

Node ops set the `act_add` / `act_remove` input channels; edge ops set the
`act_edge` channel **and** mutate the per-episode adjacency. The three data
settings are simply which ops you enable via `--action-ops` (omit = diffusion-only;
`add_node remove_node` = node; `add_edge remove_edge set_edge_weight` = edge).

---

## Datasets

**Synthetic** (`--dataset`): `er` (Erdős–Rényi), `ba` (Barabási–Albert), `ws`
(Watts–Strogatz), `karate`. Generated in bulk via `--num-graphs` with
`log1p(degree)` node features.

**Real** (downloaded on first use via `data/datasets/`):

| Dataset        | Nodes  | Edges  | Type                        | Node Features          |
| -------------- | ------ | ------ | --------------------------- | ---------------------- |
| **Cora-ML**    | 2,995  | 8,416  | Directed (citations)        | 2,879-dim bag-of-words |
| **Jazz**       | 198    | 2,742  | Undirected (collaborations) | log(1 + degree)        |
| **NetScience** | 1,589  | 2,742  | Undirected (coauthorship)   | log(1 + degree)        |
| **Power Grid** | 4,941  | 6,594  | Undirected (power lines)    | log(1 + degree)        |
| **NetHEPT**    | 15,229 | 62,752 | Directed (citations)        | log(1 + total degree)  |

---

## Quick start

```bash
source .venv/bin/activate

# 1. Generate data — 20 BA graphs, IC + LT, all five action ops, four cheap
#    seed selectors, MC-marginal targets
python -m data.generate_wm_data --dataset ba --num-graphs 20 \
    --action-ops add_node remove_node add_edge remove_edge set_edge_weight \
    --models IC LT --algorithms random degree pagerank betweenness \
    --out-dir data/output/ba20_marg_structured

# 2. Train the world model — GraphSAGE, structured IC head
python -m world_model.train_wm \
    --data-dir data/output/ba20_marg_structured --diffusion-model IC \
    --model sage --head structured --pos-weight off \
    --hidden-dim 64 --n-layers 3 --epochs 400 --batch-size 32 --patience 50 \
    --seed 42 --device cuda --plan-demo \
    --results world_model/checkpoints/ba20_marg_structured_sage_IC.json

# 3. (optional) re-evaluate a checkpoint without retraining
python -m world_model.eval_rollout_ensemble \
    --results world_model/checkpoints/ba20_marg_structured_sage_IC.json --device cpu
python -m world_model.eval_structured_oracle \
    --data-dir data/output/ba20_marg_structured --diffusion-model IC --device cpu
```

### Key training flags

| flag | default | meaning |
| ---- | ------- | ------- |
| `--data-dir` | — | generated dataset directory |
| `--diffusion-model` | `IC` | `IC` or `LT` (selects the structured head too) |
| `--model` | `gcn` | backbone: `gcn`, `sage`, `gt`, `gat`, `gcnii` |
| `--head` | `linear` | `linear` or `structured` (use `structured` for a faithful simulator) |
| `--pos-weight` | `auto` | class-imbalance up-weighting; use `off` with the structured head |
| `--hidden-dim` / `--n-layers` | `64` / `3` | encoder width / depth |
| `--n-heads` / `--ffn-dim` | `4` / `128` | attention backbones (GAT/GT) |
| `--gcnii-alpha` / `--gcnii-lamda` | `0.1` / `0.5` | GCNII initial-residual / decay |
| `--epochs` / `--patience` | `200` / `30` | training length / early-stop on val `delta_f1` |
| `--plan-demo` / `--plan-graphs` | off / `5` | run multi-graph planning-regret eval |

---

## Current results (BA-100, structured SAGE, seed 42)

| | IC | LT |
| --- | --- | --- |
| one-step `delta_f1` | **0.829** | **0.539** (partial-observability cap) |
| `brier_infected` (calibration) | **0.0012** | **0.0320** |
| rollout `count_bias` (≈0 = no saturation) | **−0.58** | **−1.43** |
| final count model / true | **36.6 / 36.7** | **47.3 / 46.2** |
| planning regret model / degree / random | 0.244 / 0.269 / 3.11 | 0.196 / 0.271 / 3.37 |

The structured SAGE world model is an accurate, calibrated, **non-saturating**
one-step simulator on both dynamics and beats random planning decisively. Full
metric-by-metric analysis: [`world_model/checkpoints/RESULTS.md`](world_model/checkpoints/RESULTS.md).

---

## Coding Agent (planned)

The world model is designed to become the **predictive environment** in a
coding-agent loop for graph-algorithm evolution:

```
π        = A_φ(G, task, history)          # agent proposes a candidate algorithm
ŝ_next   = f_θ(s_t, a_t, task)            # GWM predicts the state transition (cheap)
rollout  = Rollout_GWM(π, G, task)  ──▶  Refine(π)   # predicted insights guide refinement
```

The agent operates over parameterized graph action primitives (candidate
expansion, node selection, score propagation, subgraph update, termination), so
algorithms are compositions of the same `(operator, arguments)` actions the world
model already simulates. Planned baselines: native coding agent (real execution
only), pure graph algorithms (greedy IM, BFS, PageRank), GA routing over a fixed
pool, and the full coding-agent + GWM. The non-saturating rollout demonstrated
above is the prerequisite this component was waiting on.

---

## Project structure

```
GraphWorldModel/
├── data/
│   ├── generate_wm_data.py     # action-conditioned WM data generator (orchestrator)
│   ├── wm_simulator.py         # NDlib stepwise IC/LT sim + State/ActionOp + MC marginals
│   ├── wm_graphs.py            # graph providers (real + synthetic) → GraphBundle
│   ├── wm_actions.py           # spine seed selectors + injection + counterfactuals
│   ├── graph_utils.py          # adjacency → edge_index + IC/LT edge probs
│   ├── validate_wm_data.py     # post-hoc gate-check harness
│   ├── datasets/               # per-dataset download/load helpers
│   ├── README.md               # ← data-generation technical reference
│   └── old/                    # legacy diffusion-only CND/IM/SL generators (archived)
├── world_model/
│   ├── train_wm.py             # teacher-forced one-step training + results JSON
│   ├── wm_model.py             # WorldModel + backbone registry + structured heads
│   ├── wm_data.py              # X feature builder + GraphInput + dataset + collate
│   ├── wm_metrics.py           # F1 / accuracy / Brier / persistence primitives
│   ├── wm_eval.py              # one-step eval + ensemble rollout + planning regret
│   ├── eval_planning.py        # recompute planning regret on a checkpoint
│   ├── eval_rollout_ensemble.py# recompute ensemble rollout on a checkpoint
│   ├── eval_structured_oracle.py# IC structural-form oracle check
│   ├── model/                  # 5 backbone encoders + model_utils (encoder files also retain unused legacy *ForwardModel classes)
│   ├── checkpoints/            # trained weights, results JSONs, RESULTS.md
│   └── README.md               # ← world-model technical reference
├── baselines/DeepIM/           # external DeepIM baseline (reference)
└── requirements.txt
```

> **Legacy.** The original seed→outcome inverse-problem world-model training and
> the VAE joint-training pipeline have been removed (`world_model/old/`,
> `world_model/model/vae.py`). The diffusion-only CND/IM/SL data generators remain
> archived under `data/old/` for reference. The encoder files in
> `world_model/model/` still contain the old `*ForwardModel` classes, now unused.
> The action-conditioned world model above is the only active path.
