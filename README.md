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
> are [`data/README.md`](data/README.md) (data generation),
> [`world_model/README.md`](world_model/README.md) (features, models, training,
> evaluation), and [`coding_agent/README.md`](coding_agent/README.md) (the outer
> loop). Worked results are in
> [`world_model/checkpoints/RESULTS.md`](world_model/checkpoints/RESULTS.md).
>
> **Prior work.** [`IM_RESEARCH.md`](IM_RESEARCH.md) is the unified Influence
> Maximization literature review: every classical and learning-based method,
> which published results are directly comparable to ours (and which are not,
> because the graph versions differ), the full DeepIM/MOEIM/IRIE result tables,
> and links to every paper and code repo.

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
- **counterfactual forks** re-apply _different_ actions from the same `s_t` (same
  state, different action) — the signal that forces action-conditioning.
- **Monte-Carlo marginals**: each step is re-run `--mc-marginals` times (default 30) to estimate the _true_ one-step probability `P(infected)` / `P(frontier)`
  per node — these soft marginals are the training targets.

Full details, the JSONL schema, and every flag: [`data/README.md`](data/README.md).

### 2 — Feature construction (`world_model/wm_data.py`)

Each transition becomes a per-node feature matrix and a graph view.

**`X` — shape `(N, 6)`**, one row per node, six channels describing time `t`
before diffusion:

| col | channel      | represents                                              | type   |
| --- | ------------ | ------------------------------------------------------- | ------ |
| 0   | `infected`   | node ever-activated at `t` (state)                      | binary |
| 1   | `frontier`   | node in the current spreading wave at `t` (state)       | binary |
| 2   | `degree`     | `log1p(total degree)` in the time-`t` graph (structure) | float  |
| 3   | `act_add`    | target of an `add_node` op this step (action)           | binary |
| 4   | `act_remove` | target of a `remove_node` op this step (action)         | binary |
| 5   | `act_edge`   | endpoint of an edge op this step (action)               | binary |

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

| backbone          | `--model` | mechanism                                                      | graph view                   |
| ----------------- | --------- | -------------------------------------------------------------- | ---------------------------- |
| GCN               | `gcn`     | `Â X W`, pre-norm residual                                     | `adj_norm`                   |
| GraphSAGE         | `sage`    | `concat(self, weighted-mean(neighbors))`                       | `edge_index` + `edge_weight` |
| GATv2             | `gat`     | dynamic attention `aᵀLeakyReLU(W_l h_v + W_r h_u)`, multi-head | `edge_index` + `edge_weight` |
| Graph Transformer | `gt`      | scaled dot-product attention + FFN, pre-norm                   | `edge_index` + `edge_weight` |
| GCNII             | `gcnii`   | initial residual + identity mapping (deep)                     | `adj_norm`                   |

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

| op                | record fields                       | IC effect                           | LT effect                              |
| ----------------- | ----------------------------------- | ----------------------------------- | -------------------------------------- |
| `add_node`        | `target=v`                          | activate `v` (spreader next step)   | activate `v`                           |
| `remove_node`     | `target=v`                          | mark Removed/spent (stays counted)  | back to Susceptible (can re-activate)  |
| `add_edge`        | `target=u, destination=v, weight=w` | add arc `u→v` with transmission `w` | add edge structurally (weight ignored) |
| `remove_edge`     | `target=u, destination=v`           | remove arc `u→v`                    | remove edge structurally               |
| `set_edge_weight` | `target=u, destination=v, weight=w` | set arc `u→v` transmission `w`      | no-op (LT ignores edge weights)        |

Node ops set the `act_add` / `act_remove` input channels; edge ops set the
`act_edge` channel **and** mutate the per-episode adjacency. The three data
settings are simply which ops you enable via `--action-ops` (omit = diffusion-only;
`add_node remove_node` = node; `add_edge remove_edge set_edge_weight` = edge).

---

## Datasets

**Synthetic** (`--dataset`): `er` (Erdős–Rényi), `ba` (Barabási–Albert), `ws`
(Watts–Strogatz), `sbm` (stochastic block model — planted communities via
`--sbm-blocks/--sbm-p-in/--sbm-p-out`), `karate`. Generated in bulk via
`--num-graphs` with `log1p(degree)` node features.

**Real** (downloaded on first use via `data/datasets/`; Weibo needs a manual
AMiner download — see `data/datasets/weibo.py`):

| `--dataset`       | Nodes     | Edges     | Type                              | Node features                       |
| ----------------- | --------- | --------- | --------------------------------- | ----------------------------------- |
| `jazz`            | 198       | 2,742     | Undirected (collaborations)       | log(1 + degree)                     |
| `email_eu_core`   | 1,005     | 24,929    | Directed (emails)                 | log(1 + total degree) · **42 department labels** |
| `netscience`      | 1,589     | 2,742     | Undirected (coauthorship)         | log(1 + degree)                     |
| `cora_ml`         | 2,810     | 7,981     | Undirected (citations, standardized) | 2,879-dim bag-of-words, 7 labels    |
| `facebook`        | 4,039     | 88,234    | Undirected (friendships)          | log(1 + degree)                     |
| `power_grid`      | 4,941     | 6,594     | Undirected (power lines)          | log(1 + degree)                     |
| `ca_grqc`         | 5,242     | 14,484    | Undirected (coauthorship)         | log(1 + degree)                     |
| `wiki_vote`       | 7,115     | 103,689   | Directed (adminship votes)        | log(1 + total degree)               |
| `lastfm_asia`     | 7,624     | 27,806    | Undirected (mutual follows)       | log(1 + degree) · **18 country labels** |
| `nethept`         | 15,229    | 62,752    | Directed (coauthorship, both arcs)| log(1 + total degree)               |
| `netphy`          | 37,154    | 174,161   | Undirected (coauthorship)         | log(1 + degree)                     |
| `twitter`         | 81,306    | ≈1.3M     | Undirected (follows, symmetrized) | log(1 + degree)                     |
| `digg`            | 116,893   | ≈2.6M     | Undirected (friendships)          | log(1 + degree)                     |
| `youtube`         | 1,134,890 | 2,987,624 | Undirected (friendships)          | log(1 + degree)                     |
| `weibo`           | 1,787,443 | ≈216M     | Directed (influence u→v)          | log(1 + total degree)               |

Undirected rows quote undirected edges; directed rows quote arcs. `cora_ml` is
loaded through graph2gauss's `standardize()` (symmetrize → drop self-loops →
largest connected component), so it is byte-for-byte the graph DeepIM and MOEIM
report; the raw 2,995-node file is not comparable to any published table.

The four large graphs (Twitter, Digg, YouTube, Weibo) load fine but exceed what
the current NDlib rollout + selector pipeline can simulate in reasonable time —
they are targets for a future scalable-simulation pass, not day-one datasets.

**[`IM_DATASETS.md`](IM_DATASETS.md)** is the full catalogue: every graph in the
IM literature with source URLs and exact counts, which paper uses which, the
seven dataset names that denote more than one graph, what to add next, and the
loader contract for adding one.

---

## Quick start — the whole experiment in one command

`pipeline/run.py` runs every stage end to end and writes a single
`results/<tag>/report.md` with the tables and figures. Swapping datasets is a
one-flag change.

```bash
source .venv/bin/activate

# Everything: generate -> train the WM -> run all six baseline conditions at
# 1/5/10/20% budgets -> plot -> report
python -m pipeline.run --dataset ba --tag ba40 --num-graphs 40 --syn-nodes 100 \
    --compare

# Same thing on a real graph — only the dataset changes
python -m pipeline.run --dataset netscience --compare

# Outer-loop development without training a world model: drop the one arm that
# needs it and the train stage is skipped automatically
python -m pipeline.run --dataset sbm --tag sbm40 --num-graphs 40 --compare \
    --arms routing one_shot_free@native one_shot_free@monte_carlo one_shot_free@oracle
```

### The six baseline conditions

Two orthogonal axes — who designs the algorithm, and what feedback the designer
gets while designing. Each condition is one **arm**; every arm carries its own
evaluator, so they all land in one `report.md` table.

| # | condition | arm spec | designer | inner-loop feedback |
| --- | --- | --- | --- | --- |
| 1 | Pure GA | `--baselines celf_pp …` | fixed algorithm | none |
| 2 | GA routing | `routing` | LLM picks from the pool | none (one selection call) |
| 3 | Native coding agent | `one_shot_free@native` | LLM writes code | real executions only |
| 4 | Agent + MC simulation | `one_shot_free@monte_carlo` | LLM writes code | averaged simulator rollouts |
| 5 | Agent + oracle dynamics | `one_shot_free@oracle` | LLM writes code | true transition dynamics |
| 6 | **Ours: agent + learned GWM** | `one_shot_free@world_model` | LLM writes code | learned `f_θ` rollouts |

Conditions 3–6 hold the method fixed, so the **only** thing varying down that
ladder is the inner-loop evaluator — which is what makes it a clean ablation.
`@native` means the real simulator at `--native-mc-runs` episode(s) per candidate:
model-free trial and error that pays real experience for every noisy number it
gets back. Swap `one_shot_free` for `evolve_scored` (or any
`<method>_<mode>`) to run a different synthesis method down the same ladder.

**`--compare` is effectively mandatory for a multi-condition sweep.** Each arm's
own `reward` is measured by its own evaluator — a native arm's is one noisy
episode, ours is a model estimate — so those numbers cannot be compared to each
other. `--compare` replays every winning strategy on the same ground-truth Monte
Carlo referee, and that replay is the number the tables and plots use. Without
it the pipeline warns and the report is marked as not comparable.

```bash
# Add a classical baseline, drop an expensive one
python -m pipeline.run --dataset sbm --baselines celf_pp imm community_im \
    --arms one_shot_free@oracle one_shot_free@world_model --compare

# Give the native agent a bigger real-episode budget per candidate
python -m pipeline.run --dataset ba --native-mc-runs 5 --compare
```

### Progress and logging

Every stage announces itself and reports progress:

```
========================================================================
[pipeline] STAGE 3/5: AGENT   (elapsed 142s)
========================================================================
agent runs:  47%|████▋     | 7/15 [02:11<02:30, 18.8s/run, pct10/external_moeim]
[agent] pct10/baseline_celf_pp: done in 4.1s -> spread 24.50 (40.8% of N)
[agent] 15 results (10 reused, 5 new), 0 skipped
[pipeline] stage agent done in 25.7s
```

`data` shows an episode bar, `train` shows an epoch bar with live
`val_delta_f1` plus a nested batch bar, `agent` shows one bar over the whole
(budget × arm) grid with ETA and the arm currently running, `plots` names each
figure as it lands. `pipeline.json` marks each stage `running` → `done` (or
`failed` with the exception), so an interrupted run is distinguishable from a
clean one.

### Stages, resuming, and skipping

`data → train → agent → plots → report`. Every stage writes its artifacts before
the next begins, and a rerun **detects finished work on disk and skips it** — so a
killed job resumes at the exact arm and budget it died on. LLM arms are the
expensive part, and they are checkpointed one file per `(budget, arm)`.

```bash
# Only the reporting half of a finished run
python -m pipeline.run --dataset ba --tag ba40 --start-stage plots

# Re-run one stage from scratch
python -m pipeline.run --dataset ba --tag ba40 \
    --start-stage agent --end-stage agent --force

# Reuse an existing world model, skip straight to the agent sweep
python -m pipeline.run --dataset ba --tag ba40 --skip-stages data train
```

| flag | default | meaning |
| ---- | ------- | ------- |
| `--dataset` | — | synthetic family (`er ba ws sbm karate`) or real dataset name |
| `--tag` | the dataset name | `results/<tag>/` directory to write |
| `--start-stage` / `--end-stage` | `data` / `report` | inclusive stage range |
| `--skip-stages` | none | stages to omit from that range |
| `--force` | off | recompute stages whose outputs already exist |
| `--baselines` | 6 classical | condition 1: which algorithms from the pool to run |
| `--arms` | conditions 2–6 | `routing`, `<method>_<mode>[@<evaluator>]`, or extra `baseline:<algorithm>` |
| `--evaluator` | `oracle` | fallback for arms that do not name one with `@` |
| `--native-mc-runs` | `1` | real episodes per candidate for an `@native` arm |
| `--budget-pcts` | `1 5 10 20` | budget sweep as % of nodes; `--budgets` for absolute k |
| `--compare` | off | ground-truth referee replay — required for a valid cross-condition table |
| `--llm-model` / `--outer-iters` | `gpt-5.6-terra` / `5` | coding-agent model and refinement budget |

Generation, training, and agent hyperparameters are all exposed too
(`--rollouts`, `--mc-marginals`, `--wm-model`, `--head`, `--epochs`, `--n-samples`,
…) — see `python -m pipeline.run --help`.

### On SLURM

`sbatch/pipeline.sbatch` is the only job script. Run it directly and it queues
**itself**, with every one of the 67 pipeline flags reachable as an environment
variable:

```bash
DATASET=ba TAG=ba40 ./sbatch/pipeline.sbatch                    # everything
DATASET=sbm ARMS=one_shot_free@oracle BASELINES=none ./sbatch/pipeline.sbatch
DATASET=ba TAG=ba40 START_STAGE=plots ./sbatch/pipeline.sbatch  # re-plot only
DATASET=ba TAG=ba40 DRY_RUN=1 ./sbatch/pipeline.sbatch          # show, submit nothing
```

**GPU is requested only when the run needs one.** Before submitting, the script
asks `pipeline.run --print-resources` whether these flags ever put a tensor on a
device: training `f_θ`, any `@world_model` arm, or a learned external repo get
`--gres=gpu:1`; data generation, plots, the classical pool, and the
`routing`/`native`/`monte_carlo`/`oracle` arms queue CPU-only and start sooner.
`DEVICE` follows the same decision. Override with `GRES=gpu:2` or `GRES=none`.

Its header carries copy-pasteable templates for every scenario — full run,
oracle-only, evaluator ablation, baselines-only, specific baselines, no
baselines, single stages, plots-only, every LLM knob, every world-model knob,
every generation knob, LT, absolute budgets, and resume.

### Where everything lands

```
results/<tag>/
├── data/                        transitions + graph store          (stage: data)
├── world_model/                 wm_<model>_<dm>.pt, <model>_<dm>.json,
│                                history_<model>_<dm>.json          (stage: train)
├── agent/<budget>/<arm>.json    our conditions 1-6                 (stage: agent)
├── baselines/<budget>/<name>.json   external published baselines   (stage: agent)
│   └── _runs/<name>/<budget>/   raw stdout/stderr per external run
├── plots/*.png                  paper figures                      (stage: plots)
├── summary.csv / summary.json   ONE FLAT ROW PER (arm, budget)     (stage: report)
├── report.md                    tables + figures + winning program  (stage: report)
├── pipeline.json                config + per-stage status/timings
└── environment.json             git commit, host, python/torch/CUDA versions
```

**Nothing is recomputed and nothing is lost.** `summary.csv` is rewritten after
*every single arm*, so a killed sweep still leaves a readable table of everything
finished. The training curve flushes every 5 epochs, so a SLURM timeout at epoch
380/400 keeps the history. Per-node `final_marginals` (which cost `n_samples`
rollouts to produce) are serialized rather than recomputed. `environment.json`
records the git commit and library versions so a results tree stays
self-describing after the working copy moves on.

Raw dataset downloads live outside the results tree, in `data/raw/<dataset>/`,
since they are inputs shared across every run.

### Figures produced

`budget_vs_spread` (the headline: spread vs k per arm, with MC error bars),
`budget_vs_spread_pct` (normalized), `condition_comparison` (the baseline table
as a grouped bar chart, coloured by condition), `sample_efficiency` (spread vs
real-environment episodes burned — the axis the whole taxonomy hangs on),
`evaluator_fidelity` (model estimate vs Monte Carlo parity), `runtime`
(per-rollout cost by evaluator), `convergence` (best-so-far reward per outer
iteration), `cascade` (infected count over time), and — when a world model was
trained — `wm_training`, `wm_one_step`, `wm_rollout`. Each is skipped silently
when its inputs are absent, so a partial run just yields fewer figures.

---

## Running the stages individually

```bash
source .venv/bin/activate

# 1. Generate data — 20 BA graphs, IC + LT, all five action ops, four cheap
#    seed selectors, MC-marginal targets
python -m data.generate_wm_data --dataset ba --num-graphs 20 \
    --action-ops add_node remove_node add_edge remove_edge set_edge_weight \
    --models IC LT --algorithms random degree pagerank betweenness \
    --out-dir results/ba40/data

# 2. Train the world model — GraphSAGE, structured IC head
python -m world_model.train_wm \
    --data-dir results/ba40/data --diffusion-model IC \
    --model sage --head structured --pos-weight off \
    --hidden-dim 64 --n-layers 3 --epochs 400 --batch-size 32 --patience 50 \
    --seed 42 --device cuda --plan-demo
# checkpoint + results JSON default to results/ba40/world_model/

# 3. (optional) re-evaluate a checkpoint without retraining
python -m world_model.eval_rollout_ensemble \
    --results results/ba40/world_model/sage_IC.json --device cpu
python -m world_model.eval_structured_oracle \
    --data-dir results/ba40/data --diffusion-model IC --device cpu
```

### Key training flags

| flag                              | default       | meaning                                                              |
| --------------------------------- | ------------- | -------------------------------------------------------------------- |
| `--data-dir`                      | —             | generated dataset directory                                          |
| `--diffusion-model`               | `IC`          | `IC` or `LT` (selects the structured head too)                       |
| `--model`                         | `gcn`         | backbone: `gcn`, `sage`, `gt`, `gat`, `gcnii`                        |
| `--head`                          | `linear`      | `linear` or `structured` (use `structured` for a faithful simulator) |
| `--pos-weight`                    | `auto`        | class-imbalance up-weighting; use `off` with the structured head     |
| `--hidden-dim` / `--n-layers`     | `64` / `3`    | encoder width / depth                                                |
| `--n-heads` / `--ffn-dim`         | `4` / `128`   | attention backbones (GAT/GT)                                         |
| `--gcnii-alpha` / `--gcnii-lamda` | `0.1` / `0.5` | GCNII initial-residual / decay                                       |
| `--epochs` / `--patience`         | `200` / `30`  | training length / early-stop on val `delta_f1`                       |
| `--plan-demo` / `--plan-graphs`   | off / `5`     | run multi-graph planning-regret eval                                 |

---

## Current results (BA-100, structured SAGE, seed 42)

|                                           | IC                   | LT                                    |
| ----------------------------------------- | -------------------- | ------------------------------------- |
| one-step `delta_f1`                       | **0.829**            | **0.539** (partial-observability cap) |
| `brier_infected` (calibration)            | **0.0012**           | **0.0320**                            |
| rollout `count_bias` (≈0 = no saturation) | **−0.58**            | **−1.43**                             |
| final count model / true                  | **36.6 / 36.7**      | **47.3 / 46.2**                       |
| planning regret model / degree / random   | 0.244 / 0.269 / 3.11 | 0.196 / 0.271 / 3.37                  |

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
├── pipeline/
│   ├── run.py                  # ← END-TO-END DRIVER: stages, resume, CLI
│   ├── conditions.py           # the six-condition taxonomy + arm-spec grammar
│   ├── layout.py               # the results/<tag>/ directory contract
│   ├── plots.py                # every paper figure
│   └── report.py               # results/<tag>/report.md generator
├── data/
│   ├── generate_wm_data.py     # action-conditioned WM data generator (orchestrator)
│   ├── wm_simulator.py         # NDlib stepwise IC/LT sim + State/ActionOp + MC marginals
│   ├── wm_graphs.py            # graph providers (real + synthetic) → GraphBundle
│   ├── wm_actions.py           # spine seed selectors + injection + counterfactuals
│   ├── graph_utils.py          # adjacency → edge_index + IC/LT edge probs
│   ├── validate_wm_data.py     # post-hoc gate-check harness
│   ├── datasets/               # per-dataset download/load helpers
│   ├── raw/                    # raw downloads (gitignored, shared across runs)
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
│   ├── checkpoints/            # historical RESULTS.md (new runs write to results/)
│   └── README.md               # ← world-model technical reference
├── coding_agent/               # ← outer-loop coding agent (see its README)
├── sbatch/pipeline.sbatch      # the single SLURM entry point (self-submitting)
├── results/                    # ALL generated artifacts, one subtree per --tag
├── baselines/                  # published-baseline runners (registry + adapters);
│                               # external repos fetched into baselines/external/
└── requirements.txt
```

> **Legacy.** The original seed→outcome inverse-problem world-model training and
> the VAE joint-training pipeline have been removed (`world_model/old/`,
> `world_model/model/vae.py`). The diffusion-only CND/IM/SL data generators remain
> archived under `data/old/` for reference. The encoder files in
> `world_model/model/` still contain the old `*ForwardModel` classes, now unused.
> The action-conditioned world model above is the only active path.
