# Task-family generalization

Scope of this document: the taxonomy, the compatibility matrix, the transfer-level
definitions, the three generalization axes, the protocol, and completed results.
Cells with no run behind them say `NOT_RUN`.

Generated artefacts:

```bash
python -m scripts.transfer_matrix          # prints the matrix, writes the manifest
                                           # -> results/task_transfer/manifest.json
python -m scripts.inspect_registry         # the five registries + counts
```

---

## 1. Taxonomy

Tasks are organised first by **world** — the state layout and dynamics a head
must have — and only then by task role. World family decides whether a
`state_dict` can load at all.

| family | layout (in/out) | dynamics | tasks |
| --- | --- | --- | --- |
| **W1** single cascade | 6 / 2 | IC, LT | influence_maximization, adaptive_online_im, source_localization, cascade_reconstruction, cascade_prediction, critical_node_detection |
| **W2** competitive cascade | 8 / 4 | IC, LT | influence_blocking |
| **W3** compartmental | 9 / 5 | SIR, SIS, SEIR | epidemic_control |

Roles inside W1:

| role | tasks | what is different |
| --- | --- | --- |
| **F1** seed / intervention control | influence_maximization, adaptive_online_im | decision timing only |
| **F2** inverse / recovery | source_localization, cascade_reconstruction | solves the inverse of the same forward world |
| **F3** passive forecasting | cascade_prediction | observational data, no interventions |
| **F4** single-cascade containment | critical_node_detection | `blocked` removal instead of `spent` seeding |

**Containment overlay** (non-exclusive, spans all three world families):
`critical_node_detection`, `influence_blocking`, `epidemic_control`. They share
`blocked` removal semantics and a minimize objective. This is an action-semantics
grouping and is **never** a checkpoint-compatibility claim.

### Port status

`registry/task_families.py` classifies all eight. This repository can currently
run three of them (`influence_maximization`, `adaptive_online_im`,
`critical_node_detection`); the W2/W3 heads and the F2/F3 task modules live on
the 8-task branch and have not been ported. `scripts/transfer_matrix` prints the
distinction, and the manifest carries `target_runnable_here` per row. A task is
classifiable without being runnable — but a classification is not a result.

---

## 2. Transfer levels

| level | means | world-model fine-tuning |
| --- | --- | --- |
| `EXACT_CHECKPOINT` | the frozen checkpoint loads and is used directly; nothing rebuilt | forbidden |
| `FORWARD_DYNAMICS` | the frozen forward model is reused as an oracle, but the target never exercises action-conditioning | forbidden |
| `MECHANISM_ONLY` | some mechanism carries over; the checkpoint cannot load unchanged. The experiment must state which parameter groups are reused and which are rebuilt | n/a — it is not a checkpoint transfer |
| `INCOMPATIBLE` | no shared process and no shared action semantics | n/a |

Classification is **derived**, never keyed on a task name. The inputs are
`dynamics`, `competitive`, `epidemic`, `remove_semantics`, `action_ops` and —
the load-bearing one — `gen_action_ops`, which is what the data generator
actually injects mid-cascade.

Two rules the derivation enforces that shape alone would miss:

* **Same layout is not sufficient.** IM and CND are both W1 6/2 on IC/LT, and are
  still `MECHANISM_ONLY`, because `T_exo` branches on `remove_semantics` inside
  the head. A `spent` checkpoint under `blocked` loads with no shape error and
  applies the wrong deterministic semantics silently.
* **Declaring an action op is not exercising it.** `source_localization` declares
  `add_node` (a recovered source set is replayed as a seed bag) but its generator
  injects nothing after t=0, so a transferred model is never asked an
  action-conditioned question there. `FORWARD_DYNAMICS`, not `EXACT_CHECKPOINT`.

---

## 3. Transfer matrix

Derived, 56 ordered pairs. The rows that matter:

| Source | Target | Level | Why |
| --- | --- | --- | --- |
| influence_maximization | adaptive_online_im | **EXACT_CHECKPOINT** | identical W1 world, IC/LT, `spent`; target's vocabulary covered |
| adaptive_online_im | influence_maximization | **EXACT_CHECKPOINT** | same world, different decision schedule |
| influence_maximization | source_localization | FORWARD_DYNAMICS | no mid-cascade interventions in target data |
| influence_maximization | cascade_reconstruction | FORWARD_DYNAMICS | frozen `T_endo` reused |
| influence_maximization | cascade_prediction | FORWARD_DYNAMICS | + observational shift |
| influence_maximization | critical_node_detection | MECHANISM_ONLY | IC/LT shared; `spent` vs `blocked` differs |
| critical_node_detection | influence_blocking | MECHANISM_ONLY | W1 6/2 vs W2 8/4 |
| critical_node_detection | epidemic_control | MECHANISM_ONLY | containment action semantics only; no shared dynamics |
| influence_maximization | epidemic_control | **INCOMPATIBLE** | no shared dynamics, not a containment task |

`EXACT_CHECKPOINT` appears for exactly one pair, in both directions.

---

## 4. Generalization axes

| axis | question | configs |
| --- | --- | --- |
| **G1** graph | does it generalize to unseen graphs? | `configs/generalization/graph_size.yaml`, `graph_topology.yaml`, `synthetic_to_real.yaml` |
| **G2** policy | to algorithms that did not generate its training trajectories? | `configs/generalization/policy_unseen.yaml` |
| **G3** task | can one frozen model be reused across tasks? | `configs/task_transfer/*.yaml` |

**G1.1 (train BA-100 / test held-out BA-100) is a control, not evidence.** Twenty
BA(100,3) graphs are near-exchangeable; `docs/comparison_split_protocol.md` shows
that removing a 20/20 graph leak barely moved the prediction metrics. Every real
G1 claim needs size, topology or real-data shift.

**Single real graphs are held-out transfer sets only.** They admit no
graph-disjoint split, so splitting one episode-randomly would produce an in-graph
number dressed as a test number. `synthetic_to_real.yaml` trains on synthetic
families and never lets a real graph into training.

---

## 5. Completed results

### 5.1 Stage A — correctness (DONE)

Both blockers fixed, verified, and covered by regression tests
(`tests/test_stage_a_correctness.py`, 19 tests).

**A1 planning-evaluator split leakage.** `planning_regret_multi` and
`planning_regret_budget_multi` scored `list(store)[:n]` — training graphs
included. They now select from the test split and emit provenance. Measured on
the corrected baseline:

```
planning_split=test   n_planning_graphs=3   train_overlap=0   val_overlap=0
graphs: ba_n100_m3_s43, ba_n100_m3_s50, ba_n100_m3_s55
```

**A2 LT `remove_node` frontier semantics.** `Simulator.advance` labels LT
frontier as `active - previous_active`, snapshotting *before* the action;
`LTThresholdHead` computed it against the *post*-action state. Wrong for both
node ops in opposite directions, confirmed against the generated dataset:

| | true next-frontier at target | head before fix |
| --- | --- | --- |
| LT `add_node` | 1.0 | 0 |
| LT `remove_node` | 0.0 | up to 0.92 |
| IC `add_node` | 0.0 | 0 (correct, unchanged) |

The head now keeps `active_pre` and `active_post` separate; the frontier channel
is `clamp(y_inf - active_pre, min=0)`. IC is untouched.

### 5.2 Stage A3 — corrected canonical baseline (DONE)

BA-100 × 20 graphs, graph-disjoint, SAGE + `structured`, seed 42, one A6000 each.
Run on `grandriver`, `hongji_env`.

| metric | IC | LT | LT before A2 |
| --- | --- | --- | --- |
| `delta_f1` | 0.8715 | 0.6716 | 0.6710 |
| `brier_infected` | 0.00130 | — | — |
| `add_seed_success` | 1.000 | 1.000 | 1.000 |
| **`remove_frontier_success`** | 1.000 | **1.0000** | **0.064** |
| **`removed_p_frontier_worst`** | 1.0e-6 | **1.0e-6** | **0.92** |
| `seeded_p_infected_worst` | 0.9999989 | 0.9999989 | 0.9999989 |
| `ens_marg_mae` | 0.0933 | 0.1424 | 0.1173 |
| `ens_count_bias` | −0.460 | **+12.17** | +7.53 |
| final count model / true | 47.30 / 47.81 | 85.94 / 64.71 | 79.82 / 64.71 |
| `effect_mae_norm` (null 1.0) | 0.5294 | **0.8316** | 0.886 |
| `effect_pearson` | 0.9140 | 0.7732 | 0.757 |
| `shuffle_delta_f1_drop` | 0.6556 | 0.2614 | 0.263 |
| `plan_regret_model` (held-out) | 0.2479 ± 0.083 | 0.2063 ± 0.062 | — |
| `plan_regret_degree` (held-out) | 0.2229 | 0.2104 | — |
| `budget_regret_norm` (held-out) | −0.165 ± 0.227 | −0.082 ± 0.193 | — |

Readings:

1. **A2 is verified on exact structural grounds.** `remove_frontier_success`
   0.064 → 1.000 and `removed_p_frontier_worst` 0.92 → 1.0e-6, the clamp floor.
   LT action-conditioning also improved (`effect_mae_norm` 0.886 → 0.832).
2. **A2 made the LT rollout bias worse, and that is probably it being revealed
   rather than caused.** One-step accuracy is unchanged (0.6710 → 0.6716), so
   nothing about the fit moved; only the frontier channel did, and the ensemble
   rollout consumes exactly that channel. The most likely reading is that the
   previously-wrong channel was cancelling part of a pre-existing over-prediction
   (model 86 vs true 65). **This is now the top open defect.** It does not block
   Stage B or D, which run on IC.
3. **On held-out graphs the model no longer beats degree on one-step planning
   regret** (IC 0.2479 vs degree 0.2229; it read 0.2437 vs 0.2687 under the leaky
   selection). Both differences are inside the error bars at n = 3.
4. **Three test graphs is too thin for the planning metrics.** 20 graphs at
   70/15/15 leaves 3, so `--plan-graphs 5` silently clipped. Stage B configs
   should raise `--num-graphs` before any planning claim is made.

### 5.3 Everything else

`NOT_RUN`. See `results/task_transfer/manifest.json`.

---

## 6. Protocol

Stage order is A → B → C → D → E → F → G. Stage A is complete. Stage D1
(IM → Adaptive IM) is the headline experiment and is next.

For every transfer run, record: git commit, checkpoint format and spec, source
and target task, transfer classification, train/test graph families and sizes,
train/test policies, split mode, graph ids, seed, world-model metrics, task
metrics, runtime, trusted-evaluator call count, and train-graph overlap (which
must be 0).

Report three layers separately and never collapse them into one score:

1. **world prediction** — did the transition model survive the shift?
2. **action/causal fidelity** — does it still respond to interventions correctly?
   `NOT_APPLICABLE` for a target with no interventions; never a fabricated zero.
3. **task/decision utility** — task-specific metrics, plus `transfer_gap`,
   `oracle_gap`, `decision_retention`.
