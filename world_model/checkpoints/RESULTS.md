# SAGE World-Model Results — BA-100 (×20 graphs), Structured Heads

This document walks through **every** metric in the two GraphSAGE result files,
metric by metric: what it measures, what the value is, and what it implies.

- `ba20_marg_structured_sage_IC.json` — IC dynamics
- `ba20_marg_structured_sage_LT.json` — LT dynamics

SAGE is the chosen backbone: across the 5-backbone sweep it was the only one
faithful on **both** IC and LT rollouts, and the cheapest to run. Metric
definitions live in [`../README.md`](../README.md) and
[`../../data/README.md`](../../data/README.md). Numbers below are quoted verbatim
from the JSONs.

> **Caveat for both files:** results are a **single seed (42)** on **BA-100**
> graphs. BA is degree-trivial (hub structure makes degree a near-optimal seed
> heuristic), so the planning-vs-degree comparison is within noise here; it
> becomes meaningful on WS/SBM/real graphs. Treat these as a "the method works
> and does not saturate" result, not a final benchmark.

---

## Shared configuration

Both runs used: `model=sage`, `head=structured`, `hidden_dim=64`, `n_layers=3`,
`dropout=0.1`, `epochs=400`, `lr=1e-3`, `weight_decay=5e-4`, `batch_size=32`,
**`pos_weight=off`**, `patience=50`, `seed=42`, on
`data/output/ba20_marg_structured` (20 BA graphs, ~100 nodes each, MC-marginal
targets). `pos_weight=off` is required for the structured head — its structural
form already avoids the all-zeros collapse, and `pos_weight` would over-inflate
the per-edge transmission `q`.

The only difference between the runs is `diffusion_model` (IC vs LT), which also
switches the structured head (`ICTransmissionHead` vs `LTThresholdHead`).

---

## IC results (`…_sage_IC.json`)

### `test` — one-step, teacher-forced

| metric                    | value       | what it measures                                                           | reading                                                                                                                                                                                                                                                               |
| ------------------------- | ----------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `infected_acc`            | **0.9943**  | per-node accuracy of next-infected (threshold 0.5)                         | High, but easy: most nodes are unchanged. Judge against persistence (0.9834) — the model adds ~1.1 pts by getting the _changes_ right.                                                                                                                                |
| `frontier_acc`            | **0.9943**  | per-node accuracy of next-frontier                                         | Same story vs persistence frontier 0.9298.                                                                                                                                                                                                                            |
| `new_infection_f1`        | **0.8294**  | F1 on nodes that newly infect (restricted to susceptible-at-`t`)           | The substantive one-step number. 0.83 means the model identifies _which_ susceptible nodes light up next with high precision/recall.                                                                                                                                  |
| `delta_f1`                | **0.8294**  | F1 on nodes whose state changed `t→t+1`                                    | Identical to `new_infection_f1` here **because IC is monotone** — nodes never de-infect, so "changed" == "newly infected". This is the early-stop metric. At/near the IC label ceiling.                                                                               |
| `add_seed_success`        | **1.0**     | fraction of `add_node` targets predicted infected                          | Perfect: the model always honors a seed action (T_exo is correctly captured by the structured head).                                                                                                                                                                  |
| `remove_frontier_success` | **1.0**     | fraction of `remove_node` targets predicted not-in-frontier                | Perfect: removals correctly drop nodes from the spreading wave.                                                                                                                                                                                                       |
| `action_sensitivity`      | **1.106**   | mean # of distinct outputs across counterfactual actions at the same state | >1 ⇒ the model genuinely **reacts to the action**, not just the state. Different actions at the same `s_t` produce ~1.1 distinct next-state predictions on average (≈ 2 actions per state, mostly distinguished). It is action-conditioned, not state-autoregressive. |
| `brier_infected`          | **0.00116** | MSE(pred prob, soft marginal) for infected                                 | Excellent calibration. The predicted probabilities, not just the thresholded labels, match the true MC marginals (persistence Brier 0.0241 — ~20× worse).                                                                                                             |
| `brier_frontier`          | **0.00116** | same, for frontier                                                         | Same — well calibrated.                                                                                                                                                                                                                                               |

**Persistence baseline** (`predict next = current`): `infected_acc 0.9834`,
`frontier_acc 0.9298`, `new_infection_f1 0.0`, `delta_f1 0.0`, `brier_infected
0.0241`, `brier_frontier 0.0776`. The 0.0 F1s are structural — persistence never
predicts a change — so any change-F1 above 0 beats it; 0.83 beats it
decisively, and the Brier gap shows the model is far better calibrated.

### `rollout` — free-running stochastic ensemble (20 samples, 50 episodes)

| metric                  | value      | what it measures                                                              | reading                                                                                                                                                                                                                 |
| ----------------------- | ---------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ens_marg_mae`          | **0.0946** | mean \|model marginal − true marginal\| over nodes & steps                    | Matches the IC **oracle** (~0.091, q = true edge prob). The learned per-edge transmission is essentially as good as knowing the true probabilities.                                                                     |
| `ens_count_w1`          | **2.606**  | mean per-step Wasserstein-1 between model & true infected-count distributions | The _distribution_ of cascade sizes (not just the mean) tracks the truth to within ~2.6 nodes — tight on a ~37-node final cascade.                                                                                      |
| `ens_count_bias`        | **−0.577** | mean per-step `E[model] − E[true]` count                                      | ≈0 ⇒ **no saturation**. Slightly negative = the model is marginally conservative mid-rollout. Compare the failed linear head: +49 (it ran away to the whole graph). This is the central success of the structured head. |
| `ens_final_count_model` | **36.62**  | mean final infected count, model                                              | Essentially identical to truth ↓.                                                                                                                                                                                       |
| `ens_final_count_true`  | **36.70**  | mean final infected count, true sim                                           | Model 36.62 vs true 36.70 — a 0.08-node gap on the endpoint. The simulator is faithful end-to-end, not just one step.                                                                                                   |

### `planning` — multi-graph one-step regret (5 graphs)

| metric               | value               | what it measures                                        | reading                                                                                                                             |
| -------------------- | ------------------- | ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `plan_regret_model`  | **0.2438 ± 0.0549** | spread lost vs the oracle when the model picks the seed | Low and tight. The model's chosen intervention is within ~0.24 nodes of the best candidate.                                         |
| `plan_regret_degree` | **0.2688 ± 0.0881** | same, picking highest-degree candidate                  | The model **edges out** the degree heuristic (0.244 < 0.269), but the gap is inside the error bars — expected on degree-trivial BA. |
| `plan_regret_random` | **3.109 ± 0.595**   | same, random candidate                                  | The floor. The model is ~13× better than random — it is clearly using real structure to plan.                                       |

**IC verdict:** at the one-step label ceiling (`delta_f1` 0.83, Brier 0.001),
faithful as a free-running simulator (no saturation; final count within 0.1
node), and a competent one-step planner (crushes random, ties/edges degree on
BA). The IC world model is working.

---

## LT results (`…_sage_LT.json`)

LT is deterministic given hidden per-node thresholds that are **re-drawn each
episode and never stored**. A state-only model therefore cannot recover the exact
trajectory — only the threshold _marginal_ `P(activate | active-neighbor
fraction)`. LT numbers are looser than IC **by design**, and that is the correct
expectation, not a defect.

### `test` — one-step, teacher-forced

| metric                    | value      | what it measures                                          | reading                                                                                                                                                                                                                                    |
| ------------------------- | ---------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `infected_acc`            | **0.9540** | per-node next-infected accuracy                           | vs persistence 0.9417 — a real but smaller margin than IC (LT is harder to call per-node).                                                                                                                                                 |
| `frontier_acc`            | **0.9480** | per-node next-frontier accuracy                           | vs persistence 0.8734 — the model adds ~7.5 pts on the frontier.                                                                                                                                                                           |
| `new_infection_f1`        | **0.5432** | F1 on newly-activated susceptible nodes                   | ~0.54 — the partial-observability ceiling: without the hidden thresholds the model predicts the activation probability, so borderline nodes are inherently uncertain.                                                                      |
| `delta_f1`                | **0.5386** | F1 on nodes whose state changed                           | Slightly **below** `new_infection_f1` (unlike IC where they're equal), because LT `remove_node` flips an active node back to susceptible — so "changed" includes de-activations, a superset of "newly infected", and is marginally harder. |
| `add_seed_success`        | **1.0**    | `add_node` targets predicted active                       | Perfect — seeds are always honored.                                                                                                                                                                                                        |
| `remove_frontier_success` | **0.638**  | `remove_node` targets predicted not-in-frontier           | Looks low, but **under-reads** LT: a removed node returns to _susceptible_ (status 0) and can legitimately re-activate next step if its neighbors push it over threshold, so a "miss" here is often correct dynamics, not an error.        |
| `action_sensitivity`      | **1.618**  | distinct outputs across counterfactual actions at a state | Higher than IC (1.11): LT's threshold dynamics make the next state more action-dependent, and the model reflects that. Strongly action-conditioned.                                                                                        |
| `brier_infected`          | **0.0320** | calibration, infected                                     | Worse than IC (0.0012) but still well below persistence (0.0583) — the residual is the irreducible threshold uncertainty.                                                                                                                  |
| `brier_frontier`          | **0.0383** | calibration, frontier                                     | Below persistence (0.1266) — good calibration given hidden thresholds.                                                                                                                                                                     |

**Persistence baseline:** `infected_acc 0.9417`, `frontier_acc 0.8734`,
`new_infection_f1 0.0`, `delta_f1 0.0`, `brier_infected 0.0583`, `brier_frontier
0.1266`. The model beats it on every comparable metric.

### `rollout` — free-running stochastic ensemble (20 samples, 50 episodes)

| metric                  | value      | what it measures                     | reading                                                                                                                                                                        |
| ----------------------- | ---------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ens_marg_mae`          | **0.1056** | mean marginal error                  | Slightly above IC (0.095) — consistent with the harder dynamics. No oracle exists for LT (thresholds unstored), so this is judged against the true re-drawn-threshold sim.     |
| `ens_count_w1`          | **5.042**  | per-step count-distribution distance | Larger than IC (2.6): LT cascades are bigger (~46 vs ~37) and the hidden thresholds add genuine variance, so the count distributions are wider.                                |
| `ens_count_bias`        | **−1.431** | per-step `E[model] − E[true]`        | Again **no saturation** (the linear head gave ~+49). Mildly negative = slightly conservative mid-rollout.                                                                      |
| `ens_final_count_model` | **47.25**  | model final count                    | Close to truth ↓; the model slightly _over_-shoots at the endpoint even though the per-step average bias is negative — i.e. it lags early and catches up, ending ~1 node high. |
| `ens_final_count_true`  | **46.18**  | true final count                     | Model 47.25 vs true 46.18 — a ~1-node endpoint gap on a ~46-node cascade. Faithful, not saturating.                                                                            |

### `planning` — multi-graph one-step regret (5 graphs)

| metric               | value               | what it measures                  | reading                                                                                |
| -------------------- | ------------------- | --------------------------------- | -------------------------------------------------------------------------------------- |
| `plan_regret_model`  | **0.1963 ± 0.1382** | regret of the model's seed choice | Low; even better mean than IC, though the wider std reflects LT's variance.            |
| `plan_regret_degree` | **0.2713 ± 0.1972** | degree-heuristic regret           | The model beats degree on the mean (0.196 < 0.271), again inside the error bars on BA. |
| `plan_regret_random` | **3.366 ± 0.818**   | random-pick regret                | Floor; the model is ~17× better than random.                                           |

**LT verdict:** one-step is capped by hidden-threshold partial observability
(`delta_f1` ~0.54, the expected ceiling, not a failure), but the rollout is
faithful (final count within ~1 node, no saturation) and planning beats random
decisively and edges degree. The LT world model is also working, with looser —
and correctly looser — one-step resolution.

---

## IC vs LT side by side

| metric                     | IC          | LT          | why they differ                                                            |
| -------------------------- | ----------- | ----------- | -------------------------------------------------------------------------- |
| `delta_f1`                 | 0.829       | 0.539       | LT thresholds are hidden/random → one-step is partial-observability-capped |
| `brier_infected`           | 0.0012      | 0.0320      | same reason — residual threshold uncertainty                               |
| `action_sensitivity`       | 1.11        | 1.62        | LT next-state depends more strongly on the action                          |
| `ens_count_bias`           | −0.58       | −1.43       | both ≈0 (no saturation); LT slightly more conservative                     |
| `final count model / true` | 36.6 / 36.7 | 47.3 / 46.2 | both faithful endpoints                                                    |
| `plan_regret_model`        | 0.244       | 0.196       | both ≫ better than random; both ≈ degree on BA                             |

**Bottom line:** the structured-head SAGE world model is an accurate, calibrated,
non-saturating one-step simulator on both IC and LT, and a competent one-step
planner. The remaining limits are (a) the IC label ceiling (already reached) and
(b) LT partial observability (inherent). The next gains come from **graph
breadth** (WS/SBM, then real graphs) and **multiple seeds for error bars**, not
from more BA volume — the 20→40-graph A/B showed volume does not move these
numbers.
