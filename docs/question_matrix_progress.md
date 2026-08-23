# Task × Question progress matrix

Every experiment states which question it answers, what would have falsified it,
and what it does NOT support. Empty cells say `NOT_RUN`; nothing is estimated.

Status: `DONE` · `PARTIAL` · `NOT_RUN` · `NOT_APPLICABLE` · `BLOCKED`

Canonical setting: `influence_maximization`, BA-100 × **100 graphs**,
graph-disjoint (70/15/15 → **15 held-out test graphs**), SAGE, seed 42, run on
grandriver (RTX A6000, `hongji_env`).

---

## 1. The matrix

| Task | Q1 Action | Q2 Structure | Q3 Generalization | Q4 Decision utility | Q5 Transfer |
| --- | --- | --- | --- | --- | --- |
| **IM + IC** | `DONE` ID **and** OOD | `DONE` structured ≫ linear | `DONE` prediction transfers; action mechanism decays with size | `DONE` **negative** | → Adaptive `NOT_RUN` |
| **IM + LT** | `PARTIAL` weak (0.83 vs 1.0) | `DONE` + oracle diagnostic | `NOT_RUN` | `NOT_RUN` | later |
| **Adaptive IM** | `NOT_RUN` | shared with IM | `NOT_RUN` | `NOT_RUN` | ← IM `EXACT_CHECKPOINT`, `NOT_RUN` |
| **Source Localization** | `NOT_APPLICABLE` | forward dynamics | `NOT_RUN` | `NOT_RUN` | ← IM `FORWARD_DYNAMICS`, `BLOCKED` |
| **Cascade Reconstruction** | `NOT_APPLICABLE` | forward dynamics | `NOT_RUN` | `NOT_RUN` | ← IM `FORWARD_DYNAMICS`, `BLOCKED` |
| **Cascade Prediction** | `NOT_APPLICABLE` | forward dynamics | `NOT_RUN` real OOD | `NOT_RUN` | ← synthetic WM, `BLOCKED` |
| **CND** | `NOT_RUN` | `blocked` T_exo | `NOT_RUN` | `NOT_RUN` | ← IM `MECHANISM_ONLY`, `NOT_RUN` |
| **Influence Blocking** | `NOT_RUN` | competitive head | `NOT_RUN` | `NOT_RUN` | ← CND `MECHANISM_ONLY`, `BLOCKED` |
| **Epidemic Control** | `NOT_RUN` | compartment head | `NOT_RUN` | `NOT_RUN` | mechanism only, `BLOCKED` |

`BLOCKED` = the head or task module lives on the 8-task branch and is not ported.
Classification exists; execution does not.

---

## 2. Q2 — structured vs linear (`DONE`)

```
Experiment:          SAGE, BA-100 x 100, graph-disjoint, IC and LT,
                     head in {structured, linear}, linear at pos_weight in
                     {off, auto} because `off` was chosen FOR the structured head
Hypothesis:          the structured head preserves intervention semantics and
                     avoids runaway rollouts; it need NOT win one-step accuracy
```

| arm | ΔF1 | marg_mae | count_bias | model/true | seeded_worst | removed_frontier_worst |
| --- | --- | --- | --- | --- | --- | --- |
| **structured IC** | 0.8559 | **0.0913** | **−0.197** | 45.2 / 45.7 | **0.9999989** | **1.0e-6** |
| linear IC (auto) | 0.8526 | 0.4661 | **+49.28** | 96.7 / 45.7 | 0.9908 | 0.0078 |
| **structured LT** | 0.6736 | 0.1538 | +15.13 | 81.6 / 60.8 | **0.9999989** | **1.0e-6** |
| linear LT (auto) | 0.6754 | 0.3806 | +40.46 | 98.4 / 60.8 | 0.9930 | 0.4410 |
| linear LT (off) | 0.6669 | **0.0884** | **+0.33** | 61.8 / 60.8 | 0.9962 | 0.2661 |

```
Supports:            one-step accuracy is TIED (0.8559 vs 0.8526) -- the structure
                     buys nothing there, as predicted. It buys rollout stability:
                     the linear IC head saturates at +49.28, predicting 96.7 of
                     100 nodes infected against a true 45.7. 5.1x worse marg_mae.
                     And it buys exactness: structured sits at the clamp bounds
                     at ANY parameter value, linear is off by 0.008 to 0.44.
Trap this exposes:   the THRESHOLDED add_seed_success / remove_frontier_success
                     are 1.0000 for EVERY arm including linear. Only the
                     probability-level worst case separates them, which is what
                     the rollout consumes and why exogenous_fidelity reports it.
Does NOT support:    structured winning everywhere. On LT, linear+pos_weight-off
                     has the best rollout calibration of any arm (+0.33).
```

**The LT anomaly, chased down.** linear+off wins LT rollout calibration by
*violating monotonicity*: measured on already-active nodes, it predicts
`P(still infected) = 0.91` on average and violates on **100%** of them, i.e. it
drops ~9% of the infected set every step. LT never de-activates a node, so that
leak is a modelling error — and it happens to cancel LT's over-prediction. Two
errors cancelling, not calibration. The structured head violates on **0%** by
construction, under ×1/×20/×(−20) perturbation (regression test).

**LT bias is structural, not a fitting failure.** `--head structured_oracle` is
IC-only by design, but LT has an analytic oracle: thresholds are drawn U(0,1),
so `P(activate | f_v) = f_v` exactly. On the same data:

| LT predictor | marg_mae | count_bias |
| --- | --- | --- |
| analytic memoryless optimum (zero learning) | 0.2302 | +24.36 |
| **learned structured** | **0.1538** | **+15.13** |

The learned head beats the memoryless floor. LT is deterministic given θ and θ
*persists*; a memoryless predictor redraws `P(θ<f)` each step and ignores the
evidence in earlier non-activations, so it over-activates. Implied fix is state
augmentation (carry a posterior over θ), not more fitting.

---

## 3. Q3 — generalization (`DONE` for IC graph shift)

Frozen BA-100 IC checkpoint, no fine-tuning, evaluated on 6 target
distributions. All targets `split_mode=eval_only` (no train split exists) and
**structurally disjoint** from the source: every graph hashed by
`(num_nodes, canonical sorted edge list)`, zero collisions against ANY source
split. Density matched so topology shift is not confounded with density shift
(target mean degree 5.80–6.00 vs source 5.82).

| target | N | ΔF1 | marg_mae | count_bias | model/true | **effect_mae_norm** | pearson | shuffle | T_exo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BA-100 (ID control) | 100 | 0.8559 | 0.0913 | −0.20 | 45.2 / 45.7 | **0.5325** | 0.911 | 0.636 | exact |
| BA-200 | 200 | 0.8529 | 0.0921 | −0.60 | 91.8 / 92.3 | **0.6346** | 0.870 | 0.654 | exact |
| BA-500 | 500 | 0.8537 | 0.0950 | +1.19 | 238.4 / 236.1 | **0.7861** | 0.770 | 0.668 | exact |
| BA-1000 | 1000 | 0.8501 | 0.1039 | +2.86 | 493.2 / 489.1 | **0.8676** | 0.667 | 0.675 | exact |
| ER-100 | 100 | 0.8775 | 0.0849 | −1.92 | 35.7 / 38.1 | **0.5186** | 0.915 | 0.770 | exact |
| WS-100 | 100 | 0.9515 | 0.0681 | −0.91 | 29.8 / 30.9 | **0.4764** | 0.926 | 0.898 | exact |
| SBM-100 | 100 | 0.8871 | 0.0791 | −1.26 | 34.2 / 35.6 | **0.5045** | 0.919 | 0.781 | exact |

```
Supports:            PREDICTION transfers. delta_f1 0.850-0.952, marg_mae
                     0.068-0.104, count bias within +-2.9 nodes on a 490-node
                     cascade -- at 10x the training graph size and on three
                     unseen topologies, frozen.
                     T_exo holds EXACTLY on every target (0.9999989 / 1.0e-6),
                     which is expected: it is algebraic, not learned.
                     Action conditioning PASSES on all seven.

Key finding:         the action MECHANISM degrades with SIZE and not with
                     TOPOLOGY. effect_mae_norm 0.53 -> 0.63 -> 0.79 -> 0.87 as
                     N goes 100 -> 200 -> 500 -> 1000, approaching the 1.0
                     "predicts no action effect" null; pearson falls 0.91 ->
                     0.67 in step. Meanwhile all three unseen topologies at
                     N=100 sit at 0.48-0.52, i.e. no worse than in-distribution.
                     Topology is free; scale is not.

Does NOT support:    comparing delta_f1 ACROSS topologies. WS scores 0.9515
                     because WS cascades are intrinsically more predictable, not
                     because the model transfers better. Only metrics with a
                     distribution-independent null -- effect_mae_norm against
                     1.0 -- are comparable across families, which is why the
                     Q1-under-shift claim rests on that column.
Next:                synthetic -> real; and the same table for LT.
```

---

## 4. Q4 — decision utility (`DONE`, negative)

```
Experiment:          15 held-out graphs, 16 real IM algorithms from
                     coding_agent.tools.algorithms, budget 5% of N, NDlib MC
                     oracle at 32 runs/candidate -- 7680 trusted simulator calls
Hypothesis:          the coding agent needs the ORDER, not per-node accuracy, so
                     ranking quality may be high where prediction is mediocre
```

| metric | value | null |
| --- | --- | --- |
| **preference_accuracy** | **0.392 ± 0.103** | 0.5 |
| Kendall tau | +0.293 ± 0.183 | 0 |
| Precision@1 | 0.133 ± 0.340 | 1/16 = 0.063 |
| calls-to-first-win (pooled) | WM 43 · random 45 · degree 55 · oracle 15 | — |
| **trusted-call reduction (pooled)** | **+4.4% vs random** | 0 |

```
Result:              negative. The model does not order candidates usefully.
Cause, measured:     DISCRIMINATION, not direction. 43% of pairs (50.9 of 119.3
                     per graph) receive an IDENTICAL predicted score -- only 3-9
                     distinct values across 16 candidates. On one graph, 10 of 16
                     candidates all scored 44.550 while their true spreads ranged
                     41.25 to 46.28.
                     Accuracy by oracle margin: 0.210 (<0.5 nodes), 0.303, 0.311,
                     and 0.503 on the half of pairs the oracle separates by >= 2
                     nodes. Even on the easiest half the model is AT CHANCE.
                     The positive Kendall tau is not a contradiction: tau-b drops
                     model ties from its denominator, preference_accuracy counts
                     them as failures. Among pairs it does separate, direction is
                     better than chance; it just cannot separate.
Supports:            nothing positive about decision utility on BA-100.
Does NOT support:    the earlier "world model beats degree" claim, already
                     withdrawn after the planning-evaluator fix.

ARTEFACT CAUGHT:     the seen-vs-unseen policy gap looked like policy
                     generalization failing --
                        seen 0.590 vs unseen 0.300.
                     It is one candidate. `random_seeds` spreads 17.5 against 40+
                     for every other algorithm, so it is a free pair on every
                     graph and it is in the seen set. Excluding it:
                        seen 0.311 vs unseen 0.300 -- the gap vanishes.
                     Correct statement: the model ranks neither seen nor unseen
                     policies better than chance.
Next:                this is a BA-100 result. The likely cause is that
                     degree-like heuristics pick heavily overlapping seed sets on
                     a scale-free graph, so candidates really are near-identical
                     and the ensemble rollout cannot resolve them. Rerun on ER /
                     WS / SBM, where candidate seed sets diverge, before
                     concluding the model cannot rank at all.
```

---

## 5. Two measurement bugs found by running the experiments

**Thread thrashing, 40×.** Torch grabs every core; on a 100-node graph evaluated
one record at a time that costs **80.0 ms/record at 14 threads against 2.02
ms/record at 1** — intra-op synchronisation dwarfs the arithmetic. Eval scripts
now default to 1 thread. The first Stage-B launch appeared hung for 20 minutes
because of this.

**Mean-of-ratios call reduction.** `aggregate()` averaged per-graph reduction
ratios whose denominator is often 1–2 calls, so one lucky random baseline swings
the mean. Same Q4 data: mean-of-ratios **−0.366**, pooled **+0.044**. Pooled is
now reported and is the number to quote.

---

## 6. Question-level completion

| | status | established | missing |
| --- | --- | --- | --- |
| **Q1** | **~90%** (IC) / ~40% (LT) | IC uses actions (0.53 vs 1.0 null, ρ 0.91, shuffle −0.66); both trivial baselines fail at exactly 1.0; **survives all 6 shifts** | LT margin thin; the size decay needs a cause |
| **Q2** | **~95%** | linear saturates at +49.28 vs structured −0.197 on identical data; guarantees exact at any parameter; LT bias diagnosed as partial observability with the learned head beating the analytic floor | GAT/GT/GCNII controls (low priority) |
| **Q3** | **~60%** | prediction transfers to 10× size and 3 unseen topologies, frozen; action mechanism decays with size, not topology | synthetic→real; LT; policy axis is confounded on BA |
| **Q4** | **~70%** | full chain implemented and run; result is negative with a measured cause | rerun on non-BA graphs before concluding |
| **Q5** | ~40% | 56 pairs classified; only IM↔Adaptive is EXACT_CHECKPOINT; zero-shot enforced by 4 mechanisms + 4 tests | every transfer run |

266 tests pass.

---

## 7. Next

1. **IM → Adaptive IM frozen zero-shot** — the headline Q5 result; everything is in place.
2. **Q4 on ER / WS / SBM** — the negative result may be BA-specific; candidate seed sets overlap far less off scale-free graphs.
3. **Why does the action mechanism decay with size?** BA-1000's effect_mae_norm 0.87 is close to the null while its prediction is fine. Most likely the 3-layer receptive field covers a shrinking fraction of a larger graph. Testable by varying `--n-layers` at fixed N.
4. **Synthetic → real** — configs exist.
