# Task × Question progress matrix

Every experiment states which question it answers, what would have falsified it,
and what it does NOT support. Empty cells say `NOT_RUN`; nothing is estimated.

Canonical setting: `influence_maximization`, BA-100 × **100 graphs**,
graph-disjoint (70/15/15 → **15 held-out test graphs**), SAGE + `structured`,
seed 42, run on a single RTX A6000. 266 tests pass.

---

## 0. The claim this evidence supports

> An action-conditioned graph world model learns **reusable graph dynamics**. The
> learned transition survives 10× graph-size and three unseen topologies frozen;
> the same frozen checkpoint transfers to a different task without fine-tuning
> and matches a 372,480-call Monte-Carlo oracle; and where candidate algorithms
> are actually distinguishable it ranks them well enough to cut trusted simulator
> calls by **43–49%**.

Each clause below is a measurement with its own null, and each carries the
setting where it fails.

---

## 1. The matrix

| Task | Q1 Action | Q2 Structure | Q3 Generalization | Q4 Decision utility | Q5 Transfer |
| --- | --- | --- | --- | --- | --- |
| **IM + IC** | `DONE` ID **and** OOD | `DONE` structured ≫ linear | `DONE` size + topology | `DONE` **+43–49% on WS/SBM** | `DONE` **→ Adaptive, oracle-parity** |
| **IM + LT** | `PARTIAL` weak (0.83 vs 1.0) | `DONE` + analytic-oracle diagnostic | `NOT_RUN` | `NOT_RUN` | later |
| **Adaptive IM** | inherited, frozen | shared with IM | `NOT_RUN` | `DONE` (as Q5 target) | ← IM `EXACT_CHECKPOINT` ✅ |
| **Source Localization** | `NOT_APPLICABLE` | forward dynamics | `NOT_RUN` | `NOT_RUN` | ← IM `FORWARD_DYNAMICS`, `BLOCKED` |
| **Cascade Reconstruction** | `NOT_APPLICABLE` | forward dynamics | `NOT_RUN` | `NOT_RUN` | ← IM `FORWARD_DYNAMICS`, `BLOCKED` |
| **Cascade Prediction** | `NOT_APPLICABLE` | forward dynamics | `NOT_RUN` | `NOT_RUN` | ← synthetic WM, `BLOCKED` |
| **CND** | `NOT_RUN` | `blocked` T_exo | `NOT_RUN` | `NOT_RUN` | ← IM `MECHANISM_ONLY`, `NOT_RUN` |
| **Influence Blocking** | `NOT_RUN` | competitive head | `NOT_RUN` | `NOT_RUN` | `BLOCKED` |
| **Epidemic Control** | `NOT_RUN` | compartment head | `NOT_RUN` | `NOT_RUN` | `BLOCKED` |

`BLOCKED` = head or task module lives on the 8-task branch, not ported.

---

## 2. Q5 — cross-task transfer, frozen zero-shot ✅

```
Experiment:  IM-trained checkpoint -> adaptive_online_im, 15 held-out graphs,
             3 rounds x gap 2, budget 5% of N, 8 campaigns/graph, shortlist 20
Hypothesis:  if action conditioning is a real mechanism and not a fit to t=0
             seeding, the frozen model should answer the mid-cascade question
             "which node is worth seeding NOW" -- which its training
             distribution only ever contained as uniformly-injected noise
LLM-free:    every arm is a (state, graph, batch) -> seeds policy; only the
             scorer varies, and NDlib rolls every arm forward
```

| arm | spread | gap vs oracle | span recovered | **trusted calls** |
| --- | --- | --- | --- | --- |
| oracle (CRN greedy, MC-32) | **38.73** | 0 | 1.000 | **372,480** |
| **world_model (frozen IM)** | **36.85** | +1.88 | **0.906** | **0** |
| degree | 35.93 | +2.79 | 0.860 | 0 |
| static_degree | 29.16 | +9.57 | 0.521 | 0 |
| random | 18.74 | +19.98 | 0.000 | 0 |

Paired over the same 15 graphs and campaigns:

| comparison | Δ | t (df 14) | separated |
| --- | --- | --- | --- |
| WM − random | +18.11 | 19.15 | **yes** (15/15 graphs) |
| WM − static_degree | +7.69 | 6.11 | **yes** |
| **oracle − WM** | +1.88 | **1.05** | **no** |
| WM − degree | +0.92 | 1.40 | no (11/15 graphs) |

```
Supports:    EXACT_CHECKPOINT transfer works. The frozen IM checkpoint is
             statistically indistinguishable from a Monte-Carlo oracle that spent
             372,480 simulator rollouts, while spending zero. It beats the
             non-adaptive control decisively (t=6.11), so it is exploiting the
             adaptive structure rather than replaying a static seeding rule.
             `frozen_parameters_unchanged: true`, verified tensor by tensor;
             required_semantic_overrides = 0.
Does NOT support:
             "beats degree" -- +0.92 at t=1.40 is not separated, on 11/15 graphs.
             Single seed, one graph family, IC only.
```

**Two design bugs this experiment surfaced, both fixed before the number above.**
(1) Candidate sets were not shared: degree and random saw all N susceptible nodes
while the oracle and the world model saw a random shortlist, which on a
scale-free graph made the *oracle the worst arm*. (2) The oracle had no common
random numbers: greedy compares total spread (~34 nodes) while candidates differ
by 1–2 at a per-rollout sd around 8, so at 32 independent draws its argmax was
near noise and it scored 34.00, *below* both degree and the world model. With CRN
it scores 38.73 and is a ceiling. **An oracle that loses is a broken oracle, not
a finding.**

---

## 3. Q4 — decision utility ✅ (and the BA negative, explained)

```
Experiment:  15 held-out graphs per family, 16 real IM algorithms from
             coding_agent.tools.algorithms, budget 5% of N, NDlib MC oracle at
             32 runs/candidate
Hypothesis:  the coding agent needs the ORDER, not per-node accuracy
```

| dataset | pref. accuracy (null 0.5) | Kendall τ | model ties | **resolution** | **calls-to-first-win** | **saving vs random** |
| --- | --- | --- | --- | --- | --- | --- |
| BA-100 (source, ID) | **0.392** | 0.293 | **42.7%** | **0.458** | 43 vs 45 | +4.4% |
| ER-100 (OOD) | 0.614 | 0.338 | 9.3% | 0.705 | 47 vs 53 | +11.3% |
| **SBM-100 (OOD)** | **0.704** | 0.448 | 3.3% | 0.869 | **30 vs 53** | **+43.4%** |
| **WS-100 (OOD)** | **0.722** | 0.473 | **2.4%** | **0.956** | **28 vs 55** | **+49.1%** |

"resolution" = predicted spread sd / true spread sd over the 15 non-trivial
candidates (`random_seeds` excluded). The world model beats **degree ranking** on
all four (43<55, 47<62, 30<44, 28<56).

**The BA failure is candidate degeneracy, measured.** On a scale-free graph the
degree-like IM heuristics converge on the same hubs: 15 of 16 candidates have
true spreads within ~1.6 nodes (vs 2.2–2.8 elsewhere), and the model resolves
only 46% of that already-small gap, so 43% of pairs tie and ordering falls below
chance. Resolution and preference accuracy are **monotone in the same order**
across all four families — 0.458 → 0.705 → 0.869 → 0.956 against 0.392 → 0.614 →
0.704 → 0.722.

```
Supports:    where candidates are distinguishable, the frozen model orders them
             well above the 0.5 null and cuts trusted calls 43-49%. ER/WS/SBM are
             all OOD for a BA-trained model, so this is Q4 and Q3 at once.
Does NOT support:
             any ranking claim on BA-100. Single seed; call-reduction pooled
             across 15 graphs, no per-graph CI.
```

**Policy generalization holds.** Unseen-policy preference accuracy matches seen on
every dataset — 0.579/0.567 (ER), 0.725/0.644 (WS), 0.705/0.596 (SBM) — with
`random_seeds` excluded from the seen side. An earlier apparent gap (0.590 vs
0.300 on BA) was **entirely** that one candidate: it spreads 17.5 against 40+ for
everything else, a free pair on every graph. Excluding it, 0.311 vs 0.300.
**The model ranks by state-action dynamics, not algorithm identity.**

---

## 4. Q3 — generalization ✅

Frozen BA-100 IC checkpoint on 6 targets, all `split_mode=eval_only` (no train
split exists) and **structurally disjoint** from the source: every graph hashed
by `(num_nodes, canonical sorted edge list)`, zero collisions against any source
split. Density matched (target mean degree 5.80–6.00 vs source 5.82).

| target | N | ΔF1 | marg_mae | count_bias | model/true | **effect_mae_norm** | pearson |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BA-100 (ID control) | 100 | 0.8559 | 0.0913 | −0.20 | 45.2 / 45.7 | **0.5325** | 0.911 |
| BA-200 | 200 | 0.8529 | 0.0921 | −0.60 | 91.8 / 92.3 | 0.6346 | 0.870 |
| BA-500 | 500 | 0.8537 | 0.0950 | +1.19 | 238.4 / 236.1 | 0.7861 | 0.770 |
| BA-1000 | 1000 | 0.8501 | 0.1039 | +2.86 | 493.2 / 489.1 | **0.8676** | 0.667 |
| ER-100 | 100 | 0.8775 | 0.0849 | −1.92 | 35.7 / 38.1 | 0.5186 | 0.915 |
| WS-100 | 100 | 0.9515 | 0.0681 | −0.91 | 29.8 / 30.9 | **0.4764** | 0.926 |
| SBM-100 | 100 | 0.8871 | 0.0791 | −1.26 | 34.2 / 35.6 | 0.5045 | 0.919 |

```
Supports:    prediction transfers -- count bias within +-2.9 nodes on a 490-node
             cascade at 10x training size, frozen. T_exo exact everywhere
             (0.9999989 / 1.0e-6). Action conditioning PASSES on all seven.
Key finding: the action MECHANISM degrades with SIZE, not TOPOLOGY.
             effect_mae_norm 0.53 -> 0.63 -> 0.79 -> 0.87 as N goes 100 -> 1000,
             approaching the 1.0 null; the three unseen topologies at N=100 sit
             at 0.48-0.52, no worse than in-distribution.
Does NOT support:
             comparing delta_f1 ACROSS topologies. WS scores 0.9515 because WS
             cascades are intrinsically more predictable. Only a metric with a
             distribution-independent null -- effect_mae_norm against 1.0 -- is
             comparable across families, which is why the Q1-under-shift claim
             rests on that column.
```

### 4.1 Why the action mechanism decays with size — the receptive-field
hypothesis is REFUTED

```
Experiment:  n_layers in {1, 2, 3, 8}, everything else identical, each frozen
             checkpoint evaluated on BA-100 (ID) and BA-1000 (10x OOD)
Hypothesis:  a 3-layer receptive field covers a shrinking fraction of a larger
             graph, so depth should recover the lost action fidelity
```

| layers | BA-100 ΔF1 | BA-100 eff_mae | BA-1000 ΔF1 | **BA-1000 eff_mae** |
| --- | --- | --- | --- | --- |
| 1 | 0.8558 | 0.5328 | 0.8502 | **0.8677** |
| 2 | 0.8560 | 0.5325 | 0.8499 | **0.8677** |
| 3 | 0.8559 | 0.5325 | 0.8501 | **0.8676** |
| 8 | 0.8552 | 0.5333 | 0.8493 | **0.8681** |

**Flat to four decimals.** Depth buys nothing, at either size. The hypothesis is
refuted, and the flatness says something sharper: a 1-layer encoder matches an
8-layer one, so the ENCODER contributes almost nothing here. The structured IC
head feeds `w_uv` straight into the edge MLP and largely reproduces `q = w` —
which is the "IC consumes the true w" caveat, now measured rather than suspected.

Decomposing the normalised metric shows what actually moves:

| N | E\|d_true\| (signal) | E\|error\| | effect_mae_norm | magnitude_ratio |
| --- | --- | --- | --- | --- |
| 100 | 0.02253 | 0.01200 | 0.5325 | 0.549 |
| 200 | 0.01468 | 0.00932 | 0.6346 | 0.428 |
| 500 | 0.01030 | 0.00810 | 0.7861 | 0.251 |
| 1000 | 0.00861 | **0.00747** | 0.8676 | **0.155** |

```
Supports:    the model's ABSOLUTE counterfactual error IMPROVES with graph size
             (0.01200 -> 0.00747, -38%). Prediction quality does not degrade.
Actual cause: under-reaction. The predicted effect magnitude falls to 15.5% of
             the true one, while the true per-node effect also shrinks (one seed
             touches a smaller fraction of a bigger graph). effect_mae_norm rises
             because the denominator shrinks faster than the numerator AND the
             model increasingly under-predicts -- not because it got worse at
             predicting.
Does NOT support:
             "the action mechanism fails at scale". It attenuates. The right
             follow-up is calibration of the effect magnitude, not capacity.
Next:        RESOLVED -- see section 5.2. The `--hide-edge-weights` ablation was
             vacuous as configured (w was exactly 1/in-degree, r = 1.000000, and
             degree is an input channel). Rerun on i.i.d. weights: hiding w costs
             0.110 delta_f1 (t = 197.7), and effect_mae_norm still reaches 0.6167
             against the 1.0 null. Both halves hold -- w is used, and the model
             does not merely read the answer off it.
```

---

## 5. Q2 — structured vs linear ✅

### 5.1 Single-seed table

Same data, split, seed, budget, evaluation suite. `linear` run at both
`pos_weight` settings because `off` was chosen *for* the structured head.

| arm | ΔF1 | marg_mae | count_bias | model/true | seeded_worst | removed_frontier_worst |
| --- | --- | --- | --- | --- | --- | --- |
| **structured IC** | 0.8559 | **0.0913** | **−0.197** | 45.2 / 45.7 | **0.9999989** | **1.0e-6** |
| linear IC | 0.8526 | 0.4661 | **+49.28** | **96.7** / 45.7 | 0.9908 | 0.0078 |
| **structured LT** | 0.6736 | 0.1538 | +15.13 | 81.6 / 60.8 | **0.9999989** | **1.0e-6** |
| linear LT (auto) | 0.6754 | 0.3806 | +40.46 | 98.4 / 60.8 | 0.9930 | 0.4410 |
| linear LT (off) | 0.6669 | **0.0884** | **+0.33** | 61.8 / 60.8 | 0.9962 | 0.2661 |

One-step accuracy is **tied** — the structure buys nothing there, as predicted.
It buys rollout stability (linear IC saturates, predicting 96.7 of 100 nodes
infected against a true 45.7) and exactness at any parameter value.

**Trap worth reporting:** the *thresholded* `add_seed_success` /
`remove_frontier_success` are 1.0000 for **every** arm including linear. Only the
probability-level worst case separates them — and the rollout consumes
probabilities.

**The LT anomaly, chased down.** linear+off wins LT rollout calibration by
*violating monotonicity*: on already-active nodes it predicts P(still infected)
= 0.91 and violates on **100%** of them, dropping ~9% of the infected set every
step, which cancels LT's over-prediction. Two errors cancelling, not calibration.
The structured head violates on **0%** under ×1/×20/×(−20) perturbation
(regression test).

**LT bias is partial observability, not fitting.** LT has an analytic oracle —
thresholds are U(0,1), so `P(activate | f_v) = f_v` exactly. On the same data it
scores marg_mae 0.2302 / bias +24.36 against the learned head's 0.1538 / +15.13:
**the learned head beats the memoryless optimum**. LT is deterministic given θ
and θ persists; a memoryless predictor ignores the evidence in earlier
non-activations. The fix is state augmentation, not more fitting.

---

### 5.2 Multi-seed confirmation (paired, n = 5 seeds) ✅

Seeds 0-4 per arm, identical data and hyper-parameters. Seed is a blocking
factor, so differences are PAIRED within seed. 95% t-intervals.

| metric | structured | linear | paired diff | t | excl. 0 |
| --- | --- | --- | --- | --- | --- |
| delta_f1 (one-step) | 0.8558 ± 0.0003 | 0.8512 ± 0.0008 | +0.0046 | 12.65 | yes |
| **ens_marg_mae** | **0.0948 ± 0.0017** | 0.4832 ± 0.0124 | **−0.3884** | **−85.93** | yes |
| **ens_count_bias** | **−0.64 ± 1.05** | **+49.07 ± 1.12** | **−49.70** | **−88.13** | yes |
| effect_mae_norm | 0.5328 ± 0.0002 | 0.5412 ± 0.0060 | −0.0084 | −3.98 | yes |
| add_seed_success | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 | — | no |
| action_sensitivity | 1.7126 ± 0.0177 | 1.9126 ± 0.0759 | −0.2000 | −7.83 | yes |

```
Question:    is the single-seed +49.28 a fluke of one run?
Answer:      no. +49.07 ± 1.12 across five seeds, t = -88.1 paired.
Key point:   one-step delta_f1 separates the heads by 0.0046 and rollout count
             bias separates them by 49.7 nodes on a 100-node graph. A reader who
             saw only the one-step row would call the heads interchangeable.
Careful:     add_seed_success is 1.0000 for BOTH arms. For structured that is the
             algebraic identity; for linear it is a side effect of saturation --
             a model that infects everything marks seeds infected for free. Same
             number, two different mechanisms. Likewise action_sensitivity favours
             linear (1.91 vs 1.71) and means nothing: a saturating model is
             trivially sensitive.
```

Artefact: `results/seeds/summary.json` (15 runs).

---

## 5.3 Privileged information — learned dynamics vs reading the input ✅

| generator | corr(w, 1/in-deg) | cost of hiding w |
| --- | --- | --- |
| `weighted` (deterministic) | **1.000000** | none -- 9/9 metrics not significant (n=5) |
| `random` (i.i.d. U(0.02,0.4)) | 0.0342 | delta_f1 −0.110, t = 197.7 (n=3) |

On the corrected generator, paired over 3 seeds:

| metric | visible w | hidden w | paired diff | t | excl. 0 |
| --- | --- | --- | --- | --- | --- |
| delta_f1 | 0.8225 ± 0.0023 | **0.7125 ± 0.0001** | **+0.1100** | 197.72 | yes |
| brier_infected | 0.0014 | 0.0041 | −0.0027 | −260.70 | yes |
| ens_marg_mae | 0.0962 ± 0.0035 | 0.1187 ± 0.0035 | −0.0224 | −83.15 | yes |
| **effect_mae_norm** | **0.5635 ± 0.0017** | **0.6167 ± 0.0002** | **−0.0533** | −125.86 | yes |

```
Question:    does the world model learn propagation, or read the transmission
             probability off its own input channel?
Answer:      it learns it. Both halves are needed and both hold.
             (1) w IS used -- hiding it costs 0.110 delta_f1.
             (2) w is NOT the whole story -- with w fully withheld,
                 effect_mae_norm = 0.6167, still far below the reachable 1.0
                 null, i.e. ~38% of the counterfactual action effect survives
                 from structure and state alone.
Supports:    "the encoder learns reusable graph dynamics".
Does NOT     "the model is robust to losing edge weights" -- it degrades
support:     significantly and measurably.
Method note: the first configuration of this ablation ran to completion over five
             seeds and reported a clean null, because the masked quantity was
             exactly reconstructible from a channel left in place. An ablation is
             only as strong as the independence of what it removes.
```

Artefact: `results/priv/summary.json` (6 runs).

---

## 6. Q1 — action conditioning ✅

| | IC (ID) | IC (worst OOD, BA-1000) | LT |
| --- | --- | --- | --- |
| effect_mae_norm (null 1.0) | **0.5325** | 0.8676 | 0.8316 |
| effect_pearson | 0.9140 | 0.667 | 0.7732 |
| shuffle_delta_f1_drop | 0.6556 | 0.6748 | 0.2614 |
| ConstantModel / StateOnlyModel | **1.0000 FAIL** | — | — |

Both trivial baselines land exactly on the null and fail, so the test can fail
and does. The mechanism survives all six distribution shifts; it weakens with
graph size (§4).

---

## 7. Measurement bugs found by running the experiments

| bug | effect | fix |
| --- | --- | --- |
| Torch thread thrashing | **40× slower** on small graphs (80.0 vs 2.02 ms/record at 14 vs 1 thread) | eval scripts default to 1 thread |
| Mean-of-ratios call reduction | −0.366 vs pooled +0.044 on the same data | pooled estimator reported |
| Unshared candidate sets (Q5) | made the **oracle the worst arm** | every arm ranks the same shortlist |
| No common random numbers (Q5) | oracle at 34.00, *below* degree and WM | CRN → 38.73, a real ceiling |
| Planning evaluator split leakage | model "beat" degree 0.2437 vs 0.2687; corrected 0.2479 vs 0.2229 | test-split only + overlap provenance |
| LT frontier semantics | `removed_p_frontier_worst` 0.92 | `active_pre` / `active_post` split |
| `episode_random` splitting | 20/20 graphs straddled train/test | `graph_disjoint` default |

---

## 8. Completion

| | status | headline |
| --- | --- | --- |
| **Q1** | 🟩 ~95% (IC) / 🟨 40% (LT) | 0.53 vs a 1.0 null; both trivial baselines fail; survives all 6 shifts; **withholding edge weights entirely still leaves 0.6167 < 1.0**, so it is not reading the answer off the input |
| **Q2** | 🟩 ~98% | **5 paired seeds**: linear +49.07 ± 1.12 vs structured −0.64 ± 1.05 count bias, t = −88.1, while one-step ΔF1 differs by 0.0046; guarantees exact at any parameter |
| **Q3** | 🟩 ~75% | prediction transfers to 10× size and 3 unseen topologies frozen; action mechanism decays with size, not topology |
| **Q4** | 🟩 ~80% | **+43–49% trusted-call reduction on WS/SBM**; BA negative explained by candidate degeneracy |
| **Q5** | 🟩 ~80% | **frozen IM checkpoint indistinguishable from a 372,480-call oracle** (t=1.05) at zero cost |

Remaining: LT extensions, synthetic→real, and the six unported tasks. The depth
study is complete (receptive-field hypothesis refuted, §4.1). Large-graph
wall-clock break-even is **deliberately not reported**: the only large graphs
available are BA, and BA is the family where ranking is degenerate, so the
break-even denominator (calls saved) is ~0 and the threshold is unbounded and
uninterpretable. That is a limitation of the available graph pool, not a
negative result.
