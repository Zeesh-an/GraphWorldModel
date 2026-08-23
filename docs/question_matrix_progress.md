# Task × Question progress matrix

Every experiment in this file states which question it answers, what would have
falsified it, and what it does NOT support. Cells with no run behind them say
`NOT_RUN`; nothing is estimated.

Status vocabulary: `DONE` · `PARTIAL` · `RUNNING` · `NOT_RUN` · `NOT_APPLICABLE` · `BLOCKED`

---

## 1. The matrix

| Task | Q1 Action | Q2 Structure | Q3 Generalization | Q4 Decision utility | Q5 Transfer |
| --- | --- | --- | --- | --- | --- |
| **IM + IC** | `DONE` (ID) · `NOT_RUN` (OOD) | `RUNNING` structured vs linear | `NOT_RUN` | `PARTIAL` metrics built, unrun | → Adaptive `NOT_RUN` |
| **IM + LT** | `PARTIAL` weak (0.83 vs 1.0 null) | `DONE` diagnostic · `RUNNING` linear | `NOT_RUN` | `NOT_RUN` | later |
| **Adaptive IM** | `NOT_RUN` | shared with IM | `NOT_RUN` | `NOT_RUN` | ← IM `EXACT_CHECKPOINT`, `NOT_RUN` |
| **Source Localization** | `NOT_APPLICABLE` (base task) | forward dynamics | `NOT_RUN` | `NOT_RUN` | ← IM `FORWARD_DYNAMICS`, `BLOCKED` |
| **Cascade Reconstruction** | `NOT_APPLICABLE` | forward dynamics | `NOT_RUN` | `NOT_RUN` | ← IM `FORWARD_DYNAMICS`, `BLOCKED` |
| **Cascade Prediction** | `NOT_APPLICABLE` | forward dynamics | `NOT_RUN` real OOD | `NOT_RUN` | ← synthetic WM, `BLOCKED` |
| **CND** | `NOT_RUN` removal action | `blocked` T_exo | `NOT_RUN` | `NOT_RUN` | ← IM `MECHANISM_ONLY`, `NOT_RUN` |
| **Influence Blocking** | `NOT_RUN` | competitive head | `NOT_RUN` | `NOT_RUN` | ← CND `MECHANISM_ONLY`, `BLOCKED` |
| **Epidemic Control** | `NOT_RUN` | compartment head | `NOT_RUN` | `NOT_RUN` | mechanism only, `BLOCKED` |

`BLOCKED` = the task's head or module lives on the 8-task branch and has not been
ported into this repository. Classification is available; execution is not.

---

## 2. Experiments

### E1 — Action conditioning, IM + IC, in-distribution

```
Experiment:          action_conditioning block, SAGE + structured, IC, BA-100 x 20
Scientific question: Q1 — does the model use the action, or only the state?
Hypothesis:          a model that reads the action scores effect_mae_norm well
                     below the 1.0 null and loses accuracy when actions are shuffled
Setup:               graph-disjoint split, held-out test graphs, seed 42
Result:              effect_mae_norm 0.5294 · pearson 0.9140 ·
                     shuffle_delta_f1_drop 0.6556 · null_delta_f1_drop 0.7510
                     ConstantModel 1.0000 FAIL · StateOnlyModel 1.0000 FAIL
Supports:            the IC model genuinely uses the action; it is not exploiting
                     state persistence. The two trivial baselines land exactly on
                     the null, so the test can fail and does.
Does NOT support:    that the action MECHANISM generalizes. Every number here is
                     in-distribution.
Next:                rerun these three metrics under graph OOD (D1/D2)
```

### E2 — Action conditioning, IM + LT

```
Result:              effect_mae_norm 0.8316 · pearson 0.7732 · shuffle drop 0.2614
Supports:            detectably better than predicting no action effect
Does NOT support:    that LT action effects are predicted WELL. 0.83 against a 1.0
                     null is a weak margin and must not be quoted beside IC's 0.53.
Next:                after the LT partial-observability question (E5) is settled
```

### E3 — Algebraic intervention guarantees

```
Experiment:          exogenous_fidelity + weight-perturbation tests
Scientific question: Q2 — what does the structured transition buy?
Hypothesis:          T_exo is a property of the computation graph, so seeded and
                     removed nodes must be exact at ANY parameter value
Setup:               untrained and trained models, weights scaled x1/x50/x(-50)/x0
Result:              seeded_p_infected_worst 0.9999989 (the clamp bound) and
                     removed_p_frontier_worst 1.0e-6, unchanged under every
                     perturbation, for both IC and LT
Supports:            intervention semantics are guaranteed by construction, not
                     learned. An untrained model scores identically.
Does NOT support:    anything about how much was LEARNED. These properties hold
                     with random weights, which is exactly the point and also
                     exactly the limit of the claim.
Next:                the linear control (E4) — without it "structured avoids
                     pathological rollouts" has no same-protocol comparison
```

### E4 — structured vs linear control · `RUNNING`

```
Experiment:          SAGE, BA-100 x 100 graphs, graph-disjoint, IC and LT,
                     head in {structured, linear}
Scientific question: Q2 — value of the structured transition
Hypothesis:          the structured head preserves deterministic intervention
                     semantics and avoids runaway rollouts; it need NOT win every
                     one-step metric
Setup:               one canonical 100-graph manifest, 70/15/15, 15 held-out test
                     graphs, seed 42, identical training budget and eval suite.
                     linear is run at pos_weight in {off, auto}: `off` was chosen
                     FOR the structured head, and a control crippled by the
                     treatment's hyperparameter is not a control.
Result:              RUNNING
Falsified if:        linear matches structured on rollout calibration AND honours
                     seed/remove semantics — that would make the structure
                     decorative
Next:                fills the Q2 column for IM+IC and IM+LT
```

### E5 — LT analytic-oracle diagnostic · `DONE`

```
Experiment:          scripts/lt_oracle_diagnostic.py
Scientific question: Q2 — is the +12 LT rollout bias the learned model's fault or
                     the LT formulation's?
Hypothesis:          if the bias is structural, a zero-learning optimal predictor
                     shows it too
Setup:               --head structured_oracle is IC-only by design (LT thresholds
                     are never stored). But the simulator draws theta ~ U(0,1), so
                     P(activate | f_v) = f_v EXACTLY: p_new = f_v is the
                     Bayes-optimal memoryless LT predictor. Fed through the same
                     rollout_ensemble, matched seed, same test split.
Result:              analytic oracle    ens_marg_mae 0.2261  count_bias +22.66
                     learned structured ens_marg_mae 0.1424  count_bias +12.17
                     (true final count 64.71 in both)
Supports:            the LT rollout bias is PARTIAL OBSERVABILITY, not a fitting
                     failure — and the learned head is about twice as well
                     calibrated as the analytic floor. Mechanism: LT is
                     deterministic given theta, and theta persists within an
                     episode. A memoryless predictor redraws P(theta < f) every
                     step and so ignores the evidence carried by earlier
                     non-activations: a node that failed at f=0.3 has theta>0.3,
                     making its true conditional at f=0.4 equal to 0.143, not 0.4.
                     Ignoring that systematically over-activates.
Does NOT support:    that the learned LT model is good in absolute terms. +12 on a
                     65-node cascade is still large.
Next:                the implied fix is state augmentation — carry a posterior
                     over theta through the rollout — NOT more fitting. Report LT
                     as a harder setting with a named information-theoretic cause.
```

### E6 — Corrected planning baseline · `DONE`

```
Experiment:          Stage A1/A2 rerun, BA-100 x 20, held-out planning graphs
Scientific question: Q4 — is the model useful for decisions?
Hypothesis (prior):  the model beats the degree heuristic on planning regret
Setup:               planning_split=test, train_overlap=0, val_overlap=0
Result:              IC  model 0.2479 +- 0.083   degree 0.2229
                     (under the previous leaky selection: 0.2437 vs 0.2687)
Supports:            nothing positive. The direction REVERSED once the evaluator
                     stopped scoring training graphs.
Does NOT support:    "the world model beats degree". That claim is withdrawn.
Limitation:          n = 3 test graphs at 20 graphs total; and under the leaky
                     selection both arms produced identical plan_regret_model
                     (0.2437 vs 0.2437), so on BA the metric is dominated by the
                     candidate sampler, not the model.
Next:                E7 replaces planning regret as the primary Q4 metric; E4's
                     100-graph pool raises the held-out count from 3 to 15
```

### E7 — Q4 ranking evaluator · `PARTIAL` (built, not run)

```
Experiment:          world_model/wm_ranking.py + scripts/eval_ranking.py
Scientific question: Q4 — can the model ORDER candidates well enough to replace
                     trusted evaluation?
Hypothesis:          the coding agent does not need per-node accuracy; it needs
                     the ordering. Ranking quality may be high where per-node
                     accuracy is mediocre.
Setup:               real IM algorithms from coding_agent.tools.algorithms produce
                     seed sets on each HELD-OUT graph; the oracle is the NDlib MC
                     spread; the model score is a frozen-model rollout. The runner
                     refuses if any scoring graph appears in train or val.
                     Chain: preference accuracy -> top-K -> calls-to-first-win ->
                     trusted-call reduction, each strictly stronger.
Result:              NOT_RUN (implementation DONE, 26 tests)
Guards (tested):     a constant predictor scores 0.0 not 1.0 (model ties on a
                     separable pair are failures); oracle-tied pairs are excluded
                     rather than resolved; a ranking where nothing wins costs
                     len+1 rather than being dropped; accuracy is bucketed by
                     oracle margin so a number earned on sub-noise gaps is visible.
Next:                run on E4's checkpoints. Same run yields Workstream E
                     (seen vs unseen policy) as a filter, not a second protocol.
```

### E8 — Split protocol correction · `DONE`

```
Scientific question: methodological — were the previous numbers measuring what
                     they were read as?
Result:              legacy episode_random put 20/20 graphs in more than one
                     split; graph_disjoint puts 0/20. Prediction metrics barely
                     moved.
Supports:            the correction is real and cheap. AND the negative reading:
                     BA-100 -> BA-100 is too homogeneous to be generalization
                     evidence — if removing a total leak changes nothing, the
                     setting had nothing to leak.
Does NOT support:    "leakage does not matter". It matters wherever the
                     distribution is not near-exchangeable, which is every Q3
                     setting.
Next:                G1.1 is a control only. Real Q3 needs size/topology/real shift.
```

---

## 3. Question-level completion

| | status | what is established | what is missing |
| --- | --- | --- | --- |
| **Q1** | ~80% (IC) / ~40% (LT) | IC uses actions, with both trivial baselines failing at exactly 1.0 | OOD action fidelity; LT margin is thin |
| **Q2** | ~65% | algebraic guarantees hold at any parameter value; LT bias diagnosed as partial observability, learned head beats the analytic floor | the linear control (running) |
| **Q3** | ~5% | one negative result: BA-100 cannot test generalization | every OOD run |
| **Q4** | ~35% | leakage corrected; prior claim withdrawn; full ranking chain implemented and guarded | every ranking run |
| **Q5** | ~40% | 56 pairs classified from derived properties; only IM↔Adaptive IM is EXACT_CHECKPOINT; zero-shot enforced by 4 mechanisms + 4 tests | every transfer run |

---

## 4. Next experiments, in order

1. **E4 completes** → fills Q2 for IC and LT (running, ~2–4 h)
2. **E7 on E4's checkpoints** → first real Q4 numbers, 15 held-out graphs
3. **IM → Adaptive IM frozen zero-shot** → the headline Q5 result
4. **D1/D2 graph OOD, with Q1 metrics rerun on OOD graphs** → Q3, and upgrades
   Q1 from "uses actions" to "the action mechanism generalizes"
5. **Policy OOD** as a filter over E7's output → Q3's second axis
