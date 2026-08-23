# Action-Conditioned Graph World Model

## Structured dynamics, falsifiable action tests, and measured decision value

---

## Core statement

The model does not learn every consequence of an intervention from scratch. It
applies the immediate, deterministic effect of an action with a closed-form
structured transition, then learns the uncertain diffusion that follows. This
makes direct action semantics a property of the **computation graph** rather than
a behaviour training may or may not discover.

```
        trusted simulator                    structured world model
        IC / LT rollouts        ──────▶      T_exo fixed,  T_endo learned
               │                                      │
               │  transition data                     │  cheap prediction
               │  (G, s_t, a_t, y_{t+1})              │  and ranking
               ▼                                      ▼
        counterfactual forks                   coding agent / planner
               │                                      │
               └──────────────▶  true-environment evaluation  ◀────────┘
                                 fidelity · action effect · decision value
```

Every number in this report was measured on this repository. Where a result is
negative it is reported as negative, and where a measurement turned out to be
testing the wrong thing that is stated with the evidence that revealed it —
§9 lists thirteen such cases, several of which had already produced a conclusion.

**On uncertainty.** The two ablations the argument actually rests on — structured
versus unconstrained head (§6.1.1) and privileged information (§6.3) — are run
across seeds with everything else held fixed, so seed is a *blocking factor* and
differences are reported as **paired** within-seed intervals. On five runs an
unpaired interval is wide enough to hide effects the paired test resolves at
$t > 80$; quoting two independent means would have understated every result
below.

---

## 1. Problem formulation

Learn an action-conditioned transition operator that can replace repeated Monte
Carlo diffusion calls inside a planner:

$$f_\theta(G, s_t, a_t) \longrightarrow s_{t+1}$$

Here $G=(V,E,w)$ is a weighted graph, $s_t = (x^{\text{inf}}_t, x^{\text{fr}}_t)$
records all previously infected nodes and the current propagation frontier, and
$a_t$ is a bag of node or edge interventions drawn from
`add_node`, `remove_node`, `add_edge`, `remove_edge`, `set_edge_weight`.

### 1.1 The central decomposition

$$s_{t+1} \;=\; T_{\text{endo}}\big(T_{\text{exo}}(s_t, a_t)\big)$$

| component | meaning | treatment |
| --- | --- | --- |
| $T_{\text{exo}}$ | immediate deterministic action effect: seed a node, remove one from the frontier, change the graph | **closed form, structurally enforced** |
| $T_{\text{endo}}$ | the stochastic diffusion that follows | **learned from transition data** |

For Independent Cascade the one-step probability of a new infection is

$$P(v \text{ newly infected at } t{+}1) \;=\; 1 - \prod_{u \to v}\Big(1 - w_{uv}\,\mathbb{1}[u \in \text{frontier}_t]\Big)$$

For Linear Threshold, activation is governed by a hidden threshold and the
fraction of active incoming weight. The threshold is redrawn per episode and
never observed, so an LT model can recover only the marginal activation
distribution — §6.3 turns that limitation into a measurement.

### 1.2 Action semantics must be task-consistent

`remove_node` has two meanings and they are different problems:

| semantics | meaning | correct use |
| --- | --- | --- |
| `spent` | a propagator has exhausted its one IC attempt; it stays counted as infected | influence maximization |
| `blocked` | the node leaves the process; it cannot propagate or be infected | containment, immunization, blocking |

The generator, the model head and the simulator cross-check this before
training. A mismatch is rejected rather than silently changing the measured
spread. §7.4 shows this single field is what makes IM → CND transfer fail.

---

## 2. Data

### 2.1 Transition data with observable counterfactual effects

Each transition records the graph, the state, an action, and Monte Carlo
**marginal** labels for the next infected and frontier states. The load-bearing
design element is the **counterfactual fork**: from the same state $s_t$ the
generator applies several alternative actions and records each outcome, making
the causal quantity

$$y(s,a) - y(s,a')$$

directly observable. Without same-state action pairs a model can appear accurate
while ignoring actions entirely — §5 measures exactly that failure in two
control models.

Targets are soft marginals estimated from repeated simulator calls, not single
Bernoulli realizations:

$$\hat y^{\text{inf}}_v = \widehat{P}(v \text{ infected at } t{+}1), \qquad
\hat y^{\text{fr}}_v = \widehat{P}(v \text{ in frontier at } t{+}1)$$

### 2.2 Splitting

Splits are **graph-disjoint**: every episode of a graph lands in one split.
Two episodes on one graph share its structure, its per-edge probabilities and
its cascade, so splitting per episode leaks.

> **Measured.** The historical per-episode split put **20 of 20** graphs in more
> than one split. Under graph-disjoint splitting: **0 of 20**.

Prediction metrics barely moved, and that is itself a finding: 20 BA(100,3)
graphs are near-exchangeable, so **BA-100 → held-out BA-100 is not
generalization evidence** and is used here only as an in-distribution control.

---

## 3. Model

A backbone encoder maps node features to embeddings, and a structured head
composes the transition:

$$h = \text{Encoder}(X, G), \qquad X \in \mathbb{R}^{N \times 6}$$

Channels: `infected`, `frontier`, `log1p(degree)`, `act_add`, `act_remove`,
`act_edge`. Channels 0–1 are state, 2 is structure, 3–5 are the action projected
onto nodes. Edge interventions additionally enter through the **post-action
adjacency**, which for IC and LT is the sufficient dynamical object.

### 3.1 The structured IC head

$T_{\text{exo}}$ first, in closed form:

$$\tilde x^{\text{inf}} = \min\big(x^{\text{inf}} + x^{\text{add}},\, 1\big), \qquad
\tilde x^{\text{fr}} = \min\big(x^{\text{fr}} + x^{\text{add}},\, 1\big)\cdot\big(1 - x^{\text{rem}}\big)$$

then the learned per-edge transmission:

$$q_{uv} = \sigma\big(\text{MLP}[h_u, h_v, w_{uv}]\big), \qquad
p^{\text{new}}_v = 1 - \prod_{u \to v}\big(1 - q_{uv}\,\tilde x^{\text{fr}}_u\big)$$

composed monotonically:

$$\hat y^{\text{inf}}_v = \tilde x^{\text{inf}}_v + \big(1 - \tilde x^{\text{inf}}_v\big)\,p^{\text{new}}_v,
\qquad
\hat y^{\text{fr}}_v = \big(1 - \tilde x^{\text{inf}}_v\big)\,p^{\text{new}}_v$$

The product is evaluated in log space with a scatter-add, so long products do not
underflow.

### 3.2 The structured LT head

LT thresholds are hidden and redrawn per episode, so the head predicts activation
as a monotone function of the active-neighbour fraction $f_v$:

$$f_v = \frac{\sum_{u\to v} w_{uv}\,\tilde x^{\text{act}}_u}{\sum_{u\to v} w_{uv}},
\qquad
p^{\text{new}}_v = \mathbb{1}[f_v > 0]\cdot\sigma\big(\tau (f_v - \hat\theta_v)\big)$$

**Correction made in this work.** `Simulator.advance` defines the LT frontier as
`active − previous_active`, snapshotting *before* the action. The head computed
"newly active" against the *post*-action state, which is wrong for both node ops
at once. The dataset says so directly:

| | true next-frontier at target | head, before fix |
| --- | --- | --- |
| LT `add_node` | 1.0 | 0 |
| LT `remove_node` | 0.0 | up to **0.92** |
| IC `add_node` | 0.0 | 0 (correct, unchanged) |

The fix keeps the two activities separate — propagation and the infected channel
use post-action activity, the frontier channel is
$\max(\hat y^{\text{inf}} - x^{\text{inf}}, 0)$ — and moves
`removed_p_frontier_worst` from 0.92 to $10^{-6}$.

### 3.3 Training objective

$$\mathcal{L} = \text{BCE}\big(\hat y^{\text{inf}}, y^{\text{inf}}\big) + \text{BCE}\big(\hat y^{\text{fr}}, y^{\text{fr}}\big)$$

against the MC soft marginals, with checkpoint selection on validation
`delta_f1`. `pos_weight` is off for structured heads: the structural form already
prevents the all-zeros collapse, and reweighting inflates the per-edge $q$.

---

## 4. The algebraic guarantee

Suppose node $v$ is selected by `add_node`. Then $\tilde x^{\text{inf}}_v = 1$ and

$$\hat y^{\text{inf}}_v = 1 + (1-1)\,p^{\text{new}}_v = 1$$

The learned parameters appear only inside $p^{\text{new}}_v$, whose coefficient is
**exactly zero**. Therefore

$$\frac{\partial \hat y^{\text{inf}}_v}{\partial \theta} = 0 \quad \forall \theta$$

The model cannot forget that a seeded node is infected, whatever training does.

> **Measured, under weight perturbation ×1 / ×50 / ×(−50) / ×0, on IC and LT, in
> distribution and on all six OOD targets:**
> `seeded_p_infected_worst` = **0.9999989**, `removed_p_frontier_worst` = **1.0e-6**.
> Both are the numerical clamp bounds. An *untrained* model scores identically.

Two consequences follow structurally rather than by fitting: **monotonicity**
(an infected node stays infected) and **self-termination** (a node with no active
in-neighbour has $p^{\text{new}} = 0$, so a free rollout cannot saturate).

**What the guarantee does not cover.** It fixes only the immediate exogenous
effect. The downstream propagation strength $q_{uv}$, the long-horizon effect of
an intervention, and any planner ranking built on them are learned. That is why
§5–§8 exist.

---

## 5. Q1 — Does the model use the action?

**Hypothesis.** A model that reads the action scores far below the "predicts no
action effect" null, and loses accuracy when actions are corrupted.

**Metric with a reachable null.** For same-state action pairs,

$$d_{\text{true}} = y_{MC}(s,a) - y_{MC}(s,a'), \qquad d_{\text{pred}} = f_\theta(s,a) - f_\theta(s,a')$$

$$\texttt{effect\_mae\_norm} = \frac{\mathbb{E}\lvert d_{\text{pred}} - d_{\text{true}}\rvert}{\mathbb{E}\lvert d_{\text{true}}\rvert}$$

A model predicting $d_{\text{pred}} \equiv 0$ scores **exactly 1.0**. The null is
not "greater than zero" — it is reachable, and two arms must reach it.

### 5.1 Result — four arms, four distributions

| distribution | arm | `delta_f1` | `effect_mae_norm` | pearson |
| --- | --- | --- | --- | --- |
| **BA-100** | full | 0.8559 | **0.5325** | 0.911 |
| | state_only | 0.1398 | **0.9998** | 0.014 |
| | constant | 0.0000 | **1.0000** | 0.000 |
| | shuffle | 0.2197 | — (drop **0.636**) | |
| **WS-100** | full | 0.9515 | **0.4764** | 0.926 |
| | state_only | 0.0097 | 0.9999 | 0.024 |
| | constant | 0.0000 | 1.0000 | 0.000 |
| | shuffle | 0.0538 | — (drop **0.898**) | |
| **SBM-100** | full | 0.8871 | **0.5045** | 0.919 |
| | state_only | 0.0958 | 0.9999 | 0.007 |
| | shuffle | 0.1063 | — (drop **0.781**) | |
| **BA-1000** | full | 0.8501 | 0.8676 | 0.667 |
| | state_only | 0.1276 | 1.0000 | 0.030 |
| | shuffle | 0.1754 | — (drop **0.675**) | |

`state_only` is **the same weights with the action channels zeroed**, not a
separately trained network, so it isolates the action's contribution rather than
confounding it with a different fit. `shuffle` permutes actions between records,
holding the action distribution fixed and destroying only the state–action
pairing.

**Conclusion.** The model uses the action. Both trivial arms land on the null to
four decimals on every distribution, so the test can fail and does not.
Zeroing the action channels costs 6.1× of `delta_f1`; shuffling costs 0.64–0.90.

**What this does not support.** Nothing about how much of the effect is
*correctly* predicted at scale — see §6.2.

---

## 6. Q2/Q3 — What the structure buys, and what survives distribution shift

### 6.1 Structured vs unconstrained head

Identical data, split, seed, training budget and evaluation suite. The linear
control is run at both `pos_weight` settings, because `off` was chosen *for* the
structured head and a control crippled by the treatment's hyperparameter is not a
control.

| arm | `delta_f1` | `ens_marg_mae` | **count bias** | model / true | `seeded_worst` | `removed_fr_worst` |
| --- | --- | --- | --- | --- | --- | --- |
| **structured IC** | 0.8559 | **0.0913** | **−0.197** | 45.2 / 45.7 | **0.9999989** | **1.0e-6** |
| linear IC | 0.8526 | 0.4661 | **+49.28** | **96.7** / 45.7 | 0.9908 | 0.0078 |
| **structured LT** | 0.6736 | 0.1538 | +15.13 | 81.6 / 60.8 | **0.9999989** | **1.0e-6** |
| linear LT (auto) | 0.6754 | 0.3806 | +40.46 | 98.4 / 60.8 | 0.9930 | 0.4410 |
| linear LT (off) | 0.6669 | **0.0884** | **+0.33** | 61.8 / 60.8 | 0.9962 | 0.2661 |

**One-step accuracy is tied** (0.8559 vs 0.8526). The structure buys nothing
there, as expected. It buys **rollout stability** — the linear IC head saturates,
predicting 96.7 of 100 nodes infected against a true 45.7 — and **exactness at
any parameter value**.

> **A trap worth naming.** The *thresholded* `add_seed_success` and
> `remove_frontier_success` are **1.0000 for every arm including linear**. Only
> the probability-level worst case separates them, and the rollout consumes
> probabilities. Reporting the thresholded version would have shown nothing.

**The LT anomaly, chased down.** `linear + pos_weight off` has the best LT rollout
calibration of any arm (+0.33). It achieves it by **violating monotonicity**:
measured on already-active nodes it predicts $P(\text{still infected}) = 0.91$ and
violates on **100%** of them, dropping ~9% of the infected set every step, which
cancels LT's over-prediction. Two errors cancelling, not calibration. The
structured head violates on **0%**, under ×1/×20/×(−20) perturbation.

**LT bias is partial observability, not fitting.** LT thresholds are $U(0,1)$, so
$P(\text{activate}\mid f_v) = f_v$ exactly and $p^{\text{new}} = f_v$ is the
Bayes-optimal *memoryless* predictor. On identical data:

| LT predictor | `ens_marg_mae` | count bias |
| --- | --- | --- |
| analytic memoryless optimum (zero learning) | 0.2302 | **+24.36** |
| **learned structured** | **0.1538** | **+15.13** |

The learned head **beats the memoryless floor**. LT is deterministic given
$\theta$ and $\theta$ *persists*: a memoryless predictor redraws $P(\theta<f)$
each step and ignores the evidence in earlier non-activations, so it
over-activates. The implied fix is state augmentation — carrying a posterior over
$\theta$ — not more fitting.

### 6.1.1 Multi-seed confirmation (paired, n = 5)

The single-run table above is one seed. Seeds 0–4 were trained per arm on
identical data and hyper-parameters, so **seed is a blocking factor**: the honest
comparison is the paired within-seed difference, not two independent means. All
intervals are 95% $t$-intervals, $\pm$ half-width.

| metric | structured | linear | paired diff | $t$ | 95% CI excl. 0 |
| --- | --- | --- | --- | --- | --- |
| `delta_f1` (one-step) | 0.8558 ± 0.0003 | 0.8512 ± 0.0008 | **+0.0046** | 12.65 | yes |
| `brier_infected` | 0.0013 ± 0.0000 | 0.0014 ± 0.0001 | −0.0002 | −4.87 | yes |
| `add_seed_success` | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 | — | no |
| `remove_frontier_success` | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 | — | no |
| `action_sensitivity` | 1.7126 ± 0.0177 | 1.9126 ± 0.0759 | −0.2000 | −7.83 | yes |
| **`ens_marg_mae`** (rollout) | **0.0948 ± 0.0017** | 0.4832 ± 0.0124 | **−0.3884** | **−85.93** | yes |
| **`ens_count_bias`** (rollout) | **−0.64 ± 1.05** | **+49.07 ± 1.12** | **−49.70** | **−88.13** | yes |
| `effect_mae_norm` | 0.5328 ± 0.0002 | 0.5412 ± 0.0060 | −0.0084 | −3.98 | yes |

**The separation is entirely in the rollout, and it is not marginal.** One-step
`delta_f1` differs by 0.0046 — statistically resolvable only *because* the pairing
removes seed variance, and practically negligible. Two rows down, the same models
differ by **49.7 nodes** of terminal count bias on a 100-node graph. A reader who
saw only the one-step row would conclude the heads are interchangeable.

> **Two ways to score 1.0000.** `add_seed_success` and
> `remove_frontier_success` are $1.0000 \pm 0.0000$ for **both** arms across all
> five seeds. For the structured head this is the algebraic identity of §4,
> which holds at *any* parameter value. For the linear head it is a **side effect
> of saturation**: a model that drives nearly every node toward infection marks
> seeded nodes infected for free. The same number, earned two different ways —
> which is why §4 is stated as a proof about the functional form and not as an
> empirical row. For the same reason the `action_sensitivity` row, on which the
> linear head scores *higher* (1.91 vs 1.71), carries no directional meaning: a
> saturating model is trivially sensitive.

---

### 6.2 Graph generalization

Frozen BA-100 checkpoint, no fine-tuning, six targets. Every target has
`split_mode=eval_only` (no train split exists) and is **structurally disjoint**
from the source: each graph hashed by (num_nodes, canonical sorted edge list),
zero collisions against any source split. Density matched (target mean degree
5.80–6.00 against source 5.82) so topology shift is not confounded with density.

| target | N | `delta_f1` | `ens_marg_mae` | count bias | model / true | **`effect_mae_norm`** |
| --- | --- | --- | --- | --- | --- | --- |
| BA-100 (control) | 100 | 0.8559 | 0.0913 | −0.20 | 45.2 / 45.7 | **0.5325** |
| BA-200 | 200 | 0.8529 | 0.0921 | −0.60 | 91.8 / 92.3 | 0.6346 |
| BA-500 | 500 | 0.8537 | 0.0950 | +1.19 | 238.4 / 236.1 | 0.7861 |
| BA-1000 | 1000 | 0.8501 | 0.1039 | +2.86 | 493.2 / 489.1 | **0.8676** |
| ER-100 | 100 | 0.8775 | 0.0849 | −1.92 | 35.7 / 38.1 | 0.5186 |
| WS-100 | 100 | 0.9515 | 0.0681 | −0.91 | 29.8 / 30.9 | **0.4764** |
| SBM-100 | 100 | 0.8871 | 0.0791 | −1.26 | 34.2 / 35.6 | 0.5045 |

**Prediction transfers.** Count bias stays within ±2.9 nodes on a 490-node
cascade at 10× the training size, frozen. $T_{\text{exo}}$ is exact everywhere.

**The action mechanism decays with SIZE, not TOPOLOGY.** `effect_mae_norm` runs
0.53 → 0.63 → 0.79 → 0.87 as $N$ goes 100 → 1000, while all three unseen
topologies at $N=100$ sit at 0.48–0.52, no worse than in distribution.

> **`delta_f1` is NOT comparable across topologies.** WS scores 0.9515 because WS
> cascades are intrinsically more predictable, not because transfer is better.
> Only a metric with a distribution-independent null — `effect_mae_norm` against
> 1.0 — is comparable across families, which is why the claim rests on that column.

**Why the decay?** The receptive-field hypothesis is **refuted**: at $n_{\text{layers}} \in \{1,2,3,8\}$ on BA-1000, `effect_mae_norm` is 0.8677 / 0.8677 / 0.8676 / 0.8681 — flat to four decimals. A 1-layer encoder matches an 8-layer one. Decomposing the ratio:

| N | $\mathbb{E}\lvert d_{\text{true}}\rvert$ | $\mathbb{E}\lvert\text{error}\rvert$ | `effect_mae_norm` | magnitude ratio |
| --- | --- | --- | --- | --- |
| 100 | 0.02253 | 0.01200 | 0.5325 | 0.549 |
| 1000 | 0.00861 | **0.00747** | 0.8676 | **0.155** |

The **absolute** counterfactual error *improves* with size (−38%). What grows is
**under-reaction**: predicted effect magnitude falls to 15.5% of true. The rising
ratio reflects that plus a shrinking per-node signal, not worse prediction. The
follow-up is effect-magnitude calibration, not capacity.

### 6.3 Privileged information — does it learn dynamics or read the answer?

The ablation is `--hide-edge-weights`: feed ones instead of $w_{uv}$, so the model
must infer transmission from structure and state alone. The question it settles is
the one a skeptic asks first — *is the world model learning propagation, or is it
simply reading the transmission probability off its own input?*

**The ablation was vacuous as originally configured, and the fix is part of the
result.** Under `--prob-model weighted` the generator sets

$$p(u \to v) \;=\; \frac{1}{\deg_{\text{in}}(v)}$$

**exactly** — measured correlation between $w_{uv}$ and $1/\deg_{\text{in}}(v)$ is
$1.000000$, maximum absolute error $9.93\times10^{-9}$ — while $\log(1+\deg)$ is
input channel 2. Masking $w$ therefore removes nothing the model cannot rebuild
from a channel it still has. The five-seed measurement says so unambiguously:

| metric (n = 5 seeds) | visible $w$ | hidden $w$ | paired diff | $t$ | excl. 0 |
| --- | --- | --- | --- | --- | --- |
| `delta_f1` | 0.8558 ± 0.0003 | 0.8559 ± 0.0001 | −0.0001 | −0.88 | **no** |
| `brier_infected` | 0.0013 ± 0.0000 | 0.0013 ± 0.0000 | +0.0000 | 2.65 | **no** |
| `action_sensitivity` | 1.7126 ± 0.0177 | 1.7162 ± 0.0203 | −0.0036 | −2.00 | **no** |
| `ens_marg_mae` | 0.0948 ± 0.0017 | 0.0946 ± 0.0017 | +0.0002 | 0.95 | **no** |
| `ens_count_bias` | −0.64 ± 1.05 | −0.24 ± 0.66 | −0.39 | −0.80 | **no** |
| `effect_mae_norm` | 0.5328 ± 0.0002 | 0.5327 ± 0.0002 | +0.0000 | 0.36 | **no** |

**Nine of nine metrics fail to separate.** This row is not a discarded experiment;
it is the **control that establishes the confound**, and without it the corrected
number below cannot be interpreted.

A `--prob-model random` mode was therefore added, drawing each edge's probability
i.i.d. $w_{uv} \sim U(0.02,\,0.4)$, which drops the correlation with
$1/\deg_{\text{in}}(v)$ from $1.0000$ to $0.0342$. Now the edge weight carries
information no other channel supplies. Rerunning the identical ablation on it
(three seeds, paired):

| metric (n = 3 seeds) | visible $w$ | hidden $w$ | paired diff | $t$ | excl. 0 |
| --- | --- | --- | --- | --- | --- |
| `delta_f1` | 0.8225 ± 0.0023 | **0.7125 ± 0.0001** | **+0.1100** | 197.72 | yes |
| `brier_infected` | 0.0014 ± 0.0000 | 0.0041 ± 0.0000 | −0.0027 | −260.70 | yes |
| `action_sensitivity` | 1.7223 ± 0.0203 | 1.6607 ± 0.0129 | +0.0616 | 8.04 | yes |
| `ens_marg_mae` | 0.0962 ± 0.0035 | 0.1187 ± 0.0035 | −0.0224 | −83.15 | yes |
| `ens_count_bias` | +1.24 ± 2.39 | +0.79 ± 0.39 | +0.45 | 0.71 | no |
| **`effect_mae_norm`** | **0.5635 ± 0.0017** | **0.6167 ± 0.0002** | **−0.0533** | −125.86 | yes |

**Both halves of the answer are required, and both hold.**

1. **The edge weight is genuinely used.** Hiding it costs 0.110 `delta_f1` and
   moves `effect_mae_norm` from 0.5635 to 0.6167. The earlier "hiding costs
   nothing" reading was an artefact of the generator, not a property of the model.

2. **The model is not merely reading the answer.** With the transmission
   probability entirely withheld, `effect_mae_norm` is **0.6167, still far below
   the reachable null of 1.0** — the value a model that predicts *no action
   effect* attains by construction (§5). Roughly **38% of the counterfactual
   action effect survives** with no access to $w$ at all. The degradation is
   real; the collapse is not.

> **Why the null matters here.** Against an unbounded error metric, 0.6167 would
> be uninterpretable. Against a null that a trivial model provably attains, it is
> a statement with content: the hidden-weight model is 38% of the way from
> "predicts nothing" to "predicts the effect exactly", using structure and state
> only.

| generator | $\mathrm{corr}(w,\,1/\deg_{\text{in}})$ | cost of hiding $w$ |
| --- | --- | --- |
| `weighted` (deterministic) | **1.000000** | none — 9/9 metrics not significant |
| `random` (i.i.d.) | 0.0342 | `delta_f1` −0.110, $t = 197.7$ |

The lesson generalises past this ablation: **an ablation is only as strong as the
independence of the thing it removes.** The first configuration removed a variable
that three other channels reconstructed, and reported a null result that meant
nothing.

---

## 7. Q5 — Cross-task reuse

### 7.1 Compatibility is derived, not declared

`registry/task_families.py` classifies every ordered task pair from properties
`pipeline/tasks.py` already owns — dynamics, competitive/epidemic layout,
`remove_semantics`, `action_ops`, and the load-bearing one, `gen_action_ops`
(what the generator actually injects mid-cascade). No rule is keyed on a task
name. Four levels:

| level | meaning | fine-tuning |
| --- | --- | --- |
| `EXACT_CHECKPOINT` | the frozen checkpoint loads and is used directly | forbidden |
| `FORWARD_DYNAMICS` | the frozen forward model is reused as an oracle; action-conditioning is never exercised | forbidden |
| `MECHANISM_ONLY` | some mechanism carries over; the checkpoint cannot load unchanged | n/a |
| `INCOMPATIBLE` | no shared process and no shared action semantics | n/a |

Two rules that tensor shape alone would miss: **same layout is not sufficient**
(IM and CND are both 6/2 on IC/LT and are `MECHANISM_ONLY`, because
$T_{\text{exo}}$ branches on `remove_semantics` inside the head), and **declaring
an action op is not exercising it** (source localization declares `add_node` but
its generator injects nothing after $t=0$).

### 7.2 EXACT_CHECKPOINT — IM → Adaptive Online IM

The source commits its whole seed set at $t=0$; the target commits a batch,
watches the cascade, and commits the next from the realised mid-cascade state.
The frozen model is asked a question its training distribution contained only as
uniformly-injected noise. Every arm is a policy
`(state, graph, batch) → seeds`, so only the scorer varies and NDlib rolls every
arm forward.

| arm | spread | span recovered | **trusted calls** |
| --- | --- | --- | --- |
| oracle (CRN greedy, MC-32) | **38.73** | 1.000 | **372,480** |
| **world model (frozen IM)** | **36.85** | **0.906** | **0** |
| degree | 35.93 | 0.860 | 0 |
| static_degree | 29.16 | 0.521 | 0 |
| random | 18.74 | 0.000 | 0 |

Paired over the same 15 graphs and campaigns:

| comparison | Δ | t (df 14) | separated |
| --- | --- | --- | --- |
| WM − random | +18.11 | 19.15 | **yes**, 15/15 graphs |
| WM − static_degree | +7.69 | 6.11 | **yes** |
| **oracle − WM** | +1.88 | **1.05** | **no** |
| WM − degree | +0.92 | 1.40 | no, 11/15 graphs |

**Conclusion.** The frozen IM checkpoint is statistically indistinguishable from a
Monte Carlo oracle that spent 372,480 simulator rollouts, while spending zero,
and it beats the non-adaptive control decisively — so it is exploiting the
adaptive structure rather than replaying a static rule. `frozen_parameters_unchanged`
verified tensor by tensor; `required_semantic_overrides = 0`.

**Not supported:** "beats degree". +0.92 at t = 1.40 is not separated.

### 7.3 FORWARD_DYNAMICS — IM → Cascade Reconstruction

Hidden timesteps are filled by rolling the frozen model forward from the last
observed anchor. Scoring is F1 over nodes that **changed**, with every arm handed
the true count and asked only *which* — otherwise persistence reaches ~0.95 on a
monotone cascade and nothing is distinguishable.

| arm | F1 | trusted calls |
| --- | --- | --- |
| **world model** | **0.1738** | **0** |
| oracle | 0.1807 | 4,640 |
| degree_growth | 0.0577 | 0 |
| persistence | **0.0000** | 0 |

The frozen model recovers **96.2%** of the oracle at zero simulator cost and beats
the structural baseline 3.0×. Persistence scoring exactly 0 confirms the scoring
restriction bites.

### 7.4 MECHANISM_ONLY — IM → Critical Node Detection

Same W1 layout, so the `state_dict` loads — but `remove_semantics` differs and
$T_{\text{exo}}$ branches on it. `load_checkpoint` **refuses** the override by
default; `strict_spec=False` performs it deliberately. Reused:
graph_backbone, edge_propensity, $T_{\text{endo}}$. Rebuilt: $T_{\text{exo}}$.

| arm | spread (lower better) | span closed | trusted |
| --- | --- | --- | --- |
| oracle | 15.01 | 1.000 | 64,800 |
| degree | 16.23 | 0.756 | 0 |
| **cnd_trained** (target-trained) | 17.05 | **0.594** | 0 |
| **im_transferred** | 17.93 | **0.420** | 0 |
| random | 17.74 | 0.458 | 0 |

**The transfer fails, exactly as classified.** The transferred model is no better
than random (0.420 vs 0.458) while the target-trained model reaches 0.594 — so it
is the *transfer* that fails, not the task that is unlearnable. One field,
`remove_semantics`, is the whole difference.

**A separate finding:** even the target-trained model loses to degree (0.594 vs
0.756), so this world model is not a strong containment planner on BA.

### 7.5 A task that is not solvable this way — IM → Source Localization

Across three settings the **true-dynamics oracle never beat random**:

| setting | $k/\lvert y\rvert$ | random | **oracle** | WM |
| --- | --- | --- | --- | --- |
| weighted, k=5% | 0.334 | 0.314 | 0.306 | 0.263 |
| weighted, k=1–3% (amplification ≥5×) | 0.16 | 0.069 | **0.028** | 0.111 |
| uniform 0.25, k=5% | 0.08 | 0.080 | **0.067** | 0.053 |

Weak diffusion makes a third of the observation *be* the answer; strong diffusion
floods the graph and erases it. A single IC realisation leaves the likelihood over
seed sets nearly flat, which is why the SL literature adds either multiple
observations or a learned source prior. Bare forward inversion has neither.

**The oracle failing is the proof.** This is reported as a property of the task
setup, not of the world model.

---

## 8. Q4 — Decision value

Three claims of increasing strength, each with its own null.

### 8.1 Ranking candidate algorithms

16 real IM algorithms per held-out graph, NDlib MC oracle at 32 runs as ground
truth. Ties are never counted as correct: a model tie on a separable pair is a
failure, and oracle-tied pairs are excluded rather than resolved.

| dataset | preference accuracy (null 0.5) | Kendall τ | model ties | **resolution** | **calls-to-first-win** | **trusted-call saving** |
| --- | --- | --- | --- | --- | --- | --- |
| BA-100 (source) | **0.392** | 0.293 | **42.7%** | **0.458** | 43 vs 45 | +4.4% |
| ER-100 (OOD) | 0.614 | 0.338 | 9.3% | 0.705 | 47 vs 53 | +11.3% |
| **SBM-100 (OOD)** | **0.704** | 0.448 | 3.3% | 0.869 | **30 vs 53** | **+43.4%** |
| **WS-100 (OOD)** | **0.722** | 0.473 | **2.4%** | **0.956** | **28 vs 55** | **+49.1%** |

"resolution" is predicted-spread sd over true-spread sd across the 15 non-trivial
candidates. **Resolution and preference accuracy are monotone in the same order.**

**The BA failure is candidate degeneracy, measured.** On a scale-free graph the
degree-like heuristics converge on the same hubs: 15 of 16 candidates have true
spreads within ~1.6 nodes against 2.2–2.8 elsewhere, and the model resolves only
46% of that already-small gap.

**Policy generalization holds.** Unseen-policy preference accuracy matches seen on
every dataset — 0.579/0.567 (ER), 0.725/0.644 (WS), 0.705/0.596 (SBM) — with the
trivially-separable `random_seeds` excluded from the seen side. An apparent gap of
0.590 vs 0.300 on BA was **entirely** that one candidate.

### 8.2 The closed loop under a fixed budget

The question in the form a practitioner asks it: *given B expensive evaluations,
does an agent that consults the world model walk away with a better algorithm?*
Identical K candidates; each agent orders them and spends B trusted evaluations on
its first B; its result is the best **true** value among those B. Only the ordering
differs. 15 graphs × 5 seeds = 75 paired samples.

| dataset | B=1 | B=2 | B=3 | B=4 |
| --- | --- | --- | --- | --- |
| **WS-100** | +1.59 (t=**5.75**) | +1.43 (t=**7.02**) | +0.98 (t=**4.66**) | +0.85 (t=**4.39**) |
| **SBM-100** | +1.18 (t=**3.27**) | +1.03 (t=**4.31**) | +0.83 (t=**3.67**) | +0.89 (t=**4.03**) |
| ER-100 | +0.15 (t=0.50) | +0.26 (t=0.98) | +0.01 (t=0.03) | +0.38 (t=1.61) |
| BA-100 | **−0.40 (t=−2.10)** | +0.06 (t=0.45) | +0.38 (t=2.37) | +0.51 (t=3.20) |

(Δ is world-model agent minus degree-prior agent, paired.)

**Budget equivalence, WS-100.** The world-model agent reaches with **1** trusted
evaluation what the degree agent needs **3–4** for — a 3–4× budget efficiency at
the tight end, converging as the budget grows, which is the shape the claim
predicts.

The pattern matches §8.1 exactly: strong where candidates are distinguishable
(WS, SBM), absent on ER, negative at B=1 on the degenerate BA. The same cause
explains both.

### 8.3 Trusted-call efficiency and wall-clock efficiency are different claims

They must not be merged. Measured end to end on WS-100:

```
trusted evaluation (32 MC)        73.8 ms
world-model scoring, 5 samples    30.0 ms per candidate
calls-to-first-win                1.87 (WM) vs 3.67 (random)

with WM:   16 × 30ms scoring + 1.87 × 74ms  =  618 ms
without:                3.67 × 74ms         =  271 ms
```

**Trusted calls fall 49.1%. Wall-clock rises 128%.** Saving 1.8 calls requires
scoring all 16 candidates first, and the model costs 40% of a trusted call.

The break-even is computable and belongs in the claim:

$$t_{\text{trusted}} \;>\; \frac{K \cdot t_{\text{model}}}{\text{calls}_{\text{random}} - \text{calls}_{\text{model}}} \;=\; \frac{16 \times 30\,\text{ms}}{1.80} \;=\; \mathbf{267\ ms}$$

NDlib on WS-100 costs 73.8 ms, so this synthetic benchmark sits 3.6× short. **The
threshold is not transferable** — every term moves with graph size — and is
recomputed per dataset; BA-1000 already costs 855 ms per trusted evaluation.

**Ensemble size is the cost lever**, and quality is nearly flat while cost is
super-linear:

| $n_{\text{samples}}$ | preference accuracy | cost | trusted-call saving | wall-clock |
| --- | --- | --- | --- | --- |
| 1 | 0.529 | 7.8 ms | 47.6% | **+12.5%** |
| 3 | 0.553 | 17.8 ms | 66.7% | −12.9% |
| **5** | 0.613 | 25.6 ms | **73.8%** | −40.8% |
| 10 | 0.677 | 49.7 ms | 66.7% | −155.9% |
| 20 | 0.700 | 114.7 ms | 61.9% | −451.7% |

(WS-100 **validation** graphs; the operating point is selected here and reported
on test.) Preference accuracy rises monotonically but the **trusted-call saving
does not** — it peaks at $n=5$. Ranking quality and decision value are not the
same objective, which is why §8.2 is the headline and not §8.1.

---

## 9. Measurement faults found and fixed

Each of these produced a plausible number that meant something other than it
appeared to. They are listed because several early conclusions rested on them.

| fault | effect | fix |
| --- | --- | --- |
| Per-episode splitting | 20/20 graphs straddled train/test | graph-disjoint default |
| Planning evaluator scored `list(store)[:n]` | "beats degree" 0.2437 vs 0.2687; corrected **0.2479 vs 0.2229**, direction reversed | test-split only + overlap provenance |
| LT frontier vs post-action state | `removed_p_frontier_worst` 0.92 | `active_pre` / `active_post` split |
| Bare `state_dict` checkpoints | a `spent` checkpoint loads under `blocked` with no error | self-describing `wm-ckpt-v2`, refuses mismatch |
| Torch thread thrashing | **40× slower** (80.0 vs 2.02 ms/record) | 1 thread by default |
| Mean-of-ratios call reduction | −0.366 against pooled **+0.044** | pooled estimator |
| Unshared candidate sets | made the **oracle the worst arm** | all arms rank the same shortlist |
| Oracle without common random numbers | oracle 34.00, below degree and the model | CRN → 38.73, a real ceiling |
| Oracle under-resourced (8 MC) | quantised marginals, oracle F1 **0.000** | 32–64 draws |
| Log-likelihood inversion objective | unreached nodes clamp to −13.8 and dominate | linear in/out-of-observation score |
| SL sources chosen by degree | degree baseline F1 **0.808** by construction | `--algorithms random` |
| SL amplification too low | $k/\lvert y\rvert = 0.334$; a third of the observation *is* the answer | require $\lvert y\rvert \ge 5k$ |
| **`hide-edge-weights` vacuous** | $w = 1/\text{in-degree}$ exactly ($r = 1.000000$), and degree is an input channel, so 9/9 metrics showed no effect | added `--prob-model random` ($r = 0.0342$); rerun gives `delta_f1` −0.110, $t = 197.7$ (§6.3) |

The pattern is consistent enough to state as two rules.

**An oracle that loses is a broken oracle, not a finding.** Four separate faults
were caught by exactly that check — every one of them presented as an interesting
negative result before it was traced.

**An ablation is only as strong as the independence of what it removes.** The
`hide-edge-weights` ablation ran to completion, across five seeds, and reported a
clean null — because the quantity it masked was reconstructible from a channel it
left in place. No amount of statistical care would have caught that; only
measuring $\mathrm{corr}(w,\,1/\deg_{\text{in}}) = 1.000000$ did.

---

## 10. What the evidence supports

The intended chain, with each link's status:

| # | claim | status |
| --- | --- | --- |
| 1 | The model genuinely responds to interventions | **established** — 0.53 against a reachable 1.0 null; both trivial arms reach it |
| 1b | It learns dynamics rather than reading them off privileged input | **established** — withholding $w$ entirely still leaves `effect_mae_norm` at 0.6167, far below the 1.0 null, while costing a significant 0.110 `delta_f1` ($t = 197.7$) |
| 2 | Structured transition guarantees correct intervention semantics | **established** — exact at any parameter value; over 5 paired seeds the unconstrained head saturates at **+49.07 ± 1.12** count bias against **−0.64 ± 1.05** ($t = -88.1$), while one-step `delta_f1` differs by only 0.0046 |
| 3 | Those dynamics survive graph and policy shift | **established for prediction**; the action mechanism attenuates with size, not topology |
| 4 | The same frozen model transfers between tasks sharing the world | **established at two levels** — oracle-parity on Adaptive IM, 96.2% of oracle on Reconstruction; fails at `MECHANISM_ONLY` exactly as classified |
| 5 | The resulting rankings reduce expensive evaluation | **established in trusted calls** (43–49%, and 3–4× budget efficiency in the closed loop); **not established in wall-clock** on this benchmark, with the break-even quantified |

**Boundaries.** Single graph family for training. IC is the vehicle; LT is weaker
and diagnosed. BA-100 is degenerate for ranking, which is also why no large-graph
wall-clock break-even is reported: the only large graphs available are BA, and a
break-even computed where the ranking does not separate has an unbounded and
uninterpretable denominator. Every claim above is IM-trained; five of the eight
registered tasks have no experiment. The privileged-information result rests on
three seeds, not five.

**The honest one-sentence version.** An action-conditioned graph world model with
a structured transition learns reusable dynamics that survive 10× size and three
unseen topologies frozen, transfers to a second task at oracle parity without
fine-tuning, and cuts the *number* of trusted evaluations an algorithm-design loop
needs by 43–49% — with the wall-clock benefit arriving only once a trusted
evaluation costs more than a few hundred milliseconds.
