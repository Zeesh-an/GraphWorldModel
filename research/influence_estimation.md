# Influence Spread Estimation — Prior Work, Datasets, and Published Results

Given a graph `G` and a seed set `S`, predict the expected spread `σ(S)` — the number of nodes eventually activated — **without** running Monte Carlo. This file covers exact/hardness results, bound- and sketch-based estimators, learned (GNN) estimators, and the other half of the problem: **where `p(u→v)` comes from**, i.e. influence-probability learning from observed cascades.

This is the one task in `research/` that costs us **nothing to enter**. Our `rollout_ensemble` evaluator (`world_model/wm_eval.py`) already reports `ens_final_count_model` against `ens_final_count_true` — that *is* influence spread estimation, reported under a different name (§2).

All URLs checked for HTTP status on **2026-07-28**.

---

## 0. Verification policy

| Tier | Meaning |
| ---- | ------- |
| **[verified]** | Read from the paper's own table via text extraction, or from the data repository's own statistics page. Trustworthy. |
| **[derived]** | Computed by us from a file we downloaded, or reconstructed arithmetically from published splits. |
| **[figure]** | Read off a plotted figure — the paper published no table. Approximate, direction only. |
| **[claim]** | Stated in prose by a paper or a secondary source; not cross-checked against a file or table. |

**Automated PDF summarizers hallucinate plausible numbers from these papers.** While compiling the IM file, a summarizer reported IRIE scoring `142.8` on NetHEPT at k=50; the published value in that paper's Table 3 is **724.67**. Extract text (`pdftotext -layout`), read the table, then transcribe. Never let a summary supply a number.

**Edge-count convention.** Undirected graphs are quoted as **undirected edges**; directed graphs as **arcs**. Our loaders report `adjacency.nnz`, which for a symmetrized undirected graph is **2× the undirected edge count**. This one convention difference explains most apparent "discrepancies" between our numbers and published tables — check it before concluding two graphs differ.

This convention bites *hard* in this literature specifically, because the two most-cited learned estimators both **append reverse edges before reporting**:

- GLIE's Table I lists GR Colab as `5,242 / 28,980` — that is ca-GrQc's 14,490 undirected edges stored as arcs, not a different graph [derived]. Same for its YouTube `1,134,891 / 5,975,246` = SNAP `com-Youtube` (2,987,624 edges) doubled [derived]. GLIE states the rule explicitly: *"we turn all undirected graphs into directed ones by appending reverse edges"* [verified].
- SKIM's Table 1 column is literally headed `|A|` (arcs): Orkut `234,370.2 ·10³` = 2 × `com-Orkut`'s 117,185,083 undirected edges [derived].

---

## 1. Task definition

**Influence estimation (IE).** Given `G = (V, E)`, a diffusion model `M` (IC or LT) with its parameters, and a seed set `S ⊆ V`, compute

```
σ(S) = E[ |{v : v activated at termination}| ]
```

Two strictly harder / finer variants matter for us:

| Variant | Output | Who studies it |
| ------- | ------ | -------------- |
| **Spread estimation** | one scalar `σ(S)` | GLIE, MONSTOR, SIEA, SKIM, ConTinEst |
| **Susceptibility estimation** | per-node vector `x_v = P(v activated)` | DeepIS, DySuse, MONSTOR (`π` vector), NMF |
| **Influence oracle** | `σ(S)` for *arbitrary, repeated* `S` after one preprocessing pass | SKIM, MONSTOR+, GLIE |

The susceptibility variant is the one our model natively produces — our targets `y_inf, y_fr` *are* per-node marginals, and `σ(S) = Σ_v x_v` recovers the scalar.

**Why the task exists at all: computing `σ(S)` exactly is #P-hard.**

| Model | Result | Source |
| ----- | ------ | ------ |
| **IC** | *"Theorem 1: Computing the influence spread σ_I(S) given a seed set S is #P-hard."* Reduction from counting s-t connectedness in a directed graph, which is #P-complete. | Chen, Wang & Wang, KDD 2010 [verified] |
| **LT** | *"computing the exact influence spread in the LT model is #P-hard, even if there is only one seed in the network."* Reduction uses interpolation; strictly more involved than the IC reduction. | Chen, Yuan & Zhang, ICDM 2010 [verified] |
| **LT on DAGs** | Linear time in graph size — the tractable island LDAG exploits. | Chen, Yuan & Zhang, ICDM 2010 [verified] |

Both papers frame this as closing an open question left by Kempe et al. (2003). The consequence everyone builds on: **`σ` must be approximated**, and the default approximation — 10,000 Monte Carlo rollouts per seed set — is what every method in §3 and §4 is trying to replace.

Chen et al. also note the approximation problem is itself hard: *"finding an efficient approximation algorithm for computing the probability of s-t connectivity is a long-standing open problem"* [verified].

---

## 2. Fit with our methodology — ✅ already computed

**Implementation cost: zero.** Not "small" — zero. The numbers this literature reports are the numbers `rollout_ensemble` already writes into every results JSON. What is missing is not code; it is the *framing* and the baseline table.

### 2.1 The mapping is exact

`world_model/wm_eval.py::rollout_ensemble` rolls `n_samples` trajectories, sampling each step from the predicted frontier marginal, and compares against `n_samples` true NDlib rollouts under the same recorded action sequence. Its return dict is, verbatim:

| our metric (`wm_eval.py`) | what the IE literature calls it |
| ------------------------- | ------------------------------- |
| `ens_final_count_model` | the estimate `σ̂(S)` |
| `ens_final_count_true` | the ground truth `σ(S)` (MC replay) |
| `ens_count_bias` | signed error `σ̂(S) − σ(S)`, averaged per step |
| `ens_count_w1` | Wasserstein-1 between the model's and the simulator's *distribution* of cascade sizes — strictly more than the field reports, which is almost always a point estimate |
| `ens_marg_mae` | **susceptibility MAE** — the DeepIS / DySuse metric exactly (per-node `|x̂_v − x_v|`) |

`ens_count_bias` is the load-bearing one and it is *signed*, where the field almost universally reports unsigned relative error. Signed is the better choice and we should say so: a saturating estimator and a collapsing estimator both score badly on MAE, but only the signed statistic tells you which failure you have. Our own history is the argument — the linear head's `count_bias ≈ +49` diagnosed runaway saturation that an MAE column would have merely called "large".

### 2.2 Reading `σ(S)` off the world model

Seed a state, run with `NULL` actions, read the final count:

```
s_0 = (infected = S, frontier = S);  a_t = NULL for all t
σ̂(S) = ens_final_count_model
```

That is the whole procedure. `rollout_ensemble` already accepts episodes whose action bag is empty, so a diffusion-only dataset (`--action-ops` omitted) is already an influence-estimation benchmark — we have simply never labelled it one.

Per-node susceptibility comes out one layer earlier: `model_marginal` inside `rollout_ensemble` is `x̂`, and `Σ_v x̂_v` is a lower-variance estimator of `σ(S)` than counting sampled activations. Worth switching to if we report `σ̂`.

### 2.3 The calibration angle — our stronger claim ⭐

This is the part worth writing a paper section about, because **GLIE, MONSTOR, and DeepIS are all black-box regressors on `σ` (or on `x`), and we are not.**

`ICTransmissionHead` does not regress the answer. It predicts the *mechanism* — a per-edge transmission propensity `q(u→v) = sigmoid(MLP([h_u, h_v, w_uv]))` — and then composes the IC form analytically: `p_new(v) = 1 − ∏_{u→v}(1 − q_uv · frontier_u)`. Three consequences the black-box line cannot claim:

1. **The estimator is structurally non-saturating.** A susceptible node with no active in-neighbour has `p_new = 0`, so the cascade self-terminates. GLIE instead leans on a *theoretical upper bound tightened by supervised training* — an upper bound has no such guarantee once you leave the training regime.
2. **The structural form is separately validatable.** `structured_oracle` (`ICTransmissionHead(oracle=True)`, `q = edge_weight`, no learning, exposed via `eval_structured_oracle.py`) isolates the question *"is the composition right?"* from *"did the network learn?"*. Expected `count_bias ≈ 0`. No published learned estimator in §4 ships this ablation, and it is cheap for us because the head is already written.
3. **`q` is directly comparable to the true `p(u→v)`.** That makes our model an *influence-probability estimator* (§4.3) as well as a spread estimator — the two halves of this file collapse into one head. `structured_residual` (`q = sigmoid(logit(w) + MLP(...))`) makes the comparison explicit: zero correction *is* the oracle.

### 2.4 What we would still have to build

Honest list — all small, none blocking:

| Gap | Cost |
| --- | ---- |
| Rank correlation (Spearman/Kendall) of `σ̂` across many seed sets | ~10 lines; `scipy.stats` is already imported in `wm_eval.py` for `wasserstein_distance` |
| MAPE / relative error, to speak the field's units | 1 line from `ens_count_bias` and `ens_final_count_true` |
| Wall-clock speedup vs the MC referee | we already run both sides in `rollout_ensemble`; just time them |
| Seed sets sampled at *IM-relevant* budgets (1/5/10/20% of N) rather than episode states | reuse `--budget-pcts` from `pipeline/run.py` |

The fourth is the only one with a research trap attached, and it is the trap in §8.4 — an estimator tuned on random seed sets can be useless for optimisation.

---

## 3. Classical and heuristic estimators

Four families. Only the first two produce a *reusable* estimator; RR-sets and snapshots are estimators that exist only inside an IM loop.

### 3.1 Hardness — the reason the rest of this section exists

| Result | Year | Venue | Statement | Paper | Code |
| ------ | ---- | ----- | --------- | ----- | ---- |
| **IC #P-hardness** | 2010 | KDD | Computing `σ_I(S)` is #P-hard (reduction from s-t connectedness counting) | [Chen, Wang & Wang — MSR-TR-2010-2](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/msr-tr-2010-2_v2.pdf) · [ACM](https://dl.acm.org/doi/10.1145/1835804.1835934) | — |
| **LT #P-hardness** | 2010 | ICDM | #P-hard **even with a single seed**; linear-time on DAGs | [Chen, Yuan & Zhang](http://snap.stanford.edu/class/cs224w-readings/chen10influence.pdf) · [IEEE](https://ieeexplore.ieee.org/document/5693962) | — |

### 3.2 Local-structure bounds (a tractable subgraph replaces the graph)

| Method | Year | Venue | Estimation idea | Paper | Code |
| ------ | ---- | ----- | --------------- | ----- | ---- |
| **MIA / PMIA** | 2010 | KDD | Restrict propagation to the **maximum-influence arborescence** per node — the single most probable path, thresholded at `θ`. `σ` becomes an exact linear-time recursion on a tree. | [MSR-TR-2010-2](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/msr-tr-2010-2_v2.pdf) | no public code found (third-party: [nd7141/influence-maximization](https://github.com/nd7141/influence-maximization)) |
| **LDAG** | 2010 | ICDM | LT counterpart: build a **local DAG** per node, where `σ` is linear-time exact (§3.1). | [PDF](http://snap.stanford.edu/class/cs224w-readings/chen10influence.pdf) | no public code found |
| **SIMPATH** | 2011 | ICDM | LT-specific: enumerate simple paths within a spread threshold `η`; `σ` = sum of path probabilities. Exploits LT's live-edge equivalence to simple paths. | [IEEE](https://ieeexplore.ieee.org/document/6137213) | no public code found |
| **IRIE** | 2012 | ICDM | Influence *ranking* by a belief-propagation-style linear system + influence estimation, no MC at all. | [arXiv 1111.4795](https://arxiv.org/abs/1111.4795) | no public code found |
| **UBLF** | 2013 | ICDM | Upper bound on `σ` from the spectral radius of `P`; prunes greedy. GLIE reports its bound is **violated by the weighted-cascade model** — *"its central condition is violated … the computed influence is exaggerated to the point it surpasses the nodes of the network"* [verified]. | [IEEE](https://ieeexplore.ieee.org/document/6729583) | no public code found |
| **DMP** | 2015 | — | Dynamic message passing: exact on trees, a good approximation on sparse loopy graphs. **GLIE's own estimation baseline.** | [Lokhov et al., PRE](https://arxiv.org/abs/1407.1255) | [mateuszwilinski/dynamic-message-passing](https://github.com/mateuszwilinski/dynamic-message-passing) |

### 3.3 Sketches and sampling with guarantees

| Method | Year | Venue | Estimation idea | Paper | Code |
| ------ | ---- | ----- | --------------- | ----- | ---- |
| **StaticGreedy** | 2013 | CIKM | Fix `R` live-edge **snapshots** once; `σ̂` = mean reachability over that fixed set. Removes the per-evaluation MC cost; ~2 orders of magnitude faster. | [arXiv 1212.4779](https://arxiv.org/abs/1212.4779) | no public code found |
| **RIS / RR-sets** | 2014 | SODA | Reverse reachable sets: `σ(S) ∝ ` fraction of random RR-sets that `S` intersects. Unbiased, and the foundation of TIM/IMM/SSA/OPIM. **Estimator only within an IM loop** — no cheap query for an arbitrary `S`. | [Borgs et al., arXiv 1212.0884](https://arxiv.org/abs/1212.0884) | — |
| **SKIM** ⭐ | 2014 | CIKM | Bottom-`k` min-hash **reachability sketches** over `ℓ` sampled instances, combined by Cohen's union-cardinality estimator. Gives a true **influence oracle**: linear preprocessing, then microsecond queries for *any* `S`. | [arXiv 1408.6282](https://arxiv.org/abs/1408.6282) | no public code found |
| **ConTinEst** | 2013 | NeurIPS | Continuous-time analogue: randomized least-label-list neighbourhood sketches over sampled transmission times. Estimates `σ(S, T)` for a time horizon `T`. | [NeurIPS 2013](https://papers.nips.cc/paper_files/paper/2013/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html) | no public code found |
| **INFEST** | 2015 | KDD | First `σ` estimator with a *rigorous* per-seed-set quality guarantee (Lucier, Oren & Singer). Distributed/MapReduce. Guaranteed relative error is loose in practice — 320% (§5.3). | [arXiv 1506.01188](https://arxiv.org/abs/1506.01188) | no public code found |
| **SIEA / SIEA-LT** ⭐ | 2017 | SIGMETRICS | *Outward influence* + a robust mean estimator (RSA) with adaptive stopping. **Currently the strongest non-learned estimator**: ≤1% avg relative error to 65M nodes / 1.8B edges (§5.3). | [arXiv 1704.04794](https://arxiv.org/abs/1704.04794) | no public code found |
| **Ohsaka et al.** | 2016 | VLDB | Dynamic index over RR-sets: maintains `σ` estimates under **edge insertions/deletions** — the closest classical analogue to our `add_edge` / `remove_edge` ops. | [PVLDB 9(12)](https://www.vldb.org/pvldb/vol9/p1077-ohsaka.pdf) | no public code found |

**Why SKIM and Ohsaka matter to us specifically.** SKIM is the only classical method that is an *oracle* in our sense — amortize once, then answer arbitrary queries — which is precisely the economics a learned world model claims. Ohsaka is the only one whose index survives **graph edits**, which is what our five action ops do to the graph; every other method in this section must be rebuilt from scratch after a single `remove_edge`. That is a concrete, defensible advantage for an action-conditioned model and nobody in §4 claims it either.

---

## 4. Learning-based methods

### 4.1 Neural spread / susceptibility estimators

| Method | Year | Venue | Approach | Output | Paper | Code |
| ------ | ---- | ----- | -------- | ------ | ----- | ---- |
| **Inf2vec** | 2018 | ICDE | Embed users from cascades + network context; influence = embedding similarity. Predecessor of the whole line. | scores | [IEEE](https://ieeexplore.ieee.org/document/8509310) | no public code found |
| **MONSTOR** ⭐ | 2020 | ASONAM | *"Monte Carlo simulator that is not a Monte Carlo simulator."* Stacks `s` copies of a **one-step** GCN: each copy maps `π_{i-1} → π_i` (per-node infection probabilities at step `i`), so the stack is an end-to-end IC simulation. **Inductive** — generalizes to unseen graphs. Trains against the analytic upper bound `u_i = π_{i-1} + (π_{i-1} − π_{i-2})P` rather than raw probabilities. | per-node `π` + `σ` | [arXiv 2001.08853](https://arxiv.org/abs/2001.08853) · [IEEE](https://ieeexplore.ieee.org/document/9381460) | [jihoonko/asonam20-monstor](https://github.com/jihoonko/asonam20-monstor) |
| **MONSTOR+** ⭐ | 2025 | DMKD | Journal extension: adds structural node features, a better pooling function, and **LT support**. Contains the only published **head-to-head against GLIE** (§5.2). | per-node `π` + `σ` | [DMKD 39:55](https://link.springer.com/article/10.1007/s10618-025-01137-z) · [PDF](http://dmlab.kaist.ac.kr/~kijungs/papers/monstorDAMI2025.pdf) | [jihoonko/asonam20-monstor](https://github.com/jihoonko/asonam20-monstor) (v1 only) |
| **DeepIS** | 2021 | WSDM | Two stages: coarse per-node susceptibility from features + NN, then a **propagation scheme** that aggregates neighbours' coarse estimates into a fine-grained one. Trained end-to-end. | per-node `x` | [ACM](https://dl.acm.org/doi/10.1145/3437963.3441829) | [xiawenwen49/DeepIS](https://github.com/xiawenwen49/DeepIS) |
| **GLIE** ⭐ | 2021→2023 | ASONAM | GNN whose architecture encodes a **theoretical upper bound on `σ`**, tightened by supervised regression on CELF-labelled seed sets. Trained on 100–500-node BA/Holme-Kim graphs, generalizes ~10× larger. Spawns `GLIE-CELF` (drop-in MC replacement) and `PUN` (submodular proxy for marginal gain). | scalar `σ` | [arXiv 2108.04623](https://arxiv.org/abs/2108.04623) · [ASONAM'23](https://dl.acm.org/doi/10.1145/3625007.3627293) · [SNAM'24](https://link.springer.com/article/10.1007/s13278-024-01311-z) | ✅ [geopanag/learn_im](https://github.com/geopanag/learn_im) (see §11) |
| **NMF (neural mean-field)** | 2021 | — | Mori–Zwanzig formalism → delay differential equation for `x(t)`, memory integral learned as time convolutions. Does **network inference and influence estimation jointly** from cascades only. | `x(t)` | [arXiv 2106.02608](https://arxiv.org/abs/2106.02608) | no public code found |
| **InfluLearner** | 2014 | NeurIPS | Learns the *coverage function* directly from cascades; must be re-run per time horizon. NMF's main baseline. | `σ(S,T)` | Du, Liang, Balcan & Song, NeurIPS 2014 — *no verified URL, see §11* | no public code found |
| **CoupledGNN** | 2020 | AAAI | Two coupled GNNs — one over node states, one over influence — to capture the interplay. Built for cascade popularity, adapted to susceptibility by DySuse. | cascade size | [arXiv 1906.09032](https://arxiv.org/abs/1906.09032) | no public code found |
| **DySuse** | 2023 | Expert Syst. Appl. | Susceptibility estimation on **dynamic** graphs: a structure-feature module per snapshot + a self-attention block across snapshots. Best variant `DySuseC` uses CoupledGNN. | per-node `x` | [arXiv 2308.10442](https://arxiv.org/abs/2308.10442) | no public code found |

**The two closest analogues to us are MONSTOR and DeepIS, not GLIE.** GLIE regresses a scalar; MONSTOR and DeepIS both predict the **per-node marginal vector**, which is what our heads output. MONSTOR is closest of all: its stacked one-step GCN is literally a teacher-forced one-step transition model unrolled — the same factorization we use, minus the action conditioning. Nobody in this table conditions on an intervention. That is our gap in the literature.

### 4.2 Where these are actually used

Every one of these is proposed as **an MC replacement inside CELF/greedy**, not as a standalone product: `GLIE-CELF`, `C-MON`/`U-MON` (MONSTOR + CELF/UBLF), `C-MON+`. The claimed win is always the same shape — near-identical seed sets, 2–4 orders of magnitude less time (§5).

### 4.3 Influence-probability learning — *where `p(u→v)` comes from*

Directly relevant: our `--prob-model weighted` (`p = 1/in-degree`) is an **assumption, not a measurement**. This literature is the measurement.

| Method | Year | Venue | Idea | Paper | Code |
| ------ | ---- | ----- | ---- | ----- | ---- |
| **Saito et al. (EM)** | 2008 | KES | Maximum-likelihood `p(u→v)` for IC via **Expectation-Maximization** over observed cascades. The origin of the problem. | [Springer](https://link.springer.com/chapter/10.1007/978-3-540-85567-5_9) | no public code found |
| **Goyal, Bonchi & Lakshmanan** ⭐ | 2010 | WSDM | The reference. From a social graph + an **action log** `(user, action, time)`, learn `p(v→u)` under three model families: **Static** (Bernoulli, Jaccard, Partial Credit), **Continuous-Time (CT)** with exponential decay, **Discrete-Time (DT)**. Also predicts *when* a user acts. | [WSDM'10 PDF](http://www.wsdm-conference.org/2010/proceedings/docs/p241.pdf) · [ACM](https://dl.acm.org/doi/10.1145/1718487.1718518) | no public code found |
| **Credit Distribution** | 2011 | VLDB | Skips `p(u→v)` entirely — assign **direct credit** for each propagation from the action log, and maximize the credit-based objective. Sidesteps the whole estimation problem. | [arXiv 1109.6886](https://arxiv.org/abs/1109.6886) | no public code found |
| **NetRate** | 2011 | ICML | Continuous-time: learn per-edge **transmission rates** `α_ij` by convex MLE over cascade timings. Infers structure and rates together. | [arXiv 1105.0697](https://arxiv.org/abs/1105.0697) | [Networks-Learning/netrate](https://github.com/Networks-Learning/netrate) |

Goyal et al.'s three families map cleanly onto MONSTOR's three activation probability matrices — **BT (Bernoulli Trial), JI (Jaccard Index), LP (Linear Probability)** [verified] — which is why MONSTOR reports every result three times. That is a better protocol than ours: we report one probability model and call it the answer.

**The honest position for our paper:** weighted-cascade `p = 1/in-degree` is the field's default (DeepIM, MOEIM, IMM, ToupleGDD all use it) and is defensible as a *convention*, but it is not learned from anything. `structured_residual` (§2.3) is the natural bridge — it learns a correction *on top of* the assumed `w`, so if we ever get logged trajectories, the same head absorbs them.

---

## 5. Published results

### 5.1 GLIE (arXiv 2108.04623) ⭐ the direct comparison

Protocol: train on 100 BA + Holme-Kim graphs of 100–200 nodes and 30 of 300–500 nodes (60/20/20 split); labels from **CELF with 1,000 MC ICs**, seed sets 1–5, storing the CELF optimum plus **30 random negatives per size** = 20,150 training samples. Weighted cascade `p = 1/deg(u)`. Undirected graphs made directed by appending reverse edges. [verified]

Datasets [verified, Table I]:

| Class | Graph | Nodes | Edges (arcs) |
| ----- | ----- | ----- | ------------ |
| Sim | Test/Train | 100 – 500 | 950 – 4,810 |
| Sim | Large | 1,000 – 2,000 | 11,066 – 19,076 |
| Small | Crime (CR) | 829 | 2,946 |
| Small | HI-II-14 (HI) | 4,165 | 26,172 |
| Small | GR Colab (GR) ✅ | 5,242 | 28,980 |
| Large | Enron (EN) | 33,697 | 361,622 |
| Large | Facebook (FB) | 63,393 | 1,633,660 |
| Large | Youtube (YT) ✅ | 1,134,891 | 5,975,246 |

**Influence estimation error.** "MAE divided by the average influence" (i.e. a normalized MAE / relative error), and time in seconds, over all seed set sizes and samples [verified, Table II]:

| Graph (seeds) | DMP MAE | DMP time | GLIE MAE | GLIE time |
| ------------- | ------- | -------- | -------- | --------- |
| Test (1–5) | 0.076 | 0.05 | **0.046** | **0.0042** |
| Large (1–5) | **0.086** | 0.44 | 0.102 | **0.0034** |
| CR (1–10) | **0.009** | 0.11 | 0.044 | **0.0029** |
| HI (1–10) | **0.041** | 2.84 | 0.056 | **0.0034** |
| GR (1–10) ✅ | 0.122 | 4.32 | **0.084** | **0.0042** |

Read this honestly: **GLIE is not uniformly more accurate than DMP** — it loses on Large, CR and HI. What it wins is time, by 10–1,000×, and it holds accuracy roughly constant as the graph grows where DMP degrades (0.009 → 0.122).

**Downstream IM at 20 seeds**, CELF driven by each estimator, evaluated with 10,000 MC ICs [verified, Table III]:

| Graph | Seed overlap | DMP-CELF infl | DMP-CELF time | GLIE-CELF infl | GLIE-CELF time |
| ----- | ------------ | ------------- | ------------- | -------------- | -------------- |
| CR (20) | 14 | 221 | 83 | **229** | **1.0** |
| HI (20) | 13 | 1,235 | 8,362 | **1,281** | **5.49** |
| GR (20) ✅ | 12 | 295 | 16,533 | **393** | **7.01** |

GR is the headline: **2,358× faster and 33% better spread**. The `seed overlap` column (12–14 of 20) is a rank-agreement statistic and is the seed of §8.4 — two estimators agreeing on only 60–70% of the chosen set still land within 4% spread on CR, but 33% apart on GR.

**Final IM quality at 200 seeds**, spread by 10,000 MC ICs [verified, Table IV]:

| Graph | GLIE-CELF | PUN | K-CORE | PMIA | DEGDISC | IMM | DEEPIS-CELF | FINDER |
| ----- | --------- | --- | ------ | ---- | ------- | --- | ----------- | ------ |
| CR | 661 | 657 | 647 | 656 | 644 | 650 | 501.61 ⚠ | 642 |
| GR ✅ | 1,617 | **1,626** | 701 | 1,566 | 1415 | 835.40 ⚠ | 1,617 | 1,286 |
| HI | 2,685 | **2,688** | 2,540 | 2,685 | 2,614 | 2,668 | 1602.5 ⚠ | 2,625 |
| EN | 17,601 | **17,614** | 13,015 | 17,534 | 16,500 | 17,497 | – | 17,244 |
| FB | 10,981 | 10,626 | 6,434 | 7,688 | 10,309 | **11,007** | – | 10,801 |
| YT ✅ | 246,439 | 244,579 | 110,409 | 242,057 | 236,726 | **247,178** | – | 50,435 |

⚠ **Three cells in GLIE's own Table IV are almost certainly runtimes, not spreads.** The paper's prose states DeepIS *"required 501.61, 835.4, and 1,602.5 seconds for the CR, GR, and HI datasets"* [verified] — and 501.61, 835.40 and 1602.5 are exactly the three anomalous cells. Verified against two independent `pdftotext` passes on page 7: the table really does print them. Do not quote those three cells. Logged in §11.

**Runtime, seconds** [verified, Tables V–VII]:

| Graph | GLIE-CELF | PUN | IMM | FINDER | DEGDISC | K-CORE | PMIA | PUN CPU (100 seeds) |
| ----- | --------- | --- | --- | ------ | ------- | ------ | ---- | ------------------- |
| CR | 2.00 | 0.25 | 0.19 | 0.41 | 0.21 | 0.06 | 0.04 | 0.17 |
| GR ✅ | 4.55 | 0.26 | 0.95 | 2.36 | 0.80 | 0.13 | 1.5 | 0.27 |
| HI | 2.19 | 0.27 | 1.29 | 1.01 | 1.36 | 0.14 | 0.12 | 0.20 |
| EN | 15.49 | 0.97 | 10.47 | 9.30 | 26.74 | 2.06 | 2.17 | 2.44 |
| FB | 287.7 | 3.1 | 171.25 | 56.80 | 22.77 | 9.29 | 10.62 | 17.5 |
| YT ✅ | 151.33 | 28.92 | 82.13 | 191.00 | 4006.29 | 54.38 | 74.91 | 97.5 |

GLIE's own conclusion: `GLIE-CELF` has the best quality but is *"quite slower"*; `PUN` is the accuracy-efficiency sweet spot at 3–60× faster than IMM [verified].

### 5.2 MONSTOR (ASONAM 2020) and MONSTOR+ (DMKD 2025) ⭐

Three Twitter-derived networks, trained on two and tested on the third [verified, Table 1]. Columns are `Σp(u,v)/|E|` under each activation-probability model, train / test:

| Graph | \|V\| | \|E\| | BT train/test | JI train/test | LP train/test |
| ----- | ----- | ----- | ------------- | ------------- | ------------- |
| Extended | 11,409 | 58,972 | 0.07974 / 0.09194 | 0.03345 / 0.04095 | 0.16138 / 0.18371 |
| WannaCry | 35,627 | 169,419 | 0.07255 / 0.09466 | 0.02977 / 0.04494 | 0.19785 / 0.16297 |
| Celebrity | 15,184 | 56,538 | 0.03206 / 0.02787 | 0.00163 / 0.00159 | 0.26142 / 0.256 |

⚠ MONSTOR+ reprints this table with WannaCry's LP column **swapped** (0.16297 / 0.19785) [verified, both papers]. One of the two is a transcription error; §11.

**MONSTOR estimation accuracy** [verified, Table 2]: Pearson and Spearman rank correlation with ground-truth influence are **1.000 in 51 of 54 cells**; the three exceptions are Celebrity (0.998, 0.999, 0.999). Blue rows = the test network was *not* in training, i.e. these are inductive.

**MONSTOR+ vs GLIE, head-to-head** — the single most useful table in this file, because it is the only published direct comparison of the two learned estimators [verified, MONSTOR+ Table 2]:

| Dataset | MON+ Pearson BT / JI / LP | MON+ Rank BT / JI / LP | GLIE Pearson BT / JI / LP | GLIE Rank BT / JI / LP |
| ------- | ------------------------- | ---------------------- | ------------------------- | ---------------------- |
| Extended | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.997 / 0.992 / 0.998 | 0.996 / 0.986 / 0.993 |
| WannaCry | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.999 / 0.989 / 0.997 | 0.998 / 0.983 / 0.994 |
| Celebrity | 0.995 / 1.000 / 1.000 | 0.998 / 1.000 / 0.999 | **0.872 / 0.745 / 0.989** | **0.884 / 0.767 / 0.937** |

MONSTOR+ additionally reports LT (GLIE is IC-only): Pearson 1.000 on all three, rank 0.989 / 0.955 / 0.995 [verified].

**Celebrity is where the two separate**, and it is the graph with the most extreme probability profile (JI ≈ 0.0016, LP ≈ 0.26). GLIE's Jaccard rank correlation collapses to 0.745/0.767 there. **A scalar-`σ` regressor degrades under probability-model shift; a per-node one does not.** That is the empirical argument for our per-node marginal targets, made by someone else's experiment.

**Runtime** [verified, MONSTOR Table 4]: 1,000 estimations, seconds, vs `|E|`:

| \|E\| | 2²⁰ | 2²¹ | 2²² | 2²³ | 2²⁴ | 2²⁵ | 2²⁶ |
| ----- | --- | --- | --- | --- | --- | --- | --- |
| MONSTOR | 11.5 | 17.7 | 31.0 | 56.3 | 108.9 | 411.0 | 819.7 |

MONSTOR+ splits this into one-off preprocessing and per-query cost [verified, Table 11] — the oracle economics of §3.3, restated for a neural model:

| \|E\| | 2²⁰ | 2²¹ | 2²² | 2²³ | 2²⁴ |
| ----- | --- | --- | --- | --- | --- |
| MON+ preprocessing (s, once) | 408.68 | 1629.10 | 5584.23 | 25669.33 | 135908.99 |
| MON+ estimation (ms, per seed set) | 56.87 | 104.30 | 179.06 | 311.59 | 638.63 |
| MONSTOR estimation (ms, per seed set) | 32.3 | 58.5 | 100.0 | 137.7 | 222.5 |

**Submodularity is preserved empirically but not guaranteed.** MONSTOR: the condition `f(S)+f(T) ≥ f(S∪T)+f(S∩T)` holds in ≥99.9% of pairs under BT/JI and ≥99.5% under LP, with MAPE on the violations of order `1e−7` to `2e−4` [verified, Tables 5–6]. MONSTOR+ is *worse* here — Celebrity drops to 0.904 (BT), **0.670 (JI)**, 0.959 (LP), 0.871 (LT), with MAPE up to 0.0239 [verified, Tables 12–13]. Relevant to us: greedy's (1−1/e) guarantee is void against a learned oracle, and this is how the field measures the damage.

### 5.3 SIEA / Outward Influence (arXiv 1704.04794) — the non-learned ceiling

Metric is **relative error against ground truth**, WC model, `|S| = 1` [verified, Table 4]. `MC10K` = 10,000 MC simulations, i.e. exactly the "ground truth" everyone else trusts:

| Dataset | SIEA avg / max % | MC10K avg / max % | INFEST avg / max % | SIEA time (s) | MC10K time (s) | INFEST time (s) |
| ------- | ---------------- | ----------------- | ------------------ | ------------- | -------------- | --------------- |
| NetHEP ✅ | **0.2 / 1.5** | 1.2 / 6.6 | 17.7 / 82.7 | 0.1 | 0.0 | 3417.6 |
| NetPHY ✅ | **0.1 / 0.6** | 0.4 / 5.3 | 22.9 / 43.0 | 0.1 | 0.0 | 8517.7 |
| Epinions | **0.9 / 5.2** | 5.3 / 19.7 | n/a | 0.2 | 0.0 | n/a |
| DBLP | **0.3 / 1.9** | 1.2 / 8.7 | n/a | 2.8 | 0.1 | n/a |
| Orkut | **0.5 / 3.2** | 3.0 / 16.0 | n/a | 54.2 | 2.9 | n/a |
| Twitter | **1.0 / 3.1** | 37.1 / 240.8 | n/a | 1272.3 | 7.9 | n/a |
| Friendster | **0.1 / 0.6** | 3.1 / 23.6 | n/a | 1510.1 | 2.8 | n/a |

**The single most important row in this file is Twitter**: `MC10K` has 37.1% average and **240.8% maximum** relative error. Ten thousand Monte Carlo runs — the field's standard ground truth, and ours — is *not* ground truth on a large graph with a single seed. Our `--mc-marginals 30` default is three orders of magnitude below that. See §8.3.

At `|S| = 5%|V|` the picture inverts: MC10K reaches 0.0–1.8% avg error and matches SIEA [verified, Table 5]. **Estimation is hard at small seed sets and easy at large ones**, which is the same budget effect the IM file records at 1% vs 20%.

LT, `|S| = 1` [verified, Table 6]: SIEA-LT 1.6/1.2/1.5/0.4/0.5/2.4/0.2% avg over the same seven graphs, vs MC10K 1.6/0.5/4.3/1.0/3.3/36.1/3.1%. Note MC10K *beats* SIEA-LT on NetHEP and NetPHY — LT is deterministic given thresholds, so sampling is cheaper there. Consistent with our own LT-vs-IC asymmetry.

### 5.4 SKIM influence oracle (arXiv 1408.6282)

`ℓ = 64` sampled instances, sketch size `k = 64`, IC-WC. Error is relative, averaged over 100 uniformly-sampled seed sets [verified, Table 3]:

| Instance | preproc (s) | space (MiB) | 1 seed: µs / err% | 50 seeds: µs / err% | 1000 seeds: µs / err% |
| -------- | ----------- | ----------- | ----------------- | ------------------- | --------------------- |
| AstroPh | 4 | 7.2 | 1.6 / 8.5 | 166.7 / 2.1 | 4,658.3 / 0.5 |
| Epinions | 10 | 37.1 | 1.3 / 5.2 | 155.0 / 3.4 | 5,011.1 / 1.1 |
| Slashdot | 20 | 37.8 | 1.5 / 6.0 | 155.2 / 3.9 | 4,982.3 / 1.0 |
| Gowalla | 46 | 96.0 | 1.5 / 7.3 | 179.8 / 3.2 | 5,275.6 / 1.1 |
| TwitterFollowers | 229 | 223.0 | 2.1 / 7.0 | 190.2 / 3.3 | 5,061.8 / 0.8 |
| LiveJournal | 2,064 | 2,367.0 | 2.0 / 7.1 | 189.6 / 3.0 | 5,168.3 / 0.9 |

**Error falls monotonically with seed set size** (8.5% → 2.1% → 0.5% on AstroPh) and query time is essentially graph-size-independent. This is the cleanest statement in the literature of the accuracy/`|S|` relationship, and it is the error profile any learned estimator should be plotted against.

### 5.5 DySuse (arXiv 2308.10442) — the susceptibility-MAE protocol

Ground truth = **1,000 MC simulations**, per-node average activation probability — i.e. our `next_marginal_infected` with `--mc-marginals 1000`. Metric = MAE of per-node susceptibility = our `ens_marg_mae`. IC, best and worst variants [verified, Table 2], seed size 250:

| Dataset | EvolveGCN | DySAT | TSGNet | DySuse-MONSTOR | DySuse-DeepIS | DySuse-CoupledGNN |
| ------- | --------- | ----- | ------ | -------------- | ------------- | ----------------- |
| BA | 0.390 | 0.227 | 0.242 | 0.234 | 0.065 | **0.067** |
| Epinions | 0.436 | 0.310 | 0.366 | 0.233 | 0.098 | **0.075** |
| Enron | 0.527 | 0.317 | 0.422 | 0.208 | 0.083 | **0.073** |
| Facebook ✅ | 0.603 | 0.216 | 0.296 | 0.233 | 0.072 | **0.029** |
| Digg | 0.560 | 0.301 | 0.342 | 0.260 | 0.070 | **0.050** |

LT on Enron at seed 250: 0.421 / 0.353 / 0.395 / 0.227 / 0.107 / **0.050** [verified, Table 3]. TR (trivalency) on Enron at 250: 0.566 / 0.403 / 0.513 / 0.274 / – / **0.017** [verified, Table 4].

**Calibrate our own number against this**: a susceptibility MAE of 0.03–0.10 is publishable SOTA on 4K–1M-node graphs. Our IC structured-oracle `ens_marg_mae` is ≈0.091 (`world_model/README.md`) — i.e. *the true-probability oracle itself* sits at the weak end of DySuse's learned range. That is a strong hint the two are not measuring the same thing (different seed sizes, different graphs, different step counts) and the comparison needs care before it goes in a paper.

---

## 6. Datasets

Authoritative metadata for most social graphs lives in [`influence_maximization.md`](influence_maximization.md) §6. This section owns only the graphs that are **specific to the estimation literature**, plus the forensics needed to compare against the tables in §5.

### 6.1 What we already load that this literature also uses

| `--dataset` | Nodes | Our edges | Type | Avg deg | Used by (as) | Source |
| ----------- | ----- | --------- | ---- | ------- | ------------ | ------ |
| `facebook` ✅ | 4,039 | 88,234 | Undirected | 43.7 | DySuse ("Facebook", edge count disputed — §6.3) | [SNAP ego-Facebook](https://snap.stanford.edu/data/ego-Facebook.html) |
| `ca_grqc` ✅ | 5,242 | 14,484 | Undirected | 5.5 | **GLIE ("GR Colab", 28,980 arcs)** | [SNAP ca-GrQc](https://snap.stanford.edu/data/ca-GrQc.html) |
| `nethept` ✅ | 15,229 | 62,752 arcs | Directed (both arcs) | 4.1 | **SIEA ("NetHEP", 15K/59K)** | [SparklyYS mirror](https://github.com/SparklyYS/Simultaneous-IMM) |
| `netphy` ✅ | 37,154 | 174,161 | Undirected | 9.4 | **SIEA ("NetPHY", 37K/181K)** | [Wei Chen `weic-graphdata.zip`](https://www.microsoft.com/en-us/research/people/weic/selected-projects/) |
| `youtube` ✅ | 1,134,890 | 2,987,624 | Undirected | 5.3 | **GLIE ("YT", 5,975,246 arcs)** | [SNAP com-Youtube](https://snap.stanford.edu/data/com-Youtube.html) |
| `digg` ✅ | 116,893 | ≈2.6M | Undirected | ≈45 | DySuse ("Digg", 1.1M/4.2M — different graph) | [Syracuse](https://datasets.syr.edu/datasets/Digg.html) |
| `cora_ml` ✅ | 2,810 | 7,981 | Undirected | 5.7 | **DeepIS** (ships `data/cora_ml.npz` in its own repo) | [graph2gauss](https://github.com/abojchevski/graph2gauss) |

**Four clean matches: `ca_grqc`, `nethept`, `netphy`, `youtube`.** These are the graphs where our numbers can be placed beside a published estimation result without a version caveat — GLIE on ca-GrQc and YouTube, SIEA on NetHEP and NetPHY. That is a better intersection than the IM file gets (which has only Jazz and Power Grid).

The `ca_grqc` match needs the arcs convention: GLIE's 28,980 = 2 × 14,490 [derived], ours is 14,484 undirected (12 fewer, self-loop handling). YouTube's 5,975,246 = 2 × 2,987,623 vs our 2,987,624 [derived] — a one-edge difference.

### 6.2 Graphs specific to this literature

| Dataset | Nodes | Edges | Type | Avg deg | Used by | Source |
| ------- | ----- | ----- | ---- | ------- | ------- | ------ |
| **Crime (CR)** | 829 | 2,946 arcs | bipartite→projected | 7.1 [derived] | GLIE | ⚠ no verified direct URL — guessed [Network Repository](https://networkrepository.com/) slugs 404; ships in [geopanag/learn_im](https://github.com/geopanag/learn_im) `data.zip` |
| **HI-II-14** | 4,165 | 26,172 arcs | protein interaction | 12.6 [derived] | GLIE | ⚠ same — no verified direct URL (§11) |
| **Enron (GLIE)** | 33,697 | 361,622 arcs | Undirected email | 10.7 [derived] | GLIE | [SNAP email-Enron](https://snap.stanford.edu/data/email-Enron.html) — this is the **LCC** (33,696), not the raw 36,692 |
| **Facebook (GLIE)** | 63,393 | 1,633,660 arcs | Undirected | 25.8 [derived] | GLIE | ⚠ **not** ego-Facebook; ≈ the Viswanath/KONECT Facebook wall graph. Unresolved — §11 |
| **Extended** | 11,409 | 58,972 | Twitter cascade graph | 5.2 [derived] | MONSTOR(+) | Liu et al. 2019 via [asonam20-monstor](https://github.com/jihoonko/asonam20-monstor) |
| **WannaCry** | 35,627 | 169,419 | Twitter cascade graph | 4.8 [derived] | MONSTOR(+) | same |
| **Celebrity** | 15,184 | 56,538 | Twitter cascade graph | 3.7 [derived] | MONSTOR(+) | same |
| **AstroPh** | 14,800 | 239,300 arcs | Undirected coauthorship | 16.2 [derived] | SKIM | [SNAP ca-AstroPh](https://snap.stanford.edu/data/ca-AstroPh.html) |
| **TwitterFollowers** | 456,600 | 14,855,900 arcs | Directed | 32.5 [derived] | SKIM | [SNAP higgs-twitter](https://snap.stanford.edu/data/higgs-twitter.html) |
| **Slovakia (sk-2005)** | 50,636,200 | 1,930,292,900 arcs | Directed web | 38.1 [derived] | SKIM | [LAW sk-2005](https://law.di.unimi.it/webdata/sk-2005/) — ⚠ did not resolve on 2026-07-28 |
| **Flickr (action log)** | 1,300,000 | 40,000,000 | Undirected + 35M action tuples over 300K actions | 61.5 [derived] | Goyal et al. WSDM'10 | not publicly redistributed [claim] |
| **Citeseer / Pubmed / MS-Academic** | — | — | citation | — | DeepIS (`data/*.npz`) | [xiawenwen49/DeepIS](https://github.com/xiawenwen49/DeepIS) |

**MONSTOR's three graphs are the closest thing this field has to a canonical estimation benchmark** — every MONSTOR/MONSTOR+ number and the only GLIE head-to-head live on them, and the repo ships them. They are 11K–36K nodes, well inside what our NDlib pipeline simulates. Adding them is a `data/datasets/*.py` copy-and-edit (the three-step contract in [`influence_maximization.md`](influence_maximization.md) §6.4) and it is the single highest-value dataset addition this review found.

### 6.3 Dataset forensics — do not quote these without checking

| Name | The problem | Status |
| ---- | ----------- | ------ |
| **DySuse "Enron"** 1.9K / 2.3M | 1,900 nodes cannot carry 2.3M undirected edges (max ≈1.8M). SNAP email-Enron is 36,692 / 183,831. | ❌ internally impossible |
| **DySuse "Facebook"** 4.0K / 8.8M | Node count matches ego-Facebook (4,039) exactly; 8.8M vs the true 88,234 is a **×100 unit error** ("8.8M" for "88.2K"). | ❌ edge count wrong, graph identifiable |
| **DySuse "Epinions"** 1.2K / 5.0K | Neither soc-Epinions1 (75,879) nor signed (131,828). A sampled subgraph, unstated. | ❌ unidentified |
| **SIEA "Epinions"** 75K / 841K, avg deg 13.4 | 75K nodes is soc-Epinions1; 841K edges is **signed** epinions. Their own avg degree 13.4 = 2×508,837/75,879 [derived], which matches the *node* count, not the edge count. | ⚠ edge cell belongs to the other version |
| **GLIE "Facebook"** 63,393 | Not SNAP ego-Facebook (4,039). Not stated which. | ❓ unresolved |
| **MONSTOR vs MONSTOR+ WannaCry LP** | 0.19785/0.16297 vs 0.16297/0.19785 — train and test swapped between the two papers. | ⚠ one is a typo |

DySuse also states its snapshots are built by *"randomly delete 0∼2‰ of nodes and 0∼2‰ of edges at each timestamp"* [verified] — its graphs are perturbed subgraphs, not the named originals, which explains but does not excuse the table.

---

## 7. Which paper uses which

Cells mark the dataset **as that paper reports it** — check §6.3 first.

| Dataset | PMIA'10 | IRIE'12 | SKIM'14 | SIEA'17 | MONSTOR'20 | DeepIS'21 | GLIE'21/23 | DySuse'23 | MONSTOR+'25 |
| ------- | ------- | ------- | ------- | ------- | ---------- | --------- | ---------- | --------- | ----------- |
| **NetHEPT** ✅ | ✔ | ✔ | | ✔ | | | | | |
| **NetPHY** ✅ | ✔ | ✔ | | ✔ | | | | | |
| **ca-GrQc** ✅ | | | | | | | ✔ | | |
| **YouTube** ✅ | | | | | | | ✔ | | |
| **Cora-ML** ✅ | | | | | | ✔ | | | |
| **Facebook** ✅ | | | | | | | | ✔ (disputed) | |
| **Digg** ✅ | | | | | | | | ✔ (different) | |
| Crime, HI-II-14 | | | | | | | ✔ | | |
| Enron | | | | | | | ✔ | ✔ | |
| Epinions | ✔ | ✔ | ✔ | ✔ | | | | ✔ | |
| AstroPh | | | ✔ | | | | | | |
| Slashdot | | ✔ | ✔ | | | | | | |
| Gowalla | | | ✔ | | | | | | |
| TwitterFollowers | | | ✔ | | | | | | |
| DBLP | ✔ | ✔ | | ✔ | | | | | |
| LiveJournal | | ✔ | ✔ | | | | | | |
| Orkut | | | ✔ | ✔ | | | | | |
| Friendster | | | ✔ | ✔ | | | | | |
| Twitter (Kwak) | | | ✔ | ✔ | | | | | |
| Slovakia | | | ✔ | | | | | | |
| Extended / WannaCry / Celebrity | | | | | ✔ | | ✔ (via MON+) | | ✔ |
| Citeseer / Pubmed / MS-Academic | | | | | | ✔ | | | |
| Synthetic BA / Holme-Kim | | | | ✔ | | | ✔ | ✔ | |

**Bold = we already load it.** Seven rows, versus four in the IM file — this task has the better dataset overlap of the two.

---

## 8. Evaluation protocol

### 8.1 The metrics the field uses, and what we already compute

| Field metric | Definition | Reported by | Our equivalent |
| ------------ | ---------- | ----------- | -------------- |
| **Relative error** | `\|σ̂ − σ\| / σ`, avg and **max** | SIEA, SKIM | `ens_count_bias / ens_final_count_true` — 1 line |
| **Normalized MAE** | MAE ÷ average influence | GLIE | same |
| **MAPE** | mean abs. % error | MONSTOR(+) | same |
| **RMSE** | on `σ` and on `π` | MONSTOR+ | not computed; trivial |
| **Susceptibility MAE** | mean `\|x̂_v − x_v\|` over nodes | DeepIS, DySuse | ✅ **`ens_marg_mae`** — identical |
| **Pearson / Spearman** | correlation of `σ̂` vs `σ` over many seed sets | MONSTOR(+) | ❌ **missing** — the one real gap (§2.4) |
| **Seed overlap** | \|chosen ∩ reference\| at fixed k | GLIE | ❌ missing; cheap |
| **Downstream spread** | MC spread of seeds chosen using `σ̂` | GLIE, MONSTOR(+) | ✅ `plan_regret_model` measures the same thing as regret |
| **Speedup vs MC** | wall-clock ratio | all | ❌ not timed, though both sides already run |
| **Distributional** | Wasserstein-1 over cascade sizes | **nobody** | ✅ `ens_count_w1` — we report strictly more |

Two asymmetries worth stating in a paper: we report a **signed** bias where the field reports unsigned error, and a **distributional** distance where the field reports a point estimate. Both are defensible upgrades, and both are already in `rollout_ensemble`'s return dict.

### 8.2 Protocol conventions

- **Ground truth is 10,000 MC ICs** — GLIE, SIEA and DySuse (1,000) all use MC as the reference. Our `--mc-marginals` default is **30**. §8.3.
- **Seed set sizes.** GLIE trains on 1–5 and tests to 10, then evaluates IM at 20, 100 and 200. SKIM sweeps 1 / 50 / 1000. SIEA reports `|S| = 1` and `|S| = 5%|V|`. MONSTOR samples `|S|` uniformly in `[1, |V|/50]`.
- **Negative sampling matters.** GLIE trains on 30 random seed sets **plus the CELF optimum** per size, because random-only supervision cannot reach optimal values (§8.4). MONSTOR uses half uniform-random, half degree-proportional.
- **Probability models.** MONSTOR reports every result three times (BT / JI / LP, §4.3). SIEA reports WC and LT. We report one.

### 8.3 The trap we are already standing in: MC is not ground truth

SIEA's Table 4 (§5.3) shows `MC10K` at **37.1% average / 240.8% maximum** relative error on Twitter with a single seed. Two consequences for us:

1. **Our `ens_final_count_true` is itself an estimate**, computed from `n_samples=20` simulator rollouts by default. On a large sparse graph with a small frontier, 20 draws is nowhere near enough for `ens_count_bias` to mean what its name says. On BA-100 it is fine; on `nethept` it is not.
2. **`--mc-marginals 30`** sets the *training target* quality. SIEA's numbers say the error is worst exactly where our cascades start — small active sets. Before we publish an estimation table, sweep `--mc-marginals` and show the metric has converged.

This is a cheap, honest experiment nobody in §4 ran, and it doubles as a robustness result.

### 8.4 The ranking trap — accuracy on random seed sets ≠ usefulness ⭐

**An estimator can be accurate everywhere that does not matter.** IM does not need `σ̂ ≈ σ`; it needs the *argmax* to be right. Those come apart because random seed sets and near-optimal seed sets occupy different parts of the range.

The literature states this directly. GLIE, on why random supervision is insufficient [verified]:

> *"In IM however, the difference in σ between an average seed set and the optimal can be significant, hence training solely on the random sets would render our model unable to predict larger values that correspond to the optimum. That is why we added the aforementioned samples of the optimum seed set computed using CELF."*

Three pieces of corroborating evidence in §5:

- **GLIE Table III**: `seed overlap` is only 12–14 of 20 against DMP-CELF, yet spread differs by 4% on CR and 33% on GR. Rank agreement and outcome agreement are loosely coupled.
- **MONSTOR+ Table 2**: GLIE's Pearson on Celebrity-JI is 0.745 while MONSTOR+ is 1.000 — but both still drive CELF to usable seed sets. High correlation is not necessary; correct *local* ordering is.
- **MONSTOR Tables 5–6 / MONSTOR+ Tables 12–13**: submodularity fails on 0.1–33% of pairs. Greedy's guarantee is gone, and the failures are exactly where the ordering is wrong.

**What this means for our evaluation.** `ens_count_bias` measured over test *episodes* samples the state distribution our simulator produced — mostly mid-cascade states from spine selectors, not candidate seed sets near the IM optimum. A good `ens_count_bias` therefore does **not** license a claim about planning. `plan_regret_model` is the metric that does, and we already have it. Report both, and never let the first stand in for the second.

The mirror-image warning is already recorded in [`influence_maximization.md`](influence_maximization.md): IMINFECTOR found IMM underperforming on real cascade traces because it optimizes against a simulator. Our estimator is trained against NDlib. Same class of assumption.

---

## 9. Implications for this project

Ordered by value per hour. The first three are hours, not days.

1. **Relabel what we already print.** `ens_final_count_model` vs `ens_final_count_true` is an influence-estimation result. Add relative error and MAPE (one line each, §8.1) and the table speaks the field's units. This is the entire cost of entering this task.

2. **Add Spearman/Kendall over seed sets.** The only genuinely missing metric, and the one MONSTOR+ uses for its GLIE comparison (§5.2). `scipy.stats` is already imported in `wm_eval.py`. Without it we cannot be placed on the same axis as the two closest analogues.

3. **Time both sides of `rollout_ensemble`.** It already runs the model ensemble and the true MC ensemble in the same function. A speedup number is a stopwatch, and speedup is the headline claim of every method in §4.

4. **Run `structured_oracle` and report `count_bias ≈ 0` as a validation of the structural form** (§2.3). No published learned estimator ships this ablation. It costs one `eval_structured_oracle.py` invocation and it is the strongest differentiator we have against a black-box σ regressor.

5. **Add MONSTOR's three graphs** (Extended, WannaCry, Celebrity — §6.2). They are 11K–36K nodes, the repo ships them, and they are the only benchmark where *both* MONSTOR+ and GLIE have published numbers. Highest-value dataset addition in this review.

6. **Sweep `--mc-marginals` and `n_samples` and show convergence** (§8.3). SIEA's Twitter row proves MC10K can be 240% off; our defaults are 30 and 20. This is a cheap robustness experiment that also protects every other number we report.

7. **Frame the action conditioning as the contribution.** Every estimator in §3 and §4 answers `σ(S)` on a *fixed* graph. Ours answers `σ(s_t, a_t)` — spread after an intervention — and Ohsaka et al. (§3.3) is the only prior work that even maintains an index across edge edits. "Influence estimation under interventions" is an unoccupied slot.

8. **Do not claim planning quality from estimation quality.** §8.4. Report `plan_regret_model` alongside, always.

**What not to do:** do not build a separate σ-regressor to compete with GLIE on its own terms. We would be a worse GLIE. The per-node marginal + mechanism head is the differentiated position, and MONSTOR+'s Celebrity result (§5.2) is published evidence that the per-node formulation is the more robust one.

---

## 10. Reference list

**Hardness** [Chen, Wang & Wang, KDD'10 (MSR-TR-2010-2)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/msr-tr-2010-2_v2.pdf) · [Chen, Yuan & Zhang, ICDM'10 (LT)](http://snap.stanford.edu/class/cs224w-readings/chen10influence.pdf)

**Bounds and sketches** [IRIE (arXiv 1111.4795)](https://arxiv.org/abs/1111.4795) · [StaticGreedy (arXiv 1212.4779)](https://arxiv.org/abs/1212.4779) · [RIS / Borgs (arXiv 1212.0884)](https://arxiv.org/abs/1212.0884) · [SKIM (arXiv 1408.6282)](https://arxiv.org/abs/1408.6282) · [ConTinEst (NeurIPS'13)](https://papers.nips.cc/paper_files/paper/2013/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html) · [INFEST / Lucier, Oren & Singer (arXiv 1506.01188)](https://arxiv.org/abs/1506.01188) · [Ohsaka et al., PVLDB 9(12)](https://www.vldb.org/pvldb/vol9/p1077-ohsaka.pdf) · [SIEA / Outward Influence (arXiv 1704.04794)](https://arxiv.org/abs/1704.04794) · [DMP (arXiv 1407.1255)](https://arxiv.org/abs/1407.1255) · [code](https://github.com/mateuszwilinski/dynamic-message-passing)

**Learned estimators** [MONSTOR (arXiv 2001.08853)](https://arxiv.org/abs/2001.08853) · [code](https://github.com/jihoonko/asonam20-monstor) · [MONSTOR+ (DMKD 2025)](https://link.springer.com/article/10.1007/s10618-025-01137-z) · [PDF](http://dmlab.kaist.ac.kr/~kijungs/papers/monstorDAMI2025.pdf) · [DeepIS (WSDM'21)](https://dl.acm.org/doi/10.1145/3437963.3441829) · [code](https://github.com/xiawenwen49/DeepIS) · [GLIE (arXiv 2108.04623)](https://arxiv.org/abs/2108.04623) · [ASONAM'23](https://dl.acm.org/doi/10.1145/3625007.3627293) · [SNAM'24](https://link.springer.com/article/10.1007/s13278-024-01311-z) · [code](https://github.com/geopanag/learn_im) · [NMF (arXiv 2106.02608)](https://arxiv.org/abs/2106.02608) · [DySuse (arXiv 2308.10442)](https://arxiv.org/abs/2308.10442) · [CoupledGNN (arXiv 1906.09032)](https://arxiv.org/abs/1906.09032) · [Inf2vec (ICDE'18)](https://ieeexplore.ieee.org/document/8509310)

**Influence-probability learning** [Goyal, Bonchi & Lakshmanan, WSDM'10](http://www.wsdm-conference.org/2010/proceedings/docs/p241.pdf) · [ACM](https://dl.acm.org/doi/10.1145/1718487.1718518) · [Saito et al., KES'08](https://link.springer.com/chapter/10.1007/978-3-540-85567-5_9) · [Credit Distribution (arXiv 1109.6886)](https://arxiv.org/abs/1109.6886) · [NetRate (arXiv 1105.0697)](https://arxiv.org/abs/1105.0697) · [code](https://github.com/Networks-Learning/netrate)

**Our own** `world_model/wm_eval.py::rollout_ensemble` · `world_model/eval_structured_oracle.py` · [`influence_maximization.md`](influence_maximization.md)

---

## 11. Open gaps

Honest list of what this review could **not** establish.

- **GLIE's code — RESOLVED, and the IM file's note is now out of date.** The repo previously cited 404s, and the arXiv PDF only says *"the source code can be found in the supplementary files"* [verified]. But the code **is** public: **[github.com/geopanag/learn_im](https://github.com/geopanag/learn_im)** (HTTP 200, last pushed 2024-07-18, 8 stars). Its `readme.md` opens *"The code to reproduce the analysis for 'Maximizing Influence with Graph Neural Networks'"* and ships `train_glie.py`, `celf_glie.py`, `pun.py`, `qnet_im.py` [verified]. `geopanag/glie`, `geopanag/GLIE` and `geopanag/pun` all 404 — those are the dead links. Update `influence_maximization.md`.
- **Three cells of GLIE's Table IV are runtimes, not spreads** (CR/DEEPIS-CELF 501.61, GR/IMM 835.40, HI/DEEPIS-CELF 1602.5). Confirmed present in the PDF by two extraction passes and matched to a prose sentence listing those exact three numbers as seconds. The paper's true IMM-on-GR spread is unknown.
- **DeepIS's own result tables were not transcribed.** The paper is paywalled (`dl.acm.org` returns 403 to automated fetch); no arXiv or author-hosted PDF found. Its headline claims — *5× smaller estimation error than SOTA GNN approaches, 2 orders of magnitude faster than MC* — are **[claim]**, from the abstract via secondary sources. The only [verified] DeepIS numbers here are GLIE's Table IV column, which is itself suspect (above).
- **ConTinEst, SIMPATH, LDAG, UBLF, Inf2vec, Saito et al., Goyal et al. result tables** — methods identified and linked, cells not transcribed. Goyal et al. publish ROC curves rather than tables, so their comparison of Static vs CT vs DT models is **[figure]**/**[claim]** only; their stated finding is that the **continuous-time model performs best** and **Bernoulli slightly beats Jaccard** [claim].
- **InfluLearner has no verified URL here.** The NeurIPS 2014 proceedings hash was not confirmed, and a guessed hash was removed rather than left in. Cite by title (Du, Liang, Balcan & Song, NeurIPS 2014) until checked.
- **URL verification results (2026-07-28).** Everything in §10 returned HTTP 200 except: `dl.acm.org` and `doi.org` return **403** and `ieeexplore.ieee.org` returns **202** to automated fetch — these are bot challenges, not dead links, and all resolve in a browser. `law.di.unimi.it/webdata/sk-2005/` did not respond at all. Four URLs were **found broken and fixed**: the ConTinEst NeurIPS hash (a guessed hash 404'd; the real one is `8fb21ee7a2207526da55a679f0332de2`, confirmed from the NeurIPS 2013 index), the ConTinEst code repo (`Networks-Learning/influence-estimation-and-maximization` 404s — downgraded to "no public code found"), and the two Network Repository slugs for Crime and HI-II-14 (both 404; the site root is up, so the slugs are wrong — the data ships in GLIE's repo instead).
- **GLIE's "Facebook" (63,393 nodes)** is not ego-Facebook and the paper does not say which graph it is (§6.3).
- **MONSTOR's Extended / WannaCry / Celebrity provenance** traces to Liu et al. 2019 via the repo; the original collection paper was not read, and the exact crawl for Extended is described only as *"we crawled more tweets and retweets in addition to those used in (Sabottke et al. 2015)"* [verified].
- **DySuse's Table 1 is not usable** — two of five edge counts are internally impossible or off by ×100 (§6.3). Its *result* tables are [verified]; its dataset table is not.
- **No published estimator conditions on an intervention.** The claim in §9.7 that "influence estimation under interventions" is unoccupied is an absence of evidence from this review, not a proven absence. Ohsaka et al. (dynamic index under edge edits) is the nearest neighbour found.
- **No estimation paper reports Jazz, Power Grid, NetScience, Cora-ML (as a diffusion graph), wiki-Vote, LastFM or email-Eu-core.** Our overlap with this literature is the seven rows in §7 and no more.
