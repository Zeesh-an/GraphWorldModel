# Cascade / Popularity Prediction — Prior Work, Datasets, and Published Results

Given the early portion of a **real observed** information cascade — its first `t` minutes/hours/years, or its first `k` adopters — predict its eventual size, its growth curve, or which specific node adopts next. This is the one task in this folder whose literature refuses to use a simulator: it is evaluated on Weibo retweets, Twitter hashtags, and APS citations, never on IC/LT traces. That makes it a poor headline task for us and an unusually good **stress test** of the assumption every other file here inherits.

All URLs checked for HTTP 200 on **2026-07-28** unless annotated otherwise.

---

## 0. Verification policy

| Tier | Meaning |
| ---- | ------- |
| **[verified]** | Read from the paper's own table via text extraction, or from the data repository's own statistics page. Trustworthy. |
| **[derived]** | Computed by us from a file we downloaded, or reconstructed arithmetically from published splits. |
| **[figure]** | Read off a plotted figure — the paper published no table. Approximate, direction only. |
| **[claim]** | Stated in prose by a paper or a secondary source; not cross-checked against a file or table. |

Two extraction hazards specific to *this* literature, both encountered while compiling this file:

1. **CasFlow's headline result table (TKDE'21 Table 3) is a raster image.** `pdftotext -layout` returns the caption and nothing else. Every "CasFlow result" quoted below therefore comes from a *later* paper that re-ran it, and is labelled with which one. Do not trust any source that hands you CasFlow's own Table 3 as text — it cannot have read it.
2. **Three different corpora are all called "Twitter"** and three different preprocessings are all called "Weibo" (§6.4). MSLE values from two papers are not comparable until you have matched the cascade count *and* the observation window. This is the single most common way to build an invalid table here.

**Edge-count convention.** Undirected graphs are quoted as *undirected edges*; directed graphs as *arcs*. Our loaders report `adjacency.nnz`, which for a symmetrized undirected graph is **2× the undirected edge count**.

---

## 1. Task definition

A cascade `c` is a set of adoption events `{(u_1, t_1), (u_2, t_2), …}` on an underlying social graph `G`, ordered by time, seeded by an originator `u_1` at `t_1 = 0`. Let `P_c(t)` be its popularity (number of adopters) at time `t`. Fix an **observation window** `t_o` and a **prediction horizon** `t_p > t_o`.

### 1.1 Macroscopic formulation (the dominant one)

Predict the **incremental popularity** over the horizon:

```
ΔP_c = P_c(t_p) − P_c(t_o)          given only events in [0, t_o]
```

Practically every deep-learning paper since DeepCas (2017) solves exactly this, regresses it in log space, and reports MSLE (§8). Some older work predicts total `P_c(t_p)` instead of the increment — a difference that silently changes the metric's scale and is a frequent source of non-comparable tables (§8.2).

### 1.2 Microscopic formulation

Two sub-variants, both node-level:

- **Next-adopter prediction** — rank all inactive nodes by `P(v adopts next)`, score with Hits@k / MAP@k. Topo-LSTM, DeepDiffuse, FOREST, MS-HGAT.
- **Will-user-x-adopt** — binary classification for a *given* `(v, c)` pair, scored with AUC / F1. DeepInf (KDD'18) is the canonical instance.

### 1.3 Classification variants

- **Will it double?** — given `k` observed reshares, predict `P_c(∞) ≥ 2k`. Cheng et al. (WWW'14) chose this framing precisely because it holds the base rate at 50% and makes accuracy interpretable.
- **Will it go viral / breakout?** — top-`k` set-coverage of the largest cascades (SEISMIC's "breakout coverage").

### 1.4 Why the task exists

Two negative results motivated it. Salganik, Dodds & Watts (Science 2006) showed market outcomes for songs were dominated by social-influence-driven randomness, and Watts (2007) argued cascade size is close to inherently unpredictable. Cheng et al. (WWW'14) is the direct rebuttal: reframe the question so a base rate exists, and the *relative* growth of a cascade is in fact strongly predictable. Every result in §5 lives downstream of that reframing.

---

## 2. Fit with our methodology

**Status: Warning: moderate fit — and the reason matters more than the verdict.**

| Element          | How cascade prediction maps onto `f_θ(G, s_t, a_t) → s_{t+1}`                                                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **State** `s_t`  | `(infected, frontier)` — exactly our two channels. `infected` = adopted by `t`; `frontier` = adopted in the current step. No change needed.                                     |
| **Action** `a_t` | **`NULL`, every step.** Nothing intervenes; the cascade is just watched.                                                                                                        |
| **`T_exo`**      | identity.                                                                                                                                                                       |
| **`T_endo`**     | the real-world adoption process — *not* IC, *not* LT, and not known.                                                                                                            |
| **Objective**    | `ΔP = \|infected at t_p\| − \|infected at t_o\|`, i.e. our rollout's final-count readout under a different name.                                                                   |
| **Evaluation**   | already implemented: `world_model/wm_eval.py::rollout_ensemble` reports `ens_final_count_model` vs `ens_final_count_true`. MSLE over those two numbers *is* the field's metric. |

### 2.1 The action space goes idle

With `a_t = NULL` the model degenerates from a world model to a **forecaster**. None of the five ops (`add_node`, `remove_node`, `add_edge`, `remove_edge`, `set_edge_weight`) fire, `action_sensitivity` is undefined, and the counterfactual forks that carry most of our training signal (`cf_i` branches in `data/generate_wm_data.py`) have nothing to fork on. Conditions 3–6 of the baseline ladder collapse: with no decision to make there is no planning regret, so five of our six arms have nothing to distinguish them.

That alone would make this a weak task. It is not the interesting objection.

### 2.2 The decisive issue is the data, and it is falsifiable

Every number in §5 was measured on **real observed traces**: Sina Weibo retweets, Twitter hashtag adoptions, APS citation cascades. Not one is a simulated IC or LT rollout. Our transitions come from NDlib. Those are different distributions, and the literature has already told us which way the difference cuts:

> IMINFECTOR's authors report that **IMM — tuned against a diffusion simulator — underperformed badly under cascade-based evaluation**, which they attribute to diffusion-model misspecification. See [`influence_maximization.md` §5.5](influence_maximization.md).

Our structured IC head hard-codes `p_new(v) = 1 − ∏_u (1 − q(u→v)·frontier_u)`. That composition rule is a *modelling commitment*, not a learned fact. Under real retweet dynamics it is wrong in at least three known ways: adoption is **not memoryless** (Hawkes self-excitation and power-law reaction-time decay dominate — §3), a node gets **repeated** exposure rather than one shot per neighbour, and **exogenous** arrivals (search, front pages, off-platform sharing) inject adopters with no infected in-neighbour at all.

So: cascade prediction is the place where our IC/LT assumption is most directly **falsifiable**. That is a bad property for a headline task and an excellent one for a stress test. Running our structured head against a real Weibo cascade corpus and reporting the MSLE it achieves — even if it loses to CasFlow — converts §5.5's warning from a cited anecdote into a measured limitation of *this* model. That is worth a subsection of the paper.

### 2.3 This is the concrete form of "logged trajectories"

`research_notes/Baselines - Oracle vs MC vs GWM.md` §4 Q2 argues that outside the benchmark the GWM's training data comes from "historical observations, logged interventions, or online interaction", and that the simulator is a stand-in for that logbook. Cascade corpora **are** that logbook, in public, downloadable form. Weibo/APS/Twitter give `(s_t, NULL, s_{t+1})` transitions with no simulator anywhere in the loop — the only datasets in this folder for which the claim "trains from transition data alone" can be demonstrated rather than asserted. Note the same document's honest wrinkle: the IC heads consume the true `w` as an input feature. On real cascades **there is no `w`**, so this task forces the `w`-hidden variant that §4 flags as the clean ablation.

### 2.4 Honest cost

| Item | Cost | Note |
| ---- | ---- | ---- |
| Cascade-corpus loader (`data/datasets/weibo_cascades.py`) | **~1 day** | parse `(root, adopter, timestamp)` triples → per-cascade event list. The formats are trivial; §6.3 gives them. |
| Replay a cascade as `(s_t, NULL, s_{t+1})` transitions | **~1 day** | bin events into steps, emit our JSONL. Reuses `State`/`ActionOp` unchanged. |
| **No MC marginals available** | **blocker-ish** | our targets are soft `P(infected)` from `--mc-marginals` re-runs. A real cascade happened **once**. Targets become hard 0/1, which is a genuine change to the training signal, not a config flag. |
| Global graph at Weibo scale | **hard** | 1.17M nodes / 226M arcs. Already flagged in `influence_maximization.md` §6.1 as beyond our simulation pipeline. APS (616K nodes / 3.3M edges) is the tractable entry point. |
| Metric layer (MSLE / MALE / MAPE / PCC / COV-k) | **~2 hours** | thin wrapper over `rollout_ensemble`'s existing counts. |
| Reaching published-table parity | **not worth it** | CasFlow/CasFT are 2M-parameter models tuned for this one task. We would lose, and losing is fine — the point is the diagnostic, not the leaderboard. |

**Recommended scope: the smallest honest version.** APS only, MSLE only, our structured head vs a persistence baseline vs one published number. Framed as "what happens to an IC-structured world model when the dynamics are not IC" — not as a cascade-prediction contribution.

---

## 3. Classical and heuristic methods

Three families, in the order the field discovered them.

### 3.1 Feature-driven regression / classification

Extract hand-crafted features from the observed prefix, feed a linear model, random forest, or MLP. Still competitive — CasFlow's own ablation notes that feature models "in some cases even beat deep learning models" [verified, §5.2 O2].

| Method | Year | Venue | Idea | Paper | Code |
| ------ | ---- | ----- | ---- | ----- | ---- |
| **Szabo & Huberman (S&H)** | 2008/2010 | CACM | The origin: `log P(t_p)` is near-linear in `log P(t_o)`. One feature, one line. Still the baseline every paper calls "Feature-S&H". | [arXiv 0811.0405](https://arxiv.org/abs/0811.0405) | no public code found |
| **Kupavskii et al.** | 2012 | CIKM | Retweet-cascade size over time from user/flow/temporal features; introduced the "predict at fixed elapsed time" protocol | [ACM 10.1145/2396761.2398634](https://dl.acm.org/doi/10.1145/2396761.2398634) Warning: ACM blocks `curl` (403 with and without browser UA); resolves in a browser | no public code found |
| **Cui et al.** | 2013 | KDD | "Cascading outbreak prediction" — selects a small set of *sensor* nodes whose early activation predicts outbreak; a structural, not temporal, view | [ACM 10.1145/2487575.2487639](https://dl.acm.org/doi/10.1145/2487575.2487639) Warning: ACM 403 to `curl` | no public code found |
| **Cheng et al.** (key) | 2014 | WWW | *Can cascades be predicted?* Reframes size prediction as balanced binary "will it double". Five feature classes: content, root, structural, temporal, community. **Temporal features dominate.** | [arXiv 1403.4608](https://arxiv.org/abs/1403.4608) | no public code found (Facebook-internal data) |
| **Weng, Menczer & Ahn** | 2014 | ICWSM / Sci. Rep. | Community structure of the early adopter set predicts meme virality better than volume — early diffusion across many communities ⇒ viral | [arXiv 1403.6199](https://arxiv.org/abs/1403.6199) | no public code found |
| **Martin et al.** | 2016 | WWW | Measures the *ceiling*: even with perfect features, an irreducible-noise bound caps achievable `R²`. The sober counterweight to §5's leaderboard. | [arXiv 1602.01013](https://arxiv.org/abs/1602.01013) | no public code found |

### 3.2 Generative point processes

Model the arrival process itself; predict by integrating the fitted intensity to `t_p`. These are the closest classical analogue to a *world model*: they are explicit generative dynamics, fit per cascade, then rolled forward.

| Method | Year | Venue | Idea | Paper | Code |
| ------ | ---- | ----- | ---- | ----- | ---- |
| **RPP (Reinforced Poisson)** | 2014 | AAAI / Sci. Rep. | `λ_t = c · f_γ(t) · r_α(R_t)` — fitness × aging × rich-get-richer. Built for citation counts. | [arXiv 1401.0778](https://arxiv.org/abs/1401.0778) | no public code found |
| **SEISMIC** (key) | 2015 | KDD | Self-exciting point process with a *time-varying infectiousness* `p_t` estimated online; closed-form final-size estimator, O(n) per cascade, no features | [arXiv 1506.02594](https://arxiv.org/abs/1506.02594) | [CRAN `seismic`](https://cran.r-project.org/package=seismic) |
| **Hawkes + predictive layer** (key) | 2016 | CIKM | Fits a marked Hawkes process per cascade, then trains a random forest on `{c, θ, A_1, n*}` to correct the generative estimate. Beats SEISMIC on both mean ARE and on *how many* cascades it can score at all. | [arXiv 1608.04862](https://arxiv.org/abs/1608.04862) | [github.com/s-mishra/featuredriven-hawkes](https://github.com/s-mishra/featuredriven-hawkes) |
| **HIP** | 2017 | WWW | Hawkes Intensity Process: adds *exogenous* promotion (search, shares) as an external stimulus — the term IC has no place for | [arXiv 1602.06033](https://arxiv.org/abs/1602.06033) | [github.com/andrei-rizoiu/hip-popularity](https://github.com/andrei-rizoiu/hip-popularity) |

**Why §3.2 matters to us more than §3.1.** SEISMIC and HIP are exactly what our world model is: a stated transition mechanism plus a fitted parameter, rolled forward. Their failure mode is also ours — SEISMIC produces **no prediction at all** for supercritical cascades (`p ≥ 1/n*`), 507 of ~30K on Tweet-1Mo at 5 minutes [verified, §5.4], because the branching factor exceeds 1 and the expected size diverges. That is the same runaway our `ens_count_bias` metric was built to catch, and the same reason our structured head gates on `frontier_u`. The field's answer — a learned corrective layer on top of a generative core (Mishra et al.) — is structurally identical to our `structured_residual` head.

---

## 4. Learning-based methods

### 4.1 Macroscopic (final size / increment) — the main line

| Method | Year | Venue | Approach | Paper | Code |
| ------ | ---- | ----- | -------- | ----- | ---- |
| **DeepCas** (key) | 2017 | WWW | The founding deep model. Samples random walks over the cascade graph, encodes with bi-GRU + attention, regresses `log ΔP`. Kills hand-crafted features. | [arXiv 1611.05373](https://arxiv.org/abs/1611.05373) | [github.com/chengli-um/DeepCas](https://github.com/chengli-um/DeepCas) |
| **DeepHawkes** (key) | 2017 | CIKM | Injects the three Hawkes ingredients (user influence, self-excitation, time decay) into a GRU over *diffusion paths*. The interpretability-vs-accuracy bridge; still the most-reproduced baseline. | [ACM 10.1145/3132847.3132973](https://dl.acm.org/doi/10.1145/3132847.3132973) Warning: ACM 403 to `curl` | [github.com/CaoQi92/DeepHawkes](https://github.com/CaoQi92/DeepHawkes) |
| **Topo-LSTM** | 2017 | ICDM | LSTM whose gates are wired to the cascade's dynamic DAG rather than a linear sequence. Microscopic, but the structural idea seeded CasCN. | [arXiv 1711.10162](https://arxiv.org/abs/1711.10162) | [github.com/vwz/topolstm](https://github.com/vwz/topolstm) |
| **CasCN** (key) | 2019 | ICDE | Cascade as a *sequence of sub-cascade graphs*; GCN each snapshot, LSTM across them. First to use both structure and time properly. | [PDF via NSF-PAR](https://par.nsf.gov/servlets/purl/10122600) Warning: `curl` rejects the cert (hostname mismatch); downloads fine in a browser and with `-k` | [github.com/ChenNed/CasCN](https://github.com/ChenNed/CasCN) |
| **CoupledGNN** (key) | 2020 | WSDM | **The closest published method to our formulation.** Two coupled GNNs: one propagates node *activation state*, one propagates *influence*, iterated over `K` layers to imitate the cascading effect on the global graph. | [arXiv 1906.09032](https://arxiv.org/abs/1906.09032) | [github.com/CaoQi92/CoupledGNN](https://github.com/CaoQi92/CoupledGNN) |
| **VaCas** | 2020 | INFOCOM | Hierarchical VAE over cascade graph + Bayesian node embeddings; the first to model *diffusion uncertainty* rather than a point estimate | [IEEE 10.1109/INFOCOM41043.2020.9155349](https://doi.org/10.1109/INFOCOM41043.2020.9155349) Warning: IEEE returns 202 to `curl` | no public code found |
| **CasFlow** (key) | 2021 | TKDE | VaCas + **normalizing flows** over the latent, plus a global (whole social network) embedding alongside the local cascade graph. The reference SOTA of 2021–23 and the baseline every later paper reports. | [PDF (author copy)](https://www.xoveexu.com/file/paper/21-11-TKDE-CasFlow.pdf) Warning: `curl` cert hostname mismatch; fetches fine with `-k` and in a browser · [IEEE 9611000](https://ieeexplore.ieee.org/document/9611000) | [github.com/Xovee/casflow](https://github.com/Xovee/casflow) (mirror: [kpzhang/casflow](https://github.com/kpzhang/casflow)) |
| **CasSeqGCN** | 2021 | ESWA | Snapshot GCN + LSTM, but the node state (not the structure) is what varies across snapshots — a cheaper CasCN | [arXiv 2110.06836](https://arxiv.org/abs/2110.06836) | [github.com/MrYansong/CasSeqGCN](https://github.com/MrYansong/CasSeqGCN) |
| **TempCas** | 2021 | IPM | Adds an explicit *macroscopic temporal* branch (full-size-sequence CNN + attention) on top of cascade-graph learning | DOI [10.1016/j.ipm.2021.102593](https://doi.org/10.1016/j.ipm.2021.102593) | no public code found |
| **CCasGNN** | 2021/22 | CSCWD | Collaborative framework: GAT + GCN with positional encoding, fused in sequence | [arXiv 2112.03644](https://arxiv.org/abs/2112.03644) | [github.com/MrYansong/CCasGNN](https://github.com/MrYansong/CCasGNN) |
| **MUCas** | 2022 | IJCAI | Multi-scale **graph capsule** network with influence attention; directional/dynamic/position-aware cascade encoding | DOI [10.24963/ijcai.2022/300](https://doi.org/10.24963/ijcai.2022/300) | [github.com/ChenNed/MUCas](https://github.com/ChenNed/MUCas) |
| **CCGL** | 2022 | TKDE | Contrastive **self-supervised** pretraining on augmented cascade graphs, then fine-tune — the transfer-learning entry | [github README](https://github.com/Xovee/ccgl) | [github.com/Xovee/ccgl](https://github.com/Xovee/ccgl) |
| **CTCP** (key) | 2023 | IJCAI | **Continuous-time**, cross-cascade: one evolving user/cascade state updated event-by-event, shared across *all* cascades instead of per-cascade encoding | [arXiv 2306.03756](https://arxiv.org/abs/2306.03756) | [github.com/lxd99/CTCP](https://github.com/lxd99/CTCP) |
| **CasDO** | 2024 | TKDE | Probabilistic **diffusion model** denoiser + neural ODE for irregular event times; models both temporal and label uncertainty | DOI [10.1109/TKDE.2024.3465241](https://doi.org/10.1109/TKDE.2024.3465241) Warning: IEEE 202 to `curl` | no public code found |
| **CasFT** (key) | 2024 | AAAI-25 | Neural-ODE growth rate → *dynamic cues* → conditional **DDIM** that generates the future incremental-popularity **sequence**, not just the endpoint. Current best on the standard protocol. | [arXiv 2409.16619](https://arxiv.org/abs/2409.16619) | no public code found |
| **CasTemp** (key) | 2025/26 | (preprint) | Temporal random walks + time-aware attention + an inter-cascade *competition* graph. Deliberately lightweight; its real contribution is the **leak-free split** (§8.3). | [arXiv 2510.25348](https://arxiv.org/abs/2510.25348) | [github.com/Lucas-PJ/CasTemp-ALGO](https://github.com/Lucas-PJ/CasTemp-ALGO) |

### 4.2 Microscopic (next adopter / will-x-adopt)

| Method | Year | Venue | Approach | Paper | Code |
| ------ | ---- | ----- | -------- | ----- | ---- |
| **Topo-LSTM** | 2017 | ICDM | see above; scored with Hits@k / MAP@k | [arXiv 1711.10162](https://arxiv.org/abs/1711.10162) | [github.com/vwz/topolstm](https://github.com/vwz/topolstm) |
| **DeepInf** | 2018 | KDD | GNN over each user's `r`-hop ego network + active-neighbour states → binary "will `v` adopt". The purest *node-level* analogue of our head. | [arXiv 1807.05560](https://arxiv.org/abs/1807.05560) | [github.com/xptree/DeepInf](https://github.com/xptree/DeepInf) |
| **FOREST** | 2019 | IJCAI | RL-based multi-scale: sequential next-adopter decoding with a macroscopic size reward | [IJCAI 2019](https://www.ijcai.org/proceedings/2019/560) | [github.com/albertyang33/FOREST](https://github.com/albertyang33/FOREST) |
| **MS-HGAT** | 2022 | AAAI | Memory-enhanced sequential **hypergraph** attention over user-cascade interaction | [AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/20334) | [github.com/slingling/MS-HGAT](https://github.com/slingling/MS-HGAT) |

### 4.3 Surveys

| Work | Year | Venue | What it gives you |
| ---- | ---- | ----- | ----------------- |
| **Zhou, Xu, Trajcevski & Zhang** (key) | 2021 | ACM CSUR 54(2) | *A Survey of Information Cascade Analysis: Models, Predictions, and Recent Advances* — the standard taxonomy (feature-based / generative / deep) and the source of most "macroscopic vs microscopic" phrasing. [arXiv 2005.11041](https://arxiv.org/abs/2005.11041) · [ACM 10.1145/3433000](https://dl.acm.org/doi/10.1145/3433000) Warning: ACM 403 to `curl` |
| **Gao, Zhou et al.** | 2022 | arXiv | *Graph Representation Learning for Popularity Prediction Problem: A Survey* — narrower, GNN-focused, tabulates which model uses which graph. [arXiv 2203.07632](https://arxiv.org/abs/2203.07632) |

---

## 5. Published results

Four transcribable tables exist. They are **not mutually comparable** — §5.5 explains exactly why. Read §8 before quoting any cell.

### 5.1 CasFT (AAAI-25) (key) the most usable table in the literature

The widest baseline set under the standard protocol, and the only recent paper whose result table survives `pdftotext`. Lower is better throughout.

**Performance comparison, MSLE / MAPE** [verified, Table 2]:

| Method | Twitter 1d MSLE | MAPE | Twitter 2d MSLE | MAPE | APS 3y MSLE | MAPE | APS 5y MSLE | MAPE | Weibo 0.5h MSLE | MAPE | Weibo 1h MSLE | MAPE |
| ------ | --------------- | ---- | --------------- | ---- | ----------- | ---- | ----------- | ---- | --------------- | ---- | ------------- | ---- |
| Feature-based | 7.8268 | 0.7073 | 6.5154 | 0.6514 | 1.9881 | 0.3085 | 1.9696 | 0.3193 | 4.0788 | 0.4094 | 3.6380 | 0.4268 |
| SEISMIC | 10.687 | 0.9689 | 8.1851 | 0.8147 | 2.0583 | 0.3013 | 2.3013 | 0.4320 | 5.0300 | 0.4819 | 4.0594 | 0.5003 |
| DeepCas | 6.3297 | 0.6566 | 5.7146 | 0.6671 | 2.1051 | 0.2869 | 1.9260 | 0.3458 | 4.6460 | 0.3258 | 3.5532 | 0.3532 |
| DeepHawkes | 5.9341 | 0.5017 | 4.8489 | 0.5189 | 1.9142 | 0.2823 | 1.8145 | 0.3368 | 2.8741 | 0.3041 | 2.7434 | 0.3346 |
| CasCN | 5.8742 | 0.4894 | 4.7154 | 0.4974 | 1.8930 | 0.2763 | 1.7494 | 0.3208 | 2.7931 | 0.2940 | 2.6831 | 0.3255 |
| VaCas | 5.5124 | 0.4796 | 4.2147 | 0.4871 | 1.7764 | 0.2697 | 1.6945 | 0.3012 | 2.5246 | 0.2847 | 2.3451 | 0.2997 |
| CasFlow | 4.7799 | 0.4150 | 3.6888 | 0.4222 | 1.4370 | 0.2401 | 1.3346 | 0.2624 | 2.3370 | 0.2665 | 2.2232 | 0.2949 |
| CTCP | 5.3991 | 0.3757 | 3.6016 | 0.3773 | 1.7676 | 0.3054 | 1.3751 | 0.2908 | 2.5572 | 0.3056 | 2.2968 | 0.3010 |
| **CasFT** (key) | **3.8546** | **0.3674** | **3.4496** | **0.3605** | **1.2468** | **0.2282** | **1.1748** | **0.2561** | **2.1728** | **0.2448** | **2.0655** | **0.2695** |

Dataset sizes **as used by CasFT** [verified, Table 1]:

| | Twitter | APS | Weibo |
| --- | --- | --- | --- |
| Cascades | 86,764 | 207,685 | 119,313 |
| Avg. popularity | 94 | 51 | 240 |
| Train (1d / 3y / 0.5h) | 7,308 | 18,511 | 21,463 |
| Val | 1,566 | 3,967 | 4,599 |
| Test | 1,566 | 3,966 | 4,599 |
| Train (2d / 5y / 1h) | 10,983 | 32,102 | 29,908 |
| Val | 2,353 | 6,879 | 6,409 |
| Test | 2,353 | 6,879 | 6,408 |

Protocol [verified]: prediction horizon 15 days (Twitter) / 20 years (APS) / 24 hours (Weibo); cascades with < 10 participants in the observation window are dropped; **70/15/15 random split**; `MSLE = 1/M Σ (log₂(P+1) − log₂(P̂+1))²`, `MAPE = 1/M Σ |log₂(P+2) − log₂(P̂+2)| / log₂(P+2)`.

**Read the MAPE definition twice.** It is *not* relative error on popularity — it is relative error on `log₂` popularity. A "0.25 MAPE" here is not 25% off.

### 5.2 CTCP (IJCAI 2023) — different preprocessing, four metrics

[verified, Table 2]. Note the dataset sizes differ from §5.1 by 3–10× (§5.5).

| Model | Tw MSLE | Tw MALE | Tw MAPE | Tw PCC | Wb MSLE | Wb MALE | Wb MAPE | Wb PCC | APS MSLE | APS MALE | APS MAPE | APS PCC |
| ----- | ------- | ------- | ------- | ------ | ------- | ------- | ------- | ------ | -------- | -------- | -------- | ------- |
| XGBoost | 11.5330 | 2.9871 | 0.8571 | 0.3792 | 3.6253 | 1.3736 | 0.3571 | 0.6493 | 2.5808 | 1.2559 | 0.3437 | 0.4762 |
| MLP | 11.9105 | 2.9712 | 0.9324 | 0.3733 | 3.9370 | 1.4409 | 0.3812 | 0.6098 | 2.6075 | 1.2577 | 0.3516 | 0.4787 |
| DeepHawkes | 7.7795 | 2.1553 | 0.5547 | 0.6500 | 4.2520 | 1.4658 | 0.3998 | 0.5670 | 2.3356 | 1.2001 | 0.3158 | 0.5524 |
| DFTC | 5.9173 | 1.8426 | 0.4851 | 0.7495 | 2.9370 | 1.2046 | 0.2959 | 0.7296 | 2.0357 | 1.1159 | 0.2943 | 0.6247 |
| CasCN | 7.1021 | 2.0567 | 0.5231 | 0.6940 | 3.7714 | 1.4040 | 0.3612 | 0.6707 | 2.1248 | 1.1358 | 0.3035 | 0.6062 |
| MS-HGAT | 5.9992 | 1.9006 | 0.4741 | 0.7507 | OOM | OOM | OOM | OOM | OOM | OOM | OOM | OOM |
| TempCas | 5.5870 | 1.7584 | 0.4574 | 0.7651 | 2.7453 | 1.1702 | 0.2786 | 0.7500 | 2.0043 | 1.1022 | 0.2957 | 0.6346 |
| CasFlow | 5.2549 | 1.5775 | 0.4031 | 0.7847 | 2.6336 | 1.1230 | 0.2687 | 0.7619 | 2.0064 | 1.1053 | 0.2936 | 0.6320 |
| **CTCP** (key) | **4.6916** | **1.5668** | **0.3562** | **0.8136** | **2.5929** | 1.1414 | 0.2723 | **0.7667** | **1.6289** | **0.9906** | **0.2611** | **0.7176** |

CTCP's datasets [verified, Table 1]: Twitter 199,005 users / 19,718 cascades / 602,253 retweets · Weibo 918,852 / 39,076 / 1,572,287 · APS 218,323 / 48,575 / 939,686. Features for XGBoost/MLP follow Cheng et al. (edge count, max depth, avg depth, breadth, publication time) [verified].

### 5.3 CasTemp / "Beyond Leakage" (2025) (key) the table that changes the story

Same six baselines, re-run under a **leak-free time-ordered split** (§8.3). Mean ± std over three runs, lower is better [verified, Table 4]:

| Method | Twitter MSLE | MALE | Weibo MSLE | MALE | APS MSLE | MALE | Taoke MSLE | MALE |
| ------ | ------------ | ---- | ---------- | ---- | -------- | ---- | ---------- | ---- |
| MLP | 1.614 ± 0.010 | 0.965 ± 0.003 | 2.066 ± 0.004 | 0.954 ± 0.004 | 2.982 ± 0.042 | 1.294 ± 0.014 | 5.376 ± 0.094 | 1.991 ± 0.028 |
| DeepHawkes | 1.408 ± 0.036 | 0.948 ± 0.003 | 1.751 ± 0.004 | 1.060 ± 0.014 | 2.528 ± 0.041 | 1.296 ± 0.007 | 7.324 ± 0.148 | 2.239 ± 0.052 |
| CasCN | 1.206 ± 0.018 | 0.913 ± 0.005 | 1.981 ± 0.757 | 0.992 ± 0.060 | 2.283 ± 0.031 | 1.183 ± 0.010 | 2.795 ± 0.057 | 1.308 ± 0.033 |
| CasFlow | 1.329 ± 0.009 | 0.930 ± 0.004 | 1.685 ± 0.017 | 0.950 ± 0.004 | 2.438 ± 0.038 | 1.438 ± 0.024 | 3.300 ± 0.036 | 1.436 ± 0.049 |
| CTCP | 1.446 ± 0.001 | 0.928 ± 0.001 | 1.890 ± 0.003 | 0.966 ± 0.000 | 2.807 ± 0.013 | 1.248 ± 0.003 | 3.308 ± 0.022 | 1.398 ± 0.024 |
| CasDO | 2.130 ± 0.032 | 0.972 ± 0.001 | 2.490 ± 0.413 | 1.063 ± 0.005 | 4.815 ± 0.253 | 1.723 ± 0.036 | 9.921 ± 0.168 | 2.638 ± 0.071 |
| **CasTemp** (key) | **1.171 ± 0.002** | **0.905 ± 0.003** | **1.475 ± 0.007** | **0.919 ± 0.006** | **1.926 ± 0.018** | **1.074 ± 0.010** | **0.685 ± 0.038** | **0.548 ± 0.015** |

**The finding that matters more than the winner:** under leak-free splits, **CasFlow and CasDO fall below a plain MLP** on the toy diagnostic [verified, Table 2 — CasFlow 4.6503 / CasDO 5.6532 vs MLP 2.9478 MSLE on Scenario 1], and on APS the whole field compresses into 2.28–4.82 where §5.1 reported 1.19–2.11. The authors' diagnosis, from train-vs-test loss curves: "CasFlow and CasDo exhibit low training losses but significantly higher test losses… their complex architectures have likely learned dataset-specific shortcuts enabled by temporal leakage" [verified, §6.1 prose].

The new **Taoke** corpus (Taobao product promotion, with ground-truth purchase conversions) [verified, Table 3]: Twitter 67,760 cascades / 145,188 nodes · Weibo 48,693 / 353,504 · APS 90,768 / 118,312 · Taoke 2,862 / 29,711. Avg. path length 3.86 / 3.51 / 5.12 / 5.61. Taoke is the only one of the four with cascade *and* node features.

### 5.4 Point-process era — SEISMIC vs Hawkes

Mean Absolute Relative Error (ARE), lower is better [verified, Mishra et al. Table 2]. `n_failed` = cascades the generative model could not score at all:

| Dataset | Approach | ARE 5 min | ARE 10 min | ARE 1 hour | failed 5m | 10m | 1h |
| ------- | -------- | --------- | ---------- | ---------- | --------- | --- | -- |
| Tweet-1Mo | SEISMIC | 2.61 ± 55.80 | 0.70 ± 15.58 | 0.51 ± 10.81 | 507 | 164 | 71 |
| Tweet-1Mo | Hawkes | **0.36 ± 0.52** | **0.33 ± 0.41** | **0.30 ± 0.38** | 302 | 105 | 58 |
| News | SEISMIC | 11.13 ± 282.96 | 0.84 ± 11.77 | 0.33 ± 0.92 | 1022 | 155 | 123 |
| News | Hawkes | **0.42 ± 6.83** | **0.25 ± 0.60** | **0.22 ± 1.16** | 149 | 45 | 37 |

Adding hand-crafted features on top [verified, Table 3, News July'15]:

| Approach | 5 min | 10 min | 1 hour |
| -------- | ----- | ------ | ------ |
| SEISMIC | 15.16 ± 375.08 | 0.71 ± 4.89 | 0.32 ± 0.40 |
| Feature-driven | 0.25 ± 0.18 | 0.22 ± 0.17 | 0.17 ± 0.14 |
| Hawkes | 0.27 ± 1.83 | 0.22 ± 0.80 | 0.17 ± 0.36 |
| **Hybrid** (key) | **0.17 ± 0.16** | **0.15 ± 0.14** | **0.11 ± 0.12** |

SEISMIC's own numbers, on its own corpus [verified, prose §5.5.2]: after 10 minutes of observation the 95th/75th/50th APE percentiles are 71% / 44% / 25%; after 1 hour, 62% / 30% / 15%. Breakout coverage: **78 of the top-100** and **281 of the top-500** most-reshared tweets identified within 10 minutes.

### 5.5 CoupledGNN (WSDM 2020) — the network-aware, no-temporal protocol

The only table here that uses **MRSE / mRSE / MAPE / WroPerc** instead of MSLE, because CoupledGNN deliberately withholds timestamps and uses only the early adopter set plus the global graph [verified, Table 1, Sina Weibo]:

| Observation | 1 h MRSE | mRSE | MAPE | WroPerc | 2 h MRSE | mRSE | MAPE | WroPerc | 3 h MRSE | mRSE | MAPE | WroPerc |
| ----------- | -------- | ---- | ---- | ------- | -------- | ---- | ---- | ------- | -------- | ---- | ---- | ------- |
| SEISMIC | — | 0.2112 | — | 48.63% | — | 0.1347 | — | 34.59% | — | 0.0823 | — | 27.15% |
| Feature-based | 0.2106 | 0.1254 | 0.3749 | 35.17% | 0.1796 | 0.1041 | 0.3557 | 28.86% | 0.1581 | 0.0804 | 0.3147 | 18.97% |
| DeepCas | 0.2077 | **0.0930** | 0.3633 | 30.00% | 0.1650 | 0.0670 | 0.3134 | 20.55% | 0.1365 | 0.0361 | 0.2813 | 17.24% |
| **CoupledGNN** (key) | **0.1816** | 0.0946 | **0.3515** | **25.68%** | **0.1397** | **0.0519** | **0.2989** | **17.81%** | **0.1120** | **0.0333** | **0.2611** | **13.01%** |

`WroPerc` = fraction of cascades whose relative error exceeds ε = 0.5 [verified]. SEISMIC has no MRSE/MAPE cell because it predicts **infinite** popularity for some cascades — the supercritical failure again.

CoupledGNN's Weibo subset [verified, §5.1]: sampled from the AMiner following network (1.78M users / 308M following relations / 300K cascades) down to **23,681 users / 1,802,146 edges / 3,228 cascades** (≥5 active users), split **80/10/10**, observation windows **1 h, 2 h, 3 h**.

### 5.6 Cheng et al. (WWW 2014) — the classification framing

[verified, §3.3 prose] `N_c = 150,572` Facebook photos each reshared ≥5 times, `N_r = 9,233,300` reshares.

| Task (observe first k=5 reshares) | Accuracy | AUC |
| --------------------------------- | -------- | --- |
| Will it reach median size (double)? | **0.795** | **0.877** |
| Top quartile vs bottom quartile | **0.926** | **0.976** |
| Best single feature — reshare rate in 2nd half | 0.73 | — |
| Next best — views of original, time-to-5th-reshare | 0.72 | — |
| Best structural feature — `did_leave`, `outdeg(v0)` | 0.65 | — |
| Content features alone | 0.558 | — |

Accuracy **increases** with k [figure, Fig 5] — predicting whether a 25-reshare cascade doubles is easier than whether a 5-reshare one does, even though the target moves with k. This is the closest thing the field has to a "more observation ⇒ better" law, and it is the reason every protocol in §8 sweeps at least two observation windows.

### 5.7 Why §5.1–§5.5 cannot be merged into one table

Five independent incompatibilities, any one of which invalidates a merge:

| # | Difference | Concretely |
| - | ---------- | ---------- |
| 1 | **Different "Twitter"** | CasFlow: 88,440 cascades (2012 hashtags). CasFT: 86,764 (2022 hashtags). CTCP: 19,718. CasTemp: 67,760. Four corpora, one name. |
| 2 | **Different "Weibo" preprocessing** | CasFlow/CasFT: 119,313 cascades. CTCP: 39,076. CasTemp: 48,693. CoupledGNN: 3,228. All from the same AMiner source. |
| 3 | **Total vs incremental target** | CasFlow regresses `ΔP = P(t_p) − P(t_o)` [verified]; CasFT's Eq. 26 regresses `P` [verified]. Same symbol, different quantity. |
| 4 | **Different metric definitions** | CasFlow's MSLE uses `log₂ ΔP`; CasFT's uses `log₂(P+1)`; CTCP's loss uses natural `log ΔP`. CoupledGNN uses MRSE, not MSLE at all. |
| 5 | **Different splits** | §5.1–§5.2 use 70/15/15 **random over cascades**; §5.3 uses 1:1:1 **chronological**; §5.4 uses time-ordered by day; §5.5 uses 80/10/10 random. |

Practical rule: **quote a row only together with the paper it came from.** The only legitimate cross-paper comparison in this file is *within* §5.3, because CasTemp re-ran every baseline itself under one protocol.

---

## 6. Datasets

### 6.1 What we already load yes

**None of them carry cascades.** Every graph in [`influence_maximization.md` §6.1](influence_maximization.md) is topology only — `jazz` 198/2,742 · `email_eu_core` 1,005/24,929 arcs · `netscience` 1,589/2,742 · `cora_ml` 2,810/7,981 · `facebook` 4,039/88,234 · `power_grid` 4,941/6,594 · `ca_grqc` 5,242/14,484 · `wiki_vote` 7,115/103,689 arcs · `lastfm_asia` 7,624/27,806 · `nethept` 15,229/62,752 arcs · `netphy` 37,154/174,161 · `twitter` 81,306/1.77M arcs · `digg` 116,893/≈2.6M · `youtube` 1,134,890/2,987,624 · `weibo` 1,787,443/≈216M arcs. Plus synthetic `er`, `ba`, `ws`, `sbm`, `karate`. All counts [verified] by running the loaders.

Two of those names collide with cascade corpora and **are not the same data**:

| We load | The cascade literature's version | Same? |
| ------- | -------------------------------- | ----- |
| `digg` — Syracuse friendship graph, 116,893 / ≈2.6M | ISI/Lerman Digg 2009 — 279,632 nodes / 2,617,993 edges **+ 3,553 vote cascades** [verified, Topo-LSTM Table II] | no different graph, and ours has no cascades |
| `weibo` — AMiner *following network*, 1,787,443 / ≈216M arcs | DeepHawkes Weibo — the **retweet cascades** built from the same AMiner release | Warning: same source, different artefact: we load the graph, they load the traces |

**The `weibo` overlap is the cheapest possible entry point.** We already download and parse the AMiner Influence-Locality release. The retweet cascades live in the same release; adding them means a second parser, not a second download. See §6.3.

### 6.2 The cascade corpora (key)

Everything a comparison needs, in one place. "Cascades" is the count *after* the paper's own filtering unless noted; observation windows and splits are the two columns that make or break comparability.

| Corpus | Underlying graph | Cascades | Avg size | Obs. windows | Horizon `t_p` | Split | Used by |
| ------ | ---------------- | -------- | -------- | ------------ | ------------- | ----- | ------- |
| **Weibo** (DeepHawkes release) | 6,738,040 nodes / 15,249,636 edges [verified, CasFlow Table 2] | 119,313 [verified] | 240 [verified] | **0.5 h, 1 h** (CasFlow/CasFT); **1 h, 2 h, 3 h** (CoupledGNN/DeepHawkes) | 24 h [verified] | 70/15/15 random [verified] | DeepHawkes, CasCN, VaCas, CasFlow, CTCP, CasDO, CasFT |
| **Twitter** (Weng 2013 hashtags) | 490,474 / 1,903,230 [verified, CasFlow Table 2] | 88,440 [verified] | 142 [verified] | **1 d, 2 d** [verified] | 32 d [verified] | 70/15/15 random [verified] | CasFlow, CCGL, MUCas |
| **APS** (citation cascades) | 616,316 / 3,304,400 [verified, CasFlow Table 2] | 207,685 [verified] | 51 [verified] | **3 y, 5 y** [verified] | 20 y [verified] | 70/15/15 random [verified] | CasFlow, CTCP, CasDO, CasFT |
| **Digg 2009** | 279,632 / 2,617,993 [verified, Topo-LSTM Table II] | 3,553 [verified] | 30.0 [verified] | first-`k`-adopters (microscopic) | — | 75/25, 10% of train held out for val [verified] | Topo-LSTM, DeepDiffuse, FOREST |
| **Twitter URLs** (Hodas–Lerman) | 137,093 / 3,589,811 [verified, Topo-LSTM Table II] | 569 [verified] | 5.7 [verified] | first-`k`-adopters | — | 75/25 [verified] | Topo-LSTM, FOREST, MS-HGAT |
| **Memes** (MemeTracker) | 5,000 / 313,669 [verified, Topo-LSTM Table II] | 54,847 [verified] | 17.0 [verified] | first-`k`-adopters | — | 75/25 [verified] | Topo-LSTM, FOREST |
| **Tweet-1Mo** (SEISMIC release) | follower counts only, no graph [verified] | 30,463 sampled, length ∈ [50, 5000] [verified] | mean 160, **median 95** [verified] | 5 min, 10 min, 1 h [verified] | ∞ (final size) | 40% train / 60% test [verified] | SEISMIC, Mishra et al. |
| **News** (Mishra 2016) | 49,735,271 tweets crawled [verified] | 110,954 (length ≥ 50) [verified] | mean 158, **median 90** [verified] | 5 min, 10 min, 1 h [verified] | ∞ | July 1–15 train / 16–31 test [verified] | Mishra et al. |
| **Weibo-CoupledGNN** | 23,681 users / 1,802,146 edges [verified] | 3,228 (≥5 adopters) [verified] | not reported | **1 h, 2 h, 3 h** [verified] | final size | 80/10/10 [verified] | CoupledGNN |
| **Taoke** (e-commerce, 2025) | 29,711 nodes [verified, CasTemp Table 3] | 2,862 [verified] | avg path length 5.61 [verified] | 1 d segments [verified] | next segment | **chronological 1:1:1** [verified] | CasTemp |
| **Facebook photos** | Facebook internal | 150,572 photos / 9,233,300 reshares [verified] | ≥5 by construction | k = 5…100 reshares [verified] | 2k (doubling) | not stated | Cheng et al. — **not public** |

**Three corpora dominate.** Weibo + Twitter + APS is *the* triple: CasFlow established it, and CasFT, CasDO, CTCP and CasTemp all report on exactly those three. If we ever produce a number here, it must be on one of them.

### 6.3 Where to download, and in what shape

| Corpus | Direct link | Auto-DL? | Format |
| ------ | ----------- | -------- | ------ |
| **Weibo, Twitter, APS** — CasFlow's preprocessed bundle (key) | [Google Drive `1o4KAZs…`](https://drive.google.com/file/d/1o4KAZs19fl4Qa5LUtdnmNy57gHa15AF-/view) · [Baidu mirror, pw `1msd`](https://pan.baidu.com/s/1tWcEefxoRHj002F0s9BCTQ) | no Drive interstitial | one line per cascade: `cascade_id \t root \t pub_time \t n \t path1:t1 path2:t2 …` where a path is `u1/u2/u3` |
| **Weibo** — original DeepHawkes release | [github.com/CaoQi92/DeepHawkes](https://github.com/CaoQi92/DeepHawkes) · [Drive `1fgkLeF…`](https://drive.google.com/file/d/1fgkLeFRYQDQOKPujsmn61sGbJt6PaERF/view) | no | same path format |
| **Weibo** — raw source (what *we* already fetch) | [AMiner Influence Locality](https://www.aminer.cn/influencelocality) | no registration ([`data/datasets/weibo.py`](../data/datasets/weibo.py) documents the manual steps) | `weibo_network.txt` (graph, already parsed by us) + `total.txt` (retweet traces, **not** currently parsed) |
| **Twitter** — original | [carl.cs.indiana.edu/data #virality2013](http://carl.cs.indiana.edu/data/#virality2013) Warning: **404 as of 2026-07-28** — use CasFlow's mirror. Paper: [Weng et al., Sci. Rep. 2013](https://www.nature.com/articles/srep02522) | no dead | — |
| **APS** | [journals.aps.org/datasets](https://journals.aps.org/datasets) Warning: 403 to `curl`, loads in a browser; requires a request form | no manual | citation edge list + metadata; cascade = a paper and its citers |
| **Digg 2009** | [ISI/Lerman](https://www.isi.edu/~lerman/downloads/digg2009.html) | yes plain HTTP | vote log `(story, user, timestamp)` + friendship graph |
| **MemeTracker** | [SNAP memetracker9](https://snap.stanford.edu/data/memetracker9.html) | yes | phrase-cluster time series |
| **Tweet-1Mo** | [SNAP SEISMIC page](https://snap.stanford.edu/seismic/) | yes | `(cascade_id, relative_time, n_followers)` — **no graph at all** |
| **Taoke** | [github.com/Lucas-PJ/CasTemp-ALGO](https://github.com/Lucas-PJ/CasTemp-ALGO) | Warning: release-dependent | cascade + node + product features, plus purchase conversions |

### 6.4 Warning: Name collisions — read before quoting any MSLE

Same three names, seven different artefacts. This is §5.7 made concrete.

| Name | Version | Cascades | Who reports it |
| ---- | ------- | -------- | -------------- |
| **Twitter** A | Weng 2013 hashtags, Mar 24 – Apr 25 2012 | 88,440 [verified] | CasFlow, CCGL, MUCas |
| **Twitter** B | Jing et al. own crawl, Mar 1 – Apr 15 2022 | 86,764 [verified] | CasFT |
| **Twitter** C | CTCP's re-preprocessing (cascades before Apr 4) | 19,718 [verified] | CTCP |
| **Twitter** D | CasTemp's re-preprocessing | 67,760 [verified] | CasTemp |
| **Twitter** E | Hodas–Lerman URL diffusion, 2010 | 569 [verified] | Topo-LSTM, FOREST, MS-HGAT |
| **Weibo** A | DeepHawkes release, posts 8 a.m.–6 p.m. | 119,313 [verified] | DeepHawkes, CasCN, VaCas, CasFlow, CasDO, CasFT |
| **Weibo** B | CTCP re-preprocessing | 39,076 [verified] | CTCP |
| **Weibo** C | CasTemp re-preprocessing | 48,693 [verified] | CasTemp |
| **Weibo** D | CoupledGNN's coverage-sampled subset, ≥5 adopters | 3,228 [verified] | CoupledGNN |
| **APS** A | CasFlow preprocessing, papers 1893–1997 | 207,685 [verified] | CasFlow, CasDO, CasFT |
| **APS** B | CTCP preprocessing (papers before 1997) | 48,575 [verified] | CTCP |
| **APS** C | CasTemp preprocessing | 90,768 [verified] | CasTemp |

The **filters** that produce these differences are all stated and all different: CasFlow drops cascades with `|C(t_o)| < 10` and truncates at the first 100 participants [verified]; CasFT drops `< 10` participants [verified]; CoupledGNN drops `< 5` active users [verified]; SEISMIC/Mishra require `≥ 50` retweets [verified]. A `< 10` vs `< 50` threshold alone moves MSLE by more than the gap between any two consecutive rows of §5.1.

### 6.5 What we would actually load

Ranked, cheapest first:

1. **APS** — 616K nodes / 3.3M edges, cascades are citation lists, no rate limits, no Chinese-platform registration. The 3-year/5-year windows are long enough that a *step* can be a year, which maps onto our discrete-time simulator without resampling. **This is the one to do.**
2. **Weibo (DeepHawkes bundle)** — we already fetch the AMiner release for the `weibo` graph; the cascades are one more file. The global graph (6.7M nodes) exceeds our pipeline, but cascade-local graphs do not.
3. **Digg 2009** — small, public, `(story, user, time)` triples, and it doubles as a [`source_localization.md`](source_localization.md) corpus.
4. **Twitter (Weng)** — only via CasFlow's Drive mirror; the canonical host is dead. Avoid unless a Twitter row is specifically needed.

**Not recommended:** Tweet-1Mo (no graph — our model needs one), Facebook photos (not public), Taoke (too new, single-paper).

---

## 7. Which paper uses which

`yes` = reports results on it. Superscript marks *which version* per §6.4.

| Method | Weibo | Twitter | APS | Digg | Memes | Tweet-1Mo / News | Taoke | Other |
| ------ | ----- | ------- | --- | ---- | ----- | ---------------- | ----- | ----- |
| Cheng et al. 2014 | | | | | | | | Facebook photos |
| Weng et al. 2014 | | yes | | | | | | |
| RPP (Shen 2014) | | | | | | | | APS/citations |
| SEISMIC 2015 | | | | | | yes | | |
| Mishra et al. 2016 | | | | | | yes (both) | | |
| HIP 2017 | | | | | | | | YouTube views |
| DeepCas 2017 | yesᴬ | yes | | | | | | |
| DeepHawkes 2017 | yesᴬ | | | | | | | |
| Topo-LSTM 2017 | | yesᴱ | | yes | yes | | | |
| DeepInf 2018 | yes | yes | | yes | | | | OAG |
| CasCN 2019 | yesᴬ | | yes | | | | | |
| CoupledGNN 2020 | yesᴰ | | | | | | | synthetic |
| VaCas 2020 | yesᴬ | yesᴬ | yes | | | | | |
| **CasFlow 2021** (key) | yesᴬ | yesᴬ | yesᴬ | | | | | |
| CCGL 2022 | yesᴬ | yesᴬ | yesᴬ | | | | | |
| MUCas 2022 | yesᴬ | yesᴬ | yesᴬ | | | | | |
| **CTCP 2023** | yesᴮ | yesᶜ | yesᴮ | | | | | |
| CasDO 2024 | yesᴬ | yesᴬ | yesᴬ | | | | | |
| **CasFT 2024** (key) | yesᴬ | yesᴮ | yesᴬ | | | | | |
| **CasTemp 2025** (key) | yesᶜ | yesᴰ | yesᶜ | | | | yes | |

Reading it: **the Weibo-A / Twitter-A / APS-A row is the field's spine**, and CTemp/CTCP each broke it by re-preprocessing. Any table that lists CasFlow, CTCP and CasFT side by side without saying which version it used is wrong.

---

## 8. Evaluation protocol

### 8.1 Metrics

| Metric | Definition | Who uses it | Note |
| ------ | ---------- | ----------- | ---- |
| **MSLE** (key) | `1/M Σ (log₂ P̂ − log₂ P)²` | **everyone since DeepCas** | The field standard. Log base **2**, not `e`, in CasFlow/CasFT; CTCP's loss uses natural log [verified] — a constant factor `(ln2)² ≈ 0.48` between them. Check before comparing. |
| **MALE** | `1/M Σ \|log₂ P̂ − log₂ P\|` | CTCP, CasTemp | L1 version; more robust to the few huge cascades |
| **MAPE** | `1/M Σ \|log₂(P+2) − log₂(P̂+2)\| / log₂(P+2)` | CasFlow, CasFT, CTCP | Warning: **relative error in log space, not on popularity.** Named MAPE, is not MAPE. |
| **MRSE / mRSE** | mean / median `((P̂−P)/P)²` | CoupledGNN | genuinely relative, on raw counts |
| **APE / ARE** | `\|P̂ − P\| / P` | SEISMIC, Mishra | reported as *quantiles* because the mean is outlier-dominated |
| **R²** | coefficient of determination | CasFlow | reported only in figures |
| **COV-k** | \|top-k predicted ∩ top-k true\| / k, `k = ⌊N/10⌋` | CasFlow | "did we find the viral ones" |
| **WroPerc** | fraction with relative error ≥ ε (ε = 0.5) | CoupledGNN | the practitioner's metric |
| **Hits@k / MAP@k** | ranking of the next adopter, k ∈ {10, 50, 100} | Topo-LSTM, FOREST, MS-HGAT | microscopic only |
| **Accuracy / AUC** | binary "will it double" | Cheng et al., DeepInf | balanced by construction |

**Get MSLE right or nothing else matters.** Three independent choices hide in it: (a) log base 2 vs `e`; (b) `P` = total popularity `P(t_p)` vs increment `ΔP = P(t_p) − P(t_o)`; (c) the `+1` / `+2` smoothing constant. CasFlow uses `log₂ ΔP` with no offset [verified]; CasFT uses `log₂(P+1)` [verified]. Those two are *not* the same metric, yet §5.1 prints them in one table.

### 8.2 Observation windows and horizons — the standard settings

| Corpus | `t_o` (the two standard values) | `t_p` | Rationale [verified, CasFlow §5.1] |
| ------ | ------------------------------- | ----- | --------------------------------- |
| **Weibo** | **0.5 h and 1 h** | 24 h | Diurnal rhythm: only posts published 8 a.m.–6 p.m. are kept, so every tweet has ≥ 6 h to accrue retweets |
| **Weibo** (network-aware line) | **1 h, 2 h, 3 h** | final size | CoupledGNN/DeepHawkes convention; no `t_p` because they predict the final count |
| **Twitter** | **1 d and 2 d** | 32 d | hashtags tracked before Apr 10 so each has ≥ 15 d to grow |
| **APS** | **3 y and 5 y** | 20 y | papers published 1893–1997 so each has ≥ 20 y (1997–2017) of citations |
| Tweet-1Mo / News | 5 min, 10 min, 1 h | ∞ | point-process line; prediction starts once a cascade hits 50 retweets |
| Digg / Memes / Twitter-URL | first `k` adopters | next adopter | microscopic line; no clock at all |

The pairing is deliberate: every macroscopic paper reports **two** windows so a reader can see whether a method's edge survives more observation. A single-window result is not publishable in this literature.

### 8.3 Warning: The split is the trap — and it may invalidate a decade of numbers

The standard protocol is **70/15/15 random over cascades** [verified, CasFlow §5.1; CasFT "Datasets and Preprocessing"]. CasTemp (§5.3) argues this **leaks the future**: cascades overlap in wall-clock time, so a training cascade's *prediction* window can sit inside a test cascade's *observation* window. The model then learns "there was a burst around time T" — a global temporal shortcut that is unavailable at deployment.

Their proposed fix: cut the corpus into **four consecutive equal-duration segments** and use `seg1 → seg2` for train, `seg2 → seg3` for val, `seg3 → seg4` for test (1:1:1 in duration, temporally disjoint) [verified]. Segment lengths: 2 d Twitter, 1 h Weibo, 5 y APS, 1 d Taoke [verified].

What changes under the fix [verified, §5.3]:

- CasFlow and CasDO drop **below a plain MLP** on the diagnostic scenarios.
- CasDO — a 2024 TKDE paper — is the *worst* method on all four corpora.
- APS MSLE for the whole field moves from the 1.19–2.11 band (§5.1) to 2.28–4.82 (§5.3). Not a small correction.
- Train-vs-test loss curves show CasFlow/CasDO fitting the training set hard and generalising badly, while CTCP and CasTemp stay flat [verified, Fig 2].

**For us this is the single most transferable lesson in the file.** Our `--seed`-based train/val/test split over transitions (`data/generate_wm_data.py`) splits by *episode*, not by *time*, which is structurally the same choice CasTemp is criticising. It is harmless in our benchmark because every episode is an independent NDlib run with no shared clock — but the moment we replay real logged cascades, it stops being harmless. Any cascade-corpus experiment we run should use the time-ordered split from day one.

### 8.4 Other protocol landmines

| Trap | Detail |
| ---- | ------ |
| **Filter threshold** | `< 10` participants (CasFlow, CasFT) vs `< 5` (CoupledGNN) vs `≥ 50` retweets (SEISMIC, Mishra). Dropping small cascades removes the hardest, most numerous cases and inflates every metric. |
| **Truncation** | CasFlow keeps only the **first 100 participants** of any cascade [verified]. A method that exploits long tails cannot show it. |
| **Unscoreable cascades** | Generative models decline to predict supercritical cascades — SEISMIC failed on 1,022 / ~20K News cascades at 5 min [verified]. Papers report the mean over *scoreable* cascades only, which silently favours the model that gives up more often. Mishra et al. publish the failure counts; almost nobody else does. |
| **Diurnal filtering** | Weibo keeps only 8 a.m.–6 p.m. posts [verified]. A model tested there has never seen an overnight cascade. |
| **Metric direction** | MSLE/MALE/MAPE/MRSE/WroPerc: lower better. R²/PCC/COV-k/Hits@k: higher better. Mixed within single tables. |

### 8.5 Is there a unified public benchmark?

**No.** This review found no equivalent of OGB or TGB for cascade prediction. What exists instead:

| Artefact | What it actually is | Link |
| -------- | ------------------- | ---- |
| **CasFlow's dataset bundle** | The *de facto* standard: one Drive archive with Weibo + Twitter + APS already in the common `cascade_id \t root \t time \t n \t paths` format. Every later paper starts here. **This is the closest thing to a benchmark.** | [github.com/Xovee/casflow](https://github.com/Xovee/casflow) |
| **CCGL** | Ships augmentation + pretraining code over the same three corpora; a *method* repo with benchmark-ish tooling | [github.com/Xovee/ccgl](https://github.com/Xovee/ccgl) |
| **CasTemp-ALGO** | Re-implements six baselines under one leak-free harness — the only multi-method comparison run by one group under one protocol | [github.com/Lucas-PJ/CasTemp-ALGO](https://github.com/Lucas-PJ/CasTemp-ALGO) |
| **DiffusionPapers** | Curated reading list, not code | [github.com/yangchengbupt/DiffusionPapers](https://github.com/yangchengbupt/DiffusionPapers) |

Consequence: there is **no leaderboard to enter and no split to inherit**. If we report a number we choose the preprocessing ourselves, which means the only honest comparison is one we run ourselves against a released baseline. Budget for that, or do not report cross-paper numbers at all.

---

## 9. Implications for this project

1. **This is the falsification test for our IC/LT assumption, and nothing else in this folder is.** Every other task file evaluates a learned model against traces drawn from the same NDlib simulator that trained it — a closed loop that can only measure learning error, never modelling error. Cascade corpora break the loop: the dynamics that produced Weibo retweets are whatever they are, and our structured head's `p_new(v) = 1 − ∏_u (1 − q(u→v)·frontier_u)` is either an adequate approximation of them or it is not. That is a measurable, falsifiable claim. It is the only one of its kind available to us.

2. **The literature has already run a version of this experiment, and the simulator-tuned method lost.** IMINFECTOR's authors report IMM — a near-optimal RIS method tuned against IC simulations — underperforming badly under cascade-based evaluation, attributed to diffusion-model misspecification ([`influence_maximization.md` §5.5](influence_maximization.md)). Our world model inherits exactly that assumption class. §3.2 gives three specific mechanisms by which real cascades violate it: Hawkes self-excitation, repeated exposure, and exogenous arrivals. Expect to lose, and design the experiment so that losing is informative.

3. **As a headline task it is a poor fit, for a structural reason.** `a_t` is `NULL` at every step (§2.1). The world model degenerates to a forecaster, the five action ops go unexercised, `action_sensitivity` is undefined, and conditions 3–6 of the baseline ladder collapse into one arm because there is no decision to plan over. Do not position this as a GWM capability demonstration — it cannot demonstrate the capability the project is about.

4. **It is the concrete instantiation of "logged trajectories".** `research_notes/Baselines - Oracle vs MC vs GWM.md` §4 Q2 answers the chicken-and-egg objection by asserting that in deployment the training data comes from historical observations rather than a simulator. Weibo/APS/Twitter are that assertion made downloadable — `(s_t, NULL, s_{t+1})` transitions with no simulator in the loop. One paragraph reporting a real-cascade number converts a rhetorical answer into an empirical one.

5. **It forces the `w`-hidden ablation the notes already flag as the honest gap.** The same document's §4 wrinkle concedes that our IC heads consume the true transmission probability `w`, and that `structured_residual` literally anchors on `logit(w)` — so the "recovers unknown dynamics" claim currently rests on LT, not IC. **Real cascades have no `w`.** Running on APS is not a variant of the `w`-hidden ablation; it *is* the ablation, with the additional property that the ground-truth dynamics are not ours either.

6. **Adopt the time-ordered split now, everywhere.** §8.3 is the most transferable finding here: a decade of cascade results shifted materially when the random-over-cascades split was replaced by a chronological one, and two recent SOTA models fell below a plain MLP. Our split is by episode (`data/generate_wm_data.py`), which is safe only because NDlib episodes share no clock. That safety evaporates the instant we replay logged data. Build the real-cascade loader with a chronological split from the first commit.

7. **Our rollout metric is already the field's metric, under another name.** `rollout_ensemble` reports `ens_final_count_model` and `ens_final_count_true`; MSLE is `(log₂ model − log₂ true)²` over those two numbers. The evaluation layer is ~2 hours of work (§2.4), not a new subsystem. What is genuinely missing is soft targets: a real cascade happened **once**, so `--mc-marginals` has nothing to average and our targets become hard 0/1. That is a real change to the training signal and the main technical risk in the whole exercise.

8. **The cheapest honest version of the test.** APS only. Load the citation cascades, replay each as `(s_t, NULL, s_{t+1})` with a one-year step, train the existing structured IC head with `w` masked to ones, and report three numbers under a chronological split: our MSLE, a persistence baseline's MSLE, and CasFlow's published 1.4370 at `t_o = 3 y` (§5.1) as a context marker — explicitly *not* as a like-for-like comparison, since our preprocessing will differ (§6.4). Roughly two days of loader plus one training run. Frame the result as "how far does an IC-structured world model degrade when the dynamics are not IC", report `ens_count_bias` alongside MSLE, and put it in the limitations section rather than the results table.

9. **Do not chase the leaderboard.** CasFlow is a 2M-parameter model tuned for this one task [verified, §5 of that paper]; CasFT adds a neural ODE and a diffusion decoder. We will not beat them and should not try. The value of this task to us is diagnostic, exactly as the oracle condition's value is diagnostic — see the same framing in `research_notes/Baselines - Oracle vs MC vs GWM.md` §4 Q1.

---

## 10. Reference list

**Feature-driven and classical** [Szabo & Huberman 2008 (arXiv 0811.0405)](https://arxiv.org/abs/0811.0405) · [Kupavskii 2012 CIKM (ACM 10.1145/2396761.2398634)](https://dl.acm.org/doi/10.1145/2396761.2398634) Warning: ACM 403 to `curl` · [Cui 2013 KDD (ACM 10.1145/2487575.2487639)](https://dl.acm.org/doi/10.1145/2487575.2487639) Warning: ACM 403 · [Cheng 2014 WWW *Can Cascades be Predicted?* (arXiv 1403.4608)](https://arxiv.org/abs/1403.4608) · [Weng 2014 (arXiv 1403.6199)](https://arxiv.org/abs/1403.6199) · [Sci. Rep. 2013](https://www.nature.com/articles/srep02522) · [Martin 2016 WWW *Exploring limits to prediction* (arXiv 1602.01013)](https://arxiv.org/abs/1602.01013)

**Point processes** [Shen 2014 RPP (arXiv 1401.0778)](https://arxiv.org/abs/1401.0778) · [Zhao 2015 SEISMIC (arXiv 1506.02594)](https://arxiv.org/abs/1506.02594) · [CRAN `seismic`](https://cran.r-project.org/package=seismic) · [SNAP data](https://snap.stanford.edu/seismic/) · [Mishra 2016 CIKM (arXiv 1608.04862)](https://arxiv.org/abs/1608.04862) · [code](https://github.com/s-mishra/featuredriven-hawkes) · [Rizoiu 2017 HIP (arXiv 1602.06033)](https://arxiv.org/abs/1602.06033) · [code](https://github.com/andrei-rizoiu/hip-popularity)

**Deep macroscopic** [DeepCas 2017 (arXiv 1611.05373)](https://arxiv.org/abs/1611.05373) · [code](https://github.com/chengli-um/DeepCas) · [DeepHawkes 2017 (ACM 10.1145/3132847.3132973)](https://dl.acm.org/doi/10.1145/3132847.3132973) Warning: ACM 403 · [code](https://github.com/CaoQi92/DeepHawkes) · [CasCN 2019 (NSF-PAR PDF)](https://par.nsf.gov/servlets/purl/10122600) Warning: cert hostname mismatch · [code](https://github.com/ChenNed/CasCN) · [CoupledGNN 2020 (arXiv 1906.09032)](https://arxiv.org/abs/1906.09032) · [code](https://github.com/CaoQi92/CoupledGNN) · [VaCas 2020 (10.1109/INFOCOM41043.2020.9155349)](https://doi.org/10.1109/INFOCOM41043.2020.9155349) Warning: IEEE 202 · [CasFlow 2021 (author PDF)](https://www.xoveexu.com/file/paper/21-11-TKDE-CasFlow.pdf) Warning: cert mismatch · [IEEE 9611000](https://ieeexplore.ieee.org/document/9611000) · [code](https://github.com/Xovee/casflow) · [mirror](https://github.com/kpzhang/casflow) · [CasSeqGCN 2021 (arXiv 2110.06836)](https://arxiv.org/abs/2110.06836) · [code](https://github.com/MrYansong/CasSeqGCN) · [TempCas 2021 (10.1016/j.ipm.2021.102593)](https://doi.org/10.1016/j.ipm.2021.102593) · [CCasGNN 2021 (arXiv 2112.03644)](https://arxiv.org/abs/2112.03644) · [code](https://github.com/MrYansong/CCasGNN) · [MUCas 2022 (10.24963/ijcai.2022/300)](https://doi.org/10.24963/ijcai.2022/300) · [code](https://github.com/ChenNed/MUCas) · [CCGL 2022](https://github.com/Xovee/ccgl) · [CTCP 2023 (arXiv 2306.03756)](https://arxiv.org/abs/2306.03756) · [code](https://github.com/lxd99/CTCP) · [CasDO 2024 (10.1109/TKDE.2024.3465241)](https://doi.org/10.1109/TKDE.2024.3465241) Warning: IEEE 202 · [CasFT 2024 (arXiv 2409.16619)](https://arxiv.org/abs/2409.16619) · [CasTemp / Beyond Leakage 2025 (arXiv 2510.25348)](https://arxiv.org/abs/2510.25348) · [code](https://github.com/Lucas-PJ/CasTemp-ALGO)

**Deep microscopic** [Topo-LSTM 2017 (arXiv 1711.10162)](https://arxiv.org/abs/1711.10162) · [code](https://github.com/vwz/topolstm) · [DeepInf 2018 (arXiv 1807.05560)](https://arxiv.org/abs/1807.05560) · [code](https://github.com/xptree/DeepInf) · [FOREST 2019 (IJCAI)](https://www.ijcai.org/proceedings/2019/560) · [code](https://github.com/albertyang33/FOREST) · [MS-HGAT 2022 (AAAI)](https://ojs.aaai.org/index.php/AAAI/article/view/20334) · [code](https://github.com/slingling/MS-HGAT)

**Surveys** [Zhou, Xu, Trajcevski & Zhang, ACM CSUR 2021 (arXiv 2005.11041)](https://arxiv.org/abs/2005.11041) · [ACM 10.1145/3433000](https://dl.acm.org/doi/10.1145/3433000) Warning: ACM 403 · [Graph representation learning for popularity prediction (arXiv 2203.07632)](https://arxiv.org/abs/2203.07632) · [DiffusionPapers reading list](https://github.com/yangchengbupt/DiffusionPapers)

**Data** [CasFlow bundle — Weibo/Twitter/APS](https://drive.google.com/file/d/1o4KAZs19fl4Qa5LUtdnmNy57gHa15AF-/view) · [DeepHawkes Weibo](https://drive.google.com/file/d/1fgkLeFRYQDQOKPujsmn61sGbJt6PaERF/view) · [APS datasets](https://journals.aps.org/datasets) Warning: 403 to `curl`, browser-only · [AMiner Influence Locality](https://www.aminer.cn/influencelocality) · [ISI/Lerman Digg 2009](https://www.isi.edu/~lerman/downloads/digg2009.html) · [SNAP MemeTracker](https://snap.stanford.edu/data/memetracker9.html) · [Weng Twitter (carl.cs.indiana.edu)](http://carl.cs.indiana.edu/data/#virality2013) Warning: **404 as of 2026-07-28**

---

## 11. Open gaps

Honest list of what this review could **not** establish.

### Tables that could not be extracted

- **CasFlow's own Table 3 (TKDE 2021)** — the headline result table of the reference method is a **raster image** in the published PDF. `pdftotext -layout` returns the caption and the significance footnote only. Every CasFlow number in §5 is therefore a *re-run by a later paper* (CasFT §5.1, CTCP §5.2, CasTemp §5.3), and those three disagree with each other because they use different preprocessing (§6.4). **CasFlow's self-reported numbers are not in this file and cannot be, without OCR or a code run.**
- **CasFlow's Figure 4** (R² and COV-10%) — figure-only, no table. R² and Coverage are named in §8.1 but no cell for them is transcribed anywhere.
- **DeepCas, DeepHawkes, CasCN, VaCas, MUCas, CCGL, TempCas, CasDO self-reported tables** — not transcribed. All appear in this file only as rows in *other* papers' comparison tables. Their own papers may report different (usually better) numbers under their own preprocessing.
- **Cheng et al. Figure 5** (accuracy vs observed `k`) — the shape is described [figure], the per-`k` values are not published as a table.
- **SEISMIC's Figures 7–8** (median APE and rank correlation vs the four baselines over time) — figure-only. The §5.4 SEISMIC row comes from Mishra et al.'s independent re-run, not from SEISMIC's own plots.

### Facts not established

- **Is there a unified public benchmark? No.** §8.5 is the finding: no OGB/TGB analogue exists for cascade prediction. CasFlow's Drive bundle is the de facto standard artefact, CasTemp-ALGO is the only one-group-one-protocol multi-method harness, and neither is a maintained benchmark with a leaderboard. This was searched for specifically; the absence is a result, not a gap in effort.
- **MUCas, TempCas and CasTformer** — MUCas and TempCas are catalogued from their DOIs and from CTCP's baseline descriptions; neither PDF was read. **"CasTformer" could not be identified at all** — no paper by that title was found via Semantic Scholar or arXiv. It may be a mis-remembered name for CasFT, or for one of the several time-embedding cascade-attention papers. Treat as unverified.
- **CasDO has no public code** that this review could locate (GitHub search returned nothing), despite being a 2024 TKDE paper and despite CasTemp re-running it — CasTemp presumably obtained it privately.
- **APS licensing.** `journals.aps.org/datasets` 403s to `curl` and requires a request form. Whether the redistributed CasFlow bundle is licensed for our use was not checked. Do this before building on it — §6.5 recommends APS as the entry point, and a licence problem would invalidate that recommendation.
- **CasTemp's venue and peer-review status.** arXiv 2510.25348 carries an ACM template with placeholder conference metadata ("Conference acronym 'XX, Woodstock, NY"). §5.3 and §8.3 lean heavily on it. Its claims are internally well-evidenced but **not yet independently replicated**, and the leakage finding is strong enough that it deserves a second source before being treated as settled.
- **Whether our structured IC head is competitive on real cascades** — this is the entire point of §9 and it is, of course, unmeasured. No published work applies an IC-structured neural head to Weibo/APS cascade prediction, so there is no prior to anchor an expectation against.
- **Hard-target training.** §2.4 flags that a real cascade happened once, so `--mc-marginals` has nothing to average and our soft targets become 0/1. How much that degrades our one-step `delta_f1` is not established by anything in this literature — they never had soft targets to lose.
