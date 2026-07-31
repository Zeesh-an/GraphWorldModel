# Graph Completion under Incompleteness — Prior Work, Datasets, and Published Results

Graphs in the wild arrive incomplete: node attributes are missing for most nodes, edges were never observed, and the observation window is a time slice rather than the whole history. This file catalogues the four literatures that attack that problem — **node-feature imputation**, **structure completion / graph structure learning (GSL)**, **static link prediction**, and **dynamic link prediction** — with their methods, datasets, protocols and published tables. It is filed under ❌ **poor fit**: §2 argues rigorously why none of it is a headline task for us, and then salvages the one thing that is genuinely useful — _structure incompleteness as a robustness condition_ for our world model and planner.

All URLs verified for HTTP 200 on **2026-07-28** unless annotated otherwise.

---

## 0. Verification policy

| Tier           | Meaning                                                                                                              |
| -------------- | -------------------------------------------------------------------------------------------------------------------- |
| **[verified]** | Read from the paper's own table via text extraction, or from the data repository's own statistics page. Trustworthy. |
| **[derived]**  | Computed by us from a file we downloaded, or reconstructed arithmetically from published splits.                     |
| **[figure]**   | Read off a plotted figure — the paper published no table. Approximate, direction only.                               |
| **[claim]**    | Stated in prose by a paper or a secondary source; not cross-checked against a file or table.                         |

**Automated PDF summarizers hallucinate plausible numbers from these papers.** Every `[verified]` cell below was produced by `curl` → `pdftotext -layout` → transcription from the extracted table. No number in this file came from a WebFetch summary.

**Edge-count convention.** Undirected graphs are quoted as **undirected edges**; directed graphs as **arcs**. This literature is the worst offender in the whole folder for mixing the two — see §6.4, where the _same_ OGB graph is quoted as 61,859,140 and as 123,718,280 edges by two papers that both call it `ogbn-products`, and the ratio is exactly 2.

---

## 1. Task definition

Let `G = (V, E, X)` with adjacency `A ∈ {0,1}^{N×N}` and node features `X ∈ R^{N×F}`. Incompleteness is a **mask**: `M_X ∈ {0,1}^{N×F}` marks which feature entries were observed, `M_A ∈ {0,1}^{N×N}` marks which node pairs were observed. The four variants differ in which mask is non-trivial and in what you are asked to recover.

| #       | Variant                                                 | Given                          | Recover                             | Evaluated by                                                           |
| ------- | ------------------------------------------------------- | ------------------------------ | ----------------------------------- | ---------------------------------------------------------------------- |
| **(a)** | **Feature imputation / learning with missing features** | `A`, `X ⊙ M_X`                 | `X` (or just downstream labels `y`) | node-classification accuracy at a missing rate; sometimes feature RMSE |
| **(b)** | **Structure completion / GSL**                          | `X`, a corrupted or absent `A` | a better `A*`                       | node-classification accuracy on `A*`                                   |
| **(c)** | **Static link prediction**                              | `A ⊙ M_A`, optionally `X`      | the held-out entries of `A`         | AUC / Hits@K / MRR over held-out pairs vs sampled negatives            |
| **(d)** | **Dynamic link prediction**                             | an event stream up to `t`      | which edges appear at `t+1`         | AP / AUC / MRR against negatives                                       |
| **(e)** | **Joint**                                               | `X ⊙ M_X` **and** `A ⊙ M_A`    | both                                | node classification (T2-GNN)                                           |

**Missingness mechanisms.** Almost every paper here uses MCAR — each feature entry is dropped independently with probability `r` (FP's protocol [verified], footnote 5: _"Each entry of the feature matrix is independently missing with a probability equal to the missing rate"_). Two harder mechanisms appear: **structural missingness**, where _whole nodes_ have no attributes at all (SAT, Amer, ITR, MEGAE — the "attribute-missing graph" line), and **biased missingness**, where the drop probability depends on the value. The distinction matters: uniform entry-wise dropping leaves every node with _some_ signal, so propagation-based methods stay strong to 99% missing; whole-node dropping does not.

**Why the field exists.** Two facts drive it. First, real graphs are observed through a partial lens — a social network crawl misses edges, a user profile misses fields. Second, GNNs are unusually good at exploiting the _other_ modality to fill in the missing one: homophily means a node's neighbours predict its features, and feature similarity predicts its edges. The whole literature is the exploration of that trade.

**A vocabulary warning that matters for us.** In this literature, _"add_edge"_ means **"predict that an unobserved edge exists"** — an inference about a static ground truth. In our repo, `add_edge` means **"intervene on the graph and change what happens next"**. The words coincide; the semantics do not. §2.2 makes this precise, because the surface similarity is the single most misleading thing about this task family.

---

## 2. Fit with our methodology — ❌ poor fit, and here is the proof

Our object is `f_θ(G, s_t, a_t) → s_{t+1}` with `s_{t+1} = T_endo(T_exo(s_t, a_t))`. A task earns a slot in this folder if it has **(a)** a node- or edge-level state that evolves, **(b)** an intervention expressible in our five ops, and **(c)** a simulator we can harvest transitions from. Graph completion fails (a) and (b), and (c) is vacuous.

### 2.1 There is no `s_t → s_{t+1}` — `T_endo` has nothing to do

The target of every task in §1 is a **static reconstruction**: the true `X`, the true `A`, or a held-out entry of the true `A`. There is no time index on the label. In (a) and (b) the ground truth exists before the model runs and does not change while it runs; in (c) the "future" edge is a held-out entry of a fixed adjacency, not a state that evolved.

Concretely, our state is `s_t = (infected, frontier)` — channels 0 and 1 of `X` (`world_model/wm_data.py:17`). Neither channel has any counterpart here. Setting them to zero and asking the model to reconstruct `A` reduces `WorldModel.forward` to `head(encoder(X, graph))` on a constant input — i.e. a GNN autoencoder with four dead input channels and an `ICTransmissionHead` whose `frontier_u` gate (`world_model/wm_model.py`, the structured IC head) multiplies everything by zero. **The two components that make our model a world model — the recurrence and the mechanism-structured head — are exactly the two that do nothing on this task.** A plain GAE does it better with less machinery.

### 2.2 None of the five action ops appear

| Our op            | Meaning in our repo                                                          | Nearest thing in this literature                                            | Same?                                            |
| ----------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------ |
| `add_node`        | activate `v` — it spreads next step                                          | —                                                                           | ✗ no analogue                                    |
| `remove_node`     | de-activate / spend `v`                                                      | —                                                                           | ✗ no analogue                                    |
| `add_edge`        | **intervene**: create arc `u→v` with transmission `w`, changing the dynamics | **infer**: predict that `(u,v)` was in the unobserved ground truth          | ✗ **different semantics, same word**             |
| `remove_edge`     | intervene: delete arc `u→v`                                                  | (the _masking_ step of the evaluation protocol, not an action of the model) | ✗ it is the experimenter's move, not the agent's |
| `set_edge_weight` | intervene on `p(u→v)`                                                        | GSL learns a weighted `A*`, but as a _belief_ about the true graph          | ✗ belief, not intervention                       |

The `add_edge` row is the trap. A GSL method that "adds edges" is estimating `P(edge exists | data)`; our `add_edge` **changes the world** and is scored by what the cascade does afterwards. One is an epistemic update, the other is a causal one. There is no counterfactual fork in this literature — nothing corresponding to our `cf_i` branches — because there is nothing to intervene _on_. The action-conditioning that our whole data-generation design exists to supply (`data/wm_actions.py`) has no target here.

### 2.3 Our model never reads node features at all — so (a) cannot help us

This is decisive and it is checkable in four lines of code.

- `world_model/wm_data.py:16-17` — `in_channels = 6`, and the six channels are `ch_infected, ch_frontier, ch_degree, ch_add, ch_remove, ch_edge`. Two are state, one is structure, three are the action bag. **Zero are dataset features.**
- `world_model/wm_data.py:174-180` — `ch_degree` is computed inside `build_features` by counting occurrences in `edge_index` (`np.add.at(degrees, edge_index[0], 1.0)` … `X[:, ch_degree] = np.log1p(degrees)`). It is derived from topology, never read from a file.
- `data/generate_wm_data.py:92-99` — data generation _does_ write `node_feats=bundle.node_feats` and `node_labels=bundle.node_labels` into each `graphs/*.npz`.
- `world_model/wm_data.py:225-241` — `load_graph_store` opens that same `.npz` and reads **only** `edge_index`, `ic_probs`, `lt_weights`. `node_feats` and `node_labels` are never touched. `coding_agent/types.py:33-40` (`GraphInfo.from_store_entry`) likewise reads only `num_nodes`, `edge_index`, `ic_probs`, `directed`.

So the node features every paper in §4.1 exists to impute are **written to disk by our pipeline and read back by nothing**. Imputing them perfectly would change not one number we report. Cora-ML's 2,879-dim bag-of-words is the clearest case: it is the only real feature matrix in our whole suite (§6.3), and it is dead weight on disk.

Two honest caveats. (i) This is a property of the _current_ architecture, not a law — a future variant that conditions transmission on node attributes (topic- aware IC, where `p(u→v)` depends on content similarity) would make feature imputation load-bearing. That variant does not exist in this repo and is not on the roadmap in `CLAUDE.md`. (ii) The `node_feats` write is not a bug worth fixing; it costs a few MB and keeps the graph store self-describing.

### 2.4 What survives: structure incompleteness as a **robustness condition**

The salvageable half. Real influence networks are _observed_, and observation misses edges — a crawl truncates, a privacy setting hides a follow, a contact tracing log drops a contact. Our entire pipeline assumes `A` is the truth. It has never been measured under a hidden-edge budget.

That is a legitimate, cheap experiment, and this literature supplies the protocol off the shelf (§8.1): sample a fraction `r` of undirected edges uniformly at random, delete them, and run the whole pipeline on the residual graph while the **referee still uses the full graph**. Three things to measure:

1. **World-model fidelity under a corrupted `A`** — retrain (or just re-evaluate) on `A_obs` and score `delta_f1`, `ens_count_bias`, `ens_marg_mae` against the MC ground truth computed on the _full_ `A`. Our IC head is per-edge (`q(u→v) = σ(MLP([h_u, h_v, w_uv]))`), so hidden edges are missing transmission channels — the prediction should be _under_-confident, i.e. `ens_count_bias` should go **negative**, the opposite failure mode from the saturation we spent months fixing (`MEMORY.md`, rollout-saturation entry). That sign prediction is itself a falsifiable check.
2. **Planner regret under a corrupted `A`** — `planning_regret_multi` with seeds chosen on `A_obs` but spread measured on `A_full`. This is the number that matters for the paper: how much spread do we lose per 10% of hidden edges?
3. **Degree-baseline crossover** — degree is computed from `A_obs` too, so it degrades as well. The interesting question is whether our margin over degree _widens_ (the model's learned structure priors compensate) or _collapses_.

Budgeted at `r ∈ {0, 0.1, 0.2, 0.5}` over the small graphs (jazz, netscience, cora_ml, power_grid) this is four extra data-gen runs per graph and no new code beyond an edge-dropping flag. It converts a hidden assumption into a table.

### 2.5 The overlap that _does_ fit us lives in another file

Recovering edges **from cascades** — observing who got infected when and inferring the diffusion network — is real structure completion that is also action-conditioned and dynamic. That is **[`network_inference.md`](network_inference.md)** (NETINF / NETRATE / MultiTree / InfoPath), rated ⚠️ moderate rather than ❌. The difference is the evidence: this file's methods complete `A` from `X` and the observed part of `A`; network inference completes `A` from _cascade traces_, which is exactly the data our simulator produces. If you came here looking for "can the world model recover missing edges", go there.

**Verdict.** Keep this file as (i) the written record of why feature imputation and link prediction are not our task, (ii) the source of the masking protocol for the robustness experiment in §2.4, and (iii) the Cora name-collision forensics in §6.3. Do not build a headline result on it.

---

## 3. Classical and heuristic methods

### 3.1 Link-prediction heuristics

`Γ(x)` = neighbours of `x`. Order = how many hops of the neighbourhood the score reads; SEAL's Table 3 is the canonical listing [verified].

| Method                           | Year | Formula                              | Order  | Paper                                                                                                 |
| -------------------------------- | ---- | ------------------------------------ | ------ | ----------------------------------------------------------------------------------------------------- |
| **Common Neighbors (CN)**        | 1953 | `\|Γ(x) ∩ Γ(y)\|`                    | first  | [Newman 2001 / Liben-Nowell & Kleinberg 2003](https://www.cs.cornell.edu/home/kleinber/link-pred.pdf) |
| **Jaccard**                      | 1901 | `\|Γ(x)∩Γ(y)\| / \|Γ(x)∪Γ(y)\|`      | first  | same                                                                                                  |
| **Preferential Attachment (PA)** | 1999 | `\|Γ(x)\| · \|Γ(y)\|`                | first  | same                                                                                                  |
| **Adamic–Adar (AA)**             | 2003 | `Σ_{z∈Γ(x)∩Γ(y)} 1/log\|Γ(z)\|`      | second | [Adamic & Adar](https://www.sciencedirect.com/science/article/abs/pii/S0378873303000091)              |
| **Resource Allocation (RA)**     | 2009 | `Σ_{z∈Γ(x)∩Γ(y)} 1/\|Γ(z)\|`         | second | [Zhou, Lü & Zhang](https://arxiv.org/abs/0901.0553)                                                   |
| **Katz**                         | 1953 | `Σ_{l≥1} β^l · \|walks^{(l)}(x,y)\|` | high   | [Katz 1953](https://link.springer.com/article/10.1007/BF02289026)                                     |
| **PageRank (PR)**                | 1998 | rooted PageRank, damping α           | high   | —                                                                                                     |
| **SimRank (SR)**                 | 2002 | recursive structural similarity, γ   | high   | [Jeh & Widom](https://dl.acm.org/doi/10.1145/775047.775126) (403 to non-browser clients)              |

SEAL's hyperparameters when reproducing these [verified]: Katz `β = 0.001`, PageRank `α = 0.85`, SimRank `γ = 0.8`. **These heuristics are not obsolete** — BUDDY's Table 2 (§5.7) shows RA beating a plain GCN on `ogbl-collab` (64.00 vs 47.14 Hits@50) and on `ogbl-ppa` (49.33 vs 18.67 Hits@100) [verified].

### 3.2 Feature-imputation baselines

The baselines every §4.1 paper must beat. All are one-liners; the point of the FP paper is that the best of them is nearly unbeatable.

| Baseline                   | Rule                                                    | Note                                                                                                                 |
| -------------------------- | ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Zero**                   | missing entries ← 0                                     | FP: at ≤50% missing this loses almost nothing — _"node features are redundant"_ [verified, prose]                    |
| **Mean / global**          | missing ← column mean                                   |                                                                                                                      |
| **Neighbor Mean**          | missing ← mean over observed neighbours                 | first-order approximation of Feature Propagation; FP reports it beats GCNMF and PaGNN consistently [verified, prose] |
| **Label Propagation (LP)** | ignore features entirely, propagate labels              | feature-agnostic; the honest floor                                                                                   |
| **Positional Encodings**   | replace `X` with a structural embedding                 | feature-agnostic                                                                                                     |
| **kNN graph**              | build `A` from feature similarity, discard the real `A` | the GSL floor (`kNN-GCN` in §5.3/§5.4)                                                                               |
| **Matrix completion**      | low-rank `X` or `A` (SVD, libFM)                        | GCN-SVD in Pro-GNN; MF in SEAL                                                                                       |

---

## 4. Learning-based methods

### 4.1 Missing node features / attribute imputation

| Method                          | Year      | Venue     | Idea                                                                                                                                                                                                                                                                               | Paper                                                                                                                 | Code                                                                                            |
| ------------------------------- | --------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **GCNmf**                       | 2021      | FGCS      | Represent missing entries with a Gaussian Mixture Model; compute the _expected_ activation of GCN's first hidden layer analytically. End-to-end, no separate imputation step.                                                                                                      | [arXiv 2007.04583](https://arxiv.org/abs/2007.04583) · [FGCS](https://doi.org/10.1016/j.future.2020.11.016)           | [marblet/GCNmf](https://github.com/marblet/GCNmf)                                               |
| **PaGNN**                       | 2020→2024 | IEEE TAI  | Partial message passing: propagate only over observed entries, with a "partial aggregation" that renormalizes by how much was observed.                                                                                                                                            | [arXiv 2003.10130](https://arxiv.org/abs/2003.10130)                                                                  | no public code found                                                                            |
| **SAT**                         | 2022      | TPAMI     | _Structure-Attribute Transformer._ Shared-latent-space distribution matching between the structure branch and the attribute branch; built for **structurally** missing attributes (whole nodes with no features).                                                                  | [arXiv 2011.01623](https://arxiv.org/abs/2011.01623)                                                                  | [xuChenSJTU/SAT-master-online](https://github.com/xuChenSJTU/SAT-master-online)                 |
| **WGNN**                        | 2021      | arXiv     | Represent each node as a _distribution_ from attribute-matrix decomposition; aggregate neighbours in Wasserstein space; adversarial loss with gradient penalty.                                                                                                                    | [arXiv 2102.03450](https://arxiv.org/abs/2102.03450)                                                                  | no public code found                                                                            |
| **Amer**                        | 2022      | IEEE TCYB | Complete attributes _and_ learn the embedding in one model, coupled by mutual-information maximization rather than run in sequence.                                                                                                                                                | [IEEE 9765782](https://ieeexplore.ieee.org/document/9765782) (returns 202 to non-browser clients)                     | no public code found                                                                            |
| **ITR**                         | 2022      | IJCAI     | _Initializing Then Refining._ Structure-based initial imputation, then adaptive refinement using observed attributes **and** structure.                                                                                                                                            | [IJCAI 2022/485](https://www.ijcai.org/proceedings/2022/485) · [PDF](https://www.ijcai.org/proceedings/2022/0485.pdf) | [WxTu/ITR](https://github.com/WxTu/ITR)                                                         |
| **RITR**                        | 2024      | TNNLS     | Journal extension of ITR to jointly incomplete _and_ missing attributes.                                                                                                                                                                                                           | [arXiv 2302.07524](https://arxiv.org/abs/2302.07524)                                                                  | [WxTu/RITR](https://github.com/WxTu/RITR)                                                       |
| **MEGAE**                       | 2023      | AAAI      | _Max-Entropy regularized Graph AutoEncoder._ Diagnoses **spectral concentration** as the failure mode of GAE imputation; wavelet-based encoder + maximum graph-spectral-entropy regularizer.                                                                                       | [arXiv 2211.16771](https://arxiv.org/abs/2211.16771)                                                                  | [zqgao22/max-entropy-gae](https://github.com/zqgao22/max-entropy-gae)                           |
| **FP (Feature Propagation)** ⭐ | 2022      | LoG       | Minimize Dirichlet energy over the unknown entries → a diffusion PDE on the graph; discretize → iterate `X ← ÃX` while resetting known rows. ~40 lines, no parameters, 10 s on 2.5M nodes.                                                                                         | [arXiv 2111.12128](https://arxiv.org/abs/2111.12128) · [PMLR v198](https://proceedings.mlr.press/v198/rossi22a.html)  | [twitter-research/feature-propagation](https://github.com/twitter-research/feature-propagation) |
| **PCFI** ⭐                     | 2023      | ICLR      | Adds _channel-wise pseudo-confidence_ (shortest-path distance to the nearest known-feature node per channel) on top of FP-style diffusion; survives 99.5% missing.                                                                                                                 | [arXiv 2305.16618](https://arxiv.org/abs/2305.16618)                                                                  | [daehoum1/pcfi](https://github.com/daehoum1/pcfi)                                               |
| **T2-GNN** ⭐                   | 2023      | AAAI      | Two _separate_ teachers — a feature teacher (MLP on observed features) and a structure teacher (GCN on a PPR-enhanced graph) — distilled into one student, so the two incompletenesses never interfere. The only method here that targets **joint** feature+structure missingness. | [arXiv 2212.12738](https://arxiv.org/abs/2212.12738)                                                                  | [jindi-tju/T2-GNN](https://github.com/jindi-tju/T2-GNN)                                         |
| **AmGCL**                       | 2023      | arXiv     | Dirichlet-energy-based feature imputation + self-supervised contrastive objective.                                                                                                                                                                                                 | [arXiv 2305.03741](https://arxiv.org/abs/2305.03741)                                                                  | no public code found                                                                            |
| **FairAC**                      | 2023      | ICLR      | Attribute completion with a fairness constraint on the completed attributes.                                                                                                                                                                                                       | [arXiv 2302.12977](https://arxiv.org/abs/2302.12977)                                                                  | —                                                                                               |
| **AttriReBoost**                | 2025      | arXiv     | Gradient-free propagation optimization for the cold-start (whole-node-missing) case.                                                                                                                                                                                               | [arXiv 2501.00743](https://arxiv.org/abs/2501.00743)                                                                  | —                                                                                               |
| **CGAI**                        | 2025      | ACM MM    | Clustering-oriented _generative_ attribute-graph imputation.                                                                                                                                                                                                                       | [arXiv 2507.19085](https://arxiv.org/abs/2507.19085)                                                                  | —                                                                                               |
| **Divide-Then-Rule**            | 2025      | arXiv     | Cluster-driven hierarchical interpolator for attribute-missing graphs.                                                                                                                                                                                                             | [arXiv 2507.10595](https://arxiv.org/abs/2507.10595)                                                                  | —                                                                                               |

**Diffusion-model-based imputation (2023–2026).** The denoising-diffusion line has largely landed on _tabular_ and _spatiotemporal_ data rather than node attributes: **DiffPuter** ([arXiv 2405.20690](https://arxiv.org/abs/2405.20690), EM + diffusion for tabular missing data) and [arXiv 2407.02549](https://arxiv.org/abs/2407.02549) (tabular imputation + synthesis) are the two most cited. Graph-native diffusion imputation exists but is thin and recent: **DDFI** ([arXiv 2512.06356](https://arxiv.org/abs/2512.06356), two-step reconstruction with diffusion-style feature propagation) and **FSD-CAP** (fractional subgraph diffusion with class-aware propagation, OpenReview 2026). Treat this sub-line as **[claim]** — no table in it was extracted for this review. The reliable survey-level statement is that on the standard MCAR benchmarks nothing has displaced FP/PCFI, whose numbers are in §5.1.

**Surveys.** [Incomplete Graph Learning: A Comprehensive Survey (arXiv 2502.12412)](https://arxiv.org/abs/2502.12412) is the current one and covers both feature and structure incompleteness; the companion reading list [cherry-a11y/Incomplete-graph-learning](https://github.com/cherry-a11y/Incomplete-graph-learning) is maintained and was used to cross-check the code links above.

### 4.2 Structure completion / graph structure learning (GSL)

Two scenarios, and papers are careless about which they are in: **structure refinement** (a real `A` exists, possibly corrupted, learn a better `A*`) and **structure inference** (no `A` at all — learn one from `X`). SUBLIME's Table 1 vs Table 2 is the cleanest split of the two [verified] (§5.4).

| Method         | Year | Venue   | Idea                                                                                                                                                                                                                            | Paper                                                                                                                                                                | Code                                                            |
| -------------- | ---- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **LDS**        | 2019 | ICML    | Treat each entry of `A` as a Bernoulli parameter; **bilevel** optimization — inner loop trains the GCN, outer loop trains the graph. Founding paper of the area; `O(N²)` parameters, so it OOMs above ~5K nodes.                | [PMLR v97](https://proceedings.mlr.press/v97/franceschi19a.html) · [arXiv 1903.11960](https://arxiv.org/abs/1903.11960)                                              | [lucfra/LDS-GNN](https://github.com/lucfra/LDS-GNN)             |
| **Pro-GNN**    | 2020 | KDD     | Jointly learn `S ≈ A` and the GNN, with **low-rank** (nuclear norm), **sparsity** (ℓ1) and **feature-smoothness** (`tr(XᵀL̂X)`) priors. Framed as adversarial defence; the machinery is generic structure denoising.             | [KDD](https://dl.acm.org/doi/10.1145/3394486.3403049) (403 to non-browser clients) · [arXiv 2005.10203](https://arxiv.org/abs/2005.10203)                            | [ChandlerBang/Pro-GNN](https://github.com/ChandlerBang/Pro-GNN) |
| **IDGL**       | 2020 | NeurIPS | _Iterative_ deep graph learning: alternate between refining `A` by multi-head metric learning on embeddings and refining embeddings on the new `A`, until a stopping criterion. Anchor approximation gives near-linear scaling. | [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/e05c7ba4e087beea9410929698dc41a6-Abstract.html) · [arXiv 2006.13009](https://arxiv.org/abs/2006.13009) | [hugochan/IDGL](https://github.com/hugochan/IDGL)               |
| **SLAPS**      | 2021 | NeurIPS | Diagnoses the **supervision starvation** problem: with only node labels, most learned edges get no gradient. Fix = a _self-supervised_ denoising-autoencoder task on `X` that supervises the graph generator.                   | [arXiv 2102.05034](https://arxiv.org/abs/2102.05034)                                                                                                                 | [BorealisAI/SLAPS-GNN](https://github.com/BorealisAI/SLAPS-GNN) |
| **SUBLIME**    | 2022 | WWW     | Fully **unsupervised** GSL — no labels at all. Contrastive alignment between a learner view and an anchor view, plus a structure-bootstrapping schedule that slowly moves the anchor.                                           | [ACM](https://dl.acm.org/doi/10.1145/3485447.3512186) (403 to non-browser clients) · [arXiv 2201.06367](https://arxiv.org/abs/2201.06367)                            | [GRAND-Lab/SUBLIME](https://github.com/GRAND-Lab/SUBLIME)       |
| **NodeFormer** | 2022 | NeurIPS | All-pair message passing via kernelized Gumbel-softmax attention in **`O(N)`**; the learned attention _is_ the learned structure. Scales GSL to 2M nodes; runs with no input graph at all.                                      | [OpenReview](https://openreview.net/forum?id=sMezXGG5So) · [arXiv 2306.08385](https://arxiv.org/abs/2306.08385)                                                      | [qitianwu/NodeFormer](https://github.com/qitianwu/NodeFormer)   |
| **GRCN**       | 2020 | ECML    | Graph-revised convolution: predict a residual adjacency and add it to `A`.                                                                                                                                                      | —                                                                                                                                                                    | —                                                               |
| **GEN**        | 2021 | WWW     | Bayesian graph estimation with an observation model over the given `A`.                                                                                                                                                         | —                                                                                                                                                                    | —                                                               |
| **PTDNet**     | 2021 | WSDM    | Learn to _drop_ task-irrelevant edges (denoising rather than completion).                                                                                                                                                       | —                                                                                                                                                                    | —                                                               |
| **GDC**        | 2019 | NeurIPS | Graph diffusion convolution — replace `A` with a PPR/heat-kernel diffusion, sparsified. A fixed, unlearned `A*`.                                                                                                                | [arXiv 1911.05485](https://arxiv.org/abs/1911.05485)                                                                                                                 | —                                                               |

**Surveys.** [Zhu et al., _A Survey on Graph Structure Learning: Progress and Opportunities_ (arXiv 2103.03036)](https://arxiv.org/abs/2103.03036) is the standard taxonomy (metric-based / neural / direct-optimization graph learners); Fatemi et al.'s SLAPS paper carries the sharpest _diagnostic_ contribution (supervision starvation) even though it is not a survey.

### 4.3 Static link prediction

| Method              | Year | Venue     | Idea                                                                                                                                                                                                                                                                  | Paper                                                                                                                                                                | Code                                                                            |
| ------------------- | ---- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **WLNM**            | 2017 | KDD       | Weisfeiler-Lehman Neural Machine — encode the enclosing subgraph as a fixed adjacency and feed an MLP.                                                                                                                                                                | —                                                                                                                                                                    | —                                                                               |
| **SEAL** ⭐         | 2018 | NeurIPS   | Extract the `h`-hop **enclosing subgraph** around `(u,v)`, apply _double-radius node labeling_, classify with a GNN. Proves the γ-decaying-heuristic theory: all first/second-order heuristics are approximable from small enclosing subgraphs. The reference method. | [NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/53f0d7c537d99b3824f0f99d62ea2428-Abstract.html) · [arXiv 1802.09691](https://arxiv.org/abs/1802.09691) | [muhanzhang/SEAL](https://github.com/muhanzhang/SEAL)                           |
| **Neo-GNN**         | 2021 | NeurIPS   | Learn a _generalized_ neighborhood-overlap score from the adjacency, and add it to a GNN's pairwise score — heuristics and GNNs in one model, without per-pair subgraph extraction.                                                                                   | [arXiv 2206.04216](https://arxiv.org/abs/2206.04216)                                                                                                                 | [seongjunyun/Neo-GNNs](https://github.com/seongjunyun/Neo-GNNs)                 |
| **ELPH / BUDDY** ⭐ | 2023 | ICLR      | Replace SEAL's explicit subgraphs with **MinHash + HyperLogLog sketches** of the neighbourhoods, giving subgraph-level expressivity at message-passing cost. BUDDY precomputes the sketches, so inference is a lookup.                                                | [OpenReview](https://openreview.net/forum?id=m1oqEOAozQU) · [arXiv 2209.15486](https://arxiv.org/abs/2209.15486)                                                     | [melifluos/subgraph-sketching](https://github.com/melifluos/subgraph-sketching) |
| **NBFNet**          | 2021 | NeurIPS   | Neural Bellman-Ford — path-based reasoning; strong on knowledge graphs, OOMs on large OGB link tasks (§5.7).                                                                                                                                                          | [arXiv 2106.06935](https://arxiv.org/abs/2106.06935)                                                                                                                 | —                                                                               |
| **GAE / VGAE**      | 2016 | NeurIPS-W | Inner-product decoder over GCN embeddings. The baseline everything is measured against.                                                                                                                                                                               | [arXiv 1611.07308](https://arxiv.org/abs/1611.07308)                                                                                                                 | —                                                                               |

**Benchmark.** The [OGB link-property-prediction leaderboards](https://ogb.stanford.edu/docs/leader_linkprop/) are where this sub-field actually competes (`ogbl-ppa`, `ogbl-collab`, `ogbl-ddi`, `ogbl-citation2`, `ogbl-wikikg2`, `ogbl-biokg`). Dataset docs: [ogb.stanford.edu/docs/linkprop](https://ogb.stanford.edu/docs/linkprop/).

### 4.4 Dynamic link prediction (adjacency completion over time)

| Method           | Year | Venue       | Idea                                                                                                                                                                                                                                                                                   | Paper                                                                                                            | Code                                                                                                                                                 |
| ---------------- | ---- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **JODIE**        | 2019 | KDD         | Two coupled RNNs (user / item) plus a _projection_ operator that extrapolates an embedding trajectory to a future time. Bipartite interaction networks.                                                                                                                                | [arXiv 1908.01207](https://arxiv.org/abs/1908.01207)                                                             | [srijankr/jodie](https://github.com/srijankr/jodie)                                                                                                  |
| **DyRep**        | 2019 | ICLR        | Two-time-scale temporal point process — _association_ (topology) and _communication_ (interaction) events drive a recurrent node-state update.                                                                                                                                         | [OpenReview](https://openreview.net/forum?id=HyePrhR5KX)                                                         | no public code found (author code not released)                                                                                                      |
| **TGAT**         | 2020 | ICLR        | Self-attention over the temporal neighbourhood with a **functional time encoding** (Bochner). No memory; inductive by construction.                                                                                                                                                    | [OpenReview](https://openreview.net/forum?id=rJeW1yHYwH) · [arXiv 2002.07962](https://arxiv.org/abs/2002.07962)  | [Inductive-representation-learning-on-temporal-graphs](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs) |
| **TGN** ⭐       | 2020 | ICML-W      | Generalizes JODIE/DyRep/TGAT: per-node **memory** + message function + one attention layer. Raw Message Store prevents future leakage. Already reviewed in `research_notes/Literature Review.md` §TGN — that entry is consistent with what is below and adds the architectural detail. | [arXiv 2006.10637](https://arxiv.org/abs/2006.10637)                                                             | [twitter-research/tgn](https://github.com/twitter-research/tgn)                                                                                      |
| **CAWN**         | 2021 | ICLR        | Causal Anonymous Walks — sample temporal walks, anonymize node identities into relative-position encodings, encode with an RNN. Fully inductive.                                                                                                                                       | [OpenReview](https://openreview.net/forum?id=KYPz4YsCPj) · [arXiv 2101.05974](https://arxiv.org/abs/2101.05974)  | [snap-stanford/CAW](https://github.com/snap-stanford/CAW)                                                                                            |
| **EdgeBank**     | 2022 | NeurIPS D&B | _Not a model_ — a memorization heuristic: predict an edge iff it was seen before (∞) or within a time window (tw). Exposed that the standard random-negative protocol was too easy. Any dynamic-LP claim that does not beat EdgeBank is not a claim.                                   | [arXiv 2207.10128](https://arxiv.org/abs/2207.10128)                                                             | [fpour/DGB](https://github.com/fpour/DGB)                                                                                                            |
| **GraphMixer**   | 2023 | ICLR        | _"Do we really need complicated model architectures for temporal networks?"_ — a fixed (non-learned) time encoding + MLP-Mixer link encoder + neighbour mean-pooling. Beats several attention models.                                                                                  | [OpenReview](https://openreview.net/forum?id=ayPPc0SyLv1) · [arXiv 2302.11636](https://arxiv.org/abs/2302.11636) | [CongWeilin/GraphMixer](https://github.com/CongWeilin/GraphMixer)                                                                                    |
| **DyGFormer** ⭐ | 2023 | NeurIPS D&B | Transformer over the _concatenated_ 1-hop histories of both endpoints, with a **neighbor co-occurrence encoding** and history patching for long sequences. Best average rank in DyGLib (§5.8).                                                                                         | [arXiv 2303.13047](https://arxiv.org/abs/2303.13047)                                                             | [yule-BUAA/DyGLib](https://github.com/yule-BUAA/DyGLib)                                                                                              |
| **TCL**          | 2021 | arXiv       | Transformer with contrastive pretraining on temporal graphs.                                                                                                                                                                                                                           | [arXiv 2105.07944](https://arxiv.org/abs/2105.07944)                                                             | —                                                                                                                                                    |
| **NAT**          | 2022 | LoG         | Neighborhood-aware temporal network — dictionary-style per-node caches of recent neighbours.                                                                                                                                                                                           | [arXiv 2209.01084](https://arxiv.org/abs/2209.01084)                                                             | —                                                                                                                                                    |

**The two benchmarks that matter.**

- **DyGLib** ([yule-BUAA/DyGLib](https://github.com/yule-BUAA/DyGLib)) — one training pipeline, 13 datasets, 9 methods, three negative-sampling strategies (random / historical / inductive). Re-ran everything, so its numbers supersede the originals. §5.8.
- **TGB** — _Temporal Graph Benchmark_, NeurIPS 2023 D&B ([arXiv 2307.01026](https://arxiv.org/abs/2307.01026), [tgb.complexdatalab.com](https://tgb.complexdatalab.com/), [shenyangHuang/TGB](https://github.com/shenyangHuang/TGB)). **The current standard.** Nine datasets up to 67M edges, an automated leaderboard, and — the substantive contribution — a _hard_ negative-sampling protocol that replaced the trivially-easy random negatives everyone had been using. Under it, ranking changes: AP near 0.99 becomes MRR near 0.4. §5.9.

---

## 5. Published results

### 5.1 Feature Propagation (LoG 2022) ⭐ the headline of §4.1

Protocol: features dropped **MCAR entry-wise** at rate `r`; FP reconstructs; a plain GCN then does semi-supervised node classification. Metric = accuracy.

**Table 1 — FP(+GCN) vs the same GCN with all features** [verified]:

| Dataset       | Full features | 50% missing         | 90% missing         | 99% missing         |
| ------------- | ------------- | ------------------- | ------------------- | ------------------- |
| Cora          | 80.39%        | 79.70% (−0.86%)     | 79.77% (−0.77%)     | 78.22% (−2.70%)     |
| CiteSeer      | 67.48%        | 65.74% (−2.57%)     | 65.57% (−2.82%)     | 65.40% (−3.08%)     |
| PubMed        | 77.36%        | 76.68% (−0.89%)     | 75.85% (−1.96%)     | 74.29% (−3.97%)     |
| Photo         | 91.73%        | 91.29% (−0.48%)     | 89.48% (−2.46%)     | 87.73% (−4.36%)     |
| Computers     | 85.65%        | 84.77% (−1.04%)     | 82.71% (−3.43%)     | 80.94% (−5.51%)     |
| OGBN-Arxiv    | 72.22%        | 71.42% (−1.10%)     | 70.47% (−2.43%)     | 69.09% (−4.33%)     |
| OGBN-Products | 78.70%        | 77.16% (−1.96%)     | 75.94% (−3.51%)     | 74.94% (−4.78%)     |
| **Average**   | **79.08%**    | **78.11% (−1.27%)** | **77.11% (−2.48%)** | **75.80% (−4.10%)** |

**Table 2 — at 99% missing, against the competition** [verified]:

| Dataset       | GCNMF      | PaGNN      | Label Prop. | Pos. Enc.      | **FP**         |
| ------------- | ---------- | ---------- | ----------- | -------------- | -------------- |
| Cora          | 34.54±2.07 | 58.03±0.57 | 74.68±0.36  | 76.33±0.26     | **78.22±0.32** |
| CiteSeer      | 30.65±1.12 | 46.02±0.58 | 64.60±0.40  | **65.87±0.37** | 65.40±0.54     |
| PubMed        | 39.80±0.25 | 54.25±0.70 | 73.81±0.56  | 73.70±0.29     | **74.29±0.55** |
| Photo         | 29.64±2.78 | 85.41±0.28 | 83.45±0.94  | 83.45±0.26     | **87.73±0.27** |
| Computers     | 30.74±1.95 | 77.91±0.33 | 74.48±0.61  | 75.77±0.47     | **80.94±0.37** |
| OGBN-Arxiv    | OOM        | 53.98±0.08 | 67.56±0.00  | 65.08±0.04     | **69.09±0.06** |
| OGBN-Products | OOM        | OOM        | 74.42±0.00  | OOM            | **74.94±0.07** |

**Reading it.** (i) Relative degradation at 99% missing: GCNMF −58.33%, PaGNN −21.25%, FP −4.12% [verified, prose]. (ii) The two _feature-agnostic_ baselines (Label Prop., Positional Encodings) beat both learned imputers on five of seven datasets — a blunt statement that the learned methods were adding negative value. (iii) The Zero baseline loses almost nothing up to 50% missing, which the authors read as _"node features are redundant"_. **That last point is the one with a bearing on us: on a homophilous graph, topology already carries most of what the features say — which is consistent with our model reading only `log1p(degree)` and still working (§2.3).**

### 5.2 T2-GNN (AAAI 2023) — the joint feature+structure table

Protocol: drop a fraction of **both** features and edges; report node classification accuracy. Eight datasets (Table 1 of that paper, transcribed in §6.4 below).

**Table 3 — comparison at the paper's default missing setting** [verified]:

| Method          | Texas     | Cornell | Wisconsin | Chameleon | Cora      | CiteSeer | Squirrel | PubMed    | avg       |
| --------------- | --------- | ------- | --------- | --------- | --------- | -------- | -------- | --------- | --------- |
| GCN             | 46.48     | 48.90   | 50.39     | 58.37     | 82.77     | 68.84    | 40.09    | 83.51     | 59.90     |
| SAT             | 63.89     | 76.68   | 67.49     | 56.57     | 64.59     | 45.87    | 35.23    | —         | 58.61     |
| GCNMF           | 57.28     | 55.43   | 53.02     | 47.75     | 83.08     | 71.75    | 30.56    | 67.08     | 58.24     |
| IDGL            | 62.16     | 54.05   | 60.78     | 48.68     | 79.88     | 65.80    | 33.11    | 82.60     | 60.88     |
| PTDNet          | 61.62     | 64.87   | 73.14     | 48.42     | 74.93     | 72.29    | 30.45    | 82.49     | 63.52     |
| singleT         | 53.81     | 54.76   | 57.10     | 55.93     | 80.19     | 67.22    | 40.32    | 81.03     | 61.29     |
| T2-GCN (online) | 60.53     | 74.01   | 69.42     | 54.22     | 80.11     | 65.31    | 45.19    | 83.04     | 66.47     |
| **T2-GCN**      | **65.40** | 67.29   | 73.39     | **60.81** | **83.84** | 72.32    | 44.33    | **84.85** | **69.02** |

`—` = OOM.

**Table 4 — average accuracy over the eight datasets vs missing rate** [verified]:

| Missing rate | 0%        | 20%       | 40%       | 60%       | 80%       |
| ------------ | --------- | --------- | --------- | --------- | --------- |
| GCN          | 62.33     | 60.82     | 57.85     | 52.43     | 45.17     |
| **T2-GCN**   | **71.30** | **69.93** | **66.54** | **60.44** | **54.60** |

**The finding worth carrying over:** plain GCN beats _both_ the feature-completion methods (SAT, GCNMF) _and_ the structure-enhancement methods (IDGL, PTDNet) on many cells. The authors' explanation is that those methods assume feature↔ structure coupling helps, but when _both_ are corrupted the coupling propagates the corruption. Under joint incompleteness, **doing nothing beats doing the wrong repair.** For our §2.4 experiment this is a direct warning: the correct control arm is "train on the corrupted graph, change nothing else", not "add a structure-repair module".

### 5.3 SLAPS (NeurIPS 2021) — structure inference from features alone

Cora / CiteSeer here are the **Planetoid** graphs with only 20 labels per class; `Cora390` / `Citeseer370` are the larger-label-budget variants from LDS. Accuracy [verified, Table 1]:

| Model                       | Cora         | CiteSeer     | Cora390  | Citeseer370  | PubMed   | ogbn-arxiv |
| --------------------------- | ------------ | ------------ | -------- | ------------ | -------- | ---------- |
| MLP                         | 56.1±1.6     | 56.7±1.7     | 65.8±0.4 | 67.1±0.5     | 71.4±0.0 | 54.7±0.1   |
| LP                          | 37.6±0.0     | 23.2±0.0     | 36.2±0.0 | 29.1±0.0     | 41.3±0.0 | OOM        |
| kNN-GCN                     | 66.5±0.4     | 68.3±1.3     | 72.5±0.5 | 71.8±0.8     | 70.4±0.4 | 49.1±0.3   |
| LDS                         | —            | —            | 71.5±0.8 | 71.5±1.1     | OOM      | OOM        |
| GRCN                        | 67.4±0.3     | 67.3±0.8     | 71.3±0.9 | 70.9±0.7     | 67.3±0.3 | OOM        |
| IDGL                        | 70.9±0.6     | 68.2±0.6     | 73.4±0.5 | 72.7±0.4     | 72.3±0.4 | OOM        |
| **SLAPS (MLP-D)**           | **73.4±0.3** | **72.6±0.6** | 75.1±0.5 | **73.9±0.4** | 73.1±0.7 | 52.9±0.1   |
| SLAPS (MLP) + self-training | 74.2±0.5     | 73.1±1.0     | 75.5±0.7 | 73.3±0.6     | 74.3±1.4 | NA         |

OOM = out of memory, OOT = out of time (24 h), NA = not applicable.

### 5.4 SUBLIME (WWW 2022) — how much is the true graph worth?

The single most useful pair of tables in this file for our purposes, because it prices the graph. Same models, same datasets; only the availability of `A` changes.

**Table 1 — structure _inference_ (no `A` given)** [verified]:

| Method      | Cora         | CiteSeer     | PubMed       | ogbn-arxiv   | Wine         | Cancer       | Digits       | 20news       |
| ----------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| LR          | 60.8±0.0     | 62.2±0.0     | 72.4±0.0     | 52.5±0.0     | 92.1±1.3     | 93.3±0.5     | 85.5±1.5     | 42.7±1.7     |
| MLP         | 56.1±1.6     | 56.7±1.7     | 71.4±0.0     | 54.7±0.1     | 89.7±1.9     | 92.9±1.2     | 36.3±0.3     | 38.6±1.4     |
| GCN_kNN     | 66.5±0.4     | 68.3±1.3     | 70.4±0.4     | 54.1±0.3     | 93.2±3.1     | 83.8±1.4     | 91.3±0.5     | 41.3±0.6     |
| LDS         | 71.5±0.8     | 71.5±1.1     | OOM          | OOM          | 97.3±0.4     | 94.4±1.9     | 92.5±0.7     | 46.4±1.6     |
| Pro-GNN     | 69.2±1.4     | 69.8±1.7     | OOM          | OOM          | 95.1±1.5     | 96.5±0.1     | 93.9±1.9     | 45.7±1.4     |
| IDGL        | 70.9±0.6     | 68.2±0.6     | 70.1±1.3     | 55.0±0.2     | **98.1±1.1** | 95.1±1.0     | 93.2±0.9     | 48.5±0.6     |
| SLAPS       | **73.4±0.3** | 72.6±0.6     | **74.4±0.6** | **56.6±0.1** | 96.6±0.4     | 96.6±0.2     | **94.4±0.7** | **50.4±0.7** |
| **SUBLIME** | 73.0±0.6     | **73.1±0.3** | 73.8±0.6     | 55.5±0.1     | **98.2±1.6** | **97.2±0.2** | 94.3±0.4     | 49.2±0.6     |

**Table 2 — structure _refinement_ (the true `A` is given)** [verified]:

| Method      | Cora         | CiteSeer     | PubMed       | ogbn-arxiv   |
| ----------- | ------------ | ------------ | ------------ | ------------ |
| GCN         | 81.5         | 70.3         | 79.0         | 71.7±0.3     |
| GAT         | 83.0±0.7     | 72.5±0.7     | 79.0±0.3     | OOM          |
| LDS         | 83.9±0.6     | **74.8±0.3** | OOM          | OOM          |
| Pro-GNN     | 82.1±0.4     | 71.3±0.4     | OOM          | OOM          |
| IDGL        | 84.0±0.5     | 73.1±0.7     | **83.0±0.2** | **72.0±0.3** |
| **SUBLIME** | **84.2±0.5** | 73.5±0.6     | 81.0±0.6     | 71.8±0.3     |

⭐ **The price of the graph.** Cora: 73.0 (best inferred structure) → 84.2 (true structure refined) = **11.2 points**. ogbn-arxiv: 56.6 → 72.0 = **15.4 points**. No amount of GSL recovers what the real adjacency carries. Directly relevant to §2.4: if hiding _all_ edges costs 11–15 points on node classification, hiding 10–50% of them should cost our planner a measurable and monotone amount, and if it does not, our planner was not using the structure.

### 5.5 Pro-GNN (KDD 2020) — structure denoising under perturbation

Metattack, `Ptb Rate` = fraction of edges perturbed. Accuracy±std [verified, Table 2], Cora and CiteSeer rows:

| Dataset  | Ptb % | GCN        | GAT            | RGCN       | GCN-Jaccard | GCN-SVD    | Pro-GNN-fs     | **Pro-GNN**    |
| -------- | ----- | ---------- | -------------- | ---------- | ----------- | ---------- | -------------- | -------------- |
| Cora     | 0     | 83.50±0.44 | **83.97±0.65** | 83.09±0.44 | 82.05±0.51  | 80.63±0.45 | 83.42±0.52     | 82.98±0.23     |
| Cora     | 5     | 76.55±0.79 | 80.44±0.74     | 77.42±0.39 | 79.13±0.59  | 78.39±0.54 | **82.78±0.39** | 82.27±0.45     |
| Cora     | 10    | 70.39±1.28 | 75.61±0.59     | 72.22±0.38 | 75.16±0.76  | 71.47±0.83 | 77.91±0.86     | **79.03±0.59** |
| Cora     | 15    | 65.10±0.71 | 69.78±1.28     | 66.82±0.39 | 71.03±0.64  | 66.69±1.18 | 76.01±1.12     | **76.40±1.27** |
| Cora     | 20    | 59.56±2.72 | 59.94±0.92     | 59.27±0.37 | 65.71±0.89  | 58.94±1.13 | 68.78±5.84     | **73.32±1.56** |
| Cora     | 25    | 47.53±1.96 | 54.78±0.74     | 50.51±0.78 | 60.82±1.08  | 52.06±1.19 | 56.54±2.58     | **69.72±1.69** |
| CiteSeer | 0     | 71.96±0.55 | 73.26±0.83     | 71.20±0.83 | 72.10±0.63  | 70.65±0.32 | 73.26±0.38     | **73.28±0.69** |
| CiteSeer | 10    | 67.55±0.89 | 70.63±0.48     | 67.71±0.30 | 69.54±0.56  | 68.87±0.62 | 72.43±0.52     | **72.51±0.75** |
| CiteSeer | 25    | 56.94±2.09 | 61.85±1.12     | 55.35±0.66 | 59.89±1.47  | 57.18±1.87 | 66.40±2.57     | **68.95±2.78** |

The shape to notice: at 0% perturbation Pro-GNN is _no better_ than plain GCN (82.98 vs 83.50 on Cora) — structure learning buys nothing on a clean graph and everything on a dirty one (69.72 vs 47.53 at 25%). Pro-GNN's _random_-attack results, the closest analogue to our uniform edge hiding, are published **only as Figure 4** — `[figure]`, no per-cell numbers.

### 5.6 SEAL (NeurIPS 2018) ⭐ — and two of its graphs are ours

Protocol: remove 10% of existing links as positive test data, sample an equal number of non-edges as negatives, train on the remaining 90%. Metric AUC, 10 runs [verified, prose].

**Table 1 — heuristics vs learned, AUC** [verified]:

| Data         | CN         | Jaccard    | PA         | AA         | RA         | Katz       | PR         | SR         | ENS        | WLK            | WLNM       | **SEAL**       |
| ------------ | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | -------------- | ---------- | -------------- |
| USAir        | 93.80±1.22 | 89.79±1.61 | 88.84±1.45 | 95.06±1.03 | 95.77±0.92 | 92.88±1.42 | 94.67±1.08 | 78.89±2.31 | 88.96±1.44 | **96.63±0.73** | 95.95±1.10 | 96.62±0.72     |
| **NS** ✅    | 94.42±0.95 | 94.43±0.93 | 68.65±2.03 | 94.45±0.93 | 94.45±0.93 | 94.85±1.10 | 94.89±1.08 | 94.79±1.08 | 97.64±0.25 | 98.57±0.51     | 98.61±0.49 | **98.85±0.47** |
| PB           | 92.04±0.35 | 87.41±0.39 | 90.14±0.45 | 92.36±0.34 | 92.46±0.37 | 92.92±0.35 | 93.54±0.41 | 77.08±0.80 | 90.15±0.45 | 93.83±0.59     | 93.49±0.47 | **94.72±0.46** |
| Yeast        | 89.37±0.61 | 89.32±0.60 | 82.20±1.02 | 89.43±0.62 | 89.45±0.62 | 92.24±0.61 | 92.76±0.55 | 91.49±0.57 | 82.36±1.02 | 95.86±0.54     | 95.62±0.52 | **97.91±0.52** |
| C.ele        | 85.13±1.61 | 80.19±1.64 | 74.79±2.04 | 86.95±1.40 | 87.49±1.41 | 86.34±1.89 | 90.32±1.49 | 77.07±2.00 | 74.94±2.04 | 89.72±1.67     | 86.18±1.72 | **90.30±1.35** |
| **Power** ✅ | 58.80±0.88 | 58.79±0.88 | 44.33±1.02 | 58.79±0.88 | 58.79±0.88 | 65.39±1.59 | 66.00±1.59 | 76.15±1.06 | 79.52±1.78 | 82.41±3.43     | 84.76±0.98 | **87.61±1.57** |
| Router       | 56.43±0.52 | 56.40±0.52 | 47.58±1.47 | 56.43±0.51 | 56.43±0.51 | 38.62±1.35 | 38.76±1.39 | 37.40±1.27 | 47.58±1.48 | 87.42±2.08     | 94.41±0.88 | **96.38±1.45** |
| E.coli       | 93.71±0.39 | 81.31±0.61 | 91.82±0.58 | 95.36±0.34 | 95.95±0.35 | 93.50±0.44 | 95.57±0.44 | 62.49±1.43 | 91.89±0.58 | 96.94±0.29     | 97.21±0.27 | **97.64±0.22** |

✅ **`NS` is our `netscience` and `Power` is our `power_grid`, byte-for-byte.** SEAL's Appendix C states _"NS … 1,589 nodes and 2,742 edges … average node degree 3.45"_ and _"Power … 4,941 nodes and 6,594 edges … average node degree 2.67"_ [verified] — identical on all three statistics to our loaders (§6.2). `PB` (1,222 / 16,714) is also identical to Pro-GNN's `Polblogs`.

**Why the `Power` row is the interesting one.** Every neighbourhood heuristic collapses to near-chance on the power grid (CN/AA/RA all 58.79, PA 44.33 — _worse_ than random) because it has almost no triangles; only the subgraph-learning methods recover (SEAL 87.61). This is the same structural property that makes power_grid the hardest graph in our IM suite, and it is a ready-made sanity check: **if our world model is asked to score hidden edges on power_grid, the heuristic floor is ~58 AUC and the published learned ceiling is ~88.**

### 5.7 BUDDY / ELPH (ICLR 2023) — the static link-prediction state of the art

Metrics differ per dataset (first row). Planetoid splits are random; OGB uses the fixed OGB splits, and where possible baselines are taken from the OGB leaderboard [verified, Table 2]:

| Method    | Cora HR@100    | CiteSeer HR@100 | PubMed HR@100  | Collab HR@50   | PPA HR@100     | Citation2 MRR  | DDI HR@20      |
| --------- | -------------- | --------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| CN        | 33.92±0.46     | 29.79±0.90      | 23.13±0.15     | 56.44±0.00     | 27.65±0.00     | 51.47±0.00     | 17.73±0.00     |
| AA        | 39.85±1.34     | 35.19±1.33      | 27.38±0.11     | 64.35±0.00     | 32.45±0.00     | 51.89±0.00     | 18.61±0.00     |
| RA        | 41.07±0.48     | 33.56±0.17      | 27.03±0.35     | 64.00±0.00     | 49.33±0.00     | 51.98±0.00     | 27.60±0.00     |
| transE    | 67.40±1.60     | 60.19±1.15      | 36.67±0.99     | 29.40±1.15     | 22.69±0.49     | 76.44±0.18     | 6.65±0.20      |
| DistMult  | 41.38±2.49     | 47.65±1.68      | 40.32±0.89     | 51.00±0.54     | 28.61±1.47     | 66.95±0.40     | 11.01±0.49     |
| GCN       | 66.79±1.65     | 67.08±2.94      | 53.02±1.39     | 47.14±1.45     | 18.67±1.32     | 84.74±0.21     | 37.07±5.07     |
| SAGE      | 55.02±4.03     | 57.01±3.74      | 39.66±0.72     | 54.63±1.12     | 16.55±2.40     | 82.60±0.36     | 53.90±4.74     |
| Neo-GNN   | 80.42±1.31     | 84.67±2.16      | 73.93±1.19     | 62.13±0.5      | 49.13±0.60     | 87.26±0.84     | 63.57±3.52     |
| SEAL      | 81.71±1.30     | 83.89±2.15      | **75.54±1.32** | 64.74±0.43     | 48.80±3.16     | **87.67±0.32** | 30.56±3.86     |
| NBFnet    | 71.65±2.27     | 74.07±1.75      | 58.73±1.99     | OOM            | OOM            | OOM            | 4.00±0.58      |
| ELPH      | 87.72±2.13     | **93.44±0.53**  | 72.99±1.43     | **66.32±0.40** | OOM            | OOM            | **83.19±2.12** |
| **BUDDY** | **88.00±0.44** | 92.93±0.27      | 74.10±0.78     | 65.94±0.58     | **49.85±0.20** | 87.56±0.11     | 78.51±1.36     |

⚠️ **A transcription trap in this table's companion.** BUDDY's Table 6 lists PubMed as **18,717** nodes [verified]; PubMed is 19,717 (Kipf Table 1, SLAPS Table 5, OGB-independent sources — §6.4). That is a digit error in the paper, exactly the class of thing §0 exists to catch. Their edge counts (Cora 5,278, CiteSeer 4,676, PubMed 44,327) are the deduplicated-undirected convention.

### 5.8 DyGLib (NeurIPS 2023 D&B) — the unified dynamic-LP re-run

AP × 100 for **transductive** dynamic link prediction under **random** negative sampling [verified, Table 1, `rnd` block]. Every number was produced by one pipeline, so this table supersedes the original papers' self-reported numbers.

| Dataset       | JODIE      | DyRep      | TGAT       | TGN            | CAWN       | EdgeBank   | TCL        | GraphMixer | DyGFormer      |
| ------------- | ---------- | ---------- | ---------- | -------------- | ---------- | ---------- | ---------- | ---------- | -------------- |
| Wikipedia     | 96.50±0.14 | 94.86±0.06 | 96.94±0.06 | 98.45±0.06     | 98.76±0.03 | 90.37±0.00 | 96.47±0.16 | 97.25±0.03 | **99.03±0.02** |
| Reddit        | 98.31±0.14 | 98.22±0.04 | 98.52±0.02 | 98.63±0.06     | 99.11±0.01 | 94.86±0.00 | 97.53±0.02 | 97.31±0.01 | **99.22±0.01** |
| MOOC          | 80.23±2.44 | 81.97±0.49 | 85.84±0.15 | **89.15±1.60** | 80.15±0.25 | 57.97±0.00 | 82.38±0.24 | 82.78±0.15 | 87.52±0.49     |
| LastFM        | 70.85±2.13 | 71.92±2.21 | 73.42±0.21 | 77.07±3.97     | 86.99±0.06 | 79.29±0.00 | 67.27±2.16 | 75.61±0.24 | **93.00±0.12** |
| Enron         | 84.77±0.30 | 82.38±3.36 | 71.12±0.97 | 86.53±1.11     | 89.56±0.09 | 83.53±0.00 | 79.70±0.71 | 82.25±0.16 | **92.47±0.12** |
| Social Evo.   | 89.89±0.55 | 88.87±0.30 | 93.16±0.17 | 93.57±0.17     | 84.96±0.09 | 74.95±0.00 | 93.13±0.16 | 93.37±0.07 | **94.73±0.01** |
| UCI           | 89.43±1.09 | 65.14±2.30 | 79.63±0.70 | 92.34±1.04     | 95.18±0.06 | 76.20±0.00 | 89.57±1.63 | 93.25±0.57 | **95.79±0.17** |
| Flights       | 95.60±1.73 | 95.29±0.72 | 94.03±0.18 | 97.95±0.14     | 98.51±0.01 | 89.35±0.00 | 91.23±0.02 | 90.99±0.05 | **98.91±0.01** |
| Can. Parl.    | 69.26±0.31 | 66.54±2.76 | 70.73±0.72 | 70.88±2.34     | 69.82±2.34 | 64.55±0.00 | 68.67±2.67 | 77.04±0.46 | **97.36±0.45** |
| US Legis.     | 75.05±1.52 | 75.34±0.39 | 68.52±3.16 | **75.99±0.58** | 70.58±0.48 | 58.39±0.00 | 69.59±0.48 | 70.74±1.02 | 71.11±0.59     |
| UN Trade      | 64.94±0.31 | 63.21±0.93 | 61.47±0.18 | 65.03±1.37     | 65.39±0.12 | 60.41±0.00 | 62.21±0.03 | 62.61±0.27 | **66.46±1.29** |
| UN Vote       | 63.91±0.81 | 62.81±0.80 | 52.21±0.98 | **65.72±2.17** | 52.84±0.10 | 58.49±0.00 | 51.90±0.30 | 52.11±0.16 | 55.55±0.42     |
| Contact       | 95.31±1.33 | 95.98±0.15 | 96.28±0.09 | 96.89±0.56     | 90.26±0.28 | 92.58±0.00 | 92.44±0.12 | 91.92±0.03 | **98.29±0.01** |
| **Avg. Rank** | 5.08       | 5.85       | 5.69       | 2.54           | 4.31       | 7.54       | 6.92       | 5.46       | **1.62**       |

Under **historical** and **inductive** negative sampling the ordering changes substantially (DyGFormer's avg rank 2.62 / 3.23; CAWN falls to 7.54 under `hist`) [verified] — the negative sampler, not the model, decides half the leaderboard. That is the methodological lesson to carry: an evaluation whose negatives are too easy ranks memorization above modelling.

### 5.9 TGB (NeurIPS 2023 D&B) — the current standard, and it is harsher

MRR under TGB's hard-negative protocol [verified, Tables 2a/2b/3]:

| Method      | tgbl-wiki (val/test)          | tgbl-review (val/test)        | tgbl-coin (val/test)          | tgbl-comment (val/test)       | tgbl-flight (val/test)        |
| ----------- | ----------------------------- | ----------------------------- | ----------------------------- | ----------------------------- | ----------------------------- |
| DyRep       | 0.072±0.009 / 0.050±0.017     | 0.216±0.031 / 0.220±0.030     | 0.512±0.014 / 0.452±0.046     | 0.291±0.028 / 0.289±0.033     | 0.573±0.013 / 0.556±0.014     |
| TGN         | 0.435±0.069 / 0.396±0.060     | 0.313±0.012 / 0.349±0.020     | **0.607±0.014 / 0.586±0.037** | **0.356±0.019 / 0.379±0.021** | **0.731±0.010 / 0.705±0.020** |
| CAWN        | 0.743±0.004 / 0.711±0.006     | 0.200±0.001 / 0.193±0.001     | OOM                           | OOM                           | OOM                           |
| TCL         | 0.198±0.016 / 0.207±0.025     | 0.199±0.007 / 0.193±0.009     | OOM                           | OOM                           | OOM                           |
| GraphMixer  | 0.113±0.003 / 0.118±0.002     | 0.428±0.019 / **0.521±0.015** | OOM                           | OOM                           | OOM                           |
| TGAT        | 0.131±0.008 / 0.141±0.007     | 0.324±0.006 / 0.355±0.012     | OOM                           | OOM                           | OOM                           |
| NAT         | **0.773±0.011 / 0.749±0.010** | 0.302±0.011 / 0.341±0.020     | —                             | —                             | —                             |
| EdgeBank_tw | 0.600 / 0.571                 | 0.0242 / 0.0253               | 0.492 / 0.580                 | 0.124 / 0.149                 | 0.363 / 0.387                 |
| EdgeBank_∞  | 0.527 / 0.495                 | 0.0229 / 0.0229               | 0.315 / 0.359                 | 0.109 / 0.129                 | 0.166 / 0.167                 |

`tgbl-wiki` uses **all** possible negatives; `tgbl-review` uses 100 negatives per positive [verified]. Two findings the paper stresses: the memorization heuristic **EdgeBank beats DyRep on `tgbl-coin`** (0.580 vs 0.452 test MRR), and on the long-horizon datasets validation and test MRR diverge because of genuine distribution shift over the 5-year span [verified, prose].

Compare with §5.8: the _same_ Wikipedia graph scores AP 0.96–0.99 under random negatives (DyGLib) and MRR 0.05–0.75 under TGB's negatives. **Do not mix the two number systems.**

### 5.10 NodeFormer (NeurIPS 2022) — GSL at scale

Testing ROC-AUC on OGB-Proteins (batch 10K) and accuracy on Amazon2M (batch 100K), with training memory [verified, Tables 2 and 3]:

| Method         | OGB-Proteins ROC-AUC (%) | Train Mem | Amazon2M Accuracy (%) | Train Mem |
| -------------- | ------------------------ | --------- | --------------------- | --------- |
| MLP            | 72.04±0.48               | 2.0 GB    | 63.46±0.10            | 1.4 GB    |
| GCN            | 72.51±0.35               | 2.5 GB    | 83.90±0.10            | 5.7 GB    |
| SGC            | 70.31±0.23               | 1.2 GB    | 81.21±0.12            | 1.7 GB    |
| GraphSAINT-GAT | 74.63±1.24               | 5.2 GB    | 85.17±0.32            | 2.2 GB    |
| **NodeFormer** | **77.45±1.15**           | 3.2 GB    | **87.85±0.24**        | 4.0 GB    |

The point of the table is the memory column: learned all-pair structure at 2M nodes in 4 GB. It is the only method in §4.2 that does not OOM on our larger graphs.

---

## 6. Datasets

### 6.1 What we already load, and where it appears here

✅ marks a dataset we load via `data/datasets/<name>.py`. The authoritative rows for all of these live in [`influence_maximization.md`](influence_maximization.md) §6.1 — this section records only the **link between our graph and this literature**.

| `--dataset`                                                                                                                            | Ours                                | Appears in this literature as                                                                        | Same graph?                              |
| -------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| `netscience` ✅                                                                                                                        | 1,589 / 2,742 undirected            | SEAL's **NS** — _"1,589 nodes and 2,742 edges, average node degree 3.45"_ [verified, SEAL App. C]    | ✅ **identical on all three statistics** |
| `power_grid` ✅                                                                                                                        | 4,941 / 6,594 undirected            | SEAL's **Power** — _"4,941 nodes and 6,594 edges, average node degree 2.67"_ [verified, SEAL App. C] | ✅ **identical on all three statistics** |
| `cora_ml` ✅                                                                                                                           | 2,810 / 7,981, 2,879-dim, 7 classes | **nothing** — the "Cora" of this literature is a different graph                                     | ❌ **name collision, see §6.3**          |
| `jazz`, `facebook`, `email_eu_core`, `lastfm_asia`, `ca_grqc`, `wiki_vote`, `nethept`, `netphy`, `twitter`, `digg`, `youtube`, `weibo` | —                                   | not found in any §4 paper surveyed                                                                   | —                                        |

Two exact overlaps is more than this file expected to find, and both are on the **structure** side, not the feature side — consistent with §2's verdict. It also means the §2.4 robustness experiment has a published reference point: on `netscience` and `power_grid`, §5.6 gives the AUC that CN/AA/RA/Katz/SEAL reach when 10% of edges are hidden.

One more incidental match: Pro-GNN's **Polblogs** (1,222 / 16,714, no node features, 2 classes) is SEAL's **PB** (1,222 / 16,714) [verified, both tables]. We do not load it.

### 6.2 ⭐ The Cora name collision — read this before quoting any "Cora" number

**Our `cora_ml` is not the Cora that GNN papers benchmark on.** They are different graphs from different preprocessing lineages of the same 2000-era McCallum crawl, and they disagree on node count, edge count, and — decisively — feature dimension.

|             | **Planetoid "Cora"** (this literature)                                                                                                | **Our `cora_ml`** (the IM literature)                     |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| Nodes       | **2,708**                                                                                                                             | **2,810**                                                 |
| Edges       | **5,429** as published                                                                                                                | **7,981** undirected                                      |
| Feature dim | **1,433** binary bag-of-words                                                                                                         | **2,879** bag-of-words                                    |
| Classes     | 7                                                                                                                                     | 7                                                         |
| Directed?   | undirected (citations symmetrized)                                                                                                    | undirected (symmetrized on load)                          |
| Provenance  | Sen et al. 2008 → Yang et al. 2016 split                                                                                              | graph2gauss `cora_ml.npz` → `standardize()`               |
| Source      | [linqs cora.tgz](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz) · [kimiyoung/planetoid](https://github.com/kimiyoung/planetoid) | [graph2gauss](https://github.com/abojchevski/graph2gauss) |
| Loader      | PyG `Planetoid(root, "Cora")`                                                                                                         | `data/datasets/cora_ml.py`                                |
| Used by     | FP, GCNmf, T2-GNN, PCFI, LDS, Pro-GNN, IDGL, SLAPS, SUBLIME, SEAL-successors, BUDDY                                                   | DeepIM, MOEIM, SL-VAE, us                                 |

The 7-class agreement is a coincidence of both being topic-labelled subsets of the same corpus; **the feature vocabularies are unrelated** (1,433 vs 2,879 terms), so no feature-imputation result transfers between them. This is exactly the failure mode catalogued in [`influence_maximization.md`](influence_maximization.md) **§6.3** for Epinions, DBLP, Twitter, Wiki-Vote, Digg, Weibo and LiveJournal — one name, two graphs. Add **Cora** to that list.

**Worse: "Planetoid Cora" is itself quoted three ways**, and all three are the same file under different conventions [verified, four independent tables]:

| Quoted as                   | Nodes     | Edges     | Who                                                | What it is                                                                                          |
| --------------------------- | --------- | --------- | -------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| raw                         | 2,708     | **5,429** | Kipf & Welling Table 1; SLAPS Table 5              | the citation-link count in the Planetoid file, _"as reported in Yang et al. (2016)"_                |
| deduplicated undirected     | 2,708     | **5,278** | T2-GNN Table 1; BUDDY Table 6                      | after symmetrize + drop self-loops + collapse multi-edges (PyG reports 10,556 directed = 2 × 5,278) |
| largest connected component | **2,485** | **5,069** | Shchur et al. Table 3; Pro-GNN Table 1; FP Table 3 | LCC after standardization                                                                           |

So a paper reporting "Cora, 2,485 nodes" and one reporting "Cora, 2,708 nodes" are using the same graph; one reporting "Cora, 2,810 nodes" is not. The 2,485 / 5,069 variant traces to **Shchur et al., _Pitfalls of GNN Evaluation_ ([arXiv 1811.05868](https://arxiv.org/abs/1811.05868), code [shchur/gnn-benchmark](https://github.com/shchur/gnn-benchmark))**, whose standardized loader is what FP and Pro-GNN both consume.

**CiteSeer has the same three-way split and one unresolved disagreement:** raw 3,327 / 4,732 [Kipf, SLAPS]; deduplicated 3,327 / 4,676 [T2-GNN, BUDDY]; LCC quoted as **2,110 / 3,668** by Pro-GNN but **2,120 / 3,679** by Shchur _and_ FP [all verified]. A 10-node / 11-edge gap between two "LCC of CiteSeer" figures — cause not established, logged in §11.

### 6.3 Static node-classification benchmarks (the §4.1 / §4.2 suite)

Feature dim and class count are load-bearing here — they are what a feature imputer imputes. `Edges` follows each source's own convention; the ⚠️ column flags where conventions differ.

| Dataset              | Nodes     | Edges                                                                        | Feat. dim      | Classes   | Type            | Source                                                                                                                      | ⚠️                                |
| -------------------- | --------- | ---------------------------------------------------------------------------- | -------------- | --------- | --------------- | --------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| **Cora** (Planetoid) | 2,708     | 5,429 raw / 5,278 dedup / 5,069 LCC(2,485)                                   | 1,433          | 7         | undirected      | [linqs](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz) · [planetoid](https://github.com/kimiyoung/planetoid)          | 3 counts, §6.2                    |
| **CiteSeer**         | 3,327     | 4,732 raw / 4,676 dedup / 3,668–3,679 LCC(2,110–2,120)                       | 3,703          | 6         | undirected      | [planetoid](https://github.com/kimiyoung/planetoid)                                                                         | LCC disputed                      |
| **PubMed**           | 19,717    | 44,338 [Kipf, SLAPS, Pro-GNN] / 44,324 [Shchur, FP] / 44,327 [T2-GNN, BUDDY] | 500            | 3         | undirected      | [planetoid](https://github.com/kimiyoung/planetoid)                                                                         | BUDDY misprints nodes as 18,717   |
| **Cora-Full**        | 18,703    | 62,421                                                                       | 8,710          | 67        | undirected      | [gnn-benchmark](https://github.com/shchur/gnn-benchmark)                                                                    | 3 tiny classes dropped by Shchur  |
| **Coauthor CS**      | 18,333    | 81,894                                                                       | 6,805          | 15        | undirected      | [gnn-benchmark](https://github.com/shchur/gnn-benchmark)                                                                    |                                   |
| **Coauthor Physics** | 34,493    | 247,962                                                                      | 8,415          | 5         | undirected      | [gnn-benchmark](https://github.com/shchur/gnn-benchmark)                                                                    |                                   |
| **Amazon Computers** | 13,381    | 245,778                                                                      | 767            | 10        | undirected      | [gnn-benchmark](https://github.com/shchur/gnn-benchmark)                                                                    |                                   |
| **Amazon Photo**     | 7,487     | 119,043                                                                      | 745            | 8         | undirected      | [gnn-benchmark](https://github.com/shchur/gnn-benchmark)                                                                    |                                   |
| **ogbn-arxiv**       | 169,343   | 1,166,243                                                                    | 128            | 40        | directed (arcs) | [OGB nodeprop](https://ogb.stanford.edu/docs/nodeprop/)                                                                     |                                   |
| **ogbn-products**    | 2,449,029 | 61,859,140 [OGB] / **123,718,280** [FP]                                      | 100            | 47        | undirected      | [OGB nodeprop](https://ogb.stanford.edu/docs/nodeprop/)                                                                     | **exactly 2×** — see below        |
| **ogbn-proteins**    | 132,534   | 39,561,252                                                                   | 8 (edge feat.) | 112 tasks | undirected      | [OGB nodeprop](https://ogb.stanford.edu/docs/nodeprop/)                                                                     |                                   |
| **Texas**            | 183       | 295                                                                          | 1,703          | 5         | directed        | [WebKB](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-11/www/wwkb)                                                      | heterophilous                     |
| **Cornell**          | 183       | 295                                                                          | 1,703          | 5         | directed        | same                                                                                                                        | heterophilous                     |
| **Wisconsin**        | 251       | 499                                                                          | 1,703          | 5         | directed        | same                                                                                                                        | heterophilous                     |
| **Chameleon**        | 2,277     | 31,421                                                                       | 2,325          | 5         | undirected      | [Rozemberczki wiki](https://github.com/benedekrozemberczki/MUSAE)                                                           | heterophilous                     |
| **Squirrel**         | 5,201     | 198,493                                                                      | 2,089          | 5         | undirected      | same                                                                                                                        | heterophilous                     |
| **Polblogs / PB**    | 1,222     | 16,714                                                                       | **none**       | 2         | directed        | via [Pro-GNN](https://github.com/ChandlerBang/Pro-GNN) / [SEAL](https://github.com/muhanzhang/SEAL/tree/master/Python/data) | featureless — Jaccard/Pro-GNN N/A |

Node/edge/feature/class counts: Cora, CiteSeer, PubMed [verified, Kipf Table 1 and SLAPS Table 5]; Cora-Full, Coauthor, Amazon [verified, Shchur Table 3]; ogbn-\* [verified, OGB Table 1 and FP Table 3]; WebKB, Chameleon, Squirrel [verified, T2-GNN Table 1]; Polblogs [verified, Pro-GNN Table 1 + SEAL App. C].

⭐ **`ogbn-products` is the cleanest edge-convention case in the folder.** OGB's own Table 1 says **61,859,140**; FP's Table 3 says **123,718,280**. `2 × 61,859,140 = 123,718,280` exactly [derived] — undirected edges vs stored arcs, same file. This is the same arithmetic that resolved our NetHEPT question ([`influence_maximization.md`](influence_maximization.md) §6.4). Check the factor of 2 before ever concluding two graphs differ.

### 6.4 Static link-prediction benchmarks

**SEAL's eight small graphs** [verified, SEAL App. C] — no node features; these are pure-topology benchmarks, which is why they are the relevant ones for us:

| Dataset   | Nodes     | Edges     | Avg deg | Domain                     | Ours?             |
| --------- | --------- | --------- | ------- | -------------------------- | ----------------- |
| USAir     | 332       | 2,126     | 12.81   | airline routes             |                   |
| **NS**    | **1,589** | **2,742** | 3.45    | coauthorship (Newman 2006) | ✅ = `netscience` |
| PB        | 1,222     | 16,714    | 27.36   | US political blogs         |                   |
| Yeast     | 2,375     | 11,693    | 9.85    | protein–protein            |                   |
| C.ele     | 297       | 2,148     | 14.46   | C. elegans neural          |                   |
| **Power** | **4,941** | **6,594** | 2.67    | US western power grid      | ✅ = `power_grid` |
| Router    | 5,022     | 6,258     | 2.49    | router-level Internet      |                   |
| E.coli    | 1,805     | 14,660    | 12.55   | metabolite reactions       |                   |

Bundled with the code at [muhanzhang/SEAL/Python/data](https://github.com/muhanzhang/SEAL/tree/master/Python/data).

**OGB link-property prediction** [verified, OGB Table 1] — node features vary by dataset; the leaderboard metric is fixed per dataset:

| Dataset        | Nodes     | Edges      | Node features          | Split          | Metric   | Source                                              |
| -------------- | --------- | ---------- | ---------------------- | -------------- | -------- | --------------------------------------------------- |
| ogbl-ppa       | 576,289   | 30,326,273 | 58-dim species one-hot | throughput     | Hits@100 | [linkprop](https://ogb.stanford.edu/docs/linkprop/) |
| ogbl-collab    | 235,868   | 1,285,465  | 128-dim word2vec       | time           | Hits@50  | same                                                |
| ogbl-ddi       | 4,267     | 1,334,889  | **none**               | protein-target | Hits@20  | same                                                |
| ogbl-citation2 | 2,927,963 | 30,561,187 | 128-dim word2vec       | time           | MRR      | same                                                |
| ogbl-wikikg2   | 2,500,604 | 17,137,181 | KG entities/relations  | time           | MRR      | same                                                |
| ogbl-biokg     | 93,773    | 5,088,434  | heterogeneous KG       | random         | MRR      | same                                                |

Leaderboards: [ogb.stanford.edu/docs/leader_linkprop](https://ogb.stanford.edu/docs/leader_linkprop/). Note `ogbl-ddi` has **no node features at all** — which is why the feature-free heuristics in §5.7 stay competitive there and why it is the OGB task most like our setting.

### 6.5 Dynamic / temporal benchmarks

**DyGLib's thirteen** [verified, DyGLib Table 6]. `#N&L Feat` = dimensions of (node feature, link feature); `–` means none. The first four are the classic JODIE datasets, downloadable from [snap.stanford.edu/jodie](https://snap.stanford.edu/jodie/); all thirteen ship preprocessed with [yule-BUAA/DyGLib](https://github.com/yule-BUAA/DyGLib).

| Dataset     | Domain      | Nodes  | Links     | Node & link feat. | Bipartite | Duration      | Unique steps | Granularity |
| ----------- | ----------- | ------ | --------- | ----------------- | --------- | ------------- | ------------ | ----------- |
| Wikipedia   | social      | 9,227  | 157,474   | – & 172           | yes       | 1 month       | 152,757      | Unix ts     |
| Reddit      | social      | 10,984 | 672,447   | – & 172           | yes       | 1 month       | 669,065      | Unix ts     |
| MOOC        | interaction | 7,144  | 411,749   | – & 4             | yes       | 17 months     | 345,600      | Unix ts     |
| LastFM      | interaction | 1,980  | 1,293,103 | – & –             | yes       | 1 month       | 1,283,614    | Unix ts     |
| Enron       | social      | 184    | 125,235   | – & –             | no        | 3 years       | 22,632       | Unix ts     |
| Social Evo. | proximity   | 74     | 2,099,519 | – & 2             | no        | 8 months      | 565,932      | Unix ts     |
| UCI         | social      | 1,899  | 59,835    | – & –             | no        | 196 days      | 58,911       | Unix ts     |
| Flights     | transport   | 13,169 | 1,927,145 | – & 1             | no        | 4 months      | 122          | days        |
| Can. Parl.  | politics    | 734    | 74,478    | – & 1             | no        | 14 years      | 14           | years       |
| US Legis.   | politics    | 225    | 60,396    | – & 1             | no        | 12 congresses | 12           | congresses  |
| UN Trade    | economics   | 255    | 507,497   | – & 1             | no        | 32 years      | 32           | years       |
| UN Vote     | politics    | 201    | 1,035,742 | – & 1             | no        | 72 years      | 72           | years       |
| Contact     | proximity   | 692    | 2,426,279 | – & 1             | no        | 1 month       | 8,064        | 5 minutes   |

None of these carry node features — the "features" are edge attributes. DyGLib notes its Contact counts differ slightly from the source paper's (694 nodes / 2,426,280 links) despite using the same released file [verified, prose].

**TGB's nine** [verified, TGB Table 1]. `Surprise` = fraction of test edges never seen in training — the single most predictive statistic for whether EdgeBank wins. Chronological 70/15/15 split throughout. Download and leaderboards: [tgb.complexdatalab.com](https://tgb.complexdatalab.com/) · code [shenyangHuang/TGB](https://github.com/shenyangHuang/TGB).

| Dataset      | Task | Domain      | Nodes   | Edges      | Steps      | Surprise  | Edge props (W/Di/A) |
| ------------ | ---- | ----------- | ------- | ---------- | ---------- | --------- | ------------------- |
| tgbl-wiki    | link | interaction | 9,227   | 157,474    | 152,757    | 0.108     | ✘ / ✓ / ✓           |
| tgbl-review  | link | rating      | 352,637 | 4,873,540  | 6,865      | **0.987** | ✓ / ✓ / ✘           |
| tgbl-coin    | link | transaction | 638,486 | 22,809,486 | 1,295,720  | 0.120     | ✓ / ✓ / ✘           |
| tgbl-comment | link | social      | 994,790 | 44,314,507 | 30,998,030 | 0.823     | ✓ / ✓ / ✓           |
| tgbl-flight  | link | traffic     | 18,143  | 67,169,570 | 1,385      | 0.024     | ✘ / ✓ / ✓           |
| tgbn-trade   | node | trade       | 255     | 468,245    | 32         | 0.023     | ✓ / ✓ / ✘           |
| tgbn-genre   | node | interaction | 1,505   | 17,858,395 | 133,758    | 0.005     | ✓ / ✓ / ✘           |
| tgbn-reddit  | node | social      | 11,766  | 27,174,118 | 21,889,537 | 0.013     | ✓ / ✓ / ✘           |
| tgbn-token   | node | transaction | 61,756  | 72,936,998 | 2,036,524  | 0.014     | ✓ / ✓ / ✓           |

`tgbl-wiki` is the _same_ Wikipedia graph as DyGLib's row above (9,227 / 157,474) — the only dataset shared between the two benchmarks, and the reason §5.9's closing warning matters: one graph, two number systems.

---

## 7. Which paper uses which

Cells mark the dataset **as that paper reports it**. Check §6.2 before assuming two "Cora" cells are the same graph.

| Dataset                                      | GCNmf'21 | SAT'22 | FP'22 | PCFI'23 | T2-GNN'23 | LDS'19 | Pro-GNN'20 | IDGL'20 | SLAPS'21 | SUBLIME'22 | NodeFormer'22 | SEAL'18 | Neo-GNN'21 | BUDDY'23 | DyGLib'23 | TGB'23 |
| -------------------------------------------- | -------- | ------ | ----- | ------- | --------- | ------ | ---------- | ------- | -------- | ---------- | ------------- | ------- | ---------- | -------- | --------- | ------ |
| Cora (Planetoid)                             | ✔        | ✔      | ✔     | ✔       | ✔         | ✔      | ✔          | ✔       | ✔        | ✔          | ✔             |         | ✔          | ✔        |           |        |
| CiteSeer                                     | ✔        | ✔      | ✔     | ✔       | ✔         | ✔      | ✔          | ✔       | ✔        | ✔          | ✔             |         | ✔          | ✔        |           |        |
| PubMed                                       | ✔        |        | ✔     | ✔       | ✔         |        | ✔          | ✔       | ✔        | ✔          | ✔             |         | ✔          | ✔        |           |        |
| Amazon Photo                                 | ✔        | ✔      | ✔     | ✔       |           |        |            |         |          |            |               |         |            |          |           |        |
| Amazon Computers                             | ✔        | ✔      | ✔     | ✔       |           |        |            |         |          |            |               |         |            |          |           |        |
| Coauthor CS / Physics                        |          |        |       | ✔       |           |        |            |         |          |            | ✔             |         |            |          |           |        |
| ogbn-arxiv                                   |          |        | ✔     | ✔       |           |        |            |         | ✔        | ✔          |               |         |            |          |           |        |
| ogbn-products                                |          |        | ✔     |         |           |        |            |         |          |            |               |         |            |          |           |        |
| ogbn-proteins                                |          |        |       |         |           |        |            |         |          |            | ✔             |         |            |          |           |        |
| WebKB (Texas/Cornell/Wisc.)                  |          |        |       |         | ✔         |        |            |         |          |            |               |         |            |          |           |        |
| Chameleon / Squirrel                         |          |        |       |         | ✔         |        |            |         |          |            |               |         |            |          |           |        |
| Polblogs / PB                                |          |        |       |         |           |        | ✔          |         |          |            |               | ✔       |            |          |           |        |
| **NetScience (NS)** ✅                       |          |        |       |         |           |        |            |         |          |            |               | ✔       |            |          |           |        |
| **Power Grid** ✅                            |          |        |       |         |           |        |            |         |          |            |               | ✔       |            |          |           |        |
| USAir / Yeast / C.ele / Router / E.coli      |          |        |       |         |           |        |            |         |          |            |               | ✔       |            |          |           |        |
| ogbl-collab / ppa / ddi / citation2          |          |        |       |         |           |        |            |         |          |            |               |         | ✔          | ✔        |           |        |
| Wine / Cancer / Digits / 20news (no graph)   |          |        |       |         |           | ✔      |            |         | ✔        | ✔          |               |         |            |          |           |        |
| Wikipedia / Reddit / MOOC / LastFM           |          |        |       |         |           |        |            |         |          |            |               |         |            |          | ✔         | ✔      |
| Enron / UCI / Contact / Flights / UN / Parl. |          |        |       |         |           |        |            |         |          |            |               |         |            |          | ✔         |        |
| tgbl-review / coin / comment / flight        |          |        |       |         |           |        |            |         |          |            |               |         |            |          |           | ✔      |

**Bold = we already have it.** The entire intersection between our loaded suite and this literature is two rows, both in SEAL's column.

---

## 8. Evaluation protocol

### 8.1 Masking protocols — how incompleteness is manufactured

These are the recipes to copy for §2.4 / §9. All are experimenter-side corruptions applied _before_ training.

| Protocol                           | Rule                                                                      | Rates used                   | Used by                                    |
| ---------------------------------- | ------------------------------------------------------------------------- | ---------------------------- | ------------------------------------------ |
| **Uniform feature MCAR**           | each entry of `X` dropped independently w.p. `r`                          | 10–99%, headline at 50/90/99 | FP [verified, footnote 5], PCFI (to 99.5%) |
| **Structural / attribute-missing** | a fraction of _nodes_ have **no** attributes at all                       | typically 40–60% of nodes    | SAT, Amer, ITR, MEGAE                      |
| **Biased feature missing**         | drop probability depends on the value                                     | —                            | GCNmf's third setting                      |
| **Joint feature + edge missing**   | drop both at the same rate                                                | 0/20/40/60/80%               | T2-GNN [verified, Table 4]                 |
| **Uniform edge deletion**          | delete a fraction of edges uniformly at random                            | 25/50/75% typical            | IDGL, GSL robustness studies               |
| **Adversarial edge perturbation**  | metattack / nettack rewires edges                                         | 0/5/10/15/20/25%             | Pro-GNN [verified, Table 2]                |
| **Random edge injection**          | add random fake edges                                                     | 0–100%                       | Pro-GNN §5.2.3 [figure only]               |
| **No graph at all**                | discard `A`; optionally substitute a kNN graph on `X`                     | 100%                         | LDS, SLAPS, SUBLIME, NodeFormer            |
| **Link-prediction split**          | hold out 10% of edges as positives + equal sampled non-edges as negatives | 10%                          | SEAL [verified, prose]                     |
| **Chronological split**            | 70/15/15 by time                                                          | fixed                        | TGB [verified], DyGLib                     |

**The one to use for us** is _uniform edge deletion_, because it is the honest model of an under-observed influence network and because §5.6 gives published reference numbers for exactly that protocol on two graphs we already load. Follow SEAL's split ratio (10%) as the low rung, then extend to 20% and 50%.

### 8.2 Metrics

| Task                            | Metric                                                                                          | Note                                                            |
| ------------------------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| Feature imputation (downstream) | node-classification **accuracy** at each missing rate, plus **relative drop** vs full features  | FP reports both; the relative drop is the readable one          |
| Feature imputation (direct)     | RMSE / recall@k on the reconstructed `X`                                                        | rarer; SAT/ITR/MEGAE report it                                  |
| GSL                             | node-classification accuracy on the learned `A*`                                                | never edge-recovery accuracy — nobody scores the graph directly |
| Static link prediction          | **AUC** (small graphs, SEAL-style), **Hits@K** (OGB collab/ppa/ddi), **MRR** (OGB citation2/KG) | AUC saturates; OGB moved to Hits@K for exactly that reason      |
| Dynamic link prediction         | **AP** and **AUC-ROC** (DyGLib), **MRR** (TGB)                                                  | not interchangeable — §5.9                                      |

### 8.3 The traps

1. **Negative sampling decides the leaderboard.** DyGLib's three strategies reorder the same nine methods (avg rank of DyGFormer 1.62 → 2.62 → 3.23 across rnd/hist/ind) [verified]. TGB's hard negatives turn AP≈0.99 into MRR≈0.4 on the same Wikipedia graph. Always state the sampler.
2. **AUC saturates on easy splits.** Every heuristic in §5.6 exceeds 0.93 AUC on USAir/E.coli; the graphs that separate methods are the sparse, triangle-poor ones (Power, Router). Same lesson as the IM file's budget discussion — report the regime where methods actually differ.
3. **EdgeBank is the floor nobody expects to matter.** A method that does not beat "predict what you saw before" has not demonstrated dynamics modelling [verified, §5.9: EdgeBank 0.580 vs DyRep 0.452 on tgbl-coin].
4. **Missing-rate curves are flat until ~60%.** FP: _"most methods perform extremely well up to 50% of missing features … the gap between methods opens up from around 60%"_ [verified, prose]. Reporting only 20–50% hides every difference.
5. **Do the ×2 check before claiming two graphs differ.** §6.3's `ogbn-products` case; the same arithmetic that resolved NetHEPT in [`influence_maximization.md`](influence_maximization.md) §6.4.
6. **Feature-agnostic baselines are not a formality.** Label Propagation and positional encodings beat both learned imputers in five of seven columns of §5.1 Table 2. Any imputation claim needs them in the table.

---

## 9. Implications for this project

**Do not implement graph completion as a task.** §2 gives the argument; this section gives the three things to actually do with the file.

### 9.1 Feature imputation is not applicable — state it and move on

Our model has **six input channels and none of them is a dataset feature** (`world_model/wm_data.py:16-17`). `ch_degree` is computed from `edge_index` inside `build_features` (`wm_data.py:174-180`), not read from a file. `data/generate_wm_data.py:92-99` writes `node_feats` and `node_labels` into every `graphs/*.npz`, and `load_graph_store` (`wm_data.py:225-241`) reads back only `edge_index`, `ic_probs`, `lt_weights`; `GraphInfo.from_store_entry` (`coding_agent/types.py:33-40`) reads only `num_nodes`, `edge_index`, `ic_probs`, `directed`. **The features are written to disk and read by nothing.**

Perfect imputation of Cora-ML's 2,879-dim bag-of-words would change zero reported numbers. That belongs in the limitations paragraph of any writeup, one sentence, with the channel count as evidence. It becomes false only if we ever build a content-conditioned transmission model (`p(u→v)` a function of node attributes) — not on the roadmap.

### 9.2 ⭐ The one experiment worth running: robustness to hidden edges

**Claim to test:** our world model and planner assume the observed adjacency is the true one. Real influence networks are not observed completely. How fast do we degrade?

**Protocol** (from §8.1, uniform edge deletion; SEAL's 10% as the low rung):

1. For `r ∈ {0, 0.10, 0.20, 0.50}` build `A_obs` by deleting a uniformly random `r` fraction of undirected edges from `A_full`. One seed per `r` to start; three seeds if the trend is noisy.
2. Generate transitions and train the world model on `A_obs` **only**.
3. Score against ground truth computed on `A_full` — the referee never sees the corruption. This is the whole design: the corruption is epistemic, the world is not.
4. Report, per `r`: one-step **`delta_f1`** and **`new_infection_f1`**; rollout **`ens_count_bias`** and **`ens_marg_mae`**; and **`planning_regret_multi`** with seeds chosen on `A_obs` and spread measured on `A_full`.
5. Baselines in the same table: `degree` and `random` computed on `A_obs` too (they degrade as well — the question is the _margin_), and the T2-GNN control (§5.2): **train on the corrupted graph and change nothing else**, because that paper's headline finding is that adding a repair module under joint corruption can be worse than doing nothing.

**Falsifiable prediction.** Our IC head is per-edge (`q(u→v) = σ(MLP([h_u, h_v, w_uv]))`) with a `frontier_u` gate, so hidden edges remove transmission channels rather than adding them. `ens_count_bias` should go **negative** and monotonically more so with `r` — the opposite of the saturation failure we fixed with the structured head (`MEMORY.md`, rollout-saturation entry). If it instead goes positive, the head is not using the adjacency the way we think it is, and that is a finding.

**Cost.** Four data-gen runs per graph on jazz / netscience / cora_ml / power_grid, no new model code — an edge-dropping flag in `data/wm_graphs.py` and a `--drop-edge-frac` argument threaded through `pipeline/run.py`. Everything downstream is dataset-agnostic already.

**Why these graphs.** `netscience` and `power_grid` are **byte-identical to SEAL's NS and Power** (§6.1), so the same experiment also yields a link-scoring row directly comparable to §5.6's published AUCs — the only place in this whole file where our suite meets the literature number-for-number.

### 9.3 Where structure completion actually fits us

Recovering edges **from cascades** is the version of this problem that is dynamic, action-conditioned, and fed by data our simulator already produces. That is [`network_inference.md`](network_inference.md) (NETINF, NETRATE, MultiTree, InfoPath), rated ⚠️ moderate. If the robustness experiment in §9.2 shows we degrade badly under hidden edges, the _fix_ lives in that file, not this one: infer the missing edges from observed cascades, then plan on the completed graph.

### 9.4 Two transferable lessons, independent of the task

- **Report the regime where methods separate.** FP: differences are invisible below 60% missing. SEAL: every heuristic is >0.93 AUC on dense graphs and ~0.58 on the power grid. Same shape as the IM file's argument for 1% and 5% budgets over 20%.
- **Include the memorization floor.** EdgeBank exists because a whole subfield reported gains over baselines that could not beat "predict what you already saw". Our analogue is the `persistence` baseline in `world_model/wm_eval.py` — already present, and it should stay in every table for the same reason.

---

## 10. Reference list

**Missing node features / attribute imputation** [FP (LoG'22, arXiv 2111.12128)](https://arxiv.org/abs/2111.12128) · [PMLR](https://proceedings.mlr.press/v198/rossi22a.html) · [code](https://github.com/twitter-research/feature-propagation) · [PCFI (ICLR'23, arXiv 2305.16618)](https://arxiv.org/abs/2305.16618) · [code](https://github.com/daehoum1/pcfi) · [GCNmf (FGCS'21, arXiv 2007.04583)](https://arxiv.org/abs/2007.04583) · [DOI](https://doi.org/10.1016/j.future.2020.11.016) · [code](https://github.com/marblet/GCNmf) · [SAT (TPAMI'22, arXiv 2011.01623)](https://arxiv.org/abs/2011.01623) · [code](https://github.com/xuChenSJTU/SAT-master-online) · [PaGNN (arXiv 2003.10130)](https://arxiv.org/abs/2003.10130) · [WGNN (arXiv 2102.03450)](https://arxiv.org/abs/2102.03450) · [Amer (IEEE TCYB'22)](https://ieeexplore.ieee.org/document/9765782) (202 to non-browser clients) · [ITR (IJCAI'22)](https://www.ijcai.org/proceedings/2022/485) · [PDF](https://www.ijcai.org/proceedings/2022/0485.pdf) · [code](https://github.com/WxTu/ITR) · [RITR (TNNLS'24, arXiv 2302.07524)](https://arxiv.org/abs/2302.07524) · [code](https://github.com/WxTu/RITR) · [MEGAE (AAAI'23, arXiv 2211.16771)](https://arxiv.org/abs/2211.16771) · [code](https://github.com/zqgao22/max-entropy-gae) · [T2-GNN (AAAI'23, arXiv 2212.12738)](https://arxiv.org/abs/2212.12738) · [code](https://github.com/jindi-tju/T2-GNN) · [AmGCL (arXiv 2305.03741)](https://arxiv.org/abs/2305.03741) · [FairAC (ICLR'23, arXiv 2302.12977)](https://arxiv.org/abs/2302.12977) · [AttriReBoost (arXiv 2501.00743)](https://arxiv.org/abs/2501.00743) · [CGAI (ACM MM'25, arXiv 2507.19085)](https://arxiv.org/abs/2507.19085) · [Divide-Then-Rule (arXiv 2507.10595)](https://arxiv.org/abs/2507.10595) · [DiffPuter (arXiv 2405.20690)](https://arxiv.org/abs/2405.20690) · [DDFI (arXiv 2512.06356)](https://arxiv.org/abs/2512.06356)

**Graph structure learning** [LDS (ICML'19)](https://proceedings.mlr.press/v97/franceschi19a.html) · [arXiv 1903.11960](https://arxiv.org/abs/1903.11960) · [code](https://github.com/lucfra/LDS-GNN) · [Pro-GNN (KDD'20)](https://dl.acm.org/doi/10.1145/3394486.3403049) (403 to non-browser clients) · [arXiv 2005.10203](https://arxiv.org/abs/2005.10203) · [code](https://github.com/ChandlerBang/Pro-GNN) · [IDGL (NeurIPS'20)](https://proceedings.neurips.cc/paper/2020/hash/e05c7ba4e087beea9410929698dc41a6-Abstract.html) · [arXiv 2006.13009](https://arxiv.org/abs/2006.13009) · [code](https://github.com/hugochan/IDGL) · [SLAPS (NeurIPS'21, arXiv 2102.05034)](https://arxiv.org/abs/2102.05034) · [code](https://github.com/BorealisAI/SLAPS-GNN) · [SUBLIME (WWW'22)](https://dl.acm.org/doi/10.1145/3485447.3512186) (403 to non-browser clients) · [arXiv 2201.06367](https://arxiv.org/abs/2201.06367) · [code](https://github.com/GRAND-Lab/SUBLIME) · [NodeFormer (NeurIPS'22)](https://openreview.net/forum?id=sMezXGG5So) · [arXiv 2306.08385](https://arxiv.org/abs/2306.08385) · [code](https://github.com/qitianwu/NodeFormer) · [GDC (NeurIPS'19, arXiv 1911.05485)](https://arxiv.org/abs/1911.05485)

**Static link prediction** [Liben-Nowell & Kleinberg 2003](https://www.cs.cornell.edu/home/kleinber/link-pred.pdf) · [Adamic & Adar 2003](https://www.sciencedirect.com/science/article/abs/pii/S0378873303000091) · [Resource Allocation (arXiv 0901.0553)](https://arxiv.org/abs/0901.0553) · [Katz 1953](https://link.springer.com/article/10.1007/BF02289026) · [SEAL (NeurIPS'18)](https://proceedings.neurips.cc/paper/2018/hash/53f0d7c537d99b3824f0f99d62ea2428-Abstract.html) · [arXiv 1802.09691](https://arxiv.org/abs/1802.09691) · [code](https://github.com/muhanzhang/SEAL) · [Neo-GNN (NeurIPS'21, arXiv 2206.04216)](https://arxiv.org/abs/2206.04216) · [code](https://github.com/seongjunyun/Neo-GNNs) · [ELPH/BUDDY (ICLR'23)](https://openreview.net/forum?id=m1oqEOAozQU) · [arXiv 2209.15486](https://arxiv.org/abs/2209.15486) · [code](https://github.com/melifluos/subgraph-sketching) · [NBFNet (NeurIPS'21, arXiv 2106.06935)](https://arxiv.org/abs/2106.06935) · [VGAE (arXiv 1611.07308)](https://arxiv.org/abs/1611.07308)

**Dynamic link prediction** [JODIE (KDD'19, arXiv 1908.01207)](https://arxiv.org/abs/1908.01207) · [code](https://github.com/srijankr/jodie) · [DyRep (ICLR'19)](https://openreview.net/forum?id=HyePrhR5KX) · [TGAT (ICLR'20)](https://openreview.net/forum?id=rJeW1yHYwH) · [arXiv 2002.07962](https://arxiv.org/abs/2002.07962) · [code](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs) · [TGN (arXiv 2006.10637)](https://arxiv.org/abs/2006.10637) · [code](https://github.com/twitter-research/tgn) · [CAWN (ICLR'21)](https://openreview.net/forum?id=KYPz4YsCPj) · [arXiv 2101.05974](https://arxiv.org/abs/2101.05974) · [code](https://github.com/snap-stanford/CAW) · [EdgeBank / DGB (arXiv 2207.10128)](https://arxiv.org/abs/2207.10128) · [code](https://github.com/fpour/DGB) · [GraphMixer (ICLR'23)](https://openreview.net/forum?id=ayPPc0SyLv1) · [arXiv 2302.11636](https://arxiv.org/abs/2302.11636) · [code](https://github.com/CongWeilin/GraphMixer) · [DyGFormer / DyGLib (NeurIPS'23 D&B, arXiv 2303.13047)](https://arxiv.org/abs/2303.13047) · [code](https://github.com/yule-BUAA/DyGLib) · [TCL (arXiv 2105.07944)](https://arxiv.org/abs/2105.07944) · [NAT (LoG'22, arXiv 2209.01084)](https://arxiv.org/abs/2209.01084) · [TGB (NeurIPS'23 D&B, arXiv 2307.01026)](https://arxiv.org/abs/2307.01026) · [site](https://tgb.complexdatalab.com/) · [code](https://github.com/shenyangHuang/TGB)

**Benchmarks, surveys, dataset sources** [OGB (arXiv 2005.00687)](https://arxiv.org/abs/2005.00687) · [nodeprop](https://ogb.stanford.edu/docs/nodeprop/) · [linkprop](https://ogb.stanford.edu/docs/linkprop/) · [leaderboards](https://ogb.stanford.edu/docs/leader_linkprop/) · [Shchur et al., Pitfalls of GNN Evaluation (arXiv 1811.05868)](https://arxiv.org/abs/1811.05868) · [code](https://github.com/shchur/gnn-benchmark) · [Kipf & Welling GCN (arXiv 1609.02907)](https://arxiv.org/abs/1609.02907) · [Planetoid](https://github.com/kimiyoung/planetoid) · [linqs Cora](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz) · [graph2gauss (our Cora-ML)](https://github.com/abojchevski/graph2gauss) · [JODIE datasets (SNAP)](https://snap.stanford.edu/jodie/) · [GSL survey (arXiv 2103.03036)](https://arxiv.org/abs/2103.03036) · [Incomplete Graph Learning survey (arXiv 2502.12412)](https://arxiv.org/abs/2502.12412) · [Incomplete-graph-learning reading list](https://github.com/cherry-a11y/Incomplete-graph-learning)

---

## 11. Open gaps

Honest list of what this review could **not** establish.

- **IDGL's own tables were never extracted.** Three separate download attempts (arXiv v1/v3, NeurIPS proceedings) returned truncated or corrupt PDFs. Every IDGL number in §5 is a **third-party reproduction** (T2-GNN Table 3, SLAPS Table 1, SUBLIME Tables 1–2). Those three agree with each other, so the numbers are probably right, but IDGL's own **edge-deletion robustness experiment** — the single most on-point published result for §9.2 — was not read and is not transcribed here.
- **Pro-GNN under random attack** is published only as Figure 4. The random-edge-injection curve, the closest published analogue to our uniform edge deletion, has no per-cell numbers.
- **GCNmf's, SAT's, Neo-GNN's and MEGAE's own result tables** were not transcribed. Their numbers appear here only as reported by FP, T2-GNN and BUDDY. GCNmf's PDF extracted its prose but not its tables; SAT's and Neo-GNN's downloads truncated.
- **The CiteSeer LCC disagreement** — Pro-GNN says 2,110 / 3,668, Shchur and FP say 2,120 / 3,679, all [verified] from their own tables. A 10-node gap with no stated cause. Not resolved.
- **Cora-ML's 2,879 feature dimension and 7 classes** are taken from our own loader docstring (`data/datasets/cora_ml.py`), not from a run of the loader or from Bojchevski & Günnemann's own table — the graph2gauss paper was not downloaded. Treat as **[claim]** until someone prints `node_feats.shape` after a load.
- **Diffusion-model feature imputation on graphs** is **[claim]** throughout. The concrete recent work (DDFI, FSD-CAP) is too new and too thin for a transcribed table; the well-benchmarked diffusion imputers (DiffPuter and friends) are tabular, not graph-native.
- **No paper found evaluates a _world model_ under hidden edges.** The robustness experiment in §9.2 has no direct precedent in this literature — GSL papers corrupt the graph and measure node classification; nobody corrupts the graph and measures _planning regret_. That makes §9.2 a novel setting, but it also means there is no baseline number to position it against.
- **DyRep has no public code.** Every DyRep number in §5.8 and §5.9 comes from a third-party reimplementation (DyGLib's and TGB's).
