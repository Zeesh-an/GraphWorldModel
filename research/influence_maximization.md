# Influence Maximization: Prior Work, Datasets, and Published Results

Unified reference for the IM literature relevant to this project: every method we might baseline against, every graph any of them evaluates on, published result tables with budgets and links and code, and the version forensics that determine which published numbers are legitimately comparable to ours.

This file merges the former `IM_RESEARCH.md` (methods and results) and `IM_DATASETS.md` (dataset catalogue). It is the reference implementation of the format every other file in this folder follows: see [`README.md`](README.md).

All URLs returned HTTP 200 on **2026-07-28** unless annotated otherwise.

---

## 0. Verification policy

| Tier           | Meaning                                                                                                              |
| -------------- | -------------------------------------------------------------------------------------------------------------------- |
| **[verified]** | Read from the paper's own table via text extraction, or from the data repository's own statistics page. Trustworthy. |
| **[derived]**  | Computed by us from a file we downloaded, or reconstructed arithmetically from published splits.                     |
| **[figure]**   | Read off a plotted figure: the paper published no table. Approximate, direction only.                               |
| **[claim]**    | Stated in prose by a paper or a secondary source; not cross-checked against a file or table.                         |

This matters more than it sounds. Automated PDF summarizers hallucinate plausible-looking numbers from IM papers. While compiling this file, a summarizer reported IRIE scoring `142.8` on NetHEPT at k=50; the actual published value in that paper's Table 3 is **724.67**. A second summarizer scrambled IMINFECTOR's Table 3 columns, which produced a false conclusion about the Weibo graph that took a primary-source check to undo (§6.4.5). Every number below marked [verified] was read from the paper's own table via `pdftotext -layout`.

**Edge-count convention.** Undirected graphs are quoted as _undirected edges_; directed graphs as _arcs_. Our loaders report `adjacency.nnz`, which for a symmetrized undirected graph is **2× the undirected edge count**. This single convention difference explains most apparent "discrepancies" between our numbers and published tables: check it before concluding two graphs differ.

---

## 1. Task definition

Given a graph `G = (V, E)`, a stochastic diffusion model, and a budget `k`, choose a seed set `S ⊆ V` with `|S| = k` maximizing the **expected spread** `σ(S)`: the expected number of nodes eventually activated when the diffusion is initiated from `S`.

```
S* = argmax_{S ⊆ V, |S| = k}  σ(S)
```

Kempe, Kleinberg & Tardos (KDD 2003) established the two things that define the field: the problem is **NP-hard**, and `σ` is **monotone and submodular** under both canonical diffusion models, so the greedy algorithm gives a `(1 − 1/e)` approximation. Everything since is either a faster way to evaluate `σ` or a learned substitute for it.

### The two canonical diffusion models

| Model                        | Determinism                    | Rule                                                                                                                                |
| ---------------------------- | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| **IC** (Independent Cascade) | stochastic                     | each newly activated `u` gets one chance to activate each out-neighbour `v` with probability `p(u→v)`. Monotone: no de-activation. |
| **LT** (Linear Threshold)    | deterministic given thresholds | `v` activates when the summed weight of its active in-neighbours ≥ its threshold `θ_v`; `θ_v` is drawn per node per episode.        |

Computing `σ(S)` exactly is **#P-hard** under both (Chen, Wang & Wang KDD 2010 for IC; Chen, Yuan & Zhang ICDM 2010 for LT), which is why every practical method either Monte-Carlo estimates it, bounds it, or learns it. That sub-literature (GLIE, MONSTOR, SIEA) was reviewed as its own task and not pursued: `world_model/wm_eval.py::rollout_ensemble` already prints the quantity it optimizes.

### Variants catalogued elsewhere in this folder

| Variant                                                                           | File                                                       |
| --------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Minimize a _negative_ cascade instead of maximizing a positive one                | [`influence_blocking.md`](influence_blocking.md)           |
| Seeds chosen in rounds with feedback; unknown edge probabilities; continuous time | [`adaptive_online_im.md`](adaptive_online_im.md)           |
| Estimating `σ(S)` as a task in its own right                                      | reviewed, not pursued: `rollout_ensemble` already prints it |
| Removing nodes to _reduce_ spread or connectivity                                 | [`critical_node_detection.md`](critical_node_detection.md) |
| Recovering `S` from an observed final state                                       | [`source_localization.md`](source_localization.md)         |

---

## 2. Fit with our methodology

**Status: implemented.** IM is the task this repo was built around, so this section documents what exists rather than what it would cost.

| Element          | How IM maps onto `f_θ(G, s_t, a_t) → s_{t+1}`                                                                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **State** `s_t`  | `(infected, frontier)` per node: ever-activated, and activated on this step. Channels 0-1 of `X`.                                                                              |
| **Action** `a_t` | `add_node(v)`: commit `v` to the seed set. At `t = 0` a whole seed set is committed at once; at `t > 0` an action is injected with probability `--inject-p`. Channel 3 of `X`. |
| **`T_exo`**      | deterministic: mark the targeted nodes active before the diffusion step.                                                                                                        |
| **`T_endo`**     | the IC or LT step itself, harvested from NDlib and learned by the structured head.                                                                                              |
| **Objective**    | `σ(S) = ` final `\|infected\|`, evaluated by ground-truth Monte-Carlo replay.                                                                                                   |
| **Planning**     | `planning_regret_multi`: use the model to pick a one-step intervention, score regret against the oracle.                                                                       |

The other four ops (`remove_node`, `remove_edge`, `add_edge`, `set_edge_weight`) are supported by the simulator and the feature builder but are **not exercised by pure IM**. That is the single strongest argument for adding [`influence_blocking.md`](influence_blocking.md) next: it is the same simulator and the same graphs, and it puts the idle three-quarters of the action space to work.

**What the world model buys here.** IM's cost is dominated by `σ` evaluation: greedy needs `O(k · N · R)` Monte-Carlo simulations. The learned model replaces each with one forward pass. Our six-condition comparison (`pipeline/conditions.py`) is built to measure exactly that trade: conditions 3-6 hold the _method_ fixed and vary only the evaluator (native / MC / oracle / world model), so the table isolates the evaluator's contribution.

---

## 3. Classical and heuristic methods

| Method              | Year | Venue  | Idea                                                                               | Paper                                                                                                                                                             | Code                                               |
| ------------------- | ---- | ------ | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| **Greedy + MC**     | 2003 | KDD    | Submodular greedy, (1−1/e) guarantee. Defines the problem.                         | [Kempe et al.](https://www.cs.cornell.edu/home/kleinber/kdd03-inf.pdf)                                                                                            |                                                    |
| **CELF**            | 2007 | KDD    | Lazy-forward marginal gains, ~700× faster than greedy                              | [Leskovec et al.](https://www.cs.cmu.edu/~jure/pubs/detect-kdd07.pdf)                                                                                             |                                                    |
| **Degree Discount** | 2009 | KDD    | Discount degree by already-chosen neighbours. Still the strongest cheap heuristic. | [Chen et al.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/weic-kdd09_influence.pdf)                                                       |                                                    |
| **PMIA / MIA**      | 2010 | KDD    | Maximum-influence-arborescence local trees                                         | [Chen et al.](https://www.microsoft.com/en-us/research/publication/scalable-influence-maximization-for-prevalent-viral-marketing-in-large-scale-social-networks/) |                                                    |
| **LDAG**            | 2010 | ICDM   | Local DAG under LT                                                                 | [Chen et al.](https://ieeexplore.ieee.org/document/5693962)                                                                                                       |                                                    |
| **CELF++**          | 2011 | WWW    | CELF with a secondary-gain cache                                                   | [Goyal et al.](https://snap.stanford.edu/class/cs224w-readings/goyal11celf.pdf)                                                                                   |                                                    |
| **SIMPATH**         | 2011 | ICDM   | LT-specific path enumeration                                                       | [Goyal et al.](https://ieeexplore.ieee.org/document/6137213)                                                                                                      |                                                    |
| **IRIE**            | 2012 | ICDM   | Influence-rank iteration + influence estimation, no MC                             | [Jung et al. (arXiv 1111.4795)](https://arxiv.org/abs/1111.4795)                                                                                                  |                                                    |
| **StaticGreedy**    | 2013 | CIKM   | Fixed live-edge snapshots, ~2 orders faster                                        | [Cheng et al. (arXiv 1212.4779)](https://arxiv.org/abs/1212.4779)                                                                                                 |                                                    |
| **RIS**             | 2014 | SODA   | Reverse Influence Sampling: the foundation of everything below                    | [Borgs et al.](https://arxiv.org/abs/1212.0884)                                                                                                                   |                                                    |
| **TIM / TIM+**      | 2014 | SIGMOD | RIS with estimated sample size, near-optimal time                                  | [Tang et al. (arXiv 1404.0900)](https://arxiv.org/abs/1404.0900)                                                                                                  | [code](https://sourceforge.net/projects/timplus/)  |
| **SKIM**            | 2014 | CIKM   | Sketch-based reachability                                                          | [Cohen et al.](https://arxiv.org/abs/1408.6282)                                                                                                                   |                                                    |
| **IMM**             | 2015 | SIGMOD | Martingale RIS. **The standard classical reference point.**                        | [Tang et al.](https://dl.acm.org/doi/10.1145/2723372.2723734)                                                                                                     | [code](https://sourceforge.net/projects/im-imm/)   |
| **SSA / D-SSA**     | 2016 | SIGMOD | Stop-and-stare RIS, fewer samples                                                  | [Nguyen et al. (arXiv 1605.07990)](https://arxiv.org/abs/1605.07990)                                                                                              | [code](https://github.com/hungnt55/Stop-and-Stare) |
| **OPIM-C**          | 2018 | SIGMOD | Online processing RIS, near-optimal anytime                                        | [Tang et al.](https://dl.acm.org/doi/10.1145/3183713.3183749)                                                                                                     | [code](https://github.com/tangj90/OPIM)            |
| **SubSIM**          | 2020 | SIGMOD | Sublinear-time RIS                                                                 | [Guo et al.](https://dl.acm.org/doi/10.1145/3318464.3389740)                                                                                                      | [code](https://github.com/qtguo/subsim)            |

Our library (`coding_agent/tools/algorithms.py`) implements simplified versions of most of these: see `research_notes/ALGORITHMS.md` for per-function fidelity notes.

---

## 4. Learning-based methods

| Method                 | Year      | Venue           | Approach                                                                                             | Paper                                                                                                                 | Code                                                                     |
| ---------------------- | --------- | --------------- | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| **Inf2vec**            | 2018      | ICDE            | Cascade + network embedding                                                                          | [Feng et al.](https://ieeexplore.ieee.org/document/8509310)                                                           |                                                                          |
| **IMINFECTOR**         | 2020      | TKDE            | Multi-task NN on real cascades; influencer + susceptible vectors. **Model-free**: no IC/LT assumed. | [arXiv 1904.08804](https://arxiv.org/abs/1904.08804)                                                                  | [github.com/geopanag/IMINFECTOR](https://github.com/geopanag/IMINFECTOR) |
| **DISCO**              | 2019      |                 | structure2vec + DQN, marginal-gain reward                                                            | [Li et al.](https://arxiv.org/abs/1906.07378)                                                                         |                                                                          |
| **GCOMB**              | 2020      | NeurIPS         | GraphSAGE pruning + Q-learning; scales to billion-edge                                               | [Manchanda et al.](https://proceedings.neurips.cc/paper/2020/hash/e7532dbeff7ef901f2e70daacb3f452d-Abstract.html)     | [github.com/idea-iitd/GCOMB](https://github.com/idea-iitd/GCOMB)         |
| **PIANO**              | 2022      | TCSS            | Evolved DISCO with fuller analysis                                                                   | [Li et al.](https://ieeexplore.ieee.org/document/9769766)                                                             |                                                                          |
| **GLIE**               | 2021→2023 | ASONAM          | GNN learns an upper bound on spread; replaces MC                                                     | [arXiv 2108.04623](https://arxiv.org/abs/2108.04623)                                                                  | [github.com/geopanag/learn_im](https://github.com/geopanag/learn_im)     |
| **LeNSE**              | 2022      | ICML            | Learns to prune to a solvable subgraph                                                               | [Ireland & Montana](https://proceedings.mlr.press/v162/ireland22a.html)                                               | [github.com/davidireland3/LeNSE](https://github.com/davidireland3/LeNSE) |
| **ToupleGDD**          | 2023      | TCSS            | 3 coupled GNNs + Double DQN. Trains on <300-node graphs, generalizes to 1M+.                         | [arXiv 2210.07500](https://arxiv.org/abs/2210.07500)                                                                  | [github.com/Dtrycode/ToupleGDD](https://github.com/Dtrycode/ToupleGDD)   |
| **DeepIM** (key)          | 2023      | **ICML**        | Autoencoder over seed sets + GNN diffusion; optimizes in latent space. IC/LT/SIS.                    | [arXiv 2305.02200](https://arxiv.org/abs/2305.02200) · [PMLR](https://proceedings.mlr.press/v202/ling23b/ling23b.pdf) | [github.com/triplej0079/DeepIM](https://github.com/triplej0079/DeepIM)   |
| **MOEIM** (key)           | 2024      | GECCO           | Many-objective EA (spread, budget, fairness, communities, time) with graph-aware operators           | [arXiv 2403.18755](https://arxiv.org/abs/2403.18755)                                                                  | [github.com/eliacunegatti/MOEIM](https://github.com/eliacunegatti/MOEIM) |
| **DeepIM-accelerated** | 2024      | Neural Networks | DeepIM with faster inference                                                                         | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0893608024005732)                              |                                                                          |
| **HIM**                | 2025      |                 | Influence strength estimation in hyperbolic space                                                    | [arXiv 2502.13571](https://arxiv.org/abs/2502.13571)                                                                  |                                                                          |
| **REM**                | 2025      |                 | Seed2Vec VAE + RL, **multiplex** networks (different problem)                                        | [arXiv 2501.00779](https://arxiv.org/abs/2501.00779)                                                                  |                                                                          |
| **Topic-aware IM**     | 2025      | DAMI            | GAT + DRL, magnetic Laplacian PE                                                                     | [Springer](https://link.springer.com/article/10.1007/s10618-025-01133-3)                                              |                                                                          |

**Surveys:** [ML-based IM survey, TKDD 2023 (arXiv 2211.03074)](https://arxiv.org/abs/2211.03074) · [IM survey 2023 (arXiv 2309.04668)](https://arxiv.org/abs/2309.04668) · [Deep-RL for max-coverage benchmark (arXiv 2406.14697)](https://arxiv.org/abs/2406.14697)

---

## 5. Published results

### 5.1 DeepIM (ICML 2023) (key) the most comparable table

**The single most directly comparable published result to our setup**: same budget convention (1/5/10/20% of nodes), same IC weighted-cascade probability (`p = 1/in-degree`), same reported metric (**% of nodes infected**), and two of its graphs are byte-identical to ours.

Protocol: seed 1/5/10/20% of nodes, simulate to termination, average influence spread over **100 rounds**. LT thresholds ~ U[0.3, 0.6]. `−` = out-of-memory.

The **OIM** row is Lei, Maniu, Mo, Cheng & Senellart, _"Online Influence Maximization"_, KDD 2015 [verified, from DeepIM's own reference list]: [code](https://github.com/smaniu/oim). It is the only sequential-decision method in DeepIM's comparison; see [`adaptive_online_im.md`](adaptive_online_im.md).

Dataset sizes **as used by DeepIM** (Table 1) [verified]:

|       | Jazz  | Cora-ML | Network Science | Power Grid | Synthetic | Digg      | Weibo       |
| ----- | ----- | ------- | --------------- | ---------- | --------- | --------- | ----------- |
| Nodes | 198   | 2,810   | 1,565           | 4,941      | 50,000    | 279,613   | 2,251,166   |
| Edges | 2,742 | 7,981   | 13,532          | 6,594      | 250,000   | 1,170,689 | 225,877,808 |

> The Digg and Weibo columns contain a transcription error: see §6.4.5.

#### IC diffusion: % of nodes infected [verified, Table 2]

| Method        | Cora-ML 1% | 5%       | 10%      | 20%      | NetSci 1% | 5%       | 10%      | 20%      | PowerGrid 1% | 5%       | 10%      | 20%      | Jazz 1% | 5%       | 10%      | 20%  |
| ------------- | ---------- | -------- | -------- | -------- | --------- | -------- | -------- | -------- | ------------ | -------- | -------- | -------- | ------- | -------- | -------- | ---- |
| IMM           | 8.1        | 26.2     | 37.3     | 50.2     | 5.2       | 16.8     | 26.0     | 45.7     | 4.3          | 17.4     | 31.5     | 51.1     | 2.6     | 20.1     | 34.4     | 42.8 |
| OPIM          | 13.4       | 26.9     | 37.4     | 50.9     | 6.6       | 19.4     | 28.9     | 48.6     | 5.7          | 17.7     | 29.7     | 50.1     | 2.4     | 20.1     | 34.4     | 46.8 |
| SubSIM        | 10.1       | 25.7     | 36.8     | 51.1     | 4.8       | 15.4     | 27.9     | 44.8     | 4.6          | 19.2     | 31.7     | 50.2     | 3.6     | 18.8     | 37.6     | 44.7 |
| OIM           | 8.9        | 27.6     | 38.0     | 51.3     | 4.2       | 16.7     | 26.5     | 48.2     | 5.7          | 17.5     | 31.9     | 50.8     | 2.0     | 18.5     | 36.3     | 42.2 |
| IMINFECTOR    | 9.6        | 26.8     | 37.7     | 50.6     | 5.4       | 17.9     | 27.8     | 47.6     | 5.4          | 18.2     | 31.6     | 50.9     | 3.6     | 19.7     | 37.5     | 45.9 |
| PIANO         | 9.8        | 25.2     | 37.4     | 51.1     | 4.7       | 16.3     | 27.1     | 47.2     | 5.3          | 18.1     | 31.7     | 50.2     | 2.2     | 19.2     | 36.6     | 43.2 |
| ToupleGDD     | 10.6       | 27.5     | 38.5     | 51.5     | 6.3       | 17.8     | 28.3     | 50.5     | 5.4          | 19.3     | 31.6     | 51.3     | 3.3     | 20.4     | 37.2     | 45.7 |
| DeepIM_s      | 13.6       | 27.7     | 38.5     | 51.8     | 6.9       | 19.1     | 29.3     | 50.5     | 5.9          | 20.2     | 31.7     | 51.5     | 4.6     | 22.4     | 41.4     | 49.9 |
| **DeepIM** (key) | **14.1**   | **28.1** | **39.6** | **52.4** | **7.8**   | **20.9** | **31.5** | **51.2** | **6.3**      | **21.0** | **32.5** | **52.4** | **4.9** | **23.3** | **41.5** | 49.9 |

Large graphs, IC [verified]:

| Method        | Synthetic 1% | 5%       | 10%      | 20%      | Digg 1% | 5%       | 10%      | 20%      | Weibo 1% | 5%       | 10%      | 20%      |
| ------------- | ------------ | -------- | -------- | -------- | ------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| IMM           | 9.2          | 26.2     | 36.3     | 51.2     | 7.4     | 18.4     | 32.8     | 46.9     | 9.5      | 23.8     | 36.4     | 50.3     |
| OPIM          | 9.6          | 25.3     | 36.6     | 51.7     | 7.6     | 18.5     | 32.9     | 48.9     | 9.7      | 23.7     | 36.6     | 50.3     |
| SubSIM        | 9.5          | 26.7     | 36.5     | 51.5     | 7.5     | 18.9     | 33.3     | 49.4     | 9.3      | 23.1     | 36.5     | 50.6     |
| OIM           | 9.6          | 26.2     | 36.7     | 51.3     | 7.8     | 18.2     | 33.1     | 49.6     |          |          |          |          |
| IMINFECTOR    | 9.1          | 26.2     | 36.1     | 51.5     | 7.9     | 18.6     | 33.5     | 49.8     | 9.4      | 23.5     | 36.9     | 50.3     |
| PIANO         | 9.1          | 26.4     | 36.2     | 51.6     |         |          |          |          |          |          |          |          |
| ToupleGDD     | 9.5          | 26.8     | 37.1     | 51.4     |         |          |          |          |          |          |          |          |
| **DeepIM** (key) | **11.6**     | **27.4** | **38.7** | **52.1** | **8.4** | **19.3** | **34.2** | **51.3** | **11.2** | **26.5** | **37.9** | **51.8** |

#### LT diffusion: % of nodes infected [verified, Table 3]

| Method        | Cora-ML 1% | 5%       | 10%      | 20%      | NetSci 1% | 5%       | 10%      | 20%      | PowerGrid 1% | 5%       | 10%      | 20%      | Jazz 1% | 5%      | 10%      | 20%      |
| ------------- | ---------- | -------- | -------- | -------- | --------- | -------- | -------- | -------- | ------------ | -------- | -------- | -------- | ------- | ------- | -------- | -------- |
| IMM           | 1.7        | 34.8     | 52.2     | 66.4     | 2.5       | 11.9     | 18.1     | 33.6     | 4.6          | 19.9     | 31.7     | 56.9     | 1.4     | 5.7     | 13.4     | 24.5     |
| OPIM          | 2.3        | 36.9     | 51.2     | 71.5     | 1.6       | 12.0     | 18.8     | 34.1     | 4.4          | 17.6     | 30.1     | 55.5     | 1.4     | 6.9     | 12.6     | 20.9     |
| SubSIM        | 1.7        | 33.6     | 54.7     | 70.1     | 1.8       | 10.4     | 19.2     | 34.1     | 4.5          | 21.1     | 31.2     | 57.4     | 1.4     | 5.9     | 11.4     | 21.2     |
| IMINFECTOR    | 2.1        | 33.9     | 51.3     | 70.6     | 2.1       | 11.8     | 18.7     | 34.5     | 4.2          | 21.3     | 31.6     | 56.2     | 1.4     | 6.2     | 13.5     | 22.8     |
| PIANO         | 2.1        | 33.5     | 53.3     | 69.8     | 2.1       | 11.3     | 19.1     | 33.9     | 4.3          | 21.3     | 31.4     | 57.1     | 1.1     | 6.2     | 12.1     | 22.4     |
| ToupleGDD     | 2.3        | 36.2     | 54.5     | 70.9     | 2.8       | 12.4     | 19.8     | 34.6     | 4.8          | 21.9     | 32.6     | 58.1     | 1.4     | 6.5     | 12.9     | 23.6     |
| DeepIM_s      | 10.7       | 65.6     | 75.1     | 85.2     | 3.5       | 14.6     | 23.8     | 37.8     | 6.1          | 24.1     | 45.2     | 71.5     | 1.9     | 6.5     | 16.1     | 97.1     |
| **DeepIM** (key) | **13.4**   | **69.2** | **83.5** | **94.1** | **4.1**   | **16.6** | **26.7** | **41.5** | **6.3**      | **24.4** | **46.8** | **71.7** | **1.9** | **6.5** | **16.4** | **99.1** |

Large graphs, LT [verified]:

| Method        | Synthetic 1% | 5%      | 10%      | 20%      | Digg 1% | 5%       | 10%      | 20%      | Weibo 1% | 5%      | 10%      | 20%      |
| ------------- | ------------ | ------- | -------- | -------- | ------- | -------- | -------- | -------- | -------- | ------- | -------- | -------- |
| IMM           | 1.1          | 5.2     | 13.1     | 66.9     | 2.4     | 10.8     | 37.4     | 55.6     | 1.6      | 6.7     | 19.3     | 45.2     |
| OPIM          | 1.3          | 5.2     | 12.6     | 62.1     | 2.1     | 11.3     | 38.2     | 57.1     | 1.8      | 6.1     | 18.7     | 46.6     |
| SubSIM        | 1.4          | 5.5     | 13.1     | 69.6     | 2.4     | 11.3     | 37.9     | 56.9     | 1.7      | 6.7     | 19.2     | 46.8     |
| IMINFECTOR    | 1.3          | 5.5     | 13.5     | 67.4     | 2.2     | 11.1     | 38.9     | 58.7     | 1.8      | 6.4     | 18.6     | 47.5     |
| ToupleGDD     | 1.3          | 5.5     | 13.4     | 70.2     |         |          |          |          |          |         |          |          |
| **DeepIM** (key) | **1.5**      | **6.5** | **15.5** | **99.9** | **3.5** | **15.9** | **41.3** | **76.2** | **3.1**  | **7.6** | **39.3** | **72.4** |

#### Reading these tables

- **The LT gap is where DeepIM's headline claim lives.** Cora-ML LT at 20%: DeepIM 94.1% vs best baseline 71.5% (OPIM). Synthetic LT at 20%: 99.9% vs 70.2%. Jazz LT at 20%: 99.1% vs 24.5%. These are the ~200% improvements the paper advertises.
- **Under IC the field is nearly tied.** At 20% budget every method on Cora-ML lands in 50.2-52.4%. DeepIM's IC margin is ~1-2 points. This is the same saturation effect we hit on BA-100: **at 20% budget on a well-connected graph, the problem is close to solved by any reasonable method.** It is a strong argument for reporting the 1% and 5% columns, where the spread between methods is 8.1→14.1 (74% relative).
- **1% budget is where methods actually separate.** Cora-ML IC 1%: IMM 8.1 vs DeepIM 14.1.

### 5.2 MOEIM (GECCO 2024) (key) beats DeepIM

MOEIM is a many-objective evolutionary algorithm (spread ↑, seed-set size ↓, communities ↑, fairness ↑, budget ↓, time ↓) with graph-aware mutation and smart initialization. It is the most recent method found that **directly compares against DeepIM on our datasets** and claims to beat it.

Setting-2 datasets (chosen to match DeepIM exactly) [verified, Table 1]: Jazz 198/2,742 · Cora-ML 2,810/7,981 · Power Grid 4,941/6,594. Budgets k ∈ {1%, 5%, 10%, 20%}, propagation WC and LT, `τ = ∞`.

**Result [claim + figure]:** MOEIM outperforms DeepIM on _almost all_ propagation models and datasets. Reported specifics:

- **Jazz + LT**: MOEIM reaches whole-network influence at **k = 10%**, where DeepIM needs **k = 20%** for the same result. (DeepIM's own table shows Jazz LT 16.4% at k=10 and 99.1% at k=20, so this is a large gap.)
- **Power Grid + LT**: the one exception: near-identical up to k = 10%, after which **DeepIM is superior**.
- Setting-1 (6 other graphs, hypervolume over 6 objectives): MOEIM wins 61 of 72 cases vs GDD, CELF, MOEA.

Per-cell spread numbers are published only as non-dominated fronts in Figure 2: no table. To use MOEIM as a numeric baseline you would need to run [their code](https://github.com/eliacunegatti/MOEIM).

MOEIM's setting-1 datasets [verified]: email-eu-Core 986/25,552 · facebook-combined 4,039/88,234 · gnutella 6,299/20,776 · wiki-vote 7,066/103,663 · lastfm 7,624/27,806 · CA-HepTh 8,638/24,827. **We now load four of these six**, see §6.5.

### 5.3 Classical methods on NetHEPT

NetHEPT (arXiv High-Energy-Physics-Theory collaboration) is _the_ classical IM benchmark. Standard protocol: **k = 1…50**, IC with weighted cascade `p = 1/in-degree`, spread averaged over 10,000 MC simulations.

**IRIE paper, k = 50 [verified, Table 3]**: "ArXiv" here is NetHEPT (15,233 / 58,891):

| Dataset     | Model | SAEDV    | IRIE         |
| ----------- | ----- | -------- | ------------ |
| **NetHEPT** | WC    | 669.76   | **724.67**   |
| **NetHEPT** | TR    | 185.37   | **190.01**   |
| Epinions    | WC    | 11,177.3 | **12,063**   |
| Slashdot    | WC    | 14,803   | **16,712.3** |
| Amazon      | WC    | 487.67   | **824.80**   |
| DBLP        | WC    | 33,730   | **53,334.8** |

**NetHEPT k=50 under WC ≈ 700-725 nodes ≈ 4.6-4.8% of the graph** [figure, Fig 4.3a]: Greedy/CELF, PMIA, IR, IRIE all converge to ≈700-730; **Degree alone reaches only ≈250**, a 3× gap that makes NetHEPT a genuinely discriminative benchmark, unlike BA at high budget.

LiveJournal, k = 50 [verified, Table 2]:

| Algorithm | Weighted Cascade | Trivalency  |
| --------- | ---------------- | ----------- |
| IR        | **75,861.2**     | 629,484     |
| IRIE      | 74,830.5         | **629,694** |
| PMIA      | 71,566.5         | 629,512     |
| PageRank  | 51,162.3         | **629,892** |
| Degree    | 52,162.3         | 629,498     |

> Note the budget convention clash: classical IM sweeps **absolute k = 1…50** (k=50 is ~0.33% of NetHEPT), while the learning-based line sweeps **percentages, 1-20%** (20% of NetHEPT would be k=3,046). See §8.

### 5.4 ToupleGDD (TCSS 2023)

Datasets [verified, Table I]: almost entirely disjoint from ours, **except YouTube, wiki-Vote and ca-GrQc**:

| Dataset        | n         | m      | Type       | Avg degree |
| -------------- | --------- | ------ | ---------- | ---------- |
| soc-dolphins   | 62        | 159    | directed   | 5          |
| Twitter        | 0.8k      | 1k     | directed   | 2          |
| Wiki-1         | 0.9k      | 3k     | directed   | 6          |
| caGr yes        | 4.2k      | 13.4k  | undirected | 5          |
| Wiki-2 yes      | 7.1k      | 103.7k | directed   | 29         |
| Epinions       | 76k       | 509k   | directed   | 13         |
| Buzznet        | 101k      | 3M     | directed   | 55         |
| **YouTube** yes | **1.13M** | **3M** | undirected | 5          |

Budgets: **b ∈ {10, 20, 30, 40, 50}** (absolute). Edge weights: in-degree (= weighted cascade), plus 0.1 and 0.5 uniform settings for generalization tests.

Expected spread, in-degree setting, iterative selection, train+test with initial embedding [verified, Table II]:

| Dataset        | b=10     | b=20     | b=30     | b=40      | b=50      |
| -------------- | -------- | -------- | -------- | --------- | --------- |
| Twitter (0.8k) | 147.71   | 210.86   | 252.11   | 287.59    | 315.88    |
| caGr (4.2k)    | 213.13   | 368.74   | 489.15   | 602.95    | 696.95    |
| Wiki-2 (7.1k)  | 290.48   | 423.96   | 521.79   | 601.39    | 669.43    |
| Epinions (76k) | 6,022.85 | 8,303.34 | 9,693.69 | 10,866.88 | 11,781.69 |

**YouTube results are figure-only** [figure, Fig 3i]: all methods (IMM, OPIM-C, ToupleGDD, S2V-DQN, PIANO, GCOMB) reach ≈50,000-60,000 spread at b=50, i.e. ≈5% of the 1.13M nodes. ToupleGDD ≈ IMM, both above OPIM-C.

**Claim:** ToupleGDD achieves spread "almost equal to IMM", outperforms OPIM-C on Wiki-2/Buzznet/YouTube, and beats all other DRL methods. Its selling point is **generalization**, trained on tiny random graphs, tested on 1M-node graphs.

### 5.5 IMINFECTOR (TKDE 2020): a different protocol

**Do not compare its numbers to simulated spread.** IMINFECTOR never assumes IC/LT. It splits real cascades 80/20 by time, picks seeds from train, and measures **DNI (Distinct Nodes Influenced)** = the union of nodes appearing in _held-out real cascades_ started by the chosen seeds.

Seed set sizes [verified]: **Digg k=50 · Weibo k=1,000 · MAG k=10,000.**

Results [figure, Fig 7]:

| Dataset         | Best method                  | ≈DNI at max k   | Notes                                            |
| --------------- | ---------------------------- | --------------- | ------------------------------------------------ |
| Digg (k=50)     | **Credit Distribution** ≈43k | IMINFECTOR ≈42k | CD wins but is ~10× slower; CELFIE flat at ≈31k  |
| MAG (k=10,000)  | **IMINFECTOR** ≈235k         | CELFIE ≈160k    | IMM-DB / Simpath-DB ≈110-125k                    |
| Weibo (k=1,000) | **IMINFECTOR** ≈450k         | K-cores ≈400k   | Only IMINFECTOR + K-cores + CELFIE scaled at all |

Notable finding the paper stresses: **IMM underperformed badly on cascade-based evaluation**, which the authors attribute to diffusion-model misspecification, "it performs poorly in this type of evaluation… IMM optimizes diffusion simulations as part of its solution." Directly relevant to us: a method tuned on a _simulator_ can lose on _real observed traces_. See [`cascade_prediction.md`](cascade_prediction.md), where that failure mode is the central issue.

### 5.6 SOTA summary per dataset

| Dataset             | Best published                                          | Model | Status                                                                |
| ------------------- | ------------------------------------------------------- | ----- | --------------------------------------------------------------------- |
| **Jazz**            | **MOEIM** (2024) > DeepIM (2023)                        | WC/LT | MOEIM reaches full spread at k=10% vs DeepIM's 20%                    |
| **Cora-ML**         | **MOEIM** (2024) > DeepIM (2023)                        | WC/LT | DeepIM LT@20% = 94.1%, IC@20% = 52.4%                                 |
| **Power Grid**      | **DeepIM** at k≥10% under LT; MOEIM below that          | WC/LT | The one dataset where DeepIM holds                                    |
| **Network Science** | **DeepIM** (2023)                                       | IC/LT | but on the 13,532-edge version, not ours (§6.4.6)                     |
| **NetHEPT**         | IMM/OPIM-C/IRIE tier (classical)                        | IC-WC | ≈725 nodes @ k=50; no learning-based paper reports it in the % regime |
| **Digg / Weibo**    | **DeepIM** (simulated) / **IMINFECTOR** (cascade-based) |       | two incomparable protocols                                            |
| **YouTube**         | IMM ≈ ToupleGDD                                         | IC    | figure-only                                                           |
| **ER / BA / WS**    | ToupleGDD, GCOMB, DeepIM (synthetic)                    | IC/LT | training/generalization only, no canonical table                      |
| **SBM / Karate**    | **none found**                                          |       | no published IM baseline                                              |

**Overall ranking (single-layer IM, simulated-spread evaluation):**

```
MOEIM (GECCO'24)          ← current best on Jazz/Cora-ML
DeepIM (ICML'23)          ← best DL method; the reference everyone now cites
ToupleGDD (TCSS'23)       ← best generalization; ≈IMM quality
GCOMB (NeurIPS'20)        ← best scalability among learned
PIANO / DISCO             ← weaker, unstable across graphs
IMINFECTOR (TKDE'20)      ← best when only real cascades exist (different protocol)
S2V-DQN                   ← weakest DRL baseline
────────────────────────────────────────────
IMM / OPIM-C / SubSIM     ← classical ceiling, near-optimal guarantees
CELF / degree-discount    ← cheap strong baselines, tie at high budget
```

---

## 6. Datasets

### 6.0 What `--dataset` accepts for this task

**Every one of the 71 real graph loaders and all 7 synthetic families (`er`, `ba`, `ws`, `sbm`, `powerlaw_cluster`, `kronecker`, `karate`) runs on this task.** The loaders are task-agnostic; nothing in `pipeline/tasks.py` restricts a task to a dataset. The lists below are therefore an experimental CHOICE, not a constraint, and the only place that choice is currently encoded is `sbatch/`. Where the sweep and the benchmark family disagree, the sweep is the accident and the family is the intent.

| | Datasets |
| --- | --- |
| **Benchmark family** (§6) | `jazz`, `email_eu_core`, `netscience`, `cora_ml`, `facebook`, `power_grid`, `ca_grqc`, `wiki_vote`, `lastfm_asia`, `nethept`, `netphy` |
| **Synthetic** | `er`, `ba`, `ws`, `sbm`, `karate`, `powerlaw_cluster`, `kronecker` |
| **Scale targets** (load, too slow to simulate today) | `twitter`, `digg`, `youtube`, `epinions`, `orkut`, `livejournal`, `weibo` |
| **In `sbatch/influence_maximization/` today** | `ba`, `sbm`, `jazz`, `netscience`, `power_grid`, `nethept` |

There is no `sweep_datasets.sbatch` for this task; the six above are its `generate_data_*.sbatch` targets. `karate`, `er` and `ws` are fully supported and simply have no generation script yet.

`data/datasets/<name>.py`, dispatched by `data/wm_graphs.py::real_directed`. Raw downloads land in `data/raw/<dataset>/` (gitignored, shared across runs).

### 6.1 What we already load

**Real graphs (15).** All counts below were **produced by running the loader**, not transcribed. Undirected rows quote undirected edges; directed rows quote arcs.

| `--dataset`     | Nodes     | Edges                        | Type                                 | Node features                                  | Source                                                                                                                                                 | Auto-DL       |
| --------------- | --------- | ---------------------------- | ------------------------------------ | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------- |
| `jazz`          | 198       | 2,742                        | Undirected (collaboration)           | log1p(degree)                                  | [nrvis `arenas-jazz.zip`](https://nrvis.com/download/data/misc/arenas-jazz.zip) · [page](https://networkrepository.com/arenas-jazz.php)                | yes            |
| `email_eu_core` | 1,005     | 24,929 arcs                  | Directed (emails)                    | log1p(total degree) · **42 department labels** | [SNAP email-Eu-core](https://snap.stanford.edu/data/email-Eu-core.html)                                                                                | yes            |
| `netscience`    | 1,589     | 2,742                        | Undirected (coauthorship)            | log1p(degree)                                  | [Netzschleuder `netscience`](https://networks.skewed.de/net/netscience)                                                                                | yes            |
| `cora_ml`       | 2,810     | 7,981                        | Undirected (citations, standardized) | 2,879-dim bag-of-words, 7 labels               | [graph2gauss `cora_ml.npz`](https://github.com/abojchevski/graph2gauss/raw/master/data/cora_ml.npz)                                                    | yes            |
| `facebook`      | 4,039     | 88,234                       | Undirected (friendships)             | log1p(degree)                                  | [SNAP ego-Facebook](https://snap.stanford.edu/data/ego-Facebook.html)                                                                                  | yes            |
| `power_grid`    | 4,941     | 6,594                        | Undirected (power lines)             | log1p(degree)                                  | [nrvis `opsahl-powergrid.zip`](https://nrvis.com/download/data/misc/opsahl-powergrid.zip) · [page](https://networkrepository.com/opsahl-powergrid.php) | yes            |
| `ca_grqc`       | 5,242     | 14,484                       | Undirected (coauthorship)            | log1p(degree)                                  | [SNAP ca-GrQc](https://snap.stanford.edu/data/ca-GrQc.html)                                                                                            | yes            |
| `wiki_vote`     | 7,115     | 103,689 arcs                 | Directed (adminship votes)           | log1p(total degree)                            | [SNAP wiki-Vote](https://snap.stanford.edu/data/wiki-Vote.html)                                                                                        | yes            |
| `lastfm_asia`   | 7,624     | 27,806                       | Undirected (mutual follows)          | log1p(degree) · **18 country labels**          | [SNAP feather-lastfm-social](https://snap.stanford.edu/data/feather-lastfm-social.html)                                                                | yes            |
| `nethept`       | 15,229    | 62,752 arcs                  | Directed (coauthorship, both arcs)   | log1p(total degree)                            | [SparklyYS/Simultaneous-IMM mirror](https://github.com/SparklyYS/Simultaneous-IMM)                                                                     | yes            |
| `netphy`        | 37,154    | 174,161                      | Undirected (coauthorship)            | log1p(degree)                                  | [Wei Chen `weic-graphdata.zip`](https://www.microsoft.com/en-us/research/people/weic/selected-projects/)                                               | yes            |
| `twitter`       | 81,306    | 1,768,149 arcs → symmetrized | Undirected (follows)                 | log1p(degree)                                  | [SNAP ego-Twitter](https://snap.stanford.edu/data/ego-Twitter.html)                                                                                    | yes            |
| `digg`          | 116,893   | ≈2.6M                        | Undirected (friendships)             | log1p(degree)                                  | [Syracuse `Digg-dataset.zip`](https://datasets.syr.edu/datasets/Digg.html)                                                                             | yes            |
| `youtube`       | 1,134,890 | 2,987,624                    | Undirected (friendships)             | log1p(degree)                                  | [SNAP com-Youtube](https://snap.stanford.edu/data/com-Youtube.html)                                                                                    | yes            |
| `weibo`         | 1,787,443 | ≈216M arcs                   | Directed (influence u→v)             | log1p(total degree)                            | [AMiner Influence Locality](https://www.aminer.cn/influencelocality)                                                                                   | no **manual** |

**Simulable today:** everything down to and including `netphy`, eleven graphs. The four large ones (`twitter`, `digg`, `youtube`, `weibo`) load fine but exceed what the NDlib rollout + CELF/local-search selector pipeline finishes in reasonable time; they are scalable-simulation targets, not day-one datasets.

**Weibo needs a human**: AMiner requires registration. See `data/datasets/weibo.py` for the three-step manual procedure.

**Cora-ML is standardized on load.** `data/datasets/cora_ml.py` applies graph2gauss's `SparseGraph.standardize()` (`make_unweighted`, `make_undirected`, `no_self_loops`, `select_lcc`: all defaulting to True), so it produces DeepIM's and MOEIM's 2,810 / 7,981 exactly rather than the raw file's 2,995 / 8,416. The raw graph is not comparable to any published table, its 185 extra nodes are ~60 disconnected fragments that only add noise to a spread metric, and the 2,879-dim features that distinguished it are never read by the model (`IN_CHANNELS = 6`; `build_features` derives `CH_DEGREE` from `edge_index`). Nothing is lost by dropping it.

**Three graphs carry real node labels**: `cora_ml` (7 topics), `email_eu_core` (42 departments), `lastfm_asia` (18 countries). The rest get placeholder zeros. `email_eu_core`'s departments are the only ground-truth _community_ labels in the suite, which makes it the natural graph for community-aware evaluation (MOEIM's community objective; our SBM experiments, which have no published baseline).

**First download of `cora_ml` takes minutes**: `cora_ml.npz` is 85 MB. That loader writes to a `.part` file and renames on completion, so an interrupted run retries cleanly. The other loaders skip when the target file exists, so a run killed mid-download can leave a truncated cache; delete `data/raw/<dataset>/` and re-run.

**Synthetic families (5).** Generated in bulk via `--num-graphs`; `log1p(degree)` node features throughout.

| `--dataset` | Generator                   | Tunables                              | Literature status                            |
| ----------- | --------------------------- | ------------------------------------- | -------------------------------------------- |
| `er`        | `nx.gnp_random_graph`       | `--er-p` (0.05)                       | standard (GCOMB, ToupleGDD, DeepIM, S2V-DQN) |
| `ba`        | `nx.barabasi_albert_graph`  | `--ba-m` (3)                          | standard                                     |
| `ws`        | `nx.watts_strogatz_graph`   | `--ws-k` (6), `--ws-p` (0.1)          | standard                                     |
| `sbm`       | `nx.stochastic_block_model` | `--sbm-blocks/--sbm-p-in/--sbm-p-out` | Warning: **no published IM baseline found**        |
| `karate`    | `nx.karate_club_graph`      |                                       | Warning: used by SL-VAE, not by IM papers          |

DeepIM's "Synthetic" row (50,000 nodes / 250,000 edges, avg degree 10) is a **random graph of that density**: an ER/BA-class graph we can reproduce with `--dataset ba --syn-nodes 50000 --ba-m 5`. It is the only synthetic row in the literature with a published per-cell table (§5.1).

### 6.2 Full catalogue

yes = already implemented in `data/datasets/`.

#### Small graphs (< 10K nodes)

| Dataset                 | Nodes | Edges                  | Type       | Avg deg | Source                                                                      | Used by                            |
| ----------------------- | ----- | ---------------------- | ---------- | ------- | --------------------------------------------------------------------------- | ---------------------------------- |
| soc-dolphins            | 62    | 159                    | undirected | 5       | [NetRepo](https://networkrepository.com/soc-dolphins.php)                   | ToupleGDD                          |
| Karate                  | 34    | 78                     | undirected | 4.6     | `nx.karate_club_graph()`                                                    | SL-VAE                             |
| **Jazz** yes             | 198   | 2,742                  | undirected | 27.7    | [nrvis](https://nrvis.com/download/data/misc/arenas-jazz.zip)               | DeepIM, MOEIM, SL-VAE              |
| soc-wiki-Vote (NetRepo) | 889   | 2,900                  | directed   | 6       | [NetRepo](https://networkrepository.com/soc-wiki-Vote.php)                  | ToupleGDD ("Wiki-1")               |
| **email-Eu-core** yes    | 1,005 | 25,571 (24,929 usable) | directed   | 51      | [SNAP](https://snap.stanford.edu/data/email-Eu-core.html)                   | MOEIM                              |
| **NetScience** yes       | 1,589 | 2,742                  | undirected | 3.45    | [Netzschleuder](https://networks.skewed.de/net/netscience)                  | DeepIM, SL-VAE                     |
| **Cora-ML** yes          | 2,810 | 7,981                  | undirected | 5.7     | [graph2gauss](https://github.com/abojchevski/graph2gauss) + `standardize()` | DeepIM, MOEIM, SL-VAE              |
| **ego-Facebook** yes     | 4,039 | 88,234                 | undirected | 43.7    | [SNAP](https://snap.stanford.edu/data/ego-Facebook.html)                    | MOEIM, LeNSE                       |
| **Power Grid** yes       | 4,941 | 6,594                  | undirected | 2.67    | [nrvis](https://nrvis.com/download/data/misc/opsahl-powergrid.zip)          | DeepIM, MOEIM, SL-VAE              |
| **ca-GrQc** yes          | 5,242 | 14,496 (14,484 usable) | undirected | 5.5     | [SNAP](https://snap.stanford.edu/data/ca-GrQc.html)                         | ToupleGDD                          |
| p2p-Gnutella08          | 6,301 | 20,777                 | directed   | 6.6     | [SNAP](https://snap.stanford.edu/data/p2p-Gnutella08.html)                  | MOEIM                              |
| **wiki-Vote (SNAP)** yes | 7,115 | 103,689                | directed   | 29.1    | [SNAP](https://snap.stanford.edu/data/wiki-Vote.html)                       | ToupleGDD ("Wiki-2"), MOEIM, LeNSE |
| **LastFM Asia** yes      | 7,624 | 27,806                 | undirected | 7.3     | [SNAP](https://snap.stanford.edu/data/feather-lastfm-social.html)           | MOEIM                              |
| ca-HepTh                | 9,877 | 25,998                 | undirected | 5.3     | [SNAP](https://snap.stanford.edu/data/ca-HepTh.html)                        | MOEIM                              |

#### Classical mid-size IM benchmarks (10K: 500K)

| Dataset                | Nodes   | Edges                                                   | Type       | Avg deg | Source                                                                                                   | Used by                                                              |
| ---------------------- | ------- | ------------------------------------------------------- | ---------- | ------- | -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| **NetHEPT** yes         | 15,233  | 31,376 undirected (= 62,752 arcs)                       | undirected | 4.18    | [SparklyYS mirror](https://github.com/SparklyYS/Simultaneous-IMM)                                        | Kempe, CELF, DegreeDiscount, PMIA, IRIE, TIM, IMM, SSA, OPIM, SubSIM |
| ca-CondMat             | 23,133  | 93,497                                                  | undirected | 8.1     | [SNAP](https://snap.stanford.edu/data/ca-CondMat.html)                                                   | misc                                                                 |
| Flixster [claim]       | 29,357  | 425,228 arcs                                            | directed   | 14.5    | topic-aware IM literature (Barbieri/Goyal)                                                               | TIM+, topic-aware IM                                                 |
| Enron                  | 36,692  | 183,831                                                 | undirected | 10.0    | [SNAP](https://snap.stanford.edu/data/email-Enron.html)                                                  | SSA, D-SSA, IMM, GLIE                                                |
| **NetPHY** yes          | 37,154  | 174,161 (231,584 raw lines; SSA quotes 180,826 ordered) | undirected | 9.38    | [Wei Chen `weic-graphdata.zip`](https://www.microsoft.com/en-us/research/people/weic/selected-projects/) | PMIA, IRIE, TIM, IMM, SSA                                            |
| Brightkite             | 58,228  | 214,078                                                 | undirected | 7.4     | [SNAP](https://snap.stanford.edu/data/loc-Brightkite.html)                                               | GCOMB                                                                |
| Deezer HR              | 54,573  | 498,202                                                 | undirected | 18.3    | [SNAP gemsec-Deezer](https://snap.stanford.edu/data/gemsec-Deezer.html)                                  | LeNSE                                                                |
| soc-Epinions1          | 75,879  | 508,837                                                 | directed   | 13.4    | [SNAP](https://snap.stanford.edu/data/soc-Epinions1.html)                                                | ToupleGDD                                                            |
| **Twitter (ego)** yes   | 81,306  | 1,768,149 arcs                                          | directed   | 43.5    | [SNAP](https://snap.stanford.edu/data/ego-Twitter.html)                                                  | GCOMB ("TW-ew"), LeNSE                                               |
| Slashdot0902           | 82,168  | 948,464                                                 | directed   | 23.1    | [SNAP](https://snap.stanford.edu/data/soc-Slashdot0902.html)                                             | IRIE, LeNSE                                                          |
| Buzznet                | 101,200 | ≈2,800,000                                              | undirected | 54      | [NetRepo](https://networkrepository.com/soc-buzznet.php)                                                 | ToupleGDD                                                            |
| **Digg (Syracuse)** yes | 116,893 | ≈2.6M                                                   | undirected | ≈45     | [Syracuse](https://datasets.syr.edu/datasets/Digg.html)                                                  | (ours only)                                                          |
| Epinions (signed)      | 131,828 | 841,372                                                 | directed   | 13.4    | [SNAP](https://snap.stanford.edu/data/soc-sign-epinions.html)                                            | IMM, SSA, D-SSA, TIM, IRIE                                           |
| Gowalla                | 196,591 | 950,327                                                 | undirected | 9.7     | [SNAP](https://snap.stanford.edu/data/loc-Gowalla.html)                                                  | GCOMB                                                                |
| Digg 2009 (ISI)        | 279,631 | 2,251,166                                               | directed   | 16.1    | [ISI/Lerman](https://www.isi.edu/~lerman/downloads/digg2009.html)                                        | IMINFECTOR, DeepIM                                                   |
| com-DBLP               | 317,080 | 1,049,866                                               | undirected | 6.6     | [SNAP](https://snap.stanford.edu/data/com-DBLP.html)                                                     | LeNSE, GCOMB                                                         |
| com-Amazon             | 334,863 | 925,872                                                 | undirected | 5.5     | [SNAP](https://snap.stanford.edu/data/com-Amazon.html)                                                   | IRIE (different Amazon), misc                                        |
| Higgs Twitter          | 456,631 | 14,855,875                                              | directed   | 65.1    | [SNAP](https://snap.stanford.edu/data/higgs-twitter.html)                                                | cascade/IM literature                                                |

#### Large scale (> 500K nodes)

| Dataset                 | Nodes      | Edges                                   | Type       | Avg deg | Source                                                                                                                                                                                                                                                                         | Used by                                                              |
| ----------------------- | ---------- | --------------------------------------- | ---------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------- |
| DBLP (classical)        | 655,000    | 1,990,000                               | undirected | 6.08    | Wei Chen / arnetminer release                                                                                                                                                                                                                                                  | IMM, SSA, TIM, PMIA, IRIE                                            |
| **YouTube** yes          | 1,134,890  | 2,987,624                               | undirected | 5.3     | [SNAP](https://snap.stanford.edu/data/com-Youtube.html)                                                                                                                                                                                                                        | ToupleGDD, GCOMB, LeNSE, GLIE                                        |
| Sina Weibo (IMINFECTOR) | 1,170,689  | 225,877,808                             | directed   | 386     | AMiner via IMINFECTOR                                                                                                                                                                                                                                                          | IMINFECTOR, DeepIM                                                   |
| MAG (CS)                | 1,436,158  | 15,928,078                              | directed   | 22.2    | Microsoft Academic Graph                                                                                                                                                                                                                                                       | IMINFECTOR                                                           |
| soc-Pokec               | 1,632,803  | 30,622,564                              | directed   | 37.5    | [SNAP](https://snap.stanford.edu/data/soc-Pokec.html)                                                                                                                                                                                                                          | misc                                                                 |
| as-Skitter              | 1,696,415  | 11,095,298                              | undirected | 13.1    | [SNAP](https://snap.stanford.edu/data/as-Skitter.html)                                                                                                                                                                                                                         | LeNSE                                                                |
| **Weibo (raw)** yes      | 1,787,443  | ≈216M arcs                              | directed   | ≈242    | [AMiner](https://www.aminer.cn/influencelocality)                                                                                                                                                                                                                              | (ours only)                                                          |
| Weibo (DeepIM)          | 2,251,166  | 225,877,808                             | directed   | 200     |: see §6.4.5 (transcription error)                                                                                                                                                                                                                                             | DeepIM                                                               |
| wiki-Talk               | 2,394,385  | 5,021,410                               | directed   | 4.2     | [SNAP](https://snap.stanford.edu/data/wiki-Talk.html)                                                                                                                                                                                                                          | LeNSE ("Talk")                                                       |
| sx-stackoverflow        | 2,601,977  | 63,497,050 temporal / 36,233,450 static | directed   | 27.8    | [SNAP](https://snap.stanford.edu/data/sx-stackoverflow.html)                                                                                                                                                                                                                   | GCOMB ("Stack", quotes 2.69M/5.9M after their action-log derivation) |
| com-Orkut               | 3,072,441  | 117,185,083                             | undirected | 76.2    | [SNAP](https://snap.stanford.edu/data/com-Orkut.html)                                                                                                                                                                                                                          | IMM, SSA, GCOMB                                                      |
| cit-Patents             | 3,774,768  | 16,518,948                              | directed   | 8.8     | [SNAP](https://snap.stanford.edu/data/cit-Patents.html)                                                                                                                                                                                                                        | adjacent literature                                                  |
| com-LiveJournal         | 3,997,962  | 34,681,189                              | undirected | 17.3    | [SNAP](https://snap.stanford.edu/data/com-LiveJournal.html)                                                                                                                                                                                                                    | misc                                                                 |
| soc-LiveJournal1        | 4,847,571  | 68,993,773                              | directed   | 28.5    | [SNAP](https://snap.stanford.edu/data/soc-LiveJournal1.html)                                                                                                                                                                                                                   | IMM, SSA, TIM, IRIE                                                  |
| Twitter (Kwak'10)       | ≈41.7M     | ≈1.47G                                  | directed   | 70.5    | [KAIST](https://anlab-kaist.github.io/traces/WWW2010) · [LAW mirror](https://law.di.unimi.it/webdata/twitter-2010/), Warning: the LAW host timed out from our network on 2026-07-29; [Wayback copy](https://web.archive.org/web/2024/https://law.di.unimi.it/webdata/twitter-2010/) | IMM, SSA, TIM, GCOMB                                                 |
| com-Friendster          | 65,608,366 | 1,806,067,135                           | undirected | 55.1    | [SNAP](https://snap.stanford.edu/data/com-Friendster.html)                                                                                                                                                                                                                     | SSA, GCOMB                                                           |

#### Cascade datasets: a _different_ evaluation protocol

These carry observed diffusion traces, not just topology. They enable the IMINFECTOR-style **DNI** protocol, which never assumes IC/LT. They are the shared substrate for [`cascade_prediction.md`](cascade_prediction.md) and [`cascade_reconstruction.md`](cascade_reconstruction.md).

| Dataset         | Nodes                      | Edges        | Cascades              | Avg cascade | Source                                                                                                     |
| --------------- | -------------------------- | ------------ | --------------------- | ----------- | ---------------------------------------------------------------------------------------------------------- |
| **Digg 2009**   | 279,631                    | 2,251,166    | 3,553                 | 847         | [ISI/Lerman](https://www.isi.edu/~lerman/downloads/digg2009.html)                                          |
| **Sina Weibo**  | 1,170,689                  | 225,877,808  | 115,686               | 148         | [AMiner](https://www.aminer.cn/influencelocality) via [IMINFECTOR](https://github.com/geopanag/IMINFECTOR) |
| **MAG (CS)**    | 1,436,158                  | 15,928,078   | 181,020               | 29          | Microsoft Academic Graph, CS subset, via IMINFECTOR                                                        |
| **Memetracker** | 12,529 (as used by SL-VAE) |              | 96M phrases           |             | [SNAP MemeTracker](https://snap.stanford.edu/data/memetracker9.html)                                       |
| **Flixster**    | 29,357                     | 425,228 arcs | action log, 10 topics |             | topic-aware IM literature (Barbieri/Goyal)                                                                 |

All five counts above are **[verified]** from IMINFECTOR's Table 3 by direct text extraction, except Memetracker/Flixster which are **[claim]**.

### 6.3 Warning: Name collisions, read this before comparing any number

**Seven dataset names in this literature denote more than one graph.** Quoting a published number against the wrong version is the single most common way to produce an invalid comparison table. Each row below is two _different graphs_ that share a name.

| Name            | Version A                                     | Version B                                          | Who uses which                                               |
| --------------- | --------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------ |
| **Epinions**    | `soc-Epinions1` **75,879 / 508,837** directed | `soc-sign-epinions` **131,828 / 841,372** directed | A: ToupleGDD · B: IMM, SSA, D-SSA, TIM, IRIE                 |
| **DBLP**        | `com-DBLP` **317,080 / 1,049,866** undirected | Chen/arnetminer DBLP **655K / 1.99M** undirected   | A: LeNSE, GCOMB · B: IMM, SSA, TIM, PMIA, IRIE               |
| **Twitter**     | SNAP `ego-Twitter` **81,306 / 1,768,149**     | Kwak 2010 crawl **41.7M / 1.47G**                  | A: ours, GCOMB (TW-ew), LeNSE · B: IMM, SSA, TIM, GCOMB (TW) |
| **Twitter**     | (above)                                       | NetRepo `soc-twitter-follows` **0.8K / 1K**        | ToupleGDD's "Twitter" is this third one                      |
| **Wiki-Vote**   | NetRepo `soc-wiki-Vote` **889 / 2,900**       | SNAP `wiki-Vote` **7,115 / 103,689**               | ToupleGDD calls A "Wiki-1" and B "Wiki-2"                    |
| **Digg**        | Syracuse friendship **116,893 / ≈2.6M**       | ISI/Lerman 2009 **279,631 / 2,251,166** + cascades | A: ours · B: IMINFECTOR, DeepIM                              |
| **NetScience**  | Newman 2006 **1,589 / 2,742**                 | DeepIM's **1,565 / 13,532** (unidentified)         | A: ours, NetRepo · B: DeepIM (§6.4.6, unresolved)           |
| **Weibo**       | AMiner raw network **1,787,443 / ≈216M**      | IMINFECTOR-derived **1,170,689 / 225,877,808**     | A: ours · B: IMINFECTOR, DeepIM                              |
| **LiveJournal** | `com-LiveJournal` **3,997,962 / 34,681,189**  | `soc-LiveJournal1` **4,847,571 / 68,993,773**      | B is the one IMM/SSA report                                  |

Additionally, several papers silently report the **largest connected component** rather than the raw file. This is not a different graph, just a different preprocessing switch, but it changes the numbers:

| Graph             | Raw file        | LCC as reported                    | Reported by                    |
| ----------------- | --------------- | ---------------------------------- | ------------------------------ |
| Cora-ML           | 2,995 / 8,416   | **2,810 / 7,981**                  | DeepIM, MOEIM                  |
| email-Eu-core     | 1,005 / 25,571  | **986 / 25,552**                   | MOEIM                          |
| wiki-Vote         | 7,115 / 103,689 | **7,066 / 103,663**                | MOEIM                          |
| CA-HepTh          | 9,877 / 25,998  | **8,638 / 24,827**                 | MOEIM                          |
| CA-GrQc           | 5,242 / 14,496  | **4,158 / 13,422**                 | ToupleGDD ("caGr", 4.2k/13.4k) |
| p2p-Gnutella08    | 6,301 / 20,777  | **6,299 / 20,776**                 | MOEIM ("gnutella")             |
| facebook-combined | 4,039 / 88,234  | 4,039 / 88,234 (already connected) | MOEIM                          |

### 6.4 Version forensics

Each discrepancy between our version of a graph and the literature's, traced to its cause.

| Dataset        | Cause                                                        | Status                       |
| -------------- | ------------------------------------------------------------ | ---------------------------- |
| **NetHEPT**    | Edge-count _convention_, not a different graph               | yes **resolved: same graph** |
| **Cora-ML**    | Largest-connected-component extraction; same source file     | yes **applied in the loader** |
| **Digg**       | Different source repository (Syracuse vs ISI/Lerman)         | no different graph           |
| **Twitter**    | Different repository _and_ different graph (SNAP vs NetRepo) | no unrelated graph           |
| **Weibo**      | Same source; DeepIM's table has a **transcription error**    | Warning: see §6.4.5                |
| **NetScience** | Their numbers match neither the cited source nor its LCC     | **unresolved**            |

#### 6.4.1 NetHEPT: resolved, it is the same graph

An earlier version of this review flagged our NetHEPT (15,229 / 62,752) as differing from the literature's "15,233 / 58,891". It does not. We now have the **primary source**: Wei Chen's own `weic-graphdata.zip` (the archive `netphy.py` downloads) ships `hep.txt`, whose header line reads `15233 58891` [verified].

Deduplicating that file ourselves [derived]:

```
hep.txt header               : 15,233 nodes, 58,891 edge LINES
unique undirected pairs      : 31,398   (including 39 self-loops)
minus self-loops             : 31,359 edges  -> 62,718 arcs
our SparklyYS mirror         : 31,376 edges  -> 62,752 arcs, 15,229 nodes
```

So the three published figures are three counts of one file: **58,891 = raw lines** (a pair appears once per co-authored paper), **31.4K = deduplicated undirected edges** (which is what the SSA/D-SSA table reports [verified, `p913-Huang.pdf` Table 2]) and our 62,752 is that same edge set stored as both arcs. Our mirror differs from Chen's original by **17 edges and 4 nodes (0.05%)**.

**NetHEPT is therefore directly comparable**, and the classical k=1…50 tables in §5.3 apply to our graph without a caveat.

Corollary: NetHEPT is a **co-authorship** network, not a citation network. Our loader's docstring said "edge (a, b) means paper a cites paper b", that was wrong and is now fixed. The adjacency was always symmetric, so nothing about the data or the simulation changes.

#### 6.4.2 Cora-ML: solved, and applied

Both we and DeepIM use **the same file**: `cora_ml.npz` from [graph2gauss](https://github.com/abojchevski/graph2gauss). DeepIM loads it through that project's `SparseGraph` class and calls `standardize()`:

```python
def standardize(self, make_unweighted=True, make_undirected=True,
                no_self_loops=True, select_lcc=True) -> "SparseGraph":
```

`select_lcc=True` is the **default**. Verified empirically on our own file [derived]:

```
raw adjacency:                 2,995 nodes,  8,416 directed arcs   <- our version
as undirected simple graph:    2,995 nodes,  8,158 edges
connected components:          61, largest = 2,810 nodes
largest connected component:   2,810 nodes,  7,981 edges           <- DeepIM's Table 1, exactly
```

`data/datasets/cora_ml.py` now applies symmetrize → drop self-loops → keep LCC, reproducing their graph bit-for-bit. This converted Cora-ML from a caveated row into a directly comparable one.

#### 6.4.3 Digg: different source repository

|                         | Source                                                                          | Nodes / Edges       |
| ----------------------- | ------------------------------------------------------------------------------- | ------------------- |
| **Ours**                | [Syracuse](https://datasets.syr.edu/datasets/Digg.html) `Digg-dataset.zip`      | 116,893 / ≈2.6M     |
| **IMINFECTOR & DeepIM** | [ISI / Lerman "Digg 2009"](https://www.isi.edu/~lerman/downloads/digg2009.html) | 279,631 / 2,251,166 |

The [IMINFECTOR README](https://github.com/geopanag/IMINFECTOR) names the ISI URL as its Digg source; DeepIM cites IMINFECTOR for its Digg graph. The ISI version also carries **diffusion cascades**, which is why IMINFECTOR could use it, ours is a friendship graph only.

#### 6.4.4 Twitter: different repository, unrelated graph

|                 | Source                                                              | Nodes / Edges      |
| --------------- | ------------------------------------------------------------------- | ------------------ |
| **Ours**        | [SNAP ego-Twitter](https://snap.stanford.edu/data/ego-Twitter.html) | 81,306 / 1,768,149 |
| **ToupleGDD**   | [Network Repository](https://networkrepository.com)                 | 0.8K / 1K          |
| **IMM/SSA/TIM** | [Kwak et al. 2010](https://anlab-kaist.github.io/traces/WWW2010)    | 41.7M / 1.47G      |

ToupleGDD §VI-A: _"Twitter, Wiki-1, caGr and Buzznet are from [37], while Wiki-2, Epinions and Youtube are available on [38]"_, where **[37] = Rossi & Ahmed, Network Repository** and **[38] = Leskovec & Krevl, SNAP**. This same reference split **explains the YouTube match**: their YouTube comes from SNAP, exactly the `com-Youtube` file our loader downloads.

#### 6.4.5 Weibo: DeepIM's table contains a transcription error

IMINFECTOR's own Table 3 reads [verified, direct text extraction]:

|       | Digg          | MAG        | Sina Weibo    |
| ----- | ------------- | ---------- | ------------- |
| Nodes | 279,631       | 1,436,158  | **1,170,689** |
| Edges | **2,251,166** | 15,928,078 | 225,877,808   |

DeepIM's Table 1 (which cites IMINFECTOR for both graphs) reads Digg **279,613 / 1,170,689** and Weibo **2,251,166 / 225,877,808**.

Those are **the same four numbers reassigned across cells**: Digg's edge count (2,251,166) became Weibo's node count, and Weibo's node count (1,170,689) became Digg's edge count. (279,631 → 279,613 is an additional digit transposition.)

**Consequence:** DeepIM's Digg and Weibo _column headers_ are wrong, but the graphs it ran on are IMINFECTOR's. Our Weibo (1,787,443 nodes, read straight from `weibo_network.txt`) is still a different construction from IMINFECTOR's 1,170,689, but the gap is much smaller than DeepIM's table implies, and an earlier "union of follower network and cascade users" explanation is not supported by the numbers.

#### 6.4.6 NetScience: unresolved, evidence is contradictory

**Our graph is provably the one DeepIM cites.** DeepIM attributes Network Science to Rossi & Ahmed (Network Repository). Our loader pulls from [Netzschleuder](https://networks.skewed.de/net/netscience), and its statistics match [networkrepository.com/netscience.php](https://networkrepository.com/netscience.php) on four independent measures:

| Statistic      | Network Repository | Ours (computed) |
| -------------- | ------------------ | --------------- |
| Nodes / Edges  | 1.6K / 2.7K        | 1,589 / 2,742   |
| Max degree     | 34                 | 34              |
| Density        | 0.00217332         | 0.00217         |
| Avg clustering | 0.637791           | 0.638           |

Both are M. Newman's 2006 co-authorship network. But DeepIM reports **1,565 / 13,532**: 4.9× the edges. Two hypotheses were tested and **both fail**:

- _Not LCC extraction._ The LCC of this graph is **379 nodes**, not 1,565. (`ca-netscience` on Network Repository is exactly that 379/914 component.)
- _Not a typo._ Their Table 1 avg-degree column reads 17.28, and 2 × 13,532 / 1,565 = 17.29: internally consistent, so it describes a real graph they actually ran on.

**Conclusion: DeepIM's "Network Science" is a denser graph than the Newman network it cites, and the paper does not say how it was produced.** Resolving it needs their `netscience_25c.SG` pickle, which is not in the public repo (their `data/` folder ships empty).

**Narrowed by the source-localization literature.** The same anomalous graph appears in exactly one other place: SL-VAE (KDD 2022) also reports Network Science as **1,565 / 13,532**, with avg clustering 0.741 [verified]. Every other paper in that literature (IVGD (WWW 2022), GraphSL, SIDSL (2025)) reports **1,589 / 2,742** with clustering 0.638, which is ours. See [`source_localization.md`](source_localization.md) §6.

So the anomaly is not a DeepIM idiosyncrasy but a shared preprocessing lineage between DeepIM and SL-VAE, and it is the **minority** version within its own literature. Our graph is the one three of the four SL papers use. That does not recover DeepIM's column, but it does mean our NetScience row is comparable to most of the field: the caveat attaches to DeepIM and SL-VAE specifically, not to us.

### 6.5 What to add: ranked recommendation

Ordered by _research value per hour of implementation_.

#### Tier 1: yes DONE (small, canonical, published numbers exist)

| `--dataset`     | Nodes  | Edges   | Why it earned a slot                                                                                                               |
| --------------- | ------ | ------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `netphy`        | 37,154 | 174,161 | The other half of the classical Chen/IRIE/PMIA/TIM/SSA canon. NetHEPT's sibling; every classical paper reports both.               |
| `wiki_vote`     | 7,115  | 103,689 | Dense (avg total degree 29) and directed. ToupleGDD (as "Wiki-2") and MOEIM both report it. Our suite had no dense directed graph. |
| `email_eu_core` | 1,005  | 24,929  | MOEIM setting-1. Tiny, directed, **42 ground-truth departments** = the only real community labels in the suite.                    |
| `facebook`      | 4,039  | 88,234  | MOEIM setting-1. Strong community structure, already connected, no preprocessing decisions to get wrong.                           |
| `ca_grqc`       | 5,242  | 14,484  | ToupleGDD's "caGr". Sparse collaboration graph: the regime where degree heuristics are weakest.                                   |
| `lastfm_asia`   | 7,624  | 27,806  | MOEIM setting-1. Ships **18 country labels**; matches MOEIM's count with no preprocessing.                                         |
| `cora_ml`       | 2,810  | 7,981   | DeepIM's and MOEIM's Cora-ML, reproduced exactly via graph2gauss `standardize()`.                                                  |

This gives **direct number-for-number comparability with MOEIM's setting-1 table** (four of its six graphs) plus a comparable Cora-ML row against DeepIM's Table 2/3. Before this we could compare against **none** of setting-1.

Still missing from MOEIM setting-1: `p2p-Gnutella08` (6,301 / 20,777) and `ca-HepTh` (9,877 / 25,998). Both are SNAP `.txt.gz` files in exactly the format `ca_grqc.py` already parses, each is a copy-and-edit of that loader.

#### Tier 2: add when scalable simulation lands

| Dataset               | Nodes   | Edges      | Why                                                                                                    |
| --------------------- | ------- | ---------- | ------------------------------------------------------------------------------------------------------ |
| **Epinions (signed)** | 131,828 | 841,372    | The Epinions that IMM/SSA/TIM/IRIE actually report. IRIE publishes k=50 WC spread = 12,063 [verified]. |
| **Slashdot0902**      | 82,168  | 948,464    | LeNSE IM benchmark; classical directed social graph.                                                   |
| **Enron**             | 36,692  | 183,831    | In the SSA/D-SSA table; small enough to be near-Tier-1.                                                |
| **Brightkite**        | 58,228  | 214,078    | GCOMB IM/MCP benchmark.                                                                                |
| **Gowalla**           | 196,591 | 950,327    | GCOMB IM/MCP benchmark.                                                                                |
| **com-DBLP**          | 317,080 | 1,049,866  | LeNSE + GCOMB. Note this is _not_ the classical DBLP (§6.3).                                           |
| **Deezer HR**         | 54,573  | 498,202    | LeNSE IM benchmark.                                                                                    |
| **Buzznet**           | 101,200 | ≈2,800,000 | ToupleGDD; extremely high avg degree (54): a saturation stress test.                                  |

#### Tier 3: only if a scalability claim is being made

`com-Orkut` (3.07M/117M), `soc-LiveJournal1` (4.85M/69.0M), `as-Skitter` (1.70M/11.1M), `wiki-Talk` (2.39M/5.02M), `sx-stackoverflow` (2.60M/63.5M), `soc-Pokec` (1.63M/30.6M), Kwak Twitter (41.7M/1.47G), `com-Friendster` (65.6M/1.81G). We already have four graphs in this class (`twitter`, `digg`, `youtube`, `weibo`) that we cannot simulate: adding more before the simulator scales buys nothing.

#### Explicitly _not_ recommended

- **More synthetic families.** We already have five and `sbm`/`karate` have no published IM baseline. Adding a sixth generator adds no comparability.
- **cit-Patents, com-Amazon, com-Friendster.** Present in adjacent literature (max-cover, vertex-cover) but not in any IM table we baseline against.
- **NetScience "DeepIM version".** Cannot be added: it does not exist publicly (§6.4.6).

### 6.6 How to add a dataset

The contract is two files and one line. `--dataset` takes a free string; the registry that validates it is `real_directed` in `data/wm_graphs.py`.

1. **Write `data/datasets/<name>.py`** exposing exactly two functions:

   ```python
   def download_<name>() -> Path: ...
   def load_<name>(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
       # returns (adjacency (N,N), node_feats (N,F), node_labels (N,), num_nodes)
   ```

   Download into `data/raw/<name>/`, skip if present, print `[ok]`/`[get]` like the existing loaders. `data/graph_utils.py` carries the four steps every loader needs, so the body is usually ~6 lines:

   ```python
   raw_edges = np.loadtxt(path, comments="#", dtype=np.int64)   # shape: (E, 2)
   remapped, num_nodes = remap_to_contiguous(raw_edges)         # sparse SNAP ids
   adjacency = edges_to_adjacency(remapped[:, 0], remapped[:, 1],
                                  num_nodes, directed=False)    # dedups multi-edges
   node_feats = degree_features(adjacency, directed=False)      # shape: (N, 1)
   ```

   plus `largest_connected_component(adjacency)` when reproducing a paper that reports the LCC (it returns the surviving indices so features and labels can be subset identically). `ca_grqc.py` is the shortest complete example; `lastfm_asia.py` shows the real-labels variant.

2. **Add one line to `data/wm_graphs.py::real_directed`**: `"<name>": True|False`. This is the whole registration, `make_real_bundle` imports the module lazily by name and calls the two functions by convention.

3. **Add the row** to §6.1 above, to `README.md`, and to `CLAUDE.md`.

Nothing else needs to change: `pipeline/run.py`, the simulator, the feature builder, and every baseline adapter all consume `GraphBundle`/`GraphInfo` and are dataset-agnostic.

**Preprocessing decisions to make explicitly** (they are why §6.3 exists): keep or drop the LCC, symmetrize or keep directed, drop self-loops, collapse multi-edges. Record the choice in the loader docstring, because it _is_ the difference between two "same-name" graphs.

---

## 7. Which paper uses which

Cells mark the dataset **as that paper reports it**: check §6.3 before assuming two papers with the same cell used the same graph.

| Dataset              | Kempe'03 | PMIA'10 | IRIE'12 | TIM'14 | IMM'15 | SSA'16 | GCOMB'20 | IMINFECTOR'20 | LeNSE'22 | GLIE'23 | ToupleGDD'23 | DeepIM'23 | MOEIM'24 |
| -------------------- | -------- | ------- | ------- | ------ | ------ | ------ | -------- | ------------- | -------- | ------- | ------------ | --------- | -------- |
| NetHEPT              | yes        | yes       | yes       | yes      | yes      | yes      |          |               |          |         |              |           |          |
| NetPHY               |          | yes       | yes       | yes      | yes      | yes      |          |               |          |         |              |           |          |
| Enron                |          |         |         | yes      | yes      | yes      |          |               |          | yes       |              |           |          |
| Epinions             |          | yes       | yes       | yes      | yes      | yes      |          |               |          |         | yes            |           |          |
| Slashdot             |          |         | yes       |        |        |        |          |               | yes        |         |              |           |          |
| Amazon               |          |         | yes       |        |        |        |          |               |          |         |              |           |          |
| DBLP                 |          | yes       | yes       | yes      | yes      | yes      |          |               | yes        |         |              |           |          |
| LiveJournal          |          |         | yes       | yes      | yes      | yes      |          |               |          |         |              |           |          |
| Orkut                |          |         |         | yes      | yes      | yes      | yes        |               |          |         |              |           |          |
| Twitter              |          |         |         | yes      | yes      | yes      | yes        |               | yes        |         | yes            |           |          |
| Friendster           |          |         |         |        |        | yes      | yes        |               |          |         |              |           |          |
| YouTube              |          |         |         |        |        |        | yes        |               | yes        | yes       | yes            |           |          |
| Brightkite           |          |         |         |        |        |        | yes        |               |          |         |              |           |          |
| Gowalla              |          |         |         |        |        |        | yes        |               |          |         |              |           |          |
| Stack                |          |         |         |        |        |        | yes        |               |          |         |              |           |          |
| Skitter              |          |         |         |        |        |        |          |               | yes        |         |              |           |          |
| wiki-Talk            |          |         |         |        |        |        |          |               | yes        |         |              |           |          |
| Deezer               |          |         |         |        |        |        |          |               | yes        |         |              |           |          |
| **Facebook** yes      |          |         |         |        |        |        |          |               | yes        | yes       |              |           | yes        |
| **wiki-Vote** yes     |          |         |         |        |        |        |          |               | yes        |         | yes            |           | yes        |
| **ca-GrQc** yes       |          |         |         |        |        |        |          |               |          | yes       | yes            |           |          |
| ca-HepTh             |          |         |         |        |        |        |          |               |          |         |              |           | yes        |
| **email-Eu-core** yes |          |         |         |        |        |        |          |               |          |         |              |           | yes        |
| **LastFM** yes        |          |         |         |        |        |        |          |               |          |         |              |           | yes        |
| Gnutella             |          |         |         |        |        |        |          |               |          |         |              |           | yes        |
| soc-dolphins         |          |         |         |        |        |        |          |               |          |         | yes            |           |          |
| Buzznet              |          |         |         |        |        |        |          |               |          |         | yes            |           |          |
| Crime, HI-II-14      |          |         |         |        |        |        |          |               |          | yes       |              |           |          |
| **Jazz** yes          |          |         |         |        |        |        |          |               |          |         |              | yes         | yes        |
| **Cora-ML** yes       |          |         |         |        |        |        |          |               |          |         |              | yes         | yes        |
| **NetScience** yes    |          |         |         |        |        |        |          |               |          |         |              | yes         |          |
| **Power Grid** yes    |          |         |         |        |        |        |          |               |          |         |              | yes         | yes        |
| Digg                 |          |         |         |        |        |        |          | yes             |          |         |              | yes         |          |
| Weibo                |          |         |         |        |        |        |          | yes             |          |         |              | yes         |          |
| MAG                  |          |         |         |        |        |        |          | yes             |          |         |              |           |          |
| Synthetic ER/BA/WS   | yes        |         |         |        |        |        | yes        |               |          | yes       | yes            | yes         |          |

**yes = we already load it.** Nine rows now intersect our suite; before the Tier-1 additions it was four.

GLIE additionally uses **Crime** (829 / 2,946) and **HI-II-14** (4,165 / 26,172), both small and both [verified] from its Table I.

---

## 8. Evaluation protocol

### Budget conventions: the comparability trap

The two literatures barely overlap in budget regime:

| Convention       | Range                    | Used by                                                                |
| ---------------- | ------------------------ | ---------------------------------------------------------------------- |
| **Absolute k**   | k ∈ {10, 20, 30, 40, 50} | Kempe, CELF, PMIA, IRIE, TIM, IMM, SSA, OPIM, SubSIM, ToupleGDD, GCOMB |
| **Percent of N** | 1%, 5%, 10%, 20%         | DeepIM, MOEIM, and our `--budget-pcts`                                 |

k=50 is **0.33% of NetHEPT**; 20% of NetHEPT is k=3,046. A paper reporting "NetHEPT k=50 → 725 nodes influenced" and a paper reporting "NetHEPT 20% → 51%" are not measuring comparable things.

**Recommendation:** run `--budgets 10 20 30 40 50` alongside `--budget-pcts 1 5 10 20` on any graph where we want to speak to both.

### Edge probabilities

Two settings dominate, and they are not interchangeable:

| Setting                   | Definition                                  | Used by                                                                  |
| ------------------------- | ------------------------------------------- | ------------------------------------------------------------------------ |
| **WC (weighted cascade)** | `p(u→v) = 1 / in-degree(v)`                 | DeepIM, MOEIM, IMM, ToupleGDD, and our `--prob-model weighted` (default) |
| **TR (trivalency)**       | `p` drawn uniformly from {0.1, 0.01, 0.001} | IRIE, PMIA, classical line                                               |
| **Uniform**               | `p = 0.1` or `p = 0.01`                     | ToupleGDD generalization tests, our `--prob-model uniform --uniform-p`   |

### LT thresholds

DeepIM uses `θ_v ~ U[0.3, 0.6]`; we use `θ_v ~ U(0,1)` (NDlib default). **This is a real difference**: a narrower threshold band makes LT cascades substantially easier to ignite, and it partly explains DeepIM's very high LT numbers (Jazz LT@20% = 99.1%).

### Metrics

| Metric                     | Definition                                                         | Who reports it          |
| -------------------------- | ------------------------------------------------------------------ | ----------------------- |
| **Expected spread** `σ(S)` | absolute node count at termination, averaged over MC runs          | classical line          |
| **% of nodes infected**    | `σ(S) / N`                                                         | DeepIM, MOEIM, and us   |
| **DNI**                    | distinct nodes appearing in held-out _real_ cascades seeded by `S` | IMINFECTOR (§5.5)       |
| **Hypervolume**            | over 6 objectives                                                  | MOEIM                   |
| **Runtime / #MC calls**    | wall-clock or simulation count to produce `S`                      | every scalability paper |

MC replication counts: DeepIM uses **100 rounds**; the classical NetHEPT protocol uses **10,000**. Our `--mc-marginals` default is 30 for target estimation, with a separate ground-truth replay for scoring.

---

## 9. Implications for this project

1. **Our comparable-baseline set is Jazz, Power Grid and Cora-ML.** Identical graphs, identical budget convention, published numbers in §5.1. Running our six conditions on those three graphs at 1/5/10/20% gives a table that drops straight into a paper alongside DeepIM's. Cora-ML joined that set when the loader started applying `standardize()`.

2. **NetScience remains uncomparable** (§6.4.6); DeepIM's version has 4.9× more edges and cannot be obtained. Report ours as self-contained.

3. **The saturation problem is confirmed by the literature, not unique to us.** Every method on Cora-ML IC at 20% lands in 50.2-52.4%. Our BA-100 six-way tie is the same phenomenon. **The fix is to report low budgets (1%, 5%) where published methods separate by 74% relative**, and to prefer LT where the gaps are enormous.

4. **NetHEPT is the discriminative benchmark.** Degree gets ≈250 where greedy/PMIA/IRIE get ≈725 at k=50. If our agent has to beat degree by 3×, that's a real test: unlike BA where degree is already near-optimal.

5. **Add absolute-k budgets for the classical comparison.** The classical literature lives at k ≤ 50; the learning literature at 1-20% of N. Running `--budgets 10 20 30 40 50` alongside `--budget-pcts 1 5 10 20` lets us speak to both.

6. **SBM has no published IM baseline.** Our SBM experiments can't be positioned against prior work: they are a novel setting (and community structure is exactly where MOEIM's community objective and DeepIM's weakness live). Worth framing as a contribution rather than a comparison.

7. **IMINFECTOR's finding is a warning for us.** A method tuned against a simulator underperformed on real observed cascades. Our world model is trained on NDlib-generated transitions: the same class of assumption. Worth stating as a limitation, and worth testing directly via [`cascade_prediction.md`](cascade_prediction.md).

8. **Pure IM exercises one of five action ops.** The strongest argument for the next task is that it puts the rest to work: see [`influence_blocking.md`](influence_blocking.md).

---

## 10. Reference list

**Classical** [Kempe 2003 (KDD)](https://www.cs.cornell.edu/home/kleinber/kdd03-inf.pdf) · [Leskovec 2007 CELF (KDD)](https://www.cs.cmu.edu/~jure/pubs/detect-kdd07.pdf) · [Chen 2009 DegreeDiscount (KDD)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/weic-kdd09_influence.pdf) · [Goyal 2011 CELF++ (WWW)](https://snap.stanford.edu/class/cs224w-readings/goyal11celf.pdf) · [Jung 2012 IRIE (arXiv 1111.4795)](https://arxiv.org/abs/1111.4795) · [Cheng 2013 StaticGreedy (arXiv 1212.4779)](https://arxiv.org/abs/1212.4779) · [Borgs 2014 RIS (arXiv 1212.0884)](https://arxiv.org/abs/1212.0884) · [Tang 2014 TIM (arXiv 1404.0900)](https://arxiv.org/abs/1404.0900) · [Tang 2015 IMM (SIGMOD)](https://dl.acm.org/doi/10.1145/2723372.2723734) · [Nguyen 2016 SSA (arXiv 1605.07990)](https://arxiv.org/abs/1605.07990) · [Tang 2018 OPIM (SIGMOD)](https://dl.acm.org/doi/10.1145/3183713.3183749) · [Guo 2020 SubSIM (SIGMOD)](https://dl.acm.org/doi/10.1145/3318464.3389740)

**Learning-based** [IMINFECTOR (arXiv 1904.08804)](https://arxiv.org/abs/1904.08804) · [code](https://github.com/geopanag/IMINFECTOR) · [GCOMB (NeurIPS 2020)](https://proceedings.neurips.cc/paper/2020/hash/e7532dbeff7ef901f2e70daacb3f452d-Abstract.html) · [code](https://github.com/idea-iitd/GCOMB) · [GLIE (arXiv 2108.04623)](https://arxiv.org/abs/2108.04623) · [code](https://github.com/geopanag/learn_im), the commonly-cited `geopanag/GLIE` 404s; this is the live repo · [LeNSE (ICML 2022)](https://proceedings.mlr.press/v162/ireland22a.html) · [code](https://github.com/davidireland3/LeNSE) · [ToupleGDD (arXiv 2210.07500)](https://arxiv.org/abs/2210.07500) · [code](https://github.com/Dtrycode/ToupleGDD) · [DeepIM (arXiv 2305.02200)](https://arxiv.org/abs/2305.02200) · [PMLR](https://proceedings.mlr.press/v202/ling23b/ling23b.pdf) · [code](https://github.com/triplej0079/DeepIM) · [MOEIM (arXiv 2403.18755)](https://arxiv.org/abs/2403.18755) · [code](https://github.com/eliacunegatti/MOEIM) · [HIM (arXiv 2502.13571)](https://arxiv.org/abs/2502.13571) · [REM multiplex (arXiv 2501.00779)](https://arxiv.org/abs/2501.00779) · [Topic-aware IM (DAMI 2025)](https://link.springer.com/article/10.1007/s10618-025-01133-3) · [DeepIM accelerated (Neural Networks 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0893608024005732)

**Surveys / benchmarks** [ML-based IM survey (arXiv 2211.03074, TKDD'23)](https://arxiv.org/abs/2211.03074) · [IM survey (arXiv 2309.04668)](https://arxiv.org/abs/2309.04668) · [Behaviour-aware IM survey (arXiv 2108.03438)](https://arxiv.org/abs/2108.03438) · [Deep-RL max-coverage benchmark (arXiv 2406.14697)](https://arxiv.org/abs/2406.14697)

**Data repositories** [SNAP](https://snap.stanford.edu/data/) · [Network Repository](https://networkrepository.com/) · [Netzschleuder](https://networks.skewed.de/) · [KONECT](http://konect.cc/networks/) · [AMiner](https://www.aminer.cn/data-sna) · [ISI/Lerman](https://www.isi.edu/~lerman/downloads/) · [Syracuse](https://datasets.syr.edu/) · [LAW (Twitter-2010 mirror)](https://law.di.unimi.it/webdata/twitter-2010/), Warning: the LAW host timed out from our network on 2026-07-29; [Wayback copy](https://web.archive.org/web/2024/https://law.di.unimi.it/webdata/twitter-2010/)

**Dataset tables transcribed for this file** [Revisiting SSA (VLDB'17) Table 2](http://www.vldb.org/pvldb/vol10/p913-Huang.pdf), NetHEPT/NetPHY/Enron/Epinions/DBLP/Orkut/LiveJournal/Twitter · [IMINFECTOR (TKDE'20) Table 3](https://arxiv.org/abs/1904.08804) (Digg/MAG/Weibo + cascades · [LeNSE (ICML'22) Table 2](https://arxiv.org/abs/2205.10106)) train/test edge splits (full graphs reconstructed by summation) · [GCOMB (NeurIPS'20) Table 1a](https://arxiv.org/abs/1903.03332) · [GLIE (arXiv 2108.04623) Table I](https://arxiv.org/abs/2108.04623) · [ToupleGDD (TCSS'23) Table I](https://arxiv.org/abs/2210.07500) · [DeepIM (ICML'23) Table 1](https://proceedings.mlr.press/v202/ling23b/ling23b.pdf) · [MOEIM (GECCO'24) Table 1](https://arxiv.org/abs/2403.18755) · [IRIE (ICDM'12) Table 3](https://arxiv.org/abs/1111.4795) · [PMIA (KDD'10)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/msr-tr-2010-2_v2.pdf), NetHEPT/NetPHY sizes

---

## 11. Open gaps

Honest list of what this review could **not** establish.

**Results**

- **MOEIM per-cell numbers**: published only as Pareto-front figures. Needs a code run to get a comparable table.
- **NetHEPT under the percentage-budget convention**: no learning-based paper reports NetHEPT at 1/5/10/20%. Our numbers there will have no direct precedent.
- **GCOMB, GLIE, LeNSE per-dataset _result_ tables**: identified and linked, but their result cells were not transcribed (they benchmark on YouTube, Stack, and billion-edge graphs largely disjoint from ours). Their _dataset_ tables **are** transcribed, in §6.2.
- **SBM and Karate**: no IM baselines found in any surveyed paper.

**Datasets**

- **DeepIM's Network Science graph** (1,565 / 13,532) matches neither the source it cites nor that source's LCC, and their `data/` folder ships empty (§6.4.6).
- ~~**NetPHY edge count**~~: **RESOLVED.** Chen's `phy.txt` deduplicates three ways, and all three published figures are accounted for [derived]: **231,584** raw lines (Chen et al. Table 1) → **180,826** unique _ordered_ pairs, which is SSA's "181K" and reproduces their avg-degree 9.73 exactly (2 × 180,826 / 37,154 = 9.73) → **174,161** unique _undirected_ pairs, which is what our loader builds.
- **Our Digg edge count** (≈2.6M) has never been pinned exactly: the loader prints it at load time but no number is recorded here.
- **Our Weibo edge count** (≈216M) likewise.
- **Memetracker and Flixster sizes** are [claim] only; no table was transcribed.
- **GCOMB's "Stack"** (2.69M / 5.9M) is a derived action-log graph, not raw `sx-stackoverflow` (2.60M / 63.5M static-36.2M). Their derivation is not reproduced here.
- **Twitter and Digg** version gaps are identified but unresolved: we hold different graphs from the literature and cannot obtain theirs cheaply (§6.4.3, §6.4.4).
