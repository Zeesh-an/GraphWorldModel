# Influence Maximization — Prior Work, Datasets, and Published Results

Unified reference for the IM literature relevant to this project: every method we
might baseline against, every paper that reports results on a dataset or synthetic
family we use, with published numbers, dataset metadata, budgets, links, and code.

---

## 0. How to read this file (verification policy)

Numbers in this document fall into three tiers, and every table says which it is:

| Tier           | Meaning                                                                                |
| -------------- | -------------------------------------------------------------------------------------- |
| **[verified]** | Transcribed by reading the paper's own table in the PDF. Trustworthy.                  |
| **[figure]**   | Read off a plotted figure — the paper published no table. Approximate, direction only. |
| **[claim]**    | The paper's prose claim ("we outperform X by Y%"), no per-cell numbers available.      |

This matters: automated PDF summarizers hallucinate plausible-looking numbers from
IM papers. While compiling this, a summarizer reported IRIE scoring `142.8` on
NetHEPT at k=50; the actual published value in the paper's Table 3 is **724.67**.
Every number below marked [verified] was read from the paper's own table.

---

## 1. Dataset cross-reference — what we use vs. what the literature uses

**This is the most important table in the file.** "Cora-ML" and "NetScience" do
not denote a single graph in this literature — different papers use different
versions with substantially different edge counts. Comparing our spread numbers
to a published table without matching the graph version is invalid.

| Dataset        | **Our version** (nodes / edges) | Literature version                | Match?                             | Who uses it                     |
| -------------- | ------------------------------- | --------------------------------- | ---------------------------------- | ------------------------------- |
| **Jazz**       | 198 / 2,742                     | 198 / 2,742                       | ✅ **exact**                       | DeepIM, MOEIM                   |
| **Power Grid** | 4,941 / 6,594                   | 4,941 / 6,594                     | ✅ **exact**                       | DeepIM, MOEIM                   |
| **Cora-ML**    | 2,995 / 8,416                   | 2,810 / 7,981                     | ⚠️ **differs** (+6.6% nodes)       | DeepIM, MOEIM                   |
| **NetScience** | 1,589 / 2,742                   | 1,565 / **13,532**                | ❌ **very different** (4.9× edges) | DeepIM                          |
| **NetHEPT**    | 15,229 / 62,752                 | 15,233 / 58,891 ("Arxiv"/NetHEPT) | ⚠️ close, +6.6% edges              | IRIE, PMIA, TIM, IMM, SSA, CELF |
| **YouTube**    | 1,134,890 / 2,987,624           | 1.13M / 3M                        | ✅ **matches**                     | ToupleGDD                       |
| **Digg**       | 116,893 / ≈2.6M                 | 279,613 / 1,170,689               | ❌ **different graph**             | DeepIM, IMINFECTOR              |
| **Weibo**      | 1,787,443 / ≈216M               | 2,251,166 / 225,877,808           | ⚠️ differs (−21% nodes)            | DeepIM, IMINFECTOR              |
| **Twitter**    | 81,306 / ≈1.3M                  | 0.8k / 1k (ToupleGDD's "Twitter") | ❌ **unrelated graph**             | —                               |

**Practical consequence.** **Jazz** and **Power Grid** support a direct
number-for-number comparison against the published DeepIM/MOEIM tables today, and
**Cora-ML** joins them the moment we add largest-connected-component extraction
(§1.1 — their graph is provably ours after `standardize()`). NetHEPT is close
enough for trend comparison with a footnote. NetScience, Digg, and Twitter are
_not_ comparable — either adopt the literature's version of the graph, or report
ours as a self-contained result and say so.

Synthetic families in the literature: **ER**, **BA**, **WS** are all standard
(GCOMB, ToupleGDD, DeepIM, S2V-DQN). **SBM** and **Karate** are, as far as this
review found, **not** used as IM benchmarks in the major learning-based papers —
our SBM experiments have no published baseline to compare against.

### 1.1 Why the numbers differ — root cause per dataset

Each discrepancy was traced to its source. Three distinct causes, and they have
very different implications: one is a trivial preprocessing switch we can flip,
two are genuinely different graphs that happen to share a name.

| Dataset        | Cause                                                                     | Status                       |
| -------------- | ------------------------------------------------------------------------- | ---------------------------- |
| **Cora-ML**    | **Largest-connected-component extraction.** Same source file.             | ✅ **solved — reproducible** |
| **Digg**       | **Different source repository** (Syracuse vs ISI/Lerman)                  | ❌ different graph           |
| **Twitter**    | **Different repository AND different graph** (SNAP vs Network Repository) | ❌ unrelated graph           |
| **Weibo**      | Same source (AMiner), different graph construction                        | ⚠️ derived vs raw            |
| **NetHEPT**    | Different mirrors / dedup of Wei Chen's file                              | ⚠️ minor                     |
| **NetScience** | Same cited source, but their numbers match neither the source nor its LCC | ❓ **unresolved**            |

#### Cora-ML — solved, and we can match them exactly

Both we and DeepIM use **the same file**: `cora_ml.npz` from
[Bojchevski & Günnemann's graph2gauss](https://github.com/abojchevski/graph2gauss)
(our `data/datasets/cora_ml.py` downloads it directly). DeepIM loads it through
that project's `SparseGraph` class —
[`data/sparsegraph.py`](https://github.com/triplej0079/DeepIM/blob/main/data/sparsegraph.py)
in their repo — and calls `standardize()`, whose signature is:

```python
def standardize(self, make_unweighted=True, make_undirected=True,
                no_self_loops=True, select_lcc=True) -> "SparseGraph":
```

`select_lcc=True` is the **default**. Verified empirically on our own downloaded file:

```
raw adjacency:                 2,995 nodes,  8,416 directed arcs   <- our version
as undirected simple graph:    2,995 nodes,  8,158 edges
connected components:          61, largest = 2,810 nodes
largest connected component:   2,810 nodes,  7,981 edges           <- DeepIM's Table 1, exactly
```

DeepIM's "2,810 / 7,981" is our graph with `standardize()` applied. **Applying
symmetrize → drop self-loops → keep LCC to our loader reproduces their graph
bit-for-bit**, which makes Cora-ML a directly comparable dataset instead of a
caveated one.

#### Digg — different source repository

Two unrelated collections are both called "Digg":

|                         | Source                                                                                       | Nodes / Edges       |
| ----------------------- | -------------------------------------------------------------------------------------------- | ------------------- |
| **Ours**                | [Syracuse data repository](https://datasets.syr.edu/datasets/Digg.html) — `Digg-dataset.zip` | 116,893 / ≈2.6M     |
| **DeepIM & IMINFECTOR** | [ISI / Lerman, "Digg 2009"](https://www.isi.edu/~lerman/downloads/digg2009.html)             | 279,613 / 1,170,689 |

Confirmed from the [IMINFECTOR README](https://github.com/geopanag/IMINFECTOR),
which names the ISI URL as its Digg source; DeepIM cites Panagopoulos et al.
(IMINFECTOR) for its Digg graph. The ISI version also carries **diffusion
cascades**, which is why IMINFECTOR could use it — ours is a friendship graph only.

#### Twitter — different repository, unrelated graph

|               | Source                                                                                        | Nodes / Edges  |
| ------------- | --------------------------------------------------------------------------------------------- | -------------- |
| **Ours**      | [SNAP ego-Twitter](https://snap.stanford.edu/data/ego-Twitter.html) `twitter_combined.txt.gz` | 81,306 / ≈1.3M |
| **ToupleGDD** | [Network Repository](https://networkrepository.com) (Rossi & Ahmed, AAAI'15)                  | 0.8k / 1k      |

ToupleGDD's §VI-A states: _"Twitter, Wiki-1, caGr and Buzznet are from [37],
while Wiki-2, Epinions and Youtube are available on [38]"_, where **[37] = Rossi &
Ahmed, Network Repository, AAAI 2015** and **[38] = Leskovec & Krevl, SNAP**
(verified in their reference list). A 800-node Network-Repository graph and SNAP's
81k-node ego-network share nothing but the word "Twitter".

This same reference split **explains the YouTube match**: their YouTube comes from
[38] = SNAP, which is exactly the `com-Youtube` file our loader downloads.

#### Weibo — same source, different construction

Both trace to AMiner's ["Influence Locality"](https://www.aminer.cn/influencelocality)
release (`aminer.cn` and `aminer.org` are the same site). Ours reads
`weibo_network.txt` directly → **1,787,443 users**, the raw file's own count.
DeepIM's 2,251,166 comes via IMINFECTOR, which builds its graph as the **union of
the follower network and every user appearing in the cascade files** — so it has
more nodes than the network file alone. Same underlying release, different graph
construction.

#### NetHEPT — mirrors of the same file

Ours is a [mirror of Wei Chen's original NetHEPT](https://github.com/SparklyYS/Simultaneous-IMM)
(15,229 / 62,752); the classical papers report 15,233 / 58,891. The node counts
agree to 0.03%; the edge gap is consistent with different multi-edge/self-loop
dedup. Fine for trend comparison, worth a footnote in a table.

#### NetScience — unresolved, and the evidence is contradictory

**Our graph is provably the one DeepIM cites.** DeepIM attributes Network Science
to Rossi & Ahmed (Network Repository). Our loader pulls from
[Netzschleuder](https://networks.skewed.de/net/netscience), and its statistics
match [networkrepository.com/netscience.php](https://networkrepository.com/netscience.php)
on four independent measures:

| Statistic      | Network Repository | Ours (computed) |
| -------------- | ------------------ | --------------- |
| Nodes / Edges  | 1.6K / 2.7K        | 1,589 / 2,742   |
| Max degree     | 34                 | 34              |
| Density        | 0.00217332         | 0.00217         |
| Avg clustering | 0.637791           | 0.638           |

Both are M. Newman's 2006 co-authorship network. But DeepIM reports **1,565 /
13,532** — 4.9× the edges. Two hypotheses were tested and **both fail**:

- _Not LCC extraction._ The LCC of this graph is **379 nodes**, not 1,565.
  (`ca-netscience` on Network Repository is exactly that 379/914 component.)
- _Not a typo._ Their Table 1 avg-degree column reads 17.28, and
  2 × 13,532 / 1,565 = 17.29 — internally consistent, so it describes a real
  graph they actually ran on.

**Conclusion: DeepIM's "Network Science" is a denser graph than the Newman
network it cites, and the paper does not say how it was produced.** Resolving it
requires their `netscience_25c.SG` pickle, which is not in the public repo
([`main/utils.py::load_dataset`](https://github.com/triplej0079/DeepIM/blob/main/main/utils.py)
reads `data/<name>_25c.SG`, and that folder ships empty).

Until then, our NetScience results cannot be compared to their column.

---

## 2. Classical methods

| Method              | Year | Venue  | Idea                                                                               | Paper                                                                                                                                                             | Code                                               |
| ------------------- | ---- | ------ | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| **Greedy + MC**     | 2003 | KDD    | Submodular greedy, (1−1/e) guarantee. Defines the problem.                         | [Kempe et al.](https://www.cs.cornell.edu/home/kleinber/kdd03-inf.pdf)                                                                                            | —                                                  |
| **CELF**            | 2007 | KDD    | Lazy-forward marginal gains, ~700× faster than greedy                              | [Leskovec et al.](https://www.cs.cmu.edu/~jure/pubs/detect-kdd07.pdf)                                                                                             | —                                                  |
| **Degree Discount** | 2009 | KDD    | Discount degree by already-chosen neighbours. Still the strongest cheap heuristic. | [Chen et al.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/weic-kdd09_influence.pdf)                                                       | —                                                  |
| **PMIA / MIA**      | 2010 | KDD    | Maximum-influence-arborescence local trees                                         | [Chen et al.](https://www.microsoft.com/en-us/research/publication/scalable-influence-maximization-for-prevalent-viral-marketing-in-large-scale-social-networks/) | —                                                  |
| **LDAG**            | 2010 | ICDM   | Local DAG under LT                                                                 | [Chen et al.](https://ieeexplore.ieee.org/document/5693962)                                                                                                       | —                                                  |
| **CELF++**          | 2011 | WWW    | CELF with a secondary-gain cache                                                   | [Goyal et al.](https://snap.stanford.edu/class/cs224w-readings/goyal11celf.pdf)                                                                                   | —                                                  |
| **SIMPATH**         | 2011 | ICDM   | LT-specific path enumeration                                                       | [Goyal et al.](https://ieeexplore.ieee.org/document/6137213)                                                                                                      | —                                                  |
| **IRIE**            | 2012 | ICDM   | Influence-rank iteration + influence estimation, no MC                             | [Jung et al. (arXiv 1111.4795)](https://arxiv.org/abs/1111.4795)                                                                                                  | —                                                  |
| **StaticGreedy**    | 2013 | CIKM   | Fixed live-edge snapshots, ~2 orders faster                                        | [Cheng et al. (arXiv 1212.4779)](https://arxiv.org/abs/1212.4779)                                                                                                 | —                                                  |
| **RIS**             | 2014 | SODA   | Reverse Influence Sampling — the foundation of everything below                    | [Borgs et al.](https://arxiv.org/abs/1212.0884)                                                                                                                   | —                                                  |
| **TIM / TIM+**      | 2014 | SIGMOD | RIS with estimated sample size, near-optimal time                                  | [Tang et al. (arXiv 1404.0900)](https://arxiv.org/abs/1404.0900)                                                                                                  | [code](https://sourceforge.net/projects/timplus/)  |
| **SKIM**            | 2014 | CIKM   | Sketch-based reachability                                                          | [Cohen et al.](https://arxiv.org/abs/1408.6282)                                                                                                                   | —                                                  |
| **IMM**             | 2015 | SIGMOD | Martingale RIS. **The standard classical reference point.**                        | [Tang et al.](https://dl.acm.org/doi/10.1145/2723372.2723734)                                                                                                     | [code](https://sourceforge.net/projects/im-imm/)   |
| **SSA / D-SSA**     | 2016 | SIGMOD | Stop-and-stare RIS, fewer samples                                                  | [Nguyen et al. (arXiv 1605.07990)](https://arxiv.org/abs/1605.07990)                                                                                              | [code](https://github.com/hungnt55/Stop-and-Stare) |
| **OPIM-C**          | 2018 | SIGMOD | Online processing RIS, near-optimal anytime                                        | [Tang et al.](https://dl.acm.org/doi/10.1145/3183713.3183749)                                                                                                     | [code](https://github.com/tangj90/OPIM)            |
| **SubSIM**          | 2020 | SIGMOD | Sublinear-time RIS                                                                 | [Guo et al.](https://dl.acm.org/doi/10.1145/3318464.3389740)                                                                                                      | [code](https://github.com/qtguo/subsim)            |

Our library (`coding_agent/tools/algorithms.py`) implements simplified versions of
most of these — see `research_notes/ALGORITHMS.md` for per-function fidelity notes.

---

## 3. Learning-based methods

| Method                 | Year      | Venue           | Approach                                                                                             | Paper                                                                                                                 | Code                                                                     |
| ---------------------- | --------- | --------------- | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| **Inf2vec**            | 2018      | ICDE            | Cascade + network embedding                                                                          | [Feng et al.](https://ieeexplore.ieee.org/document/8509310)                                                           | —                                                                        |
| **IMINFECTOR**         | 2020      | TKDE            | Multi-task NN on real cascades; influencer + susceptible vectors. **Model-free** — no IC/LT assumed. | [arXiv 1904.08804](https://arxiv.org/abs/1904.08804)                                                                  | [github.com/geopanag/IMINFECTOR](https://github.com/geopanag/IMINFECTOR) |
| **DISCO**              | 2019      | —               | structure2vec + DQN, marginal-gain reward                                                            | [Li et al.](https://arxiv.org/abs/1906.07378)                                                                         | —                                                                        |
| **GCOMB**              | 2020      | NeurIPS         | GraphSAGE pruning + Q-learning; scales to billion-edge                                               | [Manchanda et al.](https://proceedings.neurips.cc/paper/2020/hash/e7532dbeff7ef901f2e70daacb3f452d-Abstract.html)     | [github.com/idea-iitd/GCOMB](https://github.com/idea-iitd/GCOMB)         |
| **PIANO**              | 2022      | TCSS            | Evolved DISCO with fuller analysis                                                                   | [Li et al.](https://ieeexplore.ieee.org/document/9769766)                                                             | —                                                                        |
| **GLIE**               | 2021→2023 | ASONAM          | GNN learns an upper bound on spread; replaces MC                                                     | [arXiv 2108.04623](https://arxiv.org/abs/2108.04623)                                                                  | code link unverified (previously cited repo 404s)                        |
| **LeNSE**              | 2022      | ICML            | Learns to prune to a solvable subgraph                                                               | [Ireland & Montana](https://proceedings.mlr.press/v162/ireland22a.html)                                               | [github.com/davidireland3/LeNSE](https://github.com/davidireland3/LeNSE) |
| **ToupleGDD**          | 2023      | TCSS            | 3 coupled GNNs + Double DQN. Trains on <300-node graphs, generalizes to 1M+.                         | [arXiv 2210.07500](https://arxiv.org/abs/2210.07500)                                                                  | [github.com/Dtrycode/ToupleGDD](https://github.com/Dtrycode/ToupleGDD)   |
| **DeepIM** ⭐          | 2023      | **ICML**        | Autoencoder over seed sets + GNN diffusion; optimizes in latent space. IC/LT/SIS.                    | [arXiv 2305.02200](https://arxiv.org/abs/2305.02200) · [PMLR](https://proceedings.mlr.press/v202/ling23b/ling23b.pdf) | [github.com/triplej0079/DeepIM](https://github.com/triplej0079/DeepIM)   |
| **MOEIM** ⭐           | 2024      | GECCO           | Many-objective EA (spread, budget, fairness, communities, time) with graph-aware operators           | [arXiv 2403.18755](https://arxiv.org/abs/2403.18755)                                                                  | [github.com/eliacunegatti/MOEIM](https://github.com/eliacunegatti/MOEIM) |
| **DeepIM-accelerated** | 2024      | Neural Networks | DeepIM with faster inference                                                                         | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0893608024005732)                              | —                                                                        |
| **HIM**                | 2025      | —               | Influence strength estimation in hyperbolic space                                                    | [arXiv 2502.13571](https://arxiv.org/abs/2502.13571)                                                                  | —                                                                        |
| **REM**                | 2025      | —               | Seed2Vec VAE + RL, **multiplex** networks (different problem)                                        | [arXiv 2501.00779](https://arxiv.org/abs/2501.00779)                                                                  | —                                                                        |
| **Topic-aware IM**     | 2025      | DAMI            | GAT + DRL, magnetic Laplacian PE                                                                     | [Springer](https://link.springer.com/article/10.1007/s10618-025-01133-3)                                              | —                                                                        |

**Surveys:** [ML-based IM survey, TKDD 2023 (arXiv 2211.03074)](https://arxiv.org/abs/2211.03074) ·
[IM survey 2023 (arXiv 2309.04668)](https://arxiv.org/abs/2309.04668) ·
[Deep-RL for max-coverage benchmark (arXiv 2406.14697)](https://arxiv.org/abs/2406.14697)

---

## 4. Results — DeepIM (ICML 2023) ⭐ SOTA table

**The single most directly comparable published result to our setup**: same
budget convention (1/5/10/20% of nodes), same IC weighted-cascade probability
(`p = 1/in-degree`), same reported metric (**% of nodes infected**), and two of
its graphs are byte-identical to ours.

Protocol: seed 1/5/10/20% of nodes, simulate to termination, average influence
spread over **100 rounds**. LT thresholds ~ U[0.3, 0.6]. `−` = out-of-memory.

Dataset sizes **as used by DeepIM** (Table 1) [verified]:

|       | Jazz  | Cora-ML | Network Science | Power Grid | Synthetic | Digg      | Weibo       |
| ----- | ----- | ------- | --------------- | ---------- | --------- | --------- | ----------- |
| Nodes | 198   | 2,810   | 1,565           | 4,941      | 50,000    | 279,613   | 2,251,166   |
| Edges | 2,742 | 7,981   | 13,532          | 6,594      | 250,000   | 1,170,689 | 225,877,808 |

### 4.1 IC diffusion — % of nodes infected [verified, Table 2]

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
| **DeepIM** ⭐ | **14.1**   | **28.1** | **39.6** | **52.4** | **7.8**   | **20.9** | **31.5** | **51.2** | **6.3**      | **21.0** | **32.5** | **52.4** | **4.9** | **23.3** | **41.5** | 49.9 |

Large graphs, IC [verified]:

| Method        | Synthetic 1% | 5%       | 10%      | 20%      | Digg 1% | 5%       | 10%      | 20%      | Weibo 1% | 5%       | 10%      | 20%      |
| ------------- | ------------ | -------- | -------- | -------- | ------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| IMM           | 9.2          | 26.2     | 36.3     | 51.2     | 7.4     | 18.4     | 32.8     | 46.9     | 9.5      | 23.8     | 36.4     | 50.3     |
| OPIM          | 9.6          | 25.3     | 36.6     | 51.7     | 7.6     | 18.5     | 32.9     | 48.9     | 9.7      | 23.7     | 36.6     | 50.3     |
| SubSIM        | 9.5          | 26.7     | 36.5     | 51.5     | 7.5     | 18.9     | 33.3     | 49.4     | 9.3      | 23.1     | 36.5     | 50.6     |
| OIM           | 9.6          | 26.2     | 36.7     | 51.3     | 7.8     | 18.2     | 33.1     | 49.6     | —        | —        | —        | —        |
| IMINFECTOR    | 9.1          | 26.2     | 36.1     | 51.5     | 7.9     | 18.6     | 33.5     | 49.8     | 9.4      | 23.5     | 36.9     | 50.3     |
| PIANO         | 9.1          | 26.4     | 36.2     | 51.6     | —       | —        | —        | —        | —        | —        | —        | —        |
| ToupleGDD     | 9.5          | 26.8     | 37.1     | 51.4     | —       | —        | —        | —        | —        | —        | —        | —        |
| **DeepIM** ⭐ | **11.6**     | **27.4** | **38.7** | **52.1** | **8.4** | **19.3** | **34.2** | **51.3** | **11.2** | **26.5** | **37.9** | **51.8** |

### 4.2 LT diffusion — % of nodes infected [verified, Table 3]

| Method        | Cora-ML 1% | 5%       | 10%      | 20%      | NetSci 1% | 5%       | 10%      | 20%      | PowerGrid 1% | 5%       | 10%      | 20%      | Jazz 1% | 5%      | 10%      | 20%      |
| ------------- | ---------- | -------- | -------- | -------- | --------- | -------- | -------- | -------- | ------------ | -------- | -------- | -------- | ------- | ------- | -------- | -------- |
| IMM           | 1.7        | 34.8     | 52.2     | 66.4     | 2.5       | 11.9     | 18.1     | 33.6     | 4.6          | 19.9     | 31.7     | 56.9     | 1.4     | 5.7     | 13.4     | 24.5     |
| OPIM          | 2.3        | 36.9     | 51.2     | 71.5     | 1.6       | 12.0     | 18.8     | 34.1     | 4.4          | 17.6     | 30.1     | 55.5     | 1.4     | 6.9     | 12.6     | 20.9     |
| SubSIM        | 1.7        | 33.6     | 54.7     | 70.1     | 1.8       | 10.4     | 19.2     | 34.1     | 4.5          | 21.1     | 31.2     | 57.4     | 1.4     | 5.9     | 11.4     | 21.2     |
| IMINFECTOR    | 2.1        | 33.9     | 51.3     | 70.6     | 2.1       | 11.8     | 18.7     | 34.5     | 4.2          | 21.3     | 31.6     | 56.2     | 1.4     | 6.2     | 13.5     | 22.8     |
| PIANO         | 2.1        | 33.5     | 53.3     | 69.8     | 2.1       | 11.3     | 19.1     | 33.9     | 4.3          | 21.3     | 31.4     | 57.1     | 1.1     | 6.2     | 12.1     | 22.4     |
| ToupleGDD     | 2.3        | 36.2     | 54.5     | 70.9     | 2.8       | 12.4     | 19.8     | 34.6     | 4.8          | 21.9     | 32.6     | 58.1     | 1.4     | 6.5     | 12.9     | 23.6     |
| DeepIM_s      | 10.7       | 65.6     | 75.1     | 85.2     | 3.5       | 14.6     | 23.8     | 37.8     | 6.1          | 24.1     | 45.2     | 71.5     | 1.9     | 6.5     | 16.1     | 97.1     |
| **DeepIM** ⭐ | **13.4**   | **69.2** | **83.5** | **94.1** | **4.1**   | **16.6** | **26.7** | **41.5** | **6.3**      | **24.4** | **46.8** | **71.7** | **1.9** | **6.5** | **16.4** | **99.1** |

Large graphs, LT [verified]:

| Method        | Synthetic 1% | 5%      | 10%      | 20%      | Digg 1% | 5%       | 10%      | 20%      | Weibo 1% | 5%      | 10%      | 20%      |
| ------------- | ------------ | ------- | -------- | -------- | ------- | -------- | -------- | -------- | -------- | ------- | -------- | -------- |
| IMM           | 1.1          | 5.2     | 13.1     | 66.9     | 2.4     | 10.8     | 37.4     | 55.6     | 1.6      | 6.7     | 19.3     | 45.2     |
| OPIM          | 1.3          | 5.2     | 12.6     | 62.1     | 2.1     | 11.3     | 38.2     | 57.1     | 1.8      | 6.1     | 18.7     | 46.6     |
| SubSIM        | 1.4          | 5.5     | 13.1     | 69.6     | 2.4     | 11.3     | 37.9     | 56.9     | 1.7      | 6.7     | 19.2     | 46.8     |
| IMINFECTOR    | 1.3          | 5.5     | 13.5     | 67.4     | 2.2     | 11.1     | 38.9     | 58.7     | 1.8      | 6.4     | 18.6     | 47.5     |
| ToupleGDD     | 1.3          | 5.5     | 13.4     | 70.2     | —       | —        | —        | —        | —        | —       | —        | —        |
| **DeepIM** ⭐ | **1.5**      | **6.5** | **15.5** | **99.9** | **3.5** | **15.9** | **41.3** | **76.2** | **3.1**  | **7.6** | **39.3** | **72.4** |

### Reading these tables

- **The LT gap is where DeepIM's headline claim lives.** Cora-ML LT at 20%:
  DeepIM 94.1% vs best baseline 71.5% (OPIM). Synthetic LT at 20%: 99.9% vs
  70.2%. Jazz LT at 20%: 99.1% vs 24.5%. These are the ~200% improvements the
  paper advertises.
- **Under IC the field is nearly tied.** At 20% budget every method on Cora-ML
  lands in 50.2–52.4%. DeepIM's IC margin is ~1–2 points. This is the same
  saturation effect we hit on BA-100 — **at 20% budget on a well-connected
  graph, the problem is close to solved by any reasonable method.** It is a
  strong argument for reporting the 1% and 5% columns, where the spread between
  methods is 8.1→14.1 (74% relative).
- **1% budget is where methods actually separate.** Cora-ML IC 1%: IMM 8.1 vs
  DeepIM 14.1.

---

## 5. Results — MOEIM (GECCO 2024) ⭐ beats DeepIM

MOEIM is a many-objective evolutionary algorithm (spread ↑, seed-set size ↓,
communities ↑, fairness ↑, budget ↓, time ↓) with graph-aware mutation and
smart initialization. It is the most recent method found that **directly
compares against DeepIM on our datasets** and claims to beat it.

Setting-2 datasets (chosen to match DeepIM exactly) [verified, Table 1]:
Jazz 198/2,742 · Cora-ML 2,810/7,981 · Power Grid 4,941/6,594.
Budgets k ∈ {1%, 5%, 10%, 20%}, propagation WC and LT, `τ = ∞`.

**Result [claim + figure]:** MOEIM outperforms DeepIM on _almost all_ propagation
models and datasets. Reported specifics:

- **Jazz + LT** — MOEIM reaches whole-network influence at **k = 10%**, where
  DeepIM needs **k = 20%** for the same result. (DeepIM's own table shows Jazz LT
  16.4% at k=10 and 99.1% at k=20, so this is a large gap.)
- **Power Grid + LT** — the one exception: near-identical up to k = 10%, after
  which **DeepIM is superior**.
- Setting-1 (6 other graphs, hypervolume over 6 objectives): MOEIM wins 61 of 72
  cases vs GDD, CELF, MOEA.

Per-cell spread numbers are published only as non-dominated fronts in Figure 2 —
no table. To use MOEIM as a numeric baseline you would need to run
[their code](https://github.com/eliacunegatti/MOEIM).

MOEIM's setting-1 datasets (not ours, listed for completeness) [verified]:
email-eu-Core 986/25,552 · facebook-combined 4,039/88,234 · gnutella 6,299/20,776 ·
wiki-vote 7,066/103,663 · lastfm 7,624/27,806 · CA-HepTh 8,638/24,827.

---

## 6. Results — classical methods on NetHEPT

NetHEPT (arXiv High-Energy-Physics-Theory collaboration) is _the_ classical IM
benchmark. Standard protocol: **k = 1…50**, IC with weighted cascade
`p = 1/in-degree`, spread averaged over 10,000 MC simulations.

**IRIE paper, k = 50 [verified, Table 3]** — "ArXiv" here is NetHEPT (15,233 / 58,891):

| Dataset     | Model | SAEDV    | IRIE         |
| ----------- | ----- | -------- | ------------ |
| **NetHEPT** | WC    | 669.76   | **724.67**   |
| **NetHEPT** | TR    | 185.37   | **190.01**   |
| Epinions    | WC    | 11,177.3 | **12,063**   |
| Slashdot    | WC    | 14,803   | **16,712.3** |
| Amazon      | WC    | 487.67   | **824.80**   |
| DBLP        | WC    | 33,730   | **53,334.8** |

**NetHEPT k=50 under WC ≈ 700–725 nodes ≈ 4.6–4.8% of the graph** [figure, Fig 4.3a]:
Greedy/CELF, PMIA, IR, IRIE all converge to ≈700–730; **Degree alone reaches only
≈250** — a 3× gap that makes NetHEPT a genuinely discriminative benchmark, unlike
BA at high budget.

LiveJournal, k = 50 [verified, Table 2]:

| Algorithm | Weighted Cascade | Trivalency  |
| --------- | ---------------- | ----------- |
| IR        | **75,861.2**     | 629,484     |
| IRIE      | 74,830.5         | **629,694** |
| PMIA      | 71,566.5         | 629,512     |
| PageRank  | 51,162.3         | **629,892** |
| Degree    | 52,162.3         | 629,498     |

> Note the budget convention clash: classical IM sweeps **absolute k = 1…50**
> (k=50 is ~0.33% of NetHEPT), while the learning-based line sweeps
> **percentages, 1–20%** (20% of NetHEPT would be k=3,046). The two literatures
> barely overlap in budget regime. Our `--budget-pcts 1 5 10 20` follows the
> learning-based convention; add `--budgets 10 20 30 40 50` to also speak to the
> classical one.

---

## 7. Results — ToupleGDD (TCSS 2023)

Datasets [verified, Table I] — note these are almost entirely disjoint from ours,
**except YouTube**:

| Dataset      | n         | m      | Type       | Avg degree |
| ------------ | --------- | ------ | ---------- | ---------- |
| soc-dolphins | 62        | 159    | directed   | 5          |
| Twitter      | 0.8k      | 1k     | directed   | 2          |
| Wiki-1       | 0.9k      | 3k     | directed   | 6          |
| caGr         | 4.2k      | 13.4k  | undirected | 5          |
| Wiki-2       | 7.1k      | 103.7k | directed   | 29         |
| Epinions     | 76k       | 509k   | directed   | 13         |
| Buzznet      | 101k      | 3M     | directed   | 55         |
| **YouTube**  | **1.13M** | **3M** | undirected | 5          |

Budgets: **b ∈ {10, 20, 30, 40, 50}** (absolute). Edge weights: in-degree
(= weighted cascade), plus 0.1 and 0.5 uniform settings for generalization tests.

Expected spread, in-degree setting, iterative selection, train+test with initial
embedding [verified, Table II]:

| Dataset        | b=10     | b=20     | b=30     | b=40      | b=50      |
| -------------- | -------- | -------- | -------- | --------- | --------- |
| Twitter (0.8k) | 147.71   | 210.86   | 252.11   | 287.59    | 315.88    |
| caGr (4.2k)    | 213.13   | 368.74   | 489.15   | 602.95    | 696.95    |
| Wiki-2 (7.1k)  | 290.48   | 423.96   | 521.79   | 601.39    | 669.43    |
| Epinions (76k) | 6,022.85 | 8,303.34 | 9,693.69 | 10,866.88 | 11,781.69 |

**YouTube results are figure-only** [figure, Fig 3i]: all methods (IMM, OPIM-C,
ToupleGDD, S2V-DQN, PIANO, GCOMB) reach ≈50,000–60,000 spread at b=50, i.e.
≈5% of the 1.13M nodes. ToupleGDD ≈ IMM, both above OPIM-C.

**Claim:** ToupleGDD achieves spread "almost equal to IMM", outperforms OPIM-C on
Wiki-2/Buzznet/YouTube, and beats all other DRL methods. Its selling point is
**generalization** — trained on tiny random graphs, tested on 1M-node graphs.

---

## 8. Results — IMINFECTOR (TKDE 2020)

**Different evaluation protocol — do not compare its numbers to simulated spread.**
IMINFECTOR never assumes IC/LT. It splits real cascades 80/20 by time, picks seeds
from train, and measures **DNI (Distinct Nodes Influenced)** = the union of nodes
appearing in _held-out real cascades_ started by the chosen seeds.

Seed set sizes [verified]: **Digg k=50 · Weibo k=1,000 · MAG k=10,000.**

Results [figure, Fig 7]:

| Dataset         | Best method                  | ≈DNI at max k   | Notes                                            |
| --------------- | ---------------------------- | --------------- | ------------------------------------------------ |
| Digg (k=50)     | **Credit Distribution** ≈43k | IMINFECTOR ≈42k | CD wins but is ~10× slower; CELFIE flat at ≈31k  |
| MAG (k=10,000)  | **IMINFECTOR** ≈235k         | CELFIE ≈160k    | IMM-DB / Simpath-DB ≈110–125k                    |
| Weibo (k=1,000) | **IMINFECTOR** ≈450k         | K-cores ≈400k   | Only IMINFECTOR + K-cores + CELFIE scaled at all |

Notable finding the paper stresses: **IMM underperformed badly on cascade-based
evaluation**, which the authors attribute to diffusion-model misspecification —
"it performs poorly in this type of evaluation… IMM optimizes diffusion
simulations as part of its solution." Directly relevant to us: a method tuned on
a _simulator_ can lose on _real observed traces_.

---

## 9. SOTA summary per dataset

| Dataset             | Best published                                          | Model | Status                                                                |
| ------------------- | ------------------------------------------------------- | ----- | --------------------------------------------------------------------- |
| **Jazz**            | **MOEIM** (2024) > DeepIM (2023)                        | WC/LT | MOEIM reaches full spread at k=10% vs DeepIM's 20%                    |
| **Cora-ML**         | **MOEIM** (2024) > DeepIM (2023)                        | WC/LT | DeepIM LT@20% = 94.1%, IC@20% = 52.4%                                 |
| **Power Grid**      | **DeepIM** at k≥10% under LT; MOEIM below that          | WC/LT | The one dataset where DeepIM holds                                    |
| **Network Science** | **DeepIM** (2023)                                       | IC/LT | but on the 13,532-edge version, not ours                              |
| **NetHEPT**         | IMM/OPIM-C/IRIE tier (classical)                        | IC-WC | ≈725 nodes @ k=50; no learning-based paper reports it in the % regime |
| **Digg / Weibo**    | **DeepIM** (simulated) / **IMINFECTOR** (cascade-based) | —     | two incomparable protocols                                            |
| **YouTube**         | IMM ≈ ToupleGDD                                         | IC    | figure-only                                                           |
| **ER / BA / WS**    | ToupleGDD, GCOMB, DeepIM (synthetic)                    | IC/LT | training/generalization only, no canonical table                      |
| **SBM / Karate**    | **none found**                                          | —     | no published IM baseline                                              |

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

## 10. Implications for this project

1. **Our comparable-baseline set is Jazz and Power Grid.** Identical graphs,
   identical budget convention, published numbers in §4. Running our six
   conditions on those two graphs at 1/5/10/20% gives a table that drops straight
   into a paper alongside DeepIM's.

2. **Fix the Cora-ML and NetScience versions** if we want those rows to be
   comparable. Ours differ (§1); DeepIM's Network Science has **4.9× more edges**
   than ours. Either switch to their version or report ours as self-contained.

3. **The saturation problem is confirmed by the literature, not unique to us.**
   Every method on Cora-ML IC at 20% lands in 50.2–52.4%. Our BA-100 six-way tie
   is the same phenomenon. **The fix is to report low budgets (1%, 5%) where
   published methods separate by 74% relative**, and to prefer LT where the gaps
   are enormous.

4. **NetHEPT is the discriminative benchmark.** Degree gets ≈250 where
   greedy/PMIA/IRIE get ≈725 at k=50. If our agent has to beat degree by 3×,
   that's a real test — unlike BA where degree is already near-optimal.

5. **Add absolute-k budgets for the classical comparison.** The classical
   literature lives at k ≤ 50; the learning literature at 1–20% of N. Running
   `--budgets 10 20 30 40 50` alongside `--budget-pcts 1 5 10 20` lets us speak
   to both.

6. **SBM has no published IM baseline.** Our SBM experiments can't be positioned
   against prior work — they are a novel setting (and community structure is
   exactly where MOEIM's community objective and DeepIM's weakness live). Worth
   framing as a contribution rather than a comparison.

7. **IMINFECTOR's finding is a warning for us.** A method tuned against a
   simulator underperformed on real observed cascades. Our world model is trained
   on NDlib-generated transitions — the same class of assumption. Worth stating
   as a limitation.

---

## 11. Reference list

**Classical**
[Kempe 2003 (KDD)](https://www.cs.cornell.edu/home/kleinber/kdd03-inf.pdf) ·
[Leskovec 2007 CELF (KDD)](https://www.cs.cmu.edu/~jure/pubs/detect-kdd07.pdf) ·
[Chen 2009 DegreeDiscount (KDD)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/weic-kdd09_influence.pdf) ·
[Goyal 2011 CELF++ (WWW)](https://snap.stanford.edu/class/cs224w-readings/goyal11celf.pdf) ·
[Jung 2012 IRIE (arXiv 1111.4795)](https://arxiv.org/abs/1111.4795) ·
[Cheng 2013 StaticGreedy (arXiv 1212.4779)](https://arxiv.org/abs/1212.4779) ·
[Borgs 2014 RIS (arXiv 1212.0884)](https://arxiv.org/abs/1212.0884) ·
[Tang 2014 TIM (arXiv 1404.0900)](https://arxiv.org/abs/1404.0900) ·
[Tang 2015 IMM (SIGMOD)](https://dl.acm.org/doi/10.1145/2723372.2723734) ·
[Nguyen 2016 SSA (arXiv 1605.07990)](https://arxiv.org/abs/1605.07990) ·
[Tang 2018 OPIM (SIGMOD)](https://dl.acm.org/doi/10.1145/3183713.3183749) ·
[Guo 2020 SubSIM (SIGMOD)](https://dl.acm.org/doi/10.1145/3318464.3389740)

**Learning-based**
[IMINFECTOR (arXiv 1904.08804)](https://arxiv.org/abs/1904.08804) · [code](https://github.com/geopanag/IMINFECTOR) ·
[GCOMB (NeurIPS 2020)](https://proceedings.neurips.cc/paper/2020/hash/e7532dbeff7ef901f2e70daacb3f452d-Abstract.html) · [code](https://github.com/idea-iitd/GCOMB) ·
[GLIE (arXiv 2108.04623)](https://arxiv.org/abs/2108.04623) · code link 404 as of 2026-07 ·
[LeNSE (ICML 2022)](https://proceedings.mlr.press/v162/ireland22a.html) · [code](https://github.com/davidireland3/LeNSE) ·
[ToupleGDD (arXiv 2210.07500)](https://arxiv.org/abs/2210.07500) · [code](https://github.com/Dtrycode/ToupleGDD) ·
[DeepIM (arXiv 2305.02200)](https://arxiv.org/abs/2305.02200) · [PMLR](https://proceedings.mlr.press/v202/ling23b/ling23b.pdf) · [code](https://github.com/triplej0079/DeepIM) ·
[MOEIM (arXiv 2403.18755)](https://arxiv.org/abs/2403.18755) · [code](https://github.com/eliacunegatti/MOEIM) ·
[HIM (arXiv 2502.13571)](https://arxiv.org/abs/2502.13571) ·
[REM multiplex (arXiv 2501.00779)](https://arxiv.org/abs/2501.00779) ·
[Topic-aware IM (DAMI 2025)](https://link.springer.com/article/10.1007/s10618-025-01133-3) ·
[DeepIM accelerated (Neural Networks 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0893608024005732)

**Surveys / benchmarks**
[ML-based IM survey (arXiv 2211.03074, TKDD'23)](https://arxiv.org/abs/2211.03074) ·
[IM survey (arXiv 2309.04668)](https://arxiv.org/abs/2309.04668) ·
[Behaviour-aware IM survey (arXiv 2108.03438)](https://arxiv.org/abs/2108.03438) ·
[Deep-RL max-coverage benchmark (arXiv 2406.14697)](https://arxiv.org/abs/2406.14697)

**Data sources**
[SNAP](https://snap.stanford.edu/data/) ·
[Network Repository](https://networkrepository.com/) ·
[KONECT](http://konect.cc/) ·
[AMiner (Weibo)](https://www.aminer.cn/data-sna)

---

## 12. Open gaps in this review

Honest list of what this review could **not** establish:

- **MOEIM per-cell numbers** — published only as Pareto-front figures. Needs a
  code run to get a comparable table.
- **NetHEPT under the percentage-budget convention** — no learning-based paper
  reports NetHEPT at 1/5/10/20%. Our numbers there will have no direct precedent.
- **SBM and Karate** — no IM baselines found in any surveyed paper.
- **Our Twitter/Digg versions** — traced to different source repositories
  (§1.1); the literature's graphs of those names are different graphs, so no
  usable published baseline exists unless we switch sources.
- **DeepIM's Network Science graph** — 1,565 / 13,532 matches neither the source
  it cites nor that source's LCC, and their data folder ships empty (§1.1). The
  only way to close this is to obtain their `netscience_25c.SG` file.
- **GCOMB, GLIE, LeNSE per-dataset tables** — identified and linked, but their
  result tables were not transcribed (they benchmark on YouTube, Stack, and
  billion-edge graphs largely disjoint from ours).
