# Influence Maximization — Dataset Catalogue

Every graph used as an IM benchmark in the literature we baseline against, plus
every graph we already load. Node/edge counts, directedness, source URLs,
diffusion-relevant metadata, which paper uses which, and where the same name
denotes two different graphs.

Companion to `IM_RESEARCH.md` (methods, published result tables). This file owns
**datasets**; that file owns **results**.

---

## 0. Verification policy

| Tier           | Meaning                                                                       |
| -------------- | ----------------------------------------------------------------------------- |
| **[verified]** | Read from the repository's own statistics page, or from the paper's own table via text extraction (not a summarizer). |
| **[derived]**  | Computed by us from a file we downloaded, or reconstructed arithmetically from published splits. |
| **[claim]**    | Stated in prose by a paper or a secondary source; not cross-checked against the file. |

All URLs in this file returned HTTP 200 on **2026-07-28** unless annotated otherwise.

**Edge-count convention.** Undirected graphs are quoted as *undirected edges*;
directed graphs as *arcs*. Our loaders report `adjacency.nnz`, which for a
symmetrized undirected graph is **2× the undirected edge count**. This single
convention difference explains most apparent "discrepancies" between our numbers
and published tables — check it before concluding two graphs differ.

---

## 1. What we already load

`data/datasets/<name>.py`, dispatched by `data/wm_graphs.py::real_directed`.
Raw downloads land in `data/raw/<dataset>/` (gitignored, shared across runs).

### 1.1 Real graphs (15)

All counts below were **produced by running the loader**, not transcribed.
Undirected rows quote undirected edges; directed rows quote arcs.

| `--dataset`     | Nodes     | Edges           | Type                      | Node features          | Source | Auto-DL |
| --------------- | --------- | --------------- | ------------------------- | ---------------------- | ------ | ------- |
| `jazz`          | 198       | 2,742           | Undirected (collaboration)| log1p(degree)          | [nrvis `arenas-jazz.zip`](https://nrvis.com/download/data/misc/arenas-jazz.zip) · [page](https://networkrepository.com/arenas-jazz.php) | ✅ |
| `email_eu_core` | 1,005     | 24,929 arcs     | Directed (emails)         | log1p(total degree) · **42 department labels** | [SNAP email-Eu-core](https://snap.stanford.edu/data/email-Eu-core.html) | ✅ |
| `netscience`    | 1,589     | 2,742           | Undirected (coauthorship) | log1p(degree)          | [Netzschleuder `netscience`](https://networks.skewed.de/net/netscience) | ✅ |
| `cora_ml`       | 2,810     | 7,981           | Undirected (citations, standardized) | 2,879-dim bag-of-words, 7 labels | [graph2gauss `cora_ml.npz`](https://github.com/abojchevski/graph2gauss/raw/master/data/cora_ml.npz) | ✅ |
| `facebook`      | 4,039     | 88,234          | Undirected (friendships)  | log1p(degree)          | [SNAP ego-Facebook](https://snap.stanford.edu/data/ego-Facebook.html) | ✅ |
| `power_grid`    | 4,941     | 6,594           | Undirected (power lines)  | log1p(degree)          | [nrvis `opsahl-powergrid.zip`](https://nrvis.com/download/data/misc/opsahl-powergrid.zip) · [page](https://networkrepository.com/opsahl-powergrid.php) | ✅ |
| `ca_grqc`       | 5,242     | 14,484          | Undirected (coauthorship) | log1p(degree)          | [SNAP ca-GrQc](https://snap.stanford.edu/data/ca-GrQc.html) | ✅ |
| `wiki_vote`     | 7,115     | 103,689 arcs    | Directed (adminship votes)| log1p(total degree)    | [SNAP wiki-Vote](https://snap.stanford.edu/data/wiki-Vote.html) | ✅ |
| `lastfm_asia`   | 7,624     | 27,806          | Undirected (mutual follows)| log1p(degree) · **18 country labels** | [SNAP feather-lastfm-social](https://snap.stanford.edu/data/feather-lastfm-social.html) | ✅ |
| `nethept`       | 15,229    | 62,752 arcs     | Directed (coauthorship, both arcs) | log1p(total degree) | [SparklyYS/Simultaneous-IMM mirror](https://github.com/SparklyYS/Simultaneous-IMM) | ✅ |
| `netphy`        | 37,154    | 174,161         | Undirected (coauthorship) | log1p(degree)          | [Wei Chen `weic-graphdata.zip`](https://www.microsoft.com/en-us/research/people/weic/selected-projects/) | ✅ |
| `twitter`       | 81,306    | 1,768,149 arcs → symmetrized | Undirected (follows) | log1p(degree)  | [SNAP ego-Twitter](https://snap.stanford.edu/data/ego-Twitter.html) | ✅ |
| `digg`          | 116,893   | ≈2.6M           | Undirected (friendships)  | log1p(degree)          | [Syracuse `Digg-dataset.zip`](https://datasets.syr.edu/datasets/Digg.html) | ✅ |
| `youtube`       | 1,134,890 | 2,987,624       | Undirected (friendships)  | log1p(degree)          | [SNAP com-Youtube](https://snap.stanford.edu/data/com-Youtube.html) | ✅ |
| `weibo`         | 1,787,443 | ≈216M arcs      | Directed (influence u→v)  | log1p(total degree)    | [AMiner Influence Locality](https://www.aminer.cn/influencelocality) | ❌ **manual** |

**Simulable today:** everything down to and including `netphy` — twelve graphs.
The four large ones (`twitter`, `digg`, `youtube`, `weibo`) load fine but exceed
what the NDlib rollout + CELF/local-search selector pipeline finishes in
reasonable time; they are scalable-simulation targets, not day-one datasets.

**Weibo needs a human**: AMiner requires registration. See
`data/datasets/weibo.py` for the three-step manual procedure.

**Cora-ML is standardized on load.** `data/datasets/cora_ml.py` applies
graph2gauss's `SparseGraph.standardize()` (`make_unweighted`, `make_undirected`,
`no_self_loops`, `select_lcc` — all defaulting to True), so it produces DeepIM's
and MOEIM's 2,810 / 7,981 exactly rather than the raw file's 2,995 / 8,416. The
raw graph is not comparable to any published table, its 185 extra nodes are ~60
disconnected fragments that only add noise to a spread metric, and the 2,879-dim
features that distinguished it are never read by the model (`IN_CHANNELS = 6`;
`build_features` derives `CH_DEGREE` from `edge_index`). Nothing is lost by
dropping it.

**Three graphs carry real node labels** — `cora_ml` (7 topics),
`email_eu_core` (42 departments), `lastfm_asia` (18 countries). The rest get
placeholder zeros. `email_eu_core`'s departments are the only ground-truth
*community* labels in the suite, which makes it the natural graph for
community-aware evaluation (MOEIM's community objective; our SBM experiments,
which have no published baseline).

**First download of `cora_ml` takes minutes** — `cora_ml.npz` is 85 MB. That
loader writes to a `.part` file and renames on completion, so an interrupted
run retries cleanly. The other loaders skip when the target file exists, so a
run killed mid-download can leave a truncated cache; delete
`data/raw/<dataset>/` and re-run.

### 1.2 Synthetic families (5)

Generated in bulk via `--num-graphs`; `log1p(degree)` node features throughout.

| `--dataset` | Generator                     | Tunables                                    | Literature status |
| ----------- | ----------------------------- | ------------------------------------------- | ----------------- |
| `er`        | `nx.gnp_random_graph`         | `--er-p` (0.05)                             | standard (GCOMB, ToupleGDD, DeepIM, S2V-DQN) |
| `ba`        | `nx.barabasi_albert_graph`    | `--ba-m` (3)                                | standard |
| `ws`        | `nx.watts_strogatz_graph`     | `--ws-k` (6), `--ws-p` (0.1)                | standard |
| `sbm`       | `nx.stochastic_block_model`   | `--sbm-blocks/--sbm-p-in/--sbm-p-out`       | ⚠️ **no published IM baseline found** |
| `karate`    | `nx.karate_club_graph`        | —                                           | ⚠️ used by SL-VAE, not by IM papers |

DeepIM's "Synthetic" row (50,000 nodes / 250,000 edges, avg degree 10) is a
**random graph of that density** — an ER/BA-class graph we can reproduce with
`--dataset ba --syn-nodes 50000 --ba-m 5`. It is the only synthetic row in the
literature with a published per-cell table (see `IM_RESEARCH.md` §4).

---

## 2. ⚠️ Name collisions — read this before comparing any number

**Seven dataset names in this literature denote more than one graph.** Quoting a
published number against the wrong version is the single most common way to
produce an invalid comparison table. Each row below is two *different graphs*
that share a name.

| Name           | Version A                                     | Version B                                      | Who uses which |
| -------------- | --------------------------------------------- | ---------------------------------------------- | -------------- |
| **Epinions**   | `soc-Epinions1` **75,879 / 508,837** directed | `soc-sign-epinions` **131,828 / 841,372** directed | A: ToupleGDD · B: IMM, SSA, D-SSA, TIM, IRIE |
| **DBLP**       | `com-DBLP` **317,080 / 1,049,866** undirected | Chen/arnetminer DBLP **655K / 1.99M** undirected | A: LeNSE, GCOMB · B: IMM, SSA, TIM, PMIA, IRIE |
| **Twitter**    | SNAP `ego-Twitter` **81,306 / 1,768,149**     | Kwak 2010 crawl **41.7M / 1.47G**              | A: ours, GCOMB (TW-ew), LeNSE · B: IMM, SSA, TIM, GCOMB (TW) |
| **Twitter**    | (above)                                        | NetRepo `soc-twitter-follows` **0.8K / 1K**    | ToupleGDD's "Twitter" is this third one |
| **Wiki-Vote**  | NetRepo `soc-wiki-Vote` **889 / 2,900**        | SNAP `wiki-Vote` **7,115 / 103,689**           | ToupleGDD calls A "Wiki-1" and B "Wiki-2" |
| **Digg**       | Syracuse friendship **116,893 / ≈2.6M**       | ISI/Lerman 2009 **279,631 / 2,251,166** + cascades | A: ours · B: IMINFECTOR, DeepIM |
| **NetScience** | Newman 2006 **1,589 / 2,742**                  | DeepIM's **1,565 / 13,532** (unidentified)     | A: ours, NetRepo · B: DeepIM (§6.6 — unresolved) |
| **Weibo**      | AMiner raw network **1,787,443 / ≈216M**      | IMINFECTOR-derived **1,170,689 / 225,877,808** | A: ours · B: IMINFECTOR, DeepIM |
| **LiveJournal**| `com-LiveJournal` **3,997,962 / 34,681,189**  | `soc-LiveJournal1` **4,847,571 / 68,993,773**  | B is the one IMM/SSA report |

Additionally, several papers silently report the **largest connected component**
rather than the raw file. This is not a different graph, just a different
preprocessing switch — but it changes the numbers:

| Graph          | Raw file        | LCC as reported          | Reported by |
| -------------- | --------------- | ------------------------ | ----------- |
| Cora-ML        | 2,995 / 8,416   | **2,810 / 7,981**        | DeepIM, MOEIM |
| email-Eu-core  | 1,005 / 25,571  | **986 / 25,552**         | MOEIM |
| wiki-Vote      | 7,115 / 103,689 | **7,066 / 103,663**      | MOEIM |
| CA-HepTh       | 9,877 / 25,998  | **8,638 / 24,827**       | MOEIM |
| CA-GrQc        | 5,242 / 14,496  | **4,158 / 13,422**       | ToupleGDD ("caGr", 4.2k/13.4k) |
| p2p-Gnutella08 | 6,301 / 20,777  | **6,299 / 20,776**       | MOEIM ("gnutella") |
| facebook-combined | 4,039 / 88,234 | 4,039 / 88,234 (already connected) | MOEIM |

---

## 3. What to add — ranked recommendation

Ordered by *research value per hour of implementation*. Every Tier-1 graph is
small enough that the full pipeline (data gen → WM train → agent → MC referee)
finishes in the same order of time as NetScience.

### Tier 1 — ✅ DONE (small, canonical, published numbers exist)

All six are implemented and verified by running the loader, plus the `cora_ml`
standardization that makes Cora-ML directly comparable for the first time.

| `--dataset`     | Nodes  | Edges   | Why it earned a slot                                                            |
| --------------- | ------ | ------- | -------------------------------------------------------------------------------- |
| `netphy`        | 37,154 | 174,161 | The other half of the classical Chen/IRIE/PMIA/TIM/SSA canon. NetHEPT's sibling; every classical paper reports both. |
| `wiki_vote`     | 7,115  | 103,689 | Dense (avg total degree 29) and directed. ToupleGDD (as "Wiki-2") and MOEIM both report it. Our suite had no dense directed graph. |
| `email_eu_core` | 1,005  | 24,929  | MOEIM setting-1. Tiny, directed, **42 ground-truth departments** = the only real community labels in the suite. |
| `facebook`      | 4,039  | 88,234  | MOEIM setting-1. Strong community structure, already connected, no preprocessing decisions to get wrong. |
| `ca_grqc`       | 5,242  | 14,484  | ToupleGDD's "caGr". Sparse collaboration graph — the regime where degree heuristics are weakest. |
| `lastfm_asia`   | 7,624  | 27,806  | MOEIM setting-1. Ships **18 country labels**; matches MOEIM's count with no preprocessing. |
| `cora_ml`       | 2,810  | 7,981   | DeepIM's and MOEIM's Cora-ML, reproduced exactly via graph2gauss `standardize()` — replaces the raw 2,995-node graph. |

This gives **direct number-for-number comparability with MOEIM's setting-1
table** — it uses email-Eu-core, facebook, gnutella, wiki-vote, lastfm and
CA-HepTh, four of which we now have — plus a comparable Cora-ML row against
DeepIM's Table 2/3. Before this we could compare against **none** of setting-1.

Still missing from MOEIM setting-1: `p2p-Gnutella08` (6,301 / 20,777) and
`ca-HepTh` (9,877 / 25,998). Both are SNAP `.txt.gz` files in exactly the format
`ca_grqc.py` already parses — each is a copy-and-edit of that loader.

### Tier 2 — add when scalable simulation lands

| Dataset            | Nodes   | Edges      | Why                                                        |
| ------------------ | ------- | ---------- | ---------------------------------------------------------- |
| **Epinions (signed)** | 131,828 | 841,372 | The Epinions that IMM/SSA/TIM/IRIE actually report. IRIE publishes k=50 WC spread = 12,063 [verified]. |
| **Slashdot0902**   | 82,168  | 948,464    | LeNSE IM benchmark; classical directed social graph.       |
| **Enron**          | 36,692  | 183,831    | In the SSA/D-SSA table; small enough to be near-Tier-1.    |
| **Brightkite**     | 58,228  | 214,078    | GCOMB IM/MCP benchmark.                                    |
| **Gowalla**        | 196,591 | 950,327    | GCOMB IM/MCP benchmark.                                    |
| **com-DBLP**       | 317,080 | 1,049,866  | LeNSE + GCOMB. Note this is *not* the classical DBLP (§2). |
| **Deezer HR**      | 54,573  | 498,202    | LeNSE IM benchmark.                                        |
| **Buzznet**        | 101,200 | ≈2,800,000 | ToupleGDD; extremely high avg degree (54) — a saturation stress test. |

### Tier 3 — only if a scalability claim is being made

`com-Orkut` (3.07M/117M), `soc-LiveJournal1` (4.85M/69.0M), `as-Skitter`
(1.70M/11.1M), `wiki-Talk` (2.39M/5.02M), `sx-stackoverflow` (2.60M/63.5M),
`soc-Pokec` (1.63M/30.6M), Kwak Twitter (41.7M/1.47G), `com-Friendster`
(65.6M/1.81G). We already have three graphs in this class (`twitter`, `digg`,
`youtube`, `weibo`) that we cannot simulate — adding more before the simulator
scales buys nothing.

### Tier 4 — cascade datasets (a *different* evaluation protocol)

These carry observed diffusion traces, not just topology. They enable the
IMINFECTOR-style **DNI (Distinct Nodes Influenced)** protocol, which never
assumes IC/LT. Relevant because Yuntong scoped "real-world data = logged
trajectories" — this is the concrete form of that.

| Dataset        | Nodes     | Edges       | Cascades | Avg cascade | Source |
| -------------- | --------- | ----------- | -------- | ----------- | ------ |
| **Digg 2009**  | 279,631   | 2,251,166   | 3,553    | 847         | [ISI/Lerman](https://www.isi.edu/~lerman/downloads/digg2009.html) |
| **Sina Weibo** | 1,170,689 | 225,877,808 | 115,686  | 148         | [AMiner](https://www.aminer.cn/influencelocality) via [IMINFECTOR](https://github.com/geopanag/IMINFECTOR) |
| **MAG (CS)**   | 1,436,158 | 15,928,078  | 181,020  | 29          | Microsoft Academic Graph, CS subset, via IMINFECTOR |
| **Memetracker**| 12,529 (as used by SL-VAE) | — | 96M phrases | — | [SNAP MemeTracker](https://snap.stanford.edu/data/memetracker9.html) |
| **Flixster**   | 29,357    | 425,228 arcs| action log, 10 topics | — | topic-aware IM literature (Barbieri/Goyal) |

All five counts above are **[verified]** from IMINFECTOR's Table 3 by direct
text extraction, except Memetracker/Flixster which are **[claim]**.

### Explicitly *not* recommended

- **More synthetic families.** We already have five and `sbm`/`karate` have no
  published IM baseline. Adding a sixth generator adds no comparability.
- **cit-Patents, com-Amazon, com-Friendster.** Present in adjacent literature
  (max-cover, vertex-cover) but not in any IM table we baseline against.
- **NetScience "DeepIM version".** Cannot be added — it does not exist publicly
  (§6.6).

---

## 4. Full catalogue

### 4.1 Small graphs (< 10K nodes)

| Dataset            | Nodes | Edges   | Type       | Avg deg | Source | Used by |
| ------------------ | ----- | ------- | ---------- | ------- | ------ | ------- |
| soc-dolphins       | 62    | 159     | undirected | 5       | [NetRepo](https://networkrepository.com/soc-dolphins.php) | ToupleGDD |
| Karate             | 34    | 78      | undirected | 4.6     | `nx.karate_club_graph()` | SL-VAE |
| **Jazz** ✅        | 198   | 2,742   | undirected | 27.7    | [nrvis](https://nrvis.com/download/data/misc/arenas-jazz.zip) | DeepIM, MOEIM, SL-VAE |
| soc-wiki-Vote (NetRepo) | 889 | 2,900 | directed | 6      | [NetRepo](https://networkrepository.com/soc-wiki-Vote.php) | ToupleGDD ("Wiki-1") |
| **email-Eu-core** ✅ | 1,005 | 25,571 (24,929 usable) | directed   | 51      | [SNAP](https://snap.stanford.edu/data/email-Eu-core.html) | MOEIM |
| **NetScience** ✅  | 1,589 | 2,742   | undirected | 3.45    | [Netzschleuder](https://networks.skewed.de/net/netscience) | DeepIM, SL-VAE |
| **Cora-ML** ✅     | 2,810 | 7,981   | undirected | 5.7     | [graph2gauss](https://github.com/abojchevski/graph2gauss) + `standardize()` | DeepIM, MOEIM, SL-VAE |
| **ego-Facebook** ✅ | 4,039 | 88,234  | undirected | 43.7    | [SNAP](https://snap.stanford.edu/data/ego-Facebook.html) | MOEIM, LeNSE |
| **Power Grid** ✅  | 4,941 | 6,594   | undirected | 2.67    | [nrvis](https://nrvis.com/download/data/misc/opsahl-powergrid.zip) | DeepIM, MOEIM, SL-VAE |
| **ca-GrQc** ✅     | 5,242 | 14,496 (14,484 usable) | undirected | 5.5     | [SNAP](https://snap.stanford.edu/data/ca-GrQc.html) | ToupleGDD |
| p2p-Gnutella08     | 6,301 | 20,777  | directed   | 6.6     | [SNAP](https://snap.stanford.edu/data/p2p-Gnutella08.html) | MOEIM |
| **wiki-Vote (SNAP)** ✅ | 7,115 | 103,689 | directed   | 29.1    | [SNAP](https://snap.stanford.edu/data/wiki-Vote.html) | ToupleGDD ("Wiki-2"), MOEIM, LeNSE |
| **LastFM Asia** ✅ | 7,624 | 27,806  | undirected | 7.3     | [SNAP](https://snap.stanford.edu/data/feather-lastfm-social.html) | MOEIM |
| ca-HepTh           | 9,877 | 25,998  | undirected | 5.3     | [SNAP](https://snap.stanford.edu/data/ca-HepTh.html) | MOEIM |

### 4.2 Classical mid-size IM benchmarks (10K – 500K)

| Dataset            | Nodes   | Edges     | Type       | Avg deg | Source | Used by |
| ------------------ | ------- | --------- | ---------- | ------- | ------ | ------- |
| **NetHEPT** ✅     | 15,233  | 31,376 undirected (= 62,752 arcs) | undirected | 4.18 | [SparklyYS mirror](https://github.com/SparklyYS/Simultaneous-IMM) | Kempe, CELF, DegreeDiscount, PMIA, IRIE, TIM, IMM, SSA, OPIM, SubSIM |
| ca-CondMat         | 23,133  | 93,497    | undirected | 8.1     | [SNAP](https://snap.stanford.edu/data/ca-CondMat.html) | misc |
| Flixster [claim]   | 29,357  | 425,228 arcs | directed | 14.5    | topic-aware IM literature (Barbieri/Goyal) | TIM+, topic-aware IM |
| Enron              | 36,692  | 183,831   | undirected | 10.0    | [SNAP](https://snap.stanford.edu/data/email-Enron.html) | SSA, D-SSA, IMM, GLIE |
| **NetPHY** ✅      | 37,154  | 174,161 (231,584 raw lines; SSA quotes 180,826 ordered) | undirected | 9.38 | [Wei Chen `weic-graphdata.zip`](https://www.microsoft.com/en-us/research/people/weic/selected-projects/) | PMIA, IRIE, TIM, IMM, SSA |
| Brightkite         | 58,228  | 214,078   | undirected | 7.4     | [SNAP](https://snap.stanford.edu/data/loc-Brightkite.html) | GCOMB |
| Deezer HR          | 54,573  | 498,202   | undirected | 18.3    | [SNAP gemsec-Deezer](https://snap.stanford.edu/data/gemsec-Deezer.html) | LeNSE |
| soc-Epinions1      | 75,879  | 508,837   | directed   | 13.4    | [SNAP](https://snap.stanford.edu/data/soc-Epinions1.html) | ToupleGDD |
| **Twitter (ego)** ✅ | 81,306 | 1,768,149 arcs | directed | 43.5 | [SNAP](https://snap.stanford.edu/data/ego-Twitter.html) | GCOMB ("TW-ew"), LeNSE |
| Slashdot0902       | 82,168  | 948,464   | directed   | 23.1    | [SNAP](https://snap.stanford.edu/data/soc-Slashdot0902.html) | IRIE, LeNSE |
| Buzznet            | 101,200 | ≈2,800,000| undirected | 54      | [NetRepo](https://networkrepository.com/soc-buzznet.php) | ToupleGDD |
| **Digg (Syracuse)** ✅ | 116,893 | ≈2.6M | undirected | ≈45     | [Syracuse](https://datasets.syr.edu/datasets/Digg.html) | (ours only) |
| Epinions (signed)  | 131,828 | 841,372   | directed   | 13.4    | [SNAP](https://snap.stanford.edu/data/soc-sign-epinions.html) | IMM, SSA, D-SSA, TIM, IRIE |
| Gowalla            | 196,591 | 950,327   | undirected | 9.7     | [SNAP](https://snap.stanford.edu/data/loc-Gowalla.html) | GCOMB |
| Digg 2009 (ISI)    | 279,631 | 2,251,166 | directed   | 16.1    | [ISI/Lerman](https://www.isi.edu/~lerman/downloads/digg2009.html) | IMINFECTOR, DeepIM |
| com-DBLP           | 317,080 | 1,049,866 | undirected | 6.6     | [SNAP](https://snap.stanford.edu/data/com-DBLP.html) | LeNSE, GCOMB |
| com-Amazon         | 334,863 | 925,872   | undirected | 5.5     | [SNAP](https://snap.stanford.edu/data/com-Amazon.html) | IRIE (different Amazon), misc |
| Higgs Twitter      | 456,631 | 14,855,875| directed   | 65.1    | [SNAP](https://snap.stanford.edu/data/higgs-twitter.html) | cascade/IM literature |

### 4.3 Large scale (> 500K nodes)

| Dataset            | Nodes      | Edges         | Type       | Avg deg | Source | Used by |
| ------------------ | ---------- | ------------- | ---------- | ------- | ------ | ------- |
| DBLP (classical)   | 655,000    | 1,990,000     | undirected | 6.08    | Wei Chen / arnetminer release | IMM, SSA, TIM, PMIA, IRIE |
| **YouTube** ✅     | 1,134,890  | 2,987,624     | undirected | 5.3     | [SNAP](https://snap.stanford.edu/data/com-Youtube.html) | ToupleGDD, GCOMB, LeNSE, GLIE |
| Sina Weibo (IMINFECTOR) | 1,170,689 | 225,877,808 | directed | 386     | AMiner via IMINFECTOR | IMINFECTOR, DeepIM |
| MAG (CS)           | 1,436,158  | 15,928,078    | directed   | 22.2    | Microsoft Academic Graph | IMINFECTOR |
| soc-Pokec          | 1,632,803  | 30,622,564    | directed   | 37.5    | [SNAP](https://snap.stanford.edu/data/soc-Pokec.html) | misc |
| as-Skitter         | 1,696,415  | 11,095,298    | undirected | 13.1    | [SNAP](https://snap.stanford.edu/data/as-Skitter.html) | LeNSE |
| **Weibo (raw)** ✅ | 1,787,443  | ≈216M arcs    | directed   | ≈242    | [AMiner](https://www.aminer.cn/influencelocality) | (ours only) |
| Weibo (DeepIM)     | 2,251,166  | 225,877,808   | directed   | 200     | — see §6.5 (transcription error) | DeepIM |
| wiki-Talk          | 2,394,385  | 5,021,410     | directed   | 4.2     | [SNAP](https://snap.stanford.edu/data/wiki-Talk.html) | LeNSE ("Talk") |
| sx-stackoverflow   | 2,601,977  | 63,497,050 temporal / 36,233,450 static | directed | 27.8 | [SNAP](https://snap.stanford.edu/data/sx-stackoverflow.html) | GCOMB ("Stack", quotes 2.69M/5.9M after their action-log derivation) |
| com-Orkut          | 3,072,441  | 117,185,083   | undirected | 76.2    | [SNAP](https://snap.stanford.edu/data/com-Orkut.html) | IMM, SSA, GCOMB |
| cit-Patents        | 3,774,768  | 16,518,948    | directed   | 8.8     | [SNAP](https://snap.stanford.edu/data/cit-Patents.html) | adjacent literature |
| com-LiveJournal    | 3,997,962  | 34,681,189    | undirected | 17.3    | [SNAP](https://snap.stanford.edu/data/com-LiveJournal.html) | misc |
| soc-LiveJournal1   | 4,847,571  | 68,993,773    | directed   | 28.5    | [SNAP](https://snap.stanford.edu/data/soc-LiveJournal1.html) | IMM, SSA, TIM, IRIE |
| Twitter (Kwak'10)  | ≈41.7M     | ≈1.47G        | directed   | 70.5    | [KAIST](https://anlab-kaist.github.io/traces/WWW2010) · [LAW mirror](https://law.di.unimi.it/webdata/twitter-2010/) | IMM, SSA, TIM, GCOMB |
| com-Friendster     | 65,608,366 | 1,806,067,135 | undirected | 55.1    | [SNAP](https://snap.stanford.edu/data/com-Friendster.html) | SSA, GCOMB |

✅ = already implemented in `data/datasets/`.

---

## 5. Which paper uses which

Cells mark the dataset **as that paper reports it** — check §2 before assuming
two papers with the same cell used the same graph.

| Dataset       | Kempe'03 | PMIA'10 | IRIE'12 | TIM'14 | IMM'15 | SSA'16 | GCOMB'20 | IMINFECTOR'20 | LeNSE'22 | GLIE'23 | ToupleGDD'23 | DeepIM'23 | MOEIM'24 |
| ------------- | -------- | ------- | ------- | ------ | ------ | ------ | -------- | ------------- | -------- | ------- | ------------ | --------- | -------- |
| NetHEPT       | ✔        | ✔       | ✔       | ✔      | ✔      | ✔      |          |               |          |         |              |           |          |
| NetPHY        |          | ✔       | ✔       | ✔      | ✔      | ✔      |          |               |          |         |              |           |          |
| Enron         |          |         |         | ✔      | ✔      | ✔      |          |               |          | ✔       |              |           |          |
| Epinions      |          | ✔       | ✔       | ✔      | ✔      | ✔      |          |               |          |         | ✔            |           |          |
| Slashdot      |          |         | ✔       |        |        |        |          |               | ✔        |         |              |           |          |
| Amazon        |          |         | ✔       |        |        |        |          |               |          |         |              |           |          |
| DBLP          |          | ✔       | ✔       | ✔      | ✔      | ✔      |          |               | ✔        |         |              |           |          |
| LiveJournal   |          |         | ✔       | ✔      | ✔      | ✔      |          |               |          |         |              |           |          |
| Orkut         |          |         |         | ✔      | ✔      | ✔      | ✔        |               |          |         |              |           |          |
| Twitter       |          |         |         | ✔      | ✔      | ✔      | ✔        |               | ✔        |         | ✔            |           |          |
| Friendster    |          |         |         |        |        | ✔      | ✔        |               |          |         |              |           |          |
| YouTube       |          |         |         |        |        |        | ✔        |               | ✔        | ✔       | ✔            |           |          |
| Brightkite    |          |         |         |        |        |        | ✔        |               |          |         |              |           |          |
| Gowalla       |          |         |         |        |        |        | ✔        |               |          |         |              |           |          |
| Stack         |          |         |         |        |        |        | ✔        |               |          |         |              |           |          |
| Skitter       |          |         |         |        |        |        |          |               | ✔        |         |              |           |          |
| wiki-Talk     |          |         |         |        |        |        |          |               | ✔        |         |              |           |          |
| Deezer        |          |         |         |        |        |        |          |               | ✔        |         |              |           |          |
| Facebook      |          |         |         |        |        |        |          |               | ✔        | ✔       |              |           | ✔        |
| wiki-Vote     |          |         |         |        |        |        |          |               | ✔        |         | ✔            |           | ✔        |
| ca-GrQc       |          |         |         |        |        |        |          |               |          | ✔       | ✔            |           |          |
| ca-HepTh      |          |         |         |        |        |        |          |               |          |         |              |           | ✔        |
| email-Eu-core |          |         |         |        |        |        |          |               |          |         |              |           | ✔        |
| LastFM        |          |         |         |        |        |        |          |               |          |         |              |           | ✔        |
| Gnutella      |          |         |         |        |        |        |          |               |          |         |              |           | ✔        |
| soc-dolphins  |          |         |         |        |        |        |          |               |          |         | ✔            |           |          |
| Buzznet       |          |         |         |        |        |        |          |               |          |         | ✔            |           |          |
| Crime, HI-II-14 |        |         |         |        |        |        |          |               |          | ✔       |              |           |          |
| **Jazz**      |          |         |         |        |        |        |          |               |          |         |              | ✔         | ✔        |
| **Cora-ML**   |          |         |         |        |        |        |          |               |          |         |              | ✔         | ✔        |
| **NetScience**|          |         |         |        |        |        |          |               |          |         |              | ✔         |          |
| **Power Grid**|          |         |         |        |        |        |          |               |          |         |              | ✔         | ✔        |
| Digg          |          |         |         |        |        |        |          | ✔             |          |         |              | ✔         |          |
| Weibo         |          |         |         |        |        |        |          | ✔             |          |         |              | ✔         |          |
| MAG           |          |         |         |        |        |        |          | ✔             |          |         |              |           |          |
| Synthetic ER/BA/WS | ✔   |         |         |        |        |        | ✔        |               |          | ✔       | ✔            | ✔         |          |

**Bold = we already have it.** The four bold rows are the entire intersection
between our suite and the learning-based IM literature.

GLIE additionally uses **Crime** (829 / 2,946) and **HI-II-14** (4,165 / 26,172),
both small and both [verified] from its Table I.

---

## 6. Version forensics

*Moved here from `IM_RESEARCH.md` §1.1.* Each discrepancy between our version of
a graph and the literature's, traced to its cause.

| Dataset        | Cause                                                              | Status |
| -------------- | ------------------------------------------------------------------ | ------ |
| **NetHEPT**    | Edge-count *convention*, not a different graph                     | ✅ **resolved — same graph** |
| **Cora-ML**    | Largest-connected-component extraction; same source file           | ✅ **applied in the loader** |
| **Digg**       | Different source repository (Syracuse vs ISI/Lerman)               | ❌ different graph |
| **Twitter**    | Different repository *and* different graph (SNAP vs NetRepo)       | ❌ unrelated graph |
| **Weibo**      | Same source; DeepIM's table has a **transcription error**          | ⚠️ see §6.5 |
| **NetScience** | Their numbers match neither the cited source nor its LCC           | ❓ **unresolved** |

### 6.1 NetHEPT — resolved, it is the same graph

`IM_RESEARCH.md` previously flagged our NetHEPT (15,229 / 62,752) as differing
from the literature's "15,233 / 58,891". It does not. We now have the **primary
source**: Wei Chen's own `weic-graphdata.zip` (the archive `netphy.py`
downloads) ships `hep.txt`, whose header line reads `15233 58891` [verified].

Deduplicating that file ourselves [derived]:

```
hep.txt header               : 15,233 nodes, 58,891 edge LINES
unique undirected pairs      : 31,398   (including 39 self-loops)
minus self-loops             : 31,359 edges  -> 62,718 arcs
our SparklyYS mirror         : 31,376 edges  -> 62,752 arcs, 15,229 nodes
```

So the three published figures are three counts of one file: **58,891 = raw
lines** (a pair appears once per co-authored paper), **31.4K = deduplicated
undirected edges** — which is what the SSA/D-SSA table reports [verified,
`p913-Huang.pdf` Table 2] — and our 62,752 is that same edge set stored as both
arcs. Our mirror differs from Chen's original by **17 edges and 4 nodes
(0.05%)**.

**NetHEPT is therefore directly comparable**, and the classical k=1…50 tables in
`IM_RESEARCH.md` §6 apply to our graph without a caveat.

Corollary: NetHEPT is a **co-authorship** network, not a citation network. Our
loader's docstring said "edge (a, b) means paper a cites paper b" — that was
wrong and is now fixed. The adjacency was always symmetric, so nothing about the
data or the simulation changes.

### 6.2 Cora-ML — solved, and applied

Both we and DeepIM use **the same file**: `cora_ml.npz` from
[graph2gauss](https://github.com/abojchevski/graph2gauss). DeepIM loads it
through that project's `SparseGraph` class and calls `standardize()`:

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

**Adding symmetrize → drop self-loops → keep LCC to `data/datasets/cora_ml.py`
reproduces their graph bit-for-bit.** This is a ~10-line change and it converts
Cora-ML from a caveated row into a directly comparable one.

### 6.3 Digg — different source repository

|                         | Source | Nodes / Edges |
| ----------------------- | ------ | ------------- |
| **Ours**                | [Syracuse](https://datasets.syr.edu/datasets/Digg.html) `Digg-dataset.zip` | 116,893 / ≈2.6M |
| **IMINFECTOR & DeepIM** | [ISI / Lerman "Digg 2009"](https://www.isi.edu/~lerman/downloads/digg2009.html) | 279,631 / 2,251,166 |

The [IMINFECTOR README](https://github.com/geopanag/IMINFECTOR) names the ISI URL
as its Digg source; DeepIM cites IMINFECTOR for its Digg graph. The ISI version
also carries **diffusion cascades**, which is why IMINFECTOR could use it — ours
is a friendship graph only.

### 6.4 Twitter — different repository, unrelated graph

|               | Source | Nodes / Edges |
| ------------- | ------ | ------------- |
| **Ours**      | [SNAP ego-Twitter](https://snap.stanford.edu/data/ego-Twitter.html) | 81,306 / 1,768,149 |
| **ToupleGDD** | [Network Repository](https://networkrepository.com) | 0.8K / 1K |
| **IMM/SSA/TIM** | [Kwak et al. 2010](https://anlab-kaist.github.io/traces/WWW2010) | 41.7M / 1.47G |

ToupleGDD §VI-A: *"Twitter, Wiki-1, caGr and Buzznet are from [37], while Wiki-2,
Epinions and Youtube are available on [38]"*, where **[37] = Rossi & Ahmed,
Network Repository** and **[38] = Leskovec & Krevl, SNAP**. This same reference
split **explains the YouTube match**: their YouTube comes from SNAP, exactly the
`com-Youtube` file our loader downloads.

### 6.5 Weibo — DeepIM's table contains a transcription error

**This corrects the earlier "different graph construction" explanation.**

IMINFECTOR's own Table 3 reads [verified, direct text extraction]:

| | Digg | MAG | Sina Weibo |
| --- | --- | --- | --- |
| Nodes | 279,631 | 1,436,158 | **1,170,689** |
| Edges | **2,251,166** | 15,928,078 | 225,877,808 |

DeepIM's Table 1 — which cites IMINFECTOR for both graphs — reads Digg
**279,613 / 1,170,689** and Weibo **2,251,166 / 225,877,808**.

Those are **the same four numbers reassigned across cells**: Digg's edge count
(2,251,166) became Weibo's node count, and Weibo's node count (1,170,689) became
Digg's edge count. (279,631 → 279,613 is an additional digit transposition.)

**Consequence:** DeepIM's Digg and Weibo *column headers* are wrong, but the
graphs it ran on are IMINFECTOR's. Our Weibo (1,787,443 nodes, read straight from
`weibo_network.txt`) is still a different construction from IMINFECTOR's
1,170,689 — but the gap is much smaller than DeepIM's table implies, and the
earlier "union of follower network and cascade users" explanation is not
supported by the numbers.

### 6.6 NetScience — unresolved, evidence is contradictory

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

- *Not LCC extraction.* The LCC of this graph is **379 nodes**, not 1,565.
  (`ca-netscience` on Network Repository is exactly that 379/914 component.)
- *Not a typo.* Their Table 1 avg-degree column reads 17.28, and
  2 × 13,532 / 1,565 = 17.29 — internally consistent, so it describes a real
  graph they actually ran on.

**Conclusion: DeepIM's "Network Science" is a denser graph than the Newman
network it cites, and the paper does not say how it was produced.** Resolving it
needs their `netscience_25c.SG` pickle, which is not in the public repo (their
`data/` folder ships empty).

Until then, our NetScience results cannot be compared to their column.

---

## 7. Budget conventions — the other comparability trap

The two literatures barely overlap in budget regime:

| Convention | Range | Used by |
| ---------- | ----- | ------- |
| **Absolute k** | k ∈ {10, 20, 30, 40, 50} | Kempe, CELF, PMIA, IRIE, TIM, IMM, SSA, OPIM, SubSIM, ToupleGDD, GCOMB |
| **Percent of N** | 1%, 5%, 10%, 20% | DeepIM, MOEIM, and our `--budget-pcts` |

k=50 is **0.33% of NetHEPT**; 20% of NetHEPT is k=3,046. A paper reporting
"NetHEPT k=50 → 725 nodes influenced" and a paper reporting "NetHEPT 20% → 51%"
are not measuring comparable things.

**Recommendation:** run `--budgets 10 20 30 40 50` alongside
`--budget-pcts 1 5 10 20` on any graph where we want to speak to both.

**Edge probabilities.** Two settings dominate, and they are not interchangeable:

| Setting | Definition | Used by |
| ------- | ---------- | ------- |
| **WC (weighted cascade)** | `p(u→v) = 1 / in-degree(v)` | DeepIM, MOEIM, IMM, ToupleGDD, and our `--prob-model weighted` (default) |
| **TR (trivalency)** | `p` drawn uniformly from {0.1, 0.01, 0.001} | IRIE, PMIA, classical line |
| **Uniform** | `p = 0.1` or `p = 0.01` | ToupleGDD generalization tests, our `--prob-model uniform --uniform-p` |

LT thresholds: DeepIM uses `θ_v ~ U[0.3, 0.6]`; we use `θ_v ~ U(0,1)` (NDlib
default). **This is a real difference** — a narrower threshold band makes LT
cascades substantially easier to ignite, and it partly explains DeepIM's very
high LT numbers (Jazz LT@20% = 99.1%).

---

## 8. How to add a dataset

The contract is two files and one line. `--dataset` takes a free string; the
registry that validates it is `real_directed` in `data/wm_graphs.py`.

1. **Write `data/datasets/<name>.py`** exposing exactly two functions:

   ```python
   def download_<name>() -> Path: ...
   def load_<name>(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
       # returns (adjacency (N,N), node_feats (N,F), node_labels (N,), num_nodes)
   ```

   Download into `data/raw/<name>/`, skip if present, print `[✓]`/`[↓]` like the
   existing loaders. `data/graph_utils.py` carries the four steps every loader
   needs, so the body is usually ~6 lines:

   ```python
   raw_edges = np.loadtxt(path, comments="#", dtype=np.int64)   # shape: (E, 2)
   remapped, num_nodes = remap_to_contiguous(raw_edges)         # sparse SNAP ids
   adjacency = edges_to_adjacency(remapped[:, 0], remapped[:, 1],
                                  num_nodes, directed=False)    # dedups multi-edges
   node_feats = degree_features(adjacency, directed=False)      # shape: (N, 1)
   ```

   plus `largest_connected_component(adjacency)` when reproducing a paper that
   reports the LCC (it returns the surviving indices so features and labels can
   be subset identically). `ca_grqc.py` is the shortest complete example;
   `lastfm_asia.py` shows the real-labels variant.

2. **Add one line to `data/wm_graphs.py::real_directed`**: `"<name>": True|False`.
   This is the whole registration — `make_real_bundle` imports the module lazily
   by name and calls the two functions by convention.

3. **Add the row** to §1.1 above, to `README.md`, and to `CLAUDE.md`.

Nothing else needs to change: `pipeline/run.py`, the simulator, the feature
builder, and every baseline adapter all consume `GraphBundle`/`GraphInfo` and are
dataset-agnostic.

**Preprocessing decisions to make explicitly** (they are why §2 exists): keep or
drop the LCC, symmetrize or keep directed, drop self-loops, collapse multi-edges.
Record the choice in the loader docstring, because it *is* the difference between
two "same-name" graphs.

---

## 9. Sources

**Repositories**
[SNAP](https://snap.stanford.edu/data/) ·
[Network Repository](https://networkrepository.com/) ·
[Netzschleuder](https://networks.skewed.de/) ·
[KONECT](http://konect.cc/networks/) ·
[AMiner](https://www.aminer.cn/data-sna) ·
[ISI/Lerman](https://www.isi.edu/~lerman/downloads/) ·
[Syracuse](https://datasets.syr.edu/) ·
[LAW (Twitter-2010 mirror)](https://law.di.unimi.it/webdata/twitter-2010/)

**Dataset tables transcribed for this file**
[Revisiting SSA (VLDB'17) Table 2](http://www.vldb.org/pvldb/vol10/p913-Huang.pdf) — NetHEPT/NetPHY/Enron/Epinions/DBLP/Orkut/LiveJournal/Twitter ·
[IMINFECTOR (TKDE'20) Table 3](https://arxiv.org/abs/1904.08804) — Digg/MAG/Weibo + cascades ·
[LeNSE (ICML'22) Table 2](https://arxiv.org/abs/2205.10106) — train/test edge splits (full graphs reconstructed by summation) ·
[GCOMB (NeurIPS'20) Table 1a](https://arxiv.org/abs/1903.03332) ·
[GLIE (arXiv 2108.04623) Table I](https://arxiv.org/abs/2108.04623) ·
[ToupleGDD (TCSS'23) Table I](https://arxiv.org/abs/2210.07500) ·
[DeepIM (ICML'23) Table 1](https://proceedings.mlr.press/v202/ling23b/ling23b.pdf) ·
[MOEIM (GECCO'24) Table 1](https://arxiv.org/abs/2403.18755) ·
[IRIE (ICDM'12) Table 3](https://arxiv.org/abs/1111.4795) ·
[PMIA (KDD'10)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/msr-tr-2010-2_v2.pdf) — NetHEPT/NetPHY sizes

---

## 10. Open gaps

- **DeepIM's Network Science graph** (1,565 / 13,532) matches neither the source
  it cites nor that source's LCC, and their `data/` folder ships empty (§6.6).
- ~~**NetPHY edge count**~~ — **RESOLVED.** Chen's `phy.txt` deduplicates three
  ways, and all three published figures are accounted for [derived]:
  **231,584** raw lines (Chen et al. Table 1) → **180,826** unique *ordered*
  pairs, which is SSA's "181K" and reproduces their avg-degree 9.73 exactly
  (2 × 180,826 / 37,154 = 9.73) → **174,161** unique *undirected* pairs, which
  is what our loader builds.
- **Our Digg edge count** (≈2.6M) has never been pinned exactly — the loader
  prints it at load time but no number is recorded here.
- **Our Weibo edge count** (≈216M) likewise.
- **Memetracker and Flixster sizes** are [claim] only; no table was transcribed.
- **GCOMB's "Stack"** (2.69M / 5.9M) is a derived action-log graph, not raw
  `sx-stackoverflow` (2.60M / 63.5M static-36.2M). Their derivation is not
  reproduced here.
- **SBM and Karate** have no published IM baseline in any paper surveyed.
