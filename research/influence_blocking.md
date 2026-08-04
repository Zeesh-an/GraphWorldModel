# Influence Blocking — Prior Work, Datasets, and Published Results

Reference for the **influence blocking maximization (IBM)** literature and its aliases: rumour blocking, misinformation containment, contamination / influence minimization, competitive influence maximization, and multi-cascade (multi-campaign) diffusion. They are one literature — every one of them is a two-or-more-cascade diffusion in which one party moves to reduce what another party spreads, and they differ only in _which lever_ the blocker pulls and _who wins a tie_.

This is the task our five-op action space was built for and does not yet exercise: `remove_node`, `remove_edge` and `set_edge_weight` are all first-class interventions here and all idle in pure IM (see [`influence_maximization.md`](influence_maximization.md) §2).

All URLs returned HTTP 200 on **2026-07-28** unless annotated otherwise.

---

## 0. Verification policy

| Tier           | Meaning                                                                                                              |
| -------------- | -------------------------------------------------------------------------------------------------------------------- |
| **[verified]** | Read from the paper's own table via text extraction, or from the data repository's own statistics page. Trustworthy. |
| **[derived]**  | Computed by us from a file we downloaded, or reconstructed arithmetically from published splits.                     |
| **[figure]**   | Read off a plotted figure — the paper published no table. Approximate, direction only.                               |
| **[claim]**    | Stated in prose by a paper or a secondary source; not cross-checked against a file or table.                         |

Every number below marked [verified] was read from the paper's own table via `pdftotext -layout`, never from a PDF summarizer. This literature is unusually easy to get wrong because papers use the _same symbol_ `σ` for three different quantities (spread of the negative cascade, spread of the positive cascade, and the _reduction_ in negative spread), and because "edges" is quoted sometimes as arcs and sometimes as undirected pairs in the same table.

**Edge-count convention.** Undirected graphs are quoted as _undirected edges_; directed graphs as _arcs_. Our loaders report `adjacency.nnz`, which for a symmetrized undirected graph is **2× the undirected edge count**.

---

## 1. Task definition

Given a graph `G = (V, E)`, a diffusion model, a **negative seed set** `S_N` already committed (the rumour / misinformation / competitor), and a budget `k`, choose an intervention that minimizes how far the negative cascade gets.

The standard objective is stated as a _maximization of prevented influence_, so that greedy has something to climb:

```
σ(S_P, S_N) = E |IBS(S_P, S_N)|        "blocked influence" / "prevented influence"
IBS(S_P, S_N) = { v : v is negatively activated under S_N alone,
                      but NOT negatively activated under (S_P, S_N) }
S_P* = argmax_{S_P ⊆ V \ S_N, |S_P| ≤ k}  σ(S_P, S_N)
```

That is exactly `σ(negative alone) − σ(negative | blockers)`. The survey states the definition in this form [verified, Chen et al., *Physics Reports* 976 (2022), Definitions 1–3]:

> **Definition 1. Blocking set:** … the set of nodes that can be negatively influenced by `S_N` when there is no positive seed, but cannot be negatively influenced by `S_N` when the positive seed set is `S_P`. **Definition 2. Blocking influence:** `σ(S_P, S_N) = E(|IBS(S_P, S_N)|)`.

### 1.1 The four intervention levers

The literature splits cleanly by _what the blocker is allowed to do_, and this split maps one-to-one onto our action ops.

| Lever                       | Our op            | Names in the literature                                                                          | Representative work                                     |
| --------------------------- | ----------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------- |
| Seed a **counter-cascade**  | `add_node`        | influence blocking maximization, eventual influence limitation, rumour blocking, adversarial IBM | Budak 2011, He 2012, Fan 2013, Tong 2017                |
| **Delete nodes**            | `remove_node`     | node blocking, vertex blocking, influence minimization, node immunization                        | Xie 2023, Wang 2024 (VLDB)                              |
| **Delete edges**            | `remove_edge`     | link blocking, contamination minimization, edge deletion                                         | Kimura 2008/2009, Khalil 2014, DiffIM 2025              |
| **Reduce edge probability** | `set_edge_weight` | link-weight reduction, "blocking" as `p → 0`, permeability control                               | DiffIM 2025 (continuous relaxation), Kimura's `p`-sweep |

Seeding is by far the largest sub-literature; node/edge deletion is the second; explicit _continuous_ weight reduction is the smallest and newest, and is usually a relaxation of edge deletion rather than a modelling choice in its own right.

### 1.2 Why it exists as a separate problem

Kempe-style IM is monotone and submodular, so greedy is a `(1 − 1/e)` approximation and the field's remaining work is speed. **Blocking is not that problem.** Introducing a second cascade destroys monotonicity or submodularity in most natural model variants, which is why the literature is dominated by model-design papers rather than by faster solvers. §5.1 lays that fault line out explicitly; it is the single most decision-relevant fact here.

### 1.3 Naming

The survey notes that the same problem carries at least three names [verified, *Physics Reports* 976, §2.1]: Kimura et al. call it **pollution / contamination minimization** (link blocking); Budak et al. call it **negative influence limitation**; He et al. (SDM 2012) coined **influence blocking maximization**, which is now the standard term. Recent work adds **AIBM** (adversarial IBM) for the counter-seeding variant, reserving "node immunization" for node/edge removal [verified, arXiv 2511.16068 §1].

---

## 2. Fit with our methodology

**Verdict: direct fit, and the best-motivated next task.** Same graphs, same `(G, s_t, a_t, s_{t+1})` harvest, same structured-head trick. It costs one new simulator class and two extra state channels. Nothing about the encoder, the collate, the training loop, or the planning evaluator has to change.

### 2.1 The state extension — 2 channels become 4

Today `s_t = (infected, frontier)` and `X` is `(N, 6)` (`world_model/wm_data.py`: `in_channels = 6`, `ch_infected, ch_frontier, ch_degree, ch_add, ch_remove, ch_edge`). A two-cascade state needs the same pair per cascade:

| col | channel        | represents                                                             |
| --- | -------------- | ---------------------------------------------------------------------- |
| 0   | `NEG_INFECTED` | negatively activated at `t`                                            |
| 1   | `NEG_FRONTIER` | negatively activated _this step_ (still spreading)                     |
| 2   | `POS_INFECTED` | positively activated at `t`                                            |
| 3   | `POS_FRONTIER` | positively activated this step                                         |
| 4   | `DEGREE`       | `log1p(total degree)` in `A_t`                                         |
| 5   | `ADD`          | target of `add_node` this step — under IBM this _is_ the positive seed |
| 6   | `REMOVE`       | target of `remove_node` this step                                      |
| 7   | `EDGE`         | endpoint of an edge op this step                                       |

So `IN_CHANNELS = 6 → 8`. The three action channels keep their current meaning: the blocker only ever seeds _positively_, and the negative seed set `S_N` is an input to the episode, not an action — it is committed at `t = 0` exactly the way our spine selectors commit an IM seed set. **Nothing in the encoders changes** — they take `(N, in_channels)` and are agnostic to what the columns mean; only `build_features` and the `in_channels` constant move.

Targets double the same way: `y_inf, y_fr` become `y_inf_N, y_fr_N, y_inf_P, y_fr_P`, i.e. `logits (N, 2) → (N, 4)`, and the BCE loss gains two terms. `advance_marginal` already averages per-node hit counts over MC draws; it needs to count four dictionaries instead of two.

### 2.2 What it does to `ICTransmissionHead`

The structured head is where the interesting work is, and it stays closed-form. Run the existing IC product form **twice** off the shared encoding `h`, then compose with an explicit tie-break:

```
q^N_uv = σ(MLP_N([h_u, h_v, w_uv]))        p^N_new(v) = 1 − Π(1 − q^N_uv · frN_u)
q^P_uv = σ(MLP_P([h_u, h_v, w_uv]))        p^P_new(v) = 1 − Π(1 − q^P_uv · frP_u)
```

and then, for a susceptible `v` (not yet activated by either cascade):

| Tie-break                       | `P(v → negative)`           | `P(v → positive)`               |
| ------------------------------- | --------------------------- | ------------------------------- |
| **Negative dominance (ND)**     | `p^N`                       | `p^P · (1 − p^N)`               |
| **Positive dominance (PD)**     | `p^N · (1 − p^P)`           | `p^P`                           |
| **Fixed/random dominance (FD)** | `p^N·(1−p^P) + γ_v·p^N·p^P` | `p^P·(1−p^N) + (1−γ_v)·p^N·p^P` |

with `γ_v ∈ {0,1}` the per-node priority draw (or a learned probability, which is the natural relaxation). All three are differentiable and all three preserve the property that made the current head work: a susceptible node with no active in-neighbour in _either_ cascade has `p^N = p^P = 0`, so the rollout is still structurally self-terminating and cannot saturate.

Two honest caveats.

1. **The two-MLP factorization silently assumes MCICM, not COICM.** Under Budak's _Multi-Campaign_ IC each edge carries two independent probabilities `p_{L,u,v}` and `p_{C,u,v}` with independent coin flips, so the product form above is exact. Under _Campaign-Oblivious_ IC there is **one** coin per edge shared by both campaigns — an edge is live or blocked for both — and `p^N`/`p^P` are then _dependent_, so multiplying them is wrong. This matters because COICM is precisely the variant where Budak proves submodularity (§5.1). Getting the head right means picking the model first.
2. **LT is worse off than IC here, not better.** `LTThresholdHead` already pays a partial-observability tax because `θ_v` is hidden; under CLT every node draws _two_ hidden thresholds `θ^+_v, θ^-_v` [verified, He et al. SDM 2012 §3], so the one-step ceiling drops again. Expect IC to carry this task.

### 2.3 Which papers pull which lever

| Lever                     | Papers                                                                                                                                                                                                                                                           |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `add_node` (counter-seed) | Budak WWW'11 · Bharathi WINE'07 · Carnes ICEC'07 · Borodin WINE'10 · He SDM'12 (CLDAG) · Nguyen WebSci'12 · Fan ICDCS'13 · Wang ICDCS'13 · Wu & Pan 2017 (CMIA-H/CMIA-O) · Tong INFOCOM'17 · Tong NeurIPS'18 · StratLearner NeurIPS'20 · NIE 2023 · TC-AIBM 2025 |
| `remove_node`             | Xie et al. ICDE'23 (vertex blocking) · Wang et al. VLDB'24 (node blocking) · Xie et al. IJoC'25                                                                                                                                                                  |
| `remove_edge`             | Kimura AAAI'08 / TKDD'09 (link blocking) · Khalil KDD'14 · DiffIM AAAI'25                                                                                                                                                                                        |
| `set_edge_weight`         | DiffIM AAAI'25 — its continuous relaxation _is_ `p̃(u,v) = p(u,v)·r̃(u,v)` with `r̃ ∈ [0,1]`, i.e. literally our `set_edge_weight` [verified, arXiv 2502.01031 §5.3]                                                                                                |

DiffIM is the closest published analogue to what we would build: a GNN surrogate for influence, optimized by gradient descent through a continuous relaxation of the edge decisions. It is the existence proof that the differentiable-blocking idea works; it is _not_ action-conditioned or rolled forward in time.

### 2.4 What we must build — NDlib ships nothing

**No NDlib model has two competing cascade labels.** Enumerating `available_statuses` across every model in `ndlib/models/` [derived, read from the installed package] gives single-cascade status sets only: `IndependentCascadesModel` `{Susceptible, Infected, Removed}`, `ThresholdModel` `{Susceptible, Infected}`, and the `Blocked: -1` that appears in `KerteszThresholdModel` / `ProfileModel` is a **static, exogenously chosen set of permanent non-adopters**, not a competing cascade.

`CompositeModel` lets you declare arbitrary statuses and rules, but its `iteration()` loops the rules in registration order and `break`s on the first one that fires — so its only expressible tie-break is a single global "first rule wins" (i.e. fixed dominance with one global priority), and it has no path to per-edge IC thresholds the way `IndependentCascadesModel` does. It is not a usable competitive IC.

So the work is a `CompetitiveSimulator` alongside `Simulator` in `data/wm_simulator.py`. Concretely:

- `State` gains two lists (`pos_infected`, `pos_frontier`) — or becomes two `State`s; the dataclass is 15 lines and `to_dict` is trivial.
- `advance(bag)` runs one synchronous step: collect both frontiers, flip transmission coins per (edge, cascade), resolve simultaneous arrivals under the configured tie-break rule, commit. Roughly 120 lines, and it must be _synchronous_ — NDlib's own IC iterates nodes in dict order, which is fine for one cascade but would leak an implicit arbitrary tie-break for two.
- `advance_marginal` is unchanged in structure: same MC loop, four count dicts.
- `apply_actions` needs no change at all — the five ops mutate the graph and the status map exactly as now, with `add_node` writing the _positive_ label.

Reusing NDlib by running two `IndependentCascadesModel` instances and reconciling per step is tempting and **wrong**: two instances flip independent coins per edge, which silently commits you to MCICM even if you meant COICM, and NDlib mutates `model.status` in place so the reconciliation would have to re-enter both models every step anyway. Writing the step is cheaper than fighting it.

### 2.5 Cost estimate

| Item                                                        | Change                                                                                                                                    | Estimate   |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `CompetitiveSimulator` (+ 4-marginal MC)                    | new class in `data/wm_simulator.py`                                                                                                       | ~150 lines |
| Negative-seed selectors                                     | reuse the six spine selectors in `wm_actions.py` verbatim — the literature's attacker models are `degree`, `pagerank`, `random`, IMM (§8) | ~10 lines  |
| `build_features` → 8 channels, 4 targets                    | `world_model/wm_data.py`                                                                                                                  | ~40 lines  |
| `CompetitiveICHead` (two edge MLPs + tie-break)             | `world_model/wm_model.py`                                                                                                                 | ~80 lines  |
| Loss: 2 BCE terms → 4                                       | `train_wm.py`                                                                                                                             | ~5 lines   |
| `blocked_influence` metric + planning regret vs `σ(S_N, ∅)` | `wm_eval.py` / `wm_metrics.py`                                                                                                            | ~60 lines  |
| Encoders, collate, `GraphInput`, pipeline stages, report    | **unchanged**                                                                                                                             | 0          |

Call it **two to three focused days** for a working IC-only vertical slice on the small graphs, plus a data-regeneration run. LT/CLT is a further day and is worth deferring — the partial-observability tax (§2.2) makes it the weaker demonstration.

---

## 3. Classical and heuristic methods

Publisher landing pages marked Warning: return **403 to automated clients** (ACM DL, SIAM, ScienceDirect) and **202 for IEEE Xplore**; all open fine in a browser. Where an open PDF exists it is linked first.

### 3.1 Counter-seeding (`add_node`)

| Method                                                | Year  | Venue             | Idea                                                                                                                                                                                 | Paper                                                                                                                                                             | Code                                                                                                 |
| ----------------------------------------------------- | ----- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **Bharathi–Kempe–Salek**                              | 2007  | WINE              | First competitive IC. Game-theoretic: last player's best response is submodular; FPTAS on trees. Conjectured the two-edge-weight case stays submodular — later **disproved** (§5.1). | [Springer](https://link.springer.com/chapter/10.1007/978-3-540-77105-0_31)                                                                                        | —                                                                                                    |
| **Carnes et al.** (Wave Propagation / Distance-Based) | 2007  | ICEC              | Follower's perspective: two IC extensions, both submodular in the follower's seed set under a shared edge probability                                                                | Warning: [ACM `10.1145/1282100.1282167`](https://doi.org/10.1145/1282100.1282167)                                                                                       | —                                                                                                    |
| **Budak et al.** (EIL, MCICM / COICM)                 | 2011  | WWW               | _The_ founding blocking paper: limit a bad campaign with a good one; NP-hard; submodular only under restrictions                                                                     | [WWW'11 PDF](https://archives.iw3c2.org/www2011/proceedings/proceedings/p665.pdf) · Warning: [ACM](https://doi.org/10.1145/1963405.1963499)                             | —                                                                                                    |
| **Borodin–Filmus–Oren**                               | 2010  | WINE              | Four competitive **LT** models; proves non-submodularity for all but the OR model, and `Ω(N^{1/2−ε})` inapproximability                                                              | [Toronto PDF](http://www.cs.toronto.edu/~oren/cs_toronto/Publications_files/doc.pdf) · [Springer](https://link.springer.com/chapter/10.1007/978-3-642-17572-5_48) | —                                                                                                    |
| **CLDAG** (He, Song, Chen, Jiang)                     | 2012  | SDM               | Coined IBM. **CLT** model; proves submodularity; local-DAG solver, 2 orders faster than greedy                                                                                       | [MSR PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/weic-sdm12_infblockingmax.pdf) · [arXiv 1110.4723](https://arxiv.org/abs/1110.4723) | —                                                                                                    |
| **Greedy Viral Stopper** (Nguyen et al., βᵀᴵ)         | 2012  | WebSci            | Containment as "decontaminate a β-fraction within T rounds"                                                                                                                          | Warning: [ACM `10.1145/2380718.2380746`](https://doi.org/10.1145/2380718.2380746)                                                                                       | —                                                                                                    |
| **Least-Cost Rumor Blocking** (Fan et al.)            | 2013  | ICDCS             | Minimum _number_ of protectors to shield a target community; set-cover flavoured                                                                                                     | Warning: [IEEE `10.1109/ICDCS.2013.34`](https://doi.org/10.1109/ICDCS.2013.34)                                                                                          | —                                                                                                    |
| **Positive-influence maximization** (Wang et al.)     | 2013  | ICDCS             | Dual framing: maximize the positive cascade rather than minimize the negative                                                                                                        | Warning: [IEEE `10.1109/ICDCS.2013.37`](https://doi.org/10.1109/ICDCS.2013.37)                                                                                          | —                                                                                                    |
| **CMIA-H / CMIA-O** (Wu & Pan)                        | 2017  | Computer Networks | MIA-style local arborescences under two competitive IC variants. **The standard scalable IBM baseline** — NIE 2023 still benchmarks against CMIA-O.                                  | Warning: [`10.1016/j.comnet.2017.05.004`](https://doi.org/10.1016/j.comnet.2017.05.004)                                                                                 | —                                                                                                    |
| **RPS / Randomized rumor blocking** (Tong et al.)     | 2017  | INFOCOM → TNSE    | Reverse-reachable _R-tuples_ → `(1−1/e−ε)` with RIS-style sampling                                                                                                                   | [arXiv 1701.02368](https://arxiv.org/abs/1701.02368) · Warning: [TNSE](https://doi.org/10.1109/TNSE.2017.2783190)                                                       | —                                                                                                    |
| **Distributed rumor blocking** (Tong et al.)          | 2017  | —                 | Multiple _independent_ positive cascades, no coordination                                                                                                                            | [arXiv 1711.07412](https://arxiv.org/abs/1711.07412)                                                                                                              | —                                                                                                    |
| **Multi-cascade containment** (Tong & Wu)             | 2018  | NeurIPS           | ≥3 cascades with per-node **cascade priority**; the general problem is non-monotone _and_ non-submodular; submodular under three special priorities                                  | [arXiv 1809.06486](https://arxiv.org/abs/1809.06486)                                                                                                              | —                                                                                                    |
| **Reverse Prevention Sampling** (Simpson et al.)      | 2018→ | —                 | RIS adapted to _prevention_; the first `(1−1/e−ε)` mitigation algorithm at scale                                                                                                     | [arXiv 1807.01162](https://arxiv.org/abs/1807.01162)                                                                                                              | [github.com/stamps](https://github.com/stamps) (link as printed in the paper; not a resolvable repo) |
| **NAMM** (Simpson, Hashemi, Lakshmanan)               | 2022  | —                 | Differential propagation rates + temporal penalties (truth arrives late and travels slower)                                                                                          | [arXiv 2206.11419](https://arxiv.org/abs/2206.11419)                                                                                                              | —                                                                                                    |
| **Proactive rumor control** (Xu, Peng, Wang)          | 2023  | —                 | Impressions rather than adoptions; branch-and-bound                                                                                                                                  | [arXiv 2303.10068](https://arxiv.org/abs/2303.10068)                                                                                                              | —                                                                                                    |
| **BIS / TC-AIBM** (Shi et al.)                        | 2025  | —                 | Adds a **time constraint**; proves submodularity under all three tie-break rules; bidirectional (forward + reverse) sampling                                                         | [arXiv 2511.16068](https://arxiv.org/abs/2511.16068)                                                                                                              | —                                                                                                    |
| **CELF-R / fair IBM** (Fang et al.)                   | 2026  | —                 | Community-fair blocking via approximately-monotone submodular optimization                                                                                                           | [arXiv 2601.22584](https://arxiv.org/abs/2601.22584)                                                                                                              | —                                                                                                    |

### 3.2 Structural blocking (`remove_node`, `remove_edge`, `set_edge_weight`)

| Method                                               | Year | Venue                | Lever                      | Idea                                                                                                                                 | Paper                                                                                                                        | Code                                                                       |
| ---------------------------------------------------- | ---- | -------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **Kimura–Saito–Motoda** (contamination minimization) | 2008 | AAAI                 | `remove_edge`              | Block `k` links to minimize expected contamination; bond-percolation estimator; **plain greedy, no approximation guarantee claimed** | [AAAI PDF](https://cdn.aaai.org/AAAI/2008/AAAI08-186.pdf)                                                                    | —                                                                          |
| -> journal version                                    | 2009 | TKDD                 | `remove_edge`              | Same, extended                                                                                                                       | Warning: [ACM `10.1145/1514888.1514892`](https://doi.org/10.1145/1514888.1514892)                                                  | —                                                                          |
| **Khalil–Dilkina–Song**                              | 2014 | KDD                  | `add_edge` / `remove_edge` | Diffusion-aware topology optimization under LT; edge _addition_ is submodular, edge _deletion_ is supermodular                       | Warning: [ACM `10.1145/2623330.2623704`](https://doi.org/10.1145/2623330.2623704)                                                  | —                                                                          |
| **Vertex blocking** (Xie, Zhang, Wang, Lin, Zhang)   | 2023 | ICDE                 | `remove_node`              | IMIN by deleting vertices; proves NP-hard **and APX-hard**; sampling + `GreedyReplace`                                               | [arXiv 2302.13529](https://arxiv.org/abs/2302.13529) · Warning: [IEEE](https://doi.org/10.1109/ICDE55515.2023.00066)               | —                                                                          |
| **SandIMIN** (Wang et al.)                           | 2024 | **PVLDB**            | `remove_node`              | Sandwich approximation over a submodular lower bound of the non-submodular objective; `(1−1/e−ε)` on the bound                       | [arXiv 2405.12871](https://arxiv.org/abs/2405.12871) · Warning: [PVLDB](https://doi.org/10.14778/3675034.3675042)                  | [github.com/wjh0116/IMIN](https://github.com/wjh0116/IMIN)                 |
| **Blocking strategies** (Xie et al.)                 | 2025 | INFORMS J. Computing | `remove_node`              | Journal extension of the ICDE'23 line; `AdvancedGreedy` + `GreedyReplace` under IC/LT/TR                                             | [arXiv 2312.17488](https://arxiv.org/abs/2312.17488) · Warning: [`10.1287/ijoc.2024.0591`](https://doi.org/10.1287/ijoc.2024.0591) | [github.com/INFORMSJoC/2024.0591](https://github.com/INFORMSJoC/2024.0591) |

### 3.3 Competitive / multi-cascade models (not blocking per se, but the model layer)

| Model                                | Year | Venue | What it adds                                                                                                                                                                       | Paper                                                |
| ------------------------------------ | ---- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| **Com-IC** (Lu, Chen, Lakshmanan)    | 2015 | VLDB  | Unifies competition _and_ complementarity via four Global Adoption Probabilities `(q_{A\|∅}, q_{A\|B}, q_{B\|∅}, q_{B\|A})`; self-/cross-submodularity holds only in special cases | [arXiv 1507.00317](https://arxiv.org/abs/1507.00317) |
| **OCIM** (Online competitive IM)     | 2020 | —     | Unknown edge probabilities, bandit feedback, competitive setting                                                                                                                   | [arXiv 2006.13411](https://arxiv.org/abs/2006.13411) |
| **Algorithmic design for CIM**       | 2014 | —     | Survey/design space of competitive IM formulations                                                                                                                                 | [arXiv 1410.8664](https://arxiv.org/abs/1410.8664)   |
| **Cost-effective rumor containment** | 2014 | —     | Budget/cost variants of containment                                                                                                                                                | [arXiv 1403.6315](https://arxiv.org/abs/1403.6315)   |

**Survey.** Chen, Jiang, Chen et al., _Influence blocking maximization on networks: models, methods and applications_, **Physics Reports 976 (2022) 1–54** — [author PDF](https://www.cse.wustl.edu/~yixin.chen/public/survey.pdf) (200, but the server truncates: re-request until `pdftotext` succeeds) · Warning: [ScienceDirect](https://doi.org/10.1016/j.physrep.2022.05.003). Its Tables 1 and 2 (transcribed in §5.2) are the cleanest available taxonomy of competitive IC and LT variants.

---

## 4. Learning-based methods

**This literature is thin, and that is the headline.** IM has DeepIM, ToupleGDD, GCOMB, LeNSE, GLIE, IMINFECTOR, PIANO and a TKDD survey; blocking has _three_ serious learned methods (StratLearner, NIE, DiffIM) plus a handful of DRL competitive-IM papers that optimize the positive cascade rather than the prevented negative one. An arXiv full-text sweep for `"influence blocking" AND "learning"` returns **two** papers total, one of which is a fairness survey [derived, arXiv API query 2026-07-29]. Nobody has published an action-conditioned learned _simulator_ of a competitive cascade.

| Method                                                            | Year | Venue                | Approach                                                                                                                                                                                | What it learns                                                      | Paper                                                                                                                                                                | Code                                                                                   |
| ----------------------------------------------------------------- | ---- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **StratLearner** (key)                                               | 2020 | **NeurIPS**          | Structured prediction: learn a weighted combination of random subgraph features so that `argmax` over them reproduces the _optimal blocker set_. No diffusion parameters assumed known. | a map `2^V → 2^V` (attacker set → protector set)                    | [arXiv 2009.14337](https://arxiv.org/abs/2009.14337)                                                                                                                 | [github.com/cdslabamotong/stratLearner](https://github.com/cdslabamotong/stratLearner) |
| **NIE (Neural Influence Estimator)** (key)                           | 2023 | IEEE Trans.          | MLP over hand-built topological features of `(S_f, S_t)` replaces Monte-Carlo inside CELF. Real-time IBM.                                                                               | the _objective_ `f(S_t \| S_f)` — a surrogate, not a dynamics model | [arXiv 2308.14012](https://arxiv.org/abs/2308.14012)                                                                                                                 | no public code found                                                                   |
| **DiffIM** (key)                                                     | 2025 | AAAI                 | GNN surrogate for influence **+ continuous relaxation of edge removal** `p̃(u,v) = p(u,v)·r̃(u,v)`, optimized by gradient descent. IC primary; LT and G-SIR in the appendix.              | a surrogate `σ̂` _and_ a differentiable decision variable per edge   | [arXiv 2502.01031](https://arxiv.org/abs/2502.01031)                                                                                                                 | [github.com/junghunl/DiffIM](https://github.com/junghunl/DiffIM)                       |
| **DRL rumor influence minimization** (Jiang, Chen, Huang, Li, Du) | 2023 | Applied Intelligence | Deep RL agent selects blockers under a rumour-spread environment                                                                                                                        | a policy                                                            | [Springer OA PDF](https://link.springer.com/content/pdf/10.1007/s10489-023-04555-y.pdf) · [`10.1007/s10489-023-04555-y`](https://doi.org/10.1007/s10489-023-04555-y) | no public code found                                                                   |
| **SandIMIN**                                                      | 2024 | PVLDB                | Not learned, but the sandwich-bound structure is the natural place to drop a surrogate                                                                                                  | —                                                                   | [arXiv 2405.12871](https://arxiv.org/abs/2405.12871)                                                                                                                 | [github.com/wjh0116/IMIN](https://github.com/wjh0116/IMIN)                             |
| **Uncertainty-aware competitive IM (DRL)**                        | 2025 | —                    | Subjective-logic opinions + DRL; competitive IM, spreads _true_ information                                                                                                             | a policy                                                            | [arXiv 2504.15131](https://arxiv.org/abs/2504.15131)                                                                                                                 | no public code found                                                                   |
| -> earlier version                                                 | 2024 | —                    | Same line, "Winning the Social Media Influence Battle"                                                                                                                                  |                                                                     | [arXiv 2404.18826](https://arxiv.org/abs/2404.18826)                                                                                                                 | no public code found                                                                   |
| **JCCIM**                                                         | 2023 | —                    | Joint complementary & competitive IM with ally-boosting and rival-preventing                                                                                                            | —                                                                   | [arXiv 2302.09620](https://arxiv.org/abs/2302.09620)                                                                                                                 | no public code found                                                                   |
| **OCIM**                                                          | 2020 | —                    | Online (bandit) competitive IM: edge probabilities unknown, learned from feedback                                                                                                       | edge probabilities                                                  | [arXiv 2006.13411](https://arxiv.org/abs/2006.13411)                                                                                                                 | no public code found                                                                   |

### 4.1 How the three serious ones differ from us

|                                 | StratLearner               | NIE                       | DiffIM                            | **GWM (proposed)**                    |
| ------------------------------- | -------------------------- | ------------------------- | --------------------------------- | ------------------------------------- |
| Learns                          | solution map               | objective value           | objective + relaxed decision      | **transition kernel `s_t → s_{t+1}`** |
| Action-conditioned              | via the input attacker set | no                        | via `r̃` on edges                  | **yes, all five ops**                 |
| Rolls forward in time           | no                         | no                        | no                                | **yes**                               |
| Needs the true diffusion params | no                         | yes (for training labels) | yes                               | yes (for training labels)             |
| Lever                           | `add_node`                 | `add_node`                | `remove_edge` / `set_edge_weight` | all five                              |

The gap is exactly the one our project targets: all three learn a **static scoring function**, then hand it to a combinatorial outer loop (CELF, greedy, gradient descent). None of them can answer "what does the state look like next step if I do _this_", which is what a coding-agent loop needs to plan against.

**Warning taken from NIE, though**: NIE-CELF beats `MCSs-CELF` by `11,920×` on email-Eu-core runtime [verified, Table IV] while landing at an average _quality ratio_ of `0.88` against HMP\*-CELF [verified, Table V] — i.e. the surrogate bought four orders of magnitude of speed at a ~12% quality cost. That is the trade our rollout-vs-MC comparison will be measured against, and it is a real bar, not a strawman.

---

## 5. Published results

### 5.1 Warning: The submodularity fault line — read this first

In single-cascade IM, `σ` is monotone and submodular, full stop. In blocking it depends on the model _and on the tie-breaking rule_, and the difference decides whether greedy carries a `(1 − 1/e)` guarantee at all. Every row below was read from the paper's own theorem statement.

| Model / variant                                                                              | Monotone?          | Submodular?                                                 | Source                                                                           |
| -------------------------------------------------------------------------------------------- | ------------------ | ----------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **MCICM**, limiting campaign has the _high-effectiveness_ property (`p_L = 1` on every edge) | yes                | yes **yes**                                                  | [verified] Budak Thm 4.2                                                         |
| **MCICM**, general `p_L`                                                                     | yes                | **no** (explicit counter-example, Fig. 1)                | [verified] Budak §4.2                                                            |
| **COICM** (one campaign-independent probability per edge)                                    | yes                | yes **yes**                                                  | [verified] Budak Claim 2                                                         |
| **CLT** (competitive linear threshold, negative dominance)                                   | yes                | yes **yes**                                                  | [verified] He et al. SDM'12 §4.2                                                 |
| **Weight-proportional competitive LT**                                                       | **no**          | no                                                       | [verified] Borodin Thm 2.1, 2.2                                                  |
| **Separated-threshold competitive LT**                                                       | yes                | **no**, _for any tie-breaking rule and any forcing rule_ | [verified] Borodin Thm 3.2, 4.2                                                  |
| **OR model** (cascades spread independently, decide at the end)                              | yes                | yes **yes** → `(1−e⁻¹−ε)`                                    | [verified] Borodin Thm 6.1                                                       |
| **Wave Propagation** with technology-specific edge weights (Carnes' conjecture)              | —                  | **no** — conjecture disproved                            | [verified] Borodin Thm 5.2                                                       |
| **≥3 cascades, general per-node cascade priority**                                           | **no**          | **no**                                                   | [verified] Tong & Wu NeurIPS'18 §5, Fig. 3                                       |
| -> with **M-dominant** or **P-dominant** priority                                             | yes                | yes **yes**                                                  | [verified] Tong & Wu Thm 2                                                       |
| -> with **homogeneous** priority                                                              | yes                | yes **yes**                                                  | [verified] Tong & Wu Thm 3                                                       |
| **Time-constrained AIBM (TC-IC)** under PD, ND _or_ FD                                       | yes                | yes **yes**                                                  | [verified] TC-AIBM Thm 1                                                         |
| **IMIN by node blocking**                                                                    | —                  | **no**                                                   | [verified] SandIMIN §1 ("the objective function of IMIN is non-submodular [46]") |
| **Contamination minimization by link blocking**                                              | —                  | no guarantee claimed; plain greedy                          | [verified] Kimura AAAI'08                                                        |
| **Com-IC** self-/cross-submodularity                                                         | special cases only | no not in general                                           | [verified] Lu et al. §5.3                                                        |
| Edge **addition** submodular / edge **deletion** supermodular under LT                       | —                  | —                                                           | [claim] Khalil KDD'14 — ACM 403s, not verified from the PDF                      |

Hardness, quoted verbatim:

> **Theorem 5.1** [Borodin]. It is NP-hard to give an approximation with a ratio better than `Ω(N^{1/2−ε})`, for all `ε > 0`, for the Separated-Threshold Competitive Influence problem.

> **Theorem 1** [Tong & Wu]. For any `ε > 0`, there is no polynomial-time approximation algorithm for the Min-M problem with an approximation factor of `Ω(2^{log^{1−ε}|V*|})` unless `NP ⊆ DTIME(n^{polylog n})`.

Vertex blocking is **NP-hard and APX-hard unless P=NP** [verified, arXiv 2302.13529 §1].

**Two consequences for us.**

1. **Pick COICM or CLT if you want a defensible greedy oracle.** Our `--baselines` arm and the oracle evaluator both assume a greedy reference with a known guarantee. Under general MCICM there is none, and a greedy "oracle" would be a heuristic wearing a crown.
2. **The tie-break is not a detail, it is a modelling axis.** Borodin's Thm 4.2 says non-submodularity survives _every_ tie-break for separated thresholds; Tong & Wu's Thms 2–3 say submodularity is _created_ by restricting the priority. Any competitive simulator we write must make the rule an explicit, recorded parameter, not an artifact of node iteration order (§2.4).

### 5.2 Model taxonomy — survey Tables 1 and 2

[verified, Chen et al., *Physics Reports* 976 (2022), Tables 1–2]. Node-status column verbatim; "Notes" abridged.

| Model                | Node status                              | Note                                                                                                        |
| -------------------- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| IC [Kempe'03]        | inactive, active                         | no competition                                                                                              |
| **CIC**              | inactive, positive, negative             | "if a node is activated by both positive and negative influence, the node will be activated **negatively**" |
| HCIC                 | inactive, positive, negative             | CIC with homogeneous probabilities                                                                          |
| PIC                  | inactive, k-active                       | each active user activates at most one neighbour per step                                                   |
| **MCICM** [Budak'11] | inactive, positive, negative             | two probabilities per edge; "too complicated"                                                               |
| **COICM** [Budak'11] | inactive, positive, negative             | one probability per edge, independent of information type                                                   |
| TCO-ICM              | inactive, positive, negative             | adds node login probability                                                                                 |
| SNIC                 | inactive, positive, negative             | signed networks                                                                                             |
| TCC                  | inactive, active                         | adds a time factor                                                                                          |
| LT [Kempe'03]        | inactive, active                         | no competition                                                                                              |
| **CLT** [He'12]      | inactive, positive, negative             | simultaneous activation → **negative wins**                                                                 |
| CA                   | inactive, positive, negative             | adds a per-user preference parameter                                                                        |
| LT1DT                | inactive, R-active, T-active, influenced | activation probability changes over time                                                                    |
| MT-LT / K-LT         | inactive, k-active                       | `k` information types                                                                                       |
| WPCLT                | inactive, positive, negative             | adds "affected" pre-activation states                                                                       |
| DLT                  | infected, protected, inactive            | user's mind changes over time                                                                               |

Note that **CIC and CLT both hard-code negative dominance**, which is why so many papers report only that rule — and why TC-AIBM's three-rule treatment (§5.6) is the exception worth copying.

### 5.3 SandIMIN (PVLDB 2024) (key) the most comparable table

**Why this one.** It reports **decreased spread** — literally `σ(S) − σ(S | blockers)`, our prevented-influence metric — under **IC with weighted cascade `p = 1/in-degree`** (our `--prob-model weighted` default) at **absolute budgets `k = 10…50`** (the classical convention we can already run via `--budgets`), and two of its six result columns are graphs we load byte-comparably: **EmailCore** (1,005 / 25,571 ≈ our `email_eu_core`) and **YouTube** (1,134,890 / 2,987,624, identical to ours). And because its lever is `remove_node`, **it needs none of the §2 two-cascade machinery** — we could reproduce this table with today's simulator.

Protocol [verified, arXiv 2405.12871 §6]: `p(u,v) = 1/in-degree(v)`; misinformation seed set `|S| = 10` by default, drawn **randomly from the top-200 most influential nodes**; spread estimated by the average over **10⁵ Monte-Carlo simulations**; each setting repeated 10 times and averaged; 24-hour cutoff.

**Decreased spread by varying `k`** [verified, Table 5]:

| `k` | EmailCore yes GSBM | LSBM       | LHGA       | EmailAll GSBM | LSBM       | LHGA   | DBLP GSBM | LSBM       | LHGA       |
| --- | ----------------- | ---------- | ---------- | ------------- | ---------- | ------ | --------- | ---------- | ---------- |
| 10  | 26.726            | 31.292     | **33.140** | **313.66**    | 307.643    | 153.96 | 49.160    | 57.260     | **59.350** |
| 20  | **55.870**        | 52.544     | 49.614     | 405.77        | **455.15** | 240.40 | 65.790    | 67.150     | **74.980** |
| 30  | 66.408            | 70.629     | **72.460** | 538.61        | **582.23** | 332.03 | 93.000    | **98.620** | 80.850     |
| 40  | 81.816            | **83.650** | 80.379     | 588.76        | **643.87** | 455.00 | 114.45    | **116.67** | 92.290     |
| 50  | 90.907            | **102.50** | 97.018     | 664.88        | **714.80** | 505.38 | 128.21    | 136.95     | **143.41** |

| `k` | Stanford GSBM | LSBM       | LHGA   | YouTube yes GSBM | LSBM       | LHGA       | Pokec GSBM | LSBM       | LHGA       |
| --- | ------------- | ---------- | ------ | --------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| 10  | 6921.6        | **7123.6** | 217.35 | **1150.5**      | 705.20     | 980.40     | 785.90     | 656.50     | **1000.2** |
| 20  | 7631.3        | **8108.0** | 689.32 | **1962.6**      | 1795.9     | 1295.1     | **1506.9** | 1302.4     | 1111.6     |
| 30  | 8319.7        | **8991.2** | 940.73 | 2246.9          | 2284.5     | **2310.5** | **1922.9** | 1591.2     | 1545.3     |
| 40  | 8699.7        | **9688.3** | 1302.6 | **2743.6**      | 2694.1     | 2406.6     | 2071.9     | **2193.1** | 1580.2     |
| 50  | 9169.3        | **10168**  | 1991.2 | 3046.4          | **3271.5** | 2551.7     | **2734.1** | 2590.6     | 2179       |

`GSBM` = global-sampling bound maximization, `LSBM` = lower-bound sampling maximization (the `(1−1/e−ε)` component), `LHGA` = a trivial "highest-gain" heuristic. `SandIMIN` returns the best of the three.

**What jumps out — and it is the same lesson as our BA-100 six-way tie.** The trivial heuristic `LHGA` **wins outright in 6 of 30 cells** and is within a few percent in many more. On EmailCore at `k = 10`, `LHGA` (33.140) beats both principled methods; on Pokec at `k = 10` it beats them by 27%. A published VLDB paper's own table says its guaranteed algorithm is beaten by a greedy heuristic a fifth of the time. **Any blocking result we publish needs a degree/proximity heuristic in the table or it is not evidence.**

Second: the _scale_ of prevented influence is small relative to the graph. EmailCore has 1,005 nodes and blocking 50 of them prevents ~102 activations; YouTube has 1.13M nodes and blocking 50 prevents ~3,272 (0.29%). Blocking is a **low-signal regime** compared to IM, where 20% seeding infects half the graph. That has a direct consequence for us: `delta_f1` on a blocking dataset will be computed over far fewer positive labels, so class imbalance gets worse, not better, and the `--pos-weight off` guidance for structured heads needs re-checking on this task.

### 5.4 CLDAG (SDM 2012) — how hard blocking gets as the rumour grows

Datasets [verified, Table 1]: Mobile **15.5K / 37.0K**, avg degree 4.77 · NetHEPT **15.2K / 58.9K**, avg 7.75 · NetPHY **37.1K / 231.5K**, avg 12.48. (NetHEPT and NetPHY here are Chen's own files — see [`influence_maximization.md`](influence_maximization.md) §6.4.1 for why 58.9K is raw edge _lines_, and §6.2 for our matching 15,229 / 62,752 arcs yes.)

Setting for the table below [verified, §6.4]: **NetHEPT**, negative seeds chosen by **largest degree**, `p⁺ = p⁻ = 1`, positive seeds capped at 1,000. `σ_N(S, N₀)` = expected negative activations given positive seeds `S`.

| `\|N₀\|` | `σ_N(∅, N₀)` | `\|S\|` | `σ_N(S, N₀)` |
| -------- | ------------ | ------- | ------------ |
| 1        | 72.8979      | 23      | 6.7396       |
| 2        | 77.4516      | 68      | 6.0182       |
| 5        | 156.48       | 145     | 15.6667      |
| 10       | 213.077      | 199     | 20.6628      |
| 20       | 581.366      | 557     | 57.6617      |
| 50       | 963.633      | 926     | 95.8451      |
| 100      | 1006.37      | 1000    | 108.823      |
| 200      | 1669.85      | 1000    | 680.518      |
| 500      | 3635.95      | 1000    | 2640.8       |
| 1000     | 5836.48      | 1000    | 4845.58      |

[verified, Table 2]. The paper's own reading of it:

> The result shows that it requires about **20 to 30 times of positive seeds to reduce negative influence to about 10% level**, and it becomes increasingly hard to block negative influence. For example, with 1000 negative seeds, we spend an equal number of 1000 positive seeds but **can only reduce 17%** negative influence. Therefore, **first mover has a clear advantage**.

This is the single most useful design fact in the whole file for choosing an experimental regime: **blocking is only interesting when `|N₀|` is small.** At `|N₀| = 1000` the problem is effectively unsolvable at any budget we would run, and at `|N₀| ≤ 20` a ~25× budget buys a 90% reduction. Our `--budget-pcts` convention (1/5/10/20% of `N`) is _the wrong axis here_ — the meaningful ratio is `|S_P| / |S_N|`, not `|S_P| / |V|`.

Also verified from the same paper: the **degree heuristic is useless for blocking** — "Traditional degree heuristic cannot be used for influence blocking maximization at all from our test results" — while **proximity** (pick out-neighbours of the negative seeds) is a strong cheap baseline that only falls behind CLDAG when the negative cascade is strong enough to traverse long paths [verified, §6.3]. That is the opposite of IM, where degree is the strong cheap baseline. Our `wm_actions.py` spine selectors therefore need a **proximity** selector added for this task; `degree` alone would be a strawman.

### 5.5 NIE (2023) — the learned-surrogate speed/quality trade

Networks [verified, Table II]: power-law graph **768 / 1,532** (avg 1.99) · email-Eu-core yes **1,005 / 25,571** (25.44) · p2p-Gnutella08 **6,301 / 20,777** (3.30) · p2p-Gnutella24 **26,518 / 65,369** (2.47) · web-Stanford **281,903 / 2,312,497** (8.20).

Runtime in seconds to reach a fixed target quality, **email-Eu-core** yes [verified, Table IV]:

| Problem         | NIE-CELF | MCSs-CELF     | HMP\*-CELF | StratLearner\*-CELF | CMIA-O  | ACO-GE |
| --------------- | -------- | ------------- | ---------- | ------------------- | ------- | ------ |
| 1               | 0.78     | 9943.50       | 51.27      | 166.74              | 86.43   | 17.92  |
| 2               | 0.60     | 16092.73      | 50.74      | 253.14              | 77.16   | 16.96  |
| 3               | 1.06     | 3576.73       | 36.25      | 189.28              | 93.67   | 16.36  |
| 4               | 0.72     | 4839.09       | 33.82      | 192.95              | 109.01  | 16.08  |
| 5               | 0.65     | 6458.17       | 36.34      | 163.83              | 91.35   | 16.29  |
| **avg speedup** | —        | **11920.03×** | 57.48×     | 266.85×             | 123.94× | 22.81× |

Blocked influence within a **one-minute** budget, same graph [verified, Table V]:

| Problem               | NIE-CELF | MCSs-CELF | HMP\*-CELF | StratLearner\*-CELF | CMIA-O | ACO-GE |
| --------------------- | -------- | --------- | ---------- | ------------------- | ------ | ------ |
| 1                     | 51.67    | —         | 49.71      | —                   | —      | 28.90  |
| 2                     | 55.50    | —         | 55.72      | —                   | —      | 25.18  |
| 3                     | 38.89    | —         | 49.43      | —                   | —      | 19.68  |
| 4                     | 38.05    | —         | 52.60      | —                   | —      | 21.82  |
| 5                     | 41.05    | —         | 49.29      | —                   | —      | 19.79  |
| **avg quality ratio** | —        | —         | **0.88**   | —                   | —      | 1.96   |

`—` = failed to produce any solution inside the budget. Read the two tables together: the surrogate is ~10⁴× faster and ~12% _worse_ than the best method that still finishes, and 2× better than the only other method that finishes at scale. **Speed is the product; quality is the cost.** That is exactly the framing our six-condition table is built to measure.

### 5.6 StratLearner (NeurIPS 2020) — learned solution map vs. off-the-shelf DL

Graphs [verified, §4.1]: Kronecker **1024 / 2655**, Erdős–Rényi **512 / 6638**, power-law **768 / 1532**; plus a SNAP Facebook graph with **4,039 nodes** yes (Appendix E.4). Triggering model with Weibull transmission times, `N_u(S) = 1/d_v`; the protector budget `k` is set to `|M|`, the attacker's size. Metric is the **performance ratio** `f(M, P_pred | ∅) / f(M, P_true | ∅) ∈ [0,1]`, with `f` computed by 10,000 simulations and `P_true` from a best-known approximation.

Row shown is **training size 1080**; StratLearner columns are 100/400/800/1600 random-subgraph features [verified, Table 1]:

| Graph       | StratL-100 | 400   | 800   | 1600      | NB    | MLP   | GCN   | DSPN  | Rand  | HD    | Pro   |
| ----------- | ---------- | ----- | ----- | --------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Kronecker   | 0.708      | 0.760 | 0.782 | **0.817** | 0.658 | 0.632 | 0.657 | 0.650 | 0.190 | 0.639 | 0.670 |
| Power-law   | 0.680      | 0.823 | 0.890 | **0.920** | 0.294 | 0.418 | 0.281 | 0.242 | 0.047 | 0.318 | 0.770 |
| Erdős–Rényi | 0.688      | 0.844 | 0.870 | **0.899** | 0.111 | 0.410 | 0.091 | 0.090 | 0.052 | 0.102 | 0.776 |

Facebook yes [verified, Table 3]: StratL **0.725** · NB 0.662 · MLP 0.651 · GCN 0.625 · DSPN 0.446 · HD 0.656 · Pro 0.170 · Rand 0.011.

**A plain GCN scores 0.091–0.657 depending only on the graph family.** Same architecture, same task, 7× spread in quality. The paper's diagnosis is that "GCN … merely uses the adjacency between nodes without considering the triggering model" [verified, §4.2]. That is a direct argument for our structured head: a backbone that does not encode the diffusion mechanism does not transfer across graph families, which is precisely the failure `ICTransmissionHead` was built to avoid. It also means an unstructured GNN baseline is a fair and informative arm for us to run.

Note also the **proximity** heuristic scoring 0.770/0.776 on power-law and ER — above every learned method except StratLearner — and then collapsing to 0.170 on Facebook. Consistent with §5.4: proximity is the baseline to beat, and it is unstable.

### 5.7 TC-AIBM / BIS (2025) — the three-tie-break study

Datasets [verified, Table 1], all from [KONECT](http://konect.cc/networks):

| Dataset | Nodes   | Edges     | Avg degree | Type       |
| ------- | ------- | --------- | ---------- | ---------- |
| net-sci | 1,461   | 2,742     | 3.79       | Undirected |
| kw-wiki | 8,623   | 160,255   | 37.16      | Undirected |
| marvel  | 25,914  | 96,662    | 7.46       | Undirected |
| amazon  | 400,727 | 3,200,440 | 7.99       | Directed   |

Warning: **`net-sci` 1,461 / 2,742 is a near-collision with our `netscience` 1,589 / 2,742** — identical edge count, 128 fewer nodes. Almost certainly the same Newman 2006 co-authorship graph with isolated/degree-0 nodes dropped, but we did not confirm it against the KONECT file (§11).

Protocol [verified, §5.2]: `k` from 10 to 100; `|S| ∈ {50, 100, 200}`; time constraint `τ ∈ {3,4,5}`; negative seeds by **Degree, IMM, and PageRank**; tie-breaks **Negative / Positive / Fixed Dominance**. Baselines: Degree, Forward (simulate from negative seeds, pick most-frequently-infected), Reverse (RR-set coverage), SSR-PEA, PCMCC, Greedy-B.

Results are published **only as bar charts** (Figs. 2–4), so no per-cell numbers [figure]. The paper's stated outcome: BIS matches Greedy-B, beats every other baseline under all three tie-break rules, except on kw-wiki under positive and fixed dominance where SSR-PEA/PCMCC beat it "by approximately 2%" [verified, §5.3], and BIS is "up to three orders of magnitude faster than the Greedy algorithm" [verified, abstract].

### 5.8 Smaller verified tables

**Tong et al., randomized rumour blocking** — datasets [verified, Table 2]: Power2500 **2.5K / 26K** (avg out-degree 20.8) · Wiki **7K / 30K** (12.0) · Epinions **75K / 508K** (13.4) · **Youtube 1.1M / 6.0M** (5.4). That YouTube row is our `youtube` yes quoted as **arcs** (2 × 2,987,624 ≈ 6.0M) — a clean example of the edges-vs-arcs convention. Results are figure-only ("Number of rumor-activated nodes" plots) [figure].

**Tong & Wu, NeurIPS 2018** — Higgs-10K and Higgs-100K (subgraphs of the SNAP Higgs Twitter activity graph) and **HepPh, 34,546 papers** [verified, §6.1]. Probabilities: activity-proportional on Higgs-10K, uniform `p = 0.1` on Higgs-100K, weighted cascade `1/deg(v)` on HepPh. 5,000 MC per objective call, 10,000 to evaluate the final solution.

**Vertex blocking (ICDE 2023), exact vs. GreedyReplace** [verified, Tables V–VI] — expected spread and runtime (s), tiny instance:

| `b` | TR: Exact | GR     | ratio  | Exact time | GR time | WC: Exact | GR     | ratio  | Exact time | GR time |
| --- | --------- | ------ | ------ | ---------- | ------- | --------- | ------ | ------ | ---------- | ------- |
| 1   | 12.614    | 12.614 | 100%   | 3.07       | 0.12    | 11.185    | 11.185 | 100%   | 2.63       | 0.10    |
| 2   | 12.328    | 12.334 | 99.95% | 130.91     | 0.21    | 11.077    | 11.078 | 99.99% | 110.92     | 0.18    |
| 3   | 12.112    | 12.119 | 99.94% | 3828.2     | 0.25    | 10.997    | 10.998 | 99.99% | 3284.0     | 0.23    |
| 4   | 11.889    | 11.903 | 99.88% | 80050      | 0.33    | 10.922    | 10.925 | 99.97% | 69415      | 0.33    |

Exact blocking of **four** nodes takes 22 hours; a greedy heuristic gets within 0.12% in a third of a second. Blocking is combinatorially brutal and empirically easy — which is why heuristics keep winning (§5.3).

---

## 6. Datasets

Authoritative metadata for the social graphs below lives in [`influence_maximization.md`](influence_maximization.md) §6 (loader contract §6.6, name collisions §6.3). This section records **what the blocking literature evaluates on** and how it differs.

### 6.1 What we already load, and who blocks on it yes

Seven of our fifteen graphs appear in this literature — a **much better overlap than IM gave us** (four). Counts are our loaders' own.

| `--dataset`        | Nodes     | Edges          | Type       | Avg deg | Blocking papers using it                                       |
| ------------------ | --------- | -------------- | ---------- | ------- | -------------------------------------------------------------- |
| `netscience`       | 1,589     | 2,742          | Undirected | 3.45    | TC-AIBM (as `net-sci`, **1,461** nodes — near-collision, §6.3) |
| `email_eu_core` yes | 1,005     | 24,929 arcs    | Directed   | 49.6    | SandIMIN · Xie IJoC'25 · Xie ICDE'23 · **NIE**                 |
| `facebook` yes      | 4,039     | 88,234         | Undirected | 43.7    | SandIMIN · Xie IJoC'25 · Xie ICDE'23 · **StratLearner**        |
| `wiki_vote` yes     | 7,115     | 103,689 arcs   | Directed   | 29.1    | SandIMIN · Xie IJoC'25 · Xie ICDE'23                           |
| `nethept` yes       | 15,229    | 62,752 arcs    | Directed   | 4.1     | **CLDAG (SDM'12)** · RPS (exact match, §6.3)                   |
| `netphy` yes        | 37,154    | 174,161        | Undirected | 9.38    | CLDAG (SDM'12)                                                 |
| `twitter` yes       | 81,306    | 1,768,149 arcs | Directed   | 59.5    | SandIMIN · Xie IJoC'25 · Xie ICDE'23 · DiffIM                  |
| `youtube` yes       | 1,134,890 | 2,987,624      | Undirected | 5.3     | SandIMIN · Xie IJoC'25 · Xie ICDE'23 · Tong INFOCOM'17         |

**Three of these are simulable today** at the sizes blocking papers use (`email_eu_core`, `facebook`, `wiki_vote`), and two more (`nethept`, `netphy`) are in reach. That is enough for a comparable table without adding a single loader — the strongest practical argument for doing this task next.

`jazz`, `cora_ml`, `power_grid`, `ca_grqc`, `lastfm_asia`, `digg`, `weibo` and all five synthetic families appear in **no** blocking paper found here.

### 6.2 Full catalogue

Undirected rows quote undirected edges; directed rows quote arcs, **as the citing paper reports them**. Where a paper's convention is ambiguous it is flagged.

#### Small (< 10K nodes)

| Dataset               | Nodes       | Edges         | Type       | Avg deg  | Download                                                                                                                                         | Used by                                        |
| --------------------- | ----------- | ------------- | ---------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------- |
| power-law graph       | 768         | 1,532         | directed   | 1.99     | generated (`networkx`)                                                                                                                           | NIE, StratLearner                              |
| **email-Eu-core** yes  | 1,005       | 25,571        | directed   | 49.6     | [SNAP](https://snap.stanford.edu/data/email-Eu-core.html) · [`email-Eu-core.txt.gz`](https://snap.stanford.edu/data/email-Eu-core.txt.gz)        | SandIMIN, Xie×2, NIE                           |
| Kronecker             | 1,024       | 2,655         | directed   | 2.6      | [SNAP `krongen`](https://github.com/snap-stanford/snap/tree/master/examples/krongen)                                                             | StratLearner                                   |
| net-sci (KONECT)      | 1,461       | 2,742         | undirected | 3.79     | [KONECT](http://konect.cc/networks)                                                                                                              | TC-AIBM                                        |
| **NetScience** yes     | 1,589       | 2,742         | undirected | 3.45     | [Netzschleuder](https://networks.skewed.de/net/netscience)                                                                                       | (ours)                                         |
| Erdős–Rényi           | 512         | 6,638         | directed   | 25.9     | generated (`networkx`)                                                                                                                           | StratLearner                                   |
| Power2500             | 2,500       | 26,000        | directed   | 20.8 out | not stated in the paper                                                                                                                          | Tong INFOCOM'17                                |
| **ego-Facebook** yes   | 4,039       | 88,234        | undirected | 43.7     | [SNAP](https://snap.stanford.edu/data/ego-Facebook.html) · [`facebook_combined.txt.gz`](https://snap.stanford.edu/data/facebook_combined.txt.gz) | SandIMIN, Xie×2, StratLearner                  |
| Facebook (NetRepo)    | 4,500       | 161,400       | undirected | 141.5    | [networkrepository.com](https://networkrepository.com/)                                                                                          | fair IBM 2026 — **a different Facebook**, §6.3 |
| Extended (ET)         | 5,636→5,413 | 31,826→27,146 | directed   | —        | Ko et al. 2020 via [DiffIM repo](https://github.com/junghunl/DiffIM)                                                                             | DiffIM                                         |
| Gnutella (Simpson)    | 6,300       | 20,800        | directed   | 3.3      | [SNAP p2p-Gnutella08](https://snap.stanford.edu/data/p2p-Gnutella08.html)                                                                        | NAMM                                           |
| p2p-Gnutella08        | 6,301       | 20,777        | directed   | 3.30     | [`p2p-Gnutella08.txt.gz`](https://snap.stanford.edu/data/p2p-Gnutella08.txt.gz)                                                                  | NIE                                            |
| Wiki (Tong)           | 7,000       | 30,000        | directed   | 12.0     | not stated — **not** SNAP wiki-Vote (§6.3)                                                                                                       | Tong INFOCOM'17                                |
| **wiki-Vote** yes      | 7,115       | 103,689       | directed   | 29.1     | [SNAP](https://snap.stanford.edu/data/wiki-Vote.html) · [`wiki-Vote.txt.gz`](https://snap.stanford.edu/data/wiki-Vote.txt.gz)                    | SandIMIN, Xie×2                                |
| Flixster (NAMM)       | 7,600       | 75,700        | undirected | 9.43     | crawl, not redistributed                                                                                                                         | NAMM                                           |
| Celebrity (CL)        | 7,848→7,336 | 28,839→27,699 | directed   | —        | [DiffIM repo](https://github.com/junghunl/DiffIM)                                                                                                | DiffIM                                         |
| Gnutella (proactive)  | 8,800       | 63,000        | directed   | 7.2      | [SNAP p2p-Gnutella](https://snap.stanford.edu/data/)                                                                                             | Proactive rumour control                       |
| kw-wiki               | 8,623       | 160,255       | undirected | 37.16    | [KONECT](http://konect.cc/networks)                                                                                                              | TC-AIBM                                        |
| Wikipedia-ja (Kimura) | 9,481       | 245,044 arcs  | directed   | 25.8     | not redistributed                                                                                                                                | Kimura AAAI'08                                 |

#### Mid-size (10K – 500K nodes)

| Dataset              | Nodes         | Edges          | Type       | Avg deg | Download                                                                                                                                                                     | Used by                                  |
| -------------------- | ------------- | -------------- | ---------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| word_assoc           | 10,617        | 72,172         | directed   | 6.8     | [LAW](http://law.di.unimi.it/datasets.php)                                                                                                                                   | RPS                                      |
| blog (Kimura)        | 12,047        | 79,920 arcs    | directed   | 6.6     | not redistributed                                                                                                                                                            | Kimura AAAI'08                           |
| Flixster (Com-IC)    | 12,900        | 192,000        | directed   | 14.8    | crawl, not redistributed                                                                                                                                                     | Com-IC                                   |
| **NetHEPT** yes       | 15,229        | 62,752 arcs    | directed   | 4.1     | [SparklyYS mirror](https://github.com/SparklyYS/Simultaneous-IMM) · [Wei Chen `weic-graphdata.zip`](https://www.microsoft.com/en-us/research/people/weic/selected-projects/) | CLDAG, RPS                               |
| Mobile (CLDAG)       | 15,500        | 37,000         | directed   | 4.77    | **proprietary — not released**                                                                                                                                               | CLDAG                                    |
| WannaCry (WC)        | 16,246→19,381 | 84,217→85,202  | directed   | —       | [DiffIM repo](https://github.com/junghunl/DiffIM)                                                                                                                            | DiffIM                                   |
| Douban-Book          | 23,300        | 141,000        | directed   | 6.5     | crawl, not redistributed                                                                                                                                                     | Com-IC, NAMM                             |
| marvel               | 25,914        | 96,662         | undirected | 7.46    | [KONECT](http://konect.cc/networks)                                                                                                                                          | TC-AIBM                                  |
| p2p-Gnutella24       | 26,518        | 65,369         | directed   | 2.47    | [`p2p-Gnutella24.txt.gz`](https://snap.stanford.edu/data/p2p-Gnutella24.txt.gz)                                                                                              | NIE                                      |
| cit-HepTh            | 27,770        | 352,807        | directed   | 12.7    | [`cit-HepTh.txt.gz`](https://snap.stanford.edu/data/cit-HepTh.txt.gz)                                                                                                        | DiffIM                                   |
| HepPh (Tong)         | 34,546        | —              | directed   | —       | [SNAP cit-HepPh](https://snap.stanford.edu/data/cit-HepPh.html)                                                                                                              | Tong NeurIPS'18                          |
| Douban-Movie         | 34,900        | 274,000        | directed   | 7.9     | crawl, not redistributed                                                                                                                                                     | Com-IC, NAMM                             |
| Email-Enron          | 36,692        | 183,831        | undirected | 10.0    | [`email-Enron.txt.gz`](https://snap.stanford.edu/data/email-Enron.txt.gz)                                                                                                    | Proactive rumour control                 |
| **NetPHY** yes        | 37,154        | 174,161        | undirected | 9.38    | [Wei Chen `weic-graphdata.zip`](https://www.microsoft.com/en-us/research/people/weic/selected-projects/)                                                                     | CLDAG                                    |
| Last.fm (Com-IC)     | 61,000        | 584,000        | directed   | 9.6     | crawl, not redistributed                                                                                                                                                     | Com-IC                                   |
| Slashdot             | 70,000        | 358,600        | undirected | 20.5    | [`soc-Slashdot0902.txt.gz`](https://snap.stanford.edu/data/soc-Slashdot0902.txt.gz)                                                                                          | fair IBM 2026                            |
| Epinions             | 75,879        | 508,837        | directed   | 13.4    | [SNAP soc-Epinions1](https://snap.stanford.edu/data/soc-Epinions1.html)                                                                                                      | Tong INFOCOM'17                          |
| **Twitter (ego)** yes | 81,306        | 1,768,149 arcs | directed   | 59.5    | [SNAP ego-Twitter](https://snap.stanford.edu/data/ego-Twitter.html)                                                                                                          | SandIMIN, Xie×2, DiffIM                  |
| Gowalla              | 196,591       | 950,327        | undirected | 19.3    | [`loc-gowalla_edges.txt.gz`](https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz)                                                                                        | fair IBM, Proactive                      |
| email-EuAll          | 265,214       | 420,045        | directed   | 3.2     | [`email-EuAll.txt.gz`](https://snap.stanford.edu/data/email-EuAll.txt.gz)                                                                                                    | SandIMIN, Xie×2, DiffIM                  |
| web-Stanford         | 281,903       | 2,312,497      | directed   | 16.4    | [`web-Stanford.txt.gz`](https://snap.stanford.edu/data/web-Stanford.txt.gz)                                                                                                  | SandIMIN, Xie×2, NIE                     |
| com-DBLP             | 317,080       | 1,049,866      | undirected | 6.6     | [`com-dblp.ungraph.txt.gz`](https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz)                                                                      | SandIMIN, Xie×2, NAMM (quotes 2.1M arcs) |
| cnr-2000             | 325,557       | 3,216,152      | directed   | 9.9     | [LAW](http://law.di.unimi.it/datasets.php)                                                                                                                                   | RPS                                      |
| dblp-2010            | 326,186       | 1,615,400      | directed   | 6.1     | [LAW](http://law.di.unimi.it/datasets.php)                                                                                                                                   | RPS                                      |
| amazon (KONECT)      | 400,727       | 3,200,440      | directed   | 7.99    | [KONECT](http://konect.cc/networks)                                                                                                                                          | TC-AIBM                                  |
| Higgs Twitter        | 456,631       | 14,855,875     | directed   | 65.1    | [`higgs-social_network.edgelist.gz`](https://snap.stanford.edu/data/higgs-social_network.edgelist.gz)                                                                        | Tong NeurIPS'18 (10K/100K subgraphs)     |

#### Large (> 500K nodes)

| Dataset        | Nodes     | Edges      | Type       | Avg deg | Download                                                                                                      | Used by                                             |
| -------------- | --------- | ---------- | ---------- | ------- | ------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| **YouTube** yes | 1,134,890 | 2,987,624  | undirected | 5.3     | [`com-youtube.ungraph.txt.gz`](https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz) | SandIMIN, Xie×2, Tong INFOCOM'17 (as 6.0M **arcs**) |
| soc-Pokec      | 1,632,803 | 30,622,564 | directed   | 37.5    | [`soc-pokec-relationships.txt.gz`](https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz)             | SandIMIN, fair IBM                                  |
| ljournal-2008  | 5,363,260 | 79,023,142 | directed   | 28.5    | [LAW](http://law.di.unimi.it/datasets.php)                                                                    | RPS                                                 |

#### Not obtainable

Budak's four Facebook regional snapshots — SB09 **26,455 / 453,132**, SB08 **12,814 / 184,482**, MB09 **14,144 / 186,582**, MB08 **6,117 / 62,750** (bidirectional edges counted twice, i.e. arcs) [verified, WWW'11 §5] — were crawled in 2008–09 and are not released. CLDAG's `Mobile` is proprietary call-detail data. Kimura's blog and Japanese-Wikipedia graphs are not redistributed. **The three founding papers of this literature are all irreproducible on their own data.**

### 6.3 Warning: Name collisions and near-collisions

| Name           | Version A                                               | Version B                                         | Who uses which                                                                                                                                   |
| -------------- | ------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **NetScience** | Newman 2006, **1,589 / 2,742** (ours yes)                | KONECT `net-sci`, **1,461 / 2,742**               | A: us · B: TC-AIBM. Identical edge count, 128 fewer nodes — almost certainly the same graph with degree-0 nodes dropped, **not confirmed** (§11) |
| **Facebook**   | SNAP `ego-Facebook` **4,039 / 88,234** (ours yes)        | NetRepo **4,500 / 161,400**                       | A: SandIMIN, Xie×2, StratLearner · B: fair IBM 2026                                                                                              |
| **Wiki**       | SNAP `wiki-Vote` **7,115 / 103,689** (ours yes)          | Tong's "Wiki" **7,000 / 30,000**                  | A: SandIMIN, Xie×2 · B: Tong INFOCOM'17 — 3.5× fewer edges, a different graph                                                                    |
| **Gnutella**   | `p2p-Gnutella08` **6,301 / 20,777**                     | Proactive's **8,800 / 63,000** (a later snapshot) | A: NIE, NAMM · B: Proactive rumour control                                                                                                       |
| **YouTube**    | **1,134,890 / 2,987,624** undirected edges (ours yes)    | same graph quoted as **6.0M arcs**                | Tong INFOCOM'17 quotes arcs — _not_ a different graph                                                                                            |
| **DBLP**       | `com-DBLP` **317,080 / 1,049,866** undirected           | NAMM's **317K / 2.1M**                            | same graph, NAMM quotes arcs                                                                                                                     |
| **NetHEPT**    | **15,229 / 62,752 arcs** (ours yes, RPS's exact numbers) | CLDAG's **15.2K / 58.9K** raw edge lines          | see [`influence_maximization.md`](influence_maximization.md) §6.4.1 — same file, three counting conventions                                      |

**RPS is the cleanest match in this literature**: its Table 2 reports `nethept 15,229 / 62,752 / avg degree 4.1` [verified] — digit-for-digit our loader's output. Any RPS number is directly comparable to ours without a caveat.

---

## 7. Which paper uses which

Cells mark the dataset **as that paper reports it** — check §6.3 first. Bold rows are graphs we already load.

| Dataset                                           | Kimura'08 | Budak'11 | CLDAG'12 | Com-IC'15 | Tong'17 | RPS'18 | Tong'18 | StratL'20 | NAMM'22 | Xie'23 | NIE'23 | SandIMIN'24 | DiffIM'25 | Xie'25 | TC-AIBM'25 | fairIBM'26 |
| ------------------------------------------------- | --------- | -------- | -------- | --------- | ------- | ------ | ------- | --------- | ------- | ------ | ------ | ----------- | --------- | ------ | ---------- | ---------- |
| **email-Eu-core**                                 |           |          |          |           |         |        |         |           |         | yes      | yes      | yes           |           | yes      |            |            |
| **NetScience / net-sci**                          |           |          |          |           |         |        |         |           |         |        |        |             |           |        | yes          |            |
| **ego-Facebook**                                  |           |          |          |           |         |        |         | yes         |         | yes      |        | yes           |           | yes      |            |            |
| **wiki-Vote**                                     |           |          |          |           |         |        |         |           |         | yes      |        | yes           |           | yes      |            |            |
| **NetHEPT**                                       |           |          | yes        |           |         | yes      |         |           |         |        |        |             |           |        |            |            |
| **NetPHY**                                        |           |          | yes        |           |         |        |         |           |         |        |        |             |           |        |            |            |
| **Twitter (ego)**                                 |           |          |          |           |         |        |         |           |         | yes      |        | yes           | yes         | yes      |            |            |
| **YouTube**                                       |           |          |          |           | yes       |        |         |           |         | yes      |        | yes           |           | yes      |            |            |
| p2p-Gnutella08/24                                 |           |          |          |           |         |        |         |           | yes       |        | yes      |             |           |        |            |            |
| web-Stanford                                      |           |          |          |           |         |        |         |           |         | yes      | yes      | yes           |           | yes      |            |            |
| email-EuAll                                       |           |          |          |           |         |        |         |           |         | yes      |        | yes           | yes         | yes      |            |            |
| com-DBLP                                          |           |          |          |           |         |        |         |           | yes       | yes      |        | yes           |           | yes      |            |            |
| soc-Pokec                                         |           |          |          |           |         |        |         |           |         |        |        | yes           |           |        |            | yes          |
| Epinions                                          |           |          |          |           | yes       |        |         |           |         |        |        |             |           |        |            |            |
| Slashdot / Gowalla                                |           |          |          |           |         |        |         |           |         |        |        |             |           |        |            | yes          |
| Higgs / HepPh                                     |           |          |          |           |         |        | yes       |           |         |        |        |             |           |        |            |            |
| cit-HepTh                                         |           |          |          |           |         |        |         |           |         |        |        |             | yes         |        |            |            |
| Flixster / Douban / Last.fm                       |           |          |          | yes         |         |        |         |           | yes       |        |        |             |           |        |            |            |
| LAW graphs (cnr, dblp-2010, ljournal)             |           |          |          |           |         | yes      |         |           |         |        |        |             |           |        |            |            |
| KONECT (kw-wiki, marvel, amazon)                  |           |          |          |           |         |        |         |           |         |        |        |             |           |        | yes          |            |
| Kronecker / ER / power-law                        |           |          |          |           |         |        |         | yes         |         |        | yes      |             |           |        |            |            |
| WC / CL / ET (Ko et al.)                          |           |          |          |           |         |        |         |           |         |        |        |             | yes         |        |            |            |
| Private (blog, Wikipedia-ja, Mobile, FB-regional) | yes         | yes        | yes        |           |         |        |         |           |         |        |        |             |           |        |            |            |

**Read it as a scoreboard for us:** eight of our fifteen graphs appear somewhere, and `email-Eu-core`, `ego-Facebook`, `wiki-Vote`, `web-Stanford`, `email-EuAll`, `com-DBLP`, `Twitter`, `YouTube` form a _consistent_ suite used by the Xie/SandIMIN node-blocking line across four papers. Running our three simulable ones (`email_eu_core`, `facebook`, `wiki_vote`) against §5.3 gives a paper-ready table with zero new loaders.

There is **no dataset used by both the seeding line and the structural line** other than the SNAP staples, and **no paper in either line reports the other's metric.** The two halves of this literature do not talk to each other, which is an opening: a single evaluator that scores `add_node`, `remove_node` and `remove_edge` interventions on the same graph under the same metric does not currently exist.

---

## 8. Evaluation protocol

### 8.1 The metric — prevented / blocked influence

One quantity, five names. All are `σ(negative alone) − σ(negative | blockers)`:

| Name                                        | Used by       | Note                                                            |
| ------------------------------------------- | ------------- | --------------------------------------------------------------- |
| **blocked influence** `f(S_t \| S_f)`       | NIE           |                                                                 |
| **negative influence reduction** `σ_NIR(S)` | CLDAG         | expectation over both threshold vectors                         |
| **π(A_L)**, "saved" nodes                   | Budak         | explicitly _only_ nodes that would otherwise have been infected |
| **decreased spread** `D_S(B)`               | SandIMIN, Xie | node-blocking line                                              |
| **prevented influence**                     | RPS, NAMM     |                                                                 |

Budak's framing is the sharp one and worth copying verbatim into our metric docstring [verified, WWW'11 §4]: _"Note that we are not necessarily interested in the number of inoculated nodes but the inoculated nodes that would be infected otherwise. We will refer to this set of nodes as **saved**."_ A blocker that protects nodes the rumour would never have reached scores zero.

Warning: **Trap: the same symbol `σ` means three things.** CLDAG's `σ_N(S, N₀)` is the _remaining_ negative spread (lower is better); its `σ_NIR(S)` is the _reduction_ (higher is better); the competitive-IM line's `σ_A(S_A, S_B)` is the _positive_ spread (higher is better, and is **not** the same objective — maximizing your own cascade is not minimizing theirs). Never compare across these without checking.

Secondary metrics in use: **runtime to a target quality** (NIE Table IV), **quality within a time budget** (NIE Table V), **performance ratio vs. a best-known solution** (StratLearner), and **number of positive seeds needed to reach a 10% residual** (CLDAG Table 2).

### 8.2 Budgets — the convention clash is worse than in IM

| Convention                            | Range                           | Used by                       |
| ------------------------------------- | ------------------------------- | ----------------------------- |
| Absolute `k` blockers                 | `k ∈ {10,…,50}` or `{10,…,100}` | SandIMIN, Xie×2, TC-AIBM      |
| `k` tied to the attacker, `k = \|M\|` | —                               | StratLearner                  |
| Ratio `\|S_P\| / \|S_N\|`             | up to 30×                       | CLDAG                         |
| Wall-clock budget (1 min, 48 h)       | —                               | NIE                           |
| Percent of `N`                        | 1/5/10/20%                      | **nobody** in this literature |

**Our `--budget-pcts 1 5 10 20` speaks to no blocking paper at all.** Use `--budgets 10 20 30 40 50` here. And per §5.4, the _informative_ axis is `|S_P| / |S_N|`, not `|S_P| / |V|`: at `|S_N| = 1000` on NetHEPT even 1,000 blockers only remove 17% of the negative spread, so a percentage budget silently lands you in the regime where every method ties at "barely anything works".

### 8.3 The attacker — a second experimental axis IM does not have

Every blocking result is conditioned on **how `S_N` was chosen**, and papers disagree:

| Paper        | `S_N` selection                                         | `\|S_N\|`           |
| ------------ | ------------------------------------------------------- | ------------------- |
| SandIMIN     | random from the **top-200 most influential** nodes      | 10 (default), 10–50 |
| TC-AIBM      | **Degree, IMM, and PageRank** (all three reported)      | 50, 100, 200        |
| CLDAG        | largest degree, and random                              | 1 → 1000 (swept)    |
| NIE          | random from the top-`ρ` out-degree set `V*`             | sampled             |
| StratLearner | size ~ power-law(2.5), members uniform at random        | variable            |
| Budak        | uniformly at random, plus a high-degree-adversary study | 1 (single source)   |

Our six spine selectors in `data/wm_actions.py` (`random`, `degree`, `pagerank`, `betweenness`, `celf`, `local_search`) already cover the first four of these verbatim — the attacker model is a **free reuse**, not new code (§2.5).

Budak adds a third axis, **detection delay `r`**: the bad campaign is detected `r` steps late, and the blocker only acts from then. That maps exactly onto our `--inject-p` / mid-rollout injection machinery and is the natural way to make the task non-trivial. CLDAG's "first mover has a clear advantage" is the same statement.

### 8.4 Tie-breaking — must be reported, is usually not

Three rules, named consistently only by TC-AIBM [verified, §2.2]: **positive dominance (PD)**, **negative dominance (ND)**, **fixed dominance (FD)** by a predefined priority order. Under the live-edge characterisation a node `w` is saved iff [verified, TC-AIBM Lemma 1]:

```
PD:  d_L(v, w) ≤ d_L(n_s, w)
ND:  d_L(v, w) <  d_L(n_s, w)
FD:  ≤ if γ(v) > γ(n_s), else <
```

— i.e. the rules differ _only_ in how ties in shortest-path distance resolve. That is a one-line change in a simulator and a large change in the reported numbers, and most papers never state which they used. **CIC and CLT hard-code ND; Budak's MCICM and COICM hard-code PD** ("if the 'bad information' and the 'good information' reach a node `w` at the same step, 'good information' takes effect") [verified, WWW'11 §3.1]; Borodin's separated-threshold model uses a **uniform coin flip**; Com-IC proves the rule is irrelevant in the mutually complementary case [verified, Lemma 2]. Record it explicitly.

### 8.5 Simulation counts and edge probabilities

Weighted cascade `p(u,v) = 1/in-degree(v)` is the default in the node-blocking line (SandIMIN, Xie) and matches our `--prob-model weighted`. Trivalency `{0.1, 0.01, 0.001}` appears alongside it in Xie IJoC'25. Uniform `p = 0.1` appears in Tong NeurIPS'18. CLDAG instead normalises **per-cascade** edge weights so that each node's in-weights sum to `p⁺` (resp. `p⁻`), giving a per-cascade "strength" knob — that is the CLT analogue of our `set_edge_weight`.

MC counts: SandIMIN **10⁵** simulations for the final spread estimate; CLDAG **10,000** per influence estimate inside greedy; Tong NeurIPS'18 **5,000** inside the objective and **10,000** to score the final solution; StratLearner **10,000**; NAMM **20,000**. Our `--mc-marginals 30` is for _one-step_ targets, which is a different thing — but the ground-truth referee in `pipeline/conditions.py` should be at the 10⁴ end to be comparable.

---

## 9. Implications for this project

### 9.1 Build order

yes **BUILT, ALL PHASES, 2026-08-04.** `influence_blocking` is `implemented` in `pipeline/tasks.py` and runs end to end through the same five stages as every other task. What shipped, against what this section proposed:

| Proposed here | Shipped as | Note |
| ------------- | ---------- | ---- |
| `CompetitiveSimulator` alongside `Simulator` | `data/wm_competitive.py` | Its own module rather than inside `wm_simulator.py`; synchronous step, all three tie-breaks, COICM/MCICM via `--positive-prob` |
| 8 channels, 4 targets | `wm_data.build_competitive_features`, `channels_for()` | Additive: the single-cascade layout is byte-identical, and `dataset_is_competitive()` reads the choice back off the data |
| `CompetitiveICHead` | `CompetitiveICHead` **and** `CompetitiveLTHead` | §9.1 Phase 2 said defer CLT; it shipped anyway, because the registry declares both dynamics and `--diffusion-model LT` otherwise has no head at all. Expect it to be the weaker demonstration for exactly the reason §2.2 gives |
| 4 BCE terms | one term per target column | Generalized rather than special-cased, so 2 and 4 are one loop |
| `blocked_influence` + planning regret | `blocking_metrics`, `wm_eval.blocking_regret` | Regret is measured against **prevented** influence and reports `proximity` as a comparison baseline, not just degree and random |
| Phase 0 (single-cascade node blocking) | the `node_block` LEVER of the full task | Subsumed rather than built separately: it is the same simulator with `budget_op = remove_node`, so §5.3's table is reproducible without a second code path |

**§2.2's COICM caveat is resolved, not inherited.** That section warns the two-MLP product form "silently assumes MCICM, not COICM". The warning is right about the live-edge characterisation and does not bite on the stepwise transition: a node activates in at most ONE campaign, so each arc is ever attempted by exactly one of them, and the two arrival probabilities at a susceptible `v` are products over disjoint in-edge sets. The factorization is therefore exact under both models, and what actually separates them in a forward simulation is only whether `p_positive == p_negative`. `check_influence_blocking.check_head_matches_simulator` holds the oracle head to within 0.03 of 4,000 simulator draws under both dominance rules, so this is measured rather than argued.

**Phase 0 — node-blocking IMIN. Almost free, and it produces a comparable table.** `remove_node` under single-cascade IC needs **no** two-cascade machinery: the "negative" cascade is just our existing cascade, and the intervention is deleting nodes. §5.3 is then directly reproducible on `email_eu_core` yes, `facebook` yes and `wiki_vote` yes at `k = 10…50` with weighted-cascade probabilities. yes **Shipped as `--blocking-lever node_block`.**

One correctness bug to fix first. yes **Done.** `Simulator.apply_actions` used to implement `remove_node` for IC as `status = 2` (NDlib _Removed_) while `active_nodes()` returned `status in (1, 2)`, so a "removed" node still counted as infected: right for "spent spreader", wrong for "blocked". Fixed as this section proposed, down to the flag name: `--remove-semantics spent|blocked`, with a `blocked: set[int]` on the simulator excluded from `active_nodes()`. Transmission is stopped structurally instead of by a special case in `advance`, because node deletion is emitted as `remove_node(v)` plus a `remove_edge` per incident arc (`wm_actions.delete_node_bag`), which also makes the feature builder and the adjacency replay correct with no changes. `Simulator._enforce_blocked()` backstops a bare `remove_node`. This task's registry entry defaults to `blocked`.

**Phase 1 — competitive IC, the real task.** `CompetitiveSimulator`, 8 feature channels, 4 targets, `CompetitiveICHead` with an explicit tie-break parameter. §2.5 costs it at two to three days. Pick **COICM** (one shared probability per edge) if you want Budak's submodularity to hold and a greedy oracle with a guarantee; pick **MCICM** if you want the two-MLP head to be exact. You cannot have both — that is the §5.1 fault line, and it is a decision to make before writing the simulator, not after.

**Phase 2 — CLT.** Defer. Two hidden thresholds per node compounds the partial-observability tax `LTThresholdHead` already pays (§2.2).

### 9.2 What it buys us

1. **It exercises three-quarters of the action space.** `remove_node`, `remove_edge` and `set_edge_weight` are implemented, tested and idle. This is the task that makes them load-bearing, and it is the strongest available answer to "why five ops?".
2. **A non-submodular objective is where a learned planner has room to win.** In IM, greedy is `(1−1/e)`-optimal and every method ties at high budget ([`influence_maximization.md`](influence_maximization.md) §5.1). Here, greedy has _no guarantee_ under most models (§5.1), exact solutions are hopeless (§5.8: 22 hours for `b = 4`), and a trivial heuristic beats a VLDB algorithm in 6 of 30 cells (§5.3). That is a genuinely open scoreboard.
3. **Eight of our graphs are already in the literature, three are simulable today, and RPS's NetHEPT row matches ours digit-for-digit** (§6.3). No new loaders.
4. **The seeding line and the structural line share no metric.** A world model that scores `add_node`, `remove_node` and `remove_edge` interventions on one graph under one metric is a contribution the field does not have (§7).
5. **A published speed/quality bar to hit.** NIE: ~10⁴× faster than MC at ~12% quality cost on `email-Eu-core` yes (§5.5). That is what our world-model evaluator arm has to beat or match.

### 9.3 What will bite

- **Degree is useless here and proximity is strong** — the reverse of IM (§5.4). Add a `proximity` spine selector before running anything, or the baseline table is a strawman.
- **Blocking is a low-signal regime.** Blocking 50 YouTube nodes prevents 0.29% of the spread (§5.3). Positive labels get rarer, so the class imbalance our `--pos-weight off` guidance was tuned against gets worse. Re-check calibration (`brier_*`) rather than assuming it carries over.
- **Percentage budgets are the wrong axis** (§8.2). Switch to absolute `k` and report the `|S_P| / |S_N|` ratio.
- **The tie-break is a reported hyperparameter**, not an implementation detail (§8.4). Bake it into the episode metadata in `graphs_index.json`.
- **Heuristics winning outright is the normal outcome.** Plan the experiment so that a heuristic win is informative rather than embarrassing: sweep `|S_N|` small (§5.4) where the problem is actually solvable and methods separate.

---

## 10. Reference list

**Foundational models and hardness** [Bharathi, Kempe & Salek, WINE 2007](https://link.springer.com/chapter/10.1007/978-3-540-77105-0_31) · [Carnes et al., ICEC 2007](https://doi.org/10.1145/1282100.1282167) Warning: · [Borodin, Filmus & Oren, WINE 2010](http://www.cs.toronto.edu/~oren/cs_toronto/Publications_files/doc.pdf) · [Springer](https://link.springer.com/chapter/10.1007/978-3-642-17572-5_48) · [Budak, Agrawal & El Abbadi, WWW 2011](https://archives.iw3c2.org/www2011/proceedings/proceedings/p665.pdf) · [ACM](https://doi.org/10.1145/1963405.1963499) Warning: · [He, Song, Chen & Jiang, SDM 2012 (CLDAG)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/weic-sdm12_infblockingmax.pdf) · [arXiv 1110.4723](https://arxiv.org/abs/1110.4723) · [Lu, Chen & Lakshmanan, VLDB 2015 (Com-IC), arXiv 1507.00317](https://arxiv.org/abs/1507.00317) · [Tong & Wu, NeurIPS 2018, arXiv 1809.06486](https://arxiv.org/abs/1809.06486)

**Counter-seeding algorithms** [Nguyen et al., WebSci 2012](https://doi.org/10.1145/2380718.2380746) Warning: · [Fan et al., ICDCS 2013](https://doi.org/10.1109/ICDCS.2013.34) Warning: · [Wang et al., ICDCS 2013](https://doi.org/10.1109/ICDCS.2013.37) Warning: · [Wu & Pan, Computer Networks 2017 (CMIA-H / CMIA-O)](https://doi.org/10.1016/j.comnet.2017.05.004) · [Tong et al., INFOCOM 2017, arXiv 1701.02368](https://arxiv.org/abs/1701.02368) · [Tong et al., distributed rumour blocking, arXiv 1711.07412](https://arxiv.org/abs/1711.07412) · [Simpson et al., RPS, arXiv 1807.01162](https://arxiv.org/abs/1807.01162) · [Simpson, Hashemi & Lakshmanan, NAMM, arXiv 2206.11419](https://arxiv.org/abs/2206.11419) · [Xu, Peng & Wang, proactive rumour control, arXiv 2303.10068](https://arxiv.org/abs/2303.10068) · [Erd, Vignatti & da Silva, ASONAM 2020](https://web.ntpu.edu.tw/~myday/doc/ASONAM2020/ASONAM2020_Proceedings/pdf/papers/037_015_232.pdf) · [Shi et al., TC-AIBM, arXiv 2511.16068](https://arxiv.org/abs/2511.16068) · [Fang et al., fair IBM, arXiv 2601.22584](https://arxiv.org/abs/2601.22584) · [Cost-effective rumour containment, arXiv 1403.6315](https://arxiv.org/abs/1403.6315) · [Algorithmic design for CIM, arXiv 1410.8664](https://arxiv.org/abs/1410.8664)

**Structural blocking** [Kimura, Saito & Motoda, AAAI 2008](https://cdn.aaai.org/AAAI/2008/AAAI08-186.pdf) · [TKDD 2009 journal version](https://doi.org/10.1145/1514888.1514892) Warning: · [Khalil, Dilkina & Song, KDD 2014](https://doi.org/10.1145/2623330.2623704) Warning: · [Xie et al., ICDE 2023, arXiv 2302.13529](https://arxiv.org/abs/2302.13529) · [Wang et al., PVLDB 2024 (SandIMIN), arXiv 2405.12871](https://arxiv.org/abs/2405.12871) · [code](https://github.com/wjh0116/IMIN) · [Xie et al., INFORMS J. Computing 2025, arXiv 2312.17488](https://arxiv.org/abs/2312.17488) · [code](https://github.com/INFORMSJoC/2024.0591)

**Learning-based** [StratLearner, NeurIPS 2020, arXiv 2009.14337](https://arxiv.org/abs/2009.14337) · [code](https://github.com/cdslabamotong/stratLearner) · [NIE, arXiv 2308.14012](https://arxiv.org/abs/2308.14012) · [DiffIM, AAAI 2025, arXiv 2502.01031](https://arxiv.org/abs/2502.01031) · [code](https://github.com/junghunl/DiffIM) · [Jiang et al., DRL rumour influence minimization, Applied Intelligence 2023](https://link.springer.com/content/pdf/10.1007/s10489-023-04555-y.pdf) · [OCIM, arXiv 2006.13411](https://arxiv.org/abs/2006.13411) · [Uncertainty-aware competitive IM (DRL), arXiv 2504.15131](https://arxiv.org/abs/2504.15131) · [earlier, arXiv 2404.18826](https://arxiv.org/abs/2404.18826) · [JCCIM, arXiv 2302.09620](https://arxiv.org/abs/2302.09620)

**Survey** Chen, Jiang, Chen et al., _Influence blocking maximization on networks: models, methods and applications_, **Physics Reports 976 (2022) 1–54** — [author PDF](https://www.cse.wustl.edu/~yixin.chen/public/survey.pdf) · [ScienceDirect](https://doi.org/10.1016/j.physrep.2022.05.003) Warning:

**Data sources** [SNAP](https://snap.stanford.edu/data/) · [KONECT](http://konect.cc/networks/) · [LAW](http://law.di.unimi.it/datasets.php) — Warning: host timed out from our network on 2026-07-29; retry or use the Wayback Machine · [Network Repository](https://networkrepository.com/)

Warning: = publisher landing page; 403 to automated clients (ACM DL, SIAM, ScienceDirect) or 202 (IEEE Xplore). Opens normally in a browser.

---

## 11. Open gaps

Honest list of what this review could **not** establish.

**Papers behind paywalls — no result cell transcribed**

- **Carnes et al. (ICEC 2007)** — the Wave Propagation and Distance-Based models are named in every survey, but the ACM page 403s and no open PDF was found. The submodularity claims here are second-hand via Borodin's Thm 5.2, which _disproves_ their conjecture; their own positive results are **[claim]**.
- **Wu & Pan (Computer Networks 2017)** — CMIA-H / CMIA-O are the standard scalable IBM baseline and are still benchmarked in 2023 (NIE). No open PDF; **no dataset table, no result table, no code**. This is the largest single hole in §5.
- **Khalil, Dilkina & Song (KDD 2014)** — the edge-addition-submodular / edge-deletion-supermodular result is [claim] only. It matters because it is the only theory result found for the `add_edge` lever.
- **Nguyen et al. (WebSci 2012)**, **Fan et al. (ICDCS 2013)**, **Wang et al. (ICDCS 2013)** — cited everywhere, all paywalled, nothing transcribed.
- **Kimura & Saito TKDD 2009** — the AAAI 2008 version was transcribed (datasets, method); the journal extension's tables were not.

**Figure-only results**

- **TC-AIBM (§5.7)** publishes every effectiveness result as bar charts. The three-tie-break comparison — the most directly relevant experiment in the whole file for our §2.2 head design — has **no numeric table**.
- **Tong et al. INFOCOM 2017** rumour-blocking results are figure-only.
- **Kimura AAAI 2008** results are figure-only.
- **CLDAG's** main comparison (Figs. 2–5) is figure-only; only Table 2 (§5.4) is numeric.

**Dataset provenance**

- **`net-sci` 1,461 / 2,742 vs our `netscience` 1,589 / 2,742** — identical edge count, 128 fewer nodes. The degree-0-node hypothesis was **not** tested against the KONECT file. Until it is, TC-AIBM's net-sci column is not comparable to ours.
- **Tong's "Wiki" 7K / 30K** — source not stated in the paper; it is not SNAP `wiki-Vote` (103,689 arcs). Unidentified.
- **Power2500 (2.5K / 26K)** — generator and parameters not stated.
- **The three founding papers are irreproducible on their own data**: Budak's four Facebook regional snapshots, CLDAG's Mobile call graph, and Kimura's blog and Japanese-Wikipedia graphs were never released (§6.2).
- **`WC` / `CL` / `ET`** (DiffIM) come from Ko et al. 2020 and were not traced to a primary download URL beyond the DiffIM repo.

**Code**

- **No public code found** for NIE, CMIA-H/CMIA-O, CLDAG, the DRL rumour- minimization paper, OCIM, JCCIM, or the uncertainty-aware competitive IM line — after searching GitHub for each. The three repos in this file (StratLearner, DiffIM, SandIMIN + the INFORMS artifact) are the entire reproducible surface of this literature.
- **RPS's printed code link is `github.com/stamps`**, which is not a resolvable repository. Recorded as printed, not verified as working.

**Wiring status of the four repos [verified 2026-08-04, by cloning and running each]**

| Repo | Status | What it took |
| ---- | ------ | ------------ |
| **SandIMIN** (`wjh0116/IMIN`) | yes wired, runs | 4 patches. Its `rdtsc` inline asm is x86-only (dead code — every real timing is `std::chrono`), and **it throws its own blocker set away**: it prints only `(influence, influence-after, decrease, time)` and its `OutputSeedSetToFile` call is commented out. Its shipped `el2bin` is a prebuilt x86 binary with no source, so the adapter writes the packed `(int, int, double)` `graph_ic.inf` directly, verified against `graph.h::readGraph`'s own mmap stride |
| **Xie IJoC** (`INFORMSJoC/2024.0591`) | yes wired, runs (both AdvancedGreedy and GreedyReplace) | 6 patches. `bits/stdc++.h` is GCC-only; **both binaries draw their own rumour from `mt19937 rand_num(20220708)`**, so unpatched they would answer a different rumour than every arm they are compared against; and neither writes the blocked set |
| **DiffIM** (`junghunl/DiffIM`) | yes wired, runs | The repo is **notebooks only** — no `.py` anywhere — and its cross-imports need `algorithms/` to be a package it is not, so the adapter execs the code cells into one namespace. Its own `requirements.txt` pins `torch-scatter==2.1.0+pt112cu113` and `torch-sparse==0.6.16+pt112cu113`, CUDA-11.3 wheels that are not on PyPI, so the spec installs a trimmed set instead — including `optuna`, which only `train.ipynb::hparam_tuning` uses and we never call, but whose top-level import still has to resolve for the runner to reach `train`. It is also the one repo here whose intervention is an ARC, so it carries `returns_edges=True`: without it the budget check counts endpoints and rejects a legal k-arc answer as 2k seeds. `DIFFIM_ALG` selects the member; the paper's own `DiffIM+` trains a surrogate on our graph first, and its `BPM` / `RIS` / `MDS` / `KED` / `greedy` baselines need no model at all |
| **StratLearner** (`cdslabamotong/stratLearner`) | Warning: `blocked`, verified reason | Running it on our graphs is a re-derivation, not an adapter. Its `data/` is a separate download (README says so; the URL 301s); each feature needs a full pairwise **distance matrix** (`DiffusionGraph.__init__` reads `<i>_distance.txt` per feature), i.e. `O(F·N²)` numbers on disk, which is why its own graphs are 512-1024 nodes; and training needs **2,500 labelled attacker/protector pairs whose protector is a best-known approximation** — solving the blocking problem near-optimally 2,500 times is the prerequisite for running the method being benchmarked. Its Table 1 (§5.6) stays usable as a reference without running it |

**Theory not chased down**

- Whether **COICM submodularity survives a detection delay `r > 0`** — Budak proves it for the delayed setting, but the interaction with _multiple_ negative seeds (`|A_C| > 1`) is stated as "the proofs can easily be generalized" and was not checked.
- Whether the **two-MLP factorization in §2.2 is exactly right under MCICM** in the presence of edge actions that change `w` mid-episode. Our `reconstruct_episode_adjacency` replays edge ops per step, but with two cascades there are two weight vectors to replay and only one graph store.
- **No paper found learns an action-conditioned competitive transition kernel.** If one exists it was not surfaced by the arXiv full-text sweep, OpenAlex search, or the 2022 Physics Reports survey's reference list.

**Metric**

- No blocking paper reports a **one-step** transition metric of any kind, so there is **no published precedent for `delta_f1` / `new_infection_f1` on a competitive cascade**. Our one-step numbers on this task will have to be self-contained; only the rollout and planning numbers are comparable to the literature.
