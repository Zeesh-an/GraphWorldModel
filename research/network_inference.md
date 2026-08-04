# Diffusion Network Inference — Prior Work, Datasets, and Published Results

Recovering a **latent influence network** from nothing but activation traces: given cascades `{(node, infection time)}` and no adjacency at all, infer which edges exist and how fast each one transmits. This is the _inverse_ of everything else in this folder — here `G` is the unknown, not the condition. Two sub-tasks travel together and are often conflated: **edge existence** (a discrete recovery problem, scored with precision/recall/F1/AUC/break-even) and **edge rate or weight** (a continuous estimation problem, scored with MAE/MSE/KL). The headline theoretical result of the field is a **sample complexity**: how many cascades you need before recovery is possible at all.

All URLs in this file returned the stated HTTP code on **2026-07-28**.

---

## 0. Verification policy

| Tier           | Meaning                                                                                                              |
| -------------- | -------------------------------------------------------------------------------------------------------------------- |
| **[verified]** | Read from the paper's own table via text extraction, or from the data repository's own statistics page. Trustworthy. |
| **[derived]**  | Computed by us from a file we downloaded, or reconstructed arithmetically from published splits.                     |
| **[figure]**   | Read off a plotted figure — the paper published no table. Approximate, direction only.                               |
| **[claim]**    | Stated in prose by a paper or a secondary source; not cross-checked against a file or table.                         |

Every number in this file was extracted with `pdftotext -layout` from the PDF and read out of the paper's own table. **No number here came from a WebFetch summary or an automated summarizer.** This literature is unusually prone to that failure mode because most of its results are published _only as precision-recall figures_ — there are far fewer real tables here than in the IM literature, so the temptation to let a summarizer "read" a plot is high. Where a paper published no table, the row is tagged **[figure]** and states direction only.

**Edge-count convention.** Undirected graphs are quoted as _undirected edges_; directed graphs as _arcs_. Our loaders report `adjacency.nnz`, which for a symmetrized undirected graph is **2× the undirected edge count**. Synthetic Kronecker benchmarks in this literature are **directed**, so their quoted edge counts are arcs.

---

## 1. Task definition

**Given.** A set of cascades `C = {c_1, …, c_n}`. Each cascade is an `N`-vector of infection times `t^c = (t^c_1, …, t^c_N)`, where `t^c_i ∈ [0, T]` is when node `i` was activated in cascade `c`, and `t^c_i = ∞` for nodes never activated inside the observation window `T`. **We observe _when_, never _from whom_.**

**Recover.** The directed edge set `E*` and, in the continuous-time variants, the per-edge transmission rate `α*_{j→i}` (or probability `A*_{ji}`).

**The generative model** that almost every method assumes is the _continuous-time independent cascade_ (CIC): once `j` is infected at `t_j`, it draws a transmission delay from a pairwise transmission function `f(t_i | t_j; α_{ji})`, and `i` takes the _minimum_ over all its infected parents. The three standard parametric choices are:

| Model           | Transmission density `f(t_i \| t_j; α)` | Hazard `H(t_i \| t_j; α)` |
| --------------- | --------------------------------------- | ------------------------- |
| **Exponential** | `α · e^{−α(t_i − t_j)}`                 | `α` (constant)            |
| **Power-law**   | `(α/δ) · ((t_i − t_j)/δ)^{−1−α}`        | `α / (t_i − t_j)`         |
| **Rayleigh**    | `α(t_i − t_j) · e^{−α(t_i − t_j)²/2}`   | `α(t_i − t_j)`            |

[verified, NETRATE Table 1 / InfoPath Tables 1–2]

**Why the problem is hard, in one sentence:** the likelihood of a cascade must sum over every possible propagation tree consistent with the observed times, and the number of such trees is super-exponential in the cascade length. The whole methodological history of this field is different ways to dodge that sum — take only the _most likely_ tree (NetInf), take _all_ trees via Kirchhoff's matrix-tree theorem (MultiTree), or reformulate so the sum has a closed form (NETRATE, ConNIe).

### 1.1 Variants

| Variant                            | What changes                                                                 | Representative work                                    |
| ---------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------ |
| **Structure only**                 | Recover `E*`, assume one shared transmission rate                            | NetInf, First-Edge                                     |
| **Structure + rates**              | Recover `E*` and per-edge `α*_{ji}` jointly                                  | NETRATE, ConNIe, MultiTree                             |
| **Non-parametric rates**           | Do not assume exponential/power-law/Rayleigh; learn `f` itself               | KernelCascade                                          |
| **Time-varying network**           | `α_{ji}(t)` evolves; infer one network per time step                         | InfoPath                                               |
| **Discrete-time IC probabilities** | Learn `p(u→v)` from discrete action logs, no continuous clock                | Saito EM, Goyal et al.                                 |
| **Point-process / Hawkes**         | Model recurrent events, not one-shot infection; infectivity matrix = network | Zhou-Zha-Song, Linderman-Adams                         |
| **Noisy / partial traces**         | Infection times corrupted, or only the infected _set_ observed               | Hoffmann-Caramanis, Braunstein et al.                  |
| **Sample complexity**              | Prove how many cascades suffice / are necessary                              | Netrapalli-Sanghavi, Abrahao et al., Daneshmand et al. |

### 1.2 The evaluation loop this field actually runs

Because ground truth is unobservable on real cascades, essentially every paper in §3–§4 does the same thing: **generate a synthetic network with known `E*` (Kronecker or Forest Fire), simulate cascades on it, infer, and score against `E*`.** Real MemeTracker experiments then substitute the _hyperlink graph_ as a proxy ground truth. Both protocols are catalogued in §8, and the Kronecker parameter matrices — the reproducible core of the benchmark — are in §6.3.

---

## 2. Fit with our methodology

**Verdict: Warning: moderate, and the reason is architectural, not incidental.** This task inverts `G`'s role. Everywhere else in this folder `G` is a _condition_ the model reads; here `G` is the _variable being solved for_. That is a real cost and it is worth being precise about exactly where our existing machinery carries us and exactly where it stops.

### 2.1 What our structured IC head already gives us — for free

`world_model/wm_model.py::ICTransmissionHead` predicts a **per-edge** transmission propensity and derives the state update from it:

```
q_uv     = sigmoid(MLP([h_u, h_v, w_uv]))
p_new(v) = 1 − Π_{u→v} (1 − q_uv · frontier_u)
```

Three consequences matter here:

1. **We already have a per-edge parameter with the right semantics.** `q_uv` is exactly the `A_{uv}` that ConNIe estimates and the discretisation of the `α_{uv}` that NETRATE estimates. Our `structured_oracle` variant sets `q = edge_weight` (the true IC probability) and validates that the structural form is correct — which is the same sanity check ConNIe's MSE experiment runs.
2. **`structured_residual` is literally a rate-refinement estimator.** `q = sigmoid(logit(w) + MLP([h_u, h_v, w]))` anchors on the current edge weight and learns a correction. Zero correction = the oracle. Point a likelihood at it and it becomes a _transmission-rate estimator_ on a known support — the ConNIe/NETRATE weight sub-problem, restricted to edges we already believe in.
3. **`GraphInput.edge_weight` is a plain `torch.Tensor` `(E,)`.** Nothing stops `edge_weight.requires_grad_(True)` and taking `∂loss/∂edge_weight`. The BCE loss flows through `torch.log1p(-gated)` → `scatter_add_` → `p_new`, all differentiable. **Continuous edge-weight recovery is a solved plumbing problem for us.**

### 2.2 Where it stops — edge _existence_ is discrete

Everything above optimises weights **on a support that `edge_index` fixes in advance**. `edge_index` is an integer tensor; there is no gradient into it. To recover _which_ edges exist we would need one of:

| Route                                                                                                       | What it costs                                                                                      | Precedent                                 |
| ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| **Dense `N×N` relaxation** — score all pairs, threshold at `λ*`                                             | `O(N²)` memory. Fine at `N ≤ 2,048` (the entire synthetic benchmark), impossible at `netphy` scale | FIM does exactly this (`λ* = 0.01`, §5.6) |
| **ℓ1-regularised MLE + soft-thresholding** — keep the dense parameter, sparsify with a prox operator        | A new loss (survival likelihood, not BCE) and a prox step in the optimiser                         | Daneshmand et al. (§3), ConNIe            |
| **Greedy submodular edge addition** — add the edge with the highest marginal log-likelihood gain, `k` times | Not gradient-based at all; a different program from our training loop                              | NetInf, MultiTree                         |
| **Gumbel/concrete edge sampler** — a learned Bernoulli per pair, reparameterised                            | Genuinely new machinery, and still `O(N²)`                                                         | GSL literature (§4.5)                     |

**Honest summary: our head buys us the rate half of the problem and none of the existence half.** The rate half is the half the theory papers call "much harder" (NETRATE: "estimating transmission rates is considerably harder than simply discovering edges" — §5.2), so this is not a trivial share. But claiming the task without an existence mechanism would be claiming half of it.

### 2.3 The action-op collision — `add_edge` as _search_, not _intervention_

Our five ops include `add_edge`, `remove_edge`, `set_edge_weight`. That is **exactly this task's action vocabulary**, and the coincidence is tempting. It is also the one place to be careful:

- In IM/blocking/CND, an edge op is an **intervention**: the world changes, and `s_{t+1}` changes because of it. `T_exo` is real.
- In network inference, an edge op is a **search move**: the world is fixed and unknown, and we are editing our _hypothesis_ about it. Nothing about the observed cascade changes when we propose an edge.

A coding agent proposing `add_edge(u, v, w)` moves is therefore doing _hypothesis search over `G`_, scored by cascade likelihood — not planning. The `(operator, arguments)` shape transfers; the semantics of `T_exo` do not. Worth stating explicitly in any writeup, because a reader who sees the same five ops will assume the former.

### 2.4 Metrics we would have to add

Nothing in `wm_eval.py` scores an edge set. The field's metric suite is:

| Metric                      | Definition                                                                                                 | Who reports it                  |
| --------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------- |
| **Precision / Recall / F1** | over inferred vs true edge sets                                                                            | everyone                        |
| **Break-even point (BEP)**  | precision at the sweep point where precision = recall                                                      | NetInf, ConNIe                  |
| **AUC**                     | area under the PR or ROC curve as `k` sweeps                                                               | NetInf (Table II), MultiTree    |
| **Accuracy**                | `1 − Σ\|I(α*) − I(α̂)\| / (Σ I(α*) + Σ I(α̂))` — a symmetric-difference score, _not_ classification accuracy | NETRATE, MultiTree, InfoPath    |
| **Normalised MAE**          | `E[\|α* − α̂\|] / α*` on rates                                                                              | NETRATE                         |
| **MSE**                     | on transmission probabilities                                                                              | ConNIe, InfoPath                |
| **KL divergence**           | between estimated and true transmission _function_                                                         | KernelCascade                   |
| **Sample complexity**       | #cascades to reach a target recovery probability                                                           | Netrapalli, Abrahao, Daneshmand |

The last one is the headline in the theory papers and the one our pipeline is best positioned to measure empirically — `--num-graphs` × rollouts already sweeps cascade counts.

### 2.5 Cost estimate and ranking

| Component                             | Effort | Notes                                                                                                               |
| ------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------- |
| Cascade export (times, not states)    | **S**  | `data/generate_wm_data.py` already records per-step infection sets; a first-infection-time reduction is a few lines |
| Continuous-time simulator             | **L**  | NDlib IC is discrete-time. The entire NETRATE/InfoPath line needs real-valued delays — a new `T_endo`               |
| Dense `N×N` edge scorer + threshold   | **M**  | Feasible only at `N ≤ ~2K`; rules out most of our dataset suite                                                     |
| Survival / ℓ1 likelihood loss         | **M**  | Replaces BCE; not compatible with the current teacher-forced loop                                                   |
| Edge-set metrics (P/R/F1/BEP/AUC/MAE) | **S**  | Self-contained additions to `wm_metrics.py`                                                                         |

**Ranking against the other candidate tasks:** below `source_localization` (free — inverts the model we already have), below `influence_estimation` (already computed), below `influence_blocking` / `epidemic_control` / `cascade_reconstruction` (all keep `G` as a condition), and roughly level with `cascade_prediction` — both need data our simulator does not currently emit. **Recommended role: a §-length robustness/limitation section, or a weights-only experiment on a fixed support**, not a headline task. §9 spells out the cheap version.

### 2.6 Verdict on the two-loop formulation

[`source_localization.md`](source_localization.md) §2.3 formalizes inverse tasks as **amortized program search**: the coding-agent outer loop searches the space of inference algorithms, the world model is the forward oracle those algorithms call, and the reward is ground-truth accuracy over labelled training instances. That framing rescued source localization from being a component swap, and [`cascade_reconstruction.md`](cascade_reconstruction.md) §2.4 applies it again. **It does not work here**, and it is worth recording exactly why, because this is the only task in the folder where the failure is on the _inner_ loop rather than the outer one.

The framing needs four things. This task has one and a half of them.

| #   | Requirement                                                          | Here                                                                                                                                                                                                                                                           |
| --- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | An inference algorithm with real design freedom                      | yes **yes.** §2.2 enumerates four route families and §3–§4 differ almost entirely in search strategy: greedy submodular (NetInf, MultiTree), convex relaxation (NETRATE, ConNIe), dense threshold (FIM), non-parametric (KernelCascade). Wide, contested space. |
| 2   | The world model callable as a forward oracle, with `G` a _condition_ | no **no.** See below.                                                                                                                                                                                                                                          |
| 3   | A dense reward with ground truth we own                              | Warning: edge-F1 against `G*` is well defined, but needs a Kronecker loader (§9.3) and a continuous-time simulator (§2.5, effort **L**) before any instance exists.                                                                                                  |
| 4   | No cheaper exact alternative                                         | no **no.** NETRATE's likelihood is **convex and closed-form**.                                                                                                                                                                                                 |

**Why (2) fails.** §2.1's coincidence is real: an agent proposing `add_edge(u, v, w)` against a cascade likelihood is doing hypothesis search over `G`, and §2.3 already frames the ops correctly as search moves. The outer loop would work. The problem is what scores those moves. Our model is `f_θ(G, s_t, a_t) → s_{t+1}`; it _reads_ `G`. A hypothesis search hands it a candidate `Ĝ` and asks how well that explains the observed cascades, and **the search spends most of its time on wrong hypotheses.** Our kernel was trained on transitions from one graph distribution, so scoring adversarially-chosen off-distribution graphs is exactly where a learned model is least trustworthy, and the signal is corrupted precisely where the search most needs guidance. Source localization and cascade reconstruction never face this: there `G` is fixed and correct and only the state varies, so the model is always being asked about the graph it was trained on.

**Why (4) fails, and this one is decisive.** §1 states that the whole methodological history of this field is dodging the super-exponential sum over propagation trees, and that NETRATE and ConNIe **succeeded** by reformulating so the sum has a closed form. NETRATE's objective is convex. Replacing an exact, convex, cheap likelihood with an approximate, non-convex, expensive learned one is a regression, not a contribution. Contrast source localization, whose posterior has no closed form, and cascade reconstruction, whose tree sum is genuinely intractable: in both, a learned kernel is the only affordable option. Here the field solved tractability in 2011.

Three further blockers, each independently sufficient: the observation type is wrong (continuous infection times vs our discrete steps, and a continuous-time `T_endo` is rated **L** in §2.5); the dense `N×N` scorer is feasible only at `N ≤ 2,048`; and the benchmark is Kronecker synthetics we have no loader for, so there is no comparable published table on a graph we load.

**Verdict: the agent half fits, the world-model half does not.** This is the mirror image of `influence_estimation` and `cascade_prediction`, where the world model fits and there is nothing for an agent to write. **What would flip it**: a continuous-time simulator, a Kronecker loader, and an accepted off-distribution-scoring risk. That is weeks of work, landing us competing against FIM's `F1 = 0.58–0.75` (§5.6) with a strictly worse objective than NETRATE's convex one. Not worth it. §9 items 2 and 4 remain the right things to take from this file, and both use only the inner loop.

---

## 3. Classical and heuristic methods

The likelihood-based line, in the order it was published. **All four Gomez-Rodriguez-line code releases still resolve** (verified 2026-07-28, HTTP 206 on a ranged GET — see §3.1).

| Method                                   | Year | Venue | Idea                                                                                                                                                                                                           | Paper                                                                                                                                                | Code                                                                                                                                                                                                                                                                                                                         |
| ---------------------------------------- | ---- | ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Saito EM**                             | 2008 | KES   | EM for discrete-time IC probabilities `p(u→v)` from observed activation sequences. The first cascade→probability estimator.                                                                                    | [Springer](https://link.springer.com/chapter/10.1007/978-3-540-85567-5_9)                                                                            | no public code found                                                                                                                                                                                                                                                                                                         |
| **Goyal et al.**                         | 2010 | WSDM  | Static / continuous-time / discrete-time influence-probability models from an _action log_; introduces the Credit Distribution idea.                                                                           | [ACM DOI](https://dl.acm.org/doi/10.1145/1718487.1718518) Warning: 403 to `curl` (Cloudflare); resolves in a browser                                       | no public code found                                                                                                                                                                                                                                                                                                         |
| **NetInf**                               | 2010 | KDD   | Submodular greedy over edges: for each cascade take the **single most likely propagation tree**, add the edge with the largest marginal log-likelihood gain, `k` times. `(1−1/e)` guarantee + an online bound. | [KDD'10 via ACM](https://dl.acm.org/doi/10.1145/1835804.1835933) Warning: 403 to `curl` · [TKDD version, arXiv 1006.0234](https://arxiv.org/abs/1006.0234) | [snap.stanford.edu/netinf](http://snap.stanford.edu/netinf/) → [`netinf.tgz`](http://snap.stanford.edu/netinf/netinf.tgz) · [`netinf-macos.tgz`](http://snap.stanford.edu/netinf/netinf-macos.tgz) · also [github.com/snap-stanford/snap/examples/netinf](https://github.com/snap-stanford/snap/tree/master/examples/netinf) |
| **ConNIe**                               | 2010 | NIPS  | Convex MLE for _transmission probabilities_ with an ℓ1 sparsity penalty `ρ`. First method to estimate per-edge weights rather than assume homogeneity.                                                         | [connie-nips10.pdf](http://snap.stanford.edu/connie/connie-nips10.pdf)                                                                               | [`connie_matlab.zip`](http://snap.stanford.edu/connie/connie_matlab.zip) (MATLAB + SNOPT)                                                                                                                                                                                                                                    |
| **NETRATE** (key)                           | 2011 | ICML  | Continuous-time survival-analysis MLE for per-edge **rates** `α_{ji}`. Convex, **no tunable parameter**, sparsity falls out of the likelihood. Decouples into `N` independent per-node problems.               | [arXiv 1105.0697](https://arxiv.org/abs/1105.0697)                                                                                                   | [people.tuebingen.mpg.de/manuelgr/netrate](http://people.tuebingen.mpg.de/manuelgr/netrate/) → [`netrate.tgz`](http://people.tuebingen.mpg.de/manuelgr/netrate/netrate.tgz) (MATLAB + CVX)                                                                                                                                   |
| **MultiTree**                            | 2012 | ICML  | Same greedy shape as NetInf but sums over **all** propagation trees via Kirchhoff's matrix-tree theorem, not just the most likely one. Wins in the _small-cascade_ regime.                                     | [arXiv 1205.1671](https://arxiv.org/abs/1205.1671)                                                                                                   | no standalone public release found (C++ described in the paper)                                                                                                                                                                                                                                                              |
| **KernelCascade**                        | 2012 | NIPS  | Drops the parametric transmission function entirely: kernel-expands the hazard, so each edge can have its own arbitrary (e.g. bimodal) delay distribution.                                                     | [NIPS 2012](https://papers.nips.cc/paper_files/paper/2012/hash/d759175de8ea5b1d9a2660e45554894f-Abstract.html)                                       | no public code found                                                                                                                                                                                                                                                                                                         |
| **InfoPath**                             | 2013 | WSDM  | Stochastic-gradient NETRATE with cascade-age-decayed sampling → **time-varying** `α_{ji}(t)`, one network per day. Scales to 43K nodes / 1.4M cascades.                                                        | [arXiv 1212.1464](https://arxiv.org/abs/1212.1464)                                                                                                   | [snap.stanford.edu/infopath](http://snap.stanford.edu/infopath/) → [`infopath.tgz`](http://snap.stanford.edu/infopath/infopath.tgz)                                                                                                                                                                                          |
| **First-Edge / First-Edge+**             | 2013 | KDD   | Return only the edge between the _first two_ nodes of each trace, discard the tail. Provably near-optimal in the worst case, and trivially cheap. `+` variant adds a degree-distribution prior.                | [arXiv 1308.2954](https://arxiv.org/abs/1308.2954)                                                                                                   | no public code found                                                                                                                                                                                                                                                                                                         |
| **Soft-thresholding ℓ1-MLE**             | 2014 | ICML  | ℓ1-regularised continuous-time MLE solved with a proximal-gradient (soft-threshold) step; the paper that supplies the recovery conditions and the `O(d³ log N)` bound.                                         | [arXiv 1405.2936](https://arxiv.org/abs/1405.2936)                                                                                                   | no public code found                                                                                                                                                                                                                                                                                                         |
| **Sparse-recovery / compressed sensing** | 2015 | ICML  | Casts inference as sparse recovery; separates _what the cascade model gives you_ from _what any ℓ1 method can give you_.                                                                                       | [arXiv 1505.05663](https://arxiv.org/abs/1505.05663)                                                                                                 | no public code found                                                                                                                                                                                                                                                                                                         |
| **Belief propagation reconstruction**    | 2016 | —     | BP/message-passing reconstruction from a **single** (or few) partially observed cascade(s); handles the "only the infected set observed" regime.                                                               | [arXiv 1609.00432](https://arxiv.org/abs/1609.00432)                                                                                                 | no public code found                                                                                                                                                                                                                                                                                                         |

### 3.1 Code-release status — verified

Because the task brief singled this out: **the Gomez-Rodriguez / SNAP releases are all alive.** Checked with `curl -r 0-500 -L`:

| Artifact               | URL                                                                 | Status 2026-07-28 |
| ---------------------- | ------------------------------------------------------------------- | ----------------- |
| NetInf (Linux/Windows) | `http://snap.stanford.edu/netinf/netinf.tgz`                        | **206** yes        |
| NetInf (macOS)         | `http://snap.stanford.edu/netinf/netinf-macos.tgz`                  | **206** yes        |
| NETRATE (MATLAB)       | `http://people.tuebingen.mpg.de/manuelgr/netrate/netrate.tgz`       | **206** yes        |
| InfoPath               | `http://snap.stanford.edu/infopath/infopath.tgz`                    | **206** yes        |
| ConNIe (MATLAB)        | `http://snap.stanford.edu/connie/connie_matlab.zip`                 | **200** yes        |
| SNAP `examples/netinf` | `https://github.com/snap-stanford/snap/tree/master/examples/netinf` | **200** yes        |

Warning: **One dead link worth recording:** the NetInf landing page still points at `http://www.stanford.edu/~manuelgr/netrate/` for NETRATE — that URL now **404s**. The live NETRATE home moved to `http://people.tuebingen.mpg.de/manuelgr/netrate/`, which the InfoPath page links correctly. Anyone following the SNAP page's own link will hit a dead end.

### 3.2 Theory and sample complexity

The results this literature is actually cited for. `d` = node in-degree, `N` (or `n`) = number of nodes, `Δ` = maximum degree, `D` = super-graph degree.

| Result                                       | Bound                                                                                        | Setting                               | Source                                                                                     |
| -------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------ |
| ML estimator, general graphs                 | **`O(d² log n)`** infections per node                                                        | discrete-time IC / SIR                | [Netrapalli & Sanghavi 2012](https://arxiv.org/abs/1202.1779) [verified, abstract + Thm 1] |
| Greedy, **trees**                            | **`O(d log n)`** infections per node                                                         | same                                  | ibid. [verified, Thm 2]                                                                    |
| Information-theoretic **lower bound**        | **`Ω(d log n)`**                                                                             | same                                  | ibid. [verified, Thm 3] — so the greedy tree bound is tight                                |
| With a known super-graph                     | becomes **independent of `n`** (`log D` replaces `log n`)                                    | same                                  | ibid. [verified, abstract]                                                                 |
| Exact inference, worst case                  | **`Ω(n Δ^{1−ε})`** necessary, **`O(nΔ log n)`** sufficient                                   | continuous-time IC                    | [Abrahao et al. 2013](https://arxiv.org/abs/1308.2954) [verified, §1 + Thm 4.1]            |
| **Trees**                                    | **`O(log n)`** traces                                                                        | ibid.                                 | [verified, §1]                                                                             |
| Bounded degree `Δ`                           | **`O(poly(Δ) log n)`**; the paper's own upper bound is `Õ(Δ⁹)` vs a `Ω(Δ^{2−ε})` lower bound | ibid.                                 | [verified, §1 + §7]                                                                        |
| **Degree distribution only** (no edges)      | **`O(n)`** traces                                                                            | ibid.                                 | [verified, §1]                                                                             |
| ℓ1-MLE, incoherence condition                | **`O(d³ log N)`** cascades; **`O(d² log N)`** under a stronger condition                     | continuous-time, general transmission | [Daneshmand et al. 2014](https://arxiv.org/abs/1405.2936) [verified, abstract + §6]        |
| Uniform-random sources                       | `Ω(d⁹ log² d log N)` cascades                                                                | ibid.                                 | [verified, §1]                                                                             |
| NETRATE's own bound (as cited by Daneshmand) | `O(N d log N)` cascades                                                                      | continuous-time                       | [verified, Daneshmand §1]                                                                  |

**The two bounds to remember:** `Ω(d log n)` is the floor, and `O(d² log n)` / `O(d³ log N)` is what practical estimators achieve — i.e. **polynomial in local degree, only logarithmic in network size.** That is the reason this field believes inference is tractable on large graphs at all.

Additional theory: [Khim & Loh 2018, _A theory of maximum likelihood for weighted infection graphs_](https://arxiv.org/abs/1806.05273) (MLE and hypothesis testing for _weighted_ infection graphs — treats "did this cascade spread over graph `G`?" as a testing problem rather than a recovery one); [Hoffmann & Caramanis 2019, _Learning Graphs from Noisy Epidemic Cascades_](https://arxiv.org/abs/1903.02650) (corrupted / missing infection times).

---

## 4. Learning-based and probabilistic methods

### 4.1 Point-process / Hawkes network inference

The Hawkes line solves a _sibling_ problem: nodes fire **repeatedly**, and the network is the `N×N` **infectivity matrix** `A` in `λ_i(t) = μ_i + Σ_j Σ_{t_j<t} a_{ij} g(t − t_j)`. Recovering `A` is edge-weight inference under a different generative model. Directly relevant to us because `a_{ij}` plays exactly the role our `q(u→v)` plays.

| Method                              | Year | Venue   | Idea                                                                                                                                                                                                                                                        | Paper                                                      | Code                                                                                        |
| ----------------------------------- | ---- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **ADM4 (low-rank + sparse Hawkes)** | 2013 | AISTATS | MLE for the infectivity matrix with a **nuclear-norm + ℓ1** penalty — social influence is both community-structured (low rank) and sparse. ADMM + majorisation-minimisation.                                                                                | [PMLR v31](https://proceedings.mlr.press/v31/zhou13a.html) | third-party: [Hawkes-Process-Toolkit](https://github.com/HongtengXu/Hawkes-Process-Toolkit) |
| **Linderman & Adams**               | 2014 | ICML    | Fully Bayesian: a **random-graph prior** (Erdős–Rényi, SBM, latent-distance) over the adjacency, combined with a Hawkes likelihood; MCMC via a Poisson-superposition data augmentation. Gives posterior _uncertainty_ over edges, which no MLE method does. | [arXiv 1402.0914](https://arxiv.org/abs/1402.0914)         | [github.com/slinderman/pyhawkes](https://github.com/slinderman/pyhawkes)                    |

### 4.2 Bayesian / posterior-over-graphs

| Method                       | Year | Idea                                                                                                                                                                                               | Paper                                                | Code                 |
| ---------------------------- | ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | -------------------- |
| **Gray, Mitchell & Roughan** | 2019 | MCMC over the _graph itself_ for the discrete-time IC model; returns a posterior distribution over networks rather than a point estimate, so it quantifies which edges are genuinely identifiable. | [arXiv 1908.03318](https://arxiv.org/abs/1908.03318) | no public code found |
| **Braunstein et al.**        | 2016 | Belief propagation reconstruction from a **single** partially-observed cascade — the extreme low-data end of the sample-complexity curve.                                                          | [arXiv 1609.00432](https://arxiv.org/abs/1609.00432) | no public code found |

### 4.3 Neural / scalable methods

| Method                      | Year | Venue    | Idea                                                                                                                                                                                                                                                                                                                                                              | Paper                                                | Code                                                                                                              |
| --------------------------- | ---- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **NMF (Neural Mean-Field)** | 2020 | NeurIPS  | Learns the diffusion dynamics with a neural mean-field ODE whose parameters _are_ the influence matrix; jointly does network inference and influence estimation. The immediate predecessor and main baseline of FIM.                                                                                                                                              | [arXiv 2006.09449](https://arxiv.org/abs/2006.09449) | no public code found at a verified URL (FIM reports using "the official implementation published by the authors") |
| **FIM** (key)                  | 2024 | **WWW**  | Treats diffusion as a **continuous-time dynamical system**; approximates the propagation operator, trains a dense parameter matrix `A` with a per-node BCE loss over simulated states, then thresholds `A ≥ λ*` to get edges. Adds a sampling trick (SDTS) for influence estimation. **The only method in this review that runs on ≥10K-node real cascade data.** | [arXiv 2403.02867](https://arxiv.org/abs/2403.02867) | [github.com/kkhuang81/FIM](https://github.com/kkhuang81/FIM)                                                      |
| **Debiased Jacobian ML**    | 2026 | preprint | Recovers the network from cascade data by estimating the **Jacobian** of the transition map with a debiased ML estimator; frames inference as a causal-estimation problem with inference guarantees (CIs on edges).                                                                                                                                               | [arXiv 2606.07483](https://arxiv.org/abs/2606.07483) | no public code found                                                                                              |

**This is the paper the project note's "FIM" means.** Keke Huang, Ruize Gao, Bogdan Cautis, Xiaokui Xiao, _Scalable Continuous-time Diffusion Framework for Network Inference and Influence Estimation_, WWW 2024. Its Table 5 (§5.6) is the single most usable modern baseline table in this file: F1 on six synthetic Kronecker graphs with published ground truth and published generator parameters.

### 4.4 Discrete-time IC probability learning — cross-link

The Saito-EM and Goyal-et-al. line (§3) learns `p(u→v)` from **action logs** with no continuous clock. It is catalogued authoritatively in [`influence_estimation.md`](influence_estimation.md), because in that literature the learned probabilities are an _input_ to spread estimation rather than the object of study. The distinction that matters for us:

- **Network inference proper** (§3, §4.1–4.3): the _support_ is unknown.
- **Influence-probability learning** (Saito, Goyal): the support is **given** (you have the social graph); only the weights are unknown.

The second is a much easier problem, and — see §2.1 — it is the one our `structured_residual` head is already shaped for.

### 4.5 Adjacent, explicitly not core

Flagged so nobody mistakes them for baselines:

| Area                                                                                                                                      | Why it is adjacent, not core                                                                                                                                                                              |
| ----------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Graph structure learning (GSL)** — LDS, IDGL, Pro-GNN, SLAPS, and the [GSL survey (arXiv 2103.03036)](https://arxiv.org/abs/2103.03036) | Learns a graph to **improve downstream node classification**, from node _features_. No cascades, no diffusion likelihood, no ground-truth edge recovery metric. Shares only the phrase "learn the graph". |
| **DeepInf** ([KDD 2018](https://arxiv.org/abs/1807.05560), [code](https://github.com/xptree/DeepInf))                                     | Predicts _social influence outcome_ for an ego-network given the graph. The graph is an **input**. Opposite direction.                                                                                    |
| **Cascade / popularity prediction**                                                                                                       | Predicts cascade size from a prefix; see [`cascade_prediction.md`](cascade_prediction.md). Graph again an input.                                                                                          |
| **Link prediction**                                                                                                                       | Predicts missing edges from _observed_ edges. Network inference sees **zero** edges. See [`graph_completion.md`](graph_completion.md).                                                                    |
| **Source localization**                                                                                                                   | Also inverts the forward model, but solves for the _seed set_ with `G` known. Much cheaper for us — see [`source_localization.md`](source_localization.md).                                               |

**Transformer/VAE cascade→graph work:** this review found **no** established method that encodes raw cascades with a transformer or a VAE and decodes an adjacency matrix scored against ground-truth edges on the Kronecker benchmark. The neural line (§4.3) parameterises a dense matrix directly rather than generating one. Recorded as an open gap (§11).

---

## 5. Published results

Fewer real tables than the IM literature — most of this field publishes precision-recall _curves_. Everything below marked [verified] was read out of the paper's own table with `pdftotext -layout`; everything marked [figure] is direction-only.

### 5.1 NetInf (TKDD) (key) — the field's one clean sample-complexity table

**The most reproducible table in this file.** All networks: **1,024 nodes, 1,446 edges** (directed). Exponential incubation, `α = 1`, `β` chosen per row so that mean cascade size `r/|C|` is neither tiny nor huge (`β ∈ (0.1, 0.6)`). `f` = fraction of true edges that participate in ≥1 cascade; `|C|` = number of cascades generated to reach that `f`; `r` = total edge transmissions (so mean cascade size = `r/|C|`).

| Network                  | `f`  | `\|C\|` | `r`    | **BEP**  | **AUC**  |
| ------------------------ | ---- | ------- | ------ | -------- | -------- |
| Forest Fire              | 0.5  | 388     | 2,898  | 0.393    | 0.29     |
| Forest Fire              | 0.9  | 2,017   | 14,027 | 0.75     | 0.67     |
| Forest Fire              | 0.95 | 2,717   | 19,418 | 0.82     | 0.74     |
| Forest Fire              | 0.99 | 4,038   | 28,663 | 0.92     | 0.86     |
| Hierarchical Kronecker   | 0.5  | 289     | 1,341  | 0.37     | 0.30     |
| Hierarchical Kronecker   | 0.9  | 1,209   | 5,502  | 0.81     | 0.80     |
| Hierarchical Kronecker   | 0.95 | 1,972   | 9,391  | 0.90     | 0.90     |
| Hierarchical Kronecker   | 0.99 | 5,078   | 25,643 | **0.98** | **0.98** |
| Core-periphery Kronecker | 0.5  | 140     | 1,392  | 0.31     | 0.23     |
| Core-periphery Kronecker | 0.9  | 884     | 9,498  | 0.84     | 0.80     |
| Core-periphery Kronecker | 0.95 | 1,506   | 14,125 | 0.93     | 0.91     |
| Core-periphery Kronecker | 0.99 | 3,110   | 30,453 | **0.98** | 0.96     |
| Flat (random) Kronecker  | 0.5  | 200     | 1,324  | 0.34     | 0.26     |
| Flat (random) Kronecker  | 0.9  | 1,303   | 7,707  | 0.84     | 0.83     |
| Flat (random) Kronecker  | 0.95 | 1,704   | 9,749  | 0.89     | 0.88     |
| Flat (random) Kronecker  | 0.99 | 3,652   | 21,153 | 0.97     | 0.97     |

[verified, TKDD Table II]

**Read this table as a sample-complexity curve, because that is what it is.** Recovery is governed by _cascade coverage of the edge set_, not by raw cascade count: at `f = 0.5` every topology sits at BEP ≈ 0.31–0.39, and at `f = 0.99` every topology sits at BEP ≈ 0.92–0.98. **An edge that never transmitted leaves no trace and cannot be recovered** — this is the practical face of the `Ω(d log n)` lower bound in §3.2. Note also that the cost of reaching `f = 0.99` is topology-dependent: core-periphery needs 3,110 cascades, hierarchical needs 5,078.

Warning: **Internal inconsistency to be aware of:** Table II's caption says all networks have **1,446 edges**, but Fig. 5's caption describes the Forest Fire network as **1,024 nodes / 1,477 edges** [verified, both captions]. The gap is ~2%; it does not change any conclusion, but do not quote "1,446" as though the paper were unambiguous.

**NetInf's optimality bound** [verified, TKDD §4.1]: at 2,000 inferred edges, the greedy solution is **≥ 97% of the (NP-hard) optimum** on synthetic data and **≥ 84%** on real data, via the online bound.

### 5.2 NETRATE (ICML 2011)

Setup [verified, §4.1]: Kronecker networks **1,024 nodes / 2,048 edges**; Forest Fire **1,024 nodes / 2,422 edges** (the paper's own text reads "1,024 edges and 2,422 edges" — a typo for nodes/edges). Rates drawn `α ∈ [0.01, 1]` for exponential and Rayleigh, `α ∈ [0.01, 2]` for power-law. Observation window `T = 10`. 5,000 cascades unless stated.

| Result                                                          | Value                                                                                                                           | Tier                   |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| Normalised MAE on rates, 5,000 cascades, most networks × models | **< 25%**                                                                                                                       | [verified, §4.1 prose] |
| Cascades needed for normalised MAE **< 20%**                    | **≈ 5,000**                                                                                                                     | [verified, §4.1 prose] |
| Precision-recall vs NetInf and ConNIe                           | NETRATE dominates both **in the Pareto sense**; "ConNIe and NetInf do not achieve NETRATE's recall for **any** precision value" | [verified, §4.1 prose] |
| Accuracy vs ConNIe                                              | NETRATE higher **for every** penalty factor `ρ`                                                                                 | [verified, §4.1 prose] |
| Runtime, single CPU, all incoming edges of one node             | **≈ 20 s** at 20,000 nodes                                                                                                      | [figure, Fig 3(c)]     |
| Cluster scaling                                                 | 25 CPUs → **16,000 nodes / 32,000 edges in < 4 hours**                                                                          | [verified, §4.1 prose] |

**The single most quotable sentence in this literature, for our purposes:** "Estimating transmission rates is considerably harder than simply discovering edges and therefore more cascades are needed for accurate estimates." [verified, §4.1]. That is the direct justification for §2's split verdict.

**Real data** [verified, §4.2]: MemeTracker hyperlink cascades — top **500** media sites/blogs by document count, **5,000** hyperlink edges as ground truth, **116,234** cascades. Per-cell precision/recall published only as Fig. 4.

### 5.3 ConNIe (NIPS 2010)

Setup [verified, §3.1]: directed scale-free (preferential attachment) and Erdős–Rényi graphs, both **512 nodes / 1,024 edges**. Edge probabilities `A_ij ~ U[0.05, 1]`. Transmission-time models: exponential, power-law, and **Weibull** (`α = 9.5`, `k = 2.3`, fitted to the Hong Kong SARS outbreak — the only non-monotone delay density in this literature). Cascades generated until 99% of edges transmitted at least once; "on the same order of cascades as there are nodes".

| Dataset                                 | Nodes / Edges   | ConNIe BEP | NetInf BEP | Weight error                       | Tier                   |
| --------------------------------------- | --------------- | ---------- | ---------- | ---------------------------------- | ---------------------- |
| Synthetic scale-free                    | 512 / 1,024     | **> 0.85** | —          | MSE **< 0.05** (NetInf > 2× worse) | [verified, §3.1 prose] |
| Collaboration net (network scientists)  | **379** nodes   | **≈ 0.95** | —          | error **< 0.03**                   | [verified, §3.2 prose] |
| Email net (European research institute) | **593 / 2,824** | **≈ 0.95** | —          | error **< 0.03**                   | [verified, §3.2 prose] |
| Product-recommendation net              | **275 / 1,522** | **0.74**   | **0.55**   | n/a (no ground-truth weights)      | [verified, §3.2 prose] |

**Robustness** [verified, §3.1]: ConNIe still recovers the network at a **noise-to-signal ratio of 0.4** on infection times (Gaussian perturbation of observed times / mean transmission time). This is the only published noise-robustness number in the classical line, and it is the number to beat if we ever claim our learned model degrades gracefully.

**Runtime** [verified, §3.2]: the 275-node recommendation network took **< 20 seconds**; the abstract claims "thousand-node networks in a matter of minutes".

> **Cross-link worth noticing:** ConNIe's "collaboration network between 379 scientists doing research on networks" is the **largest connected component of the NetScience graph we already load** (`ca-netscience`, 379 / 914 — see [`influence_maximization.md`](influence_maximization.md) §6). Our `netscience` loader gives the full 1,589 / 2,742 graph; extracting its LCC reproduces ConNIe's graph exactly. That makes ConNIe's `BEP ≈ 0.95` the one published network-inference number in this entire file that sits on a graph already in our suite.

### 5.4 MultiTree (ICML 2012)

Setup [verified, §4.1]: three **1,024-node** Kronecker networks; rates `α ~ U(0.5, 1.5)`; `β = 0.5`; **200 observed cascades** for the headline precision-recall figures; the AUC-gain figures use **1,024 nodes / 1,024 edges**. The deliberate design choice is the _small_-cascade regime, on the argument that real social networks change faster than you can record cascades.

| Claim                                               | Value                                                                                                                   | Tier                   |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| Recall vs NetInf / ConNIe / NETRATE at 200 cascades | MultiTree reaches **higher recall than all three**; at recalls NetInf can reach, precision is "very similar"            | [verified, §4.1 prose] |
| Accuracy vs NetInf                                  | beats NetInf on **> half** of NetInf's solutions, matches the rest                                                      | [verified, §4.1 prose] |
| ConNIe / NETRATE accuracy at 200 cascades           | "typically significantly lower" — **ConNIe degrades the most** under cascade scarcity                                   | [verified, §4.1 prose] |
| Exception                                           | NETRATE beats everything on the **hierarchical** Kronecker network                                                      | [verified, §4.1 prose] |
| AUC gain over NetInf vs #cascades                   | large at small `\|C\|`, → 0 (or slightly negative) at large `\|C\|`                                                     | [figure, Fig 3]        |
| Runtime vs NETRATE                                  | MultiTree and NetInf ≈ **1 order of magnitude faster**; even one full-gradient NETRATE iteration is slower              | [verified, §4.1 prose] |
| Scalability                                         | 100,000-node and 200,000-node graphs (avg 2 edges/node), 10,000 cascades → **10.12 ms** and **12.14 ms** per edge added | [verified, §4.1 prose] |

**Real data** [verified, §4.2]: top **1,000** media sites/blogs, **10,000** hyperlink edges, **500 longest** hyperlink cascades, power-law transmission. Per-cell results published only as Fig. 5.

### 5.5 KernelCascade (NIPS 2012) — the only clean real-data method table

Setup [verified, §5.1]: Kronecker core-periphery and Erdős–Rényi; per-edge transmission is a **mixture of two Rayleighs** `f(t|θ, a₁, b₁, a₂, b₂) = θR₁ + (1−θ)R₂`, with `p(t) = f(t|0.5, 10, 1, 20, 1)`, `q(t) = f(t|0.5, 0, 1, 20, 1)`, or a per-edge random choice of the two. Cascade counts swept over **50, 100, 200, 400, 800, 1000**; 10 random instantiations per setting. NetInf is given the **true edge count** as an advantage.

**MemeTracker, top 500 sites / 6,466 edges / 11,530 cascades from 7,181,406 posts in one month** [verified, Table 1]:

| Method            | Precision | Recall   | **F1**   | Predicted edges |
| ----------------- | --------- | -------- | -------- | --------------- |
| NETINF            | 0.62      | 0.62     | 0.62     | 6,466           |
| NETRATE (exp)     | **0.93**  | 0.23     | 0.37     | 1,600           |
| **KernelCascade** | 0.79      | **0.66** | **0.72** | 5,368           |

[verified, Table 1]

This is **the only head-to-head precision/recall/F1 table on real cascades in the entire classical line**, and it is the one to cite for "how good is network inference on real data" — the answer is F1 ≈ 0.6–0.7, not 0.95.

Synthetic [figure, Figs 3–4]: KernelCascade beats NetInf and both NETRATE variants in every one of the six settings, and **fully recovers the network at ≈ 1,000 cascades**, which the competitors do not. NETRATE's performance is "very sensitive to the choice of transmission function", ranging from second-best to worst depending on the true generator.

### 5.6 FIM (WWW 2024) (key) — the best modern table

Datasets [verified, Table 2]:

|               | HR     | Hier1024 / Hier2048 | Core1024 / Core2048 | Rand1024 / Rand2048 | MemeTracker | Weibo      | Twitter    |
| ------------- | ------ | ------------------- | ------------------- | ------------------- | ----------- | ---------- | ---------- |
| **#Nodes**    | 128    | 1024 / 2048         | 1024 / 2048         | 1024 / 2048         | **498**     | **8,190**  | **12,677** |
| **#Cascades** | 10,000 | 20,000 / 10,000     | 20,000 / 10,000     | 20,000 / 10,000     | **8,304**   | **43,365** | **3,461**  |

Synthetic edge counts = **4× node count** (so Hier1024 = 4,096 arcs); `λ_uv ~ U(0, 0.1)`; `T = 10`, `ε = 1.0`; edge threshold `λ* = 0.01`; cascades split **80/10/10** train/val/test [verified, §5 + §5.1].

**Network inference F1 and BCE loss on the six synthetic networks** [verified, Table 5]:

| Metric   | Method  | Rand1024 | Hier1024 | Core1024 | Rand2048 | Hier2048 | Core2048 |
| -------- | ------- | -------- | -------- | -------- | -------- | -------- | -------- |
| BCE loss | **FIM** | **0.08** | **0.04** | **0.17** | **0.06** | **0.04** | **0.14** |
| BCE loss | NMF     | 0.11     | 0.06     | 0.20     | 0.08     | 0.06     | 0.19     |
| **F1**   | **FIM** | **0.60** | **0.75** | **0.58** | **0.41** | **0.42** | **0.34** |
| **F1**   | NMF     | 0.44     | 0.49     | 0.36     | 0.32     | 0.36     | 0.16     |

**Read this carefully — it is the most important sober fact in this file.** On 1,024-node Kronecker graphs with **20,000 cascades**, the 2024 state of the art reaches **F1 = 0.58–0.75**. Doubling the graph to 2,048 nodes while _halving_ cascades to 10,000 drops it to **F1 = 0.34–0.42**. NetInf's `BEP ≈ 0.98` (§5.1) is not comparable — it is measured at `f = 0.99` edge coverage, a data regime FIM does not assume. **Network inference on realistic cascade budgets is not a solved problem.**

Scalability [verified, §5.1]: on the authors' server, **NMF and NETRATE are both OOM on Weibo and Twitter**, and NETRATE could not finish within 12 hours on MemeTracker or on any synthetic set except the 128-node HR. FIM speedup over NMF up to **8.24×** (Hier2048); over ConTinEst for influence estimation up to **100–120×**.

Downstream influence maximization on the inferred `A` [verified, Table 4], spread at seed-set size 4 → 10:

| Dataset     | Method  | 4          | 5          | 6          | 7          | 8          | 9          | 10         |
| ----------- | ------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| MemeTracker | **FIM** | **65.85**  | **75.18**  | **88.33**  | **108.88** | **113.90** | **127.11** | **136.67** |
| MemeTracker | NMF     | 53.14      | 64.78      | 70.66      | 76.10      | 83.39      | 88.23      | 94.95      |
| Core2048    | **FIM** | **517.12** | **576.19** | **559.19** | **634.83** | **636.26** | 639.38     | **723.88** |
| Core2048    | NMF     | 444.25     | 506.17     | 476.44     | 570.90     | 591.80     | **637.40** | 567.28     |

### 5.7 InfoPath (WSDM 2013)

Synthetic [verified, §4.1]: two Kronecker networks, both **1,024 nodes / 2,048 edges** — core-periphery `[0.9, 0.5; 0.5, 0.3]` and hierarchical `[0.9, 0.1; 0.1, 0.9]`. Each edge's rate follows one of four evolution patterns — **Slab, Square, Chainsaw, Hump** — over 200 time units with **1,000 cascades per time unit** [verified, Fig 2 caption]. Continuous patterns (Chainsaw, Hump) are tracked "near perfectly"; discontinuous ones (Slab, Square) are harder [verified, §4.1 prose]. Per-cell numbers are figure-only.

Real, per-topic [verified, Table 3] — sites and meme cascades per query `Q`:

| Topic / news event | # sites | # memes (cascades) |
| ------------------ | ------- | ------------------ |
| Amy Winehouse      | 1,207   | 109,650            |
| Fukushima          | 1,666   | 383,745            |
| Gaddafi            | 1,358   | 440,646            |
| Kate Middleton     | 1,427   | 191,777            |
| NBA                | 2,087   | 1,543,630          |
| Occupy             | 1,875   | 655,183            |
| Strauss-Kahn       | 1,263   | 204,238            |
| Syria              | 1,565   | 615,176            |

Scale [verified, §4.2]: 38 topics × 365 days = **> 13,000** network-inference solves, on a **1,000-core / 6 TB** cluster, in **< 4 hours**. The largest single run: "Occupy Wall Street", a **43,415-node** time-varying network over 18 months (Jan 2011 – Jun 2012) from **1,381,793** cascades.

### 5.8 Trace complexity — Abrahao et al. (KDD 2013)

Networks [verified, §6]: Facebook-Rice **graduate 503 nodes (Δ = 48)** and **undergraduate 1,220 nodes (Δ = 287)**; synthetic **1,024-node** Barabási–Albert (Δ = 174), `G(n, p)` with `p = 0.2` (Δ = 253), and a power-law tree with exponent 3 (Δ = 94). Degree-distribution reconstruction used **10n** traces. NetInf is given the **true edge count** as an advantage. `First-Edge+` threshold `p = 0.5`.

| Finding                                                                                                                                                           | Tier                   |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| Degree-distribution CCDF from `10n` traces shows "almost perfect overlap" with the truth on BA and both Facebook graphs                                           | [verified, §6]         |
| First-Edge returns ≈ `ℓ` edges from `ℓ` traces, **all true positives, never a false positive**                                                                    | [verified, §6]         |
| First-Edge+ ≈ NetInf in F1 on BA and Facebook-Undergrad, and **approaches perfect inference as traces → `Ω(nΔ)`** while NetInf plateaus                           | [verified, §6 + Fig 2] |
| Power-law tree: First-Edge / First-Edge+ reach **perfect** inference at ≈ **5,000 traces**                                                                        | [verified, §6]         |
| `G(n, p)` with `p = 0.2`: **neither** First-Edge+ **nor NetInf** reaches high F1 at any trace count tested — dense random graphs have very large trace complexity | [verified, §6]         |
| Runtime: First-Edge+ "a matter of seconds"; NetInf "a couple of hours" on the same networks                                                                       | [verified, §6]         |

**The `G(n,p)` row is the useful warning.** Our `er` synthetic family is exactly this generator. If we ever run a network-inference experiment on `--dataset er`, the published expectation is that **everything fails**.

### 5.9 Daneshmand et al. (ICML 2014)

Setup [verified, §8]: Forest Fire and Kronecker networks with **128 nodes**, rates `α ~ U(0.5, 1.5)`, `T = 5` (theory illustrations) or `T = 10` (comparisons), success probability estimated over **100 independent cascade sets**, `λ_n = K√(log p / n)`.

| Result                                                                                  | Value                                                                     | Tier               |
| --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ------------------ |
| F1 vs #cascades, hierarchical Kronecker (POW) and Forest Fire (EXP), 500–2,500 cascades | soft-thresholding > NETRATE > **First-Edge (clearly worst)**              | [figure, Fig 4]    |
| Success probability vs `β` in `n = 10βd log p`, for `p ∈ {6, 26, 40}`                   | the three curves **line up**, confirming the `log p` scaling of Theorem 2 | [figure, Fig 3(a)] |

The `p`-collapse result is the empirical confirmation that sample complexity is **logarithmic in super-neighbourhood size** — the claim that makes large-graph inference plausible.

### 5.10 Netrapalli & Sanghavi (SIGMETRICS 2012)

All figure-only. Networks [verified, §6 figure captions]: 2-D grids **10×10, 15×15, 20×20**, and a **200-node random 4-regular** graph with a **degree-8 super-graph**.

The one qualitative result worth carrying: **recovery probability collapses onto a single curve when plotted against the _average number of infections per node_, not against the total number of cascades** [figure, Fig 3]. Different grid sizes need visibly different cascade counts but the same per-node infection count. Super-graph side information helps only **moderately**, matching the `log D` vs `log n` prediction [figure, Fig 4].

---

## 6. Datasets

This literature's dataset story is unlike every other file in this folder. **Real graphs are almost irrelevant here** — you cannot evaluate edge recovery without ground truth, so the benchmark is _synthetic graphs with published generator parameters_ (§6.3), and the "real" experiments substitute a hyperlink graph as a ground-truth proxy (§6.4). The authoritative rows for the social graphs themselves live in [`influence_maximization.md`](influence_maximization.md) §6.

### 6.1 What we already load that applies

Only two of our fifteen graphs appear anywhere in this literature, and one of them only after a preprocessing step:

| Our `--dataset`          | Nodes / Edges                           | Where it appears                                                                                                                                                                                                                   | Note                                                                                                                                                                     |
| ------------------------ | --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| yes `netscience`          | 1,589 / 2,742 undirected (avg deg 3.45) | **ConNIe §3.2** uses its **379-node LCC** ("collaboration network between 379 scientists doing research on networks"), BEP ≈ 0.95                                                                                                  | `largest_connected_component()` on our loader reproduces it; NetRepo ships that component as [`ca-netscience`](https://networkrepository.com/netscience.php) (379 / 914) |
| yes `email_eu_core`       | 1,005 / 24,929 arcs (avg total deg 51)  | **ConNIe §3.2**'s "email social network of a small European research institute" is the **same institute's** email log at an earlier snapshot: **593 nodes / 2,824 edges**                                                          | Warning: **Not the same file.** ConNIe's is a 593-node cut; SNAP's `email-Eu-core` is 1,005 nodes. Different graphs, same institution. Do not quote ConNIe's BEP against ours  |
| yes `weibo`               | 1,787,443 / ≈216M arcs                  | **FIM (WWW'24)** uses the AMiner Weibo cascade release — same primary source ([Zhang et al. IJCAI'13](https://www.aminer.cn/influencelocality)) — but reduced to **8,190 nodes / 43,365 cascades** by their ≥5-cascade node filter | Our loader takes the _network_ file; FIM takes the _cascade_ file and prunes hard                                                                                        |
| yes `er` (`--dataset er`) | tunable                                 | **Abrahao et al. §6** tests `G(n, 0.2)`, `n = 1024`, `Δ = 253`                                                                                                                                                                     | Warning: **The published result is that inference fails.** Neither First-Edge+ nor NetInf reaches high F1 at any trace count tested (§5.8)                                     |
| yes `ba`                  | tunable                                 | **Abrahao et al. §6** (BA, `n = 1024`, `Δ = 174`); **ConNIe §3.1** (directed preferential attachment, 512 / 1,024)                                                                                                                 | BA is used, but never with a published per-cell table — figure-only                                                                                                      |

**Everything else in our suite — `jazz`, `cora_ml`, `facebook`, `power_grid`, `ca_grqc`, `wiki_vote`, `lastfm_asia`, `nethept`, `netphy`, `twitter`, `digg`, `youtube`, `sbm`, `karate`, `ws` — has no network-inference baseline at all.** That is not an oversight in this review; the task needs cascades, and a bare topology file supplies none.

### 6.2 Full catalogue of graphs used as ground truth

Synthetic rows are directed; **edge counts are arcs**. Real rows follow the convention stated per row.

| Graph                               | Nodes               | Edges                                            | Directed?                      | Avg deg               | Used by                         | Source                                                                                                     |
| ----------------------------------- | ------------------- | ------------------------------------------------ | ------------------------------ | --------------------- | ------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Kronecker (3 variants)              | **1,024**           | **1,446** arcs                                   | directed                       | 1.41 out              | NetInf (TKDD Table II)          | generator, §6.3                                                                                            |
| Kronecker (3 variants)              | **1,024**           | **2,048** arcs                                   | directed                       | 2.0 out               | NETRATE, MultiTree, InfoPath    | generator, §6.3                                                                                            |
| Kronecker (3 variants)              | **1,024**           | **1,024** arcs                                   | directed                       | 1.0 out               | MultiTree (AUC-gain figures)    | generator, §6.3                                                                                            |
| Kronecker Hier/Core/Rand            | **1,024 / 2,048**   | **4,096 / 8,192** arcs                           | directed                       | 4.0 out               | **FIM (Table 5)**               | generator, §6.3                                                                                            |
| Kronecker Core                      | 4,096               | 16,384 arcs                                      | directed                       | 4.0 out               | FIM (Appendix A.3, `Core4096`)  | generator, §6.3                                                                                            |
| Kronecker hierarchical              | **128**             | — (not stated)                                   | directed                       | —                     | Daneshmand et al.               | generator, §6.3                                                                                            |
| Forest Fire                         | **1,024**           | **1,477** arcs (Fig 5) / **1,446** (Table II) Warning: | directed                       | ≈1.4 out              | NetInf                          | generator, §6.3                                                                                            |
| Forest Fire                         | **1,024**           | **2,422** arcs                                   | directed                       | 2.37 out              | NETRATE                         | generator, §6.3                                                                                            |
| Forest Fire                         | **128**             | —                                                | directed                       | —                     | Daneshmand et al.               | generator, §6.3                                                                                            |
| Scale-free (pref. attachment)       | **512**             | **1,024** arcs                                   | directed                       | 2.0 out               | ConNIe                          | `nx.barabasi_albert_graph` equivalent                                                                      |
| Erdős–Rényi                         | **512**             | **1,024** arcs                                   | directed                       | 2.0 out               | ConNIe                          | `nx.gnp_random_graph`                                                                                      |
| `G(n, p=0.2)`                       | **1,024**           | ≈**104,858** arcs [derived: `p·n(n−1)`]          | directed                       | ≈102 out, Δ = 253     | Abrahao et al.                  | `nx.gnp_random_graph(1024, 0.2)`                                                                           |
| Barabási–Albert                     | **1,024**           | — (Δ = 174)                                      | directed                       | —                     | Abrahao et al.                  | `nx.barabasi_albert_graph`                                                                                 |
| Power-law tree (exp 3)              | **1,024**           | **1,023** edges (tree)                           | —                              | ≈2 undirected, Δ = 94 | Abrahao et al.                  | custom generator, not released                                                                             |
| 2-D grids                           | **100 / 225 / 400** | 180 / 420 / 760 undirected [derived]             | undirected                     | ≈3.8                  | Netrapalli & Sanghavi           | `nx.grid_2d_graph`                                                                                         |
| Random 4-regular                    | **200**             | **400** undirected                               | undirected                     | 4                     | Netrapalli & Sanghavi           | `nx.random_regular_graph(4, 200)`                                                                          |
| Facebook-Rice graduate              | **503**             | — (Δ = 48)                                       | undirected                     | —                     | Abrahao et al.                  | Rice University Facebook crawl; **no working public URL found**                                            |
| Facebook-Rice undergraduate         | **1,220**           | — (Δ = 287)                                      | undirected                     | —                     | Abrahao et al.                  | same; **no working public URL found**                                                                      |
| Collaboration (network scientists)  | **379**             | **914** undirected                               | undirected                     | 4.82                  | ConNIe                          | [NetRepo `ca-netscience`](https://networkrepository.com/netscience.php) · **= LCC of our yes `netscience`** |
| Email (European research institute) | **593**             | **2,824**                                        | directed (weighted by #emails) | 9.5                   | ConNIe                          | **no public URL found** (predates SNAP `email-Eu-core`)                                                    |
| Product-recommendation net          | **275**             | **1,522**                                        | directed                       | 5.5                   | ConNIe                          | subset of the Leskovec 4M-person / 16M-recommendation corpus; **no public URL for the subset**             |
| Blog/news hyperlink graph           | **500**             | **4,000** arcs                                   | directed                       | 8.0                   | NetInf (real-data ground truth) | derived from MemeTracker, §6.4                                                                             |
| Blog/news hyperlink graph           | **500**             | **5,000** arcs                                   | directed                       | 10.0                  | NETRATE                         | §6.4                                                                                                       |
| Blog/news hyperlink graph           | **500**             | **6,466** arcs                                   | directed                       | 12.9                  | KernelCascade                   | §6.4                                                                                                       |
| Blog/news hyperlink graph           | **1,000**           | **10,000** arcs                                  | directed                       | 10.0                  | MultiTree                       | §6.4                                                                                                       |

Warning: **Note the four different "top-500 MemeTracker hyperlink graphs"** — 4,000 / 5,000 / 6,466 arcs, from four papers, all called "the top 500 sites". They are built from different time slices with different thresholds and **are not the same graph**. This is the network-inference equivalent of the seven name collisions catalogued in [`influence_maximization.md`](influence_maximization.md) §6.3, and it means NetInf's `BEP = 0.28`, NETRATE's Fig 4, and KernelCascade's `F1 = 0.72` are **not** three numbers on one benchmark.

### 6.3 (key) The Kronecker ground-truth benchmark — exact settings

**This is the single most actionable section in the file.** These are reproducible in ~10 lines: a Kronecker graph is `K^{⊗k}`, the `k`-fold Kronecker power of a 2×2 initiator matrix `Θ`, sampled edge-by-edge with probability `Π Θ[b_i(u), b_i(v)]`. `k = 10` gives 1,024 nodes, `k = 11` gives 2,048. Three initiators are standard, each producing a qualitatively different global structure.

#### The initiator matrices

| Structure                            | `Θ`                            | Produces                     | Used by                                              |
| ------------------------------------ | ------------------------------ | ---------------------------- | ---------------------------------------------------- |
| **Random / flat** (Erdős–Rényi-like) | `[0.5, 0.5; 0.5, 0.5]`         | homogeneous degrees, no core | **NetInf, NETRATE, MultiTree, FIM** — unanimous      |
| **Hierarchical community**           | `[0.9, 0.1; 0.1, 0.9]`         | nested block communities     | **NETRATE, MultiTree, InfoPath, FIM, Daneshmand**    |
| **Hierarchical community**           | `[0.962, 0.107; 0.107, 0.962]` | same shape, denser           | **NetInf only** (fitted, per Clauset et al. 2008)    |
| **Core-periphery**                   | `[0.9, 0.5; 0.5, 0.3]`         | dense core + sparse rim      | **NETRATE, MultiTree, InfoPath, FIM, KernelCascade** |
| **Core-periphery**                   | `[0.962, 0.535; 0.535, 0.107]` | same shape, denser core      | **NetInf only** (fitted, per Leskovec et al. 2008)   |

[verified: NetInf TKDD §4.1; NETRATE §4.1; MultiTree §4.1; InfoPath §4.1; FIM §5]

Warning: **NetInf is the odd one out and it matters.** NetInf (2010) used the _fitted_ initiators from the Kronecker-graphs papers (`[0.962, 0.535; 0.535, 0.107]` and `[0.962, 0.107; 0.107, 0.962]`); **every paper after it** switched to the rounder `[0.9, 0.5; 0.5, 0.3]` and `[0.9, 0.1; 0.1, 0.9]`. NetInf's Table II (§5.1) therefore sits on **different graphs** from NETRATE Fig 1, MultiTree Fig 2, InfoPath Fig 2, and FIM Table 5, even though all five say "core-periphery Kronecker, 1,024 nodes". Reproduce NetInf's numbers only with NetInf's initiators.

#### Cascade-generation parameters, per paper

| Paper             | Nodes / arcs                  | Rate distribution                                                                                   | Transmission model(s)                                    | `T`                                        | #cascades                                   |
| ----------------- | ----------------------------- | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | ------------------------------------------ | ------------------------------------------- |
| **NetInf** (TKDD) | 1,024 / 1,446                 | single shared `α = 1`                                                                               | exponential incubation                                   | —                                          | 140–5,078 (swept to hit edge coverage `f`)  |
| **ConNIe**        | 512 / 1,024                   | `A_ij ~ U[0.05, 1]` (probabilities)                                                                 | exponential, power-law, **Weibull** (`α = 9.5, k = 2.3`) | —                                          | ≈ `N` (until 99% of edges fired)            |
| **NETRATE**       | 1,024 / 2,048                 | `α ~ U[0.01, 1]` (EXP, RAY); `U[0.01, 2]` (POW)                                                     | exponential, power-law, Rayleigh                         | **10**                                     | 2,500 / 5,000 / 7,500 / 10,000              |
| **MultiTree**     | 1,024 / 2,048 (and /1,024)    | `α ~ U(0.5, 1.5)`, `β = 0.5`                                                                        | exponential, power-law, Rayleigh                         | —                                          | **200** (headline), swept to 10⁵ for timing |
| **KernelCascade** | Kronecker core-periphery + ER | mixture of two Rayleighs: `p(t)=f(t\|0.5,10,1,20,1)`, `q(t)=f(t\|0.5,0,1,20,1)`, or per-edge random | **non-parametric** (kernel)                              | `T^c`                                      | 50, 100, 200, 400, 800, 1000                |
| **InfoPath**      | 1,024 / 2,048                 | rates follow **Slab / Square / Chainsaw / Hump** evolution over 200 time units                      | exponential, Rayleigh                                    | —                                          | **1,000 per time unit**                     |
| **Daneshmand**    | 128                           | `α ~ U(0.5, 1.5)`                                                                                   | exponential, power-law, Rayleigh                         | **5** (theory) / **10** (comparison)       | 150 (theory), 500–2,500 (comparison)        |
| **FIM**           | 1,024 & 2,048 / 4× nodes      | `λ_uv ~ U(0, 0.1)`                                                                                  | continuous-time IC                                       | **5, 8, 10, 12, 15** (`ε ∈ {0.5,1,1.5,2}`) | **20,000 / 10,000**, split 80/10/10         |

[verified, each paper's experimental-setup section]

#### Forest Fire

The scale-free counterpart, used alongside Kronecker by NetInf, NETRATE and Daneshmand. NetInf: 1,024 nodes / 1,477 edges (Fig 5) with a power-law cascade- size distribution; mean **9.1** and median **8** cascades per edge over 4,038 cascades [verified, TKDD §4.1]. NETRATE: 1,024 nodes / 2,422 edges [verified, Fig 2 caption]. **Neither paper publishes its forward/backward burning probabilities**, so Forest Fire is _less_ reproducible than Kronecker — recorded in §11.

#### What we could reproduce today

`nx.stochastic_block_model` will not give a Kronecker graph, but the generator is ~10 lines of `numpy` (sample `A[u,v] ~ Bernoulli(Π Θ[bit_i(u), bit_i(v)])`). Adding `--dataset kronecker --kron-theta hier|core|rand` would put our pipeline on the **only benchmark this entire field agrees on**, and — unlike our `sbm` family, which has no published IM baseline ([`influence_maximization.md`](influence_maximization.md) §6) — it comes with five papers' worth of published numbers. That is the highest-value dataset addition this review found.

### 6.4 Cascade corpora — counts, span, and timestamp resolution

The task brief asks for cascade counts, average cascade length, time span and timestamp resolution. Here they are, per corpus and per paper, because **every paper cuts the same raw dumps differently**.

#### MemeTracker — the raw corpus

| Property                   | Value                                                                                                                                                                                                          | Tier                              |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| Source                     | [SNAP `memetracker9`](https://snap.stanford.edu/data/memetracker9.html) — direct files e.g. [`quotes_2008-09.txt.gz`](https://snap.stanford.edu/data/bigdata/memetracker9/quotes_2008-09.txt.gz) (HTTP 206 yes) | [verified]                        |
| Files                      | 9 monthly `.txt.gz`: `quotes_2008-08` … `quotes_2009-04`                                                                                                                                                       | [verified, SNAP page]             |
| Corpus size (SNAP page)    | **96 million memes**; **> 17 million** distinct phrases; 54% blogs / 46% news media                                                                                                                            | [verified, SNAP page]             |
| Corpus size (NetInf's cut) | **172 million** articles + blog posts from **1 million** online sources                                                                                                                                        | [verified, TKDD §4.2]             |
| Time span (NetInf)         | **1 Sep 2008 – 31 Aug 2009** (one year)                                                                                                                                                                        | [verified, TKDD §4.2]             |
| Phrases extracted (NetInf) | **343 million**; **8 million** distinct appeared > 10 times; **150 million** cumulative mentions                                                                                                               | [verified, TKDD §4.2]             |
| **Timestamp resolution**   | **1 second** — the record format is `T 2008-09-09 22:35:24`                                                                                                                                                    | [verified, SNAP page format spec] |
| Record format              | `P` document URL · `T` timestamp · `Q` phrase · `L` hyperlink                                                                                                                                                  | [verified, SNAP page]             |
| Warning: Dead mirror             | `http://memetracker.org/data.html` — the URL cited by NetInf, NETRATE and MultiTree — **does not resolve** (`memetracker.org` fails DNS as of 2026-07-28). Use the SNAP mirror                                 | [verified]                        |

#### MemeTracker — the derived cascade sets, per paper

| Paper                  | Sites (nodes) | Ground-truth arcs                                     | #Cascades                                              | Avg cascade length | Cascade type                                   |
| ---------------------- | ------------- | ----------------------------------------------------- | ------------------------------------------------------ | ------------------ | ---------------------------------------------- |
| **NetInf** (hyperlink) | 500           | 4,000                                                 | not stated                                             | not stated         | hyperlink chains                               |
| **NetInf** (meme)      | 500           | 4,000                                                 | **top 5,000** phrase clusters                          | not stated         | phrase clusters                                |
| **NETRATE**            | 500           | 5,000                                                 | **116,234**                                            | not stated         | hyperlink                                      |
| **MultiTree**          | 1,000         | 10,000                                                | **500 longest**                                        | not stated         | hyperlink                                      |
| **KernelCascade**      | 500           | 6,466                                                 | **11,530** (from **7,181,406** posts in **one month**) | not stated         | hyperlink                                      |
| **FIM**                | **498**       | ground truth not used (F1 reported on synthetic only) | **8,304**                                              | not stated         | phrase clusters, nodes filtered to ≥5 cascades |

[verified, each paper's real-data section]

#### InfoPath's corpus — a different, later crawl

| Property                                        | Value                                                                                                  | Tier                |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------- |
| Corpus                                          | **300 million** blog posts + news articles from **3.3 million** websites                               | [verified, §4.2]    |
| Time span                                       | **March 2011 – February 2012** (one year)                                                              | [verified, §4.2]    |
| Memes extracted                                 | **179 million** (longer than 4 words); **34 million** distinct appeared ≥ 2× → **34 million cascades** | [verified, §4.2]    |
| Sites used                                      | top **5,000** by memes mentioned                                                                       | [verified, §4.2]    |
| **Temporal resolution of the inferred network** | **1 day** (one network per day, 365 per topic)                                                         | [verified, §4.2]    |
| Per-topic cascade counts                        | 109,650 – 1,543,630, see §5.7 Table 3                                                                  | [verified, Table 3] |
| Largest single run                              | Occupy Wall Street: **43,415 nodes**, **1,381,793 cascades**, **18 months** (Jan 2011 – Jun 2012)      | [verified, §4.2]    |
| Data page                                       | [snap.stanford.edu/infopath/data.html](http://snap.stanford.edu/infopath/data.html) (HTTP 206 yes)      | [verified]          |

#### Other cascade corpora used in this literature

| Corpus                       | Nodes      | Edges            | #Cascades   | Avg cascade | Span / resolution                           | Source                                                                            |
| ---------------------------- | ---------- | ---------------- | ----------- | ----------- | ------------------------------------------- | --------------------------------------------------------------------------------- |
| **Sina Weibo** (AMiner)      | 1,170,689  | 225,877,808 arcs | **115,686** | **148**     | 2012 retweet log, second-resolution [claim] | [AMiner](https://www.aminer.cn/influencelocality) — registration required         |
| Weibo **as FIM cuts it**     | **8,190**  | —                | **43,365**  | not stated  | same                                        | ibid., ≥5-cascade filter                                                          |
| **Digg 2009**                | 279,631    | 2,251,166 arcs   | **3,553**   | **847**     | Jun–Jul 2009, **1-second** votes [claim]    | [ISI / Lerman](https://www.isi.edu/~lerman/downloads/digg2009.html)               |
| **Twitter** (Hodas & Lerman) | —          | —                | —           | —           | **all URL tweets, October 2010**            | [Sci. Rep. 4:4343](https://www.nature.com/articles/srep04343)                     |
| Twitter **as FIM cuts it**   | **12,677** | —                | **3,461**   | not stated  | same                                        | ibid., ≥5-cascade filter                                                          |
| **MAG (CS)**                 | 1,436,158  | 15,928,078 arcs  | **181,020** | **29**      | citation years — **1-year** resolution      | Microsoft Academic Graph via [IMINFECTOR](https://github.com/geopanag/IMINFECTOR) |
| **Higgs Twitter**            | 456,631    | 14,855,875 arcs  | —           | —           | 1–7 Jul 2012, **1-second**                  | [SNAP](https://snap.stanford.edu/data/higgs-twitter.html)                         |

Digg / Weibo / MAG counts are [verified] from IMINFECTOR's Table 3, transcribed in [`influence_maximization.md`](influence_maximization.md) §6.2. Timestamp resolutions marked [claim] were not confirmed against the files.

Warning: **Two Weibo graphs and two Digg graphs exist** — our `weibo` loader and our `digg` loader read _different files_ from the ones above. See [`influence_maximization.md`](influence_maximization.md) §6.3 (name collisions) before quoting any number across the two.

**The practical point for us:** none of these corpora carry the `(G, s_t, a_t, s_{t+1})` shape our pipeline consumes, and none carry interventions at all. Using them means writing a _second_ data path — logged trajectories, per Yuntong's scoping — not extending `generate_wm_data.py`.

---

## 7. Which paper uses which

Cells mark the dataset **as that paper reports it**. Read §6.3's initiator warning and §6.2's four-hyperlink-graphs warning before treating any column as comparable to another.

| Dataset / family                   | NetInf'10 | ConNIe'10 | NETRATE'11 | MultiTree'12 | KernelCascade'12 | InfoPath'13 | Netrapalli'12 | Abrahao'13 | Daneshmand'14 | FIM'24 |
| ---------------------------------- | --------- | --------- | ---------- | ------------ | ---------------- | ----------- | ------------- | ---------- | ------------- | ------ |
| Kronecker **random / flat**        | yes         |           | yes          | yes            | yes                |             |               |            |               | yes      |
| Kronecker **hierarchical**         | yes         |           | yes          | yes            |                  | yes           |               |            | yes             | yes      |
| Kronecker **core-periphery**       | yes         |           | yes          | yes            | yes                | yes           |               |            |               | yes      |
| **Forest Fire**                    | yes         |           | yes          | yes            |                  |             |               |            | yes             |        |
| Scale-free / BA                    |           | yes         |            |              |                  |             |               | yes          |               |        |
| Erdős–Rényi / `G(n,p)`             |           | yes         |            |              | yes                |             |               | yes          |               |        |
| Power-law tree                     |           |           |            |              |                  |             |               | yes          |               |        |
| 2-D grid                           |           |           |            |              |                  |             | yes             |            |               |        |
| Random 4-regular                   |           |           |            |              |                  |             | yes             |            |               |        |
| Facebook-Rice (503 / 1,220)        |           |           |            |              |                  |             |               | yes          |               |        |
| yes NetScience LCC (379)            |           | yes         |            |              |                  |             |               |            |               |        |
| Email institute (593)              |           | yes         |            |              |                  |             |               |            |               |        |
| Recommendation net (275)           |           | yes         |            |              |                  |             |               |            |               |        |
| **MemeTracker** hyperlink cascades | yes         |           | yes          | yes            | yes                |             |               |            |               |        |
| **MemeTracker** phrase cascades    | yes         |           |            |              |                  | yes           |               |            |               | yes      |
| yes Sina Weibo                      |           |           |            |              |                  |             |               |            |               | yes      |
| Twitter (Hodas–Lerman)             |           |           |            |              |                  |             |               |            |               | yes      |

**Column-by-column takeaway:** the Kronecker triple is the only thing every generation agrees on. Real-graph coverage is thin, non-overlapping, and — with the single exception of ConNIe's 379-node NetScience LCC — disjoint from our suite.

---

## 8. Evaluation protocol

### 8.1 The standard synthetic protocol

Six steps, essentially identical in every paper from NetInf to FIM:

1. **Generate `G*`** from a Kronecker initiator (§6.3) or Forest Fire.
2. **Draw per-edge rates** `α*_{ji}` from a uniform distribution (the range varies — see §6.3's table; it is _not_ standardised).
3. **Pick a source** uniformly at random per cascade. (Abrahao et al. §3 notes this is a modelling assumption, not a fact about real cascades.)
4. **Simulate** to a time horizon `T`, recording only **first** infection times. Nodes not infected by `T` get `t = ∞`.
5. **Infer** `Ĝ` (and `α̂`).
6. **Score** against `G*` by sweeping the method's sparsity knob — `k` (NetInf, MultiTree), `ρ` (ConNIe), `λ*` (FIM) — while NETRATE produces one point with no knob.

### 8.2 Metrics, and the trap in each

| Metric                | Definition                                            | Trap                                                                                                                                                                                               |
| --------------------- | ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Precision**         | fraction of `Ĝ`'s edges present in `G*`               | trivially 1.0 if you output one edge — never quote alone                                                                                                                                           |
| **Recall**            | fraction of `G*`'s edges present in `Ĝ`               | greedy methods _get exhausted_ (no edge left with positive marginal gain) and **cannot reach recall 1.0 at any `k`** — MultiTree's whole selling point                                             |
| **Break-even point**  | precision at the sweep point where precision = recall | requires a knob to sweep; **NETRATE has none**, so its results are a single point, not a curve                                                                                                     |
| **F1**                | `2PR/(P+R)`                                           | the only one comparable across methods with and without knobs                                                                                                                                      |
| **AUC**               | area under the PR/ROC curve                           | NetInf reports it alongside BEP; the two rank topologies the same in Table II                                                                                                                      |
| **"Accuracy"**        | `1 − Σ\|I(α*)−I(α̂)\| / (Σ I(α*) + Σ I(α̂))`            | Warning: **This is not classification accuracy.** It is a normalised symmetric difference over the _support_. A method predicting no edges scores **0**, not 1. Reported by NETRATE, MultiTree, InfoPath |
| **Normalised MAE**    | `E[\|α*−α̂\|]/α*`                                      | only defined on edges that exist in both; says nothing about the support                                                                                                                           |
| **MSE**               | on transmission probabilities                         | Warning: ConNIe takes it over the **union** of true-edge positions and predicted-edge positions, with absent edges set to 0 — not over all `N²` pairs                                                    |
| **KL divergence**     | between estimated and true transmission _function_    | KernelCascade only; the others assume a parametric family so their KL is degenerate                                                                                                                |
| **Sample complexity** | #cascades to hit a target recovery probability        | **plot against per-node infections, not total cascades** — Netrapalli & Sanghavi Fig 3 shows the curves collapse only under the former                                                             |

### 8.3 Handicaps and asymmetries to declare

Three protocol details silently favour some methods:

- **NetInf and MultiTree are given `k` = the true edge count.** Abrahao et al. and KernelCascade both state they hand NetInf the true `|E|` "to give it an advantage" [verified]. NETRATE, ConNIe and FIM get no such hint.
- **NETRATE has no tunable parameter**, so it yields one solution while NetInf and ConNIe yield a whole curve from which a point must be "selected blindly (or at best heuristically)" [verified, NETRATE §4.1]. Comparing a curve's best point to a single point is not a fair comparison, and NETRATE says so.
- **Cascade coverage `f` is the real independent variable.** NetInf's Table II is indexed by it; most later papers index by raw cascade count instead, which hides the topology-dependence (core-periphery reaches `f = 0.99` in 3,110 cascades, hierarchical needs 5,078).

### 8.4 The real-data protocol, and why its ceiling is low

There is no ground truth on real cascades, so every paper substitutes the **hyperlink graph** — an edge `u→v` if some post on `u` linked to a post on `v` — and then infers from _phrase_ cascades. This is a proxy, and a weak one: NetInf itself says its assumption is "sites prefer to create links to sites that recently mentioned information while completely ignoring the authority of the site", which "is not satisfied in real life", and on that basis calls `BEP = 0.44` "a good result" [verified, TKDD §4.2].

The honest ceiling on real data is therefore **F1 ≈ 0.6–0.7** (KernelCascade Table 1, §5.5) — not the `0.9+` the synthetic tables report. Any claim we make about real cascades has to live under that ceiling.

---

## 9. Implications for this project

1. **Do not claim this task whole.** §2 is the honest version: our `ICTransmissionHead` gives us the _rate_ half and none of the _existence_ half. Claiming both would require a dense `N×N` scorer or an ℓ1-prox loss, neither of which is in the codebase.

2. **The cheap, defensible version is a weights-only experiment on a known support.** Freeze `edge_index`, perturb `edge_weight` away from the truth, train `structured_residual` with the existing BCE loss, and report normalised MAE `E[|w* − ŵ|]/w*` against NETRATE's `< 25%` at 5,000 cascades (§5.2). This is **one evaluation script and no new simulator** — the residual head already anchors on `w`, so the gradient path exists today. It also directly tests a claim we currently make implicitly: that `structured_residual`'s learned correction is small because `w` is right.

3. **Add a Kronecker generator.** §6.3 gives three initiator matrices that five papers agree on, `k = 10 → 1,024` nodes, and published numbers on each. A `--dataset kronecker --kron-theta hier|core|rand` flag would put our _existing_ IM and rollout experiments on the one synthetic family this literature has standardised — which is more than our `sbm` family can say ([`influence_maximization.md`](influence_maximization.md) §6). Highest value-per-hour item in this file.

4. **Sample complexity is the experiment our pipeline is uniquely good at.** Every theory paper's headline (`Ω(d log n)`, `O(d² log n)`, `O(d³ log N)`) is about _how much data_ recovery needs. We can already sweep cascade counts by varying `--num-graphs` × rollouts. Plotting our one-step `delta_f1` against **per-node infection count** (not cascade count — §8.2) would give the world-model analogue of Netrapalli & Sanghavi Fig 3, which nobody has published.

5. **Report `er` results as a negative control, if at all.** Abrahao et al. §6 found that on `G(n, 0.2)` **neither First-Edge+ nor NetInf** achieves high F1 at any trace count. Our `er` family is the same generator. A poor result there is the published expectation, not a defect.

6. **The realistic bar is much lower than the classical tables suggest.** FIM (2024), with 20,000 cascades on 1,024 nodes, reaches **F1 = 0.58–0.75**; at 2,048 nodes on 10,000 cascades, **F1 = 0.34–0.42** (§5.6). NetInf's `BEP = 0.98` is measured at 99% edge coverage — a data regime nobody has in practice. Any comparison we publish must state which regime it is in.

7. **The `add_edge` / `remove_edge` / `set_edge_weight` ops mean something different here.** §2.3: a search move over a hypothesis, not an intervention on the world. If the coding-agent loop ever proposes edge ops against a cascade likelihood, that is _hypothesis search_, and the writeup should say so — a reader seeing the same five ops will assume otherwise.

8. **The two-loop formulation does not apply, and the failure is on the world-model side.** §2.6 is the full argument. The coding-agent outer loop fits fine: hypothesis search over `G`, scored by cascade likelihood, is a legitimate program-search problem and §2.2 shows the algorithm space is wide. It fails on the inner loop for two independent reasons: **`G` is the variable rather than a condition**, so the search asks our kernel to score off-distribution graphs precisely where it is least reliable; and **NETRATE's likelihood is already convex and closed-form**, so a learned approximation is a regression rather than a contribution. This makes network inference the mirror image of `influence_estimation` and `cascade_prediction`: there the world model fits and there is no algorithm to write; here there is an algorithm to write and no role for the world model. Worth stating explicitly in any writeup that enumerates which tasks use which loop.

9. **This is a limitation section, not a chapter.** Ranked in §2.5 below `source_localization`, `influence_estimation`, `influence_blocking`, `epidemic_control` and `cascade_reconstruction`. The one paragraph worth writing: _"our world model conditions on a known `G`; recovering `G` from traces alone is a distinct inverse problem with its own literature, whose modern state of the art reaches F1 ≈ 0.6 at 20K cascades on 1K nodes."_

---

## 10. Reference list

**Foundational — structure inference** [NetInf, KDD 2010 (ACM DOI, 403 to `curl`)](https://dl.acm.org/doi/10.1145/1835804.1835933) · [NetInf, TKDD (arXiv 1006.0234)](https://arxiv.org/abs/1006.0234) · code: [snap.stanford.edu/netinf](http://snap.stanford.edu/netinf/) · [`netinf.tgz`](http://snap.stanford.edu/netinf/netinf.tgz) · [github.com/snap-stanford/snap/examples/netinf](https://github.com/snap-stanford/snap/tree/master/examples/netinf)

[ConNIe, NIPS 2010](http://snap.stanford.edu/connie/connie-nips10.pdf) · code: [`connie_matlab.zip`](http://snap.stanford.edu/connie/connie_matlab.zip)

[NETRATE, ICML 2011 (arXiv 1105.0697)](https://arxiv.org/abs/1105.0697) · code: [people.tuebingen.mpg.de/manuelgr/netrate](http://people.tuebingen.mpg.de/manuelgr/netrate/) · [`netrate.tgz`](http://people.tuebingen.mpg.de/manuelgr/netrate/netrate.tgz)

[MultiTree, ICML 2012 (arXiv 1205.1671)](https://arxiv.org/abs/1205.1671) · no public code found · [KernelCascade, NIPS 2012](https://papers.nips.cc/paper_files/paper/2012/hash/d759175de8ea5b1d9a2660e45554894f-Abstract.html) · no public code found · [InfoPath, WSDM 2013 (arXiv 1212.1464)](https://arxiv.org/abs/1212.1464) · code: [snap.stanford.edu/infopath](http://snap.stanford.edu/infopath/) · [`infopath.tgz`](http://snap.stanford.edu/infopath/infopath.tgz)

**Theory and sample complexity** [Netrapalli & Sanghavi, SIGMETRICS 2012 (arXiv 1202.1779)](https://arxiv.org/abs/1202.1779) · no public code found · [Abrahao et al., KDD 2013 (arXiv 1308.2954)](https://arxiv.org/abs/1308.2954) · no public code found · [Daneshmand et al., ICML 2014 (arXiv 1405.2936)](https://arxiv.org/abs/1405.2936) · no public code found · [Pouget-Abadie & Horel, ICML 2015 (arXiv 1505.05663)](https://arxiv.org/abs/1505.05663) · no public code found · [Khim & Loh 2018 (arXiv 1806.05273)](https://arxiv.org/abs/1806.05273) · no public code found · [Hoffmann & Caramanis 2019 (arXiv 1903.02650)](https://arxiv.org/abs/1903.02650) · no public code found

**Probabilistic / EM / Bayesian** [Saito, Nakano & Kimura, KES 2008](https://link.springer.com/chapter/10.1007/978-3-540-85567-5_9) · no public code found · [Goyal, Bonchi & Lakshmanan, WSDM 2010 (ACM DOI, 403 to `curl`)](https://dl.acm.org/doi/10.1145/1718487.1718518) · no public code found · [Gray, Mitchell & Roughan 2019 (arXiv 1908.03318)](https://arxiv.org/abs/1908.03318) · no public code found · [Braunstein et al. 2016 (arXiv 1609.00432)](https://arxiv.org/abs/1609.00432) · no public code found

**Hawkes / point-process network inference** [Zhou, Zha & Song, AISTATS 2013 (PMLR v31)](https://proceedings.mlr.press/v31/zhou13a.html) · third-party code: [Hawkes-Process-Toolkit](https://github.com/HongtengXu/Hawkes-Process-Toolkit) · [Linderman & Adams, ICML 2014 (arXiv 1402.0914)](https://arxiv.org/abs/1402.0914) · code: [github.com/slinderman/pyhawkes](https://github.com/slinderman/pyhawkes)

**Modern / neural** [NMF — Neural Mean-Field, NeurIPS 2020 (arXiv 2006.09449)](https://arxiv.org/abs/2006.09449) · no public code found at a verified URL · [**FIM**, WWW 2024 (arXiv 2403.02867)](https://arxiv.org/abs/2403.02867) · code: [github.com/kkhuang81/FIM](https://github.com/kkhuang81/FIM) · [Debiased Jacobian ML 2026 (arXiv 2606.07483)](https://arxiv.org/abs/2606.07483) · no public code found

**Adjacent (not baselines — §4.5)** [Graph structure learning survey (arXiv 2103.03036)](https://arxiv.org/abs/2103.03036) · [DeepInf, KDD 2018 (arXiv 1807.05560)](https://arxiv.org/abs/1807.05560) · [code](https://github.com/xptree/DeepInf)

**Data** [SNAP MemeTracker (96M memes)](https://snap.stanford.edu/data/memetracker9.html) · [InfoPath data](http://snap.stanford.edu/infopath/data.html) · [ISI/Lerman Digg 2009](https://www.isi.edu/~lerman/downloads/digg2009.html) · [AMiner Weibo](https://www.aminer.cn/influencelocality) · [Hodas & Lerman Twitter (Sci. Rep. 4:4343)](https://www.nature.com/articles/srep04343) · [SNAP Higgs Twitter](https://snap.stanford.edu/data/higgs-twitter.html) · [NetRepo `ca-netscience`](https://networkrepository.com/netscience.php)

---

## 11. Open gaps

Honest list of what this review could **not** establish.

### 11.1 Code releases — the one thing that came out clean

**The Gomez-Rodriguez line's code releases all still resolve** (checked 2026-07-28 with `curl -r 0-500 -L`, browser UA):

| Artifact                | URL                                                           | Status     |
| ----------------------- | ------------------------------------------------------------- | ---------- |
| NetInf (Linux/Windows)  | `http://snap.stanford.edu/netinf/netinf.tgz`                  | **206** yes |
| NetInf (macOS)          | `http://snap.stanford.edu/netinf/netinf-macos.tgz`            | **206** yes |
| NETRATE (MATLAB + CVX)  | `http://people.tuebingen.mpg.de/manuelgr/netrate/netrate.tgz` | **206** yes |
| InfoPath                | `http://snap.stanford.edu/infopath/infopath.tgz`              | **206** yes |
| ConNIe (MATLAB + SNOPT) | `http://snap.stanford.edu/connie/connie_matlab.zip`           | **200** yes |
| SNAP `examples/netinf`  | `github.com/snap-stanford/snap/tree/master/examples/netinf`   | **200** yes |

Warning: **One broken link inside SNAP's own pages:** the NetInf landing page links NETRATE as `http://www.stanford.edu/~manuelgr/netrate/`, which **404s**. The live home is the MPI Tübingen URL above (the InfoPath page links it correctly). Anyone following SNAP's link hits a dead end.

Warning: **`memetracker.org` is gone** — DNS resolution fails entirely. NetInf, NETRATE and MultiTree all cite `http://memetracker.org/data.html` as their data source. Use the [SNAP mirror](https://snap.stanford.edu/data/memetracker9.html) instead.

**No public code found** for: MultiTree, KernelCascade, First-Edge/First-Edge+, Daneshmand's soft-thresholding, Pouget-Abadie & Horel, Khim & Loh, Saito EM, Goyal et al., Braunstein et al., Gray et al., the debiased-Jacobian preprint, and **NMF** (FIM says it used "the official implementation published by the authors", but this review found no URL that resolves). Only **FIM** among the learning-based methods has a verified repo.

### 11.2 Result tables that could not be extracted

- **NETRATE has no result table at all.** Every precision/recall/accuracy/MAE number in the ICML paper is a figure. §5.2 reports the paper's own prose claims (`< 25%` MAE, `< 20%` at 5,000 cascades) and marks the rest [figure].
- **MultiTree, InfoPath, Netrapalli & Sanghavi, Daneshmand, Abrahao, Pouget-Abadie & Horel** — same: precision-recall and F1 curves only. Their _setup_ tables are transcribed; their _result_ cells do not exist as numbers.
- **KernelCascade's synthetic F1 and KL curves** (Figs 3–4, six settings × six cascade counts) are figure-only; only its MemeTracker Table 1 is transcribed.
- **FIM's Table 7 (`Core4096` MAE/runtime), Table 8 and Table 9** were located but not transcribed — they are appendix ablations on influence estimation, not network inference.
- **NetInf's runtime and scalability numbers** were not transcribed (they exist in the TKDD paper but were not needed for any claim here).

### 11.3 Datasets marked [claim] or unresolved

- **Timestamp resolution for Weibo, Digg 2009 and MAG** is [claim] — inferred from the corpora's known formats, not confirmed by downloading a file. Only MemeTracker's **1-second** resolution is [verified], from SNAP's own format spec (`T 2008-09-09 22:35:24`).
- **Average cascade length is missing for every MemeTracker cut** — none of NetInf, NETRATE, MultiTree, KernelCascade or FIM publishes it. Only Digg/Weibo/MAG have published averages (847 / 148 / 29), and those come from IMINFECTOR's table, not from a network-inference paper.
- **Facebook-Rice (503 and 1,220 nodes)** — Abrahao et al.'s only real graphs. **No working public download URL found.** Node counts and max degrees are [verified] from their §6; edge counts are not published at all.
- **ConNIe's email network (593 / 2,824)** and **recommendation subset (275 / 1,522)** — no public URL. The email graph is _not_ SNAP's `email-Eu-core` (1,005 nodes); same institution, different snapshot.
- **Daneshmand's 128-node Kronecker and Forest Fire edge counts** are never stated.
- **Twitter as FIM uses it** — FIM cites Hodas & Lerman (Sci. Rep. 2014) but publishes only its post-filter counts (12,677 nodes / 3,461 cascades). The raw corpus size and a direct download URL were not established.

### 11.4 Kronecker / Forest Fire settings that could not be pinned down

- **Forest Fire burning probabilities are unpublished** in NetInf, NETRATE and Daneshmand. All three say "Forest Fire model (Leskovec et al.)" and give node and edge counts, but never the forward/backward burn probabilities. **Forest Fire is therefore not reproducible from these papers**, unlike Kronecker.
- **Daneshmand's Kronecker initiator is not stated** — the paper says "hierarchical Kronecker" and cites Leskovec et al., so it is _presumably_ `[0.9, 0.1; 0.1, 0.9]`, but that is inference, not extraction.
- **KernelCascade's Kronecker initiators are not stated.** It says "core-periphery structure [11]" and "Erdős–Rényi random", citing the same sources as NETRATE — again presumably `[0.9, 0.5; 0.5, 0.3]` and `[0.5, 0.5; 0.5, 0.5]`, but not written down.
- **NetInf's Table II edge count is internally inconsistent** — caption says 1,446 edges for all networks, Fig. 5's caption says the Forest Fire graph has 1,477. Both are [verified]; the paper does not reconcile them.
- **`k` (Kronecker power) is never stated by any paper** — it is inferred from the node counts (`2^10 = 1,024`, `2^11 = 2,048`). Consistent across all five papers, so this is safe, but it is [derived] not [verified].
- **How the Kronecker sampler is seeded / thresholded to hit an exact edge count** (1,024 / 1,446 / 2,048 / 4,096 arcs from the same initiator) is not described anywhere. Reproducing an exact edge count will need a rejection or rescaling step the papers do not specify.

### 11.5 Method coverage gaps

- **No transformer-based or VAE-based cascade→graph method was found** that scores edge recovery on the Kronecker benchmark. §4.3's neural line parameterises a dense matrix directly. If such a method exists it is not cited by FIM (2024), which is the most recent survey-adjacent work here.
- **Pouget-Abadie & Horel's and Khim & Loh's experimental sections were not read** — both were confirmed as theory contributions and cited for their framing, but no numbers from either appear in this file.
- **Wang et al. Bayesian network inference** (named in the task brief) could not be disambiguated to a specific paper; §4.2 covers the Bayesian line via Gray/Mitchell/Roughan and Linderman & Adams instead.
