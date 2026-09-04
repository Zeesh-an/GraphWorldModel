# Cascade Reconstruction: Prior Work, Datasets, and Published Results

Cascade reconstruction recovers the **unobserved portion of a diffusion that already happened**: which nodes were actually infected, _when_ they were activated, and _who infected whom_. It is the smoothing problem that sits between [`source_localization.md`](source_localization.md) (recover `s_0` only) and network inference (recover `G` itself). Here `G` is known, the dynamics are known or learned, and the unknown is the hidden trajectory `s_0 … s_T`. Our world model already emits a per-step `frontier` channel and stores it on disk, so the ground truth these papers spend whole sections approximating is something we can just read back.

**The same trap applies here as in source localization, and the same escape.** Every method in §4.2 obtains a forward kernel `p(s_{t+1} | s_t, G)` and then inverts it, so dropping ours into that slot is a component swap rather than a contribution ([`source_localization.md`](source_localization.md) §2.2, and `README.md` cross-cutting finding 8). §2.4 therefore formalizes cascade reconstruction the same way: the coding-agent outer loop searches the space of **trajectory decoders**, and the world model is the transition oracle those decoders call. On the four criteria that framing needs, this task scores _better_ than source localization on every one (§2.3), and it is blocked on exactly one concrete simulator change (§2.6).

All URLs returned HTTP 200 on **2026-07-29** unless annotated otherwise. **Every `dl.acm.org` link in this file returns 403 to a scripted request** and opens normally in a browser: that is ACM's bot policy, not a dead link. IEEE Xplore returns 202 for the same reason. Both are treated as resolving.

---

## 0. Verification policy

| Tier           | Meaning                                                                                                              |
| -------------- | -------------------------------------------------------------------------------------------------------------------- |
| **[verified]** | Read from the paper's own table via text extraction, or from the data repository's own statistics page. Trustworthy. |
| **[derived]**  | Computed by us from a file we downloaded, or reconstructed arithmetically from published splits.                     |
| **[figure]**   | Read off a plotted figure: the paper published no table. Approximate, direction only.                               |
| **[claim]**    | Stated in prose by a paper or a secondary source; not cross-checked against a file or table.                         |

**Automated PDF summarizers hallucinate plausible numbers from these papers.** This literature is unusually easy to scramble for two reasons. First, **most of its headline results are figures, not tables**: Rozenshtein et al. (KDD 2016), Xiao et al. (SDM/ICDM 2018) and Farajtabar et al. (AISTATS 2015) publish **zero** result tables between them [verified by full-text extraction of all four PDFs]; any per-cell number a summarizer offers for those papers is fabricated. Second, the two dominant metrics point in opposite directions (`F1↑`, `NRMSE↓`) and DITTO's tables interleave them column-by-column, so a mis-parse silently inverts the ranking. Every number below marked [verified] was read from `pdftotext -layout` output of the paper's own table.

**Edge-count convention.** Undirected graphs are quoted as _undirected edges_; directed graphs as _arcs_. Our loaders report `adjacency.nnz`, which for a symmetrized undirected graph is **2× the undirected edge count**. This single convention difference explains most apparent "discrepancies" between our numbers and published tables: check it before concluding two graphs differ.

**"Cascade" means two different objects in this file.** An _influence cascade_ is a tree whose edges are who-infected-whom; a _network cascade_ is the set of infected nodes with edges induced from `G`. Sadikov et al. (WSDM 2011) show the two behave differently under sampling and report separate columns for each [verified]: never compare a number from one to the other.

---

## 1. Task definition

Given a graph `G = (V, E)`, a diffusion model `M` (known, parameterized, or learned), and a **partial observation** `O` of one episode of diffusion, recover the hidden trajectory:

```
Ŷ = argmax_Y  p(Y | O, G, M),     Y = (s_0, s_1, …, s_T)
```

where `s_t ∈ {S, I, R}^{|V|}` (or `{0,1}^{|V|}` for monotone SI/IC). Depending on what `O` contains and what part of `Y` is scored, the literature splits into four sub-tasks that are routinely conflated:

| Sub-task                               | `O` contains                                      | Recover                                                                    | Canonical papers                                                |
| -------------------------------------- | ------------------------------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **Missing-node completion**            | a uniform-random subsample of the infected set    | the rest of the infected set, and cascade statistics (size, depth, width)  | Sadikov WSDM'11 · Sundareisan SDM'15 (NetFill) · Xiao ICDM'18   |
| **Infection-time inference**           | infected set, timestamps partly or wholly missing | the per-node activation time `t(v)`                                        | Sefer & Kingsford ICDM'14 (DHREC) · Chen TKDE'19 · DITTO KDD'23 |
| **Propagation-tree inference**         | infected nodes (± times)                          | the who-infected-whom forest                                               | Zong ICDM'12 · Xiao SDM'18 · DIPT 2025                          |
| **Retrospective / "back to the past"** | a late, sparse observation                        | the _whole_ history back to `s_0`, including the source and its start time | Farajtabar AISTATS'15 · Rozenshtein KDD'16 · DITTO KDD'23       |

### The four axes that determine which methods apply

| Axis                | Settings                                                           | Why it matters                                                                                                                                                                                           |
| ------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Topology**        | `G` known (almost all of this file) · `G` unknown                  | If `G` is unknown the problem _is_ network inference: NetInf/NETRATE recover `G` from many cascades. See network inference.                                                |
| **Time**            | full timestamps · partial timestamps · **final snapshot only**     | The hardest and most recent setting is DASH: "reconstruct history from **A Single Snapshot**", DITTO's formulation, which drops _both_ the known-parameter and the known-source assumptions [verified]. |
| **Observation**     | subsample of infected · frontier reports only · pooled/group tests | Rozenshtein's `RS` (each active node reported w.p. β) vs `FR` (frontier reported after θ interactions) is the cleanest published statement of this axis [verified].                                      |
| **Node visibility** | all nodes visible · hidden nodes exist                             | Hidden nodes break the tree and cause systematic under-estimation of cascade size: "phantom cascades" (arXiv 1502.01602).                                                                               |

### Why the task exists

Epidemiology needs the transmission tree, not just the case count: contact tracing, super-spreader attribution, and counterfactual "what if we had quarantined at day 5" all require who-infected-whom. Platform-side, APIs return a _sample_ (Twitter's public stream is exactly the uniform-random subsample Sadikov et al. model) so every measured cascade statistic is biased low unless corrected. And methodologically it is the strictest test of a learned forward model: source localization only asks the model to rank `|V|` candidates, reconstruction asks it to produce a coherent `T × |V|` trajectory.

---

## 2. Fit with our methodology

**Status: strong fit, real work, and the best both-loops fit in this folder.** Nothing new to simulate, no new action op, no new data-generation run, but it needs an _inference procedure_ written on top of the trained model, which is the honest cost (§2.11).

That missing inference procedure is not a gap to be plugged with the cheapest thing that runs. **It is the contribution**, and §2.4 formalizes writing it as a search the coding agent performs. §2.1 and §2.2 establish the two assets that make the formulation viable (a supervised transition kernel, and free ground truth); §2.3 explains why the obvious use of them is not enough.

### 2.1 The world model is the transition kernel these papers are missing

Every method in §3 needs `p(s_{t+1} | s_t, G)` and gets it in one of three unsatisfying ways: assume IC/SI with a hand-set global `β` (NetFill, NETSLEUTH, DHREC), assume a parametric continuous-time kernel and fit it from _many other_ cascades (NETRATE, Farajtabar), or estimate `β` by a mean-field approximation from the single observed snapshot (DITTO §4.1). Ours is **learned, per-edge, and already trained**:

```
p(s_{t+1} | s_t, G)   =   ICTransmissionHead(encoder(X, graph))
                          q(u→v) = sigmoid(MLP([h_u, h_v, w_uv]))
                          p_new(v) = 1 − ∏_u (1 − q(u→v)·frontier_u)
```

Cascade reconstruction is then textbook **filtering/smoothing against that kernel**: given observations on a subset of `(node, time)` pairs, infer the hidden trajectory. Actions are `NULL` throughout (`--inject-p 0`, no `--action-ops`), so `T_exo = identity` and the factorization collapses to `s_{t+1} = T_endo(s_t)`, the special case we already train and already evaluate.

The `structured` head matters here specifically. A `linear` head's free-running rollout saturates to the whole graph, which makes any smoothing objective degenerate (the model believes every hidden node was infected). The self-terminating `structured` head (`count_bias −0.58` on our BA-100 IC run) is what makes trajectory likelihood a meaningful score at all. This is the same prerequisite the coding-agent loop was waiting on.

### 2.2 The `frontier` channel _is_ the quantity these papers reconstruct

`s_t = (infected, frontier)` where `frontier` = **activated at step `t`**. That is verbatim the signal Rozenshtein's `FR` reporting scheme samples, the `hitting time` DITTO's NRMSE scores, and the per-level structure Zong's consistent trees enumerate. Our generator writes it to disk every step:

| What a reconstruction benchmark needs        | Where it already is                                                                                                                                                                  |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Ground-truth per-step activation `s_0 … s_T` | `transitions_<IC\|LT>_<split>.jsonl`, one record per `t`, keyed by `episode_id`; `branch == "main"` is the real trajectory. Group by `episode_id`, sort by `t` → the full history.   |
| Ground-truth infection _time_ per node       | `t` at which a node first appears in `next_state.frontier`. No extra bookkeeping.                                                                                                    |
| Soft per-node targets (not just binary)      | `next_marginal_infected` / `next_marginal_frontier`: the `--mc-marginals` estimates of `P(infected at t+1)`. DITTO/DIPT have no equivalent; they score against one stochastic draw. |
| A masking protocol                           | Drop records / drop nodes from `state` at load time. Zero generator changes.                                                                                                         |
| The graph the episode ran on                 | `graphs/*.npz` + `reconstruct_episode_adjacency` already replays edge ops per episode.                                                                                               |

**Consequence: supervised evaluation is free.** Mask a fraction of one episode, reconstruct, score against what we already wrote. Every paper in §5 had to either simulate its own ground truth (DITTO, DIPT, Xiao, Rozenshtein, all of them) or accept that on real data there is none. We are in the first camp already, at zero marginal cost.

### 2.3 Why "DITTO with our kernel" is not the contribution

The tempting move is the one §4.2's table sets up: DITTO estimates a single scalar `β̂` by mean-field from one snapshot, we have a learned per-edge `q(u→v)` supervised against MC marginals at every step, so swap ours in and report the delta. Resist it for the same reason the source-localization file gives (§2.2 there): the forward kernel is a _component_ of these methods, and improving a component reproduces a paper its authors already wrote.

The case is admittedly weaker here than it is for SL-VAE, and it is worth being precise about why, because the difference is real. SL-VAE **states** its forward model is pluggable and demonstrates it across GAT, MONSTOR and DeepIS. DITTO makes no such claim, and the gap between a scalar `β̂` and a per-edge learned `q` is far larger than the gap between two black-box regressors. A kernel-swap paper here would not be empty. It would still be a second-order result in someone else's frame, and it forfeits the outer loop entirely.

**There is also a specific reason the reframing has more traction here than in source localization.** DITTO's central design decision is that plain MLE over histories is unstable because the likelihood is nearly flat in `β̂` (their Fig. 2), which is what motivates the barycenter formulation [claim, argued in prose and figure]. **We have no scalar `β` to be flat in.** Our kernel is per-edge and learned, so the premise behind DITTO's objective may simply not hold for us, and the right objective becomes an open question rather than a settled one. "Which decoding objective is correct for a per-edge learned kernel" is not a question to guess at once; it is a question to search.

### 2.4 The formalization: amortized trajectory decoding

The skeleton is the one in [`source_localization.md`](source_localization.md) §2.3 and is not repeated here: search the space of inversion programs offline against a labelled training set, bind a forward-model primitive to one of four oracles, and run the winner once per instance. What follows is what changes when the recovered object is a _trajectory_ rather than a set.

#### 2.4.1 The objective

Published methods solve, per instance, for one observation `O` on one graph:

$$\hat{Y} \;=\; \arg\max_{Y} \; p(Y \mid O, G, M), \qquad Y = (s_0, s_1, \dots, s_T)$$

with `M` supplied by a hand-set `β` (NetFill, DHREC, NETSLEUTH), a mean-field estimate off the snapshot (DITTO), or a fit on other cascades (Farajtabar/NETRATE). We instead search over decoders. Let $\mathcal{A}$ be the space of programs expressible in the agent's contract, each

$$A : (G,\; O,\; T) \;\longmapsto\; \hat{Y}$$

and let $A_f$ denote $A$ with its transition primitive bound to oracle $f$ (§2.5.2). Over labelled episodes $\mathcal{D}_{\text{train}} = \{(G^{(i)}, O^{(i)}, Y^{(i)})\}$ obtained by masking our own stored trajectories:

$$A^{*} \;=\; \arg\max_{A \in \mathcal{A}} \; \frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{i} \operatorname{Score}\!\left(A_{f_\theta}\!\left(G^{(i)}, O^{(i)}, T\right),\; Y^{(i)}\right)$$

The choice of $\operatorname{Score}$ is not a detail. It is the single most consequential decision in this design and §2.6 is devoted to it.

#### 2.4.2 Why the inner loop runs hotter here

Source localization's program tests $C$ candidate source sets per instance, each one rollout. A trajectory decoder tests candidate _histories_, and the history space is exponentially larger. DITTO's answer is Metropolis-Hastings MCMC with a learned proposal at $O(T(n \log n + m))$ per sample [verified]. If a decoder draws $S$ samples and each touches $T$ steps, kernel evaluations per instance are $O(S \cdot T)$.

| Task                       | Oracle calls per instance        | Rough magnitude                |
| -------------------------- | -------------------------------- | ------------------------------ |
| Source localization        | $C$ candidate source sets        | $10^2$                         |
| **Cascade reconstruction** | $S \cdot T$ trajectory proposals | $10^4$ at $S = 10^3$, $T = 10$ |

Multiply by $P$ programs and $M$ instances and the feasibility argument for a learned kernel is two orders of magnitude stronger than it is for source localization. **This is the task where the world model most clearly earns its place**, and it is worth saying plainly that this is a program-search-time claim rather than an inference-time one, exactly as in [`source_localization.md`](source_localization.md) §2.3.2.

#### 2.4.3 What a generated decoder looks like

The space is genuinely wide, which is the argument for searching it rather than picking. Four families the agent can express, all of them published positions:

| Family                                 | Sketch                                                                                                                                     | Published relative                                              |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| **Greedy backward decode**             | from the observed snapshot, at each step pick the $s_t$ maximizing $p(s_{t+1} \mid s_t)$ under the frozen head, subject to IC monotonicity | the cheap first cut of §2.11                                    |
| **Forward-filter / backward-sample**   | propagate marginals forward under the kernel, sample a consistent history backward                                                         | classical smoothing; nobody in §3 does it with a learned kernel |
| **MCMC with a hand-written proposal**  | Metropolis-Hastings over histories, proposal written as code rather than learned                                                           | DITTO, minus the learned $Q_\theta$                             |
| **Steiner-tree / ordering heuristics** | build a minimal consistent tree over observed reports, then fill times                                                                     | Xiao SDM'18 `OrderedSteinerTree`, `delayed-bfs`                 |

A mid-search candidate combines them, and that is the point: a decoder that runs `delayed-bfs` for an initialization and then refines it by a few hundred MCMC steps scored under $f_\theta$ is not a paper anyone has written, and it is one program in $\mathcal{A}$.

### 2.5 The contract the agent writes

#### 2.5.1 The strategy method

```python
def reconstruct(self, graph: GraphInfo, observation: Observation, horizon: int) -> dict[int, tuple[int, int | None]]: ...
```

The return is `node -> (activation timestep, inferred parent)`, with `parent = None` marking a source and uninfected nodes simply absent. That single object carries everything §8.1's metric suite scores: `(node, t)` pairs give Event F1 and infection-time MAE, and the parent assignment gives path precision, Jaccard and order accuracy.

**Note what falls out.** The subset of nodes with `parent = None` _is_ the recovered source set, so [`source_localization.md`](source_localization.md)'s task is the projection of this one, and the same `PR / RE / F1 / AUC` can be computed from a reconstruction without any extra machinery. The two tasks share a contract, and a decoder is strictly more general than a localizer.

`Observation` is a small dataclass covering the four regimes of §2.7, all of which are masks over data already on disk:

| Field                              | Meaning                                                                                                     |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `reported: dict[int, int \| None]` | observed nodes; value is the observed activation time, or `None` for "known infected, time unknown"         |
| `final_state: np.ndarray \| None`  | the terminal snapshot, when that is all that is given (DITTO's DASH setting)                                |
| `visible: np.ndarray \| None`      | node mask for the hidden-node regime, where the node is absent from the adjacency and not merely unobserved |

#### 2.5.2 The primitives, and their four bindings


> **Update 2026-09-04.** Generated programs are offline: the bindings described below are attached to canned library baselines only, and the four bindings now describe the HARNESS's oracle, which computes the reward and the feedback. The text below is the original design record.

Source localization needed one primitive, $x \mapsto$ marginals. A decoder needs the raw kernel, evaluated on **arbitrary proposed states** rather than only on seed sets:

```python
def step_marginals(graph, state, diffusion_model) -> np.ndarray:      # shape: (N,) P(newly infected at t+1)
def transition_logprob(graph, state, next_state, diffusion_model) -> float
```

`transition_logprob` derives from `step_marginals`, so there is one kernel evaluation behind both. Under the world-model binding each is a single forward pass of `ICTransmissionHead`, which is what it already computes.

**The sampling bindings are implementable, and this was not obvious.** Evaluating the kernel at an arbitrary mid-cascade state requires putting NDlib _into_ that state, which is not something a fresh simulation does. `data/wm_simulator.py::Simulator` already exposes `snapshot() -> tuple[dict, int]` and `restore(snapshot)`, built for the counterfactual forks the generator writes, so arms 4 and 5 are `restore` plus `advance_marginal(bag=[], num_mc=R)`. No new simulator work.

| Binding        | `step_marginals` resolves to                                                                                 |
| -------------- | ------------------------------------------------------------------------------------------------------------ |
| `@native`      | **absent.** The decoder must be a pure structural/temporal heuristic: Steiner trees, BFS orderings, PageRank |
| `@monte_carlo` | `restore` + `advance_marginal` with `R` draws                                                                |
| `@oracle`      | the same, treated as ground truth                                                                            |
| `@world_model` | $f_\theta$, one forward pass                                                                                 |

### 2.6 The reward, and the metric-gaming hazard

**This is the section to read twice.** In source localization the outer-loop reward was the obvious one, F1 against the true source set, and the only subtlety was preferring it over re-simulation error. Here the obvious reward is actively dangerous.

§8.1 and the thirteen years between Zong ICDM'12 and DIPT 2025 say the same thing: **the node set is easy and the tree is hard.** Zong reports `prec_v = 100%` alongside `prec_e = 78-86%` [verified]; DIPT's best path precision across five graphs is `0.680` and its worst is `0.421` [verified, Table 1], while its source-localization F1 on the same graphs runs `0.518-0.839`. Our own eval learned the identical lesson in a different guise, which is why we report `delta_f1` rather than `infected_acc`.

For a human writing one decoder, reporting only node-level numbers is a presentation failure: §9 already warns that we "will look excellent and say nothing". **For a program search it is worse than a presentation failure, because the reward is the specification.** An outer loop rewarded on Event F1 will discover that recovering the infected set is nearly free and that the tree contributes nothing to its score, and it will converge on decoders that do not even attempt the hard half. The search does not merely report the easy metric; it optimizes for it and discards the capability.

So the reward must be tree-weighted:

$$\operatorname{Score}(\hat{Y}, Y) \;=\; \lambda \cdot \text{PathPrecision} \;+\; (1-\lambda)\cdot \text{EventF}_1, \qquad \lambda \geq 0.5$$

with infection-time NRMSE reported alongside but kept out of the reward, since it is on a different scale and DITTO's tables already interleave `F1↑` with `NRMSE↓` in a way that inverts rankings when mis-read (§8.2 trap 5).

**And that requires a ground-truth parent, which we do not have.** §9 item 9 establishes it: NDlib's `IndependentCascadesModel.iteration` sets `actual_status[v] = 1` on a successful coin flip **without recording which `u` caused it** [verified, read from the installed source]. So the `parents` field is not a nice-to-have that unlocks extra metrics later. It is a **precondition for the outer loop to have a correct objective at all**, and it moves to the front of the build order (§2.10).

Two properties of that ground truth, carried from §9 item 9 and both needing to be documented wherever the numbers are: NDlib iterates spreaders in node order and skips any `v` already flipped this step, so the recorded parent is _the first successful `u` in node order_ rather than a uniformly random one among the successes, which is a real and documentable bias. And **LT has no transmission edge at all**: activation is a threshold crossing over the whole active neighbourhood, so the honest ground truth there is a parent _set_. The tree half of this task is IC-only; under LT the reward falls back to Event F1 plus a set-valued parent score.

### 2.7 Which settings we can run today, and which we cannot

| Setting                                       | Can we?                                     | Why                                                                                                                                                                                           |
| --------------------------------------------- | ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **(a) `G` known + partial node observations** | yes **today**                                | The default. Mask nodes from `state`, keep `G`. This is the Xiao/Sadikov/NetFill regime.                                                                                                      |
| **(c-i) Timestamps observed**                 | yes **today**                                | `t` is on every record; masking a _subset_ of `(node, t)` pairs gives Xiao's `OrderedSteinerTree` input exactly.                                                                              |
| **(c-ii) Final state only**                   | yes **today**, and it is the interesting one | Keep only the last record's `next_state`. This is DITTO's DASH setting: the hardest published formulation, and the one where a learned kernel should beat a hand-set `β`.                    |
| **(d) Hidden nodes**                          | Warning: needs a loader flag                      | Delete nodes from the _adjacency_, not just the observation. `reconstruct_episode_adjacency` already rebuilds per-episode adjacency, so this is a mask argument, not new machinery. Untested. |
| **(b) `G` unknown**                           | no out of scope                             | That is network inference: `G` flips from condition to variable and our encoder has nothing to condition on. Reviewed and NOT pursued (see `research/README.md`).                            |

### 2.8 Metrics we would report

Directly on top of `wm_metrics.py`, which already computes per-node precision/recall/F1 against binary and soft targets:

| Metric                                  | Definition                                                                      | Precedent                                                                      |
| --------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **Event F1**                            | macro-F1 over `(node, t)` infection events in the reconstructed vs true history | DITTO's `F1` [verified] · DIPT's source-localization `F1`                      |
| **Infection-time MAE / NRMSE**          | error on hitting times `h_u(Y)`                                                 | DITTO's `NRMSE`: normalized by `2n(T+1)²` under the root [verified]           |
| **Tree edit distance / path precision** | fraction of who-infected-whom edges recovered                                   | DIPT's `Path Precision` + `Jaccard Index` [verified] · Xiao's `order accuracy` |
| **Trajectory likelihood**               | `Σ_t log p(ŝ_{t+1} \| ŝ_t, G)` under our own head                               | no direct precedent; it is what our structured head uniquely affords           |
| **MCC**                                 | for heavily class-imbalanced masks                                              | Rozenshtein's only reported measure [verified]                                 |

Note the trap our existing eval already documents: `infected_acc` is dominated by unchanged nodes and looks great for free. The substantive numbers are `delta_f1`-shaped (scored on the _changes_) which is what Event F1 above is.

Under the formulation of §2.4, **Path Precision and Event F1 are the reward** (weighted per §2.6) and everything else in that table is reported but not optimized. Trajectory likelihood deserves a note: it is the one metric nobody in this literature can compute, because it needs a kernel to evaluate under, and it is therefore the natural _internal_ signal for a decoder to use on unlabelled data. The same split as source localization applies. Labels select the program; the program itself runs on likelihood and structure alone, so `A*` is deployable on a logged trajectory with no ground truth.

### 2.9 The six baseline conditions

Identical in structure to [`source_localization.md`](source_localization.md) §2.6. Arms 3 through 6 hold the generated decoder fixed and vary only the binding of §2.5.2.

| #     | Arm                  | What it is                                                                         | What it isolates                                              |
| ----- | -------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| **1** | Pure algorithm       | `delayed-bfs` / `OrderedSteinerTree` (Xiao SDM'18), NetFill, Personalized PageRank | the no-learning floor                                         |
| **2** | GA routing           | routing over that fixed pool                                                       | is decoder _generation_ worth more than decoder _selection_?  |
| **3** | agent `@native`      | agent writes structural/temporal heuristics; no kernel available                   | **is a transition kernel in the decode loop worth anything?** |
| **4** | agent `@monte_carlo` | same decoders, `restore` + `advance_marginal` sampling                             | the cost axis, at $10^4$ oracle calls per instance            |
| **5** | agent `@oracle`      | the same, treated as exact                                                         | the fidelity ceiling                                          |
| **6** | agent `@world_model` | same decoders, $f_\theta$                                                          | **ours**                                                      |
| **A** | ablation             | §2.3's framing: DITTO's decoder with our kernel substituted for `β̂`                | what decoder _search_ buys over the best published decoder    |

**4 vs 6 is the sharpest comparison in either file.** At $10^4$ kernel calls per instance (§2.4.2), the Monte Carlo arm cannot reach the same $P$ within any realistic budget, so the axis is not "which scores higher" but "how much search fits". Arm A is a stronger control than source localization's, since DITTO beats a supervised model trained with the true `β` on two of eight rows (§5.1.1) and is genuinely hard to beat.

### 2.10 What has to be built

Ordered by dependency. The first item is a precondition rather than an enhancement, per §2.6.

| #     | Piece                                                                                                                                                                                         | Where                                         | Est.                                                |
| ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- | --------------------------------------------------- |
| **1** | **`parents` ground truth.** Subclass `IndependentCascadesModel`, override `iteration` to log `(u, v)` when `flip <= threshold`, thread it through `Simulator.advance` into a new record field | `data/wm_simulator.py`, `generate_wm_data.py` | **~1 day, and nothing tree-level works without it** |
| 2     | `step_marginals` / `transition_logprob` primitives plus four bindings; the sampling arms are `restore` + `advance_marginal`                                                                   | `coding_agent/tools/primitives.py`, both envs | ~80 lines                                           |
| 3     | `reconstruct` on the `Strategy` Protocol, the `Observation` dataclass, executor presence check                                                                                                | `coding_agent/types.py`, `executor.py`        | ~60 lines                                           |
| 4     | Tree-weighted reward per §2.6, replacing final spread for this task                                                                                                                           | `coding_agent/methods/base.py`                | ~50 lines                                           |
| 5     | Masked-episode harness: group `transitions_*.jsonl` by `episode_id`, sort by `t`, apply a mask, emit `(G, O, Y)`                                                                              | `world_model/wm_data.py`                      | ~80 lines                                           |
| 6     | Metrics: Event F1, Path Precision, Jaccard, order accuracy, infection-time NRMSE, trajectory log-likelihood                                                                                   | `world_model/wm_eval.py`                      | ~90 lines                                           |
| 7     | Prompt scaffolding: the `reconstruct` contract, the two primitives, the four decoder families of §2.4.3 as worked examples                                                                    | `coding_agent/prompts.py`                     | ~50 lines                                           |
| 8     | **Arm A**: DITTO-style decoder with our kernel                                                                                                                                                | `world_model/wm_reconstruct.py`               | ~250 lines                                          |
| 9     | Registry entry: flip `cascade_reconstruction` from `planned`                                                                                                                                  | `pipeline/tasks.py`                           | ~10 lines                                           |
| 10    | Bump `--mc-marginals` on the eval split only, per §9 item 7                                                                                                                                   | `pipeline/run.py`                             | ~5 lines                                            |

### 2.11 Honest cost and honest risks

**Cost.** Item 1 is a day of simulator work. Items 2 through 7 and 9 through 10 are roughly 3 to 5 days. Item 8 is the expensive control: a greedy backward decode is a day, but a defensible DITTO-parity implementation is weeks, and §2.9 needs the control to be good rather than convenient. Budget **1 week for a first table, and treat arm A's fidelity as the schedule risk.**

**Risk 1: the reward specifies the task, and the easy half is nearly solved.** §2.6 in full. Zong's `prec_v = 100%` is the warning: a search rewarded on node-level metrics will find that the infected set is almost free and stop there. Weight the reward toward Path Precision, and confirm before running the full search that a trivial decoder (predict everyone reachable, assign parents by BFS) scores badly under the chosen $\operatorname{Score}$. If it does not, the reward is wrong.

**Risk 2: no tree ground truth exists until item 1 is done, and never exists for LT.** The bias in NDlib's parent attribution (first successful `u` in node order) and LT's parent-set semantics both need documenting alongside any number (§2.6).

**Risk 3: `--mc-marginals 30` is too low to serve as a reconstruction reference.** Farajtabar uses 400 to 500 samples, Xiao uses 1,000 sampled trees and reports marginal gains beyond that [verified, §8.2 trap 6]. Thirty is fine for training targets and will read as noise in a masked-episode ground truth. Bump it on the eval split only.

**Risk 4: the simulated-to-real cliff, which is steeper here than anywhere else in this folder.** DITTO's Table 4 has GCN and GIN collapsing to `F1 ≈ 0.32` on real diffusion after being fine on synthetic [verified]. Our kernel is trained on NDlib transitions. Any reconstruction claim is a claim about simulated dynamics until tested on a logged trajectory, which is exactly the "real-world data = logged trajectories" scope already agreed.

**Risk 5: assortativity lets a trivial baseline win.** Xiao ICDM'18 found Personalized PageRank beats tree sampling on `grqc` (assortativity 0.164) and loses elsewhere [verified]. Our suite contains exactly that graph. Run PageRank as arm 1 on `ca_grqc` specifically; if the search cannot beat it there, the result is not real.

**Risk 6: arm A is hard to beat and easy to strawman.** DITTO outscores a supervised model trained with the true `β` on BA-SIR and ER-SIR (`Gap` −3.49% and −32.41% on NRMSE [verified]). A weak arm A makes 6-vs-A meaningless. This is the honest reason the cost estimate is a week rather than two days.

**Risk 7: zero action ops exercised.** Actions are `NULL` throughout and `T_exo = identity`, so this task adds no coverage of the four idle ops, exactly as in source localization. The analogous fix is the same: an _active_ reconstruction variant where the decoder chooses which nodes to query, which turns the observation mask into an action. Same trade as [`source_localization.md`](source_localization.md) §2.8, and the same recommendation to defer it.

---

## 3. Classical and heuristic methods

Three families, distinguished by what they optimize: **statistical correction** (fit a generative cascade shape, invert the sampling bias), **combinatorial** (find the minimum structure consistent with the observations, almost always a Steiner-tree variant), and **MLE** (maximize the likelihood of the history under an assumed diffusion model).

| Method                           | Year        | Venue       | Family                    | Idea                                                                                                                                                                                                                                                                                                                                         | Paper                                                                                                                                        | Code                                                                                                                                                                                        |
| -------------------------------- | ----------- | ----------- | ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **NetInf**                       | 2010        | KDD         | tree/MLE                  | Submodular greedy over the most likely propagation _trees_; recovers `G` from many cascades. The per-cascade tree machinery is identical to reconstruction: see network inference.                                                                                                                             | [arXiv 1006.0234](https://arxiv.org/abs/1006.0234)                                                                                           | [SNAP netinf](https://snap.stanford.edu/netinf/)                                                                                                                                            |
| **Effectors**                    | 2010        | KDD         | combinatorial             | Given a partially activated graph, find `k` nodes that best "explain" the activation. The static ancestor of retrospective reconstruction.                                                                                                                                                                                                   | [ACM 10.1145/1835804.1835882](https://dl.acm.org/doi/10.1145/1835804.1835882) (403 to bots)                                                  | no public code found                                                                                                                                                                        |
| **Sadikov k-tree** (key)            | 2011        | WSDM        | statistical               | Model the cascade as a `k`-tree `Γ(p,b,h,k)`; derive closed forms for how size/edges/isolated-nodes/components degrade under uniform sampling; invert to estimate the _complete_ cascade's properties.                                                                                                                                       | [Stanford PDF](https://cs.stanford.edu/~jure/pubs/cascades-wsdm11.pdf) · [ACM](https://dl.acm.org/doi/10.1145/1935826.1935861) (403 to bots) | no public code found                                                                                                                                                                        |
| **NETRATE**                      | 2011        | ICML        | MLE                       | Continuous-time transmission rates `α_ji` by convex MLE over cascades. Supplies the `p(y\|x,G)` that Farajtabar then inverts.                                                                                                                                                                                                                | [arXiv 1105.0697](https://arxiv.org/abs/1105.0697)                                                                                           | [Networks-Learning/netrate](https://github.com/Networks-Learning/netrate)                                                                                                                   |
| **NETSLEUTH**                    | 2012        | ICDM        | MDL                       | MDL-optimal `(k, seed set)` from a single SI snapshot; eigenvector-based ranking. The standard "how many and which ones" baseline.                                                                                                                                                                                                           | [GaTech PDF](https://faculty.cc.gatech.edu/~badityap/papers/netsleuth-icdm12.pdf) · [IEEE](https://ieeexplore.ieee.org/document/6413786)     | no public code found                                                                                                                                                                        |
| **Consistent trees (WPCT/WBCT)** | 2012        | ICDM        | combinatorial             | Inferred cascade = a tree connecting sources to observed nodes with paths satisfying **temporal constraints**. Decision problems NP-complete, optimization hard to approximate; gives approximation algorithms + heuristics.                                                                                                                 | [arXiv 1210.3587](https://arxiv.org/abs/1210.3587)                                                                                           | no public code found                                                                                                                                                                        |
| **DHREC**                        | 2014 → 2016 | ICDM → KAIS | MLE                       | "Diffusion archaeology": reduce history-reconstruction MLE to **Prize-Collecting Dominating Set Vertex Cover**, solve greedily. Handles SEIR. Requires known diffusion parameters. **DITTO's MLE baseline.**                                                                                                                                 | [CMU PDF](https://www.cs.cmu.edu/~ckingsf/software/dhrec/icdm2014.pdf) · [KAIS](https://link.springer.com/article/10.1007/s10115-015-0904-x) | [project page](http://www.cs.cmu.edu/~ckingsf/research/3cde/paper.html)                                                                                                                     |
| **Back to the Past** (key)          | 2015        | AISTATS     | MLE + importance sampling | Two stages: learn a continuous-time diffusion network (NETRATE), then maximize the _incomplete-cascade_ likelihood over `(source, start time)`, a high-dimensional integral over hidden infection times, approximated by importance sampling with a shortest-path proposal. Piecewise-unimodal in `t_s`, so line search is exact per piece. | [arXiv 1501.06582](https://arxiv.org/abs/1501.06582) · [PMLR v38](http://proceedings.mlr.press/v38/farajtabar15.pdf)                         | no public code found                                                                                                                                                                        |
| **NetFill**                      | 2015        | SDM         | MDL                       | Finds missing _nodes_ in a partially observed SI epidemic; assumes a single global `β` and a known observed-fraction. Binary output (no per-node probability).                                                                                                                                                                               | [DOI 10.1137/1.9781611974010.47](https://doi.org/10.1137/1.9781611974010.47) (403 to bots)                                                   | no public code found                                                                                                                                                                        |
| **CulT** (key)                      | 2016        | KDD         | combinatorial             | `α`-**TempSteinerTree** over a _temporal_ network: a forest of ≤`k` temporal Steiner trees spanning the reports, with `α` trading tree cost against seed count and binary-searched to hit a target `k`. Makes **no assumption about the propagation model** and works on interaction streams, not a static `G`.                              | [KDD PDF](https://www.kdd.org/kdd2016/papers/files/rpp0920-rozenshteinAT3.pdf) · [ACM](https://dl.acm.org/doi/10.1145/2939672.2939865)       | [polinapolina/reconstructing-an-epidemic-over-time](https://github.com/polinapolina/reconstructing-an-epidemic-over-time)                                                                   |
| **CRI**                          | 2016        | TNSE        | heuristic                 | Clustering + reverse infection under SIR; estimates infection times but **not** recovery times. **DITTO's second MLE baseline.**                                                                                                                                                                                                             | [DOI 10.1109/TNSE.2016.2523804](https://doi.org/10.1109/TNSE.2016.2523804)                                                                   | no public code found                                                                                                                                                                        |
| **OJC / AJC**                    | 2017        | AAAI        | combinatorial             | Optimal-Jordan-Cover: candidate selection by infected-neighbour count, then the min-radius set covering all observed infections. Provably locates all sources w.p.→1 for heterogeneous SIR on ER graphs **with partial observations**. `AJC` = K-Means approximation.                                                                        | [arXiv 1611.06963](https://arxiv.org/abs/1611.06963) · [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/10746)                        | no public code found                                                                                                                                                                        |
| **OrderedSteinerTree** (key)        | 2018        | SDM         | combinatorial             | Reconstruct from **reported nodes + their activation times**: smallest tree spanning the reports in which every rooted path respects the observed order. `closure` gives `O(√k)`; `delayed-bfs` gives `k`-approx in `O(m + k log k)`, linear-time and the scalable option.                                                                  | [arXiv 1801.08586](https://arxiv.org/abs/1801.08586)                                                                                         | [xiaohan2012/reconstructing-cascade](https://github.com/xiaohan2012/reconstructing-cascade)                                                                                                 |
| **Tree-sampling (LERW)** (key)      | 2018        | ICDM        | probabilistic             | Instead of one tree, **sample** Steiner trees proportional to `w(T)` via loop-erased random walk / cycle-popping, then read off per-node **marginal** infection probabilities. Robust to noise; the only classical method here that outputs calibrated probabilities rather than a binary set.                                               | [arXiv 1809.05812](https://arxiv.org/abs/1809.05812)                                                                                         | [cascade-reconstruction-by-tree-samples](https://github.com/xiaohan2012/cascade-reconstruction-by-tree-samples) · [random_steiner_tree](https://github.com/xiaohan2012/random_steiner_tree) |
| **Partial-timestamp MLE**        | 2019        | TKDE        | MLE                       | Full diffusion history from _partial_ timestamps; MLE on trees, Gromov-matrix optimization heuristic for general graphs, plus a multi-source variant.                                                                                                                                                                                        | [DOI 10.1109/TKDE.2019.2905210](https://doi.org/10.1109/TKDE.2019.2905210) (32(7):1378-1392)                                                 | no public code found                                                                                                                                                                        |
| **Risk-aware reconstruction**    | 2022        | KAIS        | combinatorial             | Temporal cascade reconstruction targeted at surfacing **asymptomatic** (never-reported) cases; risk-weighted objective.                                                                                                                                                                                                                      | [Springer](https://link.springer.com/article/10.1007/s10115-022-01748-8)                                                                     | no public code found                                                                                                                                                                        |
| **PoolMLE**                      | 2026        | arXiv       | MLE                       | Reconstruction under **group surveillance** (wastewater/pooled testing): a negative pool clears many individuals, a positive pool identifies none. New observation model, not just a new solver.                                                                                                                                             | [arXiv 2602.11419](https://arxiv.org/abs/2602.11419)                                                                                         | no public code found                                                                                                                                                                        |

**Theory, not algorithms**: three results that bound what any of the above can do:

- **Learning from Contagion (Without Timestamps)**, Amin, Heidari & Kearns, ICML 2014: learn the contagion structure when you see only `(seed set, final infected set)`, no vertex-by-vertex timestamps. Exact, efficient algorithms for **trees**; empirically good on sparse graphs. [PMLR v32](https://proceedings.mlr.press/v32/amin14.html) · [UPenn PDF](https://www.cis.upenn.edu/~mkearns/papers/LearningFromContagion.pdf). No public code found. This is the formal statement of our setting (c-ii): final-state-only is learnable, but the guarantees stop at trees.
- **Learning Influence Functions from Incomplete Observations**, He, Xu, Kempe & Liu, NeurIPS 2016: proper PAC-learnability of influence functions under randomly missing activations for DLT/DIC; improper for CIC. [arXiv 1611.02305](https://arxiv.org/abs/1611.02305) · [NeurIPS PDF](https://proceedings.neurips.cc/paper/2016/file/68b1fbe7f16e4ae3024973f12f3cb313-Paper.pdf). No public code found.
- **Network forensics: random infection vs spreading epidemic**, Milling, Caramanis, Mannor & Shakkottai, SIGMETRICS 2012, the prior question: is there a cascade at all? Distinguishes contact-spread from independent random illness using only local information, no full topology, tolerant to false positives. Relevant as a null model for any reconstruction claim.

---

## 4. Learning-based methods

Four groups. The last one (**methods that put a learned forward model in the inner loop**) is the direct analogue of what we would build, and is called out separately in §4.2.

### 4.1 The catalogue

| Method                          | Year        | Venue           | Approach                                                                                                                                                                                                                                                                                                                                                     | Paper                                                                                                                                                                                           | Code                                                                                                                                                                        |
| ------------------------------- | ----------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **BRITS**                       | 2018        | NeurIPS         | Bidirectional RNN for multivariate time-series imputation. Graph-agnostic; DITTO's weakest supervised baseline and OOMs on `Pol` [verified].                                                                                                                                                                                                                 | [arXiv 1805.10572](https://arxiv.org/abs/1805.10572)                                                                                                                                            | [caow13/BRITS](https://github.com/caow13/BRITS)                                                                                                                             |
| **GRIN**                        | 2022        | ICLR            | "Filling the G_ap_s": bidirectional GNN + recurrent imputation on **graph** time series. DITTO's strongest supervised baseline, and the one used as the _ideal_ upper bound when trained with true `β` [verified].                                                                                                                                           | [arXiv 2108.00298](https://arxiv.org/abs/2108.00298)                                                                                                                                            | [Graph-Machine-Learning-Group/grin](https://github.com/Graph-Machine-Learning-Group/grin)                                                                                   |
| **SPIN**                        | 2022        | NeurIPS         | Sparse spatiotemporal **attention** for imputation; beats GRIN on small graphs, OOMs on `Oregon2`/`Prost`/`Pol` [verified].                                                                                                                                                                                                                                  | [arXiv 2205.13479](https://arxiv.org/abs/2205.13479)                                                                                                                                            | [Graph-Machine-Learning-Group/spin](https://github.com/Graph-Machine-Learning-Group/spin)                                                                                   |
| **SL-VAE**                      | 2022        | KDD             | VAE over seed sets + a learned propagation model; the forward model is pluggable (GAT / MONSTOR / DeepIS). Reconstructs sources, not trajectories.                                                                                                                                                                                                           | [KDD](https://dl.acm.org/doi/10.1145/3534678.3539267) (403 to bots)                                                                                                                             | [triplej0079/SLVAE](https://github.com/triplej0079/SLVAE)                                                                                                                   |
| **DDMIX / Deep Demixing**       | 2021 → 2023 | EUSIPCO → TSIPN | Conditional-VAE GNN that "demixes" a single aggregated snapshot into node states at **all** `T` steps. Accuracy degrades as `T` grows because the solution space blows up [claim, per DIPT §1].                                                                                                                                                              | [arXiv 2011.09583](https://arxiv.org/abs/2011.09583) · [journal version arXiv 2306.07938](https://arxiv.org/abs/2306.07938)                                                                     | [gojkoc54/Deep_demixing](https://github.com/gojkoc54/Deep_demixing)                                                                                                         |
| **DDMSL**                       | 2023        | NeurIPS         | Discrete **diffusion model** for graph inverse problems: forward SIR as a Markov chain, reversible residual GCN blocks for denoising; does source localization **and** diffusion-path reconstruction. Does **not** output the propagation tree.                                                                                                              | [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/46ab9d9645b6975b947231ddb48da1ab-Abstract-Conference.html) · [OpenReview](https://openreview.net/forum?id=5Fr8Nwi5KF) | no public code found                                                                                                                                                        |
| **PGSL**                        | 2024        | ESWA            | Probabilistic graph **diffusion** model for source localization; same family as DDMSL.                                                                                                                                                                                                                                                                       | [DOI 10.1016/j.eswa.2023.122028](https://doi.org/10.1016/j.eswa.2023.122028)                                                                                                                    | no public code found                                                                                                                                                        |
| **DITTO** (key)                    | 2023        | KDD             | **DASH**: reconstruct history from **a single snapshot**, with neither known diffusion parameters nor known sources. Reduces to estimating posterior expected hitting times, solves by Metropolis, Hastings MCMC with an _unsupervised GNN-learned proposal_; a **barycenter** objective replaces unstable MLE. `O(T(n log n + m))`.                          | [arXiv 2306.00488](https://arxiv.org/abs/2306.00488) · [ACM](https://dl.acm.org/doi/abs/10.1145/3580305.3599488)                                                                                | [q-rz/KDD23-DITTO](https://github.com/q-rz/KDD23-DITTO)                                                                                                                     |
| **SIDSL**                       | 2025        |                 | Structure-prior informed diffusion model for source localization with limited data; sibling of DDMSL.                                                                                                                                                                                                                                                        | [arXiv 2502.17928](https://arxiv.org/abs/2502.17928)                                                                                                                                            | no public code found                                                                                                                                                        |
| **DIPT** (key)                     | 2025        |                 | **Deep Identification of Propagation Trees.** First method that outputs the _explicit_ who-infected-whom edges rather than per-step node states: cross-attention influence scores between node pairs + a VAE prior over seeds, optimized by alternating latent inference. Authors are the Emory group (Memon, Ling, Kong, Seshagiri, Zufle, **Liang Zhao**). | [arXiv 2503.00646](https://arxiv.org/abs/2503.00646)                                                                                                                                            | no public code found ([anonymous 4open.science link](https://anonymous.4open.science/r/Compartmental-Infectious-Disease-Simulation-4F7C) is the _simulator_, not the model) |
| **Distribution Classification** | 2025→2026   | ICLR            | Learn spreading-model **parameters** when node statuses are unobservable, using observable proxy indicators; beats ABC and GNN baselines. Parameter recovery, not trajectory recovery.                                                                                                                                                                       | [arXiv 2505.11228](https://arxiv.org/abs/2505.11228)                                                                                                                                            | no public code found                                                                                                                                                        |

**Adjacent measurement work** (not reconstruction methods, but they quantify why reconstruction is needed): _Phantom cascades_, hidden nodes cause systematic under-estimation of cascade size across five datasets ([arXiv 1502.01602](https://arxiv.org/abs/1502.01602), no public code found); _How the cascade inference problem distorts information diffusion_ ([arXiv 2410.21554](https://arxiv.org/abs/2410.21554), no public code found).

### 4.2 The group that matters to us: a learned forward model in the inner loop

Three methods invert a _learned_ propagation operator rather than a hand-set one. This is exactly our position, and the difference is where the operator comes from:

| Method                      | Forward model                              | How it is obtained                                                                        | Inversion                                        |
| --------------------------- | ------------------------------------------ | ----------------------------------------------------------------------------------------- | ------------------------------------------------ |
| **Back to the Past** (2015) | continuous-time transmission rates `α_ji`  | NETRATE, fit on **other historical cascades** of the same network                         | importance-sampled MLE over `(source, t_s)`      |
| **DITTO** (2023)            | SI/SIR with `β̂`                            | mean-field estimate from the **single observed snapshot**                                 | MCMC with learned proposal, barycenter objective |
| **DIPT** (2025)             | pairwise influence score `MLP([h_u, h_v])` | learned end-to-end from cascades                                                          | alternating optimization of a latent seed vector |
| **ours (proposed)**         | `q(u→v) = sigmoid(MLP([h_u, h_v, w_uv]))`  | **already trained** on `(G, s_t, a_t, s_{t+1})` transitions with MC-marginal soft targets | **searched, not hand-written**: §2.4            |

Two things fall out of this table. First, **our forward model is strictly better supervised than any of theirs**: DITTO gets one scalar `β̂` per graph, DIPT learns influence scores with no per-step supervision, and we train per-edge transmission against MC-estimated `P(infected)` marginals at every step. Second, **they all had to build the inversion**, and so do we, that is the whole remaining cost, and none of the three released a reusable inversion component (DITTO's repo is the only public code in the group).

---

## 5. Published results

**Read this first: only four papers in this literature publish result tables at all.** DITTO, DIPT, Sadikov and Zong do. Rozenshtein (KDD'16), Xiao (SDM'18), Xiao (ICDM'18) and Farajtabar (AISTATS'15) publish **figures only**, verified by full-text extraction; `grep -c "Table" rozenshtein.txt` returns **0**. Their sections below are marked [figure] throughout and give directions, not cells.

### 5.1 DITTO (KDD 2023) (key) the most comparable table

**Why it is the reference point:** it reconstructs a full `T`-step history, it reports `F1` on the reconstructed states and `NRMSE` on hitting times (both of which we can compute from what we already store) and **two of its four synthetic rows are BA and ER graphs we generate natively** (`--dataset ba`, `--dataset er`).

Protocol [verified, §5.1.1 + Appendix D.1]: BA (`n=1,000`, attachment 4) and ER (`n=1,000`, `p=0.008`); SI and SIR for `T=10`, infection rate 0.1, recovery rate 0.1, **5% of nodes as sources**. Oregon2/Prost: `T=15`, infection 0.1, recovery 0.05, **10% sources**. `NRMSE` is over infection _and_ recovery hitting times, normalized by `2n(T+1)²`.

Datasets [verified, Table 2]:

| Dataset   | #Nodes | #Edges | Timespan `T` | Graph     | Diffusion |
| --------- | ------ | ------ | ------------ | --------- | --------- |
| BA        | 1,000  | 3,984  | 10           | synthetic | synthetic |
| ER        | 1,000  | 3,987  | 10           | synthetic | synthetic |
| Oregon2   | 11,461 | 32,730 | 15           | real      | synthetic |
| Prost     | 15,810 | 38,540 | 15           | real      | synthetic |
| BrFarmers | 82     | 230    | 16           | real      | real SI   |
| Pol       | 18,470 | 48,053 | 40           | real      | real SI   |
| Covid     | 344    | 2,044  | 10           | real      | real SIR  |
| Hebrew    | 3,521  | 18,064 | 9            | real      | real SIR  |

#### 5.1.1 Synthetic SI/SIR vs the MLE baselines [verified, Table 5]

`Gap` is relative to `GRIN` trained with the _true_ `β` (the "ideal" row). `F1↑`, `NRMSE↓`.

| Type       | Method       | BA-SI F1  | NRMSE     | ER-SI F1  | NRMSE     | Oregon2-SI F1 | NRMSE     | Prost-SI F1 | NRMSE     |
| ---------- | ------------ | --------- | --------- | --------- | --------- | ------------- | --------- | ----------- | --------- |
| Ideal      | GRIN         | .8404     | .2123     | .8317     | .2166     | .8320         | .2249     | .8482       | .2155     |
| MLE        | DHREC        | .6026     | .4644     | .6281     | .4495     | .6038         | .4101     | .6558       | .4138     |
| MLE        | CRI          | .7502     | .3012     | .7797     | .2744     | .8183         | .2438     | .8083       | .2491     |
| Barycenter | **DITTO** (key) | **.8384** | **.2139** | **.8269** | **.2225** | **.8280**     | **.2289** | **.8327**   | **.2317** |

| Type       | Method       | BA-SIR F1 | NRMSE     | ER-SIR F1 | NRMSE     | Oregon2-SIR F1 | NRMSE     | Prost-SIR F1 | NRMSE     |
| ---------- | ------------ | --------- | --------- | --------- | --------- | -------------- | --------- | ------------ | --------- |
| Ideal      | GRIN         | .7867     | .1692     | .7626     | .2484     | .8024          | .1651     | .8067        | .1652     |
| MLE        | DHREC        | .5080     | .4722     | .5500     | .4423     | .6044          | .4478     | .6268        | .4326     |
| MLE        | CRI          | .5994     | .3356     | .6129     | .3109     | .5761          | .3576     | .5738        | .3406     |
| Barycenter | **DITTO** (key) | **.7783** | **.1633** | **.7734** | **.1679** | **.7928**      | **.1707** | **.7929**    | **.1690** |

**DITTO beats the "ideal" supervised model on BA-SIR and ER-SIR** (`Gap` −3.49% and −32.41% on NRMSE [verified]): an unsupervised MCMC method outscoring a supervised imputer trained on the true parameters. That is the strongest single claim in this literature.

#### 5.1.2 Real diffusion [verified, Table 4]

| Type       | Method    | BrFarmers F1 | NRMSE     | Pol F1    | NRMSE     | Covid F1  | NRMSE     | Hebrew F1 | NRMSE     |
| ---------- | --------- | ------------ | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| Supervised | GCN       | .5409        | .6660     | .4458     | .4946     | .3162     | .5214     | .3350     | .6070     |
| Supervised | GIN       | .4548        | .6565     | .5203     | .4767     | .3226     | .4951     | .3704     | .7816     |
| Supervised | BRITS     | .5207        | .3995     | OOM       | OOM       | .3524     | .5333     | .3120     | .6584     |
| Supervised | GRIN      | .8003        | .2425     | .6518     | .3731     | .5448     | .3040     | .5916     | .2212     |
| Supervised | SPIN      | **.8268**    | **.2084** | OOM       | OOM       | .5917     | .2932     | .5178     | .3330     |
| MLE        | DHREC     | .6131        | .4150     | .7023     | .3398     | .3540     | .6023     | .6251     | .4169     |
| MLE        | CRI       | .6058        | .4444     | .7468     | .2942     | .4170     | .5487     | .5344     | .3552     |
| Barycenter | **DITTO** | .8206        | .2142     | **.7471** | **.2903** | **.6240** | **.2637** | **.6411** | **.2983** |

Reading it: on **real** diffusion the supervised imputers collapse (GCN/GIN `F1 ≈ 0.32-0.54`) because they were trained on simulated SI/SIR that does not match reality, the same warning IMINFECTOR raises for IM in [`influence_maximization.md`](influence_maximization.md) §5.5, and a direct caution for us, since our world model is trained on NDlib transitions. BrFarmers is the one exception, and the paper says why: its dynamics are "very close to the SI model" [verified, §5.3].

### 5.2 DIPT (2025) (key) the only propagation-**tree** table, and it is Emory's

Protocol [verified, §4.1.1]: for Cora-ML, CiteSeer and Power Grid (**graphs we already load**) there is no real diffusion data, so they **simulate**: pick 10% of nodes as sources, run **SI for 200 iterations to convergence**. MemeTracker uses real cascades: top **583 sites, 6,700 cascades**, top 5% of nodes per cascade (earliest appearance) as sources. IDSS is their own US-county SIR mobility simulation (3,143 counties, `R0 = 1.2`, 90 days, ≈500-1,000 infected counties per run).

Propagation-tree identification [verified, Table 1]:

| Method      | Cora-ML PathPrec | Jaccard   | Memetracker PathPrec | Jaccard   | CiteSeer PathPrec | Jaccard   | Power Grid PathPrec | Jaccard   | IDSS PathPrec | Jaccard   |
| ----------- | ---------------- | --------- | -------------------- | --------- | ----------------- | --------- | ------------------- | --------- | ------------- | --------- |
| DDMIX       | 0.327            | 0.195     | 0.062                | 0.041     | 0.236             | 0.133     | 0.081               | 0.031     | 0.109         | 0.057     |
| DDMSL       | 0.412            | 0.259     | 0.119                | 0.063     | 0.405             | 0.253     | 0.130               | 0.069     | 0.121         | 0.064     |
| **DIPT** (key) | **0.622**        | **0.452** | **0.602**            | **0.430** | **0.593**         | **0.421** | **0.680**           | **0.515** | **0.421**     | **0.266** |

Source localization on the same graphs [verified, Table 2]: note the column order is `RE · PR · F1 · AUC`:

| Method   | Cora-ML F1 | AUC       | Memetracker F1 | AUC       | CiteSeer F1 | AUC       | Power Grid F1 | AUC       | IDSS F1   | AUC       |
| -------- | ---------- | --------- | -------------- | --------- | ----------- | --------- | ------------- | --------- | --------- | --------- |
| LPSI     | 0.301      | 0.592     | 0.014          | 0.529     | 0.306       | 0.598     | 0.474         | **0.934** | 0.037     | 0.540     |
| OJC      | 0.121      | 0.534     | 0.026          | 0.517     | 0.117       | 0.530     | 0.153         | 0.501     | 0.028     | 0.520     |
| GCNSI    | 0.401      | 0.687     | 0.035          | 0.422     | 0.387       | 0.680     | 0.330         | 0.639     | 0.045     | 0.430     |
| SL-VAE   | 0.764      | 0.831     | 0.488          | 0.624     | 0.749       | 0.825     | 0.797         | 0.879     | 0.494     | 0.630     |
| DDMIX    | 0.221      | 0.247     | 0.022          | 0.417     | 0.215       | 0.250     | 0.280         | 0.340     | 0.029     | 0.425     |
| DDMSL    | 0.750      | **0.873** | **0.515**      | **0.641** | 0.742       | 0.870     | **0.831**     | 0.866     | **0.527** | **0.645** |
| **DIPT** | **0.839**  | **0.881** | 0.518          | 0.629     | **0.832**   | **0.880** | 0.828         | 0.864     | 0.525     | 0.630     |

Partial supervision [verified, Table 3], with only 10/20/30% of the true propagation tree visible during training, `Path Precision` on Power Grid goes **0.683 → 0.718 → 0.759** (vs 0.680 fully unsupervised). A little tree supervision buys a lot.

Warning: **Correction to an earlier version of this line.** It previously read "we have 100% of the tree, for free". **We do not.** We have 100% of the _level structure_ for free (which node activated at which step, from the stored `frontier` channel), but **no parent edges at all**, because NDlib never records which `u` caused a successful flip (§9 item 9, verified against the installed source). Once the `parents` field of §2.10 item 1 is built we would have 100% of the tree, and DIPT's Table 3 is then the argument for why that is worth a day of simulator work: 30% tree supervision alone is worth `+0.079` Path Precision over none.

### 5.3 Sadikov et al. (WSDM 2011): the canonical missing-data reference

Data [verified, §5.1]: a Twitter follow graph of **71,804,410 nodes and 2,040,072,198 directed edges** (avg degree 28.4), crawled BFS from the public stream, June, December 2009 (Topsy). **250 retweet cascades** and **100 blog influence cascades** (Spinn3r, Aug, Nov 2008, English blogosphere, cascades of ≥100 nodes); synthetic runs use 1,000 cascades of 127 nodes each.

Relative error of _estimated_ (`ê`) vs _observed_ (`e′`) `k`-tree parameters at sample ratio `σ* = 0.5` [verified, Table 2: synthetic cascades on the Twitter network]:

| Param           | Network cascade `ê` | `e′` | Influence cascade `ê` | `e′` |
| --------------- | ------------------- | ---- | --------------------- | ---- |
| `p`             |                     |      | 0.02                  |      |
| `b` (branching) | 0.03                | 0.29 | 0.03                  | 0.32 |
| `k` (in-degree) | 0.14                | 0.21 |                       |      |
| `h` (height)    | 0.00                | 0.39 | 0.00                  | 0.46 |

Same experiment on pure `k`-trees of 127 nodes, `b ~ Normal(2,1)`, `k = 3.5` [verified, Table 3]:

| Param | Spurious edges `ê` | `e′` | No spurious edges `ê` | `e′` |
| ----- | ------------------ | ---- | --------------------- | ---- |
| `p`   |                    |      | 0.02                  |      |
| `b`   | 0.03               | 0.86 | 0.02                  | 0.31 |
| `k`   | 0.02               | 0.43 |                       |      |
| `h`   | 0.05               | 0.66 | 0.10                  | 0.43 |

Cascade-property recovery (nodes, edges, width, participation) is **figure only** [figure, Fig. 7]. The paper's stated findings: correction beats naive observation for **σ ≤ 0.7**, gives **20-30% relative error even at σ = 0.1** (90% missing), and **does worse than doing nothing for σ > 0.9** [verified, prose]. The one failure mode is **width on retweet influence cascades**, which they attribute to those trees being imbalanced [verified].

The takeaway that matters for us: **cascades are fragile.** A small fraction of missing nodes disconnects a tree, so _observed_ statistics are biased low by 30-86% at σ = 0.5, which is why a reconstruction step is not optional if you want an unbiased cascade statistic.

### 5.4 Zong et al. (ICDM 2012): consistent trees

Data [verified, §V]: **Enron** email graph, 86,808 nodes; **Twitter** retweets from >17M users over 7 months from June 2009, giving **321 cascades of depth >4, node size 10-81**; synthetic cascades on an anonymous Facebook social graph. Uncertainty is `σ = 1 − |X|/|V_T|`, i.e. the fraction of the true cascade _removed_.

Node- and edge-level precision [verified, Table I]:

| Algorithm | Precision | Enron `d=3` | Enron `d=4` | Twitter `d=4` | Twitter `d=5` |
| --------- | --------- | ----------- | ----------- | ------------- | ------------- |
| **WPCT**  | `prec_v`  | 100%        | 100%        | 97.2%         | 93.2%         |
| **WPCT**  | `prec_e`  | 78.2%       | 82.4%       | 86.1%         | 82.6%         |
| WBCT      | `prec_v`  | 100%        | 70.1%       | 73.6%         | 66.1%         |
| WBCT      | `prec_e`  | 69%         | 55.7%       | 60.6%         | 41.7%         |

Robustness to missing data is **figure only** [figure, Fig. 6]; the paper states WPCT holds `prec ≥ 70%` and `rec ≥ 25%` even when **85% of cascade nodes are removed** [verified, prose]. Note the gap between `prec_v` (100%) and `prec_e` (78-86%): **getting the node set right is much easier than getting the edges right**, which is the same finding DIPT reports 13 years later and the reason §8 insists on scoring edges separately.

### 5.5 Rozenshtein et al. (KDD 2016) (CulT) **no tables published**

Full-text extraction finds **zero occurrences of "Table"** in the paper [verified]. Everything below is [figure] or [verified] prose.

Setup [verified, §5.1]: synthetic power-law backgrounds of `n = 100` with `δ = 100` random interactions injected between consecutive real activations; plus **100-node BFS subgraphs** of Facebook (New Orleans wall posts), Tumblr (MemeTracker quotes), Students (UC Irvine message log) and Enron. Four propagation models: **SI** (`p = 0.1`), **shortest path**, **IC** (`p` = inverse largest eigenvalue), **forest fire** (threshold 1), each run until half the nodes activate. Two reporting schemes: **RS** (each active node reported w.p. β) and **FR** (frontier reported after θ interactions). Metric: **MCC**. Baselines: `Reports` (trivially precision 1.0) and `Baseline` (one-hop cascade from each report). Real-cascade case study: Flixster movie ID 54053, rated by 10K users, first 10 raters designated seeds, reports sampled among frontier nodes w.p. 0.5 with θ = 1000.

Directions [figure]: MCC ≈ **0.60-0.90** across the four real graphs, CulT above both baselines throughout; accuracy is best for power-law exponent **γ ∈ [1.5, 3.5]**, the range real graphs occupy [verified, Fig. 2]; the `α ↔ k` relation is monotonic so binary search on `α` recovers a target seed count, and tree cost shows an "elbow" at the true `k = 5` [figure, Fig. 1].

**Why it matters to us despite having no table:** CulT is the only method here that assumes **no propagation model at all** and operates on a temporal interaction stream. It is the honest ceiling for "what can you do without a kernel", and therefore the right thing to beat with a learned one.

### 5.6 Xiao et al. (SDM 2018 and ICDM 2018): **no result tables**

**SDM 2018, `OrderedSteinerTree`** [verified, §6]: graphs are **email-eu 986/25,552**, **grqc 4,158/13,428**, **arxiv-hep-th 8,638/24,827**, **facebook 4,039/88,234**, three of which we already load. Cascade models: SI (`p = 0.5`), IC (tuned to activate half the graph), continuous-time (exponential, `β = 1`), shortest path. Report probabilities `q = 0.001 × 2^i`, `i = 0…8`; 100 runs per setting. Real cascades: **Digg 2009, 279,631 nodes / 1,548,131 edges, top 18 cascades, average size 1,965**, 8 runs.

Directions [figure]: all four methods reach **node precision > 0.8**, usually near 1.0; `closure`/`greedy`/`delayed-bfs` beat plain `steiner` on **order accuracy** under every model and graph; `greedy` scales roughly linearly in `|E|`. On **real** Digg cascades node precision **drops significantly**, which the authors attribute to infected nodes being densely interconnected with uninfected ones and to the parsimony assumption failing [verified, prose].

**ICDM 2018, tree sampling**: datasets [verified, Table I]:

| Name       | \|V\|   | \|E\|     | Assortativity |
| ---------- | ------- | --------- | ------------- |
| infectious | 410     | 2,765     | 0.0121        |
| email-univ | 1,133   | 5,451     | −0.0007       |
| student    | 1,266   | 6,451     | −0.0039       |
| grqc       | 4,158   | 13,428    | 0.1641        |
| Digg       | 279,631 | 1,548,131 | 0.0015        |

Setup [verified]: SI with `β = 0.1`; IC with per-edge `p ~ U[0,1]`; 1,000 sampled Steiner trees per instance; metric is **average precision (AP)** on nodes _and_ edges; 100 runs per setting; Digg uses the top-10 largest real cascades, average size 1,868. Results are [figure] only. Their reported qualitative finding: on `grqc` (the one graph with high assortativity (0.164)) a **Personalized PageRank baseline beats tree sampling**, because assortative graphs make the infected subgraph densely connected and a random walker exploits that [verified, prose]. Worth internalizing: on assortative graphs a trivial centrality baseline is competitive, so any reconstruction claim must report PageRank alongside.

### 5.7 Farajtabar et al. (AISTATS 2015) ("Back to the Past") **no tables**

Synthetic [verified, §5.1]: three Kronecker families, core-periphery `[0.9 0.5; 0.5 0.3]`, random `[0.5 0.5; 0.5 0.5]`, hierarchical `[0.9 0.1; 0.1 0.9]`, 10 networks each of **256 nodes / 512 edges**, `α ~ U(10, 5)`; a cascade counts as "large" at >40 nodes; **400 Monte Carlo samples, 10% of infected nodes observed**.

Real [verified, §5.2]: the MemeTracker meme dataset over **1,700 mainstream media sites and blogs**; a diffusion network is inferred **per topic with NETRATE** first, then sources are recovered; 15 sources each with ≥10 long cascades (>27 nodes), 5 runs, **10% observed, 500 samples**.

Directions [figure, Figs. 3-5]: on synthetic Kronecker graphs success probability reaches **≈0.6** and top-10 success **≈1.0**, "dramatically" above NaiveMC / OutDeg / NETSLEUTH / Pinto's method. On MemeTracker the numbers are far smaller but the paper quantifies them in prose [verified]: their method is **≈20× better than a uniform guesser over all 1,700 nodes** (`1/1700 = 5.8 × 10⁻⁴`) and **≈5× better than guessing uniformly among the 425 nodes from which the observations are reachable** (`1/425 = 2.4 × 10⁻³`), so absolute success probability is on the order of **1%**. Source-time MSE ≈ 2,000 days², i.e. **≈45 days** error on cascades that unfold over a year [verified].

That 1% is the sober number to keep in mind: **retrospective reconstruction on real data is very hard**, and the published wins are ratios against random, not absolute accuracy.

### 5.8 SOTA summary

| Setting                                      | Best published                         | Metric reported                    | Notes                                                           |
| -------------------------------------------- | -------------------------------------- | ---------------------------------- | --------------------------------------------------------------- |
| **Single snapshot → full history**           | **DITTO** (KDD'23)                     | F1 .78, .84, NRMSE .16, .23 on BA/ER | beats supervised GRIN trained with true `β` on SIR              |
| **Propagation tree (explicit edges)**        | **DIPT** (2025)                        | Path Precision .42, .68             | the only method that outputs edges; 3.5×/4.37× over DDMSL/DDMIX |
| **Reports + timestamps → ordered tree**      | Xiao SDM'18 (`closure`, `delayed-bfs`) | node precision >0.8 [figure]       | `delayed-bfs` is `O(m + k log k)`                               |
| **Reports → marginal infection probs**       | Xiao ICDM'18 (tree sampling)           | AP [figure]                        | loses to Personalized PageRank on assortative graphs            |
| **Temporal network, no model assumed**       | **CulT** (KDD'16)                      | MCC 0.6-0.9 [figure]               | only model-free method                                          |
| **Cascade statistics under sampling**        | **Sadikov** (WSDM'11)                  | 2-14% param error at σ=0.5         | vs 21-86% observed                                              |
| **Source + start time, partial cascade**     | Farajtabar (AISTATS'15)                | SP ≈0.6 synthetic, ≈1% real        | see [`source_localization.md`](source_localization.md)          |
| **BA / ER / Power Grid / Cora-ML**           | DITTO / DIPT                           |                                    | **the four rows we can reproduce directly**                     |
| **Jazz, NetHEPT, NetPHY, LastFM, wiki-Vote** | **none found**                         |                                    | no cascade-reconstruction baseline exists                       |

---

## 6. Datasets

### 6.0 What `--dataset` accepts for this task

**Every one of the 71 real graph loaders and all 7 synthetic families (`er`, `ba`, `ws`, `sbm`, `powerlaw_cluster`, `kronecker`, `karate`) runs on this task.** The loaders are task-agnostic; nothing in `pipeline/tasks.py` restricts a task to a dataset. The lists below are therefore an experimental CHOICE, not a constraint, and the only place that choice is currently encoded is `sbatch/`. Where the sweep and the benchmark family disagree, the sweep is the accident and the family is the intent.

| | Datasets |
| --- | --- |
| **Benchmark family** (§6, the 7 CR benchmarks) | `infectious`, `email_univ`, `uci_students`, `oregon2`, `rt_pol`, `ca_hepth`, `citeseer` |
| **Also reported on** | `cora_ml`, `power_grid` (the two DIPT rows where our graph is known to match), `ca_grqc` (§8.2 trap 4's assortativity case), `jazz` |
| **Synthetic** | `ba`, `er` at DITTO's own parameters (`generate_data_ba_er.sbatch`), plus `ws`, `sbm`, `karate`, `powerlaw_cluster`, `kronecker` |
| **In `sbatch/cascade_reconstruction/sweep_datasets.sbatch` today** | `jazz`, `infectious`, `email_univ`, `uci_students`, `citeseer`, `cora_ml`, `ca_grqc`, `power_grid`, `ca_hepth`, `oregon2`, `rt_pol` |

The sweep covers all 7 of the family plus the 4 context graphs. **`ca_hepth` is not `cit_hepth`** (8,638-node co-authorship vs 27,769-node citation).

Cascade reconstruction needs **two things a plain IM graph does not have**: a per-node activation _time_, and (for tree metrics) a ground-truth who-infected-whom edge. Almost no real corpus has the second. That is why every paper in §5 that reports tree metrics **simulates** the cascade, and why our generator's stored `frontier` is a genuine asset rather than a shortcut.

### 6.1 What we already load yes

Authoritative rows live in [`influence_maximization.md`](influence_maximization.md) §6.1. The ones this literature actually uses:

| `--dataset`        | Nodes   | Edges         | Used here by                                                                                | Timestamps?         |
| ------------------ | ------- | ------------- | ------------------------------------------------------------------------------------------- | ------------------- |
| `cora_ml` yes       | 2,810   | 7,981         | **DIPT** (simulated SI, 200 iters)                                                          | no: simulated       |
| `power_grid` yes    | 4,941   | 6,594         | **DIPT** (simulated SI)                                                                     | no: simulated       |
| `facebook` yes      | 4,039   | 88,234        | **Xiao SDM'18** (SI/IC/CT/SP)                                                               | no: simulated       |
| `email_eu_core` yes | 1,005   | 24,929 arcs   | **Xiao SDM'18** as "email-eu", quoted **986 / 25,552** (the LCC)                            | no: simulated       |
| `ca_grqc` yes       | 5,242   | 14,484        | **Xiao SDM'18 + ICDM'18** as "grqc", quoted **4,158 / 13,428** (the LCC)                    | no: simulated       |
| `digg` yes          | 116,893 | ≈2.6M         | Warning: **wrong Digg**, this literature uses the ISI/Lerman 279,631-node cascade version (§6.2) | no (friendship only) |
| `ba`, `er` yes      | 1,000   | 3,984 / 3,987 | **DITTO** (`--ba-m 4`, `--er-p 0.008` reproduces their graphs                              | no) simulated       |

**Five of our loaders are already in this literature's tables, plus both synthetic families DITTO uses.** No new loader is needed to produce a comparable first result. The two LCC discrepancies are the preprocessing switch documented in [`influence_maximization.md`](influence_maximization.md) §6.3, not different graphs.

### 6.2 Cascade corpora: the ones with real traces

Rows carried over from [`influence_maximization.md`](influence_maximization.md) §6.2 and re-checked, plus every other corpus this literature uses. **Cascades**, **avg cascade**, **span** and **timestamps** are the columns that matter here.

| Dataset                      | Nodes                                                                                                                  | Edges              | Cascades                                                         | Avg cascade     | Time span                                           | Timestamps                        | Source                                                                                                                                             |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------ | ---------------------------------------------------------------- | --------------- | --------------------------------------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Digg 2009**                | 279,631                                                                                                                | 2,251,166 arcs     | 3,553                                                            | 847             | 2009 (~1 month of stories)                          | yes vote timestamps                | [ISI/Lerman](https://www.isi.edu/~lerman/downloads/digg2009.html)                                                                                  |
| **Sina Weibo**               | 1,170,689                                                                                                              | 225,877,808 arcs   | 115,686                                                          | 148             | Sep 2012: Jul 2013 [claim]                         | yes repost times                   | [AMiner](https://www.aminer.cn/influencelocality) via [IMINFECTOR](https://github.com/geopanag/IMINFECTOR)                                         |
| **MAG (CS)**                 | 1,436,158                                                                                                              | 15,928,078 arcs    | 181,020                                                          | 29              | citations to 2017 [claim]                           | yes publication year only (coarse) | Microsoft Academic Graph, CS subset, via [IMINFECTOR](https://github.com/geopanag/IMINFECTOR)                                                      |
| **MemeTracker**              | 96M phrases over ~1.7M sites; DIPT uses a **583-site / 6,700-cascade** subnetwork [verified]; SL-VAE uses 12,529 nodes |                    | 6,700 (DIPT slice)                                               |                 | Aug 2008: Apr 2009                                 | yes per-mention time               | [SNAP memetracker9](https://snap.stanford.edu/data/memetracker9.html)                                                                              |
| **Flixster**                 | 29,357                                                                                                                 | 425,228 arcs       | action log, 10 topics                                            |                 | Nov 2005: Nov 2009 [claim]                         | yes rating times                   | topic-aware IM literature (Barbieri/Goyal); Rozenshtein uses movie ID 54053, 10K raters [verified]                                                 |
| **Higgs Twitter**            | 456,631                                                                                                                | 14,855,875 arcs    | 1 event (retweet/reply/mention streams)                          |                 | 1-7 Jul 2012                                        | yes second-resolution              | [SNAP higgs-twitter](https://snap.stanford.edu/data/higgs-twitter.html)                                                                            |
| **Pol (rt-pol)**             | 18,470                                                                                                                 | 48,053             | 1 retweet network, `T=40` as used by DITTO [verified]            |                 | one US political event                              | yes retweet times                  | [NetRepo rt-pol](https://networkrepository.com/rt-pol.php)                                                                                         |
| **Hebrew**                   | 3,521                                                                                                                  | 18,064             | 1 retweet network, `T=9` [verified]                              |                 | an Israeli election event                           | yes retweet times                  | via DITTO [repo](https://github.com/q-rz/KDD23-DITTO)                                                                                              |
| **BrFarmers**                | 82                                                                                                                     | 230                | 1 adoption trace, `T=16` [verified]                              |                 | 1943-1966 innovation study                          | yes adoption year                  | [netdiffuseR brfarmers](https://usccana.github.io/netdiffuseR/reference/brfarmers.html)                                                            |
| **Covid (CDC levels)**       | 344 counties                                                                                                           | 2,044 (10-NN geo)  | 1 trace, `T=10` [verified]                                       |                 | 23 Feb: 21 Dec 2022                                | yes weekly level changes           | [CDC Community Levels](https://data.cdc.gov/)                                                                                                      |
| **SocioPatterns Infectious** | 410                                                                                                                    | 2,765              | face-to-face contact stream                                      |                 | Infectious exhibition, Dublin 2009                  | yes 20-second contact resolution   | [KONECT sociopatterns-infectious](http://konect.cc/networks/sociopatterns-infectious/) · [NetRepo](https://networkrepository.com/infect-hyper.php) |
| **UCI Students**             | 1,266 (Xiao) / 1,899 (full)                                                                                            | 6,451 (Xiao)       | message stream                                                   |                 | Apr, Oct 2004                                        | yes message times                  | [NetRepo ia-fb-messages](http://networkrepository.com/ia-fb-messages.php) · [Opsahl](https://toreopsahl.com/datasets/)                             |
| **email-univ**               | 1,133                                                                                                                  | 5,451              | email stream                                                     |                 |                                                     | yes                                | [NetRepo ia-email-univ](http://networkrepository.com/ia-email-univ.php)                                                                            |
| **Enron**                    | 86,808 (Zong) / 36,692 (SNAP)                                                                                          | 183,831 (SNAP)     | Zong: emails containing "California" as infections [verified]    |                 | 1999-2002                                           | yes email times                    | [CMU Enron](https://www.cs.cmu.edu/~enron/)                                                                                                        |
| **Oregon2**                  | 11,461                                                                                                                 | 32,730             | none: DITTO simulates SI/SIR, `T=15`                            |                 | AS peering, 26 May 2001                             | no                                 | [SNAP Oregon-2](http://snap.stanford.edu/data/Oregon-2.html)                                                                                       |
| **Prost**                    | 15,810                                                                                                                 | 38,540             | none: DITTO simulates, `T=15`                                   |                 | online forum reviews                                | no                                 | via DITTO [repo](https://github.com/q-rz/KDD23-DITTO)                                                                                              |
| **IDSS**                     | 3,143 US counties                                                                                                      | mobility flows     | ≈500-1,000 infected counties/run, 12k, 48k individuals [verified] | 90 days         | synthetic, `R0=1.2`, 6-day infectious period        | yes simulated day                  | [DIPT simulator](https://anonymous.4open.science/r/Compartmental-Infectious-Disease-Simulation-4F7C)                                               |
| **CiteSeer**                 | 3,327                                                                                                                  | 4,732 [claim]      | none: DIPT simulates SI                                         |                 |                                                     | no                                 | standard citation benchmark                                                                                                                        |
| **Twitter (Sadikov)**        | 71,804,410                                                                                                             | 2,040,072,198 arcs | 250 retweet + 100 blog cascades [verified]                       | ≥100 nodes each | Jun, Dec 2009 (tweets); Aug, Nov 2008 (Spinn3r blogs) | yes                                | Topsy / Spinn3r, **not publicly redistributed**                                                                                                   |
| **Twitter7 (Yang, Leskovec)** |                                                                                                                        |                    | 476M tweets                                                      |                 | Jun, Dec 2009                                        | yes                                | [SNAP twitter7](https://snap.stanford.edu/data/twitter7.html)                                                                                      |

Digg 2009 / Sina Weibo / MAG counts are **[verified]** from IMINFECTOR's Table 3 (re-checked against `influence_maximization.md` §6.2); MemeTracker and Flixster sizes remain **[claim]** except DIPT's 583/6,700 slice which is **[verified]**.

**Corpora named in the brief that this review could not pin down**: recorded so nobody re-searches them: a public **Telegram** or **Reddit** cascade corpus with who-infected-whom labels; **Android / Christianity StackExchange** cascade splits (they appear in cascade-_prediction_ work, not in any reconstruction paper found here, see [`cascade_prediction.md`](cascade_prediction.md)); the **APS citation** corpus as a reconstruction benchmark; and **SMS/call-log** contact traces. Xiao's own [cascade-dataset](https://github.com/xiaohan2012/cascade-dataset) index lists Higgs, Digg 2009, [Flickr (MPI-SWS)](http://socialnetworks.mpi-sws.org/data-www2009.html), [EPFL tweets-with-URL](http://lsir.epfl.ch/research/datasets/socialnetwork/), [NEWS](https://github.com/s-mishra/featuredriven-hawkes) and [AMiner citations](https://aminer.org/citation), and flags SEISMIC's SNAP link as **broken** [verified from the repo README].

### 6.3 What a reconstruction benchmark needs that none of these give

Real corpora give you **timestamps** but never the **transmission edge**: Digg records that user `v` voted at time `t`, not that `u` caused it. Consequently:

| Requirement                                 | Real corpora                            | Our generator                                                    |
| ------------------------------------------- | --------------------------------------- | ---------------------------------------------------------------- |
| Per-node activation time                    | yes                                      | yes (`t` of first appearance in `frontier`)                       |
| Ground-truth infected set                   | yes                                      | yes (`next_state.infected`)                                       |
| Ground-truth **who-infected-whom**          | no: nobody has it                      | Warning: not stored today; the simulator knows it, we discard it (§11) |
| Soft `P(infected)` per node                 | no                                      | yes (`next_marginal_*`)                                           |
| Counterfactual branches from the same `s_t` | no                                      | yes (`branch == cf_i`)                                            |
| Controllable observation rate               | no (fixed by the API that collected it) | yes (mask at load)                                                |

Three of six are things **only** a simulator can give, which is exactly why DITTO, DIPT, Xiao and Rozenshtein all simulate. We are already in that position and can additionally offer soft targets and counterfactual branches, neither of which appears anywhere in §5.

---

## 7. Which paper uses which

Cells mark the dataset **as that paper reports it**. Check [`influence_maximization.md`](influence_maximization.md) §6.3 before assuming two papers with the same cell used the same graph: Digg in particular denotes two different objects here.

| Dataset                          | Sadikov'11 | Zong'12 | Farajtabar'15 | Rozenshtein'16   | Xiao SDM'18     | Xiao ICDM'18 | DITTO'23 | DIPT'25 |
| -------------------------------- | ---------- | ------- | ------------- | ---------------- | --------------- | ------------ | -------- | ------- |
| **BA** yes                        |            |         |               |                  | yes (scalability) |              | yes        |         |
| **ER** yes                        |            |         |               |                  |                 |              | yes        |         |
| Kronecker (CP/rand/hier)         |            |         | yes             |                  |                 |              |          |         |
| Power-law synthetic              |            |         |               | yes                |                 |              |          |         |
| **Cora-ML** yes                   |            |         |               |                  |                 |              |          | yes       |
| **Power Grid** yes                |            |         |               |                  |                 |              |          | yes       |
| CiteSeer                         |            |         |               |                  |                 |              |          | yes       |
| **email-Eu-core** yes             |            |         |               |                  | yes               |              |          |         |
| **ca-GrQc** yes                   |            |         |               |                  | yes               | yes            |          |         |
| **facebook** yes                  |            |         |               | yes (100-node BFS) | yes               |              |          |         |
| arxiv-hep-th                     |            |         |               |                  | yes               |              |          |         |
| infectious                       |            |         |               |                  |                 | yes            |          |         |
| email-univ                       |            |         |               |                  |                 | yes            |          |         |
| Students (UCI)                   |            |         |               | yes                |                 | yes            |          |         |
| Enron                            |            | yes       |               | yes                |                 |              |          |         |
| Tumblr / MemeTracker             |            |         | yes             | yes                |                 |              |          | yes       |
| Flixster                         |            |         |               | yes                |                 |              |          |         |
| **Digg (ISI 2009)**              |            |         |               |                  | yes               | yes            |          |         |
| Twitter (retweets)               | yes          | yes       |               |                  |                 |              |          |         |
| Blogs (Spinn3r)                  | yes          |         |               |                  |                 |              |          |         |
| Oregon2                          |            |         |               |                  |                 |              | yes        |         |
| Prost                            |            |         |               |                  |                 |              | yes        |         |
| BrFarmers / Pol / Covid / Hebrew |            |         |               |                  |                 |              | yes        |         |
| IDSS                             |            |         |               |                  |                 |              |          | yes       |

**Bold + yes = we already load it.** Seven of the graphs in this table are ours, spread across five of the eight papers: a far better intersection than the IM literature gives us (four rows, two papers).

---

## 8. Evaluation protocol

### 8.1 Metrics, and what each one is blind to

| Metric                            | Definition                                                                  | Blind to                                        | Who reports it                                     |
| --------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------- | -------------------------------------------------- |
| **Event precision / recall / F1** | over `(node, t)` infection events in `Ŷ` vs `Y`; macro-averaged over states | _which_ neighbour caused it                     | DITTO (`F1`) · DIPT (`F1` for sources)             |
| **Node-set precision / recall**   | over the inferred infected set, ignoring time                               | timing and structure entirely: the easy metric | Zong (`prec_v` = 100%) · Xiao SDM'18               |
| **Edge / path precision**         | fraction of inferred who-infected-whom edges that are correct               | nodes you never reached                         | Zong (`prec_e` = 78-86%) · DIPT (`Path Precision`) |
| **Jaccard index on tree edges**   | \|Ê ∩ E\| / \|Ê ∪ E\|                                                       | edge _direction_ if computed undirected         | DIPT                                               |
| **Order accuracy**                | fraction of tree edges `u→v` with `t(u) ≤ t(v)`                             | absolute times; only checks ordering            | Xiao SDM'18                                        |
| **Infection-time MAE / NRMSE**    | error on hitting times, DITTO normalizes by `2n(T+1)²`                      | topology of the tree                            | DITTO                                              |
| **Average precision (AP)**        | ranking quality of per-node infection marginals                             | calibration, only ranking                       | Xiao ICDM'18                                       |
| **MCC**                           | Matthews correlation over infected/not                                      | timing and structure                            | Rozenshtein (its **only** metric)                  |
| **Trajectory log-likelihood**     | `Σ_t log p(ŝ_{t+1} \| ŝ_t, G)`                                              | nothing (but needs a kernel to evaluate under  | **nobody**) this is ours to add                   |

The `prec_v = 100%` / `prec_e = 78%` split in Zong's Table I and DIPT's 0.68-path-precision-at-best are the same message thirteen years apart: **the node set is easy, the tree is hard.** Any protocol that reports only node-level numbers is reporting the easy half. Our own eval already learned this lesson in a different guise, `infected_acc` looks great because it is dominated by unchanged nodes, which is why we report `delta_f1`.

### 8.2 The traps

1. **Observation rate is the axis, and papers disagree on how to sweep it.** Xiao sweeps report probability geometrically, `q = 0.001 × 2^i` for `i = 0…8` [verified]; Sadikov sweeps sample ratio `σ = 0.1 … 1.0` linearly [verified]; Zong sweeps _uncertainty_ `σ = 1 − |X|/|V_T|` from 0.25 to 0.85 [verified], which is the **complement** of Sadikov's `σ`. Two papers using the symbol `σ` mean opposite things. State the direction explicitly or the curve reads backwards.
2. **"Correction helps" is only true below a threshold.** Sadikov's method is _worse than doing nothing_ for `σ > 0.9` [verified]. Report the crossover, not just the best point.
3. **Simulated vs real is a cliff, not a gap.** DITTO's Table 4: GCN/GIN drop to `F1 ≈ 0.32` on real diffusion after being fine on synthetic. Any claim trained on NDlib transitions inherits this risk and must say so.
4. **Assortativity decides whether a trivial baseline wins.** Xiao ICDM'18: Personalized PageRank beats tree sampling on `grqc` (assortativity 0.164) and loses elsewhere [verified]. Always run PageRank alongside; on our suite `ca_grqc` is exactly the graph where it will bite.
5. **`NRMSE↓` next to `F1↑`.** DITTO's tables interleave them per column. Sort direction flips mid-row.
6. **Number of MC samples is a hyperparameter of the _metric_, not just the method.** Farajtabar uses 400 samples synthetic / 500 real; Xiao ICDM'18 uses 1,000 sampled trees and notes gains beyond that are marginal [verified]. Our `--mc-marginals 30` is an order of magnitude below both: fine for training targets, probably too low for a reconstruction _reference_.
7. **Averaging counts.** Xiao SDM'18 averages 100 runs on simulated cascades but only **8** on real ones [verified]; Rozenshtein averages 100; Zong averages 10. Error bars are not comparable across papers.

### 8.3 Protocol for the decoder-search formulation

§8.1 and §8.2 are the protocol this literature runs, and every comparable number must follow it. The formulation of §2.4 needs additions the literature has no reason to define, because no published method produces a reusable decoder. The shared requirements (two-axis splitting, budget parity across arms, cost accounting, baseline parity) are specified in [`source_localization.md`](source_localization.md) §8.5 and are not repeated. Four things differ here.

**The mask is a protocol parameter, and §8.2 trap 1 says the field cannot agree on its direction.** Xiao sweeps report probability geometrically (`q = 0.001 × 2^i`), Sadikov sweeps sample ratio `σ` linearly, and Zong sweeps _uncertainty_ `σ = 1 − |X|/|V_T|`, which is the **complement** of Sadikov's `σ` [all verified]. Two papers use the symbol `σ` for opposite quantities. State the masking direction explicitly and sweep it, because a decoder selected at one observation rate is not necessarily good at another, and that sweep is itself a generalization axis in the sense of §8.5.1.

**The four settings of §2.7 are four separate protocols, not one.** A decoder selected under (c-ii) final-snapshot-only is solving a different problem from one selected under (c-i) partial timestamps. Report them as separate rows and never pool them. The DASH setting (c-ii) is the one worth leading with, since it is DITTO's formulation and the hardest published one.

**Report the reward function.** Because §2.6 makes $\operatorname{Score}$ a design decision with a documented failure mode, the value of $\lambda$ and the confirmation that a trivial decoder scores badly under it belong in the results, not in a footnote. A reader cannot interpret a program-search result without knowing what the search was told to want.

**The MC-sample count is a hyperparameter of the _metric_, not just the method** (§8.2 trap 6), and under the decoder-search formulation it is also a hyperparameter of arm 4. Arm 4's `R` and the eval split's `--mc-marginals` are different numbers with different jobs; state both.

---

## 9. Implications for this project

1. **The world model is the missing ingredient, and supplying it is still not the contribution.** Every method in §3 needs `p(s_{t+1} | s_t, G)` and gets it from a hand-set global `β` (NetFill, DHREC, NETSLEUTH), a mean-field estimate off one snapshot (DITTO), or a fit on _other_ cascades (Farajtabar/NETRATE). Ours is learned per-edge against MC-estimated marginals at every step, which is strictly better supervised than any of them. But the kernel is a _component_ of these methods, so improving it lands us inside someone else's frame (§2.3). **The contribution is the decoder**, and §2.4 formalizes searching for it with the coding-agent outer loop while the world model serves as the transition oracle. **No new simulator for the dynamics, no new action op** (actions are `NULL`, `T_exo = identity`), but one simulator change for the ground truth, see item 9.

   **This is the best both-loops fit in the folder**, better than `source_localization` on all four criteria: the decoder space is wider and genuinely contested (§2.4.3), the inner loop runs two orders of magnitude hotter at $10^4$ oracle calls per instance (§2.4.2), the reward is dense and free (item 2), and there is no closed-form alternative because the tree sum is super-exponential (§1).

2. **Supervised evaluation is free: this is the single biggest asymmetry.** Our `frontier` channel is literally "activated at step `t`", the quantity DITTO's NRMSE scores and Rozenshtein's `FR` scheme samples, and `generate_wm_data.py` already writes it every step keyed by `episode_id`. Mask part of an episode at load time, reconstruct, score against what is already on disk. Every paper in §5 had to build its own ground truth; we regenerate ours by grouping a JSONL file.

3. **Do the three easy settings first, and say which is which.**
   - **(a) `G` known + partial node observations**: today. Mask nodes from `state`. The Xiao/Sadikov/NetFill regime.
   - **(c-i) timestamps observed**: today. `t` is on every record; masking `(node, t)` pairs gives Xiao's `OrderedSteinerTree` input exactly.
   - **(c-ii) final snapshot only**: today, and this is the one worth publishing. DITTO's DASH setting: no parameters, no sources, one snapshot. It is where a learned kernel should beat a mean-field `β̂`.
   - **(d) hidden nodes**: a loader mask on `reconstruct_episode_adjacency`; untested, cheap, do it third.
   - **(b) `G` unknown**: no not this file. That is network inference, reviewed and not pursued.

4. **BA, ER, Cora-ML and Power Grid give a comparable table on day one.** DITTO publishes BA/ER at `n=1,000` (attachment 4 / `p=0.008`, SI+SIR, `T=10`, 5% sources) and DIPT publishes Cora-ML/Power Grid (SI, 10% sources). Both configurations are reachable from flags we already have. That is **four rows against two 2023-2025 papers** without adding a loader: better than our IM position.

5. **Report the tree, not just the node set, and under decoder search _reward_ the tree.** Zong's `prec_v = 100%` vs `prec_e = 78%` and DIPT's best-in-class `0.68` path precision say the node set is nearly solved and the edges are not. This is the same trap as `infected_acc` vs `delta_f1` in our own eval. If we report only node-level F1 we will look excellent and say nothing.

   **§2.6 upgrades this from a reporting rule to a correctness requirement.** For a human writing one decoder, reporting the easy metric is a presentation failure. For a program search the reward _is_ the specification: an outer loop scored on Event F1 will discover that the infected set is nearly free, that the tree contributes nothing, and will converge on decoders that never attempt the hard half. Weight the reward toward Path Precision, and verify before the full search that a trivial decoder (everyone reachable, parents by BFS) scores badly under it.

6. **The structured head is a prerequisite, again.** A `linear` head's rollout saturates, which makes any smoothing objective degenerate: the model concludes every hidden node was infected, and every reconstruction metric collapses to "predict everything". The self-terminating `ICTransmissionHead` (`count_bias −0.58` on BA-100 IC) is what makes trajectory likelihood meaningful at all. Same prerequisite the coding-agent loop was waiting on; one fix, two tasks.

7. **`--mc-marginals 30` is too low to serve as a reconstruction _reference_.** Farajtabar uses 400-500 samples, Xiao 1,000 trees. 30 is fine for training targets but will show up as noise in a masked-episode ground truth. Bump it for the eval split only.

8. **Run PageRank as a baseline, specifically on `ca_grqc`.** Xiao ICDM'18 found Personalized PageRank beats tree sampling on exactly that graph because of its assortativity (0.164). If our method cannot beat PageRank there, the result is not real.

9. **Store the transmission edge.** `build_record` has no field for it, but the cost is larger than "add a field", because **NDlib never produces it**. `IndependentCascadesModel.iteration` sets `actual_status[v] = 1` on a successful coin flip without recording which `u` caused it [verified, read from the installed source]. Capturing it means subclassing `IndependentCascadesModel` and overriding `iteration` to log the `(u, v)` pair when `flip <= threshold`, then threading that through `Simulator.advance` into a new `parents` field on the record.

   Two subtleties for whoever does it. First, NDlib iterates spreaders in node order and skips any `v` already flipped this step, so the recorded parent is _the first successful `u` in node order_, not a uniformly random one among the successes: a real but documentable bias. Second, LT has no transmission edge at all: activation is a threshold crossing over the whole active neighbourhood, so the honest ground truth there is a parent _set_, not a parent.

   **Under the formulation of §2.4 this is a hard prerequisite, not the highest-value optional item.** Item 5 above establishes that the outer-loop reward has to be tree-weighted or the search optimizes the wrong thing, and a tree-weighted reward is not computable without a ground-truth parent. So this moves to the front of the build order (§2.10 item 1) and everything else waits on it. It also remains what it always was: the unlock for every tree metric in §8.1, and the thing that would give us the only corpus in this literature carrying both soft MC marginals and ground-truth propagation edges (§6.3). Call it a simulator change of a day.

10. **Cost is honest and bounded, and the schedule risk is the control rather than the method.** Item 9 is a day. The agent loop, contract, primitives, masked-episode harness, metrics and prompts are 3 to 5 days (§2.10). The expensive piece is **arm A**, the DITTO-style decoder with our kernel that 6-vs-A is measured against: a greedy backward decode is a day, but a defensible DITTO-parity implementation is weeks, and a weak control makes the comparison meaningless. DITTO beats a supervised model trained with the _true_ `β` on BA-SIR and ER-SIR (§5.1.1), so it is genuinely hard to beat. Budget **a week for a first table** and treat arm A's fidelity as the thing that slips.

    The encouraging half is unchanged: §5.1 shows the MLE baselines (DHREC, CRI) sit 10-35% behind the supervised ideal, so the bar for an _interesting_ number is low even though the bar for beating DITTO is not.

11. **The IMINFECTOR warning applies here too, harder.** DITTO's Table 4 shows supervised models trained on simulated SI/SIR collapsing to `F1 ≈ 0.32` on real diffusion. We train on NDlib transitions. Any reconstruction claim we make is a claim about simulated dynamics unless and until we test on a logged trajectory, which is exactly the "real-world data = logged trajectories" scope already agreed.

12. **DIPT is an Emory paper (Memon, Ling, Kong, Seshagiri, Zufle, Liang Zhao).** It is the only method that outputs explicit propagation-tree edges, it has no public code, and it uses two of our graphs. That is a collaboration surface, not just a citation.

---

## 10. Reference list

**Statistical correction / missing data** [Sadikov 2011 WSDM, Correcting for Missing Data in Information Cascades](https://cs.stanford.edu/~jure/pubs/cascades-wsdm11.pdf) · [ACM](https://dl.acm.org/doi/10.1145/1935826.1935861) (403 to bots) · no public code found · [Belák 2015, Phantom cascades (arXiv 1502.01602)](https://arxiv.org/abs/1502.01602) · no public code found · [How the cascade inference problem distorts information diffusion (arXiv 2410.21554)](https://arxiv.org/abs/2410.21554) · no public code found

**Combinatorial / Steiner-tree** [Lappas 2010 KDD, Finding Effectors in Social Networks](https://dl.acm.org/doi/10.1145/1835804.1835882) (403 to bots) · no public code found · [Zong 2012 ICDM, Inferring the Underlying Structure of Information Cascades (arXiv 1210.3587)](https://arxiv.org/abs/1210.3587) · [IEEE](https://ieeexplore.ieee.org/document/6413726) · no public code found · [Rozenshtein 2016 KDD, Reconstructing an Epidemic over Time](https://www.kdd.org/kdd2016/papers/files/rpp0920-rozenshteinAT3.pdf) · [ACM](https://dl.acm.org/doi/10.1145/2939672.2939865) · [code](https://github.com/polinapolina/reconstructing-an-epidemic-over-time) · [Xiao 2018 SDM, Reconstructing a cascade from temporal observations (arXiv 1801.08586)](https://arxiv.org/abs/1801.08586) · [code](https://github.com/xiaohan2012/reconstructing-cascade) · [Xiao 2018 ICDM, Robust Cascade Reconstruction by Steiner Tree Sampling (arXiv 1809.05812)](https://arxiv.org/abs/1809.05812) · [code](https://github.com/xiaohan2012/cascade-reconstruction-by-tree-samples) · [random_steiner_tree](https://github.com/xiaohan2012/random_steiner_tree) · [active variant](https://github.com/xiaohan2012/active-cascade-reconstruction) · [Risk-aware temporal cascade reconstruction (KAIS 2022)](https://link.springer.com/article/10.1007/s10115-022-01748-8) · no public code found · [PoolMLE, Reconstructing Network Outbreaks under Group Surveillance (arXiv 2602.11419)](https://arxiv.org/abs/2602.11419) · no public code found

**MLE / probabilistic** [Gomez-Rodriguez 2010 KDD, NetInf (arXiv 1006.0234)](https://arxiv.org/abs/1006.0234) · [code](https://snap.stanford.edu/netinf/) · [Gomez-Rodriguez 2011 ICML, NETRATE (arXiv 1105.0697)](https://arxiv.org/abs/1105.0697) · [code](https://github.com/Networks-Learning/netrate) · [Prakash 2012 ICDM, NETSLEUTH](https://faculty.cc.gatech.edu/~badityap/papers/netsleuth-icdm12.pdf) · [IEEE](https://ieeexplore.ieee.org/document/6413786) · no public code found · [Sefer & Kingsford 2014 ICDM, DHREC](https://www.cs.cmu.edu/~ckingsf/software/dhrec/icdm2014.pdf) · [KAIS 2016](https://link.springer.com/article/10.1007/s10115-015-0904-x) · [project page](http://www.cs.cmu.edu/~ckingsf/research/3cde/paper.html) · [Farajtabar 2015 AISTATS, Back to the Past (arXiv 1501.06582)](https://arxiv.org/abs/1501.06582) · [PMLR](http://proceedings.mlr.press/v38/farajtabar15.pdf) · no public code found · [Sundareisan 2015 SDM, Hidden Hazards / NetFill (DOI 10.1137/1.9781611974010.47)](https://doi.org/10.1137/1.9781611974010.47) (403 to bots) · no public code found · [Chen 2016 TNSE, CRI (DOI 10.1109/TNSE.2016.2523804)](https://doi.org/10.1109/TNSE.2016.2523804) · no public code found · [Zhu, Chen & Ying 2017 AAAI, Catch'Em All / OJC (arXiv 1611.06963)](https://arxiv.org/abs/1611.06963) · [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/10746) · no public code found · [Chen, Tong & Ying 2019 TKDE, Inferring Full Diffusion History from Partial Timestamps (DOI 10.1109/TKDE.2019.2905210)](https://doi.org/10.1109/TKDE.2019.2905210) · no public code found

**Learning-based** [BRITS (arXiv 1805.10572)](https://arxiv.org/abs/1805.10572) · [code](https://github.com/caow13/BRITS) · [GRIN (arXiv 2108.00298)](https://arxiv.org/abs/2108.00298) · [code](https://github.com/Graph-Machine-Learning-Group/grin) · [SPIN (arXiv 2205.13479)](https://arxiv.org/abs/2205.13479) · [code](https://github.com/Graph-Machine-Learning-Group/spin) · [SL-VAE (KDD 2022)](https://dl.acm.org/doi/10.1145/3534678.3539267) (403 to bots) · [code](https://github.com/triplej0079/SLVAE) · [Deep Demixing / DDMIX (arXiv 2011.09583)](https://arxiv.org/abs/2011.09583) · [journal version (arXiv 2306.07938)](https://arxiv.org/abs/2306.07938) · [code](https://github.com/gojkoc54/Deep_demixing) · [DDMSL (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/46ab9d9645b6975b947231ddb48da1ab-Abstract-Conference.html) · [OpenReview](https://openreview.net/forum?id=5Fr8Nwi5KF) · no public code found · [DITTO (arXiv 2306.00488)](https://arxiv.org/abs/2306.00488) · [ACM](https://dl.acm.org/doi/abs/10.1145/3580305.3599488) · [code](https://github.com/q-rz/KDD23-DITTO) · [PGSL (ESWA 2024, DOI 10.1016/j.eswa.2023.122028)](https://doi.org/10.1016/j.eswa.2023.122028) · no public code found · [SIDSL (arXiv 2502.17928)](https://arxiv.org/abs/2502.17928) · no public code found · [DIPT (arXiv 2503.00646)](https://arxiv.org/abs/2503.00646) · no public code found · [Learning hidden cascades via classification (arXiv 2505.11228)](https://arxiv.org/abs/2505.11228) · no public code found

**Theory / learnability** [Amin, Heidari & Kearns 2014 ICML, Learning from Contagion (Without Timestamps)](https://proceedings.mlr.press/v32/amin14.html) · [PDF](https://www.cis.upenn.edu/~mkearns/papers/LearningFromContagion.pdf) · no public code found · [He, Xu, Kempe & Liu 2016 NeurIPS, Learning Influence Functions from Incomplete Observations (arXiv 1611.02305)](https://arxiv.org/abs/1611.02305) · [PDF](https://proceedings.neurips.cc/paper/2016/file/68b1fbe7f16e4ae3024973f12f3cb313-Paper.pdf) · no public code found · Milling, Caramanis, Mannor & Shakkottai 2012 SIGMETRICS, _Network forensics: random infection vs spreading epidemic_, no stable public PDF found; no public code found

**Data** [ISI/Lerman Digg 2009](https://www.isi.edu/~lerman/downloads/digg2009.html) · [SNAP MemeTracker](https://snap.stanford.edu/data/memetracker9.html) · [SNAP twitter7](https://snap.stanford.edu/data/twitter7.html) · [SNAP Higgs Twitter](https://snap.stanford.edu/data/higgs-twitter.html) · [SNAP Oregon-2](http://snap.stanford.edu/data/Oregon-2.html) · [NetRepo rt-pol](https://networkrepository.com/rt-pol.php) · [NetRepo ia-email-univ](http://networkrepository.com/ia-email-univ.php) · [NetRepo ia-fb-messages](http://networkrepository.com/ia-fb-messages.php) · [KONECT sociopatterns-infectious](http://konect.cc/networks/sociopatterns-infectious/) · [netdiffuseR brfarmers](https://usccana.github.io/netdiffuseR/reference/brfarmers.html) · [CMU Enron](https://www.cs.cmu.edu/~enron/) · [Opsahl datasets](https://toreopsahl.com/datasets/) · [Xiao's cascade-dataset index](https://github.com/xiaohan2012/cascade-dataset)

---

## 11. Open gaps

Honest list of what this review could **not** establish.

- **Rozenshtein (KDD'16), Xiao (SDM'18 and ICDM'18) and Farajtabar (AISTATS'15) publish no result tables at all**: confirmed by full-text extraction, `grep -c "Table"` on the KDD'16 text returns 0. Every §5.5-5.7 number is [figure] or [verified] prose. Getting per-cell baselines for those four requires running their code (three of the four have public repos; Farajtabar has none).
- **Sadikov's cascade-property recovery** (nodes / edges / width / participation vs `σ`) is Fig. 7 only. Only the parameter-error Tables 2-3 were transcribable.
- **DHREC's own numbers were not transcribed.** The CMU PDF is 2.6 MB and repeatedly truncated on download in this environment; DHREC appears here only as it is reported _inside_ DITTO's Tables 4-5. Its ICDM'14 self-reported results are unverified.
- **NETSLEUTH, OJC, NetFill, CRI, Chen-TKDE'19 result tables**: identified and linked, not transcribed. Their cells sit in source-localization space; see [`source_localization.md`](source_localization.md).
- **DIPT's graph sizes are never stated in its own paper.** It names Cora-ML, CiteSeer and Power Grid but publishes no dataset table [verified]: the counts in §6.1 are ours, not theirs, so a strict cell-for-cell comparison needs their preprocessing confirmed.
- **CiteSeer 3,327 / 4,732** is [claim]: carried from the standard citation benchmark, not read from DIPT.
- **MemeTracker and Flixster sizes** remain [claim] outside DIPT's 583-site / 6,700-cascade slice; the same gap is already logged in [`influence_maximization.md`](influence_maximization.md) §11.
- **Time spans** for Sina Weibo, MAG and Flixster are [claim]; only Digg 2009, Higgs, MemeTracker, Covid, BrFarmers and the SocioPatterns traces have spans verified from their own pages.
- **Prost and Hebrew have no independent public source** found: both reach us via DITTO's repo. Their provenance is unverified.
- **No public Telegram, Reddit, APS-citation, or SMS/call-log corpus with who-infected-whom labels was found** in this literature. If one exists it is not cited by any of the eight papers in §7.
- **Android / Christianity StackExchange** appear in cascade-_prediction_ work, not in any reconstruction paper surveyed: see [`cascade_prediction.md`](cascade_prediction.md).
- **Our pipeline has no transmission edge to discard.** `build_record` stores `state`/`next_state` but not who infected whom, and neither does the layer below it: NDlib's `IndependentCascadesModel.iteration` flips `v` to infected without recording the responsible `u` [verified, read from the installed source]. So tree-level metrics (§8.1) cannot be scored today, and getting them needs an NDlib subclass, not a record field. See §9 item 9 for the two subtleties (parent selection is biased by node order; LT has a parent _set_, not a parent). This is the one blocking gap on our side.
- **No published cascade-reconstruction baseline exists for Jazz, NetHEPT, NetPHY, LastFM, wiki-Vote, NetScience or `sbm`.** Results there would be self-contained, exactly as with SBM in the IM file.

### Gaps specific to the decoder-search formulation (§2.4)

Gaps in the literature rather than in this review. They are what makes §2.4 a contribution and simultaneously what leaves it without a comparison.

- **No published decoder is reusable across instances.** DITTO's MCMC, DIPT's alternating optimization and Farajtabar's importance-sampled MLE all re-solve from scratch per cascade. **No paper reports what happens when an inference procedure tuned on one graph is applied unchanged to another**, so the transfer axis of [`source_localization.md`](source_localization.md) §8.5.1 has no published number to sit beside here either. DIPT's partial-supervision sweep (Table 3) is the nearest relative and it is a different question.
- **No published method searches over decoding _objectives_.** DITTO argues for barycenter over MLE from a flat-likelihood-in-`β̂` premise that does not apply to a per-edge kernel (§2.3), and nobody has revisited that choice for a learned kernel. Whether MLE, barycenter, or something the search finds is right in our setting is genuinely open, and there is no prior result either way.
- **Trajectory log-likelihood has no baseline.** §8.1 marks it "**nobody**: this is ours to add", and §2.8 makes it the natural unlabelled internal signal for a decoder. That means it is both our most distinctive metric and one we cannot benchmark against anything.
- **The reward-specification hazard of §2.6 is undocumented in this literature**, because no prior method has an outer loop that could be gamed by it. Zong and DIPT both note the node-set/tree difficulty split as a _reporting_ concern; neither has reason to treat it as an objective-design concern.
- **Arm 4 is not merely expensive, it may be unrunnable at useful $P$.** At $10^4$ oracle calls per instance (§2.4.2) times $R$ Monte Carlo draws, the sampling arm's budget is dominated by a constant nobody has published, because no prior work evaluates a transition kernel this many times. The number will have to be measured before the arm can be scoped, and it is possible the honest reported result is "arm 4 does not complete", which is itself a finding but needs stating as one rather than as a missing row.
