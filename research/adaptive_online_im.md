# Adaptive, Online, and Continuous-Time Influence Maximization: Prior Work, Datasets, and Published Results

The **sequential-decision** family of influence maximization. Static IM ([`influence_maximization.md`](influence_maximization.md)) commits one seed set and walks away; every variant here keeps choosing. Five branches: **adaptive IM** (seed in rounds, observe realized activations between them), **online/bandit IM** (edge probabilities unknown, learned across repeated campaigns), **continuous-time IM** (transmission times drawn from a distribution, no discrete rounds), **dynamic/streaming IM** (the graph itself changes), and **multi-round IM** (budget split across campaigns).

This is the task family our methodology fits best: every one of them is a loop over `f_θ(G, s_t, a_t) → s_{t+1}`, which is the object we already learn. §2 makes that argument concretely against the generator code.

All URLs checked for HTTP 200 on **2026-07-28** unless annotated otherwise.

---

## 0. Verification policy

| Tier           | Meaning                                                                                                              |
| -------------- | -------------------------------------------------------------------------------------------------------------------- |
| **[verified]** | Read from the paper's own table via text extraction, or from the data repository's own statistics page. Trustworthy. |
| **[derived]**  | Computed by us from a file we downloaded, or reconstructed arithmetically from published splits.                     |
| **[figure]**   | Read off a plotted figure: the paper published no table. Approximate, direction only.                               |
| **[claim]**    | Stated in prose by a paper or a secondary source; not cross-checked against a file or table.                         |

**Automated PDF summarizers hallucinate plausible numbers from these papers.** Every `[verified]` cell below came out of `pdftotext -layout`. This literature is unusually easy to get wrong because it publishes _ratios_ (adaptivity gaps, approximation factors) that look like spread numbers: a gap of `4` and a spread of `4` are not the same kind of `4`.

**Edge-count convention.** Undirected graphs are quoted as _undirected edges_; directed graphs as _arcs_. Our loaders report `adjacency.nnz`, which for a symmetrized undirected graph is **2× the undirected edge count**.

---

## 1. Task definition

### 1.1 Adaptive IM

Non-adaptive IM picks `S` once. **Adaptive** IM picks it in `r` rounds of `k/r` seeds each, observing the realized diffusion between rounds:

```
for i = 1..r:
    S_i  = π(G, φ_{<i})            # policy sees the feedback so far
    φ_i  = realized diffusion from S_i     # observation
maximize  E_φ [ |⋃ activated| ]
```

The observation `φ` is the **feedback model**, and the choice of feedback model is the entire technical content of the subfield:

| Feedback model               | What the policy observes after seeding round `i`                                        | Consequence                                                                |
| ---------------------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **Full-adoption**            | the complete cascade from `S_i`: every node it ever activated, and every edge it tried | adaptive submodularity holds under some conditions; greedy has a guarantee |
| **Myopic**                   | only the nodes activated **one step** out of `S_i`                                      | _not_ adaptively submodular; needs its own analysis                        |
| **Partial / τ-delayed**      | the cascade truncated at `τ` steps                                                      | interpolates the two above                                                 |
| **Node-level / contingency** | whether a chosen seed actually _accepted_ the invitation                                | the "unknown willingness" variant                                          |

`s_t = (infected, frontier)` in our state encoding is exactly the boundary between these: `infected` is full-adoption feedback, `frontier` is myopic feedback. The distinction the literature spent a decade on is two channels of our input tensor.

The headline quantity is the **adaptivity gap**: the ratio between the optimal adaptive policy's expected spread and the optimal non-adaptive seed set's:

```
gap = max_π E[σ(π)]  /  max_{|S|=k} E[σ(S)]     ≥ 1
```

A gap near 1 means adaptivity is worthless and static IM already solved it. Every theoretical paper in §3.1 either upper- or lower-bounds this number.

### 1.2 Online / bandit IM

The graph is known, the **edge probabilities are not**. A campaign is repeated over `T` rounds; each round the learner picks `S_t`, observes some feedback (semi-bandit = which edges activated), and updates its estimate. The objective is **cumulative regret** against the `(1−1/e)`-approximate benchmark:

```
R(T) = Σ_{t=1..T} [ (1−1/e)·σ(S*) − E σ_t(S_t) ]
```

Two feedback regimes: **edge-level semi-bandit** (see which edges fired (IMLinUCB, CUCB) and **node-level** (see only who got activated) harder).

### 1.3 Continuous-time IM

Replace "steps" with time. Each edge `u→v` carries a **transmission-time density** `f(t_v − t_u ; α_uv)` (exponential, Rayleigh, or Weibull) and a node activates at the earliest arrival. Spread is measured within a **time horizon `T`**: `σ(S, T) = E[|{v : t_v ≤ T}|]`. IC is the discrete-time degenerate case. This is the one branch that needs a genuinely different `T_endo` (§2.4).

### 1.4 Dynamic / streaming IM

The graph changes: edge/node insertions and deletions arrive as a stream, and the seed set must be maintained (or re-picked) without recomputing from scratch. The metric is usually _update throughput_ and _index memory_, with spread as a quality check.

### 1.5 Multi-round IM

`r` independent campaigns, each with budget `k`, on the same graph; a node activated in round `i` counts once in the union but can be re-seeded. Distinct from adaptive IM in that rounds are separate _diffusions_, not one diffusion observed in stages.

---

## 2. Fit with our methodology

**This is the purest showcase for a world model in the whole folder.** Static IM is a one-shot combinatorial problem that a world model happens to help with. Adaptive IM _is_ a sequential decision problem over a transition function: the exact object `f_θ` is.

### 2.1 The mapping is an identity, not an analogy

| Element            | Adaptive IM                                   | What we already have                                                     |
| ------------------ | --------------------------------------------- | ------------------------------------------------------------------------ |
| **State** `s_t`    | realized activations after round `t`          | `(infected, frontier)`: channels 0-1 of `X`                             |
| **Feedback model** | full-adoption vs myopic                       | `infected` = full-adoption, `frontier` = myopic. Both are inputs already |
| **Action** `a_t`   | the round-`t` seed batch, `b = k/r` nodes     | a **bag** of `add_node` ops: the same object `t = 0` already commits    |
| **`T_exo`**        | mark the batch active                         | already deterministic, already implemented                               |
| **`T_endo`**       | one IC/LT step (or the cascade to quiescence) | the structured head                                                      |
| **Objective**      | `E[\|⋃ activated\|]` over the policy          | final `\|infected\|`, MC-replayed by `--referee` (with `--mc-agreement` for the NDlib replay)                         |
| **Metric**         | adaptivity gap, spread vs non-adaptive        | needs a multi-round evaluation loop (§9)                                 |

### 2.2 Our generator already emits adaptive-IM transitions

This is not aspirational: it is what `data/generate_wm_data.py` does today. Verified against the episode loop (`run_episode`, lines ~260-340):

- **`t = 0`** commits `seed_bag = [ActionOp("add_node", node) for node in seeds]`: a whole seed set as one bag of `add_node` ops. That is round 1 of an adaptive policy.
- **`t > 0`** calls `sample_injection(...)` with `p_inject=config.inject_p`, which returns a fresh bag of ops with probability `--inject-p` and `NULL` otherwise. **That is round `i > 1`**: a mid-cascade seed injection made after the previous round's diffusion has been observed.
- **Counterfactual forks** (`--cf-prob`, default 0.2; `--cf-branches`, default 2) snapshot the simulator, replay a _different_ action bag from the _same_ `s_t`, and write it as a `cf_i` branch. That is the counterfactual signal an adaptive policy needs: "what would round `i` have yielded had I seeded elsewhere?"
- **`--mc-marginals` (default 30)** re-runs each step to give soft `P(infected)` / `P(frontier)` targets: a per-node estimate of exactly the quantity an adaptive greedy step is trying to rank candidates by.

The one thing missing is that `sample_injection` picks the injected action **at random**, not from a policy. Adaptive IM needs the round-`i` batch to be _chosen_, and evaluation needs the rounds to be _budgeted_ (`Σ_i b_i = k`). That is an evaluation-loop change, not a model change: the transition function it would call is already trained and already accepts a bag of `add_node` ops at any `t`.

### 2.3 The value proposition is exact, and it scales with rounds

Non-adaptive greedy pays `O(k · N · R)` Monte-Carlo simulations once. **Adaptive greedy pays that per round**, because after each round the realized state `φ_i` has changed and every candidate's marginal gain must be re-estimated _under the new state_. Han et al.'s own setting is `k = 500` seeds in `r = 50` batches [verified, §6.1]: fifty complete re-estimations. Cached marginal gains (CELF's trick) do not survive a state change, which is precisely why the adaptive literature had to reinvent RR-set machinery (§3.1) instead of reusing CELF.

Our model replaces each re-estimation with **one forward pass per step of a batched rollout**. The saving is not "one MC call → one forward pass"; it is "`r × (candidates × R)` MC calls → `r × (candidates × horizon)` forward passes, all batched". The ratio grows linearly in `r`, which is the parameter that defines the task.

**This is where the six-condition table earns its keep.** Conditions 3-6 (`pipeline/conditions.py`) hold the method fixed and vary only the evaluator: native (real executions) / `monte_carlo` (`--mc-runs`, default 200) / `oracle` / `world_model`. On static IM the MC arm's cost is a constant; on adaptive IM it is **multiplied by `r`**, so the cost axis of the table finally has a slope. The oracle arm stays the ceiling of model-based guidance and remains IC-only (LT has no oracle, see `research_notes/Baselines - Oracle vs MC vs GWM.md` §2).

### 2.4 What each branch actually costs us

| Branch                         | New `T_endo`? | New simulator? | Cost                                                                                                                                                                                                                                                       |
| ------------------------------ | ------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **(a) Adaptive IM**            | no            | no             | a multi-round evaluation loop + budget bookkeeping. Cheapest item in this folder.                                                                                                                                                                          |
| **(b) Online / bandit IM**     | no            | no             | hide `edge_weight` from the encoder, add a per-round exploration policy and a regret accumulator. The **w-hidden ablation** already flagged in `research_notes/Baselines - Oracle vs MC vs GWM.md` §4 is literally the bandit setting's information state. |
| **(c) Continuous-time IM**     | **yes**       | **yes**        | see below: the honest cost.                                                                                                                                                                                                                               |
| **(d) Dynamic / streaming IM** yes | no            | no             | yes **built**: `coding_agent/stream.py`. `--edit-rate` puts a deterministic schedule of exogenous edge edits into the action bag, so no environment changed. Applied to every arm; an adaptive policy is handed the current graph, a static plan cannot react. |
| **(e) Multi-round IM** yes       | no            | no             | yes **built**: `coding_agent/envs/multi_round_env.py`. `--campaigns r` runs r separate diffusions and scores their union. Wraps any evaluator, so it composes with all four.                                                                                 |

**Be honest about (c).** Continuous-time IM is _not_ a loop over our transition function. It needs:

1. A **transmission-time density** per edge (exponential / Rayleigh / Weibull) replacing the scalar probability `p(u→v)`. Our `GraphInput.edge_weight` is a scalar and the IC head's `q(u→v) = sigmoid(MLP([h_u, h_v, w_uv]))` composes probabilities, not hazards.
2. A **new simulator**: NDlib's IC/LT are discrete-step. Continuous-time simulation is an event queue over sampled arrival times, i.e. a new `T_endo` and a new `wm_simulator.py` path.
3. A **time horizon `T`** as a first-class objective parameter; spread is `σ(S, T)`, not `σ(S)`.
4. A different state: `s_t` becomes _activation times_, not binary flags. Two channels become one continuous one.

That is a new dynamics family, on the order of the DC power-flow cost the screened-out cascading-failure review priced, not the near-free reuse that (a), (b), (d), (e) are. **Our recommendation is to treat continuous-time IM as documented-but-deferred** and spend the effort on (a), where the payoff is largest and the marginal cost is a loop.

---

## 3. Classical and heuristic methods

> `ACM/IEEE/SIAM` links below return **403/202 to `curl`** (bot protection) but resolve in a browser; each was confirmed through the Crossref DOI API instead and is annotated `[DOI-verified]`. arXiv links returned HTTP 200 directly.

### 3.1 Adaptive IM: theory and algorithms

| Method / result                             | Year    | Venue        | Idea                                                                                                                                                              | Paper                                                                                                                                              | Code                                                                                                                         |
| ------------------------------------------- | ------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Adaptive submodularity** (key)               | 2011    | JAIR         | The framework the whole subfield stands on: defines adaptive monotonicity + adaptive submodularity, proves adaptive greedy is `(1 − 1/e)`-optimal when both hold. | [arXiv 1003.3967](https://arxiv.org/abs/1003.3967) · [JAIR](https://www.jair.org/index.php/jair/article/view/10714)                                |                                                                                                                              |
| **Adaptive Seeding**                        | 2013    | FOCS         | Two-stage: spend part of the budget recruiting nodes, then seed their _neighbours_ once revealed. A different adaptivity axis from round-based seeding.           | [DOI 10.1109/FOCS.2013.56](https://doi.org/10.1109/FOCS.2013.56) `[DOI-verified]`                                                                  | no public code found                                                                                                         |
| **Locally Adaptive Optimization**           | 2015/16 | SODA         | Approximation algorithms for adaptive seeding of monotone submodular functions.                                                                                   | [DOI 10.1137/1.9781611974331.ch31](https://doi.org/10.1137/1.9781611974331.ch31) `[DOI-verified]`                                                  | no public code found                                                                                                         |
| **Adaptive IM in Dynamic Social Networks**  | 2017    | IEEE/ACM ToN | Adaptive seeding where the graph _also_ evolves: the (a)+(d) intersection.                                                                                       | [arXiv 1506.06294](https://arxiv.org/abs/1506.06294) · [DOI 10.1109/TNET.2016.2563397](https://doi.org/10.1109/TNET.2016.2563397) `[DOI-verified]` | no public code found                                                                                                         |
| **Why Commit when You can Adapt?**          | 2016    | arXiv        | Early empirical case that adaptivity helps; introduces the adaptive-vs-non-adaptive experimental protocol most later papers reuse.                                | [arXiv 1604.08171](https://arxiv.org/abs/1604.08171)                                                                                               | no public code found                                                                                                         |
| **Partial feedback (`No Time to Observe`)** | 2017    | IJCAI        | Don't wait for the cascade to finish: seed after `τ` steps. Interpolates myopic↔full-adoption.                                                                   | [arXiv 1609.00427](https://arxiv.org/abs/1609.00427)                                                                                               | no public code found                                                                                                         |
| **AdaptGreedy / EPIC** (key)                   | 2018    | **PVLDB 11** | The practical one: adaptive greedy in batches of `b = k/r`, instantiated with an RR-set IM algorithm per batch; first _scalable_ adaptive IM.                     | [PVLDB vol11 p1029](http://www.vldb.org/pvldb/vol11/p1029-han.pdf)                                                                                 | [kkhuang81/AdaptiveIM](https://github.com/kkhuang81/AdaptiveIM) (author's)                                                   |
| **Multi-Round IM (`MRIM`)**                 | 2018    | KDD          | `r` separate campaigns of `k` seeds; non-adaptive and adaptive variants, both with approximation guarantees.                                                      | [arXiv 1802.04189](https://arxiv.org/abs/1802.04189) · [DOI 10.1145/3219819.3220101](https://doi.org/10.1145/3219819.3220101) `[DOI-verified]`     | [lichao-sun/Multi-Round-Influence-Maximization](https://github.com/lichao-sun/Multi-Round-Influence-Maximization) (author's) |
| **Myopic feedback gap** (key)                  | 2019    | NeurIPS      | Settles the myopic case: adaptive greedy is a `(1 − 1/e)/4`-approximation even though adaptive submodularity fails; adaptivity gap bounded in `[e/(e−1), 4]`.     | [arXiv 1905.11663](https://arxiv.org/abs/1905.11663)                                                                                               | no public code found                                                                                                         |
| **Full-adoption gap**                       | 2019    | ISAAC        | Bounds the adaptivity gap under full-adoption feedback for general graphs and several graph classes.                                                              | [arXiv 1907.01707](https://arxiv.org/abs/1907.01707)                                                                                               | no public code found                                                                                                         |
| **Better full-adoption bounds**             | 2021    | AAAI         | Tightens the above; the current reference numbers for the full-adoption gap.                                                                                      | [arXiv 2006.15374](https://arxiv.org/abs/2006.15374)                                                                                               | no public code found                                                                                                         |

**What the field agrees on.** Three separate results (§5.1) all conclude the adaptivity gap is a **small constant**, not a factor that grows with `n` or `k`. That is the single most important fact for us: adaptive IM is worth doing for _cost_ reasons (you can stop early, you can react to failure) far more than for _spread_ reasons. A paper claiming a large spread win from adaptivity should be read sceptically.

### 3.2 Online / bandit IM

| Method                                         | Year | Venue   | Idea                                                                                                                                                                                | Paper                                                                                                                                                     | Code                                                                                                                                    |
| ---------------------------------------------- | ---- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **CUCB** (key)                                    | 2013 | ICML    | Combinatorial UCB: the general CMAB framework IM bandits are an instance of. Requires only an approximation oracle.                                                                 | [PMLR v28](https://proceedings.mlr.press/v28/chen13a.html)                                                                                                | no public code found                                                                                                                    |
| **CMAB with probabilistically triggered arms** | 2016 | JMLR    | Extends CUCB to arms that fire only probabilistically: the property that makes IC an instance.                                                                                     | [arXiv 1407.8339](https://arxiv.org/abs/1407.8339)                                                                                                        | no public code found                                                                                                                    |
| **OIM (Lei et al.)** (key)                        | 2015 | KDD     | **This is DeepIM's "OIM" baseline** (§5.3). Repeated campaigns; maintains a Beta posterior per edge, explores/exploits, updates from observed activations.                          | [arXiv 1506.01188 (extended)](https://arxiv.org/abs/1506.01188) · [DOI 10.1145/2783258.2783271](https://doi.org/10.1145/2783258.2783271) `[DOI-verified]` | [smaniu/oim](https://github.com/smaniu/oim) (co-author's)                                                                               |
| **IM with Bandits**                            | 2015 | arXiv   | The first framing of IM as a CMAB with per-edge arms; precursor to IMLinUCB.                                                                                                        | [arXiv 1503.00024](https://arxiv.org/abs/1503.00024)                                                                                                      | no public code found                                                                                                                    |
| **IMLinUCB** (key)                                | 2017 | NeurIPS | Linear generalization over edge features → regret **independent of the number of edges**, scaling with a graph-topology constant instead. The strongest theory result in bandit IM. | [arXiv 1605.06593](https://arxiv.org/abs/1605.06593)                                                                                                      | no author code; third-party temporal port [olety/TIMLinUCB](https://github.com/olety/TIMLinUCB)                                         |
| **DILinUCB (diffusion-independent)**           | 2017 | ICML    | Learns per-node _reachability_ rather than edge probabilities: no diffusion model assumed.                                                                                         | [arXiv 1703.00557](https://arxiv.org/abs/1703.00557)                                                                                                      | no public code found                                                                                                                    |
| **Improved regret for triggered arms**         | 2017 | NeurIPS | Removes a `1/p*` factor from CUCB-style bounds under a triggering-probability-modulated condition.                                                                                  | [arXiv 1703.01610](https://arxiv.org/abs/1703.01610)                                                                                                      | no public code found                                                                                                                    |
| **IMFB (factorization bandits)**               | 2019 | KDD     | Factorizes edge probability into node latent vectors; bandit over the factors.                                                                                                      | [arXiv 1906.03737](https://arxiv.org/abs/1906.03737)                                                                                                      | no public code found                                                                                                                    |
| **OIM under LT**                               | 2020 | NeurIPS | Online IM for the **linear threshold** model: the LT counterpart of IMLinUCB.                                                                                                      | [arXiv 2011.06378](https://arxiv.org/abs/2011.06378)                                                                                                      | [Ritchiegit/…LinearThresholdModel](https://github.com/Ritchiegit/Online_Influence_Maximization_under_Linear_Threshold_Model) (official) |

### 3.3 Continuous-time diffusion and IM

| Method                        | Year | Venue    | Idea                                                                                                                                                    | Paper                                                                                                                                                              | Code                 |
| ----------------------------- | ---- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------- |
| **NETINF**                    | 2010 | KDD      | Infer the diffusion network from observed infection times. The ancestor of the whole line.                                                              | see network inference                                                                                                                 |                      |
| **NETRATE**                   | 2011 | ICML     | Infer per-edge **transmission rates** (not just structure) from cascades; exponential/power-law/Rayleigh densities.                                     | [arXiv 1105.0697](https://arxiv.org/abs/1105.0697)                                                                                                                 | no public code found |
| **INFLUMAX** (key)               | 2012 | ICML     | Defines continuous-time IM: `σ(S, T)` under a time horizon; proves submodularity in continuous time; greedy with a continuous-time influence estimator. | [arXiv 1205.1682](https://arxiv.org/abs/1205.1682)                                                                                                                 | no public code found |
| **ConTinEst** (key)              | 2013 | NeurIPS  | Randomized neighbourhood-sketch estimator for continuous-time influence: `O(1)` per query after preprocessing; the scalability breakthrough for (c).   | [NeurIPS 2013](https://proceedings.neurips.cc/paper/2013/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html) · [arXiv 1311.3669](https://arxiv.org/abs/1311.3669) | no public code found |
| **ConTinEst journal version** | 2016 | ACM TOIS | Extended treatment of estimation **and** maximization in continuous time.                                                                               | [DOI 10.1145/2824253](https://doi.org/10.1145/2824253) `[DOI-verified]`                                                                                            |                      |
| **InfluLearner**              | 2014 | ICML     | Learns the influence _function_ directly from cascades, model-free: the continuous-time analogue of what our world model does discretely.              | [PMLR v32](https://proceedings.mlr.press/v32/du14.pdf)                                                                                                             | no public code found |
| **Time-critical IM**          | 2012 | AAAI     | Discrete-time with **meeting/delay** probabilities and a deadline: the cheap bridge between IC and continuous time.                                    | [arXiv 1204.3074](https://arxiv.org/abs/1204.3074)                                                                                                                 | no public code found |

### 3.4 Dynamic / streaming IM

| Method                                     | Year | Venue    | Idea                                                                                                                        | Paper                                                                                                                                                           | Code                 |
| ------------------------------------------ | ---- | -------- | --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| **Dynamic influence analysis** (key)          | 2016 | PVLDB 9  | Fully dynamic sketch index; maintains influence estimates under edge insertions/deletions. The reference dynamic-IM system. | [PVLDB vol9 p1077](http://www.vldb.org/pvldb/vol9/p1077-ohsaka.pdf) · [DOI 10.14778/2994509.2994525](https://doi.org/10.14778/2994509.2994525) `[DOI-verified]` | no public code found |
| **Tracking influential nodes**             | 2017 | TKDE     | Maintains a top-`k` influential set under a stream of graph updates.                                                        | [arXiv 1602.04490](https://arxiv.org/abs/1602.04490)                                                                                                            | no public code found |
| **Real-time IM on dynamic social streams** | 2017 | PVLDB 10 | Sliding-window IM over an _edge stream_, not a static snapshot.                                                             | [arXiv 1702.01586](https://arxiv.org/abs/1702.01586)                                                                                                            | no public code found |
| **Dynamic IM**                             | 2021 | NeurIPS  | Theory: amortized-time algorithms maintaining a `(1 − 1/e − ε)` seed set under updates.                                     | [arXiv 2110.13355](https://arxiv.org/abs/2110.13355)                                                                                                            | no public code found |

---

## 4. Learning-based methods

### 4.1 The "is it actually sequential?" audit

`influence_maximization.md` §4 catalogues DISCO, PIANO, GCOMB, and ToupleGDD as deep-RL IM methods. **They are RL over seed-set construction, not over diffusion**: a distinction that matters enormously here and is routinely blurred in survey tables.

| Method                           | MDP state                                                                                                      | Feedback between actions                             | Genuinely adaptive?               |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | --------------------------------- |
| **S2V-DQN / DISCO / PIANO**      | partial seed set                                                                                               | none: the graph never changes and no cascade is run | no sequential _construction_ only |
| **ToupleGDD**                    | `S_t` = a `\|V\|`-dimensional binary vector, component `u` is 1 iff `u ∈ S_t` [verified, arXiv 2210.07500 §IV] | none                                                 | no sequential _construction_ only |
| **GCOMB**                        | partial solution over a pruned candidate set                                                                   | none                                                 | no sequential _construction_ only |
| **DeepIM**                       | latent seed-set code                                                                                           | n/a (optimizes in latent space)                      | no one-shot                       |
| **RL4IM (contingency-aware)** (key) | invited-so-far + observed **willingness** of previously invited nodes                                          | yes: observes who accepted                          | yes multi-round with feedback      |
| **CHANGE / Kamarthi et al.**     | sampled subgraph + queried nodes                                                                               | yes: observes query results                         | yes adaptive _sampling_, then seed |

The consequence: **the entire deep-RL-for-IM line is a baseline for static IM, not for this file.** Only the bottom two rows are prior work for adaptive IM. That is a small field, and it is the gap our method is aimed at.

### 4.2 Learning methods for the sequential variants

| Method                                                         | Year | Venue      | Approach                                                                                                                                                                                                             | Paper                                                  | Code                                                                             |
| -------------------------------------------------------------- | ---- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ | -------------------------------------------------------------------------------- |
| **Maximizing Influence in an Unknown Social Network (CHANGE)** | 2018 | AAAI       | Sample the network by querying nodes, then seed. The "unknown graph" adaptive setting from HIV-prevention fieldwork.                                                                                                 | see RL4IM's citation                                   | no public code found                                                             |
| **Learning Policies for Effective Graph Sampling**             | 2020 | AAMAS      | RL policy for _which nodes to query_ when the graph is unknown, then IM on the discovered subgraph.                                                                                                                  | [arXiv 1907.11625](https://arxiv.org/abs/1907.11625)   | no public code found                                                             |
| **RL4IM (Contingency-Aware IM)** (key)                            | 2021 | UAI        | The closest published thing to our setting: IM as a genuine multi-round MDP, `T` intervention rounds × `B` seeds per round, observing whether each invited node _accepted_. S2V embedding + DQN + state abstraction. | [arXiv 2106.07039](https://arxiv.org/abs/2106.07039)   | [wmd3i/RL4IM-Contingency](https://github.com/wmd3i/RL4IM-Contingency) (official) |
| **InfluLearner**                                               | 2014 | ICML       | Learns the continuous-time influence function from cascades: model-free, no IC/LT assumed.                                                                                                                          | [PMLR v32](https://proceedings.mlr.press/v32/du14.pdf) | no public code found                                                             |
| **Neural mean-field dynamics**                                 | 2021 | arXiv/JMLR | Learns influence estimation _and_ maximization by unrolling a mean-field ODE: a learned continuous-time `T_endo`. The nearest published analogue to a continuous-time world model.                                  | [arXiv 2106.02608](https://arxiv.org/abs/2106.02608)   | no public code found                                                             |

**Nothing in this table learns a transition function and plans against it.** RL4IM learns a _policy_ (`Q(s, a)`), not a _model_ (`f_θ(s, a) → s'`). The model-based/model-free split that defines our contribution has, as far as this review found, no adaptive-IM precedent at all: see §11.

---

## 5. Published results

### 5.1 Adaptivity gaps (key): how much adaptivity is worth at all

These are **ratios, not spreads**. Every entry is `E[adaptive OPT] / E[non-adaptive OPT]` under the IC model. Read this table before designing any adaptive-IM experiment: it tells you the size of the effect you are chasing.

| Feedback          | Graph class              | Lower bound       | Upper bound                                              | Source                                     |
| ----------------- | ------------------------ | ----------------- | -------------------------------------------------------- | ------------------------------------------ |
| **Myopic**        | general                  | `e/(e−1)` ≈ 1.582 | **4**                                                    | Peng & Chen 2019 [verified, abstract + §1] |
| **Full-adoption** | general                  |                   | **`⌈n^{1/3}⌉`** (first sub-linear bound for _any_ graph) | D'Angelo et al. 2021 [verified, abstract]  |
| **Full-adoption** | in-arborescence          | `(e−1)/e`         | `2e/(e−1)` ≈ 3.16                                        | Chen & Peng 2019 [verified, abstract]      |
| **Full-adoption** | in-arborescence          |                   | **`2e²/(e²−1)` ≈ 2.31** (improved)                       | D'Angelo et al. 2021 [verified, abstract]  |
| **Full-adoption** | out-arborescence         | `(e−1)/e`         | `2`                                                      | Chen & Peng 2019 [verified, abstract]      |
| **Full-adoption** | α-bounded undirected     |                   | `√α + O(1)`                                              | D'Angelo et al. 2021 [verified, abstract]  |
| **Full-adoption** | 0-bounded (paths/cycles) |                   | `3e³/(e³−1)` ≈ 3.16                                      | D'Angelo et al. 2021 [verified, abstract]  |

Approximation ratios **with respect to the adaptive optimum**, myopic feedback [verified, Peng & Chen 2019 abstract]:

| Algorithm           | Lower bound          | Upper bound                         |
| ------------------- | -------------------- | ----------------------------------- |
| non-adaptive greedy | `¼(1 − 1/e)` ≈ 0.158 | `(e²+1)/(e+1)²` ≈ 0.579 `< 1 − 1/e` |
| adaptive greedy     | `¼(1 − 1/e)` ≈ 0.158 | `(e²+1)/(e+1)²` ≈ 0.579 `< 1 − 1/e` |

> **The finding to internalize.** Peng & Chen prove that under myopic feedback _"the approximation ratio of the non-adaptive greedy algorithm is no worse than that of the adaptive greedy algorithm, when considering all graphs"_ [verified, abstract]. Adaptivity, in the worst case, buys **nothing** in approximation quality, and at most a constant factor `≤ 4` in spread. Any adaptive-IM experiment we run that reports a large spread win over non-adaptive is measuring an instance effect, not a general one, and any that reports a _small_ win is consistent with theory, not a failure.
>
> This reframes what our contribution should claim. **The win from a world model in adaptive IM is not spread: it is cost.** The MC arm must pay `r` times over to reach a ceiling that theory caps at 4× (and typically far less); we pay once in training. That is the honest headline, and §2.3 is where it is argued.

### 5.2 Han et al., PVLDB 2018: the scalable adaptive-IM reference

Datasets [verified, Table 1]: all five are already catalogued in [`influence_maximization.md`](influence_maximization.md) §6.2:

| Dataset          | n     | m     | Type       | Avg. degree |
| ---------------- | ----- | ----- | ---------- | ----------- |
| **NetHEPT** yes   | 15.2K | 31.4K | undirected | 4.18        |
| Epinions         | 132K  | 841K  | directed   | 13.4        |
| DBLP (classical) | 655K  | 1.99M | undirected | 6.08        |
| LiveJournal      | 4.85M | 69.0M | directed   | 28.5        |
| Orkut            | 3.07M | 117M  | undirected | 76.2        |

Protocol [verified, §6.1]: `δ = 1/n`, `ξ = 0.1`. Two sweeps,

- **b-setting**: fix `k = 500`, vary batch size `b ∈ {1, 2, 5, 10, 20, 50, 500}`. `b = 500` is the non-adaptive case (IMM and D-SSA can only be run there).
- **k-setting**: fix `r = 50` rounds, vary `k ∈ {50, 100, 200, …, 500}`.

**All spread and running-time results are figure-only** (Figs. 3-6); the paper publishes no result table. Two prose findings are transcribable: AdaptIM-2 runs _"almost 4 times faster than AdaptIM-1 on the Epinions dataset when `b = 1`"_, and AdaptIM-1 _"cannot finish under the case of `b < 5` for the largest datasets LiveJournal and Orkut, due to the memory overflow"_ [both verified, §6.2 prose].

That second sentence is the whole argument for this file in one line: **the state-of-the-art scalable adaptive-IM algorithm runs out of memory when the number of rounds gets large.** Rounds are the expensive axis, and rounds are exactly what a learned transition function makes cheap.

### 5.3 OIM (Lei et al., KDD 2015): **this is DeepIM's `OIM` baseline** (key)

**Identification, settled.** DeepIM's reference list contains exactly one entry matching its `OIM` row: _"Lei, S., Maniu, S., Mo, L., Cheng, R., and Senellart, P. Online influence maximization. In Proc. of the KDD, 2015"_ [verified, DeepIM PMLR PDF, references]. DeepIM's own §Baselines groups it as **"Online IM: OIM (Lei et al., 2015)"** [verified, DeepIM PMLR PDF §5.1], and its discussion says `OIM` _"achieves better performance than traditional methods in most datasets because it can automatically update the edge weight iteratively"_ while _"it is tailored for the specific IC diffusion model"_ [verified, same PDF §5.2], which is why `OIM` has **no LT row** in DeepIM's Table 3.

**Do not re-transcribe its numbers here.** The `OIM` row of DeepIM's Tables 2 and 4 is already transcribed in [`influence_maximization.md`](influence_maximization.md) §5.1 (both the small- and large-graph IC tables). Those cells are the only published OIM numbers on graphs we load, and they live in exactly one place by design.

The one thing worth adding is **OIM's own evaluation**, which is a _different protocol_ from the one DeepIM re-ran it under:

Datasets [verified, Lei et al. Table 2]:

| Dataset          | # Nodes | # Edges | Avg. degree | Max. degree |
| ---------------- | ------- | ------- | ----------- | ----------- |
| **NetHEPT** yes   | 15K     | 59K     | 7.73        | 341         |
| **NetPHY** yes    | 37K     | 231K    | 12.46       | 286         |
| DBLP (classical) | 655K    | 2.1M    | 6.1         | 588         |

Protocol [verified, Lei et al. §8]: over `N` trials the simulator runs **a single IC simulation per trial**, returns edge-level feedback `F_n = (i, j, a_ij)` and the activated set `A_n`; the reported metric is the **union over trials**, `|⋃_{n=1..N} A_n|`, averaged over **10 repetitions**.

That union-over-trials metric is _not_ expected spread: it is cumulative distinct reach across a campaign sequence, the online-IM analogue of IMINFECTOR's DNI. **Comparing OIM's own published numbers to a single-campaign spread number is invalid.** DeepIM sidestepped this by re-running OIM under its own single-campaign protocol, which is why the DeepIM table is the usable one.

### 5.4 IMLinUCB (Wen et al., NeurIPS 2017) (key): the regret result

Setting: IC with **edge-level semi-bandit feedback**, `n` rounds, seed budget `K`, `L = |V|`, an `(α, γ)`-approximation offline oracle, and a `d`-dimensional linear generalization `w(e) = x_eᵀ θ*`.

| Quantity                          | Bound                                                | Tier                        |
| --------------------------------- | ---------------------------------------------------- | --------------------------- |
| Scaled cumulative regret, general | `R^{αγ}(n) ≤ Õ( d · C* · √(\|E\| · n) / (αγ) )`      | [verified, Theorem 1 eq. 5] |
| Tabular case (`X = I`)            | `R^{αγ}(n) ≤ Õ( (L − K) · \|E\|^{3/2} · √n / (αγ) )` | [verified, Theorem 1 eq. 8] |
| Star topology                     | `R(n) = Õ(L²)`                                       | [verified, Table 1 / §5.1]  |
| Ray topology                      | `R(n) = Õ(L^{9/4})`                                  | [verified, Table 1 / §5.1]  |

`C*` is the paper's new complexity metric, **maximum observed relevance**: a function of graph topology and activation probabilities. The headline is that the bound is **polynomial in every quantity of interest, has near-optimal dependence on `n`, and does not depend on `1/p_min` or on the cardinality of the action set** [verified, §1], the two quantities that made earlier IM-bandit bounds vacuous.

Empirical validation of the topology exponents, `n = 10⁴` steps, `K = 1`, uniform edge weight `ω` [verified, §5.1]:

| Topology | `ω` | Estimated growth | Predicted    |
| -------- | --- | ---------------- | ------------ |
| star     | 0.8 | `O(L^2.040)`     | `Õ(L²)`      |
| star     | 0.7 | `O(L^2.056)`     | `Õ(L²)`      |
| ray      | 0.8 | `O(L^2.488)`     | `Õ(L^{9/4})` |
| ray      | 0.7 | `O(L^2.467)`     | `Õ(L^{9/4})` |

Real-graph experiment [verified, §5.2]: a **Facebook subgraph** with `L = 327` nodes and `|E| = 5,038` directed arcs; `w(e) ~ U(0, 0.1)` sampled and treated as ground truth; `n = 5,000` rounds; `K = 10`; `d = 10` edge features built as the element-wise product of node2vec node embeddings; cumulative regret averaged over **10 independent runs**. Result is **figure-only** (Fig. 2b): IMLinUCB significantly below CUCB [figure].

Note the scale. The strongest theoretical result in bandit IM was validated on a **327-node** graph. Every graph we load except `jazz` is larger.

### 5.5 RL4IM (Contingency-Aware IM, UAI 2021)

The only published RL method that is genuinely multi-round (§4.1). Setup
[verified, §5.1]: synthetic **powerlaw-cluster** graphs (BA growth + triangle
step), average degree **3**, triangle probability **0.05**; IC propagation probability **0.1**; each influence number is an average over **100** simulator runs. Defaults: `|V| = 200`, `T = 2` intervention rounds, `B = 4` seeds per round, 200 training graphs, willingness `q = 0.6` [verified, §5.2]. Graph sizes swept: `{50, 100, 200, 500, 1000}` [verified, §5.2]. Test = 10 unseen graphs × 20 runs = **200 runs** per setting [verified, §5.1].

All results are **figure-only** (Figs. 3-5); no result table is published [verified: the PDF contains no `Table` float in §5].

**The scale point again.** State of the art in RL-for-adaptive-IM is `|V| = 200`, `T = 2` rounds, `B = 4`. Our `jazz` graph (198 nodes) is the same size, and our `--horizon` default is 10. There is a lot of headroom here.

---

## 6. Datasets

### 6.0 What `--dataset` accepts for this task

**Every one of the 71 real graph loaders and all 7 synthetic families (`er`, `ba`, `ws`, `sbm`, `powerlaw_cluster`, `kronecker`, `karate`) runs on this task.** The loaders are task-agnostic; nothing in `pipeline/tasks.py` restricts a task to a dataset. The lists below are therefore an experimental CHOICE, not a constraint, and the only place that choice is currently encoded is `sbatch/`. Where the sweep and the benchmark family disagree, the sweep is the accident and the family is the intent.

| | Datasets |
| --- | --- |
| **Benchmark family** (§6.1, the static IM suite re-used) | `jazz`, `cora_ml`, `facebook`, `power_grid`, `nethept`, `netphy`, `digg`, `netscience` |
| **Synthetic** | `powerlaw_cluster` (RL4IM's own family), plus `er`, `ba`, `ws`, `sbm`, `karate`, `kronecker` |
| **Scale targets** (Han et al.'s, where their own method OOMs) | `twitter`, `youtube`, `livejournal` |
| **In `sbatch/adaptive_online_im/` today** | `powerlaw_cluster`, `nethept` |

This literature adds almost no new graphs: it re-uses the static IM suite and changes the PROTOCOL. `netscience` and `digg` both carry version caveats (§6.1).

**Good news up front: this literature adds almost no new graphs.** Adaptive, online, and dynamic IM re-use the _static_ IM benchmark suite and change the _protocol_, not the data. The authoritative rows for every graph below live in [`influence_maximization.md`](influence_maximization.md) §6.2, this section records only which of them this family uses, plus the two genuinely new items (§6.3).

### 6.1 What we already load

| Dataset                   | Nodes   | Edges                             | Type                       | Avg deg | Used here by                                                     |
| ------------------------- | ------- | --------------------------------- | -------------------------- | ------- | ---------------------------------------------------------------- |
| `jazz` yes                 | 198     | 2,742 undirected                  | undirected                 | 27.7    | DeepIM's OIM row [verified]                                      |
| `cora_ml` yes              | 2,810   | 7,981 undirected                  | undirected                 | 5.7     | DeepIM's OIM row [verified]                                      |
| `facebook` yes             | 4,039   | 88,234 undirected                 | undirected                 | 43.7    | IMLinUCB uses a **327-node subgraph** of it (§6.2)               |
| `power_grid` yes           | 4,941   | 6,594 undirected                  | undirected                 | 2.67    | DeepIM's OIM row [verified]                                      |
| `nethept` yes              | 15,229  | 62,752 arcs (= 31,376 undirected) | undirected, stored as arcs | 4.18    | **Han et al. 2018** [verified] · **Lei et al. OIM** [verified]   |
| `netphy` yes               | 37,154  | 174,161 undirected                | undirected                 | 9.38    | **Lei et al. OIM** [verified]                                    |
| `digg` yes (ours ≠ theirs) | 116,893 | ≈2.6M                             | undirected                 | ≈45     | DeepIM's OIM row uses the **ISI/Lerman** Digg, a different graph |

`netscience` yes also appears in DeepIM's OIM row, but on DeepIM's 1,565/13,532 version, which is **not the graph we load**: see [`influence_maximization.md`](influence_maximization.md) §6.3.

> **Two of our graphs are the exact graphs the two most usable adaptive/online papers report on.** Han et al. (PVLDB 2018) and Lei et al. (KDD 2015) both use NetHEPT; Lei et al. also uses NetPHY. Note the edge-count convention: Lei et al. quote NetHEPT as `59K` edges, which is Wei Chen's **raw line count**, not deduplicated pairs, the same 58,891-vs-31,376 reconciliation already worked out in [`influence_maximization.md`](influence_maximization.md) §6.4.

### 6.2 Graphs this literature adds

| Dataset                          | Nodes     | Edges                  | Type       | Avg deg                  | Download                                                                                                                    | Used by                                                               |
| -------------------------------- | --------- | ---------------------- | ---------- | ------------------------ | --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Epinions (signed) yes             | 131,828   | 841,372 arcs           | directed   | 13.4                     | [SNAP](https://snap.stanford.edu/data/soc-sign-epinions.html)                                                               | Han et al. 2018 (as `132K / 841K`) [verified]                         |
| DBLP (classical) no              | 655,000   | 1.99M undirected       | undirected | 6.08                     | Wei Chen / arnetminer release, **no URL located**, see §11                                                                 | Han et al. 2018 [verified] · Lei et al. (as `655K / 2.1M`) [verified] |
| soc-LiveJournal1 yes              | 4,847,571 | 68,993,773 arcs        | directed   | 28.5                     | [SNAP](https://snap.stanford.edu/data/soc-LiveJournal1.html)                                                                | Han et al. 2018 (as `4.85M / 69.0M`) [verified]                       |
| com-Orkut yes                     | 3,072,441 | 117,185,083 undirected | undirected | 76.2                     | [SNAP](https://snap.stanford.edu/data/com-Orkut.html)                                                                       | Han et al. 2018 (as `3.07M / 117M`) [verified]                        |
| **Facebook subgraph (IMLinUCB)** | **327**   | **5,038 arcs**         | directed   | 15.4 [derived: 5038/327] | derived from [SNAP ego-Facebook](https://snap.stanford.edu/data/ego-Facebook.html); **the exact subgraph is not published** | IMLinUCB [verified, §5.2]                                             |

Every row except the last is already catalogued authoritatively in [`influence_maximization.md`](influence_maximization.md) §6.2, and the first four carry the _same_ node/edge counts there: Han et al.'s Table 1 agrees with SNAP to the digit. **No version collision was found in this family.**

The IMLinUCB Facebook subgraph is the one irreproducible item: the paper states `L = 327`, `|E| = 5,038` and cites the SNAP ego-Facebook release, but does not say which ego-network or how it was extracted. Reproducing its regret curve exactly is not possible from the paper alone.

### 6.3 Task-specific data

Two things here are genuinely not graphs-with-a-budget.

**(a) MemeTracker: cascade traces for continuous-time IM.** ConTinEst's real-world evaluation uses **10,967 hyperlink cascades among 600 media sites** [verified, ConTinEst §5], split 80/20 train/test **5 times**, with NETRATE fitting exponential transmission functions on each training split. Time windows swept `T = 5` and `T = 10`; source counts `1…50`.

- Raw MemeTracker: [SNAP memetracker9](https://snap.stanford.edu/data/memetracker9.html) (HTTP 200)
- The 600-site / cascade-formatted release used by this line: [SNAP InfoPath data](https://snap.stanford.edu/infopath/data.html) (HTTP 200)

This is **not simulable data**: it is observed traces with timestamps, which is precisely Yuntong's "real-world data = logged trajectories" scope (`research_notes/Baselines - Oracle vs MC vs GWM.md` §4, Q2). It is the natural holdout for the claim that a learned transition function beats a fitted parametric one.

**(b) Synthetic families with a _continuous-time_ generator.**

| Family                                            | Generator                                                   | Sizes reported                                                             | Source                     |
| ------------------------------------------------- | ----------------------------------------------------------- | -------------------------------------------------------------------------- | -------------------------- |
| Kronecker **core-periphery** `[0.9 0.5; 0.5 0.3]` | Kronecker graph model                                       | 128 / 320 edges; 1,024 / 2,048 edges; up to 1,000,000 nodes at density 1.5 | ConTinEst [verified, §5]   |
| Kronecker **random** `[0.5 0.5; 0.5 0.5]`         | same                                                        | same sweep                                                                 | ConTinEst [verified, §5]   |
| Kronecker **hierarchical** `[0.9 0.1; 0.1 0.9]`   | same                                                        | same sweep                                                                 | ConTinEst [verified, §5]   |
| **powerlaw-cluster**                              | BA growth + triangle step, avg degree 3, triangle prob 0.05 | `\|V\| ∈ {50, 100, 200, 500, 1000}`                                        | RL4IM [verified, §5.1-5.2] |

yes **Both are implemented.** `--dataset powerlaw_cluster` (`--plc-m`, `--plc-p`) is `nx.powerlaw_cluster_graph`, the only synthetic family in this file with a matching published protocol. `--dataset kronecker` (`--kron-variant core_periphery|random|hierarchical`) is a stochastic Kronecker sampler written for this (NetworkX has no generator), carrying ConTinEst's three seed matrices. The construction only defines graphs on `2^k` nodes, so a non-power-of-two `--syn-nodes` is induced from the next power up and the `graph_id` records both (`kron_core_periphery_n100_o128_s42`). Sampling is `O(N²)` because every pair carries its own probability, guarded at 20K nodes.

**Defaults are RL4IM's own, taken from its repo rather than its paper.** `basic_env.yaml` sets `m: 3, p: 0.05`; the paper's "average degree 3" is inconsistent with that (m=3 gives 5.91 at n=200) and the code is the authoritative source. `--plc-m` defaults to 3 accordingly. See §11.

---

## 7. Which paper uses which

Cells mark the dataset **as that paper reports it**. `yes` = reported; `(sub)` = a subgraph, not the full release. Bold rows are graphs we already load.

| Dataset                                          | Lei'15 OIM | IMLinUCB'17        | Han'18 | MRIM'18 | ConTinEst'13 | RL4IM'21 | DeepIM'23 (OIM row) |
| ------------------------------------------------ | ---------- | ------------------ | ------ | ------- | ------------ | -------- | ------------------- |
| **NetHEPT** yes                                   | yes          |                    | yes      |         |              |          |                     |
| **NetPHY** yes                                    | yes          |                    |        |         |              |          |                     |
| **Jazz** yes                                      |            |                    |        |         |              |          | yes                   |
| **Cora-ML** yes                                   |            |                    |        |         |              |          | yes                   |
| **Power Grid** yes                                |            |                    |        |         |              |          | yes                   |
| **Facebook** yes                                  |            | yes (sub, 327 nodes) |        |         |              |          |                     |
| NetScience (DeepIM version)                      |            |                    |        |         |              |          | yes                   |
| Digg (ISI/Lerman)                                |            |                    |        |         |              |          | yes                   |
| Epinions (signed)                                |            |                    | yes      |         |              |          |                     |
| DBLP (classical)                                 | yes          |                    | yes      |         |              |          |                     |
| soc-LiveJournal1                                 |            |                    | yes      |         |              |          |                     |
| com-Orkut                                        |            |                    | yes      |         |              |          |                     |
| MemeTracker (600 sites)                          |            |                    |        |         | yes            |          |                     |
| Kronecker core-periphery / random / hierarchical |            |                    |        |         | yes            |          |                     |
| powerlaw-cluster                                 |            |                    |        |         |              | yes        |                     |
| Synthetic (DeepIM 50K/250K)                      |            |                    |        |         |              |          | yes                   |

**MRIM (KDD 2018) is blank on purpose**: its dataset table was not extracted in this review; see §11.

Two readings of this matrix:

1. **The overlap with our suite is real but narrow.** NetHEPT and NetPHY put us in direct contact with the two best-documented adaptive/online papers (Han et al., Lei et al.); Jazz / Cora-ML / Power Grid put us in contact with DeepIM's OIM row. Nothing else lines up.
2. **There is no shared benchmark across the five branches.** Adaptive IM benchmarks on the classical RIS suite, bandit IM on tiny synthetic topologies plus one Facebook subgraph, continuous-time IM on MemeTracker cascades, and RL-for-adaptive-IM on 200-node synthetics. **This family has no canonical table.** That is a gap, and it is an opportunity: a single suite evaluated across all five protocols would be a contribution in itself.

---

## 8. Evaluation protocol

### 8.1 The metrics, and which branch owns each

| Metric                              | Definition                                                  | Owned by                  | Higher/lower better                                   |
| ----------------------------------- | ----------------------------------------------------------- | ------------------------- | ----------------------------------------------------- |
| **Adaptivity gap**                  | `E[adaptive OPT] / E[non-adaptive OPT]`                     | adaptive theory (§5.1)    | higher = adaptivity matters more                      |
| **Expected spread vs non-adaptive** | `σ(policy) / σ(best static seed set)` at matched budget `k` | adaptive empirics         | higher                                                |
| **Cumulative regret** `R(n)`        | `Σ_{t=1..n} [ αγ·σ(S*) − E σ_t(S_t) ]`                      | bandit IM (§5.4)          | lower; report the `O(·)` in `n`, `L`, `\|E\|` too     |
| **Union reach** `\|⋃_n A_n\|`       | distinct nodes activated across all `n` campaigns           | Lei et al. OIM (§5.3)     | higher, **not** comparable to single-campaign spread |
| **`σ(S, T)`**                       | expected spread **within time horizon `T`**                 | continuous-time IM        | higher; meaningless without `T`                       |
| **# MC calls**                      | simulator invocations to produce the answer                 | everyone, rarely reported | lower                                                 |
| **Wall-clock / memory**             | end-to-end runtime; peak RSS                                | Han et al., Ohsaka et al. | lower                                                 |
| **Update throughput**               | graph edits absorbed per second                             | dynamic/streaming IM      | higher                                                |

### 8.2 The traps

**(1) Regret is scaled by the oracle's approximation factor.** Every bandit-IM bound is `α·γ`-scaled regret, not raw regret: the benchmark is `(1−1/e)·σ(S*)`, not `σ(S*)`. IMLinUCB states this explicitly: _"If ORACLE solves the IM problem exactly (i.e., `α = γ = 1`), then `R^{αγ}(n) = R(n)`"_ [verified, §4]. Quoting a regret number without its `(α, γ)` is meaningless.

**(2) Union reach ≠ spread.** §5.3 already flags this for Lei et al. The same trap appears whenever a paper runs `N` campaigns and reports the union: the number grows monotonically in `N` and cannot be compared against a single-campaign spread. **DeepIM's OIM row avoids this only because DeepIM re-ran OIM under its own protocol.**

**(3) The budget convention splits three ways here**, one worse than [`influence_maximization.md`](influence_maximization.md) §8:

| Convention                                        | Example                  | Who                         |
| ------------------------------------------------- | ------------------------ | --------------------------- |
| absolute `k`, split into `r` batches of `b = k/r` | `k = 500`, `b ∈ {1…500}` | Han et al. [verified, §6.1] |
| `T` rounds × `B` per round                        | `T = 2`, `B = 4`         | RL4IM [verified, §5.2]      |
| percentage of `\|V\|`                             | 1/5/10/20%               | DeepIM's OIM row            |

`k = r·b = T·B` makes the first two commensurable; the third needs `\|V\|`. Our `--budget-pcts 1 5 10 20` speaks only to the third. **To compare against Han et al. we must also report `(k, b)` pairs.**

**(4) Feedback model must be stated or the number is uninterpretable.** A "spread under adaptive greedy" figure means different things under myopic vs full-adoption feedback, and the two have different theory (§5.1). Our state carries both channels, so we have no excuse for leaving it ambiguous: report which channel the policy was allowed to read.

**(5) Continuous-time results are horizon-conditional.** `σ(S, T)` at `T = 5` and `T = 10` are different numbers for the same seed set; ConTinEst reports both [verified, §5]. Any continuous-time row without `T` is unusable.

### 8.3 What our harness already reports vs what this needs

| Needed                                  | Have it? | Where                                                                                                                                                    |
| --------------------------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| expected spread, ground-truth MC replay | yes       | `--referee` (with `--mc-agreement` for the NDlib replay), `--mc-runs` (default 200)                                                                                                                   |
| per-condition cost accounting           | yes       | `real_env_episodes`, `evaluator_calls`, `evaluator_seconds`, `forward_passes` per arm, in `summary.csv` and the report                                    |
| adaptivity gap                          | yes       | `pipeline/conditions.py::adaptivity_gaps`, pairing each `adaptive_*@E` arm with `evolve_*@E` at matched `k` on the shared ground-truth replay             |
| `(k, b, r)` reported together           | yes       | `rounds` / `round_batches` / `round_gap` in the results JSON and `summary.csv`                                                                            |
| feedback model recorded                 | yes       | `--feedback-model`, stamped into every adaptive result                                                                                                   |
| per-round spread curve                  | yes       | `round_spreads`, plus the `round_spreads.png` figure                                                                                                     |
| cumulative regret                       | no       | still needs a repeated-campaign loop and an `(α, γ)`-scaled benchmark: the online/bandit branch (b), which §9.3 never scheduled. See §11                 |
| `σ(S, T)` under a horizon               | yes       | `spread_curve` in every result: ensemble-mean `E\|infected\|` at each `T`, padded to `horizon + 2`, so `σ(S, T)` is readable at any `T ≤ horizon`. `spread_at_horizon` in `summary.csv` |

The cost axis is no longer prose: `evaluator_seconds` is the number §2.3's argument is read on, and the report's adaptivity table prints it for both sides of every pair.

---

## 9. Implications for this project

**Adaptive IM is the purest showcase for a world model in this whole folder, and it is also the cheapest thing left to build.** Those two facts rarely coincide.

### 9.1 Why it is the purest showcase

1. **Every round _is_ one `(s_t, a_t) → s_{t+1}` step.** Not an analogy: the adaptive-IM round structure and our transition function have the same type. `a_t` is a bag of `add_node` ops; `s_t` is `(infected, frontier)`; the feedback model (myopic vs full-adoption) is a choice of which state channel the policy reads (§1.1, §2.1). Static IM had to be _cast_ as a sequential problem to use a world model; adaptive IM already is one.

2. **The MC arm's cost scales with `r`; a forward pass does not.** Adaptive greedy must re-estimate every candidate's marginal gain _after each round_, because the realized state changed and CELF's cached gains are invalid. Han et al. run `r = 50` rounds at `k = 500` [verified, §6.1] and report that their own AdaptIM-1 **cannot finish at `b < 5` on LiveJournal and Orkut due to memory overflow** [verified, §6.2 prose]. That is the cost curve our method flattens, published by the state of the art itself.

3. **The three-way oracle / MC / world-model comparison is most meaningful here.** `pipeline/conditions.py` already holds the method fixed across conditions 3-6 and varies only the evaluator. On static IM the MC arm pays a fixed price and the cost axis is flat. On adaptive IM it pays `r` times, so the axis finally has a slope, and the oracle arm remains the ceiling of model-based guidance (IC-only; LT has no oracle, per `research_notes/Baselines - Oracle vs MC vs GWM.md` §2).

### 9.2 The generator already emits this structure: verified

Checked against `data/generate_wm_data.py` (the `run_episode` timestep loop) and its argparse block:

| Claim                                                                | Code evidence                                                                                                                                                                                                                                            |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `t = 0` commits a seed set as a bag of `add_node` ops                | `seed_bag = [ActionOp("add_node", node) for node in seeds]`, used as `action` when `t == 0`                                                                                                                                                              |
| `t > 0` injects a further action with probability `--inject-p`       | `else sample_injection(s_t, …, p_inject=config.inject_p, …)`; `--inject-p` **default 0.3** [verified, argparse]                                                                                                                                          |
| counterfactual forks re-apply a different action from the same `s_t` | `if t > 0 and config.cf_prob > 0 and injection_rng.random() < config.cf_prob:` → `simulator.snapshot()` → `counterfactual_actions(...)` → `simulator.restore(snapshot)`; `--cf-prob` **default 0.2**, `--cf-branches` **default 2** [verified, argparse] |
| soft per-node targets estimated by repeated simulation               | `simulator.advance_marginal(action, config.mc_marginals)`; `--mc-marginals` **default 30** [verified, argparse]                                                                                                                                          |

So the transitions an adaptive-IM policy would need are **already in the training data**. The gap is one-directional and small: `sample_injection` chooses the injected action _at random_, and there is no budget bookkeeping tying `Σ_i |a_i| = k`. Both are evaluation-loop concerns. **No change to the model, the features, the heads, or the simulator is required.**

### 9.3 Build order

**Items 1-7 are implemented; the task is `implemented` in `pipeline/tasks.py` and runs end to end as `--task adaptive_online_im`.** Item 8 stays deferred, on this file's own recommendation (§2.4).

| #   | Item                                                                                                                             | Status | Where it landed                                                                                                                                                                                   |
| --- | -------------------------------------------------------------------------------------------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **MC-call + wall-clock counter per arm**                                                                                         | yes done | `evaluator_calls` / `evaluator_seconds` / `forward_passes` / `real_env_episodes` on every arm, captured before the `--credit` and `--referee` (with `--mc-agreement` for the NDlib replay) replays. §2.3's cost claim is now a column          |
| 2   | **Multi-round evaluation loop**: `--rounds r`, `--per-round-budget b`, policy re-queries `f_θ` each round on the realized `s_t` | yes done | `coding_agent/rounds.py` + the `adaptive` method. `--rounds` is Han's k-sweep, `--per-round-budget` his b-sweep, `--round-gap` sets the diffusion between rounds                                  |
| 3   | **Non-adaptive control arm at matched `k = r·b`**                                                                                | yes done | the task's registry `default_arms` pair `adaptive_free@E` with `evolve_free@E` for all four evaluators; `pipeline/conditions.py::adaptivity_gaps` divides them on the shared ground-truth replay  |
| 4   | **Report `(k, b)` alongside `--budget-pcts`**                                                                                    | yes done | `rounds`, `round_batches`, `round_gap`, `feedback_model`, `round_spreads` in the results JSON, `summary.csv`, and the report's adaptivity table                                                   |
| 5   | **Feedback-model switch**: restrict the policy to `frontier` (myopic) or allow `infected` (full-adoption)                       | yes done | `--feedback-model`; myopic blanks `state.infected` before `act()` sees it. See the asymmetry note below                                                                                           |
| 6   | `nx.powerlaw_cluster_graph` in `data/wm_graphs.py`                                                                               | yes done | `--dataset powerlaw_cluster`, with `--plc-m` / `--plc-p`. `kronecker` (§6.3b) shipped alongside it with ConTinEst's three seed matrices                                                           |
| 7   | **w-hidden ablation** (mask `edge_weight`, feed ones)                                                                            | yes done | `--hide-edge-weights`, masking at `build_graph_input` so training, one-step eval, rollout, planning and the agent's WM env all see ones. Refuses `structured_residual` / `structured_oracle`      |
| 8   | Continuous-time `T_endo`                                                                                                         | deferred | new simulator, new state, new head (§2.4c); unchanged recommendation                                                                                                                            |

**Baselines (§3.1, §3.2, §4.2) are implemented on both sides of the ladder.**

*Condition 1, our own library*: `coding_agent/tools/adaptive_algorithms.py`, seven per-round policies with signature `(state, graph, batch, dynamics) -> seeds`. This is a separate namespace from `algorithms.py` because every member there returns a static seed set from the graph alone; without it, a `--baselines` arm on this task deals all `k` at `t=0` and is the *control*, not a baseline.

| name | what it is |
| --- | --- |
| `adapt_greedy` | **AdaptGreedy** (Golovin & Krause 2011; Han et al. 2018): greedy marginal gain re-estimated each round against the realized state. The reference algorithm |
| `adapt_epic` | **EPIC** (Han et al. 2018): AdaptGreedy instantiated with RIS per batch. RR sets already covered by the active set are dropped before selection |
| `adapt_degree_discount` | DegreeDiscount restricted to susceptibles and discounted by the realized state |
| `adapt_degree`, `adapt_pagerank`, `adapt_random` | the per-round heuristic floor |
| `static_split` | §9.4's "non-adaptive wearing a costume": one static ranking dealt a batch per round. The control that isolates the timing penalty from the adaptivity benefit |

They are the task's registry `default_baselines`, `--baselines adapt_greedy` runs one as a condition-1 arm, and `build_adaptive_reference` puts them in the agent's prompt so a generated policy is asked to beat AdaptGreedy rather than reinvent it. Warning: `algorithms.adaptive_greedy` already existed and is adaptive *sampling* (coarse then refined MC), not adaptive *seeding*; Han et al.'s own name AdaptGreedy is used here, so the two never collide.

**A first measurement, on `powerlaw_cluster` at N=80, k=8, r=3, IC, MC-20 referee.** Every gap came out at or below 1 (`adapt_epic` 0.993, `adapt_degree_discount` and `static_split` 0.977, `adapt_greedy` 0.934, `adapt_random` 0.565), which is §5.1's prediction reproduced rather than a null result. The cost column is where the separation is: **`adapt_greedy` spent 70.5 evaluator-seconds against `adapt_epic`'s 0.3**, on a graph of 80 nodes. That ratio is exactly why Han et al. wrote EPIC, and it is the axis §2.3 argues a learned transition function flattens.

*Condition 7, published repos*: six registered under `task="adaptive_online_im"`. All were cloned and read on 2026-08-03, which changed the picture materially.

| repo | state |
| --- | --- |
| `adaptiveim` (Han et al.) | **WIRED, built and run.** Condition-7 arm, replays per round |
| `rl4im` (Chen et al.) | **WIRED**, adapter written from source; install blocked on this machine only (below). On the cluster it then died at `import networkx`: its requirements pin `networkx==2.3` beside a bare `numpy`, which resolves to 1.24 on 3.8 and drops the `np.int` alias that version's graphml writer reads at import. Patched to `numpy<1.24` in the registry |
| `mrim` (Sun et al.) | **BLOCKED: the code is not in the repository.** The README says to email the author. What ships is Wei Chen's older single-round `maxinf` toolkit plus Windows `.exe` files; no file under `code/` mentions multi-round. Wiring it would put static IM code in the table under MRIM's name |
| `oim`, `oim_lt`, `timlinucb` | interfaces recorded, not wired: all three are **repeated-campaign** methods whose published metric is union-over-trials or cumulative regret, so there is no single seed set to return. `oim_lt`'s `Main.py` is a simulation harness with no selector entry point at all |

**Wiring `adaptiveim` needed a contract extension, and the extension is the interesting part.** The external contract returned `list[int]`, replayed as one `t=0` plan. For a batched selector that destroys the adaptivity and lands AdaptGreedy on the non-adaptive side of the gap table. `ExternalBaseline.rounds_aware` now marks a repo that returns seeds in SELECTION order; `run_baseline` writes the round schedule into the work dir as JSON (a file rather than a signature change, so the seven wired IM adapters are untouched); and `round_seed_script` slices the flat list back into batches so `act()` hands back batch `i` at round `i`.

**Read the arm precisely.** Its batches were chosen against ITS realizations, not against the state our simulator goes on to produce, so this is AdaptGreedy's **schedule** replayed under our referee, not AdaptGreedy adapting inside our environment. True cross-process adaptivity would need it to accept an already-active set per round, which its CLI does not expose. A visible consequence: on a measured run at `k=12, r=3` it spent 10 of 12 seeds, because our realization had already activated two of its round-3 picks. That under-spend is real and shows up in `round_spreads`.

**Two obstacles resolved rather than worked around.** The README defers the "realization" files to an unlinked Tools repo; they are IC live-edge samples and the format is read verbatim from `load_possible_world()`, so the adapter generates them from OUR probabilities. And the repo does not compile on arm64 at all (`head.h`'s `rdtsc()` is x86 inline asm); since it feeds only a diagnostic macro, a `steady_clock` counter is functionally identical and portable, applied through the existing `patches` mechanism.

**First measurement, `powerlaw_cluster` N=120, k=12, r=3, MC-20 referee.** The authors' binary scored **62.70**; our own `adapt_epic` reimplementation scored **62.55**. Landing within 0.15 of the original is the strongest evidence yet that the condition-1 policies are faithful.

**`rl4im` install, verified as far as this machine allows.** Two patches were needed and both are recorded: `ipdb==0.12` builds with `use_2to3`, removed in setuptools 58, so it is unpinned rather than dropped (it is imported at module scope in `src/tasks/task_rl4im.py`); and the venv is pinned to python 3.9 for the 2020 torch/PyG pins. The remaining failure is `torch==1.7.0 has no wheels with a matching Python ABI tag`, because PyTorch published no arm64 macOS wheels before 1.12. On linux x86_64 torch 1.7.0 ships cp36-cp39 wheels, so this is expected to install on the cluster. Two honest limits on the arm regardless: it generates its own graphs (no flag for "run on this graph"), so the adapter matches ours on size and family and lets it train there, which is zero-shot transfer from a distribution; and it models willingness (`q=0.6`), so `parse_seeds` reads `invited` rather than `present`, since scoring the survivors would credit it with a budget it never spent.

Warning: **A registry error this surfaced and fixed.** The existing `oim` entry claimed *"No public code release found"* and sat under `influence_maximization`. Both were wrong: §3.2 records the co-author's release at `github.com/smaniu/oim`, and the method is online IM. It is now `task="adaptive_online_im"` with the honest blocker (its union-over-trials protocol is not comparable to single-campaign spread, §8.2 trap 2).

**Branches (d) and (e) are built too, and neither was in §9.3.** The build order only ever scheduled branch (a) plus item 7, which is why a build-order-driven implementation left two of this file's own five branches untouched. Both are now in, at the cost §2.4 predicted.

*(d) Dynamic / streaming*: `coding_agent/stream.py`, `--edit-rate`. A deterministic schedule of exogenous edge insertions and deletions, applied through the action bag, which is why no environment changed: `Simulator.advance` and `apply_edge_ops` already do the right thing with an edge op. Three properties that are load-bearing rather than decorative:

- **Pure function of (base graph, seed, timestep).** `MonteCarloEnvironment` loops (episode, timestep) while `WorldModelEnvironment` loops (timestep, sample), so a stream carrying mutable state would hand the two different graph histories. A deterministic schedule delivers the same edit at `t` to every episode and every sample, which is what "exogenous" means.
- **Applied to EVERY arm.** Comparing an arm whose graph moved against one whose graph did not measures the stream, not the method. The edits are appended *after* the policy's bag is validated, since they are not the policy's to be charged for and `--allowed-ops add_node` would otherwise reject them.
- **Balanced.** Deletions and insertions are matched on undirected *edges*, not arcs, and insertions retry on collision. The first version deleted one arc per two inserted and grew the graph 23% over six steps, which would have shown up as an adaptivity win rather than as densification.

*(e) Multi-round*: `coding_agent/envs/multi_round_env.py`, `--campaigns r`. §2.4e's "reset `frontier` between campaigns, keep `infected` as the union" is the right idea and slightly optimistic about where it goes. Clearing the frontier mid-episode is not a fresh diffusion under either dynamics: under IC it means rewriting every status-1 node to status 2, and under LT there is no spent state at all, so an active node keeps feeding its neighbours' thresholds and the campaigns never separate. A campaign is therefore a genuinely fresh inner rollout, which separates cleanly under both dynamics (LT even re-draws its hidden thresholds, which is correct: they are per-episode) and needs no environment change. The union uses `P(v never activated) = ∏_i (1 − p_i(v))` over the per-campaign marginals, which is **exact** for a fixed schedule rather than an approximation.

*`σ(S, T)`*: §8.3's last Warning: is closed. The reward was already `σ(S, horizon)` under monotone dynamics (the rollout only breaks at a fixed point), but `infected_counts` stopped when the representative cascade died, so `σ(S, T)` was unreadable at any smaller `T`. Every result now carries `spread_curve`: the ensemble-mean count at each `T`, padded to `horizon + 2` by holding the last value, which is exact for the same reason the break is safe.

**Two bugs these branches surfaced, both found by running them rather than by reading:**

1. `static_split` requested a static ranking of exactly `k`, so under multi-round campaign 2 started with campaign 1's `k` activations already in `active`, found no fresh candidates, and seeded nothing. Measured as `campaign_rewards [33.0, 0.0, 0.0]`. The ranking now accounts for the active set.
2. The `--referee` (with `--mc-agreement` for the NDlib replay) ground-truth referee was not wrapped for multi-round, so an arm reported a 3-campaign union while the shared referee reported a 1-campaign spread and the report put the two in one column. The referee now runs the same number of campaigns.

**The one design decision this file did not anticipate.** Under myopic feedback the policy cannot see `state.infected`, so it cannot tell that a candidate is already active. Erroring on a re-seed would make the myopic arm unrunnable; letting it through would be worse, because `add_node` writes NDlib status `1` over status `2` and hands a spent IC spreader a second round of transmission. The seed is therefore **dropped**: the slot is spent, nothing activates, and blind re-seeding costs the policy budget. That is the price of the weaker observation, and it is the mechanism by which the feedback model does any work at all. Under full-adoption the same proposal raises, because the policy was handed the set it failed to filter.

### 9.4 Two honest calibrations

- **Do not promise a spread win.** Theory caps the myopic adaptivity gap at `4` and proves non-adaptive greedy is _no worse_ than adaptive greedy across all graphs [verified, §5.1]. If our adaptive arm beats the static arm by 2%, that is consistent with the literature, not a weak result. **Claim cost, not spread.**
- **Do not overclaim novelty against RL-for-IM.** DISCO / PIANO / GCOMB / ToupleGDD are RL over _seed-set construction_, not over diffusion (§4.1, verified for ToupleGDD from its own state definition). The genuine prior work is RL4IM, at `|V| = 200`, `T = 2`, `B = 4` [verified, §5.5]. Our claim should be **model-based vs model-free**, which no adaptive-IM paper found in this review does at all.

---

## 10. Reference list

All links verified HTTP 200 on **2026-07-28** unless marked. `[DOI-verified]` means the publisher page returns 403/202 to `curl` (bot protection) but the DOI resolved to the correct title, venue, and year through the Crossref API.

**Adaptive IM** [Golovin & Krause, Adaptive Submodularity, JAIR 2011 (arXiv 1003.3967)](https://arxiv.org/abs/1003.3967) · [JAIR](https://www.jair.org/index.php/jair/article/view/10714) · [Seeman & Singer, Adaptive Seeding in Social Networks, FOCS 2013 (DOI 10.1109/FOCS.2013.56)](https://doi.org/10.1109/FOCS.2013.56) `[DOI-verified]` · [Badanidiyuru et al., Locally Adaptive Optimization, SODA 2016 (DOI 10.1137/1.9781611974331.ch31)](https://doi.org/10.1137/1.9781611974331.ch31) `[DOI-verified]` · [Tong et al., Adaptive IM in Dynamic Social Networks, IEEE/ACM ToN 2017 (arXiv 1506.06294)](https://arxiv.org/abs/1506.06294) · [DOI 10.1109/TNET.2016.2563397](https://doi.org/10.1109/TNET.2016.2563397) `[DOI-verified]` · [Vaswani & Lakshmanan, Why Commit when You can Adapt? (arXiv 1604.08171)](https://arxiv.org/abs/1604.08171) · [Yuan & Tang, No Time to Observe (partial feedback), IJCAI 2017 (arXiv 1609.00427)](https://arxiv.org/abs/1609.00427) · [**Han et al., Efficient Algorithms for Adaptive IM, PVLDB 11, 2018**](http://www.vldb.org/pvldb/vol11/p1029-han.pdf) · [code: kkhuang81/AdaptiveIM](https://github.com/kkhuang81/AdaptiveIM) · [Sun et al., Multi-Round IM, KDD 2018 (arXiv 1802.04189)](https://arxiv.org/abs/1802.04189) · [DOI 10.1145/3219819.3220101](https://doi.org/10.1145/3219819.3220101) `[DOI-verified]` · [code](https://github.com/lichao-sun/Multi-Round-Influence-Maximization) · [**Peng & Chen, Adaptive IM with Myopic Feedback, NeurIPS 2019 (arXiv 1905.11663)**](https://arxiv.org/abs/1905.11663) · [Chen & Peng, Adaptivity Gaps under Full-Adoption Feedback, ISAAC 2019 (arXiv 1907.01707)](https://arxiv.org/abs/1907.01707) · [D'Angelo, Poddar & Vinci, Better Bounds on the Adaptivity Gap, AAAI 2021 (arXiv 2006.15374)](https://arxiv.org/abs/2006.15374)

**Online / bandit IM** [Chen, Wang & Yuan, CUCB, ICML 2013 (PMLR v28)](https://proceedings.mlr.press/v28/chen13a.html) · [Chen et al., CMAB with Probabilistically Triggered Arms, JMLR 2016 (arXiv 1407.8339)](https://arxiv.org/abs/1407.8339) · [**Lei, Maniu, Mo, Cheng & Senellart, Online Influence Maximization, KDD 2015**, extended version (arXiv 1506.01188)](https://arxiv.org/abs/1506.01188) · [DOI 10.1145/2783258.2783271](https://doi.org/10.1145/2783258.2783271) `[DOI-verified]` · [code: smaniu/oim](https://github.com/smaniu/oim) · [Vaswani, Lakshmanan & Schmidt, Influence Maximization with Bandits (arXiv 1503.00024)](https://arxiv.org/abs/1503.00024) · [**Wen, Kveton, Valko & Vaswani, IMLinUCB, NeurIPS 2017 (arXiv 1605.06593)**](https://arxiv.org/abs/1605.06593) · third-party port [olety/TIMLinUCB](https://github.com/olety/TIMLinUCB) · [Vaswani et al., Model-Independent Online Learning for IM, ICML 2017 (arXiv 1703.00557)](https://arxiv.org/abs/1703.00557) · [Wang & Chen, Improving Regret Bounds for CMAB with Triggered Arms, NeurIPS 2017 (arXiv 1703.01610)](https://arxiv.org/abs/1703.01610) · [Wu et al., Factorization Bandits for Online IM, KDD 2019 (arXiv 1906.03737)](https://arxiv.org/abs/1906.03737) · [Online IM under the Linear Threshold Model, NeurIPS 2020 (arXiv 2011.06378)](https://arxiv.org/abs/2011.06378) · [code](https://github.com/Ritchiegit/Online_Influence_Maximization_under_Linear_Threshold_Model)

**Continuous-time** [Gomez-Rodriguez et al., NETRATE, ICML 2011 (arXiv 1105.0697)](https://arxiv.org/abs/1105.0697) · [Gomez-Rodriguez & Schölkopf, INFLUMAX, ICML 2012 (arXiv 1205.1682)](https://arxiv.org/abs/1205.1682) · [**Du, Song, Gomez-Rodriguez & Zha, ConTinEst, NeurIPS 2013**](https://proceedings.neurips.cc/paper/2013/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html) · [arXiv 1311.3669](https://arxiv.org/abs/1311.3669) · [Du et al., InfluLearner, ICML 2014 (PMLR v32)](https://proceedings.mlr.press/v32/du14.pdf) · [Influence Estimation and Maximization in Continuous-Time Diffusion Networks, ACM TOIS 2016 (DOI 10.1145/2824253)](https://doi.org/10.1145/2824253) `[DOI-verified]` · [Chen, Lu & Zhang, Time-Critical IM, AAAI 2012 (arXiv 1204.3074)](https://arxiv.org/abs/1204.3074) · [Neural Mean-Field Dynamics for Influence Estimation and Maximization (arXiv 2106.02608)](https://arxiv.org/abs/2106.02608)

**Dynamic / streaming** [Ohsaka et al., Dynamic Influence Analysis in Evolving Networks, PVLDB 9, 2016](http://www.vldb.org/pvldb/vol9/p1077-ohsaka.pdf) · [DOI 10.14778/2994509.2994525](https://doi.org/10.14778/2994509.2994525) `[DOI-verified]` · [Yang et al., Tracking Influential Nodes in Dynamic Networks, TKDE 2017 (arXiv 1602.04490)](https://arxiv.org/abs/1602.04490) · [Wang, Fan, Li & Tan, Real-Time IM on Dynamic Social Streams, PVLDB 10, 2017 (arXiv 1702.01586)](https://arxiv.org/abs/1702.01586) · [Peng, Dynamic Influence Maximization, NeurIPS 2021 (arXiv 2110.13355)](https://arxiv.org/abs/2110.13355)

**RL for adaptive IM** [Kamarthi et al., Learning Policies for Effective Graph Sampling, AAMAS 2020 (arXiv 1907.11625)](https://arxiv.org/abs/1907.11625) · [**Chen et al., Contingency-Aware IM: A Reinforcement Learning Approach, UAI 2021 (arXiv 2106.07039)**](https://arxiv.org/abs/2106.07039) · [code: wmd3i/RL4IM-Contingency](https://github.com/wmd3i/RL4IM-Contingency)

**Data sources** [SNAP MemeTracker](https://snap.stanford.edu/data/memetracker9.html) · [SNAP InfoPath (cascade-formatted MemeTracker)](https://snap.stanford.edu/infopath/data.html) · [SNAP ego-Facebook](https://snap.stanford.edu/data/ego-Facebook.html) · [SNAP soc-sign-epinions](https://snap.stanford.edu/data/soc-sign-epinions.html) · [SNAP com-Orkut](https://snap.stanford.edu/data/com-Orkut.html) · [SNAP soc-LiveJournal1](https://snap.stanford.edu/data/soc-LiveJournal1.html)

---

## 11. Open gaps

Honest list of what this review could **not** establish.

### Settled: recorded here so it is not re-litigated

- **DeepIM's `OIM` baseline is Lei, Maniu, Mo, Cheng & Senellart, "Online Influence Maximization", KDD 2015.** Determined by text extraction from DeepIM's own PDF: the baseline paragraph reads _"Online IM: OIM (Lei et al., 2015)"_ and the reference list contains exactly one matching entry [verified, DeepIM PMLR PDF §5.1 + references]. Its numbers are already transcribed in [`influence_maximization.md`](influence_maximization.md) §5.1 and are deliberately **not** duplicated here.

### Found while implementing (2026-08-01)

- **The first cluster sweep's adaptive arm was scoring fifty different policies (found 2026-08-23).** `WorldModelEnvironment` advances its ensemble in lockstep and calls `act()` once per member per round, interleaved; the generated policy kept `self._chosen` across calls, so member 1's second batch excluded member 0's picks and by member 49 every possible world had committed a different seed set. Netscience at `k=318`: 506 in the loop, 724 on the ground-truth referee, and a gap column of 0.66-0.78 that was this artefact rather than adaptivity. `State.sample` now names the member and `rounds.adaptive_action_fn` hands each one its own copy of the policy; `check_adaptive.a_stateful_policy_gets_one_copy_per_sample` reproduces the old divergence. The sweep's adaptive rows are void and the gap table has not yet been measured under the fix.

- **Epinions edge count reconciles exactly, and the 573-arc gap is self-loops.** Our loader builds **131,828 nodes / 840,799 arcs** against the `132K / 841K` Han et al. quote [verified]. The file has 841,372 lines, of which **573 are self-loops** and **zero are duplicate ordered pairs**; dropping self-loops (which every loader here does, since `p(u→v) = 1/in-degree(v)` is undefined when a node is its own in-neighbour) accounts for the difference to the arc. The sign column is discarded: Han et al.'s `132K / 841K` is the full signed arc count read as a plain directed graph, so keeping only trust arcs would give a different graph from the one their table describes.

- **DBLP (classical) has no located source.** §6.2 cites it as "Wei Chen / arnetminer release" with no URL, the only row in that table without one. Wei Chen's `weic-graphdata.zip`, which `data/datasets/netphy.py` already downloads, contains exactly `hep.txt`, `phy.txt` and a README: **no DBLP file**. SNAP's `com-DBLP` is a different graph (317,080 / 1,049,866, against the 655K / 1.99M both Han et al. and Lei et al. report), so substituting it would be a silent version collision of exactly the kind §6 exists to prevent. No loader was written. It stays the one dataset in this file's §6.2 we cannot reach.
- ~~**RL4IM's `m` for powerlaw-cluster is not published.**~~ **RESOLVED 2026-08-03 by reading the repo.** `RL4IM-Contingency/src/tasks/config/basic_env.yaml` sets `m: 3, p: 0.05`, and `src/environment/graph.py` passes them straight to `nx.powerlaw_cluster_graph(n=cur_n, m=m, p=p)`. So the protocol is **m=3**, which gives average degree **5.91** at n=200, not the "average degree 3" the paper states [verified, §5.1]. The paper disagrees with its own code; the code wins, and `--plc-m` now defaults to 3. Two further parameters came out of the same file and are worth having: `node_train: 200` with `graph_node_var: 20` (so graph size varies by ±20, not fixed at 200), and `graph_nbr_train: 200` / `graph_nbr_test: 5`.
- **The IMLinUCB Facebook subgraph was not attempted**, for the reason §6.2 already gives: `L = 327`, `|E| = 5,038` are published but the extraction procedure from SNAP ego-Facebook is not, so any 327-node subgraph we built would not be theirs.
- **Cumulative regret is still not implemented**, and this is a scope statement rather than an oversight. §9.3's build order never scheduled it; §2.4b names "a per-round exploration policy and a regret accumulator" as the cost of the online/bandit branch, and the accumulator is the easy half. The hard half is that regret needs a repeated-campaign loop in which the policy *learns between campaigns*: a bandit algorithm, i.e. a new method, not a metric. Item 7 (w-hidden) delivers that branch's *information state*; the learner is future work. Shipping a regret number against an unstated `(α, γ)` would violate §8.2 trap 1, so none is shipped.

### Could not establish

- **No result _tables_ exist for the three most relevant papers.** Han et al. (PVLDB 2018), IMLinUCB (NeurIPS 2017), and RL4IM (UAI 2021) publish their empirical results **only as figures**. Everything numeric transcribed from them in §5 is a _protocol_ parameter or a _theoretical bound_, not a measured spread. Getting comparable spread numbers requires running [kkhuang81/AdaptiveIM](https://github.com/kkhuang81/AdaptiveIM) and [wmd3i/RL4IM-Contingency](https://github.com/wmd3i/RL4IM-Contingency).
- **MRIM (KDD 2018) dataset and result tables were not extracted.** The paper is linked and its code located, but neither its Table 1 nor its result cells were read; §7's MRIM column is blank for that reason, not because it reports no data.
- **Multi-round IM vs adaptive IM boundary.** Several papers use "multi-round" and "adaptive" interchangeably. §1.5 states the distinction we adopted (separate diffusions vs one diffusion observed in stages), but this review did **not** confirm that every paper cited under one heading obeys it.
- **The IMLinUCB Facebook subgraph is not reproducible.** `L = 327`, `|E| = 5,038` [verified], but the extraction procedure from SNAP ego-Facebook is unpublished (§6.2).
- **Seeman & Singer and Badanidiyuru et al. were verified by DOI only.** Their PDFs were not obtained, so nothing numeric from either appears above. Both are cited in §3.1 for their _contribution_, not for any number.
- **No public code found** for the majority of §3: CUCB, IMLinUCB (author's), Vaswani's model-independent line, the continuous-time family (NETRATE, INFLUMAX, ConTinEst, InfluLearner), and the entire dynamic/streaming line. GitHub API searches for each method name returned no author repository. **Absence of a search hit is not proof of absence**, several of these predate the convention of releasing code, and older releases may live on author homepages that were not exhaustively crawled.
- **Web search was unavailable for the last two-thirds of this review** (session budget exhausted). Discovery from that point on ran through the Crossref DOI API and the GitHub API. A later pass with search available may find author homepages, code releases, and 2022-2026 papers this review missed: in particular, **recent (2023-2026) work on adaptive IM and on LLM agents for graph combinatorial optimization is almost certainly under-covered here.**
- **No adaptive-IM survey was located.** One is likely to exist; this review did not find it.
- **Kronecker graph parameters are partially transcribed.** ConTinEst's core-periphery matrix is quoted in §6.3 from the paper's own text, but the exact seed matrix for the core-periphery case was read as `[0.9 0.5; 0.5 0.3]` from a line that also cites a source paper: treat it as `[claim]` rather than `[verified]` until re-checked against the PDF.
- **The `(α, γ)` values used by each bandit paper's oracle were not tabulated.** §8.2 flags that regret numbers are meaningless without them; this review did not collect them per-paper.
