# Cascading Failure in Infrastructure Networks — Prior Work, Datasets, and Published Results

Reference for the cascading-failure literature: power-grid blackout propagation and prediction, load-redistribution cascades, and interdependent-network collapse. Covers both lines the field split into — the **physics-based** line (DC/AC power flow, overload, trip, re-solve) and the **abstract load-capacity** line (Motter–Lai betweenness cascades) — plus the GNN/ML surrogates now being built on top of each.

This task is scored **⚠️ moderate fit** in [`README.md`](README.md): the action space is the best match of any candidate task we have screened, and the transition kernel `T_endo` is the worst. §2 is the whole verdict.

All URLs returned the HTTP code stated on **2026-07-28**.

---

## 0. Verification policy

| Tier           | Meaning                                                                                                              |
| -------------- | -------------------------------------------------------------------------------------------------------------------- |
| **[verified]** | Read from the paper's own table via text extraction, or from the data repository's own statistics page. Trustworthy. |
| **[derived]**  | Computed by us from a file we downloaded, or reconstructed arithmetically from published splits.                     |
| **[figure]**   | Read off a plotted figure — the paper published no table. Approximate, direction only.                               |
| **[claim]**    | Stated in prose by a paper or a secondary source; not cross-checked against a file or table.                         |

Every bus/branch/generator count in §6.2 is **[derived]**: we downloaded the MATPOWER `.m` case file and counted the rows of `mpc.bus`, `mpc.branch` and `mpc.gen`, and checked non-zero `rateA` (thermal limit) and `r`/`x` (impedance) columns. Where the PGLib paper's Table I publishes the same case, the two agree exactly, so the derivation method is validated.

**Edge-count convention.** Power-system datasets quote **branches**, which include both transmission lines and transformers, and are undirected in the graph sense but carry a signed directed flow. Our loaders report `adjacency.nnz` = 2× the undirected branch count. A "2,896 branch test case" in a cascading-failure paper is 2,896 undirected branches, i.e. `nnz = 5,792`.

**Buses vs. substations.** A _bus_ is an electrical node; a _substation_ physically contains one or more buses joined by breakers. Grid2Op's action space operates on substations (splitting a substation into two buses is its core action), so its "36 substations" and MATPOWER's "118 buses" are not the same count of the same thing.

---

## 1. Task definition

Given a network `G = (V, E)` carrying a flow, a **state** recording which components are in service, and an **initial contingency** that removes some components, predict the sequence of subsequent component failures and the final damage.

```
s_0  = initial contingency (a set of tripped branches / buses)
s_{t+1} = Trip( Redistribute( s_t ) )
outcome = (blackout size, demand not served, # failed branches, # stages)
```

The defining property, and the one that separates this task from everything else in this folder: **redistribution is global**. When a line trips, power does not flow to the neighbouring line — it re-solves across the entire connected component according to Kirchhoff's laws, and the next line to fail can be arbitrarily far away, topologically and geographically. Hines, Dobson & Rezaei state it directly: _"cascading outages in power systems propagate non-locally as well as locally… the next component to fail after a particular line outages may be very distant, both geographically and topologically"_ ([arXiv 1508.01775](https://arxiv.org/abs/1508.01775), 200) [verified].

### The three model families

| Family                     | What redistributes                                                                                | Failure rule                             | Electrical data needed                |
| -------------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------- | ------------------------------------- |
| **Physics-based (DC)**     | branch power flows from a linearised `B·θ = P` solve                                              | flow `g_e > c_e` (thermal rating) → trip | impedances, ratings, generation, load |
| **Physics-based (AC)**     | full AC power flow (`P`, `Q`, `V`, `θ`) + islanding, frequency and voltage control, load shedding | overload, under-frequency, under-voltage | all of the above plus reactive limits |
| **Abstract load-capacity** | node/edge **betweenness centrality**                                                              | `L_j(t) > C_j = (1+α)·L_j(0)` → remove   | **none — topology only**              |

The physics lines are what utilities and the IEEE PES Cascading Failure Working Group use. The abstract line is what the complex-networks community built, and it is the one our current `power_grid` graph can actually support (§2, §6.1).

### Interdependent networks

A fourth family couples two networks: a node in `A` needs a node in `B` to function, and vice versa. Buldyrev et al. (Nature 2010) showed that this mutually-connected-giant-component percolation produces a **first-order** collapse where each network alone would show a second-order one, and that broad degree distributions become a _liability_ rather than an asset. The canonical instance is the Italian power grid + its control communication network, prompted by the 28 September 2003 Italian blackout.

### Why the task exists

Blackout sizes have a **power-law tail** — large blackouts are far more likely than a Gaussian risk model predicts — which is the empirical finding that launched the OPA/self-organised-criticality line (Carreras, Newman, Dobson & Poole, HICSS 2000, from NERC DAWG data). Utilities are legally required to survive all N-1 contingencies; N-2 is `O(|E|²)` (4.19M for a 2,896-branch case) and N-k is intractable, so a fast surrogate for "what does this contingency cascade into" is the operational prize.

---

## 2. Fit with our methodology

The verdict is genuinely split, and the two halves point in opposite directions.

### 2.1 The action space is the best match of any task we have screened ✅

Every one of our five ops has an exact, standard, named counterpart in power systems. No other candidate task in this folder uses more than two of them natively.

| Our op                   | Power-system meaning                                            | Status in the literature                                                                                            |
| ------------------------ | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `remove_edge(u,v)`       | **line trip / branch outage** — the N-1 contingency             | _the_ canonical action; N-1 screening is a legal requirement                                                        |
| `remove_node(v)`         | **bus or substation outage**                                    | standard, though line outages are ~10× more likely (Hines & Dobson, [1508.01775](https://arxiv.org/abs/1508.01775)) |
| `set_edge_weight(u,v,w)` | **change a line's thermal rating `rateA` or its impedance `x`** | exactly the mitigation lever in Hines & Dobson §VI ("doubled their flow limits" on the 10 most critical lines)      |
| `add_edge(u,v,w)`        | **build a new transmission line**                               | transmission expansion planning                                                                                     |
| `add_node(v)`            | **energise a bus / restore a component**                        | restoration and black-start                                                                                         |

This is a rare thing: an action space we did not design for the task lining up one-for-one with the task's own vocabulary. Contrast IM, where four of the five ops sit idle.

Two op-level caveats:

- Cascades are **edge-state** processes, not node-state ones. Our `X` is `(N, 6)` with node channels; a faithful cascade model wants a per-branch binary state `s_e[t] ∈ {0,1}`. Chadaga et al. handle this by making **edges the carriers of hidden state** and using node features only to seed them (`h⁰_(u,v) = P̃_u − P̃_v`) — a line-graph formulation, not our node-graph one.
- `set_edge_weight` currently means _IC transmission probability_. Reusing it for a thermal rating changes its semantics from "probability" to "capacity"; our `structured` IC head reads it as a probability and would need replacing.

### 2.2 `T_endo` is a different animal ⚠️ — this is the real cost

IC and LT propagate **locally**: `v`'s next state depends on `v`'s neighbours. That locality is exactly what our `ICTransmissionHead` exploits — the per-edge `q(u→v)` gated by `frontier_u` is what makes the cascade self-terminating and non-saturating (`CLAUDE.md` §Heads). It is the single design decision that made rollout work.

Cascading failure has no such locality. One step is:

```
1. solve the power flow over the whole surviving island(s)   ← GLOBAL, O(N^1.5–N^3)
2. compare every branch flow to its rating
3. trip all overloaded branches simultaneously
4. re-island, rebalance generation/load, repeat
```

Step 1 is a linear (DC) or nonlinear (AC) solve. Redistribution is governed by **line outage distribution factors** — a dense `|E|×|E|` matrix in which the outage of branch `e` shifts flow onto branch `f` by an amount with no relationship to the graph distance between them. Jhun et al. quantify the complexity gap: a single cascade step is `O(N)` for a local contact process but `O(N²)` for the nonlocal load-redistribution process ([arXiv 2208.00133](https://arxiv.org/abs/2208.00133), 200) [verified].

**Concrete consequence for our stack.** A `k`-layer GCN/SAGE/GAT/GT encoder has a receptive field of `k` hops. With `--n-layers 3` on a 4,941-node grid of diameter ≈46, the encoder physically cannot see the node that is about to fail next. This is a _structural_ mismatch, not a capacity one — more hidden units will not fix it. Our IC/LT heads never had this problem because IC/LT themselves are 1-hop.

Two honest mitigations exist in the literature, and both are usable:

1. **Hines & Dobson's influence graph.** Cascades _do_ propagate locally — in a learned _influence graph_ `H` that is not the grid topology. Build `H` from many simulated cascades (`H[i,j]` = probability `j` fails in generation `m+1` given `i` failed in `m`), and the process becomes Markovian and 1-hop **on `H`**. Validated on the 2,896-branch Polish case: all 4,191,960 N-2 contingencies simulated with DCSIMSEP; 3,170 produced at least one further outage; the influence-graph cascade-size distribution is _"remarkably similar"_ to the simulator's [verified, §V-C]. **This is the single most important paper for us**: it says our locality assumption is recoverable, at the price of learning `H` (which is itself an `add_edge` problem over a second graph, and the same shape as `network_inference.md`).
2. **Global pooling in the head.** A cheaper fix: keep the local encoder but give the head a graph-level readout, or add virtual-node / low-rank global attention so flow information can travel in one layer.

### 2.3 What a suitable head would look like

Our `ICTransmissionHead` predicts a per-edge transmission probability and composes it into a node marginal. The cascade analogue is a **per-branch overload head** with the same "predict the mechanism, not the outcome" spirit:

```
per branch e:   ĝ_e = MLP([h_u, h_v, x_e, injections])      # predicted flow
                p_trip(e) = σ( τ · (|ĝ_e| / c_e − 1) )      # margin, not raw score
                s_e[t+1] = s_e[t] · (1 − p_trip(e))          # monotone: no un-trip
```

Three properties worth keeping, each mirroring something we already do:

- **Monotone by construction** (`s_e` only decreases), the direct analogue of our IC head's `y_inf = infected + (1−infected)·p_new`. It makes the cascade self-terminating for the same reason.
- **Margin-parameterised**, so the rating `c_e` enters as a _known_ denominator rather than something to be learned — the exact trick our `structured_residual` head uses with `logit(w)`. Zero correction = the DC oracle.
- **A conservation penalty.** The one physical invariant a GNN can be handed for free is `Σ_in = Σ_out` at every bus. Adding `‖B·θ̂ − P‖²` as an auxiliary loss is what the physics-informed power-flow GNN line (§4.2) does, and it is the cheapest way to inject the global constraint the encoder cannot see.

A `structured_oracle` analogue exists too and is worth building first, exactly as we did for IC: set `ĝ_e` to the true DC flow and check `count_bias ≈ 0`. If the structural form is right, the oracle head reproduces the simulator exactly, and any gap is the encoder's fault rather than the head's.

### 2.4 Motter–Lai as the entry point — **yes, recommended**

The abstract load-capacity model needs **no electrical data at all**:

```
L_j(0) = betweenness centrality of node j        # "load"
C_j    = (1 + α) · L_j(0)                        # capacity, α = tolerance
remove a node -> recompute all betweenness -> every j with L_j > C_j fails
-> repeat until no overloads.   Damage G = N'/N (relative size of the LCC).
```

[Motter & Lai, PRE 66 065102(R), 2002](https://arxiv.org/abs/cond-mat/0301086) (arXiv 200; [APS](https://link.aps.org/doi/10.1103/PhysRevE.66.065102) returns 403 to scripted clients) [verified — model equations read from the PDF].

Why this is the pragmatic entry:

- **It runs on the graph we already have.** `power_grid` (4,941 / 6,594) is exactly the class of network Motter & Lai and the whole complex-networks cascade line evaluate on. Zero new data.
- **It is genuinely step-wise**, with a clean `(s_t, a_t, s_{t+1})` decomposition — remove a node (`T_exo`), recompute betweenness and trip overloads (`T_endo`) — which is precisely the shape `data/generate_wm_data.py` harvests. Node state, not edge state, so `X` needs one extra channel (normalised load `L_j(t)/C_j`) rather than a re-architecture.
- **The `α` sweep is a free difficulty dial.** Small `α` → catastrophic global cascade; large `α` → nothing happens. There is a critical `α_c ≈ 0.15` at which the avalanche-size distribution is a power law with exponent `τ ≈ 2.1` [verified, arXiv 2208.00133 §II-A]. That gives us a principled "interesting regime" to train in, which BA-100 never had.
- **There is a published GNN baseline on exactly this model.** Jhun et al. ([2208.00133](https://arxiv.org/abs/2208.00133)) train a GIN to predict Motter–Lai _avalanche centrality_ on small graphs and transfer to larger ones and to the real Spain / France / UK grids. Their Table I is transcribed in §5.4 — a directly comparable number-for-number target.

The honest caveat, and it must be stated in any paper we write: Hines, Cotilla-Sanchez & Blumsack (_Chaos_ 20, 033122, 2010) argue that betweenness-load models **misrepresent** real electricity infrastructure vulnerability, and Hines & Dobson show real cascades follow an influence graph rather than shortest paths. So Motter–Lai results are a result about _load-redistribution cascades on a network_, **not** a result about power grids. That is a fine contribution as long as we do not overclaim it.

**Recommendation: yes.** Start with Motter–Lai on `power_grid` + our synthetic families. It is ~2 days of work, needs no new dependency, exercises `remove_node`/`remove_edge` for the first time, and buys a comparable published baseline. Treat the physics-based version as a separate, later project.

### 2.5 Our `power_grid` dataset cannot support a physics-based cascade ❌

`data/datasets/power_grid.py` builds an unweighted symmetric adjacency from `opsahl-powergrid.edges` and sets `node_feats = log1p(degree)`, `node_labels = zeros` [verified — read from the loader]. That is **all** it has:

| Needed for a DC cascade                    | Present in our `power_grid`? |
| ------------------------------------------ | ---------------------------- |
| branch reactance `x`                       | ❌                           |
| branch thermal rating `rateA`              | ❌                           |
| generator locations + `Pmax`               | ❌                           |
| bus load `Pd`                              | ❌                           |
| bus/branch identity (line vs. transformer) | ❌                           |

It is a 1998 Watts–Strogatz small-world _topology_, published to demonstrate clustering and path length — never intended to carry a power flow. **Doing the physics-based version properly means adopting MATPOWER/pandapower cases.**

Which cases are small enough for our pipeline (branch counts and full N-2 enumeration sizes both **[derived]**):

| Case                   | Buses | Branches | Gens  | N-2 contingencies | Verdict for us                                  |
| ---------------------- | ----- | -------- | ----- | ----------------- | ----------------------------------------------- |
| `case39` (New England) | 39    | 46       | 10    | 1,035             | trivially small; good smoke test                |
| `case_ACTIVSg200`      | 200   | 245      | 49    | 29,890            | ✅ ideal first target                           |
| `case118` (IEEE 118)   | 118   | 186      | 54    | 17,205            | ✅ the field's default                          |
| `case300` (IEEE 300)   | 300   | 411      | 69    | 84,255            | ✅ comfortable                                  |
| `case_ACTIVSg500`      | 500   | 597      | 90    | 177,906           | ✅ comfortable                                  |
| `case_ACTIVSg2000`     | 2,000 | 3,206    | 544   | 5,137,615         | ⚠️ sample, don't enumerate                      |
| `case2383wp` (Polish)  | 2,383 | 2,896    | 327   | 4,191,960         | ⚠️ same; but it is _the_ published cascade case |
| `case9241pegase`       | 9,241 | 16,049   | 1,445 | 128.8M            | ❌ out of reach                                 |

`case2383wp` deserves a note: its 2,896 branches give exactly the 4,191,960 N-2 contingencies Hines & Dobson enumerate, which confirms their "2896 branch test case" _is_ the MATPOWER Polish winter-peak case [derived]. If we ever do the physics version, that is the case with a published reference distribution.

**Cost of the physics path, honestly:** a new simulator dependency (`pandapower`, or MATLAB + MATPOWER), a new `GraphBundle` carrying per-branch `(x, rateA)` and per-bus `(Pd, Pg)`, an edge-state `X`, a new head, and a new metric set. That is a multi-week project touching every layer of the stack — not a dataset swap.

### 2.6 Metrics — and how they map onto ours

| Cascading-failure metric                                     | Our nearest equivalent        | Note                                                                                       |
| ------------------------------------------------------------ | ----------------------------- | ------------------------------------------------------------------------------------------ |
| **Blackout size distribution** + its power-law tail exponent | — (new)                       | the SOC signature; needs many episodes, not many steps                                     |
| **Demand not served (DNS)**, MW or %                         | `ens_final_count_*` in spirit | PowerGraph's regression target                                                             |
| **Number of failed branches**                                | `ens_final_count_model/true`  | direct analogue of final infected count                                                    |
| **Cascade depth / # generations**                            | rollout horizon               | Chadaga predicts the per-branch failure _step_, which is `delta_f1` at a finer granularity |
| **N-1 / N-k screening accuracy**                             | `add_seed_success` analogue   | binary "does this contingency cause load shedding"                                         |
| **Time-to-predict vs. simulator**                            | already reported              | the field's headline claim (§5.2: 46–223× [derived])                                       |

Two of these are free for us — failed-branch count and time-to-predict map onto metrics `wm_eval.py` already computes. The blackout-size _distribution_ is the one genuinely new thing: it is a distributional match (a KS or `count_w1` over many episodes), not a per-step accuracy, and it is what the physics community actually validates against.

### 2.7 Verdict

| Dimension           | Grade                              | Why                                                                                      |
| ------------------- | ---------------------------------- | ---------------------------------------------------------------------------------------- |
| Action space        | ✅ **best of any task screened**   | all five ops have exact named counterparts                                               |
| State               | ⚠️                                 | edge-state, not node-state; needs a line-graph or per-edge head                          |
| `T_endo`            | ❌ **worst of any task screened**  | global load redistribution vs. our 1-hop locality assumption                             |
| Data we hold        | ❌ for physics · ✅ for Motter–Lai | `power_grid` is topology-only                                                            |
| Simulator           | ⚠️                                 | NDlib cannot do this; needs pandapower/MATPOWER, or ~80 lines of NetworkX for Motter–Lai |
| Published baselines | ✅ strong                          | PowerGraph (NeurIPS'24), Chadaga (2024), Jhun (2022), L2RPN — all with tables            |

**Overall: pursue Motter–Lai, defer the physics version.** The abstract model gives us a real cascade with global redistribution, a published GNN baseline, no new dependency, and honest framing. The physics version is a better paper and a much worse fit for the code we have.

---

## 3. Classical and physics-based methods

### 3.1 The OPA / self-organised-criticality line (ORNL – PSerc – Alaska)

The founding observation is empirical: blackout sizes in the NERC record follow a **power law**, not an exponential tail, so large blackouts are far likelier than a Gaussian risk model allows.

| Method                                              | Year  | Venue            | Idea                                                                                                                                                                            | Paper                                                                                                    | Code                 |
| --------------------------------------------------- | ----- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | -------------------- |
| **Initial evidence for SOC**                        | 2000  | HICSS            | Power-law tail in 15 years of NERC DAWG blackout data; the paper that started the line                                                                                          | [Carreras, Newman, Dobson & Poole](http://iandobson.ece.iastate.edu/PAPERS/carrerasHICSS00.pdf)          | —                    |
| **OPA model**                                       | 2001  | HICSS            | Two timescales: _fast_ = DC-flow overload/trip cascade within a day; _slow_ = load growth + line upgrades over years. Self-organises to criticality                             | [Dobson, Carreras, Lynch & Newman](http://iandobson.ece.iastate.edu/PAPERS/dobsonHICSS01.pdf)            | no public code found |
| **Criticality of blackouts**                        | 2002  | HICSS            | Critical loading transition; blackout size vs. load as an order parameter                                                                                                       | [Dobson, Chen, Thorp, Carreras & Newman](http://iandobson.ece.iastate.edu/PAPERS/dobsonHICSS02.pdf)      | —                    |
| **Critical points & transitions**                   | 2002  | Chaos 12         | The OPA phase diagram for the transmission model                                                                                                                                | [Carreras, Lynch, Dobson & Newman](http://iandobson.ece.iastate.edu/PAPERS/carrerasChaos02.pdf)          | —                    |
| **Complex systems analysis of series of blackouts** | 2007  | Chaos 17, 026103 | The consolidated OPA review; cascading failure, critical points, self-organisation                                                                                              | [Dobson, Carreras, Lynch & Newman](http://iandobson.ece.iastate.edu/PAPERS/dobsonCHAOS07.pdf)            | —                    |
| **Branching process estimator**                     | 2010→ | IEEE TPWRS       | Model the cascade as Galton–Watson with propagation `λ`; estimate `λ` from utility outage data. Gives a blackout-size distribution from _observed_ cascades without a simulator | [Dobson publication list](http://iandobson.ece.iastate.edu/publications.html) (per-paper PDF paths vary) | —                    |
| **Markovian influence graph**                       | 2019  | IEEE TPWRS       | Builds the influence graph from _real utility line-outage data_ rather than simulations                                                                                         | [arXiv 1902.00686](https://arxiv.org/abs/1902.00686)                                                     | no public code found |
| **N-k outage motifs**                               | 2022  | —                | The most frequent N-k outages occur in a small number of motifs, which improves contingency selection                                                                           | [arXiv 2209.02192](https://arxiv.org/abs/2209.02192)                                                     | —                    |

The OPA model itself has **no public code release** that this review could find; Dobson's publication page hosts the papers, not the simulator.

### 3.2 Abstract load-capacity cascades (the complex-networks line)

| Method                                | Year | Venue             | Idea                                                                                                                                                            | Paper                                                                                                                                          | Code                                         |
| ------------------------------------- | ---- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| **Motter–Lai** ⭐                     | 2002 | PRE 66, 065102(R) | Load = betweenness; capacity `C_j = (1+α)L_j(0)`; remove one node, recompute, trip overloads, repeat. Damage = LCC fraction `G = N'/N`. **The reference model** | [arXiv cond-mat/0301086](https://arxiv.org/abs/cond-mat/0301086) · [APS](https://link.aps.org/doi/10.1103/PhysRevE.66.065102) (403 to scripts) | no public code found (≈80 lines of NetworkX) |
| **Motter, cascade control**           | 2004 | PRL 93, 098701    | Defence strategy: intentionally remove low-load nodes _early_ to shed load and halt the cascade                                                                 | [arXiv cond-mat/0401074](https://arxiv.org/abs/cond-mat/0401074)                                                                               | —                                            |
| **Crucitti–Latora–Marchiori**         | 2004 | PRE 69, 045104(R) | Same skeleton but _dynamic_: efficiency of each edge degrades continuously with overload, rather than a binary trip                                             | [arXiv cond-mat/0309141](https://arxiv.org/abs/cond-mat/0309141)                                                                               | —                                            |
| **Kinney, Crucitti, Albert & Latora** | 2005 | Eur. Phys. J. B   | Applies the CLM model to the **North American power grid** (14,099 buses); introduces generator/substation/distribution roles                                   | [arXiv cond-mat/0410318](https://arxiv.org/abs/cond-mat/0410318)                                                                               | —                                            |

The whole family shares one property that matters for us: **`T_endo` is a global recompute (all-pairs shortest paths), but the state is per-node and binary, and the update is a clean step function.** It is the closest thing in the literature to "IC/LT but with global redistribution".

### 3.3 Physics-based simulators and analyses

| Method                                            | Year  | Venue            | Idea                                                                                                                                                                                                                                  | Paper                                                                                                          | Code                                                                                                                               |
| ------------------------------------------------- | ----- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **dcsimsep**                                      | 2012→ | —                | Fast DC-flow cascade simulator with islanding + proportional load shedding. The workhorse of the Vermont group and of §5.3                                                                                                            | see COSMIC below                                                                                               | no public repo found under the authors' GitHub                                                                                     |
| **Random chemistry N-k**                          | 2012  | IEEE TPWRS 27(3) | Eppstein & Hines: a sub-sampling search that finds _collections_ of `k` contingencies that initiate cascades, orders of magnitude faster than enumeration                                                                             | IEEE Xplore only; `doi.org/10.1109/TPWRS.2012.2185961` returned **404** on check — search IEEE Xplore by title | no public code found                                                                                                               |
| **COSMIC**                                        | 2016  | IEEE TPWRS       | Hines et al.: hybrid **dynamic** cascade simulator — differential-algebraic machine models + discrete protection relays, so it captures rotor-angle and voltage collapse a quasi-steady-state model cannot                            | [arXiv 1411.3990](https://arxiv.org/abs/1411.3990)                                                             | no public repo found                                                                                                               |
| **Bernstein, Bienstock, Hay, Uzunoglu & Zussman** | 2014  | INFOCOM          | Geographically correlated failures (a storm / EMP over a disc); DC cascade + control implications. The Columbia line                                                                                                                  | [arXiv 1206.1099](https://arxiv.org/abs/1206.1099)                                                             | —                                                                                                                                  |
| **Influence graph** ⭐                            | 2016  | IEEE TPWRS       | Hines, Dobson & Rezaei: cascades propagate _locally in a learned influence graph_ that is not the grid. Markovian; validated on the 2,896-branch Polish case                                                                          | [arXiv 1508.01775](https://arxiv.org/abs/1508.01775)                                                           | no public code found                                                                                                               |
| **AC-CFM**                                        | 2020  | IEEE Systems J.  | Noebels, Preece & Panteli: full **AC** cascading-failure model with islanding, UFLS/UVLS, designed for resilience analysis and large contingency sets                                                                                 | [IEEE Systems J.](https://ieeexplore.ieee.org/document/9199218) (202)                                          | **[github.com/mnoebels/AC-CFM](https://github.com/mnoebels/AC-CFM)** (200) — MATLAB on MATPOWER                                    |
| **Cascades**                                      | —     | —                | Quasi-steady-state AC model used to generate the PowerGraph dataset; AC-OPF base point, islanding, frequency/voltage control, UFLS/UVLS, overload trip. Reproduces the historical WECC blackout distribution [claim, PowerGraph §A.3] | see PowerGraph §5.1                                                                                            | via [PowerGraph-Datasets](https://github.com/PowerGraph-Datasets)                                                                  |
| **Dynamic CF simulator**                          | 2021  | IEEE PowerTech   | Dai, Noebels, Panteli & Preece: DIgSILENT PowerFactory + Python API, full dynamic cascades                                                                                                                                            | —                                                                                                              | [github.com/YitianDai/Dynamic-cascading-failure-simulator](https://github.com/YitianDai/Dynamic-cascading-failure-simulator) (200) |
| **ps-res**                                        | 2024  | —                | Gerkis et al.: power-system resilience model (outage + restoration under extreme weather), builds on AC-CFM                                                                                                                           | —                                                                                                              | [github.com/AGerkis/ps-res](https://github.com/AGerkis/ps-res) (200)                                                               |

**The IEEE PES benchmarking papers** — Vaiman et al., "Risk assessment of cascading outages: methodologies and challenges" (IEEE TPWRS 27(2), 2012) and Bialek et al., "Benchmarking and validation of cascading failure analysis tools" (IEEE TPWRS 31(6), 2016) — are the field's own statement that different cascade simulators disagree, and the reason any result must name its simulator. Both are paywalled on IEEE Xplore; no free version was located [claim].

### 3.4 Interdependent networks

| Method                                                | Year | Venue            | Idea                                                                                                                                                 | Paper                                                                                                                  | Code                    |
| ----------------------------------------------------- | ---- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| **Buldyrev, Parshani, Paul, Stanley & Havlin** ⭐     | 2010 | Nature 464:1025  | Mutually-connected-giant-component percolation between two coupled networks. **First-order** collapse; broad degree distributions become a liability | [Nature](https://www.nature.com/articles/nature08932) (200) · [arXiv 0907.1182](https://arxiv.org/abs/0907.1182) (200) | no public code found    |
| **Parshani, Buldyrev & Havlin**                       | 2010 | PRL 105, 048701  | Partial interdependence: a coupling fraction `q` interpolates between first- and second-order transitions                                            | [arXiv 1004.3989](https://arxiv.org/abs/1004.3989)                                                                     | —                       |
| **Gao, Buldyrev, Stanley & Havlin**                   | 2012 | Nature Physics 8 | Networks formed from `n` interdependent networks; the tree/loop topology of the coupling matters                                                     | —                                                                                                                      | —                       |
| **Rosato et al.**                                     | 2008 | IJCIS            | The Italian power grid + SCADA communication network that motivated the 2003 blackout analysis                                                       | —                                                                                                                      | data not public [claim] |
| **Network-of-networks for electrical infrastructure** | 2015 | —                | Applies the NoN framing to real grids with electrical, not percolation, coupling                                                                     | [arXiv 1512.01436](https://arxiv.org/abs/1512.01436) (200)                                                             | —                       |

### 3.5 The critique — read this before quoting any topology-only result

Two lines of work argue that the abstract models above **do not describe power grids**:

- **Hines, Cotilla-Sanchez & Blumsack**, "Do topological models provide good information about electricity infrastructure vulnerability?", _Chaos_ 20, 033122 (2010). Compares characteristic path length and connectivity loss (topological) against blackout size from a cascade model over 40 areas of the Eastern US interconnect, and finds the topological metrics misleading. [AIP page](https://pubs.aip.org/aip/cha/article-abstract/20/3/033122/932307) returns 403 to scripted clients; no free version located.
- **Hines, Dobson & Rezaei** ([1508.01775](https://arxiv.org/abs/1508.01775)): _"simple models of topological contagion do not accurately represent the propagation of cascades in power systems"_ [verified].

Both are why §2.4 says to frame a Motter–Lai result as a statement about load-redistribution cascades, not about power grids.

**Surveys:** [Interaction graphs for cascading failure analysis (arXiv 1911.00475)](https://arxiv.org/abs/1911.00475) (200) · [ML applications in cascading failure analysis: a review (arXiv 2305.19390)](https://arxiv.org/abs/2305.19390) (200).

---

## 4. Learning-based methods

All arXiv URLs below returned **200**; titles and author lists were resolved through the arXiv API rather than a summarizer.

### 4.1 GNN cascading-failure prediction — the direct competition

| Method                                      | Year    | Venue              | Approach                                                                                                                                                                                                                                                                             | Paper                                                                                                                                                                                                 | Code                                                                                                                                                                                                                                                                     |
| ------------------------------------------- | ------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Jhun, Choi, Lee, Lee, Kim & Kahng** ⭐    | 2022    | Chaos              | GIN predicts **avalanche centrality** under the _Motter–Lai_ model; trained on small graphs, transferred to larger ones and to the Spain/France/UK grids. The one paper that is a direct baseline for §2.4                                                                           | [arXiv 2208.00133](https://arxiv.org/abs/2208.00133)                                                                                                                                                  | no public code found                                                                                                                                                                                                                                                     |
| **Chadaga, Wu & Modiano** ⭐                | 2023/24 | IEEE SmartGridComm | **Flow-free** GNN predicting the per-branch _failure time step_ `f_e` for the whole cascade, given initial contingency + power injections. Edge-hidden-state, edge-to-edge and node-to-edge attention. Beats load-specific influence models and is 46–223× faster than the DC oracle | [arXiv 2404.16134](https://arxiv.org/abs/2404.16134) · [IEEE](https://ieeexplore.ieee.org/document/10333943/)                                                                                         | no public code found                                                                                                                                                                                                                                                     |
| **PowerGraph** ⭐                           | 2024    | NeurIPS D&B        | The benchmark: IEEE24/39/118 + UK, physics-generated (Cascades) cascade labels, binary/multiclass/regression graph-level tasks, plus ground-truth explanation masks                                                                                                                  | [arXiv 2402.02827](https://arxiv.org/abs/2402.02827) · [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/file/c7caf017cbbca1f4b368ffdc7bb8f319-Paper-Datasets_and_Benchmarks_Track.pdf) | [PowerGraph-Datasets](https://github.com/PowerGraph-Datasets) — [Graph](https://github.com/PowerGraph-Datasets/PowerGraph-Graph), [Node](https://github.com/PowerGraph-Datasets/PowerGraph-Node), [XAI](https://github.com/PowerGraph-Datasets/PowerGraph-XAI) (all 200) |
| **Gorka, Hsu, Li, Maximov & Roald** ⭐      | 2024    | PSCC               | Blackout-_size_ regression on RTS-GMLC with a statistically-augmented topology (physical edges + learned co-failure edges) and a classify-then-regress cascade. 62.7M samples                                                                                                        | [arXiv 2403.15363](https://arxiv.org/abs/2403.15363)                                                                                                                                                  | no public code found (dataset release promised in the paper)                                                                                                                                                                                                             |
| **Dwivedi & Tajer, GRNN fault chains**      | 2023    | —                  | Graph RNN predicts the _fault chain_ (the ordered sequence of failures) in real time                                                                                                                                                                                                 | [arXiv 2303.08864](https://arxiv.org/abs/2303.08864)                                                                                                                                                  | no public code found                                                                                                                                                                                                                                                     |
| **Time-varying graph RNNs**                 | 2025    | —                  | Successor: risky fault-chain search with time-varying graph RNNs                                                                                                                                                                                                                     | [arXiv 2503.09775](https://arxiv.org/abs/2503.09775)                                                                                                                                                  | —                                                                                                                                                                                                                                                                        |
| **Hyperparametric diffusion model**         | 2024    | —                  | Xiang & Cautis: models cascading failures with a _diffusion_-style parametric propagation model — closest framing to ours in the whole ML sub-literature                                                                                                                             | [arXiv 2406.08522](https://arxiv.org/abs/2406.08522)                                                                                                                                                  | —                                                                                                                                                                                                                                                                        |
| **Physics-Informed Graph Neural Jump ODEs** | 2026    | —                  | Sevak, Jadhav & Bui: continuous-time jump ODE over the grid graph for cascade prediction                                                                                                                                                                                             | [arXiv 2603.20838](https://arxiv.org/abs/2603.20838)                                                                                                                                                  | —                                                                                                                                                                                                                                                                        |
| **GRU-gated graph attention**               | 2026    | —                  | Zhou & Li: _inductive_ cascading-failure analysis — transfers across grids, not just across contingencies                                                                                                                                                                            | [arXiv 2605.07010](https://arxiv.org/abs/2605.07010)                                                                                                                                                  | —                                                                                                                                                                                                                                                                        |
| **ML for cascading failure prediction**     | 2025    | —                  | Pani & Bera; classical-ML comparison baseline                                                                                                                                                                                                                                        | [arXiv 2503.00567](https://arxiv.org/abs/2503.00567)                                                                                                                                                  | —                                                                                                                                                                                                                                                                        |

**The shape of the whole column matters more than any single row.** Every one of these is either (a) a **one-shot** graph-level predictor — `contingency → blackout size`, no intermediate state — or (b) a per-branch _failure-step_ regressor that emits the whole trajectory in a single forward pass (Chadaga). **None of them is an action-conditioned one-step transition model rolled forward.** That is the gap §11 names.

### 4.2 GNN surrogates for power flow and OPF

These matter to us because a cascade step _is_ a power-flow solve: a learned `T_endo` for the physics version would most likely be built on one of these.

| Method                             | Year    | Venue           | Approach                                                                                                                                                                                                     | Paper                                                                                                                                                | Code |
| ---------------------------------- | ------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ---- |
| **Owerko, Gama & Ribeiro**         | 2019    | ICASSP'20       | The first GNN-for-OPF paper; imitation learning of the OPF solution                                                                                                                                          | [arXiv 1910.09658](https://arxiv.org/abs/1910.09658)                                                                                                 | —    |
| **— unsupervised follow-up**       | 2022    | —               | Same group, unsupervised (constraint-violation) objective                                                                                                                                                    | [arXiv 2210.09277](https://arxiv.org/abs/2210.09277)                                                                                                 | —    |
| **Donon et al., LEAP nets**        | 2019    | —               | RTE's latent encoding of atypical perturbations — learns the _effect of a topology change_ as a latent shift. Conceptually the closest published thing to `T_exo`                                            | [arXiv 1908.08314](https://arxiv.org/abs/1908.08314)                                                                                                 | —    |
| **Böttcher, Wolf et al.**          | 2022    | —               | AC power flow with GNNs under realistic constraints                                                                                                                                                          | [arXiv 2204.07000](https://arxiv.org/abs/2204.07000)                                                                                                 | —    |
| **PowerFlowNet**                   | 2023    | —               | Message-passing power-flow approximation                                                                                                                                                                     | [arXiv 2311.03415](https://arxiv.org/abs/2311.03415)                                                                                                 | —    |
| **CANOS**                          | 2024    | —               | DeepMind: fast, scalable neural AC-OPF **robust to N-1 perturbations**                                                                                                                                       | [arXiv 2403.17660](https://arxiv.org/abs/2403.17660)                                                                                                 | —    |
| **OPFData**                        | 2024    | —               | DeepMind: large-scale AC-OPF datasets **with topological perturbations** — i.e. the `remove_edge` action already applied at scale                                                                            | [arXiv 2406.07234](https://arxiv.org/abs/2406.07234)                                                                                                 | —    |
| **DeepOPF family**                 | 2019–22 | IEEE TPWRS etc. | DNN OPF solvers; the non-graph baseline the GNN line is measured against                                                                                                                                     | [arXiv 1905.04479](https://arxiv.org/abs/1905.04479), [2007.01002](https://arxiv.org/abs/2007.01002), [2103.11793](https://arxiv.org/abs/2103.11793) | —    |
| **Fioretto, Mak & Van Hentenryck** | 2019    | AAAI'20         | Lagrangian-dual training to enforce AC-OPF constraints                                                                                                                                                       | [arXiv 1909.10461](https://arxiv.org/abs/1909.10461)                                                                                                 | —    |
| **Physics-informed GNNs**          | 2024–26 | —               | PINCO ([2410.04818](https://arxiv.org/abs/2410.04818)), KCLNet (Kirchhoff-projection, [2506.12902](https://arxiv.org/abs/2506.12902)), edge-aware attention ([2509.22458](https://arxiv.org/abs/2509.22458)) | —                                                                                                                                                    | —    |
| **LUMINA**                         | 2026    | —               | A _foundation model_ benchmark for AC-OPF surrogate learning                                                                                                                                                 | [arXiv 2605.02133](https://arxiv.org/abs/2605.02133)                                                                                                 | —    |

### 4.3 Contingency screening with ML

| Method                                               | Year    | Approach                                                                                                                                                            | Paper                                                                                          |
| ---------------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **Nakiganda & Chatzivasileiadis**                    | 2023    | GNNs for fast contingency analysis                                                                                                                                  | [arXiv 2310.04213](https://arxiv.org/abs/2310.04213)                                           |
| **Christianson, Cui & Low**                          | 2024    | **Input-convex** NNs give a _certifiable_ N-k screen: 10–20× speedup on IEEE 39-bus (pandapower) with controlled false-negative rate [verified from the paper text] | [arXiv 2410.00796](https://arxiv.org/abs/2410.00796)                                           |
| **Zhang & Karve**                                    | 2023/24 | GNN surrogates for operational **risk** assessment, incl. evolving topology                                                                                         | [2311.12309](https://arxiv.org/abs/2311.12309), [2405.07343](https://arxiv.org/abs/2405.07343) |
| **Diffusion-based dynamic screening**                | 2025    | Generative screening of dynamic contingencies                                                                                                                       | [arXiv 2510.04470](https://arxiv.org/abs/2510.04470)                                           |
| **State-aware high-impact N-k**                      | 2026    | Scalable/reliable inference of high-impact N-k sets                                                                                                                 | [arXiv 2602.09461](https://arxiv.org/abs/2602.09461)                                           |
| **Trustworthiness layer for grid foundation models** | 2026    | Alcántara & Chatzivasileiadis; conformal-style guarantees on N-k                                                                                                    | [arXiv 2602.07995](https://arxiv.org/abs/2602.07995)                                           |

### 4.4 Grid2Op / L2RPN — the closest existing "world model for grids"

**Grid2Op** is RTE France's Gym-style environment for sequential decision-making on power networks, and **L2RPN** ("Learning to Run a Power Network") is the competition series built on it. This is the single most reusable artefact in this file, and the closest analogue to what we are building.

| Component                            | URL                                                                          | HTTP |
| ------------------------------------ | ---------------------------------------------------------------------------- | ---- |
| Grid2Op (current org)                | [github.com/Grid2op/grid2op](https://github.com/Grid2op/grid2op)             | 200  |
| Grid2Op (legacy org, still resolves) | [github.com/rte-france/Grid2Op](https://github.com/rte-france/Grid2Op)       | 200  |
| Docs                                 | [grid2op.readthedocs.io](https://grid2op.readthedocs.io/)                    | 200  |
| `lightsim2grid` (fast C++ backend)   | [github.com/Grid2op/lightsim2grid](https://github.com/Grid2op/lightsim2grid) | 200  |
| Competition site                     | [l2rpn.chalearn.org](https://l2rpn.chalearn.org/)                            | 200  |
| Bundled environments directory       | [grid2op/data](https://github.com/Grid2op/grid2op/tree/master/grid2op/data)  | 200  |

**Bundled environments** [verified — read from the GitHub contents API]: `blank`, `educ_case14_redisp`, `educ_case14_storage`, `l2rpn_2019`, `l2rpn_case14_sandbox`, `l2rpn_case14_sandbox_diff_grid`, `l2rpn_icaps_2021`, `l2rpn_idf_2023`, `l2rpn_neurips_2020_track1`, `l2rpn_neurips_2020_track2`, `l2rpn_wcci_2020`, `l2rpn_wcci_2022_dev`, `rte_case118_example`, `rte_case14_opponent`, `rte_case14_realistic`, `rte_case14_redisp`, `rte_case14_test`, `rte_case5_example`. The `case5` / `case14` / `case118` naming tracks the MATPOWER cases of §6.2; per-environment substation and line counts were **not** extracted (the bundled `grid.json` resisted two parsing attempts) — treat any specific size as [claim].

**Why it is the closest thing to a world model.** Grid2Op's observation is the full grid state (per-line flows, injections, cooldowns, disconnection timers); its action space is _topological_ — split a substation into two buses, reconnect a line — plus continuous redispatch; and **its dynamics include a cascading failure**: an overloaded line auto-disconnects after a sustained overload, which can cascade to game over. Crucially it exposes `obs.simulate(action)`, a one-step lookahead through the real power-flow backend. That is precisely the interface our world model provides — except Grid2Op's is a _simulator_, not a learned model. The L2RPN retrospective states the NeurIPS 2020 setting exactly: IEEE 118 network, 5-minute resolution, 2,016 timesteps per week, **>70,000 discrete actions plus a 40-dimensional continuous space**, and 5/5 of the top teams used `simulate` at test time [verified, arXiv 2103.03104 §2.1, Table 2].

**Competition editions** [claim except where a paper is cited]: 2019 (IJCNN, `l2rpn_2019`) · WCCI 2020 (`l2rpn_wcci_2020`) · NeurIPS 2020, two tracks Robustness/Adaptability on IEEE 118 ([arXiv 2103.03104](https://arxiv.org/abs/2103.03104), 200) · ICAPS 2021 (`l2rpn_icaps_2021`) · WCCI 2022 (`l2rpn_wcci_2022_dev`) · Delft/IDF 2023 (`l2rpn_idf_2023`). Environment-design papers: [arXiv 2104.04080](https://arxiv.org/abs/2104.04080), [arXiv 2207.10330](https://arxiv.org/abs/2207.10330).

**Notable agents and follow-ups** (all arXiv 200):

| Work                                                                                                                                                | Year    | Idea                                                                                                      |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | --------------------------------------------------------------------------------------------------------- |
| [PowRL 2212.02397](https://arxiv.org/abs/2212.02397)                                                                                                | 2022    | Robust RL framework for L2RPN-style management                                                            |
| [AlphaZero topology control 2211.05612](https://arxiv.org/abs/2211.05612)                                                                           | 2022    | Dorfer et al.: MCTS + learned value over topology actions — **the only entry that plans against a model** |
| [Lehna et al. 2304.00765](https://arxiv.org/abs/2304.00765)                                                                                         | 2023    | RL vs. advanced rule-based agents, head to head                                                           |
| [HUGO 2405.00629](https://arxiv.org/abs/2405.00629)                                                                                                 | 2024    | DRL + heuristic action-space reduction                                                                    |
| [Multi-agent topology control 2310.02605](https://arxiv.org/abs/2310.02605), [2502.08681](https://arxiv.org/abs/2502.08681)                         | 2023–25 | Substation-level factorisation                                                                            |
| [Blackout mitigation via physics-guided RL 2401.09640](https://arxiv.org/abs/2401.09640)                                                            | 2024    | Dwivedi et al.                                                                                            |
| [RL for mitigating cascades 2411.18050](https://arxiv.org/abs/2411.18050)                                                                           | 2024    | Sensitivity-factor-guided exploration                                                                     |
| [Multi-stage cascade mitigation 2505.09012](https://arxiv.org/abs/2505.09012)                                                                       | 2025    | Deep RL across cascade stages                                                                             |
| [Cascade mitigation via influence graph + RL 2506.08893](https://arxiv.org/abs/2506.08893)                                                          | 2025    | Ties Hines–Dobson influence graphs to RL control                                                          |
| [RL2Grid 2503.23101](https://arxiv.org/abs/2503.23101)                                                                                              | 2025    | **A standardised RL benchmark over Grid2Op** — the cleanest published harness to plug an agent into       |
| [Graph RL for power grids: survey 2407.04522](https://arxiv.org/abs/2407.04522) · [topology-RL survey 2504.08210](https://arxiv.org/abs/2504.08210) | 2024–25 | Two surveys of the whole column                                                                           |

### 4.5 Has anyone published a learned cascading-failure _world model_?

**No — not in the sense we mean.** Evidence, and the searches behind it:

- arXiv full-text queries `abs:"world model" AND abs:"power grid"`, `all:"world model" AND all:"power grid"`, and `abs:"learned simulator" AND abs:"power grid"` return **zero** relevant hits (the only matches for "world model" + "power system" are an unrelated meter-reading paper and an unrelated agentic-tooling paper) [derived, arXiv API, 2026-07].
- Everything in §4.1 is a **one-shot** predictor: contingency in, final blackout size or full failure-step vector out. Chadaga et al. is explicitly _flow-free_ and emits the entire cascade in a single pass rather than stepping it.
- The RL column (§4.4) is overwhelmingly **model-free**; the exception, Dorfer et al.'s AlphaZero agent, plans against Grid2Op's _own simulator_, not against a learned transition model.
- Nobody trains `f_θ(G, s_t, a_t) → s_{t+1}` on cascade transitions and rolls it forward autoregressively, and nobody reports rollout-fidelity metrics (count bias, marginal MAE) against the simulator under matched actions.

That is a real, citable gap — see §9 and §11.

---

## 5. Published results

Every table below was transcribed with `pdftotext -layout` from the paper's own PDF. No summarizer supplied a cell.

### 5.1 PowerGraph (NeurIPS 2024 D&B) ⭐ the most directly comparable benchmark

Physics-generated cascade labels from the **Cascades** AC quasi-steady-state simulator, four test systems, three graph-level tasks. This is the closest thing the field has to DeepIM's role in the IM literature.

Dataset sizes [verified, Table 1]:

| Test system | Buses | Branches | Power-flow graphs (node-level) | Cascading-failure graphs (graph-level) |
| ----------- | ----- | -------- | ------------------------------ | -------------------------------------- |
| IEEE24      | 24    | 38       | 34,944                         | 21,500                                 |
| UK          | 29    | 99       | 34,944                         | 64,000                                 |
| IEEE39      | 39    | 46       | 34,944                         | 28,000                                 |
| IEEE118     | 118   | 186      | 34,944                         | 122,500                                |

Generation parameters [verified, Table 8]: **500 loading conditions** per grid × an outage list of 43 (IEEE24) / 56 (IEEE39) / 128 (IEEE118) / 245 (UK). Loading conditions come from a year of 15-minute data.

Class balance — the trap in this dataset [verified, Table 2 + Table 3]. Classes are A: `DNS>0` and a cascade occurs; B: `DNS>0`, no cascade; C: `DNS=0`, cascade occurs; D: `DNS=0`, no cascade.

| Power grid | Category A | Category B | Category C | Category D |
| ---------- | ---------- | ---------- | ---------- | ---------- |
| IEEE24     | 15.8%      | 4.3%       | 0.1%       | **79.7%**  |
| IEEE39     | 0.55%      | 8.4%       | 0.45%      | **90.6%**  |
| UK         | 3.5%       | 0          | 3.8%       | **92.7%**  |
| IEEE118    | >0.1%      | 5.0%       | 0.9%       | **93.9%**  |

Best model per task [verified, Table 5]:

| Power system | Task                 | Best model         | Metric        | Value               |
| ------------ | -------------------- | ------------------ | ------------- | ------------------- |
| IEEE24       | binary               | transformer 3h 32n | balanced acc. | **0.9828 ± 0.0056** |
| IEEE24       | multiclass           | transformer 3h 32n | balanced acc. | 0.9828 ± 0.0056     |
| IEEE24       | regression (DNS, MW) | gin 3h 32n         | MSE           | 7.82e-04 ± 1.62e-04 |
| IEEE39       | binary               | transformer 2h 32n | balanced acc. | **0.9880 ± 0.0020** |
| IEEE39       | multiclass           | transformer 3h 32n | balanced acc. | 0.9765 ± 0.0064     |
| IEEE39       | regression           | transformer 2h 16n | MSE           | 6.31e-05 ± 1.86e-05 |
| IEEE118      | binary               | gin 3h 32n         | balanced acc. | **0.9982 ± 0.0006** |
| IEEE118      | multiclass           | gin 3h 32n         | balanced acc. | 0.9962 ± 0.0014     |
| IEEE118      | regression           | gin 3h 32n         | MSE           | 2.99e-06 ± 3.19e-06 |
| UK           | binary               | gin 3h 32n         | balanced acc. | **0.9951 ± 0.0022** |
| UK           | multiclass           | gin 2h 32n         | balanced acc. | 0.9845 ± 0.0019     |
| UK           | regression           | transformer 3h 32n | MSE           | 1.07e-03 ± 3.47e-04 |

**Read this table the way we read DeepIM's IC column.** Balanced accuracy 0.98–0.998 on a task where 80–94% of instances are the majority class means the binary task is close to saturated — the same "solved at high budget" effect our IM file documents. The discriminative signal is in the **regression** row and in the explanation task, where the paper reports that _no_ explainability method recovers the cascading edges on IEEE39 or IEEE118, attributing it to Category A being <1% of instances [verified, §4].

### 5.2 Chadaga, Wu & Modiano (2024) ⭐ the one step-wise formulation

Setup [verified]: **200,000 cascade samples each** for IEEE89 and IEEE118; random N-2 initial contingency (two branches); load scaled uniformly by `α ~ U[1, 2]`; 90/10 train/test; topology, injections and branch capacities from **MATPOWER**, with _unavailable capacities set to twice the default flow_ — which is a direct admission that the IEEE base cases ship without thermal limits (§6.2 confirms this: `case118` has `rateA = 0` on all 186 branches [derived]). "IEEE89" is almost certainly MATPOWER `case89pegase` (89 buses, 210 branches); the paper does not say [claim].

Prediction time, seconds per 1,000 samples [verified, Table I]:

| System  | CFS oracle (DC, MATLAB CPU) | Influence model (GPU) | **GNN (GPU)** | GNN speedup vs. oracle |
| ------- | --------------------------- | --------------------- | ------------- | ---------------------- |
| IEEE89  | 24.18                       | 2.35                  | **0.53**      | **45.6×** [derived]    |
| IEEE118 | 62.54                       | 1.86                  | **0.28**      | **223.4×** [derived]   |

Accuracy [figure — the paper published no table]: failure-size error rate spans roughly **0–25%** across `α ∈ [1,2]` on IEEE89 and **2–3.5%** on IEEE118; final-state error roughly **0–6%** (IEEE89) and **1–3%** (IEEE118). The GNN is below every load-specific influence model at every `α` [verified, prose].

Why this paper matters to us more than PowerGraph: it predicts the **per-branch failure time step** `f_e`, from which the whole trajectory `s_e[t]` is recovered — a monotone, self-terminating parameterisation of the cascade, and the same trick our IC head uses to prevent saturation.

### 5.3 Gorka, Hsu, Li, Maximov & Roald (PSCC 2024) — blackout size on RTS-GMLC

Setup [verified]: **RTS-GMLC** (73 buses, 120 lines), **DCSIMSEP** simulator, every N-2 line contingency (**7,140**) × every hourly 2020 load/generation profile (**8,784**) = **62,717,760 samples**; 70/15/15 split. Bus features: load, generation. Line features: resistance, reactance, initial failure state. Target: blackout size in MW. **82.4%** of samples have no blackout [verified].

Blackout-size estimation, best topology [verified, Table IV] — MAE / MedAE in MW:

| Model                         | All samples MAE | MedAE | Blackout samples MAE | MedAE | Non-blackout MAE | MedAE |
| ----------------------------- | --------------- | ----- | -------------------- | ----- | ---------------- | ----- |
| R+10 (regression only)        | 8.84            | 0.069 | 39.33                | 16.67 | 2.36             | 0.069 |
| CR+5 (**perfect** classifier) | **2.99**        | 0.00  | **17.09**            | 7.28  | 0.00             | 0.00  |
| CR+5 (XGBoost classifier)     | 10.46           | 0.00  | 39.29                | 7.82  | 4.33             | 0.00  |
| **CVR+5 (XGBoost)**           | **8.67**        | 0.00  | **19.35**            | 7.44  | 6.41             | 0.00  |

The "+5"/"+10" suffix is the augmented topology (physical edges plus learned co-failure edges). The perfect-classifier row is an oracle upper bound, exactly like our `structured_oracle`. The lesson: **classify-then-regress beats direct regression**, because 82% of the mass is at zero — the same class-imbalance structure our `--pos-weight` flag exists to handle.

### 5.4 Jhun et al. (Chaos 2022) ⭐ the Motter–Lai baseline we would compete with

The GNN predicts **avalanche centrality** (a node's Motter–Lai avalanche size), then the top-ranked nodes are reinforced. `R_m` is a mitigation performance measure — **smaller is better**. SHK = Schultz–Heitzig–Kurths synthetic power grids; Spain / France / UK are real transmission topologies the GNN was **not** trained on [verified, Table I]:

| Network      | Random | Degree | EC         | BC     | Avalanche fraction | Failure fraction | AC (exact) | **AC (GNN-predicted)** |
| ------------ | ------ | ------ | ---------- | ------ | ------------------ | ---------------- | ---------- | ---------------------- |
| SHK (N=1000) | 0.4593 | 0.1923 | 0.2767     | 0.1763 | 0.1691             | 0.2697           | **0.1338** | 0.1467                 |
| SHK (N=2000) | 0.4547 | 0.1733 | 0.2582     | 0.1624 | 0.1560             | 0.3102           | **0.1273** | 0.1377                 |
| SHK (N=4000) | 0.4490 | 0.1508 | 0.2423     | 0.1415 | 0.1360             | 0.3569           | **0.1126** | 0.1234                 |
| SHK (N=8000) | 0.4458 | 0.1322 | 0.2199     | 0.1248 | 0.1202             | 0.4431           | **0.1009** | 0.1107                 |
| Spain        | 0.4887 | 0.3347 | 0.3604     | 0.2648 | 0.2815             | 0.2410           | **0.2320** | 0.2347                 |
| France       | 0.4855 | 0.3727 | 0.4593     | 0.2477 | 0.2633             | **0.2171**       | 0.2177     | 0.2306                 |
| UK           | 0.4619 | 0.4220 | **0.2720** | 0.3607 | 0.3203             | 0.2715           | 0.2787     | 0.4227                 |

Three things to take from this table:

- **The GNN tracks the exact quantity closely on synthetic grids** (0.1467 vs. 0.1338 at N=1000) and on Spain, degrades on France, and **fails on the UK** (0.4227 vs. 0.2787 — worse than plain eigenvector centrality). Transfer across real topologies is not solved.
- **Degree is a strong baseline** (0.1322–0.4220), and beats several centralities. Same lesson as our BA-100 degree-triviality result.
- **Random is ≈0.45–0.49 everywhere**, so the achievable dynamic range is roughly 0.45 → 0.10, i.e. a 4.5× spread. That is a genuinely discriminative benchmark, unlike the saturated binary task in §5.1.

Model parameters [verified]: `Pa(S) ~ S^-τ` with `τ ≈ 2.1` at the critical tolerance `a_c ≈ 0.15`; training set = 10⁴ instances; GIN layers with short skip connections; evaluation by Kendall's τ and a rank-weighted measure.

### 5.5 L2RPN NeurIPS 2020 leaderboard [verified, arXiv 2103.03104 Table 1]

| Robustness track     | Score     | Adaptability track   | Score     |
| -------------------- | --------- | -------------------- | --------- |
| **rl_agent** (Baidu) | **59.26** | **rl_agent** (Baidu) | **51.06** |
| binbinchen           | 46.89     | kunjietang           | 49.32     |
| lujixiang            | 44.62     | lujixiang            | 49.26     |
| djmax008             | 43.16     | djmax008             | 42.70     |
| tlEEmCES             | 41.38     | TonyS                | 28.02     |
| jianda               | 39.68     | taka                 | 27.94     |
| UN_aiGridOp          | 34.72     | yzm                  | 19.76     |
| yzm                  | 33.50     | UN_aiGridOp          | 16.76     |
| Konstantin           | 30.38     | var764               | 15.24     |
| panda0246            | 27.36     | PowerRangers         | 12.86     |
| taka                 | 26.73     | mwasserer            | 7.10      |

Approach breakdown for the top teams [verified, Table 2] — figures are top-5 (top-10):

| Track        | Used RL    | Mixed actions | Graph features | **Used `simulate`** |
| ------------ | ---------- | ------------- | -------------- | ------------------- |
| Robustness   | 4/5 (8/10) | 4/5 (9/10)    | 2/5 (5/10)     | **5/5 (10/10)**     |
| Adaptability | 4/5 (9/10) | 3/5 (5/10)    | 1/5 (1/10)     | **5/5 (10/10)**     |

The last column is the finding that matters to us: **every single top-10 team on both tracks used one-step lookahead through the simulator at test time.** The competition is empirically a referendum on the value of a forward model — and the forward model everyone used was the exact simulator, because no learned one was good enough. That is the market our world model is aimed at.

### 5.6 Analytic constants worth quoting

| Result                           | Value                                                                                                                                                                         | Source                                   |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| Motter–Lai capacity rule         | `C_j = (1 + α) L_j(0)`, `L_j` = betweenness; damage `G = N'/N`                                                                                                                | [verified, cond-mat/0301086 Eq. 1 & 3]   |
| Motter–Lai headline              | On a scale-free graph (`γ=3`, `5000 ≤ N ≤ 5100`, `⟨k⟩ ≈ 2.0`), removing the single highest-load node shrinks the LCC by **>20% even at α = 1** (double the required capacity) | [verified, cond-mat/0301086 Fig. 1 text] |
| Motter–Lai contrast              | A homogeneous network shows **no** cascade at `α = 0.05`; the scale-free one collapses to **<10%** of its LCC at the same `α`                                                 | [verified, prose]                        |
| Interdependent ER critical point | `f_c = 0.28467`, `p_c = 2.4554 / ⟨k⟩`, `P_∞ = 1.2564 / ⟨k⟩`                                                                                                                   | [verified, arXiv 0907.1182 §IV]          |
| — its meaning                    | A single ER network percolates at `p_c = 1/⟨k⟩`; two interdependent ones need **2.4554/⟨k⟩** — roughly **2.5× more surviving nodes** to stay connected                        | [derived]                                |
| Avalanche-size exponent          | `Pa(S) ~ S^-τ`, `τ ≈ 2.1` at `a_c ≈ 0.15`                                                                                                                                     | [verified, arXiv 2208.00133]             |
| Polish-case N-2                  | 4,191,960 contingencies; **3,170** (0.076%) produced ≥1 further outage                                                                                                        | [verified, arXiv 1508.01775 §V-C]        |

### 5.7 SOTA summary per benchmark

| Benchmark                                  | Best published                       | Task                                 | Note                                                     |
| ------------------------------------------ | ------------------------------------ | ------------------------------------ | -------------------------------------------------------- |
| **IEEE24 / 39 / 118 / UK (PowerGraph)**    | Graph Transformer / GIN              | binary + multiclass + DNS regression | binary is saturated (0.98–0.998); use the regression row |
| **IEEE89 / IEEE118 cascade sequences**     | Chadaga GNN                          | per-branch failure step              | beats load-specific influence models; 46–223× faster     |
| **RTS-GMLC blackout size**                 | CVR+5 (classify-verify-regress)      | MW regression                        | 8.67 MW MAE all-samples                                  |
| **SHK / Spain / France / UK (Motter–Lai)** | AC exact, GNN-AC close behind        | avalanche mitigation                 | GNN fails to transfer to the UK grid                     |
| **L2RPN IEEE 118**                         | Baidu `rl_agent`                     | sequential control                   | 59.26 / 51.06                                            |
| **US Western grid (4,941/6,594)**          | **none found for cascading failure** | —                                    | published only as an IM / topology benchmark             |

---

## 6. Datasets

### 6.1 What we already load — and why it is not enough

| `--dataset`     | Nodes | Edges | Type                   | Electrical parameters? | Source                                                                                                                                                                                                                                             |
| --------------- | ----- | ----- | ---------------------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `power_grid` ✅ | 4,941 | 6,594 | undirected, unweighted | ❌ **none**            | [nrvis `opsahl-powergrid.zip`](https://nrvis.com/download/data/misc/opsahl-powergrid.zip) (200) · [NetworkRepository page](https://networkrepository.com/opsahl-powergrid.php) (200) · [KONECT](http://konect.cc/networks/opsahl-powergrid/) (200) |

This is the US Western States power grid from **Watts & Strogatz, Nature 393, 440–442 (1998)** — the small-world paper. Nodes are generators, transformers and substations; edges are transmission lines. `data/datasets/power_grid.py` sets `node_feats = log1p(degree)` and `node_labels = zeros` because **there is nothing else in the file** [verified — read from the loader].

It carries no impedances, no ratings, no generator dispatch and no bus loads, so it can support **Motter–Lai** (§2.4) and nothing else in this file. It is authoritative for our purposes in [`influence_maximization.md`](influence_maximization.md) §6, where it is an IM benchmark; this file only adds that it is electrically empty.

### 6.2 Power-system test cases — the column that matters is the last one

All counts **[derived]**: downloaded from [`MATPOWER/matpower/data/`](https://github.com/MATPOWER/matpower) (200) and counted from `mpc.bus` / `mpc.branch` / `mpc.gen`. Where the PGLib paper's Table I lists the same case, the numbers agree exactly [verified, arXiv 1908.02788 Table I].

"Ratings" = number of branches with a non-zero `rateA` (thermal limit). **A zero here is the single most important fact in this table**: a case with no thermal limits cannot drive a DC cascade without you inventing the limits, which is exactly what Chadaga et al. did (§5.2).

| Case                       | Buses          | Branches        | Gens   | Impedance | Ratings (`rateA` ≠ 0) | Total load (MW) | Download                                                                                                     |
| -------------------------- | -------------- | --------------- | ------ | --------- | --------------------- | --------------- | ------------------------------------------------------------------------------------------------------------ |
| `case14` (IEEE 14)         | 14             | 20              | 5      | ✅ 20/20  | ❌ **0/20**           | 259.0           | [`case14.m`](https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case14.m)                       |
| `case24_ieee_rts`          | 24             | 38              | 33     | ✅        | ✅ 38/38              | 2,850.0         | [`case24_ieee_rts.m`](https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case24_ieee_rts.m)     |
| `case30` (IEEE 30)         | 30             | 41              | 6      | ✅        | ✅ 41/41              | 189.2           | [`case30.m`](https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case30.m)                       |
| `case39` (New England)     | 39             | 46              | 10     | ✅        | ✅ 46/46              | 6,254.2         | [`case39.m`](https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case39.m)                       |
| `case57` (IEEE 57)         | 57             | 80              | 7      | ✅        | ❌ **0/80**           | 1,250.8         | [`case57.m`](https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case57.m)                       |
| `case73` / `case_RTS_GMLC` | 73             | 120             | 158    | ✅        | ✅ 120/120            | 8,550.0         | [`case_RTS_GMLC.m`](https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case_RTS_GMLC.m)         |
| `case89pegase`             | 89             | 210             | 12     | ✅        | ⚠️ 77/210             | 5,727.9         | [`case89pegase.m`](https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case89pegase.m)           |
| `case118` (IEEE 118)       | 118            | 186             | 54     | ✅        | ❌ **0/186**          | 4,242.0         | [`case118.m`](https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case118.m)                     |
| `case_ACTIVSg200`          | 200            | 245             | 49     | ✅        | ✅ 245/245            | 1,475.7         | [`case_ACTIVSg200.m`](https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case_ACTIVSg200.m)     |
| `case300` (IEEE 300)       | 300            | 411             | 69     | ✅        | ❌ **0/411**          | 23,525.8        | [`case300.m`](https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case300.m)                     |
| `case_ACTIVSg500`          | 500            | 597             | 90     | ✅        | ✅ 597/597            | 7,750.7         | [`case_ACTIVSg500.m`](https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case_ACTIVSg500.m)     |
| `case1354pegase`           | 1,354          | 1,991           | 260    | ✅        | ⚠️ 1,432/1,991        | 73,059.7        | [`case1354pegase.m`](https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case1354pegase.m)       |
| `case1888rte`              | 1,888          | 2,531           | 298    | ✅        | ⚠️ 2,076/2,531        | 59,110.5        | [`case1888rte.m`](https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case1888rte.m)             |
| `case_ACTIVSg2000`         | 2,000          | 3,206           | 544    | ✅        | ✅ 3,206/3,206        | 67,109.2        | [`case_ACTIVSg2000.m`](https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case_ACTIVSg2000.m)   |
| **`case2383wp` (Polish)**  | 2,383          | 2,896           | 327    | ✅        | ✅ 2,896/2,896        | 24,558.4        | [`case2383wp.m`](https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case2383wp.m)               |
| `case2736sp`               | 2,736          | 3,504           | 420    | ✅        | ✅ 3,504/3,504        | 18,074.5        | [`case2736sp.m`](https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case2736sp.m)               |
| `case2869pegase`           | 2,869          | 4,582           | 510    | ✅        | ⚠️ 2,743/4,582        | 132,437.3       | [`case2869pegase.m`](https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case2869pegase.m)       |
| `case3012wp`               | 3,012          | 3,572           | 502    | ✅        | ✅ 3,566/3,572        | 27,169.7        | [`case3012wp.m`](https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case3012wp.m)               |
| `case6515rte`              | 6,515          | 9,037           | 1,389  | ✅        | ⚠️ 3,131/9,037        | 107,264.0       | [`case6515rte.m`](https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case6515rte.m)             |
| `case9241pegase`           | 9,241          | 16,049          | 1,445  | ✅        | ⚠️ 6,295/16,049       | 312,354.1       | [`case9241pegase.m`](https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case9241pegase.m)       |
| `case_ACTIVSg10k`          | 10,000         | 12,706          | 2,485  | ✅        | ⚠️ 10,244/12,706      | 150,916.9       | [`case_ACTIVSg10k.m`](https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case_ACTIVSg10k.m)     |
| `case13659pegase`          | 13,659         | 20,467          | 4,092  | ✅        | ❌ **0/20,467**       | 381,431.9       | [`case13659pegase.m`](https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case13659pegase.m)     |
| `case_ACTIVSg25k`          | 25,000         | 32,230          | 4,834  | ✅        | not checked           | —               | [`case_ACTIVSg25k.m`](https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case_ACTIVSg25k.m)     |
| `case_ACTIVSg70k`          | 70,000 [claim] | ~88,207 [claim] | —      | ✅        | not checked           | —               | download timed out twice; TAMU page is authoritative                                                         |
| `case_SyntheticUSA`        | 82,000         | 104,121         | 13,419 | ✅        | not checked           | —               | [`case_SyntheticUSA.m`](https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case_SyntheticUSA.m) |

**Every case above ships full electrical data** — impedance, generation limits, bus loads, cost curves — with the sole systematic exception of the `rateA` column, which is zero in five of the canonical IEEE cases (`case14`, `case57`, `case118`, `case300`, `case13659pegase`). Any cascade paper using those cases has invented the thermal limits, and papers that invent them differently are not comparable.

### 6.3 Benchmark libraries

| Library                 | What it is                                                                                                                                                                                | Size                                                                                                                   | Electrical parameters                                                           | URL                                                                                                                                                         |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MATPOWER**            | The reference MATLAB power-system package and its case library. Every `.m` file above lives in `data/`                                                                                    | 85 `.m` case files [verified, GitHub contents API]                                                                     | ✅ full                                                                         | [github.com/MATPOWER/matpower](https://github.com/MATPOWER/matpower) (200) · [matpower.org](https://matpower.org/) (200)                                    |
| **pglib-opf**           | IEEE PES Task Force's curated AC-OPF benchmark library, MATPOWER format, with quality-assured data and reported baselines                                                                 | **66 root cases** plus `api/` (congested) and `sad/` (small-angle-difference) variants [verified, GitHub contents API] | ✅ full, and _curated_ — this is where the missing `rateA` values get filled in | [github.com/power-grid-lib/pglib-opf](https://github.com/power-grid-lib/pglib-opf) (200) · [paper arXiv 1908.02788](https://arxiv.org/abs/1908.02788) (200) |
| **Texas A&M / ACTIVSg** | Synthetic but _geographically realistic_ US grids built on real substation footprints and load patterns; not derived from confidential data, so freely redistributable                    | ACTIVSg200 / 500 / 2000 / 10k / 25k / 70k, plus `SyntheticUSA` (82,000 buses)                                          | ✅ full; time-series load profiles ship separately per case page                | [electricgrids.engr.tamu.edu](https://electricgrids.engr.tamu.edu/electric-grid-test-cases/) (200); the MATPOWER mirrors in §6.2 are the easiest download   |
| **RTS-GMLC**            | The Reliability Test System — Grid Modernization Lab Consortium update. 73 buses, 120 branches, 158 generators [derived], plus **a full year of hourly load, wind and solar time series** | small                                                                                                                  | ✅ full + time series                                                           | [github.com/GridMod/RTS-GMLC](https://github.com/GridMod/RTS-GMLC) (200)                                                                                    |

**pglib-opf naming** maps onto §6.2 directly: `pglib_opf_case118_ieee.m`, `pglib_opf_case2383wp_k.m`, `pglib_opf_case200_activ.m`, `pglib_opf_case73_ieee_rts.m`, `pglib_opf_case9241_pegase.m`, and so on. Cases present in pglib but _not_ in MATPOWER's `data/` include the GOC series (`case179_goc` … `case30000_goc`), the `epigrids` series (`case5658` … `case78484_epigrids`), `case2853_sdet`, `case1803_snem` and `case240_pserc` [verified, GitHub contents API].

### 6.4 European network data

| Dataset              | What it contains                                                                                                                                                                           | Electrical parameters                                                  | URL                                                                                                                                       |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **PEGASE cases**     | Snapshots of the European (ENTSO-E) transmission system produced by the FP7 PEGASE project, distributed through MATPOWER: 89 / 1354 / 2869 / 8387 / 9241 / 13659 buses                     | ✅ full (but see the `rateA` gaps in §6.2)                             | in [MATPOWER `data/`](https://github.com/MATPOWER/matpower)                                                                               |
| **RTE cases**        | French TSO snapshots: 1888 / 1951 / 2848 / 2868 / 6468 / 6470 / 6495 / 6515 buses                                                                                                          | ✅ full                                                                | same                                                                                                                                      |
| **SciGRID**          | Open model of the European transmission grid derived from OpenStreetMap. The original `power.scigrid.de` host **returns 404**; the project page still resolves                             | ⚠️ line lengths and voltage levels, partial electrical parameters      | [scigrid.de](https://www.scigrid.de/) (200); `power.scigrid.de` **404**                                                                   |
| **PyPSA-Eur**        | The actively maintained successor: a full open sector-coupled European energy-system model, with a configurable clustering of the ENTSO-E-derived transmission topology                    | ✅ full, generated by the workflow rather than shipped as a fixed case | [github.com/PyPSA/pypsa-eur](https://github.com/PyPSA/pypsa-eur) (200) · [docs](https://pypsa-eur.readthedocs.io/) (200)                  |
| **UCTE study model** | The winter-peak UCTE model many 2000s cascading-failure papers cite. **No public authoritative download was located** — papers that use "the UCTE network" generally mean a reconstruction | ⚠️ varies                                                              | ENTSO-E publishes statistics ([entsoe.eu/data/power-stats](https://www.entsoe.eu/data/power-stats/), 200) but not the study model [claim] |

The practical takeaway: **use the PEGASE and RTE cases** if you want European topology with electrical data. They are one `curl` away and are what the OPF and cascade literature actually reports.

### 6.5 Historical blackout and outage records — the validation data

These are what the OPA/SOC line validated its power-law tail against, and what Dobson's branching-process and influence-graph estimators are fitted to.

| Source                                                             | What it contains                                                                                                           | Status                                                                                                                                                                                                                        |
| ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **NERC DAWG** (Disturbance Analysis Working Group)                 | US disturbance reports 1984→; the series behind the power-law blackout-size finding                                        | The DAWG database itself is **no longer served at a stable public URL**; NERC's reliability-assessment portal ([nerc.com/pa/RAPA](https://www.nerc.com/pa/RAPA/Pages/default.aspx), 200) is the surviving entry point [claim] |
| **DOE OE-417** electric emergency incident and disturbance reports | Per-incident date, area, event type, customers affected, demand lost (MW) — the modern successor to DAWG                   | The historical `oe.netl.doe.gov/oe417.aspx` host **did not resolve** on any attempt (`000`). Annual summary spreadsheets circulate; no stable canonical URL confirmed [claim]                                                 |
| **BPA transmission line outage data**                              | Bonneville Power Administration's line-outage log — the dataset Dobson's group uses for utility-data propagation estimates | BPA's outage portal resolves ([transmission.bpa.gov](https://transmission.bpa.gov/Business/Operations/Outages/), 200); the historical research extract is not obviously downloadable from it [claim]                          |
| **EIA-411**                                                        | US annual electric-reliability data                                                                                        | [eia.gov/electricity/data/eia411](https://www.eia.gov/electricity/data/eia411/) returned **503** at check time                                                                                                                |

**Do not plan around these.** Every one of them is either unavailable, moved, or requires manual acquisition. Simulated cascades are the practical training source, which is exactly what §5.1–5.3 all do.

### 6.6 Interdependent-network data

The Italian power-grid + SCADA-communication network of Rosato et al. (2008), used in Buldyrev et al.'s Nature figure, is **not publicly downloadable** as far as this review could establish [claim]. Every subsequent interdependent-networks paper works with synthetic ER/SF couplings, for which the analytic constants in §5.6 are the reference points. If we ever want this task, generating the coupling ourselves is the only route.

### 6.7 Software — as load-bearing as the data

| Tool                            | Language             | Role                                                                                                                                                   | URL                                                                                                                                |
| ------------------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| **MATPOWER**                    | MATLAB / Octave      | The reference implementation of power flow, OPF and the case format everything else reads                                                              | [github.com/MATPOWER/matpower](https://github.com/MATPOWER/matpower) (200)                                                         |
| **pandapower**                  | **Python**           | The pragmatic choice for us: pure Python, pip-installable, reads MATPOWER cases, does AC/DC power flow, contingency analysis and topology manipulation | [github.com/e2nIEE/pandapower](https://github.com/e2nIEE/pandapower) (200) · [docs](https://pandapower.readthedocs.io/) (200)      |
| **PyPSA**                       | Python               | Energy-system optimisation and linear power flow; the modelling layer above pandapower's device layer                                                  | [github.com/PyPSA/PyPSA](https://github.com/PyPSA/PyPSA) (200) · [docs](https://pypsa.readthedocs.io/) (200)                       |
| **AC-CFM**                      | MATLAB (on MATPOWER) | The only **cascading-failure simulator with a public repository** found by this review                                                                 | [github.com/mnoebels/AC-CFM](https://github.com/mnoebels/AC-CFM) (200)                                                             |
| **Dynamic CF simulator**        | Python + DIgSILENT   | Dynamic cascades; requires a commercial PowerFactory licence                                                                                           | [github.com/YitianDai/Dynamic-cascading-failure-simulator](https://github.com/YitianDai/Dynamic-cascading-failure-simulator) (200) |
| **ps-res**                      | MATLAB               | Resilience/restoration on top of AC-CFM                                                                                                                | [github.com/AGerkis/ps-res](https://github.com/AGerkis/ps-res) (200)                                                               |
| **Grid2Op** + **lightsim2grid** | Python (+C++)        | RL environment with a cascading-failure game-over condition and a `simulate()` one-step oracle                                                         | [grid2op](https://github.com/Grid2op/grid2op) (200) · [lightsim2grid](https://github.com/Grid2op/lightsim2grid) (200)              |
| **OPA**                         | Fortran/MATLAB       | **No public release found.** The papers describe it; the code is not distributed                                                                       | —                                                                                                                                  |
| **COSMIC / dcsimsep**           | MATLAB               | **No public repository found** under the authors' accounts; `github.com/eehines` returns 404                                                           | —                                                                                                                                  |
| **Manchester model**            | —                    | **No public release found**                                                                                                                            | —                                                                                                                                  |

**If we build the physics version, it is pandapower.** It is the only option in the list that is Python, pip-installable, reads every case in §6.2 unmodified, and does not require MATLAB — and it is what Christianson et al. used ([2410.00796](https://arxiv.org/abs/2410.00796)). Writing the overload-trip loop on top of `pp.rundcpp` is on the order of 100 lines; the hard parts are islanding and generation rebalancing, not the flow solve.

---

## 7. Which paper uses which

| Dataset / case                   | Motter–Lai'02 | Kinney'05 | Hines'10 | Eppstein'12 | Bernstein'14 | Hines–Dobson'16 | AC-CFM'20 | Jhun'22 | Chadaga'24 | PowerGraph'24 | Gorka'24 | L2RPN |
| -------------------------------- | ------------- | --------- | -------- | ----------- | ------------ | --------------- | --------- | ------- | ---------- | ------------- | -------- | ----- |
| Scale-free / ER synthetic        | ✔             |           |          |             |              |                 |           |         |            |               |          |       |
| SHK synthetic grids              |               |           |          |             |              |                 |           | ✔       |            |               |          |       |
| **US Western grid (ours)**       |               |           |          |             |              |                 |           |         |            |               |          |       |
| North American grid (14,099 bus) |               | ✔         |          |             |              |                 |           |         |            |               |          |       |
| Eastern US (40 areas)            |               |           | ✔        |             |              |                 |           |         |            |               |          |       |
| IEEE 24 / RTS                    |               |           |          |             |              |                 |           |         |            | ✔             |          |       |
| IEEE 39 (New England)            |               |           |          | ✔           |              |                 | ✔         |         |            | ✔             |          |       |
| IEEE 118                         |               |           |          |             |              |                 |           |         | ✔          | ✔             |          | ✔     |
| `case89pegase` ("IEEE89")        |               |           |          |             |              |                 |           |         | ✔ [claim]  |               |          |       |
| **`case2383wp` (Polish)**        |               |           |          |             |              | ✔               |           |         |            |               |          |       |
| RTS-GMLC                         |               |           |          |             |              |                 |           |         |            |               | ✔        |       |
| UK transmission                  |               |           |          |             |              |                 |           | ✔       |            | ✔             |          |       |
| Spain / France transmission      |               |           |          |             |              |                 |           | ✔       |            |               |          |       |
| Western US / WECC                |               |           |          |             | ✔            |                 |           |         |            |               |          |       |

**The US Western grid row is empty on purpose.** Our one power dataset appears in the IM literature and the small-world literature, and in **no** cascading failure result table found by this review. That is worth knowing before we propose it as a benchmark: any number we produce on it will have no precedent (the same position `sbm` occupies in the IM file).

---

## 8. Evaluation protocol

### 8.1 What gets measured

| Metric                        | Definition                                               | Who reports it                                                                |
| ----------------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **Demand not served (DNS)**   | MW of load shed at the end of the cascade                | PowerGraph (regression), AC-CFM, Cascades                                     |
| **Blackout size**             | MW shed, or fraction of load lost                        | Gorka et al., OPA line                                                        |
| **Failure size**              | number of branches failed at termination                 | Chadaga et al. (`l_size`), Hines–Dobson                                       |
| **Final state error**         | per-branch mismatch of the terminal `s[T]`               | Chadaga et al. (`l_state`)                                                    |
| **Failure step error**        | per-branch mismatch of the generation at which `e` trips | Chadaga et al. (`l_failure-step`) — **the finest granularity anyone reports** |
| **Cascade-size distribution** | the empirical PDF, and whether its tail is a power law   | OPA line, Hines–Dobson (KS-style visual match)                                |
| **Balanced accuracy**         | for the binary "does a cascade occur" screen             | PowerGraph                                                                    |
| **Prediction time**           | seconds per 1,000 samples, model vs. simulator           | Chadaga et al.                                                                |
| **`R_m`**                     | mitigation performance under a reinforcement budget      | Jhun et al.                                                                   |

### 8.2 The traps

- **Class imbalance is severe and is not incidental.** 79.7–93.9% of PowerGraph instances are "no cascade, no unserved demand"; 82.4% of Gorka et al.'s 62.7M samples have zero blackout. A model that predicts "nothing happens" scores well on accuracy and is useless. Report balanced accuracy _and_ the regression error on the positive subset separately — Gorka et al.'s Table IV (§5.3) is the right template.
- **Simulator identity is part of the result.** DC (dcsimsep, OPA) vs. AC (AC-CFM, Cascades) vs. dynamic (COSMIC) give materially different cascade sizes; the DC model is known to **underestimate** failure sizes [claim, Chadaga et al. §I]. The IEEE PES benchmarking papers (§3.3) exist because of this. Never compare a number across simulators.
- **Missing thermal limits.** Five canonical IEEE cases ship `rateA = 0` (§6.2). Every cascade result on them depends on an invented capacity rule — Chadaga used `2 ×` the base-case flow. Two papers using different rules are not comparable, and neither states it prominently.
- **Load level is the hidden variable.** Cascade size is extremely sensitive to loading; OPA's whole point is a critical loading transition. Chadaga sweeps `α ~ U[1,2]` and reports _per-`α`_ error precisely because a single-load result is meaningless. Any protocol we adopt must sweep load.
- **N-2 enumeration is the standard initiating set** (§2.5 sizes it), but contingency _lists_ differ: PowerGraph samples 43–245 outage lists per grid; Hines–Dobson enumerates all 4.19M. A "N-2 result" without the list size is uninterpretable.
- **Betweenness-based results are not power-system results** (§3.5). Say so.

---

## 9. Implications for this project

1. **Do Motter–Lai. Do not do DC power flow — yet.** The abstract load-capacity model runs on the graph we already load, needs no new dependency, gives a genuinely global-redistribution `T_endo`, and has a published GNN baseline with a transcribed table (§5.4). Estimated cost: a simulator function in `data/` (betweenness → capacity → trip → repeat), one extra `X` channel (`L_j(t)/C_j`), the existing node-state pipeline unchanged. **~2 days.**

2. **The `α` sweep replaces our budget sweep.** IM sweeps `k ∈ {1,5,10,20}%`; Motter–Lai sweeps the tolerance `α`. Small `α` gives catastrophic cascades, large `α` gives nothing, and `α_c ≈ 0.15` is where the avalanche distribution is critical. That is a principled difficulty dial our BA-100 experiments never had, and it is the direct analogue of the "report low budgets where methods separate" lesson from the IM file.

3. **Expect the k-hop encoder to hurt, and measure it.** §2.2 predicts a structural ceiling: a 3-layer GCN cannot see a node about to fail 10 hops away. The cheap experiment is an `--n-layers` sweep — if accuracy keeps climbing well past the point where it plateaus on IC/LT, that is the receptive-field bound showing itself, and it is a publishable observation in its own right.

4. **Build the influence graph as the fix.** Hines & Dobson's result (§2.2) says the locality assumption is recoverable on a _learned_ graph. Harvesting `H[i,j] = P(j fails at t+1 | i failed at t)` from our own generated cascades and running the encoder on `H` instead of `G` is a clean ablation, and it connects this task to [`network_inference.md`](network_inference.md).

5. **The physics version is a separate project, not a dataset swap.** §2.5 lists what it costs. If it happens, the order is: pandapower + `case_ACTIVSg200` (has ratings) → `case118` (needs invented ratings, but is the field default) → `case2383wp` (the one case with a published reference cascade distribution).

6. **Reuse the rollout metrics we already have.** `ens_final_count_model/true` and `ens_count_bias` map directly onto "number of failed components" and "does the learned cascade saturate". The genuinely new metric is the **blackout-size distribution**, which is distributional rather than per-step.

7. **The gap we can claim is real and narrow.** Nobody has published a step-wise, action-conditioned, autoregressively-rolled learned cascade model with rollout-fidelity metrics (§4.5). Every published GNN is one-shot; every published planner uses the exact simulator. If we frame the contribution as _"a learned forward model good enough to replace `obs.simulate()`"_ — the thing 10/10 L2RPN finalists used (§5.5) — the motivation writes itself.

8. **The `set_edge_weight` semantics need a decision.** It currently means an IC transmission probability. For cascades it means a thermal rating. That is not a rename — the `structured` head reads it as a probability. Either add a sixth op or make the head's interpretation dynamics-dependent.

---

## 10. Reference list

**Classical / physics** [Carreras, Newman, Dobson & Poole, SOC evidence (HICSS'00)](http://iandobson.ece.iastate.edu/PAPERS/carrerasHICSS00.pdf) · [Dobson, Carreras, Lynch & Newman, OPA (HICSS'01)](http://iandobson.ece.iastate.edu/PAPERS/dobsonHICSS01.pdf) · [Dobson et al., criticality (HICSS'02)](http://iandobson.ece.iastate.edu/PAPERS/dobsonHICSS02.pdf) · [Carreras, Lynch, Dobson & Newman (Chaos'02)](http://iandobson.ece.iastate.edu/PAPERS/carrerasChaos02.pdf) · [Dobson, Carreras, Lynch & Newman (Chaos'07)](http://iandobson.ece.iastate.edu/PAPERS/dobsonCHAOS07.pdf) · [Dobson publication list](http://iandobson.ece.iastate.edu/publications.html) · [Motter & Lai (PRE'02, cond-mat/0301086)](https://arxiv.org/abs/cond-mat/0301086) · [Motter, cascade control (PRL'04, cond-mat/0401074)](https://arxiv.org/abs/cond-mat/0401074) · [Crucitti, Latora & Marchiori (PRE'04, cond-mat/0309141)](https://arxiv.org/abs/cond-mat/0309141) · [Kinney, Crucitti, Albert & Latora (EPJB'05, cond-mat/0410318)](https://arxiv.org/abs/cond-mat/0410318) · [Bernstein, Bienstock, Hay, Uzunoglu & Zussman (arXiv 1206.1099)](https://arxiv.org/abs/1206.1099) · [Hines et al., COSMIC (arXiv 1411.3990)](https://arxiv.org/abs/1411.3990) · [Hines, Dobson & Rezaei, influence graph (arXiv 1508.01775)](https://arxiv.org/abs/1508.01775) ⭐ · [Zhou & Dobson, Markovian influence graph (arXiv 1902.00686)](https://arxiv.org/abs/1902.00686) · [Zhou & Dobson, N-k motifs (arXiv 2209.02192)](https://arxiv.org/abs/2209.02192) · [Buldyrev et al. (Nature'10)](https://www.nature.com/articles/nature08932) · [arXiv 0907.1182](https://arxiv.org/abs/0907.1182) · [Parshani, Buldyrev & Havlin (arXiv 1004.3989)](https://arxiv.org/abs/1004.3989) · [Network-of-networks for electrical infrastructure (arXiv 1512.01436)](https://arxiv.org/abs/1512.01436)

**Learning-based** [Chadaga, Wu & Modiano (arXiv 2404.16134)](https://arxiv.org/abs/2404.16134) ⭐ · [PowerGraph (arXiv 2402.02827)](https://arxiv.org/abs/2402.02827) ⭐ · [code](https://github.com/PowerGraph-Datasets) · [Jhun et al. (arXiv 2208.00133)](https://arxiv.org/abs/2208.00133) ⭐ · [Gorka et al. (arXiv 2403.15363)](https://arxiv.org/abs/2403.15363) ⭐ · [Dwivedi & Tajer, GRNN fault chains (arXiv 2303.08864)](https://arxiv.org/abs/2303.08864) · [follow-up 2503.09775](https://arxiv.org/abs/2503.09775) · [Hyperparametric diffusion model (arXiv 2406.08522)](https://arxiv.org/abs/2406.08522) · [Graph neural jump ODEs (arXiv 2603.20838)](https://arxiv.org/abs/2603.20838) · [GRU-gated graph attention (arXiv 2605.07010)](https://arxiv.org/abs/2605.07010) · [Owerko, Gama & Ribeiro (arXiv 1910.09658)](https://arxiv.org/abs/1910.09658) · [unsupervised 2210.09277](https://arxiv.org/abs/2210.09277) · [Donon et al., LEAP nets (arXiv 1908.08314)](https://arxiv.org/abs/1908.08314) · [CANOS (arXiv 2403.17660)](https://arxiv.org/abs/2403.17660) · [OPFData (arXiv 2406.07234)](https://arxiv.org/abs/2406.07234) · [PowerFlowNet (arXiv 2311.03415)](https://arxiv.org/abs/2311.03415) · [DeepOPF (arXiv 1905.04479)](https://arxiv.org/abs/1905.04479) · [Christianson, Cui & Low, N-k ICNN (arXiv 2410.00796)](https://arxiv.org/abs/2410.00796) · [Nakiganda & Chatzivasileiadis (arXiv 2310.04213)](https://arxiv.org/abs/2310.04213)

**Grid2Op / L2RPN** [grid2op](https://github.com/Grid2op/grid2op) · [lightsim2grid](https://github.com/Grid2op/lightsim2grid) · [docs](https://grid2op.readthedocs.io/) · [l2rpn.chalearn.org](https://l2rpn.chalearn.org/) · [L2RPN retrospective (arXiv 2103.03104)](https://arxiv.org/abs/2103.03104) ⭐ · [environment design (arXiv 2104.04080)](https://arxiv.org/abs/2104.04080) · [RL2Grid benchmark (arXiv 2503.23101)](https://arxiv.org/abs/2503.23101) · [AlphaZero topology control (arXiv 2211.05612)](https://arxiv.org/abs/2211.05612) · [PowRL (arXiv 2212.02397)](https://arxiv.org/abs/2212.02397) · [Graph RL for power grids survey (arXiv 2407.04522)](https://arxiv.org/abs/2407.04522)

**Surveys** [Interaction graphs for CF analysis (arXiv 1911.00475)](https://arxiv.org/abs/1911.00475) · [ML in cascading failure analysis (arXiv 2305.19390)](https://arxiv.org/abs/2305.19390) · [Topology-RL survey (arXiv 2504.08210)](https://arxiv.org/abs/2504.08210)

**Data and software** [MATPOWER](https://github.com/MATPOWER/matpower) · [pglib-opf](https://github.com/power-grid-lib/pglib-opf) · [PGLib paper (arXiv 1908.02788)](https://arxiv.org/abs/1908.02788) · [Texas A&M ACTIVSg](https://electricgrids.engr.tamu.edu/electric-grid-test-cases/) · [RTS-GMLC](https://github.com/GridMod/RTS-GMLC) · [pandapower](https://github.com/e2nIEE/pandapower) · [PyPSA](https://github.com/PyPSA/PyPSA) · [PyPSA-Eur](https://github.com/PyPSA/pypsa-eur) · [AC-CFM](https://github.com/mnoebels/AC-CFM) · [US Western grid (nrvis)](https://nrvis.com/download/data/misc/opsahl-powergrid.zip)

---

## 11. Open gaps

Honest list of what this review could **not** establish.

- **No learned cascading-failure world model exists to compare against.** Searches over the arXiv full-text index for `"world model"` + `"power grid"` / `"power system"` and `"learned simulator"` + `"power grid"` returned nothing relevant [derived, arXiv API 2026-07]. This is the single biggest gap, and it cuts both ways: it is our contribution, and it means **we will have no step-wise baseline** — every comparison must be against a one-shot predictor (Chadaga, PowerGraph) evaluated on its terms, not ours.
- **Chadaga et al. publish no accuracy table** — the error rates are figures only (§5.2), so our numbers can be compared to theirs on runtime but only directionally on accuracy. Their cascade-sequence data pools (2 × 200,000 samples) were not released, and no code repository was found.
- **"IEEE89" is unidentified.** Chadaga et al. never say which 89-bus case it is; `case89pegase` is the only 89-bus case in MATPOWER, so the identification is [claim], not verified.
- **The CFS oracle Chadaga et al. use** is cited to a prior paper and not named as a public tool; no repository was located.
- **No public code for OPA, COSMIC, dcsimsep, or the Manchester model.** The entire classical simulator lineage is papers-only; AC-CFM is the sole exception (§6.7). Reproducing any pre-2020 physics result means reimplementing the simulator.
- **`case_ACTIVSg70k` counts are [claim].** The MATPOWER file timed out on two download attempts; 70,000 buses is inferred from the name. `case_ACTIVSg25k` and `case_SyntheticUSA` were counted but their `rateA` columns were not checked.
- **Grid2Op per-environment sizes are [claim].** The bundled `grid.json` files resisted two parsing attempts; only the environment _names_ are verified. The IEEE 118 figure for L2RPN NeurIPS 2020 is verified from the paper, not the repo.
- **Historical outage data is effectively unavailable** (§6.5): NERC DAWG has no stable URL, DOE OE-417's host did not resolve, EIA-411 returned 503. The power-law blackout-size finding therefore cannot be re-derived by us from primary data — it is [claim] on our side, however well established in the literature.
- **The Italian power/communication interdependent network is not public** (§6.6).
- **Paywalled with no free version located:** Hines, Cotilla-Sanchez & Blumsack (_Chaos_ 2010, the topological-models critique), Vaiman et al. (TPWRS 2012), Bialek et al. (TPWRS 2016), Eppstein & Hines (TPWRS 2012, whose DOI returned 404), Yang, Nishikawa & Motter (_Science_ 2017, 403). All four are cited from their titles and abstracts only.
- **Cetinay/Van Mieghem (AC vs. DC failure effects) and Soltan/Zussman (analysis of failures in power grids)** were identified by name but no verified URL was obtained; they are omitted from §3.3 rather than guessed at.
- **`--pos-weight` behaviour on this task is untested.** Our IC experience says `off` for structured heads; the 80–94% negative-class imbalance here (§8.2) may well invert that. Unknown until run.
