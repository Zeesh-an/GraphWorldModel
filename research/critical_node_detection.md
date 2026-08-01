# Critical Node Detection — Prior Work, Datasets, and Published Results

The literature on removing a small set of nodes (or edges) to break a network: **critical node detection (CNDP/CNP)**, **network dismantling**, **network disintegration**, **optimal percolation**, **vital node identification**, the **key player problem**, and structural **immunization-by-removal**. These are one literature with four vocabularies — an operations-research branch that solves an exact integer program for a fixed budget `k`, a statistical-physics branch that sequentially removes nodes until the giant component collapses, a network-science branch that ranks nodes by centrality, and a machine-learning branch (FINDER, GDM, NIRM) that learns the removal policy.

This file catalogues all four, transcribes the published tables, and states plainly where the task fits our action-conditioned world model and where it does not. **The short version of §2: the structural variant has no diffusion dynamics and its "simulator" is `nx.connected_components`, which is cheaper than our model's forward pass — there is nothing to accelerate. The diffusion-based variant (dismantle to contain a cascade) is a direct, cheap fit.**

All URLs verified on **2026-07-28** unless annotated otherwise.

---

## 0. Verification policy

| Tier           | Meaning                                                                                                              |
| -------------- | -------------------------------------------------------------------------------------------------------------------- |
| **[verified]** | Read from the paper's own table via text extraction, or from the data repository's own statistics page. Trustworthy. |
| **[derived]**  | Computed by us from a file we downloaded, or reconstructed arithmetically from published splits.                     |
| **[figure]**   | Read off a plotted figure — the paper published no table. Approximate, direction only.                               |
| **[claim]**    | Stated in prose by a paper or a secondary source; not cross-checked against a file or table.                         |

**Automated PDF summarizers hallucinate plausible numbers from these papers.** Every `[verified]` cell below was produced by `curl -sL <url> -o x.pdf` followed by `pdftotext -layout x.pdf` and reading the extracted text. No number in this file came from a summarizer.

**Edge-count convention.** Undirected graphs are quoted as **undirected edges**; directed graphs as **arcs**. Our loaders report `adjacency.nnz`, which for a symmetrized undirected graph is **2× the undirected edge count**. Check this before concluding two graphs differ.

**This literature has a second convention trap that IM does not** — see §8.2. Half these papers remove `k` nodes _in one batch_ and report a connectivity objective; the other half remove nodes _one at a time, recomputing after each_, and report an area-under-curve. The numbers are not interconvertible, and papers routinely cite across the divide without saying so.

---

## 1. Task definition

### 1.1 The structural core

Given an undirected graph `G = (V, E)` and a budget `k`, choose `S ⊆ V`, `|S| ≤ k`, to damage the _connectivity_ of the residual graph `G[V \ S]`. The family splits by which connectivity functional you write down.

Let `C_1, …, C_m` be the connected components of `G[V \ S]`.

| # | Objective                   | Formal                                                                                  | Name in the literature                     | Introduced by          |
| --- | --------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------ | ---------------------- |
| 1 | **Pairwise connectivity**   | minimize `Σ_i\|C_i\|(\|C_i\|−1)/2`                                                      | **CNP** / CNP-1a — _the_ canonical variant | Arulselvan et al. 2009 |
| 2 | **Largest component**       | minimize `max_i\|C_i\|` s.t. `\|S\| ≤ k`                                                   | MinMaxC / β-vertex-disruptor               | Shen & Smith 2012      |
| 3 | **Cardinality-constrained** | minimize `\|S\|` s.t. `max_i\|C_i\| ≤ L`                                                   | **CC-CNP**                                 | Arulselvan et al. 2011 |
| 4 | **Component count**         | maximize `m` s.t. `\|S\| ≤ k`                                                            | MaxNum CNP                                 | Shen & Smith 2012      |
| 5 | **Distance-based**          | minimize # of node pairs at distance ≤ `D`; or minimize Harary index / graph efficiency | **DCNP**                                   | Veremyev et al. 2015   |
| 6 | **Sequential GCC decay**    | remove nodes one at a time; report `s(q) = \|GCC(G − S_q)\| / N` as a curve               | **network dismantling**                    | Braunstein et al. 2016 |

Objectives 1–5 are _one-shot batch_ formulations: you pick `S` once, you evaluate once. Objective 6 is _sequential_: `S_q` grows by one node per step and the residual graph is recomputed each step, so the "state" is the whole graph.

### 1.2 The scalar summaries used to compare dismantling curves

Objective 6 is a curve, and papers collapse it to a scalar. Three collapses are in circulation and they are **not** interchangeable:

- **Schneider's `R`** (robustness integral), Schneider et al. PNAS 2011: `R = (1/N) Σ_{q=1}^{N} s(q)` where `s(q)` is the GCC fraction after removing `q` nodes. The `1/N` normalization makes `R ∈ [1/N, 0.5]`. **Lower = better attack.** ([arXiv 1009.3125](https://arxiv.org/abs/1009.3125) · PNAS DOI `10.1073/pnas.1009440108` — pnas.org returns **403** to scripted requests)
- **ANC** (accumulated normalized connectivity), FINDER's training objective: `R(v_1,…,v_N) = (1/N) Σ_{k=1}^{N} [σ(G \ {v_1…v_k}) / σ(G)] · c(v_k)`, where `σ` is _any_ connectivity measure (pairwise connectivity, GCC size, or component count — FINDER instantiates all three) and `c(v_k)` is the normalized removal **cost** of `v_k`. With unit costs and `σ = |GCC|`, ANC is Schneider's `R`. With `σ = ` pairwise connectivity it is a different number. **Papers quoting "ANC" without naming `σ` are unusable as a baseline.**
- **Dismantling-set size at threshold `C`**: the smallest `|S|` such that `|GCC(G − S)| ≤ C·N`, conventionally `C = 0.01`. Reported as a _fraction of N_. This is the Min-Sum / CoreHD / GND convention.

`R` and "set size at C=0.01" rank methods differently: `R` weights the whole curve including the long tail after collapse; set-size only cares where the curve crosses one line.

### 1.3 Adjacent problems folded into the same literature

- **Decycling / minimum feedback vertex set.** Removing all cycles is a sufficient first step for dismantling, because a forest of `n` nodes breaks into small trees under few further removals. Min-Sum and CoreHD both decycle first. Decycling is _not_ dismantling on networks with many short loops, and both papers add a **reinsertion** pass to recover the gap (§8.2).
- **Optimal percolation** (Morone & Makse). Find the minimal set whose removal drives the giant component to zero — the percolation threshold. Framed as minimizing the largest eigenvalue of a non-backtracking matrix, giving the **Collective Influence** score.
- **Key player problem** (Borgatti 2006). KPP-Neg = remove `k` to maximally fragment (= CNP); KPP-Pos = select `k` to maximally reach (= influence maximization). The two halves are the same optimization with the sign flipped — which is exactly the relation between this file and [`influence_maximization.md`](influence_maximization.md).
- **Structural immunization.** Remove/vaccinate `k` nodes _before_ an epidemic to minimize the eventual outbreak. When the objective is measured by simulating SIR/IC rather than by a connectivity functional, this becomes the **diffusion-based** variant that §2 says is our real fit. The purely epidemiological framing lives in [`epidemic_control.md`](epidemic_control.md); the _structural proxy_ for it lives here.
- **Edge variants.** Critical edge detection / link removal, and **Generalized Network Dismantling** (GND), which prices removal by node cost (e.g. degree) rather than counting nodes.

### 1.4 Complexity

CNP is **NP-complete** on general graphs (Arulselvan et al. 2009). The tractability boundary has been mapped in detail — see §3.1 for the per-graph-class results (trees, series-parallel, bounded treewidth, split/bipartite graphs) and the parameterized-complexity picture.

### 1.5 Why the problem exists

Anti-terror / criminal-network disruption, epidemic containment by quarantine, drug-target selection in protein interaction networks, identifying single points of failure in power grids and transport networks, and — inverted — the _robustness_ question of which nodes to protect.

---

## 2. Fit with our methodology

There are two different tasks under this heading and they have opposite verdicts. Stating this precisely is the point of this section.

### 2.1 Structural CNDP: `T_endo` is trivial and there is no simulator to replace

Our world model factorizes `s_{t+1} = T_endo(T_exo(s_t, a_t))` — a deterministic action effect followed by a stochastic diffusion step. In structural CNDP:

- `T_exo` = delete the chosen node and its incident edges. Deterministic. Fine.
- `T_endo` = **nothing**. There is no diffusion. Between interventions the state does not move.

The "state" `s_t = (infected, frontier)` has no meaning here. The natural observable is _per-node membership in the largest remaining component_, and the model degenerates from a transition kernel to a **one-step connectivity regressor** `f(G, S) → [v ∈ GCC(G − S)]_{v∈V}`. That is not a world model; it is a graph-labelling task. (We already built exactly this once and deleted it — `data/old/generate_cnd_data.py` + `data/old/connectivity.py` produce `removal_sets → connectivity_vecs / n_components / largest_cc_sizes / pairwise_conn` and nothing else. See §9.)

Three concrete objections, in increasing order of how much they should change our mind:

1. **A 3-layer GNN cannot represent the target.** GCC membership is a _global_ property: two nodes are in the same component iff a path of any length joins them. A `k`-layer message-passing network has receptive field `k` hops, so it can only learn connected components on graphs of diameter `≤ k`. Our recommended sizes are 3–6 layers (`CLAUDE.md` §Backbones) against graphs of diameter 6 (`facebook`) to 46 (`power_grid`). Only GCNII at 16+ layers has the depth, and even then the initial-residual term fights the long-range propagation this needs. The one-step numbers would look fine (most nodes stay in the GCC; it is an unchanged-dominated metric exactly like our `infected_acc`) and the substantive metric would be near zero.
2. **There is no expensive simulator to replace.** The entire value proposition of a learned world model is that `f_θ` is cheaper than the ground-truth environment. Our IC/LT ground truth costs `--mc-marginals 30` NDlib rollouts per step. The structural-CNDP ground truth costs one call to `nx.connected_components`, which is `O(N + E)` — **cheaper than a single GNN forward pass** on every graph in our suite. There is no speedup to buy, and a learned approximation strictly loses to the exact answer.
3. **The action semantics do not exist in our op set.** See §2.3 — none of our five ops deletes a node from the graph.

**Verdict:** structural CNDP is a bad fit for the _transition-learning_ half of this project, and we should say so in the paper rather than force it. It remains a perfectly good fit for the **coding-agent outer loop** — evolving dismantling heuristics (a `select → score → remove → recompute` program) against an exact, cheap evaluator is a clean condition, and it is the setting where the strongest published learned baselines (FINDER, GDM) live, so there is a real table to beat. In that framing the arm is `one_shot_free@native`-style: real execution only, no world model in the loop.

### 2.2 Diffusion-based CND (dismantling for containment): a direct fit

Reformulate: choose `S`, `|S| ≤ k`, to **minimize the eventual IC/LT spread** from a given (or adversarial, or random) seed set. This is the same simulator we already harvest from, the same `(G, s_t, a_t, s_{t+1})` transitions, the same soft MC-marginal targets — only the planner's objective sign flips relative to influence maximization, and the action distribution shifts from `add_node` to `remove_node`.

Everything the model needs already exists:

| Piece            | Status                                                                                                                                                             |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Simulator        | ✅ NDlib IC/LT, already stepwise (`data/wm_simulator.py`)                                                                                                          |
| Action op        | ✅ `remove_node` already in `ACTION_OPS`, already a feature channel (`CH_REMOVE`)                                                                                  |
| Feature encoding | ✅ `X` col 4 marks `remove_node` targets                                                                                                                           |
| Structured heads | ✅ `ICTransmissionHead` / `LTThresholdHead` — the `frontier_u` gate that makes cascades self-terminating is _exactly_ what makes a containment objective learnable |
| Eval metric      | ✅ `remove_frontier_success` in `wm_eval.py` is already a containment metric                                                                                       |
| Planning loop    | ✅ `planning_regret_multi` — flip `argmax` to `argmin`                                                                                                             |

What is missing is small: a minimize-mode planner objective, a data-generation mode that injects `remove_node` bags rather than `add_node`, and honest containment metrics (§8.3). This is the variant worth implementing.

### 2.3 ⚠️ What `remove_node` actually means in our simulator

`data/wm_simulator.py::apply_actions` sets, for target `v`:

```python
self.model.status[int(v)] = 2 if self.model_name == "IC" else 0
```

and `active_nodes()` counts `status ∈ {1, 2}` as infected under IC, `status == 1` under LT. The two dynamics therefore give `remove_node` **opposite** meanings, and _neither_ is node deletion:

|                        | IC                                           | LT                                              |
| ---------------------- | -------------------------------------------- | ----------------------------------------------- |
| status after           | `2` (NDlib _Removed_)                        | `0` (_Susceptible_)                             |
| counted as infected?   | **yes** — `active_nodes()` includes status 2 | **no**                                          |
| can spread again?      | no                                           | no (not in frontier this step)                  |
| can re-activate later? | no                                           | **yes** — neighbours can re-cross its threshold |
| edges removed?         | no                                           | no                                              |

Three consequences that matter for CND and would silently corrupt any containment result:

1. **Under IC, immunizing a healthy node counts it as infected.** `remove_node` on a susceptible `v` sets `status = 2`, which `active_nodes()` reports as active. A `k`-node pre-emptive immunization therefore inflates the measured final spread by exactly `+k` before any diffusion happens. Any "minimize final infected count" objective computed against this simulator carries a systematic `+k` bias. This is fine for IM (where `remove_node` models "this spreader is spent") and wrong for CND.
2. **Under LT, `remove_node` is not immunization at all** — it is a one-step suppression. The node returns to Susceptible with all its edges intact and re-activates on the next step if its active-neighbour weight still exceeds `θ_v` (which, since `θ_v` is fixed per episode and the neighbourhood only grows under LT's monotone dynamics, it almost always will). Removing an LT node buys one timestep.
3. **Neither op changes `G`.** Structural CNDP needs `G[V \ S]`; our op set has no `delete_node`.

**Lazy fix — no new op.** Express node deletion as a _bag_: `remove_node(v)` + `remove_edge(u, v)` for every incident `u`. Our action encoding already supports multi-op bags per step, `CH_EDGE` already marks edge-op endpoints, and `reconstruct_episode_adjacency` already replays edge actions so the model sees the post-deletion adjacency. Under LT this also _fixes_ the re-activation leak: an isolated node has no active neighbours and can never cross its threshold. Under IC it does **not** fix the `+k` counting bias — that needs either a status-`0` branch for `remove_node` under IC, or (cleaner) a sixth op. Cost: `O(deg(v))` ops per deletion, and `N` never shrinks, which is fine because our features are per-node masks anyway.

**Flag for the paper:** the IC/LT asymmetry in `remove_node` is currently undocumented outside the source comment. It is defensible for IM and indefensible for CND, and whichever way we resolve it must be stated.

### 2.4 What each objective would cost us

| Objective                             | New code                                                                        | Cost                               |
| ------------------------------------- | ------------------------------------------------------------------------------- | ---------------------------------- |
| Pairwise connectivity                 | resurrect `data/old/connectivity.py` (already computes it)                      | ~0                                 |
| LCC size / GCC fraction               | same file, already computed                                                     | ~0                                 |
| Component count                       | same file, already computed                                                     | ~0                                 |
| Schneider `R` / ANC                   | a sequential-removal evaluator + integral in `wm_metrics.py`                    | ~half a day                        |
| Dismantling-set size at `C=0.01`      | same evaluator, different reduction                                             | ~0 once `R` exists                 |
| Distance-based (Harary / efficiency)  | new; `nx` has `efficiency`, but `O(N·(N+E))` per evaluation                     | ~half a day, slow on `netphy`+     |
| **Final IC/LT spread after removals** | flip the planner sign + a `remove_node` injection mode in `generate_wm_data.py` | **~1–2 days, the one worth doing** |

**Honest total.** Diffusion-based CND: **1–2 days**, reusing the entire existing stack, and it produces a second task for the same trained backbone. Structural CNDP as a _world-model_ task: **1–2 weeks**, requires a deep backbone, a new head whose target a 3-layer GNN provably cannot represent, and buys no speedup over `nx.connected_components`. Structural CNDP as a _coding-agent-only_ condition (agent evolves a dismantling heuristic, exact evaluator, no learned model): **2–3 days** and it gets us onto the FINDER/GDM table, which is the strongest published comparison in this file.

---

## 3. Classical, exact, and heuristic methods

Three traditions, three vocabularies, almost no cross-citation until ~2018. §3.1 solves an integer program for fixed `k`. §3.2 heuristically attacks the same integer program. §3.3 removes nodes sequentially until the giant component dies and never writes down an IP at all.

### 3.1 Exact / complexity line (operations research)

| Method / result                                                       | Year | Venue                                                                  | Idea                                                                                                                                                                                         | Paper                                                                                                   | Code |
| --------------------------------------------------------------------- | ---- | ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ---- |
| **CNP** — Arulselvan, Commander, Elefteriadou, Pardalos               | 2009 | Computers & OR 36(7):2193–2200                                         | **Defines the problem.** Integer program minimizing pairwise connectivity subject to `Σv_i ≤ k`; proves NP-completeness; gives a combinatorial heuristic seeded by a maximal independent set | [doi:10.1016/j.cor.2008.08.016](https://doi.org/10.1016/j.cor.2008.08.016)                              | —    |
| **CC-CNP** — Arulselvan, Commander, Shylo, Pardalos                   | 2011 | Springer chapter (Perf. Models & Risk Mgmt in Comm. Systems) pp. 79–91 | Flip the constraint: minimize `\|S\| `subject to`max\|C_i\| ≤ L`; ILP + GA heuristic                                                                                                           | [doi:10.1007/978-1-4419-0534-5_4](https://doi.org/10.1007/978-1-4419-0534-5_4)                          | —    |
| **CNP on trees** — Di Summa, Grosso, Locatelli                        | 2011 | Computers & OR 38(12):1766–1774                                        | Complexity of CNP restricted to trees; dynamic programming                                                                                                                                   | [doi:10.1016/j.cor.2011.02.016](https://doi.org/10.1016/j.cor.2011.02.016)                              | —    |
| **Branch-and-cut** — Di Summa, Grosso, Locatelli                      | 2012 | Comput. Optim. Appl. 53:649–680                                        | The triangle-inequality ILP (`e_ij + e_jk − e_ik ≥ 0` …) solved by branch-and-cut. This is the formulation the 2025 CND survey reproduces as eq. (20)                                        | [doi:10.1007/s10589-011-9420-4](https://doi.org/10.1007/s10589-011-9420-4)                              | —    |
| **Bounded treewidth** — Addis, Di Summa, Grosso                       | 2013 | Discrete Applied Math. 161(16-17):2349–2360                            | Polynomial algorithms for bounded-treewidth graphs; complexity results across graph classes                                                                                                  | [doi:10.1016/j.dam.2013.03.021](https://doi.org/10.1016/j.dam.2013.03.021)                              | —    |
| **Trees & series-parallel** — Shen & Smith                            | 2012 | Networks 60(2):103–119                                                 | Polynomial-time DP for a class of CNPs on trees and series-parallel graphs                                                                                                                   | [doi:10.1002/net.20464](https://doi.org/10.1002/net.20464) — Wiley returns **403** to scripted requests | —    |
| **Interdiction models** — Shen, Smith, Goli                           | 2012 | Discrete Optimization 9(3):172–188                                     | Exact models for disconnecting networks via node deletion, several objectives                                                                                                                | [doi:10.1016/j.disopt.2012.07.001](https://doi.org/10.1016/j.disopt.2012.07.001)                        | —    |
| **Compact formulations** — Veremyev, Boginski, Pasiliao               | 2014 | Optimization Letters 8:1245–1259                                       | New compact IP formulations; exact solutions on sparse networks an order of magnitude larger than before                                                                                     | [doi:10.1007/s11590-013-0666-x](https://doi.org/10.1007/s11590-013-0666-x)                              | —    |
| **IP framework** — Veremyev, Prokopyev, Pasiliao                      | 2014 | J. Comb. Optim. 28:233–273                                             | Unified IP framework covering critical **nodes and edges** and many connectivity objectives                                                                                                  | [doi:10.1007/s10878-014-9730-4](https://doi.org/10.1007/s10878-014-9730-4)                              | —    |
| **DCNP (distance-based)** — Veremyev, Prokopyev, Pasiliao             | 2015 | Networks 66(3):170–195                                                 | Objective = # of node pairs within distance `D`, or Harary index / efficiency. The variant closest to a _diffusion_ objective without simulating one                                         | [doi:10.1002/net.21622](https://doi.org/10.1002/net.21622) — Wiley **403**                              | —    |
| **DCNP, efficient** — Alozie, Arulselvan, Akartunalı, Pasiliao        | 2021 | Computers & OR 131:105254                                              | MIP + BFS-tree separation algorithm for the distance-based variant                                                                                                                           | [doi:10.1016/j.cor.2021.105254](https://doi.org/10.1016/j.cor.2021.105254)                              | —    |
| **Parameterized complexity** — Hermelin, Kaspi, Komusiewicz, Navon    | 2016 | Theor. Comp. Sci. 651:62–75                                            | Critical Node Cut: given `k` removals, at most `x` connected pairs remain. Maps the FPT/W[1] boundary                                                                                        | [doi:10.1016/j.tcs.2016.08.016](https://doi.org/10.1016/j.tcs.2016.08.016)                              | —    |
| **CC-CNDP on PPI** — Boginski & Commander                             | 2008 | Clustering Challenges in Biological Networks, pp. 153–167              | First application of CNP to protein interaction networks / drug targeting                                                                                                                    | (book chapter; no stable free URL found)                                                                | —    |
| **CNDP survey** ⭐ — Lalou, Tahraoui, Kheddouci                        | 2018 | Computer Science Review 28:92–117                                      | **The** taxonomy of the OR branch: classifies every CNDP by connectivity metric and by solution method                                                                                       | [doi:10.1016/j.cosrev.2018.02.002](https://doi.org/10.1016/j.cosrev.2018.02.002)                        | —    |
| **Min–max node-targeted attacks** — Fortz, Mycek, Pióro, Tomaszewski  | 2024 | Networks 83(2):256–288                                                 | Compact IP with pseudo-components; two-level model                                                                                                                                           | [doi:10.1002/net.22200](https://doi.org/10.1002/net.22200) — Wiley **403**                              | —    |
| **Stochastic CNDP** — Bayarsaikhan, Chinchuluun, Arulselvan, Pardalos | 2025 | preprint                                                               | CNDP where edges exist probabilistically; heuristics for the stochastic objective                                                                                                            | [arXiv 2512.01497](https://arxiv.org/abs/2512.01497)                                                    | —    |

### 3.2 Metaheuristics

The benchmark culture here is `k` fixed per instance and the objective is **pairwise connectivity** (CNP-1a). Instance sets originate with Ventresca 2012.

| Method                                                                          | Year | Venue                                | Idea                                                                                                                                                                                | Paper                                                                                                                               | Code                 |
| ------------------------------------------------------------------------------- | ---- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| **PBIL + combinatorial SA** — Ventresca                                         | 2012 | Computers & OR 39(11):2763–2775      | Population-based incremental learning and simulated annealing over a _combinatorial unranking_ encoding of the removal set. **Ships the benchmark instance suite everybody reuses** | [doi:10.1016/j.cor.2012.02.008](https://doi.org/10.1016/j.cor.2012.02.008)                                                          | —                    |
| **Fast greedy (DFS)** — Ventresca & Aleman                                      | 2014 | COCOA                                | `O(\|V\|+\|E\|)` DFS-based objective re-evaluation, dropped into greedy                                                                                                             | Springer LNCS 8881 (no free URL found)                                                                                              | —                    |
| **Derandomized approximation** — Ventresca & Aleman                             | 2014 | Computers & OR 43:261–270            | LP relaxation + randomized rounding with a constant-factor bound                                                                                                                    | [doi:10.1016/j.cor.2013.09.012](https://doi.org/10.1016/j.cor.2013.09.012)                                                          | —                    |
| **Region growing** — Ventresca & Aleman                                         | 2014 | COCOA                                | LP-relaxation ball-growing, logarithmic approximation                                                                                                                               | Springer LNCS 8881                                                                                                                  | —                    |
| **Multi-objective EA** — Ventresca, Harrison, Ombuki-Berman                     | 2015 | EvoApplications                      | Maximize # components **and** minimize component-size variance simultaneously                                                                                                       | [doi:10.1007/978-3-319-16549-3_14](https://doi.org/10.1007/978-3-319-16549-3_14)                                                    | —                    |
| **GRASP + path relinking** — Pullan                                             | 2015 | J. Heuristics 21(5):577–598          | Greedy randomized adaptive search with evolutionary path relinking; beats VNS and SA on sparse real graphs                                                                          | [doi:10.1007/s10732-015-9290-5](https://doi.org/10.1007/s10732-015-9290-5)                                                          | —                    |
| **Hybrid constructive heuristics** — Addis, Aringhieri, Grosso, Hosteins        | 2016 | Annals of OR 238:637–649             | Alternate node _addition_ and _deletion_ greedy rules to escape local optima                                                                                                        | [doi:10.1007/s10479-016-2110-y](https://doi.org/10.1007/s10479-016-2110-y)                                                          | —                    |
| **ILS / VNS local search** — Aringhieri, Grosso, Hosteins, Scatamacchia         | 2016 | Networks 67(3):209–221               | Two cheap neighbourhoods + iterated local search / variable neighbourhood search                                                                                                    | [doi:10.1002/net.21671](https://doi.org/10.1002/net.21671) — Wiley **403**                                                          | —                    |
| **General evolutionary framework** — Aringhieri, Grosso, Hosteins, Scatamacchia | 2016 | Eng. Appl. AI 55:128–145             | One EA covering all three CNP objectives (pairwise conn / LCC / #components)                                                                                                        | [doi:10.1016/j.engappai.2016.06.010](https://doi.org/10.1016/j.engappai.2016.06.010)                                                | —                    |
| **MA-CNP (memetic)** — Zhou, Hao, Glover                                        | 2019 | IEEE T. Cybernetics 49(10):3699–3712 | Population + crossover + a component-based neighbourhood local search. **The strongest published metaheuristic on the CNP-1a benchmark**                                            | [arXiv 1705.04119](https://arxiv.org/abs/1705.04119) · [doi:10.1109/TCYB.2018.2848116](https://doi.org/10.1109/TCYB.2018.2848116)   | no public code found |
| **Incremental evaluation** — Zhou, Wang, Hao et al.                             | 2019 | preprint                             | `O(1)`-amortized objective updates inside CNP local search                                                                                                                          | [arXiv 1908.11846](https://arxiv.org/abs/1908.11846)                                                                                | no public code found |
| **MIQP (weighted)** — Chen, Jiang, Jiang, Zhang                                 | 2020 | Physica A 538:122862                 | Non-convex mixed-integer quadratic program for edge-weighted CNP                                                                                                                    | [doi:10.1016/j.physa.2019.122862](https://doi.org/10.1016/j.physa.2019.122862)                                                      | —                    |
| **NIPA** — Li, Liu, Yang                                                        | 2020 | Expert Syst. Appl. 139:112853        | Neighborhood-information-based probabilistic algorithm for network disintegration                                                                                                   | [arXiv 2003.04713](https://arxiv.org/abs/2003.04713) · [doi:10.1016/j.eswa.2019.112853](https://doi.org/10.1016/j.eswa.2019.112853) | no public code found |
| **Targeted enumeration** — Wang, Deng, Holme, Di, Lü, Wu                        | 2021 | preprint                             | Cost-effective disintegration by enumerating a small targeted candidate pool                                                                                                        | [arXiv 2111.02655](https://arxiv.org/abs/2111.02655)                                                                                | no public code found |
| **Multitasking EA (BICND)** — Zhang, Zhang, Feng, Yang, Cheng                   | 2024 | IEEE T. Cogn. Comm. Netw.            | Bi-objective CND on **interdependent** networks via evolutionary multitasking                                                                                                       | [doi:10.1109/TCCN.2024.3395174](https://doi.org/10.1109/TCCN.2024.3395174)                                                          | —                    |
| **K2GA** — Liu, Ge, Chen, Pei, Zhu, Mei                                         | 2024 | preprint                             | A pretrained neural net _initializes_ the GA population; cut-node greedy local search. 26 CNP-1a instances                                                                          | [arXiv 2402.00404](https://arxiv.org/abs/2402.00404)                                                                                | no public code found |
| **Hopfield NN for CND** — Michos, Neocleous, Papadopoulou Lesta                 | 2024 | SETN (Hellenic Conf. AI)             | CNP as Hopfield energy minimization                                                                                                                                                 | [doi:10.1145/3688671.3688762](https://doi.org/10.1145/3688671.3688762)                                                              | —                    |

### 3.3 Physics / percolation line (dismantling)

This is the branch that actually produced the strong baselines. All of these remove nodes **sequentially**, most of them add a **reinsertion** pass, and none of them fixes `k` in advance.

| Method                                                                       | Year | Venue                        | Idea                                                                                                                                                                                                                      | Paper                                                                                                                                                 | Code                                                                                                                                                  |
| ---------------------------------------------------------------------------- | ---- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Schneider's `R`** — Schneider, Moreira, Andrade, Havlin, Herrmann          | 2011 | PNAS 108(10):3838–3841       | Defines the robustness integral `R = (1/N)Σ s(q)` — the metric half this literature reports                                                                                                                               | [arXiv 1009.3125](https://arxiv.org/abs/1009.3125) · PNAS `10.1073/pnas.1009440108` (**403** to scripts)                                              | —                                                                                                                                                     |
| **Collective Influence (CI)** ⭐ — Morone & Makse                            | 2015 | **Nature** 524(7563):65–68   | Optimal percolation: minimize `λ_max` of the **non-backtracking matrix**. Score `CI_ℓ(i) = (d_i−1)·Σ_{j∈∂Ball(i,ℓ)}(d_j−1)`, adaptive removal + greedy reinsertion                                                        | [arXiv 1506.08326](https://arxiv.org/abs/1506.08326) · [doi:10.1038/nature14604](https://doi.org/10.1038/nature14604)                                 | [makselab/Collective-Influence](https://github.com/makselab/Collective-Influence) · third-party [zhfkt/ComplexCi](https://github.com/zhfkt/ComplexCi) |
| **CI at scale** — Morone, Min, Bo, Mari, Makse                               | 2016 | Sci. Rep. 6:30062            | `O(N log N)` CI; 2×10⁸-node ER graph in <2.5 h on one CPU                                                                                                                                                                 | [arXiv 1603.08273](https://arxiv.org/abs/1603.08273) · [doi:10.1038/srep30062](https://doi.org/10.1038/srep30062)                                     | as above                                                                                                                                              |
| **Min-Sum / decycling** ⭐ — Braunstein, Dall'Asta, Semerjian, Zdeborová     | 2016 | **PNAS** 113(44):12368–12373 | Three stages: 1RSB cavity Min-Sum message passing → near-minimal **decycling** set; `O(N log N)` tree breaking; **reverse-greedy reinsertion**. Proves `θ_dis ≤ θ_dec`                                                    | [arXiv 1603.08883](https://arxiv.org/abs/1603.08883) · PNAS `10.1073/pnas.1605083113` (**403**)                                                       | [abraunst/decycler](https://github.com/abraunst/decycler)                                                                                             |
| **BPD** — Mugisha & Zhou                                                     | 2016 | Phys. Rev. E 94:012305       | Belief-propagation-guided decimation over the minimum-feedback-vertex-set mapping, reweighting `x = 12`, then greedy reinsertion                                                                                          | [arXiv 1603.05781](https://arxiv.org/abs/1603.05781) · [doi:10.1103/PhysRevE.94.012305](https://doi.org/10.1103/PhysRevE.94.012305) (**403**)         | no verified public code (author page `power.itp.ac.cn` unreachable)                                                                                   |
| **CoreHD** ⭐ — Zdeborová, Zhang, Zhou                                       | 2016 | Sci. Rep. 6:37954            | Take the **2-core**, delete its highest-degree node, repeat; then reinsert. Two to four orders of magnitude faster than CI/BPD at comparable quality                                                                      | [arXiv 1607.03276](https://arxiv.org/abs/1607.03276) · [doi:10.1038/srep37954](https://doi.org/10.1038/srep37954)                                     | third-party [hcmidt/corehd](https://github.com/hcmidt/corehd)                                                                                         |
| **Explosive Immunization (EI)** — Clusella, Grassberger, Pérez-Reche, Politi | 2016 | Phys. Rev. Lett. 117:208301  | Inverse/Achlioptas construction: start fully vaccinated, un-vaccinate the weakest blocker from `m ≈ 10³` random candidates; two-regime score with recursive effective degree                                              | [arXiv 1604.00073](https://arxiv.org/abs/1604.00073) · [doi:10.1103/PhysRevLett.117.208301](https://doi.org/10.1103/PhysRevLett.117.208301) (**403**) | [pclus/explosive-immunization](https://github.com/pclus/explosive-immunization)                                                                       |
| **Articulation points** — Tian, Bashan, Shi, Liu                             | 2017 | Nat. Commun. 8:14223         | Analytic theory for the fraction of articulation points; the structural object that dismantling exploits                                                                                                                  | [arXiv 1609.00094](https://arxiv.org/abs/1609.00094) · [doi:10.1038/ncomms14223](https://doi.org/10.1038/ncomms14223)                                 | no public code found                                                                                                                                  |
| **BI / ABI** — Wandelt, Sun, Feng, Zanin, Havlin                             | 2018 | Sci. Rep. 8:13513            | The **benchmark study**: 13 competitors on 12 synthetic families + 20 real graphs. Finds _iterative betweenness_ (BI) best in 70–80% of cases, and approximate iterative betweenness (ABI) the best quality/time tradeoff | [doi:10.1038/s41598-018-31902-8](https://www.nature.com/articles/s41598-018-31902-8)                                                                  | —                                                                                                                                                     |
| **GND / GNDR** ⭐ — Ren, Gleinig, Helbing, Antulov-Fantulin                  | 2019 | **PNAS** 116(14):6554–6559   | **Generalized**: minimize total removal _cost_ `Σ_{i∈S} w_i`, not node count. Node-weighted spectral partitioning on a Power Laplacian + weighted-vertex-cover 2-approximation, recursed. GNDR adds reinsertion           | [arXiv 1801.01357](https://arxiv.org/abs/1801.01357) · PNAS `10.1073/pnas.1806108116` (**403**)                                                       | [renxiaolong/Generalized-Network-Dismantling](https://github.com/renxiaolong/Generalized-Network-Dismantling)                                         |
| **EGND** (ensemble GND)                                                      | 2019 | —                            | Ensemble of GND cuts; listed as a distinct baseline in GDM and Artime                                                                                                                                                     | —                                                                                                                                                     | —                                                                                                                                                     |
| **BPHD** — Peng, Fan, Lü                                                     | 2024 | preprint                     | Dismantling that must destroy _higher-order_ structures (k-cores) in every branch, not just connectivity; belief propagation over **edge** removal                                                                        | [arXiv 2401.10028](https://arxiv.org/abs/2401.10028)                                                                                                  | no public code found                                                                                                                                  |
| **Embedding-aided dismantling** — Osat, Papadopoulos, Teixeira, Radicchi     | 2022 | preprint                     | Embed into Euclidean/hyperbolic space (Mercator), then remove by geometric rule                                                                                                                                           | [arXiv 2208.01087](https://arxiv.org/abs/2208.01087)                                                                                                  | uses [networkgeometry/mercator](https://github.com/networkgeometry/mercator)                                                                          |
| **Community-based dismantling landscape**                                    | 2022 | preprint                     | Shows many _node-disjoint_ but equally effective dismantling sets exist — useful when removal cost varies                                                                                                                 | [arXiv 2209.14077](https://arxiv.org/abs/2209.14077)                                                                                                  | —                                                                                                                                                     |
| **Directed dismantling (TAD)** — Liu, Hu, Wang, Liu, Zhang                   | 2025 | preprint                     | Targets the giant **strongly** connected component; trophic-incoherence centrality. The only directed-native dismantling method found                                                                                     | [arXiv 2512.11416](https://arxiv.org/abs/2512.11416)                                                                                                  | no public code found                                                                                                                                  |

**Surveys of this branch:** Lü, Chen, Ren, Zhang, Zhang, Zhou, _Vital nodes identification in complex networks_, Physics Reports 650:1–63 (2016) — [arXiv 1607.01134](https://arxiv.org/abs/1607.01134) · Artime, Grassia, De Domenico, Gleeson, Makse, Mangioni, Perc, Radicchi, _Robustness and resilience of complex networks_, Nature Reviews Physics 6:114–131 (2024) — [arXiv 2509.19867](https://arxiv.org/abs/2509.19867) · [nature.com](https://www.nature.com/articles/s42254-023-00676-y) · **runnable baseline suite** at [NetworkDismantling/review](https://github.com/NetworkDismantling/review) · Chen et al., _Critical Nodes Identification in Complex Networks: A Survey_ (2025) — [arXiv 2507.06164](https://arxiv.org/abs/2507.06164) (**seed paper 1**) · Wandelt et al., _Recent advances in network dismantling_, Chaos Solitons & Fractals (2025) — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0960077925006861) (**403** to scripts; page exists, paywalled).

---

## 4. Learning-based methods

The defining trick of this branch — and the one most relevant to us — is **train on tiny synthetic graphs, apply zero-shot to million-node real graphs**. FINDER trains on BA graphs of 30–50 nodes. NIRM trains on 20–30-node graphs where the optimal removal set can be brute-forced. MIND trains on 10,000 synthetic graphs of 100–200 nodes. This is exactly the generalization claim ToupleGDD makes on the IM side, and it is why our BA-100 training regime is defensible.

| Method                                                         | Year | Venue                                        | Approach                                                                                                                                                                                                                                                                                                                   | Paper                                                                                                                                                                                         | Code                                                                                                                                   |
| -------------------------------------------------------------- | ---- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **FINDER** ⭐ — Fan, Zeng, Sun, Liu                             | 2020 | **Nature Machine Intelligence** 2(6):317–324 | DQN over `structure2vec` embeddings; reward = ANC reduction; trained on BA graphs of **30–50 nodes**, applied to million-node graphs. Handles CN (pairwise connectivity) _and_ ND (GCC) _and_ node-weighted costs — four trained variants                                                                                  | [nature.com](https://www.nature.com/articles/s42256-020-0177-2) (paywalled) · free full text at [PMC8191335](https://pmc.ncbi.nlm.nih.gov/articles/PMC8191335/) · **no arXiv version exists** | [FFrankyy/FINDER](https://github.com/FFrankyy/FINDER) (200) · mirror [fanchangjun/FINDER](https://github.com/fanchangjun/FINDER) (200) |
| **GDM** ⭐ — Grassia, De Domenico, Mangioni                     | 2021 | **Nature Communications** 12:5190            | Geometric deep learning, **supervised** (not RL) on brute-force-optimal dismantling of small graphs. _Static_: one scoring pass, no re-computation. Optional reinsertion (`GDM+R`)                                                                                                                                         | [arXiv 2101.02453](https://arxiv.org/abs/2101.02453) · [nature.com](https://www.nature.com/articles/s41467-021-25485-8) (open access)                                                         | [NetworkScienceLab/GDM](https://github.com/NetworkScienceLab/GDM)                                                                      |
| **NIRM** — Zhang & Wang                                        | 2022 | CIKM                                         | Neural Influence Ranking Model: GAT encoder fusing local structural + global topological scores, supervised on brute-forced optimal target-attack-sets of **20–30-node** synthetic graphs                                                                                                                                  | [arXiv 2208.07792](https://arxiv.org/abs/2208.07792)                                                                                                                                          | [JiazhengZhang/NIRM](https://github.com/JiazhengZhang/NIRM)                                                                            |
| **DCRS** — Zhang & Wang                                        | 2023 | WWW                                          | Direct NIRM follow-up: encodes node **diffusion competence** _and_ **role significance** rather than pure topology. Best on 21 of 22 networks                                                                                                                                                                              | [arXiv 2301.12349](https://arxiv.org/abs/2301.12349)                                                                                                                                          | [JiazhengZhang/DCRS](https://github.com/JiazhengZhang/DCRS)                                                                            |
| **CoreGDM** — Grassia & Mangioni                               | 2023 | Complex Networks XIV (Springer)              | GDM ∘ CoreHD: run the learned score **inside the 2-core** instead of degree                                                                                                                                                                                                                                                | [doi:10.1007/978-3-031-28276-8_8](https://doi.org/10.1007/978-3-031-28276-8_8) (paywalled)                                                                                                    | no public code found                                                                                                                   |
| **Hypernetwork dismantling (DRL)** — Yan, Xie, Zhang, He, Yang | 2021 | IEEE TNSE                                    | DRL over node sequences with inductive **hypernetwork** embedding                                                                                                                                                                                                                                                          | [arXiv 2104.14332](https://arxiv.org/abs/2104.14332)                                                                                                                                          | no public code found                                                                                                                   |
| **HyperCI** — Yan et al.                                       | 2021 | —                                            | Higher-order Collective Influence for hypernetworks (non-learned baseline in the same line)                                                                                                                                                                                                                                | [arXiv 2103.06117](https://arxiv.org/abs/2103.06117)                                                                                                                                          | no public code found                                                                                                                   |
| **EML** — Rezaei, Munoz, Jalili, Khayyam                       | 2022 | Expert Syst. Appl.                           | SVR-RBF regression of node vitality from structural features, **SIR-simulation ground truth** — the only entry here whose target is a diffusion outcome rather than connectivity                                                                                                                                           | [arXiv 2202.06229](https://arxiv.org/abs/2202.06229)                                                                                                                                          | no public code found                                                                                                                   |
| **Feature-importance-aware GAT** — Tan, Zhou, Zhou, Fu         | 2024 | IEEE T. Autom. Sci. Eng.                     | Feature-sensitive GAT + adversarial double-DQN for sparse-graph CND                                                                                                                                                                                                                                                        | [doi:10.1109/TASE.2023.3286860](https://doi.org/10.1109/TASE.2023.3286860)                                                                                                                    | no public code found                                                                                                                   |
| **AdaRisk** — Li, Xu, Cheng, Wang                              | 2024 | IEEE TKDE                                    | Risk-adaptive deep RL for fragile-node detection in **uncertain** graphs                                                                                                                                                                                                                                                   | [doi:10.1109/TKDE.2023.3348493](https://doi.org/10.1109/TKDE.2023.3348493)                                                                                                                    | no public code found                                                                                                                   |
| **K2GA** — Liu, Ge, Chen, Pei, Zhu, Mei                        | 2024 | preprint                                     | Neural-net-initialized genetic algorithm for CNP-1a (also in §3.2 — it straddles)                                                                                                                                                                                                                                          | [arXiv 2402.00404](https://arxiv.org/abs/2402.00404)                                                                                                                                          | no public code found                                                                                                                   |
| **ESND** — Xie, Liu, Li, Zhan, Li                              | 2024 | preprint                                     | Embedding-based **signed** network dismantling                                                                                                                                                                                                                                                                             | [arXiv 2406.08899](https://arxiv.org/abs/2406.08899)                                                                                                                                          | no public code found                                                                                                                   |
| **MultiDismantler**                                            | 2025 | Nature Machine Intelligence                  | Deep RL + multiplex representation for **interdependent / multilayer** dismantling                                                                                                                                                                                                                                         | [nature.com](https://www.nature.com/articles/s42256-025-01070-2) (paywalled)                                                                                                                  | not located                                                                                                                            |
| **Symbolized RL (Selinda)** — Zheng, Ding, Jin, Gao, Li        | 2025 | preprint                                     | Learn the RL attack policy, then **symbolic-regress it into a closed-form resilience law** coupling topology + dynamics. Directly relevant to a coding-agent framing: the agent's output _is_ a formula                                                                                                                    | [arXiv 2507.08827](https://arxiv.org/abs/2507.08827)                                                                                                                                          | [tsinghua-fib-lab/selinda](https://github.com/tsinghua-fib-lab/selinda)                                                                |
| **MIND** ⭐ — Tian, Ferraro, Shorten, Jalili, Hamedmoghadam     | 2026 | **AAAI-26**                                  | Drops handcrafted structural input features entirely: all-to-one attention (MIND-AM) + message-iteration profiles (MIND-MP), trained with **SAC** over 8M dismantling episodes on 10,000 synthetic graphs. `O(\|V\|+\|E\|)`. **Current SOTA and the only paper with FINDER + GDM + a 2026 method on one 47-network table** | [arXiv 2508.00706](https://arxiv.org/abs/2508.00706)                                                                                                                                          | [HaozheTian/MIND-ND](https://github.com/HaozheTian/MIND-ND)                                                                            |
| **SPR / HoGNN** ⭐ — Zhou, Tan, Fang, Lü, Zhao                  | 2026 | Communications Physics 9:181                 | Higher-order GNN dismantling. **Beats FINDER, GDM, NIRM, DCRS in one printed table** (§5.4)                                                                                                                                                                                                                                | [nature.com](https://www.nature.com/articles/s42005-026-02601-y) (open access)                                                                                                                | [zhouwn/spr](https://github.com/zhouwn/spr)                                                                                            |
| **hyper-VDrank** — Hao, Liu, Wang, Han, Zheng, Tang            | 2026 | preprint                                     | Hypergraph dismantling under the **strong-deletion** rule (removing a node kills the whole hyperedge)                                                                                                                                                                                                                      | [arXiv 2606.23289](https://arxiv.org/abs/2606.23289)                                                                                                                                          | no public code found                                                                                                                   |

### 4.1 The two seed papers

- **[arXiv 2507.06164](https://arxiv.org/abs/2507.06164)** — Chen, Chen, Zhang, Jia, Liu, Sun, Lü, Yu, _Critical Nodes Identification in Complex Networks: A Survey_ (Southeast Univ. + USTC, v2 Sep 2025). Not a method — a **seven-category taxonomy**: centrality, critical-node-deletion problem, influence maximization, network control, AI, higher-order, dynamic. Its §5.2/§5.3 reproduce the CC-CNDP and Di Summa ILPs verbatim (transcribed as eqs. 19–20 there), and its Table 10 is the method-name index this file's §3–§4 expands. Co-authored by **Linyuan Lü**, which makes it the ML-era successor to the 2016 Physics Reports survey. Also mirrored at [oaepublish.com/articles/ces.2025.34](https://www.oaepublish.com/articles/ces.2025.34).
- **[arXiv 2409.15142v1](https://arxiv.org/abs/2409.15142)** — Farahi, Kamandi, Abedian, Rocha, _Critical Node Detection in Temporal Social Networks, Based on Global and Semi-local Centrality Measures_. Proposes three **temporal** centralities (temporal supra-cycle ratio TSCR, temporal semi-local integration TSLI, temporal semi-local centrality TSLC) and evaluates them three ways on SocioPatterns face-to-face contact networks: isolate the critical nodes during an epidemic, seed the epidemic from them, or remove them and measure percolation. **This is the one paper in the whole file whose evaluation is a diffusion process rather than a connectivity functional** — i.e. the closest existing work to the variant §2.2 says we should build. No public code found.

---

## 5. Published results

Read §8.2 before comparing anything across subsections. Three incompatible metric conventions are in play: **absolute set size** (§5.1), **relative AUC** (§5.2–§5.3, §5.7), and **ρ = fraction removed at threshold Θ** (§5.4–§5.6).

### 5.1 ⭐ CoreHD / BPD — 12 real networks, absolute dismantling-set sizes

**The most directly comparable published result in this file**, because row 3 (`Grid`, 4,941 / 6,594) is byte-for-byte the graph our `power_grid` loader produces. Dismantling target: largest component `< 0.01 N`. All values are **numbers of nodes removed**; lower is better.

CoreHD Table I [verified, arXiv 1607.03276]:

| Network     | N         | M          | decycling CoreHD | decycling BPD | dismantling CI | dismantling CoreHD | dismantling BPD | time CI (s) | time CoreHD (s) | time BPD (s) |
| ----------- | --------- | ---------- | ---------------- | ------------- | -------------- | ------------------ | --------------- | ----------- | --------------- | ------------ |
| RoadEU      | 1,177     | 1,417      | 90               | 91            | 209            | 148                | 152             | 0.18        | < 0.001         | 0.1          |
| PPI         | 2,361     | 6,646      | 365              | 362           | 424            | 357                | 350             | 0.91        | < 0.001         | 2.09         |
| **Grid** ✅ | **4,941** | **6,594**  | 519              | 512           | 476            | **327**            | **320**         | 1.00        | < 0.001         | 0.66         |
| IntNet1     | 6,474     | 12,572     | 217              | 215           | 198            | 156                | 161             | 5.19        | < 0.001         | 11.32        |
| Authors     | 23,133    | 93,439     | 8,311            | 8,317         | 3,588          | 2,600              | 2,583           | 87.55       | 0.09            | 40.04        |
| Citation    | 34,546    | 420,877    | 15,489           | 15,390        | 14,518         | 13,523             | 13,454          | 4,166       | 0.2             | 383.91       |
| P2P         | 62,586    | 147,892    | 9,557            | 9,285         | 10,726         | 9,561              | 9,292           | 520.59      | 0.21            | 50.24        |
| Friend      | 196,591   | 950,327    | 38,911           | 38,831        | 32,340         | 27,148             | 26,696          | 5,361       | 1.37            | 588.19       |
| Email       | 265,214   | 364,481    | 1,189            | 1,186         | 21,465         | 1,070              | 1,064           | 6,678       | 0.39            | 151.57       |
| WebPage     | 875,713   | 4,322,051  | 208,509          | 208,641       | 106,750        | 51,603             | 50,878          | 2,275       | 9.67            | 2,532        |
| RoadTX      | 1,379,917 | 1,921,660  | 243,969          | 239,885       | 133,763        | 20,289             | 20,676          | 273.69      | 4.07            | 421.15       |
| IntNet2     | 1,696,415 | 11,095,298 | 229,034          | 228,720       | 144,160        | 73,601             | 73,229          | 19,715      | 35.84           | 4,243        |

BPD's own Table I [verified, arXiv 1603.05781] reports the same 12 networks with identical `N`/`M` and identical CI/BPD dismantling columns, plus feedback-vertex-set sizes. The one visible transcription oddity is BPD's `Email` row, which prints TAS = 1,064 and FVS = 1,186 (the two columns appear swapped relative to every other row) — treat that cell as suspect.

**As fractions of `N`** [derived] on `Grid` = our `power_grid`: CI **9.63%**, CoreHD **6.62%**, BPD **6.48%**. Every one of these includes a reinsertion pass. Compare to NIRM's `UsPower` ρ of **8.58%** adaptive / **16.81%** one-pass (§5.6) — same graph, same `Θ = 0.01`, and a 2.6× spread purely from protocol. This is the cleanest single illustration of §8.2 trap 1 and 2 in the literature.

**Network identities** [verified, BPD Table I caption]: RoadEU = European express road network · RoadTX = Texas road network · Grid = western-US power grid · IntNet1/IntNet2 = Internet at AS level · WebPage = Google web graph · Email = European email network · Citation = citation network · PPI = protein–protein interaction · Authors = condensed-matter coauthorship · P2P = peer-to-peer · Friend = online friendship.

### 5.2 ⭐ MIND (AAAI-26) — relative AUC across 47 real networks

The only published table that puts **FINDER, GDM, GND, Min-Sum, EI and adaptive degree on one metric across a large, domain-stratified network set**. AUC of the dismantling curve, **normalized so MIND = 100.0 on every network**; lower is better. `nan` = method did not complete.

Overall row [verified, MIND Table 5, arXiv 2508.00706]:

|                           | AD    | BC    | PR    | MS    | EI    | GND   | FINDER    | GDM       | MIND      |
| ------------------------- | ----- | ----- | ----- | ----- | ----- | ----- | --------- | --------- | --------- |
| **Overall (47 networks)** | 119.9 | 167.8 | 129.5 | 142.8 | 108.0 | 109.1 | **115.0** | **104.2** | **100.0** |

Selected per-network rows [verified, same table] — chosen to show where the ranking inverts:

| Network            | AD    | BC    | PR    | MS    | EI       | GND   | FINDER | GDM   | MIND  |
| ------------------ | ----- | ----- | ----- | ----- | -------- | ----- | ------ | ----- | ----- |
| petster-hamster    | 128.4 | 129.6 | 141.4 | 174.5 | 108.7    | 96.8  | 129.2  | 104.6 | 100.0 |
| soc-Epinions1      | 100.7 | 120.3 | 104.1 | 117.4 | 115.6    | 99.6  | 97.0   | 99.8  | 100.0 |
| com-dblp           | 114.3 | 121.0 | 109.5 | 180.0 | 88.5     | 106.9 | 115.6  | 97.5  | 100.0 |
| web-Stanford       | 209.8 | 239.9 | 173.3 | nan   | 65.1     | 114.8 | 207.2  | 98.9  | 100.0 |
| linux              | 128.6 | 404.0 | 194.7 | 166.0 | 93.5     | 108.3 | 124.2  | 110.7 | 100.0 |
| eu-powergrid       | 151.5 | 190.5 | 196.5 | 317.5 | 80.6     | 82.8  | 161.7  | 109.1 | 100.0 |
| roads-california   | 192.8 | 882.0 | 125.1 | 114.9 | **23.4** | 78.0  | 116.3  | 92.2  | 100.0 |
| roads-sanfrancisco | 184.5 | 51.6  | 258.8 | 160.4 | **15.5** | 30.0  | 176.4  | 92.0  | 100.0 |
| opsahl-openflights | 135.2 | 131.3 | 130.9 | 167.1 | 116.8    | 107.6 | 120.5  | 106.2 | 100.0 |

**Three readings that matter to us.**

1. **FINDER (2020) beats adaptive degree by 4.9 points overall** (115.0 vs 119.9) and _loses_ to it on `soc-Epinions1` (97.0 vs 100.7 — both better than MIND there). A learned dismantler is not obviously better than recomputing the degree. This is the CND analogue of our BA-100 degree-triviality finding, and it is published, not folklore.
2. **On road and grid networks the learned methods collapse.** `roads-california`: EI 23.4, GND 78.0, GDM 92.2 — but FINDER 116.3, i.e. _worse than MIND and worse than the physics heuristics by 5×_. Mesh-like, low-degree, high-diameter graphs are where message-passing receptive fields run out. Our `power_grid` (diameter 46) is exactly this class. Expect a learned model to underperform there and say so up front.
3. **MIND's complexity table** [verified, MIND Table 1]: MS `O(|V|log|V|)+O(|E|)`, EI `O(|V|log|V|)`, GND `O(|V|log^{2+ε}|V|)`, FINDER `O(|V|log|V|+|E|)`, GDM `O(|V|⟨d²⟩+|E|)`, MIND `O(|V|+|E|)`. Every one of these is at most near-linear — reinforcing §2.1 objection 2 that there is no expensive simulator here to amortize away.

MIND's teaser figure (FilmTrust, 610 nodes) [verified, Fig. 1]: FINDER **37.40**, GDM **32.80**, MIND **29.86**.

MIND also reports that GDM's dismantling order correlates at Spearman **R = 0.762** with a PCA of its own handcrafted input features, versus **0.349** for MIND [verified, Fig. 4] — i.e. GDM is substantially reproducing its inputs rather than learning new structure. Worth remembering before we hand a GNN `CH_DEGREE` and claim the result is learned.

### 5.3 GDM (Nature Communications 2021) — relative AUC, GDM = 100

AUC of `LCC(x)/|N|` by Simpson's rule, target 10% of network size, scaled so **GDM = 100.0**; lower is better. Averages over 45 real networks [verified, GDM Table 1, arXiv 2101.02453]:

| GDM   | GND   | EGND  | Adaptive degree | EI σ1 | Pagerank | Degree | Betweenness | MS    | EI σ2  | GDM+R    | GND+R | CoreHD | MS+R  | CI ℓ=2 |
| ----- | ----- | ----- | --------------- | ----- | -------- | ------ | ----------- | ----- | ------ | -------- | ----- | ------ | ----- | ------ |
| 100.0 | 111.7 | 115.4 | 119.1           | 123.6 | 128.7    | 151.1  | 158.9       | 273.3 | 2596.4 | **91.5** | 100.8 | 106.9  | 108.7 | 112.1  |

12 large networks [verified, GDM Table 2]:

| GDM   | GND   | MS    | GDM+R    | GND+R | MS+R  | CoreHD |
| ----- | ----- | ----- | -------- | ----- | ----- | ------ |
| 100.0 | 105.6 | 148.8 | **95.3** | 102.6 | 114.1 | 114.1  |

Runtime [verified, GDM Table 3, selected]: GDM's _prediction_ pass is 1.3–13.5 s versus CoreHD 6.1–40.6 s; end-to-end dismantling of `com-dblp` takes GDM **00:22:30.7** vs GND **04:57:25.6**.

Note again the normalizer: GDM's "GND = 111.7" and Artime's "GDM grand average 158.3" (§5.7) are not in conflict — different denominators.

### 5.4 ⭐ SPR / HoGNN (Commun. Phys. 2026) — ρ at Θ_GCC = 0.01, absolute

The **only printed table** found that has FINDER, GDM, NIRM and DCRS side by side with absolute (non-self-normalized) numbers, and it includes `NetScience` and `Crime`. `ρ` = fraction of nodes removed to reach GCC ≤ 1%; lower is better [verified, HoGNN Table 1].

| Method      | FilmTrust | RoviraVirgili | DNCEmails | Figeys   | PPI       | NetScience | Crime     | KKI       | EconPoli |
| ----------- | --------- | ------------- | --------- | -------- | --------- | ---------- | --------- | --------- | -------- |
| DC          | 22.77     | 48.46         | 8.09      | 18.89    | 27.34     | 16.08      | 25.21     | 32.44     | 6.37     |
| BC          | 33.75     | 54.55         | 15.7      | 18.89    | 35.70     | 30.05      | 36.79     | 24.71     | 14.95    |
| CC          | 70.02     | 74.32         | 90.14     | 50.2     | 55.13     | 29.77      | 65.02     | 69.88     | 60.70    |
| EC          | 89.93     | 79.88         | 90.25     | 56.90    | 60.52     | 98.29      | 75.51     | 74.71     | 68.31    |
| HC          | 70.02     | 72.90         | 88.10     | 51.14    | 50.31     | 29.64      | 63.69     | 61.53     | 59.03    |
| PR          | 22.20     | 44.40         | 6.16      | 16.03    | 24.19     | 19.71      | 27.26     | 22.03     | 5.99     |
| DR          | 22.08     | 45.98         | 6.16      | 10.50    | 22.66     | 64.20      | 28.83     | 76.76     | 8.91     |
| CI          | 42.68     | 59.31         | 30.33     | 25.90    | 34.67     | 28.95      | 42.82     | 50.58     | 5.64     |
| GND         | **13.42** | 42.63         | 7.40      | 11.30    | 22.98     | 5.82       | 27.74     | 36.33     | 8.06     |
| VE          | 29.63     | 52.43         | 15.43     | 20.05    | 36.33     | 17.73      | 48.13     | 40.30     | 5.89     |
| DW          | 89.13     | 95.23         | 27.60     | 36.67    | 80.98     | 82.14      | 82.37     | 80.42     | 69.86    |
| NV          | 64.87     | 94.88         | 31.62     | 86.02    | 41.32     | 68.45      | 91.54     | 84.89     | 62.37    |
| RV          | 88.67     | 97.26         | 97.75     | 94.60    | 97.12     | 98.88      | 93.56     | 89.36     | 74.35    |
| GCN         | 98.74     | 98.94         | 7.5       | 30.64    | 98.88     | 68.86      | 82.75     | 78.06     | 34.21    |
| GAT         | 81.69     | 86.41         | 74.12     | 54.27    | 61.38     | 72.21      | 84.80     | 90.79     | 11.03    |
| **FINDER**  | 16.82     | 41.65         | 5.52      | 11.75    | 27.29     | 11.57      | 25.09     | 20.20     | 63.55    |
| **GDM**     | 14.87     | 42.36         | 5.79      | 9.30     | 22.48     | 6.84       | 25.33     | 18.77     | 5.96     |
| **NIRM**    | 44.39     | 45.01         | 6.38      | 39.53    | 25.49     | 8.28       | 25.45     | 38.78     | 8.33     |
| NEES        | 25.97     | 52.34         | 10.45     | 10.05    | 24.82     | 11.49      | 25.81     | 18.50     | 14.95    |
| **DCRS**    | 19.11     | 43.07         | 5.89      | 8.93     | 21.67     | 8.42       | 26.06     | 22.34     | 6.58     |
| **SPR**     | 14.30     | **40.86**     | **5.36**  | **8.71** | **21.31** | **5.75**   | **23.88** | **16.26** | **4.99** |
| Improv. (%) | −2.31     | 1.90          | 2.90      | 2.46     | 1.66      | 1.20       | 4.82      | 12.11     | 11.52    |

**Two things to take from this.** First, on `NetScience` **plain GND (a 2019 spectral heuristic) scores 5.82, beating every learned method except SPR's 5.75** — a 1.2% margin after six years. Second, **FINDER catastrophically fails on `EconPoli` (63.55 vs GDM's 5.96)**, a 10× miss on one network out of nine. The variance of learned dismantlers across graph families is much larger than their mean advantage.

### 5.5 DCRS (WWW 2023) — ρ at Θ = 0.01 across 22 networks

[verified, DCRS Table 2, arXiv 2301.12349]. Includes **Ca-GrQc** ✅ and **HM**. Columns abbreviated; `DW`/`NV`/`RV` are embedding baselines, `NEES` is the predecessor.

| Dataset        | DC    | BC    | CI    | PR        | GAT   | GCN   | NIRM  | NEES  | **DCRS**  | Impr.% |
| -------------- | ----- | ----- | ----- | --------- | ----- | ----- | ----- | ----- | --------- | ------ |
| Chicago        | 50.8  | 55.58 | 60.42 | 51.56     | 77.73 | 76.65 | 46.58 | 42.15 | **35.3**  | +16.25 |
| Europe         | 41.29 | 48.99 | 82.19 | 33.69     | 87.78 | 89.51 | 38.11 | 28.1  | **23.39** | +16.76 |
| AirTraffic     | 32.79 | 50.98 | 68.03 | 28.14     | 76.59 | 98.04 | 25.61 | 26.43 | **23.57** | +7.97  |
| Gnutella       | 36.64 | 38.35 | 40.35 | 35.07     | 63.78 | 98.82 | 35.03 | 43.32 | **32.45** | +7.37  |
| Route          | 4.06  | 4.59  | 6.43  | 4.49      | 60.32 | 4.46  | 55.92 | 4.79  | **3.43**  | +15.52 |
| Blog           | 45.34 | 44.12 | 52.04 | 44.53     | 89.05 | 98.94 | 55.88 | 46.73 | **37.09** | +15.93 |
| FilmTrust      | 22.77 | 33.75 | 42.68 | 22.20     | 81.69 | 98.74 | 44.39 | 25.97 | **19.11** | +13.92 |
| LastFM         | 31.69 | 40.11 | 48.6  | 27.96     | 61.23 | 98.95 | 70.96 | 31.47 | **26.02** | +6.94  |
| Flickr         | 51.76 | 45.68 | 51.14 | 48.2      | 81.33 | 99    | 57.25 | 50.27 | **42.57** | +6.81  |
| BlogCatalog    | 90.22 | 82.79 | 97.83 | 84.47     | 97.13 | 99    | 97.84 | 84.91 | **80.6**  | +2.65  |
| HM             | 40.53 | 40.64 | 57.91 | 33.48     | 87.62 | 98.98 | 34.98 | 39.77 | **31.38** | +6.27  |
| RoviraVirgili  | 48.46 | 54.55 | 59.31 | 44.40     | 86.41 | 98.94 | 45.01 | 52.34 | **43.07** | +3.00  |
| DNCEmails      | 8.09  | 15.7  | 30.33 | 6.16      | 74.12 | 7.5   | 6.38  | 10.45 | **5.89**  | +4.38  |
| HI-II-14       | 17.53 | 25.74 | 32.73 | 16.59     | 69.48 | 98.99 | 17.07 | 16.81 | **13.97** | +15.79 |
| Vidal          | 20.84 | 23.46 | 31.6  | 20.14     | 54.23 | 98.82 | 31.38 | 17.3  | **16.02** | +7.4   |
| Figeys         | 18.89 | 18.89 | 25.9  | 16.03     | 54.27 | 30.64 | 39.53 | 10.05 | **8.93**  | +11.14 |
| PPI            | 27.34 | 35.7  | 34.67 | 24.19     | 61.38 | 98.88 | 25.49 | 24.82 | **21.67** | +10.42 |
| Genefusion     | 19.24 | 21.65 | 82.82 | 13.4      | 76.63 | 98.28 | 13.75 | 19.59 | **11.34** | +15.37 |
| Bible          | 30.46 | 32.43 | 53.86 | 30.23     | 84.26 | 98.98 | 27.35 | 35.25 | **26.11** | +4.53  |
| Wikibook       | 14.65 | 15.37 | 33.27 | **11.75** | 43.22 | 16.46 | 67.63 | 13.74 | 11.93     | −1.51  |
| **Ca-GrQc** ✅ | 25.44 | 24.63 | 39.56 | 19.1      | 73.02 | 98.8  | 24.29 | 19.17 | **17.58** | +7.96  |
| UAI            | 46.2  | 55    | 54.22 | 42.13     | 86.04 | 98.99 | 75.48 | 50.77 | **40.04** | +4.96  |

Note the **`GCN` and `GAT` rows**: generic graph networks trained on the same signal score 43–99 where the specialized methods score 4–45. Architecture alone does not solve this task, which is consistent with §2.1 objection 1.

### 5.6 NIRM (CIKM 2022) — adaptive vs one-pass on `UsPower` ✅

Per-network comparison against FINDER/GND/CoreHD/BPD/CI/DC/NSKSD is published only as heatmaps (Figs. 3–4) — **no numeric table** [figure]. What _is_ printed:

Ablation Table 3, ρ (%), (A) adaptive / (B) one-pass [verified, arXiv 2208.07792]:

| Method   | UsPower ✅       | P-H               | Ca-GrQc ✅        | Infectious        | Bible             | HM                |
| -------- | ---------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| NIRM-IS  | 12.37 / 18.94    | 53.95 / 86.80     | 32.23 / 35.95     | 65.61 / 79.02     | 41.06 / 31.75     | 26.43 / 48.44     |
| NIRM-GS  | 67.86 / 98.91    | 89.35 / 96.45     | 74.43 / 97.55     | 96.83 / 98.78     | 83.42 / 94.47     | 89.24 / 97.42     |
| NIRM-LS  | 9.05 / 18.32     | 27.55 / 49.15     | 13.32 / 24.63     | 55.85 / 81.71     | 20.47 / 28.26     | 24.17 / 38.64     |
| **NIRM** | **8.58 / 16.81** | **25.70 / 47.90** | **11.98 / 24.29** | **54.88 / 76.83** | **19.35 / 27.35** | **22.60 / 34.98** |

Prose results [verified, same paper]: average ρ over 15 networks — NIRM **24.76%** adaptive vs second-best NSKSD **28.81%**; one-pass NIRM **36.58%** vs second-best **40.41%**. AUC on HM / Roget: NIRM **248.27 / 250.84** vs FINDER **259.81**, CI **253.56**.

### 5.7 Artime et al. (Nature Reviews Physics 2024) — the benchmark survey

AUC at a 10% removal target, **scaled so the best method on each network = 100.0** (different normalizer again). Domain summaries and grand average [verified, arXiv 2509.19867 Table 2]:

| Domain            | AD    | BC    | EI σ1     | GDM   | GND   | FINDER | PR    | MS    | CI ℓ=2    | CoreHD | GDM+R     | GND+R | MS+R  |
| ----------------- | ----- | ----- | --------- | ----- | ----- | ------ | ----- | ----- | --------- | ------ | --------- | ----- | ----- |
| Biological        | 114.3 | 143.0 | 125.6     | 109.3 | 123.1 | 110.8  | 129.8 | 133.9 | **101.0** | 116.7  | 103.8     | 113.7 | 117.3 |
| Information       | 162.9 | 255.4 | 150.2     | 127.2 | 147.3 | 193.7  | 173.1 | 647.8 | 262.7     | 121.0  | **110.7** | 116.8 | 126.8 |
| Technological     | 545.0 | 982.2 | **125.6** | 278.6 | 222.8 | 492.0  | 514.7 | 557.3 | 338.7     | 134.5  | 127.4     | 138.6 | 134.5 |
| **Grand average** | 240.5 | 390.0 | 137.6     | 158.3 | 153.0 | 233.2  | 240.4 | 399.7 | 207.2     | 126.6  | **112.6** | 121.2 | 127.9 |

Its stated conclusion, verbatim: _"there is **no one-size-fits-all algorithm**… the best algorithm varies within and across domains."_ Its algorithm-property table [verified, Table 1] is the cleanest reference for the reinsertion and static/adaptive confounds:

| Algorithm   | Type                      | Static  | Reinsertion | Complexity                         |
| ----------- | ------------------------- | ------- | ----------- | ---------------------------------- |
| CI          | influence maximization    | No      | Yes         | `O(\|V\| log\|V\|)`                |
| BPD         | message-passing decycling | No      | No          | `O(\|E\|T)`                        |
| Min-Sum     | message-passing decycling | No      | Yes         | `O(\|E\|T) + O(\|V\|(log\|V\|+T))` |
| GND         | spectral partitioning     | No      | Optional    | `O(\|V\| log^{2+ε}\|V\|)`          |
| EGND        | spectral partitioning     | No      | Optional    | `O(e)·O(GND)`                      |
| CoreHD      | degree-based decycling    | No      | Yes         | `O(\|V\|)` sparse                  |
| EI          | explosive percolation     | No      | No          | `O(\|V\| log\|V\|)`                |
| **GDM**     | **machine learning**      | **Yes** | Optional    | `O(h(\|V\|+\|E\|))`                |
| **CoreGDM** | **machine learning**      | **Yes** | **Yes**     | `O(h(\|V\|+\|E\|))`                |
| **FINDER**  | **machine learning**      | No      | Optional    | `O(\|E\|+\|V\|(1+log\|V\|))`       |

**Runnable baselines:** [NetworkDismantling/review](https://github.com/NetworkDismantling/review) ships every method above. If we run a dismantling condition, this is the harness to run it against rather than reimplementing eight papers.

### 5.8 Lü et al. (Physics Reports 2016) — spreading vs connectivity disagree

The single most load-bearing result in this file for our purposes, because it compares **the same centralities under a diffusion objective and a connectivity objective on the same four graphs**. Datasets [verified, Table 2]: Amazon 334,863/925,872 · Cond-mat 27,519/116,181 · Email-Enron 36,692/183,831 · Facebook 63,731/817,090.

Kendall's τ against SIR spreading influence, `β = 1.5β_c`, 100 runs — **higher is better** [verified, Table 3]:

| Method          | Amazon     | Cond-mat   | Email-Enron | Facebook   |
| --------------- | ---------- | ---------- | ----------- | ---------- |
| Degree          | 0.2675     | 0.5657     | 0.4821      | 0.7348     |
| H-index         | 0.3219     | 0.6115     | 0.4883      | 0.7619     |
| Coreness        | 0.3289     | 0.6086     | 0.4883      | 0.7745     |
| **LocalRank**   | **0.6546** | **0.8040** | 0.5336      | **0.8043** |
| ClusterRank     | 0.4521     | 0.5548     | 0.4001      | 0.7611     |
| Closeness       | 0.5968     | 0.7190     | 0.3271      | 0.7038     |
| Betweenness     | 0.2508     | 0.3277     | 0.4224      | 0.4880     |
| **Eigenvector** | 0.3161     | 0.7350     | **0.5346**  | 0.7373     |

Robustness `R` after targeted removal — **lower is better** [verified, Table 4]:

| Method      | Amazon     | Cond-mat   | Email-Enron | Facebook   |
| ----------- | ---------- | ---------- | ----------- | ---------- |
| **Degree**  | **0.1226** | **0.1404** | **0.0404**  | **0.3131** |
| H-index     | 0.2298     | 0.2156     | 0.0605      | 0.3321     |
| Coreness    | 0.3118     | 0.2826     | 0.0704      | 0.3383     |
| LocalRank   | 0.3184     | 0.2787     | 0.1114      | 0.3701     |
| ClusterRank | 0.1861     | 0.1356     | 0.0785      | 0.3404     |
| Closeness   | 0.3778     | 0.2513     | 0.1677      | 0.3625     |
| Betweenness | 0.1746     | 0.1262     | 0.0501      | 0.2974     |
| Eigenvector | 0.3789     | 0.3365     | 0.1113      | 0.3904     |

Critical removal fraction `p_c` — lower is better [verified, Table 5]:

| Method      | Amazon     | Cond-mat   | Email-Enron | Facebook   |
| ----------- | ---------- | ---------- | ----------- | ---------- |
| **Degree**  | **0.2320** | 0.2539     | **0.0948**  | **0.5289** |
| H-index     | 0.4649     | 0.4132     | 0.1496      | 0.5938     |
| Coreness    | 0.6149     | 0.5277     | 0.2045      | 0.6137     |
| LocalRank   | 0.6298     | 0.5476     | 0.4738      | 0.7884     |
| ClusterRank | 0.3190     | 0.2240     | 0.2494      | 0.6487     |
| Closeness   | 0.6562     | 0.5330     | 0.4252      | 0.7435     |
| Betweenness | 0.2960     | **0.1992** | 0.1696      | 0.5389     |
| Eigenvector | 0.7810     | 0.6450     | 0.4642      | 0.8136     |

**The two objectives rank the same measures in opposite orders.** LocalRank is best for spreading and near-worst for connectivity; degree is best for connectivity and worst-or-near-worst for spreading. The paper says so directly: for connectivity, _"degree performs the best among all the methods"_; for spreading, _"betweenness is not so good, because the nodes with high betweenness values are usually the bridges connecting two communities and may not be of high spreading influences"_ [verified, prose].

**Consequence for §2:** a world model trained to predict diffusion cannot be scored against structural-CND published numbers, and vice versa. They are different problems that happen to share a name. Any table we build must pick one.

### 5.9 Min-Sum (PNAS 2016) — decycling numbers and two real graphs

Cavity prediction vs achieved decycling number on ER graphs, `N = 10⁷` [verified, arXiv 1603.08883 Table I]:

| `d` | `θ_dec(d)` (1RSB) | `θ_dec^MS(d)` |
| --- | ----------------- | ------------- |
| 1.5 | 0.0125            | 0.0135        |
| 2.5 | 0.0912            | 0.0936        |
| 3.5 | 0.1753            | 0.1782        |
| 5   | 0.2789            | 0.2823        |

Real-graph results are prose only [verified, same paper]: on **Twitter (N = 532,000)** MS+RG dismantles to components `< C = 1000` with **3.4%** of nodes vs CI's **5.6%** (a 60% improvement); on **YouTube (1.13M nodes)** MS+RG needs **4.0%**, a 22% improvement over CI. On ER `⟨d⟩ = 3.5`, `N = 5×10⁷`: MS **17.8%**, adaptive eigenvector centrality **20.2%**, adaptive CI **20.6%**, theoretical optimum **≈17.5%**. Reverse-greedy reinsertion alone accounts for **32%** fewer nodes for CI and **20%** fewer for MS. This paper published **no multi-network statistics table** — the Hamsterster/PGP/Enron rows people attribute to it come from CoreHD and BPD.

### 5.10 GND (PNAS 2019) — cost-weighted dismantling

Objective is total removal **cost**, not node count; cost = fraction of removed edges adjacent to removed nodes. Partial dismantling to **50% of the original GCC** [verified, arXiv 1801.01357]:

| Network                | N                   | GND / GNDR      | best competitor |
| ---------------------- | ------------------- | --------------- | --------------- |
| Crime (projection GCC) | 754                 | **0.03** (GNDR) | 0.10 (Min-Sum)  |
| Corruption             | 309                 | **0.14** (GNDR) | 0.19 (Min-Sum)  |
| Political blogs (PB)   | 1,222 / 16,714      | **0.55** (GND)  | 0.65 (EGP)      |
| Pokec (PK)             | 1.63×10⁶ / 2.23×10⁷ | **0.69** (GND)  | 0.91 (Min-Sum)  |

World airport network, cost = **total passenger flux**, target GCC = 80% of initial: GND **0.06** (6% of passengers) vs Min-Sum **0.25** (25%) [verified]. On `Petster-hamster` at fixed cost 0.4, Min-Sum is **5% worse than random removal** while GND reaches 62% GCC [verified, Fig. 1] — a reminder that a strong cardinality method can be worse than random under a cost metric.

### 5.11 FINDER (Nature MI 2020) — no per-network table exists

**FINDER publishes its real-network results only as heatmaps (Fig. 5).** The per-network ANC values live in the figure; the SI (Tables S3, S10–S17) holds network descriptions and runtimes. There is **no arXiv version**, the Nature MI PDF is paywalled, the PMC copy (PMC8191335) serves an interstitial that blocks scripted PDF download, and the [FINDER repo's](https://github.com/FFrankyy/FINDER) advertised `results/` directory **does not exist in `master`** (221 files checked; only `requirements.txt` matches). Everything below is prose extracted from the PMC full text — real text, not a summary, but prose:

- **Gnutella31, ND, node-degree-weighted** [verified]: to halve the GCC, the best prior method (HDA) needs **40.3%** total cost; FINDER needs **14.1%**. At a fixed cost of 0.2, the best available method (GND) leaves **80.8%** GCC; FINDER leaves **35.3%**.
- **Flickr** [verified]: FINDER is **7,734 s** vs GND **174,363 s** (≈22×) for ND-degree-weighted; **915 s** vs RatioCut **815,411 s** (≈890×) for CN unweighted.
- Nine real networks in five domains (criminal, biological, communication, infrastructure, social). Training graphs: **BA, 30–50 nodes**.
- Baselines: HDA, CI, RatioCut (CN); MinSum, GND (ND).

If FINDER's numbers are needed as a table, the practical route is [NetworkDismantling/review](https://github.com/NetworkDismantling/review) or MIND's Table 5 (§5.2), not the paper.

### 5.12 Wandelt et al. (Sci. Rep. 2018) — the comparison nobody cites enough

13 dismantling methods on 12 synthetic families + 20 real networks, metric `R = (1/N)Σ_Q s(Q)`. **Per-network `R` values are figure-only** [figure, Figs. 6, 7, 14]. The extractable numeric claims [verified, prose]:

- On `lesmis`, `R` spans **0.09** (BI, ABI) to **0.21** (KSH) — the best algorithms are **twice as effective** as the worst.
- **"BI yields the best attack in 70−80% of the cases."** Runners-up ABI and CoreHD sit within an absolute `R` difference of **0.01–0.02** on most networks.
- Recommendation: **BI** as the quality reference on small graphs (its runtime is ~`O(N³)`), **ABI** as the default competitor on large graphs.
- Retrospective critique, verbatim: _"CI was published essentially comparing to degree-based attacks only, without a comparison to betweenness-based competitors… ND performs much better than CI in our study… the authors [of ND] did not assess their method compared to betweenness or approximate versions of betweenness (B, BI, AB, and ABI), the strongest competitors in our study."_

**Iterative (recomputed) betweenness is the strongest simple baseline in this literature and almost no learned paper reports it.** MIND's Table 5 reports plain `BC` at 167.8 — but that is _static_ betweenness, not `BI`. If we run a dismantling arm, iterated betweenness must be in it.

### 5.13 BPHD (2024) — higher-order dismantling cost

Connectivity dismantling cost at `C = 0.01 N`, `H = 0` [verified, arXiv 2401.10028 Table 1]:

| Network       | N      | M      | ⟨k⟩   | BPHD     | best baseline |
| ------------- | ------ | ------ | ----- | -------- | ------------- |
| ER            | 10,000 | 17,500 | 3.50  | **0.48** | 0.60          |
| BA            | 10,000 | 29,997 | 6.00  | **0.68** | 0.78          |
| Yeast         | 2,375  | 11,693 | 9.85  | **0.79** | 0.84          |
| Collaboration | 5,094  | 7,515  | 2.95  | **0.51** | 0.75          |
| Email         | 1,134  | 5,451  | 9.61  | **0.80** | 0.93          |
| Social        | 2,000  | 16,098 | 16.10 | **0.85** | 0.88          |

(That last row, 2,000 / 16,098, is the Hamsterster GCC — see §6.2.)

### 5.14 Seed paper 2 — temporal CND on SocioPatterns contact networks

Datasets [verified, arXiv 2409.15142 Table 1]: `N` nodes, `T` time steps of 20 s, `E` active contacts, total duration.

| Dataset     | N   | T     | E       | Duration |
| ----------- | --- | ----- | ------- | -------- |
| High school | 326 | 5,818 | 188,508 | 7,374    |
| Workplace   | 216 | 4,274 | 78,249  | 993,540  |
| Hospital    | 74  | 1,139 | 32,424  | 347,480  |
| Conference  | 402 | 9,565 | 70,262  | 347,500  |

Peak value of `Ω` (epidemic speed) after **isolating** the top `x` fraction of critical nodes — lower means the removed nodes were more critical [verified, Table 3]. TB = temporal betweenness, TC = temporal closeness, TDD = temporal degree deviation; TSLC / TSLI / TSCR are the paper's three new measures.

High school:

| `x` | TB       | TSLC     | TDD         | TSLI         | TC           | TSCR     |
| --- | -------- | -------- | ----------- | ------------ | ------------ | -------- |
| 0.1 | 1763.72  | 1573.98  | 1701.64     | 1421.4       | **1309.16**  | 1443.72  |
| 0.3 | 8312.82  | 6777.66  | **5849.78** | 7072.6       | 6744.26      | 6727.34  |
| 0.5 | 14910.1  | 12605.04 | 14469.24    | **12181.58** | 12330.9      | 12338.74 |
| 0.7 | 21241.94 | 17281.86 | 19343.34    | **16975.7**  | 17059.06     | 17128.78 |
| 0.9 | 26835.46 | 22026.92 | 24736.18    | 21554.12     | **20625.58** | 20632.20 |

Conference:

| `x` | TB          | TSLC    | TDD     | TSLI        | TC       | TSCR    |
| --- | ----------- | ------- | ------- | ----------- | -------- | ------- |
| 0.1 | **171.18**  | 344.36  | 511.42  | 307.86      | 373.7    | 831.76  |
| 0.3 | 2246.84     | 2212.98 | 2648.32 | **1707.94** | 2488.7   | 1939.36 |
| 0.5 | **3238.32** | 3287.9  | 4657.52 | 3359.54     | 5656.44  | 3442.06 |
| 0.7 | **4953.78** | 5197.4  | 6820.94 | **4931.68** | 8252.52  | 5094.18 |
| 0.9 | 6337.06     | 5947.0  | 7805.3  | **5561.74** | 10341.96 | 6208.9  |

Hospital and Workplace rows are in the paper's Table 3 as well; there the six measures land within ~2% of each other at every `x` [verified] — the paper's own explanation is that those networks have near-uniform connectivity, so no measure separates. Percolation results (LCC vs removals, Fig. 6) are **figure-only**.

**Why this table matters to us:** it is the only result set in this file measured by _epidemic dynamics under node removal_, which is precisely the §2.2 objective. It also shows the effect size is small (5–20% between the best and worst measure) on the diffusion metric, versus 2× on the structural metric in Wandelt.

---

## 6. Datasets

Counts below are tagged by source: **self-counted** = the file was downloaded and the nodes/edges counted; **repo** = the repository's own statistics page, curled and read; **paper** = the paper's own table via `pdftotext -layout`. Average degree uses `2m/n` throughout (directed rows therefore count in+out), matching KONECT's `d`. HTTP codes: **206** on a direct-download link means a range request succeeded, i.e. the file exists and is byte-servable — equivalent to 200 here. KONECT, networkrepository/nrvis, and Newman's `netdata` index require the browser `-A` string; Newman's index page 403s but every `.zip` under it serves 200.

Authoritative rows for the social graphs we already load live in [`influence_maximization.md`](influence_maximization.md) §6; this section owns the **infrastructure and biological** graphs, which the IM literature does not use.

### 6.1 ✅ What we already load that this literature also uses

| `--dataset`        | Our counts                       | Appears in                                                                                                      | Their counts                          | Verdict                                                                                        |
| ------------------ | -------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `power_grid`       | 4,941 / 6,594 undirected         | CoreHD & BPD Table I ("Grid"), NIRM Table 1 ("UsPower"), Wandelt Table 2 ("power"), Ventresca weighted set, EML | 4,941 / 6,594                         | **byte-identical.** The single best comparison target in this file                             |
| `facebook`         | 4,039 / 88,234 undirected        | Wandelt Table 2, FINDER's real set                                                                              | 4,039 / 88,234                        | **byte-identical**                                                                             |
| `netscience`       | 1,589 / 2,742 undirected         | NIRM Table 1 (1,461 / 2,742), SPR Table 1, Wandelt Table 2 (379 / 914)                                          | three different graphs share the name | ⚠️ **version collision** — see §6.4                                                            |
| `ca_grqc`          | 5,242 / 14,484 undirected        | NIRM Table 1 (4,158 / 13,422), DCRS Table 2                                                                     | LCC-extracted                         | ⚠️ ours is the raw graph, theirs the LCC                                                       |
| `jazz`             | 198 / 2,742 undirected           | Ventresca weighted CNP set; centrality papers                                                                   | 198 / 2,742                           | identical, but **no dismantling baseline reports it**                                          |
| `email_eu_core`    | 1,005 / 24,929 arcs              | not standard here; the dismantling "Email" is a different 265,214-node graph                                    | —                                     | ⚠️ name collision with CoreHD's `Email`                                                        |
| `wiki_vote`        | 7,115 / 103,689 arcs             | not used in dismantling                                                                                         | —                                     | —                                                                                              |
| `twitter`          | 81,306 / 1,768,149 arcs          | Wandelt Table 2 (81,306 / 1,342,296 — symmetrized + LCC)                                                        | same source graph                     | ⚠️ edge count differs by preprocessing                                                         |
| `youtube`          | 1,134,890 / 2,987,624 undirected | Min-Sum (1.13M), FINDER                                                                                         | same                                  | ✅ same graph                                                                                   |
| `digg`             | 116,893 / ≈2.6M undirected       | GDM/MIND use `munmun_digg_reply` (30,398) and `digg-friends` (279,630)                                          | **different graphs**                  | ⚠️ three "Digg"s                                                                               |
| `er` / `ba` / `ws` | synthetic                        | Ventresca CNP benchmark, Wandelt Table 1, MIND training set                                                     | —                                     | ✅ standard, but see the `m=1` trap in §6.3                                                     |
| `sbm`              | synthetic                        | MIND's synthetic test set (`p_intra = 0.1`, `p_inter = 5/\|V\|`, group size 100)                                | —                                     | ✅ **SBM has a dismantling baseline** — unlike IM, where `influence_maximization.md` found none |

`cora_ml`, `lastfm_asia`, `nethept`, `netphy`, and `weibo` have no counterpart in this literature.

### 6.2 Full catalogue

#### Social / collaboration

| Name                       | Nodes     | Edges                                                                     | Dir? | Avg deg | Description                                                                         | Landing                                                                          | Direct download                                                                                                                                 | Count src    |
| -------------------------- | --------- | ------------------------------------------------------------------------- | ---- | ------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| Crime (raw, bipartite)     | 1,380     | 1,476 undirected                                                          | No   | 2.14    | St. Louis homicide person–crime bipartite net (829 persons + 551 crimes)            | [konect moreno_crime](http://konect.cc/networks/moreno_crime/) (200)             | [tar.bz2](http://konect.cc/files/download.tsv.moreno_crime.tar.bz2) (206)                                                                       | repo         |
| **Crime GCC (projection)** | **754**   | **2,127 undirected**                                                      | No   | 5.64    | The person-projection GCC — **this is the "Crime" of GND, FINDER and SPR**, not 829 | [GND repo](https://github.com/renxiaolong/Generalized-Network-Dismantling) (200) | [`Crime_Gcc.txt`](https://raw.githubusercontent.com/renxiaolong/Generalized-Network-Dismantling/master/Datasets_Main_Paper/Crime_Gcc.txt) (200) | self-counted |
| **Corruption**             | **309**   | **3,281 undirected**                                                      | No   | 21.24   | Brazilian corruption scandals 1987–2014 (Ribeiro et al. 2018), GCC                  | same repo (200)                                                                  | `…/Datasets_Main_Paper/Corruption_Gcc.txt` (200)                                                                                                | self-counted |
| Hamsterster (KONECT)       | 2,426     | 16,631 undirected                                                         | No   | 13.71   | Hamsterster pet-social friendships                                                  | [konect petster-hamster](http://konect.cc/networks/petster-hamster/) (200)       | [tar.bz2](http://konect.cc/files/download.tsv.petster-hamster.tar.bz2) (206)                                                                    | repo         |
| Hamsterster (netrepo)      | 2,426     | 16,630 undirected                                                         | No   | 13.71   | Same graph, one fewer edge                                                          | [networkrepository](https://networkrepository.com/soc-hamsterster.php) (200)     | [zip](https://nrvis.com/download/data/soc/soc-hamsterster.zip) (206)                                                                            | self-counted |
| **Hamsterster GCC (P-H)**  | **2,000** | **16,098 undirected**                                                     | No   | 16.10   | The GCC used by GND, Min-Sum, NIRM ("P-H"), BPHD ("Social"), Wandelt                | GND repo (200)                                                                   | `…/Datasets_Main_Paper/Petster-Hamster_GCC.txt` (200)                                                                                           | self-counted |
| PGP web of trust           | 10,680    | 24,316 undirected                                                         | No   | 4.55    | Standard dismantling benchmark                                                      | [konect arenas-pgp](http://konect.cc/networks/arenas-pgp/) (200)                 | [tar.bz2](http://konect.cc/files/download.tsv.arenas-pgp.tar.bz2) (206)                                                                         | repo         |
| Enron email                | 36,692    | 183,831 undirected                                                        | No   | 10.02   | LCC = 33,696 / 180,811 (Wandelt)                                                    | [SNAP](https://snap.stanford.edu/data/email-Enron.html) (200)                    | [txt.gz](https://snap.stanford.edu/data/email-Enron.txt.gz) (206)                                                                               | repo         |
| ca-AstroPh                 | 18,772    | 198,110 undirected                                                        | No   | 21.10   | arXiv astro-ph coauthorship; LCC 17,903 / 196,972                                   | [SNAP](https://snap.stanford.edu/data/ca-AstroPh.html) (200)                     | [txt.gz](https://snap.stanford.edu/data/ca-AstroPh.txt.gz) (206)                                                                                | repo         |
| ca-HepTh                   | 9,877     | 25,998 undirected                                                         | No   | 5.26    | arXiv hep-th coauthorship                                                           | [SNAP](https://snap.stanford.edu/data/ca-HepTh.html) (200)                       | [txt.gz](https://snap.stanford.edu/data/ca-HepTh.txt.gz) (206)                                                                                  | repo         |
| ca-GrQc ✅                 | 5,242     | 14,496 undirected (SNAP's own count, incl. self-loops; 14,484 post-clean) | No   | 5.53    | NIRM/DCRS use the 4,158-node LCC                                                    | [SNAP](https://snap.stanford.edu/data/ca-GrQc.html) (200)                        | [txt.gz](https://snap.stanford.edu/data/ca-GrQc.txt.gz) (206)                                                                                   | repo         |
| Deezer Europe              | 28,281    | 92,752 undirected                                                         | No   | 6.56    | Mutual follows; GDM/MIND                                                            | [SNAP](https://snap.stanford.edu/data/feather-deezer-social.html) (200)          | [zip](https://snap.stanford.edu/data/deezer_europe.zip) (206)                                                                                   | repo         |
| Digg reply                 | 30,398    | 87,627 arcs (86,404 unique)                                               | Yes  | 5.77    | `munmun_digg_reply`; GDM/MIND use its 29,652-node LCC                               | [konect](http://konect.cc/networks/munmun_digg_reply/) (200)                     | [tar.bz2](http://konect.cc/files/download.tsv.munmun_digg_reply.tar.bz2) (206)                                                                  | repo         |
| Digg friends               | 279,630   | 1,731,653 arcs                                                            | Yes  | 12.39   | GDM's `digg-friends`                                                                | [konect](http://konect.cc/networks/digg-friends/) (200)                          | [tar.bz2](http://konect.cc/files/download.tsv.digg-friends.tar.bz2) (206)                                                                       | repo         |
| Twitter ego ✅             | 81,306    | 1,768,149 arcs                                                            | Yes  | 43.49   | Wandelt symmetrizes to 1,342,296                                                    | [SNAP](https://snap.stanford.edu/data/egonets-Twitter.html) (200)                | [txt.gz](https://snap.stanford.edu/data/twitter_combined.txt.gz) (206)                                                                          | repo         |
| Twitter social (GDM/MIND)  | 465,017   | 834,797 arcs                                                              | Yes  | 3.59    | `munmun_twitter_social` — a **different** Twitter                                   | [konect](http://konect.cc/networks/munmun_twitter_social/) (200)                 | [tar.bz2](http://konect.cc/files/download.tsv.munmun_twitter_social.tar.bz2) (206)                                                              | repo         |
| p2p-Gnutella31             | 62,586    | 147,892 arcs                                                              | Yes  | 4.73    | CoreHD's "P2P"; FINDER's headline ND network                                        | [SNAP](https://snap.stanford.edu/data/p2p-Gnutella31.html) (200)                 | [txt.gz](https://snap.stanford.edu/data/p2p-Gnutella31.txt.gz) (206)                                                                            | repo         |
| web-Stanford               | 281,903   | 2,312,497 arcs                                                            | Yes  | 16.41   | MIND, FINDER                                                                        | [SNAP](https://snap.stanford.edu/data/web-Stanford.html) (200)                   | [txt.gz](https://snap.stanford.edu/data/web-Stanford.txt.gz) (206)                                                                              | repo         |
| com-Youtube ✅             | 1,134,890 | 2,987,624 undirected                                                      | No   | 5.27    | Min-Sum, FINDER                                                                     | [SNAP](https://snap.stanford.edu/data/com-Youtube.html) (200)                    | [txt.gz](https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz) (206)                                                   | repo         |
| soc-Epinions1              | 75,879    | 508,837 arcs                                                              | Yes  | 13.41   | MIND, FINDER                                                                        | [SNAP](https://snap.stanford.edu/data/soc-Epinions1.html) (200)                  | [txt.gz](https://snap.stanford.edu/data/soc-Epinions1.txt.gz) (206)                                                                             | repo         |

#### Infrastructure

| Name                        | Nodes     | Edges                                  | Dir? | Avg deg | Description                                                                                                                       | Landing                                                                                                                                               | Direct download                                                                                                                                | Count src            |
| --------------------------- | --------- | -------------------------------------- | ---- | ------- | --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| **US power grid** ✅        | **4,941** | **6,594 undirected**                   | No   | 2.67    | Western US grid (Watts–Strogatz). Diameter 46                                                                                     | [konect opsahl-powergrid](http://konect.cc/networks/opsahl-powergrid/) (200) · [networkrepository](https://networkrepository.com/inf-power.php) (200) | [tar.bz2](http://konect.cc/files/download.tsv.opsahl-powergrid.tar.bz2) (206) · [zip](https://nrvis.com/download/data/inf/inf-power.zip) (206) | self-counted + repo  |
| **USAir97**                 | **332**   | **2,126 undirected**                   | No   | 12.81   | The classic 332-airport US air-transport net                                                                                      | [networkrepository](https://networkrepository.com/inf-USAir97.php) (200) · [SuiteSparse](https://sparse.tamu.edu/Pajek/USAir97) (200)                 | [zip](https://nrvis.com/download/data/inf/inf-USAir97.zip) (206)                                                                               | self-counted         |
| Openflights (Opsahl)        | 2,939     | 30,501 arcs (15,677 unique undirected) | Yes  | 20.76   | The CNP/EI "openflights"                                                                                                          | [konect opsahl-openflights](http://konect.cc/networks/opsahl-openflights/) (200)                                                                      | [tar.bz2](http://konect.cc/files/download.tsv.opsahl-openflights.tar.bz2) (206)                                                                | repo + self-counted  |
| Openflights (netrepo)       | 2,939     | 15,677 undirected                      | No   | 10.67   | Symmetrized copy of the above                                                                                                     | [networkrepository](https://networkrepository.com/inf-openflights.php) (200)                                                                          | [zip](https://nrvis.com/download/data/inf/inf-openflights.zip) (206)                                                                           | self-counted         |
| Openflights (KONECT large)  | 3,425     | 67,663 arcs (37,595 unique)            | Yes  | 39.51   | ⚠️ **A different, bigger snapshot** cited interchangeably                                                                         | [konect openflights](http://konect.cc/networks/openflights/) (200)                                                                                    | [tar.bz2](http://konect.cc/files/download.tsv.openflights.tar.bz2) (206)                                                                       | repo                 |
| Euroroad                    | 1,174     | 1,417 undirected                       | No   | 2.41    | European E-road network                                                                                                           | [konect subelj_euroroad](http://konect.cc/networks/subelj_euroroad/) (200)                                                                            | [tar.bz2](http://konect.cc/files/download.tsv.subelj_euroroad.tar.bz2) (206)                                                                   | repo + self-counted  |
| **RoadEU GCC**              | **1,039** | **1,305 undirected**                   | No   | 2.51    | The GCC GND uses; CoreHD's table prints the raw 1,177 / 1,417                                                                     | GND repo (200)                                                                                                                                        | `…/Datasets_SI/RodeEU_gcc.txt` (200)                                                                                                           | self-counted         |
| roadNet-CA                  | 1,965,206 | 2,766,607 undirected                   | No   | 2.82    | MIND's `roads-california`                                                                                                         | [SNAP](https://snap.stanford.edu/data/roadNet-CA.html) (200)                                                                                          | [txt.gz](https://snap.stanford.edu/data/roadNet-CA.txt.gz) (206)                                                                               | repo                 |
| roadNet-PA                  | 1,088,092 | 1,541,898 undirected                   | No   | 2.83    | Pennsylvania roads                                                                                                                | [SNAP](https://snap.stanford.edu/data/roadNet-PA.html) (200)                                                                                          | [txt.gz](https://snap.stanford.edu/data/roadNet-PA.txt.gz) (206)                                                                               | repo                 |
| roadNet-TX                  | 1,379,917 | 1,921,660 undirected                   | No   | 2.79    | CoreHD's "RoadTX"                                                                                                                 | [SNAP](https://snap.stanford.edu/data/roadNet-TX.html) (200)                                                                                          | [txt.gz](https://snap.stanford.edu/data/roadNet-TX.txt.gz) (206)                                                                               | repo                 |
| Oregon1 AS                  | 11,051    | 22,724 undirected                      | No   | 4.11    | AS peering, route-views                                                                                                           | [SNAP](https://snap.stanford.edu/data/oregon1.html) (200)                                                                                             | [txt.gz](https://snap.stanford.edu/data/oregon1_010331.txt.gz) (206)                                                                           | repo                 |
| as-caida                    | 8,020     | 36,406 undirected                      | No   | 9.08    | CAIDA AS relationships                                                                                                            | [SNAP](https://snap.stanford.edu/data/as-caida.html) (200)                                                                                            | [txt.gz](https://snap.stanford.edu/data/as-caida20071105.txt.gz) (206)                                                                         | repo                 |
| Internet topology           | 34,761    | 171,403 (114,496 unique)               | No   | 9.86    | MIND's `internet-topology`                                                                                                        | [konect topology](http://konect.cc/networks/topology/) (200)                                                                                          | [tar.bz2](http://konect.cc/files/download.tsv.topology.tar.bz2) (206)                                                                          | repo                 |
| **IntNet1**                 | **6,474** | **12,572 undirected**                  | No   | 3.88    | The AS instance in CoreHD / BPD Table I                                                                                           | GND repo (200)                                                                                                                                        | `…/Datasets_SI/IntNet1_gcc.txt` (200)                                                                                                          | self-counted         |
| Madrid train bombing        | 64        | 243 undirected                         | No   | 7.59    | Terrorist contact net (`moreno_train`)                                                                                            | [konect](http://konect.cc/networks/moreno_train/) (200)                                                                                               | [tar.bz2](http://konect.cc/files/download.tsv.moreno_train.tar.bz2) (206)                                                                      | repo                 |
| GridKit Europe / N. America | —         | —                                      | —    | —       | MIND's `gridkit-eupowergrid` (13,844) and `gridkit-north america` (16,167) [paper]. Zenodo ships **GIS extracts, not edge lists** | [Zenodo 47317](https://zenodo.org/records/47317) (200)                                                                                                | `…/files/gridkit_euorpe.zip/content` (206 — upstream typo _euorpe_) · `…/files/gridkit_north_america.zip/content` (206)                        | paper (MIND Table 4) |
| CAIDA ARK IPv4 AS-links     | —         | —                                      | —    | —       | GDM's `ARK201012_LCC` source                                                                                                      | [CAIDA](https://www.caida.org/catalog/datasets/ipv4_routed_topology_aslinks_dataset/) (200)                                                           | **registration-gated**                                                                                                                         | —                    |

#### Biological

| Name                           | Nodes     | Edges                                | Dir? | Avg deg     | Description                                                                                                               | Landing                                                                                              | Direct download                                                                                          | Count src              |
| ------------------------------ | --------- | ------------------------------------ | ---- | ----------- | ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ---------------------- |
| **E. coli TRN**                | **418**   | **519 arcs**                         | Yes  | 2.48        | E. coli transcriptional regulation (`coli1_1`) — the CNP "Ecoli"                                                          | [Alon collection](https://www.weizmann.ac.il/mcb/UriAlon/download/collection-complex-networks) (200) | `…/CollectionsOfComplexNetwroks/coli1_1inter_st.txt` (200)                                               | self-counted           |
| Yeast transcription            | 688       | 1,079 arcs                           | Yes  | 3.14        | S. cerevisiae transcription net                                                                                           | Alon (200)                                                                                           | `…/yeastinter_st.txt` (200)                                                                              | self-counted           |
| **Yeast PPI (CoreHD "PPI")**   | **2,361** | **6,646 undirected**                 | No   | 5.63        | The dismantling-standard yeast PPI                                                                                        | —                                                                                                    | —                                                                                                        | paper (CoreHD Table I) |
| Yeast PPI GCC (GND / NIRM)     | 2,224     | 6,609 undirected                     | No   | 5.94        | GCC of the above; NIRM Table 1's `PPI`                                                                                    | GND repo (200)                                                                                       | `…/Datasets_SI/PPI_gcc.txt` (200)                                                                        | self-counted           |
| bio-yeast                      | 1,458     | 1,948 undirected                     | No   | 2.67        | Smaller yeast PPI variant                                                                                                 | [networkrepository](https://networkrepository.com/bio-yeast.php) (200)                               | [zip](https://nrvis.com/download/data/bio/bio-yeast.zip) (206)                                           | self-counted           |
| **HI-II-14 human interactome** | **4,303** | **13,944 undirected**                | No   | 6.48        | Rolland et al. 2014 binary interactome; FINDER's headline biological net (FINDER quotes an LCC of 4,165 / 13,087 [claim]) | [NDEx](https://www.ndexbio.org/viewer/networks/6816114d-669b-11e7-a03e-0ac135e8bacf) (200)           | [NDEx REST](https://www.ndexbio.org/v2/network/6816114d-669b-11e7-a03e-0ac135e8bacf/summary) (200, JSON) | NDEx API               |
| Human PPI (Stelzl)             | 1,706     | 6,207 arcs                           | Yes  | 7.28        | `maayan-Stelzl`; GDM, MIND                                                                                                | [konect](http://konect.cc/networks/maayan-Stelzl/) (200)                                             | [tar.bz2](http://konect.cc/files/download.tsv.maayan-Stelzl.tar.bz2) (206)                               | repo                   |
| Human PPI (Vidal)              | 3,133     | 6,726 undirected                     | No   | 4.29        | `maayan-vidal`; DCRS, SPR                                                                                                 | [konect](http://konect.cc/networks/maayan-vidal/) (200)                                              | [tar.bz2](http://konect.cc/files/download.tsv.maayan-vidal.tar.bz2) (206)                                | repo                   |
| Human PPI (Figeys)             | 2,239     | 6,452 arcs                           | Yes  | 5.76        | `maayan-figeys`; DCRS, SPR                                                                                                | [konect](http://konect.cc/networks/maayan-figeys/) (200)                                             | [tar.bz2](http://konect.cc/files/download.tsv.maayan-figeys.tar.bz2) (206)                               | repo                   |
| Yeast PPI (moreno_propro)      | 1,870     | 2,277 undirected                     | No   | 2.44        | Jeong et al.; MIND                                                                                                        | [konect](http://konect.cc/networks/moreno_propro/) (200)                                             | [tar.bz2](http://konect.cc/files/download.tsv.moreno_propro.tar.bz2) (206)                               | repo                   |
| C. elegans neural              | 297       | 2,359 arcs (2,148 unique undirected) | Yes  | 15.88       | Newman GML original; Wandelt's `celegansneural`                                                                           | Newman index **403**                                                                                 | [zip](http://www-personal.umich.edu/~mejn/netdata/celegansneural.zip) (200)                              | self-counted           |
| C. elegans neural (KONECT)     | 297       | 4,296 (2× symmetrized)               | Yes  | 28.93       | `dimacs10-celegansneural`; MIND                                                                                           | [konect](http://konect.cc/networks/dimacs10-celegansneural/) (200)                                   | [tar.bz2](http://konect.cc/files/download.tsv.dimacs10-celegansneural.tar.bz2) (206)                     | repo                   |
| C. elegans metabolic           | 453       | 2,025 undirected                     | No   | 8.94        | MIND's `arenas-met`                                                                                                       | [networkrepository](https://networkrepository.com/bio-celegans.php) (200)                            | [zip](https://nrvis.com/download/data/bio/bio-celegans.zip) (206)                                        | self-counted           |
| C. elegans metabolic (KONECT)  | 453       | 4,596 (2,040 unique)                 | No   | 20.29       | `arenas-meta`, multi-edge                                                                                                 | [konect](http://konect.cc/networks/arenas-meta/) (200)                                               | [tar.bz2](http://konect.cc/files/download.tsv.arenas-meta.tar.bz2) (206)                                 | repo                   |
| **Circuit s420**               | **252**   | **399 arcs**                         | Yes  | 3.17        | Electronic circuit — **the CNP "Circuit" (252 nodes)**                                                                    | Alon (200)                                                                                           | `…/CollectionsOfComplexNetwroks/s420_st.txt` (200)                                                       | self-counted           |
| Circuit s208 / s838            | 122 / 512 | 189 / 819 arcs                       | Yes  | 3.10 / 3.20 | Sibling circuits                                                                                                          | Alon (200)                                                                                           | `…/s208_st.txt`, `…/s838_st.txt` (200)                                                                   | self-counted           |

#### Other classics

| Name                       | Nodes | Edges                                     | Dir? | Avg deg | Landing / Direct download                                                                     | Count src    |
| -------------------------- | ----- | ----------------------------------------- | ---- | ------- | --------------------------------------------------------------------------------------------- | ------------ |
| Zachary karate ✅          | 34    | 78 undirected                             | No   | 4.59    | Newman index 403 · [karate.zip](http://www-personal.umich.edu/~mejn/netdata/karate.zip) (200) | self-counted |
| Dolphins                   | 62    | 159 undirected                            | No   | 5.13    | [dolphins.zip](http://www-personal.umich.edu/~mejn/netdata/dolphins.zip) (200)                | self-counted |
| Les Misérables             | 77    | 254 undirected                            | No   | 6.60    | [lesmis.zip](http://www-personal.umich.edu/~mejn/netdata/lesmis.zip) (200)                    | self-counted |
| Political blogs            | 1,490 | 19,090 arcs (16,718 unique, 3 self-loops) | Yes  | 25.62   | [polblogs.zip](http://www-personal.umich.edu/~mejn/netdata/polblogs.zip) (200)                | self-counted |
| Political blogs (KONECT)   | 1,224 | 19,025 arcs                               | Yes  | 31.09   | [konect moreno_blogs](http://konect.cc/networks/moreno_blogs/) (200)                          | repo         |
| Political blogs (GND "PB") | 1,222 | 16,714                                    | Yes  | 27.35   | —                                                                                             | paper (GND)  |
| Football                   | 115   | 613 undirected                            | No   | 10.66   | [football.zip](http://www-personal.umich.edu/~mejn/netdata/football.zip) (200)                | self-counted |
| Jazz ✅                    | 198   | 2,742 undirected                          | No   | 27.70   | [konect arenas-jazz](http://konect.cc/networks/arenas-jazz/) (200)                            | repo         |
| Netscience (Newman)        | 1,589 | 2,742 undirected                          | No   | 3.45    | [netscience.zip](http://www-personal.umich.edu/~mejn/netdata/netscience.zip) (200)            | self-counted |
| Adjnoun                    | 112   | 425 undirected                            | No   | 7.59    | [adjnoun.zip](http://www-personal.umich.edu/~mejn/netdata/adjnoun.zip) (200)                  | self-counted |
| Polbooks                   | 105   | 441 undirected                            | No   | 8.40    | Wandelt Table 2                                                                               | paper        |

### 6.3 Task-specific data

#### Ventresca's CNP benchmark instances

The instance set every CNP-1a metaheuristic paper reuses. Landing [individual.utoronto.ca/mventresca/cnd.html](https://individual.utoronto.ca/mventresca/cnd.html) (200), direct [`cnd.zip`](https://individual.utoronto.ca/mventresca/cnd.zip) (200). Format: line 1 = `|V|`, then `vertex: neighbours`. All undirected. A weighted variant [`wCNDP.zip`](https://engineering.purdue.edu/~mventresca/wCNDP.zip) (200) adds `celegans277`, `macacanetwork`, `netscience`, `power`, `subjectA1`.

| Instance               | Nodes | Undirected edges | Avg deg |
| ---------------------- | ----- | ---------------- | ------- |
| BarabasiAlbert_n500m1  | 500   | 499              | 2.00    |
| BarabasiAlbert_n1000m1 | 1,000 | 999              | 2.00    |
| BarabasiAlbert_n2500m1 | 2,500 | 2,499            | 2.00    |
| BarabasiAlbert_n5000m1 | 5,000 | 4,999            | 2.00    |
| ErdosRenyi_n250        | 235   | 350              | 2.98    |
| ErdosRenyi_n500        | 466   | 700              | 3.00    |
| ErdosRenyi_n1000       | 941   | 1,400            | 2.98    |
| ErdosRenyi_n2500       | 2,344 | 3,500            | 2.99    |
| ForestFire_n250        | 250   | 514              | 4.11    |
| ForestFire_n500        | 500   | 828              | 3.31    |
| ForestFire_n1000       | 1,000 | 1,817            | 3.63    |
| ForestFire_n2000       | 2,000 | 3,413            | 3.41    |
| WattsStrogatz_n250     | 250   | 1,246            | 9.97    |
| WattsStrogatz_n500     | 500   | 1,496            | 5.98    |
| WattsStrogatz_n1000    | 1,000 | 4,996            | 9.99    |
| WattsStrogatz_n1500    | 1,500 | 4,498            | 6.00    |

All counts **self-counted** from the downloaded archive. ⚠️ **The BA instances use `m = 1`, so they are trees** (`n−1` edges) — trivially dismantlable and a poor discriminator. The ER names are also misleading: `ErdosRenyi_n250` has 235 nodes, not 250 (the file is the graph's LCC). This is the standard set regardless, so match the filenames, not the names.

#### Wandelt's synthetic families

[verified, Sci. Rep. 2018 Table 1] — 12 generators, `n ∈ {100, 500, 1000}`: Barabási–Albert (`m ∈ {1,3,5,7,9}`), Erdős–Rényi (`p ∈ {0.01…0.05}`), Watts–Strogatz (`k ∈ {3,5,7,9}`, rewire `p ∈ {0.1,0.4,0.7}`), regular (`d ∈ {3,4,5}`), grid (side 5–30), path, circle, wheel, ladder, binary tree (`r ∈ {2,3}`, `h ∈ {3,4}`), hypercube (`d = 2…8`), barbell. **Grid, path, circle, ladder and hypercube families do not appear anywhere in the IM literature** and are the adversarial cases for message-passing models.

#### Temporal contact networks (seed paper 2)

SocioPatterns RFID face-to-face contact networks, 20 s time bins. Counts in §5.14. Landing page [sociopatterns.org](https://www.sociopatterns.org/) (200). ⚠️ the long-cited `www.sociopatterns.org/datasets/` path now **404s** — the index moved to `sociopatterns.org/datasets.html`; see `epidemic_control.md` §6 for the per-file base URL. These are the only _temporal_ CND benchmarks found.

### 6.4 ⚠️ Name collisions in this literature

Same trap as the name-collision table in [`influence_maximization.md`](influence_maximization.md) §6, different names.

| Name                | Version A                          | Version B                                            | Version C                                              |
| ------------------- | ---------------------------------- | ---------------------------------------------------- | ------------------------------------------------------ |
| **Crime**           | KONECT bipartite **1,380 / 1,476** | GND/FINDER/SPR person-projection GCC **754 / 2,127** | —                                                      |
| **NetScience**      | ours + Newman **1,589 / 2,742**    | NIRM **1,461 / 2,742** (LCC)                         | Wandelt **379 / 914** (a different coauthorship graph) |
| **Openflights**     | Opsahl **2,939 / 15,677**          | KONECT **3,425 / 37,595**                            | —                                                      |
| **Hamsterster**     | KONECT **2,426 / 16,631**          | dismantling GCC **2,000 / 16,098**                   | netrepo **2,426 / 16,630**                             |
| **Digg**            | ours **116,893 / ≈2.6M**           | `munmun_digg_reply` **30,398**                       | `digg-friends` **279,630**                             |
| **Twitter**         | SNAP ego **81,306 / 1,768,149**    | `munmun_twitter_social` **465,017 / 834,797**        | Min-Sum's crawl **532,000**                            |
| **Email**           | our `email_eu_core` **1,005**      | CoreHD/BPD "Email" **265,214 / 364,481**             | `DNCEmails` **1,866 / 4,384**                          |
| **PPI**             | CoreHD **2,361 / 6,646**           | GND/NIRM GCC **2,224 / 6,609**                       | `bio-yeast` **1,458 / 1,948**                          |
| **Political blogs** | Newman **1,490 / 19,090**          | KONECT **1,224 / 19,025**                            | GND **1,222 / 16,714**                                 |
| **Ca-GrQc**         | ours/SNAP **5,242 / 14,484**       | NIRM/DCRS LCC **4,158 / 13,422**                     | —                                                      |

**Almost every dismantling paper silently uses the LCC.** That is the single most common source of apparent count mismatch here, and it never changes the edge count much because the dropped fragments are tiny.

---

## 7. Which paper uses which

✅ marks a graph we already load. A cell is filled only where the dataset was read out of that paper's own transcribed table or its own prose (§5) — absence of a mark means "not established", not "not used". `V-CNP` = the Ventresca CNP benchmark suite reused by every §3.2 metaheuristic; `Seed-2` = arXiv 2409.15142.

| Dataset                          | V-CNP | CI  | MS  | CoreHD | BPD | GND | Wandelt | FINDER | GDM | NIRM | DCRS | SPR | MIND | BPHD | Lü  | Seed-2 |
| -------------------------------- | :---: | :-: | :-: | :----: | :-: | :-: | :-----: | :----: | :-: | :--: | :--: | :-: | :--: | :--: | :-: | :----: |
| **US power grid** ✅             |   ●   |     |     |   ●    |  ●  |     |    ●    |        |     |  ●   |      |     |      |      |     |        |
| **Facebook (4,039)** ✅          |       |     |     |        |     |     |    ●    |   ●    |     |      |      |     |      |      |     |        |
| **NetScience** ✅                |   ●   |     |     |        |     |     |    ●    |   ●    |     |  ●   |      |  ●  |      |      |     |        |
| **Ca-GrQc** ✅                   |       |     |     |        |     |     |         |        |     |  ●   |  ●   |     |      |      |     |        |
| **Jazz** ✅                      |   ●   |     |     |        |     |     |         |        |     |      |      |     |      |      |     |        |
| **YouTube** ✅                   |       |  ●  |  ●  |        |     |     |         |   ●    |     |      |      |     |      |      |     |        |
| **Twitter (ego)** ✅             |       |     |     |        |     |     |    ●    |        |     |      |      |     |      |      |     |        |
| Yeast PPI                        |       |     |     |   ●    |  ●  |  ●  |         |   ●    |     |  ●   |  ●   |  ●  |      |  ●   |     |        |
| HI-II-14 human                   |       |     |     |        |     |     |         |   ●    |     |  ●   |  ●   |     |      |      |     |        |
| Human PPI Figeys                 |       |     |     |        |     |     |         |        |  ●  |      |  ●   |  ●  |  ●   |      |     |        |
| Human PPI Vidal / Stelzl         |       |     |     |        |     |     |         |        |  ●  |      |  ●   |     |  ●   |      |     |        |
| Crime (754 GCC)                  |       |     |     |        |     |  ●  |         |   ●    |     |  ●   |      |  ●  |      |      |     |        |
| Corruption (309)                 |       |     |     |        |     |  ●  |         |        |     |  ●   |      |     |      |      |     |        |
| Hamsterster / P-H                |       |     |     |        |     |  ●  |         |        |  ●  |  ●   |      |     |  ●   |  ●   |     |        |
| PGP web of trust                 |       |  ●  |  ●  |        |     |     |    ●    |        |     |      |      |     |      |      |     |        |
| p2p-Gnutella31                   |       |     |     |   ●    |  ●  |     |         |   ●    |     |      |  ●   |     |  ●   |      |     |        |
| Political blogs                  |       |     |     |        |     |  ●  |    ●    |        |     |      |      |     |  ●   |      |     |        |
| Enron email                      |       |     |     |        |     |     |    ●    |        |     |      |      |     |      |      |  ●  |        |
| com-DBLP                         |       |     |     |        |     |     |         |        |  ●  |      |      |     |  ●   |      |     |        |
| soc-Epinions1                    |       |     |     |        |     |     |         |   ●    |     |      |      |     |  ●   |      |     |        |
| web-Stanford                     |       |     |     |        |     |     |         |   ●    |     |      |      |     |  ●   |      |     |        |
| roadNet-TX / -CA                 |       |     |     |   ●    |  ●  |     |         |        |     |      |      |     |  ●   |      |     |        |
| RoadEU / Euroroad                |       |     |     |   ●    |  ●  |  ●  |         |        |     |      |      |     |      |      |     |        |
| Internet AS (IntNet1/2)          |       |     |     |   ●    |  ●  |  ●  |    ●    |        |  ●  |      |      |     |  ●   |      |     |        |
| Openflights                      |       |     |     |        |     |  ●  |         |        |     |      |      |     |  ●   |      |     |        |
| USAir97                          |   ●   |     |     |        |     |     |    ●    |        |     |      |      |     |      |      |     |        |
| GridKit EU / N. America          |       |     |     |        |     |     |         |        |     |      |      |     |  ●   |      |     |        |
| Pokec                            |       |     |     |        |     |  ●  |         |        |     |      |      |     |      |      |     |        |
| World airport (flux)             |       |     |     |        |     |  ●  |         |        |     |      |      |     |      |      |     |        |
| FilmTrust                        |       |     |     |        |     |     |         |        |     |      |  ●   |  ●  |  ●   |      |     |        |
| RoviraVirgili                    |       |     |     |        |     |     |         |        |     |  ●   |  ●   |  ●  |      |      |     |        |
| DNCEmails                        |       |     |     |        |     |     |         |        |     |  ●   |  ●   |  ●  |      |      |     |        |
| KKI / EconPoli                   |       |     |     |        |     |     |         |        |     |      |      |  ●  |      |      |     |        |
| Bible / Roget / HM               |       |     |     |        |     |     |         |        |     |  ●   |  ●   |     |      |      |     |        |
| Amazon / Cond-mat                |       |     |     |        |     |     |         |        |     |      |      |     |      |      |  ●  |        |
| Facebook (63,731)                |       |     |     |        |     |     |         |        |     |      |      |     |      |      |  ●  |        |
| E. coli TRN / circuits           |   ●   |     |     |        |     |     |         |        |     |      |      |     |      |      |     |        |
| SocioPatterns (4 temporal)       |       |     |     |        |     |     |         |        |     |      |      |     |      |      |     |   ●    |
| ER / BA / WS / Forest-Fire       |   ●   |  ●  |  ●  |   ●    |  ●  |  ●  |    ●    |   ●    |  ●  |  ●   |  ●   |  ●  |  ●   |  ●   |     |        |
| **SBM**                          |       |     |     |        |     |     |         |        |     |      |      |     |  ●   |      |     |        |
| Grid / path / circle / hypercube |       |     |     |        |     |     |    ●    |        |     |      |      |     |      |      |     |        |

**Three observations.**

1. **`power_grid` is the only graph we load that three separate dismantling papers report absolute numbers for** (CoreHD, BPD, NIRM). Everything else we load is either absent, version-mismatched, or figure-only.
2. **The learned branch and the OR branch share almost no data.** V-CNP's column is nearly disjoint from FINDER/GDM/MIND. There is no published table in which MA-CNP and FINDER appear together.
3. **SBM has a dismantling baseline (MIND) where it had no IM baseline.** That inverts the gap noted in [`influence_maximization.md`](influence_maximization.md) — our SBM experiments can be positioned against MIND here even though they could not be positioned against anything on the IM side.

---

## 8. Evaluation protocol

### 8.1 Metrics, by branch

| Branch                | Primary metric                                                                            | Budget convention                                                                     | Lower/higher is better  |
| --------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ----------------------- |
| OR / CNP              | pairwise connectivity `Σ\|C_i\|(\|C_i\|−1)/2`of`G − S`                                    | absolute `k`, fixed per instance (Ventresca's benchmark ships a `K` with every graph) | lower                   |
| OR / CC-CNP           | `\|S\|` s.t. `max\|C_i\| ≤ L`                                                                | `L` fixed (often `L = √N` or a percentage)                                            | lower                   |
| Physics / dismantling | fraction of nodes removed to reach `\|GCC\| ≤ C·N`, `C = 0.01`                             | none — you remove until the threshold                                                 | lower                   |
| Physics / robustness  | Schneider's `R = (1/N)Σ_q s(q)`                                                           | full sweep `q = 1…N`                                                                  | lower (for an attacker) |
| Learned (FINDER)      | **ANC** with `σ ∈ {pairwise conn, GCC size, #components}`                                 | full sweep, cost-weighted                                                             | lower                   |
| Learned (GDM, MIND)   | **AUC of the dismantling curve**, often reported _relative_ to the proposed method (=100) | full sweep                                                                            | lower                   |
| Learned (NIRM)        | `ρ = \|S\| /\|V\| `at threshold`Θ = 0.01`                                                    | none — remove until `Θ`                                                               | lower                   |
| Diffusion CND         | final infected count / fraction after removals                                            | `k` as % of `N`                                                                       | lower                   |

### 8.2 The traps

These are the ways two published numbers that look comparable are not.

1. **Batch vs sequential.** The OR line removes `k` nodes _at once_ and evaluates once. The physics/learned line removes one node at a time, **recomputing the residual graph after every removal**. Sequential-adaptive is strictly stronger at equal `k`. NIRM reports both and the gap is large: on `UsPower`, `ρ = 8.58%` adaptive vs `16.81%` one-pass [verified, NIRM Table 3] — a **1.96× difference on the same method and same graph**. Any table that mixes the two is meaningless.
2. **Reinsertion.** Min-Sum, CoreHD, and GND all ship a post-processing pass that _reinserts_ removed nodes whose reinsertion does not push the GCC back over the threshold. `GND` and `GNDR` (GND + reinsertion) are different methods with different numbers, and papers cite both under the name "GND". Same for Min-Sum with/without reinsertion. Always check.
3. **Which `σ` inside ANC.** FINDER's ANC is parametric in the connectivity measure. ANC-pairwise and ANC-GCC are different scales. A paper reporting "our ANC beats FINDER's" without naming `σ` cannot be checked.
4. **Unit cost vs weighted cost.** GND's whole contribution is _generalized_ dismantling: minimize total removal **cost** (e.g. `Σ deg(v)` over removed nodes), not node count. Comparing GND's cost-optimal set to CI's cardinality-optimal set on a cardinality metric is unfair to GND and vice versa.
5. **Relative-to-self normalization.** GDM and MIND publish AUC _relative to their own method = 100.0_. Those tables are internally consistent and externally useless: you cannot combine MIND's Table 5 with GDM's absolute numbers, because the normalizer differs per network.
6. **Threshold `C`.** `C = 0.01` is standard but `0.001` and `√N/N` both appear. The dismantling-set size is steeply sensitive to `C` near the percolation transition.
7. **LCC extraction.** Nearly every dismantling paper runs on the **largest connected component** of the raw file, silently. This is why NIRM's `Ca-GrQc` is 4,158/13,422 while SNAP's raw `ca-GrQc` — and our loader — report 5,242/14,484 [verified, NIRM Table 1 vs our loader]. Same graph, different preprocessing. `NetScience` is the same story (NIRM 1,461 vs our 1,589, both at 2,742 edges — an LCC drop of 128 isolated/fragment nodes that leaves the edge count untouched).
8. **Decycling ≠ dismantling.** Min-Sum's guarantee is about the feedback vertex set. On graphs with many short loops (dense social graphs) the decycling set is far larger than the dismantling set, which is exactly why the reinsertion pass exists.
9. **Directedness.** Almost the entire literature is undirected. Our `email_eu_core`, `wiki_vote`, `nethept`, and `weibo` are directed; the only dismantling paper found that treats direction natively is [arXiv 2512.11416](https://arxiv.org/abs/2512.11416) (Dec 2025). Symmetrizing to reuse a published baseline changes the graph.

### 8.3 What we would report

For the diffusion-based variant (§2.2), the honest metric set is:

- **Containment curve**: final infected fraction vs removal budget `k/N ∈ {1, 5, 10, 20}%`, matching the budget convention we already use for IM (`--budget-pcts`), so the CND table and the IM table share an x-axis.
- **Regret against an MC oracle**, exactly as `planning_regret_multi` already does for IM, with `argmin`.
- **Baselines that must be beaten**: `degree`, `betweenness`, `pagerank`, adaptive-degree (recompute after each removal), and `random`. Adaptive degree is the one that actually hurts — MIND's Table 5 [verified] puts plain adaptive degree (`AD`) at **119.9** overall versus FINDER at **115.0**, i.e. the strongest RL dismantler of 2020 beats recompute-the-degree by under 5% on average AUC across 47 networks. **A learned dismantler that does not clearly beat adaptive degree has not demonstrated anything**, and this is the direct analogue of the BA-100 degree-triviality result we already hit on IM.
- The **structural** metrics (`pairwise_conn`, `largest_cc_size`, `n_components`) reported alongside, computed exactly, as descriptive context — not as the learned target.

---

## 9. Implications for this project

### 9.1 Build the diffusion variant, document why we skipped the structural one

The honest position, stated once in the paper and then dropped:

> Structural CNDP is not a world-model problem. Its transition is deterministic and its ground-truth evaluator (`nx.connected_components`, `O(N+E)`) is cheaper than our model's forward pass, so a learned surrogate has nothing to amortize. We therefore evaluate the **diffusion-based** variant — minimize the eventual IC/LT cascade under a `k`-node removal budget — where the ground truth costs 30 MC rollouts per step and a learned transition kernel does buy something.

That paragraph is worth more than a mediocre connectivity regressor. §5.8 (Lü's Tables 3–5) is the citation that makes it not merely convenient: the same centralities rank in **opposite orders** under a spreading objective and a connectivity objective, so the two really are different problems.

### 9.2 Order of work

| #   | Task                                                                                                                                                                                  | Effort      | What it buys                                                                                                |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ----------------------------------------------------------------------------------------------------------- |
| 1   | Fix `remove_node` semantics (§2.3): node deletion as a bag `remove_node(v) + remove_edge(v,u) ∀u`, and an IC branch that does **not** count an immunized susceptible node as infected | ~half a day | Removes a systematic `+k` bias from every containment number we would report. **Blocking — do this first.** |
| 2   | `--action-ops remove_node` generation mode + a minimize flag on `planning_regret_multi`                                                                                               | ~1 day      | The whole diffusion-CND task, reusing the existing backbone, heads, features, and eval                      |
| 3   | Add `usair97` (332/2,126), `crime` (754/2,127) and `corruption` (309/3,281) loaders                                                                                                   | ~2 h each   | Three tiny graphs that let us report on the same networks as GND, FINDER and SPR                            |
| 4   | Sequential-removal evaluator computing `R` / ANC / ρ-at-Θ in `wm_metrics.py`, reusing `data/old/connectivity.py`                                                                      | ~half a day | Lets us print a structural column next to the diffusion column for context                                  |
| 5   | Coding-agent dismantling arm against [NetworkDismantling/review](https://github.com/NetworkDismantling/review)                                                                        | ~2–3 days   | Puts us on the FINDER/GDM/MIND table without pretending the world model is doing the work                   |

Total for a defensible CND section: **~1 week**, of which item 2 is the only part that touches the model.

### 9.3 Baselines we must beat, and the one that will hurt

From §5.2 and §5.12, in ascending order of danger:

1. `random`, `pagerank`, static `betweenness` — free wins.
2. **Adaptive degree (HDA)** — recompute degree after each removal. MIND's Table 5 puts it at 119.9 against FINDER's 115.0. **A five-point gap is the entire published advantage of 2020-era deep RL over a ten-line heuristic.**
3. **Iterative betweenness (BI/ABI)** — Wandelt found it best in 70–80% of cases, and _almost no learned dismantling paper reports it_. If we omit it we reproduce the exact methodological gap Wandelt calls out in CI and GND.
4. **GND** — on `NetScience` a 2019 spectral heuristic (5.82) still beats every learned method except SPR (5.75) [verified, §5.4].

Our existing `coding_agent/tools/algorithms.py` already has degree, pagerank and betweenness selectors; the adaptive/iterated variants are a `for` loop around them, not new algorithms.

### 9.4 Expect to lose on `power_grid`, and say so first

`power_grid` is simultaneously our best-matched benchmark (§7) and the graph class where learned dismantlers demonstrably fail. MIND Table 5: on `eu-powergrid` FINDER scores **161.7** and on `roads-california` **116.3** while EI scores **80.6** and **23.4** — the physics heuristic beats the RL method by up to 5×. Diameter-46 mesh graphs exhaust a 3–6 layer receptive field. Reporting this as a predicted-and-confirmed limitation is a stronger result than reporting a surprise.

### 9.5 What the literature validates about our design

- **Train tiny, apply large is standard here, not exotic.** FINDER trains on BA graphs of 30–50 nodes; NIRM on 20–30-node graphs; MIND on 10,000 graphs of 100–200 nodes. Our BA-100 training regime needs no defence.
- **MIND's Fig. 4 finding is a warning about `CH_DEGREE`.** GDM's dismantling order correlates at Spearman **R = 0.762** with a PCA of its own handcrafted input features (MIND: 0.349). We feed `log1p(degree)` as channel 2 and then compare against a degree baseline. If our model's ranking correlates with its own degree channel at 0.76, we have re-derived degree with extra steps — worth measuring explicitly.
- **Self-terminating heads matter more here than in IM.** The `frontier_u` gate in `ICTransmissionHead` is what makes a containment objective learnable at all: a saturating rollout reports every removal as useless because everything gets infected regardless.
- **Effect sizes are small on the diffusion metric.** Seed paper 2 (§5.14) shows 5–20% between best and worst temporal centrality on epidemic peak, versus 2× on the structural metric (Wandelt's `lesmis`, `R` 0.09 → 0.21). Budget for tighter error bars and more MC repeats than the IM experiments needed.

### 9.6 What already exists in the repo

- `data/old/connectivity.py` — `simulate_removal()` already returns `(connectivity_vec, n_components, largest_cc_size, pairwise_conn)`, i.e. three of the six objectives in §1.1, exactly.
- `data/old/generate_cnd_data.py` — the archived structural-CND generator. Its docstring documents the same `removal_sets → connectivity_vecs` schema this file argues is a _regressor_, not a world model. It is evidence for §2.1, not code to revive.
- `wm_eval.py::remove_frontier_success` — already a containment metric under another name.

---

## 10. Reference list

All links checked **2026-07-28**. `403` marks publishers that reject scripted requests (Wiley, PNAS, APS, ACM, ScienceDirect) — the pages resolve in a browser.

**Seed papers** [Chen et al. 2025, _Critical Nodes Identification in Complex Networks: A Survey_ (arXiv 2507.06164)](https://arxiv.org/abs/2507.06164) · [Farahi, Kamandi, Abedian, Rocha 2024, _Critical Node Detection in Temporal Social Networks_ (arXiv 2409.15142)](https://arxiv.org/abs/2409.15142)

**Surveys and benchmark studies** [Lü, Chen, Ren, Zhang, Zhang, Zhou 2016, _Vital nodes identification in complex networks_, Physics Reports 650:1–63 (arXiv 1607.01134)](https://arxiv.org/abs/1607.01134) · [Lalou, Tahraoui, Kheddouci 2018, _The Critical Node Detection Problem in networks: A survey_, Computer Science Review 28:92–117](https://doi.org/10.1016/j.cosrev.2018.02.002) · [Wandelt, Sun, Feng, Zanin, Havlin 2018, _A comparative analysis of approaches to network-dismantling_, Sci. Rep. 8:13513](https://www.nature.com/articles/s41598-018-31902-8) · [Artime, Grassia, De Domenico, Gleeson, Makse, Mangioni, Perc, Radicchi 2024, _Robustness and resilience of complex networks_, Nature Reviews Physics 6:114–131 (arXiv 2509.19867)](https://arxiv.org/abs/2509.19867) · [nature.com](https://www.nature.com/articles/s42254-023-00676-y) · [runnable baselines](https://github.com/NetworkDismantling/review) · [Wandelt et al. 2025, _Recent advances in network dismantling_, Chaos Solitons & Fractals](https://www.sciencedirect.com/science/article/abs/pii/S0960077925006861) (**403**)

**Exact / complexity (OR)** [Arulselvan, Commander, Elefteriadou, Pardalos 2009, Computers & OR 36(7):2193–2200](https://doi.org/10.1016/j.cor.2008.08.016) · [Arulselvan, Commander, Shylo, Pardalos 2011, CC-CNP (Springer ch.)](https://doi.org/10.1007/978-1-4419-0534-5_4) · [Di Summa, Grosso, Locatelli 2011, _CNP over trees_, Computers & OR 38(12):1766–1774](https://doi.org/10.1016/j.cor.2011.02.016) · [Di Summa, Grosso, Locatelli 2012, _Branch and cut_, Comput. Optim. Appl. 53:649–680](https://doi.org/10.1007/s10589-011-9420-4) · [Addis, Di Summa, Grosso 2013, _Bounded treewidth_, Discrete Appl. Math. 161:2349–2360](https://doi.org/10.1016/j.dam.2013.03.021) · [Shen & Smith 2012, Networks 60(2):103–119](https://doi.org/10.1002/net.20464) (**403**) · [Shen, Smith, Goli 2012, Discrete Optimization 9(3):172–188](https://doi.org/10.1016/j.disopt.2012.07.001) · [Veremyev, Boginski, Pasiliao 2014, Optimization Letters 8:1245–1259](https://doi.org/10.1007/s11590-013-0666-x) · [Veremyev, Prokopyev, Pasiliao 2014, J. Comb. Optim. 28:233–273](https://doi.org/10.1007/s10878-014-9730-4) · [Veremyev, Prokopyev, Pasiliao 2015, _Distance-based CNP_, Networks 66(3):170–195](https://doi.org/10.1002/net.21622) (**403**) · [Alozie, Arulselvan, Akartunalı, Pasiliao 2021, Computers & OR 131:105254](https://doi.org/10.1016/j.cor.2021.105254) · [Hermelin, Kaspi, Komusiewicz, Navon 2016, Theor. Comp. Sci. 651:62–75](https://doi.org/10.1016/j.tcs.2016.08.016) · [Fortz, Mycek, Pióro, Tomaszewski 2024, Networks 83(2):256–288](https://doi.org/10.1002/net.22200) (**403**) · [Bayarsaikhan, Chinchuluun, Arulselvan, Pardalos 2025, _Stochastic CNDP_ (arXiv 2512.01497)](https://arxiv.org/abs/2512.01497)

**Metaheuristics** [Ventresca 2012, Computers & OR 39(11):2763–2775](https://doi.org/10.1016/j.cor.2012.02.008) · [benchmark instances `cnd.zip`](https://individual.utoronto.ca/mventresca/cnd.zip) · [weighted `wCNDP.zip`](https://engineering.purdue.edu/~mventresca/wCNDP.zip) · [Ventresca & Aleman 2014, _Derandomized approximation_, Computers & OR 43:261–270](https://doi.org/10.1016/j.cor.2013.09.012) · [Ventresca, Harrison, Ombuki-Berman 2015, EvoApplications](https://doi.org/10.1007/978-3-319-16549-3_14) · [Pullan 2015, J. Heuristics 21(5):577–598](https://doi.org/10.1007/s10732-015-9290-5) · [Addis, Aringhieri, Grosso, Hosteins 2016, Annals of OR 238:637–649](https://doi.org/10.1007/s10479-016-2110-y) · [Aringhieri, Grosso, Hosteins, Scatamacchia 2016, Eng. Appl. AI 55:128–145](https://doi.org/10.1016/j.engappai.2016.06.010) · [Aringhieri et al. 2016, _Local search_, Networks 67(3):209–221](https://doi.org/10.1002/net.21671) (**403**) · [Zhou, Hao, Glover 2019, _Memetic search (MA-CNP)_, IEEE T. Cybernetics 49(10):3699–3712 (arXiv 1705.04119)](https://arxiv.org/abs/1705.04119) · [Zhou et al. 2019, _Incremental evaluation_ (arXiv 1908.11846)](https://arxiv.org/abs/1908.11846) · [Chen, Jiang, Jiang, Zhang 2020, Physica A 538:122862](https://doi.org/10.1016/j.physa.2019.122862) · [Li, Liu, Yang 2020, _NIPA_, Expert Syst. Appl. 139:112853 (arXiv 2003.04713)](https://arxiv.org/abs/2003.04713) · [Wang, Deng, Holme, Di, Lü, Wu 2021, _Targeted enumeration_ (arXiv 2111.02655)](https://arxiv.org/abs/2111.02655) · [Liu, Ge, Chen, Pei, Zhu, Mei 2024, _K2GA_ (arXiv 2402.00404)](https://arxiv.org/abs/2402.00404)

**Physics / percolation** [Schneider, Moreira, Andrade, Havlin, Herrmann 2011, _Mitigation of malicious attacks_, PNAS 108(10):3838–3841 (arXiv 1009.3125)](https://arxiv.org/abs/1009.3125) · [Morone & Makse 2015, _Influence maximisation through optimal percolation_, Nature 524:65–68 (arXiv 1506.08326)](https://arxiv.org/abs/1506.08326) · [doi](https://doi.org/10.1038/nature14604) · [code](https://github.com/makselab/Collective-Influence) · [Morone, Min, Bo, Mari, Makse 2016, Sci. Rep. 6:30062 (arXiv 1603.08273)](https://arxiv.org/abs/1603.08273) · [Braunstein, Dall'Asta, Semerjian, Zdeborová 2016, _Network dismantling_, PNAS 113(44):12368–12373 (arXiv 1603.08883)](https://arxiv.org/abs/1603.08883) · [code](https://github.com/abraunst/decycler) · [Mugisha & Zhou 2016, _BPD_, Phys. Rev. E 94:012305 (arXiv 1603.05781)](https://arxiv.org/abs/1603.05781) · [Zdeborová, Zhang, Zhou 2016, _CoreHD_, Sci. Rep. 6:37954 (arXiv 1607.03276)](https://arxiv.org/abs/1607.03276) · [third-party code](https://github.com/hcmidt/corehd) · [Clusella, Grassberger, Pérez-Reche, Politi 2016, _Explosive Immunization_, PRL 117:208301 (arXiv 1604.00073)](https://arxiv.org/abs/1604.00073) · [code](https://github.com/pclus/explosive-immunization) · [Tian, Bashan, Shi, Liu 2017, _Articulation points_, Nat. Commun. 8:14223 (arXiv 1609.00094)](https://arxiv.org/abs/1609.00094) · [Ren, Gleinig, Helbing, Antulov-Fantulin 2019, _Generalized Network Dismantling_, PNAS 116(14):6554–6559 (arXiv 1801.01357)](https://arxiv.org/abs/1801.01357) · [code + datasets](https://github.com/renxiaolong/Generalized-Network-Dismantling) · [Osat, Papadopoulos, Teixeira, Radicchi 2022, _Embedding-aided dismantling_ (arXiv 2208.01087)](https://arxiv.org/abs/2208.01087) · [Peng, Fan, Lü 2024, _BPHD / higher-order dismantling_ (arXiv 2401.10028)](https://arxiv.org/abs/2401.10028) · [Liu, Hu, Wang, Liu, Zhang 2025, _Optimal dismantling of directed networks_ (arXiv 2512.11416)](https://arxiv.org/abs/2512.11416)

**Learning-based** [Fan, Zeng, Sun, Liu 2020, _FINDER_, Nature Mach. Intell. 2:317–324](https://www.nature.com/articles/s42256-020-0177-2) · [free full text PMC8191335](https://pmc.ncbi.nlm.nih.gov/articles/PMC8191335/) · [code](https://github.com/FFrankyy/FINDER) · [Grassia, De Domenico, Mangioni 2021, _GDM_, Nat. Commun. 12:5190 (arXiv 2101.02453)](https://arxiv.org/abs/2101.02453) · [code](https://github.com/NetworkScienceLab/GDM) · [Yan, Xie, Zhang, He, Yang 2021, _Hypernetwork dismantling via DRL_ (arXiv 2104.14332)](https://arxiv.org/abs/2104.14332) · [Zhang & Wang 2022, _NIRM_, CIKM (arXiv 2208.07792)](https://arxiv.org/abs/2208.07792) · [code](https://github.com/JiazhengZhang/NIRM) · [Zhang & Wang 2023, _DCRS_, WWW (arXiv 2301.12349)](https://arxiv.org/abs/2301.12349) · [code](https://github.com/JiazhengZhang/DCRS) · [Grassia & Mangioni 2023, _CoreGDM_, Complex Networks XIV](https://doi.org/10.1007/978-3-031-28276-8_8) · [Rezaei, Munoz, Jalili, Khayyam 2022, _EML_ (arXiv 2202.06229)](https://arxiv.org/abs/2202.06229) · [Xie et al. 2024, _ESND_ (arXiv 2406.08899)](https://arxiv.org/abs/2406.08899) · [Zheng, Ding, Jin, Gao, Li 2025, _Symbolized RL resilience (Selinda)_ (arXiv 2507.08827)](https://arxiv.org/abs/2507.08827) · [code](https://github.com/tsinghua-fib-lab/selinda) · [_MultiDismantler_ 2025, Nature Mach. Intell.](https://www.nature.com/articles/s42256-025-01070-2) · [Tian, Ferraro, Shorten, Jalili, Hamedmoghadam 2026, _MIND_, AAAI-26 (arXiv 2508.00706)](https://arxiv.org/abs/2508.00706) · [code](https://github.com/HaozheTian/MIND-ND) · [Zhou, Tan, Fang, Lü, Zhao 2026, _SPR / HoGNN_, Commun. Phys. 9:181](https://www.nature.com/articles/s42005-026-02601-y) · [code](https://github.com/zhouwn/spr) · [Hao et al. 2026, _hyper-VDrank_ (arXiv 2606.23289)](https://arxiv.org/abs/2606.23289)

**Data sources** [SNAP](https://snap.stanford.edu/data/) · [KONECT](http://konect.cc/networks/) (needs browser `-A`) · [Network Repository / nrvis](https://networkrepository.com/) (needs browser `-A`) · [Newman `netdata`](http://www-personal.umich.edu/~mejn/netdata/) (index **403**, individual `.zip` files 200) · [Uri Alon network collection](https://www.weizmann.ac.il/mcb/UriAlon/download/collection-complex-networks) · [GND datasets](https://github.com/renxiaolong/Generalized-Network-Dismantling) · [SocioPatterns](http://www.sociopatterns.org/datasets/) · [Netzschleuder](https://networks.skewed.de/) · [Ventresca CNP instances](https://individual.utoronto.ca/mventresca/cnd.html)

**Cross-references in this repo** [`influence_maximization.md`](influence_maximization.md) (KPP-Pos, the sign-flipped twin of this task) · [`epidemic_control.md`](epidemic_control.md) (immunization measured epidemiologically rather than structurally) · [`influence_blocking.md`](influence_blocking.md) (blocking by counter-cascade rather than by removal) · [`influence_maximization.md`](influence_maximization.md) §6 (authoritative rows for the social graphs).

---

## 11. Open gaps

Honest list of what this review could **not** establish.

### Tables that exist but could not be extracted

- **FINDER's per-network ANC values — the single biggest gap.** FINDER is the most-cited learned baseline in this literature and it publishes its real-network results **only as heatmaps** (Fig. 5). There is **no arXiv version**; the Nature MI PDF is paywalled; the PMC copy (PMC8191335) serves an interstitial that defeats scripted PDF download; and the `results/` directory the repo's README advertises **does not exist in `master`** (all 221 tracked paths checked). Only prose numbers (§5.11) were recoverable. To get a FINDER table you must either read the SI in a browser or re-run [NetworkDismantling/review](https://github.com/NetworkDismantling/review).
- **NIRM's main comparison** (vs DC, CI, BPD, CoreHD, GND, FINDER, NSKSD across 15 networks) is heatmap-only (Figs. 3–4). Only its ablation table (§5.6) is numeric.
- **Wandelt 2018's per-network `R` values** are figure-only (Figs. 6, 7, 14). Given that this is the paper naming **iterative betweenness** as the strongest baseline, the absence of a table is unfortunate — its claims in §5.12 are prose.
- **Min-Sum's real-network table** does not exist. Its PNAS paper reports two graphs in prose. The Hamsterster/PGP/Enron rows commonly attributed to it actually come from CoreHD and BPD.
- **MA-CNP (Zhou, Hao, Glover 2019) objective values** on the Ventresca benchmark were not transcribed — the arXiv PDF was located but its result tables were not extracted within this review. This is the strongest metaheuristic and its numbers are the reference point for §3.2.
- **Ventresca 2012's own published objective values** for PBIL/SA on his instances were not extracted; only the instance suite itself was downloaded and counted.
- **GDM's per-network table** (its Table 1 averages 45 networks) — the per-network breakdown lives in supplementary material that was not retrieved.
- **Seed paper 2's percolation results** (LCC vs removal fraction, Fig. 6) are figure-only; only its epidemic-peak Table 3 is numeric.

### Facts not established

- **A single table containing both an OR metaheuristic and a learned dismantler.** None was found. The two branches use different objectives, budgets and graphs (§7), so "MA-CNP vs FINDER" has no published answer.
- **Complexity results per graph class**, beyond the paper titles. The precise statement of what Di Summa 2011 proves about trees (NP-hardness in the weighted case vs polynomial DP in the unweighted case) was **not** verified against the paper text — §1.4 deliberately defers to §3.1's citations rather than asserting a boundary. Same for Hermelin's FPT/W[1] split.
- **CoreHD's `Email` row in BPD's Table I** appears to have TAS and FVS swapped relative to every other row (§5.1). Not resolved; flagged as suspect.
- **GridKit node/edge counts** come from MIND's Table 4 only. Zenodo ships GIS extracts, not edge lists, so they could not be self-counted.
- **CAIDA ARK** (GDM's `ARK201012_LCC`) is registration-gated; not counted.
- **`Wikibook`, `UAI`, `KKI`, `EconPoli`, `Route`, `Chicago`, `Europe`, `AirTraffic`, `Roget`, `Innovation`, `Genefusion`** — used by DCRS/SPR/NIRM, but their source URLs and node/edge counts were not run down. They are small and probably KONECT/networkrepository, but that is an assumption.
- **Whether `netscience` in the Ventresca weighted CNP set is our 1,589-node version** — the archive contains a file by that name; it was not diffed.
- **Publisher-blocked URLs**: Wiley (`net.20464`, `net.21622`, `net.21671`, `net.22200`), PNAS (`pnas.1009440108`, `pnas.1605083113`, `pnas.1806108116`), APS (`PhysRevLett.117.208301`, `PhysRevE.94.012305`), ACM (`3688671.3688762`) and ScienceDirect all return **403** to scripted requests. The pages resolve in a browser; the DOIs are correct; they simply could not be confirmed by `curl`.
- **`10.1109/TASE.2023.3286860`** (Tan et al., feature-importance-aware GAT for CND) resolves to a **404** at doi.org as of 2026-07-29. The paper is cited by its IEEE T-ASE record; the DOI itself is dead, so the row carries no working link.
- **`10.1109/TCCN.2024.3395174`** (Zhang et al. multitasking BICND) returned **404** — the DOI in §3.2 is reconstructed from the survey's citation and is **unverified**. Treat that row's link as broken.
- **CoreGDM, MultiDismantler, Tan 2024, AdaRisk** — papers located and cited, but paywalled; no result cells transcribed and no code located.
- **Ventresca's `wCNDP.zip` contents** were listed but not individually counted.

### Things we should measure ourselves rather than cite

- Whether a 3-layer GCN/SAGE can learn GCC membership on our graphs at all. §2.1 argues it cannot on receptive-field grounds; that is an argument, not a measurement, and it is cheap to check (`data/old/connectivity.py` gives labels for free).
- The Spearman correlation between our model's node ranking and its own `CH_DEGREE` channel (§9.5). MIND measured this for GDM and got 0.762. We have not measured it for ours.
- Whether NIRM's `UsPower` ρ of 8.58% and CoreHD's `Grid` 6.62% really differ only by reinsertion + protocol, or whether the graphs differ. Both cite 4,941/6,594, so they should be identical — one run of each would settle it.
