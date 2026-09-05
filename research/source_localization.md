# Source Localization: Prior Work, Datasets, and Published Results

Source localization is the **inverse** of the diffusion process this repo already models: given a graph `G` and an observed diffusion state `y`, recover the seed set `x` that produced it. It goes by several names, source detection, rumor source identification, patient-zero inference, the graph diffusion inverse problem, and the modern learning-based line (SL-VAE, IVGD, SL-Diff, DDMSL) is built by explicitly _learning a forward propagation model_ and then inverting it. That forward model is exactly what our world model already is.

**But supplying it is not a contribution.** SL-VAE states outright that its forward operator is pluggable and reports no significant difference across GAT, MONSTOR and DeepIS [verified, §4]; dropping ours into that slot is a swap the seed paper has already declared a no-op. The contribution has to be the **inversion procedure**, not the likelihood. §2.3 therefore formalizes source localization as an **amortized program-search** problem: the coding-agent outer loop searches the space of inversion _algorithms_, and the world model is the forward oracle those algorithms query. That formulation is what puts both of this project's loops to work on the task, and it lands in a cell of §1's taxonomy that is currently empty.

All URLs returned HTTP 200 on **2026-07-28** unless annotated otherwise.

---

## 0. Verification policy

| Tier           | Meaning                                                                                                              |
| -------------- | -------------------------------------------------------------------------------------------------------------------- |
| **[verified]** | Read from the paper's own table via text extraction, or from the data repository's own statistics page. Trustworthy. |
| **[derived]**  | Computed by us from a file we downloaded, or reconstructed arithmetically from published splits.                     |
| **[figure]**   | Read off a plotted figure: the paper published no table. Approximate, direction only.                               |
| **[claim]**    | Stated in prose by a paper or a secondary source; not cross-checked against a file or table.                         |

**Automated PDF summarizers hallucinate plausible numbers from these papers.** The source-localization tables are unusually easy to scramble because two of the central papers **swap the column order between adjacent tables**: SL-VAE's Table 1 is `RE · PR · F1 · AUC` and its Table 2 is `PR · RE · F1 · AUC` [verified]. A summarizer that assumes one ordering silently transposes precision and recall for an entire table. Every number below marked [verified] was read from `pdftotext -layout` output of the paper's own table.

**Edge-count convention.** Undirected graphs are quoted as _undirected edges_; directed graphs as _arcs_. Our loaders report `adjacency.nnz`, which for a symmetrized undirected graph is **2× the undirected edge count**. This single convention difference explains most apparent "discrepancies" between our numbers and published tables: check it before concluding two graphs differ.

**Average-degree convention is not stable in this literature.** IVGD's Table 2 and SL-VAE's Table 5 both report Karate as avg degree `2.294` (= `E/N`) but Jazz as `13.848` (`E/N`) and `27.69` (`2E/N`) respectively: i.e. the two papers use _different_ conventions for the same column, and SL-VAE's own table mixes both [derived]. Recompute `2E/N` yourself before comparing an avg-degree cell.

---

## 1. Task definition

Given a graph `G = (V, E)`, a diffusion model `M`, and an observed diffusion state `y ∈ {0,1}^{|V|}` (or a partial observation of it) at some time `T`, recover the source set `x ∈ {0,1}^{|V|}` that generated `y`:

```
x* = argmax_x  p(x | y, G)   ∝   p(y | x, G) · p(x)
```

The likelihood `p(y | x, G)` **is the forward diffusion operator**. Every method in this file differs in how it obtains that operator and how it inverts it:

| Family                                     | `p(y \| x, G)` obtained by                                  | Inversion                                          | Amortized? |
| ------------------------------------------ | ----------------------------------------------------------- | -------------------------------------------------- | ---------- |
| Centrality / combinatorial                 | assumed analytically (SI tree, BFS, SIR sample path)        | closed-form maximizer                              | yes     |
| Message passing                            | dynamic message passing / belief propagation recursions     | gradient or MAP over the recursion                 | no      |
| Label propagation (LPSI)                   | none: model-free convergence of a heat/label diffusion     | local maxima of the converged label field          | yes     |
| Supervised GNN (GCNSI, GIN-SD)             | never modelled: learned discriminatively `y → x`           | one forward pass, node-level binary classification | yes     |
| Invertible (IVGD)                          | learned GNN, made **invertible** by a residual construction | run the network backwards + validity projection    | yes     |
| Generative (SL-VAE, SL-Diff, DDMSL, SIDSL) | learned GNN forward model `p_ψ(y \| x, G)`                  | optimize `x` in a learned latent/denoising prior   | no      |
| **Program search (this project, §2.3)**    | **learned action-conditioned world model `f_θ`**            | **search the space of inversion _programs_ offline; run one program per instance** | yes **yes** |

**"Amortized" means the per-instance cost is a fixed evaluation, not an optimization loop.** A non-amortized method re-solves an optimization problem from scratch for every new cascade; an amortized one does its expensive work once and then applies the result. Read the last two columns together and one cell of the cross-product is empty in the published literature:

|                             | **Uses a forward model** | **No forward model**            |
| --------------------------- | ------------------------ | ------------------------------- |
| **Per-instance inversion**  | SL-VAE, IVGD¹, DDMSL, SIDSL, DMP, BP | none                    |
| **Amortized inference**     | **← nothing published**  | GCNSI, GIN-SD, LPSI, NETSLEUTH, OJC, centralities |

¹ IVGD is the near-miss: its inversion is a single backward pass, but the validity-aware projection layers are an unrolled per-instance optimization, so its inference cost still scales with an inner loop.

**The empty cell is the target.** The generative line pays an optimization loop per cascade to exploit a forward model; the amortized line runs in one pass but discards the forward model entirely, and it is the _weakest_ family in every table in §5 (GCNSI is the lowest-scoring learned method in §5.1, §5.2, §5.3 and §5.5 without exception). Nothing published is both amortized and forward-model-using. That combination is exactly the shape of this project's two loops: the outer loop produces the amortized artifact (a program mapping `(G, y) → x̂`, written once, executed on every instance), and the inner loop is the forward model that program calls while it runs. §2.3 formalizes it.

The problem is **ill-posed**: diffusion is many-to-one (many seed sets produce the same final state) and information-destroying (a saturated cascade retains almost no trace of its origin). Hence the field's two defining moves, impose a prior over plausible source sets, and evaluate with AUC rather than accuracy, because sources are ~1-10% of nodes and accuracy is dominated by the negatives.

### The variant axes

Any published number is meaningless without all five of these pinned down.

| Axis                | Options                                                                                      | Why it matters                                                                                                                                                                                                                            |
| ------------------- | -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Source count**    | single-source · multi-source (`k` known) · multi-source (`k` unknown)                        | Classical centrality methods (rumor centrality, Jordan centre) are **single-source** and rank all nodes; NETSLEUTH and everything learning-based are **multi-source**. A single-source method evaluated as multi-source scores near zero. |
| **Observation**     | full snapshot (every node's state at `T`) · partial / sparse observers · timestamped cascade | Pinto et al. and OJC are built for _sparse observers_; SL-VAE/IVGD/GCNSI assume a **full snapshot**.                                                                                                                                      |
| **Time**            | `T` known · `T` unknown · a single snapshot vs. the whole trajectory                         | DDMSL and DDMIX reconstruct the _whole_ diffusion path, not just `x`.                                                                                                                                                                     |
| **Diffusion model** | SI · SIR · SIS · IC · LT · "real cascade, unspecified"                                       | LPSI's selling point is being model-free; NETSLEUTH is SI-only; OJC is SIR-designed. SL-VAE's Table 6 lists exactly which baseline is admissible under which model [verified].                                                            |
| **Graph knowledge** | `G` fully known (the norm) · `G` partially observed                                          | GIN-SD (AAAI 2024) is the first to drop complete node observability.                                                                                                                                                                      |

### Why the task exists

Rumour and misinformation attribution, epidemic patient-zero tracing, computer virus / malware origin detection, power-grid disturbance localization, and water-network contamination sensing are all the same inverse problem. It is also the canonical stress test for a _learned_ forward model: if `f_θ` is a faithful simulator, inverting it should recover sources; if it is not, inversion fails loudly.

---

## 2. Fit with our methodology

**Status: cheapest task in this folder to add, and the only one where a cheap addition also happens to be a methodological contribution.** It needs _no new simulator, no new action op, and no new data generation run._ It is the only task here whose training data already exists on disk in exactly the form it requires.

The rest of this section is in two halves. §2.1 establishes that the _data_ is free. §2.2 onward establishes that the _framing_ has to change: the obvious use of our world model here is a component swap the seed paper calls interchangeable, and §2.3 replaces it with a formalization that puts both the inner-loop world model and the outer-loop coding agent to work.

### 2.1 Why the data is nearly free

| Requirement                               | Where it already exists                                                                                                                                                                                                                                                                                                                    |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| A learned forward operator `p(y \| x, G)` | **This is the world model.** SL-VAE, IVGD and SL-Diff each train one from scratch (SL-VAE plugs in GAT / MONSTOR / DeepIS; IVGD pre-trains a GNN with `pretrain.py` before `main.py`). Ours is `WorldModel.forward` with a `structured` head.                                                                                              |
| `(seed set, final state)` training pairs  | Already emitted. In `data/generate_wm_data.py::_episode_transitions`, the `t = 0` record has `state = State(infected=[], frontier=[])` and `action` = the **entire seed bag** as `add_node` ops; the episode's last record carries `next_state`. Grouping `transitions_<IC\|LT>_<split>.jsonl` by `episode_id` recovers `(x, y)` directly. |
| No interventions after `t = 0`            | `--inject-p 0` and omitting `--action-ops`. The transition then degenerates to `f_θ(G, s_t) → s_{t+1}`, which is the `T_exo = identity` special case we already train.                                                                                                                                                                     |
| Soft observation targets                  | `--mc-marginals` already writes `next_marginal_infected` / `next_marginal_frontier`, i.e. `P(node infected)`: a _continuous_ `y ∈ [0,1]^{\|V\|}`, which is precisely the input type SL-VAE assumes ("the value of `y` is continuous in the range [0,1] that fit Gaussian distribution" [verified]).                                       |
| The graphs                                | Five of SL-VAE's seven datasets are graphs we already load (§6.1).                                                                                                                                                                                                                                                                         |

### 2.2 What the published methods optimize, and why swapping the forward model is not a contribution

Every generative method in §4 solves the same per-instance problem. Given one observation $y$ on one graph $G$:

$$\hat{x} \;=\; \arg\max_{x} \; \underbrace{\log p_\psi(y \mid x, G)}_{\text{learned forward model}} \;+\; \underbrace{\log p(x)}_{\text{learned prior}}$$

SL-VAE relaxes $x$ to $\tilde{x} \in [0,1]^{|V|}$ and runs Adam on it; IVGD runs its network backwards and projects; DDMSL runs a reverse denoising chain. All three differ only in _how they invert_. The forward model $p_\psi$ is a component, and SL-VAE is explicit that it is a **replaceable** one: the paper plugs in GAT, MONSTOR and DeepIS and reports no significant difference between them [verified, §4 and Fig. 3 discussion]. IVGD's `pretrain.py` exists purely so `main.py` has something to invert.

**So "SL-VAE, but with our world model as $p_\psi$" is a paper the seed authors already wrote.** It is worth building exactly once, as an ablation arm (§2.6, arm A), because it isolates what the rest of the design buys. It is not the contribution.

Two properties of our world model do survive the comparison, and they are worth stating precisely so they are not oversold:

1. **It is mechanism-shaped, not a black-box regressor.** `ICTransmissionHead` predicts per-edge transmission $q(u \to v)$ and composes $p_{\text{new}}(v) = 1 - \prod_u (1 - q \cdot \text{frontier}_u)$. SL-VAE's GAT/MONSTOR/DeepIS forward models regress $y$ from $x$ directly with no cascade structure. Ours therefore cannot saturate by construction, which matters here because saturation destroys the information the inverse problem needs (§2.9).
2. **It is action-conditioned.** $f_\theta(G, s_t, a_t) \to s_{t+1}$ accepts interventions mid-cascade. No published forward model in this literature does. In the base formulation below this buys nothing (actions are `NULL` throughout), which is an honest weakness; §2.8 is the variant that cashes it in.

### 2.3 The formalization: amortized program inversion

The reframing is to **contribute the $\arg\max$ and let the likelihood be incidental**. Instead of optimizing a source vector per instance, search the space of _inversion programs_ once, offline, and run the winning program on every instance.

#### 2.3.1 The objective

Let $\mathcal{A}$ be the space of programs expressible in the coding agent's strategy contract, where each $A \in \mathcal{A}$ is a function

$$A : (G,\; y,\; k) \;\longmapsto\; \hat{x} \subseteq V, \qquad |\hat{x}| = k$$

and let $A_{f}$ denote the program $A$ with its forward-model primitive bound to a specific simulator $f$ (§2.4.3). Let $\mathcal{D}_{\text{train}} = \{(G^{(i)}, y^{(i)}, x^{(i)})\}$ be labelled episodes harvested from our generator. The outer loop solves

$$A^{*} \;=\; \arg\max_{A \in \mathcal{A}} \; \frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{i} F_1\!\left(A_{f_\theta}\!\left(G^{(i)}, y^{(i)}, k^{(i)}\right),\; x^{(i)}\right)$$

and at test time inference is a single execution of $A^{*}$. There is no per-instance optimization loop, and no gradient with respect to $x$ anywhere in the method.

**Three things changed relative to §2.2, and each one matters independently:**

| Change | From | To | Why it matters |
| ------ | ---- | -- | -------------- |
| **Decision variable** | a continuous relaxation $\tilde{x} \in [0,1]^{\|V\|}$, thresholded at the end | a discrete set $\hat{x}$, chosen directly | No relaxation gap. SL-VAE's recall runs far ahead of its precision on five of six graphs (§5.2: IVGD hits `RE = 1.0000` on five graphs and lets precision carry F1), which is the signature of a thresholded relaxation over-predicting. |
| **The prior $p(x)$** | a learned VAE latent density | **program code**: structural heuristics the agent writes and mutates | LPSI ("sources are local maxima of a converged label field"), the Jordan centre of each infected component, one-source-per-community, minimum-hop-separation constraints. These are the classical methods of §3, and the agent can compose, parameterize and hybridize them. A VAE prior cannot express "at most one source per 3-hop ball"; ten lines of Python can. |
| **What is optimized** | $\tilde{x}$, per instance, discarded afterwards | $A$, once, reused on every instance | This is the amortization, and it is the axis §1 shows is empty. |

#### 2.3.2 The three nested loops

The single most common confusion about this design is which loop pays what. There are three, and they are strictly nested:

| Loop                         | Iterates over                                                        | One iteration costs                                                                           | Runs                                               |
| ---------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| **Outer (the coding agent)** | candidate programs $A$                                               | one LLM generation, plus one evaluation sweep of $A$ over all of $\mathcal{D}_{\text{train}}$ | offline, once per (task, dataset, dynamics)        |
| **Middle (the executor)**    | instances $(G, y, x)$ in the sweep                                   | one execution of $A$                                                                          | $\|\mathcal{D}_{\text{train}}\|$ times per program |
| **Inner (the world model)**  | candidate source sets $\hat{x}$ tested _inside_ one execution of $A$ | one call to `predict_marginals`, i.e. one $T$-step rollout                                    | as many times as $A$ chooses                       |

Write $P$ for the number of programs the outer loop evaluates (generations × population), $M = |\mathcal{D}_{\text{train}}|$, and $C$ for the candidate evaluations one program performs on one instance. Total forward-simulation work during program search is

$$P \cdot M \cdot C \quad \text{rollouts.}$$

Under the `@monte_carlo` binding each of those is $R$ independent NDlib episodes of $T$ steps, giving $P \cdot M \cdot C \cdot R$ simulated episodes; under `@world_model` each is a single batched forward pass per timestep. With $P \approx 10^2$, $M \approx 10^2$ and $C \approx 10^2$ the product is already $10^6$ rollouts before the $R$ multiplier, which is why **nobody has run a program search over an inverse problem**: without a cheap forward model it is not merely slow, it is infeasible.

**That is the load-bearing claim, and it is a program-search-time claim, not an inference-time one.** Be precise about this. At inference our program still performs $C$ forward evaluations, while SL-VAE performs $I$ gradient steps; if $C \approx I$ there is no inference-time win, and asserting one without measuring it would be dishonest. What is unconditionally true is that the world model is what makes the _search_ possible at all. Any inference-time advantage is an empirical question, and §8.5 specifies how to measure it rather than assume it.

#### 2.3.3 Why the outer reward is ground-truth F1, not re-simulation error

The natural-looking reward is the re-simulation error $\lVert y - f_\theta(\hat{x}, G) \rVert^2$, since it needs no labels. **It is the wrong outer-loop signal, and the reason is the ill-posedness stated in §1.** Diffusion is many-to-one: distinct source sets $x_1 \neq x_2$ routinely satisfy $f_\theta(x_1) \approx f_\theta(x_2) \approx y$. Under re-simulation error those two are indistinguishable, so a program that systematically recovers the wrong member of an equivalence class scores as well as one that recovers the right member, and the outer loop gets no signal on the thing we actually care about.

**We have $x$, and the real-cascade literature does not.** Our episodes are generated by our own simulator, so every training instance carries its true source set. That asymmetry is what makes the search signal usable. The split is therefore:

| Signal | Where it is used | Needs labels? |
| ------ | ---------------- | ------------- |
| **F1 against the true source set**, averaged over training episodes | the outer loop's reward, i.e. program selection | yes |
| **Re-simulation error** $\lVert y - f_\theta(\hat{x}) \rVert^2$ | _inside_ a program, to rank its own candidate hypotheses | no |

The consequence is worth stating plainly, because it is the answer to the obvious referee objection: **labels are used to select the program, never to run it.** $A^{*}$ is a self-contained algorithm whose only inputs are $(G, y, k)$. It can therefore be deployed on real cascades where no ground truth exists, which is precisely the regime SIDSL identifies as the field's hard case (§5.5: generative methods are "nearly useless without enough real cascades", DDMSL's Android F1 `0.010` without pretraining [verified]).

#### 2.3.4 What a generated program actually looks like

Made concrete, because "the agent writes an algorithm" is otherwise hand-waving. A plausible mid-search candidate, using primitives that already exist in `coding_agent/tools/primitives.py` plus the one new binding of §2.4.3:

```python
def localize(graph, observation, budget):
    # Prior: sources sit at local maxima of a label field over the infected subgraph (LPSI's idea)
    infected = [v for v in range(graph.num_nodes) if observation[v] > 0.5]
    field = propagate_labels(graph, infected, alpha=0.85)
    communities = detect_communities(graph)

    # Propose: best local-maximum candidate per community, so sources cannot cluster
    proposals = []
    for members in group_by(communities, infected):
        proposals.append(max(members, key=lambda v: field[v]))

    # Test: rank proposals by how well re-simulating them reproduces the observation
    selected = []
    for _ in range(budget):
        best, best_error = None, float("inf")
        for candidate in proposals:
            if candidate in selected:
                continue
            predicted = predict_marginals(graph, selected + [candidate], "IC")
            error = float(((predicted - observation) ** 2).sum())
            if error < best_error:
                best, best_error = candidate, error

        selected.append(best)

    return selected
```

Three properties of this example generalize to the whole design. It is **interpretable**: the recovered sources come with a stated reason, which for forensic attribution and outbreak tracing is a substantive advantage over a latent vector. It is **degenerate-friendly**: a program is free to set $C = 0$ and never call the forward model at all, which means pure-LPSI is inside the search space and the outer loop can discover that the forward model is not helping. And its expensive line is the `predict_marginals` call inside a double loop, which is exactly the inner-loop cost accounting of §2.3.2.

### 2.4 The contract the agent writes

#### 2.4.1 The strategy method

The existing `Strategy` Protocol in `coding_agent/types.py` declares `plan_horizon(graph, budget, horizon) -> list[list[ActionOp]]` and `act(state, graph, timestep) -> list[ActionOp]`, and the executor validates that the method its outer-loop method needs is present. Source localization adds one more:

```python
def localize(self, graph: GraphInfo, observation: np.ndarray, budget: int | None) -> list[int]: ...
```

| Argument | Type | Meaning |
| -------- | ---- | ------- |
| `graph` | `GraphInfo` | the same read-only view IM strategies already receive: `edge_index`, `ic_probs`, `out_neighbors`, `degree`, plus the cached community labels |
| `observation` | `np.ndarray`, shape `(N,)` | $y$. Under our data this is the MC marginal $P(\text{infected at } T) \in [0,1]$, which is continuous and is precisely the input type SL-VAE assumes [verified, §2.1]. A binarized single draw is the harder variant (§2.9). |
| `budget` | `int \| None` | $k$, the source count. An `int` matches SL-VAE's given-$k$ convention; `None` is the inferred-$k$ variant, which is NETSLEUTH's MDL setting and a strictly harder problem. |
| **returns** | `list[int]` | the recovered source node ids |

**This changes nothing else in the loop.** The environment converts $\hat{x}$ into `[[ActionOp("add_node", v) for v in x̂]]` for any re-simulation it needs, which is the identical seed-commit bag the generator already writes at `t = 0` (§2.1), so `rollout()`, the refinement loop, the checkpointing and all four evaluators are reused unchanged.

#### 2.4.2 Scored mode

`ScoredStrategy` already exists for IM: it fixes `plan_horizon` as a greedy harness the agent cannot override and permits edits only to `score()` and `schedule()`, which forces the model to edit an algorithm's internals rather than emit free-form programs. The source-localization analogue is the same trick:

```python
def source_score(self, node: int, graph: GraphInfo, observation: np.ndarray, selected: tuple) -> float: ...
```

with a fixed top-$k$ harness. This is the constrained arm to reach for if free-form `localize` generation proves too loose to converge, and it makes the search space directly comparable to the classical methods, since LPSI, the Comin, Costa centralities and rumor centrality are all exactly node-scoring functions.

#### 2.4.3 The one new primitive, and its four bindings


> **Update 2026-09-04.** Generated programs are offline: the bindings described below are attached to canned library baselines only, and the four bindings now describe the HARNESS's oracle, which computes the reward and the feedback. The text below is the original design record.

`coding_agent/tools/primitives.py` already exposes `mc_simulate_spread(graph, seeds, diffusion_model, mc_runs, horizon, seed) -> float` to generated code. That is already $x \mapsto \sigma(x)$. Source localization needs the same call returning the per-node vector rather than its sum:

```python
def predict_marginals(graph, seeds, diffusion_model, horizon) -> np.ndarray:  # shape: (N,)
```

`Trajectory.final_marginals` in `coding_agent/envs/world_model_env.py` already computes exactly this quantity ("per-node `P(infected at end)` across the ensemble"), so the world-model binding is plumbing rather than new modelling. **The four experimental conditions are four bindings of this single name**, which is what keeps the arms honest: the generated program is byte-identical across arms and only its oracle changes.

| Binding | `predict_marginals` resolves to |
| ------- | ------------------------------- |
| `@native` | **absent from the namespace.** The program must be a pure structural heuristic. |
| `@monte_carlo` | NDlib, `mc_runs` draws, the sampling estimator |
| `@oracle` | NDlib, treated as ground truth |
| `@world_model` | $f_\theta$, one batched forward pass per timestep |

### 2.5 The mapping onto `f_θ(G, s_t, a_t) → s_{t+1}`

| Element | Source localization under the world model |
| ------- | ----------------------------------------- |
| **State** $s_t$ | `(infected, frontier)`, unchanged. The observation $y$ is `infected` at the terminal step. |
| **Action** $a_t$ | `NULL` for all $t > 0$. At $t = 0$ the seed commit _is_ the unknown being solved for. |
| **$T_{\text{exo}}$** | identity in the base formulation (no interventions); the probe operator in the adaptive variant (§2.8). |
| **$T_{\text{endo}}$** | the IC/LT step the structured head already learns. |
| **Role of $f_\theta$** | a **forward oracle called from inside generated code**, not a network being inverted. It is frozen, it is never differentiated through, and no gradient with respect to $x$ is ever formed. |
| **Decision variable** | none in the model. The search happens in program space; the world model only answers "what would this seed set produce?". |
| **Objective** | $\arg\max_{A} \mathbb{E}\left[F_1(A_{f_\theta}(G,y,k), x)\right]$ over programs, evaluated by the outer loop. |

The row that matters is the fifth. In §2.2's framing the world model is the thing being inverted, which makes it the method's centrepiece and makes the method someone else's. Here it is a subroutine, and the method is the search over programs that call it.

### 2.6 The six baseline conditions

The framing of §2.3 restores the project's standard comparison table, which the component-swap framing cannot support (with no agent there is no arm 3-6). Conditions 3 through 6 hold the generated program fixed and vary only the binding of §2.4.3.

| # | Arm | What it is | What it isolates |
| - | --- | ---------- | ---------------- |
| **1** | Pure algorithm | LPSI, NETSLEUTH, OJC, GCNSI via `pip install GraphSL` | the no-learning floor |
| **2** | GA routing | genetic routing over that fixed pool | is program _generation_ worth more than program _selection_? |
| **3** | agent `@native` | agent writes structural heuristics; no forward model available | **is a forward model in the search loop worth anything at all?** |
| **4** | agent `@monte_carlo` | same programs, NDlib sampling oracle | the cost axis: how much search fits in a fixed budget |
| **5** | agent `@oracle` | same programs, exact NDlib | the fidelity ceiling any learned oracle could reach |
| **6** | agent `@world_model` | same programs, $f_\theta$ | **ours** |
| **A** | ablation | §2.2's framing: freeze $f_\theta$, gradient-descend $\tilde{x}$, VAE prior | what program search buys over per-instance descent on the same likelihood |

Reading across is the experiment. **3 vs 6** answers whether the learned simulator contributes anything beyond structure. **4 vs 6** is the cost claim of §2.3.2 and is the reason the world model exists. **5 vs 6** bounds the loss from using a learned oracle instead of the true one. **6 vs A** is the methodological claim, and it is the one that distinguishes this from SL-VAE. **1 vs everything** is the bar, and §5.5 shows that bar is higher than it looks (§2.9).

External baselines (SL-VAE, IVGD, DDMSL, SIDSL) enter as published rows alongside arm 1 rather than as the thing we are a component of. That reversal is the whole point of the reframing.

### 2.7 What has to be built

Smaller than the §2.2 framing's estimate, because it reuses the entire agent loop rather than adding a parallel one. Nothing in `data/` changes.

| # | Piece | Where | Est. |
| - | ----- | ----- | ---- |
| 1 | `predict_marginals` primitive plus the four bindings | `coding_agent/tools/primitives.py`, both envs | ~60 lines, mostly plumbing; `Trajectory.final_marginals` exists |
| 2 | `localize` on the `Strategy` Protocol, and the executor's presence check | `coding_agent/types.py`, `executor.py` | ~30 lines |
| 3 | F1-against-true-sources reward, replacing final spread for this task | `coding_agent/methods/base.py` | ~40 lines. Note the outer loop hardcodes `argmax` in three places (`one_shot.py`, `evolve.py` ×2); F1 maximizes, so no sign work is needed here, unlike the containment tasks. |
| 4 | Episode regrouping: `(x, y)` pairs from `transitions_*.jsonl` by `episode_id` | `world_model/wm_data.py` | ~50 lines, §2.1 |
| 5 | `PR / RE / F1 / AUC` plus a true re-simulated error | `world_model/wm_eval.py` | ~40 lines |
| 6 | Prompt scaffolding: task statement, the `localize` contract, the primitive's signature | `coding_agent/prompts.py` | ~40 lines |
| 7 | **Arm A** (the ablation): frozen-model gradient descent on $\tilde{x}$, plus the VAE prior | built, then removed on 2026-09-04 with condition 8; `world_model/wm_sl.py` no longer exists | superseded by the label-free consistency reward, which arms 3-6 score directly |
| 8 | Registry entry: flip `source_localization` from `planned`, set `objective`, keep `action_ops = ()` | `pipeline/tasks.py` | ~10 lines |

Items 1-6 and 8 are the method; item 7 is the control it is measured against. **Multi-source is our default**, not an extension: our seed sets are $k = 1$, $20\%$ of $N$, squarely the multi-source regime (§8.2). A single-source arm needs a `--budget 1` generation run, which is cheap but changes which baselines are admissible, since rumor centrality and the Jordan centre only make sense there.

### 2.8 The adaptive variant: probing as an action

The base formulation exercises **zero action ops**, which means the one property that distinguishes our forward model from every published one (action-conditioning, §2.2 point 2) sits idle. The variant that cashes it in is **adaptive source localization**: instead of a full snapshot, the algorithm gets a budget of $k_{\text{probe}}$ node queries and must choose them sequentially.

```
localize_adaptive(graph, probe_budget, source_budget) -> list[int]
    repeat probe_budget times:
        v ← choose the next node to observe          ← the agent writes this policy
        observe y[v]
    return the inferred source set                   ← and this inference rule
```

The world model becomes **necessary rather than merely cheap**: scoring a candidate probe means predicting what that probe would reveal under each surviving source hypothesis, which is one forward simulation per hypothesis per candidate probe, nested inside the probe loop. Expected-information-gain planning of this shape is not affordable against a sampling simulator.

| Aspect | Base (§2.3) | Adaptive (§2.8) |
| ------ | ----------- | --------------- |
| Observation | full snapshot | $k_{\text{probe}}$ chosen nodes |
| Action ops used | none | a probe/query op, or a sixth op |
| World model's role | cheap scorer | required for information-gain planning |
| Published baselines | LPSI, NETSLEUTH, OJC, GCNSI, SL-VAE, IVGD | Pinto, Thiran, Vetterli (2012), OJC (2017); both **fixed** observer sets, neither adaptive |
| Comparable table | (key) §5.1, four byte-identical graphs | none |

**Build the base version first.** Adaptive is a generalization rather than a rewrite (full observation is the $k_{\text{probe}} = |V|$ special case), but it costs a query channel or a sixth op, a regeneration run with partial observations, and it forfeits the comparability that makes §5.1 worth having. It is where this goes if the base version clears the bar in §2.9.

### 2.9 Honest cost and honest risks

**Cost.** Roughly 2-4 days for items 1-6 and 8, plus 1-2 days for arm A. The increase over the §2.2 framing's "1-2 days" buys the ablation table of §2.6; the original estimate was for arm A alone, which is now the control rather than the method.

**Risk 1: LPSI is a higher bar than it looks, and it is the one that could kill this.** §5.5 has LPSI, a 2017 label-propagation method with no learning whatsoever, scoring F1 `0.544` on Digg and beating both SL-VAE (`0.479`) and DDMSL (`0.517`) [verified]. Any method that does not clear LPSI on every dataset has not cleared the bar. The mitigating structure is that LPSI is roughly fifteen lines and sits _inside_ the agent's expressible space, so the realistic floor is "the search rediscovers LPSI" and the live question is only whether forward-model-guided refinement improves on it. **Test that first**, before building items 4-7: run the search on one graph with a small $P$, and check whether any generated program beats a hand-written LPSI. If nothing does, the rest of the plan is not worth building.

**Risk 2: the zeroth-order objection.** A referee will say the agent is doing derivative-free optimization of the same objective SL-VAE descends analytically, and is therefore strictly worse per instance. The answer must be the amortization and must be stated first, not defensively: SL-VAE optimizes $\tilde{x}$ per cascade and discards it; the program is selected once and applied to every instance, and what transfers is the _search strategy_, not just the prior. If the transfer experiments of §8.5 fail, this objection lands and the method reduces to an expensive way of doing what SL-VAE does.

**Risk 3: the ill-posedness ceiling is unknown.** The Bayes-optimal F1 for this problem is below 1 and nobody has characterized it. A disappointing absolute number might reflect the problem rather than the method. Arm 5 (`@oracle`, exact NDlib, unlimited candidate budget) is the diagnostic: it upper-bounds what _any_ program in $\mathcal{A}$ can achieve with a perfect forward model, so the gap between arm 5 and arm 6 attributes error to the learned oracle, and the gap between arm 5 and 1.0 attributes it to the problem.

**Risk 4: the forward model's failure mode propagates, and asymmetrically.** A rollout that saturates (`ens_count_bias ≫ 0`) destroys source information, so the inverse problem gets _harder_ exactly when the forward model is wrong in the direction this project already fought (`wm-rollout-saturation-diagnosis-2026-06`). This is a risk and simultaneously the strongest secondary reason to build the task: source-localization F1 fails loudly where one-step `delta_f1`, dominated by unchanged nodes, stays comfortable. It is a sharp new regression test for a bug that has already bitten once.

**Risk 5: our observation is easier than the literature's.** Our marginals are averaged over `--mc-marginals` draws; SL-VAE observes a single binary realization. Ours is a strictly more informative $y$, so our F1 will be optimistic against §5.1 unless we also evaluate on a single binarized draw. **Report both**, and treat the binarized column as the comparable one.

**Risk 6: no published SL result under IC or LT.** SL-VAE uses SI/SIR; SL-Diff, SIDSL and DDMSL use real cascades; the GNN benchmark uses SIR. Our graphs match §5.1 byte-for-byte and our dynamics do not (§11). Either add SI/SIR to `data/wm_simulator.py`, which NDlib ships and `epidemic_control.md` also wants, or report ours as self-contained and say so. This is unchanged by the reframing and is the single biggest comparability gap in this file.

**Risk 7: the absolute numbers in §5.2 are not a realistic target.** IVGD's `FS ≈ 0.97` with `RE = 1.0000` on five of six graphs comes from a very permissive simulated setting with matched train/test distributions. The 2026 GNN benchmark's ~73% top-5 accuracy on a 34-node graph (§5.6) is the more honest picture of how hard this problem is.

---

## 3. Classical and heuristic methods

`S/M` = single- or multi-source. `Obs` = full snapshot (F), sparse observers (O), timestamped cascade (T).

| Method                                      | Year    | Venue                        | Model                | S/M   | Obs       | Idea                                                                                                                                                                                                                                                                         | Paper                                                                                                                                                                                                                                           | Code                                                                                                         |
| ------------------------------------------- | ------- | ---------------------------- | -------------------- | ----- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Rumor centrality**                        | 2010-11 | SIGMETRICS / IEEE TIT 57(8)  | SI                   | S     | F         | Count the distinct spreading orders consistent with the observed infected subtree; the node maximizing that count is the ML estimator on a regular tree. **The paper that founded the field.**                                                                               | [arXiv 0909.4370](https://arxiv.org/abs/0909.4370) · SIGMETRICS'10 [`10.1145/1811039.1811063`](https://dl.acm.org/doi/10.1145/1811039.1811063) (403)                                                                                            | no public code found (reimplemented in [cosasi](https://github.com/lmiconsulting/cosasi))                    |
| **Comin, Costa centrality suite**            | 2011    | Phys. Rev. E 84, 056105      | SI                   | S     | F         | Rank candidate origins by degree / betweenness / closeness / eigenvector centrality restricted to the infected subgraph: the cheap-heuristic baseline everything is compared against.                                                                                       | [`10.1103/PhysRevE.84.056105`](https://doi.org/10.1103/PhysRevE.84.056105)                                                                                                                                                                      | no public code found                                                                                         |
| **Pinto, Thiran, Vetterli**                   | 2012    | Phys. Rev. Lett. 109, 068702 | SI + Gaussian delays | S     | **O**     | Only a sparse subset of nodes report _arrival times_; a Gaussian ML estimator over observed delay differences localizes the source. The reference method for the sparse-observer regime.                                                                                     | [arXiv 1208.2534](https://arxiv.org/abs/1208.2534)                                                                                                                                                                                              | no public code found                                                                                         |
| **Fioriti, Chinnici (dynamical age)**        | 2012    | arXiv                        | SI                   | M     | F         | Spectral: remove nodes and measure the drop in the adjacency's largest eigenvalue ("dynamical age"); the highest-age nodes are the sources.                                                                                                                                  | [arXiv 1211.2333](https://arxiv.org/abs/1211.2333)                                                                                                                                                                                              | no public code found                                                                                         |
| **NETSLEUTH** (key)                            | 2012    | ICDM (+ KAIS 2014)           | SI                   | **M** | F         | Two-part **MDL**: encode the source set + the ripple that grows from it, and pick the `k` that minimizes total description length, so the _number_ of sources is inferred, not given. Sources are found via the eigenvector of the infected subgraph's submatrix Laplacian. | [ICDM PDF](https://people.cs.vt.edu/~badityap/papers/netsleuth-icdm12.pdf) · [mirror](https://eda.mmci.uni-saarland.de/pubs/2012/netsleuth-prakash,vreeken,faloutsos.pdf) · [KAIS](https://link.springer.com/article/10.1007/s10115-013-0671-5) | in [GraphSL](https://github.com/xianggebenben/GraphSL) and [cosasi](https://github.com/lmiconsulting/cosasi) |
| **Jordan centre / sample path**             | 2013-16 | ITA / IEEE-ACM ToN 24(1)     | **SIR**              | S     | F         | Maximize over _sample paths_ rather than over sources; the optimum is shown to be the **Jordan centre** (minimizer of eccentricity) of the infected subgraph, independent of the infection parameters.                                                                       | [arXiv 1206.5421](https://arxiv.org/abs/1206.5421) · [IEEE](https://ieeexplore.ieee.org/document/6414679)                                                                                                                                       | no public code found                                                                                         |
| **Effective distance (Brockmann, Helbing)**  | 2013    | Science 342, 1337            | metapopulation       | S     | F         | Replace geographic distance with `d(u→v) = 1 − log p(u→v)`; in that metric a complex global contagion becomes a **circular wave** whose centre is the origin.                                                                                                                | [`10.1126/science.1245200`](https://www.science.org/doi/10.1126/science.1245200) (403 to bots)                                                                                                                                                  | no public code found                                                                                         |
| **Belief propagation (Altarelli et al.)**   | 2014    | Phys. Rev. Lett. 112, 118701 | SIR                  | S/M   | F/partial | Full Bayesian posterior over the epidemic's initial condition via BP on a time-unrolled factor graph: the statistically principled reference point.                                                                                                                         | [arXiv 1307.6786](https://arxiv.org/abs/1307.6786)                                                                                                                                                                                              | [github.com/sibyl-team](https://github.com/sibyl-team) (successor codebase)                                  |
| **Dynamic message passing (Lokhov et al.)** | 2014    | Phys. Rev. E 90, 012801      | SIR                  | S     | F/partial | Approximate the SIR marginals with DMP equations (exact on trees), then score each candidate origin by the DMP likelihood of the observed snapshot. Handles unknown `T`.                                                                                                     | [arXiv 1303.5315](https://arxiv.org/abs/1303.5315)                                                                                                                                                                                              | no public code found                                                                                         |
| **LPSI** (key)                                 | 2017    | AAAI 31                      | **model-free**       | **M** | F         | Treat infected/uninfected as `±1` labels and run **label propagation** to convergence; local maxima of the converged field are the sources. Requires _no_ diffusion model: the reason it is the strongest classical baseline in every learning-based paper's table.         | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/10731)                                                                                                                                                                                  | in [GraphSL](https://github.com/xianggebenben/GraphSL)                                                       |
| **OJC**                                     | 2017    | AAAI 31                      | SIR                  | **M** | **O**     | "Optimal-Jordan-Cover": cover the observed infected nodes with candidate balls, then pick the Jordan centre of the cover. Designed for **partial observation** and provably optimal on tree-like graphs.                                                                     | [arXiv 1611.06963](https://arxiv.org/abs/1611.06963)                                                                                                                                                                                            | in [GraphSL](https://github.com/xianggebenben/GraphSL)                                                       |

**The survey to cite:** Jiang, Wen, Yu, Xiang & Zhou, _"Identifying Propagation Sources in Networks: State-of-the-Art and Comparative Studies"_, **IEEE Communications Surveys & Tutorials 19(1):465-481, 2017**, [free PDF](https://nsclab.org/nsclab/esi/comst_jiang2017.pdf) · [IEEE](https://ieeexplore.ieee.org/document/7582484). It taxonomizes the pre-deep-learning field into single-source vs. multi-source and by observation model, and runs the comparative study the individual papers do not.

**Two reusable classical implementations exist** and are worth taking rather than rewriting: [**GraphSL**](https://github.com/xianggebenben/GraphSL) (JOSS 2024, `pip install GraphSL`) ships LPSI, NETSLEUTH, OJC and the GNN methods behind one `Metric` object returning accuracy/precision/recall/F1/AUC; and [**cosasi**](https://github.com/lmiconsulting/cosasi) ([JOSS 2022](https://joss.theoj.org/papers/10.21105/joss.04894), `pip install cosasi`) ships the centrality-style classical estimators.

---

## 4. Learning-based methods

| Method                           | Year | Venue                            | Approach                                                                                                                                                                                                                                                                                                                       | Forward model?                                          | Paper                                                                                                                                                                                      | Code                                                                                  |
| -------------------------------- | ---- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| **GCNSI**                        | 2019 | CIKM                             | The first GNN for this task. Build a multi-order LPSI-style input feature per node, then a spectral GCN classifies each node source / not-source. Model-free like LPSI but supervised. **Every later paper's weakest learned baseline.**                                                                                       | no                                                      | [ACM](https://dl.acm.org/doi/10.1145/3357384.3357994) (403 to bots)                                                                                                                        | in [GraphSL](https://github.com/xianggebenben/GraphSL)                                |
| **IVGD** (key)                      | 2022 | WWW                              | Make a _pretrained_ graph diffusion GNN **invertible** via a graph-residual construction with a contraction guarantee, then run it backwards; an error-compensation module offsets inversion error and **validity-aware layers** project the result onto the feasible source set by unrolled optimization.                     | **yes**: `pretrain.py` trains it, `main.py` inverts it | [arXiv 2206.09214](https://arxiv.org/abs/2206.09214) · [Emory PDF](http://cs.emory.edu/~lzhao41/materials/papers/3485447.3512155.pdf)                                                      | [xianggebenben/IVGD](https://github.com/xianggebenben/IVGD)                           |
| **SL-VAE** (key)                    | 2022 | **KDD**                          | The seed paper. Learn `p_ψ(y\|x,G)` with any diffusion estimator (GAT / MONSTOR / DeepIS are all tried), learn a **generative prior** `p_θ(x\|z) q_φ(z\|x̂)` over observed source sets, then infer `x̃` by minimizing `‖y − p_ψ(y\|x̃,G)‖² − log Σ_z p_θ(x̃\|z)q_φ(z\|x̂)`. Claims **+20% AUC on average** over prior SOTA [claim]. | **yes: explicitly pluggable**                          | [arXiv 2206.12327](https://arxiv.org/abs/2206.12327) · [KDD](https://dl.acm.org/doi/10.1145/3534678.3539288) (403)                                                                         | [triplej0079/SLVAE](https://github.com/triplej0079/SLVAE)                             |
| **SL-Diff**                      | 2023 | ECML-PKDD                        | Two-stage denoising diffusion. A **coarse** stage regresses "source proximity degrees" to initialize; a **fine** stage runs a conditional score-based diffusion (multi-head graph-attention score net) over the graph, which also quantifies dissemination uncertainty. Evaluated on real cascades only.                       | partial (proximity prior)                               | [arXiv 2304.08841](https://arxiv.org/abs/2304.08841) · [Springer](https://link.springer.com/chapter/10.1007/978-3-031-43418-1_20)                                                          | no public code found                                                                  |
| **DDMSL** (key)                     | 2023 | **NeurIPS**                      | The second seed paper. Model forward diffusion as a **Markov chain in discrete space**, then invert it with a reversible-residual denoising diffusion model. Recovers not just `x` but the **whole diffusion path** `s_0…s_T`, with theoretical guarantees.                                                                    | **yes: the Markov chain is the forward model**         | [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/46ab9d9645b6975b947231ddb48da1ab-Paper-Conference.pdf) · [OpenReview](https://openreview.net/forum?id=5Fr8Nwi5KF) | no public code found                                                                  |
| **DDMIX**                        | 2023 |                                  | VAE over node states at **every** time step, reconstructing the dissemination path. Accuracy degrades as `T` grows because the solution space explodes.                                                                                                                                                                        | yes (VAE)                                               | cited by SL-Diff and SIDSL; **no primary URL established**                                                                                                                                 | no public code found                                                                  |
| **GIN-SD**                       | 2024 | **AAAI 38**                      | First to drop the complete-observation assumption: **incomplete nodes**. Positional encoding marks which nodes are unobserved; self-attention weights nodes by transmission capacity; a class-balancing term fixes the source/non-source imbalance.                                                                            | no                                                      | [arXiv 2403.00014](https://arxiv.org/abs/2403.00014) · [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/27755)                                                                      | no public code found                                                                  |
| **CNSL**                         | 2024 | (Emory)                          | **Cross-network** source localization: diffusion crosses from an unobserved source network into an observed one; learns network-specific propagation with joint inference.                                                                                                                                                     | yes                                                     | [arXiv 2404.14668](https://arxiv.org/abs/2404.14668)                                                                                                                                       | [tanmoysr/CNSL](https://github.com/tanmoysr/CNSL)                                     |
| **Graph contrastive SL**         | 2024 | Information Sciences             | Contrastive pretraining to fix GNN over-smoothing and the purely-local receptive field in source detection.                                                                                                                                                                                                                    | no                                                      | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0020025524010041) (403 to bots)                                                                                     | no public code found                                                                  |
| **PGSL**                         | 2024 | Expert Systems with Applications | Probabilistic graph diffusion model for source localization.                                                                                                                                                                                                                                                                   | yes                                                     | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417423025307)                                                                                                   | no public code found                                                                  |
| **SIDSL**                        | 2025 | arXiv (v3)                       | Structure-**prior** informed diffusion model for the **limited-data** regime: injects topology-aware propagation priors so the model trains from few labelled cascades.                                                                                                                                                        | yes                                                     | [arXiv 2502.17928](https://arxiv.org/abs/2502.17928)                                                                                                                                       | no public code found                                                                  |
| **DIPT**                         | 2025 | (Emory)                          | "Deep Identification of Propagation Trees": recovers the _tree_ of who-infected-whom, not just the seed set. Same lab as SL-VAE and DeepIM.                                                                                                                                                                                   | yes                                                     | [arXiv 2503.00646](https://arxiv.org/abs/2503.00646)                                                                                                                                       | no public code found                                                                  |
| **LLM-advisor**                  | 2025 | **IJCAI 34**                     | Uses an LLM to embed the _semantics_ of rumour-cascade comments, contrastively separated from non-rumour comments, as an extra signal guiding source inference. First LLM-in-the-loop SL method.                                                                                                                               | no                                                      | [IJCAI PDF](https://www.ijcai.org/proceedings/2025/0326.pdf)                                                                                                                               | no public code found                                                                  |
| **Learnable-dynamics framework** | 2025 | **IJCAI 34**                     | A generalized diffusion framework whose propagation dynamics are _learned_ rather than assumed.                                                                                                                                                                                                                                | **yes**                                                 | [IJCAI](https://dl.acm.org/doi/abs/10.24963/ijcai.2025/325) (403 to bots)                                                                                                                  | no public code found                                                                  |
| **HyperDet**                     | 2025 | arXiv                            | Source detection in **hypergraphs** via interactive relationship construction + feature-rich attention fusion.                                                                                                                                                                                                                 | no                                                      | [arXiv 2505.12894](https://arxiv.org/abs/2505.12894)                                                                                                                                       | no public code found                                                                  |
| **SourceDetMamba**               | 2025 | **IJCAI 34**                     | Source detection in **sequential hypergraphs** with a graph-aware state-space (Mamba) model: reversed-order snapshots are fed to the SSM so temporal dynamics are captured before topology.                                                                                                                                   | no                                                      | [arXiv 2505.12910](https://arxiv.org/abs/2505.12910)                                                                                                                                       | no public code found                                                                  |
| **PDSL**                         | 2026 | arXiv (TNSE format)              | Propagation-dynamics-aware framework. Infers `ŝ = argmax p_ψ(Y_T\|s*,Y_t,G) · p_φ(s*\|z̄,Y_t,G) · p(z̄)`: a world-model-shaped factorization, and the closest published framing to ours.                                                                                                                                        | **yes**                                                 | [arXiv 2605.03550](https://arxiv.org/abs/2605.03550)                                                                                                                                       | [MrYansong/PDSL](https://github.com/MrYansong/PDSL)                                   |
| **GNN SD benchmark** (key)          | 2026 | arXiv (v2)                       | Not a method: an independent **review + benchmark** of GNN source detection under SIR on six contact networks, against Jordan centre, betweenness, SME and MCMF. The only reproducible third-party evaluation found in this literature.                                                                                       | n/a                                                     | [arXiv 2512.20657](https://arxiv.org/abs/2512.20657)                                                                                                                                       | [martinSter/gnn-source-detection](https://github.com/martinSter/gnn-source-detection) |

**Libraries.** [GraphSL](https://github.com/xianggebenben/GraphSL) (JOSS 9(99):6796 2024, [arXiv 2405.03724](https://arxiv.org/abs/2405.03724), [JOSS](https://joss.theoj.org/papers/10.21105/joss.06796)) is the one to start from: it ships LPSI, NETSLEUTH, OJC, GCNSI, IVGD and SL-VAE behind a common API returning accuracy / precision / recall / F1 / AUC, with **the six benchmark graphs already packaged**, Karate, Dolphins, Jazz, Network Science, Cora-ML, Power Grid: four of our loaders plus Karate and Dolphins.

---

## 5. Published results

### 5.1 SL-VAE (KDD 2022) (key) the most comparable table

**Why this is the (key) table.** Five of its seven graphs are graphs we load (Jazz, Cora-ML, Power Grid, Network Science, plus Karate, which we generate with `nx.karate_club_graph()`), and its protocol is one we can reproduce exactly: **10% of nodes chosen uniformly at random as sources**, simulate SI or SIR for **200 iterations to convergence**, `S` and `R` both counted as uninfected (`y = 0`), everything else `y = 1`; **10 repeats**, average reported [verified].

> **Column-order warning.** The paper's Table 1 is ordered `RE · PR · F1 · AUC` and its Table 2 is `PR · RE · F1 · AUC` [verified]. Both are **normalized to `PR · RE · F1 · AUC` below**, so these tables do not read left-to-right the same as the PDF. Re-check against the PDF before quoting.

#### SI diffusion [verified, Table 1: reordered to PR · RE · F1 · AUC]

| Method     | Jazz PR    | RE         | F1         | AUC        | Cora-ML PR | RE         | F1         | AUC        | PowerGrid PR | RE         | F1         | AUC        |
| ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ------------ | ---------- | ---------- | ---------- |
| LPSI       | 0.1054     | 0.4789     | 0.1716     | 0.4841     | 0.1556     | 0.5954     | 0.2466     | 0.6675     | 0.4546       | 0.4953     | 0.4737     | 0.9337     |
| GCNSI      | 0.1589     | 0.4368     | 0.2329     | 0.6428     | 0.1182     | 0.3619     | 0.1780     | 0.5383     | 0.1413       | 0.3477     | 0.2099     | 0.5044     |
| OJC        | 0.1005     | 0.1798     | 0.1289     | 0.5045     | 0.2036     | 0.2239     | 0.2133     | 0.5633     | 0.1044       | 0.2871     | 0.1531     | 0.5011     |
| NetSleuth  | 0.1087     | 0.1315     | 0.1191     | 0.5432     | 0.2647     | 0.2647     | 0.2647     | 0.4688     | 0.4975       | 0.5972     | 0.5428     | 0.7651     |
| **SL-VAE** | **0.7193** | **0.9474** | **0.8182** | **0.9777** | **0.6717** | **0.9466** | **0.7858** | **0.9582** | **0.6648**   | **0.9636** | **0.7868** | **0.9636** |

| Method     | Karate PR  | RE         | F1         | AUC        | NetSci PR  | RE         | F1         | AUC        |
| ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| LPSI       | 0.1861     | 0.4667     | 0.2855     | 0.6344     | 0.4231     | 0.6044     | 0.4978     | 0.8378     |
| GCNSI      | 0.1999     | 0.4333     | 0.2613     | 0.6022     | 0.1375     | 0.2247     | 0.1706     | 0.4759     |
| OJC        | 0.2708     | 0.3611     | 0.3095     | 0.6335     | 0.3708     | 0.1233     | 0.1851     | 0.5331     |
| NetSleuth  | 0.3333     | 0.3333     | 0.3333     | 0.4355     | 0.3283     | 0.3948     | 0.3585     | 0.6528     |
| **SL-VAE** | **0.6667** | **0.6667** | **0.6667** | **0.8172** | **0.6738** | **0.9937** | **0.8031** | **0.9705** |

#### SIR diffusion [verified, Table 2: native order PR · RE · F1 · AUC]

NetSleuth is **absent by design**: it is SI-only, which SL-VAE's Table 6 states explicitly [verified].

| Method     | Jazz PR    | RE         | F1         | AUC        | Cora-ML PR | RE         | F1         | AUC        | PowerGrid PR | RE         | F1         | AUC        |
| ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ------------ | ---------- | ---------- | ---------- |
| LPSI       | 0.1153     | 0.3632     | 0.1698     | 0.5005     | 0.1072     | 0.4779     | 0.1752     | 0.4986     | 0.4865       | 0.4721     | 0.4784     | 0.5821     |
| GCNSI      | 0.1419     | 0.3737     | 0.2055     | 0.6411     | 0.1158     | 0.3381     | 0.1725     | 0.5321     | 0.1133       | 0.2371     | 0.1533     | 0.5038     |
| OJC        | 0.1543     | 0.2201     | 0.1814     | 0.5012     | 0.1414     | 0.1679     | 0.1535     | 0.5110     | 0.1414       | 0.1679     | 0.1535     | 0.5009     |
| **SL-VAE** | **0.6667** | **0.8421** | **0.7442** | **0.9749** | **0.6695** | **0.5623** | **0.6112** | **0.9686** | **0.6846**   | **0.6458** | **0.6646** | **0.9689** |

| Method     | Karate PR  | RE         | F1         | AUC        | NetSci PR  | RE         | F1         | AUC        |
| ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| LPSI       | 0.1284     | 0.4167     | 0.1936     | 0.5144     | 0.1362     | 0.4326     | 0.2072     | 0.5614     |
| GCNSI      | 0.0721     | 0.1167     | 0.0879     | 0.5036     | 0.1047     | 0.3511     | 0.1613     | 0.5433     |
| OJC        | 0.3750     | 0.1944     | 0.2500     | 0.5771     | 0.3977     | 0.1231     | 0.1800     | 0.5097     |
| **SL-VAE** | **0.6250** | **0.8333** | **0.7143** | **0.8289** | **0.6541** | **0.5713** | **0.6099** | **0.9711** |

#### Real cascades [verified, Table 3: native order PR · RE · F1 · AUC]

Sources = top **5%** of nodes by infection time; observations = bottom 30% [verified]. `Digg-7556` and `Memetracker-7884` are random subsamples of the full graphs, used to show scaling.

| Method     | Digg-7556 PR | RE         | F1         | AUC        | Digg PR    | RE         | F1         | AUC        | Meme-7884 PR | RE         | F1         | AUC        | Memetracker PR | RE         | F1         | AUC        |
| ---------- | ------------ | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ------------ | ---------- | ---------- | ---------- | -------------- | ---------- | ---------- | ---------- |
| LPSI       | 0.0026       | 0.0123     | 0.0043     | 0.4432     | 0.0079     | 0.2727     | 0.0155     | 0.5618     | 0.0132       | 0.3184     | 0.0253     | 0.5112     | 0.0087         | 0.2913     | 0.0169     | 0.5377     |
| GCNSI      | 0.0114       | 0.3700     | 0.0221     | 0.4450     | 0.0123     | 0.2100     | 0.0232     | 0.4129     | 0.0211       | 0.3219     | 0.0396     | 0.4357     | 0.0197         | 0.2342     | 0.0363     | 0.4103     |
| OJC        | 0.0118       | 0.0107     | 0.0112     | 0.5023     | 0.0635     | 0.0696     | 0.0664     | 0.5142     | 0.0542       | 0.0433     | 0.0481     | 0.4812     | 0.0331         | 0.0207     | 0.0255     | 0.5077     |
| **SL-VAE** | **0.4131**   | **0.6217** | **0.4655** | **0.5541** | **0.4297** | **0.5421** | **0.4792** | **0.6213** | **0.5113**   | **0.6214** | **0.5610** | **0.5954** | **0.4612**     | **0.5181** | **0.4880** | **0.6245** |

#### Ablation [verified, Table 4]: the number that matters most to us

`SL-VAE (a)` = forward model + initialization only, **no generative prior**; `SL-VAE (b)` = intermediate; `SL-VAE` = full.

| Variant    | Jazz F1    | AUC        | Cora-ML F1 | AUC        | PowerGrid F1 | AUC        | Karate F1  | AUC        | NetSci F1  | AUC        |
| ---------- | ---------- | ---------- | ---------- | ---------- | ------------ | ---------- | ---------- | ---------- | ---------- | ---------- |
| SL-VAE (a) | 0.6254     | 0.8763     | 0.5180     | 0.8868     | 0.6692       | 0.9021     | 0.4936     | 0.7123     | 0.5459     | 0.9112     |
| SL-VAE (b) | 0.8072     | 0.9542     | 0.7589     | 0.9374     | 0.7124       | 0.9102     | 0.5653     | 0.7329     | 0.7264     | 0.9512     |
| **SL-VAE** | **0.8182** | **0.9777** | **0.7854** | **0.9582** | **0.7868**   | **0.9636** | **0.6667** | **0.8172** | **0.8031** | **0.9705** |

**Read this ablation as our roadmap.** `SL-VAE (a)` (a learned forward model plus gradient descent on a relaxed source vector, _no prior_) already beats every classical baseline on Jazz (`0.6254` vs LPSI's `0.1716`), Cora-ML and Karate. That is §2.3 step 1, and it is the milestone to aim at first.

> Two internal inconsistencies in the paper, for the record: Table 1 gives Cora-ML SL-VAE F1 = `0.7858` while Table 4 gives `0.7854`; and Table 5 reports Karate's average degree as `2.294` (`E/N`) while every other row uses `2E/N` [derived]. Neither affects the ranking.

### 5.2 IVGD (WWW 2022): same graphs, wildly different numbers

IVGD reports on **the same five graphs plus Dolphins**, and its numbers are far higher than SL-VAE's on the identical graph names. **They are not comparable**: IVGD trains and tests on _simulated_ diffusion with a different (unstated seed fraction) protocol, and (the load-bearing difference) **IVGD's Network Science is 1,589 / 2,742 while SL-VAE's is 1,565 / 13,532** (§6.4). Two papers from the same lab, two different graphs, one name.

Columns are native `ACC · PR · RE · FS` [verified, Table 3].

| Method    | Karate ACC | PR         | RE         | FS         | Dolphins ACC | PR         | RE         | FS         |
| --------- | ---------- | ---------- | ---------- | ---------- | ------------ | ---------- | ---------- | ---------- |
| LPSI      | 0.9559     | 0.6800     | 1.0000     | 0.7970     | 0.9677       | 0.7790     | 1.0000     | 0.8717     |
| NetSleuth | 0.9147     | 0.5371     | 0.6833     | 0.5965     | 0.9306       | 0.6454     | 0.7425     | 0.6904     |
| GCNSI     | 0.7088     | 0.1150     | 0.2667     | 0.1581     | 0.6177       | 0.1015     | 0.2548     | 0.1372     |
| **IVGD**  | **0.9853** | **0.8717** | **1.0000** | **0.9213** | **0.9935**   | **0.9444** | **1.0000** | **0.9701** |

| Method    | NetSci ACC | PR         | RE         | FS         | Cora-ML ACC | PR         | RE         | FS         |
| --------- | ---------- | ---------- | ---------- | ---------- | ----------- | ---------- | ---------- | ---------- |
| LPSI      | 0.9831     | 0.8525     | 1.0000     | 0.9202     | 0.9011      | 0.5067     | 0.9993     | 0.6724     |
| NetSleuth | 0.9595     | 0.7642     | 0.8429     | 0.8016     | 0.8229      | 0.1627     | 0.1793     | 0.1706     |
| GCNSI     | 0.8840     | 0.0582     | 0.0135     | 0.0218     | 0.8580      | 0.0970     | 0.0478     | 0.0637     |
| **IVGD**  | **0.9946** | **0.9476** | **1.0000** | **0.9730** | **0.9973**  | **0.9744** | **1.0000** | **0.9870** |

| Method    | Jazz ACC   | PR         | RE         | FS         | PowerGrid ACC | PR         | RE         | FS         |
| --------- | ---------- | ---------- | ---------- | ---------- | ------------- | ---------- | ---------- | ---------- |
| LPSI      | 0.9035     | 0.6074     | 0.9944     | 0.7371     | 0.9673        | 0.7584     | 1.0000     | 0.8624     |
| NetSleuth | 0.9222     | 0.5904     | 0.6629     | 0.6245     | 0.9276        | 0.6347     | 0.6986     | 0.6651     |
| GCNSI     | 0.7525     | 0.0685     | 0.1280     | 0.0849     | 0.7125        | 0.1022     | 0.2285     | 0.1410     |
| **IVGD**  | **0.9980** | **0.9802** | **1.0000** | **0.9899** | **0.9902**    | **0.9133** | **1.0000** | **0.9546** |

**The `ACC` column is the trap this literature sets.** GCNSI scores `ACC 0.8840` on Network Science with `FS 0.0218`: i.e. it is 88% accurate while finding essentially no sources, because sources are a tiny minority class. Report F1 and AUC; report accuracy only alongside them.

**IVGD's recall is `1.0000` on five of six graphs**, which means its validity-aware projection is tuned to over-predict and let precision carry the F1. Worth knowing before treating `FS ≈ 0.97` as a ceiling.

Runtime, simulations, seconds [verified, Table 5, partial extraction]: LPSI takes `0.26 / 0.27 / 0.76` s on Karate / Dolphins / Jazz and `52.83 / 240.88 / 899.45 / 94,541.13` s on the remaining columns (Network Science / Cora-ML / Power Grid / Deezer, **column assignment inferred from the table header order, not directly readable in the extracted text**, so treat the four large values as [claim]).

### 5.3 SL-Diff (ECML-PKDD 2023): real cascades only

SL-Diff evaluates on **five real-cascade datasets and none of the topology-only graphs**: Digg, Memetracker, Android, Christianity, Twitter. Sources are defined as the **first 5% of nodes by infection time** within a cascade; all cascade nodes are the observation; split `2:2:6` train/val/test [verified].

This is the only published table that puts **SL-VAE and IVGD side by side under one protocol**, which makes it the second-most useful table in this file.

Native order `RE · PR · F1 · ACC` [verified, Table 1]:

| Dataset      | Metric | NetSleuth | OJC    | LPSI       | GCNSI  | IVGD   | SL-VAE     | **SL-Diff** |
| ------------ | ------ | --------- | ------ | ---------- | ------ | ------ | ---------- | ----------- |
| Digg         | RE     | 0.0142    | 0.0781 | 0.2352     | 0.0135 | 0.2310 | 0.5420     | **0.7813**  |
| Digg         | PR     | 0.0023    | 0.0554 | 0.0072     | 0.2369 | 0.1397 | 0.4216     | **0.5839**  |
| Digg         | F1     | 0.0040    | 0.0648 | 0.0140     | 0.0255 | 0.1741 | 0.4743     | **0.6683**  |
| Digg         | ACC    | 0.7714    | 0.9035 | 0.9531     | 0.8064 | 0.9327 | 0.9742     | **0.9824**  |
| Memetracker  | RE     | 0.0647    | 0.0256 | 0.3047     | 0.2953 | 0.5954 | 0.5010     | **0.6902**  |
| Memetracker  | PR     | 0.0247    | 0.0360 | 0.1145     | 0.0172 | 0.1556 | 0.4592     | **0.4721**  |
| Memetracker  | F1     | 0.0358    | 0.0299 | 0.1665     | 0.0325 | 0.2467 | 0.4792     | **0.5607**  |
| Memetracker  | ACC    | 0.5688    | 0.6675 | 0.9174     | 0.8428 | 0.8947 | 0.9420     | **0.9562**  |
| Android      | RE     | 0.3172    | 0.1401 | 0.3407     | 0.7434 | 0.7253 | 0.6261     | **0.8260**  |
| Android      | PR     | 0.0422    | 0.0610 | 0.2323     | 0.3024 | 0.4105 | 0.5284     | **0.5945**  |
| Android      | F1     | 0.0745    | 0.0850 | 0.2762     | 0.4299 | 0.5243 | 0.5731     | **0.6914**  |
| Android      | ACC    | 0.6215    | 0.8337 | 0.9404     | 0.8211 | 0.9530 | 0.9245     | **0.9937**  |
| Christianity | RE     | 0.2491    | 0.3478 | 0.5309     | 0.7294 | 0.6433 | 0.8011     | **0.8352**  |
| Christianity | PR     | 0.1184    | 0.2823 | **0.6249** | 0.2300 | 0.5202 | 0.4894     | 0.5120      |
| Christianity | F1     | 0.1605    | 0.3116 | 0.5741     | 0.3497 | 0.5752 | 0.6076     | **0.6348**  |
| Christianity | ACC    | 0.7140    | 0.9304 | 0.9122     | 0.9673 | 0.9781 | 0.9529     | **0.9818**  |
| Twitter      | RE     | 0.0184    | 0.0154 | 0.2091     | 0.3770 | 0.6219 | 0.3273     | **0.9037**  |
| Twitter      | PR     | 0.0021    | 0.0238 | 0.1295     | 0.3719 | 0.4427 | 0.4210     | **0.7839**  |
| Twitter      | F1     | 0.0038    | 0.0187 | 0.1599     | 0.3744 | 0.5172 | 0.3683     | **0.8395**  |
| Twitter      | ACC    | 0.6348    | 0.8358 | 0.9149     | 0.9381 | 0.9027 | **0.9630** | **0.9630**  |

**The headline for us:** on real cascades every classical method collapses (NetSleuth F1 `0.0038-0.1605`, OJC `0.0187-0.3116`), and **the ranking flips relative to §5.2**, IVGD beats SL-VAE on Memetracker and Twitter but loses on Digg, Android and Christianity. There is no stable SOTA across protocols.

### 5.4 DDMSL (NeurIPS 2023): Warning: table not transcribed

**The second seed paper's own result table is NOT in this file.** The NeurIPS proceedings PDF is **35.5 MB** and every download attempt (four, including `curl -C -` resumption over ~20 minutes) truncated at ≈2.8 MB, which `pdftotext` recovers only 56 lines from. The OpenReview mirror (`openreview.net/pdf?id=5Fr8Nwi5KF`) returns a 12 KB JavaScript shell, not a PDF. **Rather than guess, the numbers are omitted.** What is established:

- Title: _"Diffusion Model for Graph Inverse Problems: Towards Effective Source Localization on Complex Networks"_, NeurIPS 2023 ([proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/46ab9d9645b6975b947231ddb48da1ab-Abstract-Conference.html) · [OpenReview](https://openreview.net/forum?id=5Fr8Nwi5KF) · [poster](https://neurips.cc/virtual/2023/poster/72813)).
- Method: forward diffusion as a discrete-space Markov chain, inverted by a reversible residual denoising-diffusion network; recovers the **whole diffusion path**, not only the seed set; **five real-world datasets** [claim].
- No public code repository was found.

DDMSL's numbers _as re-run by a third party_ are available and are transcribed in §5.5 below.

### 5.5 SIDSL (2025): the current cascade-protocol table, incl. DDMSL

SIDSL re-runs NetSleuth, LPSI, GCNSI, TGASI, SL-VAE and **DDMSL** on four real-cascade datasets. Sources = **top 10%** of nodes by influence time, observations = top 30% [verified]: note this is _not_ SL-Diff's 5%/100% convention, so §5.3 and §5.5 are not comparable to each other either.

Native order `F1 · RE · PR` [verified, Table 1, "without pretraining"]:

| Method    | Digg F1   | RE        | PR        | Twitter F1 | RE        | PR        | Android F1 | RE        | PR        | Christianity F1 | RE        | PR        |
| --------- | --------- | --------- | --------- | ---------- | --------- | --------- | ---------- | --------- | --------- | --------------- | --------- | --------- |
| NetSleuth | 0.006     | 0.003     | 0.000     | 0.160      | 0.181     | 0.143     | 0.142      | 0.105     | 0.219     | 0.128           | 0.099     | 0.181     |
| LPSI      | 0.544     | 0.516     | **0.575** | 0.487      | 0.495     | 0.479     | 0.348      | 0.517     | 0.268     | 0.221           | 0.282     | 0.198     |
| GCNSI     | 0.458     | 0.411     | 0.517     | 0.374      | 0.352     | 0.399     | 0.383      | 0.474     | 0.321     | 0.343           | 0.321     | 0.370     |
| TGASI     | 0.472     | 0.406     | 0.564     | 0.362      | 0.327     | 0.405     | 0.388      | 0.462     | 0.335     | 0.377           | 0.339     | **0.423** |
| SL-VAE    | 0.479     | 0.565     | 0.416     | 0.353      | 0.424     | 0.302     | 0.467      | 0.588     | 0.387     | 0.458           | 0.662     | 0.351     |
| DDMSL     | 0.517     | 0.592     | 0.459     | 0.492      | 0.504     | 0.481     | 0.448      | 0.540     | 0.432     | 0.417           | 0.481     | 0.368     |
| **SIDSL** | **0.585** | **0.605** | 0.566     | **0.546**  | **0.516** | **0.580** | **0.522**  | **0.702** | **0.439** | **0.519**       | **0.747** | 0.436     |

Two things this table settles:

1. **DDMSL > SL-VAE on real cascades, but not by much**: `0.517` vs `0.479` on Digg, `0.492` vs `0.353` on Twitter [verified]. Three years of architecture work bought ~`0.04-0.14` F1.
2. **LPSI, a 2017 label-propagation method with no learning at all, beats both on Digg** (`0.544`) [verified]. Any new method that does not beat LPSI on every dataset has not cleared the bar.

SIDSL's own contribution is the **few-shot** regime: with pretraining on synthetic propagation, DDMSL's Android F1 goes `0.010 → 0.109` and SL-VAE's `0.009 → 0.036` [verified, prose accompanying Table 2], i.e. the generative methods are nearly useless without enough real cascades, which is exactly the regime a simulator-trained forward model is meant to fix.

### 5.6 GNN source-detection benchmark (2026): the independent evaluation

A third-party benchmark (not authored by any of the method groups) under **SIR, single-source, top-`k` accuracy**, on six _contact_ networks. Different task framing from everything above (single-source ranking, not multi-source classification) so its numbers are not comparable to §5.1-§5.5. Its value is that it is reproducible and it includes non-GNN baselines.

Networks [verified, Table 3]; `β` = infection rate, `T` = snapshot time:

| Name       | \|V\| | \|E\| | ⟨k⟩   | ⟨l⟩  | diam | C    | β     | T    |
| ---------- | ----- | ----- | ----- | ---- | ---- | ---- | ----- | ---- |
| Karate     | 34    | 77    | 4.53  | 2.42 | 5    | 0.54 | 1.300 | 0.85 |
| Iceland    | 75    | 114   | 3.04  | 3.20 | 6    | 0.29 | 5.100 | 0.34 |
| Dolphin    | 62    | 159   | 5.13  | 3.36 | 8    | 0.26 | 0.900 | 2.20 |
| Fraternity | 58    | 967   | 33.34 | 1.42 | 3    | 0.75 | 0.073 | 3.50 |
| Workplace  | 92    | 755   | 16.41 | 1.96 | 3    | 0.43 | 0.165 | 3.50 |
| Highschool | 327   | 5,818 | 35.58 | 2.16 | 4    | 0.50 | 0.065 | 7.50 |

Top-5 accuracy, ± 95% CI over three seeds [verified, Table 4]:

| Model         | Karate             | Iceland            | Dolphin            |
| ------------- | ------------------ | ------------------ | ------------------ |
| Random        | 39.37% (±0.88)     | 26.51% (±0.19)     | 33.08% (±0.30)     |
| Jordan centre | 52.68% (±0.37)     | 35.84% (±0.14)     | 41.63% (±0.13)     |
| Betweenness   | 55.19% (±0.11)     | 37.39% (±0.09)     | 43.05% (±0.04)     |
| SME           | 60.67% (±0.22)     | 46.00% (±0.26)     | 45.37% (±0.52)     |
| MCMF          | 65.61% (±0.16)     | 50.95% (±0.35)     | 51.25% (±0.37)     |
| MLP-node      | 60.01% (±0.34)     | 41.50% (±0.09)     | 44.59% (±0.33)     |
| MLP-snapshot  | 67.90% (±0.84)     | 53.07% (±0.21)     | 55.94% (±0.22)     |
| IGCN          | **72.83%** (±0.15) | 55.77% (±0.09)     | 58.30% (±0.08)     |
| GCN [11]      | 68.67% (±0.27)     | 55.94% (±0.21)     | 57.81% (±0.21)     |
| GCN [12]      | **72.87%** (±0.03) | **57.95%** (±0.39) | **60.33%** (±0.10) |
| GraphSAGE     | 71.89% (±0.51)     | 55.54% (±0.60)     | 59.80% (±0.09)     |

**Note the ceiling.** The best GNN on Karate reaches **72.9% top-5 accuracy** on a 34-node graph: where random already gets 39.4%. Single-source SIR localization is _fundamentally_ hard; the near-perfect F1 scores in §5.2 come from a much easier problem setting (multi-source, full snapshot, 10% seeds, simulated data with matched train/test distributions), not from a solved task.

---

## 6. Datasets

### 6.0 What `--dataset` accepts for this task

**Every one of the 71 real graph loaders and all 7 synthetic families (`er`, `ba`, `ws`, `sbm`, `powerlaw_cluster`, `kronecker`, `karate`) runs on this task.** The loaders are task-agnostic; nothing in `pipeline/tasks.py` restricts a task to a dataset. The lists below are therefore an experimental CHOICE, not a constraint, and the only place that choice is currently encoded is `sbatch/`. Where the sweep and the benchmark family disagree, the sweep is the accident and the family is the intent.

| | Datasets |
| --- | --- |
| **Benchmark family** (§6.2) | `karate`, `dolphins`, `jazz`, `netscience`, `cora_ml`, `power_grid` |
| **Cost target** (IVGD's scalability column) | `deezer` (the HU graph, 47,538 / 222,887) |
| **Synthetic** | `er`, `ba`, `ws`, `sbm`, `powerlaw_cluster`, `kronecker` |
| **In `sbatch/source_localization/sweep_datasets.sbatch` today** | `karate`, `dolphins`, `jazz`, `netscience`, `cora_ml`, `power_grid` |

**This is the only task whose sweep matches its benchmark family exactly.** The overlap with the rest of the repo is the highest of any task here: five of the six were already loaded for other tasks.

### 6.1 What we already load

**Five of SL-VAE's seven graphs, and four of GraphSL's six, are already in `data/datasets/`.** This is the highest overlap of any task in this folder.

| Graph                   | Ours (loader output)               | SL literature reports                                                                                    | Match?                                                                    | Used by                              |
| ----------------------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------ |
| `jazz` yes               | 198 / 2,742                        | 198 / 2,742 [verified × 4: IVGD T2, SL-VAE T5, GraphSL T1, SIDSL T5]                                     | yes **identical**                                                          | SL-VAE, IVGD, GraphSL, SIDSL         |
| `netscience` yes         | 1,589 / 2,742                      | 1,589 / 2,742 [verified: IVGD T2, GraphSL T1, SIDSL T5], but **1,565 / 13,532** in SL-VAE T5 [verified] | Warning: **three of four match; SL-VAE does not**, §6.4                        | SL-VAE, IVGD, GraphSL, SIDSL         |
| `cora_ml` yes            | 2,810 / 7,981                      | 2,810 / 7,981 [verified × 3: IVGD T2, SL-VAE T5, GraphSL T1]                                             | yes **identical** (our loader already applies graph2gauss `standardize()`) | SL-VAE, IVGD, GraphSL                |
| `power_grid` yes         | 4,941 / 6,594                      | 4,941 / 6,594 [verified × 4]                                                                             | yes **identical**                                                          | SL-VAE, IVGD, GraphSL, SIDSL         |
| `karate` yes (synthetic) | `nx.karate_club_graph()` → 34 / 78 | 34 / **78** [verified: SL-VAE T5, GraphSL T1] · 34 / **77** [verified: GNN benchmark T3]                 | yes vs SL line; Warning: the benchmark drops one edge                            | SL-VAE, IVGD, GraphSL, GNN benchmark |
| `digg` yes               | 116,893 / ≈2.6M (Syracuse)         | four _different_ Diggs in this literature: §6.4                                                         | no **different graph**                                                    | SL-VAE, IVGD, SL-Diff, SIDSL         |

**Karate stops being a curiosity here.** `influence_maximization.md` §6.1 flags `karate` as "used by SL-VAE, not by IM papers": this file is why it is in the loader list at all. It is a first-class benchmark for source localization and appears in four of the papers above.

The one graph we would have to add is **Dolphins** (62 / 159, undirected, avg degree 5.13): used by IVGD, GraphSL and the GNN benchmark. It is a two-column edge list and a ~6-line loader: [NetRepo page](https://networkrepository.com/soc-dolphins.php) · [direct `soc-dolphins.zip`](https://nrvis.com/download/data/soc/soc-dolphins.zip) · [Netzschleuder](https://networks.skewed.de/net/dolphins).

### 6.2 Full catalogue

`avg deg` is quoted as `2E/N` throughout: recomputed by us, **not** copied from the papers, because their conventions disagree (§0).

#### Topology-only graphs (simulated diffusion)

| Dataset               | Nodes  | Edges   | Type                                    | Avg deg (2E/N) | Direct download                                                                                           | Used by                                        |
| --------------------- | ------ | ------- | --------------------------------------- | -------------- | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| **Karate** yes         | 34     | 78      | undirected                              | 4.59           | `nx.karate_club_graph()` · [KONECT](http://konect.cc/networks/ucidata-zachary/)                           | SL-VAE, IVGD, GraphSL, GNN bench (as 77 edges) |
| Dolphins              | 62     | 159     | undirected                              | 5.13           | [`soc-dolphins.zip`](https://nrvis.com/download/data/soc/soc-dolphins.zip)                                | IVGD, GraphSL, GNN bench                       |
| Iceland               | 75     | 114     | undirected (contact)                    | 3.04           | not established                                                                                           | GNN bench                                      |
| Fraternity            | 58     | 967     | undirected (contact)                    | 33.3           | not established                                                                                           | GNN bench                                      |
| Workplace             | 92     | 755     | undirected (contact, static projection) | 16.4           | [SocioPatterns](http://www.sociopatterns.org/datasets/) [claim]                                           | GNN bench                                      |
| **Jazz** yes           | 198    | 2,742   | undirected                              | 27.7           | [`arenas-jazz.zip`](https://nrvis.com/download/data/misc/arenas-jazz.zip)                                 | SL-VAE, IVGD, GraphSL, SIDSL                   |
| Highschool            | 327    | 5,818   | undirected (contact, static projection) | 35.6           | [SocioPatterns](http://www.sociopatterns.org/datasets/) [claim]                                           | GNN bench                                      |
| **NetScience** yes     | 1,589  | 2,742   | undirected                              | 3.45           | [Netzschleuder](https://networks.skewed.de/net/netscience)                                                | IVGD, GraphSL, SIDSL                           |
| NetScience (SL-VAE's) | 1,565  | 13,532  | undirected                              | 17.29          | no **not publicly available**: §6.4                                                                      | SL-VAE, DeepIM                                 |
| **Cora-ML** yes        | 2,810  | 7,981   | undirected                              | 5.68           | [`cora_ml.npz`](https://github.com/abojchevski/graph2gauss/raw/master/data/cora_ml.npz) + `standardize()` | SL-VAE, IVGD, GraphSL                          |
| **Power Grid** yes     | 4,941  | 6,594   | undirected                              | 2.67           | [`opsahl-powergrid.zip`](https://nrvis.com/download/data/misc/opsahl-powergrid.zip)                       | SL-VAE, IVGD, GraphSL, SIDSL                   |
| Deezer                | 47,538 | 222,887 | undirected                              | 9.38           | [SNAP gemsec-Deezer](https://snap.stanford.edu/data/gemsec-Deezer.html)                                   | IVGD (scalability only)                        |

All six graphs in the first block plus Cora-ML and Power Grid ship **pre-packaged inside GraphSL** at [`xianggebenben/GraphSL/data`](https://github.com/xianggebenben/GraphSL/tree/main/data), the fastest way to guarantee a byte-identical comparison.

#### Real-cascade datasets (observed diffusion, no simulation)

These carry **timestamped cascades**, so sources are defined by infection time rather than assigned. Every one of them appears under **two or more different node/edge counts** across papers (§6.4).

| Dataset      | Nodes  | Edges   | As reported by            | Source                                                                      |
| ------------ | ------ | ------- | ------------------------- | --------------------------------------------------------------------------- |
| Memetracker  | 1,653  | 4,267   | IVGD Table 2 [verified]   | [SNAP MemeTracker](https://snap.stanford.edu/data/memetracker9.html)        |
| Memetracker  | 12,529 | 70,466  | SL-VAE Table 5 [verified] | same source, different extraction                                           |
| Digg         | 11,240 | 47,885  | IVGD Table 2 [verified]   | [ISI/Lerman Digg 2009](https://www.isi.edu/~lerman/downloads/digg2009.html) |
| Digg         | 15,912 | 78,649  | SL-VAE Table 5 [verified] | same                                                                        |
| Digg         | 14,511 | 194,405 | SIDSL Table 5 [verified]  | same                                                                        |
| Twitter      | 12,619 | 309,621 | SIDSL Table 5 [verified]  | not established                                                             |
| Android      | 9,958  | 42,915  | SIDSL Table 5 [verified]  | Stack Exchange dump; SNAP `sx-*` family is the closest public analogue      |
| Christianity | 2,897  | 30,044  | SIDSL Table 5 [verified]  | Stack Exchange dump                                                         |

**None of these is our `digg`** (116,893 / ≈2.6M, Syracuse friendship graph): see `influence_maximization.md` §6.4.3 for that separate collision.

### 6.3 Warning: Name collisions

Four names in this literature denote more than one graph. Unlike the IM collisions (`influence_maximization.md` §6.3), **three of these four occur between papers from the same research group**, so "same lab, same name" is not evidence of "same graph".

| Name                | Version A                            | Version B                         | Version C                  | Who uses which                                                             |
| ------------------- | ------------------------------------ | --------------------------------- | -------------------------- | -------------------------------------------------------------------------- |
| **Network Science** | Newman 2006 **1,589 / 2,742**        | **1,565 / 13,532** (unidentified) |                            | A: ours, IVGD, GraphSL, SIDSL · B: **SL-VAE, DeepIM**                      |
| **Digg**            | IVGD **11,240 / 47,885**             | SL-VAE **15,912 / 78,649**        | SIDSL **14,511 / 194,405** | all three claim ISI/Lerman Digg 2009; ours (Syracuse, 116,893) is a fourth |
| **Memetracker**     | IVGD **1,653 / 4,267**               | SL-VAE **12,529 / 70,466**        |                            | both cite the same SNAP MemeTracker release                                |
| **Karate**          | **34 / 78** (`nx.karate_club_graph`) | **34 / 77** (GNN benchmark)       |                            | A: SL-VAE, GraphSL · B: GNN benchmark                                      |

The Digg and Memetracker spreads are **cascade-extraction differences**, not different source files: each paper thresholds cascades by length and keeps only the nodes that appear, so the induced subgraph size is a hyperparameter. None of the three papers states the threshold. Treat any Digg or Memetracker number as non-comparable across papers.

### 6.4 Version forensics

| Dataset                       | Cause                                                               | Status                                          |
| ----------------------------- | ------------------------------------------------------------------- | ----------------------------------------------- |
| **Jazz, Cora-ML, Power Grid** | none: four papers agree to the digit                               | yes **identical to ours**                        |
| **Network Science**           | SL-VAE and DeepIM report a graph 4.9× denser than the one they cite | **narrowed: SL-VAE is the outlier, not us** |
| **Karate**                    | one edge                                                            | Warning: cosmetic                                     |
| **Digg / Memetracker**        | undocumented cascade-extraction thresholds                          | no non-comparable across papers                 |

#### 6.4.1 Network Science: SL-VAE inherits DeepIM's anomaly, and it is the minority

`influence_maximization.md` §6.4.6 leaves this **unresolved**: DeepIM reports Network Science as **1,565 / 13,532** (avg degree 17.28) while the Newman 2006 graph it cites is **1,589 / 2,742** (avg degree 3.45), and neither LCC extraction (the LCC is 379 nodes) nor a typo explains it.

This review does not resolve where 1,565 / 13,532 came from, but it **narrows the question decisively** by adding four independent data points [all verified by `pdftotext -layout` of each paper's own table]:

| Paper                                    | Group              | Year | Network Science as reported | Avg degree     | Clustering coeff. |
| ---------------------------------------- | ------------------ | ---- | --------------------------- | -------------- | ----------------- |
| **IVGD** (WWW)                           | Zhao, Emory        | 2022 | **1,589 / 2,742**           | 3.451 (`2E/N`) |                   |
| **SL-VAE** (KDD)                         | Zhao, Emory        | 2022 | **1,565 / 13,532**          | 17.29 (`2E/N`) | **0.741**         |
| **DeepIM** (ICML)                        | Zhao, Emory        | 2023 | **1,565 / 13,532**          | 17.28          |                   |
| **GraphSL** (JOSS)                       | Wang & Zhao, Emory | 2024 | **1,589 / 2,742**           |                |                   |
| **SIDSL**                                | independent        | 2025 | **1,589 / 2,742**           | 1.72 (`E/N`)   | **0.6377**        |
| **ours** (`data/datasets/netscience.py`) |                    |      | **1,589 / 2,742**           | 3.45           | 0.638 [derived]   |

Three conclusions:

1. **The same lab publishes both versions**, four months apart, under one name. IVGD (April 2022) and SL-VAE (August 2022) are both Liang Zhao's group at Emory; IVGD uses ours, SL-VAE does not.
2. **GraphSL (written by IVGD's first author and shipping SL-VAE as one of its methods) packages the 1,589 / 2,742 graph** [verified, Table 1]. The canonical library for this task uses _our_ version.
3. **The clustering coefficient is independent confirmation that they are genuinely different graphs**, not a counting convention: SL-VAE reports `0.741`, SIDSL reports `0.6377`, and ours computes `0.638` [derived], matching [networkrepository.com/netscience.php](https://networkrepository.com/netscience.php)'s published `0.637791`. A convention difference cannot move a clustering coefficient.

**Practical consequence: our Network Science row is comparable to IVGD, GraphSL and SIDSL, and not comparable to SL-VAE or DeepIM.** That is a strict improvement on the IM file's position, where the row was comparable to nothing. Both papers cite Rossi & Ahmed's Network Repository ([27] in SL-VAE's bibliography [verified]), whose published statistics match _our_ graph, so the 1,565 / 13,532 graph is undocumented in both papers and unavailable publicly.

#### 6.4.2 Karate: 78 vs 77 edges

`nx.karate_club_graph()` gives **34 / 78**, matching SL-VAE's Table 5 and GraphSL's Table 1 [verified]. The 2026 GNN benchmark reports **34 / 77** [verified, Table 3]: one edge fewer, cause not stated. Immaterial to any metric at this scale, but it means their top-5 accuracies are on a graph one edge away from ours.

---

## 7. Which paper uses which

Cells mark the dataset **as that paper reports it**: read §6.3 before assuming two yes in one row are the same graph.

| Dataset               | GCNSI'19 | IVGD'22         | SL-VAE'22        | SL-Diff'23 | DDMSL'23 | GIN-SD'24 | GraphSL'24 | SIDSL'25 | GNN bench'26 |
| --------------------- | -------- | --------------- | ---------------- | ---------- | -------- | --------- | ---------- | -------- | ------------ |
| **Karate** yes         |          | yes               | yes                |            |          |           | yes          |          | yes (77 edges) |
| Dolphins              |          | yes               |                  |            |          |           | yes          |          | yes            |
| Iceland               |          |                 |                  |            |          |           |            |          | yes            |
| Fraternity            |          |                 |                  |            |          |           |            |          | yes            |
| Workplace             |          |                 |                  |            |          |           |            |          | yes            |
| **Jazz** yes           |          | yes               | yes                |            |          | yes [claim] | yes          | yes        |              |
| Highschool            |          |                 |                  |            |          |           |            |          | yes            |
| **NetScience** yes     |          | yes               | yes (1,565/13,532) |            |          |           | yes          | yes        |              |
| **Cora-ML** yes        |          | yes               | yes                |            |          |           | yes          |          |              |
| **Power Grid** yes     |          | yes               | yes                |            |          | yes [claim] | yes          | yes        |              |
| Deezer                |          | yes (scalability) |                  |            |          |           |            |          |              |
| Memetracker           |          | yes               | yes                | yes          |          |           |            |          |              |
| Digg                  |          | yes               | yes                | yes          |          |           |            | yes        |              |
| Twitter (cascades)    |          |                 |                  | yes          |          |           |            | yes        |              |
| Android               |          |                 |                  | yes          |          |           |            | yes        |              |
| Christianity          |          |                 |                  | yes          |          |           |            | yes        |              |
| Synthetic ER/BA/WS/SF | yes        |                 |                  |            |          | yes         |            | yes        |              |

**Bold + yes = we already load it.** Five of the six bold rows are the entire intersection between our suite and the SL literature, and unlike the IM file, where the intersection was four graphs of which two were caveated, here **four of five are byte-comparable** (§6.4).

Notes on the empty cells:

- **GCNSI'19 has no row of its own** in this matrix because its published evaluation is on synthetic graphs and a small number of network-repository graphs; every number quoted for GCNSI in §5 comes from a _later paper re-running it_. There is no first-party GCNSI table in this file.
- **DDMSL'23 has an entirely empty column**: see §5.4; its dataset table could not be extracted. Its five datasets are [claim] real-world cascade graphs, and its numbers in §5.5 come from SIDSL's re-run.
- **GIN-SD'24** entries are [claim]: the AAAI paper's dataset list was read from the abstract and related-work prose, not from a transcribed table.
- **Nothing in this literature uses NetHEPT, NetPHY, Facebook, wiki-Vote, email-Eu-core, LastFM or YouTube**: the classical IM benchmark suite and the SL benchmark suite barely overlap. The shared core is exactly Jazz / Cora-ML / Power Grid / Network Science, i.e. the DeepIM suite.

---

## 8. Evaluation protocol

### 8.1 Metrics

Source localization is scored as **node-level binary classification over `V`**, with the source set as the positive class.

| Metric                      | Definition                                                                                                     | Status in this literature                                                                                                                                                                   |
| --------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Precision (PR)**          | `TP / (TP + FP)` over predicted sources                                                                        | reported by all                                                                                                                                                                             |
| **Recall (RE)**             | `TP / (TP + FN)`                                                                                               | reported by all                                                                                                                                                                             |
| **F1 / FS**                 | harmonic mean                                                                                                  | **the headline metric**; SL-VAE calls it "the most commonly used", IVGD calls it "the most important metric for performance evaluation" [verified]                                          |
| **AUC (ROC)**               | ranking quality over the whole node set                                                                        | the tie-breaker; SL-VAE adds it explicitly _because_ sources are a small positive class [verified]                                                                                          |
| **Accuracy (ACC)**          | `(TP + TN) / \|V\|`                                                                                            | Warning: **near-useless alone**, GCNSI scores `ACC 0.8840` with `F1 0.0218` on Network Science [verified, IVGD Table 3]. Report it only beside F1.                                               |
| **RE (re-simulated error)** | re-run the forward diffusion from the _predicted_ sources and measure the discrepancy against the observed `y` | Warning: **name collision**: SL-Diff and SIDSL both use `RE` as the column header for **Recall**, not re-simulated error [verified]. When a paper says "RE", check which it means before quoting. |
| **Top-`k` accuracy**        | is the true source in the top `k` ranked nodes?                                                                | the **single-source** convention only (GNN benchmark, rumor centrality, Jordan centre). Not comparable to F1.                                                                               |
| **Error distance**          | hop distance from predicted to true source                                                                     | single-source only                                                                                                                                                                          |

**We would report the full set**: `PR / RE / F1 / AUC` for comparability plus a true re-simulated error using our own simulator as the referee, which is the metric the literature _should_ report and mostly does not.

### 8.2 Single-source vs multi-source: the hard split

These are effectively two literatures and their numbers never mix.

|                      | Single-source                                                                           | Multi-source                                                            |
| -------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **Methods**          | rumor centrality, Jordan centre, Pinto et al., DMP, Comin, Costa, the 2026 GNN benchmark | NETSLEUTH, LPSI, OJC, GCNSI, IVGD, SL-VAE, SL-Diff, DDMSL, SIDSL        |
| **Output**           | a _ranking_ of all `\|V\|` nodes                                                        | a _set_                                                                 |
| **Metric**           | top-`k` accuracy, error distance, rank of the true source                               | PR / RE / F1 / AUC                                                      |
| **Source count `k`** | fixed at 1                                                                              | given (SL-VAE: 10% of `V`) or **inferred** (NETSLEUTH infers it by MDL) |
| **Typical ceiling**  | ~73% top-5 on a 34-node graph [verified, §5.6]                                          | F1 `0.67-0.99` on the same size graph [verified, §5.1-5.2]              |

**Our seed sets are 1-20% of `N`, so we are multi-source by default.** A single-source arm would need a dedicated `--budget 1` generation run; it is cheap but it changes which baselines are admissible.

### 8.3 Full observation vs partial observation

| Setting              | What is observed                           | Methods designed for it                                                                              |
| -------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| **Full snapshot**    | every node's state at time `T`             | SL-VAE, IVGD, GCNSI, LPSI, NETSLEUTH, SL-Diff, DDMSL, SIDSL: i.e. essentially the whole modern line |
| **Sparse observers** | arrival _times_ at a small subset of nodes | Pinto, Thiran, Vetterli (2012), OJC (2017)                                                             |
| **Incomplete nodes** | some nodes' states are simply missing      | GIN-SD (AAAI 2024): the first to address it                                                         |
| **Full trajectory**  | `s_0 … s_T`, all steps                     | DDMSL, DDMIX, DIPT (they _reconstruct_ it rather than assume it)                                     |

**Full snapshot is what we would do first**: it is what our episodes already record, and it is the setting every comparable number in §5 uses. Partial observation is a masking argument on the same data (`y` observed on a random subset), which makes it a cheap second experiment rather than a new pipeline.

### 8.4 Protocol traps

1. **Source fraction is not standardized.** SL-VAE simulates from **10%** of nodes chosen uniformly at random; SL-Diff defines sources as the **first 5%** by infection time; SIDSL uses the **top 10%** by influence time [all verified]. Three papers, three definitions of "the sources".
2. **Observation fraction likewise.** SL-VAE's cascade experiments observe the bottom 30%; SIDSL observes the top 30% [verified].
3. **Train/test split.** SL-Diff uses `2:2:6` train/val/test [verified], a _smaller_ training set than test set, which is unusual and makes its absolute numbers pessimistic relative to papers that train on 70%.
4. **Diffusion model.** SL-VAE runs SI and SIR for 200 iterations to convergence, folding `S` and `R` together as `y = 0` [verified]. **Nobody in this literature evaluates on IC or LT**: the two models our simulator produces. That is the single biggest protocol gap between us and them (§11).
5. **Repeats.** SL-VAE averages 10 runs; the GNN benchmark averages 3 seeds and publishes 95% CIs [verified]. Most other papers state nothing.

### 8.5 Protocol for the program-search formulation

§8.1-§8.4 describe the protocol this literature uses, and every number we report for comparability must follow it. The formulation in §2.3 additionally needs a protocol the literature has no reason to define, because no published method produces a reusable artifact. Getting this wrong is the easiest way to publish an invalid table.

#### 8.5.1 Splitting: two axes, not one

Because the outer loop selects a program, the split has to prevent that program from having seen its test instances. There are two independent leakage paths and the literature's single train/test split only closes one.

| Axis | Train | Test | What a failure here means |
| ---- | ----- | ---- | ------------------------- |
| **Episodes** (required) | episodes used to compute the outer-loop reward | held-out episodes, same graph, same dynamics | the program memorized specific cascades |
| **Graphs** (required for the amortization claim) | program selected on graph set $\mathcal{G}_{\text{train}}$ | evaluated on a graph never seen during search | the program is a graph-specific hack, not an algorithm |
| **Dynamics** (optional, the strongest claim) | selected under IC | evaluated under LT | the program encodes IC-specific structure |
| **Source fraction** (optional) | selected at $k = 10\%$ | evaluated at $k = 5\%$ and $20\%$ | the program is budget-brittle |

**The graph axis is the one that carries §2.3's claim.** A program selected and evaluated on Jazz alone proves nothing about amortization, because per-instance methods are not disadvantaged in that setting. The headline result is a program selected on one graph set and run unmodified on another, which is a comparison no per-instance method can even enter.

#### 8.5.2 Budget parity across arms

Arms 3-6 differ only in the binding of `predict_marginals` (§2.4.3), so any difference in their _outer_ budget invalidates the comparison. This is the same requirement `--referee` (with `--mc-agreement` for the NDlib replay) already enforces for influence maximization, and it has two halves:

1. **Equal outer-loop budget.** Identical $P$ (generations × population), identical LLM model, identical prompt scaffolding, identical seeds. An arm that gets more generations wins for the wrong reason.
2. **Equal inner-loop budget, reported two ways.** Fix either the number of `predict_marginals` calls (which favours `@monte_carlo`, since each of its calls is more accurate) or the wall-clock (which favours `@world_model`). **Report both.** Only reporting the second is the version of this table that a reviewer will correctly disbelieve.

#### 8.5.3 Cost accounting

The claim in §2.3.2 is a cost claim, so cost is a headline number rather than a footnote. `Trajectory.cost` already carries a dict for this. Report, per arm:

| Quantity | Why |
| -------- | --- |
| Forward-model calls during program search, $P \cdot M \cdot C$ | the feasibility claim: shows `@monte_carlo` cannot reach the same $P$ |
| Wall-clock of the full search | what a practitioner actually pays |
| Forward-model calls **per test instance**, $C$ | the inference-cost claim of §2.3.2, which is empirical and may come out flat |
| Wall-clock per test instance | comparable against SL-VAE's per-instance optimization loop |
| LLM tokens consumed by the outer loop | the honest cost that arms 1 and 2 do not pay at all |

#### 8.5.4 Baseline parity

LPSI, NETSLEUTH, OJC and the centralities do no training, so "training cost" is not a fair axis against them and inference cost is. State the agent's offline search cost as a separate line rather than folding it into a per-instance comparison. GCNSI, IVGD and SL-VAE _do_ train, so their training cost belongs in the same column as ours; SL-VAE additionally pays a per-instance optimization loop at test time, which is the row where the amortization shows up if it shows up anywhere.

#### 8.5.5 What to report

For each (graph, dynamics, arm): `PR / RE / F1 / AUC` on the multi-source convention of §8.2, on **both** the MC-marginal observation and a binarized single draw (§2.9, risk 5); the true re-simulated error using NDlib as referee; the cost block of §8.5.3; and the source code of the winning program, since an interpretable artifact that is never shown forfeits its own advantage.

---

## 9. Implications for this project

### 9.1 The one-line case

**Nothing published is both amortized and forward-model-using, and that empty cell is the shape of our two loops.** The generative line (SL-VAE, IVGD, DDMSL, SIDSL) exploits a learned forward model but pays a fresh optimization loop for every cascade. The amortized line (GCNSI, GIN-SD, LPSI, NETSLEUTH, OJC) runs in one pass but throws the forward model away, and it is the weakest family in every table in §5. §1 shows the cross-product with one cell empty. §2.3 fills it: the coding-agent outer loop searches the space of inversion _programs_, and the world model is the forward oracle those programs query while they run.

**The obvious alternative is a trap worth naming.** "Every strong method builds a learned forward model and then inverts it, and we already have the forward model" is true, and it is a component swap that SL-VAE has already declared a no-op: the paper plugs in GAT, MONSTOR and DeepIS and reports no significant difference [verified, Fig. 3 discussion]. Building that is worth doing exactly once, as arm A of §2.6, so there is a control to measure program search against. It is not the contribution, and treating it as one would put this project's name on someone else's method.

What our `WorldModel` genuinely brings to the subroutine role is narrower than "a better $p_\psi$" but real: it is **mechanism-shaped** (per-edge transmission composed into $p_{\text{new}}$, so it cannot saturate by construction, which matters because saturation destroys source information) and **action-conditioned** (which buys nothing in the base formulation and is the entire point of the adaptive variant, §2.8).

### 9.2 Why it costs almost nothing

| What the task needs                      | What we would have to do                                                                                                                                                                                                                                     |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| A simulator to harvest from              | **nothing**: NDlib IC/LT already used                                                                                                                                                                                                                       |
| New action ops                           | **nothing**: actions are `NULL` throughout (`--inject-p 0`, omit `--action-ops`); the transition degenerates to `f_θ(G, s_t) → s_{t+1}`, a case we already train                                                                                            |
| `(seed set, final state)` training pairs | **nothing**: `data/generate_wm_data.py` already writes them. The `t = 0, branch=main` record's `action` field _is_ the seed set (a bag of `add_node` ops); the episode's last `next_state` is the observation. Group `transitions_*.jsonl` by `episode_id`. |
| Graphs with published baselines          | **nothing**: Jazz, Cora-ML, Power Grid, Network Science and Karate are all already loaded, and four of the five are byte-comparable (§6.4)                                                                                                                  |
| An outer loop over algorithms            | **nothing**: `coding_agent/` already generates, executes, scores and refines strategies against four interchangeable evaluators. Source localization adds one Protocol method and one primitive (§2.7)                                                     |
| A forward oracle callable from generated code | **almost nothing**: `primitives.mc_simulate_spread` already exposes `x → σ(x)`; the vector-valued version is the same call, and `Trajectory.final_marginals` already computes it                                                                       |
| New model code                           | items 1-6 and 8 of §2.7 are plumbing; **item 7 (arm A) is the only genuinely new modelling code, and it is the control rather than the method**                                                                                                             |

**Estimated cost: 2-4 days for the method, plus 1-2 days for the ablation arm.** Still the cheapest task in this folder by a wide margin.

### 9.3 Build order

Ordered so the risk that kills the project is tested before the work that depends on it.

1. **Kill-test LPSI first, before building anything else.** Hand-write LPSI (roughly fifteen lines), run the outer loop on one graph with a small $P$, and check whether any generated program beats it. §5.5 has LPSI beating both SL-VAE and DDMSL on Digg [verified], and LPSI sits _inside_ the agent's expressible space, so the realistic floor is "the search rediscovers LPSI". If forward-model-guided refinement cannot improve on that floor, stop: items 2-5 are not worth building. This is §2.9 risk 1 and it is cheap to run.
2. **Build the loop: items 1-6 and 8 of §2.7.** The primitive and its four bindings, `localize` on the Protocol, the F1 reward, episode regrouping, metrics, prompts, registry entry. This produces arms 3-6.
3. **Arm A was built and then removed on 2026-09-04** (condition 8, `world_model/wm_sl.py`, deleted): under the label-free consistency reward the frozen-model inversion duplicated what arms 3-6 already measure, so the 6-vs-A comparison is no longer run.
4. **Install GraphSL rather than reimplementing** (`pip install GraphSL`) for arms 1 and 2. It ships LPSI, NETSLEUTH, OJC, GCNSI, IVGD and SL-VAE behind one API returning accuracy / precision / recall / F1 / AUC, plus the six benchmark graphs, and it packages **our** Network Science version.
5. **Run the transfer experiments of §8.5.1.** A program selected and evaluated on the same graph proves nothing about amortization. Select on one graph set, run unmodified on another, and report that as the headline.
6. **Report on Jazz, Cora-ML, Power Grid and Karate** against §5.1 and §5.2 directly, on both the marginal and the binarized observation. Report Network Science against IVGD / GraphSL / SIDSL and state explicitly that SL-VAE's column is a different graph (§11).

### 9.4 What this buys beyond a new task row

- **It is the only task in this folder that turns the coding agent onto an inverse problem.** Every other both-loops task (influence maximization, blocking, critical-node detection, epidemic control) has the agent selecting interventions to steer a forward process. Here it writes an inference algorithm instead, which exercises a different half of the contract and tests whether the outer loop generalizes past intervention selection.
- **It produces an artifact that transfers.** Every published method here either re-optimizes per cascade or ships network weights. A selected program is source code that runs on a graph it was never selected on, and §8.5.1 makes that claim testable. Nothing in the literature can enter that comparison.
- **It is a sharp, new diagnostic for forward-model fidelity.** A model that saturates in rollout (`ens_count_bias ≫ 0`) has destroyed the information the inverse problem needs. Source-localization F1 fails loudly where one-step `delta_f1` (which is dominated by unchanged nodes) stays comfortable. Given that saturation was this project's hardest bug (`wm-rollout-saturation-diagnosis-2026-06`), a metric that regresses when it recurs is worth having.
- **It exercises the model in the direction it was never trained.** Everything in `wm_eval.py` scores forward prediction. Nothing yet asks whether the learned transition kernel supports _inference about its own inputs_, which is a genuinely different property.
- **The recovered algorithm is interpretable.** Source attribution is used for forensics, outbreak tracing and misinformation provenance, where "these five nodes, because they are the label-field maxima of their communities" is worth more than an equally accurate latent vector. §8.5.5 requires publishing the winning program for exactly this reason.
- **It is the natural companion to `cascade_reconstruction.md`.** DDMSL, DDMIX and DIPT all recover the whole path `s_0 … s_T`, not just `x`; that is the same inference procedure over the same kernel with a different read-out, and the same program-search framing applies unchanged.

### 9.5 Honest limitations

Task-level risks are enumerated in §2.9 (the LPSI floor, the zeroth-order objection, the unknown ill-posedness ceiling, saturation propagation, our easier observation, no IC/LT precedent, and §5.2's unrealistic absolute numbers). Three limitations are project-level rather than task-level:

- **Zero action ops exercised in the base formulation.** This task adds no coverage of the four idle ops, so it cannot double as action-space breadth. §2.8 is the variant that fixes it, at the cost of a new op and a regeneration run.
- **The outer loop's reward needs ground-truth sources, which only a simulator provides.** §2.3.3 makes the selected program label-free _at inference_, which is what allows deployment on real cascades, but program _selection_ requires labelled episodes, so this method cannot be trained directly on Digg or Memetracker. That is a real scope limit and it should be stated rather than discovered by a reviewer.
- **It adds a second stochastic search on top of an already stochastic pipeline.** Outer-loop variance across LLM seeds compounds with MC variance in the reward. §8.5.2's seed and budget parity is the minimum control; multiple outer-loop seeds with reported spread is the honest version, and it multiplies the cost of every arm.

---

## 10. Reference list

All checked **2026-07-28**. `(403)` = the host rejects automated agents; the URL is correct and resolves in a browser.

**Classical** [Shah & Zaman, _Rumors in a Network: Who's the Culprit?_ (arXiv 0909.4370, IEEE TIT 57(8) 2011)](https://arxiv.org/abs/0909.4370) · [Shah & Zaman, SIGMETRICS 2010 `10.1145/1811039.1811063`](https://dl.acm.org/doi/10.1145/1811039.1811063) (403) · [Comin & da F. Costa, Phys. Rev. E 84:056105 (2011)](https://doi.org/10.1103/PhysRevE.84.056105) · [Pinto, Thiran & Vetterli, _Locating the Source of Diffusion in Large-Scale Networks_ (arXiv 1208.2534, PRL 109:068702)](https://arxiv.org/abs/1208.2534) · [Fioriti & Chinnici, _Predicting the sources of an outbreak with a spectral technique_ (arXiv 1211.2333)](https://arxiv.org/abs/1211.2333) · [Prakash, Vreeken & Faloutsos, **NETSLEUTH**, ICDM 2012](https://people.cs.vt.edu/~badityap/papers/netsleuth-icdm12.pdf) · [mirror](https://eda.mmci.uni-saarland.de/pubs/2012/netsleuth-prakash,vreeken,faloutsos.pdf) · [KAIS 2014](https://link.springer.com/article/10.1007/s10115-013-0671-5) · [Zhu & Ying, _Information Source Detection in the SIR Model: A Sample-Path-Based Approach_ (arXiv 1206.5421)](https://arxiv.org/abs/1206.5421) · [IEEE/ACM ToN 24(1) 2016](https://ieeexplore.ieee.org/document/6414679) · [Brockmann & Helbing, _The Hidden Geometry of Complex, Network-Driven Contagion Phenomena_, Science 342:1337 (2013)](https://www.science.org/doi/10.1126/science.1245200) (403) · [Altarelli et al., _Bayesian inference of epidemics on networks via Belief Propagation_ (arXiv 1307.6786, PRL 112:118701)](https://arxiv.org/abs/1307.6786) · [Lokhov, Mézard, Ohta & Zdeborová, _Inferring the origin of an epidemic with a dynamic message-passing algorithm_ (arXiv 1303.5315, Phys. Rev. E 90:012801)](https://arxiv.org/abs/1303.5315) · [Wang, Wang, Pei & Ye, **LPSI**, _Multiple Source Detection without Knowing the Underlying Propagation Model_, AAAI 2017](https://ojs.aaai.org/index.php/AAAI/article/view/10731) · [Zhu, Chen & Ying, **OJC**, _Catch'Em All: Locating Multiple Diffusion Sources in Networks with Partial Observations_ (arXiv 1611.06963, AAAI 2017)](https://arxiv.org/abs/1611.06963)

**Learning-based** [Dong et al., **GCNSI**, CIKM 2019](https://dl.acm.org/doi/10.1145/3357384.3357994) (403) · [Wang, Jiang & Zhao, **IVGD**, WWW 2022 (arXiv 2206.09214)](https://arxiv.org/abs/2206.09214) · [Emory PDF](http://cs.emory.edu/~lzhao41/materials/papers/3485447.3512155.pdf) · [code](https://github.com/xianggebenben/IVGD) · [Ling et al., **SL-VAE**, KDD 2022 (arXiv 2206.12327)](https://arxiv.org/abs/2206.12327) · [KDD](https://dl.acm.org/doi/10.1145/3534678.3539288) (403) · [code](https://github.com/triplej0079/SLVAE) · [Huang et al., **SL-Diff**, ECML-PKDD 2023 (arXiv 2304.08841)](https://arxiv.org/abs/2304.08841) · [Springer](https://link.springer.com/chapter/10.1007/978-3-031-43418-1_20) · [**DDMSL**, _Diffusion Model for Graph Inverse Problems_, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/46ab9d9645b6975b947231ddb48da1ab-Abstract-Conference.html) · [PDF (35 MB)](https://proceedings.neurips.cc/paper_files/paper/2023/file/46ab9d9645b6975b947231ddb48da1ab-Paper-Conference.pdf) · [OpenReview](https://openreview.net/forum?id=5Fr8Nwi5KF) · [poster](https://neurips.cc/virtual/2023/poster/72813) · [Cheng et al., **GIN-SD**, AAAI 2024 (arXiv 2403.00014)](https://arxiv.org/abs/2403.00014) · [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/27755) · [Ling, Chowdhury et al., **CNSL**, _Source Localization for Cross Network Information Diffusion_ (arXiv 2404.14668)](https://arxiv.org/abs/2404.14668) · [code](https://github.com/tanmoysr/CNSL) · [_Graph contrastive learning for source localization in social networks_, Information Sciences 2024](https://www.sciencedirect.com/science/article/abs/pii/S0020025524010041) (403) · [**PGSL**, Expert Systems with Applications 2024](https://www.sciencedirect.com/science/article/abs/pii/S0957417423025307) · [Chen et al., **SIDSL**, _Structure-prior Informed Diffusion Model for Graph Source Localization with Limited Data_ (arXiv 2502.17928)](https://arxiv.org/abs/2502.17928) · [Memon, Ling, Kong et al., **DIPT**, _Deep Identification of Propagation Trees in Graph Diffusion_ (arXiv 2503.00646)](https://arxiv.org/abs/2503.00646) · [_Good Advisor for Source Localization: Using Large Language Model to Guide the Source Inference Process_, IJCAI 2025](https://www.ijcai.org/proceedings/2025/0326.pdf) · [_A generalized diffusion framework with learnable propagation dynamics for source localization_, IJCAI 2025](https://dl.acm.org/doi/abs/10.24963/ijcai.2025/325) (403) · [**HyperDet**, _Source Detection in Hypergraphs_ (arXiv 2505.12894)](https://arxiv.org/abs/2505.12894) · [Cheng et al., **SourceDetMamba**, IJCAI 2025 (arXiv 2505.12910)](https://arxiv.org/abs/2505.12910) · [**PDSL**, _Propagation Dynamics Aware Framework for Source Localization_ (arXiv 2605.03550)](https://arxiv.org/abs/2605.03550) · [code](https://github.com/MrYansong/PDSL)

**Surveys and benchmarks** [Jiang, Wen, Yu, Xiang & Zhou, _Identifying Propagation Sources in Networks: State-of-the-Art and Comparative Studies_, IEEE Comm. Surveys & Tutorials 19(1):465-481 (2017)](https://nsclab.org/nsclab/esi/comst_jiang2017.pdf) · [IEEE](https://ieeexplore.ieee.org/document/7582484) · [Sterchi, Brack & Hilfiker, _Graph Neural Networks for Source Detection: A Review and Benchmark Study_ (arXiv 2512.20657)](https://arxiv.org/abs/2512.20657) · [code](https://github.com/martinSter/gnn-source-detection)

**Libraries** [**GraphSL** (arXiv 2405.03724)](https://arxiv.org/abs/2405.03724) · [JOSS 9(99):6796](https://joss.theoj.org/papers/10.21105/joss.06796) · [code + packaged datasets](https://github.com/xianggebenben/GraphSL) · [**cosasi** (JOSS 2022)](https://joss.theoj.org/papers/10.21105/joss.04894) · [code](https://github.com/lmiconsulting/cosasi)

**Data sources** [Network Repository](https://networkrepository.com/) · [Netzschleuder](https://networks.skewed.de/) · [KONECT](http://konect.cc/) · [SNAP MemeTracker](https://snap.stanford.edu/data/memetracker9.html) · [ISI/Lerman Digg 2009](https://www.isi.edu/~lerman/downloads/digg2009.html) · [SNAP gemsec-Deezer](https://snap.stanford.edu/data/gemsec-Deezer.html)

---

## 11. Open gaps

Honest list of what this review could **not** establish.

### Does any SL paper's graph version conflict with ours?

**Yes: exactly one, and it is the seed paper.**

| Graph               | Ours            | SL literature                                                                                  | Verdict                                                        |
| ------------------- | --------------- | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| **Jazz**            | 198 / 2,742     | 198 / 2,742 in IVGD, SL-VAE, GraphSL, SIDSL [verified × 4]                                     | yes **no conflict**                                             |
| **Cora-ML**         | 2,810 / 7,981   | 2,810 / 7,981 in IVGD, SL-VAE, GraphSL [verified × 3]                                          | yes **no conflict**: our `standardize()` already reproduces it |
| **Power Grid**      | 4,941 / 6,594   | 4,941 / 6,594 in all four [verified × 4]                                                       | yes **no conflict**                                             |
| **Karate**          | 34 / 78         | 34 / 78 in SL-VAE, GraphSL [verified × 2] · 34 / **77** in the 2026 GNN benchmark [verified]   | Warning: cosmetic, one edge, cause unstated (§6.4.2)                |
| **Network Science** | 1,589 / 2,742   | 1,589 / 2,742 in IVGD, GraphSL, SIDSL [verified × 3] · **1,565 / 13,532** in SL-VAE [verified] | **CONFLICT with SL-VAE only**                               |
| **Digg**            | 116,893 / ≈2.6M | four mutually different graphs across four papers (§6.3)                                       | no **not comparable to anything**                              |

**DeepIM's unresolved NetScience case does reappear, and this file narrows it.** `influence_maximization.md` §6.4.6 could not tell whether DeepIM's 1,565 / 13,532 Network Science was an error or a real graph. **SL-VAE (KDD 2022) reports the identical anomalous numbers a year earlier, from the same lab** (Liang Zhao, Emory), with average degree `17.29` and clustering coefficient `0.741` [verified, Table 5]. Meanwhile **IVGD (WWW 2022, same lab, four months earlier), GraphSL (JOSS 2024, written by IVGD's first author) and SIDSL (2025, independent) all report 1,589 / 2,742**, our graph [verified × 3]. SIDSL's clustering coefficient `0.6377` and ours `0.638` [derived] match Network Repository's published `0.637791`; SL-VAE's `0.741` cannot be reconciled by any counting convention, so the two really are different graphs.

**Net position: the anomaly is SL-VAE's and DeepIM's, not ours, and it is the minority convention in its own literature.** What is still _not_ established is where 1,565 / 13,532 came from. Both papers cite Rossi & Ahmed's Network Repository ([27] in SL-VAE's bibliography [verified]), whose statistics match our graph. The denser graph is undocumented in both papers and not publicly downloadable; resolving it still needs SL-VAE's or DeepIM's data pickle, and neither repository ships one.

### Result tables that could not be extracted

- **DDMSL (NeurIPS 2023): the second seed paper.** Its own numbers are **not** in this file. The proceedings PDF is 35.5 MB and four download attempts (including `curl -C -` resumption over ~20 minutes) truncated at ≈2.8 MB, from which `pdftotext` recovers 56 lines. The OpenReview mirror (`openreview.net/pdf?id=5Fr8Nwi5KF`) serves a 12 KB JavaScript shell, not a PDF. Everything reported for DDMSL in §5.5 is **SIDSL's third-party re-run**. Its dataset table, source-fraction convention and five dataset names are all unestablished.
- **GCNSI (CIKM 2019).** No first-party table: the ACM page 403s to automated agents. Every GCNSI number in §5 is a later paper's re-run, and those disagree wildly (`F1 0.0218` in IVGD's Table 3 vs `0.458` in SIDSL's Table 1, on different data and protocols).
- **IVGD Table 5 (runtime).** The two-column extraction reliably yields only `0.26 / 0.27 / 0.76` s for LPSI on Karate / Dolphins / Jazz; the remaining four values (`52.83 / 240.88 / 899.45 / 94,541.13`) could not have their column headers confirmed and are marked **[claim]** in §5.2.
- **GIN-SD, HyperDet, SourceDetMamba, LLM-advisor, PDSL, CNSL, DIPT**: all identified, linked and characterized, but **no result cells transcribed**. They evaluate on hypergraphs, cross-network settings or incomplete-observation regimes largely disjoint from ours.
- **MOEIM-style figure-only results** do not occur here; no [figure]-tier number appears in this file.

### Code and URL failures

- **No public code found** for: SL-Diff, DDMSL, DDMIX, GIN-SD, SIDSL, DIPT, HyperDet, SourceDetMamba, LLM-advisor, PGSL, graph-contrastive SL, and every classical method except through GraphSL / cosasi.
- **403 to automated agents** (URLs correct, resolve in a browser): ACM DL (SL-VAE KDD, GCNSI CIKM, Shah & Zaman SIGMETRICS, IJCAI-2025 learnable-dynamics), `science.org` (Brockmann & Helbing), `sciencedirect.com` (PGSL, graph contrastive SL).
- **`202` (not `200`)** from IEEE Xplore for the Jiang survey and Zhu & Ying: the free `nsclab.org` mirror of the survey does return `200`.
- **cosasi's repository is `lmiconsulting/cosasi`**, recovered from its JOSS paper; several secondary sources cite a `qwertyjl/cosasi` URL that **404s**.

### Dataset counts that are only [claim]

- **Iceland, Fraternity, Workplace, Highschool** (GNN benchmark): node/edge counts are [verified] from its Table 3, but **no direct download URL** was confirmed. They are almost certainly [SocioPatterns](http://www.sociopatterns.org/datasets/) releases; not checked against a file.
- **Android, Christianity, Twitter** (SL-Diff, SIDSL): counts [verified] from SIDSL's Table 5, but the Stack Exchange / Twitter dumps they were derived from have **no direct download URL** established.
- **GIN-SD's dataset list** (Jazz, Power Grid) is [claim]: read from prose, not a transcribed table.
- **Digg and Memetracker at every size**: the three published Digg versions and two Memetracker versions differ by undocumented cascade-extraction thresholds (§6.3). None of the three papers states its threshold.

### Protocol gaps

- **No published SL result under IC or LT.** SL-VAE uses SI/SIR; SL-Diff, SIDSL and DDMSL use real cascades; the GNN benchmark uses SIR. **Our IC/LT numbers will have no direct precedent on any graph**, even where the graph matches byte-for-byte. This is the single biggest comparability gap in this file, and it is bigger than the NetScience one.
- **`RE` is an overloaded column name**: Recall in SL-Diff and SIDSL, re-simulated error elsewhere. **No paper surveyed reports a genuine re-simulated error**, so that metric would have no baseline.
- **Source-fraction conventions are irreconcilable**: 10% uniform-random (SL-VAE), first 5% by infection time (SL-Diff), top 10% by influence time (SIDSL). Any cross-paper table is invalid without re-running.
- **Dolphins (62 / 159) is the only benchmark graph we lack.** Not a gap in the review: a ~6-line loader (§6.1).

### Gaps specific to the program-search formulation (§2.3)

These are gaps in the _literature_, not in this review. They are what makes §2.3 a contribution and simultaneously what makes it hard to benchmark.

- **No published SL method is amortized across graphs.** Every method here is either re-optimized per instance (SL-VAE, IVGD, DDMSL) or is a network trained and tested on one graph (GCNSI, GIN-SD). **No paper reports what happens when a method selected on graph A is run unmodified on graph B**, because per-instance methods have no artifact to transfer and the supervised GNNs are not evaluated that way. The transfer experiment of §8.5.1 is therefore our headline result _and_ has no published number to sit beside. Cross-graph generalization is the closest published relative (CNSL's cross-network setting, §4), and it is a different problem.
- **No published SL method reports inference cost.** IVGD's Table 5 is the only runtime table in this file and its column assignment could not be confirmed (§5.2, marked [claim]). SL-VAE, DDMSL, SIDSL and SL-Diff report no wall-clock or call counts at all. The cost comparison of §8.5.3 will therefore be against numbers we measure ourselves by re-running their code where it exists (SL-VAE and IVGD ship repositories; DDMSL, SL-Diff and SIDSL do not), and against nothing at all for the five methods with no public code.
- **No published SL method is LLM-generated.** The only LLM-in-the-loop method found is LLM-advisor (IJCAI 2025), which uses an LLM to embed rumour-comment _semantics_ as an extra input signal, not to write the inference algorithm (§4). There is no prior work on program search for this task, so there is no baseline for arm 2 (GA routing over a fixed pool) beyond what we construct ourselves.
- **No baseline exists for re-simulated error.** Carried from above and now load-bearing: §2.3.3 uses re-simulation error as an inside-the-program signal and §8.5.5 reports it as a metric, but **no paper surveyed reports a genuine re-simulated error**, so that column is self-contained. Do not present it as a comparison.
- **The Bayes-optimal F1 is uncharacterized.** No paper in this file establishes an upper bound on achievable F1 under a given source fraction and horizon, so a mediocre absolute number cannot be attributed to method or problem without our own arm-5 diagnostic (§2.9, risk 3). The nearest published anchor is the 2026 GNN benchmark's observation that the best GNN reaches ~73% top-5 on a 34-node graph (§5.6), which is a different metric and a different source count.
