# Epidemic Control on Networks — Prior Work, Datasets, and Published Results

Reference for the epidemic-intervention literature: choosing **which nodes to immunize**, **which nodes to quarantine**, and **which contacts to cut**, under SIR / SIS / SEIR / SIRS dynamics on a fixed or temporal contact graph. Covers both the _static_ problem (pick `k` nodes up front, one shot) and the _sequential_ problem (an RL policy that intervenes every step under partial observability). Includes the spectral line (minimize `λ₁`), the simulation line (minimize expected final size), and the 2020–2026 learning line.

This is the closest sibling to [`critical_node_detection.md`](critical_node_detection.md) — CND removes nodes to break _connectivity_, epidemic control removes them to break _a dynamical process_. Where they overlap, this file owns the dynamics and CND owns the structural objective.

Style and format follow [`influence_maximization.md`](influence_maximization.md); the skeleton is defined in [`README.md`](README.md).

All URLs returned HTTP 200 on **2026-07-28** unless annotated otherwise.

---

## 0. Verification policy

| Tier           | Meaning                                                                                                              |
| -------------- | -------------------------------------------------------------------------------------------------------------------- |
| **[verified]** | Read from the paper's own table via text extraction, or from the data repository's own statistics page. Trustworthy. |
| **[derived]**  | Computed by us from a file we downloaded, or reconstructed arithmetically from published splits.                     |
| **[figure]**   | Read off a plotted figure — the paper published no table. Approximate, direction only.                               |
| **[claim]**    | Stated in prose by a paper or a secondary source; not cross-checked against a file or table.                         |

**Automated PDF summarizers hallucinate plausible numbers from these papers.** Every number below marked [verified] was extracted with `pdftotext -layout` and read out of the paper's own table. This literature is unusually prone to the error, because most immunization papers publish **only figures** — a summarizer asked for "the numbers in Table 3" will happily invent a table that does not exist. Where a paper published no table, the row says [figure] or the cell is left empty. Nothing was guessed.

**Edge-count convention.** Undirected graphs are quoted as _undirected edges_; directed graphs as _arcs_. Our loaders report `adjacency.nnz`, which for a symmetrized undirected graph is **2× the undirected edge count**. Temporal contact traces get a third number — the **contact count** (timestamped interaction events), which is neither of the above and is typically 1–2 orders of magnitude larger than the aggregated edge count.

---

## 1. Task definition

Given a graph `G = (V, E)`, a compartmental epidemic process on it, and a budget `k` (vaccine doses, test kits, quarantine slots, link-cuts), choose an intervention that minimizes the size or speed of the outbreak.

### 1.1 The static immunization problem

```
S* = argmin_{S ⊆ V, |S| = k}  E[ |R(∞)| | G − S ]
```

Remove (vaccinate) `k` nodes before the epidemic starts, minimizing the expected final number of ever-infected nodes. Equivalent edge form removes `k` edges (quarantine / contact reduction) instead of nodes. NP-hard, and the objective is **not submodular in general** — which is why the field mostly abandoned direct optimization of `E[|R(∞)|]` in favour of a spectral surrogate.

### 1.2 The spectral surrogate — the idea that organizes the whole field

For essentially every propagation model on an arbitrary graph, the epidemic dies out iff

```
λ₁(A) · (b / d)  <  1          equivalently   λ₁(A) < τ = d / b
```

where `λ₁(A)` is the largest eigenvalue of the adjacency matrix, `b` the per-contact infection rate, `d` the cure/recovery rate. Established for SIS by Wang, Chakrabarti, Wang & Faloutsos (SRDS 2003 / TISSEC 2008), generalized to an arbitrary cascade model by Prakash et al. (ICDM 2011). The consequence is that minimizing `λ₁` after deleting `k` nodes/edges is a _model-independent_ proxy for containment, and it turns a #P-hard expectation into a matrix-perturbation problem. **NetShield, NetMelt, Gelling, GreedyWalk, and the Preciado convex program are all solving this surrogate**, not the simulation objective.

The surrogate is also the reason this literature is comparable at all: `λ₁` is deterministic, so two papers reporting "eigendrop after removing k nodes" on the same graph _are_ comparable, unlike two papers reporting simulated final size under different `β`.

### 1.3 The sequential / adaptive problem

```
π* = argmax_π  E[ Σ_t r(s_t, a_t) ],   a_t ⊂ V,  |a_t| = k  every step
```

The agent observes a partially-known state each step (test results are noisy and budgeted), picks `k` nodes to test / isolate / vaccinate, and the epidemic advances. This is a POMDP, and it is where the RL line lives (RLGN, DURLECA, Libin et al.). It is also the formulation that matches our world model most directly, because it _needs_ a fast forward simulator inside the planning loop.

### 1.4 The compartmental models

| Model               | Compartments     | Recovery                  | Re-infection       | Monotone? |
| ------------------- | ---------------- | ------------------------- | ------------------ | --------- |
| **SI**              | S → I            | no                        | no                 | ✅ yes    |
| **IC** (ours today) | S → I → R(spent) | n/a (spent ≠ susceptible) | no                 | ✅ yes    |
| **SIR**             | S → I → R        | yes                       | no                 | ❌ **no** |
| **SIS**             | S → I → S        | yes                       | **yes**            | ❌ **no** |
| **SEIR**            | S → E → I → R    | yes                       | no                 | ❌ **no** |
| **SIRS**            | S → I → R → S    | yes                       | yes (after waning) | ❌ **no** |
| **SWIR**            | S → W → I → R    | yes                       | no                 | ❌ no     |

"Monotone" means a node never leaves the infected set. Our entire structured-head design assumes it. §2 is mostly about what breaks when it stops being true.

### 1.5 Why the problem exists

Vaccine doses, test kits, ICU beds, and enforceable quarantine orders are all scarce and all allocated _on a network_. Untargeted (random) immunization of a scale-free network requires an immunization fraction approaching 1 to halt spread, because `λ₁` of a power-law graph is dominated by hubs; targeted immunization of a small hub set collapses it. That asymmetry — first quantified by Pastor-Satorras & Vespignani (2002) — is the entire practical motivation.

---

## 2. Fit with our methodology

**Verdict: ✅ direct fit, and the simulator side is nearly free — but it needs a new head.** The simulator change is small and mechanical. The model change is not: the structured IC head is _provably incapable_ of representing recovery.

### 2.1 The simulator is already three-quarters there

`data/wm_simulator.py` imports `ndlib.models.epidemics as epidemics` and uses two classes from it. That same module already ships every model we need [verified, `dir(ndlib.models.epidemics)` on the installed version]:

| NDlib class | Status codes (`available_statuses`) | Model params                        |
| ----------- | ----------------------------------- | ----------------------------------- |
| `SIRModel`  | S=0, I=1, **R=2**                   | `beta`, `gamma`, `tp_rate`          |
| `SISModel`  | S=0, I=1                            | `beta`, `lambda`, `tp_rate`         |
| `SEIRModel` | S=0, I=1, **E=2**, **R=3**          | `alpha`, `beta`, `gamma`, `tp_rate` |
| `SEISModel` | S=0, I=1, E=2                       | `alpha`, `beta`, `lambda`           |
| `SWIRModel` | S=0, I=1, W=2, R=3                  | `kappa`, `mu`, `nu`                 |

Also present: `SIModel`, `SEIR_ct_Model` / `SEIS_ct_Model` (continuous time), and `UTLDRModel` (a full COVID-style model with testing, lockdown, and death).

Concretely, adding SIR/SIS/SEIR touches **five methods, all in one file**:

1. **`Simulator.reset()`** — today an `if model_name == "IC": … else: (LT)` two-branch dispatch. Becomes a five-way dispatch. SIR/SIS/SEIR configure _model-level_ scalars via `config.add_model_parameter("beta", …)` rather than IC's per-edge `config.add_edge_configuration("threshold", (u,v), p)`. The `self.model.iteration()` no-op-first-iteration convention holds for all of them [verified, `SIRModel.iteration` returns early when `actual_iteration == 0`].
2. **`Simulator.active_nodes()`** — the `status in (1, 2)` IC special-case must become per-model. For SIR, `{1, 2}` happens to be correct as "ever infected"; for SEIR it is `{1, 2, 3}`; for SIS **there is no absorbing state at all**, so "ever infected" cannot be read off `status` and must be accumulated by us.
3. **`Simulator.current_state()` / `advance()`** — the frontier definitions.
4. **`Simulator.apply_actions()`** — the `2 if self.model_name == "IC" else 0` line for `remove_node` (see §2.4).
5. **`Simulator.advance_marginal()`** — the line `draws = num_mc if self.model_name == "IC" else 1` is **wrong for all three**. SIR, SIS, and SEIR are stochastic (they draw `np.random.random_sample()` against `beta`/`gamma` per contact per step, exactly as IC does), so they all need the full `num_mc` draws. LT stays the only 1-draw model.

No change to `snapshot()` / `restore()` — they copy `model.status`, which is model-agnostic. No change to `ActionOp`. Cost estimate: **~60 lines**, one day.

### 2.2 ⚠️ `set_edge_weight` becomes a no-op

NDlib's `SIRModel`, `SISModel`, and `SEIRModel` all declare `"edges": {}` — an **empty edge-parameter dict** — and their `iteration()` compares a uniform draw against the single scalar `self.params['model']['beta']` for every susceptible neighbour [verified, read from `inspect.getsource(SIRModel.iteration)`]. There is no per-edge transmission probability to set.

Consequences, in order of severity:

- `set_edge_weight` and the `weight` field of `add_edge` have **no effect** under SIR/SIS/SEIR. Our five-op action space drops to four.
- `GraphInput.edge_weight` degenerates to all-ones (the LT situation).
- **`structured_residual` cannot exist** for these models — it is defined as `q = sigmoid(logit(w) + MLP(...))`, and there is no `w`.
- Contact-reduction interventions (the whole "social distancing" branch of this literature, §3) become expressible only as `remove_edge`, i.e. all-or-nothing rather than graded.

Getting heterogeneous `β_uv` back means either writing our own stepper (the transition rule is four lines) or using NDlib's `GeneralisedThresholdModel` / composition API. Writing our own is the honest option and is ~40 lines; it also removes the `np.random` global-RNG seeding wart the NDlib models share.

### 2.3 The state grows from 2 binary channels to 3 or 4 exclusive ones

Today `State` is `(infected, frontier)` — two overlapping binary sets — and `X` is `(N, 6)` with `CH_INFECTED`, `CH_FRONTIER` as channels 0–1, targets `y_inf`, `y_fr`.

Under SIR the natural state is the one-hot triple `(S, I, R)`; under SEIR the quadruple `(S, E, I, R)`. These are **mutually exclusive**, which is a different object from our two overlapping indicators:

|                       | today (IC/LT)              | SIR                                                         | SEIR                   |
| --------------------- | -------------------------- | ----------------------------------------------------------- | ---------------------- |
| state channels in `X` | 2 (`INFECTED`, `FRONTIER`) | 3 (`S`, `I`, `R`)                                           | 4 (`S`, `E`, `I`, `R`) |
| `IN_CHANNELS`         | 6                          | **7**                                                       | **8**                  |
| target head width     | 2 (`y_inf`, `y_fr`)        | **3**                                                       | **4**                  |
| target semantics      | two marginals              | a **simplex** — rows sum to 1                               | a simplex              |
| loss                  | 2 × BCEWithLogits          | cross-entropy over 3 classes, or 3 BCEs + a simplex penalty | same, 4-way            |

One channel is saved because `FRONTIER` (currently spreading) collapses into `I` under SIR — a node is infectious exactly while it is in `I`. So the honest count is `IN_CHANNELS = 6 → 7` for SIR (S, I, R, DEGREE, ADD, REMOVE, EDGE) and `→ 8` for SEIR. `wm_metrics.py`'s `delta_f1` and `new_infection_f1` generalize as "F1 over nodes whose compartment changed", but `add_seed_success` / `remove_frontier_success` need renaming and re-deriving.

### 2.4 ⚠️ Monotonicity breaks — `ICTransmissionHead` cannot be reused

This is the real cost, and it is not a tuning problem. `ICTransmissionHead` composes the next state as [read from `world_model/wm_model.py`]:

```python
p_newly = (1.0 - infected) * p_new     # susceptibles only
y_inf   = infected + p_newly           # monotone: infected stay infected
y_fr    = p_newly                      # new frontier = newly infected
```

`y_inf = infected + (1 − infected)·p_new` is **monotone non-decreasing in `infected` by construction**. `p_new ≥ 0`, so `y_inf ≥ infected` always. There is no assignment of encoder weights, no value of `q(u→v)`, that makes this expression predict a node leaving the infected set. The same is true of `LTThresholdHead`, whose `y_inf = active + (1 − active)·p_new` has the identical shape.

The docstring is explicit that this was the _point_ — "Locality + monotonicity are baked in, so a free-running rollout cannot saturate spuriously" — and `RESULTS.md` records that this structure is what fixed rollout saturation (`count_bias` +49 → +0.27). It is load-bearing, and it is exactly the assumption SIR and SIS violate:

- **SIR**: `I → R` at rate `γ`. A node in `I` at `t` may be in `R` at `t+1`. Under the "ever-infected" reading `y_inf` is still monotone, but the _infectious_ set is not, and the infectious set is what drives transmission. A head that cannot shrink `I` will predict a cascade that never stops transmitting — the saturation failure mode, back again through a different door.
- **SIS**: `I → S` at rate `λ`. Even "ever-infected" is not the state any more; the model must predict nodes returning to `S` and being re-infected later. Monotone composition is flatly wrong.

**What the replacement must be: a per-node transition _matrix_, not a probability.** For SIR the head has to emit, for every node `v`, a row-stochastic 3×3 matrix `P_v ∈ Δ³ˣ³` and compose `s_{t+1}(v) = P_vᵀ s_t(v)`, with structure:

```
        S                 I            R
S   1 − p_inf(v)       p_inf(v)        0            <- infection is the only S exit
I        0             1 − γ̂(v)      γ̂(v)          <- recovery is the only I exit
R        0                0            1            <- absorbing (SIR); ← σ̂(v) for SIRS
```

Only `p_inf(v)` needs the graph — it is exactly the current `p_new(v) = 1 − Π(1 − q_uv · I_u)` construction, and it can be lifted verbatim from `ICTransmissionHead`. `γ̂(v) = sigmoid(Linear(h_v))` is a per-node scalar with no graph term, structurally identical to `LTThresholdHead`'s `θ̂_v`. For SIS the matrix is 2×2 with `λ̂(v)` in the `I→S` cell; for SEIR it is 4×4 with an extra `α̂(v)` on `E→I`. Rows are enforced by a softmax over each row's free entries, which keeps the simplex constraint exact rather than penalized.

This is a **new head class** (~80 lines) reusing the existing per-edge transmission MLP, not an edit to the existing ones. `ICTransmissionHead` and `LTThresholdHead` must be left intact for IC/LT.

### 2.5 The five ops map cleanly onto the interventions — with one caveat

| Intervention in the literature                     | Our op                     | Notes                                                                                                                                                                                                                                                                                                                                                                     |
| -------------------------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Vaccination / immunization** (permanent)         | `remove_node`              | Under **SIR this is exact**: our `apply_actions` already sets status `2` for IC, and NDlib's SIR status `2` _is_ `Removed` — immune, non-infectious, cannot be re-infected. Under **SIS it is wrong**: SIS has no `R`, so the code's `else: 0` branch puts the node back to `Susceptible`. SIS vaccination needs either a real node deletion or an added absorbing state. |
| **Quarantine / isolation** (temporary)             | `remove_node`, then re-add | Not currently expressible — our `remove_node` has no duration. Either add a `duration` field to `ActionOp`, or emit `remove_edge` on all incident edges at `t` and `add_edge` at `t + d`. The second needs no schema change and is what we should do first.                                                                                                               |
| **Social distancing / contact reduction** (graded) | `set_edge_weight`          | Conceptually perfect — scale `β_uv` down by a compliance factor. **Blocked by §2.2**: NDlib's SIR/SIS/SEIR have no per-edge parameter. Needs our own stepper.                                                                                                                                                                                                             |
| **Link removal / travel ban**                      | `remove_edge`              | Direct. This is the Van Mieghem / NetMelt / Gelling intervention.                                                                                                                                                                                                                                                                                                         |
| **New contact / superspreader event**              | `add_edge`                 | Direct; the "melting" direction of Gelling.                                                                                                                                                                                                                                                                                                                               |
| **Contact tracing**                                | ❌ **not an action**       | Tracing reveals who was exposed — it changes the _observation_, not the graph or the state. It belongs in a POMDP observation model, which we do not have. RLGN and IDRLECA both model testing this way. Say so explicitly rather than forcing it into an op.                                                                                                             |

### 2.6 Metrics we would have to add

`wm_eval.py` measures one-step accuracy and rollout fidelity; epidemic control is graded on outbreak _shape_, which needs new quantities on top of the rollout we already produce:

| Metric                                | Definition                                      | Do we have it?                                                                                                                                                                        |
| ------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Final epidemic size / attack rate** | `\|R(∞)\| / N`                                  | ✅ `ens_final_count_model/true` renamed                                                                                                                                               |
| **Peak prevalence**                   | `max_t\|I(t)\| / N`                             | ❌ new — but a one-line reduction over the rollout trace                                                                                                                              |
| **Time to peak**                      | `argmax_t\|I(t)\|`                              | ❌ new, same trace                                                                                                                                                                    |
| **Area under the infection curve**    | `Σ_t\|I(t)\|`                                   | ❌ new, same trace                                                                                                                                                                    |
| **`R₀` / `R_eff(t)`**                 | mean secondary infections per case              | ❌ new; needs per-edge attribution, which our structured head actually _does_ expose via `q_uv`                                                                                       |
| **Epidemic threshold `λ₁(A)`**        | largest adjacency eigenvalue after intervention | ❌ new, but it is `scipy.sparse.linalg.eigsh(A, k=1)` — it is not a model metric at all, it is computed on the graph, so **it works as a baseline objective without any world model** |

The last row matters for positioning: the spectral line (§3) optimizes a quantity that needs no simulator. Our world model can only beat it on the objective it does _not_ optimize — the actual simulated outbreak — so **the evaluation must report simulated final size, not eigendrop**, or we hand the comparison away.

### 2.7 Honest cost, relative to the other candidate tasks

| Work item                                       | Estimate                                                         |
| ----------------------------------------------- | ---------------------------------------------------------------- |
| Simulator dispatch for SIR/SIS/SEIR (§2.1)      | ~60 lines, 1 day                                                 |
| Own stepper for heterogeneous `β_uv` (§2.2)     | ~40 lines, +0.5 day — **optional but recommended**               |
| `State`/`X`/target widening to a simplex (§2.3) | ~120 lines across `wm_simulator.py`, `wm_data.py`, `train_wm.py` |
| `CompartmentTransitionHead` (§2.4)              | ~80 lines, the only genuinely new modelling                      |
| Epidemic-curve metrics (§2.6)                   | ~60 lines in `wm_metrics.py`                                     |
| Contact-network loaders (§6.3)                  | ~40 lines each, 5–8 datasets                                     |

**Roughly 1–1.5 weeks.** That places it above [`source_localization.md`](source_localization.md) (nothing new to simulate) and [`influence_estimation.md`](influence_estimation.md) (already computed), on par with [`influence_blocking.md`](influence_blocking.md) (2-cascade state), and well below [`cascading_failure.md`](cascading_failure.md) (a whole new `T_endo`).

The distinguishing argument for doing it: **it is the only candidate task that forces us to confront non-monotone dynamics.** Every other ✅-fit task in this folder keeps the monotone assumption our structured heads are built on. If the world model is meant to be a general simulator rather than an IC simulator, this is the task that proves it — or fails to.

---

## 3. Classical and heuristic methods

### 3.1 The physics line — targeted vs. random immunization

| Method                                        | Year | Venue         | Idea                                                                                                                                                                                       | Paper                                                                                          | Code |
| --------------------------------------------- | ---- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- | ---- |
| **Epidemic spreading in scale-free networks** | 2001 | PRL 86:3200   | Establishes that a power-law network has **no epidemic threshold** as `N → ∞`. The result the whole field reacts to.                                                                       | [Pastor-Satorras & Vespignani, arXiv cond-mat/0010317](https://arxiv.org/abs/cond-mat/0010317) | —    |
| **Targeted immunization**                     | 2002 | PRE 65:036104 | Immunize the highest-degree nodes. Restores a finite threshold with a vanishing immunized fraction; **random immunization needs a fraction → 1**.                                          | [Pastor-Satorras & Vespignani, arXiv cond-mat/0107066](https://arxiv.org/abs/cond-mat/0107066) | —    |
| **Acquaintance immunization** ⭐              | 2003 | PRL 91:247901 | Pick a random node, immunize a **random neighbour** of it. Needs no global degree knowledge, still hits hubs (friendship paradox). The standard "no-information" baseline.                 | [Cohen, Havlin & ben-Avraham, arXiv cond-mat/0207387](https://arxiv.org/abs/cond-mat/0207387)  | —    |
| **Attack vulnerability of complex networks**  | 2002 | PRE 65:056109 | The four-way ID/IB/RD/RB taxonomy — **initial vs. recalculated** degree/betweenness removal. Recalculated is consistently stronger; the source of the "HDA" baseline in every later paper. | [Holme, Kim, Yoon & Han, arXiv cond-mat/0202410](https://arxiv.org/abs/cond-mat/0202410)       | —    |

### 3.2 The spectral line — minimize `λ₁`

This is the branch with real algorithms and comparable numbers.

| Method                                                   | Year        | Venue                           | Idea                                                                                                                                                                                                                                                     | Paper                                                                                                                                                                                               | Code                                                                                                      |
| -------------------------------------------------------- | ----------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **Eigenvalue threshold** ⭐                              | 2003        | SRDS                            | Proves SIS dies out iff `λ₁ · β/δ < 1`. **The foundational result.** Verified title by text extraction.                                                                                                                                                  | [Wang, Chakrabarti, Wang & Faloutsos (CMU PDF)](https://www.cs.cmu.edu/~christos/PUBLICATIONS/srds03-virus.pdf)                                                                                     | —                                                                                                         |
| **Epidemic thresholds in real networks**                 | 2008        | ACM TISSEC 10(4)                | The journal version, with the NLDS derivation and empirical validation.                                                                                                                                                                                  | [Chakrabarti, Wang, Wang, Leskovec & Faloutsos (CMU PDF)](http://www.cs.cmu.edu/~deepay/mywww/papers/tissec08.pdf) · [ACM DOI](https://dl.acm.org/doi/10.1145/1284680.1284681) (403 paywall)        | —                                                                                                         |
| **Threshold conditions for arbitrary cascade models** ⭐ | 2011        | ICDM                            | Generalizes `λ₁ · s < 1` to **any** propagation model expressible as a node-level Markov chain (SIR, SIRS, SEIR, SIV, …), with `s` a model-specific scalar. This is what licenses using one surrogate for every compartmental model.                     | [Prakash et al. ICDM'11 PDF](https://faculty.cc.gatech.edu/~badityap/papers/gen-threshold-icdm11.pdf) · [KAIS'12 extended](https://faculty.cc.gatech.edu/~badityap/papers/gen-threshold-kais12.pdf) | —                                                                                                         |
| **NetShield / NetShield+** ⭐                            | 2010 / 2015 | ICDM / TKDE                     | Node immunization. Defines the **Shield-value** `Sv(S) = Σ 2λu(i)² − Σ A(i,j)u(i)u(j)`, proves it is submodular, greedy selects `k` nodes in `O(nk² + m)`. **The canonical node baseline.**                                                              | [ICDM'10 PDF](https://faculty.cc.gatech.edu/~badityap/papers/netshield-icdm10.pdf) · [TKDE'15 PDF](https://chenannie45.github.io/netshield-tkde15.pdf)                                              | no official release; a `NetShield` function ships in [EpiLearn](https://github.com/Emory-Melody/EpiLearn) |
| **NetMelt / NetGel (Gelling)** ⭐                        | 2012        | CIKM (best paper)               | The **edge** counterpart: delete `k` edges to minimize `λ₁` (NetMelt), or add `k` edges to maximize it (NetGel). Score for edge `(i,j)` is `u(i)·v(j)` (left/right first eigenvectors). **The canonical edge baseline.**                                 | [Tong, Prakash, Eliassi-Rad, Faloutsos & Faloutsos, CIKM'12 PDF](https://faculty.cc.gatech.edu/~badityap/papers/netgel-cikm12.pdf)                                                                  | no public code found                                                                                      |
| **Decreasing `λ₁` by link removals**                     | 2011        | PRE 84:016101                   | Proves the edge version NP-hard; introduces the `deg(u)·deg(v)` (**ProductDegree**) and `\|x(u)·x(v)\|` (**EigenScore**) heuristics that every later paper baselines against.                                                                            | [Van Mieghem et al., APS](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.84.016101) (403 paywall)                                                                                           | —                                                                                                         |
| **GreedyWalk / PrimalDual** ⭐                           | 2015        | SDM                             | First **approximation guarantees** for spectral-radius minimization. Scores an edge by the number of closed `k`-walks through it; `PrimalDual` gives an `O(log n)`-type bound. Handles both edge (SRME) and node (SRMN) variants with non-uniform costs. | [Saha, Adiga, Prakash & Vullikanti, arXiv 1501.06614](https://arxiv.org/abs/1501.06614) · [PDF](https://faculty.cc.gatech.edu/~badityap/papers/greedywalk-sdm15.pdf)                                | [tinyurl.com/l3lgsq7](http://tinyurl.com/l3lgsq7) (link from the paper, **not verified**)                 |
| **Fractional Immunization**                              | 2013        | SDM                             | Drops the all-or-nothing assumption: allocate a **continuous** amount of resource per node/edge under a budget, via convex optimization on the `λ₁` surrogate. Motivated by hospital-transfer networks.                                                  | [Prakash, Adamic, Iwashyna, Tong & Faloutsos, SDM'13 PDF](https://faculty.cc.gatech.edu/~badityap/papers/smartalloc-sdm13.pdf)                                                                      | no public code found                                                                                      |
| **Optimal resource allocation** ⭐                       | 2014        | IEEE TCNS 1(1)                  | Casts vaccine + antidote allocation as a **geometric program** — convex, so globally optimal, on arbitrary directed weighted graphs. Both rate-constrained and budget-constrained forms.                                                                 | [Preciado, Zargham, Enyioha, Jadbabaie & Pappas, arXiv 1309.6270](https://arxiv.org/abs/1309.6270)                                                                                                  | no public code found                                                                                      |
| **Analysis and Control of Epidemics** (survey)           | 2016        | IEEE Control Systems Mag. 36(1) | The control-theory survey: every SIS/SIR mean-field model, every stability result, every allocation formulation, in one place.                                                                                                                           | [Nowzari, Preciado & Pappas, arXiv 1505.00768](https://arxiv.org/abs/1505.00768)                                                                                                                    | —                                                                                                         |

### 3.3 The simulation line — optimize expected outbreak size directly

| Method                          | Year        | Venue          | Idea                                                                                                                                                                                                                                                                   | Paper                                                                                                                                                                   | Code                 |
| ------------------------------- | ----------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| **Blocking links**              | 2009        | ACM TKDD 3(2)  | Greedy edge removal minimizing expected IC contamination, with a bond-percolation estimator. The IC-side ancestor of influence blocking.                                                                                                                               | [Kimura, Saito & Motoda, ACM DOI](https://dl.acm.org/doi/10.1145/1514888.1514892) (403 paywall)                                                                         | —                    |
| **DAVA / DAVA-fast** ⭐         | 2014        | SDM            | **Data-aware** vaccine allocation: given the _observed_ set of already-infected nodes, merge them into a supernode, build a dominator tree of the reachable subgraph, and solve exactly on the tree. Gets a provable approximation on trees; DAVA-fast is near-linear. | [Zhang & Prakash, SDM'14 PDF](https://faculty.cc.gatech.edu/~badityap/papers/dava-sdm14.pdf)                                                                            | no public code found |
| **DAVA journal version**        | 2015        | ACM TKDD 10(2) | Adds the uncertain-prior and multiple-strain variants.                                                                                                                                                                                                                 | [Zhang & Prakash, TKDD PDF](https://faculty.cc.gatech.edu/~badityap/papers/dava-vdp-tkdd15.pdf)                                                                         | no public code found |
| **Group immunization**          | 2015 / 2016 | ICDM / TKDE    | Immunize _groups_ (households, wards, age bands) rather than individuals — the realistic-policy variant.                                                                                                                                                               | [ICDM'15 PDF](https://faculty.cc.gatech.edu/~badityap/papers/groupvacc-icdm15.pdf) · [TKDE'16 PDF](https://faculty.cc.gatech.edu/~badityap/papers/groupimmu-tkde16.pdf) | no public code found |
| **Data-driven immunization**    | 2017 / 2018 | ICDM / KAIS    | Immunization when the graph is only partially observed and the infection log is the primary signal.                                                                                                                                                                    | [ICDM'17 PDF](https://faculty.cc.gatech.edu/~badityap/papers/dataimm-icdm17.pdf) · [KAIS'18 PDF](https://faculty.cc.gatech.edu/~badityap/papers/dataimm-kais18.pdf)     | no public code found |
| **Temporal vaccine allocation** | 2020        | AAMAS          | Allocation on a **time-varying** contact network — the SocioPatterns-style setting.                                                                                                                                                                                    | [Prakash group, AAMAS'20 PDF](https://faculty.cc.gatech.edu/~badityap/papers/tempvacc-aamas2020.pdf)                                                                    | no public code found |

**Reading the two lines against each other.** The spectral line optimizes a surrogate that needs no simulation and is therefore fast, model-agnostic, and _wrong in a specific way_: `λ₁` says nothing about **where** the epidemic currently is. DAVA's entire contribution is showing that once you condition on the observed infected set, the `λ₁`-optimal allocation is no longer the right one. **That is the same argument our world model makes** — and it is why DAVA, not NetShield, is the closest classical analogue to what we do (§9).

---

## 4. Learning-based methods

### 4.1 RL for sequential intervention — the branch that matches us

| Method                                        | Year        | Venue                       | Approach                                                                                                                                                                                                                                               | Paper                                                                                                                             | Code                                                                                                                     |
| --------------------------------------------- | ----------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **RLGN** ⭐                                   | 2021        | **ICML**                    | Two GNNs (local diffusion + long-range information) over a temporal contact graph, PPO-style RL, ranks nodes for testing/isolation each step under a `k`-per-step budget and partial observability. **The single most comparable paper to our setup.** | [arXiv 2010.05313](https://arxiv.org/abs/2010.05313) · [PMLR v139](https://proceedings.mlr.press/v139/meirom21a.html)             | **no public code found** (checked arXiv, PMLR, NVIDIA project page, GitHub search)                                       |
| **DURLECA**                                   | 2020        | KDD                         | RL over a **mobility** graph; the action is a per-edge mobility-restriction multiplier, i.e. graded `set_edge_weight`. Reward trades off infections against economic cost. Uses a GNN ("Flow-GNN") as the transition approximator.                     | [arXiv 2008.01257](https://arxiv.org/abs/2008.01257)                                                                              | [github.com/AnyLeoPeace/DURLECA](https://github.com/AnyLeoPeace/DURLECA) — **code only, Beijing mobility data withheld** |
| **IDRLECA**                                   | 2021 → 2023 | arXiv → ACM TKDD 17(3)      | Contact-tracing + intervention via deep RL; GNN state encoder over the infection graph, per-node isolate/test actions.                                                                                                                                 | [arXiv 2102.08251](https://arxiv.org/abs/2102.08251)                                                                              | no public code found                                                                                                     |
| **Deep RL for large-scale epidemic control**  | 2020        | ECML-PKDD (ADS)             | DQN over a stochastic metapopulation model of Great Britain; actions are school-closure / social-distancing per region.                                                                                                                                | [arXiv 2003.13676](https://arxiv.org/abs/2003.13676) · [Springer](https://link.springer.com/chapter/10.1007/978-3-030-67670-4_10) | none stated in the paper                                                                                                 |
| **RL for COVID-19 mitigation policies**       | 2020        | arXiv / AAAI-21 wkshp       | RL over an agent-based pandemic simulator; the _simulator_ is the contribution as much as the policy.                                                                                                                                                  | [arXiv 2010.10560](https://arxiv.org/abs/2010.10560)                                                                              | [SonyResearch/PandemicSimulator](https://github.com/SonyResearch/PandemicSimulator)                                      |
| **GNN + DRL vaccine prioritization**          | 2023        | arXiv → IEEE JBHI           | Cooperating GNNs + DRL for allocating limited doses across a spatial contact network.                                                                                                                                                                  | [arXiv 2305.05163](https://arxiv.org/abs/2305.05163)                                                                              | no public code found                                                                                                     |
| **Optimal control of epidemic spread via RL** | 2020        | Scientific Reports 10:22106 | Tabular/deep RL on compartmental dynamics; a widely-cited but non-graph baseline.                                                                                                                                                                      | [Nature SciRep](https://www.nature.com/articles/s41598-020-79147-8)                                                               | no public code found                                                                                                     |
| **Active screening for recurrent diseases**   | 2021        | AAMAS                       | RL for _whom to screen_ when disease recurs — the closest formulation to a POMDP with an observation budget.                                                                                                                                           | [arXiv 2101.02766](https://arxiv.org/abs/2101.02766)                                                                              | not located                                                                                                              |
| **Hierarchical RL on unknown networks**       | 2023        | arXiv                       | Spread control when the graph itself is not observed.                                                                                                                                                                                                  | [arXiv 2308.14311](https://arxiv.org/abs/2308.14311)                                                                              | not located                                                                                                              |
| **Graph + RL vaccination strategies**         | 2024        | Scientific Reports 14       | GNN-scored vaccination on complex networks vs. centrality baselines.                                                                                                                                                                                   | [Nature SciRep](https://www.nature.com/articles/s41598-024-78626-6)                                                               | not located                                                                                                              |

### 4.2 Adjacent: learned node-removal (structural, not dynamical)

| Method                  | Year | Venue                         | Approach                                                                                                                                                                                                                                                          | Paper                                                          | Code                                                             |
| ----------------------- | ---- | ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------- |
| **FINDER**              | 2020 | Nature Machine Intelligence 2 | GraphSAGE + DQN trained on tiny synthetic graphs, finds key players (dismantling / influencer) on million-node graphs. The strongest _learned_ node-removal baseline; objective is structural, so see [`critical_node_detection.md`](critical_node_detection.md). | [Nature MI](https://www.nature.com/articles/s42256-020-0177-2) | [github.com/FFrankyy/FINDER](https://github.com/FFrankyy/FINDER) |
| **DRL for influencers** | 2023 | arXiv                         | Successor line to FINDER, IM-flavoured.                                                                                                                                                                                                                           | [arXiv 2309.07153](https://arxiv.org/abs/2309.07153)           | not located                                                      |

### 4.3 Forecasting GNNs — _not_ control, but the source of the datasets

These predict case counts; none of them intervene. They matter here only because they define the benchmark suite the 2024–2026 "GNN for epidemics" surveys use, and because EpiLearn packages them.

| Method        | Year | Venue       | Approach                                                 | Paper                                                          | Code                                                    |
| ------------- | ---- | ----------- | -------------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------- |
| **Cola-GNN**  | 2020 | CIKM        | Cross-location attention for long-term ILI prediction    | [PDF](https://yue-ning.github.io/docs/CIKM20-colagnn.pdf)      | [amy-deng/colagnn](https://github.com/amy-deng/colagnn) |
| **STAN**      | 2021 | JAMIA 28(4) | Spatio-temporal attention + an SIR-consistency loss term | [arXiv 2008.04215](https://arxiv.org/abs/2008.04215)           | [v1xerunt/STAN](https://github.com/v1xerunt/STAN)       |
| **CausalGNN** | 2022 | AAAI 36     | Causal-module GNN with an embedded compartmental model   | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/21479) | no public code found                                    |
| **EpiGNN**    | 2022 | ECML-PKDD   | Region-aware transmission graph, learned adjacency       | [arXiv 2208.11517](https://arxiv.org/abs/2208.11517)           | [Xiefeng69/EpiGNN](https://github.com/Xiefeng69/EpiGNN) |

### 4.4 Software baselines

| Tool             | Year | What it is                                                                                                                                                                                                                   | Paper                                                                                                                       | Code                                                                                                    |
| ---------------- | ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **NDlib** ⭐     | 2018 | **What we already use.** SI/SIS/SIR/SEIR/SEIS/SWIR/UTLDR + IC/LT/threshold, on NetworkX graphs, stepwise `iteration()`.                                                                                                      | [arXiv 1801.05854](https://arxiv.org/abs/1801.05854) · [IJDSA](https://link.springer.com/article/10.1007/s41060-017-0086-6) | [GiulioRossetti/ndlib](https://github.com/GiulioRossetti/ndlib) · [docs](https://ndlib.readthedocs.io/) |
| **EpiLearn** ⭐  | 2024 | ML toolkit for epidemic modelling — forecasting + source detection, simulation, transformations, and a **`NetShield` implementation**. From the Emory Melody lab. Closest thing to a shared benchmark harness in this space. | [arXiv 2406.06016](https://arxiv.org/abs/2406.06016) · [OpenReview](https://openreview.net/forum?id=yBa0TkD99x)             | [Emory-Melody/EpiLearn](https://github.com/Emory-Melody/EpiLearn)                                       |
| **Covasim**      | 2021 | Agent-based COVID model with a full intervention API (testing, tracing, quarantine, vaccination). Networks are synthetic household/school/work layers, not a supplied graph.                                                 | [PLOS Comp Biol 17(7)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009149)                      | [InstituteforDiseaseModeling/covasim](https://github.com/InstituteforDiseaseModeling/covasim)           |
| **EpiModel**     | 2018 | R package; the reference implementation of stochastic **network** epidemic models (ERGM-based dynamic networks).                                                                                                             | [J Stat Software 84(8)](https://www.jstatsoft.org/v84/i08/)                                                                 | [EpiModel/EpiModel](https://github.com/EpiModel/EpiModel)                                               |
| **GLEaMviz**     | 2011 | Global metapopulation + air-transportation model. Client distribution now defunct.                                                                                                                                           | [BMC Inf Dis 11:37](https://link.springer.com/content/pdf/10.1186/1471-2334-11-37.pdf)                                      | no public repo                                                                                          |
| **EpiSimdemics** | 2008 | SC'08 HPC simulator behind the NDSSL Portland/Miami synthetic populations. Closed source.                                                                                                                                    | [IEEE](https://ieeexplore.ieee.org/document/5214892) (202, bot challenge)                                                   | closed source                                                                                           |

### 4.5 LLM / agent-based (2023–2026)

| Work                                         | Year    | What it is                                                                                                                                                                                                                                                                                                                                               | Paper                                                | Code        |
| -------------------------------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | ----------- |
| **EpiAgent** ⭐                              | 2026    | Multi-agent LLM system that synthesizes a **verified compartmental flow graph**, compiles it to a differentiable PyTorch simulator, then calibrates it. Already reviewed in `research_notes/Literature Review.md` — the single most architecturally relevant paper to our coding-agent loop. ⚠️ Name collides with a 2024 single-cell epigenomics model. | [arXiv 2602.00299](https://arxiv.org/abs/2602.00299) | not located |
| **Epidemic modeling with generative agents** | 2023    | LLM agents make their own protective-behaviour decisions; the epidemic curve emerges.                                                                                                                                                                                                                                                                    | [arXiv 2307.04986](https://arxiv.org/abs/2307.04986) | not located |
| **LLM-empowered ABM survey**                 | 2023/24 | Survey of the whole LLM-agent simulation space                                                                                                                                                                                                                                                                                                           | [arXiv 2312.11970](https://arxiv.org/abs/2312.11970) | —           |

**Surveys:** [A Review of GNNs in Epidemic Modeling (KDD 2024, arXiv 2403.19852)](https://arxiv.org/abs/2403.19852) · [awesome-epidemic-modeling-papers](https://github.com/Emory-Melody/awesome-epidemic-modeling-papers) · [RL for policymaking in epidemic control — scoping review (PLOS ONE 2026)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0351176) · [Nowzari, Preciado & Pappas (arXiv 1505.00768)](https://arxiv.org/abs/1505.00768)

**"ATLAS" is a mis-attribution.** No epidemic-control system by that name exists. The only verifiable ATLAS papers are a continual-learning architecture ([arXiv 2208.05388](https://arxiv.org/abs/2208.05388)) and a 2025 memory architecture — both unrelated. Likewise **"MonRe"**, **"NIRL"**, and a Wan/Kolar RL-vaccine-allocation paper were searched for and **do not exist**.

---

## 5. Published results

⚠️ **This literature is figure-dominated.** NetShield, Gelling, Fractional Immunization, DAVA, and GreedyWalk all publish their headline immunization results as _plots of eigendrop / infected-count vs. budget_, with no per-cell table. What follows is everything that **is** in a table. Where only a figure exists, the row says so and gives no number.

### 5.1 RLGN (ICML 2021) ⭐ the most comparable table

Protocol: temporal contact graph, SIR-style dynamics with an exposed/latent period, **1% of nodes tested per step**, 20 steps, metric = **mean percentile of healthy nodes** at the end. Higher is better.

Dataset sizes **as used by RLGN** [verified, Table S4]:

|       | CA-GrQc | Montreal | Portland | Email  | GEMSEC-RO |
| ----- | ------- | -------- | -------- | ------ | --------- |
| Nodes | 5,242   | 103,425  | 10,000   | 32,430 | 41,773    |
| Edges | 14,496  | 630,893  | 199,167  | 54,397 | 222,887   |

**CA-GrQc at 5,242 / 14,496 is byte-comparable to ours** ✅ (we load 5,242 / 14,484 after dropping 12 duplicate/self-loop lines — see [`influence_maximization.md`](influence_maximization.md) §6).

**Epidemic test prioritization — mean percentile of healthy nodes after 20 steps** [verified, Table 2; the paper states std < 0.1 in all cells]:

| Method          | CA-GrQc  | Montreal | Portland | Enron    | GEMSEC-RO |
| --------------- | -------- | -------- | -------- | -------- | --------- |
| Degree          | 25.5     | 12.8     | 0.7      | 71.1     | 2.4       |
| Eigenvector     | 25.4     | 8.1      | 0.04     | 55.1     | 2.4       |
| Supervised (SL) | 29.8     | 23.1     | 1.6      | 68.5     | 4.3       |
| **RLGN** ⭐     | **42.7** | **39.7** | **3.71** | **89.2** | **6.5**   |

**Synthetic ablation — % healthy on preferential-attachment (PA) and contact-tracing (CT) networks, k = 2 tests/step** [verified, Table 1]:

| Method            | PA         | CT         |
| ----------------- | ---------- | ---------- |
| Tree-based        | 10 ± 7     | 11 ± 3     |
| Counter model     | 7 ± 7      | 14 ± 5     |
| Degree            | 30 ± 2     | 16 ± 1     |
| Eigenvector       | 30 ± 1     | 16 ± 1     |
| SL (vanilla)      | 13 ± 3     | 17 ± 1     |
| SL + GNN          | 34 ± 3     | 32 ± 2     |
| SL + degree       | 15 ± 3     | 18 ± 1     |
| SL + degree + GNN | 33 ± 3     | 32 ± 1     |
| RL (vanilla)      | 17 ± 1     | 16 ± 1     |
| **RLGN** ⭐       | **52 ± 2** | **40 ± 1** |

Also [verified, prose]: on a 50,000-node PA graph (mean degree 2.8) RLGN reaches `51 ± 1` % healthy vs. `21 ± 2` for SL+GNN. Ablating the information module drops the contained-epidemic score from `0.77 ± 0.06` to `0.62 ± 0.10`.

**Reading this table.** Two things matter for us. (a) **Degree and eigenvector tie exactly** on CA-GrQc (25.5 vs 25.4) and GEMSEC-RO (2.4 vs 2.4) — the same heuristic-collapse we see on BA graphs; the discriminative graphs here are Montreal and Portland, where degree gets 12.8 and 0.7. (b) The absolute numbers are _low_ — 3.71% healthy on Portland means the epidemic wins almost everywhere even under the best policy. This is the regime where a world model has room; at Enron's 89.2% it does not.

RLGN's **Influence Maximization** table (its second task) is transcribed in [`influence_maximization.md`](influence_maximization.md)-adjacent form here for completeness — mean percentile of influenced nodes after 15 steps [verified, Table 3]:

| Method            | CA-GrQc        | Montreal       | Enron      | GEMSEC-RO     | CA-HEPTh  |
| ----------------- | -------------- | -------------- | ---------- | ------------- | --------- |
| LIR               | 7.3 ± 0.3      | 86.2 ± 0.7     | 29 ± 0.3   | 0.25 ± 0.02   | 9.2 ± 0.3 |
| LIR (filtered)    | 8.0 ± 0.2      | 86.4 ± 0.7     | 28.8 ± 0.3 | 0.22 ± 0.02   | 8.5 ± 0.3 |
| Degree            | 8.4 ± 0.2      | 85.5 ± 0.8     | 31.6 ± 0.6 | 0.07 ± 0.01   | 9.2 ± 0.3 |
| Degree Discounted | 8.7 ± 0.2      | 85.6 ± 0.7     | 26.7 ± 0.6 | 0.05 ± 0.01   | 8.4 ± 0.2 |
| Eigenvector       | 8.3 ± 0.2      | 82.9 ± 0.8     | 31.8 ± 0.5 | 0.07 ± 0.01   | 2.2 ± 0.2 |
| **RLGN**          | **10.2 ± 0.6** | **87.4 ± 0.5** | 31.3 ± 0.6 | **5.8 ± 0.3** | 9.1 ± 0.5 |

⚠️ RLGN has **no public code** — checked arXiv, PMLR, the NVIDIA project page, and GitHub. Reproducing it means reimplementing it.

### 5.2 GreedyWalk / Saha et al. (SDM 2015) ⭐ the spectral benchmark table

The only paper in the spectral line that publishes a clean network table with **`λ₁` per graph** — which is the quantity the whole branch optimizes, so this table doubles as the benchmark definition.

[verified, Table 2 — "The first two are synthetic random networks; others are taken from [SNAP] and [Network Repository]"]:

| Network                   | Nodes         | Edges         | `λ₁`      |
| ------------------------- | ------------- | ------------- | --------- |
| Barabasi-Albert           | 1,000         | 1,996         | 11.1      |
| Erdos-Renyi               | 994           | 2,526         | 6.38      |
| P2P (Gnutella05)          | 8,846         | 31,839        | 23.55     |
| P2P (Gnutella06)          | 8,717         | 31,525        | 22.38     |
| Collab. Net (HepTh)       | 9,877         | 25,998        | 31.03     |
| **Collab. Net (GrQc)** ✅ | **5,242**     | **14,496**    | **45.62** |
| AS (Oregon 1)             | 10,670        | 22,002        | 58.72     |
| AS (Oregon 2)             | 10,900        | 31,180        | 70.74     |
| Brightkite Net            | 58,228        | 214,078       | 101.49    |
| **Youtube Network** ✅    | **1,134,890** | **2,987,624** | **210.4** |
| Stanford Web graph        | 281,903       | 1,992,636     | 448.13    |

**Two of these are graphs we already load, at exactly the same counts** — `ca_grqc` (5,242 / 14,496 raw; 14,484 after dedup) and `youtube` (1,134,890 / 2,987,624, an exact match). That makes `λ₁` on our YouTube directly checkable against a published number: **210.4**.

The paper also uses the **Portland** NDSSL synthetic contact network in its figures (Fig 5g, Fig 9) and reports the age-group demographics of the removed nodes — but Portland is **not** in Table 2 and its size is never stated.

Result cells: **[figure] only** (Figs 1, 2, 5, 8). The paper's claims, all
[claim]: GreedyWalk beats EigenScore / ProductDegree / LinePagerank / Hybrid
consistently, "especially as the target threshold becomes smaller"; GreedyWalkSparse matches its quality at up to **an order of magnitude** less runtime.

### 5.3 NetShield (ICDM 2010) — datasets and approximation quality

[verified, Table 2]:

| Name                             | n         | m           |
| -------------------------------- | --------- | ----------- |
| Karate                           | 34        | 152         |
| AA (DBLP co-authorship)          | 418,236   | 2,753,798   |
| NetFlix (bipartite, unipartized) | 2,667,199 | 171,460,874 |

⚠️ **The paper contradicts itself on Karate**: Table 2 says `m = 152`, the body text says "we have n = 34 nodes and m = 156 edges". Neither is the canonical Zachary count of **78 undirected edges** — 156 = 2 × 78 arcs, so the _text_ is the arc count and Table 2's 152 is unexplained. Do not quote either.

Approximation quality of the Shield-value `Sv(S)` vs. the true eigendrop, on per-conference co-authorship subgraphs — **larger is better, max 1.0** [verified, Table 3]:

| k   | 'KDD'  | 'ICDM' | 'SDM'  | 'SIGMOD' |
| --- | ------ | ------ | ------ | -------- |
| 1   | 0.9519 | 0.9908 | 0.9995 | 1.0000   |
| 2   | 0.9629 | 0.9910 | 0.9984 | 0.9927   |
| 5   | 0.9721 | 0.9888 | 0.9992 | 0.9895   |
| 10  | 0.9726 | 0.9863 | 0.9987 | 0.9852   |
| 20  | 0.9683 | 0.9798 | 0.9929 | 0.9772   |

The **immunization comparison itself is [figure] only** (Fig 1): NetShield vs. Acquaintance, Degree, Eigenvector ("Eigs"), PageRank, at k = 5 deletions, x-axis the normalized virus strength `s = λ · b / d`, averaged over 1,000 runs. Claims: NetShield's curve is always lowest, and "the performance of 'Eigs' is much worse than the proposed NetShield" — i.e. **collective selection beats top-k of an individual score**, which is the paper's actual contribution.

### 5.4 Preciado et al. (TCNS 2014) — the air-transportation instance

[verified, prose]: airports with incoming traffic > 10 million passengers/year
→ **56 airports** connected by **1,843 directed flights**, edge weight = annual passengers (MPPY). Spectral radius of the weighted digraph **ρ(A_G) = 9.46**. Simulation bounds: `δ ∈ [0.1, 0.5]`, `β ∈ [2.1×10⁻², 4.2×10⁻³]`; without protection `λ₁(β̄ A_G − δ I) = 0.1 > 0`, i.e. the outbreak grows.

Results: **[figure] only** (allocation vs. in-degree, and vs. PageRank).

### 5.5 What has no transcribable table at all

| Paper                               | Why               | What it publishes instead                                        |
| ----------------------------------- | ----------------- | ---------------------------------------------------------------- |
| Gelling / NetMelt (CIKM'12)         | figure-only       | eigendrop vs. k curves, "melting" and "gelling" directions       |
| Fractional Immunization (SDM'13)    | figure-only       | footprint vs. budget on hospital-transfer networks               |
| DAVA (SDM'14)                       | figure-only       | expected saved nodes vs. vaccine budget                          |
| Pastor-Satorras & Vespignani (2002) | analytic          | the critical immunization fraction `g_c` as a closed form        |
| Cohen et al. (2003)                 | analytic + figure | `g_c ≈ 0.15–0.20` for scale-free with `2 < γ < 3` [claim]        |
| DURLECA (KDD'20)                    | data withheld     | infection-vs-mobility Pareto curves on a private Beijing dataset |

---

## 6. Datasets

This literature uses a **different dataset family** from IM. The social graphs that dominate `influence_maximization.md` appear only as convenience benchmarks; the graphs that carry the field are **empirical contact traces**, **air transportation networks**, and **AS/infrastructure graphs**.

### 6.1 What we already load that this literature also uses

| `--dataset`   | Nodes     | Edges               | Used by                                                                   | Match quality                                                  |
| ------------- | --------- | ------------------- | ------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `ca_grqc` ✅  | 5,242     | 14,484 (14,496 raw) | **RLGN** (5,242 / 14,496), **Saha et al.** (5,242 / 14,496, `λ₁ = 45.62`) | **exact** — the raw SNAP file; our 12-line difference is dedup |
| `youtube` ✅  | 1,134,890 | 2,987,624           | **Saha et al.** (1,134,890 / 2,987,624, `λ₁ = 210.4`)                     | **exact, byte-for-byte**                                       |
| `karate` ✅   | 34        | 78                  | **NetShield** (as "Karate")                                               | ⚠️ NetShield's counts (152 / 156) are inconsistent — see §5.3  |
| `ba`, `er` ✅ | —         | —                   | Saha et al. (BA 1,000 / 1,996, `λ₁` 11.1; ER 994 / 2,526, `λ₁` 6.38)      | reproducible with `--ba-m`/`--er-p`                            |
| `facebook` ✅ | 4,039     | 88,234              | occasional immunization benchmark                                         | [claim]                                                        |

**That is the whole intersection: two exact real-graph matches.** Everything else this literature reports on is a graph we do not have. That is the honest headline of this section.

### 6.2 ⚠️ SocioPatterns — the contact-trace family

These are the field's canonical _empirical_ networks: RFID proximity sensors worn by real people, 20-second resolution, in a school / hospital / conference / office. They are what makes epidemic control on networks an empirical problem rather than a synthetic one.

**The dataset index moved.** `http://www.sociopatterns.org/datasets/` now returns **404** [verified by curl, 2026-07-28]; the live index is [`https://sociopatterns.org/datasets.html`](https://sociopatterns.org/datasets.html) (200). Individual `/datasets/<slug>/` pages still resolve. **SocioPatterns publishes no statistics page** — every count below was **[derived]** by downloading the file and counting, cross-checked against the [Netzschleuder](https://networks.skewed.de/) mirror's `num_vertices`.

All undirected. Base URL `https://www.sociopatterns.org/assets/data/`. "Edges" = unique undirected dyads in the aggregated graph; "Contacts" = timestamped interaction events; "Steps" = distinct timestamps.

| Dataset                          | Nodes  | Edges  | Avg deg | Contacts | Steps  | Direct file (all HTTP 200)                                                                                                   |
| -------------------------------- | ------ | ------ | ------- | -------- | ------ | ---------------------------------------------------------------------------------------------------------------------------- |
| **Hospital ward (LH10)** ⭐      | 75     | 1,139  | 30.4    | 32,424   | 9,453  | [`hospital_lyon_contacts.dat.gz`](https://www.sociopatterns.org/assets/data/hospital_lyon_contacts.dat.gz)                   |
| **Primary school (temporal)** ⭐ | 242    | 8,317  | 68.7    | 125,773  | 3,100  | [`primaryschool.csv.gz`](https://www.sociopatterns.org/assets/data/primaryschool.csv.gz)                                     |
| Primary school, cumulative day 1 | 236    | 5,899  | 50.0    | —        | —      | [`sp_school_day1.gexf.gz`](https://www.sociopatterns.org/assets/data/sp_school_day1.gexf.gz)                                 |
| Primary school, cumulative day 2 | 238    | 5,539  | 46.6    | —        | —      | [`sp_school_day2.gexf.gz`](https://www.sociopatterns.org/assets/data/sp_school_day2.gexf.gz)                                 |
| **High school Thiers13** ⭐      | 327    | 5,818  | 35.6    | 188,508  | 7,375  | [`HighSchool2013_proximity_net.csv.gz`](https://www.sociopatterns.org/assets/data/HighSchool2013_proximity_net.csv.gz)       |
| High school Thiers12             | 180    | 2,220  | 24.7    | 45,047   | 11,273 | [`highschool_2012.csv.gz`](https://www.sociopatterns.org/assets/data/highschool_2012.csv.gz)                                 |
| High school Thiers11             | 126    | 1,710  | 27.1    | 28,561   | 5,609  | [`highschool_2011.csv.gz`](https://www.sociopatterns.org/assets/data/highschool_2011.csv.gz)                                 |
| **SFHH conference**              | 403    | 9,565  | 47.5    | 70,261   | 3,509  | [`SFHH_tij.dat.gz`](https://www.sociopatterns.org/assets/data/SFHH_tij.dat.gz)                                               |
| **Hypertext 2009 (HT09)**        | 113    | 2,196  | 38.9    | 20,818   | 5,246  | [`ht2009_contact_list.dat.gz`](https://www.sociopatterns.org/assets/data/ht2009_contact_list.dat.gz)                         |
| Workplace InVS15                 | 217    | 4,274  | 39.4    | 78,249   | 18,488 | [`workplace_InVS15_tij.dat.gz`](https://www.sociopatterns.org/assets/data/workplace_InVS15_tij.dat.gz)                       |
| Workplace InVS13                 | 92     | 755    | 16.4    | 9,827    | 7,104  | [`workplace_InVS_tij.dat.zip`](https://www.sociopatterns.org/assets/data/workplace_InVS_tij.dat.zip)                         |
| Kenyan households (Kilifi)       | 75     | 576    | 15.4    | 32,643   | n/a¹   | [`scc2034_household_contact_dataset.zip`](https://www.sociopatterns.org/assets/data/scc2034_household_contact_dataset.zip)   |
| Malawi village                   | 86     | 347    | 8.1     | 102,293  | 43,438 | [`tnet_malawi_pilot.csv.gz`](https://www.sociopatterns.org/assets/data/tnet_malawi_pilot.csv.gz)                             |
| **INFECTIOUS (Science Gallery)** | 10,972 | 44,517 | 8.1     | 415,912  | 76,944 | [`sciencegallery_infectious_contacts.tgz`](https://www.sociopatterns.org/assets/data/sciencegallery_infectious_contacts.tgz) |

¹ Kenya is not a `tij` trace — rows carry `duration` / `day` / `hour` over 3 days.

**Thiers13 companion networks** (Mastrandrea et al. 2015) — three _different_ relations on the same students, which makes it the only multi-layer instance here: contact diaries **120 nodes / 502 arcs (directed, weighted 1–4)**, friendship survey **134 / 668 arcs (directed)**, Facebook **156 / 1,437 undirected** [all derived].

**Co-presence networks** (Génois & Barrat 2018) — one tarball, [`colocation.tar.gz`](https://www.sociopatterns.org/assets/data/colocation.tar.gz) (63 MB, 200). Co-presence _events_, not dyads [verified from the Netzschleuder `sp_colocation` API]: LyonSchool 242 nodes / 6,594,492 events · Thiers13 332 / 18,613,039 · SFHH 403 / 1,417,485 · InVS15 232 / 1,283,194 · InVS13 100 / 394,247 · LH10 81 / 150,126. ❌ The metadata tarball 404s at both the site's own (typo'd) path and the corrected one — recover labels from Netzschleuder instead.

**Three data-vs-prose mismatches; the data wins** [derived]: SFHH's page says 405 participants, 403 appear in contacts; Thiers13's metadata lists 329 students, 327 appear; InVS15's lists 232, 217 appear.

⚠️ **Netzschleuder's `num_edges` for these is the contact-event count, not the dyad count** — which is why its "average degree" for LyonSchool reads 54,499. Use it to cross-check event counts only. Its Kenya node count (47) is also **wrong**: member IDs are numbered per household, so global dedup collapses distinct people. The correct count is 75 (B=15, E=17, F=8, H=29, L=6).

### 6.3 Air transportation, infrastructure, and the rest of the catalogue

| Dataset                             | Nodes                  | Edges                               | Type                      | Avg deg               | Stats source                      | Direct download                                                                                                                                                                                                                              |
| ----------------------------------- | ---------------------- | ----------------------------------- | ------------------------- | --------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **USAir97** ⭐                      | 332                    | 2,126 undirected                    | Undirected, **weighted**  | 12.81                 | [derived] from `.mtx`             | [`inf-USAir97.zip`](https://nrvis.com/download/data/inf/inf-USAir97.zip) · [page](https://networkrepository.com/inf-USAir97.php)                                                                                                             |
| **OpenFlights** (raw)               | 3,425                  | 37,595 **arcs** (19,257 undirected) | Directed                  | 11.0 out              | [derived] from 67,663 route rows  | [`routes.dat`](https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat) · [`airports.dat`](https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat) · [page](https://openflights.org/data.php) |
| OpenFlights (KONECT)                | 2,939                  | 30,501 **arcs**                     | Directed                  | 20.76 (KONECT's 2m/n) | [verified] KONECT stats page      | [`opsahl-openflights.tar.bz2`](http://konect.cc/files/download.tsv.opsahl-openflights.tar.bz2)                                                                                                                                               |
| **Preciado air network**            | 56                     | 1,843 arcs                          | Directed, weighted (MPPY) | 32.9                  | [verified] from the paper's prose | ❌ not published                                                                                                                                                                                                                             |
| **Oregon-1** (2001-03-31) ⭐        | 10,670                 | 22,002                              | Undirected AS             | 4.12                  | [verified] SNAP                   | [`oregon1_010331.txt.gz`](https://snap.stanford.edu/data/oregon1_010331.txt.gz) · [page](https://snap.stanford.edu/data/oregon1.html)                                                                                                        |
| **Oregon-2** (2001-03-31) ⭐        | 10,900                 | 31,180                              | Undirected AS             | 5.72                  | [verified] SNAP                   | [`oregon2_010331.txt.gz`](https://snap.stanford.edu/data/oregon2_010331.txt.gz) · [page](https://snap.stanford.edu/data/oregon2.html)                                                                                                        |
| **PGP web of trust** (`arenas-pgp`) | 10,680                 | 24,316                              | Undirected                | 4.55                  | [verified] KONECT                 | [`arenas-pgp.tar.bz2`](http://konect.cc/files/download.tsv.arenas-pgp.tar.bz2) · [page](http://konect.cc/networks/arenas-pgp/)                                                                                                               |
| **Hamsterster** (KONECT)            | 1,858                  | 12,534                              | Undirected                | 13.49                 | [verified] KONECT                 | [`petster-friendships-hamster.tar.bz2`](http://konect.cc/files/download.tsv.petster-friendships-hamster.tar.bz2)                                                                                                                             |
| Hamsterster (NetRepo)               | 2,426                  | 16,630                              | Undirected                | 13.71                 | [derived]                         | [`soc-hamsterster.zip`](https://nrvis.com/download/data/soc/soc-hamsterster.zip)                                                                                                                                                             |
| **PPI / Yeast** (`moreno_propro`)   | 1,870                  | 2,277                               | Undirected                | 2.44                  | [verified] KONECT (LCC 1,458)     | [`moreno_propro.tar.bz2`](http://konect.cc/files/download.tsv.moreno_propro.tar.bz2)                                                                                                                                                         |
| **HI-II-14** (human interactome)    | 4,182                  | 13,633                              | Undirected                | 6.52                  | [derived] from `.tsv`             | [`HI-II-14.tsv`](http://interactome-atlas.org/data/HI-II-14.tsv) · [page](http://interactome-atlas.org/download)                                                                                                                             |
| p2p-Gnutella31                      | 62,586                 | 147,892 arcs                        | Directed                  | 4.73                  | [verified] SNAP                   | [`p2p-Gnutella31.txt.gz`](https://snap.stanford.edu/data/p2p-Gnutella31.txt.gz)                                                                                                                                                              |
| p2p-Gnutella05 / 06                 | 8,846 / 8,717          | 31,839 / 31,525                     | Directed                  | —                     | [verified] Saha Table 2           | SNAP                                                                                                                                                                                                                                         |
| Email-Enron                         | 36,692                 | 183,831                             | Undirected                | 10.02                 | [verified] SNAP                   | [`email-Enron.txt.gz`](https://snap.stanford.edu/data/email-Enron.txt.gz)                                                                                                                                                                    |
| Brightkite                          | 58,228                 | 214,078                             | Undirected                | 7.35                  | [verified] SNAP                   | [`loc-brightkite_edges.txt.gz`](https://snap.stanford.edu/data/loc-brightkite_edges.txt.gz)                                                                                                                                                  |
| ca-HepTh                            | 9,877                  | 25,998                              | Undirected                | 5.26                  | [verified] Saha Table 2           | [`ca-HepTh.txt.gz`](https://snap.stanford.edu/data/ca-HepTh.txt.gz)                                                                                                                                                                          |
| Stanford Web graph                  | 281,903                | 1,992,636                           | Directed                  | 14.1                  | [verified] Saha Table 2           | [`web-Stanford.txt.gz`](https://snap.stanford.edu/data/web-Stanford.txt.gz)                                                                                                                                                                  |
| GEMSEC-RO (Deezer Romania)          | 41,773                 | 222,887                             | Undirected                | 10.67                 | [verified] RLGN Table S4          | [`gemsec_deezer_dataset.tar.gz`](https://snap.stanford.edu/data/gemsec_deezer_dataset.tar.gz)                                                                                                                                                |
| Montreal (WiFi hotspot tracing)     | 103,425                | 630,893                             | Undirected                | 12.20                 | [verified] RLGN Table S4          | ❌ source not identified (§11)                                                                                                                                                                                                               |
| Portland (NDSSL synthetic)          | 10,000 (RLGN subgraph) | 199,167                             | Undirected                | 39.8                  | [verified] RLGN Table S4          | ❌ **host dead** (§6.5)                                                                                                                                                                                                                      |
| Dolphins                            | 62                     | 159                                 | Undirected                | 5.13                  | [verified] KONECT                 | [`dolphins.zip`](https://public.websites.umich.edu/~mejn/netdata/dolphins.zip)                                                                                                                                                               |
| Football                            | 115                    | 613                                 | Undirected                | 10.66                 | [verified] KONECT                 | [`football.zip`](https://public.websites.umich.edu/~mejn/netdata/football.zip)                                                                                                                                                               |

### 6.4 ⚠️ Name collisions in this literature

| Name                    | Version A                                                                             | Version B                                                                      | Note                                            |
| ----------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------- |
| **US air traffic**      | `USAir97` **332 / 2,126** — the classic epidemic-paper graph                          | Netzschleuder `us_air_traffic` **2,278 nodes / 6.39M temporal flight records** | completely different objects                    |
| **Hamsterster**         | KONECT **1,858 / 12,534**                                                             | NetRepo **2,426 / 16,630**                                                     | different graphs; papers rarely say which       |
| **PGP**                 | `arenas-pgp` **10,680 undirected** — the immunization one                             | Netzschleuder `pgp_strong` **39,796 directed**                                 | different graphs                                |
| **Oregon**              | 9 weekly snapshots each for Oregon-1 and Oregon-2                                     | —                                                                              | papers say "Oregon" without a date; pin it      |
| **LyonSchool vs. LH10** | LH10 = **hospital ward**, 75 nodes                                                    | LyonSchool = **primary school**, 242 nodes                                     | frequently conflated; they are separate studies |
| **EpiAgent**            | epidemiological modelling agent, [arXiv 2602.00299](https://arxiv.org/abs/2602.00299) | a 2024 single-cell epigenomics foundation model                                | unrelated                                       |

### 6.5 ❌ Datasets that cannot be obtained

| Item                                             | Status                                                                                                                                                                                                                             |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **NDSSL Portland / Miami** synthetic populations | `https://ndssl.bi.vt.edu/synthetic-data/` and `https://www.epimodels.org/` both **fail DNS**. Used by Saha et al., RLGN, and most of the Prakash line. **No live source found.** This is the single biggest data gap in this file. |
| **DURLECA Beijing mobility**                     | Repo is live at [AnyLeoPeace/DURLECA](https://github.com/AnyLeoPeace/DURLECA), README states the data cannot be released for privacy reasons. **Code only.**                                                                       |
| **SafeGraph** mobility                           | Free COVID consortium tier retired; now Dewey, paid/academic licence. Not a direct download.                                                                                                                                       |
| **SocioPatterns co-presence metadata**           | 404 at both the site's own typo'd path (`/assests/`) and the corrected `/assets/` path. Labels recoverable from Netzschleuder's `sp_colocation` mirror.                                                                            |
| **Preciado's 56-airport network**                | never published as a file; reconstructable from OpenFlights + a >10 MPPY filter, but the passenger weights are not in OpenFlights.                                                                                                 |
| **Original HI-II-14 host**                       | `interactome.dfci.harvard.edu` fails DNS; superseded by `interactome-atlas.org` (⚠️ the `www.` variant has an **SSL name mismatch** — use the bare host).                                                                          |
| **Pajek archive** (`vlado.fmf.uni-lj.si`)        | connection timeout ×2. Use the NetworkRepository mirror for USAir97.                                                                                                                                                               |
| **BTER / Chung-Lu**                              | not datasets — **generators**. FEASTPACK moved to [sandialabs/feastpack](https://github.com/sandialabs/feastpack). No canonical N/E.                                                                                               |

### 6.6 What we should add, ranked

| Rank | Dataset                              | Nodes / Edges                     | Why                                                                                                                                      |
| ---- | ------------------------------------ | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | **Hospital LH10**                    | 75 / 1,139                        | 75 nodes — the entire pipeline runs in seconds. The canonical empirical contact network, and the one every intervention paper cites.     |
| 2    | **Primary school**                   | 242 / 8,317                       | Same order as `jazz` (198), strong ground-truth class structure (10 classes + teachers) — the community labels our SBM experiments lack. |
| 3    | **High school Thiers13**             | 327 / 5,818                       | Ships **three relation layers** (contact / friendship / Facebook) — the only multi-layer instance in reach.                              |
| 4    | **Oregon-1 & Oregon-2**              | 10,670 / 22,002 · 10,900 / 31,180 | The spectral line's standard benchmark, with published `λ₁` (58.72, 70.74) we can check ourselves.                                       |
| 5    | **USAir97**                          | 332 / 2,126                       | Weighted, tiny, and the archetypal "epidemic on a transport network" instance.                                                           |
| 6    | **PGP**                              | 10,680 / 24,316                   | Standard immunization benchmark; sparse (avg deg 4.55), which is where degree heuristics are weakest.                                    |
| —    | SFHH, HT09, InVS13/15, Malawi, Kenya | 92–403 nodes                      | cheap once one `tij` loader exists — they are the same format                                                                            |

**The loader work is one parser, not six.** Every SocioPatterns file is `tij` (`timestamp node_i node_j`); aggregating to a static weighted graph is `groupby(pair).size()`. One `data/datasets/sociopatterns.py` with a `--sociopatterns-trace` argument covers rows 1, 2, 3, and the last row of the table above. Oregon / USAir97 / PGP are copy-edits of `ca_grqc.py`.

**⚠️ Downloads stall, they do not fail.** SocioPatterns, KONECT, SNAP and Netzschleuder all return HTTP 200 and then time out mid-transfer on files over ~10 MB. Loaders for these must use resume (`curl -C -` / a `.part` file), the pattern `data/datasets/cora_ml.py` already implements.

---

## 7. Which paper uses which

Cells mark the dataset **as that paper reports it**. Check §6.4 before assuming two papers with the same cell used the same graph.

| Dataset                    | NetShield'10 | Gelling'12 | FracImm'13 | DAVA'14 | GreedyWalk'15 | Preciado'14 | RLGN'21 | DURLECA'20 |
| -------------------------- | ------------ | ---------- | ---------- | ------- | ------------- | ----------- | ------- | ---------- |
| Karate                     | ✔            |            |            |         |               |             |         |            |
| AA (DBLP co-auth)          | ✔            | ✔          |            |         |               |             |         |            |
| NetFlix (bipartite)        | ✔            |            |            |         |               |             |         |            |
| Oregon-1 / Oregon-2        |              | ✔          |            | ✔       | ✔             |             |         |            |
| **ca-GrQc** ✅             |              |            |            |         | ✔             |             | ✔       |            |
| ca-HepTh                   |              |            |            |         | ✔             |             | ✔       |            |
| p2p-Gnutella05/06          |              |            |            |         | ✔             |             |         |            |
| Brightkite                 |              |            |            |         | ✔             |             |         |            |
| **YouTube** ✅             |              |            |            |         | ✔             |             |         |            |
| Stanford Web               |              |            |            |         | ✔             |             |         |            |
| BA / ER synthetic          |              | ✔          | ✔          | ✔       | ✔             |             | ✔       |            |
| **Portland (NDSSL)**       |              |            | ✔          | ✔       | ✔             |             | ✔       |            |
| Hospital-transfer network  |              |            | ✔          |         |               |             |         |            |
| Air transportation         |              |            |            |         |               | ✔           |         |            |
| Montreal (WiFi)            |              |            |            |         |               |             | ✔       |            |
| Email (Leskovec'07)        |              |            |            |         |               |             | ✔       |            |
| GEMSEC-RO                  |              |            |            |         |               |             | ✔       |            |
| Beijing mobility (private) |              |            |            |         |               |             |         | ✔          |

**Bold + ✅ = we already have it.** Two rows. Contrast that with the IM file, where four of our graphs appear in the learning-based literature.

**No paper in this table uses a SocioPatterns trace.** The contact-trace family is used by the _epidemiology_ side (Vanhems, Stehlé, Génois & Barrat, Machens et al.) and by temporal-network immunization papers, not by the CS immunization line, which standardized on Oregon + Portland + collaboration graphs. If we add SocioPatterns we get realism but **no published baseline** — the same position SBM occupies for IM.

---

## 8. Evaluation protocol

### 8.1 Metrics, by branch

| Branch               | Primary metric                                                                         | Budget convention                                   | Averaging                           |
| -------------------- | -------------------------------------------------------------------------------------- | --------------------------------------------------- | ----------------------------------- |
| **Spectral**         | **eigendrop** `Δλ₁ = λ₁(A) − λ₁(A − S)`, or the fraction removed to reach a target `T` | `k` nodes or edges, swept; or `T/ρ(G) ∈ [0.2, 1.0]` | none needed — `λ₁` is deterministic |
| **Simulation**       | expected final infected count, or "footprint" `\|R(∞)\|`                               | `k` doses                                           | 1,000 MC runs (NetShield Fig 1)     |
| **Sequential / RL**  | **% healthy** or **% contained** at horizon `T`                                        | `k` tests per step, often as **1% of N**            | 300 episodes (RLGN Table S1)        |
| **Forecasting GNNs** | RMSE / MAE / PCC on case counts                                                        | n/a                                                 | k-fold over time windows            |

### 8.2 The traps

1. **Eigendrop is not containment.** A method can win on `Δλ₁` and lose on simulated final size, because `λ₁` ignores where the infection currently is. DAVA's whole contribution is that gap. **If we report only eigendrop we are grading ourselves on the surrogate the classical methods were built to optimize** — a comparison we cannot win and that does not test the world model. Report simulated outbreak size.
2. **`β` and `γ` are free parameters and nobody standardizes them.** NetShield reports against a _normalized virus strength_ `s = λ · b/d` swept on the x-axis, which is the honest way; most others fix one `(β, γ)` pair and do not justify it. Any table that fixes `β` is comparable only to itself. Our equivalent choice is `--prob-model weighted` for IC — there is no analogue, because NDlib's SIR has no per-edge `β` at all (§2.2).
3. **Budget conventions clash the same way IM's do.** The spectral line sweeps a _fraction removed_ (0–20% typical); RLGN uses **1% of nodes tested per step** over 20 steps, i.e. a cumulative 20% but re-decided each step. These are not comparable. Our `--budget-pcts 1 5 10 20` speaks to the first; the sequential setting needs a per-step budget flag we do not have.
4. **"% healthy" is `1 − attack rate`, but only at the end.** RLGN's headline numbers (42.7 on CA-GrQc) are terminal-state percentiles after exactly 20 steps. Change the horizon and the ranking can change, because a good policy flattens rather than eliminates.
5. **Temporal contact traces are not static graphs.** Aggregating a `tij` trace to a weighted static graph — which is what our pipeline would do — **throws away the ordering that makes the epidemic non-trivial**. A contact at `t=10` followed by one at `t=20` transmits; the reverse order does not. Papers that aggregate report systematically larger outbreaks. If we add SocioPatterns we must say which we did.
6. **SIS has no terminal state.** Final size is undefined; the metric is the _endemic prevalence_ `lim_{t→∞} |I(t)|/N`, estimated by time-averaging after burn-in. None of our rollout metrics do that.
7. **Recovered ≠ removed.** A node in `R` is still in the graph and still occupies a vaccine dose it did not need. Papers that report "nodes saved" against a baseline of "no intervention" and papers that report against "random vaccination" differ by a large constant.

### 8.3 What we would report

Given the above, an honest table for this task is:

| Column                                                    | Why                                                           |
| --------------------------------------------------------- | ------------------------------------------------------------- |
| `final_attack_rate` model vs. true                        | the objective, directly                                       |
| `peak_prevalence`, `time_to_peak`                         | shape, not just size — where flattening shows up              |
| `auc_infected`                                            | the integral health-system-load proxy                         |
| `count_bias` on `\|I(t)\|`                                | our existing saturation guard, per compartment                |
| `λ₁` after intervention                                   | to place ourselves against the spectral line on _its_ metric  |
| vs. **Degree**, **Acquaintance**, **NetShield**, **DAVA** | the four baselines with the strongest claim to being standard |

`Acquaintance` deserves specific mention: it needs no global information, is two lines of code, and is the baseline that most embarrasses learned methods on sparse graphs. It should be in our table.

---

## 9. Implications for this project

1. **Build it in this order.** (a) SIR only — it is the smallest departure, and `remove_node → status 2` is already exactly vaccination under NDlib's SIR (§2.5). (b) The `CompartmentTransitionHead` (§2.4), which is the only novel modelling. (c) SIS second — it breaks "ever-infected" entirely and needs the endemic-prevalence metric. (d) SEIR last, or never; it adds a latent compartment and no new mechanism.

2. **Write our own stepper rather than fighting NDlib.** NDlib's SIR/SIS/SEIR have no per-edge `β` (§2.2), which kills `set_edge_weight`, kills `structured_residual`, and makes graded social distancing inexpressible. The transition rule is four lines; writing it ourselves recovers all three and removes NDlib's global-`np.random` seeding. Keep NDlib for IC/LT where the per-edge threshold API does work.

3. **The monotone structured head is the thing that has to change, and it is the interesting part.** `y_inf = infected + (1 − infected)·p_new` cannot represent recovery — no weight assignment fixes it. Every other ✅-fit task in this folder preserves monotonicity, so **this is the only task that tests whether the "structured head" idea generalizes beyond IC**, or whether we built an IC-specific trick. That is a paper-level question, not a chore.

4. **Position against DAVA, not NetShield.** NetShield optimizes `λ₁`, needs no simulator, and runs in milliseconds — we cannot beat it on its own metric and should not try. DAVA makes exactly our argument (condition on the _observed_ infection state and the optimal allocation changes) but does it with a dominator-tree heuristic on a single observed snapshot. A learned action-conditioned model is the natural generalization: same claim, learned rather than hand-derived, and it extends to multi-step allocation where DAVA does not.

5. **Add the hospital ward and the primary school first.** 75 and 242 nodes — smaller than Jazz. They are the empirical graphs this literature is actually about, one `tij` parser covers the whole family, and the primary school ships real class labels (the community structure our SBM experiments approximate).

6. **Two exact dataset matches exist and both are worth using immediately**: `ca_grqc` (RLGN and Saha et al., 5,242 / 14,496) and `youtube` (Saha et al., 1,134,890 / 2,987,624 with a published `λ₁ = 210.4` we can verify with one `eigsh` call). The YouTube `λ₁` check costs nothing and validates our loader against a published number for the first time.

7. **RLGN is the comparison target and it has no code.** Its Table 2 is the most comparable published result to what we would produce; reproducing it means reimplementing the method. Budget for that, or report against Degree / Eigenvector / Acquaintance / NetShield only and cite RLGN's numbers as context rather than as a run.

8. **Aggregating contact traces destroys the problem.** Our pipeline consumes a static adjacency. A `tij` trace flattened to a weighted static graph loses the temporal ordering that determines whether a chain transmits at all. Either state clearly that we aggregate (and expect inflated outbreaks), or treat the trace as a sequence of per-step adjacencies — which our `reconstruct_episode_adjacency` machinery is _already shaped for_, since it replays a different adjacency per step. That is a genuinely cheap path to temporal graphs, and worth noticing.

9. **The `λ₁` baseline is free and we should compute it regardless.** It is `scipy.sparse.linalg.eigsh(A, k=1)` on a graph we already have in memory. It gives a model-independent sanity check on any node/edge-removal action the agent proposes, and it is the one number this entire literature agrees on.

---

## 10. Reference list

**Physics / classical immunization** [Pastor-Satorras & Vespignani 2001, scale-free epidemics (arXiv cond-mat/0010317)](https://arxiv.org/abs/cond-mat/0010317) · [Pastor-Satorras & Vespignani 2002, targeted immunization (arXiv cond-mat/0107066)](https://arxiv.org/abs/cond-mat/0107066) · [Cohen, Havlin & ben-Avraham 2003, acquaintance immunization (arXiv cond-mat/0207387)](https://arxiv.org/abs/cond-mat/0207387) · [Holme, Kim, Yoon & Han 2002, attack vulnerability (arXiv cond-mat/0202410)](https://arxiv.org/abs/cond-mat/0202410)

**Spectral threshold and `λ₁` minimization** [Wang, Chakrabarti, Wang & Faloutsos 2003, SRDS (CMU PDF)](https://www.cs.cmu.edu/~christos/PUBLICATIONS/srds03-virus.pdf) · [Chakrabarti et al. 2008, ACM TISSEC (CMU PDF)](http://www.cs.cmu.edu/~deepay/mywww/papers/tissec08.pdf) · [Prakash et al. 2011, arbitrary cascade thresholds (ICDM PDF)](https://faculty.cc.gatech.edu/~badityap/papers/gen-threshold-icdm11.pdf) · [KAIS'12](https://faculty.cc.gatech.edu/~badityap/papers/gen-threshold-kais12.pdf) · [Tong et al. 2010, NetShield (ICDM PDF)](https://faculty.cc.gatech.edu/~badityap/papers/netshield-icdm10.pdf) · [NetShield+ TKDE'15](https://chenannie45.github.io/netshield-tkde15.pdf) · [Tong et al. 2012, Gelling / NetMelt (CIKM PDF)](https://faculty.cc.gatech.edu/~badityap/papers/netgel-cikm12.pdf) · [Van Mieghem et al. 2011, link removals (PRE 84:016101)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.84.016101) — 403 paywall · [Saha et al. 2015, GreedyWalk (arXiv 1501.06614)](https://arxiv.org/abs/1501.06614) · [PDF](https://faculty.cc.gatech.edu/~badityap/papers/greedywalk-sdm15.pdf) · [Prakash et al. 2013, Fractional Immunization (SDM PDF)](https://faculty.cc.gatech.edu/~badityap/papers/smartalloc-sdm13.pdf) · [Preciado et al. 2014, optimal resource allocation (arXiv 1309.6270)](https://arxiv.org/abs/1309.6270) · [Nowzari, Preciado & Pappas 2016, survey (arXiv 1505.00768)](https://arxiv.org/abs/1505.00768)

**Data-aware / simulation-based allocation** [Zhang & Prakash 2014, DAVA (SDM PDF)](https://faculty.cc.gatech.edu/~badityap/papers/dava-sdm14.pdf) · [TKDD'15](https://faculty.cc.gatech.edu/~badityap/papers/dava-vdp-tkdd15.pdf) · [Group immunization ICDM'15](https://faculty.cc.gatech.edu/~badityap/papers/groupvacc-icdm15.pdf) · [TKDE'16](https://faculty.cc.gatech.edu/~badityap/papers/groupimmu-tkde16.pdf) · [Data-driven immunization ICDM'17](https://faculty.cc.gatech.edu/~badityap/papers/dataimm-icdm17.pdf) · [KAIS'18](https://faculty.cc.gatech.edu/~badityap/papers/dataimm-kais18.pdf) · [Temporal vaccine allocation AAMAS'20](https://faculty.cc.gatech.edu/~badityap/papers/tempvacc-aamas2020.pdf) · [Kimura, Saito & Motoda 2009, blocking links (ACM TKDD)](https://dl.acm.org/doi/10.1145/1514888.1514892) — 403 paywall

**Learning-based** [RLGN (arXiv 2010.05313)](https://arxiv.org/abs/2010.05313) · [PMLR v139](https://proceedings.mlr.press/v139/meirom21a.html) · no code · [DURLECA (arXiv 2008.01257)](https://arxiv.org/abs/2008.01257) · [code](https://github.com/AnyLeoPeace/DURLECA) · [IDRLECA (arXiv 2102.08251)](https://arxiv.org/abs/2102.08251) · [Libin et al. (arXiv 2003.13676)](https://arxiv.org/abs/2003.13676) · [Kompella et al. (arXiv 2010.10560)](https://arxiv.org/abs/2010.10560) · [PandemicSimulator](https://github.com/SonyResearch/PandemicSimulator) · [GNN+DRL vaccine prioritization (arXiv 2305.05163)](https://arxiv.org/abs/2305.05163) · [Active screening (arXiv 2101.02766)](https://arxiv.org/abs/2101.02766) · [FINDER (Nature MI 2020)](https://www.nature.com/articles/s42256-020-0177-2) · [code](https://github.com/FFrankyy/FINDER)

**Forecasting GNNs** [Cola-GNN (CIKM'20)](https://yue-ning.github.io/docs/CIKM20-colagnn.pdf) · [code](https://github.com/amy-deng/colagnn) · [STAN (arXiv 2008.04215)](https://arxiv.org/abs/2008.04215) · [code](https://github.com/v1xerunt/STAN) · [CausalGNN (AAAI'22)](https://ojs.aaai.org/index.php/AAAI/article/view/21479) · [EpiGNN (arXiv 2208.11517)](https://arxiv.org/abs/2208.11517) · [code](https://github.com/Xiefeng69/EpiGNN)

**Software** [NDlib (arXiv 1801.05854)](https://arxiv.org/abs/1801.05854) · [code](https://github.com/GiulioRossetti/ndlib) · [docs](https://ndlib.readthedocs.io/) · [EpiLearn (arXiv 2406.06016)](https://arxiv.org/abs/2406.06016) · [code](https://github.com/Emory-Melody/EpiLearn) · [Covasim (PLOS CB 2021)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009149) · [code](https://github.com/InstituteforDiseaseModeling/covasim) · [EpiModel (JSS 2018)](https://www.jstatsoft.org/v84/i08/) · [code](https://github.com/EpiModel/EpiModel) · [GLEaMviz (BMC Inf Dis 2011)](https://link.springer.com/content/pdf/10.1186/1471-2334-11-37.pdf)

**Agentic / LLM** [EpiAgent (arXiv 2602.00299)](https://arxiv.org/abs/2602.00299) · [Generative-agent epidemics (arXiv 2307.04986)](https://arxiv.org/abs/2307.04986) · [LLM-ABM survey (arXiv 2312.11970)](https://arxiv.org/abs/2312.11970)

**Surveys / benchmarks** [GNNs in Epidemic Modeling, KDD'24 (arXiv 2403.19852)](https://arxiv.org/abs/2403.19852) · [awesome-epidemic-modeling-papers](https://github.com/Emory-Melody/awesome-epidemic-modeling-papers) · [RL for epidemic policymaking, scoping review (PLOS ONE 2026)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0351176)

**Data sources** [SocioPatterns (index moved to `datasets.html`)](https://sociopatterns.org/datasets.html) · [Netzschleuder](https://networks.skewed.de/) · [SNAP](https://snap.stanford.edu/data/) · [KONECT](http://konect.cc/networks/) · [Network Repository](https://networkrepository.com/) · [OpenFlights](https://openflights.org/data.php) · [Human Interactome Atlas](http://interactome-atlas.org/download)

---

## 11. Open gaps

Honest list of what this review could **not** establish.

- **NDSSL Portland / Miami is unobtainable.** Both `ndssl.bi.vt.edu` and `epimodels.org` fail DNS. Portland appears in Saha et al., RLGN, DAVA and the Fractional Immunization paper — it is the field's shared synthetic contact network, and there is no live mirror. This blocks direct reproduction of four of the six most relevant papers. **Biggest single gap in this file.**
- **RLGN's "Montreal" network** (103,425 / 630,893, cited to Hoen et al. 2015, WiFi hotspot tracing) — no download URL located.
- **RLGN's "Email" network** is cited to Leskovec et al. 2007 and reported as 32,430 / 54,397, which matches **neither** `email-Enron` (36,692 / 183,831) nor `email-Eu-core` (1,005 / 25,571). Unresolved — and note RLGN's Table 2 column header says "Enron" while Table S4 says "Email".
- **No per-cell results table exists** for NetShield's immunization comparison, Gelling, Fractional Immunization, DAVA, or GreedyWalk. Every one is figure-only. To place ourselves numerically against the spectral line we would have to **run their code** — and NetShield, Gelling, FracImm and DAVA all have **no public release**. Only EpiLearn's `NetShield` reimplementation is available.
- **GreedyWalk's code link** is a `tinyurl` printed in the paper (`http://tinyurl.com/l3lgsq7`) — not verified in this review.
- **Song et al., "Node Immunization over Infectious Period" (CIKM 2015)** was in the brief but was not located; the nearest verified Prakash-group papers are the group-vaccination ICDM'15 / TKDE'16 pair.
- **"ATLAS", "MonRe", "NIRL", and a Wan/Kolar RL-vaccine paper do not exist** in this literature (searched; see §4.5). Treat any citation to them as an error to be corrected upstream.
- **Cohen et al.'s critical immunization fraction** (`g_c ≈ 0.15–0.20` for `2 < γ < 3`) is [claim] — it was not re-derived from the paper's own figure.
- **Karate's edge count in NetShield** is self-contradictory (Table 2: 152; text: 156; canonical: 78 undirected). Unexplained.
- **SocioPatterns publishes no statistics anywhere** — every count in §6.2 is [derived] from the file. They are cross-checked against Netzschleuder, but no authoritative published table exists to verify them against.
- **The epidemic-threshold constant for LT** has no analogue: the `λ₁ · s < 1` result covers Markovian cascade models, and LT's per-episode thresholds are not in that class. Our LT results have no spectral reference point.
