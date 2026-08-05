# Graph Task Research — index

One markdown file per graph task, all in the same format. Each is self-contained: task definition, fit with our methodology, every baseline method with paper and code URLs, every dataset those baselines evaluate on with node/edge counts and source URLs, transcribed result tables, evaluation protocols, and an honest list of what the review could **not** establish.

**9,600 lines · 13 tasks · ~450 methods · ~400 datasets · 1,060 unique URLs**, compiled 2026-07-28/29.

These files are **literature references**, not design documents. What we built lives in `../CLAUDE.md` and the per-package READMEs; what the field published lives here.

**The filenames are load-bearing.** Each one is a key of `pipeline/tasks.py::tasks` and the first level of the results tree, so `research/influence_maximization.md` and `results/influence_maximization/<dataset>/<run>/` always name the same task. `--task` validates against that registry, and a task that is not runnable fails up front with its blocker and a pointer back to its file here.

---

## The tasks

Screened for compatibility with our methodology: an action-conditioned world model `f_θ(G, s_t, a_t) → s_{t+1}` factorized as `s_{t+1} = T_endo(T_exo(s_t, a_t))`, five action ops (`add_node`, `remove_node`, `add_edge`, `remove_edge`, `set_edge_weight`), transitions harvested from NDlib IC/LT, and a coding-agent outer loop that plans against the learned model.

A task earns a slot if it has **(a)** a node- or edge-level state that evolves, **(b)** an intervention expressible in the five ops, and **(c)** a simulator we can harvest transitions from.

| File                                                       | Fit                     | Lines | Comparable published table on a graph we load?                          |
| ---------------------------------------------------------- | ----------------------- | ----- | ----------------------------------------------------------------------- |
| [`influence_maximization.md`](influence_maximization.md)   | yes **implemented**      | 809   | (key) DeepIM Tables 2/3 — Jazz, Cora-ML, Power Grid byte-identical         |
| [`influence_blocking.md`](influence_blocking.md)           | yes **implemented**      | 782   | (key) SandIMIN PVLDB'24 Table 5 — email-Eu-core, YouTube, our exact metric |
| [`source_localization.md`](source_localization.md)         | yes **implemented**      | 964   | (key) SL-VAE KDD'22 Tables 1–4 — 5 of its 7 graphs are ours                |
| [`critical_node_detection.md`](critical_node_detection.md) | yes **implemented**      | 1,001 | (key) CoreHD/BPD Table I — "Grid" is our `power_grid` byte-for-byte        |
| [`epidemic_control.md`](epidemic_control.md)               | yes direct               | 690   | partial — different dataset family (contact networks)                   |
| [`influence_estimation.md`](influence_estimation.md)       | yes **already computed** | 553   | (key) GLIE + SIEA — `ca_grqc`, `nethept`, `netphy`, `youtube` match        |
| [`cascade_reconstruction.md`](cascade_reconstruction.md)   | yes **implemented**      | 763   | (key) DITTO KDD'23 Tables 4–5 — BA, ER, Oregon2 and rt-pol are all ours    |
| [`adaptive_online_im.md`](adaptive_online_im.md)           | yes **implemented**      | 572   | only DeepIM's re-run OIM row; the rest are figures                      |
| [`network_inference.md`](network_inference.md)             | Warning: moderate             | 782   | no — Kronecker synthetics, not our graphs                               |
| [`cascade_prediction.md`](cascade_prediction.md)           | Warning: moderate             | 567   | no — real cascade corpora only                                          |
| [`cascading_failure.md`](cascading_failure.md)             | Warning: moderate             | 728   | Jhun et al. Table I, via Motter–Lai on our `power_grid`                 |
| [`graph_completion.md`](graph_completion.md)               | no poor fit             | 724   | SEAL NeurIPS'18 — `netscience`, `power_grid` byte-identical             |
| [`temporal_forecasting.md`](temporal_forecasting.md)       | no poor fit             | 582   | no — zero dataset overlap                                               |

The two no files exist because "we considered it and here is exactly why it does not fit" is worth more than silence — and both turned out to carry something useful anyway (§9 of each).

---

## Cross-cutting findings

Things that emerged from more than one file, or that change what we should do.

1. **`remove_node` did not mean immunization. yes FIXED.** Found independently by `critical_node_detection.md` §2.3 and `influence_blocking.md` §2. Under IC it set NDlib status `2`, which `active_nodes()` still counted as infected, so pre-emptively immunizing `k` nodes inflated measured spread by `+k` before any diffusion ran; under LT it reset to Susceptible and the node re-activated next step; neither op touched `G`. Correct for IM ("this spreader is spent"), wrong for every containment task. Resolved by `--remove-semantics {spent,blocked}` (`data/wm_simulator.py`): `blocked` deletes the node from the graph, stops counting it, and prevents transmission and re-infection under both dynamics, with deletion expressed as `remove_node(v)` plus a `remove_edge` per incident arc so nothing downstream changed. The value crosses data generation, the head's `T_exo`, and the agent's prompt, and `pipeline/tasks.py` now carries the right one per task. `spent` output is byte-identical to before, so existing checkpoints are unaffected.

2. **NDlib ships no competitive/multi-cascade model.** Verified by enumerating `available_statuses` across every model — `Blocked: -1` is a static non-adopter set, and `CompositeModel` can only express one global tie-break. Influence blocking needs a competitive step written on top of `data/wm_simulator.py`.

3. **Adaptive IM cannot be sold on spread. yes IMPLEMENTED on that basis.** The adaptivity gap is bounded (myopic ∈ [e/(e−1), 4]; full-adoption ≤ ⌈n^{1/3}⌉) and non-adaptive greedy is provably no worse than adaptive greedy on all graphs. The claim has to be **cost**: the MC arm scales with rounds, a forward pass does not. Built accordingly: every `adaptive_*@E` arm is paired with `evolve_*@E` at matched `k`, the gap is computed and printed with that calibration stated above the table, and `evaluator_seconds` carries the actual claim.

4. **Our rollout evaluation is blind to per-step drift.** From `temporal_forecasting.md` §9: DCRNN shows rollout degradation is a separate axis from one-step accuracy. We early-stop on one-step `delta_f1` and report aggregate `ens_count_bias`; neither sees a model that is sharp at `t+1` and drifting by `t+5`. Scheduled sampling is the field's standard fix and needs **zero extra simulator calls**, unlike the DAgger attempt that failed here.

5. **The DeepIM NetScience anomaly is narrowed, not solved.** Its 1,565/13,532 graph is shared with SL-VAE and _only_ SL-VAE; IVGD, GraphSL and SIDSL all use our 1,589/2,742. A shared preprocessing lineage between two papers, and the minority version in its own literature.

6. **Published tables are less trustworthy than they look.** BasicTS+ found the same method on the same data varying **33%** across papers, from normalisation and metric implementation alone. Cascade prediction's per-paper cascade filters move MSLE more than the gap between adjacent methods. Several canonical papers (Rozenshtein KDD'16, Xiao 2018, Farajtabar AISTATS'15, NETRATE) publish **no result tables at all** — any number attributed to them is fabricated.

7. **No published work is action-conditioned.** True in influence estimation (every method answers `σ(S)` on a fixed graph) and in cascading failure (every GNN is one-shot; every planner plans against the exact simulator). That is the contribution _and_ the problem: there is no step-wise baseline to compare against.

8. **Supplying a better forward model is not a contribution; supplying a better inversion is. yes IMPLEMENTED on that basis.** From `source_localization.md` §2.2: SL-VAE states outright that its forward operator is pluggable and reports no significant difference across GAT, MONSTOR and DeepIS. Every inverse task in this folder invites the same trap, because each one's strongest methods are built by learning a forward model and inverting it, and we have a forward model. Dropping ours into that slot reproduces a paper its authors already wrote. The escape is to contribute the `argmax` instead: `source_localization.md` §2.3 reformulates the task as **amortized program search**, where the coding agent searches the space of inversion algorithms and the world model is the forward oracle they call. That reframing is also what restores the six-condition table, since a task with no agent has no arms 3–6. Built exactly that way: the agent writes `localize(graph, observation, budget)`, `self.predict_marginals(seeds)` is the one new primitive and its four bindings ARE conditions 3–6, and the component-swap framing ships as **arm A** (`gradient_free@world_model`, condition 8) so `6 vs A` is a measured comparison rather than an argument. Labels select the program and are never available to it, so the selected artifact runs on cascades with no ground truth at all.

   **Four criteria decide whether the reframing applies**, and they split the inverse tasks cleanly. (1) An inference algorithm with real design freedom, (2) the world model callable as a forward oracle with `G` still a *condition*, (3) a dense reward with ground truth we own, (4) no cheaper exact alternative, because a learned approximation replacing a closed form is a regression.

   | Task | 1 | 2 | 3 | 4 | Verdict |
   | ---- | - | - | - | - | ------- |
   | `cascade_reconstruction` §2.4 | yes | yes | yes | yes | **best fit in the folder**; inner loop runs at ~10⁴ oracle calls per instance |
   | `source_localization` §2.3 | yes | yes | yes | yes | both loops, ~10² calls per instance |
   | `network_inference` §2.6 | yes | no | Warning: | no | **agent fits, world model does not**: `G` is the variable, and NETRATE's likelihood is already convex and closed-form |
   | `influence_estimation`, `cascade_prediction` | no | yes | n/a | n/a | world model only; forecasting has no algorithm to search |

   Two consequences worth carrying. **Network inference is the folder's only inverted case**: everywhere else that fails, it fails on the agent side. And **the reward is the specification**: `cascade_reconstruction.md` §2.6 shows that scoring a program search on the easy half of a metric pair (node set rather than tree) does not merely under-report, it makes the search discard the capability. Check that a trivial baseline scores badly under the chosen reward before running any search.

   **Both inverse tasks are now built on that basis, and the second one is the sharper test.** Cascade reconstruction ships as `--task cascade_reconstruction`: the agent writes `reconstruct(graph, observation, horizon)`, `self.step_marginals(infected, frontier)` is the one new primitive and its four bindings ARE conditions 3–6, and the component-swap framing ships as **arm A** (`decode_free@world_model`, condition 8) — DITTO's Metropolis-Hastings sampler with our learned kernel in place of its mean-field `β̂`, so `6 vs A` is measured rather than argued. Three differences from source localization are worth naming. The inner loop runs at ~10⁴ kernel calls per instance against ~10², which is what makes the cost axis load-bearing rather than a footnote. The reward is `λ·PathPrecision + (1−λ)·EventF1` rather than a single metric, and `trivial_decoder_reward` is computed into every results JSON because the check above is a requirement here, not advice. And it needed one thing source localization did not: a **ground-truth parent**, which NDlib does not emit at all — `--trace-parents` swaps in traced IC/LT models that record the transmission edge, and without it the tree-weighted reward is not computable and the harness refuses to run.

---

## File format

Every file follows the same skeleton. Keep it — the value of the folder is that a question ("who baselines on what?", "where do I download it?") is answered in the same section number regardless of which task you open.

```
# <Task> — Prior Work, Datasets, and Published Results

<2–4 sentence scope statement>

## 0. Verification policy          <- the shared three-tier table, verbatim
## 1. Task definition              <- formal statement + variants + why it exists
## 2. Fit with our methodology     <- state / action ops / objective / what we must build
## 3. Classical and heuristic methods    <- table: Method | Year | Venue | Idea | Paper | Code
## 4. Learning-based methods              <- same table shape
## 5. Published results             <- transcribed tables, one subsection per paper
## 6. Datasets                      <- 6.1 what we already load · 6.2 full catalogue · task-specific data
## 7. Which paper uses which        <- dataset x paper matrix
## 8. Evaluation protocol           <- metrics, budgets, splits, and the traps
## 9. Implications for this project <- what to build, in what order, and what it buys
## 10. Reference list
## 11. Open gaps                    <- what this review could NOT establish
```

### Verification policy (shared by every file)

| Tier           | Meaning                                                                                                              |
| -------------- | -------------------------------------------------------------------------------------------------------------------- |
| **[verified]** | Read from the paper's own table via text extraction, or from the data repository's own statistics page. Trustworthy. |
| **[derived]**  | Computed by us from a file we downloaded, or reconstructed arithmetically from published splits.                     |
| **[figure]**   | Read off a plotted figure — the paper published no table. Approximate, direction only.                               |
| **[claim]**    | Stated in prose by a paper or a secondary source; not cross-checked against a file or table.                         |

**Automated PDF summarizers hallucinate plausible numbers from these papers.** While compiling the IM file, a summarizer reported IRIE scoring `142.8` on NetHEPT at k=50; the published value in that paper's Table 3 is **724.67**. A second scrambled IMINFECTOR's Table 3 columns, producing a false conclusion that took a primary-source check to undo. Extract text (`pdftotext -layout`), read the table, then transcribe. Never let a summary supply a number.

### Edge-count convention

Undirected graphs are quoted as **undirected edges**; directed graphs as **arcs**. Our loaders report `adjacency.nnz`, which for a symmetrized undirected graph is **2× the undirected edge count**. This one convention difference explains most apparent "discrepancies" between our numbers and published tables — check it before concluding two graphs differ.

---

## Cross-file conventions

- **Datasets are catalogued in the file of the task that uses them.** A graph used by several tasks appears in each, but the _authoritative_ row — full metadata, version forensics, download notes — lives in the file where it is most load-bearing, and the others link to it. Most social graphs are authoritative in [`influence_maximization.md`](influence_maximization.md) §6.
- **yes marks a dataset we already load** via `data/datasets/<name>.py`. The loader contract and the three-step procedure for adding one are in [`influence_maximization.md`](influence_maximization.md) §6.6.
- **(key) marks the single most directly comparable published result** for a task — same graphs, same budget convention, same metric as our setup.
- **Name collisions are the default failure mode.** Seven dataset names in the IM literature alone denote more than one graph ([`influence_maximization.md`](influence_maximization.md) §6.3), and the pattern repeats everywhere: Planetoid Cora ≠ our Cora-ML, and "Planetoid Cora" is itself quoted three ways from one file. Check the version before quoting a number.
- Every URL was checked for HTTP 200 on the date stated at the top of each file.

---

## Adding a task file

Copy the skeleton above, keep the section numbering, and run the QA check — it verifies skeleton compliance, placeholder text, tier-tag density, URL and GitHub-link counts, and (with `--links`) that every URL still resolves:

```
uv run python <scratchpad>/qa_research.py --links
```
