# Graph Task Research: index

One markdown file per graph task, all in the same format. Each is self-contained: task definition, fit with our methodology, every baseline method with paper and code URLs, every dataset those baselines evaluate on with node/edge counts and source URLs, transcribed result tables, evaluation protocols, and an honest list of what the review could **not** establish.

**6,534 lines · 8 tasks · ~330 methods · ~300 datasets · 697 unique URLs**, compiled 2026-07-28/29, narrowed to the shipped set 2026-08-06.

Five further tasks were screened and reviewed and are **not being pursued**: cascading failure, graph completion, influence estimation, network inference and temporal forecasting. Their files were deleted rather than left as dead weight; `git log` is the record if any of that analysis is ever wanted back. What each one concluded is summarized under "Screened out" below, because two of those conclusions changed how the shipped tasks are built.

These files are **literature references**, not design documents. What we built lives in `../CLAUDE.md` and the per-package READMEs; what the field published lives here.

**The filenames are load-bearing.** Each one is a key of `pipeline/tasks.py::tasks` and the first level of the results tree, so `research/influence_maximization.md` and `results/influence_maximization/<dataset>/<run>/` always name the same task. `--task` validates against that registry, and a task that is not runnable fails up front with its blocker and a pointer back to its file here.

---

## The tasks

Screened for compatibility with our methodology: an action-conditioned world model `f_θ(G, s_t, a_t) → s_{t+1}` factorized as `s_{t+1} = T_endo(T_exo(s_t, a_t))`, five action ops (`add_node`, `remove_node`, `add_edge`, `remove_edge`, `set_edge_weight`), transitions harvested from NDlib IC/LT, and a coding-agent outer loop that plans against the learned model.

A task earns a slot if it has **(a)** a node- or edge-level state that evolves, **(b)** an intervention expressible in the five ops, and **(c)** a simulator we can harvest transitions from.

| File                                                       | Fit                     | Lines | Comparable published table on a graph we load?                          |
| ---------------------------------------------------------- | ----------------------- | ----- | ----------------------------------------------------------------------- |
| [`influence_maximization.md`](influence_maximization.md)   | yes **implemented**      | 822   | (key) DeepIM Tables 2/3: Jazz, Cora-ML, Power Grid byte-identical         |
| [`influence_blocking.md`](influence_blocking.md)           | yes **implemented**      | 784   | (key) SandIMIN PVLDB'24 Table 5: email-Eu-core, YouTube, our exact metric |
| [`source_localization.md`](source_localization.md)         | yes **implemented**      | 977   | (key) SL-VAE KDD'22 Tables 1-4-5 of its 7 graphs are ours                |
| [`critical_node_detection.md`](critical_node_detection.md) | yes **implemented**      | 1,018 | (key) CoreHD/BPD Table I: "Grid" is our `power_grid` byte-for-byte        |
| [`cascade_reconstruction.md`](cascade_reconstruction.md)   | yes **implemented**      | 776   | (key) DITTO KDD'23 Tables 4-5: BA, ER, Oregon2 and rt-pol are all ours    |
| [`epidemic_control.md`](epidemic_control.md)               | yes **implemented**      | 760   | (key) GreedyWalk SDM'15 Table 2: four published `λ₁` reproduce exactly    |
| [`adaptive_online_im.md`](adaptive_online_im.md)           | yes **implemented**      | 642   | only DeepIM's re-run OIM row; the rest are figures                      |
| [`cascade_prediction.md`](cascade_prediction.md)           | yes **implemented**      | 582   | (key) CasFT AAAI-25 Table 2 + CasTemp's leak-free re-run: context only, our preprocessing differs |

Every file above is an **implemented** task. The screening criteria are kept because they are what a
ninth task would have to pass.

### Screened out (files removed 2026-08-06)

Recorded here rather than deleted silently, because three of the five produced findings the shipped
tasks still depend on.

| Task | Verdict | What it left behind |
| ---- | ------- | ------------------- |
| **influence estimation** | not pursued | Its metrics are things we already print: `ens_final_count_model` vs `ens_final_count_true` IS an influence-estimation result. If a reviewer asks for one, relabel rather than build. |
| **network inference** | not pursued | The folder's only case where the AGENT half fits and the world-model half does not: `G` is the variable being recovered, not a condition, and NETRATE's likelihood is already convex and closed-form. That asymmetry is why criterion 2 exists. |
| **cascading failure** | not pursued | `T_endo` is global load redistribution, not local edge-wise propagation, so a k-hop encoder structurally cannot see the next failure. The clearest statement in the folder of what our architecture *cannot* represent. |
| **graph completion** | not pursued | No state evolves and `add_edge` means *infer* rather than *intervene*. Useful only as a robustness condition. |
| **temporal forecasting** | not pursued | **Cross-cutting finding 4 below came from this file**: rollout degradation is a separate axis from one-step accuracy, and scheduled sampling is the standard fix. That finding outlived the task. |

---

## Cross-cutting findings

Things that emerged from more than one file, or that change what we should do.

1. **`remove_node` did not mean immunization. yes FIXED.** Found independently by `critical_node_detection.md` §2.3 and `influence_blocking.md` §2. Under IC it set NDlib status `2`, which `active_nodes()` still counted as infected, so pre-emptively immunizing `k` nodes inflated measured spread by `+k` before any diffusion ran; under LT it reset to Susceptible and the node re-activated next step; neither op touched `G`. Correct for IM ("this spreader is spent"), wrong for every containment task. Resolved by `--remove-semantics {spent,blocked}` (`data/wm_simulator.py`): `blocked` deletes the node from the graph, stops counting it, and prevents transmission and re-infection under both dynamics, with deletion expressed as `remove_node(v)` plus a `remove_edge` per incident arc so nothing downstream changed. The value crosses data generation, the head's `T_exo`, and the agent's prompt, and `pipeline/tasks.py` now carries the right one per task. `spent` output is byte-identical to before, so existing checkpoints are unaffected.

2. **NDlib ships no competitive/multi-cascade model.** Verified by enumerating `available_statuses` across every model: `Blocked: -1` is a static non-adopter set, and `CompositeModel` can only express one global tie-break. Influence blocking needs a competitive step written on top of `data/wm_simulator.py`.

3. **Adaptive IM cannot be sold on spread. yes IMPLEMENTED on that basis.** The adaptivity gap is bounded (myopic ∈ [e/(e−1), 4]; full-adoption ≤ ⌈n^{1/3}⌉) and non-adaptive greedy is provably no worse than adaptive greedy on all graphs. The claim has to be **cost**: the MC arm scales with rounds, a forward pass does not. Built accordingly: every `adaptive_*@E` arm is paired with `evolve_*@E` at matched `k`, the gap is computed and printed with that calibration stated above the table, and `evaluator_seconds` carries the actual claim.

4. **Our rollout evaluation is blind to per-step drift.** From the temporal-forecasting review (screened out, but this is what it left behind): DCRNN shows rollout degradation is a separate axis from one-step accuracy. We early-stop on one-step `delta_f1` and report aggregate `ens_count_bias`; neither sees a model that is sharp at `t+1` and drifting by `t+5`. Scheduled sampling is the field's standard fix and needs **zero extra simulator calls**, unlike the DAgger attempt that failed here.

5. **The DeepIM NetScience anomaly is narrowed, not solved.** Its 1,565/13,532 graph is shared with SL-VAE and _only_ SL-VAE; IVGD, GraphSL and SIDSL all use our 1,589/2,742. A shared preprocessing lineage between two papers, and the minority version in its own literature.

6. **Published tables are less trustworthy than they look.** BasicTS+ found the same method on the same data varying **33%** across papers, from normalisation and metric implementation alone. Cascade prediction's per-paper cascade filters move MSLE more than the gap between adjacent methods. Several canonical papers (Rozenshtein KDD'16, Xiao 2018, Farajtabar AISTATS'15, NETRATE) publish **no result tables at all**: any number attributed to them is fabricated.

7. **No published work is action-conditioned.** Found in two of the screened-out files and true across the shipped ones: every influence estimator answers `σ(S)` on a FIXED graph, and every cascading-failure GNN is one-shot while every planner plans against the exact simulator. Ours answers `σ(s_t, a_t)`, spread after an intervention. That is the contribution _and_ the problem: there is no step-wise baseline to compare against, which is why conditions 3-6 are an internal ablation rather than a leaderboard.

8. **Supplying a better forward model is not a contribution; supplying a better inversion is. yes IMPLEMENTED on that basis.** From `source_localization.md` §2.2: SL-VAE states outright that its forward operator is pluggable and reports no significant difference across GAT, MONSTOR and DeepIS. Every inverse task in this folder invites the same trap, because each one's strongest methods are built by learning a forward model and inverting it, and we have a forward model. Dropping ours into that slot reproduces a paper its authors already wrote. The escape is to contribute the `argmax` instead: `source_localization.md` §2.3 reformulates the task as **amortized program search**, where the coding agent searches the space of inversion algorithms and the world model is the forward oracle they call. That reframing is also what restores the six-condition table, since a task with no agent has no arms 3-6. Built exactly that way: the agent writes `localize(graph, observation, budget)`, the harness's forward oracle is the one new primitive and its four bindings ARE conditions 3-6 (the program is offline and never calls it; rule of 2026-09-04), and the component-swap framing ships as **arm A** (`gradient_free@world_model`, condition 8) so `6 vs A` is a measured comparison rather than an argument. Labels select the program and are never available to it, so the selected artifact runs on cascades with no ground truth at all.

   **Four criteria decide whether the reframing applies**, and they split the inverse tasks cleanly. (1) An inference algorithm with real design freedom, (2) the world model callable as a forward oracle with `G` still a *condition*, (3) a dense reward with ground truth we own, (4) no cheaper exact alternative, because a learned approximation replacing a closed form is a regression.

   | Task | 1 | 2 | 3 | 4 | Verdict |
   | ---- | - | - | - | - | ------- |
   | `cascade_reconstruction` §2.4 | yes | yes | yes | yes | **best fit in the folder**; inner loop runs at ~10⁴ oracle calls per instance |
   | `source_localization` §2.3 | yes | yes | yes | yes | both loops, ~10² calls per instance |
   | `cascade_prediction` §2.1 | Warning: | yes | yes | yes | **the criteria say no and it shipped anyway**, see below |

   **Cascade prediction is the deliberate exception, and its own file says so.** Criterion 1 fails outright: §2.1 records that `a_t` is NULL at every step, so the action space goes idle, five of six arms have nothing to distinguish them, and §9.3 concludes it "cannot demonstrate the capability the project is about". It ships anyway for the reason §9.1 gives, which no other file in this folder can claim: **it is the only falsification test available for the IC/LT assumption every other task inherits.** Everywhere else, a learned model is evaluated against traces from the simulator that trained it, a closed loop that measures learning error and never modelling error. Real Weibo retweets break the loop, and `--compare` reports the modelling error directly. Two things follow that are worth carrying: criterion 1 turns out to be about whether the ALGORITHM search is interesting, not whether the task is, and conditions 3-6 survive an empty action space because what varies down the ladder becomes the forward model the predictor may CALL rather than the intervention it may choose.

   Two consequences worth carrying. **The criteria can fail on either side**: the screened-out network-inference review is the folder's only case where the AGENT half fits and the world model does not (`G` is the variable being recovered rather than a condition), where everywhere else that fails, fails on the agent side. And **the reward is the specification**: `cascade_reconstruction.md` §2.6 shows that scoring a program search on the easy half of a metric pair (node set rather than tree) does not merely under-report, it makes the search discard the capability. Check that a trivial baseline scores badly under the chosen reward before running any search.

   **Both inverse tasks are now built on that basis, and the second one is the sharper test.** Cascade reconstruction ships as `--task cascade_reconstruction`: the agent writes `reconstruct(graph, observation, horizon)`, the harness's transition kernel is the one new primitive and its four bindings ARE conditions 3-6 (the decoder is offline and never calls it; rule of 2026-09-04), and the component-swap framing ships as **arm A** (`decode_free@world_model`, condition 8), DITTO's Metropolis-Hastings sampler with our learned kernel in place of its mean-field `β̂`, so `6 vs A` is measured rather than argued. Three differences from source localization are worth naming. The inner loop runs at ~10⁴ kernel calls per instance against ~10², which is what makes the cost axis load-bearing rather than a footnote. The reward is `λ·PathPrecision + (1−λ)·EventF1` rather than a single metric, and `trivial_decoder_reward` is computed into every results JSON because the check above is a requirement here, not advice. And it needed one thing source localization did not: a **ground-truth parent**, which NDlib does not emit at all, `--trace-parents` swaps in traced IC/LT models that record the transmission edge, and without it the tree-weighted reward is not computable and the harness refuses to run.

---

## Evaluation matrix

What each task is evaluated on, under which dynamics, against which baselines. Assembled 2026-08-09 from `pipeline/tasks.py`, `baselines/registry.py` and `sbatch/<task>/`, not from prose. Per-task detail and the reasons behind each choice are in each file's **§6.0**.

**Read the datasets column as a choice, not a constraint.** All 71 real graph loaders and all 7 synthetic families (`er`, `ba`, `ws`, `sbm`, `powerlaw_cluster`, `kronecker`, `karate`) run on any of the seven simulator tasks; nothing in the registry ties a task to a dataset. The one exception is `cascade_prediction`, which is `observational` and accepts only the 8 replay corpora. Where "in the sweep" and "benchmark family" disagree, the sweep is the accident.

| Task | Benchmark family (from §6) | In the sweep today | Dynamics | Classical baselines (ours, condition 1) | Learned repos wired (condition 7) |
| --- | --- | --- | --- | --- | --- |
| **influence_maximization** | `jazz`, `email_eu_core`, `netscience`, `cora_ml`, `facebook`, `power_grid`, `ca_grqc`, `wiki_vote`, `lastfm_asia`, `nethept`, `netphy` · synthetic `er`, `ba`, `ws`, `sbm`, `karate`, `powerlaw_cluster`, `kronecker` · scale `twitter`, `digg`, `youtube`, `epinions`, `orkut`, `livejournal`, `weibo` | `ba`, `sbm`, `jazz`, `netscience`, `power_grid`, `nethept` | IC, LT | `high_degree`, `degree_discount`, `pagerank_seeds`, `celf_pp`, `imm`, `random_seeds` *(6 of 36)* | **learned:** `deepim`, `touplegdd`, `glie`, `moeim` · **classical:** `opim`, `ssa`, `subsim` |
| **adaptive_online_im** | the static IM suite re-used: `jazz`, `cora_ml`, `facebook`, `power_grid`, `nethept`, `netphy`, `digg`, `netscience` · scale `twitter`, `youtube`, `livejournal` | `powerlaw_cluster`, `nethept` | IC, LT | `adapt_greedy`, `adapt_epic`, `adapt_degree_discount`, `adapt_random`, `static_split` + matched controls `celf_pp`, `imm` *(7 policies)* | **learned:** `rl4im` · **classical:** `adaptiveim` |
| **critical_node_detection** | the 12 dismantling benchmarks: `corruption`, `usair97`, `crime`, `road_eu`, `euroroad`, `hamsterster`, `ppi_yeast`, `openflights`, `human_ppi_vidal`, `intnet1`, `pgp`, `p2p_gnutella` · plus `power_grid` | `crime`, `corruption`, `ppi_yeast`, `hamsterster`, `intnet1`, `usair97`, `openflights`, `road_eu`, `power_grid` | IC, LT | `adaptive_degree`, `iterative_betweenness`, `collective_influence_r`, `corehd`, `bpd_r`, `decycling`, `explosive_immunization`, `gnd`, `netshield`, `frontier_removal` (the outbreak-aware control, see that file's §8.2 trap 10), `degree_removal`, `random_removal` *(12 of 25)* | **learned:** `finder`, `gdm`, `mind`, `nirm`, `dcrs`, `selinda` · **classical:** `gnd`, `decycler`, `collective_influence`, `explosive_immunization`, `dismantling_review` |
| **source_localization** | `karate`, `dolphins`, `jazz`, `netscience`, `cora_ml`, `power_grid` · cost target `deezer` | `karate`, `dolphins`, `jazz`, `netscience`, `cora_ml`, `power_grid` | IC, LT | `lpsi`, `netsleuth`, `ojc`, `jordan_center`, `dmp_localize`, `dynamic_age`, `effective_distance`, `infected_degree`, `rumor_centrality`, `random_sources` *(10 of 14)* | **learned:** `graphsl_gcnsi`, `graphsl_ivgd`, `graphsl_slvae` · **classical:** `graphsl_lpsi`, `graphsl_netsleuth`, `graphsl_ojc`, `cosasi_jordan`, `cosasi_netsleuth`, `cosasi_lisn`, `cosasi_rumor_centrality` |
| **influence_blocking** | the 13 blocking benchmarks: `p2p_gnutella08`, `p2p_gnutella24`, `cit_hepth`, `cit_hepph`, `email_enron`, `slashdot`, `epinions1`, `gowalla`, `email_euall`, `web_stanford`, `dblp`, `higgs_twitter`, `pokec` | `email_eu_core`, `facebook`, `wiki_vote`, `netscience`, `nethept`, `netphy` | IC, LT | counter_seed `proximity`, `multi_hop_proximity`, `rps`, `cldag`, `forward_blocking`, `reverse_blocking`, `degree_blocking`, `pagerank_blocking`, `random_blocking` · node_block `imin_lhga`, `imin_lsbm`, `advanced_greedy`, `greedy_replace` · edge/weight `kimura_link_blocking`, `out_edge_blocking`, `edge_betweenness_blocking`, `random_edge_blocking` *(20 total)* | **learned:** none (`diffim` was wired, then deleted on 2026-09-04: its surrogate training stalled the sweep) · **classical:** `sandimin`, `imin_joc` |
| **cascade_reconstruction** | the 7 CR benchmarks: `infectious`, `email_univ`, `uci_students`, `oregon2`, `rt_pol`, `ca_hepth`, `citeseer` · plus `cora_ml`, `power_grid`, `ca_grqc`, `jazz` · synthetic `ba`, `er` at DITTO's parameters | `jazz`, `infectious`, `email_univ`, `uci_students`, `citeseer`, `cora_ml`, `ca_grqc`, `power_grid`, `ca_hepth`, `oregon2`, `rt_pol` | IC, LT | `delayed_bfs`, `personalized_pagerank`, `ordered_steiner_closure`, `greedy_ordered`, `tree_sampling`, `cult`, `consistent_tree_wpct`, `dhrec`, `cri`, `netfill`, `observed_only`, `random_reconstruction` *(12 of 18)* | **learned:** `ditto`, `grin`, `spin`, `deep_demixing` · **classical:** `ditto_dhrec`, `ditto_cri` |
| **epidemic_control** | 12 SocioPatterns traces `primary_school`, `high_school_2011`, `high_school_2012`, `high_school_2013`, `hypertext09`, `sfhh`, `workplace_invs13`, `workplace_invs15`, `kenya_households`, `malawi_village`, `hospital_lh10`, `infectious_sociopatterns` · 7 spectral `oregon1`, `oregon2_010331`, `brightkite`, `p2p_gnutella05`, `p2p_gnutella06`, `deezer_ro`, `youtube` · control `football` | `hospital_lh10`, `primary_school`, `malawi_village`, `football`, `oregon1`, `oregon2_010331`, `brightkite`, `ca_grqc`, `deezer_ro`, `power_grid` | **SIR, SIS, SEIR** | vaccinate `degree_immunization`, `adaptive_degree_immunization`, `acquaintance_immunization`, `netshield`, `netshield_plus`, `dava`, `dava_fast`, `frontier_immunization`, `greedy_walk`, `eigenvector_immunization`, `kshell_immunization`, `random_immunization` · edge/contact `netmelt`, `product_degree`, `eigen_score`, `greedy_walk_edge`, `frontier_edge_cut`, `random_edge_cut` *(23 total)* | **learned:** `finder_epi`, `gdm_epi` · **classical:** `netimm_netshield`, `netimm_dava`, `netimm_dava_fast`, `netimm_netshape`, `netimm_degree`, `netimm_random`, `collective_influence_epi`, `explosive_immunization_epi`, `dismantling_review_epi` |
| **cascade_prediction** | the 8 replay corpora, and nothing else: `taoke`, `digg_cascades`, `casflow_aps`, `casflow_weibo`, `casflow_twitter`, `memetracker`, `aps`, `weibo_cascades` | `taoke`, `digg_cascades`, `casflow_aps`, `casflow_weibo`, `casflow_twitter`, `memetracker` | **none (real logs replayed)** | `szabo_huberman`, `feature_linear`, `feature_gbt`, `seismic`, `hawkes`, `hawkes_hybrid`, `rpp`, `hip`, `branching_factor`, `weng_communities`, `persistence`, `mean_size`, `random_prediction` *(13 of 17)* | **learned:** `casflow`, `ccgl`, `ctcp`, `cascn`, `coupledgnn` · **classical:** none wired |

### Reading the baseline columns

**"Classical" and "learned" split twice, and the columns mean different things.** Column 5 is condition 1: our own reimplementations, all of which run. Column 6 is condition 7: published repositories with an adapter, split by whether the METHOD is learned. A lot of what is wired is a published CLASSICAL implementation (OPIM, SandIMIN, the NetImm suite, GraphSL's LPSI / NETSLEUTH / OJC), which is a cross-check on our own reimplementation rather than new coverage. Repository totals: **111 registered, 55 wired, 26 of those learned.**

### Three gaps this matrix makes visible

1. **`influence_blocking`'s sweep and its benchmark family are disjoint.** The sweep runs the IM core; none of the 13 graphs that literature publishes on is in it.
2. **Two tasks have no `sweep_datasets.sbatch` at all** (`influence_maximization`, `adaptive_online_im`); their column 3 is the set of `generate_data_*.sbatch` targets, which is why `karate`, `er` and `ws` are absent despite being fully supported.
3. **`epidemic_control` runs 3 of its 12 contact traces**, and adds two graphs (`ca_grqc`, `power_grid`) that are not in its family at all.

None of the three is a code defect. All three are the per-task evaluation set never having been decided deliberately, and they should be settled before the tables are generated.

---

## File format

Every file follows the same skeleton. Keep it: the value of the folder is that a question ("who baselines on what?", "where do I download it?") is answered in the same section number regardless of which task you open.

```
# <Task>: Prior Work, Datasets, and Published Results

<2-4 sentence scope statement>

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
| **[figure]**   | Read off a plotted figure: the paper published no table. Approximate, direction only.                               |
| **[claim]**    | Stated in prose by a paper or a secondary source; not cross-checked against a file or table.                         |

**Automated PDF summarizers hallucinate plausible numbers from these papers.** While compiling the IM file, a summarizer reported IRIE scoring `142.8` on NetHEPT at k=50; the published value in that paper's Table 3 is **724.67**. A second scrambled IMINFECTOR's Table 3 columns, producing a false conclusion that took a primary-source check to undo. Extract text (`pdftotext -layout`), read the table, then transcribe. Never let a summary supply a number.

### Edge-count convention

Undirected graphs are quoted as **undirected edges**; directed graphs as **arcs**. Our loaders report `adjacency.nnz`, which for a symmetrized undirected graph is **2× the undirected edge count**. This one convention difference explains most apparent "discrepancies" between our numbers and published tables: check it before concluding two graphs differ.

---

## Cross-file conventions

- **Datasets are catalogued in the file of the task that uses them.** A graph used by several tasks appears in each, but the _authoritative_ row (full metadata, version forensics, download notes) lives in the file where it is most load-bearing, and the others link to it. Most social graphs are authoritative in [`influence_maximization.md`](influence_maximization.md) §6.
- **yes marks a dataset we already load** via `data/datasets/<name>.py`. The loader contract and the three-step procedure for adding one are in [`influence_maximization.md`](influence_maximization.md) §6.6.
- **(key) marks the single most directly comparable published result** for a task: same graphs, same budget convention, same metric as our setup.
- **Name collisions are the default failure mode.** Seven dataset names in the IM literature alone denote more than one graph ([`influence_maximization.md`](influence_maximization.md) §6.3), and the pattern repeats everywhere: Planetoid Cora ≠ our Cora-ML, and "Planetoid Cora" is itself quoted three ways from one file. Check the version before quoting a number.
- Every URL was checked for HTTP 200 on the date stated at the top of each file.

---

## Adding a task file

Copy the skeleton above, keep the section numbering, and run the QA check: it verifies skeleton compliance, placeholder text, tier-tag density, URL and GitHub-link counts, and (with `--links`) that every URL still resolves:

```
uv run python <scratchpad>/qa_research.py --links
```
