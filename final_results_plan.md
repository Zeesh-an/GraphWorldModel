# Final Results Plan

The experiment matrix for the paper's main results, revised 2026-09-04 after the first draft was found to be capped at small graphs. It names, per graph task, the five datasets we evaluate on (three critical, two secondary; no graph under 1,000 nodes, and every synthetic row at one size, 10,000 nodes), the five baselines chosen from what the August sweep actually scored, the three algorithm-discovery systems, and the tables each claim is read from. Every value here is one the pipeline accepts today: every dataset is a `--dataset` loader or synthetic family, every baseline a `--baselines` name or a wired `external:<name>` entry in `baselines/registry.py`, every discovery system a wired `discovery:<name>` entry. `test_commands.md` holds the submission commands; `setup_guide.md` the environment.

## 1. The five claims and where each is read

| Claim | Question | Evidence | Table |
| --- | --- | --- | --- |
| C1 Dynamics learning | Does the GWM reconstruct unknown graph dynamics? | One-step `delta_f1` and `brier` against MC marginals; free-running rollout `ens_count_bias`, `ens_marg_mae`, final count model vs true; `action_sensitivity` and `add_seed_success` for action conditioning; `wm_minus_mc` on the exact winning strategy; the `structured_oracle` ceiling beside it | T1 |
| C2 Efficiency | How much faster are imagined rollouts than MC or oracle simulation, including the amortized training cost? | `rollout_seconds` vs `mc_rollout_seconds` per rollout sample; `evaluator_seconds` spent by conditions 4 and 6 in the same search; `train_seconds` plus data-generation seconds, converted to a break-even rollout count. The scale rows are where this claim is largest and where `@monte_carlo` may not finish at all | T2 |
| C3 Downstream utility | Does planning with the GWM improve graph-task performance? | The ladder replayed on the shared oracle referee: conditions 3, 4, 5, 6 with the method fixed and only the evaluator varying, beside the five baselines; `planning_regret_multi` vs degree and random | T3 (main), T1 |
| C4 Algorithm discovery | Does GWM access let the coding agent find better algorithms under the same interaction budget? | Condition 6 against condition 9 (three published discovery systems on the plain simulator) at reported `evaluator_calls`, `real_env_episodes` and `evaluator_seconds`; the write-up's trajectory, closest-classical and novelty sections as case studies | T4, case studies |
| C5 Generalization | Does this hold across datasets, sizes, topologies and dynamics? | T3 repeated over the five datasets per task (medium, large and very large tiers, real and synthetic), IC and LT for every simulator task, SIR/SIS/SEIR for epidemic control, chronological vs random split for cascade prediction, one world model per dataset reused across tasks (`scripts/transfer_matrix.py`), program transfer for source localization (`--sl-transfer-from`) | T5 |

T1 and T2 come free with every run (the `## World Model` section of each report and the timing rows), so they are populated on every dataset in the matrix. T3 is the main table. T4 runs on the critical datasets only. T5 is T3 sliced by axis.

## 2. Selection rules

**Size tiers.** Small is under 1,000 nodes, medium 1,000 to 10,000, large 10,000 to 100,000, very large above 100,000. The first draft of this file stopped at large because the project brief warns that the seven biggest graphs exceed what the NDlib rollout path simulates in reasonable time. That warning is about the Monte Carlo evaluator, and the Monte Carlo evaluator being too slow is exactly claim C2, so the scale rows are in the matrix on purpose. Small graphs are not in the matrix at all: every row is at least 1,000 nodes, because a small graph shows no evaluator speedup and duplicates what the medium rows already measure; the small loaders stay in the catalogue below for the record. Nothing above 15,229 nodes (`nethept`) has been generated or trained on yet; every large and very-large row therefore starts with a timed data-generation job before anything else is submitted on it.

**Datasets.** Five per task, ranked. The three critical datasets are the ones the paper cannot ship without, and on their own they span at least two size tiers and two structures; the third critical row of every simulator task is a large or very-large graph. Real graphs fill the critical set wherever the task's literature reports on them, because those rows are comparable to published numbers. Every simulator task carries exactly one synthetic row, at one size, 10,000 nodes, with no size sweep: 10,000 is the largest size at which the whole ladder including the Monte Carlo arm and a discovery system finishes in under a day per dynamics, it puts the synthetic row in the large tier with a structure no real row has (planted communities for `sbm`, a clean power law with tunable clustering for `ba` and `powerlaw_cluster`), and the very-large real rows already carry the 100,000-class tier. At least one community-structured graph (`sbm` or a labelled real graph) is in every task, because on a degree-trivial graph every backbone ties the degree heuristic. Cascade prediction is observational and has no synthetic option; its corpora are already the largest objects in the repo.

**Cost controls for the scale rows.** `--mc-runs` and `--n-samples` down from 200, the horizon capped, `--cp-max-nodes` for the two CasFlow corpora, and the evaluator budget of every arm recorded so a row that did not finish is reported as such rather than dropped silently. The measures are stated in the table caption.

**Baselines.** Five per task, chosen from the arms the August sweep scored on its Monte Carlo referee (the per-task survey tables below give the mean referee score across budgets, from `results/` and `results_archive/`), plus the wired learned or state-of-the-art repos that answer the same question. The external repos ran in the same sweep and their rows live under `baselines/<budget>/external_*.json` rather than `agent/`; thirteen scored, and about forty were skipped for reasons listed per task below and collected in section 7. A control that is not a method (the outbreak ring, the reports-only decoder, the two prediction floors) is always run and printed above the table but does not count toward the five. `celf_pp` is removed from influence maximization entirely.

**Discovery systems.** One common set of three for every task, chosen for three different search paradigms: `llm4ad_funsearch` (FunSearch, the origin: an island program database with best-shot prompting), `eoh` (Evolution of Heuristics, ICML 2024 oral: co-evolution of a natural-language thought and its code), and `openevolve` (the open AlphaEvolve implementation: a MAP-Elites program database with cascade evaluation). All three run their own loop at published defaults on the plain simulator through `baselines/score_program.py`; the world model is unreachable by construction. Alternates: `reevo`, `mcts_ahd`, `llamea`.

**Conditions per table row.** Every T3 cell has the five baselines (condition 1 or 7), the routed classical pool (condition 2), and the four-arm ladder `evolve_free@native`, `@monte_carlo`, `@oracle`, `@world_model` (conditions 3 to 6), all replayed on the shared referee: the exact oracle simulator at 1,000 samples (`--referee oracle --referee-samples 1000`, the defaults), with `--mc-agreement` adding the NDlib replay on the critical datasets only. Adaptive IM uses `adaptive_<mode>@<evaluator>` paired with the matching `evolve` arm so the adaptivity gap is computable.

**Budgets and dynamics** follow `pipeline/tasks.py`: 1, 5, 10 and 20 percent of $N$ for the two IM tasks and critical node detection (outbreak 10 percent of $N$), absolute $k \in \{10, 20, 30, 40, 50\}$ for influence blocking (rumour 1 percent of $N$), a single 10 percent point for the two inverse tasks and cascade prediction, and the epidemic task's percent ladder against a 1 percent outbreak. IC and LT for every simulator task, SIR, SIS and SEIR for epidemic control, with the SIR $\gamma = 1$ check against IC reported once.

## 3. Dataset catalogue

Every loader in `data/datasets/` (79 datasets; the four helper modules `cascade_common`, `casflow_bundle`, `dismantling_common`, `sociopatterns` are not datasets) plus the seven synthetic families, as loaded. Undirected rows quote edges, directed rows quote arcs. Any simulator task can run on any graph; cascade prediction accepts only the corpora marked CP. The small tier is listed for completeness and is not in the matrix.

**Small (under 1,000 nodes)**

| Dataset | Nodes | Edges | Kind | Task family |
| --- | --- | --- | --- | --- |
| `dolphins` | 62 | 159 | undirected social | source localization |
| `hospital_lh10` | 75 | 1,139 | contact trace, dense ward | epidemic control |
| `kenya_households` | 75 | 576 | contact trace, 5 household labels | epidemic control |
| `malawi_village` | 86 | 347 | contact trace | epidemic control |
| `workplace_invs13` | 92 | 755 | contact trace | epidemic control |
| `hypertext09` | 113 | 2,196 | contact trace | epidemic control |
| `football` | 115 | 613 | undirected, 12 conference labels | epidemic control (control graph) |
| `high_school_2011` | 126 | 1,709 | contact trace | epidemic control |
| `high_school_2012` | 180 | 2,220 | contact trace | epidemic control |
| `jazz` | 198 | 2,742 | undirected collaborations, dense | IM, SL, CR |
| `workplace_invs15` | 217 | 4,274 | contact trace | epidemic control |
| `primary_school` | 242 | 8,317 | contact trace, 11 class labels | epidemic control |
| `corruption` | 309 | 3,281 | undirected co-scandal, GCC | dismantling |
| `high_school_2013` | 327 | 5,818 | contact trace | epidemic control |
| `usair97` | 332 | 2,126 | undirected air routes, hub-dominated | dismantling |
| `sfhh` | 403 | 9,565 | contact trace | epidemic control |
| `infectious` | 410 | 2,765 | contact trace (Xiao's extraction) | cascade reconstruction |
| `crime` | 754 | 2,127 | undirected co-offending, GCC | dismantling |

**Medium (1,000 to 10,000 nodes)**

| Dataset | Nodes | Edges | Kind | Task family |
| --- | --- | --- | --- | --- |
| `email_eu_core` | 1,005 | 24,929 arcs | directed emails, 42 department labels | IM, IB |
| `road_eu` | 1,039 | 1,305 | undirected E-roads, GCC | dismantling |
| `email_univ` | 1,133 | 5,451 | undirected email | cascade reconstruction |
| `euroroad` | 1,174 | 1,417 | undirected E-roads, raw | dismantling |
| `uci_students` | 1,266 | 6,451 | undirected messaging | cascade reconstruction |
| `netscience` | 1,589 | 2,742 | undirected coauthorship, sparse | IM, SL |
| `citeseer` | 1,681 | 2,902 | undirected citations, LCC under standardize | cascade reconstruction |
| `hamsterster` | 2,000 | 16,098 | undirected pet social, GCC | dismantling |
| `ppi_yeast` | 2,224 | 6,609 | undirected protein interactions, GCC | dismantling |
| `cora_ml` | 2,810 | 7,981 | undirected citations, standardized, 2,879-dim features | IM, SL |
| `openflights` | 2,939 | 15,677 | undirected flight routes | dismantling |
| `human_ppi_vidal` | 3,023 | 6,149 | undirected human interactome | dismantling |
| `facebook` | 4,039 | 88,234 | undirected friendships, very dense | IM |
| `power_grid` | 4,941 | 6,594 | undirected power lines, mesh | IM, CND, SL, CR |
| `memetracker` | 5,000 hosts (published; ours differs) | 313,669 (published) | 54,847 cascades on hosts | CP only |
| `ca_grqc` | 5,242 | 14,484 | undirected coauthorship, assortative | IM, CR |
| `p2p_gnutella08` | 6,301 | 20,777 arcs | directed P2P, Aug 2002 | influence blocking |
| `intnet1` | 6,474 | 12,572 | undirected AS peering, GCC | dismantling |
| `wiki_vote` | 7,115 | 103,689 arcs | directed votes | IM |
| `lastfm_asia` | 7,624 | 27,806 | undirected, 18 country labels | IM |
| `ca_hepth` | 8,638 | 24,806 | undirected coauthorship | cascade reconstruction |
| `p2p_gnutella06` | 8,717 | 31,525 arcs | directed P2P | epidemic control (spectral) |
| `p2p_gnutella05` | 8,846 | 31,839 arcs | directed P2P | epidemic control (spectral) |

**Large (10,000 to 100,000 nodes)**

| Dataset | Nodes | Edges | Kind | Task family |
| --- | --- | --- | --- | --- |
| `oregon1` | 10,670 | 22,002 | undirected AS graph, $\lambda_1 = 58.72$ | epidemic control (spectral) |
| `pgp` | 10,680 | 24,316 | undirected web of trust | dismantling |
| `oregon2_010331` | 10,900 | 31,180 | undirected AS graph, 2001-03-31 | epidemic control (spectral) |
| `infectious_sociopatterns` | 10,972 | 44,517 | full contact trace | epidemic control |
| `oregon2` | 11,461 | 32,730 | undirected AS graph, 2001-05-26 | cascade reconstruction |
| `nethept` | 15,229 | 62,752 arcs | directed coauthorship, both arcs | IM, AOIM |
| `rt_pol` | 18,470 | 48,053 | undirected retweets | cascade reconstruction |
| `p2p_gnutella24` | 26,518 | 65,369 arcs | directed P2P | influence blocking |
| `cit_hepth` | 27,769 | 352,768 arcs | directed citations, dense | influence blocking |
| `taoke` | 29,711 | (cascade log) | 2,862 cascades | CP only |
| `cit_hepph` | 34,546 | 421,534 arcs | directed citations, dense | influence blocking |
| `email_enron` | 36,692 | 183,831 | undirected email | influence blocking |
| `netphy` | 37,154 | 174,161 | undirected coauthorship | IM |
| `deezer_ro` | 41,773 | 125,826 | undirected friendships (RO) | epidemic control (RLGN) |
| `deezer` | 47,538 | 222,887 | undirected friendships (HU) | source localization (cost) |
| `brightkite` | 58,228 | 214,078 | undirected, $\lambda_1 = 101.49$ | epidemic control (spectral) |
| `p2p_gnutella` | 62,561 | 147,878 | undirected P2P, Gnutella31 GCC | dismantling |
| `epinions1` | 75,879 | 508,837 arcs | directed trust | influence blocking |
| `twitter` | 81,306 | 1,768,149 arcs | directed follows | IM, AOIM |
| `slashdot` | 82,168 | 870,161 arcs | directed, Feb 2009 | influence blocking |

**Very large (above 100,000 nodes)**

| Dataset | Nodes | Edges | Kind | Task family |
| --- | --- | --- | --- | --- |
| `digg` | 116,893 | about 2.6M | undirected friendships | IM (scale) |
| `epinions` | 131,828 | 841,372 arcs | directed signed trust, sign dropped | IM (scale) |
| `gowalla` | 196,591 | 950,327 | undirected location friendships | influence blocking |
| `email_euall` | 265,009 | 418,956 arcs | directed email | influence blocking |
| `digg_cascades` | 279,630 | 1,731,653 arcs | 3,553 vote cascades on the ISI graph | CP only |
| `web_stanford` | 281,903 | 2,312,497 arcs | directed hyperlinks | influence blocking |
| `dblp` | 317,080 | 1,049,866 | undirected coauthorship | influence blocking |
| `higgs_twitter` | 456,631 | 14,855,875 arcs | directed followers | influence blocking (scale) |
| `casflow_twitter` | 490,474 | 1,903,230 | 88,440 hashtag cascades | CP only |
| `aps`, `casflow_aps` | 616,316 | 3,304,400 | 90,768 (ours) or 207,685 (CasFlow filter) citation cascades | CP only |
| `youtube` | 1,134,890 | 2,987,624 | undirected, $\lambda_1 = 210.4$ | IM, epidemic control (scale) |
| `pokec` | 1,632,803 | 30,622,564 arcs | directed friendships | influence blocking (scale) |
| `weibo` | 1,787,443 | about 216M arcs | directed influence graph, manual download | IM (scale) |
| `orkut` | 3,072,441 | 117,185,083 | undirected friendships | IM (scale) |
| `livejournal` | 4,847,571 | 68,993,773 arcs | directed friendships | AOIM (scale) |
| `casflow_weibo` | 6,738,040 | 15,249,636 | 119,313 retweet cascades | CP only |

**Synthetic families** (size by `--syn-nodes`, bulk by `--num-graphs`): `er`, `ba` ($m$ by flag), `ws`, `sbm` (`--sbm-blocks`, `--sbm-p-in`, `--sbm-p-out`), `powerlaw_cluster` (`--plc-m`, `--plc-p`, RL4IM's family), `kronecker` (`--kron-variant`, ConTinEst's seeds), `karate` (34 nodes, bulk).

## 4. Per-task matrix

Rank 1 to 3 is critical, 4 and 5 secondary. The survey table under each task is the August sweep on its Monte Carlo referee: mean referee reward across the budgets that ran, from one dataset each, so it ranks the pool rather than measuring it. External repos are listed after the classical pool with the reason any of them was skipped; most skips are one of two cluster-side problems, the relative `work_dir` bug fixed on 2026-08-23 (the repo's interpreter could not open the driver script, `exited 2`) and venvs whose graph libraries never installed (`ModuleNotFoundError`, `exited 1`).

### 4.1 Influence maximization (`influence_maximization`, IC and LT; higher is better)

| Rank | Dataset | Nodes | Edges | Tier | Kind | Why |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `netscience` | 1,589 | 2,742 | medium | real | the ladder already runs here; IM and adaptive IM results exist |
| 2 | `nethept` | 15,229 | 62,752 arcs | large | real | the canonical IM benchmark of Chen et al., IMM and OPIM; a generation script exists |
| 3 | `digg` | 116,893 | about 2.6M | very large | real | the scale row: the RIS baselines (IMM, OPIM, SubSIM) are built for exactly this size, and it is where MC evaluation becomes the bottleneck |
| 4 | `cora_ml` | 2,810 | 7,981 | medium | real | matches DeepIM's graph exactly, so the learned row is comparable |
| 5 | `sbm` | 10,000 | 10 blocks of 1,000, by `--sbm-p-in` and `--sbm-p-out` | large | synthetic | planted communities, where degree heuristics fail; a large-tier structure no real row has |

Survey (netscience, IC and LT runs pooled): `imm` 556, `voterank` 532, `degree_discount` 506, `pagerank_seeds` 461, `random_seeds` 327, `high_degree` 324; ladder `@world_model` 538, `@monte_carlo` 614 (LT only), `@oracle` 456 (includes small smoke runs). External: `ssa` 556, `subsim` 550, `opim` 549, `deepim` 397 (8,588 s of its own training per budget), `moeim` 367 (2,144 s per budget, and it emitted a duplicate seed at 20 percent); `glie` ran but its seed output could not be parsed, and `touplegdd` had no venv.

Baselines: `imm` (external, SIGMOD 2015), `opim` (external, SIGMOD 2018), `subsim` (external, SIGMOD 2020), `deepim` (external, learned, ICML 2023), `degree_discount` (ours, the strongest heuristic in the survey). `celf_pp` is removed. Alternates: `voterank` (second in the survey), `glie`, `touplegdd`. Stretch scale rows if `digg` finishes: `youtube`, `twitter`.

### 4.2 Adaptive online IM (`adaptive_online_im`, IC and LT; higher is better)

| Rank | Dataset | Nodes | Edges | Tier | Kind | Why |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `netscience` | 1,589 | 2,742 | medium | real | results exist; the same graph as the static IM row, so the adaptivity gap reads against a known static number |
| 2 | `nethept` | 15,229 | 62,752 arcs | large | real | Han et al.'s benchmark; a generation script exists |
| 3 | `twitter` | 81,306 | 1,768,149 arcs | large | real | Han et al.'s own scale benchmark, where per-round MC re-estimation is the cost the task exists to show |
| 4 | `powerlaw_cluster` | 10,000 | `--plc-p 0.05` (RL4IM's value) | large | synthetic | RL4IM's family; results and a generation script exist at a smaller size |
| 5 | `facebook` | 4,039 | 88,234 | medium | real | the dense regime where rounds reveal little, so the theory cap on the adaptivity gap is visible |

Survey (netscience): `imm` 518 (static), `adapt_epic` 513, `adapt_degree_discount` 453, `adapt_pagerank` 437, `static_split` 434, `adapt_degree` 344, `adapt_random` 330; `evolve_free@oracle` 527, `adaptive_free@oracle` 422. External: `adaptiveim` 468. `rl4im` was requested and skipped on every budget: it died at `import networkx`, because networkx 2.3's GraphML writer reads `np.int`, which numpy 1.24 removed; the registry now pins `numpy<1.24` in its requirements patch, so the fix is to rebuild that venv with `python -m baselines.setup_baselines --only rl4im` on the cluster and rerun.

Baselines: `adaptiveim` (external, Han et al.'s own AdaptGreedy and EPIC), `adapt_epic` (ours, the strongest adaptive row), `adapt_degree_discount` (ours), `rl4im` (external, learned, UAI 2021), `imm` (the strongest static row, kept as the non-adaptive control that turns a spread into an adaptivity gap). Alternate control: `static_split`. The bandit repos report cumulative regret and have no seed set, so they stay out of T3. Stretch: `youtube`, `livejournal` (Han et al.'s algorithm itself runs out of memory on them).

### 4.3 Critical node detection (`critical_node_detection`, IC and LT, reactive containment; lower is better)

| Rank | Dataset | Nodes | Edges | Tier | Kind | Why |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `power_grid` | 4,941 | 6,594 | medium | real | byte-identical to CoreHD, BPD and NIRM's file; results exist; the predicted loss |
| 2 | `pgp` | 10,680 | 24,316 | large | real | a dismantling benchmark with a trust-web structure unlike a grid |
| 3 | `p2p_gnutella` | 62,561 | 147,878 | large | real | the largest of the twelve dismantling benchmarks, SNAP's giant component |
| 4 | `openflights` | 2,939 | 15,677 | medium | real | hub-dominated air routes, the opposite structure to a grid, at medium size (replaces the 332-node `usair97`) |
| 5 | `ba` | 10,000 | $m = 3$, about 30,000 edges | large | synthetic | FINDER's training family, at a size FINDER itself never trains on |

Survey (power_grid, spread after removal): `bpd_r` 127.5, `collective_influence_r` 127.8, `corehd_r` 128.1, `explosive_immunization` 128.9, `decycling` 129.2, `gndr` 129.2, `corehd` 129.5, `adaptive_degree` 130.0, `degree_removal` 130.4, `betweenness_removal` 131.1, `kshell_removal` 134.9, `acquaintance_immunization` 137.2, `gnd` 138.8, `random_removal` 141.7, `netshield` 143.9; `evolve_free@oracle` 66.5 (before the ring control existed, so void). External: `explosive_immunization` 129.9, `gnd` 130.2, `collective_influence` 130.3, all within a point of our reimplementations. Every learned repo in that sweep skipped at import (`gdm` and `selinda`: `network_dismantling` missing; `mind`: `torch_scatter`; `nirm`: `torch_geometric`; `dcrs`: `dgl`), as did `dismantling_review` (`graph_tool`, which is not pip-installable); `finder` was not in that submission.

Baselines: `bpd_r` (ours, the best row), `collective_influence_r` (ours), `adaptive_degree` (ours, HDA, the row MIND's table puts within five points of deep RL), `finder` (external, learned, Nature MI 2020), `mind` (external, learned, AAAI 2026). Neither learned row has ever scored: `finder` was not in the August submission because its TF 1.14 pin needs a CPython 3.7 that uv cannot provision on the cluster (a conda, pyenv or module 3.7 unblocks it; the adapter is written), and `mind` was submitted but died at import because `torch_scatter` never built in its venv. Both are cluster setup work before the sweep; if either is not running after the first setup pass, `gdm` takes its place (its adapter was rewritten and verified locally on 2026-09-04), then `nirm` (missing only pip-installable `torch_geometric`). Alternate after that: `corehd_r`. Control printed above the table: `frontier_removal`, with `outbreak_ring` and `ring_fits` naming the trivial budgets.

### 4.4 Source localization (`source_localization`, IC and LT, label-free consistency reward; F1 reported post hoc)

| Rank | Dataset | Nodes | Edges | Tier | Kind | Why |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `cora_ml` | 2,810 | 7,981 | medium | real | the graph SL-VAE, IVGD and GCNSI report on |
| 2 | `power_grid` | 4,941 | 6,594 | medium | real | the hardest published SL graph: slow diffusion, so the observation says less about the source |
| 3 | `deezer` | 47,538 | 222,887 | large | real | IVGD's scalability column (HU, its row to the digit); a cost target and the first large SL row anyone has run |
| 4 | `netscience` | 1,589 | 2,742 | medium | real | in the SL benchmark family (the 2026 GNN benchmark); sparse with many components, the opposite of cora_ml (replaces the 198-node `jazz`, where the only results so far were run) |
| 5 | `sbm` | 10,000 | 10 blocks of 1,000, by `--sbm-p-in` and `--sbm-p-out` | large | synthetic | sources in different communities, the case where centre-of-region estimators fail |

Survey (jazz, the pre-label-free F1 reward, one budget): `infected_betweenness` 0.652, `infected_degree` 0.647, `infected_closeness` 0.630, `infected_eigenvector` 0.605, `dynamic_age` 0.565, `jordan_center` 0.428, `rumor_centrality` 0.415, `ojc` 0.408, `dmp_localize` 0.362, `netsleuth` 0.336, `effective_distance` 0.284, `lpsi` 0.282, `random_sources` 0.167; `evolve_free@oracle` 0.659. External: all six GraphSL rows and all four cosasi rows skipped by the `work_dir` bug, so none has a score yet.

Baselines: `infected_betweenness` (ours, the best row), `infected_degree` (ours), `dynamic_age` (ours), `graphsl_slvae` (external, learned, KDD 2022), `graphsl_ivgd` (external, learned, WWW 2022). `lpsi` is printed as the literature reference row because it is the method SIDSL's table puts above SL-VAE, even though it is weak on the dense jazz graph. Both observations are reported (`marginal` and the comparable `binary`); no published SL number exists under IC or LT.

### 4.5 Influence blocking (`influence_blocking`, IC and CLT, two cascades; rumour size, lower is better)

| Rank | Dataset | Nodes | Arcs | Tier | Kind | Why |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `email_eu_core` | 1,005 | 24,929 | medium | real | results exist; the department labels make the region diagnostics readable |
| 2 | `p2p_gnutella24` | 26,518 | 65,369 | large | real | matches its published count exactly; the large tier at a sparsity the competitive simulator handles |
| 3 | `epinions1` | 75,879 | 508,837 | large | real | Tong INFOCOM 2017's row digit for digit; the scale row |
| 4 | `cit_hepth` | 27,769 | 352,768 | large | real | SandIMIN and the Xie papers report on it; dense, the cost stress test |
| 5 | `powerlaw_cluster` | 10,000 | `--plc-p 0.05` | large | synthetic | StratLearner's power-law family; the rumour-to-blocker ratio at fixed structure |

Survey (email_eu_core, rumour size): `rps` 42.9, `reverse_blocking` 45.7, `degree_blocking` 46.4, `proximity` 46.6, `multi_hop_proximity` 46.9, `pagerank_blocking` 48.0, `cldag` 49.0, `betweenness_blocking` 59.4, `cmia_o` 61.2, `random_blocking` 87.0, `forward_blocking` 96.6; `evolve_free@oracle` 43.5. External (node lever, k=10 and 20 only): `imin_joc` 53.9, `sandimin` 54.7.

Baselines (counter-seed lever): `rps` (ours, the best row), `reverse_blocking` (ours), `proximity` (ours, the row StratLearner's table puts above every learned method), `degree_blocking` (ours; it beat proximity here, the opposite of CLDAG's claim, which is worth a sentence), `cldag` (ours, the founding LT method). `forward_blocking` is dropped: it scored below random blocking because its picks are the rumour's dead-end neighbours (section 7), a weakness of the published heuristic rather than a bug. The wired repos answer other levers and go in the lever table: `sandimin` (PVLDB 2024, node lever), `imin_joc` (INFORMS JoC 2025, node lever). `diffim` was deleted on 2026-09-04: it trains a surrogate for hours per budget and stalled the August sweep for two days. Stretch: `slashdot`, `gowalla`.

### 4.6 Cascade reconstruction (`cascade_reconstruction`, IC and LT, traced parents, likelihood reward; tree score reported post hoc)

| Rank | Dataset | Nodes | Edges | Tier | Kind | Why |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `uci_students` | 1,266 | 6,451 | medium | real | Xiao ICDM 2018's Table I digit for digit; the smallest row in the matrix (replaces the 410-node `infectious`, where the only results so far were run) |
| 2 | `ca_grqc` | 5,242 | 14,484 | medium | real | the graph where a random walker beats tree sampling (research file trap 4); if the search cannot clear PPR here the result is not real |
| 3 | `rt_pol` | 18,470 | 48,053 | large | real | DITTO's larger real graph; the inner loop is the hottest in the repo, so this is where arm 4 may not complete, which is itself a finding |
| 4 | `oregon2` | 11,461 | 32,730 | large | real | DITTO's Table 2 exactly, the file its own loader fetches |
| 5 | `ba` | 10,000 | $m = 3$ | large | synthetic | DITTO's own synthetic family (`generate_data_ba_er.sbatch` holds its parameters), at the matrix's one synthetic size |

Survey (infectious, the pre-likelihood tree score, one budget): `ordered_steiner_closure` 0.309, `cri` 0.251, `jordan_backward` 0.200, `observed_only` 0.200, `steiner_tree` 0.180, `dhrec` 0.148, `delayed_bfs` 0.128, `greedy_ordered` 0.100, `cult` 0.095, `random_reconstruction` 0.082, `consistent_tree_wpct` and `wbct` 0.078, `one_hop` 0.076, `netfill` 0.073, `personalized_pagerank` 0.072, `tree_sampling` 0.058; `evolve_free@oracle` 0.363. External: `ditto`, `ditto_cri`, `ditto_dhrec`, `grin`, `spin` and `deep_demixing` all skipped by the `work_dir` bug.

Baselines: `ordered_steiner_closure` (ours, the best row), `cri` (ours), `dhrec` (ours), `ditto` (external, learned, KDD 2023, the reference), `grin` (external, learned, ICLR 2022, DITTO's upper bound; fits on the selection split's labelled histories and its row says so). Alternates: `steiner_tree`, `delayed_bfs`. Control printed above the table: `observed_only`. The reward changed on 2026-09-04, so this ranking is re-checked on the first rerun. Setting `final_snapshot` leads; the other three settings are T5 protocol rows, never pooled. Stretch: `email_enron` (36,692) as a large graph outside the CR literature.

### 4.7 Epidemic control (`epidemic_control`, SIR, SIS, SEIR, per-arc $\beta$; attack size, lower is better)

| Rank | Dataset | Nodes | Edges | Tier | Kind | Why |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `infectious_sociopatterns` | 10,972 | 44,517 | large | real | the full SocioPatterns trace, the only contact network above 1,000 nodes; keeps a real contact structure in the table |
| 2 | `oregon1` | 10,670 | 22,002 | large | real | GreedyWalk's $\lambda_1 = 58.72$, reproduced exactly; the spectral line's check |
| 3 | `brightkite` | 58,228 | 214,078 | large | real | GreedyWalk's $\lambda_1 = 101.49$, reproduced; the scale row, and our stepper is not NDlib so it may go further than the IC tasks |
| 4 | `p2p_gnutella05` | 8,846 | 31,839 arcs | medium | real | GreedyWalk's own directed graph; the medium tier |
| 5 | `ba` | 10,000 | $m = 3$ | large | synthetic | the family of the BA-300 DAVA-vs-NetShield example, at the matrix's one synthetic size |

Survey (primary_school, attack size): `frontier_immunization` 130.5 (control), `dava` and `dava_fast` 132.9, `netshield_plus` 140.1, `greedy_walk` 140.2, `kshell_immunization` 140.3, `degree_immunization` 140.7, `netshield` 141.3, `betweenness_immunization` 141.7, `adaptive_degree_immunization` 141.8, `eigenvector_immunization` 143.2, `acquaintance_immunization` 147.5, `random_immunization` 150.6; `evolve_free@oracle` 127.1. External: `explosive_immunization_epi` 140.8, `collective_influence_epi` 147.2; the six `netimm_*` rows skipped by the `work_dir` bug, `gdm_epi` and `dismantling_review_epi` at import (same missing modules as in critical node detection).

Baselines: `dava` (ours, byte-identical to `netimm_dava`, the best method row and the one we position against), `netshield_plus` (ours), `greedy_walk` (ours), `degree_immunization` (ours, the row RLGN ties with eigenvector), `finder_epi` (external, learned). Alternate: `netshield`. Control printed above the table: `frontier_immunization`, the row that beat the agent in August. Lever `vaccinate` in T3; the four levers are a T5 axis. Under SIS the reported quantity is endemic prevalence. Dropping the small rows costs this task its labelled contact traces (`primary_school`, where the August results were run) and the 75-node `hospital_lh10` ward where the DAVA-vs-NetShield order reverses; that reversal can survive as one appendix row. Stretch: `youtube` (GreedyWalk's $\lambda_1 = 210.4$).

### 4.8 Cascade prediction (`cascade_prediction`, real logs replayed; MSLE, lower is better)

| Rank | Dataset | Cascades | Underlying nodes | Windows | Tier | Why |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `casflow_weibo` | 119,313 | 6,738,040 | 0.5 h and 1 h observed, 24 h horizon | very large | the de-facto benchmark; the CasFlow bundle |
| 2 | `casflow_aps` | 207,685 | 616,316 | citation years | very large | results exist; the corpus whose published band moves most under the leak-free split |
| 3 | `taoke` | 2,862 | 29,711 | 3,600 s observed | large | CasTemp's own corpus, the only one with a published leak-free number; needs `--cp-min-size 3` |
| 4 | `casflow_twitter` | 88,440 | 490,474 | 1 d and 2 d observed, 32 d horizon | very large | the third CasFlow corpus; where CasCN becomes the best baseline under the fix |
| 5 | `digg_cascades` | 3,553 | 279,630 | votes | very large | Topo-LSTM's corpus; a graph-and-cascade pair rather than a bundle |

These are already the largest objects in the repo; there is no synthetic row by construction. Survey (casflow_aps under `--cp-max-nodes`, so a censored corpus): `degree_scaled` 0.064, `rpp` 0.118, `feature_linear` 0.128, `branching_factor` 0.129, `hawkes` 0.160, `persistence` 0.168, `weng_communities` 0.211, `hip` 0.265, `feature_gbt` 0.274, `seismic` 0.279, `szabo_huberman` 0.313, `hawkes_hybrid` 0.482, `neighborhood_size` 0.574, `mean_size` 0.920, `random_prediction` 2.32, `reachability` 32.3; `evolve_free@oracle` 0.094 with zero forward-model calls. External: `casflow`, `ccgl`, `ctcp`, `cascn` and `coupledgnn` all skipped by the `work_dir` bug.

Baselines: `degree_scaled` (ours, the best row, the graph-only floor), `rpp` (ours, the generative line), `feature_linear` (ours, the feature line), `casflow` (external, learned, TKDE 2021, the reference), `cascn` (external, learned, ICDE 2019, the row whose ranking changes under the leak-free split). `szabo_huberman` is printed as the literature reference row; `mean_size` and `persistence` are printed as floors. The node cap has to be the same for every row of a corpus and stated in the caption, since a capped corpus is comparable only within itself.

## 5. The tables

**T1, dynamics learning** (one row per task, dynamics and dataset; every run contributes): `delta_f1`, `brier`, persistence change-F1 (zero by construction, the floor), rollout `ens_count_bias`, `ens_marg_mae`, `ens_final_count_model / true`, `action_sensitivity`, and `wm_minus_mc` on the winning strategy. The `structured_oracle` row for the same dynamics is the ceiling. Cascade prediction contributes its hard-target one-step numbers as the falsification test of the IC assumption.

**T2, efficiency** (one row per task and dataset, ordered by size): seconds per rollout sample for the world model, the oracle referee and NDlib (`rollout_seconds / n_samples`, `referee_rollout_seconds / referee_samples`, and `mc_rollout_seconds / mc_agreement_runs` from the agreement check), the speedup, `train_seconds` and data-generation seconds, and the break-even rollout count $\lceil (t_{\text{train}} + t_{\text{data}}) / (t_{\text{MC}} - t_{\text{WM}}) \rceil$; then the `evaluator_seconds` and `real_env_episodes` conditions 4 and 6 actually spent in the same search. The size ordering is the point: the speedup should grow with the graph. Adaptive IM adds the per-round re-estimation cost, cascade reconstruction adds `kernel_calls_per_instance`, source localization `forward_calls_per_instance`.

**T3, main results** (one block per task and dynamics; columns are budgets; rows are the five baselines, condition 2, and the four ladder arms; every cell is the oracle referee's `referee_reward` with its standard error, and the critical rows also carry `mc_reward` from the NDlib agreement check): the downstream-utility claim is the difference between the `@world_model` and `@native` rows with the `@monte_carlo` and `@oracle` rows as the ceiling, and the baseline rows are the published comparison. The three critical datasets per task fill the main-text table; the two secondary datasets go to the appendix version. A ladder arm that did not finish on a scale row is printed as such with its evaluator budget, never dropped.

**T4, algorithm discovery** (critical datasets only): condition 6 against the three discovery systems at their published budgets, each row carrying `evaluator_calls`, `real_env_episodes`, `evaluator_seconds` and the LLM call count, plus the routed classical pool and the best single baseline for scale. Beside it, the case studies: for the winner of each critical dataset, the write-up's first-attempt-to-winner trajectory, its closest classical algorithm and the parts marked new, with the top twenty programs across all runs in the supplement.

**T5, generalization**: T3 sliced four ways. By dataset and size tier (the five rows per task, medium through very large), by topology (the synthetic families against the real graphs, and the graphs with ground-truth communities, `email_eu_core` with its 42 departments and the `sbm` rows with their planted blocks, for the region diagnostics), by dynamics (IC against LT everywhere; SIR, SIS, SEIR for epidemic control; SIR at $\gamma = 1$ against IC once), and by transfer (one world model per dataset reused across the tasks that share its dynamics, from `scripts/transfer_matrix.py`; a source-localization program moved to another graph with `--sl-transfer-from`; the chronological against the random split for cascade prediction).

## 6. Run order and volume

Critical datasets first, in rank order, so the main-text table fills before the appendix. The first job on every large and very-large row is data generation alone, timed, because nothing above `nethept` has been generated yet; its wall clock decides `--mc-runs`, `--n-samples` and the horizon for that row before any arm runs. Per task that is three datasets, two dynamics (three for epidemic control, one for cascade prediction), all budgets, the five baselines, the ladder, and condition 9 on the critical rows. Rough count: 8 tasks, 5 datasets, an average of 2 dynamics, so about 80 pipeline runs plus the synthetic sweeps, of which 48 are critical and carry the discovery rows. The very-large rows (`digg`, `twitter`, `epinions1`, `brightkite`, `deezer`) run last.

Expected cost per pipeline run (one task, one dynamics, four budgets, five classical baselines, the four-arm ladder with the referee replay; no external learned repo, no discovery system), from the measured netscience runs and a Monte Carlo microbenchmark on BA graphs at 1,000, 10,000 and 100,000 nodes: about 4 to 12 hours at 1,000 to 2,000 nodes, 6 to 16 hours at 10,000, and 1 to 3 days at 100,000. The LLM latency (16 arm-budget pairs at 20 iterations, 1 to 3 minutes per reply) is the largest block at every size and does not scale with the graph; data generation and training scale linearly with edges (20 to 40 minutes and 1 to 2 hours at 10,000; 3 to 6 hours and 10 to 24 hours at 100,000, where the training batch no longer fits and shrinks); the Monte Carlo evaluator arm costs about an hour per dynamics at 10,000 and 4 to 8 hours at 100,000, where a single evaluation is 90 seconds; a generated program that does its own sampling hits the 300-second per-call cap routinely at 100,000. Outside that table, DeepIM and MOEIM took 10.6 hours between them at netscience and do not get cheaper with size, and each discovery system is LLM-bound at one to two days per budget point. Old source-localization and cascade-reconstruction results predate the label-free rewards and are not comparable; both tasks rerun from scratch.

## 7. Open decisions and things the survey turned up

- `forward_blocking` scored 96.6 against 87.0 for random blocking on email_eu_core. Reproduced on 2026-09-04 with the run's own graph and rumour: the implementation matches its definition (seed the nodes the simulated rumour reaches most often), and those nodes are the rumour's dead-end neighbours (out-degree 0 to 6, against 47 to 81 for random's picks), so they save themselves and nothing else. It is the published heuristic being weak on a dense graph, not a bug; it stays out of the five but is fine to cite.
- On jazz the infected-subgraph centralities beat LPSI by more than double; whether that holds on sparse graphs (cora_ml, power_grid) decides if LPSI stays a reference row or is dropped.
- Cascade prediction's `--cp-max-nodes` cap for `casflow_weibo` and `casflow_aps`: the same cap for every row of a corpus, stated in the caption; the August APS run went out of memory at 64 GB uncapped.
- Whether influence blocking's lever table (sandimin, imin_joc) is main text or appendix; T3 stays counter-seed either way.
- The discovery systems' budgets differ from ours by 5 to 50 times by design; T4 reports both at published defaults and states it, rather than matching budgets.
- Whether the write-up prompt should forbid literal node ids in generated programs before the case-study harvest, so the top twenty are graph-general algorithms rather than tuned seed lists.
- The scale rows have never been run; if `@monte_carlo` cannot finish on them under any budget, that row's T3 cell reports the budget it exhausted, and the C2 table is where the result lives.
- Whether the epidemic-control DAVA-vs-NetShield reversal on `hospital_lh10` (75 nodes) gets one appendix row now that no small graph is in the main matrix.
- External repos, state as of 2026-09-04. Fixed in the repo and covered by tests or a local end-to-end run: `gdm` and `gdm_epi` (adapter rewritten against the real layout, GDM's own `models/GAT.py` on its published four-feature checkpoint with the 2019 PyG state dict converted and verified numerically; `baselines/gdm_support.py`), `selinda` (same fix through the repo's own `agent/gdm.py`; it reproduces the gdm row exactly and is a cross-check, not coverage), `glie` (its runner now prints a bracketed seed list), `moeim` (duplicate seeds deduplicated in both parsers), `diffim` deleted outright (its surrogate training stalled the sweep; the user's call). Cluster setup only: `setup_baselines --all` rebuilds the venvs for `rl4im` (the `numpy<1.24` patch), `mind`, `nirm`, `dcrs` and `touplegdd`; the `work_dir` group (all CP, CR and SL repos, the six `netimm_*` rows) needs one confirming run after the 2026-08-23 fix. Not fixable here: `finder` needs a CPython 3.7 on the box; `dismantling_review` needs `graph_tool`, which pip cannot install.
