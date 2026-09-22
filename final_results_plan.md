# Final Results Plan

The experiment matrix for the paper's main results, revised 2026-09-04 after the first draft was found to be capped at small graphs. It names, per graph task, the five datasets we evaluate on (three critical, two secondary; no graph under 1,000 nodes, and every synthetic row at one size, 10,000 nodes), the five baselines chosen from what the August sweep actually scored, the three algorithm-discovery systems, and the tables each claim is read from. Every value here is one the pipeline accepts today: every dataset is a `--dataset` loader or synthetic family, every baseline a `--baselines` name or a wired `external:<name>` entry in `baselines/registry.py`, every discovery system a wired `discovery:<name>` entry. `test_commands.md` holds the submission commands; `setup_guide.md` the environment.

## 1. The five claims and where each is read

| Claim                  | Question                                                                                                   | Evidence                                                                                                                                                                                                                                                                                                                                                                           | Table            |
| ---------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| C1 Dynamics learning   | Does the GWM reconstruct unknown graph dynamics?                                                           | One-step `delta_f1` and `brier` against MC marginals; free-running rollout `ens_count_bias`, `ens_marg_mae`, final count model vs true; `action_sensitivity` and `add_seed_success` for action conditioning; `wm_minus_mc` on the exact winning strategy; the `structured_oracle` ceiling beside it                                                                                | T1               |
| C2 Efficiency          | How much faster are imagined rollouts than MC or oracle simulation, including the amortized training cost? | `rollout_seconds` vs `mc_rollout_seconds` per rollout sample; `evaluator_seconds` spent by conditions 5 and 6 in the same search (condition 4 only where it is run as an ablation); `train_seconds` plus data-generation seconds, converted to a break-even rollout count. The scale rows are where this claim is largest and where `@monte_carlo` may not finish at all                                                             | T2               |
| C3 Downstream utility  | Does planning with the GWM improve graph-task performance?                                                 | The ladder replayed on the shared oracle referee: conditions 3, 4, 5, 6 with the method fixed and only the evaluator varying, beside the five baselines; `planning_regret_multi` vs degree and random                                                                                                                                                                              | T3 (main), T1    |
| C4 Algorithm discovery | Does GWM access let the coding agent find better algorithms under the same interaction budget?             | Condition 6 against condition 9 (three published discovery systems on the plain simulator) at reported `evaluator_calls`, `real_env_episodes` and `evaluator_seconds`; the write-up's trajectory, closest-classical and novelty sections as case studies                                                                                                                           | T4, case studies |
| C5 Generalization      | Does this hold across datasets, sizes, topologies and dynamics?                                            | T3 repeated over the five datasets per task (medium, large and very large tiers, real and synthetic), IC and LT for every simulator task, SIR/SIS/SEIR for epidemic control, chronological vs random split for cascade prediction, one world model per dataset reused across tasks (`scripts/transfer_matrix.py`), program transfer for source localization (`--sl-transfer-from`) | T5               |

T1 and T2 come free with every run (the `## World Model` section of each report and the timing rows), so they are populated on every dataset in the matrix. T3 is the main table. T4 runs on the critical datasets only. T5 is T3 sliced by axis.

## 2. Selection rules

**Size tiers.** Small is under 1,000 nodes, medium 1,000 to 10,000, large 10,000 to 100,000, very large above 100,000. The first draft of this file stopped at large because the project brief warns that the seven biggest graphs exceed what the NDlib rollout path simulates in reasonable time. That warning is about the Monte Carlo evaluator, and the Monte Carlo evaluator being too slow is exactly claim C2, so the scale rows are in the matrix on purpose. Small graphs are not in the matrix at all: every row is at least 1,000 nodes, because a small graph shows no evaluator speedup and duplicates what the medium rows already measure; the small loaders stay in the catalogue below for the record. Nothing above 15,229 nodes (`nethept`) has been generated or trained on yet; every large and very-large row therefore starts with a timed data-generation job before anything else is submitted on it.

**Datasets.** Five per task, ranked. The three critical datasets are the ones the paper cannot ship without, and on their own they span at least two size tiers and two structures; the third critical row of every simulator task is a large or very-large graph. Real graphs fill the critical set wherever the task's literature reports on them, because those rows are comparable to published numbers. Every simulator task carries exactly one synthetic row, at one size, 10,000 nodes, with no size sweep: 10,000 is the largest size at which the whole ladder including the Monte Carlo arm and a discovery system finishes in under a day per dynamics, it puts the synthetic row in the large tier with a structure no real row has (planted communities for `sbm`, a clean power law with tunable clustering for `ba` and `powerlaw_cluster`), and the very-large real rows already carry the 100,000-class tier. At least one community-structured graph (`sbm` or a labelled real graph) is in every task, because on a degree-trivial graph every backbone ties the degree heuristic. Cascade prediction is observational and has no synthetic option; its corpora are already the largest objects in the repo.

**Cost controls for the scale rows.** `--mc-runs` and `--n-samples` down from 200, the horizon capped, `--cp-max-nodes` for the two CasFlow corpora, and the evaluator budget of every arm recorded so a row that did not finish is reported as such rather than dropped silently. The measures are stated in the table caption.

**Baselines.** Five per task, chosen from the arms the August sweep scored on its Monte Carlo referee (the per-task survey tables below give the mean referee score across budgets, from `results/` and `results_archive/`), plus the wired learned or state-of-the-art repos that answer the same question. The external repos ran in the same sweep and their rows live under `baselines/<budget>/external_*.json` rather than `agent/`; thirteen scored, and about forty were skipped for reasons listed per task below and collected in section 7. A control that is not a method (the outbreak ring, the reports-only decoder, the two prediction floors) is always run and printed above the table but does not count toward the five. `celf_pp` is removed from influence maximization entirely.

**Discovery systems.** One common set of three for every task, chosen for three different search paradigms: `llm4ad_funsearch` (FunSearch, the origin: an island program database with best-shot prompting), `eoh` (Evolution of Heuristics, ICML 2024 oral: co-evolution of a natural-language thought and its code), and `openevolve` (the open AlphaEvolve implementation: a MAP-Elites program database with cascade evaluation). All three run their own loop at published defaults on the plain simulator through `baselines/score_program.py`; the world model is unreachable by construction. Since 2026-09-20 they run in section 9.5 only (influence maximization on `netscience` and critical node detection on `power_grid`, both IC, one search at 10 percent transferred to the other three budgets), not on every task's critical datasets. Alternates: `reevo`, `mcts_ahd`, `llamea`.

**Conditions per table row.** Every T3 cell has the five baselines (condition 1 or 7), the routed classical pool (condition 2), and the two-arm ladder `evolve_free@oracle`, `evolve_free@world_model` (conditions 5 and 6; the native and Monte Carlo arms, conditions 3 and 4, were dropped from the matrix on 2026-09-10 and are ablation rows only), all replayed on the shared referee: the exact oracle simulator at 1,000 samples (`--referee oracle --referee-samples 1000`, the defaults, reduced to 200 samples on the very-large rows). `--mc-agreement`, the second replay on the sampled simulator, is OFF on every row of the matrix: section 8.0 states what that gives up and how to recover a single timing row if one is wanted. Adaptive IM runs the same two arms; the `adaptive_<mode>@<evaluator>` twins that make the adaptivity gap computable are ablation rows, not matrix rows, since 2026-09-10.

**Budgets and dynamics** follow `pipeline/tasks.py`: 1, 5, 10 and 20 percent of $N$ for the two IM tasks and critical node detection (outbreak 10 percent of $N$), absolute $k \in \{10, 20, 30, 40, 50\}$ for influence blocking (rumour 1 percent of $N$), a single 10 percent point for the two inverse tasks and cascade prediction, and the epidemic task's percent ladder against a 1 percent outbreak. IC and LT for every simulator task, SIR, SIS and SEIR for epidemic control, with the SIR $\gamma = 1$ check against IC reported once.

## 3. Dataset catalogue

Every loader in `data/datasets/` (79 datasets; the four helper modules `cascade_common`, `casflow_bundle`, `dismantling_common`, `sociopatterns` are not datasets) plus the seven synthetic families, as loaded. Undirected rows quote edges, directed rows quote arcs. Any simulator task can run on any graph; cascade prediction accepts only the corpora marked CP. The small tier is listed for completeness and is not in the matrix.

**Small (under 1,000 nodes)**

| Dataset            | Nodes | Edges | Kind                                 | Task family                      |
| ------------------ | ----- | ----- | ------------------------------------ | -------------------------------- |
| `dolphins`         | 62    | 159   | undirected social                    | source localization              |
| `hospital_lh10`    | 75    | 1,139 | contact trace, dense ward            | epidemic control                 |
| `kenya_households` | 75    | 576   | contact trace, 5 household labels    | epidemic control                 |
| `malawi_village`   | 86    | 347   | contact trace                        | epidemic control                 |
| `workplace_invs13` | 92    | 755   | contact trace                        | epidemic control                 |
| `hypertext09`      | 113   | 2,196 | contact trace                        | epidemic control                 |
| `football`         | 115   | 613   | undirected, 12 conference labels     | epidemic control (control graph) |
| `high_school_2011` | 126   | 1,709 | contact trace                        | epidemic control                 |
| `high_school_2012` | 180   | 2,220 | contact trace                        | epidemic control                 |
| `jazz`             | 198   | 2,742 | undirected collaborations, dense     | IM, SL, CR                       |
| `workplace_invs15` | 217   | 4,274 | contact trace                        | epidemic control                 |
| `primary_school`   | 242   | 8,317 | contact trace, 11 class labels       | epidemic control                 |
| `corruption`       | 309   | 3,281 | undirected co-scandal, GCC           | dismantling                      |
| `high_school_2013` | 327   | 5,818 | contact trace                        | epidemic control                 |
| `usair97`          | 332   | 2,126 | undirected air routes, hub-dominated | dismantling                      |
| `sfhh`             | 403   | 9,565 | contact trace                        | epidemic control                 |
| `infectious`       | 410   | 2,765 | contact trace (Xiao's extraction)    | cascade reconstruction           |
| `crime`            | 754   | 2,127 | undirected co-offending, GCC         | dismantling                      |

**Medium (1,000 to 10,000 nodes)**

| Dataset           | Nodes                                 | Edges               | Kind                                                   | Task family                 |
| ----------------- | ------------------------------------- | ------------------- | ------------------------------------------------------ | --------------------------- |
| `email_eu_core`   | 1,005                                 | 24,929 arcs         | directed emails, 42 department labels                  | IM, IB                      |
| `road_eu`         | 1,039                                 | 1,305               | undirected E-roads, GCC                                | dismantling                 |
| `email_univ`      | 1,133                                 | 5,451               | undirected email                                       | cascade reconstruction      |
| `euroroad`        | 1,174                                 | 1,417               | undirected E-roads, raw                                | dismantling                 |
| `uci_students`    | 1,266                                 | 6,451               | undirected messaging                                   | cascade reconstruction      |
| `netscience`      | 1,589                                 | 2,742               | undirected coauthorship, sparse                        | IM, SL                      |
| `citeseer`        | 1,681                                 | 2,902               | undirected citations, LCC under standardize            | cascade reconstruction      |
| `hamsterster`     | 2,000                                 | 16,098              | undirected pet social, GCC                             | dismantling                 |
| `ppi_yeast`       | 2,224                                 | 6,609               | undirected protein interactions, GCC                   | dismantling                 |
| `cora_ml`         | 2,810                                 | 7,981               | undirected citations, standardized, 2,879-dim features | IM, SL                      |
| `openflights`     | 2,939                                 | 15,677              | undirected flight routes                               | dismantling                 |
| `human_ppi_vidal` | 3,023                                 | 6,149               | undirected human interactome                           | dismantling                 |
| `facebook`        | 4,039                                 | 88,234              | undirected friendships, very dense                     | IM                          |
| `power_grid`      | 4,941                                 | 6,594               | undirected power lines, mesh                           | IM, CND, SL, CR             |
| `memetracker`     | 5,000 hosts (published; ours differs) | 313,669 (published) | 54,847 cascades on hosts                               | CP only                     |
| `ca_grqc`         | 5,242                                 | 14,484              | undirected coauthorship, assortative                   | IM, CR                      |
| `p2p_gnutella08`  | 6,301                                 | 20,777 arcs         | directed P2P, Aug 2002                                 | influence blocking          |
| `intnet1`         | 6,474                                 | 12,572              | undirected AS peering, GCC                             | dismantling                 |
| `wiki_vote`       | 7,115                                 | 103,689 arcs        | directed votes                                         | IM                          |
| `lastfm_asia`     | 7,624                                 | 27,806              | undirected, 18 country labels                          | IM                          |
| `ca_hepth`        | 8,638                                 | 24,806              | undirected coauthorship                                | cascade reconstruction      |
| `p2p_gnutella06`  | 8,717                                 | 31,525 arcs         | directed P2P                                           | epidemic control (spectral) |
| `p2p_gnutella05`  | 8,846                                 | 31,839 arcs         | directed P2P                                           | epidemic control (spectral) |

**Large (10,000 to 100,000 nodes)**

| Dataset                    | Nodes  | Edges          | Kind                                     | Task family                 |
| -------------------------- | ------ | -------------- | ---------------------------------------- | --------------------------- |
| `oregon1`                  | 10,670 | 22,002         | undirected AS graph, $\lambda_1 = 58.72$ | epidemic control (spectral) |
| `pgp`                      | 10,680 | 24,316         | undirected web of trust                  | dismantling                 |
| `oregon2_010331`           | 10,900 | 31,180         | undirected AS graph, 2001-03-31          | epidemic control (spectral) |
| `infectious_sociopatterns` | 10,972 | 44,517         | full contact trace                       | epidemic control            |
| `oregon2`                  | 11,461 | 32,730         | undirected AS graph, 2001-05-26          | cascade reconstruction      |
| `nethept`                  | 15,229 | 62,752 arcs    | directed coauthorship, both arcs         | IM, AOIM                    |
| `rt_pol`                   | 18,470 | 48,053         | undirected retweets                      | cascade reconstruction      |
| `p2p_gnutella24`           | 26,518 | 65,369 arcs    | directed P2P                             | influence blocking          |
| `cit_hepth`                | 27,769 | 352,768 arcs   | directed citations, dense                | influence blocking          |
| `taoke`                    | 29,711 | (cascade log)  | 2,862 cascades                           | CP only                     |
| `cit_hepph`                | 34,546 | 421,534 arcs   | directed citations, dense                | influence blocking          |
| `email_enron`              | 36,692 | 183,831        | undirected email                         | influence blocking          |
| `netphy`                   | 37,154 | 174,161        | undirected coauthorship                  | IM                          |
| `deezer_ro`                | 41,773 | 125,826        | undirected friendships (RO)              | epidemic control (RLGN)     |
| `deezer`                   | 47,538 | 222,887        | undirected friendships (HU)              | source localization (cost)  |
| `brightkite`               | 58,228 | 214,078        | undirected, $\lambda_1 = 101.49$         | epidemic control (spectral) |
| `p2p_gnutella`             | 62,561 | 147,878        | undirected P2P, Gnutella31 GCC           | dismantling                 |
| `epinions1`                | 75,879 | 508,837 arcs   | directed trust                           | influence blocking          |
| `twitter`                  | 81,306 | 1,768,149 arcs | directed follows                         | IM                          |
| `slashdot`                 | 82,168 | 870,161 arcs   | directed, Feb 2009                       | influence blocking          |

**Very large (above 100,000 nodes)**

| Dataset              | Nodes     | Edges           | Kind                                                        | Task family                  |
| -------------------- | --------- | --------------- | ----------------------------------------------------------- | ---------------------------- |
| `digg`               | 116,893   | about 2.6M      | undirected friendships                                      | IM, AOIM (scale)                   |
| `epinions`           | 131,828   | 841,372 arcs    | directed signed trust, sign dropped                         | IM (scale)                   |
| `gowalla`            | 196,591   | 950,327         | undirected location friendships                             | influence blocking           |
| `email_euall`        | 265,009   | 418,956 arcs    | directed email                                              | influence blocking           |
| `digg_cascades`      | 279,630   | 1,731,653 arcs  | 3,553 vote cascades on the ISI graph                        | CP only                      |
| `web_stanford`       | 281,903   | 2,312,497 arcs  | directed hyperlinks                                         | influence blocking           |
| `dblp`               | 317,080   | 1,049,866       | undirected coauthorship                                     | influence blocking           |
| `higgs_twitter`      | 456,631   | 14,855,875 arcs | directed followers                                          | influence blocking (scale)   |
| `casflow_twitter`    | 490,474   | 1,903,230       | 88,440 hashtag cascades                                     | CP only                      |
| `aps`, `casflow_aps` | 616,316   | 3,304,400       | 90,768 (ours) or 207,685 (CasFlow filter) citation cascades | CP only                      |
| `youtube`            | 1,134,890 | 2,987,624       | undirected, $\lambda_1 = 210.4$                             | IM, epidemic control (scale) |
| `pokec`              | 1,632,803 | 30,622,564 arcs | directed friendships                                        | influence blocking (scale)   |
| `weibo`              | 1,787,443 | about 216M arcs | directed influence graph, manual download                   | IM (scale)                   |
| `orkut`              | 3,072,441 | 117,185,083     | undirected friendships                                      | IM (scale)                   |
| `livejournal`        | 4,847,571 | 68,993,773 arcs | directed friendships                                        | AOIM (scale)                 |
| `casflow_weibo`      | 6,738,040 | 15,249,636      | 119,313 retweet cascades                                    | CP only                      |

**Synthetic families** (size by `--syn-nodes`, bulk by `--num-graphs`): `er`, `ba` ($m$ by flag), `ws`, `sbm` (`--sbm-blocks`, `--sbm-p-in`, `--sbm-p-out`), `powerlaw_cluster` (`--plc-m`, `--plc-p`, RL4IM's family), `kronecker` (`--kron-variant`, ConTinEst's seeds), `karate` (34 nodes, bulk).

## 4. Per-task matrix

Rank 1 to 3 is critical, 4 and 5 secondary. The survey table under each task is the August sweep on its Monte Carlo referee: mean referee reward across the budgets that ran, from one dataset each, so it ranks the pool rather than measuring it. External repos are listed after the classical pool with the reason any of them was skipped; most skips are one of two cluster-side problems, the relative `work_dir` bug fixed on 2026-08-23 (the repo's interpreter could not open the driver script, `exited 2`) and venvs whose graph libraries never installed (`ModuleNotFoundError`, `exited 1`).

### 4.1 Influence maximization (`influence_maximization`, IC and LT; higher is better)

| Rank | Dataset      | Nodes   | Edges                                                 | Tier       | Kind      | Why                                                                                                                                        |
| ---- | ------------ | ------- | ----------------------------------------------------- | ---------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| 1    | `netscience` | 1,589   | 2,742                                                 | medium     | real      | the ladder already runs here; IM and adaptive IM results exist                                                                             |
| 2    | `nethept`    | 15,229  | 62,752 arcs                                           | large      | real      | the canonical IM benchmark of Chen et al., IMM and OPIM; a generation script exists                                                        |
| 3    | `digg`       | 116,893 | about 2.6M                                            | very large | real      | the scale row: the RIS baselines (IMM, OPIM, SubSIM) are built for exactly this size, and it is where MC evaluation becomes the bottleneck |
| 4    | `cora_ml`    | 2,810   | 7,981                                                 | medium     | real      | matches DeepIM's graph exactly, so the learned row is comparable                                                                           |
| 5    | `sbm`        | 10,000  | 10 blocks of 1,000, by `--sbm-p-in` and `--sbm-p-out` | large      | synthetic | planted communities, where degree heuristics fail; a large-tier structure no real row has                                                  |

Survey (netscience, IC and LT runs pooled): `imm` 556, `voterank` 532, `degree_discount` 506, `pagerank_seeds` 461, `random_seeds` 327, `high_degree` 324; ladder `@world_model` 538, `@monte_carlo` 614 (LT only), `@oracle` 456 (includes small smoke runs). External: `ssa` 556, `subsim` 550, `opim` 549, `deepim` 397 (8,588 s of its own training per budget), `moeim` 367 (2,144 s per budget, and it emitted a duplicate seed at 20 percent); `glie` ran but its seed output could not be parsed, and `touplegdd` had no venv.

| Rank | Baseline          | Source                                   | Kind                             | Reference                       | Survey | Why                                                                                                                                                                      |
| ---- | ----------------- | ---------------------------------------- | -------------------------------- | ------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1    | `imm`             | ours, `coding_agent/tools/algorithms.py` | RIS sampler                      | Tang, Shi and Xiao, SIGMOD 2015 | 556    | the strongest row of the survey and the canonical RIS method; the library version, so it runs under every evaluator (the registry's `imm` tarball is manual and unwired) |
| 2    | `opim`            | external `opim`                          | RIS sampler with an online bound | Tang et al., SIGMOD 2018        | 549    | the second published RIS line, built for graphs the size of the scale row                                                                                                |
| 3    | `subsim`          | external `subsim`                        | RIS sampler, subset sampling     | Guo et al., SIGMOD 2020         | 550    | the fastest RIS sampler at the very-large tier, which is where `digg` reads                                                                                              |
| 4    | `deepim`          | external `deepim`                        | learned                          | Ling et al., ICML 2023          | 397    | the learned row; its published graph is `cora_ml`, so that cell is directly comparable; 8,588 s of its own training per budget                                           |
| 5    | `degree_discount` | ours, `coding_agent/tools/algorithms.py` | heuristic                        | Chen, Wang and Yang, KDD 2009   | 506    | the strongest heuristic in the survey and the row every IM paper prints                                                                                                  |

### 4.2 Adaptive online IM (`adaptive_online_im`, IC and LT; higher is better)

| Rank | Dataset            | Nodes  | Edges                          | Tier   | Kind      | Why                                                                                                           |
| ---- | ------------------ | ------ | ------------------------------ | ------ | --------- | ------------------------------------------------------------------------------------------------------------- |
| 1    | `netscience`       | 1,589  | 2,742                          | medium | real      | results exist; the same graph as the static IM row, so the adaptivity gap reads against a known static number |
| 2    | `nethept`          | 15,229 | 62,752 arcs                    | large  | real      | Han et al.'s benchmark; a generation script exists                                                            |
| 3    | `digg`             | 116,893 | 2.6M edges                    | very large | real  | the IM scale row's graph and world model reused, so the adaptivity gap is measured on the same graph as the static row; per-round MC re-estimation at 2.6M edges is the cost the task exists to show |
| 4    | `powerlaw_cluster` | 10,000 | `--plc-p 0.05` (RL4IM's value) | large  | synthetic | RL4IM's family; results and a generation script exist at a smaller size                                       |
| 5    | `facebook`         | 4,039  | 88,234                         | medium | real      | the dense regime where rounds reveal little, so the theory cap on the adaptivity gap is visible               |

Survey (netscience): `imm` 518 (static), `adapt_epic` 513, `adapt_degree_discount` 453, `adapt_pagerank` 437, `static_split` 434, `adapt_degree` 344, `adapt_random` 330; `evolve_free@oracle` 527, `adaptive_free@oracle` 422. External: `adaptiveim` 468. `rl4im` was requested and skipped on every budget: it died at `import networkx`, because networkx 2.3's GraphML writer reads `np.int`, which numpy 1.24 removed; the registry now pins `numpy<1.24` in its requirements patch, so the fix is to rebuild that venv with `python -m baselines.setup_baselines --only rl4im` on the cluster and rerun.

| Rank | Baseline                | Source                                            | Kind                          | Reference                                           | Survey     | Why                                                                                                                                                 |
| ---- | ----------------------- | ------------------------------------------------- | ----------------------------- | --------------------------------------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `adaptiveim`            | external `adaptiveim`                             | adaptive greedy, rounds-aware | Han et al., PVLDB 2018                              | 468        | the task's own published code (AdaptGreedy and EPIC), replayed one batch per round                                                                  |
| 2    | `adapt_epic`            | ours, `coding_agent/tools/adaptive_algorithms.py` | adaptive RIS policy           | Han et al., PVLDB 2018                              | 513        | our EPIC, the strongest adaptive row; the cross-check on the external row                                                                           |
| 3    | `adapt_degree_discount` | ours, `coding_agent/tools/adaptive_algorithms.py` | adaptive heuristic            | DegreeDiscount restricted to susceptibles per round | 453        | the cheap per-round policy                                                                                                                          |
| 4    | `imm`                   | ours, `coding_agent/tools/algorithms.py`          | static RIS sampler            | Tang, Shi and Xiao, SIGMOD 2015                     | 518        | the strongest static row, the non-adaptive control that turns a spread into an adaptivity gap                                                       |
| 5    | `static_split`          | ours, `coding_agent/tools/adaptive_algorithms.py` | static set dealt per round    | control                                             | 434        | the same-machinery control: one static seed set, one batch per round                                                                                |
| 6    | `rl4im`                 | external `rl4im`                                  | learned                       | Chen et al., UAI 2021                               | not scored | conditional: runs only if the venv rebuild with the `numpy<1.24` patch succeeds on the cluster (`python -m baselines.setup_baselines --only rl4im`) |

### 4.3 Critical node detection (`critical_node_detection`, IC and LT, reactive containment; lower is better)

| Rank | Dataset        | Nodes  | Edges                       | Tier   | Kind      | Why                                                                                                          |
| ---- | -------------- | ------ | --------------------------- | ------ | --------- | ------------------------------------------------------------------------------------------------------------ |
| 1    | `power_grid`   | 4,941  | 6,594                       | medium | real      | byte-identical to CoreHD, BPD and NIRM's file; results exist; the predicted loss                             |
| 2    | `pgp`          | 10,680 | 24,316                      | large  | real      | a dismantling benchmark with a trust-web structure unlike a grid                                             |
| 3    | `p2p_gnutella` | 62,561 | 147,878                     | large  | real      | the largest of the twelve dismantling benchmarks, SNAP's giant component                                     |
| 4    | `openflights`  | 2,939  | 15,677                      | medium | real      | hub-dominated air routes, the opposite structure to a grid, at medium size (replaces the 332-node `usair97`) |
| 5    | `ba`           | 10,000 | $m = 3$, about 30,000 edges | large  | synthetic | FINDER's training family, at a size FINDER itself never trains on                                            |

Survey (power_grid, spread after removal): `bpd_r` 127.5, `collective_influence_r` 127.8, `corehd_r` 128.1, `explosive_immunization` 128.9, `decycling` 129.2, `gndr` 129.2, `corehd` 129.5, `adaptive_degree` 130.0, `degree_removal` 130.4, `betweenness_removal` 131.1, `kshell_removal` 134.9, `acquaintance_immunization` 137.2, `gnd` 138.8, `random_removal` 141.7, `netshield` 143.9; `evolve_free@oracle` 66.5 (before the ring control existed, so void). External: `explosive_immunization` 129.9, `gnd` 130.2, `collective_influence` 130.3, all within a point of our reimplementations. Every learned repo in that sweep skipped at import (`gdm` and `selinda`: `network_dismantling` missing; `mind`: `torch_scatter`; `nirm`: `torch_geometric`; `dcrs`: `dgl`), as did `dismantling_review` (`graph_tool`, which is not pip-installable); `finder` was not in that submission.

| Rank | Baseline                 | Source                                               | Kind                                  | Reference                                    | Survey     | Why                                                                                                       |
| ---- | ------------------------ | ---------------------------------------------------- | ------------------------------------- | -------------------------------------------- | ---------- | --------------------------------------------------------------------------------------------------------- |
| 1    | `bpd_r`                  | ours, `coding_agent/tools/dismantling_algorithms.py` | belief propagation plus reinsertion   | Mugisha and Zhou, PRE 2016                   | 127.5      | the best row of the survey                                                                                |
| 2    | `collective_influence_r` | ours, `coding_agent/tools/dismantling_algorithms.py` | collective influence plus reinsertion | Morone and Makse, Nature 2015                | 127.8      | CI as published, with the reinsertion pass                                                                |
| 3    | `adaptive_degree`        | ours, `coding_agent/tools/dismantling_algorithms.py` | recalculated highest degree (HDA)     | classical                                    | 130.0      | the row MIND's table puts within five points of deep RL; the heuristic that has to be beaten              |
| 4    | `gndr`                   | ours, `coding_agent/tools/dismantling_algorithms.py` | spectral partition plus reinsertion   | Ren et al., PNAS 2019                        | 129.2      | GND with reinsertion, a different method from `gnd`                                                       |
| 5    | `explosive_immunization` | ours, `coding_agent/tools/dismantling_algorithms.py` | explosive percolation                 | Clusella et al., PRL 2016                    | 128.9      | the external repo scored 129.9, within a point of this reimplementation                                   |
| 6    | `finder`                 | external `finder`                                    | learned                               | Fan et al., Nature Machine Intelligence 2020 | not scored | conditional: its TF 1.14 pin needs a CPython 3.7 on the cluster; the adapter is written and has never run |

### 4.4 Source localization (`source_localization`, IC and LT, label-free consistency reward; F1 reported post hoc)

| Rank | Dataset      | Nodes  | Edges                                                 | Tier   | Kind      | Why                                                                                                                                                                              |
| ---- | ------------ | ------ | ----------------------------------------------------- | ------ | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `cora_ml`    | 2,810  | 7,981                                                 | medium | real      | the graph SL-VAE, IVGD and GCNSI report on                                                                                                                                       |
| 2    | `power_grid` | 4,941  | 6,594                                                 | medium | real      | the hardest published SL graph: slow diffusion, so the observation says less about the source                                                                                    |
| 3    | `deezer`     | 47,538 | 222,887                                               | large  | real      | IVGD's scalability column (HU, its row to the digit); a cost target and the first large SL row anyone has run                                                                    |
| 4    | `netscience` | 1,589  | 2,742                                                 | medium | real      | in the SL benchmark family (the 2026 GNN benchmark); sparse with many components, the opposite of cora_ml (replaces the 198-node `jazz`, where the only results so far were run) |
| 5    | `sbm`        | 10,000 | 10 blocks of 1,000, by `--sbm-p-in` and `--sbm-p-out` | large  | synthetic | sources in different communities, the case where centre-of-region estimators fail                                                                                                |

Survey (jazz, the pre-label-free F1 reward, one budget): `infected_betweenness` 0.652, `infected_degree` 0.647, `infected_closeness` 0.630, `infected_eigenvector` 0.605, `dynamic_age` 0.565, `jordan_center` 0.428, `rumor_centrality` 0.415, `ojc` 0.408, `dmp_localize` 0.362, `netsleuth` 0.336, `effective_distance` 0.284, `lpsi` 0.282, `random_sources` 0.167; `evolve_free@oracle` 0.659. External: all six GraphSL rows and all four cosasi rows skipped by the `work_dir` bug, so none has a score yet.

| Rank | Baseline               | Source                                                | Kind                            | Reference                       | Survey     | Why                                                                                                                                      |
| ---- | ---------------------- | ----------------------------------------------------- | ------------------------------- | ------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `infected_betweenness` | ours, `coding_agent/tools/localization_algorithms.py` | infected-subgraph centrality    | Comin and Costa, PRE 2011       | 0.652      | the best row on jazz                                                                                                                     |
| 2    | `infected_degree`      | ours, `coding_agent/tools/localization_algorithms.py` | infected-subgraph centrality    | Comin and Costa, PRE 2011       | 0.647      | the cheap floor, and what the agent's jazz program reduced to                                                                            |
| 3    | `dynamic_age`          | ours, `coding_agent/tools/localization_algorithms.py` | spectral multi-source estimator | Fioriti and Chinnici, 2012      | 0.565      | the strongest non-centrality row                                                                                                         |
| 4    | `lpsi`                 | ours, `coding_agent/tools/localization_algorithms.py` | label propagation               | Wang et al., AAAI 2017          | 0.282      | the literature reference row: SIDSL's table puts it above SL-VAE on Digg; weak on dense jazz, so the sparse rows decide whether it stays |
| 5    | `rumor_centrality`     | ours, `coding_agent/tools/localization_algorithms.py` | rumor centrality                | Shah and Zaman, SIGMETRICS 2010 | 0.415      | the method that founded the field                                                                                                        |
| 6    | `graphsl_slvae`        | external `graphsl_slvae`                              | learned                         | Ling et al., KDD 2022           | not scored | conditional: skipped by the `work_dir` bug in August and needs one confirming run; the reference learned method of the SL literature     |

### 4.5 Influence blocking (`influence_blocking`, IC and CLT, two cascades; rumour size, lower is better)

| Rank | Dataset            | Nodes  | Arcs           | Tier   | Kind      | Why                                                                                                 |
| ---- | ------------------ | ------ | -------------- | ------ | --------- | --------------------------------------------------------------------------------------------------- |
| 1    | `email_eu_core`    | 1,005  | 24,929         | medium | real      | results exist; the department labels make the region diagnostics readable                           |
| 2    | `p2p_gnutella24`   | 26,518 | 65,369         | large  | real      | matches its published count exactly; the large tier at a sparsity the competitive simulator handles |
| 3    | `epinions1`        | 75,879 | 508,837        | large  | real      | Tong INFOCOM 2017's row digit for digit; the scale row                                              |
| 4    | `cit_hepth`        | 27,769 | 352,768        | large  | real      | SandIMIN and the Xie papers report on it; dense, the cost stress test                               |
| 5    | `powerlaw_cluster` | 10,000 | `--plc-p 0.05` | large  | synthetic | StratLearner's power-law family; the rumour-to-blocker ratio at fixed structure                     |

Survey (email_eu_core, rumour size): `rps` 42.9, `reverse_blocking` 45.7, `degree_blocking` 46.4, `proximity` 46.6, `multi_hop_proximity` 46.9, `pagerank_blocking` 48.0, `cldag` 49.0, `betweenness_blocking` 59.4, `cmia_o` 61.2, `random_blocking` 87.0, `forward_blocking` 96.6; `evolve_free@oracle` 43.5. External (node lever, k=10 and 20 only): `imin_joc` 53.9, `sandimin` 54.7.

| Rank | Baseline           | Source                                            | Kind                                          | Reference                                     | Survey | Why                                                                                                                   |
| ---- | ------------------ | ------------------------------------------------- | --------------------------------------------- | --------------------------------------------- | ------ | --------------------------------------------------------------------------------------------------------------------- |
| 1    | `rps`              | ours, `coding_agent/tools/blocking_algorithms.py` | reverse prevention sampling                   | Tong et al., INFOCOM 2017                     | 42.9   | the best row of the survey                                                                                            |
| 2    | `reverse_blocking` | ours, `coding_agent/tools/blocking_algorithms.py` | RR-set coverage inside the rumour's reach     | TC-AIBM's `Reverse` baseline                  | 45.7   | the second-best row                                                                                                   |
| 3    | `proximity`        | ours, `coding_agent/tools/blocking_algorithms.py` | rumour's out-neighbours, highest degree first | CLDAG's cheap baseline, He et al., SDM 2012   | 46.6   | the row StratLearner's table puts above every learned method                                                          |
| 4    | `cldag`            | ours, `coding_agent/tools/blocking_algorithms.py` | local-DAG blocking under CLT                  | He et al., SDM 2012                           | 49.0   | the founding LT method                                                                                                |
| 5    | `imin_joc`         | external `imin_joc`                               | AdvancedGreedy and GreedyReplace, node lever  | Xie et al., INFORMS Journal on Computing 2025 | 53.9   | node lever, scored at $k \in \{10, 20\}$; answers a different lever than the counter-seed rows and says so in its row |
| 6    | `sandimin`         | external `sandimin`                               | sandwich approximation, node lever            | PVLDB 2024                                    | 54.7   | node lever; the C++ repo, patched to emit its blocker set                                                             |

### 4.6 Cascade reconstruction (`cascade_reconstruction`, IC and LT, traced parents, likelihood reward; tree score reported post hoc)

| Rank | Dataset        | Nodes  | Edges   | Tier   | Kind      | Why                                                                                                                                                   |
| ---- | -------------- | ------ | ------- | ------ | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `uci_students` | 1,266  | 6,451   | medium | real      | Xiao ICDM 2018's Table I digit for digit; the smallest row in the matrix (replaces the 410-node `infectious`, where the only results so far were run) |
| 2    | `ca_grqc`      | 5,242  | 14,484  | medium | real      | the graph where a random walker beats tree sampling (research file trap 4); if the search cannot clear PPR here the result is not real                |
| 3    | `rt_pol`       | 18,470 | 48,053  | large  | real      | DITTO's larger real graph; the inner loop is the hottest in the repo, so this is where arm 4 may not complete, which is itself a finding              |
| 4    | `oregon2`      | 11,461 | 32,730  | large  | real      | DITTO's Table 2 exactly, the file its own loader fetches                                                                                              |
| 5    | `ba`           | 10,000 | $m = 3$ | large  | synthetic | DITTO's own synthetic family at the matrix's one synthetic size: one graph, $m = 3$, diffusion-only episodes with traced parents (the registry sets the last two), which is what the deleted `generate_data_ba_er.sbatch` ran at 1,000 nodes                                  |

Survey (infectious, the pre-likelihood tree score, one budget): `ordered_steiner_closure` 0.309, `cri` 0.251, `jordan_backward` 0.200, `observed_only` 0.200, `steiner_tree` 0.180, `dhrec` 0.148, `delayed_bfs` 0.128, `greedy_ordered` 0.100, `cult` 0.095, `random_reconstruction` 0.082, `consistent_tree_wpct` and `wbct` 0.078, `one_hop` 0.076, `netfill` 0.073, `personalized_pagerank` 0.072, `tree_sampling` 0.058; `evolve_free@oracle` 0.363. External: `ditto`, `ditto_cri`, `ditto_dhrec`, `grin`, `spin` and `deep_demixing` all skipped by the `work_dir` bug.

| Rank | Baseline          | Source                                                  | Kind                                               | Reference                          | Survey     | Why                                                                                                                                                     |
| ---- | ----------------- | ------------------------------------------------------- | -------------------------------------------------- | ---------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `steiner_tree`    | ours, `coding_agent/tools/reconstruction_algorithms.py` | minimum Steiner tree over the reports              | Xiao et al., SDM 2018              | 0.180      | the order-blind Steiner baseline                                                                                                                        |
| 2    | `jordan_backward` | ours, `coding_agent/tools/reconstruction_algorithms.py` | backward decode from the Jordan centres            | ours (research file, section 2.11) | 0.200      | the cheap first cut                                                                                                                                     |
| 3    | `cri`             | ours, `coding_agent/tools/reconstruction_algorithms.py` | cluster the infected subgraph, then reverse-infect | Chen et al., TNSE 2016             | 0.251      | DITTO's first MLE baseline; the best of the five here                                                                                                   |
| 4    | `dhrec`           | ours, `coding_agent/tools/reconstruction_algorithms.py` | DHREC                                              | Sefer and Kingsford, ICDM 2014     | 0.148      | DITTO's second MLE baseline                                                                                                                             |
| 5    | `observed_only`   | ours, `coding_agent/tools/reconstruction_algorithms.py` | reports only, infers nothing                       | Rozenshtein's `Reports` control    | 0.200      | precision 1.0 by construction; the concrete answer to whether the reward is gameable                                                                    |
| 6    | `ditto`           | external `ditto`                                        | learned                                            | KDD 2023                           | not scored | conditional: skipped by the `work_dir` bug in August; the reference method, which always solves the final-snapshot problem whatever `--cr-setting` runs |
| 7    | `grin`            | external `grin`                                         | learned imputer                                    | Cini et al., ICLR 2022             | not scored | conditional: same skip; DITTO's upper bound, fits on the selection split's labelled histories and its row says so                                       |

### 4.7 Epidemic control (`epidemic_control`, SIR, SIS, SEIR, per-arc $\beta$; attack size, lower is better)

| Rank | Dataset                    | Nodes  | Edges       | Tier   | Kind      | Why                                                                                                                               |
| ---- | -------------------------- | ------ | ----------- | ------ | --------- | --------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `infectious_sociopatterns` | 10,972 | 44,517      | large  | real      | the full SocioPatterns trace, the only contact network above 1,000 nodes; keeps a real contact structure in the table             |
| 2    | `oregon1`                  | 10,670 | 22,002      | large  | real      | GreedyWalk's $\lambda_1 = 58.72$, reproduced exactly; the spectral line's check                                                   |
| 3    | `brightkite`               | 58,228 | 214,078     | large  | real      | GreedyWalk's $\lambda_1 = 101.49$, reproduced; the scale row, and our stepper is not NDlib so it may go further than the IC tasks |
| 4    | `p2p_gnutella05`           | 8,846  | 31,839 arcs | medium | real      | GreedyWalk's own directed graph; the medium tier                                                                                  |
| 5    | `ba`                       | 10,000 | $m = 3$     | large  | synthetic | the family of the BA-300 DAVA-vs-NetShield example, at the matrix's one synthetic size                                            |

Survey (primary*school, attack size): `frontier_immunization` 130.5 (control), `dava` and `dava_fast` 132.9, `netshield_plus` 140.1, `greedy_walk` 140.2, `kshell_immunization` 140.3, `degree_immunization` 140.7, `netshield` 141.3, `betweenness_immunization` 141.7, `adaptive_degree_immunization` 141.8, `eigenvector_immunization` 143.2, `acquaintance_immunization` 147.5, `random_immunization` 150.6; `evolve_free@oracle` 127.1. External: `explosive_immunization_epi` 140.8, `collective_influence_epi` 147.2; the six `netimm*\*`rows skipped by the`work_dir`bug,`gdm_epi`and`dismantling_review_epi` at import (same missing modules as in critical node detection).

| Rank | Baseline                     | Source                                                | Kind                       | Reference                                | Survey | Why                                                                                   |
| ---- | ---------------------------- | ----------------------------------------------------- | -------------------------- | ---------------------------------------- | ------ | ------------------------------------------------------------------------------------- |
| 1    | `dava`                       | ours, `coding_agent/tools/immunization_algorithms.py` | data-aware dominator tree  | Zhang and Prakash, SDM 2014              | 132.9  | the best method row, byte-identical to `netimm_dava`, and the one we position against |
| 2    | `netshield_plus`             | ours, `coding_agent/tools/immunization_algorithms.py` | spectral, $\lambda_1$ drop | Chen et al., TKDE 2015                   | 140.1  | the spectral line's reference                                                         |
| 3    | `greedy_walk`                | ours, `coding_agent/tools/immunization_algorithms.py` | spectral, walk counting    | Saha et al., SDM 2015                    | 140.2  | the graphs `oregon1`, `brightkite` and `p2p_gnutella05` are its own                   |
| 4    | `degree_immunization`        | ours, `coding_agent/tools/immunization_algorithms.py` | targeted degree            | Pastor-Satorras and Vespignani, PRE 2002 | 140.7  | the row RLGN ties with eigenvector; the gap to random is the founding result          |
| 5    | `explosive_immunization_epi` | external `explosive_immunization_epi`                 | explosive percolation      | Clusella et al., PRL 2016                | 140.8  | the physics line, run through its own repo                                            |
| 6    | `collective_influence_epi`   | external `collective_influence_epi`                   | collective influence       | Morone and Makse, Nature 2015            | 147.2  | verified end to end under this task                                                   |

### 4.8 Cascade prediction (`cascade_prediction`, real logs replayed; MSLE, lower is better)

| Rank | Dataset           | Cascades | Underlying nodes | Windows                              | Tier       | Why                                                                                                                                                                                                                 |
| ---- | ----------------- | -------- | ---------------- | ------------------------------------ | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `casflow_aps`     | 207,685  | 616,316          | citation years                       | very large | results exist; the corpus whose published band moves most under the leak-free split                                                                                                                                 |
| 2    | `taoke`           | 2,862    | 29,711           | 3,600 s observed                     | large      | CasTemp's own corpus, the only one with a published leak-free number; needs `--cp-min-size 3`, and its only download source returned 404 on 2026-09-06, so the file has to be obtained by hand before this row runs |
| 3    | `digg_cascades`   | 3,553    | 279,630          | votes                                | very large | Topo-LSTM's corpus; a graph-and-cascade pair rather than a bundle                                                                                                                                                   |
| 4    | `casflow_twitter` | 88,440   | 490,474          | 1 d and 2 d observed, 32 d horizon   | very large | the third CasFlow corpus; where CasCN becomes the best baseline under the fix                                                                                                                                       |
| 5    | `casflow_weibo`   | 119,313  | 6,738,040        | 0.5 h and 1 h observed, 24 h horizon | very large | the de-facto benchmark; the CasFlow bundle                                                                                                                                                                          |

These are already the largest objects in the repo; there is no synthetic row by construction. Survey (casflow_aps under `--cp-max-nodes`, so a censored corpus): `degree_scaled` 0.064, `rpp` 0.118, `feature_linear` 0.128, `branching_factor` 0.129, `hawkes` 0.160, `persistence` 0.168, `weng_communities` 0.211, `hip` 0.265, `feature_gbt` 0.274, `seismic` 0.279, `szabo_huberman` 0.313, `hawkes_hybrid` 0.482, `neighborhood_size` 0.574, `mean_size` 0.920, `random_prediction` 2.32, `reachability` 32.3; `evolve_free@oracle` 0.094 with zero forward-model calls. External: `casflow`, `ccgl`, `ctcp`, `cascn` and `coupledgnn` all skipped by the `work_dir` bug.

| Rank | Baseline           | Source                                              | Kind                                     | Reference                              | Survey | Why                                              |
| ---- | ------------------ | --------------------------------------------------- | ---------------------------------------- | -------------------------------------- | ------ | ------------------------------------------------ |
| 1    | `feature_linear`   | ours, `coding_agent/tools/prediction_algorithms.py` | feature line, ridge regression           | Cheng et al., WWW 2014 feature classes | 0.128  | the feature line, CTCP's MLP row's honest cousin |
| 2    | `rpp`              | ours, `coding_agent/tools/prediction_algorithms.py` | generative, reinforced Poisson           | Shen et al., AAAI 2014                 | 0.118  | the best of the six                              |
| 3    | `szabo_huberman`   | ours, `coding_agent/tools/prediction_algorithms.py` | one-feature log-linear                   | Szabo and Huberman, CACM 2010          | 0.313  | the row every paper prints as Feature-S&H        |
| 4    | `hawkes`           | ours, `coding_agent/tools/prediction_algorithms.py` | generative, marked Hawkes process        | Mishra et al., CIKM 2016               | 0.160  | the self-exciting line                           |
| 5    | `persistence`      | ours, `coding_agent/tools/prediction_algorithms.py` | floor, $P(t_p) = P(t_o)$                 | control                                | 0.168  | the cascade is over                              |
| 6    | `weng_communities` | ours, `coding_agent/tools/prediction_algorithms.py` | structural, early cross-community spread | Weng, Menczer and Ahn, 2014            | 0.211  | the community line                               |

## 5. The tables

**T1, dynamics learning** (one row per task, dynamics and dataset; every run contributes): `delta_f1`, `brier`, persistence change-F1 (zero by construction, the floor), rollout `ens_count_bias`, `ens_marg_mae`, `ens_final_count_model / true`, `action_sensitivity`, and `wm_minus_mc` on the winning strategy. The `structured_oracle` row for the same dynamics is the ceiling. Cascade prediction contributes its hard-target one-step numbers as the falsification test of the IC assumption.

**T2, efficiency** (one row per task and dataset, ordered by size): seconds per rollout sample for the world model and for the oracle referee (`rollout_seconds / n_samples` and `referee_rollout_seconds / referee_samples`), the speedup, `train_seconds` and data-generation seconds, and the break-even rollout count $\lceil (t_{\text{train}} + t_{\text{data}}) / (t_{\text{MC}} - t_{\text{WM}}) \rceil$; then the `evaluator_seconds` and `real_env_episodes` conditions 4 and 6 actually spent in the same search, which is the load-bearing pair now that the agreement replay is off and its `mc_rollout_seconds` column is empty. The size ordering is the point: the speedup should grow with the graph. Adaptive IM adds the per-round re-estimation cost, cascade reconstruction adds `kernel_calls_per_instance`, source localization `forward_calls_per_instance`.

**T3, main results** (one block per task and dynamics; columns are budgets; rows are the five baselines, condition 2, and the two ladder arms; every cell is the oracle referee's `referee_reward` with its standard error): the downstream-utility claim is the `@world_model` row against the baseline rows, with the `@oracle` row as the ceiling (an agent with perfect dynamics), and the baseline rows are the published comparison. The three critical datasets per task fill the main-text table; the two secondary datasets go to the appendix version. A ladder arm that did not finish on a scale row is printed as such with its evaluator budget, never dropped.

**T4, algorithm discovery** (the two hosts of section 9.5, since the per-task discovery loops were removed on 2026-09-20): condition 6 against the three discovery systems at their published budgets, each row carrying `evaluator_calls`, `real_env_episodes`, `evaluator_seconds` and the LLM call count, plus the routed classical pool and the best single baseline for scale. Beside it, the case studies: for the winner of each critical dataset, the write-up's first-attempt-to-winner trajectory, its closest classical algorithm and the parts marked new, with the top twenty programs across all runs in the supplement.

**T5, generalization**: T3 sliced four ways. By dataset and size tier (the five rows per task, medium through very large), by topology (the synthetic families against the real graphs, and the graphs with ground-truth communities, `email_eu_core` with its 42 departments and the `sbm` rows with their planted blocks, for the region diagnostics), by dynamics (IC against LT everywhere; SIR, SIS, SEIR for epidemic control; SIR at $\gamma = 1$ against IC once), and by transfer (one world model per dataset reused across the tasks that share its dynamics, from `scripts/transfer_matrix.py`; a source-localization program moved to another graph with `--sl-transfer-from`; the chronological against the random split for cascade prediction).

## 6. Run order and volume

Critical datasets first, in rank order, so the main-text table fills before the appendix. The first job on every large and very-large row is data generation alone, timed, because nothing above `nethept` has been generated yet; its wall clock decides `--mc-runs`, `--n-samples` and the horizon for that row before any arm runs. Per task that is three datasets, two dynamics (three for epidemic control, one for cascade prediction), all budgets, the five baselines, the ladder, and condition 9 on the critical rows. Rough count: 8 tasks, 5 datasets, an average of 2 dynamics, so about 80 pipeline runs plus the synthetic sweeps, of which 48 are critical (the discovery rows they used to carry moved to section 9.5 on 2026-09-20). The very-large rows (`digg`, `epinions1`, `brightkite`, `deezer`) run last.

Expected cost per pipeline run (one task, one dynamics, four budgets, five classical baselines, the two-arm ladder with the referee replay; no external learned repo, no discovery system), from the measured netscience runs and a Monte Carlo microbenchmark on BA graphs at 1,000, 10,000 and 100,000 nodes: about 4 to 12 hours at 1,000 to 2,000 nodes, 6 to 16 hours at 10,000, and 1 to 3 days at 100,000. The LLM latency (the arm-budget pairs that search, at 10 iterations, 1 to 3 minutes per reply) is the largest block at every size and does not scale with the graph; data generation and training scale linearly with edges (20 to 40 minutes and 1 to 2 hours at 10,000; 3 to 6 hours and 10 to 24 hours at 100,000, where the training batch no longer fits and shrinks); the Monte Carlo evaluator arm costs about an hour per dynamics at 10,000 and 4 to 8 hours at 100,000, where a single evaluation is 90 seconds; a generated program that does its own sampling hits the 300-second per-call cap routinely at 100,000. Outside that table, DeepIM and MOEIM took 10.6 hours between them at netscience and do not get cheaper with size, and each discovery system is LLM-bound at one to two days per budget point. Old source-localization and cascade-reconstruction results predate the label-free rewards and are not comparable; both tasks rerun from scratch.

## 7. Open decisions and things the survey turned up

- `forward_blocking` scored 96.6 against 87.0 for random blocking on email_eu_core. Reproduced on 2026-09-04 with the run's own graph and rumour: the implementation matches its definition (seed the nodes the simulated rumour reaches most often), and those nodes are the rumour's dead-end neighbours (out-degree 0 to 6, against 47 to 81 for random's picks), so they save themselves and nothing else. It is the published heuristic being weak on a dense graph, not a bug; it stays out of the five but is fine to cite.
- On jazz the infected-subgraph centralities beat LPSI by more than double; whether that holds on sparse graphs (cora_ml, power_grid) decides if LPSI stays a reference row or is dropped.
- Cascade prediction's `--cp-max-nodes` cap for `casflow_weibo` and `casflow_aps`: the same cap for every row of a corpus, stated in the caption; the August APS run went out of memory at 64 GB uncapped.
- Whether influence blocking gets a lever table (the three agent arms on the node lever beside `sandimin` and `imin_joc`). The `final_<dynamics>_node` runs carry the two repos only; the table needs the same commands resubmitted with `ARMS="routing evolve_free@oracle evolve_free@world_model"` and the world-model symlink from section 8.5, and the pipeline resumes over the finished repo rows and runs only the arms. T3 stays counter-seed either way.
- The discovery systems' budgets differ from ours by 5 to 50 times by design; T4 reports both at published defaults and states it, rather than matching budgets.
- Whether the write-up prompt should forbid literal node ids in generated programs before the case-study harvest, so the top twenty are graph-general algorithms rather than tuned seed lists.
- The scale rows have never been run. The `@monte_carlo` arm is no longer in the matrix (dropped 2026-09-10), so the C2 efficiency comparison reads the oracle arm's `evaluator_seconds` against the world model's; a sampled-simulator timing row needs a separate `MC_AGREEMENT=1` replay on one critical dataset per task, as section 8.0 describes.
- Whether the epidemic-control DAVA-vs-NetShield reversal on `hospital_lh10` (75 nodes) gets one appendix row now that no small graph is in the main matrix.
- External repos, state as of 2026-09-04. Fixed in the repo and covered by tests or a local end-to-end run: `gdm` and `gdm_epi` (adapter rewritten against the real layout, GDM's own `models/GAT.py` on its published four-feature checkpoint with the 2019 PyG state dict converted and verified numerically; `baselines/gdm_support.py`), `selinda` (same fix through the repo's own `agent/gdm.py`; it reproduces the gdm row exactly and is a cross-check, not coverage), `glie` (its runner now prints a bracketed seed list), `moeim` (duplicate seeds deduplicated in both parsers), `diffim` deleted outright (its surrogate training stalled the sweep; the user's call). Cluster setup only: `setup_baselines --all` rebuilds the venvs for `rl4im` (the `numpy<1.24` patch), `mind`, `nirm`, `dcrs` and `touplegdd`; the `work_dir` group (all CP, CR and SL repos, the six `netimm_*` rows) needs one confirming run after the 2026-08-23 fix. Not fixable here: `finder` needs a CPython 3.7 on the box; `dismantling_review` needs `graph_tool`, which pip cannot install.

## 8. Commands

One `pipeline.sbatch` submission per (task, dataset, dynamics) cell: 80 cells, 10 for each of the six IC/LT simulator tasks, 15 for epidemic control, 5 for cascade prediction, plus a second, baselines-only lever command per influence-blocking cell. Every command is complete and every knob is written out, including the ones that would come from a default, so a command can be read as the caption of the row it produces. Copy it, run it from the repo root on a login node, and it submits itself.

### 8.0 Conventions, and what to add per cluster

**Names.** `RUN` names the dynamics (`final_ic`, `final_lt`, `final_sir`, `final_sis`, `final_seir`) because a run directory holds one dynamics' data, checkpoint and arm results, so IC and LT on the same graph must never share one. `JOB_NAME` is `<task>_<dataset>_<dynamics>` with the task abbreviated (`im`, `aim`, `cnd`, `sl`, `ib`, `cr`, `ec`, `cp`); the script's default would be `gwm_<dataset>_<run>`, which two tasks on one graph would share, and Slurm's `singleton` dependency keys on the name. `RUN_JOBID=0` keeps the directory stable so a resubmission resumes it. **Never pass `FORCE=1`**: every command is idempotent, resubmitting it skips finished stages and finished arms, and an evolve arm resumes from its per-generation checkpoint. `FORCE=1` throws all of that away and, on a requeue, regenerates the data and retrains before reaching the agent stage, which is what cost three of the five attempts on the September digg job.

**Identical in every command**, because they are the protocol rather than a resource: `LLM_MODEL=gpt-6-astra`, `REASONING_EFFORT=high`, `WM_MODEL=sage`, `HEAD=structured` (the one head defined under IC, LT, CLT and all three compartmental dynamics; `structured_residual` raises under LT, so using it anywhere would make the IC and LT rows different models), `EVALUATOR=oracle`, `REFEREE=oracle`, `HORIZON=10`, `OUTER_ITERS=10`, `N_SAMPLES=200`, `MC_RUNS=200`, `SEED=42`, `ACTION_CONDITIONING=message`, `FEEDBACK=default`, and `ARMS`, which is the ladder for the task (condition 2 plus conditions 5 and 6, `routing evolve_free@oracle evolve_free@world_model`; the same three arms on adaptive IM, whose `adaptive_free@<evaluator>` twins were removed on 2026-09-10; the adaptivity gap is therefore not computed by the matrix and needs an ablation cell that adds `adaptive_free@oracle adaptive_free@world_model` back). The native and Monte Carlo arms (conditions 3 and 4) were removed from every command on 2026-09-10; they remain runnable as ablation rows by adding `evolve_free@native evolve_free@monte_carlo` back to `ARMS` on a cell. Also fixed and not written out because no command changes them: learning rate 1e-3, weight decay 5e-4, `pos_weight` off, 3 layers, dropout 0.1, `basic` action encoding, inject 0.4, counterfactual 0.4 with 2 branches, budgets drawn from 1 to 20 percent of $N$ at generation, weighted cascade probabilities, and the 70/15/15 split, which resolves graph-disjoint for a synthetic family and per-episode for a single real graph.

**Also identical, and on by default since 2026-09-10, so no command names them:** the probe turn before every generation (`PROBE_TURN=1`, one extra LLM call per generation, so a 10-generation search now makes about 20 generation-level calls plus reflections and idea searches; the LLM call count in T4 is read from `llm_usage.calls`, not from `OUTER_ITERS`), the paired acceptance band, the incumbent-pinned parent with the paired population ranking, the spurious-accept ledger, the unbiased incumbent curve, the `# EXPECTED:` edit forecast, counterexample realizations in the feedback, and the post-search rediscovery distance (`PROVENANCE=1`, capped at `PROVENANCE_TIMEOUT` seconds per library member, 60 by default; on the very-large rows the slow structural members such as exact betweenness will time out and the report says which). Two knobs stay OFF everywhere because they change the protocol: `STOP_WHEN_FLAT` (early stop on the unbiased curve; the matrix fixes `OUTER_ITERS=10`) and `CALIBRATION_STEERING` (forecast errors feeding the operator schedule). The report's "Search protocol" section and the `edit_calibration` figure read these per arm.

**Two knobs from Hongji's branch, wired through `pipeline.sbatch` on 2026-09-10 and set explicitly in every command below:** `ACTION_CONDITIONING=message` (the action-conditioned message passing; `global` is its locality ablation, `message_blind` the capacity control, `none` the historical model; sage only) and `FEEDBACK=default` (the full feedback; `f0` to `f3` is the controlled ladder of diagnostic blocks, an ablation variable rather than the protocol). Two consequences to keep in view. Under `HEAD=structured` with the true edge weights Hongji's sweep found all four conditioning arms identical to four decimals, so `message` is run for consistency with the method the paper describes, not because it is expected to move the table. And the checkpoint records the conditioning while the train stage refuses to reuse one trained under another value, so any `final_<dyn>` run whose checkpoint was trained before 2026-09-10 under `none` (the September `digg` runs, for example) has to be retrained: either resubmit with `FORCE=1` on the train stage or give the conditioned run a new `RUN` name. A run that skips the train stage (`START_STAGE=agent`, `SKIP_STAGES=train`) never checks, and the agent stage loads whatever conditioning the checkpoint on disk carries.

**Varied per cell**: `TASK`, `DATASET`, `DIFFUSION_MODEL` with `GEN_MODELS` equal to it, `RUN`, `JOB_NAME`, `BASELINES` (section 4's rows, plus the control section 2 says is always run: `frontier_removal` on critical node detection and `frontier_immunization` on epidemic control; `observed_only` and `persistence` are already in their lists), `GEN_ACTION_OPS` (the task's own, empty for the three diffusion-only tasks), the task's protocol knobs at their T3 values, and everything in the tier table below. `MC_AGREEMENT` is 0 on every row, for the reason two paragraphs down.

**The size tiers.** These are cost controls, not protocol, and the table is what each command writes out. Only the first two rows of it are free choices; the rest exist because a row has to fit. `MEM` is 31G on the medium tier and 62G from the large tier up since 2026-09-18. The medium value is measured rather than guessed: a cascade-prediction training set is 1.1 GB resident for taoke's 68,000 transitions and 1.8 GB for digg's 106,000, and taoke's agent stage with the six classical baselines and the referee peaked at 2.3 GB; a large simulator row's host-side terms, transitions times nodes and referee features, come to a few GB at 62,000 nodes. The one host OOM on record at 64G, APS in August, was the per-record adjacency copies fixed on 2026-09-11. 62G is what lets two jobs share finetuner's 125 GB node, and a 31G row can sit beside one of them; if a medium row is ever killed for memory, the manifest marks the stage failed and a resubmit at 62G reuses every finished stage and row. The large rows went back up from 31G to 62G on 2026-09-18 after two cascade-prediction training jobs were killed at 31G at the end of their first epoch: the one-step validation pass holds every record's per-node arrays until it finishes, about 20 GB on a 30,000-node corpus. The very-large rows ask 62G as well: a digg row's own peak is about 15 GB (the training set is 3.9 GB resident at 10 rollouts, so about 8 GB at 20; a rollout block is 2 GB in transit; imm's sample count is capped at 20,000 sets on any graph), and the headroom is for the external repos in their own processes, DeepIM and rl4im, which are unmeasured at this size.

| knob | medium, 1k to 10k | large, 10k to 100k | very large, above 100k | synthetic at 10k |
| --- | --- | --- | --- | --- |
| `CPUS` | 4 | 8 | 8 | 8 |
| `MEM` | 31G | 62G | 62G | 62G |
| `TIME` | 24:00:00 | 48:00:00 | 48:00:00 | 48:00:00 |
| `ROLLOUTS` | 100 | 50 | 20 | 20 (times 40 graphs) |
| `MC_MARGINALS` | 30 | 20 | 20 | 30 |
| `HIDDEN_DIM` | 128 | 256 | 128 | 256 |
| `BATCH_SIZE` | 32 | 8 | 2 | 8 |
| `EPOCHS` | 400 | 200 | 60 | 200 |
| `PATIENCE` | 50 | 25 | 10 | 25 |
| `NO_PLAN_DEMO` | 0 | 1 | 1 | 0 |
| `STRATEGY_TIMEOUT` | 900 | 1800 | 1800 | 1800 |
| `REFEREE_SAMPLES` | 1000 | 1000 | 200 | 1000 |
| `CREDIT` | 1 | 1 | 1 | 1 |

Credit is capped by evaluator cost since 2026-09-12 (`coding_agent/credit.py::credit_limit`, the `best_swap` rule): 128 credited actions on a small graph, 85 on nethept, 4 on digg, evenly spaced by target degree above the cap, and the feedback block says for how many of $k$. Uncapped it was $2k + 1$ batched rollouts per generation, 6,000 at nethept's 20 percent budget and 51 hours per generation at digg's 1 percent; the September digg run had `CREDIT=0`, so this is the first time credit runs on the large rows.
| `BASELINE_TIMEOUT` | 14400 | 21600 | 43200 | 21600 |
| `MC_AGREEMENT` | 0 | 0 | 0 | 0 |

Four of those rows deserve their reason stated, because a reader will otherwise read them as arbitrary. **The very-large column is anchored on a measurement**: the finished `digg` run trained a 128-wide model at batch 2 for 60 epochs and reached one-step `delta_f1` 0.9008 with a Brier score of 0.00127, so that width is a demonstrated floor rather than a hope. It stays 128 rather than rising to the large column's 256 because the structured head's edge MLP materialises an `(E, 2H + 1)` tensor over every arc, so width is the one knob that multiplies the largest allocation on exactly the rows where memory has already forced the batch down to 2, and because the digg diagnosis says capacity is not what is missing. Its `ROLLOUTS` and `MC_MARGINALS` are 20 rather than the 10 that run used, which is the change that addresses what IS missing: that model was excellent one-step and 12.3 percent optimistic over a full rollout (`ens_count_bias` +4,980, final 53,514 against 47,651), which is compounding error and noisy soft targets rather than too few parameters. Ten Monte Carlo draws is a noisy estimate of `P(infected)` to fit against, and 30 episodes is a thin training set. Doubling both costs about an hour of generation and roughly doubles training, which the tier's 48 hours absorbs. **`STRATEGY_TIMEOUT` is 900 or 1800 rather than the 300-second default** because a generated program that does its own sampling routinely blows 300 seconds per call above 10,000 nodes, and a timed-out call is a lost generation rather than a slow one. **`NO_PLAN_DEMO=1` above the medium tier** because the planning demo spends real simulator episodes on five graphs after training to produce one diagnostic figure, which is worth its price on a 2,000-node graph and not on a 60,000-node one. **`REFEREE_SAMPLES=200` only at the very-large tier**: it is a per-arm cost that scales with arcs, its standard error on `digg` is about 27 nodes against arm differences in the hundreds, and the caption says which rows ran reduced.

**`MC_AGREEMENT=0` on every row, and what that gives up.** With `=0` an arm's final number is the shared referee alone: every winner replayed on the exact batched oracle simulator at `REFEREE_SAMPLES` samples, which is the only column that may be read across rows. `=1` would replay each winner a SECOND time on the sampled simulator and add `mc_reward`, `mc_reward_se`, `mc_rollout_seconds` and `referee_minus_mc`. Two things go with it. First, the independent check that the analytic oracle is not quietly wrong, which is answered instead by `coding_agent/check_*.py`, where each head and each simulator is measured against thousands of draws, (the `@monte_carlo` arm, which would have searched on sampled episodes and been scored on the same referee, is no longer in the matrix). Second, T2's per-rollout timing column for the sampled simulator, which `pipeline/plots.py` builds only from `mc_rollout_seconds / mc_agreement_runs`: with the agreement off that column is empty, so the efficiency claim is read from `evaluator_seconds` and `real_env_episodes`, which is what conditions 4 and 6 actually spent inside the same search and is the stronger evidence anyway. If a single sampled-timing row is wanted for the paper, rerun ONE critical dataset per task with `MC_AGREEMENT=1 MC_AGREEMENT_RUNS=50 START_STAGE=agent` rather than paying for it on all 80. Note also that the agreement replay is not NDlib on every task: it is our two-cascade stepper under influence blocking, our compartmental stepper under epidemic control (NDlib carries no per-arc $\beta$, which is the whole reason that stepper exists), and on cascade prediction it is not a replay at all but `model_msle` beside the program's own error.

**`NO_PLAN_DEMO=1` versus `=0`.** `NO_PLAN_DEMO=0` (the default) runs `planning_regret_multi` after training: the model picks a one-step intervention on each of `PLAN_GRAPHS` graphs, the TRUE simulator scores what it picked, and the regret against an oracle choice is reported beside `degree` and `random` references. It is the one training-time number that asks whether the model is useful for choosing rather than merely accurate at predicting, and it feeds `wm_planning_regret.png`. It also costs real simulator episodes on every one of those graphs, which is why `NO_PLAN_DEMO=1` turns it off above the medium tier. Turning it off costs one diagnostic figure and nothing in T1 or T3. Note the separate `PLAN_BUDGET_K`, which is the full $k$-seed horizon version of the same idea and is left at 0 everywhere because it is far more expensive again.

**Synthetic rows.** `SYN_NODES=10000` is the only synthetic knob a command passes. At that size the class table in `pipeline.sbatch` sets 40 graphs (so the split is graph-disjoint), $m = 3$ for `ba`, RL4IM's $m = 3$ and triangle probability $0.05$ for `powerlaw_cluster`, and for `sbm` ten blocks of 1,000 at $p_{\text{in}} = 0.01$ and $p_{\text{out}} = 0.0005$, a mean degree near 15; the 100-node defaults of 0.15 and 0.01 would give a near-complete graph at this size. Cascade reconstruction's `ba` is the exception and passes `NUM_GRAPHS=1 BA_M=3`, DITTO's single-graph protocol, which the pipeline then splits per episode.

**Order of operations on a large or very-large row.** Section 6's rule: the first job on the row is the data stage alone, timed, because nothing above `nethept` has been generated yet. Run the row's command with `START_STAGE=data END_STAGE=data GRES=none` added in front, read the wall clock in the log, then submit the command as written; it finds the data on disk and continues from training. If the data stage alone takes more than a quarter of the row's `TIME`, cut `ROLLOUTS` and `MC_MARGINALS` further for that row and say so in the caption.

**One follow-up per row.** The discovery systems (condition 9) are not run per row: the per-task discovery loops that used to close each subsection were removed on 2026-09-20, and the only discovery runs are section 9.5's, on the two ablation hosts. When every job on a row has finished, rebuild its figures and report from the login node, which takes seconds and needs no GPU:

```bash
python -m pipeline.run --task <task> --dataset <dataset> --run final_<dynamics> --start-stage plots
```

**Conditional rows.** The externals section 4 flags as conditional (`rl4im`, `finder`, `graphsl_slvae`, `ditto`, `grin`) are in the commands. If `python -m baselines.setup_baselines --only <name>` has not produced a working venv on the cluster, the pipeline writes a `.skipped.json` with the reason and the report says the row is absent; nothing else in the job is affected. Influence blocking's two repos (`imin_joc`, `sandimin`) are node-blocking methods and their registry notes say to run them under `--blocking-lever node_block`; under the counter-seed lever their output would be read as positive seeds. Each blocking cell therefore has a second command, `RUN=final_<dynamics>_node`, that runs the two repos alone under `ARMS=none` (no arm needs the world model, so the pipeline skips the train stage itself and nothing in the job touches a GPU); section 8.5 gives the symlink that lets it reuse the counter-seed run's episodes instead of regenerating them.

**Delta** (the commands are written for it). Export once per shell before submitting: `SBATCH_ARGS="--account=<code>-delta-gpu"` (every job needs an account, and `accounts` prints yours; the CPU-only jobs (`GRES=none`) use `--account=<code>-delta-cpu` with `PARTITION=cpu`), `PARTITION="gpuA40x4,gpuA100x4"` (the A40 partition is charged at half the A100 rate and either GPU is enough for every row except the two giant ones: under message conditioning one digg graph needs 53 GB of activations at hidden 128 and one twitter graph 46 GB at hidden 256, so neither fits a 40 GB A100 or a 48 GB A40 and both belong on PDE's 96 GB cards), and `VENV=.venv` if the checkout was set up with `uv`. Keep the checkout under `/work/hdd/<code>/$USER` or `/projects/<code>`, not the 100 GB home. The walltime cap is 48 hours, so for the `TIME=48:00:00` rows that may not finish, submit the identical command four times with `--dependency=singleton` appended to `SBATCH_ARGS`; the copies queue behind each other by `JOB_NAME` and each one resumes from disk. Check once that the LLM gateway is reachable from a compute node (`srun --account=... --partition=cpu-interactive --time=00:05:00 curl -s "$GATEWAY_BASE_URL/models"`), because the agent stage cannot run without it.

**PDE** (the Math department's cluster; what to change instead). Log in with `ssh -J <netid>@lab0z.mathcs.emory.edu <netid>@pdelogin` and keep the checkout in `/local/scratch2/<netid>/GraphWorldModel` (a quota applies, so delete the `data/` directory of any row whose report is final). It is a single node, `pde`, with 8 RTX PRO 6000 GPUs of 96 GB, 80 CPUs and 756 GB of RAM on a PCIe bus, which suits these single-GPU jobs. The GPU count binds first, so at most eight of these run at once, and with `CPUS=8` the 80 cores bind at ten; the `MEM` of whatever is running must also sum to under 756G, so eight 62G jobs fit but only five 128G ones. The card matters on the two giant rows: under message conditioning training costs 103 bytes per arc per hidden unit against 29 without it (measured 2026-09-12; Hongji's modulator keeps its per-arc MLP inputs for backward), so one digg graph at hidden 128 is 53 GB and one twitter graph at hidden 256 is 46 GB where the September `none` runs needed 15 and 13, and the train stage now runs those rows one graph per micro-batch (`MAX_BATCH_ARC_HIDDEN`, gradient accumulated exactly, `BATCH_SIZE` unchanged); batch 2 of digg was 106 GB and batch 8 of twitter 371 GB, which is what the 2026-09-12 CUDA OOMs were. Host RAM is the other constraint: the dataset now holds one adjacency per graph rather than one per record, which is what ended the digg train stage before that. The changes to each command:

**One search per row, transferred across budgets** (`SEARCH_BUDGET`, since 2026-09-14). The two LLM arms search only at the named point, `pct10` on the percentage ladders and `k30` on influence blocking's absolute one; at every other budget the pipeline replans that winner at the new k and replays it on the referee through the canned path a library baseline takes, and the row carries `transferred_from`. Routing and every baseline still run at every budget. The reason is measured, not assumed: on the two finished netscience adaptive-IM runs, fifteen of the sixteen evolved programs replanned at the other three budgets landed within half a node of the winner evolved there, on the same 1,000-sample referee; the one failure was a pct1 winner that did not scale up, which is why the search point is the middle of the ladder rather than its bottom. The transfer cuts each row's gateway-bound block by four; a transferred row's cost columns are the replay's alone, so evaluator-cost comparisons between arms are read at the search point. The report marks transferred rows and the summary CSV carries the column. Set `SEARCH_BUDGET` empty to search at every point again.

**Single GPU without SLURM** (a Lambda instance, or any workstation). The same commands run inline when `SLURM_JOB_ID` is set to anything: put `SLURM_JOB_ID=local` in front of the command and `nohup` or a `parallel -j 3` queue behind it; `CPUS`, `GRES`, `MEM` and `TIME` are read only by the submit path and are ignored. On a 24 GB GPU add `MAX_BATCH_ARC_HIDDEN=67108864 MAX_BLOCK_ARC_HIDDEN=268435456` to every large row, which holds the training micro-batch and the world-model rollout block near 7 GB each so three jobs share one card; both caps change memory only, never the numbers. The scale rows (digg, epinions1, cit_hepth, brightkite, deezer, p2p_gnutella, casflow_aps) stay on an 80 GB GPU.

- **No `--account`**, and therefore no `SBATCH_ARGS` for accounting.
- **Add `PARTITION`**: `pdeday` for the `TIME=24:00:00` rows and `pdeweek` for the `TIME=48:00:00` ones. A shorter partition schedules sooner, so take the shortest that still covers the row. `pdemonth` is the 31-day default and is only for a row that genuinely needs more than a week; `pdehour` is for smoke tests.
- **Add `SBATCH_ARGS="--no-requeue --open-mode=append"`**. On 2026-09-09 the node dropped off the controller five times in a row on one job (`NODE_FAIL`, not a launch failure), and each requeue restarted the batch script from line 1 with the log truncated, until Slurm gave up and left the job in `RH`. With these two flags a failed job simply ends, its log survives, and resubmitting the identical command resumes it from disk.
- **Set `VENV`** to match that checkout: `venv` is the script's default, `.venv` for a `uv` environment.
- **`TIME` can go up rather than down.** With a seven-day partition the Delta singleton chaining is unnecessary; `TIME=7-00:00:00` on `pdeweek` replaces `48:00:00` on the very-large rows, which is what the September digg run used.
- Math's ECM2 group has priority on these GPUs, so expect queue waits to vary and prefer `pdeday` wherever the row fits in a day.

### 8.1 Influence maximization (`influence_maximization`, IC, LT; 10 commands)

Baselines from section 4.1: `imm`, `degree_discount`, `external:opim`, `external:subsim`, `external:deepim`.

**`netscience`** (rank 1, medium, critical).

```bash
TASK=influence_maximization DATASET=netscience RUN=final_ic RUN_JOBID=0 JOB_NAME=im_netscience_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="imm degree_discount external:opim external:subsim external:deepim" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=100 MC_MARGINALS=30 \
HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
CPUS=4 GRES=gpu:1 MEM=31G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=influence_maximization DATASET=netscience RUN=final_lt RUN_JOBID=0 JOB_NAME=im_netscience_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="imm degree_discount external:opim external:subsim external:deepim" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=100 MC_MARGINALS=30 \
HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
CPUS=4 GRES=gpu:1 MEM=31G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`nethept`** (rank 2, large, critical; run the data stage alone first).

```bash
TASK=influence_maximization DATASET=nethept RUN=final_ic RUN_JOBID=0 JOB_NAME=im_nethept_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="imm degree_discount external:opim external:subsim external:deepim" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=influence_maximization DATASET=nethept RUN=final_lt RUN_JOBID=0 JOB_NAME=im_nethept_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="imm degree_discount external:opim external:subsim external:deepim" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`digg`** (rank 3, very large, critical; a PDE row on `pdeweek` with a seven-day walltime, since message conditioning puts each epoch near two hours and early stopping lands around epoch 30; the finished IC run under `RUN=scale_ic` used this tier's knobs without action conditioning and can be reused by passing that run name instead of `final_ic`).

```bash
TASK=influence_maximization DATASET=digg RUN=final_ic RUN_JOBID=0 JOB_NAME=im_digg_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="imm degree_discount external:opim external:subsim external:deepim" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=20 MC_MARGINALS=20 \
HIDDEN_DIM=128 BATCH_SIZE=2 EPOCHS=60 PATIENCE=10 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=200 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=43200 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=7-00:00:00 PARTITION=pdeweek \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=influence_maximization DATASET=digg RUN=final_lt RUN_JOBID=0 JOB_NAME=im_digg_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="imm degree_discount external:opim external:subsim external:deepim" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=20 MC_MARGINALS=20 \
HIDDEN_DIM=128 BATCH_SIZE=2 EPOCHS=60 PATIENCE=10 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=200 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=43200 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=7-00:00:00 PARTITION=pdeweek \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`cora_ml`** (rank 4, medium).

```bash
TASK=influence_maximization DATASET=cora_ml RUN=final_ic RUN_JOBID=0 JOB_NAME=im_cora_ml_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="imm degree_discount external:opim external:subsim external:deepim" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=100 MC_MARGINALS=30 \
HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
CPUS=4 GRES=gpu:1 MEM=31G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=influence_maximization DATASET=cora_ml RUN=final_lt RUN_JOBID=0 JOB_NAME=im_cora_ml_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="imm degree_discount external:opim external:subsim external:deepim" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=100 MC_MARGINALS=30 \
HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
CPUS=4 GRES=gpu:1 MEM=31G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`sbm`** (rank 5, synthetic: 40 SBM graphs of 10,000 nodes in ten blocks).

```bash
TASK=influence_maximization DATASET=sbm RUN=final_ic RUN_JOBID=0 JOB_NAME=im_sbm_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="imm degree_discount external:opim external:subsim external:deepim" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=20 MC_MARGINALS=30 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 SYN_NODES=10000 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=influence_maximization DATASET=sbm RUN=final_lt RUN_JOBID=0 JOB_NAME=im_sbm_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="imm degree_discount external:opim external:subsim external:deepim" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=20 MC_MARGINALS=30 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 SYN_NODES=10000 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

### 8.2 Adaptive online IM (`adaptive_online_im`, IC, LT; 10 commands)

Baselines from section 4.2: `adapt_epic`, `adapt_degree_discount`, `imm`, `static_split`, `external:adaptiveim`, `external:rl4im`.

**`netscience`** (rank 1, medium, critical).

```bash
TASK=adaptive_online_im DATASET=netscience RUN=final_ic RUN_JOBID=0 JOB_NAME=aim_netscience_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="adapt_epic adapt_degree_discount imm static_split external:adaptiveim external:rl4im" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=100 MC_MARGINALS=30 \
HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 ROUNDS=4 ROUND_GAP=1 FEEDBACK_MODEL=full_adoption \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
CPUS=4 GRES=gpu:1 MEM=31G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=adaptive_online_im DATASET=netscience RUN=final_lt RUN_JOBID=0 JOB_NAME=aim_netscience_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="adapt_epic adapt_degree_discount imm static_split external:adaptiveim external:rl4im" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=100 MC_MARGINALS=30 \
HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 ROUNDS=4 ROUND_GAP=1 FEEDBACK_MODEL=full_adoption \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
CPUS=4 GRES=gpu:1 MEM=31G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`nethept`** (rank 2, large, critical; data stage alone first).

```bash
TASK=adaptive_online_im DATASET=nethept RUN=final_ic RUN_JOBID=0 JOB_NAME=aim_nethept_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="adapt_epic adapt_degree_discount imm static_split external:adaptiveim external:rl4im" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 ROUNDS=4 ROUND_GAP=1 FEEDBACK_MODEL=full_adoption \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=adaptive_online_im DATASET=nethept RUN=final_lt RUN_JOBID=0 JOB_NAME=aim_nethept_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="adapt_epic adapt_degree_discount imm static_split external:adaptiveim external:rl4im" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 ROUNDS=4 ROUND_GAP=1 FEEDBACK_MODEL=full_adoption \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`digg`** (rank 3, very large, critical; `REFEREE_SAMPLES=200` like the IM scale row, since a canned row is rolled out at the referee's count and 1,000 samples on this graph is 31 resident graph blocks, which is what OOMed a 96 GB card on 2026-09-17; the same graph, generation knobs and training knobs as the IM scale row, so its `data` and `world_model` directories are the IM run's, copied over with the two path fields in the results JSON rewritten, exactly as section 8.5 does for the node-lever runs; only the agent stage runs. A PDE row on `pdeweek` with a seven-day walltime. Twitter, Han et al.'s own scale benchmark, was the row here until 2026-09-16 and was dropped because its own data and training cost more than the whole row is worth when digg's can be reused; its loader stays).

```bash
TASK=adaptive_online_im DATASET=digg RUN=final_ic RUN_JOBID=0 JOB_NAME=aim_digg_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="adapt_epic adapt_degree_discount imm static_split external:adaptiveim external:rl4im" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=20 MC_MARGINALS=20 \
HIDDEN_DIM=128 BATCH_SIZE=2 EPOCHS=60 PATIENCE=10 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 ROUNDS=4 ROUND_GAP=1 FEEDBACK_MODEL=full_adoption \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=200 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=7-00:00:00 PARTITION=pdeweek \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=adaptive_online_im DATASET=digg RUN=final_lt RUN_JOBID=0 JOB_NAME=aim_digg_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="adapt_epic adapt_degree_discount imm static_split external:adaptiveim external:rl4im" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=20 MC_MARGINALS=20 \
HIDDEN_DIM=128 BATCH_SIZE=2 EPOCHS=60 PATIENCE=10 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 ROUNDS=4 ROUND_GAP=1 FEEDBACK_MODEL=full_adoption \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=200 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=7-00:00:00 PARTITION=pdeweek \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`powerlaw_cluster`** (rank 4, synthetic: 40 graphs of 10,000 nodes at RL4IM's $m = 3$, $p = 0.05$).

```bash
TASK=adaptive_online_im DATASET=powerlaw_cluster RUN=final_ic RUN_JOBID=0 JOB_NAME=aim_powerlaw_cluster_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="adapt_epic adapt_degree_discount imm static_split external:adaptiveim external:rl4im" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=20 MC_MARGINALS=30 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 ROUNDS=4 ROUND_GAP=1 FEEDBACK_MODEL=full_adoption SYN_NODES=10000 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=adaptive_online_im DATASET=powerlaw_cluster RUN=final_lt RUN_JOBID=0 JOB_NAME=aim_powerlaw_cluster_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="adapt_epic adapt_degree_discount imm static_split external:adaptiveim external:rl4im" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=20 MC_MARGINALS=30 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 ROUNDS=4 ROUND_GAP=1 FEEDBACK_MODEL=full_adoption SYN_NODES=10000 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`facebook`** (rank 5, medium).

```bash
TASK=adaptive_online_im DATASET=facebook RUN=final_ic RUN_JOBID=0 JOB_NAME=aim_facebook_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="adapt_epic adapt_degree_discount imm static_split external:adaptiveim external:rl4im" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=100 MC_MARGINALS=30 \
HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 ROUNDS=4 ROUND_GAP=1 FEEDBACK_MODEL=full_adoption \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
CPUS=4 GRES=gpu:1 MEM=31G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=adaptive_online_im DATASET=facebook RUN=final_lt RUN_JOBID=0 JOB_NAME=aim_facebook_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="adapt_epic adapt_degree_discount imm static_split external:adaptiveim external:rl4im" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=100 MC_MARGINALS=30 \
HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 ROUNDS=4 ROUND_GAP=1 FEEDBACK_MODEL=full_adoption \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
CPUS=4 GRES=gpu:1 MEM=31G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

### 8.3 Critical node detection (`critical_node_detection`, IC, LT; 10 commands)

Baselines from section 4.3: `bpd_r`, `collective_influence_r`, `adaptive_degree`, `gndr`, `explosive_immunization`, `frontier_removal`, `external:finder`.

**`power_grid`** (rank 1, medium, critical).

```bash
TASK=critical_node_detection DATASET=power_grid RUN=final_ic RUN_JOBID=0 JOB_NAME=cnd_power_grid_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="bpd_r collective_influence_r adaptive_degree gndr explosive_immunization frontier_removal external:finder" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=100 MC_MARGINALS=30 \
HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 OUTBREAK_PCT=10 OUTBREAK_SELECTOR=random \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
CPUS=4 GRES=gpu:1 MEM=31G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=critical_node_detection DATASET=power_grid RUN=final_lt RUN_JOBID=0 JOB_NAME=cnd_power_grid_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="bpd_r collective_influence_r adaptive_degree gndr explosive_immunization frontier_removal external:finder" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=100 MC_MARGINALS=30 \
HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 OUTBREAK_PCT=10 OUTBREAK_SELECTOR=random \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
CPUS=4 GRES=gpu:1 MEM=31G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`pgp`** (rank 2, large, critical; data stage alone first).

```bash
TASK=critical_node_detection DATASET=pgp RUN=final_ic RUN_JOBID=0 JOB_NAME=cnd_pgp_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="bpd_r collective_influence_r adaptive_degree gndr explosive_immunization frontier_removal external:finder" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 OUTBREAK_PCT=10 OUTBREAK_SELECTOR=random \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=critical_node_detection DATASET=pgp RUN=final_lt RUN_JOBID=0 JOB_NAME=cnd_pgp_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="bpd_r collective_influence_r adaptive_degree gndr explosive_immunization frontier_removal external:finder" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 OUTBREAK_PCT=10 OUTBREAK_SELECTOR=random \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`p2p_gnutella`** (rank 3, large (62,561 nodes), critical; data stage alone first).

```bash
TASK=critical_node_detection DATASET=p2p_gnutella RUN=final_ic RUN_JOBID=0 JOB_NAME=cnd_p2p_gnutella_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="bpd_r collective_influence_r adaptive_degree gndr explosive_immunization frontier_removal external:finder" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 OUTBREAK_PCT=10 OUTBREAK_SELECTOR=random \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=critical_node_detection DATASET=p2p_gnutella RUN=final_lt RUN_JOBID=0 JOB_NAME=cnd_p2p_gnutella_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="bpd_r collective_influence_r adaptive_degree gndr explosive_immunization frontier_removal external:finder" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 OUTBREAK_PCT=10 OUTBREAK_SELECTOR=random \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`openflights`** (rank 4, medium).

```bash
TASK=critical_node_detection DATASET=openflights RUN=final_ic RUN_JOBID=0 JOB_NAME=cnd_openflights_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="bpd_r collective_influence_r adaptive_degree gndr explosive_immunization frontier_removal external:finder" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=100 MC_MARGINALS=30 \
HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 OUTBREAK_PCT=10 OUTBREAK_SELECTOR=random \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
CPUS=4 GRES=gpu:1 MEM=31G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=critical_node_detection DATASET=openflights RUN=final_lt RUN_JOBID=0 JOB_NAME=cnd_openflights_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="bpd_r collective_influence_r adaptive_degree gndr explosive_immunization frontier_removal external:finder" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=100 MC_MARGINALS=30 \
HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 OUTBREAK_PCT=10 OUTBREAK_SELECTOR=random \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
CPUS=4 GRES=gpu:1 MEM=31G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`ba`** (rank 5, synthetic: 40 BA graphs of 10,000 nodes, $m = 3$).

```bash
TASK=critical_node_detection DATASET=ba RUN=final_ic RUN_JOBID=0 JOB_NAME=cnd_ba_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="bpd_r collective_influence_r adaptive_degree gndr explosive_immunization frontier_removal external:finder" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=20 MC_MARGINALS=30 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 OUTBREAK_PCT=10 OUTBREAK_SELECTOR=random SYN_NODES=10000 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=critical_node_detection DATASET=ba RUN=final_lt RUN_JOBID=0 JOB_NAME=cnd_ba_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="bpd_r collective_influence_r adaptive_degree gndr explosive_immunization frontier_removal external:finder" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
ROLLOUTS=20 MC_MARGINALS=30 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 OUTBREAK_PCT=10 OUTBREAK_SELECTOR=random SYN_NODES=10000 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

### 8.4 Source localization (`source_localization`, IC, LT; 10 commands)

Baselines from section 4.4: `infected_betweenness`, `infected_degree`, `dynamic_age`, `lpsi`, `rumor_centrality`, `external:graphsl_slvae`.

`GEN_ACTION_OPS` is deliberately EMPTY. This task's episodes are diffusion-only, and the registry asserts it: an episode carrying a mid-cascade injection would have an observation its seed set did not produce, so its labelled pair would be a lie.

**`cora_ml`** (rank 1, medium, critical).

```bash
TASK=source_localization DATASET=cora_ml RUN=final_ic RUN_JOBID=0 JOB_NAME=sl_cora_ml_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="infected_betweenness infected_degree dynamic_age lpsi rumor_centrality external:graphsl_slvae" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="" \
ROLLOUTS=100 MC_MARGINALS=30 \
HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="10" SL_OBSERVATION=binary SL_BUDGET_MODE=episode SL_SELECT_SPLIT=train SL_EVAL_SPLIT=test \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
CPUS=4 GRES=gpu:1 MEM=31G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=source_localization DATASET=cora_ml RUN=final_lt RUN_JOBID=0 JOB_NAME=sl_cora_ml_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="infected_betweenness infected_degree dynamic_age lpsi rumor_centrality external:graphsl_slvae" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="" \
ROLLOUTS=100 MC_MARGINALS=30 \
HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="10" SL_OBSERVATION=binary SL_BUDGET_MODE=episode SL_SELECT_SPLIT=train SL_EVAL_SPLIT=test \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
CPUS=4 GRES=gpu:1 MEM=31G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`power_grid`** (rank 2, medium, critical).

```bash
TASK=source_localization DATASET=power_grid RUN=final_ic RUN_JOBID=0 JOB_NAME=sl_power_grid_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="infected_betweenness infected_degree dynamic_age lpsi rumor_centrality external:graphsl_slvae" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="" \
ROLLOUTS=100 MC_MARGINALS=30 \
HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="10" SL_OBSERVATION=binary SL_BUDGET_MODE=episode SL_SELECT_SPLIT=train SL_EVAL_SPLIT=test \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
CPUS=4 GRES=gpu:1 MEM=31G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=source_localization DATASET=power_grid RUN=final_lt RUN_JOBID=0 JOB_NAME=sl_power_grid_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="infected_betweenness infected_degree dynamic_age lpsi rumor_centrality external:graphsl_slvae" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="" \
ROLLOUTS=100 MC_MARGINALS=30 \
HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="10" SL_OBSERVATION=binary SL_BUDGET_MODE=episode SL_SELECT_SPLIT=train SL_EVAL_SPLIT=test \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
CPUS=4 GRES=gpu:1 MEM=31G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`deezer`** (rank 3, large (47,538 nodes), critical; data stage alone first).

```bash
TASK=source_localization DATASET=deezer RUN=final_ic RUN_JOBID=0 JOB_NAME=sl_deezer_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="infected_betweenness infected_degree dynamic_age lpsi rumor_centrality external:graphsl_slvae" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="10" SL_OBSERVATION=binary SL_BUDGET_MODE=episode SL_SELECT_SPLIT=train SL_EVAL_SPLIT=test \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=source_localization DATASET=deezer RUN=final_lt RUN_JOBID=0 JOB_NAME=sl_deezer_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="infected_betweenness infected_degree dynamic_age lpsi rumor_centrality external:graphsl_slvae" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="10" SL_OBSERVATION=binary SL_BUDGET_MODE=episode SL_SELECT_SPLIT=train SL_EVAL_SPLIT=test \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`netscience`** (rank 4, medium).

```bash
TASK=source_localization DATASET=netscience RUN=final_ic RUN_JOBID=0 JOB_NAME=sl_netscience_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="infected_betweenness infected_degree dynamic_age lpsi rumor_centrality external:graphsl_slvae" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="" \
ROLLOUTS=100 MC_MARGINALS=30 \
HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="10" SL_OBSERVATION=binary SL_BUDGET_MODE=episode SL_SELECT_SPLIT=train SL_EVAL_SPLIT=test \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
CPUS=4 GRES=gpu:1 MEM=31G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=source_localization DATASET=netscience RUN=final_lt RUN_JOBID=0 JOB_NAME=sl_netscience_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="infected_betweenness infected_degree dynamic_age lpsi rumor_centrality external:graphsl_slvae" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="" \
ROLLOUTS=100 MC_MARGINALS=30 \
HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="10" SL_OBSERVATION=binary SL_BUDGET_MODE=episode SL_SELECT_SPLIT=train SL_EVAL_SPLIT=test \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
CPUS=4 GRES=gpu:1 MEM=31G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`sbm`** (rank 5, synthetic: 40 SBM graphs of 10,000 nodes in ten blocks).

```bash
TASK=source_localization DATASET=sbm RUN=final_ic RUN_JOBID=0 JOB_NAME=sl_sbm_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="infected_betweenness infected_degree dynamic_age lpsi rumor_centrality external:graphsl_slvae" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="" \
ROLLOUTS=20 MC_MARGINALS=30 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="10" SL_OBSERVATION=binary SL_BUDGET_MODE=episode SL_SELECT_SPLIT=train SL_EVAL_SPLIT=test SYN_NODES=10000 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=source_localization DATASET=sbm RUN=final_lt RUN_JOBID=0 JOB_NAME=sl_sbm_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="infected_betweenness infected_degree dynamic_age lpsi rumor_centrality external:graphsl_slvae" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="" \
ROLLOUTS=20 MC_MARGINALS=30 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="10" SL_OBSERVATION=binary SL_BUDGET_MODE=episode SL_SELECT_SPLIT=train SL_EVAL_SPLIT=test SYN_NODES=10000 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

### 8.5 Influence blocking (`influence_blocking`, IC, LT; 20 commands: 10 counter-seed and 10 node-lever)

Baselines from section 4.5: `rps`, `reverse_blocking`, `proximity`, `cldag`.

Each dataset has FOUR commands: the counter-seed ladder per dynamics, which is the T3 row, and the node-lever pair that runs section 4.5's other two repos alone. The node-lever commands set `ARMS=none`: no agent arm, so the pipeline skips the train stage on its own, the agent stage is the two repos and their referee replays, and the job asks for no GPU. The training knobs still written in those commands are inert and stay only so the command reads as a complete caption. A lever changes only what the planner may emit and what one unit of budget buys; the generator injects the same four ops whatever the lever, so the node-lever run's episodes are identical to the counter-seed run's and should be shared rather than rebuilt. Do that once per dataset and dynamics, after the counter-seed run's data stage has finished and before submitting the node-lever command. The data stage then prints `reusing` and nothing is written into the shared directory:

```bash
d=results/influence_blocking/<dataset>
for dyn in ic lt; do
  mkdir -p "$d/final_${dyn}_node"
  ln -s ../final_${dyn}/data "$d/final_${dyn}_node/data"
done
```

Without the symlink the node-lever command regenerates the episodes from scratch, which on `epinions1` and `cit_hepth` is hours spent twice for byte-identical files.

The three agent arms are deliberately absent from the node-lever runs. On that lever the agent writes deletion programs, a different action space from the T3 row rather than a repeat of it, so those rows would be a separate table (the agent on SandIMIN's and IMIN's own problem) that section 7 has not decided to include. If it is wanted, resubmit the same node-lever command with `ARMS="routing evolve_free@oracle evolve_free@world_model"`, `GRES=gpu:1`, and a second symlink `ln -s ../final_${dyn}/world_model "$d/final_${dyn}_node/world_model"` made after the counter-seed run's training has finished; the pipeline resumes over the finished repo rows and runs only the arms.

**`email_eu_core`** (rank 1, medium, critical).

```bash
TASK=influence_blocking DATASET=email_eu_core RUN=final_ic RUN_JOBID=0 JOB_NAME=ib_email_eu_core_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="rps reverse_blocking proximity cldag" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node remove_edge set_edge_weight" \
ROLLOUTS=100 MC_MARGINALS=30 \
HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGETS="10 20 30 40 50" SEARCH_BUDGET=k30 BLOCKING_LEVER=counter_seed OUTBREAK_PCT=1 TIE_BREAK=auto \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
CPUS=4 GRES=gpu:1 MEM=31G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=influence_blocking DATASET=email_eu_core RUN=final_lt RUN_JOBID=0 JOB_NAME=ib_email_eu_core_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="rps reverse_blocking proximity cldag" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node remove_edge set_edge_weight" \
ROLLOUTS=100 MC_MARGINALS=30 \
HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGETS="10 20 30 40 50" SEARCH_BUDGET=k30 BLOCKING_LEVER=counter_seed OUTBREAK_PCT=1 TIE_BREAK=auto \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
CPUS=4 GRES=gpu:1 MEM=31G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=influence_blocking DATASET=email_eu_core RUN=final_ic_node RUN_JOBID=0 JOB_NAME=ib_email_eu_core_ic_node \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="external:imin_joc external:sandimin" \
ARMS=none \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node remove_edge set_edge_weight" \
ROLLOUTS=100 MC_MARGINALS=30 \
HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGETS="10 20 30 40 50" SEARCH_BUDGET=k30 BLOCKING_LEVER=node_block OUTBREAK_PCT=1 TIE_BREAK=auto \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
CPUS=4 GRES=none MEM=31G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=influence_blocking DATASET=email_eu_core RUN=final_lt_node RUN_JOBID=0 JOB_NAME=ib_email_eu_core_lt_node \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="external:imin_joc external:sandimin" \
ARMS=none \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node remove_edge set_edge_weight" \
ROLLOUTS=100 MC_MARGINALS=30 \
HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGETS="10 20 30 40 50" SEARCH_BUDGET=k30 BLOCKING_LEVER=node_block OUTBREAK_PCT=1 TIE_BREAK=auto \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
CPUS=4 GRES=none MEM=31G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`p2p_gnutella24`** (rank 2, large, critical; data stage alone first).

```bash
TASK=influence_blocking DATASET=p2p_gnutella24 RUN=final_ic RUN_JOBID=0 JOB_NAME=ib_p2p_gnutella24_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="rps reverse_blocking proximity cldag" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node remove_edge set_edge_weight" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGETS="10 20 30 40 50" SEARCH_BUDGET=k30 BLOCKING_LEVER=counter_seed OUTBREAK_PCT=1 TIE_BREAK=auto \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=influence_blocking DATASET=p2p_gnutella24 RUN=final_lt RUN_JOBID=0 JOB_NAME=ib_p2p_gnutella24_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="rps reverse_blocking proximity cldag" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node remove_edge set_edge_weight" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGETS="10 20 30 40 50" SEARCH_BUDGET=k30 BLOCKING_LEVER=counter_seed OUTBREAK_PCT=1 TIE_BREAK=auto \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=influence_blocking DATASET=p2p_gnutella24 RUN=final_ic_node RUN_JOBID=0 JOB_NAME=ib_p2p_gnutella24_ic_node \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="external:imin_joc external:sandimin" \
ARMS=none \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node remove_edge set_edge_weight" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGETS="10 20 30 40 50" SEARCH_BUDGET=k30 BLOCKING_LEVER=node_block OUTBREAK_PCT=1 TIE_BREAK=auto \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=none MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=influence_blocking DATASET=p2p_gnutella24 RUN=final_lt_node RUN_JOBID=0 JOB_NAME=ib_p2p_gnutella24_lt_node \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="external:imin_joc external:sandimin" \
ARMS=none \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node remove_edge set_edge_weight" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGETS="10 20 30 40 50" SEARCH_BUDGET=k30 BLOCKING_LEVER=node_block OUTBREAK_PCT=1 TIE_BREAK=auto \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=none MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`epinions1`** (rank 3, large (75,879 nodes, 508,837 arcs), critical; data stage alone first).

```bash
TASK=influence_blocking DATASET=epinions1 RUN=final_ic RUN_JOBID=0 JOB_NAME=ib_epinions1_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="rps reverse_blocking proximity cldag" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node remove_edge set_edge_weight" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGETS="10 20 30 40 50" SEARCH_BUDGET=k30 BLOCKING_LEVER=counter_seed OUTBREAK_PCT=1 TIE_BREAK=auto \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=influence_blocking DATASET=epinions1 RUN=final_lt RUN_JOBID=0 JOB_NAME=ib_epinions1_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="rps reverse_blocking proximity cldag" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node remove_edge set_edge_weight" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGETS="10 20 30 40 50" SEARCH_BUDGET=k30 BLOCKING_LEVER=counter_seed OUTBREAK_PCT=1 TIE_BREAK=auto \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=influence_blocking DATASET=epinions1 RUN=final_ic_node RUN_JOBID=0 JOB_NAME=ib_epinions1_ic_node \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="external:imin_joc external:sandimin" \
ARMS=none \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node remove_edge set_edge_weight" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGETS="10 20 30 40 50" SEARCH_BUDGET=k30 BLOCKING_LEVER=node_block OUTBREAK_PCT=1 TIE_BREAK=auto \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=none MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=influence_blocking DATASET=epinions1 RUN=final_lt_node RUN_JOBID=0 JOB_NAME=ib_epinions1_lt_node \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="external:imin_joc external:sandimin" \
ARMS=none \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node remove_edge set_edge_weight" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGETS="10 20 30 40 50" SEARCH_BUDGET=k30 BLOCKING_LEVER=node_block OUTBREAK_PCT=1 TIE_BREAK=auto \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=none MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`cit_hepth`** (rank 4, large (352,768 arcs); data stage alone first).

```bash
TASK=influence_blocking DATASET=cit_hepth RUN=final_ic RUN_JOBID=0 JOB_NAME=ib_cit_hepth_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="rps reverse_blocking proximity cldag" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node remove_edge set_edge_weight" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGETS="10 20 30 40 50" SEARCH_BUDGET=k30 BLOCKING_LEVER=counter_seed OUTBREAK_PCT=1 TIE_BREAK=auto \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=influence_blocking DATASET=cit_hepth RUN=final_lt RUN_JOBID=0 JOB_NAME=ib_cit_hepth_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="rps reverse_blocking proximity cldag" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node remove_edge set_edge_weight" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGETS="10 20 30 40 50" SEARCH_BUDGET=k30 BLOCKING_LEVER=counter_seed OUTBREAK_PCT=1 TIE_BREAK=auto \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=influence_blocking DATASET=cit_hepth RUN=final_ic_node RUN_JOBID=0 JOB_NAME=ib_cit_hepth_ic_node \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="external:imin_joc external:sandimin" \
ARMS=none \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node remove_edge set_edge_weight" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGETS="10 20 30 40 50" SEARCH_BUDGET=k30 BLOCKING_LEVER=node_block OUTBREAK_PCT=1 TIE_BREAK=auto \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=none MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=influence_blocking DATASET=cit_hepth RUN=final_lt_node RUN_JOBID=0 JOB_NAME=ib_cit_hepth_lt_node \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="external:imin_joc external:sandimin" \
ARMS=none \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node remove_edge set_edge_weight" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGETS="10 20 30 40 50" SEARCH_BUDGET=k30 BLOCKING_LEVER=node_block OUTBREAK_PCT=1 TIE_BREAK=auto \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=none MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`powerlaw_cluster`** (rank 5, synthetic: 40 graphs of 10,000 nodes at $m = 3$, $p = 0.05$).

```bash
TASK=influence_blocking DATASET=powerlaw_cluster RUN=final_ic RUN_JOBID=0 JOB_NAME=ib_powerlaw_cluster_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="rps reverse_blocking proximity cldag" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node remove_edge set_edge_weight" \
ROLLOUTS=20 MC_MARGINALS=30 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGETS="10 20 30 40 50" SEARCH_BUDGET=k30 BLOCKING_LEVER=counter_seed OUTBREAK_PCT=1 TIE_BREAK=auto SYN_NODES=10000 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=influence_blocking DATASET=powerlaw_cluster RUN=final_lt RUN_JOBID=0 JOB_NAME=ib_powerlaw_cluster_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="rps reverse_blocking proximity cldag" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node remove_edge set_edge_weight" \
ROLLOUTS=20 MC_MARGINALS=30 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGETS="10 20 30 40 50" SEARCH_BUDGET=k30 BLOCKING_LEVER=counter_seed OUTBREAK_PCT=1 TIE_BREAK=auto SYN_NODES=10000 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=influence_blocking DATASET=powerlaw_cluster RUN=final_ic_node RUN_JOBID=0 JOB_NAME=ib_powerlaw_cluster_ic_node \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="external:imin_joc external:sandimin" \
ARMS=none \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node remove_edge set_edge_weight" \
ROLLOUTS=20 MC_MARGINALS=30 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGETS="10 20 30 40 50" SEARCH_BUDGET=k30 BLOCKING_LEVER=node_block OUTBREAK_PCT=1 TIE_BREAK=auto SYN_NODES=10000 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=none MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=influence_blocking DATASET=powerlaw_cluster RUN=final_lt_node RUN_JOBID=0 JOB_NAME=ib_powerlaw_cluster_lt_node \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="external:imin_joc external:sandimin" \
ARMS=none \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node remove_edge set_edge_weight" \
ROLLOUTS=20 MC_MARGINALS=30 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGETS="10 20 30 40 50" SEARCH_BUDGET=k30 BLOCKING_LEVER=node_block OUTBREAK_PCT=1 TIE_BREAK=auto SYN_NODES=10000 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=none MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

### 8.6 Cascade reconstruction (`cascade_reconstruction`, IC, LT; 10 commands)

Baselines from section 4.6: `steiner_tree`, `jordan_backward`, `cri`, `dhrec`, `observed_only`, `external:ditto`, `external:grin`.

`GEN_ACTION_OPS` is deliberately EMPTY. This task's episodes are diffusion-only, and the registry asserts it: an episode carrying a mid-cascade injection would have an observation its seed set did not produce, so its labelled pair would be a lie.

**Before submitting.** The two external repos in the baseline list need their venvs built once per cluster, on a login node:

```bash
python -m baselines.setup_baselines --only ditto grin
```

Without them the job still runs: the agent stage catches the missing venv, writes a `.skipped.json` with the reason, and the report says the two rows are absent. The four real graphs download themselves inside the data stage and `ba` is synthetic. If the compute nodes have no outbound network, fetch the raw files once from a login node first; each call is a no-op when the file is already there:

```bash
python -c "from data.datasets.uci_students import download_uci_students as d; d()"
python -c "from data.datasets.ca_grqc import download_ca_grqc as d; d()"
python -c "from data.datasets.rt_pol import download_rt_pol as d; d()"
python -c "from data.datasets.oregon2 import download_oregon2 as d; d()"
```

**`uci_students`** (rank 1, medium, critical).

```bash
TASK=cascade_reconstruction DATASET=uci_students RUN=final_ic RUN_JOBID=0 JOB_NAME=cr_uci_students_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="steiner_tree jordan_backward cri dhrec observed_only external:ditto external:grin" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="" \
ROLLOUTS=100 MC_MARGINALS=30 \
HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="10" CR_SETTING=final_snapshot CR_OBSERVATION_RATE=0.3 CR_SELECT_SPLIT=train CR_EVAL_SPLIT=test \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
CPUS=4 GRES=gpu:1 MEM=31G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=cascade_reconstruction DATASET=uci_students RUN=final_lt RUN_JOBID=0 JOB_NAME=cr_uci_students_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="steiner_tree jordan_backward cri dhrec observed_only external:ditto external:grin" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="" \
ROLLOUTS=100 MC_MARGINALS=30 \
HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="10" CR_SETTING=final_snapshot CR_OBSERVATION_RATE=0.3 CR_SELECT_SPLIT=train CR_EVAL_SPLIT=test \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
CPUS=4 GRES=gpu:1 MEM=31G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`ca_grqc`** (rank 2, medium, critical).

```bash
TASK=cascade_reconstruction DATASET=ca_grqc RUN=final_ic RUN_JOBID=0 JOB_NAME=cr_ca_grqc_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="steiner_tree jordan_backward cri dhrec observed_only external:ditto external:grin" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="" \
ROLLOUTS=100 MC_MARGINALS=30 \
HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="10" CR_SETTING=final_snapshot CR_OBSERVATION_RATE=0.3 CR_SELECT_SPLIT=train CR_EVAL_SPLIT=test \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
CPUS=4 GRES=gpu:1 MEM=31G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=cascade_reconstruction DATASET=ca_grqc RUN=final_lt RUN_JOBID=0 JOB_NAME=cr_ca_grqc_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="steiner_tree jordan_backward cri dhrec observed_only external:ditto external:grin" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="" \
ROLLOUTS=100 MC_MARGINALS=30 \
HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="10" CR_SETTING=final_snapshot CR_OBSERVATION_RATE=0.3 CR_SELECT_SPLIT=train CR_EVAL_SPLIT=test \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
CPUS=4 GRES=gpu:1 MEM=31G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`rt_pol`** (rank 3, large, critical; data stage alone first, and the arm 4 completion question lives here).

```bash
TASK=cascade_reconstruction DATASET=rt_pol RUN=final_ic RUN_JOBID=0 JOB_NAME=cr_rt_pol_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="steiner_tree jordan_backward cri dhrec observed_only external:ditto external:grin" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="10" CR_SETTING=final_snapshot CR_OBSERVATION_RATE=0.3 CR_SELECT_SPLIT=train CR_EVAL_SPLIT=test \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=cascade_reconstruction DATASET=rt_pol RUN=final_lt RUN_JOBID=0 JOB_NAME=cr_rt_pol_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="steiner_tree jordan_backward cri dhrec observed_only external:ditto external:grin" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="10" CR_SETTING=final_snapshot CR_OBSERVATION_RATE=0.3 CR_SELECT_SPLIT=train CR_EVAL_SPLIT=test \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`oregon2`** (rank 4, large; data stage alone first).

```bash
TASK=cascade_reconstruction DATASET=oregon2 RUN=final_ic RUN_JOBID=0 JOB_NAME=cr_oregon2_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="steiner_tree jordan_backward cri dhrec observed_only external:ditto external:grin" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="10" CR_SETTING=final_snapshot CR_OBSERVATION_RATE=0.3 CR_SELECT_SPLIT=train CR_EVAL_SPLIT=test \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=cascade_reconstruction DATASET=oregon2 RUN=final_lt RUN_JOBID=0 JOB_NAME=cr_oregon2_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="steiner_tree jordan_backward cri dhrec observed_only external:ditto external:grin" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="10" CR_SETTING=final_snapshot CR_OBSERVATION_RATE=0.3 CR_SELECT_SPLIT=train CR_EVAL_SPLIT=test \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`ba`** (rank 5, synthetic at DITTO's single-graph protocol: one BA graph of 10,000 nodes, $m = 3$, per-episode split).

```bash
TASK=cascade_reconstruction DATASET=ba RUN=final_ic RUN_JOBID=0 JOB_NAME=cr_ba_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="steiner_tree jordan_backward cri dhrec observed_only external:ditto external:grin" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="" \
ROLLOUTS=20 MC_MARGINALS=30 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="10" CR_SETTING=final_snapshot CR_OBSERVATION_RATE=0.3 CR_SELECT_SPLIT=train CR_EVAL_SPLIT=test SYN_NODES=10000 NUM_GRAPHS=1 BA_M=3 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=cascade_reconstruction DATASET=ba RUN=final_lt RUN_JOBID=0 JOB_NAME=cr_ba_lt \
DIFFUSION_MODEL=LT GEN_MODELS=LT \
BASELINES="steiner_tree jordan_backward cri dhrec observed_only external:ditto external:grin" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="" \
ROLLOUTS=20 MC_MARGINALS=30 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="10" CR_SETTING=final_snapshot CR_OBSERVATION_RATE=0.3 CR_SELECT_SPLIT=train CR_EVAL_SPLIT=test SYN_NODES=10000 NUM_GRAPHS=1 BA_M=3 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

### 8.7 Epidemic control (`epidemic_control`, SIR, SIS, SEIR; 15 commands)

Baselines from section 4.7: `dava`, `netshield_plus`, `greedy_walk`, `degree_immunization`, `frontier_immunization`, `external:explosive_immunization_epi`, `external:collective_influence_epi`.

**`infectious_sociopatterns`** (rank 1, large (10,972 nodes), critical; data stage alone first).

```bash
TASK=epidemic_control DATASET=infectious_sociopatterns RUN=final_sir RUN_JOBID=0 JOB_NAME=ec_infectious_sociopatterns_sir \
DIFFUSION_MODEL=SIR GEN_MODELS=SIR \
BASELINES="dava netshield_plus greedy_walk degree_immunization frontier_immunization external:explosive_immunization_epi external:collective_influence_epi" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="remove_node remove_edge set_edge_weight" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 EPI_LEVER=vaccinate EPI_BETA=1.0 EPI_GAMMA=0.3 EPI_ALPHA=0.5 OUTBREAK_PCT=1 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=epidemic_control DATASET=infectious_sociopatterns RUN=final_sis RUN_JOBID=0 JOB_NAME=ec_infectious_sociopatterns_sis \
DIFFUSION_MODEL=SIS GEN_MODELS=SIS \
BASELINES="dava netshield_plus greedy_walk degree_immunization frontier_immunization external:explosive_immunization_epi external:collective_influence_epi" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="remove_node remove_edge set_edge_weight" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 EPI_LEVER=vaccinate EPI_BETA=1.0 EPI_GAMMA=0.3 EPI_ALPHA=0.5 OUTBREAK_PCT=1 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=epidemic_control DATASET=infectious_sociopatterns RUN=final_seir RUN_JOBID=0 JOB_NAME=ec_infectious_sociopatterns_seir \
DIFFUSION_MODEL=SEIR GEN_MODELS=SEIR \
BASELINES="dava netshield_plus greedy_walk degree_immunization frontier_immunization external:explosive_immunization_epi external:collective_influence_epi" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="remove_node remove_edge set_edge_weight" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 EPI_LEVER=vaccinate EPI_BETA=1.0 EPI_GAMMA=0.3 EPI_ALPHA=0.5 OUTBREAK_PCT=1 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`oregon1`** (rank 2, large, critical; data stage alone first).

```bash
TASK=epidemic_control DATASET=oregon1 RUN=final_sir RUN_JOBID=0 JOB_NAME=ec_oregon1_sir \
DIFFUSION_MODEL=SIR GEN_MODELS=SIR \
BASELINES="dava netshield_plus greedy_walk degree_immunization frontier_immunization external:explosive_immunization_epi external:collective_influence_epi" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="remove_node remove_edge set_edge_weight" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 EPI_LEVER=vaccinate EPI_BETA=1.0 EPI_GAMMA=0.3 EPI_ALPHA=0.5 OUTBREAK_PCT=1 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=epidemic_control DATASET=oregon1 RUN=final_sis RUN_JOBID=0 JOB_NAME=ec_oregon1_sis \
DIFFUSION_MODEL=SIS GEN_MODELS=SIS \
BASELINES="dava netshield_plus greedy_walk degree_immunization frontier_immunization external:explosive_immunization_epi external:collective_influence_epi" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="remove_node remove_edge set_edge_weight" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 EPI_LEVER=vaccinate EPI_BETA=1.0 EPI_GAMMA=0.3 EPI_ALPHA=0.5 OUTBREAK_PCT=1 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=epidemic_control DATASET=oregon1 RUN=final_seir RUN_JOBID=0 JOB_NAME=ec_oregon1_seir \
DIFFUSION_MODEL=SEIR GEN_MODELS=SEIR \
BASELINES="dava netshield_plus greedy_walk degree_immunization frontier_immunization external:explosive_immunization_epi external:collective_influence_epi" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="remove_node remove_edge set_edge_weight" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 EPI_LEVER=vaccinate EPI_BETA=1.0 EPI_GAMMA=0.3 EPI_ALPHA=0.5 OUTBREAK_PCT=1 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`brightkite`** (rank 3, large (58,228 nodes), critical; data stage alone first).

```bash
TASK=epidemic_control DATASET=brightkite RUN=final_sir RUN_JOBID=0 JOB_NAME=ec_brightkite_sir \
DIFFUSION_MODEL=SIR GEN_MODELS=SIR \
BASELINES="dava netshield_plus greedy_walk degree_immunization frontier_immunization external:explosive_immunization_epi external:collective_influence_epi" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="remove_node remove_edge set_edge_weight" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 EPI_LEVER=vaccinate EPI_BETA=1.0 EPI_GAMMA=0.3 EPI_ALPHA=0.5 OUTBREAK_PCT=1 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=epidemic_control DATASET=brightkite RUN=final_sis RUN_JOBID=0 JOB_NAME=ec_brightkite_sis \
DIFFUSION_MODEL=SIS GEN_MODELS=SIS \
BASELINES="dava netshield_plus greedy_walk degree_immunization frontier_immunization external:explosive_immunization_epi external:collective_influence_epi" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="remove_node remove_edge set_edge_weight" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 EPI_LEVER=vaccinate EPI_BETA=1.0 EPI_GAMMA=0.3 EPI_ALPHA=0.5 OUTBREAK_PCT=1 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=epidemic_control DATASET=brightkite RUN=final_seir RUN_JOBID=0 JOB_NAME=ec_brightkite_seir \
DIFFUSION_MODEL=SEIR GEN_MODELS=SEIR \
BASELINES="dava netshield_plus greedy_walk degree_immunization frontier_immunization external:explosive_immunization_epi external:collective_influence_epi" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="remove_node remove_edge set_edge_weight" \
ROLLOUTS=50 MC_MARGINALS=20 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 EPI_LEVER=vaccinate EPI_BETA=1.0 EPI_GAMMA=0.3 EPI_ALPHA=0.5 OUTBREAK_PCT=1 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`p2p_gnutella05`** (rank 4, medium).

```bash
TASK=epidemic_control DATASET=p2p_gnutella05 RUN=final_sir RUN_JOBID=0 JOB_NAME=ec_p2p_gnutella05_sir \
DIFFUSION_MODEL=SIR GEN_MODELS=SIR \
BASELINES="dava netshield_plus greedy_walk degree_immunization frontier_immunization external:explosive_immunization_epi external:collective_influence_epi" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="remove_node remove_edge set_edge_weight" \
ROLLOUTS=100 MC_MARGINALS=30 \
HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 EPI_LEVER=vaccinate EPI_BETA=1.0 EPI_GAMMA=0.3 EPI_ALPHA=0.5 OUTBREAK_PCT=1 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
CPUS=4 GRES=gpu:1 MEM=31G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=epidemic_control DATASET=p2p_gnutella05 RUN=final_sis RUN_JOBID=0 JOB_NAME=ec_p2p_gnutella05_sis \
DIFFUSION_MODEL=SIS GEN_MODELS=SIS \
BASELINES="dava netshield_plus greedy_walk degree_immunization frontier_immunization external:explosive_immunization_epi external:collective_influence_epi" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="remove_node remove_edge set_edge_weight" \
ROLLOUTS=100 MC_MARGINALS=30 \
HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 EPI_LEVER=vaccinate EPI_BETA=1.0 EPI_GAMMA=0.3 EPI_ALPHA=0.5 OUTBREAK_PCT=1 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
CPUS=4 GRES=gpu:1 MEM=31G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=epidemic_control DATASET=p2p_gnutella05 RUN=final_seir RUN_JOBID=0 JOB_NAME=ec_p2p_gnutella05_seir \
DIFFUSION_MODEL=SEIR GEN_MODELS=SEIR \
BASELINES="dava netshield_plus greedy_walk degree_immunization frontier_immunization external:explosive_immunization_epi external:collective_influence_epi" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="remove_node remove_edge set_edge_weight" \
ROLLOUTS=100 MC_MARGINALS=30 \
HIDDEN_DIM=128 BATCH_SIZE=32 EPOCHS=400 PATIENCE=50 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 EPI_LEVER=vaccinate EPI_BETA=1.0 EPI_GAMMA=0.3 EPI_ALPHA=0.5 OUTBREAK_PCT=1 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=14400 STRATEGY_TIMEOUT=900 \
CPUS=4 GRES=gpu:1 MEM=31G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`ba`** (rank 5, synthetic: 40 BA graphs of 10,000 nodes, $m = 3$).

```bash
TASK=epidemic_control DATASET=ba RUN=final_sir RUN_JOBID=0 JOB_NAME=ec_ba_sir \
DIFFUSION_MODEL=SIR GEN_MODELS=SIR \
BASELINES="dava netshield_plus greedy_walk degree_immunization frontier_immunization external:explosive_immunization_epi external:collective_influence_epi" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="remove_node remove_edge set_edge_weight" \
ROLLOUTS=20 MC_MARGINALS=30 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 EPI_LEVER=vaccinate EPI_BETA=1.0 EPI_GAMMA=0.3 EPI_ALPHA=0.5 OUTBREAK_PCT=1 SYN_NODES=10000 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=epidemic_control DATASET=ba RUN=final_sis RUN_JOBID=0 JOB_NAME=ec_ba_sis \
DIFFUSION_MODEL=SIS GEN_MODELS=SIS \
BASELINES="dava netshield_plus greedy_walk degree_immunization frontier_immunization external:explosive_immunization_epi external:collective_influence_epi" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="remove_node remove_edge set_edge_weight" \
ROLLOUTS=20 MC_MARGINALS=30 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 EPI_LEVER=vaccinate EPI_BETA=1.0 EPI_GAMMA=0.3 EPI_ALPHA=0.5 OUTBREAK_PCT=1 SYN_NODES=10000 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

```bash
TASK=epidemic_control DATASET=ba RUN=final_seir RUN_JOBID=0 JOB_NAME=ec_ba_seir \
DIFFUSION_MODEL=SEIR GEN_MODELS=SEIR \
BASELINES="dava netshield_plus greedy_walk degree_immunization frontier_immunization external:explosive_immunization_epi external:collective_influence_epi" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="remove_node remove_edge set_edge_weight" \
ROLLOUTS=20 MC_MARGINALS=30 \
HIDDEN_DIM=256 BATCH_SIZE=8 EPOCHS=200 PATIENCE=25 NO_PLAN_DEMO=0 \
EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 EPI_LEVER=vaccinate EPI_BETA=1.0 EPI_GAMMA=0.3 EPI_ALPHA=0.5 OUTBREAK_PCT=1 SYN_NODES=10000 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

### 8.8 Cascade prediction (`cascade_prediction`, IC; 5 commands)

Baselines from section 4.8: `feature_linear`, `rpp`, `szabo_huberman`, `hawkes`, `persistence`, `weng_communities`.

`ROLLOUTS` and `MC_MARGINALS` are absent from these commands on purpose: no simulator runs on this task, the corpus is replayed by `data/wm_cascades.py`, and neither knob is read on that path. `GEN_ACTION_OPS` is empty for the same reason the two inverse tasks leave it empty, and here it is stronger than a convention: `a_t` is NULL at every step, so `T_exo` is the identity and none of the five ops can fire.

**Before submitting.** No external repos run here, so there is no venv to build, but four of the five corpora cannot be fetched by script. The three CasFlow corpora sit behind a Google Drive interstitial: open the Drive link in `data/datasets/casflow_bundle.py` (a Baidu mirror is listed beside it), extract the archive, and copy each corpus file to `data/raw/casflow/weibo/dataset.txt`, `data/raw/casflow/aps/dataset.txt` and `data/raw/casflow/twitter/dataset.txt`. `taoke`'s only source still returns 404 (re-checked 2026-09-13), so obtain `Taoke.zip` by hand and place it at `data/raw/taoke/Taoke.zip`; the loader extracts it. A missing file makes the data stage raise with these same steps rather than run on nothing. `digg_cascades` is the one automatic corpus, two KONECT tarballs fetched by the data stage; if the compute nodes have no outbound network, fetch it once from a login node first:

```bash
python -c "from data.datasets.digg_cascades import download_digg_cascades as d; d()"
```

**`casflow_aps`** (rank 1, very large (616,316 underlying nodes), critical; capped at the 30,000 busiest participants and 5,000 sampled cascades, both stated in the caption. The August run was capped at 30,000 nodes and 20,000 cascades, not uncapped as this note used to say: its metadata records 15,420 surviving cascades, 8,909 replayed, 115,599 transitions, and it was OOM-killed in the agent stage at 64G before the per-record adjacency copies were fixed on 2026-09-11; at 5,000 sampled cascades and with that fix the row runs at 62G. Training time is linear in replayed cascades at about 13 transitions per cascade, so 5,000 cascades is roughly half a day of training on a PDE-class GPU where the uncapped corpus would be months; 5,000 is the floor, since the chronological test split is about 300 cascades at that size).

```bash
TASK=cascade_prediction DATASET=casflow_aps RUN=final_ic RUN_JOBID=0 JOB_NAME=cp_casflow_aps_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="feature_linear rpp szabo_huberman hawkes persistence weng_communities" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="" \
HIDDEN_DIM=128 BATCH_SIZE=2 EPOCHS=60 PATIENCE=10 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="10" CP_SPLIT=chronological CP_METRIC=msle CP_SELECT_SPLIT=train CP_EVAL_SPLIT=test CP_MAX_NODES=30000 CP_MAX_CASCADES=5000 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=200 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=43200 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`taoke`** (rank 2, critical; trained at hidden 128 with 60 epochs and patience 10, since at the large tier's hidden 256 and 200 epochs an epoch over its 56,008 replayed transitions measured four hours on finetuner's A6000; batch 16 and patience 5, because a taoke transition is 157,000 arcs, so a batch is a few gigabytes, and at two hours per epoch early stopping decides the bill; `CP_MAX_CASCADES=2000` samples 2,000 of the 3,231 cascades that pass the size filter, so the row's chronological test split is about 300 cascades rather than 486, which the caption states and one to two days per row, which is why it keeps a 48-hour walltime; `CP_MIN_SIZE=3` is what makes the leak-free split possible, and 6 hours is the only horizon on offer, since that split admits nothing above 28,596 seconds; under the label rules corrected on 2026-09-19 (adoptions after the prediction time are dropped, and the 100-participant cap bounds the observed prefix only) an epoch is 19,139 transitions rather than 34,837, the median test cascade gains 2 participants between one and six hours and 117 of the 300 gain none, so `persistence` is a strong floor on this row and the report says so; the file has to be placed by hand first: its source repository no longer exists, so `Taoke.zip` is copied from the lab laptop's `data/raw/taoke/` to the same path on the target machine, as `setup_guide.md` section 5.2 says).

```bash
TASK=cascade_prediction DATASET=taoke RUN=final_ic RUN_JOBID=0 JOB_NAME=cp_taoke_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="feature_linear rpp szabo_huberman hawkes persistence weng_communities" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="" \
HIDDEN_DIM=128 BATCH_SIZE=16 EPOCHS=60 PATIENCE=5 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="10" CP_SPLIT=chronological CP_METRIC=msle CP_SELECT_SPLIT=train CP_EVAL_SPLIT=test CP_MIN_SIZE=3 CP_MAX_CASCADES=2000 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=1000 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=21600 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`digg_cascades`** (rank 3, very large (279,630 underlying nodes), critical; a graph-and-cascade pair rather than a bundle: the transitions run on the real friendship graph, whose 1.7M arcs make an uncapped transition cost what a digg IM transition costs, so it is capped at the 5,000 busiest participants, 500 sampled cascades and a 6-hour horizon (`CP_HORIZON=21600`, the same as taoke; the loader's windows are our own, so no published comparison moves), with batch 16 and patience 5. Measured on 2026-09-19 under the corrected label rules: 456 cascades survive (319 train, 68 val, 69 test, which the caption states), the induced friendship graph is 139,490 edges, a cascade replays into 24 transitions and an epoch is 7,527 of them, projected at about 40 minutes on finetuner's A6000 by scaling the 20-hour measurement below by arcs times transitions; the median test cascade has 23 participants at one hour and gains 43 by six. At the loader's own 24-hour horizon with 30,000 participants and 1,000 cascades the same epoch is 65,572 transitions over 1.06M arcs and measured 20 hours. Two label bugs were fixed before this row was rerun: the 100-participant cap had been applied to the whole cascade rather than to the observed prefix, which made every digg target exactly 100 minus the observed count and scored every arm at zero error, and adoptions after the prediction time had been clipped into the last timestep rather than dropped; its literature predicts the next adopter rather than popularity, so the row is self-contained rather than a published comparison).

```bash
TASK=cascade_prediction DATASET=digg_cascades RUN=final_ic RUN_JOBID=0 JOB_NAME=cp_digg_cascades_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="feature_linear rpp szabo_huberman hawkes persistence weng_communities" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="" \
HIDDEN_DIM=128 BATCH_SIZE=16 EPOCHS=60 PATIENCE=5 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="10" CP_SPLIT=chronological CP_METRIC=msle CP_SELECT_SPLIT=train CP_EVAL_SPLIT=test CP_MAX_NODES=5000 CP_MAX_CASCADES=500 CP_HORIZON=21600 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=200 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=43200 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`casflow_twitter`** (rank 4, very large (490,474 underlying nodes); capped at the 30,000 busiest participants and 5,000 sampled cascades like APS, both stated in the caption; the corpus where the leak-free split changes a ranking, CasCN becoming the best baseline, so it is the row that reproduces section 8.3's finding).

```bash
TASK=cascade_prediction DATASET=casflow_twitter RUN=final_ic RUN_JOBID=0 JOB_NAME=cp_casflow_twitter_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="feature_linear rpp szabo_huberman hawkes persistence weng_communities" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="" \
HIDDEN_DIM=128 BATCH_SIZE=2 EPOCHS=60 PATIENCE=10 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="10" CP_SPLIT=chronological CP_METRIC=msle CP_SELECT_SPLIT=train CP_EVAL_SPLIT=test CP_MAX_NODES=30000 CP_MAX_CASCADES=5000 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=200 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=43200 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=24:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

**`casflow_weibo`** (rank 5, very large (6,738,040 underlying nodes); the de-facto benchmark, run last because even at a node cap every surviving cascade replays into about 13 transitions and the uncapped survivor pool is weeks of training; capped at the 30,000 busiest participants and 10,000 sampled cascades, both stated in the caption; APS and Twitter run at 5,000).

```bash
TASK=cascade_prediction DATASET=casflow_weibo RUN=final_ic RUN_JOBID=0 JOB_NAME=cp_casflow_weibo_ic \
DIFFUSION_MODEL=IC GEN_MODELS=IC \
BASELINES="feature_linear rpp szabo_huberman hawkes persistence weng_communities" \
ARMS="routing evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-6-astra REASONING_EFFORT=high \
WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="" \
HIDDEN_DIM=128 BATCH_SIZE=2 EPOCHS=60 PATIENCE=10 NO_PLAN_DEMO=1 \
EVALUATOR=oracle BUDGET_PCTS="10" CP_SPLIT=chronological CP_METRIC=msle CP_SELECT_SPLIT=train CP_EVAL_SPLIT=test CP_MAX_NODES=30000 CP_MAX_CASCADES=10000 \
HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=200 \
MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=43200 STRATEGY_TIMEOUT=1800 \
CPUS=8 GRES=gpu:1 MEM=62G TIME=48:00:00 \
FEEDBACK=default ACTION_CONDITIONING=message \
./sbatch/pipeline.sbatch
```

## 9. Ablations and extra experiments

Everything outside the main tables. Two hosts only, chosen for speed: influence maximization on `netscience` under IC (1,589 nodes) and critical node detection on `power_grid` under IC (4,941 nodes). On graphs this size a search is bound by the LLM, about 30 calls and 15 to 40 minutes of LLM time per arm per 10 generations (measured on finished 20-generation searches with the probe turn on, 62 calls each: 28 minutes of LLM time on adaptive IM `netscience`, 78 minutes on source localization `cora_ml`, whose prompts are far longer; that is three calls a generation, the probe turn, the edit and the reflection, plus an idea search on the few exploring generations), and the evaluator is a rounding error, so the host barely changes the cost while the task type would (adaptive IM spends 2 to 5 hours of evaluator time per budget and is not used here). Netscience trains its world model in 11 minutes (642 s measured on the finished run).

### 9.0 Conventions

**Fixed in every run of this section:** one search budget, 10 percent, one dynamics (IC), `OUTER_ITERS=10`, `BASELINES=none` and `ARMS=none` unless a run names them, and every other knob at its main-table value. A run changes exactly one thing, which its command shows.

**Budgets.** The helpers default to `BUDGET_PCTS="10"` alone. The three subsections that compare search results against the main table (9.3, 9.4, 9.5) pass `BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10` instead, which is what the main runs do: the search happens once, at 10 percent, and its winning program is replanned at 1, 5 and 20 percent and scored through the canned path, referee included, with no further LLM calls (`transferred_from` marks those rows). That costs minutes per run on these two graphs. Since 2026-09-20 the same transfer covers a `discovery:<name>` row, which before then would have rerun the whole framework at every budget.

**A training run must name a world-model arm.** The pipeline skips the train stage whenever no arm evaluates against the world model (`active_stages` in `pipeline/run.py`), and the helpers default to `ARMS=none`, so `START_STAGE=train END_STAGE=train` on its own submits a job that trains nothing and exits in seconds (that is what the first 9.6 submission on 2026-09-21 did: ten jobs, no checkpoints). Every command that trains therefore carries `ARMS="baseline:<name>@world_model"`; with `END_STAGE=train` the arm is never run, it only tells the pipeline the checkpoint is wanted.

**Nothing is overwritten.** Every run has its own `RUN` name starting with `abl_` and its own `JOB_NAME` starting with `abl_im_` or `abl_cnd_`. No command in this section writes into `final_ic`. Runs that reuse the main world model copy `data/` and `world_model/`; runs that retrain copy `data/` only, because a copied checkpoint would be reused by the train stage and the run would silently train nothing.

**GPU.** Each subsection states it. In short: training needs a GPU; a search or a canned row with the world model as evaluator runs on either (these graphs are small enough for CPU, a GPU only shortens the evaluator share); the timing runs of 9.1 and 9.2 must use a GPU, since the world model's speed on the hardware it is meant for is what they measure; oracle-only, Monte Carlo, native and discovery runs need none. The helpers default to `GRES=none`, and GPU runs pass `GRES=gpu:1`.

**Six code changes this section relies on** (in the repository since 2026-09-20): `DISCOVERY_MAX_PROGRAMS` caps every discovery system at that many evaluated programs (9.5's matched-budget rows, added 2026-09-21); `scripts/time_evaluators.py` times the world model against NDlib with nothing else in the way (9.1 and 9.2, added 2026-09-21); `SEARCH_BUDGET` transfers a discovery system's program as it does an LLM arm's; `plot_runtime` draws a second figure, `plots/runtime_with_oracle.png`, beside `plots/runtime.png`, and its title now states the ratio in whichever direction it goes; `GEN_ACTION_OPS=none` asks for a diffusion-only dataset (the pipeline flag accepts an empty list); `scripts/compare_world_models.py` prints one table across world-model results files, including the count bias after 1, 2, 5, 10 and 20 steps.

The three helpers live in `sbatch/ablation_helpers.sh`, the one copy of them, so every cluster gets the current version with `git pull`. Source it once per shell, from the repo root; the functions last only as long as that shell, so a new login, SSH session or tmux pane sources it again (`type im_abl cnd_abl copy_run` prints them when they are loaded). On a machine without SLURM, put `SLURM_JOB_ID=local` in front of a call, as in section 8.0.

```bash
source sbatch/ablation_helpers.sh
```

To see what a call would submit without submitting it, add `DRY_RUN=1` to the call: it prints the `sbatch` line and the full `python -m pipeline.run` command and exits, writing nothing. Never combine it with `SLURM_JOB_ID=local`. That variable tells the script it is already inside a job, so it skips the submit path where `DRY_RUN` is read and runs the pipeline in the foreground for real.

`im_abl` submits `./sbatch/pipeline.sbatch` for influence maximization on `netscience` under IC with every knob at its main-table value except the ones fixed above, and `cnd_abl` does the same for critical node detection on `power_grid` with `OUTBREAK_PCT=10 OUTBREAK_SELECTOR=random`. Both default to `BASELINES=none ARMS=none LLM_MODEL=gpt-6-astra CPUS=4 GRES=none MEM=31G TIME=24:00:00`, and any `KEY=value` passed to a call overrides the default, since later assignments win. `copy_run <task> <dataset> <new_run> [data]` copies `data/` and `world_model/` out of `final_ic` into the new run, or `data/` alone when the fourth argument is `data` (the runs that retrain).

`copy_run` refuses an existing destination, so rerunning a block never clobbers a run. The path rewrite uses `perl -pi`, which behaves the same on the clusters and on macOS.

### 9.1 World model against Monte Carlo, timed, on the two hosts

One script, `scripts/time_evaluators.py`, and nothing else: no LLM, no search, no oracle, no referee, no pipeline. It loads a finished run's graph and its world-model checkpoint, rolls one fixed plan forward under each evaluator (the 1 percent highest-degree nodes seeded at $t=0$, horizon 10), and prints seconds per world-model sample against seconds per NDlib episode with the ratio. The first world-model call is an untimed warm-up, each side is timed three times and the median is reported, and a CUDA rollout is synchronized before the clock stops. It reads `final_ic` and writes only `results/timing/<task>_<dataset>.json` and the matching `.png`, drawn by the pipeline's own `plot_runtime` so it looks like every other runtime figure. No `copy_run` and no helper is needed.

**Needs a GPU**, because the world model's speed on a GPU is the claim: on a laptop CPU the same model measured 106 ms per sample against 23 ms per NDlib episode, five times SLOWER, where the finished cluster runs put it at about 7.5 ms per sample on a GPU. **Under a minute per host** once the node is allocated. The world-model settings are whatever the checkpoint was trained with, which for these two rows are section 8's (`sage`, `structured`, hidden 128, `message` conditioning); the script reads them from the checkpoint, so nothing is passed.

```bash
srun --gres=gpu:1 --cpus-per-task=4 --mem=31G --time=01:00:00 $SBATCH_ARGS \
  python -m scripts.time_evaluators --run-dir results/influence_maximization/netscience/final_ic
srun --gres=gpu:1 --cpus-per-task=4 --mem=31G --time=01:00:00 $SBATCH_ARGS \
  python -m scripts.time_evaluators --run-dir results/critical_node_detection/power_grid/final_ic
```

Run them from the repo root with the project's virtualenv active, so `srun` carries it to the node. Defaults: `--wm-samples 200 --mc-episodes 50 --repeats 3 --budget-pct 1 --horizon 10`. The figures are per sample, so the two sample counts need not match.

### 9.2 World model against Monte Carlo, timed, on digg

The same script on the scale graph, in two steps. The first attempt (2026-09-21, `--mc-episodes 20 --repeats 3` under a one-hour limit) timed out with nothing written, because its 25-minute estimate rested on rates from two other machines. A sizing run then measured the rates on the timing node itself: **2.4 s per world-model sample and 36 s per NDlib episode**, about 2.5 to 3 times slower on both sides than the earlier numbers (0.8 s per sample on PDE's GPU on 2026-09-08, 14 s per episode on a laptop), so the first attempt needed about 68 minutes. Per sample the world model is **15 times faster** than Monte Carlo here, against about 3 times on Network Science, which is the widening the Cost paragraph claims.

**Step 1, sizing** (done 2026-09-21). One rollout per side, 40 world-model samples, 5 NDlib episodes, written to `results/timing_sizing/` so it does not overwrite the measurement. Its two per-sample figures are the ones above; rerun it only if the node changes.

```bash
srun --gres=gpu:1 --cpus-per-task=8 --mem=62G --time=00:30:00 $SBATCH_ARGS \
  python -m scripts.time_evaluators --run-dir results/influence_maximization/digg/final_ic \
    --wm-samples 40 --mc-episodes 5 --repeats 1 --out-dir results/timing_sizing
```

**Step 2, the measurement.** The sizing counts with three repeats, so the digg panel is a median of three like the other two. The counts stay small because the per-sample figure does not depend on them: the world model runs two samples per chunk on this graph whatever the total, and an NDlib episode is the same computation whether 5 or 20 are averaged. **About 15 minutes** at the measured rates (a 1.6 minute warm-up, 4.8 minutes of world model, 9 minutes of NDlib), under a two-hour limit.

```bash
srun --gres=gpu:1 --cpus-per-task=8 --mem=62G --time=02:00:00 $SBATCH_ARGS \
  python -m scripts.time_evaluators --run-dir results/influence_maximization/digg/final_ic \
    --wm-samples 40 --mc-episodes 5 --repeats 3
```

**Needs a GPU** and the finished digg run on the same machine. The first line of the output must say `world model on cuda`; a `cpu` there means the venv's torch has no CUDA on that node, and the world-model figure is then not the one to report.

### 9.3 LLM comparison

`evolve_free@oracle` under three other models, everything else at protocol, **budgets included**: the main table's four budgets with `SEARCH_BUDGET=pct10`, exactly as the main runs have it. The LLM searches once, at 10 percent, and the winner is replanned and scored at 1, 5 and 20 percent through the canned path, referee included, with no further LLM calls. The `gpt-6-astra` side of the comparison is the main run's own `evolve_free@oracle` rows in `final_ic`, which were searched at 10 percent and transferred to the other three budgets under the same settings, so they are not rerun here. The probe turn stays ON (the sbatch default, and what the main runs used): turning it off would save a third of the calls, ten to twenty minutes a run, and would make the three new rows a different protocol from the `gpt-6-astra` row they are compared against. The oracle evaluator needs no GPU, so these run with `GRES=none`. The main run scored its oracle arm on the GPU it held for training, so the two sides share the sampler, the seeds and the expectation but not the random stream (CPU and CUDA generators differ); that is noise of the size `reward_se` already reports, not a protocol difference. Everything else the search sees is identical: a diff of the two commands leaves only `ARMS`, `BASELINES`, `GRES` and `START_STAGE`, the search's anchors are a fixed per-task list that `BASELINES` does not feed, and no file of the search loop has changed since the commands moved to 10 generations. About 20 to 45 minutes per run: 15 to 40 of search, scaled from 20-generation runs on other tasks rather than measured on these two hosts, plus a few minutes for start-up, the three transferred points, the plots and the report. The measured figure is the main run's own `pct10` row, the same search under `gpt-6-astra`, which the check below prints; the other three models differ from it only in latency per call, which is not known in advance.

**Check the main run before relying on its row.** The commands in section 8 were at `OUTER_ITERS=20` until 2026-09-17 19:30 and had no `SEARCH_BUDGET` until 2026-09-14 16:40, so a `final_ic` submitted before then searched twice as long as these runs will, and its row is not the `gpt-6-astra` side of this comparison. This must print `10 pct10 True` on both hosts; if it does not, add `gpt-6-astra` back to the loop below (one more run per host).

```bash
for r in influence_maximization/netscience critical_node_detection/power_grid; do
  python3 -c "import json; c = json.load(open('results/$r/final_ic/pipeline.json'))['config']; print(c['outer_iters'], c['search_budget'], c['probe_turn'])"
  python3 -c "import json; r = json.load(open('results/$r/final_ic/agent/pct10/evolve_free@oracle.json')); print(round(r['elapsed_seconds'] / 60), 'min in total,', round(r['evaluator_seconds'] / 60), 'min of evaluator,', r['llm_usage']['calls'], 'LLM calls')"
done
```

```bash
for m in gpt-5.6-luna gpt-5.6-terra gpt-5.6-sol; do tag=${m##*-}
  copy_run influence_maximization netscience abl_llm_$tag
  im_abl RUN=abl_llm_$tag JOB_NAME=abl_im_llm_$tag START_STAGE=agent ARMS="evolve_free@oracle" LLM_MODEL=$m \
    BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10
  copy_run critical_node_detection power_grid abl_llm_$tag
  cnd_abl RUN=abl_llm_$tag JOB_NAME=abl_cnd_llm_$tag START_STAGE=agent ARMS="evolve_free@oracle" LLM_MODEL=$m \
    BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10
done
```

### 9.4 Evaluator noise

`N_SAMPLES` at 50, 200 and 800 for the oracle and world-model arms. Each arm's result carries `acceptance_ledger`: `naive_accepts` is what a plain greater-than rule would have accepted, `band_accepts` what the paired band accepted, and `lucky_accepts_prevented` the difference, which should grow as the sample count falls while `referee_reward` holds. The ledger exists on the searched `pct10` row only; the three transferred rows carry the winner's `referee_reward` at the other budgets, which shows whether a noisier search also picked a program that travels worse. **GPU optional** (set here, since 800 world-model samples per evaluation is the one place it shortens the run noticeably). About an hour and a half each, since a run holds two arms.

```bash
for n in 50 200 800; do
  copy_run influence_maximization netscience abl_n$n
  im_abl RUN=abl_n$n JOB_NAME=abl_im_n$n GRES=gpu:1 START_STAGE=agent \
    ARMS="evolve_free@oracle evolve_free@world_model" N_SAMPLES=$n BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10
  copy_run critical_node_detection power_grid abl_n$n
  cnd_abl RUN=abl_n$n JOB_NAME=abl_cnd_n$n GRES=gpu:1 START_STAGE=agent \
    ARMS="evolve_free@oracle evolve_free@world_model" N_SAMPLES=$n BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10
done

python3 - <<'EOF'
import glob, json
for path in sorted(glob.glob("results/*/*/abl_n*/agent/pct10/evolve_free@*.json")):
    result = json.load(open(path))
    ledger = result["acceptance_ledger"]
    others = {p: json.load(open(path.replace("/pct10/", f"/{p}/")))["referee_reward"] for p in ("pct1", "pct5", "pct20")}
    print(path.split("/")[1], path.split("/")[3], result["evaluator"], f"referee={result['referee_reward']:.2f}",
          f"naive={ledger['naive_accepts']}", f"band={ledger['band_accepts']}", f"prevented={ledger['lucky_accepts_prevented']}",
          " ".join(f"{p}={value:.2f}" for p, value in others.items()))
EOF
```

### 9.5 Algorithm-discovery baselines

Five published systems at their own defaults: FunSearch, EoH, OpenEvolve, LLaMEA and ReEvo. Nothing about their loops is overridden, `DISCOVERY_SMOKE` must be unset, and `OUTER_ITERS` does not apply to them. Their LLM is `gpt-5.6-terra` (`LLM_MODEL` on the call, which `baselines/discovery.py` hands to each framework's own config), not the helpers' `gpt-6-astra`; the matching row of ours is therefore 9.3's `abl_llm_terra`, the same search under the same model, rather than the main run's `gpt-6-astra` row. The frameworks build their own API requests, so `REASONING_EFFORT` does not reach them: they call the model at the gateway's default effort while our arms call it at `high`, and a table that compares the two says so. They search against the oracle simulator and never see the world model. Evaluated programs at default: FunSearch 20, EoH 110, OpenEvolve 100, LLaMEA 100, ReEvo 100, against about 10 for our search, so report `external.info` beside each score. **No GPU.** On the two hosts FunSearch is under half an hour and the other four one to two hours each; on digg, where one 200-sample oracle evaluation costs about a minute, FunSearch is about half an hour and the other four about three hours each (unmeasured; scaled from the per-call and per-evaluation costs). One run directory per system, so every job can run at once. Four budgets under the same rule as our own arms: the framework runs once, at 10 percent, and the program it returns is replanned at 1, 5 and 20 percent and replayed on the referee, so each side of the comparison gets exactly one search. Those three rows launch nothing and cost seconds. These are the only discovery runs in the plan: section 8's per-task discovery loops were removed on 2026-09-20.

Two systems ReEvo and LLaMEA need that the first three did not: ReEvo's initial population is one request with `n=30`, which our gateway collapses to one choice, so its adapter replicates the request (`baselines/discovery.py`); and LLaMEA has no per-evaluation timeout of its own, so a runaway candidate is bounded only by the scorer's own limit. Neither needs anything from the caller.

**The two hosts, LLaMEA and ReEvo only.** FunSearch, EoH and OpenEvolve already have their `abl_disc_*` runs on `netscience` and `power_grid`, so this loop adds the two new systems without touching those directories.

```bash
python -m baselines.setup_baselines --only llamea reevo

for s in llamea reevo; do
  copy_run influence_maximization netscience abl_disc_$s
  im_abl RUN=abl_disc_$s JOB_NAME=abl_im_disc_$s START_STAGE=agent LLM_MODEL=gpt-5.6-terra \
    BASELINES="discovery:$s" BASELINE_TIMEOUT=86400 TIME=30:00:00 BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10
  copy_run critical_node_detection power_grid abl_disc_$s
  cnd_abl RUN=abl_disc_$s JOB_NAME=abl_cnd_disc_$s START_STAGE=agent LLM_MODEL=gpt-5.6-terra \
    BASELINES="discovery:$s" BASELINE_TIMEOUT=86400 TIME=30:00:00 BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10
done
```

**All five on digg.** Influence maximization on the scale graph, everything else as the two hosts. The helpers are not used because their defaults are the medium tier; the command below carries the digg row's own settings from section 8 (`ROLLOUTS=20 MC_MARGINALS=20 HIDDEN_DIM=128 BATCH_SIZE=2 EPOCHS=60 PATIENCE=10`, referee at 200 samples, 8 CPUs, 62 GB) with `GRES=none`, since a discovery row never touches the world model. `copy_run ... data` copies `data/` alone (several GB; `ln -s` in its place is safe, the data is only read), and no `world_model/` is needed: the agent stage asks for a checkpoint only when an arm evaluates against the world model, and `discovery:<name>` never does. `BASELINE_TIMEOUT=172800` gives a framework two days, under the three-day walltime.

```bash
python -m baselines.setup_baselines --only llm4ad_funsearch eoh openevolve llamea reevo

for s in llm4ad_funsearch eoh openevolve llamea reevo; do
  copy_run influence_maximization digg abl_disc_$s data
  env TASK=influence_maximization DATASET=digg RUN=abl_disc_$s RUN_JOBID=0 JOB_NAME=abl_im_digg_disc_$s \
    DIFFUSION_MODEL=IC GEN_MODELS=IC START_STAGE=agent ARMS=none BASELINES="discovery:$s" \
    LLM_MODEL=gpt-5.6-terra REASONING_EFFORT=high \
    WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
    ROLLOUTS=20 MC_MARGINALS=20 HIDDEN_DIM=128 BATCH_SIZE=2 EPOCHS=60 PATIENCE=10 NO_PLAN_DEMO=0 \
    EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 \
    HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=200 \
    MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=172800 STRATEGY_TIMEOUT=1800 \
    FEEDBACK=default ACTION_CONDITIONING=message \
    CPUS=8 GRES=none MEM=62G TIME=3-00:00:00 ./sbatch/pipeline.sbatch
done
```

**Matched budget, five programs each.** The same five systems capped at 5 evaluated programs, which is about what our own 10-generation search evaluates once its rejected candidates are set aside, so a row here and 9.3's `abl_llm_terra` row spent the same number of scored programs. `DISCOVERY_MAX_PROGRAMS=5` is read by `baselines/discovery.py` when the context is built and applied inside each adapter to the framework's own budget knob (OpenEvolve `max_iterations`, LLaMEA `budget`, FunSearch `max_sample_nums`, ReEvo `max_fe` with the population sizes shrunk to fit inside it, EoH `pop_size` 1 with 3 evolution samples after the 2 initial ones); nothing else about a loop changes, and unset it keeps the published defaults above. The variable travels to the job through `--export=ALL` like `DISCOVERY_SMOKE` does. Run directories and job names carry `_5iter` so nothing overwrites the default-budget rows. **No GPU.** About 5 LLM calls per run, so a few minutes each on the hosts and under half an hour on digg, where each evaluation costs about a minute.

```bash
# the two hosts
for s in llm4ad_funsearch eoh openevolve llamea reevo; do
  copy_run influence_maximization netscience abl_disc_5iter_$s
  DISCOVERY_MAX_PROGRAMS=5 im_abl RUN=abl_disc_5iter_$s JOB_NAME=abl_im_disc_5iter_$s START_STAGE=agent LLM_MODEL=gpt-5.6-terra \
    BASELINES="discovery:$s" BASELINE_TIMEOUT=86400 TIME=06:00:00 BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10
  copy_run critical_node_detection power_grid abl_disc_5iter_$s
  DISCOVERY_MAX_PROGRAMS=5 cnd_abl RUN=abl_disc_5iter_$s JOB_NAME=abl_cnd_disc_5iter_$s START_STAGE=agent LLM_MODEL=gpt-5.6-terra \
    BASELINES="discovery:$s" BASELINE_TIMEOUT=86400 TIME=06:00:00 BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10
done

# digg
for s in llm4ad_funsearch eoh openevolve llamea reevo; do
  copy_run influence_maximization digg abl_disc_5iter_$s data
  DISCOVERY_MAX_PROGRAMS=5 env TASK=influence_maximization DATASET=digg RUN=abl_disc_5iter_$s RUN_JOBID=0 JOB_NAME=abl_im_digg_disc_5iter_$s \
    DIFFUSION_MODEL=IC GEN_MODELS=IC START_STAGE=agent ARMS=none BASELINES="discovery:$s" \
    LLM_MODEL=gpt-5.6-terra REASONING_EFFORT=high \
    WM_MODEL=sage HEAD=structured GEN_ACTION_OPS="add_node remove_node" \
    ROLLOUTS=20 MC_MARGINALS=20 HIDDEN_DIM=128 BATCH_SIZE=2 EPOCHS=60 PATIENCE=10 NO_PLAN_DEMO=0 \
    EVALUATOR=oracle BUDGET_PCTS="1 5 10 20" SEARCH_BUDGET=pct10 \
    HORIZON=10 OUTER_ITERS=10 N_SAMPLES=200 MC_RUNS=200 REFEREE=oracle REFEREE_SAMPLES=200 \
    MC_AGREEMENT=0 CREDIT=1 SEED=42 BASELINE_TIMEOUT=43200 STRATEGY_TIMEOUT=1800 \
    FEEDBACK=default ACTION_CONDITIONING=message \
    CPUS=8 GRES=none MEM=62G TIME=12:00:00 ./sbatch/pipeline.sbatch
done
```

### 9.6 World-model backbones

Training only. `message` conditioning exists for GraphSAGE alone, so all five train under `ACTION_CONDITIONING=none`, including a fresh GraphSAGE run: comparing the other four with the main `message` checkpoint would mix two variables. GCNII gets the 8 layers its design is for, the rest keep 3. `data/` only is copied. **Needs a GPU.** About 11 minutes each on netscience.

```bash
for b in sage gat gcn gcnii gt; do layers=3; [ "$b" = gcnii ] && layers=8
  copy_run influence_maximization netscience abl_wm_$b data
  im_abl RUN=abl_wm_$b JOB_NAME=abl_im_wm_$b GRES=gpu:1 START_STAGE=train END_STAGE=train \
    WM_MODEL=$b N_LAYERS=$layers ACTION_CONDITIONING=none ARMS="baseline:high_degree@world_model"
  copy_run critical_node_detection power_grid abl_wm_$b data
  cnd_abl RUN=abl_wm_$b JOB_NAME=abl_cnd_wm_$b GRES=gpu:1 START_STAGE=train END_STAGE=train \
    WM_MODEL=$b N_LAYERS=$layers ACTION_CONDITIONING=none ARMS="baseline:adaptive_degree@world_model"
done

for host in influence_maximization/netscience critical_node_detection/power_grid; do
  python -m scripts.compare_world_models $(for b in sage gat gcn gcnii gt; do
    echo "$b=results/$host/abl_wm_$b/world_model/${b}_IC.json"; done)
done
```

### 9.7 Hidden edge weights

With the weights visible the structured head can read the transmission probability off its input, so this trains with them replaced by ones. No coding agent: after training, one canned heuristic is scored by the world model (`ARMS="baseline:<name>@world_model"`; a row given through `BASELINES` would be scored on the oracle whatever `EVALUATOR` says, see 9.1, and these commands were written that way until 2026-09-21), which gives the referee gap (`arm_minus_referee` in the row's JSON, and the In-Loop Evaluator Fidelity table of the report) without an LLM call. `abl_wm_visible` scores the same heuristic under the main checkpoint, the visible-weights reference. **Training needs a GPU; the reference run does not.** Caveat for the write-up: these runs use the weighted-cascade model, $p(u \rightarrow v) = 1/\mathrm{indeg}(v)$, which is recoverable from the degree channel, so this measures whether the model can learn that law from cascades, not arbitrary probabilities.

```bash
copy_run influence_maximization netscience abl_wm_hidden data
im_abl RUN=abl_wm_hidden JOB_NAME=abl_im_wm_hidden GRES=gpu:1 START_STAGE=train \
  HIDE_EDGE_WEIGHTS=1 ARMS="baseline:degree_discount@world_model" CREDIT=0 PROVENANCE=0
copy_run influence_maximization netscience abl_wm_visible
im_abl RUN=abl_wm_visible JOB_NAME=abl_im_wm_visible START_STAGE=agent \
  ARMS="baseline:degree_discount@world_model" CREDIT=0 PROVENANCE=0

copy_run critical_node_detection power_grid abl_wm_hidden data
cnd_abl RUN=abl_wm_hidden JOB_NAME=abl_cnd_wm_hidden GRES=gpu:1 START_STAGE=train \
  HIDE_EDGE_WEIGHTS=1 ARMS="baseline:adaptive_degree@world_model" CREDIT=0 PROVENANCE=0
copy_run critical_node_detection power_grid abl_wm_visible
cnd_abl RUN=abl_wm_visible JOB_NAME=abl_cnd_wm_visible START_STAGE=agent \
  ARMS="baseline:adaptive_degree@world_model" CREDIT=0 PROVENANCE=0

for host in influence_maximization/netscience critical_node_detection/power_grid; do
  python -m scripts.compare_world_models visible=results/$host/final_ic/world_model/sage_IC.json \
    hidden=results/$host/abl_wm_hidden/world_model/sage_IC.json
done
```

### 9.8 Error against horizon

Steps 1 to 10 need no run: every world-model results file already holds the per-step rollout curves, and the table script reads the relative count bias after 1, 2, 5 and 10 steps from them (on the finished netscience run: $-1.5\%$, $-1.9\%$, $-2.2\%$ and $-2.2\%$, which is the no-compounding result). Step 20 needs episodes that long, so a small dataset is generated at `GEN_HORIZON=20` and the main checkpoint is evaluated on it frozen. **No GPU**: the data stage and the frozen evaluation both run on CPU. On netscience the cascades have died by about step 8, so the 20-step column showing the same value as the 10-step one is the expected and useful outcome: no late drift.

```bash
for host in influence_maximization/netscience critical_node_detection/power_grid; do
  python -m scripts.compare_world_models main=results/$host/final_ic/world_model/sage_IC.json
done

im_abl RUN=abl_wm_h20 JOB_NAME=abl_im_wm_h20 START_STAGE=data END_STAGE=data GEN_HORIZON=20 ROLLOUTS=20
cnd_abl RUN=abl_wm_h20 JOB_NAME=abl_cnd_wm_h20 START_STAGE=data END_STAGE=data GEN_HORIZON=20 ROLLOUTS=20

# after both data jobs finish
for host in influence_maximization/netscience critical_node_detection/power_grid; do
  python -m scripts.eval_frozen --checkpoint results/$host/final_ic/world_model/wm_sage_IC.pt \
    --data-dir results/$host/abl_wm_h20/data --allow-trained-target \
    --n-samples 20 --max-episodes 50 --device cpu --out results/$host/abl_wm_h20/eval_h20.json
  python -m scripts.compare_world_models horizon10=results/$host/final_ic/world_model/sage_IC.json \
    horizon20=results/$host/abl_wm_h20/eval_h20.json
done
```

### 9.9 A world model trained without actions, tested on actions

`GEN_ACTION_OPS=none` generates a diffusion-only dataset: no action after the seed commit and no counterfactual forks. A model trained on it is then evaluated, frozen, on the main run's action-rich test split, which it has never seen the like of. It trains under `ACTION_CONDITIONING=none`, because a conditioning module that never receives an action signal stays at its initialization and would add noise at test time, so the fair reference is `abl_wm_sage` from 9.6 (same architecture, trained with actions); the main checkpoint is listed beside it, copied into `abl_wm_reference` so the comparison reads nothing from `final_ic`. Read the action columns: the drop in $\Delta$F1 when actions are zeroed and the counterfactual-effect correlation say whether the model responds to actions it was never trained on, and the count bias says what the missing data costs in calibration. **Training needs a GPU; the frozen evaluation does not.** Run 9.6 first.

```bash
im_abl RUN=abl_wm_noaction JOB_NAME=abl_im_wm_noaction GRES=gpu:1 START_STAGE=data END_STAGE=train \
  ARMS="baseline:high_degree@world_model" \
  GEN_ACTION_OPS=none ACTION_CONDITIONING=none NO_PLAN_DEMO=1
cnd_abl RUN=abl_wm_noaction JOB_NAME=abl_cnd_wm_noaction GRES=gpu:1 START_STAGE=data END_STAGE=train \
  ARMS="baseline:adaptive_degree@world_model" \
  GEN_ACTION_OPS=none ACTION_CONDITIONING=none NO_PLAN_DEMO=1

copy_run influence_maximization netscience abl_wm_reference
copy_run critical_node_detection power_grid abl_wm_reference

# after both training jobs finish
for host in influence_maximization/netscience critical_node_detection/power_grid; do
  python -m scripts.eval_frozen --checkpoint results/$host/abl_wm_noaction/world_model/wm_sage_IC.pt \
    --data-dir results/$host/abl_wm_reference/data --allow-trained-target \
    --n-samples 20 --max-episodes 50 --device cpu --out results/$host/abl_wm_noaction/eval_on_actions.json
  python -m scripts.compare_world_models main=results/$host/abl_wm_reference/world_model/sage_IC.json \
    same_architecture=results/$host/abl_wm_sage/world_model/sage_IC.json \
    no_actions=results/$host/abl_wm_noaction/eval_on_actions.json
done
```

### 9.10 Summary

| Section | Runs per host | Stages run | GPU | Time per run | LLM calls per run |
| --- | --- | --- | --- | --- | --- |
| 9.1 timing on the two hosts (one script, world model against NDlib) | 1 | none, reads `final_ic` | required | under 1 min | 0 |
| 9.2 timing on digg (same script, sizing run done) | 1 left | none, reads `final_ic` | required | about 15 min | 0 |
| 9.3 LLM comparison (four budgets, search at pct10) | 3 | agent to report | no | 20 to 45 min | about 30 (1 arm, probes on) |
| 9.4 evaluator noise (four budgets, search at pct10) | 3 | agent to report | optional | about 1.5 h | about 60 (2 arms, about 30 each) |
| 9.5 discovery baselines (four budgets, search at pct10) | 2 new per host, 5 on digg, plus 5 capped runs per graph | agent to report | no | 0.5 to 2 h per host, 0.5 to 3 h on digg; minutes when capped | 20 to 130; 5 when capped |
| 9.6 backbones | 5 | train | required | about 11 min | 0 |
| 9.7 hidden weights | 2 | train to report, agent to report | training only | about 15 min | 0 |
| 9.8 horizon | 1 | data, then a script | no | under 1 h | 0 |
| 9.9 no-action model | 1 | data to train, then a script | training only | under 1 h | 0 |

**Order.** Submit the training runs (9.6, 9.7, 9.9) and the two data-only runs (9.8) first: they are minutes each and need no gateway. Then 9.1 and 9.2, the two timing scripts, which take minutes and need only a GPU node and the finished main runs. Then the searches (9.3, 9.4, 9.5), which are bound by the gateway: together about 1,800 LLM calls across both hosts, so pace them to the plan's usage limit rather than to the cluster.
