# Baselines

Everything we compare against, in two layers:

1. **Our own library** (`coding_agent/tools/algorithms.py`): 36 classical IM algorithms reimplemented in Python against our primitives. Fast, always available, no setup. These are **condition 1 (Pure GA)**.
2. **External published repos** (this folder): the original authors' code for published methods, fetched into `baselines/external/`. These are **condition 7 (Published baseline)**.

For the literature itself (what each method published, and which of their numbers are comparable to ours) see [`../research/influence_maximization.md`](../research/influence_maximization.md).

---

## The one contract

Every external baseline reduces to:

```
(graph, budget, diffusion_model)  ->  list[int] seed set
```

We then score that seed set with **our own referee**, the exact oracle simulator that judges our own arms (NDlib Monte Carlo under `--mc-agreement`). This is deliberate and it is the single most important design decision here.

A paper's reported spread depends on its simulator, its edge probabilities, its MC count, its horizon, and sometimes its own version of the graph (see `research/influence_maximization.md` §6.3: "Cora-ML" alone means two different graphs). Those numbers are **not** comparable to ours. Their _seed set_ is. Running every method's seeds through one referee is the only apples-to-apples comparison available, and it means external baselines need no special result format: the seeds are wrapped as a canned `Strategy` and flow through the identical executor, validation, rollout, MC replay, and results JSON as every other arm.

---

## Where each baseline lives

Every method is in exactly one of three states. **Some methods exist in both layers**: our library has a simplified Python `imm`, `tim`, `ssa`, and the external registry has the authors' originals. Those are different arms (`baseline_imm` vs `external_imm`) and both can run in the same sweep.

| Baseline        | Kind      | Our library?  | External folder                 | Status                        |
| --------------- | --------- | ------------- | ------------------------------- | ----------------------------- |
| **IMM**         | classical | yes simplified | `baselines/external/imm/`       | manual fetch (SourceForge) |
| **TIM / TIM+**  | classical | yes simplified | `baselines/external/tim/`       | manual fetch (SourceForge) |
| **OPIM-C**      | classical | no            | `baselines/external/opim/`      | setup (C++ `make`)         |
| **SubSIM**      | classical | no            | `baselines/external/subsim/`    | setup (C++ `make`)         |
| **SSA / D-SSA** | classical | yes simplified | `baselines/external/ssa/`       | setup (C++ `make`)         |
| **MOEIM**       | learned   | no            | `baselines/external/moeim/`     | setup (pure Python)        |
| **ToupleGDD**   | learned   | no            | `baselines/external/touplegdd/` | setup (pretrained ckpt)    |
| **DeepIM**      | learned   | no            | `baselines/external/deepim/`    | setup + `deepim_data.py`   |
| **GLIE**        | learned   | no            | `baselines/external/glie/`      | setup (stored model, inference only) |
| **LeNSE**       | learned   | no            |                                 | registered, no adapter (four-stage training pipeline, no README, no checkpoints) |
| **HIM**         | learned   | no            |                                 | registered, no adapter (per-dataset config schema inside `configs.zip`) |
| **GCOMB**       | learned   | no            |                                 | blocked: Python 2.7       |
| **IMINFECTOR**  | learned   | no            |                                 | blocked: needs cascades   |
| **PIANO**       | learned   | no            |                                 | blocked: no public code   |
| **OIM**         | classical | no            |                                 | blocked: no public code   |

Also in our Python library only (no external counterpart needed or available): CELF, CELF++, degree-discount, PMIA/MIA, LDAG, SIMPATH-style SP1M, IRIE, StaticGreedy, SKIM, RIS, community-IM, CoFIM, VoteRank, k-shell, collective influence, and the metaheuristics (simulated annealing, hill climbing, GA), 36 in total, listed by `python -m pipeline.run --help` under `--baselines`.

`baselines/external/` is **gitignored**: several repos are 100 MB+ (MOEIM alone is ~128 MB) and carry their own licences, so every baseline is fetched by `setup_baselines.py` rather than vendored into this repo. Nothing third-party is committed here; only the registry, adapters, and this README are ours.

### What each one is

**MOEIM**: _Many-Objective Evolutionary Influence Maximization_, GECCO 2024. A multi-objective evolutionary algorithm optimising spread, seed-set size, community coverage, fairness, budget, and time simultaneously, with graph-aware mutation operators and smart initialization. **The current SOTA on Jazz and Cora-ML, it beats DeepIM.** Pure Python, no training, no GPU, and its `--k` is already a fraction of |V| exactly like our `--budget-pcts`. Its `setting2` is the configuration it used for its DeepIM comparison. The most directly runnable external baseline we have. [paper](https://arxiv.org/abs/2403.18755) · [repo](https://github.com/eliacunegatti/MOEIM)

**ToupleGDD**: _Three coupled GNNs + Double DQN_, IEEE TCSS 2023. Deep RL over graph embeddings; picks seeds sequentially by Q-value. Ships **pretrained checkpoints** (`tripling.ckpt`, `s2vdqn.ckpt`), so it runs inference-only with no training. It was trained on sub-300-node random graphs and designed to generalize to million-node graphs, the same generalization claim our world model makes, which makes it a pointed comparison. Needs torch. [paper](https://arxiv.org/abs/2210.07500) · [repo](https://github.com/Dtrycode/ToupleGDD)

**OPIM-C**: _Online Processing for Influence Maximization_, SIGMOD 2018. Near-optimal Reverse Influence Sampling with anytime guarantees. C++ reference implementation. Our library's `imm` and `ris_basic` are simplified Python stand-ins; this is the faithful, fast version. [paper](https://dl.acm.org/doi/10.1145/3183713.3183749) · [repo](https://github.com/tangj90/OPIM)

**SSA / D-SSA**: _Stop-and-Stare_, SIGMOD 2016. RIS with the tightest published sample-complexity bounds. C++. [paper](https://arxiv.org/abs/1605.07990) · [repo](https://github.com/hungnt55/Stop-and-Stare)

**IMM**: _Influence Maximization in Near-Linear Time: A Martingale Approach_, SIGMOD 2015. **The** classical reference point; essentially every IM paper since 2015 compares against it, and it is the baseline in DeepIM's table. Our library's `imm` is a simplified Python reimplementation, fine as a condition-1 arm, but the authors' C++ original is what published numbers come from. Tang et al. release a **tarball on SourceForge, not a git repo**, so `setup_baselines` cannot clone it and prints manual instructions instead. A third-party git reimplementation exists at [gdelpuente/IMM](https://github.com/gdelpuente/IMM) (not author-endorsed). [paper](https://dl.acm.org/doi/10.1145/2723372.2723734) · [SourceForge](https://sourceforge.net/projects/im-imm/)

**TIM / TIM+**: SIGMOD 2014. IMM's predecessor; the paper that made RIS practical. Same SourceForge-tarball situation. [paper](https://arxiv.org/abs/1404.0900) · [SourceForge](https://sourceforge.net/projects/timplus/)

**SubSIM**: _Influence Maximization Revisited: Efficient Sampling with Bound Tightened_, SIGMOD 2020. Sublinear-time RIS, and the third of the three RIS methods in DeepIM's comparison table (IMM / OPIM / SubSIM). **Not in our Python library at all**, so this repo is the only way to run it. [paper](https://dl.acm.org/doi/10.1145/3318464.3389740) · [repo](https://github.com/qtguo/subsim)

### Why three are blocked, and why six more have no adapter

These are **not** "not done yet": each has a concrete, stated reason:

**DeepIM** (ICML 2023) was blocked until its adapter built the missing inputs itself. The public repo's `data/` folder **ships empty**, but `genim.py`'s input is a pickled `{'adj', 'inverse_pairs'}` dict of seed and influenced vectors, which is exactly what our simulator produces, so `baselines/deepim_data.py` regenerates it from our graph store at export time. It then trains a VAE plus SpGAT per (dataset, dynamics, budget), so it is not inference-only and its runtime has to be budgeted (about 8,600 s per budget point on netscience). **Its published tables are transcribed in `research/influence_maximization.md` §5.1 and are directly comparable on Jazz and Power Grid.**

**GCOMB** (NeurIPS 2020): requires **both** Python 2.7 and Python 3 environments (`requirements_python2.7.txt` + `requirements_python3.txt`) plus a multi-stage supervised-then-RL pipeline driven by shell scripts. Python 2.7 is end-of-life. Unblocking means containerising it.

**IMINFECTOR** (TKDE 2020): **structurally inapplicable**, not an engineering gap. IMINFECTOR is model-free: it learns from _observed diffusion cascades_ and never assumes IC/LT. Our BA/SBM/ER graphs are synthetic and Jazz/Power Grid/NetScience ship no cascade logs, so there is nothing for it to train on. It can only run on Digg/Weibo/MAG, which our pipeline cannot simulate at that scale anyway.

**PIANO** (IEEE TCSS 2022): **no public code release**. The paper advertises "pretrained PIANO models" but publishes no repository or download link, and no official implementation was found. Tellingly, ToupleGDD's authors state they had to _revise the S2V-DQN code themselves_ to obtain a PIANO baseline, which is why PIANO's numbers differ between papers. Unblocking means reimplementing from the paper.

**OIM** (KDD 2015): registered under `adaptive_online_im` (`oim`, `oim_lt`, with `timlinucb`) without an adapter, because it is a _different problem setting_ (Lei, Maniu, Mo, Cheng & Senellart): OIM assumes repeated campaigns with feedback that updates edge weights between rounds, whereas ours is single-shot seeding. Even with code, the comparison would need care.

**LeNSE** (ICML 2022) and **HIM** (2025) are registered as learned IM methods without an adapter. LeNSE's `IM/` variant is a four-stage training pipeline (`embedding_training.py`, `guided_exploration_training.py`, `dqn_training.py`, `dqn_test.py`) with no README and no shipped checkpoints, so every stage's arguments would have to be reverse-engineered before an adapter could be written. HIM's data format is resolved (a pickled networkx graph plus a pickled cascade file, and a `seed_ratio` that is a percentage exactly like `--budget-pcts`), but its per-dataset model config lives in a `configs.zip` whose schema cannot be read without unpacking the repo, and it trains 200 epochs. **IMM** and **TIM** are registered without adapters for the tarball reason above; the library's Python `imm` and `tim` are the condition-1 stand-ins.

GCOMB, IMINFECTOR, PIANO and OIM all appear in DeepIM's published tables, which are transcribed in [`../research/influence_maximization.md`](../research/influence_maximization.md) §5.1, so numeric comparison against them is available from the paper on Jazz and Power Grid even with their code blocked.

### Per-task registration

`ExternalBaseline.task` names the graph task an entry solves, so one task's published baselines never join another's sweep and `--baselines all` under `--task X` expands to X's repos only. All eight tasks have entries today (110 task repos registered, 54 wired, 25 of those learned; the eleven condition-9 entries carry `task="*"`, nine of them wired, and are described in the last section of this file):

| task | registered | wired |
| --- | --- | --- |
| `influence_maximization` | 14 | 7 |
| `adaptive_online_im` | 6 (`adaptiveim`, `rl4im`, `mrim`, `oim`, `oim_lt`, `timlinucb`) | 2 |
| `critical_node_detection` | 12 (`finder`, `gdm`, `mind`, `spr`, `nirm`, `dcrs`, `gnd`, `decycler`, `collective_influence`, `explosive_immunization`, `dismantling_review`, `selinda`) | 11 |
| `source_localization` | 17 (six `graphsl_*` arms, four `cosasi_*` arms, plus `graphsl`, `cosasi`, `slvae`, `ivgd`, `cnsl`, `pdsl`, `gnn_source_detection`) | 10 |
| `influence_blocking` | 3 (`sandimin`, `imin_joc`, `stratlearner`) | 2 |
| `cascade_reconstruction` | 13 (three `ditto*` arms, plus `grin`, `spin`, `deep_demixing`, `reconstructing_cascade`, `cascade_tree_samples`, `cult`, `active_cascade_reconstruction`, `brits`, `dipt`, `netrate`) | 6 |
| `cascade_prediction` | 20 (`casflow`, `ccgl`, `ctcp`, `castemp`, `deephawkes`, `deepcas`, `cascn`, `mucas`, `coupledgnn`, `seismic`, `featuredriven_hawkes`, `hip`, `topolstm`, `deepinf`, `forest`, `ms_hgat`, `casseqgcn`, `ccasgnn`, `casft`, `casdo`) | 5 |
| `epidemic_control` | 25 (six `netimm_*` arms, five shared `*_epi` dismantlers, plus `rlgn`, `idrleca`, `durleca`, `epilearn`, `covasim`, `pandemic_simulator`, `epimodel`, `greedywalk`, `netmelt`, `fractional_immunization`, `preciado`, `colagnn`, `stan`, `epignn`) | 11 |

**Fifteen of cascade prediction's twenty are blocked, and that is the literature rather than us.** This line of work peaked in 2017-19 and its reference implementations were never ported: `deephawkes`, `deepcas` and `topolstm` are **Python 2** (py2 `print` statements, so the modules do not parse at all), `mucas` needs **eleven input arrays** its `get_batch` unpacks (position encodings, Chebyshev K-orders, a sampled support, and an outbreak-classification label we have no analogue for), `seismic` is an **R package** and `featuredriven_hawkes` is **R plus a notebook** with no Python entry point. Three more (`deepinf`, `forest`, `ms_hgat`) have fine modern code and answer a **different question**: they rank the next adopter or classify "will `v` adopt", so there is no popularity to read back. And the two most recent architectures, `casft` (AAAI-25, the source of the only recent result table that survives `pdftotext`) and `casdo` (TKDE'24), have **no public code at all**. Every reason above was read from the repo at HEAD rather than assumed, because a reader who sees "DeepHawkes, code available" in §4.1 will reasonably ask why it is not a baseline.

The five that DO run share one export: CasFlow's canonical five-field line format, which §8.5 identifies as this literature's de-facto benchmark artefact since no OGB/TGB analogue exists:

```bash
python -m baselines.setup_baselines --only casflow
python -m pipeline.run --dataset casflow_weibo --task cascade_prediction \
    --baselines szabo_huberman external:casflow external:ctcp
```

**`coupledgnn` is the one worth reading the notes on.** §4.1 calls it the closest published method to our own formulation (two coupled GNNs, one propagating activation state and one propagating influence) so a difference in that row is a statement about the architecture rather than about a feature pipeline. Its TF1 API is shimmed to `compat.v1` by three registry patches (mechanical, no behavioural content), but **two of its inputs are ours rather than the authors'**: `load_data` needs a 6-d per-node feature vector and a 32-d `.emb_32` embedding and the repo ships a derivation for neither, so the driver computes six standard graph statistics (matching the shipped example's `[degree, 5 normalized floats]` shape) and a spectral embedding in place of the DeepWalk one its own comment names. Neither is the method's contribution, and the row says so.

`ctcp` is the easiest of the five and the most interesting to run beside `casflow`: its input is a plain `(id, src, dst, cas, time)` event table, its split crosses as TIME BOUNDARIES rather than as flags (so our chronological protocol is reproduced inside the repo rather than fought), and §5.3 shows it is one of only two published methods whose train-vs-test loss curves stay flat under a leak-free split while CasFlow's diverge.

**Six wired arms, one install, for epidemic control, and it is the highest-value clone in this file.** `research/epidemic_control.md` §11 lists **NetShield, DAVA and DAVA-fast** among the methods with "no public release", and all three are in [`allogn/Network-Immunization`](https://github.com/allogn/Network-Immunization), along with NetShape (Khalil KDD'14), Degree and Random. Without it, every spectral and every data-aware number in our table is our own reimplementation and nobody else's:

```bash
python -m baselines.setup_baselines --only netimm_netshield
python -m pipeline.run --dataset hospital_lh10 --task epidemic_control \
    --baselines external:netimm_netshield external:netimm_dava netshield dava
```

**Our reimplementations produce byte-identical node sets to theirs.** On `hospital_lh10` at `k=8`, our `netshield` and `external:netimm_netshield` both return `[0, 4, 6, 14, 16, 22, 28, 36]`, and our `dava` and `external:netimm_dava` both return `[7, 45, 58, 60, 61, 65, 66, 71]` [derived, 2026-08-05]. That is the cross-check §11 asks for and neither implementation is authoritative without.

Five compatibility patches are applied at setup, each a 2016-era API that no longer exists: `nx.to_numpy_matrix` (removed in networkx 3.0), `scipy.linalg.eigh(eigvals=...)` (removed in SciPy 1.12, twice) and `np.warnings` (removed in NumPy 1.24). The driver skips the repo's own `run_solver.py` for two reasons: it unpickles a NetworkX graph (a version gamble across two environments for no gain) and it then runs 100 of its OWN IC simulations to score the answer, which is work the shared referee already does.

Warning: **`epilearn` is registered as blocked because it ships NO intervention code at all.** §3.2 and §11 of the research doc both claim a `NetShield` implementation ships in it; verified by listing the whole repository tree and grepping every module, that is wrong, EpiLearn is forecasting, detection, a `NetworkSIR` forward simulator and graph transforms. The correction is recorded in §0.0 of that file rather than edited away, because every secondary source repeating the claim traces back to it.

**Five more arms, no new installs.** `finder_epi`, `gdm_epi`, `collective_influence_epi`, `explosive_immunization_epi` and `dismantling_review_epi` re-register already-wired node-removal repos under this task, sharing their critical-node twin's clone, venv, build and adapter. §2.5 maps vaccination onto `remove_node`, so a dismantler's output IS an allocation under the `vaccinate` lever, and §4.2 lists FINDER under this task explicitly. Running one under BOTH tasks and reading the two rows against each other is a comparison neither literature makes: §8.2 trap 1 says a method can win the connectivity metric and lose the epidemic one.

**Three wired arms, one install, for cascade reconstruction.** `ditto`, `ditto_dhrec` and `ditto_cri` are three entry points inside DITTO's single clone, sharing a venv through `install_name`. `ditto.py` is the KDD'23 method; `dhrec.py` and `cri.py` are that paper's own implementations of DHREC-PCDSVC and CRI: the first because the original code covers only SEIRS, the second because CRI's authors published none, so those two arms are **cross-checks on our own reimplementations** rather than new coverage:

```bash
python -m baselines.setup_baselines --only ditto
python -m pipeline.run --dataset oregon2 --task cascade_reconstruction \
    --cr-setting final_snapshot --baselines external:ditto external:ditto_dhrec
```

Warning: **`--cr-setting final_snapshot` is not optional for these three.** Verified by reading every `data.*` access in `ditto.py` and `inc/diffus.py`: DITTO conditions on the observed terminal state (`data.y[:, -1]`) and the source COUNT (`(data.y[:, 0] == 1).sum()`) and nothing else. It therefore always solves the DASH problem whatever the sweep runs, and under `partial_times` it is answering a strictly harder instance than every other arm, a row that has to say so rather than be read as a loss.

**Three more wired arms, three separate installs: the SUPERVISED imputers.** `grin`, `spin` and `deep_demixing` all fit a model on labelled histories before predicting, which is a setting none of our own arms have, so their rows are not comparable to an unsupervised decoder's without saying so, and the report has to label them:

```bash
python -m baselines.setup_baselines --only grin
python -m pipeline.run --dataset jazz --task cascade_reconstruction \
    --baselines external:grin external:spin
```

`grin` is the one to run first: DITTO uses it as the **ideal upper bound** and every `Gap` column in its Tables 4-5 is measured against it, which makes it the single most useful reference number in this literature.

All three are driven through the repo's MODEL rather than its training harness, and that trade is stated on every row: GRIN's own entry point pins `tensorflow==2.5.0` / `pytorch-lightning==1.4` / `torch==1.8` (none of which resolves today, and none of which `GRINet` needs, tracing every import in `lib/nn/layers/` gives `torch` + `einops`), SPIN's is a `tsl` Lightning experiment over hardcoded benchmarks, and Deep Demixing's `GCVAE_Trainer` wants its own pickle layout. So these are **the authors' architecture, our optimizer**, and `GWM_IMPUTE_EPOCHS` is the first number to raise before quoting any of them as parity. `baselines/drivers/imputation_common.py` holds the loading, the masked-BCE loop and the decoding all three share.

Two install hazards found by actually running them, not inferred. **SPIN needs `torch-scatter` even though nothing in SPIN imports it**: `tsl.nn.functional` does, at module load, so `from spin.models import SPINModel` dies without it; it has no universal wheel and builds from source. **Deep Demixing needs a torch built with MKL**: `CVAE_UNET_Batch` is built from torch_geometric's `GraphUNet`, whose `augment_adj` does a sparse-CSR matmul that CPU-without-MKL cannot do, which is the stock Apple Silicon wheel. It runs on Linux and on CUDA; the driver catches that specific error and says so rather than surfacing a traceback into torch_geometric. And **a SPIN OOM is a reportable result**, not a failed arm, it OOMs on Oregon2, Prost and Pol in its own published table, and our graphs sit on both sides of that ceiling.

Four repos in that set are blocked with verified reasons rather than left to fail at the first budget. Both Xiao repos (`reconstructing_cascade`, `cascade_tree_samples`) import `graph_tool`, a Boost/C++ extension distributed through conda or apt and **not pip-installable**, so it cannot go into a per-baseline venv; `cascade_tree_samples` additionally needs an unpublished Cython package. `cult` is **Python 2** (`print` statements throughout `experiments/`) and consumes a temporal interaction stream our episodes are not. `dipt` (the only method in this literature that outputs explicit propagation-tree edges) has **no public code at all**; the anonymous link in its paper is the simulator, not the model. All four are reimplemented in `coding_agent/tools/reconstruction_algorithms.py`, so the coverage survives even where the authors' code does not.

**Six wired arms, one install.** `graphsl_lpsi`, `graphsl_netsleuth`, `graphsl_ojc`, `graphsl_gcnsi`, `graphsl_ivgd` and `graphsl_slvae` are six published methods inside one pip package (JOSS 9(99):6796), so they share a clone and a venv through `install_name` rather than pulling six copies of torch. Installing any one installs all six:

```bash
python -m baselines.setup_baselines --only graphsl_lpsi
python -m pipeline.run --dataset jazz --task source_localization \
    --baselines external:graphsl_slvae external:graphsl_ivgd external:graphsl_lpsi
```

That single package covers two of the three seed papers (SL-VAE, IVGD) plus GCNSI and the three classical references, and it packages *our* Network Science (1,589 / 2,742), which [`../research/source_localization.md`](../research/source_localization.md) §6.4.1 shows is the version IVGD, SIDSL and Network Repository agree on and SL-VAE does not.

### The inverse contract

Source localization and cascade reconstruction are the two tasks whose external contract is not `G -> S`. The repo is handed a graph AND a batch of observed diffusion states and returns one source set PER OBSERVATION, so three things differ from the IM path:

1. **`run_external_baseline` takes `instances`.** Passing it selects the localization signatures of `export` (which takes the instance list where the intervention one takes a budget) and `parse_seeds` (which returns a dict keyed by episode id). A split rather than a widened signature, so the seven wired IM adapters are untouched.
2. **`localize_script` replaces `seed_script`.** The canned Strategy implements `localize`, and it finds its row by `observation_key` (the thresholded infected set) because `localize(graph, observation, budget)` is handed no episode id and the arm makes two passes over different pools.
3. **One batched invocation, not N.** Every one of these repos is a library that loops internally, so N subprocess launches would dominate the runtime we are trying to measure. Both instance pools go over together, flagged `is_train`.

Cascade reconstruction is the same shape one level up, with two additions. A decoder is handed whole MASKED HISTORIES and returns a whole trajectory per cascade, so `_collect_trajectories` replaces `_collect_sources` and `reconstruct_script` replaces `localize_script`, keyed by the nodes the decoder was actually SHOWN (`observation.infected`) because that is the only identity available on both sides of the boundary. The addition is that **the returned parents may be null**, and usually are: only DIPT emits explicit tree edges and DIPT has no public code, so every runnable repo produces per-step node STATES and the tree is built afterwards by the same `finalize` rule the library decoders use. That is deliberate, a Path Precision difference between two rows should be a difference in their inferred TIMES, not in whichever tree-building trick one of them happened to ship.

The second addition is `supervised=True` on the export, which ships the selection split's true histories so a trained imputer has something to fit. Two invariants keep that from being a leak and **both are asserted rather than trusted**: `is_train` marks the selection pool only, and every evaluation row's trajectory is exactly zero, the export raises if it is ever not. The rule is a SPLIT comparison (`registry.training_rows`) rather than a first-occurrence one, and that distinction is load-bearing: the two pools come from disjoint splits so no episode id ever repeats, and a first-occurrence rule silently marks every row trainable. The same rule now backs `graphsl`'s export, where it had been marking the evaluation episodes trainable and letting GCNSI, IVGD and SL-VAE fit on the rows they were then scored against.

**Labels cross the boundary, and that is not a leak.** These methods tune a threshold or fit weights on labelled data exactly as our outer loop selects a program on labelled episodes, so the driver gives `train()` the SELECTION split and runs a label-free prediction pass on everything. What it never calls is their `test()`/`infer()`: those score against labels internally and return only an aggregate `Metric`, so no per-instance prediction escapes them. The prediction lines are reproduced in `drivers/graphsl_driver.py` from each method's own `test` body.

**Two deviations to know before quoting a number.** GraphSL cuts its score vector at a tuned threshold; we take the TOP-K of the same vector, because our table compares every arm at matched `k`. And the driver defaults to `graphsl_epochs = 50` for the three learned methods, far below the papers' regime: raise it before treating a GCNSI/IVGD/SL-VAE row as anything but a smoke result. Two more traps apply to every entry (§8.4): the **source fraction is not standardized** (10% uniform-random in SL-VAE, first 5% by infection time in SL-Diff, top 10% by influence time in SIDSL), and **nothing in that literature evaluates under IC or LT**, which §11 calls the single biggest comparability gap in the file.

**For critical node detection, `dismantling_review` is the route that does not work in a uv venv**: its harness imports `graph_tool`, which pip cannot install, so it stays registered and unrunnable. `gdm` runs on its own instead: its adapter loads GDM's `models/GAT.py` by file path with GDM's published four-feature checkpoint, converting the 2019 PyG 1.x state dict through `baselines/gdm_support.py` (rewired 2026-09-04). The paragraph below records the original reasoning. **For critical node detection, wire `dismantling_review` first.** It is the Artime et al. survey's harness rather than a method, and it already drives CI, CoreHD, GND, EI, MinSum, FINDER and GDM behind one interface, so it is one adapter instead of seven, and it is the only practical route to a FINDER number at all, since [`../research/critical_node_detection.md`](../research/critical_node_detection.md) §11 records that FINDER publishes its real-network results as heatmaps only, has no arXiv version, and the `results/` directory its README advertises does not exist in `master`.

Two traps apply to every entry in that group before any number is quoted (§8.2): almost all of them run on the **largest connected component** of the input silently (trap 7), and several ship a **reinsertion** pass that makes `X` and `X+R` different methods cited under one name (trap 2). A third is specific to `gnd`: its contribution is *cost*-weighted dismantling, so scoring its cost-optimal set on a cardinality budget is unfair in both directions (trap 4).

Our own `tools/dismantling_algorithms.py` reimplements CI, CoreHD, EI, GND and the Min-Sum pipeline shape in Python. Those are honest condition-1 arms and are **not** substitutes for the authors' code: each docstring states where it deviates (greedy cover instead of GND's LP 2-approximation, one EI regime instead of two, greedy decycling instead of Min-Sum's 1RSB message passing). Wire the repo before quoting a comparison to the paper.

---

## Usage

```bash
# See the registry and what is installed
python -m baselines.setup_baselines --list

# Fetch and install (each gets its own venv; ~10 min, several hundred MB)
python -m baselines.setup_baselines --all
python -m baselines.setup_baselines --only moeim touplegdd

# Run one directly, outside the pipeline
python -m baselines.run_baseline --name moeim \
    --data-dir results/influence_maximization/ba/default/data --budget 5 --diffusion-model IC
```

In the pipeline, external baselines are just another `--baselines` entry:

```bash
# Everything: classical library + every installed external repo + our conditions
python -m pipeline.run --dataset ba --baselines all

# Only the external published methods
python -m pipeline.run --dataset ba --baselines all-external

# Hand-picked
python -m pipeline.run --dataset jazz \
    --baselines celf_pp imm external:moeim external:touplegdd
```

`--baselines` accepts: a library algorithm name, `external:<name>`, or the aliases `all` / `all-classical` / `all-external`. **`all` includes only external baselines that are actually installed**, printing what it skipped, so a fresh checkout runs cleanly instead of failing on missing repos.

`--baseline-timeout` (default 3600s) kills a runaway external subprocess. If a baseline fails, the pipeline logs `SKIPPED: <reason>` and continues rather than aborting the whole sweep.

---

## Where results land

```
results/<task>/<dataset>/<run>/
├── agent/<budget>/<arm>.json          our conditions 1-6
├── baselines/<budget>/<name>.json     external published baselines (condition 7)
└── baselines/_runs/<name>/<budget>/   raw stdout/stderr + artifacts per run
```

External results carry an extra `external` block recording the seed set, the repo, the paper, and `selection_seconds` (time the external code spent choosing seeds, separate from our scoring time). Everything else (`reward`, `referee_reward`, `spread_pct`, `timeline`, `condition`) is the standard schema, so plots and the report read all arms uniformly.

Two figures compare directly: `ours_vs_baselines.png` and `ours_vs_baselines_pct.png` draw our method heavy and solid, published methods dashed, classical algorithms thin and dotted, across the whole budget sweep.

---

## Adding a new baseline

Add one `ExternalBaseline(...)` entry to `baselines/registry.py` with three callables:

- `export(graph, work_dir, budget, diffusion_model) -> dict`: write our graph in whatever format the repo expects (helpers: `write_edgelist`)
- `command(work_dir, budget, diffusion_model, extras, graph) -> list[str]`: argv
- `parse_seeds(work_dir, stdout, budget) -> list[int]`: read the seeds back (helper: `parse_seed_integers` scrapes the last bracketed int list from stdout)

`work_dir` is **absolute** by the time it reaches `export` and `command` (`run_external_baseline` resolves it), and it has to be: the child runs with `cwd=spec.directory`, so a relative `results/...` path embedded in argv resolves inside the repo clone. That one slip took 26 rows (every driver-style adapter across four tasks) out of the first cluster sweep with `can't open file '<repo>/results/.../driver.py'`. Two upstream entry points have no CLI at all and are driven through a generated runner instead: `celf_glie.py` loops six hardcoded graphs at `seed_size = 100` and never prints its seeds, and DiffIM's notebooks draw their training seed sizes from a band that is empty on any graph under 1,100 nodes.

Nothing else needs to change: the arm spec, results routing, plots, and report pick it up automatically.

---

## Condition 9: published LLM algorithm-discovery systems

A third layer, and a different contract from the two above. These repos do not return a seed set: each one **searches the space of programs** with its own LLM loop, and what crosses the process boundary is the best PROGRAM it found. The rule is the same as condition 7's, applied one level up: their code, their prompts, their published defaults, our referee. None of them ever sees the world model. Everything is in `baselines/discovery.py`; the registry entries carry `kind="discovery"` and `task="*"` (they serve every task).

What we hand each system, identically:

1. **A task statement** (`discovery.problem_statement`): the objective, the instance facts (N, arcs, dynamics, k, horizon, the lever), the allowed imports, and the fitness definition.
2. **An initial program**: the same degree heuristic for every system, one function per task: `select_seeds(graph, k)`, `select_removals(graph, k, outbreak)`, `select_blockers(graph, k, rumour, lever)`, `select_doses(graph, k, outbreak, lever)`, `localize(graph, observation, k)`, `reconstruct(graph, observation, horizon)`, `predict(graph, observation)`. Adaptive IM runs the static `select_seeds` contract, which is the non-adaptive side of its own gap table.
3. **A fitness subprocess**: `python -m baselines.score_program --context <work_dir>/discovery_context.json --program candidate.py`, run under OUR interpreter from inside their venv. It wraps the candidate through `discovery.program_script` and scores it with `run_experiment` on the plain Monte Carlo simulator under the arm's own `ExperimentConfig` (same outbreak, lever, splits and seed count as the arms it is compared against). Fitness is non-negative and higher is better for every task (nodes saved for a minimizing spread task, `1/(1+MSLE)` for prediction); the raw quantity travels beside it under its own name. Their venvs never import this package.

The best program comes back through the entry's `parse_program`, the pipeline wraps it with the same `program_script`, and it flows through the identical executor, validation, referee replay and results JSON as every other arm. `external.info` in the result carries the framework's own bookkeeping (its internal score, iteration or sample count).

| Baseline | Paper | Status | What runs |
| --- | --- | --- | --- |
| `openevolve` | software (Sharma 2025), Apache-2.0 | setup | `openevolve-run.py` at `configs/default_config.yaml` defaults (100 iterations, MAP-Elites islands) |
| `codeevolve` | EMNLP 2026 Findings, arXiv 2510.14150 | setup, Python 3.13 | `codeevolve` CLI at the shipped template's defaults (50 epochs, 3 islands, meta-prompting) |
| `llamea` | IEEE TEVC 2025, arXiv 2405.20132 | setup | `llamea.LLaMEA` at constructor defaults (5+5, 100 evaluations) |
| `eoh` | ICML 2024 oral, arXiv 2401.02051 | setup | `eoh.EoH` (v0.2 API) at library defaults (pop 5, 20 generations) |
| `reevo` | NeurIPS 2024, arXiv 2402.01145 | setup, one transport patch | `main.py` at `cfg/config.yaml` defaults (max_fe 100, pop 10, init 30) |
| `mcts_ahd` | ICML 2025, arXiv 2501.08603 | setup | `main.py` at `cfg/config.yaml` defaults (max_fe 1000) |
| `llm4ad_funsearch` | Nature 2024, port in arXiv 2412.17287 | setup (shares `llm4ad`) | `llm4ad.method.funsearch.FunSearch` at constructor defaults (20 samples) |
| `llm4ad_hillclimb` | PPSN 2024, in LLM4AD | setup (shares `llm4ad`) | `llm4ad.method.hillclimb.HillClimb` at constructor defaults (20 samples) |
| `deepevolve` | arXiv 2510.06056, no license | setup | `deepevolve.py` at `configs/config.yaml` defaults (50 iterations, retrieval ON) |
| `llm4ad_next` | none | blocked | rewritten prompts, a second coder call, no progressive widening: not any method's original |
| `gs4co` | ICML 2024 | blocked | not LLM-based: PPO symbolic regression of a SCIP branching rule |

Three things the table cannot show. **Sample budgets differ by design**: every system runs at its published defaults, so `external.info` and the wall time are as much the result as the score, and our own `evolve` arm spends about 21 programs per budget against MCTS-AHD's 1000. **DeepEvolve retrieves**: it runs an OpenAI-Agents research loop with hosted web search, which our gateway serves (measured 2026-09-01), so its rows carry literature the others never see; the repo also has no license file. **ReEvo is patched**: it requests its initial population as one `n=30` call whenever the model name contains "gpt", and the gateway returns one choice for `n > 1` with no error, so `utils/llm_client/base.py` is patched to replicate the request instead (transport only).

```bash
python -m baselines.setup_baselines --only openevolve codeevolve llamea eoh reevo mcts_ahd llm4ad_funsearch deepevolve
python -m pipeline.run --task influence_maximization --dataset netscience --baselines all-discovery
python -m pipeline.run --task source_localization --dataset jazz --baselines discovery:eoh discovery:reevo
python -m baselines.check_discovery --live eoh      # smoke one adapter end to end
```

`all` and `all-external` never include these: each is a full LLM search that runs for hours, so they are opted into by name or with `all-discovery`. `--llm-models` fans them out per model like the synthesis arms. Every entry carries a timeout floor (6 h, 12 h for MCTS-AHD and DeepEvolve) that overrides a smaller `--baseline-timeout`. Work dirs are `results/<task>/<dataset>/<run>/baselines/_runs/<name>/<budget>/`: the context, the glue, every scored candidate under `candidates/`, and the framework's own output tree.

