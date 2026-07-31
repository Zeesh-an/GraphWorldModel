# Baselines

Everything we compare against, in two layers:

1. **Our own library** (`coding_agent/tools/algorithms.py`) — 41 classical IM algorithms reimplemented in Python against our primitives. Fast, always available, no setup. These are **condition 1 (Pure GA)**.
2. **External published repos** (this folder) — the original authors' code for published methods, fetched into `baselines/external/`. These are **condition 7 (Published baseline)**.

For the literature itself — what each method published, and which of their numbers are comparable to ours — see [`../research/influence_maximization.md`](../research/influence_maximization.md).

---

## The one contract

Every external baseline reduces to:

```
(graph, budget, diffusion_model)  ->  list[int] seed set
```

We then score that seed set with **our own Monte Carlo referee**, the same one that judges our own arms. This is deliberate and it is the single most important design decision here.

A paper's reported spread depends on its simulator, its edge probabilities, its MC count, its horizon, and sometimes its own version of the graph (see `research/influence_maximization.md` §6.3 — "Cora-ML" alone means two different graphs). Those numbers are **not** comparable to ours. Their _seed set_ is. Running every method's seeds through one referee is the only apples-to-apples comparison available, and it means external baselines need no special result format: the seeds are wrapped as a canned `Strategy` and flow through the identical executor, validation, rollout, MC replay, and results JSON as every other arm.

---

## Where each baseline lives

Every method is in exactly one of three states. **Some methods exist in both layers** — our library has a simplified Python `imm`, `tim`, `ssa`, and the external registry has the authors' originals. Those are different arms (`baseline_imm` vs `external_imm`) and both can run in the same sweep.

| Baseline        | Kind      | Our library?  | External folder                 | Status                        |
| --------------- | --------- | ------------- | ------------------------------- | ----------------------------- |
| **IMM**         | classical | ✅ simplified | `baselines/external/imm/`       | 📥 manual fetch (SourceForge) |
| **TIM / TIM+**  | classical | ✅ simplified | `baselines/external/tim/`       | 📥 manual fetch (SourceForge) |
| **OPIM-C**      | classical | ❌            | `baselines/external/opim/`      | ⚙️ setup (C++ `make`)         |
| **SubSIM**      | classical | ❌            | `baselines/external/subsim/`    | ⚙️ setup (C++ `make`)         |
| **SSA / D-SSA** | classical | ✅ simplified | `baselines/external/ssa/`       | ⚙️ setup (C++ `make`)         |
| **MOEIM**       | learned   | ❌            | `baselines/external/moeim/`     | ⚙️ setup (pure Python)        |
| **ToupleGDD**   | learned   | ❌            | `baselines/external/touplegdd/` | ⚙️ setup (pretrained ckpt)    |
| **DeepIM**      | learned   | ❌            | `baselines/external/deepim/`    | ⚙️ setup + `deepim_data.py`   |
| **GCOMB**       | learned   | ❌            | —                               | ⛔ blocked — Python 2.7       |
| **IMINFECTOR**  | learned   | ❌            | —                               | ⛔ blocked — needs cascades   |
| **PIANO**       | learned   | ❌            | —                               | ⛔ blocked — no public code   |
| **OIM**         | classical | ❌            | —                               | ⛔ blocked — no public code   |

Also in our Python library only (no external counterpart needed or available): CELF, CELF++, degree-discount, PMIA/MIA, LDAG, SIMPATH-style SP1M, IRIE, StaticGreedy, SKIM, RIS, community-IM, CoFIM, VoteRank, k-shell, collective influence, and the metaheuristics (simulated annealing, hill climbing, GA) — 36 in total, listed by `python -m pipeline.run --help` under `--baselines`.

`baselines/external/` is **gitignored** — several repos are 100 MB+ (MOEIM alone is ~128 MB) and carry their own licences, so every baseline is fetched by `setup_baselines.py` rather than vendored into this repo. Nothing third-party is committed here; only the registry, adapters, and this README are ours.

### What each one is

**MOEIM** — _Many-Objective Evolutionary Influence Maximization_, GECCO 2024. A multi-objective evolutionary algorithm optimising spread, seed-set size, community coverage, fairness, budget, and time simultaneously, with graph-aware mutation operators and smart initialization. **The current SOTA on Jazz and Cora-ML — it beats DeepIM.** Pure Python, no training, no GPU, and its `--k` is already a fraction of |V| exactly like our `--budget-pcts`. Its `setting2` is the configuration it used for its DeepIM comparison. The most directly runnable external baseline we have. [paper](https://arxiv.org/abs/2403.18755) · [repo](https://github.com/eliacunegatti/MOEIM)

**ToupleGDD** — _Three coupled GNNs + Double DQN_, IEEE TCSS 2023. Deep RL over graph embeddings; picks seeds sequentially by Q-value. Ships **pretrained checkpoints** (`tripling.ckpt`, `s2vdqn.ckpt`), so it runs inference-only with no training. It was trained on sub-300-node random graphs and designed to generalize to million-node graphs — the same generalization claim our world model makes, which makes it a pointed comparison. Needs torch. [paper](https://arxiv.org/abs/2210.07500) · [repo](https://github.com/Dtrycode/ToupleGDD)

**OPIM-C** — _Online Processing for Influence Maximization_, SIGMOD 2018. Near-optimal Reverse Influence Sampling with anytime guarantees. C++ reference implementation. Our library's `imm` and `ris_basic` are simplified Python stand-ins; this is the faithful, fast version. [paper](https://dl.acm.org/doi/10.1145/3183713.3183749) · [repo](https://github.com/tangj90/OPIM)

**SSA / D-SSA** — _Stop-and-Stare_, SIGMOD 2016. RIS with the tightest published sample-complexity bounds. C++. [paper](https://arxiv.org/abs/1605.07990) · [repo](https://github.com/hungnt55/Stop-and-Stare)

**IMM** — _Influence Maximization in Near-Linear Time: A Martingale Approach_, SIGMOD 2015. **The** classical reference point; essentially every IM paper since 2015 compares against it, and it is the baseline in DeepIM's table. Our library's `imm` is a simplified Python reimplementation — fine as a condition-1 arm, but the authors' C++ original is what published numbers come from. Tang et al. release a **tarball on SourceForge, not a git repo**, so `setup_baselines` cannot clone it and prints manual instructions instead. A third-party git reimplementation exists at [gdelpuente/IMM](https://github.com/gdelpuente/IMM) (not author-endorsed). [paper](https://dl.acm.org/doi/10.1145/2723372.2723734) · [SourceForge](https://sourceforge.net/projects/im-imm/)

**TIM / TIM+** — SIGMOD 2014. IMM's predecessor; the paper that made RIS practical. Same SourceForge-tarball situation. [paper](https://arxiv.org/abs/1404.0900) · [SourceForge](https://sourceforge.net/projects/timplus/)

**SubSIM** — _Influence Maximization Revisited: Efficient Sampling with Bound Tightened_, SIGMOD 2020. Sublinear-time RIS, and the third of the three RIS methods in DeepIM's comparison table (IMM / OPIM / SubSIM). **Not in our Python library at all**, so this repo is the only way to run it. [paper](https://dl.acm.org/doi/10.1145/3318464.3389740) · [repo](https://github.com/qtguo/subsim)

### Why five are blocked

These are **not** "not done yet" — each has a concrete, stated reason:

**DeepIM** (ICML 2023) — the public repo's `data/` folder **ships empty**. `main/utils.py` loads `data/<name>_25c.SG` pickles that were never published, and the model needs those preprocessed SparseGraph files plus a full training run before it can select seeds. Unblocking means obtaining the `.SG` files from the authors or reimplementing their preprocessing. **Its published tables are transcribed in `research/influence_maximization.md` §5.1 and are directly comparable on Jazz and Power Grid**, so a numeric comparison against DeepIM is available from the paper even while its code is blocked.

**GCOMB** (NeurIPS 2020) — requires **both** Python 2.7 and Python 3 environments (`requirements_python2.7.txt` + `requirements_python3.txt`) plus a multi-stage supervised-then-RL pipeline driven by shell scripts. Python 2.7 is end-of-life. Unblocking means containerising it.

**IMINFECTOR** (TKDE 2020) — **structurally inapplicable**, not an engineering gap. IMINFECTOR is model-free: it learns from _observed diffusion cascades_ and never assumes IC/LT. Our BA/SBM/ER graphs are synthetic and Jazz/Power Grid/NetScience ship no cascade logs, so there is nothing for it to train on. It can only run on Digg/Weibo/MAG — which our pipeline cannot simulate at that scale anyway.

**PIANO** (IEEE TCSS 2022) — **no public code release**. The paper advertises "pretrained PIANO models" but publishes no repository or download link, and no official implementation was found. Tellingly, ToupleGDD's authors state they had to _revise the S2V-DQN code themselves_ to obtain a PIANO baseline — which is why PIANO's numbers differ between papers. Unblocking means reimplementing from the paper.

**OIM** (KDD 2015) — **no public code release** (Lei, Maniu, Mo, Cheng & Senellart). It is also a _different problem setting_: OIM assumes repeated campaigns with feedback that updates edge weights between rounds, whereas ours is single-shot seeding. Even with code, the comparison would need care.

All five appear in DeepIM's published tables, which are transcribed in [`../research/influence_maximization.md`](../research/influence_maximization.md) §5.1 — so numeric comparison against them is available from the paper on Jazz and Power Grid even with their code blocked.

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
python -m pipeline.run --dataset ba --baselines all --compare

# Only the external published methods
python -m pipeline.run --dataset ba --baselines all-external --compare

# Hand-picked
python -m pipeline.run --dataset jazz --compare \
    --baselines celf_pp imm external:moeim external:touplegdd
```

`--baselines` accepts: a library algorithm name, `external:<name>`, or the aliases `all` / `all-classical` / `all-external`. **`all` includes only external baselines that are actually installed**, printing what it skipped — so a fresh checkout runs cleanly instead of failing on missing repos.

`--baseline-timeout` (default 3600s) kills a runaway external subprocess. If a baseline fails, the pipeline logs `SKIPPED — <reason>` and continues rather than aborting the whole sweep.

---

## Where results land

```
results/<task>/<dataset>/<run>/
├── agent/<budget>/<arm>.json          our conditions 1-6
├── baselines/<budget>/<name>.json     external published baselines (condition 7)
└── baselines/_runs/<name>/<budget>/   raw stdout/stderr + artifacts per run
```

External results carry an extra `external` block recording the seed set, the repo, the paper, and `selection_seconds` (time the external code spent choosing seeds, separate from our scoring time). Everything else — `reward`, `mc_reward`, `spread_pct`, `timeline`, `condition` — is the standard schema, so plots and the report read all arms uniformly.

Two figures compare directly: `ours_vs_baselines.png` and `ours_vs_baselines_pct.png` draw our method heavy and solid, published methods dashed, classical algorithms thin and dotted, across the whole budget sweep.

---

## Adding a new baseline

Add one `ExternalBaseline(...)` entry to `baselines/registry.py` with three callables:

- `export(graph, work_dir, budget, diffusion_model) -> dict` — write our graph in whatever format the repo expects (helpers: `write_edgelist`)
- `command(work_dir, budget, diffusion_model, extras, graph) -> list[str]` — argv
- `parse_seeds(work_dir, stdout, budget) -> list[int]` — read the seeds back (helper: `parse_seed_integers` scrapes the last bracketed int list from stdout)

Nothing else needs to change: the arm spec, results routing, plots, and report pick it up automatically.
