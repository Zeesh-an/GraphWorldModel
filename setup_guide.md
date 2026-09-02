# Cluster Setup Guide

Every command needed to take a fresh GPU cluster account from nothing to the full experiment: clone, environment, gateway credentials, datasets, external baselines (conditions 7 and 9), self-checks, a smoke run, and the real submissions. Written against the code as of 2026-09-02; every path, flag and file name below was checked against the repository rather than recalled. Run everything from the repository root unless a step says otherwise.

What you end up with:

```
~/GraphWorldModel/
├── venv/                 the project environment (gitignored; sbatch activates it by default)
├── .env                  gateway credentials (gitignored, chmod 600)
├── data/raw/<dataset>/   downloaded graphs and cascade corpora (gitignored)
├── baselines/external/   one clone plus one venv per published repo (gitignored)
├── results/<task>/<dataset>/<run>/   everything a run writes (gitignored)
└── logs/                 SLURM .out files (gitignored except .gitkeep)
```

## 1. What the cluster has to provide

- A SLURM scheduler and GPU nodes. `sbatch/pipeline.sbatch` submits itself and asks for `--gres=gpu:1` only when the run trains or uses the world model or a learned external repo.
- Internet access from the login node. Cloning, `uv`'s managed Python downloads, dataset downloads and baseline installs all happen there. Compute nodes need to reach only the LLM gateway host, and only for arms that call an LLM.
- Build tools for the C++ baselines: `git`, `gcc`/`g++`, `make`, `cmake`. `decycler` additionally needs the Boost headers; `finder` needs CPython 3.7, which `uv` cannot fetch any more. Both are known, recorded failures, not setup mistakes.
- Whatever `module load` your site uses for a compiler and CUDA. Names differ per cluster; `module avail gcc cuda boost` shows yours. The pipeline itself needs no module: `torch` from PyPI ships its own CUDA runtime.

## 2. Clone

```bash
cd ~
git clone https://github.com/Zeesh-an/GraphWorldModel.git
cd ~/GraphWorldModel
```

`~/GraphWorldModel` is the path every command in `test_commands.md` assumes. The sbatch script resolves the checkout from its own location, so another path works too, as long as you submit from inside the checkout. `CLAUDE.md` is gitignored (it is a local Claude Code instruction file), so it is absent on a fresh clone; copy it from another machine if you use Claude Code on the cluster.

## 3. Python environment

The project uses `uv` for everything and a venv named `venv`, which is the name the sbatch script activates when no environment is active (`VENV=venv` default). There is no `pyproject.toml`; dependencies are `requirements.txt`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"        # puts uv on PATH for this shell; new shells pick it up from your profile
uv --version

uv venv venv --python 3.11
source venv/bin/activate
uv pip install -r requirements.txt
uv pip install pytest ruff            # tests and lint only
python -c "import torch, ndlib, networkx, scipy; print(torch.__version__, 'cuda', torch.cuda.is_available())"
```

`cuda False` on a login node is normal. Confirm on a GPU node once:

```bash
srun --gres=gpu:1 --mem=8G --time=00:05:00 --pty bash -c 'source venv/bin/activate && python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"'
```

On the cluster, use the activated `venv` and plain `python -m ...` as every command here does. `uv run` looks for a project `.venv` and is not used on the cluster.

## 4. Gateway credentials

The coding-agent arms and the condition-9 discovery systems call the lab's OpenAI-compatible gateway. Three variables, read from `.env` by `pipeline.run` and `coding_agent.run` at start-up (`load_dotenv`) and passed on to every external subprocess by name, never by value:

```bash
cat > .env <<'EOF'
GATEWAY_BASE_URL=https://<gateway-host>/v1
CHATGPT_GATEWAY_TOKEN=<token used for gpt-* models>
CLAUDE_GATEWAY_TOKEN=<token used for claude-* models>
EOF
chmod 600 .env
```

The base URL must be exactly `https://<host>/v1` with no further path: two of the discovery systems (EoH, LLM4AD) hard-code `/v1/chat/completions` onto the host. The token is chosen by model family (`claude-*` uses the Claude token, everything else the ChatGPT token), and the default model everywhere is `gpt-5.6-sol` (`--llm-model`, `LLM_MODEL`; `--llm-models` / `LLM_MODELS` is the opt-in multi-model sweep). Verify without printing the token:

```bash
set -a; . ./.env; set +a
curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer $CHATGPT_GATEWAY_TOKEN" "$GATEWAY_BASE_URL/models"   # expect 200
```

## 5. Datasets

Every graph loader in `data/datasets/` downloads itself on first use: `--dataset <name>` calls `data.datasets.<name>.download_<name>()` and caches the file under `data/raw/<name>/`. Because compute nodes usually have no internet, pre-download on the login node every graph the submissions use. The eight submissions in `test_commands.md` section 3 use `netscience` (IM, adaptive IM), `power_grid` (critical node detection), `jazz` (source localization), `email_eu_core` (influence blocking), `infectious` (cascade reconstruction), `primary_school` (epidemic control) and the CasFlow APS corpus (cascade prediction):

```bash
python -c "from data.datasets.infectious import download_infectious; from data.datasets.jazz import download_jazz; from data.datasets.netscience import download_netscience; from data.datasets.power_grid import download_power_grid; from data.datasets.email_eu_core import download_email_eu_core; from data.datasets.primary_school import download_primary_school; print(download_infectious()); print(download_jazz()); print(download_netscience()); print(download_power_grid()); print(download_email_eu_core()); print(download_primary_school())"
```

Any other graph follows the same pattern; this loop pre-downloads a list:

```bash
python -c "import importlib; names = ['cora_ml', 'facebook', 'ca_grqc']; [print(getattr(importlib.import_module(f'data.datasets.{n}'), f'download_{n}')()) for n in names]"
```

### 5.1 Cascade prediction: the CasFlow bundle (manual)

The CasFlow archive holds the Weibo, Twitter and APS corpora behind a Google Drive interstitial that no script can pass, so `casflow_aps`, `casflow_weibo` and `casflow_twitter` raise with instructions until the files are in place. `gdown` gets past the interstitial:

```bash
cd ~/GraphWorldModel && source venv/bin/activate
uv pip install gdown
gdown 1o4KAZs19fl4Qa5LUtdnmNy57gHa15AF- -O /tmp/casflow.zip
unzip -o /tmp/casflow.zip -d /tmp/casflow
find /tmp/casflow -name dataset.txt          # one per corpus, inside a folder named after it

for corpus in aps weibo twitter; do
  source_file=$(find /tmp/casflow -ipath "*/${corpus}/dataset.txt" | head -1)
  mkdir -p "data/raw/casflow/${corpus}"
  cp "$source_file" "data/raw/casflow/${corpus}/dataset.txt"
done

python -c "from data.datasets.casflow_bundle import bundle_path; print(bundle_path('aps'))"
```

Only `aps` is needed for the section 3 submission (`DATASET=casflow_aps`); copy all three while the archive is open. If Drive blocks `gdown`, the Baidu mirror (extract code `1msd`) is in `data/datasets/casflow_bundle.py`. Use a scratch directory instead of `/tmp` if your site cleans it aggressively.

### 5.2 Other manual and known-broken corpora

- `weibo` (the 1.8M-node follower graph, not the CasFlow cascades): AMiner requires registration. Register at https://www.aminer.cn/influencelocality, download `weibo_network.tar.gz`, extract `weibo_network.txt` into `data/raw/weibo/`.
- `taoke`: its only source (`Lucas-PJ/CasTemp-ALGO`, `Taoke.zip`) returns 404; the loader stays for when the file resurfaces.
- `digg`: `datasets.syr.edu` serves an expired TLS certificate, so `urlretrieve` fails. Fetch the zip by hand if you need it and place it where `data/datasets/digg.py` looks.
- `orkut`, `livejournal`, `pokec`, `higgs_twitter`, `youtube`, `twitter`, `weibo`: load fine but are far beyond what the NDlib rollout path simulates in reasonable time. Scalability targets, not day-one datasets.

## 6. External baselines: conditions 7 and 9

One registry, `baselines/registry.py`, drives every published repo. Setup clones each into `baselines/external/<name>/`, applies its recorded patches, creates a per-repo venv at the pinned Python (uv fetches the interpreter), installs its requirements and runs its build step. Do this on the login node; it needs internet and a compiler.

```bash
python -m baselines.setup_baselines --list      # every entry: kind, status, wired, installed
python -m baselines.setup_baselines --all       # everything not blocked; tens of minutes, several GB
python -m baselines.setup_baselines --list      # confirm `installed True` on what you need
```

`--all` now includes the nine condition-9 algorithm-discovery systems (`openevolve`, `codeevolve`, `llamea`, `eoh`, `reevo`, `mcts_ahd`, `llm4ad_funsearch`, `llm4ad_hillclimb`, `deepevolve`), which need Python 3.11, 3.12 and 3.13 venvs. To install only those:

```bash
python -m baselines.setup_baselines --only openevolve codeevolve llamea eoh reevo mcts_ahd llm4ad_funsearch deepevolve
```

What to expect from the last cluster rounds (`test_commands.md` section 0 has the full table):

- `finder` and `finder_epi` fail: `tensorflow-gpu==1.14.0` needs CPython 3.7, which uv no longer ships. They are already removed from the submissions.
- `decycler` fails unless Boost headers are installed (`module load boost` on most sites, then re-run `--only decycler`).
- `imm` and `tim` print manual instructions: Tang et al. publish SourceForge tarballs, not git repos, and neither has an adapter, so they are not in any submission. The library's Python `imm` / `tim` are the condition-1 stand-ins.
- `rl4im` installs on x86_64 only (its `torch==1.7.0` has no arm64 wheels); `adaptiveim` is patched at setup from x86 inline assembly to `std::chrono`.
- DeepIM needs no data step of yours: its adapter builds the `.SG` training files it needs from our graph store at export time (`baselines/deepim_data.py` is the standalone form of that step).
- A repo that fails to build is skipped at run time with `SKIPPED: <reason>` in the log and a `.skipped.json` marker; the sweep continues.

Re-running setup is idempotent: patches test their own marker, and a venv at the wrong Python version is rebuilt rather than reused.

## 7. Self-checks and tests

Run these once after setup; each is a few minutes on the login node and needs no GPU. The last two call the gateway.

```bash
python -m pytest tests -q                       # 295 tests, about 5 s
python -m coding_agent.check_containment        # critical node detection contract
python -m coding_agent.check_source_localization
python -m coding_agent.check_influence_blocking
python -m coding_agent.check_cascade_reconstruction
python -m coding_agent.check_epidemic_control
python -m coding_agent.check_cascade_prediction
python -m coding_agent.check_adaptive
python -m data.check_remove_semantics
python -m baselines.check_discovery             # condition 9 scorer end to end, no LLM
python -m baselines.check_discovery --live codeevolve   # one discovery system at a smoke budget (Linux only; unverified on macOS)
```

`ruff check .` should be clean (the cloned repos under `baselines/external/` are excluded by `ruff.toml`).

## 8. Smoke run before the first real submission

A synthetic run on the login node exercises data generation, the oracle coding-agent arm (three LLM calls), the referee replay, plots and the report. It takes about ten minutes, almost all of it gateway latency: a `gpt-5.6-sol` reply takes one to three minutes, which is also why the real submissions are measured in hours.

```bash
python -m pipeline.run --task influence_maximization --dataset ba --run smoke \
  --num-graphs 2 --syn-nodes 60 --budget-pcts 10 --baselines high_degree --arms evolve_free@oracle \
  --outer-iters 2 --mc-runs 20 --n-samples 20 --compare
ls results/influence_maximization/ba/smoke/           # data/ agent/ plots/ report.md pipeline.json
```

Then see what a real submission would do without submitting it:

```bash
DATASET=netscience DRY_RUN=1 ./sbatch/pipeline.sbatch
```

The dry run prints the exact `sbatch` line (memory, time, `--gres` only if needed) and the exact `python -m pipeline.run` command. Every pipeline flag is an environment variable of the sbatch script; the header of `sbatch/pipeline.sbatch` documents all of them, and the rules are: list values are space-separated inside quotes, the literal `none` empties a list flag, booleans are `1` / `0`.

## 9. The real submissions

All of them are in `test_commands.md`: section 3 is one submission per task (baselines plus the oracle arm, no world model), section 6 is the ladder with the world model, section 8 is the condition-9 split. The two flagship jobs, influence maximization on netscience under IC and LT with the oracle arm, the world-model arm, every classical and published baseline, and the nine discovery systems:

```bash
TASK=influence_maximization \
DATASET=netscience \
RUN=testrun_ic \
RUN_JOBID=0 \
DIFFUSION_MODEL=IC \
BASELINES="high_degree degree_discount pagerank_seeds imm voterank random_seeds \
external:opim external:ssa external:subsim \
external:touplegdd external:deepim external:moeim external:glie \
all-discovery" \
ARMS="evolve_free@oracle evolve_free@world_model" \
LLM_MODEL=gpt-5.6-sol \
WM_MODEL=sage HEAD=structured \
GEN_ACTION_OPS="add_node remove_node" \
EVALUATOR=oracle \
BUDGET_PCTS="1 5 10 20" \
HORIZON=10 MC_RUNS=200 OUTER_ITERS=20 N_SAMPLES=200 \
COMPARE=1 CREDIT=1 FORCE=1 BASELINE_TIMEOUT=43200 STRATEGY_TIMEOUT=1800 \
GRES=gpu:1 MEM=64G TIME=7-00:00:00 \
./sbatch/pipeline.sbatch
```

Submit the LT twin with `RUN=testrun_lt DIFFUSION_MODEL=LT` and everything else identical (it is written out in `test_commands.md` section 6). The week of wall clock is for the discovery systems: each runs one full search per budget point at its published defaults, sequentially, and MCTS-AHD alone spends 1000 evaluations per budget. Drop `all-discovery` from `BASELINES` for the ladder alone (the IC run then took about 12 hours, most of it DeepIM training per budget), and run the discovery rows afterwards into the same `RUN` with section 8's split commands.

Rules that decide whether a run is comparable and resumable:

- `COMPARE=1` always. Each arm's own `reward` comes from its own evaluator; only the shared ground-truth replay (`mc_reward`) is comparable across rows.
- `RUN_JOBID=0` gives a stable `results/<task>/<dataset>/<run>/` that reruns append to. Without it the job id is appended and nothing is ever overwritten.
- Resume by resubmitting the same `RUN` without `FORCE`, optionally with `START_STAGE=agent`: finished baseline rows and the checkpoint are reused and only missing arms run. `FORCE=1` redoes everything.
- `SKIP_STAGES=train` is safe only when no arm uses `@world_model`; the pipeline also skips training on its own when no arm needs it.
- Logs land in `logs/<jobname>_<jobid>.out`; the first line holds the exact command to resume that run.

## 10. Where results land and what to read

```
results/<task>/<dataset>/<run>/
├── data/                       transitions + graph store
├── world_model/                self-describing checkpoint + train results JSON
├── agent/<budget>/<arm>.json   conditions 1-6 and 8
├── baselines/<budget>/<name>.json      conditions 7 and 9 (external_<name>, discovery_<name>)
├── baselines/_runs/<name>/<budget>/    each external repo's own stdout, artefacts, candidates
├── plots/*.png
├── report.md                   tables, condition coverage, stage timings, world-model section
└── pipeline.json               stage manifest
```

`test_commands.md` section 7 says which column to read per task, which row has to be beaten, and which results mean a bug rather than a finding.

## 11. Traps already hit on real clusters

- `mkdir: cannot create directory 'results': Permission denied`: the job was submitted from an `srun --pty` or `salloc` shell whose submit directory was not the checkout. The script now resolves the checkout from its own path and refuses to run outside one; set `REPO_ROOT` explicitly only for a copy staged outside the tree.
- A baseline venv at the wrong Python version: setup rebuilds it now, but if a repo still fails with "no wheels with a matching Python ABI tag", run `--only <name>` again after `module load`ing the toolchain it names.
- `all` and `all-external` never include the discovery systems; they are opted into by name or with `all-discovery`, because each is hours of LLM calls.
- `--baseline-timeout` (default 3600 s) is a floor for the discovery entries only in the sense that their own larger floors override it; every other external repo is killed at that value, so raise it (the submissions use 21600 or 43200) for DeepIM.
- The gateway returns a single choice for `n > 1` chat requests. ReEvo is patched for it at setup; nothing else sends `n > 1`.
- `torch-scatter` and friends are built CPU-only (`FORCE_ONLY_CPU=1`) on purpose; the three imputation baselines that need them are driven through their model, not their CUDA kernels.

## 12. Updating an existing checkout

```bash
cd ~/GraphWorldModel && source venv/bin/activate
git pull
uv pip install -r requirements.txt
python -m baselines.setup_baselines --list          # anything new shows installed False
python -m baselines.setup_baselines --only <new names>
python -m pytest tests -q
```

Nothing under `results/`, `data/raw/` or `baselines/external/` is touched by a pull.
