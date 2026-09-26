# Network World Models as Environments for Algorithm Design on Complex Systems

This repository contains the code for the Network World Model and the algorithm design framework built around it. A Network World Model is a learned, action-conditioned simulator of diffusion on a network. A coding agent uses it as a fast environment in which to write, test and revise executable algorithms that choose interventions on complex systems: seeding a campaign, removing critical nodes, blocking a rumor, allocating vaccines, or inferring the hidden cause of a spread.

## Overview

### Network World Model

Given a network $G$, a diffusion state $s_t$ and an intervention $a_t$, the model predicts the next state:

$$f_\theta(G, s_t, a_t) \to s_{t+1}$$

The transition factorizes as $s_{t+1} = T_{\mathrm{endo}}(T_{\mathrm{exo}}(s_t, a_t))$. The exogenous step $T_{\mathrm{exo}}$ applies the intervention exactly to node states, edges and edge weights, and the endogenous step $T_{\mathrm{endo}}$ is the learned diffusion step. A message-passing encoder (GCN, GraphSAGE, GATv2, Graph Transformer or GCNII) runs on the post-intervention network, and a structured head that mirrors the known form of the process reads out each node's next state: a per-edge transmission head for IC, a threshold hazard head for LT, a two-cascade head for competitive diffusion, and a per-node compartment transition head for SIR, SIS and SEIR. Because a node can only activate through an active neighbor, free-running rollouts stay self-terminating instead of saturating the network.

The model is trained on $(G, s_t, a_t, s_{t+1})$ transitions harvested from a trusted simulator, with Monte Carlo marginals as soft targets and counterfactual branches that apply different actions from the same state, so the model has to condition on the action.

### Algorithm design loop

A coding agent (an LLM) writes an executable algorithm for a task. Each round, the candidate is run on the task instances and scored by rollouts of the Network World Model. Beyond the task score, the loop returns three kinds of feedback:

- **Rollout diagnostics**: how the process evolved under the candidate's plan.
- **Action-level credit**: what each action in the plan contributed.
- **Counterfactual probes**: the agent asks what-if questions about actions the plan did not take, such as swapping a seed, before it writes the next candidate.

A candidate replaces the current best only when its paired improvement over the incumbent, measured on the same sampled realizations, clears the standard error of that paired difference. Generated algorithms are offline: they never call the world model themselves, and reach it only through the reward and feedback the harness computes.

### Experiments

Every returned algorithm, and every baseline, is replayed on one shared referee (the exact batched simulator, 1,000 samples by default), and only that replay is compared across methods. The experiments cover eight tasks, five diffusion models, and three kinds of baselines:

- the classical algorithms of each task's literature, implemented in a callable library (`coding_agent/tools/`);
- published implementations run from their own repositories (`baselines/`);
- published LLM algorithm discovery systems, each running its own search loop against the trusted simulator.

## Tasks

| Task | `--task` | Objective | Dynamics | What the algorithm returns | Metric |
| --- | --- | --- | --- | --- | --- |
| Influence maximization | `influence_maximization` | maximize | IC, LT | a seed set | spread, % of nodes activated |
| Adaptive influence maximization | `adaptive_online_im` | maximize | IC, LT | seeds committed over rounds, each after observing the last | spread, % of nodes activated |
| Critical node detection | `critical_node_detection` | minimize | IC, LT | nodes to remove ahead of an outbreak | remaining spread, % of nodes infected |
| Influence blocking | `influence_blocking` | minimize | IC, CLT | counter-seeds or blocked nodes against a rumor | rumor cascade size |
| Epidemic control | `epidemic_control` | minimize | SIR, SIS, SEIR | nodes to vaccinate, quarantine or isolate | attack rate, % of nodes ever infected |
| Source localization | `source_localization` | recover | IC, LT | the source set behind an observed spread | consistency with the observation |
| Cascade reconstruction | `cascade_reconstruction` | recover | IC, LT | the hidden history: who was infected, when, and by whom | likelihood of the decoded history |
| Cascade prediction | `cascade_prediction` | forecast | real cascades | the final size of a partly observed cascade | MSLE |

Cascade prediction is the one task that runs no simulator: it replays real cascades from platform logs, so it tests how well the IC and LT assumptions hold on real data.

## Datasets

The datasets used in the experiments, ordered by node count. Directed networks quote arcs. Every loader downloads its data to `data/raw/<dataset>/` on first use.

| Dataset | `--dataset` | Nodes | Edges | Tasks |
| --- | --- | ---: | ---: | --- |
| Email-EU | `email_eu_core` | 1,005 | 24,929 | Influence blocking |
| UCI Students | `uci_students` | 1,266 | 6,451 | Cascade reconstruction |
| Network Science | `netscience` | 1,589 | 2,742 | Influence maximization, adaptive influence maximization |
| Cora-ML | `cora_ml` | 2,810 | 7,981 | Source localization |
| Power Grid | `power_grid` | 4,941 | 6,594 | Critical node detection, source localization |
| CA-GrQc | `ca_grqc` | 5,242 | 14,484 | Cascade reconstruction |
| Oregon1 | `oregon1` | 10,670 | 22,002 | Epidemic control |
| PGP | `pgp` | 10,680 | 24,316 | Critical node detection |
| Infectious SocioPatterns | `infectious_sociopatterns` | 10,972 | 44,517 | Epidemic control |
| NetHEPT | `nethept` | 15,229 | 62,752 | Influence maximization, adaptive influence maximization |
| Gnutella24 | `p2p_gnutella24` | 26,518 | 65,369 | Influence blocking |
| Taoke | `taoke` | 29,711 | n/a | Cascade prediction |
| Deezer | `deezer` | 47,538 | 222,887 | Source localization |
| Gnutella31 | `p2p_gnutella` | 62,561 | 147,878 | Critical node detection |
| Digg | `digg` | 116,893 | 2,011,447 | Influence maximization, adaptive influence maximization |
| Digg (cascades) | `digg_cascades` | 279,630 | 1,731,653 | Cascade prediction |
| APS | `casflow_aps` | 616,316 | 3,304,400 | Cascade prediction |

Many more loaders are available in `data/datasets/`, together with the synthetic families `er`, `ba`, `ws`, `sbm`, `powerlaw_cluster` and `kronecker`. Each loader documents its source and how the network is built.

## Setup

The code is tested on Python 3.13. Create an environment and install the dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.13
uv pip install -r requirements.txt
```

The coding agent talks to any OpenAI-compatible chat completions endpoint. Put its address and key in a `.env` file at the repository root, which is gitignored:

```bash
GATEWAY_BASE_URL=https://api.openai.com/v1
CHATGPT_GATEWAY_TOKEN=<your API key>
# Only needed for models whose name starts with claude-
CLAUDE_GATEWAY_TOKEN=<your API key>
```

The model is chosen with `--llm-model`, and the pipeline checks that the endpoint serves it before any stage runs.

The published baselines are optional. Each one is cloned into `baselines/external/` with its own virtual environment, so its dependency pins never collide with ours:

```bash
uv run python -m baselines.setup_baselines --list
uv run python -m baselines.setup_baselines --only opim subsim
uv run python -m baselines.setup_baselines --all
```

## Running

One command runs a whole experiment: it generates the transitions, trains the Network World Model, runs every baseline and every agent arm at every budget, then writes the plots and a report. Finished stages are detected on disk, so a stopped run resumes where it left off.

```bash
# Influence maximization on Network Science: two library baselines, one
# published repository, and the design loop scored by the world model
uv run python -m pipeline.run --task influence_maximization --dataset netscience \
    --baselines imm degree_discount external:opim \
    --arms routing evolve_free@oracle evolve_free@world_model \
    --budget-pcts 1 5 10 20
```

Every task runs through the same entry point, and the task registry supplies its dynamics, action space and budgets:

```bash
uv run python -m pipeline.run --task critical_node_detection --dataset power_grid
uv run python -m pipeline.run --task influence_blocking --dataset email_eu_core
uv run python -m pipeline.run --task epidemic_control --dataset oregon1
uv run python -m pipeline.run --task source_localization --dataset cora_ml
uv run python -m pipeline.run --task cascade_reconstruction --dataset uci_students
uv run python -m pipeline.run --task cascade_prediction --dataset taoke
```

An agent arm is written `<method>_<mode>@<evaluator>`. The evaluator is the only thing that changes between arms, so the arms can be compared directly:

| Arm | Evaluator inside the design loop |
| --- | --- |
| `evolve_free@native` | one real simulator episode per candidate |
| `evolve_free@monte_carlo` | Monte Carlo simulation |
| `evolve_free@oracle` | the exact batched simulator |
| `evolve_free@world_model` | the Network World Model |
| `routing` | the LLM picks one library algorithm instead of writing code |

Useful flags for controlling a run:

```bash
# Search once at a 10% budget and replay the winner at the other budgets
uv run python -m pipeline.run --task influence_maximization --dataset nethept --search-budget pct10

# Rerun only the plots and the report of a finished run
uv run python -m pipeline.run --task influence_maximization --dataset netscience --start-stage plots

# Rerun one stage from scratch
uv run python -m pipeline.run --task influence_maximization --dataset netscience \
    --start-stage agent --end-stage agent --force

# Keep several variants of one task and dataset side by side
uv run python -m pipeline.run --task influence_maximization --dataset netscience \
    --run gcnii --wm-model gcnii
```

Everything a run produces is written to `results/<task>/<dataset>/<run>/`:

```
results/<task>/<dataset>/<run>/
├── data/          # generated transitions and the network store
├── world_model/   # checkpoints and training results
├── agent/         # one JSON per budget and arm, with the returned program
├── plots/         # figures
├── report.md      # results tables and world model diagnostics
└── pipeline.json  # stage status and configuration
```

Each task ships a runnable self-check of its contract:

```bash
uv run python -m coding_agent.check_containment
uv run python -m coding_agent.check_source_localization
uv run python -m coding_agent.check_influence_blocking
uv run python -m coding_agent.check_cascade_reconstruction
uv run python -m coding_agent.check_epidemic_control
uv run python -m coding_agent.check_cascade_prediction
uv run python -m coding_agent.check_adaptive
uv run python -m baselines.check_discovery
```

## Project structure

```
.
├── pipeline/                   # end-to-end driver
│   ├── run.py                  # one command: data, train, agent, plots, report
│   ├── tasks.py                # task registry: objective, dynamics, actions, budgets
│   ├── conditions.py           # agent arms and their evaluators
│   ├── layout.py               # results/<task>/<dataset>/<run>/ paths
│   ├── plots.py                # figures
│   ├── report.py               # report.md
│   └── summary.py              # cross-run summaries
├── data/                       # simulators, loaders and transition generation
│   ├── generate_wm_data.py     # simulate episodes and write transitions
│   ├── wm_simulator.py         # IC and LT dynamics and the five action ops
│   ├── wm_competitive.py       # two-cascade dynamics for influence blocking
│   ├── wm_epidemic.py          # SIR, SIS and SEIR dynamics
│   ├── wm_cascades.py          # replay of real cascades for cascade prediction
│   ├── wm_graphs.py            # network providers and synthetic families
│   ├── wm_actions.py           # seed selectors, action injection, counterfactuals
│   └── datasets/               # one loader per dataset
├── world_model/                # the Network World Model
│   ├── wm_data.py              # node features, network inputs, batching
│   ├── wm_model.py             # world model, backbone registry, structured heads
│   ├── model/                  # GCN, GraphSAGE, GATv2, Graph Transformer, GCNII
│   ├── train_wm.py             # teacher-forced training
│   ├── wm_eval.py              # one-step, rollout and planning evaluation
│   ├── wm_metrics.py           # metric primitives
│   └── scorer.py               # loads a checkpoint and scores candidate plans
├── coding_agent/               # the algorithm design loop
│   ├── agent.py                # LLM client
│   ├── run.py                  # runs one arm on one task
│   ├── methods/                # search methods, including evolve
│   ├── prompts.py              # agent prompts
│   ├── executor.py             # restricted execution of generated algorithms
│   ├── feedback.py             # task score and feedback
│   ├── diagnostics.py          # rollout diagnostics
│   ├── credit.py               # action-level credit
│   ├── probes.py               # counterfactual probes
│   ├── envs/                   # evaluator environments: Monte Carlo, multi-round, world model
│   ├── tools/                  # callable library of classical algorithms per task
│   └── check_*.py              # per-task self-checks
├── baselines/                  # published implementations and discovery systems
│   ├── registry.py             # every published baseline and how it runs
│   ├── setup_baselines.py      # clone and install into baselines/external/
│   ├── run_baseline.py         # run one baseline and score it on the referee
│   ├── discovery.py            # LLM algorithm discovery systems
│   ├── score_program.py        # fitness process for the discovery systems
│   └── drivers/                # entry points for repositories without a usable CLI
├── requirements.txt
└── ruff.toml
```
