# Coding-Agent Outer Loop

An LLM coding-agent **outer loop** that designs Influence-Maximization algorithms
emitting graph actions across **all** timesteps `t₀…t_T` (not just a `t₀` seed set),
evaluated by the trained Graph World Model (fast inner loop) with a Monte-Carlo
true-simulator baseline.

## Architecture

```
Agent (LLM, TODO) → Python Strategy script → graph actions per t → Environment → reward
                          ▲                                              │
                          └──────────── refine on reward ───────────────┘
```

- **Outer loop** (`methods/`): three interchangeable methods, all producing an
  `action_fn(state, timestep) → list[ActionOp]`.
- **Inner loop** (`envs/`): `WorldModelEnvironment` (default, fast) or
  `MonteCarloEnvironment` (NDlib ground truth) behind one `rollout()` interface.
- **Agent** (`agent.py`): provider-agnostic; `GatewayProvider` is the concrete LLM
  call (OpenAI-compatible gateway, `gpt-*` and `claude-*` models).
- **Library** (`tools/`): named algorithms over a primitive layer; the agent's script
  may call these.

## The three methods (`--method`)

| Method     | Idea                                                                | Agent calls          |
| ---------- | ------------------------------------------------------------------- | -------------------- |
| `one_shot` | one "super-algorithm" emits the whole `t₀…T` plan; refine on reward | few                  |
| `per_step` | re-prompt the agent every timestep on the current state             | one/step (expensive) |
| `windowed` | design one online algorithm; re-apply per time window               | one                  |

## Quick start

```bash
# Monte-Carlo baseline, canned strategy (no model needed) — via Python:
python -c "from coding_agent.run import ExperimentConfig, run_experiment; ..."

# World-model inner loop on a trained checkpoint:
python -m coding_agent.run \
    --data-dir data/output/ba20_marg_structured \
    --wm-results-json world_model/checkpoints/ba20_marg_structured_sage_IC.json \
    --method one_shot --evaluator world_model --budget 5 --horizon 10 --compare
```

> The LLM defaults to `claude-sonnet-5` via the gateway (`--model` switches, e.g.
> `gpt-5.6-sol`); requires `GATEWAY_BASE_URL` + tokens in `.env`. Model-less runs
> (tests) use a canned script via `run_experiment(cfg, graph=..., canned_script=...)`.

## Counterfactual credit (`--credit`)

Per-action reward attribution (`credit.py`): each action in the plan is ablated
(removed, everything else identical, same rollout seed) and re-rolled, giving
`delta = base_spread - ablated_spread` — the spread that single action is
responsible for (`~0` = wasted budget). With `--credit`:

- **`one_shot` refinement turns** include the per-action credit report, so the
  agent gets causal feedback ("your t=3 seed contributed +0.2") instead of a
  bare scalar reward.
- **The results JSON** gains `credit_base_reward` + `credit` (one entry per
  action) for the winning strategy, whatever the method. For state-dependent
  strategies (`per_step`/`windowed`) the recorded bags are replayed as a fixed
  plan, so credit is an approximation there.

Cost: one extra rollout per action per evaluation — cheap on the world-model
evaluator, noticeably slower on `--evaluator monte_carlo`.

## Implemented algorithms (`tools/algorithms.py`)

| Family        | Function              | Notes                                                   |
| ------------- | --------------------- | ------------------------------------------------------- |
| Degree        | `high_degree`         | top-k total degree                                      |
| Degree        | `weighted_degree`     | top-k by summed outgoing IC prob                        |
| Degree        | `degree_discount`     | DegreeDiscount (Chen et al. 2009)                       |
| Centrality    | `pagerank_seeds`      | top-k PageRank                                          |
| Centrality    | `eigenvector_seeds`   | top-k eigenvector centrality                            |
| Centrality    | `closeness_seeds`     | top-k closeness centrality                              |
| Greedy        | `vanilla_greedy`      | marginal-gain greedy (Kempe et al. 2003)                |
| Greedy        | `celf`                | lazy-forward greedy (Leskovec et al. 2007)              |
| Greedy        | `celf_pp`             | CELF++ (Goyal et al. 2011) — simplified                 |
| Greedy        | `adaptive_greedy`     | adaptive MC count — simplified                          |
| RIS           | `ris_basic`           | reverse influence sampling (Borgs et al. 2014), IC only |
| RIS           | `tim`                 | TIM/TIM+ (Tang et al. 2014)                             |
| RIS           | `imm`                 | IMM (Tang et al. 2015) — simplified                     |
| RIS           | `ssa`                 | SSA/D-SSA (Nguyen et al. 2016) — simplified             |
| RIS           | `filtered_ris`        | RR-size-filtered RIS                                    |
| Path          | `sp1m`                | SP1M/SPM (Kimura & Saito 2006) — truncated path-sum     |
| Path          | `mia_pmia`            | MIA/PMIA (Chen et al. 2010) — simplified                |
| Path          | `ldag`                | LDAG (Chen et al. 2010) — simplified                    |
| Sketch        | `static_greedy`       | StaticGreedy (Cheng et al. 2014)                        |
| Sketch        | `skim`                | SKIM (Cohen et al. 2014) — simplified                   |
| Community     | `community_im`        | label-propagation communities + per-community degree    |
| Community     | `cofim`               | CoFIM (Zhang et al. 2014) — bridge-aware                |
| Community     | `community_ris`       | per-community RR coverage                               |
| Metaheuristic | `simulated_annealing` | degree init + annealed swaps                            |
| Metaheuristic | `hill_climbing`       | degree init + 1-swap local search                       |
| Metaheuristic | `genetic_algorithm`   | seed-set GA (crossover + mutation)                      |
| Hybrid        | `pagerank_greedy`     | PageRank candidate pool + greedy                        |
| Hybrid        | `degree_ris_refine`   | degree init + RR-coverage swaps                         |
| Hybrid        | `celf_local_search`   | CELF + 1-swap refinement                                |
| Hybrid        | `community_celf`      | per-community CELF                                      |

Heavy algorithms (MIA/PMIA, LDAG, SKIM, StaticGreedy, SSA, IMM, Adaptive Greedy) are
**faithful-but-simplified** (noted in their docstrings).

Primitives (`tools/primitives.py`): `compute_degree`, `compute_out_degree`,
`compute_weighted_degree`, `compute_pagerank`, `compute_centrality`,
`get_top_degree_nodes`, `mc_simulate_spread`, `compute_marginal_gain`,
`batch_reverse_sample`, `ris_select`, `detect_communities`, `allocate_budget`,
`estimate_sample_size`, `sample_live_edge_graph`, `reachable_count`,
`path_influence_scores`, `build_simulator`.

## Wiring a model

`GatewayProvider` (`agent.py`) is wired to the lab's OpenAI-compatible gateway. It
reads `GATEWAY_BASE_URL` and the per-account token from the environment
(`CLAUDE_GATEWAY_TOKEN` for `claude-*` models, `CHATGPT_GATEWAY_TOKEN` otherwise) —
put them in `.env` (gitignored) and `load_dotenv()` picks them up in `run.py`.
To use a different backend, implement an `LLMProvider` with
`complete(system, user) -> str` and pass it to `CodingAgent(provider)`.

## Tests

`uv run pytest coding_agent/tests -q` — exercises the full outer+inner loop with
injected canned scripts (no model required).
