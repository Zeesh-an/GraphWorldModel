# Action-Conditioned World-Model Data (Influence Maximization)

Generates `(G, s_t, a_t, s_{t + 1}, R)` transition data for training a graph
**world model** on IM diffusion dynamics under node-level interventions.

- **State** `s_t = (infected, frontier)` — infected = ever-activated; frontier =
  newly-activated-this-step (IC: status-1 spreaders; LT: the fresh wave).
- **Action** `a_t` — a bag of `add_seed` / `remove_node` ops (empty bag = `NULL`).
- **Reward** `R` — spread gain (Δ activated-node count).

Backbone: **NDlib** IC/LT driven stepwise with mid-rollout `status` injection.
Per-episode rollouts come from classical IM "spine" seed selectors plus
random + counterfactual action injection (same-state/different-action coverage).
See the design spec:
`docs/superpowers/specs/2026-06-09-action-conditioned-wm-im-data-gen-design.md`.

## Quick start

```bash
source .venv/bin/activate

# Synthetic (5 BA graphs, IC+LT, all 6 spine algorithms)
python Data/action_conditioned_wm/generate_wm_data.py --dataset ba --num-graphs 5

# Real dataset (downloads on first use)
python Data/action_conditioned_wm/generate_wm_data.py --dataset jazz

# Tiny end-to-end check
python Data/action_conditioned_wm/generate_wm_data.py --smoke --out-dir /tmp/wm_smoke

# Validate a produced dataset (gate checks)
python Data/action_conditioned_wm/validate_wm_data.py --dir Data/action_conditioned_wm/output/ba
```

## Output (`output/<dataset>/`)

| File                                            | Contents                                                               |
| ----------------------------------------------- | ---------------------------------------------------------------------- |
| `graphs/<graph_id>.npz`                         | one graph: `edge_index, ic_probs, lt_weights, node_feats, node_labels` |
| `graphs_index.json`                             | `graph_id` → metadata (type, directed, n_nodes, n_edges, ...)          |
| `transitions_<IC\|LT>_<train\|val\|test>.jsonl` | one transition record per line                                         |
| `metadata.json`                                 | full generation config + episode count                                 |

Each transition row (JSONL):

```json
{
  "graph_id": "ba_n100_m3_s0", "diffusion_model": "IC",
  "episode_id": "ba_n100_m3_s0|IC|pagerank|k5|r2", "algorithm": "pagerank",
  "branch": "main", "t": 3,
  "state":      {"infected": [...], "frontier": [...], "infected_count": 6, "frontier_count": 2},
  "action":     [{"op": "add_seed", "target": 17}],
  "next_state": {"infected": [...], "frontier": [...], "infected_count": 9, "frontier_count": 2},
  "reward": 3.0
}
```

`branch` is `"main"` for the executed trajectory or `"cf_i"` for counterfactual
forks (same `t`/`state`, different `action`). An empty `action` list is `NULL`.

## Datasets

- **Real** (downloaded via `Data/datasets/`): `cora_ml, digg, twitter, jazz,
netscience, power_grid, nethept`.
- **Synthetic**: `er, ba, ws, karate`.

Run `python Data/action_conditioned_wm/generate_wm_data.py --help` for all flags
(`--models`, `--prob-model`, `--budget`, `--rollouts`, `--horizon`, `--inject-p`,
`--cf-prob`, `--split`, `--seed`, synthetic params, ...).

## Modules

| Module                | Responsibility                                                                               |
| --------------------- | -------------------------------------------------------------------------------------------- |
| `wm_graphs.py`        | graph providers (real + synthetic) + edge probabilities                                      |
| `wm_simulator.py`     | `State`/`ActionOp` types + NDlib stepwise IC/LT sim with action injection + snapshot/restore |
| `wm_actions.py`       | spine seed selectors + injection schedule + counterfactual candidates                        |
| `generate_wm_data.py` | graph store + JSONL transition writer + CLI orchestrator                                     |
| `validate_wm_data.py` | post-hoc gate-check harness                                                                  |

## Tests

```bash
.venv/bin/python -m pytest Data/action_conditioned_wm/tests/ -v
```
