# Action-Conditioned World-Model Data (Influence Maximization)

Generates `(G, s_t, a_t, s_{t + 1}, R)` transition data for training a graph
**world model** on IM diffusion dynamics under node- and edge-level interventions.

- **State** `s_t = (infected, frontier)` — infected = ever-activated; frontier =
  newly-activated-this-step (IC: status-1 spreaders; LT: the fresh wave).
- **Action** `a_t` — a bag of node/edge ops: `add_node`, `remove_node`,
  `add_edge`, `remove_edge`, `set_edge_weight` (empty bag = `NULL`). Which ops are
  used is set by `--action-ops`; omitting it generates diffusion-only data.
- **Reward** `R` — spread gain (Δ activated-node count).

Backbone: **NDlib** IC/LT driven stepwise with mid-rollout `status` / edge mutation.
Per-episode rollouts come from classical IM "spine" seed selectors plus
random + counterfactual action injection (same-state/different-action coverage).
See the design spec:
`docs/superpowers/specs/2026-06-09-action-conditioned-wm-im-data-gen-design.md`.

## Quick start

```bash
source .venv/bin/activate

# Diffusion-only (no actions) — the default when --action-ops is omitted
python data/generate_wm_data.py --dataset ba --num-graphs 1

# Node action interventions
python data/generate_wm_data.py --dataset ba --num-graphs 1 \
    --action-ops add_node remove_node

# Edge action interventions
python data/generate_wm_data.py --dataset ba --num-graphs 1 \
    --action-ops add_edge remove_edge set_edge_weight

# Real dataset (downloads on first use)
python data/generate_wm_data.py --dataset jazz --action-ops add_node remove_node

# Tiny end-to-end check
python data/generate_wm_data.py --smoke --out-dir /tmp/wm_smoke

# Validate a produced dataset (gate checks)
python data/validate_wm_data.py --dir data/output/ba
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
  "action":     [{"op": "add_node", "target": 17}],
  "next_state": {"infected": [...], "frontier": [...], "infected_count": 9, "frontier_count": 2},
  "reward": 3.0
}
```

`branch` is `"main"` for the executed trajectory or `"cf_i"` for counterfactual
forks (same `t`/`state`, different `action`). An empty `action` list is `NULL`.
Edge ops carry the second endpoint as `"destination"` and, for `add_edge` /
`set_edge_weight`, a `"weight"` (the IC transmission probability), e.g.
`{"op": "set_edge_weight", "target": 4, "destination": 11, "weight": 0.62}`.

## Datasets

- **Real** (downloaded via `data/datasets/`): `cora_ml, digg, twitter, jazz,
netscience, power_grid, nethept`.
- **Synthetic**: `er, ba, ws, karate`.

Run `python data/generate_wm_data.py --help` for all flags
(`--models`, `--prob-model`, `--budget`, `--rollouts`, `--horizon`, `--inject-p`,
`--action-ops`, `--weight-lo`, `--weight-hi`, `--cf-prob`, `--split`, `--seed`,
synthetic params, ...).

---

## The 5 action ops (identical set for every dataset)

| op                | fields in the record                | IC effect                                    | LT effect                              |
| ----------------- | ----------------------------------- | -------------------------------------------- | -------------------------------------- |
| `add_node`        | `target=v`                          | status→1 (node becomes a spreader next step) | status→1                               |
| `remove_node`     | `target=v`                          | status→2 (Removed/spent, stays counted)      | status→0 (back to Susceptible)         |
| `add_edge`        | `target=u, destination=v, weight=w` | add arc u→v, set transmission p=`w`          | add edge structurally (weight ignored) |
| `remove_edge`     | `target=u, destination=v`           | remove arc u→v                               | remove edge structurally               |
| `set_edge_weight` | `target=u, destination=v, weight=w` | set arc u→v transmission p=`w`               | **no-op** (LT ignores edge weights)    |

The three settings are just which ops you pass to `--action-ops`:

- **Setting 1 (diffusion-only):** omit `--action-ops`
- **Setting 2 (node):** `--action-ops add_node remove_node`
- **Setting 3 (edge):** `--action-ops add_edge remove_edge set_edge_weight`

## Customizable probabilities / knobs

| Flag                              | Default       | What it controls                                                                                         |
| --------------------------------- | ------------- | -------------------------------------------------------------------------------------------------------- |
| `--prob-model {weighted,uniform}` | `weighted`    | IC transmission prob: `weighted` = 1/in_degree(v) (weighted cascade); `uniform` = constant `--uniform-p` |
| `--uniform-p`                     | `0.1`         | the constant IC prob (and LT edge weight) when `--prob-model uniform`                                    |
| `--inject-p`                      | `0.3`         | P(an intermediate step injects an action at all vs NULL)                                                 |
| `--weight-lo` / `--weight-hi`     | `0.0` / `1.0` | range for `add_edge`/`set_edge_weight` weights, sampled U(lo,hi)                                         |
| `--cf-prob`                       | `0.2`         | P(spawn counterfactual forks at a step)                                                                  |
| `--cf-branches`                   | `2`           | # alternate-action branches per fork                                                                     |
| `--er-p`                          | `0.05`        | ER edge probability (G(n,p)) — structural                                                                |
| `--ws-p`                          | `0.1`         | WS rewire probability (structural)                                                                       |
| `--ba-m`                          | `3`           | BA attachment count (not a prob)                                                                         |
| `--budget` / `--budget-pct`       | `5` / `None`  | seed-set size k (pct overrides absolute)                                                                 |
| `--rollouts` / `--horizon`        | `10` / `10`   | episodes per (graph,model,algo) / max timesteps                                                          |
| `--seed`                          | `42`          | master RNG (graph + selection + injection all derive from it)                                            |

> LT node thresholds are drawn U(0,1) per node internally — no CLI flag.

---

## Modules

| Module                | Responsibility                                                                               |
| --------------------- | -------------------------------------------------------------------------------------------- |
| `wm_graphs.py`        | graph providers (real + synthetic) + edge probabilities                                      |
| `wm_simulator.py`     | `State`/`ActionOp` types + NDlib stepwise IC/LT sim with action injection + snapshot/restore |
| `wm_actions.py`       | spine seed selectors + injection schedule + counterfactual candidates                        |
| `generate_wm_data.py` | graph store + JSONL transition writer + CLI orchestrator                                     |
| `validate_wm_data.py` | post-hoc gate-check harness                                                                  |
