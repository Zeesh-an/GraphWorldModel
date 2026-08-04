"""
World-model data pipeline: graph store loading from data/generate_wm_data.py, weighted/directed adjacency,
per-episode A_t reconstruction for edge actions, the 6-channel feature builder,
the transition Dataset, and a block-diagonal collate.
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

# Per-node input channels
in_channels = 6
ch_infected, ch_frontier, ch_degree, ch_add, ch_remove, ch_edge = range(6)

# Numerical floor for the symmetric renormalization (avoids 0^-0.5)
degree_floor = 1e-12

# X has shape (N, 6), with one row per node (N = nodes in the graph) and one column per feature channel
# ┌─────┬─────────────┬────────────────────────────────────────────────────────────────────────────────────┬──────────────────┬───────────────────────────────────────────┐
# │ col │    name     │                                     represents                                     │       type       │                 set from                  │
# ├─────┼─────────────┼────────────────────────────────────────────────────────────────────────────────────┼──────────────────┼───────────────────────────────────────────┤
# │ 0   │ CH_INFECTED │ node is currently infected (ever-activated) at time t, before the action           │ binary 0/1       │ record["state"]["infected"]               │
# ├─────┼─────────────┼────────────────────────────────────────────────────────────────────────────────────┼──────────────────┼───────────────────────────────────────────┤
# │ 1   │ CH_FRONTIER │ node is in the current spreading wave (frontier) at time t, before the action      │ binary 0/1       │ record["state"]["frontier"]               │
# ├─────┼─────────────┼────────────────────────────────────────────────────────────────────────────────────┼──────────────────┼───────────────────────────────────────────┤
# │ 2   │ CH_DEGREE   │ log1p(total degree) of the node in the time-t graph A_t — structural connectivity  │ continuous float │ counts of the node in edge_index (in+out) │
# ├─────┼─────────────┼────────────────────────────────────────────────────────────────────────────────────┼──────────────────┼───────────────────────────────────────────┤
# │ 3   │ CH_ADD      │ node is the target of an add_node op this step (just seeded)                       │ binary 0/1       │ the action bag                            │
# ├─────┼─────────────┼────────────────────────────────────────────────────────────────────────────────────┼──────────────────┼───────────────────────────────────────────┤
# │ 4   │ CH_REMOVE   │ node is the target of a remove_node op this step                                   │ binary 0/1       │ the action bag                            │
# ├─────┼─────────────┼────────────────────────────────────────────────────────────────────────────────────┼──────────────────┼───────────────────────────────────────────┤
# │ 5   │ CH_EDGE     │ node is an endpoint of an edge op (add_edge/remove_edge/set_edge_weight) this step │ binary 0/1       │ both target and destination of edge ops   │
# └─────┴─────────────┴────────────────────────────────────────────────────────────────────────────────────┴──────────────────┴───────────────────────────────────────────┘


@dataclass
class GraphInput:
    num_nodes: int
    adj_norm: (
        torch.Tensor
    )  # sparse (N, N): adj[v, u] = normalized weight of arc u -> v + self-loops
    edge_index: torch.Tensor  # (2, E) [src, dst]
    edge_weight: torch.Tensor  # (E)


def build_graph_input(
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    num_nodes: int,
    diffusion_model: str,
    device: torch.device,
    hide_edge_weights: bool = False,
) -> GraphInput:
    edge_index_tensor = torch.as_tensor(edge_index, dtype=torch.long, device=device)

    if edge_index_tensor.numel() == 0:
        edge_index_tensor = edge_index_tensor.reshape(2, 0)

    sources, destinations = edge_index_tensor[0], edge_index_tensor[1]

    # LT ignores edge weights (structure + hidden node thresholds)
    # IC keeps p(u -> v)
    #
    # hide_edge_weights feeds ones under IC too. That is the online/bandit
    # information state: the graph is known, the transmission probabilities are
    # not, and the encoder has to infer them from structure. It is also the
    # honest ablation for the wrinkle that our IC heads otherwise consume the
    # true w as an input feature; see research/adaptive_online_im.md §2.4b, §9.3
    # item 7. The SIMULATOR still uses the true probabilities: this masks the
    # model's view of the world, not the world.
    if diffusion_model == "IC" and not hide_edge_weights:
        weights = torch.as_tensor(edge_weight, dtype=torch.float32, device=device)
    else:
        weights = torch.ones(
            edge_index_tensor.shape[1], dtype=torch.float32, device=device
        )

    # Aggregate FROM in-neighbors: row=dst (v), col=src (u), and adds self-loops
    self_loops = torch.arange(num_nodes, device=device)
    row = torch.cat([destinations, self_loops])
    col = torch.cat([sources, self_loops])
    values = torch.cat([weights, torch.ones(num_nodes, device=device)])

    # Symmetric renormalization with weighted in-degree (GCN renormalization trick, generalized)
    degrees = torch.zeros(num_nodes, device=device).scatter_add_(0, row, values)
    inv_sqrt_degrees = degrees.clamp(min=degree_floor).pow(-0.5)
    values = inv_sqrt_degrees[row] * values * inv_sqrt_degrees[col]

    adjacency = torch.sparse_coo_tensor(
        torch.stack([row, col]), values, (num_nodes, num_nodes)
    ).coalesce()

    # Sparse adjacency matrix for the GNN world model
    return GraphInput(
        num_nodes=num_nodes,
        adj_norm=adjacency,
        edge_index=edge_index_tensor,
        edge_weight=weights,
    )


edge_ops = ("add_edge", "remove_edge", "set_edge_weight")


def apply_edge_ops(edges: dict, action: list[dict]) -> dict:
    """Return a new edge dict with the bag's edge ops applied (node ops ignored)"""
    updated = dict(edges)

    for action_op in action:
        if action_op["op"] in ("add_edge", "set_edge_weight"):
            updated[(int(action_op["target"]), int(action_op["destination"]))] = float(
                action_op.get("weight", 1.0)
            )
        elif action_op["op"] == "remove_edge":
            updated.pop((int(action_op["target"]), int(action_op["destination"])), None)

    return updated


def reconstruct_episode_adjacency(
    records: list[dict], base_edges: dict
) -> dict[tuple, dict]:
    """
    Map (t, branch) -> edge dict {(u, v): weight} for one episode.
    Main transitions see the POST-action graph at their step (cumulative edge ops
    through t inclusive); cf transitions branch from the PRE-step graph (cumulative
    edge ops through t-1). Node-only / diffusion-only episodes return base for all.
    """
    has_edge_ops = any(
        action_op["op"] in edge_ops
        for record in records
        for action_op in record["action"]
    )
    adjacency_by_step = {}

    if not has_edge_ops:
        # If the episode has no edge operations (diffusion-only / node-action data), every step just reuses the base graph
        for record in records:
            adjacency_by_step[(record["t"], record["branch"])] = base_edges

        return adjacency_by_step

    main = sorted(
        (record for record in records if record["branch"] == "main"),
        key=lambda record: record["t"],
    )
    running = dict(base_edges)  # cumulative state BEFORE the current main step
    pre_edges_by_step = {}

    for record in main:
        pre_edges_by_step[record["t"]] = dict(running)  # adj_pre[t]
        # adj_post[t] becomes the next step's pre
        running = apply_edge_ops(running, record["action"])
        adjacency_by_step[(record["t"], "main")] = dict(running)  # main uses post

    # cf transitions reuse the pre-step graph at their t
    for record in records:
        if record["branch"] != "main":
            adjacency_by_step[(record["t"], record["branch"])] = pre_edges_by_step.get(
                record["t"], dict(base_edges)
            )

    return adjacency_by_step


def build_features(
    record: dict,
    edge_index: np.ndarray,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X (N, 6) float32, y_inf (N) float32, y_fr (N) float32)."""
    X = np.zeros((num_nodes, in_channels), dtype=np.float32)

    state = record["state"]
    X[np.asarray(state["infected"], dtype=np.int64), ch_infected] = 1.0
    X[np.asarray(state["frontier"], dtype=np.int64), ch_frontier] = 1.0

    # degree = log1p(total degree in A_t); input_proj + LayerNorm handle scaling.
    degrees = np.zeros(num_nodes, dtype=np.float32)

    if edge_index.size:
        np.add.at(degrees, edge_index[0], 1.0)
        np.add.at(degrees, edge_index[1], 1.0)

    X[:, ch_degree] = np.log1p(degrees)

    for action_op in record["action"]:
        if action_op["op"] == "add_node":
            X[int(action_op["target"]), ch_add] = 1.0
        elif action_op["op"] == "remove_node":
            X[int(action_op["target"]), ch_remove] = 1.0
        elif action_op["op"] in edge_ops:
            X[int(action_op["target"]), ch_edge] = 1.0
            X[int(action_op["destination"]), ch_edge] = 1.0

    # Build the ground-truth next state s_{t + 1} the model is trying to predict,
    # as soft one-step marginals (MC-estimated in data gen via --mc-marginals).
    infected_marginal = record.get("next_marginal_infected")
    frontier_marginal = record.get("next_marginal_frontier")
    if infected_marginal is None or frontier_marginal is None:
        raise KeyError(
            "build_features requires soft marginal targets (next_marginal_infected / "
            "next_marginal_frontier); regenerate the dataset with "
            "data/generate_wm_data.py --mc-marginals >= 1"
        )

    y_inf = np.zeros(num_nodes, dtype=np.float32)
    y_fr = np.zeros(num_nodes, dtype=np.float32)

    for node, probability in infected_marginal.items():
        y_inf[int(node)] = probability

    for node, probability in frontier_marginal.items():
        y_fr[int(node)] = probability

    # X: (N, 6) - node is infected/frontier at timestep t - (CH_INFECTED, CH_FRONTIER, CH_DEGREE, CH_ADD, CH_REMOVE, CH_EDGE)
    # y_inf: (N) - node is infected at timestep t + 1
    # y_fr: (N) - node is in the frontier at timestep t + 1
    return X, y_inf, y_fr


def load_graph_store(out_dir: Path) -> dict[str, dict]:
    """Load all graphs from disk and return graph_id -> {edge_index, ic_probs, lt_weights, num_nodes, base_edges, meta}."""
    out_dir = Path(out_dir)
    index = json.loads((out_dir / "graphs_index.json").read_text())
    store = {}

    for meta in index:
        graph_id = meta["graph_id"]
        npz = np.load(out_dir / "graphs" / meta["file"])
        edge_index = npz["edge_index"].astype(np.int64)
        ic_probs = npz["ic_probs"].astype(np.float32)
        num_nodes = int(meta["n_nodes"])

        base_edges = {
            (int(edge_index[0, edge]), int(edge_index[1, edge])): float(ic_probs[edge])
            for edge in range(edge_index.shape[1])
        }
        store[graph_id] = {
            "edge_index": edge_index,
            "ic_probs": ic_probs,
            "lt_weights": npz["lt_weights"],
            "num_nodes": num_nodes,
            "base_edges": base_edges,
            "meta": meta,
        }

    return store


def edges_to_arrays(
    edges: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an edge dict {(u, v): weight} to (edge_index (2, E), weights (E)) arrays."""
    if not edges:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    keys = list(edges.keys())
    edge_index = np.array(keys, dtype=np.int64).T  # (2, E) [src, dst]
    weights = np.array([edges[key] for key in keys], dtype=np.float32)

    return edge_index, weights


def load_episode_endpoints(
    out_dir: Path, diffusion_model: str, split: str
) -> list[dict]:
    """
    Regroup `transitions_<dm>_<split>.jsonl` by episode into (x, y) pairs.

    The labelled data source localization needs, recovered from transitions that
    already exist — no new simulator, no new action op, no regeneration
    (research/source_localization.md §2.1). Per episode:

      * **sources `x`** — the `t = 0`, `branch = "main"` record's action IS the seed
        commit, a bag of `add_node` ops, and the generator writes it for every
        episode of every task.
      * **observation `y`** — the LAST main record's view of the terminal state, in
        two forms. `marginal` is that record's `next_marginal_infected`, i.e. the
        MC-averaged `P(infected)`; `binary` is its realized `next_state.infected`.
        §2.9 risk 5 is why both are kept: our marginals are averaged over
        `--mc-marginals` draws while SL-VAE observes a single binary realization,
        so the marginal column is strictly MORE informative than the literature's
        and only the binarized one is comparable to §5.1.

    Warning: `marginal` is the marginal of the LAST STEP, conditioned on the realized
    trajectory up to it — not the marginal of the whole cascade from `x`. Every
    node infected earlier reads exactly 1.0 and only the final wave is fractional.
    That is still a continuous `y in [0,1]^|V|`, which is the input type SL-VAE
    assumes, and it is what the generator writes; stating it here so nobody reads
    it as `P(infected | x)`.

    Counterfactual branches are skipped: they fork the ACTION mid-episode, so
    their terminal state was not produced by the `t = 0` seed set alone.
    """
    path = Path(out_dir) / f"transitions_{diffusion_model}_{split}.jsonl"
    store = load_graph_store(out_dir)

    groups = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue

        record = json.loads(line)
        if record["branch"] != "main":
            continue

        groups[(record["graph_id"], record["episode_id"])].append(record)

    episodes = []

    for (graph_id, episode_id), records in groups.items():
        records.sort(key=lambda record: record["t"])
        num_nodes = store[graph_id]["num_nodes"]

        sources = sorted(
            {
                int(action["target"])
                for action in records[0]["action"]
                if action["op"] == "add_node"
            }
        )
        # An episode whose t=0 bag seeded nothing has no source set to recover
        if not sources or records[0]["t"] != 0:
            continue

        terminal = records[-1]
        marginal = np.zeros(num_nodes, dtype=np.float32)
        for node, probability in (terminal.get("next_marginal_infected") or {}).items():
            marginal[int(node)] = probability

        binary = np.zeros(num_nodes, dtype=np.float32)
        binary[terminal["next_state"]["infected"]] = 1.0

        # The whole observed path, one row per step. Free here (every main record
        # already carries its own next_state) and it is a genuine observation
        # SETTING rather than plumbing for one baseline: §8.3 lists the full
        # trajectory alongside the snapshot, and DDMSL / DDMIX / DIPT reconstruct
        # it rather than assuming it. PDSL conditions on intermediate snapshots
        # and cannot be run from the endpoint alone.
        trajectory = np.zeros((len(records), num_nodes), dtype=np.float32)
        for step, record in enumerate(records):
            trajectory[step, record["next_state"]["infected"]] = 1.0

        episodes.append(
            {
                "graph_id": graph_id,
                "episode_id": episode_id,
                "algorithm": records[0].get("algorithm"),
                "num_nodes": num_nodes,
                "sources": sources,
                "marginal": marginal,
                "binary": binary,
                "trajectory": trajectory,
                # Steps the cascade actually ran, so a program can be told how
                # long the diffusion it is inverting had to spread
                "horizon": int(terminal["t"]) + 1,
                "infected_count": int(binary.sum()),
            }
        )

    return sorted(episodes, key=lambda episode: episode["episode_id"])


class TransitionDataset(Dataset):
    """One item per transition (main + cf). Resolves A_t per episode."""

    def __init__(self, out_dir: Path, diffusion_model: str, split: str) -> None:
        self.diffusion_model = diffusion_model

        # Load the store and the JSONL for one (diffusion_model, split)
        self.store = load_graph_store(out_dir)
        path = Path(out_dir) / f"transitions_{diffusion_model}_{split}.jsonl"
        records = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]

        # Group records by (graph_id, episode_id)
        groups = defaultdict(list)
        for record in records:
            groups[(record["graph_id"], record["episode_id"])].append(record)

        # Flatten every transition into samples = [(record, edge_index, edge_weight), …]
        self.samples = []
        for (graph_id, _), episode_records in groups.items():
            base_edges = self.store[graph_id]["base_edges"]
            adjacency_map = reconstruct_episode_adjacency(episode_records, base_edges)

            for record in episode_records:
                edge_index, weights = edges_to_arrays(
                    adjacency_map[(record["t"], record["branch"])]
                )
                self.samples.append((record, edge_index, weights))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        record, edge_index, weights = self.samples[index]
        num_nodes = self.store[record["graph_id"]]["num_nodes"]
        X, y_inf, y_fr = build_features(record, edge_index, num_nodes)

        return {
            "X": torch.from_numpy(X),
            "y_inf": torch.from_numpy(y_inf),
            "y_fr": torch.from_numpy(y_fr),
            "edge_index": torch.from_numpy(edge_index),
            "edge_weight": torch.from_numpy(weights),
            "num_nodes": num_nodes,
            "record": record,
        }


def collate_transitions(
    batch: list[dict],
    diffusion_model: str,
    device: torch.device,
    hide_edge_weights: bool = False,
) -> dict:
    """Stack B transitions into one disjoint block-diagonal graph + a GraphInput."""
    x_parts = []
    y_inf_parts = []
    y_fr_parts = []
    edge_index_parts = []
    edge_weight_parts = []
    batch_index_parts = []
    offset = 0

    for position, item in enumerate(batch):
        num_nodes = item["num_nodes"]
        x_parts.append(item["X"])
        y_inf_parts.append(item["y_inf"])
        y_fr_parts.append(item["y_fr"])
        # Offset node ids into the block-diagonal graph
        edge_index_parts.append(item["edge_index"] + offset)
        edge_weight_parts.append(item["edge_weight"])
        batch_index_parts.append(torch.full((num_nodes,), position, dtype=torch.long))

        offset += num_nodes

    X = torch.cat(x_parts).to(device)
    y_inf = torch.cat(y_inf_parts).to(device)
    y_fr = torch.cat(y_fr_parts).to(device)

    edge_index = (
        torch.cat(edge_index_parts, dim=1)
        if edge_index_parts
        else torch.zeros(2, 0, dtype=torch.long)
    )
    edge_weight = (
        torch.cat(edge_weight_parts)
        if edge_weight_parts
        else torch.zeros(0, dtype=torch.float32)
    )
    graph_input = build_graph_input(
        edge_index.numpy(),
        edge_weight.numpy(),
        offset,
        diffusion_model,
        device,
        hide_edge_weights,
    )

    return {
        "X": X,
        "y_inf": y_inf,
        "y_fr": y_fr,
        "graph": graph_input,
        "batch_index": torch.cat(batch_index_parts).to(device),
        "records": [item["record"] for item in batch],
    }
