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
IN_CHANNELS = 6
CH_INFECTED, CH_FRONTIER, CH_DEGREE, CH_ADD, CH_REMOVE, CH_EDGE = range(6)

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
) -> GraphInput:
    ei = torch.as_tensor(edge_index, dtype=torch.long, device=device)

    if ei.numel() == 0:
        ei = ei.reshape(2, 0)

    src, dst = ei[0], ei[1]

    # LT ignores edge weights (structure + hidden node thresholds)
    # IC keeps p(u -> v)
    if diffusion_model == "IC":
        w = torch.as_tensor(edge_weight, dtype=torch.float32, device=device)
    else:
        w = torch.ones(ei.shape[1], dtype=torch.float32, device=device)

    # Aggregate FROM in-neighbors: row=dst (v), col=src (u), and adds self-loops
    loop = torch.arange(num_nodes, device=device)
    row = torch.cat([dst, loop])
    col = torch.cat([src, loop])
    val = torch.cat([w, torch.ones(num_nodes, device=device)])

    # Symmetric renormalization with weighted in-degree (GCN renormalization trick, generalized)
    deg = torch.zeros(num_nodes, device=device).scatter_add_(0, row, val)
    dinv = deg.clamp(min=1e-12).pow(-0.5)
    val = dinv[row] * val * dinv[col]

    adj = torch.sparse_coo_tensor(
        torch.stack([row, col]), val, (num_nodes, num_nodes)
    ).coalesce()

    # Sparse adjacency matrix for the GNN world model
    return GraphInput(num_nodes=num_nodes, adj_norm=adj, edge_index=ei, edge_weight=w)


EDGE_OPS = ("add_edge", "remove_edge", "set_edge_weight")


def apply_edge_ops(edges: dict, action: list[dict]) -> dict:
    """Return a new edge dict with the bag's edge ops applied (node ops ignored)"""
    out = dict(edges)

    for op in action:
        if op["op"] in ("add_edge", "set_edge_weight"):
            out[(int(op["target"]), int(op["destination"]))] = float(
                op.get("weight", 1.0)
            )
        elif op["op"] == "remove_edge":
            out.pop((int(op["target"]), int(op["destination"])), None)

    return out


def reconstruct_episode_adjacency(
    records: list[dict], base_edges: dict
) -> dict[tuple, dict]:
    """
    Map (t, branch) -> edge dict {(u, v): weight} for one episode.
    Main transitions see the POST-action graph at their step (cumulative edge ops
    through t inclusive); cf transitions branch from the PRE-step graph (cumulative
    edge ops through t-1). Node-only / diffusion-only episodes return base for all.
    """
    has_edge_ops = any(op["op"] in EDGE_OPS for r in records for op in r["action"])
    out = {}

    if not has_edge_ops:
        # If the episode has no edge operations (diffusion-only / node-action data), every step just reuses the base graph
        for r in records:
            out[(r["t"], r["branch"])] = base_edges

        return out

    main = sorted((r for r in records if r["branch"] == "main"), key=lambda r: r["t"])
    running = dict(base_edges)  # cumulative state BEFORE the current main step
    pre_by_t = {}

    for r in main:
        pre_by_t[r["t"]] = dict(running)  # adj_pre[t]
        running = apply_edge_ops(running, r["action"])  # adj_post[t] becomes next pre
        out[(r["t"], "main")] = dict(running)  # main uses post

    # cf transitions reuse the pre-step graph at their t
    for r in records:
        if r["branch"] != "main":
            out[(r["t"], r["branch"])] = pre_by_t.get(r["t"], dict(base_edges))

    return out


def build_features(
    record: dict,
    edge_index: np.ndarray,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X (N, 6) float32, y_inf (N) float32, y_fr (N) float32)."""
    X = np.zeros((num_nodes, IN_CHANNELS), dtype=np.float32)

    state = record["state"]
    X[np.asarray(state["infected"], dtype=np.int64), CH_INFECTED] = 1.0
    X[np.asarray(state["frontier"], dtype=np.int64), CH_FRONTIER] = 1.0

    # degree = log1p(total degree in A_t); input_proj + LayerNorm handle scaling.
    deg = np.zeros(num_nodes, dtype=np.float32)
    if edge_index.size:
        np.add.at(deg, edge_index[0], 1.0)
        np.add.at(deg, edge_index[1], 1.0)
    X[:, CH_DEGREE] = np.log1p(deg)

    for op in record["action"]:
        if op["op"] == "add_node":
            X[int(op["target"]), CH_ADD] = 1.0
        elif op["op"] == "remove_node":
            X[int(op["target"]), CH_REMOVE] = 1.0
        elif op["op"] in EDGE_OPS:
            X[int(op["target"]), CH_EDGE] = 1.0
            X[int(op["destination"]), CH_EDGE] = 1.0

    nxt = record["next_state"]

    # Build the ground-truth next state s_{t + 1} the model is trying to predict.
    # Targets are soft one-step marginals when present (MC-estimated in data gen);
    # otherwise fall back to the single-draw binary next-state (legacy datasets).
    y_inf = np.zeros(num_nodes, dtype=np.float32)
    y_fr = np.zeros(num_nodes, dtype=np.float32)
    inf_marg = record.get("next_marginal_infected")
    fr_marg = record.get("next_marginal_frontier")

    if inf_marg is not None:
        for v, p in inf_marg.items():
            y_inf[int(v)] = p
        for v, p in fr_marg.items():
            y_fr[int(v)] = p
    else:
        y_inf[np.asarray(nxt["infected"], dtype=np.int64)] = 1.0
        y_fr[np.asarray(nxt["frontier"], dtype=np.int64)] = 1.0

    # X: (N, 6) - node is infected/frontier at timestep t - (CH_INFECTED, CH_FRONTIER, CH_DEGREE, CH_ADD, CH_REMOVE, CH_EDGE)
    # y_inf: (N) - node is infected at timestep t + 1
    # y_fr: (N) - node is infected at timestep t + 1
    return X, y_inf, y_fr


def load_graph_store(out_dir: Path) -> dict[str, dict]:
    """Load all graphs from disk and return graph_id -> {edge_index, ic_probs, lt_weights, num_nodes, base_edges, meta}."""
    out_dir = Path(out_dir)
    index = json.loads((out_dir / "graphs_index.json").read_text())
    store = {}

    for meta in index:
        gid = meta["graph_id"]
        npz = np.load(out_dir / "graphs" / meta["file"])
        ei = npz["edge_index"].astype(np.int64)
        ic = npz["ic_probs"].astype(np.float32)
        num_nodes = int(meta["n_nodes"])

        base_edges = {
            (int(ei[0, i]), int(ei[1, i])): float(ic[i]) for i in range(ei.shape[1])
        }
        store[gid] = {
            "edge_index": ei,
            "ic_probs": ic,
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
    ei = np.array(keys, dtype=np.int64).T  # (2, E) [src, dst]
    w = np.array([edges[k] for k in keys], dtype=np.float32)

    return ei, w


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

        # groups records by (graph_id, episode_id)
        groups = defaultdict(list)
        for r in records:
            groups[(r["graph_id"], r["episode_id"])].append(r)

        # Flatten every transition into samples = [(record, edge_index, edge_weight), …]
        self.samples = []
        for (gid, _eid), recs in groups.items():
            base = self.store[gid]["base_edges"]
            adj_map = reconstruct_episode_adjacency(recs, base)

            for r in recs:
                ei, w = edges_to_arrays(adj_map[(r["t"], r["branch"])])
                self.samples.append((r, ei, w))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        r, ei, w = self.samples[i]
        n = self.store[r["graph_id"]]["num_nodes"]
        X, y_inf, y_fr = build_features(r, ei, n)

        return {
            "X": torch.from_numpy(X),
            "y_inf": torch.from_numpy(y_inf),
            "y_fr": torch.from_numpy(y_fr),
            "edge_index": torch.from_numpy(ei),
            "edge_weight": torch.from_numpy(w),
            "num_nodes": n,
            "record": r,
        }


def collate_transitions(
    batch: list[dict],
    diffusion_model: str,
    device: torch.device,
) -> dict:
    """Stack B transitions into one disjoint block-diagonal graph + a GraphInput."""
    Xs, yi, yf, eis, ews, bidx = [], [], [], [], [], []
    offset = 0

    for b, item in enumerate(batch):
        num_nodes = item["num_nodes"]
        Xs.append(item["X"])
        yi.append(item["y_inf"])
        yf.append(item["y_fr"])
        eis.append(
            item["edge_index"] + offset
        )  # offset node ids into the block-diagonal graph
        ews.append(item["edge_weight"])
        bidx.append(torch.full((num_nodes,), b, dtype=torch.long))

        offset += num_nodes

    X = torch.cat(Xs).to(device)
    y_inf = torch.cat(yi).to(device)
    y_fr = torch.cat(yf).to(device)

    edge_index = torch.cat(eis, dim=1) if eis else torch.zeros(2, 0, dtype=torch.long)
    edge_weight = torch.cat(ews) if ews else torch.zeros(0, dtype=torch.float32)
    gi = build_graph_input(
        edge_index.numpy(), edge_weight.numpy(), offset, diffusion_model, device
    )

    return {
        "X": X,
        "y_inf": y_inf,
        "y_fr": y_fr,
        "graph": gi,
        "batch_index": torch.cat(bidx).to(device),
        "records": [it["record"] for it in batch],
    }
