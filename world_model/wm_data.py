"""
World-model data pipeline: graph store loading, weighted/directed adjacency,
per-episode A_t reconstruction for edge actions, the 6-channel feature builder,
the transition Dataset, and a block-diagonal collate.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Per-node input channels (Section A of the spec)
IN_CHANNELS = 6
CH_INFECTED, CH_FRONTIER, CH_DEGREE, CH_ADD, CH_REMOVE, CH_EDGE = range(6)


@dataclass
class GraphInput:
    num_nodes: int
    adj_norm: (
        torch.Tensor
    )  # sparse (N,N): adj[v,u] = normalized weight of arc u->v (+ self loops)
    edge_index: torch.Tensor  # (2,E) long [src, dst]
    edge_weight: torch.Tensor  # (E,) float


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

    # LT ignores edge weights (structure + hidden node thresholds); IC keeps p(u->v).
    if diffusion_model == "IC":
        w = torch.as_tensor(edge_weight, dtype=torch.float32, device=device)
    else:
        w = torch.ones(ei.shape[1], dtype=torch.float32, device=device)

    # Aggregate FROM in-neighbors: row=dst (v), col=src (u). Add self-loops.
    loop = torch.arange(num_nodes, device=device)
    row = torch.cat([dst, loop])
    col = torch.cat([src, loop])
    val = torch.cat([w, torch.ones(num_nodes, device=device)])

    # Symmetric renormalization with weighted in-degree (GCN renorm trick, generalized).
    deg = torch.zeros(num_nodes, device=device).scatter_add_(0, row, val)
    dinv = deg.clamp(min=1e-12).pow(-0.5)
    val = dinv[row] * val * dinv[col]

    adj = torch.sparse_coo_tensor(
        torch.stack([row, col]), val, (num_nodes, num_nodes)
    ).coalesce()
    return GraphInput(num_nodes=num_nodes, adj_norm=adj, edge_index=ei, edge_weight=w)


_EDGE_OPS = ("add_edge", "remove_edge", "set_edge_weight")


def _apply_edge_ops(
    edges: dict[tuple[int, int], float], action: list[dict]
) -> dict[tuple[int, int], float]:
    """Return a new edge dict with the bag's edge ops applied (node ops ignored)."""
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
    records: list[dict], base_edges: dict[tuple[int, int], float]
) -> dict[tuple[int, str], dict[tuple[int, int], float]]:
    """
    Map (t, branch) -> edge dict {(u,v): weight} for one episode.
    Main transitions see the POST-action graph at their step (cumulative edge ops
    through t inclusive); cf transitions branch from the PRE-step graph (cumulative
    edge ops through t-1). Node-only / diffusion-only episodes return base for all.
    """
    has_edge_ops = any(op["op"] in _EDGE_OPS for r in records for op in r["action"])
    out: dict = {}
    if not has_edge_ops:
        for r in records:
            out[(r["t"], r["branch"])] = base_edges
        return out

    main = sorted((r for r in records if r["branch"] == "main"), key=lambda r: r["t"])
    running = dict(base_edges)  # cumulative state BEFORE the current main step
    pre_by_t: dict = {}
    for r in main:
        pre_by_t[r["t"]] = dict(running)  # adj_pre[t]
        running = _apply_edge_ops(running, r["action"])  # adj_post[t] becomes next pre
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
    """Return (X (N,6) float32, y_inf (N,) float32, y_fr (N,) float32)."""
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
        elif op["op"] in _EDGE_OPS:
            X[int(op["target"]), CH_EDGE] = 1.0
            X[int(op["destination"]), CH_EDGE] = 1.0

    nxt = record["next_state"]
    y_inf = np.zeros(num_nodes, dtype=np.float32)
    y_fr = np.zeros(num_nodes, dtype=np.float32)
    y_inf[np.asarray(nxt["infected"], dtype=np.int64)] = 1.0
    y_fr[np.asarray(nxt["frontier"], dtype=np.int64)] = 1.0
    return X, y_inf, y_fr


# ---------------------------------------------------------------------------
# Append 1: graph store loader
# ---------------------------------------------------------------------------


def load_graph_store(out_dir: Path) -> dict[str, dict]:
    """Load all graphs from disk and return graph_id -> {edge_index, ic_probs, lt_weights, num_nodes, base_edges, meta}."""
    out_dir = Path(out_dir)
    index: list[dict] = json.loads((out_dir / "graphs_index.json").read_text())
    store: dict[str, dict] = {}
    for meta in index:
        gid: str = meta["graph_id"]
        npz = np.load(out_dir / "graphs" / meta["file"])
        ei = npz["edge_index"].astype(np.int64)
        ic = npz["ic_probs"].astype(np.float32)
        n: int = int(meta["n_nodes"])
        base_edges: dict[tuple[int, int], float] = {
            (int(ei[0, i]), int(ei[1, i])): float(ic[i]) for i in range(ei.shape[1])
        }
        store[gid] = {
            "edge_index": ei,
            "ic_probs": ic,
            "lt_weights": npz["lt_weights"],
            "num_nodes": n,
            "base_edges": base_edges,
            "meta": meta,
        }
    return store


# ---------------------------------------------------------------------------
# Append 2: _edges_to_arrays helper + TransitionDataset
# ---------------------------------------------------------------------------


def _edges_to_arrays(
    edges: dict[tuple[int, int], float],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an edge dict {(u,v): weight} to (edge_index (2,E), weights (E,)) arrays."""
    if not edges:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    keys: list[tuple[int, int]] = list(edges.keys())
    ei = np.array(keys, dtype=np.int64).T  # (2, E) [src, dst]
    w = np.array([edges[k] for k in keys], dtype=np.float32)
    return ei, w


class TransitionDataset(Dataset):
    """One item per transition (main + cf). Resolves A_t per episode."""

    def __init__(self, out_dir: Path, diffusion_model: str, split: str) -> None:
        self.dm: str = diffusion_model
        self.store: dict[str, dict] = load_graph_store(out_dir)
        path = Path(out_dir) / f"transitions_{diffusion_model}_{split}.jsonl"
        records: list[dict] = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]

        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in records:
            groups[(r["graph_id"], r["episode_id"])].append(r)

        self.samples: list[tuple[dict, np.ndarray, np.ndarray]] = []
        for (gid, _eid), recs in groups.items():
            base: dict[tuple[int, int], float] = self.store[gid]["base_edges"]
            adj_map = reconstruct_episode_adjacency(recs, base)
            for r in recs:
                ei, w = _edges_to_arrays(adj_map[(r["t"], r["branch"])])
                self.samples.append((r, ei, w))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        r, ei, w = self.samples[i]
        n: int = self.store[r["graph_id"]]["num_nodes"]
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


# ---------------------------------------------------------------------------
# Append 3: block-diagonal collate
# ---------------------------------------------------------------------------


def collate_transitions(
    batch: list[dict],
    diffusion_model: str,
    device: torch.device,
) -> dict:
    """Stack B transitions into one disjoint block-diagonal graph + a GraphInput."""
    Xs: list[torch.Tensor] = []
    yi: list[torch.Tensor] = []
    yf: list[torch.Tensor] = []
    eis: list[torch.Tensor] = []
    ews: list[torch.Tensor] = []
    bidx: list[torch.Tensor] = []
    off: int = 0

    for b, item in enumerate(batch):
        n: int = item["num_nodes"]
        Xs.append(item["X"])
        yi.append(item["y_inf"])
        yf.append(item["y_fr"])
        eis.append(
            item["edge_index"] + off
        )  # offset node ids into the block-diagonal graph
        ews.append(item["edge_weight"])
        bidx.append(torch.full((n,), b, dtype=torch.long))
        off += n

    X = torch.cat(Xs).to(device)
    y_inf = torch.cat(yi).to(device)
    y_fr = torch.cat(yf).to(device)
    edge_index: torch.Tensor = (
        torch.cat(eis, dim=1) if eis else torch.zeros(2, 0, dtype=torch.long)
    )
    edge_weight: torch.Tensor = (
        torch.cat(ews) if ews else torch.zeros(0, dtype=torch.float32)
    )
    gi = build_graph_input(
        edge_index.numpy(), edge_weight.numpy(), off, diffusion_model, device
    )
    return {
        "X": X,
        "y_inf": y_inf,
        "y_fr": y_fr,
        "graph": gi,
        "batch_index": torch.cat(bidx).to(device),
        "records": [it["record"] for it in batch],
    }
