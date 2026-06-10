"""
Shared graph utilities for data generation

Common graph preprocessing functions used across inverse graph problem data generators.
"""

import numpy as np
import scipy.sparse as sp
from pathlib import Path
from collections import defaultdict


def build_edge_index(
    adj: sp.csr_matrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert adjacency to COO edge_index and compute edge probabilities.

    IC probability p( u → v) = 1 / in_degree(v) [weighted cascade model]
    LT weight w (u → v) = 1 / in_degree(v) [same, but semantics differ]

    Returns
    -------
    edge_index: (2, E) int32
    ic_probs: (E,) float32
    lt_weights: (E,) float32
    """
    # Convert to COO and extract source and destination arrays from adjacency matrix
    adj_coo = adj.tocoo()
    src = adj_coo.row.astype(np.int32)
    dst = adj_coo.col.astype(np.int32)

    N = adj.shape[0]
    in_deg = np.array(adj.sum(axis=0)).flatten()  # (N,) in-degree
    in_deg = np.where(in_deg == 0, 1, in_deg)  # avoid divide-by-zero

    # Independent Cascade (IC): each edge gets probability = 1/in_degree(dst)
    ic_probs = (1.0 / in_deg[dst]).astype(np.float32)

    # DeepIM does not use clipping
    # ic_probs = np.clip(ic_probs, 0.001, 0.5)  # cap for realism

    # Linear Threshold (LT): weights must sum to ≤ 1 per node (already true with 1/in_deg)
    lt_weights = ic_probs.copy()

    edge_index = np.stack([src, dst], axis=0)  # (2, E)

    return edge_index, ic_probs, lt_weights


def build_adjacency_lists(
    src: np.ndarray,
    dst: np.ndarray,
    ic_probs: np.ndarray,
    lt_weights: np.ndarray,
) -> tuple[dict, dict]:
    """
    Build adjacency list dicts for fast simulation.

    Returns
    -------
    out_adj: dict[int, list[tuple[int, float]]] out_adj[u] = [(v, p), ...]  used by Independent Cascade (IC)
    in_adj: dict[int, list[tuple[int, float]]] in_adj[v] = [(u, w), ...]  used by Linear Threshold (LT)
    """
    # adj_list[v] = list of (neighbor_u, prob_u_v) incoming edges
    in_adj = defaultdict(list)  # in_adj[v]  = [(u, p), ...] used by LT
    out_adj = defaultdict(list)  # out_adj[u] = [(v, p), ...] used by IC

    for i, (u, v) in enumerate(zip(src, dst)):
        # List of (v, p) for outgoing edges from u (used by IC)
        out_adj[int(u)].append((int(v), float(ic_probs[i])))

        # List of (u, w) for incoming edges to v (used by LT)
        in_adj[int(v)].append((int(u), float(lt_weights[i])))

    return dict(out_adj), dict(in_adj)


def save_graph(
    out_dir: Path,
    edge_index: np.ndarray,
    ic_probs: np.ndarray,
    lt_weights: np.ndarray,
    node_feats: np.ndarray,
    node_labels: np.ndarray,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "graph_data.npz"
    np.savez_compressed(
        out_path,
        edge_index=edge_index,
        ic_probs=ic_probs,
        lt_weights=lt_weights,
        node_feats=node_feats,
        node_labels=node_labels,
    )
    print(f"[✓] Saved graph data to {out_path}")

    return out_path


def load_graph(data_dir: Path):
    """Load graph data. Returns dict with numpy arrays."""
    d = np.load(data_dir / "graph_data.npz")

    return {k: d[k] for k in d.files}
