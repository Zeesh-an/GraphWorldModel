"""
Shared graph utilities for data generation

Common graph preprocessing functions used across inverse graph problem data generators.
"""

import os
from collections import defaultdict
from pathlib import Path
import numpy as np
import scipy.sparse as sp


def build_edge_index(
    adjacency: sp.csr_matrix,
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
    coo = adjacency.tocoo()
    sources = coo.row.astype(np.int32)
    destinations = coo.col.astype(np.int32)

    in_degrees = np.array(adjacency.sum(axis=0)).flatten()  # (N,) in-degree
    in_degrees = np.where(in_degrees == 0, 1, in_degrees)  # avoid divide-by-zero

    # Independent Cascade (IC): each edge gets probability = 1/in_degree(dst)
    ic_probs = (1.0 / in_degrees[destinations]).astype(np.float32)

    # DeepIM does not use clipping
    # ic_probs = np.clip(ic_probs, 0.001, 0.5)  # cap for realism

    # Linear Threshold (LT): weights must sum to ≤ 1 per node (already true with 1/in_deg)
    lt_weights = ic_probs.copy()

    edge_index = np.stack([sources, destinations], axis=0)  # (2, E)

    return edge_index, ic_probs, lt_weights


def build_adjacency_lists(
    sources: np.ndarray,
    destinations: np.ndarray,
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
    # in_adjacency[v] = list of (neighbor_u, prob_u_v) incoming edges
    in_adjacency = defaultdict(list)  # in_adjacency[v]  = [(u, p), ...] used by LT
    out_adjacency = defaultdict(list)  # out_adjacency[u] = [(v, p), ...] used by IC

    for edge, (source, destination) in enumerate(zip(sources, destinations)):
        # Outgoing edges from the source (used by IC)
        out_adjacency[int(source)].append((int(destination), float(ic_probs[edge])))

        # Incoming edges to the destination (used by LT)
        in_adjacency[int(destination)].append((int(source), float(lt_weights[edge])))

    return dict(out_adjacency), dict(in_adjacency)


def save_graph(
    out_dir: Path,
    edge_index: np.ndarray,
    ic_probs: np.ndarray,
    lt_weights: np.ndarray,
    node_feats: np.ndarray,
    node_labels: np.ndarray,
) -> Path:
    os.makedirs(out_dir, exist_ok=True)
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


def load_graph(data_dir: Path) -> dict:
    """Load graph data. Returns dict with numpy arrays."""
    data = np.load(data_dir / "graph_data.npz")

    return {key: data[key] for key in data.files}
