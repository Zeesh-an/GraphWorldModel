"""
NetHEPT Dataset Loader

Downloads and loads the NetHEPT high-energy physics theory citation network.

Source: https://github.com/SparklyYS/Simultaneous-IMM (mirror of Wei Chen's data)
    - 15,229 nodes (papers), 62,752 directed edges (citations)
    - Directed: edge (a, b) means paper a cites paper b
    - Standard benchmark for Influence Maximization
    - No inherent node features — uses log(1 + total degree) as synthetic features
    - No node labels

Original paper: Wei Chen et al., "Efficient Influence Maximization
    in Social Networks," KDD 2009
"""

import urllib.request
import os
import numpy as np
import scipy.sparse as sp
from pathlib import Path

nethept_graph_url = (
    "https://raw.githubusercontent.com/SparklyYS/Simultaneous-IMM/"
    "master/nethept/graph.txt"
)
nethept_attr_url = (
    "https://raw.githubusercontent.com/SparklyYS/Simultaneous-IMM/"
    "master/nethept/attribute.txt"
)
data_dir = Path(__file__).resolve().parent.parent / "nethept"


def download_nethept() -> Path:
    """Download NetHEPT edge list if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    graph_path = data_dir / "graph.txt"
    attr_path = data_dir / "attribute.txt"

    if graph_path.exists():
        print(f"[✓] NetHEPT already downloaded at {graph_path}")
        return graph_path

    print("[↓] Downloading NetHEPT edge list ...")
    urllib.request.urlretrieve(nethept_graph_url, graph_path)
    print(f"[✓] Saved edge list to {graph_path}")

    print("[↓] Downloading NetHEPT attributes ...")
    urllib.request.urlretrieve(nethept_attr_url, attr_path)
    print(f"[✓] Saved attributes to {attr_path}")

    return graph_path


def load_nethept(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Load NetHEPT citation network. Edges are 0-indexed space-separated pairs.
    Kept as a directed graph (like Cora-ML).

    Returns
    -------
    adj: scipy.sparse.csr_matrix (N, N) directed adjacency
    node_feats: np.ndarray (N, 1) float32 -- log(1 + degree) features
    node_labels: np.ndarray (N,) int32 -- placeholder zeros
    N: int -- number of nodes
    """
    # Read attribute.txt to get expected node/edge counts
    attr_path = path.parent / "attribute.txt"
    expected_n = None
    if attr_path.exists():
        with open(attr_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("n="):
                    expected_n = int(line.split("=")[1])

    # Parse space-separated directed edge list (0-indexed)
    src_list = []
    dst_list = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            a, b = int(parts[0]), int(parts[1])
            src_list.append(a)
            dst_list.append(b)

    # Determine N from attribute file or from max node ID
    all_ids = set(src_list) | set(dst_list)
    max_id = max(all_ids)

    if expected_n is not None:
        N = expected_n
    else:
        N = max_id + 1

    # IDs are already 0-indexed contiguous (verified from data source)
    src = np.array(src_list, dtype=np.int32)
    dst = np.array(dst_list, dtype=np.int32)
    data = np.ones(len(src), dtype=np.float32)

    # Build directed sparse adjacency
    adj = sp.csr_matrix((data, (src, dst)), shape=(N, N))
    adj = (adj > 0).astype(np.float32)
    adj.setdiag(0)
    adj.eliminate_zeros()

    # Degree-based node features using total degree (in + out)
    in_degrees = np.array(adj.sum(axis=0)).flatten()
    out_degrees = np.array(adj.sum(axis=1)).flatten()
    total_degrees = in_degrees + out_degrees
    node_feats = (
        np.log1p(total_degrees).reshape(-1, 1).astype(np.float32)
    )  # shape: (N, 1)

    # Placeholder labels
    node_labels = np.zeros(N, dtype=np.int32)

    print(f"[✓] NetHEPT loaded: {N} nodes, {adj.nnz} directed edges")
    print(f"    In-degree  — avg: {in_degrees.mean():.1f}, max: {in_degrees.max():.0f}")
    print(
        f"    Out-degree — avg: {out_degrees.mean():.1f}, max: {out_degrees.max():.0f}"
    )

    return adj, node_feats, node_labels, N
