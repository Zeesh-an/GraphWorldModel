"""
NetScience Dataset Loader
=========================

Downloads and loads the Network Science coauthorship network.

Source: https://networks.skewed.de/net/netscience
    - 1,589 nodes (authors), 2,742 edges (coauthorships)
    - Undirected, weighted (collaboration strength)
    - Weights are discarded — we use binary adjacency
    - No inherent node features — uses log(1 + degree) as synthetic features
    - No node labels

Original paper: M. E. J. Newman, Phys. Rev. E 74, 036104 (2006)
"""

import csv
import zipfile
import urllib.request
import numpy as np
import scipy.sparse as sp
from pathlib import Path


NETSCIENCE_URL = "https://networks.skewed.de/net/netscience/files/netscience.csv.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "netscience"


def download_netscience() -> Path:
    """Download and extract NetScience dataset if not already present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "netscience.csv.zip"
    edges_path = DATA_DIR / "edges.csv"

    if edges_path.exists():
        print(f"[✓] NetScience already downloaded at {DATA_DIR}")
        return DATA_DIR

    if not zip_path.exists():
        print(f"[↓] Downloading NetScience from {NETSCIENCE_URL} ...")
        urllib.request.urlretrieve(NETSCIENCE_URL, zip_path)
        print(f"[✓] Saved to {zip_path}")

    print("[↓] Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)
    print(f"[✓] Extracted to {DATA_DIR}")

    return DATA_DIR


def load_netscience(data_path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Load NetScience coauthorship network. Edges are 0-indexed CSV with columns
    (source, target, value). The value (edge weight) is discarded.

    Returns
    -------
    adj: scipy.sparse.csr_matrix (N, N) undirected adjacency
    node_feats: np.ndarray (N, 1) float32 -- log(1 + degree) features
    node_labels: np.ndarray (N,) int32 -- placeholder zeros
    N: int -- number of nodes
    """
    # Read nodes.csv to determine N (nodes may be isolates not in edges.csv)
    node_ids = set()
    nodes_path = data_path / "nodes.csv"
    if nodes_path.exists():
        with open(nodes_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                node_ids.add(int(row["index"]))

    # Read edges
    src_list = []
    dst_list = []

    with open(data_path / "edges.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            a, b = int(row["source"]), int(row["target"])
            src_list.append(a)
            dst_list.append(b)
            node_ids.add(a)
            node_ids.add(b)

    # Remap to contiguous 0..N-1 (should already be 0-indexed but may have gaps from isolates)
    sorted_ids = sorted(node_ids)
    id_to_idx = {nid: idx for idx, nid in enumerate(sorted_ids)}
    N = len(id_to_idx)

    # Build symmetric (undirected) edge arrays
    sym_src = []
    sym_dst = []

    for a, b in zip(src_list, dst_list):
        ia, ib = id_to_idx[a], id_to_idx[b]
        sym_src.append(ia)
        sym_dst.append(ib)
        sym_src.append(ib)
        sym_dst.append(ia)

    src = np.array(sym_src, dtype=np.int32)
    dst = np.array(sym_dst, dtype=np.int32)
    data = np.ones(len(src), dtype=np.float32)

    # Build sparse adjacency, deduplicate via csr conversion
    adj = sp.csr_matrix((data, (src, dst)), shape=(N, N))
    adj = (adj > 0).astype(np.float32)
    adj.setdiag(0)
    adj.eliminate_zeros()

    # Degree-based node features (no natural features in this dataset)
    degrees = np.array(adj.sum(axis=1)).flatten()
    node_feats = np.log1p(degrees).reshape(-1, 1).astype(np.float32)  # shape: (N, 1)

    # Placeholder labels
    node_labels = np.zeros(N, dtype=np.int32)

    n_edges_undirected = adj.nnz // 2
    n_isolates = int((degrees == 0).sum())
    print(f"[✓] NetScience loaded: {N} nodes, {n_edges_undirected} undirected edges")
    print(f"    Avg degree: {degrees.mean():.1f}, max degree: {degrees.max():.0f}")
    if n_isolates > 0:
        print(f"    Isolate nodes (degree 0): {n_isolates}")

    return adj, node_feats, node_labels, N
