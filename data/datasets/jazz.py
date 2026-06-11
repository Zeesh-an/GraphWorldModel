"""
Jazz Musicians Dataset Loader

Downloads and loads the Jazz musicians collaboration network.

Source: https://networkrepository.com/arenas-jazz.php
    - 198 nodes (musicians), 2,742 edges (collaborations)
    - Undirected, unweighted
    - No inherent node features — uses log(1 + degree) as synthetic features
    - No node labels

Original paper: Gleiser & Danon, "Community Structure in Jazz",
    Advances in Complex Systems 6, 565 (2003)
"""

import zipfile
import urllib.request
import numpy as np
import scipy.sparse as sp
from pathlib import Path

JAZZ_URL = "https://nrvis.com/download/data/misc/arenas-jazz.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "jazz"


def download_jazz() -> Path:
    """Download and extract Jazz dataset if not already present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "arenas-jazz.zip"
    edges_path = DATA_DIR / "arenas-jazz.edges"

    if edges_path.exists():
        print(f"[✓] Jazz already downloaded at {edges_path}")
        return edges_path

    if not zip_path.exists():
        print(f"[↓] Downloading Jazz from {JAZZ_URL} ...")
        urllib.request.urlretrieve(JAZZ_URL, zip_path)
        print(f"[✓] Saved to {zip_path}")

    print("[↓] Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)
    print(f"[✓] Extracted to {DATA_DIR}")

    return edges_path


def load_jazz(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Load Jazz collaboration network. Edges are 1-indexed comma-separated pairs.
    Remaps IDs to contiguous 0..N-1.

    Returns
    -------
    adj: scipy.sparse.csr_matrix (N, N) undirected adjacency
    node_feats: np.ndarray (N, 1) float32 -- log(1 + degree) features
    node_labels: np.ndarray (N,) int32 -- placeholder zeros
    N: int -- number of nodes
    """
    src_list = []
    dst_list = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            # Skip header/comment lines (start with %)
            if not line or line.startswith("%"):
                continue
            parts = line.split(",")
            a, b = int(parts[0]), int(parts[1])
            src_list.append(a)
            dst_list.append(b)

    # Collect unique node IDs and remap to contiguous 0..N-1
    all_ids = sorted(set(src_list) | set(dst_list))
    id_to_idx = {nid: idx for idx, nid in enumerate(all_ids)}
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
    print(f"[✓] Jazz loaded: {N} nodes, {n_edges_undirected} undirected edges")
    print(f"    Avg degree: {degrees.mean():.1f}, max degree: {degrees.max():.0f}")

    return adj, node_feats, node_labels, N
