"""
Digg Dataset Loader

Downloads and loads the Digg social network.

Source: https://datasets.syr.edu/datasets/Digg.html
    - 116,893 core users, ~2.6M friendship edges (undirected)
    - Undirected: mutual friendships
    - Edges to users outside the core set are dropped
    - No inherent node features — uses log(1 + degree) as synthetic features
    - No node labels
"""

import csv
import zipfile
import urllib.request
import os
import numpy as np
import scipy.sparse as sp
from pathlib import Path

digg_url = "https://datasets.syr.edu/uploads/1296588940/Digg-dataset.zip"
data_dir = Path(__file__).resolve().parent.parent / "digg"


def download_digg() -> Path:
    """Download and extract Digg dataset if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    zip_path = data_dir / "Digg-dataset.zip"
    nodes_path = data_dir / "Digg-dataset" / "data" / "nodes.csv"

    if nodes_path.exists():
        print(f"[✓] Digg already downloaded at {data_dir}")
        return data_dir / "Digg-dataset" / "data"

    if not zip_path.exists():
        print(f"[↓] Downloading Digg from {digg_url} ...")
        urllib.request.urlretrieve(digg_url, zip_path)
        print(f"[✓] Saved to {zip_path}")

    print("[↓] Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    print(f"[✓] Extracted to {data_dir / 'Digg-dataset'}")

    return data_dir / "Digg-dataset" / "data"


def load_digg(data_path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Load Digg social network. Keeps only edges between core nodes
    (those listed in nodes.csv). Remaps IDs to contiguous 0..N-1.

    Returns
    -------
    adj: scipy.sparse.csr_matrix (N, N) undirected adjacency
    node_feats: np.ndarray (N, 1) float32 -- log(1 + degree) features
    node_labels: np.ndarray (N,) int32 -- placeholder zeros
    N: int -- number of core nodes
    """
    # Load core node IDs
    core_ids = []
    with open(data_path / "nodes.csv") as f:
        for line in f:
            line = line.strip()
            if line:
                core_ids.append(int(line))

    core_set = set(core_ids)

    # Remap non-contiguous IDs to contiguous IDs from 0, ..., N - 1
    id_to_idx: dict[int, int] = {nid: idx for idx, nid in enumerate(sorted(core_ids))}
    N = len(id_to_idx)

    # Load edges, keep only those between core nodes
    src_list = []
    dst_list = []
    n_dropped = 0

    with open(data_path / "edges.csv") as f:
        reader = csv.reader(f)

        for row in reader:
            # Edge (a - b)
            a, b = int(row[0]), int(row[1])

            if a in core_set and b in core_set:
                ia, ib = id_to_idx[a], id_to_idx[b]

                # Store both directions for undirected graph
                src_list.append(ia)
                dst_list.append(ib)
                src_list.append(ib)
                dst_list.append(ia)
            else:
                n_dropped += 1

    src = np.array(src_list, dtype=np.int32)
    dst = np.array(dst_list, dtype=np.int32)
    data = np.ones(len(src), dtype=np.float32)

    # Build sparse adjacency, deduplicate via csr conversion
    adj = sp.csr_matrix((data, (src, dst)), shape=(N, N))
    adj = (adj > 0).astype(np.float32)
    adj.setdiag(0)
    adj.eliminate_zeros()

    # Degree-based node features (no natural features in Digg)
    degrees = np.array(adj.sum(axis=1)).flatten()
    node_feats = np.log1p(degrees).reshape(-1, 1).astype(np.float32)  # shape: (N, 1)

    # Placeholder labels (Digg has no node labels)
    node_labels = np.zeros(N, dtype=np.int32)

    n_edges_undirected = adj.nnz // 2
    print(f"[✓] Digg loaded: {N} nodes, {n_edges_undirected} undirected edges")
    print(f"    Dropped {n_dropped} edges to external nodes")
    print(f"    Avg degree: {degrees.mean():.1f}, max degree: {degrees.max():.0f}")

    return adj, node_feats, node_labels, N
