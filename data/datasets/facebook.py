"""
Facebook Ego-Networks Dataset Loader

Downloads and loads the SNAP ego-Facebook combined friendship network.

Source: https://snap.stanford.edu/data/ego-Facebook.html
    - 4,039 nodes (users), 88,234 edges (friendships)
    - Undirected, unweighted; the union of 10 ego networks
    - Already a single connected component, so no LCC decision to make — the
      one graph in the suite where our numbers and MOEIM's agree with no
      preprocessing at all
    - Strong community structure (avg degree 43.7, high clustering)
    - No inherent node features — uses log(1 + degree) as synthetic features
    - No node labels

Original paper: McAuley & Leskovec, "Learning to Discover Social Circles in Ego
    Networks," NIPS 2012
"""

import gzip
import os
import urllib.request
from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.graph_utils import degree_features, edges_to_adjacency

facebook_url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
data_dir = Path(__file__).resolve().parent.parent / "raw" / "facebook"


def download_facebook() -> Path:
    """Download and extract the combined ego-network edge list."""
    os.makedirs(data_dir, exist_ok=True)
    gz_path = data_dir / "facebook_combined.txt.gz"
    txt_path = data_dir / "facebook_combined.txt"

    if txt_path.exists():
        print(f"[✓] Facebook already downloaded at {txt_path}")
        return txt_path

    if not gz_path.exists():
        print(f"[↓] Downloading Facebook from {facebook_url} ...")
        urllib.request.urlretrieve(facebook_url, gz_path)
        print(f"[✓] Saved to {gz_path}")

    print("[↓] Extracting ...")
    with gzip.open(gz_path, "rb") as gz_file:
        txt_path.write_bytes(gz_file.read())
    print(f"[✓] Extracted to {txt_path}")

    return txt_path


def load_facebook(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Load the combined Facebook ego networks. Space-separated undirected pairs
    listed once, ids already contiguous 0..4038.

    Returns
    -------
    adjacency: scipy.sparse.csr_matrix (N, N) undirected adjacency
    node_feats: np.ndarray (N, 1) float32 -- log(1 + degree) features
    node_labels: np.ndarray (N,) int32 -- placeholder zeros
    num_nodes: int -- number of nodes
    """
    raw_edges = np.loadtxt(path, comments="#", dtype=np.int64)  # shape: (E, 2)
    num_nodes = int(raw_edges.max()) + 1

    adjacency = edges_to_adjacency(
        raw_edges[:, 0], raw_edges[:, 1], num_nodes, directed=False
    )

    node_feats = degree_features(adjacency, directed=False)  # shape: (N, 1)
    node_labels = np.zeros(num_nodes, dtype=np.int32)

    degrees = np.array(adjacency.sum(axis=1)).flatten()
    n_edges_undirected = adjacency.nnz // 2
    print(
        f"[✓] Facebook loaded: {num_nodes} nodes, {n_edges_undirected} undirected edges"
    )
    print(f"    Avg degree: {degrees.mean():.1f}, max degree: {degrees.max():.0f}")

    return adjacency, node_feats, node_labels, num_nodes
