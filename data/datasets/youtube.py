"""
YouTube Dataset Loader

Downloads and loads the SNAP com-Youtube social network.

Source: https://snap.stanford.edu/data/com-Youtube.html
    - 1,134,890 nodes (users), 2,987,624 edges (friendships)
    - Undirected: mutual friendships
    - No inherent node features: uses log(1 + degree) as synthetic features
    - No node labels

Original paper: Yang & Leskovec, "Defining and Evaluating Network
    Communities based on Ground-truth," ICDM 2012
"""

import gzip
import os
import urllib.request
from pathlib import Path
import numpy as np
import scipy.sparse as sp

youtube_url = (
    "https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz"
)
data_dir = Path(__file__).resolve().parent.parent / "raw" / "youtube"


def download_youtube() -> Path:
    """Download and extract YouTube edge list if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    gz_path = data_dir / "com-youtube.ungraph.txt.gz"
    txt_path = data_dir / "com-youtube.ungraph.txt"

    if txt_path.exists():
        print(f"[ok] YouTube already downloaded at {txt_path}")
        return txt_path

    if not gz_path.exists():
        print(f"[get] Downloading YouTube from {youtube_url} ...")
        urllib.request.urlretrieve(youtube_url, gz_path)
        print(f"[ok] Saved to {gz_path}")

    print("[get] Extracting ...")
    with gzip.open(gz_path, "rb") as gz_file:
        with open(txt_path, "wb") as txt_file:
            txt_file.write(gz_file.read())
    print(f"[ok] Extracted to {txt_path}")

    return txt_path


def load_youtube(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Load YouTube friendship network. Edges are tab-separated pairs (each
    undirected edge listed once, # comment header). Remaps IDs to contiguous
    0..N-1.

    Returns
    -------
    adjacency: scipy.sparse.csr_matrix (N, N) undirected adjacency
    node_feats: np.ndarray (N, 1) float32 -- log(1 + degree) features
    node_labels: np.ndarray (N,) int32 -- placeholder zeros
    num_nodes: int -- number of nodes
    """
    # ~3M rows: vectorized parse instead of a Python line loop
    raw_edges = np.loadtxt(path, comments="#", dtype=np.int64)  # shape: (E, 2)

    all_ids = np.unique(raw_edges)
    num_nodes = int(all_ids.shape[0])
    remapped = np.searchsorted(all_ids, raw_edges)  # shape: (E, 2) contiguous ids

    # Store both directions for the undirected graph
    sources = np.concatenate([remapped[:, 0], remapped[:, 1]]).astype(np.int32)
    destinations = np.concatenate([remapped[:, 1], remapped[:, 0]]).astype(np.int32)
    values = np.ones(len(sources), dtype=np.float32)

    # Build sparse adjacency, deduplicate via csr conversion
    adjacency = sp.csr_matrix(
        (values, (sources, destinations)), shape=(num_nodes, num_nodes)
    )
    adjacency = (adjacency > 0).astype(np.float32)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    # Degree-based node features (no natural features in this dataset)
    degrees = np.array(adjacency.sum(axis=1)).flatten()
    node_feats = np.log1p(degrees).reshape(-1, 1).astype(np.float32)  # shape: (N, 1)

    # Placeholder labels
    node_labels = np.zeros(num_nodes, dtype=np.int32)

    n_edges_undirected = adjacency.nnz // 2
    print(
        f"[ok] YouTube loaded: {num_nodes} nodes, "
        f"{n_edges_undirected} undirected edges"
    )
    print(f"    Avg degree: {degrees.mean():.1f}, max degree: {degrees.max():.0f}")

    return adjacency, node_feats, node_labels, num_nodes
