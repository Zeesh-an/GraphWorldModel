"""
Orkut Dataset Loader

Downloads and loads the SNAP com-Orkut friendship network: the largest of Han
et al.'s adaptive-IM benchmarks (research/adaptive_online_im.md §5.2, §6.2).

Source: https://snap.stanford.edu/data/com-Orkut.html
    - 3,072,441 nodes (members), 117,185,083 undirected friendships
    - Undirected; the densest graph in the whole catalogue at average degree 76.2
    - Han et al. quote `3.07M / 117M`, average degree 76.2: agrees with SNAP to
      the digit, so there is no version collision here
    - The SNAP release also ships ground-truth communities; we load the ungraph
      edge list only, since nothing in this pipeline consumes community labels
    - No inherent node features: uses log(1 + degree) as synthetic features
    - No node labels

SCALE WARNING. 117M undirected edges is ~234M arcs after symmetrization, which
is beyond both the NDlib rollout and the dense-ish structures the selector
pipeline builds. Same standing caveat as `livejournal`, `weibo` and `twitter`
(see CLAUDE.md): the loader exists because Han et al. report on this graph, not
because day-one experiments run on it.

Original paper: Yang & Leskovec, "Defining and Evaluating Network Communities
    Based on Ground-Truth," ICDM 2012
"""

import gzip
import os
import urllib.request
from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.graph_utils import degree_features, edges_to_adjacency, remap_to_contiguous

orkut_url = "https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz"
data_dir = Path(__file__).resolve().parent.parent / "raw" / "orkut"


def download_orkut() -> Path:
    """Download and extract the friendship edge list (~1.7 GB uncompressed)."""
    os.makedirs(data_dir, exist_ok=True)
    gz_path = data_dir / "com-orkut.ungraph.txt.gz"
    txt_path = data_dir / "com-orkut.ungraph.txt"

    if txt_path.exists():
        print(f"[ok] Orkut already downloaded at {txt_path}")
        return txt_path

    if not gz_path.exists():
        print(f"[get] Downloading Orkut from {orkut_url} ...")
        urllib.request.urlretrieve(orkut_url, gz_path)
        print(f"[ok] Saved to {gz_path}")

    print("[get] Extracting (this file is large) ...")
    with gzip.open(gz_path, "rb") as gz_file:
        txt_path.write_bytes(gz_file.read())
    print(f"[ok] Extracted to {txt_path}")

    return txt_path


def load_orkut(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Load com-Orkut as an undirected graph.

    Returns
    -------
    adjacency: scipy.sparse.csr_matrix (N, N) symmetric adjacency
    node_feats: np.ndarray (N, 1) float32 -- log(1 + degree) features
    node_labels: np.ndarray (N,) int32 -- placeholder zeros
    num_nodes: int -- number of nodes
    """
    raw_edges = np.loadtxt(path, comments="#", dtype=np.int64)  # shape: (E, 2)

    remapped, num_nodes = remap_to_contiguous(raw_edges)
    adjacency = edges_to_adjacency(
        remapped[:, 0], remapped[:, 1], num_nodes, directed=False
    )

    node_feats = degree_features(adjacency, directed=False)  # shape: (N, 1)
    node_labels = np.zeros(num_nodes, dtype=np.int32)

    print(
        f"[ok] Orkut loaded: {num_nodes} nodes, "
        f"{adjacency.nnz // 2} undirected edges"
    )

    return adjacency, node_feats, node_labels, num_nodes
