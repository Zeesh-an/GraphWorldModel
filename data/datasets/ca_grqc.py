"""
CA-GrQc Dataset Loader

Downloads and loads the SNAP arXiv General Relativity collaboration network.

Source: https://snap.stanford.edu/data/ca-GrQc.html
    - 5,242 nodes (authors), 14,496 undirected edges (co-authorships)
    - Undirected: an edge means the two authors co-wrote at least one paper
    - The file's own header says "each unordered pair of nodes is saved once",
      but it actually lists 28,980 lines — both directions — plus 12 self-loops,
      so the deduplicated undirected count is 14,484
    - Sparse original ids (max 26,196) — remapped to 0..N-1
    - No inherent node features — uses log(1 + degree) as synthetic features
    - No node labels

ToupleGDD reports this graph as "caGr" at 4.2k / 13.4k, which is its largest
connected component (4,158 / 13,422); GLIE reports "GR Colab" at 5,242 / 28,980,
counting the raw lines (research/influence_maximization.md §6.3).

Original paper: Leskovec et al., "Graph Evolution: Densification and Shrinking
    Diameters," ACM TKDD 2007
"""

import gzip
import os
import urllib.request
from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.graph_utils import degree_features, edges_to_adjacency, remap_to_contiguous

ca_grqc_url = "https://snap.stanford.edu/data/ca-GrQc.txt.gz"
data_dir = Path(__file__).resolve().parent.parent / "raw" / "ca_grqc"


def download_ca_grqc() -> Path:
    """Download and extract the CA-GrQc edge list if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    gz_path = data_dir / "ca-GrQc.txt.gz"
    txt_path = data_dir / "ca-GrQc.txt"

    if txt_path.exists():
        print(f"[✓] CA-GrQc already downloaded at {txt_path}")
        return txt_path

    if not gz_path.exists():
        print(f"[↓] Downloading CA-GrQc from {ca_grqc_url} ...")
        urllib.request.urlretrieve(ca_grqc_url, gz_path)
        print(f"[✓] Saved to {gz_path}")

    print("[↓] Extracting ...")
    with gzip.open(gz_path, "rb") as gz_file:
        txt_path.write_bytes(gz_file.read())
    print(f"[✓] Extracted to {txt_path}")

    return txt_path


def load_ca_grqc(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Load the CA-GrQc collaboration network. Tab-separated pairs with a `#`
    header; ids are sparse so they are remapped to 0..N-1.

    Returns
    -------
    adjacency: scipy.sparse.csr_matrix (N, N) undirected adjacency
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

    degrees = np.array(adjacency.sum(axis=1)).flatten()
    n_edges_undirected = adjacency.nnz // 2
    print(
        f"[✓] CA-GrQc loaded: {num_nodes} nodes, "
        f"{n_edges_undirected} undirected edges (from {raw_edges.shape[0]} raw lines)"
    )
    print(f"    Avg degree: {degrees.mean():.1f}, max degree: {degrees.max():.0f}")

    return adjacency, node_feats, node_labels, num_nodes
