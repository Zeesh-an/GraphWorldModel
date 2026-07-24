"""
Cora-ML Dataset Loader

Downloads and loads the Cora-ML citation network.

Source: https://github.com/abojchevski/graph2gauss
    - 2,995 nodes (scientific papers), 8,416 directed edges (citations)
    - Directed: edge (a, b) means paper a cites paper b
    - Node features: 2,879-dim bag-of-words vectors
    - Node labels: 7 topic classes

Original paper: A. McCallum et al., "Automating the Construction of
    Internet Portals with Machine Learning," Information Retrieval (2000)
"""

import os
import urllib.request
from pathlib import Path
import numpy as np
import scipy.sparse as sp

cora_ml_url = "https://github.com/abojchevski/graph2gauss/raw/master/data/cora_ml.npz"
data_dir = Path(__file__).resolve().parent.parent / "raw" / "cora_ml"


def download_cora_ml() -> Path:
    os.makedirs(data_dir, exist_ok=True)
    destination = data_dir / "cora_ml.npz"

    if destination.exists():
        print(f"[✓] Cora-ML already downloaded at {destination}")
        return destination

    print(f"[↓] Downloading Cora-ML from {cora_ml_url} ...")
    urllib.request.urlretrieve(cora_ml_url, destination)
    print(f"[✓] Saved to {destination}")

    return destination


def load_cora_ml(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Returns
    -------
    adjacency: scipy.sparse.csr_matrix (N, N) binary directed adjacency
    features: np.ndarray (N, F) float32 bag-of-words
    labels: np.ndarray (N,) int32 class labels
    num_nodes: int -- number of nodes
    """
    raw = np.load(path, allow_pickle=True)

    # The npz stores the adjacency as a sparse matrix in COO format
    # Reconstruct the (N, N) sparse binary adjacency matrix from CSR components
    adjacency = sp.csr_matrix(
        (raw["adj_data"], raw["adj_indices"], raw["adj_indptr"]),
        shape=raw["adj_shape"],
    ).astype(np.float32)

    features = (
        sp.csr_matrix(
            (raw["attr_data"], raw["attr_indices"], raw["attr_indptr"]),
            shape=raw["attr_shape"],
        )
        .toarray()
        .astype(np.float32)
    )

    labels = raw["labels"].astype(np.int32)

    print(
        f"[✓] Cora-ML loaded: {adjacency.shape[0]} nodes, "
        f"{adjacency.nnz} edges, {features.shape[1]} features, "
        f"{len(np.unique(labels))} classes"
    )
    num_nodes = adjacency.shape[0]

    return adjacency, features, labels, num_nodes
