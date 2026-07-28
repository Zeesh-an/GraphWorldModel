"""
Cora-ML Dataset Loader

Downloads and loads the Cora-ML citation network, standardized the way the IM
literature uses it.

Source: https://github.com/abojchevski/graph2gauss
    - raw file:  2,995 nodes, 8,416 directed arcs
    - as loaded: 2,810 nodes, 7,981 undirected edges
    - Node features: 2,879-dim bag-of-words vectors
    - Node labels: 7 topic classes

We apply graph2gauss's own `SparseGraph.standardize()`, whose four arguments
(`make_unweighted`, `make_undirected`, `no_self_loops`, `select_lcc`) all default
to True:

    binarize -> symmetrize -> drop self-loops -> keep largest component

DeepIM and MOEIM load this same `cora_ml.npz` through that class, so their
published "2,810 / 7,981" is exactly what this loader produces — which is the
whole reason Cora-ML is in this repo. The raw 2,995-node graph is not comparable
to any published table, and its 185 extra nodes sit in ~60 tiny fragments that
only add noise to a spread metric. See IM_DATASETS.md §6.2.

Original paper: A. McCallum et al., "Automating the Construction of
    Internet Portals with Machine Learning," Information Retrieval (2000)
"""

import os
import urllib.request
from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.graph_utils import largest_connected_component

cora_ml_url = "https://github.com/abojchevski/graph2gauss/raw/master/data/cora_ml.npz"
data_dir = Path(__file__).resolve().parent.parent / "raw" / "cora_ml"


def download_cora_ml() -> Path:
    os.makedirs(data_dir, exist_ok=True)
    destination = data_dir / "cora_ml.npz"

    if destination.exists():
        print(f"[✓] Cora-ML already downloaded at {destination}")
        return destination

    # 85 MB — the slowest download in the suite. A run killed part-way leaves a
    # truncated file that later fails with BadZipFile, so write to a .part file
    # and only rename once it is complete.
    print(f"[↓] Downloading Cora-ML from {cora_ml_url} (85 MB) ...")
    partial = destination.with_suffix(".npz.part")
    urllib.request.urlretrieve(cora_ml_url, partial)
    partial.rename(destination)
    print(f"[✓] Saved to {destination}")

    return destination


def load_cora_ml(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Returns
    -------
    adjacency: scipy.sparse.csr_matrix (N, N) undirected adjacency, LCC only
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
    raw_num_nodes = adjacency.shape[0]

    # standardize(): make_unweighted + make_undirected + no_self_loops
    adjacency = ((adjacency + adjacency.T) > 0).astype(np.float32)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    # standardize(): select_lcc — features and labels take the same indices
    adjacency, keep = largest_connected_component(adjacency)
    features = features[keep]
    labels = labels[keep]
    num_nodes = int(keep.shape[0])

    degrees = np.array(adjacency.sum(axis=1)).flatten()
    print(
        f"[✓] Cora-ML loaded: {num_nodes} nodes, {adjacency.nnz // 2} undirected "
        f"edges, {features.shape[1]} features, {len(np.unique(labels))} classes"
    )
    print(
        f"    Standardized from {raw_num_nodes} nodes "
        f"(dropped {raw_num_nodes - num_nodes} outside the largest component)"
    )
    print(f"    Avg degree: {degrees.mean():.1f}, max degree: {degrees.max():.0f}")

    return adjacency, features, labels, num_nodes
