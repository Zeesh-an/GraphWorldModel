"""
Twitter Dataset Loader

Downloads and loads the SNAP ego-Twitter social network.

Source: https://snap.stanford.edu/data/ego-Twitter.html
    - 81,306 nodes, 1,768,149 directed edges (follow graph)
    - Union of 973 ego networks
    - Originally directed: edge (a, b) means a follows b
    - Symmetrized to undirected during loading
    - No inherent node features — uses log(1 + degree) as synthetic features
    - No node labels
"""

import gzip
import os
import urllib.request
from pathlib import Path
import numpy as np
import scipy.sparse as sp

twitter_url = "https://snap.stanford.edu/data/twitter_combined.txt.gz"
data_dir = Path(__file__).resolve().parent.parent / "twitter"


def download_twitter() -> Path:
    """Download and extract Twitter edge list if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    gz_path = data_dir / "twitter_combined.txt.gz"
    txt_path = data_dir / "twitter_combined.txt"

    if txt_path.exists():
        print(f"[✓] Twitter already downloaded at {txt_path}")
        return txt_path

    if not gz_path.exists():
        print(f"[↓] Downloading Twitter from {twitter_url} ...")
        urllib.request.urlretrieve(twitter_url, gz_path)
        print(f"[✓] Saved to {gz_path}")

    print("[↓] Extracting ...")
    with gzip.open(gz_path, "rb") as gz_file:
        with open(txt_path, "wb") as txt_file:
            txt_file.write(gz_file.read())
    print(f"[✓] Extracted to {txt_path}")

    return txt_path


def load_twitter(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Load Twitter ego network. Symmetrizes directed follow edges
    into an undirected graph. Remaps IDs to contiguous 0..N-1.

    Returns
    -------
    adjacency: scipy.sparse.csr_matrix (N, N) undirected adjacency
    node_feats: np.ndarray (N, 1) float32 -- log(1 + degree) features
    node_labels: np.ndarray (N,) int32 -- placeholder zeros
    num_nodes: int -- number of nodes
    """
    # Parse space-separated directed edge list
    raw_sources = []
    raw_destinations = []

    with open(path) as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            raw_sources.append(int(parts[0]))
            raw_destinations.append(int(parts[1]))

    # Collect unique node IDs and remap to contiguous 0..N-1
    all_ids = sorted(set(raw_sources) | set(raw_destinations))
    id_to_index = {node_id: index for index, node_id in enumerate(all_ids)}
    num_nodes = len(id_to_index)

    # Build symmetric (undirected) edge arrays
    source_list = []
    destination_list = []

    for source, destination in zip(raw_sources, raw_destinations):
        source_index = id_to_index[source]
        destination_index = id_to_index[destination]

        # Store both directions for undirected graph
        source_list.append(source_index)
        destination_list.append(destination_index)
        source_list.append(destination_index)
        destination_list.append(source_index)

    sources = np.array(source_list, dtype=np.int32)
    destinations = np.array(destination_list, dtype=np.int32)
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
        f"[✓] Twitter loaded: {num_nodes} nodes, "
        f"{n_edges_undirected} undirected edges"
    )
    print(f"    Avg degree: {degrees.mean():.1f}, max degree: {degrees.max():.0f}")

    return adjacency, node_feats, node_labels, num_nodes
