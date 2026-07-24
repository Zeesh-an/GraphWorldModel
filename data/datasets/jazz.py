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

import os
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import scipy.sparse as sp

jazz_url = "https://nrvis.com/download/data/misc/arenas-jazz.zip"
data_dir = Path(__file__).resolve().parent.parent / "raw" / "jazz"


def download_jazz() -> Path:
    """Download and extract Jazz dataset if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    zip_path = data_dir / "arenas-jazz.zip"
    edges_path = data_dir / "arenas-jazz.edges"

    if edges_path.exists():
        print(f"[✓] Jazz already downloaded at {edges_path}")
        return edges_path

    if not zip_path.exists():
        print(f"[↓] Downloading Jazz from {jazz_url} ...")
        urllib.request.urlretrieve(jazz_url, zip_path)
        print(f"[✓] Saved to {zip_path}")

    print("[↓] Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(data_dir)
    print(f"[✓] Extracted to {data_dir}")

    return edges_path


def load_jazz(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Load Jazz collaboration network. Edges are 1-indexed comma-separated pairs.
    Remaps IDs to contiguous 0..N-1.

    Returns
    -------
    adjacency: scipy.sparse.csr_matrix (N, N) undirected adjacency
    node_feats: np.ndarray (N, 1) float32 -- log(1 + degree) features
    node_labels: np.ndarray (N,) int32 -- placeholder zeros
    num_nodes: int -- number of nodes
    """
    source_list = []
    destination_list = []

    with open(path) as file:
        for line in file:
            line = line.strip()
            # Skip header/comment lines (start with %)
            if not line or line.startswith("%"):
                continue
            parts = line.split(",")
            source, destination = int(parts[0]), int(parts[1])
            source_list.append(source)
            destination_list.append(destination)

    # Collect unique node IDs and remap to contiguous 0..N-1
    all_ids = sorted(set(source_list) | set(destination_list))
    id_to_index = {node_id: index for index, node_id in enumerate(all_ids)}
    num_nodes = len(id_to_index)

    # Build symmetric (undirected) edge arrays
    symmetric_sources = []
    symmetric_destinations = []

    for source, destination in zip(source_list, destination_list):
        source_index = id_to_index[source]
        destination_index = id_to_index[destination]
        symmetric_sources.append(source_index)
        symmetric_destinations.append(destination_index)
        symmetric_sources.append(destination_index)
        symmetric_destinations.append(source_index)

    sources = np.array(symmetric_sources, dtype=np.int32)
    destinations = np.array(symmetric_destinations, dtype=np.int32)
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
    print(f"[✓] Jazz loaded: {num_nodes} nodes, {n_edges_undirected} undirected edges")
    print(f"    Avg degree: {degrees.mean():.1f}, max degree: {degrees.max():.0f}")

    return adjacency, node_feats, node_labels, num_nodes
