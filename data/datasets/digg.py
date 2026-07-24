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
import os
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import scipy.sparse as sp

digg_url = "https://datasets.syr.edu/uploads/1296588940/Digg-dataset.zip"
data_dir = Path(__file__).resolve().parent.parent / "raw" / "digg"


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
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(data_dir)

    print(f"[✓] Extracted to {data_dir / 'Digg-dataset'}")

    return data_dir / "Digg-dataset" / "data"


def load_digg(data_path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Load Digg social network. Keeps only edges between core nodes
    (those listed in nodes.csv). Remaps IDs to contiguous 0..N-1.

    Returns
    -------
    adjacency: scipy.sparse.csr_matrix (N, N) undirected adjacency
    node_feats: np.ndarray (N, 1) float32 -- log(1 + degree) features
    node_labels: np.ndarray (N,) int32 -- placeholder zeros
    num_nodes: int -- number of core nodes
    """
    # Load core node IDs
    core_ids = []
    with open(data_path / "nodes.csv") as file:
        for line in file:
            line = line.strip()
            if line:
                core_ids.append(int(line))

    core_set = set(core_ids)

    # Remap non-contiguous IDs to contiguous IDs from 0, ..., N - 1
    id_to_index = {node_id: index for index, node_id in enumerate(sorted(core_ids))}
    num_nodes = len(id_to_index)

    # Load edges, keep only those between core nodes
    source_list = []
    destination_list = []
    n_dropped = 0

    with open(data_path / "edges.csv") as file:
        reader = csv.reader(file)

        for row in reader:
            # Edge (a - b)
            source, destination = int(row[0]), int(row[1])

            if source in core_set and destination in core_set:
                source_index = id_to_index[source]
                destination_index = id_to_index[destination]

                # Store both directions for undirected graph
                source_list.append(source_index)
                destination_list.append(destination_index)
                source_list.append(destination_index)
                destination_list.append(source_index)
            else:
                n_dropped += 1

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

    # Degree-based node features (no natural features in Digg)
    degrees = np.array(adjacency.sum(axis=1)).flatten()
    node_feats = np.log1p(degrees).reshape(-1, 1).astype(np.float32)  # shape: (N, 1)

    # Placeholder labels (Digg has no node labels)
    node_labels = np.zeros(num_nodes, dtype=np.int32)

    n_edges_undirected = adjacency.nnz // 2
    print(f"[✓] Digg loaded: {num_nodes} nodes, {n_edges_undirected} undirected edges")
    print(f"    Dropped {n_dropped} edges to external nodes")
    print(f"    Avg degree: {degrees.mean():.1f}, max degree: {degrees.max():.0f}")

    return adjacency, node_feats, node_labels, num_nodes
