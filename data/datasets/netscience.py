"""
NetScience Dataset Loader

Downloads and loads the Network Science coauthorship network.

Source: https://networks.skewed.de/net/netscience
    - 1,589 nodes (authors), 2,742 edges (coauthorships)
    - Undirected, weighted (collaboration strength)
    - Weights are discarded — we use binary adjacency
    - No inherent node features — uses log(1 + degree) as synthetic features
    - No node labels

Original paper: M. E. J. Newman, Phys. Rev. E 74, 036104 (2006)
"""

import csv
import os
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import scipy.sparse as sp

netscience_url = "https://networks.skewed.de/net/netscience/files/netscience.csv.zip"
data_dir = Path(__file__).resolve().parent.parent / "netscience"


def download_netscience() -> Path:
    """Download and extract NetScience dataset if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    zip_path = data_dir / "netscience.csv.zip"
    edges_path = data_dir / "edges.csv"

    if edges_path.exists():
        print(f"[✓] NetScience already downloaded at {data_dir}")
        return data_dir

    if not zip_path.exists():
        print(f"[↓] Downloading NetScience from {netscience_url} ...")
        urllib.request.urlretrieve(netscience_url, zip_path)
        print(f"[✓] Saved to {zip_path}")

    print("[↓] Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(data_dir)
    print(f"[✓] Extracted to {data_dir}")

    return data_dir


def load_netscience(
    data_path: Path,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Load NetScience coauthorship network. Edges are 0-indexed CSV with columns
    (source, target, value). The value (edge weight) is discarded.

    Returns
    -------
    adjacency: scipy.sparse.csr_matrix (N, N) undirected adjacency
    node_feats: np.ndarray (N, 1) float32 -- log(1 + degree) features
    node_labels: np.ndarray (N,) int32 -- placeholder zeros
    num_nodes: int -- number of nodes
    """
    # Read nodes.csv to determine N (nodes may be isolates not in edges.csv)
    node_ids = set()
    nodes_path = data_path / "nodes.csv"
    if nodes_path.exists():
        with open(nodes_path) as file:
            reader = csv.DictReader(file)
            # Header lines look like "# index, label, _pos" — strip the comment marker and padding
            reader.fieldnames = [name.strip(" #") for name in reader.fieldnames]
            for row in reader:
                node_ids.add(int(row["index"]))

    # Read edges
    source_list = []
    destination_list = []

    with open(data_path / "edges.csv") as file:
        reader = csv.DictReader(file)
        reader.fieldnames = [name.strip(" #") for name in reader.fieldnames]
        for row in reader:
            source, destination = int(row["source"]), int(row["target"])
            source_list.append(source)
            destination_list.append(destination)
            node_ids.add(source)
            node_ids.add(destination)

    # Remap to contiguous 0..N-1 (should already be 0-indexed but may have gaps from isolates)
    sorted_ids = sorted(node_ids)
    id_to_index = {node_id: index for index, node_id in enumerate(sorted_ids)}
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
    n_isolates = int((degrees == 0).sum())
    print(
        f"[✓] NetScience loaded: {num_nodes} nodes, "
        f"{n_edges_undirected} undirected edges"
    )
    print(f"    Avg degree: {degrees.mean():.1f}, max degree: {degrees.max():.0f}")
    if n_isolates > 0:
        print(f"    Isolate nodes (degree 0): {n_isolates}")

    return adjacency, node_feats, node_labels, num_nodes
