"""
NetHEPT Dataset Loader

Downloads and loads the NetHEPT high-energy-physics-theory COLLABORATION network.

Source: https://github.com/SparklyYS/Simultaneous-IMM (mirror of Wei Chen's data)
    - 15,229 nodes (authors), 62,752 arcs = 31,376 undirected co-authorships
    - Undirected in nature; the mirror stores both arcs, so we load it as a
      directed graph with a symmetric adjacency (identical IC/LT behaviour,
      since p(u→v) = 1/in-degree(v) and in-degree == out-degree == degree here)
    - Standard benchmark for Influence Maximization
    - No inherent node features — uses log(1 + total degree) as synthetic features
    - No node labels

Wei Chen's original `hep.txt` (see `netphy.py`, same archive) declares
15,233 nodes / 58,891 edge LINES; deduplicating its multi-edges and dropping
39 self-loops gives 31,359 undirected edges. This mirror has 31,376 — the same
graph to within 17 edges (0.05%). The "31.4K undirected" in the SSA/D-SSA
benchmark table is this count; the "58,891" in IRIE's table is the raw line
count. See research/influence_maximization.md §6.4.1.

Original paper: Wei Chen et al., "Efficient Influence Maximization
    in Social Networks," KDD 2009
"""

import os
import urllib.request
from pathlib import Path
import numpy as np
import scipy.sparse as sp

nethept_graph_url = (
    "https://raw.githubusercontent.com/SparklyYS/Simultaneous-IMM/"
    "master/nethept/graph.txt"
)
nethept_attribute_url = (
    "https://raw.githubusercontent.com/SparklyYS/Simultaneous-IMM/"
    "master/nethept/attribute.txt"
)
data_dir = Path(__file__).resolve().parent.parent / "raw" / "nethept"


def download_nethept() -> Path:
    """Download NetHEPT edge list if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    graph_path = data_dir / "graph.txt"
    attribute_path = data_dir / "attribute.txt"

    if graph_path.exists():
        print(f"[✓] NetHEPT already downloaded at {graph_path}")
        return graph_path

    print("[↓] Downloading NetHEPT edge list ...")
    urllib.request.urlretrieve(nethept_graph_url, graph_path)
    print(f"[✓] Saved edge list to {graph_path}")

    print("[↓] Downloading NetHEPT attributes ...")
    urllib.request.urlretrieve(nethept_attribute_url, attribute_path)
    print(f"[✓] Saved attributes to {attribute_path}")

    return graph_path


def load_nethept(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Load the NetHEPT collaboration network. Edges are 0-indexed space-separated
    pairs, both directions present. Kept as a directed graph (like Cora-ML).

    Returns
    -------
    adjacency: scipy.sparse.csr_matrix (N, N) directed adjacency
    node_feats: np.ndarray (N, 1) float32 -- log(1 + degree) features
    node_labels: np.ndarray (N,) int32 -- placeholder zeros
    num_nodes: int -- number of nodes
    """
    # Read attribute.txt to get expected node/edge counts
    attribute_path = path.parent / "attribute.txt"
    expected_num_nodes = None
    if attribute_path.exists():
        with open(attribute_path) as file:
            for line in file:
                line = line.strip()
                if line.startswith("n="):
                    expected_num_nodes = int(line.split("=")[1])

    # Parse space-separated directed edge list (0-indexed)
    source_list = []
    destination_list = []

    with open(path) as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            source, destination = int(parts[0]), int(parts[1])
            source_list.append(source)
            destination_list.append(destination)

    # Determine N from attribute file or from max node ID
    all_ids = set(source_list) | set(destination_list)
    max_id = max(all_ids)

    if expected_num_nodes is not None:
        num_nodes = expected_num_nodes
    else:
        num_nodes = max_id + 1

    # IDs are already 0-indexed contiguous (verified from data source)
    sources = np.array(source_list, dtype=np.int32)
    destinations = np.array(destination_list, dtype=np.int32)
    values = np.ones(len(sources), dtype=np.float32)

    # Build directed sparse adjacency
    adjacency = sp.csr_matrix(
        (values, (sources, destinations)), shape=(num_nodes, num_nodes)
    )
    adjacency = (adjacency > 0).astype(np.float32)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    # Degree-based node features using total degree (in + out)
    in_degrees = np.array(adjacency.sum(axis=0)).flatten()
    out_degrees = np.array(adjacency.sum(axis=1)).flatten()
    total_degrees = in_degrees + out_degrees
    node_feats = (
        np.log1p(total_degrees).reshape(-1, 1).astype(np.float32)
    )  # shape: (N, 1)

    # Placeholder labels
    node_labels = np.zeros(num_nodes, dtype=np.int32)

    print(f"[✓] NetHEPT loaded: {num_nodes} nodes, {adjacency.nnz} directed edges")
    print(f"    In-degree  — avg: {in_degrees.mean():.1f}, max: {in_degrees.max():.0f}")
    print(
        f"    Out-degree — avg: {out_degrees.mean():.1f}, max: {out_degrees.max():.0f}"
    )

    return adjacency, node_feats, node_labels, num_nodes
