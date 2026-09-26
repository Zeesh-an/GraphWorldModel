"""
Deezer Dataset Loader

Downloads and loads the SNAP gemsec-Deezer friendship network.

Source: https://snap.stanford.edu/data/gemsec-Deezer.html
    - 47,538 nodes (users), 222,887 undirected edges (mutual friendships)
    - Undirected, unweighted, ids already contiguous 0..47537
    - No inherent node features: uses log(1 + degree) as synthetic features
    - No node labels (the archive ships genre lists we do not densify)

Warning: **"Deezer" in this literature is the HUNGARY subgraph, not the whole release.**
IVGD reports its Deezer as 47,538 / 222,887 without saying which file that
is, and the archive ships three separate country
graphs with overlapping 0-based ids. Counted from the downloaded files [derived]:

    HR   54,573 nodes   498,202 edges
    HU   47,538 nodes   222,887 edges   <- IVGD's row, to the digit
    RO   41,773 nodes   125,826 edges
    union (offset)   143,884 nodes   846,915 edges

So `--dataset deezer` is **HU**, and the union is a different graph that matches
nothing published. The other two files stay in `data/raw/deezer/` for anyone who
wants them; point `country` at one to load it instead.

IVGD uses this graph for its SCALABILITY column and nothing else, so its only
published number here is a runtime, and even that could not have its column
header confirmed (marked [claim]). It is loaded for the same purpose: a size
at which the cost claim has room to separate the arms, not a head-to-head
F1 comparison there is no row to make.

Original paper: Rozemberczki, Davies, Sarkar & Sutton, "GEMSEC: Graph Embedding
    with Self Clustering," ASONAM 2019
"""

import os
import tarfile
import urllib.request
from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.graph_utils import degree_features, edges_to_adjacency, read_pairs

deezer_url = "https://snap.stanford.edu/data/gemsec_deezer_dataset.tar.gz"
data_dir = Path(__file__).resolve().parent.parent / "raw" / "deezer"
# Which country file `--dataset deezer` means. HU is the one IVGD reports.
country = "HU"


def download_deezer() -> Path:
    """Download and extract the gemsec-Deezer archive, returning the HU edge list."""
    os.makedirs(data_dir, exist_ok=True)
    archive_path = data_dir / "gemsec_deezer_dataset.tar.gz"
    edges_path = data_dir / "deezer_clean_data" / f"{country}_edges.csv"

    if edges_path.exists():
        print(f"[ok] Deezer already downloaded at {edges_path}")
        return edges_path

    if not archive_path.exists():
        print(f"[get] Downloading Deezer from {deezer_url} ...")
        urllib.request.urlretrieve(deezer_url, archive_path)
        print(f"[ok] Saved to {archive_path}")

    print("[get] Extracting ...")
    # tarfile rather than zipfile: SNAP serves this one as a gzipped tar, unlike
    # the neighbouring datasets
    with tarfile.open(archive_path, "r:gz") as tar_file:
        tar_file.extractall(data_dir, filter="data")
    print(f"[ok] Extracted to {data_dir}")

    return edges_path


def load_deezer(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Load one country's Deezer friendship graph. The CSV has a `node_1,node_2`
    header and 0-based contiguous ids.

    Returns
    -------
    adjacency: scipy.sparse.csr_matrix (N, N) undirected adjacency
    node_feats: np.ndarray (N, 1) float32 -- log(1 + degree) features
    node_labels: np.ndarray (N,) int32 -- placeholder zeros
    num_nodes: int -- number of nodes
    """
    raw_edges = read_pairs(path, skip_rows=1)  # shape: (E, 2)
    num_nodes = int(raw_edges.max()) + 1

    adjacency = edges_to_adjacency(
        raw_edges[:, 0], raw_edges[:, 1], num_nodes, directed=False
    )
    node_feats = degree_features(adjacency, directed=False)  # shape: (N, 1)
    node_labels = np.zeros(num_nodes, dtype=np.int32)

    degrees = np.array(adjacency.sum(axis=1)).flatten()
    print(
        f"[ok] Deezer loaded: {num_nodes} nodes, {adjacency.nnz // 2} undirected "
        f"edges ({path.stem})"
    )
    print(f"    Avg degree: {degrees.mean():.1f}, max degree: {degrees.max():.0f}")

    return adjacency, node_feats, node_labels, num_nodes
