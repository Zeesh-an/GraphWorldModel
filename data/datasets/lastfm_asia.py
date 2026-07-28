"""
LastFM Asia Dataset Loader

Downloads and loads the SNAP LastFM Asia mutual-follower network.

Source: https://snap.stanford.edu/data/feather-lastfm-social.html
    - 7,624 nodes (users), 27,806 edges (mutual follows)
    - Undirected, unweighted, ids already contiguous 0..7623
    - Node labels: 18 country classes (the `target` column)
    - No inherent node features — uses log(1 + degree) as synthetic features.
      The archive also ships `lastfm_asia_features.json` (a per-user list of
      liked-artist ids), which we do NOT load: densifying it would be a
      7,624 x ~7,800 float32 matrix for no IM-relevant gain. It stays in
      data/raw/lastfm_asia/ for anyone who wants it.

MOEIM reports this graph in its setting-1 table at 7,624 / 27,806 — matching
the raw file, so no preprocessing decision is needed.

Original paper: Rozemberczki & Sarkar, "Characteristic Functions on Graphs:
    Birds of a Feather, from Statistical Descriptors to Parametric Models,"
    CIKM 2020
"""

import csv
import os
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.graph_utils import degree_features, edges_to_adjacency

lastfm_asia_url = "https://snap.stanford.edu/data/lastfm_asia.zip"
data_dir = Path(__file__).resolve().parent.parent / "raw" / "lastfm_asia"
# The directory inside the archive is misspelled by the publisher
archive_dir = "lasftm_asia"


def download_lastfm_asia() -> Path:
    """Download and extract the LastFM Asia edges and country targets."""
    os.makedirs(data_dir, exist_ok=True)
    zip_path = data_dir / "lastfm_asia.zip"
    edges_path = data_dir / archive_dir / "lastfm_asia_edges.csv"

    if edges_path.exists():
        print(f"[✓] LastFM Asia already downloaded at {edges_path}")
        return edges_path

    if not zip_path.exists():
        print(f"[↓] Downloading LastFM Asia from {lastfm_asia_url} ...")
        urllib.request.urlretrieve(lastfm_asia_url, zip_path)
        print(f"[✓] Saved to {zip_path}")

    print("[↓] Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        # Skip the 17 MB features JSON we never read
        for member in ("lastfm_asia_edges.csv", "lastfm_asia_target.csv"):
            zip_file.extract(f"{archive_dir}/{member}", data_dir)
    print(f"[✓] Extracted to {data_dir / archive_dir}")

    return edges_path


def _read_csv_pairs(path: Path, first: str, second: str) -> np.ndarray:
    with open(path) as file:
        reader = csv.DictReader(file)

        return np.array(
            [(int(row[first]), int(row[second])) for row in reader], dtype=np.int64
        )


def load_lastfm_asia(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Load the LastFM Asia network. Both CSVs carry a header row; edges are
    `node_1,node_2` and targets are `id,target`.

    Returns
    -------
    adjacency: scipy.sparse.csr_matrix (N, N) undirected adjacency
    node_feats: np.ndarray (N, 1) float32 -- log(1 + degree) features
    node_labels: np.ndarray (N,) int32 -- country ids (18 classes)
    num_nodes: int -- number of nodes
    """
    raw_edges = _read_csv_pairs(path, "node_1", "node_2")  # shape: (E, 2)
    raw_targets = _read_csv_pairs(
        path.parent / "lastfm_asia_target.csv", "id", "target"
    )  # shape: (N, 2) as (node, country)

    # The target file covers every user, so it is the authority on N
    num_nodes = int(raw_targets.shape[0])

    adjacency = edges_to_adjacency(
        raw_edges[:, 0], raw_edges[:, 1], num_nodes, directed=False
    )

    node_labels = np.zeros(num_nodes, dtype=np.int32)
    node_labels[raw_targets[:, 0]] = raw_targets[:, 1]

    node_feats = degree_features(adjacency, directed=False)  # shape: (N, 1)

    degrees = np.array(adjacency.sum(axis=1)).flatten()
    n_edges_undirected = adjacency.nnz // 2
    print(
        f"[✓] LastFM Asia loaded: {num_nodes} nodes, "
        f"{n_edges_undirected} undirected edges, "
        f"{len(np.unique(node_labels))} countries"
    )
    print(f"    Avg degree: {degrees.mean():.1f}, max degree: {degrees.max():.0f}")

    return adjacency, node_feats, node_labels, num_nodes
