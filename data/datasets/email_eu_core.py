"""
email-Eu-core Dataset Loader

Downloads and loads the SNAP email-Eu-core communication network.

Source: https://snap.stanford.edu/data/email-Eu-core.html
    - 1,005 nodes (researchers), 25,571 directed edges (emails)
    - 642 of those lines are self-loops and are dropped, leaving 24,929 arcs
    - Directed: edge (a, b) means a sent b at least one email
    - Node labels: 42 ground-truth departments — the ONLY real community labels
      in our suite, which makes this the natural graph for community-aware
      evaluation (cf. MOEIM's community objective, and our SBM experiments,
      which have no published baseline)
    - No inherent node features — uses log(1 + total degree) as synthetic features

MOEIM reports the largest connected component, 986 / 25,552.

Original paper: Yin et al., "Local Higher-order Graph Clustering," KDD 2017
"""

import gzip
import os
import urllib.request
from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.graph_utils import degree_features, edges_to_adjacency

email_eu_core_url = "https://snap.stanford.edu/data/email-Eu-core.txt.gz"
email_eu_core_labels_url = (
    "https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz"
)
data_dir = Path(__file__).resolve().parent.parent / "raw" / "email_eu_core"


def _fetch(url: str, gz_path: Path, txt_path: Path) -> None:
    if not gz_path.exists():
        print(f"[↓] Downloading {gz_path.name} from {url} ...")
        urllib.request.urlretrieve(url, gz_path)

    with gzip.open(gz_path, "rb") as gz_file:
        txt_path.write_bytes(gz_file.read())


def download_email_eu_core() -> Path:
    """Download and extract the edge list and the department labels."""
    os.makedirs(data_dir, exist_ok=True)
    txt_path = data_dir / "email-Eu-core.txt"
    labels_path = data_dir / "email-Eu-core-department-labels.txt"

    if txt_path.exists() and labels_path.exists():
        print(f"[✓] email-Eu-core already downloaded at {txt_path}")
        return txt_path

    _fetch(email_eu_core_url, data_dir / "email-Eu-core.txt.gz", txt_path)
    _fetch(
        email_eu_core_labels_url,
        data_dir / "email-Eu-core-department-labels.txt.gz",
        labels_path,
    )
    print(f"[✓] Extracted to {data_dir}")

    return txt_path


def load_email_eu_core(
    path: Path,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Load email-Eu-core. Space-separated directed pairs, ids already contiguous
    0..1004. Labels come from the sibling department-labels file.

    Returns
    -------
    adjacency: scipy.sparse.csr_matrix (N, N) directed adjacency
    node_feats: np.ndarray (N, 1) float32 -- log(1 + total degree) features
    node_labels: np.ndarray (N,) int32 -- department ids (42 classes)
    num_nodes: int -- number of nodes
    """
    raw_edges = np.loadtxt(path, comments="#", dtype=np.int64)  # shape: (E, 2)
    raw_labels = np.loadtxt(
        path.parent / "email-Eu-core-department-labels.txt", dtype=np.int64
    )  # shape: (N, 2) as (node, department)

    # The label file covers every node, so it is the authority on N — the edge
    # list alone would miss any node that neither sent nor received mail
    num_nodes = int(raw_labels.shape[0])

    adjacency = edges_to_adjacency(
        raw_edges[:, 0], raw_edges[:, 1], num_nodes, directed=True
    )

    node_labels = np.zeros(num_nodes, dtype=np.int32)
    node_labels[raw_labels[:, 0]] = raw_labels[:, 1]

    node_feats = degree_features(adjacency, directed=True)  # shape: (N, 1)

    in_degrees = np.array(adjacency.sum(axis=0)).flatten()
    out_degrees = np.array(adjacency.sum(axis=1)).flatten()
    print(
        f"[✓] email-Eu-core loaded: {num_nodes} nodes, {adjacency.nnz} directed edges, "
        f"{len(np.unique(node_labels))} departments"
    )
    print(f"    In-degree  — avg: {in_degrees.mean():.1f}, max: {in_degrees.max():.0f}")
    print(
        f"    Out-degree — avg: {out_degrees.mean():.1f}, max: {out_degrees.max():.0f}"
    )

    return adjacency, node_feats, node_labels, num_nodes
