"""
Epinions Dataset Loader

Downloads and loads the SNAP signed Epinions who-trusts-whom network: Han et
al.'s adaptive-IM benchmark (research/adaptive_online_im.md §5.2, §6.2).

Source: https://snap.stanford.edu/data/soc-sign-epinions.html
    - 131,828 nodes (members), 841,372 arcs (trust/distrust statements)
    - Directed, and SIGNED: column 3 is +1 (trust) or -1 (distrust)

Reconciling the edge count [derived, verified against the downloaded file]:

    841,372   raw lines in soc-sign-epinions.txt  <- SNAP's and Han et al.'s count
      - 573   self-loops (a member trusting themselves)
    = 840,799   arcs we build

    There are no duplicate ordered pairs, so nothing else is lost. The node count
    matches SNAP exactly at 131,828. Self-loops are dropped by edges_to_adjacency
    for the same reason as everywhere else: p(u->v) = 1/in-degree(v) is not well
    defined when a node is its own in-neighbour.

    - We DROP the sign and keep the arc. Han et al. report this graph as
      `132K / 841K` with average degree 13.4, i.e. the full signed arc count
      treated as a plain directed graph: dropping only the distrust arcs would
      give a different graph from the one their table describes.
    - No inherent node features: uses log(1 + total degree) as synthetic features
    - No node labels

Original paper: Leskovec, Huttenlocher & Kleinberg, "Signed Networks in Social
    Media," CHI 2010
"""

import gzip
import os
import urllib.request
from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.graph_utils import degree_features, edges_to_adjacency, remap_to_contiguous

epinions_url = "https://snap.stanford.edu/data/soc-sign-epinions.txt.gz"
data_dir = Path(__file__).resolve().parent.parent / "raw" / "epinions"


def download_epinions() -> Path:
    """Download and extract the signed trust edge list."""
    os.makedirs(data_dir, exist_ok=True)
    gz_path = data_dir / "soc-sign-epinions.txt.gz"
    txt_path = data_dir / "soc-sign-epinions.txt"

    if txt_path.exists():
        print(f"[ok] Epinions already downloaded at {txt_path}")
        return txt_path

    if not gz_path.exists():
        print(f"[get] Downloading Epinions from {epinions_url} ...")
        urllib.request.urlretrieve(epinions_url, gz_path)
        print(f"[ok] Saved to {gz_path}")

    print("[get] Extracting ...")
    with gzip.open(gz_path, "rb") as gz_file:
        txt_path.write_bytes(gz_file.read())
    print(f"[ok] Extracted to {txt_path}")

    return txt_path


def load_epinions(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Load Epinions as a directed graph, sign discarded.

    Returns
    -------
    adjacency: scipy.sparse.csr_matrix (N, N) directed adjacency
    node_feats: np.ndarray (N, 1) float32 -- log(1 + total degree) features
    node_labels: np.ndarray (N,) int32 -- placeholder zeros
    num_nodes: int -- number of nodes
    """
    # Three columns (from, to, sign); the sign is dropped, see the module note
    raw_edges = np.loadtxt(path, comments="#", dtype=np.int64)[:, :2]  # shape: (E, 2)

    # SNAP ids are sparse (deleted accounts leave holes), so the raw max id is
    # larger than the node count every published table quotes
    remapped, num_nodes = remap_to_contiguous(raw_edges)
    adjacency = edges_to_adjacency(
        remapped[:, 0], remapped[:, 1], num_nodes, directed=True
    )

    node_feats = degree_features(adjacency, directed=True)  # shape: (N, 1)
    node_labels = np.zeros(num_nodes, dtype=np.int32)

    print(f"[ok] Epinions loaded: {num_nodes} nodes, {adjacency.nnz} directed arcs")

    return adjacency, node_feats, node_labels, num_nodes
