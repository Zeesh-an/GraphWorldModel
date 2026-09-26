"""
p2p-Gnutella24 Dataset Loader

Source: https://snap.stanford.edu/data/p2p-Gnutella24.html
    - 26,518 nodes, 65,369 arcs
    - Directed (Gnutella peer-to-peer hosts, August 2002)
    - No inherent node features: uses log(1 + degree) as synthetic features
    - No node labels

NIE's mid-size network (Table II: 26,518 / 65,369, avg degree 2.47). Sparse
and near-tree-like, which is the regime where blocking a single cut vertex removes a
whole branch: the opposite of the hub-dominated social graphs.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_gzip, load_edge_list

p2p_gnutella24_url = "https://snap.stanford.edu/data/p2p-Gnutella24.txt.gz"


def download_p2p_gnutella24() -> Path:
    return download_gzip("p2p_gnutella24", p2p_gnutella24_url, "p2p-Gnutella24.txt.gz")


def load_p2p_gnutella24(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("p2p-Gnutella24", path, directed=True)
