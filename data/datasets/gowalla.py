"""
loc-Gowalla Dataset Loader

Source: https://snap.stanford.edu/data/loc-gowalla.html
    - 196,591 nodes, 950,327 undirected edges
    - Undirected (Gowalla location-based friendships)
    - No inherent node features: uses log(1 + degree) as synthetic features
    - No node labels

Used by both fair IBM 2026 and the proactive-rumour-control paper, and the
only large graph in this literature with a geographic embedding: its communities are
spatial, so a blocker that separates regions behaves differently here than on a
social graph whose communities are interest-based.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_gzip, load_edge_list

gowalla_url = "https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz"


def download_gowalla() -> Path:
    return download_gzip("gowalla", gowalla_url, "loc-gowalla_edges.txt.gz")


def load_gowalla(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("loc-Gowalla", path, directed=False)
