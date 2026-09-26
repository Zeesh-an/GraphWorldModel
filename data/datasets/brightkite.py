"""
Brightkite Dataset Loader (location-based social network)

Source: https://snap.stanford.edu/data/loc-Brightkite.html
    - 58,228 nodes, 214,078 undirected edges [verified, SNAP]
    - Undirected, unweighted, tab-separated with a `#` header
    - No inherent node features: uses log(1 + degree)

GreedyWalk's Table 2 row at 58,228 / 214,078 with a published **lambda_1 = 101.49**
[verified], and the largest graph in that table
we can run the whole pipeline on. The three above it (Stanford Web, YouTube) are
scalability targets rather than day-one datasets.

Warning: NOT `gowalla`. Both are Brightkite-era location-based social networks and
both are in SNAP's `loc-` family; Gowalla is 196,591 / 950,327 and is loaded for
influence blocking. Different services, different graphs.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_gzip, load_edge_list

brightkite_url = "https://snap.stanford.edu/data/loc-brightkite_edges.txt.gz"


def download_brightkite() -> Path:
    return download_gzip("brightkite", brightkite_url, "loc-brightkite_edges.txt.gz")


def load_brightkite(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("Brightkite (location social network)", path)
