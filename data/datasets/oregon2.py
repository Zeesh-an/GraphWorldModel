"""
Oregon-2 Dataset Loader (AS peering, 26 May 2001)

Source: https://snap.stanford.edu/data/Oregon-2.html
    - 11,461 nodes, 32,730 undirected edges
    - Undirected, unweighted, tab-separated with a `#` header
    - No inherent node features: uses log(1 + degree) as synthetic features
    - No node labels

DITTO's Table 2 row, at 11,461 / 32,730 [verified], and one of the two REAL graphs
it simulates on (T = 15, infection 0.1, recovery 0.05, 10% sources). Loading it is
what makes `research/cascade_reconstruction.md` §5.1's `Oregon2-SI` and
`Oregon2-SIR` columns reachable: DITTO reports F1 .8280 / .7928 there against a
supervised ideal of .8320 / .8024.

`oregon2_010526` is the LAST of the nine weekly snapshots SNAP publishes under
this name, and it is the one DITTO's `inc/data.py` fetches by URL; the other eight
are different graphs with the same page. Stated because the collision is the norm
in this literature (§6.4).
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_gzip, load_edge_list

oregon2_url = "https://snap.stanford.edu/data/oregon2_010526.txt.gz"


def download_oregon2() -> Path:
    return download_gzip("oregon2", oregon2_url, "oregon2_010526.txt.gz")


def load_oregon2(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    # DITTO takes the largest connected component before simulating, so the
    # comparable node count is the GCC's rather than the file's
    return load_edge_list("Oregon-2 (AS peering, GCC)", path, lcc=True)
