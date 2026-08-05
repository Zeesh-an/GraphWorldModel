"""
Oregon-2 Dataset Loader (AS peering, 31 March 2001)

Source: https://snap.stanford.edu/data/Oregon-2.html
    - 10,900 nodes, 31,180 undirected edges [verified, SNAP]
    - Undirected, unweighted, tab-separated with a `#` header
    - No inherent node features — uses log(1 + degree)

GreedyWalk's Table 2 row, at 10,900 / 31,180 with a published **lambda_1 = 70.74**
[verified, `research/epidemic_control.md` §5.2]. Loaded alongside `oregon1` because
the two are the spectral line's standard pair and the lambda_1 gap between them
(58.72 vs 70.74) is a controlled test of whether our eigensolver reproduces
published numbers on two graphs of nearly the same size.

Warning: THREE Oregon-2 snapshots are in play here and the key names disambiguate
them. `oregon2_010331` is this one, the 31 March 2001 week GreedyWalk, Gelling and
DAVA report. `oregon2` in this repo is `oregon2_010526` at 11,461 / 32,730, the LAST
of SNAP's nine weekly snapshots and the file DITTO's own loader fetches — a
different graph under the same paper-facing name, which is §6.4's collision hazard
in its purest form.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_gzip, load_edge_list

oregon2_010331_url = "https://snap.stanford.edu/data/oregon2_010331.txt.gz"


def download_oregon2_010331() -> Path:
    return download_gzip("oregon2_010331", oregon2_010331_url, "oregon2_010331.txt.gz")


def load_oregon2_010331(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("Oregon-2 (AS peering, 2001-03-31)", path)
