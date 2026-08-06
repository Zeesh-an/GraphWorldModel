"""
Oregon-1 Dataset Loader (AS peering, 31 March 2001)

Source: https://snap.stanford.edu/data/Oregon-1.html
    - 10,670 nodes, 22,002 undirected edges [verified, SNAP]
    - Undirected, unweighted, tab-separated with a `#` header
    - No inherent node features: uses log(1 + degree)

GreedyWalk's Table 2 row, at 10,670 / 22,002 with a published **lambda_1 = 58.72**
[verified, `research/epidemic_control.md` §5.2], which makes it one of only four
graphs where our `wm_metrics.spectral_radius` can be checked against a number
somebody else computed. §7 also has it in Gelling, DAVA and GreedyWalk, so it is
the closest thing the spectral line has to a shared benchmark.

Warning: SNAP publishes NINE weekly Oregon-1 snapshots on one page and the papers
say "Oregon" without a date (§6.4). This is `oregon1_010331`, the 31 March 2001
snapshot, which is the one whose counts match GreedyWalk's table. Its sibling
`oregon2_010331` is the Oregon-2 graph from the same week; the `oregon2` key in
this repo is a THIRD snapshot (`oregon2_010526`, 11,461 / 32,730), loaded for
cascade reconstruction because that is the file DITTO fetches.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_gzip, load_edge_list

oregon1_url = "https://snap.stanford.edu/data/oregon1_010331.txt.gz"


def download_oregon1() -> Path:
    return download_gzip("oregon1", oregon1_url, "oregon1_010331.txt.gz")


def load_oregon1(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    # No LCC: SNAP's own count of 10,670 is the raw file, and that is the number
    # GreedyWalk's table and its published lambda_1 are quoted at
    return load_edge_list("Oregon-1 (AS peering, 2001-03-31)", path)
