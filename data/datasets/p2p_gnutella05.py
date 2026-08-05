"""
Gnutella P2P Dataset Loader (5 August 2002 snapshot)

Source: https://snap.stanford.edu/data/p2p-Gnutella05.html
    - 8,846 nodes, 31,839 arcs [verified, GreedyWalk Table 2 and SNAP]
    - Directed, tab-separated with a `#` header
    - No inherent node features — uses log(1 + total degree)

GreedyWalk's Table 2 row exactly (8,846 / 31,839), loaded as the pair with
`p2p_gnutella06` because that table reports both and the two are consecutive
daily snapshots of one network — the closest thing the spectral line has to a
same-graph replicate.

Warning: FOUR Gnutella snapshots now live in this repo under names that differ only
by a suffix (§6.4 of both `research/epidemic_control.md` and
`research/influence_blocking.md`). `p2p_gnutella05` and `06` are GreedyWalk's;
`p2p_gnutella08` and `24` are the influence-blocking benchmarks; `p2p_gnutella` is
Gnutella31's giant component at 62,561. All five are different graphs.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_gzip, load_edge_list

p2p_gnutella05_url = "https://snap.stanford.edu/data/p2p-Gnutella05.txt.gz"


def download_p2p_gnutella05() -> Path:
    return download_gzip("p2p_gnutella05", p2p_gnutella05_url, "p2p-Gnutella05.txt.gz")


def load_p2p_gnutella05(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("Gnutella P2P (2002-08-05)", path, directed=True)
