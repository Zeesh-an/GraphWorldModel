"""
p2p-Gnutella08 Dataset Loader

Source: https://snap.stanford.edu/data/p2p-Gnutella08.html
    - 6,301 nodes, 20,777 arcs
    - Directed (Gnutella peer-to-peer hosts, August 2002)
    - No inherent node features: uses log(1 + degree) as synthetic features
    - No node labels

NIE's smallest real network (its Table II: 6,301 / 20,777, avg degree 3.30) and
NAMM's `Gnutella`. There is a NAME COLLISION
worth stating in the loader: our `p2p_gnutella` is SNAP's Gnutella31 giant component
at 62,561 nodes, the proactive-rumour-control paper's "Gnutella" is a third snapshot
at 8,800 / 63,000, and this is the 08 file the NIE and NAMM rows are computed on.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_gzip, load_edge_list

p2p_gnutella08_url = "https://snap.stanford.edu/data/p2p-Gnutella08.txt.gz"


def download_p2p_gnutella08() -> Path:
    return download_gzip("p2p_gnutella08", p2p_gnutella08_url, "p2p-Gnutella08.txt.gz")


def load_p2p_gnutella08(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("p2p-Gnutella08", path, directed=True)
