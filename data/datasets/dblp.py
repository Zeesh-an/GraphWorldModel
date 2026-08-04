"""
com-DBLP Dataset Loader

Source: https://snap.stanford.edu/data/com-DBLP.html
    - 317,080 nodes, 1,049,866 undirected edges
    - Undirected (DBLP co-authorship)
    - No inherent node features — uses log(1 + degree) as synthetic features
    - No node labels

SandIMIN's, both Xie papers' and NAMM's `DBLP` (§6.2). Warning: NAMM quotes it as
317K / 2.1M, which is the same graph counted as ARCS; §6.3 records the collision.
Ours reports undirected edges, so 1,049,866 is the number to compare against
SandIMIN's row rather than NAMM's.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_gzip, load_edge_list

dblp_url = "https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz"


def download_dblp() -> Path:
    return download_gzip("dblp", dblp_url, "com-dblp.ungraph.txt.gz")


def load_dblp(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("com-DBLP", path, directed=False)
