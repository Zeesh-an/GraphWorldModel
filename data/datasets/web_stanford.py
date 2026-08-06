"""
web-Stanford Dataset Loader

Source: https://snap.stanford.edu/data/web-Stanford.html
    - 281,903 nodes, 2,312,497 arcs
    - Directed (Stanford.edu web hyperlinks)
    - No inherent node features: uses log(1 + degree) as synthetic features
    - No node labels

Reported by SandIMIN, both Xie papers and NIE (§7): the only large graph shared by
the node-blocking line and the counter-seeding line, which makes it the one place the
two halves of this literature can be put on the same axis (§7 notes they otherwise
share no metric at all).
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_gzip, load_edge_list

web_stanford_url = "https://snap.stanford.edu/data/web-Stanford.txt.gz"


def download_web_stanford() -> Path:
    return download_gzip("web_stanford", web_stanford_url, "web-Stanford.txt.gz")


def load_web_stanford(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("web-Stanford", path, directed=True)
