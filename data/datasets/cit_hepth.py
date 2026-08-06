"""
cit-HepTh Dataset Loader

Source: https://snap.stanford.edu/data/cit-HepTh.html
    - 27,769 nodes, 352,768 arcs [derived]: SNAP publishes 27,770 / 352,807; we
      drop 39 self-citations and the one node left isolated by that
    - Directed (arXiv hep-th citations)
    - No inherent node features: uses log(1 + degree) as synthetic features
    - No node labels

DiffIM's citation graph (§6.2). A citation network is nearly acyclic, so the
rumour's reachable set is shallow and wide rather than deep, which is where an edge
lever should do best and a counter-seeding one worst.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_gzip, load_edge_list

cit_hepth_url = "https://snap.stanford.edu/data/cit-HepTh.txt.gz"


def download_cit_hepth() -> Path:
    return download_gzip("cit_hepth", cit_hepth_url, "cit-HepTh.txt.gz")


def load_cit_hepth(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("cit-HepTh", path, directed=True)
