"""
cit-HepPh Dataset Loader

Source: https://snap.stanford.edu/data/cit-HepPh.html
    - 34,546 nodes [verified, matches Tong & Wu's own count], 421,534 arcs
      [derived]: SNAP publishes 421,578; we drop 44 self-citations
    - Directed (arXiv hep-ph citations)
    - No inherent node features: uses log(1 + degree) as synthetic features
    - No node labels

Tong & Wu's `HepPh` (NeurIPS 2018, §5.8), which they quote as 34,546 PAPERS and
run under weighted-cascade 1/deg(v): the same probability model as our
`--prob-model weighted`. They give no edge count, so ours is the file's own.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_gzip, load_edge_list

cit_hepph_url = "https://snap.stanford.edu/data/cit-HepPh.txt.gz"


def download_cit_hepph() -> Path:
    return download_gzip("cit_hepph", cit_hepph_url, "cit-HepPh.txt.gz")


def load_cit_hepph(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("cit-HepPh", path, directed=True)
