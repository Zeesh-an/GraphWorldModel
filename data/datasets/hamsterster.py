"""
Hamsterster Dataset Loader (dismantling GCC, "P-H")

Source: https://github.com/renxiaolong/Generalized-Network-Dismantling
    - 2,000 nodes, 16,098 undirected edges
    - Undirected, unweighted, 1-indexed
    - No inherent node features: uses log(1 + degree) as synthetic features
    - No node labels

The single most-shared graph in this literature: GND, Min-Sum, NIRM ("P-H"),
GDM, MIND, BPHD ("Social") and Wandelt all report it.

Warning: Name collision. KONECT's
`petster-hamster` is 2,426 / 16,631 and networkrepository's `soc-hamsterster` is
2,426 / 16,630: both the RAW graph. The dismantling number is the 2,000-node
giant component, which is what this file is.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_plain, load_edge_list

hamsterster_url = (
    "https://raw.githubusercontent.com/renxiaolong/Generalized-Network-Dismantling/"
    "master/Datasets_Main_Paper/Petster-Hamster_GCC.txt"
)


def download_hamsterster() -> Path:
    return download_plain("hamsterster", hamsterster_url, "Petster-Hamster_GCC.txt")


def load_hamsterster(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("Hamsterster (GCC)", path)
