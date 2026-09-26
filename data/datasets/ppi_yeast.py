"""
Yeast PPI Dataset Loader (dismantling GCC)

Source: https://github.com/renxiaolong/Generalized-Network-Dismantling
    - 2,224 nodes, 6,609 undirected edges
    - Undirected, unweighted, 1-indexed
    - No inherent node features: uses log(1 + degree) as synthetic features
    - No node labels

The most widely shared biological benchmark in this literature: CoreHD, BPD, GND,
FINDER, NIRM ("PPI"), DCRS, SPR and BPHD all report it.

Warning: Three graphs go by "PPI": CoreHD's Table I quotes the RAW 2,361 / 6,646
network, this file is its 2,224-node giant component, and networkrepository's
`bio-yeast` is a different 1,458 / 1,948 graph entirely.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_plain, load_edge_list

ppi_yeast_url = (
    "https://raw.githubusercontent.com/renxiaolong/Generalized-Network-Dismantling/"
    "master/Datasets_SI/PPI_gcc.txt"
)


def download_ppi_yeast() -> Path:
    return download_plain("ppi_yeast", ppi_yeast_url, "PPI_gcc.txt")


def load_ppi_yeast(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("Yeast PPI (GCC)", path)
