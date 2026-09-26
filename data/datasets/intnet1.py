"""
IntNet1 Dataset Loader

Source: https://github.com/renxiaolong/Generalized-Network-Dismantling
    - 6,474 nodes, 12,572 undirected edges
    - Undirected, unweighted, 1-indexed
    - No inherent node features: uses log(1 + degree) as synthetic features
    - No node labels

The autonomous-system peering graph in CoreHD / BPD Table I, also reported by
GND, Wandelt, GDM and MIND. Extremely
hub-dominated, so it is where a degree heuristic is hardest to beat.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_plain, load_edge_list

intnet1_url = (
    "https://raw.githubusercontent.com/renxiaolong/Generalized-Network-Dismantling/"
    "master/Datasets_SI/IntNet1_gcc.txt"
)


def download_intnet1() -> Path:
    return download_plain("intnet1", intnet1_url, "IntNet1_gcc.txt")


def load_intnet1(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("IntNet1 (GCC)", path)
