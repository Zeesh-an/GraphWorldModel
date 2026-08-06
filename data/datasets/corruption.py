"""
Corruption Dataset Loader

Source: https://github.com/renxiaolong/Generalized-Network-Dismantling
    - 309 nodes, 3,281 undirected edges
    - Undirected, unweighted, 1-indexed
    - No inherent node features: uses log(1 + degree) as synthetic features
    - No node labels

Brazilian corruption scandals 1987-2014: nodes are people, an edge means two
people appeared in the same scandal. The giant component, as GND and NIRM report
it (research/critical_node_detection.md §6.2).

Original paper: Ribeiro, Alves, Martins, Lenzi, Perc, "The dynamical structure of
    political corruption networks", Journal of Complex Networks 6(6), 2018
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_plain, load_edge_list

corruption_url = (
    "https://raw.githubusercontent.com/renxiaolong/Generalized-Network-Dismantling/"
    "master/Datasets_Main_Paper/Corruption_Gcc.txt"
)


def download_corruption() -> Path:
    return download_plain("corruption", corruption_url, "Corruption_Gcc.txt")


def load_corruption(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("Corruption (GCC)", path)
