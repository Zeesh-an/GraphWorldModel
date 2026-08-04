"""
Crime Dataset Loader (dismantling version)

Source: https://github.com/renxiaolong/Generalized-Network-Dismantling
    - 754 nodes, 2,127 undirected edges
    - Undirected, unweighted, 1-indexed
    - No inherent node features — uses log(1 + degree) as synthetic features
    - No node labels

Warning: Name collision (research/critical_node_detection.md §6.4). KONECT's
`moreno_crime` is the RAW BIPARTITE St. Louis person-crime network, 1,380 nodes
(829 persons + 551 crimes) and 1,476 edges. The "Crime" that GND, FINDER, NIRM
and SPR all report is this one: the PERSON PROJECTION restricted to its giant
component, 754 / 2,127. Loading the bipartite file instead would produce numbers
that look like theirs and are not.

Original data: Decker, Moore et al. St. Louis homicide network, via
    Ribeiro et al. and the GND supplementary datasets.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_plain, load_edge_list

crime_url = (
    "https://raw.githubusercontent.com/renxiaolong/Generalized-Network-Dismantling/"
    "master/Datasets_Main_Paper/Crime_Gcc.txt"
)


def download_crime() -> Path:
    return download_plain("crime", crime_url, "Crime_Gcc.txt")


def load_crime(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("Crime (GCC)", path)
