"""
PGP Web of Trust Dataset Loader

Source: http://konect.cc/networks/arenas-pgp/
    - 10,680 nodes, 24,316 undirected edges (PGP key signatures)
    - Undirected, unweighted, 1-indexed
    - No inherent node features: uses log(1 + degree) as synthetic features
    - No node labels

A standard dismantling benchmark: Collective Influence, Min-Sum and Wandelt all
report it. Sparse (mean degree 4.6) with
a pronounced community structure, so it separates connectivity-driven methods
from degree-driven ones far better than the hub-dominated social graphs do.

Original paper: Boguñá, Pastor-Satorras, Díaz-Guilera, Arenas, "Models of social
    networks based on social distance attachment", Phys. Rev. E 70, 2004
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_archive, load_edge_list

pgp_url = "http://konect.cc/files/download.tsv.arenas-pgp.tar.bz2"


def download_pgp() -> Path:
    return download_archive("pgp", pgp_url, "download.tsv.arenas-pgp.tar.bz2")


def load_pgp(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("PGP web of trust", path)
