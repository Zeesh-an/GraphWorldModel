"""
Openflights Dataset Loader

Source: https://networkrepository.com/inf-openflights.php (Opsahl snapshot)
    - 2,939 nodes (airports), 30,501 arcs -> 15,677 undirected edges once
      symmetrized, which is the count GND and MIND report
    - Loaded UNDIRECTED: the whole dismantling literature treats it that way
      (research/critical_node_detection.md §8.2 trap 9)
    - No inherent node features — uses log(1 + degree) as synthetic features
    - No node labels

Warning: Name collision (§6.4). KONECT publishes a bigger, later `openflights`
snapshot at 3,425 / 37,595 arcs and papers cite the two interchangeably. This is
the Opsahl one.

Original data: Opsahl, "Why Anchorage is not (that) important", 2011
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_archive, load_edge_list

openflights_url = "https://nrvis.com/download/data/inf/inf-openflights.zip"


def download_openflights() -> Path:
    return download_archive(
        "openflights",
        openflights_url,
        "inf-openflights.zip",
        member="inf-openflights.edges",
    )


def load_openflights(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("Openflights", path)
