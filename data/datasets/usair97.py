"""
USAir97 Dataset Loader

Source: https://networkrepository.com/inf-USAir97.php (Pajek / SuiteSparse 1529)
    - 332 nodes (airports), 2,126 undirected edges (1997 US air routes)
    - Undirected; the file carries passenger-flow weights, which we drop —
      IC/LT probabilities come from `build_edge_index` like every other graph
    - No inherent node features — uses log(1 + degree) as synthetic features
    - No node labels

Matrix Market coordinate format: `%` comment lines, then ONE dimension header
(`332 332 2126`) that is not commented, then `row col value` triples in the lower
triangle only. `skip_rows=1` drops the header; `edges_to_adjacency` mirrors.

The smallest graph in the Ventresca CNP benchmark that Wandelt also dismantles
(research/critical_node_detection.md §7), so it is one of the few networks where
the OR branch and the physics branch actually meet.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_archive, load_edge_list

usair97_url = "https://nrvis.com/download/data/inf/inf-USAir97.zip"


def download_usair97() -> Path:
    return download_archive(
        "usair97", usair97_url, "inf-USAir97.zip", member="inf-USAir97.mtx"
    )


def load_usair97(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("USAir97", path, skip_rows=1)
