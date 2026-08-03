"""
RoadEU Dataset Loader

Source: https://github.com/renxiaolong/Generalized-Network-Dismantling
    - 1,039 nodes, 1,305 undirected edges
    - Undirected, unweighted, 1-indexed
    - No inherent node features — uses log(1 + degree) as synthetic features
    - No node labels

The European E-road network's giant component, as GND uses it. CoreHD and BPD's
Table I print the RAW `subelj_euroroad` counts (1,174 / 1,417) under the same
name, so the two are not interchangeable — `--dataset euroroad` is the raw one.

Mesh-like and near-planar (mean degree 2.5), which is the graph class
research/critical_node_detection.md §9.4 predicts learned dismantlers lose on.

Original data: Šubelj & Bajec, "Robust network community detection using
    balanced propagation", Eur. Phys. J. B 81, 2011
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_plain, load_edge_list

# Upstream filename really is `RodeEU` — a typo in the GND repository
road_eu_url = (
    "https://raw.githubusercontent.com/renxiaolong/Generalized-Network-Dismantling/"
    "master/Datasets_SI/RodeEU_gcc.txt"
)


def download_road_eu() -> Path:
    return download_plain("road_eu", road_eu_url, "RodeEU_gcc.txt")


def load_road_eu(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("RoadEU (GCC)", path)
