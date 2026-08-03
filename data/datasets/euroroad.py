"""
Euroroad Dataset Loader (raw)

Source: http://konect.cc/networks/subelj_euroroad/
    - 1,174 nodes (cities), 1,417 undirected edges (E-road segments)
    - Undirected, unweighted, 1-indexed
    - No inherent node features — uses log(1 + degree) as synthetic features
    - No node labels

The RAW file, which is what CoreHD and BPD's Table I print (1,177 / 1,417 there;
the three-node difference is their own preprocessing). `--dataset road_eu` is the
1,039-node giant component GND uses. Both are in the literature under the same
name, so which one a number belongs to has to be stated
(research/critical_node_detection.md §6.4).

Original paper: Šubelj & Bajec, "Robust network community detection using
    balanced propagation", Eur. Phys. J. B 81, 2011
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_archive, load_edge_list

euroroad_url = "http://konect.cc/files/download.tsv.subelj_euroroad.tar.bz2"


def download_euroroad() -> Path:
    return download_archive(
        "euroroad", euroroad_url, "download.tsv.subelj_euroroad.tar.bz2"
    )


def load_euroroad(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("Euroroad", path)
