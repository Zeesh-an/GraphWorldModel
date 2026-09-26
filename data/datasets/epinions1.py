"""
soc-Epinions1 Dataset Loader

Source: https://snap.stanford.edu/data/soc-Epinions1.html
    - 75,879 nodes, 508,837 arcs
    - Directed (who-trusts-whom on Epinions)
    - No inherent node features: uses log(1 + degree) as synthetic features
    - No node labels

Tong et al.'s `Epinions` (INFOCOM 2017: 75,879 / 508,837, avg degree 13.4),
digit for digit this file. Warning: NOT the same graph as our `epinions`, which is SNAP's
SIGNED soc-sign-epinions at 131,828 / 841,372 with the sign dropped. Two Epinions
graphs, one name, a 74% difference in node count.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_gzip, load_edge_list

epinions1_url = "https://snap.stanford.edu/data/soc-Epinions1.txt.gz"


def download_epinions1() -> Path:
    return download_gzip("epinions1", epinions1_url, "soc-Epinions1.txt.gz")


def load_epinions1(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("soc-Epinions1", path, directed=True)
