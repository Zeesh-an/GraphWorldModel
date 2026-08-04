"""
soc-Pokec Dataset Loader

Source: https://snap.stanford.edu/data/soc-Pokec.html
    - 1,632,803 nodes, 30,622,564 arcs [verified, SNAP's own statistics page —
      NOT counted from the file, which we have deliberately not downloaded]
    - Directed (Pokec social network friendships, Slovakia)
    - No inherent node features — uses log(1 + degree) as synthetic features
    - No node labels

SandIMIN's and fair IBM's largest graph (§6.2). SCALE WARNING: 30.6M arcs is past
what the NDlib rollout path simulates in reasonable time, so this is a scalability
target rather than a day-one dataset — the same standing as `twitter`, `digg` and
`youtube`. SandIMIN's Table 5 row on it is the one where its trivial LHGA heuristic
beats both principled methods by 27% at k = 10.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_gzip, load_edge_list

pokec_url = "https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz"


def download_pokec() -> Path:
    return download_gzip("pokec", pokec_url, "soc-pokec-relationships.txt.gz")


def load_pokec(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("soc-Pokec", path, directed=True)
