"""
p2p-Gnutella31 Dataset Loader

Source: https://snap.stanford.edu/data/p2p-Gnutella31.html
    - SNAP's raw crawl is 62,586 nodes / 147,892 arcs; we load its giant
      component **62,561 / 147,878**, UNDIRECTED, as CoreHD's "P2P" and FINDER's
      headline ND network both do
      (research/critical_node_detection.md §8.2 traps 7 and 9)
    - Sparse original ids — remapped by `load_edge_list` via the adjacency
    - No inherent node features — uses log(1 + degree) as synthetic features
    - No node labels

Warning: SCALE WARNING. At 62.6K nodes this is well past what the NDlib rollout +
selector pipeline simulates in reasonable time; it is registered because it is the
one large graph CoreHD, BPD, FINDER, DCRS and MIND all report, i.e. the target for
a scalable-simulation follow-up, not a day-one dataset. Generation on it will be
slow, and `--baselines greedy_blocking` will not finish.

`lcc=True` because SNAP ships the raw crawl including disconnected fragments and
every dismantling paper runs on the giant component (§8.2 trap 7).

Original paper: Ripeanu, Foster, Iamnitchi, "Mapping the Gnutella network",
    IEEE Internet Computing 6(1), 2002
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_archive, load_edge_list

p2p_gnutella_url = "https://snap.stanford.edu/data/p2p-Gnutella31.txt.gz"


def download_p2p_gnutella() -> Path:
    return download_archive(
        "p2p_gnutella",
        p2p_gnutella_url,
        "p2p-Gnutella31.txt.gz",
        member="p2p-Gnutella31.txt",
    )


def load_p2p_gnutella(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("p2p-Gnutella31", path, lcc=True)
