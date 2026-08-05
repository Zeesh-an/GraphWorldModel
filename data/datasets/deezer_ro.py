"""
Deezer Romania (GEMSEC-RO) Dataset Loader

Source: https://snap.stanford.edu/data/gemsec-Deezer.html
    - 41,773 nodes, 125,826 undirected edges [derived, counted from the download]
    - Undirected, unweighted CSV with a `node_1,node_2` header, 0-based ids
    - No inherent node features — uses log(1 + degree)

RLGN's `GEMSEC-RO` column (`research/epidemic_control.md` §5.1), and one of only
five graphs that paper reports — the single most comparable published table to what
this task produces, since it runs SIR-style dynamics with a per-step test budget
and reports % healthy at a fixed horizon.

Warning: RLGN'S OWN TABLE S4 IS INTERNALLY INCONSISTENT for this row, and our
version forensics is where that shows. It lists GEMSEC-RO at **41,773 / 222,887**.
Counted from the SNAP download [derived]: RO is 41,773 / 125,826 and **HU** is
47,538 / 222,887 — so RLGN's node count is Romania's and its edge count is
Hungary's. We load Romania at its own true edge count and say so, rather than
matching a number that describes neither file.

Warning: NOT `deezer`. That key is the HUNGARY subgraph (47,538 / 222,887), which
is IVGD's scalability row for source localization. Same release, three country
files with overlapping 0-based ids, and the offset union (143,884 / 846,915)
matches nothing published — see `data/datasets/deezer.py` for the full forensics.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.deezer import download_deezer
from data.graph_utils import degree_features, edges_to_adjacency, read_pairs

country = "RO"


def download_deezer_ro() -> Path:
    """Reuse `deezer`'s archive — one download serves all three country files."""
    return download_deezer().parent / f"{country}_edges.csv"


def load_deezer_ro(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    raw_edges = read_pairs(path, skip_rows=1)  # shape: (E, 2)
    num_nodes = int(raw_edges.max()) + 1

    adjacency = edges_to_adjacency(
        raw_edges[:, 0], raw_edges[:, 1], num_nodes, directed=False
    )
    node_feats = degree_features(adjacency, directed=False)  # shape: (N, 1)
    node_labels = np.zeros(num_nodes, dtype=np.int32)

    degrees = np.array(adjacency.sum(axis=1)).flatten()
    print(
        f"[✓] Deezer Romania (GEMSEC-RO) loaded: {num_nodes} nodes, "
        f"{adjacency.nnz // 2} undirected edges "
        f"(RLGN's Table S4 quotes 41,773 / 222,887 — its edge count is Hungary's)"
    )
    print(f"    Avg degree: {degrees.mean():.1f}, max degree: {degrees.max():.0f}")

    return adjacency, node_feats, node_labels, num_nodes
