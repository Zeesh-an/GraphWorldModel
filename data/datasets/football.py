"""
American College Football Dataset Loader

Source: https://public.websites.umich.edu/~mejn/netdata/football.zip
    - 115 nodes, 613 undirected edges [verified, KONECT]
    - Undirected, unweighted GML
    - 12 conference labels: the ground-truth communities

Girvan & Newman's (PNAS 2002) football graph: Division I-A teams, an edge per
regular-season game, and conference membership as the ground-truth partition. Part
of the epidemic-control benchmark catalogue and loaded here for one reason: at
115 nodes with 12 known communities and near-regular degree (mean 10.7, max 12), it
is the one graph in this repo where the DEGREE heuristic has almost nothing to
grab. A heuristic-collapse result is expected on power-law graphs; this is the
control that shows what happens when the tail is gone.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.graph_utils import degree_features, edges_to_adjacency, extract, fetch

raw_root = Path(__file__).resolve().parent.parent / "raw"
football_url = "https://public.websites.umich.edu/~mejn/netdata/football.zip"


def download_football() -> Path:
    directory = raw_root / "football"
    inner = directory / "football.gml"

    if inner.exists():
        return inner

    extract(fetch(football_url, directory / "football.zip"), directory)

    return inner


def load_football(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Parse the GML by hand rather than through networkx.

    `nx.read_gml` on this file needs `label='id'` and still trips over the
    duplicate team names Newman's original carries; the format is three token types
    and reading them directly is shorter than the workaround.
    """
    nodes, conferences, edges = [], [], []
    current = {}

    for line in Path(path).read_text(errors="replace").splitlines():
        parts = line.strip().split()
        if not parts:
            continue

        key = parts[0]
        if key in ("node", "edge"):
            current = {}
        elif key == "id":
            nodes.append(int(parts[1]))
        elif key == "value":
            conferences.append(int(parts[1]))
        elif key == "source":
            current["source"] = int(parts[1])
        elif key == "target":
            current["target"] = int(parts[1])
            edges.append((current["source"], current["target"]))

    num_nodes = len(nodes)
    sources = np.array([edge[0] for edge in edges], dtype=np.int64)
    targets = np.array([edge[1] for edge in edges], dtype=np.int64)
    adjacency = edges_to_adjacency(sources, targets, num_nodes, directed=False)

    node_feats = degree_features(adjacency, directed=False)  # shape: (N, 1)
    node_labels = np.array(conferences, dtype=np.int32)

    degrees = np.array(adjacency.sum(axis=1)).flatten()
    print(
        f"[ok] American College Football loaded: {num_nodes} nodes, "
        f"{adjacency.nnz // 2} undirected edges, "
        f"{len(set(conferences))} conference labels"
    )
    print(f"    Avg degree: {degrees.mean():.1f}, max degree: {degrees.max():.0f}")

    return adjacency, node_feats, node_labels, num_nodes
