"""
LiveJournal Dataset Loader

Downloads and loads the SNAP soc-LiveJournal1 friendship network: the second
largest of Han et al.'s adaptive-IM benchmarks
(research/adaptive_online_im.md §5.2, §6.2).

Source: https://snap.stanford.edu/data/soc-LiveJournal1.html
    - 4,847,571 nodes (members), 68,993,773 arcs (friendship declarations)
    - Directed: friendship is declared one way and need not be reciprocated
    - Han et al. quote `4.85M / 69.0M`, average degree 28.5: agrees with SNAP
      to the digit, so there is no version collision here
    - No inherent node features: uses log(1 + total degree) as synthetic features
    - No node labels

SCALE WARNING. This graph loads but is far beyond what the NDlib rollout and
selector pipeline can simulate in reasonable time: the same standing caveat
that applies to `twitter`, `digg`, `youtube` and `weibo` (see CLAUDE.md). It is
here because Han et al. report on it and because their own AdaptIM-1 "cannot
finish under the case of b < 5 for the largest datasets LiveJournal and Orkut,
due to the memory overflow": that sentence is the cost argument this task
exists to make, so the graph is worth having a loader for even before the
simulation side can reach it.

Original paper: Backstrom, Huttenlocher, Kleinberg & Lan, "Group Formation in
    Large Social Networks," KDD 2006
"""

import gzip
import os
import urllib.request
from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.graph_utils import degree_features, edges_to_adjacency, remap_to_contiguous

livejournal_url = "https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz"
data_dir = Path(__file__).resolve().parent.parent / "raw" / "livejournal"


def download_livejournal() -> Path:
    """Download and extract the friendship edge list (~1 GB uncompressed)."""
    os.makedirs(data_dir, exist_ok=True)
    gz_path = data_dir / "soc-LiveJournal1.txt.gz"
    txt_path = data_dir / "soc-LiveJournal1.txt"

    if txt_path.exists():
        print(f"[ok] LiveJournal already downloaded at {txt_path}")
        return txt_path

    if not gz_path.exists():
        print(f"[get] Downloading LiveJournal from {livejournal_url} ...")
        urllib.request.urlretrieve(livejournal_url, gz_path)
        print(f"[ok] Saved to {gz_path}")

    print("[get] Extracting (this file is large) ...")
    with gzip.open(gz_path, "rb") as gz_file:
        txt_path.write_bytes(gz_file.read())
    print(f"[ok] Extracted to {txt_path}")

    return txt_path


def load_livejournal(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Load soc-LiveJournal1 as a directed graph.

    Returns
    -------
    adjacency: scipy.sparse.csr_matrix (N, N) directed adjacency
    node_feats: np.ndarray (N, 1) float32 -- log(1 + total degree) features
    node_labels: np.ndarray (N,) int32 -- placeholder zeros
    num_nodes: int -- number of nodes
    """
    raw_edges = np.loadtxt(path, comments="#", dtype=np.int64)  # shape: (E, 2)

    remapped, num_nodes = remap_to_contiguous(raw_edges)
    adjacency = edges_to_adjacency(
        remapped[:, 0], remapped[:, 1], num_nodes, directed=True
    )

    node_feats = degree_features(adjacency, directed=True)  # shape: (N, 1)
    node_labels = np.zeros(num_nodes, dtype=np.int32)

    print(f"[ok] LiveJournal loaded: {num_nodes} nodes, {adjacency.nnz} directed arcs")

    return adjacency, node_feats, node_labels, num_nodes
