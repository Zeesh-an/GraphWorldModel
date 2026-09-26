"""
CiteSeer Dataset Loader

Downloads and loads the CiteSeer citation network, standardized exactly as
`cora_ml` is so the two are preprocessed identically.

Source: https://github.com/abojchevski/graph2gauss
    - raw file:  4,230 nodes
    - as loaded: 1,681 nodes, 2,902 undirected edges (the largest connected
      component after symmetrize + drop self-loops) [derived, counted from the
      download on 2026-08-04]
    - Node features: bag-of-words vectors
    - Node labels: topic classes

DIPT's third graph, reported at Path
Precision 0.593 and Jaccard 0.421: between its Cora-ML (0.622) and its Power Grid
(0.680), both of which we already load. Adding it makes THREE of DIPT's five rows
reachable.

Warning: OUR COUNT IS NOT THE ONE THAT CIRCULATES, and the gap is large. DIPT
publishes NO dataset table, and the "3,327 / 4,732" quoted for CiteSeer
everywhere is the LINQS release without a largest-component filter. graph2gauss's
file standardized the way `cora_ml` is standardized gives 1,681 / 2,902: half the
nodes, because CiteSeer is unusually fragmented and 2,549 of its 4,230 nodes sit
outside the giant component. Both numbers are right about different objects.

The consequence is concrete: a DIPT comparison on this row is NOT cell-for-cell
until their preprocessing is confirmed, and the 3,327 / 4,732 pair is marked
[claim] for exactly this reason. `cora_ml` and `power_grid` are the two DIPT
rows where our graph is known to match, and they are the ones to lead with.

Original paper: Giles, Bollacker & Lawrence, "CiteSeer: An Automatic Citation
    Indexing System," ACM DL 1998
"""

import os
import urllib.request
from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.graph_utils import largest_connected_component

citeseer_url = "https://github.com/abojchevski/graph2gauss/raw/master/data/citeseer.npz"
data_dir = Path(__file__).resolve().parent.parent / "raw" / "citeseer"


def download_citeseer() -> Path:
    os.makedirs(data_dir, exist_ok=True)
    destination = data_dir / "citeseer.npz"

    if destination.exists():
        print(f"[ok] CiteSeer already downloaded at {destination}")
        return destination

    # Written to a .part file and renamed, so a run killed part-way does not leave
    # a truncated archive that later fails with BadZipFile
    print(f"[get] Downloading CiteSeer from {citeseer_url} ...")
    partial = destination.with_suffix(".npz.part")
    urllib.request.urlretrieve(citeseer_url, partial)
    partial.rename(destination)
    print(f"[ok] Saved to {destination}")

    return destination


def load_citeseer(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Returns
    -------
    adjacency: scipy.sparse.csr_matrix (N, N) undirected adjacency, LCC only
    features: np.ndarray (N, F) float32 bag-of-words
    labels: np.ndarray (N,) int32 class labels
    num_nodes: int -- number of nodes
    """
    raw = np.load(path, allow_pickle=True)

    adjacency = sp.csr_matrix(
        (raw["adj_data"], raw["adj_indices"], raw["adj_indptr"]),
        shape=raw["adj_shape"],
    ).astype(np.float32)

    features = (
        sp.csr_matrix(
            (raw["attr_data"], raw["attr_indices"], raw["attr_indptr"]),
            shape=raw["attr_shape"],
        )
        .toarray()
        .astype(np.float32)
    )

    labels = raw["labels"].astype(np.int32)
    raw_num_nodes = adjacency.shape[0]

    # standardize(): make_unweighted + make_undirected + no_self_loops
    adjacency = ((adjacency + adjacency.T) > 0).astype(np.float32)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    # standardize(): select_lcc, features and labels take the same indices
    adjacency, keep = largest_connected_component(adjacency)
    features = features[keep]
    labels = labels[keep]
    num_nodes = int(keep.shape[0])

    degrees = np.array(adjacency.sum(axis=1)).flatten()
    print(
        f"[ok] CiteSeer loaded: {num_nodes} nodes, {adjacency.nnz // 2} undirected "
        f"edges, {features.shape[1]} features, {len(np.unique(labels))} classes"
    )
    print(
        f"    Standardized from {raw_num_nodes} nodes "
        f"(dropped {raw_num_nodes - num_nodes} outside the largest component)"
    )
    print(f"    Avg degree: {degrees.mean():.1f}, max degree: {degrees.max():.0f}")

    return adjacency, features, labels, num_nodes
