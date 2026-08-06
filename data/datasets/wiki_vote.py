"""
Wikipedia Vote Dataset Loader

Downloads and loads the SNAP wiki-Vote administrator-election network.

Source: https://snap.stanford.edu/data/wiki-Vote.html
    - 7,115 nodes (users), 103,689 directed edges (votes)
    - Directed: edge (a, b) means user a voted on b's adminship promotion
    - Dense for its size (avg total degree 29): the only dense directed graph
      in our suite
    - No inherent node features: uses log(1 + total degree) as synthetic features
    - No node labels

ToupleGDD reports this graph as "Wiki-2"; its "Wiki-1" is a DIFFERENT 889-node
graph from Network Repository (research/influence_maximization.md §6.3).
MOEIM reports the largest
connected component, 7,066 / 103,663.

Original paper: Leskovec et al., "Signed Networks in Social Media," CHI 2010
"""

import gzip
import os
import urllib.request
from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.graph_utils import degree_features, edges_to_adjacency, remap_to_contiguous

wiki_vote_url = "https://snap.stanford.edu/data/wiki-Vote.txt.gz"
data_dir = Path(__file__).resolve().parent.parent / "raw" / "wiki_vote"


def download_wiki_vote() -> Path:
    """Download and extract the wiki-Vote edge list if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    gz_path = data_dir / "wiki-Vote.txt.gz"
    txt_path = data_dir / "wiki-Vote.txt"

    if txt_path.exists():
        print(f"[ok] Wiki-Vote already downloaded at {txt_path}")
        return txt_path

    if not gz_path.exists():
        print(f"[get] Downloading Wiki-Vote from {wiki_vote_url} ...")
        urllib.request.urlretrieve(wiki_vote_url, gz_path)
        print(f"[ok] Saved to {gz_path}")

    print("[get] Extracting ...")
    with gzip.open(gz_path, "rb") as gz_file:
        txt_path.write_bytes(gz_file.read())
    print(f"[ok] Extracted to {txt_path}")

    return txt_path


def load_wiki_vote(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Load the wiki-Vote network. Tab-separated directed pairs with a `#` header;
    the original ids are sparse (max 8,297 for 7,115 nodes) so they are remapped.

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

    in_degrees = np.array(adjacency.sum(axis=0)).flatten()
    out_degrees = np.array(adjacency.sum(axis=1)).flatten()
    print(f"[ok] Wiki-Vote loaded: {num_nodes} nodes, {adjacency.nnz} directed edges")
    print(f"    In-degree: avg: {in_degrees.mean():.1f}, max: {in_degrees.max():.0f}")
    print(
        f"    Out-degree: avg: {out_degrees.mean():.1f}, max: {out_degrees.max():.0f}"
    )

    return adjacency, node_feats, node_labels, num_nodes
