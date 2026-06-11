"""
Twitter Dataset Loader

Downloads and loads the SNAP ego-Twitter social network.

Source: https://snap.stanford.edu/data/ego-Twitter.html
    - 81,306 nodes, 1,768,149 directed edges (follow graph)
    - Union of 973 ego networks
    - Originally directed: edge (a, b) means a follows b
    - Symmetrized to undirected during loading
    - No inherent node features — uses log(1 + degree) as synthetic features
    - No node labels
"""

import gzip
import urllib.request
import numpy as np
import scipy.sparse as sp
from pathlib import Path

TWITTER_URL = "https://snap.stanford.edu/data/twitter_combined.txt.gz"
DATA_DIR = Path(__file__).resolve().parent.parent / "twitter"


def download_twitter() -> Path:
    """Download and extract Twitter edge list if not already present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    gz_path = DATA_DIR / "twitter_combined.txt.gz"
    txt_path = DATA_DIR / "twitter_combined.txt"

    if txt_path.exists():
        print(f"[✓] Twitter already downloaded at {txt_path}")
        return txt_path

    if not gz_path.exists():
        print(f"[↓] Downloading Twitter from {TWITTER_URL} ...")
        urllib.request.urlretrieve(TWITTER_URL, gz_path)
        print(f"[✓] Saved to {gz_path}")

    print("[↓] Extracting ...")
    with gzip.open(gz_path, "rb") as f_in:
        with open(txt_path, "wb") as f_out:
            f_out.write(f_in.read())
    print(f"[✓] Extracted to {txt_path}")

    return txt_path


def load_twitter(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Load Twitter ego network. Symmetrizes directed follow edges
    into an undirected graph. Remaps IDs to contiguous 0..N-1.

    Returns
    -------
    adj: scipy.sparse.csr_matrix (N, N) undirected adjacency
    node_feats: np.ndarray (N, 1) float32 -- log(1 + degree) features
    node_labels: np.ndarray (N,) int32 -- placeholder zeros
    N: int -- number of nodes
    """
    # Parse space-separated directed edge list
    raw_src = []
    raw_dst = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            raw_src.append(int(parts[0]))
            raw_dst.append(int(parts[1]))

    # Collect unique node IDs and remap to contiguous 0..N-1
    all_ids = sorted(set(raw_src) | set(raw_dst))
    id_to_idx = {nid: idx for idx, nid in enumerate(all_ids)}
    N = len(id_to_idx)

    # Build symmetric (undirected) edge arrays
    src_list = []
    dst_list = []

    for a, b in zip(raw_src, raw_dst):
        ia, ib = id_to_idx[a], id_to_idx[b]
        # Store both directions for undirected graph
        src_list.append(ia)
        dst_list.append(ib)
        src_list.append(ib)
        dst_list.append(ia)

    src = np.array(src_list, dtype=np.int32)
    dst = np.array(dst_list, dtype=np.int32)
    data = np.ones(len(src), dtype=np.float32)

    # Build sparse adjacency, deduplicate via csr conversion
    adj = sp.csr_matrix((data, (src, dst)), shape=(N, N))
    adj = (adj > 0).astype(np.float32)
    adj.setdiag(0)
    adj.eliminate_zeros()

    # Degree-based node features (no natural features in this dataset)
    degrees = np.array(adj.sum(axis=1)).flatten()
    node_feats = np.log1p(degrees).reshape(-1, 1).astype(np.float32)  # shape: (N, 1)

    # Placeholder labels
    node_labels = np.zeros(N, dtype=np.int32)

    n_edges_undirected = adj.nnz // 2
    print(f"[✓] Twitter loaded: {N} nodes, {n_edges_undirected} undirected edges")
    print(f"    Avg degree: {degrees.mean():.1f}, max degree: {degrees.max():.0f}")

    return adj, node_feats, node_labels, N
