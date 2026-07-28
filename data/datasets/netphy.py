"""
NetPHY Dataset Loader

Downloads and loads the arXiv Physics-section collaboration network — NetHEPT's
sibling, and the other half of the classical IM benchmark pair.

Source: https://www.microsoft.com/en-us/research/people/weic/selected-projects/
    (Wei Chen's own release, `weic-graphdata.zip`, which ships hep.txt + phy.txt)
    - 37,154 nodes (authors), 174,161 undirected edges after deduplication
    - Undirected: an edge means the two authors co-wrote at least one paper
    - The raw file lists 231,584 edge LINES — one per co-authored paper, so a
      frequently-collaborating pair appears many times. See the note below.
    - No inherent node features — uses log(1 + degree) as synthetic features
    - No node labels

Reconciling the three published edge counts for this graph:

    231,584   raw lines in phy.txt (multi-edges kept)  <- Chen et al. Table 1
    180,826   unique ORDERED pairs                     <- SSA/D-SSA's "181K",
                                                          avg degree 9.73
    174,161   unique UNDIRECTED pairs                  <- what we build

We keep unique undirected pairs, which is the only count that makes
`p(u→v) = 1/in-degree(v)` well defined.

Original paper: Wei Chen et al., "Scalable Influence Maximization for Prevalent
    Viral Marketing in Large-Scale Social Networks," KDD 2010
"""

import os
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.graph_utils import degree_features, edges_to_adjacency

netphy_url = (
    "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/"
    "weic-graphdata.zip"
)
# Microsoft's CDN answers 403 to the default Python-urllib agent
browser_agent = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)
data_dir = Path(__file__).resolve().parent.parent / "raw" / "netphy"


def download_netphy() -> Path:
    """Download and extract Wei Chen's graph data if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    zip_path = data_dir / "weic-graphdata.zip"
    edges_path = data_dir / "phy.txt"

    if edges_path.exists():
        print(f"[✓] NetPHY already downloaded at {edges_path}")
        return edges_path

    if not zip_path.exists():
        print(f"[↓] Downloading NetPHY from {netphy_url} ...")
        request = urllib.request.Request(
            netphy_url, headers={"User-Agent": browser_agent}
        )
        with urllib.request.urlopen(request) as response:
            zip_path.write_bytes(response.read())
        print(f"[✓] Saved to {zip_path}")

    print("[↓] Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(data_dir)
    print(f"[✓] Extracted to {data_dir}")

    return edges_path


def load_netphy(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Load the NetPHY collaboration network. The first line is `n m`; the
    remaining m lines are 0-indexed space-separated pairs with multi-edges.

    Returns
    -------
    adjacency: scipy.sparse.csr_matrix (N, N) undirected adjacency
    node_feats: np.ndarray (N, 1) float32 -- log(1 + degree) features
    node_labels: np.ndarray (N,) int32 -- placeholder zeros
    num_nodes: int -- number of nodes
    """
    with open(path) as file:
        declared_nodes, declared_lines = (
            int(value) for value in file.readline().split()
        )

    raw_edges = np.loadtxt(path, skiprows=1, dtype=np.int64)  # shape: (E, 2)

    # Trust the header for N: the file's last few ids are isolated authors that
    # never appear in an edge, and dropping them would shift every node index
    num_nodes = declared_nodes

    adjacency = edges_to_adjacency(
        raw_edges[:, 0], raw_edges[:, 1], num_nodes, directed=False
    )

    node_feats = degree_features(adjacency, directed=False)  # shape: (N, 1)
    node_labels = np.zeros(num_nodes, dtype=np.int32)

    degrees = np.array(adjacency.sum(axis=1)).flatten()
    n_edges_undirected = adjacency.nnz // 2
    n_isolates = int((degrees == 0).sum())
    print(
        f"[✓] NetPHY loaded: {num_nodes} nodes, "
        f"{n_edges_undirected} undirected edges "
        f"(from {declared_lines} raw lines — multi-edges collapsed)"
    )
    print(f"    Avg degree: {degrees.mean():.1f}, max degree: {degrees.max():.0f}")
    if n_isolates > 0:
        print(f"    Isolate nodes (degree 0): {n_isolates}")

    return adjacency, node_feats, node_labels, num_nodes
