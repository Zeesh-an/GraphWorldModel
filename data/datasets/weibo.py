"""
Weibo Dataset Loader

Loads the Sina Weibo follower network (Influence Locality dataset).

Source: https://www.aminer.cn/influencelocality
    - 1,787,443 users, ~216M directed follow edges
    - Directed: edge u -> v means v follows u (influence flows u -> v)
    - No inherent node features — uses log(1 + total degree) as synthetic features
    - No node labels

NO auto-download: AMiner gates the archive behind registration, so
weibo_network.txt must be downloaded manually (see download_weibo error).

Original paper: Zhang et al., "Social Influence Locality for Modeling
    Retweeting Behaviors," IJCAI 2013
"""

from array import array
from pathlib import Path
import numpy as np
import scipy.sparse as sp

weibo_page_url = "https://www.aminer.cn/influencelocality"
data_dir = Path(__file__).resolve().parent.parent / "raw" / "weibo"


def download_weibo() -> Path:
    """Locate the manually-downloaded Weibo network file (no auto-download)."""
    network_path = data_dir / "weibo_network.txt"

    if network_path.exists():
        print(f"[✓] Weibo network found at {network_path}")
        return network_path

    raise FileNotFoundError(
        f"Weibo network file not found at {network_path}. AMiner requires "
        f"registration, so download it manually: (1) register at "
        f"{weibo_page_url}, (2) download weibo_network.tar.gz, (3) extract "
        f"weibo_network.txt into {data_dir}/"
    )


def load_weibo(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Load the Weibo follower network. Format (Influence Locality release):
    first line `<num_users> <num_edges>`; each following line
    `<user_id> <k> <followee_1> <flag_1> ... <followee_k> <flag_k>` where
    flag is 1 for reciprocal follows. User ids are already contiguous 0..N-1.

    Emits influence-direction edges followee -> follower.

    Returns
    -------
    adjacency: scipy.sparse.csr_matrix (N, N) binary directed adjacency
    node_feats: np.ndarray (N, 1) float32 -- log(1 + total degree) features
    node_labels: np.ndarray (N,) int32 -- placeholder zeros
    num_nodes: int -- number of nodes
    """
    # ~216M edges: compact typed buffers instead of Python int lists
    sources = array("i")
    destinations = array("i")

    with open(path) as file:
        header = file.readline().split()
        num_nodes = int(header[0])

        for line in file:
            parts = line.split()
            if not parts:
                continue

            follower = int(parts[0])
            followee_count = int(parts[1])

            for pair in range(followee_count):
                followee = int(parts[2 + 2 * pair])
                # Influence flows from the followed account to the follower
                sources.append(followee)
                destinations.append(follower)

    source_array = np.frombuffer(sources, dtype=np.int32)
    destination_array = np.frombuffer(destinations, dtype=np.int32)
    values = np.ones(len(source_array), dtype=np.float32)

    # Build sparse adjacency, deduplicate via csr conversion
    adjacency = sp.csr_matrix(
        (values, (source_array, destination_array)), shape=(num_nodes, num_nodes)
    )
    adjacency = (adjacency > 0).astype(np.float32)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    # Degree-based node features (no natural features in this dataset)
    out_degrees = np.array(adjacency.sum(axis=1)).flatten()
    in_degrees = np.array(adjacency.sum(axis=0)).flatten()
    total_degrees = out_degrees + in_degrees
    node_feats = (
        np.log1p(total_degrees).reshape(-1, 1).astype(np.float32)
    )  # shape: (N, 1)

    # Placeholder labels
    node_labels = np.zeros(num_nodes, dtype=np.int32)

    print(f"[✓] Weibo loaded: {num_nodes} nodes, {adjacency.nnz} directed edges")
    print(
        f"    Avg out-degree: {out_degrees.mean():.1f}, "
        f"max out-degree: {out_degrees.max():.0f}"
    )

    return adjacency, node_feats, node_labels, num_nodes
