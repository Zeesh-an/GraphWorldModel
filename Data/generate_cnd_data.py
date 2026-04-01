"""
Critical Node Detection Data Generation — Digg
================================================
Generates training/eval data for CND models on the Digg social network.

Source: https://datasets.syr.edu/datasets/Digg.html
    - 116,893 core users, ~2.6M friendship edges (undirected)
    - Edges to users outside the core set are dropped

Problem: Given graph G and budget k, find k nodes whose removal
causes maximum damage to network connectivity.

Forward process (deterministic):
    removal_set -> remove nodes + edges -> measure residual connectivity

Inverse problem:
    desired fragmentation -> infer optimal removal set

What this produces
------------------
    - samples_cnd.npz: (removal_sets, connectivity_vecs, damage metrics)
    - graph_data.npz: adjacency, node features (degree-based)
    - metadata_cnd.json: graph stats and generation config

Data format
-----------
graph_data.npz:
    edge_index: (2, E) int32  -- COO format [src, dst] (both directions stored)
    node_feats: (N, 1) float32 -- log(1 + degree) per node
    node_labels: (N,) int32  -- placeholder zeros (Digg has no node labels)
    ic_probs: (E,) float32 -- IC propagation prob = 1/in_degree(v)
    lt_weights: (E,) float32 -- LT edge weights (same as ic_probs)

samples_cnd.npz:
    removal_sets: (S, k) int32 -- S samples, each removing k nodes
    connectivity_vecs: (S, N) float32 -- 1.0 if node is in largest CC after removal
    n_components: (S,) int32 -- number of connected components after removal
    largest_cc_sizes: (S,) int32 -- size of largest remaining component
    pairwise_conn: (S,) int64 -- number of reachable node pairs after removal
    n_nodes: (1,) int32 -- total nodes in graph

Training tensor (built by loader): shape (S, N, 2)
    [:, :, 0] = binary removal vector (1 = node removed)
    [:, :, 1] = binary connectivity vector (1 = node in largest CC after removal)

Usage
-----
    python Data/generate_cnd_data.py [--samples 1000] [--k 10]

Then load with:
    from generate_cnd_data import load_samples
    removal_sets, conn_vecs, n_comp, lcc, pw = load_samples(Path('Data/digg'))
"""

import csv
import json
import time
import zipfile
import argparse
import urllib.request
import numpy as np
import networkx as nx
import scipy.sparse as sp
from pathlib import Path

# Download and load Digg
DIGG_URL = "https://datasets.syr.edu/uploads/1296588940/Digg-dataset.zip"
DATA_DIR = Path(__file__).parent / "digg"


def download_digg() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "Digg-dataset.zip"
    nodes_path = DATA_DIR / "Digg-dataset" / "data" / "nodes.csv"

    if nodes_path.exists():
        print(f"[✓] Digg already downloaded at {DATA_DIR}")
        return DATA_DIR / "Digg-dataset" / "data"

    if not zip_path.exists():
        print(f"[↓] Downloading Digg from {DIGG_URL} ...")
        urllib.request.urlretrieve(DIGG_URL, zip_path)
        print(f"[✓] Saved to {zip_path}")

    print("[↓] Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)

    print(f"[✓] Extracted to {DATA_DIR / 'Digg-dataset'}")

    return DATA_DIR / "Digg-dataset" / "data"


def load_digg(data_path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Load Digg social network. Keeps only edges between core nodes
    (those listed in nodes.csv). Remaps IDs to contiguous 0..N-1.

    Returns
    -------
    adj: scipy.sparse.csr_matrix (N, N) undirected adjacency
    node_feats: np.ndarray (N, 1) float32 -- log(1 + degree) features
    node_labels: np.ndarray (N,) int32 -- placeholder zeros
    N: int -- number of core nodes
    """
    # Load core node IDs
    core_ids = []
    with open(data_path / "nodes.csv") as f:
        for line in f:
            line = line.strip()
            if line:
                core_ids.append(int(line))

    core_set = set(core_ids)

    # Remap non-contiguous IDs to contiguous IDs from 0 to N - 1
    id_to_idx = {nid: idx for idx, nid in enumerate(sorted(core_ids))}
    N = len(id_to_idx)

    # Load edges, keep only those between core nodes
    src_list = []
    dst_list = []
    n_dropped = 0

    with open(data_path / "edges.csv") as f:
        reader = csv.reader(f)

        for row in reader:
            # Edge (a - b)
            a, b = int(row[0]), int(row[1])

            if a in core_set and b in core_set:
                ia, ib = id_to_idx[a], id_to_idx[b]

                # Store both directions for undirected graph
                src_list.append(ia)
                dst_list.append(ib)
                src_list.append(ib)
                dst_list.append(ia)
            else:
                n_dropped += 1

    src = np.array(src_list, dtype=np.int32)
    dst = np.array(dst_list, dtype=np.int32)
    data = np.ones(len(src), dtype=np.float32)

    # Build sparse adjacency, deduplicate via csr conversion, remove self-loops
    adj = sp.csr_matrix((data, (src, dst)), shape=(N, N))
    adj = (adj > 0).astype(np.float32)
    adj.setdiag(0)
    adj.eliminate_zeros()

    # Digg has no natural features, so degree-based node features are computed
    degrees = np.array(adj.sum(axis=1)).flatten()
    node_feats = np.log1p(degrees).reshape(-1, 1).astype(np.float32)  # (N, 1)

    # Placeholder labels (Digg has no node labels)
    node_labels = np.zeros(N, dtype=np.int32)

    n_edges_undirected = adj.nnz // 2
    print(f"[✓] Digg loaded: {N} nodes, {n_edges_undirected} undirected edges")
    print(f"    Dropped {n_dropped} edges to external nodes")
    print(f"    Avg degree: {degrees.mean():.1f}, max degree: {degrees.max():.0f}")

    return adj, node_feats, node_labels, N


def build_graph_data(
    adj: sp.csr_matrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build edge_index and edge probabilities from undirected adjacency.

    Returns
    -------
    edge_index: (2, E) int32  -- COO [src, dst] (both directions)
    ic_probs: (E,) float32 -- IC propagation prob per edge
    lt_weights: (E,) float32 -- LT edge weights
    """
    # Convert to COO and extract source and destination arrays from adjacency matrix
    adj_coo = adj.tocoo()
    src = adj_coo.row.astype(np.int32)
    dst = adj_coo.col.astype(np.int32)

    in_deg = np.array(adj.sum(axis=0)).flatten()  # (N,) in-degree
    in_deg = np.where(in_deg == 0, 1, in_deg)  # avoid divide-by-zero

    # Independent Cascade (IC): each edge gets probability = 1/in_degree(dst)
    ic_probs = (1.0 / in_deg[dst]).astype(np.float32)
    ic_probs = np.clip(ic_probs, 0.001, 0.5)  # cap for realism

    # Linear Threshold (LT): weights must sum to ≤ 1 per node (already true with 1/in_deg)
    lt_weights = ic_probs.copy()

    edge_index = np.stack([src, dst], axis=0)  # (2, E)

    # Note: For CND, ic_probs and lt_weights are not used. They are included for easier use with IM
    return edge_index, ic_probs, lt_weights


# Node Removal Simulation
def simulate_removal(
    removal_set: list[int],
    G: nx.Graph,
    N: int,
) -> tuple[np.ndarray, int, int, int]:
    """
    Remove nodes from graph and measure residual connectivity.

    Deterministic: given a fixed removal set, the outcome is fully
    determined by the graph topology. No stochastic process.

    Parameters
    ----------
    removal_set: nodes to remove (contiguous 0-indexed IDs)
    G: undirected NetworkX graph (not modified in-place)
    N: total number of nodes in the original graph

    Returns
    -------
    connectivity_vec: (N,) float32 -- 1.0 if node in largest CC, 0.0 otherwise. removed nodes always get 0.
    n_components: number of connected components after removal
    largest_cc_size: size of largest remaining component
    pairwise_conn: reachable node pairs = sum_i (|C_i| choose 2)
    """
    remaining = set(G.nodes()) - set(removal_set)
    G_residual = G.subgraph(remaining)

    components = list(nx.connected_components(G_residual))

    if not components:
        return np.zeros(N, dtype=np.float32), 0, 0, 0

    largest_cc = max(components, key=len)

    connectivity_vec = np.zeros(N, dtype=np.float32)
    for v in largest_cc:
        connectivity_vec[v] = 1.0

    pairwise_conn = sum(len(c) * (len(c) - 1) // 2 for c in components)

    return connectivity_vec, len(components), len(largest_cc), pairwise_conn


def generate_samples(
    N: int,
    G: nx.Graph,
    k: int = 10,
    n_samples: int = 1000,
    rng_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate (removal_set, connectivity) samples by:
        1. Sampling random removal sets of size k
        2. Computing residual connectivity (deterministic, no MC needed)

    Returns
    -------
    removal_sets: (n_samples, k) int32
    connectivity_vecs: (n_samples, N) float32
    n_components: (n_samples,  int32
    largest_cc_sizes: (n_samples,) int32
    pairwise_conn: (n_samples,) int64
    """
    np.random.seed(rng_seed)
    nodes = np.arange(N, dtype=np.int32)

    removal_sets_list = []
    connectivity_vecs_list = []
    n_components_list = []
    largest_cc_sizes_list = []
    pairwise_conn_list = []

    # Baseline stats: connectivity with no removals
    print("  Computing baseline connectivity ...")
    baseline_components = list(nx.connected_components(G))
    baseline_pairwise = sum(len(c) * (len(c) - 1) // 2 for c in baseline_components)
    baseline_largest = max(len(c) for c in baseline_components)

    print(
        f"  Baseline (no removals): {len(baseline_components)} components, "
        f"largest CC={baseline_largest}, pairwise={baseline_pairwise}"
    )

    time_o = time.time()
    for i in range(n_samples):
        if (i + 1) % 10 == 0:
            elapsed = time.time() - time_o
            eta = elapsed / (i + 1) * (n_samples - i - 1)

            print(
                f"  [CND] sample {i+1}/{n_samples} | "
                f"elapsed={elapsed:.0f}s | ETA={eta:.0f}s"
            )

        removal = sorted(np.random.choice(nodes, size=k, replace=False).tolist())
        conn_vec, n_comp, lcc_size, pw_conn = simulate_removal(removal, G, N)

        removal_sets_list.append(removal)
        connectivity_vecs_list.append(conn_vec)
        n_components_list.append(n_comp)
        largest_cc_sizes_list.append(lcc_size)
        pairwise_conn_list.append(pw_conn)

    removal_sets = np.array(removal_sets_list, dtype=np.int32)
    connectivity_vecs = np.array(connectivity_vecs_list, dtype=np.float32)
    n_components = np.array(n_components_list, dtype=np.int32)
    largest_cc_sizes = np.array(largest_cc_sizes_list, dtype=np.int32)
    pairwise_conn = np.array(pairwise_conn_list, dtype=np.int64)

    return (
        removal_sets,
        connectivity_vecs,
        n_components,
        largest_cc_sizes,
        pairwise_conn,
    )


def save_samples(
    out_dir: Path,
    removal_sets: np.ndarray,
    connectivity_vecs: np.ndarray,
    n_components: np.ndarray,
    largest_cc_sizes: np.ndarray,
    pairwise_conn: np.ndarray,
    N: int,
) -> Path:
    """Save CND samples to npz."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "samples_cnd.npz"
    np.savez_compressed(
        out_path,
        removal_sets=removal_sets,
        connectivity_vecs=connectivity_vecs,
        n_components=n_components,
        largest_cc_sizes=largest_cc_sizes,
        pairwise_conn=pairwise_conn,
        n_nodes=np.array([N], dtype=np.int32),
    )

    print(
        f"[✓] Saved CND samples -> {out_path}  "
        f"({out_path.stat().st_size / 1e6:.1f} MB)"
    )

    return out_path


def save_graph(
    out_dir: Path,
    edge_index: np.ndarray,
    ic_probs: np.ndarray,
    lt_weights: np.ndarray,
    node_feats: np.ndarray,
    node_labels: np.ndarray,
) -> Path:
    """Save graph_data.npz (shared format with training pipeline)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "graph_data.npz"
    np.savez_compressed(
        out_path,
        edge_index=edge_index,
        ic_probs=ic_probs,
        lt_weights=lt_weights,
        node_feats=node_feats,
        node_labels=node_labels,
    )
    print(f"[✓] Saved graph data -> {out_path}")

    return out_path


def load_graph(data_dir: Path):
    """Load graph data. Returns dict with numpy arrays."""
    d = np.load(data_dir / "graph_data.npz")

    return {k: d[k] for k in d.files}


def load_samples(
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load CND samples from npz.

    Returns
    -------
    removal_sets: (S, k) int32
    connectivity_vecs: (S, N) float32
    n_components: (S,) int32
    largest_cc_sizes: (S,) int32
    pairwise_conn: (S,) int64
    """
    d = np.load(data_dir / "samples_cnd.npz")

    return (
        d["removal_sets"],
        d["connectivity_vecs"],
        d["n_components"],
        d["largest_cc_sizes"],
        d["pairwise_conn"],
    )


def print_dataset_stats(
    removal_sets: np.ndarray,
    connectivity_vecs: np.ndarray,
    n_components: np.ndarray,
    largest_cc_sizes: np.ndarray,
    pairwise_conn: np.ndarray,
) -> None:
    S, k = removal_sets.shape
    N = connectivity_vecs.shape[1]
    remaining = N - k

    avg_conn_density = connectivity_vecs.sum(axis=1).mean() / remaining

    print(f"\n{'='*50}")
    print("Dataset stats [CND — Digg]")
    print(f"{'='*50}")
    print(f"  Samples           : {S}")
    print(f"  Removal size (k)  : {k}")
    print(f"  Nodes (N)         : {N}")
    print(f"  Remaining nodes   : {remaining}")
    print(
        f"  Components        : {n_components.mean():.1f} +/- {n_components.std():.1f}"
    )
    print(f"                      range [{n_components.min()}, {n_components.max()}]")
    print(
        f"  Largest CC size   : {largest_cc_sizes.mean():.1f} +/- {largest_cc_sizes.std():.1f}"
    )
    print(
        f"                      range [{largest_cc_sizes.min()}, {largest_cc_sizes.max()}]"
    )
    print(
        f"  Pairwise conn     : {pairwise_conn.mean():.0f} +/- {pairwise_conn.std():.0f}"
    )
    print(f"                      range [{pairwise_conn.min()}, {pairwise_conn.max()}]")
    print(f"  Avg largest CC %  : {avg_conn_density*100:.1f}% of remaining nodes")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate CND data on Digg social network")
    p.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of (removal_set, connectivity) samples (default: 1000)",
    )
    p.add_argument(
        "--k",
        type=int,
        default=10,
        help="Removal set size / budget (default: 10)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).parent / "digg"),
        help="Output directory (default: Data/digg)",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    # Download and load Digg graph
    raw_path = download_digg()
    adj, node_feats, node_labels, N = load_digg(raw_path)

    # Save graph_data.npz
    graph_path = out_dir / "graph_data.npz"
    if graph_path.exists():
        print(f"[✓] graph_data.npz already exists at {graph_path}, skipping")
    else:
        edge_index, ic_probs, lt_weights = build_graph_data(adj)
        save_graph(out_dir, edge_index, ic_probs, lt_weights, node_feats, node_labels)

    # Build NetworkX graph for connectivity computation
    # adj is already undirected (symmetric) from load_digg
    print("[→] Building NetworkX graph ...")
    G = nx.from_scipy_sparse_array(adj)
    print(
        f"[✓] NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )

    # Generate CND samples
    print(f"\n[→] Generating {args.samples} CND samples (k={args.k}) ...")

    removal_sets, connectivity_vecs, n_components, largest_cc_sizes, pairwise_conn = (
        generate_samples(
            N=N,
            G=G,
            k=args.k,
            n_samples=args.samples,
            rng_seed=args.seed,
        )
    )

    # Save samples and print stats
    save_samples(
        out_dir,
        removal_sets,
        connectivity_vecs,
        n_components,
        largest_cc_sizes,
        pairwise_conn,
        N,
    )
    print_dataset_stats(
        removal_sets,
        connectivity_vecs,
        n_components,
        largest_cc_sizes,
        pairwise_conn,
    )

    # Save metadata
    n_edges_undirected = adj.nnz // 2
    metadata = {
        "task": "CND",
        "dataset": "digg",
        "n_nodes": N,
        "n_edges_undirected": n_edges_undirected,
        "k": args.k,
        "n_samples": args.samples,
        "seed": args.seed,
    }

    meta_path = out_dir / "metadata_cnd.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[✓] Metadata saved -> {meta_path}")
    print("\n[✓] Done! Load your data with:")
    print("    from generate_cnd_data import load_samples")
    print(
        f"    removal_sets, conn_vecs, n_comp, lcc, pw = load_samples(Path('{out_dir}'))"
    )


if __name__ == "__main__":
    main()
