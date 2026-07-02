"""
Critical Node Detection Data Generation

Generates training/eval data for CND models on a chosen graph dataset.

Problem: Given graph G and budget k, find k nodes whose removal
causes maximum damage to network connectivity.

Forward process (deterministic):
    removal_set -> remove nodes + edges -> measure residual connectivity

Inverse problem:
    desired fragmentation -> infer optimal removal set

What this produces
------------------
    - samples_cnd_k{k}.npz: (removal_sets, connectivity_vecs, damage metrics)
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

samples_cnd_k{k}.npz:
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
    (k = 10, 20, 50)
    python Data/generate_cnd_data.py --dataset cora_ml --samples 1000 --k 10

    (k = 50, 100)
    python Data/generate_cnd_data.py --dataset digg --samples 5000 --k 50

    (k = 50, 100)
    python Data/generate_cnd_data.py --dataset twitter --samples 5000 --k 50

    (k = 5, 10, 15)
    python Data/generate_cnd_data.py --dataset jazz --samples 500 --k 5

    (k = 10, 20, 30)
    python Data/generate_cnd_data.py --dataset netscience --samples 1000 --k 10

    (k = 10, 20, 50)
    python Data/generate_cnd_data.py --dataset power_grid --samples 1000 --k 10

    (k = 20, 50)
    python Data/generate_cnd_data.py --dataset nethept --samples 5000 --k 20
"""

import json
from tqdm.auto import tqdm
import argparse
import numpy as np
import networkx as nx
from pathlib import Path

from datasets.cora_ml import download_cora_ml, load_cora_ml
from datasets.digg import download_digg, load_digg
from datasets.twitter import download_twitter, load_twitter
from datasets.jazz import download_jazz, load_jazz
from datasets.netscience import download_netscience, load_netscience
from datasets.power_grid import download_power_grid, load_power_grid
from datasets.nethept import download_nethept, load_nethept
from graph_utils import build_edge_index, save_graph
from connectivity import simulate_removal

DATASET_CHOICES = [
    "cora_ml",
    "digg",
    "twitter",
    "jazz",
    "netscience",
    "power_grid",
    "nethept",
]


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

    for i in tqdm(range(n_samples), desc=f"CND samples (k={k})"):
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
    k = removal_sets.shape[1]
    out_path = out_dir / f"samples_cnd_k{k}.npz"
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
        f"[✓] Saved CND samples -> {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)"
    )

    return out_path


def load_samples(
    data_dir: Path,
    k: int = 10,
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
    data = np.load(data_dir / f"samples_cnd_k{k}.npz")

    return (
        data["removal_sets"],
        data["connectivity_vecs"],
        data["n_components"],
        data["largest_cc_sizes"],
        data["pairwise_conn"],
    )


def print_dataset_stats(
    removal_sets: np.ndarray,
    connectivity_vecs: np.ndarray,
    n_components: np.ndarray,
    largest_cc_sizes: np.ndarray,
    pairwise_conn: np.ndarray,
    dataset: str = "digg",
) -> None:
    S, k = removal_sets.shape
    N = connectivity_vecs.shape[1]
    remaining = N - k

    avg_conn_density = connectivity_vecs.sum(axis=1).mean() / remaining

    print(f"\n{'=' * 50}")
    print(f"Dataset stats [CND — {dataset}]")
    print(f"{'=' * 50}")
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
    print(f"  Avg largest CC %  : {avg_conn_density * 100:.1f}% of remaining nodes")


def load_dataset(dataset: str) -> tuple:
    """
    Download and load the specified dataset.

    Returns
    -------
    adj: scipy.sparse.csr_matrix (N, N)
    node_feats: np.ndarray (N, F) float32
    node_labels: np.ndarray (N,) int32
    N: int
    """
    loaders = {
        "cora_ml": (download_cora_ml, load_cora_ml),
        "digg": (download_digg, load_digg),
        "twitter": (download_twitter, load_twitter),
        "jazz": (download_jazz, load_jazz),
        "netscience": (download_netscience, load_netscience),
        "power_grid": (download_power_grid, load_power_grid),
        "nethept": (download_nethept, load_nethept),
    }

    if dataset not in loaders:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {DATASET_CHOICES}")

    download_fn, load_fn = loaders[dataset]
    raw_path = download_fn()
    return load_fn(raw_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate CND data")
    p.add_argument(
        "-d",
        "--dataset",
        default="digg",
        choices=DATASET_CHOICES,
        help="Dataset to generate CND data for (default: digg)",
    )
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
    p.add_argument(
        "--k-pct",
        type=float,
        default=None,
        help="Removal set size as percentage of N (e.g., 5 = 5%%). Overrides --k.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: Data/<dataset>)",
    )

    args = p.parse_args()

    # Default out-dir based on dataset
    if args.out_dir is None:
        args.out_dir = str(Path(__file__).parent / args.dataset)

    return args


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    # Download and load graph
    adj, node_feats, node_labels, N = load_dataset(args.dataset)

    # Resolve --k-pct to a concrete k
    if args.k_pct is not None:
        args.k = max(1, round(N * args.k_pct / 100))
        print(f"[config] --k-pct={args.k_pct}% of N={N} → k={args.k}")

    # Save graph_data.npz
    graph_path = out_dir / "graph_data.npz"
    if graph_path.exists():
        print(f"[✓] graph_data.npz already exists at {graph_path}, skipping")
    else:
        edge_index, ic_probs, lt_weights = build_edge_index(adj)
        save_graph(out_dir, edge_index, ic_probs, lt_weights, node_feats, node_labels)

    # Build NetworkX graph for connectivity computation
    print("[→] Building NetworkX graph ...")

    # Converts every directed edge (u → v)into an undirected one (u - v)
    G = nx.from_scipy_sparse_array(adj)
    print(
        f"[✓] NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )

    # Generate CND samples
    metadata = {
        "task": "CND",
        "dataset": args.dataset,
        "n_nodes": N,
        "n_edges": adj.nnz,
        "k": args.k,
        "n_samples": args.samples,
        "seed": args.seed,
    }

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
        dataset=args.dataset,
    )

    metadata["n_components_mean"] = float(n_components.mean())
    metadata["largest_cc_sizes_mean"] = float(largest_cc_sizes.mean())

    # Save metadata
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
