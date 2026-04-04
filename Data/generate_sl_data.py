"""
Source Localization Data Generation
====================================

Generates training/eval data for Source Localization (SL) models on a chosen graph dataset.

Problem: Given an observed partial infection snapshot, infer which nodes
were the original sources (seeds) that started the cascade.

Forward process (stochastic):
    seed_set -> simulate diffusion -> take partial snapshot at random time t

Inverse problem:
    observed snapshot -> infer original seed set

What this produces
------------------
For each diffusion model (IC, LT):
    - samples_sl_<model>.npz: (seed_sets, snapshots, observation metadata)
    - graph_data.npz: adjacency, edge_probs, node_features, node_labels (shared with IM)

Data format
-----------
samples_sl_ic.npz / samples_sl_lt.npz:
    seed_sets: (S, k) int32 — S samples, each with k source nodes (ground truth)
    snapshots: (S, N) float32 — binary partial observation at random time t
    observe_times: (S,) int32 — timestep at which the snapshot was taken
    cascade_lengths: (S,) int32 — total cascade depth (how long diffusion ran)
    spreads: (S,) int32 — total number of activated nodes at observation time
    n_nodes: (1,) int32 — total nodes in graph

Training tensor (built by loader): shape (S, N, 2)
    [:, :, 0] = binary seed vector (ground truth sources — the label)
    [:, :, 1] = binary snapshot vector (partial observation at time t — the input to invert)

Usage
-----
    python generate_sl_data.py [--dataset cora_ml] [--samples 1000] [--k 10] [--mc-runs 100]

Then train with:
    python World_Model/train.py --task SL -d cora_ml -dm IC --npz-dir Data/cora_ml
"""

import json
import time
import argparse
import numpy as np
import scipy.sparse as sp
from pathlib import Path

from datasets.cora_ml import download_cora_ml, load_cora_ml
from datasets.digg import download_digg, load_digg
from datasets.twitter import download_twitter, load_twitter
from datasets.jazz import download_jazz, load_jazz
from datasets.netscience import download_netscience, load_netscience
from datasets.power_grid import download_power_grid, load_power_grid
from datasets.nethept import download_nethept, load_nethept
from graph_utils import build_edge_index, build_adjacency_lists, save_graph
from diffusion import simulate_IC, simulate_LT

DATASET_CHOICES = [
    "cora_ml",
    "digg",
    "twitter",
    "jazz",
    "netscience",
    "power_grid",
    "nethept",
]


def build_graph(adj: sp.csr_matrix):
    """
    Convert the sparse adjacency matrix to COO edge_index and compute edge probabilities.

    Independent Cascade (IC) probability p(u → v) = 1 / in_degree(v) (weighted cascade model)
    Linear Threshold (LT) weight w(u → v) = 1 / in_degree(v) (same, but semantics differ)
    """
    edge_index, ic_probs, lt_weights = build_edge_index(adj)
    src, dst = edge_index[0], edge_index[1]
    N = adj.shape[0]

    # Build adjacency lists for fast simulation
    # adj_list[v] = list of (neighbor_u, prob_u_v) incoming edges
    out_adj, in_adj = build_adjacency_lists(src, dst, ic_probs, lt_weights)

    print(f"[✓] Graph built: {N} nodes, {len(src)} edges")
    print(
        f"    IC probs  — mean={ic_probs.mean():.4f}, "
        f"max={ic_probs.max():.4f}, min={ic_probs.min():.4f}"
    )

    return edge_index, ic_probs, lt_weights, dict(out_adj), dict(in_adj), N


def generate_samples(
    N: int,
    out_adj: dict,
    in_adj: dict,
    model: str = "IC",
    k: int = 10,
    n_samples: int = 1000,
    max_steps: int = 50,
    rng_seed: int = 42,
):
    """
    Generate (seed_set, partial_snapshot) samples for source localization.

    For each sample:
        1. Sample a random seed set of size k
        2. Run ONE diffusion simulation (each cascade is a valid observation)
        3. Pick a random observation time t in [1, T] (must see at least one propagation step)
        4. Record the cumulative activation snapshot up to time t

    Returns
    -------
    seed_sets: (n_samples, k) int32
    snapshots: (n_samples, N) float32 — binary partial observation
    observe_times: (n_samples,) int32
    cascade_lengths: (n_samples,) int32
    spreads: (n_samples,) int32 — number of activated nodes at observation time
    """
    np.random.seed(rng_seed)
    nodes = np.arange(N, dtype=np.int32)

    seed_sets_list = []
    snapshots_list = []
    observe_times_list = []
    cascade_lengths_list = []
    spreads_list = []

    sim_fn = simulate_IC if model == "IC" else simulate_LT

    time_0 = time.time()
    for i in range(n_samples):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - time_0
            eta = elapsed / (i + 1) * (n_samples - i - 1)
            print(
                f"  [SL-{model}] sample {i+1}/{n_samples} | "
                f"elapsed={elapsed:.0f}s | ETA={eta:.0f}s"
            )

        seed = sorted(np.random.choice(nodes, size=k, replace=False).tolist())

        kwargs = (
            dict(out_adj=out_adj, N=N) if model == "IC" else dict(in_adj=in_adj, N=N)
        )
        trace, _ = sim_fn(seed, **kwargs, max_steps=max_steps)

        # trace is a list of sets: trace[0] = seeds, trace[1] = first wave, ...
        T = len(trace)

        # Pick a random observation time — at least t = 1 so we see propagation beyond seeds
        # If the cascade didn't propagate (T = 1, only seeds), observe at t = 1 (just the seeds)
        min_t = min(2, T)
        observe_t = np.random.randint(min_t, T + 1)

        # Build cumulative snapshot up to time t: union of trace[0..observe_t)
        snapshot = np.zeros(N, dtype=np.float32)
        for t_step in range(observe_t):
            for v in trace[t_step]:
                snapshot[v] = 1.0

        spread = int(snapshot.sum())

        seed_sets_list.append(seed)
        snapshots_list.append(snapshot)
        observe_times_list.append(observe_t)
        cascade_lengths_list.append(T)
        spreads_list.append(spread)

    seed_sets = np.array(seed_sets_list, dtype=np.int32)
    snapshots = np.array(snapshots_list, dtype=np.float32)
    observe_times = np.array(observe_times_list, dtype=np.int32)
    cascade_lengths = np.array(cascade_lengths_list, dtype=np.int32)
    spreads = np.array(spreads_list, dtype=np.int32)

    return seed_sets, snapshots, observe_times, cascade_lengths, spreads


def save_samples(
    out_dir: Path,
    model: str,
    seed_sets: np.ndarray,
    snapshots: np.ndarray,
    observe_times: np.ndarray,
    cascade_lengths: np.ndarray,
    spreads: np.ndarray,
    N: int,
) -> Path:
    """Save SL samples to npz."""
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"samples_sl_{model.lower()}.npz"
    np.savez_compressed(
        out_path,
        seed_sets=seed_sets,
        snapshots=snapshots,
        observe_times=observe_times,
        cascade_lengths=cascade_lengths,
        spreads=spreads,
        n_nodes=np.array([N], dtype=np.int32),
    )

    print(
        f"[✓] Saved SL-{model} samples -> {out_path}  "
        f"({out_path.stat().st_size / 1e6:.1f} MB)"
    )

    return out_path


def load_samples(
    data_dir: Path,
    model: str = "IC",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load SL samples from npz.

    Returns
    -------
    seed_sets: (S, k) int32
    snapshots: (S, N) float32
    observe_times: (S,) int32
    cascade_lengths: (S,) int32
    spreads: (S,) int32
    """
    data = np.load(data_dir / f"samples_sl_{model.lower()}.npz")

    return (
        data["seed_sets"],
        data["snapshots"],
        data["observe_times"],
        data["cascade_lengths"],
        data["spreads"],
    )


def print_dataset_stats(
    seed_sets: np.ndarray,
    snapshots: np.ndarray,
    observe_times: np.ndarray,
    cascade_lengths: np.ndarray,
    spreads: np.ndarray,
    model: str,
) -> None:
    S, k = seed_sets.shape
    N = snapshots.shape[1]

    print(f"\n{'='*50}")
    print(f"Dataset stats [SL — {model}]")
    print(f"{'='*50}")
    print(f"  Samples          : {S}")
    print(f"  Source size (k)  : {k}")
    print(f"  Nodes (N)        : {N}")
    print(
        f"  Observe time     : {observe_times.mean():.1f} +/- {observe_times.std():.1f}"
    )
    print(f"                     range [{observe_times.min()}, {observe_times.max()}]")
    print(
        f"  Cascade depth    : {cascade_lengths.mean():.1f} +/- {cascade_lengths.std():.1f}"
    )
    print(f"  Snapshot spread  : {spreads.mean():.1f} +/- {spreads.std():.1f}")
    print(f"                     range [{spreads.min()}, {spreads.max()}]")
    print(f"  Avg density      : {spreads.mean()/N*100:.1f}% of graph observed active")


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
    p = argparse.ArgumentParser(description="Generate Source Localization data")
    p.add_argument(
        "-d",
        "--dataset",
        default="cora_ml",
        choices=DATASET_CHOICES,
        help="Dataset to generate SL data for (default: cora_ml)",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of (source_set, snapshot) samples per model (default: 1000)",
    )
    p.add_argument("--k", type=int, default=10, help="Source set size (default: 10)")
    p.add_argument(
        "--models",
        nargs="+",
        default=["IC", "LT"],
        choices=["IC", "LT"],
        help="Diffusion models to generate data for (default: IC LT)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Max cascade propagation steps (default: 50)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: Data/<dataset>)",
    )

    args = p.parse_args()

    if args.out_dir is None:
        args.out_dir = str(Path(__file__).parent / args.dataset)

    return args


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    # Download and load graph data
    adj, node_feats, node_labels, N = load_dataset(args.dataset)

    # Build propagation structures
    edge_index, ic_probs, lt_weights, out_adj, in_adj, N = build_graph(adj)

    # Save graph data (shared with IM — will skip if already exists)
    graph_path = out_dir / "graph_data.npz"
    if graph_path.exists():
        print(f"[✓] graph_data.npz already exists at {graph_path}, skipping")
    else:
        save_graph(out_dir, edge_index, ic_probs, lt_weights, node_feats, node_labels)

    # Generate SL samples for each diffusion model
    metadata = {
        "task": "SL",
        "dataset": args.dataset,
        "n_nodes": N,
        "n_edges": int(edge_index.shape[1]),
        "k": args.k,
        "n_samples": args.samples,
        "max_steps": args.max_steps,
        "models": args.models,
        "seed": args.seed,
    }

    for model in args.models:
        print(
            f"\n[->] Generating {args.samples} SL samples under {model} model "
            f"(k={args.k}) ..."
        )

        seed_sets, snapshots, observe_times, cascade_lengths, spreads = (
            generate_samples(
                N=N,
                out_adj=out_adj,
                in_adj=in_adj,
                model=model,
                k=args.k,
                n_samples=args.samples,
                max_steps=args.max_steps,
                rng_seed=args.seed,
            )
        )

        save_samples(
            out_dir,
            model,
            seed_sets,
            snapshots,
            observe_times,
            cascade_lengths,
            spreads,
            N,
        )
        print_dataset_stats(
            seed_sets,
            snapshots,
            observe_times,
            cascade_lengths,
            spreads,
            model,
        )

        metadata[f"snapshot_spread_mean_{model}"] = float(spreads.mean())
        metadata[f"snapshot_spread_std_{model}"] = float(spreads.std())
        metadata[f"observe_time_mean_{model}"] = float(observe_times.mean())

    # Save metadata
    meta_path = out_dir / "metadata_sl.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[✓] Metadata saved -> {meta_path}")
    print("\n[✓] Done! Load your data with:")
    print("    from Data.generate_sl_data import load_samples")
    print(
        f"    seeds, snapshots, obs_t, lengths, spreads = "
        f"load_samples(Path('{out_dir}'), model='IC')"
    )


if __name__ == "__main__":
    main()
