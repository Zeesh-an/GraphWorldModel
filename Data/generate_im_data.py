"""
Influence Maximization Data Generation
=======================================

Generates training/eval data for IM models on a chosen graph dataset.

What this produces
------------------
For each diffusion model (IC, LT):
    - samples_<model>.npz: (seed_sets, spreads, cascade_traces) arrays
    - graph_data.npz: adjacency, edge_probs, node_features, node_labels
    - metadata.json: graph stats and generation config

Data format
-----------
graph_data.npz:
    edge_index: (2, E) int32 — COO format [src, dst]
    edge_probs: (E,) float32 — IC propagation probability per edge
    lt_weights: (E,) float32 — LT edge weights (row-normalized)
    node_feats: (N, F) float32 — original bag-of-words features
    node_labels: (N,) int32 — class labels (7 classes)

samples_ic.npz / samples_lt.npz:
    seed_sets: (S, k) int32  — S samples, each with k seed nodes
    spreads: (S,)  float32 — mean spread over R MC runs
    spread_std: (S,)  float32 — std over R MC runs
    cascades_src: (S*T_max, N) bool — activated nodes at each timestep
                        stored as flat array, reshape with cascade_lengths
    cascade_lengths: (S,) int32 — actual number of timesteps per sample

Usage
-----
    python generate_im_data.py [--samples 1000] [--k 10] [--mc-runs 100]

Then load with:
    data = np.load('graph_data.npz')
    ic = np.load('samples_ic.npz')
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
from graph_utils import build_edge_index, build_adjacency_lists, save_graph
from diffusion import simulate_IC, simulate_LT

DATASET_CHOICES = ["cora_ml", "digg", "twitter"]


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
    mc_runs: int = 100,
    max_steps: int = 50,
    rng_seed: int = 42,
):
    """
    Generate (seed_set, spread) samples by:
        1. Sampling random seed sets of size k
        2. Running MC simulations to estimate spread
        3. Recording one representative cascade trace per sample

    Returns
    -------
    seed_sets: (n_samples, k) int32
    spreads: (n_samples,) float32 mean spread
    spread_stds: (n_samples,) float32  std of spread
    cascade_arrays: list of np.ndarray each (T_i, N) bool
                    T_i = number of timesteps in cascade i
    """
    np.random.seed(rng_seed)
    nodes = np.arange(N, dtype=np.int32)

    seed_sets = []
    spreads = []
    spread_stds = []
    cascade_list = []  # one representative trace per sample (first MC run)

    sim_fn = simulate_IC if model == "IC" else simulate_LT

    time_0 = time.time()
    for i in range(n_samples):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - time_0
            eta = elapsed / (i + 1) * (n_samples - i - 1)

            print(
                f"  [IM-{model}] sample {i+1}/{n_samples} | "
                f"elapsed={elapsed:.0f}s | ETA={eta:.0f}s"
            )

        # Sample k random nodes as the seed set
        seed = sorted(np.random.choice(nodes, size=k, replace=False).tolist())

        spread_vals = []
        rep_trace = None

        kwargs = (
            dict(out_adj=out_adj, N=N) if model == "IC" else dict(in_adj=in_adj, N=N)
        )

        # Run mc_runs Monte Carlo simulations for spread estimate
        for run in range(mc_runs):
            trace, spread = sim_fn(seed, **kwargs)
            spread_vals.append(spread)

            if run == 0:
                rep_trace = trace  # Save the first run as representative cascade

        mean_sp = float(np.mean(spread_vals))
        std_sp = float(np.std(spread_vals))

        # Convert the representative trace to a dense, boolean timestep matrix: (T, N)
        T = len(rep_trace)
        cascade_mat = np.zeros((T, N), dtype=bool)

        for t, wave in enumerate(rep_trace):
            for v in wave:
                cascade_mat[t, v] = True

        seed_sets.append(seed)
        spreads.append(mean_sp)
        spread_stds.append(std_sp)
        cascade_list.append(cascade_mat)

    seed_sets = np.array(seed_sets, dtype=np.int32)
    spreads = np.array(spreads, dtype=np.float32)
    spread_stds = np.array(spread_stds, dtype=np.float32)

    return seed_sets, spreads, spread_stds, cascade_list


def save_samples(
    out_dir: Path,
    model: str,
    seed_sets: np.ndarray,
    spreads: np.ndarray,
    spread_stds: np.ndarray,
    cascade_list: list,
):
    """
    Save IM samples to npz. Cascades are saved as a ragged structure:
        cascade_data: concatenated (T_i * N,) booleans (flattened rows)
        cascade_offsets: (S + 1,) int — start index in cascade_data for sample i
        cascade_lengths: (S,) int — number of timesteps T_i for sample i
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pack ragged cascade array
    cascade_data = []
    cascade_offsets = [0]
    cascade_lengths = []
    N = cascade_list[0].shape[1]

    for mat in cascade_list:
        T = mat.shape[0]
        cascade_data.append(mat.flatten())
        cascade_offsets.append(cascade_offsets[-1] + T * N)
        cascade_lengths.append(T)

    cascade_data = np.concatenate(cascade_data)
    cascade_offsets = np.array(cascade_offsets, dtype=np.int64)
    cascade_lengths = np.array(cascade_lengths, dtype=np.int32)

    out_path = out_dir / f"samples_im_{model.lower()}.npz"
    np.savez_compressed(
        out_path,
        seed_sets=seed_sets,
        spreads=spreads,
        spread_stds=spread_stds,
        cascade_data=cascade_data,
        cascade_offsets=cascade_offsets,
        cascade_lengths=cascade_lengths,
        n_nodes=np.array([N], dtype=np.int32),
    )

    print(
        f"[✓] Saved {model} samples → {out_path}  "
        f"({out_path.stat().st_size / 1e6:.1f} MB)"
    )

    return out_path


def load_samples(data_dir: Path, model: str = "IC"):
    """
    Load IM samples from npz and reconstruct cascade traces.

    Returns
    -------
    seed_sets: (S, k) int32
    spreads: (S,) float32
    spread_stds: (S,) float32
    cascades: list of (T_i, N) bool arrays — one per sample
    """
    data = np.load(data_dir / f"samples_im_{model.lower()}.npz")
    N = int(data["n_nodes"][0])
    cascade_data = data["cascade_data"]
    cascade_offsets = data["cascade_offsets"]
    cascade_lengths = data["cascade_lengths"]

    cascades = []
    for i, T in enumerate(cascade_lengths):
        start = cascade_offsets[i]
        end = cascade_offsets[i + 1]
        mat = cascade_data[start:end].reshape(T, N)
        cascades.append(mat)

    return data["seed_sets"], data["spreads"], data["spread_stds"], cascades


def print_dataset_stats(seed_sets, spreads, spread_stds, cascades, model):
    S, k = seed_sets.shape
    N = cascades[0].shape[1]
    avg_T = np.mean([c.shape[0] for c in cascades])

    print(f"\n{'='*50}")
    print(f"Dataset stats [{model}]")
    print(f"{'='*50}")
    print(f"  Samples        : {S}")
    print(f"  Seed size (k)  : {k}")
    print(f"  Nodes (N)      : {N}")
    print(f"  Spread mean    : {spreads.mean():.1f} ± {spreads.std():.1f}")
    print(f"  Spread range   : [{spreads.min():.0f}, {spreads.max():.0f}]")
    print(f"  Avg cascade len: {avg_T:.1f} timesteps")
    print(f"  Avg density    : {spreads.mean()/N*100:.1f}% of graph reached")


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
    if dataset == "cora_ml":
        raw_path = download_cora_ml()
        return load_cora_ml(raw_path)
    elif dataset == "digg":
        raw_path = download_digg()
        return load_digg(raw_path)
    elif dataset == "twitter":
        raw_path = download_twitter()
        return load_twitter(raw_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {DATASET_CHOICES}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate IM data")
    p.add_argument(
        "-d",
        "--dataset",
        default="cora_ml",
        choices=DATASET_CHOICES,
        help="Dataset to generate IM data for (default: cora_ml)",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of (seed_set, spread) samples per model (default: 1000)",
    )
    p.add_argument("--k", type=int, default=10, help="Seed set size (default: 10)")
    p.add_argument(
        "--mc-runs",
        type=int,
        default=100,
        help="Monte Carlo runs per sample for spread estimation (default: 100)",
    )
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

    # Default out-dir based on dataset
    if args.out_dir is None:
        args.out_dir = str(Path(__file__).parent / args.dataset)

    return args


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    # Download and load graph data
    adj, node_feats, node_labels, N = load_dataset(args.dataset)

    # Build propagation structures with edge index, propagation probabilities, and adjacency lists
    edge_index, ic_probs, lt_weights, out_adj, in_adj, N = build_graph(adj)

    # Save the graph data
    save_graph(out_dir, edge_index, ic_probs, lt_weights, node_feats, node_labels)

    # Generate samples for each diffusion model (IC, LT)
    metadata = {
        "task": "IM",
        "dataset": args.dataset,
        "n_nodes": N,
        "n_edges": int(edge_index.shape[1]),
        "k": args.k,
        "n_samples": args.samples,
        "mc_runs": args.mc_runs,
        "max_steps": args.max_steps,
        "models": args.models,
        "seed": args.seed,
    }

    for model in args.models:
        print(
            f"\n[→] Generating {args.samples} samples under {model} model "
            f"(k={args.k}, mc_runs={args.mc_runs}) ..."
        )

        seed_sets, spreads, spread_stds, cascades = generate_samples(
            N=N,
            out_adj=out_adj,
            in_adj=in_adj,
            model=model,
            k=args.k,
            n_samples=args.samples,
            mc_runs=args.mc_runs,
            max_steps=args.max_steps,
            rng_seed=args.seed,
        )

        save_samples(out_dir, model, seed_sets, spreads, spread_stds, cascades)
        print_dataset_stats(seed_sets, spreads, spread_stds, cascades, model)

        metadata[f"spread_mean_{model}"] = float(spreads.mean())
        metadata[f"spread_std_{model}"] = float(spreads.std())

    # Save metadata as JSON
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[✓] Metadata saved → {meta_path}")
    print("\n[✓] Done! Load your data with:")
    print(f"    graph = load_graph(Path('{out_dir}'))")
    print(
        f"    seeds, spreads, stds, cascades = load_samples(Path('{out_dir}'), model='IC')"
    )


if __name__ == "__main__":
    main()
