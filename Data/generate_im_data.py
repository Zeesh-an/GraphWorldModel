"""
Influence Maximization Data Generation — Cora-ML
=================================================
Generates training/eval data for IM models on the Cora-ML graph.

What this produces
------------------
For each diffusion model (IC, LT):
  - samples_<model>.npz  : (seed_sets, spreads, cascade_traces) arrays
  - graph_data.npz       : adjacency, edge_probs, node_features, node_labels
  - metadata.json        : graph stats and generation config

Data format
-----------
graph_data.npz:
  edge_index  : (2, E) int32   — COO format [src, dst]
  edge_probs  : (E,)  float32  — IC propagation probability per edge
  lt_weights  : (E,)  float32  — LT edge weights (row-normalized)
  node_feats  : (N, F) float32 — original bag-of-words features
  node_labels : (N,)  int32    — class labels (7 classes)

samples_ic.npz / samples_lt.npz:
  seed_sets       : (S, k) int32   — S samples, each with k seed nodes
  spreads         : (S,)   float32 — mean spread over R MC runs
  spread_std      : (S,)   float32 — std over R MC runs
  cascades_src    : (S*T_max, N) bool — activated nodes at each timestep
                    stored as flat array, reshape with cascade_lengths
  cascade_lengths : (S,)   int32   — actual number of timesteps per sample

Usage
-----
  python generate_im_data.py [--samples 1000] [--k 10] [--mc-runs 100]

Then load with:
  data = np.load('graph_data.npz')
  ic   = np.load('samples_ic.npz')
"""

import os
import json
import time
import argparse
import urllib.request
import numpy as np
import networkx as nx
import scipy.sparse as sp
from pathlib import Path
from collections import defaultdict


# ─────────────────────────────────────────────
# 1.  Download + Load Cora-ML
# ─────────────────────────────────────────────

CORA_ML_URL = (
    "https://github.com/abojchevski/graph2gauss/raw/master/data/cora_ml.npz"
)
DATA_DIR = Path(__file__).parent / "cora_ml"


def download_cora_ml() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / "cora_ml.npz"
    if dest.exists():
        print(f"[✓] Cora-ML already downloaded at {dest}")
        return dest
    print(f"[↓] Downloading Cora-ML from {CORA_ML_URL} ...")
    urllib.request.urlretrieve(CORA_ML_URL, dest)
    print(f"[✓] Saved to {dest}")
    return dest


def load_cora_ml(path: Path):
    """
    Returns
    -------
    adj       : scipy.sparse.csr_matrix  (N, N) binary directed adjacency
    features  : np.ndarray               (N, F) float32 bag-of-words
    labels    : np.ndarray               (N,)   int32   class labels
    """
    raw = np.load(path, allow_pickle=True)

    # The npz stores the adjacency as a sparse matrix in COO format
    adj = sp.csr_matrix(
        (raw["adj_data"], raw["adj_indices"], raw["adj_indptr"]),
        shape=raw["adj_shape"],
    ).astype(np.float32)

    features = sp.csr_matrix(
        (raw["attr_data"], raw["attr_indices"], raw["attr_indptr"]),
        shape=raw["attr_shape"],
    ).toarray().astype(np.float32)

    labels = raw["labels"].astype(np.int32)

    print(f"[✓] Cora-ML loaded: {adj.shape[0]} nodes, "
          f"{adj.nnz} edges, {features.shape[1]} features, "
          f"{len(np.unique(labels))} classes")
    return adj, features, labels


# ─────────────────────────────────────────────
# 2.  Graph Preprocessing
# ─────────────────────────────────────────────

def build_graph(adj: sp.csr_matrix):
    """
    Convert adjacency to COO edge_index and compute edge probabilities.

    IC probability  p(u→v) = 1 / in_degree(v)   [weighted cascade model]
    LT weight       w(u→v) = 1 / in_degree(v)   [same, but semantics differ]
    """
    adj_coo = adj.tocoo()
    src = adj_coo.row.astype(np.int32)
    dst = adj_coo.col.astype(np.int32)

    N = adj.shape[0]
    in_deg = np.array(adj.sum(axis=0)).flatten()  # (N,) in-degree
    in_deg = np.where(in_deg == 0, 1, in_deg)     # avoid divide-by-zero

    # IC: each edge gets probability = 1/in_degree(dst)
    ic_probs = (1.0 / in_deg[dst]).astype(np.float32)
    ic_probs = np.clip(ic_probs, 0.001, 0.5)  # cap for realism

    # LT: weights must sum to ≤ 1 per node (already true with 1/in_deg)
    lt_weights = ic_probs.copy()

    edge_index = np.stack([src, dst], axis=0)  # (2, E)

    # Build adjacency list for fast simulation
    # adj_list[v] = list of (neighbor_u, prob_u_v) incoming edges
    in_adj  = defaultdict(list)   # in_adj[v]  = [(u, p), ...]  used by LT
    out_adj = defaultdict(list)   # out_adj[u] = [(v, p), ...]  used by IC

    for i, (u, v) in enumerate(zip(src, dst)):
        out_adj[int(u)].append((int(v), float(ic_probs[i])))
        in_adj[int(v)].append((int(u), float(lt_weights[i])))

    print(f"[✓] Graph built: {N} nodes, {len(src)} edges")
    print(f"    IC probs  — mean={ic_probs.mean():.4f}, "
          f"max={ic_probs.max():.4f}, min={ic_probs.min():.4f}")

    return edge_index, ic_probs, lt_weights, dict(out_adj), dict(in_adj), N


# ─────────────────────────────────────────────
# 3.  Diffusion Simulations
# ─────────────────────────────────────────────

def simulate_IC(seed_set: list[int], out_adj: dict, N: int, max_steps: int = 50):
    """
    Independent Cascade simulation.

    At each step, every newly activated node tries to activate each
    inactive out-neighbor independently with probability p(u→v).

    Returns
    -------
    cascade_trace : list of sets  — nodes newly activated at each timestep
                    cascade_trace[0] == seed_set (t=0)
    final_spread  : int           — total number of activated nodes
    """
    active = set(seed_set)
    newly_active = set(seed_set)
    cascade_trace = [set(seed_set)]

    for _ in range(max_steps):
        next_wave = set()
        for u in newly_active:
            for v, p in out_adj.get(u, []):
                if v not in active and np.random.random() < p:
                    next_wave.add(v)
        if not next_wave:
            break
        active |= next_wave
        newly_active = next_wave
        cascade_trace.append(next_wave)

    return cascade_trace, len(active)


def simulate_LT(seed_set: list[int], in_adj: dict, N: int, max_steps: int = 50):
    """
    Linear Threshold simulation.

    Each node v has threshold θ_v ~ Uniform[0, 1].
    v activates when Σ_{active u ∈ N_in(v)} w(u,v) ≥ θ_v.

    Returns
    -------
    cascade_trace : list of sets
    final_spread  : int
    """
    thresholds = np.random.uniform(0, 1, N)
    active = set(seed_set)
    newly_active = set(seed_set)
    cascade_trace = [set(seed_set)]

    # Track accumulated influence per node
    influence_sum = np.zeros(N, dtype=np.float32)
    for u in seed_set:
        for v, w in in_adj.get(u, []):
            if v not in active:
                influence_sum[v] += w

    for _ in range(max_steps):
        next_wave = set()
        for v in range(N):
            if v not in active and influence_sum[v] >= thresholds[v]:
                next_wave.add(v)
        if not next_wave:
            break
        active |= next_wave
        newly_active = next_wave
        cascade_trace.append(next_wave)

        # Update influence sums for newly activated nodes
        for u in newly_active:
            for v, w in in_adj.get(u, []):
                if v not in active:
                    influence_sum[v] += w

    return cascade_trace, len(active)


def estimate_spread(seed_set: list[int], out_adj: dict, in_adj: dict,
                    N: int, model: str = "IC", mc_runs: int = 100):
    """
    Monte Carlo estimate of influence spread.
    Returns mean and std over mc_runs simulations.
    """
    spreads = []
    all_traces = []

    sim_fn = simulate_IC if model == "IC" else simulate_LT
    kwargs = dict(out_adj=out_adj, N=N) if model == "IC" else dict(in_adj=in_adj, N=N)

    for _ in range(mc_runs):
        trace, spread = sim_fn(seed_set, **kwargs)
        spreads.append(spread)
        all_traces.append(trace)

    return np.mean(spreads), np.std(spreads), all_traces


# ─────────────────────────────────────────────
# 4.  Dataset Generation
# ─────────────────────────────────────────────

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
    seed_sets       : (n_samples, k)       int32
    spreads         : (n_samples,)         float32   mean spread
    spread_stds     : (n_samples,)         float32   std of spread
    cascade_arrays  : list of np.ndarray   each (T_i, N) bool
                      T_i = number of timesteps in cascade i
    """
    np.random.seed(rng_seed)
    nodes = np.arange(N, dtype=np.int32)

    seed_sets    = []
    spreads      = []
    spread_stds  = []
    cascade_list = []  # one representative trace per sample (first MC run)

    sim_fn = simulate_IC if model == "IC" else simulate_LT

    t0 = time.time()
    for i in range(n_samples):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_samples - i - 1)
            print(f"  [{model}] sample {i+1}/{n_samples} | "
                  f"elapsed={elapsed:.0f}s | ETA={eta:.0f}s")

        seed = sorted(np.random.choice(nodes, size=k, replace=False).tolist())

        # Run mc_runs simulations for spread estimate
        spread_vals = []
        rep_trace = None

        kwargs = dict(out_adj=out_adj, N=N) if model == "IC" else dict(in_adj=in_adj, N=N)
        for r in range(mc_runs):
            trace, spread = sim_fn(seed, **kwargs)
            spread_vals.append(spread)
            if r == 0:
                rep_trace = trace  # save first run as representative cascade

        mean_sp = float(np.mean(spread_vals))
        std_sp  = float(np.std(spread_vals))

        # Convert trace to dense timestep matrix: (T, N) bool
        T = len(rep_trace)
        cascade_mat = np.zeros((T, N), dtype=bool)
        for t, wave in enumerate(rep_trace):
            for v in wave:
                cascade_mat[t, v] = True

        seed_sets.append(seed)
        spreads.append(mean_sp)
        spread_stds.append(std_sp)
        cascade_list.append(cascade_mat)

    seed_sets   = np.array(seed_sets,   dtype=np.int32)
    spreads     = np.array(spreads,     dtype=np.float32)
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
    Save samples to npz. Cascades are saved as a ragged structure:
      cascade_data    : concatenated (T_i * N,) booleans (flattened rows)
      cascade_offsets : (S+1,) int — start index in cascade_data for sample i
      cascade_lengths : (S,)   int — number of timesteps T_i for sample i
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pack ragged cascade array
    cascade_data    = []
    cascade_offsets = [0]
    cascade_lengths = []
    N               = cascade_list[0].shape[1]

    for mat in cascade_list:
        T = mat.shape[0]
        cascade_data.append(mat.flatten())
        cascade_offsets.append(cascade_offsets[-1] + T * N)
        cascade_lengths.append(T)

    cascade_data    = np.concatenate(cascade_data)
    cascade_offsets = np.array(cascade_offsets, dtype=np.int64)
    cascade_lengths = np.array(cascade_lengths, dtype=np.int32)

    out_path = out_dir / f"samples_{model.lower()}.npz"
    np.savez_compressed(
        out_path,
        seed_sets       = seed_sets,
        spreads         = spreads,
        spread_stds     = spread_stds,
        cascade_data    = cascade_data,
        cascade_offsets = cascade_offsets,
        cascade_lengths = cascade_lengths,
        n_nodes         = np.array([N], dtype=np.int32),
    )
    print(f"[✓] Saved {model} samples → {out_path}  "
          f"({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def save_graph(
    out_dir: Path,
    edge_index: np.ndarray,
    ic_probs: np.ndarray,
    lt_weights: np.ndarray,
    node_feats: np.ndarray,
    node_labels: np.ndarray,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "graph_data.npz"
    np.savez_compressed(
        out_path,
        edge_index   = edge_index,
        ic_probs     = ic_probs,
        lt_weights   = lt_weights,
        node_feats   = node_feats,
        node_labels  = node_labels,
    )
    print(f"[✓] Saved graph data → {out_path}")
    return out_path


# ─────────────────────────────────────────────
# 5.  Loading Utilities (for downstream use)
# ─────────────────────────────────────────────

def load_graph(data_dir: Path):
    """Load graph data. Returns dict with numpy arrays."""
    d = np.load(data_dir / "graph_data.npz")
    return {k: d[k] for k in d.files}


def load_samples(data_dir: Path, model: str = "IC"):
    """
    Load samples and reconstruct cascade traces.

    Returns
    -------
    seed_sets  : (S, k) int32
    spreads    : (S,)   float32
    spread_stds: (S,)   float32
    cascades   : list of (T_i, N) bool arrays — one per sample
    """
    d = np.load(data_dir / f"samples_{model.lower()}.npz")
    N               = int(d["n_nodes"][0])
    cascade_data    = d["cascade_data"]
    cascade_offsets = d["cascade_offsets"]
    cascade_lengths = d["cascade_lengths"]

    cascades = []
    for i, T in enumerate(cascade_lengths):
        start = cascade_offsets[i]
        end   = cascade_offsets[i + 1]
        mat   = cascade_data[start:end].reshape(T, N)
        cascades.append(mat)

    return d["seed_sets"], d["spreads"], d["spread_stds"], cascades


# ─────────────────────────────────────────────
# 6.  Quick Stats / Sanity Check
# ─────────────────────────────────────────────

def print_dataset_stats(seed_sets, spreads, spread_stds, cascades, model):
    S, k = seed_sets.shape
    N    = cascades[0].shape[1]
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


# ─────────────────────────────────────────────
# 7.  Main
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate IM data on Cora-ML")
    p.add_argument("--samples",  type=int, default=1000,
                   help="Number of (seed_set, spread) samples per model (default: 1000)")
    p.add_argument("--k",        type=int, default=10,
                   help="Seed set size (default: 10)")
    p.add_argument("--mc-runs",  type=int, default=100,
                   help="Monte Carlo runs per sample for spread estimation (default: 100)")
    p.add_argument("--models",   nargs="+", default=["IC", "LT"],
                   choices=["IC", "LT"],
                   help="Diffusion models to generate data for (default: IC LT)")
    p.add_argument("--max-steps", type=int, default=50,
                   help="Max cascade propagation steps (default: 50)")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--out-dir",  type=str,
                   default=str(Path(__file__).parent / "cora_ml"),
                   help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    # Step 1: Download and load graph
    npz_path = download_cora_ml()
    adj, node_feats, node_labels = load_cora_ml(npz_path)

    # Step 2: Build propagation structures
    edge_index, ic_probs, lt_weights, out_adj, in_adj, N = build_graph(adj)

    # Step 3: Save graph data
    save_graph(out_dir, edge_index, ic_probs, lt_weights, node_feats, node_labels)

    # Step 4: Generate samples for each diffusion model
    metadata = {
        "dataset": "cora_ml",
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
        print(f"\n[→] Generating {args.samples} samples under {model} model "
              f"(k={args.k}, mc_runs={args.mc_runs}) ...")

        seed_sets, spreads, spread_stds, cascades = generate_samples(
            N         = N,
            out_adj   = out_adj,
            in_adj    = in_adj,
            model     = model,
            k         = args.k,
            n_samples = args.samples,
            mc_runs   = args.mc_runs,
            max_steps = args.max_steps,
            rng_seed  = args.seed,
        )

        save_samples(out_dir, model, seed_sets, spreads, spread_stds, cascades)
        print_dataset_stats(seed_sets, spreads, spread_stds, cascades, model)

        metadata[f"spread_mean_{model}"] = float(spreads.mean())
        metadata[f"spread_std_{model}"]  = float(spreads.std())

    # Step 5: Save metadata
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[✓] Metadata saved → {meta_path}")
    print("\n[✓] Done! Load your data with:")
    print(f"    graph = load_graph(Path('{out_dir}'))")
    print(f"    seeds, spreads, stds, cascades = load_samples(Path('{out_dir}'), model='IC')")


if __name__ == "__main__":
    main()
