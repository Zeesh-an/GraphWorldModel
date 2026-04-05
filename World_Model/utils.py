"""
Data utilities — mirrors baseline's main/utils.py
===================================================

Supports loading from:
    1. Our generated .npz files  (Data/cora_ml/)         ← preferred
    2. Baseline .SG pickle files (Baselines/DeepIM/data/) ← for comparison

Both converge to the same inverse_pairs tensor: (S, N, 2) float32

IM  task:
    [:, :, 0] = binary seed vector
    [:, :, 1] = binary influence vector (nodes activated at any cascade timestep)

CND task:
    [:, :, 0] = binary removal vector (1 = node removed)
    [:, :, 1] = binary connectivity vector (1 = node in largest CC after removal)

SL  task (source localization):
    [:, :, 0] = binary seed vector (ground truth sources — the label)
    [:, :, 1] = binary snapshot vector (partial observation at time t — the input to invert)
"""

import math
import random
import pickle
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import torch.nn.functional as F


# Adjacency helpers (identical to baseline)
def normalize_adj(mx: sp.spmatrix) -> sp.spmatrix:
    """
    Symmetric normalization: normalizes node features based on their degrees, reducing the influence of high-degree nodes

    Computes: D^{-1/2} A D^{-1/2}
    Each edge (i, j) becomes 1 / sqrt(deg_i * deg_j)
    """
    rowsum = np.array(mx.sum(1)).flatten()  # (N,) Degree of each node

    r_inv_sqrt = np.power(rowsum, -0.5)
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(r_inv_sqrt)

    return mx.dot(D_inv_sqrt).transpose().dot(D_inv_sqrt)


def adj_process(adj: sp.spmatrix) -> torch.Tensor:
    """
    Symmetrize, add self-loops, normalise, convert to sparse COO torch tensor.
    Mirrors baseline adj_process() exactly.
    """
    # Symmetrize the graph (directed -> undirected) - if edge (u, v) exists but (v, u) does not, then add it
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # Add self-loops
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    # Convert to COO sparse matrix format
    coo = adj.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data)

    return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()


def load_npz(graph_data_path: Path, samples_data_path: Path):
    """
    Load graph data and samples from generated .npz files.

    Returns
    -------
    samples: all data samples
    N: int - number of nodes
    adj: scipy sparse CSR  (N, N)
    """
    graph = np.load(graph_data_path)
    samples = np.load(samples_data_path)

    # Reconstruct scipy sparse adj from edge_index and unit weights
    edge_index = graph["edge_index"]  # (2, E)
    N = int(graph["node_feats"].shape[0])  # N = number of nodes

    # Unit weight 1.0 for all edges
    data = np.ones(edge_index.shape[1], dtype=np.float32)

    # Build sparse adjacency matrix
    adj = sp.csr_matrix((data, (edge_index[0], edge_index[1])), shape=(N, N))

    return samples, N, adj


def load_npz_im(data_dir: Path, diffusion_model: str = "IC", k: int = 10):
    """
    Load graph and Influence Maximization (IM) samples from generated .npz files.

    Returns
    -------
    adj: scipy sparse CSR  (N, N)
    inverse_pairs: torch.Tensor (S, N, 2) float32
                    [:,:,0] = binary seed vector
                    [:,:,1] = binary influence vector
    """
    samples, N, adj = load_npz(
        graph_data_path=data_dir / "graph_data.npz",
        samples_data_path=data_dir / f"samples_im_{diffusion_model.lower()}_k{k}.npz",
    )

    # Build (S, N, 2) inverse_pairs
    seed_sets = samples["seed_sets"]  # (S, k)
    cascade_data = samples["cascade_data"]
    cascade_offsets = samples["cascade_offsets"]
    cascade_lengths = samples["cascade_lengths"]
    S = seed_sets.shape[0]  # Number of examples/seed sets

    seed_vecs = np.zeros((S, N), dtype=np.float32)
    influ_vecs = np.zeros((S, N), dtype=np.float32)

    for i in range(S):
        # Seed vector
        seed_vecs[i, seed_sets[i]] = 1.0

        # Influence vector — union of all activated nodes across all timesteps
        start = cascade_offsets[i]
        end = cascade_offsets[i + 1]
        T = int(cascade_lengths[i])
        mat = cascade_data[start:end].reshape(T, N)  # (T, N) bool

        # Take the union across all timesteps, so a node is influenced if it was activated at any timestep
        influ_vecs[i] = mat.any(axis=0).astype(np.float32)

    inverse_pairs = torch.from_numpy(
        np.stack([seed_vecs, influ_vecs], axis=-1)  # (S, N, 2)
    )

    return adj, inverse_pairs


def load_npz_cnd(data_dir: Path, k: int = 10):
    """
    Load graph and Critical Node Detection (CND) samples from generated .npz files.

    Returns
    -------
    adj: scipy sparse CSR  (N, N)
    inverse_pairs: torch.Tensor (S, N, 2) float32
                    [:,:,0] = binary removal vector
                    [:,:,1] = binary connectivity vector (largest CC membership)
    """
    samples, N, adj = load_npz(
        graph_data_path=data_dir / "graph_data.npz",
        samples_data_path=data_dir / f"samples_cnd_k{k}.npz",
    )

    removal_sets = samples["removal_sets"]  # (S, k)
    connectivity_vecs = samples["connectivity_vecs"]  # (S, N)
    S = removal_sets.shape[0]  # Number of examples/seed sets

    # Build removal vectors from removal_sets indices
    removal_vecs = np.zeros((S, N), dtype=np.float32)
    for i in range(S):
        removal_vecs[i, removal_sets[i]] = 1.0

    inverse_pairs = torch.from_numpy(
        np.stack([removal_vecs, connectivity_vecs], axis=-1)  # (S, N, 2)
    )

    return adj, inverse_pairs


def load_npz_sl(data_dir: Path, diffusion_model: str = "IC", k: int = 10):
    """
    Load graph and Source Localization (SL) samples from generated .npz files.

    SL shares the same forward model as IM (seeds → activation), but channel 1
    stores a partial snapshot (observed at a random timestep) rather than the
    full cascade union.

    Returns
    -------
    adj: scipy sparse CSR  (N, N)
    inverse_pairs: torch.Tensor (S, N, 2) float32
                    [:,:,0] = binary seed vector (ground truth sources)
                    [:,:,1] = binary snapshot vector (partial observation)
    """
    samples, N, adj = load_npz(
        graph_data_path=data_dir / "graph_data.npz",
        samples_data_path=data_dir / f"samples_sl_{diffusion_model.lower()}_k{k}.npz",
    )

    seed_sets = samples["seed_sets"]  # (S, k)
    snapshots = samples["snapshots"]  # (S, N)
    S = seed_sets.shape[0]

    # Build seed vectors from seed_sets indices
    seed_vecs = np.zeros((S, N), dtype=np.float32)
    for i in range(S):
        seed_vecs[i, seed_sets[i]] = 1.0

    inverse_pairs = torch.from_numpy(
        np.stack([seed_vecs, snapshots], axis=-1)  # (S, N, 2)
    )

    return adj, inverse_pairs


def load_sg(sg_path: Path):
    """
    Load data from the baseline's .SG pickle format.

    Returns
    -------
    adj: scipy sparse
    inverse_pairs: torch.Tensor  (S, N, 2)
    """
    sys.path.insert(0, str(sg_path.parent))
    with open(sg_path, "rb") as f:
        graph = pickle.load(f)

    adj = graph["adj"]
    pairs_np = graph["inverse_pairs"]  # numpy (S, N, 2)
    inverse_pairs = torch.FloatTensor(pairs_np)

    return adj, inverse_pairs


def load_data(
    dataset: str = "cora_ml",
    diffusion_model: str = "IC",
    seed_rate: int = 1,
    k: int = 10,
    task: str = "IM",
    npz_dir: Path = None,
    sg_dir: Path = None,
):
    """
    Unified data loader function.
    Tries npz_dir first; falls back to sg_dir (baseline .SG files).

    Parameters
    ----------
    task: "IM", "CND", or "SL" — selects which samples file to load
    k: seed/removal/source set size — used in the filename (e.g. samples_im_ic_k10.npz)

    Returns: adj (scipy sparse), inverse_pairs (S, N, 2) torch float
    """
    if task == "IM":
        im_file = f"samples_im_{diffusion_model.lower()}_k{k}.npz"
        if (
            npz_dir is not None
            and (npz_dir / "graph_data.npz").exists()
            and (npz_dir / im_file).exists()
        ):
            print(f"[data] Loading IM from npz: {npz_dir}/{im_file}")
            return load_npz_im(npz_dir, diffusion_model, k)

        if sg_dir is not None:
            sg_name = f"{dataset}_mean_{diffusion_model.lower()}{10 * seed_rate}.SG"
            sg_path = sg_dir / sg_name
            if sg_path.exists():
                print(f"[data] Loading from SG: {sg_path}")
                return load_sg(sg_path)

        raise FileNotFoundError(
            f"No IM data found. Run generate_im_data.py first. "
            f"Expected: {npz_dir}/{im_file}"
        )

    if task == "CND":
        cnd_file = f"samples_cnd_k{k}.npz"
        if (
            npz_dir is not None
            and (npz_dir / "graph_data.npz").exists()
            and (npz_dir / cnd_file).exists()
        ):
            print(f"[data] Loading CND from npz: {npz_dir}/{cnd_file}")
            return load_npz_cnd(npz_dir, k)

        raise FileNotFoundError(
            f"No CND data found. Run generate_cnd_data.py first. "
            f"Expected: {npz_dir}/{cnd_file}"
        )

    if task == "SL":
        sl_file = f"samples_sl_{diffusion_model.lower()}_k{k}.npz"
        if (
            npz_dir is not None
            and (npz_dir / "graph_data.npz").exists()
            and (npz_dir / sl_file).exists()
        ):
            print(f"[data] Loading SL from npz: {npz_dir}/{sl_file}")
            return load_npz_sl(npz_dir, diffusion_model, k)

        raise FileNotFoundError(
            f"No SL data found. Run generate_sl_data.py first. "
            f"Expected: {npz_dir}/{sl_file}"
        )


class InverseProblemDataset(torch.utils.data.Dataset):
    """
    Wraps inverse_pairs tensor. Each item is (N, 2): [seed_vec, influ_vec].
    Identical interface to baseline InverseProblemDataset.
    """

    def __init__(self, inverse_pairs: torch.Tensor):
        self.data = inverse_pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def top_sampling_init_indices(
    inverse_pairs: torch.Tensor, frac: float = 0.1, largest: bool = True
) -> torch.Tensor:
    """
    Return indices of the top or bottom percent samples by influence spread (or residual connectivity).
    Used to initialize the optimal latent z in Phase 2.
    """
    counts = inverse_pairs[:, :, 1].sum(dim=1)  # (S,)
    k = max(1, int(frac * len(inverse_pairs)))

    return counts.topk(k, largest=largest).indices


def diffusion_evaluation(
    adj: sp.spmatrix,
    seed: list,
    diffusion: str = "IC",
    n_runs: int = 10,
    max_steps: int = 100,
) -> float:
    """
    Monte Carlo diffusion evaluation using our simulator to evaluate a seed set's actual influence spread.
    (avoids the ndlib dependency from baseline)

    Returns the mean influence spread over n_runs.
    """
    # Build out-adjacency from the sparse matrix
    # Convert to a Compressed Sparse Row (CSR) graph for efficient representation of the sparse graph
    adj_csr = adj.tocsr()
    N = adj_csr.shape[0]

    # Build an out-adjacency list
    out_adj = defaultdict(list)
    if diffusion in ("IC", "LT"):
        coo = adj_csr.tocoo()

        in_deg = np.array(adj_csr.sum(axis=0)).flatten()
        in_deg = np.where(in_deg == 0, 1, in_deg)

        for u, v in zip(coo.row, coo.col):
            # For each edge (u, v), the propagation probability is 1/in_degree(v), capped at 0.5.
            p = float(1.0 / in_deg[v])
            out_adj[int(u)].append((int(v), min(p, 0.5)))

    total = 0

    for run in range(n_runs):
        # Independent Cascade (IC) Diffusion Simulation
        if diffusion == "IC":
            # Initialize with the seed set
            active = set(seed)
            newly_active = set(seed)

            for _ in range(max_steps):
                wave = set()

                # For every newly active node u, attempt to activate each neighbor node v with probability p
                for u in newly_active:
                    for v, p in out_adj[u]:
                        if v not in active and np.random.random() < p:
                            wave.add(v)

                if not wave:
                    break

                # Merge wave into the active set
                active |= wave
                newly_active = wave

            total += len(active)

        # Linear Threshold (LT) Diffusion Simulation
        elif diffusion == "LT":
            # Get a random threshold for each node
            thresholds = np.random.uniform(0, 1, N)

            # Initialize with the seed set
            active = set(seed)
            newly_active = set(seed)
            influence = np.zeros(N)

            # For each seed node, add its edge weight to each neighbor's influence score
            for u in seed:
                for v, w in out_adj[u]:
                    influence[v] += w

            for _ in range(max_steps):
                # Find all inactive nodes whose accumulated influence meets their threshold for wave
                wave = {
                    v
                    for v in range(N)
                    if v not in active and influence[v] >= thresholds[v]
                }

                if not wave:
                    break

                # Merge wave into the active set
                active |= wave
                newly_active = wave

                # Propagate the influence from newly active nodes to their inactive neighbors
                for u in newly_active:
                    for v, w in out_adj[u]:
                        if v not in active:
                            influence[v] += w

            total += len(active)

        else:
            raise ValueError(f"Unsupported diffusion model: {diffusion}")

    # Average spread over all Monte Carlo simulations
    return total / n_runs


# Critical Node Detection (CND) Evaluation
def connectivity_evaluation(
    adj: sp.spmatrix,
    removal_set: list[int],
) -> dict[str, float]:
    """
    Evaluate a CND solution by removing nodes and measuring residual connectivity.
    This is deterministic: no Monte Carlo runs are needed.

    Parameters
    ----------
    adj: scipy sparse adjacency (directed, will be symmetrized)
    removal_set: list of node indices to remove

    Returns
    -------
    dict with keys:
        largest_cc: size of largest connected component after removal
        n_components: number of connected components after removal
        pairwise_conn: number of reachable node pairs
        frac_removed: largest_cc / (N - k), fraction of remaining nodes still connected
    """
    # Symmetrize the graph adjacency matrix to ensure undirected connectivity, remove self-loops, and clean up zeros
    adj_sym = adj + adj.T
    adj_sym = (adj_sym > 0).astype(np.float32)
    adj_sym.setdiag(0)
    adj_sym.eliminate_zeros()

    N = adj_sym.shape[0]

    # Build a NetworkX graph from the sparse matrix
    G = nx.from_scipy_sparse_array(adj_sym)

    # Remove the specified nodes and take the subgraph of remaining nodes
    remaining = set(G.nodes()) - set(removal_set)
    G_residual = G.subgraph(remaining)

    # Find all connected components
    components = list(nx.connected_components(G_residual))

    if not components:
        # If (all nodes have been removed, return zeros
        return {
            "largest_cc": 0,
            "n_components": 0,
            "pairwise_conn": 0,
            "frac_connected": 0.0,
        }

    # Compute the size of the biggest connected component
    largest_cc_size = max(len(c) for c in components)

    # Compute the total reachable node pairs
    pairwise_conn = sum(len(c) * (len(c) - 1) // 2 for c in components)
    n_remaining = N - len(removal_set)

    return {
        "largest_cc": largest_cc_size,
        "n_components": len(components),
        "pairwise_conn": pairwise_conn,
        "frac_connected": largest_cc_size / max(n_remaining, 1),
    }


# Source Localization (SL) Evaluation
def source_localization_evaluation(
    predicted_sources: list[int],
    true_sources: list[int],
    observed_snapshot: np.ndarray,
    adj: sp.spmatrix,
    diffusion: str = "IC",
    n_runs: int = 10,
    max_steps: int = 100,
) -> dict[str, float]:
    """
    Evaluate a source localization solution by comparing predicted sources
    against ground truth and optionally checking if the predicted sources
    reproduce the observed snapshot via forward simulation.

    Parameters
    ----------
    predicted_sources: list of predicted source node indices
    true_sources: list of ground truth source node indices
    observed_snapshot: (N,) float32 binary — the observed infection snapshot
    adj: scipy sparse adjacency matrix
    diffusion: diffusion model used ("IC" or "LT")
    n_runs: Monte Carlo runs for forward simulation check
    max_steps: max cascade propagation steps

    Returns
    -------
    dict with keys:
        precision: fraction of predicted sources that are true sources
        recall: fraction of true sources that were predicted
        f1: harmonic mean of precision and recall
        jaccard: |predicted ∩ true| / |predicted ∪ true|
        predicted_spread: mean spread when simulating from predicted sources
        observed_spread: number of activated nodes in the observed snapshot
        spread_error: |predicted_spread - observed_spread| / observed_spread
    """
    pred_set = set(predicted_sources)
    true_set = set(true_sources)

    # Set-based source recovery metrics
    true_positives = len(pred_set & true_set)
    precision = true_positives / max(len(pred_set), 1)
    recall = true_positives / max(len(true_set), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    jaccard = true_positives / max(len(pred_set | true_set), 1)

    # Forward simulation: do the predicted sources reproduce the observation?
    predicted_spread = diffusion_evaluation(
        adj,
        predicted_sources,
        diffusion=diffusion,
        n_runs=n_runs,
        max_steps=max_steps,
    )
    observed_spread = float(observed_snapshot.sum())
    spread_error = abs(predicted_spread - observed_spread) / max(observed_spread, 1.0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "predicted_spread": predicted_spread,
        "observed_spread": observed_spread,
        "spread_error": spread_error,
    }
