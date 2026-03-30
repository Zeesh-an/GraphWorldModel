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
    """Symmetric normalization:  D^{-1/2} A D^{-1/2}"""
    rowsum = np.array(mx.sum(1)).flatten()
    r_inv_sqrt = np.power(rowsum, -0.5)
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(r_inv_sqrt)

    return mx.dot(D_inv_sqrt).transpose().dot(D_inv_sqrt)


def adj_process(adj: sp.spmatrix) -> torch.Tensor:
    """
    Symmetrize, add self-loops, normalise, convert to sparse COO torch tensor.
    Mirrors baseline adj_process() exactly.
    """
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    coo = adj.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data)

    return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()


# Load from our .npx format
def load_npz(data_dir: Path, diffusion_model: str = "IC"):
    """
    Load graph and samples from our generated .npz files.

    Returns
    -------
    adj: scipy sparse CSR  (N, N)
    inverse_pairs: torch.Tensor (S, N, 2) float32
                    [:,:,0] = binary seed vector
                    [:,:,1] = binary influence vector
    """
    g = np.load(data_dir / "graph_data.npz")
    smp = np.load(data_dir / f"samples_{diffusion_model.lower()}.npz")

    # Reconstruct scipy sparse adj from edge_index + unit weights
    edge_index = g["edge_index"]  # (2, E)
    N = int(g["node_feats"].shape[0])
    data = np.ones(edge_index.shape[1], dtype=np.float32)
    adj = sp.csr_matrix((data, (edge_index[0], edge_index[1])), shape=(N, N))

    # Build (S, N, 2) inverse_pairs
    seed_sets = smp["seed_sets"]  # (S, k)
    cascade_data = smp["cascade_data"]
    cascade_offsets = smp["cascade_offsets"]
    cascade_lengths = smp["cascade_lengths"]
    S = seed_sets.shape[0]

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
        influ_vecs[i] = mat.any(axis=0).astype(np.float32)

    inverse_pairs = torch.from_numpy(
        np.stack([seed_vecs, influ_vecs], axis=-1)  # (S, N, 2)
    )

    return adj, inverse_pairs


# Load critical node detection (CND) samples from .npz format
def load_npz_cnd(data_dir: Path):
    """
    Load graph and CND samples from generated .npz files.

    Returns
    -------
    adj: scipy sparse CSR  (N, N)
    inverse_pairs: torch.Tensor (S, N, 2) float32
                    [:,:,0] = binary removal vector
                    [:,:,1] = binary connectivity vector (largest CC membership)
    """
    g = np.load(data_dir / "graph_data.npz")
    smp = np.load(data_dir / "samples_cnd.npz")

    # Reconstruct scipy sparse adj from edge_index
    edge_index = g["edge_index"]  # (2, E)
    N = int(g["node_feats"].shape[0])
    data = np.ones(edge_index.shape[1], dtype=np.float32)
    adj = sp.csr_matrix((data, (edge_index[0], edge_index[1])), shape=(N, N))

    removal_sets = smp["removal_sets"]  # (S, k)
    connectivity_vecs = smp["connectivity_vecs"]  # (S, N)
    S = removal_sets.shape[0]

    # Build removal vectors from removal_sets indices
    removal_vecs = np.zeros((S, N), dtype=np.float32)
    for i in range(S):
        removal_vecs[i, removal_sets[i]] = 1.0

    inverse_pairs = torch.from_numpy(
        np.stack([removal_vecs, connectivity_vecs], axis=-1)  # (S, N, 2)
    )

    return adj, inverse_pairs


# Load from our baseline .SG pickle format
def load_sg(sg_path: Path):
    """
    Load from baseline .SG pickle.

    Returns
    -------
    adj           : scipy sparse
    inverse_pairs : torch.Tensor  (S, N, 2)
    """
    sys.path.insert(0, str(sg_path.parent))
    with open(sg_path, "rb") as f:
        graph = pickle.load(f)

    adj = graph["adj"]
    pairs_np = graph["inverse_pairs"]  # numpy (S, N, 2)
    inverse_pairs = torch.FloatTensor(pairs_np)

    return adj, inverse_pairs


# Unfied data loader
def load_data(
    dataset: str = "cora_ml",
    diffusion_model: str = "IC",
    seed_rate: int = 1,
    task: str = "IM",
    npz_dir: Path = None,
    sg_dir: Path = None,
):
    """
    Try npz_dir first; fall back to sg_dir (baseline .SG files).

    Parameters
    ----------
    task : "IM" or "CND" — selects which samples file to load

    Returns: adj (scipy sparse), inverse_pairs (S, N, 2) torch float
    """
    if task == "CND":
        if npz_dir is not None and (npz_dir / "samples_cnd.npz").exists():
            print(f"[data] Loading CND from npz: {npz_dir}")
            return load_npz_cnd(npz_dir)

        raise FileNotFoundError(
            f"No CND data found. Run generate_cnd_data.py first, "
            f"then provide npz_dir containing samples_cnd.npz"
        )

    if npz_dir is not None and (npz_dir / "graph_data.npz").exists():
        print(f"[data] Loading from npz: {npz_dir}")
        return load_npz(npz_dir, diffusion_model)

    if sg_dir is not None:
        sg_name = f"{dataset}_mean_{diffusion_model}{10 * seed_rate}.SG"
        sg_path = sg_dir / sg_name
        if sg_path.exists():
            print(f"[data] Loading from SG: {sg_path}")
            return load_sg(sg_path)

    raise FileNotFoundError(
        f"No data found. Provide npz_dir (with graph_data.npz) "
        f"or sg_dir (with {dataset}_mean_{diffusion_model}*.SG)"
    )


# Dataset
class InverseProblemDataset(torch.utils.data.Dataset):
    """
    Wraps inverse_pairs tensor.  Each item is (N, 2): [seed_vec, influ_vec].
    Identical interface to baseline InverseProblemDataset.
    """

    def __init__(self, inverse_pairs: torch.Tensor):
        self.data = inverse_pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Sampling helper (mirrors baseline sampling())
def top_diffusion_sampling(
    inverse_pairs: torch.Tensor, frac: float = 0.1
) -> torch.Tensor:
    """
    Return indices of the top-frac% samples by influence spread.
    Used to initialise latent z in Phase 2 for IM.
    """
    counts = inverse_pairs[:, :, 1].sum(dim=1)  # (S,)
    k = max(1, int(frac * len(inverse_pairs)))

    return counts.topk(k).indices


def bottom_connectivity_sampling(
    inverse_pairs: torch.Tensor, frac: float = 0.1
) -> torch.Tensor:
    """
    Return indices of the bottom-frac% samples by residual connectivity.
    These are the most destructive removal sets.
    Used to intiialize the best latent z in Phase 2 for CND.
    """
    counts = inverse_pairs[:, :, 1].sum(dim=1)  # (S,)
    k = max(1, int(frac * len(inverse_pairs)))

    return counts.topk(k, largest=False).indices


# Evaluation (mirrors baseline diffusion_evaluation)
def diffusion_evaluation(
    adj: sp.spmatrix,
    seed: list,
    diffusion: str = "IC",
    n_runs: int = 10,
    max_steps: int = 100,
) -> float:
    """
    Monte Carlo diffusion evaluation using our simulator
    (avoids ndlib dependency from baseline).

    Returns mean influence spread over n_runs.
    """
    # Build out-adjacency from sparse matrix
    adj_csr = adj.tocsr()
    N = adj_csr.shape[0]

    out_adj = defaultdict(list)
    if diffusion in ("IC", "LT"):
        coo = adj_csr.tocoo()
        in_deg = np.array(adj_csr.sum(axis=0)).flatten()
        in_deg = np.where(in_deg == 0, 1, in_deg)
        for u, v in zip(coo.row, coo.col):
            p = float(1.0 / in_deg[v])
            out_adj[int(u)].append((int(v), min(p, 0.5)))

    total = 0
    for _ in range(n_runs):
        if diffusion == "IC":
            active = set(seed)
            newly_active = set(seed)

            for _t in range(max_steps):
                wave = set()
                for u in newly_active:
                    for v, p in out_adj[u]:
                        if v not in active and np.random.random() < p:
                            wave.add(v)
                if not wave:
                    break
                active |= wave
                newly_active = wave

            total += len(active)

        elif diffusion == "LT":
            thresholds = np.random.uniform(0, 1, N)
            active = set(seed)
            newly_active = set(seed)
            influence = np.zeros(N)
            for u in seed:
                for v, w in out_adj[u]:
                    influence[v] += w

            for _t in range(max_steps):
                wave = {
                    v
                    for v in range(N)
                    if v not in active and influence[v] >= thresholds[v]
                }
                if not wave:
                    break
                active |= wave
                newly_active = wave
                for u in newly_active:
                    for v, w in out_adj[u]:
                        if v not in active:
                            influence[v] += w

            total += len(active)

        else:
            raise ValueError(f"Unsupported diffusion model: {diffusion}")

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
    adj         : scipy sparse adjacency (directed, will be symmetrized)
    removal_set : list of node indices to remove

    Returns
    -------
    dict with keys:
        largest_cc    : size of largest connected component after removal
        n_components  : number of connected components after removal
        pairwise_conn : number of reachable node pairs
        frac_removed  : largest_cc / (N - k), fraction of remaining nodes still connected
    """
    # Symmetrize for undirected connectivity
    adj_sym = adj + adj.T
    adj_sym = (adj_sym > 0).astype(np.float32)
    adj_sym.setdiag(0)
    adj_sym.eliminate_zeros()

    N = adj_sym.shape[0]
    G = nx.from_scipy_sparse_array(adj_sym)

    remaining = set(G.nodes()) - set(removal_set)
    G_residual = G.subgraph(remaining)

    components = list(nx.connected_components(G_residual))
    if not components:
        return {
            "largest_cc": 0,
            "n_components": 0,
            "pairwise_conn": 0,
            "frac_connected": 0.0,
        }

    largest_cc_size = max(len(c) for c in components)
    pairwise_conn = sum(len(c) * (len(c) - 1) // 2 for c in components)
    n_remaining = N - len(removal_set)

    return {
        "largest_cc": largest_cc_size,
        "n_components": len(components),
        "pairwise_conn": pairwise_conn,
        "frac_connected": largest_cc_size / max(n_remaining, 1),
    }
