"""
Connectivity Simulations
=====================

Deterministic node removal and residual connectivity.
Used for data generation and evaluation across CND tasks.
"""

import numpy as np
import networkx as nx


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
