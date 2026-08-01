"""
Pure classical-IM primitives over GraphInfo.

These are the building blocks the named algorithms (algorithms.py) compose, and
the surface the coding agent is told it may call. Spread estimation uses the real
NDlib simulator (data/wm_simulator.py) — i.e. the honest classical cost; the
outer-loop Environment (envs/) is what the world model accelerates.
"""

import math
import networkx as nx
import numpy as np

from coding_agent.types import GraphInfo
from data.wm_simulator import ActionOp, Simulator, spent

seed_upper_bound = 1 << 30
convergence_tol = 1e-9
norm_floor = 1e-12
power_iterations = 100


def build_simulator(
    graph: GraphInfo,
    diffusion_model: str,
    seed: int = 0,
    remove_semantics: str = spent,
) -> Simulator:
    """Construct an NDlib Simulator from a GraphInfo."""
    nx_graph = nx.DiGraph() if graph.directed else nx.Graph()
    nx_graph.add_nodes_from(range(graph.num_nodes))
    ic_prob_map = {}

    for edge in range(graph.edge_index.shape[1]):
        source = int(graph.edge_index[0, edge])
        target = int(graph.edge_index[1, edge])
        nx_graph.add_edge(source, target)
        ic_prob_map[(source, target)] = float(graph.ic_probs[edge])

    simulator = Simulator(
        nx_graph,
        ic_prob_map=ic_prob_map,
        seed=seed,
        remove_semantics=remove_semantics,
    )
    simulator.reset(diffusion_model)

    return simulator


# Scoring / ranking
def compute_degree(graph: GraphInfo) -> np.ndarray:
    """Total (in + out) degree per node, shape (N,)."""
    degrees = np.zeros(graph.num_nodes, dtype=np.float64)

    np.add.at(degrees, graph.edge_index[0], 1.0)
    np.add.at(degrees, graph.edge_index[1], 1.0)

    return degrees


def compute_out_degree(graph: GraphInfo) -> np.ndarray:
    out_degrees = np.zeros(graph.num_nodes, dtype=np.float64)
    np.add.at(out_degrees, graph.edge_index[0], 1.0)

    return out_degrees


def compute_weighted_degree(graph: GraphInfo) -> np.ndarray:
    """Sum of outgoing IC transmission probabilities per node, shape (N,)."""
    weighted_degrees = np.zeros(graph.num_nodes, dtype=np.float64)
    np.add.at(weighted_degrees, graph.edge_index[0], graph.ic_probs.astype(np.float64))

    return weighted_degrees


def get_top_degree_nodes(graph: GraphInfo, count: int) -> list[int]:
    degrees = compute_degree(graph)
    return [int(node) for node in np.argsort(-degrees)[:count]]


def compute_pagerank(
    graph: GraphInfo, damping: float = 0.85, iters: int = 100
) -> np.ndarray:
    """Power-iteration PageRank over the directed graph, shape (N,)."""
    num_nodes = graph.num_nodes
    out_degrees = compute_out_degree(graph)
    safe_out_degrees = np.where(out_degrees == 0, 1.0, out_degrees)
    pagerank = np.full(num_nodes, 1.0 / num_nodes, dtype=np.float64)
    sources, targets = graph.edge_index[0], graph.edge_index[1]

    for _ in range(iters):
        contributions = pagerank[sources] / safe_out_degrees[sources]
        incoming = np.zeros(num_nodes, dtype=np.float64)
        np.add.at(incoming, targets, contributions)
        dangling = pagerank[out_degrees == 0].sum()
        next_pagerank = (1.0 - damping) / num_nodes + damping * (
            incoming + dangling / num_nodes
        )

        if np.abs(next_pagerank - pagerank).sum() < convergence_tol:
            pagerank = next_pagerank
            break

        pagerank = next_pagerank

    return pagerank


def compute_centrality(graph: GraphInfo, kind: str = "eigenvector") -> np.ndarray:
    """Eigenvector (power iteration) or closeness (BFS) centrality, shape (N,)."""
    num_nodes = graph.num_nodes

    if kind == "eigenvector":
        scores = np.full(num_nodes, 1.0 / num_nodes, dtype=np.float64)
        sources, targets = graph.edge_index[0], graph.edge_index[1]

        for _ in range(power_iterations):
            next_scores = np.zeros(num_nodes, dtype=np.float64)
            np.add.at(next_scores, targets, scores[sources])
            norm = np.linalg.norm(next_scores)

            if norm < norm_floor:
                break

            next_scores = next_scores / norm
            if np.abs(next_scores - scores).sum() < convergence_tol:
                scores = next_scores
                break

            scores = next_scores

        return scores

    if kind == "closeness":
        nx_graph = nx.DiGraph() if graph.directed else nx.Graph()
        nx_graph.add_nodes_from(range(num_nodes))
        nx_graph.add_edges_from(
            (int(graph.edge_index[0, edge]), int(graph.edge_index[1, edge]))
            for edge in range(graph.edge_index.shape[1])
        )
        closeness = nx.closeness_centrality(nx_graph)
        return np.array(
            [closeness.get(node, 0.0) for node in range(num_nodes)], dtype=np.float64
        )

    raise ValueError(f"unknown centrality kind {kind!r}; choose eigenvector|closeness")


# Sampling / estimation
def mc_simulate_spread(
    graph: GraphInfo,
    seeds: list[int],
    diffusion_model: str,
    mc_runs: int = 50,
    horizon: int = 20,
    seed: int = 0,
) -> float:
    """Mean final activated-node count over mc_runs rollouts (seed at t=0, then NULL)."""
    if not seeds:
        return 0.0

    rng = np.random.default_rng(seed)
    totals = []
    seed_bag = [ActionOp("add_node", int(node)) for node in seeds]

    for _ in range(mc_runs):
        simulator = build_simulator(
            graph, diffusion_model, seed=int(rng.integers(seed_upper_bound))
        )
        state = simulator.advance(seed_bag)

        for _ in range(horizon):
            if not state.frontier:
                break

            state = simulator.advance([])

        totals.append(len(state.infected))

    return float(np.mean(totals))


def compute_marginal_gain(
    graph: GraphInfo,
    seeds: list[int],
    node: int,
    diffusion_model: str,
    mc_runs: int = 50,
    horizon: int = 20,
    seed: int = 0,
) -> float:
    """spread(seeds + [node]) - spread(seeds)."""
    base_spread = mc_simulate_spread(
        graph, seeds, diffusion_model, mc_runs, horizon, seed
    )
    spread_with_node = mc_simulate_spread(
        graph, list(seeds) + [int(node)], diffusion_model, mc_runs, horizon, seed
    )

    return float(spread_with_node - base_spread)


def batch_reverse_sample(graph: GraphInfo, theta: int, seed: int = 0) -> list[set[int]]:
    """
    Generate theta reverse-reachable sets under the IC live-edge model.

    For a random root, keep each in-edge (u->root) live with prob ic_prob(u->root)
    and BFS backwards over live edges. Each RR set is the set of nodes that could
    have activated the root.
    """
    rng = np.random.default_rng(seed)

    # Build in-adjacency with probabilities once.
    in_adjacency = {node: [] for node in range(graph.num_nodes)}
    for edge in range(graph.edge_index.shape[1]):
        source = int(graph.edge_index[0, edge])
        target = int(graph.edge_index[1, edge])
        in_adjacency[target].append((source, float(graph.ic_probs[edge])))

    rr_sets = []
    for _ in range(theta):
        root = int(rng.integers(graph.num_nodes))
        seen = {root}
        stack = [root]

        while stack:
            current = stack.pop()
            for source, probability in in_adjacency[current]:
                if source not in seen and rng.random() < probability:
                    seen.add(source)
                    stack.append(source)

        rr_sets.append(seen)

    return rr_sets


def ris_select(rr_sets: list[set[int]], budget: int, num_nodes: int) -> list[int]:
    """Greedy max-coverage over RR sets -> top-k seeds (RIS selection rule)."""
    covers = {node: set() for node in range(num_nodes)}
    for rr_index, rr_set in enumerate(rr_sets):
        for node in rr_set:
            covers[node].add(rr_index)

    chosen = []
    covered = set()
    for _ in range(budget):
        best_node, best_gain = -1, -1
        for node in range(num_nodes):
            if node in chosen:
                continue
            gain = len(covers[node] - covered)
            if gain > best_gain:
                best_gain, best_node = gain, node

        if best_node < 0:
            break

        chosen.append(best_node)
        covered |= covers[best_node]

    return chosen


# Structural analysis
def detect_communities(graph: GraphInfo) -> dict[int, int]:
    """Label-propagation communities -> {node: community_id}."""
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(graph.num_nodes))
    nx_graph.add_edges_from(
        (int(graph.edge_index[0, edge]), int(graph.edge_index[1, edge]))
        for edge in range(graph.edge_index.shape[1])
    )
    communities = nx.community.label_propagation_communities(nx_graph)

    labels = {}
    for community_id, members in enumerate(communities):
        for node in members:
            labels[int(node)] = community_id

    for node in range(graph.num_nodes):
        labels.setdefault(node, len(labels))

    return labels


def allocate_budget(communities: dict[int, int], budget: int) -> dict[int, int]:
    """Distribute budget seeds across communities proportionally to size (largest-remainder)."""
    sizes = {}
    for community_id in communities.values():
        sizes[community_id] = sizes.get(community_id, 0) + 1

    total = sum(sizes.values())
    raw_shares = {
        community_id: budget * size / total for community_id, size in sizes.items()
    }
    allocation = {
        community_id: int(np.floor(share)) for community_id, share in raw_shares.items()
    }

    remaining = budget - sum(allocation.values())
    by_remainder = sorted(
        raw_shares,
        key=lambda community_id: raw_shares[community_id] - allocation[community_id],
        reverse=True,
    )
    for community_id in by_remainder[:remaining]:
        allocation[community_id] += 1

    return allocation


# Look-ahead / sketch / path primitives (heavy-family support)
def estimate_sample_size(
    graph: GraphInfo,
    budget: int,
    epsilon: float = 0.2,
    delta: float = 0.1,
    min_theta: int = 200,
    max_theta: int = 20000,
) -> int:
    """RIS sample-size (theta) heuristic; grows ~ n*log(n)/epsilon^2, clamped."""
    num_nodes = graph.num_nodes
    raw_theta = (budget + 1) * num_nodes * math.log(max(num_nodes, 2)) / (epsilon**2)

    return int(max(min_theta, min(max_theta, raw_theta)))


def sample_live_edge_graph(
    graph: GraphInfo, rng: np.random.Generator
) -> dict[int, list[int]]:
    """One IC live-edge realization -> out-adjacency of the live edges."""
    live = {node: [] for node in range(graph.num_nodes)}
    for edge in range(graph.edge_index.shape[1]):
        if rng.random() < float(graph.ic_probs[edge]):
            live[int(graph.edge_index[0, edge])].append(int(graph.edge_index[1, edge]))

    return live


def reachable_count(live_adjacency: dict[int, list[int]], seeds: list[int]) -> int:
    """Nodes reachable from seeds in a live-edge graph (forward BFS)."""
    seen = {int(seed) for seed in seeds}
    stack = list(seen)

    while stack:
        current = stack.pop()
        for neighbor in live_adjacency.get(current, ()):
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)

    return len(seen)


def path_influence_scores(graph: GraphInfo, max_hops: int = 2) -> np.ndarray:
    """
    Per-node truncated path-product influence proxy (for path-based methods), shape (N,).

    score[v] = sum over nodes reachable within max_hops of the best path-product of
    IC transmission probabilities from v.
    """
    num_nodes = graph.num_nodes
    out_adjacency = {node: [] for node in range(num_nodes)}
    for edge in range(graph.edge_index.shape[1]):
        out_adjacency[int(graph.edge_index[0, edge])].append(
            (int(graph.edge_index[1, edge]), float(graph.ic_probs[edge]))
        )

    scores = np.zeros(num_nodes, dtype=np.float64)
    for source in range(num_nodes):
        best = {source: 1.0}
        frontier = [(source, 1.0)]

        for _ in range(max_hops):
            next_frontier = []
            for current, path_probability in frontier:
                for neighbor, probability in out_adjacency[current]:
                    candidate = path_probability * probability
                    if candidate > best.get(neighbor, 0.0):
                        best[neighbor] = candidate
                        next_frontier.append((neighbor, candidate))
            frontier = next_frontier

        scores[source] = sum(
            probability for node, probability in best.items() if node != source
        )

    return scores
