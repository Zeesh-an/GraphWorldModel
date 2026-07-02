"""
Pure classical-IM primitives over GraphInfo.

These are the building blocks the named algorithms (algorithms.py) compose, and
the surface the coding agent is told it may call. Spread estimation uses the real
NDlib simulator (data/wm_simulator.py) — i.e. the honest classical cost; the
outer-loop Environment (envs/) is what the world model accelerates.
"""

import networkx as nx
import numpy as np

from coding_agent.types import GraphInfo
from data.wm_simulator import ActionOp, Simulator


def build_simulator(g: GraphInfo, diffusion_model: str, seed: int = 0) -> Simulator:
    """Construct an NDlib Simulator from a GraphInfo."""
    graph = nx.DiGraph() if g.directed else nx.Graph()
    graph.add_nodes_from(range(g.num_nodes))
    ic_prob_map = {}

    for i in range(g.edge_index.shape[1]):
        u, v = int(g.edge_index[0, i]), int(g.edge_index[1, i])
        graph.add_edge(u, v)
        ic_prob_map[(u, v)] = float(g.ic_probs[i])

    sim = Simulator(graph, ic_prob_map=ic_prob_map, seed=seed)
    sim.reset(diffusion_model)

    return sim


# Scoring/Ranking
def compute_degree(g: GraphInfo) -> np.ndarray:
    """Total (in + out) degree per node, shape (N,)."""
    deg = np.zeros(g.num_nodes, dtype=np.float64)

    np.add.at(deg, g.edge_index[0], 1.0)
    np.add.at(deg, g.edge_index[1], 1.0)

    return deg


def compute_out_degree(g: GraphInfo) -> np.ndarray:
    deg = np.zeros(g.num_nodes, dtype=np.float64)
    np.add.at(deg, g.edge_index[0], 1.0)

    return deg


def compute_weighted_degree(g: GraphInfo) -> np.ndarray:
    """Sum of outgoing IC transmission probabilities per node, shape (N,)."""
    wd = np.zeros(g.num_nodes, dtype=np.float64)
    np.add.at(wd, g.edge_index[0], g.ic_probs.astype(np.float64))

    return wd


def get_top_degree_nodes(g: GraphInfo, k: int) -> list[int]:
    deg = compute_degree(g)
    return [int(v) for v in np.argsort(-deg)[:k]]


def compute_pagerank(
    g: GraphInfo, damping: float = 0.85, iters: int = 100
) -> np.ndarray:
    """Power-iteration PageRank over the directed graph, shape (N,)."""
    n = g.num_nodes
    out_deg = compute_out_degree(g)
    out_deg_safe = np.where(out_deg == 0, 1.0, out_deg)
    pr = np.full(n, 1.0 / n, dtype=np.float64)
    src, dst = g.edge_index[0], g.edge_index[1]

    for _ in range(iters):
        contrib = pr[src] / out_deg_safe[src]
        incoming = np.zeros(n, dtype=np.float64)
        np.add.at(incoming, dst, contrib)
        dangling = pr[out_deg == 0].sum()
        pr_new = (1.0 - damping) / n + damping * (incoming + dangling / n)
        if np.abs(pr_new - pr).sum() < 1e-9:
            pr = pr_new
            break

        pr = pr_new

    return pr


def compute_centrality(g: GraphInfo, kind: str = "eigenvector") -> np.ndarray:
    """Eigenvector (power iteration) or closeness (BFS) centrality, shape (N,)."""
    n = g.num_nodes
    if kind == "eigenvector":
        x = np.full(n, 1.0 / n, dtype=np.float64)
        src, dst = g.edge_index[0], g.edge_index[1]

        for _ in range(100):
            nxt = np.zeros(n, dtype=np.float64)
            np.add.at(nxt, dst, x[src])
            norm = np.linalg.norm(nxt)

            if norm < 1e-12:
                break

            nxt = nxt / norm
            if np.abs(nxt - x).sum() < 1e-9:
                x = nxt
                break

            x = nxt
        return x

    if kind == "closeness":
        graph = nx.DiGraph() if g.directed else nx.Graph()
        graph.add_nodes_from(range(n))
        graph.add_edges_from(
            (int(g.edge_index[0, i]), int(g.edge_index[1, i]))
            for i in range(g.edge_index.shape[1])
        )
        cc = nx.closeness_centrality(graph)
        return np.array([cc.get(v, 0.0) for v in range(n)], dtype=np.float64)

    raise ValueError(f"unknown centrality kind {kind!r}; choose eigenvector|closeness")


# Sampling/Estimation
def mc_simulate_spread(
    g: GraphInfo,
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
    seed_bag = [ActionOp("add_node", int(v)) for v in seeds]

    for _ in range(mc_runs):
        sim = build_simulator(g, diffusion_model, seed=int(rng.integers(1 << 30)))
        state = sim.advance(seed_bag)
        for _t in range(horizon):
            if not state.frontier:
                break

            state = sim.advance([])
        totals.append(len(state.infected))

    return float(np.mean(totals))


def compute_marginal_gain(
    g: GraphInfo,
    seeds: list[int],
    v: int,
    diffusion_model: str,
    mc_runs: int = 50,
    horizon: int = 20,
    seed: int = 0,
) -> float:
    """spread(seeds + [v]) - spread(seeds)."""
    base = mc_simulate_spread(g, seeds, diffusion_model, mc_runs, horizon, seed)
    with_v = mc_simulate_spread(
        g, list(seeds) + [int(v)], diffusion_model, mc_runs, horizon, seed
    )

    return float(with_v - base)


def batch_reverse_sample(g: GraphInfo, theta: int, seed: int = 0) -> list[set[int]]:
    """
    Generate theta reverse-reachable sets under the IC live-edge model.

    For a random root, keep each in-edge (u->root) live with prob ic_prob(u->root)
    and BFS backwards over live edges. Each RR set is the set of nodes that could
    have activated the root.
    """
    rng = np.random.default_rng(seed)

    # Build in-adjacency with probabilities once.
    in_adj: dict[int, list[tuple[int, float]]] = {v: [] for v in range(g.num_nodes)}
    for i in range(g.edge_index.shape[1]):
        u, v = int(g.edge_index[0, i]), int(g.edge_index[1, i])
        in_adj[v].append((u, float(g.ic_probs[i])))

    rr_sets: list[set[int]] = []
    for _ in range(theta):
        root = int(rng.integers(g.num_nodes))
        seen = {root}
        stack = [root]
        while stack:
            x = stack.pop()
            for u, p in in_adj[x]:
                if u not in seen and rng.random() < p:
                    seen.add(u)
                    stack.append(u)

        rr_sets.append(seen)

    return rr_sets


def ris_select(rr_sets: list[set[int]], k: int, num_nodes: int) -> list[int]:
    """Greedy max-coverage over RR sets -> top-k seeds (RIS selection rule)."""
    covers: dict[int, set[int]] = {v: set() for v in range(num_nodes)}
    for idx, s in enumerate(rr_sets):
        for v in s:
            covers[v].add(idx)
    chosen: list[int] = []
    covered: set[int] = set()
    for _ in range(k):
        best_v, best_gain = -1, -1
        for v in range(num_nodes):
            if v in chosen:
                continue
            gain = len(covers[v] - covered)
            if gain > best_gain:
                best_gain, best_v = gain, v
        if best_v < 0:
            break
        chosen.append(best_v)
        covered |= covers[best_v]
    return chosen


# --- E. Structural Analysis -------------------------------------------------
def detect_communities(g: GraphInfo) -> dict[int, int]:
    """Label-propagation communities -> {node: community_id}."""
    graph = nx.Graph()
    graph.add_nodes_from(range(g.num_nodes))
    graph.add_edges_from(
        (int(g.edge_index[0, i]), int(g.edge_index[1, i]))
        for i in range(g.edge_index.shape[1])
    )
    communities = nx.community.label_propagation_communities(graph)
    out: dict[int, int] = {}
    for cid, members in enumerate(communities):
        for v in members:
            out[int(v)] = cid
    for v in range(g.num_nodes):
        out.setdefault(v, len(out))
    return out


def allocate_budget(communities: dict[int, int], k: int) -> dict[int, int]:
    """Distribute k seeds across communities proportionally to size (largest-remainder)."""
    sizes: dict[int, int] = {}
    for cid in communities.values():
        sizes[cid] = sizes.get(cid, 0) + 1
    total = sum(sizes.values())
    raw = {c: k * n / total for c, n in sizes.items()}
    alloc = {c: int(np.floor(x)) for c, x in raw.items()}
    remaining = k - sum(alloc.values())
    for c in sorted(raw, key=lambda c: raw[c] - alloc[c], reverse=True)[:remaining]:
        alloc[c] += 1
    return alloc


# --- Look-ahead / sketch / path primitives (heavy-family support) -----------
def estimate_sample_size(
    g: GraphInfo, k: int, epsilon: float = 0.2, delta: float = 0.1
) -> int:
    """RIS sample-size (theta) heuristic; grows ~ n*log(n)/epsilon^2, clamped."""
    import math

    n = g.num_nodes
    return int(max(200, min(20000, (k + 1) * n * math.log(max(n, 2)) / (epsilon**2))))


def sample_live_edge_graph(
    g: GraphInfo, rng: np.random.Generator
) -> dict[int, list[int]]:
    """One IC live-edge realization -> out-adjacency of the live edges."""
    live: dict[int, list[int]] = {v: [] for v in range(g.num_nodes)}
    for i in range(g.edge_index.shape[1]):
        if rng.random() < float(g.ic_probs[i]):
            live[int(g.edge_index[0, i])].append(int(g.edge_index[1, i]))
    return live


def reachable_count(live_adj: dict[int, list[int]], seeds: list[int]) -> int:
    """Nodes reachable from seeds in a live-edge graph (forward BFS)."""
    seen = {int(s) for s in seeds}
    stack = list(seen)
    while stack:
        x = stack.pop()
        for w in live_adj.get(x, ()):
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return len(seen)


def path_influence_scores(g: GraphInfo, max_hops: int = 2) -> np.ndarray:
    """Per-node truncated path-product influence proxy (for path-based methods), shape (N,).

    score[v] = sum over nodes reachable within max_hops of the best path-product of
    IC transmission probabilities from v.
    """
    n = g.num_nodes
    out_adj: dict[int, list[tuple[int, float]]] = {v: [] for v in range(n)}
    for i in range(g.edge_index.shape[1]):
        out_adj[int(g.edge_index[0, i])].append(
            (int(g.edge_index[1, i]), float(g.ic_probs[i]))
        )
    scores = np.zeros(n, dtype=np.float64)
    for s in range(n):
        best = {s: 1.0}
        frontier = [(s, 1.0)]
        for _ in range(max_hops):
            nxt = []
            for x, pp in frontier:
                for w, p in out_adj[x]:
                    cand = pp * p
                    if cand > best.get(w, 0.0):
                        best[w] = cand
                        nxt.append((w, cand))
            frontier = nxt
        scores[s] = sum(v for node, v in best.items() if node != s)
    return scores
