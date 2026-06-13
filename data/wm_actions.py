"""
Action policy: the 6 classical IM seed selectors (each episode's t=0 seed
set), an NDlib Monte-Carlo spread oracle used by CELF / local-search, plus
intermediate-step action injection and counterfactual candidate generation.
"""

import networkx as nx
import numpy as np

from wm_graphs import GraphBundle
from wm_simulator import ActionOp, State, Simulator

SPINE_ALGORITHMS = (
    "random",
    "degree",
    "pagerank",
    "betweenness",
    "celf",
    "local_search",
)


def estimate_spread(
    bundle: GraphBundle,
    seeds: list[int],
    model: str,
    mc_runs: int,
    horizon: int,
    rng: np.random.Generator,
) -> float:
    """
    Mean final activated-node count over mc_runs rollouts: seed at t=0,
    then NULL actions until the frontier dies or horizon is reached.
    """
    if not seeds:
        return 0.0

    # Seed nodes are initialized as add_node actions on the seed node set
    seed_bag = [ActionOp("add_node", int(v)) for v in seeds]
    totals = []

    for run in range(mc_runs):
        simulator = Simulator(
            bundle.nx_graph,
            ic_prob_map=bundle.ic_prob_map,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        simulator.reset(model)
        state = simulator.advance(seed_bag)

        for _ in range(horizon):
            if not state.frontier:
                break
            state = simulator.advance([])

        totals.append(len(state.infected))

    return float(np.mean(totals))


def _out_degree(graph: nx.Graph | nx.DiGraph) -> dict[int, int]:
    degrees = graph.out_degree() if graph.is_directed() else graph.degree()
    return {int(node): int(degree) for node, degree in degrees}


def select_seeds(
    bundle: GraphBundle,
    k: int,
    algorithm: str,
    model: str,
    rng: np.random.Generator,
    mc_runs: int = 16,
    horizon: int = 20,
) -> list[int]:
    graph = bundle.nx_graph

    if algorithm == "random":
        # Get k distinct random nodes
        return sorted(
            int(v) for v in rng.choice(graph.number_of_nodes(), size=k, replace=False)
        )

    if algorithm == "degree":
        # Get the top-k highest-degree nodes
        degrees = _out_degree(graph)
        return sorted(sorted(degrees, key=lambda v: degrees[v], reverse=True)[:k])

    if algorithm == "pagerank":
        # Get k nodes using PageRank (influence by random-walk importance)
        pr = nx.pagerank(graph)
        return sorted(int(v) for v in sorted(pr, key=lambda v: pr[v], reverse=True)[:k])

    if algorithm == "betweenness":
        # Get k nodes using betweenness (nodes that sit on many shortest paths)
        bc = nx.betweenness_centrality(graph)
        return sorted(int(v) for v in sorted(bc, key=lambda v: bc[v], reverse=True)[:k])

    if algorithm == "celf":
        # Get k nodes by building the seed set one node at a time, each round adding the node with the highest marginal gain (how much it increases expected spread on top of the seeds already chosen)

        # Marginal-gain greedy (same seed set as CELF's lazy heap). Telescoping
        # marginal gains mean `base` always equals spread(seeds).
        seeds = []
        base = 0.0
        candidates = list(range(graph.number_of_nodes()))

        for _ in range(k):
            # -inf so the highest-gain candidate is always chosen, even when MC noise
            # makes every marginal gain negative (otherwise best_v can stay None).
            best_v, best_gain = None, float("-inf")
            for v in candidates:
                gain = (
                    estimate_spread(
                        bundle,
                        seeds + [v],
                        model=model,
                        mc_runs=mc_runs,
                        horizon=horizon,
                        rng=rng,
                    )
                    - base
                )

                if gain > best_gain:
                    best_gain, best_v = gain, v

            seeds.append(int(best_v))
            candidates.remove(best_v)
            base += best_gain

        return sorted(seeds)

    if algorithm == "local_search":
        # Get k nodes starting from the degree heuristic, then tries 1-swaps: replace one seed with a non-seed and keep the swap if it improves estimated spread

        seeds = select_seeds(bundle, k=k, algorithm="degree", model=model, rng=rng)
        best = estimate_spread(
            bundle, seeds, model=model, mc_runs=mc_runs, horizon=horizon, rng=rng
        )
        improved, rounds = True, 0
        while improved and rounds < 3:
            improved, rounds = False, rounds + 1
            for i in range(len(seeds)):
                for v in range(graph.number_of_nodes()):
                    if v in seeds:
                        continue
                    trial = list(seeds)
                    trial[i] = v
                    sp = estimate_spread(
                        bundle,
                        trial,
                        model=model,
                        mc_runs=mc_runs,
                        horizon=horizon,
                        rng=rng,
                    )
                    if sp > best:
                        best, seeds, improved = sp, trial, True
                        break
                if improved:
                    break

        return sorted(seeds)

    raise ValueError(f"Unknown algorithm {algorithm}; choose from {SPINE_ALGORITHMS}")


def _random_edge(
    graph: nx.Graph | nx.DiGraph, rng: np.random.Generator
) -> tuple | None:
    edges = list(graph.edges())

    if not edges:
        return None
    u, v = edges[int(rng.integers(len(edges)))]

    return int(u), int(v)


def _random_non_edge(
    graph: nx.Graph | nx.DiGraph, rng: np.random.Generator
) -> tuple | None:
    # A few tries to land a missing edge
    # The graphs are sparse so this almost always hits
    for _ in range(10):
        u, v = int(rng.integers(graph.number_of_nodes())), int(
            rng.integers(graph.number_of_nodes())
        )

        if u != v and not graph.has_edge(u, v):
            return u, v

    return None


def _build_action(
    op: str,
    state: State,
    graph: nx.Graph | nx.DiGraph,
    rng: np.random.Generator,
    weight_range: tuple,
) -> list[ActionOp]:
    """One injected action of type op with a randomly chosen valid target."""
    infected = set(state.infected)

    if op == "add_node":
        # Build an action that randomly adds a susceptible node
        susceptible = [v for v in range(graph.number_of_nodes()) if v not in infected]
        if susceptible:
            return [ActionOp("add_node", int(rng.choice(susceptible)))]
    elif op == "remove_node":
        # Build an action that randomly removes an infected/frontier node
        active = list(state.frontier) if state.frontier else list(state.infected)
        if active:
            return [ActionOp("remove_node", int(rng.choice(active)))]
    elif op == "add_edge":
        # Build an action that randomly adds an edge where there previously was not one
        edge = _random_non_edge(graph, rng)
        if edge is not None:
            return [
                ActionOp(
                    "add_edge", edge[0], edge[1], float(rng.uniform(*weight_range))
                )
            ]
    elif op == "remove_edge":
        # Build an action that randomly removes an edge
        edge = _random_edge(graph, rng)
        if edge is not None:
            return [ActionOp("remove_edge", edge[0], edge[1])]
    elif op == "set_edge_weight":
        # Build an action that randomly modifies a pre-existing edge weight
        edge = _random_edge(graph, rng)
        if edge is not None:
            return [
                ActionOp(
                    "set_edge_weight",
                    edge[0],
                    edge[1],
                    float(rng.uniform(*weight_range)),
                )
            ]

    return []


def sample_injection(
    state: State,
    graph: nx.Graph | nx.DiGraph,
    rng: np.random.Generator,
    p_inject: float,
    action_ops: list[str],
    weight_range: tuple[float, float],
) -> list[ActionOp]:
    """
    NULL (no action, just diffusion dynamics) with prob (1 - p_inject), else a
    single op drawn uniformly from action_ops. Empty action_ops => always NULL.
    """
    if not action_ops or rng.random() >= p_inject:
        return []

    # Pick one enabled op uniformly, then a random valid target from the live graph
    op = action_ops[int(rng.integers(len(action_ops)))]
    return _build_action(op, state, graph, rng, weight_range)


def counterfactual_actions(
    state: State,
    n_nodes: int,
    main_bag: list[ActionOp],
    n: int,
    rng: np.random.Generator,
    action_ops: list[str],
) -> list[list[ActionOp]]:
    """
    Up to n action bags distinct from main_bag and each other. Forks cover
    node ops only (NULL, a random add_node, a random remove_node) so the
    snapshot/restore branch never has to undo an edge mutation.
    """
    infected = set(state.infected)
    susceptible = [v for v in range(n_nodes) if v not in infected]
    active = list(state.frontier) if state.frontier else list(state.infected)

    def key(bag: list[ActionOp]) -> tuple:
        return tuple(
            sorted(
                (a.op, a.target, a.destination if a.destination is not None else -1)
                for a in bag
            )
        )

    seen = {key(main_bag)}
    pool = [[]]

    if "add_node" in action_ops and susceptible:
        pool.append([ActionOp("add_node", int(rng.choice(susceptible)))])

    if "remove_node" in action_ops and active:
        pool.append([ActionOp("remove_node", int(rng.choice(active)))])

    out = []

    for bag in pool:
        if key(bag) in seen:
            continue

        seen.add(key(bag))
        out.append(bag)

        if len(out) == n:
            break

    return out
