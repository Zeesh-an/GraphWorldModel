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
    Mean final activated-node count over ``mc_runs`` rollouts: seed at t=0,
    then NULL actions until the frontier dies or ``horizon`` is reached.
    """
    if not seeds:
        return 0.0

    seed_bag = [ActionOp("add_seed", int(v)) for v in seeds]
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
            best_v, best_gain = None, -1.0
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


def sample_injection(
    state: State,
    n_nodes: int,
    rng: np.random.Generator,
    p_inject: float,
    p_add: float,
    p_remove: float,
) -> list[ActionOp]:
    """
    NULL (no action, just diffusion dynamics) with prob (1 - p_inject), else add_seed (random susceptible) or
    remove_node (random active), mixed by p_add:p_remove.
    """
    if rng.random() >= p_inject:
        return []

    # Get the action to inject at an intermediate step
    infected = set(state.infected)
    susceptible = [v for v in range(n_nodes) if v not in infected]
    active = list(state.frontier) if state.frontier else list(state.infected)
    do_add = (
        rng.random() < (p_add / (p_add + p_remove)) if (p_add + p_remove) > 0 else True
    )

    if do_add and susceptible:
        return [ActionOp("add_seed", int(rng.choice(susceptible)))]

    if not do_add and active:
        return [ActionOp("remove_node", int(rng.choice(active)))]

    return []


def counterfactual_actions(
    state: State,
    n_nodes: int,
    main_bag: list[ActionOp],
    n: int,
    rng: np.random.Generator,
) -> list[list[ActionOp]]:
    """
    Up to n action bags distinct from ``main_bag`` and each other
    (pool: NULL, a random add_seed, a random remove_node).
    """
    infected = set(state.infected)
    susceptible = [v for v in range(n_nodes) if v not in infected]
    active = list(state.frontier) if state.frontier else list(state.infected)

    def key(bag: list[ActionOp]) -> tuple[tuple[str, int], ...]:
        return tuple(sorted((a.op, a.target) for a in bag))

    seen = {key(main_bag)}
    pool = [[]]

    if susceptible:
        pool.append([ActionOp("add_seed", int(rng.choice(susceptible)))])

    if active:
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
