"""
Action policy: the 6 classical IM seed selectors (each episode's t=0 seed
set), an NDlib Monte-Carlo spread oracle used by CELF / local-search, plus
intermediate-step action injection and counterfactual candidate generation.
"""

import networkx as nx
import numpy as np

from data.wm_graphs import GraphBundle
from data.wm_simulator import ActionOp, State, Simulator, blocked, spent

spine_algorithms = (
    "random",
    "degree",
    "pagerank",
    "betweenness",
    "celf",
    "local_search",
)

# How an episode's t=0 BLOCKER set is chosen when generating competitive
# (influence-blocking) data. Not the same list as the spine selectors, and the
# difference is the single most useful design fact in
# research/influence_blocking.md §5.4: the degree heuristic "cannot be used for
# influence blocking maximization at all", while PROXIMITY (out-neighbours of the
# negative seeds) is the strong cheap baseline. `degree` is kept so the data
# contains the failure mode too, and `none` so some episodes carry an unopposed
# rumour: the sigma(S_N, empty) reference every prevented-influence number needs.
blocking_selectors = ("none", "random", "proximity", "degree", "pagerank")

# Epidemic control's t=0 dose allocation is the SAME shape as a blocker set,
# "given the outbreak's own sources, choose k nodes that are not sources", so the
# five rules and `select_blockers` cover both tasks rather than each needing its
# own. `proximity` is the ring the outbreak reaches first, which is the family DAVA
# belongs to (research/epidemic_control.md §3.3), and `none` leaves the outbreak
# unprotected and supplies the sigma(outbreak, empty) reference every
# prevented-infections number divides by.
immunizer_selectors = blocking_selectors

seed_upper_bound = 2**31 - 1


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
    seed_bag = [ActionOp("add_node", int(node)) for node in seeds]
    totals = []

    for _ in range(mc_runs):
        simulator = Simulator(
            bundle.nx_graph,
            ic_prob_map=bundle.ic_prob_map,
            seed=int(rng.integers(0, seed_upper_bound)),
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
    num_seeds: int,
    algorithm: str,
    model: str,
    rng: np.random.Generator,
    mc_runs: int = 16,
    horizon: int = 20,
) -> list[int]:
    graph = bundle.nx_graph

    if algorithm == "random":
        # Get num_seeds distinct random nodes
        return sorted(
            int(node)
            for node in rng.choice(
                graph.number_of_nodes(), size=num_seeds, replace=False
            )
        )

    if algorithm == "degree":
        # Get the top-k highest-degree nodes
        degrees = _out_degree(graph)
        ranked = sorted(degrees, key=lambda node: degrees[node], reverse=True)
        return sorted(ranked[:num_seeds])

    if algorithm == "pagerank":
        # Get num_seeds nodes using PageRank (influence by random-walk importance)
        pagerank_scores = nx.pagerank(graph)
        ranked = sorted(
            pagerank_scores, key=lambda node: pagerank_scores[node], reverse=True
        )
        return sorted(int(node) for node in ranked[:num_seeds])

    if algorithm == "betweenness":
        # Get num_seeds nodes using betweenness (nodes that sit on many shortest paths)
        betweenness_scores = nx.betweenness_centrality(graph)
        ranked = sorted(
            betweenness_scores,
            key=lambda node: betweenness_scores[node],
            reverse=True,
        )
        return sorted(int(node) for node in ranked[:num_seeds])

    if algorithm == "celf":
        # Get num_seeds nodes by building the seed set one node at a time, each round adding the node with the highest marginal gain (how much it increases expected spread on top of the seeds already chosen)

        # Marginal-gain greedy (same seed set as CELF's lazy heap). Telescoping
        # marginal gains mean `base_spread` always equals spread(seeds).
        seeds = []
        base_spread = 0.0
        candidates = list(range(graph.number_of_nodes()))

        for _ in range(num_seeds):
            # -inf so the highest-gain candidate is always chosen, even when MC noise
            # makes every marginal gain negative (otherwise best_node can stay None).
            best_node, best_gain = None, float("-inf")
            for node in candidates:
                gain = (
                    estimate_spread(
                        bundle,
                        seeds + [node],
                        model=model,
                        mc_runs=mc_runs,
                        horizon=horizon,
                        rng=rng,
                    )
                    - base_spread
                )

                if gain > best_gain:
                    best_gain, best_node = gain, node

            seeds.append(int(best_node))
            candidates.remove(best_node)
            base_spread += best_gain

        return sorted(seeds)

    if algorithm == "local_search":
        # Get num_seeds nodes starting from the degree heuristic, then try 1-swaps: replace one seed with a non-seed and keep the swap if it improves estimated spread

        seeds = select_seeds(
            bundle, num_seeds=num_seeds, algorithm="degree", model=model, rng=rng
        )
        best_spread = estimate_spread(
            bundle, seeds, model=model, mc_runs=mc_runs, horizon=horizon, rng=rng
        )
        improved, rounds = True, 0

        while improved and rounds < 3:
            improved, rounds = False, rounds + 1
            for position in range(len(seeds)):
                for node in range(graph.number_of_nodes()):
                    if node in seeds:
                        continue
                    trial = list(seeds)
                    trial[position] = node
                    spread = estimate_spread(
                        bundle,
                        trial,
                        model=model,
                        mc_runs=mc_runs,
                        horizon=horizon,
                        rng=rng,
                    )
                    if spread > best_spread:
                        best_spread, seeds, improved = spread, trial, True
                        break
                if improved:
                    break

        return sorted(seeds)

    raise ValueError(f"Unknown algorithm {algorithm}; choose from {spine_algorithms}")


def select_blockers(
    bundle: GraphBundle,
    negative_seeds: list[int],
    num_blockers: int,
    algorithm: str,
    rng: np.random.Generator,
) -> list[int]:
    """
    The positive seed set an episode commits at t=0, given the rumour's own seeds.

    Every rule here excludes `negative_seeds` themselves: a node the rumour already
    owns is committed, so seeding it positively is a no-op the simulator drops and a
    unit of budget the episode never spent.
    """
    graph = bundle.nx_graph
    excluded = {int(node) for node in negative_seeds}
    candidates = [node for node in range(graph.number_of_nodes()) if node not in excluded]

    if algorithm == "none" or num_blockers <= 0 or not candidates:
        return []

    if algorithm == "random":
        chosen = rng.choice(len(candidates), size=min(num_blockers, len(candidates)), replace=False)
        return sorted(int(candidates[int(index)]) for index in chosen)

    if algorithm == "proximity":
        # §5.4: pick the out-neighbours of the negative seeds, the nodes the rumour
        # reaches FIRST: ranked by degree, topped up by hop 2 and then by degree
        degrees = _out_degree(graph)
        ring = []
        for node in sorted(excluded):
            if node not in graph:
                continue

            ring += [int(other) for other in graph.successors(node)] if graph.is_directed() else [
                int(other) for other in graph.neighbors(node)
            ]

        ordered = sorted(
            dict.fromkeys(node for node in ring if node not in excluded),
            key=lambda node: degrees.get(node, 0),
            reverse=True,
        )
        chosen = ordered[:num_blockers]

        if len(chosen) < num_blockers:
            picked = set(chosen) | excluded
            for node in sorted(degrees, key=lambda node: degrees[node], reverse=True):
                if len(chosen) >= num_blockers:
                    break
                if node not in picked:
                    chosen.append(int(node))

        return sorted(chosen)

    if algorithm in ("degree", "pagerank"):
        scores = (
            _out_degree(graph)
            if algorithm == "degree"
            else {int(node): value for node, value in nx.pagerank(graph).items()}
        )
        ranked = sorted(candidates, key=lambda node: scores.get(node, 0.0), reverse=True)
        return sorted(int(node) for node in ranked[:num_blockers])

    raise ValueError(
        f"unknown blocking selector {algorithm!r}; choose from {blocking_selectors}"
    )


# Same function, task-appropriate name at the call site: an epidemic episode's t=0
# dose allocation is a blocker set given the outbreak's sources
select_immunizers = select_blockers


def _random_edge(
    graph: nx.Graph | nx.DiGraph, rng: np.random.Generator
) -> tuple | None:
    edges = list(graph.edges())

    if not edges:
        return None
    source, destination = edges[int(rng.integers(len(edges)))]

    return int(source), int(destination)


def _random_non_edge(
    graph: nx.Graph | nx.DiGraph, rng: np.random.Generator
) -> tuple | None:
    # A few tries to land a missing edge
    # The graphs are sparse so this almost always hits
    for _ in range(10):
        source, destination = (
            int(rng.integers(graph.number_of_nodes())),
            int(rng.integers(graph.number_of_nodes())),
        )

        if source != destination and not graph.has_edge(source, destination):
            return source, destination

    return None


def delete_node_bag(
    graph: nx.Graph | nx.DiGraph, node: int
) -> list[ActionOp]:
    """
    Node deletion written in the existing op set: remove_node(v) plus one
    remove_edge per incident arc. This is what `blocked` semantics means
    structurally, and it needs no sixth op.

    The edge ops go in the BAG rather than into the simulator, which is what
    keeps every downstream consumer correct for free: build_features marks
    CH_EDGE on both endpoints, reconstruct_episode_adjacency replays the
    removals so each step sees the post-deletion adjacency, and both rollout
    paths already call apply_edge_ops.

    Both orientations are emitted for undirected graphs because the edge store
    is keyed by arc; the extra key is a no-op wherever it does not exist.
    """
    node = int(node)
    bag = [ActionOp("remove_node", node)]

    if graph.is_directed():
        incident = [(node, int(other)) for other in graph.successors(node)]
        incident += [(int(other), node) for other in graph.predecessors(node)]
    else:
        neighbours = [int(other) for other in graph.neighbors(node)]
        incident = [(node, other) for other in neighbours]
        incident += [(other, node) for other in neighbours]

    return bag + [
        ActionOp("remove_edge", source, destination)
        for source, destination in incident
    ]


def _build_action(
    op: str,
    state: State,
    graph: nx.Graph | nx.DiGraph,
    rng: np.random.Generator,
    weight_range: tuple,
    remove_semantics: str = spent,
    blocked_nodes: set[int] | None = None,
) -> list[ActionOp]:
    """One injected action of type op with a randomly chosen valid target."""
    # Under a competitive task a node the POSITIVE cascade already owns is committed
    # too, so it is no more seedable than a negatively-infected one. Empty for every
    # single-cascade task, which leaves those pools exactly as they were. A blocked
    # node is out of the graph and is not in `state` at all, so it has to be
    # excluded here or the injection records an add_node the simulator ignores.
    infected = set(state.infected) | set(state.pos_infected) | set(blocked_nodes or ())

    if op == "add_node":
        # Build an action that randomly adds a susceptible node
        susceptible = [
            node for node in range(graph.number_of_nodes()) if node not in infected
        ]
        if susceptible:
            return [ActionOp("add_node", int(rng.choice(susceptible)))]
    elif op == "remove_node":
        if remove_semantics == blocked:
            # Containment blocks a node BEFORE the cascade reaches it, so the
            # target pool is the susceptibles, not the active nodes that `spent`
            # draws from. Without this the data would carry no example of the
            # semantics at all.
            # ponytail: uniform over susceptibles, so most blocks land far from
            # the frontier and change nothing. Restrict to susceptible
            # neighbours of the frontier if the containment signal is too sparse.
            pool = [
                node for node in range(graph.number_of_nodes()) if node not in infected
            ]
            if pool:
                return delete_node_bag(graph, int(rng.choice(pool)))
        else:
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
    remove_semantics: str = spent,
    blocked_nodes: set[int] | None = None,
) -> list[ActionOp]:
    """
    NULL (no action, just diffusion dynamics) with prob (1 - p_inject), else a
    single op drawn uniformly from action_ops. Empty action_ops => always NULL.
    """
    if not action_ops or rng.random() >= p_inject:
        return []

    # Pick one enabled op uniformly, then a random valid target from the live graph
    op = action_ops[int(rng.integers(len(action_ops)))]
    return _build_action(
        op, state, graph, rng, weight_range, remove_semantics, blocked_nodes
    )


def counterfactual_actions(
    state: State,
    graph: nx.Graph | nx.DiGraph,
    main_bag: list[ActionOp],
    count: int,
    rng: np.random.Generator,
    action_ops: list[str],
    remove_semantics: str = spent,
    blocked_nodes: set[int] | None = None,
) -> list[list[ActionOp]]:
    """
    Up to `count` action bags distinct from main_bag and each other: NULL, a
    random add_node, and a random remove_node.

    Under `blocked` the removal fork is a full node-deletion bag, so it mutates
    the graph as well as the status: `Simulator.restore()` rewinds status and the
    blocked set but not the graph. The caller must therefore pair every fork with
    `Simulator.revert_edges(bag)`, which re-adds the stripped arcs at their
    original probabilities. Skipping the fork instead (what this used to do) left
    a containment dataset with NO two actions from the same state, which is
    exactly what `action_sensitivity` measures: it read 0.0.
    """
    num_nodes = graph.number_of_nodes()
    infected = set(state.infected) | set(state.pos_infected) | set(blocked_nodes or ())
    susceptible = [node for node in range(num_nodes) if node not in infected]
    active = list(state.frontier) if state.frontier else list(state.infected)

    def key(bag: list[ActionOp]) -> tuple:
        return tuple(
            sorted(
                (
                    action.op,
                    action.target,
                    action.destination if action.destination is not None else -1,
                )
                for action in bag
            )
        )

    seen = {key(main_bag)}
    pool = [[]]

    if "add_node" in action_ops and susceptible:
        pool.append([ActionOp("add_node", int(rng.choice(susceptible)))])

    if "remove_node" in action_ops:
        if remove_semantics == blocked:
            # Containment blocks a node BEFORE the cascade reaches it, so the fork
            # draws from the susceptibles the main-branch injection draws from
            if susceptible:
                pool.append(delete_node_bag(graph, int(rng.choice(susceptible))))
        elif active:
            pool.append([ActionOp("remove_node", int(rng.choice(active)))])

    bags = []

    for bag in pool:
        if key(bag) in seen:
            continue

        seen.add(key(bag))
        bags.append(bag)

        if len(bags) == count:
            break

    return bags
