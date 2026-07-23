"""
Named classical IM algorithms, composed from primitives.

Every algorithm has the signature (graph, budget, diffusion_model, **kw) -> list[int]
and returns a seed set of size `budget`. These are the callable surface offered to
the coding agent; it may call one directly or compose several across timesteps.
"""

import heapq
import networkx as nx
import numpy as np

from coding_agent.types import GraphInfo
from coding_agent.tools import primitives

min_temperature = 1e-6


def _top_k_by_score(scores: np.ndarray, count: int) -> list[int]:
    return [int(node) for node in np.argsort(-scores)[:count]]


def high_degree(
    graph: GraphInfo, budget: int, diffusion_model: str = "IC", **_
) -> list[int]:
    """Top-k highest total-degree nodes (Degree heuristic)."""
    return primitives.get_top_degree_nodes(graph, budget)


def weighted_degree(
    graph: GraphInfo, budget: int, diffusion_model: str = "IC", **_
) -> list[int]:
    """Top-k by summed outgoing IC transmission probability."""
    return _top_k_by_score(primitives.compute_weighted_degree(graph), budget)


def degree_discount(
    graph: GraphInfo, budget: int, diffusion_model: str = "IC", **_
) -> list[int]:
    """DegreeDiscount (Chen et al. 2009): discount a node's degree for already-chosen neighbors."""
    discounted = primitives.compute_degree(graph)
    chosen = []

    for _ in range(budget):
        # -inf mask so an already-chosen node can never win argmax, however far
        # the discounting pushes the remaining scores down.
        node = int(
            np.argmax(
                [
                    discounted[candidate] if candidate not in chosen else float("-inf")
                    for candidate in range(graph.num_nodes)
                ]
            )
        )
        chosen.append(node)
        for neighbor in graph.out_neighbors(node) + graph.in_neighbors(node):
            discounted[neighbor] -= 1

    return chosen


def pagerank_seeds(
    graph: GraphInfo, budget: int, diffusion_model: str = "IC", **_
) -> list[int]:
    """Top-k PageRank nodes."""
    return _top_k_by_score(primitives.compute_pagerank(graph), budget)


def vanilla_greedy(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    mc_runs: int = 30,
    horizon: int = 20,
    **_,
) -> list[int]:
    """Greedy marginal-gain (Kempe et al. 2003)."""
    seeds = []
    for _ in range(budget):
        best_node, best_gain = -1, float("-inf")
        for node in range(graph.num_nodes):
            if node in seeds:
                continue
            gain = primitives.compute_marginal_gain(
                graph, seeds, node, diffusion_model, mc_runs, horizon
            )
            if gain > best_gain:
                best_gain, best_node = gain, node

        seeds.append(best_node)

    return seeds


def celf(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    mc_runs: int = 30,
    horizon: int = 20,
    **_,
) -> list[int]:
    """CELF (Leskovec et al. 2007): lazy-forward greedy with a max-heap of marginal gains."""
    seeds = []
    heap = []  # (-gain, node, last_updated_round)
    for node in range(graph.num_nodes):
        gain = primitives.mc_simulate_spread(
            graph, [node], diffusion_model, mc_runs, horizon
        )
        heapq.heappush(heap, (-gain, node, 0))

    for round_index in range(1, budget + 1):
        while True:
            _, node, last_updated = heapq.heappop(heap)
            if last_updated == round_index:
                seeds.append(node)
                break

            fresh_gain = primitives.mc_simulate_spread(
                graph, seeds + [node], diffusion_model, mc_runs, horizon
            ) - primitives.mc_simulate_spread(
                graph, seeds, diffusion_model, mc_runs, horizon
            )
            heapq.heappush(heap, (-fresh_gain, node, round_index))

    return seeds


def ris_basic(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    theta: int = 2000,
    **_,
) -> list[int]:
    """Basic Reverse Influence Sampling (Borgs et al. 2014). IC only."""
    rr_sets = primitives.batch_reverse_sample(graph, theta=theta, seed=0)
    seeds = primitives.ris_select(rr_sets, budget, graph.num_nodes)

    # Pad with degree if coverage ran out.
    if len(seeds) < budget:
        for node in primitives.get_top_degree_nodes(graph, graph.num_nodes):
            if node not in seeds:
                seeds.append(node)
            if len(seeds) == budget:
                break

    return seeds


def community_im(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    **_,
) -> list[int]:
    """Community-IM: split budget across label-propagation communities, degree within each."""
    communities = primitives.detect_communities(graph)
    allocation = primitives.allocate_budget(communities, budget)
    members = {}
    for node, community_id in communities.items():
        members.setdefault(community_id, []).append(node)

    degrees = primitives.compute_degree(graph)
    seeds = []
    for community_id, community_budget in allocation.items():
        ranked = sorted(
            members[community_id], key=lambda node: degrees[node], reverse=True
        )
        seeds.extend(ranked[:community_budget])

    return seeds[:budget]


def pagerank_greedy(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    pool: int = 30,
    mc_runs: int = 30,
    horizon: int = 20,
    **_,
) -> list[int]:
    """Hybrid: narrow to top PageRank candidates, then greedy marginal-gain over them."""
    candidates = _top_k_by_score(
        primitives.compute_pagerank(graph), min(pool, graph.num_nodes)
    )

    seeds = []
    for _ in range(budget):
        best_node, best_gain = -1, float("-inf")
        for node in candidates:
            if node in seeds:
                continue
            gain = primitives.compute_marginal_gain(
                graph, seeds, node, diffusion_model, mc_runs, horizon
            )
            if gain > best_gain:
                best_gain, best_node = gain, node

        if best_node < 0:
            break

        seeds.append(best_node)

    return _pad_seeds(seeds, graph, budget)


def hill_climbing(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    mc_runs: int = 30,
    horizon: int = 20,
    rounds: int = 3,
    **_,
) -> list[int]:
    """Local search: start from degree, 1-swap while spread improves."""
    seeds = primitives.get_top_degree_nodes(graph, budget)
    best_spread = primitives.mc_simulate_spread(
        graph, seeds, diffusion_model, mc_runs, horizon
    )

    for _ in range(rounds):
        improved = False
        for position in range(len(seeds)):
            for node in range(graph.num_nodes):
                if node in seeds:
                    continue
                trial = list(seeds)
                trial[position] = node
                spread = primitives.mc_simulate_spread(
                    graph, trial, diffusion_model, mc_runs, horizon
                )
                if spread > best_spread:
                    best_spread, seeds, improved = spread, trial, True
                    break
            if improved:
                break
        if not improved:
            break

    return seeds


# Shared selection helpers (used by several families)
def _pad_seeds(seeds: list[int], graph: GraphInfo, budget: int) -> list[int]:
    """Dedupe + pad to budget with high-degree fillers."""
    padded = list(dict.fromkeys(int(seed) for seed in seeds))
    if len(padded) < budget:
        for node in primitives.get_top_degree_nodes(graph, graph.num_nodes):
            if node not in padded:
                padded.append(node)
            if len(padded) == budget:
                break

    return padded[:budget]


def _greedy_discount_select(
    graph: GraphInfo, scores: np.ndarray, budget: int, discount: float = 0.5
) -> list[int]:
    """Pick top-k by score, discounting an out-neighbor's score when its source is chosen."""
    discounted = scores.astype(float).copy()
    chosen = []

    for _ in range(budget):
        node = int(
            np.argmax(
                [
                    discounted[candidate] if candidate not in chosen else float("-inf")
                    for candidate in range(graph.num_nodes)
                ]
            )
        )
        chosen.append(node)

        for neighbor in graph.out_neighbors(node):
            discounted[neighbor] *= discount

    return chosen


# Greedy family (additions)
def celf_pp(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    mc_runs: int = 30,
    horizon: int = 20,
    **_,
) -> list[int]:
    """CELF++ (Goyal et al. 2011): CELF with a secondary cache (simplified; selection = CELF)."""
    seeds = []
    heap = []
    for node in range(graph.num_nodes):
        gain = primitives.mc_simulate_spread(
            graph, [node], diffusion_model, mc_runs, horizon
        )
        heapq.heappush(heap, (-gain, node, 0))

    for round_index in range(1, budget + 1):
        while True:
            _, node, last_updated = heapq.heappop(heap)
            if last_updated == round_index:
                seeds.append(node)
                break

            fresh_gain = primitives.mc_simulate_spread(
                graph, seeds + [node], diffusion_model, mc_runs, horizon
            ) - primitives.mc_simulate_spread(
                graph, seeds, diffusion_model, mc_runs, horizon
            )
            heapq.heappush(heap, (-fresh_gain, node, round_index))

    return seeds


def adaptive_greedy(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    base_mc: int = 10,
    refine_mc: int = 40,
    horizon: int = 20,
    refine_top: int = 5,
    **_,
) -> list[int]:
    """Adaptive Greedy: coarse MC to rank, refined MC on the top few (adaptive sample count)."""
    seeds = []
    for _ in range(budget):
        coarse = sorted(
            (
                (
                    primitives.compute_marginal_gain(
                        graph, seeds, node, diffusion_model, base_mc, horizon
                    ),
                    node,
                )
                for node in range(graph.num_nodes)
                if node not in seeds
            ),
            reverse=True,
        )

        best_node, best_gain = coarse[0][1], float("-inf")
        for _, node in coarse[:refine_top]:
            gain = primitives.compute_marginal_gain(
                graph, seeds, node, diffusion_model, refine_mc, horizon
            )
            if gain > best_gain:
                best_gain, best_node = gain, node

        seeds.append(best_node)

    return seeds


# Centrality (additions)
def eigenvector_seeds(
    graph: GraphInfo, budget: int, diffusion_model: str = "IC", **_
) -> list[int]:
    """Top-k eigenvector-centrality nodes."""
    return _top_k_by_score(primitives.compute_centrality(graph, "eigenvector"), budget)


def closeness_seeds(
    graph: GraphInfo, budget: int, diffusion_model: str = "IC", **_
) -> list[int]:
    """Top-k closeness-centrality nodes."""
    return _top_k_by_score(primitives.compute_centrality(graph, "closeness"), budget)


# RIS family (additions; IC)
def tim(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    epsilon: float = 0.2,
    **_,
) -> list[int]:
    """TIM/TIM+ (Tang et al. 2014): RIS with an estimated sample size."""
    theta = primitives.estimate_sample_size(graph, budget, epsilon=epsilon)
    rr_sets = primitives.batch_reverse_sample(graph, theta=theta, seed=0)

    return _pad_seeds(
        primitives.ris_select(rr_sets, budget, graph.num_nodes), graph, budget
    )


def imm(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    epsilon: float = 0.2,
    **_,
) -> list[int]:
    """IMM (Tang et al. 2015): two-phase sample-size refinement over RIS (simplified)."""
    initial_theta = primitives.estimate_sample_size(graph, budget, epsilon=epsilon * 2)
    rr_sets = primitives.batch_reverse_sample(graph, theta=initial_theta, seed=0)

    theta = max(
        initial_theta, primitives.estimate_sample_size(graph, budget, epsilon=epsilon)
    )

    if theta > initial_theta:
        rr_sets = rr_sets + primitives.batch_reverse_sample(
            graph, theta=theta - initial_theta, seed=1
        )

    return _pad_seeds(
        primitives.ris_select(rr_sets, budget, graph.num_nodes), graph, budget
    )


def ssa(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    initial_theta: int = 500,
    max_doublings: int = 4,
    stability: float = 0.9,
    **_,
) -> list[int]:
    """SSA/D-SSA (Nguyen et al. 2016): doubling RIS until the top-k stabilizes (simplified)."""
    theta = initial_theta
    rr_sets = primitives.batch_reverse_sample(graph, theta=theta, seed=0)
    previous = set(primitives.ris_select(rr_sets, budget, graph.num_nodes))

    for iteration in range(1, max_doublings + 1):
        theta *= 2
        rr_sets = rr_sets + primitives.batch_reverse_sample(
            graph, theta=theta, seed=iteration
        )
        current = set(primitives.ris_select(rr_sets, budget, graph.num_nodes))

        if len(current & previous) >= max(1, int(stability * budget)):
            previous = current
            break

        previous = current

    return _pad_seeds(sorted(previous), graph, budget)


def filtered_ris(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    theta: int = 2000,
    min_size: int = 2,
    **_,
) -> list[int]:
    """Filtered RIS: drop tiny RR sets before coverage selection."""
    rr_sets = [
        rr_set
        for rr_set in primitives.batch_reverse_sample(graph, theta=theta, seed=0)
        if len(rr_set) >= min_size
    ]

    if not rr_sets:
        rr_sets = primitives.batch_reverse_sample(graph, theta=theta, seed=0)

    return _pad_seeds(
        primitives.ris_select(rr_sets, budget, graph.num_nodes), graph, budget
    )


# Path-based (simplified via truncated path-sum)
def sp1m(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    max_hops: int = 3,
    **_,
) -> list[int]:
    """SP1M/SPM (Kimura & Saito 2006): top-k by shortest-path influence (truncated path-sum)."""
    return _top_k_by_score(
        primitives.path_influence_scores(graph, max_hops=max_hops), budget
    )


def mia_pmia(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    max_hops: int = 2,
    **_,
) -> list[int]:
    """MIA/PMIA (Chen et al. 2010): local-influence-tree score (simplified), discounted selection."""
    return _greedy_discount_select(
        graph, primitives.path_influence_scores(graph, max_hops=max_hops), budget
    )


def ldag(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    max_hops: int = 3,
    **_,
) -> list[int]:
    """LDAG (Chen et al. 2010): local-DAG influence (simplified deeper path-sum), discounted selection."""
    return _greedy_discount_select(
        graph, primitives.path_influence_scores(graph, max_hops=max_hops), budget
    )


# Sketch-based (over sampled live-edge graphs)
def static_greedy(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    snapshots: int = 20,
    **_,
) -> list[int]:
    """StaticGreedy (Cheng et al. 2014): greedy over a fixed set of live-edge snapshots."""
    rng = np.random.default_rng(0)
    live_graphs = [
        primitives.sample_live_edge_graph(graph, rng) for _ in range(snapshots)
    ]

    seeds = []
    for _ in range(budget):
        best_node, best_gain = -1, -1.0
        for node in range(graph.num_nodes):
            if node in seeds:
                continue

            gain = float(
                np.mean(
                    [
                        primitives.reachable_count(live_graph, seeds + [node])
                        - primitives.reachable_count(live_graph, seeds)
                        for live_graph in live_graphs
                    ]
                )
            )
            if gain > best_gain:
                best_gain, best_node = gain, node

        seeds.append(best_node)

    return seeds


def skim(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    snapshots: int = 32,
    **_,
) -> list[int]:
    """SKIM (Cohen et al. 2014): rank by average single-seed reachability over sketches (simplified)."""
    rng = np.random.default_rng(0)

    live_graphs = [
        primitives.sample_live_edge_graph(graph, rng) for _ in range(snapshots)
    ]

    scores = np.array(
        [
            float(
                np.mean(
                    [
                        primitives.reachable_count(live_graph, [node])
                        for live_graph in live_graphs
                    ]
                )
            )
            for node in range(graph.num_nodes)
        ]
    )

    return _greedy_discount_select(graph, scores, budget)


# Community (additions)
def cofim(graph: GraphInfo, budget: int, diffusion_model: str = "IC", **_) -> list[int]:
    """CoFIM (Zhang et al. 2014): per-community budget + degree with cross-community bridge bonus."""
    communities = primitives.detect_communities(graph)
    allocation = primitives.allocate_budget(communities, budget)
    members = {}
    for node, community_id in communities.items():
        members.setdefault(community_id, []).append(node)

    degrees = primitives.compute_degree(graph)

    def bridge_score(node: int) -> float:
        neighbors = graph.out_neighbors(node) + graph.in_neighbors(node)
        cross_community = sum(
            1 for neighbor in neighbors if communities[neighbor] != communities[node]
        )
        return float(degrees[node] + cross_community)

    seeds = []
    for community_id, community_budget in allocation.items():
        seeds.extend(
            sorted(members[community_id], key=bridge_score, reverse=True)[
                :community_budget
            ]
        )

    return _pad_seeds(seeds, graph, budget)


def community_ris(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    theta: int = 2000,
    **_,
) -> list[int]:
    """Community-RIS hybrid: RR-set coverage selection restricted to per-community budgets."""
    communities = primitives.detect_communities(graph)
    allocation = primitives.allocate_budget(communities, budget)
    rr_sets = primitives.batch_reverse_sample(graph, theta=theta, seed=0)

    members = {}
    for node, community_id in communities.items():
        members.setdefault(community_id, []).append(node)

    covers = {node: set() for node in range(graph.num_nodes)}
    for rr_index, rr_set in enumerate(rr_sets):
        for node in rr_set:
            covers[node].add(rr_index)

    seeds = []
    covered = set()
    for community_id, community_budget in allocation.items():
        for _ in range(community_budget):
            best_node, best_gain = -1, -1
            for node in members[community_id]:
                if node in seeds:
                    continue
                gain = len(covers[node] - covered)
                if gain > best_gain:
                    best_gain, best_node = gain, node

            if best_node >= 0:
                seeds.append(best_node)
                covered |= covers[best_node]

    return _pad_seeds(seeds, graph, budget)


# Metaheuristics (additions)
def simulated_annealing(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    mc_runs: int = 20,
    horizon: int = 20,
    iters: int = 40,
    cooling: float = 0.9,
    **_,
) -> list[int]:
    """Simulated annealing over seed sets (degree init, swap moves, geometric cooling)."""
    rng = np.random.default_rng(0)
    seeds = primitives.get_top_degree_nodes(graph, budget)
    current_spread = best_spread = primitives.mc_simulate_spread(
        graph, seeds, diffusion_model, mc_runs, horizon
    )
    best_seeds = list(seeds)
    temperature = 1.0

    for _ in range(iters):
        candidates = [node for node in range(graph.num_nodes) if node not in seeds]
        if not candidates:
            break

        trial = list(seeds)
        trial[int(rng.integers(len(seeds)))] = int(rng.choice(candidates))
        spread = primitives.mc_simulate_spread(
            graph, trial, diffusion_model, mc_runs, horizon
        )

        if spread > current_spread or rng.random() < np.exp(
            (spread - current_spread) / max(temperature, min_temperature)
        ):
            seeds, current_spread = trial, spread
            if spread > best_spread:
                best_spread, best_seeds = spread, list(trial)

        temperature *= cooling

    return best_seeds


def genetic_algorithm(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    population_size: int = 8,
    generations: int = 6,
    mc_runs: int = 15,
    horizon: int = 20,
    mutation_rate: float = 0.3,
    **_,
) -> list[int]:
    """Genetic algorithm over seed sets (degree-biased init, crossover + mutation, elitism)."""
    rng = np.random.default_rng(0)
    pool = primitives.get_top_degree_nodes(
        graph, min(graph.num_nodes, max(budget * 4, 10))
    )

    def fill(child: list[int]) -> list[int]:
        child = list(dict.fromkeys(child))

        while len(child) < budget:
            node = int(rng.choice(pool))
            if node not in child:
                child.append(node)

        return child[:budget]

    def fitness(individual: list[int]) -> float:
        return primitives.mc_simulate_spread(
            graph, list(individual), diffusion_model, mc_runs, horizon
        )

    population = [
        fill(list(rng.choice(pool, size=budget, replace=False)))
        for _ in range(population_size)
    ]
    scored = sorted(
        ((fitness(individual), individual) for individual in population),
        key=lambda pair: pair[0],
        reverse=True,
    )

    for _ in range(generations):
        survivors = [
            individual for _, individual in scored[: max(2, population_size // 2)]
        ]
        children = []

        while len(children) < population_size - len(survivors):
            parent_a = survivors[int(rng.integers(len(survivors)))]
            parent_b = survivors[int(rng.integers(len(survivors)))]

            cut = budget // 2
            child = fill(parent_a[:cut] + parent_b[cut:])

            if rng.random() < mutation_rate:
                child[int(rng.integers(budget))] = int(rng.choice(pool))
                child = fill(child)

            children.append(child)

        population = survivors + children
        scored = sorted(
            ((fitness(individual), individual) for individual in population),
            key=lambda pair: pair[0],
            reverse=True,
        )

    return list(scored[0][1])


# Hybrids (additions)
def degree_ris_refine(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    theta: int = 2000,
    **_,
) -> list[int]:
    """Degree init refined by RR-set coverage swaps (Degree + RIS)."""
    seeds = primitives.get_top_degree_nodes(graph, budget)
    rr_sets = primitives.batch_reverse_sample(graph, theta=theta, seed=0)

    covers = {node: set() for node in range(graph.num_nodes)}
    for rr_index, rr_set in enumerate(rr_sets):
        for node in rr_set:
            covers[node].add(rr_index)

    covered = set()
    for seed in seeds:
        covered |= covers[seed]

    for position, seed in enumerate(seeds):
        rest = covered - covers[seed]
        for node in range(graph.num_nodes):
            if node in seeds:
                continue

            if len(covers[node] - rest) > len(covers[seed] - rest):
                covered = rest | covers[node]
                seeds[position] = node
                break

    return seeds


def celf_local_search(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    mc_runs: int = 30,
    horizon: int = 20,
    **_,
) -> list[int]:
    """CELF seed set refined by 1-swap local search (CELF + LocalSearch)."""
    seeds = celf(graph, budget, diffusion_model, mc_runs, horizon)
    best_spread = primitives.mc_simulate_spread(
        graph, seeds, diffusion_model, mc_runs, horizon
    )

    for position in range(len(seeds)):
        for node in range(graph.num_nodes):
            if node in seeds:
                continue

            trial = list(seeds)
            trial[position] = node
            spread = primitives.mc_simulate_spread(
                graph, trial, diffusion_model, mc_runs, horizon
            )

            if spread > best_spread:
                best_spread, seeds = spread, trial
                break

    return seeds


def community_celf(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    mc_runs: int = 20,
    horizon: int = 20,
    **_,
) -> list[int]:
    """Community-scoped marginal-gain greedy (no CELF lazy heap — simplified)."""
    communities = primitives.detect_communities(graph)
    allocation = primitives.allocate_budget(communities, budget)
    members = {}
    for node, community_id in communities.items():
        members.setdefault(community_id, []).append(node)

    seeds = []
    for community_id, community_budget in allocation.items():
        local_seeds = []

        for _ in range(community_budget):
            best_node, best_gain = -1, float("-inf")
            for node in members[community_id]:
                if node in local_seeds:
                    continue

                gain = primitives.compute_marginal_gain(
                    graph, seeds + local_seeds, node, diffusion_model, mc_runs, horizon
                )

                if gain > best_gain:
                    best_gain, best_node = gain, node

            if best_node >= 0:
                local_seeds.append(best_node)

        seeds.extend(local_seeds)

    return _pad_seeds(seeds, graph, budget)


def _neighbor_sets(graph: GraphInfo) -> list[set[int]]:
    neighbors = [set() for _ in range(graph.num_nodes)]
    for edge in range(graph.edge_index.shape[1]):
        source = int(graph.edge_index[0, edge])
        target = int(graph.edge_index[1, edge])
        neighbors[source].add(target)
        neighbors[target].add(source)

    return neighbors


def voterank(
    graph: GraphInfo, budget: int, diffusion_model: str = "IC", **_
) -> list[int]:
    """VoteRank (Zhang et al. 2016): iterative voting where a winner's neighbors lose voting power — anti-overlap seed selection."""
    neighbors = _neighbor_sets(graph)
    voting_ability = np.ones(graph.num_nodes)
    average_degree = max(
        float(np.mean([len(nbrs) for nbrs in neighbors])), 1.0
    )

    chosen = []
    for _ in range(min(budget, graph.num_nodes)):
        votes = np.array(
            [
                sum(voting_ability[neighbor] for neighbor in neighbors[node])
                for node in range(graph.num_nodes)
            ]
        )
        votes[chosen] = float("-inf")

        winner = int(np.argmax(votes))
        chosen.append(winner)

        # The winner stops voting; its neighbors' voting power decays by 1/<k>
        voting_ability[winner] = 0.0
        for neighbor in neighbors[winner]:
            voting_ability[neighbor] = max(
                0.0, voting_ability[neighbor] - 1.0 / average_degree
            )

    return chosen


def kshell_seeds(
    graph: GraphInfo, budget: int, diffusion_model: str = "IC", **_
) -> list[int]:
    """k-shell seeding (Kitsak et al. 2010): top-k by core number (degree breaks ties) — core position beats raw degree for spreading."""
    neighbors = _neighbor_sets(graph)
    degrees = np.array([len(nbrs) for nbrs in neighbors], dtype=np.int64)

    # Standard peeling: repeatedly strip nodes of degree <= k, assign core = k
    working_degrees = degrees.copy()
    core = np.zeros(graph.num_nodes, dtype=np.int64)
    remaining = set(range(graph.num_nodes))
    shell = 0

    while remaining:
        shell = max(shell, int(working_degrees[list(remaining)].min()))
        queue = [node for node in remaining if working_degrees[node] <= shell]

        while queue:
            node = queue.pop()
            if node not in remaining:
                continue

            remaining.discard(node)
            core[node] = shell

            for neighbor in neighbors[node]:
                if neighbor in remaining:
                    working_degrees[neighbor] -= 1
                    if working_degrees[neighbor] <= shell:
                        queue.append(neighbor)

    order = np.lexsort((-degrees, -core))
    return [int(node) for node in order[:budget]]


def collective_influence(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    radius: int = 2,
    **_,
) -> list[int]:
    """Collective Influence (Morone & Makse 2015): (k_i-1) * sum of (k_j-1) over the ball boundary at `radius`; adaptive removal."""
    neighbors = _neighbor_sets(graph)
    removed = set()
    chosen = []

    for _ in range(min(budget, graph.num_nodes)):
        alive_degrees = {
            node: len(neighbors[node] - removed)
            for node in range(graph.num_nodes)
            if node not in removed
        }

        best_node, best_score = -1, float("-inf")
        for node in alive_degrees:
            # BFS over alive nodes; the frontier at exact depth `radius` is the ball boundary
            frontier = {node}
            visited = {node}
            for _ in range(radius):
                frontier = {
                    neighbor
                    for member in frontier
                    for neighbor in neighbors[member]
                    if neighbor not in removed and neighbor not in visited
                }
                visited |= frontier

            score = (alive_degrees[node] - 1) * sum(
                alive_degrees[boundary_node] - 1 for boundary_node in frontier
            )
            if score > best_score:
                best_node, best_score = node, score

        chosen.append(best_node)
        removed.add(best_node)

    return chosen


def irie(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    alpha: float = 0.7,
    iterations: int = 20,
    **_,
) -> list[int]:
    """IRIE (Jung et al. 2012, simplified): influence-rank iteration r = (1-AP)(1 + a*sum p*r), no MC inside."""
    sources = graph.edge_index[0]
    targets = graph.edge_index[1]
    probs = graph.ic_probs.astype(np.float64)

    activation = np.zeros(graph.num_nodes)
    chosen = []

    for _ in range(min(budget, graph.num_nodes)):
        rank = np.ones(graph.num_nodes)
        for _ in range(iterations):
            spread_in = np.zeros(graph.num_nodes)
            np.add.at(spread_in, sources, probs * rank[targets])
            rank = (1.0 - activation) * (1.0 + alpha * spread_in)

        rank[chosen] = float("-inf")
        winner = int(np.argmax(rank))
        chosen.append(winner)

        # Damp future ranks by the new seed's one-hop activation probability
        activation[winner] = 1.0
        winner_edges = sources == winner
        for target, prob in zip(targets[winner_edges], probs[winner_edges]):
            activation[target] = 1.0 - (1.0 - activation[target]) * (1.0 - prob)

    return chosen


def random_seeds(
    graph: GraphInfo, budget: int, diffusion_model: str = "IC", seed: int = 42, **_
) -> list[int]:
    """Uniform random seed set — the trivial floor baseline."""
    rng = np.random.default_rng(seed)
    count = min(budget, graph.num_nodes)

    return [int(node) for node in rng.choice(graph.num_nodes, size=count, replace=False)]


def betweenness_seeds(
    graph: GraphInfo, budget: int, diffusion_model: str = "IC", **_
) -> list[int]:
    """Top-k betweenness-centrality nodes (bridge/broker positions between regions)."""
    nx_graph = nx.DiGraph() if graph.directed else nx.Graph()
    nx_graph.add_nodes_from(range(graph.num_nodes))
    nx_graph.add_edges_from(
        (int(graph.edge_index[0, edge]), int(graph.edge_index[1, edge]))
        for edge in range(graph.edge_index.shape[1])
    )

    centrality = nx.betweenness_centrality(nx_graph)
    scores = np.array([centrality[node] for node in range(graph.num_nodes)])

    return _top_k_by_score(scores, budget)


# Registry enumerated by the library API and README
algorithms = {
    # degree
    "high_degree": high_degree,
    "weighted_degree": weighted_degree,
    "degree_discount": degree_discount,
    # centrality
    "pagerank_seeds": pagerank_seeds,
    "eigenvector_seeds": eigenvector_seeds,
    "closeness_seeds": closeness_seeds,
    # greedy
    "vanilla_greedy": vanilla_greedy,
    "celf": celf,
    "celf_pp": celf_pp,
    "adaptive_greedy": adaptive_greedy,
    # RIS
    "ris_basic": ris_basic,
    "tim": tim,
    "imm": imm,
    "ssa": ssa,
    "filtered_ris": filtered_ris,
    # path
    "sp1m": sp1m,
    "mia_pmia": mia_pmia,
    "ldag": ldag,
    # sketch
    "static_greedy": static_greedy,
    "skim": skim,
    # community
    "community_im": community_im,
    "cofim": cofim,
    "community_ris": community_ris,
    # metaheuristic
    "simulated_annealing": simulated_annealing,
    "hill_climbing": hill_climbing,
    "genetic_algorithm": genetic_algorithm,
    # hybrid
    "pagerank_greedy": pagerank_greedy,
    "degree_ris_refine": degree_ris_refine,
    "celf_local_search": celf_local_search,
    "community_celf": community_celf,
    # structure / diversity
    "voterank": voterank,
    "kshell_seeds": kshell_seeds,
    "collective_influence": collective_influence,
    "irie": irie,
    # simple baselines
    "random_seeds": random_seeds,
    "betweenness_seeds": betweenness_seeds,
}
