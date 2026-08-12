"""
Named classical INFLUENCE BLOCKING baselines: the condition-1 floor for
`--task influence_blocking`.

Two contracts, because this literature has two shapes of intervention
(`research/influence_blocking.md` §1.1) and no published work scores them under one
metric (§7):

    blocking_algorithms[name](graph, budget, diffusion_model, negative_seeds=(), **kw)
        -> list[int]              nodes to counter-seed, or nodes to delete

    edge_blocking_algorithms[name](graph, budget, diffusion_model, negative_seeds=(), **kw)
        -> list[(int, int)]       arcs to cut, or arcs to drive to p = 0

`blocking_levers` says which lever each member belongs to, so the harness emits the
right op and charges the right budget.

Read §5.4 and §5.3 before treating any of these as a weak floor. In ascending order
of danger:

  1. `random_blocking`, `degree_blocking`: the free wins, and `degree_blocking` is
     free for a REASON worth internalizing: CLDAG's own §6.3 reports that "the
     traditional degree heuristic cannot be used for influence blocking maximization
     at all". This is the exact opposite of influence maximization, where degree is
     the strong cheap baseline. A blocking result that only beats degree has beaten
     nothing.
  2. **`proximity`**: pick the out-neighbours of the rumour's own seeds. CLDAG finds
     it strong enough to trail only CLDAG itself until the negative cascade can
     traverse long paths, and StratLearner's Table 1 has it at 0.770 / 0.776 on
     power-law and ER, above every learned method except StratLearner. It is also
     UNSTABLE (0.170 on Facebook) which is what makes it interesting rather than
     merely strong. This is the row to beat.
  3. **`imin_lhga`**: SandIMIN's own trivial highest-gain heuristic, which wins
     outright in 6 of the 30 cells of its Table 5 against the two principled methods
     in the same paper. A published VLDB algorithm is beaten by a greedy heuristic a
     fifth of the time, so any table we publish needs this row or it is not evidence.
  4. **`cmia_o`**: Wu & Pan's MIA-style method is still the standard scalable IBM
     baseline and was still being benchmarked against in 2023 (NIE).

Two conventions everything here obeys:

  * **The candidate set is the rumour's reachable region, not V.** §8.1: prevented
    influence counts only nodes that WOULD have been infected, so a blocker placed
    where the cascade never arrives scores exactly zero however central it is. Every
    member below either restricts to that region or weights by it.
  * **One-shot, unless the name says otherwise.** `advanced_greedy` and
    `greedy_replace` recompute after each pick because their published algorithms do;
    everything else scores once and takes the top `budget`, which is a different
    algorithm with different numbers and is stated per member.
"""

import numpy as np

from coding_agent.blocking import exposure_scores, proximity_ring
from coding_agent.tools import primitives
from coding_agent.types import GraphInfo

# Live-edge samples the percolation-based members draw. Kimura's own bond-percolation
# estimator and Xie's AdvancedGreedy both average over sampled realizations; 200 is
# where the ranking stops moving on our graphs and is cheap enough to run per pick.
percolation_samples = 200

# Reverse-reachable samples for the RIS-family members. Tong's RPS needs theta to
# scale with N for its (1 - 1/e - eps) guarantee; this ranks candidates for a
# baseline table rather than certifying one, so it is capped.
rr_per_node = 10
rr_max_theta = 20_000

# MIA / LDAG truncation. Both families keep only paths whose product of transmission
# probabilities exceeds a threshold, which is what makes them local and therefore
# scalable. Wu & Pan and Chen et al. both use 1/320 and report insensitivity.
mia_threshold = 1.0 / 320.0

# The MC blocker scores this many candidates per pick. A full scan is
# O(N x mc_runs) episodes per pick and does not finish past a few thousand nodes.
greedy_prevention_candidates = 30
greedy_prevention_mc_runs = 20
greedy_prevention_horizon = 15

# Forward simulation samples for TC-AIBM's `Forward` baseline
forward_samples = 200

# Betweenness on a graph this large is O(N x E); past it the edge-betweenness member
# samples pivots instead of running every source
betweenness_exact_nodes = 3_000


def _candidate_pool(
    graph: GraphInfo, negative_seeds, budget: int
) -> list[int]:
    """
    Nodes worth spending budget on: the rumour's reachable region, sources excluded.

    Falls back to the whole graph when `negative_seeds` is empty, which only happens
    in a degenerate configuration: every real blocking instance has a rumour.
    """
    sources = {int(node) for node in negative_seeds}

    if not sources:
        return list(range(graph.num_nodes))

    reachable = set(sources)
    frontier = set(sources)

    while frontier:
        frontier = {
            int(other) for node in frontier for other in graph.out_neighbors(node)
        } - reachable
        reachable |= frontier

    pool = sorted(reachable - sources)

    return pool if len(pool) >= budget else pool + [
        node for node in range(graph.num_nodes) if node not in reachable
    ]


def _pad(chosen: list[int], graph: GraphInfo, budget: int, protected=()) -> list[int]:
    """
    Top up a short blocker set with the highest-degree nodes not already in it.

    Every structural member can run out before the budget does: the rumour's
    out-neighbourhood is smaller than `k`, the RR sets are exhausted, the MIA
    arborescences are all covered. A short set silently under-spends and reads as a
    weak method rather than as a small candidate pool.
    """
    if len(chosen) >= budget:
        return chosen[:budget]

    picked = set(chosen) | {int(node) for node in protected}
    for node in primitives.get_top_degree_nodes(graph, graph.num_nodes):
        if len(chosen) >= budget:
            break

        if node not in picked:
            picked.add(node)
            chosen.append(int(node))

    return chosen


# Live-edge sampling and dominators: shared by the percolation family
def _sample_live_graph(graph: GraphInfo, rng: np.random.Generator) -> dict[int, list[int]]:
    """One IC live-edge realization as an out-adjacency."""
    live = {node: [] for node in range(graph.num_nodes)}
    keep = rng.random(graph.edge_index.shape[1]) < graph.ic_probs

    for edge in np.flatnonzero(keep):
        live[int(graph.edge_index[0, edge])].append(int(graph.edge_index[1, edge]))

    return live


def _reachable_order(
    live: dict[int, list[int]], sources, removed: set | None = None
) -> list[int]:
    """BFS order of everything the sources reach in one live-edge graph."""
    removed = removed or set()
    seen = {int(node) for node in sources if int(node) not in removed}
    order = list(seen)
    queue = list(seen)

    while queue:
        node = queue.pop(0)
        for other in live[node]:
            if other not in seen and other not in removed:
                seen.add(other)
                order.append(other)
                queue.append(other)

    return order


def dominator_tree(
    live: dict[int, list[int]], sources, removed: set | None = None
) -> tuple[dict[int, int], dict[int, int]]:
    """
    `(idom, subtree sizes)` for the reachable region of one live-edge graph.

    A node `v` dominates `w` when EVERY path from the sources to `w` runs through
    `v`, so deleting `v` disconnects exactly `v`'s dominator subtree, and that
    subtree size is precisely the reduction in reachable set from cutting `v`. One
    tree therefore scores every candidate at once, which is why it is the right
    primitive for both Xie's AdvancedGreedy (their Algorithm 2 builds the same tree,
    via Lengauer-Tarjan) and Kimura's link blocking, where the credited ARC is
    `(idom(v), v)`.

    Iterative Cooper-Harvey-Kennedy rather than Lengauer-Tarjan: the same tree, a few
    lines instead of a hundred, and at these graph sizes the difference is not
    measurable.
    """
    removed = removed or set()
    order = _reachable_order(live, sources, removed)
    if not order:
        return {}, {}

    position = {node: index for index, node in enumerate(order)}
    predecessors = {node: [] for node in order}

    for node in order:
        for other in live[node]:
            if other in position:
                predecessors[other].append(node)

    roots = {
        int(node)
        for node in sources
        if int(node) in position and int(node) not in removed
    }
    # A virtual super-source, encoded as "the seeds have no predecessor and are
    # their own immediate dominator", which is what makes the tree well defined for
    # a seed SET rather than a single root
    idom = {node: (node if node in roots else None) for node in order}

    def intersect(first: int, second: int) -> int:
        while first != second:
            while position[first] > position[second]:
                first = idom[first]
            while position[second] > position[first]:
                second = idom[second]

        return first

    changed = True
    while changed:
        changed = False
        for node in order:
            if node in roots:
                continue

            candidates = [
                other for other in predecessors[node] if idom[other] is not None
            ]
            if not candidates:
                continue

            new_idom = candidates[0]
            for other in candidates[1:]:
                new_idom = intersect(other, new_idom)

            if idom[node] != new_idom:
                idom[node] = new_idom
                changed = True

    # Subtree sizes, accumulated leaf-to-root so one reverse pass is enough
    counts = {node: 1 for node in order}
    for node in reversed(order):
        parent = idom[node]
        if parent is not None and parent != node:
            counts[parent] += counts[node]

    # A seed is not a blocker candidate, and its "dominated" count is the whole
    # cascade, which would otherwise make the sources the top-ranked blockers
    for node in roots:
        counts.pop(node, None)
        idom.pop(node, None)

    return idom, counts


def _dominator_scores(
    graph: GraphInfo,
    negative_seeds,
    samples: int,
    seed: int = 0,
    removed: set | None = None,
) -> np.ndarray:
    """
    Mean dominated-subtree size per node over `samples` live-edge realizations.

    `removed` is excluded from the traversal rather than only from the ranking: a
    sequential deleter has to score the NEXT pick against the graph its earlier picks
    left behind, and scoring against the intact graph would re-derive the same
    ranking every round and pick a set with no complementarity in it at all.
    """
    rng = np.random.default_rng(seed)
    scores = np.zeros(graph.num_nodes, dtype=np.float64)

    for _ in range(samples):
        live = _sample_live_graph(graph, rng)
        _, counts = dominator_tree(live, negative_seeds, removed)
        for node, count in counts.items():
            scores[node] += count

    return scores / max(samples, 1)


# Counter-seeding (lever: add_node)
def proximity(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    negative_seeds=(),
    **kwargs: object,
) -> list[int]:
    """Seed the rumour's own out-neighbours, highest degree first (CLDAG's strong cheap baseline)."""
    ring = proximity_ring(graph, negative_seeds, hops=1)

    return _pad(list(ring[:budget]), graph, budget, negative_seeds)


def multi_hop_proximity(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    negative_seeds=(),
    hops: int = 4,
    **kwargs: object,
) -> list[int]:
    """Rank by damped reach from the rumour times onward degree: proximity past hop 1."""
    exposure = exposure_scores(graph, negative_seeds, hops=hops)
    onward = primitives.compute_out_degree(graph)
    scores = exposure * (1.0 + onward)

    for node in negative_seeds:
        scores[int(node)] = -1.0

    return _pad(
        [int(node) for node in np.argsort(-scores)[:budget]],
        graph,
        budget,
        negative_seeds,
    )


def degree_blocking(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    negative_seeds=(),
    **kwargs: object,
) -> list[int]:
    """Top-degree nodes, ignoring the rumour entirely: CLDAG reports this fails outright."""
    sources = {int(node) for node in negative_seeds}
    ranked = [
        node
        for node in primitives.get_top_degree_nodes(graph, graph.num_nodes)
        if node not in sources
    ]

    return ranked[:budget]


def pagerank_blocking(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    negative_seeds=(),
    **kwargs: object,
) -> list[int]:
    """Top PageRank nodes inside the rumour's reachable region."""
    scores = primitives.compute_pagerank(graph)
    pool = _candidate_pool(graph, negative_seeds, budget)
    ranked = sorted(pool, key=lambda node: scores[node], reverse=True)

    return _pad(ranked[:budget], graph, budget, negative_seeds)


def betweenness_blocking(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    negative_seeds=(),
    **kwargs: object,
) -> list[int]:
    """Top eigenvector-centrality nodes in the reachable region (the cheap centrality floor)."""
    scores = primitives.compute_centrality(graph, kind="eigenvector")
    pool = _candidate_pool(graph, negative_seeds, budget)
    ranked = sorted(pool, key=lambda node: scores[node], reverse=True)

    return _pad(ranked[:budget], graph, budget, negative_seeds)


def forward_blocking(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    negative_seeds=(),
    samples: int = forward_samples,
    seed: int = 0,
    **kwargs: object,
) -> list[int]:
    """TC-AIBM's `Forward`: simulate the rumour, seed the most frequently infected nodes."""
    rng = np.random.default_rng(seed)
    counts = np.zeros(graph.num_nodes, dtype=np.float64)

    for _ in range(samples):
        live = _sample_live_graph(graph, rng)
        for node in _reachable_order(live, negative_seeds):
            counts[node] += 1.0

    for node in negative_seeds:
        counts[int(node)] = -1.0

    return _pad(
        [int(node) for node in np.argsort(-counts)[:budget]],
        graph,
        budget,
        negative_seeds,
    )


def _prevention_sets(
    graph: GraphInfo,
    negative_seeds,
    theta: int,
    tie_break: str,
    seed: int = 0,
) -> list[set[int]]:
    """
    Tong's R-tuples: for a sampled target `w`, which nodes could have SAVED it.

    Under the live-edge characterisation a node `w` reached by the rumour is saved by
    a positive seed `v` iff `v` reaches it no later than the rumour does: `<=` under
    positive dominance and `<` under negative dominance (§8.4, TC-AIBM Lemma 1). So
    one sample is: draw a live-edge graph, draw a target `w` the rumour reaches,
    compute its distance from `S_N`, and collect every node whose distance to `w`
    clears that bar. Greedy max-coverage over these sets is exactly the RIS argument
    transplanted from maximization to PREVENTION, which is what makes the family
    `(1 - 1/e - eps)` for blocking.
    """
    rng = np.random.default_rng(seed)
    reverse = {node: [] for node in range(graph.num_nodes)}
    for edge in range(graph.edge_index.shape[1]):
        reverse[int(graph.edge_index[1, edge])].append(int(graph.edge_index[0, edge]))

    sources = {int(node) for node in negative_seeds}
    strict = tie_break == "negative"
    sets = []

    for _ in range(theta):
        live = _sample_live_graph(graph, rng)

        # Distance from S_N in this realization
        distance = {node: 0 for node in sources}
        queue = list(sources)
        while queue:
            node = queue.pop(0)
            for other in live[node]:
                if other not in distance:
                    distance[other] = distance[node] + 1
                    queue.append(other)

        targets = [node for node in distance if node not in sources]
        if not targets:
            continue

        target = int(targets[int(rng.integers(len(targets)))])
        limit = distance[target]

        # Reverse BFS from the target over the SAME realization: how many live hops
        # each node is from being able to reach it
        back = {target: 0}
        queue = [target]
        savers = set()

        while queue:
            node = queue.pop(0)
            hops = back[node]

            if node not in sources and (hops < limit if strict else hops <= limit):
                savers.add(node)

            if hops >= limit:
                continue

            for other in reverse[node]:
                # Only traverse arcs live in THIS realization: the whole point of the
                # R-tuple is that saving is judged on one shared live-edge world
                if other not in back and node in live[other]:
                    back[other] = hops + 1
                    queue.append(other)

        if savers:
            sets.append(savers)

    return sets


def rps(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    negative_seeds=(),
    tie_break: str = "positive",
    seed: int = 0,
    **kwargs: object,
) -> list[int]:
    """Tong et al. INFOCOM'17: greedy max-coverage over reverse PREVENTION sets."""
    theta = min(rr_max_theta, rr_per_node * graph.num_nodes)
    sets = _prevention_sets(graph, negative_seeds, theta, tie_break, seed)

    covers = {}
    for index, saver_set in enumerate(sets):
        for node in saver_set:
            covers.setdefault(node, set()).add(index)

    chosen, covered = [], set()
    for _ in range(budget):
        best_node, best_gain = None, 0
        for node, cover in covers.items():
            gain = len(cover - covered)
            if gain > best_gain:
                best_node, best_gain = node, gain

        if best_node is None:
            break

        chosen.append(int(best_node))
        covered |= covers[best_node]

    return _pad(chosen, graph, budget, negative_seeds)


def reverse_blocking(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    negative_seeds=(),
    seed: int = 0,
    **kwargs: object,
) -> list[int]:
    """TC-AIBM's `Reverse`: RR-set coverage restricted to the rumour's reachable region."""
    theta = min(rr_max_theta, rr_per_node * graph.num_nodes)
    pool = set(_candidate_pool(graph, negative_seeds, budget))
    covers = {}

    for index, rr_set in enumerate(
        primitives.batch_reverse_sample(graph, theta=theta, seed=seed)
    ):
        for node in rr_set:
            if int(node) in pool:
                covers.setdefault(int(node), set()).add(index)

    chosen, covered = [], set()
    for _ in range(budget):
        best_node, best_gain = None, 0
        for node, cover in covers.items():
            gain = len(cover - covered)
            if gain > best_gain:
                best_node, best_gain = node, gain

        if best_node is None:
            break

        chosen.append(int(best_node))
        covered |= covers[best_node]

    return _pad(chosen, graph, budget, negative_seeds)


def _max_probability_paths(
    graph: GraphInfo, sources, threshold: float = mia_threshold
) -> tuple[np.ndarray, np.ndarray]:
    """
    (best path product, hop count) from `sources` to every node, truncated at `threshold`.

    The MIA family's core object: the maximum-influence path, found by a Dijkstra on
    `-log p` and cut off once the product falls below the threshold, which is what
    keeps the arborescences local.
    """
    import heapq

    best = np.zeros(graph.num_nodes, dtype=np.float64)
    hops = np.full(graph.num_nodes, np.inf)
    weights = {}

    for edge in range(graph.edge_index.shape[1]):
        weights[(int(graph.edge_index[0, edge]), int(graph.edge_index[1, edge]))] = float(
            graph.ic_probs[edge]
        )

    heap = []
    for node in sources:
        node = int(node)
        best[node] = 1.0
        hops[node] = 0.0
        heapq.heappush(heap, (0.0, 0, node))

    while heap:
        cost, hop, node = heapq.heappop(heap)
        if cost > -np.log(max(best[node], 1e-300)) + 1e-12:
            continue

        for other in graph.out_neighbors(node):
            probability = best[node] * weights.get((node, other), 0.0)
            if probability <= threshold or probability <= best[other]:
                continue

            best[other] = probability
            hops[other] = hop + 1
            heapq.heappush(heap, (-np.log(probability), hop + 1, other))

    return best, hops


def cmia_o(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    negative_seeds=(),
    tie_break: str = "positive",
    **kwargs: object,
) -> list[int]:
    """
    Wu & Pan (Computer Networks 2017) CMIA-O: greedy over MIA arborescences.

    Warning: WHAT THIS IMPLEMENTS. The paper is paywalled with no open PDF, no result
    table and no code (research/influence_blocking.md §11 names it the largest single
    hole in §5), so this is the MIA construction it is built on rather than a
    transcription: maximum-influence paths from `S_N` and from the blocker set,
    truncated at `mia_threshold`, with a node counted as saved when the positive path
    arrives no later than the negative one: `<=` under positive dominance and `<`
    under negative, which is the same live-edge criterion `rps` uses. Greedy over
    candidates, recomputing the positive arborescence after each pick.

    It is faithful to the FAMILY and is the right thing to have in the table (NIE was
    still benchmarking against CMIA-O in 2023); it is not a reproduction of their
    exact numbers, and no such reproduction is available to anyone.
    """
    negative_reach, negative_hops = _max_probability_paths(graph, negative_seeds)
    pool = [
        node
        for node in _candidate_pool(graph, negative_seeds, budget)
        if negative_reach[node] > 0.0
    ]
    strict = tie_break == "negative"
    chosen = []

    for _ in range(min(budget, len(pool))):
        best_node, best_gain = None, 0.0

        for candidate in pool:
            if candidate in chosen:
                continue

            reach, hops = _max_probability_paths(graph, chosen + [candidate])
            saved = (hops < negative_hops) if strict else (hops <= negative_hops)
            # COUNT the nodes covered, for the same reason `cldag` does: the
            # threshold already decides membership of the arborescence, and
            # re-weighting by `negative_reach * reach` applies it twice and
            # discounts high-degree nodes by about 10x (see `cldag`). Counting is
            # also what makes the greedy loop below a submodular coverage
            # maximization, which is what CMIA-O's guarantee rests on.
            gain = float(((negative_reach > 0.0) & (reach > 0.0) & saved).sum())

            if gain > best_gain:
                best_node, best_gain = candidate, gain

        if best_node is None:
            break

        chosen.append(int(best_node))

    return _pad(chosen, graph, budget, negative_seeds)


def _dag_reach(
    graph: GraphInfo, sources, threshold: float = mia_threshold
) -> tuple[np.ndarray, np.ndarray]:
    """
    (reach probability, hop count) aggregating EVERY path in the local DAG.

    `_max_probability_paths` returns the single best path's product, which is the
    MIA family's arborescence and is what `proximity`-style ranking needs. It is a
    severe underestimate wherever a node is reached by many weak paths rather than
    one strong one, and under the weighted-cascade model `p(u->v) = 1/in_degree(v)`
    that is exactly the high-degree nodes: the best single path into a 544-in-degree
    hub is about 1/544, while the hub is in practice reached almost surely.

    CLDAG's own contribution is that per-node influence is computed EXACTLY within
    the thresholded DAG rather than along one path, so its score needs this. Nodes
    are composed in hop order with the independent-cascade form the structured head
    uses, `1 - prod(1 - p_uv * reach_u)`, which is exact on a DAG.
    """
    best, hops = _max_probability_paths(graph, sources, threshold)

    reach = np.zeros(graph.num_nodes, dtype=np.float64)
    for node in sources:
        reach[int(node)] = 1.0

    weights = {}
    for edge in range(graph.edge_index.shape[1]):
        weights[(int(graph.edge_index[0, edge]), int(graph.edge_index[1, edge]))] = float(
            graph.ic_probs[edge]
        )

    # Hop order makes the local DAG acyclic, so one sweep is exact on it
    order = [node for node in np.argsort(hops) if np.isfinite(hops[node])]
    for node in order:
        node = int(node)
        if reach[node] >= 1.0 or best[node] <= 0.0:
            continue

        survive = 1.0
        for other in graph.in_neighbors(node):
            other = int(other)
            if hops[other] >= hops[node] or reach[other] <= 0.0:
                continue

            survive *= 1.0 - weights.get((other, node), 0.0) * reach[other]

        reach[node] = 1.0 - survive

    return reach, hops


def cldag(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "LT",
    negative_seeds=(),
    **kwargs: object,
) -> list[int]:
    """
    He et al. SDM'12 (CLDAG): local-DAG blocking under the competitive linear threshold model.

    Warning: WHAT THIS IMPLEMENTS. CLDAG's contribution is a local-DAG solver whose
    per-node influence is computed exactly within a thresholded DAG; the published
    comparison is figure-only (§11) and no code was ever released. This is that
    scoring rule with the local DAG built as the thresholded maximum-weight path
    expansion the MIA family uses, under NEGATIVE dominance, which is what CLT
    hard-codes. Ranking is one-shot rather than greedy, because CLDAG's own
    contribution is the speed of the local computation, not the outer loop.
    """
    negative_reach, negative_hops = _dag_reach(graph, negative_seeds)
    pool = _candidate_pool(graph, negative_seeds, budget)
    scores = np.zeros(graph.num_nodes, dtype=np.float64)

    for candidate in pool:
        reach, hops = _dag_reach(graph, [candidate])
        # COUNT the nodes saved inside the local DAG; do not re-weight them by the
        # path probability. The threshold is what makes the DAG local, so a node is
        # either in it or it is not, and multiplying by `negative_reach * reach`
        # applies that cut a second time. Measured on email-Eu-core, the max-influence
        # path underestimates P(infected) by 24x at a degree-544 hub against 2.5x at
        # a degree-31 node, so the re-weighting is not a neutral rescaling: it
        # discounts hubs by about 10x relative to the periphery and the ranking
        # collapses onto low-degree nodes next to a source, which scored barely
        # better than `random_blocking` under BOTH IC and LT.
        #
        # CLT resolves a tie in favour of the rumour, so the blocker has to arrive
        # STRICTLY earlier, and a node the rumour never reaches is worth nothing.
        saved = (negative_reach > 0.0) & (reach > 0.0) & (hops < negative_hops)
        scores[candidate] = float(saved.sum())

    return _pad(
        [int(node) for node in np.argsort(-scores)[:budget] if scores[node] > 0.0],
        graph,
        budget,
        negative_seeds,
    )


def random_blocking(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    negative_seeds=(),
    seed: int = 0,
    **kwargs: object,
) -> list[int]:
    """Uniformly random non-source nodes: the floor every table needs."""
    rng = np.random.default_rng(seed)
    pool = [
        node
        for node in range(graph.num_nodes)
        if node not in {int(other) for other in negative_seeds}
    ]
    chosen = rng.choice(len(pool), size=min(budget, len(pool)), replace=False)

    return sorted(int(pool[int(index)]) for index in chosen)


def greedy_prevention(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    negative_seeds=(),
    n_candidates: int = greedy_prevention_candidates,
    mc_runs: int = greedy_prevention_mc_runs,
    horizon: int = greedy_prevention_horizon,
    seed: int = 0,
    lever: str = "counter_seed",
    **kwargs: object,
) -> list[int]:
    """
    Budak's Greedy / TC-AIBM's `Greedy-B`: marginal-gain greedy on simulated prevented influence.

    The honest classical cost of this problem and the method every scalable paper
    exists to avoid: `budget x candidates x mc_runs` competitive episodes. Blocked
    from generated scripts by default for the same reason `celf` is: its episodes run
    on a private simulator and never reach `real_env_episodes`.
    """
    candidates = proximity(graph, n_candidates, diffusion_model, negative_seeds)
    chosen = []

    for _ in range(min(budget, len(candidates))):
        best_node, best_spread = None, float("inf")

        for candidate in candidates:
            if candidate in chosen:
                continue

            spread = primitives.mc_simulate_blocking(
                graph,
                list(negative_seeds),
                chosen + [candidate],
                diffusion_model,
                lever=lever,
                mc_runs=mc_runs,
                horizon=horizon,
                seed=seed,
            )
            if spread < best_spread:
                best_node, best_spread = candidate, spread

        if best_node is None:
            break

        chosen.append(int(best_node))

    return _pad(chosen, graph, budget, negative_seeds)


# Node blocking (lever: remove_node), the IMIN line
def imin_lhga(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    negative_seeds=(),
    **kwargs: object,
) -> list[int]:
    """
    SandIMIN's LHGA: the degree-based heuristic over the rumour's out-neighbour set.

    Read from the authors' own C++ rather than inferred: `Sandwich.h` builds `CB` as
    the out-neighbours of `rumorSet` and then calls `deg_based_heuristic(k, CB)`, so
    LHGA is exactly "rank the rumour's out-neighbours by degree". That is the row
    that wins outright in 6 of the 30 cells of the paper's own Table 5, including
    beating both principled methods on EmailCore at k = 10 and by 27% on Pokec.
    """
    ring = proximity_ring(graph, negative_seeds, hops=1)

    return _pad(list(ring[:budget]), graph, budget, negative_seeds)


def imin_lsbm(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    negative_seeds=(),
    seed: int = 0,
    **kwargs: object,
) -> list[int]:
    """
    SandIMIN's lower-bound sampling maximization: RR coverage over the blockable candidates.

    Warning: WHAT THIS IMPLEMENTS. SandIMIN's contribution is a SANDWICH, a submodular
    lower bound and upper bound around the non-submodular IMIN objective, with the
    `(1 - 1/e - eps)` guarantee holding on the bound rather than on the objective.
    This is the lower-bound component's selection rule (reverse-reachable coverage
    restricted to the same candidate set `CB` the C++ builds), not the full sandwich,
    which also runs the upper bound and the original and returns whichever of the
    three re-estimates best. Run `external:sandimin` for the authors' own code.
    """
    ring = set(proximity_ring(graph, negative_seeds, hops=1))
    theta = min(rr_max_theta, rr_per_node * graph.num_nodes)
    covers = {}

    for index, rr_set in enumerate(
        primitives.batch_reverse_sample(graph, theta=theta, seed=seed)
    ):
        # A blocker only helps on a realization the rumour would actually have used
        if not rr_set & {int(node) for node in negative_seeds}:
            continue

        for node in rr_set:
            if int(node) in ring:
                covers.setdefault(int(node), set()).add(index)

    chosen, covered = [], set()
    for _ in range(budget):
        best_node, best_gain = None, 0
        for node, cover in covers.items():
            gain = len(cover - covered)
            if gain > best_gain:
                best_node, best_gain = node, gain

        if best_node is None:
            break

        chosen.append(int(best_node))
        covered |= covers[best_node]

    return _pad(chosen, graph, budget, negative_seeds)


def advanced_greedy(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    negative_seeds=(),
    samples: int = percolation_samples,
    seed: int = 0,
    **kwargs: object,
) -> list[int]:
    """
    Xie et al. ICDE'23 / IJoC'25 AdvancedGreedy: delete the vertex that dominates the most reach.

    Their Algorithm 2 accelerates the baseline greedy by building a DOMINATOR TREE
    over sampled live-edge graphs: verified from `src/AdvancedGreedy.cpp`, which
    ships a Lengauer-Tarjan implementation (`struct tl` with `semi` / `idom` / `dt`)
    and restricts candidates to the sources' out-neighbours. A node's dominated
    subtree size IS the reduction in reachable set from deleting it, which is why one
    tree per sample scores every candidate at once.

    Sequential: the graph is re-sampled and re-scored after each deletion, because
    their loop deletes the node and rebuilds `e_tmp` before the next pick.
    """
    removed = []
    ring = set(proximity_ring(graph, negative_seeds, hops=1))

    for step in range(budget):
        scores = _dominator_scores(
            graph, negative_seeds, samples, seed + step, removed=set(removed)
        )

        for node in list(removed) + [int(node) for node in negative_seeds]:
            scores[node] = -1.0

        # Their candidate set is the sources' out-neighbourhood; fall back to the
        # whole reachable region once that is exhausted
        candidates = [node for node in ring if node not in removed] or [
            node for node in range(graph.num_nodes) if scores[node] > 0
        ]
        if not candidates:
            break

        best = max(candidates, key=lambda node: scores[node])
        if scores[best] <= 0:
            break

        removed.append(int(best))

    return _pad(removed, graph, budget, negative_seeds)


def greedy_replace(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    negative_seeds=(),
    samples: int = percolation_samples,
    seed: int = 0,
    **kwargs: object,
) -> list[int]:
    """
    Xie et al. GreedyReplace: AdvancedGreedy, then a reverse pass that re-picks each slot.

    Their Algorithm 4, and the shape is verified from `src/GreedyReplace.cpp`: run the
    greedy forward, then walk the picks in REVERSE, un-remove each one and let the
    candidate rule choose again, stopping the moment it re-chooses the same node. The
    ICDE'23 Tables V-VI put it within 0.12% of the exact optimum at `b = 4` in a
    third of a second against 22 hours, which is the sharpest single statement of
    why this problem is combinatorially brutal and empirically easy.
    """
    removed = advanced_greedy(
        graph, budget, diffusion_model, negative_seeds, samples, seed
    )

    for index in range(len(removed) - 1, -1, -1):
        held = removed[index]
        rest = removed[:index] + removed[index + 1 :]
        scores = _dominator_scores(
            graph, negative_seeds, samples, seed + 1000 + index, removed=set(rest)
        )

        for node in list(rest) + [int(node) for node in negative_seeds]:
            scores[node] = -1.0

        replacement = int(np.argmax(scores))
        if replacement == held or scores[replacement] <= 0:
            break

        removed = rest + [replacement]

    return _pad(list(dict.fromkeys(removed)), graph, budget, negative_seeds)


# Edge blocking (levers: remove_edge, set_edge_weight)
def kimura_link_blocking(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    negative_seeds=(),
    samples: int = percolation_samples,
    seed: int = 0,
    **kwargs: object,
) -> list[tuple]:
    """
    Kimura, Saito & Motoda (AAAI 2008): block `k` links to minimize expected contamination.

    Their estimator is the BOND PERCOLATION method: sample live-edge graphs and
    measure reachability on them, and their solver is plain greedy with no
    approximation guarantee claimed. Implemented here as: on each sampled realization
    build the dominator tree from `S_N`, and credit the arc `(idom(v), v)` with `v`'s
    dominated subtree size, which is exactly the reduction in contamination from
    cutting that arc on that realization. Averaged over samples, top `k`.
    """
    rng = np.random.default_rng(seed)
    scores = {}

    for _ in range(samples):
        live = _sample_live_graph(graph, rng)
        idom, counts = dominator_tree(live, negative_seeds)

        # The arc into v from its immediate dominator is the one whose removal
        # disconnects v's whole subtree on this realization
        for node, parent in idom.items():
            if parent is None or parent == node:
                continue

            scores[(parent, node)] = scores.get((parent, node), 0.0) + counts[node]

    ranked = sorted(scores, key=lambda edge: scores[edge], reverse=True)

    return _pad_edges(ranked[:budget], graph, budget)


def out_edge_blocking(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    negative_seeds=(),
    **kwargs: object,
) -> list[tuple]:
    """Cut the highest `exposure(u) * p(u,v)` arcs: the cheapest sensible edge floor."""
    exposure = exposure_scores(graph, negative_seeds)
    scores = {}

    for edge in range(graph.edge_index.shape[1]):
        source = int(graph.edge_index[0, edge])
        target = int(graph.edge_index[1, edge])
        scores[(source, target)] = exposure[source] * float(graph.ic_probs[edge])

    ranked = sorted(scores, key=lambda edge: scores[edge], reverse=True)

    return _pad_edges(ranked[:budget], graph, budget)


def edge_betweenness_blocking(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    negative_seeds=(),
    seed: int = 0,
    **kwargs: object,
) -> list[tuple]:
    """Cut the arcs carrying the most shortest paths OUT of the rumour's region."""
    rng = np.random.default_rng(seed)
    sources = [int(node) for node in negative_seeds] or list(range(graph.num_nodes))

    if graph.num_nodes > betweenness_exact_nodes:
        sources = [
            int(node)
            for node in rng.choice(sources, size=min(len(sources), 64), replace=False)
        ]

    scores = {}
    for source in sources:
        # Shortest-path counts by BFS, credited to the arc each node was first
        # reached through: the sampled-pivot edge betweenness restricted to the
        # rumour's own sources, which is the only region that can matter (§8.1)
        distance = {source: 0}
        parent = {}
        queue = [source]

        while queue:
            node = queue.pop(0)
            for other in graph.out_neighbors(node):
                if other not in distance:
                    distance[other] = distance[node] + 1
                    parent[other] = node
                    queue.append(other)

        for node, previous in parent.items():
            scores[(previous, node)] = scores.get((previous, node), 0.0) + 1.0

    ranked = sorted(scores, key=lambda edge: scores[edge], reverse=True)

    return _pad_edges(ranked[:budget], graph, budget)


def random_edge_blocking(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    negative_seeds=(),
    seed: int = 0,
    **kwargs: object,
) -> list[tuple]:
    """Uniformly random arcs: the edge-lever floor."""
    rng = np.random.default_rng(seed)
    total = graph.edge_index.shape[1]
    chosen = rng.choice(total, size=min(budget, total), replace=False)

    return [
        (int(graph.edge_index[0, edge]), int(graph.edge_index[1, edge]))
        for edge in chosen
    ]


def _pad_edges(chosen: list, graph: GraphInfo, budget: int) -> list[tuple]:
    """Top up a short arc set with the highest-probability arcs not already in it."""
    if len(chosen) >= budget:
        return [tuple(edge) for edge in chosen[:budget]]

    picked = {tuple(edge) for edge in chosen}
    order = np.argsort(-graph.ic_probs)

    for edge in order:
        if len(chosen) >= budget:
            break

        pair = (int(graph.edge_index[0, edge]), int(graph.edge_index[1, edge]))
        if pair not in picked:
            picked.add(pair)
            chosen.append(pair)

    return [tuple(edge) for edge in chosen]


blocking_algorithms = {
    # counter-seeding: the founding sub-literature
    "proximity": proximity,
    "multi_hop_proximity": multi_hop_proximity,
    "degree_blocking": degree_blocking,
    "pagerank_blocking": pagerank_blocking,
    "betweenness_blocking": betweenness_blocking,
    "forward_blocking": forward_blocking,
    "reverse_blocking": reverse_blocking,
    "rps": rps,
    "cmia_o": cmia_o,
    "cldag": cldag,
    "random_blocking": random_blocking,
    # node blocking: the IMIN line
    "imin_lhga": imin_lhga,
    "imin_lsbm": imin_lsbm,
    "advanced_greedy": advanced_greedy,
    "greedy_replace": greedy_replace,
    # simulation-based
    "greedy_prevention": greedy_prevention,
}

edge_blocking_algorithms = {
    "kimura_link_blocking": kimura_link_blocking,
    "out_edge_blocking": out_edge_blocking,
    "edge_betweenness_blocking": edge_betweenness_blocking,
    "random_edge_blocking": random_edge_blocking,
}

# What a member RETURNS, which is what decides whether a lever can emit it: a node
# selector serves both node levers and an arc selector serves both edge levers.
# `degree_blocking` is the clearest case: "the top-degree nodes" is the same
# computation whether you seed them or delete them.
node_shape = "node"
edge_shape = "edge"

lever_shape = {
    "counter_seed": node_shape,
    "node_block": node_shape,
    "edge_block": edge_shape,
    "weight_block": edge_shape,
}


def blocking_shape(name: str) -> str:
    """`node` or `edge`: what this member hands back."""
    return edge_shape if name in edge_blocking_algorithms else node_shape


def emittable(name: str, lever: str) -> bool:
    """Whether a lever can spend its budget on what this member returns."""
    return blocking_shape(name) == lever_shape[lever]


# The sub-literature each member comes from, for the menus and the report. NOT the
# compatibility rule (that is `emittable` above) because several members are
# published in one line and perfectly usable in the other.
blocking_levers = {
    "proximity": "counter_seed",
    "multi_hop_proximity": "counter_seed",
    "degree_blocking": "counter_seed",
    "pagerank_blocking": "counter_seed",
    "betweenness_blocking": "counter_seed",
    "forward_blocking": "counter_seed",
    "reverse_blocking": "counter_seed",
    "rps": "counter_seed",
    "cmia_o": "counter_seed",
    "cldag": "counter_seed",
    "random_blocking": "counter_seed",
    "greedy_prevention": "counter_seed",
    "imin_lhga": "node_block",
    "imin_lsbm": "node_block",
    "advanced_greedy": "node_block",
    "greedy_replace": "node_block",
} | {name: "edge_block" for name in edge_blocking_algorithms}

# Simulation-based, so one run costs budget x candidates x mc_runs real competitive
# episodes. Charged honestly to the arm like any other baseline, and blocked from
# generated scripts by default for the same reason `celf` and `greedy_blocking` are:
# its episodes run on a private simulator and bypass the metered evaluator.
mc_blocking_algorithms = ("greedy_prevention",)

# Condition 1's pool PER LEVER, because a lever can only emit what its own members
# return: handing a counter-seeding arm `imin_lhga` gives it a node-deletion set, and
# handing an edge arm `proximity` gives it node ids. Each list is ordered by how
# dangerous the row is (§9.3), and each one leads with the baseline that actually has
# to be beaten rather than with a floor.
#
#   counter_seed  `proximity`: above every learned method except StratLearner on
#                 two of that paper's three graphs, and CLDAG finds it trails only
#                 CLDAG until the rumour can traverse long paths
#   node_block    `imin_lhga`: SandIMIN's own trivial heuristic, which beats both of
#                 that paper's principled methods in 6 of its 30 cells
#   edge/weight   `kimura_link_blocking`: the founding link-blocking method, and the
#                 only published one for this lever with a stated estimator
#
# `degree_blocking` is in the node pools as the published FAILURE mode, not as a
# floor: CLDAG reports plain degree cannot be used for this task at all, which is the
# exact opposite of its role in influence maximization.
default_blocking_baselines = {
    "counter_seed": (
        "proximity",
        "multi_hop_proximity",
        "rps",
        "reverse_blocking",
        "forward_blocking",
        "cldag",
        "degree_blocking",
        "pagerank_blocking",
        "random_blocking",
    ),
    "node_block": (
        "imin_lhga",
        "imin_lsbm",
        "advanced_greedy",
        "greedy_replace",
        "degree_blocking",
        "pagerank_blocking",
        "random_blocking",
    ),
    "edge_block": (
        "kimura_link_blocking",
        "out_edge_blocking",
        "edge_betweenness_blocking",
        "random_edge_blocking",
    ),
}
default_blocking_baselines["weight_block"] = default_blocking_baselines["edge_block"]

all_blocking_algorithms = blocking_algorithms | edge_blocking_algorithms
blocking_algorithm_names = list(all_blocking_algorithms)
