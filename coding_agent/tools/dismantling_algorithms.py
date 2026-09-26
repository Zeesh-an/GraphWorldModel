"""
Named classical NETWORK DISMANTLING baselines: the condition-1 floor for
`--task critical_node_detection`.

Every algorithm has the signature

    (graph, budget, diffusion_model, **kw) -> list[int]

and returns a REMOVAL set of size `budget`: the nodes to delete from the graph,
not the nodes to seed. That is the whole difference from `algorithms.py`, whose
members return a seed set to maximize with.

None of these is a weak floor. In ascending order of danger:

  1. `random_removal`, `pagerank_removal`, static `betweenness_removal`: free wins.
  2. **`adaptive_degree` (HDA)**: MIND's Table 5 puts plain recompute-the-degree at
     **119.9** against FINDER's **115.0** across 47 networks. A five-point gap is
     the entire published advantage of 2020-era deep RL over a ten-line heuristic,
     and a learned dismantler that does not clearly beat HDA has demonstrated
     nothing. This is the direct analogue of the BA-100 degree-triviality result
     we already hit on IM.
  3. **`iterative_betweenness` (BI)**: Wandelt et al. found it best in 70-80% of
     cases across 13 competitors, and almost no learned dismantling paper reports
     it. Omitting it reproduces the exact methodological gap that survey calls out.
  4. **`gnd`**: on NetScience a 2019 spectral heuristic still beats every learned
     method except SPR.

Two conventions everything here obeys:

  * **Undirected.** The entire dismantling literature is undirected,
    so every routine works over `containment.neighbour_sets`, i.e. the symmetrized
    view. Our directed datasets are symmetrized to be scored at all.
  * **Sequential-adaptive, unless the name says otherwise.** Removing `k` nodes at
    once and recomputing after every removal are DIFFERENT algorithms and their
    numbers are not interconvertible: NIRM's own ablation puts the
    gap at 1.96x on `UsPower`. `degree_removal` and `betweenness_removal` are the
    one-pass controls; everything else recomputes.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from coding_agent.containment import core_numbers, neighbour_sets
from coding_agent.tools import primitives
from coding_agent.types import GraphInfo

# Explosive Immunization draws this many candidates per un-vaccination round
# rather than scanning all N. Clusella et al. use m ~ 10^3 and show the result is
# insensitive above ~100.
ei_candidates = 200

# Approximate iterative betweenness recomputes every k/abi_blocks removals rather
# than every removal: Wandelt's quality/time tradeoff, and the only reason BI is
# affordable past a few thousand nodes
abi_blocks = 10

# Candidates the MC blocker scores per pick. A full scan is O(N x mc_runs)
# episodes per pick and does not finish on anything past a few thousand nodes.
# mc_runs below ~20 puts the whole spread difference between two candidates inside
# the sampling noise and the greedy chases it, which is the same winner's-curse
# failure the native arm has at --native-mc-runs 1.
greedy_blocking_candidates = 30
greedy_blocking_mc_runs = 20
greedy_blocking_horizon = 15

# Betweenness on a graph this large is O(N x E) per recomputation; past it the
# iterative variants fall back to their approximate form with a sampled pivot set
betweenness_exact_nodes = 3_000
betweenness_pivots = 300

# EGND sweeps the Fiedler split point instead of always cutting at the median
egnd_ensemble = 5
egnd_quantile_spread = 0.2

# BPD (Mugisha & Zhou, Phys. Rev. E 94:012305). x is the paper's re-weighting
# parameter, fixed to 12 for ER networks and 7 for random-regular ones; 12 is the
# value for the heterogeneous graphs we run. `fraction` is its decimation step,
# "a tiny fraction f of the nodes is deleted" per round.
bpd_reweight = 12.0
bpd_iterations = 40
bpd_fraction = 0.01
bpd_tolerance = 1e-4
# exp() overflows past ~709; the messages saturate long before that, and clipping
# keeps a fully-connected hub from turning the normalization into inf/inf = nan
bpd_max_exponent = 500.0

def _directed_edges(
    neighbours: list[set[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    (source, target, reverse-index) arrays over both orientations of every edge.

    Message passing is defined on DIRECTED edges, and every update needs the
    reverse message, so the pairing is precomputed once rather than looked up per
    iteration.
    """
    pairs = [
        (node, neighbour)
        for node, group in enumerate(neighbours)
        for neighbour in sorted(group)
    ]
    index = {pair: position for position, pair in enumerate(pairs)}

    source = np.fromiter((pair[0] for pair in pairs), dtype=np.int64, count=len(pairs))
    target = np.fromiter((pair[1] for pair in pairs), dtype=np.int64, count=len(pairs))
    reverse = np.fromiter(
        (index[(pair[1], pair[0])] for pair in pairs), dtype=np.int64, count=len(pairs)
    )

    return source, target, reverse


def _pad(chosen: list[int], graph: GraphInfo, budget: int) -> list[int]:
    """
    Top up a short removal set with the highest-degree nodes not already in it.

    Every structural method can run out before the budget does: CoreHD empties
    the 2-core, decycling finishes the forest, GND's cut is smaller than k. A
    short set would silently under-spend the budget and make the arm look better
    per node than it is.
    """
    if len(chosen) >= budget:
        return chosen[:budget]

    picked = set(chosen)
    for node in primitives.get_top_degree_nodes(graph, graph.num_nodes):
        if len(chosen) >= budget:
            break

        if node not in picked:
            picked.add(node)
            chosen.append(int(node))

    return chosen


def _alive_degrees(neighbours: list[set[int]], removed: set[int]) -> dict[int, int]:
    return {
        node: len(group - removed)
        for node, group in enumerate(neighbours)
        if node not in removed
    }


def _components(neighbours: list[set[int]], removed: set[int]) -> list[set[int]]:
    """Connected components of the residual graph, by BFS."""
    seen = set(removed)
    found = []

    for start in range(len(neighbours)):
        if start in seen:
            continue

        component = {start}
        seen.add(start)
        queue = [start]

        while queue:
            node = queue.pop()
            for neighbour in neighbours[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    component.add(neighbour)
                    queue.append(neighbour)

        found.append(component)

    return found


def _giant_size(neighbours: list[set[int]], removed: set[int]) -> int:
    components = _components(neighbours, removed)

    return max((len(component) for component in components), default=0)


def _two_core(neighbours: list[set[int]], removed: set[int]) -> tuple[set[int], dict]:
    """
    The 2-core of the residual graph, with its degrees, by peeling degree-<=1 to fixpoint.

    The cyclic part of the graph, and therefore the only place stage 1 of any
    decycling pipeline (CoreHD, BPD, Min-Sum) has work to do: a node outside the
    2-core sits on a tree and cannot be holding a cycle together.
    """
    core = {node for node in range(len(neighbours)) if node not in removed}
    degrees = {node: len(neighbours[node] - removed) for node in core}
    queue = [node for node in core if degrees[node] <= 1]

    while queue:
        node = queue.pop()
        if node not in core:
            continue

        core.discard(node)
        for neighbour in neighbours[node]:
            if neighbour in core:
                degrees[neighbour] -= 1
                if degrees[neighbour] <= 1:
                    queue.append(neighbour)

    return core, degrees


def _break_trees(
    neighbours: list[set[int]], removed: set[int], chosen: list[int], limit: int
) -> list[int]:
    """
    Stage 2 of the decycling pipelines: split the surviving trees at their centroids.

    Once the residual graph is a forest, the giant component is a tree, and the node
    whose removal splits it most evenly is the cheapest next cut. Both Min-Sum and
    BPD describe this stage the same way: "delete an appropriately chosen node from
    this tree to achieve maximal decrease in the tree size", and it is what makes
    decycling FIRST worthwhile: a forest of n nodes breaks into small components
    under few further removals.

    Mutates `removed` and returns `chosen` extended in place, so the three stage-1
    implementations can share it.
    """
    while len(chosen) < limit:
        components = _components(neighbours, removed)
        largest = max(components, key=len, default=set())

        if len(largest) <= 2:
            break

        best_node, best_size = None, len(largest)
        for node in largest:
            pieces = _components(neighbours, removed | {node})
            biggest = max((len(piece) for piece in pieces if piece & largest), default=0)
            if biggest < best_size:
                best_node, best_size = node, biggest

        if best_node is None:
            break

        chosen.append(int(best_node))
        removed.add(int(best_node))

    return chosen


def _reinsert(
    neighbours: list[set[int]], removed: list[int], threshold: float
) -> list[int]:
    """
    Reverse-greedy reinsertion (Min-Sum / CoreHD / GNDR).

    Put back, cheapest first, every removed node whose return does not push the
    giant component past the bar. What survives is the reduced removal set. GND and
    GNDR are DIFFERENT methods with different numbers and papers cite both under
    one name, which is why the reinserting variants are registered separately here
    rather than folded in.

    Warning: The bar is `max(threshold * N, the giant component this set already
    achieved)`, and the second term is ours. The published pass is defined only
    once the graph is dismantled below `threshold`, because its job is to shrink
    `|S|` at a FIXED threshold: an unbounded-budget setting. Our budget is a fixed
    `k` that rarely reaches 1% of N (measured: on BA-500 at k=15% the giant
    component is still 21 nodes against a bar of 5), so under the published bar
    every reinserting variant was a bit-for-bit copy of its base. Holding the
    achieved size instead keeps the paper's rule wherever the paper's rule applies
    and extends the same idea: return the nodes that were not buying anything,
    to the budgets we actually run.
    """
    kept = set(removed)
    bar = max(threshold * len(neighbours), _giant_size(neighbours, kept))

    # Cheapest first: a node whose reinsertion merges the fewest / smallest
    # clusters is the one most likely to fit under the bar
    order = sorted(removed, key=lambda node: len(neighbours[node]))

    for node in order:
        trial = kept - {node}
        if _giant_size(neighbours, trial) <= bar:
            kept = trial

    return [node for node in removed if node in kept]


def _reinsert_and_refill(
    graph: GraphInfo, removals: list[int], budget: int, threshold: float
) -> list[int]:
    """
    Reinsert, then spend the freed budget on the residual giant component.

    Reinsertion on its own returns a SHORTER set, which at a fixed budget would
    silently under-spend it and make the arm look better per node than it is. The
    freed slots go back through tree breaking, which targets whatever component is
    now largest, so the pass reads as "stop paying for nodes that bought nothing,
    and re-spend on what is still standing".
    """
    neighbours = neighbour_sets(graph)
    limit = min(budget, graph.num_nodes)

    chosen = _reinsert(neighbours, removals, threshold)
    _break_trees(neighbours, set(chosen), chosen, limit)

    return _pad(chosen, graph, budget)


def degree_removal(
    graph: GraphInfo, budget: int, diffusion_model: str = "IC", **_: object
) -> list[int]:
    """Top-k highest-degree nodes, computed ONCE (the one-pass degree control)."""
    return primitives.get_top_degree_nodes(graph, budget)


def adaptive_degree(
    graph: GraphInfo, budget: int, diffusion_model: str = "IC", **_: object
) -> list[int]:
    """HDA: remove the highest-degree node, recompute degrees, repeat, the baseline that hurts."""
    neighbours = neighbour_sets(graph)
    degrees = np.array([len(group) for group in neighbours], dtype=np.int64)
    removed = set()
    chosen = []

    for _ in range(min(budget, graph.num_nodes)):
        degrees[list(removed)] = -1
        node = int(np.argmax(degrees))

        if degrees[node] < 0:
            break

        chosen.append(node)
        removed.add(node)

        # The recompute HDA is named for: every neighbour is one edge lighter now
        for neighbour in neighbours[node]:
            if neighbour not in removed:
                degrees[neighbour] -= 1

    return _pad(chosen, graph, budget)


def pagerank_removal(
    graph: GraphInfo, budget: int, diffusion_model: str = "IC", **_: object
) -> list[int]:
    """Top-k PageRank nodes, computed once."""
    scores = primitives.compute_pagerank(graph)

    return [int(node) for node in np.argsort(-scores)[:budget]]


def kshell_removal(
    graph: GraphInfo, budget: int, diffusion_model: str = "IC", **_: object
) -> list[int]:
    """Top-k by k-core number, degree breaking ties (Kitsak et al. 2010)."""
    core = core_numbers(graph)
    degrees = primitives.compute_degree(graph)

    return [int(node) for node in np.lexsort((-degrees, -core))[:budget]]


def _betweenness(neighbours: list[set[int]], removed: set[int]) -> np.ndarray:
    """
    Brandes betweenness over the residual undirected graph, shape (N,).

    Sampled from `betweenness_pivots` sources on a large graph: exact Brandes is
    O(N x E) per call, and the iterative variants call it once per removal.
    """
    num_nodes = len(neighbours)
    alive = [node for node in range(num_nodes) if node not in removed]
    scores = np.zeros(num_nodes, dtype=np.float64)

    if not alive:
        return scores

    if len(alive) <= betweenness_exact_nodes:
        sources = alive
        scale = 1.0
    else:
        rng = np.random.default_rng(0)
        sources = [
            int(alive[index])
            for index in rng.choice(len(alive), size=betweenness_pivots, replace=False)
        ]
        scale = len(alive) / len(sources)

    for source in sources:
        stack = []
        predecessors = {node: [] for node in alive}
        sigma = dict.fromkeys(alive, 0.0)
        distance = dict.fromkeys(alive, -1)
        sigma[source] = 1.0
        distance[source] = 0
        queue = [source]
        head = 0

        while head < len(queue):
            node = queue[head]
            head += 1
            stack.append(node)

            for neighbour in neighbours[node]:
                if neighbour in removed:
                    continue

                if distance[neighbour] < 0:
                    distance[neighbour] = distance[node] + 1
                    queue.append(neighbour)

                if distance[neighbour] == distance[node] + 1:
                    sigma[neighbour] += sigma[node]
                    predecessors[neighbour].append(node)

        delta = dict.fromkeys(alive, 0.0)
        while stack:
            node = stack.pop()
            for predecessor in predecessors[node]:
                delta[predecessor] += (
                    sigma[predecessor] / sigma[node] * (1.0 + delta[node])
                )

            if node != source:
                scores[node] += delta[node]

    return scores * scale


def betweenness_removal(
    graph: GraphInfo, budget: int, diffusion_model: str = "IC", **_: object
) -> list[int]:
    """Top-k betweenness nodes, computed ONCE (the one-pass betweenness control)."""
    scores = _betweenness(neighbour_sets(graph), set())

    return [int(node) for node in np.argsort(-scores)[:budget]]


def iterative_betweenness(
    graph: GraphInfo, budget: int, diffusion_model: str = "IC", **_: object
) -> list[int]:
    """BI (Wandelt et al. 2018): remove the highest-betweenness node, RECOMPUTE, repeat, best in 70-80% of their cases."""
    neighbours = neighbour_sets(graph)
    removed = set()
    chosen = []

    for _ in range(min(budget, graph.num_nodes)):
        scores = _betweenness(neighbours, removed)
        scores[list(removed)] = -1.0
        node = int(np.argmax(scores))

        if node in removed:
            break

        chosen.append(node)
        removed.add(node)

    return _pad(chosen, graph, budget)


def approx_iterative_betweenness(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    blocks: int = abi_blocks,
    **_: object,
) -> list[int]:
    """ABI (Wandelt et al. 2018): BI recomputing every k/blocks removals, their quality/time tradeoff."""
    neighbours = neighbour_sets(graph)
    removed = set()
    chosen = []
    per_block = max(1, min(budget, graph.num_nodes) // max(1, blocks))

    while len(chosen) < min(budget, graph.num_nodes):
        scores = _betweenness(neighbours, removed)
        scores[list(removed)] = -1.0
        ranked = [int(node) for node in np.argsort(-scores) if node not in removed]

        if not ranked:
            break

        for node in ranked[: min(per_block, budget - len(chosen))]:
            chosen.append(node)
            removed.add(node)

    return _pad(chosen, graph, budget)


def collective_influence_removal(
    graph: GraphInfo, budget: int, diffusion_model: str = "IC", radius: int = 2, **_: object
) -> list[int]:
    """CI (Morone & Makse, Nature 2015): (k_i-1) * sum of (k_j-1) over the ball boundary at `radius`, removed adaptively."""
    neighbours = neighbour_sets(graph)
    removed = set()
    chosen = []

    for _ in range(min(budget, graph.num_nodes)):
        degrees = _alive_degrees(neighbours, removed)
        best_node, best_score = -1, float("-inf")

        for node in degrees:
            # BFS over alive nodes; the frontier at exact depth `radius` is the
            # ball boundary the CI score sums over
            frontier = {node}
            visited = {node}

            for _ in range(radius):
                frontier = {
                    other
                    for member in frontier
                    for other in neighbours[member]
                    if other not in removed and other not in visited
                }
                visited |= frontier

            score = (degrees[node] - 1) * sum(
                degrees[boundary] - 1 for boundary in frontier
            )
            if score > best_score:
                best_node, best_score = node, score

        if best_node < 0:
            break

        chosen.append(int(best_node))
        removed.add(int(best_node))

    return _pad(chosen, graph, budget)


def collective_influence_r(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    radius: int = 2,
    threshold: float = 0.01,
    **_: object,
) -> list[int]:
    """
    CI as Morone & Makse actually publish it: the adaptive removal above, PLUS the
    greedy reinsertion pass.

    `collective_influence_removal` is only the first half. The paper's CI is
    "adaptive removal + greedy reinsertion", and a method and its reinserting
    variant are different methods routinely cited under one name, so the bare
    version should not be reported as "CI".
    """
    return _reinsert_and_refill(
        graph,
        collective_influence_removal(graph, budget, diffusion_model, radius),
        budget,
        threshold,
    )


def corehd(
    graph: GraphInfo, budget: int, diffusion_model: str = "IC", **_: object
) -> list[int]:
    """
    CoreHD (Zdeborová, Zhang, Zhou, Sci. Rep. 2016): take the 2-core, delete its
    highest-degree node, repeat.

    Two to four orders of magnitude faster than CI or BPD at comparable quality,
    because a node outside the 2-core is on a tree and cannot be holding the giant
    component together. When the 2-core empties, `_pad` finishes the job: the
    paper's own tree-breaking stage, minus the reinsertion pass that `corehd_r`
    adds separately.
    """
    neighbours = neighbour_sets(graph)
    removed = set()
    chosen = []

    for _ in range(min(budget, graph.num_nodes)):
        core, degrees = _two_core(neighbours, removed)

        if not core:
            break

        node = max(core, key=lambda candidate: degrees[candidate])
        chosen.append(int(node))
        removed.add(int(node))

    return _pad(chosen, graph, budget)


def corehd_r(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    threshold: float = 0.01,
    **_: object,
) -> list[int]:
    """CoreHD plus the paper's reverse-greedy reinsertion pass, refilled to budget."""
    return _reinsert_and_refill(
        graph, corehd(graph, budget, diffusion_model), budget, threshold
    )


def _bpd_marginals(
    neighbours: list[set[int]],
    removed: set[int],
    reweight: float,
    iterations: int,
    tolerance: float,
) -> np.ndarray:
    """
    Run BPD's belief propagation to a fixed point and return q_i^0 per node.

    The equations are Mugisha & Zhou's, on the feedback-vertex-set spin model where
    each vertex is either deleted, the root of a tree, or a child pointing at one
    parent. Two numbers per directed edge carry the whole message:

        q^0_{i->j}  probability i is suitable for DELETION, with j absent
        q^i_{i->j}  probability i is suitable to be a tree ROOT, with j absent

    and the remaining mass is i having some parent in ∂i \\ j. Writing
    P = prod_{k in ∂i\\j} [q^0_{k->i} + q^k_{k->i}] and
    S = sum_{l in ∂i\\j} (1 - q^0_{l->i}) / (q^0_{l->i} + q^l_{l->i}), the paper's
    eqs. (3a), (3b) and (4) are

        z_{i->j} = 1 + e^x * P * (1 + S)
        q^0_{i->j} = 1 / z_{i->j}
        q^i_{i->j} = e^x * P / z_{i->j}

    and the node marginal (eq. 2) is the same expression over the full neighbourhood.

    Computed over directed-edge arrays with the per-node aggregate divided down to
    the "excluding j" form, so one iteration is O(E) rather than O(sum deg^2). The
    product is accumulated in LOG space: e^12 is ~1.6e5 and P underflows on any
    high-degree node, so the direct product loses the ratio the update depends on.
    """
    num_nodes = len(neighbours)
    source, target, reverse = _directed_edges(neighbours)

    if source.size == 0:
        return np.zeros(num_nodes)

    alive_node = np.ones(num_nodes, dtype=bool)
    if removed:
        alive_node[list(removed)] = False
    # A deleted endpoint takes the edge out of ∂i(t) entirely
    alive_edge = alive_node[source] & alive_node[target]

    q_zero = np.full(source.size, 0.5)
    q_root = np.full(source.size, 0.25)

    for _ in range(iterations):
        # Incoming message on edge e = (i -> j) is the reverse edge (j -> i)
        incoming_zero = q_zero[reverse]
        incoming_root = q_root[reverse]

        weight = np.clip(incoming_zero + incoming_root, 1e-12, None)
        # Neutral (log 1, sum 0) wherever the neighbour is gone
        log_weight = np.where(alive_edge, np.log(weight), 0.0)
        parent_term = np.where(alive_edge, (1.0 - incoming_zero) / weight, 0.0)

        # Aggregate over ALL of ∂i, then divide out j to get ∂i \ j
        log_product = np.zeros(num_nodes)
        parent_sum = np.zeros(num_nodes)
        np.add.at(log_product, source, log_weight)
        np.add.at(parent_sum, source, parent_term)

        log_excluded = log_product[source] - log_weight
        sum_excluded = parent_sum[source] - parent_term

        scale = np.exp(np.clip(reweight + log_excluded, None, bpd_max_exponent))
        normalizer = 1.0 + scale * (1.0 + sum_excluded)

        updated_zero = 1.0 / normalizer
        updated_root = scale / normalizer

        shift = np.max(np.abs(updated_zero - q_zero)) if source.size else 0.0
        q_zero, q_root = updated_zero, updated_root

        if shift < tolerance:
            break

    weight = np.clip(q_zero[reverse] + q_root[reverse], 1e-12, None)
    log_weight = np.where(alive_edge, np.log(weight), 0.0)
    parent_term = np.where(alive_edge, (1.0 - q_zero[reverse]) / weight, 0.0)

    log_product = np.zeros(num_nodes)
    parent_sum = np.zeros(num_nodes)
    np.add.at(log_product, source, log_weight)
    np.add.at(parent_sum, source, parent_term)

    scale = np.exp(np.clip(reweight + log_product, None, bpd_max_exponent))

    return 1.0 / (1.0 + scale * (1.0 + parent_sum))


def bpd(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    reweight: float = bpd_reweight,
    fraction: float = bpd_fraction,
    **_: object,
) -> list[int]:
    """
    BPD (Mugisha & Zhou, Phys. Rev. E 94:012305, 2016): belief-propagation-guided
    decimation over the minimum-feedback-vertex-set mapping, then tree breaking.

    Three stages, the same shape as Min-Sum: infer which vertices are most
    "suitable for deletion" by BP on the FVS spin model, delete a tiny fraction of
    them, repeat until the graph is a forest; then break the surviving trees. The
    inference is what separates it from CoreHD, which makes the same kind of move
    using degree alone, so `bpd` against `corehd` is a clean read on what the
    message passing buys.

    Decimation follows the paper: iterate BP to a fixed point, delete the fraction
    `f` of vertices with the highest q^0, advance. Candidates are restricted to the
    2-core, which is where every cycle lives and therefore the only place stage 1
    can make progress; a tree vertex gets a low q^0 anyway, so this narrows the
    argmax without changing it.

    `bpd_r` adds the reinsertion stage the paper runs afterwards.
    """
    neighbours = neighbour_sets(graph)
    removed = set()
    chosen = []
    limit = min(budget, graph.num_nodes)
    per_round = max(1, int(fraction * graph.num_nodes))

    while len(chosen) < limit:
        core, _ = _two_core(neighbours, removed)
        if not core:
            break

        deletable = _bpd_marginals(
            neighbours, removed, reweight, bpd_iterations, bpd_tolerance
        )
        ranked = sorted(core, key=lambda node: -deletable[node])

        for node in ranked[: min(per_round, limit - len(chosen))]:
            chosen.append(int(node))
            removed.add(int(node))

    _break_trees(neighbours, removed, chosen, limit)

    return _pad(chosen, graph, budget)


def bpd_r(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    threshold: float = 0.01,
    **_: object,
) -> list[int]:
    """BPD plus the greedy reinsertion the paper runs after tree breaking."""
    return _reinsert_and_refill(
        graph, bpd(graph, budget, diffusion_model), budget, threshold
    )


def decycling(
    graph: GraphInfo, budget: int, diffusion_model: str = "IC", **_: object
) -> list[int]:
    """
    Greedy decycling then tree breaking: the Min-Sum / BPD pipeline with a greedy
    stage 1 instead of message passing.

    Stage 1 removes the highest-degree node of the 2-core until the residual is a
    FOREST; stage 2 breaks the surviving trees at their centroids. `bpd` is the
    same two stages with the published message passing in stage 1, so comparing
    this against it isolates exactly what the inference buys over greed.

    ponytail: greedy stage 1, so this reproduces the pipeline's SHAPE and not
    Min-Sum's set sizes. Min-Sum itself is NOT implemented here: a reconstruction
    from the published equations came out erratic and worse than this greedy
    version, so `abraunst/decycler` is registered as the external route instead.
    `decycling_r` adds the third stage.
    """
    neighbours = neighbour_sets(graph)
    removed = set()
    chosen = []
    limit = min(budget, graph.num_nodes)

    # Stage 1: decycle. The 2-core is exactly the cyclic part.
    while len(chosen) < limit:
        core, degrees = _two_core(neighbours, removed)

        if not core:
            break

        node = max(core, key=lambda candidate: degrees[candidate])
        chosen.append(int(node))
        removed.add(int(node))

    # Stage 2: break the surviving trees
    _break_trees(neighbours, removed, chosen, limit)

    return _pad(chosen, graph, budget)


def decycling_r(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    threshold: float = 0.01,
    **_: object,
) -> list[int]:
    """Greedy decycling plus stage 3, the reverse-greedy reinsertion, refilled to budget."""
    return _reinsert_and_refill(
        graph, decycling(graph, budget, diffusion_model), budget, threshold
    )


def articulation_removal(
    graph: GraphInfo, budget: int, diffusion_model: str = "IC", **_: object
) -> list[int]:
    """
    Articulation points, ranked by how much of the giant component each one splits off.

    The structural object dismantling exploits (Tian, Bashan, Shi, Liu, Nat.
    Commun. 2017): a cut vertex is by definition the only route between two parts
    of its component, so deleting one is the cheapest possible cut. Recomputed
    after every removal, because cutting one open creates others.
    """
    neighbours = neighbour_sets(graph)
    removed = set()
    chosen = []

    for _ in range(min(budget, graph.num_nodes)):
        components = _components(neighbours, removed)
        largest = max(components, key=len, default=set())

        if len(largest) <= 2:
            break

        best_node, best_size = None, len(largest)
        for node in largest:
            pieces = _components(neighbours, removed | {node})
            biggest = max(
                (len(piece) for piece in pieces if piece & largest), default=0
            )
            if biggest < best_size:
                best_node, best_size = node, biggest

        if best_node is None:
            break

        chosen.append(int(best_node))
        removed.add(int(best_node))

    return _pad(chosen, graph, budget)


def _fiedler_split(
    neighbours: list[set[int]], component: list[int], quantile: float = 0.5
) -> tuple[set, set]:
    """
    Spectral bisection of one component by the Fiedler vector of its normalized
    Laplacian. Falls back to an index split when ARPACK does not converge.

    `quantile` is where along the Fiedler vector the cut is made; 0.5 is the median
    (the balanced bisection GND describes), and `egnd` sweeps it.
    """
    index = {node: position for position, node in enumerate(component)}
    rows, columns = [], []

    for node in component:
        for neighbour in neighbours[node]:
            if neighbour in index:
                rows.append(index[node])
                columns.append(index[neighbour])

    size = len(component)
    if not rows:
        half = size // 2
        return set(component[:half]), set(component[half:])

    adjacency = sp.csr_matrix(
        (np.ones(len(rows)), (rows, columns)), shape=(size, size)
    )
    degrees = np.asarray(adjacency.sum(axis=1)).flatten()
    inverse_root = sp.diags(1.0 / np.sqrt(np.maximum(degrees, 1.0)))
    normalized = sp.identity(size) - inverse_root @ adjacency @ inverse_root

    try:
        # Fixed start vector: ARPACK's default is random, which made every GND
        # variant non-reproducible across calls
        _, vectors = spla.eigsh(
            normalized, k=2, sigma=0.0, which="LM", v0=np.linspace(0.1, 1.0, size)
        )
        fiedler = vectors[:, 1]
    except Exception:
        # ARPACK fails on tiny or numerically awkward blocks; an arbitrary split
        # still produces a cut for the vertex cover to work on
        half = size // 2
        return set(component[:half]), set(component[half:])

    cut = float(np.quantile(fiedler, quantile))
    left = {component[position] for position in range(size) if fiedler[position] <= cut}

    # A degenerate Fiedler vector can put everything on one side, which leaves no
    # cut edges and stalls the recursion
    if not left or len(left) == size:
        half = size // 2
        return set(component[:half]), set(component[half:])

    return left, set(component) - left


def gnd(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    min_component: int = 8,
    cost: str = "unit",
    split_quantile: float = 0.5,
    **_: object,
) -> list[int]:
    """
    GND (Ren, Gleinig, Helbing, Antulov-Fantulin, PNAS 2019): spectral bisection of
    the largest component, then a weighted vertex cover of the cut, recursed.

    `cost="unit"` is the default and the only one comparable to everything else
    here, because our budget is CARDINALITY (`k` nodes). GND's contribution is the
    generalized `cost="degree"` instantiation, which prices removal by degree and
    so prefers many cheap separators over one expensive hub: a genuinely different
    optimum that a cardinality metric scores unfairly in both directions.
    Pass `cost="degree"` to get it, and report it against a cost budget or not at
    all.

    ponytail: the cut is covered by a greedy max-coverage-per-cost rule rather than
    the paper's LP 2-approximation, and the bisection uses the plain normalized
    Laplacian rather than their node-weighted Power Laplacian. Same two stages,
    same recursion, looser constant. `renxiaolong/Generalized-Network-Dismantling`
    is registered as an external baseline for the authors' numbers.
    """
    neighbours = neighbour_sets(graph)
    costs = (
        np.ones(graph.num_nodes, dtype=np.float64)
        if cost == "unit"
        else np.array([max(len(group), 1) for group in neighbours], dtype=np.float64)
    )
    removed = set()
    chosen = []
    limit = min(budget, graph.num_nodes)

    while len(chosen) < limit:
        components = _components(neighbours, removed)
        largest = max(components, key=len, default=set())

        if len(largest) <= min_component:
            break

        left, _ = _fiedler_split(neighbours, sorted(largest), split_quantile)
        cut = [
            (node, neighbour)
            for node in largest
            for neighbour in neighbours[node]
            if neighbour in largest and (node in left) != (neighbour in left)
        ]

        if not cut:
            break

        # Weighted vertex cover of the cut, greedy by coverage per unit cost
        uncovered = {tuple(sorted(edge)) for edge in cut}
        incident = {}
        for edge in uncovered:
            for endpoint in edge:
                incident.setdefault(endpoint, set()).add(edge)

        while uncovered and len(chosen) < limit:
            node = max(
                incident,
                key=lambda candidate: len(incident[candidate] & uncovered)
                / costs[candidate],
            )
            covers = incident[node] & uncovered

            if not covers:
                break

            uncovered -= covers
            chosen.append(int(node))
            removed.add(int(node))

    return _pad(chosen, graph, budget)


def gndr(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    threshold: float = 0.01,
    **_: object,
) -> list[int]:
    """GND plus reinsertion (GNDR): a different method from `gnd`, with different numbers."""
    return _reinsert_and_refill(
        graph, gnd(graph, budget, diffusion_model), budget, threshold
    )


def egnd(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    ensemble: int = egnd_ensemble,
    **_: object,
) -> list[int]:
    """
    EGND: an ensemble of GND cuts, keeping the one that damages the graph most.

    GND's bisection cuts at the MEDIAN of the Fiedler vector, which is one
    arbitrary point on a continuum: a slightly unbalanced cut is often far cheaper
    to cover. This sweeps the split quantile and keeps the removal set with the
    smallest residual giant component.

    Warning: EGND appears as a distinct baseline in the GDM and Artime tables but
    has **no paper and no code**, so "ensemble of GND cuts" is the entire
    published specification and the quantile sweep is our reading of it. Report it
    as our variant, not as the EGND of those tables.
    """
    neighbours = neighbour_sets(graph)
    best_set, best_size = None, None

    # Symmetric around the median, so the first member IS plain `gnd`
    offsets = np.linspace(0.0, egnd_quantile_spread, ensemble)

    for index, offset in enumerate(offsets):
        quantile = 0.5 + (offset if index % 2 else -offset)
        candidate = gnd(
            graph, budget, diffusion_model, split_quantile=float(np.clip(quantile, 0.05, 0.95))
        )
        size = _giant_size(neighbours, set(candidate))

        if best_size is None or size < best_size:
            best_set, best_size = candidate, size

    return _pad(list(best_set or []), graph, budget)


def explosive_immunization(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    candidates: int = ei_candidates,
    seed: int = 0,
    **_: object,
) -> list[int]:
    """
    Explosive Immunization (Clusella, Grassberger, Perez-Reche, Politi, PRL 2016).

    An INVERSE construction: start with every node vaccinated, then un-vaccinate,
    one at a time, the candidate whose return grows the giant component least. Run
    it until N - k nodes are back and the k still vaccinated are the removal set.

    Scored the Achlioptas way: a candidate's cost is the summed size of the
    DISTINCT clusters it would merge, which is why it beats degree rules near the
    percolation transition: a low-degree node bridging two large clusters is the
    expensive one, and a high-degree node buried inside one is free.

    ponytail: `candidates` random draws per round instead of all N, which is the
    paper's own m ~ 10^3 sampling; and one regime rather than their two, since the
    second only matters deep past the transition.
    """
    neighbours = neighbour_sets(graph)
    num_nodes = graph.num_nodes
    limit = min(budget, num_nodes)
    rng = np.random.default_rng(seed)

    parent = list(range(num_nodes))
    size = [1] * num_nodes

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]

        return node

    def union(first: int, second: int) -> None:
        first, second = find(first), find(second)
        if first == second:
            return

        if size[first] < size[second]:
            first, second = second, first

        parent[second] = first
        size[first] += size[second]

    vaccinated = set(range(num_nodes))
    active = set()

    while len(vaccinated) > limit:
        pool = list(vaccinated)
        sample = (
            pool
            if len(pool) <= candidates
            else [
                int(pool[index])
                for index in rng.choice(len(pool), size=candidates, replace=False)
            ]
        )

        best_node, best_score = None, float("inf")
        for node in sample:
            roots = {find(other) for other in neighbours[node] if other in active}
            score = 1 + sum(size[root] for root in roots)

            if score < best_score:
                best_node, best_score = node, score

        vaccinated.discard(best_node)
        active.add(best_node)
        for neighbour in neighbours[best_node]:
            if neighbour in active:
                union(best_node, neighbour)

    return sorted(int(node) for node in vaccinated)


def netshield(
    graph: GraphInfo, budget: int, diffusion_model: str = "IC", **_: object
) -> list[int]:
    """
    NetShield (Tong et al. ICDM 2010): greedily maximize the drop in the adjacency
    matrix's leading eigenvalue.

    The only member here whose objective is EPIDEMIC rather than structural: the
    epidemic threshold of both SIS and IC-like dynamics scales as 1/lambda_max, so
    shrinking lambda_max is directly shrinking the outbreak. That makes it the
    strongest non-simulation baseline for the diffusion variant this task actually
    evaluates, and the one whose
    ranking a connectivity-driven method is least likely to reproduce.

    Shield value of adding node j to S:
        v(j) = (2*lambda - A_jj) * u_j^2 - 2 * u_j * sum_{i in S} A_ij * u_i
    """
    vector = primitives.compute_centrality(graph, kind="eigenvector")
    norm = np.linalg.norm(vector)
    vector = vector / norm if norm > 0 else vector

    sources, targets = graph.edge_index[0], graph.edge_index[1]
    # Rayleigh quotient u^T A u with u normalized. edge_index already carries both
    # orientations of an undirected edge, so the sum over arcs IS u^T A u: no
    # factor of two, which would inflate the first term and turn the rule into
    # plain eigenvector centrality.
    eigenvalue = float(np.sum(vector[sources] * vector[targets]))

    neighbours = neighbour_sets(graph)
    base = (2.0 * eigenvalue) * vector**2
    penalty = np.zeros(graph.num_nodes, dtype=np.float64)
    chosen = []
    picked = set()

    for _ in range(min(budget, graph.num_nodes)):
        scores = base - 2.0 * vector * penalty
        scores[list(picked)] = float("-inf")
        node = int(np.argmax(scores))

        if node in picked:
            break

        chosen.append(node)
        picked.add(node)

        # Discount every neighbour by the overlap it now has with the chosen set
        for neighbour in neighbours[node]:
            penalty[neighbour] += vector[node]

    return _pad(chosen, graph, budget)


def acquaintance_immunization(
    graph: GraphInfo, budget: int, diffusion_model: str = "IC", seed: int = 0, **_: object
) -> list[int]:
    """
    Acquaintance immunization (Cohen, Havlin, ben-Avraham, PRL 2003): pick a random
    node, immunize a random NEIGHBOUR of it.

    The classic zero-knowledge floor. It needs no global degree information and
    still beats random by a wide margin, because following an edge lands on a node
    with probability proportional to its degree. Any method that reads the whole
    graph and does not clearly beat this has bought nothing with the information.
    """
    neighbours = neighbour_sets(graph)
    rng = np.random.default_rng(seed)
    chosen = []
    picked = set()

    for _ in range(budget * 20):
        if len(chosen) >= min(budget, graph.num_nodes):
            break

        node = int(rng.integers(graph.num_nodes))
        group = sorted(neighbours[node])

        if not group:
            continue

        target = int(group[int(rng.integers(len(group)))])
        if target not in picked:
            picked.add(target)
            chosen.append(target)

    return _pad(chosen, graph, budget)


def frontier_removal(
    graph: GraphInfo, budget: int, diffusion_model: str = "IC", outbreak: tuple[int, ...] | list[int]=(), **_: object
) -> list[int]:
    """
    Delete the susceptible boundary of the observed outbreak, highest-degree first.

    The only member of this pool that conditions on WHERE the outbreak is, and the
    control every outbreak-aware arm has to beat: the first power_grid sweep had the
    agent arm at exactly the source count (49.0) against 98-160 for every published
    dismantler, and its whole program was this one-hop ring. Without this row that
    gap reads as a method and is in fact the information asymmetry. Same rule as
    `immunization_algorithms.frontier_immunization`, reused rather than copied.
    """
    # Function-local: immunization_algorithms imports from this module
    from coding_agent.tools.immunization_algorithms import frontier_immunization

    return frontier_immunization(graph, budget, diffusion_model, outbreak=outbreak)


def random_removal(
    graph: GraphInfo, budget: int, diffusion_model: str = "IC", seed: int = 42, **_: object
) -> list[int]:
    """Uniform random removal set: the trivial floor."""
    rng = np.random.default_rng(seed)
    count = min(budget, graph.num_nodes)

    return [
        int(node) for node in rng.choice(graph.num_nodes, size=count, replace=False)
    ]


def greedy_blocking(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    outbreak: tuple = (),
    mc_runs: int = greedy_blocking_mc_runs,
    horizon: int = greedy_blocking_horizon,
    n_candidates: int = greedy_blocking_candidates,
    seed: int = 0,
    **_: object,
) -> list[int]:
    """
    Greedy marginal blocking: add the node whose deletion most reduces the SIMULATED
    spread of the outbreak. CELF with the sign flipped, and the honest strong bar.

    Every other member of this library optimizes a structural proxy; this one
    optimizes the quantity the task is actually scored on, which is why it is the
    reference, and why it is unaffordable. One pick costs
    n_candidates x mc_runs episodes, so a k=20 run is thousands of real rollouts.
    That cost is the entire argument for replacing the simulator with f_theta.

    ponytail: candidates are pre-filtered to the top `n_candidates` by adaptive
    degree rather than scanning all N. Raise it for a faithful run, and expect the
    wall clock to scale linearly.
    """
    if not outbreak:
        # Nothing to contain means nothing to measure; fall back to the strongest
        # structural rule rather than returning a set scored against noise
        return adaptive_degree(graph, budget, diffusion_model)

    chosen = []

    for _ in range(min(budget, graph.num_nodes)):
        pool = [
            node
            for node in adaptive_degree(
                graph, min(graph.num_nodes, n_candidates + len(chosen)), diffusion_model
            )
            if node not in chosen
        ][:n_candidates]

        best_node, best_spread = None, float("inf")
        for node in pool:
            spread = primitives.mc_simulate_containment(
                graph,
                list(outbreak),
                chosen + [node],
                diffusion_model,
                mc_runs=mc_runs,
                horizon=horizon,
                seed=seed,
            )
            if spread < best_spread:
                best_node, best_spread = node, spread

        if best_node is None:
            break

        chosen.append(int(best_node))

    return _pad(chosen, graph, budget)


dismantling_algorithms = {
    # degree
    "degree_removal": degree_removal,
    "adaptive_degree": adaptive_degree,
    # centrality
    "pagerank_removal": pagerank_removal,
    "betweenness_removal": betweenness_removal,
    "iterative_betweenness": iterative_betweenness,
    "approx_iterative_betweenness": approx_iterative_betweenness,
    "kshell_removal": kshell_removal,
    # percolation / physics
    "collective_influence_removal": collective_influence_removal,
    "collective_influence_r": collective_influence_r,
    "corehd": corehd,
    "corehd_r": corehd_r,
    "decycling": decycling,
    "decycling_r": decycling_r,
    "bpd": bpd,
    "bpd_r": bpd_r,
    "articulation_removal": articulation_removal,
    "explosive_immunization": explosive_immunization,
    "gnd": gnd,
    "gndr": gndr,
    "egnd": egnd,
    # spectral / epidemic
    "netshield": netshield,
    # data-aware
    "frontier_removal": frontier_removal,
    # floors
    "acquaintance_immunization": acquaintance_immunization,
    "random_removal": random_removal,
    # simulation-based
    "greedy_blocking": greedy_blocking,
}

# Simulation-based, so one run costs budget x n_candidates x mc_runs real
# episodes. Charged honestly to the arm like any other baseline, and blocked from
# generated scripts by default for the same reason `celf` is: it bypasses the
# metered evaluator.
mc_dismantling_algorithms = ("greedy_blocking",)

dismantling_algorithm_names = list(dismantling_algorithms)
