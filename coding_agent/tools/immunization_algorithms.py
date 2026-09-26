"""
Named classical EPIDEMIC-CONTROL baselines: the condition-1 floor for
`--task epidemic_control`.

Every algorithm has the signature

    (graph, budget, diffusion_model, **kw) -> list[int] | list[tuple[int, int]]

and returns what its LEVER spends: node ids under `vaccinate` and `quarantine`,
`(u, v)` arcs under `edge_cut` and `contact_reduce`. `epidemic.immunization_plan`
reconciles the two shapes into one plan, which is what keeps a published algorithm
runnable without rewriting it to know what a plan is.

This literature splits into three lines and all three are here, because they
optimize DIFFERENT objectives and a table with only one of them is a table that
cannot see its own blind spot:

  * **The physics line**: targeted degree immunization
    (Pastor-Satorras & Vespignani PRE'02), acquaintance immunization
    (Cohen PRL'03), and Holme's recalculated-degree/betweenness taxonomy. These
    are two-line heuristics, and `acquaintance` is specifically "the baseline
    that most embarrasses learned methods on sparse graphs".
  * **The spectral line**: NetShield, NetShield+, NetMelt, ProductDegree,
    EigenScore, GreedyWalk, Preciado's allocation. They minimize `lambda_1`, which
    is a MODEL-INDEPENDENT surrogate that needs no simulator at all. Warning: they
    are optimizing something we do not score them on, and that is the whole
    point: a method can win on eigendrop and lose on simulated final size, because
    `lambda_1` says nothing about WHERE the infection currently is.
  * **The simulation / data-aware line**: DAVA and DAVA-fast
    (Zhang & Prakash SDM'14), frontier immunization, and the MC greedy. These
    condition on the OBSERVED infected set, and this is the line to position
    against: DAVA makes exactly our argument (condition on the observed infection
    state and the optimal allocation changes) but does it with a dominator-tree
    heuristic on a single observed snapshot.

Three conventions everything here obeys:

  * **Undirected**, like the dismantling literature and for the same reason:
    every routine works over `containment.neighbour_sets`, the symmetrized
    view, and our directed datasets are symmetrized to be scored at all.
  * **The outbreak is an input, not a discovery.** Every member takes
    `outbreak=(...)` and the data-aware ones use it; the structural ones absorb it
    through `**kw` and ignore it. That asymmetry IS the spectral-vs-data-aware split and is
    the thing the table is meant to expose.
  * **No member may return an outbreak source.** Dosing patient zero ends the
    outbreak rather than containing it, `executor.validate_actions` rejects it, and
    `immunization_plan` filters it, but the members exclude it themselves so their
    budget is spent on legal picks rather than lost to a top-up.

None of these is a weak floor. `degree_immunization` is
the row a learned method has to beat outright: RLGN's own Table 2 has Degree and
Eigenvector tying to within 0.1 on CA-GrQc (25.5 vs 25.4) and on GEMSEC-RO (2.4 vs
2.4), which is the same heuristic-collapse the BA-100 result already showed us on
influence maximization.
"""

import numpy as np

# Brandes over the residual graph, sampled on a large graph. Shared with the
# dismantling pool rather than reimplemented: it is the same functional, and two
# copies would drift into two different `betweenness` rows across two tables.
from coding_agent.containment import core_numbers, neighbour_sets
from coding_agent.tools import primitives
from coding_agent.tools.dismantling_algorithms import _betweenness
from coding_agent.types import GraphInfo

# Explosive-style candidate sampling and the MC greedy's shortlist. A full scan is
# O(N x mc_runs) episodes per pick and does not finish past a few thousand nodes.
mc_greedy_candidates = 30
mc_greedy_runs = 20
mc_greedy_horizon = 15

# NetShield+ recomputes the leading eigenvector after every batch of `b`. The TKDE
# paper sweeps b and reports the quality/time knee around k/10; 5 is that at our
# budgets and is the value every number here is quoted at.
netshield_plus_batch = 5

# GreedyWalk scores a node or an arc by the number of closed `k`-walks through it
# (Saha et al. SDM'15). `walk_length` is their `k`; 4 is the value their scaling
# results are quoted at and the largest that stays affordable at our sizes.
walk_length = 4

# Preciado's geometric program allocates a CONTINUOUS budget per node. We discretize
# by taking the top-k of the allocation, and the allocation itself is the
# first-order perturbation of lambda_1 under a per-node rate reduction, which is the
# GP's own gradient. `iterations` is the fixed-point sweep over that gradient.
preciado_iterations = 20


def _undirected(graph: GraphInfo) -> list[set[int]]:
    return neighbour_sets(graph)


def _excluded(graph: GraphInfo, outbreak: tuple[int, ...] | list[int]) -> set[int]:
    """The outbreak's own sources, which no member may return."""
    return {int(node) for node in (outbreak or ()) if 0 <= int(node) < graph.num_nodes}


def _pad(chosen: list[int], graph: GraphInfo, budget: int, banned: set[int]) -> list[int]:
    """
    Top up a short allocation with the highest-degree nodes not already in it.

    Every structural method can run out before the budget does: the infected
    frontier is smaller than k, a dominator tree has fewer than k children, a cut
    is smaller than the budget. Returning fewer doses than every other arm reads as
    a weak method in the attack-rate column, so the shortfall is filled with the
    obvious pick and the caller reports `n_spent` either way.
    """
    picked = list(dict.fromkeys(int(node) for node in chosen if int(node) not in banned))

    if len(picked) >= budget:
        return picked[:budget]

    seen = set(picked) | banned
    degrees = primitives.compute_degree(graph)

    for node in np.argsort(-degrees):
        if len(picked) >= budget:
            break

        node = int(node)
        if node not in seen:
            seen.add(node)
            picked.append(node)

    return picked[:budget]


def _leading_eigenvector(graph: GraphInfo) -> tuple[np.ndarray, float]:
    """(unit leading eigenvector, lambda_1) of the symmetrized adjacency."""
    vector = primitives.compute_centrality(graph, kind="eigenvector")
    norm = float(np.linalg.norm(vector))
    vector = vector / norm if norm > 0 else vector

    sources, targets = graph.edge_index[0], graph.edge_index[1]
    # `edge_index` already carries both orientations of an undirected edge, so the
    # sum over arcs IS the Rayleigh quotient u^T A u: no factor of two, which
    # would inflate the shield value and collapse NetShield into plain eigenvector
    # centrality
    eigenvalue = float(np.sum(vector[sources] * vector[targets]))

    return vector, eigenvalue


def _arc_list(graph: GraphInfo) -> list[tuple[int, int]]:
    """Every undirected dyad once, as a `(u, v)` with `u < v`."""
    seen = set()

    for edge in range(graph.edge_index.shape[1]):
        source = int(graph.edge_index[0, edge])
        target = int(graph.edge_index[1, edge])

        if source == target:
            continue

        seen.add((min(source, target), max(source, target)))

    return sorted(seen)


def _rank_arcs(
    graph: GraphInfo, scores: dict, budget: int, protected: set[int]
) -> list[tuple[int, int]]:
    """
    Top-`budget` arcs by score, emitted in BOTH orientations' canonical form.

    An arc incident to an outbreak source is kept: cutting the source's own contacts
    is a legitimate quarantine of patient zero and is not the same intervention as
    dosing the node, which is what the no-patient-zero rule forbids.
    """
    ranked = sorted(scores, key=lambda arc: -scores[arc])

    return [tuple(int(node) for node in arc) for arc in ranked[:budget]]


# The physics line ---------------------------------------------------------------
def degree_immunization(
    graph: GraphInfo, budget: int, diffusion_model: str = "SIR", outbreak: tuple[int, ...] | list[int]=(), **_: object
) -> list[int]:
    """
    Targeted immunization (Pastor-Satorras & Vespignani, PRE 65:036104, 2002).

    Immunize the highest-degree nodes, computed ONCE on the intact graph. The
    founding result of this literature is that this restores a finite epidemic
    threshold on a power-law network with a vanishing immunized fraction, while
    random immunization needs a fraction approaching 1, and it is the row a learned
    method has to beat outright.
    """
    degrees = primitives.compute_degree(graph)
    banned = _excluded(graph, outbreak)

    return _pad(
        [int(node) for node in np.argsort(-degrees)], graph, budget, banned
    )


def adaptive_degree_immunization(
    graph: GraphInfo, budget: int, diffusion_model: str = "SIR", outbreak: tuple[int, ...] | list[int]=(), **_: object
) -> list[int]:
    """
    Recalculated-degree immunization (Holme, Kim, Yoon & Han, PRE 65:056109, 2002).

    The `RD` arm of their four-way initial-vs-recalculated taxonomy: dose the
    current highest-degree node, delete it, recompute. Consistently stronger than
    the static version and the source of the `HDA` baseline every later paper
    reports: removing k at once and recomputing after each are DIFFERENT
    algorithms whose numbers are not interconvertible.
    """
    neighbours = _undirected(graph)
    banned = _excluded(graph, outbreak)
    alive = {node: len(group - banned) for node, group in enumerate(neighbours)}
    chosen = []

    for _ in range(min(budget, graph.num_nodes)):
        candidates = [node for node in alive if node not in banned]
        if not candidates:
            break

        node = max(candidates, key=lambda candidate: (alive[candidate], -candidate))
        if alive[node] <= 0:
            break

        chosen.append(int(node))
        for neighbour in neighbours[node]:
            if neighbour in alive:
                alive[neighbour] -= 1

        alive.pop(node, None)

    return _pad(chosen, graph, budget, banned)


def acquaintance_immunization(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "SIR",
    outbreak: tuple[int, ...] | list[int]=(),
    seed: int = 0,
    **_: object,
) -> list[int]:
    """
    Acquaintance immunization (Cohen, Havlin & ben-Avraham, PRL 91:247901, 2003).

    Pick a random node, immunize a random NEIGHBOUR of it. It needs no global degree
    knowledge at all and still hits hubs, because following an edge lands on a node
    with probability proportional to its degree (the friendship paradox).

    It needs no global information, is two lines of code, and is the baseline that
    most embarrasses learned methods on sparse graphs, so it belongs in the table. A
    method that reads the whole graph and does not clearly beat this has bought
    nothing with the information.
    """
    neighbours = _undirected(graph)
    banned = _excluded(graph, outbreak)
    rng = np.random.default_rng(seed)
    chosen = []
    picked = set(banned)

    for _ in range(max(budget, 1) * 40):
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

    return _pad(chosen, graph, budget, banned)


def pagerank_immunization(
    graph: GraphInfo, budget: int, diffusion_model: str = "SIR", outbreak: tuple[int, ...] | list[int]=(), **_: object
) -> list[int]:
    """Top-k PageRank: the centrality control NetShield's Fig 1 reports beside `Eigs`."""
    scores = primitives.compute_pagerank(graph)
    banned = _excluded(graph, outbreak)

    return _pad([int(node) for node in np.argsort(-scores)], graph, budget, banned)


def eigenvector_immunization(
    graph: GraphInfo, budget: int, diffusion_model: str = "SIR", outbreak: tuple[int, ...] | list[int]=(), **_: object
) -> list[int]:
    """
    Top-k eigenvector centrality: NetShield's `Eigs` row and RLGN's `Eigenvector`.

    In the table rather than the appendix for one reason: NetShield's own
    contribution is that COLLECTIVE selection beats top-k of an individual score,
    and this is the top-k it beats. RLGN's Table 2 also has it tying with Degree to
    0.1 on two of five graphs, which is the heuristic-collapse signature
    worth being able to see.
    """
    scores = primitives.compute_centrality(graph, kind="eigenvector")
    banned = _excluded(graph, outbreak)

    return _pad([int(node) for node in np.argsort(-scores)], graph, budget, banned)


def betweenness_immunization(
    graph: GraphInfo, budget: int, diffusion_model: str = "SIR", outbreak: tuple[int, ...] | list[int]=(), **_: object
) -> list[int]:
    """Top-k betweenness: the `IB` arm of Holme et al.'s taxonomy."""
    scores = _betweenness(_undirected(graph), set())
    banned = _excluded(graph, outbreak)

    return _pad([int(node) for node in np.argsort(-scores)], graph, budget, banned)


def kshell_immunization(
    graph: GraphInfo, budget: int, diffusion_model: str = "SIR", outbreak: tuple[int, ...] | list[int]=(), **_: object
) -> list[int]:
    """
    Top-k by k-core number (Kitsak et al., Nature Physics 2010).

    The "influential spreaders are in the core, not the hubs" result, which is a
    genuinely different ranking from degree on a graph with a dense core and is the
    cheapest way to see whether this graph is one of those.
    """
    core = core_numbers(graph).astype(np.float64)
    banned = _excluded(graph, outbreak)

    return _pad([int(node) for node in np.argsort(-core)], graph, budget, banned)


def random_immunization(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "SIR",
    outbreak: tuple[int, ...] | list[int]=(),
    seed: int = 42,
    **_: object,
) -> list[int]:
    """
    Uniform random doses: the control Pastor-Satorras & Vespignani's whole line exists to beat.

    Not a throwaway floor here the way it is elsewhere: their 2002 result is
    precisely that on a power-law graph this needs an immunized fraction approaching
    1 to halt spread, so the GAP between this row and `degree_immunization` is the
    quantity that founded the field.
    """
    banned = _excluded(graph, outbreak)
    pool = [node for node in range(graph.num_nodes) if node not in banned]
    rng = np.random.default_rng(seed)
    count = min(budget, len(pool))

    if not pool:
        return []

    return [int(pool[index]) for index in rng.choice(len(pool), size=count, replace=False)]


# The spectral line --------------------------------------------------------------
def netshield(
    graph: GraphInfo, budget: int, diffusion_model: str = "SIR", outbreak: tuple[int, ...] | list[int]=(), **_: object
) -> list[int]:
    """
    NetShield (Tong, Prakash, Tsourakakis, Eliassi-Rad, Faloutsos & Chau, ICDM 2010).

    The canonical node baseline of the spectral line, and the one method in this
    file with a proof attached: the Shield-value

        Sv(S) = sum_i 2*lambda*u(i)^2 - sum_{i,j in S} A(i,j)*u(i)*u(j)

    is submodular, so a greedy selection is `(1 - 1/e)`-optimal against it, and
    Table 3 of the paper measures its approximation to the true eigendrop at
    0.977-1.000 across four conference co-authorship graphs [verified].

    Warning: it optimizes `lambda_1`, which is a surrogate that ignores WHERE the
    outbreak currently is. It is expected to beat every centrality here on the
    eigendrop column and it is NOT expected to beat `dava` on the attack rate once
    the outbreak is small and localized: that disagreement between the spectral
    surrogate and the simulated objective is the figure `eigendrop_vs_attack.png`
    exists to show.

    Warning: NO PUBLIC RELEASE by the authors. There is a widely-repeated claim
    that EpiLearn ships a NetShield implementation, and it is WRONG: that repo's
    tree has no shield, immunization or intervention code at all [derived,
    2026-08-05]. `external:netimm_netshield` is the one runnable third-party
    version, and this is our own.
    """
    vector, eigenvalue = _leading_eigenvector(graph)
    neighbours = _undirected(graph)
    banned = _excluded(graph, outbreak)

    base = (2.0 * eigenvalue) * vector**2
    penalty = np.zeros(graph.num_nodes, dtype=np.float64)
    chosen = []
    picked = set(banned)

    for _ in range(min(budget, graph.num_nodes)):
        scores = base - 2.0 * vector * penalty
        scores[list(picked)] = float("-inf")
        node = int(np.argmax(scores))

        if node in picked or not np.isfinite(scores[node]):
            break

        chosen.append(node)
        picked.add(node)

        # Discount every neighbour by the overlap it now has with the chosen set
        for neighbour in neighbours[node]:
            penalty[neighbour] += vector[node]

    return _pad(chosen, graph, budget, banned)


def netshield_plus(
    graph: GraphInfo, budget: int, diffusion_model: str = "SIR", outbreak: tuple[int, ...] | list[int]=(), **_: object
) -> list[int]:
    """
    NetShield+ (Chen, Tong, Prakash, Tsourakakis, Eliassi-Rad, Faloutsos & Chau, TKDE 2015).

    NetShield in batches of `b`, recomputing the leading eigenvector of the REDUCED
    graph after each batch. The whole content of the journal version, and it exists
    because NetShield's eigenvector is computed once on the intact graph, after a
    few high-`u` nodes are gone, that vector no longer describes the graph the next
    pick is being made on.

    A separate entry rather than a flag on `netshield` for the same reason the
    dismantling library keeps reinsertion variants apart: `X` and `X+` get cited
    under one name and are not the same method.
    """
    banned = _excluded(graph, outbreak)
    removed = set(banned)
    chosen = []

    while len(chosen) < min(budget, graph.num_nodes):
        batch = min(netshield_plus_batch, budget - len(chosen))
        view = _reduced_graph(graph, removed)
        picks = netshield(view, batch, diffusion_model, outbreak=sorted(removed))
        fresh = [int(node) for node in picks if node not in removed]

        if not fresh:
            break

        chosen += fresh
        removed |= set(fresh)

    return _pad(chosen, graph, budget, banned)


def _reduced_graph(graph: GraphInfo, removed: set[int]) -> GraphInfo:
    """
    `G - S` as a GraphInfo on the SAME node ids.

    Node ids are preserved rather than compacted, so a pick made on the reduced
    graph is directly a pick on the original and no index map has to be threaded
    through. The removed nodes become isolated, which is what every score here
    already treats as "not a candidate".
    """
    if not removed:
        return graph

    keep = np.array(
        [
            int(graph.edge_index[0, edge]) not in removed
            and int(graph.edge_index[1, edge]) not in removed
            for edge in range(graph.edge_index.shape[1])
        ],
        dtype=bool,
    )

    return GraphInfo(
        num_nodes=graph.num_nodes,
        edge_index=graph.edge_index[:, keep],
        ic_probs=graph.ic_probs[keep],
        directed=graph.directed,
    )


def greedy_walk(
    graph: GraphInfo, budget: int, diffusion_model: str = "SIR", outbreak: tuple[int, ...] | list[int]=(), **_: object
) -> list[int]:
    """
    GreedyWalk, node variant (SRMN): Saha, Adiga, Prakash & Vullikanti, SDM 2015.

    Score a node by the number of closed `k`-walks through it and remove greedily,
    recomputing after each removal. The first spectral method with approximation
    guarantees, and the walk count is the natural surrogate: `lambda_1^k` is the
    dominant term of `trace(A^k)`, so the walks through a node bound its
    contribution to the spectral radius.

    Computed by repeated sparse matvec rather than by forming `A^k`, so it stays
    `O(k * E)` per removal.

    Warning: the paper's code link is a `tinyurl` printed in the PDF and is
    unverified. `allogn/Network-Immunization`'s README states independently
    that the Walk8 solver was supplied by these authors and is MATLAB, so this is a
    reimplementation from the paper's prose and is labelled as one.
    """
    banned = _excluded(graph, outbreak)
    removed = set(banned)
    chosen = []

    for _ in range(min(budget, graph.num_nodes)):
        scores = _closed_walks(graph, removed)
        scores[list(removed)] = -1.0
        node = int(np.argmax(scores))

        if scores[node] <= 0.0:
            break

        chosen.append(node)
        removed.add(node)

    return _pad(chosen, graph, budget, banned)


def _closed_walks(graph: GraphInfo, removed: set[int]) -> np.ndarray:
    """
    Closed `walk_length`-walks through each node, up to a constant.

    `(A^(k/2) 1)^2` elementwise for even `k`: the number of walks of length `k/2`
    leaving a node, squared, is the number of closed `k`-walks through it under the
    symmetric adjacency. Kept as repeated matvec so nothing dense is ever formed.
    """
    num_nodes = graph.num_nodes
    alive = np.ones(num_nodes, dtype=np.float64)
    alive[list(removed)] = 0.0

    sources = graph.edge_index[0]
    targets = graph.edge_index[1]
    mask = alive[sources] * alive[targets]

    vector = alive.copy()
    for _ in range(max(1, walk_length // 2)):
        following = np.zeros(num_nodes, dtype=np.float64)
        np.add.at(following, targets, vector[sources] * mask)
        # Normalize each sweep so a high-degree graph does not overflow float64
        norm = following.max()
        vector = following / norm if norm > 0 else following

    return vector**2


def preciado_allocation(
    graph: GraphInfo, budget: int, diffusion_model: str = "SIR", outbreak: tuple[int, ...] | list[int]=(), **_: object
) -> list[int]:
    """
    Optimal resource allocation, discretized (Preciado, Zargham, Enyioha, Jadbabaie
    & Pappas, IEEE TCNS 1(1), 2014).

    Their contribution is a GEOMETRIC PROGRAM (convex, so globally optimal) that
    allocates a continuous amount of vaccine and antidote per node under a budget.
    Our budget is a cardinality budget (`k` doses), so what runs here is the GP's
    own first-order structure discretized: the marginal value of protecting node `v`
    is `u(v)^2` weighted by its residual degree, iterated to a fixed point, then
    thresholded at the top `k`.

    Labelled a DISCRETIZATION rather than the method, and the difference is real,
    the same way cost-weighted and cardinality budgets differ in the dismantling
    literature. Their own instance is 56 airports with passenger weights, which is
    not reconstructable, so there is no cell of theirs to match anyway.

    Warning: no public code.
    """
    banned = _excluded(graph, outbreak)
    vector, _ = _leading_eigenvector(graph)
    neighbours = _undirected(graph)
    degrees = np.array(
        [max(len(group), 1) for group in neighbours], dtype=np.float64
    )

    allocation = vector**2 * degrees
    for _ in range(preciado_iterations):
        pressure = np.zeros(graph.num_nodes, dtype=np.float64)
        for node, group in enumerate(neighbours):
            for neighbour in group:
                pressure[node] += allocation[neighbour]

        allocation = vector**2 * (degrees + pressure)
        norm = allocation.max()
        allocation = allocation / norm if norm > 0 else allocation

    return _pad(
        [int(node) for node in np.argsort(-allocation)], graph, budget, banned
    )


# The data-aware / simulation line -----------------------------------------------
def dava(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "SIR",
    outbreak: tuple[int, ...] | list[int]=(),
    fast: bool = False,
    **_: object,
) -> list[tuple] | list[int]:
    """
    DAVA (Zhang & Prakash, SDM 2014): data-aware vaccine allocation.

    Warning: THE ROW THIS TASK IS POSITIONED AGAINST. NetShield optimizes
    `lambda_1`, needs no simulator, and runs in milliseconds: we cannot beat it on
    its own metric and should not try. DAVA makes exactly our argument (condition on
    the OBSERVED infection state and the optimal allocation changes) but does it
    with a dominator-tree heuristic on a single observed snapshot. A learned
    action-conditioned model is the natural generalization: same claim, learned
    rather than hand-derived, and it extends to multi-step allocation where DAVA
    does not.

    The algorithm, from the paper:

      1. Merge every infected node into one SUPERSEED, with the arc `(superseed, n)`
         carrying `1 - prod(1 - p(i, n))` over the infected `i` adjacent to `n`,
         the probability at least one of them reaches `n`.
      2. Build the DOMINATOR TREE of the reachable subgraph rooted at the superseed.
         A node dominates everything below it: cutting it cuts every path.
      3. The benefit of dosing a child `c` of the root is the expected number of
         nodes saved, which on a tree is `p(root -> c)` times the sum over `c`'s
         subtree of the product of edge probabilities down to each descendant.
      4. Take the best, delete it, and rebuild. `fast=True` (DAVA-fast) skips the
         rebuild and takes the top `k` from one tree, which is the near-linear
         variant.

    Warning: no public code by the authors. `external:netimm_dava` is
    `allogn/Network-Immunization`'s implementation and is the cross-check on this
    one; both are readings of the same prose and neither is authoritative.
    """
    banned = _excluded(graph, outbreak)
    infected = sorted(banned)

    if not infected:
        # With no observed outbreak DAVA has nothing to condition on and degenerates
        # to a structural method; saying so beats returning a silently arbitrary set
        return degree_immunization(graph, budget, diffusion_model)

    probabilities = {}
    for edge in range(graph.edge_index.shape[1]):
        source = int(graph.edge_index[0, edge])
        target = int(graph.edge_index[1, edge])
        probabilities[(source, target)] = float(graph.ic_probs[edge])

    removed = set()
    chosen = []
    rounds = 1 if fast else min(budget, graph.num_nodes)

    for _ in range(rounds):
        tree, weights = _dominator_tree(graph, infected, probabilities, removed)
        if not tree:
            break

        ranked = sorted(
            (
                (_subtree_benefit(tree, weights, child), child)
                for child in tree.get(None, ())
            ),
            reverse=True,
        )
        if not ranked:
            break

        take = budget - len(chosen) if fast else 1
        for _, node in ranked[: max(take, 1)]:
            if node not in removed and node not in banned:
                chosen.append(int(node))
                removed.add(int(node))

        if len(chosen) >= budget:
            break

    return _pad(chosen, graph, budget, banned)


def dava_fast(
    graph: GraphInfo, budget: int, diffusion_model: str = "SIR", outbreak: tuple[int, ...] | list[int]=(), **_: object
) -> list[int]:
    """DAVA-fast: one dominator tree, top `k` children. The near-linear variant."""
    return dava(graph, budget, diffusion_model, outbreak=outbreak, fast=True)


def _dominator_tree(
    graph: GraphInfo, infected: list[int], probabilities: dict, removed: set[int]
) -> tuple[dict, dict]:
    """
    (children map, arc-probability map) of the dominator tree rooted at the superseed.

    `None` is the superseed's key. Built by the iterative Cooper-Harvey-Kennedy
    algorithm over the BFS order of the reachable subgraph: `O(N * d)` in practice
    and needing no external dependency, where networkx's `immediate_dominators`
    would need the superseed materialized into a graph object.
    """
    infected_set = {int(node) for node in infected} - removed
    if not infected_set:
        return {}, {}

    # Step 1: the superseed's arcs, with the "at least one infected neighbour
    # reaches n" probability
    root_weight = {}
    for node in infected_set:
        for neighbour in set(graph.out_neighbors(node)) | set(graph.in_neighbors(node)):
            neighbour = int(neighbour)
            if neighbour in infected_set or neighbour in removed:
                continue

            probability = probabilities.get(
                (node, neighbour), probabilities.get((neighbour, node), 0.0)
            )
            root_weight[neighbour] = 1.0 - (1.0 - root_weight.get(neighbour, 0.0)) * (
                1.0 - probability
            )

    if not root_weight:
        return {}, {}

    # BFS over the susceptible subgraph from the superseed's children
    order = []
    seen = set(root_weight)
    queue = sorted(root_weight)

    while queue:
        node = queue.pop(0)
        order.append(node)

        for neighbour in set(graph.out_neighbors(node)) | set(graph.in_neighbors(node)):
            neighbour = int(neighbour)
            if neighbour in seen or neighbour in infected_set or neighbour in removed:
                continue

            seen.add(neighbour)
            queue.append(neighbour)

    position = {node: index for index, node in enumerate(order)}
    predecessors = {node: [] for node in order}

    for node in order:
        for neighbour in set(graph.out_neighbors(node)) | set(graph.in_neighbors(node)):
            neighbour = int(neighbour)
            if neighbour in position:
                predecessors[neighbour].append(node)

    # `-1` stands for the superseed in the immediate-dominator table
    dominator = {node: (-1 if node in root_weight else None) for node in order}

    changed = True
    while changed:
        changed = False

        for node in order:
            candidates = [
                predecessors[node][index]
                for index in range(len(predecessors[node]))
                if dominator.get(predecessors[node][index]) is not None
            ]
            if node in root_weight:
                candidates.append(-1)

            if not candidates:
                continue

            new = candidates[0]
            for candidate in candidates[1:]:
                new = _intersect(new, candidate, dominator, position)

            if dominator.get(node) != new:
                dominator[node] = new
                changed = True

    children = {}
    weights = {}

    for node, parent in dominator.items():
        if parent is None:
            continue

        key = None if parent == -1 else parent
        children.setdefault(key, []).append(node)
        weights[node] = (
            root_weight.get(node, 0.0)
            if parent == -1
            else probabilities.get(
                (parent, node), probabilities.get((node, parent), 0.0)
            )
        )

    return children, weights


def _intersect(first: int, second: int, dominator: dict, position: dict) -> int:
    """Cooper-Harvey-Kennedy's `intersect`, walking up by reverse BFS order."""

    def rank(node: int) -> int:
        return -1 if node == -1 else position.get(node, 1 << 30)

    while first != second:
        while rank(first) > rank(second):
            following = dominator.get(first)
            if following is None:
                return second

            first = following

        while rank(second) > rank(first):
            following = dominator.get(second)
            if following is None:
                return first

            second = following

    return first


def _subtree_benefit(children: dict, weights: dict, node: int) -> float:
    """
    Expected nodes saved by cutting `node`: itself plus its discounted subtree.

    Iterative rather than recursive, because a dominator tree on a path-like graph
    is as deep as the graph and Python's recursion limit is 1000: the reference
    implementation recurses and falls over on exactly those graphs.
    """
    order = []
    stack = [node]
    seen = {node}

    while stack:
        current = stack.pop()
        order.append(current)

        for child in children.get(current, ()):
            if child not in seen:
                seen.add(child)
                stack.append(child)

    benefit = {}
    for current in reversed(order):
        total = 1.0
        for child in children.get(current, ()):
            if child in benefit:
                total += benefit[child] * weights.get(child, 0.0)

        benefit[current] = total

    return benefit[node] * weights.get(node, 0.0)


def frontier_immunization(
    graph: GraphInfo, budget: int, diffusion_model: str = "SIR", outbreak: tuple[int, ...] | list[int]=(), **_: object
) -> list[int]:
    """
    Dose the susceptible boundary of the observed outbreak, highest-degree first.

    The simplest data-aware rule there is, and the control DAVA has to beat to have
    contributed anything: it conditions on WHERE the outbreak is, which is the whole
    spectral-vs-data-aware distinction, but spends nothing on figuring out which
    boundary nodes matter. If `dava` does not clearly beat this, its dominator tree
    bought nothing.
    """
    banned = _excluded(graph, outbreak)
    if not banned:
        return degree_immunization(graph, budget, diffusion_model)

    degrees = primitives.compute_degree(graph)
    ring = []
    seen = set(banned)
    wave = set(banned)

    # Grow outward one hop at a time so the nearest boundary is exhausted first
    while len(ring) < budget and wave:
        following = set()
        for node in sorted(wave):
            for neighbour in set(graph.out_neighbors(node)) | set(
                graph.in_neighbors(node)
            ):
                if int(neighbour) not in seen:
                    following.add(int(neighbour))

        if not following:
            break

        seen |= following
        ring += sorted(following, key=lambda node: -degrees[node])
        wave = following

    return _pad(ring, graph, budget, banned)


def mc_greedy_immunization(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "SIR",
    outbreak: tuple[int, ...] | list[int]=(),
    n_candidates: int = mc_greedy_candidates,
    mc_runs: int = mc_greedy_runs,
    horizon: int = mc_greedy_horizon,
    seed: int = 0,
    **_: object,
) -> list[int]:
    """
    Greedy on the SIMULATED attack rate: add the dose that lowers it most, repeat.

    The objective every other member here approximates, computed directly, and
    therefore the practical ceiling for a classical method: the problem is
    NP-hard and, unlike influence maximization, NOT submodular in general, so this
    carries no approximation guarantee and is a strong heuristic rather than a
    `(1 - 1/e)` bound.

    Candidates are shortlisted by recalculated degree because a full scan is
    `O(N * mc_runs)` episodes per pick. `mc_runs` below ~20 puts the difference
    between two candidates inside the sampling noise and the greedy chases it, which
    is the same winner's-curse failure the native arm has at `--native-mc-runs 1`.
    """
    banned = _excluded(graph, outbreak)
    chosen = []

    for _ in range(min(budget, graph.num_nodes)):
        pool = [
            node
            for node in adaptive_degree_immunization(
                graph,
                min(graph.num_nodes, n_candidates + len(chosen)),
                diffusion_model,
                outbreak=outbreak,
            )
            if node not in chosen
        ][:n_candidates]

        best_node, best_attack = None, float("inf")
        for node in pool:
            attack, _ = primitives.mc_simulate_epidemic(
                graph,
                list(banned),
                chosen + [node],
                diffusion_model,
                mc_runs=mc_runs,
                horizon=horizon,
                seed=seed,
            )
            if attack < best_attack:
                best_node, best_attack = node, attack

        if best_node is None:
            break

        chosen.append(int(best_node))

    return _pad(chosen, graph, budget, banned)


# The edge levers ----------------------------------------------------------------
def netmelt(
    graph: GraphInfo, budget: int, diffusion_model: str = "SIR", outbreak: tuple[int, ...] | list[int]=(), **_: object
) -> list[tuple[int, int]]:
    """
    NetMelt (Tong, Prakash, Eliassi-Rad, Faloutsos & Faloutsos, CIKM 2012 best paper).

    The EDGE counterpart of NetShield: delete `k` arcs to minimize `lambda_1`. The
    score of arc `(i, j)` is `u(i) * v(j)`: the product of the left and right
    leading eigenvector entries, which on a symmetric adjacency is `u(i) * u(j)`.
    This is the canonical edge baseline of the spectral line.

    Warning: no public code found.
    """
    vector, _ = _leading_eigenvector(graph)
    scores = {arc: vector[arc[0]] * vector[arc[1]] for arc in _arc_list(graph)}

    return _rank_arcs(graph, scores, budget, _excluded(graph, outbreak))


def product_degree(
    graph: GraphInfo, budget: int, diffusion_model: str = "SIR", outbreak: tuple[int, ...] | list[int]=(), **_: object
) -> list[tuple[int, int]]:
    """
    ProductDegree (Van Mieghem et al., PRE 84:016101, 2011): score arc `(u, v)` by `deg(u) * deg(v)`.

    Their paper proves the edge version NP-hard and introduces this and
    `eigen_score` as the two heuristics every later paper baselines against. It
    needs no eigen-decomposition at all, which is why it survives on graphs where
    `netmelt` does not.
    """
    degrees = primitives.compute_degree(graph)
    scores = {arc: degrees[arc[0]] * degrees[arc[1]] for arc in _arc_list(graph)}

    return _rank_arcs(graph, scores, budget, _excluded(graph, outbreak))


def eigen_score(
    graph: GraphInfo, budget: int, diffusion_model: str = "SIR", outbreak: tuple[int, ...] | list[int]=(), **_: object
) -> list[tuple[int, int]]:
    """
    EigenScore (Van Mieghem et al., PRE 84:016101, 2011): score arc `(u, v)` by `|x(u) * x(v)|`.

    Algebraically the same expression as `netmelt` on a symmetric graph, and kept as
    its own row because two papers proposing the same rule under different names is
    a fact worth being able to read off a table rather than one a reader has to
    already know. Any gap between
    the two rows here is our eigenvector solver's tolerance, not a method
    difference.
    """
    vector = primitives.compute_centrality(graph, kind="eigenvector")
    scores = {arc: abs(vector[arc[0]] * vector[arc[1]]) for arc in _arc_list(graph)}

    return _rank_arcs(graph, scores, budget, _excluded(graph, outbreak))


def greedy_walk_edge(
    graph: GraphInfo, budget: int, diffusion_model: str = "SIR", outbreak: tuple[int, ...] | list[int]=(), **_: object
) -> list[tuple[int, int]]:
    """
    GreedyWalk, edge variant (SRME): Saha et al., SDM 2015.

    Score an arc by the number of closed `k`-walks through it, which for arc
    `(u, v)` is the walks reaching `u` times the walks leaving `v`. Adaptive: the
    scores are recomputed after each cut, which is what separates it from
    `eigen_score`'s single pass.
    """
    walks = _closed_walks(graph, set())
    cut = set()
    chosen = []

    for _ in range(min(budget, graph.edge_index.shape[1])):
        best_arc, best_score = None, -1.0

        for arc in _arc_list(graph):
            if arc in cut:
                continue

            score = float(walks[arc[0]] * walks[arc[1]])
            if score > best_score:
                best_arc, best_score = arc, score

        if best_arc is None:
            break

        chosen.append(best_arc)
        cut.add(best_arc)
        # Discount both endpoints so the next cut is not the same hub's next arc
        walks[best_arc[0]] *= 0.5
        walks[best_arc[1]] *= 0.5

    return chosen


def edge_betweenness_cut(
    graph: GraphInfo, budget: int, diffusion_model: str = "SIR", outbreak: tuple[int, ...] | list[int]=(), **_: object
) -> list[tuple[int, int]]:
    """
    Cut the highest edge-betweenness arcs: the bridges, approximated by endpoint betweenness.

    A structural control for the edge levers, and the direct analogue of
    `betweenness_immunization`. The endpoint product is an approximation of true
    edge betweenness and is used because the exact version is `O(N * E)` per arc;
    the two agree on which arcs are bridges, which is what this row is for.
    """
    scores_by_node = _betweenness(_undirected(graph), set())
    scores = {
        arc: scores_by_node[arc[0]] * scores_by_node[arc[1]]
        for arc in _arc_list(graph)
    }

    return _rank_arcs(graph, scores, budget, _excluded(graph, outbreak))


def frontier_edge_cut(
    graph: GraphInfo, budget: int, diffusion_model: str = "SIR", outbreak: tuple[int, ...] | list[int]=(), **_: object
) -> list[tuple[int, int]]:
    """
    Cut the arcs LEAVING the observed outbreak, highest-degree destination first.

    The data-aware edge control, and the edge lever's answer to
    `frontier_immunization`: it conditions on where the outbreak is rather than on
    the graph's global structure. On a small localized outbreak it should beat
    `netmelt` outright, which is the spectral-surrogate trap in its sharpest form:
    the spectral method is optimizing a quantity that does not know where the
    infection is.
    """
    banned = _excluded(graph, outbreak)
    if not banned:
        return product_degree(graph, budget, diffusion_model)

    degrees = primitives.compute_degree(graph)
    scores = {}

    for arc in _arc_list(graph):
        inside = (arc[0] in banned, arc[1] in banned)

        if inside[0] == inside[1]:
            continue

        outward = arc[1] if inside[0] else arc[0]
        scores[arc] = float(degrees[outward])

    if len(scores) < budget:
        # Not enough boundary arcs: fall back to the structural ranking for the rest
        for arc in _arc_list(graph):
            scores.setdefault(arc, -1.0 + 1e-9 * degrees[arc[0]] * degrees[arc[1]])

    return _rank_arcs(graph, scores, budget, banned)


def random_edge_cut(
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "SIR",
    outbreak: tuple[int, ...] | list[int]=(),
    seed: int = 42,
    **_: object,
) -> list[tuple[int, int]]:
    """Uniform random arc cuts: the edge levers' floor."""
    arcs = _arc_list(graph)
    if not arcs:
        return []

    rng = np.random.default_rng(seed)
    count = min(budget, len(arcs))

    return [arcs[int(index)] for index in rng.choice(len(arcs), size=count, replace=False)]


immunization_algorithms = {
    # physics line
    "degree_immunization": degree_immunization,
    "adaptive_degree_immunization": adaptive_degree_immunization,
    "acquaintance_immunization": acquaintance_immunization,
    "pagerank_immunization": pagerank_immunization,
    "eigenvector_immunization": eigenvector_immunization,
    "betweenness_immunization": betweenness_immunization,
    "kshell_immunization": kshell_immunization,
    "random_immunization": random_immunization,
    # spectral line
    "netshield": netshield,
    "netshield_plus": netshield_plus,
    "greedy_walk": greedy_walk,
    "preciado_allocation": preciado_allocation,
    # data-aware / simulation line
    "dava": dava,
    "dava_fast": dava_fast,
    "frontier_immunization": frontier_immunization,
    "mc_greedy_immunization": mc_greedy_immunization,
    # edge levers
    "netmelt": netmelt,
    "product_degree": product_degree,
    "eigen_score": eigen_score,
    "greedy_walk_edge": greedy_walk_edge,
    "edge_betweenness_cut": edge_betweenness_cut,
    "frontier_edge_cut": frontier_edge_cut,
    "random_edge_cut": random_edge_cut,
}

# What each member's OUTPUT is, so a lever can never be handed something it cannot
# emit. `epidemic.immunization_plan` reconciles the two shapes into one plan, but a
# `node` selector under `--epi-lever edge_cut` would hand back node ids the arm has
# no op for, and the executor would reject every bag.
immunization_shape = {
    name: ("arc" if name in (
        "netmelt",
        "product_degree",
        "eigen_score",
        "greedy_walk_edge",
        "edge_betweenness_cut",
        "frontier_edge_cut",
        "random_edge_cut",
    ) else "node")
    for name in immunization_algorithms
}

# Which lever each member belongs to, for the error message when one is run under
# the wrong one. `vaccinate` and `quarantine` share every node selector: they
# differ in what the HARNESS does with the pick, not in how it is chosen.
immunization_levers = {
    name: ("edge_cut" if shape == "arc" else "vaccinate")
    for name, shape in immunization_shape.items()
}

# Simulation-based, so one run costs budget x n_candidates x mc_runs real episodes.
# Charged honestly to the arm like any other baseline, and blocked from generated
# scripts by default for the same reason `celf` and `greedy_blocking` are: it
# bypasses the metered evaluator.
mc_immunization_algorithms = ("mc_greedy_immunization",)

immunization_algorithm_names = list(immunization_algorithms)


def emittable(name: str, lever: str) -> bool:
    """Whether `name`'s output is something an arm on `lever` may emit."""
    from coding_agent.epidemic import lever_shape

    return immunization_shape[name] == lever_shape[lever]


# Condition-1 pool per lever, each led by the row that actually has to be beaten.
#
#   * `vaccinate` / `quarantine` lead with `degree_immunization` (RLGN's own table
#     has Degree tying Eigenvector to 0.1 on two of five graphs) and carry
#     `acquaintance_immunization` because it is the row that most embarrasses
#     learned methods. `netshield` and `dava` are the two published
#     methods this task is positioned between, and `random_immunization` is not a
#     throwaway floor: the GAP between it and degree is Pastor-Satorras &
#     Vespignani's founding result.
#   * The edge levers lead with `netmelt`, the canonical spectral edge baseline, and
#     carry `frontier_edge_cut` as the data-aware control that should beat it on a
#     localized outbreak (the spectral-surrogate trap in its sharpest form).
#
# `mc_greedy_immunization` is deliberately absent for the same reason `celf`,
# `greedy_blocking` and `resim_greedy` are: it re-simulates every candidate on its
# own private simulator and would dominate startup for a table that exists to set a
# bar. Run it as its own --baselines arm when you want its number.
default_immunization_baselines = {
    "vaccinate": (
        "degree_immunization",
        "adaptive_degree_immunization",
        "acquaintance_immunization",
        "netshield",
        "netshield_plus",
        "dava",
        "dava_fast",
        "frontier_immunization",
        "greedy_walk",
        "eigenvector_immunization",
        "kshell_immunization",
        "random_immunization",
    ),
    "quarantine": (
        "degree_immunization",
        "adaptive_degree_immunization",
        "acquaintance_immunization",
        "netshield",
        "dava",
        "frontier_immunization",
        "betweenness_immunization",
        "random_immunization",
    ),
    "edge_cut": (
        "netmelt",
        "product_degree",
        "eigen_score",
        "greedy_walk_edge",
        "frontier_edge_cut",
        "edge_betweenness_cut",
        "random_edge_cut",
    ),
    "contact_reduce": (
        "netmelt",
        "product_degree",
        "eigen_score",
        "greedy_walk_edge",
        "frontier_edge_cut",
        "random_edge_cut",
    ),
}
