"""
Named classical SOURCE LOCALIZATION baselines: the condition-1 floor for
`--task source_localization`.

Every algorithm has the signature

    (graph, observation, budget, **kw) -> list[int]

and returns a SOURCE set of size `budget`: the nodes it believes STARTED the
cascade, given the observed diffusion state `observation` (a float vector in
[0, 1], either the MC marginal or a binarized single draw). That is the whole
difference from `algorithms.py`, whose members return a seed set to maximize
with, and `dismantling_algorithms.py`, whose members return nodes to delete.

Each name also has a paired scorer in `localization_scorers` returning a per-node
score vector. The two answer different questions and the literature reports both:
F1 scores the SET the selector returns, AUC scores the RANKING the scorer
produces. Where a method's selection is
intrinsically sequential (NETSLEUTH's deflation, OJC's ball cover) the two are
deliberately not top-k of one another, and that is the honest reading.

None of these is a weak floor. In ascending order of danger:

  1. `random_sources`, the Comin-Costa infected-subgraph centralities: free wins.
  2. **`lpsi` (LPSI, AAAI 2017)**: SIDSL's Table 1 puts this 2017 label-propagation
     method at F1 **0.544** on Digg, beating both SL-VAE (0.479) and DDMSL (0.517)
     [verified]. It has no learning whatsoever and it sits INSIDE the coding
     agent's expressible space, so "the search rediscovers LPSI" is the realistic
     floor and anything that does not clear it has not cleared the bar.
  3. **`netsleuth`**: the multi-source MDL reference, and the only classical member
     that beats LPSI on Power Grid in SL-VAE's Table 1 (F1 0.5428 vs 0.4737).
  4. `rumor_centrality` / `jordan_center`: SINGLE-SOURCE estimators. They rank all
     N nodes and are exact on trees; evaluated as multi-source at k = 10% of N they
     score near zero by construction, which is a property of the protocol
     rather than of the method. They are here because they are the founding
     estimators of the field and because a `--budgets 1` arm makes them admissible.

**These are honest condition-1 arms, not substitutes for the authors' code.** Each
docstring states where it deviates from its paper, and three deviations are shared
widely enough to name up front:

  * **`k` is given, never inferred.** The harness hands every member the source
    count, which removes NETSLEUTH's MDL half outright.
  * **Full observation only.** Our episodes record a state snapshot, so the
    sparse-observer methods (OJC here, Pinto, Thiran, Vetterli not implemented at
    all) run outside the regime they were designed for.
  * **Bounded candidate pools** wherever a method costs one solve per candidate
    (`dmp_localize`, `dynamic_age`, `resim_greedy`). The bounds are module globals
    at the top of this file, not buried constants.

Two published classical methods are absent and neither is an
oversight. **Pinto, Thiran, Vetterli (2012)** estimates from per-node ARRIVAL TIMES
at a sparse observer set; our transitions record states, not timestamps, so it has
no input to consume without a generation change. **Belief propagation (Altarelli
et al. 2014)** is a full posterior over initial conditions on a time-unrolled
factor graph; `dmp_localize` is its tractable relative and is implemented instead.
The published LEARNED methods are registered as external repos in
`baselines/registry.py` rather than reimplemented here.

Three conventions everything here obeys:

  * **Undirected.** Like the dismantling literature, essentially every published
    source-localization method is defined on the undirected contact graph, so each
    routine works over `containment.neighbour_sets` (the symmetrized view). Our
    directed datasets are symmetrized to be scored at all.
  * **The infected subgraph is the search space.** A node the observation says was
    never infected cannot be a source under a progressive cascade, so every method
    restricts to `observation >= infected_threshold` and only tops up outside it
    when the infected set is smaller than the budget.
  * **No labels.** These run at inference time on `(G, y, k)` alone. The ground
    truth source set exists only in the outer loop's reward, never here.
"""

import numpy as np

from coding_agent.containment import neighbour_sets
from coding_agent.tools import primitives
from coding_agent.types import GraphInfo

# An observation entry at or above this counts as "this node was infected". Our
# marginals are continuous, so the threshold is what turns y into the infected
# SET every classical method is defined over; 0.5 is the natural cut for a
# binarized draw and for an MC marginal alike.
infected_threshold = 0.5

# LPSI's diffusion coefficient. Wang et al. use alpha = 0.5 in the paper and
# report the method is insensitive over [0.3, 0.9]; 0.5 is their value.
lpsi_alpha = 0.5
lpsi_iterations = 100
lpsi_tolerance = 1e-6

# Power-iteration budget for the spectral methods (NETSLEUTH's submatrix
# Laplacian, dynamical age's largest adjacency eigenvalue)
power_iterations = 200
power_tolerance = 1e-9
norm_floor = 1e-12

# Dynamical age recomputes the leading eigenvalue once per candidate, so the
# candidate pool is capped rather than scanning the whole infected subgraph
dynamic_age_candidates = 200

# DMP scores a candidate by running a deterministic mean-field IC forward from it
# and comparing against the observation. One forward is O(T x E), so the pool is
# bounded and the depth is the observed cascade's own reach.
dmp_candidates = 40
dmp_steps = 10
probability_floor = 1e-9

# The re-simulation greedy scores every candidate against a full forward per pick.
# Bounded on both axes for the same reason `greedy_blocking` is.
resim_candidates = 30
resim_mc_runs = 20
resim_horizon = 15


def infected_set(observation: np.ndarray) -> list[int]:
    """Nodes the observation reports as infected, in id order."""
    return [int(node) for node in np.flatnonzero(np.asarray(observation) >= infected_threshold)]


def _restricted(scores: np.ndarray, observation: np.ndarray) -> np.ndarray:
    """
    Push every uninfected node below every infected one, without losing its order.

    A source that the observation says was never infected is impossible under a
    progressive cascade, so this is a hard constraint rather than a preference,
    but the uninfected nodes still have to be RANKED for AUC, so they are shifted
    below the infected block instead of being zeroed into one tie.
    """
    scores = np.asarray(scores, dtype=np.float64)
    infected_mask = np.asarray(observation) >= infected_threshold

    if not infected_mask.any():
        return scores

    span = float(scores.max() - scores.min()) + 1.0

    return scores - (~infected_mask).astype(np.float64) * span


def _top_k(scores: np.ndarray, budget: int) -> list[int]:
    """Highest-scoring `budget` nodes, ties broken by node id for determinism."""
    scores = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((np.arange(scores.size), -scores))

    return [int(node) for node in order[: max(0, budget)]]


def _pad(chosen: list[int], scores: np.ndarray, budget: int) -> list[int]:
    """
    Top up a short source set from the highest-scoring nodes not already in it.

    Every constrained method can run out before the budget does: LPSI finds fewer
    local maxima than k, OJC's cover is smaller, the infected set itself may be
    smaller than k. A short set silently under-spends the budget and makes the
    arm's precision look better than the protocol allows.
    """
    if len(chosen) >= budget:
        return chosen[:budget]

    picked = set(chosen)
    for node in _top_k(scores, len(scores)):
        if len(chosen) >= budget:
            break

        if node not in picked:
            picked.add(node)
            chosen.append(int(node))

    return chosen


def _components(neighbours: list[set[int]], members: set[int]) -> list[list[int]]:
    """Connected components of the subgraph induced on `members`."""
    seen = set()
    found = []

    for start in sorted(members):
        if start in seen:
            continue

        component = [start]
        seen.add(start)
        queue = [start]

        while queue:
            node = queue.pop()
            for neighbour in neighbours[node]:
                if neighbour in members and neighbour not in seen:
                    seen.add(neighbour)
                    component.append(neighbour)
                    queue.append(neighbour)

        found.append(sorted(component))

    return found


def _bfs_distances(
    neighbours: list[set[int]], source: int, members: set[int]
) -> dict[int, int]:
    """Hop distances from `source` within the subgraph induced on `members`."""
    distances = {source: 0}
    queue = [source]
    head = 0

    while head < len(queue):
        node = queue[head]
        head += 1

        for neighbour in neighbours[node]:
            if neighbour in members and neighbour not in distances:
                distances[neighbour] = distances[node] + 1
                queue.append(neighbour)

    return distances


def _normalized_adjacency(graph: GraphInfo) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    (sources, targets, values) of the symmetric renormalization D^-1/2 A D^-1/2.

    LPSI is defined on exactly this operator, and it is also what our encoders
    consume, so a generated program that reimplements label propagation is
    working over the same matrix the world model does.
    """
    neighbours = neighbour_sets(graph)
    pairs = [
        (node, other) for node, group in enumerate(neighbours) for other in sorted(group)
    ]

    if not pairs:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0, dtype=np.float64)

    sources = np.fromiter((pair[0] for pair in pairs), dtype=np.int64, count=len(pairs))
    targets = np.fromiter((pair[1] for pair in pairs), dtype=np.int64, count=len(pairs))

    degrees = np.array([len(group) for group in neighbours], dtype=np.float64)
    inverse_sqrt = np.where(degrees > 0, degrees, 1.0) ** -0.5
    values = inverse_sqrt[sources] * inverse_sqrt[targets]

    return sources, targets, values


# Label propagation ----------------------------------------------------------
def lpsi_scores(graph: GraphInfo, observation: np.ndarray, alpha: float = lpsi_alpha, **_kw: object) -> np.ndarray:
    """LPSI's converged label field F = (1 - a) (I - a S)^-1 Y, by iteration."""
    sources, targets, values = _normalized_adjacency(graph)
    # +1 infected, -1 uninfected: the paper's label encoding, and the reason the
    # field has interior maxima at all
    labels = np.where(np.asarray(observation) >= infected_threshold, 1.0, -1.0)
    field = labels.copy()

    for _ in range(lpsi_iterations):
        propagated = np.zeros(graph.num_nodes, dtype=np.float64)

        if sources.size:
            np.add.at(propagated, targets, values * field[sources])

        updated = alpha * propagated + (1.0 - alpha) * labels

        if np.abs(updated - field).max() < lpsi_tolerance:
            field = updated
            break

        field = updated

    return field


def lpsi(
    graph: GraphInfo, observation: np.ndarray, budget: int, alpha: float = lpsi_alpha, **_kw: object
) -> list[int]:
    """LPSI (AAAI 2017): sources are LOCAL MAXIMA of a converged label field."""
    field = lpsi_scores(graph, observation, alpha=alpha)
    neighbours = neighbour_sets(graph)
    infected = set(infected_set(observation))

    # The paper's selection rule: a source is a node whose converged label value
    # is at least as large as every neighbour's. That constraint is what stops the
    # method from returning one blob of adjacent high-field nodes.
    maxima = [
        node
        for node in sorted(infected)
        if all(field[node] >= field[neighbour] for neighbour in neighbours[node])
    ]
    maxima.sort(key=lambda node: (-field[node], node))

    return _pad(maxima[:budget], _restricted(field, observation), budget)


# NETSLEUTH ------------------------------------------------------------------
def _submatrix_eigenvector(
    neighbours: list[set[int]], members: list[int]
) -> np.ndarray:
    """
    Eigenvector of the SMALLEST eigenvalue of the infected subgraph's submatrix
    Laplacian, by inverse-free shifted power iteration on (dI - L).

    NETSLEUTH's key result: that eigenvector's largest entry names the best single
    seed for the observed ripple. The shift makes the smallest eigenvalue of L the
    LARGEST of (dI - L), so plain power iteration finds it.
    """
    index = {node: position for position, node in enumerate(members)}
    size = len(members)

    if size == 0:
        return np.zeros(0, dtype=np.float64)

    # FULL degrees, not infected-subgraph degrees: L_I is the submatrix of the whole
    # graph's Laplacian, so an infected node with many uninfected neighbours is
    # penalized. With subgraph degrees L_I * 1 = 0 and the smallest eigenvector is
    # the constant vector, which ranks every infected node equally.
    degrees = np.array([len(neighbours[node]) for node in members], dtype=np.float64)
    pairs = [
        (index[node], index[other])
        for node in members
        for other in neighbours[node]
        if other in index
    ]
    rows = np.fromiter((pair[0] for pair in pairs), dtype=np.int64, count=len(pairs))
    columns = np.fromiter((pair[1] for pair in pairs), dtype=np.int64, count=len(pairs))

    shift = float(degrees.max()) * 2.0 + 1.0
    vector = np.ones(size, dtype=np.float64) / np.sqrt(size)

    for _ in range(power_iterations):
        # (shift*I - L) v = (shift - D) v + A v
        product = (shift - degrees) * vector

        if rows.size:
            np.add.at(product, rows, vector[columns])

        norm = np.linalg.norm(product)
        if norm < norm_floor:
            break

        product = product / norm
        if np.abs(product - vector).max() < power_tolerance:
            vector = product
            break

        vector = product

    # Sign is arbitrary out of power iteration; the Perron entry of the smallest
    # Laplacian eigenvector is non-negative, so orient it that way
    return np.abs(vector)


def netsleuth_scores(graph: GraphInfo, observation: np.ndarray, **_kw: object) -> np.ndarray:
    """NETSLEUTH: submatrix-Laplacian eigenvector entry per infected node."""
    neighbours = neighbour_sets(graph)
    infected = infected_set(observation)
    scores = np.zeros(graph.num_nodes, dtype=np.float64)

    for component in _components(neighbours, set(infected)):
        vector = _submatrix_eigenvector(neighbours, component)
        for position, node in enumerate(component):
            scores[node] = float(vector[position])

    return scores


def netsleuth(
    graph: GraphInfo, observation: np.ndarray, budget: int, **_kw: object
) -> list[int]:
    """
    NETSLEUTH (ICDM 2012): submatrix-Laplacian eigenvector, seeds picked by deflation.

    Warning: **The MDL half of the paper is not implemented.** NETSLEUTH's headline
    contribution is that it INFERS the number of sources, by encoding the source
    set plus the ripple that grows from it and picking the `k` that minimizes total
    description length. This harness is given `k` (`budget` is the source
    count, and the whole arm set is compared at matched `k`), so only the
    seed-SELECTION half runs. Under the given-k convention that is the right
    comparison and it is what SL-VAE's own table reports NETSLEUTH under, but it
    is strictly less than the published method, and an inferred-k arm would need a
    `budget=None` path through `localize` that does not exist.
    """
    neighbours = neighbour_sets(graph)
    infected = set(infected_set(observation))
    chosen = []

    # The paper picks seeds one at a time and re-solves on the residual ripple.
    # Deleting the picked node from the infected set IS that deflation: the
    # eigenvector of the remaining subgraph no longer sees the ripple it explains.
    remaining = set(infected)
    for _ in range(min(budget, len(infected))):
        best_node, best_score = None, -np.inf

        for component in _components(neighbours, remaining):
            vector = _submatrix_eigenvector(neighbours, component)
            position = int(np.argmax(vector))

            if float(vector[position]) > best_score:
                best_node, best_score = component[position], float(vector[position])

        if best_node is None:
            break

        chosen.append(int(best_node))
        remaining.discard(int(best_node))

    return _pad(chosen, _restricted(netsleuth_scores(graph, observation), observation), budget)


# Jordan centre / sample path ------------------------------------------------
def jordan_center_scores(graph: GraphInfo, observation: np.ndarray, **_kw: object) -> np.ndarray:
    """Negative eccentricity within the infected subgraph (higher = more central)."""
    neighbours = neighbour_sets(graph)
    infected = set(infected_set(observation))
    scores = np.full(graph.num_nodes, -np.inf, dtype=np.float64)

    for component in _components(neighbours, infected):
        members = set(component)
        for node in component:
            distances = _bfs_distances(neighbours, node, members)
            scores[node] = -float(max(distances.values()))

    finite = scores[np.isfinite(scores)]
    floor = float(finite.min()) - 1.0 if finite.size else 0.0

    return np.where(np.isfinite(scores), scores, floor)


def jordan_center(
    graph: GraphInfo, observation: np.ndarray, budget: int, **_kw: object
) -> list[int]:
    """
    Jordan centre (Zhu & Ying, ToN 2016): the eccentricity minimizer of the
    infected subgraph.

    Single-source by construction. The multi-source generalization used here is
    the natural one and is stated rather than hidden: one centre per infected
    COMPONENT first (components cannot share a source under a progressive
    cascade), then the remaining budget by global eccentricity rank.
    """
    neighbours = neighbour_sets(graph)
    infected = set(infected_set(observation))
    scores = jordan_center_scores(graph, observation)
    chosen = []

    components = sorted(
        _components(neighbours, infected), key=len, reverse=True
    )
    for component in components:
        if len(chosen) >= budget:
            break

        chosen.append(max(component, key=lambda node: (scores[node], -node)))

    return _pad(chosen, _restricted(scores, observation), budget)


# Rumor centrality -----------------------------------------------------------
def rumor_centrality_scores(
    graph: GraphInfo, observation: np.ndarray, **_kw: object
) -> np.ndarray:
    """
    Shah & Zaman's rumor centrality, in logs, over a BFS tree per infected
    component.

    R(v) counts the distinct spreading orders consistent with the observed
    infected subtree; on a regular tree it is the ML source estimator. The linear
    -time evaluation is the paper's own: compute R at one root by a post-order
    pass, then push it outward with

        log R(child) = log R(parent) + log(subtree_child) - log(n - subtree_child)

    Logs rather than the factorial ratio because n! overflows past n ~ 170 and the
    infected sets here run to thousands of nodes.
    """
    neighbours = neighbour_sets(graph)
    infected = set(infected_set(observation))
    scores = np.zeros(graph.num_nodes, dtype=np.float64)

    for component in _components(neighbours, infected):
        members = set(component)
        size = len(component)

        if size == 1:
            scores[component[0]] = 0.0
            continue

        # BFS tree from an arbitrary root; rumor centrality is defined on a tree,
        # and the BFS tree is the standard reduction for a general graph
        root = component[0]
        parent = {root: -1}
        order = [root]
        head = 0

        while head < len(order):
            node = order[head]
            head += 1
            for neighbour in sorted(neighbours[node]):
                if neighbour in members and neighbour not in parent:
                    parent[neighbour] = node
                    order.append(neighbour)

        subtree = {node: 1 for node in order}
        for node in reversed(order[1:]):
            subtree[parent[node]] += subtree[node]

        # log R(root) = log(n!) - sum_u log(subtree_u)
        log_root = float(
            np.sum(np.log(np.arange(1, size + 1)))
            - np.sum(np.log([subtree[node] for node in order]))
        )
        log_scores = {root: log_root}

        for node in order[1:]:
            up = parent[node]
            log_scores[node] = (
                log_scores[up]
                + np.log(subtree[node])
                - np.log(max(size - subtree[node], 1))
            )

        for node, value in log_scores.items():
            scores[node] = float(value)

    return scores


def rumor_centrality(
    graph: GraphInfo, observation: np.ndarray, budget: int, **_kw: object
) -> list[int]:
    """
    Rumor centrality (Shah & Zaman, SIGMETRICS 2010): the paper that founded the field.

    SINGLE-SOURCE. Evaluated as multi-source at k = 10% of N it scores near zero by
    construction, which is a property of
    the protocol, not of the estimator: run it at `--budgets 1` for its own number.
    """
    scores = _restricted(rumor_centrality_scores(graph, observation), observation)

    return _top_k(scores, budget)


# Optimal Jordan Cover -------------------------------------------------------
def _cover_number(neighbours: list[set[int]], infected: set[int]) -> int:
    """
    OJC's Y: the largest k for which some node has at least k infected neighbours.

    The paper's candidate rule is "nodes that cover at least Y of the observed
    infected nodes", and Y is read off the graph rather than tuned.
    """
    counts = [len(neighbours[node] & infected) for node in sorted(infected)]

    return max(counts) if counts else 0


def ojc_scores(graph: GraphInfo, observation: np.ndarray, **_kw: object) -> np.ndarray:
    """OJC: negative eccentricity to the infected set, over the full graph."""
    neighbours = neighbour_sets(graph)
    infected = set(infected_set(observation))
    scores = np.full(graph.num_nodes, -np.inf, dtype=np.float64)

    if not infected:
        return np.zeros(graph.num_nodes, dtype=np.float64)

    # Eccentricity measured over the WHOLE graph, not the infected subgraph: OJC
    # is the partial-observation method, so a source need not itself be observed
    universe = set(range(graph.num_nodes))
    candidates = sorted(
        {node for member in infected for node in neighbours[member]} | infected
    )

    for node in candidates:
        distances = _bfs_distances(neighbours, node, universe)
        reach = [distances[member] for member in infected if member in distances]
        scores[node] = -float(max(reach)) if len(reach) == len(infected) else -np.inf

    finite = scores[np.isfinite(scores)]
    floor = float(finite.min()) - 1.0 if finite.size else 0.0

    return np.where(np.isfinite(scores), scores, floor)


def ojc(graph: GraphInfo, observation: np.ndarray, budget: int, **_kw: object) -> list[int]:
    """
    OJC (AAAI 2017): cover the observed infected nodes with balls, then take the
    Jordan centre of the cover.

    Built for PARTIAL observation, which is why its candidate set is the infected
    nodes plus their neighbours rather than the infected nodes alone.

    Warning: **The cover here is GREEDY, tie-broken by Jordan centrality.** The paper
    solves for an optimal cover and proves optimality on tree-like graphs; this is
    the standard greedy set-cover approximation, made sequential so it returns a
    ranked set of exactly `budget` sources. It is also run under FULL observation,
    which is the regime our episodes record: OJC's own advantage is the sparse
    -observer setting we do not generate, so this row understates it by design
    rather than by accident.
    """
    neighbours = neighbour_sets(graph)
    infected = set(infected_set(observation))

    if not infected:
        return _top_k(np.zeros(graph.num_nodes), budget)

    cover = _cover_number(neighbours, infected)
    candidates = [
        node
        for node in range(graph.num_nodes)
        if len(neighbours[node] & infected) >= cover
    ] or sorted(infected)

    scores = ojc_scores(graph, observation)
    chosen = []
    uncovered = set(infected)

    # Greedy set cover over the candidate balls, tie-broken by Jordan centrality,
    # the paper's "optimal cover, then centre of the cover", made sequential so it
    # produces a ranked set of exactly `budget` sources
    for _ in range(min(budget, len(candidates))):
        best_node, best_key = None, None

        for node in candidates:
            if node in chosen:
                continue

            gain = len((neighbours[node] | {node}) & uncovered)
            key = (gain, scores[node], -node)

            if best_key is None or key > best_key:
                best_node, best_key = node, key

        if best_node is None:
            break

        chosen.append(int(best_node))
        uncovered -= neighbours[best_node] | {best_node}

    return _pad(chosen, _restricted(scores, observation), budget)


# Spectral: dynamical age ----------------------------------------------------
def _leading_eigenvalue(neighbours: list[set[int]], members: list[int]) -> float:
    """Largest adjacency eigenvalue of the subgraph induced on `members`."""
    index = {node: position for position, node in enumerate(members)}
    pairs = [
        (index[node], index[other])
        for node in members
        for other in neighbours[node]
        if other in index
    ]

    if not pairs:
        return 0.0

    rows = np.fromiter((pair[0] for pair in pairs), dtype=np.int64, count=len(pairs))
    columns = np.fromiter((pair[1] for pair in pairs), dtype=np.int64, count=len(pairs))
    vector = np.ones(len(members), dtype=np.float64) / np.sqrt(len(members))
    eigenvalue = 0.0

    for _ in range(power_iterations):
        product = np.zeros(len(members), dtype=np.float64)
        np.add.at(product, rows, vector[columns])
        norm = float(np.linalg.norm(product))

        if norm < norm_floor:
            return 0.0

        vector = product / norm
        if abs(norm - eigenvalue) < power_tolerance:
            eigenvalue = norm
            break

        eigenvalue = norm

    return float(eigenvalue)


def dynamic_age_scores(
    graph: GraphInfo, observation: np.ndarray, n_candidates: int = dynamic_age_candidates, **_kw: object
) -> np.ndarray:
    """Fioriti-Chinnici: drop in the infected subgraph's leading eigenvalue when v is removed."""
    neighbours = neighbour_sets(graph)
    infected = infected_set(observation)
    scores = np.zeros(graph.num_nodes, dtype=np.float64)

    if not infected:
        return scores

    baseline = _leading_eigenvalue(neighbours, infected)
    # One eigen-solve per candidate, so the pool is the highest-degree slice of
    # the infected subgraph rather than all of it
    degrees = {node: len(neighbours[node] & set(infected)) for node in infected}
    candidates = sorted(infected, key=lambda node: (-degrees[node], node))[:n_candidates]

    for node in candidates:
        residual = [member for member in infected if member != node]
        scores[node] = baseline - _leading_eigenvalue(neighbours, residual)

    return scores


def dynamic_age(
    graph: GraphInfo, observation: np.ndarray, budget: int, **_kw: object
) -> list[int]:
    """Dynamical age (Fioriti & Chinnici 2012): the spectral multi-source estimator."""
    return _top_k(_restricted(dynamic_age_scores(graph, observation), observation), budget)


# Effective distance ---------------------------------------------------------
def effective_distance_scores(
    graph: GraphInfo, observation: np.ndarray, **_kw: object
) -> np.ndarray:
    """
    Brockmann-Helbing: negative spread of effective distance to the infected set.

    In the metric d(u->v) = 1 - log P(u->v) a complex contagion becomes a circular
    wave, so the origin is the node from which every infected node sits at a
    SIMILAR effective distance. Scoring by the standard deviation of that distance
    (negated, so higher is better) is that circularity, made numeric.
    """
    infected = infected_set(observation)
    scores = np.full(graph.num_nodes, -np.inf, dtype=np.float64)

    if not infected:
        return np.zeros(graph.num_nodes, dtype=np.float64)

    # Outgoing flux fractions, so P(u->v) is a probability over u's out-edges
    weights = np.asarray(graph.ic_probs, dtype=np.float64)
    out_total = np.zeros(graph.num_nodes, dtype=np.float64)
    np.add.at(out_total, graph.edge_index[0], weights)

    lengths = 1.0 - np.log(
        np.clip(weights / np.clip(out_total[graph.edge_index[0]], probability_floor, None),
                probability_floor, 1.0)
    )

    adjacency = {node: [] for node in range(graph.num_nodes)}
    for edge in range(graph.edge_index.shape[1]):
        source = int(graph.edge_index[0, edge])
        target = int(graph.edge_index[1, edge])
        adjacency[source].append((target, float(lengths[edge])))
        adjacency[target].append((source, float(lengths[edge])))

    import heapq

    infected_index = set(infected)
    for start in infected:
        distances = {start: 0.0}
        heap = [(0.0, start)]

        while heap:
            distance, node = heapq.heappop(heap)
            if distance > distances.get(node, np.inf):
                continue

            for neighbour, length in adjacency[node]:
                candidate = distance + length
                if candidate < distances.get(neighbour, np.inf):
                    distances[neighbour] = candidate
                    heapq.heappush(heap, (candidate, neighbour))

        reach = [distances.get(member) for member in infected_index]
        if any(value is None for value in reach):
            continue

        scores[start] = -float(np.std(reach))

    finite = scores[np.isfinite(scores)]
    floor = float(finite.min()) - 1.0 if finite.size else 0.0

    return np.where(np.isfinite(scores), scores, floor)


def effective_distance(
    graph: GraphInfo, observation: np.ndarray, budget: int, **_kw: object
) -> list[int]:
    """Effective distance (Brockmann & Helbing, Science 2013): the wave-centre estimator."""
    return _top_k(
        _restricted(effective_distance_scores(graph, observation), observation), budget
    )


# Dynamic message passing ----------------------------------------------------
def _dmp_forward(
    graph: GraphInfo, seeds: list[int], steps: int
) -> np.ndarray:
    """
    Deterministic mean-field IC marginals from `seeds` after `steps` rounds.

    This is the DMP recursion in its independent-cascade form: a frontier
    probability per node, composed through the true per-edge transmission
    probabilities. Exact on trees, an approximation with loops, which is exactly
    what Lokhov et al. rely on. NOT a call to the metered evaluator: it is
    analytic, consumes no simulator episodes, and is the classical method's own
    forward model rather than ours.
    """
    infected = np.zeros(graph.num_nodes, dtype=np.float64)
    frontier = np.zeros(graph.num_nodes, dtype=np.float64)
    infected[list(seeds)] = 1.0
    frontier[list(seeds)] = 1.0

    sources, targets = graph.edge_index[0], graph.edge_index[1]
    weights = np.asarray(graph.ic_probs, dtype=np.float64)

    for _ in range(steps):
        if frontier.max() <= probability_floor:
            break

        transmitted = np.clip(weights * frontier[sources], 0.0, 1.0 - probability_floor)
        log_survival = np.zeros(graph.num_nodes, dtype=np.float64)
        np.add.at(log_survival, targets, np.log1p(-transmitted))

        p_new = (1.0 - infected) * (1.0 - np.exp(log_survival))
        infected = infected + p_new
        frontier = p_new

    return np.clip(infected, 0.0, 1.0)


def dmp_scores(
    graph: GraphInfo,
    observation: np.ndarray,
    n_candidates: int = dmp_candidates,
    steps: int = dmp_steps,
    **_kw: object,
) -> np.ndarray:
    """DMP likelihood of the observation, per single-source hypothesis."""
    observation = np.clip(np.asarray(observation, dtype=np.float64), 0.0, 1.0)
    scores = np.full(graph.num_nodes, -np.inf, dtype=np.float64)
    field = lpsi_scores(graph, observation)
    candidates = _top_k(_restricted(field, observation), n_candidates)

    for node in candidates:
        predicted = np.clip(_dmp_forward(graph, [node], steps), probability_floor, 1.0 - probability_floor)
        scores[node] = float(
            np.sum(
                observation * np.log(predicted) + (1.0 - observation) * np.log1p(-predicted)
            )
        )

    finite = scores[np.isfinite(scores)]
    floor = float(finite.min()) - 1.0 if finite.size else 0.0

    return np.where(np.isfinite(scores), scores, floor)


def dmp_localize(
    graph: GraphInfo,
    observation: np.ndarray,
    budget: int,
    n_candidates: int = dmp_candidates,
    steps: int = dmp_steps,
    **_kw: object,
) -> list[int]:
    """
    Dynamic message passing (Lokhov et al. 2014): greedy MAP over the DMP likelihood.

    Sequential rather than top-k of `dmp_scores`, because a second source is only
    worth adding where the first one's predicted cascade FAILS to explain the
    observation: scoring hypotheses independently would return `budget` copies of
    the same region.
    """
    observation = np.clip(np.asarray(observation, dtype=np.float64), 0.0, 1.0)
    field = lpsi_scores(graph, observation)
    candidates = _top_k(_restricted(field, observation), n_candidates)
    chosen = []

    for _ in range(min(budget, len(candidates))):
        best_node, best_score = None, -np.inf

        for node in candidates:
            if node in chosen:
                continue

            predicted = np.clip(
                _dmp_forward(graph, chosen + [node], steps),
                probability_floor,
                1.0 - probability_floor,
            )
            likelihood = float(
                np.sum(
                    observation * np.log(predicted)
                    + (1.0 - observation) * np.log1p(-predicted)
                )
            )

            if likelihood > best_score:
                best_node, best_score = node, likelihood

        if best_node is None:
            break

        chosen.append(int(best_node))

    return _pad(chosen, _restricted(dmp_scores(graph, observation, n_candidates, steps), observation), budget)


# Comin-Costa centrality suite ----------------------------------------------
def _infected_centrality(
    graph: GraphInfo, observation: np.ndarray, kind: str
) -> np.ndarray:
    """One of degree / betweenness / closeness / eigenvector, on the infected subgraph."""
    neighbours = neighbour_sets(graph)
    infected = set(infected_set(observation))
    scores = np.zeros(graph.num_nodes, dtype=np.float64)

    if not infected:
        return scores

    if kind == "degree":
        for node in infected:
            scores[node] = float(len(neighbours[node] & infected))

        return scores

    for component in _components(neighbours, infected):
        members = set(component)

        if kind == "closeness":
            for node in component:
                distances = _bfs_distances(neighbours, node, members)
                total = sum(distances.values())
                scores[node] = (len(distances) - 1) / total if total else 0.0
        elif kind == "betweenness":
            # Brandes on the induced subgraph, unweighted
            contribution = {node: 0.0 for node in component}

            for start in component:
                stack, predecessors = [], {node: [] for node in component}
                path_counts = {node: 0.0 for node in component}
                distance = {node: -1 for node in component}
                path_counts[start], distance[start] = 1.0, 0
                queue, head = [start], 0

                while head < len(queue):
                    node = queue[head]
                    head += 1
                    stack.append(node)

                    for neighbour in sorted(neighbours[node] & members):
                        if distance[neighbour] < 0:
                            distance[neighbour] = distance[node] + 1
                            queue.append(neighbour)

                        if distance[neighbour] == distance[node] + 1:
                            path_counts[neighbour] += path_counts[node]
                            predecessors[neighbour].append(node)

                dependency = {node: 0.0 for node in component}
                while stack:
                    node = stack.pop()
                    for predecessor in predecessors[node]:
                        dependency[predecessor] += (
                            path_counts[predecessor] / path_counts[node]
                        ) * (1.0 + dependency[node])

                    if node != start:
                        contribution[node] += dependency[node]

            for node, value in contribution.items():
                scores[node] = value / 2.0
        else:
            vector = np.ones(len(component), dtype=np.float64) / np.sqrt(len(component))
            index = {node: position for position, node in enumerate(component)}
            pairs = [
                (index[node], index[other])
                for node in component
                for other in neighbours[node]
                if other in index
            ]
            rows = np.fromiter((pair[0] for pair in pairs), dtype=np.int64, count=len(pairs))
            columns = np.fromiter((pair[1] for pair in pairs), dtype=np.int64, count=len(pairs))

            for _ in range(power_iterations):
                product = np.zeros(len(component), dtype=np.float64)
                if rows.size:
                    np.add.at(product, rows, vector[columns])

                norm = float(np.linalg.norm(product))
                if norm < norm_floor:
                    break

                product = product / norm
                if np.abs(product - vector).max() < power_tolerance:
                    vector = product
                    break

                vector = product

            for position, node in enumerate(component):
                scores[node] = float(abs(vector[position]))

    return scores


def infected_degree_scores(graph: GraphInfo, observation: np.ndarray, **_kw: object) -> np.ndarray:
    return _infected_centrality(graph, observation, "degree")


def infected_degree(graph: GraphInfo, observation: np.ndarray, budget: int, **_kw: object) -> list[int]:
    """Comin-Costa: degree within the infected subgraph, the cheap-heuristic floor."""
    return _top_k(_restricted(infected_degree_scores(graph, observation), observation), budget)


def infected_betweenness_scores(graph: GraphInfo, observation: np.ndarray, **_kw: object) -> np.ndarray:
    return _infected_centrality(graph, observation, "betweenness")


def infected_betweenness(graph: GraphInfo, observation: np.ndarray, budget: int, **_kw: object) -> list[int]:
    """Comin-Costa: betweenness within the infected subgraph."""
    return _top_k(
        _restricted(infected_betweenness_scores(graph, observation), observation), budget
    )


def infected_closeness_scores(graph: GraphInfo, observation: np.ndarray, **_kw: object) -> np.ndarray:
    return _infected_centrality(graph, observation, "closeness")


def infected_closeness(graph: GraphInfo, observation: np.ndarray, budget: int, **_kw: object) -> list[int]:
    """Comin-Costa: closeness within the infected subgraph."""
    return _top_k(
        _restricted(infected_closeness_scores(graph, observation), observation), budget
    )


def infected_eigenvector_scores(graph: GraphInfo, observation: np.ndarray, **_kw: object) -> np.ndarray:
    return _infected_centrality(graph, observation, "eigenvector")


def infected_eigenvector(graph: GraphInfo, observation: np.ndarray, budget: int, **_kw: object) -> list[int]:
    """Comin-Costa: eigenvector centrality within the infected subgraph."""
    return _top_k(
        _restricted(infected_eigenvector_scores(graph, observation), observation), budget
    )


# Floors and the simulation-based member ------------------------------------
def random_sources_scores(
    graph: GraphInfo, observation: np.ndarray, seed: int = 0, **_kw: object
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    return rng.random(graph.num_nodes)


def random_sources(
    graph: GraphInfo, observation: np.ndarray, budget: int, seed: int = 0, **_kw: object
) -> list[int]:
    """Uniformly random nodes from the infected set: the floor every number is read against."""
    return _top_k(
        _restricted(random_sources_scores(graph, observation, seed), observation), budget
    )


def resim_greedy_scores(graph: GraphInfo, observation: np.ndarray, **_kw: object) -> np.ndarray:
    """Ranking proxy for `resim_greedy`; the selection itself re-simulates."""
    return lpsi_scores(graph, observation)


def resim_greedy(
    graph: GraphInfo,
    observation: np.ndarray,
    budget: int,
    diffusion_model: str = "IC",
    predict: object = None,
    n_candidates: int = resim_candidates,
    mc_runs: int = resim_mc_runs,
    horizon: int = resim_horizon,
    seed: int = 0,
    **_kw: object,
) -> list[int]:
    """
    Greedy minimization of the re-simulation error ||y - f(x_hat)||^2.

    The forward-model-using classical baseline. Its
    `predict` argument is the one axis that matters: pass a forward oracle (a
    canned baseline receives one; a generated program is offline) and the calls go
    through the arm's own METERED evaluator; leave it None and it falls back to a private NDlib estimator, which
    is the honest classical cost and is invisible to `real_env_episodes`. That
    invisibility is why it is blocked from generated scripts by default, exactly
    as `celf` and `greedy_blocking` are.
    """
    observation = np.clip(np.asarray(observation, dtype=np.float64), 0.0, 1.0)
    field = lpsi_scores(graph, observation)
    candidates = _top_k(_restricted(field, observation), n_candidates)

    def forward(seeds: list[int]) -> np.ndarray:
        if predict is not None:
            return np.asarray(predict(seeds), dtype=np.float64)

        return primitives.predict_marginals_mc(
            graph, seeds, diffusion_model, mc_runs=mc_runs, horizon=horizon, seed=seed
        )

    chosen = []

    for _ in range(min(budget, len(candidates))):
        best_node, best_error = None, np.inf

        for node in candidates:
            if node in chosen:
                continue

            error = float(np.sum((forward(chosen + [node]) - observation) ** 2))
            if error < best_error:
                best_node, best_error = node, error

        if best_node is None:
            break

        chosen.append(int(best_node))

    return _pad(chosen, _restricted(field, observation), budget)


localization_algorithms = {
    # label propagation: the bar
    "lpsi": lpsi,
    # MDL / spectral, multi-source
    "netsleuth": netsleuth,
    "dynamic_age": dynamic_age,
    # distance / cover, multi-source
    "ojc": ojc,
    "jordan_center": jordan_center,
    "effective_distance": effective_distance,
    # message passing
    "dmp_localize": dmp_localize,
    # single-source, ranking estimators
    "rumor_centrality": rumor_centrality,
    # Comin-Costa centrality suite on the infected subgraph
    "infected_degree": infected_degree,
    "infected_betweenness": infected_betweenness,
    "infected_closeness": infected_closeness,
    "infected_eigenvector": infected_eigenvector,
    # floor
    "random_sources": random_sources,
    # simulation-based
    "resim_greedy": resim_greedy,
}

# Per-node score vectors for the same names. F1 scores the SET, AUC scores the
# RANKING, so both are needed and they are
# deliberately not top-k of one another for the sequential members.
localization_scorers = {
    "lpsi": lpsi_scores,
    "netsleuth": netsleuth_scores,
    "dynamic_age": dynamic_age_scores,
    "ojc": ojc_scores,
    "jordan_center": jordan_center_scores,
    "effective_distance": effective_distance_scores,
    "dmp_localize": dmp_scores,
    "rumor_centrality": rumor_centrality_scores,
    "infected_degree": infected_degree_scores,
    "infected_betweenness": infected_betweenness_scores,
    "infected_closeness": infected_closeness_scores,
    "infected_eigenvector": infected_eigenvector_scores,
    "random_sources": random_sources_scores,
    "resim_greedy": resim_greedy_scores,
}

# Simulation-based, so one call costs budget x n_candidates x mc_runs real
# episodes on its private simulator. Charged honestly to the arm like any other
# baseline, and blocked from generated scripts by default: a generated program
# is offline by construction.
mc_localization_algorithms = ("resim_greedy",)

localization_algorithm_names = list(localization_algorithms)

_missing = set(localization_algorithms) - set(localization_scorers)
if _missing:
    raise ValueError(
        f"every localization algorithm needs a paired scorer for the AUC column; "
        f"missing: {sorted(_missing)}"
    )
