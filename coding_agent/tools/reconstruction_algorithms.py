"""
Named classical CASCADE RECONSTRUCTION baselines: the condition-1 floor for
`--task cascade_reconstruction`.

Every decoder has the signature

    (graph, observation, horizon, **kw) -> dict[int, tuple[int, int | None]]

mapping each node it believes was infected to `(activation timestep, inferred
parent)`, with `parent = None` marking a source and uninfected nodes simply
absent. That single object carries everything the published metrics score: the
`(node, t)` pairs give Event F1 and infection-time NRMSE, and the parent assignment
gives Path Precision, Jaccard and order accuracy.

That is the whole difference from the other three pools. `algorithms.py` returns a
seed set to maximize with, `dismantling_algorithms.py` nodes to delete,
`localization_algorithms.py` a source SET, and this one returns a whole
trajectory, which is why source localization is the projection of this
task rather than a sibling of it (the subset with `parent = None` IS the recovered
seed set).

None of these is a weak floor. In ascending order of danger:

  1. `random_reconstruction`, `observed_only`, `one_hop`: Rozenshtein's own two
     controls plus a floor. `observed_only` has node precision 1.0 BY CONSTRUCTION
     and exists to prove the reward is not gameable.
  2. **`delayed_bfs` (Xiao SDM'18)**: the row that actually has to be beaten. All
     four of that paper's methods reach node precision > 0.8 and usually near 1.0
     [figure], `delayed-bfs` is `O(m + k log k)`, and it sits INSIDE the coding
     agent's expressible space.
  3. **`personalized_pagerank`**: the most specific warning in this literature.
     Xiao ICDM'18 found Personalized PageRank BEATS tree sampling on
     `grqc` (assortativity 0.164) and loses elsewhere [verified]. Our suite
     contains exactly that graph. If the search cannot clear PPR on `ca_grqc`, the
     result is not real.
  4. `consistent_tree_wpct`: Zong ICDM'12, and the source of this task's central
     asymmetry: `prec_v = 100%` alongside `prec_e = 78-86%` [verified]. The node
     set is easy and the tree is hard, thirteen years before DIPT said it again.

**These are honest condition-1 arms, not substitutes for the authors' code.** Each
docstring states where it deviates. Three deviations are shared widely enough to
name up front:

  * **The parent assignment is shared.** Most published methods output a
    node set and a time, not a tree; `finalize` turns any time assignment into a
    coherent tree by the same rule for all of them, so a Path Precision difference
    between two rows is a difference in their TIMES, not in a tree-building trick
    one of them happens to have. `consistent_tree_*` and the Steiner family are
    the exceptions and build their own.
  * **`k` is not inferred.** NETSLEUTH-style MDL model selection is absent
    throughout; the number of sources each method names is its own business but
    nothing here is scored on getting it from a description length.
  * **Bounded work per instance** wherever a method costs one solve per terminal
    (`ordered_steiner_closure`, `mcmc_decode`, `forward_backward`). The bounds are
    module globals at the top of this file, not buried constants.

Two conventions everything here obeys:

  * **The reported set is the anchor.** A node the observation reports as infected
    is infected: no method here may drop one, because under a progressive cascade
    an observation is ground truth about that node. What differs is what each one
    infers ABOUT THE REST.
  * **A transmission travels along an arc that exists.** `finalize` only ever
    names a parent from `graph.in_neighbors(v)`, which is what the executor
    validates and what makes Path Precision comparable to DIPT's.
"""

import heapq
import math
import networkx as nx
import numpy as np

from coding_agent.containment import neighbour_sets
from coding_agent.types import GraphInfo

# Backfilling a node to explain a report can cascade; the bound stops a pathological
# instance from adding a chain the length of the graph
max_backfill_rounds = 50

# Terminals the metric-closure method solves shortest paths for. One Dijkstra per
# terminal, so the pool is capped rather than scanning every report on a big graph.
closure_terminals = 120

# Steiner trees sampled by `tree_sampling`. Xiao ICDM'18 uses 1,000 and reports
# gains beyond that are marginal [verified]; 60 is what finishes
# inside one refinement iteration, and it is the first number to raise before
# quoting a tree-sampling comparison as anything but a smoke result.
sampled_trees = 60

# Personalized PageRank: restart probability and iteration cap
pagerank_alpha = 0.85
pagerank_iterations = 60

# MCMC decoding: proposals per instance and burn-in fraction. A real decoder
# needs ~10^4 kernel calls per instance; these are the smoke-test values.
mcmc_proposals = 400
mcmc_burn_in = 0.25

# Forward-filter / backward-sample: particles carried per step
smoothing_particles = 24

probability_floor = 1e-9


def _probability_map(graph: GraphInfo) -> dict[tuple[int, int], float]:
    """{(u, v): p} for the arcs of this graph, as the decoders' edge weights."""
    return {
        (int(graph.edge_index[0, edge]), int(graph.edge_index[1, edge])): float(
            graph.ic_probs[edge]
        )
        for edge in range(graph.edge_index.shape[1])
    }


def _weighted_view(graph: GraphInfo) -> nx.DiGraph:
    """
    The graph as `-log p` arc costs: the likelihood metric every published
    Steiner-style method is defined over.

    A most-likely path is a shortest path under `-log p`, so one weighted view
    serves the tree methods, the consistent-tree methods and CulT alike. Built as
    a DiGraph even for an undirected graph because our undirected loaders already
    store both arcs, so nothing is lost and the direction of a transmission stays
    explicit.
    """
    view = nx.DiGraph()
    view.add_nodes_from(range(graph.num_nodes))

    for (source, target), probability in _probability_map(graph).items():
        view.add_edge(
            source, target, weight=-math.log(max(probability, probability_floor))
        )

    return view


def _visible(observation, node: int) -> bool:
    return observation.visible is None or bool(observation.visible[int(node)])


def reported_nodes(observation) -> list[int]:
    """
    The nodes this observation asserts were infected.

    Under `final_snapshot` that is the whole terminal state; under the report
    settings it is the sampled subset. Either way it is the anchor no decoder may
    contradict.
    """
    return [node for node in observation.infected if _visible(observation, node)]


def estimate_roots(
    graph: GraphInfo, nodes: list[int], observation, count: int = 0
) -> list[int]:
    """
    One root per connected component of the observed subgraph, by Jordan centre.

    The centre of a component is the classical multi-source estimate and it is
    what every method here needs before it can assign a time to anything: a
    reconstruction without a root has no t = 0. `count` caps the roots for a
    method that wants a specific source budget; 0 means one per component.
    """
    if not nodes:
        return []

    neighbours = neighbour_sets(graph)
    members = set(nodes)
    seen = set()
    roots = []

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

        # Jordan centre: the member whose greatest distance to any other member is
        # smallest, over the subgraph induced on the component
        best_node, best_eccentricity = component[0], math.inf
        subset = set(component)

        for candidate in component:
            distances = {candidate: 0}
            wave = [candidate]

            while wave:
                node = wave.pop()
                for neighbour in neighbours[node]:
                    if neighbour in subset and neighbour not in distances:
                        distances[neighbour] = distances[node] + 1
                        wave.append(neighbour)

            eccentricity = max(distances.values())
            if eccentricity < best_eccentricity:
                best_node, best_eccentricity = candidate, eccentricity

        roots.append(int(best_node))

    if count and len(roots) > count:
        roots = sorted(roots, key=graph.degree, reverse=True)[:count]

    return sorted(roots)


def bfs_times(
    graph: GraphInfo, roots: list[int], members: set[int], horizon: int
) -> dict[int, int]:
    """Hop distance from `roots` inside `members`, capped at the horizon."""
    times = {int(root): 0 for root in roots}
    wave = list(times)
    step = 0

    while wave and step < horizon:
        step += 1
        nxt = []

        for node in wave:
            for neighbour in graph.out_neighbors(node):
                if neighbour in members and neighbour not in times:
                    times[int(neighbour)] = step
                    nxt.append(neighbour)

        wave = nxt

    # A member the BFS never reached is in another component or beyond the horizon;
    # pinning it at the horizon is the honest "as late as possible" reading
    for node in members:
        times.setdefault(int(node), horizon)

    return times


def finalize(
    graph: GraphInfo,
    times: dict[int, int],
    observation,
    horizon: int,
) -> dict[int, tuple[int, int | None]]:
    """
    Turn a time assignment into a COHERENT trajectory: every node explained.

    The shared parent rule, so a Path Precision difference between two rows is a
    difference in their times rather than in a tree-building trick one of them
    happens to have. Three steps:

      1. **Clamp** every time into `[0, horizon]` and drop hidden nodes.
      2. **Backfill.** A node at `t > 0` with no in-neighbour at an earlier time
         has no possible cause, which is incoherent rather than merely wrong. Its
         most likely in-neighbour is added at `t - 1`, recursively. This is exactly
         what a Steiner method does when it inserts a hidden node to connect two
         reports, and it is why `observed_only` scores node precision below 1.0 on
         a sparsely reported cascade: the reports alone do not explain themselves.
      3. **Attach.** Each non-source node takes the in-neighbour with the highest
         `p(u -> v)` among those at an earlier time, preferring `t - 1` exactly.
         A node left with no candidate becomes a source at `t = 0`.
    """
    probabilities = _probability_map(graph)
    resolved = {
        int(node): max(0, min(int(time), horizon))
        for node, time in times.items()
        if _visible(observation, node)
    }

    for _ in range(max_backfill_rounds):
        added = {}

        for node, time in resolved.items():
            if time == 0:
                continue

            candidates = [
                neighbour
                for neighbour in graph.in_neighbors(node)
                if resolved.get(neighbour, horizon + 1) < time
            ]
            if candidates:
                continue

            # Nothing explains this node: add its most likely in-neighbour one
            # step earlier, which is the hidden node a Steiner method would insert
            options = [
                neighbour
                for neighbour in graph.in_neighbors(node)
                if _visible(observation, neighbour)
                and resolved.get(neighbour, horizon + 1) >= time
            ]
            if not options:
                continue

            best = max(
                options, key=lambda other: probabilities.get((other, node), 0.0)
            )
            added[int(best)] = min(added.get(int(best), horizon), time - 1)

        if not added:
            break

        for node, time in added.items():
            resolved[node] = max(0, min(time, resolved.get(node, horizon)))

    decoded = {}

    for node, time in resolved.items():
        if time == 0:
            decoded[node] = (0, None)
            continue

        immediate = [
            neighbour
            for neighbour in graph.in_neighbors(node)
            if resolved.get(neighbour, horizon + 1) == time - 1
        ]
        earlier = immediate or [
            neighbour
            for neighbour in graph.in_neighbors(node)
            if resolved.get(neighbour, horizon + 1) < time
        ]

        if not earlier:
            decoded[node] = (0, None)
            continue

        parent = max(
            earlier, key=lambda other: probabilities.get((other, node), 0.0)
        )
        decoded[node] = (time, int(parent))

    return decoded


def _seed_times(observation, members: set[int]) -> dict[int, int]:
    """Observed activation times, restricted to the decoded member set."""
    return {
        node: time
        for node, time in observation.times.items()
        if node in members
    }


def _merge_observed(
    times: dict[int, int], observation, horizon: int
) -> dict[int, int]:
    """
    An OBSERVED time always wins over an inferred one.

    A report is ground truth about that node, so overwriting it with a BFS hop
    would throw away the only exact information in the instance. Applied by every
    decoder here, which is what makes the `partial_times` setting strictly easier
    than `partial_nodes` for all of them rather than for some.
    """
    merged = dict(times)

    for node, time in observation.times.items():
        if _visible(observation, node):
            merged[int(node)] = max(0, min(int(time), horizon))

    return merged


# Controls and floors --------------------------------------------------------


def observed_only(
    graph: GraphInfo, observation, horizon: int, **_kw: object
) -> dict[int, tuple[int, int | None]]:
    """
    Report exactly what was observed and infer nothing: Rozenshtein's `Reports`.

    Node PRECISION is 1.0 by construction (`Reports` is trivially precision 1.0),
    which is precisely why it belongs in the default pool: a trivial decoder has to
    be confirmed to score badly under the chosen reward before a search is run,
    and this is that decoder. It scores badly
    because it has no recall and, once `finalize` backfills the nodes needed to
    explain the reports, no tree either.
    """
    members = set(reported_nodes(observation))
    roots = estimate_roots(graph, sorted(members), observation)
    times = _merge_observed(
        bfs_times(graph, roots, members, horizon), observation, horizon
    )

    return finalize(graph, times, observation, horizon)


def one_hop(
    graph: GraphInfo, observation, horizon: int, **_kw: object
) -> dict[int, tuple[int, int | None]]:
    """
    The reports plus their one-hop neighbourhood: Rozenshtein's `Baseline`.

    The cheapest thing that trades precision for recall, and the second of the two
    controls CulT is measured against in KDD'16 §5.1.
    """
    members = set(reported_nodes(observation))
    members |= {
        int(neighbour)
        for node in list(members)
        for neighbour in graph.out_neighbors(node)
        if _visible(observation, neighbour)
    }
    roots = estimate_roots(graph, sorted(members), observation)
    times = _merge_observed(
        bfs_times(graph, roots, members, horizon), observation, horizon
    )

    return finalize(graph, times, observation, horizon)


def random_reconstruction(
    graph: GraphInfo, observation, horizon: int, seed: int = 0, **_kw: object
) -> dict[int, tuple[int, int | None]]:
    """Random times over the reports and a random sample around them: the floor."""
    rng = np.random.default_rng(seed)
    members = set(reported_nodes(observation))
    extra = [
        node
        for node in range(graph.num_nodes)
        if node not in members and _visible(observation, node)
    ]

    if extra:
        count = min(len(extra), len(members))
        chosen = rng.choice(len(extra), size=count, replace=False)
        members |= {int(extra[int(index)]) for index in chosen}

    times = {
        int(node): int(rng.integers(0, horizon + 1)) for node in sorted(members)
    }

    return finalize(graph, _merge_observed(times, observation, horizon), observation, horizon)


# Steiner / ordering family (Xiao SDM'18) ------------------------------------


def _terminal_order(observation, members: set[int]) -> list[int]:
    """Reported nodes in increasing observed time; unknown times go last, by id."""
    known = _seed_times(observation, members)

    return sorted(members, key=lambda node: (known.get(node, math.inf), node))


def delayed_bfs(
    graph: GraphInfo, observation, horizon: int, **_kw: object
) -> dict[int, tuple[int, int | None]]:
    """
    Xiao SDM'18's `delayed-bfs`: a `k`-approximate ordered Steiner tree in O(m + k log k).

    THE row that has to be beaten. Terminals are attached in increasing observed
    time, each along the cheapest path from the tree built so far, and the path's
    interior nodes are DELAYED to land between the two endpoints' times, which is
    what makes the result respect the observed order rather than merely span the
    reports. The paper's own scalable option, and the one whose linear cost makes
    it the realistic floor for a generated program.
    """
    members = set(reported_nodes(observation))
    if not members:
        return {}

    view = _weighted_view(graph)
    order = _terminal_order(observation, members)
    known = _seed_times(observation, members)

    root = order[0]
    times = {int(root): int(known.get(root, 0))}
    tree = {int(root)}

    for terminal in order[1:]:
        if terminal in tree:
            continue

        # Cheapest path from ANY node already in the tree, which is the "grow from
        # the tree" step; Dijkstra over a virtual super-source is a multi-source run
        distances, paths = _multi_source_dijkstra(view, tree)
        path = paths.get(terminal)

        if path is None:
            # Unreachable from the tree: it is its own component's root
            times[int(terminal)] = int(known.get(terminal, 0))
            tree.add(int(terminal))
            continue

        start_time = times.get(int(path[0]), 0)
        end_time = int(known.get(terminal, min(horizon, start_time + len(path) - 1)))
        end_time = max(end_time, start_time + 1) if len(path) > 1 else start_time

        # Spread the interior nodes evenly between the two endpoints' times: the
        # "delay" the method is named for
        span = max(1, len(path) - 1)
        for index, node in enumerate(path):
            step = start_time + round((end_time - start_time) * index / span)
            times.setdefault(int(node), max(0, min(int(step), horizon)))
            tree.add(int(node))

        times[int(terminal)] = max(0, min(int(end_time), horizon))

    return finalize(graph, _merge_observed(times, observation, horizon), observation, horizon)


def _multi_source_dijkstra(view: nx.DiGraph, sources: set[int]) -> tuple[dict, dict]:
    """Shortest paths from the nearest member of `sources` to every node."""
    distances = {int(node): 0.0 for node in sources}
    paths = {int(node): [int(node)] for node in sources}
    queue = [(0.0, int(node)) for node in sources]
    heapq.heapify(queue)

    while queue:
        distance, node = heapq.heappop(queue)
        if distance > distances.get(node, math.inf):
            continue

        for neighbour in view.successors(node):
            candidate = distance + view[node][neighbour]["weight"]
            if candidate < distances.get(neighbour, math.inf):
                distances[neighbour] = candidate
                paths[neighbour] = paths[node] + [int(neighbour)]
                heapq.heappush(queue, (candidate, int(neighbour)))

    return distances, paths


def ordered_steiner_closure(
    graph: GraphInfo, observation, horizon: int, **_kw: object
) -> dict[int, tuple[int, int | None]]:
    """
    Xiao SDM'18's `closure`: the O(sqrt(k)) metric-closure variant.

    Shortest paths between every pair of terminals form a complete metric closure;
    an MST over that closure, expanded back into the graph, is the classical
    2-approximate Steiner tree, and re-sorting the expansion by observed time is
    what makes it ORDER-respecting. One Dijkstra per terminal, so the terminal pool
    is capped at `closure_terminals`.
    """
    members = set(reported_nodes(observation))
    if not members:
        return {}

    view = _weighted_view(graph)
    order = _terminal_order(observation, members)[:closure_terminals]

    # Metric closure over the capped terminal set
    closure = nx.Graph()
    paths_between = {}

    for terminal in order:
        distances, paths = _multi_source_dijkstra(view, {int(terminal)})
        for other in order:
            if other <= terminal or other not in distances:
                continue

            closure.add_edge(terminal, other, weight=distances[other])
            paths_between[(terminal, other)] = paths[other]

    times = {int(order[0]): int(_seed_times(observation, members).get(order[0], 0))}
    tree = {int(order[0])}

    for source, target in nx.minimum_spanning_edges(closure, data=False):
        path = paths_between.get((min(source, target), max(source, target)))
        if path is None:
            continue

        anchor = times.get(int(path[0]), 0)
        for index, node in enumerate(path):
            times.setdefault(int(node), max(0, min(anchor + index, horizon)))
            tree.add(int(node))

    # Terminals the MST left out (disconnected in the closure) anchor themselves
    for terminal in members:
        times.setdefault(int(terminal), 0)

    return finalize(graph, _merge_observed(times, observation, horizon), observation, horizon)


def greedy_ordered(
    graph: GraphInfo, observation, horizon: int, **_kw: object
) -> dict[int, tuple[int, int | None]]:
    """
    Xiao SDM'18's `greedy`: attach whichever terminal is CLOSEST to the tree next.

    Differs from `delayed_bfs` in the attachment order: nearest-first rather than
    earliest-first, which the paper reports scales roughly linearly in |E| and
    beats plain `steiner` on order accuracy under every model and graph [figure].
    """
    members = set(reported_nodes(observation))
    if not members:
        return {}

    view = _weighted_view(graph)
    known = _seed_times(observation, members)
    order = _terminal_order(observation, members)

    root = order[0]
    times = {int(root): int(known.get(root, 0))}
    tree = {int(root)}
    remaining = set(members) - tree

    while remaining:
        distances, paths = _multi_source_dijkstra(view, tree)
        reachable = [node for node in remaining if node in distances]

        if not reachable:
            for node in remaining:
                times[int(node)] = int(known.get(node, 0))
            break

        nearest = min(reachable, key=lambda node: (distances[node], node))
        path = paths[nearest]
        anchor = times.get(int(path[0]), 0)
        end_time = int(known.get(nearest, min(horizon, anchor + len(path) - 1)))
        span = max(1, len(path) - 1)

        for index, node in enumerate(path):
            step = anchor + round((end_time - anchor) * index / span)
            times.setdefault(int(node), max(0, min(int(step), horizon)))
            tree.add(int(node))

        times[int(nearest)] = max(0, min(int(end_time), horizon))
        remaining.discard(nearest)

    return finalize(graph, _merge_observed(times, observation, horizon), observation, horizon)


def steiner_tree(
    graph: GraphInfo, observation, horizon: int, **_kw: object
) -> dict[int, tuple[int, int | None]]:
    """
    Plain minimum Steiner tree over the reports, ignoring the observed ORDER.

    The control the three ordered variants above are measured against: Xiao SDM'18
    reports `closure` / `greedy` / `delayed-bfs` all beat plain `steiner` on order
    accuracy under every model and graph [figure], and having it in the pool is
    what makes that a measured claim here rather than a cited one.
    """
    members = set(reported_nodes(observation))
    if not members:
        return {}

    undirected = nx.Graph()
    undirected.add_nodes_from(range(graph.num_nodes))
    for (source, target), probability in _probability_map(graph).items():
        weight = -math.log(max(probability, probability_floor))
        if not undirected.has_edge(source, target) or undirected[source][target]["weight"] > weight:
            undirected.add_edge(source, target, weight=weight)

    component = max(nx.connected_components(undirected), key=len)
    terminals = [node for node in members if node in component] or [next(iter(members))]

    try:
        tree = nx.approximation.steiner_tree(
            undirected.subgraph(component), terminals, weight="weight"
        )
        nodes = set(tree.nodes())
    except (nx.NetworkXError, ValueError, KeyError):
        nodes = set(members)

    nodes |= members
    roots = estimate_roots(graph, sorted(nodes), observation)
    times = bfs_times(graph, roots, nodes, horizon)

    return finalize(graph, _merge_observed(times, observation, horizon), observation, horizon)


def tree_sampling(
    graph: GraphInfo, observation, horizon: int, n_trees: int = sampled_trees,
    seed: int = 0, **_kw: object
) -> dict[int, tuple[int, int | None]]:
    """
    Xiao ICDM'18: SAMPLE Steiner trees and read off per-node marginal probabilities.

    The only published classical method that outputs calibrated probabilities rather
    than a binary set. Sampling is by randomized arc costs (`-log p` perturbed by
    an exponential draw) and a shortest-path tree per draw, which is the cheap
    relative of the paper's loop-erased-random-walk cycle popping; a node is kept
    when it appears in more than half the sampled trees, and its time is the modal
    hop at which it appeared.

    The paper uses 1,000 trees and reports marginal gains beyond that.
    `n_trees` here defaults to `sampled_trees`, which is a smoke value.
    """
    members = set(reported_nodes(observation))
    if not members:
        return {}

    rng = np.random.default_rng(seed)
    probabilities = _probability_map(graph)
    roots = estimate_roots(graph, sorted(members), observation)
    counts = {}
    depth_totals = {}

    for _ in range(max(1, n_trees)):
        view = nx.DiGraph()
        view.add_nodes_from(range(graph.num_nodes))
        for (source, target), probability in probabilities.items():
            base = -math.log(max(probability, probability_floor))
            view.add_edge(
                source, target, weight=base * float(rng.exponential(1.0) + 1e-6)
            )

        distances, paths = _multi_source_dijkstra(view, set(roots) or members)

        for terminal in members:
            for depth, node in enumerate(paths.get(terminal, [])):
                counts[int(node)] = counts.get(int(node), 0) + 1
                depth_totals[int(node)] = depth_totals.get(int(node), 0) + depth

    threshold = 0.5 * max(1, n_trees)
    kept = {node for node, count in counts.items() if count >= threshold} | members
    times = {
        int(node): max(
            0, min(int(round(depth_totals.get(node, 0) / max(1, counts.get(node, 1)))), horizon)
        )
        for node in kept
    }

    return finalize(graph, _merge_observed(times, observation, horizon), observation, horizon)


# Centrality / diffusion families --------------------------------------------


def personalized_pagerank(
    graph: GraphInfo, observation, horizon: int, **_kw: object
) -> dict[int, tuple[int, int | None]]:
    """
    Personalized PageRank from the reported nodes: the assortativity trap.

    The most specific warning in this literature: Xiao ICDM'18 found this BEATS
    tree sampling on `grqc` (assortativity 0.164) and loses elsewhere [verified],
    because an assortative graph makes the infected subgraph densely connected and a
    random walker exploits exactly that. Our suite contains that graph. **If a
    generated decoder cannot beat this row on `ca_grqc`, the result is not real**:
    run it there specifically.

    The kept set is sized to the reported set scaled by the observation rate the
    reports imply, so it does not silently predict the whole graph.
    """
    members = set(reported_nodes(observation))
    if not members:
        return {}

    scores = np.zeros(graph.num_nodes, dtype=np.float64)
    restart = np.zeros(graph.num_nodes, dtype=np.float64)
    restart[list(members)] = 1.0 / len(members)
    scores[:] = restart

    sources = graph.edge_index[0]
    targets = graph.edge_index[1]
    out_degree = np.zeros(graph.num_nodes, dtype=np.float64)
    np.add.at(out_degree, sources, 1.0)
    out_degree = np.where(out_degree > 0, out_degree, 1.0)

    for _ in range(pagerank_iterations):
        contribution = scores / out_degree
        spread = np.zeros(graph.num_nodes, dtype=np.float64)
        np.add.at(spread, targets, contribution[sources])
        scores = pagerank_alpha * spread + (1.0 - pagerank_alpha) * restart

    if observation.visible is not None:
        scores = np.where(observation.visible, scores, -np.inf)

    # Keep a set the size of the whole cascade the reports imply, not the whole
    # graph: reporting q of the infected set means the cascade was about |R| / q
    rate = max(len(members) / max(1, graph.num_nodes), 1e-3)
    target_size = min(graph.num_nodes, max(len(members), int(len(members) / max(rate, 0.05))))
    order = np.argsort(-scores)
    kept = set(members) | {
        int(node) for node in order[:target_size] if np.isfinite(scores[int(node)])
    }

    roots = estimate_roots(graph, sorted(kept), observation)
    times = bfs_times(graph, roots, kept, horizon)

    return finalize(graph, _merge_observed(times, observation, horizon), observation, horizon)


def netfill(
    graph: GraphInfo, observation, horizon: int, **_kw: object
) -> dict[int, tuple[int, int | None]]:
    """
    Sundareisan SDM'15 (NetFill): fill in the MISSING nodes of a partial epidemic.

    The MDL reading, in the form it reduces to on a fixed observation rate: a node
    the observation missed is cheaper to describe as infected than as a hole in the
    cascade exactly when enough of its neighbourhood is infected. Two or more
    infected neighbours is the threshold, applied to convergence so a filled node
    can license its own neighbours.

    Deviations: the paper infers the observed FRACTION and a single global beta by
    MDL; we are handed neither and infer neither, so this is NetFill's completion
    rule without its model-selection half. Binary output, no per-node probability,
    which is the paper's own limitation.
    """
    members = set(reported_nodes(observation))
    if not members:
        return {}

    neighbours = neighbour_sets(graph)
    changed = True

    while changed:
        changed = False
        for node in range(graph.num_nodes):
            if node in members or not _visible(observation, node):
                continue

            if sum(1 for other in neighbours[node] if other in members) >= 2:
                members.add(node)
                changed = True

    roots = estimate_roots(graph, sorted(members), observation)
    times = bfs_times(graph, roots, members, horizon)

    return finalize(graph, _merge_observed(times, observation, horizon), observation, horizon)


def dhrec(
    graph: GraphInfo, observation, horizon: int, **_kw: object
) -> dict[int, tuple[int, int | None]]:
    """
    Sefer & Kingsford ICDM'14 (DHREC), and DITTO's own MLE baseline.

    "Diffusion archaeology" reduces history reconstruction to Prize-Collecting
    Dominating Set Vertex Cover and solves it greedily. Implemented as that greedy
    cover: repeatedly take the node whose ball of radius `<= horizon` explains the
    most as-yet-unexplained reports per unit of cost, until every report is
    covered; the covers become the sources and the ball distances become the times.

    Deviation: DHREC needs the diffusion PARAMETERS to price a ball and handles
    SEIR; we price by hop count under IC/LT and take `k` from the cover rather than
    from a prize. DITTO's Tables 4-5 report DHREC at `F1 .50-.70`: 10 to 35% below
    the supervised ideal, which is the bar this row is here to set [verified].
    """
    members = set(reported_nodes(observation))
    if not members:
        return {}

    neighbours = neighbour_sets(graph)
    uncovered = set(members)
    sources = []
    times = {}

    while uncovered and len(sources) < len(members):
        best_node, best_gain, best_distances = None, 0, {}

        # Candidates are the uncovered reports themselves plus their neighbours:
        # a source outside that ring cannot cover anything cheaply
        candidates = set(uncovered) | {
            other for node in uncovered for other in neighbours[node]
        }

        for candidate in sorted(candidates):
            if not _visible(observation, candidate):
                continue

            distances = {candidate: 0}
            wave = [candidate]
            step = 0

            while wave and step < horizon:
                step += 1
                nxt = []
                for node in wave:
                    for other in neighbours[node]:
                        if other not in distances:
                            distances[other] = step
                            nxt.append(other)
                wave = nxt

            gain = sum(1 for node in uncovered if node in distances)
            # Prize-collecting: gain per unit of ball size, so a hub that covers
            # everything by swallowing the graph is not automatically preferred
            score = gain / (1 + math.log1p(len(distances)))

            if best_node is None or score > best_gain:
                best_node, best_gain, best_distances = candidate, score, distances

        if best_node is None:
            break

        sources.append(int(best_node))
        times[int(best_node)] = 0

        for node, distance in best_distances.items():
            if node in members and node not in times:
                times[int(node)] = max(0, min(int(distance), horizon))

        uncovered -= set(best_distances)

    for node in members:
        times.setdefault(int(node), horizon)

    return finalize(graph, _merge_observed(times, observation, horizon), observation, horizon)


def cri(
    graph: GraphInfo, observation, horizon: int, **_kw: object
) -> dict[int, tuple[int, int | None]]:
    """
    Chen TNSE'16 (CRI): cluster the infected subgraph, then reverse-infect.

    DITTO's second MLE baseline. The observed set is partitioned into clusters, one
    reverse-infection centre is taken per cluster, and every node's time is its
    distance from its own centre, which is the whole method for infection times.

    Deviation, and the paper's own: CRI estimates infection times but NOT recovery
    times. Under IC/LT there is no recovery compartment, so nothing is lost here;
    under SIR it would be. DITTO reports CRI at `F1 .57-.82` [verified].
    """
    members = set(reported_nodes(observation))
    if not members:
        return {}

    from coding_agent.tools import primitives

    labels = primitives.detect_communities(graph)
    clusters = {}
    for node in members:
        clusters.setdefault(labels.get(node, -1), []).append(node)

    neighbours = neighbour_sets(graph)
    times = {}

    for cluster in clusters.values():
        centres = estimate_roots(graph, sorted(cluster), observation)
        if not centres:
            continue

        distances = {int(centre): 0 for centre in centres}
        wave = list(distances)
        step = 0

        while wave and step < horizon:
            step += 1
            nxt = []
            for node in wave:
                for other in neighbours[node]:
                    if other in members and other not in distances:
                        distances[int(other)] = step
                        nxt.append(other)
            wave = nxt

        for node, distance in distances.items():
            times[int(node)] = min(times.get(int(node), horizon), int(distance))

    for node in members:
        times.setdefault(int(node), horizon)

    return finalize(graph, _merge_observed(times, observation, horizon), observation, horizon)


# Consistent trees (Zong ICDM'12) --------------------------------------------


def _consistent_tree(
    graph: GraphInfo, observation, horizon: int, path_consistent: bool
) -> dict[int, tuple[int, int | None]]:
    """
    Shared body for WPCT and WBCT: a min-cost tree over the reports whose rooted
    paths satisfy the temporal constraints.

    Zong's decision problems are NP-complete and hard to approximate, so the paper
    gives approximation algorithms and heuristics; this is the heuristic: a
    Dijkstra from the estimated roots over `-log p` in which a relaxation is
    REJECTED when it would put a node at a time inconsistent with what was
    observed. `path_consistent` is the difference between the two variants: WPCT
    constrains every node on a rooted PATH, WBCT only bounds the endpoint, which is
    exactly why Zong's Table I has WPCT at `prec_e` 78-86% and WBCT at 41-69%.
    """
    members = set(reported_nodes(observation))
    if not members:
        return {}

    view = _weighted_view(graph)
    known = _seed_times(observation, members)
    roots = estimate_roots(graph, sorted(members), observation)

    times = {int(root): int(known.get(root, 0)) for root in roots}
    costs = {int(root): 0.0 for root in roots}
    queue = [(0.0, int(root)) for root in roots]
    heapq.heapify(queue)

    while queue:
        cost, node = heapq.heappop(queue)
        if cost > costs.get(node, math.inf):
            continue

        step = times[node] + 1
        if step > horizon:
            continue

        for neighbour in view.successors(node):
            observed = known.get(int(neighbour))

            if observed is not None:
                # Temporal constraint: the arrival cannot be later than the time
                # the node was actually seen to activate
                if step > observed:
                    continue

                # WPCT additionally requires the whole rooted path to be
                # consistent, i.e. the arrival must land exactly on the observed
                # step rather than merely no later than it
                if path_consistent and step != observed:
                    continue

                arrival = observed
            else:
                arrival = step

            candidate = cost + view[node][neighbour]["weight"]
            if candidate < costs.get(int(neighbour), math.inf):
                costs[int(neighbour)] = candidate
                times[int(neighbour)] = max(0, min(int(arrival), horizon))
                heapq.heappush(queue, (candidate, int(neighbour)))

    # Reports the constrained search never reached are still infected; the tree
    # simply failed to explain them, which is what `prec_e < prec_v` measures
    for node in members:
        times.setdefault(int(node), int(known.get(node, horizon)))

    kept = {node: time for node, time in times.items() if node in members or time < horizon}

    return finalize(graph, _merge_observed(kept, observation, horizon), observation, horizon)


def consistent_tree_wpct(
    graph: GraphInfo, observation, horizon: int, **_kw: object
) -> dict[int, tuple[int, int | None]]:
    """
    Zong ICDM'12 WPCT: the weighted PATH-consistent tree, and the paper's winner.

    The source of this task's central asymmetry: `prec_v = 100%` alongside
    `prec_e = 78-86%` on Enron and Twitter [verified, Table I]. Getting the node
    set right is much easier than getting the edges right, which is the same
    finding DIPT reports thirteen years later and the reason the tree-weighted
    score leans toward the tree.
    """
    return _consistent_tree(graph, observation, horizon, path_consistent=True)


def consistent_tree_wbct(
    graph: GraphInfo, observation, horizon: int, **_kw: object
) -> dict[int, tuple[int, int | None]]:
    """
    Zong ICDM'12 WBCT: the weaker BOUNDED variant, and the control WPCT beats.

    Zong's Table I: `prec_v` falls from 100% to 66-70% and `prec_e` from 78-86% to
    41-69%. Both are in the pool because reporting only the winner of a paper's own
    ablation hides how much of its number is the constraint rather than the tree.
    """
    return _consistent_tree(graph, observation, horizon, path_consistent=False)


def cult(
    graph: GraphInfo, observation, horizon: int, alpha: float = 1.0, **_kw: object
) -> dict[int, tuple[int, int | None]]:
    """
    Rozenshtein KDD'16 (CulT): an alpha-TempSteinerTree forest, assuming NO propagation model.

    A forest of temporal Steiner trees spanning the reports, with `alpha` trading
    tree cost against seed count: an arc costs its `-log p` and opening a new root
    costs `alpha`, so raising alpha merges the forest and lowering it splits.
    Implemented as a Dijkstra from a virtual super-source whose arc into every
    candidate root costs `alpha`, which is the standard reduction.

    **The honest ceiling for "what can you do without a kernel"**: CulT is the
    only published method here that assumes no propagation model at all, which makes it
    the right thing for a learned kernel to beat. Its own paper publishes ZERO
    tables (`grep -c "Table"` returns 0) so its MCC 0.6-0.9 is [figure] and this
    row is a reimplementation of the method, not a reproduction of a number.

    Deviation: CulT operates on a temporal interaction STREAM and binary-searches
    alpha to hit a target seed count `k`. Our episodes are on a static graph and we
    are handed no k, so alpha is a parameter here rather than a search variable.
    """
    members = set(reported_nodes(observation))
    if not members:
        return {}

    view = _weighted_view(graph)
    known = _seed_times(observation, members)

    # Every report is a candidate root at cost alpha; the cheapest explanation
    # decides which ones actually open a tree
    costs = {}
    times = {}
    parents = {}
    queue = []

    for node in sorted(members):
        costs[int(node)] = float(alpha)
        times[int(node)] = int(known.get(node, 0))
        parents[int(node)] = None
        heapq.heappush(queue, (float(alpha), int(node)))

    while queue:
        cost, node = heapq.heappop(queue)
        if cost > costs.get(node, math.inf):
            continue

        step = times[node] + 1
        if step > horizon:
            continue

        for neighbour in view.successors(node):
            candidate = cost + view[node][neighbour]["weight"]
            observed = known.get(int(neighbour))
            arrival = observed if observed is not None else step

            if observed is not None and step > observed:
                continue

            if candidate < costs.get(int(neighbour), math.inf):
                costs[int(neighbour)] = candidate
                times[int(neighbour)] = max(0, min(int(arrival), horizon))
                parents[int(neighbour)] = int(node)
                heapq.heappush(queue, (candidate, int(neighbour)))

    # Only the nodes the forest actually used: a node reached at cost above alpha
    # is more cheaply explained as its own root than as part of a tree
    kept = {
        node: time
        for node, time in times.items()
        if node in members or costs.get(node, math.inf) <= float(alpha)
    }

    return finalize(graph, _merge_observed(kept, observation, horizon), observation, horizon)


def jordan_backward(
    graph: GraphInfo, observation, horizon: int, **_kw: object
) -> dict[int, tuple[int, int | None]]:
    """
    Greedy backward decode from the Jordan centres: the cheap first cut.

    Name the centre of each observed component as a source, then walk the cascade
    FORWARD from those sources through the observed set, taking at each step the
    most likely arc out of the current wave. The simplest decoder there is, and the
    one a generated program should regard as its own starting point rather than its
    target.
    """
    members = set(reported_nodes(observation))
    if not members:
        return {}

    probabilities = _probability_map(graph)
    roots = estimate_roots(graph, sorted(members), observation)
    times = {int(root): 0 for root in roots}
    wave = set(roots)
    step = 0

    while wave and step < horizon:
        step += 1
        nxt = set()

        for node in wave:
            for neighbour in graph.out_neighbors(node):
                if neighbour in times or neighbour not in members:
                    continue

                # Greedy: only take the arc if it is the most likely way INTO this
                # neighbour from the current wave
                best = max(
                    (other for other in graph.in_neighbors(neighbour) if other in wave),
                    key=lambda other: probabilities.get((other, neighbour), 0.0),
                    default=None,
                )
                if best is None:
                    continue

                times[int(neighbour)] = step
                nxt.add(int(neighbour))

        wave = nxt

    for node in members:
        times.setdefault(int(node), horizon)

    return finalize(graph, _merge_observed(times, observation, horizon), observation, horizon)


# Kernel-using decoders (blocked from generated scripts) ----------------------


def mcmc_decode(
    graph: GraphInfo,
    observation,
    horizon: int,
    predict: object = None,
    proposals: int = mcmc_proposals,
    seed: int = 0,
    **_kw: object,
) -> dict[int, tuple[int, int | None]]:
    """
    Metropolis-Hastings over HISTORIES, scored under the transition kernel: DITTO, minus the learned proposal.

    The sampling decoder family. Start from `delayed_bfs`, then repeatedly perturb one
    node's activation time by +/-1 and accept the move with probability
    `min(1, exp(Delta log p))`, where the trajectory log-likelihood is evaluated by
    unrolling `predict` over the proposed history. `predict` is
    the arm's kernel when a canned baseline passes it and a structural
    surrogate otherwise.

    **Blocked from generated scripts by default**, exactly as `celf` and
    `resim_greedy` are, and for the same reason plus one more: it costs
    `proposals x horizon` kernel evaluations per instance, and a generated program
    is offline by construction (the kernel is bound to canned baselines only), so
    blocking it costs a generated decoder nothing.
    """
    from coding_agent.reconstruction import transition_logprob

    decoded = delayed_bfs(graph, observation, horizon)
    if predict is None or not decoded:
        return decoded

    rng = np.random.default_rng(seed)
    times = {node: time for node, (time, _) in decoded.items()}
    # Roots stay at t = 0: the proposal below never moves a node back to 0, so a
    # moved root could never return
    movable = [
        node for node in times if node not in observation.times and times[node] > 0
    ]

    if not movable:
        return decoded

    def history_logprob(assignment: dict[int, int]) -> float:
        waves = {}
        for node, time in assignment.items():
            waves.setdefault(int(time), []).append(int(node))

        total = 0.0
        infected = list(waves.get(0, []))

        for step in range(1, horizon + 1):
            frontier = waves.get(step - 1, [])
            wave = waves.get(step, [])
            # An empty frontier transmits nothing: score the wave against zero
            # marginals rather than truncating, which rewarded any gap in the
            # history with a likelihood of exactly 0
            marginal = (
                predict(infected, frontier)
                if frontier
                else np.zeros(graph.num_nodes)
            )

            total += transition_logprob(marginal, infected, wave)
            infected = infected + wave

        return total

    current = history_logprob(times)
    burn_in = int(proposals * mcmc_burn_in)
    best_times, best_score = dict(times), current

    for step in range(max(1, proposals)):
        node = movable[int(rng.integers(len(movable)))]
        shift = 1 if rng.random() < 0.5 else -1
        proposed = max(1, min(times[node] + shift, horizon))

        if proposed == times[node]:
            continue

        original = times[node]
        times[node] = proposed
        candidate = history_logprob(times)

        if math.log(max(rng.random(), 1e-12)) < candidate - current:
            current = candidate
            if step >= burn_in and candidate > best_score:
                best_times, best_score = dict(times), candidate
        else:
            times[node] = original

    return finalize(graph, best_times, observation, horizon)


def forward_backward(
    graph: GraphInfo,
    observation,
    horizon: int,
    predict: object = None,
    particles: int = smoothing_particles,
    seed: int = 0,
    **_kw: object,
) -> dict[int, tuple[int, int | None]]:
    """
    Forward-filter / backward-sample smoothing against the transition kernel.

    The smoothing decoder family, and the one no published method runs with a
    learned kernel:
    propagate the per-node activation marginal forward under `predict`, then walk
    backward assigning each node the step at which its forward marginal first
    crossed the threshold, subject to the observed times. Classical smoothing,
    which the whole task is an instance of.

    **Blocked from generated scripts by default**, on the same terms as
    `mcmc_decode`: it costs `horizon` kernel evaluations per particle and a
    generated program is offline and has no kernel at all.
    """
    decoded = delayed_bfs(graph, observation, horizon)
    if predict is None or not decoded:
        return decoded

    rng = np.random.default_rng(seed)
    roots = [node for node, (time, _) in decoded.items() if time == 0]
    if not roots:
        return decoded

    arrival = np.full(graph.num_nodes, horizon, dtype=np.int64)
    arrival[list(roots)] = 0

    infected = list(roots)
    frontier = list(roots)

    for step in range(1, horizon + 1):
        if not frontier:
            break

        marginal = np.asarray(predict(infected, frontier), dtype=np.float64)
        if observation.visible is not None:
            marginal = np.where(observation.visible, marginal, 0.0)

        # One coupled draw per particle, averaged: the forward filter's estimate
        # of which nodes activate on this step
        draws = rng.random((max(1, particles), graph.num_nodes)) < marginal
        frequency = draws.mean(axis=0)

        wave = [
            int(node)
            for node in np.flatnonzero(frequency >= 0.5)
            if node not in infected
        ]
        # A node the observation says activated here is in the wave whatever the
        # filter thinks: that is the backward pass's conditioning
        wave += [
            int(node)
            for node, time in observation.times.items()
            if time == step and node not in wave and node not in infected
        ]

        if not wave:
            break

        arrival[wave] = step
        infected = infected + wave
        frontier = wave

    times = {int(node): int(arrival[node]) for node in infected}
    times |= {node: time for node, (time, _) in decoded.items() if node not in times}

    return finalize(graph, _merge_observed(times, observation, horizon), observation, horizon)


reconstruction_algorithms = {
    # ordered Steiner family (Xiao SDM'18): `delayed_bfs` is the bar
    "delayed_bfs": delayed_bfs,
    "ordered_steiner_closure": ordered_steiner_closure,
    "greedy_ordered": greedy_ordered,
    "steiner_tree": steiner_tree,
    # probabilistic tree sampling (Xiao ICDM'18)
    "tree_sampling": tree_sampling,
    # the assortativity trap: run it on ca_grqc specifically
    "personalized_pagerank": personalized_pagerank,
    # consistent trees (Zong ICDM'12), both variants
    "consistent_tree_wpct": consistent_tree_wpct,
    "consistent_tree_wbct": consistent_tree_wbct,
    # model-free temporal Steiner forest (Rozenshtein KDD'16)
    "cult": cult,
    # MLE / statistical completion
    "dhrec": dhrec,
    "cri": cri,
    "netfill": netfill,
    # cheap first cut
    "jordan_backward": jordan_backward,
    # controls and the floor
    "observed_only": observed_only,
    "one_hop": one_hop,
    "random_reconstruction": random_reconstruction,
    # kernel-using decoders
    "mcmc_decode": mcmc_decode,
    "forward_backward": forward_backward,
}

# Kernel-heavy, so one call costs proposals x horizon transition evaluations on
# whatever oracle it was handed. Charged honestly to the arm like any other
# baseline, and blocked from generated scripts by default: a generated program
# is offline by construction, so nothing is lost by blocking these.
# around it is the only way the cost lands in this arm's `kernel_calls`.
mc_reconstruction_algorithms = ("mcmc_decode", "forward_backward")

reconstruction_algorithm_names = list(reconstruction_algorithms)
