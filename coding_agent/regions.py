"""
Graph regions: the partitions node marginals are aggregated over.

The world model already predicts `p_v = P(v influenced | S)` for every node. Two
thousand probabilities are not feedback; a handful of REGIONAL sums are. What
makes an aggregate meaningful is entirely the partition, so this module holds
the partitions and nothing else — the aggregation itself is one `sum` in
`coding_agent/diagnostics.py`.

Every scheme here is derived from an actual graph statistic. There is no
"core area" label that is not the top decile of a named centrality, and no
region whose definition is a round number chosen to make a story work: the
thresholds are module constants with their reasoning attached, and they travel
into the report so a reader can move them.

| scheme       | partition by                                | extra WM calls |
| ------------ | ------------------------------------------- | -------------- |
| `community`  | label-propagation communities               | 0              |
| `structural` | inter-community incidence, betweenness, degree | 0           |
| `seed_basin` | which seed dominates a node's predicted activation | 1 per seed |

`community` is the default and is what the first experiment uses: it is the
partition influence maximization is actually about (a seed set that covers three
communities and misses the fourth is the canonical failure), and the repository
already computes it for the existing feedback, so the two agree by construction.

`structural` and `seed_basin` are alternatives rather than replacements. Only
`seed_basin` costs world-model queries, and it says so in `wm_calls`.
"""

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from coding_agent.tools import primitives
from coding_agent.types import GraphInfo

#: Scheme names accepted by `build_regions`.
community_scheme = "community"
structural_scheme = "structural"
basin_scheme = "seed_basin"
valid_schemes = (community_scheme, structural_scheme, basin_scheme)

#: A node is CORE if its total degree is at or above this percentile. The decile
#: is the conventional cut for "hub" in the IM literature and is the same
#: quantile the bridge cut below uses, so the two bands are comparable.
core_degree_percentile = 90.0
#: ...and BRIDGE if it also carries at least one inter-community edge and sits at
#: or above this betweenness percentile. Betweenness alone over-selects on a
#: hub-dominated graph (every hub is on many shortest paths); requiring an actual
#: inter-community incident edge is what makes it a bridge rather than a hub.
bridge_betweenness_percentile = 90.0
#: A node is PERIPHERY at or below this degree percentile.
periphery_degree_percentile = 50.0

#: Above this node count betweenness is estimated from a pivot sample rather than
#: computed exactly; exact Brandes is O(NE) and the diagnostics run every turn.
exact_betweenness_max_nodes = 1500
#: Pivots for the sampled estimate. NetworkX's own guidance is that a few hundred
#: pivots put the RANKING (which is all this is used for) close to exact.
betweenness_pivots = 256


@dataclass(frozen=True)
class Regions:
    """A partition of the node set, plus what produced it."""

    scheme: str
    #: {region key: sorted member nodes}. Every node appears exactly once.
    members: dict[int | str, list[int]]
    #: {node: region key}, the inverse.
    labels: dict[int, int | str]
    #: Human-readable name per region key, for the prompt text.
    names: dict[int | str, str]
    #: World-model rollouts this partition cost to build (0 for the two
    #: structure-only schemes).
    wm_calls: int = 0
    #: Free-form provenance, e.g. the seed each basin belongs to.
    detail: dict = field(default_factory=dict)

    def size(self, key) -> int:
        return len(self.members[key])

    def ranked(self) -> list:
        """Region keys, largest first — the order feedback lists them in."""
        return sorted(self.members, key=lambda key: len(self.members[key]), reverse=True)


def undirected_view(graph: GraphInfo) -> nx.Graph:
    """
    Simple undirected view, cached on the GraphInfo.

    Betweenness and the community detector both want one, and rebuilding it per
    turn on a 30K-edge graph was measurable next to the rollouts it annotates.
    """
    cached = getattr(graph, "_nx_undirected", None)

    if cached is not None:
        return cached

    view = nx.Graph()
    view.add_nodes_from(range(graph.num_nodes))
    view.add_edges_from(
        (int(graph.edge_index[0, edge]), int(graph.edge_index[1, edge]))
        for edge in range(graph.edge_index.shape[1])
    )
    # GraphInfo is a plain dataclass, so the cache is an attribute rather than a
    # declared field: it is derived, and adding it to the schema would put it in
    # every repr and every equality check
    graph._nx_undirected = view

    return view


def betweenness(graph: GraphInfo) -> np.ndarray:
    """(N,) betweenness centrality, cached; sampled above the exact-size cap."""
    cached = getattr(graph, "_betweenness", None)

    if cached is not None:
        return cached

    view = undirected_view(graph)
    scores = nx.betweenness_centrality(
        view,
        k=(
            None
            if graph.num_nodes <= exact_betweenness_max_nodes
            else min(betweenness_pivots, graph.num_nodes)
        ),
        # Fixed pivots: the diagnostics are re-derived every turn and a resampled
        # estimate would make a region's membership drift for no reason
        seed=0,
        normalized=True,
    )
    values = np.zeros(graph.num_nodes, dtype=np.float64)
    for node, score in scores.items():
        values[int(node)] = float(score)

    graph._betweenness = values

    return values


def community_labels(graph: GraphInfo) -> dict[int, int]:
    """{node: community_id}. The same cache `methods.base._communities` fills, so
    the new diagnostics and the existing feedback can never disagree."""
    if graph._community_labels is None:
        graph._community_labels = primitives.detect_communities(graph)

    return graph._community_labels


def inter_community_incidence(graph: GraphInfo) -> np.ndarray:
    """(N,) how many incident edges leave the node's own community."""
    labels = community_labels(graph)
    counts = np.zeros(graph.num_nodes, dtype=np.int64)

    for edge in range(graph.edge_index.shape[1]):
        source = int(graph.edge_index[0, edge])
        destination = int(graph.edge_index[1, edge])

        if labels.get(source, -1) != labels.get(destination, -2):
            counts[source] += 1
            counts[destination] += 1

    return counts


def bridge_nodes(graph: GraphInfo) -> list[int]:
    """
    Nodes that are both structurally central AND actually between communities.

    Returned ranked by betweenness, which is the order the bridge feedback lists
    them in and the order a reviser would work down.
    """
    scores = betweenness(graph)
    incidence = inter_community_incidence(graph)

    if not scores.size:
        return []

    cut = float(np.percentile(scores, bridge_betweenness_percentile))
    candidates = [
        node
        for node in range(graph.num_nodes)
        if incidence[node] > 0 and scores[node] >= cut
    ]

    return sorted(candidates, key=lambda node: -scores[node])


def betweenness_percentile(graph: GraphInfo, node: int) -> float:
    scores = betweenness(graph)

    if not scores.size:
        return 0.0

    return float(100.0 * (scores <= scores[int(node)]).mean())


def _community_regions(graph: GraphInfo) -> Regions:
    labels = dict(community_labels(graph))
    members: dict[int | str, list[int]] = {}

    for node in range(graph.num_nodes):
        members.setdefault(labels.get(node, -1), []).append(node)

    return Regions(
        scheme=community_scheme,
        members={key: sorted(group) for key, group in members.items()},
        labels={node: labels.get(node, -1) for node in range(graph.num_nodes)},
        names={key: f"C{key}" for key in members},
    )


def _structural_regions(graph: GraphInfo) -> Regions:
    """
    Four structural roles, assigned by PRECEDENCE so the result is a partition.

    bridge > core > periphery > interior. Bridge outranks core because a node
    that is both is interesting for the bridge reason; interior is the residual
    and is named as such rather than given a story.
    """
    scores = betweenness(graph)
    incidence = inter_community_incidence(graph)
    degrees = np.array(
        [graph.degree(node) for node in range(graph.num_nodes)], dtype=np.float64
    )

    bridge_cut = float(np.percentile(scores, bridge_betweenness_percentile)) if scores.size else 0.0
    core_cut = float(np.percentile(degrees, core_degree_percentile)) if degrees.size else 0.0
    periphery_cut = (
        float(np.percentile(degrees, periphery_degree_percentile)) if degrees.size else 0.0
    )

    labels: dict[int, int | str] = {}
    for node in range(graph.num_nodes):
        if incidence[node] > 0 and scores[node] >= bridge_cut:
            labels[node] = "bridge"
        elif degrees[node] >= core_cut:
            labels[node] = "core"
        elif degrees[node] <= periphery_cut:
            labels[node] = "periphery"
        else:
            labels[node] = "interior"

    members: dict[int | str, list[int]] = {}
    for node, key in labels.items():
        members.setdefault(key, []).append(node)

    return Regions(
        scheme=structural_scheme,
        members={key: sorted(group) for key, group in members.items()},
        labels=labels,
        names={
            "bridge": (
                f"bridge (inter-community edge, betweenness >= p"
                f"{bridge_betweenness_percentile:.0f})"
            ),
            "core": f"core (degree >= p{core_degree_percentile:.0f})",
            "periphery": f"periphery (degree <= p{periphery_degree_percentile:.0f})",
            "interior": "interior (the residual)",
        },
        detail={
            "bridge_betweenness_cut": bridge_cut,
            "core_degree_cut": core_cut,
            "periphery_degree_cut": periphery_cut,
        },
    )


def basin_regions(
    seeds: list[int], solo_marginals: dict[int, list[float]], num_nodes: int
) -> Regions:
    """
    Group each node under the seed that most strongly predicts its activation.

    `solo_marginals[s][v]` is `P(v influenced | {s})` from a single-seed
    world-model rollout, so the caller has already paid (and counted) one query
    per seed. A node no seed reaches above `basin_floor` goes to `unreached`,
    which is a real region and the one a reviser most wants named.
    """
    #: Below this predicted activation a node is not in anyone's basin. Matches
    #: `methods.base.unreached_threshold`, so "unreached" means one thing.
    basin_floor = 0.10

    labels: dict[int, int | str] = {}
    for node in range(num_nodes):
        best_seed, best_value = None, basin_floor
        for seed in seeds:
            value = solo_marginals[seed][node]
            if value > best_value:
                best_seed, best_value = seed, value

        labels[node] = "unreached" if best_seed is None else best_seed

    members: dict[int | str, list[int]] = {}
    for node, key in labels.items():
        members.setdefault(key, []).append(node)

    return Regions(
        scheme=basin_scheme,
        members={key: sorted(group) for key, group in members.items()},
        labels=labels,
        names={
            key: (f"unreached (no seed above {basin_floor:.0%})" if key == "unreached" else f"basin of seed {key}")
            for key in members
        },
        wm_calls=len(seeds),
        detail={"basin_floor": basin_floor, "seeds": list(seeds)},
    )


def build_regions(graph: GraphInfo, scheme: str = community_scheme) -> Regions:
    """The two structure-only schemes. `seed_basin` needs marginals, so it has
    its own entry point (`basin_regions`) that takes them."""
    if scheme == community_scheme:
        return _community_regions(graph)

    if scheme == structural_scheme:
        return _structural_regions(graph)

    if scheme == basin_scheme:
        raise ValueError(
            "seed_basin is plan-dependent: build it with basin_regions(seeds, "
            "solo_marginals, num_nodes), which also reports its world-model cost"
        )

    raise ValueError(f"unknown region scheme {scheme!r}; choose from {list(valid_schemes)}")


__all__ = [
    "Regions",
    "basin_regions",
    "basin_scheme",
    "betweenness",
    "betweenness_percentile",
    "bridge_nodes",
    "build_regions",
    "community_labels",
    "community_scheme",
    "inter_community_incidence",
    "structural_scheme",
    "valid_schemes",
]
