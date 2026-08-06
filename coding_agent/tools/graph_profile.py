"""
Aggregate structural statistics describing the graph in the agent's prompt.

Deliberately NOT the adjacency list. The generated code reads the real topology
through `graph` when it runs, so serializing edges into the prompt would only ask
the model to eyeball what its own program computes exactly, and would let it
hardcode node ids for one instance instead of writing an algorithm that
generalizes. These statistics are what a human expert actually uses to choose an
approach: is it hub-dominated or uniform, one component or many, community-
structured or not, above or below the cascade threshold.

Degrees are reported with `GraphInfo.degree()` semantics (total, in + out) so the
numbers here match what the generated code observes.
"""

import networkx as nx
import numpy as np

from coding_agent.types import GraphInfo

# Above this many arcs the neighbourhood-heavy statistics are sampled or skipped
heavy_stats_max_arcs = 200_000
clustering_sample_trials = 2000
listed_community_sizes = 5
profile_seed = 42
# Measured on 100-node synthetic families: ba cv=0.95, sbm 0.46, er 0.40, ws 0.11,
# so this cleanly separates hub-dominated graphs from the rest
heavy_tail_cv = 0.7


def _undirected_view(graph: GraphInfo) -> nx.Graph:
    """Simple undirected view: reciprocal arc pairs collapse to one edge."""
    view = nx.Graph()
    view.add_nodes_from(range(graph.num_nodes))
    view.add_edges_from(zip(graph.edge_index[0], graph.edge_index[1], strict=True))
    view.remove_edges_from(nx.selfloop_edges(view))

    return view


def _degree_line(degrees: np.ndarray) -> str:
    mean = float(degrees.mean())
    # CV separates BA-style hub graphs from ER/SBM-style uniform ones
    coefficient_of_variation = float(degrees.std() / mean) if mean > 0 else 0.0
    tail = (
        "heavy-tailed, hub-dominated: degree-based seeding is strong"
        if coefficient_of_variation > heavy_tail_cv
        else "fairly uniform: degree carries little seed signal"
    )

    return (
        f"degree (total, in+out): mean={mean:.2f} median={np.median(degrees):.0f} "
        f"max={degrees.max()} p90={np.percentile(degrees, 90):.0f} "
        f"p99={np.percentile(degrees, 99):.0f} cv={coefficient_of_variation:.2f} "
        f"max/mean={degrees.max() / mean if mean > 0 else 0:.1f}  ({tail})"
    )


def _transmission_lines(graph: GraphInfo) -> list[str]:
    """IC edge weights and the per-node expected out-transmission (an R0 proxy)."""
    probabilities = np.asarray(graph.ic_probs, dtype=np.float64)
    if probabilities.size == 0:
        return []

    out_transmission = np.zeros(graph.num_nodes, dtype=np.float64)
    np.add.at(out_transmission, graph.edge_index[0], probabilities)
    active = out_transmission[out_transmission > 0]

    lines = [
        f"IC edge prob: mean={probabilities.mean():.3f} "
        f"min={probabilities.min():.3f} max={probabilities.max():.3f}"
    ]

    if active.size:
        # The MEAN is pinned to ~1.0 by the weighted-cascade construction
        # (p = 1/in_degree makes every node's in-sum exactly 1), so it carries no
        # information about this graph: the median and the supercritical
        # fraction are what distinguish one graph from another
        median_transmission = float(np.median(active))
        supercritical = 100.0 * float((active >= 1.0).mean())
        # Exactly 1.0 is critical, not subcritical, and under weighted cascade a
        # median of exactly 1.0 is common, so the boundary has to fall this way
        regime = (
            "the typical node at least replaces itself: cascades sustain themselves"
            if median_transmission >= 1.0
            else "the typical node is subcritical, so cascades run on the "
            "supercritical minority: reaching those hubs is what decides spread"
        )
        lines.append(
            f"expected out-transmission per node (sum of p over out-edges): "
            f"median={median_transmission:.2f}, {supercritical:.0f}% of nodes "
            f"at or above 1.0  ({regime})"
        )

    return lines


def _component_lines(view: nx.Graph, degrees: np.ndarray, num_nodes: int) -> list[str]:
    components = sorted(nx.connected_components(view), key=len, reverse=True)
    isolates = int((degrees == 0).sum())
    largest = len(components[0]) if components else 0

    lines = [
        f"components: {len(components)} (largest holds "
        f"{100.0 * largest / max(num_nodes, 1):.1f}% of nodes, {isolates} isolates)"
    ]

    if len(components) > 1:
        lines.append(
            "  budget spent inside one component cannot reach the others: spread "
            "is capped by which components you seed"
        )

    return lines


def build_graph_profile(graph: GraphInfo) -> str:
    """Cached on the GraphInfo: per_step rebuilds its prompt on every LLM call."""
    cached = getattr(graph, "_profile", None)
    if cached is not None:
        return cached

    num_arcs = int(graph.edge_index.shape[1])
    degrees = np.array(
        [graph.degree(node) for node in range(graph.num_nodes)], dtype=np.int64
    )

    lines = [
        f"num_nodes = {graph.num_nodes}",
        f"num_arcs = {num_arcs}"
        + ("" if graph.directed else "  (undirected: each edge stored both ways)"),
        f"directed = {graph.directed}",
        _degree_line(degrees),
    ]

    view = _undirected_view(graph)
    lines.append(f"density = {nx.density(view):.5f}")
    lines += _component_lines(view, degrees, graph.num_nodes)

    core_numbers = nx.core_number(view)
    max_core = max(core_numbers.values(), default=0)
    lines.append(
        f"k-core: max core = {max_core}, "
        f"{sum(1 for value in core_numbers.values() if value == max_core)} nodes in it "
        f"(the densely interconnected spreading core)"
    )

    if graph.directed:
        directed_view = nx.DiGraph(
            zip(graph.edge_index[0], graph.edge_index[1], strict=True)
        )
        lines.append(
            f"reciprocity = {nx.reciprocity(directed_view):.3f}  "
            f"(fraction of arcs whose reverse also exists)"
        )

    if num_arcs <= heavy_stats_max_arcs:
        lines.append(
            f"clustering coefficient (avg) = {nx.average_clustering(view):.3f}  "
            f"(high = tightly knit neighbourhoods, so neighbouring seeds overlap)"
        )
        lines.append(
            f"degree assortativity = "
            f"{nx.degree_assortativity_coefficient(view):+.3f}  "
            f"(positive = hubs attach to hubs)"
        )

        communities = sorted(
            nx.community.label_propagation_communities(view), key=len, reverse=True
        )
        modularity = nx.community.modularity(view, communities)
        sizes = [len(community) for community in communities[:listed_community_sizes]]
        lines.append(
            f"communities (label propagation): {len(communities)}, "
            f"modularity = {modularity:.3f}, largest sizes = {sizes}"
        )
        if modularity > 0.3:
            lines.append(
                "  strong community structure: allocating budget ACROSS communities "
                "usually beats picking globally top-scoring nodes"
            )
    else:
        sampled_clustering = nx.approximation.average_clustering(
            view, trials=clustering_sample_trials, seed=profile_seed
        )
        lines.append(
            f"clustering coefficient (~sampled, {clustering_sample_trials} trials) = "
            f"{sampled_clustering:.3f}"
        )
        lines.append(
            f"assortativity and communities: SKIPPED (graph exceeds "
            f"{heavy_stats_max_arcs} arcs)"
        )

    lines += _transmission_lines(graph)

    profile = "GRAPH PROFILE:\n" + "\n".join(lines)
    graph._profile = profile

    return profile
