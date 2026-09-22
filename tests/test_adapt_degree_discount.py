import time

import networkx as nx
import numpy as np

from coding_agent.tools import primitives
from coding_agent.tools.adaptive_algorithms import adapt_degree_discount
from coding_agent.types import GraphInfo, State


def graph_of(built: nx.Graph) -> GraphInfo:
    arcs = [arc for u, v in built.edges() for arc in ((u, v), (v, u))]
    edge_index = np.array(arcs, dtype=np.int64).T
    return GraphInfo(num_nodes=built.number_of_nodes(), edge_index=edge_index, ic_probs=np.full(edge_index.shape[1], 0.1, dtype=np.float32), directed=False)


def reference(state: State, graph: GraphInfo, batch: int) -> list[int]:
    """The loop the vectorized version replaced, with ties broken by lowest id."""
    active = set(state.infected) | set(state.frontier)
    discounted = primitives.compute_degree(graph)
    for node in range(graph.num_nodes):
        discounted[node] -= sum(1 for neighbour in graph.out_neighbors(node) + graph.in_neighbors(node) if neighbour in active)

    chosen = []
    available = set(range(graph.num_nodes)) - active
    for _ in range(min(batch, len(available))):
        node = max(sorted(available), key=lambda candidate: discounted[candidate])
        chosen.append(node)
        available.discard(node)
        for neighbour in graph.out_neighbors(node) + graph.in_neighbors(node):
            discounted[neighbour] -= 1

    return chosen


def test_adapt_degree_discount_matches_the_reference_loop() -> None:
    rng = np.random.default_rng(0)

    for seed in range(6):
        graph = graph_of(nx.barabasi_albert_graph(120, 3, seed=seed))
        infected = set(rng.choice(graph.num_nodes, size=int(rng.integers(0, 30)), replace=False).tolist())
        frontier = set(rng.choice(graph.num_nodes, size=int(rng.integers(0, 10)), replace=False).tolist())
        state = State(infected=infected, frontier=frontier)

        for batch in (1, 7, 40, 500):
            picks = adapt_degree_discount(state, graph, batch)
            assert picks == reference(state, graph, batch)
            assert not (set(picks) & (infected | frontier))


def test_adapt_degree_discount_is_fast_on_a_large_batch() -> None:
    graph = graph_of(nx.barabasi_albert_graph(20_000, 5, seed=0))
    state = State(infected=set(range(0, 2_000)), frontier=set(range(2_000, 2_500)))

    start = time.perf_counter()
    picks = adapt_degree_discount(state, graph, 4_000)
    elapsed = time.perf_counter() - start

    assert len(picks) == 4_000 and len(set(picks)) == 4_000
    # the old loop took about a minute here
    assert elapsed < 5.0
