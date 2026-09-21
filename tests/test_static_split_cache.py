import networkx as nx
import numpy as np
import pytest

from coding_agent.tools import algorithms
from coding_agent.tools.adaptive_algorithms import static_split
from coding_agent.types import GraphInfo, State


def small_graph() -> GraphInfo:
    built = nx.barabasi_albert_graph(60, 3, seed=1)
    arcs = [arc for u, v in built.edges() for arc in ((u, v), (v, u))]
    edge_index = np.array(arcs, dtype=np.int64).T
    return GraphInfo(num_nodes=60, edge_index=edge_index, ic_probs=np.full(edge_index.shape[1], 0.1, dtype=np.float32), directed=False)


def test_static_split_ranks_once_per_graph_and_matches_a_fresh_ranking(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = small_graph()
    base = algorithms.algorithms["degree_discount"]
    lengths = []

    def counting(graph: GraphInfo, budget: int, diffusion_model: str = "IC") -> list[int]:
        lengths.append(budget)
        return base(graph, budget, diffusion_model)

    monkeypatch.setitem(algorithms.algorithms, "degree_discount", counting)

    # the harness calls a policy once per round per ensemble member: same graph, growing active set
    for active in ([], [3, 4], [3, 4, 9, 11, 20], [3, 4], []):
        picks = static_split(State(infected=set(active), frontier=set()), graph, batch=4, total_budget=8)
        fresh = [node for node in base(graph, 8 + len(active)) if node not in active][:4]
        assert picks == fresh

    # only a request longer than anything cached ranks again: 8, then 10, then 13
    assert lengths == [8, 10, 13]


def test_static_split_never_reads_another_graph_objects_ranking() -> None:
    first, second = small_graph(), small_graph()
    static_split(State(infected=set(), frontier=set()), first, batch=4, total_budget=8)

    assert first._static_rankings and second._static_rankings is None
