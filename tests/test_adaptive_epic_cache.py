import numpy as np

from coding_agent.tools import adaptive_algorithms, primitives
from coding_agent.types import GraphInfo, State


def _graph(seed: int) -> GraphInfo:
    rng = np.random.default_rng(seed)
    edges = rng.integers(0, 40, size=(2, 200))
    return GraphInfo(num_nodes=40, edge_index=edges, ic_probs=np.full(200, 0.2, dtype=np.float32), directed=True)


def test_epic_samples_once_per_graph_and_seed(monkeypatch) -> None:
    calls = []
    real = primitives.batch_reverse_sample
    monkeypatch.setattr(primitives, "batch_reverse_sample", lambda graph, theta, seed=0: calls.append(seed) or real(graph, theta=theta, seed=seed))
    adaptive_algorithms.epic_rr_cache.clear()
    graph = _graph(1)
    state = State(infected=set(), frontier=set())

    first = adaptive_algorithms.adapt_epic(state, graph, 3)
    second = adaptive_algorithms.adapt_epic(State(infected={first[0]}, frontier={first[0]}), graph, 3)
    assert calls == [0]
    assert first[0] not in second

    adaptive_algorithms.adapt_epic(state, graph, 3, seed=7)
    adaptive_algorithms.adapt_epic(state, _graph(2), 3)
    assert calls == [0, 7, 0]
