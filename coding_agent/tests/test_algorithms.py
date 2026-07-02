# coding_agent/tests/test_algorithms.py
import numpy as np
import pytest
from coding_agent.types import GraphInfo
from coding_agent.tools import algorithms as A


def _hub_graph() -> GraphInfo:
    src = [0, 0, 0, 0, 0, 1, 2, 3, 4, 5]
    dst = [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]
    ei = np.array([src, dst], dtype=np.int64)
    ic = np.full(len(src), 0.5, dtype=np.float32)
    return GraphInfo(num_nodes=6, edge_index=ei, ic_probs=ic, directed=True)


@pytest.mark.parametrize("name", list(A.ALGORITHMS.keys()))
def test_algo_returns_k_valid_distinct_seeds(name):
    g = _hub_graph()
    seeds = A.ALGORITHMS[name](g, budget=3, diffusion_model="IC")
    assert len(seeds) == 3, name
    assert len(set(seeds)) == 3, name
    assert all(0 <= s < g.num_nodes for s in seeds), name


def test_degree_picks_hub_first():
    g = _hub_graph()
    assert 0 in A.high_degree(g, budget=1, diffusion_model="IC")


def test_pagerank_seeds_picks_hub():
    g = _hub_graph()
    assert 0 in A.pagerank_seeds(g, budget=1, diffusion_model="IC")
