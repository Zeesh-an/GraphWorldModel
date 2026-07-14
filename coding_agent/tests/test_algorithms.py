import numpy as np
import pytest

from coding_agent.tools import algorithms
from coding_agent.types import GraphInfo


def _hub_graph() -> GraphInfo:
    sources = [0, 0, 0, 0, 0, 1, 2, 3, 4, 5]
    destinations = [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]
    edge_index = np.array([sources, destinations], dtype=np.int64)
    ic_probs = np.full(len(sources), 0.5, dtype=np.float32)
    return GraphInfo(
        num_nodes=6, edge_index=edge_index, ic_probs=ic_probs, directed=True
    )


@pytest.mark.parametrize("name", list(algorithms.algorithms.keys()))
def test_algo_returns_k_valid_distinct_seeds(name):
    graph = _hub_graph()
    seeds = algorithms.algorithms[name](graph, budget=3, diffusion_model="IC")
    assert len(seeds) == 3, name
    assert len(set(seeds)) == 3, name
    assert all(0 <= seed < graph.num_nodes for seed in seeds), name


def test_degree_picks_hub_first():
    graph = _hub_graph()
    assert 0 in algorithms.high_degree(graph, budget=1, diffusion_model="IC")


def test_pagerank_seeds_picks_hub():
    graph = _hub_graph()
    assert 0 in algorithms.pagerank_seeds(graph, budget=1, diffusion_model="IC")


def test_degree_discount_full_budget_stays_distinct():
    # budget == num_nodes exercises the chosen-node mask across every pick.
    graph = _hub_graph()
    seeds = algorithms.degree_discount(graph, budget=6, diffusion_model="IC")
    assert sorted(seeds) == list(range(6))
