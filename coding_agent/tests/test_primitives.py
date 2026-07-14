import numpy as np

from coding_agent.tools import primitives
from coding_agent.types import GraphInfo


def _star() -> GraphInfo:
    # Hub 0 -> {1,2,3,4}; mirrored edges so node 0 has high degree.
    sources = [0, 0, 0, 0, 1, 2, 3, 4]
    destinations = [1, 2, 3, 4, 0, 0, 0, 0]
    edge_index = np.array([sources, destinations], dtype=np.int64)
    ic_probs = np.full(len(sources), 0.5, dtype=np.float32)
    return GraphInfo(
        num_nodes=5, edge_index=edge_index, ic_probs=ic_probs, directed=True
    )


def test_compute_degree_hub_is_max():
    graph = _star()
    degrees = primitives.compute_degree(graph)
    assert int(np.argmax(degrees)) == 0


def test_get_top_degree_nodes():
    graph = _star()
    assert primitives.get_top_degree_nodes(graph, 1) == [0]
    assert len(primitives.get_top_degree_nodes(graph, 3)) == 3


def test_compute_pagerank_hub_is_top():
    graph = _star()
    pagerank_scores = primitives.compute_pagerank(graph)
    assert pagerank_scores.shape == (5,) and int(np.argmax(pagerank_scores)) == 0


def test_weighted_degree_matches_prob_sum():
    graph = _star()
    weighted_degrees = primitives.compute_weighted_degree(graph)
    assert np.isclose(weighted_degrees[0], 0.5 * 4)  # four out-edges of prob 0.5


def test_mc_simulate_spread_monotone_in_seeds():
    graph = _star()
    spread_with_seed = primitives.mc_simulate_spread(
        graph, [0], "IC", mc_runs=20, horizon=5
    )
    spread_without_seed = primitives.mc_simulate_spread(
        graph, [], "IC", mc_runs=20, horizon=5
    )
    assert spread_with_seed >= spread_without_seed and spread_with_seed >= 1.0


def test_marginal_gain_nonnegative_ic():
    graph = _star()
    gain = primitives.compute_marginal_gain(graph, [], 0, "IC", mc_runs=20, horizon=5)
    assert gain >= 1.0  # seeding the hub activates at least itself


def test_batch_reverse_sample_returns_rr_sets():
    graph = _star()
    rr_sets = primitives.batch_reverse_sample(graph, theta=50, seed=0)
    assert len(rr_sets) == 50 and all(isinstance(rr_set, set) for rr_set in rr_sets)


def test_detect_and_allocate_budget():
    graph = _star()
    communities = primitives.detect_communities(graph)
    assert set(communities.keys()) == set(range(5))
    allocation = primitives.allocate_budget(communities, 4)
    assert sum(allocation.values()) == 4


def test_estimate_sample_size_positive_and_monotone():
    graph = _star()
    assert primitives.estimate_sample_size(graph, 1) >= 200
    assert primitives.estimate_sample_size(graph, 5) >= primitives.estimate_sample_size(
        graph, 1
    )


def test_sample_live_edge_graph_and_reachable():
    graph = _star()
    rng = np.random.default_rng(0)
    live = primitives.sample_live_edge_graph(graph, rng)
    assert set(live.keys()) == set(range(5))
    assert primitives.reachable_count(live, [0]) >= 1


def test_path_influence_hub_is_top():
    graph = _star()
    scores = primitives.path_influence_scores(graph, max_hops=2)
    assert scores.shape == (5,) and int(np.argmax(scores)) == 0
