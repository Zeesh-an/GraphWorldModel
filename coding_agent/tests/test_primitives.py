# coding_agent/tests/test_primitives.py
import numpy as np
from coding_agent.types import GraphInfo
from coding_agent.tools import primitives as P


def _star() -> GraphInfo:
    # hub 0 -> {1,2,3,4}; undirected-style mirrored edges so 0 has high degree
    src = [0, 0, 0, 0, 1, 2, 3, 4]
    dst = [1, 2, 3, 4, 0, 0, 0, 0]
    ei = np.array([src, dst], dtype=np.int64)
    ic = np.full(len(src), 0.5, dtype=np.float32)
    return GraphInfo(num_nodes=5, edge_index=ei, ic_probs=ic, directed=True)


def test_compute_degree_hub_is_max():
    g = _star()
    deg = P.compute_degree(g)
    assert int(np.argmax(deg)) == 0


def test_get_top_degree_nodes():
    g = _star()
    assert P.get_top_degree_nodes(g, 1) == [0]
    assert len(P.get_top_degree_nodes(g, 3)) == 3


def test_compute_pagerank_hub_is_top():
    g = _star()
    pr = P.compute_pagerank(g)
    assert pr.shape == (5,) and int(np.argmax(pr)) == 0


def test_weighted_degree_matches_prob_sum():
    g = _star()
    wd = P.compute_weighted_degree(g)
    assert np.isclose(wd[0], 0.5 * 4)  # four out-edges of prob 0.5


def test_mc_simulate_spread_monotone_in_seeds():
    g = _star()
    s1 = P.mc_simulate_spread(g, [0], "IC", mc_runs=20, horizon=5)
    s0 = P.mc_simulate_spread(g, [], "IC", mc_runs=20, horizon=5)
    assert s1 >= s0 and s1 >= 1.0


def test_marginal_gain_nonnegative_ic():
    g = _star()
    gain = P.compute_marginal_gain(g, [], 0, "IC", mc_runs=20, horizon=5)
    assert gain >= 1.0  # seeding the hub activates at least itself


def test_batch_reverse_sample_returns_rr_sets():
    g = _star()
    rr = P.batch_reverse_sample(g, theta=50, seed=0)
    assert len(rr) == 50 and all(isinstance(s, set) for s in rr)


def test_detect_and_allocate_budget():
    g = _star()
    comm = P.detect_communities(g)
    assert set(comm.keys()) == set(range(5))
    alloc = P.allocate_budget(comm, 4)
    assert sum(alloc.values()) == 4


def test_estimate_sample_size_positive_and_monotone():
    g = _star()
    assert P.estimate_sample_size(g, 1) >= 200
    assert P.estimate_sample_size(g, 5) >= P.estimate_sample_size(g, 1)


def test_sample_live_edge_graph_and_reachable():
    g = _star()
    rng = np.random.default_rng(0)
    live = P.sample_live_edge_graph(g, rng)
    assert set(live.keys()) == set(range(5))
    assert P.reachable_count(live, [0]) >= 1


def test_path_influence_hub_is_top():
    g = _star()
    scores = P.path_influence_scores(g, max_hops=2)
    assert scores.shape == (5,) and int(np.argmax(scores)) == 0
