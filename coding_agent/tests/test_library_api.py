# coding_agent/tests/test_library_api.py
from coding_agent.tools.library_api import algorithm_names, build_api_reference


def test_api_reference_lists_algorithms_and_primitives():
    ref = build_api_reference()
    assert "pagerank_seeds" in ref
    assert "compute_pagerank" in ref
    assert "mc_simulate_spread" in ref


def test_algorithm_names_match_registry():
    assert "celf" in algorithm_names and "high_degree" in algorithm_names
