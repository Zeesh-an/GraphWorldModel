# coding_agent/tests/test_library_api.py
from coding_agent.tools.library_api import build_api_reference, ALGORITHM_NAMES


def test_api_reference_lists_algorithms_and_primitives():
    ref = build_api_reference()
    assert "pagerank_seeds" in ref
    assert "compute_pagerank" in ref
    assert "mc_simulate_spread" in ref


def test_algorithm_names_match_registry():
    assert "celf" in ALGORITHM_NAMES and "high_degree" in ALGORITHM_NAMES
