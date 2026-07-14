"""
Introspect the tools library into a human/agent-readable API reference string,
injected into the coding-agent prompts so the model sees the exact callable surface.
"""

import inspect
from typing import Callable

from coding_agent.tools import algorithms, primitives

algorithm_names = list(algorithms.algorithms.keys())

# Primitives we advertise to the agent (pure, safe to call).
primitives_list = [
    primitives.compute_degree,
    primitives.compute_out_degree,
    primitives.compute_weighted_degree,
    primitives.get_top_degree_nodes,
    primitives.compute_pagerank,
    primitives.compute_centrality,
    primitives.mc_simulate_spread,
    primitives.compute_marginal_gain,
    primitives.batch_reverse_sample,
    primitives.ris_select,
    primitives.detect_communities,
    primitives.allocate_budget,
    primitives.estimate_sample_size,
    primitives.sample_live_edge_graph,
    primitives.reachable_count,
    primitives.path_influence_scores,
]


def _signature_line(function: Callable) -> str:
    signature = inspect.signature(function)
    doc_lines = (function.__doc__ or "").strip().splitlines()
    summary = doc_lines[0] if doc_lines else ""
    return f"  {function.__name__}{signature}\n      {summary}"


def build_api_reference() -> str:
    """Return a formatted reference of named algorithms + primitives."""
    algorithm_lines = [
        _signature_line(algorithms.algorithms[name]) for name in algorithm_names
    ]
    primitive_lines = [_signature_line(function) for function in primitives_list]

    return (
        "NAMED ALGORITHMS  (from coding_agent.tools.algorithms, imported as `algorithms`)\n"
        + "\n".join(algorithm_lines)
        + "\n\nPRIMITIVES  (from coding_agent.tools.primitives, imported as `primitives`)\n"
        + "\n".join(primitive_lines)
    )
