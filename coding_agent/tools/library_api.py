"""
Introspect the tools library into a human/agent-readable API reference string,
injected into the coding-agent prompts so the model sees the exact callable surface.
"""

import inspect

from coding_agent.tools import algorithms
from coding_agent.tools import primitives

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


def _sig_line(fn) -> str:
    sig = inspect.signature(fn)
    doc = (fn.__doc__ or "").strip().splitlines()
    summary = doc[0] if doc else ""
    return f"  {fn.__name__}{sig}\n      {summary}"


def build_api_reference() -> str:
    """Return a formatted reference of named algorithms + primitives."""
    algo_lines = [_sig_line(algorithms.algorithms[name]) for name in algorithm_names]
    prim_lines = [_sig_line(fn) for fn in primitives_list]

    return (
        "NAMED ALGORITHMS  (from coding_agent.tools.algorithms, imported as `algorithms`)\n"
        + "\n".join(algo_lines)
        + "\n\nPRIMITIVES  (from coding_agent.tools.primitives, imported as `primitives`)\n"
        + "\n".join(prim_lines)
    )
