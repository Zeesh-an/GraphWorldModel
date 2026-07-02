"""Introspect the tools library into a human/agent-readable API reference string,
injected into the coding-agent prompts so the model sees the exact callable surface.
"""

from __future__ import annotations

import inspect

from coding_agent.tools import algorithms as A
from coding_agent.tools import primitives as P

ALGORITHM_NAMES = list(A.ALGORITHMS.keys())

# Primitives we advertise to the agent (pure, safe to call).
_PRIMITIVES = [
    P.compute_degree, P.compute_out_degree, P.compute_weighted_degree,
    P.get_top_degree_nodes, P.compute_pagerank, P.compute_centrality,
    P.mc_simulate_spread, P.compute_marginal_gain, P.batch_reverse_sample,
    P.ris_select, P.detect_communities, P.allocate_budget,
    P.estimate_sample_size, P.sample_live_edge_graph, P.reachable_count,
    P.path_influence_scores,
]


def _sig_line(fn) -> str:
    sig = inspect.signature(fn)
    doc = (fn.__doc__ or "").strip().splitlines()
    summary = doc[0] if doc else ""
    return f"  {fn.__name__}{sig}\n      {summary}"


def build_api_reference() -> str:
    """Return a formatted reference of named algorithms + primitives."""
    algo_lines = [_sig_line(A.ALGORITHMS[name]) for name in ALGORITHM_NAMES]
    prim_lines = [_sig_line(fn) for fn in _PRIMITIVES]
    return (
        "NAMED ALGORITHMS  (from coding_agent.tools.algorithms, imported as `algorithms`)\n"
        + "\n".join(algo_lines)
        + "\n\nPRIMITIVES  (from coding_agent.tools.primitives, imported as `primitives`)\n"
        + "\n".join(prim_lines)
    )
