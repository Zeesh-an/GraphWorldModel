"""
Introspect the tools library into a human/agent-readable API reference string,
injected into the coding-agent prompts so the model sees the exact callable surface.
"""

import inspect
from typing import Callable

from coding_agent.tools import algorithms, primitives

algorithm_names = list(algorithms.algorithms.keys())

# Primitives we advertise to the agent (pure, safe to call)
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


def build_algorithm_menu() -> str:
    """Return one `- name: summary` line per library algorithm (routing prompt)."""
    lines = []
    for name in algorithm_names:
        doc_lines = (algorithms.algorithms[name].__doc__ or "").strip().splitlines()
        summary = doc_lines[0] if doc_lines else ""
        lines.append(f"- {name}: {summary}")

    return "\n".join(lines)


def build_primitives_reference(exclude: tuple = ()) -> str:
    """Return the primitives signature list, minus any excluded names (scored mode)."""
    return "\n".join(
        _signature_line(function)
        for function in primitives_list
        if function.__name__ not in exclude
    )


def build_api_reference(exclude: tuple = ()) -> str:
    """Return a formatted reference of named algorithms + primitives."""
    available = [name for name in algorithm_names if name not in exclude]
    algorithm_lines = [
        _signature_line(algorithms.algorithms[name]) for name in available
    ]

    # Named explicitly rather than silently omitted: the model knows these
    # algorithms and will reach for one, and a listed reason costs a line
    # here versus a whole wasted refinement iteration
    blocked_note = (
        "\n\nNOT AVAILABLE (calling one raises): "
        + ", ".join(name for name in algorithm_names if name in exclude)
        + "\n  These simulate the cascade for every candidate node, which is far "
        "too slow and\n  bypasses the metered evaluator. Do not try to reimplement "
        "them either —\n  primitives.mc_simulate_spread over all nodes has the same "
        "cost."
        if exclude
        else ""
    )

    return (
        "NAMED ALGORITHMS  (from coding_agent.tools.algorithms, imported as `algorithms`)\n"
        + "\n".join(algorithm_lines)
        + blocked_note
        + "\n\nPRIMITIVES  (from coding_agent.tools.primitives, imported as `primitives`)\n"
        + build_primitives_reference()
    )
