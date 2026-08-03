"""
Introspect the tools library into a human/agent-readable API reference string,
injected into the coding-agent prompts so the model sees the exact callable surface.
"""

import inspect
from typing import Callable

from coding_agent.tools import (
    adaptive_algorithms,
    algorithms,
    dismantling_algorithms,
    primitives,
)

algorithm_names = list(algorithms.algorithms.keys())
adaptive_names = list(adaptive_algorithms.adaptive_algorithms.keys())
dismantling_names = list(dismantling_algorithms.dismantling_algorithms.keys())

# Three implementations shown in full, one per idiom the library uses: neighbour
# discounting, per-community budget allocation, and RIS-then-refine. Signatures
# alone leave the model guessing at how a seed set is actually built here.
sourced_algorithms = ("degree_discount", "cofim", "degree_ris_refine")

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


def build_algorithm_sources(names: tuple = sourced_algorithms) -> str:
    """Return the full source of a few library algorithms, as worked examples."""
    blocks = "\n\n".join(
        f"```python\n{inspect.getsource(algorithms.algorithms[name]).rstrip()}\n```"
        for name in names
    )

    return (
        "LIBRARY SOURCE (how these are actually written — same primitives and the "
        "same GraphInfo you get):\n" + blocks
    )


def build_primitives_reference(exclude: tuple = ()) -> str:
    """Return the primitives signature list, minus any excluded names (scored mode)."""
    return "\n".join(
        _signature_line(function)
        for function in primitives_list
        if function.__name__ not in exclude
    )


def build_adaptive_reference() -> str:
    """The per-round policy surface, shown only to an adaptive task's prompt."""
    lines = [
        _signature_line(adaptive_algorithms.adaptive_algorithms[name])
        for name in adaptive_names
    ]

    return (
        "ADAPTIVE POLICIES  (from coding_agent.tools.adaptive_algorithms, imported\n"
        "as `adaptive_algorithms`). Each returns AT MOST `batch` seeds for ONE\n"
        "round, chosen against the state you were handed. These are the published\n"
        "baselines you are being compared against: call one, beat one, or take\n"
        "its idea and improve on it:\n" + "\n".join(lines)
    )


def build_dismantling_menu() -> str:
    """Return one `- name: summary` line per dismantling algorithm (routing prompt)."""
    lines = []
    for name in dismantling_names:
        doc_lines = (
            dismantling_algorithms.dismantling_algorithms[name].__doc__ or ""
        ).strip().splitlines()
        lines.append(f"- {name}: {doc_lines[0] if doc_lines else ''}")

    return "\n".join(lines)


def build_dismantling_reference(exclude: tuple = ()) -> str:
    """The node-REMOVAL surface, shown only to a containment task's prompt."""
    lines = [
        _signature_line(dismantling_algorithms.dismantling_algorithms[name])
        for name in dismantling_names
        if name not in exclude
    ]

    blocked_note = (
        "\n  NOT AVAILABLE (calling one raises): "
        + ", ".join(name for name in dismantling_names if name in exclude)
        + " — it simulates the contained cascade for every candidate node, which\n"
        "  bypasses the metered evaluator."
        if exclude
        else ""
    )

    return (
        "DISMANTLING ALGORITHMS  (from coding_agent.tools.dismantling_algorithms,\n"
        "imported as `dismantling_algorithms`). Each returns a list of nodes to\n"
        "REMOVE, of length `budget`. These are the published baselines you are being\n"
        "compared against: call one, beat one, or take its idea and improve on it.\n"
        "`adaptive_degree` is the one that actually has to be beaten:\n" + "\n".join(lines)
        + blocked_note
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
