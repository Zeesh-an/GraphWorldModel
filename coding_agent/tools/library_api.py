"""
Introspect the tools library into a human/agent-readable API reference string,
injected into the coding-agent prompts so the model sees the exact callable surface.
"""

import inspect
from typing import Callable

from coding_agent.tools import (
    adaptive_algorithms,
    algorithms,
    blocking_algorithms,
    dismantling_algorithms,
    localization_algorithms,
    primitives,
    reconstruction_algorithms,
)

algorithm_names = list(algorithms.algorithms.keys())
adaptive_names = list(adaptive_algorithms.adaptive_algorithms.keys())
dismantling_names = list(dismantling_algorithms.dismantling_algorithms.keys())
localization_names = list(localization_algorithms.localization_algorithms.keys())
reconstruction_names = list(
    reconstruction_algorithms.reconstruction_algorithms.keys()
)
blocking_names = list(blocking_algorithms.all_blocking_algorithms.keys())

# Which library members a given lever can actually spend its budget on. Showing an
# edge selector to a counter-seeding arm is not merely noise: its output is arcs, so
# a program that composed one would emit an op the executor rejects, and the search
# would spend an iteration on a repair turn for a menu we handed it.
blocking_lever_of_op = {
    "add_node": "counter_seed",
    "remove_node": "node_block",
    "remove_edge": "edge_block",
    "set_edge_weight": "weight_block",
}


def blocking_names_for(budget_op: str) -> list[str]:
    """Library members whose output the given lever can emit."""
    lever = blocking_lever_of_op.get(budget_op, "counter_seed")

    return [
        name
        for name in blocking_names
        if blocking_algorithms.emittable(name, lever)
    ]

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


def build_blocking_menu(budget_op: str = "add_node") -> str:
    """Return one `- name: summary` line per blocking algorithm for this lever."""
    lines = []
    for name in blocking_names_for(budget_op):
        doc_lines = (
            blocking_algorithms.all_blocking_algorithms[name].__doc__ or ""
        ).strip().splitlines()
        lines.append(f"- {name}: {doc_lines[0] if doc_lines else ''}")

    return "\n".join(lines)


def build_blocking_reference(budget_op: str = "add_node", exclude: tuple = ()) -> str:
    """The blocking surface for ONE lever, shown only to an influence-blocking prompt."""
    names = blocking_names_for(budget_op)
    lines = [
        _signature_line(blocking_algorithms.all_blocking_algorithms[name])
        for name in names
        if name not in exclude
    ]
    returns = (
        "a list of `(u, v)` ARCS"
        if budget_op in ("remove_edge", "set_edge_weight")
        else "a list of node ids"
    )
    blocked_note = (
        "\n  NOT AVAILABLE (calling one raises): "
        + ", ".join(name for name in names if name in exclude)
        + " — it simulates the whole competitive cascade for every candidate node,\n"
        "  which bypasses the metered evaluator."
        if any(name in exclude for name in names)
        else ""
    )

    return (
        "BLOCKING ALGORITHMS  (from coding_agent.tools.blocking_algorithms, imported\n"
        "as `blocking_algorithms`). Each takes `negative_seeds=` — the rumour's own\n"
        f"seed set, which is also `self.outbreak` — and returns {returns} of length\n"
        "`budget`. These are the published baselines you are being compared against:\n"
        "call one, beat one, or take its idea and improve on it. `proximity` is the\n"
        "one that actually has to be beaten — it is 'seed the rumour's own\n"
        "out-neighbours' and it outscores every learned method except StratLearner on\n"
        "two of that paper's three graphs, while `degree_blocking` is on the list\n"
        "because the published finding is that plain degree FAILS at this task:\n"
        + "\n".join(lines)
        + blocked_note
    )


def build_localization_menu() -> str:
    """Return one `- name: summary` line per localization algorithm (routing prompt)."""
    lines = []
    for name in localization_names:
        doc_lines = (
            localization_algorithms.localization_algorithms[name].__doc__ or ""
        ).strip().splitlines()
        lines.append(f"- {name}: {doc_lines[0] if doc_lines else ''}")

    return "\n".join(lines)


def build_localization_reference(exclude: tuple = ()) -> str:
    """The source-SET inference surface, shown only to an inverse task's prompt."""
    lines = [
        _signature_line(localization_algorithms.localization_algorithms[name])
        for name in localization_names
        if name not in exclude
    ]

    blocked_note = (
        "\n  NOT AVAILABLE (calling one raises): "
        + ", ".join(name for name in localization_names if name in exclude)
        + " — it re-simulates every candidate on a PRIVATE simulator, which\n"
        "  bypasses the metered evaluator. You already have the metered version:\n"
        "  `self.predict_marginals(seeds)`."
        if exclude
        else ""
    )

    return (
        "SOURCE-LOCALIZATION ALGORITHMS  (from\n"
        "coding_agent.tools.localization_algorithms, imported as\n"
        "`localization_algorithms`). Each returns a list of `budget` node ids it\n"
        "believes STARTED the observed cascade. These are the published baselines\n"
        "you are being compared against: call one, beat one, or take its idea and\n"
        "improve on it. `lpsi` is the one that actually has to be beaten — it is a\n"
        "2017 label-propagation method with NO learning and it beats both SL-VAE\n"
        "and DDMSL on real cascades:\n" + "\n".join(lines) + blocked_note + "\n\n"
        "PER-NODE SCORERS  (imported as `localization_scorers`, same names). Each\n"
        "returns a float vector of length num_nodes, higher meaning more likely to\n"
        "be a source. Return one of these (or your own) from source_scores() and\n"
        "your AUC is measured on the real ranking instead of a rank-derived\n"
        "stand-in:\n  "
        + ", ".join(f"localization_scorers.{name}" for name in localization_names)
    )


def build_reconstruction_menu() -> str:
    """Return one `- name: summary` line per decoder (routing prompt)."""
    lines = []
    for name in reconstruction_names:
        doc_lines = (
            reconstruction_algorithms.reconstruction_algorithms[name].__doc__ or ""
        ).strip().splitlines()
        lines.append(f"- {name}: {doc_lines[0] if doc_lines else ''}")

    return "\n".join(lines)


def build_reconstruction_reference(exclude: tuple = ()) -> str:
    """The TRAJECTORY-decoder surface, shown only to a reconstruction task's prompt."""
    lines = [
        _signature_line(reconstruction_algorithms.reconstruction_algorithms[name])
        for name in reconstruction_names
        if name not in exclude
    ]

    blocked_note = (
        "\n  NOT AVAILABLE (calling one raises): "
        + ", ".join(name for name in reconstruction_names if name in exclude)
        + " — each evaluates the transition kernel thousands of times per\n"
        "  instance. You already have the metered kernel:\n"
        "  `self.step_marginals(infected, frontier)`."
        if any(name in exclude for name in reconstruction_names)
        else ""
    )

    return (
        "CASCADE-RECONSTRUCTION DECODERS  (from\n"
        "coding_agent.tools.reconstruction_algorithms, imported as\n"
        "`reconstruction_algorithms`). Each returns "
        "`{node: (activation timestep, parent)}`\n"
        "for the nodes it believes were infected. These are the published "
        "baselines\nyou are being compared against: call one, beat one, or take its "
        "idea and\nimprove on it. `delayed_bfs` is the one that actually has to be "
        "beaten — it is\nan O(m + k log k) ordered-Steiner heuristic with no learning "
        "in it —\nand `personalized_pagerank` is on the list because the published "
        "finding is\nthat a plain random walker BEATS tree sampling on assortative "
        "graphs:\n" + "\n".join(lines) + blocked_note
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
