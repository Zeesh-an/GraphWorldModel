"""
Counterexample-guided feedback over realizations.

Formal synthesis refines a program against the inputs it failed on
(Solar-Lezama's counterexample-guided loop). Here the objective is stochastic,
so the "input" a candidate fails on is a realization: the same coin flips scored
both the candidate and the incumbent, and the samples where the candidate lost
most are the concrete cases the next edit should fix. They are described in
the task's own terms (which seeds produced nothing, which nodes the incumbent
reached and the candidate did not, which instance a localizer or decoder
explained worst) and shown to the model; acceptance still reads the fresh
realization of the next generation, so the counterexamples inform the edit
without biasing the estimate.
"""

import numpy as np

from coding_agent.types import GraphInfo, TaskSpec, Trajectory, minimize

# Realizations listed per generation, and node ids named per realization
max_counterexamples = 3
max_named_nodes = 6


def _worst_samples(
    candidate: Trajectory, incumbent: Trajectory, sense: str, count: int
) -> list[tuple[int, float, float]]:
    """(sample, candidate reward, incumbent reward) for the samples the candidate lost most."""
    a = candidate.sample_rewards
    b = incumbent.sample_rewards
    if a is None or b is None or len(a) != len(b) or not a:
        return []

    sign = -1.0 if sense == minimize else 1.0
    losses = [
        (sign * (float(b[index]) - float(a[index])), index) for index in range(len(a))
    ]
    losses.sort(reverse=True)

    return [
        (index, float(a[index]), float(b[index])) for loss, index in losses[:count] if loss > 0
    ]


def _node_list(nodes: list[int], graph: GraphInfo, communities: dict | None) -> str:
    ranked = sorted(nodes, key=lambda node: -graph.degree(node))[:max_named_nodes]
    parts = []
    for node in ranked:
        community = (
            f", c{communities[node]}" if communities is not None and node in communities else ""
        )
        parts.append(f"{node}(d={graph.degree(node)}{community})")

    return ", ".join(parts)


def describe_counterexamples(
    candidate: Trajectory,
    incumbent: Trajectory,
    candidate_sets: list | None,
    incumbent_sets: list | None,
    graph: GraphInfo | None,
    task: TaskSpec,
    communities: dict | None = None,
) -> str | None:
    """
    The feedback block, or None when the two trajectories carry no aligned
    samples (the multi-round union, or a legacy checkpoint).
    """
    worst = _worst_samples(candidate, incumbent, task.sense, max_counterexamples)
    if not worst:
        return None

    if task.recovers or task.decodes or task.forecasts:
        return _describe_instances(candidate, incumbent, worst, task)

    unit = task.reward_unit
    lines = [
        "COUNTEREXAMPLES (the realizations where this candidate lost most to the "
        "incumbent; both were scored on the SAME coin flips, so the difference is "
        "the programs', not the dice):"
    ]
    seeds = sorted(
        {
            int(action.target)
            for bag in candidate.actions[:1]
            for action in bag
            if action.op == task.budget_op
        }
    )

    for sample, own, theirs in worst:
        line = f"  realization {sample}: yours {own:.0f} vs incumbent {theirs:.0f} {unit}"
        if (
            candidate_sets is not None
            and incumbent_sets is not None
            and sample < len(candidate_sets)
            and sample < len(incumbent_sets)
            and graph is not None
        ):
            mine = candidate_sets[sample]
            other = incumbent_sets[sample]
            if task.contains:
                # Under containment the loss is what the cascade reached on the
                # candidate's watch and not on the incumbent's
                broke_through = sorted(mine - other)
                if broke_through:
                    line += (
                        f"; the outbreak reached {len(broke_through)} nodes here that the "
                        f"incumbent kept clean: {_node_list(broke_through, graph, communities)}"
                    )
            else:
                missed = sorted(other - mine)
                if missed:
                    line += (
                        f"; the incumbent reached {len(missed)} nodes you did not: "
                        f"{_node_list(missed, graph, communities)}"
                    )
                dead = [
                    seed
                    for seed in seeds
                    if not any(int(neighbour) in mine for neighbour in graph.out_neighbors(seed))
                ]
                if dead:
                    line += f"; your seeds {dead} produced no new infections in this realization"
        lines.append(line)

    return "\n".join(lines)


def _describe_instances(
    candidate: Trajectory,
    incumbent: Trajectory,
    worst: list[tuple[int, float, float]],
    task: TaskSpec,
) -> str:
    """The inverse and forecast families: the counterexample is an instance."""
    own_rows = candidate.cost.get("per_instance") or []
    their_rows = incumbent.cost.get("per_instance") or []
    label = "cascade" if task.forecasts else "episode"
    lines = [
        f"COUNTEREXAMPLES (the {label}s where this candidate scored worst against "
        "the incumbent; both were scored on the same instances):"
    ]

    for index, own, theirs in worst:
        row = own_rows[index] if index < len(own_rows) else {}
        other = their_rows[index] if index < len(their_rows) else {}
        name = row.get("episode_id", row.get("cascade_id", index))
        line = f"  {label} {name}: yours {own:.4f} vs incumbent {theirs:.4f}"
        if task.recovers:
            over = [entry[0] if isinstance(entry, (list, tuple)) else entry for entry in row.get("over", [])][:max_named_nodes]
            under = [entry[0] if isinstance(entry, (list, tuple)) else entry for entry in row.get("under", [])][:max_named_nodes]
            if over:
                line += f"; your sources over-explain {over}"
            if under:
                line += f"; they under-explain {under}"
            if other.get("predicted") is not None:
                line += f"; the incumbent named {sorted(other['predicted'])[:max_named_nodes]}"
        elif task.decodes:
            weakest = row.get("weakest", [])[:max_named_nodes]
            silent = row.get("silent", [])[:max_named_nodes]
            if weakest:
                line += f"; least probable asserted transmissions {weakest}"
            if silent:
                line += f"; activations the kernel expected that you left out {silent}"
        elif task.forecasts:
            if row.get("predicted") is not None:
                line += (
                    f"; you predicted {row['predicted']} against actual {row.get('actual')} "
                    f"(observed {row.get('observed')})"
                )
            if other.get("predicted") is not None:
                line += f"; the incumbent predicted {other['predicted']}"
        lines.append(line)

    return "\n".join(lines)


def sample_sets(environment: object) -> list | None:
    """The last rollout's per-sample final sets, when the evaluator exposes them."""
    sets = getattr(environment, "last_sample_final_infected", None)
    if not sets:
        return None

    return list(sets)


def realization_gap(candidate: Trajectory, incumbent: Trajectory) -> float | None:
    """Fraction of aligned samples the candidate lost, for the results JSON."""
    a = candidate.sample_rewards
    b = incumbent.sample_rewards
    if a is None or b is None or len(a) != len(b) or not a:
        return None

    return float(np.mean(np.asarray(a) < np.asarray(b)))
