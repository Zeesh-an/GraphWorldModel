"""
Rediscovery distance: how far the winning program's OUTPUT sits from every
library algorithm's on the same instance, and which library members its code
calls.

MIND measured that GDM's dismantling order correlates at 0.762 with a projection
of its own input features, and this repository already prints that number for
critical node detection (`degree_rank_spearman`). This is the same
self-measurement generalized to every task: after the search, every callable
member of the task's library pool is run on exactly the call the condition-1
baseline row uses (`run.canned_baseline_script`), its output is compared with
the winner's, and the nearest member is named beside the similarity. Read with
the referee reward of that member (the report joins the two): high similarity
and no gain is a rediscovery, high similarity and a gain is a refined variant,
low similarity and a gain is a discovery in the strict sense. A library member
that fails or times out is recorded, never raised; a finished search is not
lost to a slow classical baseline.
"""

import re
import time
from typing import Callable
import numpy as np

from coding_agent.executor import (
    _time_limit,
    build_strategy,
    mc_blocked_algorithms,
)
from coding_agent.localization import instance_budget
from coding_agent.tools.blocking_algorithms import emittable as blocking_emittable
from coding_agent.tools.blocking_algorithms import mc_blocking_algorithms
from coding_agent.tools.dismantling_algorithms import mc_dismantling_algorithms
from coding_agent.tools.immunization_algorithms import (
    emittable as immunization_emittable,
)
from coding_agent.tools.immunization_algorithms import mc_immunization_algorithms
from coding_agent.tools.library_api import (
    algorithm_names,
    blocking_names,
    dismantling_names,
    immunization_names,
    localization_names,
    prediction_names,
    reconstruction_names,
)
from coding_agent.tools.localization_algorithms import mc_localization_algorithms
from coding_agent.tools.prediction_algorithms import mc_prediction_algorithms
from coding_agent.tools.reconstruction_algorithms import mc_reconstruction_algorithms
from coding_agent.types import GraphInfo, TaskSpec, Trajectory

# Seconds one library member may take before it is recorded as timed out
default_timeout_seconds = 60.0
# Selection instances an inverse-family or forecast comparison runs over
max_instances = 5
# Common items a rank correlation needs before it is reported
min_common_for_rank = 3
# A forecast counts as agreeing when the two predictions are within this ratio
forecast_agreement_ratio = 1.10


def library_pool(task: TaskSpec, lever: str | None = None) -> list[str]:
    """
    The callable library members a winner on this task is compared against: the
    task's pool minus the simulation-based members (blocked from generated code,
    so a winner cannot be one of them) and minus what the lever cannot emit.
    """
    if task.adaptive:
        # Per-round policies have no output without a rollout; the adaptivity
        # gap table already compares them on the referee
        return []
    if task.forecasts:
        names, blocked = prediction_names, mc_prediction_algorithms
    elif task.decodes:
        names, blocked = reconstruction_names, mc_reconstruction_algorithms
    elif task.recovers:
        names, blocked = localization_names, mc_localization_algorithms
    elif task.blocks:
        names = [name for name in blocking_names if lever is None or blocking_emittable(name, lever)]
        blocked = mc_blocking_algorithms
    elif task.immunizes:
        names = [
            name for name in immunization_names if lever is None or immunization_emittable(name, lever)
        ]
        blocked = mc_immunization_algorithms
    elif task.contains:
        names, blocked = dismantling_names, mc_dismantling_algorithms
    else:
        names, blocked = algorithm_names, mc_blocked_algorithms

    return [name for name in names if name not in blocked]


def static_calls(script: str, names: list[str]) -> list[str]:
    """Library members the program's source calls by name, in first-call order."""
    found = []
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", script or ""):
        name = match.group(1)
        if name in names and name not in found:
            found.append(name)

    return found


def budgeted_items(plan: list, task: TaskSpec) -> list:
    """What the plan spends its budget on, in plan order: node ids, or arcs on an edge lever."""
    items = []
    for bag in plan:
        for action in bag:
            if action.op != task.budget_op:
                continue
            item = (
                int(action.target)
                if action.destination is None
                else (int(action.target), int(action.destination))
            )
            if item not in items:
                items.append(item)

    return items


def jaccard(a: list, b: list) -> float:
    left, right = set(a), set(b)
    union = left | right
    if not union:
        return 1.0

    return len(left & right) / len(union)


def order_correlation(a: list, b: list) -> float | None:
    """Spearman correlation of the positions of the items both orders contain."""
    common = [item for item in a if item in set(b)]
    if len(common) < min_common_for_rank:
        return None

    position_a = {item: index for index, item in enumerate(a)}
    position_b = {item: index for index, item in enumerate(b)}
    ranks_a = np.asarray([position_a[item] for item in common], dtype=np.float64)
    ranks_b = np.asarray([position_b[item] for item in common], dtype=np.float64)
    if ranks_a.std() == 0 or ranks_b.std() == 0:
        return None

    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def event_f1(a: dict, b: dict) -> float:
    """F1 between two decoded histories' (node, time) events."""
    events_a = {(int(node), int(entry[0])) for node, entry in a.items()}
    events_b = {(int(node), int(entry[0])) for node, entry in b.items()}
    if not events_a and not events_b:
        return 1.0

    hit = len(events_a & events_b)
    if hit == 0:
        return 0.0
    precision = hit / len(events_a)
    recall = hit / len(events_b)

    return 2 * precision * recall / (precision + recall)


def forecast_similarity(a: list, b: list) -> dict:
    """Agreement of two predictors over the cascades both scored."""
    pairs = [
        (float(x), float(y))
        for x, y in zip(a, b, strict=True)
        if x is not None and y is not None
    ]
    if not pairs:
        return {"similarity": 0.0, "n": 0}

    xs = np.log1p(np.asarray([x for x, _ in pairs]))
    ys = np.log1p(np.asarray([y for _, y in pairs]))
    male = float(np.abs(xs - ys).mean())
    within = float(
        np.mean(
            [
                max(x, y) <= forecast_agreement_ratio * max(min(x, y), 1e-9)
                for x, y in pairs
            ]
        )
    )
    pearson = (
        float(np.corrcoef(xs, ys)[0, 1])
        if len(pairs) >= min_common_for_rank and xs.std() > 0 and ys.std() > 0
        else None
    )

    # One number in (0, 1] for the nearest-member ranking: 1 at identical
    # predictions, falling with the mean absolute log error between them
    return {
        "similarity": 1.0 / (1.0 + male),
        "male": male,
        "within_10pct": within,
        "pearson_log": pearson,
        "n": len(pairs),
    }


def _member_output(
    strategy: object,
    task: TaskSpec,
    graph: GraphInfo,
    instances: list,
    timeout: float,
) -> object:
    """The member's output in the family's shape, under its own time cap."""
    with _time_limit(timeout):
        if task.forecasts:
            strategy.fit_examples = list(instances)
            return [
                strategy.predict(graph, instance.observation, instance.observation.horizon)
                for instance in instances
            ]
        if task.decodes:
            return [
                strategy.reconstruct(graph, instance.observation, instance.horizon)
                for instance in instances
            ]
        if task.recovers:
            return [
                [
                    int(node)
                    for node in strategy.localize(
                        graph,
                        instance.observation.copy(),
                        instance_budget(instance, task, task.source_budget_mode),
                    )
                ]
                for instance in instances
            ]

        return strategy.plan_horizon(graph, task.budget, task.horizon)


def _winner_output(trajectory: Trajectory, task: TaskSpec, count: int) -> object:
    rows = (trajectory.cost.get("per_instance") or [])[:count]
    if task.forecasts:
        return [row.get("predicted") for row in rows]
    if task.decodes:
        return [row.get("decoded") or {} for row in rows]
    if task.recovers:
        return [[int(node) for node in row.get("predicted", [])] for row in rows]

    return list(trajectory.actions)


def _compare(winner: object, member: object, task: TaskSpec) -> dict:
    """The similarity block for one member, keyed by the family's metric."""
    if task.forecasts:
        return forecast_similarity(list(winner), list(member))
    if task.decodes:
        scores = [
            event_f1(w, {int(node): entry for node, entry in m.items()})
            for w, m in zip(winner, member, strict=True)
        ]
        nodes = [
            jaccard(list(w.keys()), [int(node) for node in m.keys()])
            for w, m in zip(winner, member, strict=True)
        ]
        return {
            "similarity": float(np.mean(scores)) if scores else 0.0,
            "event_f1": float(np.mean(scores)) if scores else 0.0,
            "node_jaccard": float(np.mean(nodes)) if nodes else 0.0,
            "n": len(scores),
        }
    if task.recovers:
        scores = [jaccard(w, m) for w, m in zip(winner, member, strict=True)]
        return {
            "similarity": float(np.mean(scores)) if scores else 0.0,
            "jaccard": float(np.mean(scores)) if scores else 0.0,
            "n": len(scores),
        }

    mine = budgeted_items(winner, task)
    theirs = budgeted_items(member, task)
    return {
        "similarity": jaccard(mine, theirs),
        "jaccard": jaccard(mine, theirs),
        "common": len(set(mine) & set(theirs)),
        "order_spearman": order_correlation(mine, theirs),
    }


def compute_provenance(
    task: TaskSpec,
    graph: GraphInfo,
    trajectory: Trajectory,
    script: str,
    script_for: Callable[[str], str],
    lever: str | None = None,
    timeout: float = default_timeout_seconds,
    allow_mc_algorithms: bool = False,
) -> dict:
    """
    The provenance block for one winner: every pool member's similarity, the
    nearest, the members the source calls, and the members that failed.
    """
    pool = library_pool(task, lever)
    instances = list(task.instances)[:max_instances]
    winner = _winner_output(trajectory, task, len(instances))
    ranking, errors = [], {}
    start = time.perf_counter()

    for name in pool:
        member_start = time.perf_counter()
        try:
            strategy = build_strategy(
                script_for(name), "free", allow_mc_algorithms=True, canned=True
            )
            output = _member_output(strategy, task, graph, instances, timeout)
            block = _compare(winner, output, task)
        # A library member that fails, times out or returns the wrong shape is a
        # data point of the provenance table, never a failure of the finished
        # search; the message is recorded verbatim for the report
        except Exception as error:
            errors[name] = f"{type(error).__name__}: {str(error).splitlines()[0][:200]}"
            continue
        block["name"] = name
        block["seconds"] = round(time.perf_counter() - member_start, 3)
        ranking.append(block)

    ranking.sort(key=lambda block: -float(block["similarity"]))
    nearest = ranking[0] if ranking else None

    return {
        "metric": (
            "forecast agreement (1 / (1 + MALE))"
            if task.forecasts
            else "event F1"
            if task.decodes
            else "source-set Jaccard"
            if task.recovers
            else f"Jaccard of the {task.budget_op} targets"
        ),
        "nearest": None if nearest is None else nearest["name"],
        "nearest_similarity": None if nearest is None else float(nearest["similarity"]),
        "ranking": ranking,
        "calls": static_calls(script, pool + list(_blocked_for(task))),
        "errors": errors,
        "n_instances": len(instances) if (task.forecasts or task.decodes or task.recovers) else None,
        "pool_size": len(pool),
        "seconds": round(time.perf_counter() - start, 3),
        "timeout_seconds": timeout,
        "skipped": (
            "adaptive policies have no output without a rollout" if task.adaptive else None
        ),
    }


def _blocked_for(task: TaskSpec) -> tuple:
    if task.forecasts:
        return mc_prediction_algorithms
    if task.decodes:
        return mc_reconstruction_algorithms
    if task.recovers:
        return mc_localization_algorithms
    if task.blocks:
        return mc_blocking_algorithms
    if task.immunizes:
        return mc_immunization_algorithms
    if task.contains:
        return mc_dismantling_algorithms

    return mc_blocked_algorithms

