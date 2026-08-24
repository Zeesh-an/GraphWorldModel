"""
Pareto fronts over the world-model / evaluator results.

Why this and not a leaderboard. The project's own taxonomy has two axes that trade
off against each other and no defensible way to weight them into one number:

- **fidelity** — how close the evaluator's answer is to the truth
  (`ens_marg_mae`, `ens_count_bias`, `budget_regret_norm`, …)
- **cost** — what the answer costs (`real_env_episodes`, seconds per rollout,
  forward passes)

An oracle evaluator wins every fidelity column and loses every cost column; a
one-episode native evaluator does the reverse. Ranking them by fidelity alone
silently assumes cost is free, which is the exact assumption the world model
exists to attack. The Pareto front is the honest summary: it reports which
configurations are not dominated, and — just as usefully — which ARE, with the
configuration that dominates them named.

Everything here is pure numpy/stdlib over already-computed result dicts, so it can
be run on a finished results tree without a GPU or a rerun.
"""

from dataclasses import dataclass, field

import numpy as np

# Objective sense. `minimize` covers error and cost columns, `maximize` covers
# spread and F1 columns. Every objective must state one — guessing from the name
# is how a sign error gets into a headline table.
minimize = "min"
maximize = "max"


@dataclass(frozen=True)
class Objective:
    key: str
    sense: str
    label: str = ""

    def better(self, first: float, second: float) -> bool:
        """Is `first` strictly better than `second` on this objective?"""
        return first < second if self.sense == minimize else first > second

    def at_least_as_good(self, first: float, second: float) -> bool:
        return first <= second if self.sense == minimize else first >= second


@dataclass
class Point:
    """One configuration and its objective values."""

    name: str
    values: dict[str, float]
    meta: dict = field(default_factory=dict)


# The fidelity/cost pairs the report cares about, named so a caller picks a
# vocabulary instead of assembling one and getting the senses wrong.
fidelity_objectives = {
    "ens_marg_mae": Objective("ens_marg_mae", minimize, "rollout marginal MAE"),
    "abs_count_bias": Objective("abs_count_bias", minimize, "|rollout count bias|"),
    "ens_count_w1": Objective("ens_count_w1", minimize, "rollout count W1"),
    "delta_f1": Objective("delta_f1", maximize, "one-step delta F1"),
    "brier_infected": Objective("brier_infected", minimize, "Brier (infected)"),
    "budget_regret_norm": Objective(
        "budget_regret_norm", minimize, "k-seed regret (fraction)"
    ),
}

cost_objectives = {
    "real_env_episodes": Objective(
        "real_env_episodes", minimize, "real-environment episodes"
    ),
    "evaluator_seconds": Objective("evaluator_seconds", minimize, "evaluator seconds"),
    "rollout_seconds": Objective("rollout_seconds", minimize, "seconds per rollout"),
    "train_seconds": Objective("train_seconds", minimize, "training seconds"),
    "forward_passes": Objective("forward_passes", minimize, "model forward passes"),
}


def dominates(
    first: Point, second: Point, objectives: list[Objective]
) -> bool:
    """Standard Pareto dominance: at least as good everywhere, strictly better somewhere.

    A point missing an objective never dominates on it: an absent number is not
    evidence of a good one. That makes partial result rows safe to pass in.
    """
    strictly_better = False

    for objective in objectives:
        first_value = first.values.get(objective.key)
        second_value = second.values.get(objective.key)

        if first_value is None or not np.isfinite(first_value):
            return False

        if second_value is None or not np.isfinite(second_value):
            # `second` has no number here, so `first` cannot be shown worse on it
            strictly_better = True
            continue

        if not objective.at_least_as_good(first_value, second_value):
            return False

        if objective.better(first_value, second_value):
            strictly_better = True

    return strictly_better


def pareto_front(
    points: list[Point], objectives: list[Objective]
) -> dict:
    """Split `points` into the non-dominated front and the dominated rest.

    Returns the front, and for every dominated point the configurations that
    dominate it — "worse than X on everything" is far more actionable in a report
    than "not on the front".
    """
    if not objectives:
        raise ValueError("pareto_front needs at least one objective")

    front, dominated = [], {}

    for candidate in points:
        dominators = [
            other.name
            for other in points
            if other is not candidate and dominates(other, candidate, objectives)
        ]

        if dominators:
            dominated[candidate.name] = dominators
        else:
            front.append(candidate.name)

    return {
        "objectives": [
            {"key": objective.key, "sense": objective.sense, "label": objective.label}
            for objective in objectives
        ],
        "front": front,
        "dominated": dominated,
        "n_points": len(points),
        "values": {point.name: point.values for point in points},
    }


def hypervolume_2d(
    points: list[Point], first: Objective, second: Objective, reference: tuple
) -> float:
    """Dominated area of the 2-D front against a reference point.

    Only defined for two objectives on purpose. In higher dimensions hypervolume
    depends on a reference point nobody can justify, and a single scalar re-imposes
    exactly the weighting the front exists to avoid. Two axes (one fidelity, one
    cost) is the case where it is a fair summary, so it is the only case offered.
    """
    def to_min(objective: Objective, value: float) -> float:
        return value if objective.sense == minimize else -value

    reference_first = to_min(first, reference[0])
    reference_second = to_min(second, reference[1])

    coordinates = sorted(
        (to_min(first, point.values[first.key]), to_min(second, point.values[second.key]))
        for point in points
        if first.key in point.values
        and second.key in point.values
        and np.isfinite(point.values[first.key])
        and np.isfinite(point.values[second.key])
    )

    area, previous_second = 0.0, reference_second

    for value_first, value_second in coordinates:
        if value_first >= reference_first or value_second >= previous_second:
            continue

        area += (reference_first - value_first) * (previous_second - value_second)
        previous_second = value_second

    return float(area)


def point_from_results(name: str, results: dict, meta: dict | None = None) -> Point:
    """Flatten one train_wm.py results JSON into a Pareto point.

    Pulls from the `test`, `rollout` and `planning_budget` blocks and derives
    `abs_count_bias`, because a count bias of -3 and one of +3 are equally
    unfaithful and the signed value would put over- and under-prediction on
    opposite ends of one axis.
    """
    test = results.get("test", {})
    rollout = results.get("rollout", {})
    budget = results.get("planning_budget", {})

    values = {
        "delta_f1": test.get("delta_f1"),
        "brier_infected": test.get("brier_infected"),
        "ens_marg_mae": rollout.get("ens_marg_mae"),
        "ens_count_w1": rollout.get("ens_count_w1"),
        "budget_regret_norm": budget.get("budget_regret_norm"),
        "train_seconds": results.get("train_seconds"),
    }

    if rollout.get("ens_count_bias") is not None:
        values["abs_count_bias"] = abs(rollout["ens_count_bias"])

    return Point(
        name=name,
        values={
            key: float(value) for key, value in values.items() if value is not None
        },
        meta=meta or {},
    )


def fidelity_cost_front(
    points: list[Point],
    fidelity: str = "ens_marg_mae",
    cost: str = "train_seconds",
) -> dict:
    """The headline two-axis front: one fidelity column against one cost column."""
    if fidelity not in fidelity_objectives:
        raise ValueError(
            f"unknown fidelity objective {fidelity!r}; "
            f"choose from {sorted(fidelity_objectives)}"
        )

    if cost not in cost_objectives:
        raise ValueError(
            f"unknown cost objective {cost!r}; choose from {sorted(cost_objectives)}"
        )

    return pareto_front(
        points, [fidelity_objectives[fidelity], cost_objectives[cost]]
    )
