"""
Paper figures from the JSONs the pipeline already wrote. Every figure is guarded
by the data it needs, so a partial run just produces fewer plots instead of failing.
"""

import os
from pathlib import Path
import matplotlib
import numpy as np

from pipeline.conditions import (
    adaptivity_gaps,
    condition_names,
    ground_truth_reward,
    is_ground_truth,
    is_reconstruct,
    is_recover,
    result_sense,
)
from pipeline.tasks import minimize

# Headless: SLURM nodes have no display
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

figure_dpi = 200
figure_size = (7.0, 4.5)
marker_cycle = ("o", "s", "^", "D", "v", "P", "X", "*")
condition_colors = {
    1: "#8C8C8C",
    2: "#DD8452",
    3: "#C44E52",
    4: "#CCB974",
    5: "#55A868",
    6: "#4C72B0",
    7: "#9370DB",
}


def _arm_style(index: int) -> dict:
    return {"marker": marker_cycle[index % len(marker_cycle)], "linewidth": 1.8}


def _sorted_arms(results: list[dict]) -> list[str]:
    """Condition order (1..6), then alphabetical — the paper's table order."""
    ranked = {}
    for result in results:
        ranked[result["arm"]] = (result.get("condition", 99), result["arm"])

    return sorted(ranked, key=ranked.get)


def _by_arm(results: list[dict], arm: str) -> list[dict]:
    return sorted(
        (result for result in results if result["arm"] == arm),
        key=lambda result: result["budget"],
    )


def _condition_of(results: list[dict], arm: str) -> int:
    return next(
        (result.get("condition", 99) for result in results if result["arm"] == arm), 99
    )


def _reward_se(result: dict) -> float:
    """SE of the number _ground_truth_reward returns, so error bars match the series."""
    if result.get("mc_reward") is not None:
        return float(result.get("mc_reward_se", 0.0) or 0.0)

    return float(result.get("cost", {}).get("reward_se", 0.0) or 0.0)


def _reward_label(results: list[dict], normalize: bool = False) -> str:
    """
    Y-axis text, including which direction is the good one.

    Stating it on the axis rather than only in the report: a containment figure
    with an unlabelled y-axis is read as a win for the highest curve, which is
    exactly backwards.
    """
    # A decoder's reward is a tree-weighted score in [0, 1] and an inverse task's
    # is an F1 in the same range; neither is a node count and neither needs a
    # referee, since both are measured against something we stored
    if is_reconstruct(results):
        return (
            "tree-weighted reconstruction score — higher is better "
            "(held-out cascades)"
        )

    if is_recover(results):
        return "F1 against the true sources — higher is better (held-out episodes)"

    unit = "% of nodes" if normalize else "nodes"
    judge = "ground-truth MC" if is_ground_truth(results) else "mixed evaluators"

    if result_sense(results) == minimize:
        return f"final infected ({unit}, {judge}) — LOWER IS BETTER"

    return f"final spread ({unit}, {judge})"


def _budget_label(results: list[dict], normalize: bool = False) -> str:
    unit = "% of nodes" if normalize else "k"
    noun = "removal" if result_sense(results) == minimize else "seed"

    return f"{noun} budget ({unit})" if normalize else f"{noun} budget {unit}"


def _spread_title(results: list[dict]) -> str:
    return (
        "contained spread vs removal budget"
        if result_sense(results) == minimize
        else "influence spread vs seed budget"
    )


def _save(figure: plt.Figure, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(figure)

    return path


def plot_budget_vs_spread(
    results: list[dict], out_path: Path, title_prefix: str, normalize: bool = False
) -> Path | None:
    if not results:
        return None

    figure, axes = plt.subplots(figsize=figure_size)

    for index, arm in enumerate(_sorted_arms(results)):
        runs = _by_arm(results, arm)
        if not runs:
            continue

        nodes = [run["graph"]["num_nodes"] for run in runs]
        x_values = [run["budget_pct"] if normalize else run["budget"] for run in runs]
        y_values = [
            100.0 * ground_truth_reward(run) / count if normalize
            else ground_truth_reward(run)
            for run, count in zip(runs, nodes, strict=True)
        ]
        errors = [
            100.0 * _reward_se(run) / count if normalize else _reward_se(run)
            for run, count in zip(runs, nodes, strict=True)
        ]

        axes.errorbar(
            x_values,
            y_values,
            yerr=errors,
            label=arm,
            capsize=3,
            color=condition_colors.get(_condition_of(results, arm)),
            **_arm_style(index),
        )

    axes.set_xlabel(_budget_label(results, normalize))
    axes.set_ylabel(_reward_label(results, normalize))
    axes.set_title(f"{title_prefix}: {_spread_title(results)}")
    axes.grid(alpha=0.3)
    axes.legend(fontsize=8)

    return _save(figure, out_path)


def plot_ours_vs_baselines(
    results: list[dict], out_path: Path, title_prefix: str, normalize: bool = False
) -> Path | None:
    """
    The paper's headline figure: our method against every baseline across the
    budget sweep.

    Ours (condition 6) is drawn heavy and solid; published external methods
    (condition 7) dashed; classical library algorithms (condition 1) thin and
    faded. All series use the ground-truth MC replay so the curves are
    commensurable.
    """
    if not results:
        return None

    ours = [result for result in results if result.get("condition") == 6]
    if not ours:
        # Without our own arm this is just budget_vs_spread under another name
        return None

    figure, axes = plt.subplots(figsize=(7.6, 4.8))

    for index, arm in enumerate(_sorted_arms(results)):
        runs = _by_arm(results, arm)
        if len(runs) < 1:
            continue

        condition = _condition_of(results, arm)
        nodes = [run["graph"]["num_nodes"] for run in runs]
        x_values = [run["budget_pct"] if normalize else run["budget"] for run in runs]
        y_values = [
            100.0 * ground_truth_reward(run) / count if normalize
            else ground_truth_reward(run)
            for run, count in zip(runs, nodes, strict=True)
        ]

        if condition == 6:
            style = {"linewidth": 3.0, "linestyle": "-", "zorder": 5, "alpha": 1.0}
            label = f"OURS — {arm}"
        elif condition == 7:
            style = {"linewidth": 2.0, "linestyle": "--", "zorder": 4, "alpha": 0.95}
            label = f"published — {arm.replace('external_', '')}"
        elif condition == 1:
            style = {"linewidth": 1.2, "linestyle": ":", "zorder": 2, "alpha": 0.7}
            label = f"classical — {arm.replace('baseline_', '')}"
        else:
            style = {"linewidth": 1.5, "linestyle": "-.", "zorder": 3, "alpha": 0.8}
            label = arm

        axes.plot(
            x_values,
            y_values,
            marker=marker_cycle[index % len(marker_cycle)],
            markersize=5,
            color=condition_colors.get(condition),
            label=label,
            **style,
        )

    axes.set_xlabel(_budget_label(results, normalize))
    axes.set_ylabel(_reward_label(results, normalize))
    axes.set_title(f"{title_prefix}: our method vs all baselines")
    axes.grid(alpha=0.3)
    axes.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    return _save(figure, out_path)


def plot_condition_comparison(
    results: list[dict], out_path: Path, title_prefix: str
) -> Path | None:
    """The baseline table as a figure: every arm at the largest budget, ground truth."""
    if not results:
        return None

    largest = max(result["budget"] for result in results)
    at_largest = [result for result in results if result["budget"] == largest]
    arms = _sorted_arms(at_largest)
    if len(arms) < 2:
        return None

    values, errors, colors = [], [], []
    for arm in arms:
        run = _by_arm(at_largest, arm)[0]
        values.append(ground_truth_reward(run))
        errors.append(_reward_se(run))
        colors.append(condition_colors.get(run.get("condition", 99), "#4C72B0"))

    figure, axes = plt.subplots(figsize=(max(8.0, 0.9 * len(arms)), 4.8))
    bars = axes.bar(range(len(arms)), values, yerr=errors, capsize=3, color=colors)

    for bar, value, error in zip(bars, values, errors, strict=True):
        axes.text(
            bar.get_x() + bar.get_width() / 2,
            value + error,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    axes.set_xticks(range(len(arms)))
    axes.set_xticklabels(arms, rotation=35, ha="right", fontsize=8)
    axes.set_ylim(0, max(v + e for v, e in zip(values, errors, strict=True)) * 1.15)
    axes.set_ylabel(_reward_label(at_largest))
    # Bars start at 0 either way, so the shortest bar is the winner under minimize
    axes.set_title(f"{title_prefix}: baseline conditions at k={largest}")
    axes.grid(alpha=0.3, axis="y")

    present = sorted({result.get("condition", 99) for result in at_largest})
    axes.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=condition_colors[condition])
            for condition in present
            if condition in condition_colors
        ],
        labels=[
            f"{condition}. {condition_names[condition]}"
            for condition in present
            if condition in condition_names
        ],
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )

    return _save(figure, out_path)


def plot_sample_efficiency(
    results: list[dict], out_path: Path, title_prefix: str
) -> Path | None:
    """
    Spread against real-environment episodes burned to get there.

    This is the axis the whole taxonomy hangs on: the native agent pays real
    episodes for every noisy evaluation, while the model-based conditions pay none.
    """
    largest = max((result["budget"] for result in results), default=None)
    if largest is None:
        return None

    at_largest = [result for result in results if result["budget"] == largest]
    if not any(result.get("real_env_episodes") for result in at_largest):
        return None

    figure, axes = plt.subplots(figsize=figure_size)

    for index, arm in enumerate(_sorted_arms(at_largest)):
        run = _by_arm(at_largest, arm)[0]
        # 0 real episodes cannot be drawn on a log axis; nudge the free arms apart
        # so oracle and world_model do not land on the exact same point
        episodes = run.get("real_env_episodes", 0) or (0.4 + 0.04 * index)

        axes.scatter(
            episodes,
            ground_truth_reward(run),
            s=70,
            color=condition_colors.get(run.get("condition", 99), "#4C72B0"),
            marker=marker_cycle[index % len(marker_cycle)],
            label=arm,
            zorder=3,
        )

    axes.set_xscale("log")
    axes.set_xlabel("real-environment episodes consumed (log; left edge = none)")
    axes.set_ylabel(_reward_label(at_largest))
    # Bars start at 0 either way, so the shortest bar is the winner under minimize
    axes.set_title(f"{title_prefix}: quality vs real-experience cost at k={largest}")
    axes.grid(alpha=0.3)
    axes.legend(fontsize=7)

    return _save(figure, out_path)


def plot_adaptivity_gap(
    results: list[dict], out_path: Path, title_prefix: str
) -> Path | None:
    """
    Gap and cost, side by side: the two halves of the adaptive-IM claim.

    Left: the spread ratio against budget, with 1.0 drawn in. Right: what each
    side spent inside its evaluator to produce that ratio. The right panel is the
    one the argument rests on; the left is there so a gap near 1 is visible as
    the predicted outcome rather than read as a null result.
    """
    paired = adaptivity_gaps(results)
    if not paired:
        return None

    figure, (gap_axes, cost_axes) = plt.subplots(1, 2, figsize=(11.0, 4.5))
    evaluators = sorted({entry["evaluator"] for entry in paired})

    for index, evaluator in enumerate(evaluators):
        rows = [entry for entry in paired if entry["evaluator"] == evaluator]
        budgets = [entry["budget"] for entry in rows]
        style = _arm_style(index)

        gap_axes.plot(
            budgets, [entry["gap"] for entry in rows], label=evaluator, **style
        )
        # Seconds inside the evaluator, adaptive over control: >1 means the
        # rounds cost this evaluator more, which is exactly what MC should show
        # and the world model should not
        cost_axes.plot(
            budgets,
            [
                (entry["adaptive_evaluator_seconds"] or 0.0)
                / (entry["control_evaluator_seconds"] or float("inf"))
                for entry in rows
            ],
            label=evaluator,
            **style,
        )

    gap_axes.axhline(1.0, color="#8C8C8C", linestyle="--", linewidth=1.0)
    gap_axes.set_xlabel("budget k")
    gap_axes.set_ylabel("adaptive spread / non-adaptive spread")
    gap_axes.set_title("adaptivity gap (theory caps myopic at 4)")

    cost_axes.axhline(1.0, color="#8C8C8C", linestyle="--", linewidth=1.0)
    cost_axes.set_xlabel("budget k")
    cost_axes.set_ylabel("adaptive evaluator seconds / non-adaptive")
    cost_axes.set_title("what the rounds cost each evaluator")

    for axes in (gap_axes, cost_axes):
        axes.grid(alpha=0.3)
        axes.legend(fontsize=8)

    figure.suptitle(f"{title_prefix}: adaptivity")

    return _save(figure, out_path)


def plot_round_spreads(
    results: list[dict], out_path: Path, title_prefix: str
) -> Path | None:
    """Realized spread after each round: where the budget actually paid off."""
    adaptive = [result for result in results if result.get("round_spreads")]
    if not adaptive:
        return None

    largest = max(result["budget"] for result in adaptive)
    at_largest = [result for result in adaptive if result["budget"] == largest]

    figure, axes = plt.subplots(figsize=figure_size)

    for index, run in enumerate(at_largest):
        spreads = run["round_spreads"]
        axes.plot(
            range(1, len(spreads) + 1),
            spreads,
            label=run["arm"],
            color=condition_colors.get(run.get("condition", 99), "#4C72B0"),
            **_arm_style(index),
        )

    axes.set_xlabel("round")
    axes.set_ylabel("infected after this round")
    axes.set_title(f"{title_prefix}: per-round spread at k={largest}")
    axes.grid(alpha=0.3)
    axes.legend(fontsize=7)

    return _save(figure, out_path)


def plot_evaluator_fidelity(
    results: list[dict], out_path: Path, title_prefix: str
) -> Path | None:
    """
    Model estimate vs the Monte Carlo referee — points off the diagonal are model error.

    Only the model-based conditions (oracle, world_model) belong here: a
    monte_carlo arm's estimate IS the simulator, so it sits on the diagonal by
    construction and would pad the plot with a meaningless perfect result.
    """
    paired = [
        result
        for result in results
        if result.get("mc_reward") is not None
        and result.get("evaluator") in ("oracle", "world_model")
    ]
    if not paired:
        return None

    figure, axes = plt.subplots(figsize=(5.2, 5.0))

    for index, arm in enumerate(_sorted_arms(paired)):
        runs = [result for result in paired if result["arm"] == arm]
        axes.scatter(
            [run["mc_reward"] for run in runs],
            # The unbiased estimate when it exists: `reward` carries the winner's curse
            [run.get("wm_reeval_mean", run["reward"]) for run in runs],
            label=arm,
            marker=marker_cycle[index % len(marker_cycle)],
            color=condition_colors.get(_condition_of(paired, arm)),
            alpha=0.85,
        )

    limits = [
        min(run["mc_reward"] for run in paired) * 0.95,
        max(run["mc_reward"] for run in paired) * 1.05,
    ]
    axes.plot(limits, limits, "k--", linewidth=1, label="perfect fidelity")

    axes.set_xlabel("Monte Carlo spread (ground truth)")
    axes.set_ylabel("model estimate of spread")
    axes.set_title(f"{title_prefix}: evaluator fidelity")
    axes.grid(alpha=0.3)
    axes.legend(fontsize=8)

    return _save(figure, out_path)


def plot_runtime(results: list[dict], out_path: Path, title_prefix: str) -> Path | None:
    """Per-rollout wall-clock by evaluator: the amortization claim for the model."""
    timed = [result for result in results if result.get("cost", {}).get("rollout_seconds")]
    if not timed:
        return None

    # One bar per evaluator, averaged over the arms that used it
    by_evaluator = {}
    for result in timed:
        by_evaluator.setdefault(result["evaluator"], []).append(
            result["cost"]["rollout_seconds"]
        )

    # The referee replay is the same simulator every arm is judged on
    referee_seconds = [
        result["mc_rollout_seconds"]
        for result in results
        if result.get("mc_rollout_seconds")
    ]
    if referee_seconds:
        by_evaluator.setdefault("monte_carlo (referee)", referee_seconds)

    if len(by_evaluator) < 2:
        return None

    labels = sorted(by_evaluator)
    means = [sum(by_evaluator[label]) / len(by_evaluator[label]) for label in labels]

    figure, axes = plt.subplots(figsize=(max(5.5, 1.5 * len(labels)), 4.2))
    bars = axes.bar(labels, means, color="#4C72B0", width=0.55)

    for bar, value in zip(bars, means, strict=True):
        axes.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axes.set_yscale("log")
    axes.set_ylabel("seconds per rollout (log scale)")
    axes.set_title(
        f"{title_prefix}: rollout cost, {max(means) / max(min(means), 1e-9):.0f}x spread"
    )
    axes.grid(alpha=0.3, axis="y")
    plt.setp(axes.get_xticklabels(), rotation=20, ha="right", fontsize=8)

    return _save(figure, out_path)


def plot_convergence(
    results: list[dict], out_path: Path, title_prefix: str
) -> Path | None:
    """Best-so-far reward per outer iteration, at the largest budget in the sweep."""
    with_history = [result for result in results if result.get("history")]
    if not with_history:
        return None

    largest = max(result["budget"] for result in with_history)
    at_largest = [result for result in with_history if result["budget"] == largest]

    figure, axes = plt.subplots(figsize=figure_size)
    plotted = 0

    for index, result in enumerate(at_largest):
        points = [entry for entry in result["history"] if entry.get("reward") is not None]
        if len(points) < 2:
            continue

        axes.plot(
            [entry["iteration"] for entry in points],
            [entry.get("best", entry["reward"]) for entry in points],
            label=result["arm"],
            **_arm_style(index),
        )
        plotted += 1

    if not plotted:
        plt.close(figure)
        return None

    axes.set_xlabel("outer-loop iteration")
    axes.set_ylabel(
        "best F1 so far (selection split)"
        if is_recover(at_largest)
        else "best spread so far (nodes)"
    )
    axes.set_title(f"{title_prefix}: outer-loop convergence at k={largest}")
    axes.grid(alpha=0.3)
    axes.legend(fontsize=8)

    return _save(figure, out_path)


def plot_cascade(results: list[dict], out_path: Path, title_prefix: str) -> Path | None:
    """Infected count per timestep for each arm's winning strategy, largest budget."""
    with_timeline = [result for result in results if result.get("timeline")]
    if not with_timeline:
        return None

    largest = max(result["budget"] for result in with_timeline)
    at_largest = [result for result in with_timeline if result["budget"] == largest]

    figure, axes = plt.subplots(figsize=figure_size)

    for index, result in enumerate(sorted(at_largest, key=lambda r: r["arm"])):
        counts = [entry["infected_count"] for entry in result["timeline"]]
        axes.plot(range(len(counts)), counts, label=result["arm"], **_arm_style(index))

    axes.set_xlabel("timestep")
    axes.set_ylabel("infected nodes")
    axes.set_title(f"{title_prefix}: cascade over time at k={largest}")
    axes.grid(alpha=0.3)
    axes.legend(fontsize=8)

    return _save(figure, out_path)


def plot_wm_training(
    wm_results: dict, out_path: Path, title_prefix: str
) -> Path | None:
    history = wm_results.get("history") or []
    if len(history) < 2:
        return None

    figure, loss_axes = plt.subplots(figsize=figure_size)
    epochs = [entry["epoch"] for entry in history]

    loss_axes.plot(
        epochs, [entry["train_loss"] for entry in history], color="#4C72B0", label="train loss"
    )
    loss_axes.set_xlabel("epoch")
    loss_axes.set_ylabel("train loss", color="#4C72B0")
    loss_axes.tick_params(axis="y", labelcolor="#4C72B0")
    loss_axes.grid(alpha=0.3)

    f1_axes = loss_axes.twinx()
    f1_axes.plot(
        epochs,
        [entry["val_delta_f1"] for entry in history],
        color="#C44E52",
        label="val delta_f1",
    )
    f1_axes.set_ylabel("val delta_f1", color="#C44E52")
    f1_axes.tick_params(axis="y", labelcolor="#C44E52")

    loss_axes.set_title(f"{title_prefix}: world-model training")

    return _save(figure, out_path)


def plot_wm_one_step(
    wm_results: dict, out_path: Path, title_prefix: str
) -> Path | None:
    test = wm_results.get("test") or {}
    keys = [
        "delta_f1",
        "new_infection_f1",
        "infected_acc",
        "frontier_acc",
        "action_sensitivity",
    ]
    present = [key for key in keys if key in test]
    if not present:
        return None

    figure, axes = plt.subplots(figsize=figure_size)
    values = [float(test[key]) for key in present]

    bars = axes.bar(present, values, color="#4C72B0", width=0.6)
    for bar, value in zip(bars, values, strict=True):
        axes.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    axes.set_ylim(0, 1.05)
    axes.set_ylabel("score")
    axes.set_title(f"{title_prefix}: world-model one-step accuracy (test)")
    axes.grid(alpha=0.3, axis="y")
    plt.setp(axes.get_xticklabels(), rotation=20, ha="right")

    return _save(figure, out_path)


def plot_wm_rollout(wm_results: dict, out_path: Path, title_prefix: str) -> Path | None:
    """Final cascade size, model vs truth — the saturation check."""
    rollout = wm_results.get("rollout") or {}
    model_count = rollout.get("ens_final_count_model")
    true_count = rollout.get("ens_final_count_true")
    if model_count is None or true_count is None:
        return None

    figure, axes = plt.subplots(figsize=(5.0, 4.0))
    values = [float(model_count), float(true_count)]

    bars = axes.bar(
        ["world model", "true simulator"], values, color=["#4C72B0", "#55A868"], width=0.55
    )
    for bar, value in zip(bars, values, strict=True):
        axes.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    bias = rollout.get("ens_count_bias")
    axes.set_ylabel("final infected count")
    axes.set_title(
        f"{title_prefix}: rollout fidelity"
        + (f" (count bias {float(bias):+.2f})" if bias is not None else "")
    )
    axes.grid(alpha=0.3, axis="y")

    return _save(figure, out_path)


def plot_dismantling_curve(
    results: list[dict], out_path: Path, title_prefix: str
) -> Path | None:
    """
    s(q) = |GCC| / N after each sequential removal — the physics branch's own figure.

    None for every sweep that removes nothing. This is the curve the published
    dismantling numbers collapse to a scalar, so plotting it is what lets a reader
    place our arms next to that literature at all. It can and does disagree with
    the spread figure: research/critical_node_detection.md §5.8 shows the same
    centralities rank in opposite orders under the two objectives.
    """
    with_curve = [
        result
        for result in results
        if (result.get("structural") or {}).get("gcc_curve")
    ]
    if not with_curve:
        return None

    largest = max(result["budget"] for result in with_curve)
    at_largest = [result for result in with_curve if result["budget"] == largest]

    figure, axes = plt.subplots(figsize=figure_size)

    for index, arm in enumerate(_sorted_arms(at_largest)):
        run = _by_arm(at_largest, arm)[0]
        curve = run["structural"]["gcc_curve"]
        nodes = run["graph"]["num_nodes"]

        axes.plot(
            [100.0 * step / nodes for step in range(len(curve))],
            curve,
            marker=marker_cycle[index % len(marker_cycle)],
            markersize=4,
            markevery=max(1, len(curve) // 8),
            color=condition_colors.get(_condition_of(at_largest, arm)),
            label=arm,
            linewidth=1.8,
        )

    threshold = at_largest[0]["structural"].get("gcc_threshold", 0.01)
    axes.axhline(
        threshold,
        color="#888888",
        linestyle="--",
        linewidth=1.0,
        label=f"dismantled threshold ({threshold:.0%} of N)",
    )

    axes.set_xlabel("nodes removed (% of N), sequentially")
    axes.set_ylabel("|GCC| / N — LOWER IS BETTER")
    axes.set_title(f"{title_prefix}: dismantling curve at k={largest}")
    axes.set_ylim(0, 1.02)
    axes.grid(alpha=0.3)
    axes.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    return _save(figure, out_path)


def plot_structural_vs_spread(
    results: list[dict], out_path: Path, title_prefix: str
) -> Path | None:
    """
    Every arm as one point: what its removals did to CONNECTIVITY against what they
    did to the CASCADE.

    The figure that makes §5.8 visible. If the two objectives agreed the points
    would lie on a line; the literature says they do not, and this is where our own
    data either confirms that or does not.
    """
    with_structural = [
        result for result in results if result.get("structural") is not None
    ]
    if len(with_structural) < 3:
        return None

    largest = max(result["budget"] for result in with_structural)
    at_largest = [result for result in with_structural if result["budget"] == largest]

    figure, axes = plt.subplots(figsize=(7.0, 4.8))

    for arm in _sorted_arms(at_largest):
        run = _by_arm(at_largest, arm)[0]
        axes.scatter(
            run["structural"]["largest_cc_drop_pct"],
            100.0 * ground_truth_reward(run) / run["graph"]["num_nodes"],
            s=70,
            color=condition_colors.get(run.get("condition", 99), "#4C72B0"),
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
        axes.annotate(
            arm,
            (
                run["structural"]["largest_cc_drop_pct"],
                100.0 * ground_truth_reward(run) / run["graph"]["num_nodes"],
            ),
            fontsize=6,
            xytext=(4, 4),
            textcoords="offset points",
        )

    axes.set_xlabel("giant-component drop (%) — structural objective, higher is better")
    axes.set_ylabel("final infected (% of N) — diffusion objective, lower is better")
    axes.set_title(
        f"{title_prefix}: connectivity vs containment at k={largest} "
        f"(bottom-right is best on both)"
    )
    axes.grid(alpha=0.3)

    return _save(figure, out_path)


def plot_prevented_influence(
    results: list[dict], out_path: Path, title_prefix: str
) -> Path | None:
    """
    Prevented influence per arm across the budget sweep — this literature's own figure.

    None for every sweep that is not two-cascade. The y-axis is
    `sigma(S_N, empty) - sigma(S_N | blockers)` on the SHARED referee, which is the
    quantity all five published names in research/influence_blocking.md §8.1 refer to,
    and it is what makes our table readable next to SandIMIN's Table 5 (which reports
    exactly this, as "decreased spread"). The dashed line is the whole cascade: an arm
    touching it stopped the rumour outright.
    """
    scored = [
        result
        for result in results
        if result.get("blocking") and result.get("mc_prevented_influence") is not None
    ]
    if not scored:
        return None

    figure, axes = plt.subplots(figsize=figure_size)

    for index, arm in enumerate(_sorted_arms(scored)):
        runs = sorted(_by_arm(scored, arm), key=lambda result: result["budget"])
        axes.plot(
            [result["budget"] for result in runs],
            [result["mc_prevented_influence"] for result in runs],
            color=condition_colors.get(_condition_of(scored, arm)),
            label=arm,
            **_arm_style(index),
        )

    unopposed = max(result.get("mc_unopposed_spread", 0.0) for result in scored)
    if unopposed:
        axes.axhline(
            unopposed,
            color="#888888",
            linestyle="--",
            linewidth=1.0,
            label=f"whole cascade ({unopposed:.0f} nodes)",
        )

    lever = scored[0].get("lever", "?")
    negative = scored[0].get("n_negative_seeds", "?")
    axes.set_xlabel("blocker budget k (absolute — this literature's own convention)")
    axes.set_ylabel("prevented influence (nodes) — HIGHER IS BETTER")
    axes.set_title(
        f"{title_prefix}: prevented influence, lever={lever}, |S_N|={negative}"
    )
    axes.grid(alpha=0.3)
    axes.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    return _save(figure, out_path)


def plot_blocking_ratio(
    results: list[dict], out_path: Path, title_prefix: str
) -> Path | None:
    """
    Prevented fraction against the `|S_P| / |S_N|` ratio — CLDAG's Table 2 as a figure.

    §8.2: the informative budget axis here is the ratio to the ATTACKER's budget, not
    the fraction of the graph. CLDAG's own reading of this curve is the design fact
    the whole task is configured around — it takes 20-30x the rumour's seeds to cut it
    to 10%, and "first mover has a clear advantage".
    """
    scored = [
        result
        for result in results
        if result.get("blocking")
        and result.get("budget_ratio")
        and result.get("mc_prevented_pct_of_unopposed") is not None
    ]
    if not scored:
        return None

    figure, axes = plt.subplots(figsize=figure_size)

    for index, arm in enumerate(_sorted_arms(scored)):
        runs = sorted(_by_arm(scored, arm), key=lambda result: result["budget_ratio"])
        axes.plot(
            [result["budget_ratio"] for result in runs],
            [result["mc_prevented_pct_of_unopposed"] for result in runs],
            color=condition_colors.get(_condition_of(scored, arm)),
            label=arm,
            **_arm_style(index),
        )

    axes.set_xlabel("|S_P| / |S_N| — blockers per rumour seed")
    axes.set_ylabel("% of the cascade prevented — HIGHER IS BETTER")
    axes.set_title(f"{title_prefix}: prevented fraction vs the attacker's budget")
    axes.set_ylim(0, 100)
    axes.grid(alpha=0.3)
    axes.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    return _save(figure, out_path)


localization_metric_keys = ("precision", "recall", "f1", "auc")


def plot_localization_metrics(
    results: list[dict], out_path: Path, title_prefix: str
) -> Path | None:
    """
    PR / RE / F1 / AUC per arm, on held-out episodes — the SL literature's own figure.

    None for every sweep that does not invert. Four bars per arm rather than one,
    because the split between precision and recall is where this literature's
    failure modes live: IVGD hits `RE = 1.0000` on five of six graphs and lets
    precision carry F1, which is the signature of a thresholded relaxation
    over-predicting (research/source_localization.md §5.2).
    """
    scored = [
        result
        for result in results
        if result.get("metrics") and not result.get("reconstruction")
    ]
    if not scored:
        return None

    arms = _sorted_arms(scored)
    figure, axes = plt.subplots(figsize=(max(7.0, 0.9 * len(arms)), 4.8))

    positions = np.arange(len(arms))
    width = 0.8 / len(localization_metric_keys)
    hatches = ("", "///", "...", "xxx")

    for index, key in enumerate(localization_metric_keys):
        values = [
            float(_by_arm(scored, arm)[0]["metrics"].get(key) or 0.0) for arm in arms
        ]
        axes.bar(
            positions + index * width - 0.4 + width / 2,
            values,
            width=width,
            color=[
                condition_colors.get(_condition_of(scored, arm), "#4C72B0")
                for arm in arms
            ],
            edgecolor="white",
            linewidth=0.6,
            hatch=hatches[index % len(hatches)],
        )

    # Proxy patches, because each bar carries a LIST of colours (one per arm) and
    # matplotlib would draw the legend swatch in whichever arm happened to be
    # first — saying "precision is grey" on a figure where it is also blue
    legend_handles = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="white", edgecolor="#444444",
            hatch=hatches[index % len(hatches)],
        )
        for index in range(len(localization_metric_keys))
    ]

    axes.set_xticks(positions)
    axes.set_xticklabels(arms, rotation=30, ha="right", fontsize=7)
    axes.set_ylabel("score on held-out episodes — HIGHER IS BETTER")
    axes.set_ylim(0, 1.02)
    axes.axhline(0.5, color="#888888", linestyle=":", linewidth=0.8)

    # PR = RE = F1 whenever every arm spends its whole budget at the instance's own
    # k, which is the published given-k convention and the default. Stated on the
    # figure rather than left to look like a plotting bug — and `--sl-budget-mode
    # sweep` genuinely separates them, so it cannot just be dropped.
    given_k = all(
        abs(
            float(_by_arm(scored, arm)[0]["metrics"].get("precision") or 0.0)
            - float(_by_arm(scored, arm)[0]["metrics"].get("recall") or 0.0)
        )
        < 1e-9
        for arm in arms
    )
    note = (
        "  —  PR = RE = F1 by construction: each arm is given the instance's own k"
        if given_k
        else ""
    )
    axes.set_title(
        f"{title_prefix}: source recovery{note}",
        fontsize=10,
    )
    axes.grid(alpha=0.3, axis="y")
    # Below the axes rather than inside them: AUC routinely sits above 0.9 on this
    # figure, so an in-axes legend covers the bars it is labelling
    axes.legend(
        legend_handles,
        [key.upper() if key == "auc" else key for key in localization_metric_keys],
        fontsize=7,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.42),
        title="hatch = metric; colour = condition",
        title_fontsize=7,
    )

    return _save(figure, out_path)


def plot_localization_cost(
    results: list[dict], out_path: Path, title_prefix: str
) -> Path | None:
    """
    F1 against what each arm spent to get it — the cost claim, as one figure.

    The load-bearing claim of §2.3.2 is a SEARCH-time one: with `P ~ 100` programs,
    `M ~ 100` instances and `C ~ 100` candidate evaluations the product is a
    million rollouts before the MC multiplier, which is why nobody has run a
    program search over an inverse problem. This plots the axis that shows it.
    """
    scored = [
        result
        for result in results
        if result.get("metrics")
        and not result.get("reconstruction")
        and result.get("evaluator_seconds") is not None
    ]
    if len(scored) < 2:
        return None

    figure, axes = plt.subplots(figsize=(7.0, 4.8))

    for arm in _sorted_arms(scored):
        run = _by_arm(scored, arm)[0]
        # Clamped away from zero so a native arm (which never calls an evaluator)
        # still has a position on a log axis instead of vanishing
        seconds = max(float(run.get("evaluator_seconds") or 0.0), 1e-2)
        axes.scatter(
            seconds,
            float(run["metrics"].get("f1") or 0.0),
            s=80,
            color=condition_colors.get(run.get("condition", 99), "#4C72B0"),
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
        axes.annotate(
            arm,
            (seconds, float(run["metrics"].get("f1") or 0.0)),
            fontsize=6,
            xytext=(5, 4),
            textcoords="offset points",
        )

    axes.set_xscale("log")
    axes.set_xlabel("seconds inside the evaluator (log scale) — lower is cheaper")
    axes.set_ylabel("held-out F1 — higher is better")
    axes.set_title(
        f"{title_prefix}: recovery quality against search cost (top-left is best)"
    )
    axes.grid(alpha=0.3, which="both")

    return _save(figure, out_path)


def plot_generalization_gap(
    results: list[dict], out_path: Path, title_prefix: str
) -> Path | None:
    """
    Selection F1 against held-out F1 — did the program learn an algorithm or memorize?

    §8.5.1's episode axis, plotted. A point on the diagonal transferred perfectly;
    a point far below it scored well on the episodes the outer loop optimized
    against and badly on ones it never saw, which is the failure that would make
    the whole amortization claim vacuous.
    """
    paired = [
        result
        for result in results
        if result.get("metrics") and result.get("selection_metrics")
    ]
    if len(paired) < 2:
        return None

    figure, axes = plt.subplots(figsize=(5.8, 5.4))
    axes.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1.0,
              label="perfect transfer")

    # A decoder is selected on the tree-weighted score and a localizer on F1;
    # both live in [0, 1] and both are stored the same way, so one figure serves
    # them once it reads the right key
    decodes = is_reconstruct(paired)
    key = "reward" if decodes else "f1"

    for arm in _sorted_arms(paired):
        run = _by_arm(paired, arm)[0]
        selection = float(run["selection_metrics"].get(key) or 0.0)
        heldout = float(
            (run.get("mc_reward") if decodes else run["metrics"].get(key)) or 0.0
        )
        axes.scatter(
            selection,
            heldout,
            s=80,
            color=condition_colors.get(run.get("condition", 99), "#4C72B0"),
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
        axes.annotate(
            arm, (selection, heldout), fontsize=6, xytext=(5, 4),
            textcoords="offset points",
        )

    unit = "score" if decodes else "F1"
    noun = "cascades" if decodes else "episodes"
    axes.set_xlabel(f"{unit} on the {noun} the search optimized against")
    axes.set_ylabel(f"{unit} on held-out {noun}")
    axes.set_xlim(0, 1.02)
    axes.set_ylim(0, 1.02)
    axes.set_title(f"{title_prefix}: generalization across {noun}")
    axes.grid(alpha=0.3)
    axes.legend(fontsize=7, loc="lower right")

    return _save(figure, out_path)


reconstruction_metric_keys = (
    "path_precision",
    "path_recall",
    "jaccard",
    "event_f1",
    "node_f1",
)


def plot_reconstruction_metrics(
    results: list[dict], out_path: Path, title_prefix: str
) -> Path | None:
    """
    The tree half and the node half per arm, on held-out cascades.

    None for every sweep that does not decode. Five bars per arm rather than one,
    because the whole point of this task is the GAP between them: Zong ICDM'12
    reports 100% node precision alongside 78% edge precision and DIPT's best path
    precision thirteen years later is 0.680, so a figure showing only the score
    would hide the finding it exists to show
    (research/cascade_reconstruction.md §5.2, §8.1).

    `path_recall` and `jaccard` are drawn beside `path_precision` on purpose: an
    arm that names few edges scores well on precision alone, which is the
    under-prediction corner a program search would otherwise find.
    """
    scored = [
        result
        for result in results
        if result.get("reconstruction") and result.get("metrics")
    ]
    if not scored:
        return None

    arms = _sorted_arms(scored)
    figure, axes = plt.subplots(figsize=(max(7.5, 1.0 * len(arms)), 4.8))

    positions = np.arange(len(arms))
    width = 0.8 / len(reconstruction_metric_keys)
    hatches = ("", "///", "...", "xxx", "\\\\\\")

    for index, key in enumerate(reconstruction_metric_keys):
        values = [
            float(_by_arm(scored, arm)[0]["metrics"].get(key) or 0.0) for arm in arms
        ]
        axes.bar(
            positions + index * width - 0.4 + width / 2,
            values,
            width=width,
            color=[
                condition_colors.get(_condition_of(scored, arm), "#4C72B0")
                for arm in arms
            ],
            edgecolor="white",
            linewidth=0.6,
            hatch=hatches[index % len(hatches)],
        )

    # Proxy patches: each bar carries a LIST of colours (one per arm), so
    # matplotlib would draw the legend swatch in whichever arm came first
    legend_handles = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="white", edgecolor="#444444",
            hatch=hatches[index % len(hatches)],
        )
        for index in range(len(reconstruction_metric_keys))
    ]

    # The reward-sanity line §2.11 risk 1 requires: if an arm sits near it, the
    # reward is wrong rather than the arm good
    trivial = next(
        (
            result["trivial_decoder_reward"]
            for result in scored
            if result.get("trivial_decoder_reward") is not None
        ),
        None,
    )
    if trivial is not None:
        axes.axhline(
            trivial,
            color="#B03030",
            linestyle="--",
            linewidth=1.0,
            label="trivial decoder",
        )
        axes.text(
            len(arms) - 0.5,
            trivial + 0.01,
            f"trivial decoder ({trivial:.3f})",
            fontsize=6,
            color="#B03030",
            ha="right",
        )

    axes.set_xticks(positions)
    axes.set_xticklabels(arms, rotation=30, ha="right", fontsize=7)
    axes.set_ylabel("score on held-out cascades — HIGHER IS BETTER")
    axes.set_ylim(0, 1.02)
    axes.set_title(
        f"{title_prefix}: trajectory recovery — the tree half against the node half",
        fontsize=10,
    )
    axes.grid(alpha=0.3, axis="y")
    axes.legend(
        legend_handles,
        [key.replace("_", " ") for key in reconstruction_metric_keys],
        fontsize=7,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.42),
        title="hatch = metric; colour = condition",
        title_fontsize=7,
    )

    return _save(figure, out_path)


def plot_tree_vs_node(
    results: list[dict], out_path: Path, title_prefix: str
) -> Path | None:
    """
    Every arm as one point: how well it recovers the NODES against the TREE.

    The figure that makes this task's central asymmetry visible in one look. The
    diagonal is where an arm recovers both equally well; every published method
    sits far BELOW it, because the node set is nearly free and the edges are not.
    An arm hugging the right wall with nothing above the floor has solved the easy
    half and reported it as a reconstruction, which is the failure §2.6 weights the
    reward to prevent.
    """
    scored = [
        result
        for result in results
        if result.get("reconstruction")
        and (result.get("metrics") or {}).get("path_precision") is not None
        and np.isfinite(result["metrics"]["path_precision"])
    ]
    if len(scored) < 2:
        return None

    figure, axes = plt.subplots(figsize=figure_size)

    for index, arm in enumerate(_sorted_arms(scored)):
        result = _by_arm(scored, arm)[0]
        metrics = result["metrics"]
        axes.scatter(
            metrics.get("node_f1", 0.0),
            metrics.get("path_precision", 0.0),
            s=110,
            marker=_arm_style(index)["marker"],
            color=condition_colors.get(_condition_of(scored, arm), "#4C72B0"),
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        axes.annotate(
            arm,
            (metrics.get("node_f1", 0.0), metrics.get("path_precision", 0.0)),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
        )

    axes.plot([0, 1], [0, 1], color="#888888", linestyle=":", linewidth=0.9)
    axes.text(
        0.62, 0.66, "equal difficulty", fontsize=7, color="#888888", rotation=45
    )
    axes.set_xlabel("node F1 — WHICH nodes were infected (the easy half)")
    axes.set_ylabel("path precision — WHO infected whom (the hard half)")
    axes.set_xlim(0, 1.02)
    axes.set_ylim(0, 1.02)
    axes.set_title(
        f"{title_prefix}: the node set is easy, the tree is hard", fontsize=10
    )
    axes.grid(alpha=0.3)

    return _save(figure, out_path)


def build_plots(
    agent_results: list[dict],
    wm_results: dict | None,
    plots_dir: Path,
    title_prefix: str,
) -> list[Path]:
    os.makedirs(plots_dir, exist_ok=True)

    # Every figure below whose y-axis is a NODE COUNT. An inverse task's reward is
    # an F1 in [0, 1], and drawing it under a "spread (nodes)" axis would be off by
    # three orders of magnitude with a label that hides it — so those builders are
    # skipped outright rather than relabelled.
    spread_figures = (
        []
        if is_recover(agent_results)
        else [
            plot_budget_vs_spread(
                agent_results, plots_dir / "budget_vs_spread.png", title_prefix
            ),
            plot_budget_vs_spread(
                agent_results,
                plots_dir / "budget_vs_spread_pct.png",
                title_prefix,
                normalize=True,
            ),
            plot_ours_vs_baselines(
                agent_results, plots_dir / "ours_vs_baselines.png", title_prefix
            ),
            plot_ours_vs_baselines(
                agent_results,
                plots_dir / "ours_vs_baselines_pct.png",
                title_prefix,
                normalize=True,
            ),
            plot_condition_comparison(
                agent_results, plots_dir / "condition_comparison.png", title_prefix
            ),
            plot_evaluator_fidelity(
                agent_results, plots_dir / "evaluator_fidelity.png", title_prefix
            ),
            plot_cascade(agent_results, plots_dir / "cascade.png", title_prefix),
        ]
    )

    builders = spread_figures + [
        plot_sample_efficiency(
            agent_results, plots_dir / "sample_efficiency.png", title_prefix
        ),
        plot_runtime(agent_results, plots_dir / "runtime.png", title_prefix),
        plot_convergence(agent_results, plots_dir / "convergence.png", title_prefix),
        # Both return None on a non-adaptive sweep, so no guard is needed here
        plot_adaptivity_gap(
            agent_results, plots_dir / "adaptivity_gap.png", title_prefix
        ),
        plot_round_spreads(
            agent_results, plots_dir / "round_spreads.png", title_prefix
        ),
        # Both return None on a sweep that removes nothing, same as the adaptive
        # pair above, so no guard is needed here either
        plot_dismantling_curve(
            agent_results, plots_dir / "dismantling_curve.png", title_prefix
        ),
        plot_structural_vs_spread(
            agent_results, plots_dir / "structural_vs_spread.png", title_prefix
        ),
        # Both return None on a sweep with only one cascade
        plot_prevented_influence(
            agent_results, plots_dir / "prevented_influence.png", title_prefix
        ),
        plot_blocking_ratio(
            agent_results, plots_dir / "blocking_ratio.png", title_prefix
        ),
        # All three return None on a sweep that recovers nothing, same as the
        # adaptive and dismantling pairs above
        plot_localization_metrics(
            agent_results, plots_dir / "localization_metrics.png", title_prefix
        ),
        plot_localization_cost(
            agent_results, plots_dir / "localization_cost.png", title_prefix
        ),
        plot_generalization_gap(
            agent_results, plots_dir / "generalization_gap.png", title_prefix
        ),
        # Both return None on a sweep that decodes nothing, same as every pair above
        plot_reconstruction_metrics(
            agent_results, plots_dir / "reconstruction_metrics.png", title_prefix
        ),
        plot_tree_vs_node(
            agent_results, plots_dir / "tree_vs_node.png", title_prefix
        ),
    ]

    if wm_results is not None:
        builders += [
            plot_wm_training(wm_results, plots_dir / "wm_training.png", title_prefix),
            plot_wm_one_step(wm_results, plots_dir / "wm_one_step.png", title_prefix),
            plot_wm_rollout(wm_results, plots_dir / "wm_rollout.png", title_prefix),
        ]

    return [figure for figure in builders if figure is not None]
