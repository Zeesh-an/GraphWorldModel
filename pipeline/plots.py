"""
Paper figures from the JSONs the pipeline already wrote. Every figure is guarded
by the data it needs, so a partial run just produces fewer plots instead of failing.
"""

import os
from pathlib import Path
import matplotlib

from pipeline.conditions import condition_names, ground_truth_reward, is_ground_truth

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


def _reward_label(results: list[dict]) -> str:
    return (
        "final spread (nodes, ground-truth MC)"
        if is_ground_truth(results)
        else "final spread (nodes, mixed evaluators)"
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

    axes.set_xlabel("seed budget (% of nodes)" if normalize else "seed budget k")
    axes.set_ylabel(
        "final spread (% of nodes)" if normalize else _reward_label(results)
    )
    axes.set_title(f"{title_prefix}: influence spread vs seed budget")
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

    axes.set_xlabel("seed budget (% of nodes)" if normalize else "seed budget k")
    axes.set_ylabel(
        "final spread (% of nodes)" if normalize else _reward_label(results)
    )
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
    axes.set_title(f"{title_prefix}: quality vs real-experience cost at k={largest}")
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
    axes.set_ylabel("best spread so far (nodes)")
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


def build_plots(
    agent_results: list[dict],
    wm_results: dict | None,
    plots_dir: Path,
    title_prefix: str,
) -> list[Path]:
    os.makedirs(plots_dir, exist_ok=True)
    builders = [
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
        plot_sample_efficiency(
            agent_results, plots_dir / "sample_efficiency.png", title_prefix
        ),
        plot_evaluator_fidelity(
            agent_results, plots_dir / "evaluator_fidelity.png", title_prefix
        ),
        plot_runtime(agent_results, plots_dir / "runtime.png", title_prefix),
        plot_convergence(agent_results, plots_dir / "convergence.png", title_prefix),
        plot_cascade(agent_results, plots_dir / "cascade.png", title_prefix),
    ]

    if wm_results is not None:
        builders += [
            plot_wm_training(wm_results, plots_dir / "wm_training.png", title_prefix),
            plot_wm_one_step(wm_results, plots_dir / "wm_one_step.png", title_prefix),
            plot_wm_rollout(wm_results, plots_dir / "wm_rollout.png", title_prefix),
        ]

    return [figure for figure in builders if figure is not None]
