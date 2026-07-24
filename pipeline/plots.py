"""
Paper figures from the JSONs the pipeline already wrote. Every figure is guarded
by the data it needs, so a partial run just produces fewer plots instead of failing.
"""

import os
from pathlib import Path
import matplotlib

# Headless: SLURM nodes have no display
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

figure_dpi = 200
figure_size = (7.0, 4.5)
marker_cycle = ("o", "s", "^", "D", "v", "P", "X", "*")


def _arm_style(index: int) -> dict:
    return {"marker": marker_cycle[index % len(marker_cycle)], "linewidth": 1.8}


def _sorted_arms(results: list[dict]) -> list[str]:
    # Baselines first, then routing, then the synthesis arms — matches the paper's table order
    def rank(arm: str) -> tuple:
        return (0 if arm.startswith("baseline_") else 1 if arm == "routing" else 2, arm)

    return sorted({result["arm"] for result in results}, key=rank)


def _by_arm(results: list[dict], arm: str) -> list[dict]:
    return sorted(
        (result for result in results if result["arm"] == arm),
        key=lambda result: result["budget"],
    )


def _reward_se(result: dict) -> float:
    return float(result.get("cost", {}).get("reward_se", 0.0) or 0.0)


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

        x_values = [
            run["budget_pct"] if normalize else run["budget"] for run in runs
        ]
        y_values = [
            run["spread_pct"] if normalize else run["reward"] for run in runs
        ]
        errors = [
            _reward_se(run) / run["graph"]["num_nodes"] * 100 if normalize
            else _reward_se(run)
            for run in runs
        ]

        axes.errorbar(
            x_values, y_values, yerr=errors, label=arm, capsize=3, **_arm_style(index)
        )

    axes.set_xlabel("seed budget (% of nodes)" if normalize else "seed budget k")
    axes.set_ylabel("final spread (% of nodes)" if normalize else "final spread (nodes)")
    axes.set_title(f"{title_prefix}: influence spread vs seed budget")
    axes.grid(alpha=0.3)
    axes.legend(fontsize=8)

    return _save(figure, out_path)


def plot_evaluator_fidelity(
    results: list[dict], out_path: Path, title_prefix: str
) -> Path | None:
    """Evaluator reward vs the Monte Carlo referee — points off the diagonal are model error."""
    paired = [result for result in results if result.get("mc_reward") is not None]
    if not paired:
        return None

    figure, axes = plt.subplots(figsize=(5.0, 5.0))

    for index, arm in enumerate(_sorted_arms(paired)):
        runs = [result for result in paired if result["arm"] == arm]
        axes.scatter(
            [run["mc_reward"] for run in runs],
            # The unbiased estimate when it exists: `reward` carries the winner's curse
            [run.get("wm_reeval_mean", run["reward"]) for run in runs],
            label=arm,
            marker=marker_cycle[index % len(marker_cycle)],
            alpha=0.85,
        )

    limits = [
        min(run["mc_reward"] for run in paired) * 0.95,
        max(run["mc_reward"] for run in paired) * 1.05,
    ]
    axes.plot(limits, limits, "k--", linewidth=1, label="perfect fidelity")

    axes.set_xlabel("Monte Carlo spread (ground truth)")
    axes.set_ylabel(f"{paired[0]['evaluator']} spread (estimate)")
    axes.set_title(f"{title_prefix}: evaluator fidelity")
    axes.grid(alpha=0.3)
    axes.legend(fontsize=8)

    return _save(figure, out_path)


def plot_runtime(results: list[dict], out_path: Path, title_prefix: str) -> Path | None:
    """Per-rollout wall-clock: the amortization claim for replacing the simulator."""
    paired = [result for result in results if result.get("mc_rollout_seconds")]
    if not paired:
        return None

    evaluator_seconds = [
        result["cost"].get("rollout_seconds", 0.0) for result in paired
    ]
    mc_seconds = [result["mc_rollout_seconds"] for result in paired]

    figure, axes = plt.subplots(figsize=(5.5, 4.0))
    labels = [paired[0]["evaluator"], "monte_carlo"]
    means = [
        sum(evaluator_seconds) / len(evaluator_seconds),
        sum(mc_seconds) / len(mc_seconds),
    ]

    bars = axes.bar(labels, means, color=["#4C72B0", "#C44E52"], width=0.55)
    for bar, value in zip(bars, means, strict=True):
        axes.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axes.set_yscale("log")
    axes.set_ylabel("seconds per rollout (log scale)")
    axes.set_title(f"{title_prefix}: rollout cost, {means[1] / max(means[0], 1e-9):.1f}x speedup")
    axes.grid(alpha=0.3, axis="y")

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
    figures = [
        plot_budget_vs_spread(
            agent_results, plots_dir / "budget_vs_spread.png", title_prefix
        ),
        plot_budget_vs_spread(
            agent_results,
            plots_dir / "budget_vs_spread_pct.png",
            title_prefix,
            normalize=True,
        ),
        plot_evaluator_fidelity(
            agent_results, plots_dir / "evaluator_fidelity.png", title_prefix
        ),
        plot_runtime(agent_results, plots_dir / "runtime.png", title_prefix),
        plot_convergence(agent_results, plots_dir / "convergence.png", title_prefix),
        plot_cascade(agent_results, plots_dir / "cascade.png", title_prefix),
    ]

    if wm_results is not None:
        figures += [
            plot_wm_training(wm_results, plots_dir / "wm_training.png", title_prefix),
            plot_wm_one_step(wm_results, plots_dir / "wm_one_step.png", title_prefix),
            plot_wm_rollout(wm_results, plots_dir / "wm_rollout.png", title_prefix),
        ]

    return [figure for figure in figures if figure is not None]
