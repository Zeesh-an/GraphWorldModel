"""
The appendix's world-model figures in the paper style: the four encoders'
free-running rollouts and their count bias by step on Network Science, and the
count bias by step of the end-to-end checkpoints on Network Science and Power
Grid. Read from the saved evaluation JSONs, so nothing is retrained; entry i of a
count curve is the count after transition i + 1.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from pipeline.plots import figure_dpi, figure_formats
from scripts.paper_figures.paper_style import apply_style, baseline_shades, ours_color

figures_dir = Path("figures")
backbone_host = Path("results/influence_maximization/netscience")
# (label in the paper, run, encoder file stem, color, marker); GraphSAGE is the paper's encoder
backbones = [
    ("GraphSAGE", "abl_wm_sage", "sage", ours_color, "o"),
    ("GCN", "abl_wm_gcn", "gcn", baseline_shades[0], "s"),
    ("GATv2", "abl_wm_gat", "gat", baseline_shades[2], "^"),
    ("GCNII", "abl_wm_gcnii", "gcnii", baseline_shades[3], "D"),
]
horizon_hosts = [
    ("Network Science (1,589)", Path("results/influence_maximization/netscience/final_ic/world_model/sage_IC.json")),
    ("Power Grid (4,941)", Path("results/critical_node_detection/power_grid/final_ic/world_model/sage_IC.json")),
]
simulator_color = "#333333"
panel_size = (3.4, 2.4)
horizon_size = (6.8, 2.3)
font_size = 9


def save(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    for suffix in figure_formats:
        figure.savefig(path.with_suffix(f".{suffix}"), dpi=figure_dpi, bbox_inches="tight")
    plt.close(figure)


def rollout(path: Path) -> tuple[np.ndarray, np.ndarray]:
    curves = json.loads(path.read_text())["rollout"]
    return np.asarray(curves["count_model_curve"], dtype=float), np.asarray(curves["count_true_curve"], dtype=float)


def relative_bias(model: np.ndarray, true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = true > 0
    return np.arange(1, len(model) + 1)[keep], 100.0 * (model[keep] - true[keep]) / true[keep]


def draw_backbones() -> None:
    curves_figure, curves_axes = plt.subplots(figsize=panel_size)
    bias_figure, bias_axes = plt.subplots(figsize=panel_size)
    true_curve = None
    for label, run, stem, color, marker in backbones:
        model, true = rollout(backbone_host / run / "world_model" / f"{stem}_IC.json")
        steps = np.arange(1, len(model) + 1)
        curves_axes.plot(steps, model, marker=marker, markersize=3.5, linewidth=1.5, color=color, label=label)
        bias_steps, bias = relative_bias(model, true)
        bias_axes.plot(bias_steps, bias, marker=marker, markersize=3.5, linewidth=1.5, color=color, label=label)
        true_curve = true
    curves_axes.plot(np.arange(1, len(true_curve) + 1), true_curve, color=simulator_color, linestyle="--", linewidth=1.1, label="simulator")
    bias_axes.axhline(0.0, color=simulator_color, linewidth=0.8)
    for axes, ylabel in ((curves_axes, "mean infected count"), (bias_axes, "relative count bias (%)")):
        axes.set_xlabel("rollout step", fontsize=font_size)
        axes.set_ylabel(ylabel, fontsize=font_size)
        axes.tick_params(labelsize=font_size - 1)
    curves_axes.legend(fontsize=font_size - 2, loc="lower right", handlelength=1.6)
    # The bias lines span the whole width, so the legend goes in one row above them
    low, high = bias_axes.get_ylim()
    bias_axes.set_ylim(low, high + 0.25 * (high - low))
    bias_axes.legend(fontsize=font_size - 2, loc="upper center", ncol=4, handlelength=1.4, columnspacing=0.8, handletextpad=0.4)
    save(curves_figure, figures_dir / "backbones" / "netscience_rollout_curves")
    save(bias_figure, figures_dir / "backbones" / "netscience_bias_by_step")


def draw_horizon() -> None:
    figure, axes_pair = plt.subplots(1, 2, figsize=horizon_size)
    for axes, (title, path) in zip(axes_pair, horizon_hosts, strict=True):
        steps, bias = relative_bias(*rollout(path))
        axes.plot(steps, bias, marker="o", markersize=3.5, linewidth=1.8, color=ours_color)
        axes.axhline(0.0, color=simulator_color, linewidth=0.8)
        axes.set_title(title, fontsize=font_size)
        axes.set_xlabel("rollout step", fontsize=font_size)
        axes.tick_params(labelsize=font_size - 1)
    axes_pair[0].set_ylabel("relative count bias (%)", fontsize=font_size)
    save(figure, figures_dir / "horizon" / "bias_by_step")


if __name__ == "__main__":
    apply_style()
    draw_backbones()
    draw_horizon()
    print(f"wrote the backbone and horizon figures under {figures_dir}")
