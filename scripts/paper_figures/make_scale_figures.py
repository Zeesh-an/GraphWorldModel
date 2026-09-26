"""
Influence maximization across network sizes, from the full results
table (tab:results_full): the designed algorithm against the strongest row of
the table in each cell, on Network Science, NetHEPT and Digg. Numbers are typed
from the table at its one-decimal resolution.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from pipeline.plots import figure_dpi, figure_formats
from scripts.paper_figures.paper_style import apply_style, baseline_color, ours_color

out_dir = Path("figures/scale")
budgets = ["1%", "5%", "10%", "20%"]
# (network, nodes, {dynamics: (ours per budget, strongest baseline per budget)})
networks = [
    ("Network Science", 1589, {
        "IC": ([8.9, 25.3, 39.1, 60.1], [8.8, 24.8, 38.0, 59.2]),
        "LT": ([11.3, 30.9, 46.8, 69.3], [10.9, 30.2, 45.3, 68.5]),
    }),
    ("NetHEPT", 15229, {
        "IC": ([12.9, 33.0, 47.8, 68.5], [12.6, 31.4, 44.8, 64.5]),
        "LT": ([16.9, 41.4, 58.1, 80.0], [16.2, 39.2, 54.5, 75.7]),
    }),
    ("Digg", 116893, {
        "IC": ([30.4, 45.8, 56.5, 69.4], [27.4, 41.7, 51.8, 62.7]),
        "LT": ([56.8, 73.2, 83.1, 93.5], [50.5, 68.4, 77.7, 85.5]),
    }),
]
panel_size = (3.4, 2.6)
font_size = 9
line_budget = 2  # index of the 10% budget


def gain(ours: list[float], baseline: list[float]) -> float:
    # Mean over budgets of the per-budget ratio, as a percent gain
    return 100.0 * float(np.mean([o / b for o, b in zip(ours, baseline, strict=True)])) - 100.0


def save(figure: plt.Figure, name: str) -> None:
    figure.tight_layout()
    for suffix in figure_formats:
        figure.savefig(out_dir / f"{name}.{suffix}", dpi=figure_dpi, bbox_inches="tight")
    plt.close(figure)


def draw_reward_lines() -> None:
    figure, axes = plt.subplots(figsize=panel_size)
    nodes = [count for _, count, _ in networks]
    ours = [cells["IC"][0][line_budget] for _, _, cells in networks]
    baseline = [cells["IC"][1][line_budget] for _, _, cells in networks]
    axes.plot(nodes, ours, marker="o", markersize=6, linewidth=2.2, color=ours_color, label="Network World Model", zorder=4)
    axes.plot(nodes, baseline, marker="s", markersize=5, linewidth=1.4, color=baseline_color, label="strongest baseline", zorder=3)
    axes.fill_between(nodes, baseline, ours, color=ours_color, alpha=0.12, linewidth=0, zorder=2)
    for x, o, b in zip(nodes, ours, baseline, strict=True):
        axes.text(x, o + 1.2, f"{o:.1f}", ha="center", va="bottom", fontsize=font_size - 2.5, color=ours_color)
        axes.text(x, b - 1.2, f"{b:.1f}", ha="center", va="top", fontsize=font_size - 2.5, color="#C9553F")
    axes.set_xscale("log")
    axes.set_xticks(nodes)
    axes.set_xticklabels([f"{name}\n{count:,}" for name, count, _ in networks], fontsize=font_size - 1.5)
    axes.minorticks_off()
    axes.set_xlabel("network size (nodes)", fontsize=font_size)
    axes.set_ylabel("spread (% of nodes, IC)", fontsize=font_size)
    axes.set_ylim(35, 60)
    axes.set_yticks([35, 40, 45, 50, 55, 60])
    axes.tick_params(axis="y", labelsize=font_size - 1)
    axes.legend(fontsize=font_size - 2, loc="upper left", framealpha=0.9)
    save(figure, "scale_reward_lines")


if __name__ == "__main__":
    apply_style()
    draw_reward_lines()
    print(f"wrote {out_dir}/scale_reward_lines.pdf")
