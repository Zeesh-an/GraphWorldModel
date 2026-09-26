"""
Two cost panels from the cost table of the experiments section: seconds per
rollout sample under the world model and under Monte Carlo on the two timed
networks, and cumulative trusted-simulator episodes as searches accumulate.
Numbers are typed from the table (results/timing for the per-rollout times).
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
import numpy as np

from pipeline.plots import figure_dpi, figure_formats
from scripts.paper_figures.paper_style import apply_style, baseline_color, ours_color

out_dir = Path("figures/cost")
# (network, nodes, world-model seconds per sample, Monte Carlo seconds per episode)
timings = [("Power Grid", 4941, 0.022, 0.089), ("Digg", 116893, 2.467, 35.694)]
training_episodes = 38800
search_episodes = 7800
searches_shown = 10
wm_color, mc_color = ours_color, baseline_color
panel_size = (3.4, 2.6)
font_size = 9
bar_width = 0.34


def save(figure: plt.Figure, name: str) -> None:
    figure.tight_layout()
    for suffix in figure_formats:
        figure.savefig(out_dir / f"{name}.{suffix}", dpi=figure_dpi, bbox_inches="tight")
    plt.close(figure)


def draw_time_per_rollout() -> None:
    figure, axes = plt.subplots(figsize=panel_size)
    positions = np.arange(len(timings))
    wm = [row[2] for row in timings]
    mc = [row[3] for row in timings]
    axes.bar(positions - bar_width / 2, wm, bar_width, color=wm_color, label="Network World Model", zorder=3)
    axes.bar(positions + bar_width / 2, mc, bar_width, color=mc_color, label="Monte Carlo simulation", zorder=3)
    for x, (fast, slow) in zip(positions, zip(wm, mc, strict=True), strict=True):
        axes.text(x - bar_width / 2, fast * 1.25, f"{fast:.3f}s", ha="center", va="bottom", fontsize=font_size - 2)
        axes.text(x + bar_width / 2, slow * 1.25, f"{slow:.3f}s", ha="center", va="bottom", fontsize=font_size - 2)
        # The speedup sits above the pair, in the same units the text quotes
        axes.text(x, slow * 4.2, f"{slow / fast:.1f}$\\times$ faster", ha="center", va="bottom", fontsize=font_size - 1, fontweight="bold", color="#333333")
    axes.set_yscale("log")
    axes.set_ylim(0.008, 900)
    ticks = [0.01, 0.1, 1, 10, 100]
    axes.yaxis.set_major_locator(FixedLocator(ticks))
    axes.yaxis.set_major_formatter(FixedFormatter(["0.01", "0.1", "1", "10", "100"]))
    axes.yaxis.set_minor_locator(NullLocator())
    axes.set_xticks(positions)
    axes.set_xticklabels([f"{name}\n({nodes:,} nodes)" for name, nodes, _, _ in timings], fontsize=font_size - 1)
    axes.set_ylabel("time per rollout (s)", fontsize=font_size)
    axes.tick_params(axis="y", labelsize=font_size - 1)
    axes.tick_params(axis="x", length=0)
    axes.grid(alpha=0.3, axis="y", which="major")
    axes.set_axisbelow(True)
    axes.legend(fontsize=font_size - 2, loc="upper left", framealpha=0.9, handlelength=1.2)
    save(figure, "cost_time_per_rollout")


def draw_simulator_calls() -> None:
    figure, axes = plt.subplots(figsize=panel_size)
    searches = np.arange(searches_shown + 1)
    mc = search_episodes * searches / 1000.0
    wm = np.full_like(mc, training_episodes / 1000.0)
    axes.plot(searches, mc, marker="s", markersize=4, linewidth=2.0, color=mc_color, label=f"Monte Carlo ({search_episodes:,} per search)", zorder=4)
    axes.plot(searches, wm, marker="o", markersize=4, linewidth=2.0, color=wm_color, label=f"Network World Model ({training_episodes:,} once)", zorder=4)
    break_even = training_episodes / search_episodes
    # Shade where the world model has paid for itself
    axes.fill_between(searches, wm, mc, where=mc >= wm, color=wm_color, alpha=0.12, linewidth=0, zorder=2)
    # The drop line stops at the crossing, clear of the legend above it
    axes.plot([break_even, break_even], [0, training_episodes / 1000.0], color="black", linestyle="--", linewidth=0.9, zorder=3)
    axes.plot(break_even, training_episodes / 1000.0, marker="o", markersize=9, markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.2, linestyle="none", zorder=5)
    axes.set_xlim(0, searches_shown)
    axes.set_ylim(0, 90)
    axes.set_xticks(range(0, searches_shown + 1, 2))
    axes.set_xlabel("searches completed", fontsize=font_size)
    axes.set_ylabel("Monte Carlo simulation\nepisodes (thousands)", fontsize=font_size)
    axes.tick_params(labelsize=font_size - 1)
    axes.legend(fontsize=font_size - 2, loc="upper left", framealpha=0.9, handlelength=1.6)
    save(figure, "cost_simulator_calls")


if __name__ == "__main__":
    apply_style()
    draw_time_per_rollout()
    draw_simulator_calls()
    print(f"wrote two figures to {out_dir}")
