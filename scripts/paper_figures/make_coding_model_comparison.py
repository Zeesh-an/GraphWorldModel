"""
Figures comparing the coding models of the design loop, read from the result files
in the style of the budget-curve figures. The paper's model labels differ from
the run names on disk: the paper's Sol is the main run (gpt-6-astra on disk),
Terra is abl_llm_sol and Luna is abl_llm_terra.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from pipeline.plots import figure_dpi, figure_formats
from scripts.paper_figures.paper_style import apply_style, baseline_color, ours_shades

out_dir = Path("figures/coding_model_comparison")
models = [("GPT-5.6 Sol", "final_ic", ours_shades[0]), ("GPT-5.6 Terra", "abl_llm_sol", ours_shades[1]), ("GPT-5.6 Luna", "abl_llm_terra", ours_shades[2])]
budgets = ["pct1", "pct5", "pct10", "pct20"]
budget_labels = ["1%", "5%", "10%", "20%"]
# Strongest row of the main table per budget: the reference the text compares against
hosts = [
    ("influence_maximization/netscience", "netscience", "max",
     {"pct1": "external_opim", "pct5": "baseline_imm", "pct10": "baseline_imm", "pct20": "baseline_imm"}, "IMM"),
    ("critical_node_detection/power_grid", "power_grid", "min",
     {b: "baseline_frontier_removal" for b in budgets}, "Frontier"),
]
rounds_shown = 10
# Both hosts in one square figure, each panel half its width
square_size = (3.4, 2.5)
panel_titles = {"netscience": "IM, Network Science\nhigher is better", "power_grid": "CND, Power Grid\nlower is better"}
# The convergence panels are wide and short: ten rounds side by side need little height
convergence_size = (3.5, 2.05)
convergence_headroom = 1.22
mean_margins = {"left": 0.16, "right": 0.98, "bottom": 0.2, "top": 0.84, "wspace": 0.45}
convergence_margins = {"left": 0.17, "right": 0.98, "bottom": 0.24, "top": 0.97}
legend_size = 8
font_size = 9
# Room above the data for a legend that would otherwise sit on a bar or a line
headroom = 1.3
better_color = ours_shades[0]
worse_color = baseline_color
shade_alpha = 0.08
worse_strip = 0.12


def load(host: str, run: str, budget: str, arm: str) -> dict:
    for sub in ("agent", "baselines"):
        path = Path("results") / host / run / sub / budget / f"{arm}.json"
        if path.exists():
            return json.load(open(path))
    raise FileNotFoundError(f"{host}/{run}/{budget}/{arm}")


def pct(record: dict, value: float) -> float:
    return 100.0 * value / record["graph"]["num_nodes"]


def gain(sense: str, ours: float, reference: float) -> float:
    # Sign-corrected so that a positive gain is an improvement on both hosts
    return 100.0 * ((ours - reference) if sense == "max" else (reference - ours)) / reference


def save(figure: plt.Figure, name: str, margins: dict) -> None:
    # Fixed margins and no tight crop, so the two hosts' panels have identical page sizes
    figure.subplots_adjust(**margins)
    for suffix in figure_formats:
        figure.savefig(out_dir / f"{name}.{suffix}", dpi=figure_dpi)
    plt.close(figure)


def draw_mean_bars() -> None:
    # One bar per coding model and one for the strongest baseline, each the reward
    # averaged over the four budgets; the baseline is the strongest table row per cell
    figure, axes_pair = plt.subplots(1, 2, figsize=square_size)
    for axes, (host, stem, sense, reference_rows, _) in zip(axes_pair, hosts, strict=True):
        values, colors = [], []
        for _, run, color in models:
            records = [load(host, run, budget, "evolve_free@oracle") for budget in budgets]
            values.append(float(np.mean([pct(record, record["referee_reward"]) for record in records])))
            colors.append(color)
        references = [load(host, "final_ic", budget, reference_rows[budget]) for budget in budgets]
        values.append(float(np.mean([pct(record, record["referee_reward"]) for record in references])))
        colors.append(baseline_color)
        positions = np.arange(len(values))
        # Full-width bars, so the four touch
        axes.bar(positions, values, 1.0, color=colors, linewidth=0, zorder=3)
        axes.set_xlim(-0.5, len(values) - 0.5)
        axes.set_xticks([])
        # A floor just under the smallest bar, so the differences between bars are visible
        low, high = min(values), max(values)
        axes.set_ylim(low - 0.6 * (high - low) - 0.5, high + 0.25 * (high - low) + 0.3)
        axes.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axes.tick_params(axis="y", labelsize=font_size - 1)
        axes.set_title(panel_titles[stem], fontsize=font_size, pad=4)
    axes_pair[0].set_ylabel("mean reward (% of nodes)", fontsize=font_size)
    # One legend for both panels, under them: the three models and the baseline
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in [color for _, _, color in models] + [baseline_color]]
    names = [label.replace("GPT-5.6 ", "") for label, _, _ in models] + ["strongest baseline"]
    figure.legend(handles, names, loc="lower center", ncol=4, fontsize=legend_size, handlelength=1.0, columnspacing=0.9, handletextpad=0.4, bbox_to_anchor=(0.55, 0.0))
    save(figure, "coding_model_mean", mean_margins)


def draw_convergence() -> None:
    for host, stem, sense, reference_rows, reference_label in hosts:
        figure, axes = plt.subplots(figsize=convergence_size)
        for label, run, color in models:
            record = load(host, run, "pct10", "evolve_free@oracle")
            history = record["history"][:rounds_shown]
            # A round whose candidate failed validation has no score and leaves a gap
            values = [np.nan if entry.get("best") is None else pct(record, entry["best"]) for entry in history]
            # Short names keep the legend on one row; the caption gives the family
            axes.plot([entry["iteration"] for entry in history], values, marker="o", markersize=4, linewidth=1.6, color=color, label=label.replace("GPT-5.6 ", ""))
        reference = load(host, "final_ic", "pct10", reference_rows["pct10"])
        baseline = pct(reference, reference["referee_reward"])
        axes.axhline(baseline, color=baseline_color, linestyle="--", linewidth=1.4, label=f"{reference_label} (strongest baseline)")
        low, high = axes.get_ylim()
        high = low + convergence_headroom * (high - low)
        # Keep a visible strip on the far side of the baseline, so both shades show
        strip = worse_strip * (high - low)
        low, high = min(low, baseline - strip), max(high, baseline + strip)
        axes.set_ylim(low, high)
        # Faint green on the better side of the baseline and red on the worse side;
        # which side is which flips between a maximized and a minimized task
        better_above = sense == "max"
        axes.axhspan(baseline, high, color=better_color if better_above else worse_color, alpha=shade_alpha, linewidth=0, zorder=0)
        axes.axhspan(low, baseline, color=worse_color if better_above else better_color, alpha=shade_alpha, linewidth=0, zorder=0)
        axes.set_xticks(range(1, rounds_shown + 1))
        # The same number of y ticks on both hosts, whatever their reward range
        axes.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axes.set_xlabel("round")
        axes.set_ylabel("reward (%)")
        axes.legend(fontsize=legend_size - 2, loc="upper center", ncol=4, columnspacing=0.9, handlelength=1.4, handletextpad=0.4, borderpad=0.3)
        save(figure, f"coding_model_convergence_{stem}", convergence_margins)


if __name__ == "__main__":
    apply_style()
    draw_mean_bars()
    draw_convergence()
    print(f"wrote the square mean figure and two convergence panels to {out_dir}")
