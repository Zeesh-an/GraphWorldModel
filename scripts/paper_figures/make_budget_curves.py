"""
The main table's IC column of one dataset per task as a budget curve, drawn the
way pipeline.plots.plot_budget_vs_spread draws the report's figure (same canvas,
markers, grid, legend size and tight bounding box). Numbers are typed from
tab:results_main so the figure and the table cannot disagree; a None is the
table's dash.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from pipeline.plots import _arm_style, figure_dpi, figure_formats, figure_size
from scripts.paper_figures.paper_style import apply_style, baseline_shades, ours_color

out_dir = Path("figures/budget_curves")
ours_label = "Ours (Network World Model)"
# Salmon shades for the baselines; distinct markers keep each one identifiable
baseline_colors = baseline_shades
ours_style = {"linewidth": 3.8, "alpha": 1.0, "zorder": 5, "markersize": 8}
baseline_style = {"linewidth": 1.4, "alpha": 0.9, "zorder": 3, "markersize": 5}
# The panels print at a quarter of the text width, so the legend is set well
# above the report's size to stay readable; it is drawn beneath the curves and
# semi-transparent so a curve it overlaps stays visible
legend_size = 12
legend_style = {"frameon": True, "edgecolor": "none", "framealpha": 0.75, "handlelength": 1.8, "markerscale": 1.2, "borderpad": 0.3, "labelspacing": 0.25, "handletextpad": 0.5}

panels = {
    "im_digg_ic": {
        "budgets": [1, 5, 10, 20],
        "xlabel": "seed budget (% of nodes)",
        "ylabel": "spread (% of nodes activated)",
        "legend": {"loc": "upper left", "fontsize": legend_size},
        "headroom": ("top", 1.04),
        "rows": {
            "IMM": [25.9, 36.0, 44.2, 55.7],
            "OPIM": [27.4, 41.6, 51.8, 62.7],
            "SubSIM": [27.3, 41.7, 51.8, 62.7],
            "DegreeDiscount": [26.0, 37.0, 45.8, 60.0],
            ours_label: [30.4, 45.8, 56.5, 69.4],
        },
    },
    "aim_digg_ic": {
        "budgets": [1, 5, 10, 20],
        "xlabel": "seed budget (% of nodes)",
        "ylabel": "spread (% of nodes activated)",
        "legend": {"loc": "upper left", "fontsize": legend_size},
        "headroom": ("top", 1.04),
        "rows": {
            "EPIC": [27.5, 40.0, 49.0, 60.7],
            "Adaptive DegreeDiscount": [29.0, 43.5, 54.8, 69.3],
            "IMM": [25.9, 36.0, 44.2, 55.7],
            "Static-Split": [26.4, 38.7, 49.0, 65.1],
            ours_label: [30.4, 45.6, 56.4, 71.9],
        },
    },
    "cnd_pgp_ic": {
        "budgets": [1, 5, 10, 20],
        "xlabel": "removal budget (% of nodes)",
        "ylabel": "remaining spread (% of nodes infected)",
        "legend": {"loc": "lower left", "fontsize": legend_size},
        "headroom": ("bottom", 1.04),
        "rows": {
            "HDA": [26.4, 23.3, 20.8, 18.0],
            "BPD+R": [26.4, None, 21.2, 18.0],
            "CI+R": [26.6, 23.4, 21.2, 18.1],
            "EI": [26.5, 23.5, 21.1, 18.2],
            "Frontier": [26.7, 23.4, 20.2, 14.2],
            ours_label: [25.8, 21.3, 16.5, 11.0],
        },
    },
    "ib_gnutella24_ic": {
        "budgets": [10, 20, 40, 50],
        "xlabel": "blocker budget $k$",
        "ylabel": "rumor cascade size (nodes)",
        # Between the Reverse and Proximity curves on the right, the one empty band
        "legend": {"loc": "upper right", "bbox_to_anchor": (0.995, 0.85), "fontsize": legend_size - 1},
        "headroom": ("top", 1.0),
        "rows": {
            "RPS": [1520.6, 1311.6, 1029.6, 939.8],
            "Reverse": [1817.6, 1713.1, 1462.0, 1413.1],
            "Proximity": [2338.9, 2289.9, 2196.5, 2141.1],
            "GreedyReplace": [1551.8, 1364.9, 1088.2, 990.4],
            "SandIMIN": [1564.9, 1391.2, 1106.4, 1017.8],
            ours_label: [1503.3, 1275.0, 985.6, 896.0],
        },
    },
}


def draw_panel(name: str, panel: dict) -> None:
    figure, axes = plt.subplots(figsize=figure_size)
    budgets = panel["budgets"]

    for index, (label, values) in enumerate(panel["rows"].items()):
        ours = label == ours_label
        # A dash in the table becomes a gap in the line rather than a guessed point
        y_values = [np.nan if value is None else value for value in values]
        axes.plot(
            budgets,
            y_values,
            label=label,
            color=ours_color if ours else baseline_colors[index],
            marker=_arm_style(index)["marker"],
            **(ours_style if ours else baseline_style),
        )

    axes.set_xticks(budgets)
    # Almost no margin above the top point or below the bottom one, so the gaps
    # between the curves fill the axes; the legend's side gets a small extra margin
    axes.margins(y=0.015)
    side, factor = panel["headroom"]
    low, high = axes.get_ylim()
    if side == "top":
        axes.set_ylim(low, low + factor * (high - low))
    else:
        axes.set_ylim(high - factor * (high - low), high)
    axes.set_xlabel(panel["xlabel"])
    axes.set_ylabel(panel["ylabel"])
    legend = axes.legend(**legend_style, **panel["legend"])
    legend.set_zorder(2)
    figure.tight_layout()

    for suffix in figure_formats:
        figure.savefig(out_dir / f"{name}.{suffix}", dpi=figure_dpi, bbox_inches="tight")

    plt.close(figure)


if __name__ == "__main__":
    apply_style()
    for name, panel in panels.items():
        draw_panel(name, panel)
        print(f"wrote {out_dir / name}.pdf")
