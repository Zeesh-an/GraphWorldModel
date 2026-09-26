"""
Two panels from the transfer table (tab:transfer), one task each: the program
designed on one network and replayed unchanged on the task's other networks,
against the strongest baseline of each target. Rewards are typed from the table
(the strongest-baseline values from the main and appendix tables for the same
cells) and averaged over the four budgets as per-budget ratios, so each bar is
relative to the strongest baseline of its own network (100).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from pipeline.plots import figure_dpi, figure_formats
from scripts.paper_figures.paper_style import apply_style, baseline_color, ours_color

out_dir = Path("figures/transfer")
# (stem, source network, sense, direction note, [(target, transferred rewards, strongest-baseline rewards) per target])
panels = [
    ("im", "Digg", "max", "higher reward is better", [
        ("Network Science", [8.9, 25.3, 39.1, 60.0], [8.8, 24.8, 38.0, 59.2]),
        ("NetHEPT", [12.9, 33.0, 47.8, 68.6], [12.6, 31.4, 44.8, 64.5]),
    ]),
    ("cnd", "PGP", "min", "lower reward is better", [
        ("Power Grid", [26.2, 22.1, 17.3, 10.2], [26.3, 22.7, 18.7, 10.7]),
        ("Gnutella31", [31.1, 24.7, 19.5, 12.3], [31.3, 26.0, 21.6, 16.0]),
    ]),
]
panel_size = (3.4, 2.7)
bar_width = 0.34
# One axis for both panels so the two tasks read on the same scale
axis_floor, axis_top = 85.0, 116.5
font_size = 9


def relative(sense: str, transferred: list[float], baseline: list[float]) -> float:
    # Mean over budgets of the per-budget ratio, sign-corrected so above 100 is better
    ratios = [(moved / reference) if sense == "max" else (reference / moved) for moved, reference in zip(transferred, baseline, strict=True)]
    return 100.0 * float(np.mean(ratios))


def draw_panel(stem: str, source: str, sense: str, direction: str, targets: list) -> None:
    figure, axes = plt.subplots(figsize=panel_size)
    positions = np.arange(len(targets))
    values = [relative(sense, transferred, baseline) for _, transferred, baseline in targets]
    for index, ((target, _, _), value) in enumerate(zip(targets, values, strict=True)):
        axes.bar(index - bar_width / 2, value, bar_width, color=ours_color, zorder=3, label=f"designed on {source}" if index == 0 else None)
        axes.bar(index + bar_width / 2, 100.0, bar_width, color=baseline_color, zorder=3, label="strongest baseline" if index == 0 else None)
        axes.text(index - bar_width / 2, value + 0.5, f"+{value - 100:.1f}%", ha="center", va="bottom", fontsize=font_size - 1, color=ours_color, fontweight="bold")
    axes.set_xticks(positions)
    axes.set_xticklabels([target for target, _, _ in targets], fontsize=font_size)
    axes.set_ylim(axis_floor, axis_top)
    axes.set_yticks([85, 90, 95, 100, 105, 110, 115])
    axes.set_ylabel("reward relative to the\nstrongest baseline (%)", fontsize=font_size)
    axes.tick_params(axis="y", labelsize=font_size - 1)
    axes.tick_params(axis="x", length=0)
    axes.grid(alpha=0.3, axis="y", zorder=0)
    axes.set_axisbelow(True)
    # The legend hangs below the axes so nothing inside the plot is covered
    axes.legend(fontsize=font_size - 2, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, handlelength=1.2, columnspacing=1.2)
    figure.subplots_adjust(left=0.2, right=0.98, bottom=0.12, top=0.98)
    for suffix in figure_formats:
        figure.savefig(out_dir / f"transfer_{stem}.{suffix}", dpi=figure_dpi, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    apply_style()
    for stem, source, sense, direction, targets in panels:
        draw_panel(stem, source, sense, direction, targets)
        print(f"wrote {out_dir / f'transfer_{stem}'}.pdf")
