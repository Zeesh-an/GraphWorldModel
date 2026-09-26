"""
The one visual style every paper figure uses: a cornflower blue for the Network
World Model against salmon shades for the baselines, set in Times to match the
body text, with no top or right spine and a faint grid. Every generator imports
from here, so a change of palette is one edit.
"""

import matplotlib
import matplotlib.pyplot as plt

ours_color = "#4C7FE6"
# Lighter blues for other variants of ours (the coding models after the main one)
ours_shades = ("#4C7FE6", "#86A6EE", "#BACCF6")
# The single-baseline color, and a set of distinct muted hues for plots with
# several baselines: salmon first, so a figure with one baseline keeps it, then
# amber, plum, sage and rose, none of them blue so ours stays the only cool color
baseline_color = "#F4A088"
baseline_shades = ("#F4A088", "#E9B75C", "#A98BC9", "#8FBF9A", "#D97A9A", "#C0664A")
text_color = "#333333"
spine_color = "#555555"
grid_alpha = 0.25


def apply_style() -> None:
    # TrueType fonts in the PDFs, as the pipeline's own figures use
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": spine_color,
        "axes.labelcolor": text_color,
        "xtick.color": text_color,
        "ytick.color": text_color,
        "axes.grid": True,
        # Horizontal lines only: vertical ones cut through bars and add clutter to lines
        "axes.grid.axis": "y",
        "grid.alpha": grid_alpha,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
    })
