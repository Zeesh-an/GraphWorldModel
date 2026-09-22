"""
Compare the world-model backbones of one host: a markdown report plus figures.

    python -m scripts.compare_backbones --host influence_maximization/netscience \
        --runs sage=abl_wm_sage gat=abl_wm_gat gcn=abl_wm_gcn gcnii=abl_wm_gcnii gt=abl_wm_gt

Each run is `results/<host>/<run>/world_model/<backbone>_IC.json`, the results file
`train_wm.py` writes. The report and the figures go to results/<host>/backbone_comparison/
(or --out-dir), the figures as PNG and PDF, and the markdown links the PNGs.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts.compare_world_models import horizon_steps, table_row  # noqa: E402

figure_dpi = 200
figure_formats = ("png", "pdf")
colors = ("#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860")
report_name = "comparison.md"


def save(figure: plt.Figure, out_dir: Path, stem: str) -> str:
    figure.tight_layout()
    for suffix in figure_formats:
        figure.savefig(out_dir / f"{stem}.{suffix}", dpi=figure_dpi, bbox_inches="tight")
    plt.close(figure)

    return f"{stem}.png"


def bar_figure(labels: list[str], values: list[float | None], ylabel: str, title: str, out_dir: Path, stem: str, zero_line: bool = False) -> str:
    figure, axes = plt.subplots(figsize=(6, 3.8))
    heights = [0.0 if value is None else value for value in values]
    bars = axes.bar(labels, heights, color=colors[: len(labels)], width=0.6)

    for bar, value in zip(bars, values, strict=True):
        text = "n/a" if value is None else f"{value:.3f}"
        axes.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), text, ha="center", va="bottom", fontsize=9)

    if zero_line:
        axes.axhline(0.0, color="black", linewidth=0.8)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.grid(alpha=0.3, axis="y")

    return save(figure, out_dir, stem)


def curve_figure(runs: dict[str, dict], out_dir: Path) -> str:
    """Rollout count against the simulator's, per step, one line per backbone."""
    figure, axes = plt.subplots(figsize=(6.5, 4))

    true_curve = None
    for (label, results), color in zip(runs.items(), colors, strict=False):
        rollout = results.get("rollout", {})
        model = rollout.get("count_model_curve") or []
        if not model:
            continue
        axes.plot(range(1, len(model) + 1), model, marker="o", markersize=3, color=color, label=label)
        true_curve = rollout.get("count_true_curve") or true_curve

    if true_curve:
        axes.plot(range(1, len(true_curve) + 1), true_curve, color="black", linestyle="--", linewidth=1.2, label="simulator")

    axes.set_xlabel("rollout step")
    axes.set_ylabel("mean infected count")
    axes.set_title("Free-running rollout against the simulator")
    axes.legend(fontsize=8)
    axes.grid(alpha=0.3)

    return save(figure, out_dir, "rollout_curves")


def bias_figure(runs: dict[str, dict], out_dir: Path) -> str:
    """Relative count bias by step: flat means one-step error does not compound."""
    figure, axes = plt.subplots(figsize=(6.5, 4))

    for (label, results), color in zip(runs.items(), colors, strict=False):
        rollout = results.get("rollout", {})
        model = np.asarray(rollout.get("count_model_curve") or [], dtype=float)
        true = np.asarray(rollout.get("count_true_curve") or [], dtype=float)
        if not model.size:
            continue
        keep = true > 0
        steps = np.arange(1, len(model) + 1)[keep]
        axes.plot(steps, 100.0 * (model[keep] - true[keep]) / true[keep], marker="o", markersize=3, color=color, label=label)

    axes.axhline(0.0, color="black", linewidth=0.8)
    axes.set_xlabel("rollout step")
    axes.set_ylabel("relative count bias (%)")
    axes.set_title("Rollout bias by step")
    axes.legend(fontsize=8)
    axes.grid(alpha=0.3)

    return save(figure, out_dir, "bias_by_step")


def training_figure(runs: dict[str, dict], out_dir: Path) -> str:
    """Validation delta F1 per epoch, which shows how fast each backbone converges."""
    figure, axes = plt.subplots(figsize=(6.5, 4))

    for (label, results), color in zip(runs.items(), colors, strict=False):
        history = results.get("history") or []
        if not history:
            continue
        axes.plot([entry["epoch"] for entry in history], [entry["val_delta_f1"] for entry in history], color=color, label=label)

    axes.set_xlabel("epoch")
    axes.set_ylabel("validation delta F1")
    axes.set_title("Training curves")
    axes.legend(fontsize=8)
    axes.grid(alpha=0.3)

    return save(figure, out_dir, "training_curves")


def calibration_figure(runs: dict[str, dict], out_dir: Path) -> str:
    """Reliability diagram of the next-infected probability, one line per backbone."""
    figure, axes = plt.subplots(figsize=(5, 5))

    for (label, results), color in zip(runs.items(), colors, strict=False):
        calibration = results.get("test", {}).get("calibration_infected")
        if not calibration:
            continue
        predicted = np.asarray(calibration["mean_predicted"], dtype=float)
        target = np.asarray(calibration["mean_target"], dtype=float)
        keep = ~np.isnan(predicted) & ~np.isnan(target)
        axes.plot(predicted[keep], target[keep], marker="o", markersize=4, color=color, label=f"{label} (ECE {calibration['ece']:.4f})")

    axes.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.0, label="perfect")
    axes.set_xlabel("mean predicted P(infected)")
    axes.set_ylabel("mean observed")
    axes.set_title("Calibration on the test split")
    axes.legend(fontsize=8)
    axes.grid(alpha=0.3)

    return save(figure, out_dir, "calibration")


def effect_figure(runs: dict[str, dict], out_dir: Path) -> str:
    """Predicted action effect against the simulator's, per op, one bar group per backbone."""
    labels = list(runs)
    ops = sorted({op for results in runs.values() for op in (results.get("action_conditioning", {}).get("counterfactual_effect", {}).get("per_op") or {})})
    if not ops:
        ops = ["all"]

    figure, axes = plt.subplots(figsize=(6.5, 4))
    width = 0.8 / len(ops)
    positions = np.arange(len(labels))

    for index, op in enumerate(ops):
        values = []
        for results in runs.values():
            effect = results.get("action_conditioning", {}).get("counterfactual_effect", {})
            per_op = (effect.get("per_op") or {}).get(op, effect if op == "all" else {})
            values.append(per_op.get("effect_pearson") or 0.0)
        axes.bar(positions + (index - (len(ops) - 1) / 2) * width, values, width, label=op, color=colors[index % len(colors)])

    axes.set_xticks(positions)
    axes.set_xticklabels(labels)
    axes.set_ylabel("Pearson r, predicted against true effect")
    axes.set_title("Action effect fidelity per op")
    axes.set_ylim(0, 1.05)
    axes.legend(fontsize=8)
    axes.grid(alpha=0.3, axis="y")

    return save(figure, out_dir, "action_effect")


def metric(results: dict, *keys: str) -> float | None:
    value = results
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare world-model backbones: markdown report and figures")
    parser.add_argument("--host", type=str, required=True, help="task/dataset under results/, e.g. influence_maximization/netscience (default: None).")
    parser.add_argument("--runs", type=str, nargs="+", required=True, help="label=run pairs; the file read is results/<host>/<run>/world_model/<label>_IC.json (default: None).")
    parser.add_argument("--diffusion-model", type=str, default="IC", help="dynamics in the results file name (default: IC).")
    parser.add_argument("--out-dir", type=Path, default=None, help="where the report and figures go (default: results/<host>/backbone_comparison).")
    args = parser.parse_args()

    host = Path("results") / args.host
    out_dir = args.out_dir or host / "backbone_comparison"
    os.makedirs(out_dir, exist_ok=True)

    runs = {}
    for pair in args.runs:
        label, run = pair.split("=", 1)
        runs[label] = json.loads((host / run / "world_model" / f"{label}_{args.diffusion_model}.json").read_text())
    labels = list(runs)

    figures = {
        "One-step delta F1 (higher is better)": bar_figure(labels, [metric(r, "test", "delta_f1") for r in runs.values()], "delta F1", "One-step delta F1", out_dir, "delta_f1"),
        "Brier score on next infected (lower is better)": bar_figure(labels, [metric(r, "test", "brier_infected") for r in runs.values()], "Brier", "Brier score, next infected", out_dir, "brier"),
        "Final rollout count bias (closer to zero is better)": bar_figure(
            labels,
            [None if not metric(r, "rollout", "ens_final_count_true") else 100.0 * metric(r, "rollout", "ens_count_bias") / metric(r, "rollout", "ens_final_count_true") for r in runs.values()],
            "relative bias (%)", "Final count bias of the free-running rollout", out_dir, "final_count_bias", zero_line=True,
        ),
        "Rollout marginal MAE (lower is better)": bar_figure(labels, [metric(r, "rollout", "ens_marg_mae") for r in runs.values()], "MAE of P(infected)", "Rollout marginal MAE", out_dir, "marginal_mae"),
        "Delta F1 drop with the action channels zeroed (higher means the model uses the action)": bar_figure(labels, [metric(r, "action_conditioning", "ablation", "null_delta_f1_drop") for r in runs.values()], "delta F1 drop", "Action dependence", out_dir, "action_dependence"),
        "Training time (minutes)": bar_figure(labels, [None if r.get("train_seconds") is None else r["train_seconds"] / 60.0 for r in runs.values()], "minutes", "Training time", out_dir, "train_minutes"),
        "Rollout against the simulator": curve_figure(runs, out_dir),
        "Rollout bias by step": bias_figure(runs, out_dir),
        "Training curves": training_figure(runs, out_dir),
        "Calibration": calibration_figure(runs, out_dir),
        "Action effect per op": effect_figure(runs, out_dir),
    }

    header = ["model", "delta F1", "Brier", "final count bias", "marginal MAE", *[f"bias @ {step}" for step in horizon_steps], "delta F1 drop, actions zeroed", "effect Pearson", "train min"]
    lines = [
        f"# World-model backbones on {args.host}",
        "",
        "Every checkpoint is trained on the same transitions and evaluated on the same test split; only the encoder differs. "
        "Delta F1 and Brier are one-step metrics on the nodes whose state changes; the rollout columns come from a free-running sampled rollout against the simulator; "
        "the action columns come from zeroing the action channels at evaluation and from the predicted effect of an action against the simulator's counterfactual forks.",
        "",
        f"| {' | '.join(header)} |",
        f"| {' | '.join('---' for _ in header)} |",
        *[f"| {' | '.join(table_row(label, results))} |" for label, results in runs.items()],
        "",
        "## Configuration",
        "",
        "| model | hidden | layers | heads | action conditioning | best val delta F1 | epochs run |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label, results in runs.items():
        config = results.get("config", {})
        lines.append(
            f"| {label} | {config.get('hidden_dim')} | {config.get('n_layers')} | {config.get('n_heads') if label in ('gat', 'gt') else '-'} | "
            f"{config.get('action_conditioning') or 'none'} | {results.get('best_val_delta_f1', float('nan')):.4f} | {len(results.get('history') or [])} |"
        )
    lines += ["", "## Figures", ""]
    for caption, file_name in figures.items():
        lines += [f"### {caption}", "", f"![{caption}]({file_name})", ""]

    (out_dir / report_name).write_text("\n".join(lines))
    print(f"wrote {out_dir / report_name} and {len(figures)} figures (png and pdf) to {out_dir}")
