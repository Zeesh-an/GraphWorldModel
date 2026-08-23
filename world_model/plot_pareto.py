"""
Fidelity-vs-cost Pareto figure over a set of train_wm.py results JSONs.

Reads the same `pareto.point_from_results` view the aggregator uses, so the figure
and the JSON front can never disagree. Dominated points are drawn hollow and the
front is joined by a step line -- a smooth interpolation between two Pareto points
would assert configurations that were never run.

    python -m world_model.plot_pareto <dir>/*.json --out pareto.png \\
        --fidelity ens_marg_mae --cost train_seconds
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from world_model.pareto import (
    cost_objectives,
    fidelity_cost_front,
    fidelity_objectives,
    point_from_results,
)

# Colour-blind-safe, and distinguishable in greyscale print
front_colour = "#0072B2"
dominated_colour = "#999999"
annotation_offset = (6, 6)


def build_points(paths: list[Path]) -> list:
    points = []

    for path in paths:
        results = json.loads(path.read_text())
        config = results.get("config", {})
        # Name by what actually varies, so the legend is readable without the path
        name = "/".join(
            str(part)
            for part in (
                config.get("model", path.stem),
                config.get("head", ""),
                "w-hidden" if config.get("hide_edge_weights") else "",
                f"s{config.get('seed')}" if config.get("seed") is not None else "",
            )
            if part
        )
        points.append(point_from_results(name or path.stem, results, meta=config))

    return points


def plot(points: list, fidelity: str, cost: str, out: Path, title: str | None) -> dict:
    front = fidelity_cost_front(points, fidelity, cost)
    on_front = set(front["front"])

    fidelity_objective = fidelity_objectives[fidelity]
    cost_objective = cost_objectives[cost]

    usable = [
        point
        for point in points
        if fidelity in point.values and cost in point.values
    ]
    skipped = [point.name for point in points if point not in usable]

    figure, axes = plt.subplots(figsize=(7.5, 5.2), dpi=160)

    for point in usable:
        x = point.values[cost]
        y = point.values[fidelity]
        is_front = point.name in on_front

        axes.scatter(
            x,
            y,
            s=110 if is_front else 70,
            facecolors=front_colour if is_front else "none",
            edgecolors=front_colour if is_front else dominated_colour,
            linewidths=1.8,
            zorder=3 if is_front else 2,
            label="_nolegend_",
        )
        axes.annotate(
            point.name,
            (x, y),
            textcoords="offset points",
            xytext=annotation_offset,
            fontsize=8,
            color="#222222" if is_front else dominated_colour,
        )

    # Step line through the front: a smooth curve would assert unrun configurations
    front_points = sorted(
        (point.values[cost], point.values[fidelity])
        for point in usable
        if point.name in on_front
    )
    if len(front_points) > 1:
        axes.step(
            [x for x, _ in front_points],
            [y for _, y in front_points],
            where="post",
            color=front_colour,
            linewidth=1.4,
            alpha=0.55,
            zorder=1,
        )

    axes.set_xlabel(f"{cost_objective.label or cost}  (lower is better) →")
    axes.set_ylabel(f"{fidelity_objective.label or fidelity}  (lower is better) →")
    axes.set_title(
        title or f"Pareto front: {fidelity_objective.label} vs {cost_objective.label}",
        fontsize=11,
    )
    axes.grid(alpha=0.25, linewidth=0.6)
    axes.spines[["top", "right"]].set_visible(False)

    handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="", markerfacecolor=front_colour,
            markeredgecolor=front_colour, markersize=9, label="on the Pareto front"
        ),
        plt.Line2D(
            [], [], marker="o", linestyle="", markerfacecolor="none",
            markeredgecolor=dominated_colour, markersize=8, label="dominated"
        ),
    ]
    axes.legend(handles=handles, frameon=False, fontsize=9, loc="best")

    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)

    front["skipped_missing_axes"] = skipped
    return front


def main() -> None:
    parser = argparse.ArgumentParser(description="Fidelity/cost Pareto figure")
    parser.add_argument("results", nargs="+", help="train_wm.py results JSON paths")
    parser.add_argument("--out", type=str, default="pareto.png")
    parser.add_argument(
        "--fidelity", type=str, default="ens_marg_mae", choices=sorted(fidelity_objectives)
    )
    parser.add_argument(
        "--cost", type=str, default="train_seconds", choices=sorted(cost_objectives)
    )
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument(
        "--front-json", type=str, default=None, help="also write the front as JSON"
    )
    args = parser.parse_args()

    points = build_points([Path(path) for path in args.results])
    front = plot(points, args.fidelity, args.cost, Path(args.out), args.title)

    print(json.dumps(front, indent=2))
    print(f"[pareto] figure -> {args.out}")

    if args.front_json:
        Path(args.front_json).write_text(json.dumps(front, indent=2))
        print(f"[pareto] front -> {args.front_json}")


if __name__ == "__main__":
    main()
