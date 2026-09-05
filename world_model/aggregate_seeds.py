"""
Aggregate several single-seed results JSONs into mean +/- std, and build the
Pareto front over them.

The published numbers are a single seed (42) on one graph family. A single seed
cannot tell a real 0.025 improvement from run-to-run noise, which is exactly the
gap the current planning result sits in (model 0.244 vs degree 0.269). This turns
N runs into a table that can: every scalar gets a mean, a std, and n, and any
comparison the report makes can be checked against the spread.

    # train the same config under several seeds
    for s in 1 2 3 4 5; do
      python -m world_model.train_wm --data-dir <data> --seed $s \\
          --results <dir>/sage_IC_seed$s.json ...
    done

    python -m world_model.aggregate_seeds <dir>/sage_IC_seed*.json \\
        --out <dir>/sage_IC_aggregate.json

    # or compare configurations on the fidelity/cost plane
    python -m world_model.aggregate_seeds <dir>/*.json --pareto \\
        --fidelity ens_marg_mae --cost train_seconds
"""

import argparse
import json
from pathlib import Path
import numpy as np

from world_model.pareto import (
    cost_objectives,
    fidelity_cost_front,
    fidelity_objectives,
    point_from_results,
)

# Result blocks whose scalars are worth aggregating. `history` is per-epoch and
# `config` is not numeric, so both are excluded rather than silently mangled.
aggregated_blocks = (
    "test",
    "rollout",
    "planning",
    "planning_budget",
)

# Top-level scalars outside any block
aggregated_scalars = ("best_val_delta_f1", "train_seconds")


def _flatten(results: dict) -> dict[str, float]:
    """Pull every aggregatable scalar out of one results JSON, as dotted keys."""
    flat = {}

    for key in aggregated_scalars:
        value = results.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[key] = float(value)

    for block in aggregated_blocks:
        for key, value in (results.get(block) or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                flat[f"{block}.{key}"] = float(value)

    # The action-conditioning verdict is a bool per run; aggregate it as the
    # fraction of seeds that passed, because "4 of 5 seeds pass" is a materially
    # different claim from "it passes"
    conditioning = results.get("action_conditioning") or {}
    if "action_conditioned" in conditioning:
        flat["action_conditioning.pass"] = float(
            bool(conditioning["action_conditioned"])
        )

    for key, value in (conditioning.get("counterfactual_effect") or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[f"action_conditioning.{key}"] = float(value)

    return flat


def aggregate(paths: list[Path]) -> dict:
    """mean / std / n / min / max for every scalar present in at least one run."""
    runs = [json.loads(path.read_text()) for path in paths]
    flats = [_flatten(run) for run in runs]

    keys = sorted({key for flat in flats for key in flat})
    summary = {}

    for key in keys:
        values = np.array(
            [flat[key] for flat in flats if key in flat and np.isfinite(flat[key])],
            dtype=np.float64,
        )

        if values.size == 0:
            continue

        summary[key] = {
            "mean": float(values.mean()),
            # Sample std (ddof=1): with n seeds we are estimating the spread of the
            # procedure, not describing this particular set of runs
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "n": int(values.size),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    seeds = [run.get("config", {}).get("seed") for run in runs]

    return {
        "n_runs": len(runs),
        "seeds": seeds,
        "sources": [str(path) for path in paths],
        "metrics": summary,
        "warning": (
            "single run: std is 0 by construction and no comparison here is "
            "supported against noise"
            if len(runs) < 2
            else None
        ),
    }


def separated(summary: dict, first: str, second: str) -> dict | None:
    """Is `first` separated from `second` by more than the two spreads combined?

    The check the current planning table needs and does not have. Reports the gap
    against the pooled standard error rather than declaring a winner, so a
    difference inside the noise reads as inside the noise.
    """
    metrics = summary.get("metrics", {})

    if first not in metrics or second not in metrics:
        return None

    left, right = metrics[first], metrics[second]
    gap = left["mean"] - right["mean"]
    pooled = float(
        np.sqrt(
            left["std"] ** 2 / max(left["n"], 1) + right["std"] ** 2 / max(right["n"], 1)
        )
    )

    return {
        "first": first,
        "second": second,
        "gap": gap,
        "pooled_se": pooled,
        # Not a p-value: with 3-5 seeds a t-test would be theatre. This is the raw
        # ratio, and the caller can see for itself whether the gap clears the noise.
        "gap_over_se": float(gap / pooled) if pooled > 0 else float("inf"),
        "separated": bool(pooled > 0 and abs(gap) > 2.0 * pooled),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate world-model results JSONs across seeds, "
        "and/or build a Pareto front over configurations"
    )
    parser.add_argument("results", nargs="+", help="results JSON paths (default: required).")
    parser.add_argument(
        "--out", type=str, default=None, help="write the aggregate JSON here (default: None)."
    )
    parser.add_argument(
        "--pareto",
        action="store_true",
        help="treat each input as a distinct CONFIGURATION and report the "
        "fidelity/cost Pareto front instead of a seed aggregate (default: False).",
    )
    parser.add_argument(
        "--fidelity",
        type=str,
        default="ens_marg_mae",
        choices=sorted(fidelity_objectives),
        help="fidelity axis for --pareto (default: ens_marg_mae).",
    )
    parser.add_argument(
        "--cost",
        type=str,
        default="train_seconds",
        choices=sorted(cost_objectives),
        help="cost axis for --pareto (default: train_seconds).",
    )
    args = parser.parse_args()

    paths = [Path(path) for path in args.results]

    if args.pareto:
        points = [
            point_from_results(path.stem, json.loads(path.read_text()))
            for path in paths
        ]
        output = fidelity_cost_front(points, args.fidelity, args.cost)
    else:
        output = aggregate(paths)

    text = json.dumps(output, indent=2)
    print(text)

    if args.out:
        Path(args.out).write_text(text)
        print(f"[aggregate] -> {args.out}")
