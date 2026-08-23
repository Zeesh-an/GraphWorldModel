"""
Paired multi-seed comparison of ABLATION ARMS.

Not the same thing as `world_model.aggregate_seeds`, and the distinction is the
point. That module answers "how noisy is one configuration?" -- it takes N result
JSONs of the SAME arm and reports mean +/- std, plus a Pareto front over the
fidelity/cost plane. This module answers "is arm A better than arm B?" across a
grid of (arm, seed).

Both arms are trained on identical data with the same seeds, so **seed is a
blocking factor**: run-to-run variation is shared and cancels in the difference.
The honest comparison is therefore the paired within-seed difference, not two
independent means. On five runs an unpaired interval is wide enough to hide an
effect the paired test resolves at t > 80, which is not a hypothetical -- it is
what the structured-vs-linear rollout comparison does.

    python -m scripts.paired_seed_ablation \\
        --root results/seeds --arms structured linear hidew \\
        --seeds 0 1 2 3 4 --baseline structured \\
        --out results/seeds/summary.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

# t_{0.975, df} for the small df we actually hit; no scipy dependency.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}

METRICS = [
    ("test.delta_f1", "delta_f1", True),
    ("test.new_infection_f1", "new_inf_f1", True),
    ("test.brier_infected", "brier_inf", False),
    ("test.add_seed_success", "add_seed", True),
    ("test.remove_frontier_success", "rm_frontier", True),
    ("test.action_sensitivity", "act_sens", True),
    ("rollout.ens_marg_mae", "marg_mae", False),
    ("rollout.ens_count_bias", "count_bias", None),
    ("action_conditioning.counterfactual_effect.effect_mae_norm",
     "effect_mae_norm", False),
]


def dig(blob: dict, dotted: str):
    node = blob
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return float(node) if isinstance(node, (int, float)) else None


def ci95(values: np.ndarray) -> tuple[float, float]:
    """Mean and half-width of the 95% t-interval."""
    n = values.size
    if n < 2:
        return float(values.mean()), float("nan")
    sem = values.std(ddof=1) / math.sqrt(n)
    return float(values.mean()), float(T975.get(n - 1, 1.96) * sem)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Paired multi-seed ablation comparison")
    parser.add_argument("--root", type=Path, default=Path("results/seeds"))
    parser.add_argument("--arms", nargs="+",
                        default=["structured", "linear", "hidew"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--baseline", type=str, default="structured",
                        help="arm every other arm is paired against")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    runs, provenance = {}, {}
    for arm in args.arms:
        runs[arm] = {}
        for seed in args.seeds:
            path = args.root / f"{arm}_s{seed}" / "sage_IC.json"
            if not path.exists():
                print(f"  MISSING {path}")
                continue
            blob = json.loads(path.read_text())
            runs[arm][seed] = blob
            provenance.setdefault(arm, {
                "data_dir": blob["config"]["data_dir"],
                "head": blob["config"]["head"],
                "hide_edge_weights": blob["config"]["hide_edge_weights"],
                "split_mode": blob.get("split_mode"),
            })

    per_arm, paired = {}, {}

    for arm in args.arms:
        per_arm[arm] = {}
        for dotted, short, _ in METRICS:
            values = np.array(
                [v for s in args.seeds
                 if (v := dig(runs[arm].get(s, {}), dotted)) is not None],
                dtype=float,
            )
            if values.size == 0:
                continue
            mean, half = ci95(values)
            per_arm[arm][short] = {"mean": mean, "ci95": half, "n": int(values.size)}

    base = args.baseline
    for arm in args.arms:
        if arm == base:
            continue
        paired[f"{base}_minus_{arm}"] = {}
        for dotted, short, higher_better in METRICS:
            pairs = [
                (a, b) for s in args.seeds
                if (a := dig(runs[base].get(s, {}), dotted)) is not None
                and (b := dig(runs[arm].get(s, {}), dotted)) is not None
            ]
            if len(pairs) < 2:
                continue
            diff = np.array([a - b for a, b in pairs], dtype=float)
            mean, half = ci95(diff)
            sem = diff.std(ddof=1) / math.sqrt(diff.size)
            paired[f"{base}_minus_{arm}"][short] = {
                "mean_diff": mean,
                "ci95": half,
                "t": float(mean / sem) if sem else float("nan"),
                "n_pairs": int(diff.size),
                "excludes_zero": bool(abs(mean) > half) if half == half else None,
                "higher_is_better": higher_better,
            }

    blob = {
        "question": "Q2_structured_head_multi_seed",
        "design": "same data, same hyper-parameters, seeds 0-4; seed is a "
                  "blocking factor so differences are PAIRED within seed",
        "provenance": provenance,
        "seeds": args.seeds,
        "per_arm": per_arm,
        "paired_vs_" + base: paired,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(blob, indent=2))

    print("\nprovenance")
    for arm, info in provenance.items():
        print(f"  {arm:12s} head={info['head']:10s} "
              f"hide_w={str(info['hide_edge_weights']):5s} {info['data_dir']}")

    shorts = [s for _, s, _ in METRICS if s in per_arm.get(args.arms[0], {})]
    print(f"\n{'metric':>16s} " + " ".join(f"{a:>21s}" for a in args.arms))
    print("-" * (17 + 22 * len(args.arms)))
    for short in shorts:
        cells = []
        for arm in args.arms:
            cell = per_arm[arm].get(short)
            cells.append(f"{cell['mean']:12.4f}±{cell['ci95']:<8.4f}"
                         if cell else f"{'--':>21s}")
        print(f"{short:>16s} " + " ".join(cells))

    for key, table in paired.items():
        print(f"\npaired {key} (n={next(iter(table.values()))['n_pairs']})")
        print(f"{'metric':>16s} {'diff':>12s} {'95% CI':>12s} {'t':>8s} "
              f"{'excl. 0':>8s} {'favours':>10s}")
        print("-" * 72)
        for short, cell in table.items():
            better = cell["higher_is_better"]
            if better is None:
                favours = "--"
            else:
                wins = (cell["mean_diff"] > 0) == better
                favours = base if wins else key.split("_minus_")[1]
                if not cell["excludes_zero"]:
                    favours = "(ns)"
            print(f"{short:>16s} {cell['mean_diff']:+12.4f} "
                  f"±{cell['ci95']:<11.4f} {cell['t']:8.2f} "
                  f"{str(cell['excludes_zero']):>8s} {favours:>10s}")

    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
