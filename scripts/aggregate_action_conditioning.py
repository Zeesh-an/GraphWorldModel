"""
Experiment 1's results table, from the sweep's per-run JSONs.

    python -m scripts.aggregate_action_conditioning \
        --results experiments/ba24/wm --out experiments/ba24/action_conditioning.json

Reads every `<setting>_<arm>_s<seed>.json` written by
`scripts/run_action_conditioning_sweep.sh`, groups by (setting, arm), and reports
mean ± SE over seeds for each metric family the brief asks for:

  one-step        delta_f1, new_infection_f1, brier_infected, ece_infected
  rollout         ens_marg_mae, ens_count_w1, |ens_count_bias|, final-count gap,
                  and the marginal-error-vs-horizon curve
  action          action_sensitivity, shuffle/null delta_f1 drop
  counterfactual  effect_pearson, effect_mae_norm, effect_sign_agree
  downstream      plan_regret_model (and the degree / random references)

Nothing here decides a winner. It prints the arm difference beside the pooled
standard error of that difference, because at three seeds most of these gaps do
not clear it and a table that hid that would be the whole problem.
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

run_name = re.compile(r"^(?P<setting>[A-Za-z]+)_(?P<arm>none|message_blind|global|message)_s(?P<seed>\d+)$")

arm_order = ("none", "message_blind", "global", "message")

#: (label, dotted path into the results JSON, lower_is_better)
metrics = (
    ("delta_f1", "test.delta_f1", False),
    ("new_infection_f1", "test.new_infection_f1", False),
    ("brier_infected", "test.brier_infected", True),
    ("ece_infected", "test.ece_infected", True),
    ("action_sensitivity", "test.action_sensitivity", False),
    ("ens_marg_mae", "rollout.ens_marg_mae", True),
    ("ens_count_w1", "rollout.ens_count_w1", True),
    ("abs_count_bias", "rollout.ens_count_bias", True),
    ("effect_pearson", "action_conditioning.counterfactual_effect.effect_pearson", False),
    ("effect_mae_norm", "action_conditioning.counterfactual_effect.effect_mae_norm", True),
    ("effect_sign_agree", "action_conditioning.counterfactual_effect.effect_sign_agree", False),
    ("shuffle_delta_f1_drop", "action_conditioning.ablation.shuffle_delta_f1_drop", False),
    ("plan_regret_model", "planning.plan_regret_model", True),
    ("plan_regret_degree", "planning.plan_regret_degree", True),
    ("epochs_run", "_epochs", False),
)


def dig(blob: dict, path: str):
    if path == "_epochs":
        return float(len(blob.get("history", [])))

    node = blob
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]

    return float(node) if isinstance(node, (int, float)) else None


def summarize(values: list[float]) -> dict:
    array = np.array([value for value in values if value is not None], dtype=float)

    if not array.size:
        return {"n": 0, "mean": None, "sd": None, "se": None}

    sd = float(array.std(ddof=1)) if array.size > 1 else 0.0

    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sd": sd,
        "se": sd / math.sqrt(array.size) if array.size > 1 else 0.0,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--baseline-arm",
        type=str,
        default="none",
        help="arm every other arm's difference is reported against (default: none).",
    )
    args = parser.parse_args(argv)

    runs = defaultdict(list)
    curves = defaultdict(list)

    for path in sorted(args.results.glob("*.json")):
        match = run_name.match(path.stem)

        if match is None:
            continue

        blob = json.loads(path.read_text())
        key = (match["setting"], match["arm"])
        runs[key].append((int(match["seed"]), blob))

        curve = blob.get("rollout", {}).get("marginal_mae_curve")
        if curve:
            curves[key].append(curve)

    settings = sorted({setting for setting, _ in runs})
    table = {}

    for setting in settings:
        table[setting] = {}
        for arm in arm_order:
            entries = runs.get((setting, arm), [])

            if not entries:
                continue

            block = {
                "seeds": sorted(seed for seed, _ in entries),
                "metrics": {
                    label: summarize(
                        [
                            (
                                abs(dig(blob, path))
                                if label == "abs_count_bias" and dig(blob, path) is not None
                                else dig(blob, path)
                            )
                            for _, blob in entries
                        ]
                    )
                    for label, path, _ in metrics
                },
            }

            group = curves.get((setting, arm), [])
            if group:
                width = min(len(curve) for curve in group)
                block["marginal_mae_curve"] = [
                    float(np.mean([curve[step] for curve in group]))
                    for step in range(width)
                ]

            # Final-count gap: |model - true| at the rollout endpoint, which the
            # two raw counts do not show on their own
            gaps = [
                abs(
                    blob["rollout"]["ens_final_count_model"]
                    - blob["rollout"]["ens_final_count_true"]
                )
                for _, blob in entries
                if "rollout" in blob
            ]
            block["metrics"]["final_count_gap"] = summarize(gaps)

            table[setting][arm] = block

    # Differences against the baseline arm, with the SE of the DIFFERENCE
    for setting, arms in table.items():
        reference = arms.get(args.baseline_arm)

        if reference is None:
            continue

        for arm, block in arms.items():
            if arm == args.baseline_arm:
                continue

            deltas = {}
            for label in block["metrics"]:
                mine = block["metrics"][label]
                theirs = reference["metrics"][label]

                if mine["mean"] is None or theirs["mean"] is None:
                    continue

                pooled = math.sqrt(mine["se"] ** 2 + theirs["se"] ** 2)
                difference = mine["mean"] - theirs["mean"]
                deltas[label] = {
                    "delta": difference,
                    "se_of_delta": pooled,
                    # The only verdict this script issues, and it is a NOISE test,
                    # not a win: does the gap clear two pooled standard errors
                    "separated": bool(abs(difference) > 2.0 * pooled) if pooled > 0 else False,
                }

            block["vs_" + args.baseline_arm] = deltas

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=2))

    headline = (
        "delta_f1",
        "brier_infected",
        "ens_marg_mae",
        "final_count_gap",
        "effect_pearson",
        "effect_mae_norm",
        "plan_regret_model",
    )
    for setting, arms in table.items():
        print(f"\n=== {setting} ===")
        header = f"{'arm':<15}" + "".join(f"{label:>20}" for label in headline)
        print(header)
        for arm in arm_order:
            block = arms.get(arm)

            if block is None:
                continue

            cells = []
            for label in headline:
                stats = block["metrics"].get(label, {"mean": None, "se": 0.0})
                cells.append(
                    "        n/a" if stats["mean"] is None
                    else f"{stats['mean']:.4f}±{stats['se']:.4f}"
                )
            print(f"{arm:<15}" + "".join(f"{cell:>20}" for cell in cells))

        separated = [
            (arm, label)
            for arm, block in arms.items()
            for label, entry in block.get(f"vs_{args.baseline_arm}", {}).items()
            if entry["separated"]
        ]
        print(
            f"metrics clearing 2 pooled SE vs `{args.baseline_arm}`: "
            f"{separated if separated else 'none'}"
        )

    print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
