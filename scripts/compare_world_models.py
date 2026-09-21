"""
One markdown table across world-model results files, for the ablations that only train.

    python -m scripts.compare_world_models \
        sage=results/.../world_model/sage_IC.json gat=results/.../world_model/gat_IC.json

A file is either the results JSON `train_wm.py` writes or the output of
`scripts.eval_frozen`: both carry `test`, `rollout` and `action_conditioning`.
"""

import argparse
import json
from pathlib import Path

horizon_steps = (1, 2, 5, 10, 20)


def bias_at(rollout: dict, step: int) -> str:
    """Relative count bias after `step` transitions, from the per-step rollout curves."""
    model = rollout.get("count_model_curve") or []
    true = rollout.get("count_true_curve") or []

    # Entry i is the count after transition i + 1; a run generated at a shorter
    # horizon has no entry there and says so rather than repeating its last value
    if step > len(model) or step > len(true) or not true[step - 1]:
        return "n/a"

    return f"{100.0 * (model[step - 1] - true[step - 1]) / true[step - 1]:+.1f}%"


def table_row(label: str, results: dict) -> list[str]:
    test = results.get("test", {})
    rollout = results.get("rollout", {})
    conditioning = results.get("action_conditioning", {})
    ablation = conditioning.get("ablation", {})
    effect = conditioning.get("counterfactual_effect", {})
    final_true = rollout.get("ens_final_count_true")

    def number(value: float | None, digits: int) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    return [
        label,
        number(test.get("delta_f1"), 4),
        number(test.get("brier_infected"), 4),
        (
            "n/a"
            if not final_true or rollout.get("ens_count_bias") is None
            else f"{100.0 * rollout['ens_count_bias'] / final_true:+.1f}%"
        ),
        number(rollout.get("ens_marg_mae"), 3),
        *[bias_at(rollout, step) for step in horizon_steps],
        number(ablation.get("null_delta_f1_drop"), 3),
        number(effect.get("effect_pearson"), 3),
        number(None if results.get("train_seconds") is None else results["train_seconds"] / 60.0, 1),
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare world-model results files in one table")
    parser.add_argument(
        "runs",
        type=str,
        nargs="+",
        help="label=path pairs, one per world-model results file (default: required).",
    )
    args = parser.parse_args()

    header = [
        "model", "delta F1", "Brier", "final count bias", "marginal MAE",
        *[f"bias @ {step}" for step in horizon_steps],
        "delta F1 drop, actions zeroed", "effect Pearson", "train min",
    ]
    print(f"| {' | '.join(header)} |")
    print(f"| {' | '.join('---' for _ in header)} |")

    for run in args.runs:
        label, path = run.split("=", 1)
        print(f"| {' | '.join(table_row(label, json.loads(Path(path).read_text())))} |")
