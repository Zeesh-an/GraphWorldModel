"""
Experiment 1 — the action ablation, as a results table.

    python -m scripts.eval_action_ablation \
        --results .../sage_IC.json --data-dir results/ood/ws100/data \
        --out results/action/ws100.json

Four arms, one question: does the model USE the action, or is it exploiting the
fact that most nodes do not change?

    full          the trained model, actions intact
    shuffle       identical states, actions permuted BETWEEN records. The action
                  distribution is unchanged and only the state-action pairing is
                  destroyed, so a model cannot recover the score from a prior over
                  actions.
    state_only    a model that sees the state and the graph but has the action
                  channels zeroed. Its purpose is to be the trap: on sparse,
                  persistence-dominated data it scores WELL on one-step accuracy
                  while being useless for choosing an action.
    constant      predicts a constant. The floor.

`state_only` and `constant` are the ones that make the test falsifiable: both
must land at exactly `effect_mae_norm = 1.0`, because a model that predicts no
action effect scores mean|d_true| / mean|d_true|. They are computed here as real
measurements on real data rather than asserted in a unit test.

The null is not "greater than zero" — it is 1.0, and it is reachable.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from world_model.checkpoint import load_checkpoint
from world_model.wm_action_eval import (
    action_ablation,
    counterfactual_effect,
    exogenous_fidelity,
)
from world_model.wm_data import TransitionDataset, ch_add, ch_edge, ch_remove
from world_model.wm_eval import evaluate_one_step

default_threads = 1


class ConstantModel(nn.Module):
    """Ignores everything. The floor: `effect_mae_norm` must be exactly 1.0."""

    head_type = "linear"

    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)), requires_grad=False)

    def forward(self, X: torch.Tensor, graph) -> torch.Tensor:
        return self.value.expand(X.shape[0], 2)


class StateOnlyModel(nn.Module):
    """
    The trained model with the ACTION CHANNELS ZEROED.

    Not a separately trained network: the same weights, denied the action. That
    isolates the action's contribution rather than confounding it with a
    different fit, and it is the strongest form of the control — if the full
    model and this one score the same, the action was decorative.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.inner = model
        self.head_type = getattr(model, "head_type", "structured")

    def forward(self, X: torch.Tensor, graph) -> torch.Tensor:
        blinded = X.clone()
        blinded[:, ch_add] = 0.0
        blinded[:, ch_remove] = 0.0
        blinded[:, ch_edge] = 0.0

        if blinded.shape[1] > 6:  # typed encoding's three extra action channels
            blinded[:, 6:] = 0.0

        return self.inner(blinded, graph)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Action ablation results table")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threads", type=int, default=default_threads)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)

    config = json.loads(args.results.read_text())["config"]
    diffusion_model = config["diffusion_model"]
    device = torch.device(args.device)
    checkpoint = (
        Path(config["ckpt_dir"]) / f"wm_{config['model']}_{diffusion_model}.pt"
    )
    model, spec, train_meta = load_checkpoint(
        checkpoint, config=config, device=device, strict_spec=False
    )
    dataset = TransitionDataset(
        args.data_dir, diffusion_model, args.split, spec.action_encoding
    )

    print(f"[action] {spec.backbone}/{spec.head} {diffusion_model} on "
          f"{args.data_dir} {args.split} ({len(dataset)} transitions)")

    arms = {
        "full": model,
        "state_only": StateOnlyModel(model).to(device).eval(),
        "constant": ConstantModel().to(device).eval(),
    }
    rows = {}
    started = time.perf_counter()

    for name, arm in arms.items():
        effect = counterfactual_effect(
            arm, dataset, diffusion_model, device, spec.hide_edge_weights
        )
        one_step = evaluate_one_step(
            arm, dataset, diffusion_model, device,
            hide_edge_weights=spec.hide_edge_weights,
        )
        exogenous = exogenous_fidelity(
            arm, dataset, diffusion_model, device, spec.hide_edge_weights
        )
        rows[name] = {
            "delta_f1": one_step["delta_f1"],
            "brier_infected": one_step["brier_infected"],
            "effect_mae_norm": effect["effect_mae_norm"],
            "effect_pearson": effect["effect_pearson"],
            "effect_magnitude_ratio": effect["effect_magnitude_ratio"],
            "n_pairs": effect["n_pairs"],
            "seeded_p_infected_worst": exogenous.get("seeded_p_infected_worst"),
        }
        print(f"  {name:12s} delta_f1={rows[name]['delta_f1']:.4f} "
              f"effect_mae_norm={rows[name]['effect_mae_norm']:.4f}")

    # `shuffle` is a corruption of the INPUT, not a different model, so it comes
    # from the ablation harness rather than from an arm.
    ablation = action_ablation(
        model, dataset, diffusion_model, device, spec.hide_edge_weights, args.seed
    )
    rows["shuffle"] = {
        "delta_f1": ablation.get("shuffle_delta_f1"),
        "delta_f1_drop_vs_full": ablation.get("shuffle_delta_f1_drop"),
        "brier_infected": ablation.get("shuffle_brier_infected"),
        "brier_rise_vs_full": ablation.get("shuffle_brier_rise"),
        "testable": ablation.get("shuffle_testable"),
        "n_with_action": ablation.get("n_with_action"),
    }
    rows["null_actions"] = {
        "delta_f1": ablation.get("null_delta_f1"),
        "delta_f1_drop_vs_full": ablation.get("null_delta_f1_drop"),
    }

    # The falsification check, stated as a property rather than left to a reader.
    verdict = {
        "constant_at_null": abs(rows["constant"]["effect_mae_norm"] - 1.0) < 1e-3,
        "state_only_at_null": abs(rows["state_only"]["effect_mae_norm"] - 1.0) < 1e-3,
        "full_below_null": rows["full"]["effect_mae_norm"] < 1.0,
        "shuffle_hurts": (rows["shuffle"].get("delta_f1_drop_vs_full") or 0) > 0,
    }
    verdict["passes"] = all(verdict.values())

    blob = {
        "question": "Q1_action_understanding",
        "model": {"backbone": spec.backbone, "head": spec.head,
                  "diffusion_model": diffusion_model,
                  "hide_edge_weights": spec.hide_edge_weights},
        "source": {"data_dir": train_meta.get("data_dir"),
                   "seed": train_meta.get("seed")},
        "target": {"data_dir": str(args.data_dir), "split": args.split,
                   "n_transitions": len(dataset)},
        "null_hypothesis": "effect_mae_norm = 1.0 means the model predicts NO "
                           "action effect; both trivial arms must reach it",
        "arms": rows,
        "verdict": verdict,
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(blob, indent=2, default=str))

    print(f"\n{'arm':14s} {'delta_f1':>9s} {'brier':>9s} {'effect_mae_norm':>17s} "
          f"{'pearson':>9s} {'mag_ratio':>10s}")
    print("-" * 72)
    for name in ("full", "state_only", "constant"):
        row = rows[name]
        print(f"{name:14s} {row['delta_f1']:9.4f} {row['brier_infected']:9.5f} "
              f"{row['effect_mae_norm']:17.4f} {row['effect_pearson']:9.4f} "
              f"{row['effect_magnitude_ratio']:10.4f}")
    print(f"{'shuffle':14s} {rows['shuffle']['delta_f1']:9.4f} "
          f"{rows['shuffle']['brier_infected']:9.5f} "
          f"{'(input corruption)':>17s} "
          f"{'drop=':>9s}{rows['shuffle']['delta_f1_drop_vs_full']:.4f}")
    print(f"{'null_actions':14s} {rows['null_actions']['delta_f1']:9.4f} "
          f"{'':>9s} {'(input corruption)':>17s} "
          f"{'drop=':>9s}{rows['null_actions']['delta_f1_drop_vs_full']:.4f}")
    print(f"\nverdict: {'PASS' if verdict['passes'] else 'FAIL'}  {verdict}")
    print(f"-> {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
