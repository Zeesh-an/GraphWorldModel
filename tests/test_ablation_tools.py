import json
import os
import subprocess
import sys
from pathlib import Path

from pipeline.plots import plot_runtime
from scripts.compare_world_models import bias_at, table_row


def _results() -> list[dict]:
    return [
        {"evaluator": "world_model", "cost": {"rollout_seconds": 2.0, "n_samples": 200},
         "referee": "oracle", "referee_rollout_seconds": 1.0, "referee_samples": 1000,
         "mc_rollout_seconds": 50.0, "mc_agreement_runs": 50},
        {"evaluator": "oracle", "cost": {"rollout_seconds": 0.4, "n_samples": 200}, "referee": "oracle"},
    ]


def test_runtime_figure_draws_the_oracle_only_when_asked(tmp_path) -> None:
    assert plot_runtime(_results(), tmp_path / "runtime.png", "t") is not None
    assert plot_runtime(_results(), tmp_path / "with_oracle.png", "t", include_oracle=True) is not None

    # No Monte Carlo timing at all: neither variant has a comparison to draw
    assert plot_runtime(_results()[1:], tmp_path / "none.png", "t", include_oracle=True) is None


def test_bias_by_horizon_reads_the_step_curves() -> None:
    rollout = {"count_model_curve": [90.0, 99.0], "count_true_curve": [100.0, 100.0],
               "ens_count_bias": -1.0, "ens_final_count_true": 100.0}

    assert bias_at(rollout, 1) == "-10.0%"
    assert bias_at(rollout, 2) == "-1.0%"
    assert bias_at(rollout, 5) == "n/a"

    row = table_row("m", {"test": {"delta_f1": 0.5}, "rollout": rollout})
    assert row[0] == "m" and row[1] == "0.5000" and row[3] == "-1.0%"


def test_timing_reports_seconds_per_sample_once_per_repeat() -> None:
    from scripts.time_evaluators import seconds_per_sample

    calls = []

    class Environment:
        def rollout(self, plan: object, horizon: int, budget: int, seed: int) -> None:
            calls.append(seed)

    timings = seconds_per_sample(Environment(), None, horizon=10, budget=3, samples=200, repeats=3, seed=42, device="cpu", label="test")

    # one timed rollout per repeat, each on its own seed, each divided by the sample count
    assert calls == [42, 43, 44]
    assert len(timings) == 3 and all(0 <= value < 1e-3 for value in timings)


def test_backbone_comparison_writes_the_report_and_every_figure_in_both_formats(tmp_path: Path) -> None:
    results = {
        "config": {"model": "sage", "hidden_dim": 8, "n_layers": 2, "action_conditioning": None},
        "test": {"delta_f1": 0.8, "brier_infected": 0.01, "calibration_infected": {"mean_predicted": [0.1, 0.9], "mean_target": [0.12, 0.88], "ece": 0.01}},
        "rollout": {"count_model_curve": [10.0, 20.0], "count_true_curve": [11.0, 21.0], "ens_count_bias": -1.0, "ens_final_count_true": 21.0, "ens_marg_mae": 0.03},
        "action_conditioning": {"ablation": {"null_delta_f1_drop": 0.7}, "counterfactual_effect": {"effect_pearson": 0.9, "per_op": {"add_node": {"effect_pearson": 0.9}}}},
        "train_seconds": 60.0,
        "best_val_delta_f1": 0.79,
        "history": [{"epoch": 0, "val_delta_f1": 0.7}, {"epoch": 1, "val_delta_f1": 0.79}],
    }
    host = tmp_path / "results" / "influence_maximization" / "toy"
    for label in ("sage", "gat"):
        run_dir = host / f"abl_wm_{label}" / "world_model"
        run_dir.mkdir(parents=True)
        (run_dir / f"{label}_IC.json").write_text(json.dumps({**results, "config": {**results["config"], "model": label}}))

    subprocess.run(
        [sys.executable, "-m", "scripts.compare_backbones", "--host", "influence_maximization/toy", "--runs", "sage=abl_wm_sage", "gat=abl_wm_gat"],
        cwd=tmp_path, check=True, env={**os.environ, "PYTHONPATH": str(Path.cwd())},
    )

    out = host / "backbone_comparison"
    report = (out / "comparison.md").read_text()
    assert "| sage |" in report and "| gat |" in report
    # the link target is what follows the caption's closing bracket: ![caption](file.png)
    linked = [line.rsplit("](", 1)[1].rstrip(")") for line in report.splitlines() if line.startswith("![")]
    assert linked and all((out / name).exists() and (out / name).with_suffix(".pdf").exists() for name in linked)

