"""Pareto dominance, the fidelity/cost front, and the seed aggregator."""

import json

import numpy as np
import pytest

from world_model.aggregate_seeds import aggregate, separated
from world_model.pareto import (
    Objective,
    Point,
    cost_objectives,
    dominates,
    fidelity_cost_front,
    fidelity_objectives,
    hypervolume_2d,
    maximize,
    minimize,
    pareto_front,
    point_from_results,
)

error = Objective("error", minimize)
cost = Objective("cost", minimize)
score = Objective("score", maximize)


class TestDominance:
    def test_strictly_better_on_both_dominates(self):
        better = Point("a", {"error": 0.1, "cost": 1.0})
        worse = Point("b", {"error": 0.2, "cost": 2.0})

        assert dominates(better, worse, [error, cost])
        assert not dominates(worse, better, [error, cost])

    def test_equal_points_do_not_dominate_each_other(self):
        first = Point("a", {"error": 0.1, "cost": 1.0})
        second = Point("b", {"error": 0.1, "cost": 1.0})

        assert not dominates(first, second, [error, cost])
        assert not dominates(second, first, [error, cost])

    def test_a_tradeoff_is_not_dominance(self):
        cheap = Point("cheap", {"error": 0.3, "cost": 1.0})
        accurate = Point("accurate", {"error": 0.1, "cost": 9.0})

        assert not dominates(cheap, accurate, [error, cost])
        assert not dominates(accurate, cheap, [error, cost])

    def test_maximise_sense_is_honoured(self):
        better = Point("a", {"score": 0.9})
        worse = Point("b", {"score": 0.5})

        assert dominates(better, worse, [score])
        assert not dominates(worse, better, [score])

    def test_a_missing_value_never_dominates(self):
        """An absent number is not evidence of a good one."""
        partial = Point("partial", {"error": 0.01})
        complete = Point("complete", {"error": 0.2, "cost": 5.0})

        assert not dominates(partial, complete, [error, cost])

    def test_nan_never_dominates(self):
        broken = Point("broken", {"error": float("nan"), "cost": 0.0})
        fine = Point("fine", {"error": 0.5, "cost": 5.0})

        assert not dominates(broken, fine, [error, cost])


class TestFront:
    def test_front_keeps_tradeoffs_and_drops_dominated_points(self):
        points = [
            Point("cheap", {"error": 0.3, "cost": 1.0}),
            Point("accurate", {"error": 0.1, "cost": 9.0}),
            Point("dominated", {"error": 0.4, "cost": 10.0}),
        ]

        front = pareto_front(points, [error, cost])

        assert set(front["front"]) == {"cheap", "accurate"}
        assert set(front["dominated"]) == {"dominated"}

    def test_dominated_points_name_what_dominates_them(self):
        points = [
            Point("good", {"error": 0.1, "cost": 1.0}),
            Point("bad", {"error": 0.5, "cost": 5.0}),
        ]

        front = pareto_front(points, [error, cost])

        assert front["dominated"]["bad"] == ["good"]

    def test_single_point_is_its_own_front(self):
        front = pareto_front([Point("only", {"error": 1.0, "cost": 1.0})], [error, cost])

        assert front["front"] == ["only"]

    def test_no_objectives_is_an_error(self):
        with pytest.raises(ValueError, match="at least one objective"):
            pareto_front([Point("a", {})], [])


class TestHypervolume:
    def test_a_better_front_dominates_more_area(self):
        weak = [Point("w", {"error": 0.4, "cost": 4.0})]
        strong = [Point("s", {"error": 0.1, "cost": 1.0})]
        reference = (1.0, 10.0)

        assert hypervolume_2d(strong, error, cost, reference) > hypervolume_2d(
            weak, error, cost, reference
        )

    def test_a_point_beyond_the_reference_contributes_nothing(self):
        beyond = [Point("b", {"error": 2.0, "cost": 20.0})]

        assert hypervolume_2d(beyond, error, cost, (1.0, 10.0)) == 0.0

    def test_area_is_computed_exactly_for_one_point(self):
        point = [Point("p", {"error": 0.5, "cost": 5.0})]

        assert hypervolume_2d(point, error, cost, (1.0, 10.0)) == pytest.approx(2.5)


class TestResultsIntegration:
    def _results(self, mae, bias, seconds, delta_f1=0.8):
        return {
            "train_seconds": seconds,
            "test": {"delta_f1": delta_f1, "brier_infected": 0.01},
            "rollout": {
                "ens_marg_mae": mae,
                "ens_count_bias": bias,
                "ens_count_w1": 2.0,
            },
        }

    def test_count_bias_is_folded_to_its_absolute_value(self):
        """-3 and +3 are equally unfaithful; a signed axis would rank one as best."""
        under = point_from_results("under", self._results(0.1, -3.0, 10.0))
        over = point_from_results("over", self._results(0.1, 3.0, 10.0))

        assert under.values["abs_count_bias"] == over.values["abs_count_bias"] == 3.0

    def test_fidelity_cost_front_over_configurations(self):
        points = [
            point_from_results("fast_rough", self._results(0.30, 1.0, 5.0)),
            point_from_results("slow_exact", self._results(0.05, 0.1, 500.0)),
            point_from_results("worst", self._results(0.40, 2.0, 600.0)),
        ]

        front = fidelity_cost_front(points, "ens_marg_mae", "train_seconds")

        assert set(front["front"]) == {"fast_rough", "slow_exact"}
        assert "worst" in front["dominated"]

    def test_unknown_axis_names_are_rejected(self):
        points = [point_from_results("a", self._results(0.1, 0.0, 1.0))]

        with pytest.raises(ValueError, match="unknown fidelity objective"):
            fidelity_cost_front(points, "accuracy", "train_seconds")

        with pytest.raises(ValueError, match="unknown cost objective"):
            fidelity_cost_front(points, "ens_marg_mae", "dollars")

    def test_every_named_objective_carries_a_sense(self):
        for objective in {**fidelity_objectives, **cost_objectives}.values():
            assert objective.sense in (minimize, maximize)


class TestSeedAggregation:
    def _write(self, tmp_path, name, seed, delta_f1, regret):
        path = tmp_path / name
        path.write_text(
            json.dumps(
                {
                    "config": {"seed": seed},
                    "train_seconds": 10.0,
                    "test": {"delta_f1": delta_f1, "brier_infected": 0.01},
                    "rollout": {"ens_marg_mae": 0.09, "ens_count_bias": -0.5},
                    "planning": {
                        "plan_regret_model": regret,
                        "plan_regret_degree": 0.27,
                    },
                    "action_conditioning": {"action_conditioned": True},
                }
            )
        )
        return path

    def test_mean_and_std_across_seeds(self, tmp_path):
        paths = [
            self._write(tmp_path, "a.json", 1, 0.80, 0.24),
            self._write(tmp_path, "b.json", 2, 0.90, 0.28),
        ]

        summary = aggregate(paths)

        assert summary["n_runs"] == 2
        assert summary["seeds"] == [1, 2]
        assert summary["metrics"]["test.delta_f1"]["mean"] == pytest.approx(0.85)
        # Sample std (ddof=1) of {0.80, 0.90}
        assert summary["metrics"]["test.delta_f1"]["std"] == pytest.approx(
            np.std([0.8, 0.9], ddof=1)
        )

    def test_single_run_is_flagged_as_unsupported(self, tmp_path):
        summary = aggregate([self._write(tmp_path, "a.json", 1, 0.8, 0.24)])

        assert summary["warning"] is not None
        assert summary["metrics"]["test.delta_f1"]["std"] == 0.0

    def test_pass_rate_is_aggregated_not_collapsed(self, tmp_path):
        paths = [
            self._write(tmp_path, "a.json", 1, 0.8, 0.24),
            self._write(tmp_path, "b.json", 2, 0.9, 0.28),
        ]

        summary = aggregate(paths)

        assert summary["metrics"]["action_conditioning.pass"]["mean"] == 1.0

    def test_separation_reports_a_gap_inside_the_noise_as_not_separated(
        self, tmp_path
    ):
        """The exact situation in the published table: model 0.244 vs degree 0.269."""
        paths = [
            self._write(tmp_path, "a.json", 1, 0.8, 0.20),
            self._write(tmp_path, "b.json", 2, 0.8, 0.30),
            self._write(tmp_path, "c.json", 3, 0.8, 0.22),
        ]

        summary = aggregate(paths)
        verdict = separated(
            summary, "planning.plan_regret_model", "planning.plan_regret_degree"
        )

        assert verdict["separated"] is False
        assert abs(verdict["gap"]) < 2.0 * verdict["pooled_se"]

    def test_separation_returns_none_for_a_missing_metric(self, tmp_path):
        summary = aggregate([self._write(tmp_path, "a.json", 1, 0.8, 0.24)])

        assert separated(summary, "planning.plan_regret_model", "nope") is None
