"""
Q4 ranking metrics.

What makes this metric family easy to fake, and what these tests block:

  * a model that outputs a constant scores 1.0 on preference accuracy if ties
    count as correct. It must score 0.
  * oracle ties inflate accuracy for free. They must be excluded, not resolved.
  * calls-to-first-win silently drops runs where nothing wins, which makes a bad
    ranking look good by removing its worst cases. It must return a value worse
    than "won last".
  * a headline accuracy earned entirely on pairs the oracle separates by 0.05
    nodes is noise. The margin buckets must expose that.

Nothing here needs a trained model: these are properties of the metrics.
"""

import numpy as np
import pytest

from world_model.wm_ranking import (
    CandidateSet,
    aggregate,
    calls_to_first_win,
    kendall_tau,
    preference_accuracy,
    ranking_orders,
    ranking_report,
    top_k_metrics,
    trusted_call_reduction,
)

true_scores = [10.0, 8.0, 6.0, 4.0, 2.0]


# ---------------------------------------------------------------------------
# B1 preference accuracy
# ---------------------------------------------------------------------------


def test_perfect_ordering_scores_one() -> None:
    assert preference_accuracy(true_scores, true_scores)["preference_accuracy"] == 1.0


def test_reversed_ordering_scores_zero() -> None:
    assert (
        preference_accuracy(true_scores[::-1], true_scores)["preference_accuracy"]
        == 0.0
    )


def test_constant_model_scores_zero_not_one() -> None:
    """
    The trivial baseline. A constant predictor ties every pair; if ties counted
    as correct it would score 1.0, which is how this metric gets faked.
    """
    result = preference_accuracy([5.0] * 5, true_scores)

    assert result["preference_accuracy"] == 0.0
    assert result["n_pairs_model_tied"] == 10


def test_oracle_ties_are_excluded_not_counted() -> None:
    """Two candidates the ORACLE cannot separate carry no ordering information."""
    result = preference_accuracy([1.0, 2.0, 3.0], [5.0, 5.0, 9.0])

    assert result["n_pairs_oracle_tied"] == 1
    assert result["n_pairs_evaluated"] == 2


def test_random_predictions_land_near_the_null() -> None:
    rng = np.random.default_rng(0)
    true = rng.normal(size=40).tolist()
    accuracies = [
        preference_accuracy(rng.normal(size=40).tolist(), true)["preference_accuracy"]
        for _ in range(20)
    ]

    assert 0.4 < float(np.mean(accuracies)) < 0.6


def test_margin_buckets_partition_the_evaluated_pairs() -> None:
    result = preference_accuracy([10.0, 1.0, 8.0, 2.0, 6.0], true_scores)
    bucketed = sum(bucket["n"] for bucket in result["by_margin"].values())

    assert bucketed == result["n_pairs_evaluated"]


def test_margin_buckets_separate_easy_from_noisy_pairs() -> None:
    """A model right only on wide margins must be visibly wrong on narrow ones."""
    true = [10.0, 9.99, 1.0, 0.99]
    predicted = [10.0, 10.01, 1.0, 1.01]  # wide gaps right, narrow ones flipped
    result = preference_accuracy(predicted, true)
    narrow = result["by_margin"]["[0.0,0.5)"]
    wide = result["by_margin"]["[2.0,inf)"]

    assert narrow["accuracy"] == 0.0
    assert wide["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# B2 rank correlation
# ---------------------------------------------------------------------------


def test_kendall_tau_endpoints() -> None:
    assert kendall_tau(true_scores, true_scores) == pytest.approx(1.0)
    assert kendall_tau(true_scores[::-1], true_scores) == pytest.approx(-1.0)


def test_kendall_tau_is_nan_for_a_constant_prediction() -> None:
    """Tau-b's denominator vanishes; NaN is honest, 0.0 would look like chance."""
    assert np.isnan(kendall_tau([1.0] * 5, true_scores))


# ---------------------------------------------------------------------------
# B3 top-K
# ---------------------------------------------------------------------------


def test_precision_at_1_detects_the_best_candidate() -> None:
    assert top_k_metrics([10, 1, 8, 2, 6], true_scores)["precision_at_1"] == 1.0
    assert top_k_metrics([1, 10, 8, 2, 6], true_scores)["precision_at_1"] == 0.0


def test_k_values_are_derived_from_candidate_count() -> None:
    """A hard-coded 3 means something different with 5 candidates than with 40."""
    small = top_k_metrics(list(range(5)), list(range(5)))
    large = top_k_metrics(list(range(40)), list(range(40)))

    assert small["k_values"] != large["k_values"]
    assert max(large["k_values"]) > max(small["k_values"])


def test_top_k_reports_the_value_obtained_not_just_the_hit() -> None:
    """
    Precision@1 is 0 when the model misses the best candidate — but picking the
    second-best is very different from picking the worst, and the planner cares
    about which.
    """
    near_miss = top_k_metrics([9, 10, 6, 4, 2], true_scores)
    disaster = top_k_metrics([1, 2, 3, 4, 10], true_scores)

    assert near_miss["precision_at_1"] == disaster["precision_at_1"] == 0.0
    assert near_miss["top_1_best_true_value"] > disaster["top_1_best_true_value"]


# ---------------------------------------------------------------------------
# B4 calls to first win
# ---------------------------------------------------------------------------


def test_best_first_ordering_wins_immediately() -> None:
    assert calls_to_first_win([0, 1, 2, 3, 4], true_scores, 8.0) == 1.0


def test_worst_first_ordering_pays_more() -> None:
    assert calls_to_first_win([4, 3, 2, 1, 0], true_scores, 8.0) == 4.0


def test_never_winning_is_worse_than_winning_last() -> None:
    """Dropping these runs would let a bad ranking hide its worst cases."""
    never = calls_to_first_win([0, 1], [1.0, 2.0], 99.0)
    last = calls_to_first_win([0, 1], [1.0, 2.0], 2.0)

    assert never > last
    assert never == 3.0


def test_win_threshold_is_an_input_not_a_derived_constant() -> None:
    order = [0, 1, 2, 3, 4]

    assert calls_to_first_win(order, true_scores, 2.0) == 1.0
    assert calls_to_first_win(order, true_scores, 10.0) == 1.0
    assert calls_to_first_win(order, true_scores, 4.0) == 1.0


def test_oracle_ordering_is_the_ceiling() -> None:
    orders = ranking_orders([1, 2, 3, 4, 5], true_scores)
    oracle = calls_to_first_win(orders["oracle"], true_scores, 8.0)
    model = calls_to_first_win(orders["world_model"], true_scores, 8.0)

    assert oracle <= model


# ---------------------------------------------------------------------------
# B5 call reduction
# ---------------------------------------------------------------------------


def test_call_reduction_is_relative_to_the_no_model_baseline() -> None:
    reduction = trusted_call_reduction(
        {"world_model": 1.0, "random": 4.0, "oracle": 1.0}
    )

    assert reduction["call_reduction_vs_random_world_model"] == pytest.approx(0.75)


def test_no_reduction_when_the_model_matches_the_baseline() -> None:
    reduction = trusted_call_reduction({"world_model": 4.0, "random": 4.0})

    assert reduction["call_reduction_vs_random_world_model"] == 0.0


def test_negative_reduction_is_reported_not_clipped() -> None:
    """A model worse than random must show as a negative saving."""
    reduction = trusted_call_reduction({"world_model": 8.0, "random": 4.0})

    assert reduction["call_reduction_vs_random_world_model"] < 0


# ---------------------------------------------------------------------------
# Report assembly and the seen/unseen policy split
# ---------------------------------------------------------------------------


def test_report_carries_every_metric_family() -> None:
    report = ranking_report(
        [10, 1, 8, 2, 6], true_scores, list("abcde"), "g0", seen_policies={"a", "b"}
    )

    for key in ("preference_accuracy", "kendall_tau", "precision_at_1",
                "calls_to_first_win_world_model", "calls_to_first_win_random",
                "win_threshold"):
        assert key in report.metrics


def test_seen_and_unseen_are_filters_over_the_same_scores() -> None:
    """
    Workstream E must not be a second evaluation under different conditions —
    otherwise a seen/unseen gap could come from the protocol, not the policies.
    """
    report = ranking_report(
        [10, 1, 8, 2, 6], true_scores, list("abcde"), "g0", seen_policies={"a", "b"}
    )

    assert report.metrics["n_candidates_seen"] == 2
    assert report.metrics["n_candidates_unseen"] == 3


def test_a_policy_subset_too_small_to_rank_reports_nan_not_a_number() -> None:
    report = ranking_report(
        [10, 1, 8, 2, 6], true_scores, list("abcde"), "g0", seen_policies={"a"}
    )

    assert np.isnan(report.metrics["preference_accuracy_seen"])
    assert report.metrics["n_candidates_seen"] == 1


def test_win_threshold_defaults_are_recorded() -> None:
    """A criterion chosen after seeing the scores is not a criterion."""
    report = ranking_report([1, 2, 3, 4, 5], true_scores, list("abcde"), "g0")

    assert report.metrics["win_quantile"] == 0.8
    assert np.isfinite(report.metrics["win_threshold"])


def test_aggregate_reports_n_alongside_mean_and_std() -> None:
    reports = [
        ranking_report([10, 1, 8, 2, 6], true_scores, list("abcde"), f"g{i}")
        for i in range(3)
    ]
    summary = aggregate(reports)

    assert summary["n_graphs"] == 3
    assert "preference_accuracy_std" in summary
    assert summary["preference_accuracy_n"] == 3


def test_candidate_set_rejects_mismatched_policy_labels() -> None:
    with pytest.raises(ValueError, match="index the same candidates"):
        CandidateSet(graph_id="g", seed_sets=[[1], [2]], policies=["a"], budget=1)
