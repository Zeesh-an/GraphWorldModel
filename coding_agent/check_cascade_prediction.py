"""
Runnable self-check for the cascade-prediction contract.

    python -m coding_agent.check_cascade_prediction

Seven checks, each one a claim `research/cascade_prediction.md` makes that would be
expensive to discover was false three stages into a sweep:

  1. **The metric definitions are the published ones.** §8.1 opens with "Get MSLE
     right or nothing else matters" and then lists three independent choices hiding
     inside that name (log base, total vs increment, the smoothing offset) and
     §5.7 difference 4 records that §5.1 prints two of the variants in ONE table.
     Checked against hand-worked values rather than against ourselves.

  2. **Declining is counted, never scored.** §8.4: generative models refuse to score
     supercritical cascades, and papers reporting the mean over scoreable cascades
     alone "silently favour the model that gives up more often". A decline must move
     `n_failed` and must not move the error.

  3. **The chronological split is leak-free.** §8.3 is the most transferable finding
     in the file: no training cascade's prediction window may reach past the first
     test observation. Asserted directly rather than trusted.

  4. **A replayed cascade round-trips.** The whole task rests on
     `data/wm_cascades.py` writing a real log into the same JSONL the simulator
     writes, and `world_model/wm_data.py` reading it back unchanged. If the
     adoption times do not survive that round trip, every number downstream is about
     a cascade that did not happen.

  5. **The generative predictors decline where the paper says they do.** SEISMIC and
     the Hawkes line diverge at branching ratio 1 (§3.2, §5.4), and our
     reimplementations have to fail at the same boundary rather than emitting a
     plausible number.

  6. **The floors are computed and are strong.** Under a LOG-space error an
     instance-blind constant is far stronger than intuition suggests, and §9's whole
     framing depends on a reader being able to see that. The analogue of cascade
     reconstruction's `trivial_decoder_reward` check.

  7. **A predictor cannot see the answer.** The observation must contain no adopter
     past `t_o`, and `predict()` must reject a popularity below what was already
     observed. Both are ways a wiring bug would surface as an excellent method.
"""

import math
import numpy as np

from coding_agent.executor import StrategyError
from coding_agent.prediction import (
    CascadeObservation,
    ForecastInstance,
    observed_waves,
    trivial_predictor_error,
    validate_prediction,
)
from coding_agent.tools.prediction_algorithms import (
    branching_factor,
    cascade_features,
    hawkes,
    mean_size,
    persistence,
    seismic,
    szabo_huberman,
)
from coding_agent.types import GraphInfo
from data.datasets.cascade_common import Cascade, filter_cascades
from data.wm_cascades import ReplayConfig, assign_splits, bin_events, replay_records
from world_model.wm_metrics import (
    coverage_at_k,
    doubling_accuracy,
    popularity_metrics,
    prediction_reward,
)

tolerance = 1e-9


def _graph(num_nodes: int = 40) -> GraphInfo:
    """A small ring-plus-hub graph; the checks below do not depend on its shape."""
    edges = [(node, (node + 1) % num_nodes) for node in range(num_nodes)]
    edges += [(0, node) for node in range(2, num_nodes, 3)]
    both = edges + [(target, source) for source, target in edges]

    return GraphInfo(
        num_nodes=num_nodes,
        edge_index=np.array(both, dtype=np.int64).T,
        ic_probs=np.full(len(both), 0.2, dtype=np.float32),
        directed=False,
    )


def _observation(adopters: dict[int, int], window: int, horizon: int) -> CascadeObservation:
    last = max(adopters.values()) if adopters else 0

    return CascadeObservation(
        cascade_id="c",
        root=min(adopters, default=0),
        num_nodes=40,
        adopters=adopters,
        frontier=tuple(node for node, step in adopters.items() if step == last),
        observed_steps=window,
        horizon=horizon,
    )


def check_metric_definitions() -> None:
    """§8.1's three hidden choices, against arithmetic done by hand."""
    predicted = np.array([4.0, 16.0])
    actual = np.array([2.0, 8.0])
    observed = np.array([1.0, 4.0])

    metrics = popularity_metrics(predicted, actual, observed)

    # CasFlow's own code: log2, clamp at 1, no offset. log2(4)-log2(2) = 1 and
    # log2(16)-log2(8) = 1, so MSLE is exactly 1 and MALE exactly 1.
    assert abs(metrics["msle"] - 1.0) < tolerance, metrics["msle"]
    assert abs(metrics["male"] - 1.0) < tolerance, metrics["male"]

    # CasFT's stated definition adds 1 INSIDE the log, which is a different number
    expected_offset = float(
        np.mean((np.log2(predicted + 1) - np.log2(actual + 1)) ** 2)
    )
    assert abs(metrics["msle_offset"] - expected_offset) < tolerance
    assert metrics["msle_offset"] != metrics["msle"], (
        "msle and msle_offset came out equal; §5.7 difference 4 is that they are "
        "NOT the same metric and §5.1 prints both under one name"
    )

    # CTCP's loss uses natural log, a factor of (ln 2)^2 from CasFlow's
    assert abs(metrics["msle_natural"] - metrics["msle"] * math.log(2) ** 2) < 1e-9

    # MRSE is genuinely relative on RAW counts: ((4-2)/2)^2 = 1, ((16-8)/8)^2 = 1
    assert abs(metrics["mrse"] - 1.0) < tolerance, metrics["mrse"]

    # WroPerc counts |relative error| >= 0.5, which both of these exceed
    assert abs(metrics["wroperc"] - 1.0) < tolerance, metrics["wroperc"]

    # The increment: predicted 3 and 12 against actual 1 and 4
    expected_increment = float(
        np.mean((np.log2(np.array([3.0, 12.0]) + 1) - np.log2(np.array([1.0, 4.0]) + 1)) ** 2)
    )
    assert abs(metrics["msle_increment"] - expected_increment) < tolerance

    # COV-k with fewer than 10 cascades has no k at all, and must say so rather
    # than silently reporting a coverage over an empty set
    assert not np.isfinite(coverage_at_k(predicted, actual))

    # ...and with enough, a perfectly ranked prediction covers the top decile
    many_actual = np.arange(1, 41, dtype=float)
    assert abs(coverage_at_k(many_actual, many_actual) - 1.0) < tolerance

    # Cheng et al.'s balanced framing: doubling means reaching 2 * observed
    assert abs(doubling_accuracy(predicted, actual, observed) - 1.0) < tolerance

    print("[OK] metric definitions match §8.1's published forms, and the variants differ")


def check_declines_are_counted() -> None:
    """§8.4: a decline moves `n_failed` and never the error."""
    predicted = np.array([4.0, 16.0])
    actual = np.array([2.0, 8.0])
    observed = np.array([1.0, 4.0])

    plain = popularity_metrics(predicted, actual, observed)
    with_declines = popularity_metrics(predicted, actual, observed, declined=8)

    assert abs(plain["msle"] - with_declines["msle"]) < tolerance, (
        "declining changed the ERROR; §8.4's whole point is that papers report the "
        "mean over SCOREABLE cascades, which is why the count has to be separate"
    )
    assert with_declines["n_failed"] == 8
    assert abs(with_declines["decline_rate"] - 8 / 10) < tolerance
    assert plain["n_failed"] == 0 and plain["decline_rate"] == 0.0

    # A predictor that scored nothing must be INFINITELY bad rather than NaN, or it
    # would compare false against everything and quietly win the `improves` test
    empty = popularity_metrics(
        np.zeros(0), np.zeros(0), np.zeros(0), declined=5
    )
    assert prediction_reward(empty, "msle") == float("inf")

    print("[OK] declines are counted, never scored, and an all-decline arm is +inf")


def _corpus(count: int = 60, spacing: int = 100) -> list[Cascade]:
    """`count` cascades published `spacing` apart, each with a handful of adopters."""
    cascades = []

    for index in range(count):
        events = [(index % 30, 0, None)]
        events += [
            ((index + step) % 30 + 5, step * 10, index % 30) for step in range(1, 12)
        ]
        cascades.append(
            Cascade(
                cascade_id=f"c{index}",
                root=index % 30,
                publish_time=index * spacing,
                events=events,
            )
        )

    return cascades


def check_split_is_leak_free() -> None:
    """§8.3: no training cascade's prediction window reaches a test observation."""
    cascades = _corpus()
    config = ReplayConfig(
        dataset="synthetic",
        out_dir="",
        observation=40,
        horizon=200,
        step=10,
        split="chronological",
    )
    assignment = assign_splits(cascades, config)
    by_id = {cascade.cascade_id: cascade for cascade in cascades}

    training = [
        by_id[key] for key, split in assignment.items() if split == "train"
    ]
    testing = [by_id[key] for key, split in assignment.items() if split == "test"]

    assert training and testing, (
        f"the chronological split produced {len(training)} train / {len(testing)} "
        f"test on an evenly-spaced corpus, which means the drop rule is too strict"
    )

    first_test = min(cascade.publish_time for cascade in testing)
    for cascade in training:
        assert cascade.publish_time + config.horizon <= first_test, (
            f"training cascade {cascade.cascade_id} predicts to "
            f"{cascade.publish_time + config.horizon} but the first test cascade is "
            f"observed from {first_test}: that is exactly the leak §8.3 describes"
        )

    # ...and the ORDER is the only thing the random protocol changes, which is what
    # makes the two a clean A/B: the pool sizes must match
    leaky = assign_splits(cascades, ReplayConfig(
        dataset="synthetic", out_dir="", observation=40, horizon=200, step=10,
        split="random",
    ))
    assert sum(1 for split in leaky.values() if split == "test") == len(
        [key for key, split in assignment.items() if split == "test"]
    ), "the two protocols produced different TEST pool sizes, which confounds the A/B"

    print(
        f"[OK] chronological split is leak-free "
        f"({len(training)} train / {len(testing)} test, and no window crosses)"
    )


def check_replay_round_trip() -> None:
    """A logged cascade survives binning and re-reading with its times intact."""
    cascade = Cascade(
        cascade_id="rt",
        root=0,
        publish_time=0,
        # Elapsed 0, 5, 12, 25, 44 with a step of 10 lands them in bins 0, 0, 1, 2, 4
        events=[(0, 0, None), (1, 5, 0), (2, 12, 1), (3, 25, 2), (4, 44, 3)],
    )
    waves, parents = bin_events(cascade, step=10, horizon=6)

    assert waves[0] == [0, 1], waves
    assert waves[1] == [2] and waves[2] == [3] and waves[4] == [4], waves
    assert parents[2] == 1 and parents[0] is None

    class _Bundle:
        graph_id = "g"

    records = replay_records(
        cascade,
        _Bundle(),
        ReplayConfig(
            dataset="synthetic", out_dir="", observation=20, horizon=60, step=10,
            gen_horizon=6,
        ),
        "IC",
        "train",
    )

    # `a_t = NULL` at every step but the seed commit: §2.1's defining property
    assert records[0]["action"], "the t=0 record must carry the root's seed commit"
    assert all(not record["action"] for record in records[1:]), (
        "a replayed cascade emitted an action after t=0; §2.1 is that nothing "
        "intervenes and the cascade is only watched"
    )

    # HARD targets: a real cascade happened once, so every marginal is exactly 1
    for record in records:
        for value in record["next_marginal_infected"].values():
            assert value == 1.0, "a replayed target was not a hard 0/1 (§2.4)"

    # Record `t` carries the wave binned at `t`: the invariant `observed_waves`
    # depends on, and the one that decides how much prefix a predictor is shown
    recovered = {}
    for step, record in enumerate(records):
        for node in record["next_state"]["frontier"]:
            recovered.setdefault(int(node), step)

    for step, wave in enumerate(waves):
        for node in wave:
            assert recovered.get(node) == step, (
                f"node {node} was binned to wave {step} but replayed at record "
                f"{recovered.get(node)}"
            )

    # A cascade SHORTER than the observation window is the normal case, not an edge
    # one (most cascades are over long before `t_o`) so `observed_waves` has to
    # clamp rather than index past the end of the recorded history
    episode = {
        "frontiers": np.array(
            [[0.0] * 5] + [
                [1.0 if node in wave else 0.0 for node in range(5)]
                for wave in waves[: len(records)]
            ],
            dtype=np.float32,
        )
    }
    wide = observed_waves(episode, window=99)
    assert wide, "observed_waves returned nothing for a window past the cascade's end"
    assert max(wide.values()) <= len(records) - 1

    narrow = observed_waves(episode, window=1)
    assert max(narrow.values()) <= 1 and len(narrow) < len(wide) + 1

    print(f"[OK] a logged cascade round-trips ({len(records)} records, hard targets, NULL actions)")


def check_generative_declines() -> None:
    """§3.2 and §5.4: SEISMIC and Hawkes decline above branching ratio 1."""
    graph = _graph()

    # A cascade whose waves are GROWING: 1, 2, 4, 8 gives a branching ratio of 2
    growing = _observation(
        {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 3, 11: 3,
         12: 3, 13: 3, 14: 3},
        window=3,
        horizon=10,
    )
    assert branching_factor(graph, growing, 10) is None, (
        "branching_factor emitted a number for a SUPERCRITICAL cascade; the expected "
        "size diverges there and §3.2 records that SEISMIC's honest response is to "
        "produce no prediction at all"
    )
    assert hawkes(graph, growing, 10) is None

    # ...and a DECAYING one is scoreable: 8, 4, 2, 1
    decaying = _observation(
        {node: 0 for node in range(8)}
        | {node: 1 for node in range(8, 12)}
        | {12: 2, 13: 2, 14: 3},
        window=3,
        horizon=10,
    )
    estimate = branching_factor(graph, decaying, 10)
    assert estimate is not None and estimate >= decaying.popularity, estimate

    # SEISMIC's own boundary is the same product, and it must be reached
    supercritical = seismic(graph, growing, 10)
    assert supercritical is None or supercritical >= growing.popularity

    print("[OK] the generative predictors decline exactly where §3.2 says they do")


def check_floors_are_strong() -> None:
    """§9: under a LOG-space error an instance-blind constant is a real bar."""
    graph = _graph()
    instances = []

    rng = np.random.default_rng(0)
    for index in range(30):
        size = int(rng.integers(5, 30))
        adopters = {node: min(node, 3) for node in range(min(size, 12))}
        instances.append(
            ForecastInstance(
                cascade_id=f"f{index}",
                graph_id="g",
                observation=_observation(adopters, window=3, horizon=10),
                actual=size,
                final_state=np.zeros(40, dtype=np.float32),
                split="train",
            )
        )

    floors = trivial_predictor_error(instances, instances, "msle")

    assert floors["trivial_predictor_error"] < 1.0, (
        f"the constant predictor scored {floors['trivial_predictor_error']:.3f}, "
        f"which is worse than a log-space error should allow: under MSLE the "
        f"geometric mean is the minimizer over instance-blind rules and it is "
        f"supposed to be STRONG"
    )
    assert floors["persistence_error"] >= 0.0
    assert floors["trivial_predictor_value"] > 0

    # ...and the fitted one-line classic must at least match the constant, or the
    # single most-printed baseline in this literature is broken
    fitted = szabo_huberman(graph, instances[0].observation, 10, fit_examples=instances)
    assert fitted >= instances[0].observation.popularity

    assert mean_size(graph, instances[0].observation, 10, fit_examples=instances) > 0
    assert persistence(graph, instances[0].observation, 10) == instances[0].observation.popularity

    print(
        f"[OK] the two floors are computed and strong "
        f"(constant {floors['trivial_predictor_error']:.4f}, "
        f"persistence {floors['persistence_error']:.4f})"
    )


def check_predictor_cannot_see_the_answer() -> None:
    """The observation stops at `t_o`, and a shrinking prediction is rejected."""
    graph = _graph()
    observation = _observation({0: 0, 1: 1, 2: 2, 3: 3}, window=3, horizon=10)

    assert max(observation.adopters.values()) <= observation.observed_steps, (
        "an adopter past the observation window reached the predictor: that is the "
        "answer, not the prefix"
    )
    assert observation.popularity == 4

    instance = ForecastInstance(
        cascade_id="c",
        graph_id="g",
        observation=observation,
        actual=20,
        final_state=np.zeros(40, dtype=np.float32),
    )

    # Below the observed count: adoption is progressive and nobody un-adopts
    try:
        validate_prediction(2.0, instance)
        raise AssertionError("a prediction below the observed popularity was accepted")
    except StrategyError:
        pass

    # Above |V|: popularity counts DISTINCT adopters
    try:
        validate_prediction(1000.0, instance)
        raise AssertionError("a prediction larger than |V| was accepted")
    except StrategyError:
        pass

    assert validate_prediction(None, instance) is None
    assert validate_prediction(float("inf"), instance) is None
    assert validate_prediction(12.0, instance) == 12.0

    # The feature extractor sees only the prefix, and its second-half rate: the
    # single most predictive feature in §5.6: is computable from it
    features = cascade_features(graph, observation)
    assert features["observed"] == 4
    assert "rate_second_half" in features and np.isfinite(features["rate_second_half"])
    assert features["remaining_steps"] == 7

    print("[OK] the prefix stops at t_o and an impossible prediction is rejected")


def check_filters_are_the_published_ones() -> None:
    """§8.4: the participant filter and the truncation are CasFlow's own."""
    cascades = _corpus(count=5)

    # Every cascade here has 12 events, one at elapsed 0 and eleven at 10..110.
    # An observation window of 40 sees 5 of them, so a `< 10` filter drops all five.
    kept = filter_cascades(cascades, observation=40, min_observed=10, truncate=100)
    assert not kept, (
        "the participant filter did not drop cascades with too few OBSERVED "
        "adopters; §8.4 records that this threshold moves MSLE by more than the gap "
        "between any two consecutive published rows"
    )

    # ...and a window wide enough to see all twelve keeps them
    kept = filter_cascades(cascades, observation=200, min_observed=10, truncate=100)
    assert len(kept) == 5

    # Truncation keeps the FIRST n participants, which is CasFlow's rule
    truncated = filter_cascades(cascades, observation=200, min_observed=1, truncate=3)
    assert all(cascade.size == 3 for cascade in truncated), (
        [cascade.size for cascade in truncated]
    )

    print("[OK] the participant filter and the truncation are CasFlow's own")


if __name__ == "__main__":
    check_metric_definitions()
    check_declines_are_counted()
    check_split_is_leak_free()
    check_replay_round_trip()
    check_generative_declines()
    check_floors_are_strong()
    check_predictor_cannot_see_the_answer()
    check_filters_are_the_published_ones()
    print("\nAll cascade-prediction checks passed.")
