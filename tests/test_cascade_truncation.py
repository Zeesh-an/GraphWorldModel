from data.datasets.cascade_common import Cascade, filter_cascades


def _cascade() -> Cascade:
    # Twelve adoptions at elapsed 0, 10, ..., 110; a window of 40 observes five
    events = [(node, node * 10, None if node == 0 else 0) for node in range(12)]
    return Cascade(cascade_id="c", root=0, publish_time=0, events=events)


def test_truncation_caps_the_observed_prefix_only() -> None:
    kept = filter_cascades([_cascade()], observation=40, min_observed=1, truncate=3)

    assert kept[0].popularity_at(40) == 3
    assert [event[0] for event in kept[0].events] == [0, 1, 2] + list(range(5, 12))


def test_truncation_off_and_full_window() -> None:
    untouched = filter_cascades([_cascade()], observation=40, min_observed=1, truncate=0)
    assert untouched[0].size == 12

    whole = filter_cascades([_cascade()], observation=200, min_observed=1, truncate=3)
    assert whole[0].size == 3
