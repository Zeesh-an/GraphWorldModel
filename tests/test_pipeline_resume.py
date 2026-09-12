import json
from pathlib import Path

from pipeline.layout import Layout
from pipeline.run import carried_seconds


def _layout(tmp_path: Path, stages: dict) -> Layout:
    layout = Layout("influence_maximization", "ba", "resume", root=tmp_path)
    layout.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    layout.manifest_path.write_text(json.dumps({"config": {}, "stages": stages}))

    return layout


def test_failed_attempt_seconds_carry_into_the_next_attempt(tmp_path: Path) -> None:
    layout = _layout(tmp_path, {"agent": {"status": "failed", "seconds": 29396.6}})
    assert carried_seconds(layout, "agent") == 29396.6
    assert carried_seconds(layout, "train") == 0.0


def test_killed_and_finished_attempts_carry_nothing(tmp_path: Path) -> None:
    # A killed process never recorded its seconds; a finished stage re-run
    # under --force is a redo, not a continuation
    layout = _layout(
        tmp_path,
        {"train": {"status": "running", "seconds": None}, "data": {"status": "done", "seconds": 5.0}},
    )
    assert carried_seconds(layout, "train") == 0.0
    assert carried_seconds(layout, "data") == 0.0


def test_missing_manifest_carries_nothing(tmp_path: Path) -> None:
    layout = Layout("influence_maximization", "ba", "fresh", root=tmp_path)
    assert carried_seconds(layout, "agent") == 0.0
