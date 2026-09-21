import json
from pathlib import Path

import pytest

from pipeline.conditions import parse_arm
from pipeline.layout import Layout
from pipeline.run import ordered_points, searches_once, transfer_source

points = [("pct1", 1.0, 16), ("pct5", 5.0, 79), ("pct10", 10.0, 159), ("pct20", 20.0, 318)]


def test_search_point_runs_first_and_the_rest_keep_their_order() -> None:
    assert [p[0] for p in ordered_points(points, "pct10")] == ["pct10", "pct1", "pct5", "pct20"]
    assert ordered_points(points, None) == points


def test_a_label_outside_the_sweep_is_rejected_with_the_choices() -> None:
    with pytest.raises(ValueError, match="pct1.*pct20"):
        ordered_points(points, "pct50")


def test_transfer_source_reads_the_searched_row_and_refuses_one_without_a_program(tmp_path: Path) -> None:
    layout = Layout("influence_maximization", "ba", "transfer", root=tmp_path)
    arm = parse_arm("evolve_free@oracle")
    assert transfer_source(layout, "pct10", arm) is None

    path = layout.agent_result("pct10", arm.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"script": "", "model": "gpt-6-astra"}))
    assert transfer_source(layout, "pct10", arm) is None

    path.write_text(json.dumps({"script": "class S: pass", "model": "gpt-6-astra"}))
    assert transfer_source(layout, "pct10", arm)["model"] == "gpt-6-astra"


def test_a_discovery_system_transfers_like_an_llm_arm_and_a_seed_set_repo_does_not(tmp_path: Path) -> None:
    layout = Layout("influence_maximization", "ba", "transfer", root=tmp_path)
    arm = parse_arm("discovery:eoh")
    assert searches_once(arm) and searches_once(parse_arm("evolve_free@oracle"))
    assert not searches_once(parse_arm("external:imm")) and not searches_once(parse_arm("routing"))

    # An external row lives in the baselines tree, not the agent one
    path = layout.agent_result("pct10", arm.name, external=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"script": "def select_seeds(graph, k): ...", "model": "discovery:eoh"}))
    assert transfer_source(layout, "pct10", arm)["model"] == "discovery:eoh"
