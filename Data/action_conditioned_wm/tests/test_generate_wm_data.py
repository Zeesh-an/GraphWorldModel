import json
from pathlib import Path

from generate_wm_data import (
    GenConfig,
    GraphStore,
    TransitionWriter,
    build_record,
    run_generation,
)
from wm_graphs import make_synthetic_bundle
from wm_simulator import ActionOp, State


def test_build_record():
    rec = build_record(
        graph_id="ba_n100_m3_s0", diffusion_model="IC", episode_id="ep1",
        algorithm="pagerank", branch="main", t=1,
        state=State(infected=[0, 3], frontier=[3]), action=[ActionOp("add_seed", 5)],
        next_state=State(infected=[0, 3, 5], frontier=[5]), reward=1.0,
    )
    assert rec["action"] == [{"op": "add_seed", "target": 5}]
    assert rec["reward"] == 1.0
    assert rec["state"]["infected_count"] == 2


def test_graph_store_and_transition_round_trip(tmp_path: Path) -> None:
    out = tmp_path / "out"
    b = make_synthetic_bundle("ba", index=0, n=30, ba_m=2, seed=0, prob_model="weighted")

    gs = GraphStore(out)
    gs.save(b)
    gs.flush()
    assert (out / "graphs" / f"{b.graph_id}.npz").exists()
    index = json.loads((out / "graphs_index.json").read_text())
    assert index[0]["graph_id"] == b.graph_id

    with TransitionWriter(out) as tw:
        rec = build_record(
            graph_id=b.graph_id, diffusion_model="IC", episode_id="e", algorithm="random",
            branch="main", t=0, state=State(infected=[0], frontier=[0]),
            action=[ActionOp("add_seed", 0)], next_state=State(infected=[0], frontier=[0]), reward=0.0,
        )
        tw.write(rec, model="IC", split="train")

    lines = (out / "transitions_IC_train.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["graph_id"] == b.graph_id


def _smoke_cfg(out_dir: Path) -> GenConfig:
    return GenConfig(
        dataset="er", num_graphs=1, syn_nodes=40, er_p=0.1,
        models=["IC", "LT"], prob_model="weighted", uniform_p=0.1,
        budget=3, budget_pct=None, algorithms=["random", "degree"],
        rollouts=2, horizon=4, inject_p=0.5, p_add=0.5, p_remove=0.5,
        cf_prob=0.5, cf_branches=2, split=(0.7, 0.15, 0.15),
        seed=123, out_dir=str(out_dir),
    )


def test_smoke_produces_valid_outputs(tmp_path: Path) -> None:
    out = tmp_path / "out"
    run_generation(_smoke_cfg(out))

    # graph store written
    assert (out / "graphs_index.json").exists()
    index = json.loads((out / "graphs_index.json").read_text())
    assert len(index) == 1

    # at least one transitions file exists and every line is a valid record
    files = list(out.glob("transitions_*.jsonl"))
    assert files, "no transition files written"
    seen_t0_bag = False
    for fp in files:
        for line in fp.read_text().strip().splitlines():
            rec = json.loads(line)
            assert rec["graph_id"] == index[0]["graph_id"]
            assert set(rec["state"]) >= {"infected", "frontier"}
            if rec["t"] == 0 and len(rec["action"]) >= 1:
                seen_t0_bag = True
    assert seen_t0_bag, "expected a t=0 seed-commit action bag"

    # metadata written
    assert (out / "metadata.json").exists()


def test_smoke_reproducible(tmp_path: Path) -> None:
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    run_generation(_smoke_cfg(out_a))
    run_generation(_smoke_cfg(out_b))
    names_a = {p.name for p in out_a.glob("transitions_*.jsonl")}
    names_b = {p.name for p in out_b.glob("transitions_*.jsonl")}
    # both runs must produce the SAME (non-empty) set of files, byte-for-byte equal
    assert names_a and names_a == names_b, "runs produced no / differing transition files"
    for name in names_a:
        assert (out_a / name).read_text() == (out_b / name).read_text(), \
            f"{name} differs across runs"
