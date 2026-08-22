"""
Graph-disjoint vs episode_random splitting.

The failure being fixed is invisible by construction: an episode-level split
produces perfectly well-formed files, trains without error, and reports test
metrics that are optimistic because the model saw the same graph — same
structure, same per-edge transmission probabilities, same cascade on the main
branch — during training.

So the tests are about the SPLIT LABELS, not about the data being loadable:

  * graph_disjoint: no graph may appear in two splits. That is the property.
  * episode_random: must keep behaving exactly as it always did, because
    reproducing a pre-2026-08-22 number requires reproducing its split.
  * the two modes must differ ONLY in the labels, so a rerun is a controlled
    comparison rather than a different dataset.
"""

import json

import numpy as np
import pytest

from data.generate_wm_data import (
    _assign_split,
    _graph_split_plan,
    episode_random_split,
    graph_disjoint_split,
    min_graphs_for_disjoint,
    valid_split_modes,
)

ratios = (0.7, 0.15, 0.15)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("graph_count", [3, 4, 5, 10, 20, 40, 137])
def test_every_graph_gets_exactly_one_split(graph_count):
    plan = _graph_split_plan(graph_count, ratios, seed=0)

    assert len(plan) == graph_count
    assert set(plan) <= {"train", "val", "test"}


@pytest.mark.parametrize("graph_count", [3, 4, 5, 10, 20, 40, 137])
def test_no_split_is_ever_empty(graph_count):
    """
    An empty val split silently disables early stopping; an empty test split
    makes the reported test metrics come from nowhere. Both are worse than a
    slightly-off ratio, so the plan clamps rather than rounds freely.
    """
    plan = _graph_split_plan(graph_count, ratios, seed=0)

    for split in ("train", "val", "test"):
        assert plan.count(split) >= 1, f"{split} empty at graph_count={graph_count}"


def test_proportions_are_honoured_for_a_realistic_graph_count():
    plan = _graph_split_plan(20, ratios, seed=0)

    assert plan.count("train") == 14
    assert plan.count("val") == 3
    assert plan.count("test") == 3


def test_the_plan_is_deterministic_given_the_seed():
    assert _graph_split_plan(20, ratios, 7) == _graph_split_plan(20, ratios, 7)


def test_a_different_seed_gives_a_different_assignment():
    """Otherwise a multi-seed sweep would train on the same split five times."""
    plans = {tuple(_graph_split_plan(20, ratios, seed)) for seed in range(5)}

    assert len(plans) > 1


def test_plan_is_stratified_not_sampled():
    """
    Independent per-graph draws would let train_count wander. Stratification
    pins it, which is the difference between a reproducible split and a lucky one.
    """
    counts = {
        _graph_split_plan(20, ratios, seed).count("train") for seed in range(25)
    }

    assert counts == {14}


# ---------------------------------------------------------------------------
# The legacy mode must not move
# ---------------------------------------------------------------------------


def test_episode_random_is_unchanged():
    """
    Byte-for-byte the historical function. If this drifts, no pre-2026-08-22
    dataset can be regenerated, and every comparison against those numbers
    becomes uncheckable.
    """
    rng = np.random.default_rng(0)
    observed = [_assign_split(rng, ratios) for _ in range(8)]

    expected = []
    replay = np.random.default_rng(0)

    for _ in range(8):
        draw = replay.random()
        expected.append(
            "train" if draw < 0.7 else "val" if draw < 0.85 else "test"
        )

    assert observed == expected


def test_both_modes_are_registered():
    assert set(valid_split_modes) == {graph_disjoint_split, episode_random_split}


# ---------------------------------------------------------------------------
# End to end, on generated data
# ---------------------------------------------------------------------------


def _generate(tmp_path, split_mode: str, num_graphs: int = 4):
    """A tiny real generation run — the only way to test the writer's behaviour."""
    from data.generate_wm_data import GenConfig, run_generation

    config = GenConfig(
        dataset="ba",
        num_graphs=num_graphs,
        syn_nodes=12,
        er_p=0.2,
        models=["IC"],
        prob_model="uniform",
        uniform_p=0.3,
        budget=2,
        budget_pct=None,
        budget_pct_range=None,
        algorithms=["degree"],
        rollouts=2,
        horizon=3,
        inject_p=0.5,
        action_ops=["add_node"],
        weight_lo=0.1,
        weight_hi=0.5,
        cf_prob=1.0,
        cf_branches=2,
        split=ratios,
        seed=11,
        mc_marginals=2,
        out_dir=str(tmp_path / split_mode),
        split_mode=split_mode,
        ba_m=2,
    )

    return run_generation(config), config


@pytest.mark.slow
def test_graph_disjoint_generation_leaks_no_graph(tmp_path):
    from world_model.wm_data import graphs_straddling_splits

    metadata, config = _generate(tmp_path, graph_disjoint_split)

    assert metadata["split_mode"] == graph_disjoint_split
    assert metadata["graphs_straddling_splits"] == []
    assert metadata["split_is_graph_disjoint"] is True
    # Verified against the WRITTEN FILES, not the generator's own claim
    assert graphs_straddling_splits(config.out_dir, "IC") == []


@pytest.mark.slow
def test_episode_random_generation_still_works_and_is_flagged(tmp_path):
    """
    Legacy mode must keep producing data — and must announce that it leaks.

    Not asserting that leakage OCCURS: with few graphs and few episodes a random
    draw can happen to be disjoint. What is asserted is that the metadata tells
    the truth either way, which is what makes an existing dataset auditable.
    """
    from world_model.wm_data import graphs_straddling_splits

    metadata, config = _generate(tmp_path, episode_random_split)

    assert metadata["split_mode"] == episode_random_split
    assert metadata["split_is_graph_disjoint"] == (
        not metadata["graphs_straddling_splits"]
    )
    assert set(metadata["graphs_straddling_splits"]) == set(
        graphs_straddling_splits(config.out_dir, "IC")
    )


@pytest.mark.slow
def test_the_two_modes_differ_only_in_split_labels(tmp_path):
    """
    Same graphs, same budgets, same simulator seeds — only the labels move.

    This is why `_assign_split` is still called in graph_disjoint mode and its
    result discarded: it keeps the `base_rng` stream aligned, so regenerating
    under the corrected split is a controlled change rather than a new dataset.
    """
    disjoint, disjoint_config = _generate(tmp_path, graph_disjoint_split)
    legacy, legacy_config = _generate(tmp_path, episode_random_split)

    assert [g["graph_id"] for g in disjoint["graphs"]] == [
        g["graph_id"] for g in legacy["graphs"]
    ]
    assert [g["budget_k_min"] for g in disjoint["graphs"]] == [
        g["budget_k_min"] for g in legacy["graphs"]
    ]
    assert disjoint["n_episodes"] == legacy["n_episodes"]

    def transitions(out_dir):
        rows = []

        for split in ("train", "val", "test"):
            path = f"{out_dir}/transitions_IC_{split}.jsonl"

            try:
                lines = open(path).read().splitlines()
            except FileNotFoundError:
                continue

            rows += [json.loads(line) for line in lines if line.strip()]

        return sorted(
            (row["episode_id"], row["t"], row["branch"]) for row in rows
        )

    assert transitions(disjoint_config.out_dir) == transitions(legacy_config.out_dir)


def test_single_graph_dataset_refuses_a_disjoint_split(tmp_path):
    """
    One graph cannot be split disjointly. Failing loudly beats emitting a
    train-only dataset that looks like a valid three-way split.
    """
    from data.generate_wm_data import GenConfig, run_generation

    config = GenConfig(
        dataset="ba",
        num_graphs=1,
        syn_nodes=10,
        er_p=0.2,
        models=["IC"],
        prob_model="uniform",
        uniform_p=0.3,
        budget=1,
        budget_pct=None,
        budget_pct_range=None,
        algorithms=["degree"],
        rollouts=1,
        horizon=2,
        inject_p=0.0,
        action_ops=[],
        weight_lo=0.1,
        weight_hi=0.5,
        cf_prob=0.0,
        cf_branches=0,
        split=ratios,
        seed=0,
        mc_marginals=1,
        out_dir=str(tmp_path / "single"),
        split_mode=graph_disjoint_split,
        ba_m=2,
    )

    with pytest.raises(ValueError, match="at least 3 graphs|at least "
                       f"{min_graphs_for_disjoint}"):
        run_generation(config)


def test_unknown_split_mode_is_rejected(tmp_path):
    from data.generate_wm_data import GenConfig, run_generation

    config = GenConfig(
        dataset="ba", num_graphs=4, syn_nodes=10, er_p=0.2, models=["IC"],
        prob_model="uniform", uniform_p=0.3, budget=1, budget_pct=None,
        budget_pct_range=None, algorithms=["degree"], rollouts=1, horizon=2,
        inject_p=0.0, action_ops=[], weight_lo=0.1, weight_hi=0.5, cf_prob=0.0,
        cf_branches=0, split=ratios, seed=0, mc_marginals=1,
        out_dir=str(tmp_path / "bad"), split_mode="whatever", ba_m=2,
    )

    with pytest.raises(ValueError, match="unknown --split-mode"):
        run_generation(config)


# ---------------------------------------------------------------------------
# Training-side detection
# ---------------------------------------------------------------------------


def test_train_reads_the_split_mode_from_metadata(tmp_path):
    from world_model.train_wm import TrainConfig, dataset_split_mode

    (tmp_path / "metadata.json").write_text(
        json.dumps({"split_mode": graph_disjoint_split, "config": {}})
    )

    assert dataset_split_mode(TrainConfig(data_dir=str(tmp_path))) == (
        graph_disjoint_split
    )


def test_a_dataset_from_before_the_flag_reads_as_episode_random(tmp_path):
    """Every pre-2026-08-22 dataset used the per-episode draw and says nothing."""
    from world_model.train_wm import TrainConfig, dataset_split_mode

    (tmp_path / "metadata.json").write_text(json.dumps({"config": {}}))

    assert dataset_split_mode(TrainConfig(data_dir=str(tmp_path))) == (
        episode_random_split
    )


def test_training_on_a_leaky_dataset_warns(tmp_path, capsys):
    from world_model.train_wm import TrainConfig, check_split_mode

    (tmp_path / "metadata.json").write_text(
        json.dumps({"split_mode": episode_random_split, "config": {}})
    )
    check_split_mode(TrainConfig(data_dir=str(tmp_path)))

    assert "optimistic" in capsys.readouterr().out


def test_training_on_a_clean_dataset_is_silent(tmp_path, capsys):
    from world_model.train_wm import TrainConfig, check_split_mode

    (tmp_path / "metadata.json").write_text(
        json.dumps({"split_mode": graph_disjoint_split, "config": {}})
    )
    check_split_mode(TrainConfig(data_dir=str(tmp_path)))

    assert capsys.readouterr().out == ""
