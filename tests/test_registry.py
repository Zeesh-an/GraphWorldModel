"""
Registry alignment and group counting.

These are the Phase-1 regression tests. What they defend:

  * the world model and the coding agent name the SAME algorithms. The two sides
    spell them differently (`degree` vs `high_degree`), so this is a real
    property that a rename on either side breaks — and until the alias map
    existed, breaking it produced no failure anywhere.

  * every count the project quotes is derived. A test that asserts "there are 67
    algorithms" would be exactly the hard-coded number §3 forbids, so these
    assert INVARIANTS (the manifest agrees with the registries, the group count
    equals the factorization) rather than values.
"""

import pytest

from registry import (
    ALGORITHM_REGISTRY,
    BASELINE_REGISTRY,
    DYNAMICS_REGISTRY,
    GRAPH_REGISTRY,
    TASK_REGISTRY,
    ExperimentGroup,
    UnknownGroupField,
    build_manifest,
    counts,
    enumerate_groups,
    explain_group_count,
    implemented_tasks,
    planner_tool,
    validate_registry_consistency,
    wm_spine,
    wm_spine_aliases,
)
from registry.core import ACTION_REGISTRY


# ---------------------------------------------------------------------------
# Registry consistency (§27 "coding-agent algorithms == registered algorithms")
# ---------------------------------------------------------------------------


def test_registry_consistency_has_no_errors():
    report = validate_registry_consistency()
    assert report.ok, "\n".join(str(finding) for finding in report.errors)


def test_every_world_model_spine_algorithm_is_emittable_by_the_coding_agent():
    """
    The §4 alignment property, stated directly.

    The world model is trained on trajectories produced by `spine_algorithms`. If
    one of those is not something the coding agent can propose, the world model
    was trained on a distribution the agent cannot generate.
    """
    from data.wm_actions import spine_algorithms

    for name in spine_algorithms:
        alias = wm_spine_aliases.get(name)
        assert alias is not None, f"spine algorithm {name!r} has no planner alias"
        assert alias in ALGORITHM_REGISTRY, f"{name!r} -> {alias!r} is not registered"
        assert planner_tool in ALGORITHM_REGISTRY[alias].roles


def test_spine_aliases_are_not_silently_dropped():
    """§4: 'Do not silently drop algorithms.' Every alias target keeps both roles."""
    spine_entries = [
        entry for entry in ALGORITHM_REGISTRY.values() if wm_spine in entry.roles
    ]

    from data.wm_actions import spine_algorithms

    assert len(spine_entries) == len(spine_algorithms)

    for entry in spine_entries:
        assert planner_tool in entry.roles
        assert "data.wm_actions.spine_algorithms" in entry.source


def test_every_task_declares_executable_action_ops():
    for name, task in TASK_REGISTRY.items():
        unknown = set(task.action_ops) - set(ACTION_REGISTRY)
        assert not unknown, f"task {name} declares unexecutable ops {unknown}"


def test_every_implemented_task_has_a_simulated_dynamics():
    for name in implemented_tasks():
        simulated = [
            d for d in TASK_REGISTRY[name].dynamics if DYNAMICS_REGISTRY[d].simulated
        ]
        assert simulated, f"implemented task {name} has no simulated dynamics"


def test_every_external_baseline_targets_a_registered_task():
    for name, baseline in BASELINE_REGISTRY.items():
        assert baseline.task in TASK_REGISTRY, f"{name} -> unknown task {baseline.task}"


def test_task_baseline_pools_reference_registered_algorithms():
    """A rename in coding_agent/tools must not leave a task's pool dangling."""
    from pipeline.conditions import default_baselines as fallback

    for name, task in TASK_REGISTRY.items():
        pool = task.default_baselines or (fallback if task.runnable else ())

        for algorithm in pool:
            assert algorithm in ALGORITHM_REGISTRY, (
                f"task {name} lists baseline {algorithm!r} which is in no "
                f"coding_agent tool pool"
            )


# ---------------------------------------------------------------------------
# Manifest (§27 "generated experiment-group count matches the manifest exactly")
# ---------------------------------------------------------------------------


def test_manifest_counts_match_the_live_registries():
    """
    No count in the manifest may be independent of the registry it describes.

    This is what makes it safe for a README or a Slurm record to quote the
    manifest: the two cannot disagree.
    """
    manifest = build_manifest(include_backbones=False)
    reported = manifest["counts"]

    assert reported["tasks_total"] == len(TASK_REGISTRY)
    assert reported["tasks_implemented"] == len(implemented_tasks())
    assert reported["algorithms_total"] == len(ALGORITHM_REGISTRY)
    assert reported["graphs_total"] == len(GRAPH_REGISTRY)
    assert reported["dynamics_total"] == len(DYNAMICS_REGISTRY)
    assert reported["baselines_total"] == len(BASELINE_REGISTRY)
    assert reported["action_ops"] == len(ACTION_REGISTRY)

    assert reported["algorithms_wm_spine"] + 0 == sum(
        wm_spine in entry.roles for entry in ALGORITHM_REGISTRY.values()
    )
    assert reported == counts()


def test_manifest_digest_is_stable_across_calls():
    assert (
        build_manifest(include_backbones=False)["digest"]
        == build_manifest(include_backbones=False)["digest"]
    )


def test_manifest_digest_ignores_consistency_findings():
    """
    The digest identifies the VOCABULARY, so a job that records it is recording
    what the names meant — not how many warnings the checker happened to emit.
    """
    manifest = build_manifest(include_backbones=False)
    mutated = dict(manifest)
    mutated["consistency"] = {"ok": False, "n_errors": 99, "n_warnings": 0,
                              "findings": []}

    from registry.manifest import digest

    assert digest(mutated) == manifest["digest"]


# ---------------------------------------------------------------------------
# Experiment groups
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> dict:
    return {
        "tasks": ["influence_maximization"],
        "datasets": {"influence_maximization": ["ba", "ws"]},
        "dynamics": {"influence_maximization": ["IC", "LT"]},
        "arms": ["evolve_free@world_model", "evolve_free@oracle"],
        "seeds": [0, 1, 2, 3, 4],
    }


def test_group_count_equals_the_factorization(config):
    """2 graphs x 2 dynamics x 2 arms x 5 seeds = 20, computed, never typed in."""
    groups = enumerate_groups(config)
    expected = (
        len(config["datasets"]["influence_maximization"])
        * len(config["dynamics"]["influence_maximization"])
        * len(config["arms"])
        * len(config["seeds"])
    )

    assert len(groups) == expected
    assert str(expected) in explain_group_count(config)


def test_groups_are_unique_and_sorted(config):
    groups = enumerate_groups(config)
    assert len(set(groups)) == len(groups)
    assert groups == sorted(groups)


def test_duplicate_seeds_do_not_launch_duplicate_runs(config):
    config["seeds"] = [0, 0, 1]
    groups = enumerate_groups(config)
    assert len({group.seed for group in groups}) == 2
    assert len(groups) == len(set(groups))


def test_run_id_is_unique_per_group(config):
    groups = enumerate_groups(config)
    assert len({group.run_id for group in groups}) == len(groups)


def test_run_id_is_filesystem_safe(config):
    for group in enumerate_groups(config):
        assert "@" not in group.run_id
        assert "/" not in group.run_id
        assert ":" not in group.run_id


def test_unknown_graph_is_rejected(config):
    config["datasets"]["influence_maximization"] = ["not_a_graph"]

    with pytest.raises(UnknownGroupField, match="datasets"):
        enumerate_groups(config)


def test_unknown_task_is_rejected(config):
    config["tasks"] = ["not_a_task"]

    with pytest.raises(UnknownGroupField, match="tasks"):
        enumerate_groups(config)


def test_dynamics_the_task_does_not_declare_is_rejected(config):
    """
    Running SIR under influence_maximization would produce a results row filed
    under a task whose research doc never claimed that dynamics.
    """
    config["dynamics"]["influence_maximization"] = ["SIR"]

    with pytest.raises(UnknownGroupField, match="declares dynamics"):
        enumerate_groups(config)


def test_empty_sweep_is_rejected(config):
    config["seeds"] = []

    with pytest.raises(UnknownGroupField):
        enumerate_groups(config)


def test_experiment_group_is_hashable_and_ordered():
    first = ExperimentGroup("influence_maximization", "ba", "IC", "a", 0)
    second = ExperimentGroup("influence_maximization", "ba", "IC", "a", 1)

    assert {first, second, first} == {first, second}
    assert first < second


def test_shipped_config_enumerates(tmp_path):
    """configs/wm_main.yaml must stay loadable and expandable."""
    import pathlib

    import yaml

    path = pathlib.Path(__file__).resolve().parent.parent / "configs" / "wm_main.yaml"
    shipped = yaml.safe_load(path.read_text())
    groups = enumerate_groups(shipped)

    assert groups
    assert all(group.task in TASK_REGISTRY for group in groups)


# ---------------------------------------------------------------------------
# Split buckets (leakage control) — a different unit, deliberately
# ---------------------------------------------------------------------------


def test_split_group_key_buckets_by_graph_not_episode():
    """
    Two episodes on the same graph share structure and edge probabilities, so
    they belong to ONE split bucket. `_assign_split` currently draws per episode;
    this test states the contract that fix has to satisfy.
    """
    from registry import split_group_key, split_groups

    same_graph = [
        {"graph_id": "ba_0", "episode_id": "ba_0|IC|degree|k5|r0"},
        {"graph_id": "ba_0", "episode_id": "ba_0|IC|celf|k5|r1"},
    ]
    other = [{"graph_id": "ba_1", "episode_id": "ba_1|IC|degree|k5|r0"}]

    assert split_group_key(same_graph[0]) == split_group_key(same_graph[1])
    assert split_groups(same_graph + other) == {"ba_0", "ba_1"}
