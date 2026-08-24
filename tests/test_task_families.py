"""
Task-family classification.

Every assertion here is about a CLASSIFICATION, and the point of the module under
test is that classifications are derived rather than declared. So these tests
pin the derivation's consequences, not a lookup table: if someone changes
`remove_semantics` on a task, `test_im_to_cnd_is_not_exact_checkpoint` should
start failing, because the transfer claim genuinely changed.

The negative tests carry the weight. Getting IM -> Adaptive IM right is easy;
what protects the paper is that IM -> CND, CND -> Influence Blocking and
IM -> Epidemic Control cannot quietly become EXACT_CHECKPOINT.
"""

import pytest

from pipeline.tasks import tasks
from registry.task_families import (
    Compatibility,
    compatibility,
    competitive_cascade,
    compartmental,
    containment_overlay,
    exact_checkpoint,
    families,
    forward_dynamics,
    incompatible,
    mechanism_only,
    observational,
    requires_actions,
    single_cascade,
    state_layout,
    transfer_matrix,
    world_family,
)

eight_tasks = (
    "influence_maximization",
    "adaptive_online_im",
    "source_localization",
    "cascade_reconstruction",
    "cascade_prediction",
    "critical_node_detection",
    "influence_blocking",
    "epidemic_control",
)


def level(source: str, target: str) -> str:
    return compatibility(tasks[source], tasks[target]).level


# ---------------------------------------------------------------------------
# World families
# ---------------------------------------------------------------------------


def test_world_families_partition_the_eight_tasks():
    grouped = families()
    placed = {name for members in grouped.values() for name in members}

    for name in eight_tasks:
        assert name in placed


@pytest.mark.parametrize(
    "task,family",
    [
        ("influence_maximization", single_cascade),
        ("adaptive_online_im", single_cascade),
        ("source_localization", single_cascade),
        ("cascade_reconstruction", single_cascade),
        ("cascade_prediction", single_cascade),
        ("critical_node_detection", single_cascade),
        ("influence_blocking", competitive_cascade),
        ("epidemic_control", compartmental),
    ],
)
def test_world_family_assignment(task, family):
    assert world_family(tasks[task]) == family


def test_state_layouts_differ_across_families():
    assert state_layout(tasks["influence_maximization"]) == (6, 2)
    assert state_layout(tasks["influence_blocking"]) == (8, 4)
    assert state_layout(tasks["epidemic_control"]) == (9, 5)


def test_inverse_and_forecast_tasks_inject_no_actions():
    """
    The property that separates FORWARD_DYNAMICS from EXACT_CHECKPOINT.

    Source localization DECLARES add_node — a recovered source set is replayed as
    a seed bag — but its generator injects nothing mid-cascade, so a transferred
    model is never asked an action-conditioned question there.
    """
    for name in ("source_localization", "cascade_reconstruction",
                 "cascade_prediction"):
        assert not requires_actions(tasks[name])

    for name in ("influence_maximization", "adaptive_online_im",
                 "critical_node_detection"):
        assert requires_actions(tasks[name])


def test_only_cascade_prediction_is_fully_observational():
    assert observational(tasks["cascade_prediction"])
    assert not observational(tasks["source_localization"])


# ---------------------------------------------------------------------------
# The classifications the brief names
# ---------------------------------------------------------------------------


def test_im_and_adaptive_im_are_exact_checkpoint_both_directions():
    """The headline cross-task claim, and the only pair that earns it."""
    assert level("influence_maximization", "adaptive_online_im") == exact_checkpoint
    assert level("adaptive_online_im", "influence_maximization") == exact_checkpoint


def test_exact_checkpoint_rebuilds_nothing():
    report = compatibility(
        tasks["influence_maximization"], tasks["adaptive_online_im"]
    )

    assert report.rebuilt_components == ()
    assert "prediction_head" in report.reusable_components


def test_im_to_source_localization_is_forward_dynamics():
    assert level("influence_maximization", "source_localization") == forward_dynamics


def test_im_to_cascade_reconstruction_is_forward_dynamics():
    assert (
        level("influence_maximization", "cascade_reconstruction") == forward_dynamics
    )


def test_im_to_cascade_prediction_is_forward_dynamics_with_domain_shift():
    report = compatibility(
        tasks["influence_maximization"], tasks["cascade_prediction"]
    )

    assert report.level == forward_dynamics
    assert report.observational_shift
    assert "observational" in report.reason


def test_im_to_cnd_is_not_exact_checkpoint():
    """
    Same 6/2 layout, same IC/LT — and still not checkpoint-compatible, because
    T_exo branches on remove_semantics inside the head. Shape is necessary, not
    sufficient, and this is the test that says so.
    """
    report = compatibility(
        tasks["influence_maximization"], tasks["critical_node_detection"]
    )

    assert report.level == mechanism_only
    assert report.same_state_layout
    assert not report.same_remove_semantics
    assert "T_exo_semantics" in report.rebuilt_components


def test_cnd_to_influence_blocking_is_not_exact_checkpoint():
    report = compatibility(
        tasks["critical_node_detection"], tasks["influence_blocking"]
    )

    assert report.level == mechanism_only
    assert not report.same_state_layout


def test_cnd_to_epidemic_control_is_mechanism_only_via_containment():
    """
    No shared dynamics, so nothing FITTED transfers — but both spend budget on
    `blocked` removals to suppress a spreading process, which is the §7 overlay.
    """
    report = compatibility(
        tasks["critical_node_detection"], tasks["epidemic_control"]
    )

    assert report.level == mechanism_only
    assert report.shared_dynamics == ()
    assert "action_semantics" in report.reusable_components
    assert "T_endo" in report.rebuilt_components


def test_im_to_epidemic_control_is_incompatible():
    """IM is not a containment task, so it does not get the overlay's benefit."""
    assert level("influence_maximization", "epidemic_control") == incompatible


def test_no_exact_checkpoint_across_world_families():
    for source in eight_tasks:
        for target in eight_tasks:
            if source == target:
                continue

            if world_family(tasks[source]) != world_family(tasks[target]):
                assert level(source, target) != exact_checkpoint


def test_no_exact_checkpoint_when_remove_semantics_differ():
    for source in eight_tasks:
        for target in eight_tasks:
            if source == target:
                continue

            if tasks[source].remove_semantics != tasks[target].remove_semantics:
                assert level(source, target) != exact_checkpoint


# ---------------------------------------------------------------------------
# Matrix shape and hygiene
# ---------------------------------------------------------------------------


def test_matrix_covers_every_ordered_pair():
    matrix = transfer_matrix(list(eight_tasks))

    assert len(matrix) == len(eight_tasks) * (len(eight_tasks) - 1)
    assert len({(c.source, c.target) for c in matrix}) == len(matrix)


def test_matrix_is_directional():
    """Transfer need not be symmetric, so the matrix must not assume it is."""
    matrix = {(c.source, c.target): c.level for c in transfer_matrix(list(eight_tasks))}
    asymmetric = [
        (s, t) for (s, t) in matrix if matrix[(s, t)] != matrix.get((t, s))
    ]

    assert asymmetric, "a fully symmetric matrix would mean direction is ignored"


def test_every_pair_carries_a_reason():
    for report in transfer_matrix(list(eight_tasks)):
        assert report.reason
        assert report.level in (
            exact_checkpoint, forward_dynamics, mechanism_only, incompatible,
        )


def test_mechanism_only_always_states_what_is_rebuilt():
    """MECHANISM_ONLY without a parameter-group split is not a usable claim."""
    for report in transfer_matrix(list(eight_tasks)):
        if report.level == mechanism_only:
            assert report.reusable_components
            assert report.rebuilt_components


def test_incompatible_transfers_nothing():
    for report in transfer_matrix(list(eight_tasks)):
        if report.level == incompatible:
            assert report.reusable_components == ()


def test_containment_overlay_spans_three_world_families():
    """
    The overlay exists to make a shared IDEA visible without it being read as
    checkpoint compatibility — so it must not collapse into one family.
    """
    overlay = [t for t in containment_overlay() if t in eight_tasks]

    assert set(overlay) == {
        "critical_node_detection", "influence_blocking", "epidemic_control",
    }
    assert len({world_family(tasks[name]) for name in overlay}) == 3


def test_compatibility_is_serialisable():
    report = compatibility(
        tasks["influence_maximization"], tasks["adaptive_online_im"]
    )
    blob = report.to_dict()

    assert blob["level"] == exact_checkpoint
    assert blob["source"] == "influence_maximization"
    assert isinstance(Compatibility(**blob), Compatibility)


# ---------------------------------------------------------------------------
# Zero-shot transfer must actually be zero-shot
# ---------------------------------------------------------------------------


def test_frozen_checkpoint_parameters_do_not_change_during_transfer(tmp_path):
    """
    The claim `freeze_world_model: true` makes, checked rather than asserted.

    Loading a checkpoint, scoring with it, and loading it again must give
    bit-identical parameters. A scorer that silently put the module in train mode
    (dropout active) or that any caller could step an optimizer through would
    fail here.
    """
    import torch

    from world_model.checkpoint import ModelSpec, build_model, save_checkpoint
    from world_model.scorer import WorldModelScorer

    spec = ModelSpec(backbone="sage", head="structured", diffusion_model="IC",
                     hidden_dim=16, n_layers=2)
    path = tmp_path / "frozen.pt"
    save_checkpoint(build_model(spec), path, spec)

    scorer = WorldModelScorer.load(path)
    before = {k: v.clone() for k, v in scorer.model.state_dict().items()}

    import numpy as np

    from coding_agent.types import GraphInfo
    from data.wm_simulator import ActionOp, State

    graph = GraphInfo(
        num_nodes=4,
        edge_index=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        ic_probs=np.array([0.5, 0.5, 0.5], dtype=np.float32),
        directed=True,
    )
    scorer.predict_transition(graph, State([0], [0]), [ActionOp("add_node", 2)])

    after = scorer.model.state_dict()

    for name, tensor in before.items():
        assert torch.equal(tensor, after[name]), f"{name} changed during scoring"


def test_scorer_loads_in_eval_mode():
    """Dropout active during transfer would make the frozen model stochastic."""

    from world_model.checkpoint import ModelSpec, build_model, save_checkpoint
    from world_model.scorer import WorldModelScorer
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        spec = ModelSpec(backbone="sage", head="structured", diffusion_model="IC",
                         hidden_dim=16, n_layers=2, dropout=0.5)
        path = Path(directory) / "m.pt"
        save_checkpoint(build_model(spec), path, spec)

        assert not WorldModelScorer.load(path).model.training


def test_transfer_config_declares_zero_shot():
    """
    The shipped IM -> Adaptive IM config must not be able to drift into
    fine-tuning without the test noticing.
    """
    from pathlib import Path

    import yaml

    config = yaml.safe_load(
        (Path(__file__).resolve().parent.parent
         / "configs" / "task_transfer" / "im_to_adaptive.yaml").read_text()
    )

    assert config["transfer_level"] == exact_checkpoint
    assert config["freeze_world_model"] is True
    assert config["target_finetuning"] is False
    assert config["required_semantic_overrides"] == 0
    assert config["reinitialized_components"] == []
    assert (
        level(config["source_task"], config["target_task"])
        == config["transfer_level"]
    )


def test_a_disagreeing_spec_cannot_be_applied_silently(tmp_path):
    """
    `required_semantic_overrides: 0` is enforceable only because the loader
    refuses a mismatch by default. Without this, a transfer run could quietly
    flip remove_semantics and still call itself zero-shot.
    """
    import pytest as _pytest

    from world_model.checkpoint import ModelSpec, build_model, load_checkpoint, save_checkpoint

    spec = ModelSpec(backbone="sage", head="structured", diffusion_model="IC",
                     hidden_dim=16, n_layers=2, remove_semantics="spent")
    path = tmp_path / "m.pt"
    save_checkpoint(build_model(spec), path, spec)
    blocked_spec = ModelSpec(**{**spec.to_dict(), "remove_semantics": "blocked"})

    with _pytest.raises(ValueError, match="Refusing to guess"):
        load_checkpoint(path, config=blocked_spec)
