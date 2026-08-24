"""
Self-describing checkpoints and the WorldModelScorer load API.

The bug class these defend against is not a crash. A `state_dict` carries no
architecture, so:

  * a `spent` checkpoint loaded under `blocked` has identically shaped tensors.
    It loads, it runs, and only the containment numbers are wrong.
  * a w-hidden checkpoint rolled out against true edge weights is the same kind
    of silence — and was a real bug in this repository once already, per the
    comment in `WorldModelEnvironment.__init__`.

So the tests below are mostly about what must FAIL, not what must load.
"""

import json

import pytest
import torch

from world_model.checkpoint import (
    LegacyCheckpointError,
    ModelSpec,
    build_model,
    checkpoint_format,
    describe,
    legacy_format,
    load_checkpoint,
    read_checkpoint,
    save_checkpoint,
)
from world_model.scorer import (
    ScoringContext,
    WorldModelScorer,
    as_action_fn,
    candidate_name,
)


@pytest.fixture
def spec() -> ModelSpec:
    return ModelSpec(
        backbone="sage",
        head="structured",
        diffusion_model="IC",
        hidden_dim=16,
        n_layers=2,
        remove_semantics="spent",
    )


@pytest.fixture
def checkpoint(tmp_path, spec):
    model = build_model(spec)
    path = tmp_path / "wm_sage_IC.pt"
    save_checkpoint(model, path, spec, train_meta={"split_mode": "graph_disjoint",
                                                   "seed": 42})

    return path


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_checkpoint_round_trips_without_any_external_config(checkpoint, spec):
    """The whole point: the file alone is enough."""
    model, loaded, meta = load_checkpoint(checkpoint)

    assert loaded == spec
    assert meta["split_mode"] == "graph_disjoint"
    assert model.head_type == spec.head


def test_saved_checkpoint_declares_its_format(checkpoint):
    assert read_checkpoint(checkpoint)["format"] == checkpoint_format


def test_weights_survive_the_round_trip(tmp_path, spec):
    model = build_model(spec)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.5)

    path = tmp_path / "wm.pt"
    save_checkpoint(model, path, spec)
    reloaded, _, _ = load_checkpoint(path)

    for before, after in zip(model.state_dict().values(),
                             reloaded.state_dict().values()):
        assert torch.allclose(before, after)


def test_train_meta_never_affects_reconstruction(tmp_path, spec):
    """Provenance is provenance: a garbage meta block must not change the model."""
    model = build_model(spec)
    clean = tmp_path / "clean.pt"
    noisy = tmp_path / "noisy.pt"
    save_checkpoint(model, clean, spec)
    save_checkpoint(model, noisy, spec, train_meta={"nonsense": [1, 2, 3]})

    assert load_checkpoint(clean)[1] == load_checkpoint(noisy)[1]


@pytest.mark.parametrize("backbone", ["gcn", "sage", "gat", "gt", "gcnii"])
def test_every_backbone_round_trips(tmp_path, backbone):
    spec = ModelSpec(backbone=backbone, head="structured", diffusion_model="IC",
                     hidden_dim=16, n_layers=2)
    path = tmp_path / f"{backbone}.pt"
    save_checkpoint(build_model(spec), path, spec)

    assert load_checkpoint(path)[1].backbone == backbone


def test_lt_checkpoint_rebuilds_the_lt_head(tmp_path):
    spec = ModelSpec(backbone="sage", head="structured", diffusion_model="LT",
                     hidden_dim=16, n_layers=2)
    path = tmp_path / "lt.pt"
    save_checkpoint(build_model(spec), path, spec)
    model, loaded, _ = load_checkpoint(path)

    assert loaded.diffusion_model == "LT"
    assert type(model.head).__name__ == "LTThresholdHead"


# ---------------------------------------------------------------------------
# The silent-mismatch class
# ---------------------------------------------------------------------------


def test_disagreeing_config_is_refused_rather_than_silently_preferred(checkpoint, spec):
    """
    A `spent` checkpoint and a `blocked` config both "work". Refuse.

    This is the regression test for the failure mode the whole module exists to
    close: shapes match, nothing raises, every containment number is biased.
    """
    lying = ModelSpec(**{**spec.to_dict(), "remove_semantics": "blocked"})

    with pytest.raises(ValueError, match="Refusing to guess"):
        load_checkpoint(checkpoint, config=lying)


def test_override_is_possible_but_must_be_explicit(checkpoint, spec):
    lying = ModelSpec(**{**spec.to_dict(), "remove_semantics": "blocked"})
    _, loaded, _ = load_checkpoint(checkpoint, config=lying, strict_spec=False)

    assert loaded.remove_semantics == "blocked"


def test_matching_config_is_accepted(checkpoint, spec):
    _, loaded, _ = load_checkpoint(checkpoint, config=spec)

    assert loaded == spec


def test_hide_edge_weights_travels_with_the_checkpoint(tmp_path):
    """
    A w-hidden model rolled out against true weights was a real past bug. The
    flag now lives in the file, so the rollout cannot disagree with training.
    """
    spec = ModelSpec(backbone="sage", head="structured", diffusion_model="IC",
                     hidden_dim=16, n_layers=2, hide_edge_weights=True)
    path = tmp_path / "hidden.pt"
    save_checkpoint(build_model(spec), path, spec)

    assert load_checkpoint(path)[1].hide_edge_weights is True


def test_action_encoding_travels_with_the_checkpoint(tmp_path):
    spec = ModelSpec(backbone="sage", head="linear", diffusion_model="IC",
                     hidden_dim=16, n_layers=2, action_encoding="typed")
    path = tmp_path / "typed.pt"
    save_checkpoint(build_model(spec), path, spec)
    _, loaded, _ = load_checkpoint(path)

    assert loaded.action_encoding == "typed"
    assert loaded.in_channels == 9


def test_unknown_backbone_and_head_are_rejected_at_spec_construction():
    with pytest.raises(ValueError, match="unknown backbone"):
        ModelSpec(backbone="nope", head="linear", diffusion_model="IC")

    with pytest.raises(ValueError, match="unknown head"):
        ModelSpec(backbone="gcn", head="nope", diffusion_model="IC")

    with pytest.raises(ValueError, match="unknown remove_semantics"):
        ModelSpec(backbone="gcn", head="linear", diffusion_model="IC",
                  remove_semantics="nope")


# ---------------------------------------------------------------------------
# Legacy v1 compatibility
# ---------------------------------------------------------------------------


def test_legacy_bare_state_dict_is_recognised(tmp_path, spec):
    path = tmp_path / "legacy.pt"
    torch.save(build_model(spec).state_dict(), path)

    blob = read_checkpoint(path)

    assert blob["format"] == legacy_format
    assert blob["spec"] is None


def test_legacy_checkpoint_without_config_raises_an_actionable_error(tmp_path, spec):
    path = tmp_path / "legacy.pt"
    torch.save(build_model(spec).state_dict(), path)

    with pytest.raises(LegacyCheckpointError, match="results JSON"):
        load_checkpoint(path)


def test_legacy_checkpoint_loads_when_given_its_config(tmp_path, spec):
    path = tmp_path / "legacy.pt"
    torch.save(build_model(spec).state_dict(), path)
    model, loaded, _ = load_checkpoint(path, config=spec)

    assert loaded == spec
    assert model.head_type == spec.head


def test_legacy_checkpoint_loads_from_a_results_json(tmp_path, spec):
    """The path every pre-v2 checkpoint on the cluster has to come back through."""
    checkpoint_dir = tmp_path / "world_model"
    checkpoint_dir.mkdir()
    torch.save(build_model(spec).state_dict(), checkpoint_dir / "wm_sage_IC.pt")

    results = checkpoint_dir / "sage_IC.json"
    results.write_text(
        json.dumps(
            {
                "config": {
                    "model": "sage",
                    "head": "structured",
                    "diffusion_model": "IC",
                    "hidden_dim": 16,
                    "n_layers": 2,
                    "dropout": 0.1,
                    "remove_semantics": "spent",
                    "action_encoding": "basic",
                    "hide_edge_weights": False,
                    "n_heads": 4,
                    "ffn_dim": 128,
                    "gcnii_alpha": 0.1,
                    "gcnii_lamda": 0.5,
                    "ckpt_dir": str(checkpoint_dir),
                }
            }
        )
    )

    scorer = WorldModelScorer.from_results_json(results)

    assert scorer.spec.backbone == "sage"
    assert scorer.spec.remove_semantics == "spent"


def test_results_json_defaults_reproduce_pre_flag_runs(tmp_path):
    """A config from before --remove-semantics / --action-encoding existed."""
    results = tmp_path / "old.json"
    results.write_text(
        json.dumps(
            {
                "config": {
                    "model": "gcn",
                    "diffusion_model": "IC",
                    "hidden_dim": 16,
                    "n_layers": 2,
                    "dropout": 0.1,
                    "n_heads": 4,
                    "ffn_dim": 128,
                    "gcnii_alpha": 0.1,
                    "gcnii_lamda": 0.5,
                }
            }
        )
    )
    spec = ModelSpec.from_results_json(results)

    assert spec.head == "linear"
    assert spec.remove_semantics == "spent"
    assert spec.action_encoding == "basic"
    assert spec.hide_edge_weights is False


def test_describe_never_raises_on_a_legacy_file(tmp_path, spec):
    path = tmp_path / "legacy.pt"
    torch.save(build_model(spec).state_dict(), path)

    assert legacy_format in describe(path)


def test_a_file_that_is_not_a_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "junk.pt"
    torch.save(["not", "a", "checkpoint"], path)

    with pytest.raises(ValueError, match="neither"):
        read_checkpoint(path)


def test_missing_checkpoint_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_checkpoint(tmp_path / "absent.pt")


# ---------------------------------------------------------------------------
# WorldModelScorer
# ---------------------------------------------------------------------------


def test_scorer_loads_from_a_self_describing_checkpoint(checkpoint):
    scorer = WorldModelScorer.load(checkpoint)

    assert scorer.spec.backbone == "sage"
    assert "sage/structured" in scorer.describe()
    assert scorer.train_meta["seed"] == 42


def test_scorer_predicts_a_transition(checkpoint):
    """One forward pass through the public API, from a non-empty state."""
    import numpy as np

    from coding_agent.types import GraphInfo
    from data.wm_simulator import ActionOp, State

    graph = GraphInfo(
        num_nodes=4,
        edge_index=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        ic_probs=np.array([0.5, 0.5, 0.5], dtype=np.float32),
        directed=True,
    )
    scorer = WorldModelScorer.load(checkpoint)
    prediction = scorer.predict_transition(
        graph, State(infected=[0], frontier=[0]), [ActionOp("add_node", 2)]
    )

    assert prediction["infected"].shape == (4,)
    # T_exo is structural: a seeded node comes out infected regardless of weights
    assert prediction["infected"][2] > 0.99
    # ... and so does one already infected
    assert prediction["infected"][0] > 0.99


def test_scorer_rejects_a_bad_sense(checkpoint):
    scorer = WorldModelScorer.load(checkpoint)

    with pytest.raises(ValueError, match="maximize or minimize"):
        scorer.rank_candidates(None, [], sense="sideways")


# ---------------------------------------------------------------------------
# Candidate adapters
# ---------------------------------------------------------------------------


def test_a_plan_becomes_an_action_fn():
    from data.wm_simulator import ActionOp, State

    plan = [[ActionOp("add_node", 0)], [ActionOp("add_node", 1)]]
    action_fn = as_action_fn(plan)
    state = State([], [])

    assert action_fn(state, 0)[0].target == 0
    assert action_fn(state, 1)[0].target == 1
    # Past the end of a finite plan the strategy stops intervening
    assert action_fn(state, 2) == []


def test_a_callable_candidate_passes_through():
    def candidate(state, timestep):
        return []

    assert as_action_fn(candidate) is candidate


def test_a_non_candidate_is_rejected():
    with pytest.raises(TypeError, match="neither callable"):
        as_action_fn(42)


def test_candidate_name_prefers_a_declared_name():
    def named(state, timestep):
        return []

    assert candidate_name(named, 3) == "named"
    assert candidate_name(object(), 3) == "candidate_3"


def test_scoring_context_coerces_a_dict_and_ignores_extras():
    context = ScoringContext.coerce({"horizon": 5, "budget": 2, "junk": 1})

    assert context.horizon == 5
    assert context.budget == 2
    assert ScoringContext.coerce(None).horizon == 20
