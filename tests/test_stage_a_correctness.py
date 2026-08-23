"""
Stage A correctness: planning-evaluator split awareness, and LT exogenous
semantics.

Both bugs shared a shape: the code ran, produced plausible numbers, and the
numbers meant something other than what they were read as.

  * `planning_regret_multi` scored `list(store)[:n]` — the first n graphs in the
    dataset, training graphs included — while being reported as evidence the
    model plans well on graphs it has not seen.

  * `LTThresholdHead` computed "newly active" against the POST-action state,
    while `Simulator.advance` labels it against the PRE-action snapshot
    (`frontier = active - previous_active`). That is wrong for both node ops at
    once, in opposite directions, and the dataset says so: the true next-frontier
    at an `add_node` target is 1.0 and at a `remove_node` target is 0.0.

The LT assertions below use an UNTRAINED model on purpose. These are structural
properties of T_exo, so they must hold at any parameter value; a test that needed
training would be testing the fit instead of the semantics.
"""

import json

import numpy as np
import pytest
import torch

from world_model.wm_data import (
    build_graph_input,
    ch_add,
    ch_infected,
    ch_remove,
)
from world_model.wm_eval import (
    graphs_in_split,
    planning_split_legacy,
    planning_split_test,
    select_planning_graphs,
)
from world_model.wm_model import WorldModel


# ---------------------------------------------------------------------------
# LT exogenous semantics
# ---------------------------------------------------------------------------


@pytest.fixture
def lt_model():
    return WorldModel(
        "sage",
        hidden_dim=8,
        n_layers=1,
        dropout=0.0,
        head_type="structured",
        diffusion_model="LT",
    ).eval()


@pytest.fixture
def path_graph():
    """0 - 1 - 2 - 3, both directions, unit weights."""
    edge_index = np.array([[0, 1, 2, 1, 2, 3], [1, 2, 3, 0, 1, 2]], dtype=np.int64)

    return build_graph_input(
        edge_index, np.ones(6, dtype=np.float32), 4, "LT", torch.device("cpu")
    )


def _probs(model, graph, infected, add=None, remove=None, num_nodes=4):
    features = torch.zeros(num_nodes, 6)

    for node in infected:
        features[node, ch_infected] = 1.0

    if add is not None:
        features[add, ch_add] = 1.0

    if remove is not None:
        features[remove, ch_remove] = 1.0

    with torch.no_grad():
        return torch.sigmoid(model(features, graph)).numpy()


def test_lt_removed_node_leaves_the_frontier(lt_model, path_graph):
    """
    The bug this file exists for. Measured before the fix: 0.92.

    A node that was active before the action cannot be "newly activated" after
    it, whatever it does to its threshold — `previous_active` already contains it.
    """
    probs = _probs(lt_model, path_graph, infected=[0, 1], remove=0)

    assert probs[0, 1] < 1e-3


def test_lt_seeded_node_enters_the_frontier(lt_model, path_graph):
    """
    The same bug in the other direction, and the one no metric was watching:
    `add_seed_success` only checks the INFECTED channel, so a seeded node
    predicted out of the frontier passed unnoticed.
    """
    probs = _probs(lt_model, path_graph, infected=[0], add=2)

    assert probs[2, 0] > 0.99
    assert probs[2, 1] > 0.99


def test_lt_already_active_node_is_not_in_the_frontier(lt_model, path_graph):
    probs = _probs(lt_model, path_graph, infected=[0])

    assert probs[0, 0] > 0.99
    assert probs[0, 1] < 1e-3


def test_lt_plain_new_activation_has_frontier_equal_to_infected(lt_model, path_graph):
    """A susceptible node's two channels must agree: becoming active IS new."""
    probs = _probs(lt_model, path_graph, infected=[0])

    assert probs[1, 1] == pytest.approx(probs[1, 0], abs=1e-6)


@pytest.mark.parametrize("scale", [1.0, 50.0, -50.0, 0.0])
def test_lt_exogenous_semantics_survive_weight_perturbation(path_graph, scale):
    """
    Structural, not learned. If any of these moved under a x(-50) perturbation
    the property would be a fitting artefact rather than a guarantee.
    """
    model = WorldModel(
        "sage", hidden_dim=8, n_layers=1, dropout=0.0,
        head_type="structured", diffusion_model="LT",
    ).eval()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(scale)

    seeded = _probs(model, path_graph, infected=[0], add=2)
    removed = _probs(model, path_graph, infected=[0, 1], remove=0)

    assert seeded[2, 0] > 0.99 and seeded[2, 1] > 0.99
    assert removed[0, 1] < 1e-3


def test_ic_frontier_semantics_are_unchanged(path_graph):
    """
    IC must NOT get the LT treatment. Its frontier is the set of status-1
    spreaders after the step, so a seeded node is correctly OUT of it — the
    dataset's true next-frontier at an IC `add_node` target is 0.0.
    """
    graph = build_graph_input(
        np.array([[0, 1, 2, 1, 2, 3], [1, 2, 3, 0, 1, 2]], dtype=np.int64),
        np.full(6, 0.5, dtype=np.float32),
        4,
        "IC",
        torch.device("cpu"),
    )
    model = WorldModel(
        "sage", hidden_dim=8, n_layers=1, dropout=0.0,
        head_type="structured", diffusion_model="IC",
    ).eval()
    probs = _probs(model, graph, infected=[0], add=2)

    assert probs[2, 0] > 0.99
    assert probs[2, 1] < 1e-3


# ---------------------------------------------------------------------------
# Planning evaluator split awareness
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset_dir(tmp_path):
    """Five graphs, split 3/1/1, written in the on-disk transition format."""
    membership = {
        "g0": "train", "g1": "train", "g2": "train", "g3": "val", "g4": "test",
    }
    lines = {"train": [], "val": [], "test": []}

    for graph_id, split in membership.items():
        lines[split].append(
            json.dumps({"graph_id": graph_id, "episode_id": f"{graph_id}|e0",
                        "t": 0, "branch": "main"})
        )

    for split, rows in lines.items():
        (tmp_path / f"transitions_IC_{split}.jsonl").write_text("\n".join(rows))

    return tmp_path


@pytest.fixture
def store():
    return {f"g{i}": {"num_nodes": 10} for i in range(5)}


def test_graphs_in_split_reads_the_files(dataset_dir):
    assert graphs_in_split(dataset_dir, "IC", "train") == ["g0", "g1", "g2"]
    assert graphs_in_split(dataset_dir, "IC", "test") == ["g4"]


def test_test_mode_selects_only_test_graphs(store, dataset_dir):
    graph_ids, provenance = select_planning_graphs(store, 5, dataset_dir, "IC")

    assert graph_ids == ["g4"]
    assert provenance["planning_split"] == planning_split_test
    assert provenance["train_overlap"] == 0
    assert provenance["val_overlap"] == 0
    assert provenance["n_planning_graphs"] == 1


def test_legacy_mode_reproduces_the_old_selection(store, dataset_dir):
    """The historical behaviour stays reachable — under its own name."""
    graph_ids, provenance = select_planning_graphs(
        store, 3, dataset_dir, "IC", planning_split_legacy
    )

    assert graph_ids == ["g0", "g1", "g2"]
    assert provenance["planning_split"] == planning_split_legacy


def test_legacy_mode_reports_unknown_overlap_not_zero(store, dataset_dir):
    """
    Reporting 0 would assert something never checked. None says "not measured",
    which is the honest value for a selection that ignores splits.
    """
    _, provenance = select_planning_graphs(
        store, 3, dataset_dir, "IC", planning_split_legacy
    )

    assert provenance["train_overlap"] is None
    assert provenance["val_overlap"] is None


def test_the_old_selection_would_have_leaked(store, dataset_dir):
    """
    States the bug directly: the historical first-n choice picks training graphs,
    so its planning regret was never a held-out number.
    """
    legacy_ids, _ = select_planning_graphs(
        store, 3, dataset_dir, "IC", planning_split_legacy
    )
    train_graphs = set(graphs_in_split(dataset_dir, "IC", "train"))

    assert set(legacy_ids) & train_graphs


def test_missing_out_dir_falls_back_to_legacy_and_says_so(store):
    """Split membership is not knowable from the store alone."""
    _, provenance = select_planning_graphs(store, 3, None, "IC")

    assert provenance["planning_split"] == planning_split_legacy
    assert provenance["train_overlap"] is None


def test_no_test_graphs_raises_rather_than_silently_using_train(store, tmp_path):
    """
    A single-graph or episode_random dataset has no held-out graphs. Falling back
    to training graphs would produce a number that reads as held-out.
    """
    (tmp_path / "transitions_IC_train.jsonl").write_text(
        json.dumps({"graph_id": "g0", "episode_id": "g0|e0", "t": 0,
                    "branch": "main"})
    )

    with pytest.raises(ValueError, match="no graph in the store"):
        select_planning_graphs(store, 3, tmp_path, "IC")


def test_unknown_planning_split_is_rejected(store, dataset_dir):
    with pytest.raises(ValueError, match="unknown planning_split"):
        select_planning_graphs(store, 3, dataset_dir, "IC", "whatever")


def test_selection_is_deterministic(store, dataset_dir):
    first = select_planning_graphs(store, 5, dataset_dir, "IC")[0]
    second = select_planning_graphs(store, 5, dataset_dir, "IC")[0]

    assert first == second


def test_train_config_defaults_to_the_test_split():
    """A run that forgets the flag must get the held-out metric, not the leaky one."""
    from world_model.train_wm import TrainConfig

    assert TrainConfig(data_dir="x").planning_split == planning_split_test
