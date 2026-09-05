"""Feature builder, adjacency reconstruction, and collate."""

from pathlib import Path
import numpy as np
import pytest
import torch

from world_model.wm_data import (
    TransitionDataset,
    basic_encoding,
    build_features,
    build_graph_input,
    ch_add,
    ch_degree,
    ch_edge,
    ch_edge_add,
    ch_edge_del,
    ch_edge_reweight,
    ch_frontier,
    ch_infected,
    ch_remove,
    collate_transitions,
    edges_to_arrays,
    num_input_channels,
    reconstruct_episode_adjacency,
    typed_encoding,
)


def _record(action, infected=(1,), frontier=(1,)) -> dict:
    return {
        "state": {"infected": list(infected), "frontier": list(frontier)},
        "action": action,
        "next_marginal_infected": {"1": 1.0, "2": 0.5},
        "next_marginal_frontier": {"2": 0.5},
    }


class TestBuildFeatures:
    def test_state_and_action_channels(self, edge_index: np.ndarray) -> None:
        X, y_inf, y_fr = build_features(
            _record([{"op": "add_node", "target": 3}]), edge_index, 6
        )

        assert X.shape == (6, 6)
        assert X[1, ch_infected] == 1.0
        assert X[1, ch_frontier] == 1.0
        assert X[3, ch_add] == 1.0
        assert X[3, ch_remove] == 0.0
        # Soft targets are scattered from the sparse marginal dicts
        assert y_inf[1] == pytest.approx(1.0)
        assert y_inf[2] == pytest.approx(0.5)
        assert y_fr[2] == pytest.approx(0.5)
        assert y_inf[0] == 0.0

    def test_degree_channel_is_log1p_total_degree(self, edge_index: np.ndarray) -> None:
        X, _, _ = build_features(_record([]), edge_index, 6)

        # Node 1 has neighbours 0, 2, 4; both arc directions are counted, so the
        # channel is log1p(in + out) = log1p(6), not log1p(3)
        assert X[1, ch_degree] == pytest.approx(np.log1p(6.0))
        assert X[0, ch_degree] == pytest.approx(np.log1p(2.0))

    def test_missing_soft_targets_is_an_error(self, edge_index: np.ndarray) -> None:
        record = _record([])
        del record["next_marginal_infected"]

        with pytest.raises(KeyError, match="soft marginal targets"):
            build_features(record, edge_index, 6)

    def test_edge_op_marks_both_endpoints(self, edge_index: np.ndarray) -> None:
        X, _, _ = build_features(
            _record([{"op": "remove_edge", "target": 4, "destination": 5}]),
            edge_index,
            6,
        )

        assert X[4, ch_edge] == 1.0
        assert X[5, ch_edge] == 1.0
        assert X[0, ch_edge] == 0.0


class TestTypedActionEncoding:
    """The gap `typed` closes: under `basic`, add_edge and remove_edge at the same
    endpoints produce byte-identical features."""

    def test_basic_cannot_distinguish_edge_ops(self, edge_index: np.ndarray) -> None:
        added, _, _ = build_features(
            _record([{"op": "add_edge", "target": 0, "destination": 3, "weight": 0.4}]),
            edge_index,
            6,
            basic_encoding,
        )
        removed, _, _ = build_features(
            _record([{"op": "remove_edge", "target": 0, "destination": 3}]),
            edge_index,
            6,
            basic_encoding,
        )

        assert np.array_equal(added, removed)

    def test_typed_distinguishes_edge_ops(self, edge_index: np.ndarray) -> None:
        added, _, _ = build_features(
            _record([{"op": "add_edge", "target": 0, "destination": 3, "weight": 0.4}]),
            edge_index,
            6,
            typed_encoding,
        )
        removed, _, _ = build_features(
            _record([{"op": "remove_edge", "target": 0, "destination": 3}]),
            edge_index,
            6,
            typed_encoding,
        )
        reweighted, _, _ = build_features(
            _record(
                [
                    {
                        "op": "set_edge_weight",
                        "target": 0,
                        "destination": 3,
                        "weight": 0.9,
                    }
                ]
            ),
            edge_index,
            6,
            typed_encoding,
        )

        assert not np.array_equal(added, removed)
        assert added[0, ch_edge_add] == 1.0 and added[0, ch_edge_del] == 0.0
        assert removed[0, ch_edge_del] == 1.0 and removed[0, ch_edge_add] == 0.0
        assert reweighted[0, ch_edge_reweight] == 1.0
        # CH_EDGE stays set under typed, so the two encodings are nested
        for X in (added, removed, reweighted):
            assert X[0, ch_edge] == 1.0

    def test_typed_is_a_superset_of_basic(self, edge_index: np.ndarray) -> None:
        action = [{"op": "add_edge", "target": 0, "destination": 3, "weight": 0.4}]
        basic, _, _ = build_features(_record(action), edge_index, 6, basic_encoding)
        typed, _, _ = build_features(_record(action), edge_index, 6, typed_encoding)

        assert typed.shape[1] == num_input_channels(typed_encoding) == 9
        assert np.array_equal(typed[:, :6], basic)

    def test_unknown_encoding_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown action_encoding"):
            num_input_channels("categorical")


class TestGraphInput:
    def test_adjacency_aggregates_from_in_neighbours(self, edge_index: np.ndarray) -> None:
        weights = np.full(edge_index.shape[1], 0.5, dtype=np.float32)
        graph = build_graph_input(
            edge_index, weights, 6, "IC", torch.device("cpu")
        )

        dense = graph.adj_norm.to_dense()
        # row = dst, col = src: arc 0 -> 1 must land at [1, 0]
        assert dense[1, 0] > 0
        # Self-loops are added by the renormalisation
        assert dense[0, 0] > 0
        assert graph.edge_weight.min() == pytest.approx(0.5)

    def test_lt_discards_edge_weights(self, edge_index: np.ndarray) -> None:
        weights = np.full(edge_index.shape[1], 0.25, dtype=np.float32)
        graph = build_graph_input(
            edge_index, weights, 6, "LT", torch.device("cpu")
        )

        assert torch.allclose(graph.edge_weight, torch.ones_like(graph.edge_weight))

    def test_hide_edge_weights_masks_ic_too(self, edge_index: np.ndarray) -> None:
        weights = np.full(edge_index.shape[1], 0.25, dtype=np.float32)
        graph = build_graph_input(
            edge_index, weights, 6, "IC", torch.device("cpu"), hide_edge_weights=True
        )

        assert torch.allclose(graph.edge_weight, torch.ones_like(graph.edge_weight))

    def test_empty_edge_list_is_survivable(self) -> None:
        graph = build_graph_input(
            np.zeros((2, 0), dtype=np.int64),
            np.zeros(0, dtype=np.float32),
            4,
            "IC",
            torch.device("cpu"),
        )

        assert graph.edge_index.shape == (2, 0)
        # Self-loops alone still give a well-formed normalised adjacency
        assert graph.adj_norm.to_dense().diagonal().min() > 0


class TestEpisodeAdjacency:
    def test_no_edge_ops_reuses_base(self, base_edges: dict) -> None:
        records = [
            {"t": 0, "branch": "main", "action": [{"op": "add_node", "target": 1}]},
            {"t": 1, "branch": "main", "action": []},
        ]
        adjacency = reconstruct_episode_adjacency(records, base_edges)

        assert adjacency[(0, "main")] is base_edges
        assert adjacency[(1, "main")] is base_edges

    def test_main_sees_post_action_and_cf_sees_pre_step(self, base_edges: dict) -> None:
        records = [
            {
                "t": 0,
                "branch": "main",
                "action": [{"op": "remove_edge", "target": 0, "destination": 1}],
            },
            {"t": 0, "branch": "cf_0", "action": []},
            {"t": 1, "branch": "main", "action": []},
        ]
        adjacency = reconstruct_episode_adjacency(records, base_edges)

        # main at t=0 sees the deletion it caused
        assert (0, 1) not in adjacency[(0, "main")]
        # the fork branches from BEFORE that step, so the arc is still there
        assert (0, 1) in adjacency[(0, "cf_0")]
        # and the deletion persists into t=1
        assert (0, 1) not in adjacency[(1, "main")]


class TestCollate:
    def test_block_diagonal_batch_keeps_samples_disconnected(self, dataset_dir: Path) -> None:
        dataset = TransitionDataset(dataset_dir, "IC", "test")
        batch = collate_transitions(
            [dataset[0], dataset[1]], "IC", torch.device("cpu")
        )

        assert batch["X"].shape[0] == 12  # two 6-node graphs
        assert batch["batch_index"].tolist() == [0] * 6 + [1] * 6

        dense = batch["graph"].adj_norm.to_dense()
        # No message can cross from block 0 into block 1
        assert dense[:6, 6:].abs().max() == 0
        assert dense[6:, :6].abs().max() == 0

    def test_edges_to_arrays_roundtrip(self, base_edges: dict) -> None:
        edge_index, weights = edges_to_arrays(base_edges)

        assert edge_index.shape == (2, len(base_edges))
        assert weights.shape == (len(base_edges),)
        assert dict(zip(map(tuple, edge_index.T.tolist()), weights.tolist())) == {
            key: pytest.approx(value) for key, value in base_edges.items()
        }

    def test_empty_edge_dict(self) -> None:
        edge_index, weights = edges_to_arrays({})

        assert edge_index.shape == (2, 0)
        assert weights.shape == (0,)


class TestTransitionDataset:
    def test_loads_main_and_counterfactual_branches(self, dataset_dir: Path) -> None:
        dataset = TransitionDataset(dataset_dir, "IC", "test")

        assert len(dataset) == 5
        branches = {sample[0]["branch"] for sample in dataset.samples}
        assert branches == {"main", "cf_0", "cf_1"}

    def test_action_encoding_sets_feature_width(self, dataset_dir: Path) -> None:
        assert TransitionDataset(dataset_dir, "IC", "test")[0]["X"].shape[1] == 6
        assert (
            TransitionDataset(dataset_dir, "IC", "test", typed_encoding)[0]["X"].shape[1]
            == 9
        )
