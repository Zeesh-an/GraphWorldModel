import numpy as np
import pytest
import torch

from baselines.gdm_support import (
    convert_legacy_gat_state,
    gdm_node_features,
    undirected_edge_index,
)


def _triangle_plus_tail() -> np.ndarray:
    # 0-1-2 triangle, 2-3 tail: degrees 2, 2, 3, 1; cores 2, 2, 2, 1; clustering 1, 1, 1/3, 0
    return np.asarray([[0, 1, 2, 2], [1, 2, 0, 3]], dtype=np.int64)


def test_features_follow_gdm_extractor_formulas():
    features = gdm_node_features(4, _triangle_plus_tail())

    degree = np.asarray([2, 2, 3, 1]) / 3.0
    mean = degree.mean()
    expected = np.column_stack(
        [(degree - mean) ** 2 / mean, [1.0, 1.0, 1 / 3, 0.0], degree, np.asarray([2, 2, 2, 1]) / 2.0]
    )

    assert features.shape == (4, 4)
    assert np.allclose(features, expected)


def test_features_ignore_direction_and_self_loops():
    both_ways = np.asarray([[0, 1, 1, 2, 2, 0, 2, 3, 3, 3], [1, 0, 2, 1, 0, 2, 3, 2, 3, 3]])

    assert np.allclose(gdm_node_features(4, both_ways), gdm_node_features(4, _triangle_plus_tail()))


@pytest.mark.parametrize("projection", ["lin.weight", "lin_src.weight"])
def test_legacy_conversion_transposes_and_splits(projection):
    weight = torch.arange(8.0).view(4, 2)  # in=4, heads*out=2
    att = torch.arange(4.0).view(1, 1, 4)  # heads=1, 2*out
    legacy = {
        "convolutional_layers.0.weight": weight,
        "convolutional_layers.0.att": att,
        "convolutional_layers.0.bias": torch.zeros(2),
        "linear_layers.0.weight": torch.ones(2, 4),
    }
    target = [
        f"convolutional_layers.0.{projection}",
        "convolutional_layers.0.att_src",
        "convolutional_layers.0.att_dst",
        "convolutional_layers.0.bias",
        "linear_layers.0.weight",
    ]

    converted = convert_legacy_gat_state(legacy, target)

    assert set(converted) == set(target)
    assert torch.equal(converted[f"convolutional_layers.0.{projection}"], weight.t())
    assert torch.equal(converted["convolutional_layers.0.att_dst"], att[..., :2])
    assert torch.equal(converted["convolutional_layers.0.att_src"], att[..., 2:])


def test_legacy_conversion_refuses_a_mismatched_model():
    legacy = {"convolutional_layers.0.weight": torch.zeros(4, 2)}

    with pytest.raises(KeyError, match="none of"):
        convert_legacy_gat_state(legacy, ["convolutional_layers.0.something_else"])


def test_undirected_edge_index_has_both_arcs_once():
    edge_index = undirected_edge_index(np.asarray([[0, 1, 1], [1, 0, 2]]))

    assert edge_index.shape == (2, 4)
    assert sorted(map(tuple, edge_index.t().tolist())) == [(0, 1), (1, 0), (1, 2), (2, 1)]
