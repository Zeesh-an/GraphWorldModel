"""
The structured heads — the project's central claim.

The IC head does not predict the next state, it predicts the MECHANISM and derives
the state from the true IC form. Everything the README credits to that choice
(no saturation, exact T_exo, correct remove semantics) is a property of the
closed form and is therefore checkable exactly, with untrained weights, on a graph
small enough to compute by hand. These tests do that: they never train, so they
cannot pass for the wrong reason.
"""

import numpy as np
import pytest
import torch

from data.wm_simulator import blocked, spent
from world_model.wm_data import (
    build_graph_input,
    ch_add,
    ch_frontier,
    ch_infected,
    ch_remove,
    in_channels,
    typed_encoding,
    num_input_channels,
)
from world_model.wm_model import WorldModel, backbones

torch.manual_seed(0)


def _graph(edge_index, num_nodes=6, weight=0.5, diffusion_model="IC"):
    return build_graph_input(
        edge_index,
        np.full(edge_index.shape[1], weight, dtype=np.float32),
        num_nodes,
        diffusion_model,
        torch.device("cpu"),
    )


def _features(infected=(), frontier=(), add=(), remove=(), num_nodes=6, channels=None):
    X = torch.zeros(num_nodes, channels or in_channels)
    for node in infected:
        X[node, ch_infected] = 1.0
    for node in frontier:
        X[node, ch_frontier] = 1.0
    for node in add:
        X[node, ch_add] = 1.0
    for node in remove:
        X[node, ch_remove] = 1.0
    return X


def _probs(model, X, graph):
    return torch.sigmoid(model(X, graph)).detach().numpy()


class TestICStructuredHead:
    @pytest.fixture
    def model(self):
        return WorldModel(
            "sage", hidden_dim=8, n_layers=2, head_type="structured", dropout=0.0
        ).eval()

    def test_no_active_frontier_means_no_new_infection(self, model, edge_index):
        """The self-termination property. This is what stops a rollout saturating."""
        graph = _graph(edge_index)
        X = _features(infected=(1,), frontier=())  # infected but nobody spreading

        probs = _probs(model, X, graph)

        # Node 1 stays infected; nobody else can be infected from an empty frontier
        assert probs[1, 0] == pytest.approx(1.0, abs=1e-5)
        susceptible = [node for node in range(6) if node != 1]
        assert np.max(probs[susceptible, 0]) == pytest.approx(0.0, abs=1e-5)
        assert np.max(probs[:, 1]) == pytest.approx(0.0, abs=1e-5)

    def test_infection_is_local_to_the_frontier(self, model, edge_index):
        """p_new > 0 only for in-neighbours of an active node — the locality bound."""
        graph = _graph(edge_index)
        X = _features(infected=(0,), frontier=(0,))

        probs = _probs(model, X, graph)

        # 0's only neighbour is 1
        assert probs[1, 1] > 0
        for far in (2, 3, 4, 5):
            assert probs[far, 1] == pytest.approx(0.0, abs=1e-5)

    def test_seeding_a_node_forces_it_infected(self, model, edge_index):
        """T_exo is exact, not learned: add_node -> P(infected) = 1."""
        graph = _graph(edge_index)
        probs = _probs(model, _features(add=(3,)), graph)

        assert probs[3, 0] == pytest.approx(1.0, abs=1e-5)

    def test_monotone_in_infection(self, model, edge_index):
        """IC never de-infects: an already-infected node stays at P = 1."""
        graph = _graph(edge_index)
        probs = _probs(model, _features(infected=(2, 3), frontier=(3,)), graph)

        assert probs[2, 0] == pytest.approx(1.0, abs=1e-5)
        assert probs[3, 0] == pytest.approx(1.0, abs=1e-5)
        # An already-infected node is not in the NEW frontier
        assert probs[2, 1] == pytest.approx(0.0, abs=1e-5)

    def test_spent_removal_keeps_the_node_counted(self, edge_index):
        """IM semantics: a spent spreader stops spreading but stays in the count."""
        model = WorldModel(
            "sage",
            hidden_dim=8,
            n_layers=2,
            head_type="structured",
            dropout=0.0,
            remove_semantics=spent,
        ).eval()
        graph = _graph(edge_index)

        probs = _probs(model, _features(infected=(1,), frontier=(1,), remove=(1,)), graph)

        assert probs[1, 0] == pytest.approx(1.0, abs=1e-5)  # still counted
        # …and it no longer transmits, so its neighbours stay clean
        for neighbour in (0, 2, 4):
            assert probs[neighbour, 1] == pytest.approx(0.0, abs=1e-5)

    def test_blocked_removal_drops_the_node_from_the_count(self, edge_index):
        """Containment semantics: a blocked node leaves the graph entirely.

        This is the asymmetry the `spent`/`blocked` split exists for. Getting it
        backwards biases every containment number by exactly +k.
        """
        model = WorldModel(
            "sage",
            hidden_dim=8,
            n_layers=2,
            head_type="structured",
            dropout=0.0,
            remove_semantics=blocked,
        ).eval()
        graph = _graph(edge_index)

        probs = _probs(model, _features(infected=(1,), frontier=(1,), remove=(1,)), graph)

        assert probs[1, 0] == pytest.approx(0.0, abs=1e-5)

    def test_more_active_in_neighbours_never_lowers_infection_risk(
        self, model, edge_index
    ):
        """Monotonicity in the frontier: 1 - prod(1 - q) is increasing in the set."""
        graph = _graph(edge_index)

        one = _probs(model, _features(infected=(3,), frontier=(3,)), graph)[4, 1]
        two = _probs(model, _features(infected=(3, 5), frontier=(3, 5)), graph)[4, 1]

        assert two >= one - 1e-6

    def test_disconnected_graph_produces_no_spread(self, model):
        graph = _graph(np.zeros((2, 0), dtype=np.int64))
        probs = _probs(model, _features(infected=(0,), frontier=(0,)), graph)

        assert np.max(probs[:, 1]) == pytest.approx(0.0, abs=1e-5)


class TestOracleHead:
    def test_oracle_reproduces_the_ic_form_exactly(self, edge_index):
        """q = w with no MLP, so p_new is computable in closed form by hand."""
        model = WorldModel(
            "gcn",
            hidden_dim=4,
            n_layers=1,
            head_type="structured_oracle",
            dropout=0.0,
        ).eval()
        weight = 0.5
        graph = _graph(edge_index, weight=weight)

        # Seed node 1, whose neighbours are 0, 2 and 4 — each with exactly one
        # active in-neighbour, so p_new = 1 - (1 - 0.5) = 0.5
        probs = _probs(model, _features(infected=(1,), frontier=(1,)), graph)

        for neighbour in (0, 2, 4):
            assert probs[neighbour, 1] == pytest.approx(weight, abs=1e-5)

        # Node 3 has two in-neighbours (2 and 4) but neither is active
        assert probs[3, 1] == pytest.approx(0.0, abs=1e-5)

    def test_two_active_in_neighbours_compose_independently(self, edge_index):
        model = WorldModel(
            "gcn",
            hidden_dim=4,
            n_layers=1,
            head_type="structured_oracle",
            dropout=0.0,
        ).eval()
        graph = _graph(edge_index, weight=0.5)

        # Node 3's in-neighbours are 2 and 4; both active => 1 - 0.5 * 0.5 = 0.75
        probs = _probs(model, _features(infected=(2, 4), frontier=(2, 4)), graph)

        assert probs[3, 1] == pytest.approx(0.75, abs=1e-5)


class TestLTStructuredHead:
    @pytest.fixture
    def model(self):
        return WorldModel(
            "sage",
            hidden_dim=8,
            n_layers=2,
            head_type="structured",
            diffusion_model="LT",
            dropout=0.0,
        ).eval()

    def test_zero_active_fraction_gate_holds(self, model, edge_index):
        graph = _graph(edge_index, diffusion_model="LT")
        probs = _probs(model, _features(infected=(), frontier=()), graph)

        assert np.max(probs[:, 1]) == pytest.approx(0.0, abs=1e-5)

    def test_lt_removal_returns_the_node_to_susceptible(self, model, edge_index):
        """Unlike spent IC, LT has no Removed state: the node resets to 0."""
        graph = _graph(edge_index, diffusion_model="LT")
        probs = _probs(
            model, _features(infected=(1,), frontier=(1,), remove=(1,)), graph
        )

        assert probs[1, 0] < 1.0

    def test_lt_rejects_the_residual_head(self):
        with pytest.raises(ValueError, match="IC-only"):
            WorldModel(
                "gcn", head_type="structured_residual", diffusion_model="LT"
            )


class TestResidualHead:
    def test_zero_correction_equals_the_oracle(self, edge_index):
        """The anchor's defining property: q = sigmoid(logit(w) + 0) = w.

        Zeroing the MLP's last layer makes the correction exactly 0, so the
        residual head must reproduce the oracle head bit for bit.
        """
        model = WorldModel(
            "gcn",
            hidden_dim=8,
            n_layers=1,
            head_type="structured_residual",
            dropout=0.0,
        ).eval()
        final = model.head.edge_mlp[-1]
        torch.nn.init.zeros_(final.weight)
        torch.nn.init.zeros_(final.bias)

        oracle = WorldModel(
            "gcn", hidden_dim=8, n_layers=1, head_type="structured_oracle", dropout=0.0
        ).eval()

        graph = _graph(edge_index, weight=0.3)
        X = _features(infected=(1,), frontier=(1,))

        assert np.allclose(_probs(model, X, graph), _probs(oracle, X, graph), atol=1e-5)


class TestLinearHead:
    def test_linear_head_has_no_structural_bound(self, edge_index):
        """The contrast case. The linear head CAN infect a node with no active
        in-neighbour — which is exactly why its free-running rollout saturates."""
        model = WorldModel(
            "gcn", hidden_dim=8, n_layers=1, head_type="linear", dropout=0.0
        ).eval()
        # Push the head hard positive so the absence of a bound is visible
        torch.nn.init.constant_(model.head.bias, 5.0)

        graph = _graph(edge_index)
        probs = _probs(model, _features(infected=(), frontier=()), graph)

        # No frontier at all, yet everything is predicted infected
        assert np.min(probs[:, 0]) > 0.9


class TestBackbones:
    @pytest.mark.parametrize("backbone", sorted(backbones))
    def test_every_backbone_runs_and_shapes_match(self, backbone, edge_index):
        model = WorldModel(
            backbone, hidden_dim=8, n_layers=2, head_type="structured", dropout=0.0
        ).eval()
        graph = _graph(edge_index)

        logits = model(_features(infected=(1,), frontier=(1,)), graph)

        assert logits.shape == (6, 2)
        assert torch.isfinite(logits).all()

    def test_typed_encoding_widens_the_input_projection(self, edge_index):
        channels = num_input_channels(typed_encoding)
        model = WorldModel(
            "sage",
            in_channels=channels,
            hidden_dim=8,
            n_layers=2,
            head_type="structured",
            dropout=0.0,
        ).eval()

        logits = model(
            _features(infected=(1,), frontier=(1,), channels=channels), _graph(edge_index)
        )

        assert logits.shape == (6, 2)

    def test_unknown_backbone_and_head_are_rejected(self):
        with pytest.raises(ValueError, match="unknown backbone"):
            WorldModel("gnn")

        with pytest.raises(ValueError, match="unknown head_type"):
            WorldModel("gcn", head_type="mlp")
