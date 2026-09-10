"""
The claim Experiment 2 rests on, as a unit test rather than only as a script.

    A: v became active through natural diffusion, action = []
    B: v was seeded,  state = (I - v, F - v), action = [add_node(v)]

Both are constructed so that `T_exo` maps them to the SAME state. If the
simulator's post-action status dicts are then identical, the two cases share one
transition kernel and provenance carries no one-step information at all — which
makes any model divergence between them spurious rather than informative.

That is a property of the dynamics, so it belongs in the suite: if a future
change to `T_exo` or to `apply_actions` breaks it, the experiment's whole
framing changes and this is where that shows up.
"""

import networkx as nx
import numpy as np
import pytest
import torch

from data.wm_simulator import Simulator
from scripts.eval_provenance import matched_pair, true_kernel_divergence
from world_model.wm_data import (
    build_features,
    build_graph_input,
    ch_add,
    ch_frontier,
    ch_infected,
)
from world_model.wm_model import WorldModel


@pytest.fixture
def graph_and_simulator():
    nx_graph = nx.barabasi_albert_graph(40, 3, seed=5)
    probability_map = {}
    for source, destination in nx_graph.edges():
        probability_map[(source, destination)] = 0.2
        probability_map[(destination, source)] = 0.2

    return nx_graph, Simulator(nx_graph, probability_map, seed=0)


class TestTheKernelIsShared:
    @pytest.mark.parametrize("dynamics", ["IC", "LT"])
    def test_the_two_cases_reach_an_identical_post_action_status(
        self, graph_and_simulator, dynamics
    ):
        nx_graph, simulator = graph_and_simulator
        simulator.reset(dynamics)
        infected, frontier, target = {1, 2, 3, 7}, {3, 7}, 7

        identical, divergence, floor = true_kernel_divergence(
            simulator, infected, frontier, target, nx_graph.number_of_nodes(), 40
        )

        assert identical
        # ...and the empirical check agrees, against its own sampling floor
        assert divergence <= max(3.0 * floor, 0.05)

    def test_the_feature_matrices_however_are_NOT_identical(self):
        """
        Which is the whole point: `T_exo` erases the difference, but the ENCODER
        still sees it, because provenance lives in different columns of X. A
        model can therefore distinguish two cases the world does not.
        """
        natural, seeded = matched_pair({1, 2, 3, 7}, {3, 7}, 7)
        edge_index = np.array([[1, 2, 3], [2, 3, 7]], dtype=np.int64)

        first, _, _ = build_features(natural, edge_index, 40, "basic")
        second, _, _ = build_features(seeded, edge_index, 40, "basic")

        assert not np.array_equal(first, second)
        assert first[7, ch_infected] == 1.0 and first[7, ch_add] == 0.0
        assert second[7, ch_infected] == 0.0 and second[7, ch_add] == 1.0
        # T_exo reconciles them: infected + add is the same on both sides
        assert (first[:, ch_infected] + first[:, ch_add]).tolist() == (
            second[:, ch_infected] + second[:, ch_add]
        ).tolist()
        assert (first[:, ch_frontier] + first[:, ch_add]).tolist() == (
            second[:, ch_frontier] + second[:, ch_add]
        ).tolist()

    def test_the_oracle_head_is_provenance_invariant_by_construction(self):
        """
        The exact-`q` head reads no embedding, so it CANNOT invent a distinction.
        It is the zero this measurement is scored against.
        """
        natural, seeded = matched_pair({0, 1}, {1}, 1)
        edge_index = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
        weights = np.full(3, 0.3, dtype=np.float32)
        graph = build_graph_input(edge_index, weights, 4, "IC", torch.device("cpu"))
        model = WorldModel(
            "sage", hidden_dim=8, n_layers=2, head_type="structured_oracle"
        ).eval()

        with torch.inference_mode():
            outputs = [
                torch.sigmoid(
                    model(
                        torch.from_numpy(
                            build_features(record, edge_index, 4, "basic")[0]
                        ),
                        graph,
                    )
                )
                for record in (natural, seeded)
            ]

        torch.testing.assert_close(outputs[0], outputs[1])
