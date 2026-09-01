"""
The LT head is a conditional hazard, and its oracle is exact in distribution.

Thresholds persist within an episode, so a node that failed to activate at
fraction f has theta > f: the head must emit 0 when f has not risen, and
per-step ensemble sampling of the oracle hazard must reproduce NDlib's LT in
distribution. The second test is the regression for the +37/step netscience
rollout bias the marginal-form head produced.
"""

import networkx as nx
import numpy as np
import pytest
import torch

from data.wm_simulator import ActionOp, Simulator
from world_model.wm_data import GraphInput, build_graph_input, ch_frontier, ch_infected
from world_model.wm_model import WorldModel

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _line_graph() -> GraphInput:
    # 0 -> 1 -> 2 -> 3 as a symmetric edge list, unit weights like LT uses
    edges = [(0, 1), (1, 2), (2, 3)]
    edge_index = np.array(edges + [(b, a) for a, b in edges]).T
    return build_graph_input(
        edge_index, np.ones(edge_index.shape[1]), 4, "LT", torch.device("cpu")
    )


def _lt_oracle() -> WorldModel:
    return WorldModel(
        "gcn",
        hidden_dim=4,
        n_layers=1,
        dropout=0.0,
        head_type="structured_oracle",
        diffusion_model="LT",
    ).eval()


def _probs(model: WorldModel, X: torch.Tensor, graph: GraphInput) -> torch.Tensor:
    with torch.inference_mode():
        return torch.sigmoid(model(X, graph))


class TestHazardForm:
    def test_survivor_with_unchanged_fraction_has_zero_hazard(self):
        graph = _line_graph()
        X = torch.zeros(4, 6)
        # Node 0 active since an EARLIER step (infected, not frontier): node 1
        # already survived theta > 1/2, so with f unchanged nothing can activate
        X[0, ch_infected] = 1.0

        probs = _probs(_lt_oracle(), X, graph)

        assert float(probs[1, 1]) == pytest.approx(0.0, abs=1e-5)

    def test_fresh_frontier_carries_the_full_marginal(self):
        graph = _line_graph()
        X = torch.zeros(4, 6)
        # Node 0 activated THIS step (infected and frontier): node 1 has survived
        # nothing, so the hazard is the full marginal f = 1/2
        X[0, ch_infected] = 1.0
        X[0, ch_frontier] = 1.0

        probs = _probs(_lt_oracle(), X, graph)

        assert float(probs[1, 1]) == pytest.approx(0.5, abs=1e-4)

    def test_rising_fraction_pays_only_the_increment(self):
        # Star: 1, 2, 3 all point at 0. With 1 active earlier and 2 newly
        # active, node 0 survived theta > 1/3 and faces f = 2/3:
        # hazard = (2/3 - 1/3) / (1 - 1/3) = 1/2
        edges = [(1, 0), (2, 0), (3, 0)]
        edge_index = np.array(edges + [(b, a) for a, b in edges]).T
        graph = build_graph_input(edge_index, np.ones(6), 4, "LT", torch.device("cpu"))
        X = torch.zeros(4, 6)
        X[1, ch_infected] = 1.0
        X[2, ch_infected] = 1.0
        X[2, ch_frontier] = 1.0

        probs = _probs(_lt_oracle(), X, graph)

        assert float(probs[0, 1]) == pytest.approx(0.5, abs=1e-4)


class TestOracleDistribution:
    def test_hazard_targets_match_simulated_lt_frequencies(self):
        # The generator's closed-form LT marginals against brute force: one
        # graph, one mid-cascade state, hazards vs 3000 fresh-threshold draws
        rng = np.random.default_rng(0)
        g = nx.barabasi_albert_graph(40, 3, seed=2)
        seeds = [0, 1]

        simulator = Simulator(g, ic_prob_map=None, seed=7)
        simulator.reset("LT")
        state = simulator.advance([ActionOp("add_node", v) for v in seeds])
        _, infected_marginal, _ = simulator.advance_marginal([], 1)

        active = set(state.infected)
        survived = active - set(state.frontier)
        counts = {v: 0 for v in g if v not in active}
        draws = 3000
        for _ in range(draws):
            theta = rng.random(40)
            for v in counts:
                neighbors = list(g.neighbors(v))
                f_now = sum(u in active for u in neighbors) / len(neighbors)
                f_prev = sum(u in survived for u in neighbors) / len(neighbors)
                # Condition on survival: theta > f_prev
                if theta[v] > f_prev and theta[v] <= f_now:
                    counts[v] += 1
                elif theta[v] <= f_prev:
                    # Rejected draw: resample by scaling into the survivor region
                    scaled = f_prev + theta[v] * (1.0 - f_prev)
                    if scaled <= f_now:
                        counts[v] += 1

        for v, count in counts.items():
            expected = infected_marginal.get(v, 0.0)
            assert count / draws == pytest.approx(expected, abs=0.035)


class TestCompetitiveHazard:
    def test_clt_oracle_rollout_matches_the_simulator_in_distribution(self):
        import networkx as nx

        from coding_agent.types import GraphInfo
        from data.wm_competitive import CompetitiveConfig, CompetitiveSimulator
        from coding_agent.envs.world_model_env import WorldModelEnvironment

        g = nx.barabasi_albert_graph(60, 3, seed=5)
        edges = [(u, v) for u, v in g.edges()] + [(v, u) for u, v in g.edges()]
        edge_index = np.array(edges).T
        graph = GraphInfo(
            num_nodes=60,
            edge_index=edge_index,
            ic_probs=np.ones(edge_index.shape[1], dtype=np.float32),
            directed=False,
        )
        rumour, blockers = (0, 1), (5, 6)
        horizon, draws = 8, 400

        finals = []
        for draw in range(draws):
            simulator = CompetitiveSimulator(
                g,
                dict.fromkeys(edges, 1.0),
                seed=500 + draw,
                config=CompetitiveConfig(),
            )
            simulator.reset("LT", negative_seeds=list(rumour))
            state = simulator.advance(
                [ActionOp("add_node", node) for node in blockers]
            )
            for _ in range(horizon - 1):
                state = simulator.advance([])
            finals.append(len(state.infected))
        reference = float(np.mean(finals))

        env = WorldModelEnvironment.oracle(
            graph,
            "LT",
            n_samples=400,
            competitive=True,
            negative_seeds=rumour,
        )
        trajectory = env.rollout(
            lambda state, t: (
                [ActionOp("add_node", node) for node in blockers] if t == 0 else []
            ),
            horizon,
            len(blockers),
        )

        assert trajectory.reward == pytest.approx(reference, abs=3.0)
