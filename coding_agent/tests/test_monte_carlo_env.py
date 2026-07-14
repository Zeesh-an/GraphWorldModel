# coding_agent/tests/test_monte_carlo_env.py
import numpy as np

from coding_agent.envs.monte_carlo_env import MonteCarloEnvironment
from coding_agent.types import Action, GraphInfo, State


def _hub() -> GraphInfo:
    sources = [0, 0, 0, 1, 2, 3]
    destinations = [1, 2, 3, 0, 0, 0]
    edge_index = np.array([sources, destinations], dtype=np.int64)
    ic_probs = np.full(len(sources), 0.9, dtype=np.float32)
    return GraphInfo(
        num_nodes=4, edge_index=edge_index, ic_probs=ic_probs, directed=True
    )


def test_mc_rollout_seeds_hub_spreads():
    graph = _hub()
    environment = MonteCarloEnvironment(graph, "IC", mc_runs=20)

    def action_fn(state: State, timestep: int) -> list[Action]:
        return [Action("add_node", 0)] if timestep == 0 else []

    trajectory = environment.rollout(action_fn, horizon=5, budget=1)
    assert trajectory.reward >= 1.0
    assert len(trajectory.infected_counts) >= 1
    assert trajectory.infected_counts[-1] == trajectory.reward


def test_mc_rollout_no_action_no_spread():
    graph = _hub()
    environment = MonteCarloEnvironment(graph, "IC", mc_runs=10)

    def no_action(state: State, timestep: int) -> list[Action]:
        return []

    trajectory = environment.rollout(no_action, horizon=3, budget=0)
    assert trajectory.reward == 0.0