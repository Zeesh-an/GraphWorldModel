# coding_agent/tests/test_methods.py
import numpy as np
import pytest

from coding_agent.agent import CodingAgent
from coding_agent.envs.monte_carlo_env import MonteCarloEnvironment
from coding_agent.executor import StrategyError
from coding_agent.methods.one_shot import OneShotSuperAlgorithm
from coding_agent.methods.per_step import PerStepReprompt
from coding_agent.methods.windowed import WindowedOnline
from coding_agent.types import GraphInfo, TaskSpec


def _hub() -> GraphInfo:
    sources = [0, 0, 0, 1, 2, 3]
    destinations = [1, 2, 3, 0, 0, 0]
    edge_index = np.array([sources, destinations], dtype=np.int64)
    return GraphInfo(4, edge_index, np.full(6, 0.9, np.float32), True)


one_shot_script = """
class S(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        seeds = algorithms.high_degree(graph, budget, "IC")
        return [[Action("add_node", node) for node in seeds]] + [[] for _ in range(horizon)]
"""

per_step_script = """
class S(Strategy):
    def act(self, state, graph, timestep):
        if timestep == 0:
            return [Action("add_node", node) for node in algorithms.high_degree(graph, 1, "IC")]
        return []
"""


class CannedProvider:
    def __init__(self, script: str) -> None:
        self.script = script

    def complete(self, system: str, user: str) -> str:
        return f"```python\n{self.script}\n```"


def test_one_shot_runs_and_scores():
    graph = _hub()
    agent = CodingAgent(CannedProvider(one_shot_script))
    environment = MonteCarloEnvironment(graph, "IC", mc_runs=10)
    method = OneShotSuperAlgorithm(outer_iters=2)
    strategy, trajectory = method.optimize(
        agent, environment, TaskSpec(budget=1, horizon=4), graph
    )
    assert trajectory.reward >= 1.0


def test_per_step_runs():
    graph = _hub()
    agent = CodingAgent(CannedProvider(per_step_script))
    environment = MonteCarloEnvironment(graph, "IC", mc_runs=10)
    strategy, trajectory = PerStepReprompt().optimize(
        agent, environment, TaskSpec(budget=1, horizon=3), graph
    )
    assert trajectory.reward >= 1.0


def test_windowed_runs():
    graph = _hub()
    agent = CodingAgent(CannedProvider(per_step_script))
    environment = MonteCarloEnvironment(graph, "IC", mc_runs=10)
    strategy, trajectory = WindowedOnline(windows=2).optimize(
        agent, environment, TaskSpec(budget=1, horizon=4), graph
    )
    assert trajectory.reward >= 1.0


# Script that always emits 2 add_node actions at t0, violating budget=1.
over_budget_script = """
class S(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        return [[Action("add_node", 0), Action("add_node", 1)]] + [[] for _ in range(horizon)]
"""


# Script with a runtime bug: compute_degree returns an ndarray, not a dict.
buggy_runtime_script = """
class S(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        degrees = primitives.compute_degree(graph)
        best = degrees.get(0)
        return [[Action("add_node", best)]] + [[] for _ in range(horizon)]
"""


def test_runtime_error_in_plan_becomes_strategy_error():
    """A crash inside generated code must convert to StrategyError (repair feedback), not escape raw."""
    graph = _hub()
    agent = CodingAgent(CannedProvider(buggy_runtime_script))
    environment = MonteCarloEnvironment(graph, "IC", mc_runs=10)
    with pytest.raises(StrategyError, match="AttributeError"):
        OneShotSuperAlgorithm(outer_iters=1).optimize(
            agent, environment, TaskSpec(budget=1, horizon=3), graph
        )


def test_over_budget_plan_triggers_validation_and_raises():
    """An over-budget plan should trigger the repair path and ultimately raise StrategyError."""
    graph = _hub()
    agent = CodingAgent(CannedProvider(over_budget_script))
    environment = MonteCarloEnvironment(graph, "IC", mc_runs=10)
    with pytest.raises(StrategyError):
        OneShotSuperAlgorithm(outer_iters=2).optimize(
            agent, environment, TaskSpec(budget=1, horizon=4), graph
        )