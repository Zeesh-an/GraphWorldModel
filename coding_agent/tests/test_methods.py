# coding_agent/tests/test_methods.py
import pytest
import numpy as np
from coding_agent.types import GraphInfo, TaskSpec
from coding_agent.agent import CodingAgent
from coding_agent.envs.monte_carlo_env import MonteCarloEnvironment
from coding_agent.executor import StrategyError
from coding_agent.methods.one_shot import OneShotSuperAlgorithm
from coding_agent.methods.per_step import PerStepReprompt
from coding_agent.methods.windowed import WindowedOnline


def _hub() -> GraphInfo:
    src = [0, 0, 0, 1, 2, 3]
    dst = [1, 2, 3, 0, 0, 0]
    ei = np.array([src, dst], dtype=np.int64)
    return GraphInfo(4, ei, np.full(6, 0.9, np.float32), True)


one_shot_script = """
class S(Strategy):
    def plan_horizon(self, g, budget, horizon):
        seeds = algorithms.high_degree(g, budget, "IC")
        return [[Action("add_node", v) for v in seeds]] + [[] for _ in range(horizon)]
"""

per_step_script = """
class S(Strategy):
    def act(self, state, g, t):
        if t == 0:
            return [Action("add_node", v) for v in algorithms.high_degree(g, 1, "IC")]
        return []
"""


class CannedProvider:
    def __init__(self, script):
        self.script = script

    def complete(self, system, user):
        return f"```python\n{self.script}\n```"


def test_one_shot_runs_and_scores():
    g = _hub()
    agent = CodingAgent(CannedProvider(one_shot_script))
    env = MonteCarloEnvironment(g, "IC", mc_runs=10)
    method = OneShotSuperAlgorithm(outer_iters=2)
    strat, tr = method.optimize(agent, env, TaskSpec(budget=1, horizon=4), g)
    assert tr.reward >= 1.0


def test_per_step_runs():
    g = _hub()
    agent = CodingAgent(CannedProvider(per_step_script))
    env = MonteCarloEnvironment(g, "IC", mc_runs=10)
    strat, tr = PerStepReprompt().optimize(agent, env, TaskSpec(budget=1, horizon=3), g)
    assert tr.reward >= 1.0


def test_windowed_runs():
    g = _hub()
    agent = CodingAgent(CannedProvider(per_step_script))
    env = MonteCarloEnvironment(g, "IC", mc_runs=10)
    strat, tr = WindowedOnline(windows=2).optimize(
        agent, env, TaskSpec(budget=1, horizon=4), g
    )
    assert tr.reward >= 1.0


# Script that always emits 2 add_node actions at t0, violating budget=1.
over_budget_script = """
class S(Strategy):
    def plan_horizon(self, g, budget, horizon):
        return [[Action("add_node", 0), Action("add_node", 1)]] + [[] for _ in range(horizon)]
"""


def test_over_budget_plan_triggers_validation_and_raises():
    """An over-budget plan should trigger the repair path and ultimately raise StrategyError."""
    g = _hub()
    agent = CodingAgent(CannedProvider(over_budget_script))
    env = MonteCarloEnvironment(g, "IC", mc_runs=10)
    with pytest.raises(StrategyError):
        OneShotSuperAlgorithm(outer_iters=2).optimize(
            agent, env, TaskSpec(budget=1, horizon=4), g
        )
