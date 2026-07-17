import numpy as np
import pytest

from coding_agent.executor import StrategyError, build_strategy
from coding_agent.types import GraphInfo

valid_script = """
class MyStrategy(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        seeds = algorithms.high_degree(graph, budget, "IC")
        plan = [[ActionOp("add_node", node) for node in seeds]]
        plan += [[] for _ in range(horizon)]
        return plan
    def act(self, state, graph, timestep):
        return [ActionOp("add_node", 0)] if timestep == 0 else []
"""

syntax_error_script = (
    "class Bad(Strategy):\n"
    "    def act(self, state, graph, timestep)\n"
    "        return []\n"
)
no_strategy_script = "x = 1 + 1\n"


def _graph() -> GraphInfo:
    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    return GraphInfo(2, edge_index, np.array([0.5, 0.5], np.float32), True)


def test_build_valid_strategy():
    strategy = build_strategy(valid_script)
    plan = strategy.plan_horizon(_graph(), budget=1, horizon=2)
    assert plan[0][0].op == "add_node"


def test_syntax_error_raises():
    with pytest.raises(StrategyError):
        build_strategy(syntax_error_script)


def test_missing_strategy_raises():
    with pytest.raises(StrategyError):
        build_strategy(no_strategy_script)
