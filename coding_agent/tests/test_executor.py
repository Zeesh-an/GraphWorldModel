import numpy as np
import pytest
from coding_agent.executor import build_strategy, StrategyError
from coding_agent.types import GraphInfo


valid_script = """
class MyStrategy(Strategy):
    def plan_horizon(self, g, budget, horizon):
        seeds = algorithms.high_degree(g, budget, "IC")
        plan = [[Action("add_node", v) for v in seeds]]
        plan += [[] for _ in range(horizon)]
        return plan
    def act(self, state, g, t):
        return [Action("add_node", 0)] if t == 0 else []
"""

syntax_err = "class Bad(Strategy):\n    def act(self, state, g, t)\n        return []\n"
no_strategy = "x = 1 + 1\n"


def _g():
    ei = np.array([[0, 1], [1, 0]], dtype=np.int64)
    return GraphInfo(2, ei, np.array([0.5, 0.5], np.float32), True)


def test_build_valid_strategy():
    strat = build_strategy(valid_script)
    plan = strat.plan_horizon(_g(), budget=1, horizon=2)
    assert plan[0][0].op == "add_node"


def test_syntax_error_raises():
    with pytest.raises(StrategyError):
        build_strategy(syntax_err)


def test_missing_strategy_raises():
    with pytest.raises(StrategyError):
        build_strategy(no_strategy)
