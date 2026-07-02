# coding_agent/tests/test_monte_carlo_env.py
import numpy as np
from coding_agent.types import GraphInfo, Action
from coding_agent.envs.monte_carlo_env import MonteCarloEnvironment


def _hub() -> GraphInfo:
    src = [0, 0, 0, 1, 2, 3]
    dst = [1, 2, 3, 0, 0, 0]
    ei = np.array([src, dst], dtype=np.int64)
    ic = np.full(len(src), 0.9, dtype=np.float32)
    return GraphInfo(num_nodes=4, edge_index=ei, ic_probs=ic, directed=True)


def test_mc_rollout_seeds_hub_spreads():
    g = _hub()
    env = MonteCarloEnvironment(g, "IC", mc_runs=20)

    def action_fn(state, t):
        return [Action("add_node", 0)] if t == 0 else []

    tr = env.rollout(action_fn, horizon=5, budget=1)
    assert tr.reward >= 1.0
    assert len(tr.infected_counts) >= 1
    assert tr.infected_counts[-1] == tr.reward


def test_mc_rollout_no_action_no_spread():
    g = _hub()
    env = MonteCarloEnvironment(g, "IC", mc_runs=10)
    tr = env.rollout(lambda s, t: [], horizon=3, budget=0)
    assert tr.reward == 0.0
