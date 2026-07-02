import numpy as np
from coding_agent.types import GraphInfo
from coding_agent.config import ExperimentConfig
from coding_agent.run import run_experiment, build_method, MONTE_CARLO


def _hub() -> GraphInfo:
    src = [0, 0, 0, 1, 2, 3]
    dst = [1, 2, 3, 0, 0, 0]
    ei = np.array([src, dst], dtype=np.int64)
    return GraphInfo(4, ei, np.full(6, 0.9, np.float32), True)


CANNED = '''
class S(Strategy):
    def plan_horizon(self, g, budget, horizon):
        return [[Action("add_node", v) for v in algorithms.high_degree(g, budget, "IC")]] + [[] for _ in range(horizon)]
    def act(self, state, g, t):
        if t == 0:
            return [Action("add_node", v) for v in algorithms.high_degree(g, 1, "IC")]
        return []
'''


def test_build_method_switch():
    for name in ("one_shot", "per_step", "windowed"):
        assert build_method(name) is not None


def test_run_experiment_mc_with_canned_script():
    cfg = ExperimentConfig(
        method="one_shot", evaluator=MONTE_CARLO, budget=1, horizon=4,
        mc_runs=10, outer_iters=1,
    )
    result = run_experiment(cfg, graph=_hub(), canned_script=CANNED)
    assert result["reward"] >= 1.0
    assert result["method"] == "one_shot"
