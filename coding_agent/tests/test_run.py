import numpy as np

from coding_agent.run import ExperimentConfig, build_method, monte_carlo, run_experiment
from coding_agent.types import GraphInfo


def _hub() -> GraphInfo:
    sources = [0, 0, 0, 1, 2, 3]
    destinations = [1, 2, 3, 0, 0, 0]
    edge_index = np.array([sources, destinations], dtype=np.int64)
    return GraphInfo(4, edge_index, np.full(6, 0.9, np.float32), True)


canned = """
class S(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        seeds = algorithms.high_degree(graph, budget, "IC")
        return [[Action("add_node", node) for node in seeds]] + [[] for _ in range(horizon)]
    def act(self, state, graph, timestep):
        if timestep == 0:
            return [Action("add_node", node) for node in algorithms.high_degree(graph, 1, "IC")]
        return []
"""


def test_build_method_switch():
    for name in ("one_shot", "per_step", "windowed"):
        assert build_method(name) is not None


def test_run_experiment_mc_with_canned_script():
    config = ExperimentConfig(
        method="one_shot",
        evaluator=monte_carlo,
        budget=1,
        horizon=4,
        mc_runs=10,
        outer_iters=1,
    )
    result = run_experiment(config, graph=_hub(), canned_script=canned)
    assert result["reward"] >= 1.0
    assert result["method"] == "one_shot"
    assert result["model"] == "canned"
    assert "class S(Strategy):" in result["script"]
    assert result["cost"]["rollout_seconds"] > 0
    assert result["elapsed_seconds"] >= result["cost"]["rollout_seconds"]

    timeline = result["timeline"]
    assert timeline[0]["t"] == 0
    assert timeline[0]["actions"][0]["op"] == "add_node"
    assert timeline[0]["infected_count"] >= 1
    assert all(
        {"t", "actions", "infected", "frontier"} <= entry.keys() for entry in timeline
    )
