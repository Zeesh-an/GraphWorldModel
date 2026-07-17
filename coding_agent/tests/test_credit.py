import numpy as np
import pytest

from coding_agent.credit import counterfactual_credit, format_credit_report
from coding_agent.envs.monte_carlo_env import MonteCarloEnvironment
from coding_agent.prompts import build_feedback_prompt
from coding_agent.run import ExperimentConfig, monte_carlo, run_experiment
from coding_agent.types import ActionOp, GraphInfo


def _hub() -> GraphInfo:
    sources = [0, 0, 0, 1, 2, 3]
    destinations = [1, 2, 3, 0, 0, 0]
    edge_index = np.array([sources, destinations], dtype=np.int64)
    return GraphInfo(4, edge_index, np.full(6, 0.9, np.float32), True)


def test_sole_seed_gets_full_credit():
    graph = _hub()
    environment = MonteCarloEnvironment(graph, "IC", mc_runs=10)
    plan = [[ActionOp("add_node", 0)], [], []]

    base_reward, entries = counterfactual_credit(
        environment, plan, horizon=3, budget=1
    )

    assert base_reward >= 1.0
    assert len(entries) == 1
    # Removing the only seed leaves an empty plan -> ablated reward 0 -> delta == base.
    assert entries[0]["delta"] == pytest.approx(base_reward)
    assert entries[0]["t"] == 0
    assert entries[0]["op"] == "add_node"
    assert entries[0]["target"] == 0


def test_empty_plan_yields_no_entries():
    graph = _hub()
    environment = MonteCarloEnvironment(graph, "IC", mc_runs=5)

    base_reward, entries = counterfactual_credit(
        environment, [[], []], horizon=2, budget=0
    )

    assert base_reward == 0.0
    assert entries == []
    assert "No actions" in format_credit_report(base_reward, entries)


def test_report_formatting_and_feedback_prompt():
    entries = [
        {"t": 0, "op": "add_node", "target": 3, "delta": 2.5},
        {"t": 2, "op": "set_edge_weight", "target": 1, "destination": 2, "delta": 0.0},
    ]
    report = format_credit_report(4.0, entries)

    assert "t=0: add_node(3)  delta=+2.50" in report
    assert "t=2: set_edge_weight(1->2)  delta=+0.00" in report

    feedback = build_feedback_prompt(4.0, "summary", credit_report=report)
    assert "add_node(3)" in feedback


canned = """
class S(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        seeds = algorithms.high_degree(graph, budget, "IC")
        return [[ActionOp("add_node", node) for node in seeds]] + [[] for _ in range(horizon)]
"""


def test_run_experiment_credit_flag():
    config = ExperimentConfig(
        method="one_shot",
        evaluator=monte_carlo,
        budget=1,
        horizon=3,
        mc_runs=5,
        outer_iters=2,
        credit=True,
    )

    result = run_experiment(config, graph=_hub(), canned_script=canned)

    assert result["credit_base_reward"] >= 1.0
    assert len(result["credit"]) == 1
    assert result["credit"][0]["op"] == "add_node"
