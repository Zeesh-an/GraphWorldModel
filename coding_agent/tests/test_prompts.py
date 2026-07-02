# coding_agent/tests/test_prompts.py
import numpy as np
from coding_agent.types import GraphInfo, TaskSpec, State
from coding_agent import prompts


def _g():
    ei = np.array([[0, 1], [1, 0]], dtype=np.int64)
    return GraphInfo(2, ei, np.array([0.5, 0.5], np.float32), True)


def test_system_prompts_exist_per_method():
    for m in ("one_shot", "per_step", "windowed"):
        assert m in prompts.SYSTEM_PROMPTS
        assert "Strategy" in prompts.SYSTEM_PROMPTS[m]


def test_user_prompt_includes_graph_and_api():
    task = TaskSpec(budget=3, horizon=5)
    u = prompts.build_user_prompt("one_shot", task, _g())
    assert "num_nodes" in u and "pagerank_seeds" in u and "budget" in u


def test_feedback_prompt_includes_reward():
    fb = prompts.build_feedback_prompt(reward=12.5, summary="cascade saturated at t=4")
    assert "12.5" in fb and "saturated" in fb
