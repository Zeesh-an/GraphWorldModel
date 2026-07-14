import numpy as np

from coding_agent import prompts
from coding_agent.types import GraphInfo, TaskSpec


def _graph() -> GraphInfo:
    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    return GraphInfo(2, edge_index, np.array([0.5, 0.5], np.float32), True)


def test_system_prompts_exist_per_method():
    for method_name in ("one_shot", "per_step", "windowed"):
        assert method_name in prompts.system_prompts
        assert "Strategy" in prompts.system_prompts[method_name]


def test_user_prompt_includes_graph_and_api():
    task = TaskSpec(budget=3, horizon=5)
    user_prompt = prompts.build_user_prompt("one_shot", task, _graph())
    assert "num_nodes" in user_prompt and "pagerank_seeds" in user_prompt
    assert "budget" in user_prompt


def test_feedback_prompt_includes_reward():
    feedback = prompts.build_feedback_prompt(
        reward=12.5, summary="cascade saturated at t=4"
    )
    assert "12.5" in feedback and "saturated" in feedback
