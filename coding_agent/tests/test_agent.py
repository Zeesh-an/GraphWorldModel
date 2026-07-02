# coding_agent/tests/test_agent.py
import pytest
from coding_agent.agent import CodingAgent, TODOProvider, extract_code_block


def test_extract_code_block():
    raw = "Here you go:\n```python\nclass S(Strategy):\n    pass\n```\nDone."
    assert "class S(Strategy):" in extract_code_block(raw)


def test_extract_code_block_no_fence_returns_stripped():
    assert extract_code_block("class S: pass") == "class S: pass"


def test_todo_provider_raises():
    agent = CodingAgent(TODOProvider())
    with pytest.raises(NotImplementedError):
        agent.generate("sys", "user")


def test_agent_uses_injected_provider():
    class Fake:
        def complete(self, system, user):
            return "```python\nclass S(Strategy):\n    pass\n```"

    agent = CodingAgent(Fake())
    assert "class S(Strategy)" in agent.generate("s", "u")
