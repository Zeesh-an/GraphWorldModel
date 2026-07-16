# coding_agent/tests/test_agent.py
from coding_agent.agent import CodingAgent, GatewayProvider, extract_code_block


def test_extract_code_block():
    raw = "Here you go:\n```python\nclass S(Strategy):\n    pass\n```\nDone."
    assert "class S(Strategy):" in extract_code_block(raw)


def test_extract_code_block_no_fence_returns_stripped():
    assert extract_code_block("class S: pass") == "class S: pass"


def test_extract_code_block_prefers_fence_with_class():
    raw = (
        "Plan:\n```python\nseeds = [1, 2]\n```\nThen:\n"
        "```python\nclass S(Strategy):\n    pass\n```"
    )
    assert extract_code_block(raw).startswith("class S(Strategy)")


def test_gateway_provider_routes_token_by_model(monkeypatch):
    monkeypatch.setenv("GATEWAY_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("CLAUDE_GATEWAY_TOKEN", "claude-token")
    monkeypatch.setenv("CHATGPT_GATEWAY_TOKEN", "chatgpt-token")
    assert GatewayProvider("claude-sonnet-5").client.api_key == "claude-token"
    assert GatewayProvider("gpt-5.6-sol").client.api_key == "chatgpt-token"


def test_agent_uses_injected_provider():
    class Fake:
        def complete(self, system: str, user: str) -> str:
            return "```python\nclass S(Strategy):\n    pass\n```"

    agent = CodingAgent(Fake())
    assert "class S(Strategy)" in agent.generate("s", "u")