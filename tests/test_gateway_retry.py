import httpx
import pytest
from openai import APIConnectionError, APIStatusError

from coding_agent import agent as agent_module
from coding_agent.agent import GatewayProvider, retryable


def _status_error(code: int) -> APIStatusError:
    response = httpx.Response(code, request=httpx.Request("POST", "https://gateway/v1/chat/completions"))
    return APIStatusError(f"Error code: {code}", response=response, body=None)


class _Reply:
    def __init__(self, content: str) -> None:
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]
        self.usage = type("Usage", (), {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4})()


class _Completions:
    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def create(self, **kwargs) -> _Reply:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome

        return _Reply(outcome)


def _provider(monkeypatch, outcomes: list) -> tuple[GatewayProvider, _Completions, list]:
    monkeypatch.setenv("GATEWAY_BASE_URL", "https://gateway/v1")
    monkeypatch.setenv("CHATGPT_GATEWAY_TOKEN", "x")
    waits = []
    monkeypatch.setattr(agent_module.time, "sleep", waits.append)
    provider = GatewayProvider("gpt-test")
    completions = _Completions(outcomes)
    provider.client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()

    return provider, completions, waits


def test_transient_errors_are_waited_out_with_backoff(monkeypatch) -> None:
    provider, completions, waits = _provider(
        monkeypatch,
        [_status_error(502), _status_error(429), APIConnectionError(request=httpx.Request("POST", "https://gateway")), "OK"],
    )
    assert provider.complete([{"role": "user", "content": "hi"}]) == "OK"
    assert completions.calls == 4
    assert waits == [10.0, 20.0, 40.0]
    assert provider.usage["calls"] == 1


def test_a_definite_refusal_raises_at_once(monkeypatch) -> None:
    provider, completions, waits = _provider(monkeypatch, [_status_error(401), "never"])
    with pytest.raises(APIStatusError):
        provider.complete([{"role": "user", "content": "hi"}])
    assert completions.calls == 1 and waits == []


def test_retryable_classification() -> None:
    assert retryable(_status_error(503)) and retryable(_status_error(429))
    assert not retryable(_status_error(400)) and not retryable(_status_error(404))
    assert retryable(ValueError("empty content"))


def test_waits_are_capped(monkeypatch) -> None:
    outcomes = [_status_error(502)] * (agent_module.gateway_retries - 1) + ["OK"]
    provider, completions, waits = _provider(monkeypatch, outcomes)
    assert provider.complete([{"role": "user", "content": "hi"}]) == "OK"
    assert max(waits) == agent_module.retry_cap_seconds
    assert sum(waits) < 3600
