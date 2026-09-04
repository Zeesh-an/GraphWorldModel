import pytest

from coding_agent.agent import default_model, verify_gateway_model


def test_default_model_is_gpt_6_astra():
    assert default_model == "gpt-6-astra"


def test_a_served_model_passes():
    verify_gateway_model("gpt-6-astra", available=["gpt-5.6-sol", "gpt-6-astra"])


def test_an_unserved_model_fails_with_the_served_list():
    with pytest.raises(RuntimeError, match="does not serve model 'gpt-6-astra'.*gpt-5.6-sol"):
        verify_gateway_model("gpt-6-astra", available=["gpt-5.6-sol", "gpt-5.6-terra"])


def test_request_kwargs_send_effort_to_the_openai_family_only():
    from coding_agent.agent import request_kwargs

    assert request_kwargs("gpt-6-astra", None, "high") == {"reasoning_effort": "high"}
    assert request_kwargs("gpt-5.6-sol", 0.7, "high") == {"temperature": 0.7, "reasoning_effort": "high"}
    assert request_kwargs("claude-opus-4-7", 0.7, "high") == {"temperature": 0.7}
    # GPT-6 Astra accepts no temperature: it is dropped, not sent and rejected
    assert request_kwargs("gpt-6-astra", 0.7, None) == {}
