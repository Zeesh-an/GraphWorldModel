"""Provider-agnostic coding agent. The concrete LLM call is the ONLY TODO in the subsystem; wire a provider once the model choice is confirmed."""

import re
from typing import Protocol


class LLMProvider(Protocol):
    def complete(self, system: str, user: str) -> str:
        """Return the model's raw text completion for the given prompts."""
        ...


class TODOProvider:
    """Placeholder provider. Raises until a real model is wired in."""

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError(
            "No LLM provider configured. Implement an LLMProvider (e.g. AnthropicProvider) "
            "with .complete(system, user) -> str and pass it to CodingAgent(provider=...). "
            "See coding_agent/agent.py."
        )


# --- Concrete providers: TODO (pending model-provider choice) ----------------
# class AnthropicProvider:
#     def __init__(self, model: str, api_key: str | None = None) -> None: ...
#     def complete(self, system: str, user: str) -> str: ...  # TODO
#
# class OpenAIProvider:
#     def __init__(self, model: str, api_key: str | None = None) -> None: ...
#     def complete(self, system: str, user: str) -> str: ...  # TODO


_code_fence = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def extract_code_block(text: str) -> str:
    """Pull the first fenced Python block; fall back to the stripped text."""
    match = _code_fence.search(text)
    return match.group(1).strip() if match else text.strip()


class CodingAgent:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def generate(self, system: str, user: str) -> str:
        """Call the provider and return the extracted Python script."""
        return extract_code_block(self.provider.complete(system, user))
