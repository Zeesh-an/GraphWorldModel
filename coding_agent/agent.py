"""Provider-agnostic coding agent. GatewayProvider is the concrete LLM call; it needs GATEWAY_BASE_URL and the per-account tokens from .env (load_dotenv first)."""

import os
import re
from typing import Protocol

from openai import OpenAI, OpenAIError


class LLMProvider(Protocol):
    def complete(self, system: str, user: str) -> str:
        """Return the model's raw text completion for the given prompts."""
        ...


gateway_retries = 3


class GatewayProvider:
    """OpenAI-compatible gateway over the lab's ChatGPT/Claude Pro subscriptions."""

    def __init__(self, model: str) -> None:
        # Each subscription account has its own bearer token, so the token is
        # picked from the model-name family (claude-* vs gpt-*).
        token_env = (
            "CLAUDE_GATEWAY_TOKEN"
            if model.startswith("claude")
            else "CHATGPT_GATEWAY_TOKEN"
        )
        # max_retries=0: retries are owned here so each failure prints a warning.
        self.client = OpenAI(
            base_url=os.environ["GATEWAY_BASE_URL"],
            api_key=os.environ[token_env],
            max_retries=0,
        )
        self.model = model

    def complete(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        for attempt in range(1, gateway_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError(
                        f"gateway returned empty content for model {self.model!r}: "
                        f"{response}"
                    )
                return content
            except (OpenAIError, ValueError) as error:
                if attempt == gateway_retries:
                    raise
                print(
                    f"warning: gateway call failed (attempt {attempt}/"
                    f"{gateway_retries}, model {self.model!r}): {error}"
                )

        raise RuntimeError("unreachable: retry loop exits via return or raise")


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
