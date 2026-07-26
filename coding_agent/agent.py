"""
Coding Agent LLM call to the gateway provider.

GatewayProvider is the concrete LLM call; it needs GATEWAY_BASE_URL and the per-account tokens from the .env file.
"""

import os
import re
import time
from typing import Protocol

from openai import OpenAI, OpenAIError

gateway_retries = 3


class LLMProvider(Protocol):
    def complete(self, messages: list[dict]) -> str:
        """Return the model's raw text completion for a full message thread."""
        ...


def fold_system(messages: list[dict]) -> list[dict]:
    """
    Merge a leading system message into the first user turn.

    The gateway silently drops the system role, so the contract has to ride in
    as user text. Isolated here so that only this function changes if the
    gateway ever starts honouring it.
    """
    if not messages or messages[0]["role"] != "system":
        return list(messages)

    system, rest = messages[0]["content"], messages[1:]
    if not rest:
        return [{"role": "user", "content": system}]

    return [
        {"role": "user", "content": f"{system}\n\n{rest[0]['content']}"}
    ] + rest[1:]


class GatewayProvider:
    """OpenAI-compatible gateway over ChatGPT/Claude Pro subscriptions."""

    def __init__(self, model: str, temperature: float | None = None) -> None:
        # Each subscription account has its own bearer token, so the token is picked from the model-name family (claude-* vs gpt-*)
        token_env = (
            "CLAUDE_GATEWAY_TOKEN"
            if model.startswith("claude")
            else "CHATGPT_GATEWAY_TOKEN"
        )

        # max_retries=0: retries are owned here so each failure prints a warning
        self.client = OpenAI(
            base_url=os.environ["GATEWAY_BASE_URL"],
            api_key=os.environ[token_env],
            max_retries=0,
        )
        self.model = model
        # None -> provider default sampling; 0.0 -> greedy decoding
        self.temperature = temperature

    def complete(self, messages: list[dict]) -> str:
        messages = fold_system(messages)

        sampling_kwargs = (
            {} if self.temperature is None else {"temperature": self.temperature}
        )

        for attempt in range(1, gateway_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, **sampling_kwargs
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


code_fence = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def extract_code_block(text: str) -> str:
    """Pull the first fenced block defining a class (else the first fence, else the stripped text)."""
    blocks = code_fence.findall(text)

    # Models sometimes emit prose snippets in extra fences; the Strategy class is the block we want
    for block in blocks:
        if "class " in block:
            return block.strip()

    return blocks[0].strip() if blocks else text.strip()


class CodingAgent:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def generate(self, messages: list[dict]) -> str:
        """Call the provider on a message thread and return the extracted script."""
        start = time.perf_counter()
        reply = self.provider.complete(messages)
        script = extract_code_block(reply)

        print(
            f"[agent] response received in {time.perf_counter() - start:.1f}s "
            f"({len(reply)} chars -> script of {len(script.splitlines())} lines, "
            f"thread of {len(messages)} messages)"
        )

        return script


# Opening turn (task + graph profile + anchor) plus this many recent exchanges
# are sent; the middle is dropped so a long refinement loop cannot grow context
# without bound. 6 exchanges covers every default --outer-iters we run.
default_history_exchanges = 6


class Conversation:
    """
    A multi-turn refinement thread with one model.

    The point of keeping a thread rather than rebuilding one prompt per
    iteration: the agent's previous script is the previous ASSISTANT turn, so
    the model can see what it wrote and edit it, instead of regenerating a
    program from the task description and a scalar reward every round.

    Only the extracted script is stored back as the assistant turn, not the raw
    prose reply — it is the artifact the next turn edits, and it keeps the
    thread compact.
    """

    def __init__(
        self,
        agent: CodingAgent,
        system: str,
        history_exchanges: int = default_history_exchanges,
    ) -> None:
        self.agent = agent
        self.history_exchanges = history_exchanges
        self.messages = [{"role": "system", "content": system}]

    def send(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        script = self.agent.generate(self.window())
        self.messages.append(
            {"role": "assistant", "content": f"```python\n{script}\n```"}
        )

        return script

    def reset(self) -> None:
        """Drop everything but the system turn — a fresh episode, same contract."""
        del self.messages[1:]

    def window(self) -> list[dict]:
        """System turn + opening task turn + the most recent exchanges."""
        kept = 2 + 2 * self.history_exchanges
        if len(self.messages) <= kept:
            return list(self.messages)

        return self.messages[:2] + self.messages[-(2 * self.history_exchanges) :]
