"""OuterLoopMethod contract + shared helpers."""

from typing import Protocol

from coding_agent.agent import CodingAgent
from coding_agent.types import GraphInfo, Strategy, TaskSpec, Trajectory


def summarize(tr: Trajectory) -> str:
    return (
        f"final_spread={tr.reward:.2f}, steps={len(tr.infected_counts)}, "
        f"counts={[round(c, 1) for c in tr.infected_counts]}"
    )


class OuterLoopMethod(Protocol):
    def optimize(
        self, agent: CodingAgent, env, task: TaskSpec, g: GraphInfo
    ) -> tuple[Strategy, Trajectory]: ...
