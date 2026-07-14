"""OuterLoopMethod contract + shared helpers."""

from typing import Protocol

from coding_agent.agent import CodingAgent
from coding_agent.types import GraphInfo, Strategy, TaskSpec, Trajectory


def summarize(trajectory: Trajectory) -> str:
    return (
        f"final_spread={trajectory.reward:.2f}, "
        f"steps={len(trajectory.infected_counts)}, "
        f"counts={[round(count, 1) for count in trajectory.infected_counts]}"
    )


class OuterLoopMethod(Protocol):
    def optimize(
        self, agent: CodingAgent, environment: object, task: TaskSpec, graph: GraphInfo
    ) -> tuple[Strategy, Trajectory]: ...
