"""Method 3: design one online algorithm; re-apply it per time window."""

from coding_agent.agent import CodingAgent
from coding_agent.executor import build_strategy, call_strategy
from coding_agent.prompts import build_user_prompt, system_prompts
from coding_agent.types import Action, GraphInfo, State, Strategy, TaskSpec, Trajectory


class WindowedOnline:
    def __init__(self, windows: int = 3) -> None:
        self.windows = windows

    def optimize(
        self, agent: CodingAgent, environment: object, task: TaskSpec, graph: GraphInfo
    ) -> tuple[Strategy, Trajectory]:
        system = system_prompts["windowed"]
        user = build_user_prompt("windowed", task, graph)
        strategy = build_strategy(agent.generate(system, user))  # designed ONCE
        window_length = max(1, (task.horizon + 1) // self.windows)

        def action_fn(state: State, timestep: int) -> list[Action]:
            # Consult the online algorithm at each window boundary only.
            if timestep % window_length == 0:
                return call_strategy(
                    strategy.act, state, graph, timestep // window_length
                )
            return []

        trajectory = environment.rollout(action_fn, task.horizon, task.budget)

        return strategy, trajectory
