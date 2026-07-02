"""Method 3: design one online algorithm; re-apply it per time window."""

from coding_agent.agent import CodingAgent
from coding_agent.executor import build_strategy
from coding_agent.prompts import build_user_prompt, system_prompts
from coding_agent.types import Action, GraphInfo, State, Strategy, TaskSpec, Trajectory


class WindowedOnline:
    def __init__(self, windows: int = 3) -> None:
        self.windows = windows

    def optimize(
        self, agent: CodingAgent, env, task: TaskSpec, g: GraphInfo
    ) -> tuple[Strategy, Trajectory]:
        system = system_prompts["windowed"]
        user = build_user_prompt("windowed", task, g)
        strat = build_strategy(agent.generate(system, user))  # designed ONCE
        window_len = max(1, (task.horizon + 1) // self.windows)

        def action_fn(state: State, t: int) -> list[Action]:
            # Consult the online algorithm at each window boundary only.
            if t % window_len == 0:
                return strat.act(state, g, t // window_len)
            return []

        tr = env.rollout(action_fn, task.horizon, task.budget)
        return strat, tr
