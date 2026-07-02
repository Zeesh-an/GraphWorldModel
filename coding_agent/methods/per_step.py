"""Method 2: re-prompt the agent at every timestep on the current state."""

from coding_agent.agent import CodingAgent
from coding_agent.executor import StrategyError, build_strategy
from coding_agent.prompts import SYSTEM_PROMPTS, build_user_prompt
from coding_agent.types import Action, GraphInfo, State, Strategy, TaskSpec, Trajectory


class PerStepReprompt:
    def optimize(
        self, agent: CodingAgent, env, task: TaskSpec, g: GraphInfo
    ) -> tuple[Strategy, Trajectory]:
        system = SYSTEM_PROMPTS["per_step"]
        last_strat: Strategy | None = None

        def action_fn(state: State, t: int) -> list[Action]:
            nonlocal last_strat
            user = build_user_prompt("per_step", task, g) + (
                f"\n\nCURRENT STATE (t={t}): infected={state.infected}, frontier={state.frontier}"
            )
            last_strat = build_strategy(agent.generate(system, user))
            return last_strat.act(state, g, t)

        tr = env.rollout(action_fn, task.horizon, task.budget)
        if last_strat is None:
            raise StrategyError("per-step method produced no strategy")
        return last_strat, tr
