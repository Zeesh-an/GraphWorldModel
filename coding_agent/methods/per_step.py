"""Method 2: re-prompt the agent at every timestep on the current state."""

from coding_agent.agent import CodingAgent
from coding_agent.executor import StrategyError, build_strategy, call_strategy
from coding_agent.prompts import build_user_prompt, system_prompts
from coding_agent.types import Action, GraphInfo, State, Strategy, TaskSpec, Trajectory


class PerStepReprompt:
    def optimize(
        self, agent: CodingAgent, environment: object, task: TaskSpec, graph: GraphInfo
    ) -> tuple[Strategy, Trajectory]:
        system = system_prompts["per_step"]
        last_strategy = None
        llm_calls = 0

        def action_fn(state: State, timestep: int) -> list[Action]:
            nonlocal last_strategy, llm_calls
            llm_calls += 1
            # One LLM call per (ensemble sample, timestep) — this line is the
            # only visibility into that cost while the rollout runs.
            print(
                f"[per_step] LLM call {llm_calls} (t={timestep}, "
                f"|infected|={len(state.infected)})"
            )
            user = build_user_prompt("per_step", task, graph) + (
                f"\n\nCURRENT STATE (t={timestep}): "
                f"infected={state.infected}, frontier={state.frontier}"
            )
            last_strategy = build_strategy(agent.generate(system, user))
            return call_strategy(last_strategy.act, state, graph, timestep)

        trajectory = environment.rollout(action_fn, task.horizon, task.budget)
        if last_strategy is None:
            raise StrategyError("per-step method produced no strategy")

        return last_strategy, trajectory
