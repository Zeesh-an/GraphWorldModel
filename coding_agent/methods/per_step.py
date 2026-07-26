"""Method 2: re-prompt the agent at every timestep on the current state."""

from coding_agent.agent import CodingAgent, Conversation
from coding_agent.executor import (
    StrategyError,
    build_strategy,
    call_strategy,
    validate_actions,
)
from coding_agent.methods.base import OuterLoopMethod
from coding_agent.prompts import build_user_prompt, system_prompts
from coding_agent.types import ActionOp, GraphInfo, State, Strategy, TaskSpec, Trajectory


class PerStepReprompt(OuterLoopMethod):
    def optimize(
        self, agent: CodingAgent, environment: object, task: TaskSpec, graph: GraphInfo
    ) -> tuple[Strategy, Trajectory]:
        system = system_prompts["per_step"]
        # The thread runs WITHIN one episode: the agent sees the states its own
        # earlier actions produced. It resets at t=0 because the next episode is
        # an independent sample and the old trajectory would be misleading.
        conversation = Conversation(agent, system)
        last_strategy = None
        llm_calls = 0

        def action_fn(state: State, timestep: int) -> list[ActionOp]:
            nonlocal last_strategy, llm_calls

            # One LLM call per (ensemble sample, timestep)
            llm_calls += 1

            print(
                f"[per_step] LLM call {llm_calls} (t={timestep}, "
                f"|infected|={len(state.infected)})"
            )

            state_text = (
                f"CURRENT STATE (t={timestep}): "
                f"infected={state.infected}, frontier={state.frontier}"
            )

            if timestep == 0:
                conversation.reset()
                user = build_user_prompt("per_step", task, graph) + "\n\n" + state_text
            else:
                user = (
                    f"{state_text}\n\nThis is the state your previous action "
                    f"produced. Reply with one ```python block for this timestep."
                )

            # Build the strategy object from the LLM generated code, and call it
            last_strategy = build_strategy(conversation.send(user))

            bag = call_strategy(last_strategy.act, state, graph, timestep)
            validate_actions(bag, graph.num_nodes, task.budget, task.allowed_ops)

            return bag

        # Roll the plan out and get the trajectory's reward, which is the score
        trajectory = environment.rollout(action_fn, task.horizon, task.budget)

        if last_strategy is None:
            raise StrategyError("per-step method produced no strategy")

        return last_strategy, trajectory
