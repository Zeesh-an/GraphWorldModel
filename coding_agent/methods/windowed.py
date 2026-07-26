"""Method 3: design one online algorithm; re-apply it per time window."""

from coding_agent.agent import CodingAgent, Conversation
from coding_agent.executor import build_strategy, call_strategy, validate_actions
from coding_agent.methods.base import OuterLoopMethod
from coding_agent.prompts import build_user_prompt, system_prompts
from coding_agent.types import ActionOp, GraphInfo, State, Strategy, TaskSpec, Trajectory


class WindowedOnline(OuterLoopMethod):
    def __init__(self, windows: int = 3) -> None:
        self.windows = windows

    def optimize(
        self, agent: CodingAgent, environment: object, task: TaskSpec, graph: GraphInfo
    ) -> tuple[Strategy, Trajectory]:
        system = system_prompts["windowed"]
        user = build_user_prompt("windowed", task, graph)
        print("[windowed] requesting online algorithm script...")

        # Designed once: the online algorithm is consulted at window boundaries by
        # the generated code, not by the LLM, so this thread is a single turn
        strategy = build_strategy(Conversation(agent, system).send(user))

        window_length = max(1, (task.horizon + 1) // self.windows)
        print(f"[windowed] rolling out {self.windows} windows of {window_length} steps")

        def action_fn(state: State, timestep: int) -> list[ActionOp]:
            # Consult the online algorithm at each window boundary only
            if timestep % window_length == 0:
                bag = call_strategy(
                    strategy.act, state, graph, timestep // window_length
                )
                # Budget applies per window call, which is exactly the per-bag check
                validate_actions(bag, graph.num_nodes, task.budget, task.allowed_ops)

                return bag

            return []

        # Roll the plan out and get the trajectory's reward, which is the score
        trajectory = environment.rollout(action_fn, task.horizon, task.budget)

        return strategy, trajectory
