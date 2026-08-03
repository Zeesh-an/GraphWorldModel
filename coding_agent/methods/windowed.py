"""Method 3: design one online algorithm; re-apply it per time window."""

from coding_agent.agent import CodingAgent, Conversation
from coding_agent.executor import build_strategy, call_strategy, validate_actions
from coding_agent.methods.base import (
    OuterLoopMethod,
    attach_context,
    wrap_exogenous,
)
from coding_agent.prompts import build_system_prompt, build_user_prompt
from coding_agent.types import ActionOp, GraphInfo, State, Strategy, TaskSpec, Trajectory


class WindowedOnline(OuterLoopMethod):
    def __init__(self, windows: int = 3, allow_mc_algorithms: bool = False) -> None:
        self.windows = windows
        self.allow_mc_algorithms = allow_mc_algorithms
        # Seeds this method may commit per episode. Unlike the other methods this
        # is NOT task.budget: the budget is per window call by design, so an arm
        # nominally at k plays k x (number of window boundaries). run.py writes it
        # into the results JSON so the sweep table is not read as equal-budget.
        self.effective_budget = None

    def optimize(
        self, agent: CodingAgent, environment: object, task: TaskSpec, graph: GraphInfo
    ) -> tuple[Strategy, Trajectory]:
        system = build_system_prompt("windowed", task=task)
        user = build_user_prompt(
            "windowed", task, graph, allow_mc_algorithms=self.allow_mc_algorithms
        )
        print("[windowed] requesting online algorithm script...")

        # Designed once: the online algorithm is consulted at window boundaries by
        # the generated code, not by the LLM, so this thread is a single turn
        conversation = Conversation(agent, system)
        # Read back by run.py for the closing plain-English write-up
        self.conversation = conversation
        strategy = attach_context(
            build_strategy(
                conversation.send(user),
                allow_mc_algorithms=self.allow_mc_algorithms,
            ),
            task,
        )

        window_length = max(1, (task.horizon + 1) // self.windows)

        # The number of boundaries, not --windows: with horizon 10 and windows 3
        # the length rounds to 3 and t=0,3,6,9 all trigger a call
        boundaries = sum(
            1 for timestep in range(task.horizon + 1) if timestep % window_length == 0
        )
        self.effective_budget = task.budget * boundaries
        print(
            f"[windowed] rolling out {boundaries} window calls of {window_length} "
            f"steps; budget {task.budget} per call = {self.effective_budget} seeds "
            f"per episode"
        )

        def action_fn(state: State, timestep: int) -> list[ActionOp]:
            # Consult the online algorithm at each window boundary only
            if timestep % window_length == 0:
                bag = call_strategy(
                    strategy.act, state, graph, timestep // window_length
                )
                # Budget applies per window call, which is exactly the per-bag check
                validate_actions(
                    bag,
                    graph.num_nodes,
                    task.budget,
                    task.allowed_ops,
                    task.budget_op,
                    task.outbreak,
                )

                return bag

            return []

        # Roll the plan out and get the trajectory's reward, which is the score
        trajectory = environment.rollout(
            wrap_exogenous(action_fn, task, graph), task.horizon, task.budget
        )

        return strategy, trajectory
