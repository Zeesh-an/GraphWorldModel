"""Method 2: re-prompt the agent at every timestep on the current state."""

from coding_agent.agent import CodingAgent, Conversation
from coding_agent.executor import (
    StrategyError,
    build_strategy,
    call_strategy,
    validate_actions,
)
from coding_agent.methods.base import (
    OuterLoopMethod,
    attach_context,
    wrap_exogenous,
)
from coding_agent.prompts import build_system_prompt, build_user_prompt
from coding_agent.types import ActionOp, GraphInfo, State, Strategy, TaskSpec, Trajectory


class PerStepReprompt(OuterLoopMethod):
    def __init__(self, allow_mc_algorithms: bool = False) -> None:
        self.allow_mc_algorithms = allow_mc_algorithms
        # Seeds this method may commit per episode; read back into the results JSON
        self.effective_budget = None

    def optimize(
        self, agent: CodingAgent, environment: object, task: TaskSpec, graph: GraphInfo
    ) -> tuple[Strategy, Trajectory]:
        system = build_system_prompt("per_step", task=task)
        self.effective_budget = task.budget
        # The thread runs WITHIN one episode: the agent sees the states its own
        # earlier actions produced. It resets at t=0 because the next episode is
        # an independent sample and the old trajectory would be misleading.
        conversation = Conversation(agent, system)
        # Read back by run.py for the closing plain-English write-up
        self.conversation = conversation
        last_strategy = None
        llm_calls = 0
        seeded = set()

        def action_fn(state: State, timestep: int) -> list[ActionOp]:
            nonlocal last_strategy, llm_calls

            # The budget is per EPISODE, not per bag: without this the policy may
            # legally emit `budget` seeds at every one of horizon+1 timesteps and
            # play the same nominal k as a one_shot arm with 11x the seeds.
            # t=0 starts a new episode under a sequential env; under a lockstep
            # ensemble env every sample's t=0 also resets, so the cap is shared
            # from t>0 onwards: stricter than per-sample, never looser.
            if timestep == 0:
                seeded.clear()

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

            remaining = task.budget - len(seeded)

            if timestep == 0:
                conversation.reset()
                user = (
                    build_user_prompt(
                        "per_step",
                        task,
                        graph,
                        allow_mc_algorithms=self.allow_mc_algorithms,
                    )
                    + "\n\n"
                    + state_text
                )
            else:
                user = (
                    f"{state_text}\n\nThis is the state your previous action "
                    f"produced. You have {remaining} of your {task.budget} seeds "
                    f"left for this episode (already seeded: {sorted(seeded)}). "
                    f"Reply with one ```python block for this timestep."
                )

            # Build the strategy object from the LLM generated code, and call it
            last_strategy = attach_context(
                build_strategy(
                    conversation.send(user),
                    allow_mc_algorithms=self.allow_mc_algorithms,
                ),
                task,
            )

            bag = call_strategy(last_strategy.act, state, graph, timestep)
            validate_actions(
                bag,
                graph.num_nodes,
                remaining,
                task.allowed_ops,
                task.budget_op,
                task.outbreak,
            )

            for action in bag:
                if action.op == task.budget_op:
                    if int(action.target) in seeded:
                        raise StrategyError(
                            f"node {action.target} was already seeded earlier in "
                            f"this episode; re-seeding it spends budget without "
                            f"adding a node. Already seeded: {sorted(seeded)}."
                        )

                    seeded.add(int(action.target))

            return bag

        # Roll the plan out and get the trajectory's reward, which is the score
        trajectory = environment.rollout(
            wrap_exogenous(action_fn, task, graph), task.horizon, task.budget
        )

        if last_strategy is None:
            raise StrategyError("per-step method produced no strategy")

        return last_strategy, trajectory
