"""Method 1: one-shot super-algorithm with reward-driven skill refinement."""

from functools import partial

from coding_agent.agent import CodingAgent
from coding_agent.credit import (
    counterfactual_credit,
    format_credit_report,
    planned_action,
)
from coding_agent.executor import (
    StrategyError,
    build_strategy,
    call_strategy,
    validate_actions,
)
from coding_agent.methods.base import summarize
from coding_agent.prompts import (
    build_feedback_prompt,
    build_user_prompt,
    system_prompts,
)
from coding_agent.types import GraphInfo, Strategy, TaskSpec, Trajectory


class OneShotSuperAlgorithm:
    def __init__(self, outer_iters: int = 3, credit: bool = False) -> None:
        self.outer_iters = outer_iters
        self.credit = credit

    def optimize(
        self, agent: CodingAgent, environment: object, task: TaskSpec, graph: GraphInfo
    ) -> tuple[Strategy, Trajectory]:
        system = system_prompts["one_shot"]
        base_user = build_user_prompt("one_shot", task, graph)
        user = base_user
        best = None
        last_error = None

        for _ in range(self.outer_iters):
            try:
                strategy = build_strategy(agent.generate(system, user))
                plan = call_strategy(
                    strategy.plan_horizon, graph, task.budget, task.horizon
                )
                total_adds = 0

                for bag in plan:
                    validate_actions(bag, graph.num_nodes, task.budget)
                    total_adds += sum(1 for action in bag if action.op == "add_node")

                if total_adds > task.budget:
                    raise StrategyError(
                        f"plan adds {total_adds} seeds in total, "
                        f"exceeds budget {task.budget}"
                    )

                # Binding the plan once avoids a late-binding closure bug.
                action_fn = partial(planned_action, plan)
                trajectory = environment.rollout(action_fn, task.horizon, task.budget)
            except StrategyError as error:
                last_error = str(error)
                user = (
                    base_user
                    + "\n\n"
                    + build_feedback_prompt(0.0, "script failed", error=last_error)
                )
                continue

            if best is None or trajectory.reward > best[1].reward:
                best = (strategy, trajectory)

            # Optional per-action counterfactual credit: costs one extra rollout per
            # action, but turns the scalar reward into causal feedback.
            report = None
            if self.credit:
                base_reward, entries = counterfactual_credit(
                    environment, plan, task.horizon, task.budget
                )
                report = format_credit_report(base_reward, entries)

            user = (
                base_user
                + "\n\n"
                + build_feedback_prompt(
                    trajectory.reward, summarize(trajectory), credit_report=report
                )
            )

        if best is None:
            raise StrategyError(
                f"all {self.outer_iters} attempts failed; last error: {last_error}"
            )

        return best
