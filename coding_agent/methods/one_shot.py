"""Method 1: one-shot super-algorithm with reward-driven skill refinement."""

from functools import partial

from coding_agent.agent import CodingAgent
from coding_agent.credit import counterfactual_credit, format_credit_report
from coding_agent.executor import StrategyError, build_strategy, validate_actions
from coding_agent.methods.base import summarize
from coding_agent.prompts import (
    build_feedback_prompt,
    build_user_prompt,
    system_prompts,
)
from coding_agent.types import GraphInfo, Strategy, TaskSpec, Trajectory


def _planned_action(plan: list[list], state: object, t: int) -> list:
    return plan[t] if t < len(plan) else []


class OneShotSuperAlgorithm:
    def __init__(self, outer_iters: int = 3, credit: bool = False) -> None:
        self.outer_iters = outer_iters
        self.credit = credit

    def optimize(
        self, agent: CodingAgent, env, task: TaskSpec, g: GraphInfo
    ) -> tuple[Strategy, Trajectory]:
        system = system_prompts["one_shot"]
        base_user = build_user_prompt("one_shot", task, g)
        user = base_user
        best: tuple[Strategy, Trajectory] | None = None
        last_err: str | None = None
        for _ in range(self.outer_iters):
            try:
                strat = build_strategy(agent.generate(system, user))
                plan = strat.plan_horizon(g, task.budget, task.horizon)
                total_adds = 0

                for bag in plan:
                    validate_actions(bag, g.num_nodes, task.budget)
                    total_adds += sum(1 for a in bag if a.op == "add_node")

                if total_adds > task.budget:
                    raise StrategyError(
                        f"plan adds {total_adds} seeds in total, exceeds budget {task.budget}"
                    )

                # Binding the plan once avoids a late-binding closure bug.
                action_fn = partial(_planned_action, plan)
                tr = env.rollout(action_fn, task.horizon, task.budget)
            except StrategyError as exc:
                last_err = str(exc)
                user = (
                    base_user
                    + "\n\n"
                    + build_feedback_prompt(0.0, "script failed", error=last_err)
                )
                continue
            if best is None or tr.reward > best[1].reward:
                best = (strat, tr)

            # Optional per-action counterfactual credit: costs one extra rollout per action, but turns the scalar reward into causal feedback
            report = None
            if self.credit:
                base, entries = counterfactual_credit(
                    env, plan, task.horizon, task.budget
                )
                report = format_credit_report(base, entries)

            user = (
                base_user
                + "\n\n"
                + build_feedback_prompt(tr.reward, summarize(tr), credit_report=report)
            )
        if best is None:
            raise StrategyError(
                f"all {self.outer_iters} attempts failed; last error: {last_err}"
            )
        return best
