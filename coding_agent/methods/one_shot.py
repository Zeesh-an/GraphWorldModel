"""Method 1: one-shot super-algorithm with reward-driven skill refinement."""

from coding_agent.agent import CodingAgent
from coding_agent.executor import StrategyError, build_strategy, validate_actions
from coding_agent.methods.base import summarize
from coding_agent.prompts import (
    SYSTEM_PROMPTS,
    build_feedback_prompt,
    build_user_prompt,
)
from coding_agent.types import GraphInfo, Strategy, TaskSpec, Trajectory


class OneShotSuperAlgorithm:
    def __init__(self, outer_iters: int = 3) -> None:
        self.outer_iters = outer_iters

    def optimize(
        self, agent: CodingAgent, env, task: TaskSpec, g: GraphInfo
    ) -> tuple[Strategy, Trajectory]:
        system = SYSTEM_PROMPTS["one_shot"]
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
                # Default-arg capture avoids late-binding closure bug.
                action_fn = lambda s, t, _p=plan: _p[t] if t < len(_p) else []
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
            user = base_user + "\n\n" + build_feedback_prompt(tr.reward, summarize(tr))
        if best is None:
            raise StrategyError(
                f"all {self.outer_iters} attempts failed; last error: {last_err}"
            )
        return best
