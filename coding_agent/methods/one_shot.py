"""Method 1: one-shot super-algorithm with reward-driven skill refinement."""

from functools import partial
from tqdm import tqdm

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
)
from coding_agent.methods.base import (
    OuterLoopMethod,
    baseline_anchor,
    summarize,
    validate_plan,
)
from coding_agent.prompts import (
    build_feedback_prompt,
    build_system_prompt,
    build_user_prompt,
)
from coding_agent.types import GraphInfo, Strategy, TaskSpec, Trajectory


class OneShotSuperAlgorithm(OuterLoopMethod):
    def __init__(
        self,
        outer_iters: int = 3,
        credit: bool = False,
        strategy_mode: str = "free",
    ) -> None:
        self.outer_iters = outer_iters
        self.credit = credit
        self.strategy_mode = strategy_mode

    def optimize(
        self, agent: CodingAgent, environment: object, task: TaskSpec, graph: GraphInfo
    ) -> tuple[Strategy, Trajectory]:
        system = build_system_prompt("one_shot", self.strategy_mode)

        # One extra rollout up front: a classical score in the same env, so
        # "increase final spread" becomes a concrete target in every prompt
        anchor = baseline_anchor(environment, task, graph)
        tqdm.write(f"[one_shot] {anchor}")

        base_user = (
            build_user_prompt("one_shot", task, graph, self.strategy_mode)
            + "\n\n"
            + anchor
        )
        user = base_user
        best = None
        last_error = None

        progress_bar = tqdm(range(self.outer_iters), desc="one_shot refinement")
        for iteration in progress_bar:
            try:
                tqdm.write(
                    f"[one_shot] iter {iteration + 1}/{self.outer_iters}: "
                    f"requesting strategy script..."
                )

                # Build the strategy object from the LLM generated code, and call it
                strategy = build_strategy(
                    agent.generate(system, user), self.strategy_mode
                )
                plan = call_strategy(
                    strategy.plan_horizon, graph, task.budget, task.horizon
                )
                validate_plan(plan, task, graph)

                # Bind the plan once into an ActionFn to avoid a late-binding closure bug
                action_fn = partial(planned_action, plan)

                # Roll the plan out and get the trajectory's reward, which is this iteration's score
                trajectory = environment.rollout(action_fn, task.horizon, task.budget)
            except StrategyError as error:
                last_error = str(error)

                # Log the first line of any StrategyError
                tqdm.write(
                    f"[one_shot] iter {iteration + 1}: script failed — "
                    f"{last_error.splitlines()[0]}"
                )

                # Rebuild the user prompt with the feedback containing a reward of 0.0 and the full error text
                user = (
                    base_user
                    + "\n\n"
                    + build_feedback_prompt(0.0, "script failed", error=last_error)
                )

                continue

            # Keep the best strategy and trajectory pair by reward
            if best is None or trajectory.reward > best[1].reward:
                best = (strategy, trajectory)

            tqdm.write(
                f"[one_shot] iter {iteration + 1}: reward={trajectory.reward:.2f} "
                f"(best={best[1].reward:.2f}, "
                f"rollout {trajectory.cost.get('rollout_seconds', 0.0):.2f}s)"
            )
            progress_bar.set_postfix(
                reward=f"{trajectory.reward:.2f}", best=f"{best[1].reward:.2f}"
            )

            # If using the credit flag, ablate each action and run the optional per-action counterfactual credit
            # Costs one extra rollout per action, but turns the scalar reward into causal feedback
            report = None

            if self.credit:
                base_reward, entries = counterfactual_credit(
                    environment, plan, task.horizon, task.budget
                )
                report = format_credit_report(base_reward, entries)

            # Rebuild the user prompt with the feedback containing this iteration's trajectory reward, summary report, and credit report
            user = (
                base_user
                + "\n\n"
                + build_feedback_prompt(
                    trajectory.reward, summarize(trajectory, graph), credit_report=report
                )
            )

        if best is None:
            raise StrategyError(
                f"all {self.outer_iters} attempts failed; last error: {last_error}"
            )

        return best
