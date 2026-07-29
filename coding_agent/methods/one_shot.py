"""Method 1: one-shot super-algorithm with reward-driven skill refinement."""

import time
from functools import partial
from tqdm import tqdm

from coding_agent.agent import CodingAgent, Conversation
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
    reference_diff,
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
        use_anchor: bool = True,
        allow_mc_algorithms: bool = False,
    ) -> None:
        self.outer_iters = outer_iters
        self.credit = credit
        self.strategy_mode = strategy_mode
        self.allow_mc_algorithms = allow_mc_algorithms
        # Canned arms (classical baselines, routing) never read a prompt, so the
        # anchor rollout would be pure cost — real episodes under an MC evaluator
        self.use_anchor = use_anchor
        # Per-iteration rewards, read back by run.py for the convergence plot
        self.history = []

    def optimize(
        self, agent: CodingAgent, environment: object, task: TaskSpec, graph: GraphInfo
    ) -> tuple[Strategy, Trajectory]:
        system = build_system_prompt("one_shot", self.strategy_mode)

        # One extra rollout up front: a classical score in the same env, so
        # "increase final spread" becomes a concrete target in every prompt
        base_user = build_user_prompt(
            "one_shot", task, graph, self.strategy_mode, self.allow_mc_algorithms
        )
        anchor_trajectory = None

        if self.use_anchor:
            anchor, anchor_trajectory = baseline_anchor(environment, task, graph)
            tqdm.write(f"[one_shot] {anchor}")
            base_user += "\n\n" + anchor

        # One thread for the whole refinement, so the base prompt is sent once and
        # every later turn is an edit against the script the model can still see
        conversation = Conversation(agent, system)
        # Read back by run.py for the closing plain-English write-up
        self.conversation = conversation
        pending = base_user
        best = None
        last_error = None
        last_script = None

        progress_bar = tqdm(range(self.outer_iters), desc="one_shot refinement")
        for iteration in progress_bar:
            try:
                tqdm.write(
                    f"[one_shot] iter {iteration + 1}/{self.outer_iters}: "
                    f"requesting strategy script..."
                )

                # Build the strategy object from the LLM generated code, and call it
                strategy = build_strategy(
                    conversation.send(pending),
                    self.strategy_mode,
                    self.allow_mc_algorithms,
                )
                last_script = strategy.source_script

                # The generated algorithm's own computation — with free-mode
                # composition scripts this internal planning dominates wall-clock
                plan_start = time.perf_counter()
                tqdm.write(
                    f"[one_shot] iter {iteration + 1}: executing plan_horizon()..."
                )
                plan = call_strategy(
                    strategy.plan_horizon, graph, task.budget, task.horizon
                )
                tqdm.write(
                    f"[one_shot] iter {iteration + 1}: plan built in "
                    f"{time.perf_counter() - plan_start:.1f}s; rolling out..."
                )

                validate_plan(plan, task, graph)

                # Bind the plan once into an ActionFn to avoid a late-binding closure bug
                action_fn = partial(planned_action, plan)

                # Roll the plan out and get the trajectory's reward, which is this iteration's score
                trajectory = environment.rollout(action_fn, task.horizon, task.budget)
            except StrategyError as error:
                last_error = str(error)
                self.history.append(
                    {"iteration": iteration + 1, "reward": None, "error": last_error}
                )

                # Log the first line of any StrategyError
                tqdm.write(
                    f"[one_shot] iter {iteration + 1}: script failed — "
                    f"{last_error.splitlines()[0]}"
                )

                # The thread already carries the task, so the next turn is only
                # the failure — plus the script that failed, when we got that far
                pending = build_feedback_prompt(
                    0.0, "script failed", error=last_error, script=last_script
                )

                continue

            # Keep the best strategy and trajectory pair by reward
            if best is None or trajectory.reward > best[1].reward:
                best = (strategy, trajectory)

            self.history.append(
                {
                    "iteration": iteration + 1,
                    "reward": trajectory.reward,
                    "best": best[1].reward,
                }
            )

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

            # Next turn: this iteration's reward, diagnostics, credit report, and
            # the script itself as the explicit edit target
            pending = build_feedback_prompt(
                trajectory.reward,
                summarize(trajectory, graph),
                credit_report=report,
                # Free: both marginal vectors are already paid for
                reference_report=(
                    reference_diff(trajectory, anchor_trajectory, graph)
                    if anchor_trajectory is not None
                    else None
                ),
                script=strategy.source_script,
            )

        if best is None:
            raise StrategyError(
                f"all {self.outer_iters} attempts failed; last error: {last_error}"
            )

        return best
