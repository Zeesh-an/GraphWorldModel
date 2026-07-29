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
    paired_delta,
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


# A script that fails to build or run teaches the next turn something, but it is
# not an evaluation — charging it against --outer-iters silently turns a 5-round
# search into a 3-round one. Repairs get their own budget instead.
default_max_repairs = 3


class OneShotSuperAlgorithm(OuterLoopMethod):
    def __init__(
        self,
        outer_iters: int = 3,
        credit: bool = False,
        strategy_mode: str = "free",
        use_anchor: bool = True,
        allow_mc_algorithms: bool = False,
        max_repairs: int = default_max_repairs,
    ) -> None:
        self.outer_iters = outer_iters
        self.credit = credit
        self.strategy_mode = strategy_mode
        self.allow_mc_algorithms = allow_mc_algorithms
        self.max_repairs = max_repairs
        # Canned arms (classical baselines, routing) never read a prompt, so the
        # anchor rollout would be pure cost — real episodes under an MC evaluator
        self.use_anchor = use_anchor
        # Per-iteration rewards, read back by run.py for the convergence plot
        self.history = []
        # Seeds this method may commit per episode; read back into the results
        # JSON so arms with different effective budgets are not silently compared
        self.effective_budget = None

    def optimize(
        self, agent: CodingAgent, environment: object, task: TaskSpec, graph: GraphInfo
    ) -> tuple[Strategy, Trajectory]:
        system = build_system_prompt("one_shot", self.strategy_mode, task)
        self.effective_budget = task.budget

        # One rollout per classical baseline up front: the score table in the
        # same env, so "increase final spread" becomes a concrete bar to clear
        base_user = build_user_prompt(
            "one_shot", task, graph, self.strategy_mode, self.allow_mc_algorithms
        )
        anchor_trajectory = None
        anchor_name = ""

        if self.use_anchor:
            anchor, anchor_trajectory, anchor_name = baseline_anchor(
                environment, task, graph
            )
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
        evaluations = 0
        repairs = 0
        iteration = 0

        progress_bar = tqdm(total=self.outer_iters, desc="one_shot refinement")
        while evaluations < self.outer_iters:
            iteration += 1
            try:
                tqdm.write(
                    f"[one_shot] iter {iteration} "
                    f"(evaluation {evaluations + 1}/{self.outer_iters}, "
                    f"repairs {repairs}/{self.max_repairs}): "
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
                tqdm.write(f"[one_shot] iter {iteration}: executing plan_horizon()...")
                plan = call_strategy(
                    strategy.plan_horizon, graph, task.budget, task.horizon
                )
                plan_seconds = time.perf_counter() - plan_start
                tqdm.write(
                    f"[one_shot] iter {iteration}: plan built in "
                    f"{plan_seconds:.1f}s; rolling out..."
                )

                validate_plan(plan, task, graph)

                # Bind the plan once into an ActionFn to avoid a late-binding closure bug
                action_fn = partial(planned_action, plan)

                # Roll the plan out and get the trajectory's reward, which is this iteration's score
                trajectory = environment.rollout(action_fn, task.horizon, task.budget)
            except StrategyError as error:
                last_error = str(error)
                repairs += 1
                self.history.append(
                    {
                        "iteration": iteration,
                        "reward": None,
                        "error": last_error,
                        "repair": repairs,
                    }
                )

                # Log the first line of any StrategyError
                tqdm.write(
                    f"[one_shot] iter {iteration}: script failed "
                    f"(repair {repairs}/{self.max_repairs}) — "
                    f"{last_error.splitlines()[0]}"
                )

                if repairs >= self.max_repairs:
                    tqdm.write(
                        f"[one_shot] repair budget exhausted after {repairs} failures"
                    )
                    break

                # The thread already carries the task, so the next turn is only
                # the failure — plus the script that failed, when we got that far,
                # and the best working script when one exists to fall back to
                pending = build_feedback_prompt(
                    0.0,
                    "script failed",
                    error=last_error,
                    script=last_script,
                    incumbent_script=best[0].source_script if best else None,
                    incumbent_reward=best[1].reward if best else None,
                )

                continue

            evaluations += 1
            progress_bar.update(1)

            # Captured before `best` moves: the paired delta and the edit target
            # both refer to the incumbent this attempt was measured against
            previous_best = best

            if best is None or trajectory.reward > best[1].reward:
                best = (strategy, trajectory)

            self.history.append(
                {
                    "iteration": iteration,
                    "reward": trajectory.reward,
                    "best": best[1].reward,
                    "plan_seconds": round(plan_seconds, 3),
                    "rollout_seconds": round(
                        trajectory.cost.get("rollout_seconds", 0.0), 3
                    ),
                }
            )

            tqdm.write(
                f"[one_shot] iter {iteration}: reward={trajectory.reward:.2f} "
                f"(best={best[1].reward:.2f}, plan {plan_seconds:.2f}s, "
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

            # Next turn: this iteration's reward and diagnostics, and the BEST
            # script so far as the edit target — editing the latest attempt
            # instead makes a regression the base for every iteration after it
            pending = build_feedback_prompt(
                trajectory.reward,
                summarize(trajectory, graph, task),
                credit_report=report,
                # Free: both marginal vectors are already paid for
                reference_report=(
                    reference_diff(trajectory, anchor_trajectory, graph, anchor_name)
                    if anchor_trajectory is not None
                    else None
                ),
                script=strategy.source_script,
                incumbent_script=(
                    best[0].source_script if best[0] is not strategy else None
                ),
                incumbent_reward=best[1].reward,
                delta_report=(
                    paired_delta(trajectory, previous_best[1])
                    if previous_best is not None
                    else None
                ),
            )

        progress_bar.close()

        if best is None:
            raise StrategyError(
                f"no attempt produced a runnable strategy after {iteration} tries; "
                f"last error: {last_error}"
            )

        return best
