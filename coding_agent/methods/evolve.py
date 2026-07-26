"""
Method 4: EvoX-lite population evolution — every generation is an EDIT of a
parent from the population (refine or restructure), never a fresh program.
Stagnation switches the operator from refine to restructure.
"""

import time
from functools import partial
from tqdm import tqdm

from coding_agent.agent import CodingAgent, Conversation
from coding_agent.credit import planned_action
from coding_agent.executor import StrategyError, build_strategy, call_strategy
from coding_agent.methods.base import (
    OuterLoopMethod,
    baseline_anchor,
    reference_diff,
    summarize,
    validate_plan,
)
from coding_agent.prompts import (
    build_evolve_prompt,
    build_system_prompt,
    build_user_prompt,
)
from coding_agent.types import GraphInfo, Strategy, TaskSpec, Trajectory

reward_improvement_epsilon = 1e-9


class EvolveSearch(OuterLoopMethod):
    def __init__(
        self,
        outer_iters: int = 10,
        strategy_mode: str = "scored",
        stagnation_patience: int = 2,
        inspiration_count: int = 2,
    ) -> None:
        self.outer_iters = outer_iters
        self.strategy_mode = strategy_mode
        self.stagnation_patience = stagnation_patience
        self.inspiration_count = inspiration_count
        # Per-generation rewards, read back by run.py for the convergence plot
        self.history = []

    def optimize(
        self, agent: CodingAgent, environment: object, task: TaskSpec, graph: GraphInfo
    ) -> tuple[Strategy, Trajectory]:
        system = build_system_prompt("evolve", self.strategy_mode)

        anchor, anchor_trajectory = baseline_anchor(environment, task, graph)
        tqdm.write(f"[evolve] {anchor}")
        base_user = (
            build_user_prompt("evolve", task, graph, self.strategy_mode)
            + "\n\n"
            + anchor
        )

        # The thread lets the model see the generations it already produced;
        # build_evolve_prompt still names the PARENT explicitly because the
        # parent is the population best, which is usually NOT the last turn
        conversation = Conversation(agent, system)
        population = []
        best = None
        stagnation = 0
        last_error = None

        progress_bar = tqdm(range(self.outer_iters), desc="evolve search")
        for iteration in progress_bar:
            if not population:
                operator = "seed"
                error_text = (
                    f"\n\nYour previous attempt failed with:\n{last_error}\n"
                    if last_error
                    else ""
                )
                # Opening turn carries the task; a retry after a failed seed only
                # needs the error, since the thread still holds everything else
                user = base_user + error_text if iteration == 0 else error_text
                user = user or base_user
            else:
                operator = (
                    "restructure"
                    if stagnation >= self.stagnation_patience
                    else "refine"
                )

                # Deterministic exploit/explore: parent is the population best;
                # the operator (not parent sampling) supplies the variation
                parent = max(population, key=lambda record: record["reward"])
                inspirations = sorted(
                    (record for record in population if record is not parent),
                    key=lambda record: record["reward"],
                    reverse=True,
                )[: self.inspiration_count]

                # No base_user here: the task is already the thread's opening turn
                user = build_evolve_prompt(
                    operator, parent, inspirations, error=last_error
                )

            tqdm.write(
                f"[evolve] iter {iteration + 1}/{self.outer_iters}: "
                f"operator={operator}, population={len(population)}"
            )

            try:
                strategy = build_strategy(
                    conversation.send(user), self.strategy_mode
                )

                plan_start = time.perf_counter()
                tqdm.write(
                    f"[evolve] iter {iteration + 1}: executing plan_horizon()..."
                )
                plan = call_strategy(
                    strategy.plan_horizon, graph, task.budget, task.horizon
                )
                tqdm.write(
                    f"[evolve] iter {iteration + 1}: plan built in "
                    f"{time.perf_counter() - plan_start:.1f}s; rolling out..."
                )

                validate_plan(plan, task, graph)

                trajectory = environment.rollout(
                    partial(planned_action, plan), task.horizon, task.budget
                )
            except StrategyError as error:
                last_error = str(error)
                stagnation += 1
                self.history.append(
                    {
                        "iteration": iteration + 1,
                        "reward": None,
                        "operator": operator,
                        "error": last_error,
                    }
                )
                tqdm.write(
                    f"[evolve] iter {iteration + 1}: script failed — "
                    f"{last_error.splitlines()[0]}"
                )
                continue

            last_error = None
            # The reference diff rides along in the summary, so it reaches the
            # prompt wherever a population record is shown as parent or inspiration
            diff = reference_diff(trajectory, anchor_trajectory, graph)
            population.append(
                {
                    "script": strategy.source_script,
                    "reward": trajectory.reward,
                    "summary": summarize(trajectory, graph)
                    + (f"\n{diff}" if diff else ""),
                }
            )

            if (
                best is None
                or trajectory.reward > best[1].reward + reward_improvement_epsilon
            ):
                best = (strategy, trajectory)
                stagnation = 0
            elif operator == "restructure":
                # A restructure opens a fresh refinement window even without improvement
                stagnation = 0
            else:
                stagnation += 1

            self.history.append(
                {
                    "iteration": iteration + 1,
                    "reward": trajectory.reward,
                    "best": best[1].reward,
                    "operator": operator,
                }
            )
            tqdm.write(
                f"[evolve] iter {iteration + 1}: reward={trajectory.reward:.2f} "
                f"(best={best[1].reward:.2f}, stagnation={stagnation})"
            )
            progress_bar.set_postfix(
                reward=f"{trajectory.reward:.2f}", best=f"{best[1].reward:.2f}"
            )

        if best is None:
            raise StrategyError(
                f"all {self.outer_iters} attempts failed; last error: {last_error}"
            )

        return best
