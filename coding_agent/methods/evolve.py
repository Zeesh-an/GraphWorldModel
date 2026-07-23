"""
Method 4: EvoX-lite population evolution — every generation is an EDIT of a
parent from the population (refine or restructure), never a fresh program.
Stagnation switches the operator from refine to restructure.
"""

from functools import partial
from tqdm import tqdm

from coding_agent.agent import CodingAgent
from coding_agent.credit import planned_action
from coding_agent.executor import StrategyError, build_strategy, call_strategy
from coding_agent.methods.base import (
    OuterLoopMethod,
    baseline_anchor,
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

    def optimize(
        self, agent: CodingAgent, environment: object, task: TaskSpec, graph: GraphInfo
    ) -> tuple[Strategy, Trajectory]:
        system = build_system_prompt("evolve", self.strategy_mode)

        anchor = baseline_anchor(environment, task, graph)
        tqdm.write(f"[evolve] {anchor}")
        base_user = (
            build_user_prompt("evolve", task, graph, self.strategy_mode)
            + "\n\n"
            + anchor
        )

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
                user = base_user + error_text
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

                user = base_user + build_evolve_prompt(
                    operator, parent, inspirations, error=last_error
                )

            tqdm.write(
                f"[evolve] iter {iteration + 1}/{self.outer_iters}: "
                f"operator={operator}, population={len(population)}"
            )

            try:
                strategy = build_strategy(
                    agent.generate(system, user), self.strategy_mode
                )
                plan = call_strategy(
                    strategy.plan_horizon, graph, task.budget, task.horizon
                )
                validate_plan(plan, task, graph)

                trajectory = environment.rollout(
                    partial(planned_action, plan), task.horizon, task.budget
                )
            except StrategyError as error:
                last_error = str(error)
                stagnation += 1
                tqdm.write(
                    f"[evolve] iter {iteration + 1}: script failed — "
                    f"{last_error.splitlines()[0]}"
                )
                continue

            last_error = None
            population.append(
                {
                    "script": strategy.source_script,
                    "reward": trajectory.reward,
                    "summary": summarize(trajectory, graph),
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
