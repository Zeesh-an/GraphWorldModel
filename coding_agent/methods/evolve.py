"""
Method 4: EvoX-lite population evolution — every generation is an EDIT of a
parent from the population (refine or restructure), never a fresh program.
Stagnation switches the operator from refine to restructure.
"""

from pathlib import Path
from tqdm import tqdm

from coding_agent import checkpoint
from coding_agent.agent import CodingAgent, Conversation
from coding_agent.executor import StrategyError, build_strategy
from coding_agent.methods.base import (
    OuterLoopMethod,
    baseline_anchor,
    evaluate_strategy,
    paired_delta,
    reference_diff,
    summarize,
)
from coding_agent.prompts import (
    build_evolve_prompt,
    build_system_prompt,
    build_user_prompt,
)
from coding_agent.types import (
    GraphInfo,
    Strategy,
    TaskSpec,
    Trajectory,
    best_by,
    improves,
    rank_by,
)

reward_improvement_epsilon = 1e-9


class EvolveSearch(OuterLoopMethod):
    def __init__(
        self,
        outer_iters: int = 10,
        strategy_mode: str = "scored",
        stagnation_patience: int = 2,
        inspiration_count: int = 2,
        label: str = "evolve",
        allow_mc_algorithms: bool = False,
        use_anchor: bool = True,
        checkpoint_path: Path | None = None,
        checkpoint_fingerprint: dict | None = None,
    ) -> None:
        self.outer_iters = outer_iters
        self.strategy_mode = strategy_mode
        # "evolve" or "adaptive": picks the system prompt (plan_horizon vs act per
        # round) and labels the logs. The search itself is the same either way.
        self.label = label
        self.allow_mc_algorithms = allow_mc_algorithms
        # Canned arms never read a prompt, so the anchor rollouts would be pure
        # cost — five real episodes under an MC evaluator, informing nothing
        self.use_anchor = use_anchor
        self.checkpoint_path = checkpoint_path
        self.checkpoint_fingerprint = checkpoint_fingerprint
        self.stagnation_patience = stagnation_patience
        self.inspiration_count = inspiration_count
        # Per-generation rewards, read back by run.py for the convergence plot
        self.history = []
        # Seeds this method may commit per episode; read back into the results JSON
        self.effective_budget = None

    def optimize(
        self, agent: CodingAgent, environment: object, task: TaskSpec, graph: GraphInfo
    ) -> tuple[Strategy, Trajectory]:
        system = build_system_prompt(self.label, self.strategy_mode, task)
        self.effective_budget = task.budget

        resumed = checkpoint.load(self.checkpoint_path, self.checkpoint_fingerprint)

        # The anchor rollouts are the expensive part of startup, so a resume
        # replays the recorded table rather than re-running the baselines
        anchor, anchor_trajectory, anchor_name = "", None, ""

        if resumed is not None and resumed["anchor"] is not None:
            anchor = resumed["anchor"]["text"]
            anchor_name = resumed["anchor"]["name"]
            anchor_trajectory = checkpoint.trajectory_from_dict(
                resumed["anchor"]["trajectory"]
            )
        elif self.use_anchor:
            anchor, anchor_trajectory, anchor_name = baseline_anchor(
                environment, task, graph
            )
            tqdm.write(f"[{self.label}] {anchor}")

        base_user = build_user_prompt(
            self.label, task, graph, self.strategy_mode, self.allow_mc_algorithms
        ) + (f"\n\n{anchor}" if anchor else "")

        # The thread lets the model see the generations it already produced;
        # build_evolve_prompt still names the PARENT explicitly because the
        # parent is the population best, which is usually NOT the last turn
        conversation = Conversation(agent, system)
        # Read back by run.py for the closing plain-English write-up
        self.conversation = conversation
        population = []
        best = None
        stagnation = 0
        last_error = None
        last_delta = None
        start_iteration = 0

        if resumed is not None:
            conversation.restore(resumed["messages"], resumed["transcript"])
            population = resumed["population"]
            stagnation = resumed["stagnation"]
            last_delta = resumed["last_delta"]
            self.history = resumed["history"]
            start_iteration = resumed["iteration"]

            if resumed["best"] is not None:
                best = (
                    build_strategy(
                        resumed["best"]["script"],
                        self.strategy_mode,
                        self.allow_mc_algorithms,
                    ),
                    checkpoint.trajectory_from_dict(resumed["best"]["trajectory"]),
                )

            best_text = "none yet" if best is None else f"{best[1].reward:.2f}"
            tqdm.write(
                f"[{self.label}] resumed from {self.checkpoint_path}: generation "
                f"{start_iteration + 1}/{self.outer_iters}, population "
                f"{len(population)}, best {best_text}"
            )

        progress_bar = tqdm(
            range(start_iteration, self.outer_iters),
            initial=start_iteration,
            total=self.outer_iters,
            desc=f"{self.label} search",
        )
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
                parent = best_by(
                    population, lambda record: record["reward"], task.sense
                )
                inspirations = rank_by(
                    [record for record in population if record is not parent],
                    lambda record: record["reward"],
                    task.sense,
                )[: self.inspiration_count]

                # No base_user here: the task is already the thread's opening turn
                user = build_evolve_prompt(
                    operator,
                    parent,
                    inspirations,
                    error=last_error,
                    last_result=last_delta,
                )

            tqdm.write(
                f"[{self.label}] iter {iteration + 1}/{self.outer_iters}: "
                f"operator={operator}, population={len(population)}"
            )

            try:
                strategy = build_strategy(
                    conversation.send(user),
                    self.strategy_mode,
                    self.allow_mc_algorithms,
                )

                entry_point = "act() per round" if task.adaptive else "plan_horizon()"
                tqdm.write(f"[{self.label}] iter {iteration + 1}: {entry_point}...")

                trajectory, plan_seconds = evaluate_strategy(
                    strategy, environment, task, graph
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
                    f"[{self.label}] iter {iteration + 1}: script failed: "
                    f"{last_error.splitlines()[0]}"
                )
                self._checkpoint(
                    iteration + 1,
                    conversation,
                    population,
                    best,
                    stagnation,
                    last_delta,
                    anchor,
                    anchor_name,
                    anchor_trajectory,
                )

                continue

            last_error = None
            # The reference diff rides along in the summary, so it reaches the
            # prompt wherever a population record is shown as parent or inspiration
            diff = reference_diff(
                trajectory, anchor_trajectory, graph, anchor_name, task.sense
            )
            last_delta = (
                paired_delta(
                    trajectory,
                    best[1],
                    "the population best",
                    task.sense,
                    unit=(
                        "score" if task.decodes else "F1" if task.recovers else "nodes"
                    ),
                )
                if best is not None
                else None
            )
            population.append(
                {
                    "script": strategy.source_script,
                    "reward": trajectory.reward,
                    "summary": summarize(trajectory, graph, task)
                    + (f"\n{last_delta}" if last_delta else "")
                    + (f"\n{diff}" if diff else ""),
                }
            )

            if best is None or improves(
                trajectory.reward,
                best[1].reward,
                task.sense,
                reward_improvement_epsilon,
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
                    "plan_seconds": round(plan_seconds, 3),
                    "rollout_seconds": round(
                        trajectory.cost.get("rollout_seconds", 0.0), 3
                    ),
                }
            )
            tqdm.write(
                f"[{self.label}] iter {iteration + 1}: reward={trajectory.reward:.2f} "
                f"(best={best[1].reward:.2f}, stagnation={stagnation})"
            )
            progress_bar.set_postfix(
                reward=f"{trajectory.reward:.2f}", best=f"{best[1].reward:.2f}"
            )
            self._checkpoint(
                iteration + 1,
                conversation,
                population,
                best,
                stagnation,
                last_delta,
                anchor,
                anchor_name,
                anchor_trajectory,
            )

        progress_bar.close()

        if best is None:
            raise StrategyError(
                f"all {self.outer_iters} attempts failed; last error: {last_error}"
            )

        return best

    def _checkpoint(
        self,
        iteration: int,
        conversation: Conversation,
        population: list[dict],
        best: tuple | None,
        stagnation: int,
        last_delta: str | None,
        anchor: str,
        anchor_name: str,
        anchor_trajectory: Trajectory | None,
    ) -> None:
        """Everything needed to continue this search, written after every generation."""
        checkpoint.save(
            self.checkpoint_path,
            {
                "fingerprint": self.checkpoint_fingerprint,
                "iteration": iteration,
                "population": population,
                "best": (
                    None
                    if best is None
                    else {
                        "script": best[0].source_script,
                        "trajectory": checkpoint.trajectory_to_dict(best[1]),
                    }
                ),
                "stagnation": stagnation,
                "last_delta": last_delta,
                "history": self.history,
                "messages": conversation.messages,
                "transcript": conversation.transcript,
                # Replayed on resume instead of re-running the baselines
                "anchor": (
                    None
                    if anchor_trajectory is None
                    else {
                        "text": anchor,
                        "name": anchor_name,
                        "trajectory": checkpoint.trajectory_to_dict(anchor_trajectory),
                    }
                ),
            },
        )
