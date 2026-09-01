"""Method 1: one-shot super-algorithm with reward-driven skill refinement."""

from pathlib import Path
from tqdm import tqdm

from coding_agent import checkpoint
from coding_agent.agent import CodingAgent, Conversation
from coding_agent.credit import credit_feedback
from coding_agent.probes import answer_probes, parse_probe_request
from coding_agent.executor import StrategyError, build_strategy
from coding_agent.methods.base import (
    OuterLoopMethod,
    baseline_anchor,
    evaluate_strategy,
    rescore,
    paired_delta,
    reference_diff,
    summarize,
)
from coding_agent.prompts import (
    build_feedback_prompt,
    probe_contract,
    build_system_prompt,
    build_user_prompt,
)
from coding_agent.types import GraphInfo, Strategy, TaskSpec, Trajectory, improves


# A script that fails to build or run teaches the next turn something, but it is
# not an evaluation: charging it against --outer-iters silently turns a 5-round
# search into a 3-round one. Repairs get their own budget instead.
default_max_repairs = 3


class OneShotSuperAlgorithm(OuterLoopMethod):
    def __init__(
        self,
        outer_iters: int = 20,
        credit: bool = False,
        strategy_mode: str = "free",
        use_anchor: bool = True,
        allow_mc_algorithms: bool = False,
        max_repairs: int = default_max_repairs,
        checkpoint_path: Path | None = None,
        checkpoint_fingerprint: dict | None = None,
        probes: bool = True,
    ) -> None:
        self.outer_iters = outer_iters
        self.credit = credit
        # False on the native arm (no forward model to ask) and on canned arms
        self.probes = probes
        # Every answered probe, read back into the results JSON by run.py
        self.probe_log = []
        self.strategy_mode = strategy_mode
        self.allow_mc_algorithms = allow_mc_algorithms
        self.max_repairs = max_repairs
        self.checkpoint_path = checkpoint_path
        self.checkpoint_fingerprint = checkpoint_fingerprint
        # Canned arms (classical baselines, routing) never read a prompt, so the
        # anchor rollout would be pure cost: real episodes under an MC evaluator
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
        # What the reward MEASURES, in the model's own terms. "final spread" is a
        # node count on the intervention tasks and meaningless on the inverse one,
        # where the number is an F1 in [0, 1].
        objective_label = (
            "the tree-weighted reconstruction score"
            if task.decodes
            else "F1 against the true source set"
            if task.recovers
            else ("final infected count" if task.contains else "final spread")
        )

        # One rollout per classical baseline up front: the score table in the
        # same env, so "increase final spread" becomes a concrete bar to clear
        base_user = build_user_prompt(
            "one_shot", task, graph, self.strategy_mode, self.allow_mc_algorithms
        )
        anchor_trajectory = None
        anchor_name = ""
        anchor = ""

        resumed = checkpoint.load(self.checkpoint_path, self.checkpoint_fingerprint)

        # The anchor rollouts are the expensive part of startup, so a resume
        # replays the recorded table rather than re-running the baselines
        if resumed is not None and resumed["anchor"] is not None:
            anchor = resumed["anchor"]["text"]
            anchor_name = resumed["anchor"]["name"]
            anchor_trajectory = checkpoint.trajectory_from_dict(
                resumed["anchor"]["trajectory"]
            )
            base_user += "\n\n" + anchor
        elif self.use_anchor:
            anchor, anchor_trajectory, anchor_name = baseline_anchor(
                environment, task, graph
            )
            tqdm.write(f"[one_shot] {anchor}")
            base_user += "\n\n" + anchor

        # One thread for the whole refinement, so the base prompt is sent once and
        # every later turn is an edit against the script the model can still see
        if self.probes:
            base_user += probe_contract

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
        probe_request, probe_note = [], None

        if resumed is not None:
            conversation.restore(resumed["messages"], resumed["transcript"])
            self.history = resumed["history"]
            pending = resumed["pending"]
            last_script = resumed["last_script"]
            last_error = resumed["last_error"]
            evaluations = resumed["evaluations"]
            repairs = resumed["repairs"]
            # .get: checkpoints from before probes existed still resume
            self.probe_log = resumed.get("probe_log", [])
            iteration = resumed["iteration"]

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
                f"[one_shot] resumed from {self.checkpoint_path}: evaluation "
                f"{evaluations + 1}/{self.outer_iters}, repairs "
                f"{repairs}/{self.max_repairs}, best {best_text}"
            )

        progress_bar = tqdm(
            total=self.outer_iters, initial=evaluations, desc="one_shot refinement"
        )
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
                script = conversation.send(pending)
                # The probes fence rides the prose send() discards; the raw
                # reply is the transcript's last generate turn
                probe_request, probe_note = parse_probe_request(
                    conversation.transcript[-1]["reply"]
                )
                strategy = build_strategy(
                    script,
                    self.strategy_mode,
                    self.allow_mc_algorithms,
                )
                last_script = strategy.source_script

                # The generated algorithm's own computation, with free-mode
                # composition scripts this internal planning dominates wall-clock
                tqdm.write(f"[one_shot] iter {iteration}: executing plan_horizon()...")

                # Same pairing rule as evolve: a fresh seed per attempt, and the
                # incumbent re-scored on it before the two are compared
                seed = task.seed + evaluations + 1
                trajectory, plan_seconds = evaluate_strategy(
                    strategy, environment, task, graph, seed=seed
                )
                best = rescore(best, environment, task, graph, seed)
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
                    f"(repair {repairs}/{self.max_repairs}): "
                    f"{last_error.splitlines()[0]}"
                )

                if repairs >= self.max_repairs:
                    tqdm.write(
                        f"[one_shot] repair budget exhausted after {repairs} failures"
                    )
                    break

                # The thread already carries the task, so the next turn is only
                # the failure, plus the script that failed, when we got that far,
                # and the best working script when one exists to fall back to
                pending = build_feedback_prompt(
                    0.0,
                    "script failed",
                    error=last_error,
                    script=last_script,
                    incumbent_script=best[0].source_script if best else None,
                    incumbent_reward=best[1].reward if best else None,
                    objective=objective_label,
                )

                probe_feedback = answer_probes(
                    environment,
                    probe_request,
                    probe_note,
                    best[1].actions if best is not None else None,
                    task,
                    self.probes,
                    self.probe_log,
                    "one_shot",
                )
                if probe_feedback:
                    pending = f"{pending}\n\n{probe_feedback}"

                # After `pending` is built, so a resume re-sends the repair turn
                # rather than the prompt that already failed
                self._checkpoint(
                    iteration,
                    conversation,
                    best,
                    pending,
                    last_script,
                    last_error,
                    evaluations,
                    repairs,
                    anchor,
                    anchor_name,
                    anchor_trajectory,
                )

                continue

            evaluations += 1
            progress_bar.update(1)

            # Captured before `best` moves: the paired delta and the edit target
            # both refer to the incumbent this attempt was measured against
            previous_best = best

            if best is None or improves(
                trajectory.reward, best[1].reward, task.sense
            ):
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

            # Counterfactual credit ablates one ACTION at a time, and an inverse
            # or forecast task emits no actions: there is nothing to ablate and
            # nothing the rollout would answer
            if self.credit:
                # The bags that actually ran, not the plan object: identical for
                # a static plan, and the only thing that exists for an adaptive
                # policy. Replaying them as a fixed plan is the approximation
                # credit.py already documents for state-dependent strategies.
                # credit_feedback guards the inverse/forecast families itself
                report = credit_feedback(environment, trajectory, task, graph)

            # Next turn: this iteration's reward and diagnostics, and the BEST
            # script so far as the edit target: editing the latest attempt
            # instead makes a regression the base for every iteration after it
            pending = build_feedback_prompt(
                trajectory.reward,
                summarize(trajectory, graph, task),
                credit_report=report,
                # Free: both marginal vectors are already paid for
                reference_report=(
                    reference_diff(
                        trajectory, anchor_trajectory, graph, anchor_name, task.sense
                    )
                    if anchor_trajectory is not None
                    else None
                ),
                script=strategy.source_script,
                incumbent_script=(
                    best[0].source_script if best[0] is not strategy else None
                ),
                incumbent_reward=best[1].reward,
                delta_report=(
                    paired_delta(
                        trajectory,
                        previous_best[1],
                        sense=task.sense,
                        unit=task.reward_unit,
                    )
                    if previous_best is not None
                    else None
                ),
                objective=objective_label,
            )

            probe_feedback = answer_probes(
                environment,
                probe_request,
                probe_note,
                trajectory.actions,
                task,
                self.probes,
                self.probe_log,
                "one_shot",
            )
            if probe_feedback:
                pending = f"{pending}\n\n{probe_feedback}"

            self._checkpoint(
                iteration,
                conversation,
                best,
                pending,
                last_script,
                last_error,
                evaluations,
                repairs,
                anchor,
                anchor_name,
                anchor_trajectory,
            )

        progress_bar.close()

        if best is None:
            raise StrategyError(
                f"no attempt produced a runnable strategy after {iteration} tries; "
                f"last error: {last_error}"
            )

        return best

    def _checkpoint(
        self,
        iteration: int,
        conversation: Conversation,
        best: tuple | None,
        pending: str,
        last_script: str | None,
        last_error: str | None,
        evaluations: int,
        repairs: int,
        anchor: str,
        anchor_name: str,
        anchor_trajectory: Trajectory | None,
    ) -> None:
        """Everything needed to continue this loop, written after every turn."""
        checkpoint.save(
            self.checkpoint_path,
            {
                "fingerprint": self.checkpoint_fingerprint,
                "iteration": iteration,
                "evaluations": evaluations,
                "repairs": repairs,
                "best": (
                    None
                    if best is None
                    else {
                        "script": best[0].source_script,
                        "trajectory": checkpoint.trajectory_to_dict(best[1]),
                    }
                ),
                "pending": pending,
                "last_script": last_script,
                "last_error": last_error,
                "history": self.history,
                "probe_log": self.probe_log,
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
