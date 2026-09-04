"""
Method 4: population evolution with a real operator set.

Every generation applies ONE operator to the population: refine, parameters or
simplify (exploit: an edit of the population best), or crossover, synthesize or
from_scratch (explore: a new mechanism). The operator is drawn with weights that
follow MCTS-AHD's schedule, an exploration term that scales with the REMAINING
budget, so restructuring happens early and refinement late, with stagnation
adding exploration back. A candidate replaces the incumbent only when its paired
delta clears the standard error of the comparison (`methods.base.accepts`).

The model sees the whole search every generation: an attempts table (operator,
reward, delta, accepted, mechanism, hint) and a population table, plus a running
memory of design rules that a short reflection call updates after every scored
generation (ReEvo's short- and long-term reflections in one call). Explore
operators run an idea search first: several candidate mechanisms, judged against
the library and the population, one chosen for implementation.
"""

import difflib
from pathlib import Path

import numpy as np
from tqdm import tqdm

from coding_agent import checkpoint
from coding_agent.agent import CodingAgent, Conversation
from coding_agent.executor import StrategyError, build_strategy
from coding_agent.credit import credit_feedback
from coding_agent.probes import answer_probes, parse_probe_request
from coding_agent.methods.base import (
    OuterLoopMethod,
    accepts,
    baseline_anchor,
    evaluate_strategy,
    rescore,
    paired_delta,
    reference_diff,
    summarize,
)
from coding_agent.prompts import (
    build_attempts_table,
    build_evolve_prompt,
    build_ideas_prompt,
    build_population_table,
    build_reflection_prompt,
    build_system_prompt,
    build_user_prompt,
    library_menu_for,
    mechanism_of,
    parse_ideas,
    parse_reflection,
    probe_contract,
    reflection_system,
)
from coding_agent.types import (
    GraphInfo,
    Strategy,
    TaskSpec,
    Trajectory,
    rank_by,
)

explore_operators = ("from_scratch", "crossover", "synthesize")
exploit_operators = ("refine", "parameters", "simplify")
# EoH's operator weights in spirit (its m1/m2 at 2, e1/e2/s1 at 1), with refine
# as the workhorse: the schedule below decides the explore/exploit split, these
# decide the mix inside each side
operator_weights = {
    "refine": 3.0,
    "parameters": 1.0,
    "simplify": 1.0,
    "crossover": 2.0,
    "from_scratch": 1.0,
    "synthesize": 1.0,
}
# MCTS-AHD: the exploration constant is lambda_0 x (remaining budget). At the
# first generation this is the probability mass on the explore operators; it
# decays linearly to zero at the last generation, and each stalled generation
# adds `stagnation_bonus` back
exploration_initial = 0.8
stagnation_bonus = 0.15
# Distinct mechanisms proposed before a from_scratch or synthesize generation
ideas_per_generation = 3
# Diff lines kept when the thread's assistant turn is compacted
max_compact_diff_lines = 200


def exploration_weight(iteration: int, total: int, stagnation: int) -> float:
    """Probability mass on the explore operators at this generation."""
    remaining = max(total - iteration, 0) / max(total, 1)

    return min(1.0, exploration_initial * remaining + stagnation_bonus * stagnation)


def choose_operator(
    rng: np.random.Generator,
    iteration: int,
    total: int,
    stagnation: int,
    population_size: int,
    patience: int,
) -> str:
    """One operator for this generation, weighted by the schedule."""
    explore = exploration_weight(iteration, total, stagnation)
    # The old rule survives as a floor: a long stall forces an explore move
    if patience > 0 and stagnation >= patience:
        explore = 1.0

    weights = {}
    for operator, weight in operator_weights.items():
        if operator in ("crossover", "synthesize") and population_size < 2:
            continue
        side = explore if operator in explore_operators else 1.0 - explore
        weights[operator] = weight * side
    if sum(weights.values()) <= 0.0:
        weights = {"refine": 1.0}

    names = list(weights)
    probabilities = np.asarray([weights[name] for name in names], dtype=np.float64)
    probabilities /= probabilities.sum()

    return str(rng.choice(names, p=probabilities))


def compact_turn(script: str, parent_script: str | None, verdict: str) -> str:
    """
    What the thread keeps of a generation: the mechanism line, one line of
    verdict, and a unified diff against the parent (the whole script only when
    there was no parent to diff against).
    """
    head = f"# {verdict}\n"
    if parent_script is None or not parent_script.strip():
        return f"{head}```python\n{script}\n```"

    diff = list(
        difflib.unified_diff(
            parent_script.splitlines(),
            script.splitlines(),
            fromfile="parent",
            tofile="this attempt",
            lineterm="",
            n=2,
        )
    )
    if not diff:
        return f"{head}(resubmitted the parent unchanged)"
    if len(diff) > max_compact_diff_lines:
        diff = diff[:max_compact_diff_lines] + [f"... {len(diff) - max_compact_diff_lines} more diff lines"]

    return f"{head}# MECHANISM: {mechanism_of(script)}\n```diff\n" + "\n".join(diff) + "\n```"


class EvolveSearch(OuterLoopMethod):
    def __init__(
        self,
        outer_iters: int = 20,
        strategy_mode: str = "scored",
        stagnation_patience: int = 2,
        inspiration_count: int = 2,
        label: str = "evolve",
        allow_mc_algorithms: bool = False,
        use_anchor: bool = True,
        checkpoint_path: Path | None = None,
        checkpoint_fingerprint: dict | None = None,
        credit: bool = False,
        probes: bool = True,
        reflect: bool = True,
    ) -> None:
        # ReEvo-style reflection call after every scored generation
        self.reflect = reflect
        self.outer_iters = outer_iters
        # Per-action counterfactual credit in every generation's feedback
        self.credit = credit
        # False on the native arm: its whole definition is no forward model to ask
        self.probes = probes
        # Every answered probe, read back into the results JSON by run.py
        self.probe_log = []
        self.strategy_mode = strategy_mode
        # "evolve" or "adaptive": picks the system prompt (plan_horizon vs act per
        # round) and labels the logs. The search itself is the same either way.
        self.label = label
        self.allow_mc_algorithms = allow_mc_algorithms
        # Canned arms never read a prompt, so the anchor rollouts would be pure
        # cost: five real episodes under an MC evaluator, informing nothing
        self.use_anchor = use_anchor
        self.checkpoint_path = checkpoint_path
        self.checkpoint_fingerprint = checkpoint_fingerprint
        self.stagnation_patience = stagnation_patience
        self.inspiration_count = inspiration_count
        # ReEvo's long-term reflection: design rules the evidence supports so far,
        # rewritten after every scored generation and shown in every prompt
        self.memory = ""
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
        if self.probes:
            base_user += probe_contract

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
        probe_feedback = None
        start_iteration = 0

        if resumed is not None:
            conversation.restore(resumed["messages"], resumed["transcript"])
            population = resumed["population"]
            stagnation = resumed["stagnation"]
            last_delta = resumed["last_delta"]
            self.history = resumed["history"]
            self.memory = resumed.get("memory", "")
            # .get: checkpoints from before probes existed still resume
            self.probe_log = resumed.get("probe_log", [])
            probe_feedback = resumed.get("probe_feedback")
            start_iteration = resumed["iteration"]

            if resumed["best"] is not None:
                best = (
                    build_strategy(
                        resumed["best"]["script"],
                        self.strategy_mode,
                        self.allow_mc_algorithms,
                        canned=getattr(agent.provider, "canned", False),
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
        library_menu = library_menu_for(task)
        attempts = build_attempts_table(self.history, task.sense)

        for iteration in progress_bar:
            rng = np.random.default_rng([task.seed, iteration])
            parent, partner, everyone, idea = None, None, None, ""

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
                operator = choose_operator(
                    rng,
                    iteration,
                    self.outer_iters,
                    stagnation,
                    len(population),
                    self.stagnation_patience,
                )
                ranked = rank_by(
                    population, lambda record: record["reward"], task.sense
                )
                parent = ranked[0]
                others = ranked[1:]
                inspirations = others[: self.inspiration_count]
                if operator == "crossover":
                    # Rank-weighted partner (EoH's selection): the best alternative
                    # most often, but not always
                    weights = np.asarray(
                        [1.0 / (rank + 2) for rank in range(len(others))], dtype=np.float64
                    )
                    partner = others[int(rng.choice(len(others), p=weights / weights.sum()))]
                if operator == "synthesize":
                    everyone = ranked[:3]

                # Idea search before code search, for the operators that need a
                # new mechanism rather than an edit
                if operator in ("from_scratch", "synthesize"):
                    ideas_reply = agent.provider.complete(
                        [
                            {"role": "system", "content": reflection_system},
                            {
                                "role": "user",
                                "content": build_ideas_prompt(
                                    task,
                                    ideas_per_generation,
                                    population,
                                    library_menu,
                                    self.memory,
                                    attempts,
                                ),
                            },
                        ]
                    )
                    ideas, idea = parse_ideas(ideas_reply)
                    tqdm.write(
                        f"[{self.label}] iter {iteration + 1}: {len(ideas)} ideas, "
                        f"chosen: {idea[:100] or '(none parsed)'}"
                    )

                # No base_user here: the task is already the thread's opening turn
                user = build_evolve_prompt(
                    operator,
                    parent,
                    inspirations,
                    error=last_error,
                    last_result=last_delta,
                    attempts=attempts,
                    population=build_population_table(population, task.sense),
                    memory=self.memory,
                    idea=idea,
                    partner=partner,
                    everyone=everyone,
                )

            if probe_feedback:
                user = f"{user}\n\n{probe_feedback}"
                probe_feedback = None

            tqdm.write(
                f"[{self.label}] iter {iteration + 1}/{self.outer_iters}: "
                f"operator={operator}, population={len(population)}, "
                f"explore={exploration_weight(iteration, self.outer_iters, stagnation):.2f}"
            )

            probe_request, probe_note = [], None
            # Recorded on a failure that happens before the script exists
            script = None
            parent_script = None if parent is None else parent["script"]

            try:
                script = conversation.send(user)
                # The probes fence rides the prose send() discards; the raw reply
                # is the transcript's last generate turn
                probe_request, probe_note = parse_probe_request(
                    conversation.transcript[-1]["reply"]
                )
                strategy = build_strategy(
                    script,
                    self.strategy_mode,
                    self.allow_mc_algorithms,
                    canned=getattr(agent.provider, "canned", False),
                )

                entry_point = "act() per round" if task.adaptive else "plan_horizon()"
                tqdm.write(f"[{self.label}] iter {iteration + 1}: {entry_point}...")

                # One fresh realization per generation, shared by challenger and
                # incumbent: see evaluate_strategy for why a fixed seed is a bug here
                seed = task.seed + iteration + 1
                trajectory, plan_seconds = evaluate_strategy(
                    strategy, environment, task, graph, seed=seed
                )
                best = rescore(best, environment, task, graph, seed)
            except StrategyError as error:
                last_error = str(error)
                stagnation += 1
                self.history.append(
                    {
                        "iteration": iteration + 1,
                        "reward": None,
                        "operator": operator,
                        "error": last_error,
                        "script": script,
                        "mechanism": mechanism_of(script or ""),
                        "parent_iteration": (
                            None if parent is None else parent.get("iteration")
                        ),
                    }
                )
                attempts = build_attempts_table(self.history, task.sense)
                if script is not None:
                    conversation.compact_last(
                        compact_turn(
                            script,
                            parent_script,
                            f"{operator}: FAILED, {last_error.splitlines()[0][:120]}",
                        )
                    )
                tqdm.write(
                    f"[{self.label}] iter {iteration + 1}: script failed: "
                    f"{last_error.splitlines()[0]}"
                )
                # A failed script can still have asked questions; answer them
                # against the incumbent so the next attempt gets both the error
                # and the information it wanted
                probe_feedback = answer_probes(
                    environment,
                    probe_request,
                    probe_note,
                    best[1].actions if best is not None else None,
                    task,
                    self.probes,
                    self.probe_log,
                    self.label,
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
                    probe_feedback,
                )

                continue

            last_error = None
            # The reference diff rides along in the summary, so it reaches the
            # prompt wherever a population record is shown as parent or inspiration
            diff = reference_diff(
                trajectory, anchor_trajectory, graph, anchor_name, task.sense
            )
            # Batched on the world-model env (a handful of calls for the whole
            # plan); sequential seed-paired rollouts on any other evaluator
            credit_report = (
                credit_feedback(environment, trajectory, task, graph)
                if self.credit
                else None
            )
            last_delta = (
                paired_delta(
                    trajectory,
                    best[1],
                    "the population best",
                    task.sense,
                    unit=task.reward_unit,
                )
                if best is not None
                else None
            )
            record = {
                "iteration": iteration + 1,
                "script": strategy.source_script,
                "mechanism": mechanism_of(strategy.source_script),
                "reward": trajectory.reward,
                "plan_seconds": round(plan_seconds, 3),
                "summary": summarize(trajectory, graph, task)
                + (f"\n{last_delta}" if last_delta else "")
                + (f"\n{diff}" if diff else "")
                + (f"\n{credit_report}" if credit_report else ""),
            }
            population.append(record)

            # The signed delta and the noise band of THIS comparison, both on the
            # same realization; the acceptance rule reads the band, not an epsilon
            delta = None if best is None else trajectory.reward - best[1].reward
            band = (
                0.0
                if best is None
                else max(
                    float(trajectory.cost.get("reward_se") or 0.0),
                    float(best[1].cost.get("reward_se") or 0.0),
                )
            )
            accepted = accepts(
                trajectory,
                None if best is None else best[1],
                task.sense,
                operator,
                strategy.source_script,
                "" if best is None else best[0].source_script,
            )

            # ReEvo's reflection: one short call on the (worse, better) pair,
            # returning a hint and the revised memory. Skipped for the seed.
            hint = ""
            if best is not None and self.reflect:
                incumbent_record = next(
                    (r for r in population if r["script"] == best[0].source_script),
                    None,
                )
                if incumbent_record is not None:
                    worse, better = (
                        (incumbent_record, record) if accepted else (record, incumbent_record)
                    )
                    if worse is not better:
                        reply = agent.provider.complete(
                            [
                                {"role": "system", "content": reflection_system},
                                {
                                    "role": "user",
                                    "content": build_reflection_prompt(
                                        worse, better, self.memory, task.sense, task.reward_unit
                                    ),
                                },
                            ]
                        )
                        hint, self.memory = parse_reflection(reply, self.memory)

            if accepted:
                best = (strategy, trajectory)
                stagnation = 0
            elif operator in explore_operators:
                # An explore move opens a fresh refinement window even without
                # improvement, exactly as restructure used to
                stagnation = 0
            else:
                stagnation += 1

            self.history.append(
                {
                    "iteration": iteration + 1,
                    "reward": trajectory.reward,
                    "best": best[1].reward,
                    "operator": operator,
                    "mechanism": record["mechanism"],
                    "delta": None if delta is None else round(delta, 4),
                    "band": round(band, 4),
                    "accepted": accepted,
                    "hint": hint,
                    # The script and what it edited: the closing write-up diffs
                    # them, since the thread is trimmed and cannot show old turns
                    "script": strategy.source_script,
                    "parent_iteration": (
                        None if parent is None else parent.get("iteration")
                    ),
                    "plan_seconds": round(plan_seconds, 3),
                    "rollout_seconds": round(
                        trajectory.cost.get("rollout_seconds", 0.0), 3
                    ),
                }
            )
            attempts = build_attempts_table(self.history, task.sense)
            conversation.compact_last(
                compact_turn(
                    strategy.source_script,
                    parent_script,
                    f"{operator}: reward={trajectory.reward:.3f}"
                    + ("" if delta is None else f", delta={delta:+.3f} (band {band:.3f})")
                    + f", {'ACCEPTED' if accepted else 'rejected'}"
                    + (f"; hint: {hint}" if hint else ""),
                )
            )
            tqdm.write(
                f"[{self.label}] iter {iteration + 1}: reward={trajectory.reward:.2f} "
                f"(best={best[1].reward:.2f}, {'accepted' if accepted else 'rejected'}, "
                f"stagnation={stagnation})"
                + (f" hint: {hint}" if hint else "")
            )
            progress_bar.set_postfix(
                reward=f"{trajectory.reward:.2f}", best=f"{best[1].reward:.2f}"
            )
            probe_feedback = answer_probes(
                environment,
                probe_request,
                probe_note,
                trajectory.actions,
                task,
                self.probes,
                self.probe_log,
                self.label,
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
                probe_feedback,
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
        probe_feedback: str | None = None,
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
                "memory": self.memory,
                "probe_log": self.probe_log,
                "probe_feedback": probe_feedback,
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
