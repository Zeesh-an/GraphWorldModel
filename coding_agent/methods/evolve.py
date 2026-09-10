"""
Method 4: population evolution with a real operator set.

Every generation applies ONE operator to the population: refine, parameters or
simplify (exploit: an edit of the population best), or crossover, synthesize or
from_scratch (explore: a new mechanism). The operator is drawn with weights that
follow MCTS-AHD's schedule, an exploration term that scales with the REMAINING
budget, so restructuring happens early and refinement late, with stagnation
adding exploration back. A candidate replaces the incumbent only when its paired
delta clears the standard error of the PAIRED difference (`methods.base.accepts`,
`search_metrics.paired_band`).

The model sees the whole search every generation: an attempts table (operator,
reward, delta, accepted, mechanism, hint, its own forecast) and a population
table ranked by the paired estimate, plus a running memory of design rules that
a short reflection call updates after every scored generation (ReEvo's short-
and long-term reflections in one call). Explore operators run an idea search
first: several candidate mechanisms, judged against the library and the
population, one chosen for implementation.

Five things the stochastic evaluator adds, none of which a deterministic-fitness
loop needs (search_metrics.py, counterexamples.py, probes.py):

- a PROBE TURN before every generation: a trace-free aside in which the model
  asks what-if questions about the incumbent's plan and gets the answers in the
  same generation's prompt;
- the parent is the INCUMBENT, never the member with the luckiest realization,
  and the rest of the population is ranked by the paired estimate;
- every generation records what the naive `delta > 0` rule would have decided
  (the spurious-accept ledger) and the incumbent's re-score on the fresh
  realization beside its accepted score (the unbiased curve);
- every candidate carries the model's forecast of its own paired delta, scored
  after the fact (edit calibration), which can steer the operator choice;
- when a candidate loses, the realizations it lost most on are described in the
  task's terms and shown as counterexamples.
"""

import difflib
from pathlib import Path
import numpy as np
from tqdm import tqdm

from coding_agent import checkpoint
from coding_agent.agent import CodingAgent, Conversation
from coding_agent.counterexamples import describe_counterexamples, sample_sets
from coding_agent.credit import credit_feedback
from coding_agent.diagnostics import PlanDiagnostics
from coding_agent.executor import StrategyError, build_strategy
from coding_agent.feedback import build_feedback
from coding_agent.feedback import resolve as resolve_feedback
from coding_agent.methods.base import (
    OuterLoopMethod,
    _communities,
    accepts,
    baseline_anchor,
    evaluate_strategy,
    paired_delta,
    reference_diff,
    rescore,
    summarize,
)
from coding_agent.probes import answer_probes, parse_probe_request
from coding_agent.prompts import (
    build_attempts_table,
    build_evolve_prompt,
    build_ideas_prompt,
    build_population_table,
    build_probe_turn_prompt,
    build_reflection_prompt,
    build_system_prompt,
    build_user_prompt,
    library_menu_for,
    mechanism_of,
    parse_ideas,
    parse_reflection,
    probe_contract_for,
    reflection_system,
)
from coding_agent.search_metrics import (
    acceptance_ledger,
    calibration_metrics,
    expected_of,
    flat_generations,
    miscalibrated,
    optimism_summary,
    paired_band,
    paired_strengths,
)
from coding_agent.types import (
    GraphInfo,
    Strategy,
    TaskSpec,
    Trajectory,
    improves,
    rank_by,
)

# The other three operators (refine, parameters, simplify) exploit the population best
explore_operators = ("from_scratch", "crossover", "synthesize")
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
    miscalibrated_now: bool = False,
) -> str:
    """
    One operator for this generation, weighted by the schedule.

    `miscalibrated_now` (with `--calibration-steering`) counts as one more
    stalled generation: the model's recent forecasts of its own edits were wrong
    in sign, so its local picture of the program is stale and an explore move is
    worth more than another refine.
    """
    explore = exploration_weight(
        iteration, total, stagnation + (1 if miscalibrated_now else 0)
    )
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
        feedback: str | None = None,
        probe_turn: bool = True,
        stop_when_flat: int = 0,
        calibration_steering: bool = False,
    ) -> None:
        # ReEvo-style reflection call after every scored generation
        self.reflect = reflect
        # The trace-free probe aside before each generation (one extra LLM call
        # per generation; answered immediately, about the incumbent's plan)
        self.probe_turn = probe_turn
        # Stop early once the UNBIASED incumbent estimate has not improved beyond
        # the band for this many consecutive generations; 0 runs every generation
        self.stop_when_flat = stop_when_flat
        # Let a run of wrong-signed forecasts add exploration mass
        self.calibration_steering = calibration_steering
        # Filled at the end of optimize(): the ledger, the calibration report and
        # the optimism summary, read back into the results JSON by run.py
        self.ledger = {}
        self.calibration = {}
        self.optimism = {}
        self.stopped_early = None
        self.outer_iters = outer_iters
        # Per-action counterfactual credit in every generation's feedback
        self.credit = credit
        # Which feedback tier each generation gets (coding_agent/feedback.py).
        # `legacy` (the default) is the repository's existing summarize() output
        # and leaves every current run byte-identical; f0-f3 are the controlled
        # ladder Experiment 3 varies.
        self.feedback = resolve_feedback(feedback)
        # False on the native arm: its whole definition is no forward model to
        # ask, and on the lower ladder rungs, which are defined by NOT having the
        # richer probes available
        self.probes = probes and self.feedback.probes_allowed
        # Per-generation diagnostic blocks, read back into the results JSON
        self.diagnostic_log = []
        # Regional coverage per generation, so the stagnation block can say WHICH
        # part of the graph has stayed uncovered rather than only that the reward
        # stopped moving
        self.coverage_history = []
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
            base_user += probe_contract_for(task)

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
                    self.calibration_steering and miscalibrated(self.history),
                )
                # The parent is the INCUMBENT: the program the paired acceptance
                # rule has kept. Each member's own reward was measured on its own
                # generation's realization, so the raw ranking put a rejected
                # candidate that drew a generous realization above the incumbent
                # and refine then edited the loser (64 percent of generations on
                # the four local runs that recorded it, with acceptance falling
                # from 33 to 3 percent). The rest are ranked by the paired
                # estimate, one scale for every member.
                strengths = paired_strengths(population)
                for record in population:
                    record["strength"] = strengths[int(record["iteration"])]
                ranked = rank_by(
                    population, lambda record: record["strength"], task.sense
                )
                incumbent_record = (
                    next(
                        (r for r in population if r["script"] == best[0].source_script),
                        None,
                    )
                    if best is not None
                    else None
                )
                parent = incumbent_record if incumbent_record is not None else ranked[0]
                others = [record for record in ranked if record is not parent]
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

            # The probe turn: ask about the incumbent's plan and get the answers
            # before writing, so a question has a payoff for THIS edit. A
            # trace-free aside, so the thread carries neither the question nor
            # the answer as a turn.
            if self.probes and self.probe_turn and best is not None and parent is not None:
                probe_reply = conversation.aside(
                    build_probe_turn_prompt(parent, task, operator), kind="probe"
                )
                turn_request, turn_note = parse_probe_request(probe_reply)
                turn_feedback = answer_probes(
                    environment,
                    turn_request,
                    turn_note,
                    best[1],
                    task,
                    graph,
                    self.probes,
                    self.probe_log,
                    self.label,
                    turn="probe",
                    about="the incumbent's plan you are about to edit",
                )
                if turn_feedback:
                    user = f"{user}\n\n{turn_feedback}"

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
                candidate_sets = sample_sets(environment)
                # The incumbent's accepted score, frozen at its own generation,
                # against its re-score on this fresh realization: the first is
                # a maximum over noisy estimates, the second is not
                incumbent_reported = None if best is None else best[1].reward
                best = rescore(best, environment, task, graph, seed)
                incumbent_sets = sample_sets(environment) if best is not None else None
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
                    best[1] if best is not None else None,
                    task,
                    graph,
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
            # The realizations this candidate lost most on, in the task's terms;
            # informs the next edit, never the estimate (the next generation is
            # scored on its own fresh realization)
            counterexamples = (
                describe_counterexamples(
                    trajectory,
                    best[1],
                    candidate_sets,
                    incumbent_sets,
                    graph,
                    task,
                    communities=_communities(graph) if not task.recovers and not task.decodes and not task.forecasts else None,
                )
                if best is not None
                else None
            )
            # LEGACY takes the existing path untouched; a ladder tier takes the
            # counted diagnostic blocks and NOTHING else, so its feedback content
            # is exactly what the tier names
            if self.feedback.is_legacy:
                summary = (
                    summarize(trajectory, graph, task)
                    + (f"\n{last_delta}" if last_delta else "")
                    + (f"\n{counterexamples}" if counterexamples else "")
                    + (f"\n{diff}" if diff else "")
                    + (f"\n{credit_report}" if credit_report else "")
                )
            else:
                diagnostics = PlanDiagnostics(
                    environment,
                    graph,
                    horizon=task.horizon,
                    budget=task.budget,
                    seed=seed,
                    budget_op=task.budget_op,
                    # Containment and blocking MINIMIZE, and the contribution
                    # signs have to follow or the feedback coaches the agent to
                    # undo its own improvements
                    sense=task.sense,
                )
                summary, diagnosis = build_feedback(
                    self.feedback,
                    diagnostics,
                    trajectory.actions,
                    history=[
                        entry.get("reward")
                        for entry in self.history
                        if entry.get("reward") is not None
                    ],
                    sense=task.sense,
                    coverage_history=list(self.coverage_history),
                )

                if diagnosis.coverage:
                    self.coverage_history.append(diagnosis.coverage)
                self.diagnostic_log.append(
                    {"iteration": iteration + 1, "tier": self.feedback.tier}
                    | diagnosis.to_dict()
                )

            # The signed delta and the noise band of THIS comparison, both on the
            # same realization; the acceptance rule reads the band, not an epsilon
            delta = None if best is None else trajectory.reward - best[1].reward
            band, band_rule = (0.0, "none") if best is None else paired_band(trajectory, best[1])
            predicted_delta, predicted_clear_p = expected_of(strategy.source_script)
            incumbent_iteration = (
                None
                if best is None
                else next(
                    (r["iteration"] for r in population if r["script"] == best[0].source_script),
                    None,
                )
            )

            record = {
                "iteration": iteration + 1,
                "script": strategy.source_script,
                "mechanism": mechanism_of(strategy.source_script),
                "reward": trajectory.reward,
                "plan_seconds": round(plan_seconds, 3),
                "summary": summary,
                # The paired comparison this member was scored by: who it was
                # compared with, on one realization, and by how much
                "compared_to": incumbent_iteration,
                "paired_delta": None if delta is None else round(delta, 6),
                "band": round(band, 6),
            }
            population.append(record)

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

            # What the naive rule would have decided, for the ledger: a positive
            # delta in the task's sense, band or no band
            naive_accepted = (
                True
                if best is None
                else improves(trajectory.reward, best[1].reward, task.sense, 0.0)
            )
            incumbent_unbiased = None if best is None else best[1].reward

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
                    "band_rule": band_rule,
                    "accepted": accepted,
                    "naive_accepted": naive_accepted,
                    # The incumbent entering this generation: its accepted score
                    # and its re-score on this generation's realization
                    "compared_to": incumbent_iteration,
                    "incumbent_reported": incumbent_reported,
                    "incumbent_unbiased": incumbent_unbiased,
                    # The model's own forecast for this edit
                    "predicted_delta": predicted_delta,
                    "predicted_clear_p": predicted_clear_p,
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
                trajectory,
                task,
                graph,
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

            if self.stop_when_flat > 0:
                flat = flat_generations(self.history, task.sense)
                if flat >= self.stop_when_flat:
                    self.stopped_early = iteration + 1
                    tqdm.write(
                        f"[{self.label}] stopping at generation {iteration + 1}: the "
                        f"unbiased incumbent estimate has been flat for {flat} generations"
                    )
                    break

        progress_bar.close()

        if best is None:
            raise StrategyError(
                f"all {self.outer_iters} attempts failed; last error: {last_error}"
            )

        self.ledger = acceptance_ledger(self.history)
        self.calibration = calibration_metrics(self.history)
        self.optimism = optimism_summary(self.history, task.sense)

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
