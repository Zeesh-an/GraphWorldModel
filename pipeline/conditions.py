"""
The baseline taxonomy: which conditions exist, how an arm spec names one, and how
to read a finished run back.

Two orthogonal axes — who designs the algorithm, and what feedback the designer
gets while designing:

| # | condition                 | designer          | inner-loop feedback        |
| - | ------------------------- | ----------------- | -------------------------- |
| 1 | Pure GA                   | fixed algorithm   | none                       |
| 2 | GA routing                | LLM picks a pool member | none (one selection call) |
| 3 | Native coding agent       | LLM synthesises   | real executions only       |
| 4 | Agent + MC simulation     | LLM synthesises   | averaged simulator rollouts |
| 5 | Agent + oracle dynamics   | LLM synthesises   | true transition dynamics   |
| 6 | Ours: agent + learned GWM | LLM synthesises   | learned f_theta rollouts   |

Arm spec grammar (what `--arms` / `--baselines` accept):

    baseline:<algorithm>[@<evaluator>]     condition 1
    routing[@<evaluator>]                  condition 2
    <method>_<mode>@<evaluator>            conditions 3-6

`<evaluator>` is `native`, `monte_carlo`, `oracle`, or `world_model`; omitting it
falls back to the pipeline-level `--evaluator`. `native` is the model-free
condition: the real simulator, but only `--native-mc-runs` episode(s) per
candidate, so the agent pays real experience for every noisy number it gets back.
"""

from dataclasses import dataclass

from coding_agent.tools.adaptive_algorithms import adaptive_algorithms

native = "native"
monte_carlo = "monte_carlo"
oracle = "oracle"
world_model = "world_model"

valid_evaluators = (native, monte_carlo, oracle, world_model)
# `adaptive` runs the same population search as `evolve` and differs in one
# thing: the program is a per-round policy called on the realized state instead
# of a static plan decided up front. Pairing `adaptive_<mode>@E` with
# `evolve_<mode>@E` at the same budget is what makes the adaptivity gap an A/B
# on that one variable (research/adaptive_online_im.md §9.3 item 3).
valid_methods = ("one_shot", "per_step", "windowed", "evolve", "adaptive")
adaptive_method = "adaptive"
# The non-adaptive counterpart an adaptive arm is divided by
non_adaptive_method = "evolve"
valid_modes = ("free", "scored")

pure_ga_condition = 1
routing_condition = 2
evaluator_conditions = {native: 3, monte_carlo: 4, oracle: 5, world_model: 6}
external_condition = 7

condition_names = {
    1: "Pure GA",
    2: "GA routing",
    3: "Native coding agent",
    4: "Agent + MC simulation",
    5: "Agent + oracle dynamics",
    6: "Ours: agent + learned GWM",
    7: "Published baseline (external repo)",
}

# Conditions 1 and 2 have no refinement loop, so their evaluator only decides how
# the single result is scored — ground truth is the honest choice
selection_evaluator = monte_carlo

# The classical pool run as condition 1: one representative per major IM family
# (heuristic, discount, centrality, greedy/CELF, RIS/sketch) plus the random floor
default_baselines = (
    "high_degree",
    "degree_discount",
    "pagerank_seeds",
    "celf_pp",
    "imm",
    "random_seeds",
)

# Conditions 2-6. The method is held fixed across 3-6 so the only thing that
# varies down that ladder is the inner-loop evaluator — the clean ablation.
#
# evolve, not one_shot: both refine a program against the same feedback, but
# evolve edits the POPULATION BEST each generation while one_shot edits the
# latest attempt, so one_shot compounds a regression instead of rejecting it.
# Same LLM calls, same evaluator, strictly better search — swap back to
# one_shot_free@* with --arms if you want the ablation.
default_arms = (
    "routing",
    "evolve_free@native",
    "evolve_free@monte_carlo",
    "evolve_free@oracle",
    "evolve_free@world_model",
)


@dataclass
class Arm:
    spec: str  # exactly what the user typed
    name: str  # filesystem-safe; becomes <budget>/<name>.json
    method: str
    strategy_mode: str
    evaluator: str
    condition: int
    baseline: str | None = None
    routing: bool = False
    external: str | None = None  # registered name in baselines/registry.py

    @property
    def condition_name(self) -> str:
        return condition_names[self.condition]

    @property
    def is_agent(self) -> bool:
        """True when an LLM actually synthesises code (conditions 3-6)."""
        return self.baseline is None and not self.routing and self.external is None


def parse_arm(spec: str, default_evaluator: str | None = None) -> Arm:
    body, separator, explicit = spec.partition("@")

    if separator and explicit not in valid_evaluators:
        raise ValueError(
            f"arm {spec!r} names evaluator {explicit!r}; choose one of {valid_evaluators}"
        )

    # An external published method is scored on ground truth like the other
    # no-refinement conditions; only its SEED SET crosses the process boundary
    if body.startswith("external:"):
        name = body.split(":", 1)[1]
        if not name:
            raise ValueError(f"arm {spec!r} is missing a name after 'external:'")

        evaluator = explicit or selection_evaluator

        return Arm(
            spec=spec,
            name=f"external_{name}",
            method="one_shot",
            strategy_mode="free",
            evaluator=evaluator,
            condition=external_condition,
            external=name,
        )

    # Conditions 1 and 2 have no refinement loop, so they default to ground truth
    # rather than to the pipeline evaluator the synthesis arms use
    if body.startswith("baseline:") or body == "routing":
        evaluator = explicit or selection_evaluator
        suffix = f"@{evaluator}" if explicit else ""

        if body == "routing":
            return Arm(
                spec=spec,
                name=f"routing{suffix}",
                method="one_shot",
                strategy_mode="free",
                evaluator=evaluator,
                condition=routing_condition,
                routing=True,
            )

        algorithm = body.split(":", 1)[1]
        if not algorithm:
            raise ValueError(f"arm {spec!r} is missing an algorithm after 'baseline:'")

        # A published ADAPTIVE algorithm (AdaptGreedy, EPIC, ...) is a per-round
        # policy, not a static seed set, so it runs down the round path like any
        # other adaptive arm. Recorded as method="adaptive" so adaptivity_gaps
        # divides it by a static baseline rather than treating it as one.
        return Arm(
            spec=spec,
            name=f"baseline_{algorithm}{suffix}",
            method=(
                adaptive_method if algorithm in adaptive_algorithms else "one_shot"
            ),
            strategy_mode="free",
            evaluator=evaluator,
            condition=pure_ga_condition,
            baseline=algorithm,
        )

    evaluator = explicit or default_evaluator
    if evaluator is None:
        raise ValueError(
            f"arm {spec!r} has no @<evaluator> and no pipeline default was supplied"
        )

    method, _, mode = body.rpartition("_")
    if method not in valid_methods or mode not in valid_modes:
        raise ValueError(
            f"unknown arm {spec!r}; expected 'baseline:<algorithm>', 'routing', or "
            f"'<method>_<mode>[@<evaluator>]' with method in {valid_methods} and "
            f"mode in {valid_modes}"
        )

    return Arm(
        spec=spec,
        name=f"{body}@{evaluator}",
        method=method,
        strategy_mode=mode,
        evaluator=evaluator,
        condition=evaluator_conditions[evaluator],
    )


def resolve_evaluator(arm: Arm, mc_runs: int, native_mc_runs: int) -> tuple[str, int]:
    """Map an arm's evaluator onto the (evaluator, mc_runs) the runner understands."""
    if arm.evaluator == native:
        return monte_carlo, native_mc_runs

    return arm.evaluator, mc_runs


def needs_world_model(arms: list[Arm]) -> bool:
    return any(arm.evaluator == world_model for arm in arms)


def ground_truth_reward(result: dict) -> float:
    """
    The only number comparable ACROSS conditions.

    Each condition's `reward` is measured by its own evaluator — a native arm's is
    one noisy episode, ours is a world-model estimate — so they cannot be plotted
    against each other. `mc_reward` is the shared ground-truth referee written by
    --compare; fall back to `reward` only when that replay was not run.
    """
    mc_reward = result.get("mc_reward")

    return float(mc_reward if mc_reward is not None else result["reward"])


def is_ground_truth(results: list[dict]) -> bool:
    return bool(results) and all(
        result.get("mc_reward") is not None for result in results
    )


def adaptivity_gaps(results: list[dict]) -> list[dict]:
    """
    Pair every adaptive arm with its matched non-adaptive control.

    gap = sigma(adaptive policy) / sigma(best static seed set) at the SAME budget
    and the SAME evaluator (research/adaptive_online_im.md §8.1). Both sides are
    read on the ground-truth MC replay, because each arm's own reward is measured
    by its own evaluator and a ratio of two different rulers means nothing.

    Calibration, so a small number is not misread as a failure: theory caps the
    myopic gap at 4 and proves non-adaptive greedy is no worse than adaptive
    greedy across all graphs (§5.1). A gap near 1 is the expected outcome; the
    claim this task makes is about COST, not spread.

    The denominator is the BEST non-adaptive arm at that (budget, evaluator), not
    a nominated one: §1.1 defines the gap against `max_{|S|=k} E[sigma(S)]`, so
    the closest available estimate is the strongest static seed set anyone
    produced under the same measurement conditions. `control_arm` records which
    it was, since that changes with the arm set.
    """
    controls = {}

    for result in results:
        if result.get("method") == adaptive_method:
            continue

        key = (result.get("budget"), result.get("evaluator"))
        best = controls.get(key)
        if best is None or ground_truth_reward(result) > ground_truth_reward(best):
            controls[key] = result

    paired = []

    for result in results:
        if result.get("method") != adaptive_method:
            continue

        control = controls.get((result.get("budget"), result.get("evaluator")))
        if control is None:
            continue

        adaptive_spread = ground_truth_reward(result)
        control_spread = ground_truth_reward(control)
        paired.append(
            {
                "budget": result.get("budget"),
                "budget_label": result.get("budget_label"),
                "evaluator": result.get("evaluator"),
                "rounds": result.get("rounds"),
                "round_batches": result.get("round_batches"),
                "feedback_model": result.get("feedback_model"),
                "adaptive_arm": result.get("arm"),
                "control_arm": control.get("arm"),
                "adaptive_spread": adaptive_spread,
                "control_spread": control_spread,
                "gap": (
                    adaptive_spread / control_spread if control_spread else None
                ),
                # The cost axis the task is actually about: how much the two arms
                # spent inside their evaluators to reach those spreads
                "adaptive_evaluator_seconds": result.get("evaluator_seconds"),
                "control_evaluator_seconds": control.get("evaluator_seconds"),
                "adaptive_real_episodes": result.get("real_env_episodes"),
                "control_real_episodes": control.get("real_env_episodes"),
            }
        )

    return sorted(paired, key=lambda entry: (entry["budget"] or 0, entry["evaluator"]))
