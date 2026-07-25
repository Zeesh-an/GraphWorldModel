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

native = "native"
monte_carlo = "monte_carlo"
oracle = "oracle"
world_model = "world_model"

valid_evaluators = (native, monte_carlo, oracle, world_model)
valid_methods = ("one_shot", "per_step", "windowed", "evolve")
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
# varies down that ladder is the inner-loop evaluator — the clean ablation
default_arms = (
    "routing",
    "one_shot_free@native",
    "one_shot_free@monte_carlo",
    "one_shot_free@oracle",
    "one_shot_free@world_model",
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

        return Arm(
            spec=spec,
            name=f"baseline_{algorithm}{suffix}",
            method="one_shot",
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
