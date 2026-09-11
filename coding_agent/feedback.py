"""
Feedback tiers: what the designer is told after each candidate is scored.

Experiment 3 asks whether richer world-model feedback changes what a reviser can
do, so "richer" has to be a controlled variable rather than a mood. This module
is that variable — five named tiers, each an explicit set of diagnostic blocks,
with the cost of each block stated:

| tier     | blocks                                                   | extra WM rollouts per turn |
| -------- | -------------------------------------------------------- | -------------------------- |
| `f0`     | scalar reward                                            | 0 (the rollout already ran) |
| `f1`     | + per-seed drop attribution                              | k                          |
| `f2`     | + regional coverage                                      | k                          |
| `f3`     | + seed overlap, bridge coverage, stagnation              | 2k                         |
| `default` | the repository's full `summarize()` feedback, untouched | 0                        |

`default` is what every run gets unless a tier is named. It is its own tier
rather than an alias for one of the others because it is not a point on this
ladder: it already carries community reach and adjacent-seed hints
(so it is richer than `f2` in places), and it also carries reverse-reachable
residual gains, which are computed from `graph.ic_probs` — the TRUE per-edge
transmission probabilities. That is fine as a graph-structure prior in the
existing design, but it means `default` is not a clean "world-model feedback
only" arm, and the f0-f3 ladder is. Anything comparing feedback content should
run on the ladder; `default` is the full feedback every other experiment uses.

The tiers deliberately do NOT change the evaluator, the candidate generator, the
budget, the horizon or the number of outer iterations. Only what comes back.
"""

from dataclasses import dataclass

#: Scalar reward only: `expected spread = X`. The floor of the ladder.
f0 = "f0"
#: + per-seed counterfactual contribution (leave-one-out, paired).
f1 = "f1"
#: + regional coverage over a graph partition.
f2 = "f2"
#: + seed overlap, bridge coverage and the stagnation verdict.
f3 = "f3"
#: The repository's full feedback (summary, paired verdict, reference diff, credit, counterexamples). The default.
default = "default"

valid_tiers = (default, f0, f1, f2, f3)

#: Which `PlanDiagnostics.explain_plan` blocks each tier requests.
tier_blocks = {
    f0: ("value",),
    f1: ("value", "drop"),
    f2: ("value", "drop", "region"),
    f3: ("value", "drop", "region", "redundancy", "bridge", "stagnation"),
}


@dataclass(frozen=True)
class FeedbackPolicy:
    """
    One tier, resolved into the switches the loop actually reads.

    `structural_hints` is what separates `default` from the ladder: the existing
    `summarize()` extras (unreached-node lists, residual gains, adjacent seed
    pairs, community reach). On the ladder they are off, so every line of
    feedback is either the reward or a counted world-model probe.
    """

    tier: str = default

    def __post_init__(self) -> None:
        if self.tier not in valid_tiers:
            raise ValueError(
                f"unknown feedback tier {self.tier!r}; choose from {list(valid_tiers)}"
            )

    @property
    def is_default(self) -> bool:
        return self.tier == default

    @property
    def blocks(self) -> tuple:
        return tier_blocks.get(self.tier, ())

    @property
    def structural_hints(self) -> bool:
        """Whether the pre-existing `summarize()` extras are included."""
        return self.is_default

    @property
    def probes_allowed(self) -> bool:
        """Agent-initiated `drop` / `swap` / `region` probes. Full tier only, so
        the ladder's lower rungs cannot buy their way back up."""
        return self.tier in (default, f3)

    def wants(self, block: str) -> bool:
        return block in self.blocks

    def describe(self) -> str:
        if self.is_default:
            return "default: the repository's full summarize() feedback"

        return f"{self.tier}: blocks {list(self.blocks)}"


def resolve(tier: str | FeedbackPolicy | None) -> FeedbackPolicy:
    if isinstance(tier, FeedbackPolicy):
        return tier

    return FeedbackPolicy(default if tier is None else str(tier))


def build_feedback(
    policy: FeedbackPolicy,
    diagnostics,
    plan: list,
    history: list | None = None,
    sense: str = "maximize",
    coverage_history: list | None = None,
) -> tuple[str, object]:
    """
    (text, Diagnosis) for one ladder tier.

    `default` is NOT handled here: it is the `methods.base.summarize()` path and
    stays where it is, so nothing about the default behaviour routes through the
    ladder's code. Callers branch on `policy.is_default` and this raises if they
    do not, rather than silently returning an impoverished summary for a default
    run.
    """
    if policy.is_default:
        raise ValueError(
            "the default tier is produced by methods.base.summarize(); branch on "
            "policy.is_default rather than calling build_feedback"
        )

    diagnosis = diagnostics.explain_plan(
        plan,
        history=history,
        sense=sense,
        blocks=policy.blocks,
        coverage_history=coverage_history,
    )

    return diagnosis.text, diagnosis


__all__ = [
    "FeedbackPolicy",
    "build_feedback",
    "f0",
    "f1",
    "f2",
    "f3",
    "default",
    "resolve",
    "tier_blocks",
    "valid_tiers",
]
