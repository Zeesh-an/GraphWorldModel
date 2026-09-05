"""
Cross-registry validation: does the coding agent's vocabulary match the world
model's, and does every config refer to things that exist?

The failure this is built to catch is not a crash — it is a run that completes
and reports a number that means something other than what the table says. Three
concrete instances already in the repository:

  * the world-model data generator rolls out six seed selectors named
    `random / degree / pagerank / ...`; the coding agent emits `random_seeds /
    high_degree / pagerank_seeds / ...`. Same algorithms, different spelling, no
    check. "The world model was trained on trajectories from algorithms the agent
    can propose" was an assumption, not a verified property.

  * `baselines.registry` already carries a `task` field with a comment saying a
    blocking or CND baseline must not join an IM sweep — but nothing enforced it.

  * a task's `default_baselines` tuple is a list of bare strings. A rename in
    `coding_agent.tools.*` leaves it pointing at nothing until the sweep runs.

Findings are returned, not raised, so `inspect_registry` can print all of them at
once. `require_consistent()` is the fail-fast wrapper for a preflight.
"""

from dataclasses import dataclass, field

from registry.core import (
    ACTION_REGISTRY,
    ALGORITHM_REGISTRY,
    BASELINE_REGISTRY,
    DYNAMICS_REGISTRY,
    GRAPH_REGISTRY,
    REMOVE_SEMANTICS,
    TASK_REGISTRY,
    planner_tool,
    wm_spine,
    wm_spine_aliases,
)

error = "error"
warning = "warning"


@dataclass(frozen=True)
class Finding:
    severity: str  # error | warning
    check: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.upper():7s}] {self.check}: {self.message}"


@dataclass()
class ConsistencyReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == error]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == warning]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, severity: str, check: str, message: str) -> None:
        self.findings.append(Finding(severity, check, message))

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "findings": [
                {"severity": f.severity, "check": f.check, "message": f.message}
                for f in self.findings
            ],
        }


def validate_registry_consistency() -> ConsistencyReport:
    """Run every cross-registry check and return all findings."""
    report = ConsistencyReport()

    _check_spine_alignment(report)
    _check_task_action_ops(report)
    _check_task_dynamics(report)
    _check_task_default_baselines(report)
    _check_external_baseline_tasks(report)
    _check_arms_parse(report)

    return report


def require_consistent() -> ConsistencyReport:
    """Raise on any error-level finding. For a preflight before a sweep."""
    report = validate_registry_consistency()

    if not report.ok:
        raise ValueError(
            "registry consistency check failed:\n"
            + "\n".join(str(f) for f in report.errors)
        )

    return report


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_spine_alignment(report: ConsistencyReport) -> None:
    """Every world-model spine selector must resolve to a planner algorithm."""
    from data.wm_actions import spine_algorithms

    for name in spine_algorithms:
        alias = wm_spine_aliases.get(name)

        if alias is None:
            report.add(
                error,
                "spine_alignment",
                f"world-model spine algorithm {name!r} has no entry in "
                f"registry.core.wm_spine_aliases, so it cannot be related to any "
                f"algorithm the coding agent emits",
            )
            continue

        if alias not in ALGORITHM_REGISTRY:
            report.add(
                error,
                "spine_alignment",
                f"spine algorithm {name!r} is aliased to {alias!r}, which is not "
                f"in any coding_agent tool pool",
            )
            continue

        entry = ALGORITHM_REGISTRY[alias]

        if planner_tool not in entry.roles:
            report.add(
                error,
                "spine_alignment",
                f"spine algorithm {name!r} resolves to {alias!r}, which the "
                f"coding agent cannot emit (roles={entry.roles})",
            )

    # The reverse direction is a warning, not an error: the agent SHOULD have a
    # wider vocabulary than the six selectors used to harvest training
    # trajectories. It is reported because that gap is the honest scope of any
    # "the world model has seen this kind of algorithm" claim.
    spine_covered = {
        name for name, entry in ALGORITHM_REGISTRY.items() if wm_spine in entry.roles
    }
    im_algorithms = {
        name
        for name, entry in ALGORITHM_REGISTRY.items()
        if "influence_maximization" in entry.tasks and planner_tool in entry.roles
    }
    uncovered = sorted(im_algorithms - spine_covered)

    if uncovered:
        report.add(
            warning,
            "spine_coverage",
            f"{len(uncovered)}/{len(im_algorithms)} influence-maximization "
            f"algorithms the coding agent can emit are NOT represented in the "
            f"world model's training trajectories "
            f"(spine covers {sorted(spine_covered)}); "
            f"off-policy fidelity for the rest is untested",
        )


def _check_task_action_ops(report: ConsistencyReport) -> None:
    """A task may only declare ops the simulator can execute."""
    for name, task in TASK_REGISTRY.items():
        unknown = sorted(set(task.action_ops) - set(ACTION_REGISTRY))

        if unknown:
            report.add(
                error,
                "task_action_ops",
                f"task {name!r} declares ops {unknown} that "
                f"data.wm_simulator cannot execute",
            )

        if task.remove_semantics not in REMOVE_SEMANTICS:
            report.add(
                error,
                "task_remove_semantics",
                f"task {name!r} declares remove_semantics "
                f"{task.remove_semantics!r}; valid: {list(REMOVE_SEMANTICS)}",
            )

        unknown_allowed = sorted(set(task.allowed_ops) - set(ACTION_REGISTRY))

        if unknown_allowed:
            report.add(
                error,
                "task_allowed_ops",
                f"task {name!r} lets the planner emit {unknown_allowed}, which "
                f"the simulator cannot execute",
            )


def _check_task_dynamics(report: ConsistencyReport) -> None:
    """Every runnable task must declare at least one SIMULATED dynamics."""
    for name, task in TASK_REGISTRY.items():
        for dynamics in task.dynamics:
            if dynamics not in DYNAMICS_REGISTRY:
                report.add(
                    error,
                    "task_dynamics",
                    f"task {name!r} declares dynamics {dynamics!r} missing from "
                    f"DYNAMICS_REGISTRY",
                )

        if not task.runnable:
            continue

        simulated = [d for d in task.dynamics if DYNAMICS_REGISTRY[d].simulated]

        if not simulated:
            report.add(
                error,
                "task_dynamics",
                f"task {name!r} is marked implemented but none of its dynamics "
                f"{list(task.dynamics)} can be stepped by data.wm_simulator",
            )


def _check_task_default_baselines(report: ConsistencyReport) -> None:
    """
    A task's baseline pool must name algorithms that exist and serve that task.

    This is the check §4 asks for by name: "no stale experiment config refers to
    missing algorithms". A task's `default_baselines` IS a stale-able config.
    """
    from coding_agent.tools.blocking_algorithms import default_blocking_baselines
    from coding_agent.tools.immunization_algorithms import (
        default_immunization_baselines,
    )
    from pipeline.conditions import default_baselines as fallback_baselines

    for name, task in TASK_REGISTRY.items():
        # Same resolution as `pipeline.run.resolve_baselines`: a lever task's
        # pool is per lever, and every lever's pool is a stale-able config.
        if task.competitive:
            pool = {a for pool in default_blocking_baselines.values() for a in pool}
        elif task.epidemic:
            pool = {
                a for pool in default_immunization_baselines.values() for a in pool
            }
        else:
            pool = task.default_baselines or (
                fallback_baselines if task.runnable else ()
            )

        for algorithm in sorted(pool):
            entry = ALGORITHM_REGISTRY.get(algorithm)

            if entry is None:
                report.add(
                    error,
                    "task_baselines",
                    f"task {name!r} lists baseline {algorithm!r}, which is in no "
                    f"coding_agent tool pool — a rename left it dangling",
                )
                continue

            if name not in entry.tasks:
                report.add(
                    warning,
                    "task_baselines",
                    f"task {name!r} lists baseline {algorithm!r}, whose registry "
                    f"entry serves {list(entry.tasks)}; either the pool mapping "
                    f"in registry.core._coding_agent_pools is too narrow or this "
                    f"baseline is solving a different problem",
                )


def _check_external_baseline_tasks(report: ConsistencyReport) -> None:
    """An external baseline must name a registered, runnable task."""
    from baselines.discovery import any_task

    for name, baseline in BASELINE_REGISTRY.items():
        # A condition-9 discovery system serves every task by construction: its
        # problem is written per run from the task's own contract
        if baseline.task == any_task:
            continue

        task = TASK_REGISTRY.get(baseline.task)

        if task is None:
            report.add(
                error,
                "external_baselines",
                f"external baseline {name!r} declares task {baseline.task!r}, "
                f"which is not registered",
            )
            continue

        if not task.runnable:
            report.add(
                warning,
                "external_baselines",
                f"external baseline {name!r} targets task {baseline.task!r}, "
                f"which is {task.status} — it cannot be swept yet",
            )


def _check_arms_parse(report: ConsistencyReport) -> None:
    """Every task's default arms must parse under the current arm grammar."""
    from pipeline.conditions import default_arms, parse_arm

    for name, task in TASK_REGISTRY.items():
        for spec in task.default_arms or default_arms:
            arm_spec = spec.spec if hasattr(spec, "spec") else spec

            try:
                parse_arm(arm_spec)
            except Exception as exception:  # noqa: BLE001 - reported, not raised
                report.add(
                    error,
                    "task_arms",
                    f"task {name!r} declares arm {arm_spec!r} which "
                    f"pipeline.conditions.parse_arm rejects: {exception}",
                )


# ---------------------------------------------------------------------------
# Experiment configs
# ---------------------------------------------------------------------------


def validate_experiment_config(config: dict) -> ConsistencyReport:
    """
    Check one experiment config (§25 YAML) against the registries.

    Separate from `validate_registry_consistency` because the registries can be
    perfectly self-consistent while a config asks for a graph that does not
    exist. `enumerate_groups` raises on the structural errors; this adds the
    checks that should warn rather than stop a dry run.
    """
    report = ConsistencyReport()

    from pipeline.conditions import parse_arm

    for spec in config.get("arms", []):
        try:
            parse_arm(spec)
        except Exception as exception:  # noqa: BLE001
            report.add(
                error, "config_arms", f"arm {spec!r} does not parse: {exception}"
            )

    for task in config.get("tasks", []):
        entry = TASK_REGISTRY.get(task)

        if entry is None:
            report.add(error, "config_tasks", f"unknown task {task!r}")
            continue

        if not entry.runnable:
            report.add(
                error,
                "config_tasks",
                f"task {task!r} is {entry.status}: {entry.blocker or 'no blocker recorded'}",
            )

        for graph in config.get("datasets", {}).get(task, []):
            if graph not in GRAPH_REGISTRY:
                report.add(error, "config_datasets", f"unknown graph {graph!r}")

    seeds = config.get("seeds", [])

    if len(seeds) < 5:
        report.add(
            warning,
            "config_seeds",
            f"{len(seeds)} seed(s) configured; §13 asks for >= 5 before any "
            f"claim that one arm beats another",
        )

    return report
