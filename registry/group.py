"""
`ExperimentGroup` — the one canonical unit of experimental work.

Why this exists: "group" was being used for two unrelated things, and neither was
written down. A count like "480 groups" then cannot be checked, reproduced, or
explained, and it silently changes meaning when the sweep changes.

The two meanings, now separated and both derived rather than typed in:

  `ExperimentGroup`   one job the launcher runs and one row the aggregator
                      reports: (task, graph, dynamics, arm, seed). This is the
                      canonical one; `enumerate_groups` is the ONLY way to obtain
                      a count, and `explain_group_count` prints the factorization
                      that produced it.

  `split_group_key`   the leakage-control bucket a world-model transition belongs
                      to when the dataset is split into train/val/test. NOT an
                      experiment: it never becomes a job. It is a function rather
                      than a second dataclass precisely so it cannot be mistaken
                      for one.

Both are computed. There is no `NUM_GROUPS` constant anywhere, and adding one
would defeat the purpose of this module.
"""

from dataclasses import asdict, dataclass

from registry.core import (
    ALGORITHM_REGISTRY,
    DYNAMICS_REGISTRY,
    GRAPH_REGISTRY,
    TASK_REGISTRY,
)


@dataclass(frozen=True, order=True)
class ExperimentGroup:
    """
    One (task, graph, dynamics, arm, seed) cell of the experiment matrix.

    `arm` rather than `algorithm` because that is the dimension this repository
    actually sweeps: `pipeline.conditions.parse_arm` turns a spec like
    `evolve_free@world_model` or `external:imm` into the thing that runs, and a
    published baseline is an arm exactly as a coding-agent condition is. An
    `algorithm` field alongside it would be empty for five of the six conditions.

    Frozen and ordered so a set of these deduplicates, sorts stably, and can key
    a resume index without a separate id.
    """

    task: str
    graph: str
    dynamics: str
    arm: str
    seed: int

    @property
    def run_id(self) -> str:
        """Filesystem-safe id — one run directory per group, per §26."""
        arm = self.arm.replace("@", "-at-").replace(":", "-")
        return f"{self.task}__{self.graph}__{self.dynamics}__{arm}__seed{self.seed}"

    def to_dict(self) -> dict:
        return asdict(self)


class UnknownGroupField(ValueError):
    """A config named something no registry knows about."""


def _validated(
    field: str, values, registry_keys, allow: tuple = ()
) -> list:
    unknown = [v for v in values if v not in registry_keys and v not in allow]

    if unknown:
        raise UnknownGroupField(
            f"experiment config lists unknown {field}: {sorted(unknown)}; "
            f"registered {field} are {sorted(registry_keys)}"
        )

    return list(values)


def enumerate_groups(config: dict) -> list[ExperimentGroup]:
    """
    Expand an experiment config into the exact set of groups it describes.

    `config` is the parsed YAML of §25, e.g.

        tasks:     [influence_maximization]
        datasets:  {influence_maximization: [ba, ws]}
        dynamics:  {influence_maximization: [IC, LT]}
        arms:      [evolve_free@world_model, evolve_free@oracle]
        seeds:     [0, 1, 2, 3, 4]

    `datasets` and `dynamics` are per-task because a task's legal dynamics are a
    property of the task (`Task.dynamics`), not of the sweep — running LT on a
    task whose registry entry declares only IC is a config bug, and this is where
    it should fail rather than three hours into a Slurm array.

    Arms are NOT validated against a registry here: `parse_arm` owns arm syntax
    and accepts open-ended `external:<name>` specs. `validate_experiment_config`
    in `registry.consistency` runs `parse_arm` over them for that check.
    """
    tasks = _validated("tasks", config.get("tasks", []), TASK_REGISTRY.keys())
    arms = list(config.get("arms", []))
    seeds = [int(s) for s in config.get("seeds", [])]

    if not (tasks and arms and seeds):
        raise UnknownGroupField(
            "experiment config must name at least one task, one arm and one seed; "
            f"got tasks={tasks} arms={arms} seeds={seeds}"
        )

    datasets_by_task = config.get("datasets", {})
    dynamics_by_task = config.get("dynamics", {})
    groups: list[ExperimentGroup] = []

    for task in tasks:
        graphs = _validated(
            f"datasets[{task}]", datasets_by_task.get(task, []), GRAPH_REGISTRY.keys()
        )
        dynamics = _validated(
            f"dynamics[{task}]", dynamics_by_task.get(task, []), DYNAMICS_REGISTRY.keys()
        )

        if not graphs or not dynamics:
            raise UnknownGroupField(
                f"task {task!r} needs at least one dataset and one dynamics; "
                f"got datasets={graphs} dynamics={dynamics}"
            )

        # A dynamics the task does not declare is a silent scientific error: the
        # data generator would happily produce it and the results table would
        # report it under a task whose research doc never claimed it.
        declared = set(TASK_REGISTRY[task].dynamics)
        undeclared = sorted(set(dynamics) - declared)

        if undeclared:
            raise UnknownGroupField(
                f"task {task!r} declares dynamics {sorted(declared)} in "
                f"pipeline.tasks, but the config asks for {undeclared}"
            )

        groups.extend(
            ExperimentGroup(task=task, graph=graph, dynamics=dyn, arm=arm, seed=seed)
            for graph in graphs
            for dyn in dynamics
            for arm in arms
            for seed in seeds
        )

    # Sorted + deduplicated: a config that lists a seed twice must not launch the
    # same run twice, and the launcher's resume index keys off this order.
    return sorted(set(groups))


def explain_group_count(config: dict) -> str:
    """
    Human-readable factorization of `len(enumerate_groups(config))`.

    §3 asks the system to be able to say WHY an experiment contains N groups.
    This is that answer, and it is printed by `scripts.inspect_registry` and by
    `--dry-run`.
    """
    groups = enumerate_groups(config)
    arms = list(config.get("arms", []))
    seeds = [int(s) for s in config.get("seeds", [])]
    lines = [f"{len(groups)} experiment groups"]

    for task in _validated("tasks", config.get("tasks", []), TASK_REGISTRY.keys()):
        graphs = config.get("datasets", {}).get(task, [])
        dynamics = config.get("dynamics", {}).get(task, [])
        subtotal = len(graphs) * len(dynamics) * len(arms) * len(seeds)
        lines.append(
            f"  {task}: {len(graphs)} graphs x {len(dynamics)} dynamics x "
            f"{len(arms)} arms x {len(seeds)} seeds = {subtotal}"
        )

    if len(groups) != sum(
        len(config.get("datasets", {}).get(t, []))
        * len(config.get("dynamics", {}).get(t, []))
        * len(arms)
        * len(seeds)
        for t in config.get("tasks", [])
    ):
        lines.append(
            "  (total is below the product because duplicate cells were removed)"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dataset split buckets — a different thing, deliberately not a dataclass
# ---------------------------------------------------------------------------


def split_group_key(record: dict) -> str:
    """
    The bucket a world-model transition must not be split across.

    A transition is identified by `graph_id` and the episode it came from. Two
    transitions from the SAME GRAPH share its structure, its edge probabilities
    and (for the main branch) its cascade, so putting one in train and another in
    test leaks. `data.generate_wm_data._assign_split` currently draws a split per
    EPISODE, which means the same graph routinely lands on both sides.

    This function states the correct bucket. It is not yet wired into the
    generator — doing that changes what every existing checkpoint was trained on,
    so it belongs in a phase with a retrain, not in a registry commit. The
    corresponding audit finding is recorded in `docs/audit_phase1.md`.
    """
    return str(record.get("graph_id", "unknown"))


def split_groups(records: list[dict]) -> set[str]:
    """Distinct leakage buckets present in a set of transition records."""
    return {split_group_key(record) for record in records}


def algorithm_of_arm(arm: str) -> str | None:
    """
    The registered algorithm an arm runs, when it names one.

    `external:imm` -> `imm`; a coding-agent condition such as
    `evolve_free@world_model` runs generated code and returns None. Used by the
    aggregator to join a results row back onto `ALGORITHM_REGISTRY`.
    """
    body = arm.partition("@")[0]

    if not body.startswith("external:"):
        return None

    name = body.split(":", 1)[1]

    return name if name in ALGORITHM_REGISTRY else None
