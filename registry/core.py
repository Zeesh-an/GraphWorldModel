"""
The five canonical registries: tasks, algorithms, graphs, dynamics, baselines.

**This module owns no facts.** Every entry is derived from the module that
already defines it — `pipeline.tasks`, `coding_agent.tools.*`, `data.wm_graphs`,
`data.generate_wm_data`, `data.wm_simulator`, `baselines.registry`. Adding a
registry that re-declares them would create a second source of truth and
guarantee they drift; the whole point of this module is to make the EXISTING
sources enumerable and cross-checkable from one place.

What is genuinely new here is the `wm_spine_aliases` map. The world-model data
generator names its six seed selectors `random / degree / pagerank / betweenness
/ celf / local_search`, while the coding agent names the same algorithms
`random_seeds / high_degree / pagerank_seeds / betweenness_seeds / celf /
hill_climbing`. Nothing in the repository related the two vocabularies, so
"the world model trains on trajectories from algorithms the coding agent can
emit" was unverifiable — and a rename on either side would have broken the
correspondence silently. See `registry.consistency.validate_registry_consistency`.
"""

from dataclasses import dataclass, field

from data.generate_wm_data import synthetic_families
from data.wm_actions import spine_algorithms
from data.wm_graphs import real_directed
from data.wm_simulator import valid_action_ops, valid_remove_semantics
from pipeline.tasks import Task, tasks

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

#: The task registry IS `pipeline.tasks.tasks`, re-exported under the name the
#: experiment tooling uses. It is not a copy: `pipeline.tasks` validates action
#: ops, remove semantics and budget ops at import time, and `Layout` keys the
#: results tree off the same dict.
TASK_REGISTRY: dict[str, Task] = tasks


def implemented_tasks() -> list[str]:
    """Tasks the pipeline can actually run today, sorted."""
    return sorted(name for name, task in TASK_REGISTRY.items() if task.runnable)


# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------

#: What an algorithm is FOR, which decides who may reference it.
planner_tool = "planner_tool"  # the coding agent may call/emit it
wm_spine = "wm_spine"  # the world-model data generator rolls it out


@dataclass(frozen=True)
class AlgorithmEntry:
    name: str
    #: `planner_tool` or `wm_spine`; an algorithm reachable both ways carries
    #: both, which is exactly the overlap `validate_registry_consistency` checks.
    roles: tuple[str, ...]
    #: Which registered tasks this algorithm is a legal move for.
    tasks: tuple[str, ...]
    #: Module-qualified home, so `inspect_registry` can point at the definition
    #: rather than making the reader grep for it.
    source: str
    summary: str = ""
    #: Set on a `wm_spine` entry whose planner-side name differs. Not cosmetic:
    #: without it the two vocabularies cannot be compared at all.
    planner_alias: str | None = None


#: Spine name -> coding-agent name.
#:
#: Verified against the implementations, not guessed. `wm_actions.select_seeds`
#: computes `local_search` as "start from the degree heuristic, then 1-swaps,
#: keep the swap if estimated spread improves"; `coding_agent.tools.algorithms`
#: documents `hill_climbing` as "Local search: start from degree, 1-swap while
#: spread improves". `celf` is spelled identically on both sides.
#:
#: `celf_local_search` is deliberately NOT the alias for `local_search`: it
#: refines a CELF seed set, whereas the spine refines a DEGREE seed set.
wm_spine_aliases: dict[str, str] = {
    "random": "random_seeds",
    "degree": "high_degree",
    "pagerank": "pagerank_seeds",
    "betweenness": "betweenness_seeds",
    "celf": "celf",
    "local_search": "hill_climbing",
}


def _coding_agent_pools() -> dict[str, tuple[dict, tuple[str, ...]]]:
    """
    Pool name -> (the dict of callables, the tasks it serves).

    Imported inside a function because `coding_agent.tools.*` pulls networkx and
    the primitive library; a registry import should stay cheap enough for a test
    or a Slurm preflight to pay it unconditionally.
    """
    from coding_agent.tools.adaptive_algorithms import adaptive_algorithms
    from coding_agent.tools.algorithms import algorithms
    from coding_agent.tools.dismantling_algorithms import dismantling_algorithms

    return {
        "coding_agent.tools.algorithms": (
            algorithms,
            ("influence_maximization", "adaptive_online_im"),
        ),
        "coding_agent.tools.dismantling_algorithms": (
            dismantling_algorithms,
            ("critical_node_detection", "influence_blocking"),
        ),
        "coding_agent.tools.adaptive_algorithms": (
            adaptive_algorithms,
            ("adaptive_online_im",),
        ),
    }


def _build_algorithm_registry() -> dict[str, AlgorithmEntry]:
    import inspect

    entries: dict[str, AlgorithmEntry] = {}

    for source, (pool, pool_tasks) in _coding_agent_pools().items():
        for name, function in pool.items():
            summary = (inspect.getdoc(function) or "").split("\n")[0]

            if name in entries:
                # One name in two pools (e.g. an IM classic reused as an
                # adaptive control): widen its task set rather than letting the
                # later pool overwrite the earlier one.
                previous = entries[name]
                entries[name] = AlgorithmEntry(
                    name=name,
                    roles=previous.roles,
                    tasks=tuple(sorted(set(previous.tasks) | set(pool_tasks))),
                    source=f"{previous.source}, {source}",
                    summary=previous.summary or summary,
                )
                continue

            entries[name] = AlgorithmEntry(
                name=name,
                roles=(planner_tool,),
                tasks=pool_tasks,
                source=source,
                summary=summary,
            )

    # Fold the world-model spine in on top, so an algorithm reachable from both
    # sides is ONE entry carrying both roles instead of two near-duplicates.
    for spine_name in spine_algorithms:
        alias = wm_spine_aliases.get(spine_name)
        target = entries.get(alias) if alias else None

        if target is not None:
            entries[alias] = AlgorithmEntry(
                name=target.name,
                roles=tuple(sorted(set(target.roles) | {wm_spine})),
                tasks=target.tasks,
                source=f"{target.source}, data.wm_actions.spine_algorithms",
                summary=target.summary,
                planner_alias=None,
            )
            continue

        # An unaliased spine algorithm is a real gap, not an error: record it so
        # `validate_registry_consistency` can report it instead of hiding it.
        entries[spine_name] = AlgorithmEntry(
            name=spine_name,
            roles=(wm_spine,),
            tasks=("influence_maximization",),
            source="data.wm_actions.spine_algorithms",
            summary="world-model spine selector with no planner-side counterpart",
            planner_alias=alias,
        )

    return dict(sorted(entries.items()))


ALGORITHM_REGISTRY: dict[str, AlgorithmEntry] = _build_algorithm_registry()


def algorithms_for_task(task: str) -> list[str]:
    return sorted(
        name for name, entry in ALGORITHM_REGISTRY.items() if task in entry.tasks
    )


def algorithms_with_role(role: str) -> list[str]:
    return sorted(
        name for name, entry in ALGORITHM_REGISTRY.items() if role in entry.roles
    )


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------

synthetic = "synthetic"
real = "real"


@dataclass(frozen=True)
class GraphEntry:
    name: str
    kind: str  # synthetic | real
    #: Synthetic families are generated at an arbitrary N (`--syn-nodes`), so
    #: size is a run parameter, not a property. A real dataset's size is fixed by
    #: the file, which is why only real entries can carry one.
    directed: bool | None = None
    source: str = ""


def _build_graph_registry() -> dict[str, GraphEntry]:
    entries = {
        name: GraphEntry(
            name=name,
            kind=synthetic,
            source="data.generate_wm_data.synthetic_families",
        )
        for name in synthetic_families
    }

    entries.update(
        {
            name: GraphEntry(
                name=name,
                kind=real,
                directed=bool(is_directed),
                source="data.wm_graphs.real_directed",
            )
            for name, is_directed in real_directed.items()
        }
    )

    return dict(sorted(entries.items()))


GRAPH_REGISTRY: dict[str, GraphEntry] = _build_graph_registry()


def graphs_of_kind(kind: str) -> list[str]:
    return sorted(name for name, entry in GRAPH_REGISTRY.items() if entry.kind == kind)


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DynamicsEntry:
    name: str
    #: True when `data.wm_simulator.Simulator` can actually step it today. The
    #: registry lists planned dynamics too (SIR/SEIR, motter_lai, ...) because
    #: `pipeline.tasks` already declares them, and silently dropping them would
    #: make a planned task look like it had no dynamics at all.
    simulated: bool
    tasks: tuple[str, ...]
    summary: str = ""


#: The two NDlib models `data.wm_simulator.Simulator.reset` actually builds.
simulated_dynamics = ("IC", "LT")

_dynamics_summaries = {
    "IC": "Independent Cascade: each active u gets one attempt at each "
    "out-neighbour v with probability w(u,v).",
    "LT": "Linear Threshold: v activates when the active fraction of its "
    "in-neighbour weight crosses v's hidden, per-episode threshold.",
}


def _build_dynamics_registry() -> dict[str, DynamicsEntry]:
    by_name: dict[str, set[str]] = {}

    for name, task in TASK_REGISTRY.items():
        for dynamics in task.dynamics:
            by_name.setdefault(dynamics, set()).add(name)

    return {
        dynamics: DynamicsEntry(
            name=dynamics,
            simulated=dynamics in simulated_dynamics,
            tasks=tuple(sorted(task_names)),
            summary=_dynamics_summaries.get(dynamics, ""),
        )
        for dynamics, task_names in sorted(by_name.items())
    }


DYNAMICS_REGISTRY: dict[str, DynamicsEntry] = _build_dynamics_registry()


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineEntry:
    name: str
    kind: str  # classical | learned
    task: str
    status: str  # ready | needs_setup | blocked
    venue: str = ""
    blocker: str | None = None


def _build_baseline_registry() -> dict[str, BaselineEntry]:
    from baselines.registry import external_baselines

    return {
        name: BaselineEntry(
            name=name,
            kind=baseline.kind,
            task=baseline.task,
            status=baseline.status,
            venue=getattr(baseline, "venue", ""),
            blocker=getattr(baseline, "blocker", None),
        )
        for name, baseline in sorted(external_baselines.items())
    }


BASELINE_REGISTRY: dict[str, BaselineEntry] = _build_baseline_registry()


def baselines_for_task(task: str) -> list[str]:
    return sorted(
        name for name, entry in BASELINE_REGISTRY.items() if entry.task == task
    )


def runnable_baselines() -> list[str]:
    return sorted(
        name for name, entry in BASELINE_REGISTRY.items() if entry.status == "ready"
    )


# ---------------------------------------------------------------------------
# Actions and evaluators
# ---------------------------------------------------------------------------

#: The world model's action vocabulary, straight from the simulator that has to
#: execute each op. Listed here so `validate_registry_consistency` can hold every
#: task's declared ops against it from one place.
ACTION_REGISTRY: tuple[str, ...] = tuple(valid_action_ops)

REMOVE_SEMANTICS: tuple[str, ...] = tuple(valid_remove_semantics)


def evaluator_modes() -> tuple[str, ...]:
    """Evaluators an arm may name — `pipeline.conditions` owns the list."""
    from pipeline.conditions import valid_evaluators

    return tuple(valid_evaluators)


def backbone_names() -> tuple[str, ...]:
    """
    World-model encoder backbones.

    Imported lazily: `world_model.wm_model` pulls torch, and `inspect_registry`
    is the only caller that needs it. Keeping it out of module import means a
    registry consistency test does not depend on a torch install.
    """
    from world_model.wm_model import backbones

    return tuple(backbones)


def head_names() -> tuple[str, ...]:
    """Output heads `world_model.wm_model.WorldModel` accepts."""
    return ("linear", "structured", "structured_residual", "structured_oracle")
