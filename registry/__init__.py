"""
Canonical registries shared by the world model and the coding agent.

One import for everything the experiment tooling needs to know about what
exists:

    from registry import (
        TASK_REGISTRY, ALGORITHM_REGISTRY, GRAPH_REGISTRY,
        DYNAMICS_REGISTRY, BASELINE_REGISTRY,
        ExperimentGroup, enumerate_groups,
        validate_registry_consistency,
    )

The registries are VIEWS over the modules that already own these facts, not
copies of them — see `registry.core`. The one thing defined here and nowhere
else is `wm_spine_aliases`, which relates the world model's algorithm names to
the coding agent's.

`python -m scripts.inspect_registry` prints all of it and writes a JSON manifest.
"""

from registry.consistency import (
    ConsistencyReport,
    Finding,
    require_consistent,
    validate_experiment_config,
    validate_registry_consistency,
)
from registry.core import (
    ACTION_REGISTRY,
    ALGORITHM_REGISTRY,
    BASELINE_REGISTRY,
    DYNAMICS_REGISTRY,
    GRAPH_REGISTRY,
    REMOVE_SEMANTICS,
    TASK_REGISTRY,
    AlgorithmEntry,
    BaselineEntry,
    DynamicsEntry,
    GraphEntry,
    algorithms_for_task,
    algorithms_with_role,
    backbone_names,
    baselines_for_task,
    evaluator_modes,
    graphs_of_kind,
    head_names,
    implemented_tasks,
    planner_tool,
    real,
    runnable_baselines,
    synthetic,
    wm_spine,
    wm_spine_aliases,
)
from registry.group import (
    ExperimentGroup,
    UnknownGroupField,
    algorithm_of_arm,
    enumerate_groups,
    explain_group_count,
    split_group_key,
    split_groups,
)
from registry.manifest import build_manifest, counts, manifest_version

__all__ = [
    # registries
    "TASK_REGISTRY",
    "ALGORITHM_REGISTRY",
    "GRAPH_REGISTRY",
    "DYNAMICS_REGISTRY",
    "BASELINE_REGISTRY",
    "ACTION_REGISTRY",
    "REMOVE_SEMANTICS",
    # entry types
    "AlgorithmEntry",
    "BaselineEntry",
    "DynamicsEntry",
    "GraphEntry",
    # lookups
    "algorithms_for_task",
    "algorithms_with_role",
    "backbone_names",
    "baselines_for_task",
    "evaluator_modes",
    "graphs_of_kind",
    "head_names",
    "implemented_tasks",
    "runnable_baselines",
    "planner_tool",
    "wm_spine",
    "wm_spine_aliases",
    "synthetic",
    "real",
    # experiment groups
    "ExperimentGroup",
    "UnknownGroupField",
    "enumerate_groups",
    "explain_group_count",
    "algorithm_of_arm",
    "split_group_key",
    "split_groups",
    # consistency
    "ConsistencyReport",
    "Finding",
    "validate_registry_consistency",
    "validate_experiment_config",
    "require_consistent",
    # manifest
    "build_manifest",
    "counts",
    "manifest_version",
]
