"""
The machine-readable manifest: one JSON snapshot of every registry.

Written by `python -m scripts.inspect_registry`. Two consumers:

  * a Slurm job, which records the manifest digest alongside its metrics so a
    results table can be traced back to the vocabulary it was produced under
    (§26). A rename that changes what `high_degree` means changes the digest.

  * `tests/test_registry.py`, which asserts the counts it reports are the counts
    the registries actually hold — so no README, report or slide can quote a
    number that the code does not produce.

Nothing in here is hand-maintained. Every count is `len()` of a live registry.
"""

import hashlib
import json
from dataclasses import asdict

from registry.consistency import validate_registry_consistency
from registry.core import (
    ACTION_REGISTRY,
    ALGORITHM_REGISTRY,
    BASELINE_REGISTRY,
    DYNAMICS_REGISTRY,
    GRAPH_REGISTRY,
    REMOVE_SEMANTICS,
    TASK_REGISTRY,
    evaluator_modes,
    head_names,
    implemented_tasks,
    planner_tool,
    real,
    synthetic,
    wm_spine,
    wm_spine_aliases,
)

manifest_version = "registry-manifest-v1"


def _task_dict(name: str) -> dict:
    task = TASK_REGISTRY[name]

    return {
        "name": task.name,
        "title": task.title,
        "status": task.status,
        "objective": task.objective,
        "dynamics": list(task.dynamics),
        "action_ops": list(task.action_ops),
        "allowed_ops": list(task.allowed_ops),
        "gen_action_ops": list(task.gen_action_ops),
        "remove_semantics": task.remove_semantics,
        "budget_op": task.budget_op,
        "runnable": task.runnable,
        "research_doc": task.research_doc,
        "blocker": task.blocker,
    }


def build_manifest(include_backbones: bool = True) -> dict:
    """
    Snapshot every registry plus the consistency verdict.

    `include_backbones` is a switch because reading them imports torch. A CI
    preflight that only needs the vocabulary can leave it off.
    """
    report = validate_registry_consistency()

    manifest = {
        "manifest_version": manifest_version,
        "tasks": {name: _task_dict(name) for name in sorted(TASK_REGISTRY)},
        "algorithms": {
            name: asdict(entry) for name, entry in ALGORITHM_REGISTRY.items()
        },
        "graphs": {name: asdict(entry) for name, entry in GRAPH_REGISTRY.items()},
        "dynamics": {name: asdict(entry) for name, entry in DYNAMICS_REGISTRY.items()},
        "baselines": {name: asdict(entry) for name, entry in BASELINE_REGISTRY.items()},
        "action_ops": list(ACTION_REGISTRY),
        "remove_semantics": list(REMOVE_SEMANTICS),
        "evaluator_modes": list(evaluator_modes()),
        "world_model_heads": list(head_names()),
        "wm_spine_aliases": dict(wm_spine_aliases),
        "counts": counts(),
        "consistency": report.to_dict(),
    }

    if include_backbones:
        from registry.core import backbone_names

        manifest["world_model_backbones"] = list(backbone_names())

    manifest["digest"] = digest(manifest)

    return manifest


def counts() -> dict:
    """
    Every count the project may quote, derived. Nothing here is typed in.

    If a document wants to say "N algorithms", it must read N from this dict.
    """
    return {
        "tasks_total": len(TASK_REGISTRY),
        "tasks_implemented": len(implemented_tasks()),
        "algorithms_total": len(ALGORITHM_REGISTRY),
        "algorithms_planner_tool": sum(
            planner_tool in entry.roles for entry in ALGORITHM_REGISTRY.values()
        ),
        "algorithms_wm_spine": sum(
            wm_spine in entry.roles for entry in ALGORITHM_REGISTRY.values()
        ),
        "graphs_total": len(GRAPH_REGISTRY),
        "graphs_synthetic": sum(
            entry.kind == synthetic for entry in GRAPH_REGISTRY.values()
        ),
        "graphs_real": sum(entry.kind == real for entry in GRAPH_REGISTRY.values()),
        "dynamics_total": len(DYNAMICS_REGISTRY),
        "dynamics_simulated": sum(
            entry.simulated for entry in DYNAMICS_REGISTRY.values()
        ),
        "baselines_total": len(BASELINE_REGISTRY),
        "baselines_ready": sum(
            entry.status == "ready" for entry in BASELINE_REGISTRY.values()
        ),
        "action_ops": len(ACTION_REGISTRY),
    }


def digest(manifest: dict) -> str:
    """
    Stable hash of the vocabulary, excluding the digest field itself.

    The consistency block is excluded too: a warning count is a property of the
    checks, not of the vocabulary, and letting it move the digest would make
    every job look like it ran under a different registry.
    """
    body = {
        key: value
        for key, value in manifest.items()
        if key not in ("digest", "consistency")
    }

    return hashlib.sha256(
        json.dumps(body, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
