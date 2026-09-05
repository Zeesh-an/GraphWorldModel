"""
Print every registry and write the JSON manifest.

    python -m scripts.inspect_registry
    python -m scripts.inspect_registry --out results/registry_manifest.json
    python -m scripts.inspect_registry --config configs/wm_main.yaml
    python -m scripts.inspect_registry --strict          # exit 1 on any error

The point is that no document has to quote a count by hand. Every number printed
here is `len()` of a live registry, so a table that disagrees with this output is
wrong about the code rather than the other way round.
"""

import argparse
import json
import os
from pathlib import Path

from registry import (
    ALGORITHM_REGISTRY,
    BASELINE_REGISTRY,
    DYNAMICS_REGISTRY,
    GRAPH_REGISTRY,
    TASK_REGISTRY,
    build_manifest,
    enumerate_groups,
    evaluator_modes,
    explain_group_count,
    planner_tool,
    real,
    synthetic,
    validate_experiment_config,
    wm_spine,
)

default_out = Path("results") / "registry_manifest.json"


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def _print_tasks() -> None:
    _rule(f"TASKS ({len(TASK_REGISTRY)})")

    for name in sorted(TASK_REGISTRY):
        task = TASK_REGISTRY[name]
        marker = "*" if task.runnable else " "
        print(
            f" {marker} {name:26s} {task.status:12s} obj={str(task.objective):9s} "
            f"dyn={','.join(task.dynamics) or '-':12s} rm={task.remove_semantics}"
        )

    print("\n   * = implemented (runnable by pipeline/run.py today)")


def _print_algorithms() -> None:
    planner = [e for e in ALGORITHM_REGISTRY.values() if planner_tool in e.roles]
    spine = [e for e in ALGORITHM_REGISTRY.values() if wm_spine in e.roles]
    _rule(
        f"ALGORITHMS ({len(ALGORITHM_REGISTRY)} total; "
        f"{len(planner)} emittable by the coding agent; "
        f"{len(spine)} used for world-model training trajectories)"
    )

    by_task = {}

    for name, entry in ALGORITHM_REGISTRY.items():
        for task in entry.tasks:
            by_task.setdefault(task, []).append(name)

    for task in sorted(by_task):
        names = sorted(by_task[task])
        print(f"  {task} ({len(names)}):")
        print(f"    {', '.join(names)}")

    print("\n  world-model spine coverage (spine name -> coding-agent name):")

    for entry in sorted(spine, key=lambda e: e.name):
        print(f"    {entry.name}")


def _print_graphs() -> None:
    syn = [e for e in GRAPH_REGISTRY.values() if e.kind == synthetic]
    rea = [e for e in GRAPH_REGISTRY.values() if e.kind == real]
    _rule(f"GRAPHS ({len(GRAPH_REGISTRY)}: {len(syn)} synthetic, {len(rea)} real)")
    print(f"  synthetic: {', '.join(sorted(e.name for e in syn))}")
    print(f"  real:      {', '.join(sorted(e.name for e in rea))}")


def _print_dynamics() -> None:
    _rule(f"DYNAMICS ({len(DYNAMICS_REGISTRY)})")

    for name, entry in DYNAMICS_REGISTRY.items():
        state = "simulated" if entry.simulated else "declared only"
        print(f"  {name:16s} {state:15s} tasks={len(entry.tasks)}")


def _print_baselines() -> None:
    _rule(f"EXTERNAL BASELINES ({len(BASELINE_REGISTRY)})")
    by_task = {}

    for entry in BASELINE_REGISTRY.values():
        by_task.setdefault(entry.task, []).append(entry)

    for task in sorted(by_task):
        entries = by_task[task]
        ready = sum(e.wired and e.status != "blocked" for e in entries)
        print(f"  {task}: {len(entries)} ({ready} wired)")
        print(f"    {', '.join(sorted(e.name for e in entries))}")


def _print_evaluators() -> None:
    _rule("EVALUATOR MODES")
    print(f"  {', '.join(evaluator_modes())}")


def _print_consistency(manifest: dict) -> None:
    consistency = manifest["consistency"]
    _rule(
        f"CONSISTENCY: {'OK' if consistency['ok'] else 'FAILED'} "
        f"({consistency['n_errors']} errors, {consistency['n_warnings']} warnings)"
    )

    for finding in consistency["findings"]:
        print(f"  [{finding['severity'].upper():7s}] {finding['check']}")
        print(f"      {finding['message']}")

    if not consistency["findings"]:
        print("  no findings")


def _print_config(path: Path) -> int:
    import yaml

    config = yaml.safe_load(path.read_text())
    _rule(f"EXPERIMENT CONFIG: {path}")
    report = validate_experiment_config(config)

    for finding in report.findings:
        print(f"  {finding}")

    try:
        groups = enumerate_groups(config)
    except Exception as exception:  # noqa: BLE001 - reported, not raised
        print(f"  [ERROR  ] group enumeration failed: {exception}")
        return 1

    print()
    print(explain_group_count(config))
    print("\n  first groups: ")

    for group in groups[:5]:
        print(f"    {group.run_id}")

    if len(groups) > 5:
        print(f"    ... and {len(groups) - 5} more")

    return 1 if report.errors else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print the canonical registries and write a JSON manifest"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help=f"manifest path (default: {default_out}).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="also validate an experiment config and explain its group count (default: None).",
    )
    parser.add_argument(
        "--no-backbones",
        action="store_true",
        help="skip reading world-model backbones (avoids importing torch) (default: False).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if any consistency check reports an error (default: False).",
    )
    parser.add_argument("--quiet", action="store_true", help="write the manifest only (default: False).")
    args = parser.parse_args()

    manifest = build_manifest(include_backbones=not args.no_backbones)

    if not args.quiet:
        _print_tasks()
        _print_algorithms()
        _print_graphs()
        _print_dynamics()
        _print_baselines()
        _print_evaluators()

        if "world_model_backbones" in manifest:
            _rule("WORLD MODEL")
            print(f"  backbones: {', '.join(manifest['world_model_backbones'])}")
            print(f"  heads:     {', '.join(manifest['world_model_heads'])}")

        _rule("COUNTS (every number the project may quote)")

        for key, value in manifest["counts"].items():
            print(f"  {key:32s} {value}")

        _print_consistency(manifest)

    os.makedirs(args.out.parent, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, default=str))

    if not args.quiet:
        print(f"\nmanifest -> {args.out}  (digest {manifest['digest']})")

    exit_code = 0

    if args.config is not None:
        exit_code |= _print_config(args.config)

    if args.strict and manifest["consistency"]["n_errors"]:
        exit_code |= 1

    raise SystemExit(exit_code)
