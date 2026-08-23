"""
Emit the source -> target transfer matrix, human- and machine-readable.

    python -m scripts.transfer_matrix
    python -m scripts.transfer_matrix --out results/task_transfer/manifest.json

Every cell is derived by `registry.task_families` from properties
`pipeline.tasks` owns. Result cells are `NOT_RUN` until a real experiment writes
them: an estimate in this table would be indistinguishable from a measurement.
"""

import argparse
import json
from pathlib import Path

from pipeline.tasks import tasks
from registry.task_families import (
    containment_overlay,
    families,
    state_layout,
    transfer_matrix,
    world_family,
)

default_out = Path("results") / "task_transfer" / "manifest.json"

eight_tasks = [
    "influence_maximization",
    "adaptive_online_im",
    "source_localization",
    "cascade_reconstruction",
    "cascade_prediction",
    "critical_node_detection",
    "influence_blocking",
    "epidemic_control",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Task transfer matrix")
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument("--source", type=str, default=None,
                        help="print only rows with this source task")
    args = parser.parse_args()

    matrix = transfer_matrix(eight_tasks)

    print("WORLD FAMILIES")
    for family, members in sorted(families().items()):
        members = [m for m in members if m in eight_tasks]
        if not members:
            continue
        layout = state_layout(tasks[members[0]])
        print(f"  {family}  {layout[0]} in / {layout[1]} out")
        for name in members:
            runnable = "runnable" if tasks[name].runnable else "not ported here"
            print(f"      {name:26s} ({runnable})")

    overlay = [t for t in containment_overlay() if t in eight_tasks]
    print(f"\nCONTAINMENT OVERLAY (action semantics only, spans "
          f"{len({world_family(tasks[t]) for t in overlay})} world families)")
    print(f"  {', '.join(overlay)}")

    print(f"\n{'source':26s} {'target':26s} {'level':18s} why")
    print("-" * 118)
    for cell in matrix:
        if args.source and cell.source != args.source:
            continue
        print(f"{cell.source:26s} {cell.target:26s} {cell.level:18s} "
              f"{cell.reason.split('.')[0][:60]}")

    family_members = {
        family: [m for m in members if m in eight_tasks]
        for family, members in sorted(families().items())
    }
    manifest = {
        "world_families": {
            family: {
                "layout_in_out": list(state_layout(tasks[members[0]])),
                "tasks": members,
            }
            for family, members in family_members.items()
            if members
        },
        "containment_overlay": overlay,
        "transfers": [
            {
                **cell.to_dict(),
                # Never an estimate. A populated cell means a run produced it.
                "prediction": "NOT_RUN",
                "action_fidelity": (
                    "NOT_APPLICABLE" if not cell.requires_actions else "NOT_RUN"
                ),
                "task_utility": "NOT_RUN",
                "transfer_gap": "NOT_RUN",
                "oracle_gap": "NOT_RUN",
                "decision_retention": "NOT_RUN",
                "target_runnable_here": tasks[cell.target].runnable,
            }
            for cell in matrix
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nmanifest -> {args.out}  ({len(matrix)} ordered pairs)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
