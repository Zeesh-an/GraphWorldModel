"""
One-off: move pre-task results trees into results/<task>/<dataset>/<run>/.

The old layout was results/<tag>/, where <tag> defaulted to the dataset name and
was often suffixed with a SLURM job id. The dataset is read from each tree's own
pipeline.json rather than parsed out of the tag, so `netscience_8251` and
`ba40_918273` both land correctly.

Dry-run by default; pass --apply to actually move anything.

    python -m pipeline.migrate_results
    python -m pipeline.migrate_results --apply
"""

import argparse
import json
import shutil
from pathlib import Path

from pipeline.tasks import default_run


def plan_moves(root: Path, task: str) -> list[tuple[Path, Path]]:
    moves = []

    for manifest in sorted(root.glob("*/pipeline.json")):
        old_dir = manifest.parent
        config = json.loads(manifest.read_text()).get("config", {})

        dataset = config.get("dataset")
        if dataset is None:
            raise ValueError(
                f"{manifest} has no config.dataset: it predates the manifest "
                f"format and must be moved by hand"
            )

        # Old tags were <dataset>, <dataset><size>, or either plus _<jobid>.
        # Whatever is left after stripping the dataset prefix is the run label.
        suffix = old_dir.name.removeprefix(dataset).lstrip("_")
        run = suffix or default_run

        moves.append((old_dir, root / task / dataset / run))

    return moves


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=str,
        default="results",
        help="top-level results directory (default: results).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="influence_maximization",
        help="task every pre-task tree belongs to (default: influence_maximization).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually move the trees; otherwise only print the plan (default: False).",
    )
    args = parser.parse_args()

    root = Path(args.results_root)
    moves = plan_moves(root, args.task)

    if not moves:
        print(f"[migrate] no pre-task trees under {root}/: nothing to do")
        raise SystemExit(0)

    for old_dir, new_dir in moves:
        marker = "!! TARGET EXISTS" if new_dir.exists() else ""
        print(f"[migrate] {old_dir}  ->  {new_dir} {marker}")

    if not args.apply:
        print(f"\n[migrate] {len(moves)} tree(s); dry run, pass --apply to move them")
        raise SystemExit(0)

    for old_dir, new_dir in moves:
        if new_dir.exists():
            raise FileExistsError(
                f"{new_dir} already exists; move or delete it before migrating "
                f"{old_dir}"
            )

        new_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_dir), str(new_dir))
        print(f"[migrate] moved {old_dir} -> {new_dir}")

    print(f"[migrate] done, {len(moves)} tree(s) moved")
