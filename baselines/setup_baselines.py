"""
Fetch and install the external baseline repos into baselines/external/.

Each baseline gets its OWN virtualenv so its dependency pins (old torch, old
networkx, python2-era code) never collide with ours. Nothing here is imported by
the pipeline at runtime — the pipeline only ever shells out to these repos.

    python -m baselines.setup_baselines --list
    python -m baselines.setup_baselines --only moeim touplegdd
    python -m baselines.setup_baselines --all

Clones are shallow (--depth 1) because some of these repos carry large data
histories; MOEIM alone is ~128 MB. baselines/external/ is gitignored.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from baselines.registry import external_baselines, external_root

clone_timeout_seconds = 1800
install_timeout_seconds = 1800


def _run(argv: list[str], cwd: Path | None = None, timeout: int = 600) -> None:
    print(f"  $ {' '.join(argv)}")
    completed = subprocess.run(argv, cwd=cwd, timeout=timeout)

    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(argv)}")


class ManualFetch(RuntimeError):
    """The authors publish a tarball, not a git repo — a human must fetch it."""


def clone(name: str) -> Path:
    spec = external_baselines[name]
    os.makedirs(external_root, exist_ok=True)

    if spec.installed():
        print(f"[setup] {name}: already present at {spec.directory}")
        return spec.directory

    if spec.fetch == "manual":
        raise ManualFetch(
            f"{name} is not on git — download it from {spec.repo} and unpack it "
            f"into {spec.directory}, then re-run setup to build it."
        )

    print(f"[setup] {name}: cloning {spec.repo}")
    _run(
        ["git", "clone", "--depth", "1", f"{spec.repo}.git", str(spec.directory)],
        timeout=clone_timeout_seconds,
    )

    return spec.directory


def build(name: str) -> None:
    """C++ baselines ship a Makefile instead of Python requirements."""
    spec = external_baselines[name]
    if not spec.build:
        return

    print(f"[setup] {name}: building ({' '.join(spec.build)})")
    _run(spec.build, cwd=spec.directory, timeout=install_timeout_seconds)


def install(name: str) -> None:
    spec = external_baselines[name]

    if spec.requirements is None:
        print(f"[setup] {name}: no Python requirements ({spec.entry}) — build manually")
        return

    requirements = spec.directory / spec.requirements
    if not requirements.exists():
        print(f"[setup] {name}: no {spec.requirements} in the repo, skipping install")
        return

    venv = spec.directory / ".venv"
    if not venv.exists():
        print(f"[setup] {name}: creating venv")
        _run([sys.executable, "-m", "venv", str(venv)])

    print(f"[setup] {name}: installing {spec.requirements}")
    _run(
        [str(venv / "bin" / "pip"), "install", "-q", "-r", str(requirements)],
        timeout=install_timeout_seconds,
    )


def setup(name: str) -> None:
    spec = external_baselines[name]

    if spec.status == "blocked":
        print(f"[setup] {name}: BLOCKED — {spec.blocker}")
        return

    try:
        clone(name)
    except ManualFetch as error:
        # Not a failure of the run: one baseline needing a human is expected,
        # and the rest of --all should still install
        print(f"[setup] {name}: MANUAL — {error}")
        return

    install(name)
    build(name)
    print(f"[setup] {name}: ready at {spec.directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install external IM baselines")
    parser.add_argument(
        "--only",
        type=str,
        nargs="+",
        default=None,
        choices=sorted(external_baselines),
        help="install just these baselines (default: None).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="install every non-blocked baseline (default: False).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the registry with status and exit (default: False).",
    )

    args = parser.parse_args()

    if args.list or (not args.only and not args.all):
        print(f"{'name':<12} {'kind':<10} {'status':<12} {'installed':<10} title")
        for name, spec in sorted(external_baselines.items()):
            print(
                f"{name:<12} {spec.kind:<10} {spec.status:<12} "
                f"{str(spec.installed()):<10} {spec.title}"
            )
            if spec.blocker:
                print(f"{'':<12} └─ blocked: {spec.blocker.splitlines()[0]}")
        raise SystemExit(0)

    targets = args.only or [
        name for name, spec in external_baselines.items() if spec.status != "blocked"
    ]

    for name in targets:
        setup(name)
