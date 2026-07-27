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
import shutil
import subprocess
import sys
from pathlib import Path

from baselines.registry import (
    available_baselines,
    external_baselines,
    external_root,
    unwired_baselines,
)

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
        print(f"[setup] {name}: already present at {spec.root}")
        return spec.root

    if spec.fetch == "manual":
        raise ManualFetch(
            f"{name} is not on git — download it from {spec.repo} and unpack it "
            f"into {spec.root}, then re-run setup to build it."
        )

    print(f"[setup] {name}: cloning {spec.repo}")
    _run(
        ["git", "clone", "--depth", "1", f"{spec.repo}.git", str(spec.root)],
        timeout=clone_timeout_seconds,
    )

    return spec.root


def patch(name: str) -> None:
    """Apply the spec's compatibility edits; a no-op once they are in place."""
    spec = external_baselines[name]
    if not spec.patches:
        return

    for relative, old, new in spec.patches:
        path = spec.root / relative
        if not path.exists():
            raise FileNotFoundError(
                f"{name}: cannot patch {path} — the repo layout changed"
            )

        text = path.read_text()

        # Test for `new` BEFORE `old`. A patch that prepends a line keeps the
        # original text inside its replacement, so `old in text` stays true
        # forever and the edit would be applied again on every setup run.
        if new in text or old not in text:
            print(f"[setup] {name}: patch already applied to {relative}")
            continue

        path.write_text(text.replace(old, new))
        print(f"[setup] {name}: patched {relative} ({old} -> {new})")


def unpack(name: str) -> None:
    """A couple of repos track a release zip rather than the source itself."""
    spec = external_baselines[name]
    if not spec.unpack:
        return

    archive = spec.root / spec.unpack
    if not archive.exists():
        raise FileNotFoundError(f"{name}: expected {archive} in the clone")

    # spec.directory is what the archive is supposed to produce, so its presence
    # means a previous run already expanded this
    if spec.directory.exists():
        print(f"[setup] {name}: already unpacked at {spec.directory}")
        return

    print(f"[setup] {name}: unpacking {spec.unpack}")
    shutil.unpack_archive(str(archive), str(spec.root))


def build(name: str) -> None:
    """C++ baselines ship a Makefile instead of Python requirements."""
    spec = external_baselines[name]
    if not spec.build:
        return

    unpack(name)

    if not spec.directory.exists():
        raise FileNotFoundError(
            f"{name}: build directory {spec.directory} does not exist — the "
            f"repo layout changed, check the spec's subdir"
        )

    print(f"[setup] {name}: building ({' '.join(spec.build)}) in {spec.directory}")
    _run(spec.build, cwd=spec.directory, timeout=install_timeout_seconds)


def _create_venv(name: str, venv: Path) -> None:
    """
    Prefer `uv venv`.

    stdlib `python -m venv` needs ensurepip, which Debian/Ubuntu ship in a
    separate python3-venv package that is usually absent on a cluster node —
    and asking for sudo on a shared machine is not a fix. uv bootstraps its own
    pip, so it works where the stdlib path cannot.
    """
    uv = shutil.which("uv")

    if uv is not None:
        _run([uv, "venv", str(venv)])
        return

    try:
        _run([sys.executable, "-m", "venv", str(venv)])
    except RuntimeError as error:
        raise RuntimeError(
            f"could not create {venv}: stdlib venv needs ensurepip and uv is not "
            f"on PATH. Install uv (curl -LsSf https://astral.sh/uv/install.sh | sh) "
            f"or the system package python3-venv, then re-run."
        ) from error


def install(name: str) -> None:
    spec = external_baselines[name]

    if spec.requirements is None:
        print(f"[setup] {name}: no Python requirements ({spec.entry}) — build manually")
        return

    requirements = spec.root / spec.requirements
    if not requirements.exists():
        print(f"[setup] {name}: no {spec.requirements} in the repo, skipping install")
        return

    venv = spec.root / ".venv"
    interpreter = spec.venv_python

    # Test for the interpreter, not the directory: a venv creation that failed
    # part-way leaves a directory behind with no python and no pip in it
    if not interpreter.exists():
        shutil.rmtree(venv, ignore_errors=True)
        print(f"[setup] {name}: creating venv")
        _create_venv(name, venv)

    print(f"[setup] {name}: installing {spec.requirements}")
    uv = shutil.which("uv")

    # `uv venv` deliberately does not put pip inside the venv, so install
    # through uv itself and point it at that interpreter
    command = (
        [uv, "pip", "install", "--python", str(interpreter)]
        if uv is not None
        else [str(venv / "bin" / "pip"), "install"]
    )
    _run(
        command + ["-q", "-r", str(requirements)],
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

    patch(name)
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

    # --all installs what can actually be RUN. A repo with no adapter would be
    # cloned, fail its install on some abandoned pin, and still be undriveable —
    # so it is skipped unless named explicitly with --only.
    if args.only:
        targets = args.only
    else:
        targets = available_baselines()
        unwired = unwired_baselines()
        if unwired:
            print(
                f"[setup] skipping {len(unwired)} registered baseline(s) with no "
                f"adapter: {' '.join(unwired)}"
            )
            print("[setup]   (install one anyway with --only <name>)")

    # One repo's broken pins or missing toolchain must not stop the other ten
    # from installing — the failures are reported together at the end
    failures = {}
    for name in targets:
        try:
            setup(name)
        except Exception as error:
            failures[name] = f"{type(error).__name__}: {error}"
            print(f"[setup] {name}: FAILED — {failures[name]}")

    installed = [name for name in targets if external_baselines[name].installed()]
    print(f"\n[setup] {len(installed)}/{len(targets)} present: {' '.join(installed)}")

    if failures:
        print(f"[setup] {len(failures)} failed:")
        for name, reason in failures.items():
            print(f"  {name}: {reason}")
        raise SystemExit(1)
