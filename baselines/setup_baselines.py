"""
Fetch and install the external baseline repos into baselines/external/.

Each baseline gets its OWN virtualenv so its dependency pins (old torch, old
networkx, python2-era code) never collide with ours. Nothing here is imported by
the pipeline at runtime: the pipeline only ever shells out to these repos.

    python -m baselines.setup_baselines --list
    python -m baselines.setup_baselines --only moeim touplegdd
    python -m baselines.setup_baselines --all

Clones are shallow (--depth 1) because some of these repos carry large data
histories; MOEIM alone is ~128 MB. baselines/external/ is gitignored.
"""

import argparse
import os
import re
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


def _abi_hint(name: str, output: str) -> str:
    """
    Turn uv's "no wheels with a matching Python ABI tag" into the pin that fixes it.

    That error names the interpreter you have and the tags the package publishes,
    but never the conclusion, so every occurrence costs a manual read of the tag
    list. These repos pin 2019-2021 dependency sets and the fix is almost always a
    lower `python=` in the registry, so say which one.
    """
    if "matching Python ABI tag" not in output:
        return ""

    # Parse ONLY the published-tag list. The same message also says "You require
    # CPython 3.10 (`cp310`)", and scanning the whole text picks that up and
    # reports the interpreter you already have as the fix.
    listed = re.search(r"following Python ABI tags:(.*)", output, re.S)
    if not listed:
        return ""

    # `\d+`, not `\d`: tags run cp37m and cp310, so a single digit reads the
    # latter as minor version 1
    tags = set(re.findall(r"cp3(\d+)", listed.group(1)))
    if not tags:
        return ""

    best = max(int(tag) for tag in tags)

    return (
        f"\n[setup] {name}: the pinned dependency publishes wheels only up to "
        f"CPython 3.{best}. Set python=\"3.{best}\" on the {name} spec in "
        f"baselines/registry.py and re-run; the venv is created at that version."
    )


def _run(
    argv: list[str],
    cwd: Path | None = None,
    timeout: int = 600,
    name: str | None = None,
    env: dict | None = None,
) -> None:
    """
    Run a setup command, streaming stdout and capturing stderr.

    stderr is captured rather than streamed so a failure can be READ before it is
    re-raised: uv's ABI-tag error carries the fix (see `_abi_hint`) and losing it
    to the terminal makes every occurrence a manual diagnosis. It is printed
    verbatim either way, so nothing is hidden; only its timing changes.
    """
    print(f"  $ {' '.join(argv)}")
    completed = subprocess.run(
        argv, cwd=cwd, timeout=timeout, stderr=subprocess.PIPE, text=True, env=env
    )
    errors = completed.stderr or ""

    if errors:
        print(errors, end="" if errors.endswith("\n") else "\n")

    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(argv)}"
            + (_abi_hint(name, errors) if name else "")
        )


class ManualFetch(RuntimeError):
    """The authors publish a tarball, not a git repo: a human must fetch it."""


def clone(name: str) -> Path:
    spec = external_baselines[name]
    os.makedirs(external_root, exist_ok=True)

    if spec.installed():
        print(f"[setup] {name}: already present at {spec.root}")
        return spec.root

    if spec.fetch == "manual":
        raise ManualFetch(
            f"{name} is not on git: download it from {spec.repo} and unpack it "
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
                f"{name}: cannot patch {path}, the repo layout changed"
            )

        text = path.read_text()

        # Which side is the "already applied?" marker depends on the patch shape,
        # and getting it wrong silently skips the edit forever.
        #
        # WRAPPING patch (`old` appears inside `new`, e.g. prepending an import):
        # `old in text` stays true after applying, so it would re-apply on every
        # run. Test `new`.
        #
        # REPLACING patch (`old` does not appear in `new`, e.g. relaxing a pin
        # `scikit-learn==0.21.1` to `scikit-learn`): `new` is often a SUBSTRING of
        # `old`, so testing `new` matches the UNPATCHED text and the edit never
        # happens. Test `old`, which is exact in both directions.
        #
        # A deletion patch has new == "", which is in every string, so it is a
        # replacing patch by this rule and tests `old`, which is what it needs.
        wrapping = bool(new) and old in new
        applied = (new in text) if wrapping else (old not in text)

        if applied:
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
            f"{name}: build directory {spec.directory} does not exist, the "
            f"repo layout changed, check the spec's subdir"
        )

    print(f"[setup] {name}: building ({' '.join(spec.build)}) in {spec.directory}")
    _run(spec.build, cwd=spec.directory, timeout=install_timeout_seconds)


# Packages whose setup.py imports torch, so they cannot be built in isolation
torch_extension_packages = frozenset(
    {"torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv"}
)


def _create_venv(name: str, venv: Path) -> None:
    """
    Prefer `uv venv`, at the interpreter version the spec asks for.

    stdlib `python -m venv` needs ensurepip, which Debian/Ubuntu ship in a
    separate python3-venv package that is usually absent on a cluster node,
    and asking for sudo on a shared machine is not a fix. uv bootstraps its own
    pip, so it works where the stdlib path cannot.

    `spec.python` is load-bearing and not decoration: these repos pin 2019-2021
    dependency sets whose wheels simply do not exist for a modern interpreter.
    `tensorflow-gpu==1.14.0` publishes cp27mu through cp37m, `torch==1.7.0`
    cp36m through cp38, `torch==1.5.1` cp35m through cp38. Creating the venv at
    whatever `python3` happens to be makes those unresolvable, and the error
    ("no wheels with a matching Python ABI tag") names the interpreter rather
    than the fix. uv fetches a managed CPython when the version is absent.
    """
    uv = shutil.which("uv")
    wanted = external_baselines[name].python

    if uv is not None:
        # A compound pin like gcomb's "2.7+3.6" names two interpreters and is a
        # note to a human, not something uv can resolve
        if wanted and "+" not in wanted:
            try:
                _run([uv, "venv", "--python", wanted, str(venv)], name=name)
                return
            except RuntimeError:
                print(
                    f"[setup] {name}: no CPython {wanted} available, falling back "
                    f"to the default interpreter (pinned dependencies may not "
                    f"resolve)"
                )

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


def _venv_version(interpreter: Path) -> str:
    """`major.minor` of an existing venv's interpreter, or "" if it cannot be read."""
    try:
        done = subprocess.run(
            [
                str(interpreter),
                "-c",
                "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return ""

    return done.stdout.strip() if done.returncode == 0 else ""


def _version_matches(interpreter: Path, wanted: str | None) -> bool:
    """A compound pin ("2.7+3.6") names two interpreters and is a note to a human."""
    if not wanted or "+" in wanted:
        return True

    found = _venv_version(interpreter)

    return not found or found == wanted


def install(name: str) -> None:
    spec = external_baselines[name]

    requirements = spec.root / spec.requirements if spec.requirements else None
    has_requirements = requirements is not None and requirements.exists()

    # A package-distributed baseline (GraphSL, cosasi) has no usable requirements
    # file in its clone: the PyPI package IS the install, so a venv is still
    # needed even though `requirements` is None
    if not has_requirements and not spec.pip_packages:
        if spec.requirements is None:
            print(
                f"[setup] {name}: no Python requirements ({spec.entry}), build manually"
            )
        else:
            print(f"[setup] {name}: no {spec.requirements} in the repo, skipping install")
        return

    venv = spec.root / ".venv"
    interpreter = spec.venv_python

    # Test for the interpreter, not the directory: a venv creation that failed
    # part-way leaves a directory behind with no python and no pip in it.
    #
    # A venv at the WRONG version is also rebuilt. Without this an existing 3.10
    # venv from an earlier run survives, `_create_venv` is never called, and the
    # `python=` pin silently does nothing: the install then fails on exactly the
    # ABI wall the pin exists to avoid, and re-running never fixes it.
    stale = interpreter.exists() and not _version_matches(interpreter, spec.python)

    if stale:
        print(
            f"[setup] {name}: venv is {_venv_version(interpreter)}, spec wants "
            f"{spec.python}; rebuilding"
        )

    if not interpreter.exists() or stale:
        shutil.rmtree(venv, ignore_errors=True)
        print(f"[setup] {name}: creating venv")
        _create_venv(name, venv)

    uv = shutil.which("uv")

    # `uv venv` deliberately does not put pip inside the venv, so install
    # through uv itself and point it at that interpreter
    command = (
        [uv, "pip", "install", "--python", str(interpreter)]
        if uv is not None
        else [str(venv / "bin" / "pip"), "install"]
    )

    # torch-scatter / torch-sparse / torch-cluster compile against torch and
    # `import torch` inside their own setup.py, so with build isolation they are
    # built in a fresh env where torch is absent and die on ModuleNotFoundError.
    # Installing torch first and then building them without isolation is the
    # upstream-documented fix, and it has to happen BEFORE the main install.
    needs_torch_build = [
        package
        for package in (spec.pip_packages or ())
        if package.split("==")[0] in torch_extension_packages
    ]

    # ...and the same packages named inside a requirements FILE. Scanning only
    # `pip_packages` missed `glie`, whose torch-sparse pin lives in
    # requirements.txt and hit the identical build-isolation wall. The line is
    # taken verbatim so the pre-install satisfies the exact pin the file asks
    # for, which makes the later full install a no-op for it rather than a
    # second, isolated, failing build.
    if has_requirements:
        for line in requirements.read_text().splitlines():
            entry = line.split("#")[0].strip()
            name_only = re.split(r"[=<>!~\[]", entry)[0].strip()
            if entry and name_only in torch_extension_packages:
                needs_torch_build.append(entry)

    if needs_torch_build and uv is not None:
        print(f"[setup] {name}: pre-installing torch for {' '.join(needs_torch_build)}")
        # numpy first: torch warns "Failed to initialize NumPy" without it and the
        # extension's own setup.py imports torch, so the warning becomes noise in
        # every subsequent build log
        _run(
            command + ["-q", "torch", "numpy"],
            timeout=install_timeout_seconds,
            name=name,
        )

        # FORCE_ONLY_CPU short-circuits torch-scatter's CUDA-version assertion,
        # which compares the CUDA that built the torch wheel against the CUDA on
        # the box and refuses to build when they differ (measured on the cluster:
        # driver 12.9 against a torch built with 13.0). These three repos are
        # driven through their MODEL by our own loop rather than through their
        # training harness, and nothing we call needs a CUDA scatter kernel, so a
        # CPU build is correct rather than merely expedient.
        cpu_build = dict(os.environ, FORCE_ONLY_CPU="1", FORCE_CUDA="0")
        _run(
            command + ["-q", "--no-build-isolation", *needs_torch_build],
            timeout=install_timeout_seconds,
            name=name,
            env=cpu_build,
        )

    if has_requirements:
        print(f"[setup] {name}: installing {spec.requirements}")
        _run(
            command + ["-q", "-r", str(requirements)],
            timeout=install_timeout_seconds,
            name=name,
        )

    _install_packages(name, spec, command)


def _install_packages(name: str, spec, command: list[str]) -> None:
    """
    The spec's PyPI packages, for a library driven as a package rather than a clone.

    GraphSL and cosasi are published on PyPI and their clones carry no usable
    requirements file, so the package IS the install. Kept separate from
    `requirements` rather than folded into it: a repo may legitimately need both.
    """
    if not spec.pip_packages:
        return

    print(f"[setup] {name}: installing {' '.join(spec.pip_packages)}")
    _run(
        command + ["-q", *spec.pip_packages],
        timeout=install_timeout_seconds,
        name=name,
    )


def setup(name: str) -> None:
    spec = external_baselines[name]

    if spec.status == "blocked":
        print(f"[setup] {name}: BLOCKED, {spec.blocker}")
        return

    try:
        clone(name)
    except ManualFetch as error:
        # Not a failure of the run: one baseline needing a human is expected,
        # and the rest of --all should still install
        print(f"[setup] {name}: MANUAL, {error}")
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
    # cloned, fail its install on some abandoned pin, and still be undriveable,
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
    # from installing: the failures are reported together at the end
    failures = {}
    for name in targets:
        try:
            setup(name)
        except Exception as error:
            failures[name] = f"{type(error).__name__}: {error}"
            print(f"[setup] {name}: FAILED, {failures[name]}")

    installed = [name for name in targets if external_baselines[name].installed()]
    print(f"\n[setup] {len(installed)}/{len(targets)} present: {' '.join(installed)}")

    if failures:
        print(f"[setup] {len(failures)} failed:")
        for name, reason in failures.items():
            print(f"  {name}: {reason}")
        raise SystemExit(1)
