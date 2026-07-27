"""
Run one external published baseline on our graph and return its seed set.

The external repo runs in its own subprocess and its own virtualenv (created by
setup_baselines.py), so its dependency pins never touch ours. All we take back
across the boundary is a list of node ids — which the caller then scores with our
Monte Carlo referee, exactly like every other arm.

    python -m baselines.run_baseline --name moeim \
        --data-dir results/ba40/data --budget 5 --diffusion-model IC
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from baselines.registry import external_baselines
from coding_agent.types import GraphInfo
from world_model.wm_data import load_graph_store

default_timeout_seconds = 3600


class BaselineError(RuntimeError):
    """Raised when an external baseline cannot run or produced no usable seeds."""


def _interpreter(spec) -> str:
    """The baseline's own venv python, falling back to ours if it has no venv."""
    venv_python = spec.venv_python

    return str(venv_python) if venv_python.exists() else sys.executable


def run_external_baseline(
    name: str,
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    work_dir: Path | None = None,
    timeout: int = default_timeout_seconds,
) -> dict:
    if name not in external_baselines:
        raise BaselineError(
            f"unknown external baseline {name!r}; registered: "
            f"{sorted(external_baselines)}"
        )

    spec = external_baselines[name]

    if spec.status == "blocked":
        raise BaselineError(f"baseline {name!r} is blocked: {spec.blocker}")

    if not spec.installed():
        raise BaselineError(
            f"baseline {name!r} is not installed at {spec.directory}. Run:\n"
            f"    python -m baselines.setup_baselines --only {name}"
        )

    if spec.export is None or spec.command is None:
        raise BaselineError(
            f"baseline {name!r} is fetched but has no adapter wired yet "
            f"(entry point: {spec.entry}); implement export/command/parse_seeds "
            f"in baselines/registry.py"
        )

    work_dir = Path(work_dir or spec.directory / "_runs" / f"k{budget}")
    os.makedirs(work_dir, exist_ok=True)

    start = time.perf_counter()

    # Everything below is third-party code and third-party file formats. ANY
    # failure here must surface as BaselineError, because that is the only
    # exception the pipeline catches — anything else aborts the whole sweep and
    # loses the arms that already succeeded.
    try:
        extras = spec.export(graph, work_dir, budget, diffusion_model)
        argv = spec.command(work_dir, budget, diffusion_model, extras, graph)
        argv[0] = _interpreter(spec) if argv[0] == "python" else argv[0]

        print(f"[baseline:{name}] {' '.join(argv)}")
        completed = subprocess.run(
            argv,
            cwd=spec.directory,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, **spec.extra_env},
        )
    except subprocess.TimeoutExpired as error:
        raise BaselineError(
            f"baseline {name!r} exceeded the {timeout}s timeout and was killed. "
            f"Raise --baseline-timeout or reduce the budget. Logs in {work_dir}."
        ) from error
    except FileNotFoundError as error:
        raise BaselineError(
            f"baseline {name!r} is missing a file or binary it needs ({error}). "
            f"Did its build step run? Try: python -m baselines.setup_baselines "
            f"--only {name}"
        ) from error
    except Exception as error:
        raise BaselineError(
            f"baseline {name!r} failed while preparing or launching "
            f"({type(error).__name__}: {error}). Logs in {work_dir}."
        ) from error

    elapsed = time.perf_counter() - start

    (work_dir / "stdout.log").write_text(completed.stdout)
    (work_dir / "stderr.log").write_text(completed.stderr)

    if completed.returncode != 0:
        raise BaselineError(
            f"baseline {name!r} exited {completed.returncode}. Logs in {work_dir}. "
            f"stderr tail:\n{completed.stderr[-1500:]}"
        )

    try:
        raw = [int(node) for node in spec.parse_seeds(work_dir, completed.stdout, budget)]
        seeds = [node for node in raw if 0 <= node < graph.num_nodes]

        # Out-of-range ids mean the repo was handed, or cached, a different
        # graph. Dropping them quietly leaves a short seed set that loses the
        # comparison for a reason that looks like poor method quality.
        if len(seeds) != len(raw):
            raise BaselineError(
                f"baseline {name!r} returned {len(raw) - len(seeds)} of "
                f"{len(raw)} seeds outside [0, {graph.num_nodes}) — it was run "
                f"on a different graph than the one being scored (stale cached "
                f"input?). Logs in {work_dir}."
            )
    except BaselineError:
        raise
    except Exception as error:
        raise BaselineError(
            f"baseline {name!r} ran but its seed output could not be parsed "
            f"({type(error).__name__}: {error}). Logs in {work_dir} — check "
            f"stdout.log and fix parse_seeds in baselines/registry.py."
        ) from error

    if not seeds:
        raise BaselineError(
            f"baseline {name!r} returned no valid seeds. Logs in {work_dir}."
        )

    if len(seeds) > budget:
        raise BaselineError(
            f"baseline {name!r} returned {len(seeds)} seeds, exceeding budget {budget}"
        )

    # Under-spending the budget is not an error — some methods legitimately
    # stop early — but it is never visible in the spread column, where it just
    # looks like a weak method. Say it out loud and record it.
    if len(seeds) < budget:
        print(
            f"[baseline:{name}] WARNING: returned {len(seeds)} seeds for budget "
            f"{budget} — it is being scored on {budget - len(seeds)} fewer seeds "
            f"than every other arm at this budget"
        )

    print(f"[baseline:{name}] {len(seeds)} seeds in {elapsed:.1f}s")

    return {
        "name": name,
        "seeds": seeds,
        "seconds": round(elapsed, 2),
        "work_dir": str(work_dir),
        # Persisted so an under-spent budget stays visible in the results JSON,
        # not just in a log line that scrolls past
        "budget": budget,
        "seeds_returned": len(seeds),
    }


def seed_script(seeds: list[int]) -> str:
    """
    Wrap a seed set as a canned Strategy.

    This is why external baselines need no special result format: the canned
    script flows through the identical executor, validation, rollout, MC referee,
    and results JSON that every other arm uses.
    """
    return f"""\
class ExternalBaseline(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        seeds = {seeds!r}[:budget]
        return [[ActionOp("add_node", node) for node in seeds]] + [
            [] for _ in range(horizon)
        ]
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one external IM baseline")
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        choices=sorted(external_baselines),
        help="registered baseline name (default: required).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="graph store directory, e.g. results/ba40/data (default: required).",
    )
    parser.add_argument(
        "--graph-id",
        type=str,
        default=None,
        help="graph id inside the store (default: the first).",
    )
    parser.add_argument(
        "--budget", type=int, default=5, help="seed budget k (default: 5)."
    )
    parser.add_argument(
        "--diffusion-model",
        type=str,
        default="IC",
        choices=["IC", "LT"],
        help="diffusion dynamics (default: IC).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=default_timeout_seconds,
        help=f"seconds before the subprocess is killed (default: {default_timeout_seconds}).",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="write the seed set here (default: None).",
    )

    args = parser.parse_args()

    store = load_graph_store(args.data_dir)
    graph_id = args.graph_id or next(iter(store))
    result = run_external_baseline(
        args.name,
        GraphInfo.from_store_entry(store[graph_id]),
        args.budget,
        args.diffusion_model,
        timeout=args.timeout,
    )

    if args.out_json:
        os.makedirs(Path(args.out_json).parent, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
