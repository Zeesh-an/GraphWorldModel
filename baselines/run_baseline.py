"""
Run one external published baseline on our graph and return what it produced.

The external repo runs in its own subprocess and its own virtualenv (created by
setup_baselines.py), so its dependency pins never touch ours. All we take back
across the boundary is node ids, which the caller then scores with our own
referee, exactly like every other arm.

Two shapes, because two kinds of task:

  * **Intervention tasks** (influence maximization, critical node detection) hand
    the repo a GRAPH and take back one seed or removal set:
    `(G, k) -> S`. `seed_script` wraps it as a canned `plan_horizon`.

  * **Inverse tasks** (source localization) hand it a graph AND a batch of
    observed diffusion states, and take back one SOURCE SET PER OBSERVATION:
    `(G, {y_i}, k) -> {x_i}`. `localize_script` wraps that as a canned
    `localize`. The batch is deliberate: every one of these repos is a library
    that loops internally, and N subprocess launches would dominate the runtime
    of the thing we are trying to measure.

    Warning: The inverse batch carries LABELS for the selection split, and that is not
    a leak: these repos tune hyperparameters on labelled data exactly as our own
    outer loop selects a program on labelled episodes. What must never happen is
    a label reaching the EVALUATION split's prediction, which is why the driver
    scripts call each repo's `train` on the selection pool and its label-free
    `predict` on the evaluation pool, and never its `test` (which scores against
    labels internally and returns no per-instance prediction anyway).

    python -m baselines.run_baseline --name moeim \
        --data-dir results/ba40/data --budget 5 --diffusion-model IC
"""

import argparse
import json
import numpy as np
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from baselines.registry import external_baselines
from coding_agent.types import GraphInfo
from world_model.wm_data import load_graph_store

default_timeout_seconds = 3600


class BaselineError(RuntimeError):
    """Raised when an external baseline cannot run or produced no usable seeds."""


@dataclass
class _Completed:
    """The subset of CompletedProcess the caller uses."""

    returncode: int
    stdout: str
    stderr: str


def _stream(
    argv: list[str],
    cwd: Path,
    timeout: int,
    env: dict,
    name: str,
    log_path: Path,
) -> _Completed:
    """
    Run a baseline with its output echoed live, and captured.

    subprocess.run(capture_output=True) holds everything in memory until the
    process exits, so a baseline that trains for hours contributes nothing to
    the job log until it is over — a hang and steady progress look identical.
    Here each line is written to the log file, flushed, and echoed with the
    baseline's name so interleaved arms stay attributable.

    stderr is merged into stdout: these repos print progress to both, and
    keeping two streams in order would need a second reader thread.
    """
    os.makedirs(log_path.parent, exist_ok=True)
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    # A watchdog rather than a per-line deadline check: a child that hangs
    # without printing would never reach the check.
    killed = threading.Event()

    def kill() -> None:
        killed.set()
        process.kill()

    watchdog = threading.Timer(timeout, kill)
    watchdog.start()
    lines = []

    try:
        with open(log_path, "w") as log:
            for line in process.stdout:
                lines.append(line)
                log.write(line)
                log.flush()
                print(f"[{name}] {line.rstrip()}", flush=True)

        process.wait()
    finally:
        watchdog.cancel()

    if killed.is_set():
        raise subprocess.TimeoutExpired(argv, timeout)

    return _Completed(process.returncode, "".join(lines), "")


def _interpreter(spec) -> str:
    """The baseline's own venv python, falling back to ours if it has no venv."""
    venv_python = spec.venv_python

    return str(venv_python) if venv_python.exists() else sys.executable


schedule_filename = "round_schedule.json"


def run_external_baseline(
    name: str,
    graph: GraphInfo,
    budget: int,
    diffusion_model: str = "IC",
    work_dir: Path | None = None,
    timeout: int = default_timeout_seconds,
    batches: list[int] | None = None,
    instances: list | None = None,
) -> dict:
    """
    Run one external repo and return the seed set it produced.

    `instances` switches this to the INVERSE contract: the repo is handed a batch
    of observed diffusion states and returns one source set per observation,
    returned under `"sources"` instead of `"seeds"`. Passing it also selects the
    localization signatures of `export` and `parse_seeds`, which take the instance
    list where the intervention ones take a budget — a split rather than a widened
    signature, so the seven already-wired IM adapters are untouched.

    `batches` is the round schedule for an adaptive task. It is written into
    work_dir as JSON rather than added to the export()/command() signatures,
    which would churn seven already-wired IM adapters for a field none of them
    read. A rounds-aware adapter loads it; everything else ignores the file.
    """
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

    if spec.rounds_aware:
        if not batches:
            raise BaselineError(
                f"baseline {name!r} is rounds-aware but no round schedule was "
                f"passed. It selects seeds in batches, and running it at one "
                f"batch of k would silently turn an adaptive method into a "
                f"static one."
            )

        (work_dir / schedule_filename).write_text(json.dumps(batches))

    start = time.perf_counter()

    # Everything below is third-party code and third-party file formats. ANY
    # failure here must surface as BaselineError, because that is the only
    # exception the pipeline catches — anything else aborts the whole sweep and
    # loses the arms that already succeeded.
    try:
        extras = (
            spec.export(graph, work_dir, instances, diffusion_model)
            if instances is not None
            else spec.export(graph, work_dir, budget, diffusion_model)
        )
        argv = spec.command(work_dir, budget, diffusion_model, extras, graph)
        argv[0] = _interpreter(spec) if argv[0] == "python" else argv[0]

        print(f"[baseline:{name}] {' '.join(argv)}", flush=True)
        completed = _stream(
            argv,
            cwd=spec.directory,
            timeout=timeout,
            env={**os.environ, **spec.extra_env},
            name=name,
            log_path=work_dir / "stdout.log",
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


    if completed.returncode != 0 and not spec.allow_nonzero_exit:
        raise BaselineError(
            f"baseline {name!r} exited {completed.returncode}. Logs in {work_dir}. "
            f"output tail:\n{completed.stdout[-1500:]}"
        )

    if instances is not None:
        return _collect_sources(
            spec, name, graph, work_dir, completed, instances, elapsed
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
        # Selection order is meaningful for a rounds-aware baseline: slicing by
        # the schedule recovers which batch each seed belonged to
        "batches": batches if spec.rounds_aware else None,
        "seconds": round(elapsed, 2),
        "work_dir": str(work_dir),
        # Persisted so an under-spent budget stays visible in the results JSON,
        # not just in a log line that scrolls past
        "budget": budget,
        "seeds_returned": len(seeds),
    }


def round_seed_script(
    seeds: list[int], batches: list[int], round_gap: int, budget_op: str = "add_node"
) -> str:
    """
    Wrap a rounds-aware repo's seed set as a canned per-round policy.

    The seeds arrive in SELECTION order, so slicing by the schedule recovers the
    batch each one belonged to, and act() hands back batch i at round i. That
    keeps the arm on the adaptive side of the gap table instead of collapsing it
    to a single t=0 plan, which is what seed_script would do.

    Be precise about what this measures. The batches were chosen against the
    repo's OWN realizations, not against the state our simulator goes on to
    produce, so the arm is the repo's SCHEDULE replayed under our referee, not
    the repo adapting inside our environment. True cross-process adaptivity
    would need the repo to accept an already-active set each round, which none
    of them expose.
    """
    plan = {}
    start = 0
    for index, size in enumerate(batches):
        plan[index * round_gap] = seeds[start : start + size]
        start += size

    return f"""\
class ExternalAdaptiveBaseline(Strategy):
    def act(self, state, graph, timestep):
        active = set(state.infected) | set(state.frontier)
        picks = [n for n in {plan!r}.get(timestep, []) if n not in active]
        return [ActionOp({budget_op!r}, node) for node in picks]
"""


def _collect_sources(
    spec, name: str, graph, work_dir: Path, completed, instances: list, elapsed: float
) -> dict:
    """
    Read one source set per instance back across the boundary and validate it.

    The same three guards the seed path applies, restated per instance because a
    batch fails one row at a time: ids inside the graph, no duplicates, and no
    more than that instance's own `k`. An over-length set would buy recall for
    free, and out-of-range ids mean the repo was run on a different graph.
    """
    try:
        returned = spec.parse_seeds(work_dir, completed.stdout, instances)
    except Exception as error:
        raise BaselineError(
            f"baseline {name!r} ran but its per-instance source output could not "
            f"be parsed ({type(error).__name__}: {error}). Logs in {work_dir} — "
            f"check stdout.log and fix parse_seeds in baselines/registry.py."
        ) from error

    sources = {}
    short = 0

    for instance in instances:
        key = observation_key(instance.observation)
        predicted = [int(node) for node in returned.get(instance.episode_id, [])]
        budget = max(1, instance.source_count)

        outside = [node for node in predicted if not 0 <= node < graph.num_nodes]
        if outside:
            raise BaselineError(
                f"baseline {name!r} returned node ids {outside[:5]} outside "
                f"[0, {graph.num_nodes}) for episode {instance.episode_id} — it "
                f"was run on a different graph than the one being scored (stale "
                f"cached input?). Logs in {work_dir}."
            )

        deduplicated = list(dict.fromkeys(predicted))[:budget]
        if len(deduplicated) < budget:
            short += 1

        sources[key] = deduplicated

    if not any(sources.values()):
        raise BaselineError(
            f"baseline {name!r} returned no sources for any of the "
            f"{len(instances)} instances. Logs in {work_dir}."
        )

    if short:
        print(
            f"[baseline:{name}] WARNING: {short}/{len(instances)} instances got "
            f"fewer sources than their k — those rows are scored on a shorter "
            f"prediction than every other arm, which reads as low recall"
        )

    print(
        f"[baseline:{name}] {len(sources)} source sets in {elapsed:.1f}s "
        f"({len(instances)} instances)"
    )

    return {
        "name": name,
        # Keyed by observation, not by episode id: the canned localize() is handed
        # an observation and nothing else (see localize_script)
        "sources": sources,
        "seconds": round(elapsed, 2),
        "work_dir": str(work_dir),
        "instances": len(instances),
        "instances_short": short,
    }


def observation_key(observation) -> str:
    """
    Canonical identity of one observed diffusion state: its infected node ids.

    `localize(graph, observation, budget)` is handed no episode id, so a canned
    external result has to find its own row by the observation itself. Keying on
    the thresholded infected SET rather than on call order is what makes that
    correct across the two passes an inverse arm makes (the selection pool, then
    the held-out pool), which visit different instances in a different order.

    Two episodes with an identical infected set collapse to one key, and that is
    the right behaviour rather than a collision to defend against: a localizer is
    a function of the observation, so identical observations must produce
    identical predictions.
    """
    infected = np.flatnonzero(np.asarray(observation, dtype=float) >= 0.5)

    return ",".join(str(int(node)) for node in infected)


def localize_script(sources: dict[str, list[int]]) -> str:
    """
    Wrap one source set per observation as a canned Strategy for an inverse task.

    The mapping is embedded in the script rather than read from a file because a
    generated script may not `open()` (`executor.forbidden_builtins`), and it is
    keyed by `observation_key` rather than by index for the reason that function
    documents.

    A missing key RAISES instead of falling back to a guess. It means the repo was
    handed a different instance pool than the one being scored, and a silent empty
    prediction would surface as a weak method rather than as the wiring bug it is.
    """
    return f"""\
import numpy as np

class ExternalLocalizer(Strategy):
    sources = {sources!r}

    def localize(self, graph, observation, budget):
        infected = np.flatnonzero(np.asarray(observation, dtype=float) >= 0.5)
        key = ",".join(str(int(node)) for node in infected)
        found = self.sources.get(key)

        if found is None:
            raise KeyError(
                "the external baseline returned no prediction for this "
                "observation (" + str(len(infected)) + " infected nodes). It was "
                "run on a different instance pool than the one being scored — "
                "check that --sl-instances, --sl-observation and --seed match "
                "between the export and the evaluation."
            )

        return [int(node) for node in found[:budget]]
"""


def seed_script(seeds: list[int], budget_op: str = "add_node") -> str:
    """
    Wrap an external repo's node set as a canned Strategy.

    This is why external baselines need no special result format: the canned
    script flows through the identical executor, validation, rollout, MC referee,
    and results JSON that every other arm uses.

    `budget_op` is what the TASK budgets. A dismantling repo returns a set to
    REMOVE, not to seed, and emitting it as `add_node` is rejected by
    `validate_actions` — so condition 7 would fail on every containment task
    while looking like a bad generated program rather than a wiring bug.
    """
    return f"""\
class ExternalBaseline(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        picks = [n for n in {seeds!r} if n not in self.outbreak][:budget]
        return [[ActionOp({budget_op!r}, node) for node in picks]] + [
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
