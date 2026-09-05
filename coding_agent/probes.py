"""
Agent-level world-model probes.

The generated algorithm stays a pure offline function; the AGENT asks the
questions. A generation's reply may carry one fenced ```probes block of JSON
beside its code block; the harness answers against the arm's own evaluator and
the answers arrive with the next iteration's feedback. Under the native
condition the request is refused by name, because condition 3 is defined by
having no forward model to consult. Every probe rollout goes through
environment.rollout, so its cost lands in evaluator_seconds like all other
evaluator work; each log entry additionally carries its own rollout count and
wall time for the results JSON.
"""

import json
import re
import time
from dataclasses import replace
from functools import partial
from tqdm import tqdm

from coding_agent.credit import (
    batched_environment,
    batched_plan_rewards,
    planned_action,
)

max_probes_per_generation = 6
max_region_nodes = 300

_fence = re.compile(r"```probes\s*(.*?)```", re.DOTALL)


def parse_probe_request(reply: str) -> tuple[list, str | None]:
    """(probes, note): the note is feedback about a malformed or oversized block."""
    matches = _fence.findall(reply)
    if not matches:
        return [], None

    try:
        payload = json.loads(matches[-1])
    except ValueError as error:
        return [], f"your probes block was ignored: invalid JSON ({error})"

    probes = payload.get("probes") if isinstance(payload, dict) else payload
    if not isinstance(probes, list):
        return [], 'your probes block was ignored: expected {"probes": [...]}'

    note = None
    if len(probes) > max_probes_per_generation:
        note = (
            f"only the first {max_probes_per_generation} of {len(probes)} "
            f"probes were answered"
        )
        probes = probes[:max_probes_per_generation]

    return probes, note


def _plan_rewards(environment, plans: list, horizon: int, budget: int) -> list[float]:
    if batched_environment(environment):
        rewards, _ = batched_plan_rewards(environment, plans, horizon, budget, None)
        return rewards

    return [
        environment.rollout(partial(planned_action, plan), horizon, budget).reward
        for plan in plans
    ]


def _drop(plan: list, node: int) -> list:
    # Every action targeting the node goes, and so do edge ops touching it, so a
    # blocked removal's deletion bag leaves with its node
    return [
        [
            action
            for action in bag
            if int(action.target) != node
            and (action.destination is None or int(action.destination) != node)
        ]
        for bag in plan
    ]


def _swap(plan: list, old: int, new: int) -> list:
    return [
        [
            replace(action, target=new) if int(action.target) == old else action
            for action in bag
        ]
        for bag in plan
    ]


def execute_probes(
    environment: object,
    plan: list | None,
    probes: list,
    horizon: int,
    budget: int,
    enabled: bool = True,
    note: str | None = None,
) -> tuple[str | None, list[dict]]:
    """
    Answer one generation's probe request: (feedback text, log entries).

    `plan` is the bags the agent's last evaluated candidate actually ran, which
    is what its feedback describes and therefore what its questions are about.
    """
    if not probes and not note:
        return None, []

    if not enabled:
        return (
            "PROBE RESULTS: probes are unavailable under the native condition: "
            "no forward model may be consulted during this search.",
            [],
        )

    if plan is None:
        return ("PROBE RESULTS: no evaluated plan exists yet to probe against.", [])

    lines = [
        "PROBE RESULTS (answered by this arm's evaluator, about your last "
        "evaluated plan):"
    ]
    if note:
        lines.append(f"  note: {note}")

    entries = []
    for probe in probes:
        start = time.perf_counter()
        before = float(getattr(environment, "evaluator_seconds", 0.0))

        try:
            operation = probe.get("op") if isinstance(probe, dict) else None
            if operation == "drop":
                node = int(probe["node"])
                base, dropped = _plan_rewards(
                    environment, [plan, _drop(plan, node)], horizon, budget
                )
                answer = (
                    f"drop({node}): plan={base:.2f}, without={dropped:.2f}, "
                    f"contribution={base - dropped:+.2f}"
                )
                rollouts = 2
            elif operation == "swap":
                old, new = int(probe["a"]), int(probe["b"])
                base, swapped = _plan_rewards(
                    environment, [plan, _swap(plan, old, new)], horizon, budget
                )
                answer = (
                    f"swap({old}->{new}): plan={base:.2f}, swapped={swapped:.2f}, "
                    f"delta={swapped - base:+.2f}"
                )
                rollouts = 2
            elif operation == "region":
                nodes = [int(node) for node in probe["nodes"]][:max_region_nodes]
                trajectory = environment.rollout(
                    partial(planned_action, plan), horizon, budget
                )
                mass = float(
                    sum(trajectory.final_marginals[node] for node in nodes)
                )
                answer = (
                    f"region({len(nodes)} nodes): expected mass captured="
                    f"{mass:.2f} ({mass / max(len(nodes), 1):.0%} of the region)"
                )
                rollouts = 1
            else:
                answer = f"unknown probe op {operation!r}: use drop / swap / region"
                rollouts = 0
        except (KeyError, TypeError, ValueError, IndexError) as error:
            # Surfaced to the model and the log rather than crashing the search:
            # a malformed probe is the agent's mistake to read about and fix
            answer = f"probe {probe!r} failed: {error}"
            rollouts = 0

        lines.append(f"  {answer}")
        entries.append(
            {
                "probe": probe,
                "answer": answer,
                "rollouts": rollouts,
                "seconds": round(time.perf_counter() - start, 3),
                "evaluator_seconds": round(
                    float(getattr(environment, "evaluator_seconds", 0.0)) - before, 3
                ),
            }
        )

    return "\n".join(lines), entries


def answer_probes(
    environment: object,
    probe_request: list,
    probe_note: str | None,
    plan: list | None,
    task,
    enabled: bool,
    log: list,
    label: str = "agent",
) -> str | None:
    """Run one generation's probe block and append its entries to `log`."""
    if not probe_request and not probe_note:
        return None

    text, entries = execute_probes(
        environment,
        plan,
        probe_request,
        task.horizon,
        task.budget,
        enabled=enabled,
        note=probe_note,
    )
    log += entries

    if entries:
        tqdm.write(f"[{label}] answered {len(entries)} probes")

    return text
