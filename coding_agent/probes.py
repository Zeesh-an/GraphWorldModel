"""
Agent-level evaluator probes.

The generated algorithm stays a pure offline function; the AGENT asks the
questions. Two channels carry them. The probe turn (`evolve --probe-turn`, the
default) is a trace-free aside before each generation is written: the model is
shown the incumbent's plan and may ask about it, and the answers arrive in the
same generation's prompt, so a question has a payoff for the edit being written
now. The write-turn channel is the older one: a fenced ```probes block beside
the code block, answered against the plan that block's script produced and
delivered with the NEXT generation's feedback. Under the native condition both
are refused by name, because condition 3 is defined by having no forward model
to consult. Every probe rollout goes through environment.rollout, so its cost
lands in evaluator_seconds like all other evaluator work; each log entry
additionally carries its own rollout count and wall time for the results JSON.

Twelve ops. drop / swap / region / add / best_swap / overlap / horizon /
frontier / robust / gradient answer what-if questions about an intervention
plan; resimulate answers a source localizer, transmission a trajectory decoder.
gradient, frontier and robust(hidden) need the forward-model evaluators and are
refused with a reason elsewhere.
"""

import json
import re
import time
from dataclasses import replace
from functools import partial
import numpy as np
from tqdm import tqdm

from coding_agent.credit import (
    batched_environment,
    batched_plan_rewards,
    planned_action,
)
from coding_agent.containment import expand_removals
from coding_agent.localization import (
    ForwardOracle,
    consistency_score,
    residual_diagnostics,
)
from coding_agent.reconstruction import StepOracle
from coding_agent.types import ActionOp

max_probes_per_generation = 6
max_region_nodes = 300
# Replacement candidates a best_swap probe scores, ranked by degree; a sampling
# evaluator pays one rollout per candidate, so it gets far fewer
max_best_swap_candidates = 200
max_best_swap_sequential = 20
max_listed_nodes = 10
# A horizon probe may look at most this many times further than the task horizon
max_horizon_multiple = 2
# robust(sigma): each transmission probability is multiplied by exp(N(0, sigma))
default_perturbation = 0.5
robust_seed = 7

intervention_ops = (
    "drop", "swap", "region", "add", "best_swap", "overlap", "horizon",
    "frontier", "robust", "gradient",
)
localization_ops = ("resimulate",)
reconstruction_ops = ("transmission",)

_fence = re.compile(r"```probes\s*(.*?)```", re.DOTALL)


def available_ops(task) -> tuple:
    """Which probe ops this task's family can answer at all."""
    if task.forecasts:
        return ()
    if task.recovers:
        return localization_ops
    if task.decodes:
        return reconstruction_ops

    return intervention_ops


def parse_probe_request(reply: str) -> tuple[list, str | None]:
    """(probes, note): the note is feedback about a malformed or oversized block."""
    match = _fence.search(reply or "")
    if match is None:
        return [], None

    try:
        payload = json.loads(match.group(1))
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


def _budgeted(plan: list, task) -> list[int]:
    """The nodes the plan spends its budget on, in plan order."""
    seen = []
    for bag in plan:
        for action in bag:
            if action.op == task.budget_op and int(action.target) not in seen:
                seen.append(int(action.target))

    return seen


def _node_actions(plan: list, node: int) -> list:
    """Every action of the plan that touches `node` (a blocked removal's whole bag)."""
    return [
        action
        for bag in plan
        for action in bag
        if int(action.target) == node
        or (action.destination is not None and int(action.destination) == node)
    ]


def _add(plan: list, node: int, task, graph) -> list:
    """The plan plus one more budgeted action on `node` at t=0."""
    bag = [ActionOp(task.budget_op, node)]
    if task.budget_op == "remove_node" and task.contains:
        bag = expand_removals(bag, graph)

    extended = [list(bag_) for bag_ in plan] or [[]]
    extended[0] = extended[0] + bag

    return extended


def _named(nodes: list, graph, values: dict | None = None) -> str:
    parts = []
    for node in nodes[:max_listed_nodes]:
        value = "" if values is None else f", {values[node]:+.3f}"
        parts.append(f"{node}(d={graph.degree(node)}{value})")

    return ", ".join(parts)


class _Perturbed:
    """
    Multiplies every transmission probability by lognormal noise for the
    duration of one rollout, on whichever evaluator is bound: the world-model
    and oracle environments read `base_edges`, the sampler reads `graph.ic_probs`.
    """

    def __init__(self, environment: object, sigma: float) -> None:
        self.environment = environment
        self.sigma = sigma
        self.saved = None

    def __enter__(self) -> "_Perturbed":
        rng = np.random.default_rng(robust_seed)
        if hasattr(self.environment, "base_edges"):
            self.saved = self.environment.base_edges
            self.environment.base_edges = {
                arc: float(min(1.0, weight * np.exp(rng.normal(0.0, self.sigma))))
                for arc, weight in self.saved.items()
            }
        else:
            graph = self.environment.graph
            self.saved = graph.ic_probs
            graph.ic_probs = np.minimum(
                1.0,
                graph.ic_probs * np.exp(rng.normal(0.0, self.sigma, size=graph.ic_probs.shape)),
            ).astype(graph.ic_probs.dtype)

        return self

    def __exit__(self, *_exc: object) -> bool:
        if hasattr(self.environment, "base_edges"):
            self.environment.base_edges = self.saved
        else:
            self.environment.graph.ic_probs = self.saved

        return False


def _answer_intervention(
    operation: str, probe: dict, plan: list, task, graph, environment
) -> tuple[str, int]:
    """(answer, rollouts) for one intervention-plan probe."""
    horizon, budget = task.horizon, task.budget
    budgeted = _budgeted(plan, task)

    if operation == "drop":
        node = int(probe["node"])
        base, dropped = _plan_rewards(environment, [plan, _drop(plan, node)], horizon, budget)
        return (
            f"drop({node}): plan={base:.2f}, without={dropped:.2f}, "
            f"contribution={base - dropped:+.2f}",
            2,
        )

    if operation == "swap":
        old, new = int(probe["a"]), int(probe["b"])
        base, swapped = _plan_rewards(environment, [plan, _swap(plan, old, new)], horizon, budget)
        return (
            f"swap({old}->{new}): plan={base:.2f}, swapped={swapped:.2f}, "
            f"delta={swapped - base:+.2f}",
            2,
        )

    if operation == "region":
        nodes = [int(node) for node in probe["nodes"]][:max_region_nodes]
        trajectory = environment.rollout(partial(planned_action, plan), horizon, budget)
        mass = float(sum(trajectory.final_marginals[node] for node in nodes))
        return (
            f"region({len(nodes)} nodes): expected mass captured="
            f"{mass:.2f} ({mass / max(len(nodes), 1):.0%} of the region)",
            1,
        )

    if operation == "add":
        node = int(probe["node"])
        if node in budgeted:
            return f"add({node}): that node is already in the plan; use drop or swap", 0
        base, extended = _plan_rewards(
            environment, [plan, _add(plan, node, task, graph)], horizon, budget
        )
        return (
            f"add({node}) at budget {budget + 1}: plan={base:.2f}, with it={extended:.2f}, "
            f"marginal gain={extended - base:+.2f} (d={graph.degree(node)})",
            2,
        )

    if operation == "best_swap":
        old = int(probe["node"])
        if old not in budgeted:
            return f"best_swap({old}): that node is not in the plan", 0
        excluded = set(budgeted) | set(int(node) for node in task.outbreak)
        limit = (
            max_best_swap_candidates
            if batched_environment(environment)
            else max_best_swap_sequential
        )
        candidates = sorted(
            (node for node in range(graph.num_nodes) if node not in excluded),
            key=lambda node: -graph.degree(node),
        )[:limit]
        if not candidates:
            return f"best_swap({old}): no replacement candidates", 0
        rewards = _plan_rewards(
            environment,
            [plan] + [_swap(plan, old, new) for new in candidates],
            horizon,
            budget,
        )
        base, swapped = rewards[0], rewards[1:]
        sign = -1.0 if task.sense == "minimize" else 1.0
        order = sorted(range(len(candidates)), key=lambda index: -sign * swapped[index])
        best = order[0]
        runner = order[1] if len(order) > 1 else None
        text = (
            f"best_swap({old}) over the {len(candidates)} highest-degree unselected "
            f"nodes: plan={base:.2f}; best replacement {candidates[best]}"
            f"(d={graph.degree(candidates[best])}) gives {swapped[best]:.2f} "
            f"(delta {swapped[best] - base:+.2f})"
        )
        if runner is not None:
            text += (
                f"; runner-up {candidates[runner]} gives {swapped[runner]:.2f} "
                f"(delta {swapped[runner] - base:+.2f})"
            )
        return text, 1 + len(candidates)

    if operation == "overlap":
        a, b = int(probe["a"]), int(probe["b"])
        if task.contains or task.blocks or task.immunizes:
            return (
                f"overlap({a},{b}): defined for seeding tasks only (the expected "
                f"nodes two seeds both reach); use drop on each instead",
                0,
            )
        solo_a, solo_b = _node_actions(plan, a), _node_actions(plan, b)
        if not solo_a or not solo_b:
            return f"overlap({a},{b}): both nodes must be in the plan", 0
        pair = _plan_rewards(
            environment, [[solo_a], [solo_b], [solo_a + solo_b]], horizon, budget
        )
        overlap = pair[0] + pair[1] - pair[2]
        return (
            f"overlap({a},{b}): alone {pair[0]:.2f} and {pair[1]:.2f}, together "
            f"{pair[2]:.2f}; expected nodes both reach={overlap:.2f} "
            f"({overlap / max(min(pair[0], pair[1]), 1e-9):.0%} of the smaller cascade)",
            3,
        )

    if operation == "horizon":
        steps = int(probe["t"])
        steps = max(1, min(steps, max_horizon_multiple * horizon))
        trajectory = environment.rollout(partial(planned_action, plan), steps, budget)
        curve = trajectory.spread_curve or []
        at_task = curve[min(horizon + 1, len(curve) - 1)] if curve else trajectory.reward
        return (
            f"horizon({steps}): expected spread {at_task:.2f} at the task horizon "
            f"{horizon} and {trajectory.reward:.2f} at {steps}; curve="
            f"{[round(value, 1) for value in curve]}",
            1,
        )

    if operation == "frontier":
        step = int(probe["t"])
        if not hasattr(environment, "capture_frontier_step"):
            return f"frontier({step}): this evaluator cannot capture a frontier", 0
        environment.capture_frontier_step = step
        environment.rollout(partial(planned_action, plan), horizon, budget)
        marginals = environment.last_frontier_marginals
        if marginals is None:
            return f"frontier({step}): step {step} is past the rollout's horizon", 1
        ranked = [int(node) for node in np.argsort(-marginals) if marginals[node] > 0]
        values = {node: float(marginals[node]) for node in ranked[:max_listed_nodes]}
        return (
            f"frontier({step}): expected frontier size after step {step}="
            f"{float(marginals.sum()):.2f}; most likely spreaders: "
            f"{_named(ranked, graph, values)}",
            1,
        )

    if operation == "robust":
        mode = probe.get("mode", "perturb")
        base = _plan_rewards(environment, [plan], horizon, budget)[0]
        if mode == "hidden":
            if not hasattr(environment, "hide_edge_weights"):
                return "robust(hidden): only a forward-model evaluator can hide its weights", 1
            saved = environment.hide_edge_weights
            environment.hide_edge_weights = True
            try:
                blind = _plan_rewards(environment, [plan], horizon, budget)[0]
            finally:
                environment.hide_edge_weights = saved
            return (
                f"robust(hidden): plan={base:.2f} with the true transmission "
                f"probabilities, {blind:.2f} with them hidden from the model "
                f"(delta {blind - base:+.2f})",
                2,
            )
        sigma = float(probe.get("sigma", default_perturbation))
        with _Perturbed(environment, sigma):
            noisy = _plan_rewards(environment, [plan], horizon, budget)[0]
        return (
            f"robust(sigma={sigma:.2f}): plan={base:.2f} on the true probabilities, "
            f"{noisy:.2f} with every p(u->v) multiplied by exp(N(0, {sigma:.2f})) "
            f"(delta {noisy - base:+.2f})",
            2,
        )

    if operation == "gradient":
        if not hasattr(environment, "seed_gradient"):
            return "gradient: only a differentiable forward model can answer this", 0
        if task.budget_op != "add_node":
            return "gradient: supported for the add_node lever only", 0
        top = int(probe.get("top", max_listed_nodes))
        report = environment.seed_gradient(plan, horizon, top=top)
        best = ", ".join(
            f"{node}(d={graph.degree(node)}, {value:+.3f})"
            for node, value in report["best_unselected"]
        )
        worst = ", ".join(
            f"{node}(d={graph.degree(node)}, {value:+.3f})"
            for node, value in report["worst_selected"]
        )
        return (
            f"gradient: d(expected spread)/d(seed) by one mean-field rollout "
            f"(mean-field spread {report['mean_field_spread']:.2f}); highest-gradient "
            f"UNSELECTED nodes: {best}; lowest-gradient SELECTED nodes: {worst}",
            1,
        )

    raise KeyError(operation)


def _answer_inverse(
    operation: str, probe: dict, trajectory, task, graph, environment
) -> tuple[str, int]:
    """(answer, rollouts) for a localization or reconstruction probe."""
    instances = list(task.instances)
    episode = int(probe.get("episode", 0))
    if not 0 <= episode < len(instances):
        return f"{operation}: episode index {episode} is out of range (0..{len(instances) - 1})", 0
    instance = instances[episode]

    if operation == "resimulate":
        nodes = [int(node) for node in probe["nodes"]]
        resimulated = ForwardOracle(environment=environment, task=task)(nodes)
        score = consistency_score(resimulated, instance.observation)
        residual = residual_diagnostics(resimulated, instance.observation, graph)
        over = [entry[0] for entry in residual["over"]][:max_listed_nodes]
        under = [entry[0] for entry in residual["under"]][:max_listed_nodes]
        return (
            f"resimulate({nodes}) on episode {instance.episode_id}: consistency="
            f"{score:.4f} (0 is perfect); over-explained {residual['n_over']} nodes "
            f"e.g. {over}; under-explained {residual['n_under']} nodes e.g. {under}",
            1,
        )

    if operation == "transmission":
        source, target, step = int(probe["u"]), int(probe["v"]), int(probe["t"])
        rows = trajectory.cost.get("per_instance") or []
        if episode >= len(rows) or "decoded" not in rows[episode]:
            return f"transmission: no decoded history for episode {episode}", 0
        decoded = rows[episode]["decoded"]
        infected = [int(node) for node, (when, _) in decoded.items() if int(when) <= step]
        frontier = [int(node) for node, (when, _) in decoded.items() if int(when) == step]
        probabilities = StepOracle(environment=environment)(infected, frontier)
        likely = [
            int(node)
            for node in np.argsort(-probabilities)
            if probabilities[node] > 0 and int(node) not in set(infected)
        ][:max_listed_nodes]
        where = "on" if source in set(frontier) else "NOT on"
        return (
            f"transmission({source}->{target} at t={step}) on episode "
            f"{instance.episode_id}: under your decoded history {source} is {where} "
            f"the frontier at t={step}; the kernel gives P({target} activates at "
            f"t={step + 1})={float(probabilities[target]):.4f}; the kernel's likeliest "
            f"activations at t={step + 1}: "
            f"{[(node, round(float(probabilities[node]), 3)) for node in likely]}",
            1,
        )

    raise KeyError(operation)


def execute_probes(
    environment: object,
    trajectory: object | None,
    probes: list,
    task,
    graph,
    enabled: bool = True,
    note: str | None = None,
    about: str = "your last evaluated plan",
) -> tuple[str | None, list[dict]]:
    """
    Answer one probe request: (feedback text, log entries).

    `trajectory` is the evaluation the questions are about: its `actions` are
    the bags that plan actually ran, and on the inverse families its
    `per_instance` rows carry the decoded histories a transmission probe reads.
    """
    if not probes and not note:
        return None, []

    if not enabled:
        return (
            "PROBE RESULTS: probes are unavailable under the native condition: "
            "no forward model may be consulted during this search.",
            [],
        )

    if trajectory is None:
        return ("PROBE RESULTS: no evaluated plan exists yet to probe against.", [])

    ops = available_ops(task)
    plan = list(trajectory.actions or [])
    lines = [f"PROBE RESULTS (answered by this arm's evaluator, about {about}):"]
    if note:
        lines.append(f"  note: {note}")

    entries = []
    for probe in probes:
        start = time.perf_counter()
        before = float(getattr(environment, "evaluator_seconds", 0.0))
        rollouts = 0

        try:
            operation = probe.get("op") if isinstance(probe, dict) else None
            if operation not in ops:
                answer = (
                    f"unknown or unavailable probe op {operation!r}: this task answers "
                    f"{', '.join(ops) or 'no probes'}"
                )
            elif operation in intervention_ops:
                if not plan:
                    answer = f"{operation}: the evaluated plan is empty"
                else:
                    answer, rollouts = _answer_intervention(
                        operation, probe, plan, task, graph, environment
                    )
            else:
                answer, rollouts = _answer_inverse(
                    operation, probe, trajectory, task, graph, environment
                )
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
    trajectory: object | None,
    task,
    graph,
    enabled: bool,
    log: list,
    label: str = "agent",
    turn: str = "write",
    about: str = "your last evaluated plan",
) -> str | None:
    """Run one probe request and append its entries, stamped with the turn, to `log`."""
    if not probe_request and not probe_note:
        return None

    text, entries = execute_probes(
        environment,
        trajectory,
        probe_request,
        task,
        graph,
        enabled=enabled,
        note=probe_note,
        about=about,
    )
    for entry in entries:
        entry["turn"] = turn
    log += entries

    if entries:
        tqdm.write(f"[{label}] answered {len(entries)} probes ({turn} turn)")

    return text
