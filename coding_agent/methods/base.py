"""OuterLoopMethod contract + shared helpers."""

import math
import time
from dataclasses import replace
from functools import partial
from typing import Protocol

from coding_agent.agent import CodingAgent
from coding_agent.containment import build_outbreak, removal_plan, removal_set
from coding_agent.credit import planned_action
from coding_agent.executor import StrategyError, call_strategy, validate_actions
from coding_agent.rounds import adaptive_action_fn, round_batches, round_schedule
from coding_agent.stream import build_stream
from coding_agent.tools import algorithms, primitives
from coding_agent.tools.adaptive_algorithms import adaptive_algorithms
from coding_agent.tools.dismantling_algorithms import dismantling_algorithms
from coding_agent.types import (
    ActionOp,
    GraphInfo,
    Strategy,
    TaskSpec,
    Trajectory,
    improves,
    rank_by,
)

# Ensemble P(infected) below this counts as "unreached" in feedback
unreached_threshold = 0.10
# ...and above this as decisively reached. The gap between the two keeps the
# reference diff from reporting nodes that merely straddle one threshold
reached_threshold = 0.50
max_listed_nodes = 20
max_listed_diff_nodes = 10
max_listed_seeds = 40
max_listed_communities = 8

# One rollout per name at the start of a refinement loop: the score table the
# agent has to top, in the same evaluator it is being judged by. Under an MC
# evaluator these episodes are charged to the arm like any other, so keep the
# list short. RIS members are dropped under LT, where reverse-reachable sets do
# not describe the dynamics.
anchor_algorithms = (
    "high_degree",
    "degree_discount",
    "pagerank_seeds",
    "imm",
    "random_seeds",
)
ic_only_anchors = ("imm",)

# Added to the leaderboard only for an adaptive task. AdaptGreedy is deliberately
# absent: it costs batch x candidates x mc_runs simulations per round, which
# would dominate startup for a table that exists to set a bar, not to be the
# result. Run it as its own --baselines arm when you want its number.
adaptive_anchor_algorithms = ("adapt_epic", "adapt_degree_discount", "static_split")

# The leaderboard for a CONTAINMENT task, which the IM anchors would be nonsense
# for: they return seed sets, and this planner spends its budget on removals.
# `adaptive_degree` heads the list because it is the row that actually has to be
# beaten (research/critical_node_detection.md §9.3 item 2). `greedy_blocking` is
# deliberately absent for the same reason `adapt_greedy` is — it costs
# k x candidates x mc_runs episodes and would dominate startup for a table that
# exists to set a bar. Run it as its own --baselines arm when you want its number.
dismantling_anchor_algorithms = (
    "adaptive_degree",
    "corehd",
    "collective_influence_removal",
    "netshield",
    "random_removal",
)

# Residual-gain feedback is built from reverse-reachable sets, so it is IC-only.
# theta is small relative to what IMM would ask for: this ranks candidates for a
# prompt, it does not select them.
rr_per_node = 10
rr_max_theta = 20_000
rr_max_nodes = 200_000
max_listed_gains = 10


def _communities(graph: GraphInfo) -> dict[int, int]:
    """{node: community_id}, cached on the graph — summarize() runs every turn."""
    if graph._community_labels is None:
        graph._community_labels = primitives.detect_communities(graph)

    return graph._community_labels


def _rr_covers(graph: GraphInfo, diffusion_model: str) -> tuple | None:
    """({node: RR-set indices it reaches}, theta), cached. None when not applicable."""
    if diffusion_model != "IC" or graph.num_nodes > rr_max_nodes:
        return None

    if graph._rr_covers is None:
        theta = min(rr_max_theta, rr_per_node * graph.num_nodes)
        covers = {}

        for index, rr_set in enumerate(
            primitives.batch_reverse_sample(graph, theta=theta, seed=0)
        ):
            for node in rr_set:
                covers.setdefault(int(node), set()).add(index)

        graph._rr_covers = (covers, theta)

    return graph._rr_covers


def _residual_gains(graph: GraphInfo, diffusion_model: str, seeds: list[int]) -> dict:
    """
    Estimated extra spread each node would add on top of `seeds`, in nodes.

    RR-set coverage not already covered by the seed set, scaled by N/theta. This
    is the marginal-gain signal CELF would compute, at RIS cost and without
    touching the metered evaluator.
    """
    covers_and_theta = _rr_covers(graph, diffusion_model)
    if covers_and_theta is None:
        return {}

    covers, theta = covers_and_theta
    covered = set()
    for seed in seeds:
        covered |= covers.get(int(seed), set())

    scale = graph.num_nodes / theta

    return {
        node: len(cover - covered) * scale
        for node, cover in covers.items()
        if node not in seeds
    }


def _seed_lines(seeds: list[int], graph: GraphInfo) -> list[str]:
    labels = _communities(graph)
    listed = ", ".join(
        f"{node}(d={graph.degree(node)},c={labels.get(node, -1)})"
        for node in seeds[:max_listed_seeds]
    )
    overflow = (
        f", … and {len(seeds) - max_listed_seeds} more"
        if len(seeds) > max_listed_seeds
        else ""
    )

    return [
        f"seeds you chose ({len(seeds)}; d=total degree, c=community id): "
        f"{listed}{overflow}"
    ]


def _community_lines(
    seeds: list[int], marginals: list[float] | None, graph: GraphInfo
) -> list[str]:
    labels = _communities(graph)
    members = {}
    for node in range(graph.num_nodes):
        members.setdefault(labels.get(node, -1), []).append(node)

    seeds_per_community = {}
    for node in seeds:
        community_id = labels.get(node, -1)
        seeds_per_community[community_id] = seeds_per_community.get(community_id, 0) + 1

    ranked = sorted(members, key=lambda key: len(members[key]), reverse=True)
    lines = [
        f"community coverage ({len(members)} communities, largest "
        f"{max_listed_communities} shown; reach = mean P(infected) over members):"
    ]

    for community_id in ranked[:max_listed_communities]:
        group = members[community_id]
        reach = (
            f"{sum(marginals[node] for node in group) / len(group):.0%}"
            if marginals is not None
            else "n/a"
        )
        lines.append(
            f"  c{community_id}: {seeds_per_community.get(community_id, 0)} seeds "
            f"/ {len(group)} nodes, reach {reach}"
        )

    unseeded = [
        community_id
        for community_id in ranked
        if community_id not in seeds_per_community and len(members[community_id]) > 1
    ]
    if unseeded:
        lines.append(
            f"  {len(unseeded)} communities got no seed at all "
            f"(largest: {[len(members[key]) for key in unseeded[:5]]} nodes)"
        )

    return lines


def _gain_lines(gains: dict, graph: GraphInfo) -> list[str]:
    if not gains:
        return []

    ranked = sorted(gains, key=lambda node: gains[node], reverse=True)
    listed = ", ".join(
        f"{node}(d={graph.degree(node)}, +{gains[node]:.1f})"
        for node in ranked[:max_listed_gains]
        if gains[node] > 0
    )

    if not listed:
        return ["no unseeded node has meaningful residual gain — the seed set already covers the reachable graph"]

    return [
        f"highest-value nodes you did NOT seed (estimated extra spread in nodes if "
        f"added on top of your seed set, from reverse-reachable sets): {listed}"
    ]


def _containment_lines(
    trajectory: Trajectory, graph: GraphInfo, task: TaskSpec
) -> list[str]:
    """
    What the removal set did, in the terms a blocker actually reasons about.

    `summarize`'s seed-centric diagnostics (residual gain, community coverage,
    unreached nodes) all describe where a cascade SHOULD go next, which is the
    wrong question here: the removals are what to explain, and the outbreak is
    fixed. This replaces them rather than adding to them.
    """
    removed = removal_set(trajectory.actions)
    outbreak = list(task.outbreak)
    lines = []

    if outbreak:
        listed = ", ".join(
            f"{node}(d={graph.degree(node)})" for node in outbreak[:max_listed_nodes]
        )
        lines.append(
            f"OUTBREAK (fixed, not yours to choose; {len(outbreak)} sources): {listed}"
        )

    if removed:
        listed = ", ".join(
            f"{node}(d={graph.degree(node)})" for node in removed[:max_listed_seeds]
        )
        overflow = (
            f", … and {len(removed) - max_listed_seeds} more"
            if len(removed) > max_listed_seeds
            else ""
        )
        lines.append(
            f"nodes you removed ({len(removed)} of budget {task.budget}, in removal "
            f"order): {listed}{overflow}"
        )

        # Distance from the outbreak is the containment-specific diagnostic: a
        # blocker the cascade never reaches did nothing, and there is no way to
        # see that from the spread number alone
        reachable = set(outbreak)
        frontier = set(outbreak)
        rings = {}
        for hop in range(1, 4):
            frontier = {
                neighbour
                for node in frontier
                for neighbour in graph.out_neighbors(node) + graph.in_neighbors(node)
            } - reachable
            reachable |= frontier
            rings[hop] = len({node for node in removed if node in frontier})

        far = len([node for node in removed if node not in reachable])
        lines.append(
            "removal distance from the outbreak: "
            + ", ".join(f"{count} at {hop} hop(s)" for hop, count in rings.items())
            + f", {far} more than 3 hops away (those can only help if the cascade "
            f"reaches that far)"
        )

    if trajectory.final_marginals is not None:
        infected = [
            node
            for node, frequency in enumerate(trajectory.final_marginals)
            if frequency >= reached_threshold
        ]
        infected.sort(key=graph.degree, reverse=True)
        listed = ", ".join(
            f"{node}(d={graph.degree(node)}, P={trajectory.final_marginals[node]:.2f})"
            for node in infected[:max_listed_nodes]
        )
        lines.append(
            f"nodes the cascade still reaches (P(infected)>={reached_threshold:.0%}): "
            f"{len(infected)}/{graph.num_nodes} — top by degree: {listed}"
        )

    return lines


def summarize(
    trajectory: Trajectory, graph: GraphInfo | None = None, task: TaskSpec | None = None
) -> str:
    reward_se = trajectory.cost.get("reward_se", 0.0)
    frontier_counts = [len(state.frontier) for state in trajectory.states[1:]]
    label = (
        "final_infected (LOWER IS BETTER)"
        if task is not None and task.contains
        else "final_spread"
    )

    lines = [
        f"{label}={trajectory.reward:.2f} (±{reward_se:.2f} SE), "
        f"steps={len(trajectory.infected_counts)}, "
        f"counts={[round(count, 1) for count in trajectory.infected_counts]}"
    ]
    lines.append(f"frontier_counts={frontier_counts}")

    if frontier_counts and frontier_counts[-1] == 0:
        death_step = len(frontier_counts) - 1 - frontier_counts[::-1].index(0)
        lines.append(
            f"cascade dead by t={death_step} — actions scheduled after that did nothing"
        )

    if graph is None:
        return "\n".join(lines)

    if task is not None and task.contains:
        return "\n".join(lines + _containment_lines(trajectory, graph, task))

    seeds = sorted(
        {
            action.target
            for bag in trajectory.actions
            for action in bag
            if action.op == "add_node"
        }
    )
    diffusion_model = task.diffusion_model if task is not None else "IC"
    gains = _residual_gains(graph, diffusion_model, seeds)

    if seeds:
        lines += _seed_lines(seeds, graph)

    adjacent_pairs = [
        (first, second)
        for index, first in enumerate(seeds)
        for second in seeds[index + 1 :]
        if second in graph.out_neighbors(first) or first in graph.out_neighbors(second)
    ]
    if adjacent_pairs:
        lines.append(
            f"adjacent seed pairs (overlapping neighborhoods, likely redundant "
            f"budget): {adjacent_pairs}"
        )

    if trajectory.final_marginals is not None:
        unreached = [
            node
            for node, frequency in enumerate(trajectory.final_marginals)
            if frequency < unreached_threshold
        ]
        # Ranked by residual gain, not degree: a high-degree node the cascade
        # never reaches is usually unreachable, whereas a high-gain one is the
        # node actually worth spending a seed on
        unreached.sort(key=lambda node: gains.get(node, float(graph.degree(node))), reverse=True)
        listed = ", ".join(
            f"{node}(d={graph.degree(node)}"
            + (f", +{gains[node]:.1f}" if node in gains else "")
            + ")"
            for node in unreached[:max_listed_nodes]
        )
        overflow = (
            f", … and {len(unreached) - max_listed_nodes} more"
            if len(unreached) > max_listed_nodes
            else ""
        )
        lines.append(
            f"unreached nodes (P(infected)<{unreached_threshold:.0%} across the "
            f"ensemble): {len(unreached)}/{graph.num_nodes} — top by estimated gain: "
            f"{listed}{overflow}"
        )

    if seeds:
        lines += _community_lines(seeds, trajectory.final_marginals, graph)

    lines += _gain_lines(gains, graph)

    return "\n".join(lines)


def paired_delta(
    trajectory: Trajectory,
    incumbent: Trajectory,
    incumbent_label: str = "your best so far",
    sense: str = "maximize",
) -> str:
    """
    Signed change against the incumbent, with the noise band that decides it.

    Both rollouts run at the same seed, so the realizations are shared and the
    difference is far better resolved than either absolute number — but on a
    hub-dominated graph the whole algorithmic spread can still sit inside this
    band, and the model needs to be told that rather than chase it.

    `sense` decides what a negative delta MEANS. Under containment fewer infected
    nodes is the win, so an unflipped verdict would coach the model to undo every
    improvement it makes.
    """
    delta = trajectory.reward - incumbent.reward
    band = 2.0 * math.sqrt(
        trajectory.cost.get("reward_se", 0.0) ** 2
        + incumbent.cost.get("reward_se", 0.0) ** 2
    )

    if abs(delta) <= band:
        verdict = (
            "INSIDE THE NOISE — this change did nothing measurable, so do not "
            "read anything into its sign"
        )
    elif improves(trajectory.reward, incumbent.reward, sense):
        verdict = "a real improvement — keep what caused it"
    else:
        verdict = "a real regression — undo what caused it"

    direction = "fewer is better" if sense == "minimize" else "more is better"

    return (
        f"CHANGE vs {incumbent_label} ({incumbent.reward:.2f}): {delta:+.2f} nodes "
        f"({direction}). The 2-sigma noise band on this comparison is ±{band:.2f}, "
        f"so this is {verdict}."
    )


def validate_plan(plan: list, task: TaskSpec, graph: GraphInfo) -> None:
    """Validate every bag and the whole-plan budgeted total against the task budget."""
    total_units = 0
    targeted = set()
    noun = "seeds" if task.budget_op == "add_node" else "removes"

    for timestep, bag in enumerate(plan):
        validate_actions(
            bag,
            graph.num_nodes,
            task.budget,
            task.allowed_ops,
            task.budget_op,
            task.outbreak,
        )

        for action in bag:
            if action.op != task.budget_op:
                continue

            # Re-targeting at a later timestep spends a second unit of budget on a
            # node the plan already acted on
            if int(action.target) in targeted:
                raise StrategyError(
                    f"plan {noun} node {action.target} again at t={timestep}; it is "
                    f"already targeted earlier in the plan and the repeat spends "
                    f"budget without changing anything."
                )

            targeted.add(int(action.target))
            total_units += 1

    if total_units > task.budget:
        raise StrategyError(
            f"plan {noun} {total_units} nodes in total, exceeds budget {task.budget}"
        )


def attach_context(strategy: Strategy, task: TaskSpec) -> Strategy:
    """
    Hand the strategy what it needs from the task that its signature cannot carry.

    Attributes rather than a signature change because `plan_horizon(graph, budget,
    horizon)` and `act(state, graph, timestep)` are the contract every existing
    method, exemplar and checkpoint is written against. `outbreak` is empty for a
    seeding task, so a generated script may read it unconditionally; `budget_op`
    is what the SCORED harness emits, which is `add_node` for seeding and
    `remove_node` for containment — hardcoding it made every scored-mode
    containment arm fail validation before it was ever scored.
    """
    strategy.outbreak = tuple(int(node) for node in task.outbreak)
    strategy.budget_op = task.budget_op

    return strategy


def wrap_exogenous(
    action_fn, task: TaskSpec, graph: GraphInfo, stream: object = None
):
    """
    Layer everything the environment must see but the policy is not charged for.

    Every path that builds an ActionFn goes through here — `evaluate_strategy`,
    `per_step`, `windowed` — because a rollout that skips it faces no outbreak and
    no edit stream, and would post a number that looks like a win against arms
    that did face both.

    Applied AFTER validation, deliberately: the outbreak's `add_node` ops would be
    rejected under `--allowed-ops remove_node`, the deletion bag's `remove_edge`
    ops are the mechanics of one removal rather than a second intervention, and
    the stream's edits are exogenous by definition.
    """
    outbreak = build_outbreak(graph, task)

    if outbreak is not None:
        action_fn = outbreak.wrap(action_fn)

    if stream is not None:
        action_fn = stream.wrap(action_fn)

    return action_fn


def evaluate_strategy(
    strategy: Strategy, environment: object, task: TaskSpec, graph: GraphInfo
) -> tuple[Trajectory, float]:
    """
    Score one generated program, and return how long building its plan took.

    The single point where adaptive and non-adaptive diverge. Non-adaptive calls
    plan_horizon() once up front and replays the result; adaptive calls act() at
    each round boundary against the state the previous round produced. Everything
    else about the search (the population, the feedback, the checkpoints) is
    identical, which is what makes the adaptivity gap an A/B on one variable.
    """
    start = time.perf_counter()
    attach_context(strategy, task)
    # Built once here and applied to EVERY arm: an arm whose graph moved against
    # one whose graph did not would be measuring the stream, not the method
    stream = build_stream(graph, task.horizon, task.edit_rate, task.seed)

    if task.adaptive:
        batches = round_batches(task.budget, task.rounds, task.per_round_budget)
        # act() runs inside the rollout, so there is no plan to time here; the
        # policy's own compute lands in the environment's rollout_seconds
        action_fn = adaptive_action_fn(strategy, task, graph, batches, stream)
    else:
        plan = call_strategy(strategy.plan_horizon, graph, task.budget, task.horizon)
        validate_plan(plan, task, graph)
        action_fn = partial(planned_action, plan)

    action_fn = wrap_exogenous(action_fn, task, graph, stream)

    plan_seconds = time.perf_counter() - start
    trajectory = environment.rollout(action_fn, task.horizon, task.budget)

    return trajectory, plan_seconds


class _PlanAnchor:
    """
    Wraps a static selector as the plan_horizon()-shaped object evaluate_strategy wants.

    Anchors go through `evaluate_strategy` rather than calling `environment.rollout`
    directly so they meet the arm they are setting a bar for under IDENTICAL
    conditions: same outbreak, same edit stream, same removal expansion. An anchor
    rolled out bare would face no outbreak at all on a containment task and post an
    unbeatable zero.
    """

    def __init__(self, selector, task: TaskSpec, graph: GraphInfo) -> None:
        self.selector = selector
        self.task = task
        self.graph = graph
        self.source_script = ""

    def plan_horizon(
        self, graph: GraphInfo, budget: int, horizon: int
    ) -> list[list[ActionOp]]:
        nodes = self.selector(
            graph,
            budget,
            self.task.diffusion_model,
            horizon=horizon,
            outbreak=self.task.outbreak,
        )

        if self.task.contains:
            # Filters the outbreak's sources and tops the set back up, which is
            # what keeps a published dismantler runnable without rewriting it to
            # know an outbreak exists
            return removal_plan(
                nodes, graph, budget, self.task.outbreak, horizon
            )

        return [
            [
                ActionOp(self.task.budget_op, int(node))
                for node in dict.fromkeys(int(node) for node in nodes)
            ]
        ] + [[] for _ in range(horizon)]


class _AdaptiveAnchor:
    """Wraps a per-round policy as the act()-shaped object evaluate_strategy wants."""

    def __init__(self, policy, task: TaskSpec) -> None:
        self.policy = policy
        self.task = task
        self.source_script = ""

    def act(self, state, graph: GraphInfo, timestep: int) -> list[ActionOp]:
        batches = round_batches(
            self.task.budget, self.task.rounds, self.task.per_round_budget
        )
        batch = round_schedule(batches, self.task.round_gap, self.task.horizon).get(
            timestep, 0
        )
        if not batch:
            return []

        return [
            ActionOp("add_node", int(node))
            for node in self.policy(
                state,
                graph,
                batch,
                self.task.diffusion_model,
                total_budget=self.task.budget,
            )
        ]


def baseline_anchor(
    environment: object, task: TaskSpec, graph: GraphInfo
) -> tuple[str, Trajectory, str]:
    """
    One rollout per classical baseline in the same env — the table to top.

    Returns the leaderboard text, the BEST baseline's trajectory, and its name.
    The trajectory rides along because its per-node marginals are what
    reference_diff() compares against, which costs no further rollouts.
    """
    if task.contains:
        # A containment task's floor is the DISMANTLING library: the IM anchors
        # return seed sets, and this planner spends its budget on removals
        pool = dismantling_algorithms
        names = list(dismantling_anchor_algorithms)
    else:
        pool = algorithms.algorithms
        names = [
            name
            for name in anchor_algorithms
            if task.diffusion_model == "IC" or name not in ic_only_anchors
        ]

    scored = []

    # A static selector is a static PLAN, so it is scored against a non-adaptive
    # view of the task even when the task has rounds: `evaluate_strategy` would
    # otherwise route it through `adaptive_action_fn` and call an `act()` it does
    # not have. That is also the right reading — a one-shot algorithm dealt at t=0
    # is exactly the control an adaptive arm is measured against.
    static_task = replace(task, rounds=None, per_round_budget=None)

    for name in names:
        selector = _PlanAnchor(pool[name], task, graph)
        trajectory, _ = evaluate_strategy(selector, environment, static_task, graph)
        scored.append((name, trajectory))

    # Under an adaptive task the static table alone sets the wrong bar: it shows
    # what a one-shot algorithm gets and says nothing about what the published
    # ADAPTIVE algorithms get on the same rounds. AdaptGreedy is the number a
    # generated policy actually has to beat.
    # ...but only on a SEEDING task. The published adaptive policies return seed
    # sets, so on a containment task they emit an op the planner may not use and
    # every anchor rollout would fail validation.
    for name in adaptive_anchor_algorithms if task.adaptive and not task.contains else ():
        policy = _AdaptiveAnchor(adaptive_algorithms[name], task)
        trajectory, _ = evaluate_strategy(policy, environment, task, graph)
        scored.append((name, trajectory))

    scored = rank_by(scored, lambda entry: entry[1].reward, task.sense)
    best_name, best_trajectory = scored[0]

    if task.contains:
        kind = "classical network-dismantling baselines"
        goal = "LOWER is better here; beating the top row is the bar"
    else:
        kind = (
            "classical baselines, static and per-round adaptive,"
            if task.adaptive
            else "classical baselines"
        )
        goal = "beating the top row is the bar"

    lines = [
        f"REFERENCE SCORES: {kind} run on THIS graph, under THIS "
        f"evaluator, at the same budget and horizon. {goal}:"
    ]
    lines += [
        f"  {name:<18} {trajectory.reward:9.2f} "
        f"(±{trajectory.cost.get('reward_se', 0.0):.2f} SE)"
        for name, trajectory in scored
    ]

    return "\n".join(lines), best_trajectory, best_name


def _diff_node_list(
    nodes: list[int], mine: list[float], theirs: list[float], graph: GraphInfo
) -> str:
    listed = ", ".join(
        f"{node}(d={graph.degree(node)}, ref P={theirs[node]:.2f} vs yours "
        f"{mine[node]:.2f})"
        for node in nodes[:max_listed_diff_nodes]
    )
    overflow = (
        f", … and {len(nodes) - max_listed_diff_nodes} more"
        if len(nodes) > max_listed_diff_nodes
        else ""
    )

    return listed + overflow


def reference_diff(
    trajectory: Trajectory,
    reference: Trajectory | None,
    graph: GraphInfo,
    reference_name: str = "degree_discount",
    sense: str = "maximize",
) -> str | None:
    """
    Where the reference algorithm's cascade went that yours did not, and vice versa.

    Both marginal vectors already exist from rollouts that have been paid for, so
    this is the richest feedback available at zero additional cost.
    """
    # No anchor at all when use_anchor is False, which is every canned arm: the
    # baseline rollouts would be pure cost for a script that never reads a prompt
    if reference is None:
        return None

    mine, theirs = trajectory.final_marginals, reference.final_marginals
    if mine is None or theirs is None:
        return None

    missed = [
        node
        for node in range(graph.num_nodes)
        if theirs[node] >= reached_threshold and mine[node] < unreached_threshold
    ]
    gained = [
        node
        for node in range(graph.num_nodes)
        if mine[node] >= reached_threshold and theirs[node] < unreached_threshold
    ]
    missed.sort(key=graph.degree, reverse=True)
    gained.sort(key=graph.degree, reverse=True)

    lines = [
        f"REFERENCE DIFF (your cascade vs {reference_name}'s — the strongest "
        f"baseline on this graph — same evaluator, per-node P(infected)). Listed "
        f"nodes are DECISIVE flips only (one side >= {reached_threshold:.0%}, the "
        f"other < {unreached_threshold:.0%}); the net line below sums every node, "
        f"so it is larger:"
    ]

    # Same two sets, opposite readings: under containment the nodes the reference
    # reaches and you do not are the ones you successfully PROTECTED
    if sense == "minimize":
        their_reach = f"  it fails to protect {len(missed)} nodes you save"
        your_reach = f"  you lose {len(gained)} nodes it protects"
        net = "  net expected infections vs the reference (negative = you contain more)"
    else:
        their_reach = f"  it reaches {len(missed)} nodes you miss"
        your_reach = f"  you reach {len(gained)} nodes it misses"
        net = "  net expected spread vs the reference"

    if missed:
        lines.append(
            f"{their_reach} — top by degree: "
            f"{_diff_node_list(missed, mine, theirs, graph)}"
        )
    if gained:
        lines.append(
            f"{your_reach} — top by degree: "
            f"{_diff_node_list(gained, mine, theirs, graph)}"
        )
    if not missed and not gained:
        lines.append("  no decisive flips either way")

    lines.append(f"{net}: {sum(mine) - sum(theirs):+.2f} nodes")

    return "\n".join(lines)


class OuterLoopMethod(Protocol):
    def optimize(
        self, agent: CodingAgent, environment: object, task: TaskSpec, graph: GraphInfo
    ) -> tuple[Strategy, Trajectory]: ...
