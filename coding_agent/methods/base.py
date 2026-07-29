"""OuterLoopMethod contract + shared helpers."""

import math
from functools import partial
from typing import Protocol

from coding_agent.agent import CodingAgent
from coding_agent.credit import planned_action
from coding_agent.executor import StrategyError, validate_actions
from coding_agent.tools import algorithms, primitives
from coding_agent.types import ActionOp, GraphInfo, Strategy, TaskSpec, Trajectory

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


def summarize(
    trajectory: Trajectory, graph: GraphInfo | None = None, task: TaskSpec | None = None
) -> str:
    reward_se = trajectory.cost.get("reward_se", 0.0)
    frontier_counts = [len(state.frontier) for state in trajectory.states[1:]]

    lines = [
        f"final_spread={trajectory.reward:.2f} (±{reward_se:.2f} SE), "
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
    trajectory: Trajectory, incumbent: Trajectory, incumbent_label: str = "your best so far"
) -> str:
    """
    Signed change against the incumbent, with the noise band that decides it.

    Both rollouts run at the same seed, so the realizations are shared and the
    difference is far better resolved than either absolute number — but on a
    hub-dominated graph the whole algorithmic spread can still sit inside this
    band, and the model needs to be told that rather than chase it.
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
    elif delta > 0:
        verdict = "a real improvement — keep what caused it"
    else:
        verdict = "a real regression — undo what caused it"

    return (
        f"CHANGE vs {incumbent_label} ({incumbent.reward:.2f}): {delta:+.2f} nodes. "
        f"The 2-sigma noise band on this comparison is ±{band:.2f}, so this is {verdict}."
    )


def validate_plan(plan: list, task: TaskSpec, graph: GraphInfo) -> None:
    """Validate every bag and the whole-plan seed total against the task budget."""
    total_adds = 0
    seeded = set()

    for timestep, bag in enumerate(plan):
        validate_actions(bag, graph.num_nodes, task.budget, task.allowed_ops)

        for action in bag:
            if action.op != "add_node":
                continue

            # Re-seeding at a later timestep spends a second unit of budget on a
            # node that is already infected
            if int(action.target) in seeded:
                raise StrategyError(
                    f"plan seeds node {action.target} again at t={timestep}; it is "
                    f"already seeded earlier in the plan and the repeat spends "
                    f"budget without adding a node."
                )

            seeded.add(int(action.target))
            total_adds += 1

    if total_adds > task.budget:
        raise StrategyError(
            f"plan adds {total_adds} seeds in total, exceeds budget {task.budget}"
        )


def baseline_anchor(
    environment: object, task: TaskSpec, graph: GraphInfo
) -> tuple[str, Trajectory, str]:
    """
    One rollout per classical baseline in the same env — the table to top.

    Returns the leaderboard text, the BEST baseline's trajectory, and its name.
    The trajectory rides along because its per-node marginals are what
    reference_diff() compares against, which costs no further rollouts.
    """
    names = [
        name
        for name in anchor_algorithms
        if task.diffusion_model == "IC" or name not in ic_only_anchors
    ]
    scored = []

    for name in names:
        seeds = algorithms.algorithms[name](
            graph, task.budget, task.diffusion_model, horizon=task.horizon
        )
        plan = [[ActionOp("add_node", int(node)) for node in dict.fromkeys(seeds)]] + [
            [] for _ in range(task.horizon)
        ]
        trajectory = environment.rollout(
            partial(planned_action, plan), task.horizon, task.budget
        )
        scored.append((name, trajectory))

    scored.sort(key=lambda entry: entry[1].reward, reverse=True)
    best_name, best_trajectory = scored[0]

    lines = [
        "REFERENCE SCORES — classical baselines run on THIS graph, under THIS "
        "evaluator, at the same budget and horizon. Beating the top row is the bar:"
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
    reference: Trajectory,
    graph: GraphInfo,
    reference_name: str = "degree_discount",
) -> str | None:
    """
    Where the reference algorithm's cascade went that yours did not, and vice versa.

    Both marginal vectors already exist from rollouts that have been paid for, so
    this is the richest feedback available at zero additional cost.
    """
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

    if missed:
        lines.append(
            f"  it reaches {len(missed)} nodes you miss — top by degree: "
            f"{_diff_node_list(missed, mine, theirs, graph)}"
        )
    if gained:
        lines.append(
            f"  you reach {len(gained)} nodes it misses — top by degree: "
            f"{_diff_node_list(gained, mine, theirs, graph)}"
        )
    if not missed and not gained:
        lines.append("  no decisive flips either way")

    lines.append(
        f"  net expected spread vs the reference: "
        f"{sum(mine) - sum(theirs):+.2f} nodes"
    )

    return "\n".join(lines)


class OuterLoopMethod(Protocol):
    def optimize(
        self, agent: CodingAgent, environment: object, task: TaskSpec, graph: GraphInfo
    ) -> tuple[Strategy, Trajectory]: ...
