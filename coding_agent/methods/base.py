"""OuterLoopMethod contract + shared helpers."""

from functools import partial
from typing import Protocol

from coding_agent.agent import CodingAgent
from coding_agent.credit import planned_action
from coding_agent.executor import StrategyError, validate_actions
from coding_agent.tools.algorithms import degree_discount
from coding_agent.types import ActionOp, GraphInfo, Strategy, TaskSpec, Trajectory

# Ensemble P(infected) below this counts as "unreached" in feedback
unreached_threshold = 0.10
# ...and above this as decisively reached. The gap between the two keeps the
# reference diff from reporting nodes that merely straddle one threshold
reached_threshold = 0.50
max_listed_nodes = 20
max_listed_diff_nodes = 10


def summarize(trajectory: Trajectory, graph: GraphInfo | None = None) -> str:
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

    if graph is not None and trajectory.final_marginals is not None:
        # Nodes the cascade rarely reaches, hubs first — where spread is being lost
        unreached = [
            node
            for node, frequency in enumerate(trajectory.final_marginals)
            if frequency < unreached_threshold
        ]
        unreached.sort(key=graph.degree, reverse=True)
        listed = ", ".join(
            f"{node}(d={graph.degree(node)})" for node in unreached[:max_listed_nodes]
        )
        overflow = (
            f", … and {len(unreached) - max_listed_nodes} more"
            if len(unreached) > max_listed_nodes
            else ""
        )
        lines.append(
            f"unreached nodes (P(infected)<{unreached_threshold:.0%} across the "
            f"ensemble): {len(unreached)}/{graph.num_nodes} — top by degree: "
            f"{listed}{overflow}"
        )

    if graph is not None:
        seeds = sorted(
            {
                action.target
                for bag in trajectory.actions
                for action in bag
                if action.op == "add_node"
            }
        )
        adjacent_pairs = [
            (first, second)
            for index, first in enumerate(seeds)
            for second in seeds[index + 1 :]
            if second in graph.out_neighbors(first)
            or first in graph.out_neighbors(second)
        ]
        if adjacent_pairs:
            lines.append(
                f"adjacent seed pairs (overlapping neighborhoods, likely redundant "
                f"budget): {adjacent_pairs}"
            )

    return "\n".join(lines)


def validate_plan(plan: list, task: TaskSpec, graph: GraphInfo) -> None:
    """Validate every bag and the whole-plan seed total against the task budget."""
    total_adds = 0
    for bag in plan:
        validate_actions(bag, graph.num_nodes, task.budget, task.allowed_ops)
        total_adds += sum(1 for action in bag if action.op == "add_node")

    if total_adds > task.budget:
        raise StrategyError(
            f"plan adds {total_adds} seeds in total, exceeds budget {task.budget}"
        )


def baseline_anchor(
    environment: object, task: TaskSpec, graph: GraphInfo
) -> tuple[str, Trajectory]:
    """
    One degree_discount rollout in the same env — a concrete score to beat.

    The trajectory is returned as well as the text: its per-node marginals are
    what reference_diff() compares against, which costs no further rollouts.
    """
    seeds = degree_discount(
        graph, task.budget, task.diffusion_model, horizon=task.horizon
    )
    plan = [[ActionOp("add_node", node) for node in seeds]] + [
        [] for _ in range(task.horizon)
    ]
    trajectory = environment.rollout(
        partial(planned_action, plan), task.horizon, task.budget
    )
    reward_se = trajectory.cost.get("reward_se", 0.0)

    text = (
        f"REFERENCE: classical degree_discount scores "
        f"{trajectory.reward:.2f} (±{reward_se:.2f} SE) on this graph under this "
        f"evaluator. Your strategy should beat it."
    )

    return text, trajectory


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
        f"REFERENCE DIFF (your cascade vs {reference_name}'s, same evaluator, "
        f"per-node P(infected)). Listed nodes are DECISIVE flips only "
        f"(one side >= {reached_threshold:.0%}, the other < {unreached_threshold:.0%}); "
        f"the net line below sums every node, so it is larger:"
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
