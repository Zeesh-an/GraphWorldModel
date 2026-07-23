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
max_listed_nodes = 20


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
) -> str:
    """One degree_discount rollout in the same env — a concrete score to beat."""
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

    return (
        f"REFERENCE: classical degree_discount scores "
        f"{trajectory.reward:.2f} (±{reward_se:.2f} SE) on this graph under this "
        f"evaluator. Your strategy should beat it."
    )


class OuterLoopMethod(Protocol):
    def optimize(
        self, agent: CodingAgent, environment: object, task: TaskSpec, graph: GraphInfo
    ) -> tuple[Strategy, Trajectory]: ...
