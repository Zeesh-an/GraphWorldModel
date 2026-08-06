"""
Influence blocking: the four levers, the negative cascade, and prevented influence.

The harness piece `influence_blocking` needs and the other tasks do not. Three
things live here, and each one is a decision
`research/influence_blocking.md` forces rather than a convenience:

  * **The four levers (§1.1).** This literature splits cleanly by what the blocker is
    allowed to DO, and that split maps one-to-one onto our action ops:
    counter-seeding (`add_node`), node blocking (`remove_node`), edge blocking
    (`remove_edge`) and weight reduction (`set_edge_weight`). `--blocking-lever`
    picks one, and it sets `budget_op` and `allowed_ops` together so an arm can never
    be budgeted for one op and permitted another. §7 is why this is worth having at
    all: the seeding line and the structural line of this literature share no metric,
    so a single evaluator that scores all three interventions on one graph under one
    metric does not currently exist.

  * **The negative cascade.** `S_N` is an INPUT to the episode, not an action (§2.1),
    so it is committed by `CompetitiveSimulator.reset` rather than injected as a bag
, which is the whole reason `add_node` can keep meaning "seed positively" and the
    three action channels keep the meaning they have in every other task.

  * **Prevented influence (§8.1).** One quantity, five published names, all of them
    `sigma(S_N alone) - sigma(S_N | blockers)`. Budak's framing is the sharp one and
    is worth restating exactly: a blocker that protects nodes the rumour would never
    have reached scores ZERO. The reward the search optimizes is the remaining
    negative spread (lower is better, so no sign work is needed anywhere); prevented
    influence is that number subtracted from the unopposed reference, and it is what
    every table in this literature reports.

Warning: THE BUDGET AXIS IS NOT PERCENT OF N HERE (§8.2). Our `--budget-pcts 1 5 10 20`
speaks to no blocking paper at all: SandIMIN, Xie and TC-AIBM all report absolute
`k in {10..50}` or `{10..100}`, and the INFORMATIVE ratio is `|S_P| / |S_N|` rather
than `|S_P| / |V|`. The task registry therefore ships `default_budgets` instead of
`default_budget_pcts`, and `blocking_metrics` reports the ratio.
"""

from dataclasses import dataclass

import numpy as np

from coding_agent.containment import expand_removals
from coding_agent.types import ActionFn, ActionOp, GraphInfo, State, TaskSpec

# §1.1's table, as the (budget_op, allowed_ops) pair each row implies. The names are
# the paper-facing ones; the ops are ours.
counter_seed = "counter_seed"
node_block = "node_block"
edge_block = "edge_block"
weight_block = "weight_block"

blocking_levers = {
    counter_seed: ("add_node", ("add_node",)),
    node_block: ("remove_node", ("remove_node",)),
    edge_block: ("remove_edge", ("remove_edge",)),
    weight_block: ("set_edge_weight", ("set_edge_weight",)),
}

lever_papers = {
    counter_seed: "Budak WWW'11, He SDM'12 (CLDAG), Tong INFOCOM'17, NIE 2023, TC-AIBM 2025",
    node_block: "Xie ICDE'23, Wang PVLDB'24 (SandIMIN), Xie IJoC'25",
    edge_block: "Kimura AAAI'08, Khalil KDD'14, DiffIM AAAI'25",
    weight_block: "DiffIM AAAI'25 (its continuous relaxation IS set_edge_weight)",
}

valid_levers = tuple(blocking_levers)


def resolve_lever(lever: str) -> tuple[str, tuple]:
    """(budget_op, allowed_ops) for one lever."""
    if lever not in blocking_levers:
        raise ValueError(
            f"unknown blocking lever {lever!r}; choose one of {valid_levers}"
        )

    return blocking_levers[lever]


def lever_of(task: TaskSpec) -> str:
    """Which lever a task is configured for, read back from its budget op."""
    for name, (budget_op, _) in blocking_levers.items():
        if task.budget_op == budget_op:
            return name

    return counter_seed


def edge_weight_caps(graph: GraphInfo) -> dict:
    """
    `{(u, v): p}`, the transmission probability each arc starts at.

    The cap the weight lever is validated against: DiffIM's relaxation is
    `p~(u,v) = p(u,v) * r~(u,v)` with `r~ in [0, 1]` (§2.3), so a blocker may only
    REDUCE an edge. Letting it raise one would be an unbudgeted boost to the
    counter-cascade dressed up as a blocking action.
    """
    return {
        (int(graph.edge_index[0, edge]), int(graph.edge_index[1, edge])): float(
            graph.ic_probs[edge]
        )
        for edge in range(graph.edge_index.shape[1])
    }


@dataclass()
class NegativeCascade:
    """The rumour a blocking policy is trying to contain."""

    sources: tuple
    graph: GraphInfo
    # Budak's DETECTION DELAY r (§8.3): the bad campaign is detected r steps late and
    # the blocker only acts from then. 0 is the usual setting and the one every
    # published table uses; larger is the axis that makes the task non-trivial, and
    # it is the same statement as CLDAG's "first mover has a clear advantage".
    detection_delay: int = 0

    def wrap(self, action_fn: ActionFn) -> ActionFn:
        """
        Apply the detection delay and expand node removals, and nothing else.

        Deliberately NOT the containment wrapper's job of seeding the outbreak:
        `S_N` starts the NEGATIVE cascade, and an `add_node` bag would start the
        positive one, which is the opposite intervention. The simulator commits it in
        `reset` instead, which is what §2.1 means by "an input to the episode, not an
        action".

        Applied AFTER validation, for the same two reasons the containment wrapper
        is: a node deletion's incident `remove_edge` ops are the mechanics of one
        removal rather than a second intervention, and the delay is exogenous.
        """

        def blocking(state: State, timestep: int) -> list[ActionOp]:
            if timestep < self.detection_delay:
                return []

            return expand_removals(list(action_fn(state, timestep)), self.graph)

        return blocking


def build_negative_cascade(
    graph: GraphInfo, task: TaskSpec
) -> NegativeCascade | None:
    """None for every task that is not two-cascade."""
    if not task.competitive:
        return None

    return NegativeCascade(
        sources=tuple(int(node) for node in task.outbreak),
        graph=graph,
        detection_delay=task.detection_delay,
    )


def blocking_plan(
    picks,
    graph: GraphInfo,
    budget: int,
    lever: str,
    horizon: int = 0,
    protected: tuple = (),
) -> list[list[ActionOp]]:
    """
    A library selector's output as a horizon-shaped plan, whatever it returns.

    The node levers hand back node ids and the edge levers hand back `(u, v)` pairs,
    so this is the single place the two shapes meet the one plan format: the same
    job `containment.removal_plan` does for a dismantler, and for the same reason:
    the published algorithms are not rewritten to know what a plan is.
    """
    budget_op, _ = resolve_lever(lever)
    guarded = {int(node) for node in protected}
    bag = []
    seen = set()

    for pick in picks:
        if len(bag) >= budget:
            break

        if budget_op in ("add_node", "remove_node"):
            node = int(pick)
            if node in seen or node in guarded:
                continue

            seen.add(node)
            bag.append(ActionOp(budget_op, node))
        else:
            source, target = int(pick[0]), int(pick[1])
            if (source, target) in seen:
                continue

            seen.add((source, target))
            # The weight lever is "blocking as p -> 0" (§1.1), which is the r~ = 0
            # end of DiffIM's own continuous relaxation
            bag.append(
                ActionOp(budget_op, source, target, 0.0)
                if budget_op == "set_edge_weight"
                else ActionOp(budget_op, source, target)
            )

    return [bag] + [[] for _ in range(horizon)]


def blocker_set(actions: list[list[ActionOp]], lever: str) -> list:
    """
    What a finished trajectory actually spent its budget on, in the order it spent it.

    Read back from the executed bags rather than re-planning, for the same reason
    `--compare` replays them: a randomized strategy returns a different set on a
    second call, and every reported number has to describe the set that earned the
    reward.
    """
    budget_op, _ = resolve_lever(lever)
    ordered = []
    seen = set()

    for bag in actions:
        for action in bag:
            if action.op != budget_op:
                continue

            key = (
                int(action.target)
                if action.destination is None
                else (int(action.target), int(action.destination))
            )
            if key in seen:
                continue

            seen.add(key)
            ordered.append(key)

    return ordered


def blocking_metrics(
    reward: float,
    unopposed: float,
    task: TaskSpec,
    graph: GraphInfo,
    actions: list[list[ActionOp]],
) -> dict:
    """
    The prevented-influence block every blocking table reports (§8.1).

    `unopposed` is `sigma(S_N, empty)` (the rumour with no blocker at all) measured
    on the same evaluator as `reward`, because a ratio of two different rulers means
    nothing. Budak's own framing is copied here deliberately: prevented influence
    counts only the nodes that WOULD have been infected, so a blocker that protects
    unreachable nodes scores zero however many of them there are.
    """
    lever = lever_of(task)
    spent = blocker_set(actions, lever)
    prevented = float(unopposed - reward)

    return {
        "lever": lever,
        "lever_papers": lever_papers[lever],
        "negative_seeds": [int(node) for node in task.outbreak],
        "n_negative_seeds": len(task.outbreak),
        "detection_delay": task.detection_delay,
        "unopposed_spread": float(unopposed),
        "prevented_influence": prevented,
        # Both normalizations, because this literature reports both and they answer
        # different questions: the fraction of the rumour stopped, and the fraction
        # of the graph saved
        "prevented_pct_of_unopposed": (
            100.0 * prevented / unopposed if unopposed else 0.0
        ),
        "prevented_pct_of_nodes": 100.0 * prevented / max(graph.num_nodes, 1),
        # §8.2: the informative budget axis here is |S_P| / |S_N|, not |S_P| / |V|.
        # CLDAG's Table 2 is read entirely off this ratio: it takes 20-30x the
        # attacker's seeds to cut the rumour to 10%.
        "budget_ratio": (
            task.budget / len(task.outbreak) if task.outbreak else None
        ),
        "spent": [list(entry) if isinstance(entry, tuple) else entry for entry in spent],
        "n_spent": len(spent),
    }


def unopposed_reference(
    environment: object, task: TaskSpec, horizon: int, budget: int
) -> float:
    """
    `sigma(S_N, empty)` on this arm's own evaluator: the rumour with nobody stopping it.

    One rollout, charged to the arm like every other, because the alternative,
    computing it on a private simulator: would make the prevented-influence column
    a comparison between two different measurement processes. It is also the floor
    any blocker has to beat, which is why it doubles as an anchor row.
    """
    return float(
        environment.rollout(lambda state, timestep: [], horizon, budget).reward
    )


def proximity_ring(graph: GraphInfo, sources, hops: int = 1) -> list[int]:
    """
    Nodes within `hops` of the rumour's own seeds, nearest first.

    §5.4's finding in one function: the degree heuristic "cannot be used for
    influence blocking maximization at all", while proximity: the out-neighbours of
    the negative seeds: is the strong cheap baseline that only falls behind CLDAG
    once the rumour is strong enough to traverse long paths. It is used by the
    library's `proximity` family, by the prompt's diagnostics, and by the anchor
    table, so it is defined once.
    """
    seen = {int(node) for node in sources}
    frontier = set(seen)
    ordered = []

    for _ in range(max(1, hops)):
        frontier = {
            int(other)
            for node in frontier
            for other in graph.out_neighbors(node)
        } - seen
        if not frontier:
            break

        seen |= frontier
        ordered += sorted(frontier, key=graph.degree, reverse=True)

    return ordered


def exposure_scores(
    graph: GraphInfo, sources, hops: int = 4, decay: float = 0.5
) -> np.ndarray:
    """
    A cheap "will the rumour ever get here" mass per node, by damped propagation from `S_N`.

    Not a published method: it is the shared building block the proximity family and
    the prompt's exemplar both need, and it exists because the single most common way
    a blocking algorithm wastes its budget is protecting nodes the cascade never
    reaches (§8.1: those score exactly zero).
    """
    exposure = np.zeros(graph.num_nodes, dtype=np.float64)
    wave = {int(node): 1.0 for node in sources}

    for _ in range(hops):
        following = {}
        for node, mass in wave.items():
            exposure[node] += mass
            for other in graph.out_neighbors(node):
                following[other] = following.get(other, 0.0) + mass * decay

        if not following:
            break

        wave = following

    return exposure
