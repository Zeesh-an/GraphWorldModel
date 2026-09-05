"""
Critical node detection: the exogenous outbreak, and node deletion as a bag.

Two harness pieces the diffusion-CND variant needs and the seeding tasks do not
(research/critical_node_detection.md §2.2). Both are wrappers around the action
function, in the same shape as `stream.GraphEditStream.wrap`, which is why no
environment changed:

  * **The outbreak.** In influence maximization the planner starts the cascade.
    In containment it does not: an outbreak is already running and the budget
    buys blockers. The outbreak is therefore injected at t=0 as `add_node` ops the
    planner neither chooses nor pays for, exactly as the edit stream injects edge
    ops it is not charged for.

  * **Deletion as a bag.** §2.3's "lazy fix": `remove_node(v)` plus one
    `remove_edge` per incident arc, so node deletion needs no sixth op. This is
    not cosmetic. Both structured heads document that a blocked node's edges are
    "gone from edge_index" and rely on it: `ICTransmissionHead` zeroes the node's
    `infected` channel under `blocked`, which makes it a fresh susceptible that
    its (still present) in-edges would happily re-infect. The data generator
    already emits full bags via `wm_actions.delete_node_bag`, so expanding here
    is also what keeps inference matched to training.

Everything is a pure function of (base graph, seed), for the same reason the edit
stream is: `MonteCarloEnvironment` loops (episode, timestep) while
`WorldModelEnvironment` loops (timestep, sample), so anything carrying mutable
state between calls delivers different outbreaks under the two.
"""

from dataclasses import dataclass
import numpy as np

from coding_agent.tools import primitives
from coding_agent.types import ActionFn, ActionOp, GraphInfo, State, TaskSpec

# How the exogenous outbreak's sources are chosen. `random` is the default and
# the honest one: a targeted outbreak makes the blocker's job trivially the same
# ranking problem, and every result would then be about the selector.
outbreak_selectors = ("random", "degree", "pagerank", "betweenness", "kshell")


def select_outbreak(
    graph: GraphInfo, size: int, selector: str = "random", seed: int = 0
) -> list[int]:
    """The cascade's source set: `size` nodes chosen by `selector`, deterministic in `seed`."""
    if selector not in outbreak_selectors:
        raise ValueError(
            f"unknown outbreak selector {selector!r}; choose one of {outbreak_selectors}"
        )

    count = max(1, min(size, graph.num_nodes))

    if selector == "random":
        rng = np.random.default_rng(seed)
        return sorted(
            int(node)
            for node in rng.choice(graph.num_nodes, size=count, replace=False)
        )

    if selector == "degree":
        scores = primitives.compute_degree(graph)
    elif selector == "pagerank":
        scores = primitives.compute_pagerank(graph)
    elif selector == "kshell":
        scores = core_numbers(graph).astype(np.float64)
    else:
        scores = primitives.compute_centrality(graph, kind="betweenness")

    return sorted(int(node) for node in np.argsort(-np.asarray(scores))[:count])


def core_numbers(graph: GraphInfo) -> np.ndarray:
    """
    k-core number per node, by peeling. Shared by the outbreak selector and CoreHD.

    Undirected view (total degree), because k-core is only defined that way and
    the whole dismantling literature is undirected anyway.
    """
    neighbours = neighbour_sets(graph)
    degrees = np.array([len(group) for group in neighbours], dtype=np.int64)
    remaining = set(range(graph.num_nodes))
    core = np.zeros(graph.num_nodes, dtype=np.int64)
    shell = 0

    while remaining:
        shell = max(shell, int(degrees[list(remaining)].min()))
        queue = [node for node in remaining if degrees[node] <= shell]

        while queue:
            node = queue.pop()
            if node not in remaining:
                continue

            remaining.discard(node)
            core[node] = shell

            for neighbour in neighbours[node]:
                if neighbour in remaining:
                    degrees[neighbour] -= 1
                    if degrees[neighbour] <= shell:
                        queue.append(neighbour)

    return core


def neighbour_sets(graph: GraphInfo) -> list[set[int]]:
    """Undirected adjacency as sets: what every dismantling routine works over."""
    groups = [set() for _ in range(graph.num_nodes)]

    for edge in range(graph.edge_index.shape[1]):
        source = int(graph.edge_index[0, edge])
        target = int(graph.edge_index[1, edge])

        if source != target:
            groups[source].add(target)
            groups[target].add(source)

    return groups


def ring_size(graph: GraphInfo, outbreak: tuple[int, ...] | list[int]) -> int:
    """
    |N_1(S) \\ S|: how many deletions wall the outbreak off completely.

    The number every containment budget has to be read against. A budget at or
    above it makes the task trivial: delete the ring and an IC cascade cannot cross
    it, so every outbreak-aware arm scores exactly |S| and the row separates
    nothing. The first power_grid sweep had three of four budgets there.
    """
    sources = {int(node) for node in outbreak}
    neighbours = neighbour_sets(graph)
    ring = set()
    for source in sources:
        ring |= neighbours[source]

    return len(ring - sources)


def delete_node_ops(graph: GraphInfo, node: int) -> list[ActionOp]:
    """
    `remove_node(v)` plus a `remove_edge` per incident arc: node deletion, in the
    existing op set.

    Both orientations are emitted on an undirected graph because the simulator's
    edge store is keyed by arc. A duplicate or already-absent arc is harmless:
    NDlib's `remove_edges` delegates to `remove_edges_from`, and
    `wm_data.apply_edge_ops` pops with a default, so both ignore a miss.
    """
    node = int(node)
    ops = [ActionOp("remove_node", node)]

    for neighbour in set(graph.out_neighbors(node)) | set(graph.in_neighbors(node)):
        ops.append(ActionOp("remove_edge", node, int(neighbour)))
        ops.append(ActionOp("remove_edge", int(neighbour), node))

    return ops


def expand_removals(bag: list[ActionOp], graph: GraphInfo) -> list[ActionOp]:
    """
    Rewrite every bare `remove_node` in `bag` as its full deletion bag.

    Idempotent: a strategy that already emitted the incident `remove_edge` ops
    gets them deduplicated rather than doubled.
    """
    if not any(action.op == "remove_node" for action in bag):
        return list(bag)

    expanded = []
    seen = set()

    for action in bag:
        candidates = (
            delete_node_ops(graph, action.target)
            if action.op == "remove_node"
            else [action]
        )

        for op in candidates:
            key = (op.op, int(op.target), op.destination)
            if key in seen:
                continue

            seen.add(key)
            expanded.append(op)

    return expanded


@dataclass()
class Outbreak:
    """The exogenous cascade a containment policy is trying to stop."""

    sources: tuple
    graph: GraphInfo

    @property
    def bag(self) -> list[ActionOp]:
        return [ActionOp("add_node", int(node)) for node in self.sources]

    def wrap(self, action_fn: ActionFn) -> ActionFn:
        """
        Seed the outbreak at t=0 and expand the policy's removals at every step.

        Applied AFTER the policy's bag has been validated, deliberately, for both
        halves: the outbreak's `add_node` ops are exogenous, so they would be
        rejected under `--allowed-ops remove_node` and must not count against the
        removal budget; and the expansion's `remove_edge` ops are the mechanics of
        one removal, not a second intervention to be charged for.

        Order matters and this is the right one: the outbreak's seeds go in FIRST,
        so a policy that blocks a source node at t=0 still wins: `apply_actions`
        walks the bag in order, so the later `remove_node` overwrites the earlier
        `add_node` and the node ends up blocked rather than infectious.
        """

        def contained(state: State, timestep: int) -> list[ActionOp]:
            policy_bag = expand_removals(list(action_fn(state, timestep)), self.graph)

            return (self.bag if timestep == 0 else []) + policy_bag

        return contained


def build_outbreak(graph: GraphInfo, task: TaskSpec) -> Outbreak | None:
    """
    None for every task whose planner starts its own cascade.

    A containment task gets one even with an EMPTY source set (`--outbreak-pct 0`),
    because the wrapper does two jobs: seeding the outbreak, and expanding removals
    into deletion bags. The second is not optional: both structured heads assume a
    blocked node's edges are gone from `edge_index`, so returning None there would
    leave the heads reading a graph the simulator no longer has.
    """
    if not task.outbreak and not task.contains:
        return None

    return Outbreak(sources=tuple(int(node) for node in task.outbreak), graph=graph)


def removal_plan(
    nodes,
    graph: GraphInfo,
    budget: int,
    protected: tuple = (),
    horizon: int = 0,
) -> list[list[ActionOp]]:
    """
    A selector's node list as a horizon-shaped plan: all removals at t=0.

    Drops outbreak sources and duplicates, then tops up from the highest-degree
    survivors so a filtered set still spends its whole budget. The filter is the
    single place a library dismantler meets the no-deleting-the-source rule
    (`executor.validate_actions`): the algorithms themselves are the published
    ones and are not rewritten to know about an outbreak.
    """
    guarded = {int(node) for node in protected}
    chosen = []
    seen = set(guarded)

    for node in nodes:
        if len(chosen) >= budget:
            break

        node = int(node)
        if node not in seen:
            seen.add(node)
            chosen.append(node)

    if len(chosen) < budget:
        degrees = primitives.compute_degree(graph)
        for node in np.argsort(-degrees):
            if len(chosen) >= budget:
                break

            node = int(node)
            if node not in seen:
                seen.add(node)
                chosen.append(node)

    return [[ActionOp("remove_node", node) for node in chosen]] + [
        [] for _ in range(horizon)
    ]


def removal_set(actions: list[list[ActionOp]]) -> list[int]:
    """
    The distinct nodes a finished trajectory removed, in the order it removed them.

    Read back from the executed bags rather than re-planning, for the same reason
    the referee replays them: a randomized strategy returns a different set on a
    second call, and the structural metrics have to describe the set that
    actually earned the reward.
    """
    ordered = []
    seen = set()

    for bag in actions:
        for action in bag:
            if action.op != "remove_node" or int(action.target) in seen:
                continue

            seen.add(int(action.target))
            ordered.append(int(action.target))

    return ordered
