"""
Named ADAPTIVE IM baselines, composed from the same primitives.

Every algorithm here has the signature

    (state, graph, batch, diffusion_model, **kw) -> list[int]

and returns at most `batch` seeds for ONE round, chosen against the realized
`state` the previous round produced. That is the whole difference from
`algorithms.py`, whose members return a single static seed set from the graph
alone (research/adaptive_online_im.md §1.1).

These are the condition-1 floor for `--task adaptive_online_im`. Without them the
task's `--baselines` arms are all static sets dealt at t=0, i.e. the CONTROL side
of the adaptivity gap with nothing on the adaptive side: the published adaptive
algorithms would be missing from their own table.

Two conventions every member obeys, because the harness enforces them anyway
(`coding_agent/rounds.py::prepare_round_bag`):

  * never return an already-active node: under full-adoption feedback that
    raises, and under myopic it silently wastes the slot;
  * never return more than `batch` nodes.

Warning: Name collision worth knowing: `algorithms.adaptive_greedy` is adaptive
*sampling* (coarse MC to rank, refined MC on the top few), not adaptive
*seeding*. Han et al.'s algorithm is called AdaptGreedy, which is the name used
here, so the two never collide in the `--baselines` namespace.
"""

import numpy as np

from coding_agent.tools import algorithms, primitives
from coding_agent.types import GraphInfo, State

# AdaptGreedy re-estimates every candidate's marginal gain from scratch each
# round, which is the cost this whole task exists to measure. These keep one
# round affordable on the graphs the pipeline can simulate; raise them for a
# quality run and watch evaluator_seconds move, which is the point.
adapt_greedy_mc_runs = 8
adapt_greedy_horizon = 10
adapt_greedy_candidates = 40

# RR sets per node for the RIS-instantiated variant (Han et al.'s EPIC shape)
epic_rr_per_node = 10
epic_max_theta = 20_000


def _susceptible(state: State, graph: GraphInfo) -> list[int]:
    """Nodes the cascade has not reached, in id order."""
    active = set(state.infected) | set(state.frontier)

    return [node for node in range(graph.num_nodes) if node not in active]


def _top_susceptible(
    scores: np.ndarray, state: State, graph: GraphInfo, batch: int
) -> list[int]:
    """Highest-scoring `batch` nodes among those still susceptible."""
    candidates = _susceptible(state, graph)
    candidates.sort(key=lambda node: -float(scores[node]))

    return candidates[:batch]


def adapt_random(
    state: State,
    graph: GraphInfo,
    batch: int,
    diffusion_model: str = "IC",
    seed: int = 0,
    **_: object,
) -> list[int]:
    """Random susceptible nodes each round: the adaptive floor."""
    candidates = _susceptible(state, graph)
    if not candidates:
        return []

    rng = np.random.default_rng(seed + len(state.infected))
    count = min(batch, len(candidates))

    return [int(node) for node in rng.choice(candidates, size=count, replace=False)]


def adapt_degree(
    state: State, graph: GraphInfo, batch: int, diffusion_model: str = "IC", **_: object
) -> list[int]:
    """Top-degree susceptible nodes each round (adaptive Degree heuristic)."""
    return _top_susceptible(primitives.compute_degree(graph), state, graph, batch)


def adapt_pagerank(
    state: State, graph: GraphInfo, batch: int, diffusion_model: str = "IC", **_: object
) -> list[int]:
    """Top-PageRank susceptible nodes each round."""
    return _top_susceptible(primitives.compute_pagerank(graph), state, graph, batch)


def adapt_degree_discount(
    state: State, graph: GraphInfo, batch: int, diffusion_model: str = "IC", **_: object
) -> list[int]:
    """
    DegreeDiscount restricted to susceptibles, discounted by the REALIZED state.

    The adaptive part is the second discount: a node whose neighbourhood the
    cascade already owns is worth less now than its raw degree says, and that is
    information the static version cannot have at t=0.
    """
    active = set(state.infected) | set(state.frontier)
    discounted = primitives.compute_degree(graph).astype(np.float64)

    for node in range(graph.num_nodes):
        covered = sum(
            1
            for neighbor in graph.out_neighbors(node) + graph.in_neighbors(node)
            if neighbor in active
        )
        discounted[node] -= covered

    chosen = []
    available = set(_susceptible(state, graph))

    for _ in range(min(batch, len(available))):
        node = max(available, key=lambda candidate: discounted[candidate])
        chosen.append(int(node))
        available.discard(node)

        # Within-round discounting, so a batch does not stack on one neighbourhood
        for neighbor in graph.out_neighbors(node) + graph.in_neighbors(node):
            discounted[neighbor] -= 1

    return chosen


def adapt_greedy(
    state: State,
    graph: GraphInfo,
    batch: int,
    diffusion_model: str = "IC",
    mc_runs: int = adapt_greedy_mc_runs,
    horizon: int = adapt_greedy_horizon,
    n_candidates: int = adapt_greedy_candidates,
    **_: object,
) -> list[int]:
    """
    AdaptGreedy (Golovin & Krause 2011; Han et al. PVLDB 2018): greedy marginal
    gain re-estimated each round against the realized state.

    The reference adaptive-IM algorithm, and the reason the task is expensive.
    `compute_marginal_gain` conditions on the already-active set by including it
    in the seed list, so a candidate is scored by what it adds ON TOP of what the
    cascade has already reached, which is exactly the quantity that changes
    between rounds and invalidates CELF's cached gains (§2.3).

    ponytail: candidates are pre-filtered to the top `n_candidates` by degree
    rather than scanning all N, because a full scan is O(N x mc_runs) simulations
    per pick and does not finish on anything past a few thousand nodes. That is a
    quality ceiling on this baseline, not on the task; raise n_candidates for a
    faithful run.
    """
    active = sorted(set(state.infected) | set(state.frontier))
    candidates = _top_susceptible(
        primitives.compute_degree(graph), state, graph, n_candidates
    )
    chosen = []

    for _ in range(min(batch, len(candidates))):
        best_node, best_gain = None, float("-inf")

        for node in candidates:
            if node in chosen:
                continue

            gain = primitives.compute_marginal_gain(
                graph,
                active + chosen,
                node,
                diffusion_model,
                mc_runs,
                horizon,
            )
            if gain > best_gain:
                best_node, best_gain = node, gain

        if best_node is None:
            break

        chosen.append(int(best_node))

    return chosen


# The RR sets depend on the graph and the seed alone, and the harness calls an
# adaptive policy once per sample per round with the same seed, so without this
# a 1,000-sample evaluation redraws the identical 20,000 sets 4,000 times; on
# nethept that was hours per row. One entry, holding the graph so its id cannot
# be reused by another object while the entry lives
epic_rr_cache = {}


def epic_rr_sets(graph: GraphInfo, theta: int, seed: int) -> list[set[int]]:
    key = (id(graph), theta, seed)
    hit = epic_rr_cache.get(key)
    if hit is None or hit[0] is not graph:
        epic_rr_cache.clear()
        hit = (graph, primitives.batch_reverse_sample(graph, theta=theta, seed=seed))
        epic_rr_cache[key] = hit

    return hit[1]


def adapt_epic(
    state: State,
    graph: GraphInfo,
    batch: int,
    diffusion_model: str = "IC",
    seed: int = 0,
    **_: object,
) -> list[int]:
    """
    EPIC (Han et al. PVLDB 2018): AdaptGreedy instantiated with an RIS selector
    per batch, which is what makes adaptive IM scalable at all.

    Reverse-reachable sets already covered by the active set are dropped before
    selection, so coverage is measured against what the cascade has NOT reached.
    IC-only: RR sets do not describe LT, and the fallback is the adaptive degree
    heuristic rather than a silently wrong number.
    """
    if diffusion_model != "IC":
        return adapt_degree(state, graph, batch, diffusion_model)

    active = set(state.infected) | set(state.frontier)
    theta = min(epic_max_theta, epic_rr_per_node * graph.num_nodes)
    rr_sets = epic_rr_sets(graph, theta, seed)

    # A set already containing an active node is satisfied: its root is reachable
    # whatever we seed now, so counting it would credit every candidate equally
    remaining = [rr_set - active for rr_set in rr_sets if not (rr_set & active)]
    if not remaining:
        return adapt_degree(state, graph, batch, diffusion_model)

    return [
        node
        for node in primitives.ris_select(remaining, batch, graph.num_nodes)
        if node not in active
    ][:batch]


def static_split(
    state: State,
    graph: GraphInfo,
    batch: int,
    diffusion_model: str = "IC",
    total_budget: int | None = None,
    base_algorithm: str = "degree_discount",
    **_: object,
) -> list[int]:
    """
    A single STATIC seed set, dealt out one batch per round. The control.

    This is §9.4's "non-adaptive strategy wearing a costume": it spends the same
    k on the same nodes a one-shot arm would, and only the timing differs. It is
    here so the adaptivity gap has a denominator computed under the identical
    round machinery, which isolates the timing penalty (a seed placed at round 3
    has fewer steps left to spread) from the adaptivity benefit.

    Stateless across rounds on purpose: the static ranking is a function of the
    graph alone, so each call recomputes it and returns the first `batch` entries
    the cascade has not already reached. No cross-call memory, which is what lets
    it behave identically under both environments' loop orders.
    """
    budget = total_budget if total_budget is not None else graph.num_nodes
    active = set(state.infected) | set(state.frontier)

    # Long enough to still yield `budget` fresh picks after the active set is
    # skipped. A ranking of exactly `budget` is right for a single campaign and
    # empties out under multi-round, where campaign 2 starts with campaign 1's
    # activations already in `active`: measured as campaign_rewards
    # [33.0, 0.0, 0.0], i.e. two campaigns that seeded nothing at all.
    length = min(graph.num_nodes, budget + len(active))
    ranking = algorithms.algorithms[base_algorithm](
        graph, length, diffusion_model
    )

    return [int(node) for node in ranking if node not in active][:batch]


adaptive_algorithms = {
    # the reference adaptive algorithms
    "adapt_greedy": adapt_greedy,
    "adapt_epic": adapt_epic,
    # per-round heuristics
    "adapt_degree": adapt_degree,
    "adapt_degree_discount": adapt_degree_discount,
    "adapt_pagerank": adapt_pagerank,
    "adapt_random": adapt_random,
    # the control: a static set, dealt in slices
    "static_split": static_split,
}

# Simulation-based, so one round costs batch x n_candidates x mc_runs episodes.
# Charged honestly to the arm like any other baseline, but worth naming: under a
# monte_carlo evaluator this is the row that makes the cost axis move.
mc_adaptive_algorithms = ("adapt_greedy",)

adaptive_algorithm_names = list(adaptive_algorithms)
