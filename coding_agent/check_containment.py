"""
Self-check for critical node detection: the outbreak, the removal budget, the
minimize sense, and the structural metrics.

Everything here is checked on graphs where the answer is known by hand, so a
regression fails as an assertion rather than as a quietly inverted result table,
which is the specific failure mode this task invites, because a maximize-shaped
harness scoring a minimize task produces perfectly plausible numbers that name the
worst arm as the winner.

    python -m coding_agent.check_containment
"""

import inspect
import networkx as nx
import numpy as np

import coding_agent.run
from coding_agent.containment import (
    Outbreak,
    delete_node_ops,
    expand_removals,
    removal_plan,
    removal_set,
    ring_size,
    select_outbreak,
)
from coding_agent.envs.monte_carlo_env import MonteCarloEnvironment
from coding_agent.executor import StrategyError, build_strategy, validate_actions
from coding_agent.methods.base import (
    attach_context,
    baseline_anchor,
    evaluate_strategy,
    validate_plan,
)
from coding_agent.tools.dismantling_algorithms import (
    dismantling_algorithms,
    frontier_removal,
)
from coding_agent.tools.immunization_algorithms import immunization_algorithms
from coding_agent.types import ActionOp, GraphInfo, State, TaskSpec, best_by, improves
from data.wm_simulator import blocked
from pipeline.conditions import result_sense
from pipeline.tasks import get_task, minimize
from world_model.wm_metrics import (
    adjacency_sets,
    containment_metrics,
    dismantling_curve,
)

# 0 -> 1 -> 2 -> 3 with certain transmission: an outbreak at 0 takes the whole
# path unless something in the middle is cut
path = [(0, 1), (1, 2), (2, 3)]


def _graph(edges: list[tuple], num_nodes: int, directed: bool = True) -> GraphInfo:
    pairs = list(edges) if directed else list(edges) + [(v, u) for u, v in edges]
    edge_index = np.asarray(pairs, dtype=np.int64).T

    return GraphInfo(
        num_nodes=num_nodes,
        edge_index=edge_index,
        ic_probs=np.ones(edge_index.shape[1], dtype=np.float32),
        directed=directed,
    )


def _task(**overrides) -> TaskSpec:
    defaults = {
        "task": "critical_node_detection",
        "sense": minimize,
        "budget_op": "remove_node",
        "allowed_ops": ("remove_node",),
        "remove_semantics": blocked,
        "budget": 1,
        "horizon": 4,
        "outbreak": (0,),
    }

    return TaskSpec(**(defaults | overrides))


class _RemovePlan:
    """The plan_horizon shape evaluate_strategy expects, with a fixed removal set."""

    def __init__(self, nodes: list[int]) -> None:
        self.nodes = nodes
        self.source_script = ""

    def plan_horizon(self, graph, budget, horizon):
        return [[ActionOp("remove_node", node) for node in self.nodes]] + [
            [] for _ in range(horizon)
        ]


def registry_says_minimize_and_blocked() -> None:
    """The one place the sign and the semantics are decided."""
    task = get_task("critical_node_detection")

    assert task.runnable, task.status
    assert task.contains
    assert task.objective == minimize
    # `spent` would count each blocker as infected and bias every number by +k
    assert task.remove_semantics == blocked
    assert task.budget_op == "remove_node"
    assert task.allowed_ops == ("remove_node",)
    assert task.outbreak_pct > 0


def outbreak_is_deterministic_in_the_seed() -> None:
    """Every arm in a sweep must face the same outbreak, or it measures the outbreak."""
    graph = _graph(path, 4)

    assert select_outbreak(graph, 2, "random", 7) == select_outbreak(graph, 2, "random", 7)
    assert select_outbreak(graph, 2, "random", 7) != select_outbreak(graph, 2, "random", 8)
    # A targeted selector ignores the seed entirely: it is a function of the graph
    assert select_outbreak(graph, 1, "degree", 0) == select_outbreak(graph, 1, "degree", 99)


def outbreak_seeds_the_cascade_the_planner_did_not() -> None:
    """
    Without the wrapper an idle containment policy scores 0 and looks perfect.
    """
    graph = _graph(path, 4)
    environment = MonteCarloEnvironment(graph, "IC", mc_runs=1, remove_semantics=blocked)

    idle = environment.rollout(lambda state, timestep: [], horizon=4, budget=1)
    assert idle.reward == 0.0, idle.reward

    outbreak = Outbreak(sources=(0,), graph=graph)
    contained = environment.rollout(
        outbreak.wrap(lambda state, timestep: []), horizon=4, budget=1
    )
    # p=1.0 on every arc, so the outbreak takes the whole path
    assert contained.reward == 4.0, contained.reward


def cutting_the_path_beats_cutting_nothing() -> None:
    """The end-to-end sanity check: a good blocker must score LOWER."""
    graph = _graph(path, 4)
    environment = MonteCarloEnvironment(graph, "IC", mc_runs=1, remove_semantics=blocked)
    task = _task()

    # Node 1 is the only route out of the source
    cut = evaluate_strategy(_RemovePlan([1]), environment, task, graph)[0]
    # Node 3 is a leaf the cascade reaches last, so blocking it saves only itself
    late = evaluate_strategy(_RemovePlan([3]), environment, task, graph)[0]

    assert cut.reward == 1.0, cut.reward
    assert late.reward == 3.0, late.reward
    assert improves(cut.reward, late.reward, minimize)
    assert not improves(cut.reward, late.reward, "maximize")
    assert best_by([cut, late], lambda run: run.reward, minimize) is cut


def removing_an_outbreak_source_is_rejected() -> None:
    """
    Deleting patient zero ENDS the outbreak rather than containing it.

    Without this rule a uniform random removal set beats every dismantler by luck
    whenever it happens to include a source, which is what the first end-to-end
    run of this task actually produced.
    """
    try:
        validate_actions(
            [ActionOp("remove_node", 0)],
            num_nodes=4,
            budget=2,
            allowed_ops=("remove_node",),
            budget_op="remove_node",
            protected=(0,),
        )
    except StrategyError as error:
        assert "OUTBREAK SOURCE" in str(error), error
    else:
        raise AssertionError("removing an outbreak source was accepted")

    # ...and a non-source is fine
    validate_actions(
        [ActionOp("remove_node", 1)],
        num_nodes=4,
        budget=2,
        allowed_ops=("remove_node",),
        budget_op="remove_node",
        protected=(0,),
    )


def budget_counts_removals_not_edges() -> None:
    """
    One removal is one unit of budget however many arcs its deletion bag carries.

    Charging the incident remove_edge ops would make k mean deg(v) different
    things per node, and a hub would cost twenty times a leaf. The planner emits
    BARE remove_node: allowing it to emit edge ops too would hand it an
    unbudgeted second intervention, and the expansion happens after validation.
    """
    graph = _graph(path, 4, directed=False)
    task = _task(budget=1)

    validate_plan([[ActionOp("remove_node", 1)], []], task, graph)

    # ...two removals against a budget of one is rejected
    try:
        validate_plan(
            [[ActionOp("remove_node", 1), ActionOp("remove_node", 2)]], task, graph
        )
    except StrategyError as error:
        assert "exceeds budget" in str(error), error
    else:
        raise AssertionError("two removals passed a budget of one")

    # ...and so is a hand-rolled deletion bag, because its edge ops are free
    try:
        validate_plan([delete_node_ops(graph, 1)], task, graph)
    except StrategyError as error:
        assert "harness expands" in str(error), error
    else:
        raise AssertionError("an unbudgeted edge op passed validation")

    # The expansion the harness does instead costs the same one unit of budget
    bag = delete_node_ops(graph, 1)
    assert sum(1 for op in bag if op.op == "remove_node") == 1
    assert sum(1 for op in bag if op.op == "remove_edge") == 4, bag


def expansion_is_idempotent_and_strips_the_edges() -> None:
    """
    Both structured heads assume a blocked node's edges are gone from edge_index.

    ICTransmissionHead zeroes the node's `infected` channel under `blocked`, which
    makes it a fresh susceptible its still-present in-edges would re-infect. A bare
    remove_node is therefore expanded here, and expanding twice must not double.
    """
    graph = _graph(path, 4, directed=False)

    once = expand_removals([ActionOp("remove_node", 1)], graph)
    twice = expand_removals(once, graph)

    assert len(once) == 5, once
    assert len(twice) == len(once), twice
    assert {(op.target, op.destination) for op in once if op.op == "remove_edge"} == {
        (1, 0),
        (0, 1),
        (1, 2),
        (2, 1),
    }


def removal_plan_filters_sources_and_refills() -> None:
    """A published dismantler that picks a source must still spend its whole budget."""
    graph = _graph(path, 4, directed=False)
    plan = removal_plan([0, 1], graph, budget=2, protected=(0,), horizon=2)

    removed = removal_set(plan)
    assert len(removed) == 2, removed
    assert 0 not in removed, removed
    assert 1 in removed, removed
    assert len(plan) == 3, plan


def every_dismantler_returns_a_full_distinct_set() -> None:
    """A short or duplicated set silently under-spends the budget."""
    graph = nx.barabasi_albert_graph(120, 3, seed=0)
    info = _graph(list(graph.edges()), 120, directed=False)
    outbreak = tuple(select_outbreak(info, 2, "random", 0))

    for name, selector in dismantling_algorithms.items():
        removals = selector(info, 12, "IC", outbreak=outbreak, horizon=6)
        assert len(removals) == 12, (name, len(removals))
        assert len(set(removals)) == 12, (name, "duplicates")
        assert all(0 <= int(node) < 120 for node in removals), name


def structural_metrics_describe_the_removal_set() -> None:
    """
    The connectivity functionals are context, and they have to be right.

    A 5-node path cut at its middle: two components of 2, pairwise connectivity
    drops from 10 to 2, and the giant component halves.
    """
    graph = _graph([(0, 1), (1, 2), (2, 3), (3, 4)], 5, directed=False)
    metrics = containment_metrics(graph.edge_index, 5, [2])

    assert metrics["pairwise_conn_intact"] == 10.0, metrics
    assert metrics["pairwise_conn"] == 2.0, metrics
    assert metrics["largest_cc_intact"] == 5.0, metrics
    assert metrics["largest_cc_size"] == 2.0, metrics
    assert metrics["n_components"] == 2.0, metrics
    assert metrics["k"] == 1

    # rho is None when the budget never reached the threshold, which is honest
    assert metrics["rho_at_threshold"] is None, metrics["rho_at_threshold"]


def dismantling_curve_is_sequential() -> None:
    """
    s(q) recomputes the residual graph after EVERY removal.

    Batch and sequential removal are different settings whose numbers are not
    interconvertible, and the
    curve is the sequential one by definition.
    """
    graph = _graph([(0, 1), (1, 2), (2, 3), (3, 4)], 5, directed=False)
    curve = dismantling_curve(adjacency_sets(graph.edge_index, 5), [2, 1])

    # intact 5/5, then cut the middle -> 2/5, then cut node 1 -> 1/5
    assert curve == [1.0, 0.4, 0.4], curve

    # ...and the order matters, which is what makes it a curve rather than a point
    reversed_curve = dismantling_curve(adjacency_sets(graph.edge_index, 5), [1, 2])
    assert reversed_curve != curve, reversed_curve


def degree_rank_spearman_catches_a_degree_ranking() -> None:
    """
    A degree-correlation self-measurement: MIND put GDM at 0.762 against its own
    input features.

    A removal order that IS the degree order must read near +1, and the reverse
    near -1, or the column cannot detect what it exists to detect.
    """
    graph = nx.barabasi_albert_graph(80, 3, seed=1)
    info = _graph(list(graph.edges()), 80, directed=False)
    by_degree = dismantling_algorithms["degree_removal"](info, 20, "IC")

    forward = containment_metrics(info.edge_index, 80, by_degree)
    backward = containment_metrics(info.edge_index, 80, list(reversed(by_degree)))

    assert forward["degree_rank_spearman"] > 0.8, forward["degree_rank_spearman"]
    assert backward["degree_rank_spearman"] < -0.8, backward["degree_rank_spearman"]


def scored_mode_emits_the_budgeted_op() -> None:
    """
    The fixed ScoredStrategy harness must emit `remove_node` on a containment task.

    It hardcoded `add_node`, so every scored-mode containment arm was rejected by
    `validate_actions` before it was ever scored: an entire strategy mode dead on
    the task, invisible because the default arm set uses free mode.
    """
    graph = _graph(path, 4, directed=False)
    task = _task(budget=2)

    strategy = build_strategy(
        "class S(ScoredStrategy):\n"
        "    def score(self, node, selected, graph):\n"
        "        return float(graph.degree(node))\n",
        "scored",
    )
    attach_context(strategy, task)
    plan = strategy.plan_horizon(graph, task.budget, task.horizon)

    assert all(op.op == "remove_node" for bag in plan for op in bag), plan
    # ...and it does not spend the budget on a source it is not allowed to remove
    assert all(op.target not in task.outbreak for bag in plan for op in bag), plan
    validate_plan(plan, task, graph)

    # A seeding task still gets add_node from the same harness
    seeding = TaskSpec(budget=2, horizon=2, allowed_ops=("add_node",))
    attach_context(strategy, seeding)
    seed_plan = strategy.plan_horizon(graph, seeding.budget, seeding.horizon)
    assert all(op.op == "add_node" for bag in seed_plan for op in bag), seed_plan


def anchors_run_under_every_task_shape() -> None:
    """
    The reference leaderboard must build for static AND adaptive, seeding AND
    containment.

    Routing the static anchors through `evaluate_strategy` made an adaptive task
    call `act()` on a plan-shaped object, which broke the leaderboard on
    `adaptive_online_im` too, not just here.
    """
    graph = _graph(path, 4, directed=False)
    environment = MonteCarloEnvironment(graph, "IC", mc_runs=2, remove_semantics=blocked)

    for task in (
        _task(budget=1),
        _task(budget=1, rounds=2, horizon=4),
        TaskSpec(budget=1, horizon=4, allowed_ops=("add_node",)),
        TaskSpec(budget=2, horizon=4, allowed_ops=("add_node",), rounds=2),
    ):
        text, best, name = baseline_anchor(environment, task, graph)
        assert name, text
        assert best is not None


def external_seed_script_emits_the_budgeted_op() -> None:
    """
    A published dismantler returns a set to REMOVE, and condition 7 must emit it
    as such.

    `seed_script` hardcoded `add_node`, so every external arm on a containment
    task failed validation and looked like a bad generated program rather than a
    wiring bug.
    """
    from baselines.run_baseline import seed_script

    graph = _graph(path, 4, directed=False)
    task = _task(budget=2)

    strategy = attach_context(
        build_strategy(seed_script([1, 2, 3], task.budget_op), "free"), task
    )
    plan = strategy.plan_horizon(graph, task.budget, task.horizon)

    assert all(op.op == "remove_node" for bag in plan for op in bag), plan
    validate_plan(plan, task, graph)

    # ...and a seeding task still gets add_node
    seeding = TaskSpec(budget=2, horizon=2, allowed_ops=("add_node",))
    seed_plan = attach_context(
        build_strategy(seed_script([1, 2], "add_node"), "free"), seeding
    ).plan_horizon(graph, seeding.budget, seeding.horizon)
    assert all(op.op == "add_node" for bag in seed_plan for op in bag), seed_plan


def readers_resolve_the_sense_from_a_result() -> None:
    """Report, plots and summary all pick a winner through this."""
    assert result_sense([{"objective": "minimize"}]) == minimize
    # ...and fall back to the registry for results written before the field existed
    assert result_sense([{"task": "critical_node_detection"}]) == minimize
    assert result_sense([{"task": "influence_maximization"}]) == "maximize"
    assert result_sense([{}]) == "maximize"


def removals_expand_even_without_an_outbreak() -> None:
    """
    `--outbreak-pct 0` must still expand removals, or the heads read a stale graph.

    The wrapper does two jobs and only one of them is about the outbreak.
    """
    from coding_agent.containment import build_outbreak

    graph = _graph(path, 4, directed=False)
    wrapper = build_outbreak(graph, _task(outbreak=()))

    assert wrapper is not None
    bag = wrapper.wrap(
        lambda state, timestep: [ActionOp("remove_node", 1)]
    )(State([], []), 0)

    assert not any(op.op == "add_node" for op in bag), bag
    assert any(op.op == "remove_edge" for op in bag), bag

    # ...and a seeding task gets no wrapper at all
    assert build_outbreak(graph, TaskSpec()) is None


def the_shared_names_resolve_to_the_dismantling_pool() -> None:
    """
    `netshield` and `acquaintance_immunization` live in TWO pools as different
    functions, and the epidemic forms refuse to dose an index case while the
    dismantling forms will happily delete one. Both take `**_`, so dispatching to
    the wrong pool cannot crash: it silently scores a different node set. A
    containment arm must get the dismantling implementation.
    """
    shared = set(dismantling_algorithms) & set(immunization_algorithms)
    assert shared == {"netshield", "acquaintance_immunization"}, shared

    for name in sorted(shared):
        assert (
            dismantling_algorithms[name] is not immunization_algorithms[name]
        ), name

    # The dispatch is gated on Task.epidemic, so the two tasks land in different
    # branches of the same elif chain rather than on whichever comes first
    from pipeline.tasks import get_task

    assert not get_task("critical_node_detection").epidemic
    assert get_task("epidemic_control").epidemic

    source = inspect.getsource(coding_agent.run)
    gate = "elif config.baseline in immunization_algorithms and get_task(config.task).epidemic:"
    assert gate in source, (
        "the immunization branch is no longer task-gated, so critical node "
        "detection will silently run the epidemic netshield"
    )


def outbreak_wrap_lets_a_removal_beat_a_source_seed() -> None:
    """
    Order inside the bag: the outbreak seeds FIRST, the policy's removals after.

    So a policy that (legally) removes a node the outbreak would have activated
    still wins that node, because apply_actions walks the bag in order.
    """
    graph = _graph(path, 4)
    outbreak = Outbreak(sources=(0,), graph=graph)
    bag = outbreak.wrap(
        lambda state, timestep: [ActionOp("remove_node", 1)] if timestep == 0 else []
    )(State([], []), 0)

    assert bag[0].op == "add_node" and bag[0].target == 0, bag
    assert any(op.op == "remove_node" and op.target == 1 for op in bag), bag
    # ...and the expansion rode along with it
    assert any(op.op == "remove_edge" for op in bag), bag


def the_ring_is_the_budget_that_makes_the_task_trivial() -> None:
    """
    `ring_size` counts |N_1(S) \\ S|, and `frontier_removal` at that budget contains
    everything: the row reads |S| exactly, whatever the graph beyond the ring does.

    That is the number the sweep has to sit under, and the one the first
    power_grid run sat over at three of four budgets, which is why the agent's
    49.00 there was an information asymmetry rather than a result.
    """
    # A star of 3 around source 0, one of which leads on to a tail 4 -> 5
    graph = _graph([(0, 1), (0, 2), (0, 3), (3, 4), (4, 5)], 6, directed=False)
    assert ring_size(graph, (0,)) == 3
    assert ring_size(graph, (0, 3)) == 3, "3 is a source now, 4 joins the ring"

    environment = MonteCarloEnvironment(graph, "IC", mc_runs=1, remove_semantics=blocked)
    task = _task(budget=3, outbreak=(0,))
    picks = frontier_removal(graph, 3, "IC", outbreak=(0,))
    assert sorted(picks) == [1, 2, 3], picks
    contained = evaluate_strategy(_RemovePlan(picks), environment, task, graph)[0]
    assert contained.reward == 1.0, contained.reward

    # One short of the ring and the cascade gets out through whichever
    # neighbour was left, so the row separates choices again
    short = frontier_removal(graph, 2, "IC", outbreak=(0,))
    leaky = evaluate_strategy(_RemovePlan(short), environment, task, graph)[0]
    assert leaky.reward > 1.0, leaky.reward


def check_generated_programs_are_offline() -> None:
    """
    A GENERATED strategy never receives an evaluator binding, whatever the arm.

    The rule since 2026-09-04: the coding agent's algorithms are offline. Only a
    canned baseline (built by the harness for a library member that needs a
    kernel) gets `predict_marginals` and friends; the plan oracle is gone for
    everyone.
    """
    from baselines.run_baseline import seed_script

    graph = _graph([(0, 1), (1, 2)], 3)
    task = _task()
    environment = MonteCarloEnvironment(
        graph, "IC", mc_runs=1, remove_semantics=blocked, negative_seeds=(0,)
    )

    generated = attach_context(
        build_strategy(seed_script([1], task.budget_op), "free"), task, environment
    )
    for name in ("score_plan", "predict_marginals", "step_marginals", "forecast_marginals"):
        assert not hasattr(generated, name), f"generated strategy must not expose {name}"

    canned = attach_context(
        build_strategy(seed_script([1], task.budget_op), "free", canned=True),
        task,
        environment,
    )
    assert hasattr(canned, "predict_marginals"), "a canned baseline keeps its bindings"
    assert not hasattr(canned, "score_plan"), "the plan oracle no longer exists"

if __name__ == "__main__":
    checks = [
        registry_says_minimize_and_blocked,
        outbreak_is_deterministic_in_the_seed,
        outbreak_seeds_the_cascade_the_planner_did_not,
        cutting_the_path_beats_cutting_nothing,
        removing_an_outbreak_source_is_rejected,
        budget_counts_removals_not_edges,
        expansion_is_idempotent_and_strips_the_edges,
        removal_plan_filters_sources_and_refills,
        every_dismantler_returns_a_full_distinct_set,
        structural_metrics_describe_the_removal_set,
        dismantling_curve_is_sequential,
        degree_rank_spearman_catches_a_degree_ranking,
        scored_mode_emits_the_budgeted_op,
        external_seed_script_emits_the_budgeted_op,
        anchors_run_under_every_task_shape,
        readers_resolve_the_sense_from_a_result,
        removals_expand_even_without_an_outbreak,
        the_shared_names_resolve_to_the_dismantling_pool,
        outbreak_wrap_lets_a_removal_beat_a_source_seed,
        the_ring_is_the_budget_that_makes_the_task_trivial,
        check_generated_programs_are_offline,
    ]

    for check in checks:
        check()
        print(f"ok  {check.__name__}")

    print(f"\n{len(checks)} checks passed")
