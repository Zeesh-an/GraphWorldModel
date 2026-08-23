"""
Self-check for influence blocking: the two cascades, the tie-break, the four levers,
prevented influence, and the competitive head.

Everything here runs on graphs where the answer is known by hand, so a regression
fails as an assertion rather than as a quietly wrong results table, which is the
specific failure mode this task invites, because a two-cascade harness that silently
drops one cascade produces perfectly plausible numbers in which every blocker looks
useless and every arm ties at the unopposed spread.

    python -m coding_agent.check_influence_blocking
"""

import networkx as nx
import numpy as np
import torch

from coding_agent.blocking import (
    blocker_set,
    blocking_metrics,
    blocking_plan,
    build_negative_cascade,
    counter_seed,
    edge_block,
    edge_weight_caps,
    exposure_scores,
    lever_of,
    node_block,
    proximity_ring,
    resolve_lever,
    unopposed_reference,
    weight_block,
)
from coding_agent.envs.monte_carlo_env import MonteCarloEnvironment
from coding_agent.executor import StrategyError, build_strategy, validate_actions
from coding_agent.methods.base import (
    attach_context,
    baseline_anchor,
    evaluate_strategy,
    validate_plan,
)
from coding_agent.tools.blocking_algorithms import (
    all_blocking_algorithms,
    blocking_levers,
    blocking_shape,
    default_blocking_baselines,
    dominator_tree,
    emittable,
    lever_shape,
)
from coding_agent.types import ActionOp, GraphInfo, State, TaskSpec
from data.wm_competitive import (
    CompetitiveConfig,
    CompetitiveSimulator,
    competitive_model_name,
    fixed_dominance,
    negative_dominance,
    positive_dominance,
    resolve_tie_break,
    run_competitive,
)
from data.wm_simulator import blocked
from pipeline.conditions import result_sense
from pipeline.tasks import get_task, minimize
from world_model.wm_data import (
    build_competitive_features,
    build_graph_input,
    ch_neg_infected,
    ch_pos_infected,
    channels_for,
    competitive_in_channels,
)
from world_model.wm_model import WorldModel

# 0 -> 1 -> 2 -> 3 with certain transmission. A rumour at 0 takes the whole path
# unless something answers it; a blocker at 1 takes 2 and 3 for itself instead.
path = [(0, 1), (1, 2), (2, 3)]

# A diamond: 0 -> 1 -> 3 and 0 -> 2 -> 3. Node 3 is reached by both branches, so
# neither 1 nor 2 dominates it: the case a naive "count my descendants" scorer
# gets wrong and the dominator tree gets right.
diamond = [(0, 1), (0, 2), (1, 3), (2, 3)]


def _graph(edges: list[tuple], num_nodes: int, directed: bool = True) -> GraphInfo:
    pairs = list(edges) if directed else list(edges) + [(v, u) for u, v in edges]
    edge_index = np.asarray(pairs, dtype=np.int64).T

    return GraphInfo(
        num_nodes=num_nodes,
        edge_index=edge_index,
        ic_probs=np.ones(edge_index.shape[1], dtype=np.float32),
        directed=directed,
    )


def _nx(edges: list[tuple], num_nodes: int) -> tuple:
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edges)

    return graph, {(u, v): 1.0 for u, v in edges}


def _task(**overrides) -> TaskSpec:
    defaults = {
        "task": "influence_blocking",
        "sense": minimize,
        "budget_op": "add_node",
        "allowed_ops": ("add_node",),
        "remove_semantics": blocked,
        "competitive": True,
        "tie_break": positive_dominance,
        "budget": 1,
        "horizon": 4,
        "outbreak": (0,),
    }

    return TaskSpec(**(defaults | overrides))


class _Plan:
    """The plan_horizon shape evaluate_strategy expects, with a fixed bag."""

    def __init__(self, bag: list[ActionOp]) -> None:
        self.bag = bag
        self.source_script = ""

    def plan_horizon(self, graph, budget, horizon):
        return [list(self.bag)] + [[] for _ in range(horizon)]


def check_registry() -> None:
    """The task is runnable, competitive, minimizes, and seeds a rumour."""
    task = get_task("influence_blocking")

    assert task.runnable, "influence_blocking should be implemented"
    assert task.competitive, "influence blocking is the two-cascade task"
    assert task.objective == minimize
    assert task.remove_semantics == blocked, (
        "a blocking task must delete blocked nodes; `spent` would count each one as "
        "infected and bias every prevented-influence number by exactly +k"
    )
    assert task.outbreak_pct > 0, "a blocker with no rumour to answer scores nothing"
    # §8.2: percent-of-N budgets speak to no blocking paper at all
    assert task.default_budgets == (10, 20, 30, 40, 50)
    assert result_sense([{"task": "influence_blocking"}]) == minimize

    # Condition 1's pool is per LEVER here, because a lever can only emit what its
    # own members return. Every pool has to be non-empty, lever-pure, and lead with
    # the row that has to be beaten.
    for lever, pool in default_blocking_baselines.items():
        assert pool, lever
        for name in pool:
            assert emittable(name, lever), (
                f"{name} returns {blocking_shape(name)}s but is pooled under "
                f"{lever}, which spends its budget on {lever_shape[lever]}s"
            )

    assert default_blocking_baselines[counter_seed][0] == "proximity", (
        "proximity is the row that has to be beaten (§5.4, §5.6)"
    )
    assert default_blocking_baselines[node_block][0] == "imin_lhga", (
        "SandIMIN's own trivial heuristic beats both of its principled methods in "
        "6 of 30 cells (§5.3)"
    )
    assert "degree_blocking" in default_blocking_baselines[counter_seed], (
        "degree is the published FAILURE mode here and belongs in the table"
    )


def check_tie_break() -> None:
    """Each dynamics resolves to its own founding paper's rule, and all three differ."""
    assert resolve_tie_break("auto", "IC") == positive_dominance, (
        "Budak's MCICM/COICM resolve a same-step arrival in favour of the GOOD campaign"
    )
    assert resolve_tie_break("auto", "LT") == negative_dominance, (
        "He et al.'s CLT resolves it in favour of the NEGATIVE one"
    )
    assert resolve_tie_break("negative", "IC") == negative_dominance

    # A node both cascades reach in one step: 0 is the rumour, 2 is the blocker,
    # both one certain hop from 1. The rule is the only thing that decides who gets it.
    graph, probs = _nx([(0, 1), (2, 1)], 3)
    outcomes = {}

    for rule in (negative_dominance, positive_dominance, fixed_dominance):
        simulator = CompetitiveSimulator(
            graph, probs, seed=0, config=CompetitiveConfig(tie_break=rule)
        )
        simulator.reset("IC", [0])
        state = simulator.advance([ActionOp("add_node", 2)])
        outcomes[rule] = (1 in state.infected, 1 in state.pos_infected)

    assert outcomes[negative_dominance] == (True, False), outcomes
    assert outcomes[positive_dominance] == (False, True), outcomes
    # Fixed dominance draws a per-node priority, so it lands on exactly one of them
    assert sum(outcomes[fixed_dominance]) == 1, outcomes


def check_disjoint_cascades() -> None:
    """A node belongs to at most one cascade, and a committed node cannot be re-taken."""
    graph, probs = _nx(path, 4)
    simulator = CompetitiveSimulator(graph, probs, seed=0)
    simulator.reset("IC", [0])

    # Seeding the rumour's own node positively is a no-op: it is already committed
    simulator.advance([ActionOp("add_node", 0)])
    assert 0 in simulator.neg_infected and 0 not in simulator.pos_infected

    for _ in range(4):
        simulator.advance([])

    overlap = simulator.neg_infected & simulator.pos_infected
    assert not overlap, f"a node ended up in both cascades: {sorted(overlap)}"


def check_blocking_works() -> None:
    """A blocker on the rumour's only route saves the rest of the path."""
    graph, probs = _nx(path, 4)

    unopposed = run_competitive(graph, probs, "IC", [0], [[]], horizon=5)
    assert sorted(unopposed.infected) == [0, 1, 2, 3], (
        f"an unopposed rumour on a certain path should take all of it, got "
        f"{sorted(unopposed.infected)}"
    )

    # Under POSITIVE dominance the blocker wins node 1 outright and the rumour stops
    blocked_state = run_competitive(
        graph, probs, "IC", [0], [[ActionOp("add_node", 1)]], horizon=5
    )
    assert sorted(blocked_state.infected) == [0], (
        f"a blocker at 1 should hold the rumour at its own seed, got "
        f"{sorted(blocked_state.infected)}"
    )
    assert 3 in blocked_state.pos_infected, "the counter-cascade should take the tail"


def check_levers() -> None:
    """All four levers stop the same path cascade, each through its own op."""
    graph, probs = _nx(path, 4)
    plans = {
        counter_seed: [[ActionOp("add_node", 1)]],
        node_block: [[ActionOp("remove_node", 1), ActionOp("remove_edge", 0, 1),
                      ActionOp("remove_edge", 1, 2)]],
        edge_block: [[ActionOp("remove_edge", 0, 1)]],
        weight_block: [[ActionOp("set_edge_weight", 0, 1, 0.0)]],
    }

    for lever, plan in plans.items():
        state = run_competitive(graph, probs, "IC", [0], plan, horizon=5)
        assert sorted(state.infected) == [0], (
            f"the {lever} lever failed to hold the rumour at its seed: "
            f"{sorted(state.infected)}"
        )

    # ...and each one declares the op it spends budget on
    for lever, (budget_op, allowed) in (
        (name, resolve_lever(name)) for name in plans
    ):
        assert budget_op in allowed, (lever, budget_op, allowed)


def check_edge_budget_key() -> None:
    """Two arcs out of one node are two interventions, not a duplicate."""
    bag = [ActionOp("remove_edge", 0, 1), ActionOp("remove_edge", 0, 2)]

    # This is the bug the edge levers would otherwise hit: a node-shaped duplicate
    # check keys on `target` alone and rejects a perfectly legal cut set
    validate_actions(bag, 4, budget=2, allowed_ops=("remove_edge",), budget_op="remove_edge")

    try:
        validate_actions(
            bag + [ActionOp("remove_edge", 0, 1)],
            4,
            budget=3,
            allowed_ops=("remove_edge",),
            budget_op="remove_edge",
        )
    except StrategyError:
        pass
    else:
        raise AssertionError("cutting the same arc twice should be rejected")


def check_weight_cap() -> None:
    """A blocker may lower an arc and never raise one."""
    graph = _graph(path, 4)
    graph.ic_probs = np.full(graph.edge_index.shape[1], 0.4, dtype=np.float32)
    caps = edge_weight_caps(graph)

    validate_actions(
        [ActionOp("set_edge_weight", 0, 1, 0.1)],
        4,
        budget=1,
        allowed_ops=("set_edge_weight",),
        budget_op="set_edge_weight",
        edge_weight_caps=caps,
    )

    for bad, why in (
        (ActionOp("set_edge_weight", 0, 1, 0.9), "raising an arc is a boost, not a block"),
        (ActionOp("set_edge_weight", 3, 0, 0.0), "reweighting a missing arc buys nothing"),
    ):
        try:
            validate_actions(
                [bad],
                4,
                budget=1,
                allowed_ops=("set_edge_weight",),
                budget_op="set_edge_weight",
                edge_weight_caps=caps,
            )
        except StrategyError:
            continue

        raise AssertionError(why)


def check_detection_delay() -> None:
    """Budak's r drops everything the blocker emits before it, and only that."""
    graph = _graph(path, 4)
    task = _task(detection_delay=2)
    cascade = build_negative_cascade(graph, task)

    wrapped = cascade.wrap(lambda state, timestep: [ActionOp("add_node", 1)])
    assert wrapped(State([], []), 0) == [], "an early action should be dropped"
    assert wrapped(State([], []), 1) == []
    assert [action.op for action in wrapped(State([], []), 2)] == ["add_node"]

    # ...and the wrapper must NOT inject S_N: `add_node` seeds the POSITIVE cascade,
    # so a containment-style outbreak wrapper here would have every arm start the
    # rumour's own counter-cascade for it
    emitted = wrapped(State([], []), 2)
    assert all(int(action.target) != 0 for action in emitted), emitted


def check_removal_expansion() -> None:
    """A bare remove_node still becomes a full deletion bag under the node lever."""
    graph = _graph(path, 4, directed=False)
    task = _task(budget_op="remove_node", allowed_ops=("remove_node",))
    cascade = build_negative_cascade(graph, task)

    expanded = cascade.wrap(lambda state, timestep: [ActionOp("remove_node", 1)])(
        State([], []), 0
    )
    ops = {action.op for action in expanded}
    assert ops == {"remove_node", "remove_edge"}, ops


def check_dominator_tree() -> None:
    """The dominator count is the reach a node ALONE stands between the sources and."""
    # On the path 0 -> 1 -> 2 -> 3 with every edge live, node 1 dominates 2 and 3
    live = {0: [1], 1: [2], 2: [3], 3: []}
    _, counts = dominator_tree(live, [0])
    assert counts[1] == 3, counts  # itself plus 2 and 3
    assert counts[3] == 1, counts

    # On the diamond, neither branch dominates the join: node 3 has two ways in
    live = {0: [1, 2], 1: [3], 2: [3], 3: []}
    _, counts = dominator_tree(live, [0])
    assert counts[1] == 1 and counts[2] == 1, counts
    assert counts[3] == 1, counts


def check_library_contracts() -> None:
    """Every library member returns the shape its lever can emit, at the right size."""
    graph = _graph(
        [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (2, 5)], 6, directed=False
    )
    budget = 2

    for name, function in all_blocking_algorithms.items():
        picks = function(graph, budget, "IC", negative_seeds=(0,), horizon=4)
        lever = blocking_levers[name]

        assert len(picks) <= budget, f"{name} returned {len(picks)} for budget {budget}"

        if lever == "edge_block":
            assert all(len(pick) == 2 for pick in picks), f"{name} must return arcs"
        else:
            assert all(isinstance(int(pick), int) for pick in picks), name
            assert 0 not in [int(pick) for pick in picks], (
                f"{name} returned the rumour's own seed as a blocker"
            )

        # ...and every one of them survives the plan builder and validation
        budget_op, allowed_ops = resolve_lever(lever)
        plan = blocking_plan(picks, graph, budget, lever, horizon=4)
        validate_plan(
            plan,
            _task(budget=budget, budget_op=budget_op, allowed_ops=allowed_ops),
            graph,
        )


def check_proximity_beats_degree() -> None:
    """
    The finding the whole baseline table is calibrated against.

    A star whose hub is far from the rumour: `degree_blocking` takes the hub and
    saves nobody, `proximity` takes the rumour's own out-neighbour and saves the
    branch behind it. §5.4 states this outright and it is the reason degree is in the
    default pool as a FAILURE mode rather than as a floor.
    """
    # 0 (rumour) -> 1 -> 2; and a high-degree hub 3 wired to 4..9, unreachable from 0
    edges = [(0, 1), (1, 2)] + [(3, node) for node in range(4, 10)]
    graph = _graph(edges, 10)

    by_degree = degree_pick = all_blocking_algorithms["degree_blocking"](
        graph, 1, "IC", negative_seeds=(0,)
    )
    by_proximity = all_blocking_algorithms["proximity"](graph, 1, "IC", negative_seeds=(0,))

    assert by_degree == [3], f"degree should chase the hub, got {by_degree}"
    assert by_proximity == [1], f"proximity should take the rumour's neighbour, got {by_proximity}"
    assert degree_pick != by_proximity, "the two heuristics must disagree here"

    graph_nx, probs = _nx(edges, 10)
    spreads = {
        name: len(
            run_competitive(
                graph_nx, probs, "IC", [0], [[ActionOp("add_node", pick[0])]], horizon=5
            ).infected
        )
        for name, pick in (("degree", by_degree), ("proximity", by_proximity))
    }
    assert spreads["proximity"] < spreads["degree"], spreads


def check_prevented_influence() -> None:
    """Prevented influence is measured against the unopposed run, and counts only saves."""
    graph = _graph(path, 4)
    task = _task(budget=1)
    environment = MonteCarloEnvironment(
        graph,
        "IC",
        mc_runs=8,
        base_seed=0,
        remove_semantics=blocked,
        negative_seeds=(0,),
        competitive_config=CompetitiveConfig(tie_break=positive_dominance),
    )

    unopposed = unopposed_reference(environment, task, task.horizon, task.budget)
    assert unopposed == 4.0, f"the unopposed rumour should take all 4 nodes, got {unopposed}"

    trajectory, _ = evaluate_strategy(
        _Plan([ActionOp("add_node", 1)]), environment, task, graph
    )
    metrics = blocking_metrics(
        trajectory.reward, unopposed, task, graph, trajectory.actions
    )

    assert metrics["prevented_influence"] == 3.0, metrics
    assert metrics["lever"] == counter_seed
    assert metrics["spent"] == [1], metrics
    assert metrics["budget_ratio"] == 1.0, metrics
    assert round(metrics["prevented_pct_of_unopposed"], 3) == 75.0, metrics


def check_no_credit_for_unreachable() -> None:
    """
    Budak's sharp framing: protecting a node the rumour could never reach scores ZERO.

    The one property that separates prevented influence from every other objective in
    this repo, and the commonest way a blocking algorithm wastes its budget.
    """
    edges = [(0, 1), (1, 2)] + [(5, 6), (6, 7)]
    graph = _graph(edges, 8)
    graph_nx, probs = _nx(edges, 8)
    task = _task(budget=1)

    unopposed = len(run_competitive(graph_nx, probs, "IC", [0], [[]], horizon=5).infected)
    useless = len(
        run_competitive(
            graph_nx, probs, "IC", [0], [[ActionOp("add_node", 6)]], horizon=5
        ).infected
    )

    assert useless == unopposed, (
        f"a blocker in a disconnected component prevented {unopposed - useless} "
        f"infections; it must prevent exactly none"
    )

    metrics = blocking_metrics(
        float(useless), float(unopposed), task, graph, [[ActionOp("add_node", 6)]]
    )
    assert metrics["prevented_influence"] == 0.0, metrics


def check_competitive_head() -> None:
    """
    The head predicts four channels, keeps the cascades disjoint, and self-terminates.

    The last property is what the whole structured-head design exists for: a
    susceptible node with no active in-neighbour in EITHER cascade has to come out at
    exactly zero, or a free-running two-cascade rollout saturates and the
    prevented-influence column becomes a difference of two runaway cascades.
    """
    graph = _graph(path, 4)
    in_channels, out_channels = channels_for(True)
    assert (in_channels, out_channels) == (8, 4)

    model = WorldModel(
        "gcn",
        in_channels=competitive_in_channels,
        hidden_dim=8,
        n_layers=1,
        dropout=0.0,
        head_type="structured_oracle",
        diffusion_model="IC",
        remove_semantics=blocked,
        competitive=True,
        tie_break=positive_dominance,
    ).eval()

    record = {
        "state": {
            "infected": [0],
            "frontier": [0],
            "pos_infected": [],
            "pos_frontier": [],
        },
        "action": [{"op": "add_node", "target": 2}],
        "next_marginal_infected": {},
        "next_marginal_frontier": {},
        "next_marginal_pos_infected": {},
        "next_marginal_pos_frontier": {},
    }
    X, targets = build_competitive_features(record, graph.edge_index, 4)

    assert X.shape == (4, 8) and targets.shape == (4, 4)
    assert X[0, ch_neg_infected] == 1.0
    assert X[2, ch_pos_infected] == 0.0, "the ADD channel carries the seed, not the state"

    graph_input = build_graph_input(
        graph.edge_index, graph.ic_probs, 4, "IC", torch.device("cpu")
    )
    with torch.inference_mode():
        probs = torch.sigmoid(model(torch.from_numpy(X), graph_input)).numpy()

    assert probs.shape == (4, 4)
    # The rumour is at 0 with a certain arc to 1, so 1 goes negative next step
    assert probs[1, 0] > 0.99, probs[1]
    # 2 was just seeded positively, so it is positively infected and NOT negatively
    assert probs[2, 2] > 0.99 and probs[2, 0] < 0.01, probs[2]
    # ...and 3 is reached by the blocker, not the rumour
    assert probs[3, 2] > 0.99 and probs[3, 0] < 0.01, probs[3]
    # Self-termination: with no cascade anywhere, every output is zero
    empty = np.zeros((4, 8), dtype=np.float32)
    with torch.inference_mode():
        idle = torch.sigmoid(model(torch.from_numpy(empty), graph_input)).numpy()

    assert idle.max() < 1e-3, f"an idle state must predict nothing, got {idle.max()}"


def check_head_matches_simulator() -> None:
    """
    The oracle head's one-step marginals agree with the simulator's, per tie-break.

    This is the claim §2.2 leaves open and this repo has to settle: the two-MLP
    product form is exact under BOTH COICM and MCICM, because a node activates in at
    most one campaign and the two arrival probabilities are products over disjoint
    in-edge sets. If that argument were wrong, the oracle head and the simulator would
    disagree exactly here.
    """
    edges = [(0, 3), (1, 3), (2, 3)]
    graph = _graph(edges, 4)
    graph.ic_probs = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    graph_nx = nx.DiGraph()
    graph_nx.add_nodes_from(range(4))
    graph_nx.add_edges_from(edges)
    probs = {edge: 0.5 for edge in edges}

    for rule in (negative_dominance, positive_dominance):
        simulator = CompetitiveSimulator(
            graph_nx, probs, seed=7, config=CompetitiveConfig(tie_break=rule)
        )
        simulator.reset("IC", [0])
        # Rumour at 0, blocker at 1 and 2, all one certain-ish hop from node 3
        _, neg_infected, _, _, _ = simulator.advance_marginal(
            [ActionOp("add_node", 1), ActionOp("add_node", 2)], num_mc=4000
        )
        empirical = neg_infected.get(3, 0.0)

        model = WorldModel(
            "gcn",
            in_channels=competitive_in_channels,
            hidden_dim=8,
            n_layers=1,
            dropout=0.0,
            head_type="structured_oracle",
            diffusion_model="IC",
            remove_semantics=blocked,
            competitive=True,
            tie_break=rule,
        ).eval()

        record = {
            "state": {"infected": [0], "frontier": [0], "pos_infected": [], "pos_frontier": []},
            "action": [
                {"op": "add_node", "target": 1},
                {"op": "add_node", "target": 2},
            ],
            "next_marginal_infected": {},
            "next_marginal_frontier": {},
            "next_marginal_pos_infected": {},
            "next_marginal_pos_frontier": {},
        }
        X, _ = build_competitive_features(record, graph.edge_index, 4)
        graph_input = build_graph_input(
            graph.edge_index, graph.ic_probs, 4, "IC", torch.device("cpu")
        )
        with torch.inference_mode():
            predicted = float(
                torch.sigmoid(model(torch.from_numpy(X), graph_input))[3, 0]
            )

        assert abs(predicted - empirical) < 0.03, (
            f"under {rule} dominance the oracle head predicts {predicted:.4f} for "
            f"node 3 but the simulator measures {empirical:.4f} over 4000 draws, the "
            f"product form and the tie-break composition disagree"
        )


def check_anchor_table() -> None:
    """The blocking anchor table runs, and `no_blocking` is the worst row on it."""
    graph = _graph(
        [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5)], 6
    )
    task = _task(budget=1, horizon=4)
    environment = MonteCarloEnvironment(
        graph,
        "IC",
        mc_runs=8,
        base_seed=0,
        remove_semantics=blocked,
        negative_seeds=(0,),
        competitive_config=CompetitiveConfig(tie_break=positive_dominance),
    )

    text, best, name = baseline_anchor(environment, task, graph)

    assert "no_blocking" in text and "prevented" in text, text
    assert name != "no_blocking", (
        "at least one baseline should beat doing nothing on a path graph"
    )
    assert best.reward <= 6.0


def check_generated_script() -> None:
    """A blocking-shaped generated script builds, validates and scores."""
    script = """
class MyBlocker(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        picks = blocking_algorithms.proximity(
            graph, budget, "IC", negative_seeds=self.outbreak
        )
        return blocking.blocking_plan(picks, graph, budget, "counter_seed", horizon)
"""
    graph = _graph(path, 4)
    task = _task(budget=1)
    strategy = build_strategy(script)
    environment = MonteCarloEnvironment(
        graph,
        "IC",
        mc_runs=4,
        base_seed=0,
        remove_semantics=blocked,
        negative_seeds=(0,),
        competitive_config=CompetitiveConfig(tie_break=positive_dominance),
    )
    trajectory, _ = evaluate_strategy(strategy, environment, task, graph)

    assert blocker_set(trajectory.actions, counter_seed) == [1], trajectory.actions
    assert trajectory.reward < 4.0, trajectory.reward


def check_helpers() -> None:
    """The shared blocking helpers behave as the prompts and the library assume."""
    graph = _graph([(0, 1), (0, 2), (1, 3)], 4)

    assert proximity_ring(graph, (0,), hops=1) == sorted([1, 2], key=graph.degree, reverse=True)
    assert 3 in proximity_ring(graph, (0,), hops=2)

    exposure = exposure_scores(graph, (0,))
    assert exposure[0] > exposure[1] > exposure[3] > 0.0, exposure

    assert lever_of(_task(budget_op="remove_edge")) == edge_block
    assert competitive_model_name("shared") == "coicm"
    assert competitive_model_name(1.0) == "mcicm"


def check_mia_scores_do_not_collapse_onto_the_periphery() -> None:
    """
    `cldag` and `cmia_o` must not re-weight their local DAG by the path probability.

    The max-influence path underestimates P(infected) far worse for high-degree
    nodes than for low-degree ones. Measured on email-Eu-core: 24x at a degree-544
    hub against 2.5x at a degree-31 node, because the weighted-cascade model puts
    p(u->v) = 1/in_degree(v) on every arc into a hub. Multiplying two such estimates
    together ranked a node that only saves ITSELF above one that saves 217 others,
    and both methods then scored barely better than `random_blocking` under IC and
    LT alike. Pinned by the SYMPTOM rather than by the score, so rewriting the
    scoring rule stays free as long as it does not reintroduce the collapse.
    """
    # A scale-free graph under the WEIGHTED-CASCADE model, which is where the
    # distortion lives: p(u->v) = 1/in_degree(v) makes every arc into a hub weak,
    # so the single best path to a hub is improbable even though the hub is reached
    # almost surely. Uniform probabilities hide the bug entirely.
    scale_free = nx.barabasi_albert_graph(160, 3, seed=5).to_directed()
    pairs = list(scale_free.edges())
    edge_index = np.asarray(pairs, dtype=np.int64).T
    in_degree = np.zeros(160)
    np.add.at(in_degree, edge_index[1], 1)
    graph = GraphInfo(
        num_nodes=160,
        edge_index=edge_index,
        ic_probs=(1.0 / np.maximum(in_degree[edge_index[1]], 1.0)).astype(np.float32),
        directed=True,
    )
    sources = sorted(range(graph.num_nodes), key=graph.degree, reverse=True)[:3]
    budget = 8
    everyone = float(np.mean([graph.degree(n) for n in range(graph.num_nodes)]))

    for name in ("cldag", "cmia_o"):
        picks = all_blocking_algorithms[name](
            graph, budget, "IC", negative_seeds=tuple(sources)
        )
        assert len(picks) == budget, (name, picks)

        chosen = float(np.mean([graph.degree(node) for node in picks]))
        assert chosen > everyone, (
            f"{name} picked mean degree {chosen:.1f} against a graph mean of "
            f"{everyone:.1f}: that is the periphery collapse the DAG re-weighting "
            f"caused"
        )


def check_plan_oracle_races_the_rumour() -> None:
    """
    `self.score_plan` under two cascades: the rumour is committed, the candidate
    counter-seeds race it, and the number returned is the number the evaluation
    would return. This is the binding that replaces the 260-seconds-per-call
    hand-rolled live-edge samplers the first sweep's programs built.
    """
    graph = _graph(path, 4)
    task = _task(budget=1)
    environment = MonteCarloEnvironment(
        graph,
        "IC",
        mc_runs=8,
        base_seed=0,
        remove_semantics=blocked,
        negative_seeds=(0,),
        competitive_config=CompetitiveConfig(tie_break=positive_dominance),
    )

    early = [[ActionOp("add_node", 1)]] + [[] for _ in range(task.horizon)]
    late = [[ActionOp("add_node", 3)]] + [[] for _ in range(task.horizon)]

    class Probe:
        source_script = ""

        def plan_horizon(self, graph, budget, horizon):
            return early

    probe = Probe()
    trajectory, _ = evaluate_strategy(probe, environment, task, graph)

    assert probe.score_plan(early) == trajectory.reward
    # Counter-seeding next to the source saves the downstream path; a leaf saves
    # only itself, so the rumour's remaining size must rank them
    assert probe.score_plan(early) < probe.score_plan(late)

    blind = attach_context(Probe(), _task(budget=1, forward_model=False), environment)
    try:
        blind.score_plan(early)
        raise AssertionError("@native score_plan must raise")
    except StrategyError as error:
        assert "@native" in str(error)


checks = (
    check_registry,
    check_tie_break,
    check_disjoint_cascades,
    check_blocking_works,
    check_levers,
    check_edge_budget_key,
    check_weight_cap,
    check_detection_delay,
    check_removal_expansion,
    check_dominator_tree,
    check_library_contracts,
    check_proximity_beats_degree,
    check_prevented_influence,
    check_no_credit_for_unreachable,
    check_competitive_head,
    check_head_matches_simulator,
    check_anchor_table,
    check_generated_script,
    check_helpers,
    check_mia_scores_do_not_collapse_onto_the_periphery,
    check_plan_oracle_races_the_rumour,
)


if __name__ == "__main__":
    for check in checks:
        check()
        print(f"[ok] {check.__name__}")

    print(f"\n{len(checks)} influence-blocking checks passed")
