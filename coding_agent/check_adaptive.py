"""Self-check for adaptive IM: the round schedule, the feedback model, and the gap.

The budget split and the state masking are the two places a silent bug would
produce a plausible number rather than a crash, so both are pinned here against
cases whose answer is known by hand.
"""

import networkx as nx
import numpy as np

from coding_agent.envs.monte_carlo_env import MonteCarloEnvironment
from coding_agent.envs.multi_round_env import MultiRoundEnvironment
from coding_agent.envs.world_model_env import WorldModelEnvironment
from coding_agent.executor import StrategyError, _namespace
from coding_agent.methods.base import _AdaptiveAnchor, evaluate_strategy
from coding_agent.prompts import build_round_block, build_system_prompt
from coding_agent.tools.adaptive_algorithms import (
    adapt_degree,
    mc_adaptive_algorithms,
    adapt_degree_discount,
    adaptive_algorithms,
    static_split,
)
from coding_agent.stream import build_stream
from coding_agent.rounds import (
    adaptive_action_fn,
    observed_state,
    round_batches,
    round_schedule,
    round_spreads,
)
from coding_agent.types import (
    ActionOp,
    GraphInfo,
    State,
    TaskSpec,
    full_adoption,
    myopic,
    pad_counts,
)
from data.wm_graphs import kronecker_graph, make_synthetic_bundle
from pipeline.conditions import adaptivity_gaps, parse_arm
from pipeline.run import expand_baselines


def batches_sum_to_the_budget() -> None:
    for budget in range(1, 40):
        for rounds in range(1, 9):
            batches = round_batches(budget, rounds)

            assert sum(batches) == budget, (budget, rounds, batches)
            # Never an empty round: r is capped at k so every batch is real work
            assert all(size >= 1 for size in batches), batches
            assert len(batches) == min(rounds, budget), (budget, rounds, batches)
            # Split as evenly as possible
            assert max(batches) - min(batches) <= 1, batches


def per_round_budget_derives_the_round_count() -> None:
    # Han's b-sweep: fix b, r follows. 10 seeds at b=3 is 3+3+3+1.
    assert round_batches(10, None, per_round_budget=3) == [3, 3, 3, 1]
    assert round_batches(9, None, per_round_budget=3) == [3, 3, 3]
    assert sum(round_batches(17, None, per_round_budget=5)) == 17
    # b >= k collapses to the non-adaptive case, one round of everything
    assert round_batches(4, None, per_round_budget=9) == [4]


def the_schedule_refuses_to_overrun_the_horizon() -> None:
    assert round_schedule([1, 1, 1], round_gap=2, horizon=4) == {0: 1, 2: 1, 4: 1}

    try:
        round_schedule([1, 1, 1], round_gap=3, horizon=4)
    except ValueError as error:
        # Silently dropping the last batch would spend less than k and report it
        # as k, so this must raise rather than clip
        assert "past --horizon" in str(error), error
    else:
        raise AssertionError("a schedule running past the horizon must raise")


def myopic_hides_the_accumulated_state() -> None:
    state = State(infected=[1, 2, 3], frontier=[3])

    assert observed_state(state, full_adoption).infected == [1, 2, 3]
    assert observed_state(state, myopic).infected == []
    # The wave survives under both: it is the myopic observation
    assert observed_state(state, myopic).frontier == [3]


def rounds_fire_only_on_schedule_and_respect_their_batch() -> None:
    graph = GraphInfo(
        num_nodes=10,
        edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
        ic_probs=np.ones(2, dtype=np.float32),
        directed=True,
    )
    task = TaskSpec(budget=4, horizon=6, rounds=2, round_gap=2)
    batches = round_batches(task.budget, task.rounds, task.per_round_budget)
    assert batches == [2, 2]

    class Greedy:
        def act(self, state, graph, timestep):
            fresh = [
                node
                for node in range(graph.num_nodes)
                if node not in set(state.infected) | set(state.frontier)
            ]
            return [ActionOp("add_node", node) for node in fresh[:2]]

    action_fn = adaptive_action_fn(Greedy(), task, graph, batches)

    assert len(action_fn(State([], []), 0)) == 2
    assert action_fn(State([0, 1], [1]), 1) == []  # between rounds: no input
    assert len(action_fn(State([0, 1], [1]), 2)) == 2
    assert action_fn(State([0, 1], [1]), 3) == []


def over_committing_a_round_is_rejected() -> None:
    graph = GraphInfo(
        num_nodes=10,
        edge_index=np.array([[0], [1]], dtype=np.int64),
        ic_probs=np.ones(1, dtype=np.float32),
        directed=True,
    )
    task = TaskSpec(budget=4, horizon=6, rounds=2)

    class Greedy:
        def act(self, state, graph, timestep):
            # Three seeds in a round whose batch is two
            return [ActionOp("add_node", node) for node in (5, 6, 7)]

    action_fn = adaptive_action_fn(Greedy(), task, graph, [2, 2])

    try:
        action_fn(State([], []), 0)
    except StrategyError:
        pass
    else:
        raise AssertionError("a round may not exceed its own batch size")


def reseeding_is_an_error_when_visible_and_a_wasted_slot_when_not() -> None:
    """The one asymmetry between the feedback models, and why it has to exist."""
    graph = GraphInfo(
        num_nodes=10,
        edge_index=np.array([[0], [1]], dtype=np.int64),
        ic_probs=np.ones(1, dtype=np.float32),
        directed=True,
    )

    class Stubborn:
        def act(self, state, graph, timestep):
            return [ActionOp("add_node", 3)]

    # full_adoption: the policy was handed state.infected, so this is its bug
    visible = TaskSpec(budget=2, horizon=4, rounds=2, feedback_model=full_adoption)
    action_fn = adaptive_action_fn(Stubborn(), visible, graph, [1, 1])
    action_fn(State([], []), 0)

    try:
        action_fn(State([3], [3]), 1)
    except StrategyError as error:
        assert "ALREADY ACTIVE" in str(error), error
    else:
        raise AssertionError("a visible re-seed must be rejected")

    # myopic: the policy could NOT see it. Erroring would make the arm
    # unrunnable, and letting the op through would re-arm a spent IC spreader
    # (add_node writes status 1 over NDlib's status 2), so the seed is dropped.
    blind = TaskSpec(budget=2, horizon=4, rounds=2, feedback_model=myopic)
    action_fn = adaptive_action_fn(Stubborn(), blind, graph, [1, 1])

    assert len(action_fn(State([], []), 0)) == 1
    assert action_fn(State([3], [3]), 1) == [], "a blind re-seed must be dropped"


def round_spreads_read_the_boundaries() -> None:
    counts = [0.0, 5.0, 9.0, 12.0, 14.0]

    assert round_spreads(counts, [2, 2], round_gap=1) == [5.0, 9.0]
    assert round_spreads(counts, [1, 1, 1], round_gap=2) == [5.0, 12.0]
    # A cascade that died early leaves fewer readings than rounds, not an error
    assert round_spreads([0.0, 3.0], [1, 1, 1], round_gap=1) == [3.0]


def evaluate_strategy_dispatches_on_the_task() -> None:
    bundle = make_synthetic_bundle("er", index=0, num_nodes=30, er_p=0.15, seed=1)
    graph = GraphInfo(
        num_nodes=bundle.nx_graph.number_of_nodes(),
        edge_index=bundle.edge_index.astype(np.int64),
        ic_probs=bundle.ic_probs.astype(np.float32),
        directed=False,
    )
    environment = MonteCarloEnvironment(graph, "IC", mc_runs=3, base_seed=0)

    class Both:
        source_script = ""

        def plan_horizon(self, graph, budget, horizon):
            return [[ActionOp("add_node", node) for node in range(budget)]] + [
                [] for _ in range(horizon)
            ]

        def act(self, state, graph, timestep):
            active = set(state.infected) | set(state.frontier)
            fresh = [node for node in range(graph.num_nodes) if node not in active]
            return [ActionOp("add_node", fresh[0])] if fresh else []

    static = TaskSpec(budget=4, horizon=5)
    adaptive = TaskSpec(budget=4, horizon=5, rounds=4)

    static_trajectory, _ = evaluate_strategy(Both(), environment, graph=graph, task=static)
    adaptive_trajectory, _ = evaluate_strategy(
        Both(), environment, graph=graph, task=adaptive
    )

    # Both spend the same k; only WHEN differs, which is the whole comparison
    for trajectory, expected in ((static_trajectory, 4), (adaptive_trajectory, 4)):
        seeded = {
            action.target
            for bag in trajectory.actions
            for action in bag
            if action.op == "add_node"
        }
        assert len(seeded) == expected, (expected, seeded)

    # The static plan commits at t=0; the adaptive one spreads across rounds
    assert len(static_trajectory.actions[0]) == 4
    assert len(adaptive_trajectory.actions[0]) == 1


def arms_and_prompts_know_about_adaptive() -> None:
    arm = parse_arm("adaptive_free@world_model")
    assert arm.method == "adaptive" and arm.condition == 6, arm

    task = TaskSpec(budget=6, horizon=8, rounds=3, allowed_ops=("add_node",))
    block = build_round_block(task)

    assert "ROUND SCHEDULE (3 rounds" in block, block
    assert "t=0" in block and "t=1" in block and "t=2" in block, block
    # A static task must not carry a round block at all
    assert build_round_block(TaskSpec(budget=6, horizon=8)) == ""

    system = build_system_prompt("adaptive", "free", task)
    assert "ADAPTIVE MULTI-ROUND POLICY" in system
    # seed_timing_note ("put every seed at t=0") is exactly wrong here
    assert "Put every seed in element 0" not in system
    assert "you do NOT choose when your seeds land" in system


def the_gap_pairs_matched_arms_only() -> None:
    def record(method, evaluator, budget, spread):
        return {
            "method": method,
            "evaluator": evaluator,
            "budget": budget,
            "arm": f"{method}_free@{evaluator}",
            "reward": spread,
            "mc_reward": spread,
            "rounds": 4 if method == "adaptive" else None,
        }

    results = [
        record("adaptive", "oracle", 10, 55.0),
        record("evolve", "oracle", 10, 50.0),
        record("adaptive", "world_model", 10, 52.0),
        record("evolve", "world_model", 10, 50.0),
        # No control at this budget, so no gap can be formed
        record("adaptive", "oracle", 20, 80.0),
    ]
    gaps = adaptivity_gaps(results)

    assert len(gaps) == 2, gaps
    assert {entry["evaluator"] for entry in gaps} == {"oracle", "world_model"}
    assert abs(gaps[0]["gap"] - 1.1) < 1e-9, gaps[0]
    # Never crosses evaluators: 55/50 and 52/50, not 55/50 and 52/50 mixed
    by_evaluator = {entry["evaluator"]: entry["gap"] for entry in gaps}
    assert abs(by_evaluator["world_model"] - 1.04) < 1e-9, by_evaluator


def every_adaptive_baseline_obeys_the_round_contract() -> None:
    """
    The two rules prepare_round_bag enforces, checked at the source.

    A policy that breaks either would fail loudly in a real run, but only on the
    graph and round where it happens; pinning it here means a new baseline cannot
    be added with an off-by-one in its batch cap.
    """
    bundle = make_synthetic_bundle("powerlaw_cluster", index=0, num_nodes=60, seed=3)
    graph = GraphInfo(
        num_nodes=bundle.nx_graph.number_of_nodes(),
        edge_index=bundle.edge_index.astype(np.int64),
        ic_probs=bundle.ic_probs.astype(np.float32),
        directed=False,
    )
    state = State(infected=[0, 1, 2, 3, 4], frontier=[3, 4])
    active = set(state.infected) | set(state.frontier)

    for name, policy in adaptive_algorithms.items():
        for batch in (1, 3):
            seeds = policy(state, graph, batch, "IC", total_budget=9)

            assert len(seeds) <= batch, (name, batch, seeds)
            assert len(set(seeds)) == len(seeds), (name, "duplicate seed", seeds)
            assert not (set(seeds) & active), (name, "seeded an active node", seeds)
            assert all(0 <= node < graph.num_nodes for node in seeds), (name, seeds)

    # ...and on an empty state, which is what round 0 always sees
    for name, policy in adaptive_algorithms.items():
        seeds = policy(State([], []), graph, 3, "IC", total_budget=9)
        assert len(seeds) == 3, (name, seeds)


def adaptive_baselines_react_and_static_split_does_not() -> None:
    """The behavioural difference the whole task turns on."""
    bundle = make_synthetic_bundle("powerlaw_cluster", index=1, num_nodes=80, seed=5)
    graph = GraphInfo(
        num_nodes=bundle.nx_graph.number_of_nodes(),
        edge_index=bundle.edge_index.astype(np.int64),
        ic_probs=bundle.ic_probs.astype(np.float32),
        directed=False,
    )
    empty = State([], [])

    # static_split hands out its own static ranking in order, so round 2's picks
    # are just the next slice of the SAME list round 1 came from
    first = static_split(empty, graph, 3, "IC", total_budget=9)
    committed = State(infected=list(first), frontier=list(first))
    second = static_split(committed, graph, 3, "IC", total_budget=9)
    whole = static_split(empty, graph, 9, "IC", total_budget=9)

    assert first == whole[:3], (first, whole)
    assert second == whole[3:6], (second, whole)

    # An adaptive policy re-ranks against what the cascade actually reached: mark
    # a hub's whole neighbourhood active and it stops being the top pick
    degrees = [graph.degree(node) for node in range(graph.num_nodes)]
    hub = int(np.argmax(degrees))
    assert adapt_degree(empty, graph, 1, "IC")[0] == hub

    reached = State(infected=graph.out_neighbors(hub), frontier=[])
    assert adapt_degree_discount(reached, graph, 1, "IC")[0] != hub


def adaptive_baselines_run_through_the_real_environment() -> None:
    """Canned-script path end to end: schedule, batch cap, recorded round spreads."""
    bundle = make_synthetic_bundle("powerlaw_cluster", index=2, num_nodes=60, seed=7)
    graph = GraphInfo(
        num_nodes=bundle.nx_graph.number_of_nodes(),
        edge_index=bundle.edge_index.astype(np.int64),
        ic_probs=bundle.ic_probs.astype(np.float32),
        directed=False,
    )
    environment = MonteCarloEnvironment(graph, "IC", mc_runs=3, base_seed=0)
    task = TaskSpec(budget=6, horizon=5, rounds=3)
    batches = round_batches(task.budget, task.rounds, task.per_round_budget)

    for name, policy in adaptive_algorithms.items():
        wrapped = _AdaptiveAnchor(policy, task)
        trajectory, _ = evaluate_strategy(wrapped, environment, task, graph)

        committed = [
            action.target
            for bag in trajectory.actions
            for action in bag
            if action.op == "add_node"
        ]
        assert len(committed) == sum(batches), (name, len(committed), batches)
        assert len(set(committed)) == len(committed), (name, "reseeded", committed)
        # Seeds land only on round timesteps
        for timestep, bag in enumerate(trajectory.actions):
            expected = timestep % task.round_gap == 0 and timestep < len(batches)
            assert bool(bag) == expected, (name, timestep, bag)


def the_gap_divides_by_the_best_static_arm() -> None:
    """The denominator is max over static arms, not a nominated one (§1.1)."""
    def record(method, arm, spread):
        return {
            "method": method,
            "arm": arm,
            "evaluator": "oracle",
            "budget": 10,
            "reward": spread,
            "mc_reward": spread,
        }

    results = [
        record("adaptive", "baseline_adapt_greedy", 60.0),
        record("one_shot", "baseline_celf_pp", 50.0),
        record("one_shot", "baseline_random_seeds", 20.0),
        record("evolve", "evolve_free@oracle", 55.0),
    ]
    gaps = adaptivity_gaps(results)

    assert len(gaps) == 1, gaps
    # 55 (the best static arm), not 50 and not 20
    assert gaps[0]["control_arm"] == "evolve_free@oracle", gaps[0]
    assert abs(gaps[0]["gap"] - 60.0 / 55.0) < 1e-9, gaps[0]


def the_stream_is_deterministic_and_balanced() -> None:
    """Exogenous means same schedule for every episode and every sample."""
    bundle = make_synthetic_bundle("powerlaw_cluster", index=0, num_nodes=120, seed=2)
    graph = GraphInfo(
        num_nodes=120,
        edge_index=bundle.edge_index.astype(np.int64),
        ic_probs=bundle.ic_probs.astype(np.float32),
        directed=False,
    )
    stream = build_stream(graph, horizon=6, rate=0.05, seed=11)
    assert build_stream(graph, 6, 0.0, 11) is None, "rate 0 must mean no stream"

    # Order-independent: the WM env asks (timestep, sample), the MC env asks
    # (episode, timestep), and both must get the same graph history
    forward = [stream.edits(t) for t in range(7)]
    backward = [stream.edits(t) for t in reversed(range(7))][::-1]
    keys = lambda ops: [(o.op, o.target, o.destination) for o in ops]  # noqa: E731
    assert [keys(o) for o in forward] == [keys(o) for o in backward]

    # Nothing at t=0: the policy picked its first seeds against the base graph
    assert forward[0] == []
    assert all(forward[t] for t in range(1, 7)), "the stream must actually edit"

    # Rewiring, not densification: edge count stays within a few percent
    base, final = len(stream.edges_at(0)), len(stream.edges_at(6))
    assert abs(final - base) / base < 0.05, (base, final)

    # ...and the graph the policy is handed actually moves
    assert stream.graph_at(6).edge_index.shape[1] != 0
    early = {tuple(pair) for pair in stream.graph_at(1).edge_index.T.tolist()}
    late = {tuple(pair) for pair in stream.graph_at(6).edge_index.T.tolist()}
    assert early != late, "graph_at must reflect the accumulated edits"


def the_stream_reaches_every_arm_and_only_adaptive_sees_it() -> None:
    bundle = make_synthetic_bundle("powerlaw_cluster", index=3, num_nodes=80, seed=4)
    graph = GraphInfo(
        num_nodes=80,
        edge_index=bundle.edge_index.astype(np.int64),
        ic_probs=bundle.ic_probs.astype(np.float32),
        directed=False,
    )
    environment = MonteCarloEnvironment(graph, "IC", mc_runs=3, base_seed=0)
    seen_edges = []

    class Watcher:
        source_script = ""

        def plan_horizon(self, graph, budget, horizon):
            return [[ActionOp("add_node", n) for n in range(budget)]] + [
                [] for _ in range(horizon)
            ]

        def act(self, state, graph, timestep):
            # Records the edge SET it was handed, not the count: the stream is
            # balanced by construction, so the count is deliberately constant and
            # only the identity of the edges moves
            seen_edges.append(
                hash(tuple(sorted(map(tuple, graph.edge_index.T.tolist()))))
            )
            active = set(state.infected) | set(state.frontier)
            fresh = [n for n in range(graph.num_nodes) if n not in active]
            return [ActionOp("add_node", fresh[0])] if fresh else []

    streamed = TaskSpec(budget=4, horizon=5, rounds=4, edit_rate=0.05, seed=3)
    trajectory, _ = evaluate_strategy(Watcher(), environment, streamed, graph)

    # The edits ride in the bag, so they show up in the executed actions even
    # though --allowed-ops would have rejected them from the policy
    edge_ops = [
        action
        for bag in trajectory.actions
        for action in bag
        if action.op in ("add_edge", "remove_edge")
    ]
    assert edge_ops, "the stream must reach the simulator"

    # An adaptive policy is handed a graph that changes between rounds
    assert len(set(seen_edges)) > 1, "graph_at must move between rounds"

    # A static arm gets the same stream (fairness) but cannot react to it
    static = TaskSpec(budget=4, horizon=5, edit_rate=0.05, seed=3)
    static_trajectory, _ = evaluate_strategy(Watcher(), environment, static, graph)
    static_edge_ops = [
        action
        for bag in static_trajectory.actions
        for action in bag
        if action.op in ("add_edge", "remove_edge")
    ]
    assert static_edge_ops, "the stream must apply to non-adaptive arms too"


def multi_round_unions_separate_campaigns() -> None:
    bundle = make_synthetic_bundle("powerlaw_cluster", index=4, num_nodes=100, seed=6)
    graph = GraphInfo(
        num_nodes=100,
        edge_index=bundle.edge_index.astype(np.int64),
        ic_probs=bundle.ic_probs.astype(np.float32),
        directed=False,
    )
    inner = MonteCarloEnvironment(graph, "IC", mc_runs=6, base_seed=0)
    seen_unions = []

    def action_fn(state, timestep):
        if timestep != 0:
            return []
        seen_unions.append(len(state.infected))
        active = set(state.infected)
        fresh = [n for n in range(graph.num_nodes) if n not in active]
        return [ActionOp("add_node", n) for n in fresh[:4]]

    single = inner.rollout(action_fn, horizon=6, budget=4)

    # The single-campaign rollout above also recorded, so drop its entries before
    # reading the per-campaign ones below
    seen_unions.clear()
    multi = MultiRoundEnvironment(inner, campaigns=3, base_seed=0)
    union = multi.rollout(action_fn, horizon=6, budget=4)

    # The union over 3 campaigns must exceed one campaign, and cannot exceed N
    assert union.reward > single.reward, (union.reward, single.reward)
    assert union.reward <= graph.num_nodes
    assert union.cost["campaigns"] == 3
    assert len(union.cost["campaign_rewards"]) == 3

    # Each campaign is a FRESH diffusion, so its own reward is single-campaign
    # sized; only the union is larger
    assert all(value < union.reward for value in union.cost["campaign_rewards"])

    # `infected` carries the union into the next campaign (§2.4e), so the policy
    # sees a growing set at t=0 of each campaign. The MC environment loops
    # (episode, then timestep), so t=0 fires once per episode and each campaign
    # contributes mc_runs entries — the union is a per-campaign constant, which
    # is what makes it safe to read under either environment's loop order.
    per_campaign = 6
    starts = [seen_unions[index * per_campaign] for index in range(3)]

    assert starts[0] == 0, "campaign 1 must start from an empty union"
    assert starts[1] > 0, "campaign 2 must see campaign 1's activations"
    assert starts[2] >= starts[1], "the union may only grow"
    # ...and it really is constant within a campaign
    assert len(set(seen_unions[:per_campaign])) == 1, seen_unions[:per_campaign]

    # Counters belong to the inner env: the wrapper runs no episodes of its own
    assert multi.episodes_used == inner.episodes_used
    assert multi.rollout_calls == inner.rollout_calls


def the_spread_curve_makes_sigma_s_t_readable() -> None:
    bundle = make_synthetic_bundle("powerlaw_cluster", index=5, num_nodes=90, seed=8)
    graph = GraphInfo(
        num_nodes=90,
        edge_index=bundle.edge_index.astype(np.int64),
        ic_probs=bundle.ic_probs.astype(np.float32),
        directed=False,
    )
    environment = MonteCarloEnvironment(graph, "IC", mc_runs=6, base_seed=0)
    horizon = 8
    trajectory = environment.rollout(
        lambda state, t: [ActionOp("add_node", n) for n in range(3)] if t == 0 else [],
        horizon,
        3,
    )
    curve = trajectory.spread_curve

    # Always horizon + 2 long, whatever the cascade did, so index t means the
    # same t across arms — the ragged infected_counts is what made sigma(S, T)
    # unreadable before
    assert len(curve) == horizon + 2, len(curve)
    assert len(trajectory.infected_counts) <= len(curve)

    # Monotone under IC, and the endpoint IS the reward
    assert all(later >= earlier for earlier, later in zip(curve, curve[1:]))
    # 1e-4, not 1e-6: the curve is rounded to 4 dp on the way into the JSON
    assert abs(curve[-1] - trajectory.reward) < 1e-4, (curve[-1], trajectory.reward)

    # Padding is exact, not smoothing: once the cascade dies the value repeats
    assert curve[-1] == curve[-2]

    # pad_counts on an already-full vector is the identity
    full = list(range(horizon + 2))
    assert pad_counts([float(v) for v in full], horizon) == [float(v) for v in full]


def adaptive_runs_under_the_world_model_loop_order() -> None:
    """
    The (timestep, sample) nesting, which MonteCarloEnvironment never exercises.

    Every other check here drives the MC environment, which loops (episode, then
    timestep). WorldModelEnvironment loops the other way and calls action_fn once
    per sample per timestep, so a round schedule that quietly depended on call
    order would pass all of them and fail on condition 6 — the arm the whole
    project is about. The oracle head needs no checkpoint, so this pins the loop
    order without a trained model.
    """
    bundle = make_synthetic_bundle("powerlaw_cluster", index=6, num_nodes=70, seed=9)
    graph = GraphInfo(
        num_nodes=70,
        edge_index=bundle.edge_index.astype(np.int64),
        ic_probs=bundle.ic_probs.astype(np.float32),
        directed=False,
    )
    environment = WorldModelEnvironment.oracle(graph, "IC", n_samples=6, base_seed=0)

    class Greedy:
        source_script = ""

        def act(self, state, graph, timestep):
            active = set(state.infected) | set(state.frontier)
            fresh = [node for node in range(graph.num_nodes) if node not in active]
            return [ActionOp("add_node", node) for node in fresh[:2]]

    task = TaskSpec(budget=6, horizon=5, rounds=3)
    trajectory, _ = evaluate_strategy(Greedy(), environment, task, graph)

    committed = [
        action.target
        for bag in trajectory.actions
        for action in bag
        if action.op == "add_node"
    ]
    # Per SAMPLE, not shared across the ensemble: a cross-call counter would make
    # sample 2 onwards run out of budget at t=0
    assert len(committed) == 6, committed
    assert environment.forward_passes > 0
    assert len(trajectory.spread_curve) == task.horizon + 2

    # ...and the same holds with a stream and with campaigns wrapped around it
    streamed = TaskSpec(budget=6, horizon=5, rounds=3, edit_rate=0.05, seed=2)
    evaluate_strategy(Greedy(), environment, streamed, graph)

    multi = MultiRoundEnvironment(environment, campaigns=3, base_seed=0)
    before = environment.forward_passes
    union, _ = evaluate_strategy(Greedy(), multi, task, graph)
    assert environment.forward_passes > before, "campaigns must drive the inner env"
    assert union.cost["campaigns"] == 3


def the_expensive_adaptive_baseline_is_metered() -> None:
    """
    adapt_greedy must be blocked from GENERATED code and runnable as a baseline.

    It re-simulates every candidate on a private simulator, so a generated act()
    calling it spends thousands of episodes that `episodes_used` never sees. That
    is the same honesty hole executor.py documents at length for celf and
    vanilla_greedy, and it stayed open until this was wired: the
    mc_adaptive_algorithms constant existed but nothing read it.
    """
    namespace = _namespace("free", allow_mc_algorithms=False)

    try:
        namespace["adaptive_algorithms"].adapt_greedy(None, None, 1)
    except StrategyError as error:
        assert "adapt_greedy" in str(error), error
    else:
        raise AssertionError("adapt_greedy must be blocked from generated scripts")

    # The cheap policies stay callable: blocking them would delete the ideas the
    # prompt is trying to hand over
    for name in ("adapt_epic", "adapt_degree_discount", "static_split"):
        assert name not in mc_adaptive_algorithms, name

    # ...and a declared --baselines arm IS the expensive algorithm, so that path
    # must still run it
    allowed = _namespace("free", allow_mc_algorithms=True)
    assert callable(allowed["adaptive_algorithms"].adapt_greedy)


def published_baselines_cannot_cross_tasks() -> None:
    """An IM repo in an adaptive table would be a category error, not a datapoint."""
    assert expand_baselines(("external:adaptiveim",), "adaptive_online_im") == [
        "external:adaptiveim"
    ]
    assert expand_baselines(("external:moeim",), "influence_maximization") == [
        "external:moeim"
    ]

    try:
        expand_baselines(("external:moeim",), "adaptive_online_im")
    except ValueError as error:
        assert "influence_maximization" in str(error), error
    else:
        raise AssertionError("an IM baseline must not join an adaptive sweep")

    # The `all` aliases already filtered; every adaptive repo is unwired, so this
    # is empty and must stay honest about that rather than expanding to the IM set
    assert expand_baselines(("all-external",), "adaptive_online_im") == []


def the_new_synthetic_families_generate() -> None:
    plc = make_synthetic_bundle("powerlaw_cluster", index=0, num_nodes=200, seed=1)
    assert plc.nx_graph.number_of_nodes() == 200
    assert plc.graph_id.startswith("plc_n200_m2"), plc.graph_id
    # Triangles are the whole reason this family exists over plain BA
    assert nx.transitivity(plc.nx_graph) > 0.0

    for variant in ("core_periphery", "random", "hierarchical"):
        graph, order = kronecker_graph(64, variant, seed=0)
        assert graph.number_of_nodes() == 64
        assert order == 64
        assert graph.number_of_edges() > 0, variant

    # Non-powers of two are induced from the next power up, and say so
    kron = make_synthetic_bundle("kronecker", index=0, num_nodes=100, seed=1)
    assert kron.nx_graph.number_of_nodes() == 100
    assert "_o128_" in kron.graph_id, kron.graph_id


if __name__ == "__main__":
    checks = [
        batches_sum_to_the_budget,
        per_round_budget_derives_the_round_count,
        the_schedule_refuses_to_overrun_the_horizon,
        myopic_hides_the_accumulated_state,
        rounds_fire_only_on_schedule_and_respect_their_batch,
        over_committing_a_round_is_rejected,
        reseeding_is_an_error_when_visible_and_a_wasted_slot_when_not,
        round_spreads_read_the_boundaries,
        evaluate_strategy_dispatches_on_the_task,
        arms_and_prompts_know_about_adaptive,
        the_gap_pairs_matched_arms_only,
        every_adaptive_baseline_obeys_the_round_contract,
        adaptive_baselines_react_and_static_split_does_not,
        adaptive_baselines_run_through_the_real_environment,
        the_gap_divides_by_the_best_static_arm,
        the_stream_is_deterministic_and_balanced,
        the_stream_reaches_every_arm_and_only_adaptive_sees_it,
        multi_round_unions_separate_campaigns,
        the_spread_curve_makes_sigma_s_t_readable,
        adaptive_runs_under_the_world_model_loop_order,
        the_expensive_adaptive_baseline_is_metered,
        published_baselines_cannot_cross_tasks,
        the_new_synthetic_families_generate,
    ]

    for check in checks:
        check()
        print(f"ok  {check.__name__}")

    print(f"\n{len(checks)} checks passed")
