"""Self-check for the remove_node semantics fix and the per-arm cost counters.

Runs the two dynamics against both readings on a graph where the answer is known
by hand, so a regression in active_nodes(), the deletion bag, or the head's T_exo
fails here rather than as a quietly biased containment number.
"""

import networkx as nx
import numpy as np
import torch

from coding_agent.envs.monte_carlo_env import MonteCarloEnvironment
from coding_agent.envs.world_model_env import WorldModelEnvironment
from coding_agent.types import GraphInfo
from data.wm_actions import counterfactual_actions, delete_node_bag, sample_injection
from data.wm_simulator import ActionOp, Simulator, State, blocked, spent
from world_model.wm_data import build_features, build_graph_input
from world_model.wm_model import WorldModel

# 0 -> 1 -> 2 with certain transmission: seeding 0 infects the whole path
path_edges = {(0, 1): 1.0, (1, 2): 1.0}
seed_bag = [ActionOp("add_node", 0)]


def build(model_name: str, remove_semantics: str) -> Simulator:
    graph = nx.DiGraph()
    graph.add_nodes_from(range(3))
    graph.add_edges_from(path_edges)

    simulator = Simulator(
        graph, ic_prob_map=dict(path_edges), seed=0, remove_semantics=remove_semantics
    )
    simulator.reset(model_name)

    return simulator


def spent_ic_counts_the_removed_node() -> None:
    simulator = build("IC", spent)
    simulator.advance(seed_bag)
    state = simulator.advance([ActionOp("remove_node", 1)])

    # This is the +k bias: node 1 was never infected, and it counts anyway
    assert 1 in state.infected, state


def blocked_ic_does_not_count_it() -> None:
    simulator = build("IC", blocked)
    simulator.advance(seed_bag)
    bag = delete_node_bag(simulator.model.graph.graph, 1)
    state = simulator.advance(bag)

    assert 1 not in state.infected, state
    # And the cascade is cut: 2 was only reachable through 1
    simulator.advance([])
    assert 2 not in simulator.current_state().infected, simulator.current_state()


def blocked_lt_cannot_reactivate() -> None:
    # Every threshold at 0, so any active in-neighbour flips a node under LT
    graph = nx.DiGraph()
    graph.add_nodes_from(range(3))
    graph.add_edges_from(path_edges)
    simulator = Simulator(graph, ic_prob_map=dict(path_edges), seed=0)
    simulator.remove_semantics = blocked
    simulator.reset("LT", lt_thresholds={node: 0.0 for node in range(3)})

    simulator.advance(seed_bag)
    simulator.advance(delete_node_bag(simulator.model.graph.graph, 1))

    for _ in range(3):
        state = simulator.advance([])
        assert 1 not in state.infected, state


def spent_lt_can_reactivate() -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from(range(3))
    graph.add_edges_from(path_edges)
    simulator = Simulator(graph, ic_prob_map=dict(path_edges), seed=0)
    simulator.reset("LT", lt_thresholds={node: 0.0 for node in range(3)})

    simulator.advance(seed_bag)
    simulator.advance([ActionOp("remove_node", 1)])
    # Node 0 is still active and 1 kept its edges, so it comes back
    state = simulator.advance([])

    assert 1 in state.infected, state


def bare_remove_node_is_still_held_down() -> None:
    """The _enforce_blocked guard: blocked without the incident-edge ops."""
    graph = nx.DiGraph()
    graph.add_nodes_from(range(3))
    graph.add_edges_from(path_edges)
    simulator = Simulator(graph, ic_prob_map=dict(path_edges), seed=0)
    simulator.remove_semantics = blocked
    simulator.reset("LT", lt_thresholds={node: 0.0 for node in range(3)})

    simulator.advance(seed_bag)
    simulator.advance([ActionOp("remove_node", 1)])

    for _ in range(3):
        state = simulator.advance([])
        assert 1 not in state.infected, state


def deletion_bag_covers_both_orientations() -> None:
    undirected = nx.Graph([(0, 1), (1, 2)])
    ops = delete_node_bag(undirected, 1)

    assert ops[0].op == "remove_node" and ops[0].target == 1
    edges = {(op.target, op.destination) for op in ops[1:]}
    assert edges == {(1, 0), (0, 1), (1, 2), (2, 1)}, edges

    directed = nx.DiGraph([(0, 1), (1, 2)])
    edges = {(op.target, op.destination) for op in delete_node_bag(directed, 1)[1:]}
    assert edges == {(1, 2), (0, 1)}, edges


def snapshot_rewinds_the_blocked_set() -> None:
    simulator = build("IC", blocked)
    simulator.advance(seed_bag)

    state = simulator.snapshot()
    simulator.apply_actions([ActionOp("remove_node", 1)])
    assert simulator.blocked == {1}

    simulator.restore(state)
    assert simulator.blocked == set(), simulator.blocked


def blocked_forks_are_deletion_bags() -> None:
    """
    A blocked remove_node fork is a full node-DELETION bag, not a bare op.

    Skipping the fork instead (what this used to assert) left a containment
    dataset with no two actions from the same state, which is exactly what
    `action_sensitivity` measures: it read 0.0.
    """
    graph = nx.DiGraph([(0, 1), (1, 2)])
    rng = np.random.default_rng(0)
    state = State(infected=[0], frontier=[0])

    bags = counterfactual_actions(
        state, graph, [], 4, rng, ["add_node", "remove_node"], remove_semantics=blocked
    )
    removals = [bag for bag in bags if any(op.op == "remove_node" for op in bag)]
    assert removals, bags
    assert any(op.op == "remove_edge" for op in removals[0]), removals[0]

    bags = counterfactual_actions(
        state, graph, [], 4, rng, ["add_node", "remove_node"], remove_semantics=spent
    )
    assert any(op.op == "remove_node" for bag in bags for op in bag), bags


def revert_edges_undoes_a_deletion_fork() -> None:
    """
    ...and the caller must put those edges back, because restore() cannot.

    Without this the main branch resumes on a graph the fork edited: node 1's
    arcs would be gone, so the cascade from 0 would stop dead at 0 forever.
    """
    simulator = build("IC", blocked)
    simulator.advance(seed_bag)
    snapshot = simulator.snapshot()

    fork = delete_node_bag(simulator.model.graph.graph, 1)
    simulator.advance(fork)
    simulator.restore(snapshot)
    simulator.revert_edges(fork)

    # The main branch's cascade still reaches node 2 through the restored arcs
    simulator.advance([])
    state = simulator.advance([])
    assert 2 in state.infected, state


def blocked_injection_expands_to_a_deletion() -> None:
    graph = nx.DiGraph([(0, 1), (1, 2)])
    state = State(infected=[0], frontier=[0])

    bag = sample_injection(
        state,
        graph,
        np.random.default_rng(0),
        p_inject=1.0,
        action_ops=["remove_node"],
        weight_range=(0.0, 1.0),
        remove_semantics=blocked,
    )

    assert any(op.op == "remove_node" for op in bag), bag
    assert any(op.op == "remove_edge" for op in bag), bag


def head_t_exo_matches_the_simulator() -> None:
    """Under blocked the IC head must zero a removed node, under spent it must not."""
    edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)
    weights = np.ones(2, dtype=np.float32)
    record = {
        "state": {"infected": [0, 1], "frontier": [1]},
        "action": [{"op": "remove_node", "target": 1}],
        "next_marginal_infected": {},
        "next_marginal_frontier": {},
    }
    X, _, _ = build_features(record, edge_index, 3)
    graph_input = build_graph_input(edge_index, weights, 3, "IC", torch.device("cpu"))

    for semantics, expect_infected in ((spent, True), (blocked, False)):
        model = WorldModel(
            "gcn",
            hidden_dim=8,
            n_layers=1,
            head_type="structured_oracle",
            diffusion_model="IC",
            remove_semantics=semantics,
        ).eval()

        with torch.inference_mode():
            probabilities = torch.sigmoid(model(torch.from_numpy(X), graph_input))

        infected = float(probabilities[1, 0]) > 0.5
        assert infected == expect_infected, (semantics, float(probabilities[1, 0]))


def environments_count_their_own_cost() -> None:
    graph_info = GraphInfo(
        num_nodes=3,
        edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
        ic_probs=np.ones(2, dtype=np.float32),
        directed=True,
    )
    action_fn = lambda state, timestep: seed_bag if timestep == 0 else []  # noqa: E731

    mc = MonteCarloEnvironment(graph_info, "IC", mc_runs=4, base_seed=0)
    mc.rollout(action_fn, horizon=3, budget=1)
    mc.rollout(action_fn, horizon=3, budget=1)

    assert mc.rollout_calls == 2, mc.rollout_calls
    assert mc.episodes_used == 8, mc.episodes_used
    assert mc.evaluator_seconds > 0.0

    wm = WorldModelEnvironment.oracle(graph_info, "IC", n_samples=4, base_seed=0)
    wm.rollout(action_fn, horizon=3, budget=1)

    assert wm.rollout_calls == 1, wm.rollout_calls
    # The model-based arms consume no real experience; that is the condition
    assert wm.episodes_used == 0, wm.episodes_used
    assert wm.forward_passes > 0, wm.forward_passes
    assert wm.evaluator_seconds > 0.0


if __name__ == "__main__":
    checks = [
        spent_ic_counts_the_removed_node,
        blocked_ic_does_not_count_it,
        blocked_lt_cannot_reactivate,
        spent_lt_can_reactivate,
        bare_remove_node_is_still_held_down,
        deletion_bag_covers_both_orientations,
        snapshot_rewinds_the_blocked_set,
        blocked_forks_are_deletion_bags,
        revert_edges_undoes_a_deletion_fork,
        blocked_injection_expands_to_a_deletion,
        head_t_exo_matches_the_simulator,
        environments_count_their_own_cost,
    ]

    for check in checks:
        check()
        print(f"ok  {check.__name__}")

    print(f"\n{len(checks)} checks passed")
