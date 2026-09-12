import numpy as np

import pytest

from coding_agent.envs import world_model_env
from coding_agent.envs.world_model_env import WorldModelEnvironment, oracle_hidden_dim
from coding_agent.types import ActionOp, GraphInfo


def ring_graph(num_nodes: int) -> GraphInfo:
    src = np.arange(num_nodes)
    dst = (src + 1) % num_nodes
    edge_index = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    return GraphInfo(
        num_nodes=num_nodes,
        edge_index=edge_index.astype(np.int64),
        ic_probs=np.full(edge_index.shape[1], 0.6, dtype=np.float32),
        directed=False,
    )


def seed_policy(state, timestep):
    return [ActionOp("add_node", 0), ActionOp("add_node", 7)] if timestep == 0 else []


def test_chunked_rollout_matches_the_single_block() -> None:
    graph = ring_graph(16)
    whole = WorldModelEnvironment.oracle(graph, "IC", n_samples=10, base_seed=3)
    # 32 arcs x hidden per sample, so a budget of 70 arcs' worth advances two
    # samples per forward pass
    chunked = WorldModelEnvironment.oracle(graph, "IC", n_samples=10, base_seed=3)
    chunked.max_block_arc_hidden = 70 * oracle_hidden_dim

    reference = whole.rollout(seed_policy, horizon=5, budget=2)
    trajectory = chunked.rollout(seed_policy, horizon=5, budget=2)

    assert trajectory.reward == reference.reward
    assert trajectory.final_marginals == reference.final_marginals
    assert trajectory.spread_curve == reference.spread_curve
    assert chunked.forward_passes == 5 * whole.forward_passes


def test_gradient_probe_refuses_a_graph_it_cannot_hold(monkeypatch) -> None:
    graph = ring_graph(16)
    environment = WorldModelEnvironment.oracle(graph, "IC", n_samples=4, base_seed=3)
    plan = [[ActionOp("add_node", 0)]] + [[] for _ in range(5)]
    assert "best_unselected" in environment.seed_gradient(plan, horizon=5)

    # 32 arcs x 8 hidden x 6 steps = 1,536 arc-hidden-steps; a budget below it refuses with the reason
    monkeypatch.setattr(world_model_env, "max_gradient_arc_hidden_steps", 1_000)
    with pytest.raises(ValueError, match="gradient probe keeps every timestep"):
        environment.seed_gradient(plan, horizon=5)
