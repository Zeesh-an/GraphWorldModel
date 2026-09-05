import numpy as np

from coding_agent.envs.world_model_env import WorldModelEnvironment
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


def test_chunked_rollout_matches_the_single_block():
    graph = ring_graph(16)
    whole = WorldModelEnvironment.oracle(graph, "IC", n_samples=10, base_seed=3)
    # 32 arcs per graph, so a 70-arc cap advances two samples per forward pass
    chunked = WorldModelEnvironment.oracle(graph, "IC", n_samples=10, base_seed=3)
    chunked.max_block_arcs = 70

    reference = whole.rollout(seed_policy, horizon=5, budget=2)
    trajectory = chunked.rollout(seed_policy, horizon=5, budget=2)

    assert trajectory.reward == reference.reward
    assert trajectory.final_marginals == reference.final_marginals
    assert trajectory.spread_curve == reference.spread_curve
    assert chunked.forward_passes == 5 * whole.forward_passes
