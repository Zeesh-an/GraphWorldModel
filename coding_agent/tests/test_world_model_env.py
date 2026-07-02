import numpy as np
import torch

from coding_agent.envs.world_model_env import WorldModelEnvironment
from coding_agent.types import Action, GraphInfo
from world_model.wm_data import in_channels
from world_model.wm_model import WorldModel


def _hub() -> GraphInfo:
    src = [0, 0, 0, 1, 2, 3]
    dst = [1, 2, 3, 0, 0, 0]
    ei = np.array([src, dst], dtype=np.int64)
    ic = np.full(len(src), 0.9, dtype=np.float32)
    return GraphInfo(num_nodes=4, edge_index=ei, ic_probs=ic, directed=True)


def test_wm_env_rollout_shapes_and_seed_effect():
    torch.manual_seed(0)
    g = _hub()
    # Untrained structured IC model: structural head still activates the seeded node.
    model = WorldModel(
        "sage",
        in_channels=in_channels,
        hidden_dim=16,
        n_layers=2,
        head_type="structured",
        diffusion_model="IC",
    )
    env = WorldModelEnvironment(model, g, "IC", device="cpu", n_samples=8)

    def action_fn(state, t):
        return [Action("add_node", 0)] if t == 0 else []

    tr = env.rollout(action_fn, horizon=4, budget=1)
    assert len(tr.infected_counts) >= 1
    assert (
        tr.reward >= 1.0
    )  # the seeded hub is infected (T_exo baked into structured head)
