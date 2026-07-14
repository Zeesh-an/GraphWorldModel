import numpy as np
import torch

from coding_agent.envs.world_model_env import WorldModelEnvironment
from coding_agent.types import Action, GraphInfo, State
from world_model.wm_data import in_channels
from world_model.wm_model import WorldModel


def _hub() -> GraphInfo:
    sources = [0, 0, 0, 1, 2, 3]
    destinations = [1, 2, 3, 0, 0, 0]
    edge_index = np.array([sources, destinations], dtype=np.int64)
    ic_probs = np.full(len(sources), 0.9, dtype=np.float32)
    return GraphInfo(
        num_nodes=4, edge_index=edge_index, ic_probs=ic_probs, directed=True
    )


def test_wm_env_rollout_shapes_and_seed_effect():
    torch.manual_seed(0)
    graph = _hub()
    # Untrained structured IC model: structural head still activates the seeded node.
    model = WorldModel(
        "sage",
        in_channels=in_channels,
        hidden_dim=16,
        n_layers=2,
        head_type="structured",
        diffusion_model="IC",
    )
    environment = WorldModelEnvironment(model, graph, "IC", device="cpu", n_samples=8)

    def action_fn(state: State, timestep: int) -> list[Action]:
        return [Action("add_node", 0)] if timestep == 0 else []

    trajectory = environment.rollout(action_fn, horizon=4, budget=1)
    assert len(trajectory.infected_counts) >= 1
    # The seeded hub is infected (T_exo baked into the structured head).
    assert trajectory.reward >= 1.0
