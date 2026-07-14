"""
World-model environment: roll the trained transition model f_theta autoregressively,
sampling the next state from its predicted marginals each step.
"""

import json
from pathlib import Path
import numpy as np
import torch

from world_model.wm_data import (
    apply_edge_ops,
    build_features,
    build_graph_input,
    edges_to_arrays,
    in_channels,
)
from world_model.wm_model import WorldModel
from coding_agent.types import ActionFn, GraphInfo, State, Trajectory

edge_ops = ("add_edge", "remove_edge", "set_edge_weight")


class WorldModelEnvironment:
    def __init__(
        self,
        model: torch.nn.Module,
        graph: GraphInfo,
        diffusion_model: str,
        device: str = "cpu",
        n_samples: int = 20,
    ) -> None:
        self.model = model.to(device).eval()
        self.graph = graph
        self.diffusion_model = diffusion_model
        self.device = torch.device(device)
        self.n_samples = n_samples
        self.base_edges = {
            (int(graph.edge_index[0, edge]), int(graph.edge_index[1, edge])): float(
                graph.ic_probs[edge]
            )
            for edge in range(graph.edge_index.shape[1])
        }

    @classmethod
    def from_results_json(
        cls,
        results_json: str,
        graph: GraphInfo,
        device: str = "cpu",
        n_samples: int = 20,
    ) -> "WorldModelEnvironment":
        """Rebuild a WorldModel from a train_wm.py results JSON config and load its checkpoint."""
        config = json.loads(Path(results_json).read_text())["config"]
        backbone_kwargs = {
            "n_heads": config["n_heads"],
            "ffn_dim": config["ffn_dim"],
            "alpha": config["gcnii_alpha"],
            "lamda": config["gcnii_lamda"],
        }
        model = WorldModel(
            config["model"],
            in_channels=in_channels,
            hidden_dim=config["hidden_dim"],
            n_layers=config["n_layers"],
            dropout=config["dropout"],
            head_type=config.get("head", "linear"),
            diffusion_model=config["diffusion_model"],
            **backbone_kwargs,
        )
        checkpoint_path = (
            Path(config["ckpt_dir"])
            / f"wm_{config['model']}_{config['diffusion_model']}.pt"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"world-model checkpoint not found: {checkpoint_path}"
            )

        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )

        return cls(
            model, graph, config["diffusion_model"], device=device, n_samples=n_samples
        )

    @torch.inference_mode()
    def _single_rollout(
        self, action_fn: ActionFn, horizon: int, rng: np.random.Generator
    ) -> tuple[list[State], list[list], list[float]]:
        num_nodes = self.graph.num_nodes
        edges = dict(self.base_edges)
        infected = set()
        frontier = set()
        states = [State([], [])]
        actions = []
        counts = [0.0]

        for timestep in range(horizon + 1):
            state = State(sorted(infected), sorted(frontier))
            bag = action_fn(state, timestep)
            actions.append(bag)

            # Apply edge ops so the model sees the post-action graph.
            bag_dicts = [action.to_dict() for action in bag]
            if any(action_dict["op"] in edge_ops for action_dict in bag_dicts):
                edges = apply_edge_ops(edges, bag_dicts)

            edge_index, edge_weights = edges_to_arrays(edges)
            record = {
                "state": {"infected": sorted(infected), "frontier": sorted(frontier)},
                "action": bag_dicts,
                "next_state": {"infected": [], "frontier": []},
                "next_marginal_infected": {},
                "next_marginal_frontier": {},
            }
            X, _, _ = build_features(record, edge_index, num_nodes)
            graph_input = build_graph_input(
                edge_index, edge_weights, num_nodes, self.diffusion_model, self.device
            )
            probabilities = (
                torch.sigmoid(
                    self.model(torch.from_numpy(X).to(self.device), graph_input)
                )
                .cpu()
                .numpy()
            )  # shape: (N, 2)
            infected_draw = rng.random(num_nodes) < probabilities[:, 0]
            frontier_draw = rng.random(num_nodes) < probabilities[:, 1]
            infected = set(np.nonzero(infected_draw)[0].tolist())
            frontier = set(np.nonzero(frontier_draw)[0].tolist())
            states.append(State(sorted(infected), sorted(frontier)))
            counts.append(float(len(infected)))
            if timestep > 0 and not frontier and not bag:
                break

        return states, actions, counts

    def rollout(
        self, action_fn: ActionFn, horizon: int, budget: int, seed: int = 0
    ) -> Trajectory:
        rng = np.random.default_rng(seed)
        final_counts = []
        representative = None

        for sample in range(self.n_samples):
            states, actions, counts = self._single_rollout(action_fn, horizon, rng)
            final_counts.append(counts[-1])
            if sample == 0:
                representative = (states, actions, counts)

        reward = float(np.mean(final_counts)) if final_counts else 0.0
        states, actions, counts = representative  # type: ignore[misc]
        counts[-1] = reward

        return Trajectory(
            states=states,
            actions=actions,
            reward=reward,
            infected_counts=counts,
            cost={"n_samples": self.n_samples, "env": "world_model"},
        )
