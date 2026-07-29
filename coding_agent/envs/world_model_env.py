"""
World-model environment: roll the trained transition model f_theta autoregressively, sampling the next state from its predicted marginals each step.
"""

import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from world_model.wm_data import (
    GraphInput,
    apply_edge_ops,
    build_features,
    build_graph_input,
    edges_to_arrays,
    in_channels,
)
from world_model.wm_model import WorldModel
from coding_agent.types import ActionFn, GraphInfo, State, Trajectory

edge_ops = ("add_edge", "remove_edge", "set_edge_weight")

# Tiny throwaway encoder for the oracle head: q = true edge weight, so the
# encoder output never reaches the transmission model and no checkpoint exists
oracle_hidden_dim = 8
oracle_n_layers = 1


class WorldModelEnvironment:
    def __init__(
        self,
        model: nn.Module,
        graph: GraphInfo,
        diffusion_model: str,
        device: str = "cpu",
        n_samples: int = 20,
        base_seed: int = 0,
    ) -> None:
        self.model = model.to(device).eval()
        self.graph = graph
        self.diffusion_model = diffusion_model
        self.device = torch.device(device)
        self.n_samples = n_samples
        # Seed every rollout uses unless one is named explicitly; shared across
        # candidates so the DIFFERENCE between two strategies is well resolved
        self.base_seed = base_seed

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
        base_seed: int = 0,
    ) -> "WorldModelEnvironment":
        """Rebuild a WorldModel from a train_wm.py results JSON config and load its checkpoint."""

        # Read the config block of a results JSON file
        config = json.loads(Path(results_json).read_text())["config"]
        backbone_kwargs = {
            "n_heads": config["n_heads"],
            "ffn_dim": config["ffn_dim"],
            "alpha": config["gcnii_alpha"],
            "lamda": config["gcnii_lamda"],
        }

        # Reconstruct the exact WorldModel architecture from the config
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

        # Get the checkpoint path
        checkpoint_path = (
            Path(config["ckpt_dir"])
            / f"wm_{config['model']}_{config['diffusion_model']}.pt"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"world-model checkpoint not found: {checkpoint_path}"
            )

        # Load the model weights from the checkpoint file
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )

        return cls(
            model,
            graph,
            config["diffusion_model"],
            device=device,
            n_samples=n_samples,
            base_seed=base_seed,
        )

    @classmethod
    def oracle(
        cls,
        graph: GraphInfo,
        diffusion_model: str,
        device: str = "cpu",
        n_samples: int = 20,
        base_seed: int = 0,
    ) -> "WorldModelEnvironment":
        """Ground-truth dynamics baseline: same rollout machinery, q = true edge weight."""
        if diffusion_model != "IC":
            raise ValueError(
                "oracle dynamics are IC-only: LT thresholds are drawn per episode "
                "and never stored, so no true LT transition function exists — use "
                "the monte_carlo evaluator with a large --mc-runs as the LT proxy"
            )

        model = WorldModel(
            "gcn",
            in_channels=in_channels,
            hidden_dim=oracle_hidden_dim,
            n_layers=oracle_n_layers,
            dropout=0.0,
            head_type="structured_oracle",
            diffusion_model=diffusion_model,
        )

        return cls(
            model,
            graph,
            diffusion_model,
            device=device,
            n_samples=n_samples,
            base_seed=base_seed,
        )

    def _block_graph_input(self, sample_arrays: list[tuple]) -> GraphInput:
        # Disjoint block-diagonal union of every sample's graph: normalization is per-component, so this equals the per-sample GraphInputs stacked
        num_nodes = self.graph.num_nodes

        edge_parts = [
            edge_index + sample * num_nodes
            for sample, (edge_index, _) in enumerate(sample_arrays)
        ]
        weight_parts = [edge_weights for _, edge_weights in sample_arrays]

        return build_graph_input(
            np.concatenate(edge_parts, axis=1),
            np.concatenate(weight_parts),
            num_nodes * len(sample_arrays),
            self.diffusion_model,
            self.device,
        )

    @torch.inference_mode()
    def rollout(
        self, action_fn: ActionFn, horizon: int, budget: int, seed: int | None = None
    ) -> Trajectory:
        start = time.perf_counter()
        seed = self.base_seed if seed is None else seed
        rng = np.random.default_rng(seed)
        num_nodes = self.graph.num_nodes
        num_samples = self.n_samples

        # All samples advance in lockstep: each timestep is ONE block-diagonal
        # forward pass instead of n_samples separate ones. Edge state is
        # copy-on-write; the block input is rebuilt only after an edge op
        base_arrays = edges_to_arrays(self.base_edges)
        sample_edges = [self.base_edges] * num_samples
        sample_arrays = [base_arrays] * num_samples
        block_input = self._block_graph_input(sample_arrays)

        # Per-sample infected/frontier sets
        infected = [set() for _ in range(num_samples)]
        frontier = [set() for _ in range(num_samples)]

        active = [True] * num_samples

        representative_states = [State([], [])]
        representative_actions = []
        representative_counts = [0.0]

        for timestep in range(horizon + 1):
            record_representative = active[0]
            bags = [[] for _ in range(num_samples)]
            bag_dicts = [[] for _ in range(num_samples)]

            for sample in range(num_samples):
                if not active[sample]:
                    continue

                state = State(sorted(infected[sample]), sorted(frontier[sample]))
                bags[sample] = action_fn(state, timestep)
                bag_dicts[sample] = [action.to_dict() for action in bags[sample]]

                # Apply edge operations so the model sees the post-action graph
                if any(
                    action_dict["op"] in edge_ops for action_dict in bag_dicts[sample]
                ):
                    sample_edges[sample] = apply_edge_ops(
                        sample_edges[sample], bag_dicts[sample]
                    )
                    sample_arrays[sample] = edges_to_arrays(sample_edges[sample])
                    block_input = None

            if block_input is None:
                block_input = self._block_graph_input(sample_arrays)

            x_parts = []
            for sample in range(num_samples):
                record = {
                    "state": {
                        "infected": sorted(infected[sample]),
                        "frontier": sorted(frontier[sample]),
                    },
                    "action": bag_dicts[sample],
                    "next_state": {"infected": [], "frontier": []},
                    "next_marginal_infected": {},
                    "next_marginal_frontier": {},
                }
                X, _, _ = build_features(record, sample_arrays[sample][0], num_nodes)
                x_parts.append(X)

            features = torch.from_numpy(np.concatenate(x_parts, axis=0))

            # One forward pass of the model given the block graph
            # Get probabilities with the sigmoid activation function
            probabilities = (
                torch.sigmoid(self.model(features.to(self.device), block_input))
                .cpu()
                .numpy()
                .reshape(num_samples, num_nodes, 2)
            )  # shape: (n_samples, N, 2)

            # Coupled sampling: draw the new infections once from the frontier
            # marginal, then derive both channels (frontier = new wave, infected
            # accumulates through the action semantics)
            # Independent per-channel draws create inconsistent states (ghost spreaders: frontier=1,
            # infected=0) that systematically inflate free-running rollouts
            new_draws = rng.random((num_samples, num_nodes)) < probabilities[:, :, 1]

            for sample in range(num_samples):
                if not active[sample]:
                    continue

                adds = {
                    action_dict["target"]
                    for action_dict in bag_dicts[sample]
                    if action_dict["op"] == "add_node"
                }
                removes = {
                    action_dict["target"]
                    for action_dict in bag_dicts[sample]
                    if action_dict["op"] == "remove_node"
                }

                post_exo_infected = infected[sample] | adds

                if self.diffusion_model == "LT":
                    # LT remove_node returns the node to Susceptible; IC keeps it counted
                    post_exo_infected -= removes

                new_nodes = (
                    set(np.nonzero(new_draws[sample])[0].tolist()) - post_exo_infected
                )

                infected[sample] = post_exo_infected | new_nodes
                frontier[sample] = new_nodes

                # Terminate early once the cascade is dead (empty frontier) and the strategy is idle (empty bag)
                if timestep > 0 and not frontier[sample] and not bags[sample]:
                    active[sample] = False

            if record_representative:
                representative_actions.append(bags[0])
                representative_states.append(
                    State(sorted(infected[0]), sorted(frontier[0]))
                )
                representative_counts.append(float(len(infected[0])))

            if not any(active):
                break

        final_counts = [float(len(infected[sample])) for sample in range(num_samples)]

        final_infected_freq = np.zeros(num_nodes)
        for sample in range(num_samples):
            final_infected_freq[list(infected[sample])] += 1.0

        # The reward is the mean of the final infected node counts
        reward = float(np.mean(final_counts)) if final_counts else 0.0

        reward_se = (
            float(np.std(final_counts, ddof=1) / np.sqrt(len(final_counts)))
            if len(final_counts) > 1
            else 0.0
        )

        states = representative_states
        actions = representative_actions
        counts = representative_counts
        counts[-1] = reward

        return Trajectory(
            states=states,
            actions=actions,
            reward=reward,
            infected_counts=counts,
            cost={
                "n_samples": self.n_samples,
                "env": "world_model",
                # The seed this rollout ran under: the whole ensemble's sampling
                # is drawn from it, so replaying it reproduces the number
                "seed": int(seed),
                "reward_se": reward_se,
                "rollout_seconds": time.perf_counter() - start,
            },
            final_marginals=(final_infected_freq / num_samples).round(3).tolist(),
        )
