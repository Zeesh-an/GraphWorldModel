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
    basic_encoding,
    build_features,
    build_graph_input,
    edges_to_arrays,
    log_degree,
    num_input_channels,
)
from world_model.checkpoint import load_checkpoint
from world_model.wm_model import WorldModel
from coding_agent.types import ActionFn, GraphInfo, State, Trajectory, pad_counts
from data.wm_simulator import blocked, spent

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
        remove_semantics: str = spent,
        hide_edge_weights: bool = False,
        action_encoding: str = basic_encoding,
    ) -> None:
        self.model = model.to(device).eval()
        self.graph = graph
        self.diffusion_model = diffusion_model
        self.device = torch.device(device)
        self.n_samples = n_samples
        self.remove_semantics = remove_semantics
        # Both MUST match what the checkpoint was trained under. hide_edge_weights
        # used not to be threaded here at all, so a w-hidden model was rolled out
        # against the true transmission probabilities — the one place in the
        # pipeline where the masking silently did not apply. action_encoding sets
        # in_channels, so a mismatch is a shape error rather than a silent one.
        self.hide_edge_weights = hide_edge_weights
        self.action_encoding = action_encoding
        # Seed every rollout uses unless one is named explicitly; shared across
        # candidates so the DIFFERENCE between two strategies is well resolved
        self.base_seed = base_seed
        # Cumulative inner-loop cost, mirroring MonteCarloEnvironment so the two
        # are directly comparable in the arm table. episodes_used stays 0 by
        # definition: this evaluator consumes no real-environment experience,
        # which is the point of the condition. forward_passes is its own unit of
        # work: one batched pass over the n_samples block graph per timestep.
        self.rollout_calls = 0
        self.evaluator_seconds = 0.0
        self.episodes_used = 0
        self.forward_passes = 0

        # CH_DEGREE depends only on the adjacency, so during a rollout with no
        # edge ops it is the same vector at every timestep for every ensemble
        # sample. Recomputing it was 11% of rollout time for a constant.
        # Invalidated (set to None) the moment an edge op fires.
        self._degree_cache: dict = {}

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
        # Absent in checkpoints trained before --remove-semantics existed, all of
        # which were spent
        remove_semantics = config.get("remove_semantics", spent)
        # Both absent in checkpoints trained before these flags existed, whose
        # behaviour the defaults reproduce exactly
        hide_edge_weights = bool(config.get("hide_edge_weights", False))
        action_encoding = config.get("action_encoding", basic_encoding)
        # Get the checkpoint path
        checkpoint_path = (
            Path(config["ckpt_dir"])
            / f"wm_{config['model']}_{config['diffusion_model']}.pt"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"world-model checkpoint not found: {checkpoint_path}"
            )

        # Reconstruction lives in world_model.checkpoint. A self-describing
        # checkpoint carries its own spec and this config is ignored; a legacy
        # bare-state_dict one is rebuilt from the config, which is the reason
        # this path still takes one. strict_spec=False because the results JSON
        # is the historical source of truth for pre-v2 runs and must keep working.
        model, spec, _ = load_checkpoint(
            checkpoint_path, config=config, device=device, strict_spec=False
        )

        # Follow the CHECKPOINT, not the config block: for a v2 file the spec is
        # what the weights were actually fit under, and a stale results JSON must
        # not be able to flip remove_semantics or hide_edge_weights underneath it.
        return cls(
            model,
            graph,
            spec.diffusion_model,
            device=device,
            n_samples=n_samples,
            base_seed=base_seed,
            remove_semantics=spec.remove_semantics,
            hide_edge_weights=spec.hide_edge_weights,
            action_encoding=spec.action_encoding,
        )

    @classmethod
    def oracle(
        cls,
        graph: GraphInfo,
        diffusion_model: str,
        device: str = "cpu",
        n_samples: int = 20,
        base_seed: int = 0,
        remove_semantics: str = spent,
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
            in_channels=num_input_channels(basic_encoding),
            hidden_dim=oracle_hidden_dim,
            n_layers=oracle_n_layers,
            dropout=0.0,
            head_type="structured_oracle",
            diffusion_model=diffusion_model,
            remove_semantics=remove_semantics,
        )

        return cls(
            model,
            graph,
            diffusion_model,
            device=device,
            n_samples=n_samples,
            base_seed=base_seed,
            remove_semantics=remove_semantics,
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
            self.hide_edge_weights,
        )

    @torch.inference_mode()
    def rollout(
        self, action_fn: ActionFn, horizon: int, budget: int, seed: int | None = None,
        num_samples: int | None = None,
    ) -> Trajectory:
        """
        `action_fn` may be a single policy, or a LIST of one policy per block.

        The list form is what makes candidate scoring affordable. Every block in
        this rollout already advances in lockstep through ONE forward pass, so
        putting K candidates x n_samples blocks in that pass costs the same FLOPs
        as K separate rollouts but pays the per-call overhead once. Measured on
        BA-100, a 2000-node forward spends about 0.56 ms in `linear` against
        roughly 50 us of arithmetic -- ten times more dispatch than compute, and
        that factor is what batching recovers.

        Per-block final infected counts land on `self.last_sample_counts` so a
        caller can split them back into per-candidate means; the returned
        Trajectory keeps its existing whole-ensemble meaning.
        """
        start = time.perf_counter()
        seed = self.base_seed if seed is None else seed
        rng = np.random.default_rng(seed)
        num_nodes = self.graph.num_nodes
        num_samples = self.n_samples if num_samples is None else num_samples
        action_fns = (
            list(action_fn)
            if isinstance(action_fn, (list, tuple))
            else [action_fn] * num_samples
        )

        if len(action_fns) != num_samples:
            raise ValueError(
                f"{len(action_fns)} action functions for {num_samples} blocks; "
                f"the list form needs exactly one policy per block"
            )

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
        # One count vector per sample, so sigma(S, T) is the ensemble mean at T
        # rather than whatever the representative sample happened to do
        sample_curves = [[0.0] for _ in range(num_samples)]

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
                bags[sample] = action_fns[sample](state, timestep)
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
                edge_array = sample_arrays[sample][0]
                key = id(edge_array)
                degrees = self._degree_cache.get(key)

                if degrees is None:
                    degrees = log_degree(edge_array, num_nodes)
                    self._degree_cache[key] = degrees

                X, _, _ = build_features(
                    record,
                    edge_array,
                    num_nodes,
                    self.action_encoding,
                    degree_column=degrees,
                )
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
            self.forward_passes += 1

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

                if self.diffusion_model == "LT" or self.remove_semantics == blocked:
                    # LT remove_node returns the node to Susceptible, and a blocked
                    # node leaves the graph under either dynamics. Only spent IC
                    # keeps the node counted.
                    post_exo_infected -= removes

                new_nodes = (
                    set(np.nonzero(new_draws[sample])[0].tolist()) - post_exo_infected
                )

                infected[sample] = post_exo_infected | new_nodes
                frontier[sample] = new_nodes

                # Terminate early once the cascade is dead (empty frontier) and the strategy is idle (empty bag)
                if timestep > 0 and not frontier[sample] and not bags[sample]:
                    active[sample] = False

            for sample in range(num_samples):
                sample_curves[sample].append(float(len(infected[sample])))

            self.last_sample_counts = [float(len(block)) for block in infected]

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

        elapsed = time.perf_counter() - start
        self.rollout_calls += 1
        self.evaluator_seconds += elapsed

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
                "rollout_seconds": elapsed,
            },
            final_marginals=(final_infected_freq / num_samples).round(3).tolist(),
            spread_curve=np.mean(
                [pad_counts(curve, horizon) for curve in sample_curves], axis=0
            )
            .round(4)
            .tolist(),
        )
