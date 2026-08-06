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
    build_competitive_features,
    build_epidemic_features,
    build_features,
    build_graph_input,
    channels_for,
    edges_to_arrays,
)
from world_model.wm_eval import sample_competitive_step, sample_epidemic_step
from world_model.wm_model import WorldModel
from coding_agent.types import ActionFn, GraphInfo, State, Trajectory, pad_counts
from data.wm_competitive import auto_dominance, shared_positive_prob
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
        negative_seeds: tuple = (),
        competitive: bool = False,
        epidemic: bool = False,
    ) -> None:
        self.model = model.to(device).eval()
        self.graph = graph
        self.diffusion_model = diffusion_model
        self.device = torch.device(device)
        self.n_samples = n_samples
        self.remove_semantics = remove_semantics
        # Influence blocking: 8-channel features, a 4-column head, and a starting
        # state in which S_N is already committed. `reward` is still the negative
        # cascade's final size because State maps it onto `infected`.
        self.competitive = competitive
        # Epidemic control: 9-channel features, a 5-column head, and a compartment
        # sampler instead of the monotone one. `reward` is still the attack set's
        # size because State maps it onto `infected`.
        self.epidemic = epidemic
        self.negative_seeds = tuple(int(node) for node in negative_seeds)
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
        negative_seeds: tuple = (),
    ) -> "WorldModelEnvironment":
        """Rebuild a WorldModel from a train_wm.py results JSON config and load its checkpoint."""

        # Read the config block of a results JSON file
        config = json.loads(Path(results_json).read_text())["config"]
        # Absent in checkpoints trained before --remove-semantics existed, all of
        # which were spent
        remove_semantics = config.get("remove_semantics", spent)
        # ...and absent in every checkpoint trained before influence blocking existed
        competitive = bool(config.get("competitive", False))
        # ...and absent in every checkpoint trained before epidemic control existed
        epidemic = bool(config.get("epidemic", False))
        backbone_kwargs = {
            "n_heads": config["n_heads"],
            "ffn_dim": config["ffn_dim"],
            "alpha": config["gcnii_alpha"],
            "lamda": config["gcnii_lamda"],
        }

        # Reconstruct the exact WorldModel architecture from the config
        model = WorldModel(
            config["model"],
            in_channels=channels_for(competitive, epidemic)[0],
            hidden_dim=config["hidden_dim"],
            n_layers=config["n_layers"],
            dropout=config["dropout"],
            head_type=config.get("head", "linear"),
            diffusion_model=config["diffusion_model"],
            remove_semantics=remove_semantics,
            competitive=competitive,
            tie_break=config.get("tie_break", auto_dominance),
            positive_prob=config.get("positive_prob"),
            epidemic=epidemic,
            # The rates the CHECKPOINT was fit under, not the ones a flag names:
            # the oracle head pins its matrix to them and a mismatch would score
            # the arm against a transition the simulator never produced
            epi_beta=config.get("beta_scale", 1.0),
            epi_gamma=config.get("gamma"),
            epi_alpha=config.get("alpha"),
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
            remove_semantics=remove_semantics,
            negative_seeds=negative_seeds,
            competitive=competitive,
            epidemic=epidemic,
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
        negative_seeds: tuple = (),
        competitive: bool = False,
        tie_break: str = auto_dominance,
        positive_prob: str | float = shared_positive_prob,
        epidemic: bool = False,
        # PREFIXED to match WorldModel's, which had to be: `backbone_kwargs` there
        # already carries GCNII's own `alpha`
        epi_beta: float = 1.0,
        epi_gamma: float | None = None,
        epi_alpha: float | None = None,
    ) -> "WorldModelEnvironment":
        """Ground-truth dynamics baseline: same rollout machinery, q = true edge weight."""
        if epidemic:
            # The compartment head's oracle form is EXACT rather than merely
            # well-shaped: q = beta_scale * w, gamma_hat = gamma, alpha_hat = alpha
            # reproduces the simulator's own one-step marginals to MC precision, so
            # this really is the ground-truth ceiling and not an approximation of it
            model = WorldModel(
                "gcn",
                in_channels=channels_for(epidemic=True)[0],
                hidden_dim=oracle_hidden_dim,
                n_layers=oracle_n_layers,
                dropout=0.0,
                head_type="structured_oracle",
                diffusion_model=diffusion_model,
                remove_semantics=remove_semantics,
                epidemic=True,
                epi_beta=epi_beta,
                epi_gamma=epi_gamma,
                epi_alpha=epi_alpha,
            )

            return cls(
                model,
                graph,
                diffusion_model,
                device=device,
                n_samples=n_samples,
                base_seed=base_seed,
                remove_semantics=remove_semantics,
                epidemic=True,
            )

        if diffusion_model != "IC":
            raise ValueError(
                "oracle dynamics are IC-only: LT thresholds are drawn per episode "
                "and never stored, so no true LT transition function exists: use "
                "the monte_carlo evaluator with a large --mc-runs as the LT proxy"
            )

        model = WorldModel(
            "gcn",
            in_channels=channels_for(competitive)[0],
            hidden_dim=oracle_hidden_dim,
            n_layers=oracle_n_layers,
            dropout=0.0,
            head_type="structured_oracle",
            diffusion_model=diffusion_model,
            remove_semantics=remove_semantics,
            competitive=competitive,
            tie_break=tie_break,
            # The oracle head pins q to the TRUE probability, so under MCICM it has
            # to be told the limiting campaign's constant: the edge weight does not
            # carry it, and pinning both campaigns to p would silently simulate COICM
            positive_prob=(
                None if positive_prob == shared_positive_prob else float(positive_prob)
            ),
        )

        return cls(
            model,
            graph,
            diffusion_model,
            device=device,
            n_samples=n_samples,
            base_seed=base_seed,
            remove_semantics=remove_semantics,
            negative_seeds=negative_seeds,
            competitive=competitive,
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
    def step_marginals(self, state: State, seed: int | None = None) -> np.ndarray:
        """
        `P(v newly infected at t + 1 | s_t)` in ONE forward pass of f_theta.

        The world-model binding of research/cascade_reconstruction.md §2.5.2, and
        the reason that file calls this the task where the model most clearly earns
        its place: the sampling bindings pay `mc_runs` real episodes per call and
        this pays one batched matmul, at ~10^4 calls per decoded instance (§2.4.2).

        Column 1 of the head is `next_frontier`: the nodes that activate on THIS
        step, which is exactly the kernel a decoder proposes against. Column 0
        (`next_infected`) is the accumulated set and would double-count everything
        already infected.

        Under the `oracle` head `q` is pinned to the true edge weight, so this is
        the analytic IC form `1 - prod(1 - p_uv * frontier_u)` evaluated exactly,
        with no sampling error at all.
        """
        start = time.perf_counter()
        num_nodes = self.graph.num_nodes
        edge_index, edge_weight = edges_to_arrays(self.base_edges)

        record = {
            "state": {
                "infected": sorted(int(node) for node in state.infected),
                "frontier": sorted(int(node) for node in state.frontier),
                "pos_infected": sorted(int(node) for node in state.pos_infected),
                "pos_frontier": sorted(int(node) for node in state.pos_frontier),
                "exposed": sorted(int(node) for node in state.exposed),
                "recovered": sorted(int(node) for node in state.recovered),
            },
            "action": [],
            "next_state": {"infected": [], "frontier": []},
            "next_marginal_infected": {},
            "next_marginal_frontier": {},
            "next_marginal_pos_infected": {},
            "next_marginal_pos_frontier": {},
            "next_marginal_incidence": {},
            "next_marginal_exposed": {},
            "next_marginal_infectious": {},
            "next_marginal_recovered": {},
        }

        if self.epidemic:
            X, _ = build_epidemic_features(record, edge_index, num_nodes)
        elif self.competitive:
            X, _ = build_competitive_features(record, edge_index, num_nodes)
        else:
            X, _, _ = build_features(record, edge_index, num_nodes)

        graph_input = build_graph_input(
            edge_index, edge_weight, num_nodes, self.diffusion_model, self.device
        )
        logits = self.model(
            torch.from_numpy(X).to(self.device), graph_input
        )  # shape: (N, 2 or 4)
        marginal = torch.sigmoid(logits[:, 1]).cpu().numpy().astype(np.float64)

        self.forward_passes += 1
        self.rollout_calls += 1
        self.evaluator_seconds += time.perf_counter() - start

        return marginal

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

        # Per-sample state. Under competition every sample starts with S_N already
        # committed and spreading, which is the premise of the task rather than an
        # initial condition to choose.
        seeds = set(self.negative_seeds) if self.competitive else set()
        infected = [set(seeds) for _ in range(num_samples)]
        frontier = [set(seeds) for _ in range(num_samples)]
        pos_infected = [set() for _ in range(num_samples)]
        pos_frontier = [set() for _ in range(num_samples)]
        exposed = [set() for _ in range(num_samples)]
        recovered = [set() for _ in range(num_samples)]

        active = [True] * num_samples
        # One count vector per sample, so sigma(S, T) is the ensemble mean at T
        # rather than whatever the representative sample happened to do
        sample_curves = [[float(len(seeds))] for _ in range(num_samples)]
        # ...and the PREVALENCE, |I(t)|, which under a compartmental task is a
        # different curve from the cumulative one and is the one §2.6 grades on
        sample_prevalence = [[0.0] for _ in range(num_samples)]

        representative_states = [State(sorted(seeds), sorted(seeds))]
        representative_actions = []
        representative_counts = [float(len(seeds))]

        for timestep in range(horizon + 1):
            record_representative = active[0]
            bags = [[] for _ in range(num_samples)]
            bag_dicts = [[] for _ in range(num_samples)]

            for sample in range(num_samples):
                if not active[sample]:
                    continue

                state = State(
                    sorted(infected[sample]),
                    sorted(frontier[sample]),
                    sorted(pos_infected[sample]),
                    sorted(pos_frontier[sample]),
                    sorted(exposed[sample]),
                    sorted(recovered[sample]),
                )
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
                        "pos_infected": sorted(pos_infected[sample]),
                        "pos_frontier": sorted(pos_frontier[sample]),
                        "exposed": sorted(exposed[sample]),
                        "recovered": sorted(recovered[sample]),
                    },
                    "action": bag_dicts[sample],
                    "next_state": {"infected": [], "frontier": []},
                    "next_marginal_infected": {},
                    "next_marginal_frontier": {},
                    "next_marginal_pos_infected": {},
                    "next_marginal_pos_frontier": {},
                    "next_marginal_incidence": {},
                    "next_marginal_exposed": {},
                    "next_marginal_infectious": {},
                    "next_marginal_recovered": {},
                }

                if self.epidemic:
                    X, _ = build_epidemic_features(
                        record, sample_arrays[sample][0], num_nodes
                    )
                elif self.competitive:
                    X, _ = build_competitive_features(
                        record, sample_arrays[sample][0], num_nodes
                    )
                else:
                    X, _, _ = build_features(record, sample_arrays[sample][0], num_nodes)

                x_parts.append(X)

            features = torch.from_numpy(np.concatenate(x_parts, axis=0))

            # One forward pass of the model given the block graph
            # Get probabilities with the sigmoid activation function
            columns = 5 if self.epidemic else 4 if self.competitive else 2
            probabilities = (
                torch.sigmoid(self.model(features.to(self.device), block_input))
                .cpu()
                .numpy()
                .reshape(num_samples, num_nodes, columns)
            )  # shape: (n_samples, N, 2 or 4)
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

                if self.epidemic:
                    # One draw per node against the cumulative (E, I, R, S)
                    # distribution, so the four EXCLUSIVE compartments cannot both
                    # claim a node: the compartmental version of the same coupling
                    state = sample_epidemic_step(
                        State(
                            sorted(infected[sample]),
                            sorted(frontier[sample]),
                            exposed=sorted(exposed[sample]),
                            recovered=sorted(recovered[sample]),
                        ),
                        bag_dicts[sample],
                        probabilities[sample],
                        rng,
                        num_nodes,
                        self.diffusion_model,
                    )
                    infected[sample] = set(state.infected)
                    frontier[sample] = set(state.frontier)
                    exposed[sample] = set(state.exposed)
                    recovered[sample] = set(state.recovered)

                    # A latent node with nobody infectious left is still going to
                    # become infectious, so E counts as alive
                    if (
                        timestep > 0
                        and not frontier[sample]
                        and not exposed[sample]
                        and not bags[sample]
                    ):
                        active[sample] = False

                    continue

                if self.competitive:
                    # One draw per node decides both whether it activates and which
                    # cascade takes it, so a node can never land in both: the same
                    # coupling the tie-break enforces in the simulator
                    state = sample_competitive_step(
                        State(
                            sorted(infected[sample]),
                            sorted(frontier[sample]),
                            sorted(pos_infected[sample]),
                            sorted(pos_frontier[sample]),
                        ),
                        bag_dicts[sample],
                        probabilities[sample],
                        rng,
                        num_nodes,
                    )
                    infected[sample] = set(state.infected)
                    frontier[sample] = set(state.frontier)
                    pos_infected[sample] = set(state.pos_infected)
                    pos_frontier[sample] = set(state.pos_frontier)

                    if (
                        timestep > 0
                        and not frontier[sample]
                        and not pos_frontier[sample]
                        and not bags[sample]
                    ):
                        active[sample] = False

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
                sample_prevalence[sample].append(float(len(frontier[sample])))

            if record_representative:
                representative_actions.append(bags[0])
                representative_states.append(
                    State(
                        sorted(infected[0]),
                        sorted(frontier[0]),
                        sorted(pos_infected[0]),
                        sorted(pos_frontier[0]),
                        sorted(exposed[0]),
                        sorted(recovered[0]),
                    )
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
            # Zero-padded rather than held, for the reason the MC environment's is:
            # a dead epidemic's prevalence IS zero, and holding the last value would
            # report a standing infectious population that ended several steps ago
            prevalence_curve=(
                np.mean(
                    [
                        curve + [0.0] * (horizon + 2 - len(curve))
                        for curve in sample_prevalence
                    ],
                    axis=0,
                )
                .round(4)
                .tolist()
                if self.epidemic
                else None
            ),
        )
