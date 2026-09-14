"""
World-model environment: roll the trained transition model f_theta autoregressively, sampling the next state from its predicted marginals each step.
"""

import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from coding_agent.types import ActionFn, GraphInfo, State, Trajectory, pad_counts
from data.wm_competitive import auto_dominance, shared_positive_prob
from data.wm_simulator import blocked, spent
from world_model.checkpoint import load_checkpoint
from world_model.wm_data import (
    GraphInput,
    apply_edge_ops,
    basic_encoding,
    build_competitive_features,
    build_epidemic_features,
    build_features,
    build_graph_input,
    ch_add,
    ch_frontier,
    ch_infected,
    channels_for,
    edges_to_arrays,
    log_degree,
)
from world_model.wm_eval import sample_competitive_step, sample_epidemic_step
from world_model.wm_model import WorldModel

edge_ops = ("add_edge", "remove_edge", "set_edge_weight")

# Tiny throwaway encoder for the oracle head: q = true edge weight, so the
# encoder output never reaches the transmission model and no checkpoint exists
oracle_hidden_dim = 8
oracle_n_layers = 1

# Arcs x hidden units per forward pass. Inference materializes per-arc tensors
# of width hidden in every layer, measured at 29 bytes per arc-hidden under
# message conditioning (17 without it), so 2^30 is about 31 GB per block;
# samples are advanced in chunks under this budget and the pass is repeated
# per chunk. The old constant was 16M arcs whatever the width, which at hidden
# 256 put a 200-sample block of nethept (12.5M arcs) at 93 GB
default_max_block_arc_hidden = 2**30
# The gradient probe keeps every timestep's activations for its one backward
# pass, measured at 103 bytes per arc-hidden-step under message conditioning,
# so this budget is about 28 GB; digg and twitter are refused with a note
max_gradient_arc_hidden_steps = 2**28
# How far inside [0, 1] the relaxed seed vector of `seed_gradient` sits, so the
# head's probability clamp (1e-6) never cuts the gradient
gradient_relaxation = 1e-3


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
        hide_edge_weights: bool = False,
        action_encoding: str = basic_encoding,
        max_block_arc_hidden: int = default_max_block_arc_hidden,
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
        # Both MUST match what the checkpoint was trained under. hide_edge_weights
        # used not to be threaded here at all, so a w-hidden model was rolled out
        # against the true transmission probabilities: the one place in the
        # pipeline where the masking silently did not apply. action_encoding sets
        # in_channels, so a mismatch is a shape error rather than a silent one.
        self.hide_edge_weights = hide_edge_weights
        self.action_encoding = action_encoding
        # CH_DEGREE depends only on the adjacency, so during a rollout with no
        # edge ops it is the same vector at every timestep for every ensemble
        # sample. Recomputing it was 11% of rollout time for a constant. Keyed by
        # id() of the edge array, holding the array itself alongside so its id
        # cannot be recycled while the entry lives, and cleared per rollout.
        self.max_block_arc_hidden = max_block_arc_hidden
        self._degree_cache = {}
        # Filled by every rollout: final infected count per sample block, and the
        # per-step count curve per sample block. The curves are what lets a
        # batched caller (credit's solo rollouts) read stagnation timing without
        # a rollout per policy; equal length across blocks because a dead sample
        # keeps appending its frozen count until the whole call terminates.
        self.last_sample_counts = []
        self.last_sample_curves = []
        # The final infected set of every sample block of the last rollout, so a
        # search loop can read WHICH nodes a candidate and the incumbent reached
        # in the realizations where they differed most (counterexamples). A fresh
        # list per rollout, never mutated in place, so a caller may keep the
        # previous rollout's reference across the next call.
        self.last_sample_final_infected = []
        # Frontier-at-t capture for the `frontier` probe: set the step before a
        # rollout and the rollout leaves P(v on the frontier after that step)
        # across the ensemble here, then clears the request
        self.capture_frontier_step = None
        self.last_frontier_marginals = None
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
        max_block_arc_hidden: int = default_max_block_arc_hidden,
    ) -> "WorldModelEnvironment":
        """Rebuild a WorldModel from a train_wm.py results JSON config and load its checkpoint."""

        # Read the config block of a results JSON file
        config = json.loads(Path(results_json).read_text())["config"]

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
        # checkpoint carries its own spec and rebuilds from it; a legacy bare
        # state_dict is rebuilt from this config, which is the reason this path
        # still takes one. strict_spec=False because the results JSON is the
        # historical source of truth for pre-v2 runs and must keep working.
        model, spec, _ = load_checkpoint(
            checkpoint_path, config=config, device=device, strict_spec=False
        )

        # Follow the resolved SPEC rather than re-reading the config block: for a
        # v2 file it is what the weights were actually fit under, and it carries
        # every layout discriminator (remove_semantics, competitive, epidemic,
        # hide_edge_weights, action_encoding) with the pre-flag defaults filled in
        return cls(
            model,
            graph,
            spec.diffusion_model,
            device=device,
            n_samples=n_samples,
            base_seed=base_seed,
            remove_semantics=spec.remove_semantics,
            negative_seeds=negative_seeds,
            competitive=spec.competitive,
            epidemic=spec.epidemic,
            hide_edge_weights=spec.hide_edge_weights,
            action_encoding=spec.action_encoding,
            max_block_arc_hidden=max_block_arc_hidden,
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
        max_block_arc_hidden: int = default_max_block_arc_hidden,
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
                max_block_arc_hidden=max_block_arc_hidden,
            )

        if diffusion_model not in ("IC", "LT"):
            raise ValueError(
                f"no oracle for diffusion_model {diffusion_model!r}: IC pins "
                f"q = w and LT pins the closed-form threshold hazard; the "
                f"compartmental dynamics take the epidemic branch above"
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
            max_block_arc_hidden=max_block_arc_hidden,
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
            # Which block each node belongs to. Only an action-conditioned encoder
            # reads it, and it MUST: the blocks are separate rollouts (and under
            # the batched candidate form, separate CANDIDATES), so a batch-wide
            # action pool would mix one candidate's seeds into another's messages.
            batch_index=torch.arange(
                len(sample_arrays), device=self.device
            ).repeat_interleave(num_nodes),
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
            X, _, _ = build_features(
                record, edge_index, num_nodes, self.action_encoding
            )

        graph_input = build_graph_input(
            edge_index,
            edge_weight,
            num_nodes,
            self.diffusion_model,
            self.device,
            self.hide_edge_weights,
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
        self,
        action_fn: ActionFn,
        horizon: int,
        budget: int,
        seed: int | None = None,
        num_samples: int | None = None,
    ) -> Trajectory:
        """
        `action_fn` may be a single policy, or a LIST of one policy per block.

        The list form is what makes candidate scoring affordable. Every block in
        this rollout already advances in lockstep through ONE forward pass, so
        putting K candidates x n_samples blocks in that pass costs the same FLOPs
        as K separate rollouts but pays the per-call overhead once. Per-block final
        infected counts land on `self.last_sample_counts` so a caller can split
        them back into per-candidate means; the returned Trajectory keeps its
        existing whole-ensemble meaning.
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

        self._degree_cache.clear()

        # All samples advance in lockstep: each timestep is ONE block-diagonal
        # forward pass instead of n_samples separate ones. Edge state is
        # copy-on-write; the block input is rebuilt only after an edge op
        base_arrays = edges_to_arrays(self.base_edges)
        sample_edges = [self.base_edges] * num_samples
        sample_arrays = [base_arrays] * num_samples
        chunk = max(
            1,
            self.max_block_arc_hidden
            // max(1, int(self.model.hidden_dim) * int(base_arrays[0].shape[1])),
        )
        chunks = [
            range(first, min(first + chunk, num_samples))
            for first in range(0, num_samples, chunk)
        ]
        # One block input per chunk, built lazily and dropped when a member edits
        block_inputs = [None] * len(chunks)

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
                    sample=sample,
                )
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
                    block_inputs[sample // chunk] = None

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
                    edge_array = sample_arrays[sample][0]
                    cached = self._degree_cache.get(id(edge_array))

                    # Identity check, not id() alone: two edge ops on one sample
                    # free the intermediate array, and a later array can land at
                    # the same address with a different degree column
                    if cached is None or cached[0] is not edge_array:
                        cached = (edge_array, log_degree(edge_array, num_nodes))
                        self._degree_cache[id(edge_array)] = cached

                    degrees = cached[1]

                    X, _, _ = build_features(
                        record,
                        edge_array,
                        num_nodes,
                        self.action_encoding,
                        degree_column=degrees,
                    )

                x_parts.append(X)

            features = np.concatenate(x_parts, axis=0)  # shape: (n_samples * N, C)

            # One forward pass per chunk of samples given that chunk's block graph
            columns = 5 if self.epidemic else 4 if self.competitive else 2
            outputs = []
            for index, members in enumerate(chunks):
                if block_inputs[index] is None:
                    block_inputs[index] = self._block_graph_input(
                        [sample_arrays[sample] for sample in members]
                    )
                rows = slice(members.start * num_nodes, members.stop * num_nodes)
                outputs.append(
                    torch.sigmoid(
                        self.model(
                            torch.from_numpy(features[rows]).to(self.device),
                            block_inputs[index],
                        )
                    )
                    .cpu()
                    .numpy()
                )
                self.forward_passes += 1
            probabilities = np.concatenate(outputs, axis=0).reshape(
                num_samples, num_nodes, columns
            )  # shape: (n_samples, N, 2, 4 or 5)

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

            self.last_sample_counts = [float(len(block)) for block in infected]
            self.last_sample_curves = [list(curve) for curve in sample_curves]

            if self.capture_frontier_step == timestep:
                frequency = np.zeros(num_nodes)
                for sample in range(num_samples):
                    frequency[list(frontier[sample])] += 1.0
                self.last_frontier_marginals = frequency / num_samples
                self.capture_frontier_step = None

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
        self.last_sample_final_infected = [set(block) for block in infected]

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
            sample_rewards=final_counts,
        )

    def seed_gradient(
        self, plan: list, horizon: int, top: int = 10
    ) -> dict:
        """
        d(expected final spread) / d(seed indicator) for every node, by one
        differentiable mean-field rollout of f_theta.

        The sampled rollout above draws coin flips and is not differentiable, so
        the gradient runs the model on RELAXED states instead: the seed vector
        `x` in [0, 1]^N is the t=0 add channel, and each step feeds the previous
        step's predicted marginals back as the infected and frontier channels.
        That is the mean-field propagation of the structured head, one forward
        pass per step and one backward pass in total, whatever k is. The
        gradient ranks the nodes NOT in the plan (where leave-one-out credit is
        blind) as well as the ones in it. Single-cascade layouts only: the
        competitive and compartmental samplers draw compartment assignments the
        relaxation has no analogue for.
        """
        if self.competitive or self.epidemic:
            raise ValueError(
                "the gradient probe supports the single-cascade IC and LT layouts "
                "only; this environment runs a competitive or compartmental model"
            )

        start = time.perf_counter()
        num_nodes = self.graph.num_nodes
        seeds = sorted(
            {
                int(action.target)
                for bag in plan[:1]
                for action in bag
                if action.op == "add_node"
            }
        )
        if not seeds:
            raise ValueError("the gradient probe needs a plan with add_node seeds at t=0")

        edge_index, edge_weight = edges_to_arrays(self.base_edges)
        cost = int(edge_index.shape[1]) * int(self.model.hidden_dim) * (horizon + 1)
        if cost > max_gradient_arc_hidden_steps:
            raise ValueError(
                f"the gradient probe keeps every timestep's activations for its "
                f"backward pass, about {cost * 103 / 1e9:.0f} GB on this graph at this "
                f"horizon, over its {max_gradient_arc_hidden_steps * 103 / 1e9:.0f} GB "
                f"budget; use best_swap or add on a graph this size"
            )
        record = {
            "state": {"infected": [], "frontier": []},
            "action": [],
            "next_state": {"infected": [], "frontier": []},
            "next_marginal_infected": {},
            "next_marginal_frontier": {},
        }
        base, _, _ = build_features(record, edge_index, num_nodes, self.action_encoding)
        graph_input = build_graph_input(
            edge_index,
            edge_weight,
            num_nodes,
            self.diffusion_model,
            self.device,
            self.hide_edge_weights,
        )

        with torch.enable_grad():
            base_x = torch.from_numpy(base).to(self.device)
            # The head clamps its probabilities to [eps, 1 - eps] before the logit,
            # which zeroes the gradient of any node sitting exactly at 0 or 1; the
            # seed vector is relaxed just inside that range so every node's
            # self-term survives
            x = torch.full((num_nodes,), gradient_relaxation, device=self.device)
            x[seeds] = 1.0 - gradient_relaxation
            x.requires_grad_(True)
            infected = torch.zeros(num_nodes, device=self.device)
            frontier = torch.zeros(num_nodes, device=self.device)

            for timestep in range(horizon + 1):
                features = base_x.clone()
                features[:, ch_infected] = infected
                features[:, ch_frontier] = frontier
                if timestep == 0:
                    features[:, ch_add] = x
                probabilities = torch.sigmoid(
                    self.model(features, graph_input)
                )  # shape: (N, 2)
                infected, frontier = probabilities[:, 0], probabilities[:, 1]
                self.forward_passes += 1

            expected = infected.sum()
            (gradient,) = torch.autograd.grad(expected, x)

        gradient = gradient.detach().cpu().numpy().astype(np.float64)
        chosen = set(seeds)
        unselected = [node for node in np.argsort(-gradient) if int(node) not in chosen]
        selected = sorted(seeds, key=lambda node: gradient[node])

        self.rollout_calls += 1
        self.evaluator_seconds += time.perf_counter() - start

        return {
            "mean_field_spread": float(expected.item()),
            "best_unselected": [(int(node), float(gradient[node])) for node in unselected[:top]],
            "worst_selected": [(int(node), float(gradient[node])) for node in selected[:top]],
        }
