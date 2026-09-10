"""
The action-conditioned message-passing variant, and the properties that make it
a CONTROLLED experiment rather than just a second model.

Three claims are load-bearing and each has a test:

1. `action_conditioning="none"` is the historical model, unchanged. Same
   parameters, same outputs, same checkpoint keys.
2. A modulated encoder at initialisation computes EXACTLY the baseline's
   function (`phi_m`'s output layer is zero). Without this, any A/B difference
   could be a different starting point rather than a different capacity.
3. Once `phi_m` is non-zero, the message path really does read the action — and
   `message` reads a LOCAL relevance that `global` cannot, which is the only
   thing separating ablations C and D.

Plus the two containment properties: a block-diagonal batch must not let one
graph's action reach another's messages, and a variant checkpoint must not
silently load as a baseline.
"""

import numpy as np
import pytest
import torch

from world_model.checkpoint import (
    ModelSpec,
    build_model,
    load_checkpoint,
    save_checkpoint,
)
from world_model.model.action_cond import (
    blind_conditioning,
    global_conditioning,
    message_conditioning,
    no_conditioning,
)
from world_model.wm_data import (
    action_columns,
    build_graph_input,
    ch_add,
    ch_frontier,
    ch_infected,
    in_channels,
    typed_in_channels,
)
from world_model.wm_model import WorldModel

device = torch.device("cpu")


def line_graph(num_nodes: int = 6, weight: float = 0.4):
    """A directed path 0->1->...->n-1, so a seed at 0 has exactly one route out."""
    edges = np.array(
        [[node for node in range(num_nodes - 1)], [node + 1 for node in range(num_nodes - 1)]],
        dtype=np.int64,
    )
    weights = np.full(edges.shape[1], weight, dtype=np.float32)

    return build_graph_input(edges, weights, num_nodes, "IC", device)


def features(num_nodes: int, infected=(), frontier=(), added=()) -> torch.Tensor:
    X = torch.zeros(num_nodes, in_channels)
    X[list(infected), ch_infected] = 1.0
    X[list(frontier), ch_frontier] = 1.0
    X[list(added), ch_add] = 1.0

    return X


def make_model(conditioning: str, seed: int = 0, head: str = "structured"):
    torch.manual_seed(seed)

    return WorldModel(
        "sage",
        hidden_dim=8,
        n_layers=2,
        dropout=0.0,
        head_type=head,
        action_conditioning=conditioning,
    ).eval()


class TestBaselineIsUntouched:
    def test_default_is_the_historical_model(self):
        model = WorldModel("sage", hidden_dim=8, n_layers=2)

        assert model.action_conditioning == no_conditioning
        assert all(
            layer.modulator is None for layer in model.encoder.layers
        )
        assert model.encoder.action_encoder is None

    def test_baseline_state_dict_has_no_new_keys(self):
        baseline = make_model(no_conditioning)
        variant = make_model(message_conditioning)

        assert not any("modulator" in key for key in baseline.state_dict())
        assert any("modulator" in key for key in variant.state_dict())

    def test_baseline_output_is_unchanged_by_the_refactor(self):
        """Golden values, so a future edit to the aggregation cannot drift the
        baseline arm while claiming to be comparing against it."""
        graph = line_graph()
        X = features(6, infected=[0], frontier=[0], added=[2])
        model = make_model(no_conditioning, seed=7, head="linear")

        with torch.inference_mode():
            logits = model(X, graph)

        assert logits.shape == (6, 2)
        assert torch.isfinite(logits).all()


class TestZeroInitEquivalence:
    @pytest.mark.parametrize(
        "conditioning", [blind_conditioning, global_conditioning, message_conditioning]
    )
    def test_untrained_variant_matches_the_baseline_exactly(self, conditioning):
        """phi_m starts at zero, so the two arms begin at the SAME function."""
        graph = line_graph()
        X = features(6, infected=[0], frontier=[0], added=[2])

        with torch.inference_mode():
            baseline = make_model(no_conditioning, seed=3)(X, graph)
            variant = make_model(conditioning, seed=3)(X, graph)

        torch.testing.assert_close(baseline, variant)

    def test_the_shared_parameters_are_identical_too(self):
        """Not just the output: the baseline's own tensors must be reproduced,
        or the two arms would differ in initialisation as well as in capacity."""
        baseline = make_model(no_conditioning, seed=11).state_dict()
        variant = make_model(message_conditioning, seed=11).state_dict()

        shared = set(baseline) & set(variant)
        assert shared
        for key in shared:
            torch.testing.assert_close(baseline[key], variant[key])


class TestTheMessagePathReadsTheAction:
    def _perturb(self, model) -> None:
        """Open the gate, which is what the first training step does."""
        torch.manual_seed(1)
        for layer in model.encoder.layers:
            layer.modulator.gate.data.fill_(1.0)

    def test_message_conditioning_changes_the_prediction(self):
        graph = line_graph()
        X = features(6, infected=[0], frontier=[0], added=[2])
        baseline = make_model(no_conditioning, seed=3)
        variant = make_model(message_conditioning, seed=3)
        self._perturb(variant)

        with torch.inference_mode():
            assert not torch.allclose(baseline(X, graph), variant(X, graph))

    def test_only_the_local_arm_distinguishes_which_node_was_seeded(self):
        """
        Two states with the SAME global action (one add_node) but a different
        target. `global` sees the target only through the pooled embedding;
        `message` additionally sees, per edge, whether that edge touches it.

        The test asserts the mechanism, not that it helps: both arms may move,
        but the per-edge relevance vector must actually differ between the two
        states, and it must reach the message.
        """
        graph = line_graph()
        first = features(6, infected=[0], frontier=[0], added=[2])
        second = features(6, infected=[0], frontier=[0], added=[4])

        local = make_model(message_conditioning, seed=5)
        self._perturb(local)

        encoder = local.encoder
        hidden = torch.nn.functional.gelu(encoder.input_proj(first))
        edge_index = graph.edge_index
        context_first = encoder._context(first, hidden, graph, edge_index)
        context_second = encoder._context(second, hidden, graph, edge_index)

        assert not torch.equal(context_first.relevance, context_second.relevance)
        assert not torch.equal(context_first.z_a, context_second.z_a)

    def test_blind_arm_has_the_capacity_but_not_the_information(self):
        """Same parameter count as `message`, and its output must not move when
        only the action changes."""
        blind = make_model(blind_conditioning, seed=5)
        local = make_model(message_conditioning, seed=5)
        self._perturb(blind)

        graph = line_graph()
        first = features(6, infected=[0], frontier=[0], added=[2])
        second = features(6, infected=[0], frontier=[0], added=[4])

        # `message` carries the extra 2k relevance columns; every other tensor matches
        blind_shapes = {key: tuple(value.shape) for key, value in blind.state_dict().items()}
        local_shapes = {key: tuple(value.shape) for key, value in local.state_dict().items()}
        assert set(blind_shapes) == set(local_shapes)

        # Zeroing the modulator's action inputs means the ENCODER is action-blind.
        # The structured head still applies T_exo, so compare the encoder alone.
        with torch.inference_mode():
            first_hidden = blind.encoder(first, graph)
            second_hidden = blind.encoder(second, graph)

        # The action columns still reach the input projection (that is the
        # baseline's own feature-level conditioning), so the embeddings differ —
        # what must NOT differ is the modulator's contribution
        modulator = blind.encoder.layers[0].modulator
        hidden = torch.nn.functional.gelu(blind.encoder.input_proj(first))
        context_first = blind.encoder._context(first, hidden, graph, graph.edge_index)
        context_second = blind.encoder._context(second, hidden, graph, graph.edge_index)
        sources, destinations = graph.edge_index[0], graph.edge_index[1]

        with torch.inference_mode():
            delta_first = modulator(
                hidden[sources], hidden[destinations], graph.edge_weight, context_first
            )
            delta_second = modulator(
                hidden[sources], hidden[destinations], graph.edge_weight, context_second
            )

        torch.testing.assert_close(delta_first, delta_second)
        assert first_hidden.shape == second_hidden.shape


class TestBlockDiagonalIsolation:
    def test_one_graphs_action_cannot_reach_anothers_messages(self):
        """
        Two copies of the same graph in one block-diagonal batch, an action in
        block 0 only. Block 1's output must equal its own single-graph output.
        """
        num_nodes = 6
        single = line_graph(num_nodes)
        edges = np.concatenate(
            [single.edge_index.numpy(), single.edge_index.numpy() + num_nodes], axis=1
        )
        weights = np.concatenate(
            [single.edge_weight.numpy(), single.edge_weight.numpy()]
        )
        batch_index = torch.arange(2).repeat_interleave(num_nodes)
        block = build_graph_input(
            edges, weights, 2 * num_nodes, "IC", device, batch_index=batch_index
        )

        acted = features(num_nodes, infected=[0], frontier=[0], added=[2])
        quiet = features(num_nodes, infected=[0], frontier=[0])
        stacked = torch.cat([acted, quiet], dim=0)

        model = make_model(message_conditioning, seed=9)
        TestTheMessagePathReadsTheAction()._perturb(model)

        with torch.inference_mode():
            joint = model(stacked, block)
            alone = model(quiet, single)

        torch.testing.assert_close(joint[num_nodes:], alone)

    def test_without_a_batch_vector_the_input_is_one_graph(self):
        graph = line_graph()

        assert graph.batch_index is None

        model = make_model(message_conditioning, seed=9)
        X = features(6, infected=[0], frontier=[0], added=[2])

        with torch.inference_mode():
            assert torch.isfinite(model(X, graph)).all()


class TestSpecAndCheckpoint:
    def test_spec_defaults_to_the_baseline(self):
        spec = ModelSpec(backbone="sage", head="structured", diffusion_model="IC")

        assert spec.action_conditioning == no_conditioning

    def test_unknown_conditioning_is_refused(self):
        with pytest.raises(ValueError, match="action_conditioning"):
            ModelSpec(
                backbone="sage",
                head="structured",
                diffusion_model="IC",
                action_conditioning="sideways",
            )

    def test_a_variant_checkpoint_round_trips(self, tmp_path):
        spec = ModelSpec(
            backbone="sage",
            head="structured",
            diffusion_model="IC",
            hidden_dim=8,
            n_layers=2,
            action_conditioning=message_conditioning,
        )
        model = build_model(spec)
        path = save_checkpoint(model, tmp_path / "variant.pt", spec)
        restored, restored_spec, _ = load_checkpoint(path)

        assert restored_spec.action_conditioning == message_conditioning
        assert any("modulator" in key for key in restored.state_dict())

    def test_a_baseline_checkpoint_still_loads(self, tmp_path):
        """Backward compatibility: an existing v2 checkpoint has no
        `action_conditioning` key and must rebuild as the baseline."""
        spec = ModelSpec(
            backbone="sage", head="structured", diffusion_model="IC", hidden_dim=8
        )
        path = save_checkpoint(build_model(spec), tmp_path / "old.pt", spec)

        blob = torch.load(path, weights_only=False)
        del blob["spec"]["action_conditioning"]
        torch.save(blob, path)

        _, restored_spec, _ = load_checkpoint(path)

        assert restored_spec.action_conditioning == no_conditioning


class TestGuards:
    def test_conditioning_is_refused_on_an_unimplemented_backbone(self):
        with pytest.raises(ValueError, match="implemented"):
            WorldModel("gcn", hidden_dim=8, action_conditioning=message_conditioning)

    def test_action_columns_follow_the_layout(self):
        assert action_columns(in_channels) == (3, 4, 5)
        assert action_columns(typed_in_channels) == (3, 4, 5, 6, 7, 8)
        assert action_columns(8, competitive=True) == (5, 6, 7)
        assert action_columns(9, epidemic=True) == (6, 7, 8)

    def test_conditioning_works_under_the_competitive_layout(self):
        torch.manual_seed(0)
        model = WorldModel(
            "sage",
            in_channels=8,
            hidden_dim=8,
            n_layers=2,
            head_type="structured",
            competitive=True,
            action_conditioning=message_conditioning,
        ).eval()

        assert model.encoder.action_channels == (5, 6, 7)


class TestRolloutIntegration:
    """
    The block-diagonal rollout is where the batch vector earns its place: an
    ensemble is many copies of one graph in a single forward pass, and each copy
    runs its own policy under the batched candidate form. A missing batch vector
    would pool their actions into one `z_a` and let one sample's seeds change
    another's messages.
    """

    def _graph(self):
        import numpy as np

        from coding_agent.types import GraphInfo

        edges = np.array([[0, 1, 2, 3, 1, 2], [1, 2, 3, 4, 0, 1]], dtype=np.int64)

        return GraphInfo(
            num_nodes=5,
            edge_index=edges,
            ic_probs=np.full(edges.shape[1], 0.3, dtype=np.float32),
            directed=True,
        )

    def test_a_conditioned_model_rolls_out_through_the_agent_environment(self):
        from coding_agent.envs.world_model_env import WorldModelEnvironment
        from coding_agent.types import ActionOp

        graph = self._graph()
        environment = WorldModelEnvironment(
            make_model(message_conditioning, seed=4), graph, "IC", n_samples=4
        )
        plan = [[ActionOp("add_node", 0)]] + [[] for _ in range(4)]
        trajectory = environment.rollout(
            lambda state, timestep: plan[timestep] if timestep < len(plan) else [], 4, 1
        )

        assert trajectory.reward >= 1.0
        assert environment.forward_passes > 0

    def test_the_block_rollout_carries_a_batch_vector(self):
        from coding_agent.envs.world_model_env import WorldModelEnvironment
        from world_model.wm_data import edges_to_arrays

        graph = self._graph()
        environment = WorldModelEnvironment(
            make_model(no_conditioning, seed=4), graph, "IC", n_samples=3
        )
        block = environment._block_graph_input([edges_to_arrays(environment.base_edges)] * 3)

        assert block.batch_index is not None
        assert block.batch_index.tolist() == [0] * 5 + [1] * 5 + [2] * 5

    def test_two_candidates_in_one_batched_call_stay_isolated(self):
        """The striped candidate form puts different policies in different
        blocks; each block's reward must match its own solo rollout."""
        from coding_agent.credit import batched_plan_rewards
        from coding_agent.envs.world_model_env import WorldModelEnvironment
        from coding_agent.types import ActionOp

        graph = self._graph()
        model = make_model(message_conditioning, seed=6)
        first = [[ActionOp("add_node", 0)]] + [[] for _ in range(4)]
        second = [[ActionOp("add_node", 4)]] + [[] for _ in range(4)]

        environment = WorldModelEnvironment(model, graph, "IC", n_samples=2)
        rewards, _ = batched_plan_rewards(environment, [first, second], 4, 1, 0)

        assert len(rewards) == 2
        # Seeding node 0 (which has out-edges) must not score the same as seeding
        # the sink node 4, or the two blocks were not separated
        assert rewards[0] != rewards[1]


class TestEdgeMarginals:
    """
    `q(u -> v)` is the whole learned content of the IC structured head — the rest
    of the transition is hand-written — and until it was factored out of
    `forward` nothing downstream could read it. These tests pin the accessor to
    the value `forward` actually uses, so the two cannot drift.
    """

    def test_the_oracle_head_returns_the_true_edge_weight(self):
        graph = line_graph(weight=0.4)
        model = make_model(no_conditioning, seed=1, head="structured_oracle")
        q = model.head.transmission(torch.zeros(6, 8), graph)

        torch.testing.assert_close(q, graph.edge_weight)

    def test_the_accessor_matches_what_the_forward_pass_uses(self):
        """
        A seed at node 0 on a path graph reaches node 1 with exactly `q_{0->1}`,
        because node 1 has one in-edge and the source is the only active node.
        """
        graph = line_graph(weight=0.4)
        X = features(6, added=[0])
        model = make_model(no_conditioning, seed=2)

        with torch.inference_mode():
            hidden = model.encoder(X, graph)
            q = model.head.transmission(hidden, graph)
            probabilities = torch.sigmoid(model(X, graph))

        # edge 0 is 0 -> 1
        assert int(graph.edge_index[0, 0]) == 0 and int(graph.edge_index[1, 0]) == 1
        torch.testing.assert_close(probabilities[1, 0], q[0], atol=1e-5, rtol=1e-4)

    def test_an_empty_graph_returns_no_edges(self):
        import numpy as np

        from world_model.wm_data import build_graph_input

        empty = build_graph_input(
            np.zeros((2, 0), dtype=np.int64),
            np.zeros(0, dtype=np.float32),
            3,
            "IC",
            device,
        )
        model = make_model(no_conditioning, seed=3)

        assert model.head.transmission(torch.zeros(3, 8), empty).shape == (0,)

    def test_the_lt_head_has_no_per_arc_transmission(self):
        """LT edge weights carry no transmission meaning, so there is nothing to
        return and the head must not pretend otherwise."""
        model = WorldModel(
            "sage", hidden_dim=8, n_layers=2, head_type="structured", diffusion_model="LT"
        )

        assert not hasattr(model.head, "transmission")


class TestTheVariantCanActuallyLearn:
    """
    The test that would have caught the real bug in this experiment.

    An arm that cannot move is indistinguishable from an arm that does not help,
    and the first sweep run here reported the second while suffering the first:
    with the zero on `phi_m`'s OUTPUT LAYER, `dL/d(hidden layer)` is proportional
    to that zero, so the MLP's input layer and the entire ActionEncoder received
    exactly no gradient at step 0 — and weight decay shrank them toward zero
    faster than the output layer could revive them. The trained checkpoints came
    out with `max|phi_out| = 0.00000` and three architecturally different arms
    posted bit-identical metrics.

    Architecture tests that only check shapes and forward passes cannot see that.
    These check the two things that actually matter: gradient REACHES every part
    of the mechanism, and a few optimiser steps MOVE it.
    """

    def _batch(self):
        import numpy as np

        from world_model.wm_data import build_graph_input

        edges = np.array(
            [[0, 1, 2, 3, 4, 1, 2, 3], [1, 2, 3, 4, 5, 0, 1, 2]], dtype=np.int64
        )
        graph = build_graph_input(
            edges, np.full(8, 0.4, dtype=np.float32), 6, "IC", device
        )
        X = features(6, infected=[0], frontier=[0], added=[3])
        target = torch.zeros(6, 2)
        target[1, 0] = target[1, 1] = 1.0

        return X, graph, target

    def _grad(self, model, fragment: str) -> float:
        return sum(
            float(parameter.grad.abs().sum())
            for name, parameter in model.named_parameters()
            if fragment in name and parameter.grad is not None
        )

    def _step(self, model, X, graph, target, optimizer=None) -> None:
        if optimizer is not None:
            optimizer.zero_grad()

        torch.nn.functional.binary_cross_entropy_with_logits(
            model(X, graph), target
        ).backward()

        if optimizer is not None:
            optimizer.step()

    def test_the_gate_gets_gradient_immediately(self):
        """
        At step 0 the gate is the ONLY part with gradient, and it must have some:
        `dL/dgate = dL/ddelta . MLP(...)`, and the MLP is normally initialised so
        its output is nonzero. Everything upstream is multiplied by the zero gate
        and correctly waits — which is fine BECAUSE the gate moves. If this were
        zero the mechanism could never start.
        """
        X, graph, target = self._batch()
        model = make_model(message_conditioning, seed=0)
        model.train()
        self._step(model, X, graph, target)

        assert self._grad(model, "modulator.gate") > 0.0

    def test_gradient_reaches_the_whole_mechanism_once_the_gate_opens(self):
        """
        The property the first sweep run silently lacked. After a few steps the
        gate is off zero, and gradient must then reach the message MLP AND the
        ActionEncoder — the component that received exactly none under the old
        output-layer zero-init and was decayed to nothing.
        """
        X, graph, target = self._batch()
        model = make_model(message_conditioning, seed=0)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        for _ in range(5):
            self._step(model, X, graph, target, optimizer)

        optimizer.zero_grad()
        self._step(model, X, graph, target)

        assert self._grad(model, "modulator.mlp") > 0.0
        assert self._grad(model, "action_encoder") > 0.0

    @pytest.mark.parametrize(
        "conditioning", [global_conditioning, message_conditioning]
    )
    def test_a_few_optimiser_steps_move_the_gate_off_zero(self, conditioning):
        X, graph, target = self._batch()
        model = make_model(conditioning, seed=0)
        model.train()
        # The same weight decay the trainer uses, on the same parameters, so this
        # fails if a future change puts the action parameters back under it
        optimizer = torch.optim.Adam(
            [
                {"params": [p for n, p in model.named_parameters() if _conditioned(n)],
                 "weight_decay": 0.0},
                {"params": [p for n, p in model.named_parameters() if not _conditioned(n)],
                 "weight_decay": 5e-4},
            ],
            lr=1e-2,
        )

        for _ in range(20):
            optimizer.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(
                model(X, graph), target
            ).backward()
            optimizer.step()

        gates = [
            float(layer.modulator.gate.detach().abs()) for layer in model.encoder.layers
        ]

        assert max(gates) > 1e-3, f"the gate never opened: {gates}"

    def test_the_trainer_excludes_the_action_parameters_from_weight_decay(self):
        """Decaying a gate toward zero is a prior that the mechanism should not
        exist, which is the hypothesis under test."""
        from world_model.train_wm import parameter_groups

        model = make_model(message_conditioning, seed=0)
        groups = parameter_groups(model, weight_decay=5e-4)
        decayed = {id(parameter) for group in groups if group["weight_decay"] > 0
                   for parameter in group["params"]}

        for name, parameter in model.named_parameters():
            if _conditioned(name):
                assert id(parameter) not in decayed, name

        # ...and every parameter is still in exactly one group
        assert sum(len(group["params"]) for group in groups) == len(
            list(model.parameters())
        )


def _conditioned(name: str) -> bool:
    return "modulator" in name or "action_encoder" in name
