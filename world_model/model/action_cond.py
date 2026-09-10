"""
Action-conditioned message passing (Variant B of the controlled experiment).

The baseline world model conditions on the action ONLY through the feature
matrix: `X[:, CH_ADD] / CH_REMOVE / CH_EDGE` are set by the feature builder and
the encoder then runs an ordinary GNN over them, while the structured head
applies `T_exo` algebraically. Written as the brief writes it:

    s~_t = T_exo(s_t, a_t)          # feature builder + head, closed form
    s^_t+1 = T_endo(s~_t)           # the learned GNN, action visible only as
                                    # three extra INPUT COLUMNS

This module supplies the missing argument, so a variant can run

    s^_t+1 = T_endo(s~_t, a_t)

with `a_t` present INSIDE the message function rather than only at the input
projection. Four modes, and the point of having four is that only their
differences are interpretable:

| mode            | phi_m sees                       | isolates                        |
| --------------- | -------------------------------- | ------------------------------- |
| `none`          | (no phi_m at all)                | the untouched baseline          |
| `message_blind` | h_u, h_v, w  (action zeroed)     | the CAPACITY phi_m adds         |
| `global`        | ... + z_a                        | global action conditioning      |
| `message`       | ... + z_a + r_uv                 | global + LOCAL action relevance |

`message_blind` exists because `message` adds parameters as well as
information; without it a win could be either. `global` vs `message` is the
locality question the brief asks (ablation D vs C).

Two properties make this a controlled experiment rather than a new model:

1. **Zero GATE.** `phi_m` is scaled by a learnable scalar initialised to zero,
   so at step 0 the modulated layer computes EXACTLY the baseline layer's
   output. Any divergence during training is learned, not an artefact of a
   different initialisation. The zero is on the gate rather than on the output
   layer for a reason worth reading in `MessageModulator`'s docstring: the
   latter placement silently starves the whole upstream path of gradient and
   lets weight decay delete the mechanism under test.
2. **Residual form.** The message stays `h_u` plus a correction:

       m_{u->v} = (h_u + phi_m([h_u, h_v, w_uv, z_a, r_uv])) * w_uv

   rather than replacing GraphSAGE's mean aggregator with an edge MLP. Swapping
   the aggregator would change the architecture on a second axis and the
   comparison would no longer be about the action.

`z_a` is per GRAPH, and the batches here are block-diagonal unions of many
graphs, so it is pooled under `GraphInput.batch_index`. With no batch vector the
whole input is treated as one graph, which is what every single-graph call site
(the rollout, the scorer, the planning eval) actually passes.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

#: The current model: action reaches the encoder only as input feature columns.
no_conditioning = "none"
#: Capacity control: the same message MLP, with its action inputs held at zero.
blind_conditioning = "message_blind"
#: Ablation D: a global action embedding, no per-edge relevance.
global_conditioning = "global"
#: Ablation C: global action embedding + per-edge action relevance.
message_conditioning = "message"

valid_action_conditioning = (
    no_conditioning,
    blind_conditioning,
    global_conditioning,
    message_conditioning,
)

#: Modes that build a modulator at all (i.e. everything but the baseline).
modulated_conditioning = (
    blind_conditioning,
    global_conditioning,
    message_conditioning,
)

#: Backbones with an action-conditioned message path. GraphSAGE is the backbone
#: the project settled on (world_model/checkpoints/RESULTS.md: the only one
#: faithful on both IC and LT rollouts), and the brief asks for the smallest
#: defensible change, not the same change five times.
conditionable_backbones = ("sage",)

#: Guards log1p/division when a graph carries no action at all.
count_floor = 1.0


@dataclass(frozen=True)
class ActionContext:
    """
    Everything the message function needs about `a_t`, computed once per forward.

    `z_edge` and `relevance` are already expanded to (E, ...) so a layer does no
    indexing of its own — the encoder holds one edge_index for all layers, so
    doing it per layer would be the same gather three times.
    """

    #: (G, H) one action embedding per graph in the block-diagonal batch.
    z_a: torch.Tensor
    #: (E, H) `z_a` broadcast to the edges of the graph each edge belongs to.
    z_edge: torch.Tensor
    #: (E, 2k) the action columns of the source and destination endpoint.
    relevance: torch.Tensor


def action_relevance(
    action_columns: torch.Tensor, edge_index: torch.Tensor
) -> torch.Tensor:
    """
    `r_uv(a)`: how this message relates to the intervention, per edge.

    Concatenating the endpoints' own action indicators answers all four
    questions the brief lists — is `u` the target, is `v` the target, was the
    edge itself modified (both endpoints carry CH_EDGE), is the message adjacent
    to the intervention — without inventing a fifth encoding for them.
    """
    if edge_index.numel() == 0:
        return action_columns.new_zeros((0, 2 * action_columns.shape[1]))

    sources, destinations = edge_index[0], edge_index[1]

    return torch.cat(
        [action_columns[sources], action_columns[destinations]], dim=-1
    )  # shape: (E, 2k)


def graph_sizes(batch_index: torch.Tensor, num_graphs: int) -> torch.Tensor:
    counts = torch.zeros(
        num_graphs, device=batch_index.device, dtype=torch.float32
    ).scatter_add_(0, batch_index, torch.ones_like(batch_index, dtype=torch.float32))

    return counts.clamp(min=count_floor)  # shape: (G,)


class ActionEncoder(nn.Module):
    """
    `a_t -> z_a`, one embedding per graph.

    Reads three things off the action columns, which between them cover the
    components the brief lists (action type, target indicator, parameters):

      * `log1p(count)` per op — HOW MANY of each op this step, on a scale that
        does not blow up when a plan seeds twenty nodes at once;
      * `count / N` per op — the same, relative to the graph;
      * the mean input embedding of the nodes each op targets — WHICH nodes,
        which is the part a count can never carry. `h0` is the input projection,
        so this is a function of the state and the structure of the target, not
        of a later layer's own output.

    An edge op sets CH_EDGE on BOTH endpoints, so its "target embedding" is the
    mean over the endpoints of every edge op in the bag; under `typed` the three
    extra columns split it by op and the same pooling applies per column.
    """

    def __init__(self, num_action_channels: int, hidden_dim: int) -> None:
        super().__init__()

        self.num_action_channels = num_action_channels
        self.hidden_dim = hidden_dim

        in_features = 2 * num_action_channels + num_action_channels * hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        action_columns: torch.Tensor,
        hidden: torch.Tensor,
        batch_index: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """
        action_columns: (N, k) the action indicator columns of X
        hidden: (N, H) input-projected node embeddings
        batch_index: (N,) which graph each node belongs to
        returns: (G, H)
        """
        num_channels = self.num_action_channels
        hidden_dim = self.hidden_dim
        index = batch_index.unsqueeze(dim=-1).expand(-1, num_channels)

        counts = torch.zeros(
            num_graphs, num_channels, device=hidden.device, dtype=hidden.dtype
        ).scatter_add_(0, index, action_columns)  # shape: (G, k)
        sizes = graph_sizes(batch_index, num_graphs).unsqueeze(dim=-1)  # (G, 1)

        # Per-channel target embedding: sum of h0 over the nodes the op targets,
        # normalised by how many there were (0 targets -> the zero vector, which
        # is the correct "this op is absent" encoding).
        weighted = action_columns.unsqueeze(dim=-1) * hidden.unsqueeze(
            dim=1
        )  # (N, k, H)
        targets = torch.zeros(
            num_graphs, num_channels, hidden_dim, device=hidden.device, dtype=hidden.dtype
        ).scatter_add_(
            0,
            batch_index.view(-1, 1, 1).expand(-1, num_channels, hidden_dim),
            weighted,
        )  # shape: (G, k, H)
        targets = targets / counts.clamp(min=count_floor).unsqueeze(dim=-1)

        features = torch.cat(
            [
                torch.log1p(counts),
                counts / sizes,
                targets.reshape(num_graphs, num_channels * hidden_dim),
            ],
            dim=-1,
        )  # shape: (G, 2k + kH)

        return self.mlp(features)  # shape: (G, H)


class MessageModulator(nn.Module):
    """
    `phi_m`: the action's contribution to one message, as a residual correction.

        delta_uv = gate * MLP([h_u, h_v, w_uv, z_a, r_uv]),   gate initialised to 0

    The gate, not the output layer, carries the zero. That distinction is the
    whole reason this class has a `gate` at all, and it was found the hard way:
    with the OUTPUT LAYER zero-initialised instead, `dL/d(hidden layer)` is
    proportional to that zero weight, so at step 0 the MLP's input layer and the
    whole ActionEncoder receive EXACTLY no gradient (measured: `|grad
    action_encoder| = 0.000e+00`). Weight decay does not wait — over a few
    thousand steps it drives those unused weights toward zero faster than the
    slowly-growing output layer can revive them, and the variant is regularised
    out of existence. A sweep run that way reports "action conditioning does not
    help" when what actually happened is that the optimiser deleted it.

    A ReZero-style scalar gate fixes it without giving up the property that
    matters: the MLP is normally initialised (so weight decay has something to
    act on and gradients flow through it as soon as the gate leaves zero),
    `dL/dgate = dL/ddelta . MLP(...)` is nonzero at step 0 (so the gate moves
    immediately), and `gate = 0` still makes an untrained variant bit-identical
    to the baseline.

    `world_model.train_wm` additionally excludes every parameter of this module
    and of `ActionEncoder` from weight decay, for the same reason biases and
    gains are conventionally excluded: decaying a gate toward zero is a prior
    that the mechanism should not exist, which is the hypothesis under test.
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        relevance_dim: int,
        use_relevance: bool,
        blind: bool,
    ) -> None:
        super().__init__()

        self.blind = blind
        self.use_relevance = use_relevance
        self.action_dim = action_dim
        self.relevance_dim = relevance_dim

        # [h_u, h_v, w_uv, z_a] (+ r_uv)
        in_features = 2 * hidden_dim + 1 + action_dim + (
            relevance_dim if use_relevance else 0
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        #: The zero that makes this a no-op at step 0, placed where it does not
        #: also zero the gradient to everything upstream of it
        self.gate = nn.Parameter(torch.zeros(1))

        for layer in (self.mlp[0], self.mlp[2]):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        source_hidden: torch.Tensor,
        destination_hidden: torch.Tensor,
        edge_weight: torch.Tensor,
        context: ActionContext,
    ) -> torch.Tensor:
        """(E, H) correction added to the source embedding before aggregation."""
        z_edge = context.z_edge
        relevance = context.relevance

        if self.blind:
            # Same tensor shapes, same parameter count, no action information:
            # the control that separates "the MLP helped" from "the ACTION helped"
            z_edge = torch.zeros_like(z_edge)
            relevance = torch.zeros_like(relevance)

        parts = [
            source_hidden,
            destination_hidden,
            edge_weight.unsqueeze(dim=-1).to(source_hidden.dtype),
            z_edge.to(source_hidden.dtype),
        ]

        if self.use_relevance:
            parts.append(relevance.to(source_hidden.dtype))

        return self.gate * self.mlp(torch.cat(parts, dim=-1))  # shape: (E, H)


def resolve_conditioning(mode: str) -> str:
    if mode not in valid_action_conditioning:
        raise ValueError(
            f"unknown action_conditioning {mode!r}; choose from "
            f"{list(valid_action_conditioning)}"
        )

    return mode


__all__ = [
    "ActionContext",
    "ActionEncoder",
    "MessageModulator",
    "action_relevance",
    "blind_conditioning",
    "conditionable_backbones",
    "global_conditioning",
    "message_conditioning",
    "modulated_conditioning",
    "no_conditioning",
    "resolve_conditioning",
    "valid_action_conditioning",
]
