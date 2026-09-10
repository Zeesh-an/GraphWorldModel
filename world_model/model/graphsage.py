"""
GraphSAGE encoder

GraphSAGE with mean aggregator as a world-model backbone: forward(X: (N, C), graph) -> (N, hidden_dim).
Uses the concat variant:

    h_N(v) = mean({ h_u : u ∈ N(v) })          # neighbor mean, no self
    h_v'   = W · concat(h_v, h_N(v))            # (2H → H)

The neighbor mean excludes self (self-loops coming from adj_process are
filtered out of edge_index) since the self term is concatenated separately.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.model.action_cond import (
    ActionContext,
    ActionEncoder,
    MessageModulator,
    action_relevance,
    blind_conditioning,
    message_conditioning,
    modulated_conditioning,
    no_conditioning,
    resolve_conditioning,
)

# Prevent division-by-zero for isolated nodes (degree 0 -> mean 0)
degree_floor = 1e-9


class GraphSAGELayer(nn.Module):
    """
    One GraphSAGE (mean-aggregator) layer with pre-norm residual structure:

        h = LayerNorm(x)
        h_neigh = scatter_mean_{u∈N(v)}(h[u])
        h = Linear(concat(h, h_neigh))  # 2H → H
        h = GELU → Dropout
        return x + h

    With a `modulator` the MESSAGE, and only the message, changes:

        BEFORE  m_{u->v} = h_u * w_uv
        AFTER   m_{u->v} = (h_u + phi_m([h_u, h_v, w_uv, z_a, r_uv])) * w_uv

    `phi_m` is zero-initialised (world_model/model/action_cond.py), so a fresh
    modulated layer and a fresh baseline layer with the same seed compute the
    same function; everything after aggregation is untouched.
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        modulator: MessageModulator | None = None,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim

        self.norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(in_features=2 * hidden_dim, out_features=hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        # None on the baseline: the module is not created at all, so a baseline
        # checkpoint has no extra keys and loads into a baseline model unchanged
        self.modulator = modulator

    def forward(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        context: ActionContext | None = None,
    ) -> torch.Tensor:
        """
        features: (N, hidden_dim)
        edge_index: (2, E) long, [src, dst] with self-loops already removed
        edge_weight: (E,) float edge weights, or None for unweighted mean
        context: ActionContext, or None on the baseline path
        returns: (N, hidden_dim)
        """
        num_nodes, hidden_dim = features.shape[0], self.hidden_dim
        sources, destinations = edge_index[0], edge_index[1]

        # Pre-norm
        hidden = self.norm(features)  # shape: (N, H)

        # Weighted neighbor mean aggregation via scatter_add / weighted degree
        weights = (
            torch.ones(sources.shape[0], device=features.device, dtype=hidden.dtype)
            if edge_weight is None
            else edge_weight.to(hidden.dtype)
        )
        neighbor_mean = torch.zeros(
            num_nodes, hidden_dim, device=features.device, dtype=hidden.dtype
        )
        messages = hidden[sources]  # shape: (E, H)

        if self.modulator is not None and context is not None:
            # The ONE line the action-conditioning experiment is about
            messages = messages + self.modulator(
                messages, hidden[destinations], weights, context
            )

        neighbor_mean.scatter_add_(
            0,
            destinations.unsqueeze(dim=1).expand(-1, hidden_dim),
            messages * weights.unsqueeze(dim=1),
        )
        degrees = torch.zeros(num_nodes, device=features.device, dtype=hidden.dtype)
        degrees.scatter_add_(0, destinations, weights)
        safe_degrees = degrees.clamp(min=degree_floor).unsqueeze(dim=-1)  # shape: (N, 1)
        neighbor_mean = neighbor_mean / safe_degrees  # shape: (N, H)

        # Concat self and neighbor mean, project back to H
        combined = torch.cat([hidden, neighbor_mean], dim=-1)  # shape: (N, 2H)
        output = self.linear(combined)  # shape: (N, H)
        output = F.gelu(output)
        output = self.dropout(output)

        return features + output  # residual


class GraphSAGEEncoder(nn.Module):
    """
    GraphSAGE encoder that maps an explicit node feature matrix X to (N, hidden_dim) embeddings.
    Used by the action-conditioned world model as the graph backbone.

    `action_conditioning` selects which of the four arms of the controlled
    experiment this encoder is (world_model/model/action_cond.py):

      * `none` (default) — the historical model. The action reaches the encoder
        only as the CH_ADD / CH_REMOVE / CH_EDGE columns of X, and every existing
        checkpoint is this arm. No extra parameters exist, so old checkpoints
        load with `strict=True` unchanged.
      * `message_blind` / `global` / `message` — build an ActionEncoder and one
        MessageModulator per layer; see the table in action_cond.py.

    `action_channels` names which columns of X are the action. It is passed by
    `WorldModel` rather than inferred here, because the same width means
    different things across the single-cascade, competitive and compartmental
    layouts and guessing from `in_channels` would silently pick one.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
        action_conditioning: str = no_conditioning,
        action_channels: tuple[int, ...] = (),
        **_: object,
    ) -> None:
        super().__init__()

        self.action_conditioning = resolve_conditioning(action_conditioning)
        self.action_channels = tuple(int(column) for column in action_channels)

        if self.action_conditioning in modulated_conditioning and not self.action_channels:
            raise ValueError(
                f"action_conditioning={self.action_conditioning!r} needs "
                f"action_channels: without them there is no a_t to condition on"
            )

        # The baseline modules are built and initialised FIRST, and the variant's
        # extra modules are then drawn from a forked RNG. Both halves matter: a
        # variant and a baseline constructed under the same `torch.manual_seed`
        # must hold bit-identical shared tensors, or the A/B would compare two
        # different starting points as well as two different capacities.
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.layers = nn.ModuleList(
            [GraphSAGELayer(hidden_dim, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        num_action_channels = len(self.action_channels)
        self.action_encoder = None

        if self.action_conditioning in modulated_conditioning:
            with torch.random.fork_rng(devices=[]):
                self.action_encoder = ActionEncoder(num_action_channels, hidden_dim)

                for layer in self.layers:
                    layer.modulator = MessageModulator(
                        hidden_dim,
                        action_dim=hidden_dim,
                        relevance_dim=2 * num_action_channels,
                        use_relevance=(
                            self.action_conditioning == message_conditioning
                        ),
                        blind=(self.action_conditioning == blind_conditioning),
                    )

    def _context(
        self,
        X: torch.Tensor,
        hidden: torch.Tensor,
        graph,
        edge_index: torch.Tensor,
    ) -> ActionContext:
        columns = X[:, self.action_channels]  # shape: (N, k)
        batch_index = getattr(graph, "batch_index", None)

        if batch_index is None:
            # Every single-graph call site (rollout, scorer, planning eval) passes
            # no batch vector; one graph is then the correct reading, and a
            # block-diagonal batch always carries one from collate_transitions
            batch_index = torch.zeros(
                X.shape[0], dtype=torch.long, device=X.device
            )

        num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() else 1
        z_a = self.action_encoder(columns, hidden, batch_index, num_graphs)

        if edge_index.numel():
            z_edge = z_a[batch_index[edge_index[1]]]  # shape: (E, H)
        else:
            z_edge = z_a.new_zeros((0, z_a.shape[1]))

        return ActionContext(
            z_a=z_a,
            z_edge=z_edge,
            relevance=action_relevance(columns, edge_index),
        )

    def forward(self, X: torch.Tensor, graph) -> torch.Tensor:
        """
        X: (N, in_channels) node feature matrix
        graph: GraphInput, uses graph.edge_index (2, E) and graph.edge_weight (E,)
        returns: (N, hidden_dim) node embeddings
        """
        # Strip self-loops so the neighbor mean excludes self (self enters via concat)
        edge_index, edge_weight = graph.edge_index, graph.edge_weight
        mask = edge_index[0] != edge_index[1]
        edge_index, edge_weight = edge_index[:, mask], edge_weight[mask]

        hidden = F.gelu(self.input_proj(X))  # shape: (N, hidden_dim)

        # Computed ONCE from the input projection rather than per layer: z_a is a
        # property of the action and the state it acts on, not of a later layer's
        # own output, and recomputing it would make it a moving target
        context = (
            self._context(X, hidden, graph, edge_index)
            if self.action_encoder is not None
            else None
        )

        for layer in self.layers:
            hidden = layer(
                hidden, edge_index, edge_weight, context
            )  # shape: (N, hidden_dim)

        return self.norm(hidden)  # shape: (N, hidden_dim)
