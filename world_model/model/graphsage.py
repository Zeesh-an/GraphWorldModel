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
    """

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim

        self.norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(in_features=2 * hidden_dim, out_features=hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        features: (N, hidden_dim)
        edge_index: (2, E) long, [src, dst] with self-loops already removed
        edge_weight: (E,) float edge weights, or None for unweighted mean
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
        neighbor_mean.scatter_add_(
            0,
            destinations.unsqueeze(dim=1).expand(-1, hidden_dim),
            hidden[sources] * weights.unsqueeze(dim=1),
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
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()

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
        for layer in self.layers:
            hidden = layer(hidden, edge_index, edge_weight)  # shape: (N, hidden_dim)

        return self.norm(hidden)  # shape: (N, hidden_dim)
