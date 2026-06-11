"""
GraphSAGE Forward Model

GraphSAGE with mean aggregator adapted to the forward diffusion task.
Uses the concat variant:

    h_N(v) = mean({ h_u : u ∈ N(v) })          # neighbor mean, no self
    h_v'   = W · concat(h_v, h_N(v))            # (2H → H)

The neighbor mean excludes self (self-loops coming from adj_process are
filtered out of edge_index) since the self term is concatenated separately.

Interface matches GraphTransformerForwardModel:
    forward(seed_vec: (N, 1), adj: sparse COO (N, N)) -> (N, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import degree_encoding


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
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (N, hidden_dim)
        edge_index: (2, E) long — [src, dst] with self-loops already removed
        edge_weight: (E,) float edge weights, or None for unweighted mean
        returns: (N, hidden_dim)
        """
        N, H = x.shape[0], self.hidden_dim
        src, dst = edge_index[0], edge_index[1]

        # Pre-norm
        h = self.norm(x)  # shape: (N, H)

        # Weighted neighbor mean aggregation via scatter_add / weighted degree
        w = (
            torch.ones(src.shape[0], device=x.device, dtype=h.dtype)
            if edge_weight is None
            else edge_weight.to(h.dtype)
        )
        h_neigh = torch.zeros(N, H, device=x.device, dtype=h.dtype)
        h_neigh.scatter_add_(0, dst.unsqueeze(1).expand(-1, H), h[src] * w.unsqueeze(1))
        deg = torch.zeros(N, device=x.device, dtype=h.dtype)
        deg.scatter_add_(0, dst, w)
        # Prevent division-by-zero for isolated nodes (deg=0 → mean=0)
        deg_safe = deg.clamp(min=1e-9).unsqueeze(dim=-1)  # shape: (N, 1)
        h_neigh = h_neigh / deg_safe  # shape: (N, H)

        # Concat self and neighbor mean, project back to H
        h_cat = torch.cat([h, h_neigh], dim=-1)  # shape: (N, 2H)
        h_out = self.linear(h_cat)  # shape: (N, H)
        h_out = F.gelu(h_out)
        h_out = self.dropout(h_out)

        return x + h_out  # residual


class GraphSAGEForwardModel(nn.Module):
    """
    GraphSAGE-based forward diffusion model.

    Architecture:
        1. Input projection: cat(seed_vec, degree_PE) -> hidden_dim
        2. n_layers x GraphSAGELayer (pre-norm residual mean-aggregator blocks)
        3. Output head: LayerNorm -> Linear(hidden_dim, 1) -> Sigmoid
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
        use_pe: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_pe = use_pe

        in_features = 1 + hidden_dim if use_pe else 1
        self.input_proj = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_dim),
            nn.GELU(),
        )

        self.layers = nn.ModuleList(
            [
                GraphSAGELayer(hidden_dim=hidden_dim, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(in_features=hidden_dim, out_features=1),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, seed_vec: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        seed_vec: (N, 1) soft action probabilities in [0, 1]
        adj: sparse COO (N, N) — normalized D^-1/2 (A+I) D^-1/2 (self-loops added)
        returns: (N, 1) predicted outcome probabilities in [0, 1]
        """
        device = seed_vec.device

        # Build edge_index from adj, stripping self-loops so GraphSAGE mean
        # aggregates over true neighbors only (self term enters via concat).
        if adj.is_sparse:
            edge_index = adj.coalesce().indices()  # shape: (2, E)
        else:
            edge_index = adj.nonzero(as_tuple=False).t()  # shape: (2, E)

        src, dst = edge_index[0], edge_index[1]
        mask = src != dst
        edge_index = torch.stack([src[mask], dst[mask]], dim=0)  # shape: (2, E')

        if self.use_pe:
            pe = degree_encoding(adj, self.hidden_dim, device)  # shape: (N, H)
            x = torch.cat([seed_vec, pe], dim=-1)  # shape: (N, 1 + H)
        else:
            x = seed_vec  # shape: (N, 1)

        x = self.input_proj(x)  # shape: (N, H)

        for layer in self.layers:
            x = layer(x, edge_index)  # shape: (N, H)

        out = F.sigmoid(self.output_proj(x))  # shape: (N, 1)
        return out


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
        **_
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.layers = nn.ModuleList(
            [GraphSAGELayer(hidden_dim, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, X: torch.Tensor, graph
    ) -> (
        torch.Tensor
    ):  # graph: GraphInput (duck-typed; avoids model/ -> wm_data coupling)
        """
        X: (N, in_channels) node feature matrix
        graph: GraphInput — uses graph.edge_index (2, E) and graph.edge_weight (E,)
        returns: (N, hidden_dim) node embeddings
        """
        # Strip self-loops so the neighbor mean excludes self (self enters via concat)
        ei, w = graph.edge_index, graph.edge_weight
        mask = ei[0] != ei[1]
        ei, w = ei[:, mask], w[mask]
        h = F.gelu(self.input_proj(X))  # shape: (N, hidden_dim)
        for layer in self.layers:
            h = layer(h, ei, w)  # shape: (N, hidden_dim)

        return self.norm(h)  # shape: (N, hidden_dim)
