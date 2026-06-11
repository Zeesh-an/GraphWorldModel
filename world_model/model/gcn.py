"""
GCN Forward Model

Classical GCN adapted to the forward diffusion task.
Interface matches GraphTransformerForwardModel:
    forward(seed_vec: (N, 1), adj: sparse COO (N, N)) -> (N, 1)

adj is expected to already be symmetrically normalized with self-loops
(i.e. the output of utils.adj_process).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import degree_encoding


class GCNLayer(nn.Module):
    """
    Pre-norm residual GCN block:
        h = LayerNorm(x)
        h = sparse.mm(adj, h) # normalized propagation
        h = Linear(h) -> GELU -> Dropout
        return x + h
    """

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (N, hidden_dim), adj: sparse COO (N, N) normalized
        h = self.norm(x)
        h = torch.sparse.mm(adj, h)  # (N, hidden_dim)
        h = self.linear(h)
        h = F.gelu(h)
        h = self.dropout(h)
        return x + h  # residual


class GCNForwardModel(nn.Module):
    """
    GCN-based forward diffusion model.

    Architecture:
        1. Input projection: cat(seed_vec, degree_PE) -> hidden_dim (or seed_vec -> hidden_dim if use_pe=False)
        2. n_layers x GCNLayer (pre-norm residual blocks)
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
            [GCNLayer(hidden_dim=hidden_dim, dropout=dropout) for _ in range(n_layers)]
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
        adj: sparse COO (N, N) — normalized D^-1/2 (A+I) D^-1/2
        returns: (N, 1) predicted outcome probabilities in [0, 1]
        """
        device = seed_vec.device

        if self.use_pe:
            pe = degree_encoding(adj, self.hidden_dim, device)  # (N, hidden_dim)
            x = torch.cat([seed_vec, pe], dim=-1)  # (N, 1 + hidden_dim)
        else:
            x = seed_vec  # (N, 1)

        x = self.input_proj(x)  # (N, hidden_dim)

        for layer in self.layers:
            x = layer(x, adj)  # (N, hidden_dim)

        out = F.sigmoid(self.output_proj(x))  # (N, 1)
        return out


class GCNEncoder(nn.Module):
    """
    GCN encoder that maps an explicit node feature matrix X to (N, hidden_dim) embeddings.
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
            [GCNLayer(hidden_dim, dropout) for _ in range(n_layers)]
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
        graph: GraphInput — uses graph.adj_norm (sparse COO, normalized)
        returns: (N, hidden_dim) node embeddings
        """
        h = F.gelu(self.input_proj(X))  # shape: (N, hidden_dim)
        for layer in self.layers:
            h = layer(h, graph.adj_norm)  # shape: (N, hidden_dim)

        return self.norm(h)  # shape: (N, hidden_dim)
