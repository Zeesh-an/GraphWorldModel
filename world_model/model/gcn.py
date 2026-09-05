"""
GCN encoder

Classical GCN as a world-model backbone: forward(X: (N, C), graph) -> (N, hidden_dim).

adj is expected to already be symmetrically normalized with self-loops
(i.e. the output of utils.adj_process).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



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

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # features: (N, hidden_dim), adjacency: sparse COO (N, N) normalized
        hidden = self.norm(features)
        hidden = torch.sparse.mm(adjacency, hidden)  # (N, hidden_dim)
        hidden = self.linear(hidden)
        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)

        return features + hidden  # residual


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
        **_,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.layers = nn.ModuleList(
            [GCNLayer(hidden_dim, dropout) for _ in range(n_layers)]
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
        graph: GraphInput, uses graph.adj_norm (sparse COO, normalized)
        returns: (N, hidden_dim) node embeddings
        """
        hidden = F.gelu(self.input_proj(X))  # shape: (N, hidden_dim)
        for layer in self.layers:
            hidden = layer(hidden, graph.adj_norm)  # shape: (N, hidden_dim)

        return self.norm(hidden)  # shape: (N, hidden_dim)
