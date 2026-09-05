"""
GCNII encoder

GCNII as a world-model backbone: forward(X: (N, C), graph) -> (N, hidden_dim).

Layer formula:
    β_l    = log(λ/l + 1)
    P      = D^{-1/2} (A + I) D^{-1/2}           # already produced by adj_process
    support = (1 - α) · P · H^l + α · H^0        # initial residual
    H^{l+1} = ReLU( (1 - β_l) · support + β_l · W^l · support )   # identity mapping

H^0 is the output of the input projection (shared across all layers).
α (alpha) controls the weight of the initial residual connection.
λ (lamda) controls the decay of β_l with depth: larger λ → larger β_l.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNIILayer(nn.Module):
    """
    One GCNII layer with initial residual + identity mapping.
    No per-layer LayerNorm and no outer residual: the (1 - α)·P·H + α·H^0 structure is the residual mechanism in GCNII.
    """

    def __init__(
        self,
        hidden_dim: int,
        alpha: float,
        lamda: float,
        layer_index: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.alpha = alpha
        # Layer indices are 1-based in the paper: β_l = log(λ/l + 1) (layer_index is l)
        self.beta = math.log(lamda / layer_index + 1.0)

        self.W = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W.weight)

    def forward(
        self,
        hidden: torch.Tensor,
        initial_hidden: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """
        hidden: (N, hidden_dim), current hidden state
        initial_hidden: (N, hidden_dim), initial projected features (shared across layers)
        adjacency: sparse COO (N, N), normalized D^-1/2 (A+I) D^-1/2
        returns: (N, hidden_dim)
        """
        # Dropout on input hidden state (matches reference GCNII implementation)
        hidden = self.dropout(hidden)

        # Propagation + initial residual
        propagated = torch.sparse.mm(adjacency, hidden)  # shape: (N, H)
        support = (
            1.0 - self.alpha
        ) * propagated + self.alpha * initial_hidden  # shape: (N, H)

        # Identity mapping: ((1 - β)·I + β·W) · support
        output = (1.0 - self.beta) * support + self.beta * self.W(
            support
        )  # shape: (N, H)

        return F.relu(output)


class GCNIIEncoder(nn.Module):
    """
    GCNII encoder that maps an explicit node feature matrix X to (N, hidden_dim) embeddings.
    Used by the action-conditioned world model as the graph backbone.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        n_layers: int = 8,
        alpha: float = 0.1,
        lamda: float = 0.5,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.layers = nn.ModuleList(
            [
                GCNIILayer(
                    hidden_dim, alpha, lamda, layer_index=index + 1, dropout=dropout
                )
                for index in range(n_layers)
            ]
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
        initial_hidden = F.relu(self.input_proj(X))  # shape: (N, hidden_dim)
        hidden = initial_hidden
        for layer in self.layers:
            hidden = layer(hidden, initial_hidden, graph.adj_norm)  # shape: (N, hidden_dim)

        return self.norm(hidden)  # shape: (N, hidden_dim)
