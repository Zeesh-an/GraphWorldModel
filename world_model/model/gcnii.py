"""
GCNII Forward Model

GCNII adapted to the forward diffusion task.

Layer formula:
    β_l    = log(λ/l + 1)
    P      = D^{-1/2} (A + I) D^{-1/2}           # already produced by adj_process
    support = (1 - α) · P · H^l + α · H^0        # initial residual
    H^{l+1} = ReLU( (1 - β_l) · support + β_l · W^l · support )   # identity mapping

H^0 is the output of the input projection (shared across all layers).
α (alpha) controls the weight of the initial residual connection.
λ (lamda) controls the decay of β_l with depth — larger λ → larger β_l.

Interface matches GraphTransformerForwardModel:
    forward(seed_vec: (N, 1), adj: sparse COO (N, N)) -> (N, 1)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_utils import degree_encoding


class GCNIILayer(nn.Module):
    """
    One GCNII layer with initial residual + identity mapping.
    No per-layer LayerNorm and no outer residual — the (1 - α)·P·H + α·H^0 structure is the residual mechanism in GCNII.
    """

    def __init__(
        self,
        hidden_dim: int,
        alpha: float,
        lamda: float,
        layer_idx: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.alpha = alpha
        # Layer indices are 1-based in the paper: β_l = log(λ/l + 1)
        self.beta = math.log(lamda / (layer_idx + 1) + 1.0)

        self.W = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W.weight)

    def forward(
        self, h: torch.Tensor, h0: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        """
        h: (N, hidden_dim) — current hidden state
        h0: (N, hidden_dim) — initial projected features (shared across layers)
        adj: sparse COO (N, N) — normalized D^-1/2 (A+I) D^-1/2
        returns: (N, hidden_dim)
        """
        # Dropout on input hidden state (matches reference GCNII implementation)
        h = self.dropout(h)

        # Propagation + initial residual
        propagated = torch.sparse.mm(adj, h)  # shape: (N, H)
        support = (1.0 - self.alpha) * propagated + self.alpha * h0  # shape: (N, H)

        # Identity mapping: ((1 - β)·I + β·W) · support
        out = (1.0 - self.beta) * support + self.beta * self.W(support)  # shape: (N, H)

        return F.relu(out)


class GCNIIForwardModel(nn.Module):
    """
    GCNII-based forward diffusion model.

    Architecture:
        1. Input projection: cat(seed_vec, degree_PE) -> hidden_dim (= H^0)
        2. n_layers x GCNIILayer (initial residual + identity mapping)
        3. Output head: LayerNorm -> Linear(hidden_dim, 1) -> Sigmoid
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 8,
        alpha: float = 0.1,
        lamda: float = 0.5,
        dropout: float = 0.1,
        use_pe: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_pe = use_pe

        in_features = 1 + hidden_dim if use_pe else 1
        self.input_proj = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_dim),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList(
            [
                GCNIILayer(
                    hidden_dim=hidden_dim,
                    alpha=alpha,
                    lamda=lamda,
                    layer_idx=layer_idx + 1,  # 1-based, matches paper
                    dropout=dropout,
                )
                for layer_idx in range(n_layers)
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
        adj: sparse COO (N, N) — normalized D^-1/2 (A+I) D^-1/2
        returns: (N, 1) predicted outcome probabilities in [0, 1]
        """
        device = seed_vec.device

        if self.use_pe:
            pe = degree_encoding(adj, self.hidden_dim, device)  # shape: (N, H)
            x = torch.cat([seed_vec, pe], dim=-1)  # shape: (N, 1 + H)
        else:
            x = seed_vec  # shape: (N, 1)

        h0 = self.input_proj(x)  # shape: (N, H)
        h = h0

        for layer in self.layers:
            h = layer(h, h0, adj)  # shape: (N, H)

        out = F.sigmoid(self.output_proj(h))  # shape: (N, 1)
        return out


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
        **_
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.layers = nn.ModuleList(
            [
                GCNIILayer(hidden_dim, alpha, lamda, layer_idx=i + 1, dropout=dropout)
                for i in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, X: torch.Tensor, graph) -> torch.Tensor:
        """
        X: (N, in_channels) node feature matrix
        graph: GraphInput — uses graph.adj_norm (sparse COO, normalized)
        returns: (N, hidden_dim) node embeddings
        """
        h0 = F.relu(self.input_proj(X))  # shape: (N, hidden_dim)
        h = h0
        for layer in self.layers:
            h = layer(h, h0, graph.adj_norm)  # shape: (N, hidden_dim)

        return self.norm(h)  # shape: (N, hidden_dim)
