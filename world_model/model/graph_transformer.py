"""
Graph Transformer — Forward Diffusion Model

Replaces the SpGAT in DeepIM with a proper Graph Transformer.

Key differences from GAT:
    - Scaled dot-product attention  (Q·K / sqrt(d_k)) vs additive LeakyReLU attention
    - Pre-LayerNorm residual blocks (more stable training)
    - Position-wise FFN after attention
    - Degree-based positional encoding injected into node features

Architecture: GraphTransformerForwardModel
    Inputs: x (N, 1) binary seed indicator per node, adj sparse COO tensor (N, N)
    Layers: node input projection → L x GTLayer → output projection → sigmoid
    Output: (N, 1) predicted activation probability per node
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.model.model_utils import degree_encoding

# Floors the attention log-bias and softmax denominator
numerical_eps = 1e-9


# Graph Transformer Layer
class GraphTransformerLayer(nn.Module):
    """
    One layer of the Graph Transformer.

    Sparse edge-masked multi-head scaled dot-product attention:
        For each edge (u → v), compute attention scores:
            score(u,v) = (Wq·h_v) · (Wk·h_u)^T / sqrt(d_k)

        Softmax over all incoming neighbors of v.
        h_v' = Σ_u  softmax_score(u,v) * (Wv · h_u)

    Followed by:
        - Residual + LayerNorm
        - 2-layer FFN (d_model → ffn_dim → d_model)
        - Residual + LayerNorm

    Pre-norm variant (norm before sublayer — more stable).
    """

    def __init__(
        self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model must be divisible by n_heads, got {d_model} / {n_heads}"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Multi-head Projections
        self.Wq = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.Wk = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.Wv = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.Wo = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        # Feed-forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, out_features=ffn_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(ffn_dim, out_features=d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in [self.Wq, self.Wk, self.Wv, self.Wo]:
            nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        features: (N, d_model)
        edge_index: (2, E) long — [src, dst]
        edge_weight: (E,) float edge weights added as log-bias to attention scores, or None for unweighted
        returns: (N, d_model)
        """
        num_nodes = features.shape[0]

        # Extract source and destination node indices
        # sources[i] → destinations[i] is the ith edge
        sources, destinations = edge_index[0], edge_index[1]  # each (E,)

        # Attention sublayer (pre-norm)
        hidden = self.norm1(features)

        # Project Q/K/V → (N, H, d_k), split by heads
        queries = self.Wq(hidden).view(num_nodes, self.n_heads, self.d_k)
        keys = self.Wk(hidden).view(num_nodes, self.n_heads, self.d_k)
        values = self.Wv(hidden).view(num_nodes, self.n_heads, self.d_k)

        # For each edge (u → v), compute attention scores per edge, per head:
        # score = Q[dst] · K[src] / sqrt(d_k)
        edge_queries = queries[destinations]  # (E, H, d_k)
        edge_keys = keys[sources]  # (E, H, d_k)
        edge_values = values[sources]  # (E, H, d_k)

        scores = (edge_queries * edge_keys).sum(dim=-1) / math.sqrt(self.d_k)  # (E, H)

        if edge_weight is not None:
            scores = scores + torch.log(
                edge_weight.clamp(min=numerical_eps)
            ).unsqueeze(1)

        # Compute softmax over incoming edges per node per head
        # Use scatter softmax: subtract max per (dst, head) for stability
        scores_max = torch.full(
            (num_nodes, self.n_heads), float("-inf"), device=features.device
        )
        scores_max.scatter_reduce_(
            0,
            destinations.unsqueeze(1).expand(-1, self.n_heads),
            scores,
            reduce="amax",
            include_self=True,
        )
        scores_exp = torch.exp(scores - scores_max[destinations])  # (E, H)

        scores_sum = torch.zeros(num_nodes, self.n_heads, device=features.device)
        scores_sum.scatter_add_(
            0, destinations.unsqueeze(1).expand(-1, self.n_heads), scores_exp
        )
        attention = scores_exp / (scores_sum[destinations] + numerical_eps)  # (E, H)
        attention = self.dropout(attention)

        # Aggregate values
        aggregated = torch.zeros(
            num_nodes, self.n_heads, self.d_k, device=features.device
        )
        aggregated.scatter_add_(
            0,
            destinations.unsqueeze(1).unsqueeze(2).expand(-1, self.n_heads, self.d_k),
            attention.unsqueeze(-1) * edge_values,
        )  # (N, H, d_k)

        # Reshape multiple attention heads and output projection
        aggregated = aggregated.reshape(num_nodes, self.d_model)  # (N, d_model)
        aggregated = self.Wo(aggregated)

        # Residual/skip connection
        features = features + self.dropout(aggregated)

        # FFN sublayer (pre-norm)
        features = features + self.dropout(self.ffn(self.norm2(features)))

        return features  # (N, d_model)


# Full Forward Model (Graph Transformer)
class GraphTransformerForwardModel(nn.Module):
    """
    Graph Transformer-based differentiable diffusion simulator.

    Mirrors the role of SpGAT in DeepIM:
        Input: seed_vec - soft seed probabilities from VAE decoder (N, 1),
                adj - sparse COO graph adjacency (symmetric, normalised)
        Output: predicted activation probability per node (N, 1)

    Architecture:
        1. Input projection: 1 → d_model, concatenated with degree PE
        2. L Graph Transformer layers
        3. Output projection: d_model → 1 → sigmoid
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        ffn_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_model = d_model

        # Project scalar node feature (seed prob) + PE → d_model
        self.input_proj = nn.Sequential(
            nn.Linear(in_features=1 + d_model, out_features=d_model),
            nn.GELU(),
        )

        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    d_model=d_model, n_heads=n_heads, ffn_dim=ffn_dim, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(in_features=d_model, out_features=1),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, seed_vec: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        seed_vec: (N, 1) soft seed probabilities
        adjacency: sparse COO (N, N) or dense (N, N)
        returns: (N, 1) influence probabilities in [0, 1]
        """
        device = seed_vec.device

        # Build edge_index from the adjacency
        if adjacency.is_sparse:
            edge_index = adjacency.coalesce().indices()  # (2, E)
        else:
            edge_index = adjacency.nonzero(as_tuple=False).t()  # (2, E)

        # Compute degree positional encodings
        positional_encoding = degree_encoding(
            adjacency, self.d_model, device
        )  # (N, d_model)

        # Input projection: cat(seed_vec, PE) → d_model
        features = self.input_proj(
            torch.cat([seed_vec, positional_encoding], dim=-1)
        )  # (N, d_model)

        # Graph Transformer layers
        for layer in self.layers:
            features = layer(features, edge_index)

        # Output projection
        return torch.sigmoid(self.output_proj(features))  # (N, 1)


class GraphTransformerEncoder(nn.Module):
    """
    Graph Transformer encoder that maps an explicit node feature matrix X to (N, hidden_dim) embeddings.
    Used by the action-conditioned world model as the graph backbone.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        **_,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(hidden_dim, n_heads, ffn_dim, dropout)
                for _ in range(n_layers)
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
        graph: GraphInput — uses graph.edge_index (2, E) and graph.edge_weight (E,)
        returns: (N, hidden_dim) node embeddings
        """
        # Add self-loops so each node attends to itself (weight 1 -> neutral log-bias)
        num_nodes = X.shape[0]
        self_loops = torch.arange(num_nodes, device=X.device)
        edge_index = torch.cat(
            [graph.edge_index, torch.stack([self_loops, self_loops])], dim=1
        )
        edge_weight = torch.cat(
            [
                graph.edge_weight,
                torch.ones(num_nodes, device=X.device, dtype=graph.edge_weight.dtype),
            ]
        )

        hidden = F.gelu(self.input_proj(X))  # shape: (N, hidden_dim)
        for layer in self.layers:
            hidden = layer(hidden, edge_index, edge_weight)  # shape: (N, hidden_dim)

        return self.norm(hidden)  # shape: (N, hidden_dim)
