"""
GAT Forward Model (GATv2)

GATv2 adapted to the forward diffusion task.
Fixes the static-attention limitation of the original GAT by applying
the attention MLP after combining the dst (self) and src (neighbor)
projections:

    z_edge = LeakyReLU(W_l h_v + W_r h_u)
    e_uv   = a^T z_edge
    alpha  = softmax_v(e_uv)   (over incoming edges of v)
    h_v'  += alpha * (W_r h_u)

Interface matches GraphTransformerForwardModel:
    forward(seed_vec: (N, 1), adj: sparse COO (N, N)) -> (N, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.model.model_utils import degree_encoding

# Floors the attention log-bias and softmax denominator
numerical_eps = 1e-9


class GATLayer(nn.Module):
    """
    One GATv2 layer with multi-head scatter-softmax attention and a pre-norm residual structure:

        h = LayerNorm(x)
        attn_out = MultiHeadGATv2(h, edge_index)
        return x + Dropout(out_proj(attn_out))
    """

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()

        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim must be divisible by n_heads, got {hidden_dim} / {n_heads}"
            )

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.d_head = hidden_dim // n_heads

        # Separate projections for self (dst) and neighbor (src)
        self.W_l = nn.Linear(
            in_features=hidden_dim, out_features=hidden_dim, bias=False
        )
        self.W_r = nn.Linear(
            in_features=hidden_dim, out_features=hidden_dim, bias=False
        )

        # Attention vector per head: (n_heads, d_head)
        self.a = nn.Parameter(torch.empty(n_heads, self.d_head))

        self.out_proj = nn.Linear(
            in_features=hidden_dim, out_features=hidden_dim, bias=False
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in [self.W_l, self.W_r, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)

        nn.init.xavier_uniform_(self.a)

    def forward(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        features: (N, hidden_dim)
        edge_index: (2, E) long — [src, dst]
        edge_weight: (E,) float edge weights added as log-bias to attention scores, or None for unweighted
        returns: (N, hidden_dim)
        """
        num_nodes = features.shape[0]
        sources, destinations = edge_index[0], edge_index[1]

        # Pre-norm
        hidden = self.norm(features)

        # Per-head projections: (N, n_heads, d_head)
        self_projection = self.W_l(hidden).view(num_nodes, self.n_heads, self.d_head)
        neighbor_projection = self.W_r(hidden).view(
            num_nodes, self.n_heads, self.d_head
        )

        # GATv2 attention logits per edge, per head
        # For edge (src=u, dst=v):  e_uv = a^T LeakyReLU(z_l[v] + z_r[u])
        edge_projection = (
            self_projection[destinations] + neighbor_projection[sources]
        )  # (E, n_heads, d_head)
        edge_projection = self.leaky_relu(edge_projection)
        scores = (edge_projection * self.a.unsqueeze(0)).sum(dim=-1)  # (E, n_heads)

        if edge_weight is not None:
            scores = scores + torch.log(
                edge_weight.clamp(min=numerical_eps)
            ).unsqueeze(1)

        # Scatter softmax over incoming edges per destination, per head
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
        scores_exp = torch.exp(scores - scores_max[destinations])  # (E, n_heads)

        scores_sum = torch.zeros(num_nodes, self.n_heads, device=features.device)
        scores_sum.scatter_add_(
            0, destinations.unsqueeze(1).expand(-1, self.n_heads), scores_exp
        )
        attention = scores_exp / (
            scores_sum[destinations] + numerical_eps
        )  # (E, n_heads)
        attention = self.dropout(attention)

        # Aggregate: value = neighbor projection
        aggregated = torch.zeros(
            num_nodes, self.n_heads, self.d_head, device=features.device
        )
        aggregated.scatter_add_(
            0,
            destinations.unsqueeze(1).unsqueeze(2).expand(-1, self.n_heads, self.d_head),
            attention.unsqueeze(-1) * neighbor_projection[sources],
        )  # (N, n_heads, d_head)

        # Concatenate heads and project
        aggregated = aggregated.reshape(num_nodes, self.hidden_dim)
        aggregated = self.out_proj(aggregated)

        return features + self.dropout(aggregated)  # residual


class GATForwardModel(nn.Module):
    """
    GATv2-based forward diffusion model.

    Architecture:
        1. Input projection: cat(seed_vec, degree_PE) -> hidden_dim
        2. n_layers x GATLayer (pre-norm residual multi-head attention)
        3. Output head: LayerNorm -> Linear(hidden_dim, 1) -> Sigmoid
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        use_pe: bool = True,
    ) -> None:
        super().__init__()

        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim must be divisible by n_heads, got {hidden_dim} / {n_heads}"
            )

        self.hidden_dim = hidden_dim
        self.use_pe = use_pe

        in_features = 1 + hidden_dim if use_pe else 1
        self.input_proj = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_dim),
            nn.GELU(),
        )

        self.layers = nn.ModuleList(
            [
                GATLayer(hidden_dim=hidden_dim, n_heads=n_heads, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(in_features=hidden_dim, out_features=1),
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
        seed_vec: (N, 1) soft action probabilities in [0, 1]
        adjacency: sparse COO (N, N) — normalized D^-1/2 (A+I) D^-1/2
        returns: (N, 1) predicted outcome probabilities in [0, 1]
        """
        device = seed_vec.device

        # Build edge_index from the adjacency
        if adjacency.is_sparse:
            edge_index = adjacency.coalesce().indices()  # (2, E)
        else:
            edge_index = adjacency.nonzero(as_tuple=False).t()  # (2, E)

        if self.use_pe:
            positional_encoding = degree_encoding(
                adjacency, self.hidden_dim, device
            )  # (N, hidden_dim)
            features = torch.cat(
                [seed_vec, positional_encoding], dim=-1
            )  # (N, 1 + hidden_dim)
        else:
            features = seed_vec  # (N, 1)

        features = self.input_proj(features)  # (N, hidden_dim)

        for layer in self.layers:
            features = layer(features, edge_index)  # (N, hidden_dim)

        return torch.sigmoid(self.output_proj(features))  # (N, 1)


class GATEncoder(nn.Module):
    """
    GATv2 encoder that maps an explicit node feature matrix X to (N, hidden_dim) embeddings.
    Used by the action-conditioned world model as the graph backbone.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        **_,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.layers = nn.ModuleList(
            [GATLayer(hidden_dim, n_heads, dropout) for _ in range(n_layers)]
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
