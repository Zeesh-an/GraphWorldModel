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

from .model_utils import degree_encoding


class GATLayer(nn.Module):
    """
    One GATv2 layer with multi-head scatter-softmax attention and a pre-norm residual structure:

        h = LayerNorm(x)
        attn_out = MultiHeadGATv2(h, edge_index)
        return x + Dropout(out_proj(attn_out))
    """

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()

        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

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
        for m in [self.W_l, self.W_r, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)

        nn.init.xavier_uniform_(self.a)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (N, hidden_dim)
        edge_index: (2, E) long — [src, dst]
        edge_weight: (E,) float edge weights added as log-bias to attention scores, or None for unweighted
        returns: (N, hidden_dim)
        """
        N = x.shape[0]
        src, dst = edge_index[0], edge_index[1]

        # Pre-norm
        h = self.norm(x)

        # Per-head projections: (N, n_heads, d_head)
        z_l = self.W_l(h).view(N, self.n_heads, self.d_head)  # self/dst
        z_r = self.W_r(h).view(N, self.n_heads, self.d_head)  # neighbor/src

        # GATv2 attention logits per edge, per head
        # For edge (src=u, dst=v):  e_uv = a^T LeakyReLU(z_l[v] + z_r[u])
        z_edge = z_l[dst] + z_r[src]  # (E, n_heads, d_head)
        z_edge = self.leaky_relu(z_edge)
        scores = (z_edge * self.a.unsqueeze(0)).sum(dim=-1)  # (E, n_heads)

        if edge_weight is not None:
            scores = scores + torch.log(edge_weight.clamp(min=1e-9)).unsqueeze(1)

        # Scatter softmax over incoming edges per dst, per head
        scores_max = torch.full((N, self.n_heads), float("-inf"), device=x.device)
        scores_max.scatter_reduce_(
            0,
            dst.unsqueeze(1).expand(-1, self.n_heads),
            scores,
            reduce="amax",
            include_self=True,
        )
        scores_exp = torch.exp(scores - scores_max[dst])  # (E, n_heads)

        scores_sum = torch.zeros(N, self.n_heads, device=x.device)
        scores_sum.scatter_add_(
            0, dst.unsqueeze(1).expand(-1, self.n_heads), scores_exp
        )
        attn = scores_exp / (scores_sum[dst] + 1e-9)  # (E, n_heads)
        attn = self.dropout(attn)

        # Aggregate: value = z_r (neighbor projection)
        agg = torch.zeros(N, self.n_heads, self.d_head, device=x.device)
        agg.scatter_add_(
            0,
            dst.unsqueeze(1).unsqueeze(2).expand(-1, self.n_heads, self.d_head),
            attn.unsqueeze(-1) * z_r[src],
        )  # (N, n_heads, d_head)

        # Concatenate heads and project
        agg = agg.reshape(N, self.hidden_dim)
        agg = self.out_proj(agg)

        return x + self.dropout(agg)  # residual


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

        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

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

        # Build edge_index from adj
        if adj.is_sparse:
            edge_index = adj.coalesce().indices()  # (2, E)
        else:
            edge_index = adj.nonzero(as_tuple=False).t()  # (2, E)

        if self.use_pe:
            pe = degree_encoding(adj, self.hidden_dim, device)  # (N, hidden_dim)
            x = torch.cat([seed_vec, pe], dim=-1)  # (N, 1 + hidden_dim)
        else:
            x = seed_vec  # (N, 1)

        x = self.input_proj(x)  # (N, hidden_dim)

        for layer in self.layers:
            x = layer(x, edge_index)  # (N, hidden_dim)

        out = F.sigmoid(self.output_proj(x))  # (N, 1)
        return out


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
        **_
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.layers = nn.ModuleList(
            [GATLayer(hidden_dim, n_heads, dropout) for _ in range(n_layers)]
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
        h = F.gelu(self.input_proj(X))  # shape: (N, hidden_dim)
        for layer in self.layers:
            h = layer(h, graph.edge_index, graph.edge_weight)  # shape: (N, hidden_dim)

        return self.norm(h)  # shape: (N, hidden_dim)
