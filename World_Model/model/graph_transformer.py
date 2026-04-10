"""
Graph Transformer — Forward Diffusion Model
============================================

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


# Fixed, degree-based Positional Encoding
def degree_encoding(
    adj: torch.Tensor, d_model: int, device: torch.device
) -> torch.Tensor:
    """
    Simple degree-based positional encoding.
    Encodes log(1 + degree) projected to d_model dims via a fixed sinusoidal scheme.

    adj: sparse COO (N, N) or dense (N, N)

    Returns: (N, d_model) float tensor
    """

    # Compute the degree of each node in the graph adjacency matrix
    if adj.is_sparse:
        deg = torch.sparse.sum(adj, dim=1).to_dense()  # (N,)
    else:
        deg = adj.sum(dim=1)

    deg = torch.log1p(deg).unsqueeze(dim=1)  # (N, 1)

    # Sinusoidal projection across d_model dimensions for positional encodings
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float, device=device)
        * -(math.log(10000.0) / d_model)
    )  # (d_model / 2,)

    pe = torch.zeros(deg.shape[0], d_model, device=device)
    pe[:, 0::2] = torch.sin(deg * div)
    pe[:, 1::2] = torch.cos(deg * div[: d_model // 2])

    return pe  # (N, d_model)


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

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

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

    def _reset_parameters(self):
        for m in [self.Wq, self.Wk, self.Wv, self.Wo]:
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: (N, d_model)
        edge_index: (2, E) long — [src, dst]
        returns: (N, d_model)
        """
        N = x.shape[0]

        # Extract source and destination node indices
        # src[i] → dst[i] is the ith edge
        src, dst = edge_index[0], edge_index[1]  # each (E,) E = number of edges

        # Attention sublayer (pre-norm)
        h = self.norm1(x)

        # Project Q/K/V → (N, H, d_k), split by heads
        Q = self.Wq(h).view(N, self.n_heads, self.d_k)
        K = self.Wk(h).view(N, self.n_heads, self.d_k)
        V = self.Wv(h).view(N, self.n_heads, self.d_k)

        # For each edge (u → v), compute attention scores per edge, per head: score = Q[dst] · K[src] / sqrt(d_k)
        # Q[dst]: (E, H, d_k),  K[src]: (E, H, d_k)
        Q_e = Q[dst]  # (E, H, d_k)
        K_e = K[src]  # (E, H, d_k)
        V_e = V[src]  # (E, H, d_k)

        scores = (Q_e * K_e).sum(dim=-1) / math.sqrt(self.d_k)  # (E, H)

        # Compute softmax over incoming edges per node per head
        # Use scatter softmax: subtract max per (dst, head) for stability
        scores_max = torch.full((N, self.n_heads), float("-inf"), device=x.device)
        scores_max.scatter_reduce_(
            0,
            dst.unsqueeze(1).expand(-1, self.n_heads),
            scores,
            reduce="amax",
            include_self=True,
        )
        scores_exp = torch.exp(scores - scores_max[dst])  # (E, H)

        scores_sum = torch.zeros(N, self.n_heads, device=x.device)
        scores_sum.scatter_add_(
            0, dst.unsqueeze(1).expand(-1, self.n_heads), scores_exp
        )
        attn = scores_exp / (scores_sum[dst] + 1e-9)  # (E, H)
        attn = self.dropout(attn)

        # Aggregate values
        agg = torch.zeros(N, self.n_heads, self.d_k, device=x.device)
        agg.scatter_add_(
            0,
            dst.unsqueeze(1).unsqueeze(2).expand(-1, self.n_heads, self.d_k),
            attn.unsqueeze(-1) * V_e,
        )  # (N, H, d_k)

        # Reshape multiple attention heads and output projection
        agg = agg.reshape(N, self.d_model)  # (N, d_model)
        agg = self.Wo(agg)

        # Residual/skip connection
        x = x + self.dropout(agg)

        # FFN sublayer (pre-norm)
        x = x + self.dropout(self.ffn(self.norm2(x)))

        return x  # (N, d_model)


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
    ):
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

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, seed_vec: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        seed_vec: (N, 1) soft seed probabilities
        adj: sparse COO (N, N) or dense (N, N)
        returns: (N, 1) influence probabilities in [0, 1]
        """
        N = seed_vec.shape[0]  # N = number of nodes
        device = seed_vec.device

        # Build edge_index from adj
        if adj.is_sparse:
            edge_index = adj.coalesce().indices()  # (2, E)
        else:
            edge_index = adj.nonzero(as_tuple=False).t()  # (2, E)

        # Compute degree positional encodings
        pe = degree_encoding(adj, self.d_model, device)  # (N, d_model)

        # Input projection: cat(seed_vec, PE) → d_model
        x = self.input_proj(torch.cat([seed_vec, pe], dim=-1))  # (N, d_model)

        # Graph Transformer layers
        for layer in self.layers:
            x = layer(x, edge_index)

        # Output projection
        out = F.sigmoid(self.output_proj(x))  # (N, 1)
        # out = F.elu(self.output_proj(x))  # (N, 1)

        return out
