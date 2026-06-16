"""Action-conditioned world model: encoder backbone and 2-logit head."""

import torch
import torch.nn as nn

from wm_data import CH_INFECTED, CH_FRONTIER, CH_ADD, CH_REMOVE
from model.gcn import GCNEncoder
from model.graphsage import GraphSAGEEncoder
from model.gat import GATEncoder
from model.graph_transformer import GraphTransformerEncoder
from model.gcnii import GCNIIEncoder

BACKBONES = {
    "gcn": GCNEncoder,
    "sage": GraphSAGEEncoder,
    "gt": GraphTransformerEncoder,
    "gat": GATEncoder,
    "gcnii": GCNIIEncoder,
}


class ICTransmissionHead(nn.Module):
    """
    Structured IC head (Lever 3): predict a per-edge transmission propensity q(u->v)
    and DERIVE the next-state marginals with the true IC form, instead of predicting
    node states directly. Locality + monotonicity are baked in, so a free-running
    rollout cannot saturate spuriously: a susceptible node with no active in-neighbor
    has p_new = 0, and the frontier self-terminates as the cascade runs.

        q_uv     = sigmoid(MLP([h_u, h_v, w_uv]))        (oracle: q_uv = w_uv)
        t_uv     = q_uv * frontier_u                     (only active sources transmit)
        p_new(v) = 1 - prod_{u->v} (1 - t_uv)            (IC infection form)
        y_inf(v) = infected_v + (1 - infected_v) * p_new(v)   (monotone)
        y_fr(v)  = (1 - infected_v) * p_new(v)               (new frontier = newly infected)

    Returns LOGITS (N, 2) so training (BCEWithLogits) and eval (sigmoid) are unchanged.
    """

    def __init__(self, hidden_dim: int, oracle: bool = False) -> None:
        super().__init__()
        self.oracle = oracle

        if not oracle:
            # [h_u, h_v, edge_weight] -> transmission logit
            self.edge_mlp = nn.Sequential(
                nn.Linear(2 * hidden_dim + 1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor, X: torch.Tensor, graph) -> torch.Tensor:
        n = h.shape[0]

        # Apply the exogenous action (T_exo) first: add_node -> infected + active spreader;
        # remove_node -> stays infected (IC) but drops out of the frontier. Edge actions
        # are already reflected in graph.edge_index / edge_weight.
        infected = torch.clamp(X[:, CH_INFECTED] + X[:, CH_ADD], max=1.0)  # shape: (N,)
        frontier = torch.clamp(X[:, CH_FRONTIER] + X[:, CH_ADD], max=1.0) * (
            1.0 - X[:, CH_REMOVE]
        )  # shape: (N,)

        ei, ew = graph.edge_index, graph.edge_weight

        if ei.numel() == 0:
            p_new = torch.zeros(n, device=h.device)
        else:
            src, dst = ei[0], ei[1]
            if self.oracle:
                q = ew.clamp(0.0, 1.0)  # true edge transmission prob (no learning)
            else:
                edge_in = torch.cat(
                    [h[src], h[dst], ew.unsqueeze(dim=-1)], dim=-1
                )  # shape: (E, 2H + 1)
                q = torch.sigmoid(self.edge_mlp(edge_in)).squeeze(dim=-1)  # shape: (E,)

            # transmission gated by an active (frontier) source
            t = (q * frontier[src]).clamp(0.0, 1.0 - 1e-6)  # shape: (E,)
            log_surv = torch.log1p(-t)  # log(1 - t), shape: (E,)
            sum_log = torch.zeros(n, device=h.device).scatter_add_(
                0, dst, log_surv
            )  # shape: (N,)
            p_new = 1.0 - torch.exp(sum_log)  # shape: (N,)

        p_newly = (1.0 - infected) * p_new  # susceptibles only
        y_inf = infected + p_newly  # monotone: infected stay infected
        y_fr = p_newly  # new frontier = newly infected

        probs = torch.stack([y_inf, y_fr], dim=1).clamp(1e-6, 1.0 - 1e-6)  # (N, 2)
        return torch.log(probs) - torch.log1p(-probs)  # -> logits (sigmoid recovers probs)


class WorldModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        in_channels: int = 6,
        hidden_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
        head_type: str = "linear",
        **bb,
    ) -> None:
        super().__init__()

        if backbone not in BACKBONES:
            raise ValueError(
                f"unknown backbone {backbone}; choose from {list(BACKBONES)}"
            )

        self.head_type = head_type

        # Encoder produces (N, hidden_dim) node embeddings
        self.encoder = BACKBONES[backbone](
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            **bb,
        )

        if head_type in ("structured", "structured_oracle"):
            # Structured IC head: predict per-edge transmission, derive next-state marginals
            self.head = ICTransmissionHead(
                hidden_dim, oracle=(head_type == "structured_oracle")
            )
        elif head_type == "linear":
            # Free head: each node embedding -> 2 logits [next_infected, next_frontier]
            self.head = nn.Linear(in_features=hidden_dim, out_features=2)
            nn.init.xavier_uniform_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        else:
            raise ValueError(
                f"unknown head_type {head_type}; choose from linear, structured, structured_oracle"
            )

    def forward(self, X: torch.Tensor, graph) -> torch.Tensor:
        # X: (N, in_channels) node features; graph: GraphInput
        h = self.encoder(X, graph)

        if self.head_type == "linear":
            return self.head(h)  # (N, 2) logits

        return self.head(h, X, graph)  # (N, 2) logits (structured)
