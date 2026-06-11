"""Plug-and-play action-conditioned world model: encoder backbone + 2-logit head."""

import torch
import torch.nn as nn

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


class WorldModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        in_channels: int = 6,
        hidden_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
        **bb,
    ) -> None:
        super().__init__()

        if backbone not in BACKBONES:
            raise ValueError(
                f"unknown backbone {backbone}; choose from {list(BACKBONES)}"
            )

        self.encoder = BACKBONES[backbone](
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            **bb,
        )
        self.head = nn.Linear(
            hidden_dim, 2
        )  # [next_infected_logit, next_frontier_logit]

        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self, X: torch.Tensor, graph
    ) -> torch.Tensor:  # graph: GraphInput (duck-typed)
        return self.head(self.encoder(X, graph))  # (N, 2) raw logits
