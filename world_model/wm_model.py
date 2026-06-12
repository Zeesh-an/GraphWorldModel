"""Action-conditioned world model: encoder backbone and 2-logit head."""

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

        # Encoder produces (N, hidden_dim) node embeddings
        self.encoder = BACKBONES[backbone](
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            **bb,
        )

        # Lienar head maps each node's embedding to 2 logits: [next_infected_logit, next_frontier_logit]
        self.head = nn.Linear(
            in_features=hidden_dim, out_features=2
        )  # [next_infected_logit, next_frontier_logit]

        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, X: torch.Tensor, graph) -> torch.Tensor:
        # X = (N, 6) node features, including the state, actions, degree, etc
        # graph = the GraphInput, with adjacency matrix and edge_index/weights

        return self.head(self.encoder(X, graph))  # (N, 2) raw logits
