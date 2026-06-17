"""Action-conditioned world model: encoder backbone and 2-logit head."""

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class LTThresholdHead(nn.Module):
    """
    Structured Linear-Threshold head (Lever 3 for LT): a susceptible node activates
    when the fraction of its active in-neighbors crosses its threshold. LT thresholds
    are hidden and drawn per episode (not stored), so the head predicts the activation
    probability as a learned monotone function of the active-neighbor fraction f_v:

        f_v       = (active in-neighbor weight) / (total in-neighbor weight)
        p_new(v)  = [f_v > 0] * sigmoid(tau * (f_v - theta_hat_v))
        theta_hat = sigmoid(Linear(h_v))   (per-node threshold proxy)
        tau       = softplus(scalar)       (learned sharpness)
        y_inf(v)  = active_v + (1 - active_v) * p_new(v)
        y_fr(v)   = (1 - active_v) * p_new(v)               (newly activated)

    The f_v>0 gate gives the same self-terminating bound as the IC head (no active
    neighbors -> no activation), so the rollout cannot saturate. Returns LOGITS (N, 2).

    Note: LT dynamics are deterministic given the (hidden, random) thresholds, so the
    best a state-only model can do is the threshold marginal P(activate | f_v); one-step
    metrics will be looser than IC, but the rollout marginal (vs the true sim re-drawing
    thresholds) is a fair comparison.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.theta = nn.Linear(hidden_dim, 1)  # per-node threshold proxy
        self.log_tau = nn.Parameter(torch.zeros(1))  # softplus -> sharpness > 0

        nn.init.xavier_uniform_(self.theta.weight)
        nn.init.zeros_(self.theta.bias)

    def forward(self, h: torch.Tensor, X: torch.Tensor, graph) -> torch.Tensor:
        n = h.shape[0]

        # Apply the exogenous action (T_exo): add_node -> active; remove_node -> susceptible
        # (LT remove resets the node to status 0, so it can re-activate).
        active = torch.clamp(X[:, CH_INFECTED] + X[:, CH_ADD], max=1.0) * (
            1.0 - X[:, CH_REMOVE]
        )  # shape: (N,)

        ei, ew = graph.edge_index, graph.edge_weight

        if ei.numel() == 0:
            f = torch.zeros(n, device=h.device)
        else:
            src, dst = ei[0], ei[1]
            num = torch.zeros(n, device=h.device).scatter_add_(
                0, dst, active[src] * ew
            )  # active in-neighbor weight, shape: (N,)
            den = torch.zeros(n, device=h.device).scatter_add_(0, dst, ew)  # total in-weight
            f = num / den.clamp(min=1e-6)  # active-neighbor fraction, shape: (N,)

        theta_hat = torch.sigmoid(self.theta(h).squeeze(dim=-1))  # shape: (N,) in [0, 1]
        tau = F.softplus(self.log_tau)  # scalar > 0
        gate = (f > 0).to(f.dtype)  # structural self-termination: no active neighbor -> no activation
        p_new = gate * torch.sigmoid(tau * (f - theta_hat))  # shape: (N,)

        p_newly = (1.0 - active) * p_new  # susceptibles only
        y_inf = active + p_newly
        y_fr = p_newly  # newly activated = new frontier

        probs = torch.stack([y_inf, y_fr], dim=1).clamp(1e-6, 1.0 - 1e-6)  # (N, 2)
        return torch.log(probs) - torch.log1p(-probs)  # -> logits


class WorldModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        in_channels: int = 6,
        hidden_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
        head_type: str = "linear",
        diffusion_model: str = "IC",
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

        if head_type == "structured":
            # Structured head matched to the dynamics: IC transmission or LT threshold
            self.head = (
                LTThresholdHead(hidden_dim)
                if diffusion_model == "LT"
                else ICTransmissionHead(hidden_dim)
            )
        elif head_type == "structured_oracle":
            # Oracle is IC-only (LT thresholds are not stored, so no oracle there)
            self.head = ICTransmissionHead(hidden_dim, oracle=True)
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
