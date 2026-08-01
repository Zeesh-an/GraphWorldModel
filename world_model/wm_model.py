"""Action-conditioned world model: encoder backbone and 2-logit head."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.wm_simulator import blocked, spent, valid_remove_semantics
from world_model.wm_data import GraphInput, ch_add, ch_frontier, ch_infected, ch_remove
from world_model.model.gcn import GCNEncoder
from world_model.model.graphsage import GraphSAGEEncoder
from world_model.model.gat import GATEncoder
from world_model.model.graph_transformer import GraphTransformerEncoder
from world_model.model.gcnii import GCNIIEncoder

backbones = {
    "gcn": GCNEncoder,
    "sage": GraphSAGEEncoder,
    "gt": GraphTransformerEncoder,
    "gat": GATEncoder,
    "gcnii": GCNIIEncoder,
}

# Probability clamp keeping logit conversion and transmission products finite
prob_epsilon = 1e-6


class ICTransmissionHead(nn.Module):
    """
    Structured IC head: predict a per-edge transmission propensity q(u -> v)
    and DERIVE the next-state marginals with the true IC form, instead of predicting
    node states directly. Locality + monotonicity are baked in, so a free-running
    rollout cannot saturate spuriously: a susceptible node with no active in-neighbor
    has p_new = 0, and the frontier self-terminates as the cascade runs.

    q_uv     = sigmoid(MLP([h_u, h_v, w_uv]))        (oracle: q_uv = w_uv)
    t_uv     = q_uv * frontier_u                     (only active sources transmit)
    p_new(v) = 1 - prod_{u->v} (1 - t_uv)            (IC infection form)
    y_inf(v) = infected_v + (1 - infected_v) * p_new(v)   (monotone)
    y_fr(v)  = (1 - infected_v) * p_new(v)               (new frontier = newly infected)

    `remove_semantics` must match the dataset's: under `spent` a removed node stays
    infected (NDlib status 2 is still counted), under `blocked` it leaves the graph
    and T_exo zeroes it. Getting this wrong puts the head's T_exo permanently at
    odds with the targets it is fit against.

    Returns logits (N, 2) so training (BCEWithLogits) and eval (sigmoid) are unchanged.
    """

    def __init__(
        self,
        hidden_dim: int,
        oracle: bool = False,
        residual: bool = False,
        remove_semantics: str = spent,
    ) -> None:
        super().__init__()
        self.oracle = oracle
        self.residual = residual
        self.remove_semantics = remove_semantics

        if not oracle:
            # [h_u, h_v, edge_weight] -> transmission logit
            self.edge_mlp = nn.Sequential(
                nn.Linear(2 * hidden_dim + 1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)

                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(
        self, hidden: torch.Tensor, X: torch.Tensor, graph: GraphInput
    ) -> torch.Tensor:
        num_nodes = hidden.shape[0]

        # Apply the exogenous action (T_exo) first: add_node -> infected + active spreader;
        # remove_node -> drops out of the frontier, and under `spent` STAYS infected
        # (NDlib status 2 is still counted). Edge actions are already reflected in
        # graph.edge_index / edge_weight.
        infected = torch.clamp(X[:, ch_infected] + X[:, ch_add], max=1.0)  # shape: (N,)

        if self.remove_semantics == blocked:
            # The node left the graph, so it is not part of the spread. Its edges
            # are gone from edge_index too (the deletion bag carries them), which
            # is what also stops it transmitting and being re-infected.
            infected = infected * (1.0 - X[:, ch_remove])  # shape: (N,)

        frontier = torch.clamp(X[:, ch_frontier] + X[:, ch_add], max=1.0) * (
            1.0 - X[:, ch_remove]
        )  # shape: (N,)

        edge_index, edge_weight = graph.edge_index, graph.edge_weight

        if edge_index.numel() == 0:
            p_new = torch.zeros(num_nodes, device=hidden.device)
        else:
            sources, destinations = edge_index[0], edge_index[1]
            if self.oracle:
                # True edge transmission prob (no learning)
                transmission = edge_weight.clamp(0.0, 1.0)
            else:
                edge_features = torch.cat(
                    [
                        hidden[sources],
                        hidden[destinations],
                        edge_weight.unsqueeze(dim=-1),
                    ],
                    dim=-1,
                )  # shape: (E, 2H + 1)
                transmission_logits = self.edge_mlp(edge_features).squeeze(
                    dim=-1
                )  # shape: (E,)

                if self.residual:
                    # Anchor q on the true IC transmission prob: the MLP learns only
                    # a residual correction, so zero correction == the oracle head
                    # Removes the need for edge-weight-diverse training data to pin down the w -> q mapping
                    anchor = edge_weight.clamp(prob_epsilon, 1.0 - prob_epsilon)
                    transmission_logits = (
                        transmission_logits + torch.log(anchor) - torch.log1p(-anchor)
                    )

                transmission = torch.sigmoid(transmission_logits)  # shape: (E,)

            # Transmission gated by an active (frontier) source
            gated = (transmission * frontier[sources]).clamp(
                0.0, 1.0 - prob_epsilon
            )  # shape: (E,)
            log_survival = torch.log1p(-gated)  # log(1 - t), shape: (E,)
            survival_log_sum = torch.zeros(
                num_nodes, device=hidden.device
            ).scatter_add_(
                0, destinations, log_survival
            )  # shape: (N,)
            p_new = 1.0 - torch.exp(survival_log_sum)  # shape: (N,)

        p_newly = (1.0 - infected) * p_new  # susceptibles only
        y_inf = infected + p_newly  # monotone: infected stay infected
        y_fr = p_newly  # new frontier = newly infected

        probs = torch.stack([y_inf, y_fr], dim=1).clamp(
            prob_epsilon, 1.0 - prob_epsilon
        )  # (N, 2)

        # -> logits (sigmoid recovers probs)
        return torch.log(probs) - torch.log1p(-probs)


class LTThresholdHead(nn.Module):
    """
    Structured Linear-Threshold head (Lever 3 for LT): a susceptible node activates
    when the fraction of its active in-neighbors crosses its threshold. LT thresholds
    are hidden and drawn per episode (not stored), so the head predicts the activation
    probability as a learned monotone function of the active-neighbor fraction f_v:

    f_v       = (active in-neighbor weight) / (total in-neighbor weight)
    p_new(v)  = [f_v > 0] * sigmoid(tau * (f_v - theta_hat_v))
    theta_hat = sigmoid(Linear(h_v))        (per-node threshold proxy)
    tau       = softplus(scalar)            (learned sharpness)
    y_inf(v)  = active_v + (1 - active_v) * p_new(v)
    y_fr(v)   = (1 - active_v) * p_new(v)   (newly activated)

    The f_v > 0 gate gives the same self-terminating bound as the IC head (no active
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

    def forward(
        self, hidden: torch.Tensor, X: torch.Tensor, graph: GraphInput
    ) -> torch.Tensor:
        num_nodes = hidden.shape[0]

        # Apply the exogenous action (T_exo): add_node -> active; remove_node -> susceptible.
        # One expression covers both remove semantics: under `spent` the node resets to
        # status 0 and may re-activate through its (intact) edges; under `blocked` those
        # edges are gone from edge_index, so active_fraction is 0 and the gate holds it
        # down. No remove_semantics branch is needed here.
        active = torch.clamp(X[:, ch_infected] + X[:, ch_add], max=1.0) * (
            1.0 - X[:, ch_remove]
        )  # shape: (N,)

        edge_index, edge_weight = graph.edge_index, graph.edge_weight

        if edge_index.numel() == 0:
            active_fraction = torch.zeros(num_nodes, device=hidden.device)
        else:
            sources, destinations = edge_index[0], edge_index[1]
            # Active in-neighbor weight, shape: (N,)
            active_weight = torch.zeros(num_nodes, device=hidden.device).scatter_add_(
                0, destinations, active[sources] * edge_weight
            )
            # Total in-weight, shape: (N,)
            total_weight = torch.zeros(num_nodes, device=hidden.device).scatter_add_(
                0, destinations, edge_weight
            )
            # Active-neighbor fraction, shape: (N,)
            active_fraction = active_weight / total_weight.clamp(min=prob_epsilon)

        theta_hat = torch.sigmoid(
            self.theta(hidden).squeeze(dim=-1)
        )  # shape: (N,) in [0, 1]
        tau = F.softplus(self.log_tau)  # scalar > 0

        # Structural self-termination: no active neighbor -> no activation
        gate = (active_fraction > 0).to(active_fraction.dtype)
        p_new = gate * torch.sigmoid(tau * (active_fraction - theta_hat))  # shape: (N,)

        p_newly = (1.0 - active) * p_new  # susceptibles only
        y_inf = active + p_newly
        y_fr = p_newly  # newly activated = new frontier

        probs = torch.stack([y_inf, y_fr], dim=1).clamp(
            prob_epsilon, 1.0 - prob_epsilon
        )  # (N, 2)
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
        remove_semantics: str = spent,
        **backbone_kwargs: object,
    ) -> None:
        super().__init__()

        if backbone not in backbones:
            raise ValueError(
                f"unknown backbone {backbone}; choose from {list(backbones)}"
            )

        if remove_semantics not in valid_remove_semantics:
            raise ValueError(
                f"unknown remove_semantics {remove_semantics!r}; "
                f"choose one of {valid_remove_semantics}"
            )

        self.head_type = head_type
        self.remove_semantics = remove_semantics

        # Encoder produces (N, hidden_dim) node embeddings
        self.encoder = backbones[backbone](
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            **backbone_kwargs,
        )

        if head_type == "structured":
            # Structured head matched to the dynamics: IC transmission or LT threshold
            self.head = (
                LTThresholdHead(hidden_dim)
                if diffusion_model == "LT"
                else ICTransmissionHead(hidden_dim, remove_semantics=remove_semantics)
            )
        elif head_type == "structured_residual":
            # IC-only: anchors q on the IC edge transmission prob (q = w at zero
            # correction), so calibration does not depend on edge-op-diverse data
            if diffusion_model == "LT":
                raise ValueError(
                    "structured_residual is IC-only: it anchors q on the IC edge "
                    "transmission prob; LT edge weights carry no transmission meaning"
                )
            self.head = ICTransmissionHead(
                hidden_dim, residual=True, remove_semantics=remove_semantics
            )
        elif head_type == "structured_oracle":
            # Oracle is IC-only (LT thresholds are not stored, so no oracle there)
            self.head = ICTransmissionHead(
                hidden_dim, oracle=True, remove_semantics=remove_semantics
            )
        elif head_type == "linear":
            # Free head: each node embedding -> 2 logits [next_infected, next_frontier]
            self.head = nn.Linear(in_features=hidden_dim, out_features=2)
            nn.init.xavier_uniform_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        else:
            raise ValueError(
                f"unknown head_type {head_type}; choose from linear, structured, "
                f"structured_residual, structured_oracle"
            )

    def forward(self, X: torch.Tensor, graph: GraphInput) -> torch.Tensor:
        # X: (N, in_channels) node features; graph: GraphInput
        hidden = self.encoder(X, graph)

        if self.head_type == "linear":
            return self.head(hidden)  # (N, 2) logits

        return self.head(hidden, X, graph)  # (N, 2) logits (structured)
