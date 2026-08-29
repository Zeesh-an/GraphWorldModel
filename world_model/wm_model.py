"""Action-conditioned world model: encoder backbone and 2-logit head."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.wm_competitive import (
    fixed_dominance,
    negative_dominance,
    positive_dominance,
    resolve_tie_break,
)
from data.wm_simulator import (
    blocked,
    epidemic_dynamics,
    spent,
    valid_remove_semantics,
)
from world_model.wm_data import (
    GraphInput,
    ch_add,
    ch_comp_add,
    ch_comp_remove,
    ch_epi_add,
    ch_epi_ever,
    ch_epi_exposed,
    ch_epi_infectious,
    ch_epi_recovered,
    ch_epi_remove,
    ch_frontier,
    ch_infected,
    ch_neg_frontier,
    ch_neg_infected,
    ch_pos_frontier,
    ch_pos_infected,
    ch_remove,
)
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

        # LT's frontier is defined relative to the state BEFORE the action:
        # `Simulator.advance` snapshots `previous_active` first, applies the bag,
        # steps, and returns `frontier = active - previous_active`. So the head
        # needs both activities, not one.
        #
        #   active_pre   who was active at t, before the action. Only the frontier
        #                channel reads it.
        #   active_post  who is active after T_exo. This is what propagates, and
        #                what the infected channel is monotone in.
        #
        # Collapsing the two (which this head used to do) breaks BOTH node ops in
        # opposite directions, and the dataset says so:
        #
        #   add_node    true next-frontier at the target is 1.0 (it was not active
        #               before, it is now) -- the head predicted 0.
        #   remove_node true next-frontier at the target is 0.0 (it WAS active
        #               before, so re-activating is not "new") -- the head
        #               predicted p_new, measured up to 0.92.
        #
        # IC is unaffected and deliberately not changed: its frontier is the set of
        # status-1 spreaders after the step, so a seeded node is correctly 0 there.
        active_pre = X[:, ch_infected]  # shape: (N,)

        # T_exo: add_node -> active; remove_node -> susceptible. One expression
        # covers both remove semantics: under `spent` the node resets to status 0
        # and may re-activate through its (intact) edges; under `blocked` those
        # edges are gone from edge_index, so active_fraction is 0 and the gate
        # holds it down.
        active = torch.clamp(active_pre + X[:, ch_add], max=1.0) * (
            1.0 - X[:, ch_remove]
        )  # shape: (N,), post-action

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

        p_newly = (1.0 - active) * p_new  # post-action susceptibles only
        y_inf = active + p_newly

        # Newly active RELATIVE TO THE PRE-ACTION STATE, which is what
        # `Simulator.advance` labels. clamp at 0 because a `spent` removal makes
        # y_inf < active_pre for exactly one step: the node was active, the action
        # reset it to susceptible, and it may or may not re-cross its threshold.
        # Negative mass there is not a frontier, it is a node leaving.
        y_fr = torch.clamp(y_inf - active_pre, min=0.0)

        probs = torch.stack([y_inf, y_fr], dim=1).clamp(
            prob_epsilon, 1.0 - prob_epsilon
        )  # (N, 2)
        return torch.log(probs) - torch.log1p(-probs)  # -> logits


class _EdgeTransmission(nn.Module):
    """
    One campaign's per-edge propensity `q(u -> v)`, shared by both competitive heads.

    Exactly `ICTransmissionHead`'s edge model, factored out because a competitive
    head needs TWO of them off one shared encoding (§2.2). `oracle` pins q to the
    true probability, `residual` anchors it there and learns only a correction, and
    `positive_prob` overrides the anchor for the limiting campaign under MCICM,
    where `p_L` is a constant the edge weight does not carry.
    """

    def __init__(
        self,
        hidden_dim: int,
        oracle: bool = False,
        residual: bool = False,
        positive_prob: float | None = None,
    ) -> None:
        super().__init__()
        self.oracle = oracle
        self.residual = residual
        self.positive_prob = positive_prob

        if not oracle:
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
        self,
        hidden: torch.Tensor,
        sources: torch.Tensor,
        destinations: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        base = (
            edge_weight
            if self.positive_prob is None
            else torch.full_like(edge_weight, float(self.positive_prob))
        )  # shape: (E,)

        if self.oracle:
            return base.clamp(0.0, 1.0)

        features = torch.cat(
            [hidden[sources], hidden[destinations], edge_weight.unsqueeze(dim=-1)],
            dim=-1,
        )  # shape: (E, 2H + 1)
        logits = self.edge_mlp(features).squeeze(dim=-1)  # shape: (E,)

        if self.residual:
            anchor = base.clamp(prob_epsilon, 1.0 - prob_epsilon)
            logits = logits + torch.log(anchor) - torch.log1p(-anchor)

        return torch.sigmoid(logits)


def _arrival_probability(
    transmission: torch.Tensor,
    frontier: torch.Tensor,
    sources: torch.Tensor,
    destinations: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """`1 - prod_{u->v} (1 - q_uv * frontier_u)`: the IC infection form, per node."""
    gated = (transmission * frontier[sources]).clamp(0.0, 1.0 - prob_epsilon)
    log_survival = torch.log1p(-gated)  # shape: (E,)
    survival = torch.zeros(num_nodes, device=transmission.device).scatter_add_(
        0, destinations, log_survival
    )  # shape: (N,)

    return 1.0 - torch.exp(survival)


def _resolve_tie(
    p_negative: torch.Tensor,
    p_positive: torch.Tensor,
    tie_break: str,
    priority: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    §2.2's tie-break table, in closed form and differentiable in both inputs.

    All three rules preserve the property that made the single-cascade head work: a
    susceptible node with no active in-neighbour in EITHER cascade has
    `p_negative = p_positive = 0`, so both outputs are 0 and the rollout is
    structurally self-terminating rather than saturating.
    """
    both = p_negative * p_positive

    if tie_break == negative_dominance:
        return p_negative, p_positive - both

    if tie_break == positive_dominance:
        return p_negative - both, p_positive

    if tie_break == fixed_dominance:
        # gamma_v is drawn per node per EPISODE and never stored, exactly like the LT
        # threshold, so the head can only learn E[gamma_v], and this is where fixed
        # dominance pays its own partial-observability tax
        gamma = priority if priority is not None else 0.5
        return (
            p_negative - both + gamma * both,
            p_positive - both + (1.0 - gamma) * both,
        )

    raise ValueError(f"unknown tie_break {tie_break!r}")


class CompetitiveICHead(nn.Module):
    """
    Structured two-cascade IC head: run the IC product form TWICE off the shared
    encoding, then compose with an explicit tie-break (§2.2).

        q^N_uv    = sigmoid(MLP_N([h_u, h_v, w_uv]))
        q^P_uv    = sigmoid(MLP_P([h_u, h_v, w_uv]))
        p^N_new(v) = 1 - prod (1 - q^N_uv * negative_frontier_u)
        p^P_new(v) = 1 - prod (1 - q^P_uv * positive_frontier_u)
        (P(v -> negative), P(v -> positive)) = tie_break(p^N_new, p^P_new)

    Warning: §2.2 warns that this factorization "silently assumes MCICM, not COICM",
    because COICM shares one coin per edge between the campaigns. That objection
    applies to the live-edge characterisation and NOT to the stepwise transition this
    head predicts: a node activates in at most one campaign, so each arc is ever
    attempted by exactly one of them, and the two arrival probabilities at a
    susceptible `v` are products over DISJOINT in-edge sets. The factorization is
    therefore exact under both models: see data/wm_competitive.py for the full
    argument, which is also why `positive_prob` (COICM vs MCICM) is the only thing
    that separates them here.

    Returns logits (N, 4) = [negative infected, negative frontier, positive infected,
    positive frontier], so columns 0-1 are the cascade being minimized.
    """

    def __init__(
        self,
        hidden_dim: int,
        tie_break: str = positive_dominance,
        oracle: bool = False,
        residual: bool = False,
        positive_prob: float | None = None,
        remove_semantics: str = blocked,
    ) -> None:
        super().__init__()
        self.tie_break = tie_break
        self.remove_semantics = remove_semantics
        self.negative = _EdgeTransmission(hidden_dim, oracle, residual)
        self.positive = _EdgeTransmission(
            hidden_dim, oracle, residual, positive_prob=positive_prob
        )

        if tie_break == fixed_dominance:
            self.gamma = nn.Linear(hidden_dim, 1)
            nn.init.xavier_uniform_(self.gamma.weight)
            nn.init.zeros_(self.gamma.bias)
        else:
            self.gamma = None

    def forward(
        self, hidden: torch.Tensor, X: torch.Tensor, graph: GraphInput
    ) -> torch.Tensor:
        num_nodes = hidden.shape[0]
        negative_infected, negative_frontier, positive_infected, positive_frontier = (
            competitive_exogenous(X, self.remove_semantics)
        )

        # A susceptible node is one NEITHER cascade owns; both heads write only here
        susceptible = (1.0 - negative_infected) * (1.0 - positive_infected)

        edge_index, edge_weight = graph.edge_index, graph.edge_weight

        if edge_index.numel() == 0:
            p_negative = torch.zeros(num_nodes, device=hidden.device)
            p_positive = torch.zeros(num_nodes, device=hidden.device)
        else:
            sources, destinations = edge_index[0], edge_index[1]
            p_negative = _arrival_probability(
                self.negative(hidden, sources, destinations, edge_weight),
                negative_frontier,
                sources,
                destinations,
                num_nodes,
            )
            p_positive = _arrival_probability(
                self.positive(hidden, sources, destinations, edge_weight),
                positive_frontier,
                sources,
                destinations,
                num_nodes,
            )

        priority = (
            torch.sigmoid(self.gamma(hidden).squeeze(dim=-1))
            if self.gamma is not None
            else None
        )
        p_new_negative, p_new_positive = _resolve_tie(
            p_negative, p_positive, self.tie_break, priority
        )

        return competitive_logits(
            negative_infected,
            positive_infected,
            susceptible * p_new_negative,
            susceptible * p_new_positive,
        )


class CompetitiveLTHead(nn.Module):
    """
    Structured CLT head: the LT threshold form per cascade, then the same tie-break.

    He et al.'s competitive linear threshold gives every node TWO hidden thresholds
    (SDM'12 §3), so this head pays the single-cascade LT partial-observability tax
    twice, which is exactly why research/influence_blocking.md §2.2 predicts IC
    carries this task and LT is the weaker demonstration. It is implemented anyway
    because the registry declares both dynamics and a `--diffusion-model LT` run
    otherwise has no head at all.

        f^N_v, f^P_v = each cascade's active in-neighbour weight fraction
        p^N_new(v)  = [f^N_v > 0] * sigmoid(tau * (f^N_v - theta^N_v))
        p^P_new(v)  = [f^P_v > 0] * sigmoid(tau * (f^P_v - theta^P_v))
    """

    def __init__(
        self,
        hidden_dim: int,
        tie_break: str = negative_dominance,
        remove_semantics: str = blocked,
    ) -> None:
        super().__init__()
        self.tie_break = tie_break
        self.remove_semantics = remove_semantics
        self.theta_negative = nn.Linear(hidden_dim, 1)
        self.theta_positive = nn.Linear(hidden_dim, 1)
        self.log_tau = nn.Parameter(torch.zeros(1))

        for module in (self.theta_negative, self.theta_positive):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        if tie_break == fixed_dominance:
            self.gamma = nn.Linear(hidden_dim, 1)
            nn.init.xavier_uniform_(self.gamma.weight)
            nn.init.zeros_(self.gamma.bias)
        else:
            self.gamma = None

    def forward(
        self, hidden: torch.Tensor, X: torch.Tensor, graph: GraphInput
    ) -> torch.Tensor:
        num_nodes = hidden.shape[0]
        negative_infected, _, positive_infected, _ = competitive_exogenous(
            X, self.remove_semantics
        )
        susceptible = (1.0 - negative_infected) * (1.0 - positive_infected)

        edge_index, edge_weight = graph.edge_index, graph.edge_weight

        if edge_index.numel() == 0:
            zeros = torch.zeros(num_nodes, device=hidden.device)
            negative_fraction = positive_fraction = zeros
        else:
            sources, destinations = edge_index[0], edge_index[1]
            total = torch.zeros(num_nodes, device=hidden.device).scatter_add_(
                0, destinations, edge_weight
            )
            negative_weight = torch.zeros(num_nodes, device=hidden.device).scatter_add_(
                0, destinations, negative_infected[sources] * edge_weight
            )
            positive_weight = torch.zeros(num_nodes, device=hidden.device).scatter_add_(
                0, destinations, positive_infected[sources] * edge_weight
            )
            safe_total = total.clamp(min=prob_epsilon)
            negative_fraction = negative_weight / safe_total
            positive_fraction = positive_weight / safe_total

        tau = F.softplus(self.log_tau)
        p_negative = (negative_fraction > 0).to(negative_fraction.dtype) * torch.sigmoid(
            tau * (negative_fraction - torch.sigmoid(self.theta_negative(hidden).squeeze(dim=-1)))
        )
        p_positive = (positive_fraction > 0).to(positive_fraction.dtype) * torch.sigmoid(
            tau * (positive_fraction - torch.sigmoid(self.theta_positive(hidden).squeeze(dim=-1)))
        )

        priority = (
            torch.sigmoid(self.gamma(hidden).squeeze(dim=-1))
            if self.gamma is not None
            else None
        )
        p_new_negative, p_new_positive = _resolve_tie(
            p_negative, p_positive, self.tie_break, priority
        )

        return competitive_logits(
            negative_infected,
            positive_infected,
            susceptible * p_new_negative,
            susceptible * p_new_positive,
        )


class CompartmentTransitionHead(nn.Module):
    """
    Structured SIR / SIS / SEIR head: a per-node row-stochastic TRANSITION MATRIX,
    not a probability.

    Warning: THIS IS THE ONE PLACE THE STRUCTURED-HEAD IDEA HAD TO CHANGE, and
    research/epidemic_control.md §2.4 is the argument. `ICTransmissionHead` composes

        y_inf = infected + (1 - infected) * p_new

    which is monotone non-decreasing in `infected` BY CONSTRUCTION: `p_new >= 0`, so
    `y_inf >= infected` for every assignment of encoder weights and every value of
    `q(u -> v)`. There is no way to make that expression predict a node LEAVING the
    infected set, and `LTThresholdHead` has the identical shape. That monotonicity
    is load-bearing rather than incidental: `RESULTS.md` records it as what fixed
    rollout saturation (`count_bias` +49 -> +0.27), and it is exactly the
    assumption `I -> R` and `I -> S` violate. So this is a NEW head reusing the
    per-edge transmission model, not an edit to the existing ones, and IC/LT keep
    theirs untouched.

    What replaces it is §2.4's matrix, with the same three structural properties:

            S              E            I            R
        S   1 - p_inf      p_inf        .            .        <- infection is S's only exit
        E   .              1 - alpha    alpha        .        <- SEIR only
        I   .              .            1 - gamma    gamma     <- SIS sends this to S
        R   .              .            .            1        <- absorbing

      * **Only `p_inf` needs the graph**, and it is exactly `ICTransmissionHead`'s
        construction lifted verbatim: `1 - prod(1 - q_uv * infectious_u)`. The rates
        are per-node scalars with no graph term, structurally identical to
        `LTThresholdHead`'s `theta_hat_v`.
      * **The rows are exact, not penalized.** Each row is composed in closed form
        from probabilities, so `S + E + I + R = 1` holds identically and no simplex
        penalty is needed.
      * **Self-termination survives.** A susceptible node with no infectious
        in-neighbour has `p_inf = 0`, and `I` decays geometrically at `gamma`, so a
        free-running rollout still cannot saturate: under SIS too, where nothing
        else would stop it.

    Columns 0-1 of the output are the EVER-infected marginal and the INCIDENCE,
    which is what lets every existing reader keep slicing `probs[:, 0]` and
    `probs[:, 1]` (`wm_data.epidemic_out_channels`). Ever-infected is still monotone
    and that is correct: a node never un-becomes ever-infected under any of the
    three. What is not monotone, and what this head can now represent, is `I`.

    Under `oracle` the whole matrix is the simulator's own: `q = beta_scale * w`,
    `gamma_hat = gamma`, `alpha_hat = alpha`. That makes the head EXACT rather than
    merely well-shaped, which is what `--head structured_oracle` is for here.
    """

    def __init__(
        self,
        hidden_dim: int,
        dynamics: str = "SIR",
        oracle: bool = False,
        residual: bool = False,
        beta_scale: float = 1.0,
        gamma: float | None = None,
        alpha: float | None = None,
        remove_semantics: str = blocked,
    ) -> None:
        super().__init__()

        if dynamics not in epidemic_dynamics:
            raise ValueError(
                f"unknown compartmental model {dynamics!r}; choose one of "
                f"{epidemic_dynamics}"
            )

        if oracle and (gamma is None or (dynamics == "SEIR" and alpha is None)):
            raise ValueError(
                "structured_oracle needs the TRUE rates to pin its matrix to; pass "
                "gamma (and alpha under SEIR) from the dataset's own metadata"
            )

        self.dynamics = dynamics
        self.oracle = oracle
        self.residual = residual
        self.beta_scale = float(beta_scale)
        self.gamma = gamma
        self.alpha = alpha
        self.remove_semantics = remove_semantics

        if not oracle:
            self.edge_mlp = nn.Sequential(
                nn.Linear(2 * hidden_dim + 1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            # Per-node rate of leaving I. One head per dynamics rather than a
            # shared one, because the destination differs (R under SIR/SEIR, S
            # under SIS) and a rate fit against one is not the other's.
            self.leave_infectious = nn.Linear(hidden_dim, 1)
            # ...and E -> I, which only SEIR has a target column for
            self.leave_exposed = (
                nn.Linear(hidden_dim, 1) if dynamics == "SEIR" else None
            )

            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)

                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def _transmission(
        self,
        hidden: torch.Tensor,
        sources: torch.Tensor,
        destinations: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """`q(u -> v)`, anchored on the arc's own `beta_uv = beta_scale * w`."""
        anchor = (self.beta_scale * edge_weight).clamp(0.0, 1.0)  # shape: (E,)

        if self.oracle:
            return anchor

        features = torch.cat(
            [hidden[sources], hidden[destinations], edge_weight.unsqueeze(dim=-1)],
            dim=-1,
        )  # shape: (E, 2H + 1)
        logits = self.edge_mlp(features).squeeze(dim=-1)  # shape: (E,)

        if self.residual:
            # The anchor is beta_uv, NOT the raw w: --epi-beta rescales every arc,
            # and anchoring on w would put the correction a constant factor off
            safe = anchor.clamp(prob_epsilon, 1.0 - prob_epsilon)
            logits = logits + torch.log(safe) - torch.log1p(-safe)

        return torch.sigmoid(logits)

    def _rate(
        self, layer: nn.Module | None, hidden: torch.Tensor, constant: float | None
    ) -> torch.Tensor:
        if self.oracle or layer is None:
            return torch.full(
                (hidden.shape[0],),
                float(constant if constant is not None else 0.0),
                device=hidden.device,
            )

        return torch.sigmoid(layer(hidden).squeeze(dim=-1))

    def forward(
        self, hidden: torch.Tensor, X: torch.Tensor, graph: GraphInput
    ) -> torch.Tensor:
        num_nodes = hidden.shape[0]

        # T_exo. `add_node` is an INDEX CASE and goes straight into I, matching the
        # simulator: a seeded outbreak is already infectious, and under SEIR putting
        # it in E instead would delay every episode's first wave by one step.
        # `remove_node` is a DOSE and under `blocked` empties every compartment
        # including S, so the node's five targets are all zero, which is what the
        # simulator writes for it.
        keep = 1.0 - X[:, ch_epi_remove] if self.remove_semantics == blocked else 1.0
        seeded = X[:, ch_epi_add]

        infectious = torch.clamp(X[:, ch_epi_infectious] + seeded, max=1.0) * keep
        # A node the action just seeded leaves whatever compartment it was in
        exposed = X[:, ch_epi_exposed] * (1.0 - seeded) * keep
        recovered = X[:, ch_epi_recovered] * (1.0 - seeded) * keep
        ever = torch.clamp(X[:, ch_epi_ever] + seeded, max=1.0) * keep
        susceptible = torch.clamp(
            1.0 - exposed - infectious - recovered, min=0.0
        ) * keep

        edge_index, edge_weight = graph.edge_index, graph.edge_weight

        if edge_index.numel() == 0:
            p_infection = torch.zeros(num_nodes, device=hidden.device)
        else:
            sources, destinations = edge_index[0], edge_index[1]
            p_infection = _arrival_probability(
                self._transmission(hidden, sources, destinations, edge_weight),
                infectious,
                sources,
                destinations,
                num_nodes,
            )

        leaving = self._rate(
            getattr(self, "leave_infectious", None), hidden, self.gamma
        )
        newly = susceptible * p_infection

        if self.dynamics == "SEIR":
            promoted = exposed * self._rate(
                getattr(self, "leave_exposed", None), hidden, self.alpha
            )
            next_exposed = exposed - promoted + newly
            next_infectious = infectious - infectious * leaving + promoted
        else:
            next_exposed = torch.zeros_like(newly)
            next_infectious = infectious - infectious * leaving + newly

        # SIS has no absorbing compartment: a node that leaves I is susceptible
        # again and may be re-infected later, which is the case plain monotone
        # composition is not merely loose about but flatly cannot represent
        next_recovered = (
            torch.zeros_like(newly)
            if self.dynamics == "SIS"
            else recovered + infectious * leaving
        )

        # `(1 - ever) * newly` rather than `ever + newly`, and the guard is not
        # cosmetic under SIS: there a node can be ever-infected AND currently
        # susceptible, so a bare sum would push it past 1 and the clamp would hide
        # it. Under SIR/SEIR an ever-infected node has `susceptible = 0` and the
        # factor is a no-op.
        probs = torch.stack(
            [
                ever + (1.0 - ever) * newly,
                newly,
                next_exposed,
                next_infectious,
                next_recovered,
            ],
            dim=1,
        ).clamp(prob_epsilon, 1.0 - prob_epsilon)  # shape: (N, 5)

        return torch.log(probs) - torch.log1p(-probs)


def competitive_exogenous(
    X: torch.Tensor, remove_semantics: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    T_exo for a two-cascade state: apply the action bag, return both (infected, frontier).

    `add_node` seeds the POSITIVE cascade and only where the negative one has not
    already committed the node: re-seeding a rumour-owned node is a no-op in the
    simulator, and a head that let it flip would be fit against a transition that
    never happens. `remove_node` under `blocked` deletes the node from both cascades;
    its incident edges are gone from `edge_index` because the deletion bag carries
    them, which is what also stops it transmitting.
    """
    removed = X[:, ch_comp_remove]
    keep = 1.0 - removed

    negative_infected = X[:, ch_neg_infected] * keep
    negative_frontier = X[:, ch_neg_frontier] * keep

    seeded = X[:, ch_comp_add] * (1.0 - X[:, ch_neg_infected])
    positive_infected = torch.clamp(X[:, ch_pos_infected] + seeded, max=1.0) * keep
    positive_frontier = torch.clamp(X[:, ch_pos_frontier] + seeded, max=1.0) * keep

    return negative_infected, negative_frontier, positive_infected, positive_frontier


def competitive_logits(
    negative_infected: torch.Tensor,
    positive_infected: torch.Tensor,
    newly_negative: torch.Tensor,
    newly_positive: torch.Tensor,
) -> torch.Tensor:
    """Compose the four monotone outputs and convert to logits. Shape: (N, 4)."""
    probs = torch.stack(
        [
            negative_infected + newly_negative,
            newly_negative,
            positive_infected + newly_positive,
            newly_positive,
        ],
        dim=1,
    ).clamp(prob_epsilon, 1.0 - prob_epsilon)  # shape: (N, 4)

    return torch.log(probs) - torch.log1p(-probs)


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
        competitive: bool = False,
        tie_break: str = "auto",
        positive_prob: float | None = None,
        epidemic: bool = False,
        # PREFIXED, and not optional: `backbone_kwargs` already carries GCNII's own
        # `alpha`, so an unprefixed compartmental rate collides with it and every
        # call raises "got multiple values for keyword argument 'alpha'"
        epi_beta: float = 1.0,
        epi_gamma: float | None = None,
        epi_alpha: float | None = None,
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
        self.in_channels = in_channels
        self.remove_semantics = remove_semantics
        self.competitive = competitive
        self.epidemic = epidemic
        # `auto` has no meaning outside a two-cascade run, and resolve_tie_break
        # would map a compartmental dynamics onto IC's rule; the field is inert
        # there, so it is left at what was passed rather than resolved against a
        # dynamics that has no founding paper for it
        self.tie_break = (
            tie_break if epidemic else resolve_tie_break(tie_break, diffusion_model)
        )

        # Encoder produces (N, hidden_dim) node embeddings
        self.encoder = backbones[backbone](
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            **backbone_kwargs,
        )

        # A compartmental task has ONE head across its three dynamics and no
        # `linear` variant, for a sharper version of the competitive reason: an
        # unstructured (N, 5) head has nothing making the four compartments a
        # simplex, nothing stopping `I` from growing without an infectious
        # neighbour, and (the point of the task) nothing that represents
        # recovery as a transition rather than as a coincidence
        # (research/epidemic_control.md §2.4).
        if epidemic:
            if competitive:
                raise ValueError(
                    "a task is either two-CASCADE or four-COMPARTMENT, never both"
                )

            if head_type == "linear":
                raise ValueError(
                    "a compartmental task has no `linear` head: nothing then keeps "
                    "S/E/I/R on the simplex or the rollout self-terminating, and a "
                    "free-running SIS cascade with no structural cap saturates the "
                    "graph. Use --head structured (or structured_residual, which "
                    "anchors q on the true per-arc beta)."
                )

            self.head = CompartmentTransitionHead(
                hidden_dim,
                dynamics=diffusion_model,
                oracle=head_type == "structured_oracle",
                residual=head_type == "structured_residual",
                beta_scale=epi_beta,
                gamma=epi_gamma,
                alpha=epi_alpha,
                remove_semantics=remove_semantics,
            )
        # A competitive task has its own head per dynamics and no `linear` variant:
        # an unstructured (N, 4) head has nothing holding the two cascades apart, so
        # its free-running rollout saturates BOTH and the blocked-influence number
        # comes out of two runaway cascades cancelling
        elif competitive:
            if head_type == "linear":
                raise ValueError(
                    "a competitive task has no `linear` head: nothing then keeps the "
                    "two cascades disjoint or self-terminating, and blocked "
                    "influence is a DIFFERENCE of two saturating rollouts. Use "
                    "--head structured (or structured_residual under IC)."
                )

            if diffusion_model == "LT":
                if head_type != "structured":
                    raise ValueError(
                        f"--head {head_type} is IC-only; CLT carries no per-edge "
                        f"transmission probability to anchor on. Use "
                        f"--head structured with --diffusion-model LT."
                    )

                self.head = CompetitiveLTHead(
                    hidden_dim,
                    tie_break=self.tie_break,
                    remove_semantics=remove_semantics,
                )
            else:
                self.head = CompetitiveICHead(
                    hidden_dim,
                    tie_break=self.tie_break,
                    oracle=head_type == "structured_oracle",
                    residual=head_type == "structured_residual",
                    positive_prob=positive_prob,
                    remove_semantics=remove_semantics,
                )
        elif head_type == "structured":
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

        if self.head_type == "linear" and not (self.competitive or self.epidemic):
            return self.head(hidden)  # (N, 2) logits

        # (N, 2) single-cascade, (N, 4) competitive, (N, 5) compartmental; columns
        # 0-1 are the same PAIR of quantities under all three: the set being scored
        # and who newly joined it, which is what keeps every downstream reader
        # working with no branch
        return self.head(hidden, X, graph)
