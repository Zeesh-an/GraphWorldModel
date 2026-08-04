"""
Arm A: freeze the world model, gradient-descend a relaxed source vector against it.

This is `research/source_localization.md` §2.2's framing, built exactly once and
on purpose. Every generative method in that literature solves the same per-instance
problem,

    x_hat = argmax_x  log p_psi(y | x, G)  +  log p(x)

and differs only in HOW it inverts. SL-VAE relaxes `x` to `x~ in [0,1]^N` and runs
Adam on it; IVGD runs its network backwards and projects; DDMSL runs a reverse
denoising chain. The forward model `p_psi` is a component, and SL-VAE is explicit
that it is a REPLACEABLE one — the paper plugs in GAT, MONSTOR and DeepIS and
reports no significant difference [verified, §4]. So "SL-VAE, but with our world
model as p_psi" is a paper the seed authors already wrote.

**It is therefore the CONTROL, not the method.** §2.3 contributes the argmax
instead: search the space of inversion PROGRAMS once, offline, and run the winner
on every instance. Arm 6 vs arm A is the methodological claim, and a control that
is strawmanned proves nothing — hence the VAE prior here rather than descent
alone. SL-VAE's own ablation (§5.1, Table 4) puts the prior at +0.19 F1 on Jazz
and +0.26 on Network Science, so `--sl-prior none` reproduces `SL-VAE (a)` and
`--sl-prior vae` reproduces the full method.

Two things make our likelihood differentiable in `x~` for free, and they are the
only reason this file is short:

  * `ICTransmissionHead` composes `p_new = 1 - prod(1 - q * frontier_u)` from
    continuous channels, so feeding it a SOFT `(infected, frontier, add)` state is
    already well defined — no relaxation of the head is needed.
  * The action enters through channel `CH_ADD`, so the decision variable is one
    column of `X` and `torch.autograd` reaches it through the whole unrolled
    rollout without a single line of custom backward code.

Standalone (no coding agent, no pipeline):

    python -m world_model.wm_sl \\
        --data-dir results/source_localization/jazz/default/data \\
        --wm-results-json results/source_localization/jazz/default/world_model/sage_IC.json \\
        --sl-prior vae --instances 20
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from coding_agent.types import GraphInfo
from world_model.wm_data import (
    build_graph_input,
    ch_add,
    ch_degree,
    ch_edge,
    ch_frontier,
    ch_infected,
    ch_remove,
    in_channels,
    load_graph_store,
)
from world_model.wm_metrics import localization_metrics, resimulation_error
from world_model.wm_model import WorldModel

no_prior = "none"
vae_prior = "vae"
valid_priors = (no_prior, vae_prior)

# Adam on the relaxed source logits. 200 steps at lr 0.1 converges on every graph
# in the benchmark suite; the count is the `I` of §2.3.2's inference-cost
# comparison, so it is reported rather than buried.
default_steps = 200
default_lr = 0.1
# Weight on the cardinality term ||sum(x~) - k||^2. SL-VAE thresholds a relaxation
# and lets the count fall where it may, which is visible in §5.2 as recall 1.0000
# on five of six graphs; we are given k, so pinning the mass is both the fairer
# comparison and the given-k convention of §8.2.
default_cardinality_weight = 0.05
# Weight on -log p(x~) under the learned prior
default_prior_weight = 1.0
# Initial logit for every node: sigmoid(-2.2) ~ 0.1, i.e. a diffuse start that
# neither pre-selects a source nor sits at a saturated gradient
initial_logit = -2.2

# Source VAE. Small on purpose: it is fit on a few hundred binary vectors of
# length N, and anything wider memorizes the training source sets outright.
default_latent_dim = 16
default_prior_hidden = 128
default_prior_epochs = 300
default_prior_lr = 1e-3
kl_weight = 1e-3

probability_floor = 1e-6


def build_graph_tensors(
    graph: GraphInfo, diffusion_model: str, device: torch.device
) -> tuple:
    """(GraphInput, degree channel) for one graph — everything the unroll reuses."""
    graph_input = build_graph_input(
        graph.edge_index,
        graph.ic_probs,
        graph.num_nodes,
        diffusion_model,
        device,
    )

    degrees = np.zeros(graph.num_nodes, dtype=np.float32)
    if graph.edge_index.size:
        np.add.at(degrees, graph.edge_index[0], 1.0)
        np.add.at(degrees, graph.edge_index[1], 1.0)

    return graph_input, torch.from_numpy(np.log1p(degrees)).to(device)


def soft_rollout(
    model: nn.Module,
    graph_input: object,
    degree_channel: torch.Tensor,
    relaxed_sources: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """
    Differentiable `steps`-step unroll of f_theta from a RELAXED source vector.

    The deterministic twin of `WorldModelEnvironment.rollout`: identical state
    recursion, with the per-timestep Bernoulli draw replaced by the marginal
    itself. That substitution is a mean-field approximation and it is the standard
    one for this construction, but it is worth naming — the sampled ensemble and
    this unroll agree in expectation only when the head is locally linear, and the
    IC head is not.

    Shapes: relaxed_sources (N,) in [0, 1]; returns P(infected at the end), (N,).
    """
    num_nodes = degree_channel.shape[0]
    infected = torch.zeros(num_nodes, device=degree_channel.device)
    frontier = torch.zeros(num_nodes, device=degree_channel.device)

    zeros = torch.zeros(num_nodes, device=degree_channel.device)

    for step in range(steps + 1):
        # Columns assembled by stacking rather than in-place writes: autograd then
        # sees one graph-building op instead of an aliased view chain, and the
        # channel order is the same CH_* layout wm_data documents
        columns = [None] * in_channels
        columns[ch_infected] = infected
        columns[ch_frontier] = frontier
        columns[ch_degree] = degree_channel
        # The whole decision variable, in one channel: the seed commit lands at
        # t=0 and nothing is injected afterwards, which is the diffusion-only
        # episode the data was generated as
        columns[ch_add] = relaxed_sources if step == 0 else zeros
        columns[ch_remove] = zeros
        columns[ch_edge] = zeros

        features = torch.stack(columns, dim=1)  # shape: (N, 6)
        probabilities = torch.sigmoid(model(features, graph_input))  # shape: (N, 2)
        infected, frontier = probabilities[:, 0], probabilities[:, 1]

    return infected


class SourceVAE(nn.Module):
    """
    The generative prior over source sets, `p_theta(x | z) q_phi(z | x)`.

    SL-VAE's second half, and the ablation row that matters: its Table 4 shows the
    prior is worth +0.19 F1 on Jazz over descent alone. Deliberately a plain
    Bernoulli-decoder VAE rather than anything clever — the point is a fair control,
    and a prior the agent's PROGRAM cannot express is exactly what §2.3.1 argues
    the trade is ("a VAE prior cannot express 'at most one source per 3-hop ball';
    ten lines of Python can").
    """

    def __init__(
        self,
        num_nodes: int,
        latent_dim: int = default_latent_dim,
        hidden_dim: int = default_prior_hidden,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_nodes, hidden_dim),
            nn.GELU(),
        )
        self.to_mean = nn.Linear(hidden_dim, latent_dim)
        self.to_log_variance = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_nodes),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)  # shape: (batch, hidden)

        return self.to_mean(hidden), self.to_log_variance(hidden)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_variance = self.encode(x)
        latent = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)

        return self.decoder(latent), mean, log_variance

    def negative_log_prior(self, x: torch.Tensor) -> torch.Tensor:
        """
        -log p(x) at the posterior mean — the term added to the inversion loss.

        The mean rather than a sample, because the inversion is a deterministic
        optimization and resampling `z` every Adam step would make the objective
        stochastic in a way SL-VAE's own procedure is not.
        """
        mean, _ = self.encode(x.unsqueeze(dim=0))
        logits = self.decoder(mean).squeeze(dim=0)  # shape: (N,)

        return F.binary_cross_entropy_with_logits(logits, x, reduction="sum")


def train_source_prior(
    source_sets: list[list[int]],
    num_nodes: int,
    latent_dim: int = default_latent_dim,
    hidden_dim: int = default_prior_hidden,
    epochs: int = default_prior_epochs,
    lr: float = default_prior_lr,
    device: str = "cpu",
    seed: int = 42,
) -> SourceVAE:
    """Fit the prior on the TRAINING split's observed source sets."""
    torch.manual_seed(seed)
    device = torch.device(device)

    data = torch.zeros(len(source_sets), num_nodes, device=device)
    for row, sources in enumerate(source_sets):
        data[row, list(sources)] = 1.0

    prior = SourceVAE(num_nodes, latent_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(prior.parameters(), lr=lr)
    prior.train()

    progress_bar = tqdm(range(epochs), desc="source prior")
    for _ in progress_bar:
        optimizer.zero_grad()
        logits, mean, log_variance = prior(data)
        reconstruction = F.binary_cross_entropy_with_logits(
            logits, data, reduction="sum"
        ) / len(source_sets)
        divergence = -0.5 * torch.sum(
            1 + log_variance - mean.pow(2) - log_variance.exp()
        ) / len(source_sets)
        loss = reconstruction + kl_weight * divergence

        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    prior.eval()

    return prior


def invert(
    model: nn.Module,
    graph_input: object,
    degree_channel: torch.Tensor,
    observation: np.ndarray,
    budget: int,
    steps: int,
    horizon: int,
    lr: float = default_lr,
    cardinality_weight: float = default_cardinality_weight,
    prior: SourceVAE | None = None,
    prior_weight: float = default_prior_weight,
) -> tuple[list[int], np.ndarray, dict]:
    """
    Recover one source set by Adam on the relaxed vector. Returns (sources, scores, info).

    `scores` is the converged `x~`, which is a genuine continuous ranking over all
    N nodes — so this arm gets the AUC column the literature reports rather than
    the rank-derived stand-in a set-only method falls back to.
    """
    device = degree_channel.device
    target = torch.from_numpy(np.asarray(observation, dtype=np.float32)).to(device)
    logits = torch.full(
        (degree_channel.shape[0],), initial_logit, device=device, requires_grad=True
    )
    optimizer = torch.optim.Adam([logits], lr=lr)

    trace = []

    for _ in range(steps):
        optimizer.zero_grad()
        relaxed = torch.sigmoid(logits)  # shape: (N,)
        predicted = soft_rollout(model, graph_input, degree_channel, relaxed, horizon)

        data_term = torch.sum((target - predicted) ** 2)
        cardinality = cardinality_weight * (relaxed.sum() - float(budget)) ** 2
        prior_term = (
            prior_weight * prior.negative_log_prior(relaxed)
            if prior is not None
            else torch.zeros((), device=device)
        )
        loss = data_term + cardinality + prior_term

        loss.backward()
        optimizer.step()
        trace.append(float(loss.item()))

    with torch.inference_mode():
        relaxed = torch.sigmoid(logits).detach()
        scores = relaxed.cpu().numpy().astype(np.float64)

    order = np.lexsort((np.arange(scores.size), -scores))
    sources = [int(node) for node in order[:budget]]

    return (
        sources,
        scores,
        {
            "final_loss": trace[-1] if trace else None,
            "loss_trace": [round(value, 6) for value in trace[:: max(1, steps // 20)]],
            # I of §2.3.2: gradient steps this instance paid, each one a full
            # `horizon`-step unroll of f_theta. The number arm 6's forward-call
            # count is compared against.
            "gradient_steps": steps,
            "forward_unrolls": steps,
        },
    )


def load_world_model(results_json: str, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild the trained WorldModel named by a train_wm.py results JSON, frozen."""
    config = json.loads(Path(results_json).read_text())["config"]
    model = WorldModel(
        config["model"],
        in_channels=in_channels,
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        dropout=0.0,
        head_type=config.get("head", "linear"),
        diffusion_model=config["diffusion_model"],
        remove_semantics=config.get("remove_semantics", "spent"),
        n_heads=config["n_heads"],
        ffn_dim=config["ffn_dim"],
        alpha=config["gcnii_alpha"],
        lamda=config["gcnii_lamda"],
    )

    checkpoint = (
        Path(config["ckpt_dir"]) / f"wm_{config['model']}_{config['diffusion_model']}.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"world-model checkpoint not found: {checkpoint}")

    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.to(device).eval()

    # Frozen: §2.5's fifth row. The world model is never differentiated INTO here,
    # only through — the gradient flows to x~ and stops.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arm A: per-instance gradient inversion of a frozen world model"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="generated data directory holding the graph store and transitions "
        "(default: required).",
    )
    parser.add_argument(
        "--wm-results-json",
        type=str,
        required=True,
        help="train_wm.py results JSON naming the frozen forward model (default: required).",
    )
    parser.add_argument(
        "--diffusion-model",
        type=str,
        default="IC",
        choices=["IC", "LT"],
        help="dynamics the checkpoint was trained on (default: IC).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="episode split to invert (default: test).",
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=20,
        help="labelled episodes to invert (default: 20).",
    )
    parser.add_argument(
        "--observation",
        type=str,
        default="marginal",
        choices=["marginal", "binary"],
        help="which observation y to invert; binary is the column comparable to "
        "the published tables (default: marginal).",
    )
    parser.add_argument(
        "--sl-prior",
        type=str,
        default=vae_prior,
        choices=list(valid_priors),
        help="none reproduces SL-VAE (a) (forward model + descent, no prior); vae "
        f"reproduces the full method (default: {vae_prior}).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=default_steps,
        help=f"Adam steps per instance (default: {default_steps}).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=default_lr,
        help=f"Adam learning rate on the source logits (default: {default_lr}).",
    )
    parser.add_argument(
        "--cardinality-weight",
        type=float,
        default=default_cardinality_weight,
        help="weight on (sum(x~) - k)^2, the given-k constraint "
        f"(default: {default_cardinality_weight}).",
    )
    parser.add_argument(
        "--prior-weight",
        type=float,
        default=default_prior_weight,
        help=f"weight on -log p(x~) (default: {default_prior_weight}).",
    )
    parser.add_argument(
        "--prior-epochs",
        type=int,
        default=default_prior_epochs,
        help=f"epochs fitting the source VAE (default: {default_prior_epochs}).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="unroll length of the soft rollout (default: 10).",
    )
    parser.add_argument(
        "--graph-id",
        type=str,
        default=None,
        help="graph id inside the store (default: the first).",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="torch device string (default: cpu)."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)."
    )
    parser.add_argument(
        "--out-json", type=str, default=None, help="results path (default: None)."
    )

    args = parser.parse_args()

    # Imported here rather than at module scope: coding_agent.localization imports
    # world_model.wm_metrics, and a top-level import would make the two modules
    # circular for anything that reaches wm_sl through the coding agent
    from coding_agent.localization import load_instances

    store = load_graph_store(args.data_dir)
    graph_id = args.graph_id or next(iter(store))
    graph = GraphInfo.from_store_entry(store[graph_id])

    model, config = load_world_model(args.wm_results_json, args.device)
    device = torch.device(args.device)
    graph_input, degree_channel = build_graph_tensors(
        graph, args.diffusion_model, device
    )

    prior = None
    if args.sl_prior == vae_prior:
        training = load_instances(
            args.data_dir,
            args.diffusion_model,
            "train",
            graph_id=graph_id,
            limit=0,
            seed=args.seed,
        )
        print(f"[wm_sl] fitting the source prior on {len(training)} training episodes")
        prior = train_source_prior(
            [instance.sources for instance in training],
            graph.num_nodes,
            epochs=args.prior_epochs,
            device=args.device,
            seed=args.seed,
        )

    instances = load_instances(
        args.data_dir,
        args.diffusion_model,
        args.split,
        graph_id=graph_id,
        observation=args.observation,
        limit=args.instances,
        seed=args.seed,
    )
    print(f"[wm_sl] inverting {len(instances)} {args.split} episodes on {graph_id}")

    start = time.perf_counter()
    per_instance = []

    for instance in tqdm(instances, desc="inverting"):
        sources, scores, info = invert(
            model,
            graph_input,
            degree_channel,
            instance.observation,
            budget=max(1, instance.source_count),
            steps=args.steps,
            horizon=args.horizon,
            lr=args.lr,
            cardinality_weight=args.cardinality_weight,
            prior=prior,
            prior_weight=args.prior_weight,
        )

        metrics = localization_metrics(
            sources, instance.sources, instance.num_nodes, scores
        )
        with torch.inference_mode():
            relaxed = torch.zeros(instance.num_nodes, device=device)
            relaxed[sources] = 1.0
            predicted = soft_rollout(
                model, graph_input, degree_channel, relaxed, args.horizon
            )

        metrics["resim_error"] = resimulation_error(
            predicted.cpu().numpy(), instance.observation
        )
        metrics |= {"episode_id": instance.episode_id, **info}
        per_instance.append(metrics)

    means = {
        key: float(np.nanmean([entry[key] for entry in per_instance]))
        for key in ("precision", "recall", "f1", "auc", "accuracy", "resim_error")
    }
    elapsed = time.perf_counter() - start

    print(
        f"[wm_sl] prior={args.sl_prior}  PR={means['precision']:.4f}  "
        f"RE={means['recall']:.4f}  F1={means['f1']:.4f}  AUC={means['auc']:.4f}  "
        f"resim_error={means['resim_error']:.5f}  "
        f"({elapsed / max(1, len(instances)):.2f}s per instance)"
    )

    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps(
                {
                    "config": vars(args),
                    "wm_config": config,
                    "graph_id": graph_id,
                    "metrics": means,
                    "per_instance": per_instance,
                    "seconds": elapsed,
                },
                indent=2,
                default=str,
            )
        )
        print(f"[wm_sl] results -> {args.out_json}")
