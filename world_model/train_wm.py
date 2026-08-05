"""
Autoregressive action-conditioned world-model training (teacher-forced one-step).

Checkpoints and the results JSON land next to the data by default:
<run>/data -> <run>/world_model/{wm_<model>_<dm>.pt, <model>_<dm>.json}, where
<run> is results/<task>/<dataset>/<run> (see pipeline/layout.py)

python -m world_model.train_wm \
    --data-dir results/ba40/data --diffusion-model IC \
    --model sage --head structured_residual --hidden-dim 64 --n-layers 3 \
    --epochs 400 --lr 1e-3 --weight-decay 5e-4 --batch-size 32 \
    --pos-weight off --patience 50 --seed 42 \
    --device cuda --plan-demo
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.wm_competitive import auto_dominance
from data.wm_simulator import epidemic_dynamics, spent, valid_remove_semantics
from world_model.wm_data import (
    TransitionDataset,
    channels_for,
    collate_transitions,
    dataset_is_competitive,
    dataset_is_epidemic,
    epidemic_rates,
)
from world_model.wm_model import WorldModel, backbones
from data.wm_competitive import CompetitiveConfig, shared_positive_prob
from data.wm_epidemic import EpidemicConfig, default_burn_in
from world_model.wm_eval import (
    blocking_regret_multi,
    competitive_rollout_ensemble,
    epidemic_rollout_ensemble,
    evaluate_one_step,
    immunization_regret_multi,
    planning_regret_multi,
    rollout_ensemble,
)

pos_weight_min = 1.0
pos_weight_max = 50.0
# Flush the training curve to disk this often so a killed job keeps its history
history_flush_epochs = 5


@dataclass
class TrainConfig:
    """Field names are the results-JSON `config` schema that world_model_env reads back."""

    data_dir: str
    diffusion_model: str = "IC"
    model: str = "gcn"
    head: str = "linear"
    # Must match the dataset's; cross-checked against metadata.json below
    remove_semantics: str = spent
    # Two-cascade (influence blocking) training: 8 input channels, 4 targets, and a
    # competitive head. None = read from the dataset's own metadata, which is what
    # keeps a 6-channel head from being fit on a competitive dataset and quietly
    # reporting the negative cascade alone. The two dynamics parameters below are
    # likewise defaulted FROM the data, because a head whose tie-break disagrees with
    # the simulator that made the targets is fit against a transition that never
    # happened (research/influence_blocking.md §8.4).
    competitive: bool | None = None
    tie_break: str = auto_dominance
    positive_prob: float | None = None
    # Compartmental (epidemic control) training: 9 input channels, 5 targets, and
    # the compartment head. None = read from the dataset's own metadata, for the
    # same reason `competitive` is: a 6-channel head fed compartmental features
    # would silently fit the attack set alone and lose recovery entirely, which is
    # the one thing this task exists to test. The three rates are likewise defaulted
    # FROM the data, because a head whose gamma disagrees with the simulator that
    # made the targets is fit against a transition that never happened.
    epidemic: bool | None = None
    beta_scale: float = 1.0
    gamma: float | None = None
    alpha: float | None = None
    burn_in: float = default_burn_in
    # Feed ones instead of p(u->v) to the encoder and the head: the online/bandit
    # information state, and the ablation for "our IC heads see the true w"
    hide_edge_weights: bool = False
    hidden_dim: int = 64
    n_layers: int = 3
    n_heads: int = 4
    ffn_dim: int = 128
    gcnii_alpha: float = 0.1
    gcnii_lamda: float = 0.5
    dropout: float = 0.1
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 5e-4
    batch_size: int = 16
    pos_weight: str = "auto"
    patience: int = 30
    seed: int = 42
    device: str = "cpu"
    ckpt_dir: str | None = None
    results: str | None = None
    plan_demo: bool = False
    plan_graphs: int = 5


def _clamp_pos_weight(value: float) -> float:
    return float(min(max(value, pos_weight_min), pos_weight_max))


def compute_pos_weight(
    dataset: TransitionDataset, device: torch.device
) -> list[torch.Tensor]:
    """
    One positive-class weight per target column — 2 single-cascade, 4 competitive.

    Diffusion changes are sparse (few infected/frontier nodes per step), so a plain
    BCE would collapse to predicting all zeros. Blocking makes that WORSE rather than
    better: §5.3 shows blocking 50 YouTube nodes prevents 0.29% of the spread, so the
    positive class in the columns that matter is rarer than it is under IM, which is
    why the calibration columns are worth re-checking rather than assuming they carry
    over from the seeding tasks.
    """
    columns = dataset[0]["y"].shape[1]
    positives = [0.0] * columns
    total = 0

    for index in range(len(dataset)):
        targets = dataset[index]["y"]
        total += targets.shape[0]

        for column in range(columns):
            positives[column] += targets[:, column].sum().item()

    # Passed into BCEWithLogitsLoss, it up-weights the rare positive class so the
    # model is pushed to actually predict the new infections
    return [
        torch.tensor(
            [_clamp_pos_weight((total - positive) / max(positive, 1.0))], device=device
        )
        for positive in positives
    ]


def resolve_paths(config: TrainConfig) -> TrainConfig:
    """Default the checkpoint dir and results JSON to siblings of the data dir."""
    if config.ckpt_dir is None:
        config.ckpt_dir = str(Path(config.data_dir).resolve().parent / "world_model")

    if config.results is None:
        config.results = str(
            Path(config.ckpt_dir) / f"{config.model}_{config.diffusion_model}.json"
        )

    return config


def check_remove_semantics(config: TrainConfig) -> None:
    """
    Refuse to train a head whose T_exo disagrees with the data that produced the
    targets. The mismatch is otherwise silent: the model simply never fits the
    removal transitions, and the containment numbers come out biased with no
    error anywhere.
    """
    metadata_path = Path(config.data_dir) / "metadata.json"
    if not metadata_path.exists():
        return

    # Datasets generated before --remove-semantics existed are all `spent`
    dataset_semantics = json.loads(metadata_path.read_text())["config"].get(
        "remove_semantics", spent
    )

    if dataset_semantics != config.remove_semantics:
        raise ValueError(
            f"--remove-semantics {config.remove_semantics!r} does not match the "
            f"dataset at {config.data_dir}, which was generated with "
            f"{dataset_semantics!r} (see {metadata_path}). Regenerate the data or "
            f"pass --remove-semantics {dataset_semantics}."
        )


def resolve_competitive(config: TrainConfig) -> TrainConfig:
    """
    Fill `competitive` / `tie_break` / `positive_prob` from the dataset that made the targets.

    Defaulted from the data rather than from a flag for the same reason
    `check_remove_semantics` refuses a mismatch: a competitive head whose tie-break
    or `p_L` disagrees with the simulator is fit against a transition that never
    happened, and nothing about the loss curve would say so.
    """
    metadata_path = Path(config.data_dir) / "metadata.json"
    dataset_competitive = dataset_is_competitive(config.data_dir)

    if config.competitive is None:
        config.competitive = dataset_competitive
    elif config.competitive != dataset_competitive:
        raise ValueError(
            f"--competitive {config.competitive} does not match the dataset at "
            f"{config.data_dir}, which is competitive={dataset_competitive}. A "
            f"competitive dataset carries 8 channels and 4 targets; regenerate with "
            f"data/generate_wm_data.py --competitive or drop the flag."
        )

    if not config.competitive or not metadata_path.exists():
        return config

    competitive = json.loads(metadata_path.read_text()).get("competitive", {})
    resolved = competitive.get(config.diffusion_model)

    if resolved is None:
        raise ValueError(
            f"the dataset at {config.data_dir} carries no competitive metadata for "
            f"--diffusion-model {config.diffusion_model}; it was generated for "
            f"{sorted(key for key in competitive if key != 'negative_pct')}"
        )

    config.tie_break = resolved["tie_break"]
    config.positive_prob = (
        None
        if resolved["positive_prob"] == "shared"
        else float(resolved["positive_prob"])
    )
    print(
        f"[train] competitive: {resolved['competitive_model']}, "
        f"tie_break={config.tie_break}, positive_prob={resolved['positive_prob']}"
    )

    return config


def resolve_epidemic(config: TrainConfig) -> TrainConfig:
    """
    Fill `epidemic` / `beta_scale` / `gamma` / `alpha` from the dataset that made the targets.

    Same rule and same reason as `resolve_competitive`: the rates are not a choice
    at training time, they are a property of the transitions on disk. Getting them
    from a flag would let `--head structured_oracle` pin its matrix to a recovery
    rate the simulator never used and report a fidelity number that means nothing.
    """
    dataset_epidemic = dataset_is_epidemic(config.data_dir)

    if config.epidemic is None:
        config.epidemic = dataset_epidemic
    elif config.epidemic != dataset_epidemic:
        raise ValueError(
            f"--epidemic {config.epidemic} does not match the dataset at "
            f"{config.data_dir}, which is epidemic={dataset_epidemic}. A "
            f"compartmental dataset carries 9 channels and 5 targets; regenerate "
            f"with data/generate_wm_data.py --models SIR (or SIS / SEIR)."
        )

    if not config.epidemic:
        return config

    resolved = epidemic_rates(config.data_dir, config.diffusion_model)
    config.beta_scale = float(resolved["beta_scale"])
    config.gamma = float(resolved["gamma"])
    config.alpha = None if resolved["alpha"] is None else float(resolved["alpha"])
    config.burn_in = float(resolved["burn_in"])
    print(
        f"[train] compartmental: {'/'.join(resolved['compartments'])}, "
        f"beta_scale={config.beta_scale}, gamma={config.gamma}, "
        f"alpha={config.alpha}"
    )

    return config


def check_hide_edge_weights(config: TrainConfig) -> None:
    """
    Both anchored heads read w directly, so masking it is not an ablation of them
    but a corruption of them: structured_residual anchors q on logit(w), which at
    w=1 pins every edge at q~1 and saturates the rollout, and structured_oracle
    IS q = w. Only the plain structured head (and linear) learn q from scratch.
    """
    if not config.hide_edge_weights:
        return

    if config.head in ("structured_residual", "structured_oracle"):
        raise ValueError(
            f"--hide-edge-weights is incompatible with --head {config.head}: that "
            f"head reads the true transmission probability directly, so masking it "
            f"to ones does not hide information, it feeds a wrong anchor. Use "
            f"--head structured for the w-hidden (bandit information state) run."
        )


def train_world_model(config: TrainConfig) -> dict:
    config = resolve_paths(config)
    check_remove_semantics(config)
    config = resolve_competitive(config)
    config = resolve_epidemic(config)
    check_hide_edge_weights(config)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device(config.device)
    diffusion_model = config.diffusion_model

    layout = dict(competitive=config.competitive, epidemic=config.epidemic)
    train_dataset = TransitionDataset(
        config.data_dir, diffusion_model, "train", **layout
    )
    validation_dataset = TransitionDataset(
        config.data_dir, diffusion_model, "val", **layout
    )
    test_dataset = TransitionDataset(config.data_dir, diffusion_model, "test", **layout)

    collate_fn = partial(
        collate_transitions,
        diffusion_model=diffusion_model,
        device=device,
        hide_edge_weights=config.hide_edge_weights,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    backbone_kwargs = {
        "n_heads": config.n_heads,
        "ffn_dim": config.ffn_dim,
        "alpha": config.gcnii_alpha,
        "lamda": config.gcnii_lamda,
    }

    model_in_channels, model_out_channels = channels_for(
        config.competitive, config.epidemic
    )
    model = WorldModel(
        config.model,
        in_channels=model_in_channels,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        dropout=config.dropout,
        head_type=config.head,
        diffusion_model=diffusion_model,
        remove_semantics=config.remove_semantics,
        competitive=config.competitive,
        tie_break=config.tie_break,
        positive_prob=config.positive_prob,
        epidemic=config.epidemic,
        epi_beta=config.beta_scale,
        epi_gamma=config.gamma,
        epi_alpha=config.alpha,
        **backbone_kwargs,
    ).to(device)

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    weights = (
        compute_pos_weight(train_dataset, device)
        if config.pos_weight == "auto"
        else [None] * model_out_channels
    )
    # One BCE term per target column: 2 single-cascade, 4 competitive (§2.5)
    losses = [nn.BCEWithLogitsLoss(pos_weight=weight) for weight in weights]

    os.makedirs(config.ckpt_dir, exist_ok=True)
    checkpoint_path = (
        Path(config.ckpt_dir) / f"wm_{config.model}_{diffusion_model}.pt"
    )
    best_delta_f1, epochs_since_best = -1.0, 0
    train_start = time.perf_counter()

    # Per-epoch curve for the training-diagnostics plot, flushed as it grows
    history = []
    history_path = Path(config.ckpt_dir) / f"history_{config.model}_{diffusion_model}.json"

    print(
        f"[train] {config.model}/{config.head} on {diffusion_model}: "
        f"{len(train_dataset)} train / {len(validation_dataset)} val / "
        f"{len(test_dataset)} test transitions, up to {config.epochs} epochs "
        f"(patience {config.patience}) on {device}"
    )

    progress_bar = tqdm(range(config.epochs), desc=f"train {config.model}/{diffusion_model}")
    for epoch in progress_bar:
        model.train()
        batch_bar = tqdm(train_dataloader, desc=f"  epoch {epoch}", leave=False)
        epoch_loss, n_batches = 0.0, 0

        for batch in batch_bar:
            optimizer.zero_grad()

            logits = model(batch["X"], batch["graph"])
            loss = sum(
                term(logits[:, column], batch["y"][:, column])
                for column, term in enumerate(losses)
            )

            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            epoch_loss += loss_value
            n_batches += 1
            batch_bar.set_postfix(loss=f"{loss_value:.6f}")

        val_metrics = evaluate_one_step(
            model,
            validation_dataset,
            diffusion_model,
            device,
            hide_edge_weights=config.hide_edge_weights,
            competitive=config.competitive,
            epidemic=config.epidemic,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / max(n_batches, 1),
                "val_delta_f1": val_metrics["delta_f1"],
                "val_new_infection_f1": val_metrics["new_infection_f1"],
            }
        )

        # Persist the curve as it is produced: a crash or a SLURM timeout at
        # epoch 380/400 otherwise loses every epoch of history
        if epoch % history_flush_epochs == 0 or epoch == config.epochs - 1:
            Path(history_path).write_text(json.dumps(history, indent=2))

        progress_bar.set_postfix(
            val_delta_f1=f"{val_metrics['delta_f1']:.4f}",
            best=f"{max(best_delta_f1, val_metrics['delta_f1']):.4f}",
        )

        if val_metrics["delta_f1"] > best_delta_f1:
            best_delta_f1, epochs_since_best = val_metrics["delta_f1"], 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_since_best += 1
            if epochs_since_best >= config.patience:
                print(
                    f"[early-stop] epoch {epoch}, best val delta_f1={best_delta_f1:.4f}"
                )
                break

    train_seconds = time.perf_counter() - train_start
    print(f"[train] total training time: {train_seconds:.1f}s")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    results = {
        "config": vars(config),
        "train_seconds": train_seconds,
        "best_val_delta_f1": best_delta_f1,
        "history": history,
        "test": evaluate_one_step(
            model,
            test_dataset,
            diffusion_model,
            device,
            hide_edge_weights=config.hide_edge_weights,
            competitive=config.competitive,
            epidemic=config.epidemic,
        ),
    }

    # Both halves fork on the same flag, and both forks measure the SAME quantity —
    # the negative cascade — so the two tasks' rollout and planning numbers sit in
    # one report column without being silently different things
    competitive_config = (
        CompetitiveConfig(
            tie_break=config.tie_break,
            positive_prob=(
                shared_positive_prob
                if config.positive_prob is None
                else config.positive_prob
            ),
            remove_semantics=config.remove_semantics,
        )
        if config.competitive
        else None
    )

    epidemic_config = (
        EpidemicConfig(
            beta_scale=config.beta_scale,
            gamma=config.gamma,
            alpha=config.alpha if config.alpha is not None else 0.5,
            remove_semantics=config.remove_semantics,
            burn_in=config.burn_in,
        )
        if config.epidemic
        else None
    )

    if config.epidemic:
        results["rollout"] = epidemic_rollout_ensemble(
            model,
            config.data_dir,
            diffusion_model,
            train_dataset.store,
            device,
            "test",
            seed=config.seed,
            hide_edge_weights=config.hide_edge_weights,
            config=epidemic_config,
        )
    elif config.competitive:
        results["rollout"] = competitive_rollout_ensemble(
            model,
            config.data_dir,
            diffusion_model,
            train_dataset.store,
            device,
            "test",
            seed=config.seed,
            hide_edge_weights=config.hide_edge_weights,
            config=competitive_config,
        )
    else:
        results["rollout"] = rollout_ensemble(
            model,
            config.data_dir,
            diffusion_model,
            train_dataset.store,
            device,
            "test",
            seed=config.seed,
            remove_semantics=config.remove_semantics,
            hide_edge_weights=config.hide_edge_weights,
        )

    if config.plan_demo:
        planner = (
            immunization_regret_multi
            if config.epidemic
            else blocking_regret_multi
            if config.competitive
            else planning_regret_multi
        )
        extra = (
            {"config": epidemic_config}
            if config.epidemic
            else {"config": competitive_config}
            if config.competitive
            else {}
        )
        results["planning"] = planner(
            model,
            train_dataset.store,
            diffusion_model,
            device,
            n_graphs=config.plan_graphs,
            seed=config.seed,
            hide_edge_weights=config.hide_edge_weights,
            **extra,
        )

    os.makedirs(Path(config.results).parent, exist_ok=True)
    Path(config.results).write_text(json.dumps(results, indent=2, default=str))
    print(f"[train] checkpoint -> {checkpoint_path}")
    print(f"[train] results -> {config.results}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Action-conditioned world-model training"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="generated dataset directory (default: required).",
    )
    parser.add_argument(
        "--diffusion-model",
        type=str,
        default="IC",
        choices=["IC", "LT"] + list(epidemic_dynamics),
        help="dynamics to train on. SIR/SIS/SEIR select the compartment head and "
        "read their rates back from the dataset's metadata (default: IC).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gcn",
        choices=list(backbones),
        help="encoder backbone (default: gcn).",
    )
    parser.add_argument(
        "--head",
        type=str,
        default="linear",
        choices=["linear", "structured", "structured_residual"],
        help="output head type; structured_residual anchors IC transmission on the true edge prob and learns only a correction (default: linear).",
    )
    parser.add_argument(
        "--remove-semantics",
        type=str,
        default=spent,
        choices=list(valid_remove_semantics),
        help="what remove_node means; must match the dataset's, which is checked "
        "against its metadata.json (default: spent).",
    )
    parser.add_argument(
        "--hide-edge-weights",
        action="store_true",
        help="feed ones instead of the true IC transmission probability to the "
        "encoder and head: the online/bandit information state, and the ablation "
        "for the IC heads otherwise seeing w. Requires --head structured or "
        "linear (default: False).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="hidden dimension (default: 64).",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=3,
        help="number of encoder layers (default: 3).",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=4,
        help="attention heads for GAT/GT (default: 4).",
    )
    parser.add_argument(
        "--ffn-dim",
        type=int,
        default=128,
        help="transformer feed-forward dimension (default: 128).",
    )
    parser.add_argument(
        "--gcnii-alpha",
        type=float,
        default=0.1,
        help="GCNII initial residual alpha (default: 0.1).",
    )
    parser.add_argument(
        "--gcnii-lamda",
        type=float,
        default=0.5,
        help="GCNII identity mapping lambda (default: 0.5).",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="dropout probability (default: 0.1)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="maximum training epochs (default: 200).",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Adam learning rate (default: 1e-3)."
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
        help="Adam weight decay (default: 5e-4).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="transition batch size (default: 16).",
    )
    parser.add_argument(
        "--pos-weight",
        type=str,
        default="auto",
        choices=["auto", "off"],
        help="positive-class weighting mode (default: auto).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="early-stop patience in epochs (default: 30).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)."
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="torch device string (default: cpu)."
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="checkpoint directory (default: <data-dir>/../world_model).",
    )
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="results JSON path (default: <ckpt-dir>/<model>_<diffusion-model>.json).",
    )
    parser.add_argument(
        "--plan-demo",
        action="store_true",
        help="run planning demo after training (default: False).",
    )
    parser.add_argument(
        "--plan-graphs",
        type=int,
        default=5,
        help="graphs used for planning demo (default: 5).",
    )

    args = parser.parse_args()
    results = train_world_model(TrainConfig(**vars(args)))

    print(
        json.dumps(
            {key: results[key] for key in ("test", "rollout") if key in results},
            indent=2,
            default=str,
        )
    )
