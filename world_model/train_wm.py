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

from data.wm_simulator import spent, valid_remove_semantics
from world_model.wm_data import TransitionDataset, collate_transitions, in_channels
from world_model.wm_model import WorldModel, backbones
from world_model.wm_eval import (
    evaluate_one_step,
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
) -> tuple[torch.Tensor, torch.Tensor]:
    # Diffusion changes are sparse (few infected/frontier nodes per step), so a plain BCE would collapse to predict all zeros
    positive_infected = total = positive_frontier = 0

    for index in range(len(dataset)):
        item = dataset[index]

        total += item["y_inf"].numel()
        positive_infected += item["y_inf"].sum().item()
        positive_frontier += item["y_fr"].sum().item()

    weight_infected = (total - positive_infected) / max(positive_infected, 1.0)
    weight_frontier = (total - positive_frontier) / max(positive_frontier, 1.0)

    # Passed into BCEWithLogitsLoss, it up-weights the rare positive class so the model is pushed to actually predict the new infections
    return (
        torch.tensor([_clamp_pos_weight(weight_infected)], device=device),
        torch.tensor([_clamp_pos_weight(weight_frontier)], device=device),
    )


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
    check_hide_edge_weights(config)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device(config.device)
    diffusion_model = config.diffusion_model

    train_dataset = TransitionDataset(config.data_dir, diffusion_model, "train")
    validation_dataset = TransitionDataset(config.data_dir, diffusion_model, "val")
    test_dataset = TransitionDataset(config.data_dir, diffusion_model, "test")

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

    model = WorldModel(
        config.model,
        in_channels=in_channels,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        dropout=config.dropout,
        head_type=config.head,
        diffusion_model=diffusion_model,
        remove_semantics=config.remove_semantics,
        **backbone_kwargs,
    ).to(device)

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    if config.pos_weight == "auto":
        pos_weight_infected, pos_weight_frontier = compute_pos_weight(
            train_dataset, device
        )
    else:
        pos_weight_infected = pos_weight_frontier = None

    infected_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_infected)
    frontier_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_frontier)

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
            loss = infected_loss(logits[:, 0], batch["y_inf"]) + frontier_loss(
                logits[:, 1], batch["y_fr"]
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
        ),
    }
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
        results["planning"] = planning_regret_multi(
            model,
            train_dataset.store,
            diffusion_model,
            device,
            n_graphs=config.plan_graphs,
            seed=config.seed,
            hide_edge_weights=config.hide_edge_weights,
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
        choices=["IC", "LT"],
        help="diffusion model to train on (default: IC).",
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
