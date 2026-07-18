"""
Autoregressive action-conditioned world-model training (teacher-forced one-step).

python world_model/train_wm.py \
    --data-dir data/output/er_node --diffusion-model IC \
    --model gcn --hidden-dim 64 --n-layers 3 \
    --epochs 300 --lr 1e-3 --weight-decay 5e-4 --batch-size 16 \
    --pos-weight auto --patience 40 --seed 42 \
    --device cuda --plan-demo \
    --ckpt-dir world_model/checkpoints \
    --results world_model/checkpoints/er_node_gcn_IC.json
"""

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from world_model.wm_data import TransitionDataset, collate_transitions, in_channels
from world_model.wm_model import WorldModel, backbones
from world_model.wm_eval import (
    evaluate_one_step,
    planning_regret_multi,
    rollout_ensemble,
)

pos_weight_min = 1.0
pos_weight_max = 50.0


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
        default=str(Path(__file__).resolve().parent / "checkpoints"),
        help="checkpoint directory (default: world_model/checkpoints).",
    )
    parser.add_argument(
        "--results", type=str, default=None, help="results JSON path (default: None)."
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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    diffusion_model = args.diffusion_model

    train_dataset = TransitionDataset(args.data_dir, diffusion_model, "train")
    validation_dataset = TransitionDataset(args.data_dir, diffusion_model, "val")
    test_dataset = TransitionDataset(args.data_dir, diffusion_model, "test")

    collate_fn = partial(
        collate_transitions, diffusion_model=diffusion_model, device=device
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )

    backbone_kwargs = {
        "n_heads": args.n_heads,
        "ffn_dim": args.ffn_dim,
        "alpha": args.gcnii_alpha,
        "lamda": args.gcnii_lamda,
    }

    model = WorldModel(
        args.model,
        in_channels=in_channels,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        head_type=args.head,
        diffusion_model=diffusion_model,
        **backbone_kwargs,
    ).to(device)

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if args.pos_weight == "auto":
        pos_weight_infected, pos_weight_frontier = compute_pos_weight(
            train_dataset, device
        )
    else:
        pos_weight_infected = pos_weight_frontier = None

    infected_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_infected)
    frontier_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_frontier)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    checkpoint_path = Path(args.ckpt_dir) / f"wm_{args.model}_{diffusion_model}.pt"
    best_delta_f1, epochs_since_best = -1.0, 0
    train_start = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"epoch {epoch}")

        for batch in progress_bar:
            optimizer.zero_grad()

            logits = model(batch["X"], batch["graph"])
            loss = infected_loss(logits[:, 0], batch["y_inf"]) + frontier_loss(
                logits[:, 1], batch["y_fr"]
            )

            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            progress_bar.set_postfix(loss=f"{loss_value:.6f}")

        val_metrics = evaluate_one_step(
            model, validation_dataset, diffusion_model, device
        )

        if val_metrics["delta_f1"] > best_delta_f1:
            best_delta_f1, epochs_since_best = val_metrics["delta_f1"], 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(
                    f"[early-stop] epoch {epoch}, best val delta_f1={best_delta_f1:.4f}"
                )
                break

    train_seconds = time.perf_counter() - train_start
    print(f"[train] total training time: {train_seconds:.1f}s")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    results = {
        "config": vars(args),
        "train_seconds": train_seconds,
        "test": evaluate_one_step(model, test_dataset, diffusion_model, device),
    }
    results["rollout"] = rollout_ensemble(
        model,
        args.data_dir,
        diffusion_model,
        train_dataset.store,
        device,
        "test",
        seed=args.seed,
    )

    if args.plan_demo:
        results["planning"] = planning_regret_multi(
            model,
            train_dataset.store,
            diffusion_model,
            device,
            n_graphs=args.plan_graphs,
            seed=args.seed,
        )

    results_path = args.results or str(
        Path(args.ckpt_dir) / f"results_{args.model}_{diffusion_model}.json"
    )
    os.makedirs(Path(results_path).parent, exist_ok=True)
    Path(results_path).write_text(json.dumps(results, indent=2, default=str))

    print(
        json.dumps(
            {key: results[key] for key in ("test", "rollout") if key in results},
            indent=2,
            default=str,
        )
    )
