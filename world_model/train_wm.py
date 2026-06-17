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
from functools import partial
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from wm_data import TransitionDataset, collate_transitions, IN_CHANNELS
from wm_model import WorldModel, BACKBONES
from wm_eval import evaluate_one_step, rollout_ensemble, planning_regret_multi


def compute_pos_weight(
    dataset: TransitionDataset, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    # Diffusion changes are sparse (few infected/frontier nodes per step), so a plain BCE would collapse to predict all zeros
    pos_i = tot = pos_f = 0

    for i in range(len(dataset)):
        it = dataset[i]

        tot += it["y_inf"].numel()
        pos_i += it["y_inf"].sum().item()
        pos_f += it["y_fr"].sum().item()

    wi = (tot - pos_i) / max(pos_i, 1.0)
    wf = (tot - pos_f) / max(pos_f, 1.0)
    clamp = lambda x: float(min(max(x, 1.0), 50.0))

    # Passed into BCEWithLogitsLoss, it up-weights the rare positive class so the model is pushed to actually predict the new infections
    return torch.tensor([clamp(wi)], device=device), torch.tensor(
        [clamp(wf)], device=device
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Action-conditioned world-model training"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--diffusion-model", default="IC", choices=["IC", "LT"])
    parser.add_argument("--model", default="gcn", choices=list(BACKBONES))
    parser.add_argument("--head", default="linear", choices=["linear", "structured"])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=128)
    parser.add_argument("--gcnii-alpha", type=float, default=0.1)
    parser.add_argument("--gcnii-lamda", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pos-weight", default="auto", choices=["auto", "off"])
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--ckpt-dir", default=str(Path(__file__).resolve().parent / "checkpoints")
    )
    parser.add_argument("--results", default=None)
    parser.add_argument("--plan-demo", action="store_true")
    parser.add_argument("--plan-graphs", type=int, default=5)

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

    bb = {
        "n_heads": args.n_heads,
        "ffn_dim": args.ffn_dim,
        "alpha": args.gcnii_alpha,
        "lamda": args.gcnii_lamda,
    }

    model = WorldModel(
        args.model,
        in_channels=IN_CHANNELS,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        head_type=args.head,
        **bb,
    ).to(device)

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if args.pos_weight == "auto":
        pw_i, pw_f = compute_pos_weight(train_dataset, device)
    else:
        pw_i = pw_f = None

    loss_i = nn.BCEWithLogitsLoss(pos_weight=pw_i)
    loss_f = nn.BCEWithLogitsLoss(pos_weight=pw_f)

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.ckpt_dir) / f"wm_{args.model}_{diffusion_model}.pt"
    best, bad = -1.0, 0

    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"epoch {epoch}")

        for batch in progress_bar:
            optimizer.zero_grad()

            logits = model(batch["X"], batch["graph"])
            loss = loss_i(logits[:, 0], batch["y_inf"]) + loss_f(
                logits[:, 1], batch["y_fr"]
            )

            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        val = evaluate_one_step(model, validation_dataset, diffusion_model, device)

        if val["delta_f1"] > best:
            best, bad = val["delta_f1"], 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[early-stop] epoch {epoch}, best val delta_f1={best:.4f}")
                break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    results = {
        "config": vars(args),
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

    out = args.results or str(
        Path(args.ckpt_dir) / f"results_{args.model}_{diffusion_model}.json"
    )
    Path(out).write_text(json.dumps(results, indent=2, default=str))

    print(
        json.dumps(
            {k: results[k] for k in ("test", "rollout") if k in results},
            indent=2,
            default=str,
        )
    )
