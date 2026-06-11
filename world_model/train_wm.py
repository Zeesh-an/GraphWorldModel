"""
Autoregressive action-conditioned world-model training (teacher-forced one-step).

python world_model/train_wm.py --data-dir data/output/er_node --diffusion-model IC \
    --model gcn --epochs 200 --lr 1e-3 --plan-demo
"""

import argparse, json, sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

THIS = Path(__file__).resolve().parent
ROOT = THIS.parent
for _p in (str(THIS), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from wm_data import TransitionDataset, collate_transitions, IN_CHANNELS
from wm_model import WorldModel, BACKBONES
from wm_eval import evaluate_one_step, rollout_episodes, planning_regret


def compute_pos_weight(
    dataset: TransitionDataset, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    pos_i = tot = pos_f = 0
    for i in range(len(dataset)):
        it = dataset[i]
        tot += it["y_inf"].numel()
        pos_i += it["y_inf"].sum().item()
        pos_f += it["y_fr"].sum().item()
    wi = (tot - pos_i) / max(pos_i, 1.0)
    wf = (tot - pos_f) / max(pos_f, 1.0)
    clamp = lambda x: float(min(max(x, 1.0), 50.0))
    return torch.tensor([clamp(wi)], device=device), torch.tensor(
        [clamp(wf)], device=device
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Action-conditioned world-model training")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--diffusion-model", default="IC", choices=["IC", "LT"])
    p.add_argument("--model", default="gcn", choices=list(BACKBONES))
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--ffn-dim", type=int, default=128)
    p.add_argument("--gcnii-alpha", type=float, default=0.1)
    p.add_argument("--gcnii-lamda", type=float, default=0.5)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--pos-weight", default="auto", choices=["auto", "off"])
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ckpt-dir", default=str(THIS / "checkpoints"))
    p.add_argument("--results", default=None)
    p.add_argument("--plan-demo", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    dm = args.diffusion_model

    train_ds = TransitionDataset(args.data_dir, dm, "train")
    val_ds = TransitionDataset(args.data_dir, dm, "val")
    test_ds = TransitionDataset(args.data_dir, dm, "test")
    collate = partial(collate_transitions, diffusion_model=dm, device=device)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate
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
        **bb,
    ).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if args.pos_weight == "auto":
        pw_i, pw_f = compute_pos_weight(train_ds, device)
    else:
        pw_i = pw_f = None
    loss_i = nn.BCEWithLogitsLoss(pos_weight=pw_i)
    loss_f = nn.BCEWithLogitsLoss(pos_weight=pw_f)

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.ckpt_dir) / f"wm_{args.model}_{dm}.pt"
    best, bad = -1.0, 0
    for ep in range(args.epochs):
        model.train()
        bar = tqdm(train_dl, desc=f"epoch {ep}")
        for b in bar:
            opt.zero_grad()
            logits = model(b["X"], b["graph"])
            loss = loss_i(logits[:, 0], b["y_inf"]) + loss_f(logits[:, 1], b["y_fr"])
            loss.backward()
            opt.step()
            bar.set_postfix(loss=f"{loss.item():.4f}")
        val = evaluate_one_step(model, val_ds, dm, device)
        if val["delta_f1"] > best:
            best, bad = val["delta_f1"], 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[early-stop] epoch {ep}, best val delta_f1={best:.4f}")
                break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    results = {
        "config": vars(args),
        "test": evaluate_one_step(model, test_ds, dm, device),
    }
    results["rollout"] = rollout_episodes(
        model, args.data_dir, dm, train_ds.store, device, "test"
    )
    if args.plan_demo:
        gid = next(iter(train_ds.store))
        results["planning"] = planning_regret(
            model, train_ds.store[gid], dm, device, seed=args.seed
        )
    out = args.results or str(Path(args.ckpt_dir) / f"results_{args.model}_{dm}.json")
    Path(out).write_text(json.dumps(results, indent=2, default=str))
    print(
        json.dumps(
            {k: results[k] for k in ("test", "rollout") if k in results},
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
