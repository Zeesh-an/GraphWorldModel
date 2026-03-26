# -*- coding: utf-8 -*-
"""
Graph Transformer — Influence Maximization Training
=====================================================
Mirrors the baseline genim.py structure exactly:

  Phase 1 (--epochs,  default 600):
    Train VAE (Encoder + Decoder) + GraphTransformerForwardModel jointly.
    Loss = BCE(x_hat, x) + MSE(y_hat, y)

  Phase 2 (--opt-iters, default 300):
    Freeze both models. Optimise latent z directly via backprop.
    Loss = MSE(ones, y_hat) + L0 sparsity on x_hat
    Pick top-k nodes from x_hat as the predicted seed set.

Usage
-----
  python World_model/train.py -d cora_ml -dm IC -sp 1
  python World_model/train.py -d cora_ml -dm LT -sp 1 --epochs 300 --opt-iters 150
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

# ── local imports ──────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(THIS_DIR))

from model.vae               import Encoder, Decoder, VAEModel
from model.graph_transformer import GraphTransformerForwardModel
from utils                   import (
    load_data, InverseProblemDataset,
    adj_process, top_diffusion_sampling, diffusion_evaluation,
)


# ─────────────────────────────────────────────
# CLI  (same flags as baseline genim.py)
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GT-IM: Graph Transformer Influence Maximization")

    p.add_argument("-d",  "--dataset",
                   default="cora_ml",
                   choices=["jazz", "cora_ml", "power_grid", "netscience", "random5"],
                   help="Dataset name")
    p.add_argument("-dm", "--diffusion_model",
                   default="IC",
                   choices=["IC", "LT", "SIS"],
                   help="Diffusion model")
    p.add_argument("-sp", "--seed_rate",
                   default=1, type=int,
                   choices=[1, 5, 10, 20],
                   help="Seed rate (× 10 = seed set size percentage)")
    p.add_argument("-m",  "--mode",
                   default="normal",
                   choices=["normal", "budget_constraint"],
                   help="Evaluation mode")

    # Training hyper-params
    p.add_argument("--epochs",    default=600, type=int,  help="Phase 1 training epochs")
    p.add_argument("--opt-iters", default=300, type=int,  help="Phase 2 latent optimisation iters")
    p.add_argument("--lr",        default=1e-4, type=float)
    p.add_argument("--lr-z",      default=1e-4, type=float, help="Phase 2 latent z learning rate")

    # Model dims
    p.add_argument("--hidden-dim", default=1024, type=int)
    p.add_argument("--latent-dim", default=512,  type=int)
    p.add_argument("--gt-d-model", default=64,   type=int, help="GT hidden dim")
    p.add_argument("--gt-heads",   default=4,    type=int, help="GT attention heads")
    p.add_argument("--gt-layers",  default=3,    type=int, help="Number of GT layers")
    p.add_argument("--gt-ffn",     default=128,  type=int, help="GT FFN hidden dim")
    p.add_argument("--gt-dropout", default=0.1,  type=float)

    # Data paths
    p.add_argument("--npz-dir",
                   default=str(ROOT / "Data" / "cora_ml"),
                   help="Path to generated .npz data directory")
    p.add_argument("--sg-dir",
                   default=str(ROOT / "Baselines" / "DeepIM" / "data"),
                   help="Path to baseline .SG data directory (fallback)")
    p.add_argument("--ckpt-dir",
                   default=str(ROOT / "World_model" / "checkpoints"),
                   help="Directory to save checkpoints")

    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


# ─────────────────────────────────────────────
# Loss functions  (identical to baseline)
# ─────────────────────────────────────────────

def loss_phase1(x, x_hat, y, y_hat):
    """
    Joint VAE + forward model loss.
    reproduction_loss = BCE(x_hat, x)   — how well VAE reconstructs seed
    forward_loss      = MSE(y_hat, y)   — how well GT predicts influence
    """
    reproduction_loss = F.binary_cross_entropy(x_hat, x, reduction="sum")
    forward_loss      = F.mse_loss(y_hat, y, reduction="sum")
    return reproduction_loss + forward_loss, reproduction_loss, forward_loss


def loss_phase2(y_true, y_hat, x_hat):
    """
    Latent optimisation loss.
    forward_loss = MSE(y_hat, ones)  — push influence to cover all nodes
    L0_loss      = L1 sparsity on x_hat  — keep seed set small
    """
    forward_loss = F.mse_loss(y_hat, y_true)
    L0_loss      = torch.sum(torch.abs(x_hat)) / x_hat.shape[1]
    return forward_loss + L0_loss, L0_loss


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ───────────────────────────────────────────────
    adj, inverse_pairs = load_data(
        dataset         = args.dataset,
        diffusion_model = args.diffusion_model,
        seed_rate       = args.seed_rate,
        npz_dir         = Path(args.npz_dir),
        sg_dir          = Path(args.sg_dir),
    )

    N = inverse_pairs.shape[1]
    print(f"[data] {args.dataset} | {args.diffusion_model} | "
          f"N={N} | samples={len(inverse_pairs)}")

    # Adjacency: symmetrise, normalise, to sparse COO
    adj_t = adj_process(adj).to(device)

    # Dataset / loaders  (same split logic as baseline)
    if args.dataset == "random5":
        batch_size  = 2
        hidden_dim  = 4096
        latent_dim  = 1024
    else:
        batch_size  = 16
        hidden_dim  = args.hidden_dim
        latent_dim  = args.latent_dim

    n_test  = min(batch_size, len(inverse_pairs) // 10)
    n_train = len(inverse_pairs) - n_test

    train_set, test_set = random_split(inverse_pairs, [n_train, n_test])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  drop_last=False)
    test_loader  = DataLoader(test_set,  batch_size=1,          shuffle=False)

    # ── Models ─────────────────────────────────────────────
    encoder = Encoder(input_dim=N, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(input_dim=latent_dim, latent_dim=latent_dim,
                      hidden_dim=hidden_dim, output_dim=N)

    vae_model      = VAEModel(encoder, decoder).to(device)
    forward_model  = GraphTransformerForwardModel(
        d_model  = args.gt_d_model,
        n_heads  = args.gt_heads,
        n_layers = args.gt_layers,
        ffn_dim  = args.gt_ffn,
        dropout  = args.gt_dropout,
    ).to(device)

    n_vae = sum(p.numel() for p in vae_model.parameters())
    n_gt  = sum(p.numel() for p in forward_model.parameters())
    print(f"[model] VAE params={n_vae:,}  |  GT params={n_gt:,}")

    optimizer = Adam(
        [{"params": vae_model.parameters()},
         {"params": forward_model.parameters()}],
        lr=args.lr,
    )

    # ════════════════════════════════════════════
    #  PHASE 1 — Joint training
    # ════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f" Phase 1 — Joint VAE + Graph Transformer training")
    print(f" Epochs: {args.epochs}  |  batch: {batch_size}")
    print(f"{'='*60}")

    vae_model.train()
    forward_model.train()
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        total_loss_ep   = 0.0
        forward_loss_ep = 0.0
        recon_loss_ep   = 0.0
        precision_re_ep = 0.0
        recall_re_ep    = 0.0
        n_seen          = 0

        for data_pair in train_loader:
            # data_pair: (B, N, 2)
            x = data_pair[:, :, 0].float().to(device)  # seed vectors
            y = data_pair[:, :, 1].float().to(device)  # influence vectors
            B = x.size(0)

            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)

            for i in range(B):
                x_i = x[i]                                       # (N,)
                y_i = y[i]                                       # (N,)

                x_hat = vae_model(x_i.unsqueeze(0))              # (1, N)
                # Forward model: (N, 1) → (N, 1)
                y_hat = forward_model(
                    x_hat.squeeze(0).unsqueeze(-1), adj_t
                ).squeeze(-1)                                     # (N,)

                total, recon, forw = loss_phase1(
                    x_i.unsqueeze(0), x_hat,
                    y_i.unsqueeze(0), y_hat.unsqueeze(0),
                )
                batch_loss += total

                # Reconstruction metrics
                x_pred_np = x_hat.detach().cpu().numpy()
                x_pred_np = (x_pred_np > 0.01).astype(float)
                x_true_np = x_i.unsqueeze(0).cpu().numpy()

                precision_re_ep += precision_score(x_true_np[0], x_pred_np[0], zero_division=0)
                recall_re_ep    += recall_score   (x_true_np[0], x_pred_np[0], zero_division=0)

                forward_loss_ep += forw.item()
                recon_loss_ep   += recon.item()

            total_loss_ep += batch_loss.item()
            batch_loss = batch_loss / B
            batch_loss.backward()
            optimizer.step()

            # Clamp GT params ≥ 0  (mirrors baseline's clamp for SpGAT)
            for p in forward_model.parameters():
                p.data.clamp_(min=0)

            n_seen += B

        elapsed = time.time() - t0
        avg = lambda v: v / max(n_seen, 1)

        print(
            f"Epoch {epoch:>4d}/{args.epochs}"
            f"  Total={avg(total_loss_ep):.4f}"
            f"  Recon={avg(recon_loss_ep):.4f}"
            f"  Fwd={avg(forward_loss_ep):.4f}"
            f"  Prec={avg(precision_re_ep):.4f}"
            f"  Rec={avg(recall_re_ep):.4f}"
            f"  t={elapsed:.2f}s"
        )

        # Checkpoint best model
        if avg(total_loss_ep) < best_loss:
            best_loss = avg(total_loss_ep)
            ckpt = ckpt_dir / f"best_{args.dataset}_{args.diffusion_model}.pt"
            torch.save({
                "epoch":         epoch,
                "vae":           vae_model.state_dict(),
                "forward_model": forward_model.state_dict(),
                "loss":          best_loss,
                "args":          vars(args),
            }, ckpt)

    print(f"\n[✓] Phase 1 done. Best loss={best_loss:.4f}")
    print(f"[✓] Checkpoint saved → {ckpt}")

    # ════════════════════════════════════════════
    #  PHASE 2 — Latent optimisation
    # ════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f" Phase 2 — Latent z optimisation ({args.opt_iters} iters)")
    print(f"{'='*60}")

    # Freeze both models
    for p in vae_model.parameters():
        p.requires_grad = False
    for p in forward_model.parameters():
        p.requires_grad = False
    vae_model.eval()
    forward_model.eval()

    # Initialise z as weighted mean of top-spreading samples' encodings
    topk_idx = top_diffusion_sampling(inverse_pairs, frac=0.1)
    z_hat    = torch.zeros(1, latent_dim, device=device)

    with torch.no_grad():
        for i in topk_idx:
            seed_vec_i = inverse_pairs[i, :, 0].float().unsqueeze(0).to(device)
            z_hat     += encoder(seed_vec_i)
    z_hat = z_hat / len(topk_idx)
    z_hat = z_hat.detach().requires_grad_(True)

    z_optimizer = Adam([z_hat], lr=args.lr_z)

    # Seed budget: estimated from mean reconstruction
    with torch.no_grad():
        x_init = decoder(z_hat)
    seed_num = max(1, int(x_init.sum().item()))
    y_target = torch.ones(1, N, device=device)    # want to influence all nodes

    print(f"[phase2] Estimated seed budget = {seed_num}")

    for i in range(1, args.opt_iters + 1):
        x_hat = vae_model.decoder(z_hat)                     # (1, N)
        y_hat = forward_model(
            x_hat.squeeze(0).unsqueeze(-1), adj_t
        ).squeeze(-1).unsqueeze(0)                            # (1, N)

        loss, L0 = loss_phase2(y_target, y_hat, x_hat)

        z_optimizer.zero_grad()
        loss.backward()
        z_optimizer.step()

        if i % 50 == 0 or i == 1:
            print(f"  Iter {i:>4d}/{args.opt_iters}"
                  f"  Loss={loss.item():.5f}"
                  f"  L0={L0.item():.5f}"
                  f"  PredSpread={y_hat.sum().item():.1f}")

    # ── Extract final seed set ───────────────────────────
    with torch.no_grad():
        x_final = vae_model.decoder(z_hat)

    top_k  = x_final.topk(seed_num, dim=1)
    seed   = top_k.indices[0].cpu().numpy().tolist()

    print(f"\n[result] Predicted seed set (size={len(seed)}): {seed[:20]}{'...' if len(seed)>20 else ''}")

    # ── Evaluate influence spread ─────────────────────────
    print("\n[eval] Running diffusion evaluation (10 MC runs)...")
    spread = diffusion_evaluation(adj, seed, diffusion=args.diffusion_model, n_runs=10)
    print(f"[result] Influence spread = {spread:.1f}")

    # Save results
    result_path = ckpt_dir / f"results_{args.dataset}_{args.diffusion_model}.txt"
    with open(result_path, "w") as f:
        f.write(f"dataset:        {args.dataset}\n")
        f.write(f"diffusion:      {args.diffusion_model}\n")
        f.write(f"seed_rate:      {args.seed_rate}\n")
        f.write(f"seed_num:       {seed_num}\n")
        f.write(f"spread:         {spread:.2f}\n")
        f.write(f"seed_set:       {seed}\n")
    print(f"[✓] Results saved → {result_path}")


if __name__ == "__main__":
    main()
