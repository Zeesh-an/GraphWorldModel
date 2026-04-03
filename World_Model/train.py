"""
Graph Transformer — Inverse Graph Problem Training
=====================================================

Supports three tasks:
    IM  — Influence Maximization (select seeds to maximize spread)
    CND — Critical Node Detection (select nodes to maximize disruption / minimize connectivity)
    SL  — Source Localization (infer which sources caused an observed infection snapshot)

All share the same architecture (VAE + Graph Transformer) with different
Phase 2 objectives:
    IM:  MSE(y_hat, ones)  — maximize spread
    CND: MSE(y_hat, zeros) — minimize residual connectivity
    SL:  MSE(y_hat, observed_snapshot) — match predicted activation to observation

Phase 1 (--epochs,  default 600):
    Train VAE (Encoder + Decoder) + GraphTransformerForwardModel jointly.
    Loss = BCE(x_hat, x) + MSE(y_hat, y)

Phase 2 (--opt-iters, default 300):
    Freeze both models. Optimise latent z directly via backprop.
    Pick top-k nodes from x_hat as the predicted node set.

Usage
-----
    python World_Model/train.py -d cora_ml -dm IC -sp 1
    python World_Model/train.py -d cora_ml --task CND
    python World_Model/train.py -d cora_ml --task SL -dm IC
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

# Local Imports
ROOT = Path(__file__).resolve().parent.parent
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(THIS_DIR))

from model.vae import Encoder, Decoder, VAEModel
from model.graph_transformer import GraphTransformerForwardModel
from utils import (
    load_data,
    InverseProblemDataset,
    adj_process,
    top_sampling_init_indices,
    diffusion_evaluation,
    connectivity_evaluation,
    source_localization_evaluation,
)


# CLI (same flags as baseline genim.py)
def parse_args():
    p = argparse.ArgumentParser(
        description="Graph Transformer — Inverse Graph Problems"
    )

    p.add_argument(
        "-t",
        "--task",
        default="IM",
        choices=["IM", "CND", "SL"],
        help="Task: IM (influence maximization), CND (critical node detection), or SL (source localization)",
    )
    p.add_argument(
        "-d",
        "--dataset",
        default="cora_ml",
        choices=["jazz", "cora_ml", "power_grid", "netscience", "random5"],
        help="Dataset name",
    )
    p.add_argument(
        "-dm",
        "--diffusion_model",
        default="IC",
        choices=["IC", "LT", "SIS"],
        help="Diffusion model (IM only)",
    )
    p.add_argument(
        "-sp",
        "--seed_rate",
        default=1,
        type=int,
        choices=[1, 5, 10, 20],
        help="Seed rate (x 10 = seed set size percentage, IM only)",
    )
    p.add_argument(
        "-m",
        "--mode",
        default="normal",
        choices=["normal", "budget_constraint"],
        help="Evaluation mode",
    )

    # Training hyperparameters
    p.add_argument("--epochs", default=600, type=int, help="Phase 1 training epochs")
    p.add_argument(
        "--opt-iters", default=300, type=int, help="Phase 2 latent optimisation iters"
    )
    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument(
        "--lr-z", default=1e-4, type=float, help="Phase 2 latent z learning rate"
    )

    # Model dims
    p.add_argument("--hidden-dim", default=1024, type=int)
    p.add_argument("--latent-dim", default=512, type=int)
    p.add_argument("--gt-d-model", default=64, type=int, help="GT hidden dim")
    p.add_argument("--gt-heads", default=4, type=int, help="GT attention heads")
    p.add_argument("--gt-layers", default=3, type=int, help="Number of GT layers")
    p.add_argument("--gt-ffn", default=128, type=int, help="GT FFN hidden dim")
    p.add_argument("--gt-dropout", default=0.1, type=float)

    # Data paths
    p.add_argument(
        "--npz-dir",
        default=str(ROOT / "Data" / "cora_ml"),
        help="Path to generated .npz data directory",
    )
    p.add_argument(
        "--sg-dir",
        default=str(ROOT / "Baselines" / "DeepIM" / "data"),
        help="Path to baseline .SG data directory (fallback)",
    )
    p.add_argument(
        "--ckpt-dir",
        default=str(ROOT / "World_model" / "checkpoints"),
        help="Directory to save checkpoints",
    )

    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


def loss_phase_1(x, x_hat, y, y_hat):
    """
    Joint VAE + forward model loss.

    reproduction_loss = BCE(x_hat, x) — how well VAE reconstructs seed
    forward_loss = MSE(y_hat, y) — how well GT predicts influence

    total_loss = reproduction_loss + forward_loss
    """

    # Measures how well the VAE reconstructs the seed vector
    reproduction_loss = F.binary_cross_entropy(x_hat, x, reduction="sum")

    # Measures how well the Graph Transformer predicts the influence vector
    forward_loss = F.mse_loss(y_hat, y, reduction="sum")

    return reproduction_loss + forward_loss, reproduction_loss, forward_loss


def loss_phase_2(y_true, y_hat, x_hat):
    """
    Latent optimisation loss.
    forward_loss = MSE(y_hat, y_true) — push toward target state
        For IM:  y_true = ones (maximize spread)
        For CND: y_true = zeros (maximize disruption)
        For SL:  y_true = observed_snapshot (match predicted activation to observation)

    L0_loss = L1 sparsity on x_hat — keep node set small

    loss = forward_loss + L0_loss
    """

    forward_loss = F.mse_loss(y_hat, y_true)

    # L0 sparsity penalty to encourage the decoded seed vector to be sparse — sum(|x_hat|) / N
    L0_loss = torch.sum(torch.abs(x_hat)) / x_hat.shape[1]

    return forward_loss + L0_loss, L0_loss


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Data loading
    uses_diffusion = args.task in ("IM", "SL")

    # Try loading data from .npz, before falling back to .sg
    adj, inverse_pairs = load_data(
        dataset=args.dataset,
        diffusion_model=args.diffusion_model,
        seed_rate=args.seed_rate,
        task=args.task,
        npz_dir=Path(args.npz_dir),
        sg_dir=Path(args.sg_dir),
    )

    N = inverse_pairs.shape[1]  # Number of nodes
    task_label = f"{args.task}" + (
        f" | {args.diffusion_model}" if uses_diffusion else ""
    )
    print(
        f"[data] {args.dataset} | {task_label} | "
        f"N={N} | samples={len(inverse_pairs)}"
    )

    # Process the adjacency matrix: symmetrize, normalize, and convert to sparse COO
    adj_t = adj_process(adj).to(device)

    # Dataset / loaders (same split logic as baseline)
    if args.dataset == "random5":
        batch_size = 2
        hidden_dim = 4096
        latent_dim = 1024
    else:
        batch_size = 16
        hidden_dim = args.hidden_dim
        latent_dim = args.latent_dim

    # Train-test (90% - 10%) data split
    n_test = min(batch_size, len(inverse_pairs) // 10)
    n_train = len(inverse_pairs) - n_test

    train_set, test_set = random_split(inverse_pairs, [n_train, n_test])

    train_dataloader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False
    )
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

    # Models
    encoder = Encoder(input_dim=N, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(
        input_dim=latent_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=N
    )

    vae_model = VAEModel(encoder=encoder, decoder=decoder).to(device)
    forward_model = GraphTransformerForwardModel(
        d_model=args.gt_d_model,
        n_heads=args.gt_heads,
        n_layers=args.gt_layers,
        ffn_dim=args.gt_ffn,
        dropout=args.gt_dropout,
    ).to(device)

    num_vae_params = sum(param.numel() for param in vae_model.parameters())
    num_gt_params = sum(param.numel() for param in forward_model.parameters())
    print(f"[model] VAE params={num_vae_params:,}  |  GT params={num_gt_params:,}")

    optimizer = Adam(
        [{"params": vae_model.parameters()}, {"params": forward_model.parameters()}],
        lr=args.lr,
    )

    # PHASE 1 — Joint training
    print(f"\n{'='*60}")
    print(f" Phase 1 — Joint VAE + Graph Transformer training ({args.task})")
    print(f" Epochs: {args.epochs}  |  batch: {batch_size}")
    print(f"{'='*60}")

    vae_model.train()
    forward_model.train()
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        total_loss_ep = 0.0
        forward_loss_ep = 0.0
        recon_loss_ep = 0.0
        precision_re_ep = 0.0
        recall_re_ep = 0.0
        n_seen = 0

        for data_pair in train_dataloader:
            # data_pair: (B, N, 2)
            x = data_pair[:, :, 0].float().to(device)  # seed vectors
            y = data_pair[:, :, 1].float().to(device)  # influence vectors
            B = x.shape[0]  # batch size

            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)

            for i in range(B):
                x_i = x[i]  # (N,)
                y_i = y[i]  # (N,)

                x_hat = vae_model(x_i.unsqueeze(0))  # (1, N)

                # Forward model: (N, 1) → (N, 1)
                y_hat = forward_model(x_hat.squeeze(0).unsqueeze(-1), adj_t).squeeze(
                    -1
                )  # (N,)

                # BCE(x_hat, x) + MSE(y_hat, y)
                total, recon, forw = loss_phase_1(
                    x_i.unsqueeze(0),
                    x_hat,
                    y_i.unsqueeze(0),
                    y_hat.unsqueeze(0),
                )
                batch_loss += total

                # Reconstruction metrics
                x_pred_np = x_hat.detach().cpu().numpy()
                x_pred_np = (x_pred_np > 0.01).astype(float)
                x_true_np = x_i.unsqueeze(0).cpu().numpy()

                precision_re_ep += precision_score(
                    x_true_np[0], x_pred_np[0], zero_division=0
                )
                recall_re_ep += recall_score(
                    x_true_np[0], x_pred_np[0], zero_division=0
                )

                forward_loss_ep += forw.item()
                recon_loss_ep += recon.item()

            total_loss_ep += batch_loss.item()
            batch_loss = batch_loss / B

            batch_loss.backward()
            optimizer.step()

            # Clamp Graph Transformer forward mdoel params ≥ 0  (mirrors baseline's clamp for SpGAT)
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

        # Checkpoint the best model
        if avg(total_loss_ep) < best_loss:
            best_loss = avg(total_loss_ep)
            ckpt_suffix = args.diffusion_model if uses_diffusion else "CND"
            ckpt = ckpt_dir / f"best_{args.dataset}_{ckpt_suffix}.pt"

            torch.save(
                {
                    "epoch": epoch,
                    "vae": vae_model.state_dict(),
                    "forward_model": forward_model.state_dict(),
                    "loss": best_loss,
                    "args": vars(args),
                },
                ckpt,
            )

    print(f"\n[✓] Phase 1 done. Best loss={best_loss:.4f}")
    print(f"[✓] Checkpoint saved → {ckpt}")

    # PHASE 2 — Latent optimisation
    print(f"\n{'='*60}")
    print(f" Phase 2 — Latent z optimisation ({args.opt_iters} iters)")
    print(f"{'='*60}")

    # Freeze both models to prevent weight updates
    for p in vae_model.parameters():
        p.requires_grad = False

    for p in forward_model.parameters():
        p.requires_grad = False

    vae_model.eval()
    forward_model.eval()

    # Initialize z from the top-performing training samples' encodings
    # IM / SL: highest channel 1 sum (top-spreading samples)
    # CND: lowest channel 1 sum (most-destructive samples)
    init_idx = top_sampling_init_indices(
        inverse_pairs, frac=0.1, largest=(args.task != "CND")
    )

    z_hat = torch.zeros(1, latent_dim, device=device)
    with torch.no_grad():
        for i in init_idx:
            vec_i = inverse_pairs[i, :, 0].float().unsqueeze(0).to(device)
            z_hat += encoder(vec_i)

    z_hat = z_hat / len(init_idx)
    z_hat = z_hat.detach().requires_grad_(True)

    print(f"[phase2] Initialized z from top-{len(init_idx)} samples")

    # Set the node budget and y_target
    if args.task == "SL":
        # For Source Localization (SL), use the first test sample as the target
        test_sample = test_set[0]  # (N, 2)

        # Binary groud-truth seed nodes vector
        sl_true_sources = test_sample[:, 0].numpy()  # (N,)

        # Binary observation snapshot of nodes
        sl_observed_snapshot = test_sample[:, 1].numpy()  # (N,)

        # Source Localization Optimization Target
        y_target = test_sample[:, 1].float().unsqueeze(dim=0).to(device)  # (1, N)

        node_budget = int(sl_true_sources.sum())

        print(
            f"[phase2] SL query — observed spread = {int(sl_observed_snapshot.sum())}"
        )
        print(f"[phase2] Ground truth source count = {node_budget}")
    elif args.task == "IM":
        with torch.no_grad():
            x_init = decoder(z_hat)

        node_budget = max(1, int(x_init.sum().item()))

        # For Influence Maximization (IM), the target is for all nodes to be activated/influenced
        y_target = torch.ones(1, N, device=device)

        print(f"[phase2] Estimated seed budget = {node_budget}")
    elif args.task == "CND":
        with torch.no_grad():
            x_init = decoder(z_hat)

        node_budget = max(1, int(x_init.sum().item()))

        # For Critical Node Detection (CND), the target is for all nodes to be deactivated/disconnected
        y_target = torch.zeros(1, N, device=device)

        print(f"[phase2] Estimated removal budget = {node_budget}")

    z_optimizer = Adam([z_hat], lr=args.lr_z)

    # Optimization iterations for inverse graph optimization of latent optimal seed node vector z_hat
    # The smooth and continuous latent vector z_hat is optimized rather than directly optimizing the binary, sparse seed node vector because optimization works better on the smooth vector
    for i in range(1, args.opt_iters + 1):
        # Decode the current z_hat through VAE decoder
        x_hat = vae_model.decoder(z_hat)  # (1, N)

        # Pass the decoded vector through the forward Graph Transformer to predict influence/connectivity y_hat
        y_hat = (
            forward_model(x_hat.squeeze(0).unsqueeze(-1), adj_t)
            .squeeze(-1)
            .unsqueeze(0)
        )  # (1, N)

        loss, L0 = loss_phase_2(y_target, y_hat, x_hat)

        # Perform backpropagation (∂L/z_hat) and gradient descent to optimize and update z_hat
        z_optimizer.zero_grad()
        loss.backward()
        z_optimizer.step()

        if i % 50 == 0 or i == 1:
            if args.task == "SL":
                metric_label = "PredMatch"
            elif args.task == "IM":
                metric_label = "PredSpread"
            elif args.task == "CND":
                metric_label = "PredConn"

            print(
                f"  Iter {i:>4d}/{args.opt_iters}"
                f"  Loss={loss.item():.5f}"
                f"  L0={L0.item():.5f}"
                f"  {metric_label}={y_hat.sum().item():.1f}"
            )

    # Extract the final optimal node set for the inverse graph problem
    with torch.no_grad():
        # Decode the optimized z_hat to get x_final of probabilities
        x_final = vae_model.decoder(z_hat)

    # Get the optimal node set, by choosing the top-K best nodes by probability
    top_k = x_final.topk(node_budget, dim=1)
    node_set = top_k.indices[0].cpu().numpy().tolist()

    if args.task == "SL":
        set_label = "source"
    elif args.task == "IM":
        set_label = "seed"
    elif args.task == "CND":
        set_label = "removal"

    print(
        f"\n[result] Predicted {set_label} set (size={len(node_set)}): "
        f"{node_set[:20]}{'...' if len(node_set) > 20 else ''}"
    )

    # Evaluate
    ckpt_suffix = args.diffusion_model if uses_diffusion else "CND"

    if args.task == "SL":
        print("\n[eval] Running source localization evaluation...")
        true_source_indices = np.where(sl_true_sources > 0.5)[0].tolist()
        metrics = source_localization_evaluation(
            predicted_sources=node_set,
            true_sources=true_source_indices,
            observed_snapshot=sl_observed_snapshot,
            adj=adj,
            diffusion=args.diffusion_model,
            n_runs=10,
        )
        print(
            f"[result] Precision={metrics['precision']:.3f}  "
            f"Recall={metrics['recall']:.3f}  "
            f"F1={metrics['f1']:.3f}  "
            f"Jaccard={metrics['jaccard']:.3f}"
        )
        print(
            f"[result] Predicted spread={metrics['predicted_spread']:.1f}  "
            f"Observed spread={metrics['observed_spread']:.0f}  "
            f"Spread error={metrics['spread_error']:.3f}"
        )

        result_path = ckpt_dir / f"results_{args.dataset}_SL_{ckpt_suffix}.txt"

        with open(result_path, "w") as f:
            f.write(f"task:              {args.task}\n")
            f.write(f"dataset:           {args.dataset}\n")
            f.write(f"diffusion:         {args.diffusion_model}\n")
            f.write(f"source_budget:     {node_budget}\n")
            f.write(f"precision:         {metrics['precision']:.4f}\n")
            f.write(f"recall:            {metrics['recall']:.4f}\n")
            f.write(f"f1:                {metrics['f1']:.4f}\n")
            f.write(f"jaccard:           {metrics['jaccard']:.4f}\n")
            f.write(f"predicted_spread:  {metrics['predicted_spread']:.2f}\n")
            f.write(f"observed_spread:   {metrics['observed_spread']:.0f}\n")
            f.write(f"spread_error:      {metrics['spread_error']:.4f}\n")
            f.write(f"predicted_sources: {node_set}\n")
            f.write(f"true_sources:      {true_source_indices}\n")

    elif args.task == "IM":
        print("\n[eval] Running diffusion evaluation (10 MC runs)...")
        spread = diffusion_evaluation(
            adj, node_set, diffusion=args.diffusion_model, n_runs=10
        )
        print(f"[result] Influence spread = {spread:.1f}")

        result_path = ckpt_dir / f"results_{args.dataset}_{ckpt_suffix}.txt"

        with open(result_path, "w") as f:
            f.write(f"task:           {args.task}\n")
            f.write(f"dataset:        {args.dataset}\n")
            f.write(f"diffusion:      {args.diffusion_model}\n")
            f.write(f"seed_rate:      {args.seed_rate}\n")
            f.write(f"seed_num:       {node_budget}\n")
            f.write(f"spread:         {spread:.2f}\n")
            f.write(f"seed_set:       {node_set}\n")
    elif args.task == "CND":
        print("\n[eval] Running connectivity evaluation...")
        metrics = connectivity_evaluation(adj, node_set)
        print(
            f"[result] Largest CC = {metrics['largest_cc']}  "
            f"({metrics['frac_connected']*100:.1f}% of remaining)"
        )
        print(
            f"[result] Components = {metrics['n_components']}  "
            f"| Pairwise conn = {metrics['pairwise_conn']}"
        )

        result_path = ckpt_dir / f"results_{args.dataset}_{ckpt_suffix}.txt"

        with open(result_path, "w") as f:
            f.write(f"task:           {args.task}\n")
            f.write(f"dataset:        {args.dataset}\n")
            f.write(f"node_budget:    {node_budget}\n")
            f.write(f"largest_cc:     {metrics['largest_cc']}\n")
            f.write(f"n_components:   {metrics['n_components']}\n")
            f.write(f"pairwise_conn:  {metrics['pairwise_conn']}\n")
            f.write(f"frac_connected: {metrics['frac_connected']:.4f}\n")
            f.write(f"removal_set:    {node_set}\n")

    print(f"[✓] Results saved → {result_path}")


if __name__ == "__main__":
    main()
