"""
Forward Graph Model — Inverse Graph Problem Training

Models:
    GT — Graph Transformer (scatter-softmax attention, scale-invariant)
    GCN — GCN with pre-norm residual blocks
    GAT — GATv2 with scatter-softmax multi-head attention
    SAGE — GraphSAGE with mean aggregator
    GCNII — GCNII with initial residual + identity mapping

Tasks:
    IM — Influence Maximization (select seeds to maximize spread)
    CND — Critical Node Detection (select nodes to maximize disruption)
    SL — Source Localization (infer sources from observed infection snapshot)

Phase 1 (--epochs, default 600):
    Supervised training of the forward model against ground-truth (x, y) pairs.
    Loss = MSE(y_hat, y)

Phase 2 (--opt-iters, default 300):
    Freeze forward model. Optimize logits directly via backprop, where
    x_hat = sigmoid(logits). Pick top-k nodes from x_hat as the predicted node set.
    Phase 2 targets differ by task:
        IM: MSE(y_hat, ones) — maximize spread
        CND: MSE(y_hat, zeros) — minimize residual connectivity
        SL: MSE(y_hat, observed_snapshot) — match predicted activation to observation

Usage
-----
    python world_model/train.py --task IM -d cora_ml -dm IC --k 10 --k-pct 1 \\
        --model gcn --gcn-hidden 64 --gcn-layers 3 \\
        --epochs 600 --opt-iters 300 --lr 1e-4 --lr-z 1e-3 \\
        --npz-dir data/cora_ml
"""

import argparse
import sys
from tqdm.auto import tqdm
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

# Local Imports
ROOT = Path(__file__).resolve().parent.parent.parent   # project root
WM_DIR = Path(__file__).resolve().parent.parent         # world_model/ (for `from model.x`)
THIS_DIR = Path(__file__).resolve().parent              # world_model/old/ (for `from utils`)
for _p in (str(ROOT), str(WM_DIR), str(THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model.graph_transformer import GraphTransformerForwardModel
from model.gcn import GCNForwardModel
from model.gat import GATForwardModel
from model.graphsage import GraphSAGEForwardModel
from model.gcnii import GCNIIForwardModel
from utils import (
    load_data,
    adj_process,
    top_sampling_init_indices,
    diffusion_evaluation,
    connectivity_evaluation,
    source_localization_evaluation,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Forward Graph Model Training — Inverse Graph Problems"
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
        help="Diffusion model",
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
        "--k",
        default=10,
        type=int,
        help="Seed/removal/source set size k — must match the k used during data generation (default: 10)",
    )
    p.add_argument(
        "--k-pct",
        default=None,
        type=float,
        help="Seed/removal budget as percentage of N (e.g., 5 = 5%%). Overrides --k for Phase 2 node budget. "
        "Note: --k is still used for data loading (must match data generation k).",
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
        "--opt-iters", default=300, type=int, help="Phase 2 logit optimization iters"
    )
    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument(
        "--lr-z",
        default=1e-4,
        type=float,
        help="Phase 2 logit optimization learning rate",
    )
    p.add_argument(
        "--l0-weight",
        default=1.0,
        type=float,
        help="Phase 2 L0/L1 sparsity weight on x_hat. Higher values pull x_hat "
        "toward the sparse regime the forward model was trained on. Try 50-500 "
        "if Phase 2 PredSpread stays stuck near 0.",
    )

    # Model selection
    p.add_argument(
        "--model",
        default="gt",
        choices=["gt", "gcn", "gat", "sage", "gcnii"],
        help="Forward model architecture",
    )

    # Graph Transformer (GT) hyperparameters
    p.add_argument("--gt-d-model", default=64, type=int, help="GT hidden dim")
    p.add_argument("--gt-heads", default=4, type=int, help="GT attention heads")
    p.add_argument("--gt-layers", default=3, type=int, help="Number of GT layers")
    p.add_argument("--gt-ffn", default=128, type=int, help="GT FFN hidden dim")
    p.add_argument("--gt-dropout", default=0.1, type=float)

    # GCN hyperparameters
    p.add_argument("--gcn-hidden", default=64, type=int, help="GCN hidden dim")
    p.add_argument("--gcn-layers", default=3, type=int, help="Number of GCN layers")
    p.add_argument("--gcn-dropout", default=0.1, type=float)
    p.add_argument(
        "--gcn-no-pe",
        action="store_true",
        help="Disable degree positional encoding in GCN input",
    )

    # GAT hyperparameters
    p.add_argument("--gat-hidden", default=64, type=int, help="GAT hidden dim")
    p.add_argument("--gat-heads", default=4, type=int, help="GAT attention heads")
    p.add_argument("--gat-layers", default=3, type=int, help="Number of GAT layers")
    p.add_argument("--gat-dropout", default=0.1, type=float)
    p.add_argument(
        "--gat-no-pe",
        action="store_true",
        help="Disable degree positional encoding in GAT input",
    )

    # GraphSAGE hyperparameters
    p.add_argument("--sage-hidden", default=64, type=int, help="GraphSAGE hidden dim")
    p.add_argument(
        "--sage-layers", default=3, type=int, help="Number of GraphSAGE layers"
    )
    p.add_argument("--sage-dropout", default=0.1, type=float)
    p.add_argument(
        "--sage-no-pe",
        action="store_true",
        help="Disable degree positional encoding in GraphSAGE input",
    )

    # GCNII hyperparameters
    p.add_argument("--gcnii-hidden", default=64, type=int, help="GCNII hidden dim")
    p.add_argument(
        "--gcnii-layers",
        default=8,
        type=int,
        help="Number of GCNII layers (GCNII is designed for depth)",
    )
    p.add_argument(
        "--gcnii-alpha",
        default=0.1,
        type=float,
        help="GCNII initial-residual strength (α)",
    )
    p.add_argument(
        "--gcnii-lamda",
        default=0.5,
        type=float,
        help="GCNII identity-mapping decay hyperparameter (λ)",
    )
    p.add_argument("--gcnii-dropout", default=0.1, type=float)
    p.add_argument(
        "--gcnii-no-pe",
        action="store_true",
        help="Disable degree positional encoding in GCNII input",
    )

    # Data paths
    p.add_argument(
        "--npz-dir",
        default=str(ROOT / "data" / "cora_ml"),
        help="Path to generated .npz data directory",
    )
    p.add_argument(
        "--sg-dir",
        default=str(ROOT / "baselines" / "DeepIM" / "data"),
        help="Path to baseline .SG data directory (fallback)",
    )
    p.add_argument(
        "--ckpt-dir",
        default=str(ROOT / "world_model" / "checkpoints"),
        help="Directory to save checkpoints",
    )

    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


def build_forward_model(args: argparse.Namespace) -> nn.Module:
    if args.model == "gt":
        return GraphTransformerForwardModel(
            d_model=args.gt_d_model,
            n_heads=args.gt_heads,
            n_layers=args.gt_layers,
            ffn_dim=args.gt_ffn,
            dropout=args.gt_dropout,
        )
    if args.model == "gcn":
        return GCNForwardModel(
            hidden_dim=args.gcn_hidden,
            n_layers=args.gcn_layers,
            dropout=args.gcn_dropout,
            use_pe=not args.gcn_no_pe,
        )
    if args.model == "gat":
        return GATForwardModel(
            hidden_dim=args.gat_hidden,
            n_heads=args.gat_heads,
            n_layers=args.gat_layers,
            dropout=args.gat_dropout,
            use_pe=not args.gat_no_pe,
        )
    if args.model == "sage":
        return GraphSAGEForwardModel(
            hidden_dim=args.sage_hidden,
            n_layers=args.sage_layers,
            dropout=args.sage_dropout,
            use_pe=not args.sage_no_pe,
        )
    if args.model == "gcnii":
        return GCNIIForwardModel(
            hidden_dim=args.gcnii_hidden,
            n_layers=args.gcnii_layers,
            alpha=args.gcnii_alpha,
            lamda=args.gcnii_lamda,
            dropout=args.gcnii_dropout,
            use_pe=not args.gcnii_no_pe,
        )
    raise ValueError(f"Unknown --model: {args.model}")


def loss_phase_1(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """
    Forward prediction loss = MSE(y_hat, y) — how well the forward model predicts the outcome

    Argument order mirrors loss_phase_2 (target first, prediction second).
    """
    return F.mse_loss(y_hat, y, reduction="sum")


def loss_phase_2(
    y_true: torch.Tensor,
    y_hat: torch.Tensor,
    x_hat: torch.Tensor,
    l0_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Latent optimization loss.
    forward_loss = MSE(y_hat, y_true) — push toward target state
        For IM:  y_true = ones (maximize spread)
        For CND: y_true = zeros (maximize disruption)
        For SL:  y_true = observed_snapshot (match predicted activation to observation)

    L0_loss = L1 sparsity on x_hat — keep node set small
        Scaled by l0_weight to pull x_hat into the sparse regime the forward
        model was trained on (binary k-hot vectors).

    loss = forward_loss + l0_weight * L0_loss
    """

    forward_loss = F.mse_loss(y_hat, y_true)

    # L1 sparsity penalty on x_hat — sum(|x_hat|) / N
    L0_loss = torch.sum(torch.abs(x_hat)) / x_hat.shape[1]

    return forward_loss + l0_weight * L0_loss, L0_loss


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
        k=args.k,
        task=args.task,
        npz_dir=Path(args.npz_dir),
        sg_dir=Path(args.sg_dir),
    )

    N = inverse_pairs.shape[1]  # Number of nodes

    # Resolve --k-pct → node budget (requires N, so must happen after data load)
    if args.k_pct is not None:
        args.node_budget = max(1, round(N * args.k_pct / 100))
        print(
            f"[config] --k-pct={args.k_pct}% of N={N} → node budget = {args.node_budget}"
        )
    else:
        args.node_budget = args.k

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
    batch_size = 2 if args.dataset == "random5" else 16

    # Train-test (90% - 10%) data split
    n_test = min(batch_size, len(inverse_pairs) // 10)
    n_train = len(inverse_pairs) - n_test

    train_set, test_set = random_split(inverse_pairs, [n_train, n_test])

    train_dataloader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False
    )
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

    # Forward model (VAE-free)
    forward_model = build_forward_model(args).to(device)

    num_params = sum(param.numel() for param in forward_model.parameters())
    print(f"[model] {args.model.upper()} params={num_params:,}")

    optimizer = Adam(forward_model.parameters(), lr=args.lr)

    node_budget = args.node_budget

    # PHASE 1 — Forward-model training
    print(f"\n{'='*60}")
    print(f" Phase 1 — {args.model.upper()} forward-model training ({args.task})")
    print(f" Epochs: {args.epochs}  |  batch: {batch_size}")
    print(f"{'='*60}")

    forward_model.train()
    best_loss = float("inf")

    epoch_progress_bar = tqdm(
        range(1, args.epochs + 1), desc=f"Phase 1 — {args.task} training"
    )

    for epoch in epoch_progress_bar:
        total_loss_ep = 0.0
        n_seen = 0

        for data_pair in train_dataloader:
            # data_pair: (B, N, 2)
            x = data_pair[:, :, 0].float().to(device)  # action vectors
            y = data_pair[:, :, 1].float().to(device)  # outcome vectors
            B = x.shape[0]

            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)

            for i in range(B):
                x_i = x[i]  # (N,)
                y_i = y[i]  # (N,)

                # Ground-truth x straight into the forward model (no VAE)
                y_hat = forward_model(x_i.unsqueeze(-1), adj_t).squeeze(-1)  # (N,)

                loss_i = loss_phase_1(y_i.unsqueeze(0), y_hat.unsqueeze(0))
                batch_loss += loss_i

            total_loss_ep += batch_loss.item()
            batch_loss = batch_loss / B
            batch_loss.backward()
            optimizer.step()

            n_seen += B

        avg_loss = total_loss_ep / max(n_seen, 1)
        epoch_progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

        # Checkpoint the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_suffix = (
                f"{args.task}_{args.diffusion_model}" if uses_diffusion else "CND"
            )
            ckpt = (
                ckpt_dir
                / f"best_{args.dataset}_{ckpt_suffix}_{args.model}_k{node_budget}.pt"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_type": args.model,
                    "forward_model": forward_model.state_dict(),
                    "loss": best_loss,
                    "args": vars(args),
                },
                ckpt,
            )

    print(f"\n[✓] Phase 1 done. Best loss={best_loss:.4f}")
    print(f"[✓] Checkpoint saved → {ckpt}")

    # PHASE 2 — Direct logit optimization (VAE-free)
    print(f"\n{'='*60}")
    print(f" Phase 2 — Direct logit optimization ({args.opt_iters} iters)")
    print(f"{'='*60}")

    # Freeze the forward model
    for p in forward_model.parameters():
        p.requires_grad = False
    forward_model.eval()

    # Initialize logits from the mean of top-performing training samples.
    # IM / SL: highest channel 1 sum (top-spreading samples)
    # CND: lowest channel 1 sum (most-destructive samples)
    init_idx = top_sampling_init_indices(
        inverse_pairs, frac=0.1, largest=(args.task != "CND")
    )

    x_mean = inverse_pairs[init_idx, :, 0].float().mean(dim=0)  # (N,) in [0, 1]
    x_mean = x_mean.clamp(min=0.01, max=0.99)
    logits = torch.log(x_mean / (1.0 - x_mean)).unsqueeze(0).to(device)  # (1, N)
    logits = logits.detach().requires_grad_(True)

    print(f"[phase2] Initialized logits from top-{len(init_idx)} samples")

    # Set the y_target per task
    if args.task == "SL":
        # For Source Localization (SL), use the first test sample as the target
        test_sample = test_set[0]  # (N, 2)

        # Binary ground-truth seed nodes vector
        sl_true_sources = test_sample[:, 0].numpy()  # (N,)

        # Binary observation snapshot of nodes
        sl_observed_snapshot = test_sample[:, 1].numpy()  # (N,)

        # Source Localization Optimization Target
        y_target = test_sample[:, 1].float().unsqueeze(dim=0).to(device)  # (1, N)

        # SL uses ground truth source count as budget (may differ from args.k)
        node_budget = int(sl_true_sources.sum())

        print(
            f"[phase2] SL query — observed spread = {int(sl_observed_snapshot.sum())}"
        )
        print(f"[phase2] Ground truth source count = {node_budget}")
    elif args.task == "IM":
        # For Influence Maximization (IM), the target is for all nodes to be activated/influenced
        y_target = torch.ones(1, N, device=device)

        print(
            f"[phase2] Seed budget (k) = {node_budget}  ({node_budget / N * 100:.2f}% of N={N})"
        )
    elif args.task == "CND":
        # For Critical Node Detection (CND), the target is for all nodes to be deactivated/disconnected
        y_target = torch.zeros(1, N, device=device)

        print(
            f"[phase2] Removal budget (k) = {node_budget}  ({node_budget / N * 100:.2f}% of N={N})"
        )

    z_optimizer = Adam([logits], lr=args.lr_z)

    # Direct optimization of continuous action vector x_hat = sigmoid(logits).
    # Backprop flows: loss -> y_hat -> forward_model -> x_hat -> logits.
    metric_labels = {"SL": "PredMatch", "IM": "PredSpread", "CND": "PredConn"}
    optimization_progress_bar = tqdm(
        range(1, args.opt_iters + 1),
        desc=f"Phase 2 — {args.task} logit optimization",
    )

    for i in optimization_progress_bar:
        x_hat = torch.sigmoid(logits)  # (1, N)

        y_hat = (
            forward_model(x_hat.squeeze(0).unsqueeze(-1), adj_t)
            .squeeze(-1)
            .unsqueeze(0)
        )  # (1, N)

        loss, L0 = loss_phase_2(y_target, y_hat, x_hat, l0_weight=args.l0_weight)

        z_optimizer.zero_grad()
        loss.backward()
        z_optimizer.step()

        optimization_progress_bar.set_postfix(
            loss=f"{loss.item():.5f}",
            L0=f"{L0.item():.5f}",
            **{metric_labels[args.task]: f"{y_hat.sum().item():.1f}"},
        )

    # Extract the final optimal node set for the inverse graph problem
    with torch.no_grad():
        x_final = torch.sigmoid(logits)  # (1, N)

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
        source_pct = node_budget / N * 100
        pred_spread_pct = metrics["predicted_spread"] / N * 100
        obs_spread_pct = metrics["observed_spread"] / N * 100

        print(
            f"[result] Precision={metrics['precision']:.3f}  "
            f"Recall={metrics['recall']:.3f}  "
            f"F1={metrics['f1']:.3f}  "
            f"Jaccard={metrics['jaccard']:.3f}"
        )
        print(
            f"[result] Predicted spread={metrics['predicted_spread']:.1f} ({pred_spread_pct:.2f}%)  "
            f"Observed spread={metrics['observed_spread']:.0f} ({obs_spread_pct:.2f}%)  "
            f"Spread error={metrics['spread_error']:.3f}"
        )
        print(
            f"[result] Source budget = {node_budget} nodes  ({source_pct:.2f}% of N={N})"
        )

        result_path = (
            ckpt_dir / f"results_{args.dataset}_{ckpt_suffix}_k{node_budget}.txt"
        )

        with open(result_path, "w") as f:
            f.write(f"task:              {args.task}\n")
            f.write(f"dataset:           {args.dataset}\n")
            f.write(f"diffusion:         {args.diffusion_model}\n")
            f.write(f"N:                 {N}\n")
            f.write(f"source_budget:     {node_budget}\n")
            f.write(f"source_budget_pct: {source_pct:.2f}%\n")
            f.write(f"precision:         {metrics['precision']:.4f}\n")
            f.write(f"recall:            {metrics['recall']:.4f}\n")
            f.write(f"f1:                {metrics['f1']:.4f}\n")
            f.write(f"jaccard:           {metrics['jaccard']:.4f}\n")
            f.write(f"predicted_spread:  {metrics['predicted_spread']:.2f}\n")
            f.write(f"pred_spread_pct:   {pred_spread_pct:.2f}%\n")
            f.write(f"observed_spread:   {metrics['observed_spread']:.0f}\n")
            f.write(f"obs_spread_pct:    {obs_spread_pct:.2f}%\n")
            f.write(f"spread_error:      {metrics['spread_error']:.4f}\n")
            f.write(f"predicted_sources: {node_set}\n")
            f.write(f"true_sources:      {true_source_indices}\n")

    elif args.task == "IM":
        print("\n[eval] Running diffusion evaluation (10 MC runs)...")
        spread = diffusion_evaluation(
            adj, node_set, diffusion=args.diffusion_model, n_runs=10
        )
        spread_pct = spread / N * 100
        seed_pct = node_budget / N * 100

        print(
            f"[result] Influence spread = {spread:.1f} nodes  ({spread_pct:.2f}% of N={N})"
        )
        print(f"[result] Seed budget = {node_budget} nodes  ({seed_pct:.2f}% of N={N})")

        result_path = (
            ckpt_dir / f"results_{args.dataset}_{ckpt_suffix}_k{node_budget}.txt"
        )

        with open(result_path, "w") as f:
            f.write(f"task:           {args.task}\n")
            f.write(f"dataset:        {args.dataset}\n")
            f.write(f"diffusion:      {args.diffusion_model}\n")
            f.write(f"N:              {N}\n")
            f.write(f"k:              {node_budget}\n")
            f.write(f"k_pct:          {seed_pct:.2f}%\n")
            f.write(f"spread:         {spread:.2f}\n")
            f.write(f"spread_pct:     {spread_pct:.2f}%\n")
            f.write(f"seed_set:       {node_set}\n")
    elif args.task == "CND":
        print("\n[eval] Running connectivity evaluation...")
        metrics = connectivity_evaluation(adj, node_set)
        remaining_nodes = N - node_budget
        removal_pct = node_budget / N * 100
        largest_cc_pct = (
            metrics["largest_cc"] / remaining_nodes * 100
            if remaining_nodes > 0
            else 0.0
        )
        largest_cc_pct_total = metrics["largest_cc"] / N * 100

        print(
            f"[result] Largest CC = {metrics['largest_cc']} nodes  "
            f"({largest_cc_pct:.2f}% of remaining, {largest_cc_pct_total:.2f}% of N={N})"
        )
        print(
            f"[result] Components = {metrics['n_components']}  "
            f"| Pairwise conn = {metrics['pairwise_conn']}"
        )
        print(
            f"[result] Removal budget = {node_budget} nodes  ({removal_pct:.2f}% of N={N})"
        )

        result_path = (
            ckpt_dir / f"results_{args.dataset}_{ckpt_suffix}_k{node_budget}.txt"
        )

        with open(result_path, "w") as f:
            f.write(f"task:           {args.task}\n")
            f.write(f"dataset:        {args.dataset}\n")
            f.write(f"N:              {N}\n")
            f.write(f"k:              {node_budget}\n")
            f.write(f"k_pct:          {removal_pct:.2f}%\n")
            f.write(f"largest_cc:     {metrics['largest_cc']}\n")
            f.write(f"largest_cc_pct: {largest_cc_pct:.2f}%\n")
            f.write(f"n_components:   {metrics['n_components']}\n")
            f.write(f"pairwise_conn:  {metrics['pairwise_conn']}\n")
            f.write(f"frac_connected: {metrics['frac_connected']:.4f}\n")
            f.write(f"removal_set:    {node_set}\n")

    print(f"[✓] Results saved → {result_path}")


if __name__ == "__main__":
    main()
