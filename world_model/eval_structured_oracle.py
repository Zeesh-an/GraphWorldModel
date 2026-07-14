"""
Oracle structured-rollout check (Lever 3 validation): build the structured IC head
with q = the TRUE edge transmission prob (no learning, encoder ignored) and run the
ensemble rollout. This must give ens_count_bias ~ 0 and ens_final_count_model ~ true,
confirming the IC structural form fixes saturation BEFORE we train a learned head.

python world_model/eval_structured_oracle.py \
    --data-dir data/output/ba20_marg --diffusion-model IC \
    --n-samples 20 --max-episodes 50 --device cpu
"""

import argparse
import torch

from world_model.wm_data import in_channels, load_graph_store
from world_model.wm_eval import rollout_ensemble
from world_model.wm_model import WorldModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Oracle structured rollout (q = true edge prob) — validates the IC structural form"
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
        help="diffusion dynamics (default: IC).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sage",
        help="encoder backbone ignored by oracle head (default: sage).",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=64, help="hidden dimension (default: 64)."
    )
    parser.add_argument(
        "--n-layers", type=int, default=3, help="number of encoder layers (default: 3)."
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="sampled rollouts per episode (default: 20).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=50,
        help="maximum episodes to evaluate (default: 50).",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0).")
    parser.add_argument(
        "--device", type=str, default="cpu", help="torch device string (default: cpu)."
    )
    args = parser.parse_args()

    if args.diffusion_model != "IC":
        raise ValueError(
            "the oracle is IC-only: LT thresholds are not stored, so there is no true-parameter "
            "oracle for LT. Validate the LT structured head via the ensemble rollout instead."
        )

    device = torch.device(args.device)

    model = WorldModel(
        args.model,
        in_channels=in_channels,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        head_type="structured_oracle",
    ).to(device)

    store = load_graph_store(args.data_dir)
    ensemble = rollout_ensemble(
        model,
        args.data_dir,
        args.diffusion_model,
        store,
        device,
        split="test",
        n_samples=args.n_samples,
        max_episodes=args.max_episodes,
        seed=args.seed,
    )

    print(
        f"[oracle structured rollout]  marg_mae={ensemble['ens_marg_mae']:.4f}  "
        f"count_w1={ensemble['ens_count_w1']:.3f}  "
        f"count_bias={ensemble['ens_count_bias']:.3f}  "
        f"model_cnt={ensemble['ens_final_count_model']:.2f}  "
        f"true_cnt={ensemble['ens_final_count_true']:.2f}"
    )
    print(
        "(expect count_bias ~ 0 and model_cnt ~ true_cnt if the structural form is correct)"
    )
