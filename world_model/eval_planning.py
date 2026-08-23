"""
Recompute multi-graph planning regret on already-trained checkpoints, WITHOUT
retraining, and write the updated numbers back into each results JSON.

Planning regret depends only on the trained weights and the graphs, not on the
training run — so like the rollout-threshold sweep, recomputing it should reload
the checkpoint rather than retrain. Use this after changing how planning is
measured (e.g. single-graph -> multi-graph averaging).

python -m world_model.eval_planning \
    --results results/ba40/world_model/sage_IC.json \
    --plan-graphs 5 --device cpu
"""

import argparse
import json
import os
from pathlib import Path
import torch
import torch.nn as nn

from data.wm_simulator import spent
from world_model.wm_data import basic_encoding, load_graph_store, num_input_channels
from world_model.wm_eval import planning_regret_multi
from world_model.wm_model import WorldModel


def load_trained_model(config: dict, device: torch.device) -> nn.Module:
    """Rebuild the WorldModel from a saved run config and load its checkpoint."""
    backbone_kwargs = {
        "n_heads": config["n_heads"],
        "ffn_dim": config["ffn_dim"],
        "alpha": config["gcnii_alpha"],
        "lamda": config["gcnii_lamda"],
    }
    model = WorldModel(
        config["model"],
        # Absent in runs from before --action-encoding, all of which were `basic`
        in_channels=num_input_channels(config.get("action_encoding", basic_encoding)),
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        head_type=config.get("head", "linear"),
        diffusion_model=config["diffusion_model"],
        # Absent in runs trained before --remove-semantics existed, all of which
        # were spent
        remove_semantics=config.get("remove_semantics", spent),
        **backbone_kwargs,
    ).to(device)

    # train_wm.py saves to <ckpt_dir>/wm_<model>_<dm>.pt
    checkpoint_path = (
        Path(config["ckpt_dir"])
        / f"wm_{config['model']}_{config['diffusion_model']}.pt"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint_path} (derived from "
            f"model={config['model']}, diffusion_model={config['diffusion_model']}, "
            f"ckpt_dir={config['ckpt_dir']})"
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recompute multi-graph planning regret on trained checkpoints (no retraining)"
    )
    parser.add_argument(
        "--results",
        type=str,
        nargs="+",
        required=True,
        help="results JSONs written by train_wm.py (default: required).",
    )
    parser.add_argument(
        "--plan-graphs",
        type=int,
        default=5,
        help="graphs to average for planning regret (default: 5).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)."
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="torch device string (default: cpu)."
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"{'run':>28} | {'model±std':>16} | {'degree':>7} | {'random':>7}")
    for path in args.results:
        results = json.loads(Path(path).read_text())
        config = results["config"]

        model = load_trained_model(config, device)
        store = load_graph_store(config["data_dir"])
        planning = planning_regret_multi(
            model,
            store,
            config["diffusion_model"],
            device,
            n_graphs=args.plan_graphs,
            seed=args.seed,
            # All three must follow the checkpoint, not this CLI's defaults: a
            # w-hidden model re-scored against true weights, or a `blocked` model
            # scored against a `spent` oracle, produces a number for the wrong run
            hide_edge_weights=bool(config.get("hide_edge_weights", False)),
            remove_semantics=config.get("remove_semantics", spent),
            action_encoding=config.get("action_encoding", basic_encoding),
        )

        # Replace only the planning block; keep test + rollout intact.
        results["planning"] = planning
        os.makedirs(Path(path).parent, exist_ok=True)
        Path(path).write_text(json.dumps(results, indent=2, default=str))

        run = Path(path).stem
        print(
            f"{run:>28} | "
            f"{planning['plan_regret_model']:>6.3f}±{planning['plan_regret_model_std']:<5.3f} | "
            f"{planning['plan_regret_degree']:>7.3f} | "
            f"{planning['plan_regret_random']:>7.3f}"
        )
