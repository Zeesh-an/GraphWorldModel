"""
Recompute multi-graph planning regret on already-trained checkpoints, WITHOUT
retraining, and write the updated numbers back into each results JSON.

Planning regret depends only on the trained weights and the graphs, not on the
training run — so like the rollout-threshold sweep, recomputing it should reload
the checkpoint rather than retrain. Use this after changing how planning is
measured (e.g. single-graph -> multi-graph averaging).

python world_model/eval_planning.py \
    --results world_model/checkpoints/ba20_all_*_IC.json \
    --plan-graphs 5 --device cpu
"""

import argparse
import json
from pathlib import Path
import torch

from wm_data import load_graph_store, IN_CHANNELS
from wm_model import WorldModel
from wm_eval import planning_regret_multi


def load_trained_model(config: dict, device: torch.device) -> torch.nn.Module:
    """Rebuild the WorldModel from a saved run config and load its checkpoint."""
    bb = {
        "n_heads": config["n_heads"],
        "ffn_dim": config["ffn_dim"],
        "alpha": config["gcnii_alpha"],
        "lamda": config["gcnii_lamda"],
    }
    model = WorldModel(
        config["model"],
        in_channels=IN_CHANNELS,
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        head_type=config.get("head", "linear"),
        **bb,
    ).to(device)

    # train_wm.py saves to <ckpt_dir>/wm_<model>_<dm>.pt
    ckpt = (
        Path(config["ckpt_dir"])
        / f"wm_{config['model']}_{config['diffusion_model']}.pt"
    )
    if not ckpt.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {ckpt} (derived from model={config['model']}, "
            f"diffusion_model={config['diffusion_model']}, ckpt_dir={config['ckpt_dir']})"
        )

    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recompute multi-graph planning regret on trained checkpoints (no retraining)"
    )
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="results JSONs written by train_wm.py",
    )
    parser.add_argument("--plan-graphs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
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
        )

        # Replace only the planning block; keep test + rollout intact.
        results["planning"] = planning
        Path(path).write_text(json.dumps(results, indent=2, default=str))

        run = Path(path).stem
        print(
            f"{run:>28} | "
            f"{planning['plan_regret_model']:>6.3f}±{planning['plan_regret_model_std']:<5.3f} | "
            f"{planning['plan_regret_degree']:>7.3f} | "
            f"{planning['plan_regret_random']:>7.3f}"
        )
