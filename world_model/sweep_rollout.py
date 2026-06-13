"""
Standalone rollout-threshold sweep: reload a trained world-model checkpoint and
re-run the free-running rollout at several thresholds WITHOUT retraining.

The rollout threshold only affects evaluation (how the autoregressive state is
binarized each step), not the trained weights — so sweeping it by retraining is
pure waste. This loads one checkpoint and evaluates every threshold in seconds.

python world_model/sweep_rollout.py \
    --results world_model/checkpoints/ba20_all_gt_IC.json \
    --thresholds 0.5 0.6 0.7 0.8 0.9
"""

import argparse
import json
from pathlib import Path
import torch

from wm_data import TransitionDataset, IN_CHANNELS
from wm_model import WorldModel
from wm_eval import rollout_episodes


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
        description="Sweep the rollout threshold on a trained world model (no retraining)"
    )
    parser.add_argument(
        "--results", required=True, help="results JSON written by train_wm.py"
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9]
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    results = json.loads(Path(args.results).read_text())
    config = results["config"]
    diffusion_model = config["diffusion_model"]
    data_dir = config["data_dir"]

    model = load_trained_model(config, device)
    dataset = TransitionDataset(data_dir, diffusion_model, args.split)

    sweep = {}
    for thr in args.thresholds:
        sweep[f"{thr:.2f}"] = rollout_episodes(
            model, data_dir, diffusion_model, dataset.store, device, args.split, threshold=thr
        )

    out = Path(args.results).with_name(Path(args.results).stem + "_thrsweep.json")
    out.write_text(
        json.dumps({"config": config, "thresholds": sweep}, indent=2, default=str)
    )

    print(
        f"rollout-threshold sweep  model={config['model']}  dm={diffusion_model}  split={args.split}"
    )
    print(f"{'thr':>5} | {'newinf_f1':>9} | {'count_mae':>9} | {'final_f1':>9}")
    for thr in args.thresholds:
        m = sweep[f"{thr:.2f}"]
        print(
            f"{thr:>5.2f} | {m['rollout_newinf_f1']:>9.4f} | "
            f"{m['rollout_count_mae']:>9.2f} | {m['rollout_final_f1']:>9.4f}"
        )
    print(f"[sweep] -> {out}")
