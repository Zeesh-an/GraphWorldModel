"""
Stochastic ensemble rollout (Lever 1) on a trained checkpoint, WITHOUT retraining.

Treats the world model as a stochastic simulator: samples the next state from the
predicted marginals each step (instead of thresholding at 0.5), rolls n_samples
trajectories, and compares the model's marginal/count distribution to the TRUE
simulator's MC trajectory under the same recorded actions. Tests whether the
rollout saturation is mostly a thresholding artifact.

python world_model/eval_rollout_ensemble.py \
    --results world_model/checkpoints/ba20_marg_sage_IC_structured.json \
    --n-samples 20 --max-episodes 50 --device cpu
"""

import argparse
import json
from pathlib import Path
import torch

from wm_data import load_graph_store
from wm_eval import rollout_ensemble
from eval_planning import load_trained_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sampled ensemble rollout vs true MC trajectory on a trained checkpoint (no retraining)"
    )
    parser.add_argument(
        "--results", nargs="+", required=True, help="results JSONs written by train_wm.py"
    )
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--max-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    print(
        f"{'run':>34} | {'marg_mae':>9} | {'count_w1':>9} | {'count_bias':>10} | {'model_cnt':>9} | {'true_cnt':>9}"
    )
    for path in args.results:
        results = json.loads(Path(path).read_text())
        config = results["config"]

        model = load_trained_model(config, device)
        store = load_graph_store(config["data_dir"])
        ens = rollout_ensemble(
            model,
            config["data_dir"],
            config["diffusion_model"],
            store,
            device,
            split="test",
            n_samples=args.n_samples,
            max_episodes=args.max_episodes,
            seed=args.seed,
        )

        results["rollout_ensemble"] = ens
        Path(path).write_text(json.dumps(results, indent=2, default=str))

        run = Path(path).stem
        print(
            f"{run:>34} | "
            f"{ens['ens_marg_mae']:>9.4f} | {ens['ens_count_w1']:>9.3f} | "
            f"{ens['ens_count_bias']:>10.3f} | {ens['ens_final_count_model']:>9.2f} | "
            f"{ens['ens_final_count_true']:>9.2f}"
        )
