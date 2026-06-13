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

from wm_data import load_graph_store
from wm_eval import planning_regret_multi
from sweep_rollout import load_trained_model

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
