"""
Workstream F — is the large LT rollout bias the LEARNED model's, or the LT
formulation's?

    python -m scripts.lt_oracle_diagnostic \
        --data-dir results/influence_maximization/ba100/data \
        --out results/influence_maximization/ba100/lt_oracle_diagnostic.json

Why this exists instead of `--head structured_oracle`. That head is IC-only, and
deliberately so: `world_model/wm_model.py` refuses an LT oracle because LT
thresholds are drawn per episode and never stored, so there is no recorded
transition function to instantiate.

But LT does have an ANALYTIC oracle, and the simulator hands it to us. In
`data/wm_simulator.py::Simulator.reset`, thresholds are drawn
`rng.uniform(0.0, 1.0)` per node. For a node whose active in-neighbour weight
fraction is f_v, the probability it activates is therefore

    P(theta_v < f_v) = f_v        for theta ~ U(0, 1)

exactly. So `p_new(v) = f_v` is the best any model with no access to the
realised threshold can do — the Bayes-optimal LT marginal predictor. That is the
floor to compare the learned head against.

The diagnostic is a throwaway module that computes this closed form and is fed
through the SAME `rollout_ensemble` the learned model uses, so the comparison
differs in the transition function and nothing else. No architecture is changed:
this module lives here, not in `world_model/wm_model.py`.

Reading the result:

    oracle bias also large   -> the bias is structural. LT is a genuinely harder
                                setting because the thresholds are hidden, and
                                the honest move is to report it as such rather
                                than to tune it away.
    oracle bias small        -> the closed form is calibrated and the learned
                                T_endo is not. Investigate the learned q/tau.
"""

import argparse
import json
import os
from pathlib import Path
import torch
import torch.nn as nn

from world_model.wm_data import (
    GraphInput,
    basic_encoding,
    ch_add,
    ch_infected,
    ch_remove,
    load_graph_store,
)
from world_model.wm_eval import rollout_ensemble

probability_epsilon = 1e-6


class LTMarginalOracle(nn.Module):
    """
    The Bayes-optimal LT marginal predictor: `p_new(v) = f_v`.

    Mirrors `LTThresholdHead`'s exogenous path exactly — including the
    `active_pre` / `active_post` split that Stage A2 fixed — so any difference in
    the rollout is the TRANSMISSION model and not the intervention semantics.

    Carries one unused parameter so `.to(device)` and `.eval()` behave like any
    other module; nothing here is fitted.
    """

    head_type = "structured"

    def __init__(self) -> None:
        super().__init__()
        self.unused = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, X: torch.Tensor, graph: GraphInput) -> torch.Tensor:
        num_nodes = X.shape[0]
        device = X.device

        active_pre = X[:, ch_infected]
        active = torch.clamp(active_pre + X[:, ch_add], max=1.0) * (
            1.0 - X[:, ch_remove]
        )

        edge_index, edge_weight = graph.edge_index, graph.edge_weight

        if edge_index.numel() == 0:
            active_fraction = torch.zeros(num_nodes, device=device)
        else:
            sources, destinations = edge_index[0], edge_index[1]
            active_weight = torch.zeros(num_nodes, device=device).scatter_add_(
                0, destinations, active[sources] * edge_weight
            )
            total_weight = torch.zeros(num_nodes, device=device).scatter_add_(
                0, destinations, edge_weight
            )
            active_fraction = active_weight / total_weight.clamp(
                min=probability_epsilon
            )

        # theta ~ U(0,1)  =>  P(activate | f) = f. No sigmoid, no learned tau.
        p_new = active_fraction.clamp(0.0, 1.0)

        p_newly = (1.0 - active) * p_new
        y_inf = active + p_newly
        y_fr = torch.clamp(y_inf - active_pre, min=0.0)

        probs = torch.stack([y_inf, y_fr], dim=1).clamp(
            probability_epsilon, 1.0 - probability_epsilon
        )

        return torch.log(probs) - torch.log1p(-probs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LT analytic-oracle rollout diagnostic (Workstream F)"
    )
    parser.add_argument("--data-dir", type=str, required=True, help="dataset directory holding graphs/ and the transition files (default: None).")
    parser.add_argument("--split", type=str, default="test", help="dataset split to evaluate (default: test).")
    parser.add_argument("--n-samples", type=int, default=20, help="world-model ensemble size (default: 20).")
    parser.add_argument("--max-episodes", type=int, default=50, help="episodes per split to roll out (default: 50).")
    parser.add_argument("--seed", type=int, default=0, help="master random seed (default: 0).")
    parser.add_argument("--device", type=str, default="cpu", help="torch device (default: cpu).")
    parser.add_argument("--out", type=Path, required=True, help="output path (default: None).")
    args = parser.parse_args()

    device = torch.device(args.device)
    store = load_graph_store(args.data_dir)
    model = LTMarginalOracle().to(device).eval()

    print(f"[lt-oracle] analytic p_new = f_v on {args.data_dir} split={args.split}")

    rollout = rollout_ensemble(
        model,
        args.data_dir,
        "LT",
        store,
        device,
        split=args.split,
        n_samples=args.n_samples,
        max_episodes=args.max_episodes,
        seed=args.seed,
        action_encoding=basic_encoding,
    )

    blob = {
        "question": "Q2_structure / Workstream_F_LT_diagnostic",
        "predictor": "analytic LT marginal, p_new(v) = active in-neighbour "
                     "fraction f_v; exact because thresholds are U(0,1)",
        "learned": False,
        "data_dir": str(args.data_dir),
        "split": args.split,
        "rollout": rollout,
    }
    os.makedirs(args.out.parent, exist_ok=True)
    args.out.write_text(json.dumps(blob, indent=2, default=str))

    print(f"  ens_marg_mae         {rollout['ens_marg_mae']:.4f}")
    print(f"  ens_count_bias       {rollout['ens_count_bias']:+.3f}")
    print(f"  ens_count_w1         {rollout['ens_count_w1']:.3f}")
    print(f"  final model / true   {rollout['ens_final_count_model']:.2f} / "
          f"{rollout['ens_final_count_true']:.2f}")
    print(f"\n-> {args.out}")
