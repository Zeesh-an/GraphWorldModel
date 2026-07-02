"""
World-model environment: roll the trained transition model f_theta autoregressively,
sampling the next state from its predicted marginals each step.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# world_model/ on path so its bare-name imports resolve.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_model.wm_data import (
    apply_edge_ops,
    build_features,
    build_graph_input,
    edges_to_arrays,
)
from world_model.wm_model import WorldModel

from coding_agent.types import ActionFn, GraphInfo, State, Trajectory

_EDGE_OPS = ("add_edge", "remove_edge", "set_edge_weight")


class WorldModelEnvironment:
    def __init__(
        self,
        model: torch.nn.Module,
        g: GraphInfo,
        diffusion_model: str,
        device: str = "cpu",
        n_samples: int = 20,
    ) -> None:
        self.model = model.to(device).eval()
        self.g = g
        self.diffusion_model = diffusion_model
        self.device = torch.device(device)
        self.n_samples = n_samples
        self.base_edges = {
            (int(g.edge_index[0, i]), int(g.edge_index[1, i])): float(g.ic_probs[i])
            for i in range(g.edge_index.shape[1])
        }

    @classmethod
    def from_results_json(
        cls, results_json: str, g: GraphInfo, device: str = "cpu", n_samples: int = 20
    ) -> "WorldModelEnvironment":
        """Rebuild a WorldModel from a train_wm.py results JSON config and load its checkpoint."""
        import json

        cfg = json.loads(Path(results_json).read_text())["config"]
        bb = {
            "n_heads": cfg["n_heads"],
            "ffn_dim": cfg["ffn_dim"],
            "alpha": cfg["gcnii_alpha"],
            "lamda": cfg["gcnii_lamda"],
        }
        model = WorldModel(
            cfg["model"],
            in_channels=6,
            hidden_dim=cfg["hidden_dim"],
            n_layers=cfg["n_layers"],
            dropout=cfg["dropout"],
            head_type=cfg.get("head", "linear"),
            diffusion_model=cfg["diffusion_model"],
            **bb,
        )
        ckpt = Path(cfg["ckpt_dir"]) / f"wm_{cfg['model']}_{cfg['diffusion_model']}.pt"

        if not ckpt.exists():
            raise FileNotFoundError(f"world-model checkpoint not found: {ckpt}")

        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

        return cls(model, g, cfg["diffusion_model"], device=device, n_samples=n_samples)

    @torch.inference_mode()
    def _single_rollout(
        self, action_fn: ActionFn, horizon: int, rng: np.random.Generator
    ):
        n = self.g.num_nodes
        edges = dict(self.base_edges)
        infected: set[int] = set()
        frontier: set[int] = set()
        states = [State([], [])]
        actions: list[list] = []
        counts = [0.0]

        for t in range(horizon + 1):
            state = State(sorted(infected), sorted(frontier))
            bag = action_fn(state, t)
            actions.append(bag)
            # Apply edge ops to the running graph so the model sees the post-action graph.
            bag_dicts = [a.to_dict() for a in bag]
            if any(d["op"] in _EDGE_OPS for d in bag_dicts):
                edges = apply_edge_ops(edges, bag_dicts)
            ei, w = edges_to_arrays(edges)
            record = {
                "state": {"infected": sorted(infected), "frontier": sorted(frontier)},
                "action": bag_dicts,
                "next_state": {"infected": [], "frontier": []},
                "next_marginal_infected": {},
                "next_marginal_frontier": {},
            }
            X, _, _ = build_features(record, ei, n)
            gi = build_graph_input(ei, w, n, self.diffusion_model, self.device)
            prob = (
                torch.sigmoid(self.model(torch.from_numpy(X).to(self.device), gi))
                .cpu()
                .numpy()
            )  # (N, 2)
            draw_inf = rng.random(n) < prob[:, 0]
            draw_fr = rng.random(n) < prob[:, 1]
            infected = set(np.nonzero(draw_inf)[0].tolist())
            frontier = set(np.nonzero(draw_fr)[0].tolist())
            states.append(State(sorted(infected), sorted(frontier)))
            counts.append(float(len(infected)))
            if t > 0 and not frontier and not bag:
                break
        return states, actions, counts

    def rollout(
        self, action_fn: ActionFn, horizon: int, budget: int, seed: int = 0
    ) -> Trajectory:
        rng = np.random.default_rng(seed)
        finals: list[float] = []
        rep = None
        for s in range(self.n_samples):
            states, actions, counts = self._single_rollout(action_fn, horizon, rng)
            finals.append(counts[-1])
            if s == 0:
                rep = (states, actions, counts)
        reward = float(np.mean(finals)) if finals else 0.0
        states, actions, counts = rep  # type: ignore[misc]
        counts[-1] = reward
        return Trajectory(
            states=states,
            actions=actions,
            reward=reward,
            infected_counts=counts,
            cost={"n_samples": self.n_samples, "env": "world_model"},
        )
