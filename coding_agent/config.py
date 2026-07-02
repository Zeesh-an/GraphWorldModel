"""Experiment configuration for the coding-agent outer loop."""

from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    method: str = "one_shot"  # one_shot | per_step | windowed
    evaluator: str = "world_model"  # world_model | monte_carlo
    diffusion_model: str = "IC"  # IC | LT
    budget: int = 5
    horizon: int = 10
    windows: int = 3
    outer_iters: int = 3
    mc_runs: int = 30
    n_samples: int = 20
    seed: int = 42
    device: str = "cpu"
    data_dir: str | None = None  # world_model.wm_data graph store dir
    graph_id: str | None = None  # which graph in the store (default: first)
    wm_results_json: str | None = None  # train_wm.py results JSON (for the WM env)
    compare: bool = False  # also evaluate the winning strategy on the MC baseline
    out_json: str | None = None
