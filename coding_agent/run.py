"""Top-level driver: run the coding-agent outer loop over the inner-loop environment.

CLI:
    python -m coding_agent.run --data-dir data/output/ba20_marg_structured \
        --wm-results-json world_model/checkpoints/ba20_marg_structured_sage_IC.json \
        --method one_shot --evaluator world_model --budget 5 --horizon 10 --compare
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from coding_agent.agent import CodingAgent, TODOProvider
from coding_agent.envs.monte_carlo_env import MonteCarloEnvironment
from coding_agent.methods.base import OuterLoopMethod, summarize
from coding_agent.methods.one_shot import OneShotSuperAlgorithm
from coding_agent.methods.per_step import PerStepReprompt
from coding_agent.methods.windowed import WindowedOnline
from coding_agent.types import GraphInfo, TaskSpec

world_model = "world_model"
monte_carlo = "monte_carlo"


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


class _CannedProvider:
    """Used when a canned script is supplied (tests / model-less runs)."""

    def __init__(self, script: str) -> None:
        self.script = script

    def complete(self, system: str, user: str) -> str:
        return f"```python\n{self.script}\n```"


def build_method(name: str) -> OuterLoopMethod:
    if name == "one_shot":
        return OneShotSuperAlgorithm()
    if name == "per_step":
        return PerStepReprompt()
    if name == "windowed":
        return WindowedOnline()
    raise ValueError(f"unknown method {name!r}; choose one_shot|per_step|windowed")


def _load_graph(cfg: ExperimentConfig) -> GraphInfo:
    if cfg.data_dir is None:
        raise ValueError(
            "config.data_dir is required unless a graph is passed directly"
        )

    from world_model.wm_data import load_graph_store

    store = load_graph_store(cfg.data_dir)
    gid = cfg.graph_id or next(iter(store))
    return GraphInfo.from_store_entry(store[gid])


def _build_env(cfg: ExperimentConfig, g: GraphInfo) -> object:
    if cfg.evaluator == monte_carlo:
        return MonteCarloEnvironment(g, cfg.diffusion_model, mc_runs=cfg.mc_runs)
    if cfg.evaluator == world_model:
        from coding_agent.envs.world_model_env import WorldModelEnvironment

        if cfg.wm_results_json is None:
            raise ValueError("evaluator=world_model requires config.wm_results_json")
        return WorldModelEnvironment.from_results_json(
            cfg.wm_results_json, g, device=cfg.device, n_samples=cfg.n_samples
        )
    raise ValueError(f"unknown evaluator {cfg.evaluator!r}")


def _planned_action(plan: list[list], state: object, t: int) -> list:
    return plan[t] if plan and t < len(plan) else []


def _strategy_action(strategy: object, graph: GraphInfo, state: object, t: int) -> list:
    return strategy.act(state, graph, t)


def run_experiment(
    cfg: ExperimentConfig,
    graph: GraphInfo | None = None,
    canned_script: str | None = None,
) -> dict:
    g = graph if graph is not None else _load_graph(cfg)
    env = _build_env(cfg, g)
    provider = _CannedProvider(canned_script) if canned_script else TODOProvider()
    agent = CodingAgent(provider)
    task = TaskSpec(
        diffusion_model=cfg.diffusion_model, budget=cfg.budget, horizon=cfg.horizon
    )
    method = build_method(cfg.method)
    if cfg.method == "windowed":
        method = WindowedOnline(windows=cfg.windows)
    if cfg.method == "one_shot":
        method = OneShotSuperAlgorithm(outer_iters=cfg.outer_iters)

    strategy, trajectory = method.optimize(agent, env, task, g)

    result = {
        "method": cfg.method,
        "evaluator": cfg.evaluator,
        "reward": trajectory.reward,
        "summary": summarize(trajectory),
        "cost": trajectory.cost,
    }

    if cfg.compare and cfg.evaluator == world_model:
        mc_env = MonteCarloEnvironment(g, cfg.diffusion_model, mc_runs=cfg.mc_runs)
        plan = (
            strategy.plan_horizon(g, cfg.budget, cfg.horizon)
            if hasattr(strategy, "plan_horizon")
            else None
        )
        action_fn = (
            partial(_planned_action, plan)
            if plan is not None
            else partial(_strategy_action, strategy, g)
        )
        mc_tr = mc_env.rollout(action_fn, cfg.horizon, cfg.budget)
        result["mc_reward"] = mc_tr.reward
        result["wm_minus_mc"] = trajectory.reward - mc_tr.reward

    if cfg.out_json:
        os.makedirs(Path(cfg.out_json).parent, exist_ok=True)
        Path(cfg.out_json).write_text(json.dumps(result, indent=2, default=str))
    return result


def _parse_args() -> ExperimentConfig:
    p = argparse.ArgumentParser(
        description="Coding-agent outer loop over the world model"
    )
    p.add_argument(
        "--method",
        type=str,
        default="one_shot",
        choices=["one_shot", "per_step", "windowed"],
        help="outer-loop method (default: one_shot).",
    )
    p.add_argument(
        "--evaluator",
        type=str,
        default="world_model",
        choices=["world_model", "monte_carlo"],
        help="inner-loop evaluator (default: world_model).",
    )
    p.add_argument(
        "--diffusion-model",
        type=str,
        default="IC",
        choices=["IC", "LT"],
        help="diffusion dynamics (default: IC).",
    )
    p.add_argument(
        "--budget", type=int, default=5, help="seed/action budget (default: 5)."
    )
    p.add_argument(
        "--horizon", type=int, default=10, help="rollout horizon (default: 10)."
    )
    p.add_argument(
        "--windows",
        type=int,
        default=3,
        help="number of windows for windowed method (default: 3).",
    )
    p.add_argument(
        "--outer-iters",
        type=int,
        default=3,
        help="outer refinement iterations (default: 3).",
    )
    p.add_argument(
        "--mc-runs",
        type=int,
        default=30,
        help="Monte Carlo simulator runs (default: 30).",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="world-model rollout samples (default: 20).",
    )
    p.add_argument("--seed", type=int, default=42, help="random seed (default: 42).")
    p.add_argument(
        "--device", type=str, default="cpu", help="torch device string (default: cpu)."
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="generated data directory (default: None).",
    )
    p.add_argument(
        "--graph-id",
        type=str,
        default=None,
        help="graph id inside the graph store (default: None).",
    )
    p.add_argument(
        "--wm-results-json",
        type=str,
        default=None,
        help="world-model results JSON path (default: None).",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="also evaluate the final strategy on Monte Carlo (default: False).",
    )
    p.add_argument(
        "--out-json", type=str, default=None, help="output JSON path (default: None)."
    )
    a = p.parse_args()
    return ExperimentConfig(
        method=a.method,
        evaluator=a.evaluator,
        diffusion_model=a.diffusion_model,
        budget=a.budget,
        horizon=a.horizon,
        windows=a.windows,
        outer_iters=a.outer_iters,
        mc_runs=a.mc_runs,
        n_samples=a.n_samples,
        seed=a.seed,
        device=a.device,
        data_dir=a.data_dir,
        graph_id=a.graph_id,
        wm_results_json=a.wm_results_json,
        compare=a.compare,
        out_json=a.out_json,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    print(json.dumps(run_experiment(cfg), indent=2, default=str))
