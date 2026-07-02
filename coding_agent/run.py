"""Top-level driver: run the coding-agent outer loop over the inner-loop environment.

CLI:
    python -m coding_agent.run --data-dir data/output/ba20_marg_structured \
        --wm-results-json world_model/checkpoints/ba20_marg_structured_sage_IC.json \
        --method one_shot --evaluator world_model --budget 5 --horizon 10 --compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from coding_agent.agent import CodingAgent, TODOProvider
from coding_agent.config import ExperimentConfig
from coding_agent.envs.monte_carlo_env import MonteCarloEnvironment
from coding_agent.methods.base import OuterLoopMethod, summarize
from coding_agent.methods.one_shot import OneShotSuperAlgorithm
from coding_agent.methods.per_step import PerStepReprompt
from coding_agent.methods.windowed import WindowedOnline
from coding_agent.types import GraphInfo, TaskSpec

WORLD_MODEL = "world_model"
MONTE_CARLO = "monte_carlo"


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
        raise ValueError("config.data_dir is required unless a graph is passed directly")
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "world_model"))
    from wm_data import load_graph_store

    store = load_graph_store(cfg.data_dir)
    gid = cfg.graph_id or next(iter(store))
    return GraphInfo.from_store_entry(store[gid])


def _build_env(cfg: ExperimentConfig, g: GraphInfo):
    if cfg.evaluator == MONTE_CARLO:
        return MonteCarloEnvironment(g, cfg.diffusion_model, mc_runs=cfg.mc_runs)
    if cfg.evaluator == WORLD_MODEL:
        from coding_agent.envs.world_model_env import WorldModelEnvironment

        if cfg.wm_results_json is None:
            raise ValueError("evaluator=world_model requires config.wm_results_json")
        return WorldModelEnvironment.from_results_json(
            cfg.wm_results_json, g, device=cfg.device, n_samples=cfg.n_samples
        )
    raise ValueError(f"unknown evaluator {cfg.evaluator!r}")


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

    if cfg.compare and cfg.evaluator == WORLD_MODEL:
        mc_env = MonteCarloEnvironment(g, cfg.diffusion_model, mc_runs=cfg.mc_runs)
        plan = strategy.plan_horizon(g, cfg.budget, cfg.horizon) \
            if hasattr(strategy, "plan_horizon") else None
        action_fn = (
            (lambda s, t, _p=plan: _p[t] if _p and t < len(_p) else [])
            if plan is not None
            else (lambda s, t: strategy.act(s, g, t))
        )
        mc_tr = mc_env.rollout(action_fn, cfg.horizon, cfg.budget)
        result["mc_reward"] = mc_tr.reward
        result["wm_minus_mc"] = trajectory.reward - mc_tr.reward

    if cfg.out_json:
        Path(cfg.out_json).write_text(json.dumps(result, indent=2, default=str))
    return result


def _parse_args() -> ExperimentConfig:
    p = argparse.ArgumentParser(description="Coding-agent outer loop over the world model")
    p.add_argument("--method", default="one_shot", choices=["one_shot", "per_step", "windowed"])
    p.add_argument("--evaluator", default="world_model", choices=["world_model", "monte_carlo"])
    p.add_argument("--diffusion-model", default="IC", choices=["IC", "LT"])
    p.add_argument("--budget", type=int, default=5)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--windows", type=int, default=3)
    p.add_argument("--outer-iters", type=int, default=3)
    p.add_argument("--mc-runs", type=int, default=30)
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--graph-id", default=None)
    p.add_argument("--wm-results-json", default=None)
    p.add_argument("--compare", action="store_true")
    p.add_argument("--out-json", default=None)
    a = p.parse_args()
    return ExperimentConfig(
        method=a.method, evaluator=a.evaluator, diffusion_model=a.diffusion_model,
        budget=a.budget, horizon=a.horizon, windows=a.windows, outer_iters=a.outer_iters,
        mc_runs=a.mc_runs, n_samples=a.n_samples, seed=a.seed, device=a.device,
        data_dir=a.data_dir, graph_id=a.graph_id, wm_results_json=a.wm_results_json,
        compare=a.compare, out_json=a.out_json,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    print(json.dumps(run_experiment(cfg), indent=2, default=str))
