"""
Top-level driver: run the coding-agent outer loop over the inner-loop environment

python -m coding_agent.run --data-dir data/output/ba20_marg_structured \
    --model claude-haiku-4-5-20251001 \
    --wm-results-json world_model/checkpoints/ba20_marg_structured_sage_IC.json \
    --method one_shot --evaluator world_model --budget 5 --horizon 10 --compare \
    --outer-iters 3 --out-json coding_agent/results/first_llm_run.json
"""

import argparse
import json
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

from coding_agent.agent import CodingAgent, GatewayProvider
from coding_agent.credit import counterfactual_credit, planned_action
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
    model: str = "claude-sonnet-5"  # gateway model name
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
    credit: bool = False  # per-action counterfactual credit (feedback + results)
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


def _load_graph(config: ExperimentConfig) -> GraphInfo:
    if config.data_dir is None:
        raise ValueError(
            "config.data_dir is required unless a graph is passed directly"
        )

    from world_model.wm_data import load_graph_store

    store = load_graph_store(config.data_dir)
    graph_id = config.graph_id or next(iter(store))
    return GraphInfo.from_store_entry(store[graph_id])


def _build_environment(config: ExperimentConfig, graph: GraphInfo) -> object:
    if config.evaluator == monte_carlo:
        return MonteCarloEnvironment(
            graph, config.diffusion_model, mc_runs=config.mc_runs
        )
    if config.evaluator == world_model:
        from coding_agent.envs.world_model_env import WorldModelEnvironment

        if config.wm_results_json is None:
            raise ValueError("evaluator=world_model requires config.wm_results_json")
        return WorldModelEnvironment.from_results_json(
            config.wm_results_json,
            graph,
            device=config.device,
            n_samples=config.n_samples,
        )
    raise ValueError(f"unknown evaluator {config.evaluator!r}")


def _strategy_action(
    strategy: object, graph: GraphInfo, state: object, timestep: int
) -> list:
    return strategy.act(state, graph, timestep)


def run_experiment(
    config: ExperimentConfig,
    graph: GraphInfo | None = None,
    canned_script: str | None = None,
) -> dict:
    if graph is None:
        graph = _load_graph(config)

    environment = _build_environment(config, graph)
    provider = (
        _CannedProvider(canned_script)
        if canned_script
        else GatewayProvider(config.model)
    )
    agent = CodingAgent(provider)
    task = TaskSpec(
        diffusion_model=config.diffusion_model,
        budget=config.budget,
        horizon=config.horizon,
    )

    method = build_method(config.method)
    if config.method == "windowed":
        method = WindowedOnline(windows=config.windows)
    if config.method == "one_shot":
        method = OneShotSuperAlgorithm(
            outer_iters=config.outer_iters, credit=config.credit
        )

    strategy, trajectory = method.optimize(agent, environment, task, graph)

    result = {
        "method": config.method,
        "evaluator": config.evaluator,
        "reward": trajectory.reward,
        "summary": summarize(trajectory),
        "cost": trajectory.cost,
    }

    if config.credit:
        # Credit of the executed action sequence; for state-dependent strategies
        # (per_step/windowed) the recorded bags are replayed as a fixed plan.
        base_reward, entries = counterfactual_credit(
            environment, trajectory.actions, config.horizon, config.budget
        )
        result["credit_base_reward"] = base_reward
        result["credit"] = entries

    if config.compare and config.evaluator == world_model:
        mc_environment = MonteCarloEnvironment(
            graph, config.diffusion_model, mc_runs=config.mc_runs
        )
        plan = (
            strategy.plan_horizon(graph, config.budget, config.horizon)
            if hasattr(strategy, "plan_horizon")
            else None
        )
        action_fn = (
            partial(planned_action, plan)
            if plan is not None
            else partial(_strategy_action, strategy, graph)
        )
        mc_trajectory = mc_environment.rollout(action_fn, config.horizon, config.budget)
        result["mc_reward"] = mc_trajectory.reward
        result["wm_minus_mc"] = trajectory.reward - mc_trajectory.reward

    if config.out_json:
        os.makedirs(Path(config.out_json).parent, exist_ok=True)
        Path(config.out_json).write_text(json.dumps(result, indent=2, default=str))

    return result


def _parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Coding-agent outer loop over the world model"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-5",
        help="gateway model name, e.g. gpt-5.6-sol (default: claude-sonnet-5).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="one_shot",
        choices=["one_shot", "per_step", "windowed"],
        help="outer-loop method (default: one_shot).",
    )
    parser.add_argument(
        "--evaluator",
        type=str,
        default="world_model",
        choices=["world_model", "monte_carlo"],
        help="inner-loop evaluator (default: world_model).",
    )
    parser.add_argument(
        "--diffusion-model",
        type=str,
        default="IC",
        choices=["IC", "LT"],
        help="diffusion dynamics (default: IC).",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=5,
        help="seed/action budget (default: 5).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="rollout horizon (default: 10).",
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=3,
        help="number of windows for windowed method (default: 3).",
    )
    parser.add_argument(
        "--outer-iters",
        type=int,
        default=3,
        help="outer refinement iterations (default: 3).",
    )
    parser.add_argument(
        "--mc-runs",
        type=int,
        default=30,
        help="Monte Carlo simulator runs (default: 30).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="world-model rollout samples (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed (default: 42).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="torch device string (default: cpu).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="generated data directory (default: None).",
    )
    parser.add_argument(
        "--graph-id",
        type=str,
        default=None,
        help="graph id inside the graph store (default: None).",
    )
    parser.add_argument(
        "--wm-results-json",
        type=str,
        default=None,
        help="world-model results JSON path (default: None).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="also evaluate the final strategy on Monte Carlo (default: False).",
    )
    parser.add_argument(
        "--credit",
        action="store_true",
        help="per-action counterfactual credit: ablate each action, report its "
        "delta-spread in refinement feedback and the results JSON; costs one "
        "extra rollout per action (default: False).",
    )
    parser.add_argument(
        "--out-json", type=str, default=None, help="output JSON path (default: None)."
    )

    args = parser.parse_args()

    return ExperimentConfig(
        model=args.model,
        method=args.method,
        evaluator=args.evaluator,
        diffusion_model=args.diffusion_model,
        budget=args.budget,
        horizon=args.horizon,
        windows=args.windows,
        outer_iters=args.outer_iters,
        mc_runs=args.mc_runs,
        n_samples=args.n_samples,
        seed=args.seed,
        device=args.device,
        data_dir=args.data_dir,
        graph_id=args.graph_id,
        wm_results_json=args.wm_results_json,
        compare=args.compare,
        credit=args.credit,
        out_json=args.out_json,
    )


if __name__ == "__main__":
    load_dotenv()

    config = _parse_args()
    print(json.dumps(run_experiment(config), indent=2, default=str))
