"""
Runnable self-check for condition 9 (baselines/discovery.py).

Builds a tiny influence-maximization dataset, writes a discovery context exactly
as the pipeline would, and drives `baselines/score_program.py` through the same
subprocess glue every framework uses: a valid program must score, a broken one
must fail with a zero fitness and an error message, never a crash.

    python -m baselines.check_discovery
    python -m baselines.check_discovery --live eoh     # one framework, smoke budget

`--live` needs the framework installed (`python -m baselines.setup_baselines
--only <name>`) and the gateway credentials in .env; DISCOVERY_SMOKE=1 shrinks the
framework's own budget to a handful of samples so the whole adapter is exercised
end to end in minutes rather than hours.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from baselines.discovery import (
    build_context,
    contracts,
    program_script,
    write_context,
    write_scoring_glue,
)
from baselines.run_baseline import run_external_baseline
from coding_agent.envs.monte_carlo_env import MonteCarloEnvironment
from coding_agent.executor import build_strategy
from coding_agent.methods.base import evaluate_strategy
from coding_agent.run import ExperimentConfig
from coding_agent.types import GraphInfo, TaskSpec
from pipeline.conditions import monte_carlo
from world_model.wm_data import load_graph_store

task_name = "influence_maximization"
check_nodes = 40
check_budget = 3
check_horizon = 5
check_mc_runs = 20


def build_dataset(root: Path) -> Path:
    argv = [
        sys.executable, "-m", "pipeline.run",
        "--task", task_name, "--dataset", "er", "--run", "disc_check",
        "--results-root", str(root), "--end-stage", "data",
        "--num-graphs", "1", "--syn-nodes", str(check_nodes),
        "--rollouts", "2", "--mc-marginals", "1",
    ]
    print(f"[check] {' '.join(argv)}")
    subprocess.run(argv, check=True)

    return root / task_name / "er" / "disc_check" / "data"


def scoring_module(work_dir: Path):
    path = write_scoring_glue(work_dir, work_dir / "discovery_context.json")
    spec = importlib.util.spec_from_file_location("discovery_scoring", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def check_scorer(work_dir: Path, experiment: ExperimentConfig, graph: GraphInfo) -> None:
    context = build_context(experiment, task_name, "add_node", work_dir, graph)
    write_context(work_dir, context)
    glue = scoring_module(work_dir)

    good = glue.score_program(program=contracts[task_name].initial_program, tag="good")
    assert good["ok"], good
    assert good["sense"] == "maximize" and good["reward"] > 0, good
    assert good["fitness"] == good["reward"], good
    print(f"[check] valid program: reward {good['reward']:.2f} in {good['seconds']:.1f}s")

    broken = glue.score_program(program="def other(graph, k):\n    return []\n", tag="broken")
    assert not broken["ok"] and broken["fitness"] == 0.0, broken
    assert "select_seeds" in broken["error"], broken["error"][-400:]
    print("[check] broken program: fitness 0 with a named error")

    forbidden = glue.score_program(program="import os\n" + contracts[task_name].initial_program, tag="os")
    assert not forbidden["ok"] and "os" in forbidden["error"], forbidden["error"][-400:]
    print("[check] forbidden import: rejected by the executor whitelist")


def check_live(name: str, work_root: Path, experiment: ExperimentConfig, graph: GraphInfo) -> None:
    os.environ["DISCOVERY_SMOKE"] = "1"
    work_dir = work_root / task_name / "er" / "disc_check" / "baselines" / "_runs" / name / "k3"
    context = build_context(experiment, task_name, "add_node", work_dir, graph)
    result = run_external_baseline(
        name, graph, check_budget, "IC", work_dir=work_dir, timeout=3600, context=context
    )
    program = result["program"]
    print(f"[check] {name}: best program ({len(program.splitlines())} lines):")
    print("\n".join("    " + line for line in program.splitlines()[:25]))

    strategy = build_strategy(program_script(program, task_name))
    environment = MonteCarloEnvironment(graph, "IC", mc_runs=check_mc_runs, base_seed=1)
    task = TaskSpec(task=task_name, budget=check_budget, horizon=check_horizon)
    trajectory, _ = evaluate_strategy(strategy, environment, task, graph)
    print(f"[check] {name}: referee spread {trajectory.reward:.2f} (info {json.dumps(result['info'])})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-check for condition 9")
    parser.add_argument(
        "--live",
        type=str,
        default=None,
        help="also run one installed framework at a smoke budget (default: None).",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="results root to build the tiny dataset under (default: a temp dir).",
    )
    args = parser.parse_args()
    load_dotenv()

    root = Path(args.root) if args.root else Path(tempfile.mkdtemp(prefix="disc_check_"))
    os.makedirs(root, exist_ok=True)
    data_dir = build_dataset(root)
    store = load_graph_store(str(data_dir))
    graph = GraphInfo.from_store_entry(store[next(iter(store))])
    experiment = ExperimentConfig(
        task=task_name,
        method="one_shot",
        evaluator=monte_carlo,
        budget=check_budget,
        horizon=check_horizon,
        mc_runs=check_mc_runs,
        data_dir=str(data_dir),
        seed=42,
    )

    work_dir = root / task_name / "er" / "disc_check" / "baselines" / "_runs" / "scorer" / "k3"
    os.makedirs(work_dir, exist_ok=True)
    check_scorer(work_dir, experiment, graph)

    if args.live:
        check_live(args.live, root, experiment, graph)

    print(f"[check] OK (artefacts under {root})")
