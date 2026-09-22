"""
Time one world-model rollout against one NDlib Monte Carlo rollout, and nothing else.

    python -m scripts.time_evaluators --run-dir results/influence_maximization/netscience/final_ic

No LLM, no search, no oracle, no referee. Both evaluators roll the same fixed plan forward
(the highest-degree nodes seeded at t = 0) on a finished run's graph, the world model from
that run's checkpoint. Reads the run and writes only under --out-dir.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from coding_agent.envs.monte_carlo_env import MonteCarloEnvironment
from coding_agent.envs.world_model_env import WorldModelEnvironment
from coding_agent.types import GraphInfo
from data.wm_simulator import ActionOp
from pipeline.plots import plot_runtime
from world_model.wm_data import load_graph_store


def seconds_per_sample(environment: object, plan: object, horizon: int, budget: int, samples: int, repeats: int, seed: int, device: str, label: str) -> list[float]:
    timings = []

    for repeat in range(repeats):
        start = time.perf_counter()
        environment.rollout(plan, horizon, budget, seed=seed + repeat)

        # a CUDA rollout returns before its kernels finish unless the clock waits for them
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        timings.append(elapsed / samples)
        print(f"  {label} rollout {repeat + 1}/{repeats}: {elapsed:.1f} s for {samples} -> {1000 * elapsed / samples:.2f} ms each", flush=True)

    return timings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seconds per rollout sample: world model against NDlib Monte Carlo")
    parser.add_argument("--run-dir", type=Path, required=True, help="finished run holding data/ and world_model/ (default: None).")
    parser.add_argument("--wm-model", type=str, default="sage", help="backbone in the results file name (default: sage).")
    parser.add_argument("--diffusion-model", type=str, default="IC", help="dynamics in the results file name (default: IC).")
    parser.add_argument("--budget-pct", type=float, default=1.0, help="percent of nodes seeded at t = 0 (default: 1.0).")
    parser.add_argument("--horizon", type=int, default=10, help="rollout steps (default: 10).")
    parser.add_argument("--wm-samples", type=int, default=200, help="world-model samples per rollout (default: 200).")
    parser.add_argument("--mc-episodes", type=int, default=50, help="NDlib episodes per rollout (default: 50).")
    parser.add_argument("--repeats", type=int, default=3, help="timed rollouts per evaluator, the median is reported (default: 3).")
    parser.add_argument("--seed", type=int, default=42, help="master random seed (default: 42).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="torch device for the world model (default: cuda when available).")
    parser.add_argument("--out-dir", type=Path, default=Path("results/timing"), help="where the JSON and the figure go (default: results/timing).")
    args = parser.parse_args()

    store = load_graph_store(str(args.run_dir / "data"))
    graph = GraphInfo.from_store_entry(store[next(iter(store))])

    budget = max(1, round(graph.num_nodes * args.budget_pct / 100))
    degree = np.bincount(graph.edge_index[0], minlength=graph.num_nodes) + np.bincount(graph.edge_index[1], minlength=graph.num_nodes)
    seeds = np.argsort(-degree)[:budget].tolist()

    def plan(state: object, timestep: int) -> list:
        return [ActionOp("add_node", int(node)) for node in seeds] if timestep == 0 else []

    task, dataset = args.run_dir.parts[-3], args.run_dir.parts[-2]
    results_json = args.run_dir / "world_model" / f"{args.wm_model}_{args.diffusion_model}.json"
    world_model = WorldModelEnvironment.from_results_json(str(results_json), graph, device=args.device, n_samples=args.wm_samples, base_seed=args.seed)
    monte_carlo = MonteCarloEnvironment(graph, args.diffusion_model, mc_runs=args.mc_episodes, base_seed=args.seed, remove_semantics=world_model.remove_semantics)

    print(f"{dataset}: {graph.num_nodes:,} nodes, {graph.edge_index.shape[1]:,} arcs, {budget} seeds, horizon {args.horizon}, world model on {args.device}", flush=True)

    # the first call on a device pays for kernel start-up, which is not the model's speed
    start = time.perf_counter()
    world_model.rollout(plan, args.horizon, budget, seed=args.seed - 1)
    print(f"  warm-up rollout: {time.perf_counter() - start:.1f} s", flush=True)

    wm_timings = seconds_per_sample(world_model, plan, args.horizon, budget, args.wm_samples, args.repeats, args.seed, args.device, "world model")
    mc_timings = seconds_per_sample(monte_carlo, plan, args.horizon, budget, args.mc_episodes, args.repeats, args.seed, "cpu", "Monte Carlo")
    wm_seconds = statistics.median(wm_timings)
    mc_seconds = statistics.median(mc_timings)

    summary = {
        "run_dir": str(args.run_dir),
        "num_nodes": int(graph.num_nodes),
        "num_arcs": int(graph.edge_index.shape[1]),
        "budget": budget,
        "horizon": args.horizon,
        "device": args.device,
        "world_model_seconds_per_sample": wm_seconds,
        "monte_carlo_seconds_per_episode": mc_seconds,
        "speedup": mc_seconds / wm_seconds,
        "world_model_timings": wm_timings,
        "monte_carlo_timings": mc_timings,
        "wm_samples": args.wm_samples,
        "mc_episodes": args.mc_episodes,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    (args.out_dir / f"{task}_{dataset}.json").write_text(json.dumps(summary, indent=2))

    # the pipeline's own figure, fed the two medians, so every runtime figure looks alike
    rows = [
        {"evaluator": "world_model", "cost": {"rollout_seconds": wm_seconds * args.wm_samples, "n_samples": args.wm_samples}},
        {"evaluator": "monte_carlo", "cost": {"rollout_seconds": mc_seconds * args.mc_episodes, "mc_runs": args.mc_episodes}},
    ]
    plot_runtime(rows, args.out_dir / f"{task}_{dataset}.png", f"{dataset} ({graph.num_nodes:,} nodes)")

    print(f"  world model : {1000 * wm_seconds:9.2f} ms per sample   (median of {args.repeats} x {args.wm_samples} samples)")
    print(f"  Monte Carlo : {1000 * mc_seconds:9.2f} ms per episode  (median of {args.repeats} x {args.mc_episodes} NDlib episodes)")
    ratio = max(mc_seconds, wm_seconds) / min(mc_seconds, wm_seconds)
    print(f"  world model is {ratio:.1f}x {'faster' if mc_seconds > wm_seconds else 'SLOWER'} than Monte Carlo")
    print(f"  wrote {args.out_dir / f'{task}_{dataset}.json'} and .png")
