"""
Top-level driver: run the coding-agent outer loop over the inner-loop environment.

For a full dataset sweep (all arms x all budgets, plots and report) use
`python -m pipeline.run` instead; this module is the single-run entry point.

Baselines -

python -m coding_agent.run --data-dir results/ba40/data \
    --wm-results-json results/ba40/world_model/sage_IC.json \
    --method one_shot --strategy-mode free \
    --evaluator world_model --budget 5 --horizon 10 --compare \
    --baseline degree_discount --outer-iters 1 \
    --out-json results/ba40/agent/pct5.0/baseline_degree_discount.json


Graph algorithm routing (LLM picks from a list of graph algorithms, no synthesis) -

python -m coding_agent.run --data-dir results/ba40/data \
    --model gpt-5.6-terra \
    --wm-results-json results/ba40/world_model/sage_IC.json \
    --method one_shot --strategy-mode free \
    --evaluator world_model --budget 5 --horizon 10 --compare \
    --routing --outer-iters 1 \
    --out-json results/ba40/agent/pct5.0/routing.json


Oracle -

python -m coding_agent.run --data-dir results/ba40/data \
    --model gpt-5.6-terra \
    --method one_shot --strategy-mode free \
    --evaluator oracle --budget 5 --horizon 10 --compare \
    --allowed-ops add_node remove_node \
    --mc-runs 200 --n-samples 50 \
    --outer-iters 5 --out-json results/ba40/agent/pct5.0/one_shot_free.json


Scored mode + evolve (agent edits algorithm internals, population search) -

python -m coding_agent.run --data-dir results/sbm40/data \
    --model gpt-5.6-terra \
    --wm-results-json results/sbm40/world_model/sage_IC.json \
    --method evolve --strategy-mode scored \
    --evaluator world_model --budget 5 --horizon 10 --compare \
    --allowed-ops add_node remove_node \
    --outer-iters 10 --n-samples 50 \
    --out-json results/sbm40/agent/pct5.0/evolve_scored.json
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from dotenv import load_dotenv

from coding_agent.agent import CodingAgent, GatewayProvider
from coding_agent.credit import counterfactual_credit, planned_action
from coding_agent.envs.monte_carlo_env import MonteCarloEnvironment
from coding_agent.envs.world_model_env import WorldModelEnvironment
from coding_agent.methods.base import OuterLoopMethod, summarize
from coding_agent.methods.evolve import EvolveSearch
from coding_agent.methods.one_shot import OneShotSuperAlgorithm
from coding_agent.methods.per_step import PerStepReprompt
from coding_agent.methods.windowed import WindowedOnline
from coding_agent.prompts import build_routing_prompt, routing_system
from coding_agent.tools.library_api import algorithm_names
from coding_agent.types import GraphInfo, TaskSpec
from data.wm_simulator import valid_action_ops
from pipeline.conditions import parse_arm
from pipeline.layout import budget_label
from world_model.wm_data import load_graph_store

world_model = "world_model"
monte_carlo = "monte_carlo"
oracle = "oracle"

# Fresh-seed WM re-evaluations of the winning strategy during --compare
wm_reeval_seeds = 3

# Above this node count the per-node marginal vector is dropped from the results
# JSON — it would dominate the file (1M nodes ~ 7MB of floats per run)
max_serialized_marginals = 200_000


@dataclass
class ExperimentConfig:
    method: str = "one_shot"  # one_shot | per_step | windowed | evolve
    strategy_mode: str = (
        "free"  # free (whole Strategy) | scored (score/schedule hooks only)
    )
    evaluator: str = "world_model"  # world_model | monte_carlo | oracle
    model: str = "gpt-5.6-terra"  # gateway model name
    temperature: float | None = (
        None  # LLM sampling temperature; None -> provider default
    )
    diffusion_model: str = "IC"  # IC | LT
    budget: int = 5
    budget_pct: float | None = None  # overrides budget: % of the graph's num_nodes
    horizon: int = 10
    windows: int = 3
    outer_iters: int = 3
    mc_runs: int = 200
    # Runs for the --compare ground-truth replay; None -> mc_runs. Kept separate
    # so a native arm (mc_runs=1 inner loop) is still judged on a clean average
    referee_mc_runs: int | None = None
    n_samples: int = 20
    seed: int = 42
    device: str = "cpu"
    data_dir: str | None = None  # world_model.wm_data graph store dir
    graph_id: str | None = None  # which graph in the store (default: first)
    wm_results_json: str | None = None  # train_wm.py results JSON (for the WM env)
    compare: bool = False  # also evaluate the winning strategy on the MC baseline
    credit: bool = False  # per-action counterfactual credit (feedback + results)
    baseline: str | None = None  # library algorithm name; evaluates it with no LLM
    routing: bool = False  # GA routing: one LLM call picks a library algorithm
    allowed_ops: tuple = valid_action_ops  # ops the strategy may emit
    out_json: str | None = None
    # Arm spec this run represents; stamps the condition metadata into the
    # results JSON so a standalone run is readable by plots/report exactly like
    # a pipeline-produced one. The pipeline stamps these itself and leaves this
    # unset.
    arm_spec: str | None = None


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

    if name == "evolve":
        return EvolveSearch()

    raise ValueError(
        f"unknown method {name!r}; choose one_shot|per_step|windowed|evolve"
    )


def _load_graph(config: ExperimentConfig) -> tuple[GraphInfo, str]:
    if config.data_dir is None:
        raise ValueError(
            "config.data_dir is required unless a graph is passed directly"
        )

    store = load_graph_store(config.data_dir)
    graph_id = config.graph_id or next(iter(store))

    return GraphInfo.from_store_entry(store[graph_id]), graph_id


def _build_environment(config: ExperimentConfig, graph: GraphInfo) -> object:
    if config.evaluator == monte_carlo:
        return MonteCarloEnvironment(
            graph, config.diffusion_model, mc_runs=config.mc_runs
        )

    if config.evaluator == world_model:
        if config.wm_results_json is None:
            raise ValueError("evaluator=world_model requires config.wm_results_json")

        return WorldModelEnvironment.from_results_json(
            config.wm_results_json,
            graph,
            device=config.device,
            n_samples=config.n_samples,
        )

    if config.evaluator == oracle:
        # Ground-truth dynamics ceiling: no checkpoint, no --wm-results-json
        return WorldModelEnvironment.oracle(
            graph,
            config.diffusion_model,
            device=config.device,
            n_samples=config.n_samples,
        )

    raise ValueError(f"unknown evaluator {config.evaluator!r}")


def _parse_routing_choice(reply: str) -> str:
    cleaned = reply.strip().strip("`'\".")

    if cleaned in algorithm_names:
        return cleaned

    # Models sometimes wrap the name in prose; accept iff exactly one menu name appears
    mentioned = [name for name in algorithm_names if re.search(rf"\b{name}\b", reply)]
    if len(mentioned) == 1:
        return mentioned[0]

    raise ValueError(
        f"routing reply must name exactly one library algorithm, got {reply!r} "
        f"(matched: {mentioned})"
    )


def _strategy_action(
    strategy: object, graph: GraphInfo, state: object, timestep: int
) -> list:
    return strategy.act(state, graph, timestep)


def run_experiment(
    config: ExperimentConfig,
    graph: GraphInfo | None = None,
    canned_script: str | None = None,
) -> dict:
    experiment_start = time.perf_counter()

    graph_id = config.graph_id
    if graph is None:
        graph, graph_id = _load_graph(config)

    # Same resolution rule as data generation: pct of N overrides the absolute k
    if config.budget_pct is not None:
        config.budget = max(1, round(graph.num_nodes * config.budget_pct / 100))

    print(
        f"[run] graph {graph_id or 'inline'}: {graph.num_nodes} nodes, "
        f"{graph.edge_index.shape[1]} arcs | evaluator={config.evaluator} "
        f"method={config.method} mode={config.strategy_mode} "
        f"budget={config.budget} ({100.0 * config.budget / graph.num_nodes:.2f}% "
        f"of N) horizon={config.horizon}"
    )

    environment = _build_environment(config, graph)
    print(f"[run] {config.evaluator} environment ready")

    task = TaskSpec(
        diffusion_model=config.diffusion_model,
        budget=config.budget,
        horizon=config.horizon,
        allowed_ops=tuple(config.allowed_ops),
    )

    # GA routing: one LLM call selects from the algorithm pool (no synthesis),
    # then the pick runs through the identical --baseline canned path below
    routing_reply = None
    if config.routing:
        if config.baseline is not None:
            raise ValueError("--routing and --baseline are mutually exclusive")

        router = GatewayProvider(config.model, temperature=config.temperature)
        routing_reply = router.complete(
            routing_system, build_routing_prompt(task, graph)
        )
        config.baseline = _parse_routing_choice(routing_reply)
        print(f"[run] routing picked {config.baseline!r}")

    if canned_script is not None:
        provider_label = "canned"
    elif config.baseline is not None:
        # Classical-library baseline: same pipeline, envs, and metrics — no LLM
        if config.baseline not in algorithm_names:
            raise ValueError(
                f"unknown baseline {config.baseline!r}; choose one of {algorithm_names}"
            )

        # For classical baseline, construct a canned one-shot script that calls algorithms.<name>(graph, budget, dynamics, horizon=horizon)
        # Seeds at t=0
        provider_label = (
            f"routing:{config.baseline}"
            if config.routing
            else f"baseline:{config.baseline}"
        )
        canned_script = f"""\
class Baseline(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        seeds = algorithms.{config.baseline}(
            graph, budget, "{config.diffusion_model}", horizon=horizon
        )
        return [[ActionOp("add_node", node) for node in seeds]] + [
            [] for _ in range(horizon)
        ]
"""
    else:
        provider_label = config.model

    # Canned/baseline scripts are whole free-form Strategies (they call
    # algorithms.*), so they always run in free mode regardless of the flag
    effective_mode = "free" if canned_script is not None else config.strategy_mode
    if effective_mode == "scored" and config.method in ("per_step", "windowed"):
        raise ValueError(
            "strategy_mode='scored' generates plan_horizon-only strategies; "
            "use --method one_shot or evolve"
        )

    provider = (
        _CannedProvider(canned_script)
        if canned_script
        else GatewayProvider(config.model, temperature=config.temperature)
    )
    agent = CodingAgent(provider)

    method = build_method(config.method)
    if config.method == "windowed":
        method = WindowedOnline(windows=config.windows)

    if config.method == "one_shot":
        method = OneShotSuperAlgorithm(
            outer_iters=config.outer_iters,
            credit=config.credit,
            strategy_mode=effective_mode,
            # A canned script ignores its prompt, so the anchor rollout would only
            # burn real episodes and wall clock without informing anything
            use_anchor=canned_script is None,
        )

    if config.method == "evolve":
        method = EvolveSearch(
            outer_iters=config.outer_iters, strategy_mode=effective_mode
        )

    # Optimize the method with the outer-loop coding agent iteration loop to find the best strategy and trajectory result
    print(f"[run] optimizing with {config.method} (provider {provider_label})...")
    strategy, trajectory = method.optimize(agent, environment, task, graph)
    print(
        f"[run] winner: reward={trajectory.reward:.2f} "
        f"({100.0 * trajectory.reward / graph.num_nodes:.2f}% of N)"
    )

    # Per-timestep log of the representative rollout (first ensemble sample):
    # entry t holds the action bag applied at t and the resulting state
    # The reward is the ensemble MEAN, so this single sample's final infected count need not equal it exactly
    timeline = [
        {"t": timestep, "actions": [action.to_dict() for action in bag]}
        | state.to_dict()
        for timestep, (bag, state) in enumerate(
            zip(trajectory.actions, trajectory.states[1:], strict=True)
        )
    ]

    result = {
        "method": config.method,
        "evaluator": config.evaluator,
        "model": provider_label,
        # num_edges counts directed arcs (edge_index columns), matching the
        # graph stats shown in the agent prompts
        "graph": {
            "graph_id": graph_id,
            "num_nodes": graph.num_nodes,
            "num_edges": int(graph.edge_index.shape[1]),
            "directed": graph.directed,
        },
        "budget": config.budget,
        "budget_pct": round(100.0 * config.budget / graph.num_nodes, 3),
        "reward": trajectory.reward,
        "spread_pct": round(100.0 * trajectory.reward / graph.num_nodes, 2),
        "summary": summarize(trajectory, graph),
        # For per_step this is the last timestep's script (one is generated per step)
        "script": strategy.source_script,
        "cost": trajectory.cost,
        # Per-outer-iteration rewards (empty for baseline/routing arms)
        "history": getattr(method, "history", []),
        # Per-node P(infected at end) across the ensemble. Costs n_samples
        # rollouts to produce, so it is serialized rather than recomputed — it
        # is what any post-hoc spatial analysis (coverage, per-community reach)
        # needs. Suppressed on very large graphs where the list dominates the file.
        "final_marginals": (
            trajectory.final_marginals
            if trajectory.final_marginals is not None
            and graph.num_nodes <= max_serialized_marginals
            else None
        ),
        # Inner-loop real-environment episodes only (0 for model-based
        # evaluators); the --compare referee replay is deliberately excluded
        "real_env_episodes": getattr(environment, "episodes_used", 0),
        "timeline": timeline,
    }

    if routing_reply is not None:
        result["routing_reply"] = routing_reply

    # When the credit flag is enabled, ablate the executed action sequence against the same environment and measure the reward
    if config.credit:
        print("[run] per-action counterfactual credit (one rollout per action)...")
        # Credit of the executed action sequence; for state-dependent strategies (per_step/windowed) the recorded bags are replayed as a fixed plan
        base_reward, entries = counterfactual_credit(
            environment, trajectory.actions, config.horizon, config.budget
        )
        result["credit_base_reward"] = base_reward
        result["credit"] = entries

    # When the compare flag is enabled, build a Monte Carlo environment and rollout with Monte Carlo simulation to compare against the world model
    # Every evaluator gets this replay, not just the model-based ones: it is the
    # single ground-truth referee that makes rewards comparable ACROSS conditions.
    # A native arm's own reward is one noisy episode; a monte_carlo arm's carries
    # the winner's curse from being the max over outer iterations.
    if config.compare:
        referee_runs = config.referee_mc_runs or config.mc_runs
        result["referee_mc_runs"] = referee_runs
        print(f"[run] MC compare replay ({referee_runs} runs)...")

        mc_environment = MonteCarloEnvironment(
            graph, config.diffusion_model, mc_runs=referee_runs
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
        result["mc_spread_pct"] = round(
            100.0 * mc_trajectory.reward / graph.num_nodes, 2
        )
        result["mc_reward_se"] = mc_trajectory.cost["reward_se"]
        result["mc_rollout_seconds"] = mc_trajectory.cost["rollout_seconds"]
        result["wm_minus_mc"] = trajectory.reward - mc_trajectory.reward
        print(
            f"[run] mc_reward={mc_trajectory.reward:.2f} "
            f"±{mc_trajectory.cost['reward_se']:.2f} "
            f"(wm_minus_mc={result['wm_minus_mc']:+.2f})"
        )
        # Fidelity re-evaluation only means something for a model-based evaluator:
        # it measures how far the MODEL is from truth. For a monte_carlo evaluator
        # the "model" is the simulator itself, so there is nothing to measure.
        if config.evaluator in (world_model, oracle):
            print(
                f"[run] re-evaluating winner on {wm_reeval_seeds} fresh "
                f"{config.evaluator} seeds..."
            )

            # `reward` is the max over outer iterations, all evaluated at rollout
            # seed 0 — it carries selection optimism (winner's curse) plus that one
            # seed's persistent luck. Re-evaluating the winner on fresh seeds gives
            # the unbiased WM estimate: judge evaluator fidelity by
            # wm_reeval_minus_mc, not wm_minus_mc.
            reeval_rewards = [
                environment.rollout(
                    action_fn, config.horizon, config.budget, seed=reeval_seed
                ).reward
                for reeval_seed in range(1, wm_reeval_seeds + 1)
            ]
            result["wm_reeval_rewards"] = reeval_rewards
            result["wm_reeval_mean"] = sum(reeval_rewards) / len(reeval_rewards)
            result["wm_reeval_minus_mc"] = (
                result["wm_reeval_mean"] - mc_trajectory.reward
            )
            print(
                f"[run] wm_reeval_mean={result['wm_reeval_mean']:.2f} "
                f"(reeval_minus_mc={result['wm_reeval_minus_mc']:+.2f})"
            )

    # Whole experiment including LLM calls; the per-rollout WM-vs-MC timing lives in cost.rollout_seconds / mc_rollout_seconds
    result["elapsed_seconds"] = time.perf_counter() - experiment_start

    if config.arm_spec:
        arm = parse_arm(config.arm_spec, default_evaluator=config.evaluator)
        result["arm"] = arm.name
        result["arm_spec"] = arm.spec
        result["condition"] = arm.condition
        result["condition_name"] = arm.condition_name
        result["budget_label"] = budget_label(config.budget_pct, config.budget)

    if config.out_json:
        os.makedirs(Path(config.out_json).parent, exist_ok=True)
        Path(config.out_json).write_text(json.dumps(result, indent=2, default=str))
        print(f"[run] results -> {config.out_json}")

    print(f"[run] done in {result['elapsed_seconds']:.1f}s")

    return result


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Coding-agent outer loop over the world model"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.6-terra",
        help="gateway model name, e.g. gpt-5.6-sol (default: gpt-5.6-terra).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="LLM sampling temperature; 0.0 = greedy decoding, omit for the provider default (default: None).",
    )
    parser.add_argument(
        "--allowed-ops",
        type=str,
        nargs="+",
        default=list(valid_action_ops),
        choices=list(valid_action_ops),
        help="action ops the strategy may emit; e.g. add_node remove_node for a "
        "node-ops-only run (default: all five ops).",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        choices=algorithm_names,
        help="evaluate this classical library algorithm instead of an LLM strategy (default: None).",
    )
    parser.add_argument(
        "--routing",
        action="store_true",
        help="GA-routing baseline: one LLM call picks a library algorithm from the menu, then it runs through the --baseline canned path (default: False).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="one_shot",
        choices=["one_shot", "per_step", "windowed", "evolve"],
        help="outer-loop method; evolve = population edits with refine/restructure operators (default: one_shot).",
    )
    parser.add_argument(
        "--strategy-mode",
        type=str,
        default="free",
        choices=["free", "scored"],
        help="what the agent writes: free = whole Strategy program; scored = only score()/schedule() hooks inside the fixed ScoredStrategy harness, no algorithms.* (default: free).",
    )
    parser.add_argument(
        "--evaluator",
        type=str,
        default="world_model",
        choices=["world_model", "monte_carlo", "oracle"],
        help="inner-loop evaluator; oracle = true IC dynamics, no checkpoint needed (default: world_model).",
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
        "--budget-pct",
        type=float,
        default=None,
        help="seed budget as a percent of the graph's num_nodes; overrides --budget, e.g. 1 -> k=16 on netscience (default: None).",
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
        default=200,
        help="Monte Carlo simulator runs; ~(spread_std/target_se)^2, dial down for large graphs with --evaluator monte_carlo (default: 200).",
    )
    parser.add_argument(
        "--referee-mc-runs",
        type=int,
        default=None,
        help="runs for the --compare ground-truth replay; keep this high even when --mc-runs is 1 for a native-agent condition (default: --mc-runs).",
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
        help="per-action counterfactual credit: ablate each action, report its delta-spread in refinement feedback and the results JSON; costs one extra rollout per action (default: False).",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="output JSON path (default: <data-dir>/../agent/<budget>/<arm>.json, "
        "the same slot the pipeline writes).",
    )

    args = parser.parse_args()

    # A standalone run costs the same LLM calls as a pipeline run, so it lands in
    # the same slot by default instead of printing to stdout and evaporating
    if args.baseline:
        arm_spec = f"baseline:{args.baseline}"
    elif args.routing:
        arm_spec = "routing"
    else:
        arm_spec = f"{args.method}_{args.strategy_mode}@{args.evaluator}"

    if args.out_json is None and args.data_dir:
        arm = parse_arm(arm_spec, default_evaluator=args.evaluator)
        args.out_json = str(
            Path(args.data_dir).resolve().parent
            / "agent"
            / budget_label(args.budget_pct, args.budget)
            / f"{arm.name}.json"
        )

    config = ExperimentConfig(
        model=args.model,
        temperature=args.temperature,
        baseline=args.baseline,
        routing=args.routing,
        allowed_ops=tuple(args.allowed_ops),
        method=args.method,
        strategy_mode=args.strategy_mode,
        evaluator=args.evaluator,
        diffusion_model=args.diffusion_model,
        budget=args.budget,
        budget_pct=args.budget_pct,
        horizon=args.horizon,
        windows=args.windows,
        outer_iters=args.outer_iters,
        mc_runs=args.mc_runs,
        referee_mc_runs=args.referee_mc_runs,
        n_samples=args.n_samples,
        seed=args.seed,
        device=args.device,
        data_dir=args.data_dir,
        graph_id=args.graph_id,
        wm_results_json=args.wm_results_json,
        compare=args.compare,
        credit=args.credit,
        out_json=args.out_json,
        arm_spec=arm_spec,
    )

    print(json.dumps(run_experiment(config), indent=2, default=str))
