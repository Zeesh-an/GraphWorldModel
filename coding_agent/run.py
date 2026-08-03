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

from coding_agent import checkpoint, executor
from coding_agent.agent import (
    CodingAgent,
    GatewayProvider,
    empty_usage,
    merge_usage,
)
from coding_agent.containment import (
    outbreak_selectors,
    removal_set,
    select_outbreak,
)
from coding_agent.credit import counterfactual_credit, planned_action
from coding_agent.envs.monte_carlo_env import MonteCarloEnvironment
from coding_agent.envs.multi_round_env import MultiRoundEnvironment
from coding_agent.envs.world_model_env import WorldModelEnvironment
from coding_agent.methods.base import OuterLoopMethod, summarize
from coding_agent.methods.evolve import EvolveSearch
from coding_agent.methods.one_shot import OneShotSuperAlgorithm
from coding_agent.methods.per_step import PerStepReprompt
from coding_agent.methods.windowed import WindowedOnline
from coding_agent.prompts import (
    build_explanation_prompt,
    build_routing_prompt,
    build_routing_system,
)
from coding_agent.rounds import (
    describe_schedule,
    round_batches,
    round_schedule,
    round_spreads,
)
from coding_agent.tools.adaptive_algorithms import adaptive_algorithms
from coding_agent.tools.dismantling_algorithms import dismantling_algorithms
from coding_agent.tools.library_api import algorithm_names, dismantling_names
from coding_agent.types import GraphInfo, TaskSpec, full_adoption, valid_feedback_models
from data.wm_simulator import spent, valid_action_ops, valid_remove_semantics
from pipeline.conditions import parse_arm
from pipeline.layout import budget_label, checkpoint_suffix
from pipeline.tasks import get_task, maximize
from world_model.wm_data import load_graph_store
from world_model.wm_metrics import containment_metrics

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
    # Key of pipeline.tasks.tasks; reaches the agent through TaskSpec, so the
    # prompt names the problem the arm is actually being scored on
    task: str = "influence_maximization"
    method: str = "one_shot"  # one_shot | per_step | windowed | evolve | adaptive
    strategy_mode: str = (
        "free"  # free (whole Strategy) | scored (score/schedule hooks only)
    )
    evaluator: str = "world_model"  # world_model | monte_carlo | oracle
    model: str = "gpt-5.6-terra"  # gateway model name
    temperature: float | None = (
        None  # LLM sampling temperature; None -> provider default
    )
    diffusion_model: str = "IC"  # IC | LT
    # What remove_node does. Must match the checkpoint's for the world_model
    # evaluator, which reads it back from the results JSON.
    remove_semantics: str = spent
    budget: int = 5
    budget_pct: float | None = None  # overrides budget: % of the graph's num_nodes
    horizon: int = 10
    windows: int = 3
    # Adaptive IM (method="adaptive"): k is committed in `rounds` batches, each
    # chosen after seeing what the previous one activated. Ignored by every other
    # method, so a single sweep can hold adaptive and non-adaptive arms together.
    rounds: int = 4
    per_round_budget: int | None = None  # fix b and derive r instead of fixing r
    round_gap: int = 1  # timesteps of diffusion between rounds
    feedback_model: str = full_adoption
    # Dynamic / streaming IM: exogenous edge edits per timestep as a fraction of
    # |E|, applied to every arm. 0 = the static graph.
    edit_rate: float = 0.0
    # Multi-round IM: r SEPARATE campaigns of k seeds, scored on their union.
    # 1 = a single campaign.
    campaigns: int = 1
    # Critical node detection: the exogenous outbreak the planner is containing,
    # as a percentage of N and the rule that picks its sources. None -> the task
    # registry's own value, which is 0 for every seeding task (the planner starts
    # its own cascade there). Deterministic in --seed, so every arm in a sweep
    # fights the SAME outbreak — an arm facing a different one would be measuring
    # the outbreak, not the method.
    outbreak_pct: float | None = None
    outbreak_selector: str = "random"
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
    # Re-expose the per-candidate-simulation algorithms (celf, vanilla_greedy,
    # ...) to generated scripts; see executor.mc_blocked_algorithms
    allow_mc_algorithms: bool = False
    # Wall-clock cap on one generated plan_horizon()/act() call; 0 disables
    strategy_timeout: float = executor.strategy_timeout_seconds
    # USD per 1M tokens, for the cost line in the results JSON. The lab gateway
    # bills nothing per token (it fronts Pro subscriptions), so there is no rate
    # to hardcode — supply your own or the cost stays null while tokens are
    # still counted exactly.
    llm_price_in: float | None = None
    llm_price_out: float | None = None
    # Resume a killed search from its checkpoint; --force turns this off so a
    # forced re-run is genuinely fresh
    resume: bool = True
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
        # Present so usage accounting never special-cases the canned path
        self.usage = empty_usage()

    def complete(self, messages: list[dict]) -> str:
        self.usage["calls"] += 1

        return f"```python\n{self.script}\n```"


def _usage_report(providers: list, config: ExperimentConfig) -> dict:
    """Token totals for the run, and a cost only when a price was supplied."""
    usage = merge_usage(providers)
    prices = (config.llm_price_in, config.llm_price_out)

    usage["cost_usd"] = (
        None
        if any(price is None for price in prices)
        else round(
            usage["prompt_tokens"] / 1e6 * config.llm_price_in
            + usage["completion_tokens"] / 1e6 * config.llm_price_out,
            6,
        )
    )

    return usage


def build_method(
    config: "ExperimentConfig",
    strategy_mode: str,
    allow_mc_algorithms: bool,
    use_anchor: bool = True,
    checkpoint_path: Path | None = None,
    checkpoint_fingerprint: dict | None = None,
) -> OuterLoopMethod:
    """
    The single construction site for every method.

    `strategy_mode` / `allow_mc_algorithms` are the resolved values, not
    config's: a canned script overrides both. Only the two methods with a
    refinement loop take a checkpoint — per_step and windowed have nothing to
    resume.
    """
    if config.method == "one_shot":
        return OneShotSuperAlgorithm(
            outer_iters=config.outer_iters,
            credit=config.credit,
            strategy_mode=strategy_mode,
            # A canned script ignores its prompt, so the anchor rollouts would only
            # burn real episodes and wall clock without informing anything
            use_anchor=use_anchor,
            allow_mc_algorithms=allow_mc_algorithms,
            checkpoint_path=checkpoint_path,
            checkpoint_fingerprint=checkpoint_fingerprint,
        )

    # per_step has no refinement loop — it is one episode with an LLM call per
    # (sample, timestep), so outer_iters and credit do not apply to it
    if config.method == "per_step":
        return PerStepReprompt(allow_mc_algorithms=allow_mc_algorithms)

    if config.method == "windowed":
        return WindowedOnline(
            windows=config.windows, allow_mc_algorithms=allow_mc_algorithms
        )

    # Same population search for both: `adaptive` differs only in what the
    # generated program is (a per-round policy vs a static plan), which
    # methods.base.evaluate_strategy dispatches on via TaskSpec.rounds
    if config.method in ("evolve", "adaptive"):
        return EvolveSearch(
            outer_iters=config.outer_iters,
            strategy_mode=strategy_mode,
            label=config.method,
            allow_mc_algorithms=allow_mc_algorithms,
            use_anchor=use_anchor,
            checkpoint_path=checkpoint_path,
            checkpoint_fingerprint=checkpoint_fingerprint,
        )

    raise ValueError(
        f"unknown method {config.method!r}; "
        f"choose one_shot|per_step|windowed|evolve|adaptive"
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
    # --seed is the base seed for every rollout in this run. Shared across
    # candidates on purpose (common random numbers), and recorded per rollout in
    # Trajectory.cost["seed"] so any single number can be replayed exactly.
    if config.evaluator == monte_carlo:
        return MonteCarloEnvironment(
            graph,
            config.diffusion_model,
            mc_runs=config.mc_runs,
            base_seed=config.seed,
            remove_semantics=config.remove_semantics,
        )

    if config.evaluator == world_model:
        if config.wm_results_json is None:
            raise ValueError("evaluator=world_model requires config.wm_results_json")

        # Semantics comes from the checkpoint, not from --remove-semantics: the
        # head's T_exo was fixed at training time and cannot be reinterpreted here
        environment = WorldModelEnvironment.from_results_json(
            config.wm_results_json,
            graph,
            device=config.device,
            n_samples=config.n_samples,
            base_seed=config.seed,
        )

        # Everything else in the run (the prompt, the --compare referee) follows
        # config.remove_semantics, so a disagreement would have the arm plan under
        # one reading and be scored under the other
        if environment.remove_semantics != config.remove_semantics:
            raise ValueError(
                f"checkpoint {config.wm_results_json} was trained with "
                f"remove_semantics={environment.remove_semantics!r} but this run "
                f"is configured for {config.remove_semantics!r}; pass "
                f"--remove-semantics {environment.remove_semantics} or use a "
                f"checkpoint trained for {config.remove_semantics}"
            )

        return environment

    if config.evaluator == oracle:
        # Ground-truth dynamics ceiling: no checkpoint, no --wm-results-json
        return WorldModelEnvironment.oracle(
            graph,
            config.diffusion_model,
            device=config.device,
            n_samples=config.n_samples,
            base_seed=config.seed,
            remove_semantics=config.remove_semantics,
        )

    raise ValueError(f"unknown evaluator {config.evaluator!r}")


def _parse_routing_choice(reply: str, menu: list[str] = algorithm_names) -> str:
    cleaned = reply.strip().strip("`'\".")

    if cleaned in menu:
        return cleaned

    # Models sometimes wrap the name in prose; accept iff exactly one menu name
    # appears. Longest first, so `adaptive_degree` is not shadowed by a shorter
    # name that happens to be its substring.
    mentioned = [
        name
        for name in sorted(menu, key=len, reverse=True)
        if re.search(rf"\b{name}\b", reply)
    ]
    if len(mentioned) == 1:
        return mentioned[0]

    raise ValueError(
        f"routing reply must name exactly one library algorithm, got {reply!r} "
        f"(matched: {mentioned})"
    )


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

    # Module-level rather than threaded through four methods: it is a machine
    # limit on generated code, not a property of the task or the search
    executor.strategy_timeout_seconds = config.strategy_timeout

    environment = _build_environment(config, graph)

    if config.campaigns > 1:
        # Wraps any evaluator: multi-round is a scoring change (r separate
        # diffusions, union objective), not a dynamics change, so it composes
        # with monte_carlo / oracle / world_model without touching any of them
        environment = MultiRoundEnvironment(
            environment, campaigns=config.campaigns, base_seed=config.seed
        )

    print(f"[run] {config.evaluator} environment ready")

    # rounds=None for every non-adaptive method, which is what TaskSpec.adaptive
    # reads: the round machinery is inert unless this arm asked for it
    adaptive = config.method == "adaptive"
    registry = get_task(config.task)

    # The exogenous outbreak, resolved BEFORE the arm runs and derived only from
    # (graph, size, selector, seed), so every arm in the sweep fights the same one
    outbreak_pct = (
        registry.outbreak_pct if config.outbreak_pct is None else config.outbreak_pct
    )
    outbreak = (
        tuple(
            select_outbreak(
                graph,
                max(1, round(graph.num_nodes * outbreak_pct / 100)),
                config.outbreak_selector,
                config.seed,
            )
        )
        if outbreak_pct
        else ()
    )

    task = TaskSpec(
        task=config.task,
        objective=registry.summary,
        diffusion_model=config.diffusion_model,
        budget=config.budget,
        horizon=config.horizon,
        allowed_ops=tuple(config.allowed_ops),
        remove_semantics=config.remove_semantics,
        # From the registry, never from a flag: the objective sign and what a unit
        # of budget buys are properties of the TASK, and a run that disagreed with
        # its own registry entry would optimize one thing and be reported as another
        sense=registry.objective if registry.objective in (maximize, "minimize") else maximize,
        budget_op=registry.budget_op,
        outbreak=outbreak,
        rounds=config.rounds if adaptive else None,
        per_round_budget=config.per_round_budget if adaptive else None,
        round_gap=config.round_gap,
        feedback_model=config.feedback_model,
        # Both apply to every method, adaptive or not: a stream only one arm sees
        # and a union only one arm is scored on are not comparisons
        edit_rate=config.edit_rate,
        campaigns=config.campaigns,
        seed=config.seed,
    )

    if outbreak:
        print(
            f"[run] outbreak: {len(outbreak)} source(s) "
            f"({outbreak_pct:g}% of N, selector={config.outbreak_selector}, "
            f"seed={config.seed}) — MINIMIZING final infected count"
        )

    batches = (
        round_batches(task.budget, task.rounds, task.per_round_budget)
        if task.adaptive
        else None
    )
    if batches is not None:
        print(f"[run] adaptive: {describe_schedule(task, batches)}")

    # GA routing: one LLM call selects from the algorithm pool (no synthesis),
    # then the pick runs through the identical --baseline canned path below
    routing_reply = None
    # Every provider this run creates, so the token totals cover the routing call
    providers = []

    if config.routing:
        if config.baseline is not None:
            raise ValueError("--routing and --baseline are mutually exclusive")

        router = GatewayProvider(config.model, temperature=config.temperature)
        providers.append(router)
        routing_reply = router.complete(
            [
                {"role": "system", "content": build_routing_system(task)},
                {"role": "user", "content": build_routing_prompt(task, graph)},
            ]
        )
        # A containment task's menu is the DISMANTLING pool; routing into the IM
        # pool would return a seed set the executor then rejects
        config.baseline = _parse_routing_choice(
            routing_reply, dismantling_names if task.contains else algorithm_names
        )
        print(f"[run] routing picked {config.baseline!r}")

    if canned_script is not None:
        provider_label = "canned"
    elif config.baseline is not None:
        # Classical-library baseline: same pipeline, envs, and metrics — no LLM
        if config.baseline not in (
            algorithm_names + list(adaptive_algorithms) + dismantling_names
        ):
            raise ValueError(
                f"unknown baseline {config.baseline!r}; choose a static algorithm "
                f"from {algorithm_names}, an adaptive policy from "
                f"{sorted(adaptive_algorithms)}, or a dismantler from "
                f"{dismantling_names}"
            )

        provider_label = (
            f"routing:{config.baseline}"
            if config.routing
            else f"baseline:{config.baseline}"
        )

        if config.baseline in adaptive_algorithms:
            if batches is None:
                raise ValueError(
                    f"baseline {config.baseline!r} is a per-round adaptive policy, "
                    f"but this arm is not adaptive. parse_arm marks adaptive "
                    f"baselines with method='adaptive'; a hand-built "
                    f"ExperimentConfig must set method='adaptive' too."
                )

            # A published adaptive algorithm is a per-round POLICY: act() runs
            # once per round on the realized state. The schedule is inlined rather
            # than passed, because act() receives only the timestep and the batch
            # size varies between rounds when k does not divide evenly by r.
            # total_budget is for static_split, which needs to know how long its
            # static ranking should be.
            schedule = round_schedule(batches, config.round_gap, config.horizon)
            canned_script = f"""\
class AdaptiveBaseline(Strategy):
    def act(self, state, graph, timestep):
        batch = {schedule!r}.get(timestep, 0)
        if not batch:
            return []
        return [
            ActionOp("add_node", node)
            for node in adaptive_algorithms.{config.baseline}(
                state,
                graph,
                batch,
                "{config.diffusion_model}",
                total_budget={config.budget},
            )
        ]
"""
        elif config.baseline in dismantling_algorithms:
            # A dismantler is a static REMOVAL set, committed at t=0. It is handed
            # the outbreak because the simulation-based member scores candidates
            # against it; the structural members take **kw and ignore it.
            # `removal_plan` then drops any source it picked anyway and tops the
            # set back up, which is what keeps a published algorithm runnable
            # without rewriting it to know an outbreak exists.
            canned_script = f"""\
class DismantlingBaseline(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        removals = dismantling_algorithms.{config.baseline}(
            graph,
            budget,
            "{config.diffusion_model}",
            horizon=horizon,
            outbreak={tuple(outbreak)!r},
        )
        return containment.removal_plan(
            removals, graph, budget, {tuple(outbreak)!r}, horizon
        )
"""
        else:
            # A classical baseline is a static seed set, committed at t=0
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

    # A declared baseline (condition 1) or a routing pick (condition 2) IS the
    # expensive algorithm — blocking it would delete the arm rather than speed it
    # up, and its cost is honestly attributed to that arm. The block governs what
    # the agent SYNTHESIZES, not what the harness was told to run.
    effective_allow_mc = config.allow_mc_algorithms or canned_script is not None
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
    providers.append(provider)
    agent = CodingAgent(provider)

    # Beside the result this run will become; a file here means "did not finish".
    # Canned arms are a single deterministic rollout with nothing to resume.
    checkpoint_path = (
        None
        if canned_script is not None or config.out_json is None
        else Path(config.out_json).with_suffix(checkpoint_suffix)
    )
    if not config.resume:
        checkpoint.clear(checkpoint_path)

    method = build_method(
        config,
        effective_mode,
        effective_allow_mc,
        use_anchor=canned_script is None,
        checkpoint_path=checkpoint_path,
        checkpoint_fingerprint=(
            None
            if checkpoint_path is None
            else checkpoint.fingerprint(config, config.method, graph)
        ),
    )

    # Optimize the method with the outer-loop coding agent iteration loop to find the best strategy and trajectory result
    print(f"[run] optimizing with {config.method} (provider {provider_label})...")
    strategy, trajectory = method.optimize(agent, environment, task, graph)
    print(
        f"[run] winner: reward={trajectory.reward:.2f} "
        f"({100.0 * trajectory.reward / graph.num_nodes:.2f}% of N"
        f"{', lower is better' if task.contains else ''})"
    )

    # One closing turn on the generation thread: what it tried each iteration and
    # how the winner works. Canned arms (classical baselines, routing picks) are
    # library algorithms nobody synthesized, and _CannedProvider would answer any
    # question with the script itself — so they get no write-up.
    explanation = None
    if canned_script is None and hasattr(method, "conversation"):
        print("[run] requesting the plain-English algorithm write-up...")
        explanation = method.conversation.ask(
            build_explanation_prompt(
                strategy.source_script,
                trajectory.reward,
                getattr(method, "history", []),
                summarize(trajectory, graph, task),
            )
        )
        print(f"\n{'=' * 78}\nALGORITHM WRITE-UP\n{'=' * 78}\n{explanation}\n")

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
        "task": config.task,
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
        # Seeds actually committable per episode. Equals `budget` for every method
        # except windowed, whose budget is per window call by design — without
        # this the sweep table reads two different budgets as the same k.
        "effective_budget": getattr(method, "effective_budget", None) or config.budget,
        "reward": trajectory.reward,
        "spread_pct": round(100.0 * trajectory.reward / graph.num_nodes, 2),
        "summary": summarize(trajectory, graph, task),
        # For per_step this is the last timestep's script (one is generated per step)
        "script": strategy.source_script,
        # Markdown; None for canned arms, which synthesized nothing to explain
        "explanation": explanation,
        # Every turn verbatim, prose and all. The code extractor keeps only the
        # fenced block, but the prose around it is where the model says what it
        # was trying to do — irrecoverable afterwards, and the first thing worth
        # reading when a run goes wrong. Empty for canned arms.
        "llm_transcript": (
            []
            if canned_script is not None
            else getattr(getattr(method, "conversation", None), "transcript", [])
        ),
        # calls + token totals across every provider this run created (the
        # routing call included). cost_usd is None unless --llm-price-in/-out
        # were given: the lab gateway bills nothing per token, so there is no
        # rate to assume.
        "llm_usage": _usage_report(providers, config),
        "cost": trajectory.cost,
        # Base seed for every rollout in this run; each rollout also records the
        # seed it actually used in its own cost block
        "seed": config.seed,
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
        # Inner-loop cost, and the reason this block sits ABOVE the --credit and
        # --compare sections rather than below them: both call rollout() again on
        # this same environment for post-hoc analysis, and counting those would
        # charge the search for work it did not do. The referee replay is excluded
        # for the same reason, and separately by using its own environment.
        #
        # real_env_episodes  simulator episodes consumed; 0 for world_model/oracle,
        #                    the sample-efficiency axis for the native arm
        # evaluator_calls    how many times the search queried its evaluator
        # evaluator_seconds  wall clock spent inside it, the axis the six-condition
        #                    cost claim is actually read on, since elapsed_seconds
        #                    is dominated by LLM latency
        # forward_passes     the model-based unit of work (0 for monte_carlo)
        "real_env_episodes": getattr(environment, "episodes_used", 0),
        "evaluator_calls": getattr(environment, "rollout_calls", 0),
        "evaluator_seconds": round(getattr(environment, "evaluator_seconds", 0.0), 3),
        "forward_passes": getattr(environment, "forward_passes", 0),
        "timeline": timeline,
        # sigma(S, T) at every T <= horizon, ensemble-mean and padded, so a
        # horizon-conditional number is readable instead of only the endpoint
        # (research/adaptive_online_im.md §8.2 trap 5). Index t is the count
        # AFTER the step at t - 1.
        "spread_curve": trajectory.spread_curve,
        "spread_at_horizon": (
            trajectory.spread_curve[-1] if trajectory.spread_curve else None
        ),
    }

    # Sense first, because everything downstream that picks a winner needs it and
    # the per-arm JSON is read standalone by plots/report/summary
    result["objective"] = task.sense

    if task.contains:
        result["containment"] = True
        result["outbreak"] = list(outbreak)
        result["outbreak_pct"] = outbreak_pct
        result["outbreak_selector"] = config.outbreak_selector
        # §8.3: the connectivity functionals reported ALONGSIDE the diffusion
        # number, computed exactly, as context — never as the learned target.
        # Read off the executed bags rather than re-planning, for the same reason
        # --compare replays them: a randomized strategy returns a different set on
        # a second call, and this has to describe the set that earned the reward.
        result["structural"] = containment_metrics(
            graph.edge_index, graph.num_nodes, removal_set(trajectory.actions)
        )
        structural = result["structural"]
        print(
            f"[run] structural: removed {structural['k']}, "
            f"GCC {structural['largest_cc_intact']:.0f} -> "
            f"{structural['largest_cc_size']:.0f} "
            f"({structural['largest_cc_drop_pct']:.1f}% drop), "
            f"pairwise conn -{structural['pairwise_conn_drop_pct']:.1f}%, "
            f"R={structural['schneider_r']:.4f}, "
            f"degree-rank rho={structural['degree_rank_spearman']:+.3f}"
        )

    if config.edit_rate:
        result["streaming"] = True
        result["edit_rate"] = config.edit_rate

    if config.campaigns > 1:
        result["multi_round"] = True
        result["campaigns"] = config.campaigns
        # Per-campaign spread alongside the union: a union that barely grows
        # after campaign 1 means the later campaigns bought nothing
        result["campaign_rewards"] = trajectory.cost.get("campaign_rewards")

    if batches is not None:
        # (k, b, r) together, because the adaptive-IM literature splits three ways
        # on the budget convention and a spread number is not comparable without
        # all three (research/adaptive_online_im.md §8.2 trap 3)
        result["adaptive"] = True
        result["rounds"] = len(batches)
        result["round_batches"] = batches
        result["per_round_budget"] = config.per_round_budget
        result["round_gap"] = config.round_gap
        result["feedback_model"] = config.feedback_model
        result["round_schedule"] = describe_schedule(task, batches)
        # Spread after each round: the per-round benefit curve the cost argument
        # is read against. Shorter than `rounds` when the cascade died early.
        result["round_spreads"] = round_spreads(
            trajectory.infected_counts, batches, config.round_gap
        )

    if routing_reply is not None:
        result["routing_reply"] = routing_reply

    # When the credit flag is enabled, ablate the executed action sequence against the same environment and measure the reward
    if config.credit:
        print("[run] per-action counterfactual credit (one rollout per action)...")
        # Credit of the executed action sequence; for state-dependent strategies (per_step/windowed) the recorded bags are replayed as a fixed plan
        base_reward, entries = counterfactual_credit(
            environment,
            trajectory.actions,
            config.horizon,
            config.budget,
            creditable_ops=tuple(config.allowed_ops),
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
            graph,
            config.diffusion_model,
            mc_runs=referee_runs,
            base_seed=config.seed,
            remove_semantics=config.remove_semantics,
        )

        if config.campaigns > 1:
            # The referee has to score the SAME quantity the arm was scored on.
            # Without this the arm reports a 3-campaign union and the shared
            # referee reports a 1-campaign spread, and the report silently puts
            # the two in one column.
            mc_environment = MultiRoundEnvironment(
                mc_environment, campaigns=config.campaigns, base_seed=config.seed
            )

        # Replay the actions that EARNED the reward rather than re-planning. A
        # generated script that samples (RIS with a live seed, a randomized local
        # search) returns a different seed set on a second call, so re-planning
        # would referee a strategy that never ran — and for per_step it would fire
        # a fresh LLM call per (run, timestep) of the replay.
        action_fn = partial(planned_action, trajectory.actions)

        mc_trajectory = mc_environment.rollout(action_fn, config.horizon, config.budget)
        result["mc_reward"] = mc_trajectory.reward
        result["mc_seed"] = mc_trajectory.cost["seed"]
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
            # Offset from the base seed, so these are fresh realizations no
            # matter what --seed is, and each one is recorded for replay
            reeval_seed_list = [
                config.seed + offset for offset in range(1, wm_reeval_seeds + 1)
            ]
            reeval_rewards = [
                environment.rollout(
                    action_fn, config.horizon, config.budget, seed=reeval_seed
                ).reward
                for reeval_seed in reeval_seed_list
            ]
            result["wm_reeval_seeds"] = reeval_seed_list
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

    # The result supersedes the mid-search state; leaving it would make a
    # finished arm look interrupted to the next sweep
    checkpoint.clear(checkpoint_path)

    usage = result["llm_usage"]
    cost = "" if usage["cost_usd"] is None else f", ${usage['cost_usd']:.4f}"
    print(
        f"[run] llm: {usage['calls']} calls, {usage['prompt_tokens']:,} prompt + "
        f"{usage['completion_tokens']:,} completion tokens{cost}"
    )
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
        "--allow-mc-algorithms",
        action="store_true",
        help="re-expose the per-candidate-simulation algorithms (celf, celf_pp, "
        "vanilla_greedy, static_greedy, ...) to generated scripts; they are "
        "blocked by default because they exceed 60s per call and their episodes "
        "are invisible to real_env_episodes (default: False).",
    )
    parser.add_argument(
        "--strategy-timeout",
        type=float,
        default=executor.strategy_timeout_seconds,
        help="wall-clock cap in seconds on one generated plan_horizon()/act() "
        "call; an overrun becomes a repair turn instead of hanging the sweep. "
        f"0 disables (default: {executor.strategy_timeout_seconds:.0f}).",
    )
    parser.add_argument(
        "--llm-price-in",
        type=float,
        default=None,
        help="USD per 1M prompt tokens, for the cost line in the results JSON. "
        "The lab gateway fronts Pro subscriptions and bills nothing per token, so "
        "there is no rate to assume: tokens are always counted exactly, cost stays "
        "null unless both price flags are given (default: None).",
    )
    parser.add_argument(
        "--llm-price-out",
        type=float,
        default=None,
        help="USD per 1M completion tokens; see --llm-price-in (default: None).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore any checkpoint from a killed run of this arm and search from "
        "scratch (default: resume).",
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
        choices=["one_shot", "per_step", "windowed", "evolve", "adaptive"],
        help="outer-loop method; evolve = population edits with refine/restructure operators; adaptive = the same search over a per-round policy for adaptive IM (default: one_shot).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="influence_maximization",
        help="task name shown to the agent in its prompt (default: influence_maximization).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=4,
        help="adaptive IM: batches the budget is committed in, each chosen after seeing the previous one's diffusion (default: 4).",
    )
    parser.add_argument(
        "--per-round-budget",
        type=int,
        default=None,
        help="adaptive IM: fix seeds per round b and derive r = ceil(k/b), instead of fixing r with --rounds (default: None).",
    )
    parser.add_argument(
        "--round-gap",
        type=int,
        default=1,
        help="adaptive IM: timesteps of diffusion between consecutive rounds (default: 1).",
    )
    parser.add_argument(
        "--feedback-model",
        type=str,
        default=full_adoption,
        choices=list(valid_feedback_models),
        help="adaptive IM: what the policy observes at a round boundary; myopic hides state.infected and leaves only the current wave (default: full_adoption).",
    )
    parser.add_argument(
        "--edit-rate",
        type=float,
        default=0.0,
        help="dynamic/streaming IM: exogenous edge edits per timestep as a fraction of |E|, applied to every arm. 0 = static graph (default: 0.0).",
    )
    parser.add_argument(
        "--campaigns",
        type=int,
        default=1,
        help="multi-round IM: separate campaigns of `budget` seeds each, scored on the union of what they activate. 1 = a single campaign (default: 1).",
    )
    parser.add_argument(
        "--outbreak-pct",
        type=float,
        default=None,
        help="critical node detection: size of the exogenous outbreak the planner must contain, as a percentage of N. Unset = the task registry's value, 0 for every seeding task (default: None).",
    )
    parser.add_argument(
        "--outbreak-selector",
        type=str,
        default="random",
        choices=list(outbreak_selectors),
        help="how the outbreak's source nodes are chosen; deterministic in --seed so every arm faces the same one. `random` is the honest default: a targeted outbreak makes blocking the same ranking problem (default: random).",
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
        "--remove-semantics",
        type=str,
        default=spent,
        choices=list(valid_remove_semantics),
        help="what remove_node means: spent = stays counted, stops spreading; "
        "blocked = deleted from the graph, uncounted, cannot transmit or be "
        "infected. Must match the --wm-results-json checkpoint (default: spent).",
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
        allow_mc_algorithms=args.allow_mc_algorithms,
        strategy_timeout=args.strategy_timeout,
        llm_price_in=args.llm_price_in,
        llm_price_out=args.llm_price_out,
        resume=not args.no_resume,
        task=args.task,
        method=args.method,
        rounds=args.rounds,
        per_round_budget=args.per_round_budget,
        round_gap=args.round_gap,
        feedback_model=args.feedback_model,
        edit_rate=args.edit_rate,
        campaigns=args.campaigns,
        outbreak_pct=args.outbreak_pct,
        outbreak_selector=args.outbreak_selector,
        strategy_mode=args.strategy_mode,
        evaluator=args.evaluator,
        diffusion_model=args.diffusion_model,
        remove_semantics=args.remove_semantics,
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
