"""
Top-level driver: run the coding-agent outer loop over the inner-loop environment.

For a full dataset sweep (all arms x all budgets, plots and report) use
`python -m pipeline.run` instead; this module is the single-run entry point.

Baselines -

python -m coding_agent.run --data-dir results/ba40/data \
    --wm-results-json results/ba40/world_model/sage_IC.json \
    --method one_shot --strategy-mode free \
    --evaluator world_model --budget 5 --horizon 10 \
    --baseline degree_discount --outer-iters 1 \
    --out-json results/ba40/agent/pct5.0/baseline_degree_discount.json


Graph algorithm routing (LLM picks from a list of graph algorithms, no synthesis) -

python -m coding_agent.run --data-dir results/ba40/data \
    --model gpt-6-astra \
    --wm-results-json results/ba40/world_model/sage_IC.json \
    --method one_shot --strategy-mode free \
    --evaluator world_model --budget 5 --horizon 10 \
    --routing --outer-iters 1 \
    --out-json results/ba40/agent/pct5.0/routing.json


Oracle -

python -m coding_agent.run --data-dir results/ba40/data \
    --model gpt-6-astra \
    --method one_shot --strategy-mode free \
    --evaluator oracle --budget 5 --horizon 10 \
    --allowed-ops add_node remove_node \
    --mc-runs 200 --n-samples 50 \
    --outer-iters 5 --out-json results/ba40/agent/pct5.0/one_shot_free.json


Scored mode + evolve (agent edits algorithm internals, population search) -

python -m coding_agent.run --data-dir results/sbm40/data \
    --model gpt-6-astra \
    --wm-results-json results/sbm40/world_model/sage_IC.json \
    --method evolve --strategy-mode scored \
    --evaluator world_model --budget 5 --horizon 10 \
    --allowed-ops add_node remove_node \
    --outer-iters 10 --n-samples 50 \
    --out-json results/sbm40/agent/pct5.0/evolve_scored.json
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from dotenv import load_dotenv

from coding_agent import checkpoint, executor
from coding_agent.agent import (
    CodingAgent,
    GatewayProvider,
    default_model,
    default_reasoning_effort,
    empty_usage,
    merge_usage,
    reasoning_efforts,
    verify_gateway_model,
)
from coding_agent.blocking import (
    blocking_metrics,
    counter_seed,
    resolve_lever,
    unopposed_reference,
    valid_levers,
)
from coding_agent.containment import (
    outbreak_selectors,
    removal_set,
    ring_size,
    select_outbreak,
)
from coding_agent.credit import augment_solo, counterfactual_credit, credit_limit, planned_action
from coding_agent.envs.monte_carlo_env import MonteCarloEnvironment
from coding_agent.envs.multi_round_env import MultiRoundEnvironment
from coding_agent.envs.world_model_env import (
    WorldModelEnvironment,
    default_max_block_arc_hidden,
)
from coding_agent.feedback import default as feedback_default
from coding_agent.feedback import valid_tiers as valid_feedback_tiers
from coding_agent.epidemic import (
    default_contact_reduction,
    epidemic_metrics,
    unprotected_reference,
    vaccinate,
)
from coding_agent.epidemic import lever_shape as epidemic_lever_shape
from coding_agent.epidemic import (
    resolve_lever as resolve_epidemic_lever,
)
from coding_agent.epidemic import (
    valid_levers as valid_epidemic_levers,
)
from coding_agent.localization import (
    episode_budget,
    evaluate_localizer,
    load_instances,
    localization_label_metrics,
    referee_resimulation_error,
    valid_budget_modes,
    valid_observations,
)
from coding_agent.methods.base import OuterLoopMethod, summarize
from coding_agent.methods.evolve import EvolveSearch
from coding_agent.methods.one_shot import OneShotSuperAlgorithm, default_max_repairs
from coding_agent.methods.per_step import PerStepReprompt
from coding_agent.methods.windowed import WindowedOnline
from coding_agent.provenance import compute_provenance
from coding_agent.prediction import (
    default_forecast_samples,
    evaluate_predictor,
    load_forecasts,
    referee_forecast_samples,
    referee_modelling_error,
    resolve_target,
    trivial_predictor_error,
)
from coding_agent.prompts import (
    build_explanation_prompt,
    build_routing_prompt,
    build_routing_system,
)
from coding_agent.reconstruction import (
    default_hidden_rate,
    default_observation_rate,
    evaluate_reconstructor,
    load_cascades,
    partial_times,
    reconstruction_label_metrics,
    referee_likelihood,
    trivial_decoder_reward,
    valid_settings,
)
from coding_agent.reconstruction import (
    referee_resimulation_error as referee_reconstruction_error,
)
from coding_agent.rounds import (
    describe_schedule,
    round_batches,
    round_schedule,
    round_spreads,
)
from coding_agent.tools.adaptive_algorithms import adaptive_algorithms
from coding_agent.tools.blocking_algorithms import (
    all_blocking_algorithms,
    blocking_levers,
    blocking_shape,
    emittable,
    lever_shape,
)
from coding_agent.tools.dismantling_algorithms import dismantling_algorithms
from coding_agent.tools.immunization_algorithms import (
    emittable as immunization_emittable,
)
from coding_agent.tools.immunization_algorithms import (
    immunization_algorithms,
    immunization_levers,
    immunization_shape,
)
from coding_agent.tools.library_api import (
    algorithm_names,
    blocking_names,
    dismantling_names,
    immunization_names,
    localization_names,
    prediction_names,
    reconstruction_names,
)
from coding_agent.tools.localization_algorithms import localization_algorithms
from coding_agent.tools.prediction_algorithms import (
    fitted_prediction_algorithms,
    prediction_algorithms,
)
from coding_agent.tools.reconstruction_algorithms import reconstruction_algorithms
from coding_agent.types import GraphInfo, TaskSpec, full_adoption, valid_feedback_models
from data.wm_cascades import increment_target, valid_targets
from data.wm_competitive import (
    CompetitiveConfig,
    auto_dominance,
    competitive_model_name,
    resolve_tie_break,
    shared_positive_prob,
    tie_break_choices,
)
from data.wm_epidemic import EpidemicConfig, default_burn_in
from data.wm_simulator import (
    epidemic_dynamics,
    spent,
    valid_action_ops,
    valid_remove_semantics,
)
from pipeline.conditions import parse_arm
from pipeline.layout import budget_label, checkpoint_suffix
from pipeline.tasks import get_task, maximize, task_names
from world_model.wm_data import load_graph_store
from world_model.wm_metrics import (
    containment_metrics,
    default_prediction_metric,
    epidemic_curve_metrics,
    valid_prediction_metrics,
)

world_model = "world_model"
monte_carlo = "monte_carlo"
oracle = "oracle"

# Referee samples per replay: the oracle is batched, so this is cheap, and 200
# left a standard error above the gaps the small-graph smokes were comparing
referee_samples_default = 1000

# Fresh-seed WM re-evaluations of the winning strategy after the referee replay
wm_reeval_seeds = 3

# Above this node count the per-node marginal vector is dropped from the results
# JSON: it would dominate the file (1M nodes ~ 7MB of floats per run)
max_serialized_marginals = 200_000


@dataclass()
class ExperimentConfig:
    # Key of pipeline.tasks.tasks; reaches the agent through TaskSpec, so the
    # prompt names the problem the arm is actually being scored on
    task: str = "influence_maximization"
    method: str = "one_shot"  # one_shot | per_step | windowed | evolve | adaptive
    strategy_mode: str = (
        "free"  # free (whole Strategy) | scored (score/schedule hooks only)
    )
    evaluator: str = "world_model"  # world_model | monte_carlo | oracle
    model: str = default_model  # gateway model name
    temperature: float | None = (
        None  # LLM sampling temperature; None -> provider default
    )
    # Reasoning effort for the OpenAI family; None sends nothing
    reasoning_effort: str | None = default_reasoning_effort
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
    # fights the SAME outbreak: an arm facing a different one would be measuring
    # the outbreak, not the method.
    outbreak_pct: float | None = None
    outbreak_selector: str = "random"
    # Influence blocking. `outbreak_pct` / `outbreak_selector` above double as |S_N|
    # and the ATTACKER MODEL: the literature's second experimental axis, which IM
    # does not have, so nothing new is needed for those. What is new is the lever
    # (which of the four published interventions the budget buys), the tie-break,
    # `p_L`, and Budak's detection delay. All four are inert unless the task is
    # competitive.
    blocking_lever: str = counter_seed
    tie_break: str = auto_dominance
    positive_prob: str = shared_positive_prob
    detection_delay: int = 0
    # Epidemic control. `outbreak_pct` / `outbreak_selector` above double as the
    # index-case count and the OUTBREAK MODEL, so nothing new is needed for those.
    # What is new is the lever (which of the four published interventions the
    # budget buys) and the three compartmental rates, which must be reported
    # rather than left implicit: beta and gamma are free parameters nobody
    # standardizes, so a table that fixes them without saying so is comparable only
    # to itself. All are inert unless the task registry marks the task epidemic.
    epi_lever: str = vaccinate
    epi_beta: float = 1.0
    epi_gamma: float = 0.3
    epi_alpha: float = 0.5
    epi_burn_in: float = default_burn_in
    contact_reduction: float = default_contact_reduction
    # Source localization. The two SPLITS are the load-bearing pair:
    # the outer loop's reward is computed
    # on `sl_select_split` and the winning program is then re-scored, unmodified,
    # on `sl_eval_split`. Without that separation a program that memorized specific
    # cascades is indistinguishable from an algorithm, and the amortization claim
    # is the whole point of the task.
    sl_select_split: str = "train"
    sl_eval_split: str = "test"
    sl_instances: int = 20
    sl_observation: str = "marginal"
    sl_budget_mode: str = episode_budget
    sl_source_tolerance: float = 0.5
    # The GRAPH axis: run the winning program of ANOTHER run, unmodified, on
    # this dataset. That comparison is the headline of the amortization claim and
    # is one no per-instance method can even enter: SL-VAE has no artifact to
    # transfer. Points at another arm's results JSON; its `script` field is run as
    # a canned strategy here.
    transfer_from: str | None = None
    # Cascade reconstruction. `cr_setting` is the load-bearing one: the four
    # settings are four PROTOCOLS, not four knobs, and a decoder selected under
    # `final_snapshot` is solving a different problem from one selected under
    # `partial_times`. The masking DIRECTION is stated explicitly because two
    # papers in this literature use the symbol sigma for opposite quantities:
    # `cr_observation_rate` is the probability a node IS reported.
    cr_setting: str = partial_times
    cr_observation_rate: float = default_observation_rate
    cr_hidden_rate: float = default_hidden_rate
    cr_instances: int = 20
    cr_select_split: str = "train"
    cr_eval_split: str = "test"
    # lambda in the REPORTED `lambda * PathPrecision + (1 - lambda) * EventF1`
    # tree score, computed against the stored history after the search. The
    # search itself runs on the label-free kernel likelihood (the risk of
    # a search that never attempts the tree is answered by the likelihood scoring
    # the asserted transmissions, not by this weight).
    cr_tree_weight: float = 0.6
    # Cascade prediction. The two SPLITS are the load-bearing pair for the same
    # reason source localization's are, plus one this task has and that one does
    # not: a decade of published
    # numbers moved materially when the split stopped being random over cascades,
    # and our replay is chronological from the first commit for exactly that reason.
    # `cp_observation_steps` is how much of each cascade the predictor sees; 0 reads
    # it back off the dataset's own `observed` block, which is where the corpus's
    # published window landed after binning.
    cp_select_split: str = "train"
    cp_eval_split: str = "test"
    cp_instances: int = 40
    cp_observation_steps: int = 0
    cp_metric: str = default_prediction_metric
    cp_target: str = increment_target
    # Unrolls averaged inside one `forecast_marginals` call. The cost knob:
    # an @monte_carlo arm pays `steps * samples * mc_runs` real episodes per call
    # and a @world_model arm pays `steps * samples` matmuls.
    cp_forecast_samples: int = default_forecast_samples
    # True when this arm is the @native condition. The evaluator has already been
    # resolved to monte_carlo with one episode by then, so the arm identity cannot
    # be recovered from `evaluator`, and it decides whether `predict_marginals`
    # exists at all.
    native_arm: bool = False
    outer_iters: int = 20
    mc_runs: int = 200
    # The shared referee every arm's winner is replayed on: the exact oracle
    # simulator (batched, one sampled state per step) by default, NDlib as the
    # slow alternative. Its sample count is separate from the search's.
    referee: str = oracle
    referee_samples: int = referee_samples_default
    # NDlib on the same winner: the independent reference the oracle is measured
    # against, and the timing row. Off by default; None runs -> mc_runs.
    mc_agreement: bool = False
    mc_agreement_runs: int | None = None
    n_samples: int = 200
    # arcs x hidden units one world-model rollout block may hold; scoring more
    # plans than fit runs as several blocks with identical results
    max_block_arc_hidden: int = default_max_block_arc_hidden
    seed: int = 42
    device: str = "cpu"
    data_dir: str | None = None  # world_model.wm_data graph store dir
    graph_id: str | None = None  # which graph in the store (default: first)
    wm_results_json: str | None = None  # train_wm.py results JSON (for the WM env)
    credit: bool = True  # per-action counterfactual credit (feedback + results)
    # The probe turn before each evolve generation (one extra LLM call per
    # generation, answered immediately about the incumbent's plan)
    probe_turn: bool = True
    # Stop an evolve search once the unbiased incumbent estimate has been flat
    # for this many generations; 0 runs every generation
    stop_when_flat: int = 0
    # Let a run of wrong-signed edit forecasts add exploration mass
    calibration_steering: bool = False
    # Post-search rediscovery distance against the task's library pool
    provenance: bool = True
    provenance_timeout: float = 60.0
    # Which feedback tier each refinement generation receives
    # (coding_agent/feedback.py). `default` is the full feedback and the
    # default; f0-f3 are the controlled ladder Experiment 3 varies.
    feedback: str = feedback_default
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
    # to hardcode: supply your own or the cost stays null while tokens are
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

    # Read by the methods when they build the strategy: a canned baseline is the
    # only kind of strategy that receives the evaluator bindings
    canned = True

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
    canned: bool = False,
    checkpoint_path: Path | None = None,
    checkpoint_fingerprint: dict | None = None,
    instances: list | None = None,
) -> OuterLoopMethod:
    """
    The single construction site for every method.

    `strategy_mode` / `allow_mc_algorithms` are the resolved values, not
    config's: a canned script overrides both. Only the two methods with a
    refinement loop take a checkpoint: per_step and windowed have nothing to
    resume.
    """
    if config.method == "one_shot":
        return OneShotSuperAlgorithm(
            outer_iters=config.outer_iters,
            # A canned script never reads its feedback, and under @monte_carlo
            # every ablation is a sequential batch of real episodes
            credit=config.credit and not canned and config.evaluator != monte_carlo,
            probes=not config.native_arm and not canned,
            strategy_mode=strategy_mode,
            # A canned script ignores its prompt, so the anchor rollouts would only
            # burn real episodes and wall clock without informing anything
            use_anchor=use_anchor,
            # ...and for the same reason a repair turn re-sends the IDENTICAL
            # script, so it fails identically. A classical baseline or an external
            # repo that blows the wall-clock cap once will blow it three times,
            # costing 3x the timeout before the arm is abandoned anyway.
            max_repairs=0 if canned else default_max_repairs,
            allow_mc_algorithms=allow_mc_algorithms,
            checkpoint_path=checkpoint_path,
            checkpoint_fingerprint=checkpoint_fingerprint,
        )

    # per_step has no refinement loop: it is one episode with an LLM call per
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
            credit=config.credit and config.evaluator != monte_carlo,
            probes=not config.native_arm,
            feedback=config.feedback,
            probe_turn=config.probe_turn,
            stop_when_flat=config.stop_when_flat,
            calibration_steering=config.calibration_steering,
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


def _competitive_config(config: ExperimentConfig) -> CompetitiveConfig:
    """The two-cascade dynamics this run simulates, with `auto` already resolved."""
    return CompetitiveConfig(
        tie_break=config.tie_break,
        positive_prob=(
            shared_positive_prob
            if config.positive_prob == shared_positive_prob
            else float(config.positive_prob)
        ),
        remove_semantics=config.remove_semantics,
    )


def _epidemic_config(config: ExperimentConfig) -> EpidemicConfig:
    """The compartmental dynamics this run simulates."""
    return EpidemicConfig(
        beta_scale=config.epi_beta,
        gamma=config.epi_gamma,
        alpha=config.epi_alpha,
        remove_semantics=config.remove_semantics,
        burn_in=config.epi_burn_in,
    )


def _check_checkpoint_graph(
    config: ExperimentConfig, graph: GraphInfo, graph_id: str | None
) -> None:
    """
    Refuse a checkpoint whose training store does not hold this run's graph.

    Only checked when it is determinable: the results JSON names its data_dir,
    that dir differs from the run's, its index still exists, and the run knows
    its graph_id. A store that lacks the id, or holds it at different counts,
    is a different dataset, and a GNN rolled out on a graph it never saw scores
    something, silently.
    """
    trained_dir = json.loads(Path(config.wm_results_json).read_text())["config"].get(
        "data_dir"
    )
    if not trained_dir or graph_id is None or config.data_dir is None:
        return

    trained_dir, run_dir = Path(trained_dir), Path(config.data_dir)
    index_path = trained_dir / "graphs_index.json"
    if trained_dir.resolve() == run_dir.resolve() or not index_path.exists():
        return

    entries = {
        meta["graph_id"]: meta for meta in json.loads(index_path.read_text())
    }
    meta = entries.get(graph_id)
    # Node count only: the index's n_edges counts undirected edges on an
    # undirected graph and arcs on a directed one, so it is not comparable to
    # edge_index without re-deriving the loader's convention
    trained_nodes = None if meta is None else int(meta["n_nodes"])

    if trained_nodes != graph.num_nodes:
        raise ValueError(
            f"checkpoint {config.wm_results_json} was trained on the graph store "
            f"{trained_dir}, which "
            + (
                f"has no graph {graph_id!r}"
                if meta is None
                else f"holds {graph_id!r} at {trained_nodes} nodes"
            )
            + f"; this run's graph {graph_id!r} has {graph.num_nodes} nodes. Point "
            f"--wm-results-json at a checkpoint trained on this dataset, or "
            f"--data-dir at the checkpoint's."
        )


def _build_referee(config: ExperimentConfig, graph: GraphInfo, task: TaskSpec, outbreak: tuple) -> object:
    """The shared referee: the arm's environment builder pointed at the referee's evaluator and sample count."""
    referee_config = replace(
        config,
        evaluator=config.referee,
        n_samples=config.referee_samples,
        mc_runs=config.referee_samples,
    )
    referee = _build_environment(
        referee_config,
        graph,
        negative_seeds=outbreak,
        competitive=task.blocks,
        epidemic=task.immunizes,
    )

    # The referee has to score the SAME quantity the arm was scored on: a
    # 3-campaign union against a 1-campaign spread in one column reads as noise
    if config.campaigns > 1:
        referee = MultiRoundEnvironment(
            referee, campaigns=config.campaigns, base_seed=config.seed
        )

    return referee


def _agreement_keys(measured: dict) -> dict:
    """The inverse-task referee helpers' keys, re-prefixed as the NDlib agreement check."""
    return {
        (f"mc_{key[len('referee_'):]}" if key.startswith("referee_") else f"mc_{key}"): value
        for key, value in measured.items()
    }


def _build_environment(
    config: ExperimentConfig,
    graph: GraphInfo,
    negative_seeds: tuple = (),
    competitive: bool = False,
    epidemic: bool = False,
    graph_id: str | None = None,
) -> object:
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
            negative_seeds=negative_seeds,
            competitive_config=_competitive_config(config) if competitive else None,
            epidemic_config=_epidemic_config(config) if epidemic else None,
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
            negative_seeds=negative_seeds,
            max_block_arc_hidden=config.max_block_arc_hidden,
        )

        # The head is built for ONE dynamics (an LT threshold head has no per-edge
        # transmission model, an IC head no threshold), so a checkpoint that
        # disagrees with the run's dynamics would score a plan against a process
        # that never produced the referee's numbers
        if environment.diffusion_model != config.diffusion_model:
            raise ValueError(
                f"checkpoint {config.wm_results_json} was trained under "
                f"{environment.diffusion_model} dynamics but this run is "
                f"--diffusion-model {config.diffusion_model}; pass "
                f"--diffusion-model {environment.diffusion_model} or use a "
                f"checkpoint trained for {config.diffusion_model}"
            )

        _check_checkpoint_graph(config, graph, graph_id)

        # ...and a non-compartmental checkpoint cannot be rolled out against an
        # epidemic task: its head is monotone by construction, so `I` could never
        # shrink and every rollout would report a cascade that transmits forever
        if epidemic and not environment.epidemic:
            raise ValueError(
                f"checkpoint {config.wm_results_json} was trained on IC/LT "
                f"transitions but this run is a compartmental epidemic task. Its "
                f"head composes `y_inf = infected + (1 - infected) * p_new`, which "
                f"is monotone by construction and cannot represent recovery. "
                f"Regenerate with `data/generate_wm_data.py --models SIR` and "
                f"retrain, or point --wm-results-json at a compartmental checkpoint."
            )

        # A single-cascade checkpoint cannot be rolled out against a two-cascade
        # task: its head has no positive channel at all, so it would predict the
        # rumour as if the blocker's counter-cascade did not exist and every arm
        # would score the unopposed spread
        if competitive and not environment.competitive:
            raise ValueError(
                f"checkpoint {config.wm_results_json} was trained on SINGLE-cascade "
                f"transitions but this run is a two-cascade blocking task. Regenerate "
                f"with `data/generate_wm_data.py --competitive` and retrain, or point "
                f"--wm-results-json at a competitive checkpoint."
            )

        # Everything else in the run (the prompt, the referee) follows
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
        competitive_config = _competitive_config(config)

        return WorldModelEnvironment.oracle(
            graph,
            config.diffusion_model,
            device=config.device,
            n_samples=config.n_samples,
            base_seed=config.seed,
            remove_semantics=config.remove_semantics,
            negative_seeds=negative_seeds,
            competitive=competitive,
            max_block_arc_hidden=config.max_block_arc_hidden,
            tie_break=(
                config.tie_break
                if epidemic
                else resolve_tie_break(config.tie_break, config.diffusion_model)
            ),
            positive_prob=competitive_config.positive_prob,
            # The compartment head's oracle form reproduces the simulator's own
            # one-step marginals exactly, so this is a genuine ceiling here rather
            # than a well-shaped approximation of one
            epidemic=epidemic,
            epi_beta=config.epi_beta,
            epi_gamma=config.epi_gamma,
            epi_alpha=config.epi_alpha,
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


def canned_baseline_script(
    config: ExperimentConfig, name: str, outbreak: tuple, batches: list | None
) -> str:
    """
    The Strategy source that runs library member `name` through the identical
    scoring path as every other arm. One template per pool; shared by the
    condition-1 baseline arms and by the post-search provenance check, so a
    "nearest library algorithm" is measured on exactly the call the baseline
    row was.
    """
    if name in adaptive_algorithms:
        if batches is None:
            raise ValueError(
                f"baseline {name!r} is a per-round adaptive policy, "
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
        script = f"""\
class AdaptiveBaseline(Strategy):
    def act(self, state, graph, timestep):
        batch = {schedule!r}.get(timestep, 0)
        if not batch:
            return []
        return [
            ActionOp("add_node", node)
            for node in adaptive_algorithms.{name}(
                state,
                graph,
                batch,
                "{config.diffusion_model}",
                total_budget={config.budget},
            )
        ]
"""
    elif name in prediction_algorithms:
        # A published predictor returns a NUMBER, which is what makes this pool
        # different from every other one here. `fit_examples` is passed only to
        # the members that fit a constant (the pool names them explicitly), so a
        # label can never reach an unfitted member; `predict` is the metered
        # forward model, which `mc_forward` reads and everything else ignores
        # through **kw.
        script = f"""\
class PredictionBaseline(Strategy):
    def predict(self, graph, observation, horizon):
        return prediction_algorithms.{name}(
            graph,
            observation,
            horizon,
            fit_examples=(
                list(getattr(self, "fit_examples", []))
                if "{name}" in {tuple(fitted_prediction_algorithms)!r}
                else None
            ),
            predict=getattr(self, "forecast_marginals", None),
            seed={config.seed},
        )
"""
    elif name in reconstruction_algorithms:
        # A published decoder is a whole-TRAJECTORY inference: it is handed the
        # masked observation and returns `{node: (time, parent)}`. `predict` is
        # the metered kernel, which the kernel-using members read and the
        # structural ones ignore through **kw.
        script = f"""\
class ReconstructionBaseline(Strategy):
    def reconstruct(self, graph, observation, horizon):
        return reconstruction_algorithms.{name}(
            graph,
            observation,
            horizon,
            diffusion_model="{config.diffusion_model}",
            predict=getattr(self, "step_marginals", None),
        )
"""
    elif name in localization_algorithms:
        # A published localizer is a source-set INFERENCE, not a plan: it is
        # handed the observation and returns the nodes it believes started the
        # cascade. Its paired scorer rides along so the arm gets a real AUC
        # rather than the rank-derived stand-in a set-only method falls back to.
        script = f"""\
class LocalizationBaseline(Strategy):
    def localize(self, graph, observation, budget):
        return [
            int(node)
            for node in localization_algorithms.{name}(
                graph,
                observation,
                budget,
                diffusion_model="{config.diffusion_model}",
                horizon={config.horizon},
            )
        ]

    def source_scores(self, graph, observation):
        return localization_scorers.{name}(
            graph, observation, diffusion_model="{config.diffusion_model}"
        )
"""
    elif name in all_blocking_algorithms:
        # A published blocker is handed the RUMOUR's own seeds and returns the
        # intervention this lever buys: node ids on three levers, `(u, v)` arcs
        # on the fourth. `blocking.blocking_plan` reconciles the two shapes into
        # one plan, which is what keeps a library algorithm runnable without
        # rewriting it to know what a plan is.
        if not emittable(name, config.blocking_lever):
            raise ValueError(
                f"baseline {name!r} returns "
                f"{blocking_shape(name)}s and this arm's lever "
                f"({config.blocking_lever!r}) spends its budget on "
                f"{lever_shape[config.blocking_lever]}s, so its output is not "
                f"something this arm may emit. It is a "
                f"{blocking_levers[name]} method: run it with "
                f"--blocking-lever {blocking_levers[name]}, or pick a "
                f"{config.blocking_lever} member."
            )

        script = f"""\
class BlockingBaseline(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        picks = blocking_algorithms.{name}(
            graph,
            budget,
            "{config.diffusion_model}",
            negative_seeds={tuple(outbreak)!r},
            horizon=horizon,
        )
        return blocking.blocking_plan(
            picks, graph, budget, "{config.blocking_lever}", horizon
        )
"""
    elif name in immunization_algorithms and get_task(config.task).epidemic:
        # Gated on the task, NOT on pool membership alone. `netshield` and
        # `acquaintance_immunization` exist in BOTH this pool and the
        # dismantling one as genuinely different functions (the immunization
        # forms take the outbreak and refuse to dose an index case), so an
        # order-dependent chain would hand every critical-node-detection arm
        # asking for `netshield` the epidemic implementation and silently
        # score the wrong node set. Both accept **_, so it would never crash.
        #
        # A published immunizer is a static DOSE allocation, committed at t=0.
        # It is handed the outbreak because the data-aware members (dava,
        # frontier_immunization) condition on it; the structural ones take **kw
        # and ignore it. `immunization_plan` then reconciles the two output
        # shapes: node ids on the node levers, `(u, v)` arcs on the edge ones,
        # and drops any index case the algorithm picked anyway.
        if not immunization_emittable(name, config.epi_lever):
            raise ValueError(
                f"baseline {name!r} returns "
                f"{immunization_shape[name]}s and this arm's lever "
                f"({config.epi_lever!r}) spends its budget on "
                f"{epidemic_lever_shape[config.epi_lever]}s, so its output is not "
                f"something this arm may emit. It is a "
                f"{immunization_levers[name]} method: run it with "
                f"--epi-lever {immunization_levers[name]}, or pick a "
                f"{config.epi_lever} member."
            )

        script = f"""\
class ImmunizationBaseline(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        picks = immunization_algorithms.{name}(
            graph,
            budget,
            "{config.diffusion_model}",
            outbreak={tuple(outbreak)!r},
            horizon=horizon,
        )
        return epidemic.immunization_plan(
            picks,
            graph,
            budget,
            "{config.epi_lever}",
            horizon,
            {tuple(outbreak)!r},
            {config.contact_reduction!r},
        )
"""
    elif name in dismantling_algorithms:
        # A dismantler is a static REMOVAL set, committed at t=0. It is handed
        # the outbreak because the simulation-based member scores candidates
        # against it; the structural members take **kw and ignore it.
        # `removal_plan` then drops any source it picked anyway and tops the
        # set back up, which is what keeps a published algorithm runnable
        # without rewriting it to know an outbreak exists.
        script = f"""\
class DismantlingBaseline(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        removals = dismantling_algorithms.{name}(
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
        script = f"""\
class Baseline(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        seeds = algorithms.{name}(
            graph, budget, "{config.diffusion_model}", horizon=horizon
        )
        return [[ActionOp("add_node", node) for node in seeds]] + [
            [] for _ in range(horizon)
        ]
"""

    return script


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

    registry = get_task(config.task)

    # The exogenous cascade, resolved BEFORE the environment so both the simulator and
    # the arm face the same one, and derived only from (graph, size, selector, seed).
    # For influence blocking this IS S_N, and `--outbreak-selector` is the
    # attacker model rather than an outbreak rule: the same machinery, because the
    # published attacker models (degree, PageRank, random, IMM) are exactly the
    # selectors that machinery already has.
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

    environment = _build_environment(
        config,
        graph,
        negative_seeds=outbreak,
        competitive=registry.competitive,
        epidemic=registry.epidemic,
        graph_id=graph_id,
    )

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

    # The labelled (G, y, x) episodes an inverse task is scored on. Two disjoint
    # pools, and the split between them is what makes the amortization claim
    # testable at all: the outer loop's
    # reward is computed on `select`, and the winner is re-run unmodified on
    # `evaluate`. A program that memorized specific cascades scores well on the
    # first and badly on the second.
    select_instances, evaluate_instances = [], []

    if registry.forecasts:
        # The same two-pool split as both inverse tasks, and here it carries an
        # extra load: `load_forecasts` RAISES on a SIMULATED dataset rather than
        # scoring it, because the whole argument collapses if the "real cascade"
        # is an NDlib rollout. The observation window comes off the dataset's own
        # `observed` block unless overridden, so a run cannot silently show the
        # predictor a different prefix than the one the corpus's published protocol
        # defines.
        common = dict(
            graph_id=graph_id,
            observed_steps=config.cp_observation_steps,
            limit=config.cp_instances,
            seed=config.seed,
        )
        select_instances = load_forecasts(
            config.data_dir, config.diffusion_model, config.cp_select_split, **common
        )
        evaluate_instances = load_forecasts(
            config.data_dir, config.diffusion_model, config.cp_eval_split, **common
        )

        from data.wm_cascades import observed_protocol

        protocol = observed_protocol(config.data_dir)
        print(
            f"[run] cascade prediction: selecting on {len(select_instances)} "
            f"{config.cp_select_split} cascades, held out on "
            f"{len(evaluate_instances)} {config.cp_eval_split} cascades "
            f"(corpus={protocol['corpus']}, "
            f"t_o={protocol['observation']} {protocol['time_unit']}(s) = "
            f"{protocol['observed_steps']} step(s), "
            f"t_p={protocol['horizon']} {protocol['time_unit']}(s), "
            f"split={protocol['split_protocol']}, metric={config.cp_metric}) "
            f", MINIMIZING the error"
        )
        if protocol["split_protocol"] != "chronological":
            print(
                "[run] WARNING: this dataset used a RANDOM split over cascades. "
                "That protocol leaks the "
                "future: cascades overlap in wall-clock time, so a training "
                "cascade's prediction window can sit inside a test cascade's "
                "observation window, and two 2021-24 SOTA models fell BELOW a "
                "plain MLP once it was fixed. Numbers from this run are comparable "
                "to the published tables and not to a leak-free one."
            )

    elif registry.reconstructs:
        # The same two-pool split, and for the same reason: a decoder that
        # memorized specific cascades is indistinguishable from an algorithm until
        # it meets episodes the search never saw. `require_parents` RAISES on a
        # dataset generated without --trace-parents rather than silently collapsing
        # the reward onto Event F1, which is the failure this guard exists to prevent.
        common = dict(
            graph_id=graph_id,
            setting=config.cr_setting,
            observation_rate=config.cr_observation_rate,
            hidden_rate=config.cr_hidden_rate,
            limit=config.cr_instances,
            seed=config.seed,
            require_parents=config.cr_tree_weight > 0.0,
        )
        select_instances = load_cascades(
            config.data_dir, config.diffusion_model, config.cr_select_split, **common
        )
        evaluate_instances = load_cascades(
            config.data_dir, config.diffusion_model, config.cr_eval_split, **common
        )

        print(
            f"[run] cascade reconstruction: selecting on {len(select_instances)} "
            f"{config.cr_select_split} episodes, held out on "
            f"{len(evaluate_instances)} {config.cr_eval_split} episodes "
            f"(setting={config.cr_setting}, reported at "
            f"{config.cr_observation_rate:.0%}, "
            f"lambda={config.cr_tree_weight} on PathPrecision)"
        )
    elif registry.recovers:
        select_instances = load_instances(
            config.data_dir,
            config.diffusion_model,
            config.sl_select_split,
            graph_id=graph_id,
            observation=config.sl_observation,
            limit=config.sl_instances,
            budget_mode=config.sl_budget_mode,
            budget=config.budget,
            source_tolerance=config.sl_source_tolerance,
            seed=config.seed,
        )
        evaluate_instances = load_instances(
            config.data_dir,
            config.diffusion_model,
            config.sl_eval_split,
            graph_id=graph_id,
            observation=config.sl_observation,
            limit=config.sl_instances,
            budget_mode=config.sl_budget_mode,
            budget=config.budget,
            source_tolerance=config.sl_source_tolerance,
            seed=config.seed,
        )
        print(
            f"[run] source localization: selecting on {len(select_instances)} "
            f"{config.sl_select_split} episodes, held out on "
            f"{len(evaluate_instances)} {config.sl_eval_split} episodes "
            f"(observation={config.sl_observation}, k from "
            f"{config.sl_budget_mode})"
        )

    # A blocking arm's lever decides BOTH what one unit of budget buys and which ops
    # it may emit, and they are set together so an arm can never be budgeted for one
    # and permitted another. Every other task keeps the registry's own pair.
    budget_op, lever_ops = (
        resolve_lever(config.blocking_lever)
        if registry.competitive
        # Same rule on the epidemic side, and it is what stops an arm being
        # budgeted for `remove_node` while permitted `set_edge_weight`
        else resolve_epidemic_lever(config.epi_lever)
        if registry.epidemic
        else (registry.budget_op, tuple(config.allowed_ops))
    )

    task = TaskSpec(
        task=config.task,
        objective=registry.summary,
        diffusion_model=config.diffusion_model,
        budget=config.budget,
        horizon=config.horizon,
        allowed_ops=lever_ops,
        remove_semantics=config.remove_semantics,
        # `recover` reaches the prompt, the contract check and the summary through
        # this one field, so a task that inverts cannot be driven as one that
        # intervenes by forgetting a flag
        objective_kind=registry.objective or maximize,
        instances=tuple(select_instances),
        source_budget_mode=config.sl_budget_mode,
        # Cascade prediction; inert for every task that does not forecast, so one
        # code path serves all seven runnable tasks
        observational=registry.observational,
        prediction_target=resolve_target(config.cp_target),
        prediction_metric=config.cp_metric,
        observation_window=(
            select_instances[0].observation.observed_steps
            if registry.forecasts and select_instances
            else 0
        ),
        forecast_samples=config.cp_forecast_samples,
        # Cascade reconstruction; inert for every task that does not decode, so
        # one code path serves all six runnable tasks
        reconstructs=registry.reconstructs,
        tree_weight=config.cr_tree_weight,
        observation_setting=config.cr_setting,
        # False is the @native condition: no forward model in the search loop at
        # all, which is the arm that answers whether one is worth anything
        forward_model=not config.native_arm,
        # From the registry, never from a flag: the objective sign and what a unit
        # of budget buys are properties of the TASK, and a run that disagreed with
        # its own registry entry would optimize one thing and be reported as another
        # `.sense` rather than `.objective`: a recover task reports an F1 that
        # MAXIMIZES and a forecast task an error that MINIMIZES, so reading the
        # objective literally would run two of the four families backwards
        sense=registry.sense,
        budget_op=budget_op,
        outbreak=outbreak,
        # Two-cascade fields; inert for every task that is not competitive, so one
        # code path serves all five runnable tasks
        competitive=registry.competitive,
        tie_break=(
            config.tie_break
            if registry.epidemic
            else resolve_tie_break(config.tie_break, config.diffusion_model)
        ),
        detection_delay=config.detection_delay,
        # Compartmental fields; inert for every task that is not epidemic, so one
        # code path serves all six runnable tasks
        epidemic=registry.epidemic,
        epi_lever=config.epi_lever,
        contact_reduction=config.contact_reduction,
        epi_beta=config.epi_beta,
        epi_gamma=config.epi_gamma,
        epi_alpha=config.epi_alpha,
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

    if outbreak and registry.competitive:
        print(
            f"[run] rumour S_N: {len(outbreak)} seed(s) ({outbreak_pct:g}% of N, "
            f"attacker={config.outbreak_selector}, seed={config.seed}) | "
            f"lever={config.blocking_lever} (budget buys {budget_op}) | "
            f"{competitive_model_name(config.positive_prob)}, "
            f"tie_break={task.tie_break}, detection_delay={config.detection_delay} "
            f", MINIMIZING the rumour's final size"
        )
    elif outbreak and registry.epidemic:
        print(
            f"[run] outbreak: {len(outbreak)} index case(s) ({outbreak_pct:g}% of "
            f"N, selector={config.outbreak_selector}, seed={config.seed}) | "
            f"{config.diffusion_model} beta_scale={config.epi_beta} "
            f"gamma={config.epi_gamma}"
            + (f" alpha={config.epi_alpha}" if config.diffusion_model == "SEIR" else "")
            + f" | lever={config.epi_lever} (budget buys {budget_op}) "
            f", MINIMIZING the attack rate"
        )
    elif outbreak:
        print(
            f"[run] outbreak: {len(outbreak)} source(s) "
            f"({outbreak_pct:g}% of N, selector={config.outbreak_selector}, "
            f"seed={config.seed}): MINIMIZING final infected count"
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

        verify_gateway_model(config.model)
        router = GatewayProvider(
            config.model,
            temperature=config.temperature,
            reasoning_effort=config.reasoning_effort,
        )
        providers.append(router)
        routing_reply = router.complete(
            [
                {"role": "system", "content": build_routing_system(task)},
                {"role": "user", "content": build_routing_prompt(task, graph)},
            ]
        )
        # A containment task's menu is the DISMANTLING pool; routing into the IM
        # pool would return a seed set the executor then rejects
        if task.forecasts:
            menu = prediction_names
        elif task.decodes:
            menu = reconstruction_names
        elif task.recovers:
            menu = localization_names
        elif task.blocks:
            # Only the members whose OUTPUT this lever can emit: routing into the
            # edge selectors from a counter-seeding arm would return arcs the
            # executor then rejects
            from coding_agent.tools.library_api import blocking_names_for

            menu = blocking_names_for(task.budget_op)
        elif task.immunizes:
            # Only the members whose OUTPUT this lever can emit: routing into the
            # edge selectors from a vaccination arm would return arcs the executor
            # then rejects
            from coding_agent.tools.library_api import immunization_names_for

            menu = immunization_names_for(task.epi_lever)
        elif task.contains:
            menu = dismantling_names
        else:
            menu = algorithm_names

        config.baseline = _parse_routing_choice(routing_reply, menu)
        print(f"[run] routing picked {config.baseline!r}")

    # The graph axis of the amortization claim, for every task: the winning
    # program of another run, executed here unmodified. Read before the canned-baseline
    # branch so a transfer arm is a transfer arm regardless of what else was set.
    if config.transfer_from is not None:
        if canned_script is not None:
            raise ValueError(
                "--transfer-from supplies the script to run and cannot be "
                "combined with another canned script"
            )

        source = json.loads(Path(config.transfer_from).read_text())
        canned_script = source.get("script")

        if not canned_script:
            raise ValueError(
                f"{config.transfer_from} has no `script` field to transfer; "
                f"point at an arm's results JSON that synthesized a program"
            )

        print(
            f"[run] TRANSFER: running the winner of "
            f"{source.get('task')}/{source.get('graph', {}).get('graph_id')} "
            f"(arm {source.get('arm')}, its own reward "
            f"{source.get('reward')}) unmodified on this instance"
        )

    if canned_script is not None:
        provider_label = "canned"
    elif config.baseline is not None:
        # Classical-library baseline: same pipeline, envs, and metrics, no LLM
        if config.baseline not in (
            algorithm_names
            + list(adaptive_algorithms)
            + blocking_names
            + dismantling_names
            + immunization_names
            + localization_names
            + prediction_names
            + reconstruction_names
        ):
            raise ValueError(
                f"unknown baseline {config.baseline!r}; choose a static algorithm "
                f"from {algorithm_names}, an adaptive policy from "
                f"{sorted(adaptive_algorithms)}, a blocker from {blocking_names}, a "
                f"dismantler from {dismantling_names}, an immunizer from "
                f"{immunization_names}, a source localizer from "
                f"{localization_names}, a trajectory decoder from "
                f"{reconstruction_names}, or a popularity predictor from "
                f"{prediction_names}"
            )

        provider_label = (
            f"routing:{config.baseline}"
            if config.routing
            else f"baseline:{config.baseline}"
        )

        canned_script = canned_baseline_script(
            config, config.baseline, outbreak, batches
        )
    else:
        provider_label = config.model

    # Canned/baseline scripts are whole free-form Strategies (they call
    # algorithms.*), so they always run in free mode regardless of the flag
    effective_mode = "free" if canned_script is not None else config.strategy_mode

    # A declared baseline (condition 1) or a routing pick (condition 2) IS the
    # expensive algorithm: blocking it would delete the arm rather than speed it
    # up, and its cost is honestly attributed to that arm. The block governs what
    # the agent SYNTHESIZES, not what the harness was told to run.
    effective_allow_mc = config.allow_mc_algorithms or canned_script is not None
    if effective_mode == "scored" and config.method in ("per_step", "windowed"):
        raise ValueError(
            "strategy_mode='scored' generates plan_horizon-only strategies; "
            "use --method one_shot or evolve"
        )

    # Preflight: a model the gateway will not serve should fail here, in
    # seconds, not after the anchor rollouts and the first generation
    if not canned_script:
        verify_gateway_model(config.model)

    provider = (
        _CannedProvider(canned_script)
        if canned_script
        else GatewayProvider(
            config.model,
            temperature=config.temperature,
            reasoning_effort=config.reasoning_effort,
        )
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
        canned=canned_script is not None,
        checkpoint_path=checkpoint_path,
        checkpoint_fingerprint=(
            None
            if checkpoint_path is None
            else checkpoint.fingerprint(config, config.method, graph)
        ),
        instances=select_instances,
    )

    # Optimize the method with the outer-loop coding agent iteration loop to find the best strategy and trajectory result
    print(f"[run] optimizing with {config.method} (provider {provider_label})...")
    strategy, trajectory = method.optimize(agent, environment, task, graph)

    if task.forecasts:
        print(
            f"[run] winner: {config.cp_metric.upper()}={trajectory.reward:.4f} on the "
            f"{config.cp_select_split} split (selection error, LOWER is better)"
        )
    elif task.decodes:
        print(
            f"[run] winner: reward={trajectory.reward:.4f} on the "
            f"{config.cr_select_split} split (kernel likelihood per node minus "
            f"observation violations, higher is better)"
        )
    elif task.recovers:
        print(
            f"[run] winner: consistency={trajectory.reward:.5f} on the "
            f"{config.sl_select_split} split (minus re-simulation error, higher is "
            f"better)"
        )
    else:
        print(
            f"[run] winner: reward={trajectory.reward:.2f} "
            f"({100.0 * trajectory.reward / graph.num_nodes:.2f}% of N"
            f"{', lower is better' if task.contains else ''})"
        )

    # The held-out number, and the one every table reads. Selecting and reporting
    # on the same episodes proves nothing about amortization: a program that
    # memorized specific cascades is indistinguishable from an algorithm until it
    # meets episodes the search never saw.
    heldout = None
    if task.forecasts and evaluate_instances:
        print(
            f"[run] re-running the winner on {len(evaluate_instances)} held-out "
            f"{config.cp_eval_split} cascades..."
        )
        # `fit_examples` stays the SELECTION pool: a fitted predictor may calibrate
        # on the cascades the search saw and never on the ones it is scored against,
        # which is the one leak run_baseline's docstring forbids for external repos
        # and holds identically for our own library rows.
        heldout, _ = evaluate_predictor(
            strategy,
            environment,
            task,
            graph,
            evaluate_instances,
            fit_examples=select_instances,
        )
        print(
            f"[run] held-out {config.cp_metric.upper()}={heldout.reward:.4f} "
            f"(selection {trajectory.reward:.4f}, "
            f"generalization gap {heldout.reward - trajectory.reward:+.4f}, "
            f"POSITIVE means it did worse on cascades the search never saw)"
        )
    elif task.decodes and evaluate_instances:
        print(
            f"[run] re-running the winner on {len(evaluate_instances)} held-out "
            f"{config.cr_eval_split} cascades..."
        )
        heldout, _ = evaluate_reconstructor(
            strategy,
            environment,
            task,
            graph,
            evaluate_instances,
            config.cr_tree_weight,
        )
        print(
            f"[run] held-out reward={heldout.reward:.4f} "
            f"(selection {trajectory.reward:.4f}, "
            f"generalization gap {heldout.reward - trajectory.reward:+.4f})"
        )
    elif task.recovers and evaluate_instances:
        print(
            f"[run] re-running the winner on {len(evaluate_instances)} held-out "
            f"{config.sl_eval_split} episodes..."
        )
        heldout, _ = evaluate_localizer(
            strategy,
            environment,
            task,
            graph,
            evaluate_instances,
            config.sl_budget_mode,
        )
        print(
            f"[run] held-out consistency={heldout.reward:.5f} "
            f"(selection {trajectory.reward:.5f}, "
            f"generalization gap {heldout.reward - trajectory.reward:+.5f})"
        )

    # One closing turn on the generation thread: what it tried each iteration and
    # how the winner works. Canned arms (classical baselines, routing picks) are
    # library algorithms nobody synthesized, and _CannedProvider would answer any
    # question with the script itself, so they get no write-up.
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
        # The results JSON whose program this row replayed, None when the program
        # was written here; recorded for every task, not only source localization
        "transfer_from": config.transfer_from,
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
        # except windowed, whose budget is per window call by design, without
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
        # was trying to do: irrecoverable afterwards, and the first thing worth
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
        # The running design-rule memory the evolve reflections built (empty elsewhere)
        "memory": getattr(method, "memory", ""),
        # Every answered probe, plus the totals the cost story is read on
        "probes": getattr(method, "probe_log", []),
        # Which feedback tier ran, and every diagnostic block it produced. Each
        # entry carries its own `cost` block splitting world-model rollouts from
        # trusted-simulator episodes, which is what keeps the feedback experiment
        # from quietly buying information with simulator calls.
        "feedback_tier": config.feedback,
        "diagnostics": getattr(method, "diagnostic_log", []),
        "probe_calls": sum(
            entry.get("rollouts", 0) for entry in getattr(method, "probe_log", [])
        ),
        "probe_seconds": round(
            sum(entry.get("seconds", 0.0) for entry in getattr(method, "probe_log", [])),
            3,
        ),
        # Probes asked in the probe turn (about the incumbent, answered before
        # the generation was written) as opposed to beside a code block
        "probe_turns": sum(
            1
            for entry in getattr(method, "probe_log", [])
            if entry.get("turn") == "probe"
        ),
        # The search's own bookkeeping under a stochastic evaluator
        # (coding_agent/search_metrics.py): what the naive delta > 0 rule would
        # have accepted against what the band rule did; how well the model
        # forecast its own edits; the incumbent's accepted score against its
        # unbiased re-score; and whether --stop-when-flat ended the search early
        "acceptance_ledger": getattr(method, "ledger", {}),
        "calibration": getattr(method, "calibration", {}),
        "in_loop_optimism": getattr(method, "optimism", {}),
        "stopped_early": getattr(method, "stopped_early", None),
        # Per-node P(infected at end) across the ensemble. Costs n_samples
        # rollouts to produce, so it is serialized rather than recomputed: it
        # is what any post-hoc spatial analysis (coverage, per-community reach)
        # needs. Suppressed on very large graphs where the list dominates the file.
        "final_marginals": (
            trajectory.final_marginals
            if trajectory.final_marginals is not None
            and graph.num_nodes <= max_serialized_marginals
            else None
        ),
        # Inner-loop cost, and the reason this block sits ABOVE the --credit and
        # referee sections rather than below them: both call rollout() again on
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
        # horizon-conditional number is readable instead of only the endpoint.
        # Index t is the count AFTER the step at t - 1.
        "spread_curve": trajectory.spread_curve,
        "spread_at_horizon": (
            trajectory.spread_curve[-1] if trajectory.spread_curve else None
        ),
    }

    # The selection-split number, kept beside the held-out one the three inverse
    # families overwrite `reward` with below: condition 9's fitness scorer
    # (baselines/score_program.py) reads this and never the held-out split
    result["selection_reward"] = trajectory.reward

    # Sense first, because everything downstream that picks a winner needs it and
    # the per-arm JSON is read standalone by plots/report/summary
    result["objective"] = task.sense

    # Rediscovery distance: every callable library member run on the same
    # instance, its output compared with the winner's, on the exact call the
    # condition-1 rows use. Agent arms only: a canned arm IS a library member.
    result["provenance"] = None
    if config.provenance and canned_script is None:
        print("[run] provenance: running the library pool on this instance...")
        result["provenance"] = compute_provenance(
            task,
            graph,
            trajectory,
            strategy.source_script,
            partial(canned_baseline_script, config, outbreak=outbreak, batches=batches),
            lever=(
                config.blocking_lever
                if task.blocks
                else config.epi_lever
                if task.immunizes
                else None
            ),
            timeout=config.provenance_timeout,
        )
        nearest = result["provenance"].get("nearest")
        if nearest is not None:
            print(
                f"[run] provenance: nearest library member {nearest} "
                f"(similarity {result['provenance']['nearest_similarity']:.3f}; "
                f"{len(result['provenance']['errors'])} members failed or timed out)"
            )

    if task.forecasts:
        # Same reasoning as both inverse blocks: the error is measured against a
        # popularity we read off a log, so it carries no evaluator noise and is
        # already comparable across conditions. `referee_reward` is filled from the
        # held-out error so every reader that asks for "the number comparable across
        # arms" gets the right one unchanged.
        selection = trajectory.cost.get("metrics", {})
        reported = heldout if heldout is not None else trajectory
        metrics = reported.cost.get("metrics", {})
        protocol = {}
        try:
            from data.wm_cascades import observed_protocol

            protocol = observed_protocol(config.data_dir)
        except (FileNotFoundError, ValueError):
            protocol = {}

        result["prediction"] = True
        result["metrics"] = metrics
        result["selection_metrics"] = selection
        result["select_split"] = config.cp_select_split
        result["eval_split"] = config.cp_eval_split
        result["n_select_instances"] = len(select_instances)
        result["n_eval_instances"] = len(evaluate_instances)
        result["prediction_metric"] = config.cp_metric
        result["prediction_target"] = task.prediction_target
        result["observation_window"] = task.observation_window
        result["prediction_horizon"] = (
            select_instances[0].observation.horizon if select_instances else None
        )
        # The protocol block, verbatim from the dataset. Published tables differ in
        # five independent ways and four of them are here; a
        # row without them is comparable to nothing, which is why they travel with
        # the number rather than living in a config file.
        result["corpus"] = protocol.get("corpus")
        result["corpus_time_unit"] = protocol.get("time_unit")
        result["split_protocol"] = protocol.get("split_protocol")
        result["observation_seconds"] = protocol.get("observation")
        result["horizon_seconds"] = protocol.get("horizon")
        result["min_observed_filter"] = protocol.get("min_observed")
        result["truncate_filter"] = protocol.get("truncate")
        result["hard_targets"] = protocol.get("hard_targets")
        result["forecast_calls"] = reported.cost.get("forecast_calls")
        result["kernel_calls"] = reported.cost.get("kernel_calls")
        result["kernel_calls_per_instance"] = reported.cost.get(
            "kernel_calls_per_instance"
        )
        result["generalization_gap"] = (
            round(reported.reward - trajectory.reward, 6)
            if heldout is not None
            else None
        )
        result["referee_reward"] = reported.reward
        result["reward"] = reported.reward
        result["referee_reward_se"] = reported.cost.get("reward_se")
        result["spread_pct"] = None
        result["per_instance"] = reported.cost.get("per_instance")
        result["summary"] = summarize(reported, graph, task)
        # The two floors a reader needs to interpret a number at all: under a
        # LOG-space error an instance-blind constant is far stronger than intuition
        # suggests, and "predict what you already see" is right whenever a cascade
        # is finished, which most are.
        result |= trivial_predictor_error(
            evaluate_instances or select_instances,
            select_instances,
            config.cp_metric,
        )

    if task.decodes:
        # The reward was label-free (the arm's kernel likelihood of the decoded
        # history). Only NOW, with the search over and the write-up requested, is
        # the stored history read: path precision, event F1 and the tree-weighted
        # score are computed on the winner for both splits and reported beside
        # the reward, never fed to it. `referee_reward` is the referee's
        # re-measurement under the ground-truth kernel, exactly as a spread's is.
        reconstruction_label_metrics(trajectory, select_instances, config.cr_tree_weight)
        if heldout is not None:
            reconstruction_label_metrics(heldout, evaluate_instances, config.cr_tree_weight)
        selection = trajectory.cost.get("metrics", {})
        reported = heldout if heldout is not None else trajectory
        metrics = reported.cost.get("metrics", {})

        result["reconstruction"] = True
        result["metrics"] = metrics
        result["selection_metrics"] = selection
        result["select_split"] = config.cr_select_split
        result["eval_split"] = config.cr_eval_split
        result["n_select_instances"] = len(select_instances)
        result["n_eval_instances"] = len(evaluate_instances)
        result["observation_setting"] = config.cr_setting
        result["observation_rate"] = config.cr_observation_rate
        result["hidden_rate"] = (
            config.cr_hidden_rate if config.cr_setting == "hidden_nodes" else None
        )
        result["tree_weight"] = config.cr_tree_weight
        result["has_tree_truth"] = bool(
            select_instances and select_instances[0].true_parents is not None
        )
        result["kernel_calls"] = reported.cost.get("kernel_calls")
        result["kernel_calls_per_instance"] = reported.cost.get(
            "kernel_calls_per_instance"
        )
        result["scoring_kernel_calls"] = reported.cost.get("scoring_kernel_calls")
        # The gap on the REWARD (held-out minus selection, both on this arm's
        # evaluator) and on the reported tree score, side by side
        result["generalization_gap"] = (
            round(reported.reward - trajectory.reward, 6)
            if heldout is not None
            else None
        )
        result["tree_score_generalization_gap"] = (
            round(metrics.get("tree_score", 0.0) - selection.get("tree_score", 0.0), 6)
            if heldout is not None and "tree_score" in metrics
            else None
        )
        result["referee_reward"] = None
        result["reward"] = reported.reward
        result["reward_se"] = reported.cost.get("reward_se")
        result["spread_pct"] = None
        result["per_instance"] = reported.cost.get("per_instance")
        result["summary"] = summarize(reported, graph, task)
        # This is a REQUIRED check rather than a diagnostic: a
        # trivial decoder (everyone reachable, parents by BFS) must score badly
        # under the reward, or the reward is wrong. Both rewards in play are
        # reported, in the same file as the result, because a reader cannot
        # interpret a program-search number without them.
        result |= trivial_decoder_reward(
            evaluate_instances or select_instances,
            graph,
            config.cr_tree_weight,
            environment=environment,
        )

    if task.recovers and not task.decodes:
        # The reward was label-free (consistency of the recovered set with the
        # observation on this arm's evaluator). Only NOW, with the search over and
        # the write-up requested, are the stored sources read: F1, precision,
        # recall and AUC are computed on the winner for both splits and reported
        # beside the reward, never fed to it. `referee_reward` is the
        # referee's re-measurement on NDlib, exactly as a spread's is.
        localization_label_metrics(strategy, trajectory, select_instances, graph)
        if heldout is not None:
            localization_label_metrics(strategy, heldout, evaluate_instances, graph)
        selection = trajectory.cost.get("metrics", {})
        reported = heldout if heldout is not None else trajectory
        metrics = reported.cost.get("metrics", {})

        result["localization"] = True
        result["metrics"] = metrics
        result["selection_metrics"] = selection
        result["select_split"] = config.sl_select_split
        result["eval_split"] = config.sl_eval_split
        result["n_select_instances"] = len(select_instances)
        result["n_eval_instances"] = len(evaluate_instances)
        result["observation_mode"] = config.sl_observation
        result["source_budget_mode"] = config.sl_budget_mode
        result["auc_source"] = reported.cost.get("auc_source")
        result["forward_calls"] = reported.cost.get("forward_calls")
        result["forward_calls_per_instance"] = reported.cost.get(
            "forward_calls_per_instance"
        )
        result["scoring_calls"] = reported.cost.get("scoring_calls")
        result["transfer_from"] = config.transfer_from
        # The generalization gap, exposed on the REWARD and on
        # the reported F1: a large negative gap means the program memorized the
        # episodes it was selected on rather than learning an algorithm
        result["generalization_gap"] = (
            round(reported.reward - trajectory.reward, 6)
            if heldout is not None
            else None
        )
        result["f1_generalization_gap"] = (
            round(metrics.get("f1", 0.0) - selection.get("f1", 0.0), 6)
            if heldout is not None and "f1" in metrics
            else None
        )
        result["referee_reward"] = None
        result["reward"] = reported.reward
        result["reward_se"] = reported.cost.get("reward_se")
        result["spread_pct"] = None
        result["per_instance"] = reported.cost.get("per_instance")
        result["summary"] = summarize(reported, graph, task)

    if task.blocks:
        # The prevented-influence block. The reward is the rumour's remaining size
        # (lower is better); PREVENTED influence is that subtracted from the
        # unopposed reference, which is the quantity all five published names refer
        # to. The reference is measured on THIS arm's evaluator, on purpose: a ratio
        # of two different rulers means nothing, and the referee re-measures both on
        # the shared kernel.
        print("[run] unopposed reference sigma(S_N, empty) on this evaluator...")
        unopposed = unopposed_reference(
            environment, task, config.horizon, config.budget
        )
        result["blocking"] = True
        result["competitive_model"] = competitive_model_name(config.positive_prob)
        result["tie_break"] = task.tie_break
        result["attacker"] = config.outbreak_selector
        result |= blocking_metrics(
            trajectory.reward, unopposed, task, graph, trajectory.actions
        )
        print(
            f"[run] blocking: rumour {trajectory.reward:.2f} vs unopposed "
            f"{unopposed:.2f}, prevented "
            f"{result['prevented_influence']:+.2f} "
            f"({result['prevented_pct_of_unopposed']:.1f}% of the cascade, "
            f"lever={result['lever']}, |S_P|/|S_N|={result['budget_ratio']})"
        )

    if task.immunizes:
        # The immunization block. The reward is the attack rate (lower is better);
        # PREVENTED INFECTIONS is that subtracted from the unprotected reference,
        # which is what an immunization table reports. The reference is measured on THIS
        # arm's evaluator, on purpose: a difference of two different rulers is not a
        # quantity, and the referee re-measures both on the shared kernel.
        print("[run] unprotected reference |R(inf)| on this evaluator...")
        unprotected, unprotected_curve = unprotected_reference(
            environment, task, graph, config.horizon, config.budget
        )
        result["epidemic"] = True
        result["compartments"] = config.diffusion_model
        result["epi_beta"] = config.epi_beta
        result["epi_gamma"] = config.epi_gamma
        result["epi_alpha"] = (
            config.epi_alpha if config.diffusion_model == "SEIR" else None
        )
        result["outbreak_selector"] = config.outbreak_selector
        result["outbreak_pct"] = outbreak_pct
        result["contact_reduction"] = (
            config.contact_reduction if task.epi_lever == "contact_reduce" else None
        )
        result["prevalence_curve"] = trajectory.prevalence_curve
        # The curve the doses were measured against, so the report can draw the
        # flattening rather than only the totals
        result["unprotected_prevalence_curve"] = unprotected_curve
        result |= epidemic_metrics(
            trajectory.reward,
            unprotected,
            task,
            graph,
            trajectory.actions,
            trajectory.prevalence_curve,
            # The same burn-in the referee's curve is scored with, or the arm's
            # own endemic prevalence and the ground-truth column disagree
            burn_in=config.epi_burn_in,
        )
        spectral = result.get("spectral") or {}
        print(
            f"[run] epidemic: attack {trajectory.reward:.2f} vs unprotected "
            f"{unprotected:.2f}, prevented "
            f"{result['prevented_infections']:+.2f} "
            f"({result['prevented_pct_of_unprotected']:.1f}% of the outbreak, "
            f"lever={result['lever']}, peak "
            f"{result.get('peak_prevalence', 0.0):.1f} at t="
            f"{result.get('time_to_peak', 0)})"
        )
        if spectral:
            print(
                f"[run] spectral: lambda1 {spectral['lambda1_intact']:.3f} -> "
                f"{spectral['lambda1']:.3f} "
                f"(eigendrop {spectral['eigendrop']:.3f} = "
                f"{spectral['eigendrop_pct']:.1f}%), CONTEXT, not the score"
            )

    if task.contains and not task.blocks and not task.immunizes:
        result["containment"] = True
        result["outbreak"] = list(outbreak)
        result["outbreak_pct"] = outbreak_pct
        result["outbreak_selector"] = config.outbreak_selector
        # The budget's meaning: at or above the ring the row is trivial
        result["outbreak_ring"] = ring_size(graph, outbreak)
        result["ring_fits"] = config.budget >= result["outbreak_ring"]
        # The connectivity functionals reported ALONGSIDE the diffusion
        # number, computed exactly, as context: never as the learned target.
        # Read off the executed bags rather than re-planning, for the same reason
        # the referee replays them: a randomized strategy returns a different set on
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
        # all three
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
    # An inverse or forecast task emits no actions: there is nothing to ablate,
    # and the base rollout would report a spread number beside an F1 or an MSLE
    # Canned rows skip both blocks: nothing renders the credit of a library
    # algorithm, and at the referee sample count it is (k+1) rollouts for nothing
    if (
        config.credit
        and canned_script is None
        and config.evaluator == monte_carlo
        and not (task.recovers or task.forecasts)
    ):
        print(
            "[run] credit skipped under @monte_carlo: one sequential rollout per "
            "action means (k+1) x mc_runs real episodes per report; run the arm "
            "under @world_model or @oracle for batched credit"
        )

    if (
        config.credit
        and canned_script is None
        and config.evaluator != monte_carlo
        and not (task.recovers or task.forecasts)
    ):
        print("[run] per-action counterfactual credit (one rollout per action)...")
        # Credit of the executed action sequence; for state-dependent strategies (per_step/windowed) the recorded bags are replayed as a fixed plan
        limit = credit_limit(environment, graph)
        base_reward, entries = counterfactual_credit(
            environment,
            trajectory.actions,
            config.horizon,
            config.budget,
            creditable_ops=tuple(config.allowed_ops),
            limit=limit,
            graph=graph,
        )
        # Solo cascades and stagnation timing ride along on batched evaluators
        augment_solo(
            environment,
            trajectory.actions,
            entries,
            config.horizon,
            config.budget,
            creditable_ops=tuple(config.allowed_ops),
            limit=limit,
            graph=graph,
        )
        result["credit_base_reward"] = base_reward
        result["credit"] = entries
        result["credit_limit"] = limit
    # Every arm's winner is replayed on the SHARED referee, the exact oracle
    # simulator by default: the one number comparable across conditions. A native
    # arm's own reward is one noisy episode, a monte_carlo arm's carries the
    # winner's curse from being the max over outer iterations, a world-model arm's
    # is a model estimate. `--mc-agreement` adds NDlib on the same winner, which is
    # the independent reference the oracle is measured against and the timing row.
    if task.forecasts:
        # THE number this task is actually about, and the one no other task in this repo
        # can produce. The reward already measures how good a PROGRAM is; this
        # measures how good the MODEL is: roll the arm's own forward model forward
        # from each observed prefix with no program in the loop, and compare its
        # expected popularity against what the log says happened. That difference is
        # MODELLING error against a process that is not IC, which is exactly the
        # closed loop every other task cannot break: they evaluate a learned model
        # against traces drawn from the simulator that trained it.
        #
        # Reported beside the program's own error because the two are separable and
        # a reader cannot tell them apart from the reward column alone: a strong
        # program on a badly misspecified kernel and a weak program on a good one
        # post the same number.
        # Only for arms that actually HAVE a forward model in their search loop.
        # A condition-1 baseline's "own forward model" is just the shared evaluator,
        # so the number would be identical across every classical row and computing
        # it once per row is pure waste: on a 30K-node graph under @monte_carlo it
        # is `instances x samples x steps x mc_runs` real episodes, which is millions.
        if (
            config.mc_agreement
            and task.forward_model
            and canned_script is None
            and config.baseline is None
        ):
            print("[run] modelling-error check (the arm's own forward model, no program)...")
            result |= referee_modelling_error(
                environment,
                task,
                evaluate_instances or list(task.instances),
                # A DIAGNOSTIC rather than the reward, so it runs at a fixed small
                # sample count instead of the search's: doubling its precision buys
                # nothing a reader would act on, and under @monte_carlo it
                # is the single most expensive thing in the run.
                samples=referee_forecast_samples,
            )
            model_msle = result.get("model_msle")
            if model_msle is not None:
                print(
                    f"[run] model_msle={model_msle:.4f} against the program's "
                    f"{result['reward']:.4f}, the gap is what the SEARCH bought on "
                    f"top of the kernel; the LEVEL is how far an IC-shaped kernel is "
                    f"from a real adoption process"
                )
        elif config.mc_agreement and not task.forward_model:
            print(
                "[run] @native has no forward model, so there is no modelling error "
                "to measure: that is the condition, not a gap"
            )
    elif task.decodes or task.recovers:
        # The held-out reward re-measured with the referee's own kernel in place
        # of the arm's evaluator (every decoded history, or every recovered set,
        # re-scored), which is to `reward` what the replay below is to a spread.
        referee = _build_referee(config, graph, task, outbreak)
        reported = heldout if heldout is not None else trajectory
        referee_instances = evaluate_instances or list(task.instances)
        per_instance = reported.cost.get("per_instance", [])
        result["referee"] = config.referee
        result["referee_samples"] = config.referee_samples
        print(
            f"[run] {config.referee} referee ({config.referee_samples} samples per "
            f"kernel call)..."
        )

        def measure(kernel_environment: object) -> dict:
            if task.decodes:
                return referee_likelihood(
                    kernel_environment, task, referee_instances, per_instance
                ) | referee_reconstruction_error(
                    kernel_environment, task, referee_instances, per_instance
                )
            return referee_resimulation_error(
                kernel_environment, task, referee_instances, per_instance
            )

        result |= measure(referee)
        result["arm_minus_referee"] = reported.reward - result["referee_reward"]
        print(
            f"[run] referee reward={result['referee_reward']:.5f} "
            f"(arm's own {reported.reward:.5f}); resim_error="
            f"{result.get('resim_error', float('nan')):.5f} (true sources score "
            f"{result.get('resim_error_true_sources', float('nan')):.5f})"
        )

        if config.mc_agreement:
            agreement_runs = config.mc_agreement_runs or config.mc_runs
            print(f"[run] NDlib agreement check ({agreement_runs} runs per kernel call)...")
            result |= _agreement_keys(
                measure(
                    MonteCarloEnvironment(
                        graph,
                        config.diffusion_model,
                        mc_runs=agreement_runs,
                        base_seed=config.seed,
                        remove_semantics=config.remove_semantics,
                    )
                )
            )
            result["mc_agreement_runs"] = agreement_runs
            result["referee_minus_mc"] = result["referee_reward"] - result["mc_reward"]
            print(
                f"[run] mc_reward={result['mc_reward']:.5f} "
                f"(referee_minus_mc={result['referee_minus_mc']:+.5f})"
            )
    else:
        referee_environment = _build_referee(config, graph, task, outbreak)
        result["referee"] = config.referee
        result["referee_samples"] = config.referee_samples

        # Replay the actions that EARNED the reward rather than re-planning. A
        # generated script that samples (RIS with a live seed, a randomized local
        # search) returns a different seed set on a second call, so re-planning
        # would referee a strategy that never ran, and for per_step it would fire
        # a fresh LLM call per (run, timestep) of the replay.
        action_fn = partial(planned_action, trajectory.actions)

        # A canned arm (a library baseline, a routed pick, an external repo's seed
        # set) was scored exactly once, on the referee's own environment at the
        # referee's sample count, so that trajectory IS the referee replay and a
        # second rollout would buy nothing but time.
        if (
            canned_script is not None
            and config.evaluator == config.referee
            and config.n_samples >= config.referee_samples
        ):
            referee_trajectory = trajectory
            print(f"[run] canned arm: its {config.referee} evaluation is the referee replay")
        else:
            print(f"[run] {config.referee} referee replay ({config.referee_samples} samples)...")
            referee_trajectory = referee_environment.rollout(
                action_fn, config.horizon, config.budget
            )

        result["referee_reward"] = referee_trajectory.reward
        result["referee_seed"] = referee_trajectory.cost["seed"]
        result["referee_spread_pct"] = round(
            100.0 * referee_trajectory.reward / graph.num_nodes, 2
        )
        result["referee_reward_se"] = referee_trajectory.cost["reward_se"]
        result["referee_rollout_seconds"] = referee_trajectory.cost["rollout_seconds"]
        result["arm_minus_referee"] = trajectory.reward - referee_trajectory.reward
        print(
            f"[run] referee_reward={referee_trajectory.reward:.2f} "
            f"±{referee_trajectory.cost['reward_se']:.2f} "
            f"(arm_minus_referee={result['arm_minus_referee']:+.2f})"
        )

        if task.immunizes:
            # The prevented-infections column every immunization table reports, on
            # the SHARED referee. Both terms are re-measured here rather than
            # reusing the arm's own reference, because prevented infections is a
            # DIFFERENCE and a difference of two evaluators' numbers is not a
            # quantity. The curve comes back too, so peak and time-to-peak are
            # ground-truth rather than model-predicted.
            referee_unprotected, referee_unprotected_curve = unprotected_reference(
                referee_environment, task, graph, config.horizon, config.budget
            )
            result["referee_unprotected_prevalence_curve"] = referee_unprotected_curve
            result["referee_unprotected_attack_rate"] = referee_unprotected
            result["referee_prevented_infections"] = referee_unprotected - referee_trajectory.reward
            result["referee_prevented_pct_of_unprotected"] = (
                100.0 * (referee_unprotected - referee_trajectory.reward) / referee_unprotected
                if referee_unprotected
                else 0.0
            )
            result["referee_prevalence_curve"] = referee_trajectory.prevalence_curve
            result["referee_curve"] = epidemic_curve_metrics(
                referee_trajectory.prevalence_curve or [], graph.num_nodes, config.epi_burn_in
            )
            print(
                f"[run] ground-truth prevented infections: "
                f"{result['referee_prevented_infections']:+.2f} of {referee_unprotected:.2f} "
                f"({result['referee_prevented_pct_of_unprotected']:.1f}%), peak "
                f"{result['referee_curve']['peak_prevalence']:.1f} at t="
                f"{result['referee_curve']['time_to_peak']}"
            )

        if task.blocks:
            # The prevented-influence column every blocking table reports, on the
            # SHARED referee. Both terms are re-measured here rather than reusing the
            # arm's own reference, because prevented influence is a DIFFERENCE and a
            # difference of two evaluators' numbers is not a quantity.
            referee_unopposed = unopposed_reference(
                referee_environment, task, config.horizon, config.budget
            )
            result["referee_unopposed_spread"] = referee_unopposed
            result["referee_prevented_influence"] = referee_unopposed - referee_trajectory.reward
            result["referee_prevented_pct_of_unopposed"] = (
                100.0 * (referee_unopposed - referee_trajectory.reward) / referee_unopposed
                if referee_unopposed
                else 0.0
            )
            print(
                f"[run] ground-truth prevented influence: "
                f"{result['referee_prevented_influence']:+.2f} of {referee_unopposed:.2f} "
                f"({result['referee_prevented_pct_of_unopposed']:.1f}%)"
            )
        if config.mc_agreement:
            # The same winner on NDlib: the independent reference the oracle is
            # judged against, and the per-rollout timing the efficiency table reads.
            # Same dynamics as the arm (two cascades, or recovery), or the column
            # would compare processes rather than implementations.
            agreement_runs = config.mc_agreement_runs or config.mc_runs
            monte_carlo_environment = MonteCarloEnvironment(
                graph,
                config.diffusion_model,
                mc_runs=agreement_runs,
                base_seed=config.seed,
                remove_semantics=config.remove_semantics,
                negative_seeds=outbreak,
                competitive_config=(
                    _competitive_config(config) if task.blocks else None
                ),
                epidemic_config=(
                    _epidemic_config(config) if task.immunizes else None
                ),
            )
            if config.campaigns > 1:
                monte_carlo_environment = MultiRoundEnvironment(
                    monte_carlo_environment,
                    campaigns=config.campaigns,
                    base_seed=config.seed,
                )

            print(f"[run] NDlib agreement check ({agreement_runs} runs)...")
            mc_trajectory = monte_carlo_environment.rollout(
                action_fn, config.horizon, config.budget
            )
            result["mc_agreement_runs"] = agreement_runs
            result["mc_reward"] = mc_trajectory.reward
            result["mc_reward_se"] = mc_trajectory.cost["reward_se"]
            result["mc_rollout_seconds"] = mc_trajectory.cost["rollout_seconds"]
            result["mc_seed"] = mc_trajectory.cost["seed"]
            result["referee_minus_mc"] = referee_trajectory.reward - mc_trajectory.reward
            print(
                f"[run] mc_reward={mc_trajectory.reward:.2f} "
                f"±{mc_trajectory.cost['reward_se']:.2f} "
                f"(referee_minus_mc={result['referee_minus_mc']:+.2f}, "
                f"{mc_trajectory.cost['rollout_seconds']:.1f}s against the referee's "
                f"{referee_trajectory.cost['rollout_seconds']:.1f}s)"
            )

        # Fidelity re-evaluation only means something for a model-based evaluator:
        # it measures how far the MODEL is from truth. For a monte_carlo evaluator
        # the "model" is the simulator itself, so there is nothing to measure.
        if config.evaluator in (world_model, oracle) and canned_script is None:
            print(
                f"[run] re-evaluating winner on {wm_reeval_seeds} fresh "
                f"{config.evaluator} seeds..."
            )

            # `reward` is the max over outer iterations, all evaluated at rollout
            # seed 0: it carries selection optimism (winner's curse) plus that one
            # seed's persistent luck. Re-evaluating the winner on fresh seeds gives
            # the unbiased WM estimate: judge evaluator fidelity by
            # wm_reeval_minus_referee, not arm_minus_referee.
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
            result["wm_reeval_minus_referee"] = (
                result["wm_reeval_mean"] - referee_trajectory.reward
            )
            print(
                f"[run] wm_reeval_mean={result['wm_reeval_mean']:.2f} "
                f"(reeval_minus_mc={result['wm_reeval_minus_referee']:+.2f})"
            )

    # Whole experiment including LLM calls; the per-rollout WM-vs-MC timing lives in cost.rollout_seconds / referee_rollout_seconds
    result["elapsed_seconds"] = time.perf_counter() - experiment_start + getattr(
        method, "resumed_elapsed", 0.0
    )

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
        default=default_model,
        help=f"gateway model name (default: {default_model}; gpt-5.6-sol, gpt-5.6-terra and gpt-5.6-luna are the alternatives the gateway served before it).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="LLM sampling temperature; 0.0 = greedy decoding, omit for the provider default (default: None).",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=default_reasoning_effort,
        choices=[*reasoning_efforts, "none"],
        help=f"reasoning effort for the OpenAI reasoning family; 'none' sends nothing (default: {default_reasoning_effort}).",
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
        choices=(
            algorithm_names
            + list(adaptive_algorithms)
            + blocking_names
            + dismantling_names
            + immunization_names
            + localization_names
            + prediction_names
            + reconstruction_names
        ),
        metavar="NAME",
        help="evaluate this classical library algorithm instead of an LLM strategy: "
        "a static IM algorithm, a per-round adaptive policy, an influence blocker, a "
        "network dismantler, an immunizer, a source localizer, a trajectory "
        "decoder, or a popularity predictor (default: None).",
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
        choices=[
            "one_shot",
            "per_step",
            "windowed",
            "evolve",
            "adaptive",
        ],
        help="outer-loop method; evolve = population edits with refine/restructure "
        "operators; adaptive = the same search over a per-round policy for adaptive "
        "IM (default: one_shot).",
    )
    parser.add_argument(
        "--native-arm",
        action="store_true",
        help="mark this run as the @native condition: the harness scores with one "
        "real episode and no forward model exists for the canned kernel-using "
        "baselines. Pair with --evaluator monte_carlo --mc-runs 1 (default: False).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="influence_maximization",
        choices=task_names(),
        help="key of pipeline.tasks.tasks: sets the objective sign, what a unit of "
        "budget buys, the outbreak, the simulator family and the contract the "
        "agent writes against (default: influence_maximization).",
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
    # Influence blocking
    parser.add_argument(
        "--blocking-lever",
        type=str,
        default=counter_seed,
        choices=list(valid_levers),
        help="influence blocking: which of the four published interventions the "
        "budget buys. counter_seed = seed a competing cascade (the founding and "
        "largest sub-literature); node_block = delete nodes (the IMIN line); "
        "edge_block = cut arcs (Kimura); weight_block = reduce arc probabilities "
        f"(DiffIM's continuous relaxation) (default: {counter_seed}).",
    )
    parser.add_argument(
        "--tie-break",
        type=str,
        default=auto_dominance,
        choices=list(tie_break_choices),
        help="influence blocking: which cascade wins a node both reach on the same "
        "step. auto = each dynamics' own founding paper (positive under IC, negative "
        f"under LT). Must match the checkpoint's (default: {auto_dominance}).",
    )
    parser.add_argument(
        "--positive-prob",
        type=str,
        default=shared_positive_prob,
        help="influence blocking: the blocker's per-edge transmission probability. "
        "'shared' is COICM; a float is MCICM, and 1.0 is Budak's high-effectiveness "
        f"property (default: {shared_positive_prob}).",
    )
    parser.add_argument(
        "--detection-delay",
        type=int,
        default=0,
        help="influence blocking: Budak's r, the rumour is detected r steps late "
        "and anything the blocker emits before then is dropped. This is the axis "
        "that makes the first-mover advantage measurable (default: 0).",
    )
    # Epidemic control
    parser.add_argument(
        "--epi-lever",
        type=str,
        default=vaccinate,
        choices=list(valid_epidemic_levers),
        help="epidemic control: which of the four interventions the budget buys. "
        "vaccinate = the node is immune, leaves the graph and is never counted; "
        "quarantine = the node is ISOLATED but stays in the graph and stays "
        "counted; edge_cut = cut arcs (Kimura, NetMelt, Van Mieghem); "
        "contact_reduce = scale arc probabilities down (social distancing, the "
        f"lever NDlib's compartmental models cannot express) (default: {vaccinate}).",
    )
    parser.add_argument(
        "--contact-reduction",
        type=float,
        default=default_contact_reduction,
        help="epidemic control: the multiplier --epi-lever contact_reduce writes. "
        "0 is a full cut through the weight channel and is directly comparable to "
        f"edge_cut at the same k (default: {default_contact_reduction}).",
    )
    parser.add_argument(
        "--epi-beta",
        type=float,
        default=1.0,
        help="epidemic control: multiplier on the graph's per-arc probability, so "
        "beta_uv = clip(scale * p(u->v)) (default: 1.0).",
    )
    parser.add_argument(
        "--epi-gamma",
        type=float,
        default=0.3,
        help="epidemic control: rate of LEAVING I, recovery under SIR/SEIR, "
        "return-to-susceptible under SIS. 1.0 under SIR reproduces IC exactly "
        "(default: 0.3).",
    )
    parser.add_argument(
        "--epi-alpha",
        type=float,
        default=0.5,
        help="epidemic control: E -> I rate, SEIR only (default: 0.5).",
    )
    parser.add_argument(
        "--epi-burn-in",
        type=float,
        default=default_burn_in,
        help="epidemic control: fraction of the prevalence curve discarded before "
        "the endemic prevalence is time-averaged. SIS has no terminal state, so "
        f"final size is undefined there (default: {default_burn_in}).",
    )
    # Source localization
    parser.add_argument(
        "--sl-select-split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="source localization: episodes the outer loop's reward is computed on "
        "(default: train).",
    )
    parser.add_argument(
        "--sl-eval-split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="source localization: HELD-OUT episodes the winning program is re-run "
        "on, unmodified. This is the reported number; selecting and reporting on "
        "the same episodes proves nothing about amortization (default: test).",
    )
    parser.add_argument(
        "--sl-instances",
        type=int,
        default=20,
        help="source localization: labelled episodes per split. Every candidate "
        "program pays this many executions, so it is the M of the P x M x C search "
        "cost (default: 20).",
    )
    parser.add_argument(
        "--sl-observation",
        type=str,
        default="marginal",
        choices=list(valid_observations),
        help="source localization: which y the program sees. marginal is the "
        "MC-averaged P(infected); binary is a single realized draw and is the "
        "column comparable to the published tables (default: marginal).",
    )
    parser.add_argument(
        "--sl-budget-mode",
        type=str,
        default=episode_budget,
        choices=list(valid_budget_modes),
        help="source localization: where k comes from. episode = the instance's own "
        "source count (the published given-k convention); sweep = the pipeline's k, "
        "with the episode pool filtered to that source fraction "
        f"(default: {episode_budget}).",
    )
    parser.add_argument(
        "--sl-source-tolerance",
        type=float,
        default=0.5,
        help="source localization: relative band around k that --sl-budget-mode "
        "sweep keeps an episode in (default: 0.5).",
    )
    parser.add_argument(
        "--transfer-from",
        "--sl-transfer-from",
        dest="transfer_from",
        type=str,
        default=None,
        help="run the winning program named by ANOTHER arm's results JSON, "
        "unmodified, on this dataset, for any task. The graph axis of the "
        "amortization claim, and a comparison no per-instance method can enter. "
        "--sl-transfer-from is the historical spelling (default: None).",
    )
    # Cascade reconstruction
    parser.add_argument(
        "--cr-setting",
        type=str,
        default=partial_times,
        choices=list(valid_settings),
        help="cascade reconstruction: WHICH of the four observation regimes is "
        "masked. partial_times = a subsample of the infected set with activation "
        "times (the ordered-Steiner regime); partial_nodes = the same subsample "
        "with times withheld; final_snapshot = the terminal state only (DITTO's "
        "DASH, the hardest published formulation); hidden_nodes = partial_times "
        "plus nodes deleted from the graph. These are four PROTOCOLS and their "
        f"rows are never pooled (default: {partial_times}).",
    )
    parser.add_argument(
        "--cr-observation-rate",
        type=float,
        default=default_observation_rate,
        help="cascade reconstruction: probability an infected node IS REPORTED. "
        "Stated in that direction on purpose: this literature uses the symbol "
        "sigma for both the report rate and its complement, so a curve read "
        f"backwards is a real hazard (default: {default_observation_rate}).",
    )
    parser.add_argument(
        "--cr-hidden-rate",
        type=float,
        default=default_hidden_rate,
        help="cascade reconstruction: fraction of non-source nodes DELETED from "
        "the graph under --cr-setting hidden_nodes. Distinct from the observation "
        "rate: an unobserved node can still be inferred, a hidden one is not there "
        f"(default: {default_hidden_rate}).",
    )
    parser.add_argument(
        "--cr-instances",
        type=int,
        default=20,
        help="cascade reconstruction: masked cascades per split. Every candidate "
        "decoder pays this many executions (default: 20).",
    )
    parser.add_argument(
        "--cr-select-split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="cascade reconstruction: episodes the outer loop's reward is computed "
        "on (default: train).",
    )
    parser.add_argument(
        "--cr-eval-split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="cascade reconstruction: HELD-OUT episodes the winning decoder is "
        "re-run on unmodified, and the number every table reports (default: test).",
    )
    parser.add_argument(
        "--cp-select-split",
        type=str,
        default="train",
        help="cascade prediction: which logged cascades the outer loop optimizes "
        "the predictor on (default: train).",
    )
    parser.add_argument(
        "--cp-eval-split",
        type=str,
        default="test",
        help="cascade prediction: HELD-OUT cascades the winning predictor is re-run "
        "on unmodified, and the number every table reports (default: test).",
    )
    parser.add_argument(
        "--cp-instances",
        type=int,
        default=40,
        help="cascade prediction: logged cascades one reward evaluation sweeps "
        "over, per split (default: 40).",
    )
    parser.add_argument(
        "--cp-observation-steps",
        type=int,
        default=0,
        help="cascade prediction: replayed timesteps of each cascade the predictor "
        "sees. 0 reads it off the dataset's own `observed` block, which is where "
        "the corpus's published window landed after binning. The "
        "literature pairs TWO windows per corpus deliberately, so "
        "sweep this rather than picking one (default: 0).",
    )
    parser.add_argument(
        "--cp-metric",
        type=str,
        default=default_prediction_metric,
        choices=list(valid_prediction_metrics),
        help="cascade prediction: which error the reward IS. All of them MINIMIZE. "
        "`msle` is CasFlow's own code (log2, clamped at 1, no offset); "
        "`msle_offset` is CasFT's stated log2(P+1); `msle_natural` is CTCP's loss. "
        "Those three are printed "
        "under one name in one published table "
        f"(default: {default_prediction_metric}).",
    )
    parser.add_argument(
        "--cp-target",
        type=str,
        default=increment_target,
        choices=list(valid_targets),
        help="cascade prediction: which quantity the reported table is about. "
        "`increment` is CasFlow's own label (P(t_p) - P(t_o)) and `total` is "
        "CasFT's Eq. 26; 5.7 difference 3 records that the two share a symbol and "
        f"are not the same quantity (default: {increment_target}).",
    )
    parser.add_argument(
        "--cp-forecast-samples",
        type=int,
        default=default_forecast_samples,
        help="cascade prediction: unrolls averaged inside one forecast_marginals "
        "call. Each costs `steps` metered kernel evaluations, so an @monte_carlo "
        "arm pays steps x samples x mc_runs real episodes per call and a "
        f"@world_model arm pays steps x samples matmuls "
        f"(default: {default_forecast_samples}).",
    )
    parser.add_argument(
        "--cr-tree-weight",
        type=float,
        default=0.6,
        help="cascade reconstruction: lambda in "
        "lambda * PathPrecision + (1 - lambda) * EventF1. At least 0.5 is a "
        "REQUIREMENT: the node set is nearly free, so a search rewarded mostly on "
        "Event F1 discovers the tree contributes nothing and converges on decoders "
        "that never attempt it. 0 acknowledges a node-only protocol explicitly and "
        "is the only value that runs without a transmission edge in the data "
        "(default: 0.6).",
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
        choices=["IC", "LT"] + list(epidemic_dynamics),
        help="dynamics. SIR/SIS/SEIR run the compartmental simulator and select "
        "the compartment head (default: IC).",
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
        default=20,
        help="outer refinement iterations (default: 20).",
    )
    parser.add_argument(
        "--mc-runs",
        type=int,
        default=200,
        help="Monte Carlo simulator runs; ~(spread_std/target_se)^2, dial down for large graphs with --evaluator monte_carlo (default: 200).",
    )
    parser.add_argument(
        "--referee",
        type=str,
        default=oracle,
        choices=[oracle, monte_carlo],
        help="the shared referee every arm's winner is replayed on (default: oracle).",
    )
    parser.add_argument(
        "--referee-samples",
        type=int,
        default=referee_samples_default,
        help=f"rollout samples for the referee replay (default: {referee_samples_default}).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="world-model rollout samples (default: 200).",
    )
    parser.add_argument(
        "--max-block-arc-hidden",
        type=int,
        default=default_max_block_arc_hidden,
        help="arcs x hidden units one world-model rollout block may hold; more plans "
        f"run as several blocks with identical results (default: {default_max_block_arc_hidden}).",
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
        "--mc-agreement",
        action="store_true",
        help="also replay the final strategy on NDlib Monte Carlo: the independent check on the oracle referee, and the timing row (default: False).",
    )
    parser.add_argument(
        "--mc-agreement-runs",
        type=int,
        default=None,
        help="NDlib runs for --mc-agreement (default: --mc-runs).",
    )
    parser.add_argument(
        "--credit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="per-action counterfactual credit in every refinement iteration and the results JSON: leave-one-out delta per action, each action's solo cascade and when it stops growing, and the learned kernel's bottleneck nodes. Batched on the world-model and oracle evaluators; skipped with a notice under @monte_carlo, where one sequential rollout per action would add hours per arm. --no-credit turns it off (default: True).",
    )
    parser.add_argument(
        "--feedback",
        type=str,
        default=feedback_default,
        choices=list(valid_feedback_tiers),
        help="what each refinement generation is told after its candidate is "
        "scored. `default` is the full summarize() feedback with the paired verdict, "
        "reference diff, credit and counterexamples. The f0-f3 ladder is the controlled "
        "variable of the feedback experiment: f0 = scalar reward only, f1 = + "
        "per-seed leave-one-out contribution, f2 = + regional coverage, f3 = + "
        "seed overlap, bridge coverage and stagnation. Every ladder rung is "
        "world-model work only and is counted in `diagnostics` "
        "(default: default).",
    )
    parser.add_argument(
        "--probe-turn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="before every evolve generation, one trace-free aside in which the "
        "model asks up to six what-if probes about the incumbent's plan and gets "
        "the answers in the same generation's prompt. One extra LLM call per "
        "generation; the write-turn probes block still works either way. "
        "--no-probe-turn turns it off (default: True).",
    )
    parser.add_argument(
        "--stop-when-flat",
        type=int,
        default=0,
        help="stop an evolve search once the incumbent's re-score on the fresh "
        "realization (the unbiased estimate) has not improved beyond the band "
        "for this many consecutive generations; 0 runs every generation "
        "(default: 0).",
    )
    parser.add_argument(
        "--calibration-steering",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="when the model's last three forecasts of its own edits (the "
        "`# EXPECTED:` line) were wrong in sign more often than not, count one "
        "extra stalled generation in the operator schedule so an explore move "
        "becomes likelier (default: False).",
    )
    parser.add_argument(
        "--provenance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="after the search, run every callable member of the task's library "
        "pool on the same instance and record the similarity of each one's "
        "output to the winner's (the rediscovery distance), plus the members the "
        "winning source calls. --no-provenance skips it (default: True).",
    )
    parser.add_argument(
        "--provenance-timeout",
        type=float,
        default=60.0,
        help="seconds one library member may take in the provenance check before "
        "it is recorded as timed out (default: 60.0).",
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
        # A native run has already been told --evaluator monte_carlo; the arm
        # name is what the results JSON stamps as the condition, so it has to say
        # native or the row lands in condition 4
        arm_evaluator = "native" if args.native_arm else args.evaluator
        arm_spec = f"{args.method}_{args.strategy_mode}@{arm_evaluator}"

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
        reasoning_effort=None if args.reasoning_effort == "none" else args.reasoning_effort,
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
        blocking_lever=args.blocking_lever,
        tie_break=args.tie_break,
        positive_prob=args.positive_prob,
        detection_delay=args.detection_delay,
        epi_lever=args.epi_lever,
        contact_reduction=args.contact_reduction,
        epi_beta=args.epi_beta,
        epi_gamma=args.epi_gamma,
        epi_alpha=args.epi_alpha,
        epi_burn_in=args.epi_burn_in,
        sl_select_split=args.sl_select_split,
        sl_eval_split=args.sl_eval_split,
        sl_instances=args.sl_instances,
        sl_observation=args.sl_observation,
        sl_budget_mode=args.sl_budget_mode,
        sl_source_tolerance=args.sl_source_tolerance,
        transfer_from=args.transfer_from,
        cr_setting=args.cr_setting,
        cr_observation_rate=args.cr_observation_rate,
        cr_hidden_rate=args.cr_hidden_rate,
        cr_instances=args.cr_instances,
        cr_select_split=args.cr_select_split,
        cr_eval_split=args.cr_eval_split,
        cp_select_split=args.cp_select_split,
        cp_eval_split=args.cp_eval_split,
        cp_instances=args.cp_instances,
        cp_observation_steps=args.cp_observation_steps,
        cp_metric=args.cp_metric,
        cp_target=args.cp_target,
        cp_forecast_samples=args.cp_forecast_samples,
        cr_tree_weight=args.cr_tree_weight,
        native_arm=args.native_arm,
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
                n_samples=args.n_samples,
        max_block_arc_hidden=args.max_block_arc_hidden,
        seed=args.seed,
        device=args.device,
        data_dir=args.data_dir,
        graph_id=args.graph_id,
        wm_results_json=args.wm_results_json,
        referee=args.referee,
        referee_samples=args.referee_samples,
        mc_agreement=args.mc_agreement,
        mc_agreement_runs=args.mc_agreement_runs,
        credit=args.credit,
        probe_turn=args.probe_turn,
        stop_when_flat=args.stop_when_flat,
        calibration_steering=args.calibration_steering,
        provenance=args.provenance,
        provenance_timeout=args.provenance_timeout,
        out_json=args.out_json,
        arm_spec=arm_spec,
    )

    print(json.dumps(run_experiment(config), indent=2, default=str))
