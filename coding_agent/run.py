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
    select_outbreak,
)
from coding_agent.credit import counterfactual_credit, planned_action
from coding_agent.epidemic import (
    default_contact_reduction,
    epidemic_metrics,
    resolve_lever as resolve_epidemic_lever,
    unprotected_reference,
    valid_levers as valid_epidemic_levers,
    vaccinate,
)
from coding_agent.envs.monte_carlo_env import MonteCarloEnvironment
from coding_agent.envs.multi_round_env import MultiRoundEnvironment
from coding_agent.envs.world_model_env import WorldModelEnvironment
from coding_agent.localization import (
    episode_budget,
    evaluate_localizer,
    load_instances,
    referee_resimulation_error,
    valid_budget_modes,
    valid_observations,
)
from coding_agent.reconstruction import (
    default_hidden_rate,
    default_observation_rate,
    evaluate_reconstructor,
    load_cascades,
    partial_times,
    valid_settings,
)
from coding_agent.reconstruction import (
    referee_resimulation_error as referee_reconstruction_error,
)
from coding_agent.reconstruction import trivial_decoder_reward
from coding_agent.methods.base import OuterLoopMethod, summarize
from coding_agent.methods.decode import BarycenterDecoding
from coding_agent.methods.evolve import EvolveSearch
from coding_agent.methods.gradient import GradientInversion
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
from coding_agent.tools.blocking_algorithms import (
    all_blocking_algorithms,
    blocking_levers,
    blocking_shape,
    emittable,
    lever_shape,
)
from coding_agent.epidemic import lever_shape as epidemic_lever_shape
from coding_agent.tools.dismantling_algorithms import dismantling_algorithms
from coding_agent.tools.immunization_algorithms import (
    immunization_algorithms,
    immunization_levers,
    immunization_shape,
)
from coding_agent.tools.immunization_algorithms import (
    emittable as immunization_emittable,
)
from coding_agent.tools.library_api import (
    algorithm_names,
    blocking_names,
    dismantling_names,
    immunization_names,
    localization_names,
    reconstruction_names,
)
from coding_agent.tools.localization_algorithms import localization_algorithms
from coding_agent.tools.reconstruction_algorithms import reconstruction_algorithms
from coding_agent.types import GraphInfo, TaskSpec, full_adoption, valid_feedback_models
from data.wm_competitive import (
    CompetitiveConfig,
    auto_dominance,
    competitive_model_name,
    resolve_tie_break,
    shared_positive_prob,
    tie_break_choices,
)
from data.wm_epidemic import EpidemicConfig, default_burn_in, endemic_prevalence
from data.wm_simulator import (
    epidemic_dynamics,
    spent,
    valid_action_ops,
    valid_remove_semantics,
)
from pipeline.conditions import parse_arm
from pipeline.layout import budget_label, checkpoint_suffix
from pipeline.tasks import get_task, maximize
from world_model.wm_data import load_graph_store
from world_model.wm_metrics import containment_metrics, epidemic_curve_metrics
from world_model.wm_sl import valid_priors, vae_prior

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
    # Influence blocking. `outbreak_pct` / `outbreak_selector` above double as |S_N|
    # and the ATTACKER MODEL — §8.3's second experimental axis, which IM does not
    # have — so nothing new is needed for those. What is new is the lever (which of
    # §1.1's four interventions the budget buys), the tie-break, `p_L`, and Budak's
    # detection delay. All four are inert unless the task is competitive.
    blocking_lever: str = counter_seed
    tie_break: str = auto_dominance
    positive_prob: str = shared_positive_prob
    detection_delay: int = 0
    # Epidemic control. `outbreak_pct` / `outbreak_selector` above double as the
    # index-case count and the OUTBREAK MODEL, so nothing new is needed for those.
    # What is new is the lever (which of §2.5's four interventions the budget buys)
    # and the three compartmental rates, which §8.2 trap 2 says must be reported
    # rather than left implicit — beta and gamma are free parameters nobody
    # standardizes, so a table that fixes them without saying so is comparable only
    # to itself. All are inert unless the task registry marks the task epidemic.
    epi_lever: str = vaccinate
    epi_beta: float = 1.0
    epi_gamma: float = 0.3
    epi_alpha: float = 0.5
    epi_burn_in: float = default_burn_in
    contact_reduction: float = default_contact_reduction
    # Source localization. The two SPLITS are the load-bearing pair
    # (research/source_localization.md §8.5.1): the outer loop's reward is computed
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
    # Arm A knobs; read only by method="gradient". `none` reproduces SL-VAE (a)
    # (forward model + descent, no prior) and `vae` the full method, which is
    # SL-VAE's own ablation and worth +0.19 F1 on Jazz in its Table 4.
    sl_prior: str = vae_prior
    sl_steps: int = 200
    sl_lr: float = 0.1
    sl_cardinality_weight: float = 0.05
    sl_prior_weight: float = 1.0
    sl_prior_epochs: int = 300
    # §8.5.1's GRAPH axis: run the winning program of ANOTHER run, unmodified, on
    # this dataset. That comparison is the headline of the amortization claim and
    # is one no per-instance method can even enter — SL-VAE has no artifact to
    # transfer. Points at another arm's results JSON; its `script` field is run as
    # a canned strategy here.
    sl_transfer_from: str | None = None
    # Cascade reconstruction. `cr_setting` is the load-bearing one
    # (research/cascade_reconstruction.md §8.3): the four settings of §2.7 are four
    # PROTOCOLS, not four knobs, and a decoder selected under `final_snapshot` is
    # solving a different problem from one selected under `partial_times`. The
    # masking DIRECTION is stated explicitly because §8.2 trap 1 records that two
    # papers in this literature use the symbol sigma for opposite quantities:
    # `cr_observation_rate` is the probability a node IS reported.
    cr_setting: str = partial_times
    cr_observation_rate: float = default_observation_rate
    cr_hidden_rate: float = default_hidden_rate
    cr_instances: int = 20
    cr_select_split: str = "train"
    cr_eval_split: str = "test"
    # lambda in `lambda * PathPrecision + (1 - lambda) * EventF1`. >= 0.5 is a
    # REQUIREMENT rather than a taste (§2.6): the node set is nearly free, so a
    # search rewarded mostly on Event F1 discovers the tree contributes nothing to
    # its score and converges on decoders that never attempt it.
    cr_tree_weight: float = 0.6
    # Arm A knobs; read only by method="decode"
    cr_mcmc_proposals: int = 400
    cr_mcmc_burn_in: float = 0.3
    # True when this arm is the @native condition. The evaluator has already been
    # resolved to monte_carlo with one episode by then, so the arm identity cannot
    # be recovered from `evaluator` — and it decides whether `predict_marginals`
    # exists at all (§2.4.3).
    native_arm: bool = False
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
    instances: list | None = None,
    training_sources: list | None = None,
) -> OuterLoopMethod:
    """
    The single construction site for every method.

    `strategy_mode` / `allow_mc_algorithms` are the resolved values, not
    config's: a canned script overrides both. Only the two methods with a
    refinement loop take a checkpoint — per_step and windowed have nothing to
    resume, and `gradient` is a single deterministic pass.
    """
    if config.method == "decode":
        # Arm A for cascade reconstruction. No agent, no population, no
        # refinement: Metropolis-Hastings over histories against the frozen
        # kernel, reporting DITTO's posterior-mean barycenter rather than the MAP
        # sample — the component swap §2.3 warns against, hence a CONTROL.
        return BarycenterDecoding(
            instances=instances or [],
            proposals=config.cr_mcmc_proposals,
            burn_in=config.cr_mcmc_burn_in,
            tree_weight=config.cr_tree_weight,
            seed=config.seed,
        )

    if config.method == "gradient":
        # Arm A. No agent, no population, no refinement: Adam on a relaxed source
        # vector against a frozen f_theta, which is SL-VAE's own procedure with our
        # likelihood plugged in — the control program search is measured against
        # (research/source_localization.md §2.6 arm A).
        if config.evaluator in (monte_carlo,):
            raise ValueError(
                f"the gradient method needs GRADIENTS through the forward model, "
                f"and the {config.evaluator!r} evaluator is a sampler with none. "
                f"Use gradient_free@world_model (or @oracle for the analytic IC "
                f"form); arms 3 and 4 are agent conditions, not this one."
            )

        return GradientInversion(
            wm_results_json=config.wm_results_json,
            instances=instances or [],
            budget_mode=config.sl_budget_mode,
            steps=config.sl_steps,
            lr=config.sl_lr,
            cardinality_weight=config.sl_cardinality_weight,
            prior_kind=config.sl_prior,
            prior_weight=config.sl_prior_weight,
            prior_epochs=config.sl_prior_epochs,
            training_sources=training_sources or [],
            device=config.device,
            seed=config.seed,
        )

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
        f"choose one_shot|per_step|windowed|evolve|adaptive|gradient|decode"
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


def _build_environment(
    config: ExperimentConfig,
    graph: GraphInfo,
    negative_seeds: tuple = (),
    competitive: bool = False,
    epidemic: bool = False,
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
        )

        # ...and a non-compartmental checkpoint cannot be rolled out against an
        # epidemic task: its head is monotone by construction, so `I` could never
        # shrink and every rollout would report a cascade that transmits forever
        # (research/epidemic_control.md §2.4)
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
    # For influence blocking this IS S_N, and `--outbreak-selector` is §8.3's
    # attacker model rather than an outbreak rule — the same machinery, because the
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
    # testable at all (research/source_localization.md §8.5.1): the outer loop's
    # reward is computed on `select`, and the winner is re-run unmodified on
    # `evaluate`. A program that memorized specific cascades scores well on the
    # first and badly on the second.
    select_instances, evaluate_instances, training_sources = [], [], []

    if registry.reconstructs:
        # The same two-pool split, and for the same reason: a decoder that
        # memorized specific cascades is indistinguishable from an algorithm until
        # it meets episodes the search never saw. `require_parents` RAISES on a
        # dataset generated without --trace-parents rather than silently collapsing
        # the reward onto Event F1, which is the failure §2.6 exists to prevent.
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
        # Arm A's prior is fit on the SELECTION split's source sets: withholding it
        # strawmans the control, and fitting it on the held-out split leaks the
        # answer. Both mistakes flip the sign of the headline claim.
        training_sources = [instance.sources for instance in select_instances]

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
        sense=registry.objective if registry.objective in (maximize, "minimize") else maximize,
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
            f"— MINIMIZING the rumour's final size"
        )
    elif outbreak and registry.epidemic:
        print(
            f"[run] outbreak: {len(outbreak)} index case(s) ({outbreak_pct:g}% of "
            f"N, selector={config.outbreak_selector}, seed={config.seed}) | "
            f"{config.diffusion_model} beta_scale={config.epi_beta} "
            f"gamma={config.epi_gamma}"
            + (f" alpha={config.epi_alpha}" if config.diffusion_model == "SEIR" else "")
            + f" | lever={config.epi_lever} (budget buys {budget_op}) "
            f"— MINIMIZING the attack rate"
        )
    elif outbreak:
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
        if task.decodes:
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

    # §8.5.1's graph axis: the winning program of another run, executed here
    # unmodified. Read before the canned-baseline branch so a transfer arm is a
    # transfer arm regardless of what else was set.
    if config.sl_transfer_from is not None:
        if canned_script is not None:
            raise ValueError(
                "--sl-transfer-from supplies the script to run and cannot be "
                "combined with another canned script"
            )

        source = json.loads(Path(config.sl_transfer_from).read_text())
        canned_script = source.get("script")

        if not canned_script:
            raise ValueError(
                f"{config.sl_transfer_from} has no `script` field to transfer; "
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
        # Classical-library baseline: same pipeline, envs, and metrics — no LLM
        if config.baseline not in (
            algorithm_names
            + list(adaptive_algorithms)
            + blocking_names
            + dismantling_names
            + immunization_names
            + localization_names
            + reconstruction_names
        ):
            raise ValueError(
                f"unknown baseline {config.baseline!r}; choose a static algorithm "
                f"from {algorithm_names}, an adaptive policy from "
                f"{sorted(adaptive_algorithms)}, a blocker from {blocking_names}, a "
                f"dismantler from {dismantling_names}, an immunizer from "
                f"{immunization_names}, a source localizer from "
                f"{localization_names}, or a trajectory decoder from "
                f"{reconstruction_names}"
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
        elif config.baseline in reconstruction_algorithms:
            # A published decoder is a whole-TRAJECTORY inference: it is handed the
            # masked observation and returns `{node: (time, parent)}`. `predict` is
            # the metered kernel, which the kernel-using members read and the
            # structural ones ignore through **kw.
            canned_script = f"""\
class ReconstructionBaseline(Strategy):
    def reconstruct(self, graph, observation, horizon):
        return reconstruction_algorithms.{config.baseline}(
            graph,
            observation,
            horizon,
            diffusion_model="{config.diffusion_model}",
            predict=getattr(self, "step_marginals", None),
        )
"""
        elif config.baseline in localization_algorithms:
            # A published localizer is a source-set INFERENCE, not a plan: it is
            # handed the observation and returns the nodes it believes started the
            # cascade. Its paired scorer rides along so the arm gets a real AUC
            # rather than the rank-derived stand-in a set-only method falls back to.
            canned_script = f"""\
class LocalizationBaseline(Strategy):
    def localize(self, graph, observation, budget):
        return [
            int(node)
            for node in localization_algorithms.{config.baseline}(
                graph,
                observation,
                budget,
                diffusion_model="{config.diffusion_model}",
                horizon={config.horizon},
            )
        ]

    def source_scores(self, graph, observation):
        return localization_scorers.{config.baseline}(
            graph, observation, diffusion_model="{config.diffusion_model}"
        )
"""
        elif config.baseline in all_blocking_algorithms:
            # A published blocker is handed the RUMOUR's own seeds and returns the
            # intervention this lever buys — node ids on three levers, `(u, v)` arcs
            # on the fourth. `blocking.blocking_plan` reconciles the two shapes into
            # one plan, which is what keeps a library algorithm runnable without
            # rewriting it to know what a plan is.
            if not emittable(config.baseline, config.blocking_lever):
                raise ValueError(
                    f"baseline {config.baseline!r} returns "
                    f"{blocking_shape(config.baseline)}s and this arm's lever "
                    f"({config.blocking_lever!r}) spends its budget on "
                    f"{lever_shape[config.blocking_lever]}s, so its output is not "
                    f"something this arm may emit. It is a "
                    f"{blocking_levers[config.baseline]} method — run it with "
                    f"--blocking-lever {blocking_levers[config.baseline]}, or pick a "
                    f"{config.blocking_lever} member."
                )

            canned_script = f"""\
class BlockingBaseline(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        picks = blocking_algorithms.{config.baseline}(
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
        elif config.baseline in immunization_algorithms:
            # A published immunizer is a static DOSE allocation, committed at t=0.
            # It is handed the outbreak because the data-aware members (dava,
            # frontier_immunization) condition on it; the structural ones take **kw
            # and ignore it. `immunization_plan` then reconciles the two output
            # shapes — node ids on the node levers, `(u, v)` arcs on the edge ones —
            # and drops any index case the algorithm picked anyway.
            if not immunization_emittable(config.baseline, config.epi_lever):
                raise ValueError(
                    f"baseline {config.baseline!r} returns "
                    f"{immunization_shape[config.baseline]}s and this arm's lever "
                    f"({config.epi_lever!r}) spends its budget on "
                    f"{epidemic_lever_shape[config.epi_lever]}s, so its output is not "
                    f"something this arm may emit. It is a "
                    f"{immunization_levers[config.baseline]} method — run it with "
                    f"--epi-lever {immunization_levers[config.baseline]}, or pick a "
                    f"{config.epi_lever} member."
                )

            canned_script = f"""\
class ImmunizationBaseline(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        picks = immunization_algorithms.{config.baseline}(
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
        instances=select_instances,
        training_sources=training_sources,
    )

    # Optimize the method with the outer-loop coding agent iteration loop to find the best strategy and trajectory result
    print(f"[run] optimizing with {config.method} (provider {provider_label})...")
    strategy, trajectory = method.optimize(agent, environment, task, graph)

    if task.decodes:
        print(
            f"[run] winner: score={trajectory.reward:.4f} on the "
            f"{config.cr_select_split} split (selection score, higher is better)"
        )
    elif task.recovers:
        print(
            f"[run] winner: F1={trajectory.reward:.4f} on the "
            f"{config.sl_select_split} split (selection score, higher is better)"
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
    # meets episodes the search never saw (research/source_localization.md §8.5.1).
    heldout = None
    if task.decodes and evaluate_instances:
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
            f"[run] held-out score={heldout.reward:.4f} "
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
            f"[run] held-out F1={heldout.reward:.4f} "
            f"(selection {trajectory.reward:.4f}, "
            f"generalization gap {heldout.reward - trajectory.reward:+.4f})"
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

    if task.decodes:
        # Same reasoning as the localization block below: the score is measured
        # against a history we stored, so it carries no evaluator noise and is
        # already comparable across conditions. `mc_reward` is filled from the
        # held-out score so every reader that asks for "the number comparable
        # across arms" gets the right one unchanged.
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
        result["mcmc_proposals_per_instance"] = trajectory.cost.get(
            "mcmc_proposals_per_instance"
        )
        result["mcmc_acceptance_rate"] = trajectory.cost.get("mcmc_acceptance_rate")
        result["generalization_gap"] = (
            round(reported.reward - trajectory.reward, 6)
            if heldout is not None
            else None
        )
        result["mc_reward"] = reported.reward
        result["reward"] = reported.reward
        result["mc_reward_se"] = reported.cost.get("reward_se")
        result["spread_pct"] = None
        result["per_instance"] = reported.cost.get("per_instance")
        result["summary"] = summarize(reported, graph, task)
        # §2.11 risk 1 makes this a REQUIRED check rather than a diagnostic: a
        # trivial decoder (everyone reachable, parents by BFS) must score badly
        # under the chosen reward, or the reward is wrong. Reported into the same
        # file as the result, because a reader cannot interpret a program-search
        # number without it.
        result |= trivial_decoder_reward(
            evaluate_instances or select_instances, graph, config.cr_tree_weight
        )

    if task.recovers and not task.decodes:
        # The F1 an inverse task reports needs NO ground-truth referee to be
        # comparable across conditions: it is measured against a source set we
        # know, so it carries no evaluator noise at all. That is unusual for this
        # pipeline and is why `mc_reward` is filled from the held-out score rather
        # than from a Monte-Carlo replay — every reader that asks for "the number
        # comparable across arms" then gets the right one unchanged.
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
        result["gradient_steps_per_instance"] = trajectory.cost.get(
            "gradient_steps_per_instance"
        )
        result["sl_prior"] = trajectory.cost.get("sl_prior")
        result["transfer_from"] = config.sl_transfer_from
        # The generalization gap §8.5.1 exists to expose: large and positive-side
        # means the program memorized the episodes it was selected on
        result["generalization_gap"] = (
            round(reported.reward - trajectory.reward, 6)
            if heldout is not None
            else None
        )
        result["mc_reward"] = reported.reward
        result["reward"] = reported.reward
        result["mc_reward_se"] = reported.cost.get("reward_se")
        result["spread_pct"] = None
        result["per_instance"] = reported.cost.get("per_instance")
        result["summary"] = summarize(reported, graph, task)

    if task.blocks:
        # §8.1's block. The reward is the rumour's remaining size (lower is better);
        # PREVENTED influence is that subtracted from the unopposed reference, which
        # is the quantity all five published names refer to. The reference is
        # measured on THIS arm's evaluator, on purpose: a ratio of two different
        # rulers means nothing, and --compare re-measures both on the shared referee.
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
            f"{unopposed:.2f} — prevented "
            f"{result['prevented_influence']:+.2f} "
            f"({result['prevented_pct_of_unopposed']:.1f}% of the cascade, "
            f"lever={result['lever']}, |S_P|/|S_N|={result['budget_ratio']})"
        )

    if task.immunizes:
        # §8.3's block. The reward is the attack rate (lower is better); PREVENTED
        # INFECTIONS is that subtracted from the unprotected reference, which is
        # what an immunization table reports. The reference is measured on THIS
        # arm's evaluator, on purpose: a difference of two different rulers is not a
        # quantity, and --compare re-measures both on the shared referee.
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
        )
        spectral = result.get("spectral") or {}
        print(
            f"[run] epidemic: attack {trajectory.reward:.2f} vs unprotected "
            f"{unprotected:.2f} — prevented "
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
                f"{spectral['eigendrop_pct']:.1f}%) — CONTEXT, not the score"
            )

    if task.contains and not task.blocks and not task.immunizes:
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
    if config.compare and task.decodes:
        # A decoder's score is already ground truth (it is measured against a
        # history we stored), so the referee measures the other thing: re-simulate
        # each decode's RECOVERED SOURCES on NDlib and compare against what the
        # cascade actually did. Reported beside the TRUE source set's own error,
        # because on an ill-posed problem a recovered set can reproduce the
        # observation better than the truth did.
        referee_runs = config.referee_mc_runs or config.mc_runs
        result["referee_mc_runs"] = referee_runs
        print(f"[run] re-simulation referee ({referee_runs} NDlib runs per set)...")

        referee = MonteCarloEnvironment(
            graph,
            config.diffusion_model,
            mc_runs=referee_runs,
            base_seed=config.seed,
            remove_semantics=config.remove_semantics,
        )
        reported = heldout if heldout is not None else trajectory
        result |= referee_reconstruction_error(
            referee,
            task,
            evaluate_instances or list(task.instances),
            reported.cost.get("per_instance", []),
        )
        print(
            f"[run] resim_error={result.get('resim_error', float('nan')):.5f} "
            f"(true sources score "
            f"{result.get('resim_error_true_sources', float('nan')):.5f})"
        )
    elif config.compare and task.recovers:
        # An inverse task's F1 is already ground truth, so the referee measures the
        # OTHER thing §8.5.5 asks for: re-simulate the recovered sources on NDlib
        # and compare against what was observed. Reported beside the TRUE source
        # set's own error, because on an ill-posed problem a recovered set can
        # reproduce y better than the truth did, and the number is unreadable
        # without knowing that. §11: no surveyed paper reports this at all, so the
        # column is self-contained and is not a cross-paper comparison.
        referee_runs = config.referee_mc_runs or config.mc_runs
        result["referee_mc_runs"] = referee_runs
        print(f"[run] re-simulation referee ({referee_runs} NDlib runs per set)...")

        referee = MonteCarloEnvironment(
            graph,
            config.diffusion_model,
            mc_runs=referee_runs,
            base_seed=config.seed,
            remove_semantics=config.remove_semantics,
        )
        reported = heldout if heldout is not None else trajectory
        result |= referee_resimulation_error(
            referee,
            task,
            evaluate_instances or list(task.instances),
            reported.cost.get("per_instance", []),
        )
        print(
            f"[run] resim_error={result.get('resim_error', float('nan')):.5f} "
            f"(true sources score {result.get('resim_error_true_sources', float('nan')):.5f})"
        )
    elif config.compare:
        referee_runs = config.referee_mc_runs or config.mc_runs
        result["referee_mc_runs"] = referee_runs
        print(f"[run] MC compare replay ({referee_runs} runs)...")

        mc_environment = MonteCarloEnvironment(
            graph,
            config.diffusion_model,
            mc_runs=referee_runs,
            base_seed=config.seed,
            remove_semantics=config.remove_semantics,
            # The referee has to run the SAME dynamics the arm was scored on, or the
            # shared ground-truth column would compare a two-cascade result against a
            # one-cascade replay and read as a huge fidelity error
            negative_seeds=outbreak,
            competitive_config=(
                _competitive_config(config) if task.blocks else None
            ),
            # ...and the same rule for the compartmental case: refereeing an SIR
            # arm on an IC replay would compare a process with recovery against one
            # without and read as an enormous fidelity error
            epidemic_config=(
                _epidemic_config(config) if task.immunizes else None
            ),
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

        if task.immunizes:
            # The prevented-infections column every immunization table reports, on
            # the SHARED referee. Both terms are re-measured here rather than
            # reusing the arm's own reference, because prevented infections is a
            # DIFFERENCE and a difference of two evaluators' numbers is not a
            # quantity. The curve comes back too, so peak and time-to-peak are
            # ground-truth rather than model-predicted.
            mc_unprotected, mc_unprotected_curve = unprotected_reference(
                mc_environment, task, graph, config.horizon, config.budget
            )
            result["mc_unprotected_prevalence_curve"] = mc_unprotected_curve
            result["mc_unprotected_attack_rate"] = mc_unprotected
            result["mc_prevented_infections"] = mc_unprotected - mc_trajectory.reward
            result["mc_prevented_pct_of_unprotected"] = (
                100.0 * (mc_unprotected - mc_trajectory.reward) / mc_unprotected
                if mc_unprotected
                else 0.0
            )
            result["mc_prevalence_curve"] = mc_trajectory.prevalence_curve
            result["mc_curve"] = epidemic_curve_metrics(
                mc_trajectory.prevalence_curve or [], graph.num_nodes, config.epi_burn_in
            )
            print(
                f"[run] ground-truth prevented infections: "
                f"{result['mc_prevented_infections']:+.2f} of {mc_unprotected:.2f} "
                f"({result['mc_prevented_pct_of_unprotected']:.1f}%), peak "
                f"{result['mc_curve']['peak_prevalence']:.1f} at t="
                f"{result['mc_curve']['time_to_peak']}"
            )

        if task.blocks:
            # The prevented-influence column every blocking table reports, on the
            # SHARED referee. Both terms are re-measured here rather than reusing the
            # arm's own reference, because prevented influence is a DIFFERENCE and a
            # difference of two evaluators' numbers is not a quantity.
            mc_unopposed = unopposed_reference(
                mc_environment, task, config.horizon, config.budget
            )
            result["mc_unopposed_spread"] = mc_unopposed
            result["mc_prevented_influence"] = mc_unopposed - mc_trajectory.reward
            result["mc_prevented_pct_of_unopposed"] = (
                100.0 * (mc_unopposed - mc_trajectory.reward) / mc_unopposed
                if mc_unopposed
                else 0.0
            )
            print(
                f"[run] ground-truth prevented influence: "
                f"{result['mc_prevented_influence']:+.2f} of {mc_unopposed:.2f} "
                f"({result['mc_prevented_pct_of_unopposed']:.1f}%)"
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
        choices=(
            algorithm_names
            + list(adaptive_algorithms)
            + blocking_names
            + dismantling_names
            + localization_names
            + reconstruction_names
        ),
        metavar="NAME",
        help="evaluate this classical library algorithm instead of an LLM strategy: "
        "a static IM algorithm, a per-round adaptive policy, an influence blocker, a "
        "network dismantler, a source localizer, or a trajectory decoder "
        "(default: None).",
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
            "gradient",
            "decode",
        ],
        help="outer-loop method; evolve = population edits with refine/restructure "
        "operators; adaptive = the same search over a per-round policy for adaptive "
        "IM; gradient = arm A for source localization, per-instance Adam on a "
        "relaxed source vector against a frozen world model, no LLM; decode = arm A "
        "for cascade reconstruction, Metropolis-Hastings over histories against the "
        "same frozen model, also no LLM (default: one_shot).",
    )
    parser.add_argument(
        "--native-arm",
        action="store_true",
        help="mark this run as the @native condition: `predict_marginals` is "
        "removed, so a source-localization program must be a pure structural "
        "heuristic. Pair with --evaluator monte_carlo --mc-runs 1 (default: False).",
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
        help="influence blocking: Budak's r — the rumour is detected r steps late "
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
        help="epidemic control: rate of LEAVING I — recovery under SIR/SEIR, "
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
        "--sl-prior",
        type=str,
        default=vae_prior,
        choices=list(valid_priors),
        help="arm A only: none reproduces SL-VAE (a) (frozen forward model + "
        f"descent, no prior); vae reproduces the full method (default: {vae_prior}).",
    )
    parser.add_argument(
        "--sl-steps",
        type=int,
        default=200,
        help="arm A only: Adam steps per instance (default: 200).",
    )
    parser.add_argument(
        "--sl-lr",
        type=float,
        default=0.1,
        help="arm A only: Adam learning rate on the source logits (default: 0.1).",
    )
    parser.add_argument(
        "--sl-cardinality-weight",
        type=float,
        default=0.05,
        help="arm A only: weight on (sum(x~) - k)^2, the given-k constraint "
        "(default: 0.05).",
    )
    parser.add_argument(
        "--sl-prior-weight",
        type=float,
        default=1.0,
        help="arm A only: weight on -log p(x~) (default: 1.0).",
    )
    parser.add_argument(
        "--sl-prior-epochs",
        type=int,
        default=300,
        help="arm A only: epochs fitting the source VAE (default: 300).",
    )
    parser.add_argument(
        "--sl-transfer-from",
        type=str,
        default=None,
        help="source localization: run the winning program named by ANOTHER arm's "
        "results JSON, unmodified, on this dataset. The graph axis of the "
        "amortization claim, and a comparison no per-instance method can enter "
        "(default: None).",
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
        "Stated in that direction on purpose — this literature uses the symbol "
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
        "--cr-mcmc-proposals",
        type=int,
        default=400,
        help="arm A only: Metropolis-Hastings proposals per cascade. This is the "
        "first number to raise before quoting arm A as DITTO-parity (default: 400).",
    )
    parser.add_argument(
        "--cr-mcmc-burn-in",
        type=float,
        default=0.3,
        help="arm A only: fraction of the chain discarded before the barycenter "
        "starts accumulating (default: 0.3).",
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
        sl_prior=args.sl_prior,
        sl_steps=args.sl_steps,
        sl_lr=args.sl_lr,
        sl_cardinality_weight=args.sl_cardinality_weight,
        sl_prior_weight=args.sl_prior_weight,
        sl_prior_epochs=args.sl_prior_epochs,
        sl_transfer_from=args.sl_transfer_from,
        cr_setting=args.cr_setting,
        cr_observation_rate=args.cr_observation_rate,
        cr_hidden_rate=args.cr_hidden_rate,
        cr_instances=args.cr_instances,
        cr_select_split=args.cr_select_split,
        cr_eval_split=args.cr_eval_split,
        cr_tree_weight=args.cr_tree_weight,
        cr_mcmc_proposals=args.cr_mcmc_proposals,
        cr_mcmc_burn_in=args.cr_mcmc_burn_in,
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
