"""
One command, whole experiment: generate data -> train the world model -> run every
coding-agent arm at every budget -> plot -> write
results/<task>/<dataset>/<run>/report.md.

Every stage writes its artifacts before the next one starts, so a crashed or killed
run resumes exactly where it stopped (finished work is detected on disk and skipped).

All six baseline conditions in one sweep (the default arm set). Every arm carries
its own evaluator, so conditions 3-6 differ in exactly one thing: the inner-loop
feedback, and land in one report table:

python -m pipeline.run --dataset ba --num-graphs 40 --syn-nodes 100 \
    --budget-pcts 1 5 10 20 --compare \
    --llm-model gpt-5.6-sol --outer-iters 5

Just the classical pool and our method, at one budget:

python -m pipeline.run --dataset netscience \
    --baselines celf_pp degree_discount imm \
    --arms one_shot_free@world_model --budget-pcts 5 --compare

No world model anywhere (the train stage is then skipped automatically):

python -m pipeline.run --dataset sbm --num-graphs 40 \
    --arms routing one_shot_free@native one_shot_free@oracle --compare

Resume just the reporting half of a finished run:

python -m pipeline.run --dataset ba --start-stage plots

Re-run one stage from scratch:

python -m pipeline.run --dataset ba --start-stage agent --end-stage agent --force

Reuse a world model trained by an earlier run instead of training a new one:

python -m pipeline.run --dataset ba --run new_agent_sweep \
    --wm-results-json results/influence_maximization/ba/default/world_model/sage_IC.json

Hold two variants of the same (task, dataset) side by side:

python -m pipeline.run --dataset jazz --run gcnii_ablation --wm-model gcnii --n-layers 8
"""

import argparse
import json
import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from dotenv import load_dotenv
from tqdm import tqdm

from baselines.discovery import build_context, program_script
from baselines.discovery import discovery as discovery_kind
from baselines.registry import (
    available_baselines,
    baselines_for_task,
    external_baselines,
    runnable_baselines,
)
from baselines.run_baseline import (
    BaselineError,
    edge_script,
    localize_script,
    predict_script,
    reconstruct_script,
    round_seed_script,
    run_external_baseline,
    seed_script,
)
from coding_agent import executor
from coding_agent.run import ExperimentConfig, run_experiment
from data.generate_wm_data import (
    GenConfig,
    default_budget_pct_range,
    run_generation,
    synthetic_families,
)
from data.wm_actions import blocking_selectors, spine_algorithms
from data.wm_competitive import (
    auto_dominance,
    shared_positive_prob,
    tie_break_choices,
)
from coding_agent.tools.adaptive_algorithms import adaptive_algorithm_names
from coding_agent.tools.blocking_algorithms import (
    blocking_algorithm_names,
    default_blocking_baselines,
)
from coding_agent.tools.dismantling_algorithms import dismantling_algorithm_names
from coding_agent.tools.immunization_algorithms import (
    default_immunization_baselines,
    immunization_algorithm_names,
)
from coding_agent.tools.localization_algorithms import localization_algorithm_names
from coding_agent.tools.reconstruction_algorithms import (
    reconstruction_algorithm_names,
)
from coding_agent.tools.prediction_algorithms import prediction_algorithm_names
from coding_agent.tools.library_api import algorithm_names
from coding_agent.blocking import counter_seed, resolve_lever, valid_levers
from coding_agent.containment import outbreak_selectors, select_outbreak
from coding_agent.epidemic import (
    default_contact_reduction,
    resolve_lever as resolve_epidemic_lever,
    valid_levers as valid_epidemic_levers,
    vaccinate,
)
from coding_agent.localization import (
    episode_budget,
    load_instances,
    valid_budget_modes,
    valid_observations,
)
from coding_agent.prediction import (
    default_forecast_samples,
    load_forecasts,
)
from coding_agent.reconstruction import (
    default_hidden_rate,
    default_observation_rate,
    load_cascades,
    partial_times,
    valid_settings,
)
from coding_agent.rounds import round_batches
from coding_agent.types import GraphInfo, full_adoption, valid_feedback_models
from data.wm_cascades import (
    chronological_split,
    corpus_defaults,
    resolve_step,
    increment_target,
    valid_graphs,
    valid_splits,
    valid_targets,
)
from data.generate_wm_data import (
    episode_random_split,
    graph_disjoint_split,
    min_graphs_for_disjoint,
    valid_split_modes,
)
from data.wm_graphs import cascade_corpora, kronecker_seeds
from data.wm_epidemic import default_burn_in
from data.wm_simulator import (
    epidemic_dynamics,
    valid_action_ops,
    valid_remove_semantics,
)
from pipeline.conditions import (
    expand_llm_models,
    Arm,
    condition_names,
    ground_truth_reward,
    default_arms,
    default_baselines,
    native,
    needs_world_model,
    parse_arm,
    resolve_evaluator,
    valid_evaluators,
)
from pipeline.layout import Layout, budget_label, skip_marker_suffix
from pipeline.tasks import default_run, get_task, require_runnable, task_names
from pipeline.plots import build_plots
from pipeline.report import write_report
from pipeline.summary import write_environment, write_summary
from world_model.train_wm import TrainConfig, train_world_model
from world_model.wm_data import (
    basic_encoding,
    load_graph_store,
    valid_action_encodings,
)
from world_model.wm_metrics import (
    default_prediction_metric,
    valid_prediction_metrics,
)
from world_model.wm_model import backbones

stages = ("data", "train", "agent", "plots", "report")
native_mc_runs_default = 1
# The sample-count ceiling for adaptive arms: their per-round act() runs once per
# ensemble member, so evaluation cost scales linearly in this where every other
# arm's is one batched forward pass
adaptive_n_samples = 50


@dataclass
class PipelineConfig:
    dataset: str
    task: str = "influence_maximization"
    run: str = default_run
    # data stage
    num_graphs: int = 1
    syn_nodes: int = 100
    er_p: float = 0.05
    ba_m: int = 3
    ws_k: int = 6
    ws_p: float = 0.1
    sbm_blocks: int = 4
    sbm_p_in: float = 0.15
    sbm_p_out: float = 0.01
    plc_m: int = 3
    plc_p: float = 0.05
    kron_variant: str = "core_periphery"
    gen_models: tuple = ("IC", "LT")
    gen_algorithms: tuple = ("random", "degree", "pagerank", "betweenness")
    # None = the task registry's own op set, so `--task critical_node_detection`
    # generates remove_node transitions without a flag. Paired with `allowed_ops`
    # below, which resolves the same way: the ops the data teaches and the ops the
    # planner may emit come from one registry entry and cannot disagree.
    gen_action_ops: tuple | None = None
    # None = on for the one task whose reward needs a transmission edge (cascade
    # reconstruction), off otherwise. Explicit True lets a dataset generated under
    # another task be shared with that one, which the registry alone cannot say.
    trace_parents: bool | None = None
    # Crosses all three stages: it decides how the data is generated, how the
    # head's T_exo is built, and what the agent is told remove_node does. One
    # value so they cannot disagree; defaulted from the task registry.
    remove_semantics: str | None = None
    # How the generator assigns train/val/test. graph_disjoint is correct and a
    # SINGLE-graph real dataset cannot satisfy it at all, so None resolves per
    # dataset in stage_data: graph_disjoint for a synthetic family with enough
    # graphs, episode_random (in-graph metrics, stated loudly) otherwise. An
    # explicit value always wins and a bad explicit value still raises in the
    # generator. See data/generate_wm_data.py.
    split_mode: str | None = None
    prob_model: str = "weighted"
    uniform_p: float = 0.1
    budget_pct_range: tuple = default_budget_pct_range
    rollouts: int = 20
    gen_horizon: int = 10
    inject_p: float = 0.4
    cf_prob: float = 0.4
    cf_branches: int = 2
    mc_marginals: int = 30
    split: tuple = (0.7, 0.15, 0.15)
    # train stage
    wm_model: str = "sage"
    head: str = "structured"
    hidden_dim: int = 64
    n_layers: int = 3
    n_heads: int = 4
    ffn_dim: int = 128
    gcnii_alpha: float = 0.1
    gcnii_lamda: float = 0.5
    dropout: float = 0.1
    epochs: int = 400
    lr: float = 1e-3
    weight_decay: float = 5e-4
    batch_size: int = 32
    pos_weight: str = "off"
    patience: int = 50
    plan_demo: bool = True
    plan_graphs: int = 5
    # k-seed FULL-HORIZON planning regret vs greedy-MC — the IM problem as posed,
    # unlike the single-step plan_demo. Costs real simulator episodes, so it is
    # opt-in: 0 disables.
    plan_budget_k: int = 0
    plan_budget_graphs: int = 3
    plan_budget_horizon: int = 20
    # Off-policy rollout fidelity (world_model/wm_policies.py). The recorded action
    # sequence is on-policy by construction, so it says nothing about the action
    # distribution an agent actually proposes.
    ood_policies: tuple = ()
    # `typed` splits the single act_edge channel by op
    action_encoding: str = basic_encoding
    # Feed the world model ones instead of the true p(u->v): the online/bandit
    # information state (research/adaptive_online_im.md §2.4b, §9.3 item 7)
    hide_edge_weights: bool = False
    wm_results_json: str | None = None
    # agent stage
    # None = the task registry's pool, falling back to the static IM classics
    baselines: tuple | None = None
    # None = the task registry's arms, falling back to the six-condition ladder
    arms: tuple | None = None
    # None = the task registry's own sweep, falling back to the 1/5/10/20 ladder.
    # Resolved through `resolve_budget_pcts` rather than defaulted here, so a task
    # whose k is a property of the instance can say so once.
    budget_pcts: tuple | None = None
    budgets: tuple | None = None
    evaluator: str = "oracle"
    native_mc_runs: int = native_mc_runs_default
    llm_model: str = "gpt-5.6-sol"
    # Fan every LLM-driven arm out across these models, one result row each;
    # None = the single llm_model above
    llm_models: tuple | None = None
    temperature: float | None = None
    diffusion_model: str = "IC"
    horizon: int = 10
    outer_iters: int = 20
    windows: int = 3
    mc_runs: int = 200
    n_samples: int = 200
    allowed_ops: tuple | None = None
    # Adaptive IM; read only by `adaptive` arms, so one sweep can hold adaptive
    # and non-adaptive arms side by side and divide one by the other
    rounds: int = 4
    per_round_budget: int | None = None
    round_gap: int = 1
    feedback_model: str = full_adoption
    # Dynamic/streaming IM and multi-round IM; both apply to every arm
    edit_rate: float = 0.0
    campaigns: int = 1
    # Critical node detection: the exogenous outbreak every arm contains. None =
    # the task registry's value (0 for every seeding task). Deterministic in
    # --seed, so all arms in one sweep face the same one.
    outbreak_pct: float | None = None
    outbreak_selector: str = "random"
    # Influence blocking. `outbreak_pct` / `outbreak_selector` above double as |S_N|
    # and the attacker model, so only the two-cascade specifics live here. All are
    # inert unless the task registry marks the task competitive.
    blocking_lever: str = counter_seed
    tie_break: str = auto_dominance
    positive_prob: str = shared_positive_prob
    detection_delay: int = 0
    negative_selectors: tuple = ("random", "degree", "pagerank")
    blocker_selectors: tuple = blocking_selectors
    # Epidemic control. `outbreak_pct` / `outbreak_selector` above double as the
    # index-case count and the OUTBREAK MODEL, so only the compartmental specifics
    # live here. All are inert unless the task registry marks the task epidemic.
    epi_lever: str = vaccinate
    contact_reduction: float = default_contact_reduction
    epi_beta: float = 1.0
    epi_gamma: float = 0.3
    epi_alpha: float = 0.5
    epi_burn_in: float = default_burn_in
    outbreak_selectors: tuple = ("random", "degree", "pagerank")
    immunizer_selectors: tuple = blocking_selectors
    # Source localization; every field is inert unless the task inverts, so one
    # sweep configuration serves all four runnable tasks
    sl_select_split: str = "train"
    sl_eval_split: str = "test"
    sl_instances: int = 20
    sl_observation: str = "marginal"
    sl_budget_mode: str = episode_budget
    sl_source_tolerance: float = 0.5
    sl_transfer_from: str | None = None
    # Cascade reconstruction; every field is inert unless the task decodes, so one
    # sweep configuration serves all six runnable tasks
    cr_setting: str = partial_times
    cr_observation_rate: float = default_observation_rate
    cr_hidden_rate: float = default_hidden_rate
    cr_instances: int = 20
    cr_select_split: str = "train"
    cr_eval_split: str = "test"
    cr_tree_weight: float = 0.6
    # Cascade prediction; every field is inert unless the task forecasts, so one
    # sweep configuration serves all seven runnable tasks. The two that decide
    # whether a number means anything are `cp_observation` (§8.2 pairs TWO windows
    # per corpus and a single-window result is not publishable in this literature)
    # and `cp_split` (§8.3: the field's own random-over-cascades split LEAKS, and
    # fixing it moved a decade of published numbers).
    cp_observation: int = 0
    cp_horizon: int = 0
    cp_step: int = 0
    cp_min_size: int = 10
    cp_truncate: int = 100
    cp_split: str = chronological_split
    cp_target: str = increment_target
    cp_graph: str = "native"
    cp_max_cascades: int = 0
    cp_max_nodes: int = 0
    cp_select_split: str = "train"
    cp_eval_split: str = "test"
    cp_instances: int = 40
    cp_observation_steps: int = 0
    cp_metric: str = default_prediction_metric
    cp_forecast_samples: int = default_forecast_samples
    allow_mc_algorithms: bool = False
    strategy_timeout: float = executor.strategy_timeout_seconds
    # USD per 1M tokens for the cost line; None -> tokens counted, cost null
    llm_price_in: float | None = None
    llm_price_out: float | None = None
    compare: bool = False
    credit: bool = True
    graph_id: str | None = None
    # driver
    seed: int = 42
    device: str = "cpu"
    start_stage: str = stages[0]
    end_stage: str = stages[-1]
    skip_stages: tuple = ()
    force: bool = False
    baseline_timeout: int = 3600
    results_root: str = "results"
    stage_status: dict = field(default_factory=dict)
    _graph: object = field(default=None, repr=False)


def expand_baselines(
    names: tuple, task: str, lever: str = counter_seed, epi_lever: str = vaccinate
) -> list[str]:
    """
    Resolve --baselines into arm specs.

    Accepts library algorithm names (condition 1), `external:<name>` for a
    published repo (condition 7), `discovery:<name>` for a published LLM
    algorithm-discovery system (condition 9), and the aliases `all`,
    `all-classical`, `all-external`, `all-discovery`. `all` deliberately expands
    external baselines to only those actually installed, so a fresh checkout does
    not fail on missing repos, and it never includes the discovery systems: each
    of those is a full LLM search at its published defaults, hours rather than
    seconds, so they are opted into by name or through `all-discovery`.

    `all-classical` expands to the TASK's own pool, not the static IM classics: a
    dismantler and a source localizer are different KINDS of algorithm, and
    expanding the IM list under `--task source_localization` would run seed-set
    selectors against an inverse task and fail at the contract check. On a
    competitive task it narrows further still, to the LEVER's own pool.
    """
    classical = (
        default_blocking_baselines[lever]
        if get_task(task).competitive
        # Same rule on the epidemic side and for the same reason: a lever can only
        # emit what its own members return, so `netmelt`'s arcs would be rejected
        # by a vaccination arm and `netshield`'s nodes by an edge one
        else default_immunization_baselines[epi_lever]
        if get_task(task).epidemic
        else get_task(task).default_baselines or default_baselines
    )
    specs = []

    for name in names:
        if name == "all-classical":
            specs += [f"baseline:{algorithm}" for algorithm in classical]
        elif name == "all-external":
            specs += [
                f"external:{baseline}"
                for baseline in available_baselines(task=task)
                if external_baselines[baseline].kind != discovery_kind
            ]
        elif name == "all":
            specs += [f"baseline:{algorithm}" for algorithm in classical]
            installed = [
                baseline
                for baseline in runnable_baselines(task=task)
                if external_baselines[baseline].kind != discovery_kind
            ]
            specs += [f"external:{baseline}" for baseline in installed]

            skipped = sorted(
                baseline
                for baseline in available_baselines(task=task)
                if external_baselines[baseline].kind != discovery_kind
                and baseline not in installed
            )
            if skipped:
                print(
                    f"[pipeline] --baselines all: skipping not-installed external "
                    f"baselines {skipped} (run: python -m baselines.setup_baselines --all)"
                )
        elif name == "all-discovery":
            installed = [
                baseline
                for baseline in runnable_baselines(task=task)
                if external_baselines[baseline].kind == discovery_kind
            ]
            specs += [f"discovery:{baseline}" for baseline in installed]

            skipped = sorted(
                baseline
                for baseline in available_baselines(task=task)
                if external_baselines[baseline].kind == discovery_kind
                and baseline not in installed
            )
            if skipped:
                print(
                    f"[pipeline] --baselines all-discovery: skipping not-installed "
                    f"discovery systems {skipped} (run: python -m "
                    f"baselines.setup_baselines --only {' '.join(skipped)})"
                )
        elif name.startswith("discovery:"):
            baseline = name.split(":", 1)[1]
            spec = external_baselines.get(baseline)

            if spec is None:
                raise ValueError(
                    f"unknown discovery baseline {baseline!r}; registered: "
                    f"{sorted(n for n, s in external_baselines.items() if s.kind == discovery_kind)}"
                )

            if spec.kind != discovery_kind:
                raise ValueError(
                    f"{baseline!r} is a {spec.kind} baseline that returns a set, not "
                    f"a program: run it as external:{baseline}"
                )

            specs.append(name)
        elif name.startswith("external:"):
            # The `all` aliases already filter on the task field; an explicit
            # name skipped it entirely, so `--baselines external:moeim` under
            # --task adaptive_online_im would quietly drop a static IM method
            # into an adaptive table. The field exists to stop precisely that.
            baseline = name.split(":", 1)[1]
            spec = external_baselines.get(baseline)

            if spec is not None and spec.kind == discovery_kind:
                raise ValueError(
                    f"{baseline!r} is an algorithm-discovery system that returns a "
                    f"program, not a set: run it as discovery:{baseline} (condition 9)"
                )

            if spec is not None and not spec.serves(task):
                raise ValueError(
                    f"external baseline {baseline!r} solves {spec.task!r}, not "
                    f"{task!r}. Running it here would put a {spec.task} method in "
                    f"a {task} table. Registered for this task: "
                    f"{sorted(baselines_for_task(task))}"
                )

            specs.append(name)
        else:
            specs.append(f"baseline:{name}")

    return specs


def resolve_arms(config: PipelineConfig) -> tuple:
    """--arms, else the task's own arm set, else the six-condition ladder."""
    if config.arms is not None:
        return tuple(config.arms)

    return get_task(config.task).default_arms or default_arms


def resolve_baselines(config: PipelineConfig) -> tuple:
    """
    --baselines, else the task's own pool, else the static IM classics.

    A competitive task resolves per LEVER rather than per task, because a lever can
    only emit what its own members return: `proximity` hands back node ids and
    `kimura_link_blocking` hands back arcs, so one pool across all four levers would
    fail at the first baseline of any lever but the default.
    """
    if config.baselines is not None:
        return tuple(config.baselines)

    if get_task(config.task).competitive:
        return default_blocking_baselines[config.blocking_lever]

    if get_task(config.task).epidemic:
        return default_immunization_baselines[config.epi_lever]

    return get_task(config.task).default_baselines or default_baselines


def resolve_allowed_ops(config: PipelineConfig) -> tuple:
    """--allowed-ops, else the LEVER's ops, else the task's planner's own set."""
    if config.allowed_ops is not None:
        return tuple(config.allowed_ops)

    # An epidemic arm's op vocabulary is decided by which of the four interventions
    # its budget buys, not by the task; `coding_agent.run` resolves the same pair
    # again from `--epi-lever`, and this keeps the two agreeing
    if get_task(config.task).epidemic:
        return resolve_epidemic_lever(config.epi_lever)[1]

    return get_task(config.task).allowed_ops


def resolve_gen_action_ops(config: PipelineConfig) -> tuple:
    """--gen-action-ops, else the ops the task's data generator should inject."""
    if config.gen_action_ops is not None:
        return tuple(config.gen_action_ops)

    return get_task(config.task).gen_action_ops


def build_arms(config: PipelineConfig) -> list[Arm]:
    """Classical pool + external published baselines + the named conditions."""
    specs = expand_baselines(
        resolve_baselines(config),
        config.task,
        config.blocking_lever,
        config.epi_lever,
    )
    specs += list(resolve_arms(config))

    arms = [parse_arm(spec, default_evaluator=config.evaluator) for spec in specs]

    duplicates = {arm.name for arm in arms if [a.name for a in arms].count(arm.name) > 1}
    if duplicates:
        raise ValueError(
            f"arm names collide, so their results would overwrite each other: "
            f"{sorted(duplicates)}"
        )

    if config.llm_models:
        arms = expand_llm_models(arms, tuple(config.llm_models))

    return arms


def resolve_budget_pcts(config: PipelineConfig) -> tuple:
    """--budget-pcts, else the task's own sweep, else the 1/5/10/20 ladder."""
    if config.budget_pcts is not None:
        return tuple(config.budget_pcts)

    return get_task(config.task).default_budget_pcts or (1.0, 5.0, 10.0, 20.0)


def budget_points(config: PipelineConfig) -> list[tuple[str, float | None, int]]:
    """
    (label, budget_pct, budget) for each point of the sweep.

    Precedence: `--budgets`, then the task's own ABSOLUTE sweep, then `--budget-pcts`,
    then the task's percentage sweep, then the 1/5/10/20 ladder. The absolute rung
    exists for influence blocking specifically: percent-of-N budgets are used by
    nobody in that literature, while k in {10..50} is the shared convention of every
    comparable table (research/influence_blocking.md §8.2).
    """
    if config.budgets is not None:
        return [(budget_label(None, k), None, int(k)) for k in config.budgets]

    task_budgets = get_task(config.task).default_budgets
    if task_budgets is not None and config.budget_pcts is None:
        return [(budget_label(None, k), None, int(k)) for k in task_budgets]

    return [(budget_label(pct, 0), float(pct), 0) for pct in resolve_budget_pcts(config)]


def active_stages(config: PipelineConfig) -> list[str]:
    start, end = stages.index(config.start_stage), stages.index(config.end_stage)
    if start > end:
        raise ValueError(
            f"--start-stage {config.start_stage!r} comes after --end-stage {config.end_stage!r}"
        )

    selected = [stage for stage in stages[start : end + 1]]

    # Training is only needed when some arm actually evaluates against the world model
    if "train" in selected and not needs_world_model(build_arms(config)):
        print("[pipeline] no arm uses the world model: skipping the train stage")
        selected.remove("train")

    if "train" in selected and config.wm_results_json is not None:
        print(
            f"[pipeline] --wm-results-json {config.wm_results_json}: "
            f"skipping the train stage"
        )
        selected.remove("train")

    return [stage for stage in selected if stage not in config.skip_stages]


def needs_gpu(config: PipelineConfig) -> tuple[bool, str]:
    """
    Whether this exact invocation ever puts a tensor on a device.

    The submit path reads this to choose between queueing with a GPU and
    queueing CPU-only. Generation, plots, the report, every classical arm and
    every C++ external repo are pure CPU: only training f_theta, rolling it
    out, and the learned external baselines need one.
    """
    selected = active_stages(config)
    arms = build_arms(config)

    if "train" in selected:
        return True, "the train stage fits f_theta"

    if "agent" in selected:
        if needs_world_model(arms):
            return True, "a @world_model arm rolls f_theta out"

        learned = sorted(
            arm.external
            for arm in arms
            if arm.external and external_baselines[arm.external].kind == "learned"
        )
        if learned:
            return True, f"learned external baselines: {' '.join(learned)}"

    return False, f"no torch on a device in stages {selected}"


def resolve_cascade_protocol(config: PipelineConfig) -> None:
    """
    Fill `--cp-observation` / `--cp-horizon` / `--cp-step` / `--gen-horizon` for a
    replayed corpus, and refuse a dataset that carries no cascades.

    Defaults come from the CORPUS's own published windows rather than from a flag,
    for the reason research/cascade_prediction.md §8.2 gives: the standard settings
    are per corpus (Weibo 0.5 h / 1 h to 24 h; Twitter 1 d / 2 d to 32 d; APS 3 y /
    5 y to 20 y), and a run that silently used one corpus's window on another would
    produce a number comparable to nothing. `--gen-horizon` is derived last, because
    it is the STEP COUNT the replay needs and getting it wrong truncates the very
    quantity being predicted.
    """
    task = get_task(config.task)

    if not task.observational:
        return

    if config.dataset not in cascade_corpora:
        raise ValueError(
            f"--task {config.task} replays REAL logged cascades, and dataset "
            f"{config.dataset!r} carries none. Corpora: {sorted(cascade_corpora)}. "
            f"research/cascade_prediction.md §6.1 records that not one graph we load "
            f"for the other tasks carries a trace: that is the whole reason this "
            f"task needed new loaders, and §2.2 is why replaying a SIMULATOR here "
            f"would close exactly the loop the task exists to break."
        )

    observation, horizon = corpus_defaults(config.dataset)
    config.cp_observation = config.cp_observation or observation
    config.cp_horizon = config.cp_horizon or horizon
    config.cp_step = resolve_step(config.cp_observation, config.cp_step)

    needed = config.cp_horizon // config.cp_step
    if config.gen_horizon < needed:
        print(
            f"[pipeline] --gen-horizon {config.gen_horizon} is shorter than the "
            f"{needed} timesteps this protocol needs "
            f"({config.cp_horizon} / {config.cp_step}); raising it, or the replay "
            f"would stop before the quantity being predicted exists"
        )
        config.gen_horizon = needed

    # A replayed corpus holds ONE dynamics: the transitions are identical whatever
    # kernel label they carry (the log is the log), so writing both IC and LT would
    # double the file for no second experiment. The label decides which HEAD is fit
    # to them, which is the actual axis (§2.2).
    if len(config.gen_models) > 1:
        config.gen_models = (config.diffusion_model,)

    print(
        f"[pipeline] cascade protocol: t_o={config.cp_observation}, "
        f"t_p={config.cp_horizon}, step={config.cp_step} "
        f"-> {config.cp_observation // config.cp_step} observed of {needed} "
        f"timesteps, split={config.cp_split}"
    )


def resolve_split_mode(config: PipelineConfig) -> str:
    """
    graph_disjoint where the dataset can honour it, episode_random where it cannot.

    A synthetic family with >= 3 graphs gets the leakage-free split; a single real
    graph cannot be split disjointly by graph at all, so it keeps the per-episode
    draw and the choice is printed so no number gets quoted without its regime.
    """
    if config.split_mode is not None:
        return config.split_mode

    disjoint = (
        config.dataset in synthetic_families
        and config.num_graphs >= min_graphs_for_disjoint
    )
    resolved = graph_disjoint_split if disjoint else episode_random_split
    reason = (
        f"{config.num_graphs} graphs from the {config.dataset!r} family"
        if disjoint
        else (
            f"{config.dataset!r} yields {config.num_graphs} graph(s), below the "
            f"{min_graphs_for_disjoint} a graph-disjoint split needs"
        )
    )
    print(f"[data] split_mode resolved to {resolved}: {reason}")

    return resolved


def stage_data(config: PipelineConfig, layout: Layout) -> dict:
    if layout.data_metadata().exists() and not config.force:
        metadata = json.loads(layout.data_metadata().read_text())
        print(
            f"[data] reusing {layout.data_dir} "
            f"({metadata.get('n_episodes')} episodes already generated; "
            f"--force to regenerate)"
        )
        return metadata

    print(
        f"[data] generating: dataset={config.dataset} "
        f"graphs={config.num_graphs} models={list(config.gen_models)} "
        f"selectors={list(config.gen_algorithms)} rollouts={config.rollouts} "
        f"mc_marginals={config.mc_marginals}"
    )

    generation_config = GenConfig(
        dataset=config.dataset,
        num_graphs=config.num_graphs,
        syn_nodes=config.syn_nodes,
        er_p=config.er_p,
        ba_m=config.ba_m,
        ws_k=config.ws_k,
        ws_p=config.ws_p,
        sbm_blocks=config.sbm_blocks,
        sbm_p_in=config.sbm_p_in,
        sbm_p_out=config.sbm_p_out,
        plc_m=config.plc_m,
        plc_p=config.plc_p,
        kron_variant=config.kron_variant,
        models=list(config.gen_models),
        prob_model=config.prob_model,
        uniform_p=config.uniform_p,
        budget=5,
        budget_pct=None,
        budget_pct_range=tuple(config.budget_pct_range),
        algorithms=list(config.gen_algorithms),
        rollouts=config.rollouts,
        horizon=config.gen_horizon,
        inject_p=config.inject_p,
        action_ops=list(resolve_gen_action_ops(config)),
        remove_semantics=config.remove_semantics,
        split_mode=resolve_split_mode(config),
        weight_lo=0.0,
        weight_hi=1.0,
        cf_prob=config.cf_prob,
        cf_branches=config.cf_branches,
        split=tuple(config.split),
        seed=config.seed,
        mc_marginals=config.mc_marginals,
        out_dir=str(layout.data_dir),
        # Two-cascade generation, from the registry rather than a flag so a blocking
        # dataset cannot be generated single-cascade by forgetting one. The three
        # dynamics parameters land in metadata.json and train_wm reads them back.
        competitive=get_task(config.task).competitive,
        # From the registry unless --trace-parents overrides it: NDlib emits no
        # transmission edge, and cascade reconstruction's tree-weighted reward is
        # not computable without one, so a dataset generated by forgetting the
        # flag would silently collapse the reward onto its easy half
        # (research/cascade_reconstruction.md §2.6).
        trace_parents=(
            get_task(config.task).reconstructs
            if config.trace_parents is None
            else config.trace_parents
        ),
        tie_break=config.tie_break,
        positive_prob=config.positive_prob,
        negative_pct=(
            get_task(config.task).outbreak_pct
            if config.outbreak_pct is None
            else config.outbreak_pct
        ),
        negative_selectors=tuple(config.negative_selectors),
        blocker_selectors=tuple(config.blocker_selectors),
        # Compartmental generation. The three rates are what §8.2 trap 2 says must
        # be recorded rather than left implicit, and `train_wm` reads them back off
        # metadata.json rather than off a flag, so a head can never be fit against
        # transitions a different gamma produced. The generator switches simulators
        # on `--gen-models`, so nothing here needs a `--epidemic` flag.
        epi_beta=config.epi_beta,
        epi_gamma=config.epi_gamma,
        epi_alpha=config.epi_alpha,
        epi_burn_in=config.epi_burn_in,
        outbreak_pct=(
            get_task(config.task).outbreak_pct
            if config.outbreak_pct is None
            else config.outbreak_pct
        ),
        outbreak_selectors=tuple(config.outbreak_selectors),
        immunizer_selectors=tuple(config.immunizer_selectors),
        # REPLAY rather than simulate, from the registry rather than a flag, for
        # the same reason `competitive` and `trace_parents` are. A cascade-prediction
        # dataset generated by forgetting this would be an NDlib rollout wearing the
        # name of a real corpus, which is the one thing §2.2 says invalidates every
        # number the task produces.
        cascade_corpus=get_task(config.task).observational,
        cp_observation=config.cp_observation,
        cp_horizon=config.cp_horizon,
        cp_step=config.cp_step,
        cp_min_size=config.cp_min_size,
        cp_truncate=config.cp_truncate,
        cp_split=config.cp_split,
        cp_target=config.cp_target,
        cp_graph=config.cp_graph,
        cp_max_cascades=config.cp_max_cascades,
        cp_max_nodes=config.cp_max_nodes,
    )

    return run_generation(generation_config)


def stage_train(config: PipelineConfig, layout: Layout) -> dict:
    results_path = layout.wm_results(config.wm_model, config.diffusion_model)
    checkpoint_path = layout.wm_checkpoint(config.wm_model, config.diffusion_model)

    if results_path.exists() and checkpoint_path.exists() and not config.force:
        existing = json.loads(results_path.read_text())
        print(
            f"[train] reusing {results_path} "
            f"(delta_f1={existing.get('test', {}).get('delta_f1', '?')}, "
            f"--force to retrain)"
        )
        return existing

    return train_world_model(
        TrainConfig(
            data_dir=str(layout.data_dir),
            diffusion_model=config.diffusion_model,
            model=config.wm_model,
            head=config.head,
            remove_semantics=config.remove_semantics,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            ffn_dim=config.ffn_dim,
            gcnii_alpha=config.gcnii_alpha,
            gcnii_lamda=config.gcnii_lamda,
            dropout=config.dropout,
            epochs=config.epochs,
            lr=config.lr,
            weight_decay=config.weight_decay,
            batch_size=config.batch_size,
            pos_weight=config.pos_weight,
            patience=config.patience,
            seed=config.seed,
            device=config.device,
            ckpt_dir=str(layout.world_model_dir),
            results=str(results_path),
            plan_demo=config.plan_demo,
            plan_graphs=config.plan_graphs,
            plan_budget_k=config.plan_budget_k,
            plan_budget_graphs=config.plan_budget_graphs,
            plan_budget_horizon=config.plan_budget_horizon,
            ood_policies=tuple(config.ood_policies),
            action_encoding=config.action_encoding,
            hide_edge_weights=config.hide_edge_weights,
        )
    )


def _skip_path(layout: Layout, label: str, arm) -> Path:
    return layout.baselines_dir / label / f"{arm.name}{skip_marker_suffix}"


def _clear_skip(layout: Layout, label: str, arm) -> None:
    _skip_path(layout, label, arm).unlink(missing_ok=True)


def _record_skip(
    layout: Layout, label: str, arm, reason: str, seconds: float | None = None
) -> None:
    """
    Persist WHY an arm was skipped.

    Without this a skipped arm is invisible on resume: the result file is
    absent, so the next run retries it and fails identically. The marker also
    keeps the report honest about what was attempted.
    """
    path = _skip_path(layout, label, arm)
    os.makedirs(path.parent, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "arm": arm.name,
                "arm_spec": arm.spec,
                "condition": arm.condition,
                "budget_label": label,
                "skipped": True,
                # What the failed attempt cost, so the report's timing split can
                # count it instead of leaving it as unexplained stage overhead
                "seconds": seconds,
                "reason": reason,
            },
            indent=2,
        )
    )


def _load_pipeline_graph(layout: Layout, config: PipelineConfig) -> GraphInfo:
    """The graph external baselines are handed, cached on the config."""
    if config._graph is None:
        store = load_graph_store(str(layout.data_dir))
        graph_id = config.graph_id or next(iter(store))
        config._graph = GraphInfo.from_store_entry(store[graph_id])

    return config._graph


def _budget_op(config: PipelineConfig) -> str:
    """What one unit of this run's budget buys, lever-aware."""
    task = get_task(config.task)

    return (
        resolve_lever(config.blocking_lever)[0]
        if task.competitive
        else resolve_epidemic_lever(config.epi_lever)[0]
        if task.epidemic
        else task.budget_op
    )


def _lever(config: PipelineConfig) -> str | None:
    task = get_task(config.task)

    return (
        config.blocking_lever
        if task.competitive
        else config.epi_lever
        if task.epidemic
        else None
    )


def _external_script(external_seeds: dict, config: PipelineConfig) -> str:
    """
    Canned script for a published repo's seed set.

    A rounds-aware repo returns its seeds in SELECTION order, so slicing by the
    schedule it was run at recovers the batches and the arm replays them one
    round at a time. Using the flat script for those would deal every batch at
    t=0 and put an adaptive method on the non-adaptive side of the gap table.
    """
    # A PREDICTOR's repo returns one NUMBER per cascade, so its canned script is a
    # predict() keyed by cascade id
    if "popularities" in external_seeds:
        return predict_script(external_seeds["popularities"])

    # A DECODER's repo returns one whole TRAJECTORY per observation, so its canned
    # script is a reconstruct() keyed by the observed report set
    if "trajectories" in external_seeds:
        return reconstruct_script(external_seeds["trajectories"])

    # An inverse task's repo returns one SOURCE SET PER OBSERVATION rather than
    # one seed set, so the canned script is a localize() keyed by observation
    if "sources" in external_seeds:
        return localize_script(external_seeds["sources"])

    budget_op = _budget_op(config)

    # A DISCOVERY system hands back a whole PROGRAM: wrap it so the contract
    # function runs under the identical executor, validation and referee
    if "program" in external_seeds:
        return program_script(
            external_seeds["program"], config.task, budget_op, _lever(config)
        )

    batches = external_seeds.get("batches")

    # An EDGE repo hands back a flat list of arc endpoints (see
    # registry._diffim_parse), which only the edge levers can emit
    if budget_op in ("remove_edge", "set_edge_weight"):
        return edge_script(external_seeds["seeds"], budget_op)

    if batches:
        return round_seed_script(
            external_seeds["seeds"], batches, config.round_gap, budget_op
        )

    return seed_script(external_seeds["seeds"], budget_op)


def _negative_seeds(config: PipelineConfig, graph: GraphInfo) -> tuple:
    """
    S_N for this sweep, derived exactly as `run_experiment` derives it.

    A pure function of (graph, size, selector, seed), so the external repo answers
    the SAME rumour our own arms do rather than one that merely looks like it.
    """
    pct = (
        get_task(config.task).outbreak_pct
        if config.outbreak_pct is None
        else config.outbreak_pct
    )

    return tuple(
        select_outbreak(
            graph,
            max(1, round(graph.num_nodes * pct / 100)),
            config.outbreak_selector,
            config.seed,
        )
    )


def _external_instances(config: PipelineConfig, layout: Layout, budget: int) -> list:
    """
    Both instance pools an inverse task's external repo has to predict for.

    Selection AND evaluation, in one batch, because the arm makes two passes
    (`run_experiment` scores the winner on the held-out pool after selecting on
    the other) and a repo invoked once has to cover both. `load_instances` is a
    pure function of the config, so this reproduces exactly what `run_experiment`
    loads; the canned script keys by OBSERVATION, so a disagreement surfaces as a
    loud KeyError rather than as a silently wrong row.
    """
    common = dict(
        graph_id=config.graph_id,
        observation=config.sl_observation,
        limit=config.sl_instances,
        budget_mode=config.sl_budget_mode,
        budget=budget,
        source_tolerance=config.sl_source_tolerance,
        seed=config.seed,
    )

    return load_instances(
        str(layout.data_dir), config.diffusion_model, config.sl_select_split, **common
    ) + load_instances(
        str(layout.data_dir), config.diffusion_model, config.sl_eval_split, **common
    )


def _external_cascades(config: PipelineConfig, layout: Layout) -> list:
    """
    Both cascade pools an external DECODER has to reconstruct.

    Same shape and same reason as `_external_instances`: the arm makes two passes
    and a repo invoked once has to cover both. The masking is reproduced exactly,
    `load_cascades` is a pure function of the config, and the canned script keys
    by the observed report set, so a disagreement surfaces as a loud KeyError
    rather than a silently wrong row.
    """
    common = dict(
        graph_id=config.graph_id,
        setting=config.cr_setting,
        observation_rate=config.cr_observation_rate,
        hidden_rate=config.cr_hidden_rate,
        limit=config.cr_instances,
        seed=config.seed,
        require_parents=config.cr_tree_weight > 0.0,
    )

    return load_cascades(
        str(layout.data_dir), config.diffusion_model, config.cr_select_split, **common
    ) + load_cascades(
        str(layout.data_dir), config.diffusion_model, config.cr_eval_split, **common
    )


def _external_forecasts(config: PipelineConfig, layout: Layout) -> list:
    """
    Both cascade pools an external PREDICTOR has to forecast for.

    Same shape and same reason as `_external_instances` and `_external_cascades`:
    the arm makes two passes and a repo invoked once has to cover both.
    `load_forecasts` is a pure function of the config, so this reproduces exactly
    what `run_experiment` loads, and the canned script keys by cascade id, so a
    disagreement surfaces as a loud KeyError rather than a silently wrong row.

    The selection pool crosses first, which is what `registry.training_rows` reads:
    every repo here is a SUPERVISED regressor that fits on labelled cascades before
    predicting, so the split flag is what keeps a label out of an evaluation-row
    prediction.
    """
    common = dict(
        graph_id=config.graph_id,
        observed_steps=config.cp_observation_steps,
        limit=config.cp_instances,
        seed=config.seed,
    )

    return load_forecasts(
        str(layout.data_dir), config.diffusion_model, config.cp_select_split, **common
    ) + load_forecasts(
        str(layout.data_dir), config.diffusion_model, config.cp_eval_split, **common
    )


def _wm_results_path(config: PipelineConfig, layout: Layout) -> Path:
    """The results JSON every world-model reader shares: a reused checkpoint or this run's own."""
    return (
        Path(config.wm_results_json)
        if config.wm_results_json is not None
        else layout.wm_results(config.wm_model, config.diffusion_model)
    )


def _check_wm_results(config: PipelineConfig, path: Path) -> None:
    """
    Refuse a checkpoint trained under other dynamics or on another dataset.

    `coding_agent.run` checks the state layout and the remove semantics but not
    these two, and a reused --wm-results-json is the one path where they can
    disagree: an LT head rolled out as the IC evaluator scores every arm on the
    wrong transition and nothing downstream can tell.
    """
    trained = json.loads(path.read_text()).get("config") or {}
    dynamics = trained.get("diffusion_model")
    if dynamics is not None and dynamics != config.diffusion_model:
        raise ValueError(
            f"{path} was trained under {dynamics} but this run is "
            f"--diffusion-model {config.diffusion_model}; point --wm-results-json "
            f"at a {config.diffusion_model} checkpoint or change the run's dynamics"
        )

    # The checkpoint records its data directory rather than its dataset, so the
    # dataset is read off that directory's own metadata while it still exists
    data_dir = trained.get("data_dir")
    metadata = Path(data_dir) / "metadata.json" if data_dir else None
    if metadata is not None and metadata.exists():
        dataset = (json.loads(metadata.read_text()).get("config") or {}).get("dataset")
        if dataset is not None and dataset != config.dataset:
            raise ValueError(
                f"{path} was trained on dataset {dataset!r} (from {metadata}) but "
                f"this run is --dataset {config.dataset}; a world model is a "
                f"transition model of one graph store and does not transfer by "
                f"pointing at it"
            )


def stage_agent(config: PipelineConfig, layout: Layout) -> list[dict]:
    wm_results = _wm_results_path(config, layout)
    arms = build_arms(config)

    if needs_world_model(arms):
        if not wm_results.exists():
            raise FileNotFoundError(
                f"arms {[arm.spec for arm in arms if needs_world_model([arm])]} "
                f"evaluate against the world model but {wm_results} is missing; "
                f"run the train stage or drop those arms"
            )

        _check_wm_results(config, wm_results)

    # Rewards from different evaluators are not comparable, so a multi-condition
    # sweep is only readable once every arm has been replayed on the same referee.
    # An INVERSE task is the exception and the warning would be wrong there: its
    # reward is F1 against a source set we know, so it carries no evaluator noise
    # and is already comparable across conditions. --compare still buys the
    # re-simulated error column, it just is not load-bearing for the table.
    if (
        not config.compare
        and not get_task(config.task).recovers
        # ...and a FORECAST task, for the same reason and with the same wrinkle: its
        # error is measured against a popularity read off a log, so it carries no
        # evaluator noise. --compare still buys the modelling-error column, which is
        # the number §9.1 is actually about; it just is not load-bearing for the table.
        and not get_task(config.task).forecasts
        and len({arm.evaluator for arm in arms}) > 1
    ):
        print(
            "[agent] WARNING: arms span multiple evaluators without --compare, so "
            "their rewards are measured by different judges and cannot be compared. "
            "Re-run with --compare for a valid table."
        )

    points = budget_points(config)
    completed = []
    reused = failed = 0

    # One bar over the whole (budget x arm) grid: this stage dominates wall
    # clock and previously reported nothing but per-arm prints
    progress_bar = tqdm(
        total=len(points) * len(arms), desc="agent runs", unit="run"
    )

    for label, budget_pct, budget in points:
        for arm in arms:
            progress_bar.set_postfix_str(f"{label}/{arm.name}")
            out_json = layout.agent_result(
                label, arm.name, external=arm.external is not None
            )

            if out_json.exists() and not config.force:
                tqdm.write(f"[agent] {label}/{arm.name}: reusing cached result")
                completed.append(json.loads(out_json.read_text()))
                reused += 1
                progress_bar.update(1)
                continue

            evaluator, mc_runs = resolve_evaluator(
                arm, config.mc_runs, config.native_mc_runs
            )
            tqdm.write(
                f"[agent] {label}/{arm.name}: running "
                f"(condition {arm.condition}: {arm.condition_name}, "
                f"evaluator={evaluator}, mc_runs={mc_runs})"
            )
            arm_start = time.perf_counter()

            experiment = ExperimentConfig(
                task=config.task,
                method=arm.method,
                strategy_mode=arm.strategy_mode,
                evaluator=evaluator,
                model=arm.llm_model or config.llm_model,
                temperature=config.temperature,
                diffusion_model=config.diffusion_model,
                remove_semantics=config.remove_semantics,
                # Inert unless arm.method == "adaptive"; passing them always is
                # what lets one sweep hold both sides of the adaptivity gap
                rounds=config.rounds,
                per_round_budget=config.per_round_budget,
                round_gap=config.round_gap,
                feedback_model=config.feedback_model,
                edit_rate=config.edit_rate,
                campaigns=config.campaigns,
                # Inert unless the task registry gives this task an outbreak;
                # passed always so one code path serves both families
                outbreak_pct=config.outbreak_pct,
                outbreak_selector=config.outbreak_selector,
                # Inert unless the task registry marks the task competitive; passed
                # always so one code path serves the blocking family too
                blocking_lever=config.blocking_lever,
                tie_break=config.tie_break,
                positive_prob=config.positive_prob,
                detection_delay=config.detection_delay,
                # Inert unless the task registry marks the task epidemic; passed
                # always so one code path serves the compartmental family too
                epi_lever=config.epi_lever,
                contact_reduction=config.contact_reduction,
                epi_beta=config.epi_beta,
                epi_gamma=config.epi_gamma,
                epi_alpha=config.epi_alpha,
                epi_burn_in=config.epi_burn_in,
                # Inert unless the task inverts. `native_arm` cannot be recovered
                # downstream: resolve_evaluator has already rewritten a native
                # arm to monte_carlo with one episode, and it decides whether
                # `predict_marginals` exists at all, which IS condition 3.
                sl_select_split=config.sl_select_split,
                sl_eval_split=config.sl_eval_split,
                sl_instances=config.sl_instances,
                sl_observation=config.sl_observation,
                sl_budget_mode=config.sl_budget_mode,
                sl_source_tolerance=config.sl_source_tolerance,
                sl_transfer_from=config.sl_transfer_from,
                # ...and inert unless the task decodes
                cr_setting=config.cr_setting,
                cr_observation_rate=config.cr_observation_rate,
                cr_hidden_rate=config.cr_hidden_rate,
                cr_instances=config.cr_instances,
                cr_select_split=config.cr_select_split,
                cr_eval_split=config.cr_eval_split,
                cr_tree_weight=config.cr_tree_weight,
                # ...and inert unless the task forecasts
                cp_select_split=config.cp_select_split,
                cp_eval_split=config.cp_eval_split,
                cp_instances=config.cp_instances,
                cp_observation_steps=config.cp_observation_steps,
                cp_metric=config.cp_metric,
                cp_target=config.cp_target,
                cp_forecast_samples=config.cp_forecast_samples,
                native_arm=arm.evaluator == native,
                budget=budget or 5,
                budget_pct=budget_pct,
                horizon=config.horizon,
                windows=config.windows,
                # Conditions 1 and 2 have no refinement loop: one pass is the arm.
                # Nor does a transfer arm: the program is fixed and came from
                # another run, so refining it here would defeat the experiment
                # (and would re-run the same canned script --outer-iters times).
                outer_iters=(
                    1
                    if not arm.is_agent or config.sl_transfer_from is not None
                    else config.outer_iters
                ),
                mc_runs=mc_runs,
                referee_mc_runs=config.mc_runs,
                # Adaptive arms are capped: act() runs once per ensemble member per
                # round, so the policy's own compute scales linearly with the sample
                # count and 200 samples turns a 500 s evaluation into a 30-minute
                # one. Every other arm takes the flag as given; 200 is the ladder
                # default because 50 samples put the reward SE (2.5-12 nodes) above
                # the deltas the late search iterations are deciding between.
                n_samples=(
                    min(config.n_samples, adaptive_n_samples)
                    if arm.method == "adaptive"
                    else config.n_samples
                ),
                seed=config.seed,
                device=config.device,
                data_dir=str(layout.data_dir),
                graph_id=config.graph_id,
                wm_results_json=str(wm_results) if wm_results.exists() else None,
                compare=config.compare,
                credit=config.credit,
                baseline=arm.baseline,
                routing=arm.routing,
                allowed_ops=resolve_allowed_ops(config),
                allow_mc_algorithms=config.allow_mc_algorithms,
                strategy_timeout=config.strategy_timeout,
                llm_price_in=config.llm_price_in,
                llm_price_out=config.llm_price_out,
                # --force means redo, so it must not silently resume a killed
                # search from the checkpoint it left behind
                resume=not config.force,
                out_json=str(out_json),
            )

            # An external repo only hands back a seed set; wrapping it as a
            # canned Strategy routes it through the identical scoring path
            external_seeds = None
            if arm.external is not None:
                graph = _load_pipeline_graph(layout, config)
                resolved = budget or max(
                    1, round(graph.num_nodes * budget_pct / 100)
                )
                # A rounds-aware repo selects in batches, so it needs the same
                # schedule the adaptive arms run at, or it would be handed one
                # batch of k and quietly become a static method
                external_batches = (
                    round_batches(resolved, config.rounds, config.per_round_budget)
                    if external_baselines[arm.external].rounds_aware
                    else None
                )

                # An inverse task's repo predicts per OBSERVATION, so it is handed
                # both instance pools at once rather than a budget. A DECODER's
                # pools carry whole masked histories rather than endpoints, which
                # is the only difference.
                if get_task(config.task).forecasts:
                    external_instances = _external_forecasts(config, layout)
                elif get_task(config.task).reconstructs:
                    external_instances = _external_cascades(config, layout)
                elif get_task(config.task).recovers:
                    external_instances = _external_instances(config, layout, resolved)
                else:
                    external_instances = None

                # A blocking repo has to be told which rumour it is answering: two of
                # the three would otherwise draw their own from a fixed seed and
                # answer a different one than every arm it is being compared against.
                # An EPIDEMIC repo needs the same channel for a different reason,
                # DAVA and NetShape are DEFINED on the observed infected set, so a
                # run without it is answering a structural question rather than the
                # data-aware one that is the whole point of those rows.
                task_registry = get_task(config.task)
                external_negative = (
                    _negative_seeds(config, graph)
                    if task_registry.competitive or task_registry.epidemic
                    else None
                )

                # A discovery system scores its candidates under this arm's own
                # ExperimentConfig, so the task settings (outbreak, lever, splits,
                # seed count) match the arms it is compared against; a seed-set
                # repo needs nothing of the kind
                external_spec = external_baselines[arm.external]
                external_context = (
                    build_context(
                        experiment,
                        config.task,
                        _budget_op(config),
                        layout.baselines_dir / "_runs" / arm.external / label,
                        graph,
                    )
                    if external_spec.kind == discovery_kind
                    else None
                )

                attempt_start = time.perf_counter()

                try:
                    external_seeds = run_external_baseline(
                        arm.external,
                        graph,
                        resolved,
                        config.diffusion_model,
                        work_dir=layout.baselines_dir / "_runs" / arm.external / label,
                        timeout=config.baseline_timeout,
                        batches=external_batches,
                        instances=external_instances,
                        negative_seeds=external_negative,
                        context=external_context,
                    )
                except BaselineError as error:
                    tqdm.write(f"[agent] {label}/{arm.name}: SKIPPED, {error}")
                    _record_skip(
                        layout,
                        label,
                        arm,
                        str(error),
                        seconds=time.perf_counter() - attempt_start,
                    )
                    failed += 1
                    progress_bar.update(1)
                    continue

            # An agent arm that never produces a runnable program is a RESULT
            # about that arm, not an infrastructure failure, so it is recorded
            # like a dead external repo rather than killing the sweep. Letting it
            # propagate cost the plots and the report for every arm that had
            # already succeeded, which is the expensive half of the run.
            try:
                result = run_experiment(
                    experiment,
                    canned_script=(
                        _external_script(external_seeds, config)
                        if external_seeds
                        else None
                    ),
                )
            except executor.StrategyError as error:
                tqdm.write(f"[agent] {label}/{arm.name}: SKIPPED, {error}")
                _record_skip(layout, label, arm, str(error))
                failed += 1
                progress_bar.update(1)
                continue

            result["arm"] = arm.name
            result["arm_spec"] = arm.spec
            result["condition"] = arm.condition
            result["condition_name"] = arm.condition_name
            result["budget_label"] = label

            # This arm skipped on an earlier run and works now, so retract the
            # marker. Nothing else deletes them, and a stale one makes the run
            # claim a failure that the results table simultaneously disproves.
            _clear_skip(layout, label, arm)

            if external_seeds is not None:
                spec = external_baselines[arm.external]
                result["external"] = {
                    "name": arm.external,
                    "title": spec.title,
                    "venue": spec.venue,
                    "kind": spec.kind,
                    "repo": spec.repo,
                    "paper": spec.paper,
                    # One of the two, never both: a seed/removal set for an
                    # intervention task, per-instance source sets for an inverse one
                    "seeds": external_seeds.get("seeds"),
                    "instances": external_seeds.get("instances"),
                    "trajectories": (
                        len(external_seeds["trajectories"])
                        if external_seeds.get("trajectories")
                        else None
                    ),
                    "popularities": (
                        len(external_seeds["popularities"])
                        if external_seeds.get("popularities")
                        else None
                    ),
                    "instances_short": external_seeds.get("instances_short"),
                    # Time the external repo itself spent selecting; our scoring
                    # time is in elapsed_seconds as for every other arm
                    "selection_seconds": external_seeds["seconds"],
                    # Condition 9: the artefact was a PROGRAM, and `info` is the
                    # framework's own bookkeeping (its internal score, iteration)
                    "program": external_seeds.get("program") is not None,
                    "info": external_seeds.get("info"),
                }
                result["model"] = (
                    f"discovery:{arm.external}"
                    if spec.kind == discovery_kind
                    else f"external:{arm.external}"
                )

            out_json.write_text(json.dumps(result, indent=2, default=str))
            completed.append(result)

            usage = result.get("llm_usage") or {}
            token_text = (
                ""
                if not usage.get("calls")
                else (
                    f", {usage['calls']} llm calls / "
                    f"{usage['total_tokens']:,} tokens"
                    + (
                        ""
                        if usage.get("cost_usd") is None
                        else f" / ${usage['cost_usd']:.4f}"
                    )
                )
            )
            score = ground_truth_reward(result)
            score_text = (
                f"held-out {result.get('prediction_metric', 'error').upper()} "
                f"{score:.4f} (lower is better)"
                if result.get("prediction")
                else f"held-out reward {score:.4f} (kernel log-lik/node minus violations)"
                if result.get("reconstruction")
                else f"held-out consistency {score:.4f} (minus resim MSE)"
                if result.get("localization")
                else (
                    f"spread {score:.2f} "
                    f"({100.0 * score / result['graph']['num_nodes']:.1f}% of N)"
                )
            )
            tqdm.write(
                f"[agent] {label}/{arm.name}: done in "
                f"{time.perf_counter() - arm_start:.1f}s -> {score_text}{token_text}"
            )
            progress_bar.update(1)

            # Refresh the flat table after EVERY run, so a killed sweep still
            # leaves a readable summary of everything finished so far
            write_summary(layout, completed)

    progress_bar.close()
    print(
        f"[agent] {len(completed)} results ({reused} reused, "
        f"{len(completed) - reused} new), {failed} skipped"
    )

    return completed


def load_agent_results(layout: Layout) -> list[dict]:
    """Every result JSON on disk: our arms and external baselines together."""
    results = []

    for path in layout.result_globs():
        result = json.loads(path.read_text())
        result.setdefault("arm", path.stem)
        result.setdefault("budget_label", path.parent.name)
        results.append(result)

    return results


def load_skips(layout: Layout) -> list[dict]:
    """The `.skipped.json` markers: what was attempted and did not produce a row."""
    return [json.loads(path.read_text()) for path in layout.skip_globs()]


def report_skips(layout: Layout) -> None:
    """
    An arm that skipped is invisible in the figures, so say so out loud.

    Silence here reads as "everything ran", which is exactly the wrong
    impression when an external repo failed on every budget.
    """
    skips = load_skips(layout)
    if not skips:
        return

    reasons = {}
    for skip in skips:
        reasons.setdefault(skip["arm"], []).append(skip.get("budget_label", "?"))

    print(f"[plots] {len(skips)} skipped run(s) excluded from the figures:")
    for arm, labels in sorted(reasons.items()):
        reason = next(s["reason"] for s in skips if s["arm"] == arm)
        print(f"[plots]   {arm} ({', '.join(labels)}): {reason.splitlines()[0]}")


def load_wm_results(config: PipelineConfig, layout: Layout) -> dict | None:
    # The same JSON the agent stage rolled out against: a reused checkpoint lives
    # outside this run's tree, and reading only the layout path reported "not
    # trained" beside a table full of @world_model rows
    path = _wm_results_path(config, layout)

    return json.loads(path.read_text()) if path.exists() else None


def _write_manifest(layout: Layout, config: PipelineConfig) -> None:
    """Config + per-stage status, rewritten whenever anything changes."""
    # A resumed run only carries the stages IT ran; merging the previous
    # manifest's entries underneath keeps the earlier stages' recorded seconds,
    # which is what the report's total is summed from
    if layout.manifest_path.exists():
        previous = json.loads(layout.manifest_path.read_text()).get("stages") or {}
        config.stage_status = {**previous, **config.stage_status}

    layout.manifest_path.write_text(
        json.dumps(
            {
                "config": {
                    key: value
                    for key, value in vars(config).items()
                    if not key.startswith("_")
                },
                "stages": config.stage_status,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            default=str,
        )
    )


def resolve_dynamics(config: PipelineConfig) -> None:
    """
    Default `--gen-models` / `--diffusion-model` to the TASK's own dynamics family.

    IC/LT and SIR/SIS/SEIR produce different state layouts (two overlapping
    indicators vs four exclusive compartments) and different target widths, so a
    dataset holds one family or the other. Rather than making every epidemic
    invocation restate both flags, the task's own dynamics are the default and a
    genuine mismatch raises here instead of at the first matmul.
    """
    task = get_task(config.task)
    compartmental = set(config.gen_models) & set(epidemic_dynamics)

    if task.epidemic:
        if config.diffusion_model not in task.dynamics:
            config.diffusion_model = task.dynamics[0]

        if set(config.gen_models) - set(epidemic_dynamics):
            config.gen_models = (config.diffusion_model,)

        if config.diffusion_model not in config.gen_models:
            raise ValueError(
                f"--diffusion-model {config.diffusion_model} is not in --gen-models "
                f"{list(config.gen_models)}, so the agent stage would be handed a "
                f"dataset that was never generated for it"
            )
    elif compartmental or config.diffusion_model in epidemic_dynamics:
        raise ValueError(
            f"task {config.task!r} runs on IC/LT but was given compartmental "
            f"dynamics (gen_models={list(config.gen_models)}, "
            f"diffusion_model={config.diffusion_model}). SIR/SIS/SEIR produce four "
            f"exclusive compartments and a 5-target head; only "
            f"--task epidemic_control reads them."
        )
    elif config.diffusion_model not in config.gen_models:
        # Same check the epidemic branch makes, or the mismatch surfaces as a
        # missing transitions_<dm>_train.jsonl after the whole data stage has run
        raise ValueError(
            f"--diffusion-model {config.diffusion_model} is not in --gen-models "
            f"{list(config.gen_models)}, so the train and agent stages would be "
            f"handed a dataset that was never generated for it"
        )


def run_pipeline(config: PipelineConfig) -> dict:
    pipeline_start = time.perf_counter()

    # Resolved here rather than only in __main__, so a programmatically built
    # config gets the task's semantics instead of leaking None into GenConfig
    if config.remove_semantics is None:
        config.remove_semantics = get_task(config.task).remove_semantics

    resolve_dynamics(config)
    resolve_cascade_protocol(config)

    layout = Layout(
        config.task, config.dataset, config.run, root=config.results_root
    )
    os.makedirs(layout.root, exist_ok=True)

    selected = active_stages(config)
    arms = build_arms(config)
    print(f"[pipeline] {layout.label} -> {layout.root}")
    print(f"[pipeline] stages: {' -> '.join(selected)}")
    print(f"[pipeline] {len(arms)} arms x {len(budget_points(config))} budgets:")

    for condition in sorted({arm.condition for arm in arms}):
        members = [arm.name for arm in arms if arm.condition == condition]
        print(
            f"[pipeline]   {condition}. {condition_names[condition]}: "
            f"{', '.join(members)}"
        )

    # Provenance first: written before any work so even a stage-1 crash records
    # which commit and machine produced the tree
    write_environment(layout)

    for index, stage in enumerate(selected, start=1):
        stage_start = time.perf_counter()
        print(
            f"\n{'=' * 72}\n"
            f"[pipeline] STAGE {index}/{len(selected)}: {stage.upper()}"
            f"   (elapsed {time.perf_counter() - pipeline_start:.0f}s)\n"
            f"{'=' * 72}"
        )

        # Mark the stage in-flight so a killed run is distinguishable from a
        # clean one when the manifest is read back
        config.stage_status[stage] = {"status": "running", "seconds": None}
        _write_manifest(layout, config)

        try:
            if stage == "data":
                stage_data(config, layout)
            elif stage == "train":
                stage_train(config, layout)
            elif stage == "agent":
                stage_agent(config, layout)
            elif stage == "plots":
                results = load_agent_results(layout)
                print(f"[plots] building figures from {len(results)} results...")
                report_skips(layout)
                figures = build_plots(
                    results,
                    load_wm_results(config, layout),
                    layout.plots_dir,
                    title_prefix=layout.label,
                    diffusion_model=config.diffusion_model,
                )
                for figure in figures:
                    print(f"[plots]   {figure.name}")
                print(f"[plots] wrote {len(figures)} figures -> {layout.plots_dir}")
            elif stage == "report":
                results = load_agent_results(layout)
                csv_path, json_path = write_summary(layout, results)
                print(f"[report] summary table -> {csv_path} ({len(results)} rows)")
                print(f"[report] summary json  -> {json_path}")
                write_report(
                    config=vars(config),
                    layout=layout,
                    agent_results=results,
                    wm_results=load_wm_results(config, layout),
                )
                print(f"[report] markdown      -> {layout.report_path}")
        except Exception as error:
            config.stage_status[stage] = {
                "status": "failed",
                "seconds": round(time.perf_counter() - stage_start, 1),
                "error": f"{type(error).__name__}: {error}",
            }
            _write_manifest(layout, config)
            print(f"[pipeline] stage {stage} FAILED: manifest updated, re-run to resume")
            raise

        seconds = time.perf_counter() - stage_start
        config.stage_status[stage] = {
            "status": "done",
            "seconds": round(seconds, 1),
        }
        print(f"[pipeline] stage {stage} done in {seconds:.1f}s")

        # Manifest is rewritten after each stage so a crash still leaves a record
        _write_manifest(layout, config)

    elapsed = time.perf_counter() - pipeline_start
    print(f"\n[pipeline] done in {elapsed:.1f}s -> {layout.root}")

    return {
        "task": config.task,
        "dataset": config.dataset,
        "run": config.run,
        "stages": config.stage_status,
        "seconds": elapsed,
    }


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="End-to-end GWM experiment pipeline for one dataset"
    )

    # Driver
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="graph dataset or synthetic family (default: required).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="influence_maximization",
        choices=task_names(),
        help="graph task; the first level of the results tree and the name of "
        "its literature review in research/ "
        "(default: influence_maximization).",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=default_run,
        help="run label under results/<task>/<dataset>/, for holding several "
        f"variants of one dataset side by side (default: {default_run}).",
    )
    parser.add_argument(
        "--start-stage",
        type=str,
        default=stages[0],
        choices=stages,
        help=f"first stage to run (default: {stages[0]}).",
    )
    parser.add_argument(
        "--end-stage",
        type=str,
        default=stages[-1],
        choices=stages,
        help=f"last stage to run (default: {stages[-1]}).",
    )
    parser.add_argument(
        "--skip-stages",
        type=str,
        nargs="*",
        default=[],
        choices=stages,
        help="stages to omit from the selected range (default: none).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute stages whose outputs already exist (default: False).",
    )
    parser.add_argument(
        "--print-resources",
        action="store_true",
        help="print whether this run needs a GPU and exit without running "
        "anything; the sbatch submit path reads it (default: False).",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results",
        help="top-level results directory; the tree below it is "
        "<task>/<dataset>/<run> (default: results).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)."
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="torch device string (default: cpu)."
    )

    # Data stage
    parser.add_argument(
        "--num-graphs",
        type=int,
        default=1,
        help="synthetic graph instances; ignored for real datasets (default: 1).",
    )
    parser.add_argument(
        "--syn-nodes",
        type=int,
        default=100,
        help="nodes per synthetic graph (default: 100).",
    )
    parser.add_argument(
        "--er-p", type=float, default=0.05, help="ER edge probability (default: 0.05)."
    )
    parser.add_argument(
        "--ba-m", type=int, default=3, help="BA attachment count (default: 3)."
    )
    parser.add_argument(
        "--ws-k", type=int, default=6, help="WS ring neighbours (default: 6)."
    )
    parser.add_argument(
        "--ws-p", type=float, default=0.1, help="WS rewire probability (default: 0.1)."
    )
    parser.add_argument(
        "--sbm-blocks", type=int, default=4, help="SBM community count (default: 4)."
    )
    parser.add_argument(
        "--sbm-p-in",
        type=float,
        default=0.15,
        help="SBM within-block edge probability (default: 0.15).",
    )
    parser.add_argument(
        "--sbm-p-out",
        type=float,
        default=0.01,
        help="SBM across-block edge probability (default: 0.01).",
    )
    parser.add_argument(
        "--plc-m",
        type=int,
        default=3,
        help="powerlaw_cluster: edges added per new node. RL4IM's own config uses "
        "3; its paper says avg degree 3, which no integer m produces "
        "(default: 3).",
    )
    parser.add_argument(
        "--plc-p",
        type=float,
        default=0.05,
        help="powerlaw_cluster: triangle-closing probability (RL4IM: 0.05) "
        "(default: 0.05).",
    )
    parser.add_argument(
        "--kron-variant",
        type=str,
        default="core_periphery",
        choices=sorted(kronecker_seeds),
        help="kronecker seed matrix, from ConTinEst (default: core_periphery).",
    )
    parser.add_argument(
        "--gen-models",
        type=str,
        nargs="+",
        default=["IC", "LT"],
        choices=["IC", "LT"] + list(epidemic_dynamics),
        help="dynamics to generate transitions for. IC/LT run NDlib and produce two "
        "overlapping state indicators; SIR/SIS/SEIR run data/wm_epidemic.py and "
        "produce four exclusive compartments, so the two families cannot share one "
        "dataset (default: IC LT).",
    )
    parser.add_argument(
        "--gen-algorithms",
        type=str,
        nargs="+",
        default=["random", "degree", "pagerank", "betweenness"],
        choices=list(spine_algorithms),
        help="spine seed selectors driving the episodes (default: random degree pagerank betweenness).",
    )
    parser.add_argument(
        "--gen-action-ops",
        type=str,
        nargs="+",
        default=None,
        choices=list(valid_action_ops),
        help="action ops injected during generation. Unset = the task registry's own set: add_node remove_node for every intervention task except influence blocking and epidemic control, whose levers add the edge ops, and empty for the three tasks whose episodes must be pure diffusion (default: None).",
    )
    parser.add_argument(
        "--trace-parents",
        action="store_true",
        default=None,
        help="record which u infected v on every generated record: the "
        "transmission edge cascade reconstruction's tree reward needs and NDlib "
        "does not otherwise emit. Unset = on for --task cascade_reconstruction and "
        "off otherwise; pass it to generate a dataset another task can share with "
        "that one (default: None).",
    )
    parser.add_argument(
        "--remove-semantics",
        type=str,
        default=None,
        choices=list(valid_remove_semantics),
        help="what remove_node means across all three stages: spent = stays "
        "counted, stops spreading; blocked = deleted from the graph, uncounted, "
        "cannot transmit or be infected (default: the task registry's value).",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default=None,
        choices=list(valid_split_modes),
        help="how train/val/test are assigned. graph_disjoint keeps every episode "
        "of a graph in one split; episode_random draws per episode and LEAKS a "
        "graph across splits, but is the only option for a single-graph real "
        "dataset, whose test metrics are then in-graph rather than "
        "held-out-graph (default: graph_disjoint for a synthetic family with "
        "at least 3 graphs, episode_random otherwise).",
    )
    parser.add_argument(
        "--prob-model",
        type=str,
        default="weighted",
        choices=["weighted", "uniform"],
        help="IC transmission probability model (default: weighted).",
    )
    parser.add_argument(
        "--uniform-p",
        type=float,
        default=0.1,
        help="IC probability when --prob-model uniform (default: 0.1).",
    )
    parser.add_argument(
        "--budget-pct-range",
        type=float,
        nargs=2,
        default=list(default_budget_pct_range),
        metavar=("LO", "HI"),
        help=f"per-episode seed budget band for generation (default: {default_budget_pct_range[0]} {default_budget_pct_range[1]}).",
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=20,
        help="episodes per (graph, dynamics, selector) (default: 20).",
    )
    parser.add_argument(
        "--gen-horizon",
        type=int,
        default=10,
        help="max timesteps per generated episode (default: 10).",
    )
    parser.add_argument(
        "--inject-p",
        type=float,
        default=0.4,
        help="probability an intermediate step injects an action (default: 0.4).",
    )
    parser.add_argument(
        "--cf-prob",
        type=float,
        default=0.4,
        help="probability a step spawns counterfactual forks (default: 0.4).",
    )
    parser.add_argument(
        "--cf-branches",
        type=int,
        default=2,
        help="counterfactual branches per fork (default: 2).",
    )
    parser.add_argument(
        "--mc-marginals",
        type=int,
        default=30,
        help="Monte Carlo draws per step for the soft targets (default: 30).",
    )
    parser.add_argument(
        "--split",
        type=float,
        nargs=3,
        default=[0.7, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="transition split fractions (default: 0.7 0.15 0.15).",
    )

    # Train stage
    parser.add_argument(
        "--wm-model",
        type=str,
        default="sage",
        choices=list(backbones),
        help="world-model encoder backbone (default: sage).",
    )
    parser.add_argument(
        "--head",
        type=str,
        default="structured",
        choices=["linear", "structured", "structured_residual"],
        help="world-model output head (default: structured).",
    )
    # The checkpoint is located via `ckpt_dir` inside this JSON, and the
    # architecture flags are read from its `config` block, so this one path
    # replaces --wm-model/--head/--hidden-dim/... as well as the train stage
    parser.add_argument(
        "--wm-results-json",
        type=str,
        default=None,
        help="reuse an already-trained world model from this train_wm.py results "
        "JSON instead of running the train stage (default: None).",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=64, help="hidden dimension (default: 64)."
    )
    parser.add_argument(
        "--n-layers", type=int, default=3, help="encoder layers (default: 3)."
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=4,
        help="attention heads; gat and gt only (default: 4).",
    )
    parser.add_argument(
        "--ffn-dim",
        type=int,
        default=128,
        help="feed-forward width; gt only (default: 128).",
    )
    parser.add_argument(
        "--gcnii-alpha",
        type=float,
        default=0.1,
        help="GCNII initial-residual strength; gcnii only (default: 0.1).",
    )
    parser.add_argument(
        "--gcnii-lamda",
        type=float,
        default=0.5,
        help="GCNII identity-mapping decay; gcnii only (default: 0.5).",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="dropout probability (default: 0.1)."
    )
    parser.add_argument(
        "--epochs", type=int, default=400, help="max training epochs (default: 400)."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Adam learning rate (default: 1e-3)."
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
        help="Adam weight decay (default: 5e-4).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="transition batch size (default: 32)."
    )
    parser.add_argument(
        "--pos-weight",
        type=str,
        default="off",
        choices=["auto", "off"],
        help="positive-class weighting; off for structured heads (default: off).",
    )
    parser.add_argument(
        "--patience", type=int, default=50, help="early-stop patience (default: 50)."
    )
    parser.add_argument(
        "--no-plan-demo",
        action="store_true",
        help="skip the multi-graph planning demo after training (default: False).",
    )
    parser.add_argument(
        "--plan-graphs",
        type=int,
        default=5,
        help="graphs used for the planning demo (default: 5).",
    )
    parser.add_argument(
        "--plan-budget-k",
        type=int,
        default=0,
        help="k-seed FULL-HORIZON planning regret vs greedy-MC — the IM problem as "
        "posed, unlike the single-step --plan-graphs demo. 0 disables (default: 0).",
    )
    parser.add_argument(
        "--plan-budget-graphs",
        type=int,
        default=3,
        help="graphs for the k-seed planning eval (default: 3).",
    )
    parser.add_argument(
        "--plan-budget-horizon",
        type=int,
        default=20,
        help="rollout horizon for the k-seed planning eval (default: 20).",
    )
    parser.add_argument(
        "--ood-policies",
        type=str,
        nargs="*",
        default=[],
        help="off-policy rollout fidelity tests, e.g. `degree_seed null`. The "
        "recorded action sequence is on-policy by construction, so it says nothing "
        "about the action distribution an agent proposes (default: none).",
    )
    parser.add_argument(
        "--action-encoding",
        type=str,
        default=basic_encoding,
        choices=list(valid_action_encodings),
        help="`typed` adds 3 channels splitting act_edge by op, so add_edge and "
        "remove_edge on the same endpoints stop producing identical X. Changes "
        "in_channels, so checkpoints are not portable across the two "
        "(default: basic).",
    )
    parser.add_argument(
        "--hide-edge-weights",
        action="store_true",
        help="train the world model on ones instead of the true IC transmission "
        "probability: the online/bandit information state, and the ablation for "
        "the IC heads otherwise seeing w. Needs --head structured (default: False).",
    )

    # Agent stage
    parser.add_argument(
        "--rounds",
        type=int,
        default=4,
        help="adaptive IM: batches the budget is committed in, each chosen after "
        "seeing the previous one's diffusion. Read only by `adaptive` arms "
        "(default: 4).",
    )
    parser.add_argument(
        "--per-round-budget",
        type=int,
        default=None,
        help="adaptive IM: fix seeds per round b and derive r = ceil(k/b), "
        "instead of fixing r with --rounds. This is Han et al.'s b-sweep; "
        "--rounds is their k-sweep (default: None).",
    )
    parser.add_argument(
        "--round-gap",
        type=int,
        default=1,
        help="adaptive IM: timesteps of diffusion between consecutive rounds "
        "(default: 1).",
    )
    parser.add_argument(
        "--feedback-model",
        type=str,
        default=full_adoption,
        choices=list(valid_feedback_models),
        help="adaptive IM: what the policy observes at a round boundary; myopic "
        "hides state.infected and leaves only the current wave (default: "
        "full_adoption).",
    )
    parser.add_argument(
        "--edit-rate",
        type=float,
        default=0.0,
        help="dynamic/streaming IM: exogenous edge edits per timestep as a "
        "fraction of |E|, applied to EVERY arm so the comparison stays fair. "
        "0 = static graph (default: 0.0).",
    )
    parser.add_argument(
        "--campaigns",
        type=int,
        default=1,
        help="multi-round IM: separate campaigns of k seeds each, scored on the "
        "union of what they activate. Composes with any evaluator. 1 = a single "
        "campaign (default: 1).",
    )
    parser.add_argument(
        "--outbreak-pct",
        type=float,
        default=None,
        help="critical node detection: size of the exogenous outbreak every arm "
        "must contain, as a percentage of N. Unset = the task registry's value, "
        "which is 10.0 for critical_node_detection, 1.0 for influence_blocking "
        "and epidemic_control, and 0 for every seeding task (default: None).",
    )
    parser.add_argument(
        "--outbreak-selector",
        type=str,
        default="random",
        choices=list(outbreak_selectors),
        help="how the outbreak's sources are chosen; deterministic in --seed so "
        "every arm faces the same one. `random` is the honest default: a targeted "
        "outbreak makes blocking the same ranking problem (default: random).",
    )
    parser.add_argument(
        "--blocking-lever",
        type=str,
        default=counter_seed,
        choices=list(valid_levers),
        help="influence blocking: which of the four published interventions the "
        "budget buys. counter_seed = seed a competing cascade (Budak, CLDAG, RPS, "
        "NIE); node_block = delete nodes (SandIMIN, Xie); edge_block = cut arcs "
        "(Kimura); weight_block = reduce arc probabilities (DiffIM's continuous "
        f"relaxation) (default: {counter_seed}).",
    )
    parser.add_argument(
        "--tie-break",
        type=str,
        default=auto_dominance,
        choices=list(tie_break_choices),
        help="influence blocking: which cascade wins a node both reach on the same "
        "step. Crosses generation, training and evaluation, so it is one value; auto "
        "resolves to each dynamics' own founding paper, positive under IC (Budak) "
        f"and negative under LT (He) (default: {auto_dominance}).",
    )
    parser.add_argument(
        "--positive-prob",
        type=str,
        default=shared_positive_prob,
        help="influence blocking: the blocker's per-edge transmission probability. "
        "'shared' is COICM (one probability per edge, information-independent); a "
        "float is MCICM, and 1.0 is Budak's high-effectiveness property, the case "
        f"his Theorem 4.2 proves submodular (default: {shared_positive_prob}).",
    )
    parser.add_argument(
        "--detection-delay",
        type=int,
        default=0,
        help="influence blocking: Budak's r, the rumour is detected r steps late "
        "and anything the blocker emits before then is dropped. The axis that makes "
        "the first-mover advantage measurable (default: 0).",
    )
    parser.add_argument(
        "--negative-selectors",
        type=str,
        nargs="+",
        default=["random", "degree", "pagerank"],
        choices=list(spine_algorithms),
        help="influence blocking, data stage: how each episode's S_N is chosen, the "
        "attacker model, a second experimental axis IM does not have "
        "(default: random degree pagerank).",
    )
    parser.add_argument(
        "--blocker-selectors",
        type=str,
        nargs="+",
        default=list(blocking_selectors),
        choices=list(blocking_selectors),
        help="influence blocking, data stage: how each episode's t=0 blocker set is "
        "chosen. `none` leaves the rumour unopposed and is the reference every "
        f"prevented-influence number divides by (default: {' '.join(blocking_selectors)}).",
    )
    parser.add_argument(
        "--epi-lever",
        type=str,
        default=vaccinate,
        choices=list(valid_epidemic_levers),
        help="epidemic control: which of the four interventions the budget buys. "
        "vaccinate = the node is immune, leaves the graph and is never counted "
        "(NetShield, DAVA, Pastor-Satorras & Vespignani, Cohen); quarantine = the "
        "node is ISOLATED but stays in the graph and stays counted, which is §8.2 "
        "trap 7's 'recovered is not removed'; edge_cut = cut arcs (Kimura, NetMelt, "
        "Van Mieghem); contact_reduce = scale arc probabilities down (social "
        "distancing, the lever NDlib's compartmental models cannot express) "
        f"(default: {vaccinate}).",
    )
    parser.add_argument(
        "--contact-reduction",
        type=float,
        default=default_contact_reduction,
        help="epidemic control: the multiplier --epi-lever contact_reduce writes on "
        "each arc it spends budget on. 0 is a full cut through the weight channel "
        "and is directly comparable to edge_cut at the same k, which isolates "
        f"graded-vs-all-or-nothing as its own axis (default: {default_contact_reduction}).",
    )
    parser.add_argument(
        "--epi-beta",
        type=float,
        default=1.0,
        help="epidemic control: multiplier on the graph's own per-arc probability, "
        "so beta_uv = clip(scale * p(u->v)). 1.0 leaves it at the weighted-cascade "
        "value; the literature's scalar-beta regime is --prob-model uniform "
        "--uniform-p <beta> with this at 1.0. Crosses all three stages and lands in "
        "metadata.json, because §8.2 trap 2 records that a table which fixes beta "
        "without stating it is comparable only to itself (default: 1.0).",
    )
    parser.add_argument(
        "--epi-gamma",
        type=float,
        default=0.3,
        help="epidemic control: rate of LEAVING I, recovery under SIR/SEIR, "
        "return-to-susceptible under SIS. One parameter for both, because the "
        "lambda1 * beta / delta < 1 threshold uses one. 1.0 under SIR reproduces IC "
        "exactly, which is the cheapest correctness check this task has "
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
        "the endemic prevalence is time-averaged. SIS has NO terminal state, so "
        "final size is undefined there and this is the metric that replaces it "
        f"(§8.2 trap 6) (default: {default_burn_in}).",
    )
    parser.add_argument(
        "--outbreak-selectors",
        type=str,
        nargs="+",
        default=["random", "degree", "pagerank"],
        choices=list(spine_algorithms),
        help="epidemic control, data stage: how each episode's index cases are "
        "chosen: the outbreak model, a second experimental axis a seeding task "
        "does not have (default: random degree pagerank).",
    )
    parser.add_argument(
        "--immunizer-selectors",
        type=str,
        nargs="+",
        default=list(blocking_selectors),
        choices=list(blocking_selectors),
        help="epidemic control, data stage: how each episode's t=0 dose allocation "
        "is chosen. `none` leaves the outbreak unprotected and is the reference "
        f"every prevented-infections number divides by (default: {' '.join(blocking_selectors)}).",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        nargs="*",
        default=None,
        choices=(
            list(algorithm_names)
            + list(adaptive_algorithm_names)
            + list(blocking_algorithm_names)
            + list(dismantling_algorithm_names)
            + list(immunization_algorithm_names)
            + list(localization_algorithm_names)
            + list(reconstruction_algorithm_names)
            + list(prediction_algorithm_names)
            + [
                f"external:{name}"
                for name, spec in external_baselines.items()
                if spec.kind != discovery_kind
            ]
            + [
                f"discovery:{name}"
                for name, spec in external_baselines.items()
                if spec.kind == discovery_kind
            ]
            + ["all", "all-classical", "all-external", "all-discovery"]
        ),
        metavar="NAME",
        help="baselines to run: a static IM algorithm, a per-round adaptive "
        "policy, an influence blocker, a network dismantler, an epidemic "
        "immunizer, a source localizer, a trajectory decoder, a popularity "
        "predictor "
        "(all condition 1), 'external:<name>' for a published repo (condition 7), "
        "'discovery:<name>' for a published LLM algorithm-discovery system run at "
        "its own defaults with our simulator as fitness (condition 9), or the "
        "aliases 'all' / 'all-classical' / 'all-external' / 'all-discovery'. 'all' "
        "includes only external baselines already installed and never the "
        "discovery systems. Unset = the task registry's own pool "
        f"(default for influence_maximization: {' '.join(default_baselines)}).",
    )
    parser.add_argument(
        "--baseline-timeout",
        type=int,
        default=3600,
        help="seconds before an external baseline subprocess is killed (default: 3600).",
    )
    parser.add_argument(
        "--arms",
        type=str,
        nargs="*",
        default=None,
        help="conditions 2-6: 'routing', or '<method>_<mode>[@<evaluator>]' with "
        f"evaluator in {valid_evaluators}; 'native' = the real simulator at "
        f"--native-mc-runs episodes per candidate. 'adaptive_<mode>@<evaluator>' "
        f"is the multi-round policy for adaptive IM. Extra 'baseline:<algorithm>' "
        f"entries are allowed too (default: the task registry's arms, else "
        f"{' '.join(default_arms)}).",
    )
    parser.add_argument(
        "--native-mc-runs",
        type=int,
        default=native_mc_runs_default,
        help="real episodes per candidate for a '@native' arm; 1 keeps it honestly "
        f"model-free (default: {native_mc_runs_default}).",
    )
    parser.add_argument(
        "--budget-pcts",
        type=float,
        nargs="+",
        default=None,
        help="seed budgets as percent of nodes, one run per value. Unset = the "
        "task registry's own sweep, which is a single 10-percent point for source "
        "localization because k there is a property of the instance rather than a "
        "choice (default: 1 5 10 20).",
    )
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
        "on unmodified, and the number every table reports. Selecting and reporting "
        "on the same episodes proves nothing about amortization (default: test).",
    )
    parser.add_argument(
        "--sl-instances",
        type=int,
        default=20,
        help="source localization: labelled episodes per split. Every candidate "
        "program pays this many executions (default: 20).",
    )
    parser.add_argument(
        "--sl-observation",
        type=str,
        default="marginal",
        choices=list(valid_observations),
        help="source localization: which y the program sees. marginal is the "
        "MC-averaged P(infected) and is strictly MORE informative than the "
        "published protocol's; binary is a single realized draw and is the column "
        "comparable to the published tables (default: marginal).",
    )
    parser.add_argument(
        "--sl-budget-mode",
        type=str,
        default=episode_budget,
        choices=list(valid_budget_modes),
        help="source localization: where k comes from. episode = the instance's own "
        "source count (the published given-k convention); sweep = the pipeline's k, "
        "with the pool filtered to that source fraction, which is how the "
        f"source-fraction axis is run (default: {episode_budget}).",
    )
    parser.add_argument(
        "--sl-source-tolerance",
        type=float,
        default=0.5,
        help="source localization: relative band around k that --sl-budget-mode "
        "sweep keeps an episode in (default: 0.5).",
    )
    parser.add_argument(
        "--sl-transfer-from",
        type=str,
        default=None,
        help="source localization: run the winning program named by ANOTHER run's "
        "results JSON, unmodified, on this dataset. This is the graph axis of the "
        "amortization claim, and a comparison no per-instance method can enter: "
        "SL-VAE has no artifact to transfer (default: None).",
    )
    parser.add_argument(
        "--cr-setting",
        type=str,
        default=partial_times,
        choices=list(valid_settings),
        help="cascade reconstruction: which of the four observation regimes is "
        "masked. partial_times = a subsample of the infected set WITH activation "
        "times (the ordered-Steiner regime); partial_nodes = the same subsample "
        "with times withheld; final_snapshot = the terminal state only (DITTO's "
        "DASH, the hardest published formulation and the one worth leading with); "
        "hidden_nodes = partial_times plus nodes deleted from the graph. These are "
        "four separate PROTOCOLS and their rows are never pooled "
        f"(default: {partial_times}).",
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
        help="cascade reconstruction: fraction of non-source nodes DELETED from the "
        "graph under --cr-setting hidden_nodes (default: "
        f"{default_hidden_rate}).",
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
        "REQUIREMENT rather than a taste: the node set is nearly free, so a search "
        "rewarded mostly on Event F1 discovers the tree contributes nothing to its "
        "score and converges on decoders that never attempt it. 0 acknowledges a "
        "node-only protocol explicitly and is the only value that runs on data "
        "without a transmission edge (default: 0.6).",
    )
    # Cascade prediction. The whole point of this task is that the data is REAL, so
    # the flags that shape it are protocol rather than tuning: research/
    # cascade_prediction.md 5.7 lists five independent incompatibilities between
    # published tables and four of them are set here.
    parser.add_argument(
        "--cp-observation",
        type=int,
        default=0,
        help="cascade prediction: observation window t_o, in the CORPUS's own time "
        "unit (seconds for the social corpora, DAYS for APS). 0 takes the corpus's "
        "own first published window: Weibo 1800 (0.5 h), Twitter 86400 (1 d), APS "
        "1095 (3 y). 8.2 pairs TWO windows per corpus deliberately, and a "
        "single-window result is not publishable in this literature, so sweep it "
        "(default: 0).",
    )
    parser.add_argument(
        "--cp-horizon",
        type=int,
        default=0,
        help="cascade prediction: prediction horizon t_p, same units. 0 takes the "
        "corpus's own: Weibo 86400 (24 h), Twitter 2764800 (32 d), APS 7305 (20 y) "
        "(default: 0).",
    )
    parser.add_argument(
        "--cp-step",
        type=int,
        default=0,
        help="cascade prediction: corpus time units per replayed timestep. 0 derives "
        "it from the OBSERVATION window so the prefix always has ~4 bins, which is "
        "the minimum that supports the feature this literature says dominates (the "
        "adoption rate in the SECOND HALF of the window) (default: 0).",
    )
    parser.add_argument(
        "--cp-min-size",
        type=int,
        default=10,
        help="cascade prediction: drop cascades with fewer than this many "
        "participants INSIDE the observation window. CasFlow and CasFT use 10, "
        "CoupledGNN 5, SEISMIC and Mishra 50. 8.4: dropping small cascades removes "
        "the hardest and most numerous cases and inflates every metric, so this is "
        "recorded in metadata.json and reported (default: 10).",
    )
    parser.add_argument(
        "--cp-truncate",
        type=int,
        default=100,
        help="cascade prediction: keep only the first this-many participants of any "
        "cascade. CasFlow uses 100; a method that exploits long tails cannot show it "
        "under this rule, which is why it is reported. 0 disables (default: 100).",
    )
    parser.add_argument(
        "--cp-split",
        type=str,
        default=chronological_split,
        choices=list(valid_splits),
        help="cascade prediction: how the corpus is cut into train/val/test. "
        "`chronological` is CasTemp's leak-free protocol: contiguous bins of equal "
        "publication-time duration, dropping any cascade whose prediction window "
        "crosses a boundary. `random` is the field's own 70/15/15 over cascades, and "
        "8.3 shows it LEAKS: two 2021-24 SOTA models fell below a plain MLP once it "
        f"was fixed. Run both; the gap is the experiment (default: {chronological_split}).",
    )
    parser.add_argument(
        "--cp-target",
        type=str,
        default=increment_target,
        choices=list(valid_targets),
        help="cascade prediction: which quantity the table is about. `increment` is "
        "CasFlow's own label (P(t_p) - P(t_o)); `total` is CasFT's Eq. 26. 5.7 "
        f"difference 3: they share a symbol and are not the same quantity "
        f"(default: {increment_target}).",
    )
    parser.add_argument(
        "--cp-graph",
        type=str,
        default="native",
        choices=list(valid_graphs),
        help="cascade prediction: which graph the transitions run over. `native` is "
        "the corpus loader's own choice: the real published FRIENDSHIP network for "
        "digg_cascades, the union of observed propagation ties for everything else, "
        "which is all those corpora have. `paths` forces the union everywhere; on "
        "Digg that is a real second experiment (the vote log records no parent, so "
        "the union is a union of STARS) and elsewhere it is a no-op the log says so "
        "(default: native).",
    )
    parser.add_argument(
        "--cp-max-cascades",
        type=int,
        default=0,
        help="cascade prediction: cap on replayed cascades, subsampled in --seed. "
        "0 = all of them (default: 0).",
    )
    parser.add_argument(
        "--cp-max-nodes",
        type=int,
        default=0,
        help="cascade prediction: keep only the busiest this-many participants. "
        "CoupledGNN's own move (1.78M users down to 23,681), and what makes a "
        "6.7M-node corpus runnable. 0 = all of them (default: 0).",
    )
    parser.add_argument(
        "--cp-select-split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="cascade prediction: cascades the outer loop optimizes on "
        "(default: train).",
    )
    parser.add_argument(
        "--cp-eval-split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="cascade prediction: HELD-OUT cascades the winner is re-run on "
        "unmodified, and the number every table reports (default: test).",
    )
    parser.add_argument(
        "--cp-instances",
        type=int,
        default=40,
        help="cascade prediction: cascades one reward evaluation sweeps over, per "
        "split (default: 40).",
    )
    parser.add_argument(
        "--cp-observation-steps",
        type=int,
        default=0,
        help="cascade prediction: replayed timesteps the predictor sees. 0 reads it "
        "off the dataset's own `observed` block (default: 0).",
    )
    parser.add_argument(
        "--cp-metric",
        type=str,
        default=default_prediction_metric,
        choices=list(valid_prediction_metrics),
        help="cascade prediction: which error the reward IS. All of them MINIMIZE. "
        "`msle` is CasFlow's own code (log2, clamped at 1, no offset); "
        "`msle_offset` is CasFT's stated log2(P+1); `msle_natural` is CTCP's loss. "
        "5.7 difference 4 records that those variants are printed under one name in "
        f"one published table (default: {default_prediction_metric}).",
    )
    parser.add_argument(
        "--cp-forecast-samples",
        type=int,
        default=default_forecast_samples,
        help="cascade prediction: unrolls averaged inside one forecast_marginals "
        "call. An @monte_carlo arm pays steps x samples x mc_runs real episodes per "
        f"call and a @world_model arm pays steps x samples matmuls "
        f"(default: {default_forecast_samples}).",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=None,
        help="absolute seed budgets, overriding --budget-pcts (default: None).",
    )
    parser.add_argument(
        "--evaluator",
        type=str,
        default="oracle",
        choices=list(valid_evaluators),
        help="default inner-loop evaluator for arms that do not name one with "
        "@<evaluator> (default: oracle).",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-5.6-sol",
        help="gateway model name for the coding agent (default: gpt-5.6-sol).",
    )
    parser.add_argument(
        "--llm-models",
        type=str,
        nargs="+",
        default=None,
        help="fan every LLM-driven arm (conditions 2-6) out across these models, "
        "one result row per (arm, model), named <arm>+<model>; baselines and the "
        "gradient/decode arms run once regardless (default: None).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="LLM sampling temperature (default: None).",
    )
    parser.add_argument(
        "--diffusion-model",
        type=str,
        default="IC",
        choices=["IC", "LT"] + list(epidemic_dynamics),
        help="dynamics used downstream of generation. SIR/SIS/SEIR select the "
        "compartmental simulator and the compartment head, and must match "
        "--gen-models (default: IC).",
    )
    parser.add_argument(
        "--horizon", type=int, default=10, help="agent rollout horizon (default: 10)."
    )
    parser.add_argument(
        "--outer-iters",
        type=int,
        default=20,
        help="outer-loop iterations per LLM arm (default: 20).",
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=3,
        help="windows for the windowed method (default: 3).",
    )
    parser.add_argument(
        "--mc-runs",
        type=int,
        default=200,
        help="Monte Carlo runs for the referee replay (default: 200).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="world-model/oracle rollout samples (default: 200).",
    )
    parser.add_argument(
        "--allowed-ops",
        type=str,
        nargs="+",
        default=None,
        choices=list(valid_action_ops),
        help="action ops the strategies may emit. Unset = the task registry's own set, which is add_node remove_node for influence maximization and remove_node for critical node detection (default: None).",
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
        help="USD per 1M prompt tokens, for the cost column. The lab gateway "
        "fronts Pro subscriptions and bills nothing per token, so there is no rate "
        "to assume: tokens are always counted, cost stays null unless both price "
        "flags are given (default: None).",
    )
    parser.add_argument(
        "--llm-price-out",
        type=float,
        default=None,
        help="USD per 1M completion tokens; see --llm-price-in (default: None).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="replay each winning strategy on Monte Carlo for a fidelity check (default: False).",
    )
    parser.add_argument(
        "--credit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="per-action counterfactual credit for each arm; skipped with a notice on @monte_carlo arms, where it would cost (k+1) real rollouts per action. --no-credit turns it off (default: True).",
    )
    parser.add_argument(
        "--graph-id",
        type=str,
        default=None,
        help="graph id inside the store to evaluate on (default: the first).",
    )

    args = parser.parse_args()

    # Fail before any stage runs, with the runnable list in the message rather
    # than an argparse choices dump
    task = require_runnable(args.task)

    # The task registry is the default so a containment task cannot be generated
    # with maximization semantics by forgetting a flag; --remove-semantics still
    # overrides for ablations
    if args.remove_semantics is None:
        args.remove_semantics = task.remove_semantics

    if args.dataset not in synthetic_families and args.num_graphs != 1:
        raise ValueError(
            f"--num-graphs {args.num_graphs} only applies to synthetic families "
            f"{synthetic_families}; {args.dataset!r} is a single real graph"
        )

    config = PipelineConfig(
        dataset=args.dataset,
        task=args.task,
        run=args.run,
        num_graphs=args.num_graphs,
        syn_nodes=args.syn_nodes,
        er_p=args.er_p,
        ba_m=args.ba_m,
        ws_k=args.ws_k,
        ws_p=args.ws_p,
        sbm_blocks=args.sbm_blocks,
        sbm_p_in=args.sbm_p_in,
        sbm_p_out=args.sbm_p_out,
        plc_m=args.plc_m,
        plc_p=args.plc_p,
        kron_variant=args.kron_variant,
        gen_models=tuple(args.gen_models),
        gen_algorithms=tuple(args.gen_algorithms),
        gen_action_ops=(
            None if args.gen_action_ops is None else tuple(args.gen_action_ops)
        ),
        trace_parents=args.trace_parents,
        remove_semantics=args.remove_semantics,
        split_mode=args.split_mode,
        prob_model=args.prob_model,
        uniform_p=args.uniform_p,
        budget_pct_range=tuple(args.budget_pct_range),
        rollouts=args.rollouts,
        gen_horizon=args.gen_horizon,
        inject_p=args.inject_p,
        cf_prob=args.cf_prob,
        cf_branches=args.cf_branches,
        mc_marginals=args.mc_marginals,
        split=tuple(args.split),
        wm_model=args.wm_model,
        head=args.head,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        gcnii_alpha=args.gcnii_alpha,
        gcnii_lamda=args.gcnii_lamda,
        dropout=args.dropout,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        pos_weight=args.pos_weight,
        patience=args.patience,
        plan_demo=not args.no_plan_demo,
        hide_edge_weights=args.hide_edge_weights,
        plan_graphs=args.plan_graphs,
        plan_budget_k=args.plan_budget_k,
        plan_budget_graphs=args.plan_budget_graphs,
        plan_budget_horizon=args.plan_budget_horizon,
        ood_policies=tuple(args.ood_policies),
        action_encoding=args.action_encoding,
        baselines=None if args.baselines is None else tuple(args.baselines),
        arms=None if args.arms is None else tuple(args.arms),
        budget_pcts=None if args.budget_pcts is None else tuple(args.budget_pcts),
        budgets=tuple(args.budgets) if args.budgets else None,
        evaluator=args.evaluator,
        native_mc_runs=args.native_mc_runs,
        llm_model=args.llm_model,
        llm_models=(None if args.llm_models is None else tuple(args.llm_models)),
        temperature=args.temperature,
        diffusion_model=args.diffusion_model,
        horizon=args.horizon,
        outer_iters=args.outer_iters,
        windows=args.windows,
        mc_runs=args.mc_runs,
        n_samples=args.n_samples,
        allowed_ops=(
            None if args.allowed_ops is None else tuple(args.allowed_ops)
        ),
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
        negative_selectors=tuple(args.negative_selectors),
        blocker_selectors=tuple(args.blocker_selectors),
        epi_lever=args.epi_lever,
        contact_reduction=args.contact_reduction,
        epi_beta=args.epi_beta,
        epi_gamma=args.epi_gamma,
        epi_alpha=args.epi_alpha,
        epi_burn_in=args.epi_burn_in,
        outbreak_selectors=tuple(args.outbreak_selectors),
        immunizer_selectors=tuple(args.immunizer_selectors),
        sl_select_split=args.sl_select_split,
        sl_eval_split=args.sl_eval_split,
        sl_instances=args.sl_instances,
        sl_observation=args.sl_observation,
        sl_budget_mode=args.sl_budget_mode,
        sl_source_tolerance=args.sl_source_tolerance,
        sl_transfer_from=args.sl_transfer_from,
        cr_setting=args.cr_setting,
        cr_observation_rate=args.cr_observation_rate,
        cr_hidden_rate=args.cr_hidden_rate,
        cr_instances=args.cr_instances,
        cr_select_split=args.cr_select_split,
        cr_eval_split=args.cr_eval_split,
        cr_tree_weight=args.cr_tree_weight,
        cp_observation=args.cp_observation,
        cp_horizon=args.cp_horizon,
        cp_step=args.cp_step,
        cp_min_size=args.cp_min_size,
        cp_truncate=args.cp_truncate,
        cp_split=args.cp_split,
        cp_target=args.cp_target,
        cp_graph=args.cp_graph,
        cp_max_cascades=args.cp_max_cascades,
        cp_max_nodes=args.cp_max_nodes,
        cp_select_split=args.cp_select_split,
        cp_eval_split=args.cp_eval_split,
        cp_instances=args.cp_instances,
        cp_observation_steps=args.cp_observation_steps,
        cp_metric=args.cp_metric,
        cp_forecast_samples=args.cp_forecast_samples,
        allow_mc_algorithms=args.allow_mc_algorithms,
        strategy_timeout=args.strategy_timeout,
        llm_price_in=args.llm_price_in,
        llm_price_out=args.llm_price_out,
        compare=args.compare,
        credit=args.credit,
        graph_id=args.graph_id,
        wm_results_json=args.wm_results_json,
        seed=args.seed,
        device=args.device,
        start_stage=args.start_stage,
        end_stage=args.end_stage,
        skip_stages=tuple(args.skip_stages),
        force=args.force,
        baseline_timeout=args.baseline_timeout,
        results_root=args.results_root,
    )

    if args.print_resources:
        gpu, reason = needs_gpu(config)
        print(f"[resources] {'GPU' if gpu else 'CPU-only'}: {reason}")
        print(f"gpu={int(gpu)}")
    else:
        run_pipeline(config)
