"""
One command, whole experiment: generate data -> train the world model -> run every
coding-agent arm at every budget -> plot -> write
results/<task>/<dataset>/<run>/report.md.

Every stage writes its artifacts before the next one starts, so a crashed or killed
run resumes exactly where it stopped (finished work is detected on disk and skipped).

All six baseline conditions in one sweep (the default arm set). Every arm carries
its own evaluator, so conditions 3-6 differ in exactly one thing — the inner-loop
feedback — and land in one report table:

python -m pipeline.run --dataset ba --num-graphs 40 --syn-nodes 100 \
    --budget-pcts 1 5 10 20 --compare \
    --llm-model gpt-5.6-terra --outer-iters 5

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

from baselines.registry import (
    available_baselines,
    external_baselines,
    runnable_baselines,
)
from baselines.run_baseline import BaselineError, run_external_baseline, seed_script
from coding_agent import executor
from coding_agent.run import ExperimentConfig, run_experiment
from data.generate_wm_data import (
    GenConfig,
    default_budget_pct_range,
    run_generation,
    synthetic_families,
)
from coding_agent.tools.library_api import algorithm_names
from coding_agent.types import GraphInfo
from data.wm_simulator import valid_action_ops, valid_remove_semantics
from pipeline.conditions import (
    Arm,
    condition_names,
    ground_truth_reward,
    default_arms,
    default_baselines,
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
from world_model.wm_data import load_graph_store
from world_model.wm_model import backbones

stages = ("data", "train", "agent", "plots", "report")
native_mc_runs_default = 1


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
    gen_models: tuple = ("IC", "LT")
    gen_algorithms: tuple = ("random", "degree", "pagerank", "betweenness")
    gen_action_ops: tuple = ("add_node", "remove_node")
    # Crosses all three stages: it decides how the data is generated, how the
    # head's T_exo is built, and what the agent is told remove_node does. One
    # value so they cannot disagree; defaulted from the task registry.
    remove_semantics: str | None = None
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
    head: str = "structured_residual"
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
    wm_results_json: str | None = None
    # agent stage
    baselines: tuple = default_baselines
    arms: tuple = default_arms
    budget_pcts: tuple | None = (1.0, 5.0, 10.0, 20.0)
    budgets: tuple | None = None
    evaluator: str = "oracle"
    native_mc_runs: int = native_mc_runs_default
    llm_model: str = "gpt-5.6-terra"
    temperature: float | None = None
    diffusion_model: str = "IC"
    horizon: int = 10
    outer_iters: int = 5
    windows: int = 3
    mc_runs: int = 200
    n_samples: int = 50
    allowed_ops: tuple = ("add_node", "remove_node")
    allow_mc_algorithms: bool = False
    strategy_timeout: float = executor.strategy_timeout_seconds
    # USD per 1M tokens for the cost line; None -> tokens counted, cost null
    llm_price_in: float | None = None
    llm_price_out: float | None = None
    compare: bool = False
    credit: bool = False
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


def expand_baselines(names: tuple, task: str) -> list[str]:
    """
    Resolve --baselines into arm specs.

    Accepts library algorithm names (condition 1), `external:<name>` for a
    published repo (condition 7), and the aliases `all`, `all-classical`,
    `all-external`. `all` deliberately expands external baselines to only those
    actually installed, so a fresh checkout does not fail on missing repos.
    """
    specs = []

    for name in names:
        if name == "all-classical":
            specs += [f"baseline:{algorithm}" for algorithm in default_baselines]
        elif name == "all-external":
            specs += [
                f"external:{baseline}"
                for baseline in available_baselines(task=task)
            ]
        elif name == "all":
            specs += [f"baseline:{algorithm}" for algorithm in default_baselines]
            installed = runnable_baselines(task=task)
            specs += [f"external:{baseline}" for baseline in installed]

            skipped = sorted(set(available_baselines(task=task)) - set(installed))
            if skipped:
                print(
                    f"[pipeline] --baselines all: skipping not-installed external "
                    f"baselines {skipped} (run: python -m baselines.setup_baselines --all)"
                )
        elif name.startswith("external:"):
            specs.append(name)
        else:
            specs.append(f"baseline:{name}")

    return specs


def build_arms(config: PipelineConfig) -> list[Arm]:
    """Classical pool + external published baselines + the named conditions."""
    specs = expand_baselines(config.baselines, config.task)
    specs += list(config.arms)

    arms = [parse_arm(spec, default_evaluator=config.evaluator) for spec in specs]

    duplicates = {arm.name for arm in arms if [a.name for a in arms].count(arm.name) > 1}
    if duplicates:
        raise ValueError(
            f"arm names collide, so their results would overwrite each other: "
            f"{sorted(duplicates)}"
        )

    return arms


def budget_points(config: PipelineConfig) -> list[tuple[str, float | None, int]]:
    """(label, budget_pct, budget) for each point of the sweep; pct wins if both are set."""
    if config.budgets is not None:
        return [(budget_label(None, k), None, int(k)) for k in config.budgets]

    return [(budget_label(pct, 0), float(pct), 0) for pct in config.budget_pcts]


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
    every C++ external repo are pure CPU — only training f_theta, rolling it
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
        action_ops=list(config.gen_action_ops),
        remove_semantics=config.remove_semantics,
        weight_lo=0.0,
        weight_hi=1.0,
        cf_prob=config.cf_prob,
        cf_branches=config.cf_branches,
        split=tuple(config.split),
        seed=config.seed,
        mc_marginals=config.mc_marginals,
        out_dir=str(layout.data_dir),
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
        )
    )


def _skip_path(layout: Layout, label: str, arm) -> Path:
    return layout.baselines_dir / label / f"{arm.name}{skip_marker_suffix}"


def _clear_skip(layout: Layout, label: str, arm) -> None:
    _skip_path(layout, label, arm).unlink(missing_ok=True)


def _record_skip(layout: Layout, label: str, arm, reason: str) -> None:
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


def stage_agent(config: PipelineConfig, layout: Layout) -> list[dict]:
    wm_results = (
        Path(config.wm_results_json)
        if config.wm_results_json is not None
        else layout.wm_results(config.wm_model, config.diffusion_model)
    )
    arms = build_arms(config)

    if needs_world_model(arms) and not wm_results.exists():
        raise FileNotFoundError(
            f"arms {[arm.spec for arm in arms if arm.evaluator == 'world_model']} "
            f"evaluate against the world model but {wm_results} is missing; run the "
            f"train stage or drop those arms"
        )

    # Rewards from different evaluators are not comparable, so a multi-condition
    # sweep is only readable once every arm has been replayed on the same referee
    if not config.compare and len({arm.evaluator for arm in arms}) > 1:
        print(
            "[agent] WARNING: arms span multiple evaluators without --compare, so "
            "their rewards are measured by different judges and cannot be compared. "
            "Re-run with --compare for a valid table."
        )

    points = budget_points(config)
    completed = []
    reused = failed = 0

    # One bar over the whole (budget x arm) grid — this stage dominates wall
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
                f"(condition {arm.condition} — {arm.condition_name}, "
                f"evaluator={evaluator}, mc_runs={mc_runs})"
            )
            arm_start = time.perf_counter()

            # An external repo only hands back a seed set; wrapping it as a
            # canned Strategy routes it through the identical scoring path
            external_seeds = None
            if arm.external is not None:
                graph = _load_pipeline_graph(layout, config)
                resolved = budget or max(
                    1, round(graph.num_nodes * budget_pct / 100)
                )
                try:
                    external_seeds = run_external_baseline(
                        arm.external,
                        graph,
                        resolved,
                        config.diffusion_model,
                        work_dir=layout.baselines_dir / "_runs" / arm.external / label,
                        timeout=config.baseline_timeout,
                    )
                except BaselineError as error:
                    tqdm.write(f"[agent] {label}/{arm.name}: SKIPPED — {error}")
                    _record_skip(layout, label, arm, str(error))
                    failed += 1
                    progress_bar.update(1)
                    continue

            experiment = ExperimentConfig(
                method=arm.method,
                strategy_mode=arm.strategy_mode,
                evaluator=evaluator,
                model=config.llm_model,
                temperature=config.temperature,
                diffusion_model=config.diffusion_model,
                remove_semantics=config.remove_semantics,
                budget=budget or 5,
                budget_pct=budget_pct,
                horizon=config.horizon,
                windows=config.windows,
                # Conditions 1 and 2 have no refinement loop: one pass is the arm
                outer_iters=1 if not arm.is_agent else config.outer_iters,
                mc_runs=mc_runs,
                referee_mc_runs=config.mc_runs,
                n_samples=config.n_samples,
                seed=config.seed,
                device=config.device,
                data_dir=str(layout.data_dir),
                graph_id=config.graph_id,
                wm_results_json=str(wm_results) if wm_results.exists() else None,
                compare=config.compare,
                credit=config.credit,
                baseline=arm.baseline,
                routing=arm.routing,
                allowed_ops=tuple(config.allowed_ops),
                allow_mc_algorithms=config.allow_mc_algorithms,
                strategy_timeout=config.strategy_timeout,
                llm_price_in=config.llm_price_in,
                llm_price_out=config.llm_price_out,
                # --force means redo, so it must not silently resume a killed
                # search from the checkpoint it left behind
                resume=not config.force,
                out_json=str(out_json),
            )

            result = run_experiment(
                experiment,
                canned_script=(
                    seed_script(external_seeds["seeds"]) if external_seeds else None
                ),
            )
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
                    "seeds": external_seeds["seeds"],
                    # Time the external repo itself spent selecting seeds; our
                    # scoring time is in elapsed_seconds as for every other arm
                    "selection_seconds": external_seeds["seconds"],
                }
                result["model"] = f"external:{arm.external}"

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
            tqdm.write(
                f"[agent] {label}/{arm.name}: done in "
                f"{time.perf_counter() - arm_start:.1f}s -> "
                f"spread {ground_truth_reward(result):.2f} "
                f"({100.0 * ground_truth_reward(result) / result['graph']['num_nodes']:.1f}% of N)"
                f"{token_text}"
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
    """Every result JSON on disk — our arms and external baselines together."""
    results = []

    for path in layout.result_globs():
        result = json.loads(path.read_text())
        result.setdefault("arm", path.stem)
        result.setdefault("budget_label", path.parent.name)
        results.append(result)

    return results


def load_skips(layout: Layout) -> list[dict]:
    """The `.skipped.json` markers — what was attempted and did not produce a row."""
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
    path = layout.wm_results(config.wm_model, config.diffusion_model)

    return json.loads(path.read_text()) if path.exists() else None


def _write_manifest(layout: Layout, config: PipelineConfig) -> None:
    """Config + per-stage status, rewritten whenever anything changes."""
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


def run_pipeline(config: PipelineConfig) -> dict:
    pipeline_start = time.perf_counter()

    # Resolved here rather than only in __main__, so a programmatically built
    # config gets the task's semantics instead of leaking None into GenConfig
    if config.remove_semantics is None:
        config.remove_semantics = get_task(config.task).remove_semantics

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
            print(f"[pipeline] stage {stage} FAILED — manifest updated, re-run to resume")
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
        "--gen-models",
        type=str,
        nargs="+",
        default=["IC", "LT"],
        choices=["IC", "LT"],
        help="dynamics to generate transitions for (default: IC LT).",
    )
    parser.add_argument(
        "--gen-algorithms",
        type=str,
        nargs="+",
        default=["random", "degree", "pagerank", "betweenness"],
        help="spine seed selectors driving the episodes (default: random degree pagerank betweenness).",
    )
    parser.add_argument(
        "--gen-action-ops",
        type=str,
        nargs="+",
        default=["add_node", "remove_node"],
        choices=list(valid_action_ops),
        help="action ops injected during generation (default: add_node remove_node).",
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
        default="structured_residual",
        choices=["linear", "structured", "structured_residual"],
        help="world-model output head (default: structured_residual).",
    )
    # The checkpoint is located via `ckpt_dir` inside this JSON, and the
    # architecture flags are read from its `config` block — so this one path
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

    # Agent stage
    parser.add_argument(
        "--baselines",
        type=str,
        nargs="*",
        default=list(default_baselines),
        choices=(
            list(algorithm_names)
            + [f"external:{name}" for name in external_baselines]
            + ["all", "all-classical", "all-external"]
        ),
        metavar="NAME",
        help="baselines to run: a library algorithm (condition 1), "
        "'external:<name>' for a published repo (condition 7), or the aliases "
        "'all' / 'all-classical' / 'all-external'. 'all' includes only external "
        f"baselines already installed (default: {' '.join(default_baselines)}).",
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
        default=list(default_arms),
        help="conditions 2-6: 'routing', or '<method>_<mode>[@<evaluator>]' with "
        f"evaluator in {valid_evaluators}; 'native' = the real simulator at "
        f"--native-mc-runs episodes per candidate. Extra 'baseline:<algorithm>' "
        f"entries are allowed too (default: {' '.join(default_arms)}).",
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
        default=[1.0, 5.0, 10.0, 20.0],
        help="seed budgets as percent of nodes, one run per value (default: 1 5 10 20).",
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
        default="gpt-5.6-terra",
        help="gateway model name for the coding agent (default: gpt-5.6-terra).",
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
        choices=["IC", "LT"],
        help="dynamics used downstream of generation (default: IC).",
    )
    parser.add_argument(
        "--horizon", type=int, default=10, help="agent rollout horizon (default: 10)."
    )
    parser.add_argument(
        "--outer-iters",
        type=int,
        default=5,
        help="outer-loop iterations per LLM arm (default: 5).",
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
        default=50,
        help="world-model/oracle rollout samples (default: 50).",
    )
    parser.add_argument(
        "--allowed-ops",
        type=str,
        nargs="+",
        default=["add_node", "remove_node"],
        choices=list(valid_action_ops),
        help="action ops the strategies may emit (default: add_node remove_node).",
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
        action="store_true",
        help="per-action counterfactual credit for each arm (default: False).",
    )
    parser.add_argument(
        "--graph-id",
        type=str,
        default=None,
        help="graph id inside the store to evaluate on (default: the first).",
    )

    args = parser.parse_args()

    # Fail before any stage runs: a planned task has no head, no simulator, or
    # no data, and the message names which
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
        gen_models=tuple(args.gen_models),
        gen_algorithms=tuple(args.gen_algorithms),
        gen_action_ops=tuple(args.gen_action_ops),
        remove_semantics=args.remove_semantics,
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
        plan_graphs=args.plan_graphs,
        baselines=tuple(args.baselines),
        arms=tuple(args.arms),
        budget_pcts=tuple(args.budget_pcts),
        budgets=tuple(args.budgets) if args.budgets else None,
        evaluator=args.evaluator,
        native_mc_runs=args.native_mc_runs,
        llm_model=args.llm_model,
        temperature=args.temperature,
        diffusion_model=args.diffusion_model,
        horizon=args.horizon,
        outer_iters=args.outer_iters,
        windows=args.windows,
        mc_runs=args.mc_runs,
        n_samples=args.n_samples,
        allowed_ops=tuple(args.allowed_ops),
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
