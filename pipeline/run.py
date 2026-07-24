"""
One command, whole experiment: generate data -> train the world model -> run every
coding-agent arm at every budget -> plot -> write results/<tag>/report.md.

Every stage writes its artifacts before the next one starts, so a crashed or killed
run resumes exactly where it stopped (finished work is detected on disk and skipped).

Full sweep on a synthetic family:

python -m pipeline.run --dataset ba --tag ba40 \
    --num-graphs 40 --syn-nodes 100 \
    --budget-pcts 1 5 10 20 --evaluator oracle \
    --llm-model claude-sonnet-5 --outer-iters 5 --compare

Oracle only, no world model (skips the train stage automatically):

python -m pipeline.run --dataset netscience --tag netscience --evaluator oracle

Resume just the reporting half of a finished run:

python -m pipeline.run --dataset ba --tag ba40 --start-stage plots

Re-run one stage from scratch:

python -m pipeline.run --dataset ba --tag ba40 --start-stage agent --end-stage agent --force
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from dotenv import load_dotenv

from coding_agent.run import ExperimentConfig, run_experiment
from data.generate_wm_data import (
    GenConfig,
    default_budget_pct_range,
    run_generation,
    synthetic_families,
)
from data.wm_simulator import valid_action_ops
from pipeline.layout import Layout, budget_label
from pipeline.plots import build_plots
from pipeline.report import write_report
from world_model.train_wm import TrainConfig, train_world_model
from world_model.wm_model import backbones

stages = ("data", "train", "agent", "plots", "report")

# The six-condition baseline taxonomy, minus the evaluator axis (--evaluator)
default_arms = (
    "baseline:degree_discount",
    "baseline:celf_pp",
    "routing",
    "one_shot_free",
    "evolve_scored",
)
valid_methods = ("one_shot", "per_step", "windowed", "evolve")
valid_modes = ("free", "scored")


@dataclass
class Arm:
    name: str
    method: str
    strategy_mode: str
    baseline: str | None = None
    routing: bool = False


@dataclass
class PipelineConfig:
    dataset: str
    tag: str
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
    dropout: float = 0.1
    epochs: int = 400
    lr: float = 1e-3
    weight_decay: float = 5e-4
    batch_size: int = 32
    pos_weight: str = "off"
    patience: int = 50
    plan_demo: bool = True
    plan_graphs: int = 5
    # agent stage
    arms: tuple = default_arms
    budget_pcts: tuple | None = (1.0, 5.0, 10.0, 20.0)
    budgets: tuple | None = None
    evaluator: str = "oracle"
    llm_model: str = "claude-sonnet-5"
    temperature: float | None = None
    diffusion_model: str = "IC"
    horizon: int = 10
    outer_iters: int = 5
    windows: int = 3
    mc_runs: int = 200
    n_samples: int = 50
    allowed_ops: tuple = ("add_node", "remove_node")
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
    results_root: str = "results"
    stage_status: dict = field(default_factory=dict)


def parse_arm(name: str) -> Arm:
    if name.startswith("baseline:"):
        algorithm = name.split(":", 1)[1]
        return Arm(
            name=f"baseline_{algorithm}",
            method="one_shot",
            strategy_mode="free",
            baseline=algorithm,
        )

    if name == "routing":
        return Arm(name="routing", method="one_shot", strategy_mode="free", routing=True)

    method, _, mode = name.rpartition("_")
    if method not in valid_methods or mode not in valid_modes:
        raise ValueError(
            f"unknown arm {name!r}; expected 'baseline:<algorithm>', 'routing', "
            f"or '<method>_<mode>' with method in {valid_methods} and mode in {valid_modes}"
        )

    return Arm(name=name, method=method, strategy_mode=mode)


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

    # The world model is only needed when it is the evaluator
    if config.evaluator != "world_model" and "train" in selected:
        print(f"[pipeline] evaluator={config.evaluator}: skipping the train stage")
        selected.remove("train")

    return [stage for stage in selected if stage not in config.skip_stages]


def stage_data(config: PipelineConfig, layout: Layout) -> dict:
    if layout.data_metadata().exists() and not config.force:
        print(f"[data] reusing {layout.data_dir} (--force to regenerate)")
        return json.loads(layout.data_metadata().read_text())

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
        print(f"[train] reusing {results_path} (--force to retrain)")
        return json.loads(results_path.read_text())

    return train_world_model(
        TrainConfig(
            data_dir=str(layout.data_dir),
            diffusion_model=config.diffusion_model,
            model=config.wm_model,
            head=config.head,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
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


def stage_agent(config: PipelineConfig, layout: Layout) -> list[dict]:
    wm_results = layout.wm_results(config.wm_model, config.diffusion_model)
    if config.evaluator == "world_model" and not wm_results.exists():
        raise FileNotFoundError(
            f"evaluator=world_model needs a trained checkpoint but {wm_results} is "
            f"missing; run the train stage or switch to --evaluator oracle"
        )

    points = budget_points(config)
    arms = [parse_arm(name) for name in config.arms]
    completed = []

    for label, budget_pct, budget in points:
        for arm in arms:
            out_json = layout.agent_result(label, arm.name)

            if out_json.exists() and not config.force:
                print(f"[agent] {label}/{arm.name}: reusing {out_json}")
                completed.append(json.loads(out_json.read_text()))
                continue

            print(f"[agent] {label}/{arm.name}: running")
            experiment = ExperimentConfig(
                method=arm.method,
                strategy_mode=arm.strategy_mode,
                evaluator=config.evaluator,
                model=config.llm_model,
                temperature=config.temperature,
                diffusion_model=config.diffusion_model,
                budget=budget or 5,
                budget_pct=budget_pct,
                horizon=config.horizon,
                windows=config.windows,
                # Classical baselines are deterministic: one pass is the whole arm
                outer_iters=1 if arm.baseline or arm.routing else config.outer_iters,
                mc_runs=config.mc_runs,
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
                out_json=str(out_json),
            )

            result = run_experiment(experiment)
            result["arm"] = arm.name
            result["budget_label"] = label
            out_json.write_text(json.dumps(result, indent=2, default=str))
            completed.append(result)

    return completed


def load_agent_results(layout: Layout) -> list[dict]:
    """Every agent JSON on disk, tagged with its arm and budget label."""
    results = []

    for path in sorted(layout.agent_dir.glob("*/*.json")):
        result = json.loads(path.read_text())
        result.setdefault("arm", path.stem)
        result.setdefault("budget_label", path.parent.name)
        results.append(result)

    return results


def load_wm_results(config: PipelineConfig, layout: Layout) -> dict | None:
    path = layout.wm_results(config.wm_model, config.diffusion_model)

    return json.loads(path.read_text()) if path.exists() else None


def run_pipeline(config: PipelineConfig) -> dict:
    pipeline_start = time.perf_counter()
    layout = Layout(config.tag, root=config.results_root)
    os.makedirs(layout.root, exist_ok=True)

    selected = active_stages(config)
    print(f"[pipeline] tag={config.tag} dataset={config.dataset} -> {layout.root}")
    print(f"[pipeline] stages: {' -> '.join(selected)}")

    for stage in selected:
        stage_start = time.perf_counter()
        print(f"\n{'=' * 72}\n[pipeline] stage: {stage}\n{'=' * 72}")

        if stage == "data":
            stage_data(config, layout)
        elif stage == "train":
            stage_train(config, layout)
        elif stage == "agent":
            stage_agent(config, layout)
        elif stage == "plots":
            figures = build_plots(
                load_agent_results(layout),
                load_wm_results(config, layout),
                layout.plots_dir,
                title_prefix=config.tag,
            )
            print(f"[plots] wrote {len(figures)} figures -> {layout.plots_dir}")
        elif stage == "report":
            write_report(
                config=vars(config),
                layout=layout,
                agent_results=load_agent_results(layout),
                wm_results=load_wm_results(config, layout),
            )
            print(f"[report] -> {layout.report_path}")

        config.stage_status[stage] = {
            "status": "done",
            "seconds": round(time.perf_counter() - stage_start, 1),
        }

        # Manifest is rewritten after each stage so a crash still leaves a record
        layout.manifest_path.write_text(
            json.dumps(
                {"config": vars(config), "stages": config.stage_status},
                indent=2,
                default=str,
            )
        )

    elapsed = time.perf_counter() - pipeline_start
    print(f"\n[pipeline] done in {elapsed:.1f}s -> {layout.root}")

    return {"tag": config.tag, "stages": config.stage_status, "seconds": elapsed}


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
        "--tag",
        type=str,
        default=None,
        help="results/<tag> directory name (default: the dataset name).",
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
        "--results-root",
        type=str,
        default="results",
        help="top-level results directory (default: results).",
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
    parser.add_argument(
        "--hidden-dim", type=int, default=64, help="hidden dimension (default: 64)."
    )
    parser.add_argument(
        "--n-layers", type=int, default=3, help="encoder layers (default: 3)."
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
        "--arms",
        type=str,
        nargs="+",
        default=list(default_arms),
        help="conditions to evaluate: 'baseline:<algorithm>', 'routing', or "
        f"'<method>_<mode>' (default: {' '.join(default_arms)}).",
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
        choices=["world_model", "monte_carlo", "oracle"],
        help="inner-loop evaluator; oracle needs no trained checkpoint (default: oracle).",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="claude-sonnet-5",
        help="gateway model name for the coding agent (default: claude-sonnet-5).",
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

    if args.dataset not in synthetic_families and args.num_graphs != 1:
        raise ValueError(
            f"--num-graphs {args.num_graphs} only applies to synthetic families "
            f"{synthetic_families}; {args.dataset!r} is a single real graph"
        )

    config = PipelineConfig(
        dataset=args.dataset,
        tag=args.tag or args.dataset,
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
        prob_model=args.prob_model,
        uniform_p=args.uniform_p,
        budget_pct_range=tuple(args.budget_pct_range),
        rollouts=args.rollouts,
        gen_horizon=args.gen_horizon,
        inject_p=args.inject_p,
        cf_prob=args.cf_prob,
        cf_branches=args.cf_branches,
        mc_marginals=args.mc_marginals,
        wm_model=args.wm_model,
        head=args.head,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        pos_weight=args.pos_weight,
        patience=args.patience,
        plan_demo=not args.no_plan_demo,
        plan_graphs=args.plan_graphs,
        arms=tuple(args.arms),
        budget_pcts=tuple(args.budget_pcts),
        budgets=tuple(args.budgets) if args.budgets else None,
        evaluator=args.evaluator,
        llm_model=args.llm_model,
        temperature=args.temperature,
        diffusion_model=args.diffusion_model,
        horizon=args.horizon,
        outer_iters=args.outer_iters,
        windows=args.windows,
        mc_runs=args.mc_runs,
        n_samples=args.n_samples,
        allowed_ops=tuple(args.allowed_ops),
        compare=args.compare,
        credit=args.credit,
        graph_id=args.graph_id,
        seed=args.seed,
        device=args.device,
        start_stage=args.start_stage,
        end_stage=args.end_stage,
        skip_stages=tuple(args.skip_stages),
        force=args.force,
        results_root=args.results_root,
    )

    run_pipeline(config)
