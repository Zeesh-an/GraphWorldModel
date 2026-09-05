"""
Autoregressive action-conditioned world-model training (teacher-forced one-step).

Checkpoints and the results JSON land next to the data by default:
<run>/data -> <run>/world_model/{wm_<model>_<dm>.pt, <model>_<dm>.json}, where
<run> is results/<task>/<dataset>/<run> (see pipeline/layout.py)

python -m world_model.train_wm \
    --data-dir results/ba40/data --diffusion-model IC \
    --model sage --head structured --hidden-dim 64 --n-layers 3 \
    --epochs 400 --lr 1e-3 --weight-decay 5e-4 --batch-size 32 \
    --pos-weight off --patience 50 --seed 42 \
    --device cuda --plan-demo

The CLI defaults are the pipeline's (pipeline/run.py): sage / structured /
400 epochs / batch 32 / pos_weight off / patience 50.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.wm_competitive import CompetitiveConfig, auto_dominance, shared_positive_prob
from data.wm_epidemic import EpidemicConfig, default_burn_in
from data.wm_simulator import epidemic_dynamics, spent, valid_remove_semantics
from world_model.checkpoint import (
    ModelSpec,
    checkpoint_format,
    load_checkpoint,
    save_checkpoint,
)
from world_model.wm_action_eval import action_conditioning_report
from world_model.wm_data import (
    TransitionDataset,
    basic_encoding,
    channels_for,
    collate_transitions,
    dataset_is_competitive,
    dataset_is_epidemic,
    epidemic_rates,
    num_input_channels,
    valid_action_encodings,
)
from world_model.wm_eval import (
    blocking_regret_multi,
    competitive_rollout_ensemble,
    epidemic_rollout_ensemble,
    evaluate_one_step,
    immunization_regret_multi,
    planning_regret_budget_multi,
    planning_regret_multi,
    planning_split_test,
    rollout_ensemble,
    valid_planning_splits,
)
from world_model.wm_model import WorldModel, backbones
from world_model.wm_policies import resolve as resolve_policies

pos_weight_min = 1.0
pos_weight_max = 50.0
# Flush the training curve to disk this often so a killed job keeps its history
history_flush_epochs = 5


@dataclass()
class TrainConfig:
    """Field names are the results-JSON `config` schema that world_model_env reads back."""

    data_dir: str
    diffusion_model: str = "IC"
    model: str = "sage"
    head: str = "structured"
    # Must match the dataset's; cross-checked against metadata.json below
    remove_semantics: str = spent
    # Two-cascade (influence blocking) training: 8 input channels, 4 targets, and a
    # competitive head. None = read from the dataset's own metadata, which is what
    # keeps a 6-channel head from being fit on a competitive dataset and quietly
    # reporting the negative cascade alone. The two dynamics parameters below are
    # likewise defaulted FROM the data, because a head whose tie-break disagrees with
    # the simulator that made the targets is fit against a transition that never
    # happened (research/influence_blocking.md §8.4).
    competitive: bool | None = None
    tie_break: str = auto_dominance
    positive_prob: float | None = None
    # Compartmental (epidemic control) training: 9 input channels, 5 targets, and
    # the compartment head. None = read from the dataset's own metadata, for the
    # same reason `competitive` is: a 6-channel head fed compartmental features
    # would silently fit the attack set alone and lose recovery entirely, which is
    # the one thing this task exists to test. The three rates are likewise defaulted
    # FROM the data, because a head whose gamma disagrees with the simulator that
    # made the targets is fit against a transition that never happened.
    epidemic: bool | None = None
    beta_scale: float = 1.0
    gamma: float | None = None
    alpha: float | None = None
    burn_in: float = default_burn_in
    # Feed ones instead of p(u->v) to the encoder and the head: the online/bandit
    # information state, and the ablation for "our IC heads see the true w"
    hide_edge_weights: bool = False
    # `basic` = the original 6 channels; `typed` adds 3 that split CH_EDGE by op,
    # so add_edge and remove_edge at the same endpoints stop producing identical X
    action_encoding: str = basic_encoding
    # Off-policy rollout policies (world_model/wm_policies.py). Empty = skip.
    ood_policies: tuple = ()
    # k-seed full-horizon planning; 0 disables (it costs real simulator episodes)
    plan_budget_k: int = 0
    plan_budget_graphs: int = 3
    plan_budget_horizon: int = 20
    # Which graphs the planning evaluators score. `test` (the default) restricts
    # them to held-out graphs; `legacy` reproduces the historical
    # `list(store)[:n]`, which mixes splits and is not a held-out number.
    planning_split: str = planning_split_test
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
    seed: int = 42
    device: str = "cpu"
    ckpt_dir: str | None = None
    results: str | None = None
    plan_demo: bool = False
    plan_graphs: int = 5


def _clamp_pos_weight(value: float) -> float:
    return float(min(max(value, pos_weight_min), pos_weight_max))


def compute_pos_weight(
    dataset: TransitionDataset, device: torch.device
) -> list[torch.Tensor]:
    """
    One positive-class weight per target column: 2 single-cascade, 4 competitive.

    Diffusion changes are sparse (few infected/frontier nodes per step), so a plain
    BCE would collapse to predicting all zeros. Blocking makes that WORSE rather than
    better: §5.3 shows blocking 50 YouTube nodes prevents 0.29% of the spread, so the
    positive class in the columns that matter is rarer than it is under IM, which is
    why the calibration columns are worth re-checking rather than assuming they carry
    over from the seeding tasks.
    """
    columns = dataset[0]["y"].shape[1]
    positives = [0.0] * columns
    total = 0

    for index in range(len(dataset)):
        targets = dataset[index]["y"]
        total += targets.shape[0]

        for column in range(columns):
            positives[column] += targets[:, column].sum().item()

    # Passed into BCEWithLogitsLoss, it up-weights the rare positive class so the
    # model is pushed to actually predict the new infections
    return [
        torch.tensor(
            [_clamp_pos_weight((total - positive) / max(positive, 1.0))], device=device
        )
        for positive in positives
    ]


def resolve_paths(config: TrainConfig) -> TrainConfig:
    """Default the checkpoint dir and results JSON to siblings of the data dir."""
    if config.ckpt_dir is None:
        config.ckpt_dir = str(Path(config.data_dir).resolve().parent / "world_model")

    if config.results is None:
        config.results = str(
            Path(config.ckpt_dir) / f"{config.model}_{config.diffusion_model}.json"
        )

    return config


def check_remove_semantics(config: TrainConfig) -> None:
    """
    Refuse to train a head whose T_exo disagrees with the data that produced the
    targets. The mismatch is otherwise silent: the model simply never fits the
    removal transitions, and the containment numbers come out biased with no
    error anywhere.
    """
    metadata_path = Path(config.data_dir) / "metadata.json"
    if not metadata_path.exists():
        return

    # Datasets generated before --remove-semantics existed are all `spent`
    dataset_semantics = json.loads(metadata_path.read_text())["config"].get(
        "remove_semantics", spent
    )

    if dataset_semantics != config.remove_semantics:
        raise ValueError(
            f"--remove-semantics {config.remove_semantics!r} does not match the "
            f"dataset at {config.data_dir}, which was generated with "
            f"{dataset_semantics!r} (see {metadata_path}). Regenerate the data or "
            f"pass --remove-semantics {dataset_semantics}."
        )


def resolve_competitive(config: TrainConfig) -> TrainConfig:
    """
    Fill `competitive` / `tie_break` / `positive_prob` from the dataset that made the targets.

    Defaulted from the data rather than from a flag for the same reason
    `check_remove_semantics` refuses a mismatch: a competitive head whose tie-break
    or `p_L` disagrees with the simulator is fit against a transition that never
    happened, and nothing about the loss curve would say so.
    """
    metadata_path = Path(config.data_dir) / "metadata.json"
    dataset_competitive = dataset_is_competitive(config.data_dir)

    if config.competitive is None:
        config.competitive = dataset_competitive
    elif config.competitive != dataset_competitive:
        raise ValueError(
            f"--competitive {config.competitive} does not match the dataset at "
            f"{config.data_dir}, which is competitive={dataset_competitive}. A "
            f"competitive dataset carries 8 channels and 4 targets; regenerate with "
            f"data/generate_wm_data.py --competitive or drop the flag."
        )

    if not config.competitive or not metadata_path.exists():
        return config

    competitive = json.loads(metadata_path.read_text()).get("competitive", {})
    resolved = competitive.get(config.diffusion_model)

    if resolved is None:
        raise ValueError(
            f"the dataset at {config.data_dir} carries no competitive metadata for "
            f"--diffusion-model {config.diffusion_model}; it was generated for "
            f"{sorted(key for key in competitive if key != 'negative_pct')}"
        )

    config.tie_break = resolved["tie_break"]
    config.positive_prob = (
        None
        if resolved["positive_prob"] == "shared"
        else float(resolved["positive_prob"])
    )
    print(
        f"[train] competitive: {resolved['competitive_model']}, "
        f"tie_break={config.tie_break}, positive_prob={resolved['positive_prob']}"
    )

    return config


def resolve_epidemic(config: TrainConfig) -> TrainConfig:
    """
    Fill `epidemic` / `beta_scale` / `gamma` / `alpha` from the dataset that made the targets.

    Same rule and same reason as `resolve_competitive`: the rates are not a choice
    at training time, they are a property of the transitions on disk. Getting them
    from a flag would let `--head structured_oracle` pin its matrix to a recovery
    rate the simulator never used and report a fidelity number that means nothing.
    """
    dataset_epidemic = dataset_is_epidemic(config.data_dir)

    if config.epidemic is None:
        config.epidemic = dataset_epidemic
    elif config.epidemic != dataset_epidemic:
        raise ValueError(
            f"--epidemic {config.epidemic} does not match the dataset at "
            f"{config.data_dir}, which is epidemic={dataset_epidemic}. A "
            f"compartmental dataset carries 9 channels and 5 targets; regenerate "
            f"with data/generate_wm_data.py --models SIR (or SIS / SEIR)."
        )

    if not config.epidemic:
        return config

    resolved = epidemic_rates(config.data_dir, config.diffusion_model)
    config.beta_scale = float(resolved["beta_scale"])
    config.gamma = float(resolved["gamma"])
    config.alpha = None if resolved["alpha"] is None else float(resolved["alpha"])
    config.burn_in = float(resolved["burn_in"])
    print(
        f"[train] compartmental: {'/'.join(resolved['compartments'])}, "
        f"beta_scale={config.beta_scale}, gamma={config.gamma}, "
        f"alpha={config.alpha}"
    )

    return config


def dataset_split_mode(config: TrainConfig) -> str:
    """
    How the dataset under `--data-dir` assigned train/val/test.

    Datasets generated before `--split-mode` existed carry no marker, and all of
    them used the per-episode draw, so that is what an absent field means.
    """
    metadata_path = Path(config.data_dir) / "metadata.json"

    if not metadata_path.exists():
        return "unknown"

    metadata = json.loads(metadata_path.read_text())

    return metadata.get("split_mode") or metadata["config"].get(
        "split_mode", "episode_random"
    )


def check_split_mode(config: TrainConfig) -> str:
    """
    Warn — loudly — when training on a dataset whose split leaks graphs.

    A warning rather than a refusal, deliberately: the legacy mode is kept so
    that pre-2026-08-22 numbers can be reproduced, and a run whose PURPOSE is
    that reproduction must still be possible. What must not happen is a new
    number being quoted without anyone knowing which regime produced it, which is
    why the mode also lands in the checkpoint's train_meta and the results JSON.
    """
    mode = dataset_split_mode(config)

    if mode == "graph_disjoint":
        return mode

    print(
        f"[warn] dataset at {config.data_dir} was generated with "
        f"split_mode={mode!r}: train/val/test were drawn PER EPISODE, so the same "
        f"graph appears on both sides of the split. Test metrics from this run "
        f"are optimistic by an unknown amount and are not comparable to a "
        f"graph-disjoint run. Regenerate with `--split-mode graph_disjoint` "
        f"unless this is a single-graph real dataset (which cannot be split by "
        f"graph at all: its metrics are in-graph by construction) or a "
        f"deliberate reproduction of a legacy result."
    )

    return mode


def check_hide_edge_weights(config: TrainConfig) -> None:
    """
    Both anchored heads read w directly, so masking it is not an ablation of them
    but a corruption of them: structured_residual anchors q on logit(w), which at
    w=1 pins every edge at q~1 and saturates the rollout, and structured_oracle
    IS q = w. Only the plain structured head (and linear) learn q from scratch.
    """
    if not config.hide_edge_weights:
        return

    if config.head in ("structured_residual", "structured_oracle"):
        raise ValueError(
            f"--hide-edge-weights is incompatible with --head {config.head}: that "
            f"head reads the true transmission probability directly, so masking it "
            f"to ones does not hide information, it feeds a wrong anchor. Use "
            f"--head structured for the w-hidden (bandit information state) run."
        )


def train_world_model(config: TrainConfig) -> dict:
    config = resolve_paths(config)
    check_remove_semantics(config)
    config = resolve_competitive(config)
    config = resolve_epidemic(config)
    check_split_mode(config)

    if config.action_encoding != basic_encoding and (
        config.competitive or config.epidemic
    ):
        raise ValueError(
            f"--action-encoding {config.action_encoding!r} applies only to the "
            f"single-cascade 6-channel layout; competitive and compartmental "
            f"datasets have their own fixed layouts"
        )
    check_hide_edge_weights(config)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device(config.device)
    diffusion_model = config.diffusion_model

    layout = dict(
        competitive=config.competitive,
        epidemic=config.epidemic,
        action_encoding=config.action_encoding,
    )
    train_dataset = TransitionDataset(
        config.data_dir, diffusion_model, "train", **layout
    )
    validation_dataset = TransitionDataset(
        config.data_dir, diffusion_model, "val", **layout
    )
    test_dataset = TransitionDataset(config.data_dir, diffusion_model, "test", **layout)

    for split, dataset in (
        ("train", train_dataset),
        ("val", validation_dataset),
        ("test", test_dataset),
    ):
        # An empty split does not fail loudly on its own: binary_f1 returns 1.0
        # when there is nothing to predict, so an empty val split keeps the
        # epoch-0 weights and an empty test split reports delta_f1 = 1.0
        if len(dataset) == 0:
            raise ValueError(
                f"the {split} split of {config.data_dir} holds no "
                f"{diffusion_model} transitions (split_mode="
                f"{dataset_split_mode(config)!r}); a graph-disjoint split needs "
                f"at least three graphs, and a single-graph dataset must be "
                f"generated with --split-mode episode_random"
            )

    if config.pos_weight == "auto" and config.head != "linear":
        print(
            f"[warn] --pos-weight auto with --head {config.head}: the positive "
            f"weight inflates the per-edge transmission q globally and collapses "
            f"one-step accuracy on a structured head; use --pos-weight off"
        )

    collate_fn = partial(
        collate_transitions,
        diffusion_model=diffusion_model,
        device=device,
        hide_edge_weights=config.hide_edge_weights,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    backbone_kwargs = {
        "n_heads": config.n_heads,
        "ffn_dim": config.ffn_dim,
        "alpha": config.gcnii_alpha,
        "lamda": config.gcnii_lamda,
    }

    model_in_channels, model_out_channels = channels_for(
        config.competitive, config.epidemic
    )
    model = WorldModel(
        config.model,
        in_channels=(
            model_in_channels
            if config.competitive or config.epidemic
            else num_input_channels(config.action_encoding)
        ),
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        dropout=config.dropout,
        head_type=config.head,
        diffusion_model=diffusion_model,
        remove_semantics=config.remove_semantics,
        competitive=config.competitive,
        tie_break=config.tie_break,
        positive_prob=config.positive_prob,
        epidemic=config.epidemic,
        epi_beta=config.beta_scale,
        epi_gamma=config.gamma,
        epi_alpha=config.alpha,
        **backbone_kwargs,
    ).to(device)

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    weights = (
        compute_pos_weight(train_dataset, device)
        if config.pos_weight == "auto"
        else [None] * model_out_channels
    )
    # One BCE term per target column: 2 single-cascade, 4 competitive (§2.5)
    losses = [nn.BCEWithLogitsLoss(pos_weight=weight) for weight in weights]

    os.makedirs(config.ckpt_dir, exist_ok=True)
    checkpoint_path = (
        Path(config.ckpt_dir) / f"wm_{config.model}_{diffusion_model}.pt"
    )
    # Built once: the checkpoint describes itself, so nothing downstream has to
    # find this run's results JSON to know what architecture the weights belong to
    model_spec = ModelSpec.from_train_config(config)
    checkpoint_meta = {
        "data_dir": str(config.data_dir),
        "split_mode": dataset_split_mode(config),
        "seed": config.seed,
        "epochs": config.epochs,
        "patience": config.patience,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "pos_weight": config.pos_weight,
        "selection_metric": "val delta_f1",
    }
    best_delta_f1, epochs_since_best = -1.0, 0
    train_start = time.perf_counter()

    # Per-epoch curve for the training-diagnostics plot, flushed as it grows
    history = []
    history_path = Path(config.ckpt_dir) / f"history_{config.model}_{diffusion_model}.json"

    print(
        f"[train] {config.model}/{config.head} on {diffusion_model}: "
        f"{len(train_dataset)} train / {len(validation_dataset)} val / "
        f"{len(test_dataset)} test transitions, up to {config.epochs} epochs "
        f"(patience {config.patience}) on {device}"
    )

    progress_bar = tqdm(range(config.epochs), desc=f"train {config.model}/{diffusion_model}")
    for epoch in progress_bar:
        model.train()
        batch_bar = tqdm(train_dataloader, desc=f"  epoch {epoch}", leave=False)
        epoch_loss, n_batches = 0.0, 0

        for batch in batch_bar:
            optimizer.zero_grad()

            logits = model(batch["X"], batch["graph"])
            loss = sum(
                term(logits[:, column], batch["y"][:, column])
                for column, term in enumerate(losses)
            )

            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            epoch_loss += loss_value
            n_batches += 1
            batch_bar.set_postfix(loss=f"{loss_value:.6f}")

        val_metrics = evaluate_one_step(
            model,
            validation_dataset,
            diffusion_model,
            device,
            hide_edge_weights=config.hide_edge_weights,
            competitive=config.competitive,
            epidemic=config.epidemic,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / max(n_batches, 1),
                "val_delta_f1": val_metrics["delta_f1"],
                "val_new_infection_f1": val_metrics["new_infection_f1"],
            }
        )

        # Persist the curve as it is produced: a crash or a SLURM timeout at
        # epoch 380/400 otherwise loses every epoch of history
        if epoch % history_flush_epochs == 0 or epoch == config.epochs - 1:
            Path(history_path).write_text(json.dumps(history, indent=2))

        progress_bar.set_postfix(
            val_delta_f1=f"{val_metrics['delta_f1']:.4f}",
            best=f"{max(best_delta_f1, val_metrics['delta_f1']):.4f}",
        )

        if val_metrics["delta_f1"] > best_delta_f1:
            best_delta_f1, epochs_since_best = val_metrics["delta_f1"], 0
            save_checkpoint(
                model,
                checkpoint_path,
                model_spec,
                train_meta={**checkpoint_meta, "best_val_delta_f1": best_delta_f1,
                            "epoch": epoch},
            )
        else:
            epochs_since_best += 1
            if epochs_since_best >= config.patience:
                print(
                    f"[early-stop] epoch {epoch}, best val delta_f1={best_delta_f1:.4f}"
                )
                break

    train_seconds = time.perf_counter() - train_start
    print(f"[train] total training time: {train_seconds:.1f}s")

    model, _, _ = load_checkpoint(checkpoint_path, device=device)
    model.train(False)
    results = {
        "config": vars(config),
        # Which split regime produced these numbers. Recorded at the top level so
        # an aggregator can refuse to pool a leaky run with a clean one.
        "split_mode": checkpoint_meta["split_mode"],
        "checkpoint_format": checkpoint_format,
        "train_seconds": train_seconds,
        "best_val_delta_f1": best_delta_f1,
        "history": history,
        "test": evaluate_one_step(
            model,
            test_dataset,
            diffusion_model,
            device,
            hide_edge_weights=config.hide_edge_weights,
            competitive=config.competitive,
            epidemic=config.epidemic,
        ),
    }

    # IC is monotone, so "changed" == "newly infected" and delta_f1 is ALGEBRAICALLY
    # identical to new_infection_f1. Reporting both as if they were two pieces of
    # evidence overstates the one-step result; say so in the JSON rather than in a
    # footnote nobody reads.
    results["test"]["delta_f1_is_new_infection_f1"] = bool(
        abs(results["test"]["delta_f1"] - results["test"]["new_infection_f1"]) < 1e-12
    )

    # Both halves fork on the same flag, and both forks measure the SAME quantity,
    # the negative cascade, so the two tasks' rollout and planning numbers sit in
    # one report column without being silently different things
    competitive_config = (
        CompetitiveConfig(
            tie_break=config.tie_break,
            positive_prob=(
                shared_positive_prob
                if config.positive_prob is None
                else config.positive_prob
            ),
            remove_semantics=config.remove_semantics,
        )
        if config.competitive
        else None
    )

    epidemic_config = (
        EpidemicConfig(
            beta_scale=config.beta_scale,
            gamma=config.gamma,
            alpha=config.alpha if config.alpha is not None else 0.5,
            remove_semantics=config.remove_semantics,
            burn_in=config.burn_in,
        )
        if config.epidemic
        else None
    )

    rollout_kwargs = dict(
        seed=config.seed,
        remove_semantics=config.remove_semantics,
        hide_edge_weights=config.hide_edge_weights,
        action_encoding=config.action_encoding,
    )

    if config.epidemic:
        results["rollout"] = epidemic_rollout_ensemble(
            model,
            config.data_dir,
            diffusion_model,
            train_dataset.store,
            device,
            "test",
            seed=config.seed,
            hide_edge_weights=config.hide_edge_weights,
            config=epidemic_config,
        )
    elif config.competitive:
        results["rollout"] = competitive_rollout_ensemble(
            model,
            config.data_dir,
            diffusion_model,
            train_dataset.store,
            device,
            "test",
            seed=config.seed,
            hide_edge_weights=config.hide_edge_weights,
            config=competitive_config,
        )
    else:
        results["rollout"] = rollout_ensemble(
            model,
            config.data_dir,
            diffusion_model,
            train_dataset.store,
            device,
            "test",
            **rollout_kwargs,
        )

    # The action-conditioning suite and the off-policy rollouts read the plain
    # 6-channel records; the two special layouts carry their own eval branches
    if not config.competitive and not config.epidemic:
        # Does the model use the action at all, and does it use it CORRECTLY?
        results["action_conditioning"] = action_conditioning_report(
            model,
            test_dataset,
            diffusion_model,
            device,
            hide_edge_weights=config.hide_edge_weights,
            seed=config.seed,
        )

        # Fidelity under action distributions the training data never contained.
        # The recorded-sequence rollout above is on-policy by construction; this is
        # the number that speaks to using f_theta as a coding agent's inner loop.
        if config.ood_policies:
            results["rollout_ood"] = {
                name: rollout_ensemble(
                    model,
                    config.data_dir,
                    diffusion_model,
                    train_dataset.store,
                    device,
                    "test",
                    action_policy=policy,
                    **rollout_kwargs,
                )
                for name, policy in resolve_policies(
                    list(config.ood_policies)
                ).items()
            }

    if config.plan_demo:
        planner = (
            immunization_regret_multi
            if config.epidemic
            else blocking_regret_multi
            if config.competitive
            else planning_regret_multi
        )
        extra = (
            {"config": epidemic_config}
            if config.epidemic
            else {"config": competitive_config}
            if config.competitive
            else {
                "remove_semantics": config.remove_semantics,
                "action_encoding": config.action_encoding,
            }
        )
        results["planning"] = planner(
            model,
            train_dataset.store,
            diffusion_model,
            device,
            n_graphs=config.plan_graphs,
            seed=config.seed,
            # Held-out graphs only. Without out_dir the selector cannot see split
            # membership and falls back to `legacy`, which is not a test metric.
            out_dir=config.data_dir,
            planning_split=config.planning_split,
            hide_edge_weights=config.hide_edge_weights,
            **extra,
        )

    # The k-seed full-horizon planner greedily builds an IM seed set, which only
    # means something on the plain single-cascade layout
    if config.plan_budget_k > 0 and not config.competitive and not config.epidemic:
        results["planning_budget"] = planning_regret_budget_multi(
            model,
            train_dataset.store,
            diffusion_model,
            device,
            n_graphs=config.plan_budget_graphs,
            seed=config.seed,
            k=config.plan_budget_k,
            horizon=config.plan_budget_horizon,
            out_dir=config.data_dir,
            planning_split=config.planning_split,
            hide_edge_weights=config.hide_edge_weights,
            remove_semantics=config.remove_semantics,
            action_encoding=config.action_encoding,
        )

    os.makedirs(Path(config.results).parent, exist_ok=True)
    Path(config.results).write_text(json.dumps(results, indent=2, default=str))
    print(f"[train] checkpoint -> {checkpoint_path}")
    print(f"[train] results -> {config.results}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Action-conditioned world-model training"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="generated dataset directory (default: required).",
    )
    parser.add_argument(
        "--diffusion-model",
        type=str,
        default="IC",
        choices=["IC", "LT"] + list(epidemic_dynamics),
        help="dynamics to train on. SIR/SIS/SEIR select the compartment head and "
        "read their rates back from the dataset's metadata (default: IC).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sage",
        choices=list(backbones),
        help="encoder backbone (default: sage).",
    )
    parser.add_argument(
        "--head",
        type=str,
        default="structured",
        choices=["linear", "structured", "structured_residual"],
        help="output head type; structured_residual anchors IC transmission on the true edge prob and learns only a correction (default: structured).",
    )
    parser.add_argument(
        "--remove-semantics",
        type=str,
        default=spent,
        choices=list(valid_remove_semantics),
        help="what remove_node means; must match the dataset's, which is checked "
        "against its metadata.json (default: spent).",
    )
    parser.add_argument(
        "--hide-edge-weights",
        action="store_true",
        help="feed ones instead of the true IC transmission probability to the "
        "encoder and head: the online/bandit information state, and the ablation "
        "for the IC heads otherwise seeing w. Requires --head structured or "
        "linear (default: False).",
    )
    parser.add_argument(
        "--action-encoding",
        type=str,
        default=basic_encoding,
        choices=list(valid_action_encodings),
        help="action feature encoding; `typed` adds 3 channels splitting CH_EDGE "
        "by op, so add_edge and remove_edge on the same endpoints stop producing "
        "identical X. Changes in_channels, so a checkpoint is not portable across "
        "the two (default: basic).",
    )
    parser.add_argument(
        "--ood-policies",
        type=str,
        nargs="*",
        default=[],
        help="off-policy rollout tests, e.g. `degree_seed null`. The recorded "
        "action sequence is on-policy by construction; these measure fidelity "
        "under the action distribution an agent would actually propose "
        "(default: none).",
    )
    parser.add_argument(
        "--plan-budget-k",
        type=int,
        default=0,
        help="k-seed full-horizon planning regret vs greedy-MC; 0 disables. This "
        "is the IM problem as posed, unlike the single-step --plan-demo "
        "(default: 0).",
    )
    parser.add_argument(
        "--planning-split",
        type=str,
        default=planning_split_test,
        choices=list(valid_planning_splits),
        help="which graphs the planning evaluators score. `test` (default) uses "
        "held-out graphs only and reports train_overlap/val_overlap so that can "
        "be checked; `legacy` reproduces the old list(store)[:n] selection, which "
        "mixes splits and is NOT a held-out number (default: {planning_split_test}).",
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
        "--hidden-dim",
        type=int,
        default=64,
        help="hidden dimension (default: 64).",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=3,
        help="number of encoder layers (default: 3).",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=4,
        help="attention heads for GAT/GT (default: 4).",
    )
    parser.add_argument(
        "--ffn-dim",
        type=int,
        default=128,
        help="transformer feed-forward dimension (default: 128).",
    )
    parser.add_argument(
        "--gcnii-alpha",
        type=float,
        default=0.1,
        help="GCNII initial residual alpha (default: 0.1).",
    )
    parser.add_argument(
        "--gcnii-lamda",
        type=float,
        default=0.5,
        help="GCNII identity mapping lambda (default: 0.5).",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="dropout probability (default: 0.1)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=400,
        help="maximum training epochs (default: 400).",
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
        "--batch-size",
        type=int,
        default=32,
        help="transition batch size (default: 32).",
    )
    parser.add_argument(
        "--pos-weight",
        type=str,
        default="off",
        choices=["auto", "off"],
        help="positive-class weighting mode; `auto` is for the linear head only, "
        "it inflates a structured head's q globally (default: off).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="early-stop patience in epochs (default: 50).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)."
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="torch device string (default: cpu)."
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="checkpoint directory (default: <data-dir>/../world_model).",
    )
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="results JSON path (default: <ckpt-dir>/<model>_<diffusion-model>.json).",
    )
    parser.add_argument(
        "--plan-demo",
        action="store_true",
        help="run planning demo after training (default: False).",
    )
    parser.add_argument(
        "--plan-graphs",
        type=int,
        default=5,
        help="graphs used for planning demo (default: 5).",
    )

    args = parser.parse_args()
    results = train_world_model(TrainConfig(**vars(args)))

    print(
        json.dumps(
            {key: results[key] for key in ("test", "rollout") if key in results},
            indent=2,
            default=str,
        )
    )
