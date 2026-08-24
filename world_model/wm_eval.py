"""One-step eval, action-specific metrics, free-running rollout, planning."""

import json
from pathlib import Path
from collections import defaultdict
from typing import Callable
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance

from world_model.wm_data import (
    TransitionDataset,
    apply_edge_ops,
    basic_encoding,
    build_competitive_features,
    build_epidemic_features,
    build_features,
    build_graph_input,
    ch_epi_ever,
    ch_epi_infectious,
    ch_frontier,
    ch_infected,
    collate_transitions,
    edges_to_arrays,
    reconstruct_episode_adjacency,
)
from world_model.wm_metrics import (
    binary_f1,
    brier_score,
    persistence_baseline,
    reliability_bins,
    score_predictions,
)

from data.wm_competitive import CompetitiveConfig, CompetitiveSimulator
from data.wm_epidemic import EpidemicConfig, EpidemicSimulator
from data.wm_simulator import ActionOp, Simulator, State, blocked, spent

seed_upper_bound = 1 << 30

# Decimal places the action-sensitivity comparison rounds predicted probabilities
# to. Two outputs that agree to 1e-3 per node ARE the same prediction; anything
# finer would count floating-point noise as a reaction to the action.
sensitivity_decimals = 3


def _cat_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(arrays) if arrays else np.zeros(0)


# Curves that are padded with zero once an episode ends rather than held: a dead
# epidemic's prevalence IS zero, the same rule the agent environments apply
zero_padded_curves = ("prevalence_model", "prevalence_true")


def _step_curves(per_episode: dict[str, list]) -> dict[str, list[float]]:
    """
    Per-step means across ALL episodes, one list per named accumulator.

    Episodes differ in length, and averaging only over the episodes that reached
    step t makes a cumulative count appear to fall once the large cascades have
    ended. Each episode is padded to the longest one instead: a finished cascade
    holds its final size (the true state does not change after the last step)
    and a finished epidemic's prevalence is zero. This is the convention the
    agent environments' `spread_curve` already follows.
    """
    curves = {}
    for name, episodes in per_episode.items():
        if not episodes:
            curves[f"{name}_curve"] = []
            continue

        length = max(len(episode) for episode in episodes)
        fill = 0.0 if name in zero_padded_curves else None
        padded = np.array(
            [
                episode + [episode[-1] if fill is None else fill] * (length - len(episode))
                for episode in episodes
            ],
            dtype=np.float64,
        )  # shape: (n_episodes, length)
        curves[f"{name}_curve"] = padded.mean(axis=0).tolist()

    return curves


def _reskin_action(dataset: TransitionDataset, index: int, action: list[dict]) -> dict:
    """Rebuild dataset[index] with `action` substituted for the recorded bag.

    The ADJACENCY is deliberately left as reconstructed for the recorded bag. That
    is exact for node ops, which never touch the graph, and it is why
    `wm_action_eval.action_ablation` only ever substitutes node-only bags: swapping
    in an edge op without replaying it would score the model against a graph its
    input claims it does not have, and the resulting "degradation" would measure
    the inconsistency rather than the model's use of the action.
    """
    record, edge_index, weights = dataset.samples[index]
    num_nodes = dataset.store[record["graph_id"]]["num_nodes"]

    swapped = dict(record)
    swapped["action"] = action

    X, y_inf, y_fr = build_features(
        swapped, edge_index, num_nodes, dataset.action_encoding
    )
    targets = np.stack([y_inf, y_fr], axis=1)  # shape: (N, 2)

    # Mirrors TransitionDataset.__getitem__'s plain-layout item exactly; the
    # override path is guarded to plain datasets, so no competitive/epidemic branch
    return {
        "X": torch.from_numpy(X),
        "y": torch.from_numpy(targets),
        "y_inf": torch.from_numpy(targets[:, 0].copy()),
        "y_fr": torch.from_numpy(targets[:, 1].copy()),
        "edge_index": torch.from_numpy(edge_index),
        "edge_weight": torch.from_numpy(weights),
        "num_nodes": num_nodes,
        "record": swapped,
    }


@torch.inference_mode()
def evaluate_one_step(
    model: nn.Module,
    dataset: TransitionDataset,
    diffusion_model: str,
    device: torch.device,
    threshold: float = 0.5,
    hide_edge_weights: bool = False,
    competitive: bool = False,
    epidemic: bool = False,
    action_override: dict | None = None,
) -> dict[str, float]:
    """
    Teacher-forced one-step evaluation over an entire dataset.

    `action_override` maps a dataset index to a replacement action list. It is how
    `wm_action_eval.action_ablation` re-scores the same states under zeroed or
    shuffled actions; leave it None for a normal evaluation.

    Aggregates the full metric suite (score_predictions), action-specific metrics
    (Add-Seed Success, Remove-Frontier Success), action sensitivity (how much the
    model output changes under counterfactual actions at the same state), and the persistence baseline.

    Under `competitive` the headline suite is unchanged and still describes columns
    0-1, which are the NEGATIVE cascade under both layouts: the quantity a blocking
    task is scored on. What changes is the action metric: an `add_node` there seeds
    the POSITIVE cascade, so "did the model flip the target to infected" has to read
    the positive channel or it measures the exact opposite of the intervention. The
    positive cascade's own suite is reported alongside under `pos_*`.

    Under `epidemic` the headline suite is again unchanged and describes columns 0-1,
    which are the ATTACK SET and the INCIDENCE: the quantity a control task is
    scored on. What changes is that `remove_node` is a DOSE rather than a spent
    spreader, so the removal metric reads the infectious channel and is reported as
    `dose_success`; and the three compartment columns get their own accuracy under
    `compartment_acc`, because a head that predicts the attack set well while
    getting `I` wrong has failed at the only thing that makes this task different
    from critical node detection (research/epidemic_control.md §2.4).
    """
    model.eval()
    pos_pred_infected_parts = []
    pos_target_infected_parts = []
    pos_current_infected_parts = []
    block_hits = block_total = 0
    compartment_pred_parts = []
    compartment_target_parts = []
    compartment_prob_parts = []
    pred_infected_parts = []
    pred_frontier_parts = []
    prob_infected_parts = []
    prob_frontier_parts = []
    target_infected_parts = []
    target_frontier_parts = []
    current_infected_parts = []
    current_frontier_parts = []

    # sensitivity_groups[state_key][action_key] = tuple of per-node predicted infected probabilities
    sensitivity_groups = defaultdict(dict)
    add_hits = add_total = remove_hits = remove_total = 0

    for index in range(len(dataset)):
        item = dataset[index]

        if action_override is not None and index in action_override:
            item = _reskin_action(dataset, index, action_override[index])
        batch = collate_transitions(
            [item], diffusion_model, device, hide_edge_weights
        )
        logits = model(batch["X"], batch["graph"])  # (N, 2)
        probs = torch.sigmoid(logits).cpu().numpy()

        pred_infected = (probs[:, 0] > threshold).astype(np.float32)
        pred_infected_parts.append(pred_infected)
        prob_infected_parts.append(probs[:, 0])

        pred_frontier = (probs[:, 1] > threshold).astype(np.float32)
        pred_frontier_parts.append(pred_frontier)
        prob_frontier_parts.append(probs[:, 1])

        target_infected_parts.append(item["y_inf"].numpy())
        target_frontier_parts.append(item["y_fr"].numpy())
        # Under the compartmental layout the attack set is channel 4 and the
        # infectious set channel 2; under IC/LT they are 0 and 1. Both are "the set
        # being scored" and "the current wave", which is what the persistence
        # baseline and the change-F1 both need.
        current_infected_parts.append(
            item["X"][:, ch_epi_ever if epidemic else ch_infected].numpy()
        )
        current_frontier_parts.append(
            item["X"][:, ch_epi_infectious if epidemic else ch_frontier].numpy()
        )

        if epidemic:
            # Columns 2-4 are E / I / R. Scored as one flat 0/1 problem rather than
            # three, because the head composes them from one simplex and a
            # per-column number would hide that E is empty under SIR and SIS.
            compartment_pred_parts.append((probs[:, 2:5] > threshold).astype(np.float32))
            compartment_prob_parts.append(probs[:, 2:5])
            compartment_target_parts.append(item["y"][:, 2:5].numpy())

        if competitive:
            pos_pred_infected_parts.append((probs[:, 2] > threshold).astype(np.float32))
            pos_target_infected_parts.append(item["y"][:, 2].numpy())
            # Channel 2 is the positive cascade's own `infected` under the
            # competitive layout, which is where its persistence baseline reads from
            pos_current_infected_parts.append(item["X"][:, 2].numpy())

        record = item["record"]
        for action_op in record["action"]:
            if action_op["op"] == "add_node":
                if competitive:
                    # A blocker's seed makes the target POSITIVELY active, and the
                    # negative channel is what it must NOT flip
                    block_total += 1
                    block_hits += int(
                        pos_pred_infected_parts[-1][int(action_op["target"])] == 1
                    )
                else:
                    add_total += 1
                    add_hits += int(pred_infected[int(action_op["target"])] == 1)
            elif action_op["op"] == "remove_node":
                remove_total += 1
                remove_hits += int(pred_frontier[int(action_op["target"])] == 0)

        # Group main and cf transitions by (graph, episode, t, state) for sensitivity
        state_key = (
            record["graph_id"],
            record["episode_id"],
            record["t"],
            tuple(record["state"]["infected"]),
        )
        action_key = tuple(
            sorted(
                (
                    action_op["op"],
                    action_op["target"],
                    action_op.get("destination", -1),
                    action_op.get("weight", -1.0),
                )
                for action_op in record["action"]
            )
        )
        # Keyed on the rounded PROBABILITIES, not the thresholded prediction.
        # Under a seeding task the two agree: seeding A flips A itself, a
        # decisive 0 -> 1. Under containment they do not: blocking A rather than B
        # shifts its neighbours' infection probabilities without moving any of them
        # across 0.5, so the binary version reports "the model ignored the action"
        # for a model that reacted correctly. Measured at exactly 0.0 on the first
        # containment dataset, with counterfactual pairs present in the split.
        sensitivity_groups[state_key][action_key] = tuple(
            np.round(probs[:, 0], sensitivity_decimals).tolist()
        )

    # Targets are soft marginals: threshold at 0.5 for the binary F1/accuracy suite,
    # keep the raw probabilities + soft targets for Brier (calibration vs the marginal).
    target_infected = _cat_arrays(target_infected_parts)
    target_frontier = _cat_arrays(target_frontier_parts)
    current_infected = _cat_arrays(current_infected_parts)
    current_frontier = _cat_arrays(current_frontier_parts)
    target_infected_binary = (target_infected > 0.5).astype(np.float32)
    target_frontier_binary = (target_frontier > 0.5).astype(np.float32)

    results = score_predictions(
        _cat_arrays(pred_infected_parts),
        _cat_arrays(pred_frontier_parts),
        target_infected_binary,
        target_frontier_binary,
        current_infected,
        current_frontier,
    )
    results["add_seed_success"] = add_hits / add_total if add_total else float("nan")
    results["remove_frontier_success"] = (
        remove_hits / remove_total if remove_total else float("nan")
    )

    if epidemic:
        # A dose is not a spent spreader: the target must be predicted OUT of the
        # infectious set, which is the same measurement under a different name and
        # a different column
        results["dose_success"] = results["remove_frontier_success"]
        compartment_target = (
            _cat_arrays([part.ravel() for part in compartment_target_parts]) > 0.5
        ).astype(np.float32)
        compartment_pred = _cat_arrays(
            [part.ravel() for part in compartment_pred_parts]
        )
        results["compartment_acc"] = (
            float((compartment_pred.astype(bool) == compartment_target.astype(bool)).mean())
            if compartment_pred.size
            else float("nan")
        )
        results["compartment_f1"] = binary_f1(
            compartment_pred.astype(bool), compartment_target.astype(bool)
        )
        results["brier_compartment"] = brier_score(
            _cat_arrays([part.ravel() for part in compartment_prob_parts]),
            _cat_arrays([part.ravel() for part in compartment_target_parts]),
        )

    # Action sensitivity: mean number of distinct output tuples per (state, multiple actions) group
    distinct_counts = [
        len(set(group.values())) - 1
        for group in sensitivity_groups.values()
        if len(group) >= 2
    ]
    results["action_sensitivity"] = (
        float(np.mean(distinct_counts)) if distinct_counts else 0.0
    )

    results["brier_infected"] = brier_score(
        _cat_arrays(prob_infected_parts), target_infected
    )
    results["brier_frontier"] = brier_score(
        _cat_arrays(prob_frontier_parts), target_frontier
    )
    # Reliability against the simulator's own marginals: Brier alone cannot
    # separate a sharp-but-miscalibrated model from a calibrated-but-vague one
    results["calibration_infected"] = reliability_bins(
        _cat_arrays(prob_infected_parts), target_infected
    )
    results["calibration_frontier"] = reliability_bins(
        _cat_arrays(prob_frontier_parts), target_frontier
    )
    results["ece_infected"] = results["calibration_infected"]["ece"]
    results["ece_frontier"] = results["calibration_frontier"]["ece"]

    results["persistence"] = persistence_baseline(
        target_infected_binary,
        target_frontier_binary,
        current_infected,
        current_frontier,
    )
    results["persistence"]["brier_infected"] = brier_score(
        current_infected, target_infected
    )
    results["persistence"]["brier_frontier"] = brier_score(
        current_frontier, target_frontier
    )

    if competitive:
        # The counter-cascade's own accuracy, reported separately rather than
        # averaged in: the two cascades are not interchangeable and a head that
        # predicts the blocker's spread well while missing the rumour's has failed
        # at the only thing the task scores
        pos_current = _cat_arrays(pos_current_infected_parts)
        pos_target = (_cat_arrays(pos_target_infected_parts) > 0.5).astype(np.float32)
        results["pos_infected_acc"] = float(
            (_cat_arrays(pos_pred_infected_parts).astype(bool) == pos_target.astype(bool)).mean()
        )
        results["pos_new_infection_f1"] = binary_f1(
            _cat_arrays(pos_pred_infected_parts).astype(bool) & ~pos_current.astype(bool),
            pos_target.astype(bool) & ~pos_current.astype(bool),
        )
        results["block_seed_success"] = (
            block_hits / block_total if block_total else float("nan")
        )

    return results


@torch.inference_mode()
def rollout_episodes(
    model: nn.Module,
    out_dir: str,
    diffusion_model: str,
    store: dict[str, dict],
    device: torch.device,
    split: str = "test",
    threshold: float = 0.5,
    hide_edge_weights: bool = False,
    action_encoding: str = basic_encoding,
) -> dict[str, float]:
    """
    Feed the model its own thresholded prediction + recorded action; compare to truth.

    For each episode (main branch only), the model is initialised with the true state
    at t = 0 and then fed its own binary output at each subsequent step.  The recorded
    (ground-truth) action is applied at every step so the model sees the correct intervention signal.

    Three metrics are reported:
    - rollout_newinf_f1: per-step F1 on newly infected nodes only (susceptible mask)
    - rollout_count_mae: per-step |predicted count - true count| of infected nodes
    - rollout_final_f1: F1 between rolled-out final state and true final state
    """
    path = Path(out_dir) / f"transitions_{diffusion_model}_{split}.jsonl"
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]

    # Keep only main-branch records; group by (graph_id, episode_id).
    by_episode = defaultdict(list)
    for record in records:
        if record["branch"] == "main":
            by_episode[(record["graph_id"], record["episode_id"])].append(record)

    model.eval()
    per_step_f1, count_mae, final_f1 = [], [], []

    for (graph_id, _), episode_records in by_episode.items():
        episode_records.sort(key=lambda record: record["t"])
        num_nodes = store[graph_id]["num_nodes"]
        adjacency_map = reconstruct_episode_adjacency(
            episode_records, store[graph_id]["base_edges"]
        )

        # Initialise the autoregressive state from the first recorded true state
        current_infected = set(episode_records[0]["state"]["infected"])
        current_frontier = set(episode_records[0]["state"]["frontier"])

        for record in episode_records:
            edge_index, weights = edges_to_arrays(adjacency_map[(record["t"], "main")])

            # Replace the record's state with the rolled-out state for feature building
            rolled = dict(record)
            rolled["state"] = {
                "infected": sorted(current_infected),
                "frontier": sorted(current_frontier),
            }
            X, _, _ = build_features(rolled, edge_index, num_nodes, action_encoding)
            graph_input = build_graph_input(
                edge_index,
                weights,
                num_nodes,
                diffusion_model,
                device,
                hide_edge_weights,
            )

            probs = (
                torch.sigmoid(model(torch.from_numpy(X).to(device), graph_input))
                .cpu()
                .numpy()
            )
            pred_infected = probs[:, 0] > threshold  # (N,) bool
            pred_frontier = probs[:, 1] > threshold  # (N,) bool

            # Ground-truth next state from the record.
            target_infected = np.zeros(num_nodes, dtype=np.float32)
            target_infected[record["next_state"]["infected"]] = 1.0
            state_infected = np.zeros(num_nodes, dtype=np.float32)
            state_infected[record["state"]["infected"]] = 1.0

            # New-infection F1: evaluate only on susceptible nodes (state_infected == 0)
            susceptible_mask = state_infected == 0
            per_step_f1.append(
                binary_f1(
                    pred_infected & susceptible_mask,
                    target_infected.astype(bool) & susceptible_mask,
                )
            )
            count_mae.append(
                abs(float(pred_infected.sum()) - len(record["next_state"]["infected"]))
            )

            # Advance the autoregressive state.
            current_infected = set(
                int(node) for node in np.nonzero(pred_infected)[0].tolist()
            )
            current_frontier = set(
                int(node) for node in np.nonzero(pred_frontier)[0].tolist()
            )

        # Final-state F1 compares rolled-out endpoint to the true endpoint
        truth_final = np.zeros(num_nodes, dtype=np.float32)
        truth_final[episode_records[-1]["next_state"]["infected"]] = 1.0
        rolled_final = np.zeros(num_nodes, dtype=np.float32)
        rolled_final[list(current_infected)] = 1.0
        final_f1.append(binary_f1(rolled_final, truth_final))

    return {
        "rollout_newinf_f1": float(np.mean(per_step_f1)) if per_step_f1 else 0.0,
        "rollout_count_mae": float(np.mean(count_mae)) if count_mae else 0.0,
        "rollout_final_f1": float(np.mean(final_f1)) if final_f1 else 0.0,
    }


def _action_bag(action_dicts: list[dict]) -> list[ActionOp]:
    """Rebuild an ActionOp bag from a record's serialized action list."""
    return [
        ActionOp(
            op=action_op["op"],
            target=int(action_op["target"]),
            destination=action_op.get("destination"),
            weight=action_op.get("weight"),
        )
        for action_op in action_dicts
    ]


@torch.inference_mode()
def rollout_ensemble(
    model: nn.Module,
    out_dir: str,
    diffusion_model: str,
    store: dict[str, dict],
    device: torch.device,
    split: str = "test",
    n_samples: int = 20,
    max_episodes: int = 50,
    seed: int = 0,
    remove_semantics: str = spent,
    hide_edge_weights: bool = False,
    action_encoding: str = basic_encoding,
    action_policy: Callable | None = None,
) -> dict[str, float]:
    """
    Stochastic ensemble rollout (Lever 1): treat the world model as a stochastic
    simulator. At each step SAMPLE the next state from the predicted marginals
    (instead of thresholding at 0.5) and roll n_samples trajectories; compare the
    model's marginal/count distribution to the TRUE simulator's MC trajectory under
    the same recorded action sequence. Isolates the threshold->sample hypothesis:
    thresholding a monotone cascade over-commits every >0.5 node and must saturate;
    sampling commits only the predicted fraction.

    Meaningful for IC (stochastic). LT thresholds are not stored, so the true LT
    re-run draws fresh thresholds and the comparison is not faithful there.

    Metrics:
    - ens_marg_mae:   mean |model marginal - true marginal| over nodes and steps
    - ens_count_w1:   mean per-step Wasserstein-1 between model and true infected-count distributions
    - ens_count_bias: mean per-step (E[model count] - E[true count]); ~0 = unbiased, >0 = still over-predicting
    - ens_final_count_model / ens_final_count_true: mean final infected counts

    `action_policy` replaces the recorded action sequence with one this policy
    generates, on BOTH sides of the comparison, which is the off-policy
    (out-of-distribution action) test. Training injects actions uniformly at
    random; a coding agent does not, so fidelity under the recorded sequence alone
    says nothing about fidelity under the sequence the agent will actually propose.
    Policies are node-only by contract, because the adjacency map is reconstructed
    from the RECORDED edge ops and replaying different ones would desynchronise it.
    """
    rng = np.random.default_rng(seed)
    path = Path(out_dir) / f"transitions_{diffusion_model}_{split}.jsonl"
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]

    by_episode = defaultdict(list)
    for record in records:
        if record["branch"] == "main":
            by_episode[(record["graph_id"], record["episode_id"])].append(record)

    episode_keys = list(by_episode)
    if max_episodes and len(episode_keys) > max_episodes:
        episode_keys = [
            episode_keys[index]
            for index in rng.choice(len(episode_keys), size=max_episodes, replace=False)
        ]

    model.eval()
    marginal_mae, count_w1, count_bias = [], [], []
    final_model_counts, final_true_counts = [], []
    per_episode = {"count_model": [], "count_true": [], "marginal_mae": []}

    for graph_id, episode_id in episode_keys:
        episode_records = sorted(
            by_episode[(graph_id, episode_id)], key=lambda record: record["t"]
        )
        num_nodes = store[graph_id]["num_nodes"]
        adjacency_map = reconstruct_episode_adjacency(
            episode_records, store[graph_id]["base_edges"]
        )
        num_steps = len(episode_records)

        # One action sequence per episode, replayed identically by both ensembles.
        # Under the default (recorded) policy this is exactly the old behaviour.
        if action_policy is None:
            action_sequence = [record["action"] for record in episode_records]
        else:
            action_sequence = action_policy(store[graph_id], episode_records, rng)

        # True ensemble: n_samples simulator rollouts under that action sequence.
        true_infected = np.zeros((n_samples, num_steps, num_nodes), dtype=np.float32)
        for sample in range(n_samples):
            simulator = rebuild_simulator(
                store[graph_id],
                diffusion_model,
                seed=int(rng.integers(seed_upper_bound)),
                remove_semantics=remove_semantics,
            )
            for step, action in enumerate(action_sequence):
                state = simulator.advance(_action_bag(action))
                true_infected[
                    sample, step, np.asarray(state.infected, dtype=np.int64)
                ] = 1.0

        # Model ensemble: n_samples sampled rollouts (mirror rollout_episodes, but sample).
        model_infected = np.zeros((n_samples, num_steps, num_nodes), dtype=np.float32)
        for sample in range(n_samples):
            current_infected = set(episode_records[0]["state"]["infected"])
            current_frontier = set(episode_records[0]["state"]["frontier"])

            for step, record in enumerate(episode_records):
                action = action_sequence[step]
                edge_index, weights = edges_to_arrays(
                    adjacency_map[(record["t"], "main")]
                )
                rolled = dict(record)
                rolled["action"] = action
                rolled["state"] = {
                    "infected": sorted(current_infected),
                    "frontier": sorted(current_frontier),
                }
                X, _, _ = build_features(
                    rolled, edge_index, num_nodes, action_encoding
                )
                graph_input = build_graph_input(
                    edge_index,
                    weights,
                    num_nodes,
                    diffusion_model,
                    device,
                    hide_edge_weights,
                )

                probs = (
                    torch.sigmoid(model(torch.from_numpy(X).to(device), graph_input))
                    .cpu()
                    .numpy()
                )

                # Coupled sampling: draw the new infections once from the frontier
                # marginal, then derive both channels (frontier = new wave, infected
                # accumulates through the action semantics)
                # Independent per-channel draws create inconsistent states (ghost spreaders: frontier=1,
                # infected=0) that systematically inflate free-running rollouts
                adds = {
                    int(action_op["target"])
                    for action_op in action
                    if action_op["op"] == "add_node"
                }
                removes = {
                    int(action_op["target"])
                    for action_op in action
                    if action_op["op"] == "remove_node"
                }

                post_exo_infected = current_infected | adds

                if diffusion_model == "LT" or remove_semantics == blocked:
                    # LT remove_node returns the node to Susceptible, and a blocked
                    # node leaves the graph under either dynamics. Only spent IC
                    # keeps the node counted.
                    post_exo_infected -= removes

                new_draw = rng.random(num_nodes) < probs[:, 1]
                new_nodes = set(np.nonzero(new_draw)[0].tolist()) - post_exo_infected

                current_infected = post_exo_infected | new_nodes
                current_frontier = new_nodes
                model_infected[
                    sample, step, np.asarray(sorted(current_infected), dtype=np.int64)
                ] = 1.0

        # Per-node infection frequency, shape: (num_steps, num_nodes)
        model_marginal = model_infected.mean(axis=0)
        true_marginal = true_infected.mean(axis=0)
        marginal_mae.append(float(np.abs(model_marginal - true_marginal).mean()))

        model_counts = model_infected.sum(axis=2)  # (n_samples, num_steps)
        true_counts = true_infected.sum(axis=2)
        for step in range(num_steps):
            count_w1.append(
                wasserstein_distance(model_counts[:, step], true_counts[:, step])
            )
            count_bias.append(
                float(model_counts[:, step].mean() - true_counts[:, step].mean())
            )

        per_episode["count_model"].append(model_counts.mean(axis=0).tolist())
        per_episode["count_true"].append(true_counts.mean(axis=0).tolist())
        per_episode["marginal_mae"].append(
            np.abs(model_marginal - true_marginal).mean(axis=1).tolist()
        )

        final_model_counts.append(float(model_counts[:, -1].mean()))
        final_true_counts.append(float(true_counts[:, -1].mean()))

    return {
        "ens_marg_mae": float(np.mean(marginal_mae)) if marginal_mae else 0.0,
        "ens_count_w1": float(np.mean(count_w1)) if count_w1 else 0.0,
        "ens_count_bias": float(np.mean(count_bias)) if count_bias else 0.0,
        "ens_final_count_model": (
            float(np.mean(final_model_counts)) if final_model_counts else 0.0
        ),
        "ens_final_count_true": (
            float(np.mean(final_true_counts)) if final_true_counts else 0.0
        ),
        "ens_n_samples": float(n_samples),
        "ens_n_episodes": float(len(episode_keys)),
        **_step_curves(per_episode),
    }


def rebuild_competitive_simulator(
    store_entry: dict,
    diffusion_model: str,
    negative_seeds: list[int],
    seed: int = 0,
    config: CompetitiveConfig | None = None,
) -> CompetitiveSimulator:
    """A CompetitiveSimulator on a stored graph, seeded with the episode's own S_N."""
    edge_index = store_entry["edge_index"]
    ic_probs = store_entry["ic_probs"]
    num_nodes = store_entry["num_nodes"]
    directed = bool(store_entry["meta"].get("directed", False))

    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(
        (int(edge_index[0, edge]), int(edge_index[1, edge]))
        for edge in range(edge_index.shape[1])
    )
    ic_prob_map = {
        (int(edge_index[0, edge]), int(edge_index[1, edge])): float(ic_probs[edge])
        for edge in range(edge_index.shape[1])
    }

    simulator = CompetitiveSimulator(
        graph, ic_prob_map=ic_prob_map, seed=seed, config=config
    )
    simulator.reset(diffusion_model, negative_seeds)

    return simulator


@torch.inference_mode()
def competitive_rollout_ensemble(
    model: nn.Module,
    out_dir: str,
    diffusion_model: str,
    store: dict[str, dict],
    device: torch.device,
    split: str = "test",
    n_samples: int = 20,
    max_episodes: int = 50,
    seed: int = 0,
    hide_edge_weights: bool = False,
    config: CompetitiveConfig | None = None,
) -> dict[str, float]:
    """
    Two-cascade sampled-ensemble rollout against the true competitive simulator.

    The competitive twin of `rollout_ensemble`, and the metric that decides whether
    the learned head can be used as a SIMULATOR rather than only a one-step
    predictor. Two things differ from the single-cascade version and both matter:

      * The bias that counts is `ens_count_bias`, on the NEGATIVE cascade: the
        quantity a blocker is scored on. A head that over-predicts the positive
        cascade under-reports the rumour and reports containment it never achieved,
        so `ens_pos_count_bias` is reported beside it rather than folded in.
      * Sampling is COUPLED across cascades, exactly as the tie-break is: one draw
        per node decides which cascade takes it, so a node cannot be sampled into
        both and the ensemble cannot produce states the simulator never can.
    """
    rng = np.random.default_rng(seed)
    path = Path(out_dir) / f"transitions_{diffusion_model}_{split}.jsonl"
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]

    by_episode = defaultdict(list)
    for record in records:
        if record["branch"] == "main":
            by_episode[(record["graph_id"], record["episode_id"])].append(record)

    episode_keys = list(by_episode)
    if max_episodes and len(episode_keys) > max_episodes:
        episode_keys = [
            episode_keys[index]
            for index in rng.choice(len(episode_keys), size=max_episodes, replace=False)
        ]

    model.eval()
    marginal_mae, count_bias, pos_count_bias, count_w1 = [], [], [], []
    final_model_counts, final_true_counts = [], []
    per_episode = {
        "count_model": [], "count_true": [], "pos_count_model": [], "marginal_mae": [],
    }

    for graph_id, episode_id in episode_keys:
        episode_records = sorted(
            by_episode[(graph_id, episode_id)], key=lambda record: record["t"]
        )
        num_nodes = store[graph_id]["num_nodes"]
        negative_seeds = episode_records[0].get("negative_seeds", [])
        adjacency_map = reconstruct_episode_adjacency(
            episode_records, store[graph_id]["base_edges"]
        )
        num_steps = len(episode_records)

        true_infected = np.zeros((n_samples, num_steps, num_nodes), dtype=np.float32)
        for sample in range(n_samples):
            simulator = rebuild_competitive_simulator(
                store[graph_id],
                diffusion_model,
                negative_seeds,
                seed=int(rng.integers(seed_upper_bound)),
                config=config,
            )
            for step, record in enumerate(episode_records):
                state = simulator.advance(_action_bag(record["action"]))
                true_infected[
                    sample, step, np.asarray(state.infected, dtype=np.int64)
                ] = 1.0

        model_infected = np.zeros((n_samples, num_steps, num_nodes), dtype=np.float32)
        model_positive = np.zeros((n_samples, num_steps, num_nodes), dtype=np.float32)

        for sample in range(n_samples):
            state = _initial_competitive_state(episode_records[0])

            for step, record in enumerate(episode_records):
                edge_index, weights = edges_to_arrays(
                    adjacency_map[(record["t"], "main")]
                )
                rolled = dict(record) | {"state": state.to_dict()}
                X, _ = build_competitive_features(rolled, edge_index, num_nodes)
                graph_input = build_graph_input(
                    edge_index,
                    weights,
                    num_nodes,
                    diffusion_model,
                    device,
                    hide_edge_weights,
                )
                probs = (
                    torch.sigmoid(model(torch.from_numpy(X).to(device), graph_input))
                    .cpu()
                    .numpy()
                )
                state = sample_competitive_step(
                    state, record["action"], probs, rng, num_nodes
                )
                model_infected[
                    sample, step, np.asarray(state.infected, dtype=np.int64)
                ] = 1.0
                model_positive[
                    sample, step, np.asarray(state.pos_infected, dtype=np.int64)
                ] = 1.0

        marginal_mae.append(
            float(np.abs(model_infected.mean(axis=0) - true_infected.mean(axis=0)).mean())
        )

        model_counts = model_infected.sum(axis=2)
        true_counts = true_infected.sum(axis=2)
        positive_counts = model_positive.sum(axis=2)

        for step in range(num_steps):
            count_w1.append(
                wasserstein_distance(model_counts[:, step], true_counts[:, step])
            )
            count_bias.append(
                float(model_counts[:, step].mean() - true_counts[:, step].mean())
            )
            pos_count_bias.append(float(positive_counts[:, step].mean()))

        per_episode["count_model"].append(model_counts.mean(axis=0).tolist())
        per_episode["count_true"].append(true_counts.mean(axis=0).tolist())
        per_episode["pos_count_model"].append(positive_counts.mean(axis=0).tolist())
        per_episode["marginal_mae"].append(
            np.abs(model_infected.mean(axis=0) - true_infected.mean(axis=0))
            .mean(axis=1)
            .tolist()
        )

        final_model_counts.append(float(model_counts[:, -1].mean()))
        final_true_counts.append(float(true_counts[:, -1].mean()))

    return {
        "ens_marg_mae": float(np.mean(marginal_mae)) if marginal_mae else 0.0,
        "ens_count_w1": float(np.mean(count_w1)) if count_w1 else 0.0,
        "ens_count_bias": float(np.mean(count_bias)) if count_bias else 0.0,
        "ens_pos_count_mean": float(np.mean(pos_count_bias)) if pos_count_bias else 0.0,
        "ens_final_count_model": (
            float(np.mean(final_model_counts)) if final_model_counts else 0.0
        ),
        "ens_final_count_true": (
            float(np.mean(final_true_counts)) if final_true_counts else 0.0
        ),
        "ens_n_samples": float(n_samples),
        "ens_n_episodes": float(len(episode_keys)),
        **_step_curves(per_episode),
    }


def _initial_competitive_state(record: dict) -> State:
    """The first recorded state of a competitive episode, as a State."""
    state = record["state"]

    return State(
        infected=list(state["infected"]),
        frontier=list(state["frontier"]),
        pos_infected=list(state.get("pos_infected", [])),
        pos_frontier=list(state.get("pos_frontier", [])),
    )


def sample_competitive_step(
    state: State,
    action: list[dict],
    probs: np.ndarray,
    rng: np.random.Generator,
    num_nodes: int,
) -> State:
    """
    One sampled two-cascade step from the head's four marginals.

    Coupled rather than four independent draws, and for the same reason the
    single-cascade sampler couples its two channels: independent draws produce states
    the simulator cannot reach (here a node in BOTH cascades) and those states
    systematically inflate a free-running rollout. One uniform per node decides
    whether it activates at all and, if so, which cascade takes it, in proportion to
    the two frontier marginals the head already resolved through the tie-break.
    """
    adds = {
        int(op["target"]) for op in action if op["op"] == "add_node"
    }
    removes = {int(op["target"]) for op in action if op["op"] == "remove_node"}

    negative = set(state.infected) - removes
    positive = (set(state.pos_infected) | (adds - negative)) - removes

    committed = negative | positive | removes
    free = np.ones(num_nodes, dtype=bool)
    free[list(committed)] = False

    p_negative = np.where(free, probs[:, 1], 0.0)
    p_positive = np.where(free, probs[:, 3], 0.0)
    total = p_negative + p_positive

    draw = rng.random(num_nodes)
    activates = draw < np.clip(total, 0.0, 1.0)
    # ONE uniform does both jobs: `draw < total` decides that the node activates, and
    # the same draw landing in [0, p_negative) rather than [p_negative, total)
    # decides which cascade takes it, which splits it in exactly the right
    # proportion without a second random number or a second source of drift
    goes_negative = activates & (draw < p_negative)

    new_negative = set(np.flatnonzero(goes_negative).tolist())
    new_positive = set(np.flatnonzero(activates & ~goes_negative).tolist())

    return State(
        infected=sorted(negative | new_negative),
        frontier=sorted(new_negative),
        pos_infected=sorted(positive | new_positive),
        pos_frontier=sorted(new_positive),
    )


def _store_graph(store_entry: dict) -> tuple:
    """(nx graph, {(u, v): p}) from a stored graph, shared by all three rebuilders."""
    edge_index = store_entry["edge_index"]
    ic_probs = store_entry["ic_probs"]
    num_nodes = store_entry["num_nodes"]
    directed = bool(store_entry["meta"].get("directed", False))

    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(
        (int(edge_index[0, edge]), int(edge_index[1, edge]))
        for edge in range(edge_index.shape[1])
    )
    ic_prob_map = {
        (int(edge_index[0, edge]), int(edge_index[1, edge])): float(ic_probs[edge])
        for edge in range(edge_index.shape[1])
    }

    return graph, ic_prob_map


def rebuild_epidemic_simulator(
    store_entry: dict,
    diffusion_model: str,
    seed: int = 0,
    config: EpidemicConfig | None = None,
) -> EpidemicSimulator:
    """An EpidemicSimulator on a stored graph, with no outbreak committed yet."""
    graph, ic_prob_map = _store_graph(store_entry)
    simulator = EpidemicSimulator(
        graph, ic_prob_map=ic_prob_map, seed=seed, config=config
    )
    simulator.reset(diffusion_model)

    return simulator


def _initial_epidemic_state(record: dict) -> State:
    """The first recorded state of a compartmental episode, as a State."""
    state = record["state"]

    return State(
        infected=list(state["infected"]),
        frontier=list(state["frontier"]),
        exposed=list(state.get("exposed", [])),
        recovered=list(state.get("recovered", [])),
    )


def sample_epidemic_step(
    state: State,
    action: list[dict],
    probs: np.ndarray,
    rng: np.random.Generator,
    num_nodes: int,
    dynamics: str = "SIR",
) -> State:
    """
    One sampled compartmental step from the head's five marginals.

    COUPLED, exactly as the two-cascade sampler is coupled and for a stronger
    version of the same reason: S, E, I and R are mutually EXCLUSIVE, so four
    independent Bernoulli draws would put a node in two compartments at once,
    a state the simulator cannot reach and one that inflates a free-running rollout
    in whichever direction the noise happens to point. One uniform per node is drawn
    against the cumulative (E, I, R, S) distribution the head already composed, so
    the result is always a valid one-hot.

    `S` is derived rather than predicted (`1 - E - I - R`) and the four are
    renormalized, because the head's exact composition is put through a
    sigmoid/logit round trip and a clamp before it gets here: the sum is 1 up to
    floating point, not identically.

    `ever` is advanced from the SAMPLE, not from column 0: a node counts as newly
    infected exactly when it was susceptible before the step and is not after, which
    is the simulator's own bookkeeping and the only definition that stays right when
    SIS sends a node back to S.
    """
    adds = {int(op["target"]) for op in action if op["op"] == "add_node"}
    removes = {int(op["target"]) for op in action if op["op"] == "remove_node"}

    # T_exo, matching the head's: a seed is an index case straight into I, a dose
    # empties every compartment
    infectious = (set(state.frontier) | adds) - removes
    exposed = set(state.exposed) - adds - removes
    recovered = set(state.recovered) - adds - removes
    ever = (set(state.infected) | adds) - removes

    committed = np.zeros(num_nodes, dtype=bool)
    committed[list(removes)] = True

    exposed_p = np.clip(probs[:, 2], 0.0, 1.0)
    infectious_p = np.clip(probs[:, 3], 0.0, 1.0)
    recovered_p = np.clip(probs[:, 4], 0.0, 1.0)
    susceptible_p = np.clip(1.0 - exposed_p - infectious_p - recovered_p, 0.0, 1.0)

    stacked = np.stack(
        [exposed_p, infectious_p, recovered_p, susceptible_p], axis=1
    )  # shape: (N, 4)
    total = stacked.sum(axis=1, keepdims=True)
    stacked = stacked / np.where(total > 0.0, total, 1.0)

    draw = rng.random(num_nodes)[:, None]
    # First cumulative bucket the uniform lands in; 3 = susceptible
    drawn = (np.cumsum(stacked, axis=1) < draw).sum(axis=1).clip(0, 3)
    drawn[committed] = 3

    was_susceptible = np.ones(num_nodes, dtype=bool)
    was_susceptible[list(exposed | infectious | recovered | removes)] = False

    next_exposed = set(np.flatnonzero(drawn == 0).tolist()) - removes
    next_infectious = set(np.flatnonzero(drawn == 1).tolist()) - removes
    next_recovered = set(np.flatnonzero(drawn == 2).tolist()) - removes

    if dynamics != "SEIR":
        next_exposed = set()

    if dynamics == "SIS":
        next_recovered = set()

    newly = {
        node
        for node in next_exposed | next_infectious
        if was_susceptible[node]
    }

    return State(
        infected=sorted(ever | newly),
        frontier=sorted(next_infectious),
        exposed=sorted(next_exposed),
        recovered=sorted(next_recovered),
    )


@torch.inference_mode()
def epidemic_rollout_ensemble(
    model: nn.Module,
    out_dir: str,
    diffusion_model: str,
    store: dict[str, dict],
    device: torch.device,
    split: str = "test",
    n_samples: int = 20,
    max_episodes: int = 50,
    seed: int = 0,
    hide_edge_weights: bool = False,
    config: EpidemicConfig | None = None,
) -> dict[str, float]:
    """
    Compartmental sampled-ensemble rollout against the true epidemic simulator.

    The compartmental twin of `rollout_ensemble`, and the metric that decides
    whether the compartment head can be used as a SIMULATOR rather than only a
    one-step predictor. Two things it reports that the single-cascade version cannot:

      * **`ens_prevalence_bias`**, on `|I(t)|`. `ens_count_bias` is on the attack
        set, which is monotone and therefore forgiving; the prevalence is the
        non-monotone quantity, and a head that cannot shrink `I` shows up here
        first (as a positive bias that grows with `t`) while the attack-set bias
        still looks fine. This is the saturation guard for the compartmental case.
      * **`ens_peak_model` / `ens_peak_true`**, the peak prevalence, which is what
        §2.6 lists as the shape metric this literature grades on and the attack rate
        does not capture.
    """
    rng = np.random.default_rng(seed)
    path = Path(out_dir) / f"transitions_{diffusion_model}_{split}.jsonl"
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]

    by_episode = defaultdict(list)
    for record in records:
        if record["branch"] == "main":
            by_episode[(record["graph_id"], record["episode_id"])].append(record)

    episode_keys = list(by_episode)
    if max_episodes and len(episode_keys) > max_episodes:
        episode_keys = [
            episode_keys[index]
            for index in rng.choice(len(episode_keys), size=max_episodes, replace=False)
        ]

    model.eval()
    marginal_mae, count_w1, count_bias, prevalence_bias = [], [], [], []
    final_model_counts, final_true_counts = [], []
    per_episode = {
        "count_model": [], "count_true": [], "prevalence_model": [],
        "prevalence_true": [], "marginal_mae": [],
    }
    peak_model, peak_true = [], []

    for graph_id, episode_id in episode_keys:
        episode_records = sorted(
            by_episode[(graph_id, episode_id)], key=lambda record: record["t"]
        )
        num_nodes = store[graph_id]["num_nodes"]
        adjacency_map = reconstruct_episode_adjacency(
            episode_records, store[graph_id]["base_edges"]
        )
        num_steps = len(episode_records)

        true_ever = np.zeros((n_samples, num_steps, num_nodes), dtype=np.float32)
        true_infectious = np.zeros((n_samples, num_steps), dtype=np.float32)

        for sample in range(n_samples):
            simulator = rebuild_epidemic_simulator(
                store[graph_id],
                diffusion_model,
                seed=int(rng.integers(seed_upper_bound)),
                config=config,
            )
            for step, record in enumerate(episode_records):
                state = simulator.advance(_action_bag(record["action"]))
                true_ever[
                    sample, step, np.asarray(state.infected, dtype=np.int64)
                ] = 1.0
                true_infectious[sample, step] = len(state.frontier)

        model_ever = np.zeros((n_samples, num_steps, num_nodes), dtype=np.float32)
        model_infectious = np.zeros((n_samples, num_steps), dtype=np.float32)

        for sample in range(n_samples):
            state = _initial_epidemic_state(episode_records[0])

            for step, record in enumerate(episode_records):
                edge_index, weights = edges_to_arrays(
                    adjacency_map[(record["t"], "main")]
                )
                rolled = dict(record) | {"state": state.to_dict()}
                X, _ = build_epidemic_features(rolled, edge_index, num_nodes)
                graph_input = build_graph_input(
                    edge_index,
                    weights,
                    num_nodes,
                    diffusion_model,
                    device,
                    hide_edge_weights,
                )
                probs = (
                    torch.sigmoid(model(torch.from_numpy(X).to(device), graph_input))
                    .cpu()
                    .numpy()
                )
                state = sample_epidemic_step(
                    state, record["action"], probs, rng, num_nodes, diffusion_model
                )
                model_ever[
                    sample, step, np.asarray(state.infected, dtype=np.int64)
                ] = 1.0
                model_infectious[sample, step] = len(state.frontier)

        marginal_mae.append(
            float(np.abs(model_ever.mean(axis=0) - true_ever.mean(axis=0)).mean())
        )

        model_counts = model_ever.sum(axis=2)
        true_counts = true_ever.sum(axis=2)

        for step in range(num_steps):
            count_w1.append(
                wasserstein_distance(model_counts[:, step], true_counts[:, step])
            )
            count_bias.append(
                float(model_counts[:, step].mean() - true_counts[:, step].mean())
            )
            prevalence_bias.append(
                float(
                    model_infectious[:, step].mean() - true_infectious[:, step].mean()
                )
            )

        per_episode["count_model"].append(model_counts.mean(axis=0).tolist())
        per_episode["count_true"].append(true_counts.mean(axis=0).tolist())
        per_episode["prevalence_model"].append(model_infectious.mean(axis=0).tolist())
        per_episode["prevalence_true"].append(true_infectious.mean(axis=0).tolist())
        per_episode["marginal_mae"].append(
            np.abs(model_ever.mean(axis=0) - true_ever.mean(axis=0)).mean(axis=1).tolist()
        )

        final_model_counts.append(float(model_counts[:, -1].mean()))
        final_true_counts.append(float(true_counts[:, -1].mean()))
        peak_model.append(float(model_infectious.max(axis=1).mean()))
        peak_true.append(float(true_infectious.max(axis=1).mean()))

    def mean(values: list) -> float:
        return float(np.mean(values)) if values else 0.0

    return {
        "ens_marg_mae": mean(marginal_mae),
        "ens_count_w1": mean(count_w1),
        "ens_count_bias": mean(count_bias),
        # The non-monotone quantity, and the one a head that cannot represent
        # recovery gets wrong first
        "ens_prevalence_bias": mean(prevalence_bias),
        "ens_peak_model": mean(peak_model),
        "ens_peak_true": mean(peak_true),
        "ens_final_count_model": mean(final_model_counts),
        "ens_final_count_true": mean(final_true_counts),
        "ens_n_samples": float(n_samples),
        "ens_n_episodes": float(len(episode_keys)),
        **_step_curves(per_episode),
    }


def rebuild_simulator(
    store_entry: dict,
    diffusion_model: str,
    seed: int = 0,
    remove_semantics: str = spent,
) -> Simulator:
    """Reconstruct a data/wm_simulator.Simulator from a stored graph.

    `remove_semantics` MUST be the dataset's. It used to be hard-defaulted to
    `spent` here while the model side of every comparison honoured the configured
    value, so on a `blocked` (containment) dataset the ground-truth ensemble kept
    counting each removed node as infected and the model's did not: `ens_count_bias`
    picked up a spurious offset of about -k, and planning regret was measured
    against an oracle running different dynamics from the one the data came from.
    """
    edge_index = store_entry["edge_index"]
    ic_probs = store_entry["ic_probs"]
    num_nodes = store_entry["num_nodes"]
    directed = bool(store_entry["meta"].get("directed", False))

    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(
        (int(edge_index[0, edge]), int(edge_index[1, edge]))
        for edge in range(edge_index.shape[1])
    )
    ic_prob_map = {
        (int(edge_index[0, edge]), int(edge_index[1, edge])): float(ic_probs[edge])
        for edge in range(edge_index.shape[1])
    }

    simulator = Simulator(
        graph,
        ic_prob_map=ic_prob_map,
        seed=seed,
        remove_semantics=remove_semantics,
    )
    simulator.reset(diffusion_model)

    return simulator


planning_split_test = "test"
planning_split_legacy = "legacy"
valid_planning_splits = (planning_split_test, planning_split_legacy)


def graphs_in_split(
    out_dir: str | Path, diffusion_model: str, split: str
) -> list[str]:
    """Graph ids whose transitions appear in one split, in store order."""
    path = Path(out_dir) / f"transitions_{diffusion_model}_{split}.jsonl"

    if not path.exists():
        return []

    seen = {}

    for line in path.read_text().splitlines():
        if line.strip():
            seen.setdefault(json.loads(line)["graph_id"], None)

    return list(seen)


def select_planning_graphs(
    store: dict[str, dict],
    n_graphs: int,
    out_dir: str | Path | None,
    diffusion_model: str,
    planning_split: str = planning_split_test,
) -> tuple[list[str], dict]:
    """
    Choose the graphs a planning metric scores, and report where they came from.

    Returns `(graph_ids, provenance)`. The provenance block is not decoration: a
    planning number is only a generalization claim if the graphs behind it were
    held out, and `train_overlap` / `val_overlap` are what let a reader check
    that instead of taking it on trust. Both must be 0 for a reported test
    result.
    """
    if planning_split not in valid_planning_splits:
        raise ValueError(
            f"unknown planning_split {planning_split!r}; "
            f"choose one of {list(valid_planning_splits)}"
        )

    if planning_split == planning_split_legacy or out_dir is None:
        graph_ids = list(store)[:n_graphs]
        provenance = {
            "planning_split": planning_split_legacy,
            "planning_graph_ids": graph_ids,
            "n_planning_graphs": len(graph_ids),
            # Unknown rather than 0: without the dataset directory there is no
            # way to tell which split these graphs came from, and reporting 0
            # would assert something that was never checked.
            "train_overlap": None,
            "val_overlap": None,
        }

        return graph_ids, provenance

    in_split = {
        name: set(graphs_in_split(out_dir, diffusion_model, name))
        for name in ("train", "val", "test")
    }
    # Store order, not file order, so the choice is stable across reruns and
    # independent of how the JSONL happens to be written.
    candidates = [g for g in store if g in in_split["test"]]

    if not candidates:
        raise ValueError(
            f"planning evaluation asked for split={planning_split!r} but no graph "
            f"in the store appears in transitions_{diffusion_model}_test.jsonl. "
            f"With a single-graph dataset, or one generated with --split-mode "
            f"episode_random, there are no held-out graphs to plan on; pass "
            f"planning_split={planning_split_legacy!r} to reproduce the old "
            f"(non-held-out) behaviour explicitly."
        )

    graph_ids = candidates[:n_graphs]
    chosen = set(graph_ids)
    provenance = {
        "planning_split": planning_split_test,
        "planning_graph_ids": graph_ids,
        "n_planning_graphs": len(graph_ids),
        "train_overlap": len(chosen & in_split["train"]),
        "val_overlap": len(chosen & in_split["val"]),
    }

    return graph_ids, provenance


@torch.inference_mode()
def planning_regret(
    model: nn.Module,
    store_entry: dict,
    diffusion_model: str,
    device: torch.device,
    n_states: int = 20,
    n_candidates: int = 20,
    mc_runs: int = 8,
    seed: int = 0,
    threshold: float = 0.5,
    hide_edge_weights: bool = False,
    remove_semantics: str = spent,
    action_encoding: str = basic_encoding,
) -> dict[str, float]:
    """One-step greedy: model picks argmax predicted spread; compare true spread to oracle."""
    rng = np.random.default_rng(seed)
    num_nodes = store_entry["num_nodes"]
    edge_index = store_entry["edge_index"]
    ic_probs = store_entry["ic_probs"]

    graph_input = build_graph_input(
        edge_index, ic_probs, num_nodes, diffusion_model, device, hide_edge_weights
    )
    degrees = np.zeros(num_nodes)
    np.add.at(degrees, edge_index[0], 1)
    np.add.at(degrees, edge_index[1], 1)

    model.eval()
    model_regrets, random_regrets, degree_regrets = [], [], []

    def true_spread(infected: set[int], action: list) -> float:
        gains = []
        for _ in range(mc_runs):
            simulator = rebuild_simulator(
                store_entry,
                diffusion_model,
                seed=int(rng.integers(seed_upper_bound)),
                remove_semantics=remove_semantics,
            )

            for node in infected:
                simulator.model.status[node] = 1

            next_state = simulator.advance(action)
            gains.append(len(next_state.infected) - len(infected))

        return float(np.mean(gains))

    for _ in range(n_states):
        num_infected = max(1, num_nodes // 20)
        infected = set(rng.choice(num_nodes, size=num_infected, replace=False).tolist())
        susceptible = [node for node in range(num_nodes) if node not in infected]

        if not susceptible:
            continue

        candidates = list(
            rng.choice(
                susceptible, size=min(n_candidates, len(susceptible)), replace=False
            )
        )
        predicted_spreads = []
        for candidate in candidates:
            record = {
                "state": {"infected": sorted(infected), "frontier": sorted(infected)},
                "action": [{"op": "add_node", "target": int(candidate)}],
                "next_state": {"infected": [], "frontier": []},
                "next_marginal_infected": {},  # y is unused here (only X is read)
                "next_marginal_frontier": {},
            }
            X, _, _ = build_features(record, edge_index, num_nodes, action_encoding)
            probs = (
                torch.sigmoid(model(torch.from_numpy(X).to(device), graph_input))
                .cpu()
                .numpy()
            )
            susceptible_mask = np.array(
                [node not in infected for node in range(num_nodes)]
            )
            predicted_spreads.append(probs[susceptible_mask, 0].sum())

        model_choice = candidates[int(np.argmax(predicted_spreads))]
        random_choice = candidates[int(rng.integers(len(candidates)))]
        degree_choice = candidates[
            int(np.argmax([degrees[candidate] for candidate in candidates]))
        ]
        true_spreads = {
            candidate: true_spread(infected, [ActionOp("add_node", int(candidate))])
            for candidate in candidates
        }
        oracle = max(true_spreads.values())
        model_regrets.append(oracle - true_spreads[model_choice])
        random_regrets.append(oracle - true_spreads[random_choice])
        degree_regrets.append(oracle - true_spreads[degree_choice])

    return {
        "plan_regret_model": float(np.mean(model_regrets)) if model_regrets else 0.0,
        "plan_regret_random": float(np.mean(random_regrets)) if random_regrets else 0.0,
        "plan_regret_degree": float(np.mean(degree_regrets)) if degree_regrets else 0.0,
    }


@torch.inference_mode()
def blocking_regret(
    model: nn.Module,
    store_entry: dict,
    diffusion_model: str,
    device: torch.device,
    n_states: int = 10,
    n_candidates: int = 20,
    mc_runs: int = 8,
    horizon: int = 6,
    seed: int = 0,
    hide_edge_weights: bool = False,
    config: CompetitiveConfig | None = None,
) -> dict[str, float]:
    """
    One-step blocker choice: does the model pick the counter-seed that saves the most?

    The competitive analogue of `planning_regret`, and it measures PREVENTED
    influence rather than spread: `sigma(S_N, empty) - sigma(S_N, blocker)`, the
    quantity every name in research/influence_blocking.md §8.1 refers to. Regret is
    against the best candidate in the same shortlist, so 0 means the model chose the
    node the simulator agrees was best.

    Both comparison baselines are here for a reason §5.4 states outright: `degree` is
    the heuristic that "cannot be used for influence blocking maximization at all",
    and `proximity` (an out-neighbour of the rumour's own seeds) is the strong
    cheap one. Beating random is not evidence here; beating proximity is.
    """
    rng = np.random.default_rng(seed)
    num_nodes = store_entry["num_nodes"]
    edge_index = store_entry["edge_index"]
    ic_probs = store_entry["ic_probs"]

    graph_input = build_graph_input(
        edge_index, ic_probs, num_nodes, diffusion_model, device, hide_edge_weights
    )
    degrees = np.zeros(num_nodes)
    np.add.at(degrees, edge_index[0], 1)
    np.add.at(degrees, edge_index[1], 1)

    out_neighbours = defaultdict(list)
    for edge in range(edge_index.shape[1]):
        out_neighbours[int(edge_index[0, edge])].append(int(edge_index[1, edge]))

    model.eval()
    model_regrets, random_regrets, degree_regrets, proximity_regrets = [], [], [], []

    def true_prevented(negative_seeds: list[int], blocker: int | None) -> float:
        plan = [[ActionOp("add_node", int(blocker))]] if blocker is not None else [[]]
        totals = []

        for _ in range(mc_runs):
            simulator = rebuild_competitive_simulator(
                store_entry,
                diffusion_model,
                negative_seeds,
                seed=int(rng.integers(seed_upper_bound)),
                config=config,
            )
            state = simulator.current_state()
            for timestep in range(horizon + 1):
                state = simulator.advance(plan[timestep] if timestep < len(plan) else [])
                if timestep > 0 and not state.frontier and not state.pos_frontier:
                    break

            totals.append(len(state.infected))

        return float(np.mean(totals))

    for _ in range(n_states):
        size = max(1, num_nodes // 100)
        negative_seeds = sorted(
            int(node) for node in rng.choice(num_nodes, size=size, replace=False)
        )
        ring = sorted(
            {
                node
                for source in negative_seeds
                for node in out_neighbours[source]
                if node not in negative_seeds
            }
        )
        pool = [node for node in range(num_nodes) if node not in negative_seeds]

        if not pool:
            continue

        candidates = sorted(
            {int(node) for node in rng.choice(pool, size=min(n_candidates, len(pool)), replace=False)}
            | set(ring[:5])
        )

        predicted = []
        for candidate in candidates:
            record = {
                "state": {
                    "infected": negative_seeds,
                    "frontier": negative_seeds,
                    "pos_infected": [],
                    "pos_frontier": [],
                },
                "action": [{"op": "add_node", "target": int(candidate)}],
                "next_marginal_infected": {},
                "next_marginal_frontier": {},
                "next_marginal_pos_infected": {},
                "next_marginal_pos_frontier": {},
            }
            X, _ = build_competitive_features(record, edge_index, num_nodes)
            probs = (
                torch.sigmoid(model(torch.from_numpy(X).to(device), graph_input))
                .cpu()
                .numpy()
            )
            # The model's own estimate of how far the rumour gets next step; the
            # blocker it should choose is the one that MINIMIZES this
            predicted.append(probs[:, 0].sum())

        model_choice = candidates[int(np.argmin(predicted))]
        random_choice = candidates[int(rng.integers(len(candidates)))]
        degree_choice = candidates[
            int(np.argmax([degrees[candidate] for candidate in candidates]))
        ]
        in_ring = [node for node in candidates if node in set(ring)]
        proximity_choice = (
            max(in_ring, key=lambda node: degrees[node]) if in_ring else degree_choice
        )

        unopposed = true_prevented(negative_seeds, None)
        prevented = {
            candidate: unopposed - true_prevented(negative_seeds, candidate)
            for candidate in candidates
        }
        oracle = max(prevented.values())

        model_regrets.append(oracle - prevented[model_choice])
        random_regrets.append(oracle - prevented[random_choice])
        degree_regrets.append(oracle - prevented[degree_choice])
        proximity_regrets.append(oracle - prevented[proximity_choice])

    return {
        "block_regret_model": float(np.mean(model_regrets)) if model_regrets else 0.0,
        "block_regret_random": float(np.mean(random_regrets)) if random_regrets else 0.0,
        "block_regret_degree": float(np.mean(degree_regrets)) if degree_regrets else 0.0,
        "block_regret_proximity": (
            float(np.mean(proximity_regrets)) if proximity_regrets else 0.0
        ),
    }


@torch.inference_mode()
def immunization_regret(
    model: nn.Module,
    store_entry: dict,
    diffusion_model: str,
    device: torch.device,
    n_states: int = 10,
    n_candidates: int = 20,
    mc_runs: int = 8,
    horizon: int = 10,
    seed: int = 0,
    hide_edge_weights: bool = False,
    config: EpidemicConfig | None = None,
) -> dict[str, float]:
    """
    One-dose choice: does the model pick the node whose vaccination saves the most?

    The compartmental analogue of `blocking_regret`, measuring PREVENTED INFECTIONS
: `|R(inf)| unprotected - |R(inf)| with one dose`, which is what §8.3 says the
    table should report rather than the eigendrop the spectral line optimizes.
    Regret is against the best candidate in the same shortlist, so 0 means the model
    chose the node the simulator agrees was best.

    Three comparison baselines, chosen from §3.1 rather than for convenience:
    `degree` is Pastor-Satorras & Vespignani's targeted immunization (and the row a
    learned method has to beat), `acquaintance` is Cohen et al.'s no-global-
    information rule that most embarrasses learned methods on sparse graphs (§8.3),
    and `random` is the control their whole line exists to beat. Beating random is
    not evidence here; beating degree and acquaintance is.

    Each candidate's dose is emitted as a full DELETION BAG: `remove_node` plus its
    incident arcs, because the head's T_exo assumes a vaccinated node's edges are
    gone from `edge_index`, and a bare removal would leave a fresh susceptible its
    in-edges promptly re-infect.
    """
    rng = np.random.default_rng(seed)
    num_nodes = store_entry["num_nodes"]
    edge_index = store_entry["edge_index"]

    degrees = np.zeros(num_nodes)
    np.add.at(degrees, edge_index[0], 1)
    np.add.at(degrees, edge_index[1], 1)

    neighbours = defaultdict(list)
    for edge in range(edge_index.shape[1]):
        neighbours[int(edge_index[0, edge])].append(int(edge_index[1, edge]))
        neighbours[int(edge_index[1, edge])].append(int(edge_index[0, edge]))

    def dose_bag(node: int) -> list[ActionOp]:
        bag = [ActionOp("remove_node", int(node))]
        for other in set(neighbours[int(node)]):
            bag.append(ActionOp("remove_edge", int(node), int(other)))
            bag.append(ActionOp("remove_edge", int(other), int(node)))

        return bag

    model.eval()
    model_regrets, random_regrets, degree_regrets, acquaintance_regrets = [], [], [], []

    def true_attack(sources: list[int], dose: int | None) -> float:
        opening = [ActionOp("add_node", int(node)) for node in sources]
        if dose is not None:
            opening = opening + dose_bag(dose)

        totals = []
        for _ in range(mc_runs):
            simulator = rebuild_epidemic_simulator(
                store_entry,
                diffusion_model,
                seed=int(rng.integers(seed_upper_bound)),
                config=config,
            )
            state = simulator.current_state()
            for timestep in range(horizon + 1):
                state = simulator.advance(opening if timestep == 0 else [])
                if timestep > 0 and not state.frontier and not state.exposed:
                    break

            totals.append(len(state.infected))

        return float(np.mean(totals))

    for _ in range(n_states):
        size = max(1, num_nodes // 100)
        sources = sorted(
            int(node) for node in rng.choice(num_nodes, size=size, replace=False)
        )
        pool = [node for node in range(num_nodes) if node not in sources]

        if not pool:
            continue

        candidates = sorted(
            {
                int(node)
                for node in rng.choice(
                    pool, size=min(n_candidates, len(pool)), replace=False
                )
            }
        )

        predicted = []
        for candidate in candidates:
            bag = dose_bag(candidate)
            edges = apply_edge_ops(
                store_entry["base_edges"], [action.to_dict() for action in bag]
            )
            candidate_index, candidate_weights = edges_to_arrays(edges)
            record = {
                "state": {
                    "infected": sources,
                    "frontier": sources,
                    "exposed": [],
                    "recovered": [],
                },
                "action": [action.to_dict() for action in bag],
                "next_marginal_infected": {},
                "next_marginal_incidence": {},
                "next_marginal_exposed": {},
                "next_marginal_infectious": {},
                "next_marginal_recovered": {},
            }
            X, _ = build_epidemic_features(record, candidate_index, num_nodes)
            graph_input = build_graph_input(
                candidate_index,
                candidate_weights,
                num_nodes,
                diffusion_model,
                device,
                hide_edge_weights,
            )
            probs = (
                torch.sigmoid(model(torch.from_numpy(X).to(device), graph_input))
                .cpu()
                .numpy()
            )
            # The model's own estimate of how far the outbreak gets next step; the
            # dose it should choose is the one that MINIMIZES it
            predicted.append(probs[:, 0].sum())

        model_choice = candidates[int(np.argmin(predicted))]
        random_choice = candidates[int(rng.integers(len(candidates)))]
        degree_choice = candidates[
            int(np.argmax([degrees[candidate] for candidate in candidates]))
        ]
        # Cohen et al. 2003: pick a random node, immunize a random NEIGHBOUR of it.
        # Restricted to the shortlist so all four choices face the same menu.
        acquaintance_choice = degree_choice
        for _ in range(len(candidates)):
            anchor = int(rng.integers(num_nodes))
            options = [node for node in neighbours[anchor] if node in set(candidates)]
            if options:
                acquaintance_choice = int(rng.choice(options))
                break

        unprotected = true_attack(sources, None)
        prevented = {
            candidate: unprotected - true_attack(sources, candidate)
            for candidate in candidates
        }
        oracle = max(prevented.values())

        model_regrets.append(oracle - prevented[model_choice])
        random_regrets.append(oracle - prevented[random_choice])
        degree_regrets.append(oracle - prevented[degree_choice])
        acquaintance_regrets.append(oracle - prevented[acquaintance_choice])

    def mean(values: list) -> float:
        return float(np.mean(values)) if values else 0.0

    return {
        "immun_regret_model": mean(model_regrets),
        "immun_regret_random": mean(random_regrets),
        "immun_regret_degree": mean(degree_regrets),
        "immun_regret_acquaintance": mean(acquaintance_regrets),
    }


@torch.inference_mode()
def immunization_regret_multi(
    model: nn.Module,
    store: dict[str, dict],
    diffusion_model: str,
    device: torch.device,
    n_graphs: int = 5,
    seed: int = 0,
    out_dir: str | Path | None = None,
    planning_split: str = planning_split_test,
    hide_edge_weights: bool = False,
    config: EpidemicConfig | None = None,
    **kwargs: object,
) -> dict[str, float]:
    """Average immunization regret over the first n_graphs graphs, with a cross-graph std."""
    graph_ids, provenance = select_planning_graphs(
        store, n_graphs, out_dir, diffusion_model, planning_split
    )

    if not graph_ids:
        raise ValueError("immunization_regret_multi: empty graph store")

    per_graph = [
        immunization_regret(
            model,
            store[graph_id],
            diffusion_model,
            device,
            seed=seed + offset,
            hide_edge_weights=hide_edge_weights,
            config=config,
            **kwargs,
        )
        for offset, graph_id in enumerate(graph_ids)
    ]

    results = {"plan_n_graphs": float(len(per_graph))}
    for key in per_graph[0]:
        values = np.array([entry[key] for entry in per_graph], dtype=np.float64)
        results[key] = float(values.mean())
        results[f"{key}_std"] = float(values.std())

    results.update(provenance)

    return results


@torch.inference_mode()
def blocking_regret_multi(
    model: nn.Module,
    store: dict[str, dict],
    diffusion_model: str,
    device: torch.device,
    n_graphs: int = 5,
    seed: int = 0,
    out_dir: str | Path | None = None,
    planning_split: str = planning_split_test,
    hide_edge_weights: bool = False,
    config: CompetitiveConfig | None = None,
    **kwargs: object,
) -> dict[str, float]:
    """Average blocking regret over the first n_graphs graphs, with a cross-graph std."""
    graph_ids, provenance = select_planning_graphs(
        store, n_graphs, out_dir, diffusion_model, planning_split
    )

    if not graph_ids:
        raise ValueError("blocking_regret_multi: empty graph store")

    per_graph = [
        blocking_regret(
            model,
            store[graph_id],
            diffusion_model,
            device,
            seed=seed + offset,
            hide_edge_weights=hide_edge_weights,
            config=config,
            **kwargs,
        )
        for offset, graph_id in enumerate(graph_ids)
    ]

    results = {"plan_n_graphs": float(len(per_graph))}
    for key in per_graph[0]:
        values = np.array([entry[key] for entry in per_graph], dtype=np.float64)
        results[key] = float(values.mean())
        results[f"{key}_std"] = float(values.std())

    results.update(provenance)

    return results


@torch.inference_mode()
def _model_spread(
    model: nn.Module,
    seeds: list[int],
    store_entry: dict,
    graph_input,
    diffusion_model: str,
    device: torch.device,
    horizon: int,
    n_samples: int,
    rng: np.random.Generator,
    action_encoding: str = basic_encoding,
) -> float:
    """Expected final infected count under the MODEL rolled to `horizon`.

    This is the world model used the way the coding agent uses it — free-running,
    multi-step, sampled — rather than the single-step marginal sum
    `planning_regret` scores with. A one-step score can rank seeds correctly while
    the multi-step rollout it is supposed to replace does not.
    """
    num_nodes = store_entry["num_nodes"]
    edge_index = store_entry["edge_index"]
    totals = []

    for _ in range(n_samples):
        infected: set[int] = set()
        frontier: set[int] = set()

        for step in range(horizon + 1):
            action = (
                [{"op": "add_node", "target": int(node)} for node in seeds]
                if step == 0
                else []
            )
            record = {
                "state": {"infected": sorted(infected), "frontier": sorted(frontier)},
                "action": action,
                "next_state": {"infected": [], "frontier": []},
                "next_marginal_infected": {},
                "next_marginal_frontier": {},
            }
            X, _, _ = build_features(record, edge_index, num_nodes, action_encoding)
            probs = (
                torch.sigmoid(model(torch.from_numpy(X).to(device), graph_input))
                .cpu()
                .numpy()
            )

            post_exo = infected | {int(node) for node in seeds} if step == 0 else infected
            new_nodes = set(
                np.nonzero(rng.random(num_nodes) < probs[:, 1])[0].tolist()
            ) - post_exo

            infected = post_exo | new_nodes
            frontier = new_nodes

            if step > 0 and not frontier:
                break

        totals.append(float(len(infected)))

    return float(np.mean(totals))

def _true_spread_set(
    seeds: list[int],
    store_entry: dict,
    diffusion_model: str,
    horizon: int,
    mc_runs: int,
    rng: np.random.Generator,
    remove_semantics: str = spent,
) -> float:
    """Ground-truth expected final spread of a seed SET, rolled to the horizon."""
    if not seeds:
        return 0.0

    bag = [ActionOp("add_node", int(node)) for node in seeds]
    totals = []

    for _ in range(mc_runs):
        simulator = rebuild_simulator(
            store_entry,
            diffusion_model,
            seed=int(rng.integers(seed_upper_bound)),
            remove_semantics=remove_semantics,
        )
        state = simulator.advance(bag)

        for _ in range(horizon):
            if not state.frontier:
                break
            state = simulator.advance([])

        totals.append(float(len(state.infected)))

    return float(np.mean(totals))

@torch.inference_mode()
def planning_regret_budget(
    model: nn.Module,
    store_entry: dict,
    diffusion_model: str,
    device: torch.device,
    k: int = 5,
    horizon: int = 20,
    n_candidates: int = 20,
    mc_runs: int = 8,
    model_samples: int = 20,
    seed: int = 0,
    hide_edge_weights: bool = False,
    remove_semantics: str = spent,
    action_encoding: str = basic_encoding,
) -> dict[str, float]:
    """
    The IM problem as actually posed: pick k seeds, measure FULL-HORIZON spread.

    `planning_regret` scores a single `add_node` by its one-step marginal against a
    one-step oracle. That is a much easier question than the one the outer loop
    asks, and it cannot distinguish a model that ranks seeds well for one step from
    one that is a usable multi-step simulator — which is the whole claim. Here the
    model builds the seed set greedily using its OWN multi-step rollout as the
    objective, and every arm is then scored by the true simulator at the horizon.

    The reference is greedy Monte Carlo over the same candidate pool, not a true
    optimum: exact k-subset IM is NP-hard, and greedy-MC is the (1-1/e) benchmark
    the literature reports against. `budget_regret_norm` is the honest headline —
    the fraction of the achievable spread the model's seed set gives up.
    """
    rng = np.random.default_rng(seed)
    num_nodes = store_entry["num_nodes"]
    edge_index = store_entry["edge_index"]

    graph_input = build_graph_input(
        edge_index,
        store_entry["ic_probs"],
        num_nodes,
        diffusion_model,
        device,
        hide_edge_weights,
    )
    degrees = np.zeros(num_nodes)
    np.add.at(degrees, edge_index[0], 1)
    np.add.at(degrees, edge_index[1], 1)

    k = min(k, num_nodes)
    candidates = sorted(
        int(node)
        for node in rng.choice(
            num_nodes, size=min(n_candidates, num_nodes), replace=False
        )
    )

    model.eval()

    def greedy(score) -> list[int]:
        chosen: list[int] = []
        for _ in range(k):
            pool = [node for node in candidates if node not in chosen]
            if not pool:
                break
            chosen.append(max(pool, key=lambda node: score(chosen + [node])))
        return chosen

    model_seeds = greedy(
        lambda seeds: _model_spread(
            model,
            seeds,
            store_entry,
            graph_input,
            diffusion_model,
            device,
            horizon,
            model_samples,
            rng,
            action_encoding,
        )
    )
    greedy_seeds = greedy(
        lambda seeds: _true_spread_set(
            seeds,
            store_entry,
            diffusion_model,
            horizon,
            mc_runs,
            rng,
            remove_semantics,
        )
    )
    degree_seeds = sorted(candidates, key=lambda node: -degrees[node])[:k]
    random_seeds = [
        int(node) for node in rng.choice(candidates, size=k, replace=False)
    ]

    def truth(seeds: list[int]) -> float:
        # A larger mc_runs for scoring than for search: the search only needs the
        # ranking, the reported number is the result
        return _true_spread_set(
            seeds,
            store_entry,
            diffusion_model,
            horizon,
            mc_runs * 4,
            rng,
            remove_semantics,
        )

    spreads = {
        "model": truth(model_seeds),
        "greedy_mc": truth(greedy_seeds),
        "degree": truth(degree_seeds),
        "random": truth(random_seeds),
    }
    reference = spreads["greedy_mc"]

    results = {f"budget_spread_{name}": value for name, value in spreads.items()}
    results.update(
        {
            "budget_k": float(k),
            "budget_horizon": float(horizon),
            "budget_regret_model": reference - spreads["model"],
            "budget_regret_degree": reference - spreads["degree"],
            "budget_regret_random": reference - spreads["random"],
            # Fraction of the achievable spread given up. Scale-free, so it is
            # comparable across graphs and k, which the raw regret is not.
            "budget_regret_norm": (
                (reference - spreads["model"]) / reference if reference > 0 else 0.0
            ),
            "budget_seed_overlap": (
                len(set(model_seeds) & set(greedy_seeds)) / max(len(greedy_seeds), 1)
            ),
        }
    )

    return results


# Graph-selection modes for the planning evaluators.
#
#   test     score ONLY graphs whose transitions live in the test split. The
#            correct default: `store` holds every graph the dataset was built
#            from, so taking the first n of them scores the model on graphs it
#            trained on, and the resulting regret is not a held-out number.
#
#   legacy   the historical `list(store)[:n]`, which mixes splits. Kept so a
#            pre-2026-08-22 planning number can be reproduced, and named so it
#            cannot be selected by accident.
planning_split_test = "test"
planning_split_legacy = "legacy"
valid_planning_splits = (planning_split_test, planning_split_legacy)


@torch.inference_mode()
def planning_regret_multi(
    model: nn.Module,
    store: dict[str, dict],
    diffusion_model: str,
    device: torch.device,
    n_graphs: int = 5,
    n_states: int = 20,
    n_candidates: int = 20,
    mc_runs: int = 8,
    seed: int = 0,
    threshold: float = 0.5,
    hide_edge_weights: bool = False,
    remove_semantics: str = spent,
    action_encoding: str = basic_encoding,
    out_dir: str | Path | None = None,
    planning_split: str = planning_split_test,
) -> dict[str, float]:
    """
    Average planning regret over held-out graphs, with a cross-graph std.

    Single-graph planning regret ties across backbones because the per-graph
    argmax choice is coarse (most models pick the same candidate). Averaging over
    several graphs: each with its own seed offset so the sampled states differ,
    gives the metric real resolution, plus a cross-graph std as an error bar.

    The graphs come from the TEST split. Scoring `list(store)[:n]` instead — what
    this did before — puts training graphs in a metric that is read as evidence
    the model plans well on graphs it has not seen.
    """
    graph_ids, provenance = select_planning_graphs(
        store, n_graphs, out_dir, diffusion_model, planning_split
    )

    if not graph_ids:
        raise ValueError("planning_regret_multi: empty graph store")

    per_graph = [
        planning_regret(
            model,
            store[graph_id],
            diffusion_model,
            device,
            n_states=n_states,
            n_candidates=n_candidates,
            mc_runs=mc_runs,
            seed=seed + offset,
            threshold=threshold,
            hide_edge_weights=hide_edge_weights,
            remove_semantics=remove_semantics,
            action_encoding=action_encoding,
        )
        for offset, graph_id in enumerate(graph_ids)
    ]

    results = {"plan_n_graphs": float(len(per_graph))}
    for key in ("plan_regret_model", "plan_regret_random", "plan_regret_degree"):
        values = np.array(
            [graph_result[key] for graph_result in per_graph], dtype=np.float64
        )
        results[key] = float(values.mean())
        results[f"{key}_std"] = float(values.std())

    results.update(provenance)

    return results


@torch.inference_mode()
def planning_regret_budget_multi(
    model: nn.Module,
    store: dict[str, dict],
    diffusion_model: str,
    device: torch.device,
    n_graphs: int = 3,
    seed: int = 0,
    out_dir: str | Path | None = None,
    planning_split: str = planning_split_test,
    **kwargs,
) -> dict[str, float]:
    """
    Average `planning_regret_budget` over held-out graphs, with a std.

    `out_dir` is the dataset directory; without it the selection falls back to
    `legacy` and says so in the provenance, because split membership cannot be
    determined from the store alone.
    """
    graph_ids, provenance = select_planning_graphs(
        store, n_graphs, out_dir, diffusion_model, planning_split
    )

    if not graph_ids:
        raise ValueError("planning_regret_budget_multi: empty graph store")

    per_graph = [
        planning_regret_budget(
            model,
            store[graph_id],
            diffusion_model,
            device,
            seed=seed + offset,
            **kwargs,
        )
        for offset, graph_id in enumerate(graph_ids)
    ]

    results = {"budget_n_graphs": float(len(per_graph))}
    for key in per_graph[0]:
        values = np.array(
            [graph_result[key] for graph_result in per_graph], dtype=np.float64
        )
        results[key] = float(values.mean())
        results[f"{key}_std"] = float(values.std())

    results.update(provenance)

    return results
