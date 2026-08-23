"""One-step eval, action-specific metrics, free-running rollout, planning."""

import json
from pathlib import Path
from collections import defaultdict
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance

from world_model.wm_data import (
    TransitionDataset,
    basic_encoding,
    build_features,
    build_graph_input,
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
    score_predictions,
)

from data.wm_simulator import ActionOp, Simulator, blocked, spent

seed_upper_bound = 1 << 30

# Decimal places the action-sensitivity comparison rounds predicted probabilities
# to. Two outputs that agree to 1e-3 per node ARE the same prediction; anything
# finer would count floating-point noise as a reaction to the action.
sensitivity_decimals = 3


def _cat_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(arrays) if arrays else np.zeros(0)


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

    return {
        "X": torch.from_numpy(X),
        "y_inf": torch.from_numpy(y_inf),
        "y_fr": torch.from_numpy(y_fr),
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
    """
    model.eval()
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
        current_infected_parts.append(item["X"][:, ch_infected].numpy())
        current_frontier_parts.append(item["X"][:, ch_frontier].numpy())

        record = item["record"]
        for action_op in record["action"]:
            if action_op["op"] == "add_node":
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
        # Under a seeding task the two agree — seeding A flips A itself, a
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
    action_policy: "ActionPolicy | None" = None,
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


@torch.inference_mode()
def planning_regret_budget_multi(
    model: nn.Module,
    store: dict[str, dict],
    diffusion_model: str,
    device: torch.device,
    n_graphs: int = 3,
    seed: int = 0,
    **kwargs,
) -> dict[str, float]:
    """Average `planning_regret_budget` over the first n_graphs graphs, with a std."""
    graph_ids = list(store)[:n_graphs]

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

    return results


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
) -> dict[str, float]:
    """
    Average planning regret over the first n_graphs graphs in the store.

    Single-graph planning regret ties across backbones because the per-graph
    argmax choice is coarse (most models pick the same candidate). Averaging over
    several graphs — each with its own seed offset so the sampled states differ —
    gives the metric real resolution, plus a cross-graph std as an error bar.
    """
    graph_ids = list(store)[:n_graphs]
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

    return results
