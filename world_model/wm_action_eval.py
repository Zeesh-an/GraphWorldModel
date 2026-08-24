"""
Does the model actually understand the action? — the falsifiable suite.

`action_sensitivity` (wm_eval) only counts how many DISTINCT outputs a state
produces across its counterfactual actions. A model that reacts to the action
arbitrarily scores exactly as well as one that reacts correctly, and a model that
reacts correctly but sub-threshold can score 0. It answers "did the output move",
never "did it move the right way".

This module answers the second question three ways, in increasing strength:

1. `counterfactual_effect` — the data already contains, for the same (graph,
   episode, t, state), several actions each with their own MC marginal. The true
   causal effect of swapping action a for a' is therefore KNOWN:
   d_true = y(a) - y(a'). Correlate it against d_pred = f(s,a) - f(s,a'). This is
   the direct measurement, and its null is unambiguous: a model that ignores the
   action has d_pred == 0, giving corr = 0 and effect_mae_norm = 1.

2. `action_ablation` — re-score the identical states with the action channels
   zeroed, and with the actions shuffled between records. If the metrics do not
   degrade, the action was not being used. This catches the case where a model
   scores well on the marginals purely from the state and the graph.

3. `exogenous_fidelity` — the deterministic part of the transition, T_exo, has a
   closed form: a seeded node IS infected next step, a `blocked` node is NOT. Check
   the model reproduces it exactly rather than approximately.

Nothing here is a training target. All three read the same test split the one-step
metrics do.
"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from world_model.wm_data import TransitionDataset, collate_transitions
from world_model.wm_eval import evaluate_one_step

# Probability mass below which a counterfactual pair is considered to have no true
# effect at all. MC marginals estimated from `--mc-marginals` draws carry sampling
# noise of order 1/sqrt(draws); pairs whose true effect is entirely inside that
# noise would otherwise dominate the correlation with pure noise-vs-noise.
min_true_effect = 1e-3

# Node ops only. Swapping in an edge op without replaying it onto the adjacency
# would score the model against a graph its own input contradicts.
node_ops = ("add_node", "remove_node")


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 2:
        return 0.0

    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = np.linalg.norm(first_centered) * np.linalg.norm(second_centered)

    if denominator == 0:
        return 0.0

    return float(np.dot(first_centered, second_centered) / denominator)


@torch.inference_mode()
def _predict_all(
    model: nn.Module,
    dataset: TransitionDataset,
    diffusion_model: str,
    device: torch.device,
    hide_edge_weights: bool,
) -> list[np.ndarray]:
    """Per-record predicted next-infected marginal, in dataset order."""
    model.eval()
    predictions = []

    for index in range(len(dataset)):
        batch = collate_transitions(
            [dataset[index]], diffusion_model, device, hide_edge_weights
        )
        logits = model(batch["X"], batch["graph"])
        predictions.append(torch.sigmoid(logits[:, 0]).cpu().numpy())

    return predictions


def counterfactual_effect(
    model: nn.Module,
    dataset: TransitionDataset,
    diffusion_model: str,
    device: torch.device,
    hide_edge_weights: bool = False,
) -> dict:
    """
    Correlate the PREDICTED causal effect of an action swap with the TRUE one.

    For every (graph, episode, t, state) group holding >= 2 distinct actions, form
    all pairs (a, a') and compare
        d_pred = f(s,a) - f(s,a')      against      d_true = y_MC(s,a) - y_MC(s,a')
    over the nodes. Both are (N,) vectors of next-infected probability, so this
    scores the whole downstream effect of the action, not just its target node.

    | key                  | reading                                                |
    | -------------------- | ------------------------------------------------------ |
    | `effect_pearson`     | 1 = the effect is predicted exactly; 0 = ignored        |
    | `effect_mae`         | mean per-node |d_pred - d_true|                         |
    | `effect_mae_norm`    | the same, over mean|d_true|. **1.0 = predicts no effect at all**, and is what a state-only model scores. < 1 means the action carries real signal |
    | `effect_sign_agree`  | fraction of nodes with a real effect whose DIRECTION is right (chance = 0.5) |
    | `effect_magnitude_ratio` | mean|d_pred| / mean|d_true|; < 1 under-reacts, > 1 over-reacts |
    | `n_pairs`            | counterfactual pairs found. **0 means this dataset cannot support the test** — regenerate with counterfactual forks |
    """
    predictions = _predict_all(
        model, dataset, diffusion_model, device, hide_edge_weights
    )

    # Group dataset indices by the state they branch from
    groups = defaultdict(dict)
    for index in range(len(dataset)):
        record = dataset.samples[index][0]
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
        groups[state_key][action_key] = index

    predicted_deltas, true_deltas = [], []
    per_op = defaultdict(lambda: {"pred": [], "true": []})
    n_pairs = 0

    for group in groups.values():
        indices = sorted(group.values())
        if len(indices) < 2:
            continue

        for position, first in enumerate(indices):
            for second in indices[position + 1 :]:
                n_pairs += 1
                true_first = dataset[first]["y_inf"].numpy()
                true_second = dataset[second]["y_inf"].numpy()

                delta_true = true_first - true_second
                delta_pred = predictions[first] - predictions[second]

                predicted_deltas.append(delta_pred)
                true_deltas.append(delta_true)

                # Attribute the pair to the ops that differ between the two bags
                ops = {
                    action_op["op"]
                    for index in (first, second)
                    for action_op in dataset.samples[index][0]["action"]
                } or {"null"}
                for op in ops:
                    per_op[op]["pred"].append(delta_pred)
                    per_op[op]["true"].append(delta_true)

    if not predicted_deltas:
        return {
            "n_pairs": 0,
            "effect_pearson": float("nan"),
            "effect_mae": float("nan"),
            "effect_mae_norm": float("nan"),
            "effect_sign_agree": float("nan"),
            "effect_magnitude_ratio": float("nan"),
            "per_op": {},
            "note": (
                "no counterfactual pairs in this split: the dataset carries at most "
                "one action per state, so action-conditioning is untestable on it. "
                "Regenerate with data/generate_wm_data.py counterfactual forks."
            ),
        }

    results = _score_effect(
        np.concatenate(predicted_deltas), np.concatenate(true_deltas)
    )
    results["n_pairs"] = n_pairs
    results["per_op"] = {
        op: _score_effect(
            np.concatenate(values["pred"]), np.concatenate(values["true"])
        )
        for op, values in sorted(per_op.items())
    }

    return results


def _score_effect(delta_pred: np.ndarray, delta_true: np.ndarray) -> dict:
    """Compare a predicted causal-effect vector to the true one."""
    mean_true = float(np.abs(delta_true).mean())
    real = np.abs(delta_true) > min_true_effect

    return {
        "effect_pearson": _pearson(delta_pred, delta_true),
        "effect_mae": float(np.abs(delta_pred - delta_true).mean()),
        # Normalised so the "ignores the action" null is exactly 1.0: a model
        # predicting d_pred = 0 everywhere scores mean|d_true| / mean|d_true|.
        "effect_mae_norm": (
            float(np.abs(delta_pred - delta_true).mean() / mean_true)
            if mean_true > 0
            else float("nan")
        ),
        "effect_sign_agree": (
            float((np.sign(delta_pred[real]) == np.sign(delta_true[real])).mean())
            if real.any()
            else float("nan")
        ),
        "effect_magnitude_ratio": (
            float(np.abs(delta_pred).mean() / mean_true) if mean_true > 0 else float("nan")
        ),
        "n_nodes_scored": int(delta_true.size),
        "n_nodes_with_effect": int(real.sum()),
    }


def action_ablation(
    model: nn.Module,
    dataset: TransitionDataset,
    diffusion_model: str,
    device: torch.device,
    hide_edge_weights: bool = False,
    seed: int = 0,
) -> dict:
    """
    Re-score the same states under corrupted actions. A model that uses the action
    must get WORSE; one that ignores it cannot.

    Two corruptions, both restricted to records whose bag is node-only (an edge op
    cannot be swapped without desynchronising the adjacency, see `_reskin_action`):

    - `null`    — every action dropped. Isolates how much of the score the action
                  was carrying at all.
    - `shuffle` — actions permuted between records, so the same actions are present
                  in the same quantity but attached to the wrong states. This is the
                  stronger test: it holds the action DISTRIBUTION fixed and destroys
                  only the state-action pairing, so a model cannot recover the score
                  from a prior over actions.

    `*_delta_f1_drop` is the headline. **A drop of ~0 is a failed test**: it means
    the reported one-step accuracy is obtainable without reading the action.
    """
    rng = np.random.default_rng(seed)

    swappable = [
        index
        for index in range(len(dataset))
        if all(
            action_op["op"] in node_ops
            for action_op in dataset.samples[index][0]["action"]
        )
    ]
    with_action = [
        index for index in swappable if dataset.samples[index][0]["action"]
    ]

    baseline = evaluate_one_step(
        model, dataset, diffusion_model, device, hide_edge_weights=hide_edge_weights
    )

    results = {
        "n_swappable": len(swappable),
        "n_with_action": len(with_action),
        "baseline_delta_f1": baseline["delta_f1"],
        "baseline_brier_infected": baseline["brier_infected"],
    }

    if not with_action:
        results["note"] = (
            "no node-op actions in this split, so the ablation has nothing to "
            "corrupt; diffusion-only data cannot test action-conditioning"
        )
        return results

    # 1. Drop every action
    nulled = evaluate_one_step(
        model,
        dataset,
        diffusion_model,
        device,
        hide_edge_weights=hide_edge_weights,
        action_override={index: [] for index in with_action},
    )

    # 2. Permute actions between records (derangement where possible, so no record
    #    keeps its own bag and the corruption is not silently a no-op)
    #
    #    With fewer than two acting records there is nothing to permute and the
    #    "shuffle" is the identity. Say so: a drop of 0.0 from a degenerate shuffle
    #    reads identically to a drop of 0.0 from a model that ignores the action,
    #    and only one of those is a finding.
    results["shuffle_testable"] = len(with_action) >= 2
    order = list(with_action)
    permuted = list(rng.permutation(order))
    for position in range(len(order)):
        if permuted[position] == order[position] and len(order) > 1:
            partner = (position + 1) % len(order)
            permuted[position], permuted[partner] = (
                permuted[partner],
                permuted[position],
            )

    shuffled = evaluate_one_step(
        model,
        dataset,
        diffusion_model,
        device,
        hide_edge_weights=hide_edge_weights,
        action_override={
            target: dataset.samples[source][0]["action"]
            for target, source in zip(order, permuted)
        },
    )

    for name, corrupted in (("null", nulled), ("shuffle", shuffled)):
        results[f"{name}_delta_f1"] = corrupted["delta_f1"]
        results[f"{name}_delta_f1_drop"] = baseline["delta_f1"] - corrupted["delta_f1"]
        results[f"{name}_brier_infected"] = corrupted["brier_infected"]
        results[f"{name}_brier_rise"] = (
            corrupted["brier_infected"] - baseline["brier_infected"]
        )

    return results


@torch.inference_mode()
def exogenous_fidelity(
    model: nn.Module,
    dataset: TransitionDataset,
    diffusion_model: str,
    device: torch.device,
    hide_edge_weights: bool = False,
    tolerance: float = 1e-3,
) -> dict:
    """
    T_exo has a closed form. Check the model reproduces it EXACTLY, not on average.

    A seeded node must come out with P(infected) = 1; a node the action removed
    must come out of the frontier. `add_seed_success` already asks a thresholded
    version of the first question — this asks it in probability, which is what the
    rollout actually consumes, and reports the worst case rather than the mean, so
    a single violated node is visible instead of being averaged away.
    """
    model.eval()
    seed_probs, remove_frontier_probs = [], []

    for index in range(len(dataset)):
        record = dataset.samples[index][0]
        ops = {action_op["op"] for action_op in record["action"]}

        if not ops & set(node_ops):
            continue

        batch = collate_transitions(
            [dataset[index]], diffusion_model, device, hide_edge_weights
        )
        probs = torch.sigmoid(model(batch["X"], batch["graph"])).cpu().numpy()

        for action_op in record["action"]:
            target = int(action_op["target"])
            if action_op["op"] == "add_node":
                seed_probs.append(float(probs[target, 0]))
            elif action_op["op"] == "remove_node":
                remove_frontier_probs.append(float(probs[target, 1]))

    def summarise(values: list[float], prefix: str, want: float) -> dict:
        if not values:
            return {f"{prefix}_n": 0}

        array = np.array(values)
        return {
            f"{prefix}_n": int(array.size),
            f"{prefix}_mean": float(array.mean()),
            f"{prefix}_worst": float(array.min() if want == 1.0 else array.max()),
            f"{prefix}_exact_frac": float(
                (np.abs(array - want) <= tolerance).mean()
            ),
        }

    return {
        **summarise(seed_probs, "seeded_p_infected", 1.0),
        **summarise(remove_frontier_probs, "removed_p_frontier", 0.0),
        "tolerance": tolerance,
    }


def action_conditioning_report(
    model: nn.Module,
    dataset: TransitionDataset,
    diffusion_model: str,
    device: torch.device,
    hide_edge_weights: bool = False,
    seed: int = 0,
) -> dict:
    """All three tests plus a single pass/fail verdict, for the results JSON."""
    effect = counterfactual_effect(
        model, dataset, diffusion_model, device, hide_edge_weights
    )
    ablation = action_ablation(
        model, dataset, diffusion_model, device, hide_edge_weights, seed
    )
    exogenous = exogenous_fidelity(
        model, dataset, diffusion_model, device, hide_edge_weights
    )

    # The verdict is deliberately conservative: untestable is NOT a pass.
    effect_ok = (
        effect["n_pairs"] > 0
        and np.isfinite(effect["effect_mae_norm"])
        and effect["effect_mae_norm"] < 1.0
        and effect["effect_pearson"] > 0.0
    )
    ablation_ok = ablation.get("shuffle_testable", False) and (
        ablation.get("shuffle_delta_f1_drop", 0.0) > 0.0
    )

    return {
        "counterfactual_effect": effect,
        "ablation": ablation,
        "exogenous": exogenous,
        "action_conditioned": bool(effect_ok and ablation_ok),
        "verdict": _verdict(effect, ablation, effect_ok, ablation_ok),
    }


def _verdict(effect: dict, ablation: dict, effect_ok: bool, ablation_ok: bool) -> str:
    if effect["n_pairs"] == 0:
        return (
            "UNTESTABLE: the split has no counterfactual pairs, so no claim about "
            "action-conditioning is supported by this run"
        )

    if not ablation.get("n_with_action"):
        return "UNTESTABLE: no node-op actions to ablate in this split"

    if not ablation.get("shuffle_testable", False):
        return (
            f"UNTESTABLE: only {ablation['n_with_action']} record(s) carry a node "
            f"action, so the shuffle has nothing to permute and its 0.0 drop is "
            f"degenerate rather than evidence"
        )

    if effect_ok and ablation_ok:
        return (
            f"PASS: predicted action effects correlate with true ones at "
            f"r={effect['effect_pearson']:.3f} (mae_norm "
            f"{effect['effect_mae_norm']:.3f} < 1.0), and shuffling the actions "
            f"costs {ablation['shuffle_delta_f1_drop']:.4f} delta_f1"
        )

    reasons = []
    if not effect_ok:
        reasons.append(
            f"predicted effects do not track true ones "
            f"(r={effect['effect_pearson']:.3f}, "
            f"mae_norm={effect['effect_mae_norm']:.3f}; >= 1.0 means no better "
            f"than predicting no effect)"
        )
    if not ablation_ok:
        reasons.append(
            f"shuffling the actions costs "
            f"{ablation.get('shuffle_delta_f1_drop', 0.0):.4f} delta_f1, so the "
            f"score does not depend on reading the action"
        )

    return "FAIL: " + "; ".join(reasons)
