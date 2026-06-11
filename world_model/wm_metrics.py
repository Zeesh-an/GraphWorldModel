"""Hongji metric suite + persistence baseline for the world model."""

import numpy as np


def binary_f1(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()
    if tp + fp + fn == 0:
        return 1.0  # nothing to predict, nothing predicted
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


def new_infection_f1(
    pred_inf: np.ndarray,
    y_inf: np.ndarray,
    infected_t: np.ndarray,
) -> float:
    # restrict to nodes that were susceptible (not yet infected) at time t
    sus = infected_t.astype(bool) == False  # noqa: E712 — intentional element-wise
    return binary_f1(pred_inf.astype(bool) & sus, y_inf.astype(bool) & sus)


def delta_f1(
    pred_inf: np.ndarray,
    y_inf: np.ndarray,
    infected_t: np.ndarray,
) -> float:
    # restrict to nodes whose state actually changed between t and t+1
    changed = y_inf.astype(bool) != infected_t.astype(bool)
    pred_changed = pred_inf.astype(bool) != infected_t.astype(bool)
    return binary_f1(pred_changed, changed)


def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    return float((pred.astype(bool) == target.astype(bool)).mean())


def score_predictions(
    pred_inf: np.ndarray,
    pred_fr: np.ndarray,
    y_inf: np.ndarray,
    y_fr: np.ndarray,
    infected_t: np.ndarray,
    frontier_t: np.ndarray,
) -> dict[str, float]:
    """Compute all metrics. All per-node arrays are concatenated across the eval set."""
    return {
        "infected_acc": accuracy(pred_inf, y_inf),
        "frontier_acc": accuracy(pred_fr, y_fr),
        "new_infection_f1": new_infection_f1(pred_inf, y_inf, infected_t),
        "delta_f1": delta_f1(pred_inf, y_inf, infected_t),
    }


def persistence_baseline(
    y_inf: np.ndarray,
    y_fr: np.ndarray,
    infected_t: np.ndarray,
    frontier_t: np.ndarray,
) -> dict[str, float]:
    """Predict next = current (Hongji's reference baseline)."""
    return score_predictions(
        infected_t, frontier_t, y_inf, y_fr, infected_t, frontier_t
    )
