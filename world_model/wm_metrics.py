"""
Metric suite + persistence baseline for the world model, plus the per-task metric
primitives an arm is described by.

Three blocks, answering three different questions; confusing them is how a table
ends up reporting one thing under another's name.

  * Everything above `roc_auc` scores the LEARNED TRANSITION against the
    simulator — one-step forward prediction.
  * `roc_auc` through `resimulation_error` score an INVERSE prediction: a
    recovered source set against the true one, in the PR / RE / F1 / AUC form the
    source-localization literature reports (`research/source_localization.md` §8.1).
  * Everything from `gcc_threshold` down is computed exactly by BFS on the
    residual graph and is never a training target —
    `research/critical_node_detection.md` §2.1 argues at length that a k-layer
    message-passing model provably cannot represent giant-component membership on
    a diameter-46 graph, and §8.3 therefore puts the connectivity functionals in
    the report as descriptive context beside the diffusion number that the arm was
    actually optimized for. The epidemic block (`spectral_radius`,
    `epidemic_curve_metrics`, `immunization_metrics`) sits in the same category and
    carries the same warning from the other side: `research/epidemic_control.md`
    §8.2 trap 1 records that a method can win on eigendrop and lose on simulated
    final size, so the eigendrop is reported BESIDE the attack rate rather than as
    the score.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def binary_f1(pred: np.ndarray, target: np.ndarray) -> float:
    # Compute the binary F1 between the pred and target node states
    # F1 = (2 * TP) / (2 * TP + FP + FN)
    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()

    if tp + fp + fn == 0:
        return 1.0  # nothing to predict, nothing predicted

    denominator = 2 * tp + fp + fn

    return float(2 * tp / denominator) if denominator > 0 else 0.0


def new_infection_f1(
    pred_infected: np.ndarray,
    y_inf: np.ndarray,
    infected_t: np.ndarray,
) -> float:
    # Restrict to nodes that were susceptible (not yet infected) at time t, and then predict the binary F1
    susceptible = ~infected_t.astype(bool)
    return binary_f1(
        pred_infected.astype(bool) & susceptible, y_inf.astype(bool) & susceptible
    )


def delta_f1(
    pred_infected: np.ndarray,
    y_inf: np.ndarray,
    infected_t: np.ndarray,
) -> float:
    # Restrict to nodes whose state actually changed between t and t + 1, and then predict the binary F1
    changed = y_inf.astype(bool) != infected_t.astype(bool)
    pred_changed = pred_infected.astype(bool) != infected_t.astype(bool)
    return binary_f1(pred_changed, changed)


def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    return float((pred.astype(bool) == target.astype(bool)).mean())


def brier_score(prob: np.ndarray, target: np.ndarray) -> float:
    # Mean squared error between predicted probability and the (soft) target marginal — lower = better calibrated
    if prob.size == 0:
        return 0.0

    return float(np.mean((prob.astype(np.float64) - target.astype(np.float64)) ** 2))


def score_predictions(
    pred_infected: np.ndarray,
    pred_frontier: np.ndarray,
    y_inf: np.ndarray,
    y_fr: np.ndarray,
    infected_t: np.ndarray,
    frontier_t: np.ndarray,
) -> dict[str, float]:
    """Compute all metrics. All per-node arrays are concatenated across the eval set."""
    return {
        "infected_acc": accuracy(pred_infected, y_inf),
        "frontier_acc": accuracy(pred_frontier, y_fr),
        "new_infection_f1": new_infection_f1(pred_infected, y_inf, infected_t),
        "delta_f1": delta_f1(pred_infected, y_inf, infected_t),
    }


def persistence_baseline(
    y_inf: np.ndarray,
    y_fr: np.ndarray,
    infected_t: np.ndarray,
    frontier_t: np.ndarray,
) -> dict[str, float]:
    """Predict next = current."""
    return score_predictions(
        infected_t, frontier_t, y_inf, y_fr, infected_t, frontier_t
    )


def roc_auc(scores: np.ndarray, target: np.ndarray) -> float:
    """
    ROC AUC by the Mann-Whitney rank statistic, with proper tie handling.

    Ties matter more here than anywhere else in this file. Source localization is
    scored over ALL of V with a positive class of 1-10% of nodes, and an arm that
    supplies no per-node scores gets a rank-derived vector in which every
    un-nominated node ties — so an implementation that broke ties arbitrarily
    would report a number that depended on node ordering.
    """
    scores = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(target).astype(bool)
    n_positive = int(positive.sum())
    n_negative = int(scores.size - n_positive)

    if n_positive == 0 or n_negative == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)

    # Average the ranks inside each tie block, which is what makes a fully tied
    # score vector score exactly 0.5 instead of whatever the sort happened to do
    sorted_scores = scores[order]
    start = 0
    for position in range(1, scores.size + 1):
        if position == scores.size or sorted_scores[position] != sorted_scores[start]:
            if position - start > 1:
                ranks[order[start:position]] = ranks[order[start:position]].mean()
            start = position

    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


def localization_metrics(
    predicted: list[int],
    sources: list[int],
    num_nodes: int,
    scores: np.ndarray | None = None,
) -> dict[str, float]:
    """
    PR / RE / F1 / AUC / ACC for one recovered source set — the SL literature's set.

    Node-level binary classification over V with the source set as the positive
    class (research/source_localization.md §8.1). Four notes on the columns, each
    one a trap that file names explicitly:

      * **F1 is the headline.** SL-VAE calls it "the most commonly used", IVGD "the
        most important metric for performance evaluation" [both verified].
      * **ACC is near-useless alone.** IVGD's Table 3 has GCNSI at `ACC 0.8840` with
        `F1 0.0218` on Network Science, because sources are a tiny minority class.
        It is reported only beside F1.
      * **AUC needs a RANKING, not a set.** When `scores` is None the ranking is
        derived from the returned list's order, which leaves every un-nominated
        node tied — a real, monotone number, but not the continuous AUC SL-VAE
        reports. Which rule an arm used is recorded per result.
      * **`RE` is an overloaded column name in this literature** — Recall in
        SL-Diff and SIDSL, re-simulated error elsewhere (§8.1). Here `recall` is
        recall, and the re-simulated error has its own key.
    """
    truth = np.zeros(num_nodes, dtype=bool)
    truth[list(sources)] = True

    prediction = np.zeros(num_nodes, dtype=bool)
    prediction[list(predicted)] = True

    true_positive = int(np.logical_and(prediction, truth).sum())
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(sources) if len(sources) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    if scores is None:
        # Rank-derived: 1/(rank + 1) down the returned list, 0 for everything else
        scores = np.zeros(num_nodes, dtype=np.float64)
        for rank, node in enumerate(predicted):
            scores[int(node)] = 1.0 / (rank + 1.0)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((prediction == truth).mean()),
        "auc": roc_auc(np.asarray(scores, dtype=np.float64), truth),
        "n_predicted": float(len(predicted)),
        "n_sources": float(len(sources)),
        "true_positive": float(true_positive),
    }


def resimulation_error(predicted_marginal: np.ndarray, observation: np.ndarray) -> float:
    """
    Mean squared error between re-simulating the RECOVERED sources and the observed y.

    The metric this literature should report and does not: §11 records that no
    surveyed paper reports a genuine re-simulated error, so this column is
    self-contained and must not be presented as a cross-paper comparison. It is
    also NOT the outer loop's reward — diffusion is many-to-one, so a program that
    systematically recovers the wrong member of an equivalence class scores well
    here and badly on F1, which is exactly why §2.3.3 selects on F1.
    """
    predicted_marginal = np.asarray(predicted_marginal, dtype=np.float64)
    observation = np.asarray(observation, dtype=np.float64)

    return float(np.mean((predicted_marginal - observation) ** 2))


def matthews_corrcoef(predicted: np.ndarray, truth: np.ndarray) -> float:
    """
    MCC over a binary node labelling — Rozenshtein KDD'16's ONLY reported measure.

    Kept beside F1 rather than instead of it: MCC uses the true negatives, so on a
    cascade that reached 5% of the graph it is far less flattering than accuracy
    and far more stable than F1 when the predicted set is nearly empty
    (research/cascade_reconstruction.md §8.1).
    """
    predicted = np.asarray(predicted).astype(bool)
    truth = np.asarray(truth).astype(bool)

    tp = float(np.logical_and(predicted, truth).sum())
    tn = float(np.logical_and(~predicted, ~truth).sum())
    fp = float(np.logical_and(predicted, ~truth).sum())
    fn = float(np.logical_and(~predicted, truth).sum())

    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return float((tp * tn - fp * fn) / denominator) if denominator > 0 else 0.0


def _prf(true_positive: int, predicted: int, actual: int) -> tuple[float, float, float]:
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / actual if actual else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return float(precision), float(recall), float(f1)


def reconstruction_metrics(
    predicted: dict[int, tuple[int, int | None]],
    true_times: dict[int, int],
    true_parents: dict[int, list[int]] | None,
    num_nodes: int,
    horizon: int,
) -> dict[str, float]:
    """
    The metric suite for one reconstructed trajectory — §8.1's whole table.

    `predicted` and the two truth maps are `node -> activation step` and
    `node -> (activation step, inferred parent)`; a node absent from either was
    never infected. Three blocks, and confusing them is how a paper reports the
    easy half under the hard half's name:

      * **NODE level** (`node_f1`, `mcc`) — the infected SET, ignoring time. Zong
        ICDM'12 reports `prec_v = 100%` here [verified] and DIPT's best path
        precision thirteen years later is `0.680`, which is the same message twice:
        THE NODE SET IS EASY.
      * **EVENT level** (`event_f1`) — the `(node, t)` pairs, so a node recovered
        at the wrong step is a miss. DITTO's `F1` column.
      * **TREE level** (`path_precision`, `jaccard`, `order_accuracy`) — the
        who-infected-whom edges. DIPT's `Path Precision` + `Jaccard Index`. This is
        the hard half, and §2.6 is why it must carry the outer loop's reward:
        rewarded on the node set alone, a program SEARCH discovers the tree
        contributes nothing to its score and converges on decoders that never
        attempt it.

    `true_parents` is None when the dataset was generated without `--trace-parents`
    — NDlib does not emit a transmission edge — and every tree column is then NaN
    rather than 0.0, so a missing capability never reads as a failed one.

    A parent set (LT, where activation is a threshold crossing over the whole
    active neighbourhood and there is no single transmitting edge, §2.6) counts a
    predicted parent as correct if it is IN the set. That is the honest reading and
    it makes LT's path precision structurally easier than IC's; the two are never
    compared.
    """
    truth_infected = np.zeros(num_nodes, dtype=bool)
    truth_infected[list(true_times)] = True
    predicted_infected = np.zeros(num_nodes, dtype=bool)
    predicted_infected[list(predicted)] = True

    node_tp = int(np.logical_and(predicted_infected, truth_infected).sum())
    node_precision, node_recall, node_f1 = _prf(
        node_tp, int(predicted_infected.sum()), int(truth_infected.sum())
    )

    # An EVENT is a (node, step) pair, so a node recovered one step late is a
    # false positive AND a false negative rather than a hit
    event_tp = sum(
        1
        for node, (time, _) in predicted.items()
        if node in true_times and int(time) == int(true_times[node])
    )
    event_precision, event_recall, event_f1 = _prf(
        event_tp, len(predicted), len(true_times)
    )

    # Hitting times, with an un-infected node's time pinned at the horizon: that
    # is DITTO's own convention and it is what makes the normalizer 2n(T+1)^2
    horizon = max(int(horizon), 1)
    predicted_hitting = np.full(num_nodes, float(horizon), dtype=np.float64)
    true_hitting = np.full(num_nodes, float(horizon), dtype=np.float64)

    for node, (time, _) in predicted.items():
        predicted_hitting[int(node)] = min(float(time), float(horizon))

    for node, time in true_times.items():
        true_hitting[int(node)] = min(float(time), float(horizon))

    squared_error = float(np.sum((predicted_hitting - true_hitting) ** 2))
    metrics = {
        "node_precision": node_precision,
        "node_recall": node_recall,
        "node_f1": node_f1,
        "event_precision": event_precision,
        "event_recall": event_recall,
        "event_f1": event_f1,
        "mcc": matthews_corrcoef(predicted_infected, truth_infected),
        "time_mae": float(np.mean(np.abs(predicted_hitting - true_hitting))),
        # DITTO normalizes by 2n(T+1)^2 under the root [verified, §5.1]
        "time_nrmse": float(
            np.sqrt(squared_error / (2.0 * num_nodes * (horizon + 1) ** 2))
        ),
        "n_predicted": float(len(predicted)),
        "n_true": float(len(true_times)),
    }

    # The recovered SOURCES fall out for free: the nodes a decoder gave no parent
    # ARE its seed-set estimate, which is why §2.5.1 calls source localization the
    # projection of this task rather than a sibling of it
    predicted_sources = {node for node, (_, parent) in predicted.items() if parent is None}
    true_sources = {node for node, time in true_times.items() if int(time) == 0}
    source_precision, source_recall, source_f1 = _prf(
        len(predicted_sources & true_sources),
        len(predicted_sources),
        len(true_sources),
    )
    metrics |= {
        "source_precision": source_precision,
        "source_recall": source_recall,
        "source_f1": source_f1,
    }

    if true_parents is None:
        metrics |= {
            "path_precision": float("nan"),
            "path_recall": float("nan"),
            "jaccard": float("nan"),
            "order_accuracy": float("nan"),
            "n_tree_edges": 0.0,
        }

        return metrics

    predicted_edges = {
        (int(parent), int(node))
        for node, (_, parent) in predicted.items()
        if parent is not None
    }
    # One entry per non-source node in the truth. Under LT the value is a SET and
    # membership is what `correct` tests, so the unit counted is the node.
    true_causes = {
        int(node): {int(cause) for cause in causes}
        for node, causes in true_parents.items()
        if causes
    }
    correct = sum(
        1 for parent, node in predicted_edges if parent in true_causes.get(node, ())
    )
    path_precision, path_recall, _ = _prf(
        correct, len(predicted_edges), len(true_causes)
    )
    union = len(predicted_edges) + len(true_causes) - correct

    # Xiao SDM'18's order accuracy: does each inferred edge respect the times the
    # decoder itself assigned? It checks ORDERING only and is blind to absolute
    # times, which is why it is reported beside NRMSE rather than instead of it.
    #
    # Warning: it reads 1.0 for every LIBRARY decoder by construction, because
    # `reconstruction_algorithms.finalize` only ever attaches a parent that
    # activated earlier. That is not a bug and not a strong result — it is the
    # column doing its job on a GENERATED decoder, which assigns its own parents
    # and can produce an incoherent tree. Read it as a validity check on synthesis,
    # not as a quality measure across the classical pool.
    ordered = sum(
        1
        for node, (time, parent) in predicted.items()
        if parent is not None
        and parent in predicted
        and predicted[parent][0] <= time
    )

    metrics |= {
        "path_precision": path_precision,
        "path_recall": path_recall,
        "jaccard": float(correct / union) if union else 0.0,
        "order_accuracy": (
            float(ordered / len(predicted_edges)) if predicted_edges else 0.0
        ),
        "n_tree_edges": float(len(predicted_edges)),
    }

    return metrics


# Weight on Path Precision in the outer loop's reward. >= 0.5 is a REQUIREMENT
# rather than a taste (research/cascade_reconstruction.md §2.6): the node set is
# nearly free, so a search rewarded mostly on Event F1 discovers that the tree
# contributes nothing to its score and converges on decoders that do not attempt
# the hard half. The reward is the specification.
default_tree_weight = 0.6


def reconstruction_reward(metrics: dict[str, float], tree_weight: float) -> float:
    """
    `lambda * PathPrecision + (1 - lambda) * EventF1` — §2.6's Score, verbatim.

    Falls back to Event F1 alone when the dataset carries no transmission edge
    (LT-set-valued truth still has one; a dataset generated without
    `--trace-parents` does not). That fallback is the failure mode §2.6 describes,
    so every caller records `tree_weight` alongside the number and the harness
    refuses to run the full search without parents.

    Warning: PathPrecision is a PRECISION, and §2.6 does not say so because no
    published method has a search that could exploit it. A decoder that names three
    transmission edges and gets them right scores 1.0 on the half the reward is
    weighted toward, so UNDER-PREDICTING is a second gaming corner beside the one
    §2.6 names. `path_recall` and `jaccard` are computed for exactly this, and
    `reconstruction.summarize_reconstruction` prints a diagnostic whenever an arm
    names fewer than half the real edges. Keeping the published column as the
    reward and surfacing the hazard beats silently switching to a tree F1 the
    literature does not report.
    """
    path_precision = metrics.get("path_precision", float("nan"))

    if not np.isfinite(path_precision):
        return float(metrics["event_f1"])

    return float(
        tree_weight * path_precision + (1.0 - tree_weight) * metrics["event_f1"]
    )


# Cascade / popularity prediction -------------------------------------------
#
# research/cascade_prediction.md §8.1 opens with "Get MSLE right or nothing else
# matters", and it means three independent choices that all hide inside one name:
# the log BASE (2 in CasFlow and CasFT, natural in CTCP's loss — a constant factor
# of (ln 2)^2 ~ 0.48 between them), the QUANTITY (total `P(t_p)` versus the
# increment `dP`), and the SMOOTHING offset (CasFlow's code clamps to >= 1 and adds
# nothing; CasFT's stated definition adds 1). §5.7 difference 4 records that §5.1
# prints two of those variants in ONE table. So every variant this repo can report
# is computed and named, and the reward names which one it used.

# CasFlow's own `casflow.py`, verbatim:
#     predictions = [1 if p < 1 else p for p in predictions]
#     msle  = mean((log2(pred) - log2(label))^2)
#     mape  = mean(|log2(pred + 1) - log2(label + 1)| / log2(label + 2))
# Note the asymmetry in MAPE — +1 inside the absolute value, +2 in the denominator.
# §8.1 quotes CasFT's caption as `|log2(P+2) - log2(P_hat+2)| / log2(P+2)`, which is
# a THIRD form. Both are computed; `mape` is the code's and `mape_casft` the
# caption's, and a table has to say which.
popularity_floor = 1.0

# WroPerc's epsilon: CoupledGNN counts a cascade "wrong" when its relative error on
# the RAW count exceeds this [verified, §8.1]. The practitioner's metric, and the
# only one here that is not in log space.
wroperc_epsilon = 0.5

# COV-k's k, as a fraction of the scored set. CasFlow reports Coverage at
# `k = floor(N/10)` — "did we find the viral ones" [verified, §8.1].
coverage_fraction = 0.1


def _safe_log2(values: np.ndarray) -> np.ndarray:
    return np.log2(np.maximum(values, popularity_floor))


def popularity_metrics(
    predicted: np.ndarray,
    actual: np.ndarray,
    observed: np.ndarray,
    declined: int = 0,
) -> dict[str, float]:
    """
    Every metric §8.1 tabulates, over one arm's scored cascades.

    `predicted` and `actual` are TOTAL popularities at `t_p`; `observed` is
    `P(t_o)`, so the increment variants are derivable here rather than needing a
    second pass. `declined` is how many cascades the predictor refused to score at
    all, and it is a first-class column for the reason §8.4 gives: generative models
    decline on supercritical cascades (SEISMIC failed on 1,022 of ~20K News cascades
    at 5 minutes), papers report the mean over SCOREABLE cascades only, and that
    "silently favours the model that gives up more often". Mishra et al. publish the
    failure counts; almost nobody else does, so we always do.

    Direction, because §8.4 warns the table mixes them: MSLE / MALE / MAPE / MRSE /
    WroPerc are LOWER-is-better; PCC / R2 / COV-k are HIGHER-is-better.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)

    scored = int(predicted.size)
    if not scored:
        return {
            "msle": float("nan"),
            "n_scored": 0,
            "n_failed": int(declined),
            "decline_rate": 1.0 if declined else 0.0,
        }

    log_predicted = _safe_log2(predicted)
    log_actual = _safe_log2(actual)
    error = log_predicted - log_actual

    # The increment, floored at 0: a progressive cascade cannot shrink, so a
    # prediction below what was already observed is a prediction of negative growth
    predicted_increment = np.maximum(predicted - observed, 0.0)
    actual_increment = np.maximum(actual - observed, 0.0)
    increment_error = _safe_log2(predicted_increment + 1.0) - _safe_log2(
        actual_increment + 1.0
    )

    # Genuinely relative, on RAW counts — CoupledGNN's units, and the only ones here
    # that are not log-space
    relative = (predicted - actual) / np.maximum(actual, 1.0)

    absolute_relative = np.abs(relative)
    variance = float(np.var(log_actual))

    metrics = {
        # CasFlow's own code: clamp to >= 1, log base 2, no offset
        "msle": float(np.mean(error**2)),
        "male": float(np.mean(np.abs(error))),
        # CasFT's stated definition: log2(P + 1)
        "msle_offset": float(
            np.mean((np.log2(predicted + 1.0) - np.log2(actual + 1.0)) ** 2)
        ),
        # CTCP's loss, natural log — a factor of (ln 2)^2 from `msle`, and §8.1's
        # warning that the two are printed under one name
        "msle_natural": float(np.mean((np.log(np.maximum(predicted, popularity_floor))
                                       - np.log(np.maximum(actual, popularity_floor))) ** 2)),
        # CasFlow's code form, then §8.1's caption form
        "mape": float(
            np.mean(
                np.abs(np.log2(predicted + 1.0) - np.log2(actual + 1.0))
                / np.log2(actual + 2.0)
            )
        ),
        "mape_casft": float(
            np.mean(
                np.abs(np.log2(actual + 2.0) - np.log2(predicted + 2.0))
                / np.log2(actual + 2.0)
            )
        ),
        # The increment, which is what CasFlow's own label IS (§5.7 difference 3)
        "msle_increment": float(np.mean(increment_error**2)),
        "male_increment": float(np.mean(np.abs(increment_error))),
        # CoupledGNN's three, on raw counts
        "mrse": float(np.mean(relative**2)),
        "mrse_median": float(np.median(relative**2)),
        "wroperc": float(np.mean(absolute_relative >= wroperc_epsilon)),
        # SEISMIC reports APE as QUANTILES because the mean is outlier-dominated
        # [verified, §5.4]; its own 10-minute row is 71% / 44% / 25%
        "ape_median": float(np.median(absolute_relative)),
        "ape_p75": float(np.percentile(absolute_relative, 75)),
        "ape_p95": float(np.percentile(absolute_relative, 95)),
        # Higher is better from here down
        "pcc": (
            float(np.corrcoef(log_predicted, log_actual)[0, 1])
            if scored > 1 and np.std(log_predicted) > 0 and np.std(log_actual) > 0
            else float("nan")
        ),
        "r2": (
            float(1.0 - np.mean(error**2) / variance) if variance > 0 else float("nan")
        ),
        "n_scored": scored,
        "n_failed": int(declined),
        # The column §8.4 asks for and almost nobody publishes: a mean over
        # scoreable cascades favours whoever gives up more often, so the give-up
        # rate travels beside the mean rather than in a footnote
        "decline_rate": float(declined / max(scored + declined, 1)),
        "mean_predicted": float(np.mean(predicted)),
        "mean_actual": float(np.mean(actual)),
    }

    metrics["coverage"] = coverage_at_k(predicted, actual)

    return metrics


def coverage_at_k(predicted: np.ndarray, actual: np.ndarray) -> float:
    """
    `|top-k predicted ∩ top-k true| / k` at `k = floor(N/10)` — CasFlow's COV-k.

    "Did we find the viral ones", and the one metric here that asks a question MSLE
    cannot: a model can be well calibrated on the bulk and rank the tail wrong,
    which is the outcome §1.4's whole motivating debate (Salganik/Watts vs Cheng et
    al.) is about.
    """
    count = int(len(actual) * coverage_fraction)

    if count < 1:
        return float("nan")

    top_predicted = set(np.argsort(-np.asarray(predicted, dtype=np.float64))[:count])
    top_actual = set(np.argsort(-np.asarray(actual, dtype=np.float64))[:count])

    return float(len(top_predicted & top_actual) / count)


# Which error the outer loop's reward IS. Every one of them MINIMIZES, which is why
# `pipeline.tasks.objective_sense` maps `forecast` onto `minimize` rather than
# reading the objective literally.
valid_prediction_metrics = (
    "msle",
    "male",
    "msle_offset",
    "msle_natural",
    "msle_increment",
    "male_increment",
    "mape",
    "mrse",
    "wroperc",
)
default_prediction_metric = "msle"


def prediction_reward(metrics: dict[str, float], metric: str) -> float:
    """
    The scalar the outer loop minimizes, from the metric block.

    A predictor that declined every cascade produces no metric at all, and returning
    NaN there would make it compare false against everything and quietly win the
    `improves` test on some paths. Infinity is the honest reading: it predicted
    nothing, so it is worse than any prediction.
    """
    if metric not in valid_prediction_metrics:
        raise ValueError(
            f"unknown prediction metric {metric!r}; choose one of "
            f"{valid_prediction_metrics}"
        )

    value = metrics.get(metric, float("nan"))

    return float("inf") if not np.isfinite(value) else float(value)


def doubling_accuracy(
    predicted: np.ndarray, actual: np.ndarray, observed: np.ndarray
) -> float:
    """
    Cheng et al.'s balanced framing: did the cascade at least DOUBLE, and did we say so?

    §1.3 and §5.6: the WWW'14 paper reframed size prediction as "given `k` observed
    reshares, will it reach `2k`" precisely because that holds the base rate at 50%
    and makes accuracy interpretable — their own numbers are 0.795 accuracy / 0.877
    AUC at `k = 5`. Reported rather than optimized, because our reward is MSLE and a
    balanced-accuracy reward would select for a different program entirely.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)

    if not predicted.size:
        return float("nan")

    threshold = 2.0 * observed

    return float(np.mean((predicted >= threshold) == (actual >= threshold)))


# Fraction of N the giant component must fall below for the graph to count as
# dismantled. 0.01 is the Min-Sum / CoreHD / GND convention; the set size is
# steeply sensitive to it near the percolation transition, so it is stated with
# every number (research/critical_node_detection.md §8.2 trap 6).
gcc_threshold = 0.01


def adjacency_sets(edge_index: np.ndarray, num_nodes: int) -> list[set[int]]:
    """Undirected adjacency as sets — every connectivity functional here is undirected."""
    groups = [set() for _ in range(num_nodes)]

    for edge in range(edge_index.shape[1]):
        source = int(edge_index[0, edge])
        target = int(edge_index[1, edge])

        if source != target:
            groups[source].add(target)
            groups[target].add(source)

    return groups


def components(neighbours: list[set[int]], removed: set[int]) -> list[int]:
    """Component SIZES of the residual graph G[V \\ removed], by BFS."""
    seen = set(removed)
    sizes = []

    for start in range(len(neighbours)):
        if start in seen:
            continue

        size = 1
        seen.add(start)
        queue = [start]

        while queue:
            node = queue.pop()
            for neighbour in neighbours[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    size += 1
                    queue.append(neighbour)

        sizes.append(size)

    return sizes


def connectivity_profile(
    neighbours: list[set[int]], removed: set[int]
) -> dict[str, float]:
    """
    Objectives 1, 2 and 4 of research/critical_node_detection.md §1.1, computed exactly.

    pairwise_conn is THE canonical CNP objective (Arulselvan et al. 2009);
    largest_cc_size is MinMaxC / the dismantling curve's y-axis; n_components is
    MaxNum CNP. All three come from one BFS pass, so reporting all three costs
    nothing over reporting one.
    """
    sizes = components(neighbours, removed)
    num_nodes = len(neighbours)

    return {
        "pairwise_conn": float(sum(size * (size - 1) // 2 for size in sizes)),
        "largest_cc_size": float(max(sizes, default=0)),
        "gcc_fraction": float(max(sizes, default=0) / num_nodes) if num_nodes else 0.0,
        "n_components": float(len(sizes)),
    }


def dismantling_curve(
    neighbours: list[set[int]], order: list[int]
) -> list[float]:
    """
    s(q) = |GCC(G - S_q)| / N after each of the removals in `order`, sequentially.

    This is objective 6 (§1.1): the physics branch removes nodes ONE AT A TIME and
    recomputes the residual graph after each, which is a strictly stronger setting
    than the OR branch's one-batch evaluation at equal k and produces numbers that
    are NOT interconvertible with it (§8.2 trap 1). Index 0 is the intact graph, so
    a length-k order yields k + 1 entries.
    """
    num_nodes = len(neighbours)
    removed = set()
    curve = [float(max(components(neighbours, removed), default=0)) / max(num_nodes, 1)]

    for node in order:
        removed.add(int(node))
        curve.append(
            float(max(components(neighbours, removed), default=0)) / max(num_nodes, 1)
        )

    return curve


def schneider_r(curve: list[float], num_nodes: int) -> float:
    """
    Schneider's robustness integral R = (1/N) * sum_q s(q) over the removals made.

    Warning: Schneider et al. define R over the FULL sweep q = 1..N, which bounds it in
    [1/N, 0.5]. A budgeted arm removes only k nodes, so this is the partial
    integral over the prefix that was actually removed and is comparable ACROSS
    ARMS at the same k, never against a published full-sweep R. Lower = a better
    attack.
    """
    if num_nodes <= 0 or len(curve) <= 1:
        return 0.0

    return float(sum(curve[1:]) / num_nodes)


def accumulated_normalized_connectivity(
    values: list[float], initial: float
) -> float:
    """
    FINDER's ANC over one connectivity measure: mean of sigma(G - S_q) / sigma(G).

    Parametric in sigma on purpose — FINDER instantiates pairwise connectivity,
    GCC size and component count, and the three are different scales. A paper
    quoting "ANC" without naming sigma is unusable as a baseline (§8.2 trap 3), so
    every caller here records which sigma it passed.
    """
    if not values or initial <= 0:
        return 0.0

    return float(np.mean([value / initial for value in values]))


def dismantling_set_size(
    curve: list[float], threshold: float = gcc_threshold
) -> float | None:
    """
    rho: removals needed to push the GCC below `threshold * N`, as a fraction of N.

    None when the budget ran out before the threshold was crossed, which is the
    common case for us — our budgets top out at 20% of N and the published rho for
    a hard graph exceeds that. Reporting None is the honest answer; reporting the
    budget would read as "dismantled at k" when it was not.
    """
    for index, value in enumerate(curve):
        if value <= threshold:
            return float(index) / max(len(curve) - 1, 1)

    return None


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    """Rank correlation, with an average-rank tie correction."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)

    if first.size < 2:
        return 0.0

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values)
        result = np.empty(values.size, dtype=np.float64)
        result[order] = np.arange(values.size, dtype=np.float64)

        # Average rank within each tie group, so a constant vector does not
        # manufacture a correlation out of argsort's arbitrary order
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        sums = np.zeros(counts.size, dtype=np.float64)
        np.add.at(sums, inverse, result)

        return (sums / counts)[inverse]

    first_ranks, second_ranks = ranks(first), ranks(second)
    first_centered = first_ranks - first_ranks.mean()
    second_centered = second_ranks - second_ranks.mean()
    denominator = np.linalg.norm(first_centered) * np.linalg.norm(second_centered)

    if denominator == 0:
        return 0.0

    return float(np.dot(first_centered, second_centered) / denominator)


def spectral_radius(
    edge_index: np.ndarray,
    num_nodes: int,
    removed: tuple | list = (),
) -> float:
    """
    `lambda_1(A)` of the undirected adjacency after deleting `removed`.

    The one number the entire epidemic-control literature agrees on
    (research/epidemic_control.md §1.2): for essentially every propagation model,
    the epidemic dies out iff `lambda_1 * beta / delta < 1`, which is what
    NetShield, NetMelt, Gelling, GreedyWalk and Preciado are all actually
    optimizing. It needs no simulator at all, which is exactly §9.9's point —
    computing it costs one `eigsh` call on a graph we already hold, and it is the
    only bridge between our simulated-outbreak table and the spectral line's.

    Warning: it is a SURROGATE and §8.2 trap 1 is emphatic that reporting only the
    eigendrop grades us on the quantity the classical methods were built to
    optimize — a comparison we cannot win and that does not test the world model.
    It is reported BESIDE the simulated attack rate, never instead of it.
    """
    if num_nodes <= 0:
        return 0.0

    dropped = {int(node) for node in removed}
    sources, targets = [], []

    for edge in range(edge_index.shape[1]):
        source = int(edge_index[0, edge])
        target = int(edge_index[1, edge])

        if source == target or source in dropped or target in dropped:
            continue

        sources += [source, target]
        targets += [target, source]

    if not sources:
        return 0.0

    adjacency = sp.csr_matrix(
        (np.ones(len(sources), dtype=np.float64), (sources, targets)),
        shape=(num_nodes, num_nodes),
    )
    # Symmetrized above, so duplicate arcs of an undirected graph would count twice
    adjacency.data[:] = 1.0

    # eigsh needs k < n, and a graph small enough for the dense path is small
    # enough that the dense path is faster anyway
    if num_nodes <= 3:
        return float(np.abs(np.linalg.eigvalsh(adjacency.toarray())).max())

    try:
        values = spla.eigsh(
            adjacency, k=1, which="LA", return_eigenvectors=False, maxiter=5000
        )
        return float(values[0])
    except spla.ArpackNoConvergence as error:
        # ARPACK stalls on graphs with a near-degenerate top pair; its partial
        # result is still the best estimate available and is better than a
        # missing column
        return float(error.eigenvalues.max()) if error.eigenvalues.size else 0.0


def epidemic_curve_metrics(
    prevalence: list[float],
    num_nodes: int,
    burn_in: float = 0.5,
) -> dict:
    """
    The SHAPE of an outbreak, from its `|I(t)|` curve — §2.6's four new quantities.

    The attack rate says how many were infected; none of these do, and §8.3 lists
    them because flattening a curve without shrinking its integral is precisely
    what an epidemic-control policy is judged on:

      * **peak prevalence** `max_t |I(t)| / N` — the health-system-capacity metric,
        and the one "flatten the curve" names.
      * **time to peak** `argmax_t |I(t)|` — a policy that delays the peak buys
        response time even when it saves nobody.
      * **AUC** `sum_t |I(t)|` — the integrated load.
      * **endemic prevalence** — the time average after burn-in, which is the ONLY
        one of the four that is defined for SIS. §8.2 trap 6: SIS has no terminal
        state, so final size is undefined there and every rollout metric that
        assumes one is silently wrong.
    """
    curve = [float(value) for value in prevalence]

    if not curve or num_nodes <= 0:
        return {
            "peak_prevalence": 0.0,
            "peak_prevalence_pct": 0.0,
            "time_to_peak": 0,
            "auc_infectious": 0.0,
            "auc_infectious_pct": 0.0,
            "endemic_prevalence": 0.0,
            "endemic_prevalence_pct": 0.0,
        }

    peak = max(curve)
    start = min(int(len(curve) * max(0.0, min(1.0, burn_in))), len(curve) - 1)
    endemic = float(np.mean(curve[start:]))

    return {
        "peak_prevalence": peak,
        "peak_prevalence_pct": 100.0 * peak / num_nodes,
        "time_to_peak": int(np.argmax(curve)),
        "auc_infectious": float(sum(curve)),
        # Normalized by N * T so two runs at different horizons are comparable
        "auc_infectious_pct": 100.0 * sum(curve) / (num_nodes * len(curve)),
        "endemic_prevalence": endemic,
        "endemic_prevalence_pct": 100.0 * endemic / num_nodes,
    }


def immunization_metrics(
    edge_index: np.ndarray,
    num_nodes: int,
    removed: list[int],
    threshold: float = gcc_threshold,
) -> dict:
    """
    The spectral and structural description of one arm's dose allocation.

    §8.3's context columns, and the same role `containment_metrics` plays for
    dismantling: never a training target and never the arm's reward, which is the
    simulated attack rate. What this adds over that function is the EIGENDROP —
    `lambda_1(A) - lambda_1(A - S)` — because that is the quantity NetShield,
    NetMelt, Gelling and GreedyWalk report and therefore the only number our table
    and theirs share.

    Reported together with the connectivity profile on purpose. §5.8 of the
    dismantling review and §8.2 trap 1 here make the same point from two sides: a
    method can win on eigendrop and lose on simulated final size, because
    `lambda_1` says nothing about WHERE the infection currently is. Printing both
    beside the attack rate is what makes that disagreement visible rather than a
    thing a reader has to already know.
    """
    order = [int(node) for node in removed]
    neighbours = adjacency_sets(edge_index, num_nodes)

    intact = connectivity_profile(neighbours, set())
    after = connectivity_profile(neighbours, set(order))
    lambda_intact = spectral_radius(edge_index, num_nodes)
    lambda_after = spectral_radius(edge_index, num_nodes, order)

    return {
        "doses": order,
        "k": len(order),
        "lambda1_intact": lambda_intact,
        "lambda1": lambda_after,
        "eigendrop": lambda_intact - lambda_after,
        "eigendrop_pct": (
            100.0 * (lambda_intact - lambda_after) / lambda_intact
            if lambda_intact
            else 0.0
        ),
        "pairwise_conn_intact": intact["pairwise_conn"],
        "pairwise_conn": after["pairwise_conn"],
        "largest_cc_intact": intact["largest_cc_size"],
        "largest_cc_size": after["largest_cc_size"],
        "gcc_fraction": after["gcc_fraction"],
        "n_components": after["n_components"],
        "largest_cc_drop_pct": (
            100.0 * (1.0 - after["largest_cc_size"] / intact["largest_cc_size"])
            if intact["largest_cc_size"]
            else 0.0
        ),
    }


def containment_metrics(
    edge_index: np.ndarray,
    num_nodes: int,
    removed: list[int],
    threshold: float = gcc_threshold,
) -> dict:
    """
    The structural description of one arm's removal set — §8.3's context column.

    Never a training target and never the arm's reward. The reward is the
    ground-truth MC spread; this says what the same set did to the graph's
    connectivity, which is the quantity every published dismantling number is
    measured in and therefore the only bridge between our table and theirs.

    `degree_rank_spearman` is the self-measurement §9.5 asks for: MIND found GDM's
    dismantling order correlates at 0.762 with a PCA of its own handcrafted input
    features. We feed log1p(degree) as channel 2, so a removal order that
    correlates with degree at ~0.76 has re-derived the degree heuristic with extra
    steps. Measure it, do not assume it.
    """
    neighbours = adjacency_sets(edge_index, num_nodes)
    order = [int(node) for node in removed]

    intact = connectivity_profile(neighbours, set())
    after = connectivity_profile(neighbours, set(order))
    curve = dismantling_curve(neighbours, order)

    degrees = np.array([len(group) for group in neighbours], dtype=np.float64)
    # Rank the removal order against degree: position 0 is removed first, so the
    # order is negated to make "removed early" the high-rank end, matching "high
    # degree" at the other side of the correlation
    chosen_degrees = degrees[order] if order else np.zeros(0)
    positions = -np.arange(len(order), dtype=np.float64)

    metrics = {
        "removed": order,
        "k": len(order),
        "gcc_threshold": threshold,
        "pairwise_conn_intact": intact["pairwise_conn"],
        "largest_cc_intact": intact["largest_cc_size"],
        "n_components_intact": intact["n_components"],
        "pairwise_conn": after["pairwise_conn"],
        "largest_cc_size": after["largest_cc_size"],
        "gcc_fraction": after["gcc_fraction"],
        "n_components": after["n_components"],
        "gcc_curve": [round(value, 6) for value in curve],
        "schneider_r": schneider_r(curve, num_nodes),
        # sigma = GCC size, named because ANC is only meaningful with sigma stated
        "anc_sigma": "gcc_size",
        "anc": accumulated_normalized_connectivity(curve[1:], curve[0] or 1.0),
        "rho_at_threshold": dismantling_set_size(curve, threshold),
        "degree_rank_spearman": (
            spearman(positions, chosen_degrees) if len(order) > 1 else 0.0
        ),
    }

    # The reductions the report actually prints, so nothing downstream divides by
    # a possibly-zero intact value
    metrics["pairwise_conn_drop_pct"] = (
        100.0 * (1.0 - after["pairwise_conn"] / intact["pairwise_conn"])
        if intact["pairwise_conn"]
        else 0.0
    )
    metrics["largest_cc_drop_pct"] = (
        100.0 * (1.0 - after["largest_cc_size"] / intact["largest_cc_size"])
        if intact["largest_cc_size"]
        else 0.0
    )

    return metrics
