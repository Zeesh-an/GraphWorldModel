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
    actually optimized for.
"""

import numpy as np


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
