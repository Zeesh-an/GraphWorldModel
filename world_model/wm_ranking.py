"""
Q4 — decision utility: can the world model ORDER candidates well enough to
replace expensive trusted evaluation?

Why this file exists separately from `wm_eval.py`. The metrics there answer "is
the predicted next state right". That is not the question the coding agent asks.
It asks "which of these candidates should I spend a simulator call on", and a
model can be mediocre at per-node prediction while being excellent at that, or
the reverse. Planning regret half-answers it and, on BA, turned out to be
dominated by the candidate sampler rather than by the model.

So the chain here is deliberately ordered from cheapest claim to strongest:

    preference accuracy  ->  top-K quality  ->  calls-to-first-win
                                             ->  trusted-call reduction

Each step is a strictly stronger statement than the one before, and each is
falsifiable on its own: preference accuracy has a 0.5 null, top-K has a random-
selection null, calls-to-first-win has a random-ordering null.

Two things this module refuses to do:

  * count ties as correct. An oracle that cannot separate two candidates is not
    evidence the model ordered them right, and letting ties inflate accuracy is
    the easiest way to manufacture a good number here.
  * treat all pairs equally. Two candidates whose true spreads differ by 0.1
    nodes are inside MC noise; `by_margin` reports accuracy bucketed by oracle
    margin so a headline number cannot hide that it was earned on noise.
"""

from dataclasses import asdict, dataclass, field
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn

from data.wm_simulator import spent
from world_model.wm_data import basic_encoding
from world_model.wm_eval import _true_spread_set, seed_upper_bound

# Oracle spreads come from a finite Monte-Carlo average, so two candidates whose
# means differ by less than this are not separated by the oracle either. Pairs
# inside it are excluded from accuracy and reported separately.
default_tie_epsilon = 1e-6

# Margin buckets, in nodes of true expected spread.
margin_buckets = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, float("inf")))


@dataclass
class CandidateSet:
    """
    Candidates to be ordered, plus where they came from.

    `policy` is the algorithm that produced each seed set. Carrying it here is
    what makes the seen/unseen policy split (Workstream E) a filter over an
    existing result rather than a separate evaluation path.
    """

    graph_id: str
    seed_sets: list[list[int]]
    policies: list[str]
    budget: int

    def __post_init__(self) -> None:
        if len(self.seed_sets) != len(self.policies):
            raise ValueError(
                f"{len(self.seed_sets)} seed sets but {len(self.policies)} "
                f"policy labels; they index the same candidates"
            )


@dataclass
class RankingResult:
    graph_id: str
    n_candidates: int
    policies: list[str]
    true_spread: list[float]
    predicted_spread: list[float]
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def oracle_spreads(
    candidates: CandidateSet,
    store_entry: dict,
    diffusion_model: str,
    horizon: int = 20,
    mc_runs: int = 32,
    seed: int = 0,
    remove_semantics: str = spent,
) -> list[float]:
    """
    Trusted expected final spread per candidate, and the trusted-call budget it
    cost. One call = one simulator rollout, which is the unit `trusted_calls`
    counts everywhere in this module.
    """
    rng = np.random.default_rng(seed)

    return [
        _true_spread_set(
            seeds, store_entry, diffusion_model, horizon, mc_runs, rng,
            remove_semantics,
        )
        for seeds in candidates.seed_sets
    ]


@torch.inference_mode()
def model_spreads(
    model: nn.Module,
    candidates: CandidateSet,
    store_entry: dict,
    diffusion_model: str,
    device: torch.device,
    horizon: int = 20,
    n_samples: int = 20,
    seed: int = 0,
    remove_semantics: str = spent,
    hide_edge_weights: bool = False,
    action_encoding: str = basic_encoding,
) -> list[float]:
    """Predicted expected final spread per candidate, under the frozen model."""
    from coding_agent.types import GraphInfo

    from world_model.scorer import ScoringContext, WorldModelScorer
    from world_model.checkpoint import ModelSpec

    spec = ModelSpec(
        backbone="gcn",  # unused: the module is supplied, not rebuilt
        head="structured",
        diffusion_model=diffusion_model,
        remove_semantics=remove_semantics,
        action_encoding=action_encoding,
        hide_edge_weights=hide_edge_weights,
    )
    scorer = WorldModelScorer(model, spec, device=str(device))
    graph = GraphInfo.from_store_entry(store_entry)
    context = ScoringContext(
        horizon=horizon, budget=candidates.budget, n_samples=n_samples, seed=seed
    )
    from data.wm_simulator import ActionOp

    predictions = []

    for seeds in candidates.seed_sets:
        plan = [[ActionOp("add_node", int(node)) for node in seeds]]
        predictions.append(scorer.score_candidate(graph, plan, context))

    return predictions


# ---------------------------------------------------------------------------
# B1 — pairwise preference accuracy
# ---------------------------------------------------------------------------


def preference_accuracy(
    predicted: list[float],
    true: list[float],
    tie_epsilon: float = default_tie_epsilon,
) -> dict:
    """
    Over all candidate pairs: does the model order them the way the oracle does?

    Null is 0.5. Pairs the ORACLE cannot separate (|true_i - true_j| <=
    tie_epsilon) are excluded rather than counted either way — they carry no
    information about ordering, and counting them correct is how this metric
    gets inflated.

    Model ties on a separable pair ARE counted, as failures: a model that cannot
    distinguish two candidates has not ordered them.
    """
    predicted = np.asarray(predicted, dtype=float)
    true = np.asarray(true, dtype=float)
    correct = evaluated = model_ties = oracle_ties = 0
    margins: list[float] = []
    per_bucket: dict[tuple, list[int]] = {bucket: [] for bucket in margin_buckets}

    for first, second in combinations(range(len(true)), 2):
        margin = abs(true[first] - true[second])

        if margin <= tie_epsilon:
            oracle_ties += 1
            continue

        evaluated += 1
        margins.append(float(margin))
        predicted_delta = predicted[first] - predicted[second]

        if predicted_delta == 0.0:
            model_ties += 1
            hit = 0
        else:
            hit = int(
                np.sign(predicted_delta) == np.sign(true[first] - true[second])
            )

        correct += hit

        for bucket in margin_buckets:
            if bucket[0] <= margin < bucket[1]:
                per_bucket[bucket].append(hit)
                break

    return {
        "preference_accuracy": correct / evaluated if evaluated else float("nan"),
        "n_pairs_evaluated": evaluated,
        "n_pairs_oracle_tied": oracle_ties,
        "n_pairs_model_tied": model_ties,
        "oracle_margin_mean": float(np.mean(margins)) if margins else float("nan"),
        "oracle_margin_median": (
            float(np.median(margins)) if margins else float("nan")
        ),
        "by_margin": {
            f"[{low},{high})": {
                "accuracy": float(np.mean(hits)) if hits else float("nan"),
                "n": len(hits),
            }
            for (low, high), hits in per_bucket.items()
        },
    }


# ---------------------------------------------------------------------------
# B2 — ranking correlation
# ---------------------------------------------------------------------------


def kendall_tau(predicted: list[float], true: list[float]) -> float:
    """Tau-b: ties in either ranking reduce the denominator rather than counting."""
    predicted = np.asarray(predicted, dtype=float)
    true = np.asarray(true, dtype=float)
    concordant = discordant = predicted_ties = true_ties = 0

    for first, second in combinations(range(len(true)), 2):
        predicted_sign = np.sign(predicted[first] - predicted[second])
        true_sign = np.sign(true[first] - true[second])

        if predicted_sign == 0 and true_sign == 0:
            continue

        if predicted_sign == 0:
            predicted_ties += 1
            continue

        if true_sign == 0:
            true_ties += 1
            continue

        if predicted_sign == true_sign:
            concordant += 1
        else:
            discordant += 1

    denominator = np.sqrt(
        (concordant + discordant + predicted_ties)
        * (concordant + discordant + true_ties)
    )

    return float((concordant - discordant) / denominator) if denominator else float("nan")


def spearman_rho(predicted: list[float], true: list[float]) -> float:
    from world_model.wm_metrics import spearman

    return spearman(np.asarray(predicted, dtype=float), np.asarray(true, dtype=float))


# ---------------------------------------------------------------------------
# B3 — top-K
# ---------------------------------------------------------------------------


def top_k_metrics(
    predicted: list[float], true: list[float], k_values: tuple = ()
) -> dict:
    """
    Precision@K and Recall@K of the model's top-K against the oracle's.

    K defaults are derived from the candidate count, not hard-coded: a fixed
    `3` means something different with 5 candidates than with 40, and a
    Precision@3 over 4 candidates is nearly free.
    """
    predicted = np.asarray(predicted, dtype=float)
    true = np.asarray(true, dtype=float)
    count = len(true)

    if not k_values:
        k_values = tuple(
            sorted({1, max(1, count // 10), max(1, count // 4)} & set(range(1, count)))
        ) or (1,)

    predicted_order = np.argsort(-predicted, kind="stable")
    true_order = np.argsort(-true, kind="stable")
    results = {"n_candidates": count, "k_values": list(k_values)}

    for k in k_values:
        if k > count:
            continue

        predicted_top = set(predicted_order[:k].tolist())
        true_top = set(true_order[:k].tolist())
        hits = len(predicted_top & true_top)
        results[f"precision_at_{k}"] = hits / k
        results[f"recall_at_{k}"] = hits / len(true_top)
        # The value actually obtained by taking the model's top-1, relative to
        # the best available. Precision@1 says "did it pick THE best"; this says
        # "how good was what it picked", which is the quantity a planner cares
        # about when several candidates are nearly tied.
        results[f"top_{k}_best_true_value"] = float(
            max(true[index] for index in predicted_top)
        )

    results["oracle_best_true_value"] = float(true.max()) if count else float("nan")

    return results


# ---------------------------------------------------------------------------
# B4 — calls to first win
# ---------------------------------------------------------------------------


def calls_to_first_win(
    order: list[int],
    true: list[float],
    win_threshold: float,
) -> float:
    """
    Trusted calls spent walking a ranking until the first winner appears.

    `win_threshold` is an absolute value on the oracle scale and comes from the
    experiment config, never from inside this function — a criterion chosen
    after seeing the scores is not a criterion.

    Returns `len(order) + 1` when no candidate wins, so a run that never
    succeeds is worse than one that succeeds last rather than being dropped from
    the mean.
    """
    true = np.asarray(true, dtype=float)

    for position, index in enumerate(order, start=1):
        if true[index] >= win_threshold:
            return float(position)

    return float(len(order) + 1)


def ranking_orders(
    predicted: list[float],
    true: list[float],
    degrees: np.ndarray | None = None,
    seed_sets: list[list[int]] | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, list[int]]:
    """
    The orderings compared for calls-to-first-win.

    `oracle` is the ceiling (it already knows the answer), `random` the floor,
    `degree` the heuristic a world model has to beat to be worth its cost.
    """
    count = len(true)
    orders = {
        "world_model": np.argsort(-np.asarray(predicted, dtype=float),
                                  kind="stable").tolist(),
        "oracle": np.argsort(-np.asarray(true, dtype=float), kind="stable").tolist(),
    }
    rng = rng or np.random.default_rng(0)
    orders["random"] = rng.permutation(count).tolist()

    if degrees is not None and seed_sets is not None:
        totals = [float(sum(degrees[node] for node in seeds)) for seeds in seed_sets]
        orders["degree"] = np.argsort(-np.asarray(totals), kind="stable").tolist()

    return orders


# ---------------------------------------------------------------------------
# B5 — trusted-call reduction
# ---------------------------------------------------------------------------


def trusted_call_reduction(
    calls_by_order: dict[str, float], reference: str = "random"
) -> dict:
    """
    Fraction of trusted calls saved relative to a reference ordering.

    Reported against `random` by default rather than against exhaustive
    evaluation: the honest baseline for "the model saved us calls" is what you
    would have spent with no model, not what you would have spent checking
    everything.
    """
    baseline = calls_by_order.get(reference)

    if not baseline:
        return {}

    return {
        f"call_reduction_vs_{reference}_{name}": float(1.0 - calls / baseline)
        for name, calls in calls_by_order.items()
        if name != reference
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def ranking_report(
    predicted: list[float],
    true: list[float],
    policies: list[str],
    graph_id: str,
    win_threshold: float | None = None,
    win_quantile: float = 0.8,
    degrees: np.ndarray | None = None,
    seed_sets: list[list[int]] | None = None,
    seen_policies: set[str] | None = None,
    seed: int = 0,
) -> RankingResult:
    """
    Every Q4 metric for one graph's candidate set.

    `win_threshold` is absolute; if absent it is derived from `win_quantile` of
    the oracle scores ON THIS GRAPH and the derived value is recorded, so a
    reader can see the criterion was not tuned per result.
    """
    true_array = np.asarray(true, dtype=float)
    threshold = (
        float(win_threshold)
        if win_threshold is not None
        else float(np.quantile(true_array, win_quantile))
    )
    orders = ranking_orders(
        predicted, true, degrees, seed_sets, np.random.default_rng(seed)
    )
    calls = {
        name: calls_to_first_win(order, true, threshold)
        for name, order in orders.items()
    }

    metrics = {
        **preference_accuracy(predicted, true),
        "kendall_tau": kendall_tau(predicted, true),
        "spearman_rho": spearman_rho(predicted, true),
        **top_k_metrics(predicted, true),
        "win_threshold": threshold,
        "win_quantile": win_quantile if win_threshold is None else None,
        **{f"calls_to_first_win_{name}": value for name, value in calls.items()},
        **trusted_call_reduction(calls),
    }

    # Workstream E: the same candidates, filtered to the policies the world model
    # never saw in training. A filter over one result rather than a second
    # evaluation, so seen and unseen are measured under identical conditions.
    if seen_policies is not None:
        for label, wanted_seen in (("seen", True), ("unseen", False)):
            indices = [
                index
                for index, policy in enumerate(policies)
                if (policy in seen_policies) == wanted_seen
            ]

            if len(indices) < 2:
                metrics[f"preference_accuracy_{label}"] = float("nan")
                metrics[f"n_candidates_{label}"] = len(indices)
                continue

            subset_predicted = [predicted[index] for index in indices]
            subset_true = [true[index] for index in indices]
            metrics[f"preference_accuracy_{label}"] = preference_accuracy(
                subset_predicted, subset_true
            )["preference_accuracy"]
            metrics[f"kendall_tau_{label}"] = kendall_tau(subset_predicted, subset_true)
            metrics[f"n_candidates_{label}"] = len(indices)

    return RankingResult(
        graph_id=graph_id,
        n_candidates=len(true),
        policies=list(policies),
        true_spread=[float(value) for value in true],
        predicted_spread=[float(value) for value in predicted],
        metrics=metrics,
    )


def aggregate(results: list[RankingResult]) -> dict:
    """
    Mean and std across graphs for every scalar metric, plus POOLED call counts.

    Call reduction must be pooled, not averaged per graph. A per-graph ratio has
    the ranking's own call count in the denominator, and that denominator is
    often 1 or 2 — so a graph where the random baseline got lucky contributes a
    ratio of -1 or -2 and drags the mean anywhere. Measured on the first Q4 run,
    the mean of per-graph ratios read -0.366 while the pooled estimate was
    +0.044: the same data, one of them meaningless.

    Both are reported; the pooled one carries the `_pooled` suffix and is the
    one to quote.
    """
    if not results:
        return {}

    scalar_keys = [
        key
        for key, value in results[0].metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    summary = {"n_graphs": len(results)}

    for key in scalar_keys:
        values = np.array(
            [result.metrics.get(key, np.nan) for result in results], dtype=float
        )
        values = values[np.isfinite(values)]

        if values.size:
            summary[key] = float(values.mean())
            summary[f"{key}_std"] = float(values.std())
            summary[f"{key}_n"] = int(values.size)

    # Pooled call counts and the reduction derived from them.
    call_keys = {
        key.replace("calls_to_first_win_", "")
        for key in results[0].metrics
        if key.startswith("calls_to_first_win_")
    }
    totals = {
        name: float(
            sum(
                result.metrics.get(f"calls_to_first_win_{name}", 0.0)
                for result in results
            )
        )
        for name in call_keys
    }

    for name, total in totals.items():
        summary[f"total_calls_{name}"] = total

    baseline = totals.get("random")

    if baseline:
        for name, total in totals.items():
            if name != "random":
                summary[f"call_reduction_pooled_vs_random_{name}"] = float(
                    1.0 - total / baseline
                )

    return summary
