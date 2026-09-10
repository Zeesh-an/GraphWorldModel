"""
The search's own bookkeeping under a stochastic evaluator.

Four things a deterministic-fitness discovery loop never needs, all read off
data the paired protocol already produces: the noise band of one comparison
(the paired standard error, not the marginal ones), a noise-aware ranking of
the population (least squares over the paired comparisons the search made),
the ledger of what the naive `delta > 0` rule would have accepted, and the
model's own forecast of each edit against what the edit did.
"""

import math
import re
import numpy as np

from coding_agent.types import Trajectory, improves, minimize

# Rows carrying a forecast are scored on their sign and their magnitude; this
# many recent forecasts decide whether the model is currently miscalibrated
calibration_window = 3
# Comparisons with a zero band get this weight ceiling in the least-squares fit
min_band = 1e-6

_expected_line = re.compile(
    r"^\s*#\s*EXPECTED:\s*([+-]?\d+(?:\.\d+)?)(?:.*?\bp\s*=\s*(\d+(?:\.\d+)?))?",
    re.M,
)


def expected_of(script: str) -> tuple[float | None, float | None]:
    """
    The `# EXPECTED: <signed delta> p=<probability>` line a generated script
    carries under its mechanism line: the paired delta the model forecast for
    this edit and its stated probability of clearing the band. Either is None
    when absent or malformed; a probability outside [0, 1] is clamped.
    """
    match = _expected_line.search(script or "")
    if match is None:
        return None, None

    predicted = float(match.group(1))
    probability = None
    if match.group(2) is not None:
        probability = min(1.0, max(0.0, float(match.group(2))))

    return predicted, probability


def paired_band(candidate: Trajectory, incumbent: Trajectory) -> tuple[float, str]:
    """
    The noise of THIS comparison: the standard error of the mean of the paired
    per-sample differences when both trajectories carry aligned samples, else
    the larger marginal standard error.

    Under common random numbers sample s of both rollouts saw the same coin
    flips, so the realization noise cancels in the difference and the paired
    standard error is usually far below either marginal one. The old
    `max(se_c, se_i)` rule was wrong in both directions: too strict when the
    samples are positively correlated (it rejected real improvements) and too
    loose when they are not (the standard error of an independent difference is
    the root sum of squares, above the max).
    """
    a = candidate.sample_rewards
    b = incumbent.sample_rewards
    if a is not None and b is not None and len(a) == len(b) and len(a) > 1:
        differences = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        return float(np.std(differences, ddof=1) / math.sqrt(len(differences))), "paired"

    return (
        max(
            float(candidate.cost.get("reward_se") or 0.0),
            float(incumbent.cost.get("reward_se") or 0.0),
        ),
        "marginal",
    )


def paired_strengths(records: list[dict]) -> dict[int, float]:
    """
    A noise-aware score per population member, on the reward scale.

    Every scored generation compared its candidate with the incumbent of that
    generation on ONE realization, so the search's own history is a set of
    paired comparisons `reward(c) - reward(i) = delta`. The strengths are the
    weighted least-squares solution of those equations (HodgeRank, Jiang et al.
    2011), with the seed program pinned at its own reward and each comparison
    weighted by the inverse of its band. While every candidate was compared
    exactly once the comparison graph is a tree and the solution is the exact
    chain sum `s(c) = s(i) + delta`; extra comparisons (a re-scored member, the
    anchor as a standing competitor) add rows and the same call still applies.
    Members with no comparison (a legacy checkpoint) keep their raw reward.
    """
    if not records:
        return {}

    ids = [int(record["iteration"]) for record in records]
    index = {iteration: position for position, iteration in enumerate(ids)}
    seed = ids[0]
    rows, targets, weights = [], [], []

    for record in records:
        compared_to = record.get("compared_to")
        delta = record.get("paired_delta")
        if compared_to is None or delta is None or int(compared_to) not in index:
            continue

        row = np.zeros(len(ids))
        row[index[int(record["iteration"])]] = 1.0
        row[index[int(compared_to)]] = -1.0
        rows.append(row)
        targets.append(float(delta))
        weights.append(1.0 / max(float(record.get("band") or 0.0), min_band))

    # Pin the seed at its own reward so the system is determined
    pin = np.zeros(len(ids))
    pin[index[seed]] = 1.0
    rows.append(pin)
    targets.append(float(records[0]["reward"]))
    weights.append(1.0)

    design = np.asarray(rows) * np.sqrt(np.asarray(weights))[:, None]
    target = np.asarray(targets) * np.sqrt(np.asarray(weights))
    solution, _, _, _ = np.linalg.lstsq(design, target, rcond=None)

    strengths = {}
    compared = {int(record["iteration"]) for record in records if record.get("compared_to") is not None}
    for record in records:
        iteration = int(record["iteration"])
        if iteration == seed or iteration in compared:
            strengths[iteration] = float(solution[index[iteration]])
        else:
            strengths[iteration] = float(record["reward"])

    return strengths


def acceptance_ledger(history: list[dict]) -> dict:
    """
    What the naive rule (`delta > 0` in the task's sense) would have done,
    generation by generation, against what the band rule did.
    """
    scored = [
        entry
        for entry in history
        if entry.get("reward") is not None and entry.get("delta") is not None
    ]
    naive = sum(1 for entry in scored if entry.get("naive_accepted"))
    band = sum(1 for entry in scored if entry.get("accepted"))
    lucky = sum(
        1 for entry in scored if entry.get("naive_accepted") and not entry.get("accepted")
    )
    ties_kept = sum(
        1 for entry in scored if entry.get("accepted") and not entry.get("naive_accepted")
    )

    return {
        "compared_generations": len(scored),
        "naive_accepts": naive,
        "band_accepts": band,
        # Candidates with a positive delta inside the band: the coin flips the
        # old loop took as progress
        "lucky_accepts_prevented": lucky,
        # `simplify` children that tied inside the band and replaced a longer
        # incumbent
        "ties_kept": ties_kept,
        "band_rules": sorted({entry.get("band_rule", "marginal") for entry in scored}),
    }


def calibration_metrics(history: list[dict]) -> dict:
    """
    How well the model forecast its own edits: predicted against realized paired
    delta over the generations that carried an `# EXPECTED:` line.
    """
    rows = [
        entry
        for entry in history
        if entry.get("predicted_delta") is not None and entry.get("delta") is not None
    ]
    if not rows:
        return {"n": 0}

    predicted = np.asarray([entry["predicted_delta"] for entry in rows], dtype=np.float64)
    realized = np.asarray([entry["delta"] for entry in rows], dtype=np.float64)
    errors = predicted - realized
    signs = np.sign(predicted) == np.sign(realized)

    pearson = None
    if len(rows) > 2 and predicted.std() > 0 and realized.std() > 0:
        pearson = float(np.corrcoef(predicted, realized)[0, 1])

    spearman = None
    if len(rows) > 2:
        ranks_p = np.argsort(np.argsort(predicted)).astype(np.float64)
        ranks_r = np.argsort(np.argsort(realized)).astype(np.float64)
        if ranks_p.std() > 0 and ranks_r.std() > 0:
            spearman = float(np.corrcoef(ranks_p, ranks_r)[0, 1])

    # Brier score of the stated P(clears the band) against whether it did; the
    # sign Brier treats the forecast's sign as a 1/0 probability of improving
    with_probability = [
        entry for entry in rows if entry.get("predicted_clear_p") is not None
    ]
    brier = None
    if with_probability:
        brier = float(
            np.mean(
                [
                    (float(entry["predicted_clear_p"]) - float(bool(entry.get("accepted")))) ** 2
                    for entry in with_probability
                ]
            )
        )

    return {
        "n": len(rows),
        "pearson_r": pearson,
        "spearman_r": spearman,
        "sign_accuracy": float(signs.mean()),
        "mae": float(np.abs(errors).mean()),
        "mean_error": float(errors.mean()),
        "brier": brier,
        "n_with_probability": len(with_probability),
    }


def miscalibrated(history: list[dict], window: int = calibration_window) -> bool:
    """
    True when the last `window` forecasts got the sign wrong more often than not:
    the model's local picture of the program is stale and an explore move is
    worth more than another refine.
    """
    rows = [
        entry
        for entry in history
        if entry.get("predicted_delta") is not None and entry.get("delta") is not None
    ][-window:]
    if len(rows) < window:
        return False

    wrong = sum(
        1
        for entry in rows
        if np.sign(entry["predicted_delta"]) != np.sign(entry["delta"])
    )

    return wrong > len(rows) / 2


def optimism_summary(history: list[dict], sense: str) -> dict:
    """
    The incumbent's accepted score against its re-score on the next fresh
    realization: an accepted score is a maximum over noisy estimates and biased
    in the improving direction, the re-score is not. Positive optimism means the
    reported curve flattered the program.
    """
    rows = [
        entry
        for entry in history
        if entry.get("incumbent_reported") is not None
        and entry.get("incumbent_unbiased") is not None
    ]
    if not rows:
        return {"n": 0}

    sign = -1.0 if sense == minimize else 1.0
    gaps = [
        sign * (float(entry["incumbent_reported"]) - float(entry["incumbent_unbiased"]))
        for entry in rows
    ]

    return {
        "n": len(rows),
        "mean": float(np.mean(gaps)),
        "final": float(gaps[-1]),
        "max": float(np.max(gaps)),
        "final_reported": float(rows[-1]["incumbent_reported"]),
        "final_unbiased": float(rows[-1]["incumbent_unbiased"]),
    }


def flat_generations(history: list[dict], sense: str) -> int:
    """
    Consecutive trailing generations in which the unbiased incumbent estimate
    did not improve on its running best by more than that generation's band:
    the stopping signal `--stop-when-flat` reads.
    """
    rows = [
        entry
        for entry in history
        if entry.get("incumbent_unbiased") is not None and entry.get("band") is not None
    ]
    best = None
    flat = 0
    for entry in rows:
        value = float(entry["incumbent_unbiased"])
        if best is None or improves(value, best, sense, float(entry["band"])):
            best = value
            flat = 0
        else:
            flat += 1

    return flat
