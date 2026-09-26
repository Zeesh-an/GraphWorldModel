"""
Named classical CASCADE PREDICTION baselines: the condition-1 floor for
`--task cascade_prediction`.

Every predictor has the signature

    (graph, observation, horizon, **kw) -> float | None

returning the popularity the cascade will have reached by `t_p`, or **None to
DECLINE**. That second return is not politeness: generative models refuse to score
supercritical cascades (SEISMIC
produced no prediction at all for 1,022 of ~20K News cascades at five minutes),
papers report the mean over scoreable cascades only, and this "silently
favours the model that gives up more often". Mishra et al. publish their failure
counts and almost nobody else does, so `n_failed` is a column here rather than a
footnote and declining is a first-class outcome rather than an error.

That is the whole difference from the other four pools. `algorithms.py` returns a
seed set, `dismantling_algorithms.py` nodes to delete, `localization_algorithms.py`
a source set, `reconstruction_algorithms.py` a trajectory, and this one returns a
single number, because the task emits no action at all.

None of these is a weak floor. In ascending order of danger:

  1. `random_prediction`, `mean_size`, `persistence`: the three floors. Do not
     dismiss `mean_size`: MSLE is an error in LOG space, so predicting the training
     mean is far stronger than the same idea would be under squared error, and an
     arm that cannot clear it has learned nothing about the instance.
  2. **`szabo_huberman`**: the row that actually has to be beaten. One feature, one
     line, from 2008, and every published paper still prints it as "Feature-S&H".
     Its whole content is that `log P(t_p)` is near-linear in `log P(t_o)`.
  3. **`feature_linear` / `feature_gbt`**: CTCP's own MLP and XGBoost rows over
     Cheng et al.'s five feature classes, and CasFlow's own ablation found that
     feature models "in some cases even beat deep learning models".
     Cheng et al.'s central result is that the TEMPORAL features dominate.
  4. **`seismic` / `hawkes` / `hawkes_hybrid`**: the generative line, and the
     closest classical analogue to a world model: a stated transition mechanism plus
     a fitted parameter, rolled forward. Their failure mode is ours too: SEISMIC
     diverges when the branching factor exceeds 1, which is the same runaway
     `ens_count_bias` was built to catch, and the field's answer (a learned
     corrective layer on a generative core) is structurally our
     `structured_residual` head.

**These are honest condition-1 arms, not substitutes for the authors' code.** Three
deviations are shared widely enough to name up front:

  * **The supervised ones fit on the SELECTION split only.** `feature_linear`,
    `feature_gbt`, `hawkes_hybrid`, `szabo_huberman` and `mean_size` all need a
    fitted constant, and the harness hands them the selection instances through
    `fit_examples`. Fitting on the evaluation split would be the one leak
    `run_baseline`'s module docstring forbids.
  * **No content features anywhere.** Cheng et al.'s five classes include content,
    and their own result is that content alone reaches 0.558 accuracy against 0.73
    for the best temporal feature [verified]. Our corpora carry no text, so
    the content class is absent and the temporal class: the one that dominates,
    is not.
  * **Point-process fits are moment-matched, not MLE.** SEISMIC's and the Hawkes
    line's published implementations fit by maximum likelihood over the exact event
    times; ours estimate the branching factor and the decay from the observed
    per-step waves, which is what a binned replay supports. Each docstring says so.
"""

import math
import numpy as np

from coding_agent.types import GraphInfo

# Szabo & Huberman's model is `log P(t_p) = alpha * log P(t_o) + beta`. With no
# fitted pair to hand (a single-instance call, or an unfitted arm) these are the
# defaults: alpha = 1 and beta = log(2), i.e. "it doubles", which is exactly the
# median outcome Cheng et al. built their balanced classification task around.
default_sh_slope = 1.0
default_sh_intercept = math.log(2.0)

# Reinforced Poisson's aging kernel is log-normal in the published model; fitting
# its three parameters per cascade needs the exact arrival times, which a binned
# replay does not have. We fit the two that a wave series identifies (total
# intensity and decay) and hold the log-normal shape at the paper's own scale.
rpp_default_decay = 0.6

# HIP's exogenous term needs a separate stimulus series (search volume, shares) that
# none of our corpora publish, so our version runs the endogenous half alone. Stated
# here rather than in the function, because it is also why the `hip` external repo
# is registered BLOCKED rather than wired.
hip_default_theta = 1.5

# Branching-factor extrapolation stops here: a supercritical estimate diverges, and
# SEISMIC's honest response is to DECLINE rather than to emit a number. Ours
# declines at the same boundary.
supercritical_threshold = 0.999

# Reachability's BFS depth. The structural ceiling is what a cascade COULD reach if
# every remaining step transmitted perfectly, so it is bounded by the horizon.
max_reach_depth = 6

# The floor every predictor is clamped to: a progressive cascade cannot shrink, and
# an observation is ground truth about the nodes it reports.
minimum_popularity = 1.0


def _waves(observation) -> list[int]:
    """Adopters per observed timestep, index 0 = the seed wave."""
    counts = [0] * (observation.observed_steps + 1)

    for _, step in observation.adopters.items():
        index = min(max(int(step), 0), len(counts) - 1)
        counts[index] += 1

    return counts


def cascade_features(graph: GraphInfo, observation) -> dict[str, float]:
    """
    Cheng et al.'s five feature classes, minus content, over the observed prefix.

    Cheng et al. are the reason this exists as a shared function rather than inside
    one predictor: that paper's finding is not "features work" but "TEMPORAL features
    dominate", with the reshare rate in the second half of the observation window
    the single best one at 0.73 accuracy and the best STRUCTURAL feature at 0.65. A
    pool whose members each computed their own features could not show that, and the
    scored-mode harness (`ScoredStrategy.growth_factor`) takes this dict directly so
    a generated rule searches the same space the published line does.
    """
    waves = _waves(observation)
    observed = float(observation.popularity)
    steps = max(len(waves) - 1, 1)
    half = max(steps // 2, 1)

    first_half = float(sum(waves[: half + 1]))
    second_half = float(sum(waves[half + 1 :]))

    adopters = list(observation.adopters)
    frontier = list(observation.frontier)

    degrees = [float(graph.degree(node)) for node in adopters] or [0.0]
    frontier_degrees = [float(graph.degree(node)) for node in frontier] or [0.0]

    # The out-neighbourhood the cascade has NOT yet reached: the structural budget
    # the remaining horizon has to work with
    reached = set(adopters)
    exposed = {
        neighbour
        for node in frontier
        for neighbour in graph.out_neighbors(node)
        if neighbour not in reached
    }

    return {
        "observed": observed,
        "log_observed": math.log(max(observed, minimum_popularity)),
        "observed_steps": float(observation.observed_steps),
        "remaining_steps": float(max(observation.horizon - observation.observed_steps, 0)),
        # Temporal: the class that dominates
        "rate": observed / max(observation.observed_steps, 1),
        "rate_first_half": first_half / max(half, 1),
        "rate_second_half": second_half / max(steps - half, 1),
        "acceleration": (second_half / max(steps - half, 1))
        - (first_half / max(half, 1)),
        "time_to_half": float(
            next((index for index, _ in enumerate(np.cumsum(waves)) if _ >= observed / 2), steps)
        ),
        "last_wave": float(waves[-1]) if waves else 0.0,
        "peak_wave": float(max(waves)) if waves else 0.0,
        # Root
        "root_degree": float(graph.degree(observation.root)),
        "log_root_degree": math.log1p(float(graph.degree(observation.root))),
        # Structural
        "mean_degree": float(np.mean(degrees)),
        "max_degree": float(np.max(degrees)),
        "frontier_size": float(len(frontier)),
        "frontier_mean_degree": float(np.mean(frontier_degrees)),
        "exposed": float(len(exposed)),
        "exposure_ratio": len(exposed) / max(observed, 1.0),
        # Community: how many distinct 1-hop neighbourhoods the adopters span, which
        # is Weng et al.'s virality signal in the cheapest form a graph supports
        "spread_breadth": float(len(exposed)) / max(float(graph.num_nodes), 1.0),
        "graph_fraction": observed / max(float(graph.num_nodes), 1.0),
    }


feature_order = (
    "log_observed",
    "rate",
    "rate_first_half",
    "rate_second_half",
    "acceleration",
    "last_wave",
    "peak_wave",
    "log_root_degree",
    "mean_degree",
    "max_degree",
    "frontier_size",
    "frontier_mean_degree",
    "exposure_ratio",
    "spread_breadth",
    "observed_steps",
    "remaining_steps",
)


def feature_vector(graph: GraphInfo, observation) -> np.ndarray:
    features = cascade_features(graph, observation)

    return np.array([features[key] for key in feature_order], dtype=np.float64)


def _fit_matrix(graph: GraphInfo, examples: list) -> tuple[np.ndarray, np.ndarray]:
    """`(X, log-target)` over the labelled selection instances."""
    rows = [feature_vector(graph, example.observation) for example in examples]
    targets = [
        math.log(max(float(example.actual), minimum_popularity)) for example in examples
    ]

    return np.array(rows, dtype=np.float64), np.array(targets, dtype=np.float64)


# Predictors -----------------------------------------------------------------


def persistence(graph: GraphInfo, observation, horizon: int, **_: object) -> float:
    """
    `P(t_p) = P(t_o)`: the cascade is over.

    The floor every table needs and the one this literature never prints, because
    it is embarrassing how well it does: most cascades ARE over, so under a
    log-space error it is a strong baseline and an arm that does not beat it has
    learned nothing.
    """
    return float(observation.popularity)


def mean_size(
    graph: GraphInfo, observation, horizon: int, fit_examples: list | None = None, **_: object
) -> float:
    """
    The geometric mean of the training cascades' final sizes, ignoring the instance.

    The trivial predictor, and the one to check a reward against: MSLE is an error
    in LOG space, so the geometric mean is its minimizer over a constant predictor
    and this is the best any instance-blind rule can do. An arm near it has
    discovered the corpus's size distribution and nothing else.

    Instance-blind EXCEPT for the physical floor, and it has to be: a cascade that
    already has more adopters than the corpus geometric mean would otherwise get a
    prediction below its own observed count, which `prediction.validate` rejects
    outright (adoption is progressive, so that answer is known-wrong rather than
    merely bad). Without the clamp this baseline cannot run at all on any corpus
    whose largest cascades exceed its typical one, which is every corpus here.
    Every other member of this pool clamps the same way.
    """
    if not fit_examples:
        return float(observation.popularity)

    logs = [
        math.log(max(float(example.actual), minimum_popularity))
        for example in fit_examples
    ]

    return max(float(observation.popularity), float(math.exp(float(np.mean(logs)))))


def szabo_huberman(
    graph: GraphInfo, observation, horizon: int, fit_examples: list | None = None, **_: object
) -> float:
    """
    Szabo & Huberman (CACM 2010): `log P(t_p) = alpha * log P(t_o) + beta`.

    The origin of the feature line, one feature and one line, and every paper since
    still prints it as Feature-S&H. Fitted by least squares on the selection
    split when examples are available and held at the doubling constant otherwise.

    **This is the row that has to be beaten.** A learned method that does not clearly
    clear a 2008 one-parameter regression has demonstrated nothing, which is the same
    bar `adaptive_degree` sets for dismantling and `lpsi` for source localization.
    """
    slope, intercept = default_sh_slope, default_sh_intercept

    if fit_examples:
        x = np.array(
            [
                math.log(max(float(example.observation.popularity), minimum_popularity))
                for example in fit_examples
            ]
        )
        y = np.array(
            [
                math.log(max(float(example.actual), minimum_popularity))
                for example in fit_examples
            ]
        )

        if x.size > 1 and float(np.std(x)) > 0:
            slope, intercept = np.polyfit(x, y, 1)

    observed = math.log(max(float(observation.popularity), minimum_popularity))

    return max(
        float(observation.popularity), float(math.exp(slope * observed + intercept))
    )


def _ridge(rows: np.ndarray, targets: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Closed-form ridge with an intercept column; no sklearn dependency for one line."""
    design = np.hstack([rows, np.ones((rows.shape[0], 1))])
    gram = design.T @ design + strength * np.eye(design.shape[1])
    gram[-1, -1] -= strength  # never penalize the intercept

    return np.linalg.solve(gram, design.T @ targets)


def feature_linear(
    graph: GraphInfo, observation, horizon: int, fit_examples: list | None = None, **_: object
) -> float:
    """
    Ridge regression on Cheng et al.'s feature classes: CTCP's "MLP" row's honest cousin.

    Feature-driven regression is still competitive, and CasFlow's own
    ablation says it sometimes beats deep models. Linear rather than an MLP because
    the whole selling point of this row is that it is a one-screen model whose
    coefficients can be read; `feature_gbt` is the non-linear member of the pair.
    """
    if not fit_examples:
        return szabo_huberman(graph, observation, horizon)

    rows, targets = _fit_matrix(graph, fit_examples)
    weights = _ridge(rows, targets)
    features = np.append(feature_vector(graph, observation), 1.0)

    return max(
        float(observation.popularity),
        float(math.exp(min(float(features @ weights), 30.0))),
    )


def feature_gbt(
    graph: GraphInfo, observation, horizon: int, fit_examples: list | None = None, **_: object
) -> float:
    """
    Gradient-boosted trees on the same features: CTCP's XGBoost row.

    sklearn's `GradientBoostingRegressor` rather than xgboost, and stated rather than
    hidden: CTCP reports XGBoost specifically [verified] and the two are not
    identical, but adding a dependency for one baseline row is worse than a named
    substitution. Falls back to the linear fit when sklearn is unavailable.
    """
    if not fit_examples:
        return szabo_huberman(graph, observation, horizon)

    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        return feature_linear(graph, observation, horizon, fit_examples=fit_examples)

    rows, targets = _fit_matrix(graph, fit_examples)
    model = GradientBoostingRegressor(random_state=0, n_estimators=120, max_depth=3)
    model.fit(rows, targets)

    prediction = float(model.predict(feature_vector(graph, observation)[None, :])[0])

    return max(float(observation.popularity), float(math.exp(min(prediction, 30.0))))


def weng_communities(graph: GraphInfo, observation, horizon: int, **_: object) -> float:
    """
    Weng, Menczer & Ahn (2014): early diffusion ACROSS communities predicts virality.

    Their finding: community structure of the early adopter set predicts meme virality
    better than volume. Their own measure needs a community partition of the whole graph;
    ours uses the cheapest structural proxy a `GraphInfo` supports: the fraction of
    the adopter set's neighbourhood that is NOT already adopting, which is high
    exactly when the cascade has escaped its first neighbourhood. Deviation stated
    because the published measure is infection-tree entropy over Infomap communities
    and this is not that.
    """
    features = cascade_features(graph, observation)
    # Escape ratio in [0, 1]: 0 = the cascade is trapped, 1 = every exposed node is new
    escape = features["exposed"] / max(features["exposed"] + features["observed"], 1.0)
    remaining = features["remaining_steps"]

    return max(
        float(observation.popularity),
        float(observation.popularity) * (1.0 + escape * math.sqrt(max(remaining, 0.0))),
    )


def branching_factor(graph: GraphInfo, observation, horizon: int, **_: object) -> float | None:
    """
    Galton-Watson extrapolation: estimate `R` from the observed waves and sum the tail.

    The mechanism SEISMIC, our structured IC head and every generative
    branching-process model share, in its barest form: if each adopter produces `R`
    more and `R < 1`, the remaining total is `last_wave * R / (1 - R)`. **DECLINES
    when `R >= 1`**, which is the supercritical case SEISMIC also refuses: the expected
    size diverges, and emitting a number there would be a guess dressed as a model.
    """
    waves = _waves(observation)
    ratios = [
        waves[index] / waves[index - 1]
        for index in range(1, len(waves))
        if waves[index - 1] > 0
    ]

    if not ratios:
        return float(observation.popularity)

    reproduction = float(np.mean(ratios[-3:] if len(ratios) > 3 else ratios))

    if reproduction >= supercritical_threshold:
        return None

    remaining = max(horizon - observation.observed_steps, 0)
    last = float(waves[-1])
    # Truncated geometric sum over the steps that are actually left, not to infinity:
    # the horizon is finite and the infinite sum systematically over-predicts
    tail = last * reproduction * (1.0 - reproduction**remaining) / (1.0 - reproduction)

    return float(observation.popularity) + tail


def seismic(graph: GraphInfo, observation, horizon: int, **_: object) -> float | None:
    """
    SEISMIC (Zhao et al., KDD 2015): self-exciting process with time-varying infectiousness.

    The closed-form final-size estimator `P_hat = n + p_t * n_star / (1 - p_t *
    n_star)`, with the infectiousness `p_t` estimated from the recent waves and
    `n_star` the mean out-degree of the adopter set (SEISMIC's own `n*` is the mean
    follower count, which is exactly out-degree on our graphs).

    **DECLINES on supercritical cascades** (`p_t * n_star >= 1`), which is not a
    limitation added here: SEISMIC is published failing on 507 of ~30K Tweet-1Mo
    cascades at five minutes and 1,022 of ~20K News cascades, and reporting the mean
    over the rest silently favours the model that gives up more often. The decline
    is the honest behaviour and `n_failed` is where it shows.

    Deviation: the published estimator fits `p_t` by a kernel-weighted count over
    exact event times. A binned replay has waves rather than instants, so `p_t` here
    is the recent per-adopter transmission rate. The failure boundary is identical.
    """
    waves = _waves(observation)
    adopters = list(observation.adopters)

    if not adopters:
        return None

    degrees = [float(graph.degree(node)) for node in adopters]
    n_star = float(np.mean(degrees)) if degrees else 0.0

    recent = waves[-2:] if len(waves) > 2 else waves
    parents = max(float(sum(waves[:-1])), 1.0)
    infectiousness = float(sum(recent)) / parents / max(n_star, 1.0)

    branching = infectiousness * n_star

    if branching >= supercritical_threshold:
        return None

    return float(observation.popularity) + float(
        observation.popularity * branching / (1.0 - branching)
    )


def _hawkes_parameters(observation) -> tuple[float, float]:
    """
    `(branching ratio n*, decay theta)` from the observed wave series.

    A marked Hawkes process with a power-law memory kernel has expected offspring
    `n*` per event and a decay `theta`; the wave-to-wave ratio identifies the first
    and the log-slope of the decline identifies the second. Moment matching rather
    than the published MLE, for the reason the module docstring gives.
    """
    waves = _waves(observation)
    positive = [(index, count) for index, count in enumerate(waves) if count > 0]

    if len(positive) < 2:
        return 0.0, hip_default_theta

    ratios = [
        waves[index] / waves[index - 1]
        for index in range(1, len(waves))
        if waves[index - 1] > 0
    ]
    branching = float(np.mean(ratios)) if ratios else 0.0

    steps = np.array([index for index, _ in positive], dtype=np.float64)
    counts = np.log(np.array([count for _, count in positive], dtype=np.float64))
    theta = hip_default_theta

    if steps.size > 1 and float(np.std(steps)) > 0:
        slope = float(np.polyfit(steps, counts, 1)[0])
        theta = float(max(-slope, 0.05))

    return branching, theta


def hawkes(graph: GraphInfo, observation, horizon: int, **_: object) -> float | None:
    """
    Mishra et al. (CIKM 2016): a marked Hawkes process fitted per cascade.

    This row matters more than SEISMIC's: on Tweet-1Mo at five minutes
    Hawkes reaches mean ARE 0.36 against SEISMIC's 2.61, and it fails on 302
    cascades against SEISMIC's 507: better on both the error AND on how many
    cascades it can score at all, which is the pairing that should always be
    reported together.

    Same supercritical decline as `seismic`, for the same reason: `A_1 / (1 - n*)`
    diverges at `n* = 1`.
    """
    branching, _ = _hawkes_parameters(observation)

    if branching >= supercritical_threshold:
        return None

    waves = _waves(observation)
    current = float(waves[-1]) if waves else 0.0

    return float(observation.popularity) + float(current / max(1.0 - branching, 1e-6))


def hawkes_hybrid(
    graph: GraphInfo, observation, horizon: int, fit_examples: list | None = None, **_: object
) -> float | None:
    """
    Mishra et al.'s hybrid: the generative estimate CORRECTED by a fitted layer.

    Their best row by a wide margin: 0.17 / 0.15 / 0.11 mean ARE against the pure
    Hawkes 0.27 / 0.22 / 0.17 [verified], and it matters to us specifically: a
    learned corrective layer on top of a generative core is structurally identical
    to our `structured_residual` head. This is that
    architecture as a baseline, so 6-vs-this is a comparison of two corrections on
    two different cores rather than of a learned model against a heuristic.

    Declines exactly when the generative core does: a correction on no estimate is
    not an estimate.
    """
    core = hawkes(graph, observation, horizon)

    if core is None:
        return None

    if not fit_examples:
        return core

    # The correction is a ridge on (Hawkes parameters + features) -> log residual,
    # which is Mishra's own `{c, theta, A_1, n*}` feature set plus ours
    rows, corrections = [], []

    for example in fit_examples:
        estimate = hawkes(graph, example.observation, horizon)
        if estimate is None:
            continue

        branching, theta = _hawkes_parameters(example.observation)
        rows.append(
            np.concatenate(
                [
                    feature_vector(graph, example.observation),
                    [branching, theta, math.log(max(estimate, minimum_popularity))],
                ]
            )
        )
        corrections.append(
            math.log(max(float(example.actual), minimum_popularity))
            - math.log(max(estimate, minimum_popularity))
        )

    if len(rows) < 4:
        return core

    weights = _ridge(np.array(rows), np.array(corrections))
    branching, theta = _hawkes_parameters(observation)
    features = np.append(
        np.concatenate(
            [
                feature_vector(graph, observation),
                [branching, theta, math.log(max(core, minimum_popularity))],
            ]
        ),
        1.0,
    )

    return max(
        float(observation.popularity),
        float(core * math.exp(min(max(float(features @ weights), -5.0), 5.0))),
    )


def rpp(graph: GraphInfo, observation, horizon: int, **_: object) -> float:
    """
    Reinforced Poisson (Shen et al., AAAI 2014): fitness x aging x rich-get-richer.

    `lambda_t = c * f_gamma(t) * r_alpha(R_t)`: built for CITATION counts, which is
    why it is in the default pool specifically for APS. The rich-get-richer term is
    the observed count itself, the aging kernel decays the arrival rate, and the
    fitness `c` is the observed rate.

    Deviation: the published model fits a log-normal aging kernel by MLE over exact
    arrival times. Ours holds the kernel exponential at a fitted decay, which a wave
    series identifies and a log-normal does not. Never declines: the model has no
    divergence, which is the structural difference from the Hawkes line.
    """
    waves = _waves(observation)
    _, theta = _hawkes_parameters(observation)
    decay = max(theta, rpp_default_decay)

    fitness = float(observation.popularity) / max(observation.observed_steps, 1)
    remaining = max(horizon - observation.observed_steps, 0)
    total = float(observation.popularity)
    current = float(waves[-1]) if waves else fitness

    for step in range(1, remaining + 1):
        # Aging times reinforcement: the rate decays but scales with what accrued
        rate = current * math.exp(-decay * step) * (1.0 + math.log1p(total) / 10.0)
        total += rate

    return max(float(observation.popularity), total)


def hip(graph: GraphInfo, observation, horizon: int, **_: object) -> float:
    """
    Hawkes Intensity Process (Rizoiu et al., WWW 2017), endogenous half only.

    HIP's contribution is an EXOGENOUS promotion term, search, shares, off-platform
    arrivals: the term IC has no place for, and exogenous arrivals are one of three
    specific mechanisms by which real cascades violate our structured head's
    composition rule. That makes it the most diagnostically
    interesting row in this pool.

    **Deviation, and it is a large one:** none of our corpora publish a separate
    stimulus series, so the exogenous term is estimated from the cascade's own
    residual growth rather than measured. That is also why the `hip` external repo
    is registered BLOCKED rather than wired: `pyhip.HIP.initial` takes a
    `daily_share` series we do not have. A row here is HIP's kernel without HIP's
    contribution, and it is in the pool as a power-law-memory baseline rather than as
    an exogenous one.
    """
    waves = _waves(observation)
    _, theta = _hawkes_parameters(observation)
    remaining = max(horizon - observation.observed_steps, 0)

    total = float(observation.popularity)
    history = [float(count) for count in waves]

    for step in range(1, remaining + 1):
        # Power-law memory over the whole history, which is HIP's kernel shape
        intensity = sum(
            count / ((step + len(history) - index) ** (1.0 + theta))
            for index, count in enumerate(history)
        )
        history.append(intensity)
        total += intensity

    return max(float(observation.popularity), total)


def neighborhood_size(graph: GraphInfo, observation, horizon: int, **_: object) -> float:
    """
    `P(t_o)` plus the frontier's un-adopted out-neighbourhood: the one-step graph floor.

    The cheapest prediction that uses the graph at all, and the control that says
    whether a structural signal is worth anything here: if it ties with
    `persistence`, the topology is not carrying information about growth on this
    corpus.
    """
    reached = set(observation.adopters)
    exposed = {
        neighbour
        for node in observation.frontier
        for neighbour in graph.out_neighbors(node)
        if neighbour not in reached
    }

    return float(observation.popularity + len(exposed))


def reachability(graph: GraphInfo, observation, horizon: int, **_: object) -> float:
    """
    The structural CEILING: everything within `horizon - t_o` hops of the frontier.

    Not a serious predictor: it is an upper bound, and on a well-connected corpus
    it predicts most of the graph. It is in the pool because that bound is the
    single most useful diagnostic when an arm massively over-predicts: an arm at the
    ceiling has learned "the cascade reaches whatever it can reach", which is the
    saturation failure `ens_count_bias` exists to catch, arriving here as an
    outer-loop failure instead.
    """
    remaining = min(max(horizon - observation.observed_steps, 0), max_reach_depth)
    reached = set(observation.adopters)
    frontier = set(observation.frontier)

    for _ in range(remaining):
        wave = {
            neighbour
            for node in frontier
            for neighbour in graph.out_neighbors(node)
            if neighbour not in reached
        }
        if not wave:
            break

        reached |= wave
        frontier = wave

    return float(len(reached))


def degree_scaled(graph: GraphInfo, observation, horizon: int, **_: object) -> float:
    """
    `P(t_o)` scaled by the frontier's mean degree relative to the graph's.

    The structural analogue of Szabo & Huberman: a multiplier, but read off the
    graph rather than fitted from history. In the pool because CoupledGNN's is
    the only published protocol that withholds timestamps entirely and uses the
    early adopter set plus the global graph alone: this is the crudest member of
    that family and sets the bar its whole line has to clear.
    """
    features = cascade_features(graph, observation)
    mean_degree = 2.0 * graph.edge_index.shape[1] / max(graph.num_nodes, 1)
    ratio = features["frontier_mean_degree"] / max(mean_degree, 1e-6)
    remaining = max(horizon - observation.observed_steps, 0)

    return max(
        float(observation.popularity),
        float(observation.popularity) * (1.0 + ratio * remaining / max(horizon, 1)),
    )


def random_prediction(
    graph: GraphInfo, observation, horizon: int, fit_examples: list | None = None,
    seed: int = 0, **_: object
) -> float:
    """
    A draw from the training size distribution: the floor, and it is not a throwaway.

    The reason: Salganik/Dodds/Watts and Watts (2007) argued cascade size is
    close to inherently unpredictable, and Cheng et al. (WWW'14) is the direct
    rebuttal. The GAP between this row and everything else is the empirical form of
    that argument, exactly as the random-versus-degree gap is Pastor-Satorras &
    Vespignani's founding result on the immunization side.
    """
    rng = np.random.default_rng([seed, len(observation.adopters)])

    if not fit_examples:
        return float(max(observation.popularity, 1.0) * rng.uniform(1.0, 4.0))

    sizes = [float(example.actual) for example in fit_examples]

    return float(max(observation.popularity, sizes[int(rng.integers(len(sizes)))]))


def mc_forward(
    graph: GraphInfo, observation, horizon: int, predict=None, **_: object
) -> float | None:
    """
    Roll the observed prefix forward with the arm's own kernel and read the count.

    The "just simulate it" baseline, and the one that IS this repo's contribution
    wearing a baseline's clothes: under `@world_model` it is a forward pass of
    `f_theta`, under `@monte_carlo` it is `mc_runs` real episodes per step, and the
    difference between those two rows is the world model's cost claim with nothing
    else varying.

    Blocked from generated scripts by default, for the same reason `celf` and
    `mcmc_decode` are: a generated program is offline by construction and has no forward model,
    so nothing is lost by blocking it.
    the cost lands in this arm's `forecast_calls`.
    """
    if predict is None:
        return None

    steps = max(horizon - observation.observed_steps, 0)
    if steps <= 0:
        return float(observation.popularity)

    marginal = np.asarray(
        predict(list(observation.adopters), list(observation.frontier), steps),
        dtype=np.float64,
    )

    return float(max(observation.popularity, float(marginal.sum())))


prediction_algorithms = {
    # the feature line: `szabo_huberman` is the bar
    "szabo_huberman": szabo_huberman,
    "feature_linear": feature_linear,
    "feature_gbt": feature_gbt,
    "weng_communities": weng_communities,
    # the generative line, and the two that DECLINE
    "seismic": seismic,
    "hawkes": hawkes,
    "hawkes_hybrid": hawkes_hybrid,
    "rpp": rpp,
    "hip": hip,
    "branching_factor": branching_factor,
    # graph-only, the CoupledGNN family's floor
    "neighborhood_size": neighborhood_size,
    "degree_scaled": degree_scaled,
    "reachability": reachability,
    # the three floors
    "persistence": persistence,
    "mean_size": mean_size,
    "random_prediction": random_prediction,
    # kernel-using
    "mc_forward": mc_forward,
}

# Which members need the SELECTION split's labels to fit a constant. The harness
# passes `fit_examples` to these and to nothing else, so an unfitted member can
# never silently receive the evaluation pool.
fitted_prediction_algorithms = (
    "szabo_huberman",
    "feature_linear",
    "feature_gbt",
    "hawkes_hybrid",
    "mean_size",
    "random_prediction",
)

# Kernel-heavy, so one call costs `steps * forecast_samples` transition evaluations
# on whatever oracle it was handed. Charged honestly to the arm like any other
# baseline, and blocked from generated scripts by default.
mc_prediction_algorithms = ("mc_forward",)

prediction_algorithm_names = list(prediction_algorithms)
