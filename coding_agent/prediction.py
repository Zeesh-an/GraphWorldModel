"""
Cascade prediction: the observed instances, the forward-model binding, and the
error that scores one popularity predictor.

The harness for the FOURTH problem family, and the one that gives up the most.
`research/cascade_prediction.md` §2.1 is explicit: with `a_t = NULL` at every step
the world model degenerates from an action-conditioned simulator into a forecaster,
none of the five ops fire, `action_sensitivity` is undefined, and there is no
planning regret because there is no decision to plan over. §9.3 says outright that
this cannot demonstrate the capability the project is about.

It is here anyway, for the reason §9.1 gives and nothing else in `research/` can
claim: **this is the falsification test for our IC/LT assumption.** Every other task
in this repo evaluates a learned model against traces drawn from the same NDlib
simulator that trained it: a closed loop that can only ever measure learning error,
never modelling error. A real Weibo cascade breaks the loop. Our structured head's
`p_new(v) = 1 - prod(1 - q(u->v) * frontier_u)` is either an adequate approximation
of whatever produced those retweets or it is not, and §2.2 names three specific
mechanisms by which it is not: adoption is not memoryless (Hawkes self-excitation),
exposure is repeated rather than one-shot per neighbour, and exogenous arrivals
inject adopters with no infected in-neighbour at all. §9.2 says to expect to lose
and to design the experiment so that losing is informative.

Four pieces:

  * **`load_forecasts`**: the `(G, prefix, P(t_p))` instances, regrouped from
    transitions that were REPLAYED rather than simulated (`data/wm_cascades.py`). No
    new simulator and no new action op, but a hard precondition: the dataset must
    carry the `observed` block, because §2.2's whole argument collapses if the
    "real cascade" is an NDlib rollout.

  * **`bind_forecast_marginals`**: the ONE new primitive, and its four bindings.
    `@native` gets a raiser; `@monte_carlo` / `@oracle` / `@world_model` each unroll
    their own environment's `step_marginals`. The generated predictor is
    byte-identical across arms 3-6 and only its oracle changes, which is what keeps
    conditions 3-6 an ablation on one variable even though the ACTION space is
    empty. §2.1 says five of six arms have nothing to distinguish them; that is true
    of the action space and false of the forward model, and this is where the
    difference lives.

  * **`evaluate_predictor`**: the outer loop's reward, an ERROR that minimizes.
    §8.1's first sentence is "Get MSLE right or nothing else matters", and three
    independent choices hide inside that name (log base, total vs increment, the
    smoothing offset), so every variant is computed and the reward names which one
    it used.

  * **`trivial_predictor_error`**: what predicting the training geometric mean
    scores. The analogue of `trivial_decoder_reward`, and needed for a sharper
    reason than that one: MSLE is an error in LOG space, so an instance-blind
    constant is a much stronger baseline than intuition suggests, and a program
    search that only ties with it has learned the corpus's size distribution rather
    than anything about the instance.

Like the two inverse tasks, the reward is EXACT: it is measured against a
popularity we read off a log, so it carries no evaluator noise and is comparable
across conditions without the `--compare` referee. The referee still runs and
measures something else: what the ARM's own evaluator thinks the cascade would
have done, which is §9.1's modelling error in the units the rest of this repo
reports.
"""

import math
import time
from dataclasses import dataclass, field

import numpy as np

from coding_agent.executor import StrategyError, call_strategy
from coding_agent.types import GraphInfo, State, Strategy, TaskSpec, Trajectory
from data.wm_cascades import increment_target, total_target, valid_targets
from world_model.wm_data import load_episode_trajectories
from world_model.wm_metrics import (
    default_prediction_metric,
    doubling_accuracy,
    popularity_metrics,
    prediction_reward,
)

# Instances one reward evaluation sweeps over. Every candidate predictor pays this
# many calls, and under @monte_carlo each `forecast_marginals` inside one pays its
# own episodes, so it is the M of the cost story.
default_instances = 40

# Unrolls averaged inside one `forecast_marginals` call. Each costs `steps` metered
# kernel evaluations, so the total per call is `steps * samples`.
default_forecast_samples = 8

# Unrolls the `--compare` MODELLING referee averages. Far below the search's,
# deliberately: it is a diagnostic (§9.1) rather than a reward, and under
# @monte_carlo one unroll costs `steps * mc_runs` real episodes, so it is the single
# most expensive thing in a forecast run.
referee_forecast_samples = 2

# A node is treated as adopting in an unrolled realization at this probability. Only
# the SAMPLED unroll needs a threshold; the returned marginal is an average over
# realizations and stays continuous.
adoption_threshold = 0.5


@dataclass()
class CascadeObservation:
    """
    What a predictor is allowed to see: the cascade's first `t_o` steps.

    `adopters` maps every node that adopted inside the observation window to the
    timestep it did, so a predictor has the WHOLE observed history rather than a
    count, which is the input Cheng et al.'s temporal features (the class their
    paper found dominates) are computed from. `frontier` is the last observed wave,
    and it is what a forward model has to be seeded from: an adopter three steps ago
    has already transmitted.
    """

    cascade_id: str
    root: int
    num_nodes: int
    adopters: dict[int, int]
    frontier: tuple = ()
    # Steps of history shown, and steps to the prediction target. Both in replayed
    # timesteps; the corpus's own units are in `data/metadata.json`.
    observed_steps: int = 0
    horizon: int = 0
    # Wall-clock publication time, so a predictor can key off the diurnal rhythm the
    # Weibo protocol filters on (§8.2). Never the ANSWER: that is `actual`, and it
    # lives on the instance rather than here.
    publish_time: int = 0

    @property
    def popularity(self) -> int:
        """`P(t_o)`: adopters inside the observation window."""
        return len(self.adopters)

    def wave(self, step: int) -> list[int]:
        return sorted(node for node, when in self.adopters.items() if when == step)


@dataclass()
class ForecastInstance:
    """One logged cascade, split into what is shown and what must be predicted."""

    cascade_id: str
    graph_id: str
    observation: CascadeObservation
    # `P(t_p)`, the TOTAL popularity at the horizon. The increment is derivable
    # (`actual - observation.popularity`) rather than stored, so the two quantities
    # §5.7 difference 3 warns share a symbol can never disagree here.
    actual: int
    final_state: np.ndarray
    split: str = ""

    @property
    def increment(self) -> int:
        return max(self.actual - self.observation.popularity, 0)


@dataclass()
class ForecastOracle:
    """
    `forecast_marginals`, bound to one arm's evaluator and counting its own calls.

    Held as an object rather than a bare closure for the same reason
    `localization.ForwardOracle` and `reconstruction.StepOracle` are: the call count
    IS the cost claim. One call unrolls `steps` timesteps across `samples`
    realizations, so it costs `steps * samples` metered kernel evaluations, which
    an @monte_carlo arm pays in real episodes and a @world_model arm pays in
    matmuls. §2.4 lists "reaching published-table parity: not worth it" and the
    diagnostic as the point, and this counter is the diagnostic.
    """

    environment: object
    samples: int = default_forecast_samples
    seed: int = 0
    calls: int = field(default=0)
    kernel_calls: int = field(default=0)

    def __call__(self, adopters, frontier, steps: int) -> np.ndarray:
        """
        `P(v has adopted by t_o + steps)` per node, averaged over sampled unrolls.

        Mean-field would be cheaper and would be wrong in the direction that
        matters: composing expectations through a product form systematically
        under-states the variance a supercritical cascade has, which is exactly the
        regime §3.2 says the generative models fail in. Sampling realizations keeps
        the estimate honest and keeps every step a real metered kernel call.
        """
        steps = int(steps)
        self.calls += 1

        adopters = [int(node) for node in adopters]
        frontier = [int(node) for node in frontier]

        if steps <= 0 or not frontier:
            marginal = np.zeros(self._num_nodes(), dtype=np.float64)
            marginal[adopters] = 1.0

            return marginal

        totals = np.zeros(self._num_nodes(), dtype=np.float64)

        for sample in range(max(1, self.samples)):
            rng = np.random.default_rng([self.seed, self.calls, sample])
            infected = set(adopters)
            wave = list(frontier)

            for _ in range(steps):
                if not wave:
                    break

                probabilities = np.asarray(
                    self.environment.step_marginals(
                        State(sorted(infected), sorted(wave))
                    ),
                    dtype=np.float64,
                )
                self.kernel_calls += 1

                draw = rng.random(probabilities.shape[0]) < probabilities
                wave = [
                    int(node)
                    for node in np.flatnonzero(draw)
                    if node not in infected
                ]
                infected |= set(wave)

            totals[sorted(infected)] += 1.0

        return totals / max(1, self.samples)

    def _num_nodes(self) -> int:
        return int(self.environment.graph.num_nodes)


def unavailable_forecast_marginals(_adopters, _frontier, _steps) -> np.ndarray:
    """
    The `@native` binding: there is no forward model in this condition.

    Raising rather than being absent for the same reason the other two raisers do:
    an AttributeError traceback costs a whole refinement iteration and a message
    that names the condition costs one repair turn. The experimental condition is
    identical either way: §2.1 says the action space goes idle, so what conditions
    3-6 actually vary is the FORWARD MODEL, and this arm is the one that answers
    whether having one is worth anything at all. §3.1 and §5.1 both say it may not
    be: feature-driven regression "in some cases even beat[s] deep learning models",
    and Feature-based scores MSLE 1.9881 on APS-3y against CasFlow's 1.4370: closer
    than a decade of architecture would suggest.
    """
    raise StrategyError(
        "self.forecast_marginals is not available in this condition (@native): this "
        "arm has NO forward model, by design. It exists to measure whether one is "
        "worth anything at all for popularity prediction, and the literature "
        "suggests it may not be. Write a purely feature-driven or point-process "
        "predictor instead: Szabo-Huberman's log-linear rule on the observed count, "
        "the reshare rate in the second half of the window (Cheng et al.'s single "
        "best feature), a branching-factor extrapolation, a Hawkes fit on the wave "
        "series. `cascade_features(graph, observation)` gives you all of them."
    )


def expected_popularity(forecast, adopters, frontier, steps: int) -> float:
    """
    `E|adopted by t_o + steps|` from one `forecast_marginals` call.

    Derives from the primitive rather than being a second oracle, exactly as
    `transition_logprob` derives from `step_marginals`, so one kernel budget is
    behind both and the cost accounting stays honest.
    """
    marginal = np.asarray(forecast(adopters, frontier, steps), dtype=np.float64)

    return float(marginal.sum())


def observed_waves(episode: dict, window: int) -> dict[int, int]:
    """
    `{node: replay timestep}` for everything that adopted inside `[0, window]`.

    Read off `frontiers` rather than off `activation_time`, and the offset is the
    whole reason this is a function. `load_episode_trajectories` builds `frontiers`
    as `(steps + 1, N)` where row 0 is the episode's INITIAL frontier and row `r + 1`
    is record `r`'s `next_state.frontier`, so under our replay, where record `t`
    carries the wave binned at `t`, wave `t` is row `t + 1`. Its `activation_time`
    field applies a different convention (`enumerate(records, start=1)` with the
    sources forced to 0), which is correct for the simulated episodes cascade
    reconstruction scores and is off by one for a replayed one.

    Taking the rows directly means the prefix a predictor sees is exactly the
    `window` bins the protocol says it should, rather than one fewer: an error in
    the SAFE direction, which is why it would never have surfaced as a failure.
    """
    frontiers = episode["frontiers"]
    adopters = {}

    # `frontiers` has `len(records) + 1` rows, so wave `t` (row `t + 1`) exists only
    # up to `t = len(records) - 1`. A cascade SHORTER than the observation window is
    # the normal case rather than an edge one: most cascades are over long before
    # `t_o`, so the bound is a clamp rather than a guard.
    last = min(window, frontiers.shape[0] - 2)

    for step in range(last + 1):
        for node in np.flatnonzero(frontiers[step + 1] >= 0.5):
            adopters.setdefault(int(node), step)

    return adopters


def load_forecasts(
    data_dir: str,
    diffusion_model: str,
    split: str,
    graph_id: str | None = None,
    observed_steps: int = 0,
    limit: int = default_instances,
    seed: int = 0,
    require_observed: bool = True,
) -> list[ForecastInstance]:
    """
    Labelled `(G, prefix, P(t_p))` instances for one (dynamics, split).

    `require_observed` RAISES on a SIMULATED dataset rather than scoring it, and
    that guard is the whole methodological content of this loader. §2.2: "cascade
    prediction is the place where our IC/LT assumption is most directly
    falsifiable", and it is falsifiable only because the traces did not come from
    the assumption. Run this on NDlib output and every number is a measurement of
    how well an IC model fits IC data, which is what the other seven tasks already
    tell us.

    `limit` subsamples deterministically in `seed` rather than taking a prefix,
    because a replayed corpus is written in publication order and a prefix would be
    the first hours of the corpus alone.
    """
    from data.wm_cascades import dataset_is_observed, observed_protocol

    if require_observed and not dataset_is_observed(data_dir):
        observed_protocol(data_dir)  # raises with the corpus/task it actually holds

    protocol = observed_protocol(data_dir) if dataset_is_observed(data_dir) else {}
    window = observed_steps or int(protocol.get("observed_steps", 1))

    episodes = load_episode_trajectories(data_dir, diffusion_model, split)

    if graph_id is not None:
        episodes = [record for record in episodes if record["graph_id"] == graph_id]

    if not episodes:
        raise ValueError(
            f"no replayed cascades for graph {graph_id!r} in "
            f"{data_dir}/transitions_{diffusion_model}_{split}.jsonl. The data stage "
            f"must have run for this (corpus, dynamics, split)."
        )

    if limit and len(episodes) > limit:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(episodes), size=limit, replace=False)
        episodes = [episodes[int(index)] for index in sorted(chosen)]

    instances = []

    for episode in episodes:
        adopters = observed_waves(episode, window)

        # A cascade whose whole observed prefix is empty has nothing to predict from;
        # one whose root fell outside the window is a replay bug rather than a hard
        # instance, so both are dropped loudly by the count rather than scored
        if not adopters:
            continue

        instances.append(
            ForecastInstance(
                cascade_id=episode["episode_id"].rsplit("|", 1)[-1],
                graph_id=episode["graph_id"],
                observation=CascadeObservation(
                    cascade_id=episode["episode_id"].rsplit("|", 1)[-1],
                    root=int(episode["sources"][0]),
                    num_nodes=int(episode["num_nodes"]),
                    adopters=adopters,
                    # The LAST observed wave, not the whole adopter set: an adopter
                    # three steps ago has already had its chance to transmit, and
                    # seeding a forward model from all of them over-predicts badly
                    frontier=tuple(
                        node
                        for node, step in adopters.items()
                        if step == max(adopters.values())
                    ),
                    observed_steps=window,
                    horizon=int(episode["horizon"]),
                    publish_time=0,
                ),
                # `P(t_p)`: everything the log says ever adopted, which under a
                # replay IS the popularity at the horizon by construction
                actual=int(episode["infected_count"]),
                final_state=np.asarray(episode["final_state"], dtype=np.float32),
                split=split,
            )
        )

    if not instances:
        raise ValueError(
            f"every replayed cascade in {split} had an empty observation window at "
            f"{window} step(s). Raise --cp-observation or lower --cp-step."
        )

    return instances


def bind_forecast_marginals(
    environment: object, task: TaskSpec, seed: int = 0
) -> ForecastOracle | object:
    """The arm's forward model, or the raiser that stands in for it under @native."""
    if not task.forward_model:
        return unavailable_forecast_marginals

    return ForecastOracle(
        environment=environment, samples=task.forecast_samples, seed=seed
    )


def implemented(strategy: object, name: str) -> object | None:
    """
    The method `strategy` actually WROTE, or None if it only inherited the stub.

    Same identity check as the other two harnesses, and here for the same reason:
    `Strategy` is a `typing.Protocol` whose method bodies are `...`, so subclassing
    it inherits a `predict` that returns None, which this task would then read as a
    DECLINE rather than as a missing contract, and quietly score the arm as having
    refused every cascade.
    """
    written = getattr(type(strategy), name, None)

    if written is None or written is getattr(Strategy, name, None):
        return None

    return getattr(strategy, name)


def validate_prediction(value: object, instance: ForecastInstance) -> float | None:
    """
    Raise StrategyError unless `value` is a popularity or an explicit decline.

    None and non-finite both mean DECLINE, and that is a legitimate answer rather
    than an error (§8.4): SEISMIC returns no prediction for a supercritical cascade
    because the expected size diverges, and forcing a number there would be a guess
    dressed as a model. What is NOT legitimate is a popularity below what was already
    observed: a progressive cascade cannot shrink, and an observation is ground
    truth about the nodes it reports.
    """
    if value is None:
        return None

    try:
        popularity = float(value)
    except (TypeError, ValueError) as error:
        raise StrategyError(
            f"predict() must return a number (the popularity at the horizon) or "
            f"None to decline; got {type(value).__name__}."
        ) from error

    if not math.isfinite(popularity):
        return None

    observed = instance.observation.popularity

    if popularity < observed - 1e-9:
        raise StrategyError(
            f"predict() returned {popularity:.3f} for a cascade that had ALREADY "
            f"reached {observed} adopters inside the observation window. Adoption is "
            f"progressive: nobody un-adopts, so the prediction is bounded below by "
            f"`observation.popularity`. Return that value if you believe the cascade "
            f"is over."
        )

    if popularity > instance.observation.num_nodes:
        raise StrategyError(
            f"predict() returned {popularity:.3f} on a graph with "
            f"{instance.observation.num_nodes} nodes. Popularity counts DISTINCT "
            f"adopters, so it cannot exceed |V|."
        )

    return popularity


metric_keys = (
    "msle",
    "male",
    "msle_offset",
    "msle_natural",
    "msle_increment",
    "male_increment",
    "mape",
    "mape_casft",
    "mrse",
    "mrse_median",
    "wroperc",
    "ape_median",
    "ape_p75",
    "ape_p95",
    "pcc",
    "r2",
    "coverage",
    "n_scored",
    "n_failed",
    "decline_rate",
    "mean_predicted",
    "mean_actual",
)


def evaluate_predictor(
    strategy: object,
    environment: object,
    task: TaskSpec,
    graph: GraphInfo,
    instances: list[ForecastInstance],
    fit_examples: list[ForecastInstance] | None = None,
) -> tuple[Trajectory, float]:
    """
    Score one popularity predictor over the logged cascades.

    Returns the same `(Trajectory, plan_seconds)` pair `evaluate_strategy` does, so
    every method (one_shot, evolve, the checkpointing, the population feedback)
    consumes it unchanged. `reward` is the configured ERROR, which MINIMIZES: the
    one family in this repo whose reward does.
    """
    start = time.perf_counter()
    predict = implemented(strategy, "predict")

    if predict is None:
        raise StrategyError(
            "this task needs a predict() method and your class does not define one. "
            "The contract is\n"
            "    def predict(self, graph, observation, horizon) -> float | None\n"
            "returning the popularity the cascade will have reached by the horizon, "
            "or None to DECLINE scoring it (which is counted, not penalized as an "
            "error). plan_horizon(), act(), localize() and reconstruct() are other "
            "tasks' contracts and are not called here."
        )

    oracle = bind_forecast_marginals(environment, task, seed=task.seed)
    strategy.forecast_marginals = oracle
    strategy.expected_popularity = (
        (lambda adopters, frontier, steps: expected_popularity(
            oracle, adopters, frontier, steps
        ))
        if task.forward_model
        else unavailable_forecast_marginals
    )
    strategy.fit_examples = list(fit_examples or [])

    predicted, actual, observed = [], [], []
    per_instance = []
    declined = 0

    for instance in instances:
        value = validate_prediction(
            call_strategy(
                predict,
                graph,
                instance.observation,
                instance.observation.horizon,
                # None is a DECLINE here, not a missing return (§8.4)
                allow_none=True,
            ),
            instance,
        )

        entry = {
            "cascade_id": instance.cascade_id,
            "observed": instance.observation.popularity,
            "actual": instance.actual,
            "predicted": None if value is None else round(value, 4),
            "declined": value is None,
        }
        per_instance.append(entry)

        if value is None:
            declined += 1
            continue

        predicted.append(value)
        actual.append(float(instance.actual))
        observed.append(float(instance.observation.popularity))

    if not per_instance:
        raise StrategyError("no logged cascades to score this predictor against")

    metrics = popularity_metrics(
        np.array(predicted), np.array(actual), np.array(observed), declined=declined
    )
    metrics["doubling_accuracy"] = doubling_accuracy(
        np.array(predicted), np.array(actual), np.array(observed)
    )
    reward = prediction_reward(metrics, task.prediction_metric)
    elapsed = time.perf_counter() - start

    # A predictor that declined EVERY cascade is not a strong one that gave up, it
    # is a program that does not answer the question. Saying so here beats an
    # infinite reward the search then wanders around for --outer-iters turns.
    if declined == len(per_instance):
        raise StrategyError(
            f"predict() declined all {declined} cascades, so there is nothing to "
            f"score. Declining is legitimate for a GENERATIVE model on a "
            f"supercritical cascade (research/cascade_prediction.md §8.4: SEISMIC "
            f"does exactly that on ~3% of Tweet-1Mo), but a predictor that never "
            f"answers has no error to report. Fall back to a feature-driven estimate "
            f"when your generative fit diverges."
        )

    trajectory = Trajectory(
        # No cascade was rolled out; the popularity was read off a log
        states=[State([], []), State([], [])],
        actions=[[]],
        reward=reward,
        infected_counts=predicted,
        cost={
            "env": "cascade_prediction",
            "reward_se": (
                float(np.std(predicted, ddof=1) / np.sqrt(len(predicted)))
                if len(predicted) > 1
                else 0.0
            ),
            "rollout_seconds": elapsed,
            "n_instances": len(per_instance),
            "prediction_metric": task.prediction_metric,
            "prediction_target": task.prediction_target,
            "observation_window": task.observation_window,
            "metrics": metrics,
            "per_instance": per_instance,
            # The cost axis §2.4 exists to measure: an @monte_carlo arm pays
            # `mc_runs` real episodes per kernel call and a @world_model arm pays one
            # batched matmul, at `steps * forecast_samples` calls per instance
            "forecast_calls": getattr(oracle, "calls", 0),
            "kernel_calls": getattr(oracle, "kernel_calls", 0),
            "kernel_calls_per_instance": round(
                getattr(oracle, "kernel_calls", 0) / len(per_instance), 3
            ),
        },
        final_marginals=None,
        spread_curve=None,
    )

    return trajectory, elapsed


def trivial_predictor_error(
    instances: list[ForecastInstance],
    fit_examples: list[ForecastInstance],
    metric: str = default_prediction_metric,
) -> dict[str, float]:
    """
    What predicting the training GEOMETRIC MEAN scores, and what persistence scores.

    The analogue of `reconstruction.trivial_decoder_reward`, and needed for a
    sharper reason: MSLE is an error in LOG space, so its minimizer over an
    instance-blind constant is the geometric mean of the training sizes: a much
    stronger baseline than the same idea would be under squared error. An arm that
    only ties with this row has discovered the corpus's size distribution and
    nothing about the instance, and a reader cannot see that from the arm's own
    number alone. Reported into every results JSON for exactly that reason.

    `persistence_error` is the second floor and the one no paper in §5 prints: most
    cascades are over by `t_o`, so "predict what you see" is strong and an arm below
    it has at least learned that some are not.
    """
    if not instances:
        return {}

    actual = np.array([float(instance.actual) for instance in instances])
    observed = np.array(
        [float(instance.observation.popularity) for instance in instances]
    )

    pool = fit_examples or instances
    logs = [math.log(max(float(example.actual), 1.0)) for example in pool]
    constant = math.exp(float(np.mean(logs)))

    trivial = popularity_metrics(np.full(actual.shape, constant), actual, observed)
    persistence = popularity_metrics(observed.copy(), actual, observed)

    return {
        "trivial_predictor_error": prediction_reward(trivial, metric),
        "persistence_error": prediction_reward(persistence, metric),
        "trivial_predictor_value": round(constant, 3),
        "prediction_metric": metric,
    }


def referee_modelling_error(
    environment: object,
    task: TaskSpec,
    instances: list[ForecastInstance],
    samples: int = default_forecast_samples,
) -> dict[str, float]:
    """
    What the ARM'S OWN EVALUATOR predicts, rolled forward from each observed prefix.

    The `--compare` referee for this task, and the number §9.1 is actually about.
    The reward already measures how good a PROGRAM is; this measures how good the
    MODEL is: roll `f_theta` (or NDlib, or the analytic IC form) forward from the
    observed prefix with no program in the loop, and compare its expected popularity
    against what the log says happened. That difference is modelling error against a
    process that is not IC, which is the one quantity no other task in this repo can
    produce and the whole reason §9.1 calls this the falsification test.

    Reported alongside the program's own error so a reader can tell the two apart: a
    strong program on a badly misspecified kernel and a weak program on a good one
    look identical in the reward column alone.
    """
    if not task.forward_model:
        return {}

    oracle = ForecastOracle(environment=environment, samples=samples, seed=task.seed)
    predicted, actual, observed = [], [], []

    for instance in instances:
        steps = max(
            instance.observation.horizon - instance.observation.observed_steps, 0
        )
        estimate = expected_popularity(
            oracle,
            list(instance.observation.adopters),
            list(instance.observation.frontier),
            steps,
        )
        predicted.append(max(estimate, float(instance.observation.popularity)))
        actual.append(float(instance.actual))
        observed.append(float(instance.observation.popularity))

    if not predicted:
        return {}

    metrics = popularity_metrics(
        np.array(predicted), np.array(actual), np.array(observed)
    )

    return {
        "model_msle": metrics["msle"],
        "model_male": metrics["male"],
        "model_mape": metrics["mape"],
        "model_pcc": metrics["pcc"],
        "model_mean_predicted": metrics["mean_predicted"],
        "model_mean_actual": metrics["mean_actual"],
        "model_kernel_calls": oracle.kernel_calls,
    }


max_listed_cascades = 6


def summarize_prediction(
    trajectory: Trajectory, graph: GraphInfo, task: TaskSpec
) -> str:
    """
    What the predictions got right and wrong, in the direction the error actually runs.

    Neither `methods.base.summarize` (which describes where a cascade should go
    next) nor either inverse summary answers the question here. The split that
    matters is OVER- against UNDER-prediction: MSLE is symmetric in log space, so a
    model that misses every viral cascade and one that invents virality everywhere
    can post the same number, and only the sign of the residual tells them apart.
    That is also the failure §2.2 predicts specifically: a saturating IC rollout
    over-predicts, and our own `ens_count_bias` metric exists to catch exactly it.
    """
    cost = trajectory.cost
    metrics = cost.get("metrics", {})
    per_instance = cost.get("per_instance", [])
    metric = cost.get("prediction_metric", default_prediction_metric)

    scored = [entry for entry in per_instance if not entry["declined"]]
    residuals = [
        math.log2(max(entry["predicted"], 1.0)) - math.log2(max(entry["actual"], 1.0))
        for entry in scored
    ]
    over = sum(1 for value in residuals if value > 0)

    lines = [
        f"{metric.upper()}={trajectory.reward:.4f} (LOWER IS BETTER) over "
        f"{metrics.get('n_scored', 0)} logged cascades, "
        f"{metrics.get('n_failed', 0)} declined "
        f"({metrics.get('decline_rate', 0.0):.1%}, declining is legitimate for a "
        f"generative fit that diverges, and it is counted rather than penalized)",
        f"log-space errors: MSLE={metrics.get('msle', float('nan')):.4f}  "
        f"MALE={metrics.get('male', float('nan')):.4f}  "
        f"MAPE={metrics.get('mape', float('nan')):.4f}  "
        f"increment MSLE={metrics.get('msle_increment', float('nan')):.4f}",
        f"raw-count errors: median APE={metrics.get('ape_median', float('nan')):.3f}  "
        f"75th={metrics.get('ape_p75', float('nan')):.3f}  "
        f"95th={metrics.get('ape_p95', float('nan')):.3f}  "
        f"WroPerc={metrics.get('wroperc', float('nan')):.3f} "
        f"(the mean is outlier-dominated here; quantiles are what SEISMIC reports)",
        f"rank and tail: PCC={metrics.get('pcc', float('nan')):.4f}  "
        f"R2={metrics.get('r2', float('nan')):.4f}  "
        f"COV-10%={metrics.get('coverage', float('nan')):.3f} (did you find the "
        f"viral ones)  doubling accuracy="
        f"{metrics.get('doubling_accuracy', float('nan')):.3f}",
        f"forward model: {cost.get('kernel_calls', 0)} kernel calls total, "
        f"{cost.get('kernel_calls_per_instance', 0)} per cascade",
    ]

    if scored:
        lines.append(
            f"DIRECTION: you OVER-predicted {over}/{len(scored)} cascades "
            f"({over / len(scored):.0%}); mean log2 residual "
            f"{float(np.mean(residuals)):+.3f} "
            f"(predicted {metrics.get('mean_predicted', 0.0):.1f} on average against "
            f"{metrics.get('mean_actual', 0.0):.1f} actual)"
        )

        if abs(float(np.mean(residuals))) > 0.5:
            direction = "OVER" if float(np.mean(residuals)) > 0 else "UNDER"
            lines.append(
                f"DIAGNOSIS: you are systematically {direction}-predicting by about "
                f"{2 ** abs(float(np.mean(residuals))):.1f}x. MSLE is symmetric in "
                f"log space, so a constant multiplicative bias is the cheapest thing "
                f"to fix and it costs nothing structural: rescale before changing "
                f"the model."
            )

    if task.observes:
        lines.append(
            "NOTE: these are REAL logged cascades, not simulated ones. Whatever "
            "produced them is not Independent Cascade: adoption is not memoryless, "
            "exposure is repeated rather than one-shot per neighbour, and some "
            "adopters arrive with no adopting neighbour at all. A forward model that "
            "assumes otherwise will be wrong in a systematic direction, and the "
            "residual sign above is where that shows."
        )

    ranked = sorted(scored, key=lambda entry: abs(entry["predicted"] - entry["actual"]))

    for label, entry in (
        ("BEST", ranked[0] if ranked else None),
        ("WORST", ranked[-1] if ranked else None),
    ):
        if entry is None:
            continue

        lines.append(
            f"{label} cascade {entry['cascade_id']}: observed {entry['observed']} "
            f"-> predicted {entry['predicted']:.1f}, actually {entry['actual']}"
        )

    return "\n".join(lines)


class PredictAnchor:
    """
    Wraps a library predictor as the `predict()`-shaped object the sweep wants.

    Anchors go through `evaluate_predictor` rather than being scored separately, so
    they meet the arm they are setting a bar for under IDENTICAL conditions: the
    same cascades, the same observation window, the same metric.
    """

    def __init__(self, name: str, predictor, task: TaskSpec) -> None:
        self.name = name
        self.predictor = predictor
        self.task = task
        self.source_script = ""

    def predict(
        self, graph: GraphInfo, observation: CascadeObservation, horizon: int
    ) -> float | None:
        from coding_agent.tools.prediction_algorithms import (
            fitted_prediction_algorithms,
        )

        return self.predictor(
            graph,
            observation,
            horizon,
            # A label may reach a FITTED member's constant and never an unfitted
            # one's prediction: the pool is explicit about which members take the
            # selection split, so a leak would have to be a registry edit
            fit_examples=(
                list(getattr(self, "fit_examples", []))
                if self.name in fitted_prediction_algorithms
                else None
            ),
            predict=getattr(self, "forecast_marginals", None),
            seed=self.task.seed,
        )


def resolve_target(target: str) -> str:
    if target not in valid_targets:
        raise ValueError(
            f"unknown --cp-target {target!r}; choose one of {valid_targets}. "
            f"{increment_target!r} is CasFlow's own label and {total_target!r} is "
            f"CasFT's Eq. 26: research/cascade_prediction.md §5.7 difference 3 "
            f"records that the two share a symbol and are not the same quantity."
        )

    return target
