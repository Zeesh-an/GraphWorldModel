"""
Source localization: the labelled instances, the forward-oracle binding, and the
F1 sweep that scores one recovered-source program.

This is the harness for the INVERSE problem. Every other task in this repo has the
planner choose an intervention and steers a forward process; here it writes an
inference algorithm and the world model is a subroutine that algorithm calls
(`research/source_localization.md` §2.3, §2.5). Three pieces, and each one is
where a specific claim of that file lives:

  * **`load_instances`**: the labelled `(G, y, x)` episodes, regrouped from
    transitions that already exist (§2.1). No new simulator, no new action op, no
    regeneration run.

  * **`bind_predict_marginals`**: the ONE new primitive (§2.4.3), and its four
    bindings. `@native` gets a raiser, `@monte_carlo` / `@oracle` / `@world_model`
    each get their own environment's `Trajectory.final_marginals`. The generated
    program is byte-identical across those arms and only its oracle changes, which
    is what makes conditions 3-6 an ablation on one variable. Routing every call
    through `environment.rollout` is also what puts them in `evaluator_calls` /
    `evaluator_seconds` / `real_env_episodes`, i.e. §8.5.3's cost block.

  * **`evaluate_localizer`**: the outer loop's reward: mean F1 against the TRUE
    source set, over held-out episodes. §2.3.3 is emphatic that re-simulation error
    is the wrong selection signal here, because diffusion is many-to-one and a
    program that reliably recovers the wrong member of an equivalence class scores
    just as well under it. Labels select the program; they are never available to
    the program, which is what lets `A*` be deployed on real cascades.

The reward is EXACT rather than estimated, which is unusual for this pipeline and
worth stating: F1 against a known source set carries no evaluator noise, so a
recover task's `reward` is already comparable across conditions and does not need
the referee to become so. The referee still runs, measuring the
re-simulated error (§8.5.5): the metric this literature should report and does
not (§11).
"""

import time
from dataclasses import dataclass, field
from functools import partial
import numpy as np

from coding_agent.credit import planned_action
from coding_agent.executor import StrategyError, call_strategy
from coding_agent.types import (
    ActionOp,
    GraphInfo,
    State,
    Strategy,
    TaskSpec,
    Trajectory,
)
from world_model.wm_data import load_episode_endpoints
from world_model.wm_metrics import localization_metrics, resimulation_error

# Which form of `y` the program is handed. `marginal` is the MC-averaged
# P(infected), `binary` a single realized draw. §2.9 risk 5: ours is strictly more
# informative than the literature's, so the binarized column is the one comparable
# to §5.1 and both are reported.
marginal_observation = "marginal"
binary_observation = "binary"
valid_observations = (marginal_observation, binary_observation)

# Where `k` comes from. `episode` hands each localizer the episode's own source
# count, which is SL-VAE's given-k convention and makes F1 = PR = RE when the
# program spends its whole budget. `sweep` forces the pipeline's k on every
# instance and filters the pool to episodes near it, which is how §8.5.1's
# source-fraction axis is run.
episode_budget = "episode"
sweep_budget = "sweep"
valid_budget_modes = (episode_budget, sweep_budget)

# Relative band around the sweep's k under `sweep` mode: an episode is in scope
# when k / (1 + t) <= |x| <= k * (1 + t)
default_source_tolerance = 0.5

# Instances one reward evaluation sweeps over. Every candidate program pays this
# many executions, and under @monte_carlo each execution pays its own rollouts, so
# it is the multiplier M of §2.3.2's P * M * C.
default_instances = 20


@dataclass()
class SourceInstance:
    """One labelled cascade: the graph, what was observed, and what caused it."""

    episode_id: str
    graph_id: str
    num_nodes: int
    sources: list[int]
    observation: np.ndarray  # shape: (N,) in [0, 1]
    horizon: int
    infected_count: int
    algorithm: str | None = None
    # Which split this episode came from. Carried for the same reason
    # `CascadeInstance.split` is: a supervised external baseline (GCNSI, IVGD,
    # SL-VAE inside GraphSL) has to be told which rows it may FIT on, and the two
    # pools cross the process boundary concatenated. Episode ids are disjoint
    # between splits, so a "first occurrence" rule marks every row trainable and
    # leaks the evaluation labels into the model that is then scored on them.
    split: str = ""
    # The whole observed path, shape (T, N), one binary row per step. NOT given to
    # a generated program: its contract is a single snapshot, which is the
    # setting every comparable published number uses (§8.3). It is carried for the
    # external baselines that condition on intermediate observations rather than
    # on the endpoint alone, PDSL being the one wired today.
    trajectory: np.ndarray | None = None

    @property
    def source_count(self) -> int:
        return len(self.sources)


@dataclass()
class ForwardOracle:
    """
    `predict_marginals`, bound to one arm's evaluator and counting its own calls.

    Held as an object rather than a bare closure so the call count survives into
    the results JSON: `C` in §2.3.2's accounting is "forward evaluations one
    program performs on one instance", and it is the number §8.5.3 asks for
    per test instance. A closure would leave it uncounted.
    """

    environment: object
    task: TaskSpec
    calls: int = field(default=0)

    def __call__(self, seeds) -> np.ndarray:
        nodes = [int(node) for node in dict.fromkeys(int(node) for node in seeds)]

        if not nodes:
            return np.zeros(self.task_num_nodes, dtype=np.float64)

        # The seed commit is exactly the bag the generator writes at t = 0, so the
        # forward pass being asked for here is the same transition the world model
        # was trained on (§2.4.1)
        plan = [[ActionOp("add_node", node) for node in nodes]] + [
            [] for _ in range(self.task.horizon)
        ]
        trajectory = self.environment.rollout(
            partial(planned_action, plan), self.task.horizon, self.task.budget
        )
        self.calls += 1

        return np.asarray(trajectory.final_marginals, dtype=np.float64)

    @property
    def task_num_nodes(self) -> int:
        return int(self.environment.graph.num_nodes)


def unavailable_forward_oracle(_seeds) -> np.ndarray:
    """
    The `@native` binding: there is no forward model in this condition.

    Raising rather than being absent from the namespace, for the same reason
    `executor._blocked_algorithm` raises: an AttributeError traceback costs a whole
    refinement iteration, while a message that names the condition costs one repair
    turn. The experimental condition is identical either way: the program cannot
    call a forward model, so it has to be a pure structural heuristic (§2.4.3).
    """
    raise StrategyError(
        "self.predict_marginals is not available in this condition (@native): this "
        "arm has NO forward model, by design. It exists to measure whether a "
        "forward model in the search loop is worth anything at all. Write a purely "
        "structural inference rule instead: label propagation over the observed "
        "state, centralities restricted to the infected subgraph, per-component "
        "centres, community-aware separation constraints."
    )


def load_instances(
    data_dir: str,
    diffusion_model: str,
    split: str,
    graph_id: str | None = None,
    observation: str = marginal_observation,
    limit: int = default_instances,
    budget_mode: str = episode_budget,
    budget: int | None = None,
    source_tolerance: float = default_source_tolerance,
    seed: int = 0,
) -> list[SourceInstance]:
    """
    Labelled `(G, y, x)` episodes for one (dynamics, split), as SourceInstances.

    `limit` subsamples deterministically in `seed` rather than taking a prefix:
    episodes are written selector-major, so a prefix would be all-`random` or
    all-`degree` and the program would be selected against one seeding process.

    Under `sweep` budget mode the pool is filtered to episodes whose true source
    count sits within `source_tolerance` of the sweep's `k`, so a table row labelled
    "k = 5% of N" really is about 5%-of-N sources. An empty band RAISES rather than
    silently widening: a row computed over a different source fraction than its
    label claims is the single easiest way to publish an invalid comparison.
    """
    if observation not in valid_observations:
        raise ValueError(
            f"unknown observation mode {observation!r}; choose one of {valid_observations}"
        )

    if budget_mode not in valid_budget_modes:
        raise ValueError(
            f"unknown budget mode {budget_mode!r}; choose one of {valid_budget_modes}"
        )

    episodes = load_episode_endpoints(data_dir, diffusion_model, split)

    if graph_id is not None:
        episodes = [record for record in episodes if record["graph_id"] == graph_id]

    if not episodes:
        raise ValueError(
            f"no labelled episodes for graph {graph_id!r} in "
            f"{data_dir}/transitions_{diffusion_model}_{split}.jsonl. Source "
            f"localization reads the t=0 seed commit as its label, so the data "
            f"stage must have run for this (dataset, dynamics, split)."
        )

    if budget_mode == sweep_budget and budget:
        low = budget / (1.0 + source_tolerance)
        high = budget * (1.0 + source_tolerance)
        in_band = [
            record for record in episodes if low <= len(record["sources"]) <= high
        ]

        if not in_band:
            counts = sorted({len(record["sources"]) for record in episodes})
            raise ValueError(
                f"--sl-budget sweep at k={budget} keeps no episode: the {len(episodes)} "
                f"available ones have source counts {counts[:12]}"
                f"{' …' if len(counts) > 12 else ''}. Widen --sl-source-tolerance "
                f"(currently {source_tolerance}), pick a --budget-pcts value inside "
                f"that range, or regenerate with --budget-pct-range covering it."
            )

        episodes = in_band

    if limit and len(episodes) > limit:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(episodes), size=limit, replace=False)
        episodes = [episodes[int(index)] for index in sorted(chosen)]

    key = "marginal" if observation == marginal_observation else "binary"

    return [
        SourceInstance(
            episode_id=record["episode_id"],
            graph_id=record["graph_id"],
            num_nodes=record["num_nodes"],
            sources=record["sources"],
            observation=np.asarray(record[key], dtype=np.float64),
            horizon=record["horizon"],
            infected_count=record["infected_count"],
            algorithm=record.get("algorithm"),
            trajectory=record.get("trajectory"),
            split=split,
        )
        for record in episodes
    ]


def instance_budget(
    instance: SourceInstance, task: TaskSpec, budget_mode: str = episode_budget
) -> int:
    """`k` for one instance: its own source count, or the sweep's."""
    if budget_mode == episode_budget:
        return max(1, instance.source_count)

    return max(1, min(task.budget, instance.num_nodes))


def bind_predict_marginals(
    environment: object, task: TaskSpec
) -> ForwardOracle | object:
    """The arm's forward oracle, or the raiser that stands in for it under @native."""
    if not task.forward_model:
        return unavailable_forward_oracle

    return ForwardOracle(environment=environment, task=task)


def implemented(strategy: object, name: str) -> object | None:
    """
    The method `strategy` actually WROTE, or None if it only inherited the stub.

    `hasattr` is the wrong test here and the failure is silent. Generated code
    writes `class MyStrategy(Strategy)`, and `Strategy` is a `typing.Protocol`
    whose method bodies are `...`, so subclassing it inherits a `localize` and a
    `source_scores` that return `None`. `hasattr` then always says yes, which
    turns an OPTIONAL method into a required one that fails with "returned None"
    from inside validation, and turns a missing REQUIRED method into the same
    opaque error instead of the message that names the contract.

    Comparing the class's attribute against the Protocol's own function is the
    identity check `executor.build_strategy` already uses for the scored harness.
    """
    written = getattr(type(strategy), name, None)

    if written is None or written is getattr(Strategy, name, None):
        return None

    return getattr(strategy, name)


# Reward ---------------------------------------------------------------------
#
# The reward never sees the true sources. The harness rolls the RECOVERED set
# forward through the arm's own evaluator and scores how well that explains the
# observation: minus the mean squared error between the re-simulated
# P(infected) and the observed state, so 0 is a perfect explanation and higher
# is better. That is the same shape as every intervention task's reward (the
# arm's evaluator scores what the program produced) and it is computable at
# deployment, where no label exists. F1 against the true sources is computed
# AFTER the search, on the winner only, by `localization_label_metrics`.

# A node whose re-simulated infection probability differs from the observation
# by at least this much is listed in the feedback as over- or under-explained
residual_threshold = 0.25
max_listed_residuals = 12


def consistency_score(resimulated: np.ndarray, observation: np.ndarray) -> float:
    """Minus the re-simulation error: 0 means the named sources reproduce y exactly."""
    return -resimulation_error(resimulated, observation)


def residual_diagnostics(
    resimulated: np.ndarray,
    observation: np.ndarray,
    graph: GraphInfo,
    top: int = max_listed_residuals,
) -> dict:
    """
    Where the recovered set's cascade disagrees with the observation, node by node.

    `over` are nodes the re-simulation reaches that the observation says stayed
    clean (the named sources overshoot); `under` are observed infections the
    re-simulation misses (the named sources undershoot, or sit in the wrong
    component). Each carries its residual and its degree, which is what a
    localizer needs to move a pick and is label-free.
    """
    residual = np.asarray(resimulated, dtype=np.float64) - np.asarray(
        observation, dtype=np.float64
    )
    over = np.flatnonzero(residual >= residual_threshold)
    under = np.flatnonzero(residual <= -residual_threshold)
    over = over[np.argsort(-residual[over])][:top]
    under = under[np.argsort(residual[under])][:top]

    return {
        "n_over": int(np.count_nonzero(residual >= residual_threshold)),
        "n_under": int(np.count_nonzero(residual <= -residual_threshold)),
        "over": [
            [int(node), round(float(residual[node]), 3), int(graph.degree(node))]
            for node in over
        ],
        "under": [
            [int(node), round(float(residual[node]), 3), int(graph.degree(node))]
            for node in under
        ],
    }


consistency_keys = ("consistency", "resim_error", "n_named", "n_over", "n_under")


def _mean_over(per_instance: list[dict], keys: tuple) -> dict[str, float]:
    return {
        key: float(np.mean([entry[key] for entry in per_instance]))
        for key in keys
        if per_instance and all(key in entry for entry in per_instance)
    }

def evaluate_localizer(
    strategy: object,
    environment: object,
    task: TaskSpec,
    graph: GraphInfo,
    instances: list[SourceInstance],
    budget_mode: str = episode_budget,
) -> tuple[Trajectory, float]:
    """
    Score one recovered-source program over the selection episodes, label-free.

    Returns the same `(Trajectory, plan_seconds)` pair `evaluate_strategy` does, so
    every method (one_shot, evolve, the checkpointing, the population feedback)
    consumes it unchanged. `reward` is the mean consistency: minus the error with
    which the RECOVERED set, rolled forward through this arm's evaluator,
    reproduces the observation. It maximizes and it uses nothing a deployed
    localizer would not have. The true sources are read only by
    `localization_label_metrics`, after the search, on the winner.
    """
    start = time.perf_counter()
    localize = implemented(strategy, "localize")

    if localize is None:
        raise StrategyError(
            "this task needs a localize() method and your class does not define "
            "one. The contract is\n"
            "    def localize(self, graph, observation, budget) -> list[int]\n"
            "returning the node ids you believe STARTED the cascade. plan_horizon() "
            "and act() are the contracts for the intervention tasks and are not "
            "called here."
        )

    # What the PROGRAM may call: the four bindings, a raiser under @native
    # Bound to canned baselines only: a generated localizer is offline
    oracle = None
    if getattr(strategy, "canned", False):
        oracle = bind_predict_marginals(environment, task)
        strategy.predict_marginals = oracle
    # What the HARNESS scores with: the same rollout, counted apart from the
    # program's own calls, and available under every condition. The native arm
    # is scored on one real episode exactly as an intervention task's native arm
    # is, rather than on nothing.
    scorer = ForwardOracle(environment=environment, task=task)

    per_instance = []
    for instance in instances:
        budget = instance_budget(instance, task, budget_mode)
        predicted = call_strategy(
            localize, graph, instance.observation.copy(), budget
        )
        predicted = validate_sources(predicted, instance, budget)
        resimulated = scorer(predicted)

        entry = {
            "episode_id": instance.episode_id,
            "budget": budget,
            "predicted": predicted,
            "infected_count": instance.infected_count,
            "consistency": consistency_score(resimulated, instance.observation),
            "resim_error": resimulation_error(resimulated, instance.observation),
            "n_named": len(predicted),
        }
        entry |= residual_diagnostics(resimulated, instance.observation, graph)
        per_instance.append(entry)

    if not per_instance:
        raise StrategyError("no episodes to score this program against")

    means = _mean_over(per_instance, consistency_keys)
    rewards = [entry["consistency"] for entry in per_instance]
    elapsed = time.perf_counter() - start

    # A representative recovered set, so the results JSON's timeline and the
    # referee have concrete actions to replay: the seed commit that WOULD
    # reproduce the observation if the program got it right
    representative = per_instance[0]
    bag = [ActionOp("add_node", int(node)) for node in representative["predicted"]]

    trajectory = Trajectory(
        states=[State([], []), State(sorted(representative["predicted"]), [])],
        actions=[bag],
        reward=means["consistency"],
        infected_counts=rewards,
        cost={
            "env": "source_localization",
            "reward_se": (
                float(np.std(rewards, ddof=1) / np.sqrt(len(rewards)))
                if len(rewards) > 1
                else 0.0
            ),
            "rollout_seconds": elapsed,
            "n_instances": len(per_instance),
            "budget_mode": budget_mode,
            "metrics": means,
            "per_instance": per_instance,
            # C in §2.3.2: forward evaluations the PROGRAM performed, in total and
            # per instance. The inference-cost claim of §8.5.3 is read off this.
            "forward_calls": getattr(oracle, "calls", 0),
            "forward_calls_per_instance": round(
                getattr(oracle, "calls", 0) / len(per_instance), 3
            ),
            # ...and the harness's own scoring rollouts, one per instance
            "scoring_calls": scorer.calls,
            "auc_source": (
                "source_scores"
                if implemented(strategy, "source_scores") is not None
                else "rank_derived"
            ),
        },
        # No cascade was rolled out for the plan, so there is no per-node
        # P(infected) to report. reference_diff() already returns None on that.
        final_marginals=None,
        spread_curve=None,
    )

    return trajectory, elapsed


def localization_label_metrics(
    strategy: object | None,
    trajectory: Trajectory,
    instances: list[SourceInstance],
    graph: GraphInfo,
) -> dict[str, float]:
    """
    F1, precision, recall, AUC and accuracy against the TRUE sources, after the fact.

    Called once per split on the winning program only, after the search has
    finished and the closing write-up has been requested, so no label reaches a
    prompt or a selection decision. The literature reports F1, so it stays the
    reported column; it just stops being the reward. Merges the label metrics
    into the trajectory's `metrics` and `per_instance` blocks in place.
    """
    by_episode = {instance.episode_id: instance for instance in instances}
    scorer = implemented(strategy, "source_scores") if strategy is not None else None
    labelled = []

    for entry in trajectory.cost.get("per_instance", []):
        instance = by_episode.get(entry["episode_id"])
        if instance is None:
            continue

        scores = None
        if scorer is not None:
            try:
                scores = np.asarray(
                    call_strategy(scorer, graph, instance.observation.copy()),
                    dtype=np.float64,
                )
            except StrategyError as error:
                # The winner's optional ranking hook failing after the search is a
                # reporting detail, not a failed arm: fall back to the rank-derived
                # AUC and say so
                print(f"[run] source_scores() failed post hoc, AUC is rank-derived: {error}")
                scores = None
                trajectory.cost["auc_source"] = "rank_derived (source_scores failed)"
            if scores is not None and scores.shape != (instance.num_nodes,):
                print(
                    f"[run] source_scores() returned shape {scores.shape}, expected "
                    f"({instance.num_nodes},): AUC is rank-derived"
                )
                scores = None
                trajectory.cost["auc_source"] = "rank_derived (source_scores failed)"

        metrics = localization_metrics(
            entry["predicted"], instance.sources, instance.num_nodes, scores
        )
        metrics["sources"] = list(instance.sources)
        entry.update(metrics)
        labelled.append(metrics)

    means = aggregate_metrics(labelled) if labelled else {}
    trajectory.cost["metrics"] = {**trajectory.cost.get("metrics", {}), **means}

    return means


def validate_sources(
    predicted: object, instance: SourceInstance, budget: int
) -> list[int]:
    """
    Raise StrategyError unless `predicted` is a legal source set of at most `budget`.

    Deliberately NOT `executor.validate_actions`: that validates an ACTION bag
    against an op whitelist, and a recovered source set is an inference, not an
    intervention. The rules that do carry over are the ones about spending the
    budget: a duplicate wastes a slot and an over-length set inflates recall for
    free.
    """
    try:
        nodes = [int(node) for node in predicted]
    except (TypeError, ValueError) as error:
        raise StrategyError(
            f"localize() must return a list of integer node ids; got "
            f"{type(predicted).__name__} ({error})"
        ) from error

    for node in nodes:
        if not 0 <= node < instance.num_nodes:
            raise StrategyError(
                f"localize() returned node {node}, outside [0, {instance.num_nodes})."
            )

    if len(set(nodes)) != len(nodes):
        raise StrategyError(
            "localize() returned a duplicate node. A repeat spends one of your "
            "k slots on a node you already named and cannot raise recall; "
            "deduplicate before returning."
        )

    if len(nodes) > budget:
        raise StrategyError(
            f"localize() returned {len(nodes)} sources, exceeding the budget "
            f"{budget}. Returning extra nodes would buy recall at no cost to "
            f"precision, which is why it is rejected: return at most k."
        )

    return nodes


metric_keys = (
    "precision",
    "recall",
    "f1",
    "accuracy",
    "auc",
    "n_predicted",
    "n_sources",
    "true_positive",
)


def aggregate_metrics(per_instance: list[dict]) -> dict[str, float]:
    """Instance-averaged PR / RE / F1 / AUC: SL-VAE's own aggregation (§8.4 trap 5)."""
    return {
        key: float(np.nanmean([entry[key] for entry in per_instance]))
        for key in metric_keys
    }


def referee_resimulation_error(
    environment: object,
    task: TaskSpec,
    instances: list[SourceInstance],
    per_instance: list[dict],
) -> dict[str, float]:
    """
    The referee: the reward re-measured on the referee's own simulator (the exact
    oracle by default, NDlib for the agreement check).

    Re-simulate each RECOVERED source set on NDlib and compare it against what was
    observed, exactly the quantity the arm's own evaluator scored in the loop, so
    `referee_reward` here is to `reward` what the referee replay is to a world-model spread.
    Reported beside the TRUE source set's own error, because the number is
    unreadable without knowing what the oracle set scores: on an ill-posed
    problem a recovered set can reproduce `y` better than the truth did.
    """
    by_episode = {entry["episode_id"]: entry for entry in per_instance}
    recovered_errors = []
    oracle_errors = []
    oracle = ForwardOracle(environment=environment, task=task)

    for instance in instances:
        entry = by_episode.get(instance.episode_id)
        if entry is None:
            continue

        recovered_errors.append(
            resimulation_error(oracle(entry["predicted"]), instance.observation)
        )
        oracle_errors.append(
            resimulation_error(oracle(instance.sources), instance.observation)
        )

    if not recovered_errors:
        return {}

    rewards = [-error for error in recovered_errors]

    return {
        "referee_reward": float(np.mean(rewards)),
        "referee_reward_se": (
            float(np.std(rewards, ddof=1) / np.sqrt(len(rewards)))
            if len(rewards) > 1
            else 0.0
        ),
        "resim_error": float(np.mean(recovered_errors)),
        "resim_error_true_sources": float(np.mean(oracle_errors)),
        "resim_referee_calls": oracle.calls,
    }


max_listed_sources = 25


def summarize_localization(
    trajectory: Trajectory, graph: GraphInfo, task: TaskSpec
) -> str:
    """
    Where the recovered sets' cascades disagree with the observation, label-free.

    `methods.base.summarize`'s cascade diagnostics (residual gain, community reach,
    unreached nodes) all describe where a cascade SHOULD go next, which is the
    wrong question here. This describes the residual instead: the nodes the named
    sources over-explain and under-explain when rolled forward, which is what the
    reward is made of and everything a deployed localizer could also see.
    """
    cost = trajectory.cost
    means = cost.get("metrics", {})
    per_instance = cost.get("per_instance", [])

    lines = [
        f"consistency={trajectory.reward:.5f} (±{cost.get('reward_se', 0.0):.5f} SE, "
        f"HIGHER IS BETTER, 0 = the sources you named re-simulate to exactly the "
        f"observation) over {cost.get('n_instances', 0)} episodes: minus the mean "
        f"squared error between P(infected) re-simulated from your sources and the "
        f"observed state",
        f"per episode: {means.get('n_named', 0.0):.1f} sources named, "
        f"{means.get('n_over', 0.0):.1f} nodes OVER-explained (your cascade reaches "
        f"them, the observation says clean), {means.get('n_under', 0.0):.1f} nodes "
        f"UNDER-explained (observed infected, your cascade misses them)",
        f"harness scoring rollouts: {cost.get('scoring_calls', 0)} (your program is "
        f"offline and makes none)",
    ]

    if not per_instance:
        return "\n".join(lines)

    ranked = sorted(per_instance, key=lambda entry: entry["consistency"])
    worst, best = ranked[0], ranked[-1]

    for label, entry in (("WORST", worst), ("BEST", best)):
        named = ", ".join(
            f"{node}(d={graph.degree(node)})"
            for node in entry["predicted"][:max_listed_sources]
        )
        lines.append(
            f"{label} episode (consistency={entry['consistency']:.5f}, "
            f"k={entry['budget']}, {entry['infected_count']}/{graph.num_nodes} nodes "
            f"observed infected): named {named or 'nothing'}"
        )
        lines.append(
            "  over-explained (node, residual, degree): "
            + (
                ", ".join(
                    f"{node}({residual:+.2f}, d={degree})"
                    for node, residual, degree in entry["over"]
                )
                or "none"
            )
        )
        lines.append(
            "  under-explained (node, residual, degree): "
            + (
                ", ".join(
                    f"{node}({residual:+.2f}, d={degree})"
                    for node, residual, degree in entry["under"]
                )
                or "none"
            )
        )

    over = means.get("n_over", 0.0)
    under = means.get("n_under", 0.0)
    if over > under and over > 0:
        lines.append(
            "DIAGNOSIS: your sets OVERSHOOT: the sources you name spread into regions "
            "the observation never reached, which is what naming a hub the cascade "
            "merely passed through looks like. Prefer nodes whose forward reach stays "
            "inside the observed region: peripheral members of it, one per component."
        )
    elif under > over and under > 0:
        lines.append(
            "DIAGNOSIS: your sets UNDERSHOOT: parts of the observed region stay "
            "unexplained. The missing sources sit inside those under-explained "
            "components; name one node per unexplained component before refining "
            "the rest."
        )

    return "\n".join(lines)


class LocalizeAnchor:
    """
    Wraps a library localizer as the `localize()`-shaped object the sweep wants.

    Anchors go through `evaluate_localizer` rather than being scored separately, so
    they meet the arm they are setting a bar for under IDENTICAL conditions: the
    same instances, the same k, the same observation mode.
    """

    def __init__(self, name: str, selector, scorer, task: TaskSpec) -> None:
        self.name = name
        self.selector = selector
        self.scorer = scorer
        self.task = task
        self.source_script = ""

    def localize(self, graph: GraphInfo, observation: np.ndarray, budget: int) -> list[int]:
        return [
            int(node)
            for node in self.selector(
                graph,
                observation,
                budget,
                diffusion_model=self.task.diffusion_model,
                horizon=self.task.horizon,
            )
        ]

    def source_scores(self, graph: GraphInfo, observation: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.scorer(graph, observation, diffusion_model=self.task.diffusion_model),
            dtype=np.float64,
        )
