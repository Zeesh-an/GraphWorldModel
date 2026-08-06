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
the `--compare` referee to become so. The referee still runs, measuring the
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


def evaluate_localizer(
    strategy: object,
    environment: object,
    task: TaskSpec,
    graph: GraphInfo,
    instances: list[SourceInstance],
    budget_mode: str = episode_budget,
) -> tuple[Trajectory, float]:
    """
    Score one recovered-source program over the labelled episodes.

    Returns the same `(Trajectory, plan_seconds)` pair `evaluate_strategy` does, so
    every method (one_shot, evolve, the checkpointing, the population feedback)
    consumes it unchanged. `reward` is the mean F1 against the true sources, which
    maximizes: no sign work is needed anywhere, unlike the containment tasks
    (§2.7 item 3).
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

    oracle = bind_predict_marginals(environment, task)
    strategy.predict_marginals = oracle
    scorer = implemented(strategy, "source_scores")

    per_instance = []

    for instance in instances:
        budget = instance_budget(instance, task, budget_mode)
        predicted = call_strategy(
            localize, graph, instance.observation.copy(), budget
        )
        predicted = validate_sources(predicted, instance, budget)

        scores = None
        if scorer is not None:
            scores = np.asarray(
                call_strategy(scorer, graph, instance.observation.copy()),
                dtype=np.float64,
            )
            if scores.shape != (instance.num_nodes,):
                raise StrategyError(
                    f"source_scores() returned shape {scores.shape}, expected "
                    f"({instance.num_nodes},): one score per node, higher meaning "
                    f"more likely to be a source."
                )

        metrics = localization_metrics(
            predicted, instance.sources, instance.num_nodes, scores
        )
        metrics["episode_id"] = instance.episode_id
        metrics["budget"] = budget
        metrics["predicted"] = predicted
        metrics["sources"] = list(instance.sources)
        metrics["infected_count"] = instance.infected_count
        per_instance.append(metrics)

    if not per_instance:
        raise StrategyError("no labelled instances to score this program against")

    means = aggregate_metrics(per_instance)
    f1_values = [entry["f1"] for entry in per_instance]
    elapsed = time.perf_counter() - start

    # A representative recovered set, so the results JSON's timeline and the
    # --compare referee have concrete actions to replay: the seed commit that WOULD
    # reproduce the observation if the program got it right
    representative = per_instance[0]
    bag = [ActionOp("add_node", int(node)) for node in representative["predicted"]]

    trajectory = Trajectory(
        states=[State([], []), State(sorted(representative["predicted"]), [])],
        actions=[bag],
        reward=means["f1"],
        infected_counts=f1_values,
        cost={
            "env": "source_localization",
            "reward_se": (
                float(np.std(f1_values, ddof=1) / np.sqrt(len(f1_values)))
                if len(f1_values) > 1
                else 0.0
            ),
            "rollout_seconds": elapsed,
            "n_instances": len(per_instance),
            "budget_mode": budget_mode,
            "metrics": means,
            "per_instance": per_instance,
            # C in §2.3.2: forward evaluations this program performed, in total and
            # per instance. The inference-cost claim of §8.5.3 is read off this.
            "forward_calls": getattr(oracle, "calls", 0),
            "forward_calls_per_instance": round(
                getattr(oracle, "calls", 0) / len(per_instance), 3
            ),
            "auc_source": "source_scores" if scorer is not None else "rank_derived",
        },
        # No cascade was rolled out, so there is no per-node P(infected) to report.
        # reference_diff() already returns None on that, which is the right
        # behaviour: its seed-centric diff describes where a cascade went, and this
        # program did not run one.
        final_marginals=None,
        spread_curve=None,
    )

    return trajectory, elapsed


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
    Re-simulate each RECOVERED source set on the ground-truth simulator and compare
    it against what was observed.

    §8.5.5's second column, and the one metric in this literature with no baseline
    at all: §11 records that no surveyed paper reports a genuine re-simulated
    error, so it is self-contained and must never be presented as a cross-paper
    comparison. Reported beside the true-source error, because the number is
    meaningless without knowing what the ORACLE set scores: on an ill-posed
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

    return {
        "resim_error": float(np.mean(recovered_errors)),
        "resim_error_true_sources": float(np.mean(oracle_errors)),
        "resim_referee_calls": oracle.calls,
    }


max_listed_sources = 25


def summarize_localization(
    trajectory: Trajectory, graph: GraphInfo, task: TaskSpec
) -> str:
    """
    What the recovered sets got right and wrong, in the terms an inverse problem
    is actually debugged in.

    `methods.base.summarize`'s cascade diagnostics (residual gain, community reach,
    unreached nodes) all describe where a cascade SHOULD go next, which is the
    wrong question here: nothing was seeded and nothing spread. This replaces them.
    """
    cost = trajectory.cost
    means = cost.get("metrics", {})
    per_instance = cost.get("per_instance", [])
    lines = [
        f"F1={means.get('f1', 0.0):.4f} (±{cost.get('reward_se', 0.0):.4f} SE, "
        f"HIGHER IS BETTER) over {cost.get('n_instances', 0)} labelled episodes",
        f"precision={means.get('precision', 0.0):.4f}  "
        f"recall={means.get('recall', 0.0):.4f}  "
        f"auc={means.get('auc', float('nan')):.4f} "
        f"(from {cost.get('auc_source', 'rank_derived')})  "
        f"accuracy={means.get('accuracy', 0.0):.4f} "
        f"(accuracy is near-useless alone here: sources are a tiny minority class)",
        f"forward-model calls: {cost.get('forward_calls', 0)} total, "
        f"{cost.get('forward_calls_per_instance', 0)} per instance",
    ]

    if not per_instance:
        return "\n".join(lines)

    ranked = sorted(per_instance, key=lambda entry: entry["f1"])
    worst, best = ranked[0], ranked[-1]

    for label, entry in (("WORST", worst), ("BEST", best)):
        hit = sorted(set(entry["predicted"]) & set(entry["sources"]))
        missed = sorted(set(entry["sources"]) - set(entry["predicted"]))
        spurious = sorted(set(entry["predicted"]) - set(entry["sources"]))
        lines.append(
            f"{label} episode (F1={entry['f1']:.3f}, k={entry['budget']}, "
            f"{entry['infected_count']}/{graph.num_nodes} nodes infected): "
            f"hit {len(hit)}, missed {len(missed)}, spurious {len(spurious)}"
        )
        lines.append(
            "  missed sources (degree in brackets): "
            + (
                ", ".join(
                    f"{node}(d={graph.degree(node)})"
                    for node in missed[:max_listed_sources]
                )
                or "none"
            )
        )
        lines.append(
            "  nodes you named that were NOT sources: "
            + (
                ", ".join(
                    f"{node}(d={graph.degree(node)})"
                    for node in spurious[:max_listed_sources]
                )
                or "none"
            )
        )

    # The one systematic bias worth surfacing: are the false positives high-degree
    # hubs the cascade merely passed through, or peripheral nodes?
    spurious_degrees = [
        graph.degree(node)
        for entry in per_instance
        for node in set(entry["predicted"]) - set(entry["sources"])
    ]
    source_degrees = [
        graph.degree(node) for entry in per_instance for node in entry["sources"]
    ]
    if spurious_degrees and source_degrees:
        lines.append(
            f"degree bias: your false positives average degree "
            f"{np.mean(spurious_degrees):.1f} against {np.mean(source_degrees):.1f} "
            f"for the true sources: a large gap means you are naming hubs the "
            f"cascade travelled THROUGH rather than the nodes it started from"
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
