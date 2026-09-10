"""
Cascade reconstruction: the masked episodes, the transition-kernel binding, and
the tree-weighted reward that scores one trajectory decoder.

This is the harness for the second INVERSE problem, and it differs from
`localization.py` in exactly the way `research/cascade_reconstruction.md` §2.5
says it should: the recovered object is a whole history rather than a set, the
primitive is the raw kernel evaluated at arbitrary proposed states rather than a
seed-set forward pass, and the reward is weighted toward the half of the problem
that is actually hard. Four pieces:

  * **`load_cascades`**: the labelled `(G, O, Y)` episodes, regrouped from
    transitions that already exist (§2.2). No new simulator, no new action op, no
    regeneration run, but they must have been generated with `--trace-parents`,
    because NDlib emits no transmission edge and §2.6 makes one a precondition
    rather than an enhancement.

  * **`mask_observation`**: the four settings of §2.7 as four masks over data
    already on disk. Which one is running is a PROTOCOL parameter and the four are
    four separate experiments, never pooled (§8.3).

  * **`bind_step_marginals`**: the ONE new primitive (§2.5.2), and its four
    bindings. `@native` gets a raiser; `@monte_carlo` / `@oracle` / `@world_model`
    each get their own environment's `step_marginals`. The generated decoder is
    byte-identical across arms 3-6 and only its oracle changes, which is what
    makes conditions 3-6 an ablation on one variable, and at ~10^4 kernel calls
    per instance it is where the cost claim of §2.4.2 is measured.

  * **`evaluate_reconstructor`**: the outer loop's reward,
    `lambda * PathPrecision + (1 - lambda) * EventF1`. §2.6 is the section to read
    twice: the obvious reward (Event F1 alone) is actively dangerous here, because
    the node set is nearly free and a program SEARCH rewarded on it discovers that
    the tree contributes nothing to its score and converges on decoders that never
    attempt the hard half. The reward is the specification.

Like source localization, the reward is EXACT: it is computed against a history we
stored, so it carries no evaluator noise and is comparable across conditions
without the referee. The referee still runs and measures something
else: re-simulating the RECOVERED sources against what was observed (§8.5.5's
analogue), which is the column no paper in this literature reports.
"""

import time
from dataclasses import dataclass, field
import numpy as np

from coding_agent.executor import StrategyError, call_strategy
from coding_agent.types import (
    ActionOp,
    GraphInfo,
    State,
    Strategy,
    TaskSpec,
    Trajectory,
)
from world_model.wm_data import load_episode_trajectories
from world_model.wm_metrics import (
    default_tree_weight,
    reconstruction_metrics,
    reconstruction_reward,
    resimulation_error,
)

# The four settings of §2.7, and they are four PROTOCOLS rather than four knobs:
# a decoder selected under `final_snapshot` is solving a different problem from
# one selected under `partial_times`, so the rows are reported separately and
# never pooled (§8.3).
partial_nodes = "partial_nodes"  # (a) a subsample of the infected set, NO times
partial_times = "partial_times"  # (c-i) the same subsample, WITH activation times
final_snapshot = "final_snapshot"  # (c-ii) the terminal state only: DITTO's DASH
hidden_nodes = "hidden_nodes"  # (d) nodes deleted from the graph, not merely unobserved
valid_settings = (partial_nodes, partial_times, final_snapshot, hidden_nodes)

# Fraction of the infected set that is REPORTED. §8.2 trap 1: this literature
# cannot agree on the direction of its own masking parameter: Xiao sweeps report
# probability, Sadikov sweeps sample ratio, and Zong sweeps UNCERTAINTY, which is
# the complement of Sadikov's under the same symbol. Ours is the report
# probability (higher = more observed) and every table has to say so.
default_observation_rate = 0.3

# Fraction of nodes deleted from the adjacency under `hidden_nodes`. Distinct
# from the observation rate: an unobserved node is still in the graph and can
# still be inferred, a hidden one is not there at all.
default_hidden_rate = 0.1

# Instances one reward evaluation sweeps over. Every candidate decoder pays this
# many executions, and under @monte_carlo each execution pays its own kernel
# samples, so it is the M of §2.4.2's P * M * (S * T).
default_instances = 20


@dataclass()
class Observation:
    """
    What a decoder is allowed to see: §2.5.1's dataclass, verbatim.

    `reported` maps an observed node to its activation time, or to None for
    "known infected, time unknown". `final_state` is the terminal snapshot, which
    is all there is under the DASH setting. `visible` is the node mask for the
    hidden-node regime, where a node is absent from the ADJACENCY rather than
    merely unobserved.
    """

    reported: dict[int, int | None]
    horizon: int
    num_nodes: int
    setting: str = partial_times
    final_state: np.ndarray | None = None
    visible: np.ndarray | None = None

    @property
    def infected(self) -> list[int]:
        """Every node the observation says was infected, in id order."""
        if self.final_state is not None:
            return [int(node) for node in np.flatnonzero(self.final_state >= 0.5)]

        return sorted(int(node) for node in self.reported)

    @property
    def times(self) -> dict[int, int]:
        """Only the reports whose activation time is known."""
        return {
            int(node): int(time)
            for node, time in self.reported.items()
            if time is not None
        }

    def is_visible(self, node: int) -> bool:
        return self.visible is None or bool(self.visible[int(node)])


@dataclass()
class CascadeInstance:
    """One labelled episode: the graph, what was masked, and the whole truth."""

    episode_id: str
    graph_id: str
    num_nodes: int
    horizon: int
    sources: list[int]
    # node -> activation step, for every node that ever activated
    true_times: dict[int, int]
    # node -> the causing node(s). None when the dataset carries no transmission
    # edge at all, which makes every tree metric unscoreable (§2.6).
    true_parents: dict[int, list[int]] | None
    observation: Observation
    final_state: np.ndarray
    marginal: np.ndarray
    algorithm: str | None = None
    # Which split this cascade came from. Carried because a SUPERVISED external
    # baseline has to be told which rows it may fit on, and the two pools cross
    # the process boundary concatenated: episode ids are disjoint between splits,
    # so "first occurrence" cannot tell them apart and would mark every row
    # trainable. `registry._reconstruction_export` reads this instead.
    split: str = ""

    @property
    def infected_count(self) -> int:
        return len(self.true_times)

    @property
    def observed_count(self) -> int:
        return len(self.observation.reported)


@dataclass()
class StepOracle:
    """
    `step_marginals`, bound to one arm's evaluator and counting its own calls.

    Held as an object rather than a bare closure for the same reason
    `localization.ForwardOracle` is: the call count is the whole cost claim.
    §2.4.2 puts a trajectory decoder two orders of magnitude above a localizer,
    10^4 kernel evaluations per instance against 10^2, and §11 records that the
    sampling arm's budget is dominated by a constant nobody has published, so it
    has to be measured rather than assumed.
    """

    environment: object
    calls: int = field(default=0)

    def __call__(self, infected, frontier) -> np.ndarray:
        state = State(
            sorted(int(node) for node in infected),
            sorted(int(node) for node in frontier),
        )
        self.calls += 1

        return np.asarray(
            self.environment.step_marginals(state), dtype=np.float64
        )


def unavailable_step_marginals(_infected, _frontier) -> np.ndarray:
    """
    The `@native` binding: there is no transition kernel in this condition.

    Raising rather than being absent for the same reason
    `localization.unavailable_forward_oracle` does: an AttributeError traceback
    costs a whole refinement iteration, a message that names the condition costs
    one repair turn. The experimental condition is identical either way: §2.9
    arm 3 exists to answer whether a kernel in the decode loop is worth anything
    at all, so the program must be a pure structural or temporal heuristic.
    """
    raise StrategyError(
        "self.step_marginals is not available in this condition (@native): this "
        "arm has NO transition kernel, by design. It exists to measure whether a "
        "kernel in the decode loop is worth anything at all. Write a purely "
        "structural or temporal decoder instead: Steiner trees over the reported "
        "nodes, BFS/shortest-path orderings that respect the observed times, "
        "personalized PageRank from the reports, per-component centres."
    )


def transition_logprob(
    marginal: np.ndarray, infected, next_frontier
) -> float:
    """
    `log p(s_{t+1} | s_t, G)` from one kernel evaluation.

    §2.5.2: this DERIVES from `step_marginals` rather than being a second oracle,
    so one kernel call is behind both. Under IC each susceptible node activates
    independently given the frontier, so the step's log-likelihood is the sum of
    `log p_v` over the nodes that did activate and `log(1 - p_v)` over the
    susceptible ones that did not.

    §8.1 marks trajectory log-likelihood "nobody: this is ours to add": it is the
    one metric in this literature that needs a kernel to evaluate under, which is
    exactly why it is the natural INTERNAL signal for a decoder on unlabelled data
    (§2.8). Labels select the program; the program itself may run on this.
    """
    marginal = np.clip(np.asarray(marginal, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    active = np.zeros(marginal.shape[0], dtype=bool)
    active[list(int(node) for node in infected)] = True

    activated = np.zeros(marginal.shape[0], dtype=bool)
    activated[list(int(node) for node in next_frontier)] = True

    susceptible = ~active
    gained = susceptible & activated
    missed = susceptible & ~activated

    return float(
        np.log(marginal[gained]).sum() + np.log1p(-marginal[missed]).sum()
    )


def mask_observation(
    episode: dict, setting: str, observation_rate: float, hidden_rate: float, rng: np.random.Generator
) -> tuple[Observation, np.ndarray | None]:
    """
    One episode's stored history, masked into the observation a decoder gets.

    Zero generator changes: §2.2's whole asset is that the ground truth these
    papers spend sections approximating is something we can read back. The mask is
    a PROTOCOL parameter and the four settings are four experiments (§8.3):

      * `partial_nodes`: each infected node is reported w.p. `observation_rate`,
        with its time withheld. The Xiao/Sadikov/NetFill regime.
      * `partial_times`: the same subsample, times included. Xiao SDM'18's
        `OrderedSteinerTree` input exactly.
      * `final_snapshot`: no reports at all, only the terminal state. DITTO's
        DASH formulation, the hardest published one, and the one worth leading
        with because it is where a learned kernel should beat a mean-field
        beta-hat.
      * `hidden_nodes`: `partial_times` plus a node mask. A hidden node is gone
        from the ADJACENCY rather than merely unobserved, which is what makes it a
        different problem from a low observation rate; it is excluded from the
        truth as well, since a decoder cannot name a node that is not there.

    Returns `(observation, visible_mask)`; the mask is None for every setting but
    the last.
    """
    if setting not in valid_settings:
        raise ValueError(
            f"unknown reconstruction setting {setting!r}; choose one of {valid_settings}"
        )

    num_nodes = episode["num_nodes"]
    activation_time = episode["activation_time"]
    infected = [int(node) for node in np.flatnonzero(activation_time >= 0)]

    visible = None
    if setting == hidden_nodes:
        # Hide a fraction of the NON-SOURCE nodes: hiding patient zero would make
        # the instance unsolvable rather than harder, in the same way removing an
        # outbreak source is rejected on the containment side
        candidates = [node for node in range(num_nodes) if node not in episode["sources"]]
        count = int(round(len(candidates) * hidden_rate))
        visible = np.ones(num_nodes, dtype=bool)

        if count:
            chosen = rng.choice(len(candidates), size=count, replace=False)
            visible[[candidates[int(index)] for index in chosen]] = False

        infected = [node for node in infected if visible[node]]

    if setting == final_snapshot:
        return (
            Observation(
                reported={},
                horizon=episode["horizon"],
                num_nodes=num_nodes,
                setting=setting,
                final_state=np.asarray(episode["final_state"], dtype=np.float32),
                visible=visible,
            ),
            visible,
        )

    keep = rng.random(len(infected)) < observation_rate
    reported = {}

    for node, observed in zip(infected, keep, strict=True):
        if not observed:
            continue

        reported[int(node)] = (
            int(activation_time[node]) if setting != partial_nodes else None
        )

    # An empty report set is a degenerate instance rather than a hard one: nothing
    # anchors the decode and every arm scores whatever its prior does. Reporting
    # one node is the minimum that keeps the instance about reconstruction.
    if not reported and infected:
        anchor = int(infected[int(rng.integers(len(infected)))])
        reported[anchor] = (
            int(activation_time[anchor]) if setting != partial_nodes else None
        )

    return (
        Observation(
            reported=reported,
            horizon=episode["horizon"],
            num_nodes=num_nodes,
            setting=setting,
            final_state=None,
            visible=visible,
        ),
        visible,
    )


def load_cascades(
    data_dir: str,
    diffusion_model: str,
    split: str,
    graph_id: str | None = None,
    setting: str = partial_times,
    observation_rate: float = default_observation_rate,
    hidden_rate: float = default_hidden_rate,
    limit: int = default_instances,
    seed: int = 0,
    require_parents: bool = True,
) -> list[CascadeInstance]:
    """
    Labelled `(G, O, Y)` episodes for one (dynamics, split), as CascadeInstances.

    `limit` subsamples deterministically in `seed` rather than taking a prefix, for
    the same reason `localization.load_instances` does: episodes are written
    selector-major, so a prefix would be all-`random` or all-`degree` and the
    decoder would be selected against one seeding process.

    `require_parents` RAISES on a dataset generated without `--trace-parents`
    rather than quietly falling back to Event F1. §2.6 is explicit that a
    tree-weighted reward is a precondition and not an enhancement: a search
    rewarded on the node half alone converges on decoders that never attempt the
    tree, and a run that silently did that would look like a result.
    """
    episodes = load_episode_trajectories(data_dir, diffusion_model, split)

    if graph_id is not None:
        episodes = [record for record in episodes if record["graph_id"] == graph_id]

    if not episodes:
        raise ValueError(
            f"no labelled episodes for graph {graph_id!r} in "
            f"{data_dir}/transitions_{diffusion_model}_{split}.jsonl. Cascade "
            f"reconstruction reads whole stored histories, so the data stage must "
            f"have run for this (dataset, dynamics, split)."
        )

    if require_parents and episodes[0]["parents"] is None:
        raise ValueError(
            f"{data_dir} carries no transmission edge, so PathPrecision cannot be "
            f"scored and the outer loop's reward would collapse onto Event F1: "
            f"which research/cascade_reconstruction.md §2.6 shows makes a program "
            f"search discard the tree half entirely. Regenerate with "
            f"`data/generate_wm_data.py --trace-parents` (the pipeline sets it "
            f"automatically for --task cascade_reconstruction), or pass "
            f"--cr-tree-weight 0 to acknowledge the node-only protocol explicitly."
        )

    if limit and len(episodes) > limit:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(episodes), size=limit, replace=False)
        episodes = [episodes[int(index)] for index in sorted(chosen)]

    instances = []

    for index, episode in enumerate(episodes):
        # Per-episode stream so the mask is stable in `seed` and does not shift
        # when `limit` changes, which would silently re-run a different experiment
        rng = np.random.default_rng([seed, index])
        observation, visible = mask_observation(
            episode, setting, observation_rate, hidden_rate, rng
        )

        true_times = {
            int(node): int(episode["activation_time"][node])
            for node in np.flatnonzero(episode["activation_time"] >= 0)
            if visible is None or visible[int(node)]
        }
        true_parents = episode["parents"]
        if true_parents is not None and visible is not None:
            true_parents = {
                node: [cause for cause in causes if visible[cause]]
                for node, causes in true_parents.items()
                if visible[node]
            }

        instances.append(
            CascadeInstance(
                episode_id=episode["episode_id"],
                graph_id=episode["graph_id"],
                num_nodes=episode["num_nodes"],
                horizon=episode["horizon"],
                sources=list(episode["sources"]),
                true_times=true_times,
                true_parents=true_parents,
                observation=observation,
                final_state=np.asarray(episode["final_state"], dtype=np.float32),
                marginal=np.asarray(episode["marginal"], dtype=np.float32),
                algorithm=episode.get("algorithm"),
                split=split,
            )
        )

    return instances


def bind_step_marginals(environment: object, task: TaskSpec) -> StepOracle | object:
    """The arm's transition kernel, or the raiser that stands in for it under @native."""
    if not task.forward_model:
        return unavailable_step_marginals

    return StepOracle(environment=environment)


def implemented(strategy: object, name: str) -> object | None:
    """
    The method `strategy` actually WROTE, or None if it only inherited the stub.

    Same identity check as `localization.implemented`, and here for the same
    reason: `Strategy` is a `typing.Protocol` whose method bodies are `...`, so
    subclassing it inherits a `reconstruct` that returns None and `hasattr` always
    says yes.
    """
    written = getattr(type(strategy), name, None)

    if written is None or written is getattr(Strategy, name, None):
        return None

    return getattr(strategy, name)


def validate_reconstruction(
    decoded: object, instance: CascadeInstance, graph: GraphInfo
) -> dict[int, tuple[int, int | None]]:
    """
    Raise StrategyError unless `decoded` is a coherent trajectory over this graph.

    Deliberately NOT `executor.validate_actions`: nothing here is an intervention
    and there is no budget to spend. The rules that ARE enforced are the ones
    without which a metric would silently measure something else:

      * a transmission has to traverse an ARC that exists, or `path_precision`
        would be scored against edges the graph does not have;
      * `parent = None` and `time = 0` mean the same thing (a source), so a node
        claimed at t=0 with a parent, or with a parent but no t>0, is incoherent
        rather than merely wrong;
      * under `hidden_nodes` a hidden node is not in the graph, so naming one is
        naming something that is not there.
    """
    if not isinstance(decoded, dict):
        raise StrategyError(
            f"reconstruct() must return a dict mapping node id -> (activation "
            f"timestep, inferred parent or None); got {type(decoded).__name__}. "
            f"Uninfected nodes are simply ABSENT from the dict."
        )

    result = {}

    for node, value in decoded.items():
        try:
            time, parent = value
        except (TypeError, ValueError) as error:
            raise StrategyError(
                f"reconstruct() mapped node {node!r} to {value!r}; every value must "
                f"be a 2-tuple (activation timestep, inferred parent or None)."
            ) from error

        node = int(node)
        time = int(time)

        if not 0 <= node < instance.num_nodes:
            raise StrategyError(
                f"reconstruct() returned node {node}, outside "
                f"[0, {instance.num_nodes})."
            )

        if not instance.observation.is_visible(node):
            raise StrategyError(
                f"reconstruct() named node {node}, which is HIDDEN in this setting: "
                f"it is absent from the graph, not merely unobserved. "
                f"`observation.visible` is the mask: filter your candidates by it."
            )

        if not 0 <= time <= instance.horizon:
            raise StrategyError(
                f"reconstruct() put node {node} at timestep {time}, outside "
                f"[0, {instance.horizon}]. The cascade ran for {instance.horizon} "
                f"steps; `horizon` is the argument you were handed."
            )

        if parent is None:
            if time != 0:
                raise StrategyError(
                    f"reconstruct() gave node {node} no parent but put it at "
                    f"timestep {time}. `parent = None` means SOURCE, and a source "
                    f"activates at t = 0: a node infected later was infected BY "
                    f"someone, so name them."
                )

            result[node] = (time, None)
            continue

        parent = int(parent)

        if time == 0:
            raise StrategyError(
                f"reconstruct() gave node {node} parent {parent} at timestep 0. "
                f"Nothing had activated before t = 0, so a node at t = 0 is a "
                f"source and its parent must be None."
            )

        if not 0 <= parent < instance.num_nodes:
            raise StrategyError(
                f"reconstruct() named parent {parent} for node {node}, outside "
                f"[0, {instance.num_nodes})."
            )

        if parent not in graph.in_neighbors(node):
            raise StrategyError(
                f"reconstruct() claims {parent} -> {node}, which is not an arc of "
                f"this graph. A transmission has to travel along an edge that "
                f"exists; `graph.in_neighbors({node})` is the set you may pick "
                f"from."
            )

        if not instance.observation.is_visible(parent):
            raise StrategyError(
                f"reconstruct() named parent {parent} for node {node}, but {parent} "
                f"is HIDDEN in this setting: it is absent from the graph, so it "
                f"cannot have transmitted. `observation.visible` is the mask."
            )

        result[node] = (time, parent)

    if not result:
        raise StrategyError(
            "reconstruct() returned an empty trajectory: no node was inferred to "
            "have been infected at all. At minimum the nodes the observation "
            "REPORTS were infected, so returning fewer than those is always wrong."
        )

    return result


metric_keys = (
    "node_precision",
    "node_recall",
    "node_f1",
    "event_precision",
    "event_recall",
    "event_f1",
    "mcc",
    "time_mae",
    "time_nrmse",
    "path_precision",
    "path_recall",
    "jaccard",
    "order_accuracy",
    "source_precision",
    "source_recall",
    "source_f1",
    "n_predicted",
    "n_true",
    "n_tree_edges",
    "n_true_edges",
)


def aggregate_metrics(per_instance: list[dict]) -> dict[str, float]:
    """Instance-averaged, with NaN tree columns skipped rather than propagated."""
    means = {}

    for key in metric_keys:
        values = [
            entry[key] for entry in per_instance if np.isfinite(entry.get(key, np.nan))
        ]
        means[key] = float(np.mean(values)) if values else float("nan")

    return means


# Reward ---------------------------------------------------------------------
#
# The reward never sees the true history. A decoded history is scored by how
# probable the arm's own transition kernel says it is (its log-likelihood, summed
# over the steps it implies and normalized per node) minus how much of the
# observation it contradicts (reported nodes it dropped, reported times it
# moved, nodes it named that a final snapshot says stayed clean). Both parts are
# computable at deployment. Path precision, event F1 and the tree-weighted score
# against the stored history are computed AFTER the search, on the winner only,
# by `reconstruction_label_metrics`, and stay the reported columns.

# One unit of observation violation (all constraints broken) costs as much as one
# nat per node of likelihood; a decode that argues with the data loses first
observation_penalty_weight = 1.0
max_listed_transitions = 8
# A susceptible node the kernel expects to activate with at least this
# probability, and the decode says did not, is listed as a silent candidate
silent_threshold = 0.5


def observation_violations(
    decoded: dict[int, tuple[int, int | None]], observation: Observation
) -> dict:
    """How much of what was OBSERVED the decode contradicts, label-free."""
    missing = [int(node) for node in observation.infected if node not in decoded]
    mistimed = [
        [int(node), int(time), int(decoded[node][0])]
        for node, time in observation.times.items()
        if node in decoded and decoded[node][0] != time
    ]
    outside = []
    if observation.final_state is not None:
        outside = [
            int(node)
            for node in decoded
            if float(observation.final_state[int(node)]) < 0.5
        ]

    n_constraints = len(observation.infected) + len(observation.times)
    if observation.final_state is not None:
        n_constraints += len(decoded)
    violations = len(missing) + len(mistimed) + len(outside)

    return {
        "consistency": 1.0 - violations / max(1, n_constraints),
        "n_missing": len(missing),
        "n_mistimed": len(mistimed),
        "n_outside": len(outside),
        "missing": missing[:max_listed_transitions],
        "mistimed": mistimed[:max_listed_transitions],
        "outside": outside[:max_listed_transitions],
    }


def history_steps(
    decoded: dict[int, tuple[int, int | None]], horizon: int
) -> list[tuple[int, list[int], list[int], list[int]]]:
    """
    The state sequence a decoded history implies: `(t, infected_t, frontier_t,
    frontier_{t+1})` for every transition it asserts, plus one terminal
    transition asserting nothing more activated, when the horizon allows it.
    """
    by_time = {}
    for node, (when, _) in decoded.items():
        by_time.setdefault(int(when), []).append(int(node))

    last = max(by_time)
    end = min(int(horizon), last + 1)
    steps = []
    infected = []
    for when in range(end):
        infected = sorted(infected + by_time.get(when, []))
        steps.append(
            (
                when,
                list(infected),
                sorted(by_time.get(when, [])),
                sorted(by_time.get(when + 1, [])),
            )
        )

    return steps


def score_history(
    decoded: dict[int, tuple[int, int | None]],
    observation: Observation,
    horizon: int,
    kernel: object,
    num_nodes: int,
) -> dict:
    """
    One decoded history's reward and the diagnostics behind it.

    `reward = (log p(sources) + log p(transitions | kernel)) / N - w * (1 -
    consistency)`. The transition term is the kernel's log-probability of every
    activation the history asserts. The source term is the generative model's own
    prior: each node is an exogenous source with probability `1/N`, independently,
    so declaring a node a source costs about `log N` nats. Without it, "every
    reported node is a source at t=0" explains any observation without asserting
    a single transmission and outscores every real decoder (measured on the first
    smoke run). Minus the fraction of the observation the decode contradicts.
    `weakest` lists the asserted activations the kernel found least probable (a
    parent not yet active, an arc that rarely transmits), and `silent` the
    susceptible nodes the kernel expected to activate that the decode left out:
    the two places a decoder edits next.
    """
    steps = history_steps(decoded, horizon)
    parent_of = {int(node): parent for node, (_, parent) in decoded.items()}
    n_sources = sum(1 for parent in parent_of.values() if parent is None)
    source_prior = 1.0 / max(2, num_nodes)
    loglik = n_sources * np.log(source_prior) + (num_nodes - n_sources) * np.log1p(
        -source_prior
    )
    weakest = []
    silent = []

    for when, infected, frontier, next_frontier in steps:
        marginal = np.asarray(kernel(infected, frontier), dtype=np.float64)
        loglik += transition_logprob(marginal, infected, next_frontier)

        for node in next_frontier:
            weakest.append(
                [round(float(marginal[node]), 4), int(node), parent_of[node], when + 1]
            )

        active = np.zeros(num_nodes, dtype=bool)
        active[infected] = True
        active[next_frontier] = True
        candidates = np.flatnonzero(~active)
        expected = candidates[marginal[candidates] >= silent_threshold]
        for node in expected[np.argsort(-marginal[expected])][:3]:
            silent.append([round(float(marginal[node]), 4), int(node), when + 1])

    weakest.sort(key=lambda item: item[0])
    silent.sort(key=lambda item: -item[0])
    violations = observation_violations(decoded, observation)
    per_node = loglik / max(1, num_nodes)

    return {
        "reward": per_node
        - observation_penalty_weight * (1.0 - violations["consistency"]),
        "loglik": float(loglik),
        "loglik_per_node": float(per_node),
        "n_steps": len(steps),
        "n_predicted": len(decoded),
        "n_sources": n_sources,
        "weakest": weakest[:max_listed_transitions],
        "silent": silent[:max_listed_transitions],
        **violations,
    }


reward_keys = (
    "reward",
    "loglik_per_node",
    "consistency",
    "n_missing",
    "n_mistimed",
    "n_outside",
    "n_predicted",
    "n_sources",
    "n_steps",
)


def _mean_over(per_instance: list[dict], keys: tuple) -> dict[str, float]:
    return {
        key: float(np.mean([entry[key] for entry in per_instance]))
        for key in keys
        if per_instance and all(key in entry for entry in per_instance)
    }


def _decoded_tuples(decoded: dict) -> dict[int, tuple[int, int | None]]:
    """The JSON form `{node: [t, parent]}` back to the contract's tuples."""
    return {
        int(node): (int(value[0]), None if value[1] is None else int(value[1]))
        for node, value in decoded.items()
    }

def evaluate_reconstructor(
    strategy: object,
    environment: object,
    task: TaskSpec,
    graph: GraphInfo,
    instances: list[CascadeInstance],
    tree_weight: float = default_tree_weight,
) -> tuple[Trajectory, float]:
    """
    Score one trajectory decoder over the selection episodes, label-free.

    Returns the same `(Trajectory, plan_seconds)` pair `evaluate_strategy` does, so
    every method (one_shot, evolve, the checkpointing, the population feedback)
    consumes it unchanged. `reward` is the mean of `score_history`: the arm's own
    kernel's log-likelihood of the decoded history per node, minus the observation
    it contradicts. It maximizes and uses nothing a deployed decoder would not
    have. The stored history is read only by `reconstruction_label_metrics`,
    after the search, on the winner; `tree_weight` is carried for that report.
    """
    start = time.perf_counter()
    reconstruct = implemented(strategy, "reconstruct")

    if reconstruct is None:
        raise StrategyError(
            "this task needs a reconstruct() method and your class does not define "
            "one. The contract is\n"
            "    def reconstruct(self, graph, observation, horizon) "
            "-> dict[int, tuple[int, int | None]]\n"
            "mapping each node you believe was infected to (the timestep it "
            "activated, the node that infected it). `parent = None` marks a "
            "source; nodes you believe were never infected are simply ABSENT. "
            "plan_horizon(), act() and localize() are other tasks' contracts and "
            "are not called here."
        )

    # What the PROGRAM may call: the four bindings, a raiser under @native
    # Bound to canned baselines only: a generated decoder is offline
    oracle = None
    if getattr(strategy, "canned", False):
        oracle = bind_step_marginals(environment, task)
        strategy.step_marginals = oracle
    strategy.transition_logprob = transition_logprob
    # What the HARNESS scores with: the same kernel, counted apart from the
    # program's own calls, available under every condition (the native arm's is
    # one real episode per step, as an intervention task's native arm is scored)
    scorer = StepOracle(environment=environment)

    per_instance = []
    for instance in instances:
        decoded = call_strategy(
            reconstruct, graph, instance.observation, instance.horizon
        )
        decoded = validate_reconstruction(decoded, instance, graph)
        entry = score_history(
            decoded, instance.observation, instance.horizon, scorer, instance.num_nodes
        )
        entry["episode_id"] = instance.episode_id
        entry["observed"] = instance.observed_count
        entry["infected_count"] = instance.infected_count
        entry["decoded"] = {
            int(node): [int(time), None if parent is None else int(parent)]
            for node, (time, parent) in decoded.items()
        }
        per_instance.append(entry)

    if not per_instance:
        raise StrategyError("no episodes to score this decoder against")

    means = _mean_over(per_instance, reward_keys)
    rewards = [entry["reward"] for entry in per_instance]
    elapsed = time.perf_counter() - start

    # A representative recovered SOURCE set, so the results JSON's timeline and the
    # referee have concrete actions to replay: the seed commit that WOULD
    # reproduce the observation if the decode got its roots right
    representative = per_instance[0]
    recovered_sources = sorted(
        int(node)
        for node, (_, parent) in representative["decoded"].items()
        if parent is None
    )

    trajectory = Trajectory(
        states=[State([], []), State(recovered_sources, [])],
        actions=[[ActionOp("add_node", node) for node in recovered_sources]],
        reward=means["reward"],
        infected_counts=rewards,
        cost={
            "env": "cascade_reconstruction",
            "reward_se": (
                float(np.std(rewards, ddof=1) / np.sqrt(len(rewards)))
                if len(rewards) > 1
                else 0.0
            ),
            "rollout_seconds": elapsed,
            "n_instances": len(per_instance),
            "setting": instances[0].observation.setting,
            # The weight of the REPORTED tree score, computed after the search
            "tree_weight": tree_weight,
            "metrics": means,
            "per_instance": per_instance,
            # The cost axis §2.4.2 exists to measure, and the number §11 says
            # nobody has published: kernel evaluations the PROGRAM made per
            # decoded instance. The harness's own scoring steps are separate.
            "kernel_calls": getattr(oracle, "calls", 0),
            "kernel_calls_per_instance": round(
                getattr(oracle, "calls", 0) / len(per_instance), 3
            ),
            "scoring_kernel_calls": scorer.calls,
        },
        # No cascade was rolled out; nothing was seeded and nothing spread
        final_marginals=None,
        spread_curve=None,
        # One reward per selection cascade, instance-aligned across decoders
        sample_rewards=rewards,
    )

    return trajectory, elapsed


def reconstruction_label_metrics(
    trajectory: Trajectory, instances: list[CascadeInstance], tree_weight: float
) -> dict[str, float]:
    """
    Path precision, event F1, timing error and the tree-weighted score against the
    STORED history, after the fact.

    Called once per split on the winning decoder only, after the search has
    finished and the write-up has been requested, so no label reaches a prompt or
    a selection decision. Merges the label metrics into the trajectory's
    `metrics` and `per_instance` blocks in place; `tree_score` is the
    `lambda * PathPrecision + (1 - lambda) * EventF1` the literature is read in.
    """
    by_episode = {instance.episode_id: instance for instance in instances}
    labelled = []

    for entry in trajectory.cost.get("per_instance", []):
        instance = by_episode.get(entry["episode_id"])
        if instance is None:
            continue

        metrics = reconstruction_metrics(
            _decoded_tuples(entry["decoded"]),
            instance.true_times,
            instance.true_parents,
            instance.num_nodes,
            instance.horizon,
        )
        metrics["tree_score"] = reconstruction_reward(metrics, tree_weight)
        # The tree the decoder is scored against: one edge per non-source node
        # whose cause survived the mask. What `n_tree_edges` is read against.
        metrics["n_true_edges"] = float(
            sum(1 for causes in (instance.true_parents or {}).values() if causes)
        )
        entry.update(metrics)
        labelled.append(metrics)

    means = aggregate_metrics(labelled) if labelled else {}
    if labelled:
        means["tree_score"] = float(np.mean([entry["tree_score"] for entry in labelled]))
    trajectory.cost["metrics"] = {**trajectory.cost.get("metrics", {}), **means}

    return means


def trivial_decoder(instance: CascadeInstance, graph: GraphInfo) -> dict:
    """Everyone reachable from the reports, parents by BFS: the laziest decoder that type-checks."""
    observation = instance.observation
    seen = {node: (0, None) for node in observation.infected}
    queue = list(seen)
    step = 0

    while queue and step < instance.horizon:
        step += 1
        wave = []
        for node in queue:
            for neighbour in graph.out_neighbors(node):
                if neighbour in seen or not observation.is_visible(neighbour):
                    continue
                seen[neighbour] = (step, node)
                wave.append(neighbour)
        queue = wave

    return seen


def trivial_decoder_reward(
    instances: list[CascadeInstance],
    graph: GraphInfo,
    tree_weight: float,
    environment: object | None = None,
) -> dict[str, float]:
    """
    What "predict everyone reachable, assign parents by BFS" scores.

    §2.11 risk 1 makes this a REQUIRED check rather than a diagnostic: a trivial
    decoder has to score badly under the reward, or the reward is wrong. Two
    numbers, because two rewards are in play: `trivial_decoder_reward` is the
    label-free likelihood reward the search runs on (needs the arm's evaluator),
    `trivial_decoder_tree_score` the reported tree-weighted score against the
    stored history. Written into the results JSON so the number a reader needs to
    interpret a program-search result is in the same file as the result.
    """
    tree_scores = []
    rewards = []
    kernel = StepOracle(environment=environment) if environment is not None else None

    for instance in instances:
        decoded = trivial_decoder(instance, graph)
        metrics = reconstruction_metrics(
            decoded,
            instance.true_times,
            instance.true_parents,
            instance.num_nodes,
            instance.horizon,
        )
        tree_scores.append(reconstruction_reward(metrics, tree_weight))
        if kernel is not None:
            rewards.append(
                score_history(
                    decoded, instance.observation, instance.horizon, kernel,
                    instance.num_nodes,
                )["reward"]
            )

    return {
        "trivial_decoder_tree_score": (
            float(np.mean(tree_scores)) if tree_scores else float("nan")
        ),
        "trivial_decoder_reward": float(np.mean(rewards)) if rewards else None,
        "tree_weight": tree_weight,
    }


def referee_likelihood(
    environment: object,
    task: TaskSpec,
    instances: list[CascadeInstance],
    per_instance: list[dict],
) -> dict[str, float]:
    """
    The referee: the reward re-measured under the referee's own kernel (the exact
    oracle by default, NDlib for the agreement check).

    Every stored decode is re-scored by `score_history` with NDlib's own
    step marginals in place of the arm's evaluator, so `referee_reward` is to
    `reward` exactly what the referee replay is to a world-model spread.
    """
    by_episode = {entry["episode_id"]: entry for entry in per_instance}
    kernel = StepOracle(environment=environment)
    rewards = []

    for instance in instances:
        entry = by_episode.get(instance.episode_id)
        if entry is None:
            continue

        rewards.append(
            score_history(
                _decoded_tuples(entry["decoded"]),
                instance.observation,
                instance.horizon,
                kernel,
                instance.num_nodes,
            )["reward"]
        )

    if not rewards:
        return {}

    return {
        "referee_reward": float(np.mean(rewards)),
        "referee_reward_se": (
            float(np.std(rewards, ddof=1) / np.sqrt(len(rewards)))
            if len(rewards) > 1
            else 0.0
        ),
        "referee_kernel_calls": kernel.calls,
    }

def referee_resimulation_error(
    environment: object,
    task: TaskSpec,
    instances: list[CascadeInstance],
    per_instance: list[dict],
) -> dict[str, float]:
    """
    Re-simulate each decode's RECOVERED sources and compare against what happened.

    The referee's secondary column for this task. The reward is already exact (it is
    scored against a history we stored), so the referee measures the other thing:
    whether the recovered roots actually reproduce the observed cascade. Reported
    beside the TRUE source set's own error, because on an ill-posed problem a
    recovered set can reproduce `y` better than the truth did and the number is
    unreadable without knowing that.
    """
    from functools import partial

    from coding_agent.credit import planned_action

    by_episode = {entry["episode_id"]: entry for entry in per_instance}
    recovered_errors = []
    oracle_errors = []

    def simulate(nodes: list[int]) -> np.ndarray:
        plan = [[ActionOp("add_node", int(node)) for node in nodes]] + [
            [] for _ in range(task.horizon)
        ]
        trajectory = environment.rollout(
            partial(planned_action, plan), task.horizon, task.budget
        )

        return np.asarray(trajectory.final_marginals, dtype=np.float64)

    for instance in instances:
        entry = by_episode.get(instance.episode_id)
        if entry is None:
            continue

        recovered = [
            int(node)
            for node, value in entry["decoded"].items()
            if value[1] is None
        ]
        if not recovered:
            continue

        observed = np.asarray(instance.final_state, dtype=np.float64)
        recovered_errors.append(resimulation_error(simulate(recovered), observed))
        oracle_errors.append(resimulation_error(simulate(instance.sources), observed))

    if not recovered_errors:
        return {}

    return {
        "resim_error": float(np.mean(recovered_errors)),
        "resim_error_true_sources": float(np.mean(oracle_errors)),
    }


max_listed_nodes = 20


def summarize_reconstruction(
    trajectory: Trajectory, graph: GraphInfo, task: TaskSpec
) -> str:
    """
    Where the decoded histories are implausible or contradict the data, label-free.

    Neither `methods.base.summarize` (which describes where a cascade should go
    next) nor `localization.summarize_localization` (which describes a recovered
    set) answers the question here. This reads the reward's own parts back: the
    asserted transmissions the kernel found least probable, the activations it
    expected and the decode left out, and the reports the decode contradicted.
    """
    cost = trajectory.cost
    means = cost.get("metrics", {})
    per_instance = cost.get("per_instance", [])

    lines = [
        f"REWARD={trajectory.reward:.4f} (±{cost.get('reward_se', 0.0):.4f} SE, "
        f"HIGHER IS BETTER) over {cost.get('n_instances', 0)} cascades "
        f"({cost.get('setting', '?')} observation) = log-likelihood of your decoded "
        f"history per node ({means.get('loglik_per_node', 0.0):.4f}: a 1/N prior "
        f"per source you declare, plus the kernel's probability of every "
        f"transmission you assert) minus {observation_penalty_weight:g} x the "
        f"fraction of the observation you contradicted (consistency "
        f"{means.get('consistency', 0.0):.3f})",
        f"per cascade: {means.get('n_predicted', 0.0):.1f} nodes decoded "
        f"({means.get('n_sources', 0.0):.1f} as sources), "
        f"{means.get('n_missing', 0.0):.1f} reported nodes DROPPED, "
        f"{means.get('n_mistimed', 0.0):.1f} reported times MOVED, "
        f"{means.get('n_outside', 0.0):.1f} nodes named that the snapshot says stayed clean",
        f"harness scoring kernel steps: {cost.get('scoring_kernel_calls', 0)} (your "
        f"program is offline and makes none)",
    ]

    if means.get("n_missing", 0.0) > 0 or means.get("n_mistimed", 0.0) > 0:
        lines.append(
            "DIAGNOSIS: you are arguing with the data. Every reported node must be in "
            "your history at its reported time; dropping or moving one costs more "
            "than any likelihood gain can pay back."
        )

    if not per_instance:
        return "\n".join(lines)

    ranked = sorted(per_instance, key=lambda entry: entry["reward"])
    worst, best = ranked[0], ranked[-1]

    for label, entry in (("WORST", worst), ("BEST", best)):
        lines.append(
            f"{label} cascade (reward={entry['reward']:.4f}, log-lik/node "
            f"{entry['loglik_per_node']:.4f}, consistency {entry['consistency']:.3f}): "
            f"{entry['observed']} nodes reported, you decoded {entry['n_predicted']} "
            f"({entry['n_sources']} sources) over {entry['n_steps']} steps"
        )
        lines.append(
            "  least probable transmissions you asserted (p, child, parent, t): "
            + (
                ", ".join(
                    f"{node}<-{parent}@t{time} (p={probability:.3f})"
                    for probability, node, parent, time in entry["weakest"]
                )
                or "none"
            )
        )
        lines.append(
            "  activations the kernel expected that you left out (p, node, t): "
            + (
                ", ".join(
                    f"{node}@t{time} (p={probability:.3f})"
                    for probability, node, time in entry["silent"]
                )
                or "none"
            )
        )
        if entry["missing"] or entry["mistimed"] or entry["outside"]:
            lines.append(
                f"  contradicted: dropped reports {entry['missing'] or 'none'}; "
                f"moved times (node, reported, yours) {entry['mistimed'] or 'none'}; "
                f"named outside the snapshot {entry['outside'] or 'none'}"
            )

    if worst["weakest"] and worst["weakest"][0][0] < 0.05:
        lines.append(
            "DIAGNOSIS: some transmissions you asserted are near-impossible under the "
            "kernel: the parent was not active at t-1, or the arc almost never "
            "transmits. Re-time those nodes or pick the in-neighbour that was."
        )

    if means.get("n_predicted", 0.0) <= means.get("n_missing", 0.0) + (
        per_instance[0]["observed"] if per_instance else 0
    ):
        lines.append(
            "DIAGNOSIS: you inferred nothing beyond the reports. The silent "
            "candidates above are where the kernel expects hidden activations; a "
            "history that reaches the later reports through them is more probable "
            "than one that jumps."
        )

    return "\n".join(lines)


class ReconstructAnchor:
    """
    Wraps a library decoder as the `reconstruct()`-shaped object the sweep wants.

    Anchors go through `evaluate_reconstructor` rather than being scored
    separately, so they meet the arm they are setting a bar for under IDENTICAL
    conditions: the same episodes, the same mask, the same setting.
    """

    def __init__(self, name: str, decoder, task: TaskSpec) -> None:
        self.name = name
        self.decoder = decoder
        self.task = task
        self.source_script = ""

    def reconstruct(
        self, graph: GraphInfo, observation: Observation, horizon: int
    ) -> dict:
        return self.decoder(
            graph,
            observation,
            horizon,
            diffusion_model=self.task.diffusion_model,
            predict=getattr(self, "step_marginals", None),
        )
