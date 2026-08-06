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
without the `--compare` referee. The referee still runs and measures something
else: re-simulating the RECOVERED sources against what was observed (§8.5.5's
analogue), which is the column no paper in this literature reports.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from coding_agent.executor import StrategyError, call_strategy
from coding_agent.types import ActionOp, GraphInfo, State, Strategy, TaskSpec, Trajectory
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
    episode: dict, setting: str, observation_rate: float, hidden_rate: float, rng
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


def evaluate_reconstructor(
    strategy: object,
    environment: object,
    task: TaskSpec,
    graph: GraphInfo,
    instances: list[CascadeInstance],
    tree_weight: float = default_tree_weight,
) -> tuple[Trajectory, float]:
    """
    Score one trajectory decoder over the labelled episodes.

    Returns the same `(Trajectory, plan_seconds)` pair `evaluate_strategy` does, so
    every method (one_shot, evolve, the checkpointing, the population feedback)
    consumes it unchanged. `reward` is §2.6's Score, which maximizes.
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

    oracle = bind_step_marginals(environment, task)
    strategy.step_marginals = oracle
    strategy.transition_logprob = transition_logprob

    per_instance = []

    for instance in instances:
        decoded = call_strategy(
            reconstruct, graph, instance.observation, instance.horizon
        )
        decoded = validate_reconstruction(decoded, instance, graph)

        metrics = reconstruction_metrics(
            decoded,
            instance.true_times,
            instance.true_parents,
            instance.num_nodes,
            instance.horizon,
        )
        metrics["reward"] = reconstruction_reward(metrics, tree_weight)
        metrics["episode_id"] = instance.episode_id
        metrics["observed"] = instance.observed_count
        metrics["infected_count"] = instance.infected_count
        metrics["decoded"] = {
            int(node): [int(time), None if parent is None else int(parent)]
            for node, (time, parent) in decoded.items()
        }
        per_instance.append(metrics)

    if not per_instance:
        raise StrategyError("no labelled episodes to score this decoder against")

    means = aggregate_metrics(per_instance)
    rewards = [entry["reward"] for entry in per_instance]
    # The aggregate score belongs IN the metrics block, not only on the trajectory:
    # `selection_metrics` is what the report and the generalization figure read
    # back, and without it a selection score would have to be reconstructed from
    # the gap
    means["reward"] = float(np.mean(rewards))
    elapsed = time.perf_counter() - start

    # A representative recovered SOURCE set, so the results JSON's timeline and the
    # --compare referee have concrete actions to replay: the seed commit that WOULD
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
        reward=float(np.mean(rewards)),
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
            # §8.3: the reward function is a design decision with a documented
            # failure mode, so lambda belongs in the results rather than a footnote
            "tree_weight": tree_weight,
            "metrics": means,
            "per_instance": per_instance,
            # The cost axis §2.4.2 exists to measure, and the number §11 says
            # nobody has published: kernel evaluations per decoded instance
            "kernel_calls": getattr(oracle, "calls", 0),
            "kernel_calls_per_instance": round(
                getattr(oracle, "calls", 0) / len(per_instance), 3
            ),
        },
        # No cascade was rolled out; nothing was seeded and nothing spread
        final_marginals=None,
        spread_curve=None,
    )

    return trajectory, elapsed


def trivial_decoder_reward(
    instances: list[CascadeInstance], graph: GraphInfo, tree_weight: float
) -> dict[str, float]:
    """
    What "predict everyone reachable, assign parents by BFS" scores under this reward.

    §2.11 risk 1 makes this a REQUIRED check rather than a diagnostic: "confirm
    before running the full search that a trivial decoder scores badly under the
    chosen Score. If it does not, the reward is wrong." Reported into the results
    JSON so the number a reader needs to interpret a program-search result is in
    the same file as the result.
    """
    scores = []

    for instance in instances:
        observation = instance.observation
        seen = {node: (0, None) for node in observation.infected}
        queue = list(seen)
        step = 0

        # BFS out of the reports, giving every node the hop it was reached at and
        # the neighbour that reached it: the laziest decoder that type-checks
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

        metrics = reconstruction_metrics(
            seen,
            instance.true_times,
            instance.true_parents,
            instance.num_nodes,
            instance.horizon,
        )
        scores.append(reconstruction_reward(metrics, tree_weight))

    return {
        "trivial_decoder_reward": float(np.mean(scores)) if scores else float("nan"),
        "tree_weight": tree_weight,
    }


def referee_resimulation_error(
    environment: object,
    task: TaskSpec,
    instances: list[CascadeInstance],
    per_instance: list[dict],
) -> dict[str, float]:
    """
    Re-simulate each decode's RECOVERED sources and compare against what happened.

    The `--compare` referee for this task. The reward is already exact (it is
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
    What the decoded trajectories got right and wrong, split by HALF.

    Neither `methods.base.summarize` (which describes where a cascade should go
    next) nor `localization.summarize_localization` (which describes a recovered
    set) answers the question here. The split that matters is §2.6's: the node set
    against the tree, printed side by side, so the model can see that its Event F1
    is high while its Path Precision is not, which is the exact failure the
    reward is weighted to prevent it from settling into.
    """
    cost = trajectory.cost
    means = cost.get("metrics", {})
    per_instance = cost.get("per_instance", [])
    tree_weight = cost.get("tree_weight", default_tree_weight)
    path_precision = means.get("path_precision", float("nan"))

    lines = [
        f"SCORE={trajectory.reward:.4f} (±{cost.get('reward_se', 0.0):.4f} SE, "
        f"HIGHER IS BETTER) = {tree_weight:.2f} * PathPrecision + "
        f"{1.0 - tree_weight:.2f} * EventF1, over "
        f"{cost.get('n_instances', 0)} labelled cascades "
        f"({cost.get('setting', '?')} observation)",
        f"THE HARD HALF: tree: path_precision={path_precision:.4f}  "
        f"jaccard={means.get('jaccard', float('nan')):.4f}  "
        f"order_accuracy={means.get('order_accuracy', float('nan')):.4f}  "
        f"({means.get('n_tree_edges', 0.0):.0f} edges named per cascade)",
        f"THE EASY HALF: events: event_f1={means.get('event_f1', 0.0):.4f} "
        f"(PR {means.get('event_precision', 0.0):.4f} / "
        f"RE {means.get('event_recall', 0.0):.4f})  "
        f"node_f1={means.get('node_f1', 0.0):.4f}  "
        f"mcc={means.get('mcc', 0.0):.4f}",
        f"timing: MAE={means.get('time_mae', 0.0):.3f} steps, "
        f"NRMSE={means.get('time_nrmse', 0.0):.4f} (lower is better; reported, "
        f"NOT part of the score)",
        f"sources recovered as a side effect: F1={means.get('source_f1', 0.0):.4f} "
        f"(the nodes you gave no parent)",
        f"kernel calls: {cost.get('kernel_calls', 0)} total, "
        f"{cost.get('kernel_calls_per_instance', 0)} per cascade",
    ]

    if np.isfinite(path_precision) and means.get("event_f1", 0.0) - path_precision > 0.2:
        lines.append(
            "DIAGNOSIS: your event F1 is far above your path precision, which is "
            "the known failure mode of this task: you are recovering WHICH nodes "
            "were infected and WHEN, and guessing who infected them. The score is "
            "weighted toward the tree precisely because the node half is nearly "
            "free; spend your next edit on the parent assignment."
        )

    # The second reward hazard, and one the literature does not name because no
    # published method has a search to game: PathPrecision is a PRECISION, so a
    # decoder that names three edges and gets them right scores 1.0 on the half the
    # reward is weighted toward. `path_recall` and `jaccard` are reported for
    # exactly this and the diagnostic says so out loud, because a search will find
    # the under-prediction corner long before a human notices it.
    named = means.get("n_tree_edges", 0.0)
    truth = max(means.get("n_true", 0.0) - means.get("source_recall", 0.0), 1.0)
    if np.isfinite(path_precision) and named < 0.5 * truth:
        lines.append(
            f"DIAGNOSIS: you named only {named:.0f} transmission edges against "
            f"~{truth:.0f} real ones, so your path precision "
            f"({path_precision:.4f}) is measured on a small fraction of the tree "
            f"and your path RECALL is {means.get('path_recall', 0.0):.4f}. "
            f"Precision on three lucky edges is not a reconstruction: predicting "
            f"fewer nodes is the cheapest way to make this number look good and it "
            f"is the corner to avoid, not to find. Jaccard "
            f"({means.get('jaccard', 0.0):.4f}) is the honest summary of the tree."
        )

    if not per_instance:
        return "\n".join(lines)

    ranked = sorted(per_instance, key=lambda entry: entry["reward"])
    worst, best = ranked[0], ranked[-1]

    for label, entry in (("WORST", worst), ("BEST", best)):
        lines.append(
            f"{label} cascade (score={entry['reward']:.3f}): "
            f"{entry['observed']} nodes reported of {entry['infected_count']} "
            f"actually infected ({graph.num_nodes} in the graph); you named "
            f"{entry['n_predicted']:.0f}, "
            f"path_precision={entry.get('path_precision', float('nan')):.3f}, "
            f"event_f1={entry['event_f1']:.3f}"
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
