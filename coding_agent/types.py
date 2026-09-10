"""
Core contracts for the coding-agent outer loop.

Reuses the canonical ActionOp/State value types from the data simulator so the
strategies, environments, and the trained world model all speak the same action vocabulary.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol
import numpy as np

from data.wm_simulator import ActionOp, State, spent, valid_action_ops
from pipeline.tasks import forecast, maximize, minimize, recover

# ActionFn is the interface between strategies and environments (every environment's rollout() consumes one of these; every method produces one):
# ActionFn is a function mapping (current state, timestep) -> action bag for that timestep
ActionFn = Callable[[State, int], list[ActionOp]]

# Which realized activations an adaptive policy observes between rounds
full_adoption = "full_adoption"
myopic = "myopic"
valid_feedback_models = (full_adoption, myopic)


def improves(candidate: float, incumbent: float, sense: str, epsilon: float = 0.0) -> bool:
    """
    Whether `candidate` beats `incumbent` under the task's objective sense.

    Every "is this better" in the outer loop routes through here, so flipping one
    registry field flips the whole search rather than N scattered comparisons,
    and a place that forgets to ask is a place that silently maximizes.
    """
    if sense == minimize:
        return candidate < incumbent - epsilon

    return candidate > incumbent + epsilon


def best_by(items: Iterable, key: Callable, sense: str):
    """argmax or argmin over `items`, whichever the sense asks for."""
    return (min if sense == minimize else max)(items, key=key)


def rank_by(items: Iterable, key: Callable, sense: str) -> list:
    """`items` best-first under the sense."""
    return sorted(items, key=key, reverse=sense != minimize)


def pad_counts(counts: list[float], horizon: int) -> list[float]:
    """
    Extend a per-timestep count vector to horizon + 2 by holding its last value.

    A rollout breaks early only when the frontier AND the action bag are both
    empty, which is a fixed point of monotone IC/LT: no frontier means no new
    infections and no bag means no injections, so every later step would report
    the same count. Holding it is therefore exact, and it is what makes
    sigma(S, T) readable at any T <= horizon instead of only at termination.

    Index t is the count AFTER the step at timestep t - 1, so index 0 is the
    empty initial state and a full run has horizon + 2 entries.
    """
    if not counts:
        return [0.0] * (horizon + 2)

    return counts + [counts[-1]] * (horizon + 2 - len(counts))


@dataclass()
class GraphInfo:
    """Read-only graph view handed to the library and to strategies."""

    num_nodes: int
    edge_index: np.ndarray  # (2, E) int64 [src, dst]
    ic_probs: np.ndarray  # (E,) float32 per-edge IC transmission prob
    directed: bool
    _out_adjacency: dict[int, list[int]] | None = field(default=None, repr=False)
    _in_adjacency: dict[int, list[int]] | None = field(default=None, repr=False)
    _degrees: np.ndarray | None = field(default=None, repr=False)
    # Prompt profile string, cached by tools.graph_profile.build_graph_profile
    _profile: str | None = field(default=None, repr=False)
    # Feedback caches, filled by methods.base: {node: community_id} and the
    # (RR-set cover index, theta) pair behind the residual-gain hints. Both are
    # graph-level and would otherwise be recomputed on every refinement turn.
    _community_labels: dict[int, int] | None = field(default=None, repr=False)
    _rr_covers: tuple | None = field(default=None, repr=False)

    @classmethod
    def from_store_entry(cls, entry: dict) -> "GraphInfo":
        """Build GraphInfo from a world_model.wm_data.load_graph_store entry."""
        return cls(
            num_nodes=int(entry["num_nodes"]),
            edge_index=np.asarray(entry["edge_index"], dtype=np.int64),
            ic_probs=np.asarray(entry["ic_probs"], dtype=np.float32),
            directed=bool(entry.get("meta", {}).get("directed", False)),
        )

    def _ensure_adjacency(self) -> None:
        if self._out_adjacency is not None:
            return

        # Build the in and out adjacency lists
        out_adjacency = {node: [] for node in range(self.num_nodes)}
        in_adjacency = {node: [] for node in range(self.num_nodes)}

        for edge in range(self.edge_index.shape[1]):
            source = int(self.edge_index[0, edge])
            target = int(self.edge_index[1, edge])

            out_adjacency[source].append(target)
            in_adjacency[target].append(source)

        self._out_adjacency, self._in_adjacency = out_adjacency, in_adjacency

    def out_neighbors(self, node: int) -> list[int]:
        self._ensure_adjacency()
        assert self._out_adjacency is not None
        return self._out_adjacency[int(node)]

    def in_neighbors(self, node: int) -> list[int]:
        self._ensure_adjacency()
        assert self._in_adjacency is not None
        return self._in_adjacency[int(node)]

    def degree(self, node: int) -> int:
        """Total degree (in + out) of the node."""
        if self._degrees is None:
            degrees = np.zeros(self.num_nodes, dtype=np.int64)
            np.add.at(degrees, self.edge_index[0], 1)
            np.add.at(degrees, self.edge_index[1], 1)
            self._degrees = degrees

        return int(self._degrees[int(node)])


@dataclass()
class TaskSpec:
    """Experiment problem statement and task specification details"""

    task: str = "influence_maximization"
    objective: str = "maximize_final_spread"
    diffusion_model: str = "IC"  # "IC" or "LT"
    budget: int = 5
    horizon: int = 10
    allowed_ops: tuple = valid_action_ops  # ops the strategy may emit
    # maximize | minimize, from the task registry. Read by improves()/best_by()
    # everywhere the search picks a winner.
    sense: str = maximize
    # Which op one unit of budget buys: `add_node` seeding, `remove_node`
    # containment. The executor counts it and the prompt states it.
    budget_op: str = "add_node"
    # Containment tasks (critical node detection): the cascade is started by an
    # EXOGENOUS outbreak the planner does not control and cannot spend budget on,
    # injected at t=0 by coding_agent.containment. Empty for every seeding task,
    # where the planner's own add_node ops are the outbreak.
    outbreak: tuple = ()
    # What remove_node does; the system prompt states the matching rule, and
    # stating the wrong one has the agent plan against dynamics it will not get
    remove_semantics: str = spent
    # Adaptive IM: seeds are committed in `rounds` batches summing to `budget`,
    # each chosen AFTER observing the diffusion the previous batch produced.
    # None = non-adaptive, one plan decided up front (static IM). The two differ
    # only here, which is what makes the adaptivity gap a clean A/B.
    rounds: int | None = None
    # Han et al. run two sweeps: fix r and vary k, or fix b and vary k. Setting
    # this switches to the second; r is then derived as ceil(k / b).
    per_round_budget: int | None = None
    # Timesteps of diffusion between consecutive rounds. 1 = seed again on the
    # very next step; larger lets each batch's cascade run further first.
    round_gap: int = 1
    # What the policy may READ at a round boundary (research/adaptive_online_im.md
    # §1.1). full_adoption = the whole realized state; myopic = the current wave
    # only. The theory literature's central axis, and free from our channel layout.
    feedback_model: str = full_adoption
    # Dynamic / streaming IM (§1.4): exogenous edge edits per timestep, as a
    # fraction of |E|. 0 = the static graph every other setting assumes.
    edit_rate: float = 0.0
    # Multi-round IM (§1.5): r SEPARATE campaigns of k seeds each, scored on the
    # union of what they activate. 1 = a single campaign, i.e. every other task.
    campaigns: int = 1
    # Source localization: the program INVERTS the transition instead of choosing
    # an intervention, so it writes localize() rather than plan_horizon(), it is
    # scored on F1 against the true source set, and no action is ever emitted.
    # From the task registry's `objective == recover`.
    objective_kind: str = maximize
    # The labelled (G, y, x) episodes one reward evaluation sweeps over, as
    # coding_agent.localization.SourceInstance. Carried on the task for the same
    # reason `outbreak` is: it is instance data every arm must face identically,
    # and every call site that has a task already has it. Empty for every task
    # that does not invert.
    instances: tuple = ()
    # Where each instance's k comes from: `episode` (its own source count, the
    # published given-k convention) or `sweep` (the pipeline's k, for §8.5.1's
    # source-fraction axis)
    source_budget_mode: str = "episode"
    # Whether the arm's evaluator provides a forward model at all. False is the
    # `@native` condition (research/source_localization.md §2.4.3): the harness
    # scores with one real episode and the canned kernel-using baselines get a
    # raiser instead of a kernel. A GENERATED program is offline under every
    # condition and never sees the evaluator, so for it this flag only changes
    # what the harness can measure.
    forward_model: bool = True
    # Cascade reconstruction: the recovered object is a whole TRAJECTORY rather
    # than a set, so the contract is `reconstruct()` and the reward is
    # `lambda * PathPrecision + (1 - lambda) * EventF1` instead of source F1. Set
    # from the task registry's `reconstructs` flag, and read by every path that has
    # to pick the decoder contract, the masked-episode pool or the tree metrics.
    # Narrows `recovers` exactly as `blocks` narrows `contains`.
    reconstructs: bool = False
    # Weight on PathPrecision in that reward. >= 0.5 is a REQUIREMENT rather than a
    # taste (research/cascade_reconstruction.md §2.6): the node set is nearly free,
    # so a search rewarded mostly on Event F1 discovers the tree contributes
    # nothing to its score and converges on decoders that never attempt it.
    tree_weight: float = 0.6
    # Which of §2.7's four settings this run masks under. Four PROTOCOLS, not four
    # knobs: a decoder selected under final_snapshot is solving a different problem
    # from one selected under partial_times, so the rows are never pooled (§8.3).
    observation_setting: str = "partial_times"
    # Influence blocking: TWO cascades. The planner fights a rumour seeded from
    # `outbreak` (which is S_N here) and its own counter-cascade is the instrument
    # rather than the score. Set from the task registry's `competitive` flag, and
    # read by every path that has to pick the two-cascade simulator, the 8-channel
    # feature builder or the 4-target head.
    competitive: bool = False
    # Which cascade wins a node both reach on the same step, RESOLVED (never `auto`).
    # §8.4 calls this a reported hyperparameter rather than an implementation detail
    # and most papers never state theirs, so it reaches the prompt, the head and the
    # simulator from one place. Inert unless `competitive`.
    tie_break: str = "positive"
    # Budak's detection delay r (§8.3): the rumour is detected r steps late and the
    # blocker emits nothing before then. 0 is every published table's setting.
    detection_delay: int = 0
    # Epidemic control: the state is four EXCLUSIVE compartments rather than two
    # overlapping indicators, and the transition is a per-node matrix rather than a
    # probability. Set from the task registry's `epidemic` flag, and read by every
    # path that has to pick the compartmental simulator, the 9-channel feature
    # builder or the 5-target head. Narrows `contains` exactly as `blocks` does.
    epidemic: bool = False
    # Which of research/epidemic_control.md §2.5's four interventions the budget
    # buys. Carried as a FIELD rather than derived from `budget_op`, because
    # `vaccinate` and `quarantine` both spend `remove_node` and differ only in what
    # the harness expands it into: the first deletes the node, the second isolates
    # it and leaves it counted (§8.2 trap 7).
    epi_lever: str = "vaccinate"
    # Multiplier `set_edge_weight` writes under the `contact_reduce` lever. 0.0 is a
    # full cut through the weight channel, so it is directly comparable to
    # `edge_cut` at the same k; a value in (0, 1) is graded social distancing, which
    # is expressible only because we wrote our own stepper (§2.2).
    contact_reduction: float = 0.0
    # The compartmental rates, RESOLVED, so the prompt can state what is being
    # simulated. §8.2 trap 2: beta and gamma are free parameters nobody
    # standardizes, and a table that fixes them without saying so is comparable
    # only to itself. Inert unless `epidemic`.
    epi_beta: float = 1.0
    epi_gamma: float = 0.3
    epi_alpha: float = 0.5
    # Cascade prediction: the program PREDICTS a scalar the process produces rather
    # than steering or inverting it, so it writes `predict()`, is scored on a
    # prediction error against a REAL logged cascade, and emits no action ever. Set
    # from the task registry's `objective == forecast`, and read by every path that
    # has to pick the predictor contract, the forecast-instance pool or the
    # popularity metrics.
    #
    # `observes` is the narrowing one, and it narrows `forecasts` exactly as
    # `decodes` narrows `recovers`: the transitions were REPLAYED FROM A LOG, so the
    # dynamics that produced them are not IC, not LT and not known
    # (research/cascade_prediction.md §2.2). The prompt has to say so: a system
    # message asserting IC dynamics over Weibo retweets would be a lie the model
    # would then optimize against.
    observational: bool = False
    # Which quantity `predict()` is scored on. `increment` is CasFlow's own code
    # (`label = P(t_p) - P(t_o)`), `total` is CasFT's Eq. 26. §5.7 difference 3
    # records that the two share a symbol and are not the same quantity, so the
    # choice is reported rather than assumed.
    prediction_target: str = "increment"
    # Which error the reward IS. Every one of these MINIMIZES, which is why
    # `Task.sense` maps `forecast` onto `minimize` rather than reading the objective
    # literally.
    prediction_metric: str = "msle"
    # Steps of history a predictor may see, and steps it must predict over. Both in
    # the replayed corpus's own timesteps rather than seconds, so they are readable
    # beside `horizon` (§8.2 pairs two observation windows per corpus deliberately:
    # a single-window result is not publishable in this literature).
    observation_window: int = 5
    # Unrolls averaged inside one `forecast_marginals` call. The cost knob of §2.4:
    # each unroll pays `steps` metered kernel evaluations, so an @monte_carlo arm
    # pays `steps * samples * mc_runs` real episodes per call and a @world_model arm
    # pays `steps * samples` matmuls.
    forecast_samples: int = 8
    # Drives the edit stream's schedule. Carried on the task rather than read
    # from the environment so the same seed produces the same graph history for
    # every arm, which is what makes a cross-arm comparison under a stream mean
    # anything at all.
    seed: int = 0

    @property
    def adaptive(self) -> bool:
        return self.rounds is not None or self.per_round_budget is not None

    @property
    def contains(self) -> bool:
        """
        True when the planner is fighting a cascade it did not start.

        `not self.forecasts` is load-bearing rather than defensive: a forecast
        task's reward is a prediction ERROR, so its `sense` is `minimize` for a
        reason that has nothing to do with containment. Without the guard every
        reader that asks `contains`: the prompt, the summary, the structural
        metrics block, the plots: would describe a cascade-prediction arm as an
        outbreak it was trying to shrink.
        """
        return self.sense == minimize and not self.forecasts

    @property
    def blocks(self) -> bool:
        """
        True for influence blocking specifically, not for containment in general.

        Both minimize a cascade they did not start, and the difference is what the
        budget buys: a critical-node arm only ever DELETES, while a blocking arm may
        also seed a counter-cascade that spreads and competes. That second cascade is
        what needs a different simulator, a different feature layout and a different
        head, so it needs a predicate of its own rather than riding on `contains`.
        """
        return self.competitive

    @property
    def immunizes(self) -> bool:
        """
        True for epidemic control specifically, not for containment in general.

        Same relationship `blocks` has to `contains`, and the difference is what
        the DYNAMICS are rather than what the budget buys: a critical-node arm
        fights a monotone cascade, while this one fights a compartmental process in
        which nodes RECOVER and (under SIS) become susceptible again. That
        non-monotonicity is what needs a different simulator, a different feature
        layout and a head that composes a transition matrix rather than a
        probability (research/epidemic_control.md §2.4).
        """
        return self.epidemic

    @property
    def recovers(self) -> bool:
        """
        True when the program infers a hidden cause instead of choosing an action.

        The third problem family, and the one that changes the CONTRACT rather than
        only the sign: no rollout, no action bag, no budget spent on the graph,
        `localize(graph, observation, budget)` returning the nodes that started the
        cascade, scored on F1 against the truth.
        """
        return self.objective_kind == recover

    @property
    def decodes(self) -> bool:
        """
        True for cascade reconstruction specifically, not for the recover family.

        Both members of that family invert something and neither emits an action;
        the difference is WHAT is recovered. A localizer names a set of `|V|`
        candidates and is scored on F1; a decoder produces a coherent `T x |V|`
        trajectory and is scored on a tree, which needs a different contract, a
        different reward and a transmission edge in the data that NDlib does not
        otherwise produce. Same relationship `blocks` has to `contains`.
        """
        return self.reconstructs

    @property
    def forecasts(self) -> bool:
        """
        True when the program PREDICTS a scalar instead of choosing or inverting.

        The fourth problem family. It changes the contract as much as `recovers`
        does and gives up more: `predict(graph, observation, horizon) -> float`,
        scored on a prediction error, with `a_t` NULL at every step so no action is
        ever emitted and no budget is ever spent
        (research/cascade_prediction.md §2.1).
        """
        return self.objective_kind == forecast

    @property
    def observes(self) -> bool:
        """
        True for cascade prediction specifically, not for the forecast family.

        Same relationship `decodes` has to `recovers`, and the difference is where
        the DATA came from rather than what is predicted: these transitions were
        replayed from a real log, so the process behind them is not IC, not LT and
        not known. That is the one thing the prompt must not get wrong: §2.2 lists
        three specific mechanisms (Hawkes self-excitation, repeated exposure,
        exogenous arrivals) by which real adoption violates the composition rule our
        structured head hard-codes, and a model told it is predicting IC would tune
        against the wrong process.
        """
        return self.observational

    @property
    def reward_unit(self) -> str:
        """
        The noun a reward DELTA is quoted in, for the refinement feedback.

        A property rather than a conditional at each call site because there are now
        four families and the chain was already three deep in two files: a task that
        forgets a branch here reports a `+0.00 nodes` change on an MSLE, which reads
        as "nothing happened" for exactly the moves that mattered most.
        """
        if self.forecasts:
            return self.prediction_metric.upper()

        if self.decodes:
            return "score"

        if self.recovers:
            return "F1"

        return "nodes"

    @property
    def streaming(self) -> bool:
        return self.edit_rate > 0.0

    @property
    def multi_round(self) -> bool:
        return self.campaigns > 1


@dataclass()
class Trajectory:
    """What a rollout returns"""

    states: list[State]
    actions: list[list[ActionOp]]
    reward: float  # final spread (infected count)
    infected_counts: list[float]
    cost: dict = field(default_factory=dict)
    # Per-node P(infected at end) across the ensemble (feedback, not serialized)
    final_marginals: list[float] | None = None
    # E|infected| after each timestep, ENSEMBLE-MEAN and padded to horizon + 2 so
    # index t is always the same t across arms and runs. `infected_counts` is the
    # representative sample and stops when its cascade died, which makes
    # sigma(S, T) unreadable at any T past that point; this is the readable one.
    # Padding by holding the last value is exact rather than an approximation:
    # the rollout only breaks when the frontier AND the action bag are both
    # empty, which is a fixed point of monotone IC/LT.
    spread_curve: list[float] | None = None
    # E|I(t)| after each timestep: the CURRENTLY-infectious count, not the
    # cumulative one. Only a compartmental task fills it, and it is a separate
    # field rather than a reinterpretation of `spread_curve` because the two are
    # genuinely different curves there: the attack set is monotone and the
    # prevalence is not, and §2.6's peak, time-to-peak and AUC are all functions of
    # the second. Padding it by holding the last value would be WRONG for the same
    # reason (a dead epidemic's prevalence is 0, not its last non-zero value) so
    # it is padded with zeros instead.
    prevalence_curve: list[float] | None = None
    # Final reward of every ensemble sample (or every instance on the inverse and
    # forecast families), in sample order. Two rollouts at the same seed share
    # their realizations sample for sample, so the standard error of the mean of
    # the PAIRED differences is the noise of a comparison, and the samples where
    # a candidate lost most are readable as counterexamples. None where samples
    # are not aligned across calls (the multi-round union).
    sample_rewards: list[float] | None = None


class Strategy(Protocol):
    """
    Contract the agent's generated script must implement.

    A script may implement only the method its outer-loop method needs; the executor validates the required one is present.
    """

    # Method 1 (One-shot algorithm generation)
    def plan_horizon(
        self, graph: GraphInfo, budget: int, horizon: int
    ) -> list[list[ActionOp]]: ...

    # Methods 2 (Per-step algorithm generation) and 3 (Windowed algorithm generation)
    def act(self, state: State, graph: GraphInfo, timestep: int) -> list[ActionOp]: ...

    # Inverse tasks (source localization). Not an intervention: given the graph and
    # an observed diffusion state `observation` (P(infected) per node, in [0, 1]),
    # return the `budget` node ids that STARTED the cascade. The program is
    # offline: it sees the graph, `graph.ic_probs` and the observation, and the
    # harness re-simulates what it returns on the arm's evaluator to score it.
    def localize(
        self, graph: GraphInfo, observation: np.ndarray, budget: int
    ) -> list[int]: ...

    # Cascade reconstruction. Also not an intervention: given the graph and a
    # partial `Observation` of a diffusion that already happened, return
    # `{node: (activation timestep, inferred parent)}` for every node believed
    # infected, with `parent = None` marking a source and uninfected nodes simply
    # absent. The program is offline: the arm's transition kernel scores the
    # history it returns (research/cascade_reconstruction.md §2.5), never runs
    # inside it.
    def reconstruct(
        self, graph: GraphInfo, observation: object, horizon: int
    ) -> dict[int, tuple[int, int | None]]: ...

    # Cascade prediction. Neither an intervention nor an inversion: given the graph
    # and a `CascadeObservation` of a REAL cascade's first `t_o` steps, return the
    # popularity it will have reached by `t_p`. Return None (or a non-finite value)
    # to DECLINE: a generative model that cannot score a supercritical cascade is
    # counted in `n_failed` rather than charged a wild guess, which is the column
    # research/cascade_prediction.md §8.4 says almost nobody publishes. The
    # program is offline: no forward model runs inside it (§2.1).
    def predict(
        self, graph: GraphInfo, observation: object, horizon: int
    ) -> float | None: ...

    # Optional companion to localize(). F1 scores the SET, AUC scores the RANKING,
    # so a program that exposes a per-node score vector gets a true AUC; one that
    # does not gets an AUC derived from the ORDER of the list localize() returned,
    # which leaves every un-nominated node tied. Which rule was used is recorded
    # per result rather than silently averaged in.
    def source_scores(
        self, graph: GraphInfo, observation: np.ndarray
    ) -> np.ndarray: ...


class ScoredStrategy:
    """
    Scored-mode contract: plan_horizon and localize are fixed harnesses the agent
    cannot override: generated code may only override score(), schedule(), or
    source_score(). This forces edits to the algorithm's internals instead of
    free-form programs or composition over the library.

    The inverse-task half (`source_score` + the fixed top-k `localize`) is the
    analogue described in research/source_localization.md §2.4.2, and it makes the
    search space directly comparable to the classical methods: LPSI, the
    Comin-Costa centralities and rumor centrality are all exactly node-scoring
    functions over the observed state.
    """

    # `budget_op` and `outbreak` are stamped by methods.base.attach_context before
    # the harness runs; the three oracle slots are filled for CANNED baselines
    # only and stay None on a generated strategy, which is offline. Defaulted here
    # so the class is usable standalone (tests, a bare harness).
    budget_op: str = "add_node"
    outbreak: tuple = ()
    predict_marginals: Callable | None = None
    step_marginals: Callable | None = None
    forecast_marginals: Callable | None = None

    def score(self, node: int, selected: tuple, graph: GraphInfo) -> float:
        return float(graph.degree(node))

    def schedule(
        self, seeds: list[int], graph: GraphInfo, horizon: int
    ) -> list[list[ActionOp]]:
        # The op the TASK budgets, not `add_node`: this harness is shared by
        # seeding and containment, and emitting the wrong one is rejected by
        # validate_actions before the strategy is ever scored
        return [[ActionOp(self.budget_op, node) for node in seeds]] + [
            [] for _ in range(horizon)
        ]

    def plan_horizon(
        self, graph: GraphInfo, budget: int, horizon: int
    ) -> list[list[ActionOp]]:
        selected = []
        # Deleting an outbreak source ends the cascade rather than containing it
        # and is rejected; skipping them here spends the whole budget on legal
        # picks instead of losing the arm to a repair turn the scorer cannot fix
        protected = set(self.outbreak)

        for _ in range(min(budget, graph.num_nodes - len(protected))):
            best_node, best_score = -1, float("-inf")
            for node in range(graph.num_nodes):
                if node in selected or node in protected:
                    continue

                node_score = float(self.score(node, tuple(selected), graph))
                if node_score > best_score:
                    best_node, best_score = node, node_score

            if best_node < 0:
                break

            selected.append(best_node)

        # Normalize whatever schedule() returns to exactly horizon+1 bags
        plan = [list(bag) for bag in self.schedule(selected, graph, horizon)]
        plan = plan[: horizon + 1]
        plan += [[] for _ in range(horizon + 1 - len(plan))]

        return plan

    def source_score(
        self,
        node: int,
        graph: GraphInfo,
        observation: np.ndarray,
        selected: tuple,
    ) -> float:
        """
        How likely `node` is to have STARTED the observed cascade. Higher = named sooner.

        The default is the Comin-Costa floor: degree within the infected subgraph.
        `selected` is the tuple of sources already named, so a rule can penalize a
        candidate whose neighbourhood an earlier pick already explains.
        """
        if observation[node] < 0.5:
            return float("-inf")

        return float(
            sum(1 for other in graph.out_neighbors(node) if observation[other] >= 0.5)
        )

    def source_scores(
        self, graph: GraphInfo, observation: np.ndarray
    ) -> np.ndarray:
        """
        The whole score vector at the first pick, so scored mode gets a REAL AUC.

        Free for the agent: it wrote `source_score`, and this only evaluates it
        once per node with an empty `selected`. Non-finite entries (the natural way
        a rule rules a node out) are pushed below the lowest finite score rather
        than left as -inf, so the ranking stays total.
        """
        scores = np.array(
            [
                float(self.source_score(node, graph, observation, ()))
                for node in range(graph.num_nodes)
            ],
            dtype=np.float64,
        )
        finite = scores[np.isfinite(scores)]
        floor = float(finite.min()) - 1.0 if finite.size else 0.0

        return np.where(np.isfinite(scores), scores, floor)

    def localize(
        self, graph: GraphInfo, observation: np.ndarray, budget: int
    ) -> list[int]:
        """Fixed top-k harness over source_score; not overridable in scored mode."""
        selected = []

        for _ in range(min(budget, graph.num_nodes)):
            best_node, best_score = -1, float("-inf")

            for node in range(graph.num_nodes):
                if node in selected:
                    continue

                node_score = float(
                    self.source_score(node, graph, observation, tuple(selected))
                )
                if node_score > best_score:
                    best_node, best_score = node, node_score

            if best_node < 0:
                break

            selected.append(best_node)

        return selected

    def growth_factor(
        self, features: dict, graph: GraphInfo, observation: object
    ) -> float:
        """
        Multiplier on the OBSERVED popularity. 1.0 = the cascade is already over.

        Scored mode for cascade prediction, and the tightest fit of the four:
        Szabo & Huberman's founding result is that `log P(t_p)` is near-linear in
        `log P(t_o)`, i.e. that the whole problem is a multiplier, so a search over
        multipliers is a search over exactly the space §3.1's feature line occupies,
        rather than a subset of it. `features` is `cascade_features(...)`: Cheng et
        al.'s five classes (root, structural, temporal, community, and the observed
        counts) computed once per instance, so a rule can key off the reshare rate
        in the second half of the window, which is the single best feature that
        paper found.

        The default is the doubling constant Cheng et al. built their whole
        classification framing around: predict that a cascade reaches twice what it
        has, which is the median outcome their balanced task is defined by.
        """
        return 2.0

    def predict(
        self, graph: GraphInfo, observation: object, horizon: int
    ) -> float | None:
        """
        Fixed `P(t_o) * growth_factor` harness; not overridable in scored mode.

        Clamped at the observed popularity from below because a progressive cascade
        cannot shrink: an observation is ground truth about the nodes it reports, so
        a multiplier below 1 predicts adopters un-adopting.
        """
        from coding_agent.tools.prediction_algorithms import cascade_features

        observed = float(getattr(observation, "popularity", 0.0))
        factor = float(self.growth_factor(cascade_features(graph, observation), graph, observation))

        return max(observed, observed * factor)

    def edge_cost(
        self, source: int, target: int, probability: float, graph: GraphInfo,
        observation: object
    ) -> float:
        """
        How implausible the transmission `source -> target` is. LOWER = more likely.

        Scored mode for cascade reconstruction, and the tightest fit of the three:
        every ordered-Steiner method in `research/cascade_reconstruction.md` §3 IS
        exactly a shortest-path computation under an arc cost, so the constrained
        search space is directly comparable to the classical methods rather than a
        subset of them. The default is the likelihood metric they all use: a
        most-likely path is a shortest path under `-log p`.

        `observation.reported` is the observed report set, so a rule may make an
        arc into an already-reported node cheap, or penalize one leaving the
        observed region.
        """
        return -math.log(max(float(probability), 1e-9))

    def reconstruct(
        self, graph: GraphInfo, observation: object, horizon: int
    ) -> dict[int, tuple[int, int | None]]:
        """
        Fixed ordered-Steiner harness over edge_cost; not overridable in scored mode.

        Grow a tree out of the estimated roots by cheapest-arc-first under
        `edge_cost`, respecting any observed activation time, then hand the time
        assignment to the shared `finalize` so the parent rule is identical to
        every library decoder's and a score difference is a difference in the cost
        function alone.
        """
        import heapq

        from coding_agent.tools.reconstruction_algorithms import (
            _merge_observed,
            _probability_map,
            estimate_roots,
            finalize,
            reported_nodes,
        )

        members = set(reported_nodes(observation))
        if not members:
            return {}

        probabilities = _probability_map(graph)
        known = observation.times
        roots = estimate_roots(graph, sorted(members), observation)

        times = {int(root): int(known.get(root, 0)) for root in roots}
        costs = {int(root): 0.0 for root in roots}
        queue = [(0.0, int(root)) for root in roots]
        heapq.heapify(queue)

        while queue:
            cost, node = heapq.heappop(queue)
            if cost > costs.get(node, float("inf")):
                continue

            step = times[node] + 1
            if step > horizon:
                continue

            for neighbour in graph.out_neighbors(node):
                observed = known.get(int(neighbour))
                if observed is not None and step > observed:
                    continue

                candidate = cost + float(
                    self.edge_cost(
                        int(node),
                        int(neighbour),
                        probabilities.get((int(node), int(neighbour)), 0.0),
                        graph,
                        observation,
                    )
                )
                if candidate < costs.get(int(neighbour), float("inf")):
                    costs[int(neighbour)] = candidate
                    times[int(neighbour)] = max(
                        0, min(int(observed if observed is not None else step), horizon)
                    )
                    heapq.heappush(queue, (candidate, int(neighbour)))

        for node in members:
            times.setdefault(int(node), int(known.get(node, horizon)))

        kept = {
            node: time
            for node, time in times.items()
            if node in members or time < horizon
        }

        return finalize(
            graph, _merge_observed(kept, observation, horizon), observation, horizon
        )
