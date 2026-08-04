"""
Core contracts for the coding-agent outer loop.

Reuses the canonical ActionOp/State value types from the data simulator so the
strategies, environments, and the trained world model all speak the same action vocabulary.
"""

from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol
import numpy as np

from data.wm_simulator import ActionOp, State, spent, valid_action_ops
from pipeline.tasks import maximize, minimize, recover

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
    registry field flips the whole search rather than N scattered comparisons —
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


@dataclass
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


@dataclass
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
    # Where each instance's k comes from — `episode` (its own source count, the
    # published given-k convention) or `sweep` (the pipeline's k, for §8.5.1's
    # source-fraction axis)
    source_budget_mode: str = "episode"
    # Whether `predict_marginals` is bound to a real evaluator for this arm. False
    # is the `@native` condition of research/source_localization.md §2.4.3: the
    # program has NO forward model and must be a pure structural heuristic, which
    # is the arm that answers "is a forward model in the search loop worth
    # anything at all". Ignored by every task that does not invert.
    forward_model: bool = True
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
        """True when the planner is fighting a cascade it did not start."""
        return self.sense == minimize

    @property
    def recovers(self) -> bool:
        """
        True when the program infers a hidden cause instead of choosing an action.

        The third problem family, and the one that changes the CONTRACT rather than
        only the sign: no rollout, no action bag, no budget spent on the graph —
        `localize(graph, observation, budget)` returning the nodes that started the
        cascade, scored on F1 against the truth.
        """
        return self.objective_kind == recover

    @property
    def streaming(self) -> bool:
        return self.edit_rate > 0.0

    @property
    def multi_round(self) -> bool:
        return self.campaigns > 1


@dataclass
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
    # return the `budget` node ids that STARTED the cascade. The harness binds
    # `self.predict_marginals(seeds) -> np.ndarray` before calling this, which is
    # the forward oracle the program may query; under the @native condition that
    # attribute raises instead (research/source_localization.md §2.4).
    def localize(
        self, graph: GraphInfo, observation: np.ndarray, budget: int
    ) -> list[int]: ...

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
    cannot override — generated code may only override score(), schedule(), or
    source_score(). This forces edits to the algorithm's internals instead of
    free-form programs or composition over the library.

    The inverse-task half (`source_score` + the fixed top-k `localize`) is the
    analogue described in research/source_localization.md §2.4.2, and it makes the
    search space directly comparable to the classical methods: LPSI, the
    Comin-Costa centralities and rumor centrality are all exactly node-scoring
    functions over the observed state.
    """

    # All three are stamped by methods.base.attach_context before the harness runs.
    # Defaulted here so the class is usable standalone (tests, a bare harness) and
    # so a seeding task needs no context at all.
    budget_op: str = "add_node"
    outbreak: tuple = ()
    predict_marginals: Callable | None = None

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

        Free for the agent — it wrote `source_score`, and this only evaluates it
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
