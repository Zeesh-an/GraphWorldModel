"""
Two-cascade competitive IC / CLT simulator: the dynamics `influence_blocking` runs on.

A negative cascade (the rumour, `S_N`) and a positive one (the blocker's
counter-cascade) spread over the same graph, and a susceptible node reached by both
in the same step is resolved by an explicit, recorded TIE-BREAK. That last clause is
why this file exists at all rather than reusing NDlib
(`research/influence_blocking.md` §2.4): no NDlib model carries two competing
cascade labels, `Blocked: -1` is a static non-adopter set, and `CompositeModel`
loops its rules in registration order and breaks on the first that fires, so its
only expressible tie-break is a single global "first rule wins".

Three parameters decide what is being simulated, and all three are recorded in
`data/metadata.json` because §5.1 shows each one moves the published numbers:

  * **tie_break**: `negative` / `positive` / `fixed` dominance (§8.4). `auto`
    resolves to each dynamics' own founding paper: PD under IC, because Budak's
    MCICM and COICM both hard-code "if the bad and the good information reach a node
    at the same step, the good information takes effect"; ND under LT, because He
    et al.'s CLT is defined with negative dominance and their submodularity proof is
    for that rule. Most papers never state which they used, so it is a flag with a
    printed default rather than an artifact of node iteration order.

  * **positive_prob**: `shared` (COICM: one probability per edge, independent of
    information type) or a constant `c` (MCICM: the limiting campaign transmits at
    `c` on every edge; `c = 1.0` is Budak's high-effectiveness property, the case
    Theorem 4.2 proves submodular).

  * **remove_semantics**: `blocked` throughout, as every containment task needs:
    a removed node is deleted, uncounted, and cannot transmit or be infected by
    either cascade.

Warning: A NOTE ON §2.2's COICM CAVEAT, because it changes what the head has to be.
That section warns that a two-MLP product form "silently assumes MCICM, not COICM",
on the grounds that COICM shares one coin per edge between the campaigns and so
makes `p^N` and `p^P` dependent. Under the LIVE-EDGE characterisation that is right.
Under the stepwise simulation it does not bite, and the reason is worth stating
because it is what makes `CompetitiveICHead` exact rather than approximate: a node
activates in at most ONE campaign, so each arc `(u, v)` is ever attempted by exactly
one of them and carries exactly one coin either way. The two campaigns' arrivals at a
susceptible `v` come from DISJOINT in-edge sets, so `p^N(v)` and `p^P(v)` are
products over disjoint independent coins and multiplying them is correct under both
models. What actually separates COICM from MCICM in a forward simulation is only
whether `p_positive == p_negative`, which is exactly what `positive_prob` sets.
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np

from data.wm_simulator import ActionOp, State, blocked, valid_remove_semantics

# Which cascade wins a susceptible node reached by both in the same step (§8.4).
# The rules differ ONLY in how ties in arrival time resolve, which is a one-line
# change in a simulator and a large change in the reported numbers.
negative_dominance = "negative"
positive_dominance = "positive"
fixed_dominance = "fixed"
auto_dominance = "auto"
valid_tie_breaks = (negative_dominance, positive_dominance, fixed_dominance)
tie_break_choices = (auto_dominance,) + valid_tie_breaks

# The two competitive-IC variants of the survey's Table 1, and the ONE thing that
# separates them in a forward simulation (see the module docstring).
coicm = "coicm"
mcicm = "mcicm"

# `--positive-prob shared` is COICM; anything else is a constant p_L and therefore
# MCICM. Budak's high-effectiveness property is `shared_positive_prob` replaced by
# 1.0, which is the case his Theorem 4.2 proves submodular.
shared_positive_prob = "shared"


def resolve_tie_break(tie_break: str, diffusion_model: str) -> str:
    """
    `auto` -> the rule each dynamics' own founding paper hard-codes.

    Budak's MCICM/COICM resolve simultaneous arrival in favour of the GOOD campaign
    (WWW'11 §3.1); He et al.'s CLT resolves it in favour of the negative one (SDM'12).
    Running IC under ND or LT under PD is a legitimate experiment and is what the
    explicit values are for: it is just not what either paper measured.
    """
    if tie_break == auto_dominance:
        return negative_dominance if diffusion_model == "LT" else positive_dominance

    if tie_break not in valid_tie_breaks:
        raise ValueError(
            f"unknown tie_break {tie_break!r}; choose one of {tie_break_choices}"
        )

    return tie_break


def competitive_model_name(positive_prob: str | float) -> str:
    """Which row of the survey's Table 1 a `positive_prob` setting is."""
    return coicm if positive_prob == shared_positive_prob else mcicm


@dataclass()
class CompetitiveConfig:
    """Everything about the competitive dynamics that a run has to record."""

    tie_break: str = auto_dominance
    positive_prob: str | float = shared_positive_prob
    remove_semantics: str = blocked

    def resolved(self, diffusion_model: str) -> dict:
        return {
            "tie_break": resolve_tie_break(self.tie_break, diffusion_model),
            "positive_prob": self.positive_prob,
            "competitive_model": competitive_model_name(self.positive_prob),
            "remove_semantics": self.remove_semantics,
        }


class CompetitiveSimulator:
    """
    Synchronous two-cascade IC / CLT, with the same `advance(bag) -> State` contract
    the single-cascade `Simulator` has.

    Synchronous is not a style choice: NDlib's own IC iterates nodes in dict order,
    which is harmless for one cascade and would leak an implicit, unrecorded
    tie-break for two. Every node's arrivals for a step are collected against the
    PRE-STEP frontiers and only then committed.

    `add_node` always seeds the POSITIVE cascade. The negative seed set is an input
    to the episode rather than an action (§2.1) (it is committed by `reset`) so the
    three action channels keep the meaning they have in every other task and the
    blocker can only ever help itself.
    """

    def __init__(
        self,
        graph: nx.Graph | nx.DiGraph,
        ic_prob_map: dict,
        seed: int = 0,
        config: CompetitiveConfig | None = None,
    ) -> None:
        config = config or CompetitiveConfig()

        if config.remove_semantics not in valid_remove_semantics:
            raise ValueError(
                f"unknown remove_semantics {config.remove_semantics!r}; "
                f"choose one of {valid_remove_semantics}"
            )
        # Only `blocked` is implemented here: a removed node leaves both cascades.
        # Accepting `spent` would validate a value the stepper then ignores, and
        # the dataset's metadata would claim a semantics its records never had
        if config.remove_semantics != blocked:
            raise ValueError(
                f"the competitive simulator implements remove_semantics="
                f"{blocked!r} only, got {config.remove_semantics!r}"
            )

        self.graph = graph
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.num_nodes = graph.number_of_nodes()
        self.model_name = None
        self.tie_break = None

        # {(u, v): p} for the negative cascade, mutated by the edge ops. The base
        # map is copied per reset so an episode's edits never reach the bundle.
        self.base_edges = {
            (int(source), int(target)): float(probability)
            for (source, target), probability in (ic_prob_map or {}).items()
        }
        self.edges = {}
        self.out_edges = {}
        self.in_edges = {}

        self.neg_infected = set()
        self.neg_frontier = set()
        self.pos_infected = set()
        self.pos_frontier = set()
        self.blocked = set()
        # Per-node priority for FIXED dominance and the two hidden CLT thresholds.
        # Drawn per episode and NEVER stored, exactly like the single-cascade LT
        # threshold, which is where the head's partial-observability tax comes from.
        self.priority = np.zeros(self.num_nodes, dtype=np.float64)
        self.neg_threshold = {}
        self.pos_threshold = {}

    # Setup
    def reset(
        self,
        model_name: str,
        negative_seeds: list[int] | tuple = (),
        lt_thresholds: tuple[dict, dict] | None = None,
    ) -> None:
        if model_name not in ("IC", "LT"):
            raise ValueError(f"unknown diffusion model {model_name!r}; choose IC or LT")

        self.model_name = model_name
        self.tie_break = resolve_tie_break(self.config.tie_break, model_name)
        self.edges = dict(self.base_edges)
        self._rebuild_adjacency()

        self.neg_infected = {int(node) for node in negative_seeds}
        self.neg_frontier = set(self.neg_infected)
        self.pos_infected = set()
        self.pos_frontier = set()
        self.blocked = set()

        self.priority = self.rng.random(self.num_nodes)

        if model_name == "LT":
            # CLT: TWO hidden thresholds per node (He et al. SDM'12 §3), which is
            # what makes the LT one-step ceiling lower here than for single-cascade
            # LT rather than equal to it
            if lt_thresholds is None:
                self.neg_threshold = {
                    node: float(self.rng.uniform(0.0, 1.0))
                    for node in range(self.num_nodes)
                }
                self.pos_threshold = {
                    node: float(self.rng.uniform(0.0, 1.0))
                    for node in range(self.num_nodes)
                }
            else:
                self.neg_threshold, self.pos_threshold = lt_thresholds

    def _rebuild_adjacency(self) -> None:
        self.out_edges = {node: [] for node in range(self.num_nodes)}
        self.in_edges = {node: [] for node in range(self.num_nodes)}

        for (source, target), probability in self.edges.items():
            self.out_edges[source].append((target, probability))
            self.in_edges[target].append((source, probability))

    def positive_probability(self, probability: float) -> float:
        """`p_L` for one arc: the shared value under COICM, the constant under MCICM."""
        if self.config.positive_prob == shared_positive_prob:
            return probability

        return float(min(1.0, max(0.0, float(self.config.positive_prob))))

    # State
    def current_state(self) -> State:
        return State(
            infected=sorted(self.neg_infected),
            frontier=sorted(self.neg_frontier),
            pos_infected=sorted(self.pos_infected),
            pos_frontier=sorted(self.pos_frontier),
        )

    def apply_actions(self, bag: list[ActionOp]) -> None:
        """
        The five ops, with `add_node` seeding the POSITIVE cascade.

        A blocker may seed a node the rumour already owns and it simply does
        nothing: the node is already committed, exactly as re-seeding an active node
        does nothing under single-cascade IC.
        """
        rebuild = False

        for action in bag:
            target = int(action.target)

            if action.op == "add_node":
                if target in self.blocked or target in self.neg_infected:
                    continue

                self.pos_infected.add(target)
                self.pos_frontier.add(target)
            elif action.op == "remove_node":
                # Containment reading only: `spent` would leave the node counted in
                # the negative total and bias every blocking number by +k
                self.blocked.add(target)
                self.neg_infected.discard(target)
                self.neg_frontier.discard(target)
                self.pos_infected.discard(target)
                self.pos_frontier.discard(target)
            elif action.op == "add_edge":
                self.edges[(target, int(action.destination))] = float(action.weight)
                rebuild = True
            elif action.op == "remove_edge":
                self.edges.pop((target, int(action.destination)), None)
                rebuild = True
            elif action.op == "set_edge_weight":
                edge = (target, int(action.destination))
                if edge in self.edges:
                    self.edges[edge] = float(action.weight)
                    rebuild = True

        if rebuild:
            self._rebuild_adjacency()

    # Dynamics
    def _ic_arrivals(self) -> tuple[dict, dict]:
        """One synchronous IC step: which susceptible nodes each cascade reached."""
        negative, positive = {}, {}

        for source in self.neg_frontier:
            for target, probability in self.out_edges[source]:
                if self._susceptible(target) and self.rng.random() < probability:
                    negative[target] = True

        for source in self.pos_frontier:
            for target, probability in self.out_edges[source]:
                if self._susceptible(target) and self.rng.random() < self.positive_probability(
                    probability
                ):
                    positive[target] = True

        return negative, positive

    def _lt_arrivals(self) -> tuple[dict, dict]:
        """One synchronous CLT step: each cascade's active in-weight vs its threshold."""
        negative, positive = {}, {}

        for target in range(self.num_nodes):
            if not self._susceptible(target):
                continue

            neg_weight = pos_weight = total = 0.0
            for source, probability in self.in_edges[target]:
                if source in self.blocked:
                    continue

                total += probability
                if source in self.neg_infected:
                    neg_weight += probability
                elif source in self.pos_infected:
                    pos_weight += probability

            if total <= 0.0:
                continue

            if neg_weight / total >= self.neg_threshold[target] and neg_weight > 0.0:
                negative[target] = True

            if pos_weight / total >= self.pos_threshold[target] and pos_weight > 0.0:
                positive[target] = True

        return negative, positive

    def _susceptible(self, node: int) -> bool:
        return (
            node not in self.neg_infected
            and node not in self.pos_infected
            and node not in self.blocked
        )

    def _resolve(self, negative: dict, positive: dict) -> tuple[set, set]:
        """
        Commit one step's arrivals under the configured tie-break.

        The three rules differ only for a node BOTH cascades reached, which is the
        whole content of §8.4, and the reason this is one function rather than
        three scattered conditionals.
        """
        new_negative, new_positive = set(), set()

        for node in set(negative) | set(positive):
            hit_negative = node in negative
            hit_positive = node in positive

            if hit_negative and hit_positive:
                if self.tie_break == negative_dominance:
                    new_negative.add(node)
                elif self.tie_break == positive_dominance:
                    new_positive.add(node)
                elif self.priority[node] < 0.5:
                    new_negative.add(node)
                else:
                    new_positive.add(node)
            elif hit_negative:
                new_negative.add(node)
            else:
                new_positive.add(node)

        return new_negative, new_positive

    def advance(self, bag: list[ActionOp]) -> State:
        """s_{t+1} = T_endo(T_exo(s_t, a_t)), both cascades in lockstep."""
        self.apply_actions(bag)

        negative, positive = (
            self._ic_arrivals() if self.model_name == "IC" else self._lt_arrivals()
        )
        new_negative, new_positive = self._resolve(negative, positive)

        self.neg_infected |= new_negative
        self.pos_infected |= new_positive
        self.neg_frontier = new_negative
        self.pos_frontier = new_positive

        return self.current_state()

    def advance_marginal(
        self, bag: list[ActionOp], num_mc: int
    ) -> tuple[State, dict, dict, dict, dict]:
        """
        One-step marginals for both cascades: the four soft targets the head is fit on.

        Same shape as `Simulator.advance_marginal` with four count dicts instead of
        two (§2.1). The action is applied ONCE (it is deterministic) and only the
        diffusion step is redrawn.
        """
        if num_mc < 1:
            raise ValueError(f"num_mc must be >= 1, got {num_mc}")

        # What each campaign's thresholds were last survived AGAINST, captured
        # before the action mutates the sets: the conditioning of the closed-form
        # CLT hazard targets below
        previous_negative = set(self.neg_infected) - set(self.neg_frontier)
        previous_positive = set(self.pos_infected) - set(self.pos_frontier)

        self.apply_actions(bag)
        post_action = self.snapshot()

        # CLT is deterministic given its per-episode thresholds, so ONE realized
        # draw continues the trajectory and the targets are computed in closed
        # form below. (Until 2026-09-01 this redrew BOTH threshold tables per
        # draw and left the last redraw in place, so recorded CLT trajectories
        # were a memoryless process rather than the threshold-persistent CLT
        # that advance() and the referee run.)
        draws = num_mc if self.model_name == "IC" else 1
        counts = [{}, {}, {}, {}]
        last_state = None

        for _ in range(draws):
            self.restore(post_action)

            negative, positive = (
                self._ic_arrivals() if self.model_name == "IC" else self._lt_arrivals()
            )
            new_negative, new_positive = self._resolve(negative, positive)

            groups = (
                self.neg_infected | new_negative,
                new_negative,
                self.pos_infected | new_positive,
                new_positive,
            )
            for table, group in zip(counts, groups, strict=True):
                for node in group:
                    table[node] = table.get(node, 0) + 1

            last_state = State(
                infected=sorted(groups[0]),
                frontier=sorted(groups[1]),
                pos_infected=sorted(groups[2]),
                pos_frontier=sorted(groups[3]),
            )

        if self.model_name == "IC":
            marginals = [
                {int(node): count / draws for node, count in table.items()}
                for table in counts
            ]
        else:
            # The exact conditional law under the two persistent U(0,1)
            # thresholds, composed through the tie-break; a `spent` reading does
            # not exist here (the constructor refuses it), so there is no
            # re-entry approximation to make
            marginals = self._clt_hazard_marginals(
                previous_negative, previous_positive
            )

        # The simulator's own state advances to the LAST draw, matching
        # Simulator.advance_marginal: the episode continues down one realized path
        # while the targets describe the distribution over all of them
        self.neg_infected = set(last_state.infected)
        self.neg_frontier = set(last_state.frontier)
        self.pos_infected = set(last_state.pos_infected)
        self.pos_frontier = set(last_state.pos_frontier)

        return (last_state, *marginals)

    def _clt_hazard_marginals(
        self, previous_negative: set, previous_positive: set
    ) -> list[dict]:
        """Closed-form one-step CLT marginals for the four targets, post-action."""
        neg_infected_marginal, neg_frontier_marginal = {}, {}
        pos_infected_marginal, pos_frontier_marginal = {}, {}

        for node in range(self.num_nodes):
            if node in self.blocked:
                continue

            if node in self.neg_infected:
                neg_infected_marginal[node] = 1.0
                continue

            if node in self.pos_infected:
                pos_infected_marginal[node] = 1.0
                continue

            neg_now = pos_now = neg_prev = pos_prev = total = 0.0
            for source, probability in self.in_edges[node]:
                if source in self.blocked:
                    continue
                total += probability
                if source in self.neg_infected:
                    neg_now += probability
                elif source in self.pos_infected:
                    pos_now += probability
                if source in previous_negative:
                    neg_prev += probability
                elif source in previous_positive:
                    pos_prev += probability

            if total <= 0.0:
                continue

            def conditional(now: float, previous: float) -> float:
                if now <= 0.0:
                    return 0.0
                f_now, f_prev = now / total, previous / total
                return max(0.0, f_now - f_prev) / max(1e-9, 1.0 - f_prev)

            hit_negative = conditional(neg_now, neg_prev)
            hit_positive = conditional(pos_now, pos_prev)

            # Independent thresholds, so the joint factorizes and the tie-break
            # composes the two hazards exactly as _resolve composes the arrivals
            if self.tie_break == negative_dominance:
                new_negative = hit_negative
                new_positive = hit_positive * (1.0 - hit_negative)
            elif self.tie_break == positive_dominance:
                new_positive = hit_positive
                new_negative = hit_negative * (1.0 - hit_positive)
            else:
                both = hit_negative * hit_positive
                negative_wins = 1.0 if self.priority[node] < 0.5 else 0.0
                new_negative = (
                    hit_negative * (1.0 - hit_positive) + negative_wins * both
                )
                new_positive = (
                    hit_positive * (1.0 - hit_negative)
                    + (1.0 - negative_wins) * both
                )

            if new_negative > 0.0:
                neg_infected_marginal[node] = new_negative
                neg_frontier_marginal[node] = new_negative
            if new_positive > 0.0:
                pos_infected_marginal[node] = new_positive
                pos_frontier_marginal[node] = new_positive

        return [
            neg_infected_marginal,
            neg_frontier_marginal,
            pos_infected_marginal,
            pos_frontier_marginal,
        ]

    # Fork support
    def snapshot(self) -> tuple:
        return (
            set(self.neg_infected),
            set(self.neg_frontier),
            set(self.pos_infected),
            set(self.pos_frontier),
            set(self.blocked),
            dict(self.edges),
        )

    def restore(self, snapshot: tuple) -> None:
        (
            self.neg_infected,
            self.neg_frontier,
            self.pos_infected,
            self.pos_frontier,
            self.blocked,
            edges,
        ) = (
            set(snapshot[0]),
            set(snapshot[1]),
            set(snapshot[2]),
            set(snapshot[3]),
            set(snapshot[4]),
            dict(snapshot[5]),
        )

        # Unlike the NDlib-backed Simulator, the graph lives HERE, so a restore puts
        # the edge table back too and no `revert_edges` companion is needed
        if edges.keys() != self.edges.keys() or edges != self.edges:
            self.edges = edges
            self._rebuild_adjacency()


def run_competitive(
    graph: nx.Graph | nx.DiGraph,
    ic_prob_map: dict,
    diffusion_model: str,
    negative_seeds: list[int],
    plan: list[list[ActionOp]],
    horizon: int,
    seed: int = 0,
    config: CompetitiveConfig | None = None,
) -> State:
    """
    One competitive episode start to finish: seed `S_N`, apply `plan[t]`, run `horizon` steps.

    The shared driver behind the MC environment, the blocking library's own spread
    estimator, and the self-check, so all three agree on what "run this blocker set"
    means down to the tie-break.
    """
    simulator = CompetitiveSimulator(graph, ic_prob_map, seed=seed, config=config)
    simulator.reset(diffusion_model, negative_seeds)
    state = simulator.current_state()

    for timestep in range(horizon + 1):
        bag = plan[timestep] if timestep < len(plan) else []
        state = simulator.advance(bag)

        if timestep > 0 and not state.frontier and not state.pos_frontier and not bag:
            break

    return state
