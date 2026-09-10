"""
`PlanDiagnostics` — the world model as EVIDENCE rather than as one scalar.

The coding-agent loop's inner feedback has been `final_spread=143.7` plus a
handful of structural hints. The model that produced that number also produced
`p_v = P(v influenced | S)` for every node, and several counterfactual rollouts
are affordable at its price. This module turns both into a small set of probes a
reviser can act on:

    wm.evaluate(plan)                      V(S) and the marginals behind it
    wm.probe_drop(plan, node)              V(S) - V(S \\ {v})
    wm.probe_swap(plan, old, new)          V(S - old + new) - V(S)
    wm.probe_region(plan, nodes)           sum_{v in R} p_v
    wm.summarize_regions(plan)             the same, over a graph partition
    wm.redundancy(plan)                    pairwise seed overlap
    wm.bridge_report(plan)                 predicted activation on bridge nodes
    wm.stagnation(history)                 is the search still moving
    wm.explain_plan(plan)                  all of the above, as prompt text

Three rules this module is written to keep, because the experiment it serves is
worthless without them:

1. **No trusted-simulator information enters through here.** Everything comes
   from (a) the graph, (b) the bound evaluator, (c) deterministic arithmetic on
   its output. Nothing reads the simulator directly, and nothing reads
   `graph.ic_probs` — note that the EXISTING feedback in `methods.base` does,
   through its reverse-reachable residual gains, which is why that estimate is
   deliberately not reused here.
2. **Every query is counted, and counted as what it is.** `ProbeCost` reads the
   bound environment's own meters, so when the arm's evaluator IS the trusted
   simulator (condition 4) the same probe correctly reports simulator episodes
   instead of world-model rollouts. The diagnostics do not decide which
   evaluator they are attached to and never pretend a call was free.
3. **Paired evaluation by default.** A drop is a DIFFERENCE of two rollouts, and
   at the ensemble sizes this loop runs the difference is smaller than either
   term's Monte Carlo error. Both rollouts therefore run at the same seed, which
   on `WorldModelEnvironment` means the same per-step uniform draws — a common
   random number, not two independent estimates. `paired=False` switches to the
   batched striped path, which is one call for K plans but NOT matched; it is
   there for bulk work where the ranking, not the difference, is wanted.

The module produces evidence and stops. It never says which node to seed: the
closing `diagnosis` line of `explain_plan` states what the numbers show ("the
weakest region is C2 at 12% while two seeds overlap at 0.71 in C0") and leaves
the revision to the agent.
"""

import time
from dataclasses import dataclass, field, replace
from functools import partial

from coding_agent.credit import (
    batched_environment,
    batched_plan_rewards,
    planned_action,
)
from coding_agent.regions import (
    Regions,
    basin_regions,
    basin_scheme,
    betweenness_percentile,
    bridge_nodes,
    build_regions,
    community_labels,
    community_scheme,
)
from coding_agent.types import ActionOp, GraphInfo

#: Predicted activation at or above which a node counts as covered. Shared with
#: `methods.base.reached_threshold` so "reached" means one thing across the two
#: feedback paths.
reached_threshold = 0.50
#: ...and below which it counts as missed.
unreached_threshold = 0.10

#: Normalised regional coverage below this is reported as WEAK. It is the same
#: number as `unreached_threshold` on purpose: a region is weak exactly when its
#: average member looks unreached.
weak_region_coverage = 0.10

#: Fuzzy-overlap at or above which two seeds are called redundant. 0.5 means at
#: least half of the smaller seed's predicted influence mass is also claimed by
#: the larger one. Reported as a NUMBER everywhere; the threshold only decides
#: whether the word "high" appears.
high_overlap = 0.50

#: Default stagnation window and tolerance for
#:     max_{j in [t-w, t]} V_j  -  V_{t-w}  <  epsilon
#: `w = 5` matches the "best over the last 5 iterations" the brief writes, and
#: epsilon is in reward units (nodes of spread). Both are constructor arguments;
#: these are the documented defaults, not hard-coded law.
stagnation_window = 5
stagnation_epsilon = 0.5

#: Regions listed in `summarize_regions` text, largest first.
max_listed_regions = 8
#: Bridge nodes listed in the bridge report.
max_listed_bridges = 5
#: Seed pairs listed in the redundancy report.
max_listed_pairs = 5


@dataclass(frozen=True)
class ProbeCost:
    """
    What one probe consumed, split by the distinction the experiment turns on.

    `simulator_episodes` is non-zero only when the bound evaluator is the trusted
    simulator; on the world-model arm it is 0 by construction and the cost lands
    in `wm_rollouts` / `wm_forward_passes`. `postprocessing` marks work that
    consumed no evaluator query at all (region sums off marginals already paid
    for, betweenness, stagnation arithmetic).
    """

    wm_rollouts: int = 0
    wm_forward_passes: int = 0
    simulator_episodes: int = 0
    seconds: float = 0.0
    postprocessing: bool = False

    def __add__(self, other: "ProbeCost") -> "ProbeCost":
        return ProbeCost(
            wm_rollouts=self.wm_rollouts + other.wm_rollouts,
            wm_forward_passes=self.wm_forward_passes + other.wm_forward_passes,
            simulator_episodes=self.simulator_episodes + other.simulator_episodes,
            seconds=self.seconds + other.seconds,
            postprocessing=self.postprocessing and other.postprocessing,
        )

    def to_dict(self) -> dict:
        return {
            "wm_rollouts": self.wm_rollouts,
            "wm_forward_passes": self.wm_forward_passes,
            "simulator_episodes": self.simulator_episodes,
            "seconds": round(self.seconds, 4),
            "postprocessing_only": self.postprocessing,
        }


@dataclass(frozen=True)
class PlanValue:
    """`V(S)` and the per-node marginals it was computed from."""

    reward: float
    reward_se: float
    marginals: list[float]
    seeds: list[int]
    cost: ProbeCost


@dataclass(frozen=True)
class RegionCoverage:
    key: object
    name: str
    size: int
    #: `C(R | S) = sum_{v in R} p_v`, in nodes.
    mass: float
    #: `Cbar(R | S) = C / |R|`, in [0, 1].
    normalized: float
    seeds_inside: int

    @property
    def weak(self) -> bool:
        return self.normalized < weak_region_coverage


@dataclass(frozen=True)
class Diagnosis:
    """One `explain_plan` result: the blocks, their text, and their total cost."""

    value: PlanValue
    contributions: dict[int, float] = field(default_factory=dict)
    coverage: list[RegionCoverage] = field(default_factory=list)
    overlaps: dict[tuple[int, int], float] = field(default_factory=dict)
    dominant_region: dict[int, object] = field(default_factory=dict)
    bridges: dict = field(default_factory=dict)
    stagnation: dict | None = None
    text: str = ""
    cost: ProbeCost = ProbeCost()

    def to_dict(self) -> dict:
        return {
            "reward": self.value.reward,
            "contributions": {str(node): value for node, value in self.contributions.items()},
            "coverage": [
                {
                    "region": str(entry.key),
                    "name": entry.name,
                    "size": entry.size,
                    "mass": round(entry.mass, 3),
                    "normalized": round(entry.normalized, 4),
                    "seeds_inside": entry.seeds_inside,
                    "weak": entry.weak,
                }
                for entry in self.coverage
            ],
            "overlaps": {f"{a}-{b}": round(v, 4) for (a, b), v in self.overlaps.items()},
            "dominant_region": {
                str(node): str(region) for node, region in self.dominant_region.items()
            },
            "bridges": self.bridges,
            "stagnation": self.stagnation,
            "cost": self.cost.to_dict(),
        }


def plan_seeds(plan: list[list[ActionOp]], budget_op: str = "add_node") -> list[int]:
    """The nodes a plan commits budget to, in first-scheduled order."""
    seen, seeds = set(), []

    for bag in plan:
        for action in bag:
            if action.op == budget_op and int(action.target) not in seen:
                seen.add(int(action.target))
                seeds.append(int(action.target))

    return seeds


def drop_node(plan: list[list[ActionOp]], node: int) -> list[list[ActionOp]]:
    """The plan with every action touching `node` removed — including edge ops,
    so a blocked removal's deletion bag leaves with its node."""
    return [
        [
            action
            for action in bag
            if int(action.target) != node
            and (action.destination is None or int(action.destination) != node)
        ]
        for bag in plan
    ]


def swap_node(plan: list[list[ActionOp]], old: int, new: int) -> list[list[ActionOp]]:
    return [
        [replace(action, target=new) if int(action.target) == old else action for action in bag]
        for bag in plan
    ]


def _scheduled_at(plan: list[list[ActionOp]], node: int) -> int:
    """First timestep the node is acted on, or -1 if it is not in the plan."""
    for timestep, bag in enumerate(plan):
        if any(int(action.target) == node for action in bag):
            return timestep

    return -1


def solo_plan(plan: list[list[ActionOp]], node: int) -> list[list[ActionOp]]:
    """`{node}` alone, scheduled in the bag it originally occupied."""
    return [
        [action for action in bag if int(action.target) == node] for bag in plan
    ]


class PlanDiagnostics:
    """
    Probes over one (evaluator, graph, task) triple.

    `environment` is whatever the arm is running on — the world model, the
    oracle, the Monte Carlo simulator, or the native single-episode evaluator.
    Nothing here assumes it is the world model; the cost accounting reports what
    it actually was.
    """

    def __init__(
        self,
        environment: object,
        graph: GraphInfo,
        horizon: int,
        budget: int,
        seed: int | None = None,
        budget_op: str = "add_node",
        # `maximize` (influence maximization) or `minimize` (containment,
        # blocking, immunization). It decides the SIGN CONVENTION of every
        # contribution: a positive number always means "this action helps the
        # objective", so a containment reviser reading "+18.4" is not being told
        # its removal made the outbreak bigger.
        sense: str = "maximize",
        region_scheme: str = community_scheme,
        paired: bool = True,
        stagnation_window: int = stagnation_window,
        stagnation_epsilon: float = stagnation_epsilon,
    ) -> None:
        self.environment = environment
        self.graph = graph
        self.horizon = horizon
        self.budget = budget
        self.seed = seed
        self.budget_op = budget_op

        if sense not in ("maximize", "minimize"):
            raise ValueError(f"sense must be maximize or minimize, got {sense!r}")

        self.sense = sense
        #: +1 when more reward is better, -1 when less is. Multiplies every
        #: reported difference so "positive = good" holds under both.
        self._orientation = 1.0 if sense == "maximize" else -1.0
        self.region_scheme = region_scheme
        # Common random numbers: both terms of a difference roll at one seed
        self.paired = paired
        self.stagnation_window = int(stagnation_window)
        self.stagnation_epsilon = float(stagnation_epsilon)

        # Running totals across every probe this object has answered, so a run's
        # results JSON can report the diagnostics' whole overhead in one place
        self.total_cost = ProbeCost(postprocessing=True)
        # Keyed by (node, the timestep the node is seeded at) rather than by node
        # alone: P(v | {s}) is independent of the OTHER seeds, so it survives a
        # plan edit, but not of WHEN s fires — a seed scheduled at t=3 has three
        # fewer steps to spread. Correct across plans, which is what lets a
        # refinement loop keep one PlanDiagnostics.
        self._solo_cache: dict[tuple[int, int], list[float]] = {}
        # Structure-only partitions are plan-independent and cached outright;
        # `seed_basin` is not, so it is keyed by the seed set it was built from
        self._regions: Regions | None = None
        self._basin_cache: dict[tuple[int, ...], Regions] = {}

    def reset_plan_cache(self) -> None:
        """
        Drop the per-plan caches after the incumbent plan changes.

        Only the `seed_basin` partition is genuinely plan-dependent; the solo
        profiles are keyed by (node, timestep) and survive a plan edit. Exposed
        anyway so a refinement loop has a supported way to start clean rather
        than reaching into the attributes.
        """
        self._basin_cache.clear()

    # -- accounting ---------------------------------------------------------

    def _meters(self) -> tuple[int, int, int]:
        return (
            int(getattr(self.environment, "rollout_calls", 0)),
            int(getattr(self.environment, "forward_passes", 0)),
            int(getattr(self.environment, "episodes_used", 0)),
        )

    def _charge(self, before: tuple[int, int, int], start: float) -> ProbeCost:
        after = self._meters()
        cost = ProbeCost(
            wm_rollouts=after[0] - before[0],
            wm_forward_passes=after[1] - before[1],
            simulator_episodes=after[2] - before[2],
            seconds=time.perf_counter() - start,
            postprocessing=False,
        )
        self.total_cost = self.total_cost + cost

        return cost

    def _free(self, start: float) -> ProbeCost:
        """A block that queried no evaluator: arithmetic on what was already paid."""
        cost = ProbeCost(seconds=time.perf_counter() - start, postprocessing=True)
        self.total_cost = self.total_cost + cost

        return cost

    # -- rollouts -----------------------------------------------------------

    def _rollout(self, plan: list[list[ActionOp]]):
        return self.environment.rollout(
            partial(planned_action, plan), self.horizon, self.budget, seed=self.seed
        )

    def _rewards(self, plans: list[list[list[ActionOp]]]) -> list[float]:
        """
        Rewards for several plans.

        Paired (the default) means one rollout each at the SHARED seed, so any
        two of them are matched realization-for-realization. The batched path
        stripes the plans across one call's sample blocks: cheaper per plan, but
        the blocks consume different columns of the same draw matrix, so the
        difference of two of them carries the full ensemble noise of both.
        """
        if self.paired or not batched_environment(self.environment):
            return [float(self._rollout(plan).reward) for plan in plans]

        rewards, _ = batched_plan_rewards(
            self.environment, plans, self.horizon, self.budget, self.seed
        )

        return [float(reward) for reward in rewards]

    # -- probes -------------------------------------------------------------

    def evaluate(self, plan: list[list[ActionOp]]) -> PlanValue:
        """`V(S)` plus the per-node marginals the regional probes aggregate."""
        start, before = time.perf_counter(), self._meters()
        trajectory = self._rollout(plan)

        return PlanValue(
            reward=float(trajectory.reward),
            reward_se=float(trajectory.cost.get("reward_se", 0.0)),
            marginals=list(trajectory.final_marginals or []),
            seeds=plan_seeds(plan, self.budget_op),
            cost=self._charge(before, start),
        )

    def probe_drop(self, plan: list[list[ActionOp]], node: int) -> dict:
        """`Delta_i = V(S) - V(S \\ {v_i})`, paired."""
        start, before = time.perf_counter(), self._meters()
        base, dropped = self._rewards([plan, drop_node(plan, node)])

        return {
            "op": "drop",
            "node": int(node),
            "plan": base,
            "without": dropped,
            # Signed so that positive always means "keeping this action helps"
            "contribution": self._orientation * (base - dropped),
            "cost": self._charge(before, start).to_dict(),
        }

    def probe_swap(self, plan: list[list[ActionOp]], old: int, new: int) -> dict:
        """`Delta = V(S - old + new) - V(S)`, paired."""
        start, before = time.perf_counter(), self._meters()
        base, swapped = self._rewards([plan, swap_node(plan, old, new)])

        return {
            "op": "swap",
            "old": int(old),
            "new": int(new),
            "plan": base,
            "swapped": swapped,
            # ...and positive always means "the swap is an improvement"
            "delta": self._orientation * (swapped - base),
            "cost": self._charge(before, start).to_dict(),
        }

    def probe_region(
        self, plan: list[list[ActionOp]], nodes: list[int], value: PlanValue | None = None
    ) -> dict:
        """
        `C(R | S) = sum_{v in R} p_v` and `Cbar = C / |R|`.

        Costs ONE evaluator call if the caller has no `PlanValue` in hand and
        zero if it does — the marginals are already there, and every region in
        `summarize_regions` is scored off the same single vector.
        """
        start, before = time.perf_counter(), self._meters()
        value = value or self.evaluate(plan)
        # Out-of-range ids are dropped rather than raising: the node set can come
        # from a generated script, and one bad id must not lose the whole probe
        members = [
            int(node) for node in nodes if 0 <= int(node) < self.graph.num_nodes
        ]
        mass = float(sum(value.marginals[node] for node in members)) if members else 0.0

        return {
            "op": "region",
            "size": len(members),
            "mass": mass,
            "normalized": mass / len(members) if members else 0.0,
            "cost": self._charge(before, start).to_dict(),
        }

    def regions(self, plan: list[list[ActionOp]] | None = None) -> Regions:
        """The partition regional feedback is aggregated over, cached per graph."""
        if self.region_scheme == basin_scheme:
            if plan is None:
                raise ValueError("the seed_basin partition needs a plan to build from")

            seeds = plan_seeds(plan, self.budget_op)
            key = tuple(seeds)

            if key not in self._basin_cache:
                self._basin_cache[key] = basin_regions(
                    seeds,
                    {seed: self._solo_marginals(plan, seed) for seed in seeds},
                    self.graph.num_nodes,
                )

            return self._basin_cache[key]

        if self._regions is None:
            self._regions = build_regions(self.graph, self.region_scheme)

        return self._regions

    def summarize_regions(
        self, plan: list[list[ActionOp]], value: PlanValue | None = None
    ) -> list[RegionCoverage]:
        """Regional coverage for the whole partition, largest region first."""
        start = time.perf_counter()
        value = value or self.evaluate(plan)
        regions = self.regions(plan)
        labels = regions.labels
        seeds = value.seeds

        seeds_per_region: dict[object, int] = {}
        for seed in seeds:
            key = labels.get(seed)
            seeds_per_region[key] = seeds_per_region.get(key, 0) + 1

        coverage = []
        for key in regions.ranked():
            members = regions.members[key]
            mass = float(sum(value.marginals[node] for node in members))
            coverage.append(
                RegionCoverage(
                    key=key,
                    name=regions.names.get(key, str(key)),
                    size=len(members),
                    mass=mass,
                    normalized=mass / len(members) if members else 0.0,
                    seeds_inside=seeds_per_region.get(key, 0),
                )
            )

        self._free(start)

        return coverage

    # -- redundancy ---------------------------------------------------------

    def _solo_marginals(self, plan: list[list[ActionOp]], node: int) -> list[float]:
        """`P(v influenced | {node})`, one rollout, cached per (node, timestep)."""
        key = (int(node), _scheduled_at(plan, node))

        if key in self._solo_cache:
            return self._solo_cache[key]

        start, before = time.perf_counter(), self._meters()
        trajectory = self._rollout(solo_plan(plan, node))
        self._charge(before, start)
        self._solo_cache[key] = list(trajectory.final_marginals or [])

        return self._solo_cache[key]

    def redundancy(self, plan: list[list[ActionOp]]) -> dict:
        """
        Pairwise seed overlap from single-seed influence profiles.

        For seeds i and j with solo marginal vectors p^i, p^j:

            overlap(i, j) = sum_v min(p^i_v, p^j_v) / min(sum_v p^i_v, sum_v p^j_v)

        the fuzzy-set overlap coefficient. 1 means the smaller seed's whole
        predicted influence mass is already claimed by the larger one; 0 means
        they reach disjoint parts of the graph. Deliberately the simplest
        quantity that answers "are these two seeds buying the same thing" — no
        Shapley value, no attribution method.

        Costs one world-model rollout per seed (cached), and reports it.
        """
        start = time.perf_counter()
        seeds = plan_seeds(plan, self.budget_op)
        profiles = {seed: self._solo_marginals(plan, seed) for seed in seeds}
        masses = {seed: sum(profile) for seed, profile in profiles.items()}
        labels = community_labels(self.graph)

        overlaps: dict[tuple[int, int], float] = {}
        for index, first in enumerate(seeds):
            for second in seeds[index + 1 :]:
                floor = min(masses[first], masses[second])
                shared = sum(
                    min(a, b) for a, b in zip(profiles[first], profiles[second], strict=True)
                )
                overlaps[(first, second)] = float(shared / floor) if floor > 0 else 0.0

        # Where each seed's own influence mass actually lands, which is what makes
        # "these two overlap" readable as "both are working the same region"
        dominant: dict[int, object] = {}
        for seed in seeds:
            per_region: dict[object, float] = {}
            for node, probability in enumerate(profiles[seed]):
                key = labels.get(node, -1)
                per_region[key] = per_region.get(key, 0.0) + probability

            dominant[seed] = (
                max(per_region, key=lambda key: per_region[key]) if per_region else None
            )

        self._free(start)

        return {
            "overlaps": overlaps,
            "dominant_region": dominant,
            "solo_mass": masses,
            "definition": (
                "overlap(i,j) = sum_v min(p_v|i, p_v|j) / min(mass_i, mass_j), "
                "from one single-seed world-model rollout per seed"
            ),
        }

    # -- bridges ------------------------------------------------------------

    def bridge_report(
        self, plan: list[list[ActionOp]], value: PlanValue | None = None
    ) -> dict:
        """
        Predicted activation on the graph's inter-community bridge nodes.

        Structure (betweenness, inter-community incidence) comes from the graph;
        activation comes from marginals already paid for. Zero extra evaluator
        calls. It is a DIAGNOSTIC, not an oracle: a bridge with low predicted
        activation is a place to look, not a node that is known to be worth
        seeding.
        """
        start = time.perf_counter()
        value = value or self.evaluate(plan)
        labels = community_labels(self.graph)
        candidates = bridge_nodes(self.graph)

        covered, missed = [], []
        for node in candidates:
            probability = value.marginals[node] if value.marginals else 0.0
            neighbours = {
                labels.get(other, -1)
                for other in set(self.graph.out_neighbors(node)) | set(self.graph.in_neighbors(node))
            }
            entry = {
                "node": int(node),
                "p_influenced": round(float(probability), 3),
                "betweenness_percentile": round(
                    betweenness_percentile(self.graph, node), 1
                ),
                "own_community": labels.get(node, -1),
                "connects": sorted(key for key in neighbours if key != labels.get(node, -1)),
            }
            (covered if probability >= reached_threshold else missed).append(entry)

        self._free(start)

        return {
            "n_bridges": len(candidates),
            "n_covered": len(covered),
            "covered": covered,
            "missed": sorted(missed, key=lambda entry: -entry["betweenness_percentile"]),
            "definition": (
                "a bridge node has >= 1 inter-community incident edge and sits at "
                "or above the 90th betweenness percentile"
            ),
        }

    # -- stagnation ---------------------------------------------------------

    def stagnation(self, history: list[float], sense: str = "maximize") -> dict:
        """
        Is the search still moving?

            stagnating  <=>  max_{j in [t-w, t]} V_j  -  V_{t-w}  <  epsilon

        Exactly the brief's definition, with `w` and `epsilon` the constructor's
        (defaults 5 and 0.5 nodes, both documented at the top of this module).
        Under `minimize` the same test runs on negated rewards, so "improvement"
        keeps meaning "the objective got better".

        Pure arithmetic on rewards the loop already had: no evaluator call.
        """
        start = time.perf_counter()
        values = [
            -float(value) if sense == "minimize" else float(value)
            for value in history
            if value is not None
        ]
        window = self.stagnation_window

        if len(values) < window + 1:
            self._free(start)

            return {
                "status": "WARMING_UP",
                "iterations": len(values),
                "window": window,
                "epsilon": self.stagnation_epsilon,
                "note": f"needs {window + 1} scored iterations to judge",
            }

        reference = values[-(window + 1)]
        best = max(values[-(window + 1) :])
        improvement = best - reference

        self._free(start)

        return {
            "status": "STAGNATING" if improvement < self.stagnation_epsilon else "IMPROVING",
            "iterations": len(values),
            "window": window,
            "epsilon": self.stagnation_epsilon,
            "current": values[-1] if sense == "maximize" else -values[-1],
            "best_in_window": best if sense == "maximize" else -best,
            "improvement": improvement,
        }

    def persistent_weakness(
        self, coverage_history: list[list[RegionCoverage]]
    ) -> list[dict]:
        """
        Regions that stayed weak across the whole stagnation window.

        Reported beside the stagnation verdict because "no longer improving" and
        "and here is the part of the graph nothing has touched" are the two
        halves of the same message. Deterministic post-processing of coverage
        already computed.
        """
        window = coverage_history[-self.stagnation_window :]

        if len(window) < self.stagnation_window:
            return []

        weak_every_turn: dict[object, list[float]] = {}
        for index, snapshot in enumerate(window):
            weak_now = {entry.key: entry.normalized for entry in snapshot if entry.weak}

            if index == 0:
                weak_every_turn = {key: [value] for key, value in weak_now.items()}
                continue

            weak_every_turn = {
                key: values + [weak_now[key]]
                for key, values in weak_every_turn.items()
                if key in weak_now
            }

        return [
            {
                "region": str(key),
                "turns": len(values),
                "max_coverage": round(max(values), 4),
            }
            for key, values in sorted(weak_every_turn.items(), key=lambda item: str(item[0]))
        ]

    # -- composition --------------------------------------------------------

    def explain_plan(
        self,
        plan: list[list[ActionOp]],
        history: list[float] | None = None,
        sense: str | None = None,
        blocks: tuple = ("value", "drop", "region", "redundancy", "bridge", "stagnation"),
        coverage_history: list | None = None,
    ) -> Diagnosis:
        """
        Every requested block, once, sharing one `PlanValue` and one solo cache.

        `blocks` is what makes the feedback-tier experiment a controlled one: F0
        is `("value",)`, F1 adds `"drop"`, F2 adds `"region"`, F3 is the default.
        The cost of each tier is then exactly the cost of the blocks it asked
        for, with nothing computed and discarded.
        """
        sense = sense or self.sense
        value = self.evaluate(plan)
        seeds = value.seeds
        label = (
            "Expected spread" if self.sense == "maximize" else "Expected final size"
        )
        direction = "more is better" if self.sense == "maximize" else "LOWER IS BETTER"
        lines = [
            f"{label} = {value.reward:.2f} (±{value.reward_se:.2f} SE, {direction})"
        ]

        contributions: dict[int, float] = {}
        if "drop" in blocks and seeds:
            start, before = time.perf_counter(), self._meters()

            if self.paired:
                # `value.reward` came from a rollout at this same seed, so it is
                # already the matched base and re-rolling it would be one wasted
                # query per turn
                base = value.reward
                rewards = self._rewards([drop_node(plan, seed) for seed in seeds])
            else:
                # Striped mode gives each plan its own sample blocks, so a base
                # taken from the earlier paired rollout would be measured against
                # a different realization. The base rides along in the batch.
                rewards = self._rewards(
                    [plan] + [drop_node(plan, seed) for seed in seeds]
                )
                base, rewards = rewards[0], rewards[1:]

            self._charge(before, start)
            contributions = {
                seed: self._orientation * (base - reward)
                for seed, reward in zip(seeds, rewards, strict=True)
            }
            lines.append("")
            lines.append(
                f"Contribution per {self.budget_op} target (paired rollouts; "
                f"positive = keeping it helps the objective):"
            )
            for seed in sorted(contributions, key=lambda node: -contributions[node]):
                lines.append(f"  node_{seed}  {contributions[seed]:+.2f}")

        coverage: list[RegionCoverage] = []
        if "region" in blocks:
            coverage = self.summarize_regions(plan, value)
            regions = self.regions(plan)
            lines.append("")
            lines.append(
                f"Regional coverage ({regions.scheme}; "
                f"C(R|S) = sum of P(influenced) over members):"
            )
            weakest = min(coverage, key=lambda entry: entry.normalized) if coverage else None
            for entry in coverage[:max_listed_regions]:
                mark = (
                    "  <-- weakest"
                    if weakest is not None and entry.key == weakest.key
                    else ("  weakly covered" if entry.weak else "")
                )
                plural = "" if entry.seeds_inside == 1 else "s"
                lines.append(
                    f"  {entry.name}: {entry.mass:.1f} / {entry.size} "
                    f"({entry.normalized:.0%}), {entry.seeds_inside} seed{plural}{mark}"
                )

        overlaps: dict[tuple[int, int], float] = {}
        dominant: dict[int, object] = {}
        if "redundancy" in blocks and len(seeds) > 1:
            report = self.redundancy(plan)
            overlaps = report["overlaps"]
            dominant = report["dominant_region"]
            ranked = sorted(overlaps, key=lambda pair: -overlaps[pair])[:max_listed_pairs]
            lines.append("")
            lines.append(
                "Seed overlap (fuzzy-set overlap of single-seed influence "
                "profiles; 1 = the same reach):"
            )
            for first, second in ranked:
                label = " HIGH" if overlaps[(first, second)] >= high_overlap else ""
                lines.append(
                    f"  node_{first} / node_{second}: {overlaps[(first, second)]:.2f}"
                    f" (dominant regions {dominant.get(first)} / {dominant.get(second)})"
                    f"{label}"
                )

        bridges: dict = {}
        if "bridge" in blocks:
            bridges = self.bridge_report(plan, value)
            lines.append("")
            lines.append(
                f"Bridge coverage: {bridges['n_covered']} / {bridges['n_bridges']} "
                f"bridge nodes have P(influenced) >= {reached_threshold:.0%}"
            )
            for entry in bridges["missed"][:max_listed_bridges]:
                lines.append(
                    f"  missed bridge node_{entry['node']}: "
                    f"betweenness percentile {entry['betweenness_percentile']:.1f}%, "
                    f"P(influenced)={entry['p_influenced']:.2f}, "
                    f"connects C{entry['own_community']} -> "
                    f"{[f'C{key}' for key in entry['connects']]}"
                )

        stagnation_block = None
        if "stagnation" in blocks and history:
            stagnation_block = self.stagnation(history, sense)
            lines.append("")
            lines.append(
                f"Search status: {stagnation_block['status']} "
                f"(window {stagnation_block['window']}, "
                f"epsilon {stagnation_block['epsilon']:.2f})"
            )
            if "improvement" in stagnation_block:
                lines.append(
                    f"  best in window {stagnation_block['best_in_window']:.2f}, "
                    f"improvement {stagnation_block['improvement']:+.2f}"
                )

            # "no longer improving" and "and here is what nothing has touched"
            # are the two halves of the same message
            if coverage_history:
                persistent = self.persistent_weakness(
                    list(coverage_history) + ([coverage] if coverage else [])
                )
                stagnation_block["persistent_weakness"] = persistent

                for entry in persistent:
                    lines.append(
                        f"  persistent weakness: region {entry['region']} stayed "
                        f"below {weak_region_coverage:.0%} coverage for all "
                        f"{entry['turns']} turns of the window "
                        f"(best {entry['max_coverage']:.0%})"
                    )

        diagnosis = _diagnose(contributions, coverage, overlaps, dominant, bridges)
        if diagnosis:
            lines.append("")
            lines.append("What the numbers show:")
            lines.append(f"  {diagnosis}")

        return Diagnosis(
            value=value,
            contributions=contributions,
            coverage=coverage,
            overlaps=overlaps,
            dominant_region=dominant,
            bridges=bridges,
            stagnation=stagnation_block,
            text="\n".join(lines),
            cost=self.total_cost,
        )


def _diagnose(
    contributions: dict[int, float],
    coverage: list[RegionCoverage],
    overlaps: dict[tuple[int, int], float],
    dominant: dict[int, object],
    bridges: dict,
) -> str:
    """
    One sentence, entirely derived from the numbers above it.

    Deliberately NOT a recommendation. It names what the evidence is — a weak
    region, a redundant pair, a missed bridge — and stops; choosing the revision
    is the agent's job, and a world model that proposes seed sets is no longer
    supplying evidence, it is competing with the algorithm under test.
    """
    findings = []

    weak = [entry for entry in coverage if entry.weak]
    if weak:
        worst = min(weak, key=lambda entry: entry.normalized)
        findings.append(
            f"{worst.name} is at {worst.normalized:.0%} coverage over "
            f"{worst.size} nodes with {worst.seeds_inside} seeds"
        )

    redundant = [pair for pair, value in overlaps.items() if value >= high_overlap]
    if redundant:
        first, second = max(redundant, key=lambda pair: overlaps[pair])
        findings.append(
            f"node_{first} and node_{second} overlap at "
            f"{overlaps[(first, second)]:.2f} (both dominant in "
            f"{dominant.get(first)})"
        )

    if contributions:
        weakest_seed = min(contributions, key=lambda node: contributions[node])
        if contributions[weakest_seed] < 1.0:
            findings.append(
                f"node_{weakest_seed} contributes only "
                f"{contributions[weakest_seed]:+.2f} nodes"
            )

    if bridges and bridges.get("missed"):
        entry = bridges["missed"][0]
        findings.append(
            f"the highest-betweenness uncovered bridge is node_{entry['node']} "
            f"at P={entry['p_influenced']:.2f}"
        )

    return "; ".join(findings)


__all__ = [
    "Diagnosis",
    "PlanDiagnostics",
    "PlanValue",
    "ProbeCost",
    "RegionCoverage",
    "drop_node",
    "plan_seeds",
    "solo_plan",
    "swap_node",
]
