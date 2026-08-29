"""
NDlib-backed stepwise IC/LT diffusion simulator with mid-rollout action
injection, plus the shared State / ActionOp value types.

advance(bag) applies the action bag (mutating model.status) then runs one
diffusion iteration, returning s_{t + 1} = T_endo(T_exo(s_t, a_t)).

`remove_semantics` picks what remove_node means: `spent` (the IM reading, the
default) or `blocked` (the containment reading). See the constants below.
"""

from dataclasses import dataclass, field

import future.utils
import networkx as nx
import numpy as np
import ndlib.models.ModelConfig as model_config_module
import ndlib.models.epidemics as epidemics  # IC, LT

valid_action_ops = (
    "add_node",
    "remove_node",
    "add_edge",
    "remove_edge",
    "set_edge_weight",
)
# What generation injects unless a task or a flag says otherwise: the two node
# ops. The three edge ops stay available and are opted into per task
# (influence blocking and epidemic control) or by passing --action-ops.
default_action_ops = ("add_node", "remove_node")

# The three COMPARTMENTAL dynamics, which `data/wm_epidemic.py` simulates rather
# than NDlib. Named here because `--diffusion-model` is one flag across every task
# and half the pipeline has to ask "is this one of the epidemic ones" without
# importing the epidemic module (research/epidemic_control.md §2.1).
epidemic_dynamics = ("SIR", "SIS", "SEIR")

# Dynamics whose edges carry a real per-arc transmission probability, so
# `GraphInput.edge_weight` must be the true w rather than ones. LT is the only
# structural dynamics that does not, and it is what this set exists to exclude,
# the epidemic ones DO, which is the whole reason §2.2 says to write our own
# stepper instead of using NDlib's scalar-beta SIR/SIS/SEIR.
weighted_dynamics = ("IC",) + epidemic_dynamics

# What `remove_node` means. The two readings are genuinely different problems and
# the wrong one silently biases every number:
#
#   spent    the spreader is used up. IC status 2 (Removed) STAYS counted as
#            infected; LT status 0 lets the node re-cross its threshold later.
#            Correct for influence maximization, where removing an active node
#            models "this node has already spent its shot".
#
#   blocked  the node is deleted from the graph: not counted, cannot transmit,
#            cannot be (re-)infected. Correct for every containment task
#            (critical node detection, influence blocking, immunization), where
#            `spent` inflates the measured final spread by exactly +k because
#            active_nodes() counts each immunized node as infected.
spent = "spent"
blocked = "blocked"
valid_remove_semantics = (spent, blocked)


@dataclass
class ActionOp:
    op: str  # valid_action_ops
    target: int  # Target node ID (edge source u for edge operations)
    destination: int | None = None  # Edge destination v (edge operations only)
    weight: float | None = None  # Edge weight (add_edge / set_edge_weight)

    def to_dict(self) -> dict:
        action_dict = {"op": self.op, "target": int(self.target)}

        if self.destination is not None:
            action_dict["destination"] = int(self.destination)

        if self.weight is not None:
            action_dict["weight"] = float(self.weight)

        return action_dict


@dataclass
class State:
    """
    Diffusion state s_t as sorted node-id lists.

    Single-cascade tasks use `infected` / `frontier` and nothing else. A COMPETITIVE
    task (influence blocking) carries a second cascade in `pos_infected` /
    `pos_frontier`, and the mapping is deliberate rather than symmetric:

      * `infected` / `frontier` are always the **negative** cascade: the rumour, the
        thing being minimized. Every existing reader (the reward, `spread_curve`,
        `summarize`, `credit`, the plots) therefore measures the objective without
        a single change.
      * `pos_infected` / `pos_frontier` are the blocker's counter-cascade, which is
        an INSTRUMENT and never the score.

    The positive keys are omitted from `to_dict` when both are empty, so a
    single-cascade JSONL is byte-identical to what it was before competition existed.

    A COMPARTMENTAL task (epidemic control) carries two more sets, and the mapping
    is again deliberate rather than symmetric
    (research/epidemic_control.md §2.3):

      * `infected` is EVER-INFECTED: the attack set, the thing being minimized. It
        is monotone under SIR, SIS and SEIR alike (a node never un-becomes
        ever-infected), which is what lets `reward = len(state.infected)`, the
        spread curve, the plots and the summary go on reading it with no branch.
      * `frontier` is the currently INFECTIOUS set `I`. That is exactly what it
        already means under IC (status-1 spreaders), and it is the set that drives
        transmission: it is also the one that is NOT monotone, because `I -> R`
        under SIR/SEIR and `I -> S` under SIS both shrink it.
      * `exposed` is `E` (SEIR only) and `recovered` is `R` (SIR/SEIR). Susceptible
        is everything else: a node in none of the four is `S`, which under SIS is
        how a recovered-to-susceptible node reads while staying ever-infected.

    Both are omitted from `to_dict` when empty, so an IC/LT JSONL is unchanged.
    """

    infected: list[int]
    frontier: list[int]
    pos_infected: list[int] = field(default_factory=list)
    pos_frontier: list[int] = field(default_factory=list)
    exposed: list[int] = field(default_factory=list)
    recovered: list[int] = field(default_factory=list)
    # Which ensemble member (or Monte Carlo run) this state belongs to. Never
    # serialized: it exists so an adaptive policy that keeps state across act()
    # calls can be given one copy per possible world. The world-model rollout
    # advances its samples in lockstep and queried ONE policy object with all 50
    # of them interleaved; a policy tracking "what I already chose" then excluded
    # every other sample's picks, and every sample ended up committing a different
    # seed set (netscience k=318: oracle 506 in the loop, 724 on the referee).
    sample: int = 0

    def to_dict(self) -> dict:
        infected = sorted(int(node) for node in self.infected)
        frontier = sorted(int(node) for node in self.frontier)
        state = {
            "infected": infected,
            "frontier": frontier,
            "infected_count": len(infected),
            "frontier_count": len(frontier),
        }

        if self.pos_infected or self.pos_frontier:
            positive = sorted(int(node) for node in self.pos_infected)
            positive_frontier = sorted(int(node) for node in self.pos_frontier)
            state |= {
                "pos_infected": positive,
                "pos_frontier": positive_frontier,
                "pos_infected_count": len(positive),
                "pos_frontier_count": len(positive_frontier),
            }

        if self.exposed or self.recovered:
            exposed = sorted(int(node) for node in self.exposed)
            recovered = sorted(int(node) for node in self.recovered)
            state |= {
                "exposed": exposed,
                "recovered": recovered,
                "exposed_count": len(exposed),
                "recovered_count": len(recovered),
            }

        return state


class TracedICModel(epidemics.IndependentCascadesModel):
    """
    IC that records WHICH `u` caused each successful flip.

    NDlib never produces the transmission edge: its `iteration` sets
    `actual_status[v] = 1` on a successful coin flip without recording the
    responsible `u`, so every tree-level cascade-reconstruction metric is
    unscoreable without this (research/cascade_reconstruction.md §9 item 9). The
    body below is NDlib's own, line for line, plus one append.

    Two properties of the ground truth it produces, both of which must be
    documented wherever a tree number is:

      * NDlib iterates spreaders in NODE ORDER and skips any `v` already flipped
        this step, so the recorded parent is *the first successful `u` in node
        order*, not a uniformly random one among the successes. That is a real
        and documentable bias, not an artifact of this subclass.
      * A node can be reached by several spreaders in one step; only the first
        one is credited, because only the first one actually changed the state.

    `build` rather than a constructor: NDlib's own `__init__` calls
    `super(self.__class__, self).__init__`, which recurses forever the moment the
    class is subclassed. Building the base and re-classing the instance is a
    well-defined operation and avoids copying its constructor body.
    """

    @classmethod
    def build(cls, graph, seed=None) -> "TracedICModel":
        model = epidemics.IndependentCascadesModel(graph, seed=seed)
        model.__class__ = cls
        model.transmissions = []

        return model

    def iteration(self, node_status=True):
        self.clean_initial_status(list(self.available_statuses.values()))
        actual_status = {
            node: nstatus for node, nstatus in future.utils.iteritems(self.status)
        }
        self.transmissions = []

        if self.actual_iteration == 0:
            self.actual_iteration += 1
            delta, node_count, status_delta = self.status_delta(actual_status)

            return {
                "iteration": 0,
                "status": actual_status.copy() if node_status else {},
                "node_count": node_count.copy(),
                "status_delta": status_delta.copy(),
            }

        for u in self.graph.nodes:
            if self.status[u] != 1:
                continue

            neighbors = list(self.graph.neighbors(u))

            if len(neighbors) > 0:
                threshold = 1.0 / len(neighbors)

                for v in neighbors:
                    if actual_status[v] == 0:
                        key = (u, v)

                        if "threshold" in self.params["edges"]:
                            if key in self.params["edges"]["threshold"]:
                                threshold = self.params["edges"]["threshold"][key]
                            elif (
                                v,
                                u,
                            ) in self.params["edges"]["threshold"] and not self.graph.directed:
                                threshold = self.params["edges"]["threshold"][(v, u)]

                        if np.random.random_sample() <= threshold:
                            actual_status[v] = 1
                            self.transmissions.append((int(u), int(v)))

            actual_status[u] = 2

        delta, node_count, status_delta = self.status_delta(actual_status)
        self.status = actual_status
        self.actual_iteration += 1

        return {
            "iteration": self.actual_iteration - 1,
            "status": delta.copy() if node_status else {},
            "node_count": node_count.copy(),
            "status_delta": status_delta.copy(),
        }


class TracedThresholdModel(epidemics.ThresholdModel):
    """
    LT that records the active in-neighbourhood each newly activated node crossed on.

    LT has NO transmission edge: activation is a threshold crossing over the whole
    active neighbourhood, so the honest ground truth is a parent SET rather than a
    parent (research/cascade_reconstruction.md §2.6). Every tree metric that wants
    a single edge therefore reads LT as set-valued and says so.

    Same `build` trick as `TracedICModel`, for the same NDlib constructor bug.
    """

    @classmethod
    def build(cls, graph, seed=None) -> "TracedThresholdModel":
        model = epidemics.ThresholdModel(graph, seed=seed)
        model.__class__ = cls
        model.transmissions = []

        return model

    def iteration(self, node_status=True):
        self.clean_initial_status(list(self.available_statuses.values()))
        actual_status = {
            node: nstatus for node, nstatus in future.utils.iteritems(self.status)
        }
        self.transmissions = []

        if self.actual_iteration == 0:
            self.actual_iteration += 1
            delta, node_count, status_delta = self.status_delta(actual_status)

            return {
                "iteration": 0,
                "status": actual_status.copy() if node_status else {},
                "node_count": node_count.copy(),
                "status_delta": status_delta.copy(),
            }

        for u in self.graph.nodes:
            if actual_status[u] == 1:
                continue

            neighbors = list(self.graph.neighbors(u))
            if self.graph.directed:
                neighbors = list(self.graph.predecessors(u))

            infected = 0
            for v in neighbors:
                infected += self.status[v]

            if len(neighbors) > 0:
                infected_ratio = float(infected) / len(neighbors)
                if infected_ratio >= self.params["nodes"]["threshold"][u]:
                    actual_status[u] = 1
                    # The whole active in-neighbourhood is the cause; naming one
                    # of them would invent a transmission edge LT does not have
                    self.transmissions += [
                        (int(v), int(u)) for v in neighbors if self.status[v] == 1
                    ]

        delta, node_count, status_delta = self.status_delta(actual_status)
        self.status = actual_status
        self.actual_iteration += 1

        return {
            "iteration": self.actual_iteration - 1,
            "status": delta.copy() if node_status else {},
            "node_count": node_count.copy(),
            "status_delta": status_delta.copy(),
        }


class Simulator:
    def __init__(
        self,
        graph: nx.Graph | nx.DiGraph,
        ic_prob_map: dict | None = None,
        seed: int = 0,
        remove_semantics: str = spent,
        trace_parents: bool = False,
    ) -> None:
        if remove_semantics not in valid_remove_semantics:
            raise ValueError(
                f"unknown remove_semantics {remove_semantics!r}; "
                f"choose one of {valid_remove_semantics}"
            )

        self.graph = graph
        self.ic_prob_map = ic_prob_map
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.remove_semantics = remove_semantics
        # Record who infected whom. Off by default because it is only meaningful
        # to cascade reconstruction and costs a list append per successful flip.
        self.trace_parents = trace_parents
        self.model_name = None
        self.model = None
        # Nodes deleted from the graph under `blocked`; empty under `spent`
        self.blocked = set()
        # {v: [u, ...]} for the transitions of the most recent advance(); IC
        # records exactly one parent per newly infected node, LT records the set
        self.last_parents = {}

    def reset(
        self, model_name: str, lt_thresholds: dict[int, float] | None = None
    ) -> None:
        self.model_name = model_name
        self.blocked = set()
        self.last_parents = {}
        config = model_config_module.Configuration()

        if model_name == "IC":
            # Independent Cascade (IC)
            # One directed arc per ic_prob_map entry: configuring both (u,v) and
            # (v,u) on an undirected graph makes NDlib drop ALL edge thresholds.
            ic_graph = nx.DiGraph()
            ic_graph.add_nodes_from(self.graph.nodes())
            ic_graph.add_edges_from(self.ic_prob_map.keys())

            build = (
                TracedICModel.build
                if self.trace_parents
                else epidemics.IndependentCascadesModel
            )
            model = build(ic_graph, seed=self.seed)
            for (source, destination), probability in self.ic_prob_map.items():
                config.add_edge_configuration(
                    "threshold", (source, destination), float(probability)
                )
        else:
            # Linear Threshold (LT)
            # Copy so edge actions mutate this episode's graph, not the shared bundle
            build = (
                TracedThresholdModel.build
                if self.trace_parents
                else epidemics.ThresholdModel
            )
            model = build(self.graph.copy(), seed=self.seed)

            if lt_thresholds is None:
                lt_thresholds = {
                    int(node): float(self.rng.uniform(0.0, 1.0))
                    for node in self.graph.nodes()
                }

            for node in self.graph.nodes():
                config.add_node_configuration(
                    "threshold", int(node), float(lt_thresholds[int(node)])
                )

        config.add_model_initial_configuration("Infected", [])
        model.set_initial_status(config)
        self.model = model
        self.model.iteration()  # First iteration is a no-op (no diffusion)

    def active_nodes(self) -> set[int]:
        if self.model_name == "IC":
            # IC: 0 = Susceptible, 1 = Infected (currently infectious), 2 = Removed (already spread, spent)
            # Active: 1, 2
            active = {
                int(node)
                for node, status in self.model.status.items()
                if status in (1, 2)
            }
        else:
            # LT: 0 = Susceptible, 1 = Infected
            # Active: 1
            active = {
                int(node) for node, status in self.model.status.items() if status == 1
            }

        # A blocked node is no longer part of the graph, so it is not part of the
        # spread either. Returned unchanged under `spent`: the set difference
        # would rebuild the hash table and reorder iteration, which reorders the
        # marginal dicts written to the JSONL for no semantic gain.
        return active - self.blocked if self.blocked else active

    def current_state(self) -> State:
        # IC frontier = status-1 spreaders; LT's "newly flipped" delta is only available via advance()
        active = self.active_nodes()

        if self.model_name == "IC":
            frontier = {
                int(node) for node, status in self.model.status.items() if status == 1
            }
        else:
            frontier = set()

        return State(infected=sorted(active), frontier=sorted(frontier))

    def apply_actions(self, bag: list[ActionOp]) -> None:
        for action in bag:
            if action.op == "add_node":
                # A blocked node left the graph; seeding it would let a bare
                # remove_node's target transmit for one step before
                # _enforce_blocked re-zeroes it. Same rule as the other two simulators.
                if int(action.target) in self.blocked:
                    continue

                # Set status to 1 (the node becomes an active spreader, and next iteration it spreads)
                self.model.status[int(action.target)] = 1
            elif action.op == "remove_node":
                # Take it out of the frontier
                # For IC, status 2 (Removed): it stays counted as infected but won't spread again
                # For LT (no Removed state), status 0 (Susceptible)
                self.model.status[int(action.target)] = (
                    2 if self.model_name == "IC" else 0
                )

                # Under `blocked` the node also leaves the graph: active_nodes()
                # stops counting it, and the incident remove_edge ops that the
                # deletion bag carries alongside stop it transmitting
                if self.remove_semantics == blocked:
                    self.blocked.add(int(action.target))
            elif action.op == "add_edge":
                # New edge u -> v: diffusion can traverse it from the next iteration onwards
                source, destination = int(action.target), int(action.destination)
                self.model.graph.add_edges(source, [destination])

                if self.model_name == "IC":
                    # IC needs a per-edge transmission probability (the edge threshold)
                    self.model.params["edges"]["threshold"][
                        (source, destination)
                    ] = float(action.weight)
            elif action.op == "remove_edge":
                # Drop the edge u -> v so diffusion can no longer traverse it
                source, destination = int(action.target), int(action.destination)
                self.model.graph.remove_edges(source, [destination])

                if self.model_name == "IC":
                    self.model.params["edges"]["threshold"].pop(
                        (source, destination), None
                    )
            elif action.op == "set_edge_weight":
                # Perturb the edge's transmission probability (IC only because LT ignores edge weights)
                if self.model_name == "IC":
                    source, destination = int(action.target), int(action.destination)
                    self.model.params["edges"]["threshold"][
                        (source, destination)
                    ] = float(action.weight)

    def _enforce_blocked(self) -> None:
        """
        Hold blocked nodes out of the dynamics after a diffusion step.

        delete_node_bag() normally strips the incident edges too, which is what
        makes a block structural. This is the guard for a caller that emits a bare
        remove_node: without it an LT node that was blocked but not isolated
        re-crosses its threshold and starts feeding its neighbours again. NDlib
        computes each iteration from a pre-step status snapshot, so re-zeroing
        afterwards is enough to keep it out of every subsequent step.
        """
        for node in self.blocked:
            self.model.status[node] = 2 if self.model_name == "IC" else 0

    def _record_parents(self, bag: list[ActionOp]) -> None:
        """
        {v: [u, ...]} for the step that just ran, from the traced model's log.

        A node the ACTION activated has no parent at all: an `add_node` is an
        exogenous injection, not a transmission, so those are recorded with an
        empty list, which is what marks a source in the reconstructed tree.
        """
        if not self.trace_parents:
            return

        parents = {}
        for source, target in getattr(self.model, "transmissions", []):
            if target not in self.blocked and source not in self.blocked:
                parents.setdefault(target, []).append(source)

        for action in bag:
            if action.op == "add_node":
                parents[int(action.target)] = []

        self.last_parents = parents

    def advance(self, bag: list[ActionOp]) -> State:
        # s_{t + 1} = T_endo(T_exo(s_t, a_t))
        # Snapshot previous active nodes before actions
        previous_active = self.active_nodes()

        self.apply_actions(bag)  # Apply the actions (exogenous effect)
        self.model.iteration()  # Run one diffusion iteration (endogenous diffusion dynamics)
        self._enforce_blocked()
        self._record_parents(bag)

        active = self.active_nodes()

        if self.model_name == "IC":
            # IC frontier = status = 1 nodes
            frontier = {
                int(node) for node, status in self.model.status.items() if status == 1
            }
        else:
            # LT frontier = active - previous_active (the nodes that newly flipped this step)
            frontier = active - previous_active

        return State(infected=sorted(active), frontier=sorted(frontier))

    def advance_marginal(
        self, bag: list[ActionOp], num_mc: int
    ) -> tuple[State, dict, dict]:
        # Estimate the one-step marginals by Monte Carlo - s_{t + 1} = T_endo(T_exo(s_t, a_t))
        if num_mc < 1:
            raise ValueError(f"num_mc must be >= 1, got {num_mc}")

        # T_exo (action) is deterministic, so it is applied once
        previous_active = self.active_nodes()
        self.apply_actions(bag)  # edge/graph mutations persist across draws
        post_action = self.snapshot()  # status after the action, before diffusion

        # IC is stochastic, while LT is deterministic, so only 1 run is needed for LT
        draws = num_mc if self.model_name == "IC" else 1
        infected_counts = {}
        frontier_counts = {}
        last_state = None

        # T_endo (diffusion iteration) is stochastic (for IC), so it is sampled using Monte Carlo simulations from the post-action state
        for _ in range(draws):
            # Each iteration is an independent stochastic draw from the same starting state

            # Restores status only for a fresh stochastic draw
            self.restore(post_action)
            self.model.iteration()
            self._enforce_blocked()

            active = self.active_nodes()
            if self.model_name == "IC":
                frontier = {
                    int(node)
                    for node, status in self.model.status.items()
                    if status == 1
                }
            else:
                frontier = active - previous_active

            for node in active:
                infected_counts[node] = infected_counts.get(node, 0) + 1

            for node in frontier:
                frontier_counts[node] = frontier_counts.get(node, 0) + 1

            last_state = State(infected=sorted(active), frontier=sorted(frontier))
            # The LAST draw is the one whose state is returned, so the parents
            # recorded here are the parents of the state actually written
            self._record_parents(bag)

        # Averaging across Monte Carlo runs turns the target into the true probability
        infected_marginal = {
            int(node): count / draws for node, count in infected_counts.items()
        }
        frontier_marginal = {
            int(node): count / draws for node, count in frontier_counts.items()
        }

        return last_state, infected_marginal, frontier_marginal

    def revert_edges(self, bag: list[ActionOp]) -> None:
        """
        Undo the edge deletions in `bag`, restoring their original IC weights.

        `snapshot`/`restore` rewind status and the blocked set but NOT the graph,
        because the graph lives inside NDlib's model. That is fine for a fork whose
        bag is node-only, and wrong for a `blocked` removal, whose deletion bag
        strips the node's incident arcs permanently: the main branch would resume
        on a graph the fork edited.

        Weights come from `ic_prob_map`, the episode's own edge table, so a
        restored arc carries the probability it had rather than a guess. Only
        `remove_edge` is invertible here: reverting an `add_edge` or a
        `set_edge_weight` needs the pre-op weight, which nothing records, and
        silently skipping them would leave the graph subtly wrong instead of
        loudly wrong.
        """
        inverse = []

        for action in bag:
            if action.op == "remove_edge":
                source, destination = int(action.target), int(action.destination)
                weight = (self.ic_prob_map or {}).get((source, destination))

                if weight is not None:
                    inverse.append(
                        ActionOp("add_edge", source, destination, float(weight))
                    )
            elif action.op in ("add_edge", "set_edge_weight"):
                raise ValueError(
                    f"cannot revert {action.op!r}: the pre-op edge weight is not "
                    f"recorded anywhere. Only remove_edge is invertible, which is "
                    f"all a node-deletion bag emits."
                )

        self.apply_actions(inverse)

    def set_state(self, infected, frontier) -> None:
        """
        Force the model into an ARBITRARY mid-cascade state.

        Evaluating the transition kernel at a proposed state is what a trajectory
        decoder does thousands of times per instance
        (research/cascade_reconstruction.md §2.5.2), and a fresh simulation cannot
        reach one: NDlib only ever advances forward from what it already holds.
        `snapshot`/`restore` rewind to a state this simulator VISITED; this writes
        one it never did, which is the difference between replaying an episode and
        scoring a hypothesis about it.

        Under IC the frontier is status 1 (currently infectious) and the rest of
        the infected set is status 2 (spent), which is exactly the invariant
        `current_state` reads back. LT has no spent compartment, so everything
        active is status 1 and the frontier is carried by the caller.
        """
        active = {int(node) for node in infected} | {int(node) for node in frontier}
        wave = {int(node) for node in frontier}

        for node in self.model.status:
            if int(node) in wave:
                self.model.status[node] = 1
            elif int(node) in active:
                self.model.status[node] = 2 if self.model_name == "IC" else 1
            else:
                self.model.status[node] = 0

        self._enforce_blocked()

    def snapshot(self) -> tuple[dict, int, set]:
        # `blocked` belongs in here: a counterfactual fork that blocks a node
        # would otherwise leak that block back into the main branch on restore
        return (
            dict(self.model.status),
            int(self.model.actual_iteration),
            set(self.blocked),
        )

    def restore(self, snapshot: tuple[dict, int, set]) -> None:
        status, actual_iteration, blocked_nodes = snapshot
        self.model.status = dict(status)
        self.model.actual_iteration = actual_iteration
        self.blocked = set(blocked_nodes)
