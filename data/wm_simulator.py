"""
NDlib-backed stepwise IC/LT diffusion simulator with mid-rollout action
injection, plus the shared State / ActionOp value types.

advance(bag) applies the action bag (mutating model.status) then runs one
diffusion iteration, returning s_{t + 1} = T_endo(T_exo(s_t, a_t)).

`remove_semantics` picks what remove_node means: `spent` (the IM reading, the
default) or `blocked` (the containment reading). See the constants below.
"""

from dataclasses import dataclass, field

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

      * `infected` / `frontier` are always the **negative** cascade — the rumour, the
        thing being minimized. Every existing reader (the reward, `spread_curve`,
        `summarize`, `credit`, the plots) therefore measures the objective without
        a single change.
      * `pos_infected` / `pos_frontier` are the blocker's counter-cascade, which is
        an INSTRUMENT and never the score.

    The positive keys are omitted from `to_dict` when both are empty, so a
    single-cascade JSONL is byte-identical to what it was before competition existed.
    """

    infected: list[int]
    frontier: list[int]
    pos_infected: list[int] = field(default_factory=list)
    pos_frontier: list[int] = field(default_factory=list)

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

        return state


class Simulator:
    def __init__(
        self,
        graph: nx.Graph | nx.DiGraph,
        ic_prob_map: dict | None = None,
        seed: int = 0,
        remove_semantics: str = spent,
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
        self.model_name = None
        self.model = None
        # Nodes deleted from the graph under `blocked`; empty under `spent`
        self.blocked = set()

    def reset(
        self, model_name: str, lt_thresholds: dict[int, float] | None = None
    ) -> None:
        self.model_name = model_name
        self.blocked = set()
        config = model_config_module.Configuration()

        if model_name == "IC":
            # Independent Cascade (IC)
            # One directed arc per ic_prob_map entry: configuring both (u,v) and
            # (v,u) on an undirected graph makes NDlib drop ALL edge thresholds.
            ic_graph = nx.DiGraph()
            ic_graph.add_nodes_from(self.graph.nodes())
            ic_graph.add_edges_from(self.ic_prob_map.keys())

            model = epidemics.IndependentCascadesModel(ic_graph, seed=self.seed)
            for (source, destination), probability in self.ic_prob_map.items():
                config.add_edge_configuration(
                    "threshold", (source, destination), float(probability)
                )
        else:
            # Linear Threshold (LT)
            # Copy so edge actions mutate this episode's graph, not the shared bundle
            model = epidemics.ThresholdModel(self.graph.copy(), seed=self.seed)

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

    def advance(self, bag: list[ActionOp]) -> State:
        # s_{t + 1} = T_endo(T_exo(s_t, a_t))
        # Snapshot previous active nodes before actions
        previous_active = self.active_nodes()

        self.apply_actions(bag)  # Apply the actions (exogenous effect)
        self.model.iteration()  # Run one diffusion iteration (endogenous diffusion dynamics)
        self._enforce_blocked()

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
        strips the node's incident arcs permanently — the main branch would resume
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
