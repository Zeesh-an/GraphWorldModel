"""
NDlib-backed stepwise IC/LT diffusion simulator with mid-rollout action
injection, plus the shared State / ActionOp value types.

advance(bag) applies the action bag (mutating model.status) then runs one
diffusion iteration, returning s_{t + 1} = T_endo(T_exo(s_t, a_t)).
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep  # IC, LT

VALID_ACTION_OPS = (
    "add_node",
    "remove_node",
    "add_edge",
    "remove_edge",
    "set_edge_weight",
)


@dataclass
class ActionOp:
    op: str  # VALID_ACTION_OPS
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
    """Diffusion state s_t = (infected, frontier) as sorted node-id lists"""

    infected: list[int]
    frontier: list[int]

    def to_dict(self) -> dict:
        infected = sorted(int(v) for v in self.infected)
        frontier = sorted(int(v) for v in self.frontier)

        return {
            "infected": infected,
            "frontier": frontier,
            "infected_count": len(infected),
            "frontier_count": len(frontier),
        }


class Simulator:
    def __init__(
        self,
        graph: nx.Graph | nx.DiGraph,
        ic_prob_map: dict | None = None,
        seed: int = 0,
    ) -> None:
        self.graph = graph
        self.ic_prob_map = ic_prob_map
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.model_name = None
        self.model = None

    def reset(
        self, model_name: str, lt_thresholds: dict[int, float] | None = None
    ) -> None:
        self.model_name = model_name
        config = mc.Configuration()

        if model_name == "IC":
            # Independent Cascade (IC)
            # One directed arc per ic_prob_map entry: configuring both (u,v) and
            # (v,u) on an undirected graph makes NDlib drop ALL edge thresholds.
            ic_graph = nx.DiGraph()
            ic_graph.add_nodes_from(self.graph.nodes())
            ic_graph.add_edges_from(self.ic_prob_map.keys())

            model = ep.IndependentCascadesModel(ic_graph, seed=self.seed)
            for (u, v), p in self.ic_prob_map.items():
                config.add_edge_configuration("threshold", (u, v), float(p))
        else:
            # Linear Threshold (LT)
            # Copy so edge actions mutate this episode's graph, not the shared bundle
            model = ep.ThresholdModel(self.graph.copy(), seed=self.seed)

            if lt_thresholds is None:
                lt_thresholds = {
                    int(n): float(self.rng.uniform(0.0, 1.0))
                    for n in self.graph.nodes()
                }

            for n in self.graph.nodes():
                config.add_node_configuration(
                    "threshold", int(n), float(lt_thresholds[int(n)])
                )

        config.add_model_initial_configuration("Infected", [])
        model.set_initial_status(config)
        self.model = model
        self.model.iteration()  # First iteration is a no-op (no diffusion)

    def _active_nodes(self) -> set[int]:
        if self.model_name == "IC":
            # IC: 0 = Susceptible, 1 = Infected (currently infectious), 2 = Removed (already spread, spent)
            # Active: 1, 2
            return {int(n) for n, s in self.model.status.items() if s in (1, 2)}
        else:
            # LT: 0 = Susceptible, 1 = Infected
            # Active: 1
            return {int(n) for n, s in self.model.status.items() if s == 1}

    def current_state(self) -> State:
        # IC frontier = status-1 spreaders; LT's "newly flipped" delta is only available via advance()
        active = self._active_nodes()

        if self.model_name == "IC":
            frontier = {int(n) for n, s in self.model.status.items() if s == 1}
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
            elif action.op == "add_edge":
                # New edge u -> v: diffusion can traverse it from the next iteration onwards
                u, v = int(action.target), int(action.destination)
                self.model.graph.add_edges(u, [v])

                if self.model_name == "IC":
                    # IC needs a per-edge transmission probability (the edge threshold)
                    self.model.params["edges"]["threshold"][(u, v)] = float(
                        action.weight
                    )
            elif action.op == "remove_edge":
                # Drop the edge u -> v so diffusion can no longer traverse it
                u, v = int(action.target), int(action.destination)
                self.model.graph.remove_edges(u, [v])

                if self.model_name == "IC":
                    self.model.params["edges"]["threshold"].pop((u, v), None)
            elif action.op == "set_edge_weight":
                # Perturb the edge's transmission probability (IC only because LT ignores edge weights)
                if self.model_name == "IC":
                    u, v = int(action.target), int(action.destination)
                    self.model.params["edges"]["threshold"][(u, v)] = float(
                        action.weight
                    )

    def advance(self, bag: list[ActionOp]) -> State:
        # s_{t + 1} = T_endo(T_exo(s_t, a_t))
        prev_active = (
            self._active_nodes()
        )  # Snapsot previous active nodes before actions

        self.apply_actions(bag)  # Apply the actions (exogenous effect)
        self.model.iteration()  # Run one diffusion iteration (endogenous diffusion dynamics)

        active = self._active_nodes()

        if self.model_name == "IC":
            # IC frontier = status = 1 nodes
            frontier = {int(n) for n, s in self.model.status.items() if s == 1}
        else:
            # LT frontier = active - prev_active (the nodes that newly flipped this step)
            frontier = active - prev_active

        return State(infected=sorted(active), frontier=sorted(frontier))

    def snapshot(self) -> tuple[dict, int]:
        return (dict(self.model.status), int(self.model.actual_iteration))

    def restore(self, snapshot: tuple[dict, int]) -> None:
        status, actual_iteration = snapshot
        self.model.status = dict(status)
        self.model.actual_iteration = actual_iteration
