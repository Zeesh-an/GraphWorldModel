"""
Core contracts for the coding-agent outer loop.

Reuses the canonical ActionOp/State value types from the data simulator so the
strategies, environments, and the trained world model all speak the same action vocabulary.
"""

from dataclasses import dataclass, field
from typing import Callable, Protocol
import numpy as np

from data.wm_simulator import ActionOp, State, valid_action_ops

# ActionFn is the interface between strategies and environments (every environment's rollout() consumes one of these; every method produces one):
# ActionFn is a function mapping (current state, timestep) -> action bag for that timestep
ActionFn = Callable[[State, int], list[ActionOp]]


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


class ScoredStrategy:
    """
    Scored-mode contract: plan_horizon is a fixed greedy harness the agent cannot
    override — generated code may only override score() and/or schedule(). This
    forces edits to the algorithm's internals instead of free-form programs or
    composition over the library.
    """

    def score(self, node: int, selected: tuple, graph: GraphInfo) -> float:
        return float(graph.degree(node))

    def schedule(
        self, seeds: list[int], graph: GraphInfo, horizon: int
    ) -> list[list[ActionOp]]:
        return [[ActionOp("add_node", node) for node in seeds]] + [
            [] for _ in range(horizon)
        ]

    def plan_horizon(
        self, graph: GraphInfo, budget: int, horizon: int
    ) -> list[list[ActionOp]]:
        selected = []
        for _ in range(min(budget, graph.num_nodes)):
            best_node, best_score = -1, float("-inf")
            for node in range(graph.num_nodes):
                if node in selected:
                    continue

                node_score = float(self.score(node, tuple(selected), graph))
                if node_score > best_score:
                    best_node, best_score = node, node_score

            selected.append(best_node)

        # Normalize whatever schedule() returns to exactly horizon+1 bags
        plan = [list(bag) for bag in self.schedule(selected, graph, horizon)]
        plan = plan[: horizon + 1]
        plan += [[] for _ in range(horizon + 1 - len(plan))]

        return plan
