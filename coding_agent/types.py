"""
Core contracts for the coding-agent outer loop.

Reuses the canonical Action/State value types from the data simulator so the
strategies, environments, and the trained world model all speak the same action vocabulary.
"""

from dataclasses import dataclass, field
from typing import Callable, Protocol
import numpy as np

from data.wm_simulator import ActionOp, State

Action = ActionOp

# action_fn(state, timestep) -> action bag for that timestep
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

    @classmethod
    def from_store_entry(cls, entry: dict) -> "GraphInfo":
        """Build from a world_model.wm_data.load_graph_store entry."""
        return cls(
            num_nodes=int(entry["num_nodes"]),
            edge_index=np.asarray(entry["edge_index"], dtype=np.int64),
            ic_probs=np.asarray(entry["ic_probs"], dtype=np.float32),
            directed=bool(entry.get("meta", {}).get("directed", False)),
        )

    def _ensure_adjacency(self) -> None:
        if self._out_adjacency is not None:
            return

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
    task: str = "influence_maximization"
    objective: str = "maximize_final_spread"
    diffusion_model: str = "IC"  # "IC" or "LT"
    budget: int = 5
    horizon: int = 10


@dataclass
class Trajectory:
    states: list[State]
    actions: list[list[ActionOp]]
    reward: float  # final spread (infected count)
    infected_counts: list[float]
    cost: dict = field(default_factory=dict)


class Strategy(Protocol):
    """
    Contract the agent's generated script must implement.

    A script may implement only the method its outer-loop method needs; the
    executor validates the required one is present.
    """

    def plan_horizon(
        self, graph: GraphInfo, budget: int, horizon: int
    ) -> list[list[ActionOp]]:  # Method 1
        ...

    def act(
        self, state: State, graph: GraphInfo, timestep: int
    ) -> list[ActionOp]:  # Methods 2 & 3
        ...
