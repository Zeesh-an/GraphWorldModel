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

# action_fn(state, t) -> action bag for timestep t
ActionFn = Callable[[State, int], list[ActionOp]]


@dataclass
class GraphInfo:
    """Read-only graph view handed to the library and to strategies."""

    num_nodes: int
    edge_index: np.ndarray  # (2, E) int64 [src, dst]
    ic_probs: np.ndarray  # (E,) float32 per-edge IC transmission prob
    directed: bool
    _out: dict[int, list[int]] | None = field(default=None, repr=False)
    _in: dict[int, list[int]] | None = field(default=None, repr=False)
    _deg: np.ndarray | None = field(default=None, repr=False)

    @classmethod
    def from_store_entry(cls, entry: dict) -> "GraphInfo":
        """Build from a world_model.wm_data.load_graph_store entry."""
        return cls(
            num_nodes=int(entry["num_nodes"]),
            edge_index=np.asarray(entry["edge_index"], dtype=np.int64),
            ic_probs=np.asarray(entry["ic_probs"], dtype=np.float32),
            directed=bool(entry.get("meta", {}).get("directed", False)),
        )

    def _ensure_adj(self) -> None:
        if self._out is not None:
            return

        out: dict[int, list[int]] = {v: [] for v in range(self.num_nodes)}
        inn: dict[int, list[int]] = {v: [] for v in range(self.num_nodes)}

        for i in range(self.edge_index.shape[1]):
            u, v = int(self.edge_index[0, i]), int(self.edge_index[1, i])
            out[u].append(v)
            inn[v].append(u)

        self._out, self._in = out, inn

    def out_neighbors(self, v: int) -> list[int]:
        self._ensure_adj()
        assert self._out is not None
        return self._out[int(v)]

    def in_neighbors(self, v: int) -> list[int]:
        self._ensure_adj()
        assert self._in is not None
        return self._in[int(v)]

    def degree(self, v: int) -> int:
        """Total degree (in + out) of node v."""
        if self._deg is None:
            deg = np.zeros(self.num_nodes, dtype=np.int64)
            np.add.at(deg, self.edge_index[0], 1)
            np.add.at(deg, self.edge_index[1], 1)
            self._deg = deg

        return int(self._deg[int(v)])


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
        self, g: GraphInfo, budget: int, horizon: int
    ) -> list[list[ActionOp]]:  # Method 1
        ...

    def act(
        self, state: State, g: GraphInfo, t: int
    ) -> list[ActionOp]:  # Methods 2 & 3
        ...
