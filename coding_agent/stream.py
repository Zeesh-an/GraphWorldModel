"""
Dynamic / streaming IM: exogenous graph edits arriving while the policy seeds.

The dynamic branch of the IM literature. The graph is not fixed: edges
appear and disappear as the campaign runs, and the seed set has to be chosen
against a moving target. The expected cost is "no new T_endo, no new
simulator", and that holds: `reconstruct_episode_adjacency` and `apply_edge_ops`
already replay a changing `A_t`. What was missing is a stream of edits
INDEPENDENT of the action, which is what this file supplies.

The edits ride in the action bag, which is why no environment changed: the MC
environment hands the bag to `Simulator.advance` (mutating the live NDlib graph)
and the world-model environment routes it through `apply_edge_ops` (mutating that
sample's edge dict). Both already do exactly the right thing with an edge op.

Everything here is a pure function of (base graph, seed, timestep). That is not
tidiness: `MonteCarloEnvironment` loops (episode, then timestep) while
`WorldModelEnvironment` loops (timestep, then sample), so a stream carrying
mutable state between calls would deliver different graphs under the two. A
deterministic schedule delivers the same edit at timestep t to every episode and
every sample, which is what "exogenous" means.

Two consequences worth stating before reading any number:

  * The stream is applied to EVERY arm, adaptive or not. Comparing an arm whose
    graph moved against one whose graph did not would measure the stream, not the
    method.
  * A static plan cannot react to it by construction, and an adaptive policy can,
    because it is re-queried each round and handed the current graph. That gap is
    the experiment, so do not read a static arm's loss under a stream as a bug.
"""

from dataclasses import dataclass, field
import numpy as np

from coding_agent.types import ActionFn, ActionOp, GraphInfo, State

# Tries per wanted insertion before giving up. A random pair collides with an
# existing edge often on a dense graph, and skipping instead of retrying is what
# turns a rewire into a thinning.
max_insert_attempts = 20


@dataclass()
class GraphEditStream:
    """A deterministic schedule of exogenous edge insertions and deletions."""

    base_edges: dict  # {(u, v): weight}
    num_nodes: int
    horizon: int
    # Edits per timestep as a fraction of |E|; 0.02 on a 5,000-edge graph is 100
    # insertions and 100 deletions per step
    rate: float = 0.02
    seed: int = 0
    directed: bool = False
    _edits: dict = field(default_factory=dict, repr=False)
    _edges: dict = field(default_factory=dict, repr=False)
    _graphs: dict = field(default_factory=dict, repr=False)

    @property
    def edits_per_step(self) -> int:
        return max(1, round(self.rate * len(self.base_edges)))

    def edits(self, timestep: int) -> list[ActionOp]:
        """
        The exogenous edits applied at `timestep`, as edge ops.

        Nothing fires at t=0: the policy chose its first seeds against the base
        graph, and moving the graph inside that same step would score it on a
        graph it never saw. The stream starts at t=1.

        Deletions and insertions are balanced so edge count stays roughly fixed
        and a spread change is attributable to REWIRING rather than to the graph
        quietly getting denser or sparser.
        """
        if timestep <= 0 or timestep > self.horizon:
            return []

        if timestep in self._edits:
            return self._edits[timestep]

        # Seeded per timestep, so edits(5) is the same list however many times and
        # in whatever order the environments ask for it
        rng = np.random.default_rng([self.seed, timestep])
        existing = set(self.edges_at(timestep))

        # The unit of work is an UNDIRECTED EDGE on an undirected graph, not an
        # arc. The edge store keeps both orientations there, so deleting one arc
        # per insertion of two would grow the graph by an arc a step; measured at
        # +23% over six steps before this was canonicalized.
        units = sorted(
            {
                (source, destination) if self.directed or source <= destination
                else (destination, source)
                for source, destination in existing
            }
        )
        count = min(self.edits_per_step, len(units))
        operations = []

        def both_ways(op: str, pair: tuple, weight: float | None = None) -> list:
            source, destination = pair
            ops = [ActionOp(op, source, destination, weight)]
            if not self.directed:
                ops.append(ActionOp(op, destination, source, weight))

            return ops

        for index in rng.choice(len(units), size=count, replace=False):
            operations += both_ways("remove_edge", units[int(index)])

        weights = list(self.base_edges.values())
        # Match the base graph's weight distribution rather than inventing a
        # constant: a new edge at p=1.0 would be a stronger intervention than
        # anything the policy can buy
        default_weight = float(np.mean(weights)) if weights else 1.0

        # Retry rather than skip on a collision, so insertions match deletions
        # one for one and the graph is rewired rather than thinned
        added = 0
        for _ in range(count * max_insert_attempts):
            if added == count:
                break

            source = int(rng.integers(self.num_nodes))
            destination = int(rng.integers(self.num_nodes))

            if source == destination or (source, destination) in existing:
                continue

            existing.add((source, destination))
            if not self.directed:
                existing.add((destination, source))

            operations += both_ways(
                "add_edge", (source, destination), default_weight
            )
            added += 1

        self._edits[timestep] = operations

        return operations

    def edges_at(self, timestep: int) -> dict:
        """The edge dict a policy called at `timestep` is looking at."""
        if timestep in self._edges:
            return self._edges[timestep]

        # Recursive on the previous step rather than replayed from 0 each time,
        # so a horizon-10 rollout builds each step's graph exactly once
        if timestep <= 0:
            edges = dict(self.base_edges)
        else:
            edges = dict(self.edges_at(timestep - 1))
            for action in self.edits(timestep - 1):
                if action.op == "remove_edge":
                    edges.pop((action.target, action.destination), None)
                else:
                    edges[(action.target, action.destination)] = float(action.weight)

        self._edges[timestep] = edges

        return edges

    def graph_at(self, timestep: int) -> GraphInfo:
        """The GraphInfo an adaptive policy should reason over at `timestep`."""
        if timestep in self._graphs:
            return self._graphs[timestep]

        edges = self.edges_at(timestep)
        pairs = sorted(edges)
        edge_index = (
            np.asarray(pairs, dtype=np.int64).T
            if pairs
            else np.zeros((2, 0), dtype=np.int64)
        )
        self._graphs[timestep] = GraphInfo(
            num_nodes=self.num_nodes,
            edge_index=edge_index,
            ic_probs=np.asarray(
                [edges[pair] for pair in pairs], dtype=np.float32
            ),
            directed=self.directed,
        )

        return self._graphs[timestep]

    def wrap(self, action_fn: ActionFn) -> ActionFn:
        """
        Append this timestep's edits to whatever the policy returned.

        Applied AFTER the policy's bag has been validated, deliberately: the
        stream's ops are exogenous, so they are not the policy's to be charged
        for and would otherwise be rejected on an `--allowed-ops add_node` task.
        """

        def streamed(state: State, timestep: int) -> list[ActionOp]:
            return list(action_fn(state, timestep)) + self.edits(timestep)

        return streamed


def build_stream(
    graph: GraphInfo, horizon: int, rate: float, seed: int
) -> GraphEditStream | None:
    """None when the rate is zero, which is the static-graph default."""
    if not rate:
        return None

    return GraphEditStream(
        base_edges={
            (int(graph.edge_index[0, edge]), int(graph.edge_index[1, edge])): float(
                graph.ic_probs[edge]
            )
            for edge in range(graph.edge_index.shape[1])
        },
        num_nodes=graph.num_nodes,
        horizon=horizon,
        rate=rate,
        seed=seed,
        directed=graph.directed,
    )
