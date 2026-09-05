"""
Shared fixtures: a tiny synthetic dataset written to disk in the exact on-disk
format `data/generate_wm_data.py` produces.

Built by hand rather than by calling the generator, so the tests do not depend on
NDlib, do not take seconds, and — the point — fail if the READER's contract with
the writer drifts. The schema mirrored here is the one documented in
`data/README.md`; if the generator changes, this fixture is the thing that should
be updated to match, and the tests around it say what that change broke.
"""

import json
from pathlib import Path
import numpy as np
import pytest

# A 6-node path with a chord, small enough to reason about by hand:
#   0 - 1 - 2 - 3 - 4 - 5   plus  1 - 4
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 4)]
num_nodes = 6
ic_probability = 0.5


def _both_directions(pairs: list[tuple]) -> list[tuple]:
    return pairs + [(destination, source) for source, destination in pairs]


@pytest.fixture(scope="session")
def arcs() -> list[tuple]:
    return _both_directions(edges)


@pytest.fixture(scope="session")
def edge_index(arcs: list[tuple]) -> np.ndarray:
    return np.array(arcs, dtype=np.int64).T


@pytest.fixture(scope="session")
def base_edges(arcs: list[tuple]) -> dict:
    return {arc: ic_probability for arc in arcs}


def _marginals(infected: list[int], frontier: list[int]) -> dict:
    return {
        "next_marginal_infected": {str(node): 1.0 for node in infected},
        "next_marginal_frontier": {str(node): 1.0 for node in frontier},
    }


def _record(
    t: int,
    branch: str,
    state_infected: list[int],
    state_frontier: list[int],
    action: list[dict],
    next_infected: list[int],
    next_frontier: list[int],
    episode_id: str = "ep0",
    graph_id: str = "g0",
) -> dict:
    return {
        "graph_id": graph_id,
        "diffusion_model": "IC",
        "episode_id": episode_id,
        "algorithm": "degree",
        "branch": branch,
        "t": t,
        "state": {
            "infected": sorted(state_infected),
            "frontier": sorted(state_frontier),
            "infected_count": len(state_infected),
            "frontier_count": len(state_frontier),
        },
        "action": action,
        "next_state": {
            "infected": sorted(next_infected),
            "frontier": sorted(next_frontier),
            "infected_count": len(next_infected),
            "frontier_count": len(next_frontier),
        },
        "reward": float(len(next_infected) - len(state_infected)),
        **_marginals(next_infected, next_frontier),
    }


def _records() -> list[dict]:
    """One main branch of 3 steps, with two counterfactual forks off step 1.

    The forks are what makes action-conditioning testable at all: same state, two
    different actions, each with its own recorded outcome.
    """
    return [
        # t=0: seed node 1
        _record(
            0, "main", [], [], [{"op": "add_node", "target": 1}], [1], [1]
        ),
        # t=1: no action, cascade spreads from 1
        _record(1, "main", [1], [1], [], [1, 2, 4], [2, 4]),
        # t=1 forks: same state, different actions
        _record(1, "cf_0", [1], [1], [{"op": "add_node", "target": 5}], [1, 4, 5], [4, 5]),
        _record(1, "cf_1", [1], [1], [{"op": "remove_node", "target": 1}], [1], []),
        # t=2: an edge op, so the adjacency reconstruction has something to replay
        _record(
            2,
            "main",
            [1, 2, 4],
            [2, 4],
            [{"op": "remove_edge", "target": 4, "destination": 5}],
            [1, 2, 3, 4],
            [3],
        ),
    ]


@pytest.fixture
def dataset_dir(tmp_path: Path, edge_index: np.ndarray) -> Path:
    """A results/<...>/data directory the loaders accept."""
    out_dir = tmp_path / "data"
    (out_dir / "graphs").mkdir(parents=True)

    np.savez_compressed(
        out_dir / "graphs" / "g0.npz",
        edge_index=edge_index,
        ic_probs=np.full(edge_index.shape[1], ic_probability, dtype=np.float32),
        lt_weights=np.ones(edge_index.shape[1], dtype=np.float32),
        node_feats=np.zeros((num_nodes, 1), dtype=np.float32),
        node_labels=np.zeros(num_nodes, dtype=np.int64),
    )
    (out_dir / "graphs_index.json").write_text(
        json.dumps(
            [
                {
                    "graph_id": "g0",
                    "file": "g0.npz",
                    "n_nodes": num_nodes,
                    "n_edges": len(edges),
                    "directed": False,
                }
            ]
        )
    )

    records = _records()
    for split in ("train", "val", "test"):
        (out_dir / f"transitions_IC_{split}.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records) + "\n"
        )

    (out_dir / "metadata.json").write_text(
        json.dumps({"config": {"remove_semantics": "spent"}})
    )

    return out_dir
