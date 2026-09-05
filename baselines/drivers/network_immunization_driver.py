"""
Drive `allogn/Network-Immunization`'s five solvers over our graph and outbreak.

`GWM_NETIMM_SOLVER` picks which one runs: `NetShield` (Tong ICDM'10), `Dom` (DAVA,
Zhang & Prakash SDM'14), `NetShape` (Khalil KDD'14's convex hazard-matrix program),
`Degree` or `Random`. All five subclass one `Solver` base with the same
`(G, seeds, k, **params)` constructor and the same `log['Blocked nodes']` output, so
one driver serves all of them and the five rows are produced under identical
conditions.

Warning: THIS REPO IS THE ONLY PUBLIC NETSHIELD AND THE ONLY PUBLIC DAVA. Both
papers shipped no code: `research/epidemic_control.md` §11 lists them among the
five methods with "no public release", and §3.2's widely-repeated claim that
EpiLearn ships a NetShield implementation is WRONG: that repo's tree has no shield,
immunization or intervention code at all [derived, 2026-08-05]. So these arms are
not a convenience, they are the only way a NetShield or DAVA number in our table is
anyone's but ours, and they cross-check
`coding_agent/tools/immunization_algorithms.py`'s reimplementations of both.

Run inside the clone with our work directory as argv[1]. Reads `graph.txt` and
`outbreak.json` (written by `registry._netimm_export`) and writes `blocked.json`.

WHY THIS EXISTS RATHER THAN `run_solver.py`. The shipped entry point ties three
things together that we need apart: it unpickles a NetworkX graph (a pickle written
by our networkx and read by theirs is a version gamble for no gain), and it then
runs 100 of its OWN Independent Cascade simulations to score the answer, which is
work we do not want, because the whole point of the harness is that every arm's
seeds are scored by ONE referee. Building the graph from a plain edge list and
stopping after `solver.run()` skips both.

THREE COMPATIBILITY PATCHES are applied at setup rather than here, because they
edit the repo's own files; `registry.external_baselines["netimm_netshield"].patches`
carries them and each is a 2016-era API that no longer exists:

  * `nx.to_numpy_matrix` was REMOVED in networkx 3.0 -> `nx.to_numpy_array`.
  * `scipy.linalg.eigh(..., eigvals=(n-1, n-1))` was removed in SciPy 1.12 ->
    `subset_by_index=[n-1, n-1]`.
  * `np.warnings` was removed in NumPy 1.24.

The one thing this driver does fix in-process is the RECURSION LIMIT: `DomSolver`'s
`traverseTreeRec` walks the dominator tree recursively, and on a path-like graph
that tree is as deep as the graph. Python's default 1000 frames is not enough for a
10,000-node line, and the failure is a bare RecursionError with no hint of the
cause.
"""

import json
import os
import sys
import networkx as nx

# The five solvers, and which of the repo's classes each one is
solvers = {
    "NetShield": "NetShieldSolver",
    "Dom": "DomSolver",
    "NetShape": "NetShapeSolver",
    "Degree": "DegreeSolver",
    "Random": "RandomSolver",
}

# NetShape's `epsilon`: it runs `ceil((R / epsilon)^2)` iterations, each a DENSE
# eigendecomposition of the N x N hazard matrix, so this is the knob that decides
# whether the arm finishes at all. 1.0 is ~50 iterations at k=50 and is where the
# quality/time knee sits at our sizes; the paper sweeps it.
default_epsilon = 1.0

# `traverseTreeRec` recurses once per dominator-tree level. A path graph's tree is
# N deep, so the default 1000 is not a bound on the algorithm, only on Python.
recursion_headroom = 50_000


def build_graph(path: str, num_nodes: int) -> nx.DiGraph:
    """
    `u v w` per line -> the DiGraph the solvers expect, with `weight` on every arc.

    DIRECTED even for an undirected graph, and not a style choice:
    `DomSolver.build_domtree` calls `nx.algorithms.dominance.immediate_dominators`,
    which is defined on a DiGraph and raises on anything else. Our export writes
    both orientations of an undirected edge, so the dominance is computed on the
    same reachability an undirected graph has.

    `weight` is the per-arc transmission probability and is what DAVA's superseed
    construction and its `-log p` shortest paths both consume; NetShield and Degree
    ignore it (NetShield builds its adjacency with `weight=None`).
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))

    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                continue

            weight = float(parts[2]) if len(parts) > 2 else 1.0
            graph.add_edge(int(parts[0]), int(parts[1]), weight=weight)

    # `graph_id` is what the repo's own Generator stamps and its Readme asks for;
    # no solver reads it, but setting it keeps a repo-shaped object repo-shaped
    graph.graph["graph_id"] = 0

    return graph


if __name__ == "__main__":
    work_dir = sys.argv[1]
    repo_dir = os.getcwd()
    sys.path.insert(0, repo_dir)
    sys.setrecursionlimit(recursion_headroom)

    name = os.environ.get("GWM_NETIMM_SOLVER", "NetShield")
    if name not in solvers:
        raise SystemExit(f"unknown solver {name!r}; choose one of {sorted(solvers)}")

    payload = json.load(open(os.path.join(work_dir, "outbreak.json")))
    num_nodes = int(payload["num_nodes"])
    outbreak = [int(node) for node in payload["outbreak"]]
    budget = int(payload["budget"])

    graph = build_graph(os.path.join(work_dir, "graph.txt"), num_nodes)

    module = __import__(solvers[name])
    solver_class = getattr(module, solvers[name])

    # NetShape reads `self.params['epsilon']` in `clear()` and raises KeyError
    # without it; every other solver ignores extra params
    params = (
        {"epsilon": float(os.environ.get("GWM_NETIMM_EPSILON", default_epsilon))}
        if name == "NetShape"
        else {}
    )
    # `Dom` is DAVA and `Dom` with `fast=True` is DAVA-fast: the paper's own
    # near-linear variant, which builds ONE dominator tree instead of k
    if os.environ.get("GWM_NETIMM_FAST") == "1":
        params["fast"] = True

    print(
        f"[netimm] {name} on {num_nodes} nodes / {graph.number_of_edges()} arcs, "
        f"k={budget}, {len(outbreak)} index case(s), params={params}",
        flush=True,
    )

    solver = solver_class(graph, outbreak, budget, **params)
    solver.run()

    blocked = [int(node) for node in solver.log["Blocked nodes"]]
    # DAVA merges the outbreak into a SUPERSEED whose id is `len(G)` or higher, and
    # its `get_rank` fallback can hand that id back on a trivial instance. Dropping
    # it here rather than in the parser keeps the range check in
    # `run_baseline.run_external_baseline` meaningful: an out-of-range id there
    # means the repo was run on a different graph, which is a real error.
    blocked = [node for node in blocked if 0 <= node < num_nodes]

    with open(os.path.join(work_dir, "blocked.json"), "w") as handle:
        json.dump({"blocked": blocked, "log": {
            key: value for key, value in solver.log.items()
            if isinstance(value, (int, float, str, list))
        }}, handle)

    print(f"[netimm] {name} blocked {len(blocked)} nodes", flush=True)
