"""
Driver for the cosasi package: our episodes in, one source set per episode out.

Runs inside cosasi's OWN virtualenv as a subprocess, so nothing here may import
from the rest of this repo. It reads the two npz files written by
`registry._localization_export` and writes `predictions.json`.

cosasi (JOSS 2022) is the classical-estimator toolkit, and its value here is a
CROSS-CHECK rather than new coverage: `lpsi`, `netsleuth`, `jordan_center` and
`rumor_centrality` in `coding_agent/tools/localization_algorithms.py` are our own
reimplementations from the papers' prose, and rumor centrality in particular has
no first-party code anywhere. A second independent implementation disagreeing
loudly is the signal worth having.

Two things differ from the GraphSL driver, both in our favour:

  * **There is no training step.** Every estimator here is a pure function of
    `(infected subgraph, graph)`, so no labels touch this process at all and the
    selection/evaluation split is irrelevant to it.
  * **It returns a ranking over all N nodes**, not a thresholded set, so top-k is
    the natural read rather than a reinterpretation.

Warning: `cosasi.utils.estimators.source_subgraphs` calls scikit-learn's
SpectralClustering on `nx.adjacency_matrix(I)`, and modern scikit-learn rejects
the int64 sparse indices networkx now produces ("Only sparse matrices with 32-bit
integer indices are accepted"). Every multi-source estimator routes through it.
`_patch_int32_indices` below shims that one call rather than pinning an old
scikit-learn, because the pin would also have to hold torch and numpy back.

Environment:
    GWM_COSASI_METHOD    jordan | netsleuth | lisn | rumor_centrality
"""

# `cosasi` is installed in baselines/external/cosasi/.venv, NOT in ours: that
# isolation is the whole point of the external-baseline design, and here it is
# load-bearing rather than tidy: cosasi's own requirements pin numpy 1.21,
# scikit-learn 1.1 and networkx 2.8, which cannot coexist with this project's
# floors. An editor resolving this file against the project interpreter therefore
# reports every cosasi import as missing, and it is right to: this file never runs
# under that interpreter. Scoped to the file rather than set in
# .vscode/settings.json so it travels with the code and does not silence a genuine
# missing import anywhere else.
# pyright: reportMissingImports=false

import json
import os
import sys
from types import SimpleNamespace

import networkx as nx
import numpy as np

methods = ("jordan", "netsleuth", "lisn", "rumor_centrality")

# rumor_centrality is a single-source estimator: exact on trees, near zero BY
# CONSTRUCTION under a multi-source protocol at k = 10% of N (§8.2). Run it at
# --budgets 1.


def patch_int32_indices():
    """
    Give cosasi's estimators a networkx whose adjacency_matrix returns int32 indices.

    Surgical on purpose: only the `nx` name inside `cosasi.utils.estimators` is
    replaced, so nothing else in the process sees a patched networkx.
    """
    from cosasi.utils import estimators

    def adjacency_matrix(graph, *args, **kwargs):
        matrix = nx.adjacency_matrix(graph, *args, **kwargs).tocsr()
        matrix.indices = matrix.indices.astype(np.int32)
        matrix.indptr = matrix.indptr.astype(np.int32)

        return matrix

    shim = SimpleNamespace(
        **{name: getattr(nx, name) for name in dir(nx) if not name.startswith("_")}
    )
    shim.adjacency_matrix = adjacency_matrix
    estimators.nx = shim


def load_inputs(work_dir):
    graph_payload = np.load(os.path.join(work_dir, "graph.npz"))
    edges = graph_payload["edges"]
    num_nodes = int(graph_payload["num_nodes"])

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(map(tuple, edges))

    payload = np.load(os.path.join(work_dir, "instances.npz"), allow_pickle=True)

    return graph, payload


def sources_from_result(result, budget, num_nodes):
    """
    The predicted source set, from whichever of cosasi's two result shapes came back.

    They are genuinely different objects and conflating them is silent:

      * `SingleSourceResult.data[<method>]` is `{node: score}` over all N nodes,
        so the set is the top-k of that ranking.
      * `MultiSourceResult.data["scores"]` is `{(n1, n2, ...): score}`, it scores
        whole candidate SETS, not nodes. The prediction is the argmax TUPLE, which
        is cosasi's own answer; deriving per-node scores from it and re-ranking
        would be our reinterpretation rather than its output.

    Reading a MultiSourceResult as if it were the first shape yields a constant
    vector, whose top-k is `[0, 1, ..., k-1]` for every instance, which is
    exactly what three cosasi arms silently produced before this split existed.
    """
    payload = result.data

    if "scores" in payload and payload["scores"] and isinstance(
        next(iter(payload["scores"])), tuple
    ):
        best = max(payload["scores"].items(), key=lambda item: item[1])[0]

        return [int(node) for node in best][:budget]

    scores = np.full(num_nodes, -np.inf, dtype=np.float64)
    for mapping in payload.values():
        if not isinstance(mapping, dict):
            continue

        for node, value in mapping.items():
            if isinstance(node, (int, np.integer)) and np.isfinite(float(value)):
                scores[int(node)] = float(value)
        break

    # Ties broken by node id so a rerun is reproducible
    order = np.lexsort((np.arange(num_nodes), -scores))

    return [int(node) for node in order[:budget]]


def infer(method, infected_graph, graph, budget, horizon):
    import cosasi

    if method == "jordan":
        return cosasi.multiple_source.fast_multisource_jordan_centrality(
            infected_graph, graph, number_sources=budget
        )

    if method == "netsleuth":
        return cosasi.multiple_source.fast_multisource_netsleuth(
            infected_graph, graph, number_sources=budget
        )

    if method == "lisn":
        # LISN needs the elapsed time; our episodes record how long the cascade
        # actually ran, which is exactly that quantity
        return cosasi.multiple_source.fast_multisource_lisn(
            infected_graph, graph, t=max(1, int(horizon)), number_sources=budget
        )

    return cosasi.single_source.rumor_centrality(infected_graph, graph)


def main() -> int:
    work_dir = sys.argv[1]
    method = os.environ.get("GWM_COSASI_METHOD", "jordan").lower()

    if method not in methods:
        raise SystemExit(
            f"unknown GWM_COSASI_METHOD {method!r}; choose one of {methods}"
        )

    patch_int32_indices()

    graph, payload = load_inputs(work_dir)
    num_nodes = graph.number_of_nodes()
    episodes = [str(name) for name in payload["episode_ids"]]
    observations = payload["observations"]
    budgets = payload["budgets"].astype(int)
    horizons = payload["horizons"].astype(int)

    print(
        f"[cosasi] method={method} nodes={num_nodes} instances={len(episodes)}",
        flush=True,
    )

    predictions = {}
    for index, episode in enumerate(episodes):
        infected = [int(node) for node in np.flatnonzero(observations[index] >= 0.5)]

        if not infected:
            predictions[episode] = []
            continue

        result = infer(
            method,
            graph.subgraph(infected),
            graph,
            int(budgets[index]),
            int(horizons[index]),
        )
        predictions[episode] = sources_from_result(
            result, int(budgets[index]), num_nodes
        )

    with open(os.path.join(work_dir, "predictions.json"), "w") as handle:
        json.dump({"method": method, "sources": predictions}, handle)

    print(f"[cosasi] wrote {len(predictions)} predictions", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
