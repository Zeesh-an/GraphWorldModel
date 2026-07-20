"""
Graph providers: a GraphBundle (networkx graph + edge probabilities + node
features + metadata) for synthetic graphs (ER, BA, WS, and Karate) and the real datasets
Edge probabilities reuse graph_utils.build_edge_index
"""

import importlib
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import scipy.sparse as sp

from data.graph_utils import build_edge_index

real_directed = {
    "cora_ml": True,
    "digg": True,
    "twitter": True,
    "nethept": True,
    "weibo": True,
    "jazz": False,
    "netscience": False,
    "power_grid": False,
    "youtube": False,
}


@dataclass
class GraphBundle:
    graph_id: str
    nx_graph: nx.Graph | nx.DiGraph
    edge_index: np.ndarray  # (2, E) int32
    ic_probs: np.ndarray  # (E) float32
    lt_weights: np.ndarray  # (E) float32
    node_feats: np.ndarray  # (N, F) float32
    node_labels: np.ndarray  # (N) int32
    ic_prob_map: dict[
        tuple[int, int], float
    ]  # (u, v) -> p; both directions if undirected
    meta: dict[str, str | bool | int] = field(default_factory=dict)


def bundle_from_nx(
    graph_id: str,
    graph: nx.Graph | nx.DiGraph,
    graph_type: str,
    node_feats: np.ndarray | None,
    node_labels: np.ndarray | None,
    prob_model: str,
    uniform_p: float,
) -> GraphBundle:
    adjacency = nx.to_scipy_sparse_array(
        graph,
        nodelist=list(range(graph.number_of_nodes())),
        dtype=np.float32,
        format="csr",
    )
    edge_index, ic_probs, lt_weights = build_edge_index(adjacency)

    if prob_model == "uniform":
        ic_probs = np.full(ic_probs.shape, float(uniform_p), dtype=np.float32)
        lt_weights = ic_probs.copy()

    ic_prob_map = {
        (int(edge_index[0, edge]), int(edge_index[1, edge])): float(ic_probs[edge])
        for edge in range(edge_index.shape[1])
    }

    if node_feats is None:
        degrees = np.array([degree for _, degree in graph.degree()], dtype=np.float32)

        # For graphs which have no node features (such as synthetic graphs), use log(1 + degree) as a single node feature
        node_feats = np.log1p(degrees).reshape(-1, 1).astype(np.float32)

    if node_labels is None:
        node_labels = np.zeros(graph.number_of_nodes(), dtype=np.int32)

    meta = {
        "graph_type": graph_type,
        "directed": graph.is_directed(),
        "n_nodes": int(graph.number_of_nodes()),
        "n_edges": int(graph.number_of_edges()),
        "prob_model": prob_model,
    }

    return GraphBundle(
        graph_id,
        graph,
        edge_index,
        ic_probs,
        lt_weights,
        node_feats,
        node_labels,
        ic_prob_map,
        meta,
    )


def make_synthetic_bundle(
    family: str,
    index: int,
    num_nodes: int = 100,
    er_p: float = 0.05,
    ba_m: int = 3,
    ws_k: int = 6,
    ws_p: float = 0.1,
    sbm_blocks: int = 4,
    sbm_p_in: float = 0.15,
    sbm_p_out: float = 0.01,
    seed: int = 0,
    prob_model: str = "weighted",
    uniform_p: float = 0.1,
) -> GraphBundle:
    """Generate one synthetic graph instance (index is folded into the seed)."""
    instance_seed = seed + index

    if family == "er":
        graph = nx.gnp_random_graph(num_nodes, er_p, seed=instance_seed)
        graph_id = f"er_n{num_nodes}_p{er_p}_s{instance_seed}"
    elif family == "ba":
        graph = nx.barabasi_albert_graph(num_nodes, ba_m, seed=instance_seed)
        graph_id = f"ba_n{num_nodes}_m{ba_m}_s{instance_seed}"
    elif family == "ws":
        graph = nx.watts_strogatz_graph(num_nodes, ws_k, ws_p, seed=instance_seed)
        graph_id = f"ws_n{num_nodes}_k{ws_k}_p{ws_p}_s{instance_seed}"
    elif family == "sbm":
        # Even block sizes (remainder folded into the first block); dense within
        # blocks, sparse across — the community structure BA graphs lack
        sizes = [num_nodes // sbm_blocks] * sbm_blocks
        sizes[0] += num_nodes - sum(sizes)
        block_probs = [
            [sbm_p_in if row == column else sbm_p_out for column in range(sbm_blocks)]
            for row in range(sbm_blocks)
        ]
        graph = nx.stochastic_block_model(sizes, block_probs, seed=instance_seed)
        graph_id = (
            f"sbm_n{num_nodes}_b{sbm_blocks}_pin{sbm_p_in}_pout{sbm_p_out}"
            f"_s{instance_seed}"
        )
    elif family == "karate":
        graph = nx.karate_club_graph()
        graph_id = "karate"
    else:
        raise ValueError(f"Unknown synthetic family {family}")

    graph = nx.convert_node_labels_to_integers(graph)

    return bundle_from_nx(graph_id, graph, family, None, None, prob_model, uniform_p)


def adj_to_nx(adjacency: sp.spmatrix, directed: bool) -> nx.Graph | nx.DiGraph:
    """Scipy adjacency -> networkx, preserving directedness and dropping self-loops"""
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(range(adjacency.shape[0]))

    coo = adjacency.tocoo()
    for source, destination in zip(coo.row.tolist(), coo.col.tolist()):
        if source != destination:
            graph.add_edge(int(source), int(destination))

    return graph


def make_real_bundle_from_arrays(
    dataset: str,
    adjacency: sp.spmatrix,
    node_feats: np.ndarray,
    node_labels: np.ndarray,
    prob_model: str = "weighted",
    uniform_p: float = 0.1,
) -> GraphBundle:
    """Build a GraphBundle from already-loaded dataset arrays (no download)."""
    graph = adj_to_nx(adjacency, real_directed[dataset])

    return bundle_from_nx(
        dataset,
        graph,
        dataset,
        node_feats.astype(np.float32),
        node_labels.astype(np.int32),
        prob_model,
        uniform_p,
    )


def make_real_bundle(
    dataset: str, prob_model: str = "weighted", uniform_p: float = 0.1
) -> GraphBundle:
    """
    Download (if needed) + load a real dataset, then build a GraphBundle.

    The loader is imported lazily so building synthetic graphs needs no network access.
    """
    if dataset not in real_directed:
        raise ValueError(
            f"unknown real dataset {dataset!r}; choose one of {sorted(real_directed)}"
        )

    # Every loader module follows the same contract: data/datasets/<name>.py
    # exposing download_<name>() -> raw path and load_<name>(path) -> arrays
    module = importlib.import_module(f"data.datasets.{dataset}")
    raw_path = getattr(module, f"download_{dataset}")()

    adjacency, node_feats, node_labels, _ = getattr(module, f"load_{dataset}")(raw_path)

    return make_real_bundle_from_arrays(
        dataset,
        adjacency,
        node_feats,
        node_labels,
        prob_model=prob_model,
        uniform_p=uniform_p,
    )
