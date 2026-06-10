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

from graph_utils import build_edge_index

REAL_DIRECTED = {
    "cora_ml": True,
    "digg": True,
    "twitter": True,
    "nethept": True,
    "jazz": False,
    "netscience": False,
    "power_grid": False,
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
    adj = nx.to_scipy_sparse_array(
        graph,
        nodelist=list(range(graph.number_of_nodes())),
        dtype=np.float32,
        format="csr",
    )
    edge_index, ic_probs, lt_weights = build_edge_index(adj)

    if prob_model == "uniform":
        ic_probs = np.full(ic_probs.shape, float(uniform_p), dtype=np.float32)
        lt_weights = ic_probs.copy()

    ic_prob_map = {
        (int(edge_index[0, i]), int(edge_index[1, i])): float(ic_probs[i])
        for i in range(edge_index.shape[1])
    }

    if node_feats is None:
        deg = np.array([d for _, d in graph.degree()], dtype=np.float32)

        # For graphs which have no node features (such as synthethic graphs), use log(1 + degree) as a single node feature
        node_feats = np.log1p(deg).reshape(-1, 1).astype(np.float32)

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
    n: int = 100,
    er_p: float = 0.05,
    ba_m: int = 3,
    ws_k: int = 6,
    ws_p: float = 0.1,
    seed: int = 0,
    prob_model: str = "weighted",
    uniform_p: float = 0.1,
) -> GraphBundle:
    """Generate one synthetic graph instance index is folded into the seed)."""
    inst_seed = seed + index

    if family == "er":
        graph = nx.gnp_random_graph(n, er_p, seed=inst_seed)
        graph_id = f"er_n{n}_p{er_p}_s{inst_seed}"
    elif family == "ba":
        graph = nx.barabasi_albert_graph(n, ba_m, seed=inst_seed)
        graph_id = f"ba_n{n}_m{ba_m}_s{inst_seed}"
    elif family == "ws":
        graph = nx.watts_strogatz_graph(n, ws_k, ws_p, seed=inst_seed)
        graph_id = f"ws_n{n}_k{ws_k}_p{ws_p}_s{inst_seed}"
    elif family == "karate":
        graph = nx.karate_club_graph()
        graph_id = "karate"
    else:
        raise ValueError(f"Unknown synthetic family {family}")

    graph = nx.convert_node_labels_to_integers(graph)

    return bundle_from_nx(graph_id, graph, family, None, None, prob_model, uniform_p)


def adj_to_nx(adj: sp.spmatrix, directed: bool) -> nx.Graph | nx.DiGraph:
    """Scipy adjacency -> networkx, preserving directedness and dropping self-loops"""
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(range(adj.shape[0]))

    coo = adj.tocoo()
    for u, v in zip(coo.row.tolist(), coo.col.tolist()):
        if u != v:
            graph.add_edge(int(u), int(v))

    return graph


def make_real_bundle_from_arrays(
    dataset: str,
    adj: sp.spmatrix,
    node_feats: np.ndarray,
    node_labels: np.ndarray,
    prob_model: str = "weighted",
    uniform_p: float = 0.1,
) -> GraphBundle:
    """Build a GraphBundle from already-loaded dataset arrays (no download)."""
    graph = adj_to_nx(adj, REAL_DIRECTED[dataset])

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

    The loader is imported lazily so unit tests need no network access.
    """
    loaders = {
        "cora_ml": ("datasets.cora_ml", "download_cora_ml", "load_cora_ml"),
        "digg": ("datasets.digg", "download_digg", "load_digg"),
        "twitter": ("datasets.twitter", "download_twitter", "load_twitter"),
        "jazz": ("datasets.jazz", "download_jazz", "load_jazz"),
        "netscience": ("datasets.netscience", "download_netscience", "load_netscience"),
        "power_grid": ("datasets.power_grid", "download_power_grid", "load_power_grid"),
        "nethept": ("datasets.nethept", "download_nethept", "load_nethept"),
    }

    module_name, dl_name, ld_name = loaders[dataset]
    module = importlib.import_module(module_name)
    raw_path = getattr(module, dl_name)()

    adj, node_feats, node_labels, _n = getattr(module, ld_name)(raw_path)

    return make_real_bundle_from_arrays(
        dataset,
        adj,
        node_feats,
        node_labels,
        prob_model=prob_model,
        uniform_p=uniform_p,
    )
