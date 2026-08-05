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
    "digg": True,
    "twitter": True,
    "nethept": True,
    "weibo": True,
    "wiki_vote": True,
    "email_eu_core": True,
    # Adaptive/online IM benchmarks (Han et al. PVLDB 2018); all three are
    # scale targets, not day-one datasets; see each loader's SCALE WARNING
    "epinions": True,
    "livejournal": True,
    "orkut": False,
    # Cora-ML is symmetrized by standardize(), matching DeepIM/MOEIM
    "cora_ml": False,
    "jazz": False,
    "netscience": False,
    "power_grid": False,
    "youtube": False,
    "netphy": False,
    "facebook": False,
    "ca_grqc": False,
    "lastfm_asia": False,
    # Network-dismantling benchmarks (research/critical_node_detection.md §6.2).
    # Every one is loaded undirected: that literature is undirected almost
    # end to end, and symmetrizing to reuse a published baseline is a stated
    # preprocessing choice rather than an accident (§8.2 trap 9).
    "usair97": False,
    "crime": False,
    "corruption": False,
    "hamsterster": False,
    "road_eu": False,
    "euroroad": False,
    "intnet1": False,
    "ppi_yeast": False,
    "human_ppi_vidal": False,
    "pgp": False,
    "openflights": False,
    "p2p_gnutella": False,
    # Source-localization benchmarks (research/source_localization.md §6.2).
    # `dolphins` is the only graph that literature uses which the other task
    # files did not already need; `deezer` is IVGD's scalability column and has no
    # published F1 row at all, so it is a cost target rather than a comparison.
    "dolphins": False,
    "deezer": False,
    # Influence-blocking benchmarks (research/influence_blocking.md §6.2). Every one
    # is loaded with its own file's directedness rather than symmetrized, because
    # unlike the dismantling literature this one is largely DIRECTED and its
    # published probabilities are 1/in-degree — which is only defined on arcs.
    #
    # Warning: two of these collide by name with graphs we already load, and both
    # collisions are recorded in §6.3 rather than resolved by renaming theirs:
    # `epinions1` is SNAP's soc-Epinions1 (75,879) while `epinions` is the SIGNED
    # soc-sign-epinions (131,828), and `p2p_gnutella08` is the 08 snapshot NIE and
    # NAMM use while `p2p_gnutella` is Gnutella31's giant component (62,561).
    "p2p_gnutella08": True,
    "p2p_gnutella24": True,
    "cit_hepth": True,
    "cit_hepph": True,
    "email_enron": False,
    "slashdot": True,
    "epinions1": True,
    "gowalla": False,
    "email_euall": True,
    "web_stanford": True,
    "dblp": False,
    # The last two exceed what the NDlib rollout path can simulate in reasonable
    # time; they are scalability targets on the same footing as `twitter` and
    # `youtube`, and each loader's docstring says so
    "higgs_twitter": True,
    "pokec": True,
    # Cascade-reconstruction benchmarks (research/cascade_reconstruction.md §6).
    # All undirected: like the dismantling literature, this one is undirected end
    # to end — every Steiner-tree method in §3 is defined on the symmetric contact
    # graph, and DITTO's own loaders call `nx.read_edgelist` without `create_using`.
    #
    # `oregon2` and `rt_pol` are DITTO's two graphs we did not already have;
    # `ca_hepth`, `email_univ`, `uci_students` and `infectious` are the rest of
    # Xiao's two tables; `citeseer` is DIPT's third graph. Warning: `ca_hepth` is
    # the CO-AUTHORSHIP network and `cit_hepth` the CITATION one — same arXiv
    # section, different graphs, and each loader's docstring says which.
    "oregon2": False,
    "rt_pol": False,
    "ca_hepth": False,
    "email_univ": False,
    "uci_students": False,
    "infectious": False,
    "citeseer": False,
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


# ConTinEst's three 2x2 Kronecker seed matrices (research/adaptive_online_im.md
# 6.3b). The generator is the k-fold tensor power of one of these; edge (i, j)
# is sampled with the resulting probability.
kronecker_seeds = {
    "core_periphery": ((0.9, 0.5), (0.5, 0.3)),
    "random": ((0.5, 0.5), (0.5, 0.5)),
    "hierarchical": ((0.9, 0.1), (0.1, 0.9)),
}
# Kronecker sampling is O(N^2) because every pair carries its own probability.
# Fine at the sizes this pipeline can simulate; a guard beats an OOM at 1M.
kronecker_max_nodes = 20_000


def kronecker_graph(
    num_nodes: int, variant: str, seed: int
) -> tuple[nx.Graph, int]:
    """
    Stochastic Kronecker graph, returned with the power-of-two order it was
    generated at.

    The construction only defines graphs on 2^k nodes, so we generate at the
    next power of two and induce on the first `num_nodes` of them. Returning the
    order lets the graph_id record what was actually generated rather than
    implying `num_nodes` was native.
    """
    if variant not in kronecker_seeds:
        raise ValueError(
            f"unknown kronecker variant {variant!r}; "
            f"choose one of {sorted(kronecker_seeds)}"
        )

    if num_nodes > kronecker_max_nodes:
        raise ValueError(
            f"kronecker sampling is O(N^2); {num_nodes} nodes exceeds the "
            f"{kronecker_max_nodes} guard. Raise kronecker_max_nodes if you "
            f"genuinely want a dense N x N draw."
        )

    order = 1
    while order < num_nodes:
        order *= 2

    probabilities = np.array(kronecker_seeds[variant], dtype=np.float64)
    full = probabilities
    while full.shape[0] < order:
        full = np.kron(full, probabilities)

    rng = np.random.default_rng(seed)
    # Upper triangle only, then mirrored: the seed matrices are symmetric, so
    # drawing both directions independently would double every edge's chance
    draws = rng.random((order, order)) < full
    draws = np.triu(draws, k=1)

    graph = nx.Graph()
    graph.add_nodes_from(range(order))
    graph.add_edges_from(zip(*np.nonzero(draws)))

    return graph.subgraph(range(num_nodes)).copy(), order


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
    plc_m: int = 3,
    plc_p: float = 0.05,
    kron_variant: str = "core_periphery",
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
    elif family == "powerlaw_cluster":
        # RL4IM's family (research/adaptive_online_im.md 6.3b): BA growth plus a
        # triangle-closing step, so it has the clustering BA lacks while keeping
        # the heavy tail. Defaults are RL4IM's, read from its repo rather than
        # its paper: RL4IM's own basic_env.yaml sets m=3, p=0.05 (avg degree 5.9), which is what its graph.py passes to nx.powerlaw_cluster_graph. The paper's "average degree 3" disagrees with its own code; the code wins.
        graph = nx.powerlaw_cluster_graph(num_nodes, plc_m, plc_p, seed=instance_seed)
        graph_id = f"plc_n{num_nodes}_m{plc_m}_p{plc_p}_s{instance_seed}"
    elif family == "kronecker":
        graph, order = kronecker_graph(num_nodes, kron_variant, instance_seed)
        graph_id = f"kron_{kron_variant}_n{num_nodes}_o{order}_s{instance_seed}"
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
