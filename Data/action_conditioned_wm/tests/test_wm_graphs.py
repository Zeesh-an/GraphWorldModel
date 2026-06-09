import networkx as nx
import numpy as np
import scipy.sparse as sp

from wm_graphs import GraphBundle, make_synthetic_bundle, adj_to_nx, REAL_DIRECTED, make_real_bundle_from_arrays


def test_ba_bundle_shape_and_undirected():
    b = make_synthetic_bundle("ba", index=0, n=100, ba_m=3, seed=0, prob_model="weighted")
    assert isinstance(b, GraphBundle)
    assert b.nx_graph.number_of_nodes() == 100
    assert not b.nx_graph.is_directed()
    assert b.graph_id == "ba_n100_m3_s0"
    # node features = log1p(degree), shape (N, 1)
    assert b.node_feats.shape == (100, 1)
    # array dtypes/shapes the simulator depends on (guard against silent regressions)
    assert b.edge_index.shape[0] == 2
    assert b.edge_index.dtype == np.int32
    assert b.ic_probs.dtype == np.float32
    assert b.lt_weights.dtype == np.float32
    # ic_prob_map covers both directions of every undirected edge
    assert len(b.ic_prob_map) == 2 * b.nx_graph.number_of_edges()
    # weighted-cascade probs are in (0, 1]
    probs = np.array(list(b.ic_prob_map.values()))
    assert probs.min() > 0 and probs.max() <= 1.0


def test_er_ws_karate_construct():
    er = make_synthetic_bundle("er", index=1, n=50, er_p=0.1, seed=1, prob_model="weighted")
    assert er.nx_graph.number_of_nodes() == 50
    ws = make_synthetic_bundle("ws", index=0, n=60, ws_k=6, ws_p=0.1, seed=2, prob_model="weighted")
    assert ws.nx_graph.number_of_nodes() == 60
    kar = make_synthetic_bundle("karate", index=0, seed=0, prob_model="weighted")
    assert kar.nx_graph.number_of_nodes() == 34
    assert kar.graph_id == "karate"


def test_uniform_prob_model():
    b = make_synthetic_bundle("ba", index=0, n=40, ba_m=2, seed=0,
                              prob_model="uniform", uniform_p=0.1)
    probs = np.array(list(b.ic_prob_map.values()))
    assert np.allclose(probs, 0.1)


def test_meta_fields_present():
    b = make_synthetic_bundle("ba", index=0, n=30, ba_m=2, seed=0, prob_model="weighted")
    for key in ("graph_type", "directed", "n_nodes", "n_edges"):
        assert key in b.meta
    assert b.meta["graph_type"] == "ba"
    assert b.meta["directed"] is False


def test_adj_to_nx_directed_preserved():
    # 0 -> 1, 1 -> 2 (directed, asymmetric)
    adj = sp.csr_matrix(
        (np.array([1.0, 1.0]), (np.array([0, 1]), np.array([1, 2]))), shape=(3, 3)
    )
    g = adj_to_nx(adj, directed=True)
    assert g.is_directed()
    assert g.number_of_edges() == 2
    assert g.has_edge(0, 1) and not g.has_edge(1, 0)


def test_adj_to_nx_drops_self_loops():
    # diagonal entries at (0,0) and (1,1), plus a real edge (1,2)
    adj = sp.csr_matrix(
        (np.ones(3), (np.array([0, 1, 1]), np.array([0, 1, 2]))), shape=(3, 3)
    )
    g = adj_to_nx(adj, directed=True)
    assert g.number_of_edges() == 1
    assert not g.has_edge(0, 0)


def test_adj_to_nx_undirected_dedups():
    # symmetric adj for an undirected edge 0-1
    adj = sp.csr_matrix(
        (np.array([1.0, 1.0]), (np.array([0, 1]), np.array([1, 0]))), shape=(2, 2)
    )
    g = adj_to_nx(adj, directed=False)
    assert not g.is_directed()
    assert g.number_of_edges() == 1


def test_directedness_map_matches_spec():
    assert REAL_DIRECTED["cora_ml"] is True
    assert REAL_DIRECTED["nethept"] is True
    assert REAL_DIRECTED["jazz"] is False
    assert REAL_DIRECTED["power_grid"] is False


def test_make_real_bundle_from_arrays_directed():
    adj = sp.csr_matrix(
        (np.array([1.0, 1.0]), (np.array([0, 1]), np.array([1, 2]))), shape=(3, 3)
    )
    feats = np.ones((3, 4), dtype=np.float32)
    labels = np.zeros(3, dtype=np.int32)
    b = make_real_bundle_from_arrays("cora_ml", adj, feats, labels, prob_model="weighted")
    assert b.graph_id == "cora_ml"
    assert b.nx_graph.is_directed()
    assert b.node_feats.shape == (3, 4)
    assert b.meta["directed"] is True
