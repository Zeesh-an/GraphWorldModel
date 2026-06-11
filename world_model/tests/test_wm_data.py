import numpy as np
import torch

from wm_data import GraphInput, build_graph_input


def test_in_neighbor_weighted_direction():
    # arc 0->1 with p=0.5, arc 2->1 with p=0.25 ; node 1 has two in-neighbors
    edge_index = np.array([[0, 2], [1, 1]], dtype=np.int64)  # (src, dst)
    edge_weight = np.array([0.5, 0.25], dtype=np.float32)
    gi = build_graph_input(edge_index, edge_weight, num_nodes=3,
                           diffusion_model="IC", device=torch.device("cpu"))
    assert isinstance(gi, GraphInput)
    adj = gi.adj_norm.to_dense()  # (3,3): adj[v,u] holds the u->v weight (+ self loops, normalized)
    # node 1 aggregates FROM 0 and 2 (its in-neighbors): adj[1,0] and adj[1,2] are nonzero
    assert adj[1, 0] > 0 and adj[1, 2] > 0
    # node 0 has no in-neighbors except its self-loop
    assert adj[0, 1] == 0 and adj[0, 2] == 0 and adj[0, 0] > 0
    # IC keeps relative weights: stronger arc (0->1, p=0.5) outweighs weaker (2->1, p=0.25)
    assert adj[1, 0] > adj[1, 2]
    # edge_index/edge_weight preserved for SAGE/GAT/GT
    assert gi.edge_index.shape == (2, 2) and gi.edge_weight.shape == (2,)


def test_lt_is_binary():
    edge_index = np.array([[0, 2], [1, 1]], dtype=np.int64)
    edge_weight = np.array([0.5, 0.25], dtype=np.float32)  # ignored for LT
    gi = build_graph_input(edge_index, edge_weight, num_nodes=3,
                           diffusion_model="LT", device=torch.device("cpu"))
    adj = gi.adj_norm.to_dense()
    # LT ignores edge weights -> the two equal-structure in-edges to node 1 normalize equally
    assert torch.isclose(adj[1, 0], adj[1, 2])


from wm_data import reconstruct_episode_adjacency


def _rec(t, branch, action):
    return {"t": t, "branch": branch, "action": action}


def test_edge_replay_main_vs_cf():
    base = {(0, 1): 0.5}  # one base arc 0->1, p=0.5
    records = [
        _rec(0, "main", []),                                              # no edge op
        _rec(1, "main", [{"op": "add_edge", "target": 2, "destination": 3, "weight": 0.9}]),
        _rec(1, "cf_0", [{"op": "add_node", "target": 4}]),              # node-only cf
        _rec(2, "main", [{"op": "remove_edge", "target": 0, "destination": 1}]),
    ]
    out = reconstruct_episode_adjacency(records, base)
    # main@1 sees its own add_edge (post): base + (2,3)
    assert (2, 3) in out[(1, "main")] and (0, 1) in out[(1, "main")]
    # cf@1 branches from PRE state of step 1: base only, no (2,3)
    assert (2, 3) not in out[(1, "cf_0")] and (0, 1) in out[(1, "cf_0")]
    # main@2 removed (0,1); cumulative add_edge from step 1 persists
    assert (0, 1) not in out[(2, "main")] and (2, 3) in out[(2, "main")]


def test_edge_replay_static_when_no_edge_ops():
    base = {(0, 1): 0.5}
    records = [_rec(0, "main", [{"op": "add_node", "target": 1}]),
               _rec(1, "main", [])]
    out = reconstruct_episode_adjacency(records, base)
    assert out[(0, "main")] == base and out[(1, "main")] == base


from wm_data import build_features


def test_build_features_channels():
    edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)  # arcs 0->1, 1->2
    rec = {
        "state": {"infected": [0], "frontier": [0]},
        "action": [{"op": "add_node", "target": 1},
                   {"op": "set_edge_weight", "target": 0, "destination": 1, "weight": 0.7}],
        "next_state": {"infected": [0, 1, 2], "frontier": [1, 2]},
    }
    X, y_inf, y_fr = build_features(rec, edge_index, num_nodes=3)
    assert X.shape == (3, 6)
    assert X[0, 0] == 1 and X[0, 1] == 1            # node 0 pre-action infected+frontier
    assert X[1, 3] == 1                              # act_add on node 1
    assert X[0, 5] == 1 and X[1, 5] == 1            # edge-op endpoints 0 and 1
    assert (y_inf == np.array([1, 1, 1])).all()
    assert (y_fr == np.array([0, 1, 1])).all()
