import numpy as np
from coding_agent.types import Action, State, GraphInfo, TaskSpec, Trajectory


def _triangle() -> GraphInfo:
    # 0->1, 1->2, 2->0 directed triangle, uniform ic prob 0.5
    ei = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64)
    ic = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    return GraphInfo(num_nodes=3, edge_index=ei, ic_probs=ic, directed=True)


def test_graphinfo_neighbors_and_degree():
    g = _triangle()
    assert g.out_neighbors(0) == [1]
    assert g.in_neighbors(0) == [2]
    assert g.degree(0) == 2  # one in + one out


def test_graphinfo_from_store_entry():
    entry = {
        "edge_index": np.array([[0, 1], [1, 2]], dtype=np.int64),
        "ic_probs": np.array([0.3, 0.7], dtype=np.float32),
        "num_nodes": 3,
        "meta": {"directed": True},
    }
    g = GraphInfo.from_store_entry(entry)
    assert g.num_nodes == 3 and g.directed is True
    assert g.out_neighbors(1) == [2]


def test_action_and_state_reexport():
    a = Action("add_node", 1)
    assert a.op == "add_node" and a.target == 1
    s = State(infected=[0], frontier=[0])
    assert s.to_dict()["infected_count"] == 1


def test_trajectory_fields():
    tr = Trajectory(states=[State([], [])], actions=[[]], reward=4.0, infected_counts=[0.0])
    assert tr.reward == 4.0 and tr.cost == {}
