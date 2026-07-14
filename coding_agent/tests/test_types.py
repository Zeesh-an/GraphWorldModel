import numpy as np

from coding_agent.types import Action, GraphInfo, State, Trajectory


def _triangle() -> GraphInfo:
    # 0->1, 1->2, 2->0 directed triangle, uniform ic prob 0.5
    edge_index = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64)
    ic_probs = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    return GraphInfo(
        num_nodes=3, edge_index=edge_index, ic_probs=ic_probs, directed=True
    )


def test_graphinfo_neighbors_and_degree():
    graph = _triangle()
    assert graph.out_neighbors(0) == [1]
    assert graph.in_neighbors(0) == [2]
    assert graph.degree(0) == 2  # one in + one out


def test_graphinfo_from_store_entry():
    entry = {
        "edge_index": np.array([[0, 1], [1, 2]], dtype=np.int64),
        "ic_probs": np.array([0.3, 0.7], dtype=np.float32),
        "num_nodes": 3,
        "meta": {"directed": True},
    }
    graph = GraphInfo.from_store_entry(entry)
    assert graph.num_nodes == 3 and graph.directed is True
    assert graph.out_neighbors(1) == [2]


def test_action_and_state_reexport():
    action = Action("add_node", 1)
    assert action.op == "add_node" and action.target == 1
    state = State(infected=[0], frontier=[0])
    assert state.to_dict()["infected_count"] == 1


def test_trajectory_fields():
    trajectory = Trajectory(
        states=[State([], [])], actions=[[]], reward=4.0, infected_counts=[0.0]
    )
    assert trajectory.reward == 4.0 and trajectory.cost == {}
