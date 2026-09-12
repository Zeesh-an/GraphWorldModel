import numpy as np

from coding_agent.credit import (
    augment_solo,
    counterfactual_credit,
    credit_limit,
    creditable_actions,
    format_credit_report,
    max_credit_actions,
    min_credit_actions,
)
from coding_agent.envs.world_model_env import WorldModelEnvironment
from coding_agent.types import ActionOp, GraphInfo


def _graph() -> GraphInfo:
    # A star (0 to 1..5) joined to a path (5..11)
    pairs = [(0, node) for node in range(1, 6)] + [(node, node + 1) for node in range(5, 11)]
    arcs = pairs + [(destination, source) for source, destination in pairs]
    edge_index = np.array(arcs, dtype=np.int64).T

    return GraphInfo(
        num_nodes=12,
        edge_index=edge_index,
        ic_probs=np.full(edge_index.shape[1], 0.5, dtype=np.float32),
        directed=False,
    )


class _Huge:
    edge_index = np.zeros((2, 4_000_000), dtype=np.int64)


def test_credit_limit_follows_the_evaluator_cost() -> None:
    graph = _graph()
    environment = WorldModelEnvironment.oracle(graph, "IC", n_samples=200, base_seed=0)
    assert credit_limit(environment, graph) == max_credit_actions
    # digg-sized: 4.0M arcs x 200 samples per plan leaves room for the floor only
    assert credit_limit(environment, _Huge()) == min_credit_actions


def test_capped_credit_spans_the_degree_range_and_solo_matches() -> None:
    graph = _graph()
    environment = WorldModelEnvironment.oracle(graph, "IC", n_samples=8, base_seed=0)
    targets = [0, 1, 2, 3, 6, 7, 8, 11]
    plan = [[ActionOp("add_node", node) for node in targets]] + [[] for _ in range(3)]

    kept = creditable_actions(plan, 3, ("add_node",), limit=3, graph=graph)
    kept_targets = [int(action.target) for _, _, action, _ in kept]
    # Hub (degree 5), a middle node, and a leaf: evenly spaced by degree rank
    assert len(kept_targets) == 3 and 0 in kept_targets and 11 in kept_targets

    base, entries = counterfactual_credit(environment, plan, 3, 8, creditable_ops=("add_node",), limit=3, graph=graph)
    assert [entry["target"] for entry in entries] == kept_targets
    augment_solo(environment, plan, entries, 3, 8, creditable_ops=("add_node",), limit=3, graph=graph)
    assert all("solo" in entry and "delta" in entry for entry in entries)

    report = format_credit_report(base, entries, len(targets))
    assert "for 3 of 8 actions, evenly spaced by target degree" in report

    # Under the cap nothing changes
    assert len(creditable_actions(plan, 3, ("add_node",), limit=8, graph=graph)) == 8
    assert "of 8 actions" not in format_credit_report(base, entries, 3)
