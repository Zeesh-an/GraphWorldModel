import pytest

from wm_graphs import GraphBundle, make_synthetic_bundle
from wm_simulator import ActionOp, Simulator, State


def test_action_op_to_dict():
    assert ActionOp(op="add_seed", target=17).to_dict() == {"op": "add_seed", "target": 17}


def test_action_op_rejects_invalid_op():
    with pytest.raises(ValueError, match="Unknown op"):
        ActionOp(op="invalid_op", target=0)


def test_state_to_dict_sorts_and_counts():
    s = State(infected=[3, 0, 7], frontier=[7, 3])
    assert s.to_dict() == {
        "infected": [0, 3, 7], "frontier": [3, 7], "infected_count": 3, "frontier_count": 2,
    }


def _sim(model: str = "IC", seed: int = 0) -> tuple[Simulator, GraphBundle]:
    b = make_synthetic_bundle("ba", index=0, n=60, ba_m=3, seed=7,
                              prob_model="uniform", uniform_p=0.3)
    sim = Simulator(b.nx_graph, ic_prob_map=b.ic_prob_map, seed=seed)
    sim.reset(model)
    return sim, b


def test_reset_starts_all_susceptible():
    sim, _ = _sim("IC")
    s0 = sim.current_state()
    assert s0.infected == [] and s0.frontier == []


def test_add_seed_makes_node_infected():
    sim, _ = _sim("IC")
    s1 = sim.advance([ActionOp("add_seed", 5)])
    assert 5 in s1.infected  # the committed node is activated


def test_ic_injected_seed_not_in_frontier_after_spread():
    # In IC, a freshly add_seed'd node spreads once then becomes Removed (status 2),
    # so it must NOT appear in the next frontier (status-1 rule).
    sim, _ = _sim("IC")
    s1 = sim.advance([ActionOp("add_seed", 5)])
    assert 5 in s1.infected
    assert 5 not in s1.frontier


def test_ic_determinism_same_seed():
    # Run each simulator to completion before constructing the next: NDlib shares
    # numpy's GLOBAL RNG, so interleaving two live sims is not reproducible. The
    # orchestrator always runs sims non-interleaved, which is what we assert here.
    bag = [ActionOp("add_seed", 5)]

    def run() -> dict:
        b = make_synthetic_bundle("ba", index=0, n=60, ba_m=3, seed=7,
                                  prob_model="uniform", uniform_p=0.3)
        sim = Simulator(b.nx_graph, ic_prob_map=b.ic_prob_map, seed=42)
        sim.reset("IC")
        return sim.advance(bag).to_dict()

    assert run() == run()


def test_remove_node_defrontiers_ic():
    # remove_node on an IC frontier node sets it to Removed (status 2): it stays
    # counted as infected but is no longer a spreader (drops from the frontier),
    # immediately, before the next diffusion step. uniform_p=1.0 makes spread
    # deterministic so node 5 (>=3 neighbors in BA m=3) always yields a frontier.
    b = make_synthetic_bundle("ba", index=0, n=60, ba_m=3, seed=7,
                              prob_model="uniform", uniform_p=1.0)
    sim = Simulator(b.nx_graph, ic_prob_map=b.ic_prob_map, seed=0)
    sim.reset("IC")
    s1 = sim.advance([ActionOp("add_seed", 5)])
    assert s1.frontier, "expected a non-empty frontier after spread"
    target = s1.frontier[0]
    sim.apply_actions([ActionOp("remove_node", target)])
    s_now = sim.current_state()  # status after the action, before stepping
    assert target in s_now.infected  # still infected (status 2)
    assert target not in s_now.frontier  # no longer a spreader


def test_lt_add_seed_in_frontier():
    # LT has no Removed state; a freshly activated node IS part of the new wave.
    b = make_synthetic_bundle("ba", index=0, n=60, ba_m=3, seed=7, prob_model="weighted")
    sim = Simulator(b.nx_graph, ic_prob_map=b.ic_prob_map, seed=0)
    sim.reset("LT")
    s1 = sim.advance([ActionOp("add_seed", 5)])
    assert 5 in s1.infected
    # node 5 was just activated, so it is always in the newly-flipped LT frontier
    # (this checks the LT frontier rule, not stochastic onward propagation)
    assert 5 in s1.frontier


def test_snapshot_restore_round_trip():
    b = make_synthetic_bundle("ba", index=0, n=60, ba_m=3, seed=7,
                              prob_model="uniform", uniform_p=0.3)
    sim = Simulator(b.nx_graph, ic_prob_map=b.ic_prob_map, seed=1)
    sim.reset("IC")
    sim.advance([ActionOp("add_seed", 5)])
    snap = sim.snapshot()
    state_before = sim.current_state().to_dict()
    # mutate, then restore
    sim.advance([ActionOp("add_seed", 10)])
    sim.restore(snap)
    assert sim.current_state().to_dict() == state_before


def test_counterfactual_fork_independence():
    # Two different actions from the same snapshot yield independent next-states.
    b = make_synthetic_bundle("ba", index=0, n=60, ba_m=3, seed=7,
                              prob_model="uniform", uniform_p=0.3)
    sim = Simulator(b.nx_graph, ic_prob_map=b.ic_prob_map, seed=2)
    sim.reset("IC")
    sim.advance([ActionOp("add_seed", 5)])
    snap = sim.snapshot()
    s_a = sim.advance([ActionOp("add_seed", 10)])
    sim.restore(snap)
    s_b = sim.advance([])  # NULL
    # 10 is activated only in branch A
    assert 10 in s_a.infected
    assert 10 not in s_b.infected
