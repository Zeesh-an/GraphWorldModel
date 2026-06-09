import numpy as np

from wm_graphs import make_synthetic_bundle
from wm_actions import SPINE_ALGORITHMS, select_seeds, estimate_spread


def _bundle(n=30):
    # Small graph + small MC budget keeps CELF / local-search fast in tests.
    return make_synthetic_bundle("ba", index=0, n=n, ba_m=3, seed=7,
                                 prob_model="uniform", uniform_p=0.2)


def test_all_algorithms_return_k_distinct_valid_nodes():
    b = _bundle(n=30)
    for algo in SPINE_ALGORITHMS:
        rng = np.random.default_rng(0)
        seeds = select_seeds(b, k=3, algorithm=algo, model="IC", rng=rng,
                             mc_runs=4, horizon=8)
        assert len(seeds) == 3
        assert len(set(seeds)) == 3
        assert all(0 <= s < 30 for s in seeds)


def test_unknown_algorithm_raises():
    b = _bundle()
    rng = np.random.default_rng(0)
    try:
        select_seeds(b, k=3, algorithm="nope", model="IC", rng=rng)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_estimate_spread_bounds():
    # Robust (non-MC-flaky) bounds: no seeds -> 0 spread; k seeds -> at least k
    # activated (the seeds themselves are always active).
    b = _bundle()
    rng = np.random.default_rng(0)
    assert estimate_spread(b, [], model="IC", mc_runs=4, horizon=10, rng=rng) == 0.0
    s = estimate_spread(b, [0, 1, 2, 3, 4], model="IC", mc_runs=4, horizon=10, rng=rng)
    assert s >= 5.0


from wm_simulator import ActionOp, State
from wm_actions import sample_injection, counterfactual_actions


def _state():
    return State(infected=[0, 1, 2, 3], frontier=[2, 3])


def test_sample_injection_null_when_not_injecting():
    rng = np.random.default_rng(0)
    bag = sample_injection(_state(), n_nodes=10, rng=rng,
                           p_inject=0.0, p_add=0.5, p_remove=0.5)
    assert bag == []


def test_sample_injection_emits_valid_op_when_injecting():
    rng = np.random.default_rng(0)
    bag = sample_injection(_state(), n_nodes=10, rng=rng,
                           p_inject=1.0, p_add=1.0, p_remove=0.0)
    assert len(bag) == 1 and bag[0].op == "add_seed"
    assert bag[0].target not in _state().infected  # add_seed targets a susceptible node


def test_remove_targets_active_node():
    rng = np.random.default_rng(1)
    bag = sample_injection(_state(), n_nodes=10, rng=rng,
                           p_inject=1.0, p_add=0.0, p_remove=1.0)
    assert len(bag) == 1 and bag[0].op == "remove_node"
    assert bag[0].target in _state().frontier


def test_counterfactual_actions_distinct():
    rng = np.random.default_rng(2)
    main = [ActionOp("add_seed", 5)]
    cfs = counterfactual_actions(_state(), n_nodes=10, main_bag=main, n=2, rng=rng)
    assert len(cfs) == 2
    serialized = [tuple(sorted((a.op, a.target) for a in bag)) for bag in cfs + [main]]
    assert len(set(serialized)) == len(serialized)  # all distinct
