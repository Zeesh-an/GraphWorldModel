from pathlib import Path

from generate_wm_data import run_generation, GenConfig
from validate_wm_data import compute_checks


def _cfg(out_dir: Path) -> GenConfig:
    return GenConfig(
        dataset="ba", num_graphs=2, syn_nodes=60, er_p=0.1, ba_m=3,
        models=["IC"], prob_model="uniform", uniform_p=0.2,
        budget=3, budget_pct=None, algorithms=["random", "degree"],
        rollouts=3, horizon=6, inject_p=0.4, p_add=0.5, p_remove=0.5,
        cf_prob=0.6, cf_branches=2, split=(1.0, 0.0, 0.0),
        seed=7, out_dir=str(out_dir),
    )


def test_checks_run_and_report(tmp_path: Path):
    out = tmp_path / "out"
    run_generation(_cfg(out))
    checks = compute_checks(out)

    assert checks["n_transitions"] > 0
    # reward distribution is not collapsed: rewards genuinely vary
    assert checks["reward_max"] > checks["reward_min"]
    assert checks["reward_std"] > 0.0
    # action sensitivity: at least one same-state pair with differing next_state
    assert "action_sensitivity_pairs" in checks
    # monotonicity ratio reported in [0, 1]
    assert 0.0 <= checks["main_monotone_ratio"] <= 1.0
    # per-algorithm final spread reported for ranking sanity (random + degree here)
    assert set(checks["per_algorithm_final_spread"]) == {"random", "degree"}
