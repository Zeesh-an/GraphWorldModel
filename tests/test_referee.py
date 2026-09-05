from coding_agent.run import _agreement_keys
from pipeline.conditions import parse_arm


def test_canned_arms_default_to_the_oracle_referee() -> None:
    for spec in ("baseline:imm", "routing", "external:gnd", "discovery:eoh"):
        assert parse_arm(spec).evaluator == "oracle", spec


def test_an_explicit_evaluator_still_wins() -> None:
    assert parse_arm("baseline:imm@monte_carlo").evaluator == "monte_carlo"


def test_agreement_keys_reprefix_the_referee_helpers() -> None:
    measured = {"referee_reward": -0.1, "referee_reward_se": 0.01, "resim_error": 0.1}

    assert _agreement_keys(measured) == {
        "mc_reward": -0.1,
        "mc_reward_se": 0.01,
        "mc_resim_error": 0.1,
    }
