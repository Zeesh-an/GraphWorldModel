import numpy as np

from coding_agent.agent import Conversation
from coding_agent.methods.base import accepts
from coding_agent.methods.evolve import (
    choose_operator,
    compact_turn,
    exploration_weight,
    explore_operators,
)
from coding_agent.prompts import (
    build_attempts_table,
    build_evolve_prompt,
    build_population_table,
    mechanism_of,
    parse_ideas,
    parse_reflection,
)
from coding_agent.types import Trajectory


def _trajectory(reward: float, se: float) -> Trajectory:
    return Trajectory(states=[], actions=[], reward=reward, infected_counts=[], cost={"reward_se": se})


def test_exploration_decays_with_the_remaining_budget_and_stagnation_adds_back() -> None:
    assert exploration_weight(0, 20, 0) > exploration_weight(10, 20, 0) > exploration_weight(19, 20, 0)
    assert exploration_weight(19, 20, 0) < 0.1
    assert exploration_weight(19, 20, 4) > exploration_weight(19, 20, 0)


def test_operator_schedule_explores_early_exploits_late_and_respects_the_population() -> None:
    early = [choose_operator(np.random.default_rng(i), 0, 20, 0, 5, 2) for i in range(200)]
    late = [choose_operator(np.random.default_rng(i), 19, 20, 0, 5, 0) for i in range(200)]
    early_explore = sum(op in explore_operators for op in early) / len(early)
    late_explore = sum(op in explore_operators for op in late) / len(late)

    assert early_explore > 0.6 > late_explore
    # A single-member population cannot crossover or synthesize
    assert all(
        choose_operator(np.random.default_rng(i), 0, 20, 0, 1, 2)
        not in ("crossover", "synthesize")
        for i in range(100)
    )
    # A stall at the patience floor forces an explore move
    assert all(
        choose_operator(np.random.default_rng(i), 19, 20, 3, 5, 2) in explore_operators
        for i in range(50)
    )


def test_acceptance_needs_the_delta_to_clear_the_noise_band() -> None:
    incumbent = _trajectory(10.0, 0.5)

    assert accepts(_trajectory(11.0, 0.5), incumbent, "maximize")
    assert not accepts(_trajectory(10.3, 0.5), incumbent, "maximize")
    assert accepts(_trajectory(9.0, 0.5), incumbent, "minimize")
    assert accepts(_trajectory(10.3, 0.5), None, "maximize")


def test_simplify_may_replace_a_shorter_program_within_the_band() -> None:
    incumbent = _trajectory(10.0, 0.5)
    shorter, longer = "a = 1\n", "a = 1\nb = 2\n"

    assert accepts(_trajectory(9.8, 0.5), incumbent, "maximize", "simplify", shorter, longer)
    assert not accepts(_trajectory(9.8, 0.5), incumbent, "maximize", "simplify", longer, shorter)
    assert not accepts(_trajectory(9.0, 0.5), incumbent, "maximize", "simplify", shorter, longer)
    assert not accepts(_trajectory(9.8, 0.5), incumbent, "maximize", "refine", shorter, longer)


def test_mechanism_line_and_tables_and_compaction() -> None:
    script = "# MECHANISM: degree-weighted reverse reachable coverage\nclass S: pass\n"
    assert mechanism_of(script) == "degree-weighted reverse reachable coverage"
    assert mechanism_of("class S: pass") == ""

    history = [
        {"iteration": 1, "operator": "seed", "reward": 10.0, "accepted": True, "mechanism": "m1", "hint": ""},
        {"iteration": 2, "operator": "refine", "reward": 10.2, "delta": 0.2, "accepted": False, "mechanism": "m2", "hint": "bigger radius helps"},
        {"iteration": 3, "operator": "crossover", "reward": None, "error": "NameError: x", "mechanism": ""},
    ]
    table = build_attempts_table(history, "maximize")
    assert "2 | refine | 10.200 | +0.200 | no | m2 | bigger radius helps" in table
    assert "FAILED: NameError: x" in table

    population = [
        {"iteration": 1, "reward": 10.0, "plan_seconds": 1.5, "script": "a\nb\n", "mechanism": "m1"},
        {"iteration": 2, "reward": 12.0, "plan_seconds": 0.5, "script": "a\n", "mechanism": "m2"},
    ]
    rows = build_population_table(population, "maximize").splitlines()
    assert rows[1].startswith("2 | 12.000 | 0.5 | 1 | m2")

    compact = compact_turn(script, "class S: pass\n", "refine: reward=10.000, ACCEPTED")
    assert compact.startswith("# refine: reward=10.000, ACCEPTED")
    assert "```diff" in compact and "+# MECHANISM" in compact
    assert compact_turn(script, None, "seed").endswith("```python\n" + script + "\n```")


def test_reflection_and_idea_replies_are_parsed_leniently() -> None:
    hint, memory = parse_reflection('Sure: {"hint": "prefer  hubs", "memory": "rule one"}', "old")
    assert (hint, memory) == ("prefer hubs", "rule one")
    assert parse_reflection("not json", "old") == ("", "old")

    ideas, chosen = parse_ideas('{"ideas": ["a", "b", "c"], "novelty": [1, 9, 4], "chosen": 1}')
    assert ideas == ["a", "b", "c"] and chosen == "b"
    assert parse_ideas('{"ideas": ["a"], "chosen": 7}') == (["a"], "a")
    assert parse_ideas("garbage") == ([], "")


def test_evolve_prompt_carries_the_search_state_for_every_operator() -> None:
    parent = {"iteration": 1, "reward": 10.0, "script": "# MECHANISM: p\n", "summary": "diag", "mechanism": "p"}
    partner = {"iteration": 2, "reward": 9.0, "script": "# MECHANISM: q\n", "summary": "diag2", "mechanism": "q"}
    for operator in ("refine", "parameters", "simplify", "crossover", "synthesize", "from_scratch"):
        text = build_evolve_prompt(
            operator, parent, [partner], attempts="ATT", population="POP", memory="MEM",
            idea="an idea", partner=partner, everyone=[parent, partner],
        )
        assert "ATT" in text and "POP" in text and "MEM" in text and "# MECHANISM:" in text
        assert operator.replace("_", " ").upper() in text or operator.upper() in text
    assert "PARTNER" in build_evolve_prompt("crossover", parent, [], partner=partner)
    assert "CHOSEN IDEA" in build_evolve_prompt("from_scratch", None, [], idea="an idea")
    assert "PARENT" not in build_evolve_prompt("from_scratch", None, [], idea="an idea")


def test_conversation_compacts_only_the_last_assistant_turn():
    class Provider:
        def complete(self, messages):
            return "```python\nclass S: pass\n```"

    class Agent:
        provider = Provider()

        def generate(self, messages):
            return "reply", "class S: pass"

    conversation = Conversation(Agent(), "system")
    conversation.send("task")
    conversation.compact_last("# verdict")
    assert conversation.messages[-1] == {"role": "assistant", "content": "# verdict"}
    assert conversation.transcript[-1]["reply"] == "reply"
