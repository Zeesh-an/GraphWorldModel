"""
The search protocol under a stochastic evaluator: the paired band, the
noise-aware population ranking, the spurious-accept ledger, the unbiased
incumbent curve, edit calibration, counterexamples, the probe turn and its new
ops, and the rediscovery distance. The end-to-end case drives `EvolveSearch`
with a scripted provider on the exact oracle so every piece is exercised on
real trajectories rather than hand-built dicts.
"""

import json
import numpy as np
import pytest

from coding_agent.agent import CodingAgent, Conversation
from coding_agent.counterexamples import describe_counterexamples
from coding_agent.envs.monte_carlo_env import MonteCarloEnvironment
from coding_agent.envs.world_model_env import WorldModelEnvironment
from coding_agent.methods.base import accepts, paired_delta
from coding_agent.methods.evolve import EvolveSearch, choose_operator, explore_operators
from coding_agent.probes import available_ops, execute_probes, parse_probe_request
from coding_agent.prompts import (
    build_attempts_table,
    build_population_table,
    build_probe_turn_prompt,
    probe_contract_for,
)
from coding_agent.provenance import (
    budgeted_items,
    event_f1,
    forecast_similarity,
    jaccard,
    library_pool,
    order_correlation,
    static_calls,
)
from coding_agent.search_metrics import (
    acceptance_ledger,
    calibration_metrics,
    expected_of,
    flat_generations,
    miscalibrated,
    optimism_summary,
    paired_band,
    paired_strengths,
)
from coding_agent.types import ActionOp, GraphInfo, TaskSpec, Trajectory


def _trajectory(
    reward: float,
    se: float,
    samples: list | None = None,
    actions: list | None = None,
    cost: dict | None = None,
) -> Trajectory:
    return Trajectory(
        states=[],
        actions=actions or [],
        reward=reward,
        infected_counts=[],
        cost=cost or {"reward_se": se},
        sample_rewards=samples,
    )


def _graph() -> GraphInfo:
    # A star (0 to 1..5) joined to a path (5..11): degrees differ, so the
    # highest-degree program and the lowest-degree one produce different plans
    pairs = [(0, node) for node in range(1, 6)] + [(node, node + 1) for node in range(5, 11)]
    arcs = pairs + [(destination, source) for source, destination in pairs]
    edge_index = np.array(arcs, dtype=np.int64).T

    return GraphInfo(
        num_nodes=12,
        edge_index=edge_index,
        ic_probs=np.full(edge_index.shape[1], 0.5, dtype=np.float32),
        directed=False,
    )


def _task(**overrides) -> TaskSpec:
    fields = {"task": "influence_maximization", "budget": 2, "horizon": 3, "seed": 1}
    fields.update(overrides)

    return TaskSpec(**fields)


def _plan(nodes: list[int], horizon: int) -> list:
    return [[ActionOp("add_node", node) for node in nodes]] + [[] for _ in range(horizon)]


def test_paired_band_is_the_standard_error_of_the_paired_difference() -> None:
    incumbent = _trajectory(10.0, 1.0, [8.0, 12.0, 9.0, 11.0, 10.0])
    # Positively correlated samples: the realization noise cancels
    lifted = _trajectory(10.5, 1.0, [8.5, 12.5, 9.5, 11.5, 10.5])
    band, rule = paired_band(lifted, incumbent)
    assert rule == "paired" and band == pytest.approx(0.0)
    assert accepts(lifted, incumbent, "maximize")

    # Anti-correlated samples: the paired band is LARGER than either marginal SE
    swung = _trajectory(10.5, 1.0, [12.5, 8.5, 11.5, 9.5, 10.5])
    band, rule = paired_band(swung, incumbent)
    assert rule == "paired" and band > 1.0
    assert not accepts(swung, incumbent, "maximize")

    # No aligned samples: the marginal rule survives as the fallback
    band, rule = paired_band(_trajectory(11.0, 0.5), _trajectory(10.0, 0.7))
    assert (band, rule) == (0.7, "marginal")
    band, rule = paired_band(_trajectory(11.0, 0.5, [1.0, 2.0]), _trajectory(10.0, 0.7, [1.0]))
    assert rule == "marginal"


def test_the_prose_verdict_and_the_acceptance_rule_read_one_band() -> None:
    incumbent = _trajectory(10.0, 1.0, [8.0, 12.0, 9.0, 11.0, 10.0])
    lifted = _trajectory(10.5, 1.0, [8.5, 12.5, 9.5, 11.5, 10.5])
    text = paired_delta(lifted, incumbent, sense="maximize")
    assert "real improvement" in text
    swung = _trajectory(10.5, 1.0, [12.5, 8.5, 11.5, 9.5, 10.5])
    assert "INSIDE THE NOISE" in paired_delta(swung, incumbent, sense="maximize")


def test_paired_strengths_are_the_chain_sums_on_a_tree() -> None:
    population = [
        {"iteration": 1, "reward": 10.0, "compared_to": None, "paired_delta": None, "band": 0.0},
        # accepted: +2 against the seed on generation 2's realization
        {"iteration": 2, "reward": 13.0, "compared_to": 1, "paired_delta": 2.0, "band": 0.5},
        # rejected, but scored on a generous realization: raw reward above the
        # incumbent's, paired delta below it
        {"iteration": 3, "reward": 15.0, "compared_to": 2, "paired_delta": -1.0, "band": 0.5},
    ]
    strengths = paired_strengths(population)
    assert strengths == pytest.approx({1: 10.0, 2: 12.0, 3: 11.0})
    # A record without a comparison keeps its raw reward
    population.append({"iteration": 4, "reward": 9.0})
    assert paired_strengths(population)[4] == 9.0


def test_acceptance_ledger_counts_what_the_naive_rule_would_have_taken() -> None:
    history = [
        {"iteration": 1, "reward": 10.0, "accepted": True},
        {"iteration": 2, "reward": 10.2, "delta": 0.2, "accepted": False, "naive_accepted": True, "band_rule": "paired"},
        {"iteration": 3, "reward": 12.0, "delta": 2.0, "accepted": True, "naive_accepted": True, "band_rule": "paired"},
        {"iteration": 4, "reward": 11.9, "delta": -0.1, "accepted": True, "naive_accepted": False, "band_rule": "paired"},
        {"iteration": 5, "reward": None, "error": "boom"},
    ]
    ledger = acceptance_ledger(history)
    assert ledger["compared_generations"] == 3
    assert ledger["naive_accepts"] == 2 and ledger["band_accepts"] == 2
    assert ledger["lucky_accepts_prevented"] == 1 and ledger["ties_kept"] == 1
    assert ledger["band_rules"] == ["paired"]


def test_expected_line_parsing_and_calibration() -> None:
    assert expected_of("# MECHANISM: m\n# EXPECTED: +3.5 p=0.7\nclass S: pass") == (3.5, 0.7)
    assert expected_of("# EXPECTED: -2\n") == (-2.0, None)
    assert expected_of("# EXPECTED: 1.0 p=1.7\n") == (1.0, 1.0)
    assert expected_of("class S: pass") == (None, None)

    history = [
        {"iteration": 2, "delta": 1.0, "predicted_delta": 1.2, "predicted_clear_p": 0.9, "accepted": True},
        {"iteration": 3, "delta": -0.5, "predicted_delta": 0.4, "predicted_clear_p": 0.6, "accepted": False},
        {"iteration": 4, "delta": 2.0, "predicted_delta": 1.5, "predicted_clear_p": 0.8, "accepted": True},
        {"iteration": 5, "delta": -1.0, "predicted_delta": -0.8, "accepted": False},
    ]
    report = calibration_metrics(history)
    assert report["n"] == 4 and report["n_with_probability"] == 3
    assert report["sign_accuracy"] == pytest.approx(0.75)
    assert report["pearson_r"] > 0.8
    assert report["brier"] == pytest.approx(((0.9 - 1) ** 2 + (0.6 - 0) ** 2 + (0.8 - 1) ** 2) / 3)
    assert calibration_metrics([]) == {"n": 0}

    wrong = [{"delta": -1.0, "predicted_delta": 1.0}] * 3
    assert miscalibrated(wrong) and not miscalibrated(wrong[:2])
    assert not miscalibrated(history)
    # Steering adds one stalled generation's worth of exploration mass
    draws = [choose_operator(np.random.default_rng(i), 15, 20, 0, 5, 0, True) for i in range(300)]
    plain = [choose_operator(np.random.default_rng(i), 15, 20, 0, 5, 0, False) for i in range(300)]
    assert sum(op in explore_operators for op in draws) > sum(op in explore_operators for op in plain)


def test_optimism_summary_and_flat_generations() -> None:
    history = [
        {"iteration": 2, "incumbent_reported": 12.0, "incumbent_unbiased": 11.0, "band": 0.5},
        {"iteration": 3, "incumbent_reported": 12.0, "incumbent_unbiased": 11.2, "band": 0.5},
        {"iteration": 4, "incumbent_reported": 14.0, "incumbent_unbiased": 12.5, "band": 0.5},
        {"iteration": 5, "incumbent_reported": 14.0, "incumbent_unbiased": 12.6, "band": 0.5},
        {"iteration": 6, "incumbent_reported": 14.0, "incumbent_unbiased": 12.4, "band": 0.5},
    ]
    summary = optimism_summary(history, "maximize")
    assert summary["n"] == 5 and summary["final"] == pytest.approx(1.6)
    assert summary["max"] == pytest.approx(1.6) and summary["mean"] == pytest.approx(1.26)
    # Under minimize the same numbers are pessimism
    assert optimism_summary(history, "minimize")["final"] == pytest.approx(-1.6)
    # Generation 4 improved beyond the band; 5 and 6 did not
    assert flat_generations(history, "maximize") == 2
    assert optimism_summary([], "maximize") == {"n": 0}


def test_counterexamples_name_the_realizations_a_candidate_lost() -> None:
    graph = _graph()
    task = _task()
    candidate = _trajectory(5.0, 0.5, [4.0, 6.0, 5.0], actions=_plan([9, 11], 3))
    incumbent = _trajectory(7.0, 0.5, [8.0, 6.0, 7.0], actions=_plan([0, 5], 3))
    candidate_sets = [{9, 11}, {9, 10, 11, 8, 7, 6}, {9, 11, 10}]
    incumbent_sets = [{0, 1, 2, 3, 4, 5, 6, 7}, {0, 5, 6, 4, 3, 2}, {0, 5, 1, 2, 3, 4, 6}]
    text = describe_counterexamples(candidate, incumbent, candidate_sets, incumbent_sets, graph, task)
    assert text.startswith("COUNTEREXAMPLES")
    assert "realization 0: yours 4 vs incumbent 8" in text
    assert "the incumbent reached 8 nodes you did not" in text
    assert "your seeds [9, 11] produced no new infections" in text
    # Realization 1 was a win, so only the two losses are listed
    assert "realization 1:" not in text
    assert describe_counterexamples(_trajectory(5.0, 0.5), incumbent, None, None, graph, task) is None

    # The inverse family: the counterexample is an instance
    recover = _task(task="source_localization", objective_kind="recover", sense="maximize")
    assert recover.recovers
    candidate = _trajectory(
        -0.3, 0.1, [-0.5, -0.1],
        cost={"reward_se": 0.1, "per_instance": [
            {"episode_id": "ep7", "over": [[3, 0.4, 2]], "under": [[9, -0.5, 1]]},
            {"episode_id": "ep8", "over": [], "under": []},
        ]},
    )
    incumbent = _trajectory(
        -0.2, 0.1, [-0.2, -0.2],
        cost={"reward_se": 0.1, "per_instance": [{"predicted": [4, 1]}, {"predicted": [2]}]},
    )
    text = describe_counterexamples(candidate, incumbent, None, None, graph, recover)
    assert "episode ep7: yours -0.5000 vs incumbent -0.2000" in text
    assert "over-explain [3]" in text and "under-explain [9]" in text
    assert "the incumbent named [1, 4]" in text


def test_probe_ops_on_the_oracle_environment() -> None:
    graph = _graph()
    task = _task()
    environment = WorldModelEnvironment.oracle(graph, "IC", n_samples=8, base_seed=0)
    from functools import partial

    from coding_agent.credit import planned_action

    trajectory = environment.rollout(partial(planned_action, _plan([0, 11], 3)), 3, 2)
    assert len(trajectory.sample_rewards) == 8
    assert len(environment.last_sample_final_infected) == 8

    probes, note = parse_probe_request(
        '```probes\n{"probes": [{"op": "drop", "node": 0}, {"op": "swap", "a": 11, "b": 5}, '
        '{"op": "add", "node": 5}, {"op": "best_swap", "node": 11}, {"op": "overlap", "a": 0, "b": 11}, '
        '{"op": "horizon", "t": 6}, {"op": "frontier", "t": 1}, {"op": "robust", "sigma": 0.3}, '
        '{"op": "robust", "mode": "hidden"}, {"op": "gradient", "top": 3}, {"op": "region", "nodes": [1, 2]}, '
        '{"op": "bogus"}]}\n```'
    )
    assert note.startswith("only the first 6")
    probes = probes + [
        {"op": "horizon", "t": 6}, {"op": "frontier", "t": 1}, {"op": "robust", "sigma": 0.3},
        {"op": "robust", "mode": "hidden"}, {"op": "gradient", "top": 3}, {"op": "region", "nodes": [1, 2]},
        {"op": "bogus"}, {"op": "add", "node": 0},
    ]
    text, entries = execute_probes(environment, trajectory, probes, task, graph, note=note)
    answers = {entry["probe"].get("op"): entry["answer"] for entry in entries}
    assert "contribution=" in answers["drop"] and "delta=" in answers["swap"]
    assert "marginal gain=" in text and "best replacement" in answers["best_swap"]
    assert "expected nodes both reach=" in answers["overlap"]
    assert "at the task horizon 3" in answers["horizon"] and "curve=" in answers["horizon"]
    assert "expected frontier size after step 1=" in answers["frontier"]
    assert "with them hidden from the model" in text and "exp(N(0, 0.30))" in text
    assert "highest-gradient UNSELECTED nodes" in answers["gradient"]
    assert "expected mass captured=" in answers["region"]
    assert "unknown or unavailable probe op 'bogus'" in answers["bogus"]
    assert "already in the plan" in text
    assert all(entry["rollouts"] >= 0 and "seconds" in entry for entry in entries)
    # Nothing a probe touched leaked into the environment
    assert environment.hide_edge_weights is False and environment.capture_frontier_step is None
    assert environment.rollout(partial(planned_action, _plan([0, 11], 3)), 3, 2).reward == trajectory.reward

    refused, _ = execute_probes(environment, trajectory, [{"op": "drop", "node": 0}], task, graph, enabled=False)
    assert "native condition" in refused
    assert available_ops(_task(task="cascade_prediction", observational=True, objective_kind="forecast")) == ()


def test_gradient_ranks_an_unselected_hub_above_a_leaf() -> None:
    graph = _graph()
    environment = WorldModelEnvironment.oracle(graph, "IC", n_samples=4, base_seed=0)
    report = environment.seed_gradient(_plan([10, 11], 3), 3, top=3)
    best = [node for node, _ in report["best_unselected"]]
    # The star centre is the node whose seeding would add the most expected spread
    assert best[0] == 0
    assert report["mean_field_spread"] > 2.0
    with pytest.raises(ValueError):
        environment.seed_gradient([[ActionOp("remove_node", 1)]], 3)


def test_monte_carlo_environment_carries_per_sample_rewards() -> None:
    graph = _graph()
    environment = MonteCarloEnvironment(graph, "IC", mc_runs=5, base_seed=0)
    from functools import partial

    from coding_agent.credit import planned_action

    environment.capture_frontier_step = 0
    trajectory = environment.rollout(partial(planned_action, _plan([0], 3)), 3, 1)
    assert len(trajectory.sample_rewards) == 5
    assert trajectory.reward == pytest.approx(np.mean(trajectory.sample_rewards))
    assert len(environment.last_sample_final_infected) == 5
    assert environment.last_frontier_marginals is not None and environment.capture_frontier_step is None
    # Same seed, same realizations: the paired band of a plan against itself is zero
    again = environment.rollout(partial(planned_action, _plan([0], 3)), 3, 1)
    assert paired_band(again, trajectory) == (0.0, "paired")


def test_provenance_similarities_and_static_calls() -> None:
    task = _task()
    assert jaccard([1, 2, 3], [2, 3, 4]) == pytest.approx(0.5) and jaccard([], []) == 1.0
    assert order_correlation([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert order_correlation([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert order_correlation([1, 2], [1, 2]) is None
    assert event_f1({1: [0, None], 2: [1, 1]}, {1: [0, None], 2: [2, 1]}) == pytest.approx(0.5)
    similarity = forecast_similarity([10.0, 20.0, None, 40.0], [11.0, 18.0, 5.0, 40.0])
    assert similarity["n"] == 3 and 0 < similarity["similarity"] <= 1.0
    assert similarity["within_10pct"] == pytest.approx(2 / 3)
    assert budgeted_items(_plan([3, 1, 3], 2), task) == [3, 1]
    script = "seeds = algorithms.degree_discount(graph, budget, 'IC')\nx = imm(graph, 1, 'IC')\nfoo()"
    assert static_calls(script, ["degree_discount", "imm", "foo"]) == ["degree_discount", "imm", "foo"]
    pool = library_pool(task)
    assert "degree_discount" in pool and "celf" not in pool
    assert library_pool(_task(rounds=2)) == []


def test_prompt_pieces_for_the_probe_turn_and_the_forecast_line() -> None:
    task = _task()
    contract = probe_contract_for(task)
    assert "gradient" in contract and "best_swap" in contract and "PROBE TURN" in contract
    recover = _task(task="source_localization", objective_kind="recover", sense="maximize")
    assert "resimulate" in probe_contract_for(recover) and "gradient" not in probe_contract_for(recover)
    assert probe_contract_for(_task(task="cascade_prediction", observational=True, objective_kind="forecast")) == ""

    parent = {"iteration": 3, "reward": 7.5, "mechanism": "hubs", "summary": "diag"}
    turn = build_probe_turn_prompt(parent, task, "refine")
    assert "PROBE TURN" in turn and "iteration 3" in turn and "NONE" in turn and "```probes" in turn

    history = [
        {"iteration": 1, "operator": "seed", "reward": 10.0, "accepted": True, "mechanism": "m1", "hint": ""},
        {"iteration": 2, "operator": "refine", "reward": 10.2, "delta": 0.2, "accepted": False, "mechanism": "m2",
         "hint": "", "predicted_delta": 1.5, "predicted_clear_p": 0.8},
    ]
    table = build_attempts_table(history, "maximize")
    assert "you expected" in table.splitlines()[0]
    assert table.splitlines()[2].endswith("| +1.500 (p=0.80)")

    population = [
        {"iteration": 1, "reward": 10.0, "plan_seconds": 1.0, "script": "a\n", "mechanism": "m1", "strength": 10.0},
        {"iteration": 2, "reward": 15.0, "plan_seconds": 1.0, "script": "a\n", "mechanism": "m2", "strength": 9.0},
    ]
    rows = build_population_table(population, "maximize").splitlines()
    assert "paired estimate" in rows[0] and rows[1].startswith("1 | 10.000 | 10.000")


def test_conversation_aside_leaves_no_trace_in_the_thread() -> None:
    class Provider:
        def complete(self, messages):
            return f"seen {len(messages)} messages"

    class Agent:
        provider = Provider()

    conversation = Conversation(Agent(), "system")
    conversation.messages.append({"role": "user", "content": "task"})
    reply = conversation.aside("question", kind="probe")
    assert reply == "seen 3 messages"
    assert len(conversation.messages) == 2
    assert conversation.transcript[-1]["kind"] == "probe"


seed_script = """\
# MECHANISM: highest degree first
class S(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        nodes = sorted(range(graph.num_nodes), key=lambda n: -graph.degree(n))[:budget]
        return [[ActionOp("add_node", n) for n in nodes]] + [[] for _ in range(horizon)]
"""
worse_script = """\
# MECHANISM: lowest degree first
# EXPECTED: +2.0 p=0.8
class S(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        nodes = sorted(range(graph.num_nodes), key=lambda n: graph.degree(n))[:budget]
        return [[ActionOp("add_node", n) for n in nodes]] + [[] for _ in range(horizon)]
"""
better_script = """\
# MECHANISM: the hub plus the far end of the path
# EXPECTED: +1.0 p=0.6
class S(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        return [[ActionOp("add_node", 0), ActionOp("add_node", 11)]] + [[] for _ in range(horizon)]
"""


class _ScriptedProvider:
    """Answers the probe turn with probes, idea searches with JSON, and generation turns with scripts."""

    def __init__(self) -> None:
        self.scripts = [seed_script, worse_script, better_script, worse_script]
        self.calls = []

    def complete(self, messages: list[dict]) -> str:
        prompt = messages[-1]["content"]
        self.calls.append(prompt[:40])
        if prompt.startswith("PROBE TURN before"):
            return '```probes\n{"probes": [{"op": "drop", "node": 0}, {"op": "gradient", "top": 2}]}\n```'
        if '"ideas"' in prompt:
            return json.dumps({"ideas": ["a new idea"], "novelty": [8], "chosen": 0, "why": "x"})
        script = self.scripts.pop(0) if self.scripts else better_script

        return f"```python\n{script}\n```"


def test_evolve_end_to_end_records_the_whole_protocol() -> None:
    graph = _graph()
    task = _task(seed=3)
    environment = WorldModelEnvironment.oracle(graph, "IC", n_samples=16, base_seed=0)
    provider = _ScriptedProvider()
    method = EvolveSearch(
        outer_iters=4,
        strategy_mode="free",
        use_anchor=False,
        credit=False,
        reflect=False,
        probes=True,
        probe_turn=True,
    )
    strategy, trajectory = method.optimize(CodingAgent(provider), environment, task, graph)

    scored = [entry for entry in method.history if entry.get("reward") is not None]
    assert len(scored) == 4
    compared = [entry for entry in scored if entry.get("delta") is not None]
    assert compared and all(entry["band_rule"] == "paired" for entry in compared)
    # The parent is the incumbent, never the luckiest raw reward
    assert all(entry["parent_iteration"] == entry["compared_to"] for entry in compared)
    # The forecast line was parsed and scored
    assert any(entry.get("predicted_delta") == 2.0 for entry in compared)
    assert method.calibration["n"] >= 2 and "sign_accuracy" in method.calibration
    # The ledger, the unbiased curve and the strengths all exist
    assert method.ledger["compared_generations"] == len(compared)
    assert method.optimism["n"] == len(compared)
    assert all(entry.get("incumbent_unbiased") is not None for entry in compared)
    # The probe turn fired on every generation that had an incumbent
    turns = [entry for entry in method.probe_log if entry.get("turn") == "probe"]
    assert len(turns) == 2 * len(compared)
    assert any("highest-gradient" in entry["answer"] for entry in turns)
    assert sum("PROBE TURN" in call for call in provider.calls) == len(compared)
    # Counterexamples reached the candidate's own summary when it lost
    lost = [entry for entry in compared if not entry["accepted"]]
    assert lost
    assert trajectory.sample_rewards is not None and strategy.source_script


def test_provenance_runs_the_library_pool_on_the_winner() -> None:
    from functools import partial

    from coding_agent.provenance import compute_provenance
    from coding_agent.run import ExperimentConfig, canned_baseline_script

    graph = _graph()
    task = _task()
    config = ExperimentConfig(task="influence_maximization", budget=2, horizon=3, diffusion_model="IC", seed=1)
    winner = _trajectory(5.0, 0.5, actions=_plan([0, 11], 3))
    script = "seeds = degree_discount(graph, budget, 'IC')\n"
    block = compute_provenance(
        task, graph, winner, script,
        partial(canned_baseline_script, config, outbreak=(), batches=None),
        timeout=30.0,
    )
    assert block["pool_size"] > 5 and block["nearest"] is not None
    assert 0.0 <= block["nearest_similarity"] <= 1.0
    assert block["calls"] == ["degree_discount"]
    # Every member either ranked or failed with a recorded reason, never silently
    assert len(block["ranking"]) + len(block["errors"]) == block["pool_size"]
    names = {entry["name"] for entry in block["ranking"]}
    assert "high_degree" in names
    hub_first = next(entry for entry in block["ranking"] if entry["name"] == "high_degree")
    assert hub_first["common"] >= 1


def test_train_stage_refuses_a_checkpoint_trained_under_another_conditioning(tmp_path) -> None:
    import pytest as _pytest

    from pipeline.run import check_checkpoint_conditioning
    from world_model.checkpoint import ModelSpec, build_model, save_checkpoint

    spec = ModelSpec(backbone="sage", head="structured", diffusion_model="IC", hidden_dim=16, n_layers=2)
    path = tmp_path / "wm_sage_IC.pt"
    save_checkpoint(build_model(spec), path, spec)
    check_checkpoint_conditioning(path, "none")
    with _pytest.raises(ValueError, match="trained with --action-conditioning 'none'"):
        check_checkpoint_conditioning(path, "message")

    conditioned = ModelSpec(
        backbone="sage", head="structured", diffusion_model="IC", hidden_dim=16, n_layers=2,
        action_conditioning="message",
    )
    path = tmp_path / "wm_sage_IC_message.pt"
    save_checkpoint(build_model(conditioned), path, conditioned)
    check_checkpoint_conditioning(path, "message")
    with _pytest.raises(ValueError):
        check_checkpoint_conditioning(path, "global")


def test_best_swap_cap_scales_with_the_graph_and_swap_rebuilds_a_deletion_bag() -> None:
    from coding_agent.probes import _swap, best_swap_limit

    graph = _graph()
    environment = WorldModelEnvironment.oracle(graph, "IC", n_samples=8, base_seed=0)
    assert best_swap_limit(environment, graph) == 200

    class Huge:
        edge_index = np.zeros((2, 2_600_000), dtype=np.int64)

    # At the ladder's 200 samples a 2.6M-arc graph sits at the floor
    ladder = WorldModelEnvironment.oracle(graph, "IC", n_samples=200, base_seed=0)
    assert best_swap_limit(ladder, Huge()) == 10
    assert 10 < best_swap_limit(environment, Huge()) < 200
    sampler = MonteCarloEnvironment(graph, "IC", mc_runs=2, base_seed=0)
    assert best_swap_limit(sampler, graph) == 20
    assert best_swap_limit(sampler, Huge()) == 3

    containment = TaskSpec(
        task="critical_node_detection", objective_kind="minimize", sense="minimize",
        budget_op="remove_node", remove_semantics="blocked", allowed_ops=("remove_node",),
        budget=1, horizon=2,
    )
    assert containment.contains
    from coding_agent.containment import expand_removals

    plan = [expand_removals([ActionOp("remove_node", 0)], graph), [], []]
    swapped = _swap(plan, 0, 11, containment, graph)
    targets = {(action.op, int(action.target), action.destination) for action in swapped[0]}
    assert ("remove_node", 11, None) in targets and ("remove_node", 0, None) not in targets
    # The new node's own arcs, not the old node's retargeted ones
    assert all(11 in (int(a.target), int(a.destination)) for a in swapped[0] if a.op == "remove_edge")
    assert any(a.op == "remove_edge" for a in swapped[0])
