"""
Condition 9 plumbing: the program wrapper, the fitness orientation, the parsers
that read each framework's best program back, and the arm/registry rules.

None of this needs a framework installed or an LLM: every framework's output
layout is reproduced from the survey of its source (research_notes/
algorithm_discovery_baselines.md) as a fixture directory.
"""

import json

import numpy as np
import pytest

from baselines import discovery
from baselines.discovery import (
    adapters,
    any_task,
    build_context,
    contracts,
    oriented_fitness,
    problem_statement,
    program_script,
    run_uid,
)
from baselines.registry import available_baselines, external_baselines
from coding_agent.executor import StrategyError, build_strategy
from coding_agent.prediction import CascadeObservation
from coding_agent.reconstruction import Observation
from coding_agent.run import ExperimentConfig
from coding_agent.types import GraphInfo
from pipeline.conditions import (
    discovery_condition,
    expand_llm_models,
    monte_carlo,
    parse_arm,
)
from pipeline.run import expand_baselines


@pytest.fixture()
def graph(edge_index) -> GraphInfo:
    return GraphInfo(
        num_nodes=6,
        edge_index=edge_index,
        ic_probs=np.full(edge_index.shape[1], 0.5, dtype=np.float32),
        directed=False,
    )


def _strategy(task: str, budget_op: str = "add_node", lever: str | None = None, outbreak=()):
    script = program_script(contracts[task].initial_program, task, budget_op, lever)
    strategy = build_strategy(script)
    strategy.outbreak = tuple(outbreak)
    strategy.budget_op = budget_op

    return strategy


def test_seed_wrapper_emits_budget_distinct_add_node_ops(graph):
    plan = _strategy("influence_maximization").plan_horizon(graph, 2, 3)

    assert len(plan) == 4
    assert all(op.op == "add_node" for op in plan[0])
    # Degree ranking on the fixture: nodes 1 and 4 have degree 3
    assert sorted(op.target for op in plan[0]) == [1, 4]
    assert plan[1:] == [[], [], []]


def test_removal_wrapper_never_removes_an_outbreak_source(graph):
    plan = _strategy(
        "critical_node_detection", "remove_node", outbreak=(1,)
    ).plan_horizon(graph, 2, 1)

    targets = [op.target for op in plan[0]]
    assert 1 not in targets
    assert len(targets) == 2
    assert all(op.op == "remove_node" for op in plan[0])


def test_blocking_wrapper_switches_to_arcs_under_an_edge_lever(graph):
    plan = _strategy(
        "influence_blocking", "remove_edge", "edge_block", outbreak=(0,)
    ).plan_horizon(graph, 2, 1)

    assert [(op.op, op.target, op.destination) for op in plan[0]] == [
        ("remove_edge", 0, 1)
    ]


def test_epidemic_wrapper_zeroes_the_weight_under_contact_reduction(graph):
    plan = _strategy(
        "epidemic_control", "set_edge_weight", "contact_reduce", outbreak=(0,)
    ).plan_horizon(graph, 1, 1)

    op = plan[0][0]
    assert (op.op, op.target, op.destination, op.weight) == ("set_edge_weight", 0, 1, 0.0)


def test_localize_wrapper_ranks_by_observation_times_degree(graph):
    observation = np.array([0.0, 1.0, 1.0, 0.0, 0.5, 0.0])

    assert _strategy("source_localization").localize(graph, observation, 2) == [1, 2]


def test_reconstruct_wrapper_round_trips_the_history(graph):
    observation = Observation(reported={1: 0, 2: 1, 4: None}, horizon=5, num_nodes=6)
    history = _strategy("cascade_reconstruction").reconstruct(graph, observation, 5)

    assert history == {1: (0, None), 2: (1, 1), 4: (1, 1)}


def test_predict_wrapper_returns_a_float(graph):
    observation = CascadeObservation(
        cascade_id="c", root=0, num_nodes=6, adopters={0: 0, 1: 1}, frontier=(1,),
        observed_steps=2, horizon=5,
    )

    assert _strategy("cascade_prediction").predict(graph, observation, 5) == 3.0


def test_wrapper_resolves_the_versioned_name_reevo_and_mcts_ahd_emit(graph):
    program = (
        "import networkx as nx\n\n"
        "def select_seeds_v2(graph, k):\n"
        "    return [5, 0][:k]\n"
    )
    strategy = build_strategy(program_script(program, "influence_maximization"))
    strategy.outbreak = ()

    assert [op.target for op in strategy.plan_horizon(graph, 2, 0)[0]] == [5, 0]


def test_wrapper_names_a_missing_contract_function(graph):
    strategy = build_strategy(program_script("x = 1\n", "influence_maximization"))
    strategy.outbreak = ()

    with pytest.raises(ValueError, match="select_seeds"):
        strategy.plan_horizon(graph, 1, 0)


def test_wrapper_keeps_the_executor_import_whitelist():
    with pytest.raises(StrategyError):
        build_strategy(program_script("import os\n", "influence_maximization"))


def test_oriented_fitness_is_non_negative_and_higher_is_better():
    assert oriented_fitness("influence_maximization", "maximize", 41.5, 100) == 41.5
    assert oriented_fitness("critical_node_detection", "minimize", 30.0, 100) == 70.0
    assert oriented_fitness("cascade_prediction", "minimize", 1.0, 100) == 0.5
    assert oriented_fitness("epidemic_control", "minimize", 500.0, 100) == 0.0


def test_context_round_trips_through_json(graph, tmp_path):
    experiment = ExperimentConfig(
        task="influence_blocking", evaluator=monte_carlo, blocking_lever="edge_block"
    )
    context = build_context(experiment, "influence_blocking", "remove_edge", tmp_path, graph)
    loaded = json.loads(json.dumps(context))
    restored = dict(loaded["experiment"])
    restored["allowed_ops"] = tuple(restored["allowed_ops"])

    assert ExperimentConfig(**restored) == experiment
    assert loaded["lever"] == "edge_block"
    assert loaded["token_env"] == "CHATGPT_GATEWAY_TOKEN"
    assert "select_blockers(graph, k, rumour, lever)" in problem_statement(loaded)
    assert "edge_block" in problem_statement(loaded)


def test_run_uid_is_unique_per_task_dataset_run_and_budget(tmp_path):
    work = tmp_path / "critical_node_detection" / "jazz" / "r1" / "baselines" / "_runs" / "reevo" / "pct10"

    assert run_uid(work) == "d_critical_node_detection_jazz_r1_pct10"


def _work(tmp_path, framework: str):
    work = tmp_path / "t" / "d" / "r" / "baselines" / "_runs" / framework / "k5"
    work.mkdir(parents=True)

    return work


def test_openevolve_parse_reads_best_program(tmp_path):
    work = _work(tmp_path, "openevolve")
    best = work / "out" / "best"
    best.mkdir(parents=True)
    (best / "best_program.py").write_text("def select_seeds(graph, k):\n    return []\n")
    (best / "best_program_info.json").write_text(json.dumps({"iteration": 7, "metrics": {}}))

    program, info = adapters["openevolve"][1](work, "")

    assert "select_seeds" in program
    assert info["best_iteration"] == 7


def test_codeevolve_parse_follows_run_metadata_to_the_island(tmp_path):
    work = _work(tmp_path, "codeevolve")
    out = work / "out"
    (out / "island_1").mkdir(parents=True)
    (out / "island_0").mkdir()
    (out / "island_0" / "best_sol.py").write_text("wrong island\n")
    (out / "island_1" / "best_sol.py").write_text("def select_seeds(graph, k):\n    return [1]\n")
    (out / "run_metadata.json").write_text(
        json.dumps({"10": {"best_sol": {"fitness": 3.0, "island_found": 1, "iteration_found": 4}}})
    )

    program, info = adapters["codeevolve"][1](work, "")

    assert "return [1]" in program
    assert info["island_found"] == 1


def test_driver_parsers_read_best_program_py(tmp_path):
    for name in ("llamea", "eoh", "llm4ad_funsearch", "llm4ad_hillclimb"):
        work = _work(tmp_path, name)
        (work / "best_program.py").write_text("def localize(graph, observation, k):\n    return []\n")
        (work / "best_info.json").write_text(json.dumps({"score": 0.5}))

        program, info = adapters[name][1](work, "")

        assert "localize" in program
        assert info == {"score": 0.5}


def test_reevo_parse_follows_main_log(tmp_path):
    work = _work(tmp_path, "reevo")
    hydra = work / "hydra"
    hydra.mkdir()
    (hydra / "problem_iter3_code2.py").write_text("def select_seeds_v2(graph, k):\n    return [2]\n")
    (hydra / "main.log").write_text(
        "[INFO] Best Code Path Overall: problem_iter3_code2.py\n"
    )

    program, info = adapters["reevo"][1](work, "")

    assert "select_seeds_v2" in program
    assert info["best_path"].endswith("problem_iter3_code2.py")


def test_mcts_ahd_parse_reads_the_last_best_population(tmp_path):
    work = _work(tmp_path, "mcts_ahd")
    hydra = work / "hydra"
    hydra.mkdir()
    (hydra / "best_population_generation_4.json").write_text(json.dumps([{"code": "old"}]))
    (hydra / "best_population_generation_12.json").write_text(
        json.dumps([{"code": "def select_seeds(graph, k):\n    result = [3]\n    return result\n", "objective": -9.0}])
    )

    program, info = adapters["mcts_ahd"][1](work, "")

    assert "result = [3]" in program
    assert info["evaluations"] == 12


def test_deepevolve_parse_takes_main_py_only(tmp_path):
    work = _work(tmp_path, "deepevolve")
    best = work / "problems" / run_uid(work) / "ckpt" / "best"
    best.mkdir(parents=True)
    (best / "main.py").write_text("def predict(graph, observation):\n    return 2.0\n")
    (best / "deepevolve_interface.py").write_text("evolved, ignored\n")
    (best / "best_program_info.json").write_text(json.dumps({"iteration": 3}))

    program, info = adapters["deepevolve"][1](work, "")

    assert "predict" in program and "ignored" not in program
    assert info["iteration"] == 3


def test_every_adapter_is_a_wired_discovery_entry_serving_every_task():
    for name in adapters:
        spec = external_baselines[name]
        assert spec.kind == discovery.discovery
        assert spec.task == any_task
        assert spec.wired and spec.parse_program is not None and spec.parse_seeds is None
        assert spec.serves("cascade_prediction") and spec.serves("epidemic_control")
        assert spec.timeout is not None

    assert set(adapters) <= set(available_baselines(task="source_localization"))
    assert external_baselines["gs4co"].status == "blocked"
    assert external_baselines["llm4ad_next"].status == "blocked"


def test_discovery_arms_parse_to_condition_nine_and_fan_out_per_model():
    arm = parse_arm("discovery:eoh")

    assert (arm.condition, arm.name, arm.external, arm.method, arm.evaluator) == (
        discovery_condition, "discovery_eoh", "eoh", "one_shot", monte_carlo
    )
    assert not arm.is_agent

    fanned = expand_llm_models([arm], ("gpt-5.6-luna", "gpt-5.6-sol"))
    assert [a.name for a in fanned] == ["discovery_eoh+gpt-5.6-luna", "discovery_eoh+gpt-5.6-sol"]
    assert [a.llm_model for a in fanned] == ["gpt-5.6-luna", "gpt-5.6-sol"]


def test_expand_baselines_keeps_discovery_out_of_all_external():
    assert expand_baselines(("discovery:eoh",), "source_localization") == ["discovery:eoh"]

    with pytest.raises(ValueError, match="discovery:eoh"):
        expand_baselines(("external:eoh",), "influence_maximization")

    with pytest.raises(ValueError, match="external:moeim"):
        expand_baselines(("discovery:moeim",), "influence_maximization")

    assert not any(
        "eoh" in spec for spec in expand_baselines(("all-external",), "influence_maximization")
    )
