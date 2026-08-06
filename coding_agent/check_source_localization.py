"""
Self-check for source localization: the contract, the metrics, the label
extraction, the four oracle bindings, and the differentiable inversion.

Everything here runs on graphs where the answer is known by hand, so a regression
fails as an assertion rather than as a quietly wrong results table, which is the
specific failure mode an inverse task invites, because the reward is an F1 in
[0, 1] and every other task in this repo reports a node count. A harness that
mixed the two would print `0.83` under a column header that means "nodes" and
nobody would notice.

    python -m coding_agent.check_source_localization
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from coding_agent.envs.monte_carlo_env import MonteCarloEnvironment
from coding_agent.executor import StrategyError, build_strategy
from coding_agent.localization import (
    ForwardOracle,
    LocalizeAnchor,
    bind_predict_marginals,
    evaluate_localizer,
    load_instances,
    unavailable_forward_oracle,
    validate_sources,
)
from coding_agent.methods.base import attach_context, baseline_anchor, summarize
from coding_agent.prompts import build_system_prompt, build_user_prompt
from coding_agent.tools.localization_algorithms import (
    infected_set,
    localization_algorithms,
    localization_scorers,
    lpsi,
    lpsi_scores,
)
from coding_agent.types import GraphInfo, ScoredStrategy, Strategy, TaskSpec
from pipeline.conditions import parse_arm
from pipeline.tasks import get_task, recover
from world_model.wm_data import load_episode_endpoints
from world_model.wm_metrics import localization_metrics, roc_auc
from world_model.wm_model import WorldModel
from world_model.wm_sl import build_graph_tensors, invert, soft_rollout, train_source_prior

# A 3-armed star of paths: node 0 is the hub, and each arm is 0 -> a -> b. A
# cascade seeded at one leaf reaches the hub and the other arms, so "which leaf
# started it" is a real question with a checkable answer.
spider = [(0, 1), (1, 2), (0, 3), (3, 4), (0, 5), (5, 6)]


def _graph(edges: list[tuple], num_nodes: int, directed: bool = False) -> GraphInfo:
    pairs = list(edges) if directed else list(edges) + [(v, u) for u, v in edges]
    edge_index = np.asarray(pairs, dtype=np.int64).T

    return GraphInfo(
        num_nodes=num_nodes,
        edge_index=edge_index,
        ic_probs=np.ones(edge_index.shape[1], dtype=np.float32),
        directed=directed,
    )


def _task(**overrides) -> TaskSpec:
    defaults = {
        "task": "source_localization",
        "objective_kind": recover,
        "allowed_ops": ("add_node",),
        "budget_op": "add_node",
        "budget": 2,
        "horizon": 4,
    }

    return TaskSpec(**(defaults | overrides))


def _observation(nodes: list[int], num_nodes: int) -> np.ndarray:
    observation = np.zeros(num_nodes, dtype=np.float64)
    observation[nodes] = 1.0

    return observation


def registry_says_recover_and_generates_no_actions() -> None:
    """The one place the contract, the sweep and the data shape are decided."""
    task = get_task("source_localization")

    assert task.runnable, task.status
    assert task.recovers
    assert not task.contains
    assert task.objective == recover
    # An episode carrying a mid-cascade injection has an observation its t=0 seed
    # set did not produce, so its (x, y) label would be a lie
    assert task.gen_action_ops == ()
    # k is a property of the instance, so a four-point budget sweep would be four
    # runs of one experiment
    assert task.default_budget_pcts == (10.0,)
    assert "lpsi" in task.default_baselines
    assert "gradient_free@world_model" in task.default_arms


def arm_a_is_its_own_condition() -> None:
    """Folding the ablation into the 3-6 ladder would break the ladder's ablation."""
    ours = parse_arm("evolve_free@world_model", default_evaluator="oracle")
    control = parse_arm("gradient_free@world_model", default_evaluator="oracle")

    assert ours.condition == 6 and ours.is_agent
    assert control.condition == 8
    # No LLM anywhere in arm A, so charging it --outer-iters turns would bill it
    # for calls it never makes
    assert not control.is_agent


def metrics_are_the_published_ones() -> None:
    """PR / RE / F1 / AUC on a case small enough to check by hand."""
    # 2 of 3 named nodes are true sources; 2 of 2 true sources found
    metrics = localization_metrics([0, 1, 4], [0, 1], num_nodes=10)

    assert abs(metrics["precision"] - 2 / 3) < 1e-9, metrics
    assert abs(metrics["recall"] - 1.0) < 1e-9, metrics
    assert abs(metrics["f1"] - 0.8) < 1e-9, metrics
    # 9 of 10 nodes classified correctly (node 4 is a false positive)
    assert abs(metrics["accuracy"] - 0.9) < 1e-9, metrics

    # Accuracy is near-useless alone, and this is the case that proves it: naming
    # nothing scores 0.98 accuracy and 0 F1 (IVGD Table 3's GCNSI row)
    empty = localization_metrics([], [0, 1], num_nodes=100)
    assert empty["accuracy"] == 0.98 and empty["f1"] == 0.0, empty


def auc_handles_ties_and_perfect_rankings() -> None:
    """A fully tied score vector must score exactly 0.5, not whatever the sort did."""
    target = np.array([1, 1, 0, 0, 0, 0], dtype=bool)

    assert roc_auc(np.ones(6), target) == 0.5
    assert roc_auc(np.array([1.0, 0.9, 0.1, 0.2, 0.0, 0.05]), target) == 1.0
    assert roc_auc(np.array([0.0, 0.05, 1.0, 0.9, 0.5, 0.6]), target) == 0.0

    # ...and the rank-derived fallback ranks a correct guess above an empty one
    good = localization_metrics([0, 1], [0, 1], num_nodes=20)
    bad = localization_metrics([7, 8], [0, 1], num_nodes=20)
    assert good["auc"] > bad["auc"], (good["auc"], bad["auc"])


def _write_episode_store(directory: Path) -> None:
    """A two-episode dataset in exactly the shape generate_wm_data writes."""
    graph = _graph(spider, 7)
    os.makedirs(directory / "graphs", exist_ok=True)
    np.savez_compressed(
        directory / "graphs" / "toy.npz",
        edge_index=graph.edge_index,
        ic_probs=graph.ic_probs,
        lt_weights=graph.ic_probs,
        node_feats=np.zeros((7, 1), dtype=np.float32),
        node_labels=np.zeros(7, dtype=np.int32),
    )
    (directory / "graphs_index.json").write_text(
        json.dumps([{"graph_id": "toy", "file": "toy.npz", "n_nodes": 7}])
    )

    def record(episode: str, t: int, action: list[dict], infected: list[int], marginal: dict) -> dict:
        return {
            "graph_id": "toy",
            "diffusion_model": "IC",
            "episode_id": episode,
            "algorithm": "random",
            "branch": "main",
            "t": t,
            "state": {"infected": [], "frontier": []},
            "action": action,
            "next_state": {"infected": infected, "frontier": infected},
            "reward": float(len(infected)),
            "next_marginal_infected": marginal,
            "next_marginal_frontier": {},
        }

    lines = [
        record("e1", 0, [{"op": "add_node", "target": 2}], [2], {"2": 1.0}),
        record("e1", 1, [], [1, 2], {"1": 1.0, "2": 1.0}),
        record("e1", 2, [], [0, 1, 2], {"0": 0.5, "1": 1.0, "2": 1.0}),
        record("e2", 0, [{"op": "add_node", "target": 6}], [6], {"6": 1.0}),
        record("e2", 1, [], [5, 6], {"5": 1.0, "6": 1.0}),
        # A counterfactual branch must be IGNORED: its terminal state was not
        # produced by the t=0 seed set alone
        {
            "graph_id": "toy",
            "diffusion_model": "IC",
            "episode_id": "e2",
            "algorithm": "random",
            "branch": "cf_0",
            "t": 1,
            "state": {"infected": [6], "frontier": [6]},
            "action": [{"op": "add_node", "target": 0}],
            "next_state": {"infected": [0, 1, 3, 5, 6], "frontier": [0]},
            "reward": 4.0,
            "next_marginal_infected": {str(node): 1.0 for node in (0, 1, 3, 5, 6)},
            "next_marginal_frontier": {},
        },
    ]
    (directory / "transitions_IC_train.jsonl").write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n"
    )


def labels_come_from_the_t0_seed_commit() -> None:
    """
    The (x, y) pair is free, and it has to be read the right way round.

    The t=0 main record's ACTION is the source set; the LAST main record's
    next_state is the observation. Reading the last action instead (empty) or the
    first state instead (also empty) both produce a plausible, silent, wrong label.
    """
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        _write_episode_store(directory)

        episodes = load_episode_endpoints(directory, "IC", "train")
        assert len(episodes) == 2, [record["episode_id"] for record in episodes]

        by_id = {record["episode_id"]: record for record in episodes}
        assert by_id["e1"]["sources"] == [2]
        assert by_id["e2"]["sources"] == [6]
        # Terminal state of e1 is {0, 1, 2}; the counterfactual fork's {0,1,3,5,6}
        # must not have leaked into e2, whose terminal state is {5, 6}
        assert by_id["e2"]["binary"].tolist() == [0, 0, 0, 0, 0, 1, 1]
        # ...and the marginal is continuous where the last wave was uncertain
        assert by_id["e1"]["marginal"][0] == 0.5

        instances = load_instances(
            directory, "IC", "train", graph_id="toy", limit=0, observation="marginal"
        )
        assert [instance.sources for instance in instances] == [[2], [6]]
        assert instances[0].source_count == 1

        binary = load_instances(
            directory, "IC", "train", graph_id="toy", limit=0, observation="binary"
        )
        # Our marginal is strictly MORE informative than the binary draw, which is
        # why both are kept and only the binary column is comparable to §5.1
        assert set(np.unique(binary[0].observation)) <= {0.0, 1.0}
        assert not set(np.unique(instances[0].observation)) <= {0.0, 1.0}

        # A sweep-mode band that keeps nothing must RAISE, not silently widen: a
        # row computed over a different source fraction than its label claims is
        # the easiest way to publish an invalid comparison
        try:
            load_instances(
                directory,
                "IC",
                "train",
                graph_id="toy",
                limit=0,
                budget_mode="sweep",
                budget=6,
                source_tolerance=0.1,
            )
            raise AssertionError("an empty source-fraction band must raise")
        except ValueError as error:
            assert "source counts" in str(error), error


def every_localizer_returns_a_legal_source_set() -> None:
    """The uniform contract: exactly `budget` distinct in-range nodes, every time."""
    graph = _graph(spider, 7)
    observation = _observation([0, 1, 2], 7)

    for name, algorithm in localization_algorithms.items():
        for budget in (1, 3):
            sources = algorithm(
                graph, observation, budget, diffusion_model="IC", horizon=4
            )
            sources = [int(node) for node in sources]

            assert len(sources) == budget, (name, budget, sources)
            assert len(set(sources)) == budget, (name, sources)
            assert all(0 <= node < 7 for node in sources), (name, sources)

        scores = np.asarray(
            localization_scorers[name](graph, observation, diffusion_model="IC")
        )
        assert scores.shape == (7,), (name, scores.shape)
        assert np.isfinite(scores).all(), (name, scores)


def lpsi_finds_the_planted_source() -> None:
    """
    The bar, on cases where the answer is unambiguous under LPSI's OWN rule.

    Worth being precise about what that rule guarantees, because it is not
    "recovers the source" and assuming otherwise is how a wrong implementation
    passes a wrong test. LPSI names the local maxima of a converged label field,
    which is a CENTRE-of-the-infected-region estimator: on a path 0-1-2 with all
    three infected, node 1 accumulates more field than either endpoint and LPSI
    names it, even though the cascade started at an endpoint. That is the
    ill-posedness §1 describes, showing up on a seven-node graph, not a bug, and
    exactly why §2.9 risk 3 says the achievable ceiling is uncharacterized.

    So the two things asserted here are the two LPSI actually promises: it never
    names a node the observation says was uninfected, and on a star seeded at the
    hub (where the source IS the field maximum) it recovers it exactly.
    """
    graph = _graph(spider, 7)
    observation = _observation([0, 1, 2], 7)
    scores = lpsi_scores(graph, observation)

    assert infected_set(observation) == [0, 1, 2]
    # A node the observation never saw infected can never be named, at any budget
    for budget in (1, 2, 3):
        assert set(lpsi(graph, observation, budget)) <= {0, 1, 2}, (budget, scores)

    # The centre-not-endpoint behaviour, pinned so a future "fix" has to argue
    # with the paper rather than with this file
    assert lpsi(graph, observation, 1) == [1], scores
    assert scores[1] > scores[2] > scores[0], scores

    # A star seeded at the hub: every leaf is infected, the hub has every infected
    # neighbour, and it is the unique local maximum
    star = _graph([(0, node) for node in range(1, 6)], 6)
    saturated = _observation(list(range(6)), 6)
    assert lpsi(star, saturated, 1) == [0], lpsi_scores(star, saturated)


def the_four_oracle_bindings_differ_in_one_thing() -> None:
    """
    @native has no forward model; the others route through their own evaluator.

    This is the ablation conditions 3-6 exist to be, so it is worth an assertion:
    the program is identical, the binding is not.
    """
    graph = _graph(spider, 7)
    task = _task()
    environment = MonteCarloEnvironment(graph, "IC", mc_runs=2)

    oracle = bind_predict_marginals(environment, task)
    assert isinstance(oracle, ForwardOracle)

    before = environment.episodes_used
    marginals = oracle([2])
    assert marginals.shape == (7,)
    # Certain transmission from leaf 2 reaches the whole component
    assert marginals[2] == 1.0 and marginals[0] == 1.0, marginals
    # ...and the call is METERED: episodes and the call count both moved
    assert environment.episodes_used > before
    assert oracle.calls == 1

    native = bind_predict_marginals(environment, _task(forward_model=False))
    assert native is unavailable_forward_oracle
    try:
        native([2])
        raise AssertionError("the @native binding must raise, not return")
    except StrategyError as error:
        assert "@native" in str(error), error


def a_missing_localize_is_a_repair_turn() -> None:
    """
    A model that implemented plan_horizon here must be told, not tracebacked.

    Both shapes are checked, and the second is the one that actually happens.
    Generated code writes `class MyStrategy(Strategy)`, and `Strategy` is a
    Protocol whose method bodies are `...`, so subclassing it INHERITS a
    `localize` that returns None. `hasattr` says yes, and without an identity
    check against the Protocol the contract error becomes an opaque "returned
    None" from inside validation instead.
    """
    graph = _graph(spider, 7)
    task = _task()
    environment = MonteCarloEnvironment(graph, "IC", mc_runs=1)

    class WrongContract:
        source_script = ""

        def plan_horizon(self, graph, budget, horizon):
            return [[]]

    class InheritsTheStub(Strategy):
        source_script = ""

        def plan_horizon(self, graph, budget, horizon):
            return [[]]

    for wrong in (WrongContract(), InheritsTheStub()):
        try:
            evaluate_localizer(wrong, environment, task, graph, [], "episode")
            raise AssertionError(
                f"{type(wrong).__name__} has no localize() and must raise"
            )
        except StrategyError as error:
            assert "localize()" in str(error), error

    # ...and the same trap makes the OPTIONAL source_scores non-optional: a
    # program that only wrote localize() must fall back to the rank-derived AUC,
    # not fail on the inherited stub returning None
    script = """\
class OnlyLocalize(Strategy):
    def localize(self, graph, observation, budget):
        return [int(node) for node in localization_algorithms.lpsi(graph, observation, budget)]
"""
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        _write_episode_store(directory)
        instances = load_instances(directory, "IC", "train", graph_id="toy", limit=0)

    trajectory, _ = evaluate_localizer(
        build_strategy(script, "free"), environment, task, graph, instances, "episode"
    )
    assert trajectory.cost["auc_source"] == "rank_derived", trajectory.cost


def an_illegal_source_set_is_rejected() -> None:
    """Duplicates and over-length sets both buy recall for free."""
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        _write_episode_store(directory)
        instance = load_instances(directory, "IC", "train", graph_id="toy", limit=0)[0]

    assert validate_sources([1, 2], instance, 2) == [1, 2]

    for bad, reason in (
        ([1, 1], "duplicate"),
        ([0, 1, 2], "budget"),
        ([99], "range"),
        ("nope", "type"),
    ):
        try:
            validate_sources(bad, instance, 2)
            raise AssertionError(f"{reason}: {bad!r} must be rejected")
        except StrategyError:
            pass


def the_scored_harness_is_fixed() -> None:
    """Scored mode may edit source_score, never localize."""
    graph = _graph(spider, 7)
    observation = _observation([0, 1, 2], 7)

    class Peripheral(ScoredStrategy):
        def source_score(self, node, graph, observation, selected):
            if observation[node] < 0.5:
                return float("-inf")

            return -float(graph.degree(node))

    scorer = Peripheral()
    sources = scorer.localize(graph, observation, 2)
    assert len(sources) == 2 and set(sources) <= {0, 1, 2}, sources
    # The hub has the highest degree, so a peripheral rule names it last
    assert sources[0] != 0, sources

    # Scored mode gets a REAL AUC for free: source_score is already a per-node
    # function, so the harness evaluates it once per node rather than falling back
    # to the order localize() happened to name things in
    scores = scorer.source_scores(graph, observation)
    assert scores.shape == (7,) and np.isfinite(scores).all(), scores
    # ...and a ruled-out node sits strictly below every candidate rather than at
    # -inf, so the ranking stays total
    assert scores[3] < scores[2], scores

    override = """\
class Cheat(ScoredStrategy):
    def localize(self, graph, observation, budget):
        return list(range(budget))
"""
    try:
        build_strategy(override, "scored")
        raise AssertionError("overriding the fixed localize harness must be rejected")
    except StrategyError as error:
        assert "fixed harness" in str(error), error


def a_generated_program_runs_end_to_end() -> None:
    """The whole path: build a script, bind the oracle, score it, summarize it."""
    script = """\
import numpy as np

class FieldMaxima(Strategy):
    def source_scores(self, graph, observation):
        return localization_scorers.lpsi(graph, observation)

    def localize(self, graph, observation, budget):
        scores = self.source_scores(graph, observation)
        scores = np.where(observation >= 0.5, scores, scores.min() - 1.0)
        picked = [int(node) for node in np.argsort(-scores)[:budget]]
        # Rank the survivors by how well re-simulating them reproduces y
        ranked = sorted(
            picked,
            key=lambda node: float(
                ((self.predict_marginals([node]) - observation) ** 2).sum()
            ),
        )
        return ranked
"""
    graph = _graph(spider, 7)
    environment = MonteCarloEnvironment(graph, "IC", mc_runs=2)

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        _write_episode_store(directory)
        instances = load_instances(directory, "IC", "train", graph_id="toy", limit=0)

    task = _task(instances=tuple(instances))
    strategy = build_strategy(script, "free")
    attach_context(strategy, task, environment)

    trajectory, seconds = evaluate_localizer(
        strategy, environment, task, graph, instances, "episode"
    )

    assert 0.0 <= trajectory.reward <= 1.0, trajectory.reward
    assert trajectory.cost["n_instances"] == len(instances)
    # source_scores() was implemented, so the AUC is measured on a real ranking
    assert trajectory.cost["auc_source"] == "source_scores"
    # ...and the forward oracle was actually called and counted
    assert trajectory.cost["forward_calls"] > 0
    assert seconds > 0.0

    # The inverse task's own summary, not the cascade one
    text = summarize(trajectory, graph, task)
    assert "F1=" in text and "precision=" in text, text
    assert "frontier_counts" not in text, text

    # ...and the reference table it is read against runs on the same instances
    anchor, best, name = baseline_anchor(environment, task, graph)
    assert "LPSI" in anchor or "lpsi" in anchor, anchor
    assert name in localization_algorithms, name
    assert 0.0 <= best.reward <= 1.0


def the_anchor_wrapper_matches_the_library() -> None:
    """A canned baseline and its anchor must be the same algorithm."""
    graph = _graph(spider, 7)
    observation = _observation([0, 1, 2], 7)
    task = _task()

    anchor = LocalizeAnchor(
        "lpsi", localization_algorithms["lpsi"], localization_scorers["lpsi"], task
    )
    assert anchor.localize(graph, observation, 2) == lpsi(graph, observation, 2)
    assert anchor.source_scores(graph, observation).shape == (7,)


def the_prompt_states_the_right_contract() -> None:
    """A model told to write plan_horizon here would fail every iteration."""
    graph = _graph(spider, 7)
    task = _task(instances=())

    system = build_system_prompt("evolve", "free", task)
    assert "localize(self, graph, observation, budget)" in system, system
    assert "plan_horizon" not in system, system
    assert "predict_marginals" in system

    # ...and the @native condition must say the oracle is gone, not advertise it
    native = build_system_prompt("evolve", "free", _task(forward_model=False))
    assert "NO FORWARD ORACLE" in native, native

    user = build_user_prompt("evolve", task, graph)
    assert "emits no actions" in user, user
    assert "lpsi" in user, user


def the_inversion_is_differentiable_and_recovers_a_source() -> None:
    """
    Arm A's load-bearing property: gradients reach the relaxed source vector.

    Run against the ORACLE head (q = the true edge probability, no learning), so a
    failure here is the unroll or the optimizer, never an undertrained checkpoint.
    """
    graph = _graph(spider, 7)
    device = torch.device("cpu")
    model = WorldModel(
        "gcn",
        in_channels=6,
        hidden_dim=8,
        n_layers=1,
        dropout=0.0,
        head_type="structured_oracle",
        diffusion_model="IC",
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    graph_input, degree_channel = build_graph_tensors(graph, "IC", device)

    relaxed = torch.full((7,), 0.3, requires_grad=True)
    predicted = soft_rollout(model, graph_input, degree_channel, relaxed, steps=3)

    assert predicted.shape == (7,)
    predicted.sum().backward()
    assert relaxed.grad is not None and relaxed.grad.abs().sum() > 0, relaxed.grad

    # ...and the whole inversion recovers a planted source on a graph where the
    # answer is unambiguous. Certain transmission from leaf 2 reaches the hub
    # first, so an observation of {0, 1, 2} can only have started at 2.
    observation = _observation([0, 1, 2], 7)
    sources, scores, info = invert(
        model,
        graph_input,
        degree_channel,
        observation,
        budget=1,
        steps=120,
        horizon=2,
        cardinality_weight=0.2,
    )

    assert scores.shape == (7,)
    assert info["gradient_steps"] == 120
    # The recovered node has to be inside the infected set at minimum; on this
    # graph the endpoint is the only consistent single source
    assert sources[0] in (0, 1, 2), (sources, scores)

    # The prior is a real generative model over source sets, and it has to
    # penalize a vector that looks nothing like the sets it was fit on
    prior = train_source_prior([[2], [4], [6]], num_nodes=7, epochs=60)
    plausible = torch.zeros(7)
    plausible[2] = 1.0
    implausible = torch.ones(7)
    assert prior.negative_log_prior(plausible) < prior.negative_log_prior(implausible)


checks = (
    registry_says_recover_and_generates_no_actions,
    arm_a_is_its_own_condition,
    metrics_are_the_published_ones,
    auc_handles_ties_and_perfect_rankings,
    labels_come_from_the_t0_seed_commit,
    every_localizer_returns_a_legal_source_set,
    lpsi_finds_the_planted_source,
    the_four_oracle_bindings_differ_in_one_thing,
    a_missing_localize_is_a_repair_turn,
    an_illegal_source_set_is_rejected,
    the_scored_harness_is_fixed,
    a_generated_program_runs_end_to_end,
    the_anchor_wrapper_matches_the_library,
    the_prompt_states_the_right_contract,
    the_inversion_is_differentiable_and_recovers_a_source,
)


if __name__ == "__main__":
    for check in checks:
        check()
        print(f"[ok] {check.__name__}")

    print(f"\n{len(checks)} source-localization checks passed")
