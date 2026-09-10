"""
The diagnostic probe interface, and the constraints that make it usable as
evidence in the feedback experiment.

The probes themselves are easy; what has to be tested is the accounting and the
provenance, because those are the claims the experiment rests on:

  * a region / bridge / stagnation block costs ZERO extra evaluator calls once
    the marginals exist, and says so;
  * a drop or swap costs exactly two rollouts and is PAIRED, so the difference
    is not dominated by ensemble noise;
  * when the bound evaluator is the trusted simulator, the same probes report
    simulator episodes rather than world-model rollouts — the diagnostics can
    never launder a simulator call as a model call;
  * nothing in a region definition is an invented label: each is a named graph
    statistic, and the partitions really do partition.
"""

import networkx as nx
import numpy as np
import pytest

from coding_agent.diagnostics import (
    PlanDiagnostics,
    drop_node,
    plan_seeds,
    solo_plan,
    swap_node,
)
from coding_agent.envs.monte_carlo_env import MonteCarloEnvironment
from coding_agent.envs.world_model_env import WorldModelEnvironment
from coding_agent.regions import (
    bridge_nodes,
    build_regions,
    community_scheme,
    structural_scheme,
)
from coding_agent.types import ActionOp, GraphInfo


@pytest.fixture
def block_graph() -> GraphInfo:
    """Four planted communities: the structure regional coverage is about."""
    nx_graph = nx.stochastic_block_model(
        [30, 30, 30, 30],
        [
            [0.25, 0.01, 0.005, 0.005],
            [0.01, 0.25, 0.01, 0.005],
            [0.005, 0.01, 0.25, 0.01],
            [0.005, 0.005, 0.01, 0.25],
        ],
        seed=1,
    )
    edges = []
    for source, destination in nx_graph.edges():
        edges += [(source, destination), (destination, source)]

    edge_index = np.array(edges, dtype=np.int64).T

    return GraphInfo(
        num_nodes=nx_graph.number_of_nodes(),
        edge_index=edge_index,
        ic_probs=np.full(edge_index.shape[1], 0.1, dtype=np.float32),
        directed=False,
    )


@pytest.fixture
def plan() -> list:
    """Three seeds in one community, one elsewhere: the lopsided case."""
    return [
        [
            ActionOp("add_node", 3),
            ActionOp("add_node", 7),
            ActionOp("add_node", 12),
            ActionOp("add_node", 65),
        ]
    ] + [[] for _ in range(10)]


@pytest.fixture
def diagnostics(block_graph):
    environment = WorldModelEnvironment.oracle(
        block_graph, "IC", n_samples=20, base_seed=0
    )

    return PlanDiagnostics(environment, block_graph, horizon=10, budget=4, seed=0)


class TestPlanAlgebra:
    def test_seeds_come_out_in_scheduled_order(self, plan):
        assert plan_seeds(plan) == [3, 7, 12, 65]

    def test_drop_removes_every_action_touching_the_node(self, plan):
        assert plan_seeds(drop_node(plan, 7)) == [3, 12, 65]

    def test_swap_replaces_only_the_named_target(self, plan):
        assert plan_seeds(swap_node(plan, 7, 99)) == [3, 99, 12, 65]

    def test_solo_keeps_one_seed_in_its_original_bag(self, plan):
        solo = solo_plan(plan, 12)

        assert plan_seeds(solo) == [12]
        assert len(solo) == len(plan)


class TestPairing:
    def test_drop_costs_exactly_two_rollouts(self, diagnostics, plan):
        result = diagnostics.probe_drop(plan, 7)

        assert result["cost"]["wm_rollouts"] == 2
        assert result["cost"]["simulator_episodes"] == 0

    def test_paired_drop_reproduces_the_same_number_twice(self, diagnostics, plan):
        """Common random numbers: the SAME seed, so the difference is exact for
        this ensemble rather than an estimate of it."""
        first = diagnostics.probe_drop(plan, 7)
        second = diagnostics.probe_drop(plan, 7)

        assert first["contribution"] == pytest.approx(second["contribution"])

    def test_dropping_a_seed_that_is_not_in_the_plan_contributes_nothing(
        self, diagnostics, plan
    ):
        assert diagnostics.probe_drop(plan, 999)["contribution"] == pytest.approx(0.0)

    def test_swap_reports_the_signed_change(self, diagnostics, plan):
        result = diagnostics.probe_swap(plan, 65, 40)

        assert result["delta"] == pytest.approx(result["swapped"] - result["plan"])
        assert result["cost"]["wm_rollouts"] == 2

    def test_pairing_actually_resolves_the_difference_better(self, block_graph, plan):
        """
        The reason `paired` is the default, measured rather than asserted.

        The same leave-one-out contribution, re-estimated under several ensemble
        seeds. Both modes are unbiased for the same quantity; the paired one
        cancels the ensemble's own randomness out of the DIFFERENCE, so its
        spread across seeds is much smaller. On a fuller run (24 repetitions on a
        trained SBM checkpoint) this is sd 1.12 paired against 3.27 striped, with
        the striped estimate changing sign; here it is asserted only as an
        ordering, so the test stays fast and cannot flake on a tie.
        """
        target = plan_seeds(plan)[1]
        spreads = {}

        for paired in (True, False):
            estimates = []
            for repetition in range(6):
                environment = WorldModelEnvironment.oracle(
                    block_graph, "IC", n_samples=10, base_seed=repetition
                )
                diagnostics = PlanDiagnostics(
                    environment,
                    block_graph,
                    horizon=10,
                    budget=4,
                    seed=repetition,
                    paired=paired,
                )
                estimates.append(diagnostics.probe_drop(plan, target)["contribution"])

            spreads[paired] = float(np.std(estimates, ddof=1))

        assert spreads[True] < spreads[False]

    def test_unpaired_mode_uses_one_batched_call(self, block_graph, plan):
        environment = WorldModelEnvironment.oracle(
            block_graph, "IC", n_samples=20, base_seed=0
        )
        unpaired = PlanDiagnostics(
            environment, block_graph, horizon=10, budget=4, seed=0, paired=False
        )
        result = unpaired.probe_drop(plan, 7)

        assert result["cost"]["wm_rollouts"] == 1


class TestRegionsAreFree:
    def test_regional_coverage_costs_no_extra_call_once_marginals_exist(
        self, diagnostics, plan
    ):
        value = diagnostics.evaluate(plan)
        before = diagnostics.total_cost.wm_rollouts
        coverage = diagnostics.summarize_regions(plan, value)

        assert coverage
        assert diagnostics.total_cost.wm_rollouts == before

    def test_probe_region_reuses_a_supplied_plan_value(self, diagnostics, plan):
        value = diagnostics.evaluate(plan)
        result = diagnostics.probe_region(plan, list(range(30)), value)

        assert result["cost"]["wm_rollouts"] == 0
        assert result["mass"] == pytest.approx(sum(value.marginals[:30]))

    def test_probe_region_pays_one_call_without_one(self, diagnostics, plan):
        result = diagnostics.probe_region(plan, list(range(30)))

        assert result["cost"]["wm_rollouts"] == 1

    def test_coverage_masses_sum_to_the_total_predicted_mass(self, diagnostics, plan):
        value = diagnostics.evaluate(plan)
        coverage = diagnostics.summarize_regions(plan, value)

        assert sum(entry.mass for entry in coverage) == pytest.approx(
            sum(value.marginals)
        )

    def test_bridge_report_costs_nothing_extra(self, diagnostics, plan):
        value = diagnostics.evaluate(plan)
        before = diagnostics.total_cost.wm_rollouts
        report = diagnostics.bridge_report(plan, value)

        assert diagnostics.total_cost.wm_rollouts == before
        assert report["n_bridges"] >= report["n_covered"]

    def test_unknown_node_ids_are_dropped_rather_than_raising(self, diagnostics, plan):
        value = diagnostics.evaluate(plan)
        result = diagnostics.probe_region(plan, [0, 10 ** 6, -3], value)

        assert result["size"] == 1


class TestRegionSchemes:
    def test_communities_partition_the_node_set(self, block_graph):
        regions = build_regions(block_graph, community_scheme)
        covered = sorted(node for group in regions.members.values() for node in group)

        assert covered == list(range(block_graph.num_nodes))

    def test_structural_roles_partition_the_node_set(self, block_graph):
        regions = build_regions(block_graph, structural_scheme)
        covered = sorted(node for group in regions.members.values() for node in group)

        assert covered == list(range(block_graph.num_nodes))
        assert set(regions.members) <= {"bridge", "core", "periphery", "interior"}

    def test_every_bridge_node_really_leaves_its_community(self, block_graph):
        from coding_agent.regions import community_labels, inter_community_incidence

        labels = community_labels(block_graph)
        incidence = inter_community_incidence(block_graph)

        for node in bridge_nodes(block_graph):
            assert incidence[node] > 0
            assert labels[node] is not None

    def test_seed_basins_need_one_rollout_per_seed(self, block_graph, plan):
        environment = WorldModelEnvironment.oracle(
            block_graph, "IC", n_samples=20, base_seed=0
        )
        diagnostics = PlanDiagnostics(
            environment,
            block_graph,
            horizon=10,
            budget=4,
            seed=0,
            region_scheme="seed_basin",
        )
        before = environment.rollout_calls
        regions = diagnostics.regions(plan)

        assert regions.wm_calls == 4
        assert environment.rollout_calls - before == 4
        assert "unreached" in regions.members

    def test_an_unknown_scheme_is_refused(self, block_graph):
        with pytest.raises(ValueError, match="unknown region scheme"):
            build_regions(block_graph, "vibes")


class TestRedundancy:
    def test_overlap_is_one_for_a_seed_against_itself(self, diagnostics, plan):
        """Sanity on the definition: the fuzzy overlap coefficient of a profile
        with itself is exactly 1."""
        profile = diagnostics._solo_marginals(plan, 3)
        shared = sum(min(value, value) for value in profile)

        assert shared / sum(profile) == pytest.approx(1.0)

    def test_overlap_costs_one_rollout_per_seed_and_caches(self, diagnostics, plan):
        before = diagnostics.total_cost.wm_rollouts
        diagnostics.redundancy(plan)
        after_first = diagnostics.total_cost.wm_rollouts
        diagnostics.redundancy(plan)

        assert after_first - before == 4
        assert diagnostics.total_cost.wm_rollouts == after_first

    def test_overlaps_are_bounded(self, diagnostics, plan):
        report = diagnostics.redundancy(plan)

        assert report["overlaps"]
        for value in report["overlaps"].values():
            assert 0.0 <= value <= 1.0


class TestStagnation:
    def test_it_says_so_before_the_window_is_full(self, diagnostics):
        assert diagnostics.stagnation([1.0, 2.0])["status"] == "WARMING_UP"

    def test_a_flat_history_is_stagnating(self, diagnostics):
        assert (
            diagnostics.stagnation([40.0] * 6)["status"] == "STAGNATING"
        )

    def test_a_rising_history_is_improving(self, diagnostics):
        assert (
            diagnostics.stagnation([40.0, 41.0, 42.0, 43.0, 44.0, 45.0])["status"]
            == "IMPROVING"
        )

    def test_the_test_follows_the_stated_formula(self, diagnostics):
        """max over the window minus the window's first value, against epsilon."""
        history = [10.0, 20.0, 11.0, 11.0, 11.0, 11.0]
        result = diagnostics.stagnation(history)

        assert result["improvement"] == pytest.approx(20.0 - 10.0)

    def test_minimize_flips_what_improvement_means(self, diagnostics):
        falling = [45.0, 44.0, 43.0, 42.0, 41.0, 40.0]

        assert diagnostics.stagnation(falling, "minimize")["status"] == "IMPROVING"
        assert diagnostics.stagnation(falling, "maximize")["status"] == "STAGNATING"

    def test_stagnation_costs_nothing(self, diagnostics):
        before = diagnostics.total_cost.wm_rollouts
        diagnostics.stagnation([40.0] * 6)

        assert diagnostics.total_cost.wm_rollouts == before

    def test_persistent_weakness_needs_a_full_window(self, diagnostics, plan):
        coverage = diagnostics.summarize_regions(plan)

        assert diagnostics.persistent_weakness([coverage] * 2) == []
        persistent = diagnostics.persistent_weakness([coverage] * 5)

        assert all(entry["turns"] == 5 for entry in persistent)


class TestExplainPlan:
    def test_blocks_control_what_is_computed(self, block_graph, plan):
        environment = WorldModelEnvironment.oracle(
            block_graph, "IC", n_samples=20, base_seed=0
        )
        scalar_only = PlanDiagnostics(
            environment, block_graph, horizon=10, budget=4, seed=0
        ).explain_plan(plan, blocks=("value",))

        assert scalar_only.contributions == {}
        assert scalar_only.coverage == []
        assert scalar_only.cost.wm_rollouts == 1

    def test_the_full_report_names_the_weakest_region(self, diagnostics, plan):
        report = diagnostics.explain_plan(plan, history=[40.0] * 6)

        assert "Expected spread" in report.text
        assert "Regional coverage" in report.text
        assert "weakest" in report.text
        assert report.stagnation["status"] == "STAGNATING"

    def test_the_diagnosis_states_evidence_and_prescribes_nothing(
        self, diagnostics, plan
    ):
        report = diagnostics.explain_plan(plan)
        closing = report.text.split("What the numbers show:")[-1]

        assert closing.strip()
        for imperative in ("you should", "replace ", "add seed", "try seeding"):
            assert imperative not in closing.lower()

    def test_persistent_weakness_rides_with_the_stagnation_verdict(
        self, diagnostics, plan
    ):
        """"No longer improving" is only actionable beside "and here is the part
        of the graph nothing has touched"."""
        coverage = diagnostics.summarize_regions(plan)
        report = diagnostics.explain_plan(
            plan, history=[41.0] * 6, coverage_history=[coverage] * 4
        )

        assert report.stagnation["status"] == "STAGNATING"
        assert report.stagnation["persistent_weakness"]
        assert "persistent weakness" in report.text

    def test_the_report_serialises(self, diagnostics, plan):
        import json

        report = diagnostics.explain_plan(plan, history=[40.0] * 6)

        assert json.loads(json.dumps(report.to_dict()))["reward"] > 0


class TestNoLaunderedSimulatorCalls:
    def test_the_same_probe_on_the_simulator_reports_simulator_episodes(
        self, block_graph, plan
    ):
        """
        The single constraint the feedback experiment cannot survive breaking.

        Bound to the trusted simulator, a probe must report its cost as
        `simulator_episodes` — the diagnostics do not decide which evaluator they
        run on and never report a simulator call as a model call.
        """
        environment = MonteCarloEnvironment(block_graph, "IC", mc_runs=3, base_seed=0)
        diagnostics = PlanDiagnostics(
            environment, block_graph, horizon=6, budget=4, seed=0
        )
        result = diagnostics.probe_drop(plan, 7)

        assert result["cost"]["simulator_episodes"] > 0
        assert result["cost"]["wm_forward_passes"] == 0

    def test_the_world_model_arm_spends_no_simulator_episodes(self, diagnostics, plan):
        diagnostics.explain_plan(plan, history=[40.0] * 6)

        assert diagnostics.total_cost.simulator_episodes == 0
        assert diagnostics.total_cost.wm_forward_passes > 0


class TestDegenerateInputs:
    """A refinement loop hands this whatever the generated script produced,
    including a plan that seeds nothing."""

    def test_an_empty_plan_produces_a_report_rather_than_an_error(
        self, diagnostics
    ):
        empty = [[] for _ in range(6)]
        report = diagnostics.explain_plan(empty, history=[1.0] * 6)

        assert report.contributions == {}
        assert report.overlaps == {}
        assert "Expected spread" in report.text

    def test_a_single_seed_has_a_contribution_and_no_pairs(self, diagnostics):
        single = [[ActionOp("add_node", 3)]] + [[] for _ in range(5)]
        report = diagnostics.explain_plan(single)

        assert set(report.contributions) == {3}
        assert report.overlaps == {}


class TestSenseConvention:
    """
    Under containment the objective MINIMIZES, so a raw `V(S) - V(S\\{v})` reads
    backwards: the number that means "this removal worked" is negative. Every
    reported difference is oriented so positive always means "helps the
    objective", or a containment reviser would systematically undo its own wins.
    """

    def _diagnostics(self, block_graph, sense):
        return PlanDiagnostics(
            WorldModelEnvironment.oracle(block_graph, "IC", n_samples=20, base_seed=0),
            block_graph,
            horizon=10,
            budget=4,
            seed=0,
            sense=sense,
        )

    def test_the_two_senses_report_opposite_signs(self, block_graph, plan):
        maximize = self._diagnostics(block_graph, "maximize").probe_drop(plan, 3)
        minimize = self._diagnostics(block_graph, "minimize").probe_drop(plan, 3)

        assert maximize["contribution"] == pytest.approx(-minimize["contribution"])
        # ...and the underlying rollouts are the same numbers
        assert maximize["plan"] == pytest.approx(minimize["plan"])

    def test_the_swap_delta_flips_too(self, block_graph, plan):
        maximize = self._diagnostics(block_graph, "maximize").probe_swap(plan, 65, 40)
        minimize = self._diagnostics(block_graph, "minimize").probe_swap(plan, 65, 40)

        assert maximize["delta"] == pytest.approx(-minimize["delta"])

    def test_the_report_says_which_direction_is_better(self, block_graph, plan):
        text = self._diagnostics(block_graph, "minimize").explain_plan(plan).text

        assert "LOWER IS BETTER" in text

    def test_an_unknown_sense_is_refused(self, block_graph):
        with pytest.raises(ValueError, match="maximize or minimize"):
            self._diagnostics(block_graph, "sideways")
