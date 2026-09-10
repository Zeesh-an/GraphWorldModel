"""
The feedback ladder: the controlled variable of Experiment 3.

What has to hold for the tier comparison to mean anything:

  * `legacy` is the default and does not route through any new code, so every
    existing run is unchanged;
  * each rung is a strict superset of the one below, so "richer" is an ordering
    rather than four unrelated formats;
  * a rung's cost is exactly the cost of the blocks it names — no block is
    computed and thrown away, which would put f0's price at f3's;
  * no rung spends a trusted-simulator episode.
"""

import pytest

from coding_agent.feedback import (
    FeedbackPolicy,
    build_feedback,
    f0,
    f1,
    f2,
    f3,
    legacy,
    resolve,
    tier_blocks,
)


class TestPolicy:
    def test_the_default_is_legacy(self):
        assert resolve(None).tier == legacy
        assert FeedbackPolicy().is_legacy

    def test_an_unknown_tier_is_refused(self):
        with pytest.raises(ValueError, match="unknown feedback tier"):
            FeedbackPolicy("f9")

    def test_the_ladder_is_nested(self):
        for lower, higher in ((f0, f1), (f1, f2), (f2, f3)):
            assert set(tier_blocks[lower]) < set(tier_blocks[higher])

    def test_only_the_full_rung_gets_agent_initiated_probes(self):
        assert not resolve(f0).probes_allowed
        assert not resolve(f2).probes_allowed
        assert resolve(f3).probes_allowed
        # ...and legacy keeps the behaviour it already had
        assert resolve(legacy).probes_allowed

    def test_legacy_is_not_produced_here(self):
        with pytest.raises(ValueError, match="legacy tier"):
            build_feedback(resolve(legacy), None, [])


class TestCostPerTier:
    @pytest.fixture
    def setup(self):
        import networkx as nx
        import numpy as np

        from coding_agent.diagnostics import PlanDiagnostics
        from coding_agent.envs.world_model_env import WorldModelEnvironment
        from coding_agent.types import ActionOp, GraphInfo

        nx_graph = nx.stochastic_block_model(
            [25, 25, 25], [[0.25, 0.01, 0.01], [0.01, 0.25, 0.01], [0.01, 0.01, 0.25]],
            seed=3,
        )
        edges = []
        for source, destination in nx_graph.edges():
            edges += [(source, destination), (destination, source)]
        edge_index = np.array(edges, dtype=np.int64).T
        graph = GraphInfo(
            num_nodes=nx_graph.number_of_nodes(),
            edge_index=edge_index,
            ic_probs=np.full(edge_index.shape[1], 0.1, dtype=np.float32),
            directed=False,
        )
        plan = [[ActionOp("add_node", node) for node in (2, 30, 60)]] + [
            [] for _ in range(8)
        ]

        def make():
            return PlanDiagnostics(
                WorldModelEnvironment.oracle(graph, "IC", n_samples=10, base_seed=0),
                graph,
                horizon=8,
                budget=3,
                seed=0,
            )

        return make, plan

    def test_each_rung_costs_what_its_blocks_cost(self, setup):
        """
        k = 3 seeds, so with paired probes:
          f0 -> 1 (the plan's own rollout)
          f1 -> 1 + k drops
          f2 -> the same (regions are free off the marginals)
          f3 -> + k single-seed rollouts for the overlap profiles
        """
        make, plan = setup
        costs = {}

        for tier in (f0, f1, f2, f3):
            diagnostics = make()
            build_feedback(resolve(tier), diagnostics, plan, history=[1.0] * 6)
            costs[tier] = diagnostics.total_cost

        assert costs[f0].wm_rollouts == 1
        assert costs[f1].wm_rollouts == 1 + 3
        assert costs[f2].wm_rollouts == costs[f1].wm_rollouts
        assert costs[f3].wm_rollouts == costs[f2].wm_rollouts + 3

    def test_no_rung_spends_a_trusted_episode(self, setup):
        make, plan = setup

        for tier in (f0, f1, f2, f3):
            diagnostics = make()
            build_feedback(resolve(tier), diagnostics, plan, history=[1.0] * 6)

            assert diagnostics.total_cost.simulator_episodes == 0

    def test_text_grows_monotonically_up_the_ladder(self, setup):
        make, plan = setup
        lengths = []

        for tier in (f0, f1, f2, f3):
            diagnostics = make()
            text, _ = build_feedback(
                resolve(tier), diagnostics, plan, history=[1.0] * 6
            )
            lengths.append(len(text))

        assert lengths == sorted(lengths)

    def test_f0_really_is_only_the_scalar(self, setup):
        make, plan = setup
        text, diagnosis = build_feedback(resolve(f0), make(), plan)

        assert "Expected spread" in text
        for absent in ("Seed contributions", "Regional coverage", "Bridge coverage"):
            assert absent not in text
        assert diagnosis.contributions == {}


class TestSearchWiring:
    """The flag has to reach the search, or the experiment silently runs one arm
    four times."""

    def _search(self, tier):
        from coding_agent.methods.evolve import EvolveSearch

        return EvolveSearch(outer_iters=1, feedback=tier)

    def test_the_default_search_is_legacy(self):
        from coding_agent.methods.evolve import EvolveSearch

        assert EvolveSearch(outer_iters=1).feedback.is_legacy

    def test_a_tier_reaches_the_search(self):
        assert self._search(f2).feedback.tier == f2

    def test_lower_rungs_lose_agent_initiated_probes(self):
        """A tier that could still ask for a drop probe by hand would not be the
        tier it claims to be."""
        assert not self._search(f1).probes
        assert self._search(f3).probes
        assert self._search(legacy).probes

    def test_the_native_arm_still_overrides_every_tier(self):
        from coding_agent.methods.evolve import EvolveSearch

        assert not EvolveSearch(outer_iters=1, probes=False, feedback=f3).probes
