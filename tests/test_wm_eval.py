"""
Evaluation plumbing, including a regression test for the ground-truth semantics bug.

`rebuild_simulator` used to hard-default `remove_semantics` to `spent` while the
model side of every comparison honoured the configured value. On a `blocked`
(containment) dataset that made the two sides of `ens_count_bias` run different
dynamics — the ground truth kept counting each removed node as infected and the
model did not — so the metric picked up an offset of about -k that had nothing to
do with the model. `TestRemoveSemanticsPropagation` is the test that fails if it
comes back.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from data.wm_simulator import blocked, spent
from world_model.wm_data import TransitionDataset, load_graph_store
from world_model.wm_eval import (
    evaluate_one_step,
    planning_regret,
    rebuild_simulator,
    rollout_ensemble,
)
from world_model.wm_model import WorldModel
from world_model.wm_policies import policies, resolve

device = torch.device("cpu")


@pytest.fixture
def store(dataset_dir: Path) -> dict:
    return load_graph_store(dataset_dir)


@pytest.fixture
def model() -> WorldModel:
    torch.manual_seed(0)
    return WorldModel(
        "sage", hidden_dim=8, n_layers=2, head_type="structured", dropout=0.0
    ).eval()


class TestRemoveSemanticsPropagation:
    def test_rebuild_simulator_defaults_to_spent(self, store) -> None:
        simulator = rebuild_simulator(store["g0"], "IC")

        assert simulator.remove_semantics == spent

    def test_rebuild_simulator_honours_blocked(self, store) -> None:
        simulator = rebuild_simulator(store["g0"], "IC", remove_semantics=blocked)

        assert simulator.remove_semantics == blocked

    def test_blocked_node_leaves_the_count_and_spent_does_not(self, store) -> None:
        """The +k the bug was silently introducing, demonstrated directly."""
        from data.wm_simulator import ActionOp

        bag = [ActionOp("add_node", 1), ActionOp("remove_node", 1)]

        spent_sim = rebuild_simulator(store["g0"], "IC", remove_semantics=spent)
        blocked_sim = rebuild_simulator(store["g0"], "IC", remove_semantics=blocked)

        assert 1 in spent_sim.advance(bag).infected
        assert 1 not in blocked_sim.advance(bag).infected

    def test_rollout_ensemble_threads_semantics_into_the_ground_truth(
        self, model: WorldModel, dataset_dir: Path, store, monkeypatch
    ):
        """The regression proper: whatever rollout_ensemble is told must reach the
        simulator it scores against, not just the model rollout."""
        seen = []
        import world_model.wm_eval as wm_eval

        original = wm_eval.rebuild_simulator

        def spy(*args: object, **kwargs: object):
            seen.append(kwargs.get("remove_semantics", spent))
            return original(*args, **kwargs)

        monkeypatch.setattr(wm_eval, "rebuild_simulator", spy)

        rollout_ensemble(
            model,
            dataset_dir,
            "IC",
            store,
            device,
            "test",
            n_samples=2,
            max_episodes=1,
            remove_semantics=blocked,
        )

        assert seen, "the ground-truth ensemble never built a simulator"
        assert set(seen) == {blocked}

    def test_planning_regret_threads_semantics_too(
        self, model: WorldModel, store, monkeypatch
    ):
        seen = []
        import world_model.wm_eval as wm_eval

        original = wm_eval.rebuild_simulator

        def spy(*args: object, **kwargs: object):
            seen.append(kwargs.get("remove_semantics", spent))
            return original(*args, **kwargs)

        monkeypatch.setattr(wm_eval, "rebuild_simulator", spy)

        planning_regret(
            model,
            store["g0"],
            "IC",
            device,
            n_states=1,
            n_candidates=2,
            mc_runs=1,
            remove_semantics=blocked,
        )

        assert seen and set(seen) == {blocked}


class TestOneStep:
    def test_returns_the_documented_metric_suite(self, model: WorldModel, dataset_dir: Path) -> None:
        dataset = TransitionDataset(dataset_dir, "IC", "test")
        results = evaluate_one_step(model, dataset, "IC", device)

        for key in (
            "infected_acc",
            "new_infection_f1",
            "delta_f1",
            "brier_infected",
            "add_seed_success",
            "action_sensitivity",
            "persistence",
        ):
            assert key in results

    def test_add_seed_success_is_perfect_for_a_structured_head(
        self, model: WorldModel, dataset_dir: Path
    ) -> None:
        dataset = TransitionDataset(dataset_dir, "IC", "test")
        results = evaluate_one_step(model, dataset, "IC", device)

        assert results["add_seed_success"] == pytest.approx(1.0)

    def test_action_override_changes_the_score(self, model: WorldModel, dataset_dir: Path) -> None:
        dataset = TransitionDataset(dataset_dir, "IC", "test")

        baseline = evaluate_one_step(model, dataset, "IC", device)
        nulled = evaluate_one_step(
            model,
            dataset,
            "IC",
            device,
            action_override={index: [] for index in range(len(dataset))},
        )

        assert baseline["add_seed_success"] > nulled.get(
            "add_seed_success", 0.0
        ) or np.isnan(nulled["add_seed_success"])


class TestRolloutEnsemble:
    def test_reports_the_documented_metrics(self, model: WorldModel, dataset_dir: Path, store) -> None:
        results = rollout_ensemble(
            model, dataset_dir, "IC", store, device, "test", n_samples=3, max_episodes=1
        )

        for key in (
            "ens_marg_mae",
            "ens_count_w1",
            "ens_count_bias",
            "ens_final_count_model",
            "ens_final_count_true",
        ):
            assert key in results and np.isfinite(results[key])

    def test_recorded_policy_is_the_default_path(self, model: WorldModel, dataset_dir: Path, store) -> None:
        """Passing no policy must be byte-identical to the pre-existing behaviour."""
        common = dict(n_samples=3, max_episodes=1, seed=7)

        default = rollout_ensemble(
            model, dataset_dir, "IC", store, device, "test", **common
        )
        explicit = rollout_ensemble(
            model,
            dataset_dir,
            "IC",
            store,
            device,
            "test",
            action_policy=None,
            **common,
        )

        assert default == explicit


class TestActionPolicies:
    @pytest.mark.parametrize("name", sorted(policies))
    def test_policy_emits_one_bag_per_step_and_only_node_ops(self, name, store) -> None:
        rng = np.random.default_rng(0)
        episode_records = [{"t": step} for step in range(4)]

        sequence = policies[name](store["g0"], episode_records, rng)

        assert len(sequence) == len(episode_records)
        for bag in sequence:
            for action_op in bag:
                assert action_op["op"] in ("add_node", "remove_node")
                assert 0 <= action_op["target"] < store["g0"]["num_nodes"]

    def test_degree_policy_targets_the_hub_first(self, store) -> None:
        rng = np.random.default_rng(0)
        sequence = policies["degree_seed"](store["g0"], [{"t": 0}], rng)

        # Nodes 1 and 4 both have degree 3 in the fixture; the hub must be one of them
        assert sequence[0][0]["target"] in (1, 4)

    def test_null_policy_emits_nothing(self, store) -> None:
        rng = np.random.default_rng(0)
        sequence = policies["null"](store["g0"], [{"t": 0}, {"t": 1}], rng)

        assert sequence == [[], []]

    def test_off_policy_rollout_runs_end_to_end(self, model: WorldModel, dataset_dir: Path, store) -> None:
        results = rollout_ensemble(
            model,
            dataset_dir,
            "IC",
            store,
            device,
            "test",
            n_samples=3,
            max_episodes=1,
            action_policy=policies["degree_seed"],
        )

        assert np.isfinite(results["ens_marg_mae"])

    def test_unknown_policy_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown action policy"):
            resolve(["greedy_celf"])
