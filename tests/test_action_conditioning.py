"""
The action-conditioning suite, validated against models whose answer is known.

A metric is only worth reporting if it FAILS the thing it is supposed to catch.
So the suite is run against two deliberately constructed models:

- `ConstantModel` ignores its input entirely. Every action-conditioning number
  must take its null value and the verdict must be FAIL.
- the structured IC head applies T_exo in closed form, so it provably reacts to a
  node action. The same numbers must move off the null.

If both scored the same, the suite would be decoration.
"""

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

from coding_agent.types import GraphInfo
from world_model.wm_action_eval import (
    action_ablation,
    action_conditioning_report,
    counterfactual_effect,
    exogenous_fidelity,
)
from world_model.wm_data import TransitionDataset
from world_model.wm_model import WorldModel

device = torch.device("cpu")


class ConstantModel(nn.Module):
    """Ignores X and the graph. The null hypothesis, made runnable."""

    def __init__(self, value: float = 0.1) -> None:
        super().__init__()
        self.value = value

    def forward(self, X, graph: GraphInfo):
        logit = float(np.log(self.value / (1.0 - self.value)))
        return torch.full((X.shape[0], 2), logit)


class StateOnlyModel(nn.Module):
    """Predicts persistence: next = current. Uses the STATE but never the action.

    The subtler null, and the one that matters: a model can score respectably on
    one-step marginals from the state alone, so "good delta_f1" is not evidence of
    action-conditioning.
    """

    def forward(self, X, graph: GraphInfo):
        infected = X[:, 0]
        frontier = X[:, 1]
        probs = torch.stack([infected, frontier], dim=1).clamp(1e-6, 1 - 1e-6)
        return torch.log(probs) - torch.log1p(-probs)


@pytest.fixture
def dataset(dataset_dir: Path) -> TransitionDataset:
    return TransitionDataset(dataset_dir, "IC", "test")


@pytest.fixture
def structured() -> WorldModel:
    torch.manual_seed(0)
    return WorldModel(
        "sage", hidden_dim=8, n_layers=2, head_type="structured", dropout=0.0
    ).eval()


class TestCounterfactualEffect:
    def test_the_fixture_actually_contains_counterfactual_pairs(
        self, structured: WorldModel, dataset: TransitionDataset
    ) -> None:
        """Guard: with no pairs every number below would be vacuously nan."""
        effect = counterfactual_effect(structured, dataset, "IC", device)

        assert effect["n_pairs"] > 0

    def test_action_ignoring_model_scores_the_exact_null(self, dataset: TransitionDataset) -> None:
        """d_pred == 0 everywhere, so mae_norm is exactly 1.0 and r is 0."""
        effect = counterfactual_effect(ConstantModel(), dataset, "IC", device)

        assert effect["effect_mae_norm"] == pytest.approx(1.0)
        assert effect["effect_pearson"] == pytest.approx(0.0)
        assert effect["effect_magnitude_ratio"] == pytest.approx(0.0)

    def test_state_only_model_also_scores_the_null(self, dataset: TransitionDataset) -> None:
        """Two records branching from the SAME state have identical features to a
        state-only model, so its predicted effect is 0 however good it otherwise is."""
        effect = counterfactual_effect(StateOnlyModel(), dataset, "IC", device)

        assert effect["effect_mae_norm"] == pytest.approx(1.0)

    def test_structured_head_beats_the_null(self, structured: WorldModel, dataset: TransitionDataset) -> None:
        effect = counterfactual_effect(structured, dataset, "IC", device)

        assert effect["effect_magnitude_ratio"] > 0.0
        assert effect["effect_mae_norm"] < 1.0

    def test_per_op_breakdown_is_reported(self, structured: WorldModel, dataset: TransitionDataset) -> None:
        effect = counterfactual_effect(structured, dataset, "IC", device)

        assert effect["per_op"]
        for scores in effect["per_op"].values():
            assert "effect_pearson" in scores

    def test_no_pairs_is_reported_as_untestable_not_as_zero(
        self, structured: WorldModel, dataset_dir: Path, tmp_path: Path
    ) -> None:
        """A dataset with one action per state cannot support the claim. It must
        say so rather than returning a number that reads as a failing model."""
        import json

        source = dataset_dir / "transitions_IC_test.jsonl"
        records = [
            json.loads(line) for line in source.read_text().splitlines() if line.strip()
        ]
        source.write_text(
            "\n".join(
                json.dumps(record) for record in records if record["branch"] == "main"
            )
            + "\n"
        )

        effect = counterfactual_effect(
            structured, TransitionDataset(dataset_dir, "IC", "test"), "IC", device
        )

        assert effect["n_pairs"] == 0
        assert "no counterfactual pairs" in effect["note"]


class TestActionAblation:
    def test_ignoring_model_loses_nothing_when_actions_are_corrupted(self, dataset: TransitionDataset) -> None:
        results = action_ablation(ConstantModel(), dataset, "IC", device)

        assert results["null_delta_f1_drop"] == pytest.approx(0.0)
        assert results["shuffle_delta_f1_drop"] == pytest.approx(0.0)

    def test_structured_head_is_hurt_by_shuffling(self, structured: WorldModel, dataset: TransitionDataset) -> None:
        results = action_ablation(structured, dataset, "IC", device)

        assert results["n_with_action"] > 0
        assert results["shuffle_delta_f1_drop"] > 0.0

    def test_only_node_op_records_are_swappable(self, structured: WorldModel, dataset: TransitionDataset) -> None:
        """An edge op cannot be swapped without desynchronising the adjacency, so
        the t=2 remove_edge record must be excluded."""
        results = action_ablation(structured, dataset, "IC", device)

        assert results["n_swappable"] == len(dataset) - 1


class TestExogenousFidelity:
    def test_structured_head_honours_t_exo_exactly(self, structured: WorldModel, dataset: TransitionDataset) -> None:
        results = exogenous_fidelity(structured, dataset, "IC", device)

        assert results["seeded_p_infected_n"] > 0
        # Worst case, not mean: one violated node must be visible
        assert results["seeded_p_infected_worst"] == pytest.approx(1.0, abs=1e-3)
        assert results["seeded_p_infected_exact_frac"] == pytest.approx(1.0)

    def test_ignoring_model_fails_t_exo(self, dataset: TransitionDataset) -> None:
        results = exogenous_fidelity(ConstantModel(), dataset, "IC", device)

        assert results["seeded_p_infected_exact_frac"] == pytest.approx(0.0)


class TestVerdict:
    def test_ignoring_model_is_reported_as_failing(self, dataset: TransitionDataset) -> None:
        report = action_conditioning_report(ConstantModel(), dataset, "IC", device)

        assert report["action_conditioned"] is False
        assert report["verdict"].startswith("FAIL")

    def test_state_only_model_is_reported_as_failing(self, dataset: TransitionDataset) -> None:
        report = action_conditioning_report(StateOnlyModel(), dataset, "IC", device)

        assert report["action_conditioned"] is False

    def test_structured_head_is_reported_as_passing(self, structured: WorldModel, dataset: TransitionDataset) -> None:
        report = action_conditioning_report(structured, dataset, "IC", device)

        assert report["action_conditioned"] is True
        assert report["verdict"].startswith("PASS")

    def test_report_carries_all_three_blocks(self, structured: WorldModel, dataset: TransitionDataset) -> None:
        report = action_conditioning_report(structured, dataset, "IC", device)

        assert {"counterfactual_effect", "ablation", "exogenous"} <= set(report)
