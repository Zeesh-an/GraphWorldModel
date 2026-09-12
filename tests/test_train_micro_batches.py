from functools import partial
from pathlib import Path

import torch

from world_model.train_wm import micro_batches
from world_model.wm_data import TransitionDataset, collate_transitions
from world_model.wm_model import WorldModel


def _grads(model: WorldModel, batches: list[dict], total_nodes: int) -> list[torch.Tensor]:
    model.zero_grad()
    for batch in batches:
        logits = model(batch["X"], batch["graph"])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, 0], batch["y"][:, 0])
        loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(logits[:, 1], batch["y"][:, 1])
        (loss * (batch["X"].shape[0] / total_nodes)).backward()

    return [parameter.grad.clone() for parameter in model.parameters() if parameter.grad is not None]


def test_micro_batched_gradient_equals_the_whole_batch(dataset_dir: Path) -> None:
    torch.manual_seed(0)
    dataset = TransitionDataset(dataset_dir, "IC", "test")
    items = [dataset[index] for index in range(len(dataset))]
    collate = partial(collate_transitions, diffusion_model="IC", device=torch.device("cpu"))
    model = WorldModel("sage", in_channels=6, hidden_dim=8, n_layers=2, dropout=0.0,
                       head_type="structured", diffusion_model="IC", remove_semantics="spent",
                       action_conditioning="message")
    model.train()
    total_nodes = sum(item["num_nodes"] for item in items)

    # A budget below one graph's cost forces one item per micro-batch
    arcs = int(items[0]["edge_index"].shape[1])
    split = micro_batches(items, collate, budget=arcs * 8 - 1, hidden_dim=8)
    assert len(split) == len(items)
    whole = micro_batches(items, collate, budget=arcs * 8 * len(items), hidden_dim=8)
    assert len(whole) == 1

    for expected, actual in zip(_grads(model, whole, total_nodes), _grads(model, split, total_nodes), strict=True):
        assert torch.allclose(expected, actual, atol=1e-6)


def test_a_graph_over_budget_runs_alone(dataset_dir: Path) -> None:
    dataset = TransitionDataset(dataset_dir, "IC", "test")
    items = [dataset[index] for index in range(3)]
    collate = partial(collate_transitions, diffusion_model="IC", device=torch.device("cpu"))
    assert len(micro_batches(items, collate, budget=1, hidden_dim=8)) == 3
