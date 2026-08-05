"""
Shared plumbing for the three SUPERVISED cascade-reconstruction imputers.

GRIN, SPIN and Deep Demixing all fit a model on labelled histories and then
predict, which is a setting none of our own arms have. That shared shape is what
this module holds: load the export, turn each cascade into a `[steps, nodes, 1]`
tensor plus an observation mask, train on the selection rows, predict on the
evaluation rows, and write `predictions.json`.

TWO THINGS EVERY CALLER INHERITS, and both are the reason this is one module
rather than three copies.

**The label boundary is enforced by the DATA, not by the driver.**
`registry._reconstruction_export(supervised=True)` writes the true history for
training rows and exactly zero for evaluation rows, and raises if that is ever
violated. So `training_rows()` below cannot accidentally include an evaluation
cascade, and a bug in a driver reads zeros rather than the answer.

**The training signal is the SAME masking the arm is scored under.** Each
training cascade's observed entries come from its own `reported` mask, so the
model learns to fill the holes it will actually be asked to fill. Imputers are
usually trained with a synthetic mask drawn independently of evaluation; using
ours is the more favourable choice for these baselines and the honest one, since
a mismatch would hand them a harder problem than every other arm faces.

**What is NOT reproduced, and must be said wherever these rows appear:** none of
these three is driven through its own training harness. GRIN pins
`tensorflow==2.5.0` and `pytorch-lightning==1.4`, SPIN needs `tsl` plus a
Lightning `Experiment` runner, and Deep Demixing's `GCVAE_Trainer` wants its own
pickle format and DataLoader. We import each repo's MODEL and train it here with
Adam on a masked BCE. That reproduces the published ARCHITECTURE at a training
budget we control, not the published training recipe — so these are "the authors'
model, our optimizer" rows, and `--gwm-epochs` is the first number to raise
before quoting any of them as parity.
"""

import json
import os

import numpy as np
import torch

# Small enough that a 20-cascade pool finishes on CPU; every driver takes an
# override so a real run can raise it. This is the FIRST number to raise before
# quoting any of these three as a published-parity comparison.
default_epochs = 60
default_lr = 1e-3
default_hidden = 32
default_batch = 4


def settings() -> dict:
    """Training knobs, from the environment so one adapter serves every arm."""
    return {
        "epochs": int(os.environ.get("GWM_IMPUTE_EPOCHS", default_epochs)),
        "lr": float(os.environ.get("GWM_IMPUTE_LR", default_lr)),
        "hidden": int(os.environ.get("GWM_IMPUTE_HIDDEN", default_hidden)),
        "batch": int(os.environ.get("GWM_IMPUTE_BATCH", default_batch)),
        "seed": int(os.environ.get("GWM_IMPUTE_SEED", 42)),
        "device": torch.device(os.environ.get("GWM_IMPUTE_DEVICE", "cpu")),
    }


def load_export(work_dir: str) -> dict:
    """
    The export, as the tensors an imputer wants.

    `x` is `[cascades, steps, nodes, 1]` with the OBSERVED entries filled and
    everything else zero; `mask` is 1 exactly where an entry was observed. Under
    `partial_times` an observation pins one node at one step; under
    `partial_nodes` the time is unknown, so the node is pinned across every step
    it could have activated at (which is the honest encoding of "infected, time
    unknown"); under `final_snapshot` only the last step is observed.
    """
    graph = np.load(os.path.join(work_dir, "graph.npz"))
    payload = np.load(os.path.join(work_dir, "instances.npz"), allow_pickle=True)

    num_nodes = int(graph["num_nodes"])
    episode_ids = [str(value) for value in payload["episode_ids"]]
    horizons = payload["horizons"].astype(int)
    steps = int(payload["trajectories"].shape[1])

    count = len(episode_ids)
    x = np.zeros((count, steps, num_nodes, 1), dtype=np.float32)
    mask = np.zeros((count, steps, num_nodes, 1), dtype=np.float32)

    for row in range(count):
        setting = str(payload["settings"][row])
        horizon = int(horizons[row])
        times = payload["reported_times"][row]

        if setting == "final_snapshot":
            # The terminal state is the only thing observed, and it is observed
            # for every node at once
            last = min(horizon, steps - 1)
            x[row, last, :, 0] = payload["final_state"][row]
            mask[row, last, :, 0] = 1.0
            continue

        for node in np.flatnonzero(times > -2):
            time = int(times[node])

            if time >= 0:
                # A monotone cascade: once activated, activated at every later
                # step, which is what makes a per-step target well defined
                x[row, min(time, steps - 1) :, node, 0] = 1.0
                mask[row, min(time, steps - 1) :, node, 0] = 1.0
            else:
                # Infected, time unknown: the node is 1 at the terminal step and
                # nothing earlier is asserted
                last = min(horizon, steps - 1)
                x[row, last, node, 0] = 1.0
                mask[row, last, node, 0] = 1.0

    # Targets exist only for training rows; the export guarantees the rest are zero
    target = payload["trajectories"][..., None].astype(np.float32)

    return {
        "edges": graph["edges"],
        "num_nodes": num_nodes,
        "steps": steps,
        "episode_ids": episode_ids,
        "horizons": horizons,
        "visible": payload["visible"],
        "x": x,
        "mask": mask,
        "target": target,
        "is_train": payload["is_train"].astype(bool),
    }


def adjacency(edges: np.ndarray, num_nodes: int) -> np.ndarray:
    """Row-normalized dense adjacency with self-loops — GRIN's `adj` argument."""
    dense = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for source, target in edges:
        dense[int(source), int(target)] = 1.0
        dense[int(target), int(source)] = 1.0

    dense += np.eye(num_nodes, dtype=np.float32)

    return dense / np.maximum(dense.sum(axis=1, keepdims=True), 1.0)


def edge_index(edges: np.ndarray) -> torch.Tensor:
    """Both directions as a `[2, E]` COO tensor."""
    if not len(edges):
        return torch.zeros((2, 0), dtype=torch.long)

    both = np.concatenate([edges.T, edges[:, ::-1].T], axis=1)

    return torch.from_numpy(np.ascontiguousarray(both)).long()


def train_masked(model, forward, data: dict, config: dict, label: str) -> None:
    """
    Adam on a masked BCE over the training rows.

    `forward(batch_x, batch_mask) -> logits or probabilities in [0, 1]`; the loss
    is taken over EVERY entry rather than only the held-out ones, because a
    cascade's zeros are as informative as its ones — a node that never activated
    is a fact about the trajectory, not a missing value.
    """
    rows = np.flatnonzero(data["is_train"])
    if not rows.size:
        raise RuntimeError(
            "no training cascades in the export: this baseline is SUPERVISED and "
            "needs the selection split's labelled histories. Check that "
            "--cr-select-split and --cr-eval-split name different splits."
        )

    device = config["device"]
    x = torch.from_numpy(data["x"]).to(device)
    mask = torch.from_numpy(data["mask"]).to(device)
    target = torch.from_numpy(data["target"]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    model.train()
    torch.manual_seed(config["seed"])

    for epoch in range(config["epochs"]):
        order = np.random.default_rng(config["seed"] + epoch).permutation(rows)
        total = 0.0

        for start in range(0, len(order), config["batch"]):
            batch = list(order[start : start + config["batch"]])
            optimizer.zero_grad()

            predicted = forward(x[batch], mask[batch])
            loss = torch.nn.functional.binary_cross_entropy(
                predicted.clamp(1e-6, 1.0 - 1e-6), target[batch]
            )
            loss.backward()
            optimizer.step()
            total += float(loss.item())

        if epoch % 10 == 0 or epoch == config["epochs"] - 1:
            print(
                f"[{label}] epoch {epoch + 1}/{config['epochs']} "
                f"loss={total / max(1, len(order) // config['batch'] + 1):.5f}",
                flush=True,
            )

    model.eval()


def decode_all(model, forward, data: dict, config: dict, label: str) -> dict:
    """
    Predict every cascade and turn each `[steps, nodes]` field into a trajectory.

    The activation time of a node is the FIRST step at which its predicted
    probability crosses 0.5, which is the monotone reading the target was built
    under. Parents come out null: none of these three emits a propagation tree, so
    `run_baseline._collect_trajectories` builds it with the same `finalize` rule
    every library decoder uses.

    Predictions are made for EVERY row, training and evaluation alike. That is
    deliberate: the arm scores the selection pool too (it is what `selection` in
    the report table is), and a training row's prediction is honestly labelled as
    one — the generalization gap between the two is the column that exposes it.
    """
    device = config["device"]
    x = torch.from_numpy(data["x"]).to(device)
    mask = torch.from_numpy(data["mask"]).to(device)
    trajectories = {}

    with torch.no_grad():
        for start in range(0, len(data["episode_ids"]), config["batch"]):
            end = min(start + config["batch"], len(data["episode_ids"]))
            batch = list(range(start, end))
            predicted = forward(x[batch], mask[batch]).cpu().numpy()

            for offset, row in enumerate(batch):
                horizon = int(data["horizons"][row])
                visible = data["visible"][row]
                field = predicted[offset, :, :, 0]  # shape: (steps, nodes)
                # An observed entry is ground truth about that node and is pinned
                # back, exactly as GRIN's own `impute_only_holes` does
                observed = data["mask"][row, :, :, 0] > 0
                field = np.where(observed, data["x"][row, :, :, 0], field)

                times = {}
                for node in range(data["num_nodes"]):
                    if not visible[node]:
                        continue

                    hits = np.flatnonzero(field[:, node] >= 0.5)
                    if hits.size:
                        times[str(int(node))] = [int(min(hits[0], horizon)), None]

                trajectories[data["episode_ids"][row]] = times

    print(f"[{label}] decoded {len(trajectories)} cascades", flush=True)

    return trajectories


def write(work_dir: str, trajectories: dict) -> None:
    with open(os.path.join(work_dir, "predictions.json"), "w") as handle:
        json.dump({"trajectories": trajectories}, handle)
