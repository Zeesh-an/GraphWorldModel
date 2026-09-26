"""
Drive DDMIX / Deep Demixing (EUSIPCO 2021, TSIPN 2023) over our masked cascades.

Run inside the Deep_demixing clone with our work directory as argv[1]. Reads
`graph.npz` and `instances.npz` (written by
`registry._reconstruction_export(supervised=True)`) and writes `predictions.json`.

WHY THIS ONE IS DIFFERENT FROM ITS TWO SIBLINGS. GRIN and SPIN impute a partially
observed TIME SERIES; Deep Demixing demixes ONE AGGREGATED SNAPSHOT into node
states at all `T` steps. Its `x` is `[nodes, 1]` (the collapsed observation) and
its `y` is `[nodes, T]`, the uncollapsed history. That is a closer match to our
`final_snapshot` setting than to the report settings, and it is why this driver
builds `x` by collapsing rather than by masking: under any setting we hand it the
node's observed activation as a single scalar, which is exactly the input its
`CVAE_UNET_Batch.encoder` expects.

WHY THE MODEL AND NOT `scripts/exp0-minimal.py`. That script reads a pickle in the
repo's own `{'adj', 'attr', 'data'}` layout, trains four models at once, and
writes checkpoints under `models/exp_0/`. We use `CVAE_UNET_Batch` directly with
our own Adam loop, on the same terms as the GRIN and SPIN drivers: **the authors'
model, our optimizer**.

THREE THINGS VERIFIED BY READING `models.py` AT HEAD, each of which would be a
silent bug otherwise:

  * `CVAE_UNET_Batch.forward(data)` returns `(y_hat, (mu, logstd, mu_pr,
    logstd_pr))` and `y_hat` is already through a sigmoid, so no second one here.
  * **At eval it uses the PRIOR path**: `z = cat(x_encoded, p)` where `p` comes
    from `self.prior(x, A_coo)` and reads only `x`. `data.y` is still touched to
    compute the posterior's `mu`/`logstd`, which are returned and unused, so this
    driver passes ZEROS for `y` at prediction time. That is not a convenience: an
    evaluation row must never see its own history, and passing the real `y` would
    put the answer inside the forward pass even though the output does not depend
    on it.
  * `CVAE_UNET_Batch` computes `batch = repeat_interleave(arange(x.shape[0] // M),
    M)`, so every graph in a batch must have exactly `max_nodes` nodes. Ours all
    do (one graph, one node count) and the edge index is offset per batch member
    accordingly.

The KLD term is included with a small weight, which is the conditional-VAE
objective the model was designed under; dropping it would train the posterior and
prior heads against nothing and is the easiest way to get a plausible-looking but
untrained decoder.

Warning: SUPERVISED, like its two siblings. It is also the WEAKEST published row in
this literature: DIPT reports it at Path Precision
0.062-0.327 and Jaccard 0.031-0.195 across five graphs, and its own paper says
accuracy degrades as `T` grows because the solution space blows up. A low number
here is the expected outcome, not a wiring failure.
"""

import os
import sys
import numpy as np
import torch

# Weight on the KL term of the conditional-VAE objective. Small, because the
# reconstruction term is a per-node BCE over `nodes x steps` entries and an
# unweighted KL over the flattened latent swamps it at our graph sizes.
kld_weight = 1e-3


def build_batch(data: dict, rows: list, base_edges, device, with_target: bool):
    """
    One batched PyG-style `Data` in the layout `CVAE_UNET_Batch.forward` reads.

    `x` is `[batch * nodes, 1]` (the collapsed observation), `y` is
    `[batch * nodes, T]` (the uncollapsed history), and `edge_index` is
    `base_edges` repeated once per batch member with node ids offset, which is
    what makes the model's own `repeat_interleave` batch vector line up.

    `base_edges` is passed rather than built here because the helper that builds
    it is imported at RUN TIME inside `__main__` (it lives beside this file in the
    work directory, not in our tree), and a module-level function must not depend
    on that.

    `with_target=False` passes ZEROS for `y`. At eval the model takes the prior
    path and never uses the posterior, so this changes nothing about the output
    and guarantees an evaluation row's history never enters the forward pass.
    """
    num_nodes = data["num_nodes"]
    steps = data["steps"]

    # Collapse: a node's observation is whether it was ever seen infected
    collapsed = data["x"][rows].max(axis=1)[..., 0]  # shape: (batch, nodes)
    x = torch.from_numpy(collapsed.reshape(-1, 1).astype(np.float32)).to(device)

    if with_target:
        history = data["target"][rows][..., 0]  # shape: (batch, steps, nodes)
        y = torch.from_numpy(
            np.ascontiguousarray(history.transpose(0, 2, 1).reshape(-1, steps))
        ).to(device)
    else:
        y = torch.zeros(
            (len(rows) * num_nodes, steps), dtype=torch.float32, device=device
        )

    parts = [base_edges + index * num_nodes for index in range(len(rows))]

    return type(
        "Batch",
        (),
        {"x": x, "y": y, "edge_index": torch.cat(parts, dim=1), "pos": None},
    )()


if __name__ == "__main__":
    # Run-time paths, not import-time ones: `imputation_common` is copied next to
    # this file inside the work directory, and `models` is Deep Demixing's own
    # top-level module in the clone this runs from. Neither resolves when the file
    # is read as part of OUR tree, which is why the repo's other drivers import
    # `GraphSL` and `cosasi` the same way.
    sys.path.insert(0, os.getcwd())

    import imputation_common as common
    from models import CVAE_UNET_Batch

    work_dir = sys.argv[1]
    config = common.settings()
    data = common.load_export(work_dir)
    base_edges = common.edge_index(data["edges"]).to(config["device"])

    steps = data["steps"]
    model = CVAE_UNET_Batch(
        encoder_feat=steps,
        posterior_feat=steps,
        T=steps,
        max_nodes=data["num_nodes"],
    ).to(config["device"])

    train_rows = list(np.flatnonzero(data["is_train"]))
    if not train_rows:
        raise SystemExit(
            "[deep_demixing] no training cascades in the export: this baseline is "
            "SUPERVISED. Check that --cr-select-split and --cr-eval-split differ."
        )

    print(
        f"[deep_demixing] {len(train_rows)} training / "
        f"{len(data['episode_ids']) - len(train_rows)} evaluation cascades, "
        f"{data['num_nodes']} nodes, {steps} steps, {config}",
        flush=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    torch.manual_seed(config["seed"])
    model.train()

    for epoch in range(config["epochs"]):
        order = np.random.default_rng(config["seed"] + epoch).permutation(train_rows)
        total, batches = 0.0, 0

        for start in range(0, len(order), config["batch"]):
            rows = list(order[start : start + config["batch"]])
            optimizer.zero_grad()

            batch = build_batch(
                data, rows, base_edges, config["device"], with_target=True
            )

            try:
                y_hat, (mu, logstd, mu_pr, logstd_pr) = model(batch)
            except RuntimeError as error:
                # torch_geometric's GraphUNet.augment_adj does a sparse-CSR @
                # sparse-CSR matmul, which a torch built without MKL cannot do on
                # CPU: the stock Apple Silicon wheel is the common case. It is a
                # property of the environment, not of this adapter: the same four
                # lines fail standalone. Named here because the raw traceback
                # points into torch_geometric and reads like a wiring bug.
                if "without MKL" not in str(error):
                    raise

                raise SystemExit(
                    "[deep_demixing] this torch build has no MKL, and "
                    "torch_geometric's GraphUNet (which CVAE_UNET_Batch is built "
                    "from) needs a sparse-CSR matmul that CPU-without-MKL does not "
                    "implement. Reproduce it in four lines with GraphUNet alone: "
                    "it is the environment, not the export. Run this arm on the "
                    "cluster (Linux torch ships with MKL) or set "
                    "GWM_IMPUTE_DEVICE=cuda."
                ) from error

            reconstruction = torch.nn.functional.binary_cross_entropy(
                y_hat.clamp(1e-6, 1.0 - 1e-6), batch.y
            )
            # The conditional-VAE term: pull the posterior toward the prior, which
            # is what makes the prior usable at eval when no history is available
            kld = (
                logstd_pr
                - logstd
                + (torch.exp(2 * logstd) + (mu - mu_pr) ** 2)
                / (2 * torch.exp(2 * logstd_pr))
                - 0.5
            ).mean()

            loss = reconstruction + kld_weight * kld
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            batches += 1

        if epoch % 10 == 0 or epoch == config["epochs"] - 1:
            print(
                f"[deep_demixing] epoch {epoch + 1}/{config['epochs']} "
                f"loss={total / max(1, batches):.5f}",
                flush=True,
            )

    model.eval()
    trajectories = {}

    with torch.no_grad():
        for start in range(0, len(data["episode_ids"]), config["batch"]):
            rows = list(
                range(start, min(start + config["batch"], len(data["episode_ids"])))
            )
            # with_target=False: the prior path is what eval uses, and an
            # evaluation row's history must not enter the forward pass at all
            batch = build_batch(
                data, rows, base_edges, config["device"], with_target=False
            )
            y_hat, _ = model(batch)
            field = y_hat.cpu().numpy().reshape(len(rows), data["num_nodes"], steps)

            for offset, row in enumerate(rows):
                horizon = int(data["horizons"][row])
                visible = data["visible"][row]
                # (nodes, steps) -> (steps, nodes), then pin the observed entries
                predicted = field[offset].T
                observed = data["mask"][row, :, :, 0] > 0
                predicted = np.where(observed, data["x"][row, :, :, 0], predicted)

                times = {}
                for node in range(data["num_nodes"]):
                    if not visible[node]:
                        continue

                    hits = np.flatnonzero(predicted[:, node] >= 0.5)
                    if hits.size:
                        times[str(int(node))] = [int(min(hits[0], horizon)), None]

                trajectories[data["episode_ids"][row]] = times

    print(f"[deep_demixing] decoded {len(trajectories)} cascades", flush=True)
    common.write(work_dir, trajectories)
