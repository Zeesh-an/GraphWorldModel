"""
Drive SPIN (NeurIPS 2022) over our masked cascades and write one time assignment each.

Run inside the SPIN clone with our work directory as argv[1]. Reads `graph.npz`
and `instances.npz` (written by `registry._reconstruction_export(supervised=True)`)
and writes `predictions.json`.

WHY THE MODEL AND NOT `experiments/run_imputation.py`. SPIN's own entry point is a
`tsl` + PyTorch Lightning `Experiment` over that library's hardcoded benchmark
datasets, none of which is a cascade. `spin.models.SPINModel` itself needs only
`torch`, `torch_geometric` and three `tsl.nn` pieces (`StaticGraphEmbedding`,
`MLP`, and the repo's own `spin/layers/`), so this imports the published
ARCHITECTURE and trains it with our own Adam loop: **SPIN's model, our
optimizer**, on the same terms as the GRIN driver beside it.

WHY IT IS HERE. SPIN is the one method in
this literature that BEATS DITTO outright on a real-diffusion row (BrFarmers, `F1 .8268`
against `.8206`) while running OUT OF MEMORY on Oregon2, Prost and Pol. That OOM
pattern is the useful half: it is the clearest published statement of the scale
ceiling on attention-based reconstruction, and our own graphs sit on both sides
of it: `jazz` and `infectious` well below, `oregon2` and `rt_pol` at or above.
An OOM here is a REPORTABLE RESULT rather than a failed arm, and `run_baseline`
records it as a skip with the reason attached.

Warning: SUPERVISED, exactly as `grin` is, and the same caveat applies to its row.

`SPINModel.forward(x, u, mask, edge_index, ...)` takes
`[batch, steps, nodes, channels]` and returns `(x_hat, intermediate_imputations)`.
`u` is the exogenous/positional input: SPIN's own experiments pass time-of-day
encodings there, and a cascade has no calendar, so this passes the normalized step
index, which is the same information (where in the sequence this step sits) in
the form the positional encoder expects.
"""

import os
import sys
import torch

if __name__ == "__main__":
    # Run-time paths, not import-time ones: `imputation_common` is copied next to
    # this file inside the work directory, and SPIN's `spin` package lives in the
    # clone this runs from. Neither resolves when the file is read as part of OUR
    # tree, which is why the repo's other drivers import `GraphSL` and `cosasi`
    # the same way.
    sys.path.insert(0, os.getcwd())

    import imputation_common as common
    from spin.models import SPINModel

    work_dir = sys.argv[1]
    config = common.settings()
    data = common.load_export(work_dir)

    edges = common.edge_index(data["edges"]).to(config["device"])

    model = SPINModel(
        input_size=1,
        hidden_size=config["hidden"],
        n_nodes=data["num_nodes"],
        u_size=1,
        output_size=1,
        temporal_self_attention=True,
        reweight="softmax",
        n_layers=4,
        eta=3,
        message_layers=1,
    ).to(config["device"])

    # Normalized step index, broadcast over nodes: a cascade has no calendar, so
    # "where in the sequence this step sits" is the whole exogenous signal
    positions = (
        torch.arange(data["steps"], dtype=torch.float32, device=config["device"])
        / max(data["steps"] - 1, 1)
    ).reshape(1, data["steps"], 1, 1)

    def forward(batch_x, batch_mask):
        u = positions.expand(batch_x.shape[0], -1, data["num_nodes"], 1)
        x_hat, _ = model(
            x=batch_x, u=u, mask=batch_mask, edge_index=edges, edge_weight=None
        )

        return torch.sigmoid(x_hat)

    print(
        f"[spin] {int(data['is_train'].sum())} training / "
        f"{int((~data['is_train']).sum())} evaluation cascades, "
        f"{data['num_nodes']} nodes, {data['steps']} steps, {config}",
        flush=True,
    )

    try:
        common.train_masked(model, forward, data, config, "spin")
    except torch.cuda.OutOfMemoryError as error:
        raise SystemExit(
            f"[spin] OUT OF MEMORY on {data['num_nodes']} nodes x {data['steps']} "
            f"steps. That is the published behaviour rather than a wiring bug: "
            f"SPIN OOMs on Oregon2, Prost and Pol in its own comparison table. "
            f"Report it as the scale "
            f"ceiling it is, or lower GWM_IMPUTE_HIDDEN / GWM_IMPUTE_BATCH. "
            f"({error})"
        ) from error

    common.write(work_dir, common.decode_all(model, forward, data, config, "spin"))
