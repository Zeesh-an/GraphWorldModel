"""
Drive GRIN (ICLR 2022) over our masked cascades and write one time assignment each.

Run inside the GRIN clone with our work directory as argv[1]. Reads `graph.npz`
and `instances.npz` (written by `registry._reconstruction_export(supervised=True)`)
and writes `predictions.json`.

WHY THE MODEL AND NOT `scripts/run_imputation.py`. GRIN's own entry point is a
PyTorch Lightning experiment over `lib/datasets/{air,la,bay,synthetic}` — four
hardcoded traffic and air-quality datasets, none of which is a cascade — and its
`requirements.txt` pins `tensorflow==2.5.0`, `tensorflow-gpu==2.4.0`,
`pytorch-lightning==1.4` and `torch==1.8`. Those pins do not resolve on a current
Python and none of them is needed: `lib.nn.models.GRINet` imports only `torch`,
`einops` and `lib/__init__.py`'s `epsilon`, verified by tracing every import in
`lib/nn/layers/{rits,gril,gcrnn,spatial_conv,spatial_attention}.py`. So this
imports the published ARCHITECTURE and trains it with our own Adam loop.

That trade has to be stated wherever this row appears: it is **GRIN's model, our
optimizer**, not GRIN's published training recipe. `GWM_IMPUTE_EPOCHS` is the
first number to raise before quoting it as parity.

WHY IT IS WORTH THE TROUBLE ANYWAY. GRIN is the strongest supervised baseline in
`research/cascade_reconstruction.md` §5.1 and the one DITTO uses as the **ideal
upper bound** — every `Gap` column in its Tables 4-5 is measured against a GRIN
trained with the TRUE `beta`. It is therefore the single most useful reference
number in this literature, and it is the row that shows what a method WITH labels
achieves against our own unsupervised arms.

Warning: this arm is SUPERVISED and every other arm on the table is not. It fits on
the selection split's labelled histories, so its row is not comparable to a
decoder that never sees one, and §5.1.2 is the reason that matters rather than
being a technicality: the supervised family collapses from `F1 ~ 0.80` on
simulated diffusion to `F1 ~ 0.32` on real, which is §2.11 risk 4 in one table.

`GRINet.forward(x, mask)` takes `[batch, steps, nodes, channels]` and returns
`(imputation, prediction)` in training and `imputation` in eval, where
`impute_only_holes` pins the observed entries back — the same thing
`imputation_common.decode_all` does, so the two agree.
"""

import os
import sys

if __name__ == "__main__":
    # Both of these are RUN-TIME paths, not import-time ones, so they live here
    # rather than at module scope: `imputation_common` is copied next to this file
    # inside the work directory (Python puts the script's own directory on
    # sys.path[0]), and GRIN's `lib` package lives in the clone this runs from.
    # Neither resolves when the file is read as part of OUR tree, which is why the
    # repo's other drivers do the same thing with `GraphSL` and `cosasi`.
    sys.path.insert(0, os.getcwd())

    import imputation_common as common
    from lib.nn.models import GRINet

    work_dir = sys.argv[1]
    config = common.settings()
    data = common.load_export(work_dir)

    model = GRINet(
        adj=common.adjacency(data["edges"], data["num_nodes"]),
        d_in=1,
        d_hidden=config["hidden"],
        d_ff=config["hidden"],
        ff_dropout=0.0,
        n_layers=1,
        kernel_size=2,
        decoder_order=1,
        global_att=False,
        d_u=0,
        d_emb=8,
        layer_norm=False,
        merge="mlp",
        # Left ON, which is GRIN's own default: an observed entry is ground truth
        # about that node, so imputing over it would be strictly worse and would
        # also disagree with what every other arm here is allowed to assume
        impute_only_holes=True,
    ).to(config["device"])

    def forward(batch_x, batch_mask):
        out = model(batch_x, mask=batch_mask.bool())
        # Training returns (imputation, prediction); eval returns imputation alone
        imputation = out[0] if isinstance(out, tuple) else out

        return imputation.clamp(0.0, 1.0)

    print(
        f"[grin] {int(data['is_train'].sum())} training / "
        f"{int((~data['is_train']).sum())} evaluation cascades, "
        f"{data['num_nodes']} nodes, {data['steps']} steps, {config}",
        flush=True,
    )

    common.train_masked(model, forward, data, config, "grin")
    common.write(work_dir, common.decode_all(model, forward, data, config, "grin"))
