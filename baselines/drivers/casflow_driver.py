"""
Drive CasFlow (TKDE'21) and CCGL (TKDE'22) on OUR cascades and dump per-cascade
predictions.

Both repos are by the same author, take the same five-field line format, and run
the same three-stage pipeline: `gene_cas` (filter + split), `gene_emb` (graphwave
+ sparse matrix factorization embeddings), then the model, so one driver covers
both and `GWM_CASFLOW_SRC` picks which clone's `src` directory to import from.

**What this driver does NOT do is re-run the repo's own preprocessing.** `gene_cas.py`
applies CasFlow's corpus-specific publication filters and then splits 70/15/15 at
RANDOM, and both halves are wrong for us: our cascades were already filtered by
`data/wm_cascades.py` at the protocol we are reporting, and
`research/cascade_prediction.md` §8.3 is the single most transferable finding in
that file: the random-over-cascades split LEAKS, and reproducing it here would put
this arm on a different protocol from every other arm in the table. So the driver
reimplements `gene_cas.py`'s `file_write` loop (thirty lines, and the trivially
correct part) honoring OUR split, and calls the repo's own code for the two stages
that carry real content: the graphwave embeddings and the model.

Predictions come back in FILE ORDER, and that is load-bearing. `gene_emb.write_cascade`
iterates `graphs.items()` over a dict built by reading the split file line by line,
which preserves insertion order under Python 3.7+; `tools.Generator` does not shuffle
when `is_train=False`; and `model.predict(test_generator)` returns in generator
order. So row `i` of the prediction array is line `i` of `test.txt`, which is how
the cascade id is recovered.

**The repo's label is the INCREMENT** (`gene_cas.py`: `label = str(label -
len(observation_path))`), and our harness scores a TOTAL. The conversion is one line
at the end and it is stated here because §5.7 difference 3 records that the two
quantities share a symbol in this literature.
"""

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

# The repo's own modules resolve as `utils.graphwave...` from its source directory,
# so that has to be sys.path[0] before anything is imported from it. The variable is
# RELATIVE to the process CWD (which `run_baseline` sets to the clone) rather than
# absolute, because the registry cannot know the clone path at import time: "." for
# CasFlow, whose modules sit at the repo root, and "src" for CCGL, whose do not.
source_root = (Path.cwd() / os.environ.get("GWM_CASFLOW_SRC", ".")).resolve()
sys.path.insert(0, str(source_root))

# CasFlow's own defaults, from `casflow.py`'s flags rather than from the paper
cascade_embedding_dim = 40
global_embedding_dim = 40
max_sequence = 100
latent_dim = 64
rnn_units = 128
flow_transformations = 8
batch_size = 64
learning_rate = 5e-4
patience = 10
# Far below the repo's 1000, and deliberately: this has to finish inside a
# baseline's timeout on a laptop. It is the first number to raise before quoting a
# CasFlow comparison as anything but a smoke result.
max_epochs = 60


def load_instances(work_dir: Path) -> dict:
    payload = np.load(work_dir / "instances.npz", allow_pickle=True)

    return {key: payload[key] for key in payload.files}


def write_split_files(work_dir: Path, data: dict) -> tuple[dict, dict]:
    """
    Our cascades as the repo's own `train/val/test.txt`, honoring OUR split.

    Line format, from `gene_cas.py`'s `file_write`:

        cascade_id \\t n1,n2:t \\t n1,n3:t \\t ... \\t label

    where each entry is a COMMA-joined path (the repo splits on `,` in
    `gene_emb.sequence2list`) truncated to the observation window, and `label` is
    the incremental popularity. Returns the per-split cascade id order, which is
    how a prediction row is mapped back, and the observed count per cascade, which
    is how an increment becomes a total.
    """
    order = {"train": [], "val": [], "test": []}
    observed = {}

    handles = {
        split: open(work_dir / f"{split}.txt", "w", encoding="utf-8")
        for split in order
    }

    try:
        for index, cascade_id in enumerate(data["cascade_ids"]):
            split = str(data["splits"][index])
            if split not in handles:
                continue

            paths = json.loads(str(data["paths"][index]))
            label = int(data["labels"][index])

            handles[split].write(
                f"{cascade_id}\t"
                + "\t".join(f"{','.join(str(node) for node in nodes)}:{when}"
                            for nodes, when in paths)
                + f"\t{label}\n"
            )
            order[split].append(str(cascade_id))
            observed[str(cascade_id)] = len(paths)
    finally:
        for handle in handles.values():
            handle.close()

    return order, observed


def write_global_graph(work_dir: Path, data: dict) -> None:
    """The repo's `global_graph.pkl`: the union of every observed propagation tie."""
    import networkx as nx

    graph = nx.Graph()

    for index in range(len(data["cascade_ids"])):
        for nodes, _ in json.loads(str(data["paths"][index])):
            if len(nodes) < 2:
                graph.add_node(nodes[-1])
            else:
                graph.add_edge(nodes[-1], nodes[-2])

    with open(work_dir / "global_graph.pkl", "wb") as handle:
        pickle.dump(graph, handle)


def build_embeddings(work_dir: Path, observation: int) -> None:
    """
    Run the repo's OWN `gene_emb` stage: graphwave + sparse matrix factorization.

    Called through its module rather than reimplemented, because this is where the
    method's actual content is: the cascade-graph wavelet embeddings and the global
    structural embedding CasFlow's contribution is built on. `absl` flags are set
    programmatically so no CLI is involved.
    """
    from absl import flags

    import gene_emb

    flags.FLAGS([""])
    flags.FLAGS.input = f"{work_dir}{os.sep}"
    flags.FLAGS.gg_path = "global_graph.pkl"
    flags.FLAGS.observation_time = observation
    flags.FLAGS.cg_emb_dim = cascade_embedding_dim
    flags.FLAGS.gg_emb_dim = global_embedding_dim
    flags.FLAGS.max_seq = max_sequence

    gene_emb.main([])


def build_model():
    """
    CasFlow's architecture, exactly as `casflow.py` builds it.

    Rebuilt here rather than imported because `casflow.py` inlines the whole graph
    inside `main(argv)` and returns nothing: there is no model object to get hold
    of, and no prediction to read. The layer stack below is a transcription of that
    function, which is the same treatment `ditto_driver.py` gives DITTO's own CLI.
    """
    import tensorflow as tf

    from utils.tools import Sampling2D, Sampling3D, nf_transformations

    embedding_dim = cascade_embedding_dim + global_embedding_dim
    inputs = tf.keras.layers.Input(shape=(max_sequence, embedding_dim))
    normalized = tf.keras.layers.BatchNormalization()(inputs)

    node_embedding = tf.keras.layers.Dense(embedding_dim)(normalized)
    node_mean = tf.keras.layers.Dense(latent_dim)(node_embedding)
    node_log_var = tf.keras.layers.Dense(latent_dim)(node_embedding)
    node_z = Sampling3D()((node_mean, node_log_var))

    node_reconstruction = tf.keras.layers.Dense(latent_dim)(node_z)
    node_reconstruction = tf.keras.layers.Dense(embedding_dim)(node_reconstruction)

    cascade_embedding = tf.keras.layers.GRU(rnn_units)(node_z)
    cascade_mean = tf.keras.layers.Dense(latent_dim)(cascade_embedding)
    cascade_log_var = tf.keras.layers.Dense(latent_dim)(cascade_embedding)
    cascade_z = Sampling2D()((cascade_mean, cascade_log_var))

    flowed, log_determinant = nf_transformations(
        cascade_z, latent_dim, flow_transformations
    )

    cascade_reconstruction = tf.keras.layers.RepeatVector(max_sequence)(flowed)
    cascade_reconstruction = tf.keras.layers.GRU(rnn_units, return_sequences=True)(
        cascade_reconstruction
    )
    cascade_reconstruction = tf.keras.layers.Dense(latent_dim)(cascade_reconstruction)

    recurrent = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(rnn_units * 2, return_sequences=True)
    )(normalized)
    recurrent = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(rnn_units))(recurrent)

    merged = tf.keras.layers.Concatenate()([flowed, recurrent])
    hidden = tf.keras.layers.Dense(128, activation="relu")(merged)
    hidden = tf.keras.layers.Dense(64, activation="relu")(hidden)
    outputs = tf.keras.layers.Dense(1)(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.add_loss(tf.reduce_mean(tf.square(normalized - node_reconstruction)))
    model.add_loss(
        -0.5
        * tf.reduce_mean(
            node_log_var - tf.square(node_mean) - tf.exp(node_log_var) + 1
        )
    )
    model.add_loss(tf.reduce_mean(tf.square(node_z - cascade_reconstruction)))
    model.add_loss(
        -0.5
        * tf.reduce_mean(
            cascade_log_var - tf.square(cascade_mean) - tf.exp(cascade_log_var) + 1
        )
    )
    model.add_loss(-0.1 * tf.reduce_mean(log_determinant))

    model.compile(
        loss="msle",
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["msle"],
    )

    return model


def main(work_dir: Path) -> None:
    import tensorflow as tf

    from utils.tools import Generator

    data = load_instances(work_dir)
    observation = int(data["observation"])

    print(f"[casflow_driver] {len(data['cascade_ids'])} cascades, t_o={observation}")
    order, observed = write_split_files(work_dir, data)
    write_global_graph(work_dir, data)
    print(
        f"[casflow_driver] split: train {len(order['train'])} / "
        f"val {len(order['val'])} / test {len(order['test'])} (OUR protocol, not "
        f"the repo's random 70/15/15)"
    )

    build_embeddings(work_dir, observation)

    splits = {}
    for split in ("train", "val", "test"):
        with open(work_dir / f"{split}.pkl", "rb") as handle:
            splits[split] = pickle.load(handle)

    model = build_model()
    generators = {
        split: Generator(
            cascade,
            global_embedding,
            label,
            batch_size,
            max_sequence,
            is_train=(split == "train"),
        )
        for split, (cascade, global_embedding, label) in splits.items()
    }

    model.fit(
        generators["train"],
        validation_data=generators["val"],
        epochs=max_epochs,
        verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_msle", patience=patience, restore_best_weights=True
            )
        ],
    )

    predictions = {}

    # BOTH pools are predicted, not only test: the arm makes two passes (selection,
    # then held-out) and a repo invoked once has to cover both. Training on the
    # selection pool and predicting on all of it is the same thing every supervised
    # baseline in this repo does: a label may reach `fit`, never an evaluation-row
    # prediction, and the split flag is what enforces that.
    for split in ("train", "val", "test"):
        if not order[split]:
            continue

        values = np.squeeze(np.asarray(model.predict(generators[split], verbose=0)))
        values = np.atleast_1d(values)

        for row, cascade_id in enumerate(order[split]):
            if row >= values.shape[0]:
                break

            # The repo regresses the INCREMENT (gene_cas: label = P(t_p) - P(t_o)),
            # and our harness scores a TOTAL. §5.7 difference 3.
            increment = max(float(values[row]), 0.0)
            predictions[cascade_id] = observed.get(cascade_id, 0) + increment

    (work_dir / "predictions.json").write_text(
        json.dumps({"popularities": predictions})
    )
    print(f"[casflow_driver] wrote {len(predictions)} predictions")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
