"""
Drive CasCN (ICDE'19) on OUR cascades and dump per-cascade predictions.

CasCN is the first method in this literature to use both structure and time
properly: a cascade as a SEQUENCE of sub-cascade graphs, a GCN over each snapshot
and an LSTM across them. `research/cascade_prediction.md` §5.3 is why it is worth
running rather than citing: under CasTemp's leak-free split it is the BEST of the
six re-run baselines on Twitter (MSLE 1.206) and second on APS, which is a larger
reordering than any other row in that table. Under the leaky split it sits mid-pack
(2.7931 on Weibo 0.5 h), so it is the clearest single case of §8.3's finding
changing a ranking rather than just a level.

**This driver reuses the repo's own preprocessing rather than reimplementing it**,
which is possible because `preprocessing/preprocess_graph_signal.py` reads exactly
the per-split file `casflow_driver.py` already knows how to write:

    cascade_id \\t root \\t pub_time \\t n \\t path1:t1 path2:t2 ... \\t label

`read_labelANDsize` takes `profile[-1]` as the label and `profile[3]` as the size,
`seq2graph` builds the per-timestep sub-cascade graphs, and
`caslaplacian.calculate_scaled_laplacian_dir` builds the scaled directed Laplacian.
All of that is the method, and none of it is ours. What the driver supplies is the
split: OUR split, not the repo's, for the reason §8.3 gives.

**Two things my own earlier blocker got wrong, corrected here rather than quietly.**
The repo's `num_nodes` is `tf.flags.DEFINE_integer("num_nodes", 100, "number of max
nodes in cascade")`: a PER-CASCADE bound, not the global graph, and cascades past
100 participants are discarded by the preprocessing itself. So the dense
`[batch, n_steps, num_nodes, num_nodes]` placeholder is 100x100 per step, which is
trivial rather than cluster-scale. And the one file that fails to parse under
Python 3 (`preprocessing/utils.py`) fails on a genuine INDENTATION BUG in the repo
a mismatched `else:` at line 121, not on a py2 construct; the registry patches
it.

**It already dumps per-cascade predictions.** `run_graph_sequence.py` pickles
`(predict_result, y_test, test_loss)`, so the readout needs no patch: only the
mapping back to cascade ids, which the driver keeps from the file order it wrote.

**The label is `log2(dP + 1)`** (`preprocess_graph_signal.py`:
`y_data.append(np.log(y+1.0)/np.log(2.0))`), so the conversion back to the TOTAL
popularity our harness scores is `2**pred - 1 + observed`.
"""

import json
import os
import pickle
import sys
from pathlib import Path
import numpy as np

repository_root = Path(os.environ.get("GWM_CASCN_ROOT", os.getcwd())).resolve()
sys.path.insert(0, str(repository_root))

# Far below the repo's own training budget, for the same reason every other driver
# here caps: this has to finish inside a baseline timeout. First thing to raise
# before quoting a CasCN number as anything but a smoke result.
max_steps = 400
display_step = 50
batch_size = 8

# The repo's own defaults, from `model/config.py` and `run_graph_sequence.py`'s
# flags rather than from the paper
n_time_interval = 6
learning_rate = 0.005
max_cascade_nodes = 100


def load_instances(work_dir: Path) -> dict:
    payload = np.load(work_dir / "instances.npz", allow_pickle=True)

    return {key: payload[key] for key in payload.files}


def write_split_files(directory: Path, data: dict) -> tuple[dict, dict]:
    """
    Our cascades as the repo's own `cascade_{train,val,test}.txt`.

    Six tab-separated fields, which is what `read_labelANDsize` and `seq2graph`
    between them consume: `read_labelANDsize` splits on whitespace and takes
    `profile[-1]` as the label and `profile[3]` as the size, and `seq2graph` walks
    field 4's `u1/u2:t` entries. Returns the per-split id order (how a prediction
    row maps back) and the observed count per cascade (how an increment becomes a
    total).
    """
    os.makedirs(directory, exist_ok=True)
    order = {"train": [], "val": [], "test": []}
    observed = {}

    handles = {
        split: open(directory / f"cascade_{split}.txt", "w", encoding="utf-8")
        for split in order
    }

    try:
        for index, cascade_id in enumerate(data["cascade_ids"]):
            split = str(data["splits"][index])
            if split not in handles:
                continue

            paths = json.loads(str(data["paths"][index]))

            # The preprocessing DISCARDS anything past 100 observed participants
            # (`utils.py`: `if len(observation_path) > 100`), so a longer prefix
            # would silently vanish from the pool rather than be truncated
            paths = paths[:max_cascade_nodes]
            if len(paths) < 2:
                continue

            handles[split].write(
                f"{cascade_id}\t{paths[0][0][-1]}\t"
                f"{int(data['publish_times'][index])}\t{len(paths)}\t"
                + " ".join(
                    "/".join(str(node) for node in nodes) + f":{when}"
                    for nodes, when in paths
                )
                + f"\t{int(data['labels'][index])}\n"
            )
            order[split].append(str(cascade_id))
            observed[str(cascade_id)] = len(paths)
    finally:
        for handle in handles.values():
            handle.close()

    return order, observed


def write_configs(directory: Path, observation: int, horizon: int, n_steps: int) -> None:
    """
    Rewrite the repo's two `config.py` files with this run's paths and windows.

    They are files of literals and they ARE the repo's parameter surface: there is
    no CLI for the observation window. Mutating the imported modules instead would
    not survive `runpy`, which re-imports them; and `preprocess_graph_signal.py` has
    no `main()` at all (its body sits under `if __name__ == "__main__"`), so runpy
    is how its own code gets executed rather than reimplemented.
    """
    (repository_root / "preprocessing" / "config.py").write_text(
        f'DATA_PATHA = {str(directory)!r}\n'
        f'cascades = DATA_PATHA + "/dataset.txt"\n'
        f'cascade_train = DATA_PATHA + "/cascade_train.txt"\n'
        f'cascade_val = DATA_PATHA + "/cascade_val.txt"\n'
        f'cascade_test = DATA_PATHA + "/cascade_test.txt"\n'
        f'shortestpath_train = DATA_PATHA + "/shortestpath_train.txt"\n'
        f'shortestpath_val = DATA_PATHA + "/shortestpath_val.txt"\n'
        f'shortestpath_test = DATA_PATHA + "/shortestpath_test.txt"\n'
        f'train_pkl = DATA_PATHA + "/data_train.pkl"\n'
        f'val_pkl = DATA_PATHA + "/data_val.pkl"\n'
        f'test_pkl = DATA_PATHA + "/data_test.pkl"\n'
        f'information = DATA_PATHA + "/information.pkl"\n'
        f'observation = {observation}\n'
        f'pre_times = [{horizon}]\n'
    )
    (repository_root / "model" / "config.py").write_text(
        f'import math\n'
        f'DATA_PATHA = {str(directory)!r}\n'
        f'train_pkl = DATA_PATHA + "/data_train.pkl"\n'
        f'val_pkl = DATA_PATHA + "/data_val.pkl"\n'
        f'test_pkl = DATA_PATHA + "/data_test.pkl"\n'
        f'information = DATA_PATHA + "/information.pkl"\n'
        f'observation = {observation}\n'
        f'n_time_interval = {n_time_interval}\n'
        f'time_interval = math.ceil((observation + 1) * 1.0 / n_time_interval)\n'
        f'n_steps = {n_steps}\n'
        f'num_nodes = {max_cascade_nodes}\n'
        f'learning_rate = {learning_rate}\n'
        f'batch_size = {batch_size}\n'
        f'lmax = 2\n'
        f'version = "gwm"\n'
    )


def run_preprocessing() -> None:
    """
    The repo's OWN preprocessing stage, executed as `__main__`.

    Reused rather than reimplemented, because this is where the method's actual
    input lives: the per-timestep sub-cascade graphs and the scaled directed
    Laplacian (`caslaplacian.calculate_scaled_laplacian_dir`). Neither is ours.
    """
    import runpy

    runpy.run_path(
        str(repository_root / "preprocessing" / "preprocess_graph_signal.py"),
        run_name="__main__",
    )


if __name__ == "__main__":
    work_dir = Path(sys.argv[1])
    import tensorflow as tf

    data = load_instances(work_dir)
    observation = int(data["observation"])
    horizon = int(data["horizon"])

    # Every path both configs resolve hangs off one `DATA_PATHA`; keeping it inside
    # the clone puts every artifact together and out of our tree
    directory = repository_root / "data"
    order, observed = write_split_files(directory, data)
    print(
        f"[cascn_driver] {sum(len(v) for v in order.values())} cascades "
        f"(train {len(order['train'])} / val {len(order['val'])} / "
        f"test {len(order['test'])}), t_o={observation}, t_p={horizon}"
    )

    # `n_steps` is not known until preprocessing has run, so the configs are
    # written twice: once to let it run, once with the sequence length it produced
    write_configs(directory, observation, horizon, n_steps=1)
    run_preprocessing()

    from model import config as model_config

    # The 7-tuple `write_XYSIZE_data` dumps, per split
    with open(model_config.train_pkl, "rb") as handle:
        _, train_x, train_l, train_y, train_sz, train_time, _ = pickle.load(handle)
    with open(model_config.test_pkl, "rb") as handle:
        _, test_x, test_l, test_y, test_sz, test_time, _ = pickle.load(handle)
    with open(model_config.val_pkl, "rb") as handle:
        _, val_x, val_l, val_y, val_sz, val_time, _ = pickle.load(handle)

    from model.model_sparse_graph_signal import Model
    from model.run_graph_sequence import get_batch

    n_steps = max(max(train_sz, default=1), max(test_sz, default=1))
    write_configs(directory, observation, horizon, n_steps)

    import importlib

    model_config = importlib.reload(model_config)

    session = tf.Session()
    model = Model(model_config, max_cascade_nodes, session)
    session.run(tf.global_variables_initializer())

    for step in range(max_steps):
        batch = get_batch(
            train_x, train_l, train_y, train_sz, train_time,
            n_time_interval, step, batch_size, n_steps,
        )
        model.train_batch(*batch)

        if step % display_step == 0:
            print(f"[cascn_driver] step {step}/{max_steps}")

    predictions = {}

    # BOTH pools are predicted: the arm makes two passes (selection, then held-out)
    # and a repo invoked once has to cover both
    for split, (xs, ls, ys, szs, times) in (
        ("train", (train_x, train_l, train_y, train_sz, train_time)),
        ("val", (val_x, val_l, val_y, val_sz, val_time)),
        ("test", (test_x, test_l, test_y, test_sz, test_time)),
    ):
        if not order[split] or not len(ys):
            continue

        rows = []
        for step in range(int(len(ys) / batch_size) + 1):
            batch = get_batch(
                xs, ls, ys, szs, times, n_time_interval, step, batch_size, n_steps
            )
            rows.extend(np.atleast_1d(np.asarray(model.predict(*batch)).squeeze()))

        for row, cascade_id in enumerate(order[split]):
            if row >= len(rows):
                break

            # The repo regresses log2(increment + 1); our harness scores a TOTAL
            increment = max(float(2.0 ** float(rows[row]) - 1.0), 0.0)
            predictions[cascade_id] = observed.get(cascade_id, 0) + increment

    (work_dir / "predictions.json").write_text(
        json.dumps({"popularities": predictions})
    )
    print(f"[cascn_driver] wrote {len(predictions)} predictions")
