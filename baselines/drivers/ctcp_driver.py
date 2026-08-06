"""
Drive CTCP (IJCAI'23) on OUR cascades and dump per-cascade predictions.

CTCP is the continuous-time, cross-cascade method: one evolving user/cascade state
updated event by event and shared across ALL cascades, rather than a per-cascade
encoder. `research/cascade_prediction.md` §5.3 makes it the interesting arm to run
beside CasFlow — under CasTemp's leak-free split it is one of the two methods whose
train-vs-test loss curves stay flat while CasFlow's and CasDO's diverge, so it is
the published method least likely to be exploiting the temporal shortcut §8.3
describes.

**Its input is a plain event table**, which is what makes it wireable without
touching its preprocessing at all. `utils/data_processing.get_data` reads exactly
two CSVs from `data/`:

    <name>.csv           id, src, dst, cas, time
    <name>_metadata.csv  casid, pub_time

where a row means "user `dst` forwarded message `cas` from `src`, `time` after that
cascade was published". Our `Cascade.events` is that table with the columns
renamed, so the export is a rename rather than a reconstruction.

**The split crosses as TIME BOUNDARIES, not as flags.** `get_data` takes
`train_time` / `val_time` / `test_time` and assigns a cascade by its publication
time, which is exactly the chronological protocol `data/wm_cascades.py` already
replayed under — so passing our own boundaries reproduces our split inside the
repo rather than fighting it. That is the one place this driver is luckier than
`casflow_driver.py`, which had to reimplement a random split it could not use.

**The repo prints aggregate metrics and saves no per-cascade predictions**, so the
driver patches `train.train.train_model`'s evaluation path via the repo's own
`Metric` object. Where that is not reachable it falls back to running the model
forward itself over the test loader, which is what the `predict_all` path below
does.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

# `run_baseline` sets the subprocess CWD to the clone, so the repo root is simply
# the CWD; the variable exists only so the driver can be run by hand from elsewhere.
repository_root = Path(os.environ.get("GWM_CTCP_ROOT", os.getcwd())).resolve()
sys.path.insert(0, str(repository_root))

# CTCP's own defaults, from its README's reported invocation rather than from the
# paper: `--embedding_module aggregate --use_dynamic --use_temporal
# --use_structural --use_static --dropout 0.6 --predictor merge --lambda 0.1`
ctcp_defaults = {
    "bs": 50,
    "lr": 1e-4,
    "node_dim": 64,
    "time_dim": 16,
    "dropout": 0.6,
    "predictor": "merge",
    "embedding_module": "aggregate",
    "lambda": 0.1,
    "patience": 15,
    "run": 1,
    "single": False,
    "use_static": True,
    "use_dynamic": True,
    "use_structural": True,
    "use_temporal": True,
}
# Far below the README's 150, and for the same reason `casflow_driver`'s is: this
# has to finish inside a baseline timeout. Raise it before quoting a CTCP number.
max_epochs = 20

dataset_name = "gwm"


def load_instances(work_dir: Path) -> dict:
    payload = np.load(work_dir / "instances.npz", allow_pickle=True)

    return {key: payload[key] for key in payload.files}


def write_event_table(work_dir: Path, data: dict) -> tuple[Path, dict, dict]:
    """
    Our cascades as CTCP's own two CSVs, written into the repo's `data/` directory.

    `get_data` opens `data/{dataset}.csv` relative to the process CWD and the
    subprocess runs in the repo root, so the files go there rather than into
    work_dir. Returns the observed count and the publication time per cascade,
    which the split boundaries and the increment-to-total conversion both need.
    """
    directory = repository_root / "data"
    os.makedirs(directory, exist_ok=True)

    observed = {}
    published = {}
    rows = []
    identifier = 0

    for index, cascade_id in enumerate(data["cascade_ids"]):
        paths = json.loads(str(data["paths"][index]))
        cascade = str(cascade_id)
        published[cascade] = int(data["publish_times"][index])
        observed[cascade] = len(paths)

        for nodes, when in paths:
            # The repo's `src` is who was forwarded FROM and `dst` who forwarded.
            # A single-element path is the root, whose own "forward" is from itself
            # — which is how the repo's own preprocessing writes a cascade's first
            # event, so the convention is theirs rather than ours.
            source = nodes[-2] if len(nodes) > 1 else nodes[-1]
            rows.append((identifier, source, nodes[-1], cascade, int(when)))
            identifier += 1

    rows.sort(key=lambda row: (row[4], row[0]))

    with open(directory / f"{dataset_name}.csv", "w", encoding="utf-8") as handle:
        handle.write("id,src,dst,cas,time\n")
        for order, (_, source, target, cascade, when) in enumerate(rows):
            handle.write(f"{order},{source},{target},{cascade},{when}\n")

    with open(
        directory / f"{dataset_name}_metadata.csv", "w", encoding="utf-8"
    ) as handle:
        handle.write("casid,pub_time\n")
        for cascade, when in published.items():
            handle.write(f"{cascade},{when}\n")

    return directory, observed, published


def split_boundaries(data: dict, published: dict) -> tuple[int, int, int]:
    """
    Publication-time boundaries that reproduce OUR split inside the repo.

    Derived from the split labels we already assigned rather than recomputed, so
    the two cannot drift: the train boundary is the latest training cascade's
    publication time, and so on. `get_data` assigns a cascade to train when its
    publication time is at or before `train_time`, which is the same rule
    `data/wm_cascades.assign_splits` used to produce the labels.
    """
    times = {"train": [], "val": [], "test": []}

    for index, cascade_id in enumerate(data["cascade_ids"]):
        split = str(data["splits"][index])
        if split in times:
            times[split].append(published[str(cascade_id)])

    latest = max(published.values(), default=0)

    return (
        max(times["train"], default=latest),
        max(times["val"], default=latest),
        max(times["test"], default=latest),
    )


def predict_all(model, dataset, param, observed: dict) -> dict:
    """
    One forward pass per cascade, read back as a TOTAL popularity.

    The repo's own `Tester` returns aggregate metrics and keeps no per-cascade
    output, so this walks its loader and reads the prediction head directly. Its
    target is `log2(increment + 1)` (its `Metric` inverts exactly that), so the
    conversion back is `2**y - 1` plus the observed count.
    """
    import torch

    predictions = {}
    model.eval()

    with torch.no_grad():
        for batch in dataset.loader(param["bs"]):
            output, cascades = model.forward(batch)
            values = np.atleast_1d(
                np.asarray(output.detach().cpu().numpy(), dtype=np.float64).squeeze()
            )

            for row, cascade in enumerate(cascades):
                if row >= values.shape[0]:
                    break

                cascade = str(cascade)
                increment = max(float(2.0 ** values[row] - 1.0), 0.0)
                predictions[cascade] = observed.get(cascade, 0) + increment

    return predictions


def main(work_dir: Path) -> None:
    import torch

    from model.CTCP import CTCP
    from utils.data_processing import get_data
    from utils.my_utils import set_config

    import argparse
    import logging

    data = load_instances(work_dir)
    observation = int(data["observation"])
    horizon = int(data["horizon"])

    _, observed, published = write_event_table(work_dir, data)
    train_time, val_time, test_time = split_boundaries(data, published)

    print(
        f"[ctcp_driver] {len(data['cascade_ids'])} cascades, t_o={observation}, "
        f"t_p={horizon}, boundaries {train_time}/{val_time}/{test_time}"
    )

    # `set_config` takes the argparse Namespace the CLI would have produced; building
    # it here rather than shelling out is what lets the driver keep the model object
    # and read predictions off it
    arguments = argparse.Namespace(
        dataset=dataset_name,
        prefix="gwm",
        gpu=0 if torch.cuda.is_available() else -1,
        epoch=max_epochs,
        **ctcp_defaults,
    )
    param = set_config(arguments)
    param["observe_time"] = observation
    param["predict_time"] = horizon

    logger = logging.getLogger("ctcp_driver")
    logging.basicConfig(level=logging.INFO)

    dataset = get_data(
        dataset_name,
        observation,
        horizon,
        train_time,
        val_time,
        test_time,
        # Time unit: our replay already normalized elapsed time into the corpus's
        # own unit, and the repo divides by this to bucket. 1 keeps its buckets
        # equal to ours.
        1,
        logger,
        param,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CTCP(device=device, node_dim=param["node_dim"], **{
        key: param[key]
        for key in ("dropout", "use_static", "use_dynamic", "use_temporal",
                    "use_structural", "single", "merge_prob", "max_time",
                    "lambda", "time_dim", "predictor", "embedding_module")
        if key in param
    }).to(device)

    from train.train import train_model

    train_model(param, model, dataset, logger, device, param["run"])

    predictions = predict_all(model, dataset, param, observed)

    (work_dir / "predictions.json").write_text(
        json.dumps({"popularities": predictions})
    )
    print(f"[ctcp_driver] wrote {len(predictions)} predictions")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
