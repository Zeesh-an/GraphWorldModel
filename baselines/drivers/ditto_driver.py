"""
Drive DITTO (KDD 2023) and its two shipped MLE baselines over our masked cascades.

`GWM_DITTO_ENTRY` / `GWM_DITTO_FN` pick which of the repo's three entry points to
run: `ditto.py`/`main` (the paper's method), `dhrec.py`/`pcdsvc_run` (its
implementation of DHREC-PCDSVC for SI and SIR, which the original code does not
cover) or `cri.py`/`cri_run` (its implementation of CRI, whose authors published
none). All three end in the same three driver lines and all three take one `data`
object, so one driver serves them and the three rows are produced under identical
conditions, which is the whole reason to run the authors' versions beside our own
`dhrec` and `cri`.

Run inside the DITTO clone with our work directory as argv[1]. Reads `graph.npz`
and `instances.npz` (written by `registry._reconstruction_export`) and writes
`predictions.json`.

WHY THIS EXISTS RATHER THAN `scripts/ditto-*.sh`. DITTO's shipped entry point ties
three things together that we need apart: `ditto.py` ends in
`args = get_args(); tester = Tester(...); tester.test([...])`, its `DATASETS`
table only accepts twelve hardcoded names, and `Tester` runs ONE cascade per
process. Our pool is 20 to 40 cascades on a graph none of those names describe, so
launching a subprocess per cascade would put the harness's own overhead into the
number we are trying to measure.

The hook is that `ditto.py` defines `main(data)` BEFORE the three driver lines, so
splitting its source on `args = get_args()` and exec'ing the prefix gives the whole
method with none of the CLI. `main` reads `args` as a global, which we inject
afterwards: Python resolves globals at call time, so this is a supported use
rather than a trick.

WHAT DITTO IS AND IS NOT GIVEN, verified by reading every `data.*` access in
`ditto.py` and `inc/diffus.py`:

  * given: `edge_index`, `num_nodes`, `T`, the OBSERVED final state (`y[:, -1]`),
    and the number of sources (`y[:, 0]` is read only as `(y[:, 0] == 1).sum()`).
    The source count is DITTO's own protocol and matches ours, where every inverse
    method is handed its `k`.
  * NOT given: which nodes the sources are, any intermediate state, any report, and
    any activation time. Columns 1..T-1 of `y` are never read by any code path, so
    they are left at zero rather than filled with the truth.

That means DITTO always solves the DASH (final-snapshot) problem, whatever
`--cr-setting` the sweep is running. Under `partial_times` it is therefore solving
a strictly harder instance than every other arm, and the results table has to say
so, which is what `notes` in the registry entry does.

Output is a TIME ASSIGNMENT, not a tree: DITTO emits per-step node states and no
who-infected-whom edges, so the parents come out null and `run_baseline` builds
the tree with the same rule every library decoder uses.
"""

import json
import os
import sys
import numpy as np
import torch
import torch_geometric as pyg

# The three lines at the end of ditto.py that turn it into a CLI. Splitting here
# keeps main() and drops the argparse call that would consume OUR argv.
driver_marker = "args = get_args()"

# Which of the repo's three entry points to drive, and the function inside it that
# takes one `data` and returns a per-step state matrix
entry_functions = {
    "ditto.py": "main",
    "dhrec.py": "pcdsvc_run",
    "cri.py": "cri_run",
}

# Cut down from DITTO's own scripts/ditto-ba-si.sh so a 20-cascade pool finishes:
# its q_steps=500 trains a proposal network per cascade. Override any of them
# through the environment; GWM_DITTO_FAST=0 restores the paper's values.
paper_defaults = dict(
    b_pI0=1e-6,
    b_pR0=1e-6,
    b_steps=500,
    b_lr=0.003,
    q_steps=500,
    q_lr=0.001,
    q_hid=16,
    q_gnn=3,
    q_mlp=2,
    q_samples=10,
    q_zlim=16,
    p_coef=1.0,
    t_samples=100,
    t_steps=10,
    t_keep=0.5,
)
fast_overrides = dict(b_steps=100, q_steps=100, t_samples=30, t_steps=5)


class Args:
    """The `args` namespace `main` reads as a global."""

    def __init__(self, values: dict, device) -> None:
        for key, value in values.items():
            setattr(self, key, value)

        self.device = device


def load_entry(repo_dir: str, entry: str, function: str) -> dict:
    source = open(os.path.join(repo_dir, entry)).read()

    if driver_marker not in source:
        raise RuntimeError(
            f"{entry} no longer ends with {driver_marker!r}; the upstream entry "
            f"point changed and this driver's split is stale. Re-read {entry} and "
            f"fix the marker."
        )

    namespace = {"__name__": "gwm_entry_prefix"}
    exec(compile(source.split(driver_marker)[0], entry, "exec"), namespace)

    if function not in namespace:
        raise RuntimeError(
            f"exec'ing {entry}'s prefix did not define {function!r}; the upstream "
            f"file was restructured."
        )

    return namespace


def build_data(
    edges: np.ndarray, num_nodes: int, observation, sources: int, horizon: int
):
    """
    One PyG `Data` in exactly the layout `inc/data.py` produces.

    `y` is `(nodes, T + 1)` with SIR states in {0, 1, 2}. Column 0 carries `sources`
    ones so `(y[:, 0] == 1).sum()` is the source count DITTO reads; WHICH nodes
    carry them is never read, so they are placed at ids 0..k-1 rather than at the
    true sources. Column T is the observation. Everything between stays zero.
    """
    horizon = max(int(horizon), 1)
    y = torch.zeros((num_nodes, horizon + 1), dtype=torch.long)
    y[: min(int(sources), num_nodes), 0] = 1
    y[:, horizon] = torch.from_numpy((observation >= 0.5).astype(np.int64))

    edge_index = torch.from_numpy(
        np.concatenate([edges.T, edges[:, ::-1].T], axis=1)
    ).long()

    data = pyg.data.Data(edge_index=edge_index, num_nodes=num_nodes)
    data.y = y
    data.T = torch.tensor(horizon, dtype=torch.long)

    return data


def decode(y_pred: torch.Tensor, observation: np.ndarray, horizon: int) -> dict:
    """
    `{node: [activation step, null]}` from DITTO's per-step state matrix.

    `main` returns `y_pred[:, :T]` after a cummax, and `inc/test.py::test_fix_obs`
    pins the observed final column back on: reproduced here, so a node DITTO only
    commits to at the end lands at `t = T` rather than being dropped.
    """
    states = np.concatenate(
        [y_pred.cpu().numpy(), (observation >= 0.5).astype(np.int64)[:, None]], axis=1
    )
    infected = states >= 1
    times = {}

    for node in range(states.shape[0]):
        hits = np.flatnonzero(infected[node])
        if hits.size:
            times[int(node)] = [int(min(hits[0], horizon)), None]

    return times


if __name__ == "__main__":
    work_dir = sys.argv[1]
    repo_dir = os.getcwd()

    graph = np.load(os.path.join(work_dir, "graph.npz"))
    payload = np.load(os.path.join(work_dir, "instances.npz"), allow_pickle=True)

    edges = graph["edges"]
    num_nodes = int(graph["num_nodes"])
    episode_ids = [str(value) for value in payload["episode_ids"]]

    device = torch.device(os.environ.get("GWM_DITTO_DEVICE", "cpu"))
    values = dict(paper_defaults)
    if os.environ.get("GWM_DITTO_FAST", "1") != "0":
        values |= fast_overrides
    for key in list(values):
        override = os.environ.get(f"GWM_DITTO_{key.upper()}")
        if override is not None:
            values[key] = type(values[key])(override)

    seed = int(os.environ.get("GWM_DITTO_SEED", "123456789"))
    values["seed"] = seed
    torch.manual_seed(seed)

    entry = os.environ.get("GWM_DITTO_ENTRY", "ditto.py")
    function = os.environ.get("GWM_DITTO_FN", entry_functions.get(entry, "main"))
    ditto = load_entry(repo_dir, entry, function)
    ditto["args"] = Args(values, device)

    # dhrec.py and cri.py call seed_all(args.seed) at import; ditto.py does not,
    # so seeding is done here for all three rather than assumed
    if "seed_all" in ditto:
        ditto["seed_all"](seed)

    print(
        f"[ditto] {entry}:{function} on {len(episode_ids)} cascades, "
        f"{num_nodes} nodes, {values}",
        flush=True,
    )
    trajectories = {}

    for row, episode_id in enumerate(episode_ids):
        observation = payload["final_state"][row]
        horizon = int(payload["horizons"][row])
        data = build_data(
            edges, num_nodes, observation, int(payload["source_counts"][row]), horizon
        ).to(device)

        print(
            f"[ditto] cascade {row + 1}/{len(episode_ids)} ({episode_id})", flush=True
        )
        y_pred = ditto[function](data)
        trajectories[episode_id] = decode(y_pred, observation, horizon)

    with open(os.path.join(work_dir, "predictions.json"), "w") as handle:
        json.dump({"trajectories": trajectories}, handle)

    print(f"[ditto] wrote {len(trajectories)} decodes", flush=True)
