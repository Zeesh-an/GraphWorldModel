"""
Weibo Retweet Cascade Corpus Loader (from the raw AMiner release)

The retweet traces that ship in the SAME AMiner Influence-Locality archive we
already fetch for the `weibo` follower graph. `research/cascade_prediction.md` §6.1
calls this "the cheapest possible entry point" and it is exactly right: we already
download and parse that release, the cascades live in a second file inside it, so
adding them is a second parser rather than a second download.

Source: https://www.aminer.cn/influencelocality (manual; registration-gated)
    - `weibo_network.txt` - the follower graph, already parsed by
      `data/datasets/weibo.py` (1,787,443 users / ~216M arcs)
    - `total.txt`         - the retweet traces, parsed here and by nothing else
    - Undirected here: the union of the observed retweet paths
    - No inherent node features - uses log(1 + degree)
    - No node labels

**Warning: this is the raw source, NOT Weibo-A.** `casflow_weibo` is the
DeepHawkes-release preprocessing that carries every published number (119,313
cascades after an 08:00-18:00 publication filter and a `< 10` participant filter).
This loader reads the same underlying traces before any of that, so its counts are
its own: a fifth artefact for a name §6.4 already lists four of. Use it when the
bundle is unavailable or when you want the unfiltered corpus; use `casflow_weibo`
for anything compared against §5.1.

**Warning: `--dataset weibo` is a different object again.** That is the FOLLOWER
GRAPH with no traces at all, and §6.1 records the pair explicitly: "same source,
different artefact: we load the graph, they load the traces".

**Format of `total.txt`** (the Influence-Locality release's own, four lines per
cascade):

    <root user id> <original message id> <n retweets>
    <retweet time> ...              (n entries, Unix seconds)
    <retweeting user id> ...        (n entries)
    <parent user id> ...            (n entries, the user retweeted FROM)

The fourth line is a real transmission edge, which makes this corpus one of only two
here (with Taoke) whose parents are logged rather than attributed. Releases in the
wild vary in whether the parent line is present; when it is absent every adopter is
attributed to the root and this loader says so at load time rather than silently.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.cascade_common import (
    Cascade,
    cascade_arrays,
    corpus_statistics,
    manual_download,
)

data_dir = Path(__file__).resolve().parent.parent / "raw" / "weibo"
weibo_page_url = "https://www.aminer.cn/influencelocality"

instructions = f"""\
AMiner gates the Influence-Locality archive behind registration, so total.txt must
be downloaded by hand: the same manual step data/datasets/weibo.py already
documents for the follower graph:

  1. Register at {weibo_page_url}
  2. Download the Influence-Locality archive (the one containing weibo_network.txt).
  3. Extract total.txt into {data_dir}/

research/cascade_prediction.md 6.1 is why this is the cheapest corpus to add: it is
the same download the `weibo` graph already needs."""


def download_weibo_cascades() -> Path:
    return manual_download("the Weibo retweet traces", data_dir, "total.txt", instructions)


def _blocks(path: Path):
    """Yield `(root, message, times, users, parents | None)` per cascade."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        while True:
            header = handle.readline()
            if not header:
                return

            header = header.split()
            if len(header) < 3:
                continue

            try:
                root, message, count = int(header[0]), header[1], int(header[2])
            except ValueError:
                continue

            if count <= 0:
                continue

            times = handle.readline().split()
            users = handle.readline().split()

            # The parent line is present in the release we document and absent in
            # some mirrors; peek rather than assume, and never consume a header
            position = handle.tell()
            candidate = handle.readline().split()
            parents = candidate if len(candidate) == count else None
            if parents is None:
                handle.seek(position)

            if len(times) != count or len(users) != count:
                continue

            yield root, message, times, users, parents


def load_weibo_cascades_pair(path: Path) -> tuple[list[Cascade], int, bool]:
    """The retweet traces as cascades on a contiguous id space."""
    raw = []
    traced = True

    for root, message, times, users, parents in _blocks(path):
        try:
            stamps = [int(float(value)) for value in times]
            adopters = [int(value) for value in users]
            causes = [int(value) for value in parents] if parents else None
        except ValueError:
            continue

        if causes is None:
            traced = False

        raw.append((root, message, stamps, adopters, causes))

    if not traced:
        print(
            "    NOTE: this release carries no parent line, so every retweeter is "
            "attributed to the root and the observed path length is 2 throughout"
        )

    nodes = sorted(
        {root for root, _, _, _, _ in raw}
        | {adopter for _, _, _, adopters, _ in raw for adopter in adopters}
        | {
            cause
            for _, _, _, _, causes in raw
            if causes
            for cause in causes
        }
    )
    mapping = {node: index for index, node in enumerate(nodes)}

    cascades = []

    for root, message, stamps, adopters, causes in raw:
        origin_time = min(stamps)
        source = mapping[root]
        events = [(source, 0, None)]
        seen = {source}

        ordered = sorted(
            range(len(adopters)), key=lambda index: (stamps[index], adopters[index])
        )
        for index in ordered:
            adopter = mapping[adopters[index]]
            if adopter in seen:
                continue

            seen.add(adopter)
            parent = (
                mapping.get(causes[index], source) if causes is not None else source
            )
            events.append(
                (adopter, stamps[index] - origin_time, None if adopter == source else parent)
            )

        if len(events) < 2:
            continue

        events.sort(key=lambda event: (event[1], event[0]))
        cascades.append(
            Cascade(
                cascade_id=str(message),
                root=source,
                publish_time=origin_time,
                events=events,
            )
        )

    return cascades, len(mapping), traced


def load_weibo_cascades_cascades(path: Path) -> list[Cascade]:
    cascades, num_nodes, _ = load_weibo_cascades_pair(path)
    corpus_statistics("Weibo (AMiner raw)", cascades, num_nodes)

    return cascades


def load_weibo_cascades(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    The graph: the union of the observed retweet paths.

    NOT the 1.79M-node follower network. §6.5 records that the global Weibo graph
    exceeds this pipeline while cascade-local graphs do not, and `weibo.py` is where
    the follower network lives for anyone who wants it.
    """
    cascades, num_nodes, _ = load_weibo_cascades_pair(path)
    adjacency, node_feats, node_labels, _ = cascade_arrays(cascades, num_nodes)

    print(
        f"[OK] Weibo retweet graph: {num_nodes} nodes, "
        f"{adjacency.nnz // 2} undirected edges"
    )

    return adjacency, node_feats, node_labels, num_nodes


# Corpus time is SECONDS since publication; CasFlow's own windows for this corpus
observation_windows = (1800, 3600)
prediction_horizon = 86400
