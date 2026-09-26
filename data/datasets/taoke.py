"""
Taoke Cascade Corpus Loader

CasTemp's own e-commerce corpus: Taobao product promotions with real purchase
conversions, released inside the CasTemp-ALGO repository after a privacy and
ethics review.

Source: https://github.com/Lucas-PJ/CasTemp-ALGO (`Taoke.zip`, branch `master`)
    - 2,862 cascades / 29,711 nodes [verified, CasTemp Table 3]
    - Avg. path length 5.61, the longest of that paper's four corpora
    - Undirected here; the raw relation is a directed (src, dst) forward
    - No inherent node features from us - the repo ships a 128-d node embedding
      (`taoke_node_feat.npy`), which we do NOT use, for the reason below
    - No node labels

**This is the only corpus here that records a real transmission edge.** Every row
of `taoke_*_forward_relations.csv` is `(item_id, src_member, dst_member,
min_buy_timestamp, ...)` - who forwarded to whom, and when the forward converted.
Digg and MemeTracker publish adoption times with no parent, and the CasFlow bundle's
`u1/u2/u3` chains are reconstructed rather than logged. So Taoke is where a replayed
`parents` field means what `--trace-parents` means everywhere else in this repo,
and it is the corpus to reach for if a cascade-prediction result ever needs a
tree-level diagnostic.

**Warning: this corpus is "Not recommended"**: it is too new and appears in a
single paper. That verdict stands and this loader does
not overturn it: the corpus exists in exactly one paper, its own, and the only
published numbers on it are CasTemp's Table 4 (MSLE 0.685, MALE 0.548, against
CasCN's 2.795 / 1.308 - a gap far larger than any on the three standard corpora,
which is itself a reason to be careful). It is loaded because it is the one corpus
here that is BOTH auto-downloadable and carries a real tree, and because CasTemp's
leakage finding is reported on it. Lead with Weibo/Twitter/APS.

**Warning: node and item feature embeddings are ignored.** The repo ships
`taoke_node_feat.npy` (128-d per member) and `taoke_item_feat_emb.npy`. Taoke is
the only one of CasTemp's four corpora with cascade AND node features, and using
them would make our row incomparable to every other corpus we run - our feature
builder is 6 channels of state and structure by construction (`world_model/
wm_data.py`), and a corpus with private side information is not the place to break
that. We use `log1p(degree)` like every other loader here and say so.

**Warning: two of the shipped filenames end in `.csv.csv`** (train and val, not
test). That is upstream, not a typo here.
"""

import csv
import os
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.cascade_common import Cascade, cascade_arrays, corpus_statistics

taoke_url = "https://raw.githubusercontent.com/Lucas-PJ/CasTemp-ALGO/master/Taoke.zip"
data_dir = Path(__file__).resolve().parent.parent / "raw" / "taoke"

# The repo's own split files. We read all three and re-split chronologically
# ourselves, because CasTemp's whole finding is that the SPLIT is the trap and our
# protocol has to be one we control.
relation_files = (
    "taoke_train_forward_relations.csv.csv",
    "taoke_val_forward_relations.csv.csv",
    "taoke_test_forward_relations.csv",
)


def download_taoke() -> Path:
    """Download and extract `Taoke.zip` from the CasTemp-ALGO repository."""
    os.makedirs(data_dir, exist_ok=True)
    marker = data_dir / "Taoke" / relation_files[-1]

    if marker.exists():
        print(f"[OK] Taoke already downloaded at {marker}")
        return marker

    archive = data_dir / "Taoke.zip"
    if not archive.exists():
        print(f"[..] Downloading Taoke from {taoke_url} ...")
        urllib.request.urlretrieve(taoke_url, archive)

    print("[..] Extracting ...")
    with zipfile.ZipFile(archive) as zip_file:
        # `__MACOSX/` resource forks ride along in the release; skipping them keeps
        # the directory readable and stops a `._name` file shadowing a real one
        for member in zip_file.namelist():
            if member.startswith("__MACOSX/") or member.endswith("/"):
                continue

            zip_file.extract(member, data_dir)

    if not marker.exists():
        raise FileNotFoundError(
            f"{archive} did not contain {relation_files[-1]}; the release layout "
            f"may have changed. Extract it by hand into {data_dir}/"
        )

    return marker


def _read_relations(directory: Path) -> list[tuple[str, int, int, int]]:
    """`(item, src, dst, timestamp)` across all three shipped split files."""
    rows = []

    for name in relation_files:
        path = directory / name
        if not path.exists():
            continue

        with open(path, newline="") as handle:
            for record in csv.DictReader(handle):
                rows.append(
                    (
                        record["item_id_new"],
                        int(record["src_member_id_new"]),
                        int(record["dst_member_id_new"]),
                        # `min_buy_timestamp_filled` is the column CasTemp's own
                        # loader uses; the unfilled one carries gaps
                        int(
                            record.get("min_buy_timestamp_filled")
                            or record["min_buy_timestamp"]
                        ),
                    )
                )

    if not rows:
        raise FileNotFoundError(
            f"no forward-relation CSV found under {directory}; expected one of "
            f"{list(relation_files)}"
        )

    return rows


def load_taoke_pair(path: Path) -> tuple[list[Cascade], int]:
    """
    The forward relations as cascades on a contiguous `0..N-1` id space.

    One cascade per item. The root is the source of the earliest forward, elapsed
    time is seconds since that forward, and the parent is the row's own `src` -
    the real transmission edge, which is what makes this corpus different from
    every other one here.
    """
    rows = _read_relations(Path(path).parent)

    by_item = {}
    for item, source, target, when in rows:
        by_item.setdefault(item, []).append((source, target, when))

    members = sorted(
        {member for _, source, target, _ in rows for member in (source, target)}
    )
    mapping = {member: index for index, member in enumerate(members)}

    cascades = []

    for item, forwards in sorted(by_item.items()):
        forwards.sort(key=lambda row: (row[2], row[1], row[0]))
        origin_time = forwards[0][2]
        root = mapping[forwards[0][0]]

        events = [(root, 0, None)]
        seen = {root}

        for source, target, when in forwards:
            adopter = mapping[target]
            if adopter in seen:
                continue

            seen.add(adopter)
            events.append((adopter, when - origin_time, mapping[source]))

        if len(events) < 2:
            continue

        events.sort(key=lambda event: (event[1], event[0]))
        cascades.append(
            Cascade(
                cascade_id=item,
                root=root,
                publish_time=origin_time,
                events=events,
            )
        )

    return cascades, len(mapping)


def load_taoke_cascades(path: Path) -> list[Cascade]:
    cascades, num_nodes = load_taoke_pair(path)
    corpus_statistics("Taoke", cascades, num_nodes)

    return cascades


def load_taoke(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """The forward graph: every observed (src, dst) promotion tie, symmetrized."""
    cascades, num_nodes = load_taoke_pair(path)
    adjacency, node_feats, node_labels, _ = cascade_arrays(cascades, num_nodes)

    print(
        f"[OK] Taoke forward graph: {num_nodes} nodes, "
        f"{adjacency.nnz // 2} undirected edges"
    )

    return adjacency, node_feats, node_labels, num_nodes

# Corpus time is SECONDS since the item's first forward.
#
# CasTemp's own segment length for Taoke is 1 DAY and its protocol observes
# one segment and predicts over the next, which under the elapsed-time convention
# every other corpus here uses would be `t_o = 1 d, t_p = 2 d`. **That does not fit
# the released file**: `taoke_*_forward_relations.csv` spans about THREE days
# [derived, counted 2026-08-05], even though `taoke_item_forward_counts.csv` carries
# ten daily columns, so a leak-free chronological split into three contiguous bins
# leaves each bin about a day wide and a 2-day prediction window crosses every
# boundary. 1 h / 6 h is what survives that split with most of the corpus intact, and
# it is OUR setting rather than CasTemp's: a Taoke number from this loader is
# comparable to itself and not to that paper's Table 4.
observation_windows = (3600, 7200)
prediction_horizon = 21600
