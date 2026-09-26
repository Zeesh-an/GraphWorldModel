"""
Digg 2009 Cascade Corpus Loader

The ISI/Lerman Digg 2009 release: a directed friendship graph plus a month of
story votes, which is the third-cheapest of the four real cascade corpora considered
here and the only one of the four that needs no
registration, no request form and no Drive interstitial.

Source: KONECT's mirror of the ISI extraction (both halves, one `extr: digg`)
    - `digg-friends`  279,630 users / 1,731,653 directed friend arcs
    - `digg-votes`    bipartite (user, story, vote time), one month of 2009
    - Undirected here: the published count is 279,632 / 2,617,993 for the
      symmetrized graph, and every published method that uses this corpus symmetrizes
    - No inherent node features - uses log(1 + degree)
    - No node labels

**Warning: the publisher's own page is gone.** It used to be
`isi.edu/~lerman/downloads/digg2009.html` over plain HTTP, auto-downloadable
[checked 2026-07-28]; as of 2026-08-05 that URL 301s to a personal landing page and
`digg/digg2009.zip` returns an HTML 404 body under a 200 status, which is worse than
a clean failure because `urlretrieve` accepts it. KONECT mirrors both halves of the
same extraction and both were verified live, so that is the route this loader takes.
The ids are consistent between the two files because KONECT records them under one
`extr: digg` extraction.

**Warning: this is NOT the `digg` we already load.** `data/datasets/digg.py` is the
Syracuse friendship graph at 116,893 nodes and carries no cascades at all.
This is the 279,632-node ISI graph WITH its 3,553 vote cascades: a different graph,
and ours has no cascades. Quoting a number from one under the other's name is the
name-collision error this file exists to avoid.

**The diffusion tree is not observed.** A vote records that a user promoted a story
and when, never from whom - so every adopter is attributed to the story's first
voter and the observed "path" has length 2 throughout. This corpus is
classed as first-`k`-adopters (microscopic) for exactly that reason. The FRIENDSHIP graph is
still the real one, so the transitions this replays are over genuine social ties;
only the parent attribution is a convention, and `--cp-graph votes` is what switches
to the star construction if you want to see the difference.

Used by: Topo-LSTM, DeepDiffuse, FOREST.
"""

import os
import tarfile
import urllib.request
from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.cascade_common import (
    Cascade,
    cascade_arrays,
    corpus_statistics,
    events_to_cascade,
)

votes_url = "http://konect.cc/files/download.tsv.digg-votes.tar.bz2"
friends_url = "http://konect.cc/files/download.tsv.digg-friends.tar.bz2"
data_dir = Path(__file__).resolve().parent.parent / "raw" / "digg_cascades"

# KONECT's own note: "The dataset contains multiple edges, when a single user has
# apparently given multiple votes to a single item." The first vote is the adoption.


def _fetch(name: str, url: str, archive: str, member: str) -> Path:
    os.makedirs(data_dir, exist_ok=True)
    extracted = data_dir / member

    if extracted.exists():
        print(f"[OK] Digg {name} already downloaded at {extracted}")
        return extracted

    tarball = data_dir / archive
    if not tarball.exists():
        print(f"[..] Downloading Digg {name} from {url} ...")
        urllib.request.urlretrieve(url, tarball)

    print("[..] Extracting ...")
    with tarfile.open(tarball, "r:bz2") as archive_file:
        for entry in archive_file.getmembers():
            if entry.name.endswith(member):
                entry.name = member
                archive_file.extract(entry, data_dir)
                break

    if not extracted.exists():
        raise FileNotFoundError(
            f"{archive} did not contain {member}; KONECT may have changed its "
            f"layout. Extract it by hand into {data_dir}/"
        )

    return extracted


def download_digg_cascades() -> Path:
    """Both halves; returns the votes file, which is what `load_*_cascades` reads."""
    _fetch("friends", friends_url, "digg-friends.tar.bz2", "out.digg-friends")

    return _fetch("votes", votes_url, "digg-votes.tar.bz2", "out.digg-votes")


def _read_konect(path: Path) -> list[tuple[int, int, int]]:
    """`(left, right, timestamp)` triples; `%` comment lines skipped."""
    rows = []

    with open(path) as handle:
        for line in handle:
            if line.startswith("%") or not line.strip():
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            rows.append((int(parts[0]), int(parts[1]), int(parts[3])))

    return rows


def load_digg_cascades_raw(path: Path) -> tuple[list[Cascade], dict[int, int]]:
    """
    The vote log as cascades, plus the story-id map, in ORIGINAL user ids.

    One cascade per story: its adopters are the users who voted, its elapsed times
    are seconds since the story's FIRST vote, and its publish time is that first
    vote's Unix timestamp. Digg publishes no story-submission time in this
    extraction, so the first vote is the only anchor available and it is what
    Topo-LSTM's own preprocessing uses.
    """
    votes = _read_konect(path)
    by_story = {}

    for user, story, when in votes:
        by_story.setdefault(story, []).append((user, when))

    cascades = []
    story_ids = {}

    for story, entries in sorted(by_story.items()):
        entries.sort(key=lambda pair: pair[1])
        origin_time = entries[0][1]
        cascade = events_to_cascade(
            cascade_id=str(story),
            publish_time=origin_time,
            adoptions=[(user, when - origin_time) for user, when in entries],
        )

        if cascade is None:
            continue

        story_ids[story] = len(cascades)
        cascades.append(cascade)

    return cascades, story_ids


def _friendship_edges() -> list[tuple[int, int]]:
    friends = data_dir / "out.digg-friends"

    if not friends.exists():
        raise FileNotFoundError(
            f"the Digg friendship graph is missing at {friends}; run "
            f"download_digg_cascades() first"
        )

    return [(source, target) for source, target, _ in _read_konect(friends)]


def load_digg_cascades_pair(
    path: Path,
) -> tuple[list[Cascade], sp.csr_matrix, int]:
    """
    Cascades and the FRIENDSHIP adjacency, renumbered onto one shared id space.

    Only the users who actually voted survive: the full graph is 279,630 nodes and
    the point of a cascade corpus here is the traces, not a
    scalability run. The induced subgraph keeps every friend tie between two voters,
    which is the graph over which the observed adoptions actually happened.
    """
    cascades, _ = load_digg_cascades_raw(path)

    voters = sorted({user for cascade in cascades for user in cascade.participants()})
    mapping = {user: index for index, user in enumerate(voters)}
    num_nodes = len(mapping)

    sources, destinations = [], []
    for source, target in _friendship_edges():
        if source in mapping and target in mapping:
            sources += [mapping[source], mapping[target]]
            destinations += [mapping[target], mapping[source]]

    adjacency = sp.csr_matrix(
        (
            np.ones(len(sources), dtype=np.float32),
            (np.array(sources, dtype=np.int32), np.array(destinations, dtype=np.int32)),
        ),
        shape=(num_nodes, num_nodes),
    ) if sources else sp.csr_matrix((num_nodes, num_nodes), dtype=np.float32)
    adjacency = (adjacency > 0).astype(np.float32)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    remapped = []
    for cascade in cascades:
        events = [
            (mapping[adopter], elapsed, None if parent is None else mapping.get(parent))
            for adopter, elapsed, parent in cascade.events
        ]
        remapped.append(
            Cascade(
                cascade_id=cascade.cascade_id,
                root=mapping[cascade.root],
                publish_time=cascade.publish_time,
                events=events,
            )
        )

    return remapped, adjacency, num_nodes


def load_digg_cascades_cascades(path: Path) -> list[Cascade]:
    """The cascade-corpus half of the loader contract."""
    cascades, _, num_nodes = load_digg_cascades_pair(path)
    corpus_statistics("Digg 2009", cascades, num_nodes)

    return cascades


def load_digg_cascades(
    path: Path,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    The graph half: the Digg friendship network induced on the voters.

    Deliberately NOT the union of diffusion paths every other corpus here uses:
    Digg publishes the real social graph and it is the object every published
    method on this corpus runs on, so falling back to a union of stars would throw away the one thing
    this corpus has that MemeTracker does not.
    """
    _, adjacency, num_nodes = load_digg_cascades_pair(path)
    degrees = np.array(adjacency.sum(axis=1)).flatten()
    node_feats = np.log1p(degrees).reshape(-1, 1).astype(np.float32)  # shape: (N, 1)

    print(
        f"[OK] Digg 2009 friendship graph induced on voters: {num_nodes} nodes, "
        f"{adjacency.nnz // 2} undirected edges (avg degree {degrees.mean():.1f})"
    )

    return adjacency, node_feats, np.zeros(num_nodes, dtype=np.int32), num_nodes


def load_digg_cascades_star(
    path: Path,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """The `--cp-graph votes` construction: the union of the vote stars."""
    cascades, _, num_nodes = load_digg_cascades_pair(path)

    return cascade_arrays(cascades, num_nodes)

# Corpus time is SECONDS since a story's first vote. **These windows are OURS, not
# published**: this literature treats Digg as a first-`k`-adopters (microscopic)
# corpus with no clock at all, because the methods that use it rank the next adopter
# rather than forecast a size. 1 h / 24 h mirrors the Weibo macroscopic convention, which
# is the closest published analogue for a fast social corpus, and every number from
# this loader is self-contained for that reason.
observation_windows = (3600, 7200)
prediction_horizon = 86400
