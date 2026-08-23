"""
Shared machinery for the REAL cascade corpora: the only datasets in this repo
that carry diffusion traces rather than topology alone.

Every other loader here answers one question ("what is the graph"). A cascade
corpus answers two, and the second is what `--task cascade_prediction` exists for:
`research/cascade_prediction.md` §6.1 records that not one graph we already load
carries a cascade, and §9.4 that these corpora are the concrete, downloadable form
of the "logged trajectories" our whole methodology note asserts exist.

So a corpus loader exposes THREE functions instead of two:

    download_<name>()                -> raw path            (the usual contract)
    load_<name>(path)                -> (adjacency, feats, labels, n)
    load_<name>_cascades(path)       -> list[Cascade]       (this file's addition)

and `load_<name>` is built FROM the cascades rather than beside them, so a node id
means the same thing in both. That is not a convenience: the underlying social
graph published with these corpora is usually far too large to simulate on (Weibo
is 6.7M nodes), and §6.5 records the way out: the union of the observed diffusion
paths IS a graph, it is the one CasFlow's own `generate_global_graph` builds, and
it is small enough to run.

**The canonical line format.** CasFlow's preprocessed bundle is the de-facto
benchmark artefact of this literature (§8.5: "there is no leaderboard to enter and
no split to inherit"; the bundle is the closest thing), and every later paper
starts from it. One line per cascade, five tab-separated fields:

    cascade_id \t root \t publish_time \t n \t path1:t1 path2:t2 ...

where a path is `u1/u2/u3` meaning `u3` adopted from `u2` who adopted from `u1`,
and `t` is elapsed seconds since publication. `parse_casflow_line` reads it, and
every corpus here is normalized INTO it: including the ones (Digg, MemeTracker)
whose native format is a flat `(item, user, time)` log, where the diffusion tree is
unobserved and every adopter is attributed to the root. That attribution is stated
on each loader rather than hidden: it makes the path length 2 for every adopter,
which is exactly what those corpora support and no more.

**Warning: the same three names mean seven different things** (§6.4). Twitter has
five published versions (88,440 / 86,764 / 19,718 / 67,760 / 569 cascades), Weibo
four and APS three, and §5.7 shows a `< 10` vs `< 50` filter alone moves MSLE by
more than the gap between any two consecutive rows of the field's headline table.
Every loader here states which version it is in its own docstring, and
`corpus_statistics` prints the counts so a mismatch is visible in the log rather
than inferred from a number that looks wrong.
"""

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# Elapsed time is stored in the corpus's own unit (seconds for the social corpora,
# days for APS). A replayed timestep is a BIN of that unit, and the bin width is
# `--cp-step`; these are the field's own observation windows expressed in it, and
# they are quoted from CasFlow's `gene_cas.py` header rather than from any paper.
casflow_windows = {
    # corpus stem -> (observation windows, prediction horizon), corpus time units
    "weibo": ((1800, 3600), 86400),  # 0.5 h / 1 h, predicted to 24 h
    "twitter": ((86400, 172800), 2764800),  # 1 d / 2 d, predicted to 32 d
    "aps": ((1095, 1826), 7305),  # 3 y / 5 y in DAYS, predicted to 20 y
}

# CasFlow drops any cascade with fewer than this many participants inside the
# observation window, and CasFT does the same. §8.4 is why it is a named constant
# rather than a literal: dropping small cascades removes the hardest and most
# numerous cases and inflates every metric, so a table has to report it.
default_min_observed = 10

# ...and keeps only the first this-many participants of any cascade [verified,
# §8.4]. A method that exploits long tails cannot show it under this rule, which is
# also why it is reported rather than assumed.
default_truncate = 100


@dataclass()
class Cascade:
    """
    One observed diffusion, in the units its corpus publishes.

    `events` is `(adopter, elapsed, parent)` sorted by elapsed time, with `parent`
    None for the root. `parent` is the second-to-last id of the corpus's own path
    string, so it is a REAL transmission edge where the corpus records one and the
    root everywhere else, which is exactly the distinction
    `research/cascade_reconstruction.md` spends its §2.6 on, arriving here for free.
    """

    cascade_id: str
    root: int
    publish_time: int
    events: list[tuple[int, int, int | None]] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.events)

    def popularity_at(self, elapsed: int) -> int:
        """Adopters at or before `elapsed`, in the corpus's own time unit."""
        return sum(1 for _, when, _ in self.events if when <= elapsed)

    def prefix(self, elapsed: int) -> list[tuple[int, int, int | None]]:
        return [event for event in self.events if event[1] <= elapsed]

    def participants(self) -> set[int]:
        return {adopter for adopter, _, _ in self.events}


def _publish_time(token: str) -> int:
    """
    A corpus publish time as an int, in that corpus's own time unit.

    Weibo and Twitter publish unix seconds. APS publishes an ISO date, and its
    elapsed times are DAYS, so the date becomes a day ORDINAL rather than a year:
    that is the unit the leak-free split needs, since `wm_cascades` compares
    `publish_time + horizon` (7,305 days on APS) against another cascade's
    publish time, which a bare year would make meaningless.
    """
    try:
        return int(float(token))
    except ValueError:
        return date.fromisoformat(token).toordinal()


def parse_casflow_line(line: str) -> Cascade | None:
    """
    One line of the canonical five-field format into a `Cascade`.

    Returns None for a blank or malformed line rather than raising: these files run
    to millions of lines and a single truncated one at the end of a partial
    download is not worth losing the corpus over: `load_casflow_file` counts them
    and prints the total, so a systematically wrong parse is loud and a stray one is
    not fatal.

    The path `u1/u2/u3:3655` means `u3` adopted at t=3655 from `u2`. The root's own
    entry is `u1:0`, a single-element path, and it is emitted with `parent = None`.
    A repeated adopter keeps its FIRST time, which is what every method in §3 and §4
    assumes (a progressive cascade; a node adopts once).
    """
    parts = line.rstrip("\n").split("\t")

    if len(parts) < 5:
        return None

    cascade_id, root, publish_time = parts[0], parts[1], parts[2]
    events = []
    seen = set()

    for token in parts[4].strip().split(" "):
        if not token:
            continue

        path, _, elapsed = token.rpartition(":")
        if not path:
            return None

        nodes = path.split("/")
        adopter = nodes[-1]

        if adopter in seen:
            continue

        seen.add(adopter)
        try:
            events.append(
                (
                    _identifier(adopter),
                    int(float(elapsed)),
                    _identifier(nodes[-2]) if len(nodes) > 1 else None,
                )
            )
        except ValueError:
            return None

    if not events:
        return None

    events.sort(key=lambda event: (event[1], event[0]))

    # Malformed like every other field on this line rather than fatal: these files
    # run to millions of lines and one truncated tail is not worth the corpus
    try:
        published = _publish_time(publish_time)
    except ValueError:
        return None

    return Cascade(
        cascade_id=cascade_id,
        root=_identifier(root),
        publish_time=published,
        events=events,
    )


_identifier_cache: dict[str, int] = {}


def _identifier(token: str) -> int:
    """
    A corpus user id as an int, interning non-numeric ones.

    Weibo and APS publish integer ids; Digg's story/user log does too, but
    MemeTracker's nodes are hostnames. One cache rather than a per-loader remap
    keeps `parse_casflow_line` total, and `relabel_cascades` renumbers everything
    into `0..N-1` afterwards anyway, so the intermediate value never escapes.
    """
    try:
        return int(token)
    except ValueError:
        interned = _identifier_cache.get(token)

        if interned is None:
            # Negative so an interned id can never collide with a real numeric one
            interned = -(len(_identifier_cache) + 1)
            _identifier_cache[token] = interned

        return interned


def load_casflow_file(path: Path, limit: int = 0) -> list[Cascade]:
    """Every parseable cascade in a canonical-format file, in file order."""
    cascades = []
    malformed = 0

    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue

            cascade = parse_casflow_line(line)
            if cascade is None:
                malformed += 1
                continue

            cascades.append(cascade)
            if limit and len(cascades) >= limit:
                break

    if malformed:
        print(f"    skipped {malformed} malformed line(s)")

    return cascades


def events_to_cascade(
    cascade_id: str,
    publish_time: int,
    adoptions: list[tuple[int, int]],
    root: int | None = None,
) -> Cascade | None:
    """
    A flat `(user, elapsed)` adoption log as a `Cascade` rooted at its first adopter.

    The bridge for every corpus whose native format records WHEN each user adopted
    but not FROM WHOM: Digg's vote log and MemeTracker's phrase-cluster time series
    are both like this, and §6.2 lists them as first-`k`-adopters (microscopic)
    corpora for exactly that reason. Attributing every adopter to the root is the
    honest reading: it is what the data supports, it makes the observed diffusion
    "graph" a star, and each loader says so out loud rather than letting a reader
    assume a tree was recovered.
    """
    if not adoptions:
        return None

    ordered = sorted(adoptions, key=lambda pair: (pair[1], pair[0]))
    seen = set()
    events = []

    for user, elapsed in ordered:
        if user in seen:
            continue

        seen.add(user)
        events.append((int(user), int(elapsed), None))

    origin = events[0][0] if root is None else int(root)
    events = [
        (adopter, elapsed, None if adopter == origin else origin)
        for adopter, elapsed, _ in events
    ]

    return Cascade(
        cascade_id=cascade_id,
        root=origin,
        publish_time=int(publish_time),
        events=events,
    )


def filter_cascades(
    cascades: list[Cascade],
    observation: int,
    min_observed: int = default_min_observed,
    truncate: int = default_truncate,
) -> list[Cascade]:
    """
    CasFlow's own two filters, applied in its own order and reported as its own.

    `min_observed` drops cascades with fewer than that many participants INSIDE the
    observation window (CasFlow and CasFT both use 10; CoupledGNN uses 5; SEISMIC
    and Mishra require 50 retweets). `truncate` keeps only the first that-many
    participants of what survives (CasFlow keeps 100). Both are §8.4 landmines and
    both are recorded in `data/metadata.json`, because a `< 10` versus `< 50`
    threshold moves MSLE by more than the gap between any two consecutive rows of
    §5.1, so a number quoted without them is comparable to nothing.
    """
    kept = []

    for cascade in cascades:
        if cascade.popularity_at(observation) < min_observed:
            continue

        events = cascade.events[:truncate] if truncate else cascade.events
        kept.append(
            Cascade(
                cascade_id=cascade.cascade_id,
                root=cascade.root,
                publish_time=cascade.publish_time,
                events=events,
            )
        )

    return kept


def relabel_cascades(
    cascades: list[Cascade], max_nodes: int = 0
) -> tuple[list[Cascade], dict[int, int]]:
    """
    Renumber corpus ids into `0..N-1`, optionally keeping only the busiest nodes.

    `max_nodes` is what makes a 6.7M-node corpus runnable at all (§6.5): keep the
    `max_nodes` most frequently appearing participants and drop every cascade left
    with fewer than two events. That is CoupledGNN's own move: it samples the
    1.78M-user AMiner following network down to 23,681 users [verified, §5.5], and
    like every reduction here it is reported rather than silent, since a corpus
    reduced this way is a fifth version of a name that already has four.

    Nodes are ranked by appearance count and ties broken by id, so the reduction is
    deterministic in the corpus alone and two runs cannot disagree about which
    graph they are on.
    """
    counts: dict[int, int] = {}

    for cascade in cascades:
        for adopter, _, parent in cascade.events:
            counts[adopter] = counts.get(adopter, 0) + 1
            if parent is not None:
                counts[parent] = counts.get(parent, 0) + 1

    ranked = sorted(counts, key=lambda node: (-counts[node], node))
    keep = set(ranked[:max_nodes] if max_nodes else ranked)

    mapping = {node: index for index, node in enumerate(sorted(keep))}
    relabelled = []

    for cascade in cascades:
        events = [
            (
                mapping[adopter],
                elapsed,
                mapping[parent] if parent is not None and parent in mapping else None,
            )
            for adopter, elapsed, parent in cascade.events
            if adopter in mapping
        ]

        if len(events) < 2 or cascade.root not in mapping:
            continue

        # The earliest surviving adopter becomes the root when the original was
        # dropped, so `parent = None` still marks exactly one node per cascade
        root = mapping[cascade.root]
        events = [
            (adopter, elapsed, None if adopter == root else parent)
            for adopter, elapsed, parent in events
        ]
        relabelled.append(
            Cascade(
                cascade_id=cascade.cascade_id,
                root=root,
                publish_time=cascade.publish_time,
                events=events,
            )
        )

    return relabelled, mapping


def cascade_adjacency(
    cascades: list[Cascade], num_nodes: int
) -> sp.csr_matrix:
    """
    The union of the observed diffusion paths, symmetrized: the corpus's graph.

    CasFlow's `generate_global_graph` builds exactly this and calls it the global
    graph; CoupledGNN uses the published following network instead and samples it
    down. We build it from the paths because it is the only construction that
    guarantees every cascade is realizable ON the graph we hand the world model,
    a following network sampled independently leaves adopters with no in-edge at
    all, which our structured head reads as `p_new = 0` and would score as an
    impossible transition rather than as a hard one.

    Symmetrized because the encoders symmetrize anyway (`D^-1/2 (A+I) D^-1/2`) and
    because a retweet edge observed once in one direction is evidence of a tie, not
    of a one-way channel. Where the corpus records no parent (Digg, MemeTracker)
    every adopter is joined to the root, which makes the induced graph a union of
    stars and is stated on each of those loaders.
    """
    sources, destinations = [], []

    for cascade in cascades:
        for adopter, _, parent in cascade.events:
            if parent is None or parent == adopter:
                continue

            sources += [parent, adopter]
            destinations += [adopter, parent]

    if not sources:
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.float32)

    adjacency = sp.csr_matrix(
        (
            np.ones(len(sources), dtype=np.float32),
            (np.array(sources, dtype=np.int32), np.array(destinations, dtype=np.int32)),
        ),
        shape=(num_nodes, num_nodes),
    )
    adjacency = (adjacency > 0).astype(np.float32)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    return adjacency


def cascade_arrays(
    cascades: list[Cascade], num_nodes: int
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """The four-tuple every `load_<name>` returns, built off the diffusion paths."""
    adjacency = cascade_adjacency(cascades, num_nodes)
    degrees = np.array(adjacency.sum(axis=1)).flatten()
    node_feats = np.log1p(degrees).reshape(-1, 1).astype(np.float32)  # shape: (N, 1)
    node_labels = np.zeros(num_nodes, dtype=np.int32)

    return adjacency, node_feats, node_labels, num_nodes


def corpus_statistics(name: str, cascades: list[Cascade], num_nodes: int) -> None:
    """
    Print what was actually loaded, in the columns §6.2's own table uses.

    Loud rather than optional: §6.4 is a table of seven artefacts sharing three
    names, and the cascade count plus the average size is what tells them apart. A
    run whose log says 119,313 cascades is on Weibo-A; one that says 39,076 is on
    Weibo-B and its MSLE is not comparable to the first.
    """
    sizes = np.array([cascade.size for cascade in cascades], dtype=np.float64)
    print(
        f"[OK] {name}: {len(cascades)} cascades over {num_nodes} nodes, "
        f"avg size {sizes.mean():.1f}, median {np.median(sizes):.0f}, "
        f"max {sizes.max():.0f}"
        if len(cascades)
        else f"[OK] {name}: 0 cascades"
    )


def manual_download(name: str, directory: Path, expected: str, instructions: str) -> Path:
    """
    The path to a corpus the publisher will not serve to `curl`, or a clear raise.

    Four of these corpora are behind something a script cannot pass: APS behind a
    request form, the CasFlow bundle behind a Google Drive interstitial, and the
    AMiner release behind a registration wall [verified, §6.3]. Guessing a URL that
    404s is worse than not shipping a loader, so this raises with the exact steps
    instead: the same standard `data/datasets/weibo.py` already sets.
    """
    os.makedirs(directory, exist_ok=True)
    path = directory / expected

    if path.exists():
        return path

    raise FileNotFoundError(
        f"{name} has no automatic download and {path} is missing.\n{instructions}"
    )
