"""
MemeTracker Cascade Corpus Loader

SNAP's MemeTracker phrase-cluster stream: nine months of quoted phrases tracked
across news sites and blogs, one cascade per phrase.

Source: https://snap.stanford.edu/data/memetracker9.html
    - One 1.28 GB gzip per month; this loader streams ONE month (2009-01 by
      default) and never materializes it
    - Nodes are HOSTS (the netloc of each post URL), not individual posts
    - Undirected: the union of observed phrase-propagation ties
    - No inherent node features - uses log(1 + degree)
    - No node labels

Topo-LSTM's Table II reports Memes at 5,000 nodes / 313,669 edges / 54,847
cascades, avg size 17.0 [verified, §6.2]. **We do not reproduce those counts and
say so here rather than in a footnote:** that row is Topo-LSTM's own extraction,
top-5,000 hosts, their own phrase filter, their own month selection, and the
authors publish the extraction script for none of it. Ours is the same construction
applied to one month with `--cp-max-nodes` deciding the host cap, so a Memes number
from this loader is comparable to itself and to nothing published. §6.4 already
records five corpora sharing the name "Twitter"; this is the same hazard with a
different name, and the honest response is a loader that states its own version.

**The diffusion tree is not observed.** A quote record says a host carried a phrase
at a time, never which host it took it from. MemeTracker's `L` (link) lines DO
record hyperlinks, but between POSTS and often to hosts that never carried the
phrase, so they are not a transmission tree for the cascade: using them as one is
exactly the network-inference mistake: an observed link is not a transmission. Every adopter is
therefore attributed to the phrase's first host and the observed path has length 2,
identical to Digg's convention and stated for the same reason.

**Format** (SNAP's own, one record per blank-line-separated block):

    P   <post url>
    T   <post time, YYYY-MM-DD HH:MM:SS>
    Q   <quoted phrase>            (zero or more)
    L   <hyperlink>                (zero or more)

Used by: Topo-LSTM, FOREST (§7).
"""

import gzip
import os
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit
import numpy as np
import scipy.sparse as sp

from data.datasets.cascade_common import (
    Cascade,
    cascade_arrays,
    corpus_statistics,
    events_to_cascade,
)

memetracker_url = "https://snap.stanford.edu/data/bigdata/memetracker9/quotes_{month}.txt.gz"
data_dir = Path(__file__).resolve().parent.parent / "raw" / "memetracker"

# 2009-01 is the peak of the collection window (the US inauguration month) and the
# month with the densest phrase reuse, so one month buys the most cascades per byte
default_month = "2009-01"

# A phrase carried by fewer hosts than this is not a cascade, it is a quotation.
# Deliberately looser than CasFlow's observation-window filter, which is applied
# LATER by `filter_cascades`: this one only decides what is worth keeping in RAM
# while streaming 1.3 GB.
min_phrase_hosts = 5

# Phrases retained while streaming. The full month has millions; this is the first
# number to raise before quoting a MemeTracker comparison as anything but a smoke
# result, and it is a module global rather than a buried constant for that reason.
max_phrases = 60_000

# Distinct hosts a phrase may contribute before it is treated as boilerplate (a
# site-wide footer quote appears on every page of one domain and is not diffusion)
max_phrase_hosts = 5_000


def download_memetracker(month: str = default_month) -> Path:
    """Download one month of the phrase-cluster stream (~1.3 GB gzipped)."""
    os.makedirs(data_dir, exist_ok=True)
    path = data_dir / f"quotes_{month}.txt.gz"

    if path.exists():
        print(f"[OK] MemeTracker {month} already downloaded at {path}")
        return path

    url = memetracker_url.format(month=month)
    print(f"[..] Downloading MemeTracker {month} from {url} (~1.3 GB) ...")
    urllib.request.urlretrieve(url, path)
    print(f"[OK] Saved to {path}")

    return path


def _host(url: str) -> str:
    """The netloc of a post URL, lowercased and stripped of a leading `www.`."""
    netloc = urlsplit(url.strip()).netloc.lower()

    return netloc[4:] if netloc.startswith("www.") else netloc


def _parse_time(stamp: str) -> int:
    """`YYYY-MM-DD HH:MM:SS` to a Unix timestamp, or -1 when unparseable."""
    from datetime import datetime, timezone

    try:
        moment = datetime.strptime(stamp.strip(), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return -1

    return int(moment.replace(tzinfo=timezone.utc).timestamp())


def stream_phrase_adoptions(path: Path) -> dict[str, list[tuple[str, int]]]:
    """
    `{phrase: [(host, unix time), ...]}`, streamed straight off the gzip.

    Never decompresses to disk and never holds a whole month: the file is read as a
    line stream and only the retained phrases' adoption lists survive. A host that
    carries a phrase twice keeps its first time, which is the progressive-cascade
    assumption every method in §3 and §4 makes.
    """
    adoptions = {}
    seen = {}
    host = ""
    when = -1
    full = False

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue

            tag, _, value = line.partition("\t")

            if tag == "P":
                host, when = _host(value), -1
            elif tag == "T":
                when = _parse_time(value)
            elif tag == "Q" and host and when > 0:
                phrase = value.strip()
                if not phrase:
                    continue

                carriers = seen.get(phrase)
                if carriers is None:
                    if full:
                        continue

                    carriers = set()
                    seen[phrase] = carriers
                    adoptions[phrase] = []
                    full = len(seen) >= max_phrases

                if host in carriers or len(carriers) >= max_phrase_hosts:
                    continue

                carriers.add(host)
                adoptions[phrase].append((host, when))

    return {
        phrase: events
        for phrase, events in adoptions.items()
        if len(events) >= min_phrase_hosts
    }


def load_memetracker_pair(path: Path) -> tuple[list[Cascade], int]:
    """The retained phrases as cascades on a contiguous host id space."""
    adoptions = stream_phrase_adoptions(path)

    hosts = sorted({host for events in adoptions.values() for host, _ in events})
    mapping = {host: index for index, host in enumerate(hosts)}

    cascades = []

    for index, (phrase, events) in enumerate(sorted(adoptions.items())):
        events.sort(key=lambda pair: pair[1])
        origin_time = events[0][1]
        cascade = events_to_cascade(
            # The phrase itself can be a paragraph; the index keeps ids short and
            # the phrase is recoverable from the sorted order
            cascade_id=f"meme{index}",
            publish_time=origin_time,
            adoptions=[(mapping[host], when - origin_time) for host, when in events],
        )

        if cascade is not None:
            cascades.append(cascade)

    return cascades, len(mapping)


def load_memetracker_cascades(path: Path) -> list[Cascade]:
    cascades, num_nodes = load_memetracker_pair(path)
    corpus_statistics("MemeTracker", cascades, num_nodes)

    return cascades


def load_memetracker(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """The graph: the union of the phrase-propagation stars over hosts."""
    cascades, num_nodes = load_memetracker_pair(path)
    adjacency, node_feats, node_labels, _ = cascade_arrays(cascades, num_nodes)

    print(
        f"[OK] MemeTracker host graph: {num_nodes} nodes, "
        f"{adjacency.nnz // 2} undirected edges"
    )

    return adjacency, node_feats, node_labels, num_nodes

# Corpus time is SECONDS since a phrase's first appearance. **These windows are
# OURS, not published**, for the same reason Digg's are: §6.2 lists MemeTracker as a
# first-`k`-adopters corpus and the methods that use it are microscopic. 1 d / 7 d
# reflects how phrase reuse actually decays in this stream (most phrases are dead
# within a week), and it is stated as a choice rather than a convention.
observation_windows = (86400, 172800)
prediction_horizon = 604800
