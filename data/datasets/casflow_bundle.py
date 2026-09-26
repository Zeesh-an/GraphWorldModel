"""
CasFlow's preprocessed Weibo / Twitter / APS bundle: the de-facto benchmark.

This module implements one finding about the field: there
is **no** unified public benchmark for cascade prediction, no OGB or TGB analogue,
no leaderboard to enter and no split to inherit. What exists instead is one Google
Drive archive holding all three standard corpora in one line format, and every
paper since 2021 starts from it. So the closest thing to a benchmark is a file, and
this is the parser for it.

One entry point per corpus (`casflow_weibo`, `casflow_twitter`, `casflow_aps`),
because seven published artefacts share three names and a single `--dataset
casflow` would put us right back in that trap. Each states its own version, its own
published counts, and its own two observation windows.

**No auto-download.** The archive sits behind Google Drive's virus-scan
interstitial, which `urlretrieve` receives as an HTML page [verified]. Rather
than guess at a confirm-token URL that breaks whenever Drive changes it, this
raises with the exact manual steps: the standard `data/datasets/weibo.py` already
sets.

**The publisher's own preprocessing is reproduced, not approximated.** CasFlow's
`gene_cas.py` applies three corpus-specific publication filters before anything
else, and all three are protocol landmines:

  * **Weibo** keeps only posts published 08:00-18:00 Beijing time, so every cascade
    has at least six hours to accrue retweets before the 24-hour horizon. (CasFlow's
    own code uses 19 rather than 18 when the observation window is a whole number of
    hours, with the comment "end_hour is set to 19 in DeepHawkes and CasCN, but it
    should be 18". We reproduce the code, not the comment, and record which.)
  * **Twitter** drops hashtags first seen after April 10, so each has 15 days to
    grow inside the 32-day horizon.
  * **APS** drops papers published after 1997, so each has 20 years of citations.

A run that skips these is not on the same corpus as the published result tables,
whatever its cascade count says.
"""

import time
from datetime import date
from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.cascade_common import (
    Cascade,
    cascade_arrays,
    corpus_statistics,
    load_casflow_file,
    manual_download,
)

data_dir = Path(__file__).resolve().parent.parent / "raw" / "casflow"

drive_url = "https://drive.google.com/file/d/1o4KAZs19fl4Qa5LUtdnmNy57gHa15AF-/view"
baidu_url = "https://pan.baidu.com/s/1tWcEefxoRHj002F0s9BCTQ"

# CasFlow's own hour bound, and the disagreement it documents against its own
# baselines. 18 is what the paper describes; 19 is what its code uses when the
# observation window is a whole number of hours, to stay byte-compatible with
# DeepHawkes and CasCN. Both are here because `metadata.json` records which ran.
weibo_start_hour = 8
weibo_end_hour_paper = 18
weibo_end_hour_deephawkes = 19
# Beijing is UTC+8, and CasFlow's code adds the offset by hand rather than
# localizing, so the filter is timezone-invariant. Reproduced exactly.
weibo_utc_offset = 8

twitter_cutoff_month = 4
twitter_cutoff_day = 10
aps_cutoff_year = "1997"

instructions = f"""\
The CasFlow bundle is behind a Google Drive interstitial and cannot be fetched by
script. To install it:

  1. Open {drive_url}
     (Baidu mirror, extract code 1msd: {baidu_url})
  2. Download the archive and extract it.
  3. Copy the three per-corpus `dataset.txt` files into:
         {data_dir}/weibo/dataset.txt
         {data_dir}/twitter/dataset.txt
         {data_dir}/aps/dataset.txt

Each file is one cascade per line, five tab-separated fields:
    cascade_id \\t root \\t publish_time \\t n \\t path1:t1 path2:t2 ...

Each corpus name covers several published versions: confirm which one a published
row was computed on before quoting any MSLE against it."""


def bundle_path(corpus: str) -> Path:
    return manual_download(
        f"the CasFlow {corpus} corpus",
        data_dir / corpus,
        "dataset.txt",
        instructions,
    )


def _keep_weibo(publish_time: int, end_hour: int) -> bool:
    hour = int(time.strftime("%H", time.gmtime(float(publish_time)))) + weibo_utc_offset

    return weibo_start_hour <= hour < end_hour


def _keep_twitter(publish_time: int) -> bool:
    month = int(time.strftime("%m", time.localtime(float(publish_time))))
    day = int(time.strftime("%d", time.localtime(float(publish_time))))

    return not (month == twitter_cutoff_month and day > twitter_cutoff_day)


def _keep_aps(publish_time: int) -> bool:
    # CasFlow compares APS publish times as STRINGS against '1997'. Reproduced as a
    # numeric comparison on the YEAR, which agrees on every four-digit year and does
    # not depend on lexical ordering.
    #
    # The bundle's own field is an ISO date (`1893-07-01`), which
    # `parse_casflow_line` normalizes to a day ordinal so it shares the unit of the
    # elapsed times. A four-digit value is a bare year instead, which is what the
    # code here previously assumed the whole corpus was.
    year = publish_time if publish_time <= 9999 else date.fromordinal(publish_time).year

    return year <= int(aps_cutoff_year)


def publication_filter(corpus: str, deephawkes_hours: bool = False):
    """The corpus's own publication-time filter, as a predicate on publish time."""
    if corpus == "weibo":
        end_hour = (
            weibo_end_hour_deephawkes if deephawkes_hours else weibo_end_hour_paper
        )

        return lambda when: _keep_weibo(when, end_hour)

    if corpus == "twitter":
        return _keep_twitter

    if corpus == "aps":
        return _keep_aps

    raise ValueError(
        f"unknown CasFlow corpus {corpus!r}; choose one of weibo, twitter, aps"
    )


def load_bundle_cascades(
    corpus: str, path: Path, deephawkes_hours: bool = False, limit: int = 0
) -> tuple[list[Cascade], int]:
    """
    One bundle file as cascades on a contiguous id space, filtered as CasFlow does.

    Ids are renumbered here rather than in `relabel_cascades` because the bundle's
    own ids are already dense per corpus but overlap ACROSS corpora, and a graph
    store keyed by `graph_id` would otherwise silently reuse one corpus's node count
    for another.
    """
    keep = publication_filter(corpus, deephawkes_hours)
    raw = load_casflow_file(path, limit=limit)
    filtered = [cascade for cascade in raw if keep(cascade.publish_time)]

    print(
        f"    publication filter kept {len(filtered)}/{len(raw)} cascades "
        f"({corpus}: "
        + (
            f"{weibo_start_hour}:00-"
            f"{weibo_end_hour_deephawkes if deephawkes_hours else weibo_end_hour_paper}"
            ":00 Beijing"
            if corpus == "weibo"
            else "first seen on or before Apr 10"
            if corpus == "twitter"
            else f"published on or before {aps_cutoff_year}"
        )
        + ")"
    )

    participants = sorted(
        {
            node
            for cascade in filtered
            for node in cascade.participants() | {cascade.root}
        }
    )
    mapping = {node: index for index, node in enumerate(participants)}

    cascades = []
    for cascade in filtered:
        events = [
            (
                mapping[adopter],
                elapsed,
                None if parent is None else mapping.get(parent),
            )
            for adopter, elapsed, parent in cascade.events
        ]
        cascades.append(
            Cascade(
                cascade_id=cascade.cascade_id,
                root=mapping[cascade.root],
                publish_time=cascade.publish_time,
                events=events,
            )
        )

    return cascades, len(mapping)


def bundle_loaders(corpus: str) -> tuple:
    """
    `(download, load, load_cascades)` for one corpus of the bundle.

    Returned as a triple rather than defined per module because the three loader
    files differ only in a string; each one binds these and adds its own docstring,
    which is where the version, the published counts and the windows belong.
    """

    def download() -> Path:
        return bundle_path(corpus)

    def load_cascades(path: Path) -> list[Cascade]:
        cascades, num_nodes = load_bundle_cascades(corpus, path)
        corpus_statistics(f"CasFlow {corpus}", cascades, num_nodes)

        return cascades

    def load(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
        cascades, num_nodes = load_bundle_cascades(corpus, path)
        adjacency, node_feats, node_labels, _ = cascade_arrays(cascades, num_nodes)

        print(
            f"[OK] CasFlow {corpus} global graph: {num_nodes} nodes, "
            f"{adjacency.nnz // 2} undirected edges: this is the union of the "
            f"observed diffusion paths, which is what CasFlow's own "
            f"generate_global_graph builds"
        )

        return adjacency, node_feats, node_labels, num_nodes

    return download, load, load_cascades
