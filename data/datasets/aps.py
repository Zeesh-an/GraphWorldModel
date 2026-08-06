"""
APS Citation Cascade Corpus Loader (from the publisher's own release)

The American Physical Society citation dataset, parsed directly rather than through
CasFlow's redistributed bundle. `research/cascade_prediction.md` §6.5 ranks APS the
corpus to do first, and §11 flags the one reason to have this route as well as
`casflow_aps`: **whether the redistributed bundle is licensed for our use was never
established**, and a licence problem would invalidate that recommendation. Obtaining
the release directly settles it.

Source: https://journals.aps.org/datasets (manual; a request form, 403 to `curl`)
    - 616,316 papers / 3,304,400 citation arcs in CasFlow's preprocessing
      [verified, CasFlow Table 2]
    - A CASCADE is a paper and the papers that cite it; elapsed time is DAYS from
      the cited paper's publication to the citing paper's
    - Undirected here: the union of the observed citation paths
    - No inherent node features - uses log(1 + degree)
    - No node labels

**A citation cascade is a star, not a tree, and that is the data rather than a
convention.** A cites B tells you B was adopted; it does not say A learned of B
through some third paper. So every citer is attributed to the cited paper, exactly
as CasFlow's own preprocessing does when it emits `root/citer:elapsed`. Chains of
length 3 exist in the bundle only because CasFlow's generator walks citation
lineage; the release itself records pairs.

**Warning: the publisher ships two archives and both are needed.** The citation
archive alone gives `(citing_doi, cited_doi)` with no dates; the metadata archive
gives each DOI its publication date. Without the second there is no elapsed time
and no cascade at all.

**Warning: our counts will not match CasFlow's.** §6.4 lists three APS versions
(207,685 / 48,575 / 90,768 cascades) and this is a fourth: our filters are the
pipeline's own (`--cp-min-size`, `--cp-truncate`), applied to whatever release year
you were granted. `casflow_aps` is the comparable one; this one is the licensed one.
"""

import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from data.datasets.cascade_common import (
    Cascade,
    cascade_arrays,
    corpus_statistics,
    manual_download,
)

data_dir = Path(__file__).resolve().parent.parent / "raw" / "aps"
aps_page_url = "https://journals.aps.org/datasets"

# APS publishes papers back to 1893; CasFlow keeps those published on or before
# 1997 so each has 20 years of citations inside the release window
default_cutoff_year = 1997

instructions = f"""\
The APS dataset is behind a request form and cannot be fetched by script. To
install it:

  1. Request access at {aps_page_url} (403 to curl; use a browser).
  2. Download BOTH archives:
       aps-dataset-citations-*.zip   (citing DOI -> cited DOI pairs)
       aps-dataset-metadata-*.zip    (per-DOI publication dates)
  3. Extract them and produce two files in {data_dir}/:
       citations.csv   two columns, `citing_doi,cited_doi`, with a header
       metadata.json   {{"<doi>": "YYYY-MM-DD", ...}}

     The citation archive already ships `citations.csv` in that shape. The metadata
     archive ships one JSON per article under a nested directory tree; collapse it
     with, from the extracted metadata root:

       python - <<'EOF'
       import json, pathlib
       out = {{}}
       for path in pathlib.Path('.').rglob('*.json'):
           record = json.loads(path.read_text())
           if record.get('id') and record.get('date'):
               out[record['id']] = record['date']
       pathlib.Path('metadata.json').write_text(json.dumps(out))
       EOF

research/cascade_prediction.md 6.5 explains why APS is the entry point; 11 records
that its licensing for redistribution is an open question, which is what this route
exists to sidestep."""


def download_aps() -> Path:
    return manual_download("the APS citation dataset", data_dir, "citations.csv", instructions)


def _publication_days() -> dict[str, int]:
    """`{doi: days since epoch}` from the collapsed metadata file."""
    path = data_dir / "metadata.json"

    if not path.exists():
        raise FileNotFoundError(
            f"the APS metadata file is missing at {path}, so no citation has an "
            f"elapsed time and no cascade can be built.\n{instructions}"
        )

    epoch = date(1970, 1, 1)
    days = {}

    for doi, stamp in json.loads(path.read_text()).items():
        parts = str(stamp).split("-")
        if len(parts) < 3:
            continue

        try:
            days[doi] = (date(int(parts[0]), int(parts[1]), int(parts[2])) - epoch).days
        except ValueError:
            continue

    return days


def load_aps_pair(
    path: Path, cutoff_year: int = default_cutoff_year
) -> tuple[list[Cascade], int]:
    """
    Citation pairs as one cascade per cited paper, on a contiguous id space.

    Elapsed time is DAYS from the cited paper's publication to the citing paper's,
    which is the unit CasFlow's APS windows are quoted in (1095 = 3 years, 7305 =
    20 years). A citation recorded as arriving BEFORE the cited paper was published
    is dropped rather than clamped: it is a metadata error, and clamping it to 0
    would put a phantom adopter inside every observation window.
    """
    published = _publication_days()
    cutoff = (date(cutoff_year, 12, 31) - date(1970, 1, 1)).days

    citers: dict[str, list[tuple[str, int]]] = {}
    dropped = 0

    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)

        # The shipped file has a header; a file without one must not lose its first row
        if header and not header[0].lower().startswith("citing"):
            handle.seek(0)
            reader = csv.reader(handle)

        for row in reader:
            if len(row) < 2:
                continue

            citing, cited = row[0].strip(), row[1].strip()
            when, origin = published.get(citing), published.get(cited)

            if when is None or origin is None or origin > cutoff:
                continue

            if when < origin:
                dropped += 1
                continue

            citers.setdefault(cited, []).append((citing, when - origin))

    if dropped:
        print(f"    dropped {dropped} citation(s) dated before the cited paper")

    papers = sorted(
        {doi for cited, rows in citers.items() for doi in [cited] + [c for c, _ in rows]}
    )
    mapping = {doi: index for index, doi in enumerate(papers)}

    cascades = []

    for cited, rows in sorted(citers.items()):
        root = mapping[cited]
        events = [(root, 0, None)]
        seen = {root}

        for citing, elapsed in sorted(rows, key=lambda pair: (pair[1], pair[0])):
            adopter = mapping[citing]
            if adopter in seen:
                continue

            seen.add(adopter)
            events.append((adopter, elapsed, root))

        if len(events) < 2:
            continue

        cascades.append(
            Cascade(
                cascade_id=cited,
                root=root,
                # APS cascades carry a publication DAY rather than a Unix timestamp,
                # which is what the chronological split partitions on
                publish_time=published[cited],
                events=events,
            )
        )

    return cascades, len(mapping)


def load_aps_cascades(path: Path) -> list[Cascade]:
    cascades, num_nodes = load_aps_pair(path)
    corpus_statistics("APS", cascades, num_nodes)

    return cascades


def load_aps(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """The graph: the union of the observed citation ties, symmetrized."""
    cascades, num_nodes = load_aps_pair(path)
    adjacency, node_feats, node_labels, _ = cascade_arrays(cascades, num_nodes)

    print(
        f"[OK] APS citation graph: {num_nodes} nodes, "
        f"{adjacency.nnz // 2} undirected edges"
    )

    return adjacency, node_feats, node_labels, num_nodes


# Corpus time is DAYS since publication (CasFlow's own unit for this corpus)
observation_windows = (1095, 1826)
prediction_horizon = 7305
