"""
Shared loader body for the SocioPatterns empirical CONTACT TRACES.

These are the field's canonical empirical networks and what makes epidemic control
on networks an empirical problem rather than a synthetic one: RFID proximity sensors worn by real people
at 20-second resolution, in a hospital ward, a school, a conference, an office, a
village.

Warning: AGGREGATING A CONTACT TRACE DESTROYS HALF THE PROBLEM, and every loader
here does it. Put plainly: a contact at `t = 10` followed by one
at `t = 20` transmits, and the reverse order does not, so flattening a `tij` stream
to a static graph throws away the ordering that makes the epidemic non-trivial.
Papers that aggregate report systematically LARGER outbreaks. Our pipeline consumes
a static adjacency, so we aggregate, and say so here and in the loader's own print
line rather than leaving a reader to discover it. There is a cheap route out:
`reconstruct_episode_adjacency`
already replays a different adjacency per step, so a per-step contact graph is
reachable without new machinery whenever it is worth building.

Warning: THESE GRAPHS HAVE NO PUBLISHED BASELINE. No paper in the
CS immunization line uses a SocioPatterns trace: that family belongs to the
epidemiology side (Vanhems, Stehlé, Génois & Barrat, Machens) and the CS line
standardized on Oregon + Portland + collaboration graphs. Adding them buys realism
and community structure, not a number to compare against.

**Every count below is [derived]**: SocioPatterns publishes no statistics page at
all, so each loader's docstring states what we counted from the file and each was
checked against an earlier independent count. All twelve
match to the digit.

One parser covers the family because the format is: `timestamp node_i node_j
[class_i class_j]`, whitespace or tab separated. Aggregation to a simple undirected
graph is `set()` over the dyads. Where the file carries class or department labels
they become `node_labels`, exactly as `email_eu_core`'s 42 departments do: the
primary school's ten classes are the ground-truth community structure our SBM
experiments only approximate.
"""

import csv
import gzip
from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.graph_utils import degree_features, edges_to_adjacency, extract, fetch

raw_root = Path(__file__).resolve().parent.parent / "raw"
base_url = "https://www.sociopatterns.org/assets/data"


def download_trace(name: str, filename: str) -> Path:
    """
    Fetch one SocioPatterns file, expanding a `.gz` and leaving an archive alone.

    A known hazard of these hosts applies here rather than to the URL: they return
    HTTP 200 and then stall mid-transfer on anything over ~10 MB. `fetch` writes
    the whole body before renaming nothing, so a stalled download leaves a short
    file that fails to parse rather than a silently truncated graph: delete the
    file under `data/raw/<name>/` and re-run if that happens.
    """
    directory = raw_root / name
    path = fetch(f"{base_url}/{filename}", directory / filename)

    if filename.endswith(".gz") and not filename.endswith(".tar.gz"):
        inner = directory / filename[: -len(".gz")]

        if not inner.exists():
            with gzip.open(path, "rb") as archive:
                inner.write_bytes(archive.read())

        return inner

    return path


def download_archive(name: str, filename: str, member: str) -> Path:
    """Fetch and unpack a `.zip` / `.tgz`, returning one member inside it."""
    directory = raw_root / name
    target = directory / member

    if target.exists():
        return target

    extract(fetch(f"{base_url}/{filename}", directory / filename), directory)

    return target


def read_contacts(
    paths: list[Path],
    node_columns: tuple[int, int] = (1, 2),
    label_columns: tuple[int, int] | None = None,
    delimiter: str | None = None,
    skip_header: bool = False,
) -> tuple[list[tuple], dict, int]:
    """
    (unique dyads, {node id: class label}, contact count) from one or more `tij` files.

    `node_columns` is (1, 2) for every `t i j ...` file, which is all of them but
    Malawi and Kenya. Node ids are the file's OWN ids and are remapped by the
    caller, because they are sparse (the hospital's run to 1,232 for 75 people).
    """
    dyads = set()
    labels = {}
    contacts = 0

    for path in paths:
        with open(path, errors="replace") as handle:
            if skip_header:
                next(handle, None)

            for line in handle:
                line = line.strip()
                if not line:
                    continue

                parts = (
                    line.split(delimiter) if delimiter is not None else line.split()
                )
                source = parts[node_columns[0]].strip()
                target = parts[node_columns[1]].strip()

                dyads.add((min(source, target), max(source, target)))
                contacts += 1

                if label_columns is not None and len(parts) > max(label_columns):
                    labels[source] = parts[label_columns[0]].strip()
                    labels[target] = parts[label_columns[1]].strip()

    return sorted(dyads), labels, contacts


def build_contact_graph(
    title: str,
    dyads: list[tuple],
    labels: dict,
    contacts: int,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Aggregated dyads -> the (adjacency, feats, labels, N) loader contract.

    The AGGREGATION is the load-bearing step and the print line says so: `contacts`
    timestamped interaction events collapse to `len(dyads)` undirected edges, and
    the ordering that decided which of them could transmit is gone. Contact COUNTS
    are dropped too rather than kept as weights, because the pipeline derives its
    own transmission probability from in-degree (`graph_utils.build_edge_index`) and
    a weighted adjacency here would be silently overwritten.
    """
    identifiers = sorted({node for dyad in dyads for node in dyad})
    index = {node: position for position, node in enumerate(identifiers)}
    num_nodes = len(identifiers)

    sources = np.array([index[dyad[0]] for dyad in dyads], dtype=np.int64)
    targets = np.array([index[dyad[1]] for dyad in dyads], dtype=np.int64)
    adjacency = edges_to_adjacency(sources, targets, num_nodes, directed=False)

    node_feats = degree_features(adjacency, directed=False)  # shape: (N, 1)

    if labels:
        classes = sorted({labels[node] for node in identifiers if node in labels})
        class_index = {name: position for position, name in enumerate(classes)}
        node_labels = np.array(
            [class_index.get(labels.get(node), -1) for node in identifiers],
            dtype=np.int32,
        )
    else:
        classes = []
        node_labels = np.zeros(num_nodes, dtype=np.int32)

    degrees = np.array(adjacency.sum(axis=1)).flatten()
    print(
        f"[ok] {title} loaded: {num_nodes} nodes, {adjacency.nnz // 2} undirected "
        f"edges (aggregated from {contacts} timestamped contacts: the ORDERING "
        f"is discarded)"
    )
    print(f"    Avg degree: {degrees.mean():.1f}, max degree: {degrees.max():.0f}")

    if classes:
        print(f"    {len(classes)} class labels: {', '.join(classes[:8])}"
              + (" ..." if len(classes) > 8 else ""))

    return adjacency, node_feats, node_labels, num_nodes


def load_kilifi(paths: list[Path]) -> tuple[list[tuple], dict, int]:
    """
    The Kenyan household study, which is the one file in this family that is not `tij`.

    Its rows carry `h1,m1,h2,m2,...,duration,day,hour` rather than a timestamp, and
    a person is identified by the PAIR `(household, member)`: member ids restart at
    1 in every household. Netzschleuder's node count of 47 is
    wrong for exactly this reason: a global dedup on the member id collapses
    distinct people. Keying on the pair gives 75 (B=15, E=17, F=8, H=29, L=6),
    verified against the file.

    The household letter becomes the class label, which makes this the only trace
    here with a ground-truth block structure that is also a physical one.
    """
    dyads = set()
    labels = {}
    contacts = 0

    for path in paths:
        with open(path, errors="replace") as handle:
            for row in csv.DictReader(handle):
                source = f"{row['h1'].strip()}-{row['m1'].strip()}"
                target = f"{row['h2'].strip()}-{row['m2'].strip()}"

                dyads.add((min(source, target), max(source, target)))
                labels[source] = row["h1"].strip()
                labels[target] = row["h2"].strip()
                contacts += 1

    return sorted(dyads), labels, contacts
