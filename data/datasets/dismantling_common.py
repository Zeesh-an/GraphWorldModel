"""
Shared loader body for the network-dismantling benchmark graphs.

Every graph in `research/critical_node_detection.md` §6.2 arrives as a plain
1-indexed edge list: the GND repo publishes raw `.txt`, KONECT ships `out.<name>`
inside a tarball, networkrepository ships Matrix Market inside a zip. Once the
bytes are on disk the four formats differ only in which leading lines to skip, so
one function does the parse and each `data/datasets/<name>.py` supplies the URL.

Warning: Nearly every dismantling paper silently runs on the LARGEST CONNECTED
COMPONENT (§8.2 trap 7), which is why the GND repo's files are named `*_Gcc`.
Loaders that fetch a raw file therefore pass `lcc=True` to reproduce the published
node count; loaders that fetch an already-extracted GCC leave it off.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.graph_utils import (
    degree_features,
    edges_to_adjacency,
    extract,
    fetch,
    konect_edge_file,
    largest_connected_component,
    read_pairs,
)

raw_root = Path(__file__).resolve().parent.parent / "raw"


def download_plain(name: str, url: str, filename: str) -> Path:
    """A bare edge-list file (the GND repo, SNAP `.txt`)."""
    return fetch(url, raw_root / name / filename)


def download_gzip(name: str, url: str, archive: str) -> Path:
    """
    A single gzipped edge list (every SNAP `.txt.gz`), returned already expanded.

    Separate from `download_plain` rather than sniffed inside it: a `.gz` handed to
    `read_pairs` fails with a UnicodeDecodeError on byte 0x8b, which reads as a
    corrupt download rather than as a missing decompress step.
    """
    directory = raw_root / name
    inner = directory / archive[: -len(".gz")]

    if inner.exists():
        return inner

    extract(fetch(url, directory / archive), directory)

    return inner


def download_archive(name: str, url: str, archive: str, member: str | None = None) -> Path:
    """
    Download and unpack an archive, returning the edge file inside it.

    `member` is the path relative to the extraction directory; leave it None for
    a KONECT tarball, whose payload is globbed as `out.*` because the inner name
    does not always match the download name.
    """
    directory = raw_root / name
    archive_path = fetch(url, directory / archive)

    if member is not None and (directory / member).exists():
        return directory / member

    extract(archive_path, directory)

    return directory / member if member is not None else konect_edge_file(directory)


def load_edge_list(
    title: str,
    path: Path,
    directed: bool = False,
    skip_rows: int = 0,
    lcc: bool = False,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    """
    Parse an edge list into the (adjacency, feats, labels, N) loader contract.

    Node ids are remapped to 0..N-1 through the adjacency itself rather than
    `remap_to_contiguous`, because these files are 1-indexed and several of them
    have gaps; building at max-id + 1 and then dropping the isolated rows keeps
    the mapping stable when `lcc` also drops fragments.
    """
    raw_edges = read_pairs(path, skip_rows=skip_rows)  # shape: (E, 2)
    size = int(raw_edges.max()) + 1

    adjacency = edges_to_adjacency(
        raw_edges[:, 0], raw_edges[:, 1], size, directed=directed
    )

    if lcc:
        adjacency, _ = largest_connected_component(adjacency)
    else:
        # Drop the unused id 0 (these files are 1-indexed) and any other node the
        # file never mentions, so N matches the published count
        total_degree = np.array(adjacency.sum(axis=1)).flatten() + np.array(
            adjacency.sum(axis=0)
        ).flatten()
        keep = np.flatnonzero(total_degree > 0)
        adjacency = adjacency[keep][:, keep]

    num_nodes = adjacency.shape[0]
    node_feats = degree_features(adjacency, directed=directed)  # shape: (N, 1)
    node_labels = np.zeros(num_nodes, dtype=np.int32)

    degrees = np.array(adjacency.sum(axis=1)).flatten()
    unit = "arcs" if directed else "undirected edges"
    count = adjacency.nnz if directed else adjacency.nnz // 2
    print(
        f"[ok] {title} loaded: {num_nodes} nodes, {count} {unit} "
        f"(from {raw_edges.shape[0]} raw lines)"
    )
    print(f"    Avg degree: {degrees.mean():.1f}, max degree: {degrees.max():.0f}")

    return adjacency, node_feats, node_labels, num_nodes
