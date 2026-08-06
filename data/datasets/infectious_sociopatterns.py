"""
INFECTIOUS: Stay Away Contact Network Loader (Science Gallery, Dublin)

Source: https://sociopatterns.org/datasets.html
    - 10,972 nodes, 44,517 undirected edges, 415,912 timestamped contacts [derived]
    - Undirected, unweighted after aggregation
    - No labels

Isella et al. (2011), 69 days of an art-science exhibition. By far the LARGEST
SocioPatterns trace and the only one at a scale where the spectral methods'
`O(N^3)` dense eigendecomposition starts to matter, which is what makes it the
family's scaling target.

Warning: THIS IS NOT `infectious`. That dataset key is Xiao ICDM'18's 410-node /
2,765-edge extraction of the SAME study, loaded for cascade reconstruction and
matching that paper's Table I to the digit. This is the full aggregated contact
graph at 10,972 nodes: a 27x difference in node count under one name, and the
same collision hazard `research/epidemic_control.md` §6.4 documents for
Hamsterster, PGP and Oregon. Each loader states which version it is.

Ships as a tarball of one `listcontacts_YYYY_MM_DD.txt` per exhibition day, each a
plain `t i j`; the aggregation unions all 69.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.sociopatterns import (
    base_url,
    build_contact_graph,
    raw_root,
    read_contacts,
)
from data.graph_utils import extract, fetch

infectious_sociopatterns_file = "sciencegallery_infectious_contacts.tgz"


def download_infectious_sociopatterns() -> Path:
    """The extraction directory: one file per exhibition day, all of them read."""
    directory = raw_root / "infectious_sociopatterns"
    days = sorted(directory.glob("listcontacts_*.txt"))

    if days:
        return directory

    extract(
        fetch(
            f"{base_url}/{infectious_sociopatterns_file}",
            directory / infectious_sociopatterns_file,
        ),
        directory,
    )

    return directory


def load_infectious_sociopatterns(
    path: Path,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    days = sorted(Path(path).glob("listcontacts_*.txt"))

    if not days:
        raise FileNotFoundError(
            f"no listcontacts_*.txt under {path}; the tarball did not extract"
        )

    dyads, labels, contacts = read_contacts(days)

    return build_contact_graph(
        "INFECTIOUS (Science Gallery, full trace)", dyads, labels, contacts
    )
