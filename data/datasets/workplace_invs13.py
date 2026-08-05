"""
Workplace (InVS13) Contact Network Loader

Source: https://sociopatterns.org/datasets.html
    - 92 nodes, 755 undirected edges, 9,827 timestamped contacts [derived]
    - Undirected, unweighted after aggregation
    - No labels in the tij file

The 2013 wave of the same office study `workplace_invs15` covers, and the SMALLEST
graph in the SocioPatterns family at 92 nodes. Loaded for the same reason the three
Thiers years are: two waves of one protocol on one building is a variance estimate
rather than a single draw.

Warning: shipped as a ZIP rather than a `.gz`, and the member inside is named
`tij_InVS.dat` — no `15`, no year. That is the only thing separating it from its
own successor on disk, which is why the extraction names the member explicitly.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.sociopatterns import (
    build_contact_graph,
    download_archive,
    read_contacts,
)

workplace_invs13_file = "workplace_InVS_tij.dat.zip"
workplace_invs13_member = "tij_InVS.dat"


def download_workplace_invs13() -> Path:
    return download_archive(
        "workplace_invs13", workplace_invs13_file, workplace_invs13_member
    )


def load_workplace_invs13(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    dyads, labels, contacts = read_contacts([path])

    return build_contact_graph("Workplace (InVS13)", dyads, labels, contacts)
