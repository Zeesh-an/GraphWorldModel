"""
Workplace (InVS15) Contact Network Loader

Source: https://sociopatterns.org/datasets.html
    - 217 nodes, 4,274 undirected edges, 78,249 timestamped contacts [derived]
    - Undirected, unweighted after aggregation
    - No labels in the tij file (the department metadata ships separately and 404s — §6.5)
An office building over two weeks (Génois & Barrat 2018). Its structure is the one
this family otherwise lacks: departments that barely mix, so it is nearly
block-diagonal and an intervention that finds the few between-department contacts
does disproportionately well.

Warning: 217, not 232. The dataset page lists 232 participants; 217 appear in the
contacts [derived, §6.2]. §6.5 records that the department metadata tarball 404s at
both the site's own (typo'd) path and the corrected one, so the labels are not
loaded — recover them from Netzschleuder's `sp_colocation` mirror if needed.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.sociopatterns import (
    build_contact_graph,
    download_trace,
    read_contacts,
)

workplace_invs15_file = "workplace_InVS15_tij.dat.gz"


def download_workplace_invs15() -> Path:
    return download_trace("workplace_invs15", workplace_invs15_file)


def load_workplace_invs15(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    dyads, labels, contacts = read_contacts([path])

    return build_contact_graph("Workplace (InVS15)", dyads, labels, contacts)
