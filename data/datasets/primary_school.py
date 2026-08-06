"""
Primary school (Lyon, temporal) Contact Network Loader

Source: https://sociopatterns.org/datasets.html
    - 242 nodes, 8,317 undirected edges, 125,773 timestamped contacts [derived]
    - Undirected, unweighted after aggregation
    - 11 class labels (10 classes plus Teachers): the ground-truth communities
§6.6 ranks this SECOND to add and gives the reason: it is the same order as
`jazz` (198 nodes) and it ships real class structure, which is the community
partition our SBM experiments only approximate. Stehlé et al. (PLoS ONE 2011).

Its average degree is 68.7 on 242 nodes: extremely dense for a contact graph,
because a school day puts every child in a room with their whole class repeatedly.
Expect an epidemic to reach nearly everyone unless the intervention cuts across
classes, which is exactly what makes the class labels worth having.

Warning: NOT the hospital ward, see `hospital_lh10` and §6.4.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.sociopatterns import (
    build_contact_graph,
    download_trace,
    read_contacts,
)

primary_school_file = "primaryschool.csv.gz"


def download_primary_school() -> Path:
    return download_trace("primary_school", primary_school_file)


def load_primary_school(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    dyads, labels, contacts = read_contacts([path], label_columns=(3, 4))

    return build_contact_graph("Primary school (Lyon, temporal)", dyads, labels, contacts)
