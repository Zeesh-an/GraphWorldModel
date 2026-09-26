"""
High school (Thiers13) Contact Network Loader

Source: https://sociopatterns.org/datasets.html
    - 327 nodes, 5,818 undirected edges, 188,508 timestamped contacts [derived]
    - Undirected, unweighted after aggregation
    - 9 class labels (2BIO1-3, MP, MP*1-2, PC, PC*, PSI*)
Loaded because it is the only multi-layer instance within reach:
Mastrandrea et al. (2015) published three DIFFERENT relations on the same students
this proximity trace, a contact diary (120 nodes / 502 directed arcs) and a
Facebook friendship graph (156 / 1,437 undirected) [all derived]. Only the
proximity layer is loaded: the other two are different relations rather than
contact networks, and an epidemic does not travel along a Facebook edge.

Warning: THE FILE AND THE PAPER DISAGREE, and the file wins. Thiers13's metadata
lists 329 students; 327 appear in the contacts [derived]. Same discrepancy as
SFHH (405 vs 403) and InVS15 (232 vs 217).
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.sociopatterns import (
    build_contact_graph,
    download_trace,
    read_contacts,
)

high_school_2013_file = "HighSchool2013_proximity_net.csv.gz"


def download_high_school_2013() -> Path:
    return download_trace("high_school_2013", high_school_2013_file)


def load_high_school_2013(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    dyads, labels, contacts = read_contacts([path], label_columns=(3, 4))

    return build_contact_graph("High school (Thiers13)", dyads, labels, contacts)
