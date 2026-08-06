"""
High school (Thiers11) Contact Network Loader

Source: https://sociopatterns.org/datasets.html
    - 126 nodes, 1,709 undirected edges, 28,561 timestamped contacts [derived]
    - Undirected, unweighted after aggregation
    - 4 class labels (PC, PC*, PSI*, teacher)
The 2011 Thiers wave, the smallest of the three and the cheapest of the family
after LH10. See `high_school_2012` for why all three are loaded.

Warning: 1,709 EDGES, NOT 1,710. `research/epidemic_control.md` §6.2 derives 1,710
by counting unique dyads including `(43, 43)`: the file carries one SELF-CONTACT,
which `edges_to_adjacency` drops with `setdiag(0)` like every other loader here.
The two counts describe the same graph and this one is the one an epidemic can
travel on; a self-loop transmits to nobody.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.sociopatterns import (
    build_contact_graph,
    download_trace,
    read_contacts,
)

high_school_2011_file = "highschool_2011.csv.gz"


def download_high_school_2011() -> Path:
    return download_trace("high_school_2011", high_school_2011_file)


def load_high_school_2011(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    dyads, labels, contacts = read_contacts([path], label_columns=(3, 4))

    return build_contact_graph("High school (Thiers11)", dyads, labels, contacts)
