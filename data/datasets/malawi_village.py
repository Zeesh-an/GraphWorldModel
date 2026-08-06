"""
Malawi Village Contact Network Loader

Source: https://sociopatterns.org/datasets.html
    - 86 nodes, 347 undirected edges, 102,293 timestamped contacts [derived]
    - Undirected, unweighted after aggregation
    - No labels

The pilot deployment in a rural Malawian village (Ozella et al. 2021), and the
SPARSEST graph in the family by a wide margin: average degree 8.1 against the
primary school's 68.7 on a comparable node count. That is what makes it worth
loading rather than a fourth school: §8.3 names sparse graphs as exactly where
`acquaintance_immunization` most embarrasses methods that read the whole graph, and
this is the sparsest real contact graph available to test that on.

Warning: a CSV with a HEADER and an index column (`,contact_time,day,id1,id2`) so
the node ids are columns 3 and 4, not 1 and 2. It is the only file in this family
that is not `t i j`, apart from Kilifi.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.sociopatterns import (
    build_contact_graph,
    download_trace,
    read_contacts,
)

malawi_village_file = "tnet_malawi_pilot.csv.gz"


def download_malawi_village() -> Path:
    return download_trace("malawi_village", malawi_village_file)


def load_malawi_village(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    dyads, labels, contacts = read_contacts(
        [path], node_columns=(3, 4), delimiter=",", skip_header=True
    )

    return build_contact_graph("Malawi village", dyads, labels, contacts)
