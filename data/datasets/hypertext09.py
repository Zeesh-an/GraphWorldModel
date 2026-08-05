"""
Hypertext 2009 conference (HT09) Contact Network Loader

Source: https://sociopatterns.org/datasets.html
    - 113 nodes, 2,196 undirected edges, 20,818 timestamped contacts [derived]
    - Undirected, unweighted after aggregation
    - No labels
The ACM Hypertext 2009 conference (Isella et al. 2011), and the second-smallest
graph in the family. A conference trace like SFHH but at a quarter of the size,
which makes it the cheapest place to sweep a protocol axis before paying for a
bigger graph.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.sociopatterns import (
    build_contact_graph,
    download_trace,
    read_contacts,
)

hypertext09_file = "ht2009_contact_list.dat.gz"


def download_hypertext09() -> Path:
    return download_trace("hypertext09", hypertext09_file)


def load_hypertext09(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    dyads, labels, contacts = read_contacts([path])

    return build_contact_graph("Hypertext 2009 conference (HT09)", dyads, labels, contacts)
