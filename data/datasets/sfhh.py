"""
SFHH conference Contact Network Loader

Source: https://sociopatterns.org/datasets.html
    - 403 nodes, 9,565 undirected edges, 70,261 timestamped contacts [derived]
    - Undirected, unweighted after aggregation
    - No labels: a conference has no class structure to record
A two-day scientific conference (Génois & Barrat 2018), and the one trace here
with NO community structure at all: attendees mix freely, so its degree
distribution is close to homogeneous. That makes it the natural control against the
school traces: a method that only works by cutting between classes has nothing to
cut here.

Warning: 403, not 405. The dataset page says 405 participants; 403 appear in the
contacts [derived, §6.2].
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.sociopatterns import (
    build_contact_graph,
    download_trace,
    read_contacts,
)

sfhh_file = "SFHH_tij.dat.gz"


def download_sfhh() -> Path:
    return download_trace("sfhh", sfhh_file)


def load_sfhh(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    dyads, labels, contacts = read_contacts([path])

    return build_contact_graph("SFHH conference", dyads, labels, contacts)
