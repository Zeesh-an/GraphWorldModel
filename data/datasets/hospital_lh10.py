"""
Hospital ward (Lyon, LH10) Contact Network Loader

Source: https://sociopatterns.org/datasets.html
    - 75 nodes, 1,139 undirected edges, 32,424 timestamped contacts [derived]
    - Undirected, unweighted after aggregation
    - 4 role labels (MED / NUR / ADM / PAT), which is the ward's own block structure
Warning: THE CANONICAL EMPIRICAL CONTACT NETWORK of this literature and the FIRST
one to add: 75 nodes, so the entire
pipeline runs in seconds, and it is the network every intervention paper on
real contact data cites. Vanhems et al. (PLoS ONE 2013) is the study.

Warning: NOT the Lyon primary school. LH10 (75 nodes, a hospital
ward) and LyonSchool (242 nodes, a primary school) are frequently conflated and are
separate studies on separate populations. This is the ward.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.sociopatterns import (
    build_contact_graph,
    download_trace,
    read_contacts,
)

hospital_lh10_file = "hospital_lyon_contacts.dat.gz"


def download_hospital_lh10() -> Path:
    return download_trace("hospital_lh10", hospital_lh10_file)


def load_hospital_lh10(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    dyads, labels, contacts = read_contacts([path], label_columns=(3, 4))

    return build_contact_graph("Hospital ward (Lyon, LH10)", dyads, labels, contacts)
