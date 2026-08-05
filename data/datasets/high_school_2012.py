"""
High school (Thiers12) Contact Network Loader

Source: https://sociopatterns.org/datasets.html
    - 180 nodes, 2,220 undirected edges, 45,047 timestamped contacts [derived]
    - Undirected, unweighted after aggregation
    - 5 class labels (MP*1, MP*2, PC, PC*, PSI*)
The 2012 Thiers wave, and the middle size of the three. Loaded because the three
Thiers years are the closest thing this family has to a controlled repeat: same
school, same protocol, three different cohorts, so a method's spread across them
is a variance estimate rather than a single draw.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.sociopatterns import (
    build_contact_graph,
    download_trace,
    read_contacts,
)

high_school_2012_file = "highschool_2012.csv.gz"


def download_high_school_2012() -> Path:
    return download_trace("high_school_2012", high_school_2012_file)


def load_high_school_2012(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    dyads, labels, contacts = read_contacts([path], label_columns=(3, 4))

    return build_contact_graph("High school (Thiers12)", dyads, labels, contacts)
