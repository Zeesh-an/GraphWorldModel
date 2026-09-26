"""
Email-Enron Dataset Loader

Source: https://snap.stanford.edu/data/email-Enron.html
    - 36,692 nodes, 183,831 undirected edges
    - Undirected (Enron email communication)
    - No inherent node features: uses log(1 + degree) as synthetic features
    - No node labels

The proactive-rumour-control paper's mid-size graph. Its counts match the
published row digit for digit, which makes it one of the cleaner comparison targets
in a literature where most graphs are quoted three different ways.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_gzip, load_edge_list

email_enron_url = "https://snap.stanford.edu/data/email-Enron.txt.gz"


def download_email_enron() -> Path:
    return download_gzip("email_enron", email_enron_url, "email-Enron.txt.gz")


def load_email_enron(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("Email-Enron", path, directed=False)
