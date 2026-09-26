"""
email-EuAll Dataset Loader

Source: https://snap.stanford.edu/data/email-EuAll.html
    - 265,009 nodes, 418,956 arcs [derived]: SNAP publishes 265,214 / 420,045;
      we drop self-loops and the nodes left with no incident arc
    - Directed (EU research-institution email, all addresses)
    - No inherent node features: uses log(1 + degree) as synthetic features
    - No node labels

One of the eight-graph suite the Xie / SandIMIN node-blocking line reports across
four papers, and the large sibling of our `email_eu_core`: same institution, all
addresses rather than the 1,005-member core. SandIMIN's Table 5 reports decreased
spread on it at k = 10..50 as `EmailAll`, which is directly our metric.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_gzip, load_edge_list

email_euall_url = "https://snap.stanford.edu/data/email-EuAll.txt.gz"


def download_email_euall() -> Path:
    return download_gzip("email_euall", email_euall_url, "email-EuAll.txt.gz")


def load_email_euall(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("email-EuAll", path, directed=True)
