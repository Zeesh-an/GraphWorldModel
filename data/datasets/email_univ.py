"""
email-univ Dataset Loader (university email network)

Source: http://networkrepository.com/ia-email-univ.php
    - 1,133 nodes, 5,451 undirected edges
    - Undirected, unweighted, Matrix Market coordinate format
    - No inherent node features: uses log(1 + degree) as synthetic features
    - No node labels

Xiao ICDM'18's `email-univ` row, at 1,133 / 5,451 with assortativity -0.0007
[verified, Table I]. It is the ASSORTATIVITY-NEUTRAL member of that table and
therefore the control for `ca_grqc`: Xiao ICDM'18 found that Personalized
PageRank beats tree sampling on `grqc` (assortativity 0.164) and LOSES elsewhere,
and "elsewhere" is this graph. Running both is what turns that warning into a
measurement.

Matrix Market: `%` comment lines, then ONE uncommented dimension header
(`1133 1133 5451`), then `row col` pairs. `skip_rows=1` drops the header.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_archive, load_edge_list

email_univ_url = "https://nrvis.com/download/data/ia/ia-email-univ.zip"


def download_email_univ() -> Path:
    return download_archive(
        "email_univ", email_univ_url, "ia-email-univ.zip", member="ia-email-univ.mtx"
    )


def load_email_univ(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("email-univ", path, skip_rows=1)
