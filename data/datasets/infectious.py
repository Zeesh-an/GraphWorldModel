"""
SocioPatterns Infectious Dataset Loader (face-to-face contacts)

Source: http://konect.cc/networks/sociopatterns-infectious/
    - 410 nodes, 2,765 unique undirected edges (17,298 timestamped contact events)
    - Undirected; the raw file is `u v weight unix_timestamp` per CONTACT, so an
      edge repeats once per 20-second contact and the deduplicated count is what
      is quoted
    - No inherent node features: uses log(1 + degree) as synthetic features
    - No node labels

Xiao ICDM'18's `infectious` row, at 410 / 2,765 with assortativity 0.0121
[verified, Table I]. The SMALLEST graph in that table and the fastest place to
converge a reconstruction number, which is why it is worth loading even though its
own numbers in that paper are [figure] only.

Warning: the counts are the DEDUPLICATED graph. KONECT's own header says
`% 17298 410 410`, where 17,298 is the number of contact EVENTS; collapsing
repeated contacts is what produces the 2,765 the paper reports, and
`edges_to_adjacency` does it by construction.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_archive, load_edge_list

infectious_url = (
    "http://konect.cc/files/download.tsv.sociopatterns-infectious.tar.bz2"
)


def download_infectious() -> Path:
    return download_archive(
        "infectious", infectious_url, "sociopatterns-infectious.tar.bz2"
    )


def load_infectious(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("SocioPatterns Infectious", path)
