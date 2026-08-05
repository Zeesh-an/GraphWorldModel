"""
UCI Students Dataset Loader (UC Irvine online message network)

Source: http://networkrepository.com/ia-fb-messages.php (Opsahl's UC Irvine log)
    - 1,266 nodes, 6,451 undirected edges
    - Undirected, unweighted, Matrix Market coordinate format
    - No inherent node features — uses log(1 + degree) as synthetic features
    - No node labels

Xiao ICDM'18's `student` row, at 1,266 / 6,451 with assortativity -0.0039
[verified, Table I], and one of Rozenshtein KDD'16's four real graphs (as
`Students`, a 100-node BFS subgraph of it). The second assortativity-neutral
control beside `email_univ`.

Warning: NAME COLLISION, recorded rather than resolved by renaming. Opsahl's full
UC Irvine message log is 1,899 nodes; networkrepository's `ia-fb-messages` is the
1,266-node version, which is the one Xiao's table quotes. Two graphs, one study,
a 50% difference in node count — the same hazard §6.4 of the other task files
documents for Digg and Epinions.

Matrix Market: `%` comments, one uncommented dimension header, then `row col`
pairs. `skip_rows=1` drops the header.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_archive, load_edge_list

uci_students_url = "https://nrvis.com/download/data/ia/ia-fb-messages.zip"


def download_uci_students() -> Path:
    return download_archive(
        "uci_students",
        uci_students_url,
        "ia-fb-messages.zip",
        member="ia-fb-messages.mtx",
    )


def load_uci_students(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("UCI Students (messages)", path, skip_rows=1)
