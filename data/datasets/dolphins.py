"""
Dolphins Dataset Loader

Source: https://networkrepository.com/soc-dolphins.php (Lusseau et al. 2003)
    - 62 nodes (bottlenose dolphins), 159 undirected edges (frequent associations)
    - Undirected, unweighted
    - No inherent node features: uses log(1 + degree) as synthetic features
    - No node labels

THE one benchmark graph the source-localization literature uses that we did not
already load: IVGD's Table 3, the
GraphSL package's Table 1 and the 2026 GNN source-detection benchmark's Table 3
all report it, and all three report 62 / 159. IVGD's row is the most useful of the
three (`LPSI FS 0.8717` against `IVGD FS 0.9701` on this graph) because it is
the only place a classical and a learned method are printed side by side on a
graph small enough that our own numbers converge in seconds.

Matrix Market coordinate format: `%` comment lines, then ONE uncommented dimension
header (`62 62 159`), then `row col` pairs in the lower triangle only.
`skip_rows=1` drops the header; `edges_to_adjacency` mirrors.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_archive, load_edge_list

dolphins_url = "https://nrvis.com/download/data/soc/soc-dolphins.zip"


def download_dolphins() -> Path:
    return download_archive(
        "dolphins", dolphins_url, "soc-dolphins.zip", member="soc-dolphins.mtx"
    )


def load_dolphins(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("Dolphins", path, skip_rows=1)
