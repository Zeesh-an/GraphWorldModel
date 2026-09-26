"""
Slashdot0902 Dataset Loader

Source: https://snap.stanford.edu/data/soc-Slashdot0902.html
    - 82,168 nodes, 870,161 arcs [derived]: SNAP publishes 948,464 raw lines, of
      which 78,303 are repeats of a pair already seen or self-loops
    - Directed (Slashdot Zoo friend/foe links, February 2009)
    - No inherent node features: uses log(1 + degree) as synthetic features
    - No node labels

Warning: THE COUNTS DISAGREE WITH THE PAPER THAT USES IT. The fair-IBM 2026 paper
reports its Slashdot row as 70,000 / 358,600 undirected, while SNAP's
own soc-Slashdot0902 page states 82,168 nodes and 948,464 arcs, which is what this
loader reports, from the file. Either they used a different snapshot (0811 is 77,357 /
905,468, also not a match) or an undirected largest-component extraction they did not
describe. Ours is the raw 0902 file; do not present it as reproducing their row.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_gzip, load_edge_list

slashdot_url = "https://snap.stanford.edu/data/soc-Slashdot0902.txt.gz"


def download_slashdot() -> Path:
    return download_gzip("slashdot", slashdot_url, "soc-Slashdot0902.txt.gz")


def load_slashdot(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("Slashdot0902", path, directed=True)
