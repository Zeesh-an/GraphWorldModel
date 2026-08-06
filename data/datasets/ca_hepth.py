"""
CA-HepTh Dataset Loader (arXiv hep-th collaboration network)

Source: https://snap.stanford.edu/data/ca-HepTh.html
    - 8,638 nodes, 24,806 undirected edges as loaded (the GCC)
    - Undirected: an edge means the two authors co-wrote at least one paper
    - Sparse original ids: remapped through the adjacency
    - No inherent node features: uses log(1 + degree) as synthetic features
    - No node labels

Xiao SDM'18's `arxiv-hep-th` row, quoted at 8,638 / 24,827 [verified, §6], which
is this file's largest connected component: the same LCC preprocessing switch
`ca_grqc` documents, so `lcc=True` reproduces the published NODE count rather than
the file's 9,877.

Warning: our edge count is 24,806, twenty-one below Xiao's 24,827 [derived, counted
from the download on 2026-08-04]. The node count matches to the digit, so the gap is
preprocessing rather than a different graph: SNAP's file lists both directions of
every collaboration plus self-loops, and `edges_to_adjacency` drops the loops while
Xiao's pipeline evidently keeps some. Recorded rather than reconciled, because
matching it would mean guessing at a step that paper does not describe.

Warning: NOT `cit_hepth`. That is SNAP's hep-th CITATION network (27,769 nodes,
directed arcs) loaded for influence blocking; this is the CO-AUTHORSHIP network on
the same arXiv section. Two graphs, one subject area, a 3x difference in node
count.

Original paper: Leskovec et al., "Graph Evolution: Densification and Shrinking
    Diameters," ACM TKDD 2007
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_gzip, load_edge_list

ca_hepth_url = "https://snap.stanford.edu/data/ca-HepTh.txt.gz"


def download_ca_hepth() -> Path:
    return download_gzip("ca_hepth", ca_hepth_url, "ca-HepTh.txt.gz")


def load_ca_hepth(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("CA-HepTh (coauthorship, GCC)", path, lcc=True)
