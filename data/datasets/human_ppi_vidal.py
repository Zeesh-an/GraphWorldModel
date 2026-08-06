"""
Human PPI (Vidal) Dataset Loader

Source: http://konect.cc/networks/maayan-vidal/
    - KONECT quotes 3,133 nodes / 6,726 edges from the raw line count; we load
      **3,023 / 6,149** because the file carries self-loops and repeated pairs
      that `edges_to_adjacency` drops, and 110 proteins appear only in a
      self-loop. State which of the two a number belongs to before comparing.
    - Undirected, unweighted, 1-indexed
    - No inherent node features: uses log(1 + degree) as synthetic features
    - No node labels

Reported by DCRS and SPR (research/critical_node_detection.md §7), the two
strongest learned dismantlers with a printed per-network table, so this is one of
the few graphs where a learned baseline number can actually be checked.

Original paper: Rual et al., "Towards a proteome-scale map of the human
    protein-protein interaction network", Nature 437, 2005
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_archive, load_edge_list

human_ppi_vidal_url = "http://konect.cc/files/download.tsv.maayan-vidal.tar.bz2"


def download_human_ppi_vidal() -> Path:
    return download_archive(
        "human_ppi_vidal",
        human_ppi_vidal_url,
        "download.tsv.maayan-vidal.tar.bz2",
    )


def load_human_ppi_vidal(
    path: Path,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("Human PPI (Vidal)", path)
