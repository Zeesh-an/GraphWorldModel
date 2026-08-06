"""
Higgs Twitter Dataset Loader

Source: https://snap.stanford.edu/data/higgs-twitter.html
    - 456,631 nodes, 14,855,875 arcs [verified, SNAP's own statistics page, NOT
      counted from the file, which we have deliberately not downloaded]
    - Directed (Twitter follower network around the Higgs discovery)
    - No inherent node features: uses log(1 + degree) as synthetic features
    - No node labels

Tong & Wu's source graph (NeurIPS 2018, §5.8). Warning: THEY DO NOT RUN ON THIS FILE,
they run on 10K- and 100K-node SUBGRAPHS of it, with activity-proportional
probabilities on the first and uniform p = 0.1 on the second, and describe neither
extraction. Their numbers are therefore not reproducible from this loader; it is here
because the full graph is the only published artifact and because it is the densest
graph in this literature (avg degree 65).
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_gzip, load_edge_list

higgs_twitter_url = "https://snap.stanford.edu/data/higgs-social_network.edgelist.gz"


def download_higgs_twitter() -> Path:
    return download_gzip("higgs_twitter", higgs_twitter_url, "higgs-social_network.edgelist.gz")


def load_higgs_twitter(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    return load_edge_list("Higgs Twitter", path, directed=True)
