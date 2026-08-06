"""
rt-pol Dataset Loader (political retweet network)

Source: https://networkrepository.com/rt-pol.php
    - 18,470 nodes, 48,053 undirected edges
    - Undirected, unweighted; the raw file is `fr,to,unix_timestamp` per RETWEET,
      so an edge repeats once per retweet and the deduplicated count is what is
      quoted
    - No inherent node features: uses log(1 + degree) as synthetic features
    - No node labels

DITTO's `Pol` row [verified, Table 2], and one of its four REAL-diffusion
datasets: the retweet TIMES are the cascade, `T = 40` days, and it is the graph on
which DITTO's own F1 (.7471) barely clears its MLE baseline CRI (.7468) while both
supervised imputers OOM entirely (§5.1.2). That makes it the single most useful
scale reference in this literature.

Warning: we load the GRAPH only. The timestamps are dropped here, because our
episodes are simulated on the topology: using its real retweet times would be the
logged-trajectory experiment `research/cascade_reconstruction.md` §2.11 risk 4
calls the untested claim, not this loader.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.dismantling_common import download_archive, load_edge_list

rt_pol_url = "https://nrvis.com/download/data/rt/rt-pol.zip"


def download_rt_pol() -> Path:
    return download_archive("rt_pol", rt_pol_url, "rt-pol.zip", member="rt-pol.txt")


def load_rt_pol(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    # Comma-separated with a trailing timestamp column, which `read_pairs` drops;
    # DITTO takes the largest component before simulating
    return load_edge_list("rt-pol (retweets, GCC)", path, lcc=True)
