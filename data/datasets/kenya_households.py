"""
Kenyan Households (Kilifi) Contact Network Loader

Source: https://sociopatterns.org/datasets.html
    - 75 nodes, 576 undirected edges, 32,643 contact records [derived]
    - Undirected, unweighted after aggregation
    - 5 household labels (B, E, F, H, L): a PHYSICAL block structure

Kiti et al. (2016), five rural Kenyan households over three days. The one trace in
this family whose community labels are households rather than institutional roles,
which makes it the closest empirical analogue of the SBM this pipeline generates
synthetically.

Warning: THE PUBLISHED NODE COUNT OF 47 IS WRONG, and the reason is this: member
ids restart at 1 in every household, so a global dedup on the
member id alone collapses distinct people. A person is the PAIR `(household,
member)`, which gives 75: B=15, E=17, F=8, H=29, L=6, verified against the file.
Netzschleuder's mirror reports the collapsed 47.

Warning: not a `tij` trace at all. Its rows carry `duration / day / hour` rather
than a timestamp, so there is no ordering to discard here: the aggregation warning
in `sociopatterns.py` applies to the day/hour resolution instead.
"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp

from data.datasets.sociopatterns import (
    build_contact_graph,
    download_archive,
    load_kilifi,
)

kenya_households_file = "scc2034_household_contact_dataset.zip"
kenya_households_members = (
    "scc2034_kilifi_all_contacts_within_households.csv",
    "scc2034_kilifi_all_contacts_across_households.csv",
)


def download_kenya_households() -> Path:
    """
    Both CSVs, returned as the WITHIN-household one.

    The across-household file is what connects the five components at all, so
    loading only one of them would give five disconnected households and an
    epidemic that cannot leave the one it starts in. `load_kenya_households` reads
    both from the returned file's directory.
    """
    for member in kenya_households_members[1:]:
        download_archive("kenya_households", kenya_households_file, member)

    return download_archive(
        "kenya_households", kenya_households_file, kenya_households_members[0]
    )


def load_kenya_households(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, int]:
    paths = [path.parent / member for member in kenya_households_members]
    dyads, labels, contacts = load_kilifi(paths)

    return build_contact_graph("Kenyan households (Kilifi)", dyads, labels, contacts)
