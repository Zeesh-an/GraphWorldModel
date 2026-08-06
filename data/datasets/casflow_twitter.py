"""
CasFlow Twitter Cascade Corpus Loader (Twitter-A)

Weng et al.'s 2013 hashtag-adoption corpus (Mar 24 - Apr 25, 2012), as preprocessed
in CasFlow's bundle. **This is Twitter-A**, the first row of
`research/cascade_prediction.md` §6.4's five.

Source: CasFlow's Drive bundle (manual; see `casflow_bundle.instructions`)
    - Underlying graph 490,474 nodes / 1,903,230 edges [verified, CasFlow Table 2]
    - 88,440 cascades after filtering, avg size 142 [verified]
    - Observation windows 1 d (86400 s) and 2 d (172800 s); horizon 32 d (2764800 s)
    - Publication filter: hashtags first seen on or before Apr 10
    - Undirected here: the union of the observed adoption paths
    - No inherent node features - uses log(1 + degree)

Reported by: CasFlow, CCGL, MUCas (§7).

**Warning: the canonical host is dead.** §6.3 records
`carl.cs.indiana.edu/data/#virality2013` as 404 as of 2026-07-28, re-verified
2026-08-05, so CasFlow's mirror is the only route to this corpus and this loader is
the only way we reach it.

**Warning: four other things are called Twitter.** CasFT crawled its own 2022
hashtags (86,764 cascades), CTCP re-preprocessed to 19,718, CasTemp to 67,760, and
Topo-LSTM's "Twitter" is the Hodas-Lerman URL corpus at 569. A CasFT MSLE and a
CasFlow MSLE on "Twitter" are numbers about different data.
"""

from data.datasets.casflow_bundle import bundle_loaders

# Corpus time is SECONDS since publication
observation_windows = (86400, 172800)
prediction_horizon = 2764800

download_casflow_twitter, load_casflow_twitter, load_casflow_twitter_cascades = (
    bundle_loaders("twitter")
)
