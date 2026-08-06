"""
CasFlow APS Cascade Corpus Loader (APS-A)

American Physical Society citation cascades (papers 1893-1997), as preprocessed in
CasFlow's bundle. **This is APS-A**, and `research/cascade_prediction.md` §6.5 ranks
it the corpus to do FIRST: no rate limits, no Chinese-platform registration, and
3-year/5-year windows long enough that one replayed timestep can be a whole year,
which maps onto our discrete-time simulator with no resampling at all.

Source: CasFlow's Drive bundle (manual; see `casflow_bundle.instructions`)
    - Underlying graph 616,316 nodes / 3,304,400 edges [verified, CasFlow Table 2]
    - 207,685 cascades after filtering, avg size 51 [verified]
    - Observation windows 3 y (1095 d) and 5 y (1826 d); horizon 20 y (7305 d)
    - Publication filter: papers published on or before 1997
    - **Corpus time is DAYS**, not seconds - the one corpus here where it is
    - Undirected here: the union of the observed citation paths
    - No inherent node features - uses log(1 + degree)

Reported by: CasCN, VaCas, CasFlow, CCGL, MUCas, CTCP, CasDO, CasFT (§7). CasFT's
own row at `t_o = 3 y` is **MSLE 1.2468 / MAPE 0.2282** against CasFlow's 1.4370 /
0.2401 [verified, §5.1] — and §5.3 is the reason to read those two numbers with
care, because under CasTemp's leak-free split the whole field moves from that
1.19-2.11 band to 2.28-4.82.

**Warning: two other things are called APS.** CTCP re-preprocessed to 48,575
cascades and CasTemp to 90,768. **Warning: licensing is unchecked.** §11 flags it as
an open gap — `journals.aps.org/datasets` requires a request form (403 to `curl`,
re-verified 2026-08-05) and whether the redistributed CasFlow bundle is licensed for
our use was never established. `data/datasets/aps.py` is the direct route for anyone
who obtains the release themselves.
"""

from data.datasets.casflow_bundle import bundle_loaders

# Corpus time is DAYS since publication, unlike the two social corpora
observation_windows = (1095, 1826)
prediction_horizon = 7305

download_casflow_aps, load_casflow_aps, load_casflow_aps_cascades = bundle_loaders(
    "aps"
)
