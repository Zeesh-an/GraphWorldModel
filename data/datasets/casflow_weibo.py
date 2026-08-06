"""
CasFlow Weibo Cascade Corpus Loader (Weibo-A)

The Sina Weibo retweet corpus from the DeepHawkes release, as preprocessed in
CasFlow's bundle. **This is Weibo-A**, the version `research/cascade_prediction.md`
§6.4 lists first and the one that carries almost every published number.

Source: CasFlow's Drive bundle (manual; see `casflow_bundle.instructions`)
    - Underlying graph 6,738,040 nodes / 15,249,636 edges [verified, CasFlow Table 2]
    - 119,313 cascades after filtering, avg size 240 [verified]
    - Observation windows 0.5 h (1800 s) and 1 h (3600 s); horizon 24 h (86400 s)
    - Publication filter: 08:00-18:00 Beijing only
    - Undirected here: the union of the observed retweet paths
    - No inherent node features - uses log(1 + degree)

Reported by: DeepHawkes, CasCN, VaCas, CasFlow, CCGL, MUCas, CasDO, CasFT (§7).
CasFT's own row at `t_o = 0.5 h` is **MSLE 2.1728 / MAPE 0.2448**, against CasFlow's
2.3370 / 0.2665 [verified, §5.1 Table 2].

**Warning: three other things are called Weibo.** CTCP's re-preprocessing is 39,076
cascades, CasTemp's 48,693, CoupledGNN's coverage-sampled subset 3,228, and our own
`--dataset weibo` is the AMiner FOLLOWING NETWORK with no cascades at all (§6.1).
A row is comparable only to rows on the same one.

**Warning: the global graph is not the one we run on.** 6.7M nodes exceeds this
pipeline (§6.5, and `research/influence_maximization.md` §6.1 flags the same graph);
what we build is the union of the observed diffusion paths, which is exactly
CasFlow's own `generate_global_graph` and is small enough to simulate.
"""

from data.datasets.casflow_bundle import bundle_loaders

# Corpus time is SECONDS since publication
observation_windows = (1800, 3600)
prediction_horizon = 86400

download_casflow_weibo, load_casflow_weibo, load_casflow_weibo_cascades = (
    bundle_loaders("weibo")
)
