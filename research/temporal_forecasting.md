# Traffic and Temporal Graph Forecasting — Prior Work, Datasets, and Published Results

Spatio-temporal forecasting on graphs: predict the next `H` steps of a continuous node signal (traffic speed, flow, occupancy) from the last `P` steps, on a road network whose topology is fixed. This is the largest and most professionalized "dynamics on a graph" literature in existence — roughly a decade of work, six canonical datasets, four benchmark libraries, and a rigidly standardized evaluation protocol.

**Verdict up front: ❌ poor fit, and not a task to implement.** The state is continuous and periodic, not binary activation; there are **no interventions**, so `T_exo` — the action-conditioned half of our contribution — has nothing to do; the data is measured rather than simulated, so there are no counterfactual forks and no MC marginals; and the dataset overlap with our suite is exactly zero. §2 works that argument through axis by axis.

**What this file is actually for: §9.** This literature has thought harder about _autoregressive error accumulation over a multi-step horizon_ than any other graph community, because their headline metric is a 12-step rollout. Our `rollout_ensemble` saturation problem (`research_notes`, "solved but fragile") is their central engineering problem under a different name. Scheduled sampling, horizon curriculum learning, and the closed-loop/open-loop distinction are the transferable goods.

All URLs verified HTTP 200 on **2026-07-28** unless annotated otherwise.

---

## 0. Verification policy

| Tier           | Meaning                                                                                                              |
| -------------- | -------------------------------------------------------------------------------------------------------------------- |
| **[verified]** | Read from the paper's own table via text extraction, or from the data repository's own statistics page. Trustworthy. |
| **[derived]**  | Computed by us from a file we downloaded, or reconstructed arithmetically from published splits.                     |
| **[figure]**   | Read off a plotted figure — the paper published no table. Approximate, direction only.                               |
| **[claim]**    | Stated in prose by a paper or a secondary source; not cross-checked against a file or table.                         |

Every number in this file was extracted with `pdftotext -layout` from the paper's own PDF and transcribed by hand. No number here came from a summarizer.

**The comparability trap in this specific literature is the split.** METR-LA and PEMS-BAY use **70/10/20** train/val/test; PEMS03/04/07/08 use **60/20/20**. They are different conventions from different papers (DCRNN vs STSGCN) and mixing them silently invalidates a table. See §6.

---

## 1. Task definition

Given a graph `G = (V, E)` with `N = |V|` sensors and a fixed adjacency `A`, and a multivariate signal `X ∈ R^{T×N×C}` sampled at a uniform interval, learn

```
f_θ : X_{t-P+1 : t}  ─────▶  X_{t+1 : t+H}
```

with `P = H = 12` almost universally (one hour in, one hour out, at a 5-minute sampling interval). `C` is 1 for speed datasets (METR-LA, PEMS-BAY, PeMSD7) and 1 or 3 for flow datasets (PEMS03/04/07/08 carry flow, occupancy, speed; nearly every paper uses flow only).

**Loss and metrics are the same object.** Training minimizes masked MAE over the whole horizon; evaluation reports MAE / RMSE / MAPE, either averaged over the 12 steps (the PEMS0X convention) or reported per horizon at steps 3 / 6 / 12 = 15 / 30 / 60 minutes (the METR-LA / PEMS-BAY convention). Missing readings are masked out of both — a detail that materially changes the numbers, since METR-LA has a substantial fraction of zero-filled sensor gaps.

**Three structural properties define the task:**

1. **The signal is continuous and strongly periodic.** Daily and weekly cycles dominate; the single most informative feature is "what time of day is it, at which sensor". §4 covers STID, which is the paper that made this embarrassingly explicit.
2. **The topology is fixed and known.** `A` is built from road-network distance with a thresholded Gaussian kernel (DCRNN) or from raw connectivity (PEMS0X). A branch of the field (Graph WaveNet, MTGNN, AGCRN, GTS) instead _learns_ an adjacency from node embeddings, which turns out to be the main axis of architectural variation.
3. **The horizon is multi-step and autoregressive error compounds.** Every method must answer "how do I train a 12-step rollout when I only have teacher-forced targets". This is the part that matters to us.

### The two sub-literatures

|                | **Traffic forecasting**    | **General temporal graph learning**                         |
| -------------- | -------------------------- | ----------------------------------------------------------- |
| State          | continuous signal on nodes | node/edge _existence_ and embeddings                        |
| Topology       | fixed, known               | evolving — edges arrive over time                           |
| Time           | regular grid, 5-min        | discrete snapshots **or** continuous-time event stream      |
| Canonical work | DCRNN, Graph WaveNet, STID | EvolveGCN, ROLAND (discrete); TGN, JODIE, TGAT (continuous) |
| Task           | regression                 | link prediction / node classification                       |

The continuous-time branch (TGN and relatives) is already covered in `research_notes/Literature Review.md`; this file does not re-derive it and does not contradict it. §4.3 covers the discrete-time branch (EvolveGCN, ROLAND) and the taxonomy surveys that organize both.

### Why the task exists

Traffic management is the motivating application: signal timing, ramp metering, routing, incident response. Note that all four of those _are_ interventions — which is exactly what makes the **absence** of interventions from the benchmark formulation so striking, and is the subject of §2.

---

## 2. Fit with our methodology

Our model is `f_θ(G, s_t, a_t) → s_{t+1}` factorized as `s_{t+1} = T_endo(T_exo(s_t, a_t))`. Traffic forecasting is `f(X_{t-11:t}) → X_{t+1:t+12}`. The two look superficially similar — both are "next state on a graph" — and the similarity is entirely at the level of the encoder. Every other axis diverges.

### 2.1 State type — binary activation vs continuous periodic signal

|                    | **Ours**                                            | **Traffic forecasting**                            |
| ------------------ | --------------------------------------------------- | -------------------------------------------------- |
| State `s_t`        | `(infected, frontier)`, binary per node, 2 channels | speed / flow / occupancy, continuous, 1–3 channels |
| Support            | `{0,1}` (targets are MC marginals in `[0,1]`)       | `R⁺`, e.g. 0–70 mph, 0–1000 veh/5min               |
| Loss               | `BCEWithLogits` on both channels                    | masked MAE (sometimes Huber, STSGCN)               |
| Dominant structure | monotone spread from a frontier                     | daily + weekly periodicity                         |
| Head               | `ICTransmissionHead` / `LTThresholdHead`            | linear projection or a decoder RNN/CNN             |

Our two-channel BCE has no analogue: there is nothing to be "infected" by. More decisively, **our structured heads are mechanism models and the mechanism does not exist here.** `ICTransmissionHead` computes `p_new(v) = 1 − ∏_u (1 − q(u→v)·frontier_u)` — a noisy-OR over an active frontier. `LTThresholdHead` computes `[f_v > 0]·sigmoid(τ·(f_v − θ̂_v))` over an active-neighbour _fraction_. Both are built on a binary "active neighbour" set. Traffic has no frontier, no activation, no absorbing state, and no monotonicity; congestion propagates _and recedes_, which our IC head structurally cannot represent (IC is monotone by construction — `y_inf = infected + (1−infected)·p_new` can only grow).

The strongest thing you can say is that congestion propagation upstream along a road is _spread-like_. It is not IC or LT. Fitting it would mean discarding both structured heads and reverting to `--head linear`, which is precisely the configuration our own results file records as unfaithful as a simulator.

The periodicity point cuts the other way too. Our diffusion episodes have no time-of-day; theirs is the single strongest predictor in the data. Their models carry an explicit time-of-day / day-of-week embedding (STID's whole thesis, §4.2) that has no meaning in our feature table's 6 channels.

### 2.2 No interventions — the decisive objection

**None of our five action ops apply, and there is no `T_exo` to learn.**

| Op                | Traffic-forecasting analogue                                    |
| ----------------- | --------------------------------------------------------------- |
| `add_node`        | none — you cannot "activate" a road sensor                      |
| `remove_node`     | none — sensor failure exists but is missing-data, not an action |
| `add_edge`        | none — the road network is fixed in every benchmark             |
| `remove_edge`     | none — road closures are not in any benchmark dataset           |
| `set_edge_weight` | none — `A` is fixed at preprocessing time from road distance    |

Every method in §4 takes exactly one input tensor `X` and produces one output tensor `Ŷ`. There is no action argument anywhere in the field. Our feature table (`world_model/wm_data.py::build_features`, `IN_CHANNELS = 6`) would have channels 3, 4 and 5 — `CH_ADD`, `CH_REMOVE`, `CH_EDGE` — identically zero on every row of every episode. `reconstruct_episode_adjacency` would be a no-op. Three of our six input channels and one of our two transition factors would be dead code.

**This is the decisive objection, and it is worth stating bluntly: action conditioning _is_ our contribution.** A learned forward model without actions is a sequence model, and the traffic field already has fifteen better ones. Our one-step `action_sensitivity` metric, our `add_seed_success` / `remove_frontier_success` metrics, our counterfactual forks, and the entire planning-regret evaluation (`planning_regret_multi`) all become undefined. The coding-agent outer loop, which exists to _choose an action_ against the learned model, has nothing to choose.

There is a hypothetical action-conditioned traffic problem — road closures, signal timing, ramp metering as `set_edge_weight` — and it would be a genuinely interesting fit. **No benchmark for it exists in this literature.** Building one means building a traffic microsimulator (SUMO / CityFlow), at which point you have left the forecasting literature entirely and are in traffic-signal-control RL, which is a separate field with separate datasets and no comparability to anything in §5.

### 2.3 No simulator — no counterfactual forks, no MC marginals

Their data is **measured**: loop detectors on real freeways, aggregated to 5-minute bins. Ours is **generated** by NDlib. Two consequences, both fatal to our data recipe:

- **No counterfactual forks.** `data/wm_actions.py` re-applies different actions from the same `s_t` to produce `cf_i` branches — the signal that forces action-conditioning. You cannot re-run March 4th 2012 on the I-405 with a different action. There is exactly one realized trajectory.
- **No MC marginals.** `--mc-marginals 30` re-runs each step to estimate the true `P(infected)` / `P(frontier)` per node, and `build_features` **raises** if those targets are absent. Measured data has one sample per `(sensor, time)`. There is no distribution to average over, so our soft targets — recorded in `research_notes` as the change that lifted one-step `delta_f1` from 0.52 to 0.75 — are unavailable in principle, not just in practice.

`data/generate_wm_data.py` exists to produce exactly these two things. On this task it would have nothing to do.

### 2.4 Zero dataset overlap

Cross-referencing §6 against our fifteen loaded graphs (`research/influence_maximization.md` §6.1): **the intersection is empty.**

- Their graphs are 170–8,600 sensors of road network — planar, near-degree-2, geometric, with a Gaussian-kernel-weighted adjacency derived from driving distance. Ours are social / collaboration / citation graphs with heavy-tailed degree.
- Their "edge weight" is a distance kernel; ours is an IC transmission probability `p(u→v) = 1/in-degree(v)`. The two are not the same object even though both land in `edge_weight`.
- No traffic paper reports Jazz, Cora-ML, NetHEPT, or any social graph. No IM paper reports METR-LA. There is not one shared row.

Nothing in `data/datasets/` would be reused; every dataset in §6 would be a new loader with a new (T, N, C) tensor format that our `GraphBundle` contract does not express.

### 2.5 What genuinely transfers

Three things, and they are not nothing:

1. **The encoder backbones.** GCN, GAT, and Graph Transformer are shared vocabulary. STGCN uses ChebNet; Graph WaveNet uses a diffusion GCN; GMAN/PDFormer/STAEformer are attention stacks. Our five backbones would drop into their pipeline and vice versa. This is real but shallow — it is the part of the model that is already commodity.
2. **The multi-step rollout evaluation protocol.** Their headline number _is_ a 12-step free-running rollout scored per horizon. They report error at h=3, 6, 12 precisely so that error accumulation is visible in the table rather than averaged away. Our `rollout_ensemble` reports one aggregate `ens_count_bias`; reporting per-step bias would make our drift legible the way theirs is. §8.
3. **Scheduled sampling and horizon curriculum.** The actual prize. DCRNN integrates scheduled sampling explicitly to close the train/test distribution gap; the PEMS0X line uses curriculum learning on horizon length. Both attack exactly the failure mode our structured head fixed structurally and could regress on. §9 is the full treatment.

### 2.6 Verdict

**Do not implement this task.** It fails the folder's admission test on two of three criteria (`research/README.md`: a task earns a slot if it has (a) an evolving node/edge state, (b) an intervention expressible in the five ops, (c) a simulator to harvest transitions from). It has (a). It has neither (b) nor (c), and (b) is the one our contribution lives on.

**Do mine it for §9.** This is the one file in this folder whose value is a training technique rather than a benchmark.

| Axis                 | Verdict                                                                                  |
| -------------------- | ---------------------------------------------------------------------------------------- |
| State type           | ❌ continuous + periodic vs our binary/probabilistic; both structured heads inapplicable |
| Actions              | ❌ **none exist**; `T_exo` unused; 3 of 6 input channels dead                            |
| Simulator            | ❌ measured data — no counterfactual forks, no MC marginals (`build_features` raises)    |
| Datasets             | ❌ zero overlap with our fifteen graphs                                                  |
| Objective            | ❌ masked MAE regression vs BCE + planning regret                                        |
| Encoders             | ✅ shared — GCN / GAT / GT / (GCNII) all appear on both sides                            |
| Rollout evaluation   | ✅ per-horizon reporting is strictly better than our aggregate                           |
| Rollout **training** | ✅ ⭐ scheduled sampling + curriculum — the reason to read this file                     |

---

## 3. Classical and heuristic methods

These are the standing baselines in every table in §5. All are node-independent (no graph at all) except the last two rows, which is the point: the field's claim is that the graph adds something over them.

| Method                           | Idea                                                                            | Uses graph?        | Reference                                                                                                                                                      |
| -------------------------------- | ------------------------------------------------------------------------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **HA** (Historical Average)      | predict the seasonal mean for this (sensor, time-of-week)                       | no                 | folklore; DCRNN Table 1                                                                                                                                        |
| **ARIMA / ARIMA-Kalman**         | per-sensor Box-Jenkins with a Kalman filter                                     | no                 | Box & Jenkins, _Time Series Analysis_ — [doi:10.1002/9781118619193](https://doi.org/10.1002/9781118619193) (**403** to non-browser clients; publisher paywall) |
| **VAR** (Vector Auto-Regression) | linear, models pairwise cross-sensor lags                                       | implicitly (dense) | [Hamilton 1994, _Time Series Analysis_](https://press.princeton.edu/books/hardcover/9780691042893/time-series-analysis)                                        |
| **SVR**                          | linear support-vector regression per sensor                                     | no                 | [Drucker et al. 1996](https://papers.nips.cc/paper/1996/hash/d38901788c533e8286cb6400b40b386d-Abstract.html)                                                   |
| **FC-LSTM**                      | fully-connected LSTM encoder-decoder over all sensors                           | no                 | [Sutskever et al. 2014](https://arxiv.org/abs/1409.3215)                                                                                                       |
| **LGBM / DeepAR / N-BEATS**      | strong univariate/probabilistic TS baselines added by BasicTS+                  | no                 | [DeepAR](https://arxiv.org/abs/1704.04110) · [N-BEATS](https://arxiv.org/abs/1905.10437)                                                                       |
| **DLinear / NLinear / Linear**   | one linear layer on the flattened history — the "are deep models needed?" probe | no                 | [Zeng et al., AAAI 2023](https://arxiv.org/abs/2205.13504) · [code](https://github.com/cure-lab/LTSF-Linear)                                                   |

**Why they matter to us.** HA is the traffic analogue of our `persistence` baseline in `wm_eval.py` — a no-model reference whose change-metric is 0 by construction. Every table in §5 that omits a trivial baseline is unreadable for the same reason ours would be. The DLinear line is the direct ancestor of the STID result in §4.2.

---

## 4. Learning-based methods

### 4.1 Traffic forecasting — the canonical line

Every paper URL and code URL below returned HTTP 200 on **2026-07-28**.

| Method               | Year | Venue       | Idea                                                                                                                                                                                                                                     | Paper                                                              | Code                                                                                                                |
| -------------------- | ---- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| **STGCN**            | 2018 | IJCAI       | ChebNet spatial conv + 1-D gated temporal conv. First fully-convolutional ST model; no RNN.                                                                                                                                              | [arXiv 1709.04875](https://arxiv.org/abs/1709.04875)               | [VeritasYin/STGCN_IJCAI-18](https://github.com/VeritasYin/STGCN_IJCAI-18)                                           |
| **DCRNN** ⭐         | 2018 | **ICLR**    | Traffic as a **diffusion process** on a directed graph: bidirectional random-walk conv inside a GRU, seq2seq decoder, **scheduled sampling**. Defines the task, the datasets, and the 70/10/20 split.                                    | [arXiv 1707.01926](https://arxiv.org/abs/1707.01926)               | [liyaguang/DCRNN](https://github.com/liyaguang/DCRNN)                                                               |
| **ASTGCN**           | 2019 | AAAI        | Spatial + temporal attention on top of ChebNet; three parallel branches for recent / daily / weekly periodicity. Introduces PEMS04 and PEMS08.                                                                                           | [AAAI 3881](https://ojs.aaai.org/index.php/AAAI/article/view/3881) | [Davidham3/ASTGCN](https://github.com/Davidham3/ASTGCN)                                                             |
| **Graph WaveNet** ⭐ | 2019 | IJCAI       | Dilated causal TCN + diffusion conv, plus a **self-adaptive adjacency** `softmax(ReLU(E₁E₂ᵀ))` learned from node embeddings. Non-autoregressive: emits all 12 steps in one forward pass. Still competitive in 2024 (§5.2).               | [arXiv 1906.00121](https://arxiv.org/abs/1906.00121)               | [nnzhan/Graph-WaveNet](https://github.com/nnzhan/Graph-WaveNet)                                                     |
| **GMAN**             | 2020 | AAAI        | Encoder-decoder of spatial + temporal attention blocks with a transform-attention layer bridging history and horizon; targets long-horizon (60 min) error.                                                                               | [arXiv 1911.08415](https://arxiv.org/abs/1911.08415)               | [zhengchuanpan/GMAN](https://github.com/zhengchuanpan/GMAN)                                                         |
| **STSGCN**           | 2020 | AAAI        | Localized _spatio-temporal_ graph: stitches three adjacent time-slice graphs into one and convolves over it, capturing heterogeneity. **Introduces PEMS03/07 and the 60/20/20 split.** Huber loss.                                       | [AAAI 5438](https://ojs.aaai.org/index.php/AAAI/article/view/5438) | [Davidham3/STSGCN](https://github.com/Davidham3/STSGCN)                                                             |
| **AGCRN**            | 2020 | **NeurIPS** | Node-adaptive parameter learning (per-node weight factorization) + data-adaptive graph generation; needs **no predefined adjacency at all**.                                                                                             | [arXiv 2007.02842](https://arxiv.org/abs/2007.02842)               | [LeiBAI/AGCRN](https://github.com/LeiBAI/AGCRN)                                                                     |
| **MTGNN**            | 2020 | KDD         | Generalizes Graph WaveNet to arbitrary multivariate TS: graph-learning layer + mix-hop propagation + dilated inception. Trains with a **node-subset curriculum**.                                                                        | [arXiv 2005.11650](https://arxiv.org/abs/2005.11650)               | [nnzhan/MTGNN](https://github.com/nnzhan/MTGNN)                                                                     |
| **STGODE**           | 2021 | KDD         | Replaces stacked GCN layers with a **continuous-depth neural ODE** on the graph, so depth stops causing over-smoothing.                                                                                                                  | [arXiv 2106.12931](https://arxiv.org/abs/2106.12931)               | [square-coder/STGODE](https://github.com/square-coder/STGODE)                                                       |
| **D2STGNN** ⭐       | 2022 | VLDB        | **Decouples the signal into a diffusion component and an inherent component** and models each with its own branch, plus a dynamic graph learner. SOTA for STF in the BasicTS+ re-benchmark.                                              | [arXiv 2206.09112](https://arxiv.org/abs/2206.09112)               | [zezhishao/D2STGNN](https://github.com/zezhishao/D2STGNN)                                                           |
| **STID** ⭐          | 2022 | CIKM        | **An MLP plus three learned identity embeddings** (spatial, time-of-day, day-of-week). No graph convolution, no attention, no recurrence. Matches or beats most STGNNs at ~0.12M params. The field's most useful negative result — §4.2. | [arXiv 2208.05233](https://arxiv.org/abs/2208.05233)               | [zezhishao/STID](https://github.com/zezhishao/STID) · [GestaltCogTeam/STID](https://github.com/GestaltCogTeam/STID) |
| **STJGCN**           | 2023 | TKDE        | Spatio-temporal **joint** graph: connects every node to every node at every other time step within a window, over multiple ranges, with dilated aggregation.                                                                             | [arXiv 2111.13684](https://arxiv.org/abs/2111.13684)               | [zhengchuanpan/STJGCN](https://github.com/zhengchuanpan/STJGCN)                                                     |
| **PDFormer**         | 2023 | AAAI        | Transformer with a **propagation-delay-aware** attention: explicitly models that congestion takes time to travel between distant sensors, via a delay-matched key lookup.                                                                | [arXiv 2301.07945](https://arxiv.org/abs/2301.07945)               | [BUAABIGSCity/PDFormer](https://github.com/BUAABIGSCity/PDFormer)                                                   |
| **STAEformer** ⭐    | 2023 | CIKM        | Vanilla transformer + a **spatio-temporal adaptive embedding**. The ablation is the contribution: the embedding, not the architecture, carries the performance. Same lesson as STID.                                                     | [arXiv 2308.10425](https://arxiv.org/abs/2308.10425)               | [XDZhelheim/STAEformer](https://github.com/XDZhelheim/STAEformer)                                                   |

### 4.2 The "are GNNs even needed?" thread ⭐

The single most useful negative result in this field, and worth more to us than any leaderboard.

**STID's argument** (Shao et al., CIKM 2022): the reason STGNNs beat MLPs is not that they model spatial dependency — it is that they break the _indistinguishability_ of samples. Two windows with near-identical history at different sensors, or at different times of day, have different futures; a plain MLP cannot tell them apart. STID adds three learnable embedding tables — **spatial identity** (one vector per sensor), **time-of-day identity**, and **day-of-week identity** — concatenates them to the flattened history, and runs an MLP. That is the whole model.

BasicTS+ [verified, Table IV] reproduces the ablation independently:

| Data         | Metric | STID   | AGCRN  | STID\* (no identities) | AGCRN\* (identity adjacency) | Gap     |
| ------------ | ------ | ------ | ------ | ---------------------- | ---------------------------- | ------- |
| METR-LA      | MAE    | 3.12   | 3.16   | 3.58                   | 3.36                         | 12.85%↑ |
| METR-LA      | RMSE   | 6.49   | 6.44   | 7.24                   | 6.90                         | 10.35%↑ |
| PEMS-BAY     | MAE    | 1.56   | 1.60   | 1.80                   | 1.70                         | 13.33%↑ |
| PEMS-BAY     | RMSE   | 3.59   | 3.67   | 4.21                   | 3.96                         | 14.72%↑ |
| ExchangeRate | MAE    | 0.0325 | 0.0455 | **0.0312**             | 0.0421                       | 8.07%↓  |
| ETTm1        | MAE    | 1.63   | 2.29   | **1.41**               | 1.89                         | 21.16%↓ |

Read the ↓ rows: on datasets _without_ spatial indistinguishability, **removing the spatial modelling improves the model.** BasicTS+ concludes that "spatial indistinguishability is a strong indicator of spatial dependencies and that we do not always need to model spatial dependencies" [verified].

STAEformer (2023) is the same finding from the transformer side: a vanilla transformer plus an adaptive embedding reaches SOTA, so the embedding is doing the work.

**Why this matters to us.** Our BA-100 result (`research_notes`, 2026-06) is the same shape: five backbones tie at ~0.51 and the degree heuristic wins. That is our version of "the graph module is not earning its place." The traffic field diagnosed it with an _identity-embedding ablation_ — replace the spatial module with a free per-node embedding and see whether performance survives. Running that ablation on our encoder (swap the GNN for a learned per-node embedding + MLP) would be a cheap, decisive test of whether our backbones are doing structural work or just memorizing node identity.

### 4.3 Foundation models and LLM-based forecasters (2023–2026)

| Method        | Year | Venue    | Idea                                                                                                                                                                                           | Paper                                                | Code                                                                           |
| ------------- | ---- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------ |
| **TimeGPT-1** | 2023 | preprint | First "foundation model for time series": a transformer pre-trained on 100B data points, **zero-shot** forecasting on unseen series. Not graph-aware; the reference point everyone else cites. | [arXiv 2310.03589](https://arxiv.org/abs/2310.03589) | [Nixtla/nixtla](https://github.com/Nixtla/nixtla) (API client; weights closed) |
| **UniST**     | 2024 | KDD      | Universal _urban_ spatio-temporal model: masked pre-training across many cities/domains + spatio-temporal knowledge-guided prompts, for few-shot and zero-shot transfer to new cities.         | [arXiv 2402.11838](https://arxiv.org/abs/2402.11838) | [tsinghua-fib-lab/UniST](https://github.com/tsinghua-fib-lab/UniST)            |
| **UrbanGPT**  | 2024 | KDD      | Spatio-temporal encoder aligned into an LLM via instruction tuning; predicts in natural language and generalizes zero-shot across cities and tasks.                                            | [arXiv 2403.00813](https://arxiv.org/abs/2403.00813) | [HKUDS/UrbanGPT](https://github.com/HKUDS/UrbanGPT)                            |

The interesting thing about this branch for us is negative: **the move to foundation models made the graph _less_ central, not more.** UniST and UrbanGPT both compete on cross-city transfer, where a fixed road adjacency is exactly the thing that does not transfer. It is the same pressure STID applied from below.

### 4.4 General temporal graph learning (discrete-time branch)

Distinct from traffic: here the **topology** evolves and the task is link prediction or node classification, not signal regression.

**Taxonomy.** The field splits on how time is represented:

- **Discrete-time (DTDG)** — a sequence of snapshots `G_1, …, G_T`. A static GNN is applied per snapshot and an RNN ties the snapshots together. EvolveGCN, ROLAND, DySAT.
- **Continuous-time (CTDG)** — a stream of timestamped events, embeddings queryable at any `t`. TGN, JODIE, TGAT, DyRep. **Already covered in `research_notes/Literature Review.md`** (TGN entry) — this file does not restate it and does not contradict it.

| Method        | Year | Venue | Idea                                                                                                                                                                                                                                                                                                                  | Paper                                                | Code                                                            |
| ------------- | ---- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | --------------------------------------------------------------- |
| **EvolveGCN** | 2020 | AAAI  | Instead of evolving node embeddings, **evolve the GCN weight matrices themselves** with an RNN (`-H` variant uses a GRU over the weights, `-O` an LSTM). Handles nodes appearing and disappearing, which embedding-RNN methods cannot.                                                                                | [arXiv 1902.10191](https://arxiv.org/abs/1902.10191) | [IBM/EvolveGCN](https://github.com/IBM/EvolveGCN)               |
| **ROLAND** ⭐ | 2022 | KDD   | A _recipe_, not an architecture: treat node embeddings at each GNN layer as hierarchical states, update them across snapshots, and **repurpose any static GNN as a dynamic one**. Introduces the **live-update evaluation** setting and frames it as meta-learning; scales to 56M edges and 733 snapshots [verified]. | [arXiv 2208.07239](https://arxiv.org/abs/2208.07239) | [snap-stanford/roland](https://github.com/snap-stanford/roland) |

**Surveys.** Both are genuinely worth reading for the taxonomy, not the results:

- **Kazemi et al., "Representation Learning for Dynamic Graphs: A Survey"**, JMLR 2020 — the encoder/decoder framing that defines DTDG vs CTDG. [arXiv 1905.11485](https://arxiv.org/abs/1905.11485)
- **Longa et al., "Graph Neural Networks for Temporal Graphs: State of the Art, Open Challenges, and Opportunities"**, TMLR 2023 — the newer, more critical one; explicitly flags evaluation-protocol inconsistency across the field. [arXiv 2302.01018](https://arxiv.org/abs/2302.01018)

**ROLAND's live-update evaluation is the one idea in §4.4 with direct bearing on us.** It insists that at prediction time the model has only ever been trained on strictly earlier snapshots, then updates on the newly arrived snapshot — i.e. it enforces the same no-leakage discipline TGN's raw message store enforces, at snapshot granularity. If we ever train autoregressively on our own rollouts (§9), that discipline is the thing that stops us predicting the future from the future.

### 4.5 Benchmark libraries ⭐

This is where the field is unusually mature, and the reason §5 has a re-benchmarking section at all.

| Library                        | Year    | Venue      | Scope                                                                                                                                                                                                                  | Paper                                                | Code                                                                                                                                |
| ------------------------------ | ------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **PyTorch Geometric Temporal** | 2021    | CIKM       | PyG extension: discrete-time temporal GNN layers + iterators for static/dynamic signals; ships DCRNN, STGCN, EvolveGCN, A3T-GCN and others as layers. The easiest way to get these encoders.                           | [arXiv 2104.07788](https://arxiv.org/abs/2104.07788) | [benedekrozemberczki/pytorch_geometric_temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal)                 |
| **DL-Traff**                   | 2021    | CIKM       | Two paired benchmarks — grid-based (TaxiBJ, NYC-Bike/Taxi) and graph-based (METR-LA, PEMS-BAY, PEMSD7M) — under one codebase, to stop grid and graph papers being compared across incompatible setups.                 | [arXiv 2108.09091](https://arxiv.org/abs/2108.09091) | [DL-Traff-Graph](https://github.com/deepkashiwa20/DL-Traff-Graph) · [DL-Traff-Grid](https://github.com/deepkashiwa20/DL-Traff-Grid) |
| **LibCity**                    | 2021/23 | SIGSPATIAL | The broadest: **65 models** across **9 tasks** on **55 datasets** [verified] converted to a unified "atomic file" storage format. Covers traffic-state prediction, trajectory, ETA, map matching, road representation. | [arXiv 2304.14343](https://arxiv.org/abs/2304.14343) | [LibCity/Bigscity-LibCity](https://github.com/LibCity/Bigscity-LibCity)                                                             |
| **BasicTS / BasicTS+** ⭐      | 2023/24 | TKDE       | The one that matters: a unified training pipeline that **re-ran 45+ methods under a single fair protocol and overturned several published rankings**. §5.2 is its findings.                                            | [arXiv 2310.06119](https://arxiv.org/abs/2310.06119) | [GestaltCogTeam/BasicTS](https://github.com/GestaltCogTeam/BasicTS)                                                                 |

---

## 5. Published results

One representative original table (§5.1), then the re-benchmark that supersedes it (§5.2). Given the ❌ verdict in §2, transcribing the full leaderboard would be wasted effort — the two negative results in §5.2 are the payload.

### 5.1 DCRNN (ICLR 2018) — the canonical table

Protocol: `P = H = 12`, 5-min interval, **70/10/20** split, missing values excluded from the metrics. Reported at 15 / 30 / 60 min = horizon 3 / 6 / 12.

[verified, DCRNN Table 1]:

| Data     | T      | Metric | HA    | ARIMA-Kal | VAR   | SVR   | FNN   | FC-LSTM | **DCRNN** |
| -------- | ------ | ------ | ----- | --------- | ----- | ----- | ----- | ------- | --------- |
| METR-LA  | 15 min | MAE    | 4.16  | 3.99      | 4.42  | 3.99  | 3.99  | 3.44    | **2.77**  |
|          |        | RMSE   | 7.80  | 8.21      | 7.89  | 8.45  | 7.94  | 6.30    | **5.38**  |
|          |        | MAPE   | 13.0% | 9.6%      | 10.2% | 9.3%  | 9.9%  | 9.6%    | **7.3%**  |
| METR-LA  | 30 min | MAE    | 4.16  | 5.15      | 5.41  | 5.05  | 4.23  | 3.77    | **3.15**  |
|          |        | RMSE   | 7.80  | 10.45     | 9.13  | 10.87 | 8.17  | 7.23    | **6.45**  |
|          |        | MAPE   | 13.0% | 12.7%     | 12.7% | 12.1% | 12.9% | 10.9%   | **8.8%**  |
| METR-LA  | 60 min | MAE    | 4.16  | 6.90      | 6.52  | 6.72  | 4.49  | 4.37    | **3.60**  |
|          |        | RMSE   | 7.80  | 13.23     | 10.11 | 13.76 | 8.69  | 8.69    | **7.59**  |
|          |        | MAPE   | 13.0% | 17.4%     | 15.8% | 16.7% | 14.0% | 13.2%   | **10.5%** |
| PEMS-BAY | 15 min | MAE    | 2.88  | 1.62      | 1.74  | 1.85  | 2.20  | 2.05    | **1.38**  |
|          |        | RMSE   | 5.59  | 3.30      | 3.16  | 3.59  | 4.42  | 4.19    | **2.95**  |
|          |        | MAPE   | 6.8%  | 3.5%      | 3.6%  | 3.8%  | 5.19% | 4.8%    | **2.9%**  |
| PEMS-BAY | 30 min | MAE    | 2.88  | 2.33      | 2.32  | 2.48  | 2.30  | 2.20    | **1.74**  |
|          |        | RMSE   | 5.59  | 4.76      | 4.25  | 5.18  | 4.63  | 4.55    | **3.97**  |
|          |        | MAPE   | 6.8%  | 5.4%      | 5.0%  | 5.5%  | 5.43% | 5.2%    | **3.9%**  |
| PEMS-BAY | 60 min | MAE    | 2.88  | 3.38      | 2.93  | 3.28  | 2.46  | 2.37    | **2.07**  |
|          |        | RMSE   | 5.59  | 6.50      | 5.44  | 7.08  | 4.98  | 4.96    | **4.74**  |
|          |        | MAPE   | 6.8%  | 8.3%      | 6.5%  | 8.0%  | 5.89% | 5.7%    | **4.9%**  |

**Read the horizon axis, not the method axis.** DCRNN's METR-LA MAE goes 2.77 → 3.15 → 3.60 as the horizon goes 3 → 6 → 12: **+30% error over 12 autoregressive steps.** ARIMA-Kalman goes 3.99 → 6.90, **+73%**. HA is flat by construction (4.16 at every horizon) because it never rolls anything forward. That spread _is_ the error-accumulation problem, made legible by reporting per horizon instead of averaging. Our `rollout_ensemble` reports one aggregate `ens_count_bias`; this table is the argument for reporting ours per step. §8.

### 5.2 BasicTS+ (TKDE 2024) ⭐ — the re-benchmark that overturned rankings

This is the section worth reading. BasicTS+ built a unified training pipeline (unified dataloader, runner, and metric implementation) and re-ran 45+ methods. Three findings.

**Finding 1 — published numbers for the _same_ method on the _same_ dataset vary by up to 33%.** BasicTS+ compiled what different papers report for Graph WaveNet and DCRNN as baselines, all under nominally identical settings (12→12, MAE / RMSE / MAPE) [verified, Table I]:

| Method    | Source                                  | PEMS04 MAE  | RMSE        | MAPE         | PEMS08 MAE  | RMSE        | MAPE         |
| --------- | --------------------------------------- | ----------- | ----------- | ------------ | ----------- | ----------- | ------------ |
| GWNet     | as reported in [22],[62],[23],[63],[57] | 25.45       | 39.70       | 17.29%       | 19.13       | 31.05       | 12.68%       |
| GWNet     | as reported in [25],[64]                | 24.89       | 39.66       | 17.29%       | 18.28       | 30.04       | 12.15%       |
| GWNet     | as reported in [46]                     | 19.36       | 31.72       | 13.31%       | 15.07       | 23.85       | 9.51%        |
| GWNet     | as reported in [24]                     | 28.15       | 39.88       | 18.52%       | 20.30       | 30.82       | 13.84%       |
| **GWNet** | **BasicTS+**                            | **18.80**   | **30.14**   | **13.19%**   | **14.67**   | **23.55**   | **9.46%**    |
|           | _gap vs worst_                          | 33.21%↑     | 24.42%↑     | 28.78%↑      | 27.73%↑     | 23.59%↑     | 31.64%↑      |
| DCRNN     | as reported in [22],[62],[23],[63],[57] | 23.65–24.70 | 37.12–38.12 | 16.05–17.12% | 16.82–18.22 | 26.36–28.29 | 10.92–11.56% |
| **DCRNN** | **BasicTS+**                            | **19.66**   | **31.18**   | **13.45%**   | **15.23**   | **24.17**   | **10.21%**   |
|           | _gap vs worst_                          | 20.40%↑     | 12.20%↑     | 21.43%↑      | 16.41%↑     | 14.56%↑     | 11.67%↑      |

Both methods have public code. BasicTS+ attributes the spread to three "overlooked" sources [verified]: **data processing** (min-max vs z-score normalization), **training configuration** (masked MAE vs naive MAE; gradient clipping; curriculum learning), and **evaluation implementation** (outlier handling, mini-batch metric aggregation). None of these are in anyone's method section.

**The comparability lesson for us is exact.** A baseline you cite is a baseline you ran under someone else's pipeline. This is precisely why our `pipeline/conditions.py` requires `--compare`: every arm's own `reward` comes from its own evaluator, and only the shared ground-truth MC replay is comparable. BasicTS+ is independent confirmation that the discipline is necessary, from a field that learned it the hard way.

**Finding 2 — five years of architectures bought ~6%.** The full re-benchmark, MAE only, all six datasets [verified, BasicTS+ Table V]. Param counts are millions; **Speed** is seconds/epoch on their hardware.

| Data          | LGBM  | DeepAR | N-BEATS | STGCN | DCRNN | GWNet     | DGCRN  | D²STGNN   | AGCRN | MTGNN | StemGNN | GTS   | STEP      | STNorm   | **STID** |
| ------------- | ----- | ------ | ------- | ----- | ----- | --------- | ------ | --------- | ----- | ----- | ------- | ----- | --------- | -------- | -------- |
| METR-LA       | 5.03  | 3.33   | 3.79    | 3.11  | 3.03  | 3.03      | 2.94   | **2.88**  | 3.16  | 3.05  | 3.72    | 3.13  | 2.93      | 3.14     | 3.12     |
| PEMS-BAY      | 2.10  | 1.70   | 1.95    | 1.63  | 1.59  | 1.59      | 1.58   | 1.52      | 1.60  | 1.60  | 1.99    | 1.68  | **1.48**  | 1.58     | 1.56     |
| PEMS03        | 20.56 | 16.63  | 19.71   | 15.83 | 15.54 | **14.59** | 14.60  | 14.63     | 15.24 | 14.85 | 16.95   | 15.41 | N/A       | 15.32    | 15.33    |
| PEMS04        | 26.56 | 20.64  | 25.30   | 19.76 | 19.66 | 18.80     | 18.84  | **18.32** | 19.28 | 19.13 | 22.98   | 21.32 | **18.32** | 19.21    | 18.35    |
| PEMS07        | 29.64 | 22.00  | 26.14   | 22.25 | 21.16 | 20.44     | 20.04  | **19.49** | 20.68 | 21.01 | 22.50   | 22.47 | N/A       | 20.59    | 19.61    |
| PEMS08        | 21.29 | 16.80  | 18.91   | 16.19 | 15.23 | 14.67     | 14.77  | 14.10     | 15.78 | 15.25 | 16.90   | 16.92 | **14.00** | 15.39    | 14.21    |
| _Params (LA)_ | —     | 0.10   | 8.07    | 0.25  | 0.37  | 0.31      | 0.20   | 0.39      | 0.75  | 0.41  | 1.20    | 38.49 | 40.48     | 0.22     | **0.12** |
| _Speed (LA)_  | —     | 24.48  | 11.36   | 21.01 | 94.87 | 27.70     | 128.84 | 152.33    | 28.22 | 24.37 | 16.19   | 52.23 | 497.26    | **7.50** | ?        |

⚠️ **The `_Speed (LA)_` row was transcribed one cell short** and the missing value could not be recovered. Fifteen values were extracted for sixteen columns, so the cells are shown in their extracted order with a `?` appended. **This means the assignment of `**7.50**` to STNorm rather than STID is unverified** — the `?` may belong anywhere at or after the STNorm column. Do not quote a per-model speed from this row without re-reading BasicTS+ Table V. Every other row in this table has its full sixteen cells and is unaffected.

BasicTS+'s own reading, in their words [verified]:

> "although D²STGNN, published in 2022, is the state-of-the-art for STF prediction, its MAE on METR-LA is only 6% higher than that of Graph WaveNet, published in 2019. In addition, it is even more surprising that Graph WaveNet and its variant MTGNN are still able to significantly outperform many newer solutions, including StemGNN, GTS, and others."

> "compared to improving prediction accuracy by designing increasingly complex models, more progress may be achieved by focusing on other important and challenging issues, such as efficiency, graph structure learning."

And on learned adjacency specifically: **"learning a graph structure can be very challenging. Among the different solutions, only MTGNN and STEP are capable of learning effective graph structures that do not significantly degrade the prediction performance"** [verified]. Roughly half the field's architectural innovation — learn the graph instead of using the road network — is reported here as _net harmful_ when run under a fair pipeline.

**Finding 3 — STID, an MLP, is on the podium at 0.12M params and 3× the speed of anything else.** Read the STID column against GWNet: **better** on PEMS04 (18.35 vs 18.80), PEMS07 (19.61 vs 20.44), PEMS08 (14.21 vs 14.67); a hair worse on METR-LA (3.12 vs 3.03) and PEMS03. It has no graph convolution. See §4.2 for why, and for the ablation that isolates the cause.

### 5.3 Original PEMS0X numbers, for contrast

The same graphs, as first published [verified, STSGCN AAAI 2020 Table 2], MAE ± std over 10 runs, **60/20/20** split:

| Data   | VAR   | SVR        | LSTM       | DCRNN      | STGCN      | ASTGCN(r)  | STG2Seq    | Graph WaveNet | **STSGCN**     |
| ------ | ----- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ------------- | -------------- |
| PEMS03 | 23.65 | 21.97±0.00 | 21.33±0.24 | 18.18±0.15 | 17.49±0.46 | 17.69±1.43 | 19.03±0.51 | 19.85±0.03    | **17.48±0.15** |
| PEMS04 | 23.75 | 28.70±0.01 | 27.14±0.20 | 24.70±0.22 | 22.70±0.64 | 22.93±1.29 | 25.20±0.45 | 25.45±0.03    | **21.19±0.10** |
| PEMS07 | 75.63 | 32.49±0.00 | 29.98±0.42 | 25.30±0.52 | 25.38±0.49 | 28.05±2.34 | 32.77±3.21 | 26.85±0.05    | **24.26±0.14** |
| PEMS08 | 23.46 | 23.25±0.01 | 22.20±0.18 | 17.86±0.03 | 18.02±0.14 | 18.61±0.40 | 20.17±0.49 | 19.13±0.08    | **17.13±0.09** |

Compare row-wise with §5.2. Graph WaveNet on PEMS04 is **25.45** here and **18.80** under BasicTS+ — a 26% swing on the same model and the same data, with the ranking against DCRNN reversed (25.45 > 24.70 here; 18.80 < 19.66 there). **Two published tables, opposite conclusions, same experiment.** That is the whole case for §5.2 in one comparison.

---

## 6. Datasets

### 6.1 What we already load

**None.** The intersection between this literature's benchmark suite and our fifteen graphs (`research/influence_maximization.md` §6.1) is empty. Nothing in `data/datasets/` is reusable and no dataset below would be added — see §2.4.

### 6.2 The six canonical graphs

Sampling interval is **5 minutes** for every dataset in this table, so `frames = days × 288`. Edge counts are **directed arcs** in the sensor graph as shipped (these are road networks, stored directed).

| Dataset      | Nodes | Edges                                                                        | Frames | Interval | Time span                       | Signal             | **Split**    |
| ------------ | ----- | ---------------------------------------------------------------------------- | ------ | -------- | ------------------------------- | ------------------ | ------------ |
| **METR-LA**  | 207   | 1,515 [verified, LargeST T1] / 1,722 [verified, D²STGNN T2]                  | 34,272 | 5 min    | 03/01/2012 – 06/27/2012 (~4 mo) | speed              | **70/10/20** |
| **PEMS-BAY** | 325   | 2,369 [verified, LargeST] / 2,694 [verified, D²STGNN]                        | 52,116 | 5 min    | 01/01/2017 – 06/30/2017 (~6 mo) | speed              | **70/10/20** |
| **PEMS03**   | 358   | 546 [verified, LargeST]                                                      | 26,208 | 5 min    | 09/01/2018 – 11/30/2018         | flow               | **60/20/20** |
| **PEMS04**   | 307   | 338 [verified, LargeST] / 340 [verified, PDFormer T1]                        | 16,992 | 5 min    | 01/01/2018 – 02/28/2018         | flow (+occ, speed) | **60/20/20** |
| **PEMS07**   | 883   | 865 [verified, LargeST] / 866 [verified, PDFormer]                           | 28,224 | 5 min    | 05/01/2017 – 08/31/2017         | flow               | **60/20/20** |
| **PEMS08**   | 170   | 276 [verified, LargeST] / 295 [verified, PDFormer] / 548 [verified, D²STGNN] | 17,856 | 5 min    | 07/01/2016 – 08/31/2016         | flow (+occ, speed) | **60/20/20** |

Node counts, spans and frame counts are consistent across all four sources. **Edge counts are not** — see §11; the discrepancies are undirected-vs-arc and self-loop conventions, the same class of trap documented in `research/README.md`'s edge-count convention note.

**Sources.** METR-LA and PEMS-BAY are Li et al.'s DCRNN release ([liyaguang/DCRNN](https://github.com/liyaguang/DCRNN) — README links a Google Drive / Baidu archive; the `.h5` + `adj_mx.pkl` pair is mirrored inside [DL-Traff-Graph/METRLA](https://github.com/deepkashiwa20/DL-Traff-Graph/tree/main/METRLA)). PEMS03/04/07/08 are Song et al.'s STSGCN release ([Davidham3/STSGCN](https://github.com/Davidham3/STSGCN)); PEMS04 and PEMS08 also ship directly in [Davidham3/ASTGCN/data](https://github.com/Davidham3/ASTGCN/tree/master/data). All six are pre-packaged by [BasicTS](https://github.com/GestaltCogTeam/BasicTS) and [LibCity](https://github.com/LibCity/Bigscity-LibCity). Upstream raw source for all of them is [Caltrans PeMS](https://pems.dot.ca.gov/) (registration required).

### 6.3 ⚠️ The split is the comparability trap

**This is the single most important fact in §6.** The two conventions come from two different papers and are never reconciled:

| Convention                      | Datasets                       | Origin                                                                                                                             |
| ------------------------------- | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| **70 / 10 / 20** train/val/test | METR-LA, PEMS-BAY              | DCRNN, ICLR 2018 [verified: *"70% of data is used for training, 20% are used for testing while the remaining 10% for validation"*] |
| **60 / 20 / 20** train/val/test | PEMS03, PEMS04, PEMS07, PEMS08 | STSGCN, AAAI 2020 [verified: *"We split all datasets with ratio 6 : 2 : 2"*] · LargeST inherits it [verified]                      |

Splits are **chronological**, not random — the test set is the final 20% of the timeline. Applying 60/20/20 to METR-LA, or 70/10/20 to PEMS04, silently produces numbers that cannot be compared to any published table. Every paper in §4 follows its dataset's convention; BasicTS+ and LibCity encode it in config.

Two further per-dataset conventions ride along with the split:

- **PEMS04 and PEMS08 carry three channels** (flow, occupancy, speed). Nearly every paper uses **flow only**. Using all three changes the numbers.
- **METR-LA has substantial missing data** encoded as zeros. DCRNN excludes them from the metric [verified: *"Missing values are excluded in calculating these metrics"*]; a model trained with unmasked MAE learns to predict zeros. This is one of BasicTS+'s three named sources of spurious variation (§5.2).

### 6.4 Other datasets in this literature

| Dataset                         | Size                           | Frames        | Interval       | Span                        | Notes                                                                                                                                                       | Source                                                                                                                           |
| ------------------------------- | ------------------------------ | ------------- | -------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **PeMSD7(M)**                   | 228 nodes, 1,664 edges         | 12,672        | 5 min          | 05/01–06/30/2012 (weekdays) | STGCN's own dataset; the "M" and "L" variants differ only in sensor count                                                                                   | [VeritasYin/STGCN_IJCAI-18](https://github.com/VeritasYin/STGCN_IJCAI-18/tree/master/data_loader)                                |
| **PeMSD7(L)**                   | 1,026 nodes, 14,534 edges      | 12,672        | 5 min          | 05/01–06/30/2012            | as above                                                                                                                                                    | as above                                                                                                                         |
| **LargeST — CA**                | 8,600 nodes, 201,363 edges     | 525,888       | 5 min          | 01/01/2017 – 12/31/2021     | **4.52B data points**; 25× METR-LA's node count and 15× its span. The scalability benchmark. 6:2:2 split [verified]                                         | [liuxu77/LargeST](https://github.com/liuxu77/LargeST)                                                                            |
| **LargeST — GLA**               | 3,834 nodes, 98,703 edges      | 525,888       | 5 min          | as above                    | Greater Los Angeles sub-region                                                                                                                              | as above                                                                                                                         |
| **LargeST — GBA**               | 2,352 nodes, 61,246 edges      | 525,888       | 5 min          | as above                    | Greater Bay Area sub-region                                                                                                                                 | as above                                                                                                                         |
| **LargeST — SD**                | 716 nodes, 17,319 edges        | 525,888       | 5 min          | as above                    | San Diego sub-region; the entry point                                                                                                                       | as above                                                                                                                         |
| **TaxiBJ**                      | 32×32 **grid** (not a graph)   | 22,459        | 30 min         | 4 periods, 2013–2016        | inflow/outflow; the canonical _grid_ benchmark (ST-ResNet). Grid ≠ graph — DL-Traff exists to keep the two from being compared                              | [TolicWang/DeepST](https://github.com/TolicWang/DeepST)                                                                          |
| **NYC-Bike / NYC-Taxi**         | grid (typically 16×8 or 20×10) | varies        | 30–60 min      | 2014–2016 windows           | pickup/dropoff demand; every paper crops its own window, so counts vary                                                                                     | [NYC TLC trips](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) · [Citi Bike](https://citibikenyc.com/system-data) |
| **Q-Traffic**                   | 45,148 road segments           | ~5.8M records | 15 min         | 04/01–05/31/2017            | Baidu Maps, Beijing; includes query sub-dataset (user route requests) — the closest thing in this field to an _exogenous input_, though still not an action | [JingqingZ/BaiduTraffic](https://github.com/JingqingZ/BaiduTraffic) [claim]                                                      |
| **England / LA loop detectors** | varies by crop                 | varies        | 15 min / 5 min | varies                      | UK Highways England WebTRIS and the raw Caltrans D7 loop archive; used ad hoc, no standard crop or split                                                    | [WebTRIS](https://webtris.highwaysengland.co.uk/) · [PeMS](https://pems.dot.ca.gov/) [claim]                                     |

All counts in this table are [verified] from LargeST Table 1 except the last three rows, which are [claim] — no paper table was extracted for TaxiBJ, NYC-Bike/Taxi, Q-Traffic, or the loop-detector archives, and their per-paper crops are not standardized (§11).

---

## 7. Which paper uses which

✔ = the paper reports that dataset in its own main table. Note the clean split down the middle: the METR-LA/PEMS-BAY column block and the PEMS0X block are two sub-communities with two different splits (§6.3), and few papers span both.

| Dataset                                    | STGCN'18 | DCRNN'18 | ASTGCN'19 | GWNet'19 | STSGCN'20 | AGCRN'20 | MTGNN'20 | GMAN'20 | STGODE'21 | D²STGNN'22 | STID'22 | STJGCN'23 | PDFormer'23 | STAEformer'23 | LargeST'23 | BasicTS+'24 |
| ------------------------------------------ | -------- | -------- | --------- | -------- | --------- | -------- | -------- | ------- | --------- | ---------- | ------- | --------- | ----------- | ------------- | ---------- | ----------- |
| METR-LA                                    |          | ✔        |           | ✔        |           |          | ✔        | ✔       |           | ✔          | ✔       | ✔         | ✔           | ✔             | ✔          | ✔           |
| PEMS-BAY                                   |          | ✔        |           | ✔        |           |          | ✔        | ✔       |           | ✔          | ✔       | ✔         | ✔           | ✔             | ✔          | ✔           |
| PEMS03                                     |          |          |           |          | ✔         |          |          |         | ✔         |            |         |           |             |               |            | ✔           |
| PEMS04                                     |          |          | ✔         |          | ✔         | ✔        |          |         | ✔         | ✔          | ✔       |           | ✔           | ✔             |            | ✔           |
| PEMS07                                     |          |          |           |          | ✔         |          |          |         | ✔         | ✔          | ✔       |           | ✔           | ✔             |            | ✔           |
| PEMS08                                     |          |          | ✔         |          | ✔         | ✔        |          |         | ✔         | ✔          | ✔       |           | ✔           | ✔             |            | ✔           |
| PeMSD7(M/L)                                | ✔        |          |           |          |           |          |          |         |           |            |         |           |             |               | ✔          |             |
| LargeST (CA/GLA/GBA/SD)                    |          |          |           |          |           |          |          |         |           |            |         |           |             |               | ✔          |             |
| ETT / Electricity / Weather / ExchangeRate |          |          |           |          |           |          | ✔        |         |           |            |         |           |             |               |            | ✔           |

**Two things to read off this matrix.** First, **BasicTS+ is the only row that spans everything** — which is exactly why it is the one paper that could detect the inconsistencies in §5.2. Second, the LTSF datasets (ETT, Electricity, ExchangeRate, Weather) appear in only two columns, and BasicTS+'s central heterogeneity claim is that they are _the wrong datasets_ for spatio-temporal methods because they have almost no spatial indistinguishability (§4.2, Table IV, the ↓ rows).

---

## 8. Evaluation protocol

### Metrics

Three, always, all computed on **re-normalized** (original-scale) values:

```
MAE  = mean |y − ŷ|
RMSE = sqrt(mean (y − ŷ)²)
MAPE = mean |y − ŷ| / |y|          (y = 0 excluded)
```

BasicTS+ adds **WAPE** = `Σ|y−ŷ| / Σ|y|`, arguing MAPE is unstable near zero — free-flow speed sensors and low-flow night hours both drive `|y| → 0`.

### Horizon reporting — two conventions

| Convention           | Datasets          | What is reported                                                   |
| -------------------- | ----------------- | ------------------------------------------------------------------ |
| **Per-horizon**      | METR-LA, PEMS-BAY | metrics at step 3 / 6 / 12 = 15 / 30 / 60 min, reported separately |
| **Horizon-averaged** | PEMS03/04/07/08   | one number averaged over all 12 steps                              |

`P = H = 12` in both. **The per-horizon convention is the one worth copying** — it is what makes error accumulation visible (§5.1: DCRNN 2.77 → 3.15 → 3.60). The averaged convention hides it, which is a real cost: a model that is excellent at h=1 and diverges by h=12 can post a respectable average.

### The traps, in order of how often they bite

1. **The split.** 70/10/20 vs 60/20/20, chronological. §6.3. This is the big one.
2. **Normalization.** z-score is the de-facto standard; some papers use min-max. BasicTS+ names this as a primary source of the 33% spread in §5.2, and adopts z-score as its default because it "generally yields superior performance" [verified].
3. **Masking.** Whether missing values (zeros) are excluded from the **loss** as well as the metric. Masked MAE is standard in the STGNN line; BasicTS+
   [verified]: _"most studies employ masked MAE for model training… In contrast,
   some studies adopt naive MAE as their optimization function, which tends to yield inferior results."_ This is a training-pipeline choice reported as a method difference.
4. **Metric scale.** Reporting on normalized rather than re-normalized data makes errors look small and is not comparable to anything. BasicTS+ Table II demonstrates the gap directly [verified] — e.g. Autoformer on ETTh1 reads MAE 0.483 normalized and 1.74 re-normalized.
5. **Channel selection.** Flow only vs flow+occupancy+speed on PEMS04/08 (§6.3).
6. **Training tricks reported as architecture.** Gradient clipping and curriculum learning are in some codebases and not others, and are almost never in the method section. §9 is about exactly one of these.

### What our evaluation should borrow

`world_model/wm_eval.py::rollout_ensemble` currently reports aggregate `ens_marg_mae`, `ens_count_w1`, `ens_count_bias`, and final counts. **Report them per rollout step, not aggregated.** `ens_count_bias` at h=1 vs h=5 vs h=10 is the difference between "our model is calibrated" and "our model is calibrated for two steps and then runs away" — and the saturation history in `research_notes` says that is precisely the distinction we need to keep watching. This is a reporting change, not a research change, and it is the cheapest thing in this file.

---

## 9. Implications for this project

**Verdict: do not implement this task. Do steal its rollout-training technique.**

§2 establishes why the task is out: no interventions, so `T_exo` is unused and action-conditioning — our actual contribution — has nothing to condition on; no simulator, so no counterfactual forks and no MC marginals; zero dataset overlap. Nothing below argues against that. Everything below is about §2.5 item 3.

### 9.1 The problem this literature solved that we also have

Our `rollout_ensemble` free-running rollout has a documented history of saturation: `research_notes` records the outcome head diverging with `ens_count_bias = +49`, fixed **structurally** by the per-edge transmission head (`--head structured`, `--pos-weight off`) down to `+0.27`, then flipping to `+2.03` on a node-ops-only retrain. Threshold tuning, data volume, input perturbation, sampling changes and DAgger all failed before the structural fix worked. The gate on retrains is now `ens_count_bias`.

That history says the drift is **suppressed by architecture, not by training**. The structured head is a hard constraint that makes runaway impossible; nothing in the training objective ever taught the model to be stable under its own predictions. That is fragile in exactly the way the `+2.03` regression showed — change the data distribution and the constraint stops being enough.

**This literature's entire multi-step protocol is that same problem, attacked from the training side.** Their headline number is a 12-step rollout; every one of them had to answer "how do I train something that will be fed its own output". Four answers, in increasing order of how much they cost us.

### 9.2 Technique 1 — scheduled sampling ⭐ cheapest, most direct

**DCRNN, ICLR 2018** [verified]:

> "The decoder generates predictions given previous ground truth observations. At testing time, ground truth observations are replaced by predictions generated by the model itself. **The discrepancy between the input distributions of training and testing can cause degraded performance.** To mitigate this issue, we integrate scheduled sampling into the model, where we feed the model with either the ground truth observation with probability `ε_i` or the prediction by the model with probability `1 − ε_i` at the `i`-th iteration. During the training process, `ε_i` gradually decreases to 0 to allow the model to learn the testing distribution."

That paragraph is a precise description of our situation. `world_model/train_wm.py` is **purely teacher-forced**: `logits = model(X, graph)` where `X` is built from the _true_ `s_t` at every step. The model is never once trained on an input it produced. Then `rollout_ensemble` feeds it exactly that.

**Concretely, for us:** during training, with probability `1 − ε_i`, replace the `CH_INFECTED` / `CH_FRONTIER` channels of `X` with a sample drawn from the model's own previous-step prediction rather than the recorded `s_t`, annealing `ε_i → 0`. Our episodes are stored as ordered `(s_t, a_t, s_{t+1})` records on a `main` branch, so the previous step of the same episode is already available; the change is in the dataset/collate path, not the model. **This is the single highest-value item in this file.** It attacks the drift where it originates instead of constraining the head so it cannot express drift.

Note the failed-DAgger entry in `research_notes` is _related but different_: DAgger regenerates data by querying the simulator on model-visited states. Scheduled sampling needs no new simulator calls at all — it reuses recorded episodes and only changes which value goes into the input channel. Strictly cheaper, and the thing that was never tried.

### 9.3 Technique 2 — curriculum on horizon length

**MTGNN, KDD 2020** [verified]:

> "The algorithm starts with solving the easiest problem, predicting the next one-step only. It is very advantageous for the model to find a good starting point. With the increase in iteration numbers, we increase the prediction length of the model gradually so that the model can learn the hard task step by step."

MTGNN ablates it (`w/o CL`) and reports it as one of the components that "proves to be effective" [verified]. It composes with scheduled sampling rather than competing: curriculum controls _how far_ you roll, scheduled sampling controls _what you feed_ while rolling.

**For us:** train one-step (current behaviour) for the first `k` epochs, then extend the training rollout to 2, 3, … steps, backpropagating through the multi-step composition. Cost is a `for` loop in the training step and the memory to hold the unrolled graph. Pairs naturally with early stopping on a multi-step metric instead of the current one-step `delta_f1`.

### 9.4 Technique 3 — closed-loop vs open-loop (non-autoregressive) decoding

**Graph WaveNet, IJCAI 2019** sidesteps the problem instead of solving it [verified]:

> "Graph WaveNet generates 12 predictions in one run while DCRNN and STGCN have to produce the results conditioned on previous predictions."

Emitting the whole horizon from one forward pass makes error accumulation structurally impossible — there is no feedback path. It is also why Graph WaveNet is still competitive in 2024 (§5.2) and the fastest at inference.

**For us this is a genuine architectural option and also a genuine trade-off.** A head that emits `s_{t+1..t+H}` in one pass cannot compound its own error, but it also stops being a _world model_ in the sense we need: our coding-agent outer loop must be able to inject an action at an arbitrary intermediate step, which requires a closed loop. So: worth knowing, probably not worth taking. The honest framing is that the traffic field's best-performing answer to rollout drift is "don't roll out", and that answer is unavailable to us precisely because of action-conditioning.

### 9.5 Technique 4 — decouple what diffuses from what does not

**D²STGNN, VLDB 2022** splits the signal into a **diffusion** component (what propagates from neighbours) and an **inherent** component (what the node would do on its own), modelling each in a separate branch. It is the SOTA row in §5.2.

This is architecturally close to what our structured heads already do — the IC head's `p_new(v)` term _is_ a diffusion branch, and `infected` carried forward _is_ an inherent branch. Recording it here mainly as independent convergent evidence that the decomposition is the right inductive bias, from a field that arrived at it by ablation rather than by knowing the generating mechanism.

### 9.6 What the error-accumulation evidence actually shows

From §5.1 [verified, DCRNN Table 1], METR-LA MAE by horizon:

| Model                          | h=3 (15 min) | h=6 (30 min) | h=12 (60 min) | Degradation                   |
| ------------------------------ | ------------ | ------------ | ------------- | ----------------------------- |
| HA (no rollout)                | 4.16         | 4.16         | 4.16          | **0%** (flat by construction) |
| ARIMA-Kalman                   | 3.99         | 5.15         | 6.90          | **+73%**                      |
| SVR                            | 3.99         | 5.05         | 6.72          | **+68%**                      |
| FC-LSTM                        | 3.44         | 3.77         | 4.37          | **+27%**                      |
| **DCRNN** (scheduled sampling) | **2.77**     | **3.15**     | **3.60**      | **+30%**                      |

Two readings. First, **degradation is a separate axis from one-step accuracy** — FC-LSTM and DCRNN have nearly the same degradation slope but DCRNN is uniformly better; ARIMA and SVR start comparable to FNN and end far worse. A model can win at h=1 and lose at h=12. Our one-step `delta_f1` early-stopping criterion is blind to this by construction.

Second, **HA's flat row is the analogue of our `persistence` baseline** — a method that never feeds itself has zero degradation and is therefore the right control to plot degradation against.

### 9.7 Ranked recommendation

| #   | Action                                                                            | Cost                                                | What it buys                                                                                                                           |
| --- | --------------------------------------------------------------------------------- | --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Report `ens_count_bias` (and `ens_marg_mae`) per rollout step**, not aggregated | ~an afternoon; reporting only                       | Makes drift legible. Distinguishes "calibrated" from "calibrated for 2 steps". Prerequisite for 2 and 3.                               |
| 2   | **Scheduled sampling in `train_wm.py`** ⭐                                        | small; dataset/collate path, one annealing schedule | Trains the model on its own output distribution. Attacks the root cause the structured head only constrains. No extra simulator calls. |
| 3   | **Horizon curriculum** (1-step → k-step training rollout)                         | medium; unrolled loop + memory                      | Composes with 2. Lets early stopping key on a multi-step metric instead of one-step `delta_f1`.                                        |
| 4   | **Identity-embedding ablation** (swap the GNN for per-node embedding + MLP)       | small                                               | The STID test (§4.2), applied to our BA-100 backbone tie. Decisive on whether our encoders do structural work.                         |
| 5   | Non-autoregressive multi-step head                                                | large; changes the interface                        | Eliminates drift, **but breaks mid-rollout action injection.** Documented, not recommended.                                            |

Items 1 and 2 are the whole reason this file exists. Item 5 is the reason it is filed under ❌ rather than ⚠️: the traffic field's best answer to our problem is one we cannot take, because taking it would mean giving up the action conditioning that is our contribution.

---

## 10. Reference list

All links verified HTTP 200 on **2026-07-28** unless annotated.

**Traffic forecasting — the canonical line** [STGCN (IJCAI'18, arXiv 1709.04875)](https://arxiv.org/abs/1709.04875) · [code](https://github.com/VeritasYin/STGCN_IJCAI-18) · [DCRNN (ICLR'18, arXiv 1707.01926)](https://arxiv.org/abs/1707.01926) · [code](https://github.com/liyaguang/DCRNN) · [ASTGCN (AAAI'19)](https://ojs.aaai.org/index.php/AAAI/article/view/3881) · [code](https://github.com/Davidham3/ASTGCN) · [Graph WaveNet (IJCAI'19, arXiv 1906.00121)](https://arxiv.org/abs/1906.00121) · [code](https://github.com/nnzhan/Graph-WaveNet) · [GMAN (AAAI'20, arXiv 1911.08415)](https://arxiv.org/abs/1911.08415) · [code](https://github.com/zhengchuanpan/GMAN) · [STSGCN (AAAI'20)](https://ojs.aaai.org/index.php/AAAI/article/view/5438) · [code](https://github.com/Davidham3/STSGCN) · [AGCRN (NeurIPS'20, arXiv 2007.02842)](https://arxiv.org/abs/2007.02842) · [code](https://github.com/LeiBAI/AGCRN) · [MTGNN (KDD'20, arXiv 2005.11650)](https://arxiv.org/abs/2005.11650) · [code](https://github.com/nnzhan/MTGNN) · [STGODE (KDD'21, arXiv 2106.12931)](https://arxiv.org/abs/2106.12931) · [code](https://github.com/square-coder/STGODE) · [D²STGNN (VLDB'22, arXiv 2206.09112)](https://arxiv.org/abs/2206.09112) · [code](https://github.com/zezhishao/D2STGNN) · [STID (CIKM'22, arXiv 2208.05233)](https://arxiv.org/abs/2208.05233) · [code](https://github.com/zezhishao/STID) · [STJGCN (TKDE'23, arXiv 2111.13684)](https://arxiv.org/abs/2111.13684) · [code](https://github.com/zhengchuanpan/STJGCN) · [PDFormer (AAAI'23, arXiv 2301.07945)](https://arxiv.org/abs/2301.07945) · [code](https://github.com/BUAABIGSCity/PDFormer) · [STAEformer (CIKM'23, arXiv 2308.10425)](https://arxiv.org/abs/2308.10425) · [code](https://github.com/XDZhelheim/STAEformer)

**Foundation / LLM-based** [TimeGPT-1 (arXiv 2310.03589)](https://arxiv.org/abs/2310.03589) · [client](https://github.com/Nixtla/nixtla) · [UniST (KDD'24, arXiv 2402.11838)](https://arxiv.org/abs/2402.11838) · [code](https://github.com/tsinghua-fib-lab/UniST) · [UrbanGPT (KDD'24, arXiv 2403.00813)](https://arxiv.org/abs/2403.00813) · [code](https://github.com/HKUDS/UrbanGPT)

**Benchmarks and libraries** [BasicTS+ (TKDE'24, arXiv 2310.06119)](https://arxiv.org/abs/2310.06119) · [code](https://github.com/GestaltCogTeam/BasicTS) · [LibCity (arXiv 2304.14343)](https://arxiv.org/abs/2304.14343) · [code](https://github.com/LibCity/Bigscity-LibCity) · [DL-Traff (CIKM'21, arXiv 2108.09091)](https://arxiv.org/abs/2108.09091) · [graph](https://github.com/deepkashiwa20/DL-Traff-Graph) · [grid](https://github.com/deepkashiwa20/DL-Traff-Grid) · [PyTorch Geometric Temporal (CIKM'21, arXiv 2104.07788)](https://arxiv.org/abs/2104.07788) · [code](https://github.com/benedekrozemberczki/pytorch_geometric_temporal) · [LargeST (NeurIPS'23 D&B, arXiv 2306.08259)](https://arxiv.org/abs/2306.08259) · [code](https://github.com/liuxu77/LargeST)

**Temporal graphs (discrete-time) and surveys** [EvolveGCN (AAAI'20, arXiv 1902.10191)](https://arxiv.org/abs/1902.10191) · [code](https://github.com/IBM/EvolveGCN) · [ROLAND (KDD'22, arXiv 2208.07239)](https://arxiv.org/abs/2208.07239) · [code](https://github.com/snap-stanford/roland) · [Kazemi et al., "Representation Learning for Dynamic Graphs" (JMLR'20, arXiv 1905.11485)](https://arxiv.org/abs/1905.11485) · [Longa et al., "GNNs for Temporal Graphs" (TMLR'23, arXiv 2302.01018)](https://arxiv.org/abs/2302.01018)

**Baselines and the "are deep models needed" thread** [LTSF-Linear / DLinear (AAAI'23, arXiv 2205.13504)](https://arxiv.org/abs/2205.13504) · [code](https://github.com/cure-lab/LTSF-Linear) · [Scheduled sampling (Bengio et al., NeurIPS'15, arXiv 1506.03099)](https://arxiv.org/abs/1506.03099) · [Seq2seq (Sutskever et al., arXiv 1409.3215)](https://arxiv.org/abs/1409.3215) · [DeepAR (arXiv 1704.04110)](https://arxiv.org/abs/1704.04110) · [N-BEATS (arXiv 1905.10437)](https://arxiv.org/abs/1905.10437)

**Data sources** [Caltrans PeMS](https://pems.dot.ca.gov/) · [NYC TLC trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) · [Citi Bike system data](https://citibikenyc.com/system-data) · [Q-Traffic / BaiduTraffic](https://github.com/JingqingZ/BaiduTraffic) · [TaxiBJ via DeepST](https://github.com/TolicWang/DeepST) · [Highways England WebTRIS](https://webtris.highwaysengland.co.uk/)

Two links in §3 do not resolve for a non-browser client and are annotated there: the Wiley Box & Jenkins DOI returns **403** (publisher paywall). Every other URL in this file returned **200**.

**Cross-reference:** the continuous-time branch (TGN, JODIE, TGAT, DyRep) is covered in `research_notes/Literature Review.md`, not here.

---

## 11. Open gaps

Honest list of what this review could **not** establish.

- **One cell of BasicTS+ Table V's `_Speed (LA)_` row is missing** and could not be recovered from the extracted text: fifteen values for sixteen model columns. It is marked `?` in §5.3, and the consequence is that the trailing values' column assignment (STNorm vs STID) is unverified. Every other cell in that table is [verified] and full. Resolving it needs a re-read of the paper's own table.
- **Edge counts for METR-LA / PEMS-BAY / PEMS08 disagree across papers** and the discrepancy is not resolved here. METR-LA is 1,515 (LargeST Table 1) vs 1,722 (D²STGNN Table 2); PEMS-BAY 2,369 vs 2,694; PEMS08 276 vs 295 vs 548. All are [verified] transcriptions of their own tables — so the graphs differ by preprocessing, not by transcription error. Likely causes are the self-loop/threshold choice in DCRNN's Gaussian kernel (`κ`) and arcs-vs-edges, the same convention trap `research/README.md` documents. **Not resolved:** no file was downloaded and counted. Since no dataset here will be loaded (§2.4), this was not worth the download.
- **TaxiBJ, NYC-Bike/Taxi, Q-Traffic, and the loop-detector archives are [claim]-tier only.** No paper table was extracted for them, and their crops genuinely are not standardized — every paper picks its own window and grid resolution. A number quoted for "NYC-Taxi" is close to meaningless without the paper that produced it.
- **Q-Traffic's query sub-dataset was not examined.** It is the one artifact in this literature that resembles an exogenous input (user route requests preceding traffic change) and is therefore the only plausible route to an action-conditioned variant of this task. Whether it can be made to look like any of our five ops is unestablished.
- **The BasicTS+ reference numbers `[22]`, `[24]`, `[25]`, `[46]`, `[57]`, `[62]`, `[63]`, `[64]` in §5.2 Table I were not dereferenced.** Their bibliography was not extracted, so §5.2 reports "as reported in five papers" without naming which five. The _spread_ is [verified]; the attribution is not.
- **STGODE, STJGCN, PDFormer, STAEformer, GMAN, UniST and UrbanGPT result cells were not transcribed.** They are catalogued with paper and code URLs in §4 but their tables are not in §5 — deliberately, per the ❌ verdict. If a number from them is ever needed, extract it from the PDF; do not take it from a summary.
- **DL-Traff's own re-benchmark findings were not transcribed.** It is the other paper in this space that re-ran the field, and it may or may not agree with BasicTS+. Only BasicTS+ was extracted.
- **No claim is made about whether scheduled sampling actually fixes our drift.** §9.2 argues it targets the right failure mode and is cheap; DCRNN demonstrates it works for continuous autoregressive decoding. Whether it helps a binary/probabilistic structured head is untested and is a genuine empirical question — our own `research_notes` records four plausible fixes that failed before the structural one worked.
- **The LargeST-scale regime is unexplored on our side.** LargeST-CA is 8,600 nodes × 525,888 frames. Nothing in this review establishes whether the scalability lessons there (subsampling, sub-region splits) transfer to our NDlib rollout bottleneck, which is a simulator cost, not a model cost.
