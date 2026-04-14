# Literature Review

## Graph World Model - https://arxiv.org/pdf/2507.10539

**Code:** https://github.com/ulab-uiuc/GWM

### Summary

This paper introduces **GWM (Graph World Model)**, a framework that extends the classical world model paradigm to handle _graph-structured_ states with multi-modal node information (images, tables, text). Standard world models operate over unstructured or sequential data; GWM argues that the digital world is fundamentally graph-structured — entities relate to one another — and that a world model should natively reason over this relational structure.

The core idea is to treat state as a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ where each node $v \in \mathcal{V}$ carries multi-modal content $v = [v^a, v^b, v^e]$ (image, table, text), and edges encode either _explicit_ domain structure $\mathcal{E}_p$ or _implicit_ semantic similarity $\mathcal{E}_m$. State transitions follow the standard world model equation:

$$s_{t+1} = f_{\text{tr}}(s_t, a_t)$$

**Actions** are themselves modeled as nodes. A result node $v_r$ is obtained by querying the state graph with action $a$ via a retrieval function $R$:

$$v_r = R(v, a)$$

Actions come in two flavors:

- **Intended actions** $a_d$: directly linked to graph elements (node-level, edge-level, or graph-level targets)
- **Unintended actions** $a_u$: indirectly linked via semantic similarity (e.g., RAG-style retrieval)

Two instantiations are proposed to unify modalities before feeding into downstream generative models:

### GWM-T: Token-Based Architecture

Images are converted to text via LLaVA:

$$v^{ta} = \mathcal{L}(v^a)$$

Tables are linearized using column-value templates:

$$v^{tb} = \mathcal{T}(v^b) \quad \text{(format: "\{column\} is \{value\}")}$$

All modalities are concatenated into a unified token representation:

$$v_c = P_u(v^{ta}, v^{tb}, v^e)$$

Message passing is implemented at the **token level** via LLM prompting. At layer $l$:

$$h_v^{(l)} = f_v\!\left(\operatorname{Concat}\!\left(h_v^{(l-1)},\ \{h_u^{(l-1)} : u \in \mathcal{N}(v)\}\right)\right)$$

with $h_v^{(0)} = v_c$. For **generation** tasks (image synthesis), the denoising loss over a Stable Diffusion backbone is:

$$\mathcal{L} = \mathbb{E}_{z \sim \text{Enc}(x),\, c_T,\, \epsilon \sim \mathcal{N}(0,1),\, t}\left[\left\|\epsilon - \epsilon_\theta(z_t, t, h(c_T))\right\|^2\right]$$

where $h(c_T) = \text{CLIP}(c_T)$ encodes the text prompt. For **prediction** tasks (text output), the supervised fine-tuning loss is:

$$\mathcal{L}_{\text{SFT}} = -\sum_t \log P_{f_{\text{SFT}}}\!\left(y_t \mid P_{sa}(h_{v_r}, a),\, y_1, \ldots, y_{t-1}\right)$$

---

### GWM-E: Embedding-Based Architecture

Instead of text conversion, GWM-E encodes each modality with dedicated encoders:

- Text: $e_t = E_b(v^e)$ (BERT)
- Table: $e_b = E_b(\mathcal{T}(v^b))$ (BERT on linearized table)
- Image: $e_a = E_c(v^a)$ (CLIP)

Unified node embedding with zero-padding for missing modalities:

$$e_v = \operatorname{Concat}(e_a, e_t, e_b)$$

Multi-hop aggregation uses the normalized adjacency $\tilde{A} = D^{-1/2} A D^{-1/2}$:

$$X_e^{(l)} = \tilde{A}^l \cdot X_e$$

The full multi-hop feature set is retained and projected:

$$X_G = \left[X_c,\ X_c^{(1)},\ \ldots,\ X_c^{(L)}\right], \quad X_c^{(l)} = f_c(X_e^{(l)})$$

For generation (Stable Diffusion with graph conditioning):

$$\mathcal{L} = \mathbb{E}_{z \sim \text{Enc}(x),\, c_T,\, c_G,\, \epsilon,\, t}\left[\left\|\epsilon - \epsilon_\theta\!\left(z_t, t, h(c_T, c_G)\right)\right\|^2\right]$$

where $h(c_T, c_G) = [h_T(c_T),\ h_G(c_G)]$ concatenates text and graph conditions. For prediction (frozen LLM, projector-only fine-tuning):

$$\mathcal{L}_{\text{SFT}} = -\sum_t \log P_{f_{\text{SFT}}}\!\left(y_t \mid X_G, a,\, y_1, \ldots, y_{t-1}\right)$$

### Experimental Results

GWM is evaluated across six domain categories spanning **world prediction** and **world generation/optimization**:

| Domain                 | Task                  | GWM Best       | Best Baseline |
| ---------------------- | --------------------- | -------------- | ------------- |
| Multi-Modal Generation | CLIP Score (MM-Paper) | 59.92 (GWM-T)  | 56.37         |
| Multi-Modal Matching   | Accuracy (MM-Paper)   | 96.23% (GWM-E) | 51.77%        |
| Recommendation         | F1 (Baby)             | 84.74 (GWM-E)  | 82.47         |
| Graph Prediction       | Accuracy (Cora)       | 83.03% (GWM-E) | 82.76%        |
| Multi-Agent (RAG)      | LongBench v2          | 33.32% (GWM-E) | 29.01%        |
| Planning (ALFWorld)    | F1                    | 92.13 (GWM-E)  | 91.82         |

**Ablation:** Adding graph structure yields at minimum a 20% relative gain on graph-specific tasks. Excessive hops cause over-smoothing and degrade performance.

**Zero-shot transfer:** GWM trained on five of six tasks outperforms single-task baselines on the held-out sixth task in zero-shot, with 10% few-shot fine-tuning closing most remaining gaps.

**Backbone:** Llama-3-8B (LLM), SD-v1-5 (diffusion), LLaVA-1.5-7B (image-to-text), CLIP + BERT (encoders).

### Intuition

The key intuition is that a world model's state should be a _graph_, not a vector or sequence. An entity's true state depends on its relational context — what it connects to, how information flows through the graph. GWM realizes this by running message passing before any generative computation, so the LLM or diffusion model receives node representations already enriched with neighborhood context. The "action node" abstraction is elegant: by treating actions as graph queries rather than control signals, GWM handles both structured retrieval and free-form generation within the same framework.

The distinction between GWM-T and GWM-E reflects a core tension in multi-modal learning: **late fusion in token space** (GWM-T) is more flexible and leverages LLM reasoning but loses information in the image→text conversion; **early fusion in embedding space** (GWM-E) preserves raw signal fidelity and scales to larger graphs but requires a projector to bridge modality gaps.

### Relation to Our Project

#### What's Shared

| Concept                   | GWM                                                       | This Project                                                       |
| ------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------ |
| Graph-structured state    | $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ as world state | Graph $G$ as the environment; cascades/connectivity as state       |
| World model framing       | $s_{t+1} = f_{\text{tr}}(s_t, a_t)$                       | Forward model $\hat{y} = f_\theta(x, \text{adj})$ as world model   |
| Normalized adjacency      | $\tilde{A} = D^{-1/2} A D^{-1/2}$                         | Identical normalization for attention masking in GT and GCN layers |
| Multi-hop aggregation     | $X_e^{(l)} = \tilde{A}^l X_e$                             | GCN/GCNII layers perform equivalent $l$-hop propagation            |
| Optimization over actions | World optimization via planning                           | Phase 2 logit optimization: $\min_z \mathcal{L}(\sigma(z))$        |
| Action representation     | Action nodes linked to graph                              | Binary action vector $x \in \{0,1\}^N$ (seed/removal/source)       |

Both projects share the same fundamental framing: a learned graph model serves as a forward simulator, and the goal is to invert or optimize through it to recover an action (node set) that produces a target outcome.

#### What's Different

**1. Problem class.** GWM targets multi-modal prediction and generation — its "actions" are queries (what book cover to generate, what to recommend, what text to retrieve). This project targets _combinatorial inverse problems_ on graphs — the action is a binary node selection from $\{0,1\}^N$ and the outcome is a dynamical process (cascade, connectivity, infection spread). GWM's state transitions are driven by LLM/diffusion generation; this project's transitions are graph diffusion processes (IC model, connectivity).

**2. Action space.** GWM's action nodes are continuous semantic queries resolved via retrieval or attention. This project's actions are discrete node sets optimized via continuous relaxation ($\sigma(\text{logits})$) and rounded at extraction. The combinatorial constraint ($\|x\|_0 = k$) is central here and absent in GWM.

**3. Graph dynamics.** GWM treats the graph as a _static_ relational context for retrieval and generation. This project's graphs are _dynamic_ — the graph structure drives a stochastic process (IC cascade, connectivity fragments), and the forward model must learn to predict those dynamics. This is a harder problem with less training signal per graph.

**4. Scale and modality.** GWM operates on nodes with rich multi-modal content (images, text, tables) using large pretrained encoders and LLMs. This project operates on purely structural/topological graphs with scalar node features (degree PE, bag-of-words), using lightweight GNNs trained from scratch. The setting is intentionally lean to test whether graph structure alone is sufficient for inverse inference.

**5. Evaluation.** GWM evaluates on standard benchmarks across diverse task types. This project evaluates on task-specific metrics: influence spread (IM), network fragmentation (CND), and source recovery accuracy (SL), with the primary signal being whether the recovered node set achieves a better outcome than classical baselines.

#### What's Applicable

**1. Action node framing for the coding agent.** The most directly applicable idea is GWM's "action node" abstraction. The planned coding agent in this project generates candidate graph algorithms as compositions of action primitives. Representing each candidate algorithm as a _node in a graph_ — where edges encode similarity, dependency, or historical co-performance — would allow message passing to propagate feedback signals across the algorithm search space. This mirrors GWM's intended action framework.

**2. Implicit edges from embedding similarity.** GWM computes $\mathcal{E}_m$ by embedding similarity between nodes. For the coding agent loop, this could be used to build a _performance graph_ over candidate algorithms: nodes are algorithm instances, edges connect candidates with similar structure or outcome, and GNN message passing propagates reward signals across structurally similar candidates without requiring independent evaluation of each.

**3. Multi-hop feature retention for the forward model.** GWM-E retains all $L$ hops of aggregated features and feeds them jointly to the decoder: $X_G = [X_c, X_c^{(1)}, \ldots, X_c^{(L)}]$. This is essentially a learnable version of what GCNII does with its initial residual connection. If GCN/GT forward models are being deepened (to capture longer cascade paths), this multi-hop concatenation strategy is a practical alternative to GCNII's algebraic formulation — it avoids the identity mapping hyperparameters ($\alpha$, $\lambda$) while still preventing over-smoothing.

**4. Zero-shot transfer across tasks.** GWM shows strong cross-task generalization. For this project's multi-task setting (IM + CND + SL), training a _single_ forward model across tasks — with task identity as an additional node feature or action node — could improve data efficiency on smaller graphs (Jazz, NetScience). GWM's result that 10% few-shot fine-tuning closes most zero-shot gaps suggests the shared representation is meaningful and that task-specific heads need only light fine-tuning.

---

## Temporal Graph Networks (TGN) - https://arxiv.org/pdf/2006.10637

**Code:** https://github.com/twitter-research/tgn

### Summary

TGN is a generic, efficient framework for deep learning on **continuous-time dynamic graphs** represented as a stream of timestamped events. Where standard GNNs operate on static graphs (a single snapshot), TGN treats the graph as an evolving object: edges and node features arrive over time, and the model must produce up-to-date node embeddings at any timestamp $t$. The central innovation is a **per-node memory module** $\mathbf{s}_i(t)$ that compresses each node's history into a state vector, combined with standard graph attention over the temporal neighborhood to produce embeddings.

The framework unifies several earlier dynamic-graph methods (Jodie, TGAT, DyRep) as special cases and demonstrates that combining memory with a single attention layer is both more accurate and ~30× faster than multi-layer attention without memory.

### Architecture

A continuous-time dynamic graph is represented as a sequence of events:

$$\mathcal{G} = \{x(t_1), x(t_2), \ldots\}$$

where each event is either a node-wise event (feature update) or an interaction event $(i, j, t, \mathbf{e}_{ij}(t))$ between nodes $i$ and $j$ at time $t$.

TGN has five modular components:

#### 1. Memory Module

Each node $i$ maintains a state vector $\mathbf{s}_i(t) \in \mathbb{R}^d$, initialized to zero and updated whenever the node is involved in an event. Memory persists across batches and even after training, acting as a long-term per-node summary.

#### 2. Message Function

For an interaction event between $i$ and $j$ at time $t$, two messages are computed:

$$\mathbf{m}_i(t) = \text{msg}_s\!\left(\mathbf{s}_i(t^-), \mathbf{s}_j(t^-), \Delta t, \mathbf{e}_{ij}(t)\right)$$

$$\mathbf{m}_j(t) = \text{msg}_d\!\left(\mathbf{s}_j(t^-), \mathbf{s}_i(t^-), \Delta t, \mathbf{e}_{ij}(t)\right)$$

For node-wise events:

$$\mathbf{m}_i(t) = \text{msg}_n\!\left(\mathbf{s}_i(t^-), t, \mathbf{v}_i(t)\right)$$

where $t^-$ denotes the time just before $t$ and $\Delta t = t - t_i^-$ is the elapsed time since the last update of node $i$.

#### 3. Message Aggregator

Within a training batch, multiple events may target the same node. An aggregator collapses them:

$$\bar{\mathbf{m}}_i(t) = \text{agg}\!\left(\mathbf{m}_i(t_1), \ldots, \mathbf{m}_i(t_b)\right)$$

In practice the aggregator is either _most recent_ or _mean_ — both are non-learnable.

#### 4. Memory Updater

A recurrent function (LSTM or GRU) updates the memory state:

$$\mathbf{s}_i(t) = \text{mem}\!\left(\bar{\mathbf{m}}_i(t), \mathbf{s}_i(t^-)\right)$$

#### 5. Embedding Module

Embeddings $\mathbf{z}_i(t)$ are produced _on demand_ from the current memory plus a neighborhood lookup. This step is essential because it (a) prevents memory staleness and (b) lets information from the static graph features flow back in.

The most effective variant is **Temporal Graph Attention** ($L$ layers):

$$\mathbf{h}_i^{(l)}(t) = \text{MLP}^{(l)}\!\left(\mathbf{h}_i^{(l-1)}(t)\,\Vert\,\tilde{\mathbf{h}}_i^{(l)}(t)\right)$$

$$\tilde{\mathbf{h}}_i^{(l)}(t) = \text{MultiHeadAttention}^{(l)}\!\left(\mathbf{q}^{(l)}(t), \mathbf{K}^{(l)}(t), \mathbf{V}^{(l)}(t)\right)$$

with query, keys, and values built from the temporal neighborhood $\mathcal{N}_i([0, t])$:

$$\mathbf{q}^{(l)}(t) = \mathbf{h}_i^{(l-1)}(t)\,\Vert\,\phi(0)$$

$$\mathbf{K}^{(l)}(t) = \mathbf{V}^{(l)}(t) = \left[\mathbf{h}_{j_1}^{(l-1)}(t)\,\Vert\,\mathbf{e}_{ij_1}\,\Vert\,\phi(t - t_1),\ \ldots\right]$$

The function $\phi(\cdot): \mathbb{R} \to \mathbb{R}^{d_T}$ is a learnable **time encoding** (Time2Vec / cosine basis) that lets attention condition on _how long ago_ each interaction happened. Initial features are $\mathbf{h}_i^{(0)}(t) = \mathbf{s}_i(t) + \mathbf{v}_i(t)$.

A simpler variant, **Temporal Graph Sum**, replaces attention with a sum:

$$\tilde{\mathbf{h}}_i^{(l)}(t) = \text{ReLU}\!\left(\sum_{j \in \mathcal{N}_i([0,t])} \mathbf{W}_1^{(l)}\!\left(\mathbf{h}_j^{(l-1)}(t)\,\Vert\,\mathbf{e}_{ij}\,\Vert\,\phi(t - t_j)\right)\right)$$

### Training

Training uses self-supervised **future link prediction**: given events up to time $t$, predict whether a candidate edge $(i, j)$ will appear next. Loss is binary cross-entropy with negative sampling.

The subtle issue is that memory must be updated _before_ embeddings are computed for a batch, but the messages used for that update must come from earlier batches — otherwise gradients leak future information. TGN solves this with a **Raw Message Store**: raw inputs to the message function are cached from prior batches, and the memory update for batch $b$ uses only messages from batches $< b$. The order of operations per batch is:

1. Update memory with raw messages stored from prior batches
2. Compute embeddings $\mathbf{z}_i(t)$ from updated memory + temporal neighborhood
3. Predict links and compute loss
4. Compute new raw messages from current batch and store them

### Experimental Results

**Future edge prediction (Average Precision %):**

| Dataset   | Setting      | TGN-attn  | TGAT  | DyRep | Jodie |
| --------- | ------------ | --------- | ----- | ----- | ----- |
| Wikipedia | Transductive | **98.46** | 95.34 | 94.59 | 94.62 |
| Wikipedia | Inductive    | **97.81** | 93.99 | 92.05 | 93.11 |
| Reddit    | Transductive | **98.70** | 98.12 | 97.98 | 97.11 |
| Twitter   | Transductive | **94.52** | 70.02 | 83.52 | 85.20 |

**Dynamic node classification (ROC AUC):** TGN-attn reaches 87.81% on Wikipedia, 67.06% on Reddit — beating TGAT and Jodie baselines.

**Speed:** A single attention layer with memory matches or exceeds two attention layers without memory while being ~30× faster per epoch. The memory module effectively "pre-aggregates" temporal context, eliminating the need for deep $L$-hop recursion.

### Intuition

The key insight is the decomposition of _what happens to a node over time_ into two qualitatively different operations:

1. **Memory** ($\mathbf{s}_i(t)$): a slow-moving, per-node summary that captures the node's full history compressed into a fixed-size vector. Updated only when the node is touched by an event.
2. **Embedding** ($\mathbf{z}_i(t)$): a fast, on-demand computation that combines the (possibly stale) memory with a fresh look at the current temporal neighborhood. This is what is consumed by downstream prediction heads.

Without memory, you need many GNN layers to propagate temporal context — slow and prone to oversmoothing. Without embedding, memory becomes stale: a node not touched recently has an outdated representation. The two modules together are mutually reinforcing — memory provides historical compression, embedding provides freshness and structural context.

The **time encoding** $\phi(\cdot)$ is the other key trick: it lets the attention mechanism reason about _recency_ without requiring an explicit recurrence over time, which is what makes TGN work over irregular event sequences rather than discrete snapshots.

---

### Relation to Our Project

#### What's Shared

| Concept                  | TGN                                            | This Project                                                  |
| ------------------------ | ---------------------------------------------- | ------------------------------------------------------------- |
| Graph state              | Continuous-time graph $\mathcal{G}=\{x(t_k)\}$ | Static graph $G$ + dynamical process (cascade, fragmentation) |
| Forward model            | $\mathbf{z}_i(t)$ predicts future links        | $\hat{y} = f_\theta(x, \text{adj})$ predicts cascade outcome  |
| Attention over neighbors | Multi-head attention with time encoding        | Multi-head attention in `model/graph_transformer.py`          |
| Self-supervised data     | Negative-sampled link prediction               | Synthetic $(x, y)$ pairs from running IC / connectivity       |
| Message passing          | Memory + attention over temporal neighborhood  | Scatter-softmax attention / GCN propagation                   |

Both approaches train a graph model to predict outcomes generated by an underlying graph process, and both lean heavily on multi-head attention as the workhorse.

#### What's Different

**1. Time is implicit vs. explicit.** This is the largest difference. The current project's forward models are _static_: the input is a binary action vector $x \in \{0,1\}^N$ and the output is a _collapsed_ outcome ($y$ = union of activations across all timesteps for IM, residual largest CC for CND, snapshot at one fixed time $t$ for SL). The temporal dynamics of the IC cascade — _which nodes activate in which round, in what order_ — are thrown away during data generation. TGN, in contrast, treats time as a first-class input via timestamps, $\Delta t$, and the time encoding $\phi(\cdot)$.

**2. Static vs. dynamic graph topology.** TGN handles graphs whose _edges_ arrive over time (Wikipedia user-page edits, Reddit post-subreddit interactions). This project's graphs are fixed in structure — the topology never changes. What changes is the _node activation state_ under a dynamical process. So TGN's memory module would track _infection states_ per node, not per-node embeddings of an evolving graph.

**3. Inverse vs. forward problem.** TGN is purely a forward model: given history, predict what happens next. The current project's signature contribution is the **inverse loop** — Phase 2 logit optimization through a frozen forward model to recover an unknown action. TGN does not address inverse problems, optimization-through-the-model, or combinatorial action selection.

**4. Training signal.** TGN trains on raw event streams from real-world dynamic graphs via self-supervised link prediction. This project trains on _simulated_ (action, outcome) pairs from running classical algorithms (IC sampling, BFS connectivity, etc.). The setting is supervised regression, not contrastive.

**5. Embedding granularity.** TGN computes embeddings _on demand_, at arbitrary query times $t$. This project's forward models produce a single $(N, 1)$ output for the full graph in one shot. There is no notion of "embedding at time $t$" because time isn't represented at all.

#### What's Applicable

**1. Don't collapse the cascade — predict the trajectory.** This is the single most actionable idea. The IC model produces an ordered sequence of activation rounds: at $t=0$, the seed set is active; at $t=1$, some subset of their neighbors becomes active; and so on. The current pipeline collapses this to $y = \mathbb{1}\{\text{node ever activates}\}$. A TGN-style forward model would instead predict $y(t) \in [0,1]^N$ for each round, training on the full trajectory. This gives the forward model dramatically more supervision per cascade sample, and — more importantly — the gradients in Phase 2 would carry temporal information that makes the loss landscape much more informative. For graphs like NetHEPT where cascades unfold over many rounds, the temporal signal is real and currently discarded.

**2. Memory module for source localization.** SL is the most temporally natural task in the project: the input is "a partial snapshot at random time $t$" and the output is the source set. A TGN-style architecture is almost custom-built for this:

- Treat the observed snapshot as initial node states $\mathbf{v}_i(t)$
- Use the time encoding $\phi(\cdot)$ to inject the snapshot timestamp
- Run the embedding module with attention over the _static_ graph topology but using the time-aware features
- The query becomes "given the infection pattern at time $t$, which nodes were infected at $t=0$?" — this is essentially TGN's link prediction reframed as backward inference.

Concretely, the SL forward model could replace its current Channel-1 input with $[\text{snapshot}, \phi(t)]$ where $\phi(t)$ is the learned time encoding, giving the model an explicit handle on _how much diffusion has occurred_.

**3. Time2Vec / cosine time encoding for cascade rounds.** Even without a full TGN rewrite, the time encoding $\phi(\cdot)$ is a drop-in addition. For IM training with `--augment`, currently the model never sees _which round_ an activation happened in. Augmenting the forward pass to take a round index $r$ and use $\phi(r)$ as an additional positional encoding would let one forward model predict the full trajectory rather than the union. Implementation cost is low: add a learnable Time2Vec layer in `model/graph_transformer.py` and concatenate $\phi(r)$ to the degree positional encoding.

**4. Memory + shallow attention beats deep stateless attention.** TGN's empirical finding — that one attention layer with memory beats $L$ stateless layers and is 30× faster — is directly relevant to the depth vs. epochs tradeoff already observed in this project (cf. memory observation S1 from Apr 12 about `--sage-layers` depth). If cascade rounds are explicitly encoded via memory, the forward model needs fewer GNN layers to capture multi-hop diffusion, since the memory state effectively "remembers" earlier propagation. This may be a cheaper path to handling NetHEPT-scale graphs than naively deepening GCN/GraphSAGE.

**5. Raw message store for preventing target leakage.** If the project moves toward trajectory prediction, TGN's discipline around _order of operations_ (update memory only with messages from prior steps, not the current batch) is the right pattern to avoid the model trivially "predicting the future from the future." This is a subtle source of bugs in any temporal forward model and the TGN paper presents the cleanest published solution.

---

### Bottom Line

TGN solves a different problem (forward prediction on dynamic graphs) but shares this project's deepest design constraint: how do you build a graph model that respects an underlying temporal process? The current project sidesteps this by collapsing the cascade to a static union — convenient, but it throws away most of the supervisory signal the IC model produces.

The most concrete actionable change is **trajectory-aware forward models**: train $f_\theta$ to predict $y(t)$ for each cascade round, not just the final union. The cheapest path to this is adding a Time2Vec encoding and a round index to the existing GT/GCN architectures, keeping the rest of the pipeline intact. The longer-term refactor — adopting a full TGN-style memory module — is more invasive but would be the right architecture for source localization in particular, where the temporal structure of the snapshot is the entire input signal.

TGN is also a useful counterpoint to GWM: where GWM treats graphs as static relational context for multi-modal generation, TGN treats them as evolving streams of events. This project lives somewhere in between — static topology, dynamic node states — and could productively borrow primitives from both.

---

## CWM: Code World Model - https://arxiv.org/pdf/2510.02387

**Code / Weights:** https://github.com/facebookresearch/cwm (open-weights, 32B parameters)

### Summary

CWM (Code World Model) is a 32B-parameter open-weights LLM from FAIR CodeGen trained specifically to _model the dynamics of code execution_, not merely its static form. The thesis is that most code LLMs learn "what code looks like" from raw source text, but fail to learn "what code _does_ when executed" — how each line mutates program state. CWM attacks this gap by mid-training on observation–action trajectories harvested from two environments: a traced Python interpreter (120M+ traced functions, stack-frame-level variable state at every line), and a containerized agentic environment (3M ForagerAgent rollouts in Docker, performing bash/edit/create actions to fix bugs). The resulting model can act as a "neural Python debugger" (predict variable states line by line without executing the code) and as a competent agentic coder (65.8% on SWE-bench Verified, 96.6% on Math-500).

The broader claim is that **world modeling — learning $s_{t+1} = f(s_t, a_t)$ over execution states — is the right inductive bias for code reasoning**, and that scale-pretraining alone is not enough. This directly mirrors the framing in the project's own planned coding agent: use a forward model as a _cheap, differentiable simulator_ that replaces expensive real execution in the agent loop.

### Core Framing: Code as a World Model

The model is trained on sequences of (observation, action, next observation), where:

- **Observation** $o_t$ = local variable dictionary (JSON-serialized) + stack frame metadata _before_ executing a line
- **Action** $a_t$ = the Python statement to be executed
- **Next observation** $o_{t+1}$ = updated local variables after execution (values unchanged from $o_t$ are compressed to `"..."`)

This yields the standard world-model recurrence:

$$o_{t+1} = f_\theta(o_t, a_t)$$

Trajectories are encoded with custom control tokens, e.g.:

```
|trace_context_start|
|frame_sep| |call_sep|   {vars_json} |action_sep| code_line
|frame_sep| |line_sep|   {vars_json} |action_sep| code_line
|frame_sep| |return_sep|              |action_sep| code_line |arg_sep| retval
|frame_sep|
```

Two trajectory sources:

1. **Python execution traces.** Functions from 21k+ traced repo commits are fuzzed (with LLM-generated inputs) and run through an instrumented CPython; the interpreter emits a line-by-line memory state capture.
2. **ForagerAgent trajectories.** A scripted agent interacts with a Docker sandbox over a minimal tool set (`bash`, `edit`, `create`, `submit`) on two task families: _mutate-fix_ (synthetic bug injection) and _issue-fix_ (real GitHub issues).

### Training Pipeline

CWM is trained in four stages. Only mid-training is conceptually novel — the rest is standard.

**1. Pre-training (8T tokens).** Standard causal-LM next-token prediction on a general code+text corpus.

**2. Mid-training (5T tokens).** Same objective, but the data mixture is shifted toward execution-trace and agentic-trajectory data. This is where the "world model" signal enters. The loss remains next-token cross-entropy:

$$\mathcal{L}_{\text{mid}} = -\sum_{t} \log p_\theta(x_t \mid x_{<t})$$

but the sequences $x_{1:T}$ are observation–action–observation tuples rather than raw source files. The model has to predict the _next variable state_ given the prior state and the executed line — this is a sequence-modeling reframing of $o_{t+1} = f_\theta(o_t, a_t)$.

**3. SFT (100B tokens, 50k steps).** Instruction tuning with a reasoning-mode toggle. Loss is masked on reasoning-block tokens:

$$\mathcal{L}_{\text{SFT}} = -\sum_{t \notin \mathcal{M}_{\text{think}}} \log p_\theta(x_t \mid x_{<t})$$

where $\mathcal{M}_{\text{think}}$ masks spans between `|reasoning_thinking_start|` and its closing token. Constant LR $1 \times 10^{-5}$ after 1k warmup steps.

**4. RL (GRPO variant).** Task reward on coding/math benchmarks. The advantage estimator drops the $\sigma$ normalization used in vanilla GRPO:

$$\hat{A}_i = R_i - \mu$$

where $R_i$ is the cumulative return of rollout $i$ and $\mu$ is the group mean. PPO-style clipping uses asymmetric bounds $\varepsilon_{\text{low}} = 0.2$, $\varepsilon_{\text{high}} = 0.25$, no KL regularization, and length normalization by the context window $N = 131\,072$ rather than by rollout length.

The SWE-RL reward is hybrid and terminal:

$$r = \begin{cases} +1 & \text{all hidden tests pass} \\ 0 & \text{tests fail, patch similarity} \geq 0.5 \\ -1 & \text{otherwise} \end{cases}$$

A self-bootstrapping loop (RL $\to$ rejection sampling high-return traces $\to$ re-add to SFT) drives SWE-bench pass@1 from 30% to 43% across iterations.

### Compute and Scaling Laws

Compute per training run is budgeted as $C = M \cdot D$ where $M$ is FLOP per token and $D$ is total tokens. For a dense transformer:

$$M = 6 N_{\text{ne}} + 6 d L S$$

with $N_{\text{ne}}$ non-embedding parameters, $d$ hidden dim, $L$ layers, $S$ sequence length. Empirically fitted scaling laws for LR and batch size:

$$\text{LR}(C) = 19.29 \cdot C^{-0.177}, \qquad \text{BS}(C) = 30.17 \cdot C^{0.231}$$

derived from 8 quasi-random search scales between $2 \times 10^{18}$ and $2 \times 10^{20}$ FLOP.

### Architecture

CWM is a standard dense decoder-only transformer — no architectural novelty. 32B parameters, 131k context, alternating local+global attention, rotary embeddings. The research claim lives entirely in the _training data_ and the _trajectory format_, not the backbone.

### Experimental Results

| Benchmark          | Metric | CWM-32B |
| ------------------ | ------ | ------- |
| SWE-bench Verified | pass@1 | 65.8%   |
| Math-500           | acc    | 96.6%   |
| LiveCodeBench (v5) | pass@1 | 68.6%   |
| CruxEval (output)  | acc    | 94.3%   |

The CruxEval-output result is the most interesting for the world-model thesis: CruxEval asks the model to predict the output of a Python snippet given its input — i.e., to simulate execution mentally. CWM outperforms much larger closed models on this task, supporting the claim that trace mid-training produces genuine execution-simulation ability rather than just better retrieval.

### Relation to Our Project

This paper is a completely different object from GWM and TGN — it is an LLM paper, not a GNN paper — but it is **directly relevant to the Coding Agent component of this project** (see `CLAUDE.md` §"Coding Agent (Planned)"). In fact, CWM and the planned GraphWorldModel coding agent are almost exact architectural analogs, just over different state spaces.

#### What's Shared

**1. Identical world-model framing.** Both projects use the exact same recurrence:

$$s_{t+1} = f_\theta(s_t, a_t)$$

CWM's state = Python local variables + stack frame; action = executed line. This project's state = graph node activations / connectivity / infection snapshot; action = seed / remove / source vector. Both train a forward model $f_\theta$ to be a _differentiable or query-able simulator_ that replaces expensive real execution inside an agent loop.

**2. Agent-over-simulator architecture.** CWM's ForagerAgent is a direct parallel to the planned $A_\phi(G, \text{task}, \text{history})$ in CLAUDE.md. In both cases: (a) a generator proposes actions, (b) a world model predicts their effect, (c) predicted rollouts are scored against an objective, (d) the generator is refined. CWM implements (d) via RL; the planned GWM coding agent leaves (d) open (refinement strategy unspecified).

**3. Structured action space.** CWM restricts the agent to `bash / edit / create / submit`. The planned GWM agent restricts to "candidate expansion, node selection, score propagation, subgraph update, termination." Both choose a small, parameterized primitive set rather than free-form code, for the same reason: _the world model only has to learn dynamics over a tractable action space_.

**4. Trajectory data as the training signal.** CWM's key data contribution is 120M+ traced execution trajectories. The analog for this project would be classical-algorithm execution traces: run greedy IM, PageRank, BFS, etc., on each dataset and capture the sequence of (graph-state, action-primitive, next-graph-state) tuples. Training the GWM on these traces — rather than on (x, y) pairs that collapse the whole trajectory — would be the direct translation of CWM's mid-training recipe.

#### What's Different

**1. State space and substrate.** CWM operates on program state (variable dictionaries) with an autoregressive LLM; this project operates on graph state $(N, 1)$ with message-passing GNNs. The algorithms are entirely different families, and CWM's sequence-token interface is not a drop-in for sparse graph tensors.

**2. Differentiability.** CWM's world model is _queried_ via text generation (sample the next observation). This project's forward model is _differentiable_ end-to-end, which is the entire point of Phase 2 logit optimization — gradients flow back from $y_{\text{target}}$ through the frozen model to the input logits. CWM cannot do this directly; its "optimization" is RL over discrete token outputs. The project has a strictly stronger inference-time mechanism for the kinds of combinatorial problems it targets.

**3. Supervision source.** CWM's supervision is real Python execution + real GitHub issues. This project's supervision is simulated IC cascades, BFS connectivity, etc. from the classical-algorithm module in `Data/`. Both are _cheap to generate at scale_, which is the enabling property for mid-training-style regimes.

**4. Scale.** CWM is 32B parameters trained on 13T tokens. This project's forward models are small GNNs trained on thousands of (x, y) pairs. The scaling laws CWM fits are not directly transferable, but the _shape_ of the claim (more trace data $\to$ better world model) is.

**5. The reasoning-mode toggle.** CWM can optionally emit a reasoning trace before answering. For the planned graph coding agent, the analog would be having the agent emit an _algorithm sketch_ or _action plan_ before calling world-model rollouts. This is an architectural pattern worth borrowing for the $A_\phi$ generator.

#### What's Applicable

**1. Train the forward model on algorithm execution traces, not just final (x, y) pairs.** This is the single most actionable idea — and it composes cleanly with the TGN-inspired "predict trajectory, not just union" suggestion from the previous entry. Concretely:

- For each dataset, run greedy-IM, PageRank, $k$-shell, CELF, BFS fragmentation, etc.
- At each step $t$, log the tuple $(s_t, a_t, s_{t+1})$ where $s_t$ is the current graph state (activated set, selected set, residual graph) and $a_t$ is the algorithm's chosen primitive (e.g., "select node with highest marginal gain").
- Train the forward model on $s_{t+1} = f_\theta(s_t, a_t)$ rather than on $y = f_\theta(x)$.

This gives dramatically more supervision per sample, and the resulting forward model can be rolled out autoregressively — which is the exact capability the planned coding agent needs.

**2. Formalize the action primitive set before building the agent.** CWM's agent works because the action space (`bash/edit/create/submit`) was fixed _before_ training. The project's CLAUDE.md lists candidate primitives ("candidate expansion, node selection, score propagation, subgraph update, termination") but does not pin down their signatures. A concrete next step is to define each primitive as `(operator, arguments) -> state-delta` with a strict schema, and then generate trace data by running _human-written reference algorithms_ expressed in this schema. Without this, there is no training data to mid-train on.

**3. Self-bootstrapping for the refinement loop.** CWM's RL-$\to$-rejection-sample-$\to$-SFT loop (30% $\to$ 43% on SWE-bench) is a recipe that transfers. For the GWM coding agent: run the agent with the current GWM, rejection-sample the trajectories that actually achieve the task objective (e.g., high spread, strong fragmentation), fine-tune both the agent $A_\phi$ and the forward model $f_\theta$ on the successful traces, repeat. This closes the loop between generator and simulator without needing a hand-tuned reward function up front.

**4. Neural-debugger capability as an evaluation harness.** CruxEval (predict output given input) is a benchmark that directly tests world-model fidelity. The graph analog is trivial to build: hold out some (algorithm, graph, initial-state) triples, ask the trained GWM to predict the final state, compare to ground truth from a real run. This gives a clean, fast _eval metric for world-model quality_ that is independent of Phase 2 inverse optimization quality — currently the project only measures the latter, which conflates model quality with optimization quality.

**5. Hybrid terminal reward for sparse tasks.** CWM's reward design — $+1$ on full success, $-1$ on failure with a soft patch-similarity tier in the middle — is a practical pattern for the CND/IM agent where "success" is nearly all-or-nothing. A soft shaping term (e.g., fraction of connected components broken, fraction of target spread achieved) in the middle tier would prevent the agent from seeing a flat reward landscape on partial solutions.

**6. Asymmetric PPO clipping and no-KL RL.** If/when the agent $A_\phi$ is trained with RL, CWM's recipe ($\varepsilon_{\text{high}} = 0.25$, $\varepsilon_{\text{low}} = 0.2$, no KL, length-normalize by context window) is a solid starting point. It's specifically tuned to let the model explore more aggressively (high upper clip) without drifting catastrophically (tight lower clip) and without the KL term suppressing the policy's ability to move away from a weak base.

---

### Bottom Line

CWM is not a graph paper, but it is _the clearest worked example of the exact agent architecture this project's CLAUDE.md describes_ — a generator proposing actions into a structured action space, a world model predicting their effect, and a refinement loop that uses predicted rollouts instead of real execution. The single most important lesson is **don't wait for the coding agent to be the last thing you build**. The forward models already in `World_Model/model/` are training on the wrong signal for that use case: they learn $y = f(x)$ (one-shot prediction), but the coding agent needs $s_{t+1} = f(s_t, a_t)$ (autoregressive rollout). These are different objectives, and switching the forward-model training data to _algorithm execution traces_ is a change that can be made now — it would strengthen Phase 2 optimization via richer supervision _and_ produce exactly the kind of world model the planned coding agent needs.

The secondary lesson is about _action-space discipline_. CWM works because FAIR fixed a minimal, well-typed action set before generating any trajectories. The project has a candidate list in CLAUDE.md but no schema; writing that schema down is a prerequisite to everything downstream. Finally, CWM's CruxEval-style "can the model simulate execution?" eval is a cheap win that this project should adopt — it gives a metric for _forward-model fidelity_ that is currently missing, since evaluation is done end-to-end through Phase 2.

In short: CWM validates the architectural bet in CLAUDE.md and tells us what the training data and evaluation harness should look like. It does not tell us how to build GNN world models — TGN and GWM remain the relevant references there — but it does tell us what shape the GWM's input/output interface has to take to support the agent use case.

---

## CompilerDream: Learning a Compiler World Model for General Code Optimization - https://arxiv.org/abs/2404.16077

**Code:** https://github.com/thuml/CompilerDream

### Summary

CompilerDream applies **model-based reinforcement learning (DreamerV3)** to the classic LLVM pass-ordering problem: given an intermediate representation (IR), pick the sequence of optimization passes that minimizes code size (or latency). The central bet is that a _learned compiler world model_ — a latent-space simulator of "what this IR looks like after applying pass $a$" — lets the agent plan entirely inside imagination, dramatically reducing the number of real compiler invocations needed and enabling zero-shot generalization to unseen code. It trains on 110k+ C++ programs from CodeContests and evaluates across CompilerGym benchmarks (cBench, CHStone, MiBench, NPB), production kernels (BLAS, Linux, OpenCV, TensorFlow), and 109k AI-generated C programs (FormAI). Beats LLVM's `-Oz` on most benchmarks zero-shot and tops the CompilerGym leaderboard.

This is the cleanest existing instantiation of the "world model as a simulator inside a code-generation agent loop" pattern — exactly the architecture described in CLAUDE.md §"Coding Agent (Planned)". Unlike CWM (which is an LLM learning execution semantics), CompilerDream is a small latent-dynamics model learning _action-effect dynamics over a discrete pass space_, which is much closer to what the GWM would do for graph-algorithm primitives.

### MDP Formulation

Compiler pass selection is cast as a POMDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, r, p, \mu, \mathcal{O}, \phi)$:

- **State** $s_t \in \mathcal{S}$: the current LLVM IR (too large to observe directly).
- **Action** $a_t \in \mathcal{A}$: one of 42 (limited) or 124 (full) LLVM optimization passes.
- **Observation** $o_t \in \mathcal{O}$: 56-dim Autophase features + 42-dim normalized action histogram (compact summary of the IR and the pass sequence so far).
- **Reward**: normalized instruction-count reduction relative to LLVM `-Oz`:

$$r_{t+1} = \frac{C(s_t) - C(s_{t+1})}{C(s_0) - C(s_b)}$$

where $C(\cdot)$ is the IR instruction count, $s_0$ is the unoptimized input, and $s_b$ is `-Oz`'s output. This normalization is important: it makes rewards comparable across programs of wildly different sizes, which is what allows training a _single_ agent across the full CodeContests corpus.

- **Initial distribution** $\mu$: uniform sample from the training set of programs.

### World Model Architecture (DreamerV3-style)

Four neural components, all operating in a compact latent space $z_t$:

**Representation model** (posterior, conditioned on observation):

$$z_t \sim q_\theta(z_t \mid z_{t-1}, a_{t-1}, o_t)$$

**Transition model** (prior, no observation — this is what enables imagination rollouts):

$$\hat{z}_t \sim p_\theta(\hat{z}_t \mid z_{t-1}, a_{t-1})$$

**Observation decoder** and **reward decoder**:

$$\hat{o}_t \sim p_\theta(\hat{o}_t \mid z_t), \qquad \hat{r}_t \sim p_\theta(\hat{r}_t \mid z_t)$$

The key conceptual split: $q_\theta$ is used to encode _real_ transitions from replay; $p_\theta$ is used to roll out _imagined_ trajectories entirely in latent space, so the compiler is never invoked during policy improvement.

### World Model Losses

**Representation loss** (Eq. 2) — train the encoder so latent states reconstruct observations and rewards:

$$\mathcal{L}_{\text{repr}}(\theta) = -\ln p_\theta(o_t \mid z_t) - \ln p_\theta(r_t \mid z_t)$$

**Prediction loss** (Eq. 3) — force the prior (no obs) to match the posterior (with obs):

$$\mathcal{L}_{\text{pred}}(\theta) = \text{KL}\big[ q_\theta(z_t \mid z_{t-1}, a_{t-1}, o_t) \,\|\, p_\theta(\hat{z}_t \mid z_{t-1}, a_{t-1}) \big]$$

Minimizing the KL makes imagined rollouts stay close to what the encoder would produce on real data.

### Reward Smoothing

The raw reward signal is **sparse and long-tailed**: most passes produce zero or near-zero instruction-count deltas, and the occasional big win dominates. This wrecks value learning. The fix is a first-order IIR filter applied to the reward stream before training the reward head:

$$r_t' \leftarrow \alpha \, r_{t-1}' + (1 - \alpha) \, r_t, \quad \alpha \in [0, 1)$$

The reward decoder then learns $p_\theta(\hat{r}_t' \mid z_t)$ on the smoothed signal. The authors find this is load-bearing — without it, the agent's value function fails to generalize across programs.

### Actor–Critic Training in Imagination

Both the actor and critic are trained **entirely on latent rollouts from the learned world model**. The compiler is not invoked during this phase.

**Policy** $\pi_\psi(a_t \mid \hat{z}_t)$ and **value** $v_\xi(\hat{z}_t) \approx \mathbb{E}\big[\sum_{\tau \geq t} \gamma^{\tau - t} \hat{r}_\tau \big]$.

**Critic loss** (Eq. 6), regression onto a bootstrapped $\lambda$-return $V^\lambda_\tau$:

$$\mathcal{L}_{\text{critic}}(\xi) = \mathbb{E}_{p_\theta, \pi_\psi} \Bigg[ \sum_{\tau=t}^{t+H} -\log v_\xi(V^\lambda_\tau \mid \hat{z}_\tau) \Bigg]$$

**Actor loss** (Eq. 7), REINFORCE with advantage baseline and entropy bonus:

$$\mathcal{L}_{\text{actor}}(\psi) = \mathbb{E}_{p_\theta, \pi_\psi} \Bigg[ \sum_{\tau=t}^{t+H} \Big( -(V^\lambda_\tau - v_\xi(\hat{z}_\tau)) \log \pi_\psi(\hat{a}_\tau \mid \hat{z}_\tau) - \eta \, \mathbb{H}[\pi_\psi(\hat{z}_\tau)] \Big) \Bigg]$$

Horizon $H$ is the imagination rollout length. The entropy term $\eta \mathbb{H}[\pi_\psi]$ is essential for exploration over the discrete 42-action space.

### Data and Benchmarks

**Training**: CodeContests (AlphaCode), ~110,240 C++ programs from 13k competitive-programming problems. 100 solutions held out for validation.

**Evaluation**:

- **CompilerGym suites**: cBench, CHStone, MiBench, NPB
- **Production code**: BLAS, Linux kernel modules, OpenCV, TensorFlow kernels
- **Synthetic stress test**: FormAI (109,016 AI-generated C programs)
- **Cross-language transfer**: 50 Objective-C programs generated by an LLM

### Key Numerical Results

| Benchmark / Setting    | Method                        | Code size reduction            | Time   |
| ---------------------- | ----------------------------- | ------------------------------ | ------ |
| cBench (autotuning)    | CompilerDream + Guided Search | **1.073×**                     | 60.8 s |
| cBench (single-trial)  | CompilerDream                 | **1.068×**                     | 2.9 s  |
| NPB (value prediction) | CompilerDream                 | **1.140×**                     | —      |
| NPB (value prediction) | Coreset-NVP (prior SOTA)      | 1.085×                         | —      |
| Diverse zero-shot      | CompilerDream vs. LLVM `-Oz`  | beats `-Oz` on most benchmarks | —      |

Headline claim: "CompilerDream outperforms [PPO], learning an order of magnitude faster with fewer compiler interactions." The sample-efficiency gap comes from being able to train the policy in imagination — every compiler call feeds many imagined updates.

### Intuition

The paper is a clean case study of **why model-based RL wins on expensive-environment problems**: invoking LLVM to apply a pass is slow (milliseconds to seconds), and you need thousands of interactions to learn a good policy. If you can train a fast neural approximation of "IR + pass $\to$ next IR" in a compact latent space, you can do most of your policy learning there and only occasionally fall back to the real compiler to correct model drift. The agent _dreams_ about pass sequences, evaluates them against the learned reward model, and only pays for real compiler calls when it needs fresh training data for the world model.

Reward smoothing is the ugly-but-important practical detail: the raw objective (instruction-count delta) is dominated by rare large wins, and without filtering, the critic never converges. This is the kind of "the signal is too sparse to learn from" pathology that shows up in any combinatorial optimization problem — including graph inverse problems.

### Relation to Our Project

CompilerDream sits in the same architectural family as CWM, but is a _much closer_ fit to what the planned GWM coding agent actually needs: a small, fast, latent-dynamics world model over a discrete, well-typed action space, used to train an RL agent entirely inside imagination. Read together with CWM (LLM-scale world model over program state) and the TGN entry (temporal dynamics on graphs), this paper fills in the third corner: how to _train an agent_ against a learned world model using RL.

#### What's Shared

**1. World model as environment surrogate.** CompilerDream's thesis is identical to CLAUDE.md §"Coding Agent (Planned)": classical evaluation (running LLVM on the IR / running a greedy graph algorithm on the dataset) is too expensive to do inside a tight agent loop, so train a differentiable simulator and let the agent explore inside it. The underlying recurrence is the same:

$$s_{t+1} = f_\theta(s_t, a_t), \qquad r_{t+1} = g_\theta(s_t, a_t)$$

In CompilerDream, $s_t$ is an IR (observed via Autophase features), $a_t$ is an LLVM pass, $r_t$ is instruction-count delta. In this project, $s_t$ would be graph state (activated set, residual connectivity, infection snapshot), $a_t$ would be a graph action primitive, and $r_t$ would be spread / fragmentation / snapshot-match delta.

**2. Small, discrete, well-typed action space.** CompilerDream works because LLVM exposes 42 well-defined passes. The planned GWM action set ("candidate expansion, node selection, score propagation, subgraph update, termination") is the same shape: small, discrete, strongly typed. This is the regime where DreamerV3-style latent models are known to work well; it is not a coincidence that CompilerDream, CWM's agent, and the planned GWM agent all make this choice.

**3. Large-scale training across instances for zero-shot transfer.** CompilerDream trains on 110k programs and evaluates zero-shot on held-out codebases including cross-language transfer (C $\to$ Objective-C). The analog for this project is training a single forward model across _all_ datasets (Cora-ML, Jazz, NetScience, Power Grid, NetHEPT) rather than training per-dataset. GWM (prior entry) and CompilerDream both make the same empirical claim: cross-instance training gives more generalization than per-instance tuning.

**4. Normalized reward for cross-instance comparability.** CompilerDream's instruction-count reward is normalized by `C(s_0) - C(s_b)` so that a 10-instruction program and a 10,000-instruction program contribute comparable gradient signals. This project has an analogous problem: spread on NetHEPT (15k nodes) and spread on Jazz (198 nodes) are not comparable as raw counts. Normalizing task metrics by a classical baseline (e.g., greedy IM, degree-centrality CND) is the same fix.

#### What's Different

**1. RL vs. direct logit optimization.** CompilerDream uses actor-critic RL (REINFORCE with $\lambda$-return critic) over discrete actions. This project uses gradient descent on a continuous logit vector through a fully differentiable forward model (Phase 2). For the current IM/CND/SL tasks, Phase 2 logit optimization is a strictly _stronger_ tool — you get real gradients instead of REINFORCE variance. RL only becomes necessary when the action space stops being a single $(N,)$ vector and becomes a _sequence_ of discrete primitive calls, which is exactly the Coding Agent setting.

**2. Latent-space dynamics vs. full-state dynamics.** CompilerDream predicts the _next latent_ $\hat{z}_t$, not the next IR. It decodes back to Autophase features only for training the encoder. The current GWM forward models operate directly on node-vectors — they don't have a latent state that is separate from the input/output. Moving toward latent dynamics (encoder + latent recurrence + decoder) is a non-trivial architectural shift but would be the right choice if rollouts need to be long.

**3. Single-algorithm target vs. algorithm-family target.** CompilerDream targets a single well-defined objective (minimize instruction count for LLVM). This project targets a family of tasks (IM, CND, SL) that already share a forward model. Multi-task conditioning is thus more central here than in CompilerDream.

**4. Observation format.** CompilerDream uses a 98-dim hand-engineered feature vector (Autophase + histogram). This project's forward models consume an $(N, 2)$ tensor (action channel + outcome channel) plus a sparse adjacency. The substrates are incomparable; what transfers is the training recipe, not the encoder.

**5. Reward smoothing is not (yet) a concern.** The current project trains on (x, y) pairs with MSE — no temporal reward, so no sparsity pathology. But the moment training switches to algorithm-execution trajectories (the CWM-inspired change recommended in the prior entry), reward smoothing becomes directly relevant: IM cascades produce most activations in early rounds and then long tails of near-zero deltas, which is exactly CompilerDream's sparse-long-tailed regime.

#### What's Applicable

**1. Use DreamerV3-style latent dynamics for the graph world model, not direct state prediction.** For the planned coding agent loop, the right architecture is:

- **Encoder** $q_\theta(z_t \mid z_{t-1}, a_{t-1}, o_t)$: takes the current graph state (activations + features + adjacency) and produces a compact latent.
- **Latent transition** $p_\theta(\hat{z}_t \mid z_{t-1}, a_{t-1})$: predicts the next latent given the chosen graph action primitive. This is what lets the agent roll out long trajectories cheaply.
- **Decoder** $p_\theta(\hat{o}_t \mid z_t)$: reconstructs graph state for supervision during world-model training.
- **Reward head** $p_\theta(\hat{r}_t \mid z_t)$: predicts task-specific metric delta (spread, connectivity loss, snapshot-match improvement).

This is strictly more expressive than the current one-shot $y = f(x)$ architecture and is the shape the agent loop needs. The current forward models in `World_Model/model/` can be reused as the encoder backbone.

**2. Adopt reward smoothing the moment training switches to trajectories.** The first-order IIR filter $r_t' = \alpha r_{t-1}' + (1 - \alpha) r_t$ costs nothing and solves the sparse-reward problem that will appear as soon as the project starts training on per-round cascade deltas. Include it from day one rather than debugging a flat critic later.

**3. Train the agent inside imagination, not against real algorithm runs.** This is the central CompilerDream insight and applies almost verbatim. Real runs of greedy IM, CELF, or BFS fragmentation are milliseconds-cheap on Jazz but seconds-expensive on NetHEPT; for the coding agent to search over long action sequences, rollouts have to happen in the latent world model. The real algorithms should only be used (a) to generate training data for the world model, and (b) to validate final proposed policies.

**4. Use $\lambda$-return critics with REINFORCE + entropy bonus for agent training.** When the agent $A_\phi$ is eventually trained, the DreamerV3 recipe (critic on bootstrapped $\lambda$-return, actor loss is advantage-weighted log-prob with entropy regularization) is a well-tested default and transfers directly — it handles discrete action spaces cleanly, unlike many continuous-control model-based RL recipes.

**5. Normalize task rewards by a classical baseline, cross-instance.** Borrow CompilerDream's reward-normalization structure: for IM, normalize predicted spread by _greedy-IM spread_ on the same graph; for CND, by _highest-degree removal_; for SL, by _random-source snapshot match_. This lets a single agent train coherently across Jazz and NetHEPT without being dominated by scale differences.

**6. Large-scale cross-instance training.** CompilerDream's zero-shot generalization to production code and FormAI is driven by training on 110k programs, not by clever architecture. The analog here is training the forward model / agent on a large corpus of synthetic graphs (random ER, BA, SBM, Watts-Strogatz) in addition to the 5 real datasets. The GWM's world-model role benefits from scale in instance diversity, not necessarily instance size.

---

### Bottom Line

CompilerDream is the piece the CWM entry pointed at but didn't fully specify: _how_ you train an agent against a learned world model when actions are discrete primitives and real environment calls are expensive. It's a DreamerV3 adaptation, and the recipe — encoder/transition/decoder split, reward smoothing, imagination rollouts, $\lambda$-return actor-critic — is directly portable to the planned GWM coding agent. The action space, reward structure, and cross-instance training setup are all more analogous to this project's needs than CWM's LLM-scale approach.

The concrete sequence of changes this paper suggests is:

1. **Refactor the forward model into an encoder + latent-transition + decoder** (DreamerV3 structure), keeping the current GCN/GT/GAT backbones as the encoder. This is compatible with Phase 1/Phase 2 training as it exists — Phase 2 can still optimize logits through the encoder — but it now also supports autoregressive rollouts through the latent-transition model.
2. **Add reward smoothing and classical-baseline normalization** to the reward head before any trajectory-level training begins.
3. **When the agent is built, train it in imagination** using the DreamerV3 actor-critic loss, not against real algorithm executions.

Crucially, steps 1 and 2 can be done now — they strengthen the existing forward-model pipeline on its own terms — without having to commit to the agent architecture. The latent-space split is a better forward model _and_ the prerequisite for everything downstream. Combined with the trajectory-prediction data recipe from the CWM and TGN entries, this gives a concrete architectural roadmap for turning the current `World_Model/` into a usable world model in the DreamerV3 sense rather than just a learned classifier.

CompilerDream's overall lesson for this project: the barrier between "forward model trained on (x, y) pairs" and "world model usable inside an agent loop" is smaller than it looks. It's essentially a matter of (a) splitting representation from transition, (b) training on trajectories instead of endpoints, and (c) adding a reward head. Everything else — the actor-critic training, the imagination rollouts — is standard DreamerV3 code.
