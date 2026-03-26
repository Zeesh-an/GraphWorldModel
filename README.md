# Graph World Model — Influence Maximization

Research codebase for **Influence Maximization (IM) as a Graph Inverse Problem**.
Generates training data from diffusion simulations on real-world graphs and provides
a foundation for building ML models (GNN, RL, generative) that solve IM end-to-end.

---

## Problem

Given a social network `G = (V, E)` and a budget `k`, find the seed set `S ⊆ V` with
`|S| = k` that maximizes expected influence spread under a diffusion model:

```
S* = argmax_{S ⊆ V, |S| ≤ k}  σ(S, G)
```

**NP-hard.** Classical greedy achieves a `(1 - 1/e)` approximation but is too slow for
large graphs. This project targets learned surrogates that generalize across graphs.

---

## Diffusion Models

| Model | Mechanism |
|-------|-----------|
| **IC** (Independent Cascade) | Active node fires each neighbor independently with prob `p(u,v) = 1/in_deg(v)` |
| **LT** (Linear Threshold) | Node activates when `Σ w(u,v)` over active neighbors ≥ random threshold `θ ~ U[0,1]` |

---

## Dataset — Cora-ML

Citation network. Nodes = papers, edges = citations.

| Property | Value |
|----------|-------|
| Nodes | 2,995 |
| Edges | 8,416 |
| Node features | 2,879 (bag-of-words) |
| Classes | 7 |
| Mean in-degree | ~2.8 |
| Source | [graph2gauss](https://github.com/abojchevski/graph2gauss) |

### Generated Samples (1,000 per model, k=10, 100 MC runs)

| Model | Spread mean | Spread std | Spread range | Avg cascade depth |
|-------|-------------|------------|--------------|-------------------|
| IC | 18.2 | 4.3 | [11, 46] | 3.3 steps |
| LT | 18.3 | 2.7 | [10, 27] | 3.5 steps |

---

## Project Structure

```
GraphWorldModel/
├── README.md
├── RESEARCH.md              # Literature survey — methods, datasets, SOTA
├── requirements.txt
├── .venv/                   # Python 3.12 virtual environment
│
├── Data/
│   ├── generate_im_data.py  # Data generation pipeline
│   └── cora_ml/
│       ├── cora_ml.npz      # Raw graph (downloaded automatically)
│       ├── graph_data.npz   # Processed graph — edges, probs, features
│       ├── samples_ic.npz   # IC simulation samples
│       ├── samples_lt.npz   # LT simulation samples
│       └── metadata.json    # Generation config and stats
│
└── Baselines/               # Baseline model implementations (WIP)
```

---

## Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Generate Data

```bash
# Quick test (50 samples)
python Data/generate_im_data.py --samples 50 --k 5 --mc-runs 10

# Standard (1000 samples, k=10, both models)
python Data/generate_im_data.py --samples 1000 --k 10 --mc-runs 100 --models IC LT

# Large seed budget (high-spread regime)
python Data/generate_im_data.py --samples 1000 --k 50 --mc-runs 100
```

All arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--samples` | 1000 | Number of (seed_set, spread) samples per model |
| `--k` | 10 | Seed set size |
| `--mc-runs` | 100 | Monte Carlo runs per sample for spread estimation |
| `--models` | IC LT | Diffusion models to simulate |
| `--max-steps` | 50 | Max cascade propagation timesteps |
| `--seed` | 42 | RNG seed |
| `--out-dir` | Data/cora_ml | Output directory |

Cora-ML downloads automatically on first run (~2 MB).

---

## Data Format

### `graph_data.npz`

```python
import numpy as np
g = np.load("Data/cora_ml/graph_data.npz")

g["edge_index"]   # (2, E)   int32   — COO format [src_nodes, dst_nodes]
g["ic_probs"]     # (E,)     float32 — IC propagation probability per edge
g["lt_weights"]   # (E,)     float32 — LT edge weights (1/in_deg, sums ≤ 1 per node)
g["node_feats"]   # (N, F)   float32 — bag-of-words node features
g["node_labels"]  # (N,)     int32   — 7-class labels
```

### `samples_ic.npz` / `samples_lt.npz`

```python
from Data.generate_im_data import load_samples
from pathlib import Path

seeds, spreads, stds, cascades = load_samples(Path("Data/cora_ml"), model="IC")

# seeds     : (S, k)  int32   — seed node indices per sample
# spreads   : (S,)    float32 — mean influence spread over MC runs
# stds      : (S,)    float32 — std of spread over MC runs
# cascades  : list of (T_i, N) bool arrays — one per sample
```

### Cascade Trace (timestep-by-timestep)

```
cascade[i]  shape = (T_i, N)   dtype=bool

  row 0  →  seed nodes (t=0, initial activation)
  row 1  →  nodes newly activated at t=1
  row 2  →  nodes newly activated at t=2
  ...
  row T-1 → final wave (empty if cascade stopped)
```

Example for a single sample:

```python
c = cascades[0]          # shape e.g. (4, 2995)
seeds_t0 = np.where(c[0])[0]   # [321, 1374, ...]   — seed set
wave_1   = np.where(c[1])[0]   # [89, 203, ...]     — first propagation
total    = np.any(c, axis=0).sum()  # total nodes reached
```

---

## Loading for Model Training

```python
import numpy as np
from pathlib import Path
from Data.generate_im_data import load_graph, load_samples

data_dir = Path("Data/cora_ml")

# Graph structure
graph = load_graph(data_dir)
edge_index = graph["edge_index"]   # (2, 8416)
node_feats = graph["node_feats"]   # (2995, 2879)

# IC training samples
seeds, spreads, stds, cascades = load_samples(data_dir, model="IC")

# Paradigm A — direct regression (seed set → spread)
X = seeds    # (1000, 10)  input: which nodes are seeds
y = spreads  # (1000,)     target: expected spread

# Paradigm B — timestep diffusion (feed cascade sequence to sequence model)
for i, cascade in enumerate(cascades):
    T, N = cascade.shape   # (timesteps, nodes)
    # cascade[t] = active mask at step t → input to GNN per timestep
```

---

## SOTA Methods (see RESEARCH.md for full survey)

| Method | Venue | Approach | Status |
|--------|-------|----------|--------|
| IMM | SIGMOD 2015 | Reverse influence sampling | Classical ceiling |
| GCOMB | NeurIPS 2020 | GCN pruning + Q-learning | Strong baseline |
| ToupleGDD | 2022 | 3×GNN + Double DQN | Strong baseline |
| **DeepIM / PIANO** | **ICML 2023** | **VAE over seed sets + GNN diffusion** | **SOTA (single-layer)** |
| REM | 2025 | Seed2Vec VAE + RL | SOTA (multiplex) |

---

## References

- Kempe et al., "Maximizing the Spread of Influence through a Social Network," KDD 2003
- Ling et al., "Deep Graph Representation Learning and Optimization for Influence Maximization," ICML 2023 — [paper](https://proceedings.mlr.press/v202/ling23b/ling23b.pdf)
- Chen et al., "ToupleGDD: A Fine-Designed Solution of Influence Maximization by Deep Reinforcement Learning," 2022 — [arxiv](https://arxiv.org/abs/2210.07500)
- Tiara et al., "Multi-task Learning for Influence Estimation and Maximization," 2019 — [arxiv](https://arxiv.org/abs/1904.08804)
- Huang et al., "Scalable Continuous-time Diffusion Framework for Network Inference and Influence Estimation," WWW 2024 — [arxiv](https://arxiv.org/abs/2403.02867)
