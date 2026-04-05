# Graph World Model

A unified framework for solving **graph inverse problems** using a learned world model. Given a graph and a desired or observed outcome, the system infers the node-level action (seed set, removal set, source set) that produces that outcome — without running expensive classical algorithms at inference time.

---

## Graph Inverse Problems

A graph inverse problem asks: **given a graph and an outcome, what input caused it?**

The forward process is well-defined (e.g., diffusion spreads from seeds to neighbors), but the inverse is combinatorial and NP-hard. Classical approaches (greedy, CELF++, simulated annealing) require thousands of forward simulations per query. This project replaces that with a learned world model that can be queried via gradient-based optimization in continuous latent space.

### Tasks

| Task                              | Problem                                                               | Action (Channel 0)                  | Outcome (Channel 1)                              |
| --------------------------------- | --------------------------------------------------------------------- | ----------------------------------- | ------------------------------------------------ |
| **Influence Maximization (IM)**   | Select k seed nodes to maximize information spread                    | Binary seed vector                  | Full cascade activation (union across timesteps) |
| **Critical Node Detection (CND)** | Select k nodes whose removal maximally fragments the network          | Binary removal vector               | Residual connectivity (largest CC membership)    |
| **Source Localization (SL)**      | Given an observed infection snapshot, infer the original source nodes | Binary source vector (ground truth) | Partial snapshot at random time t                |

**IM** is a design problem — optimize toward an ideal outcome (activate all nodes).
**CND** is a design problem — optimize toward an ideal outcome (disconnect all nodes).
**SL** is an inference problem — optimize toward an observed outcome (match the snapshot).

---

## Datasets

| Dataset        | Nodes   | Edges  | Type                        | Node Features          | Source                                                                   |
| -------------- | ------- | ------ | --------------------------- | ---------------------- | ------------------------------------------------------------------------ |
| **Cora-ML**    | 2,995   | 8,416  | Directed (citations)        | 2,879-dim bag-of-words | [graph2gauss](https://github.com/abojchevski/graph2gauss)                |
| **Jazz**       | 198     | 2,742  | Undirected (collaborations) | log(1 + degree)        | [Network Repository](https://networkrepository.com/arenas-jazz.php)      |
| **NetScience** | 1,589   | 2,742  | Undirected (coauthorship)   | log(1 + degree)        | [Netzschleuder](https://networks.skewed.de/net/netscience)               |
| **Power Grid** | 4,941   | 6,594  | Undirected (power lines)    | log(1 + degree)        | [Network Repository](https://networkrepository.com/opsahl-powergrid.php) |
| **NetHEPT**    | 15,229  | 62,752 | Directed (citations)        | log(1 + total degree)  | [GitHub mirror](https://github.com/SparklyYS/Simultaneous-IMM)           |
| **Digg**       | 116,893 | ~2.6M  | Undirected (friendships)    | log(1 + degree)        | [Syracuse datasets](https://datasets.syr.edu/datasets/Digg.html)         |
| **Twitter**    | 81,306  | 1.77M  | Symmetrized to undirected   | log(1 + degree)        | [SNAP](https://snap.stanford.edu/data/ego-Twitter.html)                  |

IM and SL respect the original graph directionality. CND symmetrizes all graphs to undirected (connectivity is measured via undirected connected components).

---

## Architecture

The system has two jointly trained models that together form the world model:

### Model 1: Variational Autoencoder (VAE)

Encodes and decodes the action vector (seed/removal/source set) through a continuous latent space.

```
Encoder: (N,) -> hidden -> hidden -> hidden -> (latent_dim,)
Decoder: (latent_dim,) -> latent_dim -> hidden -> hidden -> (N,)
Activation: ReLU (encoder), Sigmoid output (decoder)
```

The VAE maps sparse binary node-set vectors into a smooth, continuous latent space where gradient-based optimization is tractable. Directly optimizing a binary vector is combinatorial; optimizing the latent z is smooth.

### Model 2: Graph Transformer (GT)

A differentiable forward model that predicts the outcome of an action on the graph.

```
Input: (N, 1) action probs + degree PE -> input_proj -> (N, d_model)
GT Layers x n_layers:
    LayerNorm -> Multi-Head Attention (with graph-masked softmax) -> residual
    LayerNorm -> FFN (GELU) -> residual
Output: LayerNorm -> Linear -> Sigmoid -> (N, 1)
```

The GT uses **scatter softmax** over graph neighborhoods — each node only attends to its neighbors, not all nodes. Positional encoding is degree-based sinusoidal (no learned parameters).

---

## Methodology

### Training Data Format

All tasks use the same tensor format: **(S, N, 2)**

- **Channel 0**: the action vector (binary, sparse) — seeds for IM, removals for CND, sources for SL
- **Channel 1**: the outcome vector (binary or continuous) — spread pattern, connectivity, or snapshot

Data is generated by running the forward process (diffusion simulation or connectivity computation) on random action sets, producing (action, outcome) pairs.

### Phase 1: Joint Training

Both models train together end-to-end:

```
x = inverse_pairs[:, :, 0]    # ground truth action vector
y = inverse_pairs[:, :, 1]    # ground truth outcome vector

z = encoder(x)                # encode action to latent space
x_hat = decoder(z)            # reconstruct action from latent
y_hat = GT(x_hat, adj)        # predict outcome via graph transformer

Loss = BCE(x_hat, x) + MSE(y_hat, y)
```

The VAE learns to compress and reconstruct action vectors. The GT learns the forward mapping from actions to outcomes on this specific graph. Checkpoint saved when loss improves.

### Phase 2: Latent Optimization

Both models are **frozen**. A latent vector z_hat is optimized directly via backpropagation to find the optimal action.

**Initialization:** z_hat is initialized from the top 10% of training samples (by channel 1 sum), encoded and averaged in latent space. This warm-starts the optimization in a promising region.

**Optimization loop** (gradient descent on z_hat):

```
for each iteration:
    x_hat = decoder(z_hat)          # decode to action probabilities
    y_hat = GT(x_hat, adj)          # predict outcome
    loss = MSE(y_hat, y_target) + L0_sparsity(x_hat)
    loss.backward()                 # gradients flow through GT and decoder to z_hat
    optimizer.step()                # update z_hat
```

**y_target differs by task:**

| Task | y_target          | Meaning                                       |
| ---- | ----------------- | --------------------------------------------- |
| IM   | `ones(N)`         | Activate all nodes (maximize spread)          |
| CND  | `zeros(N)`        | Disconnect all nodes (maximize fragmentation) |
| SL   | observed snapshot | Match the observed infection pattern          |

The L0 sparsity term `sum(abs(x_hat)) / N` encourages the decoded action vector to remain sparse (few selected nodes).

**Final extraction:** After optimization converges, the top-k nodes by probability in the decoded x_hat form the predicted action set.

---

## Quick Start

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.12+. Dependencies: `torch`, `numpy`, `scipy`, `networkx`, `pandas`, `scikit-learn`.

### 2. Generate data

```bash
# Influence Maximization (Cora-ML, IC diffusion, k=10)
python Data/generate_im_data.py -d cora_ml --k 10 --samples 1000

# Critical Node Detection (Jazz, k=10)
python Data/generate_cnd_data.py -d jazz --k 10 --samples 500

# Source Localization (Power Grid, IC diffusion, k=5)
python Data/generate_sl_data.py -d power_grid --k 5 --samples 1000
```

### 3. Train

```bash
# IM on Cora-ML
python World_Model/train.py --task IM -d cora_ml -dm IC --k 10 --npz-dir Data/cora_ml

# CND on Jazz
python World_Model/train.py --task CND -d jazz --k 10 --npz-dir Data/jazz

# SL on Power Grid
python World_Model/train.py --task SL -d power_grid -dm IC --k 5 --npz-dir Data/power_grid
```

### CLI flags

| Flag                        | Default                   | Description                                               |
| --------------------------- | ------------------------- | --------------------------------------------------------- |
| `--task`                    | `IM`                      | Task: `IM`, `CND`, or `SL`                                |
| `-d` / `--dataset`          | `cora_ml`                 | Dataset name                                              |
| `-dm` / `--diffusion_model` | `IC`                      | Diffusion model: `IC`, `LT`, `SIS` (IM and SL only)       |
| `--k`                       | `10`                      | Seed/removal/source set size (must match data generation) |
| `-sp` / `--seed_rate`       | `1`                       | Seed rate for baseline .SG fallback                       |
| `--epochs`                  | `600`                     | Phase 1 training epochs                                   |
| `--opt-iters`               | `300`                     | Phase 2 latent optimization iterations                    |
| `--lr`                      | `1e-4`                    | Phase 1 learning rate                                     |
| `--lr-z`                    | `1e-4`                    | Phase 2 latent z learning rate                            |
| `--hidden-dim`              | `1024`                    | VAE hidden layer width                                    |
| `--latent-dim`              | `512`                     | VAE latent space dimension                                |
| `--gt-d-model`              | `64`                      | Graph Transformer hidden dim                              |
| `--gt-heads`                | `4`                       | GT attention heads                                        |
| `--gt-layers`               | `3`                       | Number of GT layers                                       |
| `--gt-ffn`                  | `128`                     | GT FFN hidden dim                                         |
| `--gt-dropout`              | `0.1`                     | GT dropout rate                                           |
| `--npz-dir`                 | `Data/cora_ml`            | Path to generated .npz data directory                     |
| `--ckpt-dir`                | `World_Model/checkpoints` | Where to save checkpoints                                 |

---

## Project Structure

```
GraphWorldModel/
├── README.md                            # This file
├── CLAUDE.md                            # Development instructions and architecture reference
├── RESEARCH.md                          # Literature survey
├── requirements.txt                     # Python dependencies
│
├── Data/
│   ├── generate_im_data.py             # Generate IM training data (seed sets + cascade traces)
│   ├── generate_cnd_data.py            # Generate CND training data (removal sets + connectivity)
│   ├── generate_sl_data.py             # Generate SL training data (source sets + partial snapshots)
│   ├── diffusion.py                    # IC and LT diffusion simulation functions
│   ├── connectivity.py                 # Node removal and residual connectivity computation
│   ├── graph_utils.py                  # Shared utilities: edge index, adjacency lists, save/load
│   │
│   ├── datasets/                       # Dataset download and loading scripts
│   │   ├── cora_ml.py                  # Cora-ML citation network (2,995 nodes, directed, bag-of-words features)
│   │   ├── jazz.py                     # Jazz musician collaborations (198 nodes, undirected)
│   │   ├── netscience.py               # Network Science coauthorship (1,589 nodes, undirected)
│   │   ├── power_grid.py              # US Western power grid (4,941 nodes, undirected)
│   │   ├── nethept.py                  # HEP-Theory citations (15,229 nodes, directed)
│   │   ├── digg.py                     # Digg social network (116,893 nodes, undirected)
│   │   └── twitter.py                  # Twitter ego network (81,306 nodes, symmetrized to undirected)
│   │
│   ├── cora_ml/                        # Generated data output (created by generate_*_data.py)
│   │   ├── graph_data.npz             # Static graph: edge_index, ic_probs, lt_weights, node_feats
│   │   ├── samples_im_ic_k10.npz     # IM samples: seed_sets, spreads, cascade_data
│   │   ├── samples_cnd_k10.npz       # CND samples: removal_sets, connectivity_vecs
│   │   ├── samples_sl_ic_k5.npz      # SL samples: seed_sets, snapshots, observe_times
│   │   └── metadata.json              # Dataset statistics and generation config
│   └── <dataset>/                      # Same structure for jazz/, netscience/, power_grid/, etc.
│
├── World_Model/
│   ├── train.py                        # Main training script (Phase 1 joint training + Phase 2 latent optimization)
│   ├── utils.py                        # Data loaders, adjacency processing, evaluation functions
│   ├── model/
│   │   ├── vae.py                      # Encoder, Decoder, VAEModel
│   │   └── graph_transformer.py        # GraphTransformerLayer, GraphTransformerForwardModel
│   └── checkpoints/                    # Saved .pt model checkpoints and results .txt files
│
└── Baselines/
    └── DeepIM/                         # Reference implementation (DeepIM, ICML 2023)
        ├── genim.py                    # Data generation for DeepIM
        ├── data/
        │   └── sparsegraph.py          # Sparse graph utilities
        └── main/
            ├── utils.py                # Training utilities
            └── model/                  # GAT, GIN, MLP architectures
                ├── dataloader.py
                ├── gat.py
                ├── gin_parser.py
                ├── graphcnn.py
                ├── mlp.py
                └── model.py
```
