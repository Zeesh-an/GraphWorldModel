# Graph World Model

A framework for solving **graph inverse problems** using learned forward models. Given a graph and a desired or observed outcome, the system infers the node-level action (seed set, removal set, source set) that produces that outcome — without running expensive classical algorithms at inference time.

## How It Works

**Pipeline:** Generate (action, outcome) samples &rarr; Train a forward graph model &rarr; Optimize input logits &rarr; Extract best node set.

The forward model learns to predict `outcome = f(action, graph)`. At inference (Phase 2), we freeze the model and backpropagate through it to find the action that best achieves a target outcome. The action vector is parameterized as `x_hat = sigmoid(logits)`, and the top-k nodes by probability form the predicted node set.

## Tasks

| Task                              | Problem                                                 | Action (x)            | Outcome (y)                                      |
| --------------------------------- | ------------------------------------------------------- | --------------------- | ------------------------------------------------ |
| **Influence Maximization (IM)**   | Select k seed nodes to maximize spread                  | Binary seed vector    | Full cascade activation (union across timesteps) |
| **Critical Node Detection (CND)** | Select k nodes whose removal fragments the network      | Binary removal vector | Residual connectivity (largest CC membership)    |
| **Source Localization (SL)**      | Given an observed infection snapshot, infer the sources | Binary source vector  | Partial snapshot at random time t                |

**IM** and **CND** are design problems — optimize toward an ideal outcome. **SL** is an inference problem — optimize toward an observed outcome.

## Forward Models

Five forward model architectures are implemented, all sharing the same interface:

```
forward(seed_vec: (N, 1), adj: sparse COO (N, N)) -> (N, 1)
```

| Model         | Description                                      | Key Properties                                                                            |
| ------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| **GT**        | Graph Transformer with scatter-softmax attention | Scaled dot-product QKV, pre-norm residual, FFN, degree PE                                 |
| **GCN**       | Kipf-Welling GCN                                 | `X' = Â X W`, pre-norm residual blocks, degree PE                                         |
| **GAT**       | GATv2 (Brody et al. 2022)                        | Dynamic attention via `LeakyReLU(W_l h_v + W_r h_u)`, multi-head scatter-softmax          |
| **GraphSAGE** | Hamilton et al. 2017, mean aggregator            | `concat(h_self, mean(h_neighbors))`, self-loop-filtered aggregation                       |
| **GCNII**     | Chen et al. 2020                                 | Initial residual `(1-α)·Â·H + α·H⁰`, identity mapping `(1-β)·I + β·W`, designed for depth |

All models use degree-based sinusoidal positional encoding (can be disabled with `--<model>-no-pe`), sigmoid output activation, and xavier-uniform weight initialization.

## Datasets

| Dataset        | Nodes  | Edges  | Type                        | Node Features          |
| -------------- | ------ | ------ | --------------------------- | ---------------------- |
| **Cora-ML**    | 2,995  | 8,416  | Directed (citations)        | 2,879-dim bag-of-words |
| **Jazz**       | 198    | 2,742  | Undirected (collaborations) | log(1 + degree)        |
| **NetScience** | 1,589  | 2,742  | Undirected (coauthorship)   | log(1 + degree)        |
| **Power Grid** | 4,941  | 6,594  | Undirected (power lines)    | log(1 + degree)        |
| **NetHEPT**    | 15,229 | 62,752 | Directed (citations)        | log(1 + total degree)  |

## Quick Start

### Step 1 — Generate Data

> These diffusion-only generators are the **legacy** pipeline, archived under
> `data/old/`. The current action-conditioned pipeline is
> `data/generate_wm_data.py` (see [data/README.md](data/README.md)).

```bash
# Influence Maximization (Cora-ML, IC model, k=10, 1000 samples)
python data/old/generate_im_data.py -d cora_ml --k 10 --samples 1000

# Critical Node Detection (Jazz, k=10, 500 samples)
python data/old/generate_cnd_data.py -d jazz --k 10 --samples 500

# Source Localization (Power Grid, IC model, k=5, 1000 samples)
python data/old/generate_sl_data.py -d power_grid --k 5 --samples 1000
```

### Step 2 — Train Forward Model + Inverse Optimization

> The seed→final commands below are the **legacy** forward-model training, now under
> `world_model/old/`. The current path is the **action-conditioned world model**:
> `python world_model/train_wm.py --data-dir data/output/<setting> --diffusion-model IC --model gcn --plan-demo`
> (see the design spec in `docs/superpowers/specs/`).

```bash
# IM on Cora-ML with GraphSAGE
python world_model/old/train.py --task IM -d cora_ml -dm IC --k 10 \
    --model sage --sage-hidden 128 --sage-layers 3 \
    --epochs 600 --opt-iters 500 --lr 1e-4 --lr-z 1e-2 \
    --npz-dir data/cora_ml

# CND on Jazz with GCN
python world_model/old/train.py --task CND -d jazz --k 10 \
    --model gcn --gcn-hidden 64 --gcn-layers 3 \
    --epochs 600 --opt-iters 300 \
    --npz-dir data/jazz

# SL on Power Grid with Graph Transformer
python world_model/old/train.py --task SL -d power_grid -dm IC --k 5 \
    --model gt --gt-d-model 64 --gt-heads 4 --gt-layers 3 --gt-ffn 128 \
    --epochs 600 --opt-iters 300 \
    --npz-dir data/power_grid
```

### Key Training Flags

| Flag          | Default | Description                                                  |
| ------------- | ------- | ------------------------------------------------------------ |
| `--model`     | `gt`    | Forward model: `gt`, `gcn`, `gat`, `sage`, `gcnii`           |
| `--epochs`    | `600`   | Phase 1 supervised training epochs                           |
| `--opt-iters` | `300`   | Phase 2 logit optimization iterations                        |
| `--lr`        | `1e-4`  | Phase 1 learning rate                                        |
| `--lr-z`      | `1e-4`  | Phase 2 logit optimization learning rate                     |
| `--l0-weight` | `1.0`   | Phase 2 L1 sparsity penalty weight on x_hat                  |
| `--augment`   | off     | Phase 1 input augmentation (noise + label smoothing)         |
| `--k`         | `10`    | Seed/removal/source set size (must match data generation)    |
| `--k-pct`     | —       | Node budget as percentage of N (overrides `--k` for Phase 2) |

### Model-Specific Flags

```
GT:        --gt-d-model  --gt-heads  --gt-layers  --gt-ffn  --gt-dropout
GCN:       --gcn-hidden  --gcn-layers  --gcn-dropout  --gcn-no-pe
GAT:       --gat-hidden  --gat-heads  --gat-layers  --gat-dropout  --gat-no-pe
GraphSAGE: --sage-hidden  --sage-layers  --sage-dropout  --sage-no-pe
GCNII:     --gcnii-hidden  --gcnii-layers  --gcnii-alpha  --gcnii-lamda  --gcnii-dropout  --gcnii-no-pe
```

## Training Phases

### Phase 1 — Supervised Forward Training

The forward model learns to predict the outcome vector from the action vector:

```
x = ground truth action vector          # (N,) binary
y = ground truth outcome vector          # (N,) binary

y_hat = forward_model(x, adj)           # predict outcome
loss = MSE(y_hat, y)                    # supervised loss
```

With `--augment`, the input `x` is stochastically perturbed (additive noise, label smoothing) while the target `y` stays clean. This teaches the forward model to handle the soft inputs it will see in Phase 2.

### Phase 2 — Direct Logit Optimization

The forward model is frozen. A logit vector is optimized via backpropagation to find the action that best achieves the target outcome:

```
logits = initialized from top-performing training samples
x_hat = sigmoid(logits)                         # soft action probabilities
y_hat = forward_model(x_hat, adj)               # predicted outcome
loss = MSE(y_hat, y_target) + l0_weight * L1(x_hat)   # task loss + sparsity

# y_target differs by task:
#   IM:  ones(N)            — maximize spread
#   CND: zeros(N)           — maximize fragmentation
#   SL:  observed_snapshot   — match observation
```

The top-k nodes by probability in `x_hat` form the final predicted node set.

## Coding Agent (Planned)

The Graph World Model is designed to serve as a **predictive environment** inside a coding agent loop for graph algorithm evolution. This component is not yet implemented.

### Concept

A coding agent generates candidate graph algorithms tailored to a given graph and task. Evaluating these candidates by fully executing them on real graphs is expensive. The GWM provides a fast, differentiable simulator for predicted rollouts:

```
# The coding agent generates a candidate algorithm
pi = A_phi(G, task, history)

# The GWM predicts graph-state transitions without full execution
s_hat_next = f_theta(s_t, a_t, task)

# Predicted rollout provides algorithmic insights for refinement
rollout = Rollout_GWM(pi, G, task)  -->  Refine(pi)
```

### Graph Action Space

Rather than allowing unrestricted code execution, the agent operates over a structured set of parameterized graph action primitives (candidate expansion, node selection, score propagation, subgraph update, termination). Each action is represented as `(operator, arguments)`, providing a unified interface for both algorithm execution and world model prediction.

### Agent Loop

1. **Generate**: The coding agent produces a candidate graph algorithm as a composition of action primitives
2. **Simulate**: The GWM predicts the graph-state trajectory under the candidate algorithm
3. **Evaluate**: Predicted outcomes (spread, connectivity, reachability) are compared against the task objective
4. **Refine**: The agent uses the GWM's predictions as feedback to iteratively improve the algorithm

### Baselines (Planned)

| Baseline                  | Description                                                       |
| ------------------------- | ----------------------------------------------------------------- |
| **Native coding agent**   | Agent without GWM — improves only through real execution feedback |
| **Pure graph algorithms** | Fixed hand-designed algorithms (greedy IM, BFS, PageRank, etc.)   |
| **GA routing**            | Learned or heuristic selection over a predefined algorithm pool   |
| **Coding agent + GWM**    | Full method — agent guided by world model predictions             |

## Project Structure

```
GraphWorldModel/
├── data/
│   ├── generate_wm_data.py          # action-conditioned WM data generator (current)
│   ├── validate_wm_data.py          # gate-check harness for generated data
│   ├── wm_simulator.py              # NDlib stepwise IC/LT sim + State/ActionOp
│   ├── wm_graphs.py                 # graph providers (real + synthetic)
│   ├── wm_actions.py                # spine seed selectors + action injection
│   ├── graph_utils.py               # Graph utilities (shared)
│   ├── datasets/                    # Dataset loaders (cora_ml, jazz, etc.)
│   └── old/                         # legacy diffusion-only pipeline (archived)
│       ├── generate_im_data.py      # IM data generation
│       ├── generate_cnd_data.py     # CND data generation
│       ├── generate_sl_data.py      # SL data generation
│       ├── diffusion.py             # IC/LT diffusion simulators
│       └── connectivity.py          # Graph connectivity analysis
├── world_model/
│   ├── train_wm.py                  # Action-conditioned WM training (current)
│   ├── wm_model.py                  # WorldModel + backbone registry
│   ├── wm_data.py                   # Transition dataset + adjacency
│   ├── wm_metrics.py                # Metric suite (F1, Success, Sensitivity, Regret)
│   ├── wm_eval.py                   # Evaluation + rollout + planning demo
│   ├── model/
│   │   ├── graph_transformer.py     # Graph Transformer forward model
│   │   ├── gcn.py                   # GCN forward model
│   │   ├── gat.py                   # GATv2 forward model
│   │   ├── graphsage.py             # GraphSAGE forward model
│   │   ├── gcnii.py                 # GCNII forward model
│   │   ├── model_utils.py           # Shared utilities (degree encoding)
│   │   └── vae.py                   # VAE encoder/decoder (used by train_vae.py)
│   └── old/
│       ├── train.py                 # Legacy forward-model training
│       ├── train_vae.py             # Legacy VAE+GT joint training
│       └── utils.py                 # Legacy data loading & evaluation
├── baselines/
│   └── DeepIM/                      # DeepIM baseline implementation
└── requirements.txt
```
