# Data

Data generation pipelines for each graph inverse problem task.
Each task has its own generator script that produces standardised `.npz` files.

---

## Influence Maximization — Cora-ML

**Script:** [generate_im_data.py](generate_im_data.py)

Generates `(seed_set, influence_spread, cascade_trace)` samples by simulating
IC and LT diffusion on the Cora-ML citation graph.

### Quick start

```bash
# From project root
source .venv/bin/activate

# Standard run (downloads Cora-ML automatically on first use)
python Data/generate_im_data.py --samples 1000 --k 10 --mc-runs 100 --models IC LT
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--samples` | 1000 | Number of samples per diffusion model |
| `--k` | 10 | Seed set size |
| `--mc-runs` | 100 | Monte Carlo runs per sample (for spread estimation) |
| `--models` | IC LT | Diffusion models: `IC`, `LT` |
| `--max-steps` | 50 | Max cascade propagation timesteps |
| `--out-dir` | Data/cora_ml | Output directory |

### Output files

```
Data/cora_ml/
├── cora_ml.npz          # raw download (auto)
├── graph_data.npz       # processed graph
├── samples_ic.npz       # IC simulation samples
├── samples_lt.npz       # LT simulation samples
└── metadata.json        # generation config + stats
```

### graph_data.npz

| Key | Shape | Description |
|-----|-------|-------------|
| `edge_index` | `(2, E)` int32 | COO edge list |
| `ic_probs` | `(E,)` float32 | IC propagation probability per edge |
| `lt_weights` | `(E,)` float32 | LT edge weights |
| `node_feats` | `(N, 2879)` float32 | Bag-of-words node features |
| `node_labels` | `(N,)` int32 | 7-class labels |

### samples_ic.npz / samples_lt.npz

| Key | Shape | Description |
|-----|-------|-------------|
| `seed_sets` | `(S, k)` int32 | Seed node indices per sample |
| `spreads` | `(S,)` float32 | Mean influence spread (over MC runs) |
| `spread_stds` | `(S,)` float32 | Std of spread |
| `cascade_data` | flat bool | Packed cascade traces (ragged) |
| `cascade_offsets` | `(S+1,)` int64 | Index boundaries for unpacking |
| `cascade_lengths` | `(S,)` int32 | Number of timesteps per sample |

### Loading

```python
from pathlib import Path
from Data.generate_im_data import load_graph, load_samples

graph  = load_graph(Path("Data/cora_ml"))
seeds, spreads, stds, cascades = load_samples(Path("Data/cora_ml"), model="IC")

# cascades[i] → (T_i, N) bool  — activated nodes at each timestep
```

### Dataset stats (Cora-ML, k=10, 1000 samples)

| Model | Spread mean | Spread std | Range | Avg depth |
|-------|-------------|------------|-------|-----------|
| IC | 18.2 | 4.3 | [11, 46] | 3.3 steps |
| LT | 18.3 | 2.7 | [10, 27] | 3.5 steps |

---

## World-Model Trajectories — any dataset

**Script:** [generate_wm_data.py](generate_wm_data.py)

Generates action-conditioned discrete-time trajectories `(s_0, a_0, …, a_{T-1}, s_T)`
under NDlib IC/LT with exogenous graph-mutation actions injected by a mixed policy.
For training a world model on `(s_t, a_t) → s_{t+1}`.

### Quick start

```bash
python Data/generate_wm_data.py -d jazz --models IC LT --samples 2000 --horizon 25
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--samples` | 2000 | Trajectories per diffusion model |
| `--horizon` | 25 | Timesteps T per trajectory |
| `--k-seeds` | 3 | Seeds activated at t=0 |
| `--n-candidates` | 64 | Candidate non-edges added to the edge superset (enables EDGE_ADD) |
| `--max-actions` | 8 | Non-NULL action budget per trajectory |
| `--p-null` | 0.7 | Probability a timestep is NULL (no action) |
| `--poisson-lambda` | 0.5 | Extra actions per non-NULL bag ~ Poisson |
| `--models` | IC LT | NDlib diffusion models |

### Output: samples_wm_ic.npz / samples_wm_lt.npz (ragged)

| Key | Description |
|-----|-------------|
| `n_nodes` | `(1,)` N |
| `traj_lengths` | `(S,)` T_i per trajectory |
| `super_n_edges` | `(S,)` E_i per trajectory (edge superset size) |
| `state_data` + `state_offsets` | states[i] → `(T_i+1, N)` bool (activation) |
| `alive_data` (+ state_offsets) | alive[i] → `(T_i+1, N)` bool |
| `edge_mask_data` + `edge_mask_offsets` | edge_masks[i] → `(T_i+1, E_i)` bool |
| `edge_index_data` + `edge_index_offsets` | per-trajectory superset `(2, E_i)` |
| `action_op/target/param/t` + `action_offsets` | ragged action bags |

### Loading

```python
from pathlib import Path
from wm_storage import load_wm_samples, iter_transitions

trajs = load_wm_samples(Path("Data/jazz"), model="IC")
for s_t, a_t, s_next in iter_transitions(trajs[0]):
    ...  # s_t = {node_state, alive_mask, edge_mask, edge_index}; a_t = action bag
```

**Actions** (op ids): 0 NULL · 1 NODE_SET_STATE · 2 NODE_REMOVE · 3 EDGE_ADD · 4 EDGE_REMOVE.
NODE/edge targets index nodes / the per-trajectory edge superset; `param` is the new edge weight for EDGE_ADD.
