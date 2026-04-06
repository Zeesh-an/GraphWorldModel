# Baselines

Reference implementations from prior work, used for comparison against models in `World_Model/`.

---

## DeepIM (ICML 2023)

**Folder:** [DeepIM/](DeepIM/)

> Ling et al., "Deep Graph Representation Learning and Optimization for Influence Maximization," ICML 2023.
> [paper](https://proceedings.mlr.press/v202/ling23b/ling23b.pdf)

VAE + SpGAT approach for influence maximization. Two-phase training:

1. Joint VAE + GAT forward model training
2. Latent z optimization to find seed set

### Run

```bash
cd Baselines/DeepIM
python genim.py -d cora_ml -dm IC -sp 1
```

### Arguments

```
-d   dataset        jazz | cora_ml | power_grid | netscience | random5
-dm  diffusion      IC | LT | SIS
-sp  seed_rate      1 | 5 | 10 | 20   (×10 = seed set size %)
```

### Data

Pre-generated `.SG` pickle files are in `DeepIM/data/`.
Format: `<dataset>_mean_<diffusion><seed_rate*10>.SG`

Each file contains:

- `adj` — scipy sparse adjacency matrix
- `inverse_pairs` — `(S, N, 2)` array: `[:,: ,0]` = seed vectors, `[:,:,1]` = influence vectors

---

## Adding a New Baseline

1. Clone the repo into a new subfolder under `Baselines/`
2. Remove its `.git` directory: `rm -rf Baselines/<name>/.git`
3. Add a brief entry to this README
