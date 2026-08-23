# Migration: self-describing checkpoints + graph-disjoint splits

Two blocking fixes, landed before any architectural refactoring. Nothing about
the world-model architecture changed: same encoders, same heads, same T_exo /
T_endo decomposition, same loss, same metrics.

---

## 1. What changed

### Checkpoints (`world_model/checkpoint.py`)

`train_wm.py` now writes `wm-ckpt-v2`:

```python
{"format": "wm-ckpt-v2",
 "spec": {backbone, head, diffusion_model, hidden_dim, n_layers, dropout,
          remove_semantics, action_encoding, hide_edge_weights,
          n_heads, ffn_dim, gcnii_alpha, gcnii_lamda},
 "state_dict": {...},
 "train_meta": {data_dir, split_mode, seed, epochs, lr, ..., selection_metric}}
```

`spec` is exactly what `WorldModel.__init__` consumes, so a checkpoint can be
rebuilt from the checkpoint. `train_meta` is provenance and is never read during
reconstruction — a corrupt meta block cannot change which model comes back.

Reconstruction now happens in ONE place. `world_model/eval_planning.py` and
`coding_agent/envs/world_model_env.py` each carried their own copy of the
rebuild-from-results-JSON logic; both now call `load_checkpoint`.

### The scorer (`world_model/scorer.py`)

```python
from world_model.scorer import WorldModelScorer, ScoringContext

scorer = WorldModelScorer.load("results/.../world_model/wm_sage_IC.pt")
ranked = scorer.rank_candidates(graph, candidates, ScoringContext(horizon=20, budget=5))
```

`predict_transition` / `rollout` / `score_candidate` / `score_candidates` /
`rank_candidates`. Rollouts delegate to `WorldModelEnvironment` — this is an API,
not a second implementation.

### Splits (`data/generate_wm_data.py`)

`--split-mode {graph_disjoint, episode_random}`, default `graph_disjoint`.

`graph_disjoint` assigns each **graph** to one split, stratified so the ratios
are exact and the assignment is reproducible from `--seed`. `episode_random` is
the historical per-episode draw, kept byte-for-byte so pre-2026-08-22 datasets
can be regenerated.

`metadata.json` now records `split_mode`, `splits_per_graph`,
`graphs_straddling_splits` and `split_is_graph_disjoint`, so an existing dataset
is self-auditing. `train_wm.py` prints a warning when it trains on a leaky one,
and the mode lands in both the checkpoint's `train_meta` and the results JSON.

**Measured on a 5-graph smoke run: `episode_random` put 2 of 5 graphs into more
than one split.** On the production 40-graph BA config it will be most of them.

---

## 2. Which checkpoints become legacy

**Every `.pt` file produced before this change.** They are bare `state_dict`s
(`wm-ckpt-v1-bare-state-dict`); their architecture is not in the file.

They are **not broken**. `load_checkpoint` still loads them when handed the run's
results JSON, and `WorldModelScorer.from_results_json(...)` does that for you.
What they cannot do is load standalone — `WorldModelScorer.load(path)` alone
raises `LegacyCheckpointError` naming the JSON to pass.

No `.pt` files are committed to this repository, so the affected set is whatever
is on the cluster under `results/*/world_model/wm_*.pt`. The runs they correspond
to are recorded in `world_model/checkpoints/old/`:

| results JSON | backbone | dynamics |
| --- | --- | --- |
| `ba20_marg_structured_gcn_IC.json` / `_LT.json` | gcn | IC / LT |
| `ba20_marg_structured_sage_IC.json` / `_LT.json` | sage | IC / LT |
| `ba20_marg_structured_gat_IC.json` / `_LT.json` | gat | IC / LT |
| `ba20_marg_structured_gt_IC.json` / `_LT.json` | gt | IC / LT |
| `ba20_marg_structured_gcnii_IC.json` / `_LT.json` | gcnii | IC / LT |

**Scientifically, all ten are superseded**, and not because of the checkpoint
format. They were trained on an `episode_random` split, so every number in
`world_model/checkpoints/RESULTS.md` — `delta_f1` 0.8294, `ens_marg_mae` 0.0946,
`ens_count_bias` −0.577, the 36.62 / 36.70 final counts, the Pareto frontier — is
an in-distribution number measured with the training graphs present at test time.
They are optimistic by an unknown amount and are **not comparable** to anything
produced under `graph_disjoint`.

Do not pool the two. Do not quote a mix.

---

## 3. Commands to regenerate and retrain

### 3.1 Data (IC and LT in one run)

The production BA config, unchanged except that the split is now explicit:

```bash
python -m data.generate_wm_data \
    --dataset ba --num-graphs 40 --syn-nodes 100 --ba-m 3 \
    --models IC LT --prob-model weighted --budget-pct-range 1 20 \
    --algorithms random degree pagerank betweenness \
    --rollouts 20 --horizon 10 --inject-p 0.4 --cf-prob 0.4 --cf-branches 2 \
    --action-ops add_node remove_node \
    --weight-lo 0.0 --weight-hi 1.0 \
    --mc-marginals 30 \
    --split 0.7 0.15 0.15 --seed 42 --split-mode graph_disjoint \
    --out-dir results/influence_maximization/ba/default/data

python -m data.validate_wm_data --dir results/influence_maximization/ba/default/data
```

Confirm the split before spending a GPU on it:

```bash
python -c "
import json
m = json.load(open('results/influence_maximization/ba/default/data/metadata.json'))
print('split_mode :', m['split_mode'])
print('disjoint   :', m['split_is_graph_disjoint'])
print('straddling :', m['graphs_straddling_splits'])
"
```

On Slurm the same thing is:

```bash
sbatch sbatch/influence_maximization/generate_data_ba.sbatch
```

which now passes `--split-mode graph_disjoint` explicitly.

### 3.2 Training — IC

```bash
python -m world_model.train_wm \
    --data-dir results/influence_maximization/ba/default/data \
    --diffusion-model IC \
    --model sage --head structured --hidden-dim 64 --n-layers 3 \
    --epochs 400 --lr 1e-3 --weight-decay 5e-4 --batch-size 32 \
    --pos-weight off --patience 50 --seed 42 \
    --device cuda
```

### 3.3 Training — LT

Identical but for the dynamics, which also switches the structured head from
`ICTransmissionHead` to `LTThresholdHead`:

```bash
python -m world_model.train_wm \
    --data-dir results/influence_maximization/ba/default/data \
    --diffusion-model LT \
    --model sage --head structured --hidden-dim 64 --n-layers 3 \
    --epochs 400 --lr 1e-3 --weight-decay 5e-4 --batch-size 32 \
    --pos-weight off --patience 50 --seed 42 \
    --device cuda
```

On Slurm:

```bash
sbatch sbatch/influence_maximization/train.sbatch sage IC results/influence_maximization/ba/default/data
sbatch sbatch/influence_maximization/train.sbatch sage LT results/influence_maximization/ba/default/data
```

Note `sbatch/influence_maximization/train.sbatch` defaults to
`--head structured_residual`; `RESULTS.md`'s numbers used `--head structured`.
Pick one deliberately — they are different models.

`--pos-weight off` is required for a structured head: its structural form already
prevents the all-zeros collapse, and `pos_weight` inflates the per-edge
transmission `q`.

### 3.4 Confirm the new checkpoint is self-describing

```bash
python -c "
from world_model.checkpoint import describe
from world_model.scorer import WorldModelScorer
p = 'results/influence_maximization/ba/default/world_model/wm_sage_IC.pt'
print(describe(p))
print(WorldModelScorer.load(p).describe())   # no config, no results JSON
"
```

Expected shape (from the smoke run):

```
wm_sage_IC.pt: sage/structured IC hidden=64 layers=3 remove=spent
               encoding=basic hide_w=False [split_mode=graph_disjoint seed=42]
```

---

## 4. Single real graphs cannot be split disjointly

One graph admits no graph-disjoint split, so `graph_disjoint` **fails fast**
rather than emitting a train-only dataset that looks like a three-way split.
Six generation scripts hit this and now pass `episode_random` explicitly with a
comment saying why:

```
sbatch/influence_maximization/generate_data_{jazz,nethept,netscience,power_grid}.sbatch
sbatch/adaptive_online_im/generate_data_nethept.sbatch
sbatch/critical_node_detection/generate_data_{power_grid,ppi_yeast}.sbatch   (via SPLIT_MODE)
```

Their test metrics are **in-graph**, not held-out-graph, and must not be pooled
with synthetic `graph_disjoint` runs.

This is a placeholder, not a resolution. The leakage-free way to use a real graph
is as a **transfer test set** for a model trained on synthetic families — which
is exactly the Phase-6 generalization experiment. Deciding that is a separate
call and is not made here.

---

## 5. Regression tests

`tests/test_checkpoint.py` (32) and `tests/test_split_modes.py` (29). Full suite:
179 passing.

The two that matter most, because they cover failures that produce no error:

* `test_disagreeing_config_is_refused_rather_than_silently_preferred` — a `spent`
  checkpoint under a `blocked` config has identically shaped tensors, loads
  cleanly, and biases every containment number. Now refused.
* `test_the_two_modes_differ_only_in_split_labels` — the `base_rng` stream is held
  identical across modes, so regenerating under the corrected split changes the
  labels and nothing else. That makes old-vs-new a controlled comparison rather
  than two unrelated datasets.
