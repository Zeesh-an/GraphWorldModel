# Phase 1 audit — repository alignment

Date: 2026-08-22. Audited tree: this repository at commit `3205d42`
(local snapshot of `Zeesh-an/GraphWorldModel`, zip dated 2026-08-03).

Every count below is **derived**, not transcribed. Reproduce with:

```bash
python -m scripts.inspect_registry            # prints all registries + counts
python -m scripts.inspect_registry --strict   # non-zero exit on any error
```

---

## 0. Scope note: two repositories were in play

The brief referenced facts from two different trees:

| brief section | lives in |
| --- | --- |
| §4 coding-agent alignment, §16 task adapters, §19 evaluator switching, §26 Slurm, §14 `--hide-edge-weights`, §15 IC/LT structured heads | **this repository** |
| §3 group counts (160→480), §23 planner/evaluator ranking mismatch, §24 preliminary results (31.8% call reduction, Precision@3 1.96×, calls-to-first-win 2.34 vs 3.43) | a **separate** repository, `HongjiPu/world-model` (the algorithm-edit / hierarchical-planner system) |

The two share no code, no registry and no checkpoint format.

**Decision: this repository is the shared repo.** It is the only tree containing
both the coding agent and the world model, which is what §5 requires. The other
repository's planner, gate and decision-value machinery — the source of the §24
numbers — is a **pending migration**, not something Phase 1 touched. Phase 4
("reproduce the current main result") therefore cannot run until that migration
lands; see §7 below.

`origin` is set to `https://github.com/Zeesh-an/GraphWorldModel.git` but could
not be fetched from the audit environment (private repo, no credentials). **The
Phase-1 work is written against the 2026-08-03 snapshot and must be rebased onto
the real `main` before it is trusted.**

---

## 1. Answers to §32 A–L

### A. Task count

**13 registered, 3 implemented.** Source: `pipeline/tasks.py::tasks`.

Implemented (runnable by `pipeline/run.py` today): `influence_maximization`,
`critical_node_detection`, `adaptive_online_im`.

Planned: `influence_blocking`, `epidemic_control`, `source_localization`,
`influence_estimation`, `cascade_reconstruction`, `network_inference`,
`cascade_prediction`, `cascading_failure`. Out of scope: `graph_completion`,
`temporal_forecasting`. Each planned entry names its own blocker in the registry.

The brief's §16 list of eight is a subset with different spellings
(`AdaptiveInfluence` → `adaptive_online_im`). The registry names are canonical.

### B. Algorithms per task

**67 total**, all emittable by the coding agent, across three pools:

| pool | count | serves |
| --- | --- | --- |
| `coding_agent/tools/algorithms.py` | 36 | influence_maximization, adaptive_online_im |
| `coding_agent/tools/dismantling_algorithms.py` | 24 | critical_node_detection, influence_blocking |
| `coding_agent/tools/adaptive_algorithms.py` | 7 | adaptive_online_im |

**6** of the 67 are used to harvest world-model training trajectories
(`data/wm_actions.py::spine_algorithms`).

### C. Graphs

**37**: 7 synthetic families (`ba, er, karate, kronecker, powerlaw_cluster, sbm,
ws`) + 30 real datasets. Synthetic size is a run parameter (`--syn-nodes`), not a
property of the registry entry.

### D. Dynamics

**7 declared, 2 simulated.** `IC` and `LT` are the only two
`data/wm_simulator.py::Simulator.reset` can step. `SIR`/`SIS`/`SEIR`,
`motter_lai` and `dc_power_flow` are declared by planned tasks and have no
simulator.

### E. World-model backbone / head

Backbones: `gcn, sage, gt, gat, gcnii` (`world_model/wm_model.py::backbones`).
Heads: `linear, structured, structured_residual, structured_oracle`, with
`structured` dispatching to `ICTransmissionHead` or `LTThresholdHead` on the
dynamics.

There is **no `GraphBackbone` interface**. `build_graph_input` emits both
`adj_norm` (consumed by GCN/GCNII) and `edge_index`/`edge_weight` (consumed by
SAGE/GAT/GT), so the backbone choice is coupled to the data layer. Phase 3.

### F. Training loss

`world_model/train_wm.py`:

```
loss = BCEWithLogits(logits[:, 0], y_inf) + BCEWithLogits(logits[:, 1], y_fr)
```

Targets are **Monte-Carlo soft marginals** (`--mc-marginals`, default 30), not
Bernoulli realizations. `pos_weight` is `auto` (clamped to [1, 50]) or `off`;
`off` is required for structured heads, whose structural form already prevents
the all-zeros collapse.

Checkpoint selection: **`val delta_f1`**, patience-based early stopping.

Gap vs §10: only the summed loss is logged. `loss_infected` / `loss_frontier`
are computed but discarded before the history entry is written.

### G. Checkpoint format

`torch.save(model.state_dict(), ...)` — a **bare state dict**. Architecture
(backbone, head, `hidden_dim`, `n_layers`, `remove_semantics`,
`action_encoding`, `hide_edge_weights`) exists only in the sibling results JSON
and the CLI flags.

**This blocks §18.** `WorldModelScorer.load(checkpoint)` cannot reconstruct a
model from a checkpoint alone. Phase 2 must add a config block to the saved blob.

### H. Coding-agent evaluator interface

`pipeline/conditions.py`. An arm spec is `<method>_<mode>[@evaluator]`, with
`valid_evaluators = (native, monte_carlo, oracle, world_model)`.
`needs_world_model()` gates checkpoint loading; `coding_agent/envs/world_model_env.py`
rolls `f_theta` autoregressively.

**§19 (config-switchable evaluator) already exists here** and did not need
building.

### I. Group definition

**None existed.** The unit was the episode id
`{graph_id}|{model}|{algorithm}|k{budget}|r{rollout}`, and
`data/generate_wm_data.py::_assign_split` draws train/val/test **per episode**,
so the same graph routinely appears on both sides of a split.

That is a leakage bug, and it is more serious than the group-count confusion the
brief asked about. See finding F-1.

### J. Is the "20-algorithm" world-model config stale?

**There is no 20-algorithm configuration in this repository.** The live numbers
are 6 (spine) and 67 (coding agent). The "20" belongs to the other repository's
`scripts/generate_edit_transitions.py` docstring, where it describes a CLI
default for root algorithms, not a registry.

### K. Do the coding agent and world model share a registry?

**No — and the mismatch is a naming mismatch, which is why nothing caught it.**

| world-model spine | coding-agent name |
| --- | --- |
| `random` | `random_seeds` |
| `degree` | `high_degree` |
| `pagerank` | `pagerank_seeds` |
| `betweenness` | `betweenness_seeds` |
| `celf` | `celf` |
| `local_search` | `hill_climbing` |

Verified against implementations, not guessed: `wm_actions.select_seeds` computes
`local_search` as "start from degree, then 1-swaps, keep improvements", and
`algorithms.py` documents `hill_climbing` as "Local search: start from degree,
1-swap while spread improves". `celf_local_search` is *not* the counterpart — it
refines a CELF seed set, not a degree seed set.

Two further unenforced couplings: `baselines/registry.py` carries a `task` field
with a comment saying a CND baseline must not join an IM sweep, but nothing
checked it; and each task's `default_baselines` is a tuple of bare strings that a
rename would silently dangle.

### L. Files modified

Phase 1 **added** files only; no existing file was edited except `.gitignore`.

```
registry/__init__.py         public API
registry/core.py             the five registries + wm_spine_aliases
registry/group.py            ExperimentGroup, enumerate_groups, split_group_key
registry/consistency.py      validate_registry_consistency, validate_experiment_config
registry/manifest.py         build_manifest, counts, digest
scripts/__init__.py
scripts/inspect_registry.py  the §2 command
configs/wm_main.yaml         first §25 config
tests/test_registry.py       22 tests
docs/audit_phase1.md         this file
```

---

## 2. What Phase 1 delivered

- **Five registries** (`TASK_REGISTRY`, `ALGORITHM_REGISTRY`, `GRAPH_REGISTRY`,
  `DYNAMICS_REGISTRY`, `BASELINE_REGISTRY`) built as **views** over the modules
  that already own those facts. No fact is re-declared, so nothing can drift.
- **`wm_spine_aliases`** — the one genuinely new fact, relating the two
  vocabularies, with a test that fails if either side renames.
- **`ExperimentGroup(task, graph, dynamics, arm, seed)`** — one canonical unit,
  with `enumerate_groups()` as the only way to get a count and
  `explain_group_count()` printing the factorization.
- **`validate_registry_consistency()`** — six checks, returning findings rather
  than raising, plus `require_consistent()` for preflight.
- **`python -m scripts.inspect_registry`** — prints everything, writes
  `results/registry_manifest.json` with a vocabulary digest for §26.
- **22 tests**, all passing; the full suite is 118 passing, no regressions.

`arm` rather than `algorithm` in `ExperimentGroup`: that is the dimension this
repository sweeps. `parse_arm` turns `evolve_free@world_model` or `external:imm`
into the thing that runs, and a published baseline is an arm exactly as a
coding-agent condition is. `algorithm_of_arm()` recovers the algorithm when the
arm names one.

---

## 3. Findings

| id | severity | finding | fix lands in |
| --- | --- | --- | --- |
| F-1 | **error** | `_assign_split` splits per episode, so one graph straddles train/test. Every reported test number is optimistic by an unknown amount. | **FIXED** — `--split-mode graph_disjoint` is now the default; see `docs/migration_split_and_checkpoints.md`. Retrain still outstanding. |
| F-2 | **error** | Checkpoints are bare state dicts; architecture is not recoverable from the file. Blocks `WorldModelScorer.load`. | **FIXED** — `wm-ckpt-v2` + `WorldModelScorer.load`; v1 still loads with its results JSON. |
| F-3 | warning | 30 of 36 IM algorithms the coding agent can emit are absent from the world model's training trajectories. Off-policy fidelity for those is untested. | Phase 5/7 |
| F-4 | warning | Only the summed loss is logged; per-head terms are computed then dropped. | Phase 3 |
| F-5 | warning | No `GraphBackbone` interface; backbone choice is coupled to `build_graph_input`. | Phase 3 |
| F-6 | info | 0 of 32 external baselines are `ready`; 29 `needs_setup`, 3 `blocked`. Condition 7 cannot run yet. | Phase 9 |
| F-7 | info | This tree has no git history; `origin` is set but unreachable from the audit environment. §26 commit-hash recording is unverified. | before Phase 2 |

F-3 is reported by `validate_registry_consistency()` on every run, so it stays
visible rather than becoming folklore.

---

## 4. Explicitly not done

- No existing behaviour was changed. `pipeline/tasks.py`, `wm_model.py`,
  `generate_wm_data.py` and the evaluation suite are untouched.
- The split-leakage fix (F-1) is **stated as a contract in a test** but not
  wired into the generator: changing it changes what every existing checkpoint
  was trained on, so it belongs with a retrain.
- No experimental results were produced, reproduced or estimated. The §24
  numbers belong to the other repository and are not reproducible here yet.
