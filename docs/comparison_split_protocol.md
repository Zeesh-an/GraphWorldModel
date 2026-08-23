# episode_random vs graph_disjoint — controlled comparison

**Run date:** 2026-08-22. **Machine:** local, CPU (14 cores), torch 2.x, `python3.13`.
**Scope:** BA-100 × 20 graphs, SAGE + `structured`, IC and LT. One seed (42).
Five-backbone sweep NOT run. `structured_residual` NOT run.

---

## 0. How to read this document

Two arms were trained **here, in this environment**, differing in one thing:

| | arm L (legacy) | arm D (corrected) |
| --- | --- | --- |
| `--split-mode` | `episode_random` | `graph_disjoint` |
| graphs straddling splits | **20 / 20** | **0 / 20** |
| everything else | identical | identical |

"Identical" is verified, not asserted: every transition record in the two
datasets is byte-for-byte the same, keyed by `(episode_id, t, branch)`. Only the
train/val/test *labels* differ. That is by construction — `_assign_split` is
still called in `graph_disjoint` mode and its draw discarded, precisely so the
`base_rng` stream, and therefore every budget and simulator seed, stays aligned.

**Arm L is not a reproduction of `world_model/checkpoints/RESULTS.md`.** It uses
the recipe recorded in `data/README.md:41`, but the historical run's true final
cascade size (36.70) differs from arm L's (41.37), so the historical dataset was
generated with flags that were never written down. The historical numbers appear
in §4 as context only.

**Arm L's test metrics are not valid.** They are measured with all 20 training
graphs present in the test set. They are listed so the size and direction of the
distortion can be seen, not so they can be quoted.

**Arm D's one-step and rollout metrics are not paired with arm L's.** The two
arms have different test sets — different graphs, not the same graphs scored
twice. Differences below therefore mix "leakage removed" with "different graphs",
and no per-metric significance can be claimed from n = 1 seed.

---

## 1. Headline

**On BA-100, removing the leak barely moves the prediction metrics.** That is
itself the finding, and it is not the reassuring one it looks like.

Twenty BA(n=100, m=3) graphs from one generator are near-exchangeable. A model
that has seen 14 of them has effectively seen the 15th, so holding graphs out
costs it almost nothing. The leak was real — 20/20 graphs straddled — but on this
family it had little to buy.

The consequence is the important part: **BA-100 cannot demonstrate
generalization, with or without the split fix.** Whatever these numbers show, they
do not show that the world model transfers. That case can only be made by the
Phase-6 experiments (BA → WS, BA → SBM, small → large, synthetic → real).

The one place the correction moves a number materially is **budget planning
regret**, which flips sign — and that metric has its own problem (§3).

---

## 2. Measured results

### 2.1 One-step prediction (teacher-forced, test split)

| metric | IC arm L | IC arm D | LT arm L | LT arm D |
| --- | --- | --- | --- | --- |
| `delta_f1` | 0.8614 | **0.8747** | 0.6649 | **0.6710** |
| `new_infection_f1` | 0.8614 | 0.8747 | 0.6667 | 0.6714 |
| `brier_infected` | 0.00133 | 0.00133 | 0.0361 | 0.0369 |
| `infected_acc` | 0.9905 | 0.9895 | 0.9494 | 0.9487 |
| `add_seed_success` | 1.000 | 1.000 | 1.000 | 1.000 |
| `remove_frontier_success` | 1.000 | 1.000 | 0.319 | **0.064** |
| `action_sensitivity` | 1.752 | 1.699 | 1.980 | 1.979 |

`delta_f1 ≡ new_infection_f1` under IC because IC is monotone; the run flags this
itself as `delta_f1_is_new_infection_f1`.

Arm D scores *higher* on `delta_f1` in both dynamics. This is not evidence that
leakage helps — it is evidence that the two test sets differ and that graph-to-
graph variation on BA-100 exceeds the leakage effect.

### 2.2 Free-running rollout (stochastic ensemble, 20 samples × 50 episodes)

| metric | IC arm L | IC arm D | LT arm L | LT arm D |
| --- | --- | --- | --- | --- |
| `ens_marg_mae` | 0.0895 | 0.0942 | 0.1266 | **0.1173** |
| `ens_count_w1` | 2.433 | 2.699 | 10.80 | **8.62** |
| `ens_count_bias` | +0.472 | −0.807 | +10.46 | +7.53 |
| final count model / true | 42.19 / 41.37 | 46.85 / 47.81 | 80.75 / 64.35 | 79.82 / 64.71 |

**IC: no saturation in either arm.** `count_bias` stays within ±1 node on a
~45-node cascade, which is the structural head doing its job — the property comes
from the closed form, not from the split, and holds regardless.

**LT is badly biased in both arms**: +10.5 and +7.5 nodes, model 80 vs true 64.
That is a pre-existing LT problem, not something the split correction caused or
fixed. It is the largest open defect these runs surface.

### 2.3 Action conditioning (the falsifiable suite)

| metric | IC arm L | IC arm D | LT arm L | LT arm D |
| --- | --- | --- | --- | --- |
| verdict | PASS | PASS | PASS | PASS |
| `effect_mae_norm` (null = 1.0) | 0.536 | 0.529 | 0.957 | **0.886** |
| `effect_pearson` | 0.907 | 0.914 | 0.736 | 0.757 |
| `effect_sign_agree` | 0.395 | 0.506 | 1.000 | 1.000 |
| `shuffle_delta_f1_drop` | 0.671 | 0.660 | 0.273 | 0.263 |
| `null_delta_f1_drop` | 0.719 | 0.751 | 0.274 | 0.275 |
| `n_pairs` | 363 | 309 | 441 | 435 |
| `seeded_p_infected_worst` | 0.9999989 | 0.9999989 | 0.9999989 | 0.9999989 |
| `removed_p_frontier_worst` | 1.0e-6 | 1.0e-6 | **0.937** | **0.918** |

Three readings:

1. **Action conditioning survives the correction.** Both IC arms sit at roughly
   half the "ignores the action" null, with `pearson` ≈ 0.91 and a shuffle costing
   0.66 `delta_f1`. This is the strongest column in the whole table.
2. **LT action conditioning is weak.** `effect_mae_norm` 0.89–0.96 is barely
   under the 1.0 null. It passes the test, but "passes" here means "detectably
   better than predicting no action effect at all", not "predicts action effects
   well".
3. **`seeded_p_infected_worst = 0.9999989` in all four**, exactly the clamp bound.
   That is the algebraic T_exo guarantee, and it is insensitive to the split by
   construction — an untrained model scores it too.

`removed_p_frontier_worst ≈ 0.92` shows the LT head does **not** honour
`remove_node` in the frontier channel on its worst node, which is what drags
`remove_frontier_success` to 0.064 in arm D. Pre-existing, and worth a look
before LT is used for anything downstream.

### 2.4 Off-policy rollout (`--ood-policies`)

Training injects actions roughly uniformly; a planner does not. Both sides of the
comparison use the same policy.

| dynamics | policy | MAE arm L | MAE arm D | bias arm L | bias arm D |
| --- | --- | --- | --- | --- | --- |
| IC | `degree_seed` | 0.0821 | 0.0816 | +0.27 | −0.67 |
| IC | `random_seed` | 0.0399 | 0.0363 | +0.63 | +0.13 |
| IC | `null` | 0.0000 | 0.0000 | 0.00 | 0.00 |
| LT | `degree_seed` | 0.1150 | 0.1002 | +7.87 | +3.47 |
| LT | `random_seed` | 0.0572 | 0.0516 | +1.77 | +0.17 |
| LT | `null` | 0.0000 | 0.0000 | 0.00 | 0.00 |

Arm D is equal or better on every off-policy row. The `null` policy scoring
exactly 0.0 is a degenerate case, not a result: with no actions and no seeds
nothing ever activates, so model and simulator agree trivially.

### 2.5 Planning

| metric | IC arm L | IC arm D | LT arm L | LT arm D |
| --- | --- | --- | --- | --- |
| `plan_regret_model` | 0.2437 ± 0.0549 | 0.2437 ± 0.0549 | 0.1913 ± 0.118 | 0.2037 ± 0.131 |
| `plan_regret_degree` | 0.2687 ± 0.0881 | 0.2687 ± 0.0881 | 0.2712 ± 0.197 | 0.2712 ± 0.197 |
| `plan_regret_random` | 3.109 ± 0.595 | 3.109 ± 0.595 | 3.366 ± 0.818 | 3.366 ± 0.818 |
| `budget_regret_norm` | **−0.034 ± 0.028** | **+0.116 ± 0.034** | +0.042 ± 0.063 | +0.138 ± 0.045 |
| `budget_seed_overlap` | 0.533 | 0.400 | 0.467 | 0.467 |

`budget_regret_norm` is regret against greedy-MC, normalised; negative means the
model's k-seed plan beat greedy-MC. Under the leaky split the IC model reads
**slightly better than greedy-MC (−0.034)**; under the corrected split it reads
**11.6% worse (+0.116)**. LT moves the same direction, +0.042 → +0.138.

**This is the only metric the correction moves materially, and it is the one that
matters for the research question.** It is also the weakest measurement in the
suite: n = 3 graphs, one seed, and §3 below.

---

## 3. A second leakage site the split fix does NOT close

`planning_regret_multi` and `planning_regret_budget_multi` score
`list(store)[:n_graphs]` — the first 5 and first 3 graphs in the graph store,
**regardless of split**. Under `graph_disjoint` those are:

```
ba_n100_m3_s42 -> train
ba_n100_m3_s43 -> test
ba_n100_m3_s44 -> val
ba_n100_m3_s45 -> train
ba_n100_m3_s46 -> val
```

So **planning regret is measured partly on training graphs in both arms.**
Fixing the dataset split did not make it a held-out metric.

This also explains the identical IC `plan_regret_model` across arms (0.2437 vs
0.2437, to four decimals) and its agreement with the historical 0.2438: in this
regime `planning_regret` is dominated by the candidate sampler and the graphs,
not by the model. As a decision-value metric on BA it is close to uninformative.

Not fixed here — the instruction was two blocking fixes and no architecture
change, and restricting these evaluators to test graphs changes what every
planning number means. It should be the next correction, before any Phase-7
decision-value claim.

---

## 4. Historical results — context only, NOT comparable

From `world_model/checkpoints/RESULTS.md` (`ba20_marg_structured`, single seed,
different machine, generation flags never recorded):

| metric | historical | arm L (here) | arm D (here) |
| --- | --- | --- | --- |
| `delta_f1` | 0.8294 | 0.8614 | 0.8747 |
| `ens_marg_mae` | 0.0946 | 0.0895 | 0.0942 |
| `ens_count_bias` | −0.577 | +0.472 | −0.807 |
| final count model / true | 36.62 / 36.70 | 42.19 / 41.37 | 46.85 / 47.81 |
| `plan_regret_model` | 0.2438 | 0.2437 | 0.2437 |

The historical run's **true** final count is 36.70 against arm L's 41.37. Since
the true count comes from the simulator, not the model, the two runs are on
different data. The historical dataset's generation flags are therefore not
recoverable from the repository, and none of its numbers can be paired with
anything here.

**Every number in `RESULTS.md` was produced under `episode_random` and is
potentially optimistic.** They should be retired rather than re-quoted; §2 above
supersedes them for SAGE + `structured`.

---

## 5. What these runs do and do not establish

Established:

* The corrected split works end to end and costs nothing: 0/20 graphs straddle,
  vs 20/20 before.
* IC action conditioning is real and survives the correction (`effect_mae_norm`
  0.53 vs a 1.0 null, `pearson` 0.91, shuffle costs 0.66 `delta_f1`).
* The T_exo guarantee holds exactly in all four runs (`seeded_p_infected_worst`
  = 0.9999989, the clamp bound).
* IC rollouts do not saturate under either split.
* Checkpoints are self-describing: all four load with
  `WorldModelScorer.load(path)` and no results JSON.

Not established, and not claimable from these runs:

* That the world model generalizes. BA-100 is too homogeneous to test it.
* That the split correction improves or degrades prediction quality — the test
  sets differ and n = 1 seed.
* Anything about planning value, because of §3.
* Anything about the other four backbones or `structured_residual`.

Open defects surfaced:

* LT rollout count bias +7.5 to +10.5 nodes (model 80 vs true 64).
* LT `remove_frontier_success` 0.064; the LT head does not honour `remove_node`
  in the frontier channel.
* LT action conditioning barely clears its null.
* Planning evaluators are not split-aware (§3).

---

## 6. Reproducing this

```bash
for MODE in graph_disjoint episode_random; do
  python -m data.generate_wm_data \
      --dataset ba --num-graphs 20 --syn-nodes 100 --ba-m 3 \
      --models IC LT --prob-model weighted --budget-pct-range 1 20 \
      --algorithms random degree pagerank betweenness \
      --rollouts 10 --horizon 10 --inject-p 0.3 --cf-prob 0.2 --cf-branches 2 \
      --action-ops add_node remove_node add_edge remove_edge set_edge_weight \
      --weight-lo 0.0 --weight-hi 1.0 --mc-marginals 30 \
      --split 0.7 0.15 0.15 --seed 42 --split-mode $MODE \
      --out-dir results/influence_maximization/ba/$MODE/data
  for DM in IC LT; do
    python -m world_model.train_wm \
        --data-dir results/influence_maximization/ba/$MODE/data --diffusion-model $DM \
        --model sage --head structured --hidden-dim 64 --n-layers 3 --dropout 0.1 \
        --epochs 400 --lr 1e-3 --weight-decay 5e-4 --batch-size 32 \
        --pos-weight off --patience 50 --seed 42 --device cpu \
        --plan-demo --plan-graphs 5 \
        --ood-policies degree_seed random_seed null \
        --plan-budget-k 5 --plan-budget-graphs 3
  done
done
```

Wall clock, all four training runs in parallel on 14 CPU cores: **~8 minutes**
(370–470 s each, early-stopped at epochs 72–97 of 400). Data generation: 11 s per
arm. No GPU or cluster needed at this scale.
