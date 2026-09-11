# Action-conditioned message passing, and richer world-model feedback

Two questions, kept apart because they have different answers:

1. **Should the learned transition see the action explicitly**, or is modifying
   the node features before an ordinary GNN enough?
2. **Can the world model's node marginals be turned into feedback a reviser can
   act on**, without retraining anything?

Everything here is additive. `--action-conditioning none` and
`--feedback default` are the defaults and reproduce the previous behaviour
exactly; every existing checkpoint loads unchanged.

---

## 1. Where the action enters today

Tracing `s_t`, `a_t` and `T_exo` through the code as it stands:

```
record (data/generate_wm_data.py -> transitions_<dm>_<split>.jsonl)
  state.infected / state.frontier          <- s_t, PRE-action
  action[]  {op, target, destination, weight}   <- a_t
        |
        v  world_model/wm_data.py :: build_features
X : (N, 6)  [CH_INFECTED, CH_FRONTIER, CH_DEGREE, CH_ADD, CH_REMOVE, CH_EDGE]
            ^^^^^ s_t ^^^^^^^^^^^^^^^^^  ^^^^^^^^^^ a_t ^^^^^^^^^^^^^^^^^^^^
        |   (`--action-encoding typed` splits CH_EDGE by op -> 9 columns)
        |   edge ops are ALSO applied to the adjacency (apply_edge_ops)
        v
GraphInput(num_nodes, adj_norm, edge_index, edge_weight, batch_index)
        |
        v  world_model/wm_model.py :: WorldModel.forward
encoder = GraphSAGEEncoder | GCN | GAT | GraphTransformer | GCNII
        message m_{u->v} = w_uv * h_u        <- NO action term
        |
        v  head (ICTransmissionHead / LTThresholdHead / competitive / compartmental)
        T_exo applied ALGEBRAICALLY:  infected = clamp(X[:,CH_INFECTED] + X[:,CH_ADD])
        q_uv = sigmoid(MLP([h_u, h_v, w_uv]))
        p_new(v) = 1 - prod_u (1 - q_uv * frontier_u)
        |
        v  (N, 2) logits -> sigmoid -> per-node marginals
```

So the action reaches the model through **two** channels, both of which are
"preprocessing" in the brief's sense:

* three **input columns** of `X`, which the encoder sees at its input projection
  and then propagates like any other feature;
* the **closed-form `T_exo`** inside the head.

What is *not* action-conditioned is the message function itself. That is the gap
Variant B fills.

## 2. The change

`world_model/model/action_cond.py` + `world_model/model/graphsage.py`:

```
BEFORE   m_{u->v} = h_u * w_uv

AFTER    m_{u->v} = (h_u + phi_m([h_u, h_v, w_uv, z_a, r_uv])) * w_uv

         z_a  = ActionEncoder(action columns of X, h0, batch_index)   (per graph)
         r_uv = [a_u ; a_v]                                           (per edge)
```

Two properties make this a controlled comparison rather than a second model:

* `phi_m` is scaled by a learnable **gate initialised to zero**, so an untrained
  variant computes exactly the baseline's function;
* the variant's extra modules are drawn from a **forked RNG**, so a variant and
  a baseline built under one `torch.manual_seed` hold bit-identical shared
  tensors. The arms differ in capacity, not in starting point.

Four arms (`--action-conditioning`):

| arm             | `phi_m` sees                 | isolates                    |
| --------------- | ---------------------------- | --------------------------- |
| `none`          | (no `phi_m`)                 | the baseline                |
| `message_blind` | `h_u, h_v, w` (action zeroed)| the CAPACITY `phi_m` adds   |
| `global`        | `... + z_a`                  | global action conditioning  |
| `message`       | `... + z_a + r_uv`           | + LOCAL action relevance    |

`message_blind` is not in the brief's list and is the arm that makes the rest
readable: `message` adds parameters as well as information, and without a
capacity control a win could be either.

Implemented for `sage` only (the backbone `world_model/checkpoints/RESULTS.md`
settles on). The other four take `**_` and would silently drop the flag, so
`WorldModel` refuses them by name.

### The zero has to go on the gate, not on the output layer

The first sweep run here put the zero on `phi_m`'s output layer instead. That is
the obvious placement and it is wrong, in a way that silently answers the
research question:

* `dL/d(hidden layer) = W_out^T . dL/dout`, and `W_out = 0`, so at step 0 the
  message MLP's input layer and the **entire ActionEncoder** receive exactly no
  gradient. Measured on a real batch: `|grad action_encoder| = 0.000e+00`.
* Weight decay does not wait. Over a few thousand steps it drove those unused
  weights toward zero faster than `W_out` could grow, and the mechanism was
  regularised out of existence.

The trained checkpoints show it outright — every arm came back with
`max|phi_out| = 0.00000`, and under LT three architecturally different arms
posted **bit-identical** metrics across all eight columns:

```
LT_global_s42         |mod|=0.0000  max|phi_out|=0.00000000  delta_f1=0.9529
LT_message_blind_s42  |mod|=0.0001  max|phi_out|=0.00000000  delta_f1=0.9529
LT_message_s42        |mod|=0.0000  max|phi_out|=0.00000000  delta_f1=0.9529
```

That is not "action conditioning does not help": it is "the arm could not move".
Those runs are kept under `experiments/ba24/wm_zeroinit/` as a measurement of the
flawed configuration, and no conclusion is drawn from them.

Two changes fix it, and `tests/test_action_message_passing.py`'s
`TestTheVariantCanActuallyLearn` is the guard — a shape-and-forward test cannot
see this class of bug:

1. the zero moves to a **ReZero-style scalar gate**, so `dL/dgate` is nonzero at
   step 0 (the MLP is normally initialised, so its output is not) and gradient
   reaches everything as soon as the gate opens;
2. `world_model.train_wm.parameter_groups` puts every modulator / ActionEncoder
   parameter in a **no-weight-decay** group, on the same reasoning that
   conventionally exempts biases and normalisation gains.

Verified on real data: after 60 optimiser steps the gates read 0.006-0.044 (IC)
and 0.014-0.044 (LT), and the action encoder carries real weight.

### Plumbing

* `GraphInput.batch_index` (new, optional, defaults to `None`) — a block-diagonal
  batch is many graphs, and `z_a` is pooled per graph. `collate_transitions` and
  `WorldModelEnvironment._block_graph_input` fill it; every single-graph call
  site leaves it `None`, which correctly reads as one graph.
* `ModelSpec.action_conditioning` (new, defaults to `none`) — a variant
  checkpoint carries extra tensors, so loading one into a baseline raises rather
  than quietly dropping the conditioning.

## 2b. What the arms measured

Three seeds, `--head linear` on IC — the setting where the GNN IS the transition
model, chosen because the structured head leaves almost nothing to learn (its
per-arc `q` comes out within 0.0075 of the true `w`, so all four arms are
identical there to four decimals). `+/-` is the SE over seeds.

| metric                  |    none    | message_blind |   global   |  message   |
| ----------------------- | ---------: | ------------: | ---------: | ---------: |
| delta_f1                | 0.8756 +/-0.0003 | 0.8754 +/-0.0009 | 0.8732 +/-0.0015 | 0.8752 +/-0.0002 |
| ens_marg_mae            | 0.0981 +/-0.0032 | 0.0935 +/-0.0009 | 0.0935 +/-0.0010 | 0.0942 +/-0.0012 |
| ens_count_w1            | 3.582 +/-0.688 | 2.485 +/-0.140 | 2.435 +/-0.007 | 2.676 +/-0.085 |
| abs(count_bias)         | 2.692 +/-1.228 | 0.917 +/-0.438 | 0.916 +/-0.149 | 1.549 +/-0.291 |
| final_count_gap         | 3.272 +/-1.371 | 1.085 +/-0.533 | 1.255 +/-0.314 | 1.826 +/-0.241 |
| effect_pearson          | 0.9120 +/-0.0004 | 0.9126 +/-0.0001 | 0.9122 +/-0.0004 | 0.9125 +/-0.0001 |
| **effect_sign_agree**   | 0.5949 +/-0.0039 | 0.5987 +/-0.0037 | **0.6333 +/-0.0038** | **0.6273 +/-0.0028** |
| action_sensitivity      | 1.931 +/-0.030 | 1.915 +/-0.003 | 1.971 +/-0.017 | 1.967 +/-0.017 |
| plan_regret_model       | 0.1958 +/-0.0448 | 0.2007 +/-0.0400 | 0.1958 +/-0.0378 | 0.1958 +/-0.0378 |

Clearing two pooled SE against `none`: **`effect_sign_agree` for `global` and
`message`, and nothing else, for anything.**

Two readings, and the capacity control is what separates them:

**The rollout improvement is CAPACITY, not action.** Marginal error, count
distance, saturation bias and endpoint gap all improve substantially over
`none` — and `message_blind`, which has the identical MLP with its action inputs
held at zero, is as good as or better than both action arms on every one of
them. The extra edge MLP stabilises a free head's free-running rollout; the
action content contributes nothing to that.

**The counterfactual DIRECTION improvement is action, not capacity.** On
`effect_sign_agree` — does the model get the SIGN of the true effect of swapping
one action for another right, per node — the two arms that see the action inside
the message clear two pooled SE, and the capacity control does not
(0.5987 +/-0.0037, indistinguishable from `none`). Against a chance floor of 0.5,
the baseline's edge is 0.095 and the action-conditioned arms' is 0.133: about
40% more. `effect_pearson` (magnitude) does not move; only the direction does.

**Locality does not pay.** `global` (z_a only) matches or beats `message`
(z_a + per-edge r_uv) on every column, including the one that separates. Ablation
D is enough; the extra per-edge relevance vector of ablation C buys nothing here.

Caveat worth its weight: three seeds. A two-pooled-SE test on n=3 is weak, and
`effect_sign_agree` is the only column that passes it. What makes it worth
reporting anyway is that the CONTROL behaves correctly — a capacity artefact
would have lifted `message_blind` too, and it did not.

### Replicated in a second headroom setting

`--hide-edge-weights` on IC, 3 seeds: the structured head keeps its form but `q`
must be inferred from structure rather than read off `w`, so the encoder has real
work to do for a different reason than the linear head does. Same metric, same
verdict, and the effect is much LARGER:

| effect_sign_agree | none | message_blind | global | message |
| ----------------- | ---: | ------------: | -----: | ------: |
| IC, w hidden      | 0.4480 +/-0.0626 | 0.5139 +/-0.0557 | **0.6268 +/-0.0036** * | **0.6282 +/-0.0039** * |
| IC, linear head   | 0.5949 +/-0.0039 | 0.5987 +/-0.0037 | **0.6333 +/-0.0038** * | **0.6273 +/-0.0028** * |

`*` = clears two pooled SE against `none`. In BOTH settings the capacity control
fails to separate and both action arms do, which is the pattern that makes this
an action effect rather than a parameter-count effect.

Two things sharpen it:

* **The w-hidden baseline is BELOW chance** (0.4480 against a 0.5 floor). Denied
  the true transmission probability, the state-only model gets the direction of
  an intervention's effect wrong more often than a coin flip, and action
  conditioning repairs it to 0.63 — a +0.18 move, four times the linear head's
  +0.04. `action_sensitivity` separates here too (1.768 -> 1.902).
* **Both settings converge to the same place.** The baselines start 0.15 apart
  (0.448 vs 0.595) and the action-conditioned arms land within 0.007 of each
  other (0.627-0.633) in both. That is what a real mechanism looks like: the
  ceiling is a property of the action information, not of the setting.

`global` and `message` remain indistinguishable (0.6268 vs 0.6282), so the
locality conclusion replicates as well: ablation D suffices.

## 3. The feedback interface

`coding_agent/diagnostics.py` — probes over one `(evaluator, graph, task)`:

| probe                     | meaning                                      | evaluator calls          |
| ------------------------- | -------------------------------------------- | ------------------------ |
| `evaluate(plan)`          | `V(S)` + per-node `p_v`                      | 1                        |
| `probe_drop(plan, v)`     | `V(S) - V(S \ {v})`, paired                  | 2 (1 if `V(S)` is held)  |
| `probe_swap(plan, a, b)`  | `V(S - a + b) - V(S)`, paired                | 2                        |
| `probe_region(plan, R)`   | `C(R\|S) = sum_{v in R} p_v`                 | 0 after `evaluate`       |
| `summarize_regions(plan)` | the same over a whole partition              | 0 after `evaluate`       |
| `redundancy(plan)`        | pairwise seed overlap                        | 1 per seed (cached)      |
| `bridge_report(plan)`     | `p_v` on inter-community bridge nodes        | 0 after `evaluate`       |
| `stagnation(history)`     | `max_j V_j - V_{t-w} < eps`                  | 0                        |
| `explain_plan(plan)`      | all of the above as prompt text              | sum of the blocks asked  |

Three rules the module is built around:

1. **No trusted-simulator information enters through here.** Everything is the
   graph, the bound evaluator, or arithmetic on its output. Note that the
   *existing* `methods.base.summarize()` does read `graph.ic_probs` (the true
   per-edge probabilities) through its reverse-reachable residual gains; the new
   diagnostics deliberately do not reuse that estimate.
2. **Every query is counted as what it is.** `ProbeCost` reads the bound
   environment's own meters, so the same probe on a `@monte_carlo` arm reports
   `simulator_episodes` instead of `wm_rollouts`. A test asserts this.
3. **Paired by default.** A drop is a difference; both terms roll at one seed,
   which on `WorldModelEnvironment` is the same per-step uniform draw matrix.
   Measured on the trained SBM checkpoint, re-estimating one seed's leave-one-out
   contribution under 24 ensemble seeds:

   | mode              | mean  |   sd | range          |
   | ----------------- | ----: | ---: | -------------- |
   | paired (default)  | +4.79 | 1.12 | [+2.70, +8.00] |
   | striped/batched   | +4.61 | 3.27 | [−2.00, +9.75] |

   Same quantity, 2.9x the spread — and the striped estimate changes SIGN, so it
   would tell a reviser to keep a seed the paired one says is worth +2.7 nodes.

Regions (`coding_agent/regions.py`) are all named graph statistics:

| scheme       | partition by                                        | extra WM calls |
| ------------ | --------------------------------------------------- | -------------- |
| `community`  | label-propagation communities (the existing detector)| 0              |
| `structural` | inter-community incidence, betweenness, degree bands | 0              |
| `seed_basin` | which seed dominates a node's predicted activation   | 1 per seed     |

## 3b. Where provenance actually lives

`scripts/eval_provenance.py` builds matched pairs with the same post-intervention
state and different provenance:

    A: state = (I, F) with v in F,   action = []
    B: state = (I-v, F-v),           action = [add_node(v)]     ->  T_exo(A) = T_exo(B)

**One step: there is nothing to recover.** Over 105 matched pairs on six held-out
BA graphs, the simulator's post-action status dicts were identical in 100% of
cases, so the two share one transition kernel. The empirical divergence (0.0021)
sits on the Monte-Carlo noise floor of the estimator (0.0020) at 400 draws. The
hypothesis is structurally false at one step under both IC and LT: `(I, F)` after
`T_exo` is a sufficient statistic, and a model that predicts *different* next
states for A and B is inventing a distinction, not recovering one. Every arm's
spurious divergence is ~1e-4, and seed-to-seed variation exceeds arm-to-arm
variation, so no arm can be said to be cleaner than another.

**Multiple steps under LT: the channel is large, and unreachable.** LT thresholds
persist within an episode. A node that activated naturally has revealed
`theta_v <= f_v`; a seeded one has revealed nothing. That is unobservable while
the node stays active — so the probe removes it (LT `remove_node` returns it to
susceptible) and asks whether it crosses again. Over 522 constructed pairs:

| case at t                       | P(active again at t+2, after removal at t+1) |
| ------------------------------- | -------------------------------------------: |
| activated naturally             |                                    **1.000** |
| externally seeded               |                                    **0.383** |
| **provenance gap**              |                                   **+0.617** |

This is a real, large information channel, and **no arm of this experiment can
use it**. All four are Markov in `(s_t, a_t)`; the action that distinguishes the
two worlds fired at `t-1` and is not an input at `t+1`. Putting the action into
the message function does not help, because the problem is not *where* the action
enters — it is that the model has no memory. Closing this needs a recurrent or
history-conditioned state, or an explicit belief over the hidden thresholds; it
is a different change from the one tested here.

## 4. Feedback tiers

`coding_agent/feedback.py`, selected with `--feedback`:

| tier     | blocks                                          | extra WM rollouts / turn |
| -------- | ----------------------------------------------- | ------------------------ |
| `f0`     | scalar reward                                   | 0                        |
| `f1`     | + per-seed drop attribution                     | k                        |
| `f2`     | + regional coverage                             | k                        |
| `f3`     | + seed overlap, bridge coverage, stagnation     | 2k                       |
| `default` | the existing `summarize()` output (the default) | 0                       |

`default` is not a rung of this ladder: it already carries community reach and
adjacent-seed hints, so it is richer than `f2` in places, and it reads
`graph.ic_probs`, so it is not a clean "world-model-only" arm. It exists so
nothing that runs today changes.

## 4b. What the ladder measured

30 cells (6 held-out SBM-120 graphs x 5 repetitions), one trained world model,
`budget = 6`, 20 iterations, one trusted verification per run. The reviser is
SCRIPTED, not an LLM (`scripts/eval_feedback_tiers.py`), so this measures whether
each tier's feedback CONTAINS the signal a reviser needs — not what an agent
would do with it. Paired within cell; `+/-` is the SE of the paired difference.

| feedback   | trusted | vs f0        | vs f0_matched | W-L vs f0 | WM calls | trusted eps | iters | chars  |
| ---------- | ------: | -----------: | ------------: | --------: | -------: | ----------: | ----: | -----: |
| f0         |   32.28 |            - | -0.56 +/-0.24 |         - |       41 |         300 |    20 |    680 |
| f1         |   32.93 | +0.65 +/-0.29 | +0.09 +/-0.28 |     20-8 |      161 |         300 |    20 |  4,003 |
| f2         |   32.33 | +0.05 +/-0.29 | -0.51 +/-0.33 |    17-12 |      161 |         300 |    20 | 11,608 |
| f3         |   33.77 | +1.49 +/-0.25 | +0.93 +/-0.27 |     23-6 |      169 |         300 |    20 | 34,343 |
| f0_matched |   32.84 | +0.56 +/-0.24 |             - |     18-9 |      166 |         300 |    82 |  2,802 |

References on the same trusted simulator: degree 31.85, random 24.31. Trusted
episodes spent inside the diagnostics: **0**.

`f0_matched` is why this table is readable. It is f0 given as many iterations as
it takes to spend f3's world-model budget, and it alone gains +0.56 — so:

* **f1's win is mostly bought queries, not information.** Against the equal-budget
  control it is +0.09 +/- 0.28, i.e. nothing.
* **Only f3 clears the control**: +0.93 +/- 0.27 at the same budget, 23 of 29
  decided cells. Redundancy, bridge coverage and stagnation carry something extra
  search does not replace.
* **f2 is negative against the control** (-0.51 +/- 0.33). Regional coverage added
  to attribution, without the bridge and redundancy context that says WHY a
  region is uncovered, sends the reviser after weak communities the cascade
  cannot reach. More feedback is not monotonically better.

And the price: f3 costs **50x f0's context** (34.3k vs 680 characters, ~8.6k
tokens per 20-iteration run) for +0.93 nodes on a ~32-node spread.

**How far this generalises.** The `+/-` above is the SE over 30 cells, which
treats cells as independent — but five of them share each graph, so the honest
across-GRAPH statement is weaker than the cell-level SE suggests. Per graph, f3
beats the equal-budget control on **4 of 6**:

| graph | f3 - f0_matched |
| ----- | --------------: |
| s10   |           +1.47 |
| s11   |           -0.09 |
| s12   |           -0.33 |
| s13   |           +2.54 |
| s14   |           +1.19 |
| s15   |           +0.78 |

Two graphs go the other way, by small margins. Six graphs is thin, and one
graph family (SBM-120) with one trained world model and one scripted reviser is
thinner still. This is a directional result, not a benchmark.

## 5. Reproducing

```bash
PY=/opt/anaconda3/bin/python3   # 3.9 cannot parse this repo, and lacks ndlib

# data
$PY -m data.generate_wm_data --dataset ba --num-graphs 24 --syn-nodes 100 \
    --action-ops add_node remove_node --models IC LT \
    --algorithms random degree pagerank betweenness \
    --split-mode graph_disjoint --seed 42 --out-dir experiments/ba24/data

# Experiment 1: four arms x seeds, ordered by how much room the ENCODER has.
# With --head structured the transition is mostly hand-written (the learned part
# is one number per edge, and it comes out within 0.0075 of the true w), so that
# setting cannot discriminate the arms whatever they do. Run the others first.
#
#   lin  the linear head: the GNN IS the transition model
sed 's/--head structured/--head linear/' scripts/run_action_conditioning_sweep.sh > /tmp/linear.sh
chmod +x /tmp/linear.sh
TAG="lin" /tmp/linear.sh experiments/ba24/data experiments/ba24/wm "42 43 44" "IC"
#   w    structured, but q must be inferred from structure
EXTRA_FLAGS="--hide-edge-weights" TAG="w" \
  scripts/run_action_conditioning_sweep.sh experiments/ba24/data experiments/ba24/wm "42 43 44" "IC"
#   IC   structured with the true w — the ceiling setting, for continuity
scripts/run_action_conditioning_sweep.sh experiments/ba24/data experiments/ba24/wm "42 43 44" "IC"
$PY -m scripts.aggregate_action_conditioning --results experiments/ba24/wm \
    --out experiments/ba24/action_conditioning.json

# Experiment 2: intervention provenance
$PY -m scripts.eval_provenance --data-dir experiments/ba24/data --diffusion-model IC \
    --results experiments/ba24/wm/IC_none_s42.json experiments/ba24/wm/IC_message_s42.json \
    --out experiments/ba24/provenance_IC.json

# Experiment 3: the feedback ladder (scripted reviser)
$PY -m scripts.eval_feedback_tiers --data-dir experiments/sbm24/data \
    --wm-results experiments/sbm24/wm/sbm_IC.json \
    --graphs 6 --seeds 0 1 2 --iterations 20 --budget 6 \
    --out experiments/feedback/sbm.json

# Experiment 3 with the real coding agent (needs GATEWAY_BASE_URL + a token)
$PY -m coding_agent.run --method evolve --evaluator world_model --feedback f2 ...
```

`scripts/run_action_conditioning_sweep.sh` runs the four arms of a group
concurrently. Copy it before launching if you intend to edit it: bash reads a
script incrementally and editing a running one corrupts the run.
