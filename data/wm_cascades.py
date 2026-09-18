"""
Replay real logged cascades as `(s_t, NULL, s_{t+1})` transitions.

The data stage for `--task cascade_prediction`, and the only one in this repo that
runs no simulator at all. `research/cascade_prediction.md` §9.4 is what it exists
for: `research_notes/Baselines - Oracle vs MC vs GWM.md` §4 Q2 answers the
chicken-and-egg objection by ASSERTING that in deployment the world model's training
data comes from historical observations rather than a simulator, and Weibo/APS/
Twitter are that assertion made downloadable. This module is where the assertion
becomes a file on disk.

Four things it does that `generate_wm_data.py` does not, each forced rather than
chosen:

  * **The action is NULL at every step.** §2.1: nothing intervenes, the cascade is
    only watched, so `T_exo` is the identity and the factorization collapses to
    `s_{t+1} = T_endo(s_t)`. The `t = 0` record still carries the root as an
    `add_node` bag, because that is how every reader in this repo recovers an
    episode's sources, but there is no injection at any later step and no
    counterfactual fork, because a log has nothing to fork on.

  * **The targets are HARD.** §2.4 calls this the main technical risk of the whole
    exercise and it is not a config flag: our soft targets are `P(infected)`
    averaged over `--mc-marginals` re-runs, and a real cascade **happened once**.
    So `next_marginal_infected` is the realized 0/1 indicator. §11 records that
    how much that costs our one-step `delta_f1` is unestablished by anything in
    the literature: they never had soft targets to lose.

  * **The split is CHRONOLOGICAL.** §8.3 is the most transferable finding in that
    file: the field's standard 70/15/15-random-over-cascades split LEAKS, because
    cascades overlap in wall-clock time and a training cascade's prediction window
    can sit inside a test cascade's observation window. Under CasTemp's leak-free
    fix CasFlow and CasDO fall below a plain MLP and the whole field's APS band
    moves from 1.19-2.11 to 2.28-4.82. `--cp-split random` reproduces the leaky
    protocol on purpose, so the effect is measurable in our own harness rather than
    cited.

  * **Elapsed time is BINNED into timesteps.** The corpora publish seconds (or days,
    for APS) since publication; our simulator is discrete-time. `--cp-step` is the
    bin width, and §6.5 notes the happy case: APS's 3-year and 5-year windows are
    long enough that one step can be a whole year with no resampling at all.

Everything else is byte-identical to what the simulator writes, deliberately:
`world_model/wm_data.py` reads these transitions with no branch, the world model
trains on them with no branch, and the only thing that says they are not simulated
is the `observed` block this module adds to `metadata.json`.
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from tqdm import tqdm

from data.datasets.cascade_common import (
    Cascade,
    cascade_arrays,
    default_min_observed,
    default_truncate,
    filter_cascades,
    relabel_cascades,
)
from data.wm_graphs import GraphBundle, corpus_time_unit, load_cascade_corpus
from data.wm_simulator import ActionOp, State

# How the corpus is cut into train / val / test.
#
# `chronological` is CasTemp's leak-free protocol generalized to arbitrary windows
# (§8.3): partition by PUBLICATION TIME into three contiguous bins of equal
# duration, and drop any cascade whose prediction window runs past its own bin. The
# second half is what makes it leak-free rather than merely ordered, without it a
# late training cascade still predicts over a span a test cascade is observed in.
#
# `random` is the field's own 70/15/15 over cascades (CasFlow §5.1, CasFT
# "Datasets and Preprocessing"). It is here to be MEASURED against the first, not
# because it is defensible.
chronological_split = "chronological"
random_split = "random"
valid_splits = (chronological_split, random_split)

# Which quantity a predictor is scored on. §5.7 difference 3: CasFlow's own code
# regresses `label = P(t_p) - P(t_o)` and CasFT's Eq. 26 regresses `P`. Same symbol,
# different quantity, printed in one table in §5.1.
increment_target = "increment"
total_target = "total"
valid_targets = (increment_target, total_target)

# The field's own split proportions for `--cp-split random`
random_proportions = (0.7, 0.15, 0.15)

# `--cp-graph`: which graph the transitions run over.
#
# `native` is whatever the CORPUS's own loader returns, and it is the default
# because the loader is where that judgement belongs: `digg_cascades` returns the
# real published FRIENDSHIP network (the object every method in §7 runs on) while
# every other corpus returns the union of its observed propagation paths, because
# that is all it has.
#
# `paths` FORCES the path union everywhere. On Digg that is a real second
# experiment rather than a formatting choice: the vote log records no parent, so
# the union is a union of STARS, and the difference between the two rows is what
# the social graph is worth on a corpus whose diffusion tree is unobserved. On
# every other corpus the two are identical and the flag is a no-op, which the log
# says out loud rather than leaving to be inferred.
native_graph = "native"
paths_graph = "paths"
valid_graphs = (native_graph, paths_graph)


# Bins of the OBSERVATION window a replayed cascade gets. Four is the smallest
# number that supports the feature this literature says dominates: Cheng et al.'s
# single best predictor is the adoption rate in the SECOND HALF of the window, which
# needs at least two bins per half to be a rate rather than a count.
default_observed_steps = 4


def corpus_defaults(dataset: str) -> tuple[int, int]:
    """
    `(observation window, prediction horizon)` a corpus publishes, in its own units.

    Read off the loader module rather than hardcoded here, because the windows are a
    property of the CORPUS and §8.2 tabulates them per corpus: Weibo 0.5 h / 1 h
    predicted to 24 h, Twitter 1 d / 2 d to 32 d, APS 3 y / 5 y to 20 y. The first of
    each pair is the default; §8.2 pairs two deliberately, so a sweep runs both.
    """
    import importlib

    module = importlib.import_module(f"data.datasets.{dataset}")
    windows = getattr(module, "observation_windows", None)
    horizon = getattr(module, "prediction_horizon", None)

    if not windows or horizon is None:
        raise ValueError(
            f"the loader for {dataset!r} publishes no observation_windows / "
            f"prediction_horizon, so --cp-observation and --cp-horizon have no "
            f"default and must be given explicitly. research/cascade_prediction.md "
            f"§8.2 has the standard settings per corpus."
        )

    return int(windows[0]), int(horizon)


def resolve_step(observation: int, step: int) -> int:
    """
    Corpus time units per replayed timestep.

    Derived from the OBSERVATION window rather than the horizon, and that choice is
    the whole reason it is a function: these corpora have windows that are a tiny
    fraction of their horizons (Weibo observes 0.5 h of 24 h), so binning the
    horizon uniformly leaves the prefix with zero or one step and deletes the wave
    series a predictor is supposed to read. Binning the WINDOW instead keeps the
    prefix informative and lets the tail be as long as it needs; a replayed episode
    stops as soon as its frontier empties, so a long horizon costs nothing on a
    cascade that is over.
    """
    return step if step > 0 else max(1, observation // default_observed_steps)


@dataclass()
class ReplayConfig:
    """Everything that decides what a replayed corpus contains."""

    dataset: str
    out_dir: str
    # Both in the CORPUS's own time unit (seconds for the social corpora, days for
    # APS). §8.2 pairs two observation windows per corpus deliberately: "a
    # single-window result is not publishable in this literature", so a sweep runs
    # this twice rather than averaging.
    observation: int
    horizon: int
    # Corpus time units per replayed timestep. The whole window must fit inside
    # `gen_horizon` steps or the prediction target is unreachable, which is checked
    # rather than truncated.
    step: int
    gen_horizon: int = 10
    min_observed: int = default_min_observed
    truncate: int = default_truncate
    split: str = chronological_split
    target: str = increment_target
    graph: str = native_graph
    max_cascades: int = 0
    max_nodes: int = 0
    prob_model: str = "weighted"
    uniform_p: float = 0.1
    seed: int = 42
    diffusion_models: tuple = ("IC",)


def bin_events(
    cascade: Cascade, step: int, horizon: int
) -> tuple[list[list[int]], list[int]]:
    """
    A cascade's events as per-timestep adopter waves, plus each wave's parents.

    Index `t` holds the nodes whose elapsed time falls in `[t * step, (t+1) * step)`,
    capped at `horizon` steps. The root is forced into wave 0 whatever its recorded
    elapsed time, because a cascade with no adopter at `t = 0` has no seed commit
    and `load_episode_endpoints` would skip the episode entirely.
    """
    waves = [[] for _ in range(horizon + 1)]
    parents = {}
    placed = set()

    for adopter, elapsed, parent in cascade.events:
        index = min(int(elapsed) // step, horizon)

        if adopter == cascade.root:
            index = 0

        if adopter in placed:
            continue

        placed.add(adopter)
        waves[index].append(int(adopter))
        parents[int(adopter)] = None if parent is None else int(parent)

    if cascade.root not in placed:
        waves[0].insert(0, int(cascade.root))
        parents[int(cascade.root)] = None

    return waves, parents


def replay_records(
    cascade: Cascade,
    bundle: GraphBundle,
    config: ReplayConfig,
    diffusion_model: str,
    split: str,
) -> list[dict]:
    """
    One logged cascade as a list of transition records in the shipped JSONL shape.

    The `t = 0` action is the root's `add_node` bag and every later action is EMPTY,
    which is §2.1's `a_t = NULL` written down. `next_marginal_infected` carries the
    realized indicator rather than an MC average, and `parents` carries the corpus's
    own transmission edge where it logs one: free here, because two of these
    corpora (Taoke, the AMiner Weibo release) record it and the field's tree metrics
    are then computable without a second pass.
    """
    from data.generate_wm_data import build_record

    steps = max(1, config.horizon // config.step)
    waves, parents = bin_events(cascade, config.step, min(steps, config.gen_horizon))

    episode_id = f"{bundle.graph_id}|{diffusion_model}|observed|{cascade.cascade_id}"
    records = []

    infected = []
    frontier = []

    for timestep, wave in enumerate(waves):
        state = State(infected=list(infected), frontier=list(frontier))
        action = (
            [ActionOp("add_node", node) for node in wave] if timestep == 0 else []
        )

        # At t = 0 the wave IS the action's effect, so `next_state` holds the seeded
        # nodes; afterwards it holds whatever the log says adopted next. Either way
        # the pair is a real observed transition and nothing was simulated.
        gained = [node for node in wave if node not in infected]
        next_infected = sorted(set(infected) | set(gained))
        next_state = State(infected=next_infected, frontier=sorted(gained))

        records.append(
            build_record(
                graph_id=bundle.graph_id,
                diffusion_model=diffusion_model,
                episode_id=episode_id,
                algorithm="observed",
                branch="main",
                t=timestep,
                state=state,
                action=action,
                next_state=next_state,
                reward=float(len(next_infected) - len(infected)),
                # HARD 0/1: a real cascade happened once, so there is nothing to
                # average (§2.4). Written as a marginal anyway so `build_features`
                # takes the identical path it takes on simulated data.
                next_marginal_infected={node: 1.0 for node in next_infected},
                next_marginal_frontier={node: 1.0 for node in gained},
                parents={
                    node: ([] if parents.get(node) is None else [parents[node]])
                    for node in gained
                },
            )
        )

        infected, frontier = next_infected, sorted(gained)

        # A log that stops has stopped, but "stopped" here means NOTHING LATER,
        # not "nothing this step". The simulator's own break fires on an empty
        # frontier because that is a fixed point of a monotone cascade; a REPLAYED
        # log is not a Markov process and routinely goes quiet for a bin and
        # resumes, so breaking on the first gap would silently truncate the
        # cascade at its first lull and under-report every popularity after it.
        if timestep > 0 and not any(waves[timestep + 1 :]):
            break

    return records


def assign_splits(cascades: list[Cascade], config: ReplayConfig) -> dict[str, str]:
    """
    `{cascade_id: split}` under the configured protocol.

    `chronological` sorts by PUBLICATION TIME and cuts at the field's own 70/15/15
    proportions, then DROPS (maps to the empty string) any cascade whose prediction
    window `[publish, publish + t_p]` runs past its own bin's closing time. That
    second rule is what makes it leak-free rather than merely ordered, and it is the
    whole content of §8.3: without it a late training cascade still predicts over a
    span the next bin's cascades are being observed in, which is the global temporal
    shortcut CasTemp showed two SOTA models were exploiting.

    **The proportions are held at the field's own rather than at CasTemp's equal
    DURATION, and that is a deliberate deviation.** CasTemp cuts into contiguous bins
    of equal wall-clock length, which is faithful but leaves the pool sizes at the
    mercy of how a corpus is distributed in time: Taoke publishes most of its items
    in the first hours, so equal-duration bins put 98% of it in train and leave two
    cascades in test. Leak-freeness depends only on the bins being CONTIGUOUS in time
    and on the drop rule, not on where the boundaries fall, so cutting at fixed
    proportions preserves the property and holds the pool sizes fixed. That also
    makes `chronological` versus `random` a clean A/B on ORDER alone, which is the
    variable §8.3 is actually about; changing the sizes too would confound it.

    `random` is the field's own 70/15/15 shuffled over cascades (CasFlow §5.1,
    CasFT "Datasets and Preprocessing"), seeded so a rerun reproduces it. It is here
    to be MEASURED against the first, not because it is defensible.
    """
    if config.split not in valid_splits:
        raise ValueError(
            f"unknown --cp-split {config.split!r}; choose one of {valid_splits}"
        )

    if not cascades:
        return {}

    count = len(cascades)
    train_end = int(count * random_proportions[0])
    val_end = train_end + int(count * random_proportions[1])

    if config.split == random_split:
        rng = np.random.default_rng(config.seed)
        order = [int(index) for index in rng.permutation(count)]
    else:
        # Ties broken by cascade id so the order is total and a rerun reproduces it
        order = sorted(
            range(count),
            key=lambda index: (cascades[index].publish_time, cascades[index].cascade_id),
        )

    assignment = {}
    first_test = None

    for rank, index in enumerate(order):
        split = "train" if rank < train_end else "val" if rank < val_end else "test"
        assignment[cascades[index].cascade_id] = split

        if split == "test" and first_test is None:
            first_test = cascades[index].publish_time

    if config.split == random_split:
        return assignment

    # The leak §8.3 names, stated exactly: a TRAINING cascade's prediction window
    # `[p, p + t_p]` must close before the first TEST cascade is observed, or the
    # model can learn "there was a burst around time T": a global temporal shortcut
    # unavailable at deployment. Enforced train-to-test rather than bin-to-bin
    # because VAL sits between them and is a buffer by construction: tightening it to
    # every bin would drop the val pool's own tail for no leak it prevents.
    dropped = 0

    if first_test is not None:
        for cascade in cascades:
            if assignment.get(cascade.cascade_id) != "train":
                continue

            if cascade.publish_time + config.horizon > first_test:
                dropped += 1
                assignment[cascade.cascade_id] = ""

    if dropped:
        print(
            f"[replay] chronological split dropped {dropped}/{train_end} training "
            f"cascades whose prediction window reached past the first test "
            f"observation: that drop IS the leak-free protocol (§8.3), not a loss"
        )

    return assignment


def replay_corpus(config: ReplayConfig) -> dict:
    """
    Load a corpus, replay it into `transitions_<dm>_<split>.jsonl`, write metadata.

    Returns the same metadata dict `run_generation` does, so `pipeline.run`'s data
    stage consumes either one unchanged.
    """
    from data.generate_wm_data import GraphStore, TransitionWriter

    start = time.perf_counter()
    out_dir = Path(config.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if config.step <= 0:
        raise ValueError(f"--cp-step must be positive, got {config.step}")

    if config.horizon <= config.observation:
        raise ValueError(
            f"--cp-horizon ({config.horizon}) must exceed --cp-observation "
            f"({config.observation}); a prediction window of zero has nothing to "
            f"predict"
        )

    unit = corpus_time_unit.get(config.dataset, "second")
    print(
        f"[replay] {config.dataset}: observation={config.observation} {unit}(s), "
        f"horizon={config.horizon} {unit}(s), step={config.step} {unit}(s) "
        f"-> {config.observation // config.step} observed of "
        f"{config.horizon // config.step} timesteps"
    )

    if config.horizon // config.step > config.gen_horizon:
        raise ValueError(
            f"the prediction horizon is {config.horizon // config.step} timesteps at "
            f"--cp-step {config.step}, but --gen-horizon is {config.gen_horizon}, so "
            f"the replay would stop before the quantity being predicted exists. "
            f"Raise --gen-horizon or --cp-step."
        )

    bundle, cascades = load_cascade_corpus(
        config.dataset, prob_model=config.prob_model, uniform_p=config.uniform_p
    )

    total = len(cascades)
    cascades = filter_cascades(
        cascades, config.observation, config.min_observed, config.truncate
    )
    print(
        f"[replay] filters kept {len(cascades)}/{total} cascades "
        f"(>= {config.min_observed} participants inside the observation window, "
        f"first {config.truncate} observed kept): both are §8.4 landmines and both are in "
        f"metadata.json"
    )

    if config.max_cascades and len(cascades) > config.max_cascades:
        rng = np.random.default_rng(config.seed)
        chosen = sorted(
            rng.choice(len(cascades), size=config.max_cascades, replace=False)
        )
        cascades = [cascades[int(index)] for index in chosen]
        print(f"[replay] subsampled to {len(cascades)} cascades (--cp-max-cascades)")

    # CoupledGNN's own move, and what makes a 6.7M-node corpus runnable at all
    # (§6.5): keep the busiest participants and drop whatever is left with fewer
    # than two events. The bundle is renumbered onto the surviving ids, so the
    # GRAPH and the cascades cannot disagree about what a node is, which is why
    # this happens here rather than being pushed onto the loader.
    if config.max_nodes:
        before = len(cascades)
        cascades, mapping = relabel_cascades(cascades, config.max_nodes)
        print(
            f"[replay] reduced to the {len(mapping)} busiest participants "
            f"(--cp-max-nodes {config.max_nodes}), keeping {len(cascades)}/{before} "
            f"cascades: §6.4: a corpus reduced this way is a NEW version of its "
            f"own name, and its numbers are comparable only to themselves"
        )

        # The cap deletes participants, so the size filter above no longer holds on
        # what is replayed: casflow_aps at 30,000 nodes shipped test cascades with
        # 2-4 observed adopters under a metadata line claiming ">= 10", and every
        # growth-predicting baseline scored an order of magnitude worse on the
        # selection split than on the held-out one. Re-apply it so the protocol
        # the metadata states is the protocol the numbers were measured under.
        surviving = len(cascades)
        cascades = filter_cascades(
            cascades, config.observation, config.min_observed, config.truncate
        )
        print(
            f"[replay] size filter re-applied after the node cap: "
            f"{len(cascades)}/{surviving} cascades still have >= "
            f"{config.min_observed} observed participants"
        )

    if not cascades:
        raise ValueError(
            f"no cascade survived the filters on {config.dataset}. Lower "
            f"--cp-min-size (currently {config.min_observed}) or widen "
            f"--cp-observation (currently {config.observation} {unit}(s))."
        )

    if config.graph not in valid_graphs:
        raise ValueError(
            f"unknown --cp-graph {config.graph!r}; choose one of {valid_graphs}"
        )

    if config.graph == paths_graph:
        from data.wm_graphs import make_real_bundle_from_arrays

        adjacency, node_feats, node_labels, _ = cascade_arrays(
            cascades, bundle.nx_graph.number_of_nodes()
        )
        before = bundle.nx_graph.number_of_edges()
        bundle = make_real_bundle_from_arrays(
            config.dataset,
            adjacency,
            node_feats,
            node_labels,
            prob_model=config.prob_model,
            uniform_p=config.uniform_p,
        )
        after = bundle.nx_graph.number_of_edges()
        print(
            f"[replay] --cp-graph paths: rebuilt from the observed propagation "
            f"ties ({before} -> {after} undirected edges)"
            + (
                ": IDENTICAL to the loader's own graph on this corpus, which "
                "publishes no separate network"
                if before == after
                else ""
            )
        )

    if config.max_nodes:
        # The bundle still describes the FULL graph; rebuild it over the reduced
        # ids from the surviving propagation paths, or every record would index
        # into a node space its own graph no longer has
        from data.wm_graphs import make_real_bundle_from_arrays

        reduced = max(
            (node for cascade in cascades for node in cascade.participants()),
            default=0,
        ) + 1
        adjacency, node_feats, node_labels, _ = cascade_arrays(cascades, reduced)
        bundle = make_real_bundle_from_arrays(
            config.dataset,
            adjacency,
            node_feats,
            node_labels,
            prob_model=config.prob_model,
            uniform_p=config.uniform_p,
        )
        print(
            f"[replay] rebuilt graph on the reduced ids: {reduced} nodes, "
            f"{adjacency.nnz // 2} undirected edges"
        )

    assignment = assign_splits(cascades, config)
    graph_store = GraphStore(out_dir)
    graph_store.save(bundle)
    graph_store.flush()

    counts = {"train": 0, "val": 0, "test": 0}
    episodes = 0

    with TransitionWriter(out_dir) as writer:
        for cascade in tqdm(cascades, desc="cascades"):
            split = assignment.get(cascade.cascade_id, "")
            if not split:
                continue

            for diffusion_model in config.diffusion_models:
                for record in replay_records(
                    cascade, bundle, config, diffusion_model, split
                ):
                    writer.write(record, model=diffusion_model, split=split)

            counts[split] += 1
            episodes += 1

    if not counts["train"]:
        # The corpus is too SHORT for this horizon, and saying so with the number
        # that would work beats leaving the caller to bisect. A leak-free split needs
        # three contiguous publication bins each at least `t_p` wide, so the corpus's
        # own span divided by three is the largest horizon it can support at all.
        times = np.array(sorted(cascade.publish_time for cascade in cascades))
        times = times - times[0]
        # The largest horizon that leaves any training cascade at all: the gap
        # between the first test publication and the first training one
        largest = int(np.percentile(times, 100 * sum(random_proportions[:2])))

        # `t_p` must also EXCEED the observation window or there is nothing left to
        # predict, and that floor is checked elsewhere. When it sits above the gap,
        # no horizon satisfies both and recommending one would send the caller into
        # the other validator. Say so instead: on this corpus, at this observation
        # window, the leak-free protocol does not exist.
        impossible = largest <= config.observation

        remedy = (
            f"  No --cp-horizon can work here: a leak-free split needs "
            f"--cp-horizon <= {max(largest, 1)}, but --cp-horizon must also exceed "
            f"--cp-observation ({config.observation}) or the prediction window is "
            f"empty.\n"
            f"  The knob that usually opens the gap is --cp-min-size (currently "
            f"{config.min_observed}): it decides WHICH cascades survive the filter, "
            f"and the ones with many early participants are the most concentrated "
            f"in time, so a lower value widens the publication span the split gets "
            f"to work with. Lowering --cp-observation below {max(largest, 1)} "
            f"{unit}(s) also works. Both change the pool, so this recommended "
            f"bound is recomputed on the pool you end up with rather than this one."
            if impossible
            else f"  --cp-horizon {config.horizon} does not fit in that gap. Either "
            f"use --cp-horizon <= {max(largest, 1)} (which must still exceed "
            f"--cp-observation {config.observation}), or use --cp-split random."
        )

        raise ValueError(
            f"the {config.split} split left no TRAINING cascades: every training "
            f"cascade's prediction window reaches past the first test observation.\n"
            f"  publication times span {int(times[-1])} {unit}(s), but the corpus is "
            f"FRONT-LOADED: 85% of its cascades appear within {largest} {unit}(s) "
            f"of the first (median {int(np.median(times))}).\n"
            f"{remedy}\n"
            f"  --cp-split random reproduces the field's own leaky protocol and is "
            f"the other half of the A/B research/cascade_prediction.md §8.3 "
            f"describes, but it is a LAST resort here rather than the only option: "
            f"the leak-free split is what that section argues for, so widen the "
            f"pool first and fall back to random only if nothing opens the gap."
        )

    sizes = np.array([cascade.size for cascade in cascades], dtype=np.float64)
    observed = np.array(
        [cascade.popularity_at(config.observation) for cascade in cascades],
        dtype=np.float64,
    )

    metadata = {
        "task": "cascade_prediction_observed_transitions",
        "config": config.__dict__,
        "n_episodes": episodes,
        "generation_seconds": round(time.perf_counter() - start, 1),
        "graphs": [
            {
                "graph_id": bundle.graph_id,
                "num_nodes": bundle.nx_graph.number_of_nodes(),
                "num_edges": bundle.nx_graph.number_of_edges(),
                "budget_k_min": 0,
                "budget_k_max": 0,
                "budget_pct_min": 0.0,
                "budget_pct_max": 0.0,
            }
        ],
        # The block that says these transitions were REPLAYED rather than simulated.
        # Every other reader in this repo treats `transitions_IC_train.jsonl` as an
        # NDlib rollout, and §2.2's whole argument is that fitting the IC head to
        # something else is the experiment, so the file has to say which it is.
        "observed": {
            "corpus": config.dataset,
            "time_unit": unit,
            "observation": config.observation,
            "horizon": config.horizon,
            "step": config.step,
            "observed_steps": config.observation // config.step,
            "horizon_steps": config.horizon // config.step,
            "split_protocol": config.split,
            "target": config.target,
            "graph_construction": config.graph,
            "min_observed": config.min_observed,
            "truncate": config.truncate,
            "n_cascades": len(cascades),
            "n_scored": episodes,
            "split_counts": counts,
            "avg_size": float(sizes.mean()),
            "median_size": float(np.median(sizes)),
            "avg_observed": float(observed.mean()),
            # §2.4: a real cascade happened ONCE, so `--mc-marginals` has nothing to
            # average and the training signal is genuinely different from every
            # other dataset in this tree
            "hard_targets": True,
        },
    }

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(
        f"[replay] {episodes} cascades replayed "
        f"(train {counts['train']} / val {counts['val']} / test {counts['test']}) "
        f"in {metadata['generation_seconds']}s -> {out_dir}"
    )

    return metadata


def dataset_is_observed(out_dir: Path) -> bool:
    """
    Whether this dataset was REPLAYED from a log, from its own metadata.

    Read rather than passed, for the same reason `dataset_is_competitive` is: a
    checkpoint trained on hard targets and one trained on MC marginals are not the
    same object, and nothing in a loss curve would say which produced a given
    number.
    """
    metadata_path = Path(out_dir) / "metadata.json"

    if not metadata_path.exists():
        return False

    return "observed" in json.loads(metadata_path.read_text())


def observed_protocol(out_dir: Path) -> dict:
    """The `observed` block, or a raise naming what generated the dataset instead."""
    metadata_path = Path(out_dir) / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    protocol = metadata.get("observed")

    if protocol is None:
        raise ValueError(
            f"the dataset at {out_dir} was SIMULATED, not replayed from a log "
            f"(task={metadata.get('task')!r}). Cascade prediction is defined on real "
            f"observed traces: research/cascade_prediction.md §2.2 is explicit that "
            f"running it on NDlib output closes exactly the loop the task exists to "
            f"break. Regenerate with --task cascade_prediction and a corpus dataset."
        )

    return protocol
