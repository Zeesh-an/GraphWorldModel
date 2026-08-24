"""
Q5 (Stage E) — IM -> Source Localization, frozen forward-dynamics reuse.

    python -m scripts.eval_source_localization \
        --results results/influence_maximization/ba100/wm_structured_IC/sage_IC.json \
        --data-dir results/sl/ba100_clean/data \
        --out results/sl/sl_transfer_IC.json

The question this answers is NOT the one Q5's IM -> Adaptive IM answers. There
the frozen checkpoint stayed a decision model: it scored interventions. Here it
is demoted to a **forward oracle called from inside an inversion**, and its
action-conditioning is never exercised — source-localization episodes carry no
mid-cascade interventions at all (`--inject-p 0`, verified: 0 injected steps in
the dataset used). `registry.task_families` classifies the pair FORWARD_DYNAMICS
for exactly that reason, and this script must not be reported as zero-shot
action-conditioned transfer.

What transfers is T_endo. What is new is the readout, and nothing else: the world
model is frozen, no optimizer exists in this file, and no target-task training
happens.

The inversion, stated plainly:

    given  G and an observed terminal infected set y, and a budget k
    find   the seed set x, |x| = k, whose forward rollout best explains y

implemented as greedy: grow x one node at a time, each step keeping the
candidate whose predicted terminal marginals best match y. Candidates are
restricted to y itself, because under IC a source is necessarily infected --
that is a hard constraint from the dynamics, not a heuristic.

Arms differ ONLY in what plays the forward model:

    world_model   the frozen IM checkpoint (ours)
    oracle        NDlib Monte-Carlo. The ceiling, and the cost being saved.
    degree        no forward model: top-k degree within y. The structural floor
                  that any forward model has to beat to have earned its place.
    random        k drawn from y.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from coding_agent.types import GraphInfo
from data.wm_simulator import ActionOp
from world_model.checkpoint import load_checkpoint
from world_model.wm_data import load_graph_store
from world_model.wm_eval import rebuild_simulator, seed_upper_bound

default_threads = 1


# An episode is only a source-localization problem if the cascade AMPLIFIED.
# Measured on the first attempt: with k = 5.5 seeds the observed set averaged 21.9
# nodes, so k/|y| = 0.334 -- a third of the observation WAS the answer, and
# picking at random from it scored 0.314. Every method then lands in 0.26-0.33 and
# the benchmark cannot discriminate. Requiring |y| >= amplification * k keeps the
# episodes where the source set is actually hidden in the observation.
min_amplification = 5.0


def load_episodes(data_dir: Path, diffusion_model: str, split: str = "test",
                  min_amplification: float = min_amplification):
    """
    (graph_id, true_sources, observed_terminal_infected) per episode.

    `research/source_localization.md` §2.1 is right that this data is free: the
    t=0 record has an empty state and the whole seed bag as `add_node` ops, and
    the last record carries the terminal state. Grouping by episode recovers
    (x, y) with no new generation.

    Episodes with any mid-cascade action are DROPPED rather than used: an
    injected intervention breaks the x -> y relationship the inversion assumes,
    and silently keeping them would make the task easier or harder in a way no
    metric would show.
    """
    path = Path(data_dir) / f"transitions_{diffusion_model}_{split}.jsonl"
    grouped = defaultdict(list)

    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["branch"] == "main":
            grouped[(record["graph_id"], record["episode_id"])].append(record)

    episodes, dropped = [], 0

    for (graph_id, episode_id), records in grouped.items():
        records.sort(key=lambda record: record["t"])

        if any(record["action"] for record in records[1:]):
            dropped += 1
            continue

        sources = sorted(
            int(op["target"]) for op in records[0]["action"] if op["op"] == "add_node"
        )
        observed = sorted(int(v) for v in records[-1]["next_state"]["infected"])

        if not sources or len(observed) < min_amplification * len(sources):
            dropped += 1
            continue

        episodes.append(
            {"graph_id": graph_id, "episode_id": episode_id,
             "sources": sources, "observed": observed,
             "amplification": len(observed) / len(sources)}
        )

    return episodes, dropped


# ---------------------------------------------------------------------------
# Forward models. Each returns predicted terminal infection probability per node.
# ---------------------------------------------------------------------------


@torch.inference_mode()
def world_model_forward(seeds, graph_info, store_entry, scorer, horizon, n_samples,
                        cost=None):
    """Frozen world model rolled out from `seeds`; returns per-node P(infected)."""
    plan = [[ActionOp("add_node", int(node)) for node in seeds]]
    from world_model.scorer import ScoringContext

    trajectory = scorer.rollout(
        graph_info, plan,
        ScoringContext(horizon=horizon, budget=len(seeds), n_samples=n_samples,
                       seed=0),
    )

    if cost is not None:
        cost["forward"] += 1

    marginals = trajectory.final_marginals

    if marginals is None:
        return np.zeros(store_entry["num_nodes"])

    return np.asarray(marginals, dtype=float)


def oracle_forward(seeds, graph_info, store_entry, diffusion_model, horizon,
                   mc_runs, remove_semantics, crn_seeds, cost=None):
    """NDlib Monte-Carlo from `seeds`, common random numbers across candidates."""
    num_nodes = store_entry["num_nodes"]
    counts = np.zeros(num_nodes)

    for draw_seed in crn_seeds:
        simulator = rebuild_simulator(
            store_entry, diffusion_model, seed=draw_seed,
            remove_semantics=remove_semantics,
        )
        state = simulator.advance(
            [ActionOp("add_node", int(node)) for node in seeds]
        )

        for _ in range(horizon):
            if not state.frontier:
                break
            state = simulator.advance([])

        counts[list(state.infected)] += 1

        if cost is not None:
            cost["calls"] += 1

    return counts / len(crn_seeds)


def explanation_score(predicted: np.ndarray, observed_mask: np.ndarray) -> float:
    """
    How well a predicted terminal marginal explains the observation.

    Mass the forward model puts INSIDE the observed set, minus the mass it puts
    outside it. A candidate is good when its cascade covers what was seen and
    little else.

    NOT log-likelihood, which is the obvious choice and is wrong here. A
    Monte-Carlo oracle at 16-32 draws quantises its marginals, so every observed
    node it happens not to reach lands at p = 0, gets clipped to 1e-6, and
    contributes log(1e-6) = -13.8. The score is then dominated by how many nodes
    a candidate MISSED ENTIRELY rather than by which candidate is the source, and
    the argmax tracks coverage instead of explanation. Measured: with that
    objective the oracle scored F1 0.000 -- below random -- which is a broken
    objective, not a broken oracle.

    This form is linear in the marginals, bounded, and has no pathology at zero,
    so a coarse estimator degrades gracefully instead of catastrophically.
    """
    return float(np.sum(predicted * observed_mask) - np.sum(predicted * (1 - observed_mask)))


def greedy_invert(forward, observed, num_nodes, budget, candidates, rng):
    """
    Grow the seed set one node at a time, keeping the best explanation.

    Candidates come from the OBSERVED infected set: under IC a source is
    necessarily infected, so this is a constraint the dynamics impose rather than
    a prior we chose. `candidates` caps the shortlist per step so the oracle and
    the world model face identical work.
    """
    observed_mask = np.zeros(num_nodes)
    observed_mask[observed] = 1.0
    pool = list(observed)

    if len(pool) > candidates:
        pool = [int(v) for v in rng.choice(pool, size=candidates, replace=False)]

    chosen: list[int] = []

    for _ in range(min(budget, len(pool))):
        best, best_score = None, -float("inf")

        for node in pool:
            if node in chosen:
                continue

            score = explanation_score(forward(chosen + [node]), observed_mask)

            if score > best_score:
                best_score, best = score, node

        if best is None:
            break

        chosen.append(best)

    return chosen


def degree_baseline(observed, store_entry, budget, **_):
    degrees = np.zeros(store_entry["num_nodes"])
    np.add.at(degrees, store_entry["edge_index"][0], 1)
    np.add.at(degrees, store_entry["edge_index"][1], 1)

    return sorted(observed, key=lambda node: -degrees[node])[:budget]


def score_recovery(predicted: list[int], true: list[int]) -> dict:
    predicted_set, true_set = set(predicted), set(true)
    hits = len(predicted_set & true_set)
    precision = hits / len(predicted_set) if predicted_set else 0.0
    recall = hits / len(true_set) if true_set else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "n_predicted": len(predicted_set),
        "n_true": len(true_set),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Q5 IM -> Source Localization")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--n-episodes", type=int, default=60)
    parser.add_argument("--candidates", type=int, default=25)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--oracle-mc", type=int, default=16)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threads", type=int, default=default_threads)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)

    config = json.loads(args.results.read_text())["config"]
    diffusion_model = config["diffusion_model"]
    device = torch.device(args.device)
    checkpoint = (
        Path(config["ckpt_dir"]) / f"wm_{config['model']}_{diffusion_model}.pt"
    )
    model, spec, train_meta = load_checkpoint(
        checkpoint, config=config, device=device, strict_spec=False
    )
    frozen_before = {k: v.clone() for k, v in model.state_dict().items()}

    from world_model.scorer import WorldModelScorer

    scorer = WorldModelScorer(model, spec, device=str(device))
    store = load_graph_store(args.data_dir)
    episodes, dropped = load_episodes(args.data_dir, diffusion_model)
    rng = np.random.default_rng(args.seed)

    if len(episodes) > args.n_episodes:
        episodes = [
            episodes[i]
            for i in rng.choice(len(episodes), size=args.n_episodes, replace=False)
        ]

    print(f"[sl] frozen {spec.backbone}/{spec.head} {diffusion_model} as a FORWARD "
          f"oracle for source localization")
    amplifications = [episode["amplification"] for episode in episodes]
    print(f"[sl] {len(episodes)} episodes ({dropped} dropped: injected actions, or "
          f"amplification below {min_amplification}x)")
    print(f"[sl] amplification |y|/k: mean {np.mean(amplifications):.1f}x, "
          f"random-baseline precision k/|y| ~ "
          f"{np.mean([1 / a for a in amplifications]):.3f}")

    arms = ("world_model", "oracle", "degree", "random")
    scores = {name: [] for name in arms}
    costs = {name: {"calls": 0, "forward": 0} for name in arms}
    started = time.perf_counter()

    for index, episode in enumerate(episodes):
        entry = store[episode["graph_id"]]
        graph_info = GraphInfo.from_store_entry(entry)
        budget = len(episode["sources"])
        observed = episode["observed"]
        num_nodes = entry["num_nodes"]
        episode_rng = np.random.default_rng(args.seed + index)
        crn = [
            int(episode_rng.integers(seed_upper_bound)) for _ in range(args.oracle_mc)
        ]

        predictions = {
            "world_model": greedy_invert(
                lambda seeds: world_model_forward(
                    seeds, graph_info, entry, scorer, args.horizon,
                    args.n_samples, costs["world_model"],
                ),
                observed, num_nodes, budget, args.candidates,
                np.random.default_rng(args.seed + index),
            ),
            "oracle": greedy_invert(
                lambda seeds: oracle_forward(
                    seeds, graph_info, entry, diffusion_model, args.horizon,
                    args.oracle_mc, spec.remove_semantics, crn, costs["oracle"],
                ),
                observed, num_nodes, budget, args.candidates,
                np.random.default_rng(args.seed + index),
            ),
            "degree": degree_baseline(observed, entry, budget),
            "random": [
                int(v)
                for v in np.random.default_rng(args.seed + index).choice(
                    observed, size=min(budget, len(observed)), replace=False
                )
            ],
        }

        for name in arms:
            scores[name].append(score_recovery(predictions[name], episode["sources"]))

        if index < 5 or index % 20 == 0:
            line = "  ".join(
                f"{name}={scores[name][-1]['f1']:.2f}" for name in arms
            )
            print(f"  {episode['episode_id'][:38]:38s} k={budget} |y|={len(observed)}"
                  f"  {line}")

    after = model.state_dict()
    frozen_ok = all(torch.equal(frozen_before[k], after[k]) for k in frozen_before)

    summary = {
        name: {
            metric: float(np.mean([s[metric] for s in scores[name]]))
            for metric in ("precision", "recall", "f1")
        }
        | {
            "f1_std": float(np.std([s["f1"] for s in scores[name]])),
            "n_episodes": len(scores[name]),
        }
        for name in arms
    }

    blob = {
        "question": "Q5_cross_task_transfer",
        "source_task": "influence_maximization",
        "target_task": "source_localization",
        "transfer_level": "FORWARD_DYNAMICS",
        "note": "the frozen model is a forward oracle inside an inversion; its "
                "action conditioning is NOT exercised (0 mid-cascade actions in "
                "the target data), so this is not zero-shot action-conditioned "
                "transfer",
        "freeze_world_model": True,
        "target_finetuning": False,
        "frozen_parameters_unchanged": frozen_ok,
        "reused_components": ["graph_backbone", "edge_propensity", "T_endo",
                              "T_exo_semantics"],
        "reinitialized_components": ["readout", "inversion_procedure"],
        "model": {"backbone": spec.backbone, "head": spec.head,
                  "diffusion_model": diffusion_model},
        "source": {"data_dir": train_meta.get("data_dir"),
                   "split_mode": train_meta.get("split_mode")},
        "protocol": {
            "data_dir": str(args.data_dir), "n_episodes": len(episodes),
            "episodes_dropped": dropped,
            "min_amplification": min_amplification,
            "mean_amplification": float(np.mean(amplifications)),
            "candidates": args.candidates, "horizon": args.horizon,
            "oracle_mc": args.oracle_mc, "seed": args.seed,
        },
        "cost": costs,
        "summary": summary,
        "per_episode": {name: scores[name] for name in arms},
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(blob, indent=2, default=str))

    print(f"\n[sl] frozen parameters unchanged: {frozen_ok}")
    print(f"{'arm':14s} {'precision':>10s} {'recall':>8s} {'F1':>8s} {'F1 sd':>8s} "
          f"{'trusted calls':>14s}")
    print("-" * 66)
    for name in arms:
        block = summary[name]
        print(f"{name:14s} {block['precision']:10.4f} {block['recall']:8.4f} "
              f"{block['f1']:8.4f} {block['f1_std']:8.4f} "
              f"{costs[name]['calls']:14d}")

    print(f"\n-> {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
