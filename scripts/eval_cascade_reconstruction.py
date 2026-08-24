"""
Q5 (Stage E2) — IM -> Cascade Reconstruction, frozen forward-dynamics reuse.

    python -m scripts.eval_cascade_reconstruction \
        --results .../sage_IC.json --data-dir results/sl/ba100_random_src/data \
        --out results/recon/recon_transfer_IC.json

The task: an episode's trajectory `s_0 ... s_T` is partially observed — some
timesteps are hidden — and the hidden states must be filled in. Unlike source
localization, which recovers only `s_0`, this asks for the whole path.

Same transfer level as SL and for the same reason: `gen_action_ops` is empty for
this task, so the frozen model is used as a forward oracle and its
action-conditioning is never exercised. `FORWARD_DYNAMICS`, not zero-shot
action-conditioned transfer.

Reconstruction here is FORWARD FILTERING, which is what a frozen transition model
supports natively and what makes this a fair test of the transition rather than
of an inference procedure we invented:

    given the observed anchors, roll the model forward from the last observed
    state and read its marginals at each hidden timestep

Arms differ only in what propagates between anchors:

    world_model   the frozen IM checkpoint
    oracle        NDlib Monte-Carlo from the same anchor
    persistence   the hidden state equals the last observed one. The trivial
                  baseline, and a strong one for a monotone cascade -- most nodes
                  genuinely do not change in one step, so any method that cannot
                  beat it has learned nothing about the dynamics.
    degree_growth persistence plus the highest-degree susceptible neighbours,
                  matched in count to the true growth. A structural baseline that
                  knows HOW MANY nodes appear but not which.

Scoring is on the HIDDEN steps only, and on the nodes that actually changed:
counting the unchanged majority would let persistence score ~0.95 and hide every
difference.
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


def load_trajectories(data_dir: Path, diffusion_model: str, split: str = "test"):
    """Main-branch episodes as a list of per-timestep infected sets."""
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

        seeds = sorted(
            int(op["target"]) for op in records[0]["action"] if op["op"] == "add_node"
        )
        states = [set(seeds)] + [
            set(int(v) for v in record["next_state"]["infected"]) for record in records
        ]

        # A trajectory that never grows carries no reconstruction signal: every
        # method scores identically and the episode only adds noise.
        if len(states) >= 4 and len(states[-1]) > len(states[0]):
            episodes.append(
                {"graph_id": graph_id, "episode_id": episode_id,
                 "seeds": seeds, "states": states}
            )

    return episodes, dropped


def mask_steps(n_states: int, mask_fraction: float, rng) -> list[int]:
    """
    Which timesteps are hidden. Never the first or last: the first is the seed
    commit the task is given, and the last is the terminal observation every
    method anchors on.
    """
    candidates = list(range(1, n_states - 1))

    if not candidates:
        return []

    count = max(1, int(round(len(candidates) * mask_fraction)))

    return sorted(
        int(t) for t in rng.choice(candidates, size=min(count, len(candidates)),
                                   replace=False)
    )


@torch.inference_mode()
def world_model_fill(anchor_state, steps_ahead, graph_info, scorer, n_samples, seed):
    """Frozen model rolled `steps_ahead` from an anchor; per-node P(infected)."""
    from world_model.scorer import ScoringContext

    if steps_ahead <= 0:
        return None

    # The seed bag re-asserts the anchor's infected set at t=0 of this segment,
    # which is exactly T_exo: `add_node` on an already-infected node is a no-op
    # for the count and pins the state the segment starts from.
    plan = [[ActionOp("add_node", int(v)) for v in sorted(anchor_state)]]
    trajectory = scorer.rollout(
        graph_info, plan,
        ScoringContext(horizon=steps_ahead, budget=len(anchor_state),
                       n_samples=n_samples, seed=seed),
    )
    marginals = trajectory.final_marginals

    return None if marginals is None else np.asarray(marginals, dtype=float)


def oracle_fill(anchor_state, steps_ahead, store_entry, diffusion_model, mc_runs,
                remove_semantics, crn_seeds, cost=None):
    """NDlib from the same anchor, common random numbers."""
    num_nodes = store_entry["num_nodes"]
    counts = np.zeros(num_nodes)

    for draw_seed in crn_seeds[:mc_runs]:
        simulator = rebuild_simulator(
            store_entry, diffusion_model, seed=draw_seed,
            remove_semantics=remove_semantics,
        )
        state = simulator.advance(
            [ActionOp("add_node", int(v)) for v in sorted(anchor_state)]
        )

        for _ in range(steps_ahead - 1):
            if not state.frontier:
                break
            state = simulator.advance([])

        counts[list(state.infected)] += 1

        if cost is not None:
            cost["calls"] += 1

    return counts / max(len(crn_seeds[:mc_runs]), 1)


def score_hidden_step(predicted, truth, anchor, num_nodes) -> dict | None:
    """
    F1 on the nodes that CHANGED between the anchor and the hidden step.

    Scoring every node would let persistence reach ~0.95 on a monotone cascade
    where most nodes are unchanged, and no method would be distinguishable. The
    question is which NEW nodes lit up.
    """
    new_true = truth - anchor

    if not new_true:
        return None

    if predicted is None:
        new_pred = set()
    else:
        # Take exactly as many as truly appeared. This hands every arm the
        # correct COUNT and asks only WHICH, so a method cannot win or lose on
        # calibration of the total -- the thing being compared is the ordering
        # over candidate nodes.
        eligible = [v for v in range(num_nodes) if v not in anchor]
        ranked = sorted(eligible, key=lambda v: -predicted[v])
        new_pred = set(ranked[: len(new_true)])

    hits = len(new_pred & new_true)
    precision = hits / len(new_pred) if new_pred else 0.0
    recall = hits / len(new_true)

    return {
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ),
        "n_new": len(new_true),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Q5 IM -> Cascade Reconstruction")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--n-episodes", type=int, default=60)
    parser.add_argument("--mask-fraction", type=float, default=0.5)
    parser.add_argument("--oracle-mc", type=int, default=32)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threads", type=int, default=default_threads)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)

    from world_model.scorer import WorldModelScorer

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
    scorer = WorldModelScorer(model, spec, device=str(device))

    store = load_graph_store(args.data_dir)
    episodes, dropped = load_trajectories(args.data_dir, diffusion_model)
    rng = np.random.default_rng(args.seed)

    if len(episodes) > args.n_episodes:
        episodes = [
            episodes[i]
            for i in rng.choice(len(episodes), size=args.n_episodes, replace=False)
        ]

    print(f"[recon] frozen {spec.backbone}/{spec.head} {diffusion_model} as a "
          f"FORWARD oracle for cascade reconstruction")
    print(f"[recon] {len(episodes)} episodes ({dropped} dropped: injected actions "
          f"or no growth), mask_fraction={args.mask_fraction}")

    arms = ("world_model", "oracle", "persistence", "degree_growth")
    scores = {name: [] for name in arms}
    costs = {name: {"calls": 0} for name in arms}
    started = time.perf_counter()

    for index, episode in enumerate(episodes):
        entry = store[episode["graph_id"]]
        num_nodes = entry["num_nodes"]
        graph_info = GraphInfo.from_store_entry(entry)
        states = episode["states"]
        episode_rng = np.random.default_rng(args.seed + index)
        hidden = mask_steps(len(states), args.mask_fraction, episode_rng)

        if not hidden:
            continue

        degrees = np.zeros(num_nodes)
        np.add.at(degrees, entry["edge_index"][0], 1)
        np.add.at(degrees, entry["edge_index"][1], 1)
        crn = [
            int(episode_rng.integers(seed_upper_bound)) for _ in range(args.oracle_mc)
        ]

        for step in hidden:
            # Anchor = the most recent OBSERVED state before this hidden step.
            anchor_index = max(t for t in range(step) if t not in hidden)
            anchor = states[anchor_index]
            ahead = step - anchor_index
            truth = states[step]

            predictions = {
                "world_model": world_model_fill(
                    anchor, ahead, graph_info, scorer, args.n_samples,
                    args.seed + index,
                ),
                "oracle": oracle_fill(
                    anchor, ahead, entry, diffusion_model, args.oracle_mc,
                    spec.remove_semantics, crn, costs["oracle"],
                ),
                "persistence": None,
                "degree_growth": degrees,
            }

            for name in arms:
                result = score_hidden_step(
                    predictions[name], truth, anchor, num_nodes
                )

                if result is not None:
                    scores[name].append(result)

        if index < 5 or index % 20 == 0:
            line = "  ".join(
                f"{name}={np.mean([s['f1'] for s in scores[name]]):.3f}"
                for name in arms
                if scores[name]
            )
            print(f"  {episode['episode_id'][:36]:36s} T={len(states)} "
                  f"hidden={hidden}  {line}")

    after = model.state_dict()
    frozen_ok = all(torch.equal(frozen_before[k], after[k]) for k in frozen_before)

    summary = {
        name: {
            metric: float(np.mean([s[metric] for s in scores[name]]))
            for metric in ("precision", "recall", "f1")
        }
        | {
            "f1_std": float(np.std([s["f1"] for s in scores[name]])),
            "n_hidden_steps": len(scores[name]),
        }
        for name in arms
        if scores[name]
    }

    blob = {
        "question": "Q5_cross_task_transfer",
        "source_task": "influence_maximization",
        "target_task": "cascade_reconstruction",
        "transfer_level": "FORWARD_DYNAMICS",
        "note": "frozen forward model used as a filter between observed anchors; "
                "action conditioning is not exercised (target data carries no "
                "mid-cascade interventions)",
        "freeze_world_model": True,
        "target_finetuning": False,
        "frozen_parameters_unchanged": frozen_ok,
        "reused_components": ["graph_backbone", "edge_propensity", "T_endo",
                              "T_exo_semantics"],
        "reinitialized_components": ["readout", "filtering_procedure"],
        "model": {"backbone": spec.backbone, "head": spec.head,
                  "diffusion_model": diffusion_model},
        "source": {"data_dir": train_meta.get("data_dir"),
                   "split_mode": train_meta.get("split_mode")},
        "protocol": {
            "data_dir": str(args.data_dir), "n_episodes": len(episodes),
            "episodes_dropped": dropped, "mask_fraction": args.mask_fraction,
            "oracle_mc": args.oracle_mc, "n_samples": args.n_samples,
            "seed": args.seed,
            "scoring": "F1 over newly-infected nodes at each hidden step; every "
                       "arm is given the true count and asked only WHICH nodes",
        },
        "cost": costs,
        "summary": summary,
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(blob, indent=2, default=str))

    print(f"\n[recon] frozen parameters unchanged: {frozen_ok}")
    print(f"{'arm':16s} {'precision':>10s} {'recall':>8s} {'F1':>8s} {'F1 sd':>8s} "
          f"{'steps':>7s} {'trusted':>9s}")
    print("-" * 68)
    for name in arms:
        if name not in summary:
            continue
        block = summary[name]
        print(f"{name:16s} {block['precision']:10.4f} {block['recall']:8.4f} "
              f"{block['f1']:8.4f} {block['f1_std']:8.4f} "
              f"{block['n_hidden_steps']:7d} {costs[name]['calls']:9d}")

    print(f"\n-> {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
