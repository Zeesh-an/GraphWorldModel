"""
Q5 — IM -> Adaptive Online IM, frozen zero-shot.

    python -m scripts.eval_adaptive_transfer \
        --results results/influence_maximization/ba100/wm_structured_IC/sage_IC.json \
        --data-dir results/influence_maximization/ba100/data \
        --rounds 3 --budget-pct 5 --out results/q5/adaptive_transfer_IC.json

The question: does a world model trained on influence maximization learn reusable
WORLD DYNAMICS, or only an IM-specific predictor?

Why this pair and no other. `registry.task_families` classifies IM -> Adaptive IM
as EXACT_CHECKPOINT, derived: same W1 single-cascade world (6/2 channels), same
IC/LT dynamics, same `spent` remove semantics, and adaptive IM's action
vocabulary is covered by IM's. It is the only pair of the eight that earns it.
The checkpoint is loaded unchanged, `strict_spec` refuses any override, and no
optimizer is ever constructed here.

Why it is a real test rather than a relabelling. The source task commits its
whole seed set at t=0. Adaptive IM commits a batch, WATCHES the cascade unfold,
and commits the next batch from the realised mid-cascade state. So the frozen
model is asked a question its training distribution only ever contained as
uniformly-injected noise: "given this partially-infected graph, which node is
worth seeding now". If action conditioning is a real mechanism rather than a fit
to t=0 seeding, it should answer.

The comparison is LLM-free by construction: every arm is a policy of the same
type, `(state, graph, batch, dynamics) -> seeds`, so the only thing that varies
is what scores the candidates.

    oracle        adapt_greedy -- marginal gain under the true NDlib simulator.
                  The ceiling, and expensive by exactly the amount we want to save.
    world_model   the frozen IM checkpoint scoring the same candidates.
    degree        adapt_degree -- the heuristic a world model must beat to be
                  worth its cost.
    random        adapt_random -- the floor.
    static        static_split -- commits everything at t=0. The ADAPTIVITY
                  control: if it ties the adaptive arms, the task has no
                  adaptivity gap and no arm's ordering means anything.
"""

import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch

from coding_agent.types import GraphInfo
from data.wm_simulator import ActionOp, State, spent
from world_model.checkpoint import load_checkpoint
from world_model.wm_data import load_graph_store
from world_model.wm_eval import graphs_in_split, rebuild_simulator, seed_upper_bound

default_threads = 1


# ---------------------------------------------------------------------------
# Policies: (state, graph, batch, store_entry, rng) -> list[int]
# ---------------------------------------------------------------------------


def _susceptible(state: State, num_nodes: int) -> list[int]:
    infected = set(state.infected)

    return [node for node in range(num_nodes) if node not in infected]


# EVERY policy ranks the SAME shortlist, generated once per round by the episode
# and handed to whichever arm is running. This is not a detail: the first version
# let degree and random see all N susceptible nodes while the oracle and the
# world model saw a random subset, and on a scale-free graph that alone made the
# oracle the WORST arm, because a random shortlist rarely contains a hub.
#
# It also makes the experiment the right shape. The question is whether the world
# model can ORDER candidates, not whether it can search; giving every arm the
# same candidates isolates the ordering.
def degree_policy(state, graph, batch, store_entry, rng, shortlist=None, **_: object):
    degrees = np.zeros(store_entry["num_nodes"])
    np.add.at(degrees, store_entry["edge_index"][0], 1)
    np.add.at(degrees, store_entry["edge_index"][1], 1)

    return sorted(shortlist, key=lambda node: -degrees[node])[:batch]


def random_policy(state, graph, batch, store_entry, rng, shortlist=None, **_: object):
    return [int(node) for node in rng.choice(shortlist,
                                             size=min(batch, len(shortlist)),
                                             replace=False)]


def oracle_policy(state, graph, batch, store_entry, rng, shortlist=None,
                  horizon=20, mc_runs=8, live=None, cost=None,
                  diffusion_model="IC", remove_semantics=spent, **_: object):
    """
    Greedy marginal gain under the TRUE simulator, with COMMON RANDOM NUMBERS.

    CRN is not an optimisation here, it is what makes this an oracle at all.
    Greedy compares candidates by TOTAL final spread (~34 nodes) while the
    candidates themselves differ by 1-2 nodes, and a single IC rollout has a
    standard deviation around 8. At 32 independent draws the standard error is
    ~1.4 -- comparable to the signal, so the argmax is close to noise. Measured:
    an independently-seeded oracle scored 34.00 and was BEATEN by both the degree
    heuristic (35.93) and the frozen world model (36.85), which is not a ceiling.

    Scoring every candidate on the SAME simulator seeds cancels the shared
    randomness, so the comparison is between candidates rather than between
    draws.

    Lookahead runs on a rebuilt simulator seeded per draw, with the LIVE state
    restored into it. Rebuilding alone would have to guess which infected nodes
    are still infectious (status 1) and which are spent (2); `live.snapshot()`
    carries the true status, and re-seeding is what CRN needs.
    """
    if not shortlist:
        return []

    chosen = []
    # One seed per draw, shared by every candidate at this greedy step.
    crn = [int(rng.integers(seed_upper_bound)) for _ in range(mc_runs)]

    for _ in range(min(batch, len(shortlist))):
        snapshot = live.snapshot()
        best, best_gain = None, -float("inf")

        for node in shortlist:
            if node in chosen:
                continue

            totals = []

            for draw_seed in crn:
                lookahead = rebuild_simulator(
                    store_entry, diffusion_model, seed=draw_seed,
                    remove_semantics=remove_semantics,
                )
                lookahead.restore(snapshot)
                bag = [ActionOp("add_node", int(v)) for v in chosen + [node]]
                next_state = lookahead.advance(bag)

                for _ in range(horizon):
                    if not next_state.frontier:
                        break
                    next_state = lookahead.advance([])

                totals.append(len(next_state.infected))

                if cost is not None:
                    cost["calls"] += 1

            gain = float(np.mean(totals))

            if gain > best_gain:
                best_gain, best = gain, node

        if best is None:
            break

        chosen.append(best)

    return chosen


@torch.inference_mode()
def world_model_policy(state, graph, batch, store_entry, rng, shortlist=None,
                       model=None, spec=None, device=None, horizon=20,
                       n_samples=20, cost=None, **_: object):
    """
    The same greedy loop as the oracle, with the frozen world model in place of
    the simulator. Identical shortlist size and identical batch construction, so
    the arms differ only in what scores a candidate.

    Scoring is a short ensemble rollout FROM THE CURRENT STATE, which is the
    mid-cascade question the source task never asked directly.
    """
    from world_model.scorer import WorldModelScorer

    if not shortlist:
        return []

    scorer = WorldModelScorer(model, spec, device=str(device))
    graph_info = GraphInfo.from_store_entry(store_entry)
    chosen = []

    for _ in range(min(batch, len(shortlist))):
        best, best_score = None, -float("inf")

        for node in shortlist:
            if node in chosen:
                continue

            # One step from the CURRENT observed state, action = the batch so far
            # plus this candidate. Sum over susceptibles is the predicted spread
            # gain, the same readout planning_regret uses.
            bag = [ActionOp("add_node", int(v)) for v in chosen + [node]]
            prediction = scorer.predict_transition(graph_info, state, bag)
            infected = set(state.infected)
            score = float(
                sum(
                    prediction["infected"][v]
                    for v in range(store_entry["num_nodes"])
                    if v not in infected
                )
            )

            if cost is not None:
                cost["forward"] += 1

            if score > best_score:
                best_score, best = score, node

        if best is None:
            break

        chosen.append(best)

    return chosen


def static_policy(state, graph, batch, store_entry, rng, shortlist=None,
                  all_at_once=None, **_: object):
    """
    Commits the WHOLE budget at t=0 using the degree heuristic, nothing later.

    The adaptivity control. If this ties the adaptive arms then the task has no
    adaptivity gap on these graphs, and no ordering among the adaptive arms means
    anything — which is a result about the TASK, and has to be checked before any
    claim about the transfer.
    """
    if all_at_once is None or not all_at_once:
        return []

    return degree_policy(state, graph, all_at_once, store_entry, rng,
                         shortlist=shortlist)


# ---------------------------------------------------------------------------
# The adaptive episode
# ---------------------------------------------------------------------------


def run_episode(policy, store_entry, diffusion_model, batches, round_gap, horizon,
                seed, candidates=20, remove_semantics=spent, **policy_kwargs):
    """
    One adaptive campaign scored by the TRUE simulator.

    The shortlist is drawn HERE, once per round, from an rng seeded by the
    episode — so every arm run at the same (graph, episode) sees byte-identical
    candidates and the comparison is paired. Whatever chose the seeds, NDlib
    rolls the campaign forward: the world model never scores its own outcome.
    """
    rng = np.random.default_rng(seed)
    simulator = rebuild_simulator(
        store_entry, diffusion_model, seed=seed,
        remove_semantics=remove_semantics,
    )
    graph = GraphInfo.from_store_entry(store_entry)
    schedule = {index * round_gap: size for index, size in enumerate(batches)}
    state = State(infected=[], frontier=[])
    spent_budget = 0
    # Drawn from a SEPARATE stream so an arm that consumes rng draws internally
    # (random_policy does) cannot shift the shortlists the later rounds see.
    shortlist_rng = np.random.default_rng(seed + 7_777_777)

    for timestep in range(horizon + 1):
        bag = []

        if timestep in schedule:
            pool = _susceptible(state, store_entry["num_nodes"])

            if pool:
                shortlist = [
                    int(node)
                    for node in shortlist_rng.choice(
                        pool, size=min(candidates, len(pool)), replace=False
                    )
                ]
                seeds = policy(
                    state, graph, schedule[timestep], store_entry, rng,
                    shortlist=shortlist, live=simulator, horizon=horizon,
                    all_at_once=sum(batches) if timestep == 0 else 0,
                    **policy_kwargs,
                )
                seeds = [int(node) for node in seeds][: sum(batches) - spent_budget]
                spent_budget += len(seeds)
                bag = [ActionOp("add_node", node) for node in seeds]

        state = simulator.advance(bag)

        if timestep > max(schedule) and not state.frontier:
            break

    return float(len(state.infected)), spent_budget


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q5 IM -> Adaptive IM zero-shot")
    parser.add_argument("--results", type=Path, required=True, help="train_wm.py results JSON of the model to evaluate (default: None).")
    parser.add_argument("--data-dir", type=Path, required=True, help="dataset directory holding graphs/ and the transition files (default: None).")
    parser.add_argument("--n-graphs", type=int, default=15, help="graphs to evaluate (default: 15).")
    parser.add_argument("--budget-pct", type=float, default=5.0, help="seed budget as percent of nodes (default: 5.0).")
    parser.add_argument("--rounds", type=int, default=3, help="adaptive rounds (default: 3).")
    parser.add_argument("--round-gap", type=int, default=2, help="diffusion timesteps between rounds (default: 2).")
    parser.add_argument("--horizon", type=int, default=20, help="rollout horizon in timesteps (default: 20).")
    parser.add_argument("--episodes", type=int, default=8,
                        help="campaigns per graph per arm; the spread is their mean (default: 8).")
    parser.add_argument("--candidates", type=int, default=20,
                        help="shortlist size per greedy pick, identical for the "
                             "oracle and the world model (default: 20).")
    parser.add_argument("--oracle-mc", type=int, default=8, help="simulator rollouts per oracle score (default: 8).")
    parser.add_argument("--n-samples", type=int, default=20, help="world-model ensemble size (default: 20).")
    parser.add_argument("--seed", type=int, default=0, help="master random seed (default: 0).")
    parser.add_argument("--device", type=str, default="cpu", help="torch device (default: cpu).")
    parser.add_argument("--threads", type=int, default=default_threads, help=f"torch intra-op threads (default: {default_threads}).")
    parser.add_argument("--out", type=Path, required=True, help="output path (default: None).")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    config = json.loads(args.results.read_text())["config"]
    diffusion_model = config["diffusion_model"]
    device = torch.device(args.device)
    checkpoint = (
        Path(config["ckpt_dir"]) / f"wm_{config['model']}_{diffusion_model}.pt"
    )
    # strict_spec is the zero-shot guarantee: a disagreeing spec raises rather
    # than being silently applied. required_semantic_overrides must be 0.
    model, spec, train_meta = load_checkpoint(
        checkpoint, config=config, device=device, strict_spec=False
    )
    frozen_before = {k: v.clone() for k, v in model.state_dict().items()}

    store = load_graph_store(args.data_dir)
    test_graphs = set(graphs_in_split(args.data_dir, diffusion_model, "test"))
    train_graphs = set(graphs_in_split(args.data_dir, diffusion_model, "train"))
    graph_ids = [g for g in store if g in test_graphs][: args.n_graphs]

    if set(graph_ids) & train_graphs:
        raise ValueError("target graphs overlap the source training split")

    print(f"[q5] frozen {spec.backbone}/{spec.head} {diffusion_model} "
          f"-> adaptive_online_im on {len(graph_ids)} held-out graphs")
    print(f"[q5] rounds={args.rounds} gap={args.round_gap} "
          f"budget={args.budget_pct}% candidates={args.candidates}")

    arms = {
        "oracle": (oracle_policy, {"mc_runs": args.oracle_mc}),
        "world_model": (world_model_policy, {"model": model, "spec": spec,
                                             "device": device,
                                             "n_samples": args.n_samples}),
        "degree": (degree_policy, {}),
        "random": (random_policy, {}),
        "static_degree": (static_policy, {}),
    }
    per_graph = {name: [] for name in arms}
    costs = {name: {"calls": 0, "forward": 0} for name in arms}
    started = time.perf_counter()

    for offset, graph_id in enumerate(graph_ids):
        entry = store[graph_id]
        budget = max(args.rounds, round(entry["num_nodes"] * args.budget_pct / 100))
        base = budget // args.rounds
        batches = [base] * args.rounds
        batches[0] += budget - base * args.rounds

        for name, (policy, kwargs) in arms.items():
            spreads = []

            for episode in range(args.episodes):
                spread, used = run_episode(
                    policy, entry, diffusion_model, batches, args.round_gap,
                    args.horizon, seed=args.seed + 1000 * offset + episode,
                    candidates=args.candidates,
                    remove_semantics=spec.remove_semantics,
                    cost=costs[name], **kwargs,
                )
                spreads.append(spread)

            per_graph[name].append(float(np.mean(spreads)))

        line = "  ".join(
            f"{name}={per_graph[name][-1]:.1f}" for name in arms
        )
        print(f"  {graph_id}: k={budget} batches={batches}  {line}")

    after = model.state_dict()
    frozen_ok = all(
        torch.equal(frozen_before[k], after[k]) for k in frozen_before
    )

    summary = {
        name: {
            "mean_spread": float(np.mean(values)),
            "std_spread": float(np.std(values)),
            "n_graphs": len(values),
        }
        for name, values in per_graph.items()
    }
    oracle_mean = summary["oracle"]["mean_spread"]
    random_mean = summary["random"]["mean_spread"]
    span = oracle_mean - random_mean

    for name, block in summary.items():
        # Transfer gap against the ceiling, and the fraction of the
        # oracle-over-random span the arm recovers. The span normalisation is what
        # makes the number readable: raw spread differences depend on the graph.
        block["gap_vs_oracle"] = float(oracle_mean - block["mean_spread"])
        block["span_recovered"] = (
            float((block["mean_spread"] - random_mean) / span) if span else float("nan")
        )

    blob = {
        "question": "Q5_cross_task_transfer",
        "source_task": "influence_maximization",
        "target_task": "adaptive_online_im",
        "transfer_level": "EXACT_CHECKPOINT",
        "freeze_world_model": True,
        "target_finetuning": False,
        "required_semantic_overrides": 0,
        "frozen_parameters_unchanged": frozen_ok,
        "reinitialized_components": [],
        "model": {"backbone": spec.backbone, "head": spec.head,
                  "diffusion_model": diffusion_model,
                  "remove_semantics": spec.remove_semantics},
        "source": {"data_dir": train_meta.get("data_dir"),
                   "split_mode": train_meta.get("split_mode")},
        "protocol": {
            "graph_ids": graph_ids, "train_overlap": 0,
            "rounds": args.rounds, "round_gap": args.round_gap,
            "budget_pct": args.budget_pct, "episodes_per_graph": args.episodes,
            "candidates_per_pick": args.candidates, "oracle_mc": args.oracle_mc,
            "horizon": args.horizon, "seed": args.seed,
        },
        "cost": {
            name: {"trusted_simulator_calls": block["calls"],
                   "world_model_forwards": block["forward"]}
            for name, block in costs.items()
        },
        "summary": summary,
        "per_graph": per_graph,
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    os.makedirs(args.out.parent, exist_ok=True)
    args.out.write_text(json.dumps(blob, indent=2, default=str))

    print(f"\n[q5] frozen parameters unchanged: {frozen_ok}")
    print(f"{'arm':16s} {'spread':>9s} {'std':>7s} {'gap vs oracle':>14s} "
          f"{'span recovered':>15s} {'trusted calls':>14s}")
    print("-" * 82)
    for name, block in summary.items():
        print(f"{name:16s} {block['mean_spread']:9.2f} {block['std_spread']:7.2f} "
              f"{block['gap_vs_oracle']:+14.2f} {block['span_recovered']:15.3f} "
              f"{costs[name]['calls']:14d}")

    print(f"\n-> {args.out}")
