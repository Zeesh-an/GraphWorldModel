"""
Q5 (Stage F1) — IM -> Critical Node Detection, MECHANISM transfer.

    python -m scripts.eval_cnd_transfer \
        --source-results .../ba100/wm_structured_IC/sage_IC.json \
        --target-results .../cnd/ba100/wm_structured_IC/sage_IC.json \
        --data-dir results/cnd/ba100/data --out results/cnd/cnd_transfer.json

This is the weakest of the three transfer levels and must not be reported as the
other two. `registry.task_families` classifies IM -> CND `MECHANISM_ONLY`, and
the reason is precise: the two tasks share the W1 world (6/2 channels, IC/LT) so
the state_dict LOADS, but `remove_semantics` differs -- IM's `spent` keeps a
removed node counted, CND's `blocked` deletes it from the graph -- and T_exo
branches on exactly that value INSIDE the head. A source checkpoint therefore
loads with no shape error and applies the wrong deterministic semantics silently.

So the transfer here is explicit surgery, not a load:

    reused        graph_backbone, edge_propensity, T_endo  (the learned weights)
    rebuilt       T_exo_semantics                          (the blocked branch)

`load_checkpoint(..., strict_spec=False)` is what performs it, and passing
`strict_spec=False` is the whole point: the default refuses this, and the refusal
is what stops it happening by accident elsewhere.

Arms:

    cnd_trained     a checkpoint trained ON CND data under `blocked`. The
                    learned-model ceiling.
    im_transferred  the IM checkpoint's weights under CND's T_exo. Ours.
    oracle          NDlib greedy with common random numbers. The true ceiling.
    degree          remove the highest-degree nodes. The structural baseline.
    random          the floor.

Objective is containment, so LOWER final spread is better and every sign in this
file is flipped relative to the influence-maximization scripts.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from coding_agent.types import GraphInfo
from data.wm_simulator import ActionOp, blocked
from world_model.checkpoint import ModelSpec, load_checkpoint
from world_model.wm_data import load_graph_store
from world_model.wm_eval import graphs_in_split, rebuild_simulator, seed_upper_bound

default_threads = 1


def spread_after_removal(store_entry, diffusion_model, outbreak_seeds, removals,
                         horizon, mc_runs, crn_seeds, cost=None):
    """
    True expected final spread when `removals` are blocked before the cascade runs.

    The deletion bag is the full one -- `remove_node` plus every incident
    `remove_edge` -- because that is what `blocked` means structurally; a bare
    `remove_node` would leave the node isolated in status only and let LT
    re-activate it.
    """

    totals = []

    for draw_seed in crn_seeds[:mc_runs]:
        simulator = rebuild_simulator(
            store_entry, diffusion_model, seed=draw_seed, remove_semantics=blocked
        )
        graph = simulator.model.graph.graph if hasattr(
            simulator.model.graph, "graph"
        ) else None
        bag = []

        for node in removals:
            bag.append(ActionOp("remove_node", int(node)))

            if graph is not None and node in graph:
                for other in list(graph.neighbors(node)):
                    bag.append(ActionOp("remove_edge", int(node), int(other)))
                    bag.append(ActionOp("remove_edge", int(other), int(node)))

        bag += [ActionOp("add_node", int(v)) for v in outbreak_seeds]
        state = simulator.advance(bag)

        for _ in range(horizon):
            if not state.frontier:
                break
            state = simulator.advance([])

        totals.append(float(len(state.infected)))

        if cost is not None:
            cost["calls"] += 1

    return float(np.mean(totals))


@torch.inference_mode()
def model_spread_after_removal(scorer, graph_info, outbreak_seeds, removals,
                               horizon, n_samples):
    """Predicted final spread under the world model with `removals` blocked."""
    from world_model.scorer import ScoringContext

    bag = [ActionOp("remove_node", int(v)) for v in removals]
    bag += [ActionOp("add_node", int(v)) for v in outbreak_seeds]

    return float(
        scorer.rollout(
            graph_info, [bag],
            ScoringContext(horizon=horizon, budget=len(removals) or 1,
                           n_samples=n_samples, seed=0),
        ).reward
    )


def greedy_min_spread(score, shortlist, budget):
    """Greedy removal set: at each step take the node that MINIMISES spread."""
    chosen: list[int] = []

    for _ in range(min(budget, len(shortlist))):
        best, best_value = None, float("inf")

        for node in shortlist:
            if node in chosen:
                continue

            value = score(chosen + [node])

            if value < best_value:
                best_value, best = value, node

        if best is None:
            break

        chosen.append(best)

    return chosen


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Q5 IM -> CND mechanism transfer")
    parser.add_argument("--source-results", type=Path, required=True,
                        help="IM-trained results JSON (spent semantics)")
    parser.add_argument("--target-results", type=Path, default=None,
                        help="CND-trained results JSON (blocked); the ceiling")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--n-graphs", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--budget-pct", type=float, default=5.0)
    parser.add_argument("--outbreak-pct", type=float, default=5.0)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--oracle-mc", type=int, default=16)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threads", type=int, default=default_threads)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)

    from world_model.scorer import WorldModelScorer

    device = torch.device(args.device)
    source_config = json.loads(args.source_results.read_text())["config"]
    diffusion_model = source_config["diffusion_model"]
    source_checkpoint = (
        Path(source_config["ckpt_dir"])
        / f"wm_{source_config['model']}_{diffusion_model}.pt"
    )

    # THE transfer. The source was trained under `spent`; CND runs under
    # `blocked`, and T_exo branches on that inside the head. strict_spec=False is
    # what allows the override, and the default refusing it is what keeps this
    # from happening silently anywhere else.
    transferred_spec = ModelSpec.from_results_json(args.source_results)
    transferred_spec = ModelSpec(
        **{**transferred_spec.to_dict(), "remove_semantics": blocked}
    )
    transferred_model, _, source_meta = load_checkpoint(
        source_checkpoint, config=transferred_spec, device=device, strict_spec=False
    )
    scorers = {
        "im_transferred": WorldModelScorer(
            transferred_model, transferred_spec, device=str(device)
        )
    }

    if args.target_results and args.target_results.exists():
        target_config = json.loads(args.target_results.read_text())["config"]
        target_checkpoint = (
            Path(target_config["ckpt_dir"])
            / f"wm_{target_config['model']}_{target_config['diffusion_model']}.pt"
        )

        if target_checkpoint.exists():
            model, spec, _ = load_checkpoint(
                target_checkpoint, config=target_config, device=device,
                strict_spec=False,
            )
            scorers["cnd_trained"] = WorldModelScorer(model, spec, device=str(device))

    store = load_graph_store(args.data_dir)
    wanted = set(graphs_in_split(args.data_dir, diffusion_model, "test"))
    graph_ids = [g for g in store if g in wanted][: args.n_graphs]

    if not graph_ids:
        raise ValueError(f"no test graphs in {args.data_dir}")

    arms = list(scorers) + ["oracle", "degree", "random", "no_removal"]
    spreads = {name: [] for name in arms}
    costs = {name: {"calls": 0} for name in arms}
    print(f"[cnd] mechanism transfer on {len(graph_ids)} held-out graphs; "
          f"arms: {arms}")
    started = time.perf_counter()

    for offset, graph_id in enumerate(graph_ids):
        entry = store[graph_id]
        num_nodes = entry["num_nodes"]
        budget = max(1, round(num_nodes * args.budget_pct / 100))
        graph_info = GraphInfo.from_store_entry(entry)
        degrees = np.zeros(num_nodes)
        np.add.at(degrees, entry["edge_index"][0], 1)
        np.add.at(degrees, entry["edge_index"][1], 1)

        for episode in range(args.episodes):
            rng = np.random.default_rng(args.seed + 1000 * offset + episode)
            outbreak_count = max(1, round(num_nodes * args.outbreak_pct / 100))
            outbreak_seeds = [
                int(v) for v in rng.choice(num_nodes, size=outbreak_count,
                                           replace=False)
            ]
            pool = [v for v in range(num_nodes) if v not in set(outbreak_seeds)]
            shortlist = [
                int(v)
                for v in rng.choice(pool, size=min(args.candidates, len(pool)),
                                    replace=False)
            ]
            crn = [int(rng.integers(seed_upper_bound)) for _ in range(args.oracle_mc)]

            selections = {
                "oracle": greedy_min_spread(
                    lambda removals: spread_after_removal(
                        entry, diffusion_model, outbreak_seeds, removals,
                        args.horizon, args.oracle_mc, crn, costs["oracle"],
                    ),
                    shortlist, budget,
                ),
                "degree": sorted(shortlist, key=lambda v: -degrees[v])[:budget],
                "random": [
                    int(v)
                    for v in np.random.default_rng(args.seed + offset).choice(
                        shortlist, size=min(budget, len(shortlist)), replace=False
                    )
                ],
                "no_removal": [],
            }

            for name, scorer in scorers.items():
                selections[name] = greedy_min_spread(
                    lambda removals: model_spread_after_removal(
                        scorer, graph_info, outbreak_seeds, removals,
                        args.horizon, args.n_samples,
                    ),
                    shortlist, budget,
                )

            for name in arms:
                spreads[name].append(
                    spread_after_removal(
                        entry, diffusion_model, outbreak_seeds, selections[name],
                        args.horizon, args.oracle_mc, crn,
                    )
                )

        line = "  ".join(
            f"{name}={np.mean(spreads[name][-args.episodes:]):.1f}" for name in arms
        )
        print(f"  {graph_id}: k={budget} outbreak={outbreak_count}  {line}")

    summary = {
        name: {
            "mean_spread": float(np.mean(values)),
            "std_spread": float(np.std(values)),
            "n": len(values),
        }
        for name, values in spreads.items()
    }
    # Containment: lower is better, so the span runs from no_removal (worst) down
    # to oracle (best), and an arm's credit is how much of it that arm closed.
    worst = summary["no_removal"]["mean_spread"]
    best = summary["oracle"]["mean_spread"]
    span = worst - best

    for name, block in summary.items():
        block["span_recovered"] = (
            float((worst - block["mean_spread"]) / span) if span else float("nan")
        )

    blob = {
        "question": "Q5_cross_task_transfer",
        "source_task": "influence_maximization",
        "target_task": "critical_node_detection",
        "transfer_level": "MECHANISM_ONLY",
        "note": "the source checkpoint's WEIGHTS are reused under the target's "
                "T_exo (blocked). This is NOT zero-shot checkpoint transfer: "
                "remove_semantics differs and load_checkpoint refuses it by "
                "default; strict_spec=False performs the override deliberately",
        "reused_components": ["graph_backbone", "edge_propensity", "T_endo"],
        "rebuilt_components": ["T_exo_semantics"],
        "freeze_world_model": True,
        "target_finetuning": False,
        "source": {"data_dir": source_meta.get("data_dir"),
                   "remove_semantics_trained": "spent",
                   "remove_semantics_applied": blocked},
        "protocol": {
            "data_dir": str(args.data_dir), "graph_ids": graph_ids,
            "episodes_per_graph": args.episodes, "budget_pct": args.budget_pct,
            "outbreak_pct": args.outbreak_pct, "candidates": args.candidates,
            "oracle_mc": args.oracle_mc, "n_samples": args.n_samples,
            "objective": "minimize final spread",
        },
        "cost": costs,
        "summary": summary,
        "per_episode": spreads,
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(blob, indent=2, default=str))

    print(f"\n{'arm':16s} {'spread':>9s} {'std':>7s} {'span closed':>12s} "
          f"{'trusted':>9s}   (lower spread is better)")
    print("-" * 62)
    for name in arms:
        block = summary[name]
        print(f"{name:16s} {block['mean_spread']:9.2f} {block['std_spread']:7.2f} "
              f"{block['span_recovered']:12.3f} {costs[name]['calls']:9d}")

    print(f"\n-> {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
