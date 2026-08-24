"""
Ensemble size as a quality–cost ablation, and the wall-clock break-even.

    python -m scripts.eval_ensemble_ablation \
        --results .../sage_IC.json --data-dir results/val/ws100/data \
        --n-samples 1 3 5 10 20 --out results/ablation/ws100_val.json

Two things this measures that the Q4 table does not.

**Quality against cost.** `n_samples` is the world model's dominant cost knob and
its effect is super-linear, so the operating point is a real choice rather than a
default. Every setting scores the SAME candidate set against the SAME oracle —
computed once per graph and reused — so the comparison isolates ensemble size and
costs a fifth of running the sweep independently.

**The break-even, per dataset.** A trusted-call saving is only a wall-clock
saving if scoring K candidates costs less than the calls it avoids:

    K * t_model  <  (calls_random - calls_model) * t_trusted

so the model pays for itself once

    t_trusted  >  K * t_model / (calls_random - calls_model)

Both sides are measured here, on the dataset in hand. The threshold is NOT
transferable between datasets — t_model, t_trusted and the call saving all move
with graph size — which is why this script recomputes it rather than quoting one.

Select the operating point on a VALIDATION dataset. Choosing `n_samples` on the
same graphs the headline number is reported on is selection on test.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from coding_agent.types import GraphInfo
from data.wm_simulator import ActionOp
from world_model.checkpoint import load_checkpoint
from world_model.wm_data import load_graph_store
from world_model.wm_eval import graphs_in_split
from world_model.wm_ranking import (
    calls_to_first_win,
    kendall_tau,
    oracle_spreads,
    preference_accuracy,
    ranking_orders,
)

default_threads = 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Ensemble-size quality/cost ablation")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n-graphs", type=int, default=8)
    parser.add_argument("--n-samples", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    parser.add_argument("--budget-pct", type=float, default=5.0)
    parser.add_argument("--mc-runs", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--win-quantile", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threads", type=int, default=default_threads)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)

    from scripts.eval_ranking import build_candidates, default_pool
    from world_model.scorer import ScoringContext, WorldModelScorer

    config = json.loads(args.results.read_text())["config"]
    diffusion_model = config["diffusion_model"]
    device = torch.device(args.device)
    checkpoint = (
        Path(config["ckpt_dir"]) / f"wm_{config['model']}_{diffusion_model}.pt"
    )
    model, spec, train_meta = load_checkpoint(
        checkpoint, config=config, device=device, strict_spec=False
    )
    scorer = WorldModelScorer(model, spec, device=str(device))

    store = load_graph_store(args.data_dir)
    wanted = set(graphs_in_split(args.data_dir, diffusion_model, args.split))
    graph_ids = [g for g in store if g in wanted][: args.n_graphs]

    if not graph_ids:
        raise ValueError(f"no {args.split} graphs in {args.data_dir}")

    print(f"[abl] {spec.backbone}/{spec.head} {diffusion_model} on "
          f"{len(graph_ids)} {args.split} graphs from {args.data_dir}")

    per_graph = {n: [] for n in args.n_samples}
    latency = {n: [] for n in args.n_samples}
    trusted_latency = []
    candidate_counts = []

    for offset, graph_id in enumerate(graph_ids):
        entry = store[graph_id]
        budget = max(1, round(entry["num_nodes"] * args.budget_pct / 100))
        candidates, _ = build_candidates(
            graph_id, entry, default_pool, budget, diffusion_model
        )

        if len(candidates.seed_sets) < 2:
            continue

        candidate_counts.append(len(candidates.seed_sets))

        # Oracle ONCE per graph: it is the ground truth for every n_samples, and
        # its per-call cost is what the break-even is measured against.
        started = time.perf_counter()
        true = oracle_spreads(
            candidates, entry, diffusion_model, horizon=args.horizon,
            mc_runs=args.mc_runs, seed=args.seed + offset,
            remove_semantics=spec.remove_semantics,
        )
        trusted_latency.append(
            (time.perf_counter() - started) / len(candidates.seed_sets)
        )

        graph_info = GraphInfo.from_store_entry(entry)
        degrees = np.zeros(entry["num_nodes"])
        np.add.at(degrees, entry["edge_index"][0], 1)
        np.add.at(degrees, entry["edge_index"][1], 1)
        threshold = float(np.quantile(np.asarray(true), args.win_quantile))

        for n_samples in args.n_samples:
            context = ScoringContext(
                horizon=args.horizon, budget=budget, n_samples=n_samples,
                seed=args.seed + offset,
            )
            started = time.perf_counter()
            predicted = [
                scorer.score_candidate(
                    graph_info,
                    [[ActionOp("add_node", int(v)) for v in seeds]],
                    context,
                )
                for seeds in candidates.seed_sets
            ]
            latency[n_samples].append(
                (time.perf_counter() - started) / len(candidates.seed_sets)
            )

            orders = ranking_orders(
                predicted, true, degrees, candidates.seed_sets,
                np.random.default_rng(args.seed + offset),
            )
            per_graph[n_samples].append(
                {
                    "preference_accuracy": preference_accuracy(predicted, true)[
                        "preference_accuracy"
                    ],
                    "kendall_tau": kendall_tau(predicted, true),
                    **{
                        f"calls_{name}": calls_to_first_win(order, true, threshold)
                        for name, order in orders.items()
                    },
                }
            )

        print(f"  {graph_id}: k={budget} n_cand={len(candidates.seed_sets)} "
              f"trusted={1000 * trusted_latency[-1]:.1f}ms/cand  "
              + "  ".join(
                  f"n{n}={per_graph[n][-1]['preference_accuracy']:.3f}"
                  for n in args.n_samples
              ))

    mean_trusted = float(np.mean(trusted_latency))
    mean_candidates = float(np.mean(candidate_counts))
    rows = []

    for n_samples in args.n_samples:
        records = per_graph[n_samples]
        calls_model = float(np.mean([r["calls_world_model"] for r in records]))
        calls_random = float(np.mean([r["calls_random"] for r in records]))
        calls_degree = float(np.mean([r["calls_degree"] for r in records]))
        model_latency = float(np.mean(latency[n_samples]))
        saved_calls = calls_random - calls_model

        # End to end: score every candidate, then spend the calls the ranking
        # still needs. The baseline spends only calls.
        end_to_end = mean_candidates * model_latency + calls_model * mean_trusted
        baseline = calls_random * mean_trusted

        rows.append(
            {
                "n_samples": n_samples,
                "preference_accuracy": float(
                    np.mean([r["preference_accuracy"] for r in records])
                ),
                "preference_accuracy_std": float(
                    np.std([r["preference_accuracy"] for r in records])
                ),
                "kendall_tau": float(np.mean([r["kendall_tau"] for r in records])),
                "calls_world_model": calls_model,
                "calls_random": calls_random,
                "calls_degree": calls_degree,
                "trusted_call_reduction": (
                    float(saved_calls / calls_random) if calls_random else float("nan")
                ),
                "wm_latency_ms_per_candidate": 1000 * model_latency,
                "end_to_end_ms": 1000 * end_to_end,
                "baseline_ms": 1000 * baseline,
                "wall_clock_saving": (
                    float(1 - end_to_end / baseline) if baseline else float("nan")
                ),
                # The threshold at which this operating point starts paying for
                # itself ON THIS DATASET. Not transferable: every term moves with
                # graph size.
                "break_even_trusted_ms": (
                    1000 * mean_candidates * model_latency / saved_calls
                    if saved_calls > 0
                    else float("inf")
                ),
            }
        )

    blob = {
        "question": "Q4_cost_quality_ablation",
        "model": {"backbone": spec.backbone, "head": spec.head,
                  "diffusion_model": diffusion_model},
        "protocol": {
            "data_dir": str(args.data_dir), "split": args.split,
            "graph_ids": graph_ids, "n_graphs": len(candidate_counts),
            "mean_candidates_per_graph": mean_candidates,
            "budget_pct": args.budget_pct, "mc_runs": args.mc_runs,
            "horizon": args.horizon, "win_quantile": args.win_quantile,
            "seed": args.seed, "threads": args.threads, "device": args.device,
        },
        "measured_trusted_ms_per_candidate": 1000 * mean_trusted,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(blob, indent=2, default=str))

    print(f"\ntrusted evaluation: {1000 * mean_trusted:.1f} ms/candidate "
          f"({args.mc_runs} MC), {mean_candidates:.0f} candidates/graph\n")
    print(f"{'n':>3s} {'pref':>7s} {'tau':>7s} {'calls_wm':>9s} {'calls_rnd':>10s} "
          f"{'call_red':>9s} {'wm_ms':>8s} {'e2e_ms':>9s} {'base_ms':>9s} "
          f"{'wall':>8s} {'break_even':>11s}")
    print("-" * 100)
    for row in rows:
        print(f"{row['n_samples']:3d} {row['preference_accuracy']:7.4f} "
              f"{row['kendall_tau']:7.4f} {row['calls_world_model']:9.2f} "
              f"{row['calls_random']:10.2f} "
              f"{100 * row['trusted_call_reduction']:8.1f}% "
              f"{row['wm_latency_ms_per_candidate']:8.1f} "
              f"{row['end_to_end_ms']:9.0f} {row['baseline_ms']:9.0f} "
              f"{100 * row['wall_clock_saving']:7.1f}% "
              f"{row['break_even_trusted_ms']:10.0f}ms")

    print(f"\n-> {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
