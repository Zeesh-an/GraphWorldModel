"""
Q4 — run the candidate-ranking evaluation on held-out graphs.

    python -m scripts.eval_ranking \
        --results results/influence_maximization/ba100/wm_structured_IC/sage_IC.json \
        --data-dir results/influence_maximization/ba100/data \
        --n-graphs 10 --budget-pct 5 --mc-runs 32 \
        --out results/influence_maximization/ba100/ranking_structured_IC.json

Scientific question: can the world model ORDER candidate algorithms well enough
to replace trusted evaluation? Not "is the next state right" — a model can be
good at one and bad at the other.

The candidates are real IM algorithms from `coding_agent.tools.algorithms`, run
on each held-out graph to produce a seed set. The oracle is the NDlib Monte-Carlo
spread of that seed set. Nothing here is synthetic: the ordering being scored is
the one a coding agent would actually face.

Held-out graphs only. `--planning-split test` semantics: the graph ids come from
the test split and the run refuses if any of them appear in train or val.
"""

import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch

from world_model.checkpoint import load_checkpoint
from world_model.wm_data import load_graph_store
from world_model.wm_eval import graphs_in_split
from world_model.wm_ranking import (
    CandidateSet,
    aggregate,
    model_spreads,
    oracle_spreads,
    ranking_report,
)

# The six selectors that harvested the world model's training trajectories,
# under their coding-agent names. Everything else in the pool is unseen.
# Torch grabs every core by default. On the tensors this repository actually
# evaluates -- a 100-to-1000 node graph, one record at a time -- that is
# catastrophic: measured on a 100-node OOD set, 14 threads cost 80.0 ms/record
# against 2.02 ms/record at 1 thread, a 40x slowdown, because the intra-op
# synchronisation dwarfs the arithmetic. It also makes parallel eval jobs fight.
# 1 is therefore the default here, not a tuning choice.
default_threads = 1

seen_policy_names = {
    "random_seeds", "high_degree", "pagerank_seeds", "betweenness_seeds",
    "celf", "hill_climbing",
}

# Candidate pool. Deliberately mixes seen and unseen so Workstream E is a filter
# over one result rather than a separate run under different conditions.
# Excludes algorithms that need an external solver or a spread oracle of their
# own, which would make the "cheap candidate" framing false.
default_pool = (
    "random_seeds", "high_degree", "pagerank_seeds", "betweenness_seeds",
    "hill_climbing", "degree_discount", "eigenvector_seeds", "closeness_seeds",
    "kshell_seeds", "voterank", "collective_influence", "weighted_degree",
    "community_im", "cofim", "irie", "sp1m",
)


def build_candidates(graph_id, store_entry, pool, budget, diffusion_model):
    """Run each algorithm on the graph to get its seed set."""
    from coding_agent.tools.algorithms import algorithms
    from coding_agent.types import GraphInfo

    graph = GraphInfo.from_store_entry(store_entry)
    seed_sets, policies, failures = [], [], {}

    for name in pool:
        function = algorithms.get(name)

        if function is None:
            failures[name] = "not in coding_agent.tools.algorithms"
            continue

        try:
            # diffusion_model matters: several selectors branch on it, and
            # scoring LT candidates that were built as if for IC would make the
            # ranking about the wrong candidate set.
            seeds = function(graph, budget, diffusion_model=diffusion_model)
            seeds = sorted({int(node) for node in seeds})[:budget]
        except Exception as exception:  # noqa: BLE001 - recorded, not raised
            failures[name] = f"{type(exception).__name__}: {exception}"
            continue

        if not seeds:
            failures[name] = "returned an empty seed set"
            continue

        seed_sets.append(seeds)
        policies.append(name)

    return (
        CandidateSet(graph_id=graph_id, seed_sets=seed_sets, policies=policies,
                     budget=budget),
        failures,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q4 candidate-ranking evaluation")
    parser.add_argument("--results", type=Path, required=True,
                        help="train_wm.py results JSON of the model to score (default: None).")
    parser.add_argument("--data-dir", type=Path, required=True, help="dataset directory holding graphs/ and the transition files (default: None).")
    parser.add_argument("--n-graphs", type=int, default=10, help="graphs to evaluate (default: 10).")
    parser.add_argument("--budget-pct", type=float, default=5.0, help="seed budget as percent of nodes (default: 5.0).")
    parser.add_argument("--mc-runs", type=int, default=32,
                        help="simulator rollouts per oracle score (default: 32).")
    parser.add_argument("--horizon", type=int, default=20, help="rollout horizon in timesteps (default: 20).")
    parser.add_argument("--n-samples", type=int, default=20,
                        help="world-model ensemble size (default: 20).")
    parser.add_argument("--win-quantile", type=float, default=0.8,
                        help="oracle quantile a candidate must reach to count as "
                             "a win; recorded in the output (default: 0.8).")
    parser.add_argument("--pool", nargs="*", default=list(default_pool), type=str, help=f"candidate policies to rank (default: {list(default_pool)}).")
    parser.add_argument("--seed", type=int, default=0, help="master random seed (default: 0).")
    parser.add_argument("--device", type=str, default="cpu", help="torch device (default: cpu).")
    parser.add_argument("--threads", type=int, default=default_threads,
                        help=f"torch intra-op threads; more is much slower on small graphs "
                             f"(default: {default_threads}).")
    parser.add_argument("--out", type=Path, required=True, help="output path (default: None).")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    results = json.loads(args.results.read_text())
    config = results["config"]
    diffusion_model = config["diffusion_model"]
    device = torch.device(args.device)

    checkpoint_path = (
        Path(config["ckpt_dir"]) / f"wm_{config['model']}_{diffusion_model}.pt"
    )
    model, spec, train_meta = load_checkpoint(
        checkpoint_path, config=config, device=device, strict_spec=False
    )

    store = load_graph_store(args.data_dir)
    test_graphs = set(graphs_in_split(args.data_dir, diffusion_model, "test"))
    train_graphs = set(graphs_in_split(args.data_dir, diffusion_model, "train"))
    val_graphs = set(graphs_in_split(args.data_dir, diffusion_model, "val"))
    graph_ids = [g for g in store if g in test_graphs][: args.n_graphs]

    if not graph_ids:
        raise ValueError(
            f"no test graphs in {args.data_dir} for {diffusion_model}; ranking "
            f"must be measured on held-out graphs"
        )

    overlap = (set(graph_ids) & train_graphs) | (set(graph_ids) & val_graphs)

    if overlap:
        raise ValueError(f"ranking graphs leak into train/val: {sorted(overlap)}")

    print(f"[rank] {spec.backbone}/{spec.head} {diffusion_model} on "
          f"{len(graph_ids)} held-out graphs, {len(args.pool)} candidate algorithms")

    reports, all_failures, trusted_calls = [], {}, 0
    started = time.perf_counter()

    for offset, graph_id in enumerate(graph_ids):
        entry = store[graph_id]
        budget = max(1, round(entry["num_nodes"] * args.budget_pct / 100))
        candidates, failures = build_candidates(
            graph_id, entry, args.pool, budget, diffusion_model
        )

        if len(candidates.seed_sets) < 2:
            print(f"  {graph_id}: only {len(candidates.seed_sets)} candidates, skipped")
            continue

        for name, reason in failures.items():
            all_failures.setdefault(name, reason)

        true = oracle_spreads(
            candidates, entry, diffusion_model, horizon=args.horizon,
            mc_runs=args.mc_runs, seed=args.seed + offset,
            remove_semantics=spec.remove_semantics,
        )
        trusted_calls += len(candidates.seed_sets) * args.mc_runs

        predicted = model_spreads(
            model, candidates, entry, diffusion_model, device,
            horizon=args.horizon, n_samples=args.n_samples, seed=args.seed + offset,
            remove_semantics=spec.remove_semantics,
            hide_edge_weights=spec.hide_edge_weights,
            action_encoding=spec.action_encoding,
        )

        degrees = np.zeros(entry["num_nodes"])
        np.add.at(degrees, entry["edge_index"][0], 1)
        np.add.at(degrees, entry["edge_index"][1], 1)

        report = ranking_report(
            predicted, true, candidates.policies, graph_id,
            win_quantile=args.win_quantile, degrees=degrees,
            seed_sets=candidates.seed_sets, seen_policies=seen_policy_names,
            seed=args.seed + offset,
        )
        reports.append(report)
        print(f"  {graph_id}: k={budget} n={report.n_candidates} "
              f"pref={report.metrics['preference_accuracy']:.3f} "
              f"tau={report.metrics['kendall_tau']:+.3f} "
              f"c2fw_wm={report.metrics['calls_to_first_win_world_model']:.1f} "
              f"c2fw_rand={report.metrics['calls_to_first_win_random']:.1f}")

    summary = aggregate(reports)
    blob = {
        "question": "Q4_decision_utility",
        "model": {"backbone": spec.backbone, "head": spec.head,
                  "diffusion_model": diffusion_model,
                  "remove_semantics": spec.remove_semantics,
                  "split_mode": train_meta.get("split_mode")},
        "protocol": {
            "graph_ids": graph_ids, "n_graphs": len(reports),
            "train_overlap": 0, "val_overlap": 0,
            "budget_pct": args.budget_pct, "mc_runs": args.mc_runs,
            "horizon": args.horizon, "n_samples": args.n_samples,
            "win_quantile": args.win_quantile, "seed": args.seed,
            "candidate_pool": args.pool,
            "seen_policies": sorted(seen_policy_names & set(args.pool)),
            "unseen_policies": sorted(set(args.pool) - seen_policy_names),
            "candidate_failures": all_failures,
            "trusted_simulator_calls": trusted_calls,
        },
        "summary": summary,
        "per_graph": [report.to_dict() for report in reports],
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    os.makedirs(args.out.parent, exist_ok=True)
    args.out.write_text(json.dumps(blob, indent=2, default=str))

    print(f"\n[rank] {len(reports)} graphs | trusted calls {trusted_calls}")
    for key in ("preference_accuracy", "kendall_tau", "precision_at_1",
                "calls_to_first_win_world_model", "calls_to_first_win_random",
                "calls_to_first_win_degree", "call_reduction_vs_random_world_model",
                "preference_accuracy_seen", "preference_accuracy_unseen"):
        if key in summary:
            print(f"  {key:42s} {summary[key]:+.4f} "
                  f"± {summary.get(key + '_std', float('nan')):.4f} "
                  f"(n={summary.get(key + '_n', 0)})")

    print(f"\n-> {args.out}")
