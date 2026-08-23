"""
Experiment 7 — the closed loop, under a FIXED trusted-call budget.

    python -m scripts.eval_closed_loop \
        --results .../sage_IC.json --data-dir results/ood/ws100/data \
        --out results/closed_loop/ws100.json

This is the experiment the whole project is for, and the only one that answers
the question in the form a practitioner asks it:

    given B expensive evaluations, does an agent that consults the world model
    end up with a BETTER algorithm than one that does not?

Every earlier decision metric is a proxy for this. Preference accuracy asks
whether the ordering is right; calls-to-first-win asks how deep you dig before
success. Neither says what you WALK AWAY WITH when the budget runs out, which is
the quantity that matters.

The protocol, identical for every agent:

    1. K candidate algorithms are generated on the graph (the same K for all).
    2. The agent orders them, using whatever it has.
    3. It spends its B trusted evaluations on its first B, in order.
    4. Its result is the best TRUE value among those B.

Only step 2 differs between agents, so the comparison is exactly "what did
consulting the world model buy". The world model costs zero trusted calls and is
therefore free in this currency -- which is the honest framing, since the whole
premise is that a trusted evaluation is the expensive thing. Wall-clock is
reported separately and is not netted into this.

Agents:

    wm_agent      orders by frozen world-model score
    no_wm_random  orders at random. What you do with no signal.
    no_wm_degree  orders by total seed-set degree. A real practitioner's prior,
                  and the baseline that matters: beating random is easy, beating
                  the heuristic people actually use is the claim.
    oracle_agent  orders by true value. The ceiling; it cannot be beaten and
                  exists to normalise the others.

Reported as a CURVE over B = 1..K, because a single budget is a cherry-pick.
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
from world_model.wm_ranking import oracle_spreads

default_threads = 1


def best_within_budget(order: list[int], true: list[float], budget: int) -> float:
    """Best TRUE value among the first `budget` candidates of an ordering."""
    return float(max(true[index] for index in order[:budget]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fixed-budget closed-loop experiment")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n-graphs", type=int, default=15)
    parser.add_argument("--budget-pct", type=float, default=5.0)
    parser.add_argument("--mc-runs", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=5,
                        help="world-model ensemble size; 5 is the operating point "
                             "selected on validation graphs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="repeats; the random agent and the shortlist vary "
                             "with it, so this is where the CI comes from")
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
    frozen_before = {k: v.clone() for k, v in model.state_dict().items()}
    scorer = WorldModelScorer(model, spec, device=str(device))

    store = load_graph_store(args.data_dir)
    wanted = set(graphs_in_split(args.data_dir, diffusion_model, args.split))
    graph_ids = [g for g in store if g in wanted][: args.n_graphs]

    if not graph_ids:
        raise ValueError(f"no {args.split} graphs in {args.data_dir}")

    print(f"[loop] {spec.backbone}/{spec.head} {diffusion_model}, n_samples="
          f"{args.n_samples}, {len(graph_ids)} graphs x {len(args.seeds)} seeds")

    agents = ("wm_agent", "no_wm_degree", "no_wm_random", "oracle_agent")
    # per (agent, budget) -> list of best-found values, one per (graph, seed)
    records = {agent: {} for agent in agents}
    wm_seconds, trusted_seconds, candidate_counts = [], [], []

    for offset, graph_id in enumerate(graph_ids):
        entry = store[graph_id]
        budget_k = max(1, round(entry["num_nodes"] * args.budget_pct / 100))
        candidates, _ = build_candidates(
            graph_id, entry, default_pool, budget_k, diffusion_model
        )

        if len(candidates.seed_sets) < 3:
            continue

        count = len(candidates.seed_sets)
        candidate_counts.append(count)
        graph_info = GraphInfo.from_store_entry(entry)
        degrees = np.zeros(entry["num_nodes"])
        np.add.at(degrees, entry["edge_index"][0], 1)
        np.add.at(degrees, entry["edge_index"][1], 1)

        # Ground truth, once per graph: it defines every agent's payoff and is
        # also what the oracle agent orders by.
        started = time.perf_counter()
        true = oracle_spreads(
            candidates, entry, diffusion_model, horizon=args.horizon,
            mc_runs=args.mc_runs, seed=offset,
            remove_semantics=spec.remove_semantics,
        )
        trusted_seconds.append((time.perf_counter() - started) / count)

        degree_totals = [
            float(sum(degrees[node] for node in seeds))
            for seeds in candidates.seed_sets
        ]

        for seed in args.seeds:
            rng = np.random.default_rng(1000 * offset + seed)
            context = ScoringContext(
                horizon=args.horizon, budget=budget_k,
                n_samples=args.n_samples, seed=seed,
            )
            started = time.perf_counter()
            predicted = [
                scorer.score_candidate(
                    graph_info, [[ActionOp("add_node", int(v)) for v in seeds]],
                    context,
                )
                for seeds in candidates.seed_sets
            ]
            wm_seconds.append((time.perf_counter() - started) / count)

            orders = {
                "wm_agent": np.argsort(-np.asarray(predicted), kind="stable").tolist(),
                "no_wm_degree": np.argsort(
                    -np.asarray(degree_totals), kind="stable"
                ).tolist(),
                "no_wm_random": rng.permutation(count).tolist(),
                "oracle_agent": np.argsort(-np.asarray(true), kind="stable").tolist(),
            }

            for agent, order in orders.items():
                for budget in range(1, count + 1):
                    records[agent].setdefault(budget, []).append(
                        best_within_budget(order, true, budget)
                    )

        print(f"  {graph_id}: k={budget_k} K={count} "
              f"best_true={max(true):.2f} mean_true={np.mean(true):.2f}")

    after = model.state_dict()
    frozen_ok = all(torch.equal(frozen_before[k], after[k]) for k in frozen_before)
    budgets = sorted(records["oracle_agent"])
    curve = []

    for budget in budgets:
        row = {"budget": budget}
        oracle_values = np.asarray(records["oracle_agent"][budget], dtype=float)
        random_values = np.asarray(records["no_wm_random"][budget], dtype=float)

        for agent in agents:
            values = np.asarray(records[agent][budget], dtype=float)
            row[agent] = float(values.mean())
            row[f"{agent}_sem"] = float(values.std(ddof=1) / np.sqrt(values.size))

            # Paired against the same (graph, seed) draws, which is what makes a
            # 0.5-node difference measurable at all.
            if agent not in ("oracle_agent",):
                paired = values - oracle_values
                row[f"{agent}_gap_to_oracle"] = float(-paired.mean())

            if agent not in ("no_wm_random",):
                paired = values - random_values
                spread = paired.std(ddof=1) / np.sqrt(paired.size)
                row[f"{agent}_vs_random"] = float(paired.mean())
                row[f"{agent}_vs_random_t"] = (
                    float(paired.mean() / spread) if spread else float("nan")
                )

        # Paired WM vs the heuristic prior: the comparison that decides whether
        # consulting the model was worth anything a practitioner did not have.
        paired = np.asarray(records["wm_agent"][budget], dtype=float) - np.asarray(
            records["no_wm_degree"][budget], dtype=float
        )
        spread = paired.std(ddof=1) / np.sqrt(paired.size)
        row["wm_vs_degree"] = float(paired.mean())
        row["wm_vs_degree_t"] = float(paired.mean() / spread) if spread else float("nan")
        row["n_paired"] = int(paired.size)
        curve.append(row)

    blob = {
        "question": "Q4_decision_value_closed_loop",
        "protocol": "identical K candidates per graph; each agent orders them and "
                    "spends B trusted evaluations on its first B; its result is "
                    "the best TRUE value among those B",
        "model": {"backbone": spec.backbone, "head": spec.head,
                  "diffusion_model": diffusion_model,
                  "n_samples": args.n_samples},
        "frozen_parameters_unchanged": frozen_ok,
        "source": {"data_dir": train_meta.get("data_dir")},
        "target": {"data_dir": str(args.data_dir), "split": args.split,
                   "graph_ids": graph_ids,
                   "mean_candidates": float(np.mean(candidate_counts))},
        "seeds": args.seeds,
        "cost": {
            "wm_ms_per_candidate": 1000 * float(np.mean(wm_seconds)),
            "trusted_ms_per_call": 1000 * float(np.mean(trusted_seconds)),
            "wm_trusted_calls": 0,
        },
        "curve": curve,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(blob, indent=2, default=str))

    print(f"\nfrozen parameters unchanged: {frozen_ok}")
    print(f"world-model scoring {1000 * np.mean(wm_seconds):.1f} ms/candidate, "
          f"0 trusted calls; trusted evaluation "
          f"{1000 * np.mean(trusted_seconds):.1f} ms/call\n")
    print(f"{'B':>3s} {'wm_agent':>16s} {'no_wm_degree':>16s} {'no_wm_random':>16s} "
          f"{'oracle':>9s} {'wm-degree':>11s} {'t':>7s}")
    print("-" * 86)
    for row in curve:
        print(f"{row['budget']:3d} "
              f"{row['wm_agent']:9.2f}±{row['wm_agent_sem']:<5.2f} "
              f"{row['no_wm_degree']:9.2f}±{row['no_wm_degree_sem']:<5.2f} "
              f"{row['no_wm_random']:9.2f}±{row['no_wm_random_sem']:<5.2f} "
              f"{row['oracle_agent']:9.2f} "
              f"{row['wm_vs_degree']:+11.3f} {row['wm_vs_degree_t']:7.2f}")

    print(f"\n-> {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
