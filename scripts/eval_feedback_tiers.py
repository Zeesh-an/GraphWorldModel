"""
Experiment 3 — does richer world-model feedback change what a reviser can do?

    python -m scripts.eval_feedback_tiers \
        --wm-results experiments/sbm24/wm/sbm_IC.json \
        --data-dir experiments/sbm24/data \
        --graphs 6 --seeds 0 1 2 --iterations 20 --budget 6 \
        --out experiments/feedback/sbm.json

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
The reviser here is a SCRIPTED policy, not a language model. It is not a stand-in
for the coding agent and no number it produces is a claim about one. What it
isolates is the prior question, and the one an LLM result would otherwise
confound with prompt luck:

    does each tier's feedback CONTAIN the signal a reviser would need to make a
    better next move, at a fixed query budget?

Every tier runs the identical loop — same initial seed set, same number of
iterations, same accept rule, same world model, same trusted-simulator budget
(one final verification). The ONLY difference is which diagnostic blocks the
proposal rule may read, and therefore which move it makes. If f2 beats f1 here,
the regional block carries decision-relevant information that the attribution
block does not; if it does not, the extra text is overhead whatever an agent
would do with it.

FAIRNESS
--------
Graph structure (degree, communities, betweenness) is available to EVERY tier,
because it is available to any agent that has the graph. Proposals are therefore
degree-weighted at every rung, and the ladder adds only world-model-derived
TARGETING: which seed is weak (f1), which region is weak (f2), which seeds are
redundant and which bridges are missed (f3). Without this the ladder would be
measuring "does the reviser know what degree is", which it always does.

COST ACCOUNTING
---------------
Three separate meters, never pooled:

  * `wm_rollouts` — world-model queries: the loop's own evaluations plus every
    diagnostic probe, read off the environment's counters.
  * `trusted_episodes` — real NDlib episodes. Identical across tiers by
    construction (one verification of the final seed set), so the comparison is
    quality at a FIXED trusted budget, which is the decision-value question.
  * `feedback_chars` — the context each tier costs, as a token proxy.

Reference rows (`degree`, `random`) are scored on the same trusted simulator so
the tier column is readable against something.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from coding_agent.diagnostics import PlanDiagnostics, high_overlap, swap_node
from coding_agent.envs.monte_carlo_env import MonteCarloEnvironment
from coding_agent.envs.world_model_env import WorldModelEnvironment
from coding_agent.feedback import build_feedback, f0, f1, f2, f3, resolve
from coding_agent.regions import community_labels
from coding_agent.types import ActionOp, GraphInfo
from world_model.wm_data import load_graph_store

ladder = (f0, f1, f2, f3)

#: The control the ladder is worthless without: f0 again, but run for however
#: many more iterations it takes to spend the SAME number of world-model
#: rollouts f3 spent. Without it "richer feedback wins" cannot be told apart
#: from "more queries win", because a richer tier also queries more per turn.
#: It is `f0` in every other respect and is reported as its own row.
matched_arm = "f0_matched"

#: Proposal temperature: a non-seed is drawn with probability proportional to
#: (degree + 1)^alpha. 1.0 is plain degree-proportional; it is the same at every
#: rung, so the ladder never differs in how much it likes hubs.
degree_alpha = 1.0


def initial_plan(graph: GraphInfo, budget: int, horizon: int) -> list:
    """
    Top-`budget` by degree, committed at t=0.

    Identical for every tier and every repetition, so each arm starts from the
    same point and the comparison is about the REVISIONS.
    """
    ranked = sorted(range(graph.num_nodes), key=lambda node: -graph.degree(node))
    seeds = ranked[:budget]

    return [[ActionOp("add_node", node) for node in seeds]] + [
        [] for _ in range(horizon)
    ]


def degree_weights(graph: GraphInfo, candidates: list[int]) -> np.ndarray:
    weights = np.array(
        [(graph.degree(node) + 1.0) ** degree_alpha for node in candidates],
        dtype=np.float64,
    )

    return weights / weights.sum()


def _sample(rng, candidates: list[int], graph: GraphInfo) -> int | None:
    if not candidates:
        return None

    return int(rng.choice(candidates, p=degree_weights(graph, candidates)))


def propose(
    tier: str,
    diagnosis,
    diagnostics: PlanDiagnostics,
    graph: GraphInfo,
    seeds: list[int],
    rng,
) -> list[tuple[int, int]]:
    """
    (old, new) swaps this tier's information licenses. One swap normally; two
    when f3's stagnation block says the search has stopped moving, which is the
    only place a tier changes the SIZE of its move rather than its target.
    """
    non_seeds = [node for node in range(graph.num_nodes) if node not in set(seeds)]

    if not seeds or not non_seeds:
        return []

    # --- which seed to give up -------------------------------------------
    if tier == f0:
        # A scalar says nothing about which seed is weak; the honest move is a
        # uniform choice among the seeds
        outgoing = [int(rng.choice(seeds))]
    else:
        contributions = diagnosis.contributions
        ordered = sorted(seeds, key=lambda node: contributions.get(node, 0.0))
        outgoing = [ordered[0]]

        if tier == f3 and diagnosis.overlaps:
            pair, value = max(diagnosis.overlaps.items(), key=lambda item: item[1])
            if value >= high_overlap:
                # Of a redundant pair, drop the one contributing less: the two
                # buy the same reach, so the cheaper of them is the free seat
                outgoing = [
                    min(pair, key=lambda node: contributions.get(node, 0.0))
                ]

    # --- what to put in its place ----------------------------------------
    weakest = None
    if tier in (f2, f3) and diagnosis.coverage:
        weakest = min(diagnosis.coverage, key=lambda entry: entry.normalized)

    incoming: list[int] = []

    if tier == f3 and diagnosis.bridges.get("missed"):
        labels = community_labels(graph)
        missed = [entry["node"] for entry in diagnosis.bridges["missed"]]
        in_weak = [
            node
            for node in missed
            if weakest is not None and labels.get(node) == weakest.key
        ]
        # `missed` is already betweenness-ranked, so the first survivor is the
        # most central uncovered bridge into the weakest region
        pool = [node for node in (in_weak or missed) if node not in set(seeds)]

        if pool:
            incoming = [pool[0]]

    if not incoming:
        pool = non_seeds

        if weakest is not None:
            regions = diagnostics.regions()
            in_region = [
                node for node in regions.members[weakest.key] if node not in set(seeds)
            ]
            pool = in_region or non_seeds

        picked = _sample(rng, pool, graph)
        incoming = [] if picked is None else [picked]

    if not incoming:
        return []

    swaps = [(outgoing[0], incoming[0])]

    # f3 alone can see that it has stopped improving, and the documented
    # response to stagnation in this repository's own evolve loop is a bigger
    # move, so it takes one
    stagnating = (diagnosis.stagnation or {}).get("status") == "STAGNATING"

    if tier == f3 and stagnating:
        remaining = [node for node in seeds if node != outgoing[0]]
        contributions = diagnosis.contributions
        if remaining:
            second_out = min(remaining, key=lambda node: contributions.get(node, 0.0))
            second_pool = [
                node
                for node in non_seeds
                if node != incoming[0]
            ]
            second_in = _sample(rng, second_pool, graph)

            if second_in is not None:
                swaps.append((second_out, second_in))

    return swaps


def run_tier(
    tier: str,
    graph: GraphInfo,
    wm_environment,
    trusted,
    budget: int,
    horizon: int,
    iterations: int,
    repetition: int,
    label: str | None = None,
) -> dict:
    """One (graph, repetition, tier) search. Returns its row of the table."""
    policy = resolve(tier)
    rng = np.random.default_rng(1000 + repetition)
    diagnostics = PlanDiagnostics(
        wm_environment,
        graph,
        horizon=horizon,
        budget=budget,
        seed=repetition,
        stagnation_window=5,
        stagnation_epsilon=0.5,
    )

    plan = initial_plan(graph, budget, horizon)
    incumbent = diagnostics.evaluate(plan)
    history = [incumbent.reward]
    trace = [incumbent.reward]
    feedback_chars = 0
    start = time.perf_counter()

    for _ in range(iterations):
        text, diagnosis = build_feedback(
            policy, diagnostics, plan, history=history, sense="maximize"
        )
        feedback_chars += len(text)

        swaps = propose(
            tier, diagnosis, diagnostics, graph, list(incumbent.seeds), rng
        )

        if not swaps:
            history.append(incumbent.reward)
            trace.append(incumbent.reward)
            continue

        candidate_plan = plan
        for old, new in swaps:
            candidate_plan = swap_node(candidate_plan, old, new)

        candidate = diagnostics.evaluate(candidate_plan)

        # Hill climbing on the WORLD MODEL. No trusted episode is spent deciding
        # whether to accept: that is the whole point of having a forward model
        if candidate.reward > incumbent.reward:
            plan, incumbent = candidate_plan, candidate
            diagnostics.reset_plan_cache()

        history.append(incumbent.reward)
        trace.append(incumbent.reward)

    # The one trusted call: the final seed set, scored on NDlib. Identical budget
    # for every tier, so the columns compare quality at a fixed trusted budget
    before_episodes = trusted.episodes_used
    final = trusted.rollout(
        lambda state, timestep: plan[timestep] if timestep < len(plan) else [],
        horizon,
        budget,
    )

    return {
        "tier": label or tier,
        "iterations": iterations,
        "repetition": repetition,
        "seeds": list(incumbent.seeds),
        "wm_final": incumbent.reward,
        "wm_initial": trace[0],
        "trusted_final": float(final.reward),
        "trusted_final_se": float(final.cost.get("reward_se", 0.0)),
        "trusted_episodes": int(trusted.episodes_used - before_episodes),
        "wm_rollouts": diagnostics.total_cost.wm_rollouts,
        "wm_forward_passes": diagnostics.total_cost.wm_forward_passes,
        "diagnostic_simulator_episodes": diagnostics.total_cost.simulator_episodes,
        "feedback_chars": feedback_chars,
        "wm_trace": [round(value, 3) for value in trace],
        "seconds": round(time.perf_counter() - start, 2),
    }


def reference_rows(graph, trusted, budget, horizon) -> dict:
    """Degree and random seed sets on the same trusted simulator, for scale."""
    rows = {}

    for name, seeds in (
        (
            "degree",
            sorted(range(graph.num_nodes), key=lambda node: -graph.degree(node))[:budget],
        ),
        (
            "random",
            list(np.random.default_rng(0).choice(graph.num_nodes, budget, replace=False)),
        ),
    ):
        plan = [[ActionOp("add_node", int(node)) for node in seeds]] + [
            [] for _ in range(horizon)
        ]
        trajectory = trusted.rollout(
            lambda state, timestep, plan=plan: plan[timestep]
            if timestep < len(plan)
            else [],
            horizon,
            budget,
        )
        rows[name] = {
            "seeds": [int(node) for node in seeds],
            "trusted_final": float(trajectory.reward),
        }

    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--wm-results",
        type=Path,
        default=None,
        help="train_wm.py results JSON for the world model. Omit to use the "
        "structured ORACLE head, which isolates the feedback question from "
        "model error (and says so in the output).",
    )
    parser.add_argument("--diffusion-model", type=str, default="IC")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--graphs", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--mc-runs", type=int, default=300)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    store = load_graph_store(args.data_dir)
    graph_ids = sorted(store)[: args.graphs]
    rows, references = [], {}

    for graph_id in graph_ids:
        graph = GraphInfo.from_store_entry(store[graph_id])
        trusted = MonteCarloEnvironment(
            graph, args.diffusion_model, mc_runs=args.mc_runs, base_seed=0
        )
        references[graph_id] = reference_rows(
            graph, trusted, args.budget, args.horizon
        )

        # `graph` / `graph_id` / `repetition` are bound as defaults rather than
        # captured: both helpers are only ever called inside the iteration that
        # defined them, so late binding happens to be correct today, and would
        # stop being correct the moment either is deferred
        def environment(graph=graph, repetition=None):
            if args.wm_results is None:
                return WorldModelEnvironment.oracle(
                    graph,
                    args.diffusion_model,
                    device=args.device,
                    n_samples=args.n_samples,
                    base_seed=repetition,
                )

            return WorldModelEnvironment.from_results_json(
                str(args.wm_results),
                graph,
                device=args.device,
                n_samples=args.n_samples,
                base_seed=repetition,
            )

        def report(row: dict, graph_id=graph_id, repetition=None) -> None:
            row["graph_id"] = graph_id
            rows.append(row)
            print(
                f"[{graph_id} r{repetition}] {row['tier']}: "
                f"wm {row['wm_initial']:.2f} -> {row['wm_final']:.2f}, "
                f"trusted {row['trusted_final']:.2f}, "
                f"wm_rollouts {row['wm_rollouts']}, "
                f"iters {row['iterations']}, "
                f"chars {row['feedback_chars']}"
            )

        for repetition in args.seeds:
            by_tier = {}

            for tier in ladder:
                row = run_tier(
                    tier,
                    graph,
                    environment(repetition=repetition),
                    trusted,
                    args.budget,
                    args.horizon,
                    args.iterations,
                    repetition,
                )
                by_tier[tier] = row
                report(row, repetition=repetition)

            # The equal-query control: f0 for as many iterations as it takes to
            # spend f3's world-model budget
            per_iteration = by_tier[f0]["wm_rollouts"] / max(args.iterations, 1)
            matched_iterations = round(
                by_tier[f3]["wm_rollouts"] / max(per_iteration, 1e-9)
            )
            report(
                run_tier(
                    f0,
                    graph,
                    environment(repetition=repetition),
                    trusted,
                    args.budget,
                    args.horizon,
                    max(matched_iterations, args.iterations),
                    repetition,
                    label=matched_arm,
                ),
                repetition=repetition,
            )

    summary = {}
    for tier in (*ladder, matched_arm):
        subset = [row for row in rows if row["tier"] == tier]
        trusted_values = np.array([row["trusted_final"] for row in subset])
        summary[tier] = {
            "n_runs": len(subset),
            "trusted_final_mean": float(trusted_values.mean()),
            "trusted_final_sd": float(trusted_values.std(ddof=1))
            if len(subset) > 1
            else 0.0,
            "trusted_final_se": float(
                trusted_values.std(ddof=1) / np.sqrt(len(subset))
            )
            if len(subset) > 1
            else 0.0,
            "wm_rollouts_mean": float(
                np.mean([row["wm_rollouts"] for row in subset])
            ),
            "trusted_episodes_mean": float(
                np.mean([row["trusted_episodes"] for row in subset])
            ),
            "feedback_chars_mean": float(
                np.mean([row["feedback_chars"] for row in subset])
            ),
            "iterations_mean": float(np.mean([row["iterations"] for row in subset])),
            "diagnostic_simulator_episodes": int(
                sum(row["diagnostic_simulator_episodes"] for row in subset)
            ),
        }

    payload = {
        "config": {
            "data_dir": str(args.data_dir),
            "wm_results": None if args.wm_results is None else str(args.wm_results),
            "forward_model": "oracle" if args.wm_results is None else "trained",
            "diffusion_model": args.diffusion_model,
            "graphs": graph_ids,
            "repetitions": args.seeds,
            "iterations": args.iterations,
            "budget": args.budget,
            "horizon": args.horizon,
            "n_samples": args.n_samples,
            "mc_runs": args.mc_runs,
            "reviser": "scripted (NOT an LLM); see this module's docstring",
        },
        "summary": summary,
        "references": references,
        "runs": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")

    for tier in (*ladder, matched_arm):
        block = summary[tier]
        print(
            f"{tier}: trusted {block['trusted_final_mean']:.2f} "
            f"± {block['trusted_final_se']:.2f} SE, "
            f"wm_rollouts {block['wm_rollouts_mean']:.0f}, "
            f"trusted episodes {block['trusted_episodes_mean']:.0f}, "
            f"iters {block['iterations_mean']:.0f}, "
            f"feedback {block['feedback_chars_mean']:.0f} chars"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
