"""
Aggregate every run into one flat table, plus a record of the environment that
produced it.

The per-run JSONs are the source of truth, but they are nested and scattered
across two directories and one folder per budget. `summary.csv` flattens all of
it into one row per (arm, budget) so a paper table is a spreadsheet away and no
metric ever has to be recomputed by hand.
"""

import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from coding_agent.types import best_by
from pipeline.conditions import ground_truth_reward, result_sense
from pipeline.layout import Layout

# Flattened in this order; missing keys become empty cells rather than errors
summary_columns = (
    "arm",
    "arm_spec",
    "task",
    # maximize | minimize. Without it a reader has no way to know whether the
    # spread column's smallest or largest value is the good one.
    "objective",
    "method",
    "condition",
    "condition_name",
    "evaluator",
    "budget",
    "budget_pct",
    "graph_id",
    "num_nodes",
    "num_edges",
    "directed",
    "spread_ground_truth",
    "spread_pct_ground_truth",
    "reward_own_evaluator",
    "spread_pct_own_evaluator",
    "estimate_unbiased",
    "fidelity_error",
    "mc_reward",
    "mc_reward_se",
    "mc_rollout_seconds",
    "referee_mc_runs",
    "reward_se",
    "rollout_seconds",
    "n_samples",
    "mc_runs",
    "real_env_episodes",
    "evaluator_calls",
    "evaluator_seconds",
    "forward_passes",
    "llm_calls",
    "llm_prompt_tokens",
    "llm_completion_tokens",
    "llm_total_tokens",
    "llm_cost_usd",
    "elapsed_seconds",
    "seed",
    "outer_iterations",
    "best_iteration",
    "failed_iterations",
    "model",
    "external_selection_seconds",
    "cascade_steps",
    "final_infected_count",
    # Adaptive IM: (k, b, r) together, because the literature splits three ways
    # on the budget convention and a spread is not comparable without all three
    "adaptive",
    "rounds",
    "round_batches",
    "round_gap",
    "feedback_model",
    "round_spreads",
    "spread_at_horizon",
    "streaming",
    "edit_rate",
    "multi_round",
    "campaigns",
    "campaign_rewards",
    # Critical node detection: the outbreak faced, the set removed, and what that
    # set did to connectivity. Empty for every seeding arm.
    "containment",
    "outbreak_pct",
    "outbreak_selector",
    "removed_nodes",
    "pairwise_conn",
    "pairwise_conn_drop_pct",
    "largest_cc_size",
    "largest_cc_drop_pct",
    "gcc_fraction",
    "n_components",
    "schneider_r",
    "anc",
    "anc_sigma",
    "rho_at_threshold",
    "degree_rank_spearman",
)


def _row(result: dict, sense: str = "maximize") -> dict:
    graph = result.get("graph", {})
    cost = result.get("cost", {})
    history = result.get("history") or []
    external = result.get("external") or {}
    timeline = result.get("timeline") or []
    usage = result.get("llm_usage") or {}

    spread = ground_truth_reward(result)
    nodes = graph.get("num_nodes") or 1
    estimate = result.get("wm_reeval_mean", result.get("reward"))
    scored = [entry for entry in history if entry.get("reward") is not None]

    # argmin on a containment task; the search itself already picks its winner
    # this way, so taking the max here would report an iteration it discarded
    best_iteration = (
        best_by(scored, lambda entry: entry["reward"], sense)["iteration"]
        if scored
        else None
    )
    structural = result.get("structural") or {}

    return {
        "arm": result.get("arm"),
        "arm_spec": result.get("arm_spec"),
        "task": result.get("task"),
        "objective": result.get("objective"),
        "method": result.get("method"),
        "condition": result.get("condition"),
        "condition_name": result.get("condition_name"),
        "evaluator": result.get("evaluator"),
        "budget": result.get("budget"),
        "budget_pct": result.get("budget_pct"),
        "graph_id": graph.get("graph_id"),
        "num_nodes": graph.get("num_nodes"),
        "num_edges": graph.get("num_edges"),
        "directed": graph.get("directed"),
        "spread_ground_truth": round(spread, 4),
        "spread_pct_ground_truth": round(100.0 * spread / nodes, 4),
        "reward_own_evaluator": result.get("reward"),
        "spread_pct_own_evaluator": result.get("spread_pct"),
        "estimate_unbiased": estimate,
        # How far this arm's own evaluator was from the shared referee
        "fidelity_error": (
            round(estimate - spread, 4)
            if result.get("mc_reward") is not None and estimate is not None
            else None
        ),
        "mc_reward": result.get("mc_reward"),
        "mc_reward_se": result.get("mc_reward_se"),
        "mc_rollout_seconds": result.get("mc_rollout_seconds"),
        "referee_mc_runs": result.get("referee_mc_runs"),
        "reward_se": cost.get("reward_se"),
        "rollout_seconds": cost.get("rollout_seconds"),
        "n_samples": cost.get("n_samples"),
        "mc_runs": cost.get("mc_runs"),
        # Inner-loop cost, excluding the --credit and --compare post-hoc replays.
        # evaluator_seconds is the cross-condition axis: elapsed_seconds is mostly
        # LLM latency, and rollout_seconds above is only the final rollout.
        "real_env_episodes": result.get("real_env_episodes"),
        "evaluator_calls": result.get("evaluator_calls"),
        "evaluator_seconds": result.get("evaluator_seconds"),
        "forward_passes": result.get("forward_passes"),
        # LLM spend for this arm. cost_usd is None unless --llm-price-in/-out
        # were supplied; the token counts are exact either way.
        "llm_calls": usage.get("calls"),
        "llm_prompt_tokens": usage.get("prompt_tokens"),
        "llm_completion_tokens": usage.get("completion_tokens"),
        "llm_total_tokens": usage.get("total_tokens"),
        "llm_cost_usd": usage.get("cost_usd"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        # Base rollout seed; each rollout's own seed is in its cost block
        "seed": result.get("seed", cost.get("seed")),
        "outer_iterations": len(history) or None,
        "best_iteration": best_iteration,
        "failed_iterations": sum(
            1 for entry in history if entry.get("reward") is None
        )
        or None,
        "model": result.get("model"),
        "external_selection_seconds": external.get("selection_seconds"),
        "cascade_steps": len(timeline) or None,
        "final_infected_count": (
            timeline[-1].get("infected_count") if timeline else None
        ),
        # Empty for every non-adaptive arm, so one table holds both sides of the
        # adaptivity gap without a second file
        "adaptive": result.get("adaptive"),
        "rounds": result.get("rounds"),
        "round_batches": result.get("round_batches"),
        "round_gap": result.get("round_gap"),
        "feedback_model": result.get("feedback_model"),
        "round_spreads": result.get("round_spreads"),
        # sigma(S, T) at T = horizon, readable at any smaller T from spread_curve
        # in the per-arm JSON
        "spread_at_horizon": result.get("spread_at_horizon"),
        "streaming": result.get("streaming"),
        "edit_rate": result.get("edit_rate"),
        "multi_round": result.get("multi_round"),
        "campaigns": result.get("campaigns"),
        "campaign_rewards": result.get("campaign_rewards"),
        # Critical node detection. Empty for every seeding arm, so one table holds
        # both families; the connectivity functionals are DESCRIPTIVE context, never
        # the objective (research/critical_node_detection.md §8.3).
        "containment": result.get("containment"),
        "outbreak_pct": result.get("outbreak_pct"),
        "outbreak_selector": result.get("outbreak_selector"),
        "removed_nodes": structural.get("removed"),
        "pairwise_conn": structural.get("pairwise_conn"),
        "pairwise_conn_drop_pct": structural.get("pairwise_conn_drop_pct"),
        "largest_cc_size": structural.get("largest_cc_size"),
        "largest_cc_drop_pct": structural.get("largest_cc_drop_pct"),
        "gcc_fraction": structural.get("gcc_fraction"),
        "n_components": structural.get("n_components"),
        "schneider_r": structural.get("schneider_r"),
        "anc": structural.get("anc"),
        "anc_sigma": structural.get("anc_sigma"),
        "rho_at_threshold": structural.get("rho_at_threshold"),
        "degree_rank_spearman": structural.get("degree_rank_spearman"),
    }


def write_summary(layout: Layout, results: list[dict]) -> tuple[Path, Path]:
    """One row per (arm, budget), sorted the way the report table reads."""
    os.makedirs(layout.root, exist_ok=True)
    sense = result_sense(results)
    rows = sorted(
        (_row(result, sense) for result in results),
        key=lambda row: (
            row["budget"] or 0,
            row["condition"] or 99,
            row["arm"] or "",
        ),
    )

    csv_path = layout.root / "summary.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_columns))
        writer.writeheader()
        writer.writerows(rows)

    json_path = layout.root / "summary.json"
    json_path.write_text(json.dumps(rows, indent=2, default=str))

    return csv_path, json_path


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return completed.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _git_dirty() -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10
        )
        return bool(completed.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def write_environment(layout: Layout) -> Path:
    """
    Provenance for the run: which code, which machine, which library versions.

    Written once per pipeline invocation so a results tree is self-describing
    months later, when the working copy has moved on.
    """
    import numpy
    import torch

    payload = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }

    path = layout.root / "environment.json"
    os.makedirs(layout.root, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))

    return path
