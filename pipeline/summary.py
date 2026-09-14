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
    "referee",
    "referee_reward",
    "referee_reward_se",
    "referee_rollout_seconds",
    "referee_samples",
    "mc_reward",
    "mc_reward_se",
    "mc_rollout_seconds",
    "mc_agreement_runs",
    "referee_minus_mc",
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
    "transferred_from",
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
    # Influence blocking: the two-cascade columns. `blocking_prevented` is the
    # headline this literature publishes in: every other task's tables report a
    # spread, and this one reports a DIFFERENCE from the unopposed cascade.
    "blocking",
    "blocking_lever",
    "blocking_tie_break",
    "blocking_competitive_model",
    "blocking_attacker",
    "blocking_negative_seeds",
    "blocking_detection_delay",
    "blocking_unopposed",
    "blocking_prevented",
    "blocking_prevented_pct",
    "blocking_prevented_pct_of_nodes",
    "blocking_budget_ratio",
    "blocking_spent",
    "blocking_n_spent",
    # Epidemic control: the compartmental columns. `epi_prevented` is the headline
    # this literature publishes in, and the three curve columns beside it are the
    # SHAPE: §8.2 trap 4 records that a good policy flattens rather than
    # eliminates, so a terminal-state number alone can rank two policies backwards.
    # `epi_eigendrop` is the spectral line's own metric, carried as CONTEXT: §8.2
    # trap 1 is that a method can win it and lose the attack rate.
    "epidemic",
    "epi_compartments",
    "epi_lever",
    "epi_beta",
    "epi_gamma",
    "epi_alpha",
    "epi_outbreak_selector",
    "epi_attack_rate",
    "epi_attack_rate_pct",
    "epi_unprotected",
    "epi_prevented",
    "epi_prevented_pct",
    "epi_peak_prevalence",
    "epi_peak_prevalence_pct",
    "epi_time_to_peak",
    "epi_auc_infectious",
    "epi_endemic_prevalence",
    "epi_lambda1_intact",
    "epi_lambda1",
    "epi_eigendrop",
    "epi_eigendrop_pct",
    "epi_spent",
    "epi_n_spent",
    # Source localization: the inverse task's own metric set. Empty for every arm
    # that intervenes, so one table still holds all four runnable tasks.
    # `sl_f1` is the held-out score and duplicates `spread_ground_truth` on
    # purpose: the shared column keeps cross-task readers working, and the named
    # one keeps a spreadsheet from calling an F1 a spread.
    "localization",
    "sl_reward",
    "sl_reward_referee",
    "sl_f1",
    "sl_precision",
    "sl_recall",
    "sl_auc",
    "sl_accuracy",
    "sl_f1_selection",
    "sl_generalization_gap",
    "sl_f1_generalization_gap",
    "sl_auc_source",
    "sl_observation_mode",
    "sl_budget_mode",
    "sl_select_split",
    "sl_eval_split",
    "sl_select_instances",
    "sl_eval_instances",
    "sl_forward_calls",
    "sl_forward_calls_per_instance",
    "sl_transfer_from",
    "sl_resim_error",
    "sl_resim_error_true_sources",
    # Cascade reconstruction: the decoding task's own metric set. Empty for every
    # other arm, so one table still holds all six runnable tasks. `cr_score` is the
    # held-out tree-weighted score and duplicates `spread_ground_truth` on purpose,
    # for the same reason `sl_f1` does: the shared column keeps cross-task readers
    # working and the named one keeps a spreadsheet from calling a score a spread.
    "reconstruction",
    "cr_reward",
    "cr_reward_referee",
    "cr_loglik_per_node",
    "cr_consistency",
    "cr_tree_score",
    "cr_path_precision",
    "cr_path_recall",
    "cr_jaccard",
    "cr_order_accuracy",
    "cr_event_f1",
    "cr_event_precision",
    "cr_event_recall",
    "cr_node_f1",
    "cr_mcc",
    "cr_time_mae",
    "cr_time_nrmse",
    "cr_source_f1",
    "cr_n_tree_edges",
    "cr_reward_selection",
    "cr_generalization_gap",
    "cr_tree_score_generalization_gap",
    "cr_setting",
    "cr_observation_rate",
    "cr_hidden_rate",
    "cr_tree_weight",
    "cr_has_tree_truth",
    "cr_trivial_decoder_reward",
    "cr_select_split",
    "cr_eval_split",
    "cr_select_instances",
    "cr_eval_instances",
    "cr_kernel_calls",
    "cr_kernel_calls_per_instance",
    "cr_resim_error",
    "cr_resim_error_true_sources",
    # Cascade prediction: the forecasting task's own metric set. Empty for every
    # other arm, so one table still holds all seven runnable tasks. `cp_error`
    # duplicates `spread_ground_truth` for the same reason `cr_score` and `sl_f1`
    # do: the shared column keeps cross-task readers working and the named one
    # keeps a spreadsheet from calling an ERROR a spread. Every cp_* error column
    # runs LOWER-is-better; every cp_* correlation column runs higher.
    "prediction",
    "cp_error",
    "cp_metric",
    "cp_msle",
    "cp_male",
    "cp_msle_offset",
    "cp_msle_natural",
    "cp_msle_increment",
    "cp_mape",
    "cp_mape_casft",
    "cp_mrse",
    "cp_mrse_median",
    "cp_wroperc",
    "cp_ape_median",
    "cp_ape_p75",
    "cp_ape_p95",
    "cp_pcc",
    "cp_r2",
    "cp_coverage",
    "cp_doubling_accuracy",
    # The column §8.4 says almost nobody publishes: a mean over SCOREABLE cascades
    # silently favours whoever gives up more often
    "cp_n_scored",
    "cp_n_failed",
    "cp_decline_rate",
    "cp_mean_predicted",
    "cp_mean_actual",
    "cp_error_selection",
    "cp_generalization_gap",
    # The two floors a reader needs to interpret any of the above
    "cp_trivial_error",
    "cp_persistence_error",
    # The protocol. Four of §5.7's five incompatibilities live here, and a number
    # without them is comparable to nothing.
    "cp_corpus",
    "cp_time_unit",
    "cp_target",
    "cp_split_protocol",
    "cp_observation_window",
    "cp_prediction_horizon",
    "cp_observation_seconds",
    "cp_horizon_seconds",
    "cp_min_observed_filter",
    "cp_truncate_filter",
    "cp_hard_targets",
    "cp_select_split",
    "cp_eval_split",
    "cp_select_instances",
    "cp_eval_instances",
    "cp_forecast_calls",
    "cp_kernel_calls",
    "cp_kernel_calls_per_instance",
    # MODELLING error: what the arm's own forward model predicts with no program in
    # the loop, which is the falsification number §9.1 is about
    "cp_model_msle",
    "cp_model_male",
    "cp_model_pcc",
    # The search protocol's own bookkeeping (coding_agent/search_metrics.py,
    # coding_agent/provenance.py): agent arms only, empty on every canned row
    "accepted_generations",
    "lucky_accepts_prevented",
    "ties_kept",
    "band_rule",
    "in_loop_optimism_mean",
    "in_loop_optimism_final",
    "calibration_n",
    "calibration_r",
    "calibration_sign_accuracy",
    "calibration_brier",
    "probe_turns",
    "probe_calls",
    "probe_seconds",
    "nearest_library",
    "nearest_similarity",
    "library_calls",
    "stopped_early",
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
    metrics = result.get("metrics") or {}
    selection_metrics = result.get("selection_metrics") or {}
    # The spectral block is present only under a NODE lever: the edge levers spend
    # arcs, and `immunization_metrics` has nothing to delete
    spectral = result.get("spectral") or {}
    # ...and the referee curve only exists once the referee ran, so the arm's own
    # is the fallback rather than the preference
    epidemic_curve = result.get("referee_curve") or {}
    ledger = result.get("acceptance_ledger") or {}
    calibration = result.get("calibration") or {}
    optimism = result.get("in_loop_optimism") or {}
    provenance = result.get("provenance") or {}

    return {
        "arm": result.get("arm"),
        "arm_spec": result.get("arm_spec"),
        "task": result.get("task"),
        "objective": result.get("objective"),
        "method": result.get("method"),
        "condition": result.get("condition"),
        "condition_name": result.get("condition_name"),
        "transferred_from": result.get("transferred_from"),
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
            if result.get("referee_reward") is not None and estimate is not None
            else None
        ),
        "referee": result.get("referee"),
        "referee_reward": result.get("referee_reward"),
        "referee_reward_se": result.get("referee_reward_se"),
        "referee_rollout_seconds": result.get("referee_rollout_seconds"),
        "referee_samples": result.get("referee_samples"),
        # The NDlib agreement check, when --mc-agreement ran: the independent
        # reference the oracle referee is measured against, and the timing row
        "mc_reward": result.get("mc_reward"),
        "mc_reward_se": result.get("mc_reward_se"),
        "mc_rollout_seconds": result.get("mc_rollout_seconds"),
        "mc_agreement_runs": result.get("mc_agreement_runs"),
        "referee_minus_mc": result.get("referee_minus_mc"),
        "reward_se": cost.get("reward_se"),
        "rollout_seconds": cost.get("rollout_seconds"),
        "n_samples": cost.get("n_samples"),
        "mc_runs": cost.get("mc_runs"),
        # Inner-loop cost, excluding the --credit, referee and agreement replays.
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
        # Influence blocking. The `mc_*` prevented columns are preferred over the
        # arm's own because prevented influence is a DIFFERENCE, and a difference of
        # two evaluators' numbers is not a quantity: the shared referee measures
        # both terms or neither.
        "blocking": result.get("blocking"),
        "blocking_lever": result.get("lever"),
        "blocking_tie_break": result.get("tie_break"),
        "blocking_competitive_model": result.get("competitive_model"),
        "blocking_attacker": result.get("attacker"),
        "blocking_negative_seeds": result.get("n_negative_seeds"),
        "blocking_detection_delay": result.get("detection_delay"),
        "blocking_unopposed": result.get(
            "referee_unopposed_spread", result.get("unopposed_spread")
        ),
        "blocking_prevented": result.get(
            "referee_prevented_influence", result.get("prevented_influence")
        ),
        "blocking_prevented_pct": result.get(
            "referee_prevented_pct_of_unopposed", result.get("prevented_pct_of_unopposed")
        ),
        "blocking_prevented_pct_of_nodes": result.get("prevented_pct_of_nodes"),
        "blocking_budget_ratio": result.get("budget_ratio"),
        "blocking_spent": result.get("spent") if result.get("blocking") else None,
        "blocking_n_spent": result.get("n_spent") if result.get("blocking") else None,
        # Epidemic control. The `mc_*` prevented columns are preferred over the
        # arm's own for the same reason blocking's are: prevented infections is a
        # DIFFERENCE, and a difference of two evaluators' numbers is not a quantity.
        # The curve columns likewise prefer the referee's, so peak and time-to-peak
        # describe the ground-truth outbreak rather than the model's picture of it.
        "epidemic": result.get("epidemic"),
        "epi_compartments": result.get("compartments"),
        "epi_lever": result.get("lever") if result.get("epidemic") else None,
        "epi_beta": result.get("epi_beta"),
        "epi_gamma": result.get("epi_gamma"),
        "epi_alpha": result.get("epi_alpha"),
        "epi_outbreak_selector": result.get("outbreak_selector"),
        "epi_attack_rate": result.get(
            "referee_reward", result.get("attack_rate")
        ) if result.get("epidemic") else None,
        "epi_attack_rate_pct": result.get("attack_rate_pct"),
        "epi_unprotected": result.get(
            "referee_unprotected_attack_rate", result.get("unprotected_attack_rate")
        ),
        "epi_prevented": result.get(
            "referee_prevented_infections", result.get("prevented_infections")
        ),
        "epi_prevented_pct": result.get(
            "referee_prevented_pct_of_unprotected",
            result.get("prevented_pct_of_unprotected"),
        ),
        "epi_peak_prevalence": epidemic_curve.get(
            "peak_prevalence", result.get("peak_prevalence")
        ),
        "epi_peak_prevalence_pct": epidemic_curve.get(
            "peak_prevalence_pct", result.get("peak_prevalence_pct")
        ),
        "epi_time_to_peak": epidemic_curve.get(
            "time_to_peak", result.get("time_to_peak")
        ),
        "epi_auc_infectious": epidemic_curve.get(
            "auc_infectious", result.get("auc_infectious")
        ),
        "epi_endemic_prevalence": epidemic_curve.get(
            "endemic_prevalence", result.get("endemic_prevalence")
        ),
        "epi_lambda1_intact": spectral.get("lambda1_intact"),
        "epi_lambda1": spectral.get("lambda1"),
        "epi_eigendrop": spectral.get("eigendrop"),
        "epi_eigendrop_pct": spectral.get("eigendrop_pct"),
        "epi_spent": result.get("spent") if result.get("epidemic") else None,
        "epi_n_spent": result.get("n_spent") if result.get("epidemic") else None,
        # Source localization. Empty for every seeding or containment arm; the
        # metrics are exact (F1 against a known source set carries no evaluator
        # noise), so there is no fidelity column here and none is expected.
        "localization": result.get("localization"),
        # The reward is label-free consistency on the arm's own evaluator and
        # `sl_reward_referee` its re-measurement on the referee; F1 is
        # the reported metric against the stored sources, computed after the search
        "sl_reward": result.get("reward") if result.get("localization") else None,
        "sl_reward_referee": (
            result.get("referee_reward") if result.get("localization") else None
        ),
        "sl_f1": metrics.get("f1"),
        "sl_precision": metrics.get("precision"),
        "sl_recall": metrics.get("recall"),
        "sl_auc": metrics.get("auc"),
        "sl_accuracy": metrics.get("accuracy"),
        "sl_f1_selection": selection_metrics.get("f1"),
        "sl_generalization_gap": (
            result.get("generalization_gap") if result.get("localization") else None
        ),
        "sl_f1_generalization_gap": result.get("f1_generalization_gap"),
        "sl_auc_source": result.get("auc_source"),
        "sl_observation_mode": result.get("observation_mode"),
        "sl_budget_mode": result.get("source_budget_mode"),
        "sl_select_split": result.get("select_split"),
        "sl_eval_split": result.get("eval_split"),
        "sl_select_instances": result.get("n_select_instances"),
        "sl_eval_instances": result.get("n_eval_instances"),
        "sl_forward_calls": result.get("forward_calls"),
        "sl_forward_calls_per_instance": result.get("forward_calls_per_instance"),
        "sl_transfer_from": result.get("transfer_from"),
        "sl_resim_error": (
            result.get("resim_error") if not result.get("reconstruction") else None
        ),
        "sl_resim_error_true_sources": (
            result.get("resim_error_true_sources")
            if not result.get("reconstruction")
            else None
        ),
        # Cascade reconstruction. Empty for every other arm. The tree columns are
        # the ones to read: `cr_path_precision` is what DIPT publishes and
        # `cr_event_f1` what DITTO does, and the reward is a weighted sum of the two
        # (research/cascade_reconstruction.md §2.6). `cr_path_recall` and
        # `cr_jaccard` ride along because PathPrecision alone rewards naming FEW
        # edges, which is the under-prediction corner a search would otherwise find.
        "reconstruction": result.get("reconstruction"),
        # The reward is the label-free kernel likelihood per node minus observation
        # violations on the arm's own kernel, `cr_reward_referee` its re-measurement
        # under NDlib's kernel; the tree score is the reported label metric
        "cr_reward": result.get("reward") if result.get("reconstruction") else None,
        "cr_reward_referee": (
            result.get("referee_reward") if result.get("reconstruction") else None
        ),
        "cr_loglik_per_node": metrics.get("loglik_per_node"),
        "cr_consistency": metrics.get("consistency"),
        "cr_tree_score": metrics.get("tree_score"),
        "cr_path_precision": metrics.get("path_precision"),
        "cr_path_recall": metrics.get("path_recall"),
        "cr_jaccard": metrics.get("jaccard"),
        "cr_order_accuracy": metrics.get("order_accuracy"),
        "cr_event_f1": metrics.get("event_f1"),
        "cr_event_precision": metrics.get("event_precision"),
        "cr_event_recall": metrics.get("event_recall"),
        "cr_node_f1": metrics.get("node_f1"),
        "cr_mcc": metrics.get("mcc"),
        "cr_time_mae": metrics.get("time_mae"),
        "cr_time_nrmse": metrics.get("time_nrmse"),
        "cr_source_f1": metrics.get("source_f1"),
        "cr_n_tree_edges": metrics.get("n_tree_edges"),
        "cr_reward_selection": (
            selection_metrics.get("reward") if result.get("reconstruction") else None
        ),
        "cr_generalization_gap": (
            result.get("generalization_gap") if result.get("reconstruction") else None
        ),
        "cr_tree_score_generalization_gap": result.get("tree_score_generalization_gap"),
        "cr_setting": result.get("observation_setting"),
        "cr_observation_rate": result.get("observation_rate"),
        "cr_hidden_rate": result.get("hidden_rate"),
        "cr_tree_weight": result.get("tree_weight"),
        "cr_has_tree_truth": result.get("has_tree_truth"),
        # §2.11 risk 1's required check, in the same row as the result it
        # qualifies: a trivial decoder has to score badly or the reward is wrong
        "cr_trivial_decoder_reward": result.get("trivial_decoder_reward"),
        "cr_select_split": (
            result.get("select_split") if result.get("reconstruction") else None
        ),
        "cr_eval_split": (
            result.get("eval_split") if result.get("reconstruction") else None
        ),
        "cr_select_instances": (
            result.get("n_select_instances") if result.get("reconstruction") else None
        ),
        "cr_eval_instances": (
            result.get("n_eval_instances") if result.get("reconstruction") else None
        ),
        # The cost axis §2.4.2 exists to measure and §11 says nobody has published
        "cr_kernel_calls": result.get("kernel_calls"),
        "cr_kernel_calls_per_instance": result.get("kernel_calls_per_instance"),
        "cr_resim_error": (
            result.get("resim_error") if result.get("reconstruction") else None
        ),
        "cr_resim_error_true_sources": (
            result.get("resim_error_true_sources")
            if result.get("reconstruction")
            else None
        ),
        # Cascade prediction. Empty for every other arm. `cp_error` is the reward
        # and it MINIMIZES; `cp_n_failed` is the decline count, which travels beside
        # every error because a mean over scoreable cascades alone favours the model
        # that gives up more often (research/cascade_prediction.md §8.4).
        "prediction": result.get("prediction"),
        "cp_error": spread if result.get("prediction") else None,
        "cp_metric": result.get("prediction_metric"),
        "cp_msle": metrics.get("msle"),
        "cp_male": metrics.get("male"),
        "cp_msle_offset": metrics.get("msle_offset"),
        "cp_msle_natural": metrics.get("msle_natural"),
        "cp_msle_increment": metrics.get("msle_increment"),
        "cp_mape": metrics.get("mape"),
        "cp_mape_casft": metrics.get("mape_casft"),
        "cp_mrse": metrics.get("mrse"),
        "cp_mrse_median": metrics.get("mrse_median"),
        "cp_wroperc": metrics.get("wroperc"),
        "cp_ape_median": metrics.get("ape_median"),
        "cp_ape_p75": metrics.get("ape_p75"),
        "cp_ape_p95": metrics.get("ape_p95"),
        "cp_pcc": metrics.get("pcc"),
        "cp_r2": metrics.get("r2"),
        "cp_coverage": metrics.get("coverage"),
        "cp_doubling_accuracy": metrics.get("doubling_accuracy"),
        "cp_n_scored": metrics.get("n_scored"),
        "cp_n_failed": metrics.get("n_failed"),
        "cp_decline_rate": metrics.get("decline_rate"),
        "cp_mean_predicted": metrics.get("mean_predicted"),
        "cp_mean_actual": metrics.get("mean_actual"),
        "cp_error_selection": (
            selection_metrics.get(result.get("prediction_metric", "msle"))
            if result.get("prediction")
            else None
        ),
        "cp_generalization_gap": (
            result.get("generalization_gap") if result.get("prediction") else None
        ),
        "cp_trivial_error": result.get("trivial_predictor_error"),
        "cp_persistence_error": result.get("persistence_error"),
        "cp_corpus": result.get("corpus"),
        "cp_time_unit": result.get("corpus_time_unit"),
        "cp_target": result.get("prediction_target"),
        "cp_split_protocol": result.get("split_protocol"),
        "cp_observation_window": result.get("observation_window"),
        "cp_prediction_horizon": result.get("prediction_horizon"),
        "cp_observation_seconds": result.get("observation_seconds"),
        "cp_horizon_seconds": result.get("horizon_seconds"),
        "cp_min_observed_filter": result.get("min_observed_filter"),
        "cp_truncate_filter": result.get("truncate_filter"),
        "cp_hard_targets": result.get("hard_targets"),
        "cp_select_split": (
            result.get("select_split") if result.get("prediction") else None
        ),
        "cp_eval_split": (
            result.get("eval_split") if result.get("prediction") else None
        ),
        "cp_select_instances": (
            result.get("n_select_instances") if result.get("prediction") else None
        ),
        "cp_eval_instances": (
            result.get("n_eval_instances") if result.get("prediction") else None
        ),
        "cp_forecast_calls": (
            result.get("forecast_calls") if result.get("prediction") else None
        ),
        "cp_kernel_calls": (
            result.get("kernel_calls") if result.get("prediction") else None
        ),
        "cp_kernel_calls_per_instance": (
            result.get("kernel_calls_per_instance")
            if result.get("prediction")
            else None
        ),
        "cp_model_msle": result.get("model_msle"),
        "cp_model_male": result.get("model_male"),
        "cp_model_pcc": result.get("model_pcc"),
        "accepted_generations": ledger.get("band_accepts"),
        "lucky_accepts_prevented": ledger.get("lucky_accepts_prevented"),
        "ties_kept": ledger.get("ties_kept"),
        "band_rule": "+".join(ledger.get("band_rules") or []) or None,
        "in_loop_optimism_mean": optimism.get("mean"),
        "in_loop_optimism_final": optimism.get("final"),
        "calibration_n": calibration.get("n"),
        "calibration_r": calibration.get("pearson_r"),
        "calibration_sign_accuracy": calibration.get("sign_accuracy"),
        "calibration_brier": calibration.get("brier"),
        "probe_turns": result.get("probe_turns"),
        "probe_calls": result.get("probe_calls"),
        "probe_seconds": result.get("probe_seconds"),
        "nearest_library": provenance.get("nearest"),
        "nearest_similarity": provenance.get("nearest_similarity"),
        "library_calls": "+".join(provenance.get("calls") or []) or None,
        "stopped_early": result.get("stopped_early"),
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
