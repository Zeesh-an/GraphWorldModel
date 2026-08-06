"""One markdown file per dataset: config, results tables, figures, winning program."""

import json
import re
from pathlib import Path

from coding_agent.types import best_by, rank_by
from pipeline.conditions import (
    adaptivity_gaps,
    condition_names,
    ground_truth_reward,
    is_forecast,
    is_ground_truth,
    is_reconstruct,
    is_recover,
    result_sense,
    reward_direction,
    reward_name,
)
from pipeline.layout import Layout
from pipeline.tasks import minimize

# The agent writes its own `##` headings; demoting them one level keeps the
# report's outline intact instead of the write-up opening new top-level sections
markdown_heading = re.compile(r"^(#{1,5} )", re.MULTILINE)

# Config keys worth printing; the manifest JSON holds the exhaustive version
reported_config_keys = (
    "task",
    "dataset",
    "run",
    "evaluator",
    "diffusion_model",
    "llm_model",
    "outer_iters",
    "horizon",
    "n_samples",
    "mc_runs",
    "native_mc_runs",
    "allowed_ops",
    "rounds",
    "round_gap",
    "feedback_model",
    "edit_rate",
    "campaigns",
    "outbreak_pct",
    "outbreak_selector",
    "sl_select_split",
    "sl_eval_split",
    "sl_instances",
    "sl_observation",
    "sl_budget_mode",
    "sl_prior",
    "sl_steps",
    "sl_transfer_from",
    "cr_setting",
    "cr_observation_rate",
    "cr_hidden_rate",
    "cr_instances",
    "cr_select_split",
    "cr_eval_split",
    "cr_tree_weight",
    "cr_mcmc_proposals",
    "wm_model",
    "head",
    "hide_edge_weights",
    "seed",
)


def _format_number(value: object, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}" if isinstance(value, (int, float)) else "—"


def _graph_section(agent_results: list[dict], metadata: dict | None) -> list[str]:
    if not agent_results:
        return ["_No agent runs yet._", ""]

    graph = agent_results[0]["graph"]
    lines = [
        "## Graph",
        "",
        f"- **id**: `{graph.get('graph_id')}`",
        f"- **nodes**: {graph['num_nodes']:,}",
        f"- **arcs**: {graph['num_edges']:,}",
        f"- **directed**: {graph.get('directed')}",
    ]

    if metadata:
        episodes = metadata.get("n_episodes")
        seconds = metadata.get("generation_seconds")
        lines.append(
            f"- **generated episodes**: {episodes} in {_format_number(seconds, 1)}s"
        )

        for entry in metadata.get("graphs", [])[:1]:
            lines.append(
                f"- **generation budget band**: k {entry.get('budget_k_min')}–"
                f"{entry.get('budget_k_max')} "
                f"({entry.get('budget_pct_min')}–{entry.get('budget_pct_max')}% of N)"
            )

    return lines + [""]


def _taxonomy_section(agent_results: list[dict]) -> list[str]:
    """Which conditions this run actually covers, and what each one isolates."""
    present = sorted({result.get("condition", 99) for result in agent_results})
    if not present:
        return []

    isolates = {
        1: "classical floor — fixed expert algorithms, no LLM anywhere",
        2: "is *choosing* from a pool enough, versus *generating* code?",
        3: "do the gains come merely from having a coding agent?",
        4: "does simulated lookahead by itself explain the gain?",
        5: "ceiling of model-based guidance — a perfect internal model",
        6: "does the *learned* model recover the true dynamics?",
        7: "the original authors' code, seeds scored by our referee",
        8: "what program *search* buys over per-instance inversion of the same model",
    }

    lines = [
        "## Conditions covered",
        "",
        "| # | condition | arms | what it isolates |",
        "| --- | --- | --- | --- |",
    ]

    for condition in present:
        arms = sorted(
            {
                result["arm"]
                for result in agent_results
                if result.get("condition") == condition
            }
        )
        lines.append(
            f"| {condition} | {condition_names.get(condition, '—')} "
            f"| {', '.join(f'`{arm}`' for arm in arms)} "
            f"| {isolates.get(condition, '—')} |"
        )

    return lines + [""]


def _localization_table(agent_results: list[dict]) -> list[str]:
    """
    PR / RE / F1 / AUC per arm, on HELD-OUT episodes, plus the cost each one paid.

    The inverse task's results table, and it reads differently from the other two
    in one important way: F1 against a known source set carries no evaluator noise,
    so there is no ground-truth referee to fall back on and no fidelity column to
    report. Every number here is exact.
    """
    first = agent_results[0]
    lines = [
        "**F1 is the headline.** SL-VAE calls it \"the most commonly used\" metric "
        "and IVGD \"the most important metric for performance evaluation\"; AUC is "
        "the tie-breaker, added because sources are a tiny positive class. "
        "**Accuracy is near-useless alone** — IVGD's Table 3 has GCNSI at `ACC "
        "0.8840` with `F1 0.0218`, so it is reported only beside F1 "
        "([`research/source_localization.md`](../../../../research/source_localization.md) "
        "§8.1).",
        "",
        f"Every row is scored on the **held-out** `{first.get('eval_split', '?')}` "
        f"episodes, by re-running that arm's winning program unmodified. The "
        f"`selection F1` column is what it scored on the "
        f"`{first.get('select_split', '?')}` episodes the outer loop actually "
        f"optimized against, and `gap` is the difference — a large negative gap "
        f"means the program memorized specific cascades rather than learning an "
        f"algorithm, which is the failure §8.5.1 exists to catch.",
        "",
        f"Observation: `{first.get('observation_mode', '?')}`. The MC marginal is "
        f"strictly MORE informative than the single binary realization the "
        f"published protocol observes, so a `marginal` table is optimistic against "
        f"§5.1 and only the `binary` one is comparable (§2.9 risk 5).",
        "",
        "Warning: **The `AUC from` column decides whether two AUCs are comparable.** An "
        "arm that supplied a per-node ranking (`source_scores`) is scored on that "
        "ranking; one that supplied only a SET (`rank_derived`, which every "
        "external repo is, because a set is all that crosses the process "
        "boundary) has every un-nominated node tied and lands near 0.5 whatever "
        "its quality. Compare AUC only within one value of that column. **F1 is "
        "exact for every row** and is the column to read across them.",
        "",
        "| # | k | arm | evaluator | F1 (higher is better) | PR | RE | AUC | ACC | "
        "selection F1 | gap | AUC from | fwd calls / instance | eval s | total s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    # Sorted and LABELLED by budget: `--sl-budget-mode sweep` runs the same arm at
    # several source fractions, and without the k column those rows are three
    # identical arm names with different numbers
    ordered = sorted(
        agent_results,
        key=lambda result: (
            result.get("budget", 0),
            result.get("condition", 99),
            result["arm"],
        ),
    )

    for result in ordered:
        metrics = result.get("metrics") or {}
        selection = result.get("selection_metrics") or {}
        gap = result.get("generalization_gap")
        # Arm A pays gradient steps rather than forward-oracle calls, and printing
        # a 0 there would read as "this arm is free"
        per_instance = result.get("forward_calls_per_instance")
        if result.get("gradient_steps_per_instance"):
            per_instance = f"{result['gradient_steps_per_instance']:g} (Adam steps)"

        lines.append(
            f"| {result.get('condition', '—')} | {result.get('budget', '—')} "
            f"| `{result['arm']}` "
            f"| `{result.get('evaluator', '—')}` "
            f"| {_format_number(metrics.get('f1'), 4)} "
            f"| {_format_number(metrics.get('precision'), 4)} "
            f"| {_format_number(metrics.get('recall'), 4)} "
            f"| {_format_number(metrics.get('auc'), 4)} "
            f"| {_format_number(metrics.get('accuracy'), 4)} "
            f"| {_format_number(selection.get('f1'), 4)} "
            f"| {'—' if gap is None else f'{gap:+.4f}'} "
            f"| `{result.get('auc_source', '—')}` "
            f"| {per_instance if per_instance is not None else '—'} "
            f"| {_format_number(result.get('evaluator_seconds'), 1)} "
            f"| {_format_number(result.get('elapsed_seconds'), 1)} |"
        )

    if any(result.get("resim_error") is not None for result in ordered):
        lines += [
            "",
            "### Re-simulated error",
            "",
            "Re-run the ground-truth simulator from each arm's RECOVERED sources "
            "and compare against what was observed. **No surveyed paper reports "
            "this at all** (§11), so the column is self-contained and is not a "
            "cross-paper comparison. It is reported beside the TRUE source set's "
            "own error because the number is unreadable without it: on an ill-posed "
            "problem a recovered set can reproduce `y` better than the truth did, "
            "which is exactly why F1 and not this is the selection signal (§2.3.3).",
            "",
            "| arm | resim error (recovered) | resim error (true sources) | ratio |",
            "| --- | --- | --- | --- |",
        ]

        for result in ordered:
            recovered = result.get("resim_error")
            truth = result.get("resim_error_true_sources")
            if recovered is None:
                continue

            ratio = recovered / truth if truth else None
            lines.append(
                f"| `{result['arm']}` | {_format_number(recovered, 5)} "
                f"| {_format_number(truth, 5)} "
                f"| {'—' if ratio is None else f'{ratio:.3f}'} |"
            )

    transferred = [result for result in ordered if result.get("transfer_from")]
    if transferred:
        lines += [
            "",
            "### Transferred programs",
            "",
            "These rows ran a program selected on a DIFFERENT graph, unmodified. "
            "That is the graph axis of §8.5.1 and the headline of the amortization "
            "claim — and it is a comparison no per-instance method can enter, "
            "because SL-VAE, IVGD and DDMSL have no artifact to transfer.",
            "",
            "| arm | selected on | F1 here |",
            "| --- | --- | --- |",
        ]
        lines += [
            f"| `{result['arm']}` | `{result['transfer_from']}` "
            f"| {_format_number((result.get('metrics') or {}).get('f1'), 4)} |"
            for result in transferred
        ]

    return lines + [""]


def _prediction_table(agent_results: list[dict]) -> list[str]:
    """
    Every error column per arm, on HELD-OUT cascades, with the protocol above it.

    The forecasting task's results table. Two things it must do that no other table
    here does. First, it prints the PROTOCOL — corpus, window, horizon, split,
    filters — because
    [`research/cascade_prediction.md`](../../../../research/cascade_prediction.md)
    §5.7 lists five independent incompatibilities between published tables and four
    of them are protocol rather than method: three different corpora are called
    "Twitter", a `< 10` versus `< 50` participant filter moves MSLE by more than the
    gap between any two consecutive published rows, and the random-over-cascades
    split LEAKS. A number without its protocol is comparable to nothing.

    Second, it prints the DECLINE COUNT beside every error. §8.4: generative models
    refuse to score supercritical cascades, papers report the mean over scoreable
    cascades only, and that "silently favours the model that gives up more often".
    Mishra et al. publish their failure counts and almost nobody else does.
    """
    scored = [result for result in agent_results if result.get("prediction")]
    if not scored:
        return []

    first = scored[0]
    metric = str(first.get("prediction_metric", "msle")).upper()
    trivial = next(
        (
            result["trivial_predictor_error"]
            for result in scored
            if result.get("trivial_predictor_error") is not None
        ),
        None,
    )
    persistence = next(
        (
            result["persistence_error"]
            for result in scored
            if result.get("persistence_error") is not None
        ),
        None,
    )

    lines = [
        f"> **`{metric}` is an ERROR and LOWER IS BETTER** — the only column in this "
        f"pipeline that runs that way for a reason unrelated to containment. It is "
        f"measured in LOG space, so being off by a factor of two costs the same on a "
        f"cascade of 20 and one of 2000, and RELATIVE accuracy is the whole game.",
        "",
        "**These cascades are REAL, not simulated.** That is the point of the task "
        "and the one thing that makes it different from every other result in this "
        "repo: every other task evaluates a learned model against traces drawn from "
        "the same NDlib simulator that trained it, a closed loop that can only "
        "measure LEARNING error. Here the dynamics that produced the data are "
        "whatever they are, and our structured head's Independent-Cascade "
        "composition rule is either an adequate approximation of them or it is not "
        "(§9.1). Expect to lose to a method built for this task; the value is "
        "diagnostic.",
        "",
        "### Protocol",
        "",
        "| setting | value |",
        "| --- | --- |",
        f"| corpus | `{first.get('corpus')}` |",
        f"| observation window `t_o` | "
        f"`{first.get('observation_seconds')}` {first.get('corpus_time_unit')}(s) "
        f"= {first.get('observation_window')} replayed step(s) |",
        f"| prediction horizon `t_p` | "
        f"`{first.get('horizon_seconds')}` {first.get('corpus_time_unit')}(s) "
        f"= {first.get('prediction_horizon')} replayed step(s) |",
        f"| split protocol | `{first.get('split_protocol')}` |",
        f"| target quantity | `{first.get('prediction_target')}` |",
        f"| participant filter | drop `< {first.get('min_observed_filter')}` "
        f"observed, keep first `{first.get('truncate_filter')}` |",
        "| targets | "
        + ("HARD 0/1 (a real cascade happened once)" if first.get("hard_targets") else "soft")
        + " |",
        f"| cascades | {first.get('n_select_instances')} selection / "
        f"{first.get('n_eval_instances')} held out |",
        "",
    ]

    if first.get("split_protocol") == "random":
        lines += [
            "> **Warning: this run used the RANDOM split over cascades.** §8.3: "
            "cascades overlap in wall-clock time, so a training cascade's prediction "
            "window can sit inside a test cascade's observation window, and the model "
            "learns a global temporal shortcut unavailable at deployment. Under the "
            "leak-free chronological fix, two 2021-24 SOTA methods fell BELOW a plain "
            "MLP and the field's APS band moved from 1.19-2.11 to 2.28-4.82. These "
            "numbers are comparable to the published tables and NOT to a leak-free "
            "run.",
            "",
        ]

    if trivial is not None or persistence is not None:
        lines += [
            "**Two floors, and both are stronger than they sound.** Under a "
            "log-space error the geometric mean of the training sizes is the best "
            "instance-BLIND prediction"
            + (f" (`{metric} {trivial:.4f}`)" if trivial is not None else "")
            + ", and predicting the already-observed count unchanged is right "
            "whenever a cascade is finished — which most are"
            + (f" (`{metric} {persistence:.4f}`)" if persistence is not None else "")
            + ". An arm that does not clear both has learned the corpus's size "
            "distribution rather than anything about the instance.",
            "",
        ]

    lines += [
        f"| arm | condition | {metric} | MALE | MAPE | PCC | COV-10% | median APE | "
        f"declined | selection | gap | kernel calls |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for result in rank_by(scored, ground_truth_reward, minimize):
        metrics = result.get("metrics") or {}
        selection = result.get("selection_metrics") or {}
        gap = result.get("generalization_gap")
        total = (metrics.get("n_scored") or 0) + (metrics.get("n_failed") or 0)

        lines.append(
            f"| `{result.get('arm')}` "
            f"| {result.get('condition')} "
            f"| **{_format_number(ground_truth_reward(result), 4)}** "
            f"| {_format_number(metrics.get('male'), 4)} "
            f"| {_format_number(metrics.get('mape'), 4)} "
            f"| {_format_number(metrics.get('pcc'), 4)} "
            f"| {_format_number(metrics.get('coverage'), 3)} "
            f"| {_format_number(metrics.get('ape_median'), 3)} "
            f"| {metrics.get('n_failed', 0):.0f}/{total:.0f} "
            f"| {_format_number(selection.get(result.get('prediction_metric', 'msle')), 4)} "
            f"| {_format_number(gap, 4) if gap is not None else '-'} "
            f"| {result.get('kernel_calls') or 0} |"
        )

    modelling = [
        result for result in scored if result.get("model_msle") is not None
    ]
    if modelling:
        lines += [
            "",
            "### Modelling error — what the FORWARD MODEL alone predicts",
            "",
            "The number §9.1 is actually about, and the one no other task in this "
            "repo can produce. Roll each arm's own forward model forward from the "
            "observed prefix with no program in the loop, and compare its expected "
            "popularity against what the log says happened. The gap between this "
            "column and the arm's own error is what the SEARCH bought; the LEVEL is "
            "how far an Independent-Cascade-shaped kernel is from a real adoption "
            "process. §2.2 names three mechanisms by which it is wrong — adoption is "
            "not memoryless, exposure is repeated rather than one-shot per "
            "neighbour, and exogenous arrivals have no infected in-neighbour at all.",
            "",
            "| arm | model MSLE | model MALE | model PCC | program error | "
            "search gain |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for result in rank_by(modelling, lambda entry: entry["model_msle"], minimize):
            program = ground_truth_reward(result)
            lines.append(
                f"| `{result.get('arm')}` "
                f"| {_format_number(result.get('model_msle'), 4)} "
                f"| {_format_number(result.get('model_male'), 4)} "
                f"| {_format_number(result.get('model_pcc'), 4)} "
                f"| {_format_number(program, 4)} "
                f"| {_format_number(result['model_msle'] - program, 4)} |"
            )

    lines += [
        "",
        "**Published context, explicitly NOT a like-for-like comparison.** CasFT "
        "reports MSLE `2.1728` on Weibo at `t_o = 0.5 h`, `3.8546` on Twitter at "
        "1 d and `1.2468` on APS at 3 y, against CasFlow's `2.3370` / `4.7799` / "
        "`1.4370` (§5.1, all under the random split). Under CasTemp's leak-free "
        "split the same field compresses to `1.475` / `1.171` / `1.926` for its own "
        "method and `1.685` / `1.329` / `2.438` for CasFlow (§5.3). Our "
        "preprocessing differs from all of them (§6.4 lists seven artefacts sharing "
        "three names), so these are context markers for the order of magnitude and "
        "nothing more. §9.9: do not chase the leaderboard — CasFlow is a "
        "2M-parameter model tuned for this one task.",
        "",
    ]

    return lines


def _reconstruction_table(agent_results: list[dict]) -> list[str]:
    """
    The tree half and the node half side by side, on HELD-OUT cascades.

    The decoding task's results table, and the one thing it must never do is print
    only the easy half. Zong ICDM'12 reports `prec_v = 100%` alongside
    `prec_e = 78-86%` and DIPT's best path precision thirteen years later is
    `0.680` against source-localization F1 of `0.518-0.839` on the same graphs
    ([`research/cascade_reconstruction.md`](../../../../research/cascade_reconstruction.md)
    §5.2, §8.1) — the node set is easy and the tree is hard, and a table that led
    with `node F1` would look excellent and say nothing.
    """
    first = agent_results[0]
    tree_weight = first.get("tree_weight", 0.6)
    trivial = next(
        (
            result["trivial_decoder_reward"]
            for result in agent_results
            if result.get("trivial_decoder_reward") is not None
        ),
        None,
    )
    lines = [
        f"**The score is `{tree_weight:.2f} * PathPrecision + "
        f"{1.0 - tree_weight:.2f} * EventF1`, and the weighting is the "
        f"specification rather than a presentation choice.** §2.6: recovering "
        f"WHICH nodes were infected is nearly free, so an outer loop rewarded on "
        f"Event F1 alone discovers the tree contributes nothing to its score and "
        f"converges on decoders that never attempt the hard half. The two "
        f"components are printed separately because the gap between them is the "
        f"result.",
        "",
        f"Every row is scored on the **held-out** `{first.get('eval_split', '?')}` "
        f"cascades by re-running that arm's winning decoder unmodified; `selection` "
        f"is what it scored on the `{first.get('select_split', '?')}` cascades the "
        f"outer loop optimized against, and `gap` is the difference.",
        "",
        f"Masking: **`{first.get('observation_setting', '?')}`**, with each infected "
        f"node reported at probability "
        f"`{_format_number(first.get('observation_rate'), 2)}`. The direction is "
        f"stated because this literature uses the symbol `sigma` for both the "
        f"report rate and its complement (§8.2 trap 1). The four settings are four "
        f"separate protocols and their rows are never pooled (§8.3).",
        "",
    ]

    if trivial is not None:
        lines += [
            f"**Reward sanity check (§2.11 risk 1): a trivial decoder — everyone "
            f"reachable, parents by BFS — scores `{trivial:.4f}` under this reward.** "
            f"If that number were competitive with the arms below, the reward would "
            f"be wrong rather than the arms good.",
            "",
        ]

    lines += [
        "Warning: **`path precision` is a PRECISION.** An arm that names three "
        "transmission edges and gets them right scores 1.0 on the half the reward "
        "is weighted toward, so read `path recall`, `jaccard` and `tree edges` "
        "beside it — under-predicting is the second gaming corner and it is not one "
        "this literature names, because no published method has a search that could "
        "find it.",
        "",
        "| # | arm | evaluator | score | path prec | path rec | jaccard | order acc "
        "| event F1 | node F1 | MCC | NRMSE↓ | src F1 | tree edges | selection | gap "
        "| kernel calls / cascade | eval s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
        "--- | --- | --- | --- | --- | --- |",
    ]

    ordered = sorted(
        agent_results,
        key=lambda result: (result.get("condition", 99), result["arm"]),
    )

    for result in ordered:
        metrics = result.get("metrics") or {}
        gap = result.get("generalization_gap")
        score = ground_truth_reward(result)
        # Arm A pays MCMC proposals rather than free-form kernel calls, and both
        # are the cost axis §2.4.2 exists to measure
        per_instance = result.get("kernel_calls_per_instance")
        if result.get("mcmc_proposals_per_instance"):
            per_instance = (
                f"{result['kernel_calls_per_instance']} "
                f"({result['mcmc_proposals_per_instance']} MH proposals)"
            )

        lines.append(
            f"| {result.get('condition', '—')} | `{result['arm']}` "
            f"| `{result.get('evaluator', '—')}` "
            f"| {_format_number(score, 4)} "
            f"| {_format_number(metrics.get('path_precision'), 4)} "
            f"| {_format_number(metrics.get('path_recall'), 4)} "
            f"| {_format_number(metrics.get('jaccard'), 4)} "
            f"| {_format_number(metrics.get('order_accuracy'), 4)} "
            f"| {_format_number(metrics.get('event_f1'), 4)} "
            f"| {_format_number(metrics.get('node_f1'), 4)} "
            f"| {_format_number(metrics.get('mcc'), 4)} "
            f"| {_format_number(metrics.get('time_nrmse'), 4)} "
            f"| {_format_number(metrics.get('source_f1'), 4)} "
            f"| {_format_number(metrics.get('n_tree_edges'), 1)} "
            f"| {_format_number(score - gap, 4) if gap is not None else '—'} "
            f"| {'—' if gap is None else f'{gap:+.4f}'} "
            f"| {per_instance if per_instance is not None else '—'} "
            f"| {_format_number(result.get('evaluator_seconds'), 1)} |"
        )

    if not first.get("has_tree_truth", True):
        lines += [
            "",
            "Warning: **this dataset carries no transmission edge**, so every tree "
            "column above is empty and the score collapsed onto Event F1. That is "
            "the exact failure §2.6 describes — regenerate with "
            "`--trace-parents` (the pipeline sets it automatically for "
            "`--task cascade_reconstruction`) before reading any of these numbers "
            "as a reconstruction result.",
        ]

    if any(result.get("resim_error") is not None for result in ordered):
        lines += [
            "",
            "### Re-simulated error",
            "",
            "Re-run the ground-truth simulator from each decode's RECOVERED SOURCES "
            "— the nodes it gave no parent — and compare against what the cascade "
            "actually did. The score above is already exact (it is measured against "
            "a history we stored), so this measures something else: whether the "
            "recovered ROOTS reproduce the observation. Reported beside the true "
            "source set's own error, because on an ill-posed problem a recovered "
            "set can reproduce it better than the truth did.",
            "",
            "| arm | resim error (recovered) | resim error (true sources) | ratio |",
            "| --- | --- | --- | --- |",
        ]

        for result in ordered:
            recovered = result.get("resim_error")
            truth = result.get("resim_error_true_sources")
            if recovered is None:
                continue

            ratio = recovered / truth if truth else None
            lines.append(
                f"| `{result['arm']}` | {_format_number(recovered, 5)} "
                f"| {_format_number(truth, 5)} "
                f"| {'—' if ratio is None else f'{ratio:.3f}'} |"
            )

    return lines + [""]


def _results_table(agent_results: list[dict]) -> list[str]:
    if not agent_results:
        return []

    has_mc = any(result.get("mc_reward") is not None for result in agent_results)
    sense = result_sense(agent_results)
    recover = is_recover(agent_results)
    lines = ["## Results", ""]

    if is_forecast(agent_results):
        return lines + _prediction_table(agent_results)

    if is_reconstruct(agent_results):
        return lines + _reconstruction_table(agent_results)

    if recover:
        return lines + _localization_table(agent_results)

    blocking = any(result.get("blocking") for result in agent_results)

    if blocking:
        lines += [
            "> **`spread` is the RUMOUR's remaining size and LOWER IS BETTER.** Two "
            "cascades run on this graph: a rumour seeded first at nodes no arm chose, "
            "and each arm's own intervention answering it. The column counts only the "
            "rumour. The prevented-influence table below reports the same numbers as "
            "a difference from the unopposed cascade, which is the form this "
            "literature publishes in.",
            "",
        ]
    elif any(result.get("epidemic") for result in agent_results):
        lines += [
            "> **`spread` is the ATTACK RATE and LOWER IS BETTER.** This is a "
            "compartmental epidemic-control task: an outbreak is already running "
            "from index cases no arm chose, the budget buys doses, and the column "
            "counts every node that was EVER infected. Unlike every other task "
            "here the dynamics are NOT monotone — infectious nodes recover and stop "
            "transmitting — so the outbreak burns out on its own and the "
            "prevented-infections table below reports what each arm saved before "
            "that happened, together with the curve's shape.",
            "",
        ]
    elif sense == minimize:
        lines += [
            "> **`spread` is the objective and LOWER IS BETTER.** This is a "
            "containment task: an exogenous outbreak is already running, the budget "
            "buys node deletions, and the column reports how many nodes the cascade "
            "still reached. The best arm is the one with the SMALLEST number.",
            "",
        ]

    if has_mc:
        lines += [
            "**Spread** is the ground-truth Monte Carlo replay of each arm's winning "
            "strategy — the only number comparable across conditions, since each "
            "arm's own `reward` is measured by its own evaluator. **Estimate** is "
            "what that arm's evaluator believed, so estimate − spread is its "
            "fidelity error (zero by construction for a `monte_carlo` arm).",
            "",
        ]
    else:
        lines += [
            "> **Warning:** `--compare` was off, so each row is scored by its own "
            "evaluator and rows are NOT comparable across conditions. Re-run with "
            "`--compare` for a valid table.",
            "",
        ]

    header = (
        f"| # | budget k | % of N | arm | evaluator | spread "
        f"({reward_direction(sense)}) | % of N |"
    )
    divider = "| --- | --- | --- | --- | --- | --- | --- |"

    if has_mc:
        header += " estimate | est − spread |"
        divider += " --- | --- |"

    # eval calls / eval s are the inner-loop cost axis; total s is dominated by
    # LLM latency and says little about which evaluator is cheaper
    header += " real episodes | eval calls | eval s | total s |"
    divider += " --- | --- | --- | --- |"

    rows = []
    ordered = sorted(
        agent_results,
        key=lambda result: (
            result["budget"],
            result.get("condition", 99),
            result["arm"],
        ),
    )

    for result in ordered:
        spread = ground_truth_reward(result)
        nodes = result["graph"]["num_nodes"]
        row = (
            f"| {result.get('condition', '—')} | {result['budget']} "
            f"| {_format_number(result['budget_pct'])} | `{result['arm']}` "
            f"| `{result.get('evaluator', '—')}` | {_format_number(spread)} "
            f"| {_format_number(100.0 * spread / nodes)} |"
        )

        if has_mc:
            # The winner's-curse-free estimate when the arm produced one
            estimate = result.get("wm_reeval_mean", result["reward"])
            row += f" {_format_number(estimate)} |"
            row += (
                f" {estimate - spread:+.2f} |"
                if result.get("mc_reward") is not None
                else " — |"
            )

        row += f" {result.get('real_env_episodes', 0)} |"
        row += f" {result.get('evaluator_calls', 0)} |"
        row += f" {_format_number(result.get('evaluator_seconds'), 1)} |"
        row += f" {_format_number(result.get('elapsed_seconds'), 1)} |"
        rows.append(row)

    return lines + [header, divider] + rows + [""]


def _adaptivity_section(agent_results: list[dict]) -> list[str]:
    """
    The adaptivity gap, and the cost each side paid to get there.

    Empty for a non-adaptive sweep. The two calibrations in the header are not
    hedging: theory caps the myopic gap at 4 and proves non-adaptive greedy is no
    worse than adaptive greedy over all graphs, so a gap near 1 is the predicted
    outcome and the claim being made here is about cost.
    """
    paired = adaptivity_gaps(agent_results)
    if not paired:
        return []

    lines = [
        "## Adaptivity gap",
        "",
        "`gap = spread(adaptive policy) / spread(matched non-adaptive arm)` at "
        "the same budget and the same evaluator, both read on the ground-truth "
        "MC replay. Rounds commit `k` in batches, each chosen after observing "
        "what the previous batch activated. The control is the STRONGEST static "
        "arm at that budget and evaluator, which is the closest available "
        "estimate of the `max` the gap is defined against.",
        "",
        "> **Read this before reading the numbers.** Peng & Chen bound the myopic "
        "adaptivity gap in `[e/(e−1), 4]` and prove non-adaptive greedy is no "
        "worse than adaptive greedy across all graphs "
        "([`research/adaptive_online_im.md`](../../../../research/adaptive_online_im.md) "
        "§5.1). A gap near `1.00` is the expected result, not a failed run, and a "
        "large gap is an instance effect rather than a general one. The claim "
        "this task supports is the **cost** columns: the MC arm re-estimates "
        "every candidate once per round, and a forward pass does not.",
        "",
        "| budget k | evaluator | rounds | batches | feedback | adaptive arm | "
        "spread | control arm | spread | gap | adaptive eval s | control eval s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for entry in paired:
        gap = entry["gap"]
        lines.append(
            f"| {entry['budget']} | `{entry['evaluator']}` | {entry['rounds']} "
            f"| `{entry['round_batches']}` | `{entry['feedback_model']}` "
            f"| `{entry['adaptive_arm']}` "
            f"| {_format_number(entry['adaptive_spread'])} "
            f"| `{entry['control_arm']}` "
            f"| {_format_number(entry['control_spread'])} "
            f"| {'n/a' if gap is None else f'{gap:.3f}'} "
            f"| {_format_number(entry['adaptive_evaluator_seconds'], 1)} "
            f"| {_format_number(entry['control_evaluator_seconds'], 1)} |"
        )

    return lines + [""]


def _blocking_section(agent_results: list[dict]) -> list[str]:
    """
    Prevented influence per arm — the column every blocking paper actually reports.

    Empty for every task with one cascade. The results table above reports the
    rumour's REMAINING size, which is what the search minimizes; this reports the
    same numbers as a DIFFERENCE from the unopposed cascade, which is the form
    research/influence_blocking.md §8.1 records under five different names and the
    only form comparable to a published table.
    """
    scored = [result for result in agent_results if result.get("blocking")]
    if not scored:
        return []

    largest = max(result["budget"] for result in scored)
    at_largest = sorted(
        [result for result in scored if result["budget"] == largest],
        key=lambda result: -(result.get("mc_prevented_influence") or 0.0),
    )
    head = at_largest[0]
    unopposed = head.get("mc_unopposed_spread") or head.get("unopposed_spread") or 0.0

    lines = [
        f"## Prevented influence at k={largest}",
        "",
        f"`prevented = sigma(S_N, empty) - sigma(S_N | blockers)`, both terms measured "
        f"on the shared ground-truth referee. The rumour was seeded by "
        f"`{head.get('attacker', '?')}` at **{head.get('n_negative_seeds', '?')} "
        f"nodes** and reaches **{unopposed:,.1f}** of them unopposed; the lever is "
        f"`{head.get('lever', '?')}` ({head.get('lever_papers', '')}); the tie-break "
        f"is `{head.get('tie_break', '?')}` under "
        f"`{head.get('competitive_model', '?')}`.",
        "",
        "> **Read the budget column as an absolute k, not a percentage.** "
        "[`research/influence_blocking.md`](../../../../research/influence_blocking.md) "
        "§8.2 records that percent-of-N budgets are used by **nobody** in this "
        "literature, while `k` in `{10..50}` is the shared convention of SandIMIN, "
        "both Xie papers and TC-AIBM. The informative ratio is `|S_P| / |S_N|`, which "
        "is its own column below — CLDAG's Table 2 shows it takes 20-30x the rumour's "
        "own seed count to cut it to a 10% residual.",
        "",
        "> **A heuristic winning here is the NORMAL outcome, not a failed run.** "
        "SandIMIN's own Table 5 has its trivial highest-gain heuristic beating both "
        "of that paper's principled methods in 6 of 30 cells, and StratLearner's "
        "Table 1 has plain proximity above every learned method except StratLearner "
        "on two of three graphs. `proximity` and `imin_lhga` are in the default pool "
        "for exactly that reason; `degree_blocking` is there as the published FAILURE "
        "mode, since CLDAG reports plain degree cannot be used for this task at all.",
        "",
        # The ratio's own pipes have to be escaped or they close the table cell
        "| arm | rumour size | prevented | % of cascade | % of N | "
        "\\|S_P\\|/\\|S_N\\| | spent |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for result in at_largest:
        prevented = result.get("mc_prevented_influence")
        percent = result.get("mc_prevented_pct_of_unopposed")
        ratio = result.get("budget_ratio")
        lines.append(
            f"| `{result['arm']}` "
            f"| {_format_number(ground_truth_reward(result))} "
            f"| {'—' if prevented is None else f'{prevented:+.2f}'} "
            f"| {'—' if percent is None else f'{percent:.1f}%'} "
            f"| {_format_number(result.get('prevented_pct_of_nodes'))} "
            f"| {'—' if ratio is None else f'{ratio:.2f}'} "
            f"| {result.get('n_spent', '—')} |"
        )

    lines += [
        "",
        "`prevented` counts only nodes the rumour **would otherwise have infected** "
        "(Budak's \"saved\" set): a blocker that protects a node the cascade never "
        "reaches scores exactly zero, however central that node is. An arm whose "
        "`spent` is below the budget ran out of candidates — the reachable region was "
        "smaller than `k`, which is common at the high end of the sweep and is the "
        "honest answer rather than a padded set.",
        "",
    ]

    return lines


def _epidemic_section(agent_results: list[dict]) -> list[str]:
    """
    Prevented infections and the outbreak's SHAPE — what an immunization table reports.

    Empty for every non-compartmental task. Three groups, and the order is the
    argument this task makes (research/epidemic_control.md §8.3):

      1. `prevented` is the attack rate subtracted from the unprotected reference,
         both measured on the shared referee.
      2. `peak` / `t_peak` / `AUC` are the SHAPE. §8.2 trap 4 is the reason they are
         not optional: a good policy flattens rather than eliminates, so a
         terminal-state number alone can rank two policies backwards.
      3. `eigendrop` is the spectral line's own metric, and it is here as CONTEXT.
         §8.2 trap 1: a method can win it and lose the attack rate, because
         `lambda_1` does not know where the outbreak IS. Seeing NetShield above
         DAVA in this column and below it in the previous one is the point of
         printing both.
    """
    scored = [result for result in agent_results if result.get("epidemic")]
    if not scored:
        return []

    largest = max(result["budget"] for result in scored)
    at_largest = sorted(
        [result for result in scored if result["budget"] == largest],
        key=lambda result: -(result.get("mc_prevented_infections") or 0.0),
    )
    head = at_largest[0]
    unprotected = (
        head.get("mc_unprotected_attack_rate")
        or head.get("unprotected_attack_rate")
        or 0.0
    )
    alpha = head.get("epi_alpha")

    lines = [
        f"## Prevented infections at k={largest}",
        "",
        f"`prevented = |R(inf)| unprotected - |R(inf)| with doses`, both terms "
        f"measured on the shared ground-truth referee. The outbreak was seeded by "
        f"`{head.get('outbreak_selector', '?')}` at **{head.get('n_outbreak', '?')} "
        f"index case(s)** and reaches **{unprotected:,.1f}** nodes unprotected. "
        f"Dynamics: **{head.get('compartments', '?')}**, per-contact transmission "
        f"`{head.get('epi_beta', '?')}x` the arc's own probability, leaving-`I` rate "
        f"`{head.get('epi_gamma', '?')}`"
        + (f", `E -> I` rate `{alpha}`" if alpha is not None else "")
        + f". Lever: `{head.get('lever', '?')}` ({head.get('lever_papers', '')}).",
        "",
        "> **`beta` and `gamma` are free parameters and nobody standardizes them.** "
        "[`research/epidemic_control.md`](../../../../research/epidemic_control.md) "
        "§8.2 trap 2: NetShield reports against a normalized virus strength swept on "
        "the x-axis, and most other papers fix one pair without justifying it. The "
        "rates above are stated for exactly that reason — this table is comparable "
        "to another run at the same rates and to nothing else.",
        "",
        "> **Read `eigendrop` as context, never as the score.** It is what the "
        "spectral line (NetShield, NetMelt, Gelling, GreedyWalk) actually optimizes, "
        "and it needs no simulator at all — so it is the only column this table and "
        "theirs share. It is also the column §8.2 trap 1 warns about: `lambda_1` says "
        "nothing about WHERE the infection currently is, which is DAVA's entire "
        "contribution and the reason a data-aware method can post the smallest "
        "eigendrop here and still prevent the most infections.",
        "",
        "| arm | attack rate | prevented | % of outbreak | peak I | t_peak | "
        "AUC | eigendrop | spent |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for result in at_largest:
        curve = result.get("mc_curve") or result
        prevented = result.get("mc_prevented_infections")
        percent = result.get("mc_prevented_pct_of_unprotected")
        spectral = result.get("spectral") or {}
        eigendrop = spectral.get("eigendrop_pct")
        lines.append(
            f"| `{result['arm']}` "
            f"| {_format_number(ground_truth_reward(result))} "
            f"| {'—' if prevented is None else f'{prevented:+.2f}'} "
            f"| {'—' if percent is None else f'{percent:.1f}%'} "
            f"| {_format_number(curve.get('peak_prevalence'))} "
            f"| {curve.get('time_to_peak', '—')} "
            f"| {_format_number(curve.get('auc_infectious'))} "
            f"| {'—' if eigendrop is None else f'{eigendrop:.1f}%'} "
            f"| {result.get('n_spent', '—')} |"
        )

    lines += [
        "",
        "`prevented` counts only nodes the outbreak **would otherwise have reached**: "
        "a dose spent on a node the epidemic never gets to scores exactly zero, "
        "however central that node is. The infectious set also RECOVERS, so the "
        "outbreak burns out on its own and a dose is worth only what it saves before "
        "then — which is why `t_peak` moving earlier is a real result even when the "
        "attack rate barely moves.",
        "",
    ]

    if head.get("compartments") == "SIS":
        lines += [
            "> **SIS has no terminal state**, so the attack rate above is the "
            "CUMULATIVE incidence (every node ever infected) rather than a final "
            "size, and it grows monotonically with the horizon. §8.2 trap 6: the "
            "quantity this literature reports for SIS is the endemic prevalence "
            "`lim |I(t)|/N`, which is the `epi_endemic_prevalence` column of "
            "`summary.csv`. Read that one, not this one, when comparing to a "
            "published SIS number.",
            "",
        ]

    return lines


def _structural_section(agent_results: list[dict]) -> list[str]:
    """
    What each removal set did to the graph's CONNECTIVITY — context, not the score.

    Empty for every task that does not remove nodes. The framing matters as much
    as the numbers: `research/critical_node_detection.md` §2.1 argues that
    structural CNDP is not a world-model problem at all (its transition is
    deterministic and `nx.connected_components` is cheaper than one forward pass),
    so this table exists to bridge to the published dismantling literature, which
    reports these functionals and not spread. §5.8 is the reason it can disagree
    with the results table above: the same centralities rank in OPPOSITE orders
    under a spreading objective and a connectivity objective.
    """
    with_structural = [
        result for result in agent_results if result.get("structural") is not None
    ]
    if not with_structural:
        return []

    largest = max(result["budget"] for result in with_structural)
    at_largest = rank_by(
        [result for result in with_structural if result["budget"] == largest],
        ground_truth_reward,
        result_sense(with_structural),
    )
    intact = at_largest[0]["structural"]

    lines = [
        f"## Structural context at k={largest}",
        "",
        "Computed exactly by BFS on the residual graph, and **never a training "
        "target** — a k-layer message-passing model cannot represent "
        "giant-component membership on a graph of diameter > k, so the world model "
        "is fit on the diffusion transition and these describe the same removal "
        "sets afterwards "
        "([`research/critical_node_detection.md`](../../../../research/critical_node_detection.md) "
        "§2.1, §8.3). They are the units the published dismantling literature "
        "reports in, so they are the bridge to it — and §5.8 shows a method can "
        "win the column above and lose every column here.",
        "",
        f"Intact graph: pairwise connectivity {intact['pairwise_conn_intact']:,.0f}, "
        f"largest component {intact['largest_cc_intact']:,.0f}, "
        f"{intact['n_components_intact']:.0f} component(s). "
        f"`rho` is the fraction of N removed to drive the giant component below "
        f"{intact['gcc_threshold']:.0%} of N; `—` means the budget ran out first, "
        f"which is the common case at these budgets and is the honest answer.",
        "",
        "| arm | spread | GCC after | GCC drop | pairwise conn drop | components | "
        "Schneider R | ANC (sigma = GCC) | rho at threshold | degree-rank rho |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for result in at_largest:
        structural = result["structural"]
        rho = structural.get("rho_at_threshold")
        lines.append(
            f"| `{result['arm']}` "
            f"| {_format_number(ground_truth_reward(result))} "
            f"| {structural['largest_cc_size']:,.0f} "
            f"| {structural['largest_cc_drop_pct']:.1f}% "
            f"| {structural['pairwise_conn_drop_pct']:.1f}% "
            f"| {structural['n_components']:.0f} "
            f"| {_format_number(structural['schneider_r'], 4)} "
            f"| {_format_number(structural['anc'], 4)} "
            f"| {'—' if rho is None else f'{rho:.3f}'} "
            f"| {structural['degree_rank_spearman']:+.3f} |"
        )

    lines += [
        "",
        "**`degree-rank rho`** is the Spearman correlation between an arm's removal "
        "ORDER and the degree of the nodes it removed — the self-measurement "
        "§9.5 asks for. MIND found GDM's dismantling order correlates at **0.762** "
        "with a PCA of its own handcrafted input features. We feed `log1p(degree)` "
        "as feature channel 2, so an arm near that value has re-derived the degree "
        "heuristic with extra steps, whatever its spread number says.",
        "",
    ]

    return lines


def _winner_section(agent_results: list[dict]) -> list[str]:
    """Best arm at the largest budget, with the program it produced."""
    if not agent_results:
        return []

    largest = max(result["budget"] for result in agent_results)
    at_largest = [result for result in agent_results if result["budget"] == largest]
    sense = result_sense(agent_results)
    # argmin on a containment task: the winner is the arm that let the FEWEST
    # nodes get infected, and taking the max there would name the worst arm
    winner = best_by(at_largest, ground_truth_reward, sense)
    spread = ground_truth_reward(winner)

    if is_reconstruct(at_largest):
        metrics = winner.get("metrics") or {}
        headline = (
            f"held-out score {_format_number(spread, 4)} "
            f"(path precision {_format_number(metrics.get('path_precision'), 4)}, "
            f"event F1 {_format_number(metrics.get('event_f1'), 4)}) over "
            f"{winner.get('n_eval_instances', '?')} "
            f"`{winner.get('eval_split', '?')}` cascades"
        )
        title = (
            f"## Winning arm ({winner.get('observation_setting', '?')} observation)"
        )
    elif is_recover(at_largest):
        headline = (
            f"held-out {reward_name(at_largest)} {_format_number(spread, 4)} "
            f"over {winner.get('n_eval_instances', '?')} "
            f"`{winner.get('eval_split', '?')}` episodes"
        )
        title = f"## Winning arm at k={largest} (sources per episode)"
    else:
        judged = (
            "ground-truth MC" if is_ground_truth(at_largest) else "its own evaluator"
        )
        headline = (
            f"spread {_format_number(spread)} "
            f"({_format_number(100.0 * spread / winner['graph']['num_nodes'])}% of "
            f"N, {reward_direction(sense)}) by {judged}"
        )
        title = f"## Winning arm at k={largest}"

    lines = [
        title,
        "",
        f"**`{winner['arm']}`** (condition {winner.get('condition', '—')} — "
        f"{condition_names.get(winner.get('condition'), 'unknown')}) — {headline}, "
        f"model `{winner.get('model')}`",
        "",
        "```",
        str(winner.get("summary", "")).strip(),
        "```",
        "",
    ]

    # The agent's own account of how it got here — written before the program so
    # the reader knows what they are looking at when they expand it
    explanation = winner.get("explanation")
    if explanation:
        lines += [
            f"_The write-up below is the agent's own, produced by "
            f"`{winner.get('model')}` at the end of its refinement loop on the "
            f"same conversation thread as the program._",
            "",
            markdown_heading.sub(r"#\1", explanation.strip()),
            "",
        ]

    lines += [
        "<details><summary>Generated program</summary>",
        "",
        "```python",
        str(winner.get("script", "")).strip(),
        "```",
        "",
        "</details>",
        "",
    ]

    return lines


def _world_model_section(wm_results: dict | None) -> list[str]:
    if wm_results is None:
        return [
            "## World model",
            "",
            "_Not trained for this run (oracle or Monte Carlo evaluator)._",
            "",
        ]

    test = wm_results.get("test", {})
    rollout = wm_results.get("rollout", {})
    lines = [
        "## World model",
        "",
        f"- **backbone / head**: `{wm_results['config'].get('model')}` / "
        f"`{wm_results['config'].get('head')}`",
        f"- **train time**: {_format_number(wm_results.get('train_seconds'), 1)}s "
        f"over {len(wm_results.get('history', []))} epochs",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]

    for key in ("delta_f1", "new_infection_f1", "infected_acc", "frontier_acc"):
        if key in test:
            lines.append(f"| one-step `{key}` | {_format_number(test[key], 4)} |")

    for key in ("ens_count_bias", "ens_final_count_model", "ens_final_count_true"):
        if key in rollout:
            lines.append(f"| rollout `{key}` | {_format_number(rollout[key], 3)} |")

    return lines + [""]


def _figures_section(plots_dir: Path, report_parent: Path) -> list[str]:
    figures = sorted(plots_dir.glob("*.png"))
    if not figures:
        return []

    lines = ["## Figures", ""]
    for figure in figures:
        title = figure.stem.replace("_", " ")
        lines += [
            f"### {title}",
            "",
            f"![{title}]({figure.relative_to(report_parent).as_posix()})",
            "",
        ]

    return lines


def write_report(
    config: dict,
    layout: Layout,
    agent_results: list[dict],
    wm_results: dict | None,
) -> Path:
    metadata_path = layout.data_metadata()
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None

    lines = [
        f"# {layout.label} — Graph World Model results",
        "",
        f"Dataset `{config['dataset']}`, evaluator `{config['evaluator']}`, "
        f"dynamics `{config['diffusion_model']}`.",
        "",
        "## Configuration",
        "",
        "| key | value |",
        "| --- | --- |",
    ]

    for key in reported_config_keys:
        if key in config:
            lines.append(f"| `{key}` | `{config[key]}` |")

    lines += ["", f"Full config: [`pipeline.json`]({layout.manifest_path.name})", ""]
    lines += _graph_section(agent_results, metadata)
    lines += _taxonomy_section(agent_results)
    lines += _results_table(agent_results)
    lines += _adaptivity_section(agent_results)
    lines += _blocking_section(agent_results)
    lines += _epidemic_section(agent_results)
    lines += _structural_section(agent_results)
    lines += _winner_section(agent_results)
    lines += _world_model_section(wm_results)
    lines += _figures_section(layout.plots_dir, layout.root)

    layout.report_path.write_text("\n".join(lines))

    return layout.report_path
