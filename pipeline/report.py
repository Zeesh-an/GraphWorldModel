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
from pipeline.plots import pretty_arm, prettify, wm_metric_name
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
    "sl_transfer_from",
    "cr_setting",
    "cr_observation_rate",
    "cr_hidden_rate",
    "cr_instances",
    "cr_select_split",
    "cr_eval_split",
    "cr_tree_weight",
    "wm_model",
    "head",
    "hide_edge_weights",
    "seed",
)


def _format_number(value: object, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}" if isinstance(value, (int, float)) else ", "


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
                f"- **generation budget band**: k {entry.get('budget_k_min')}, "
                f"{entry.get('budget_k_max')} "
                f"({entry.get('budget_pct_min')}, {entry.get('budget_pct_max')}% of N)"
            )

    return lines + [""]


def _taxonomy_section(agent_results: list[dict]) -> list[str]:
    """Which conditions this run actually covers, and what each one isolates."""
    present = sorted({result.get("condition", 99) for result in agent_results})
    if not present:
        return []

    isolates = {
        1: "classical floor, fixed expert algorithms, no LLM anywhere",
        2: "is *choosing* from a pool enough, versus *generating* code?",
        3: "do the gains come merely from having a coding agent?",
        4: "does simulated lookahead by itself explain the gain?",
        5: "ceiling of model-based guidance, a perfect internal model",
        6: "does the *learned* model recover the true dynamics?",
        7: "the original authors' code, seeds scored by our referee",
        9: "published LLM algorithm-discovery loops at their own defaults, our simulator as the only fitness",
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
            f"| {condition} | {condition_names.get(condition, ', ')} "
            f"| {', '.join(f'`{arm}`' for arm in arms)} "
            f"| {isolates.get(condition, ', ')} |"
        )

    return lines + [""]


def _localization_table(agent_results: list[dict]) -> list[str]:
    """
    The reward, its referee re-measurement, and the label metrics per arm, on
    HELD-OUT episodes, plus the cost each one paid.

    The inverse task's table reads like an intervention task's: the reward the
    search ran on (consistency, on the arm's own evaluator), the same quantity
    re-measured by the shared referee, and beside them the
    F1 / PR / RE / AUC the literature publishes, computed against the stored
    sources AFTER the search and never fed to it.
    """
    first = agent_results[0]
    lines = [
        "**The reward is label-free.** Each arm's program names a source set per "
        "episode; the harness rolls that set forward on the arm's own evaluator and "
        "scores minus the mean squared error against the observed state "
        "(`consistency`, 0 is perfect). `referee` is the same quantity re-measured "
        "on the shared referee (the exact oracle simulator by default) and is the "
        "column comparable across arms, exactly as a spread is. **F1 is the published "
        "metric** (SL-VAE calls it \"the most commonly used\"; IVGD \"the most "
        "important\") and it is computed against the true sources only after the "
        "search, on the winner: a set that reproduces the observation need not be "
        "the true set (diffusion is many-to-one), and the gap between the two "
        "columns is that identifiability, measured. AUC is the tie-breaker; "
        "**accuracy is near-useless alone** (IVGD's Table 3 has GCNSI at `ACC "
        "0.8840` with `F1 0.0218`), so it is reported only beside F1 "
        "([`research/source_localization.md`](../../../../research/source_localization.md) "
        "§8.1).",
        "",
        f"Every row is scored on the **held-out** `{first.get('eval_split', '?')}` "
        f"episodes, by re-running that arm's winning program unmodified. "
        f"`selection` is the reward on the `{first.get('select_split', '?')}` "
        f"episodes the outer loop optimized against, and `gap` the difference: a "
        f"large negative gap means the program memorized specific cascades rather "
        f"than learning an algorithm, which is the failure §8.5.1 exists to catch.",
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
        "its quality. Compare AUC only within one value of that column.",
        "",
        "| # | k | arm | evaluator | consistency (higher is better) | referee | "
        "selection | gap | F1 | PR | RE | AUC | ACC | F1 gap | AUC from "
        "| fwd calls / instance | eval s | total s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
        "--- | --- | --- | --- | --- | --- |",
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
        gap = result.get("generalization_gap")
        f1_gap = result.get("f1_generalization_gap")
        reward = result.get("reward")
        selection = (
            reward - gap if reward is not None and gap is not None else None
        )

        lines.append(
            f"| {result.get('condition', ', ')} | {result.get('budget', ', ')} "
            f"| `{result['arm']}` "
            f"| `{result.get('evaluator', ', ')}` "
            f"| {_format_number(reward, 5)} "
            f"| {_format_number(result.get('referee_reward'), 5)} "
            f"| {_format_number(selection, 5)} "
            f"| {', ' if gap is None else f'{gap:+.5f}'} "
            f"| {_format_number(metrics.get('f1'), 4)} "
            f"| {_format_number(metrics.get('precision'), 4)} "
            f"| {_format_number(metrics.get('recall'), 4)} "
            f"| {_format_number(metrics.get('auc'), 4)} "
            f"| {_format_number(metrics.get('accuracy'), 4)} "
            f"| {', ' if f1_gap is None else f'{f1_gap:+.4f}'} "
            f"| `{result.get('auc_source', ', ')}` "
            f"| {_format_number(result.get('forward_calls_per_instance'), 1)} "
            f"| {_format_number(result.get('evaluator_seconds'), 1)} "
            f"| {_format_number(result.get('elapsed_seconds'), 1)} |"
        )

    if any(result.get("resim_error") is not None for result in ordered):
        lines += [
            "",
            "### Re-simulated error on the referee",
            "",
            "The `referee` column, unpacked: the ground-truth simulator re-run from "
            "each arm's RECOVERED sources against what was observed, beside the TRUE "
            "source set's own error. **No surveyed paper reports this at all** "
            "(§11), so the column is self-contained and is not a cross-paper "
            "comparison. A ratio below 1 means the recovered set explains the "
            "observation better than the truth did, which is what an ill-posed "
            "instance looks like.",
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
                f"| {', ' if ratio is None else f'{ratio:.3f}'} |"
            )

    transferred = [result for result in ordered if result.get("transfer_from")]
    if transferred:
        lines += [
            "",
            "### Transferred programs",
            "",
            "These rows ran a program selected on a DIFFERENT graph, unmodified. "
            "That is the graph axis of §8.5.1 and the headline of the amortization "
            "claim, and it is a comparison no per-instance method can enter, "
            "because SL-VAE, IVGD and DDMSL have no artifact to transfer.",
            "",
            "| arm | selected on | consistency here | F1 here |",
            "| --- | --- | --- | --- |",
        ]
        lines += [
            f"| `{result['arm']}` | `{result['transfer_from']}` "
            f"| {_format_number(result.get('reward'), 5)} "
            f"| {_format_number((result.get('metrics') or {}).get('f1'), 4)} |"
            for result in transferred
        ]

    return lines + [""]


def _prediction_table(agent_results: list[dict]) -> list[str]:
    """
    Every error column per arm, on HELD-OUT cascades, with the protocol above it.

    The forecasting task's results table. Two things it must do that no other table
    here does. First, it prints the PROTOCOL: corpus, window, horizon, split,
    filters, because
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
        f"> **`{metric}` is an ERROR and LOWER IS BETTER**: the only column in this "
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
            "whenever a cascade is finished, which most are"
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
        n_failed = metrics.get("n_failed") or 0
        total = (metrics.get("n_scored") or 0) + n_failed

        lines.append(
            f"| `{result.get('arm')}` "
            f"| {result.get('condition')} "
            f"| **{_format_number(ground_truth_reward(result), 4)}** "
            f"| {_format_number(metrics.get('male'), 4)} "
            f"| {_format_number(metrics.get('mape'), 4)} "
            f"| {_format_number(metrics.get('pcc'), 4)} "
            f"| {_format_number(metrics.get('coverage'), 3)} "
            f"| {_format_number(metrics.get('ape_median'), 3)} "
            f"| {n_failed:.0f}/{total:.0f} "
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
            "### Modelling error: what the FORWARD MODEL alone predicts",
            "",
            "The number §9.1 is actually about, and the one no other task in this "
            "repo can produce. Roll each arm's own forward model forward from the "
            "observed prefix with no program in the loop, and compare its expected "
            "popularity against what the log says happened. The gap between this "
            "column and the arm's own error is what the SEARCH bought; the LEVEL is "
            "how far an Independent-Cascade-shaped kernel is from a real adoption "
            "process. §2.2 names three mechanisms by which it is wrong: adoption is "
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
        "nothing more. §9.9: do not chase the leaderboard, CasFlow is a "
        "2M-parameter model tuned for this one task.",
        "",
    ]

    return lines


def _reconstruction_table(agent_results: list[dict]) -> list[str]:
    """
    The reward, its referee re-measurement, and the tree half against the node
    half, on HELD-OUT cascades.

    The decoding task's table, and the one thing it must never do is lead with
    the easy half. Zong ICDM'12 reports `prec_v = 100%` alongside `prec_e =
    78-86%` and DIPT's best path precision thirteen years later is `0.680`
    ([`research/cascade_reconstruction.md`](../../../../research/cascade_reconstruction.md)
    §5.2, §8.1): the node set is easy and the tree is hard.
    """
    first = agent_results[0]
    tree_weight = first.get("tree_weight")
    if tree_weight is None:
        tree_weight = 0.6
    trivial = next(
        (
            result["trivial_decoder_reward"]
            for result in agent_results
            if result.get("trivial_decoder_reward") is not None
        ),
        None,
    )
    trivial_tree = next(
        (
            result["trivial_decoder_tree_score"]
            for result in agent_results
            if result.get("trivial_decoder_tree_score") is not None
        ),
        None,
    )
    lines = [
        "**The reward is label-free.** Each arm's decoder returns a whole history "
        "per cascade; the harness scores its log-probability under the generative "
        "model (a `1/N` prior per declared source, plus the probability the arm's "
        "own transition kernel assigns to every transmission the history asserts), "
        "per node, minus the fraction of the observation the history contradicts "
        "(reported nodes dropped, reported times moved, nodes named that a snapshot "
        "says stayed clean). `referee` is the same quantity re-measured under the "
        "shared referee's kernel (the exact oracle by default) and is the column "
        "comparable across arms, exactly as a spread is.",
        "",
        f"**The tree score is the published metric**, `{tree_weight:.2f} * "
        f"PathPrecision + {1.0 - tree_weight:.2f} * EventF1`, computed against the "
        f"stored history AFTER the search and never fed to it. The two components "
        f"are printed separately because the gap between them is the result: "
        f"recovering WHICH nodes were infected is nearly free and the tree is not "
        f"(§2.6).",
        "",
        f"Every row is scored on the **held-out** `{first.get('eval_split', '?')}` "
        f"cascades by re-running that arm's winning decoder unmodified; "
        f"`selection` is the reward on the `{first.get('select_split', '?')}` "
        f"cascades the outer loop optimized against, and `gap` the difference.",
        "",
        f"Masking: **`{first.get('observation_setting', '?')}`**, with each infected "
        f"node reported at probability "
        f"`{_format_number(first.get('observation_rate'), 2)}`. The direction is "
        f"stated because this literature uses the symbol `sigma` for both the "
        f"report rate and its complement (§8.2 trap 1). The four settings are four "
        f"separate protocols and their rows are never pooled (§8.3).",
        "",
    ]

    if trivial is not None or trivial_tree is not None:
        lines += [
            f"**Reward sanity check (§2.11 risk 1): a trivial decoder, everyone "
            f"reachable, parents by BFS, scores `{_format_number(trivial, 4)}` under "
            f"the reward and `{_format_number(trivial_tree, 4)}` on the tree score.** "
            f"If either were competitive with the arms below, that reward would be "
            f"wrong rather than the arms good.",
            "",
        ]

    lines += [
        "Warning: **`path precision` is a PRECISION.** An arm that names three "
        "transmission edges and gets them right scores 1.0 on it, so read `path "
        "recall`, `jaccard` and `tree edges` beside it: under-predicting is the "
        "corner this literature does not name, because no published method has a "
        "search that could find it.",
        "",
        "| # | arm | evaluator | reward | referee | selection | gap | log-lik/node "
        "| consistency | tree score | path prec | path rec | jaccard | event F1 "
        "| node F1 | MCC | NRMSE↓ | src F1 | tree edges | kernel calls / cascade "
        "| eval s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
        "--- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    ordered = sorted(
        agent_results,
        key=lambda result: (result.get("condition", 99), result["arm"]),
    )

    for result in ordered:
        metrics = result.get("metrics") or {}
        gap = result.get("generalization_gap")
        reward = result.get("reward")
        selection = (
            reward - gap if reward is not None and gap is not None else None
        )

        lines.append(
            f"| {result.get('condition', ', ')} | `{result['arm']}` "
            f"| `{result.get('evaluator', ', ')}` "
            f"| {_format_number(reward, 4)} "
            f"| {_format_number(result.get('referee_reward'), 4)} "
            f"| {_format_number(selection, 4)} "
            f"| {', ' if gap is None else f'{gap:+.4f}'} "
            f"| {_format_number(metrics.get('loglik_per_node'), 4)} "
            f"| {_format_number(metrics.get('consistency'), 3)} "
            f"| {_format_number(metrics.get('tree_score'), 4)} "
            f"| {_format_number(metrics.get('path_precision'), 4)} "
            f"| {_format_number(metrics.get('path_recall'), 4)} "
            f"| {_format_number(metrics.get('jaccard'), 4)} "
            f"| {_format_number(metrics.get('event_f1'), 4)} "
            f"| {_format_number(metrics.get('node_f1'), 4)} "
            f"| {_format_number(metrics.get('mcc'), 4)} "
            f"| {_format_number(metrics.get('time_nrmse'), 4)} "
            f"| {_format_number(metrics.get('source_f1'), 4)} "
            f"| {_format_number(metrics.get('n_tree_edges'), 1)} "
            f"| {_format_number(result.get('kernel_calls_per_instance'), 1)} "
            f"| {_format_number(result.get('evaluator_seconds'), 1)} |"
        )

    if not first.get("has_tree_truth", True):
        lines += [
            "",
            "Warning: **this dataset carries no transmission edge**, so every tree "
            "column above is empty and the tree score collapsed onto Event F1. "
            "Regenerate with `--trace-parents` (the pipeline sets it automatically "
            "for `--task cascade_reconstruction`) before reading the tree columns "
            "as a reconstruction result.",
        ]

    if any(result.get("resim_error") is not None for result in ordered):
        lines += [
            "",
            "### Re-simulated error of the recovered roots",
            "",
            "Re-run the ground-truth simulator from each decode's RECOVERED SOURCES, "
            "the nodes it gave no parent, and compare against what the cascade "
            "actually did, beside the true source set's own error. A secondary "
            "column: the reward already scores the whole history.",
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
                f"| {', ' if ratio is None else f'{ratio:.3f}'} |"
            )

    return lines + [""]


def _results_table(agent_results: list[dict]) -> list[str]:
    if not agent_results:
        return []

    has_mc = any(result.get("referee_reward") is not None for result in agent_results)
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
            "here the dynamics are NOT monotone: infectious nodes recover and stop "
            "transmitting, so the outbreak burns out on its own and the "
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
        lines += _ring_note(agent_results)

    if has_mc:
        referee = next(
            (result.get("referee") for result in agent_results if result.get("referee")),
            "oracle",
        )
        samples = next(
            (result.get("referee_samples") for result in agent_results if result.get("referee_samples")),
            None,
        )
        lines += [
            f"**Spread** is the shared referee's replay of each arm's winning strategy "
            f"(`{referee}`" + (f", {samples} samples" if samples else "") + "): the only "
            "number comparable across conditions, since each arm's own `reward` is "
            "measured by its own evaluator. **Estimate** is what that arm's evaluator "
            "believed, so estimate − spread is its fidelity error (zero by construction "
            "for an arm whose evaluator is the referee).",
            "",
        ]
        agreed = [result for result in agent_results if result.get("referee_minus_mc") is not None]
        if agreed:
            worst = max(agreed, key=lambda result: abs(result["referee_minus_mc"]))
            lines += [
                f"**NDlib agreement check** on {len(agreed)} of {len(agent_results)} rows: "
                f"the same winners replayed on Monte Carlo agree with the referee to within "
                f"{max(abs(result['referee_minus_mc']) for result in agreed):.2f} nodes "
                f"(largest gap `{worst['arm']}` at k={worst.get('budget', '?')}: "
                f"{worst['referee_minus_mc']:+.2f}, referee SE "
                f"{_format_number(worst.get('referee_reward_se'), 2)}, MC SE "
                f"{_format_number(worst.get('mc_reward_se'), 2)}).",
                "",
            ]
    else:
        lines += [
            "> **Warning:** no row carries a referee number, so each row is scored by "
            "its own evaluator and rows are NOT comparable across conditions. These "
            "results predate the referee replay; re-run the arm.",
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
            f"| {result.get('condition', ', ')} | {result['budget']} "
            f"| {_format_number(result['budget_pct'])} | `{result['arm']}` "
            f"| `{result.get('evaluator', ', ')}` | {_format_number(spread)} "
            f"| {_format_number(100.0 * spread / nodes)} |"
        )

        if has_mc:
            # The winner's-curse-free estimate when the arm produced one
            estimate = result.get("wm_reeval_mean", result["reward"])
            row += f" {_format_number(estimate)} |"
            row += (
                f" {estimate - spread:+.2f} |"
                if result.get("referee_reward") is not None
                else ": |"
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
        "the same budget and the same evaluator, both read on the shared "
        "referee replay. Rounds commit `k` in batches, each chosen after observing "
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
    Prevented influence per arm: the column every blocking paper actually reports.

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
        key=lambda result: -(result.get("referee_prevented_influence") or 0.0),
    )
    head = at_largest[0]
    unopposed = head.get("referee_unopposed_spread") or head.get("unopposed_spread") or 0.0

    lines = [
        f"## Prevented influence at k={largest}",
        "",
        f"`prevented = sigma(S_N, empty) - sigma(S_N | blockers)`, both terms measured "
        f"on the shared referee. The rumour was seeded by "
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
        "is its own column below: CLDAG's Table 2 shows it takes 20-30x the rumour's "
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
        prevented = result.get("referee_prevented_influence")
        percent = result.get("referee_prevented_pct_of_unopposed")
        ratio = result.get("budget_ratio")
        lines.append(
            f"| `{result['arm']}` "
            f"| {_format_number(ground_truth_reward(result))} "
            f"| {', ' if prevented is None else f'{prevented:+.2f}'} "
            f"| {', ' if percent is None else f'{percent:.1f}%'} "
            f"| {_format_number(result.get('prevented_pct_of_nodes'))} "
            f"| {', ' if ratio is None else f'{ratio:.2f}'} "
            f"| {result.get('n_spent', ', ')} |"
        )

    lines += [
        "",
        "`prevented` counts only nodes the rumour **would otherwise have infected** "
        "(Budak's \"saved\" set): a blocker that protects a node the cascade never "
        "reaches scores exactly zero, however central that node is. An arm whose "
        "`spent` is below the budget ran out of candidates: the reachable region was "
        "smaller than `k`, which is common at the high end of the sweep and is the "
        "honest answer rather than a padded set.",
        "",
    ]

    return lines


def _epidemic_section(agent_results: list[dict]) -> list[str]:
    """
    Prevented infections and the outbreak's SHAPE: what an immunization table reports.

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
        key=lambda result: -(result.get("referee_prevented_infections") or 0.0),
    )
    head = at_largest[0]
    unprotected = (
        head.get("referee_unprotected_attack_rate")
        or head.get("unprotected_attack_rate")
        or 0.0
    )
    alpha = head.get("epi_alpha")

    lines = [
        f"## Prevented infections at k={largest}",
        "",
        f"`prevented = |R(inf)| unprotected - |R(inf)| with doses`, both terms "
        f"measured on the shared referee. The outbreak was seeded by "
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
        "rates above are stated for exactly that reason: this table is comparable "
        "to another run at the same rates and to nothing else.",
        "",
        "> **Read `eigendrop` as context, never as the score.** It is what the "
        "spectral line (NetShield, NetMelt, Gelling, GreedyWalk) actually optimizes, "
        "and it needs no simulator at all, so it is the only column this table and "
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
        curve = result.get("referee_curve") or result
        prevented = result.get("referee_prevented_infections")
        percent = result.get("referee_prevented_pct_of_unprotected")
        spectral = result.get("spectral") or {}
        eigendrop = spectral.get("eigendrop_pct")
        lines.append(
            f"| `{result['arm']}` "
            f"| {_format_number(ground_truth_reward(result))} "
            f"| {', ' if prevented is None else f'{prevented:+.2f}'} "
            f"| {', ' if percent is None else f'{percent:.1f}%'} "
            f"| {_format_number(curve.get('peak_prevalence'))} "
            f"| {curve.get('time_to_peak', ', ')} "
            f"| {_format_number(curve.get('auc_infectious'))} "
            f"| {', ' if eigendrop is None else f'{eigendrop:.1f}%'} "
            f"| {result.get('n_spent', ', ')} |"
        )

    lines += [
        "",
        "`prevented` counts only nodes the outbreak **would otherwise have reached**: "
        "a dose spent on a node the epidemic never gets to scores exactly zero, "
        "however central that node is. The infectious set also RECOVERS, so the "
        "outbreak burns out on its own and a dose is worth only what it saves before "
        "then, which is why `t_peak` moving earlier is a real result even when the "
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
    What each removal set did to the graph's CONNECTIVITY: context, not the score.

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
        "target**: a k-layer message-passing model cannot represent "
        "giant-component membership on a graph of diameter > k, so the world model "
        "is fit on the diffusion transition and these describe the same removal "
        "sets afterwards "
        "([`research/critical_node_detection.md`](../../../../research/critical_node_detection.md) "
        "§2.1, §8.3). They are the units the published dismantling literature "
        "reports in, so they are the bridge to it, and §5.8 shows a method can "
        "win the column above and lose every column here.",
        "",
        f"Intact graph: pairwise connectivity {intact['pairwise_conn_intact']:,.0f}, "
        f"largest component {intact['largest_cc_intact']:,.0f}, "
        f"{intact['n_components_intact']:.0f} component(s). "
        f"`rho` is the fraction of N removed to drive the giant component below "
        f"{intact['gcc_threshold']:.0%} of N; `, ` means the budget ran out first, "
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
            f"| {', ' if rho is None else f'{rho:.3f}'} "
            f"| {structural['degree_rank_spearman']:+.3f} |"
        )

    lines += [
        "",
        "**`degree-rank rho`** is the Spearman correlation between an arm's removal "
        "ORDER and the degree of the nodes it removed: the self-measurement "
        "§9.5 asks for. MIND found GDM's dismantling order correlates at **0.762** "
        "with a PCA of its own handcrafted input features. We feed `log1p(degree)` "
        "as feature channel 2, so an arm near that value has re-derived the degree "
        "heuristic with extra steps, whatever its spread number says.",
        "",
    ]

    return lines


def _ring_note(agent_results: list[dict]) -> list[str]:
    """
    Which budgets sit above the outbreak's one-hop ring, and so separate nothing.

    Two information states share this table. The planner and `frontier_removal`
    are handed the outbreak; every published dismantler is blind to it and would
    pick the same set for any outbreak. A budget at or above the ring makes the
    first group trivially exact (delete the ring, score |S|), so only the rows
    below it compare methods rather than information.
    """
    rings = {
        result["budget"]: result["outbreak_ring"]
        for result in agent_results
        if result.get("outbreak_ring") is not None
    }
    if not rings:
        return []

    ring = max(rings.values())
    sources = max(
        len(result.get("outbreak") or []) for result in agent_results
    )
    trivial = sorted(budget for budget in rings if budget >= ring)
    lines = [
        f"> **Two information states share this table.** The agent arms and "
        f"`frontier_removal` are told the {sources} outbreak sources; every "
        f"published dismantler is blind to them and returns the same set for any "
        f"outbreak. The outbreak's one-hop ring is **{ring} nodes**, and a budget "
        f"at or above it is trivial for any outbreak-aware arm: delete the ring "
        f"and the cascade cannot leave the sources, so the row reads exactly "
        f"{sources} and separates information, not methods.",
    ]
    if trivial:
        lines.append(
            f"> Trivial budgets in this run: k = {', '.join(str(k) for k in trivial)}. "
            f"Read only the rows below {ring}; raise `--outbreak-pct` to put the "
            f"whole sweep under the ring."
        )
    else:
        lines.append(
            "> Every budget in this run sits below the ring, so every row is a "
            "choice of WHICH neighbours to delete."
        )

    return lines + [""]


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
    elif is_forecast(at_largest):
        # An error, not a count: "spread 0.06 (0.00% of N)" is what the cascade
        # branch printed over an MSLE on the first casflow_aps run
        headline = (
            f"held-out {reward_name(at_largest)} {_format_number(spread, 4)} "
            f"(lower is better) over {winner.get('n_eval_instances', '?')} "
            f"`{winner.get('eval_split', '?')}` cascades"
        )
        title = "## Winning arm (held-out cascades)"
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
        f"**`{winner['arm']}`** (condition {winner.get('condition', ', ')}: "
        f"{condition_names.get(winner.get('condition'), 'unknown')}): {headline}, "
        f"model `{winner.get('model')}`",
        "",
        "```",
        str(winner.get("summary", "")).strip(),
        "```",
        "",
    ]

    # The agent's own account of how it got here: written before the program so
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


# Result keys that are counts of things, printed without decimals
wm_count_keys = (
    "plan_n_graphs", "ens_n_samples", "ens_n_episodes", "n_pairs", "n_nodes_scored",
    "n_nodes_with_effect", "n_with_action", "n_swappable", "seeded_p_infected_n",
    "removed_p_frontier_n", "n_planning_graphs",
)


def _wm_value(value: object, digits: int = 4, key: str = "") -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"

    if key in wm_count_keys and isinstance(value, (int, float)):
        return str(int(round(float(value))))

    if isinstance(value, (int, float)):
        return _format_number(value, digits)

    return str(value)


def _wm_table(header: tuple, rows: list[tuple]) -> list[str]:
    if not rows:
        return []

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]

    return lines + [""]


def _wm_provenance(wm_results: dict, metadata: dict | None) -> list[str]:
    config = wm_results.get("config", {})
    history = wm_results.get("history") or []
    best_epoch = (
        max(history, key=lambda entry: entry["val_delta_f1"])["epoch"] if history else None
    )
    straddling = (metadata or {}).get("graphs_straddling_splits")
    split_mode = wm_results.get("split_mode") or (metadata or {}).get("split_mode")
    split_note = (
        f"{split_mode} ({len(straddling)} graph(s) straddle train/test)"
        if split_mode and straddling
        else f"{split_mode} (no graph straddles a split)"
        if split_mode and straddling == []
        else str(split_mode)
    )
    layout = (
        "compartmental (9 channels in, 5 out)"
        if config.get("epidemic")
        else "competitive (8 channels in, 4 out)"
        if config.get("competitive")
        else f"single cascade (`{config.get('action_encoding', 'basic')}` action encoding)"
    )
    rows = [
        ("Backbone / Head", f"`{config.get('model')}` / `{config.get('head')}`"),
        ("Dynamics", f"`{config.get('diffusion_model')}`"),
        ("State Layout", layout),
        ("Hidden Dim / Layers", f"{config.get('hidden_dim')} / {config.get('n_layers')}"),
        ("Remove Semantics", f"`{config.get('remove_semantics')}`"),
        ("Edge Weights Hidden", _wm_value(bool(config.get("hide_edge_weights", False)))),
        ("Split Mode", split_note),
        ("Checkpoint Format", f"`{wm_results.get('checkpoint_format', 'bare state_dict')}`"),
        ("Epochs Run / Checkpoint Epoch", f"{len(history)} / {best_epoch}"),
        ("Best Validation Delta F1", _wm_value(wm_results.get("best_val_delta_f1"))),
        ("Training Time", f"{_format_number(wm_results.get('train_seconds'), 1)} s"),
    ]

    return ["### Provenance", ""] + _wm_table(("Setting", "Value"), rows)


def _wm_one_step(test: dict) -> list[str]:
    persistence = test.get("persistence") or {}
    scored = (
        "delta_f1", "new_infection_f1", "infected_acc", "frontier_acc",
        "brier_infected", "brier_frontier", "ece_infected", "ece_frontier",
        "compartment_acc", "compartment_f1", "brier_compartment",
        "pos_infected_acc", "pos_new_infection_f1",
    )
    rows = [
        (wm_metric_name(key), _wm_value(test[key], key=key), _wm_value(persistence[key], key=key) if key in persistence else "n/a")
        for key in scored
        if key in test
    ]
    action_rows = [
        (wm_metric_name(key), _wm_value(test[key]), "n/a")
        for key in ("add_seed_success", "remove_frontier_success", "dose_success",
                    "block_seed_success", "action_sensitivity")
        if key in test
    ]
    lines = ["### One-Step Accuracy (Test Split)", ""]
    lines += [
        "The persistence baseline predicts `s_{t+1} = s_t`; its change-F1 is zero by "
        "construction, so the Delta F1 and New-Infection F1 rows are the substantive "
        "ones and the accuracy rows are dominated by unchanged nodes. Brier and ECE "
        "score the probabilities against the simulator's own marginals (lower is better).",
        "",
    ]
    lines += _wm_table(("Metric", "World Model", "Persistence Baseline"), rows + action_rows)

    if test.get("delta_f1_is_new_infection_f1"):
        brier_pair = (test.get("brier_infected"), test.get("brier_frontier"))
        same_brier = (
            None not in brier_pair and abs(brier_pair[0] - brier_pair[1]) < 1e-9
        )
        lines += [
            "> The scored column is monotone, so a changed node is a newly infected node "
            "and Delta F1 and New-Infection F1 are the same quantity: one piece of "
            "evidence, not two."
            + (
                " The two Brier and ECE rows coincide for the same reason: the "
                "structured head composes both columns from one per-node probability, so "
                "their errors are identical node for node."
                if same_brier
                else ""
            ),
            "",
        ]

    return lines


def _wm_rollout(wm_results: dict) -> list[str]:
    rollout = wm_results.get("rollout") or {}
    if not rollout:
        return []

    rows = [
        (wm_metric_name(key), _wm_value(value, 3, key=key))
        for key, value in rollout.items()
        if not key.endswith("_curve")
    ]
    bias = rollout.get("ens_count_bias")
    reading = (
        "" if bias is None
        else "A count bias near zero means the free-running rollout neither saturates "
        f"nor dies early; this run reads {float(bias):+.2f} nodes per step against "
        f"a final cascade of {_format_number(rollout.get('ens_final_count_true'), 1)}."
    )
    lines = ["### Rollout Fidelity", ""]
    lines += [
        "A sampled ensemble of the world model is rolled forward under the recorded "
        "action sequence and compared with the same number of simulator rollouts. "
        + reading,
        "",
    ]
    lines += _wm_table(("Metric", "Value"), rows)

    ood = wm_results.get("rollout_ood") or {}
    if ood:
        lines += ["### Off-Policy Rollouts", ""]
        lines += [
            "The same comparison under action policies the training data never "
            "contained, which is the distribution a planner actually proposes.",
            "",
        ]
        keys = ["ens_marg_mae", "ens_count_w1", "ens_count_bias", "ens_final_count_model", "ens_final_count_true"]
        header = ("Policy",) + tuple(wm_metric_name(key) for key in keys)
        rows = [("recorded",) + tuple(_wm_value(rollout.get(key), 3, key=key) for key in keys)]
        rows += [
            (name,) + tuple(_wm_value(block.get(key), 3, key=key) for key in keys)
            for name, block in ood.items()
        ]
        lines += _wm_table(header, rows)

    return lines


def _wm_action_conditioning(block: dict | None) -> list[str]:
    if not block:
        return []

    effect = block.get("counterfactual_effect") or {}
    ablation = block.get("ablation") or {}
    exogenous = block.get("exogenous") or {}
    lines = ["### Action Conditioning", ""]
    lines += [
        "Three falsifiable tests of whether the prediction depends on the intervention "
        "rather than on state persistence. The counterfactual effect compares the "
        "predicted change between two actions at the same state with the simulator's "
        "true change; a model that predicts no effect scores exactly 1.0 on the "
        "normalized error, so that is the null to beat. The ablation re-scores the same "
        "states with actions zeroed and shuffled. The exogenous test checks the closed "
        "form of the immediate action effect at its worst case.",
        "",
    ]

    effect_keys = ("effect_mae_norm", "effect_pearson", "effect_sign_agree", "effect_magnitude_ratio", "effect_mae", "n_pairs", "n_nodes_with_effect")
    rows = [(wm_metric_name(key), _wm_value(effect[key], key=key)) for key in effect_keys if key in effect]
    per_op = effect.get("per_op") or {}
    for op, values in per_op.items():
        if isinstance(values, dict) and values.get("effect_mae_norm") is not None:
            rows.append((f"Normalized Effect MAE ({pretty_arm(op)})", _wm_value(values["effect_mae_norm"])))
    if rows:
        lines += ["**Counterfactual effect** (null = 1.0):", ""]
        lines += _wm_table(("Metric", "Value"), rows)

    ablation_keys = ("baseline_delta_f1", "null_delta_f1", "null_delta_f1_drop", "shuffle_delta_f1", "shuffle_delta_f1_drop",
                     "baseline_brier_infected", "null_brier_infected", "shuffle_brier_infected", "n_with_action", "n_swappable")
    rows = [(wm_metric_name(key), _wm_value(ablation[key], key=key)) for key in ablation_keys if ablation.get(key) is not None]
    if rows:
        lines += ["**Action ablation** (a drop near zero means the accuracy was obtainable without reading the action):", ""]
        lines += _wm_table(("Metric", "Value"), rows)

    exo_keys = ("seeded_p_infected_worst", "seeded_p_infected_mean", "seeded_p_infected_exact_frac", "seeded_p_infected_n",
                "removed_p_frontier_worst", "removed_p_frontier_mean", "removed_p_frontier_exact_frac", "removed_p_frontier_n")
    rows = [(wm_metric_name(key), _wm_value(exogenous[key], 6, key=key)) for key in exo_keys if exogenous.get(key) is not None]
    if rows:
        lines += ["**Exogenous fidelity** (a seeded node must read 1.0, a removed node 0.0):", ""]
        lines += _wm_table(("Metric", "Value"), rows)

    verdict = block.get("verdict")
    if verdict:
        status = "PASS" if block.get("action_conditioned") else "FAIL"
        lines += [f"> **Verdict: {status}.** {verdict}", ""]

    return lines


def _wm_planning(wm_results: dict) -> list[str]:
    lines = []
    for title, key in (("Planning Regret (One-Step)", "planning"), ("Planning Regret (k-Seed, Full Horizon)", "planning_budget")):
        block = wm_results.get(key)
        if not isinstance(block, dict):
            continue

        rows = [
            (wm_metric_name(name), _wm_value(value), _wm_value(block.get(f"{name}_std"), 4) if f"{name}_std" in block else "n/a")
            for name, value in block.items()
            if "_regret_" in name and not name.endswith("_std") and value is not None
        ]
        if not rows:
            continue

        lines += [f"### {title}", ""]
        lines += [
            "Regret is the spread lost against the oracle's choice when the world model "
            "picks the intervention, beside the same choice made by a degree heuristic "
            "and at random. A useful planner beats random decisively and at least "
            "matches degree. The one-step form scores a single intervention by its "
            "next-step marginal, which is a sanity check rather than the multi-step "
            "claim the outer loop rests on.",
            "",
        ]
        lines += _wm_table(("Chooser", "Regret", "Std Across Graphs"), rows)
        provenance = [
            ("Planning Split", block.get("planning_split")),
            ("Graphs Scored", block.get("plan_n_graphs")),
            ("Overlap with Train / Val", f"{block.get('train_overlap')} / {block.get('val_overlap')}"
             if "train_overlap" in block else None),
        ]
        lines += ["Provenance: " + "; ".join(f"{name} {_wm_value(value, 0)}" for name, value in provenance if value is not None) + ".", ""]

    return lines


def _wm_in_loop_fidelity(agent_results: list[dict]) -> list[str]:
    """The world model as the outer loop saw it: its estimate against the referee replay."""
    # An inverse or forecast reward carries no evaluator noise and `referee_reward` IS
    # the held-out score there, so a "bias" column would print +0.00 under a
    # "spread" header over an F1 or an MSLE
    if is_recover(agent_results) or is_forecast(agent_results):
        return []

    paired = [
        result for result in agent_results
        if result.get("referee_reward") is not None and result.get("evaluator") in ("world_model", "oracle")
    ]
    if not paired:
        return []

    rows = []
    for result in sorted(paired, key=lambda item: (item["arm"], item.get("budget", 0))):
        estimate = result.get("wm_reeval_mean", result["reward"])
        rows.append((
            pretty_arm(result["arm"]),
            _wm_value(result.get("budget"), 0),
            _wm_value(estimate, 2),
            _wm_value(result["referee_reward"], 2),
            f"{float(estimate) - float(result['referee_reward']):+.2f}",
        ))

    return ["### In-Loop Evaluator Fidelity", "", 
            "What each model-based evaluator believed about its own winning strategy against "
            "the shared referee's replay of that strategy. This is the number the "
            "outer loop actually depends on; the offline rollout fidelity above is its "
            "prediction.", ""] + _wm_table(
        ("Arm", "Budget", "Evaluator Estimate", "Referee Spread", "Bias"), rows
    )


def _world_model_section(
    wm_results: dict | None,
    metadata: dict | None = None,
    agent_results: list[dict] | None = None,
) -> list[str]:
    lines = ["## World Model", ""]

    if wm_results is None:
        lines += [
            "_Not trained for this run (oracle or Monte Carlo evaluator)._",
            "",
        ]
        lines += _wm_in_loop_fidelity(agent_results or [])

        return lines

    lines += [
        "The learned transition model $f_\\theta(G, s_t, a_t) \\to s_{t+1}$ that the "
        "`@world_model` arms use in place of the simulator. Every number below comes "
        "from the train stage's results JSON; the figures `wm_*.png` draw the same "
        "blocks.",
        "",
    ]
    lines += _wm_provenance(wm_results, metadata)
    lines += _wm_one_step(wm_results.get("test") or {})
    lines += _wm_action_conditioning(wm_results.get("action_conditioning"))
    lines += _wm_rollout(wm_results)
    lines += _wm_planning(wm_results)
    lines += _wm_in_loop_fidelity(agent_results or [])

    return lines


def _stage_timings(layout: Layout, agent_results: list[dict]) -> list[str]:
    """Per-stage wall time from pipeline.json, and the total the run cost so far.

    The agent stage is split from the per-arm `elapsed_seconds`: baseline rows
    (conditions 1, 7 and 9) against our method arms (conditions 2-6), with
    whatever the stage spent outside any arm shown as overhead.
    """
    if not layout.manifest_path.exists():
        return []

    stages = json.loads(layout.manifest_path.read_text()).get("stages") or {}
    if not stages:
        return []

    lines = [
        "## Stage Timings",
        "",
        "| Stage | Status | Time |",
        "| --- | --- | --- |",
    ]
    total = 0.0
    for stage, entry in stages.items():
        seconds = entry.get("seconds")
        if seconds is not None:
            total += float(seconds)
        rendered = "" if seconds is None else f"{float(seconds):,.1f} s"
        lines.append(f"| {stage} | {entry.get('status')} | {rendered} |")

        if stage == "agent" and seconds is not None and agent_results:
            # A published repo's own selection (DeepIM trains for hours) is
            # recorded beside our referee-side elapsed, not inside it
            baseline_seconds = sum(
                float(result.get("elapsed_seconds") or 0.0)
                + float((result.get("external") or {}).get("selection_seconds") or 0.0)
                for result in agent_results
                if result.get("condition") in (1, 7, 9)
            )
            method_seconds = sum(
                float(result.get("elapsed_seconds") or 0.0)
                for result in agent_results
                if result.get("condition") not in (1, 7, 9)
            )
            skipped_seconds = 0.0
            for marker in layout.baselines_dir.glob("*/*.skipped.json"):
                skipped_seconds += float(
                    json.loads(marker.read_text()).get("seconds") or 0.0
                )
            overhead = (
                float(seconds) - baseline_seconds - method_seconds - skipped_seconds
            )
            lines.append(
                f"| &nbsp;&nbsp;baselines (conditions 1, 7, 9, incl. external "
                f"selection) | | {baseline_seconds:,.1f} s |"
            )
            lines.append(
                f"| &nbsp;&nbsp;our arms (conditions 2-6) | | "
                f"{method_seconds:,.1f} s |"
            )
            if skipped_seconds > 1.0:
                lines.append(
                    f"| &nbsp;&nbsp;failed baseline attempts | | "
                    f"{skipped_seconds:,.1f} s |"
                )
            if overhead > 1.0:
                lines.append(
                    f"| &nbsp;&nbsp;stage overhead | | {overhead:,.1f} s |"
                )

    hours = total / 3600.0
    lines += [
        "",
        f"**Total recorded: {total:,.1f} s ({hours:.2f} h).** The report stage is "
        "still running when this file is written, so its own time is excluded; a "
        "resumed run's total sums every stage recorded in `pipeline.json` across "
        "submissions, not wall-clock between them.",
        "",
    ]

    return lines


def _figures_section(plots_dir: Path, report_parent: Path) -> list[str]:
    figures = sorted(plots_dir.glob("*.png"))
    if not figures:
        return []

    lines = ["## Figures", ""]
    for figure in figures:
        # Same paper-ready naming as the figure titles themselves
        title = prettify(figure.stem)
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
        f"# {layout.label}: Graph World Model results",
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
    lines += _world_model_section(wm_results, metadata, agent_results)
    lines += _stage_timings(layout, agent_results)
    lines += _figures_section(layout.plots_dir, layout.root)

    layout.report_path.write_text("\n".join(lines))

    return layout.report_path
