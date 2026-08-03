"""One markdown file per dataset: config, results tables, figures, winning program."""

import json
import re
from pathlib import Path

from coding_agent.types import best_by, rank_by
from pipeline.conditions import (
    adaptivity_gaps,
    condition_names,
    ground_truth_reward,
    is_ground_truth,
    result_sense,
    reward_direction,
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


def _results_table(agent_results: list[dict]) -> list[str]:
    if not agent_results:
        return []

    has_mc = any(result.get("mc_reward") is not None for result in agent_results)
    sense = result_sense(agent_results)
    lines = ["## Results", ""]

    if sense == minimize:
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

    judged = "ground-truth MC" if is_ground_truth(at_largest) else "its own evaluator"
    lines = [
        f"## Winning arm at k={largest}",
        "",
        f"**`{winner['arm']}`** (condition {winner.get('condition', '—')} — "
        f"{condition_names.get(winner.get('condition'), 'unknown')}) — spread "
        f"{_format_number(spread)} "
        f"({_format_number(100.0 * spread / winner['graph']['num_nodes'])}% of N, "
        f"{reward_direction(sense)}) "
        f"by {judged}, model `{winner.get('model')}`",
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
    lines += _structural_section(agent_results)
    lines += _winner_section(agent_results)
    lines += _world_model_section(wm_results)
    lines += _figures_section(layout.plots_dir, layout.root)

    layout.report_path.write_text("\n".join(lines))

    return layout.report_path
