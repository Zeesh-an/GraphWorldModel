"""One markdown file per dataset: config, results tables, figures, winning program."""

import json
from pathlib import Path

from pipeline.conditions import condition_names, ground_truth_reward, is_ground_truth
from pipeline.layout import Layout

# Config keys worth printing; the manifest JSON holds the exhaustive version
reported_config_keys = (
    "dataset",
    "tag",
    "evaluator",
    "diffusion_model",
    "llm_model",
    "outer_iters",
    "horizon",
    "n_samples",
    "mc_runs",
    "native_mc_runs",
    "allowed_ops",
    "wm_model",
    "head",
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
    lines = ["## Results", ""]

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

    header = "| # | budget k | % of N | arm | evaluator | spread | % of N |"
    divider = "| --- | --- | --- | --- | --- | --- | --- |"

    if has_mc:
        header += " estimate | est − spread |"
        divider += " --- | --- |"

    header += " real episodes | seconds |"
    divider += " --- | --- |"

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
        row += f" {_format_number(result.get('elapsed_seconds'), 1)} |"
        rows.append(row)

    return lines + [header, divider] + rows + [""]


def _winner_section(agent_results: list[dict]) -> list[str]:
    """Best arm at the largest budget, with the program it produced."""
    if not agent_results:
        return []

    largest = max(result["budget"] for result in agent_results)
    at_largest = [result for result in agent_results if result["budget"] == largest]
    winner = max(at_largest, key=ground_truth_reward)
    spread = ground_truth_reward(winner)

    judged = "ground-truth MC" if is_ground_truth(at_largest) else "its own evaluator"
    lines = [
        f"## Winning arm at k={largest}",
        "",
        f"**`{winner['arm']}`** (condition {winner.get('condition', '—')} — "
        f"{condition_names.get(winner.get('condition'), 'unknown')}) — spread "
        f"{_format_number(spread)} "
        f"({_format_number(100.0 * spread / winner['graph']['num_nodes'])}% of N) "
        f"by {judged}, model `{winner.get('model')}`",
        "",
        "```",
        str(winner.get("summary", "")).strip(),
        "```",
        "",
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
        f"# {config['tag']} — Graph World Model results",
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
    lines += _winner_section(agent_results)
    lines += _world_model_section(wm_results)
    lines += _figures_section(layout.plots_dir, layout.root)

    layout.report_path.write_text("\n".join(lines))

    return layout.report_path
