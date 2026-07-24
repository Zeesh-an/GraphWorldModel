"""One markdown file per dataset: config, results tables, figures, winning program."""

import json
from pathlib import Path

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


def _results_table(agent_results: list[dict]) -> list[str]:
    if not agent_results:
        return []

    has_mc = any(result.get("mc_reward") is not None for result in agent_results)
    header = ["| budget k | % of N | arm | spread | % of N |"]
    divider = ["| --- | --- | --- | --- | --- |"]

    if has_mc:
        header[0] += " MC spread | evaluator − MC |"
        divider[0] += " --- | --- |"

    header[0] += " seconds |"
    divider[0] += " --- |"

    rows = []
    ordered = sorted(
        agent_results, key=lambda result: (result["budget"], result["arm"])
    )

    for result in ordered:
        row = (
            f"| {result['budget']} | {_format_number(result['budget_pct'])} "
            f"| `{result['arm']}` | {_format_number(result['reward'])} "
            f"| {_format_number(result['spread_pct'])} |"
        )

        if has_mc:
            mc_reward = result.get("mc_reward")
            gap = result.get("wm_reeval_minus_mc", result.get("wm_minus_mc"))
            row += f" {_format_number(mc_reward)} |"
            row += (
                f" {float(gap):+.2f} |" if isinstance(gap, (int, float)) else " — |"
            )

        row += f" {_format_number(result.get('elapsed_seconds'), 1)} |"
        rows.append(row)

    return ["## Results", ""] + header + divider + rows + [""]


def _winner_section(agent_results: list[dict]) -> list[str]:
    """Best arm at the largest budget, with the program it produced."""
    if not agent_results:
        return []

    largest = max(result["budget"] for result in agent_results)
    at_largest = [result for result in agent_results if result["budget"] == largest]
    winner = max(at_largest, key=lambda result: result["reward"])

    lines = [
        f"## Winning arm at k={largest}",
        "",
        f"**`{winner['arm']}`** — spread {_format_number(winner['reward'])} "
        f"({_format_number(winner['spread_pct'])}% of N), model `{winner.get('model')}`",
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
    lines += _results_table(agent_results)
    lines += _winner_section(agent_results)
    lines += _world_model_section(wm_results)
    lines += _figures_section(layout.plots_dir, layout.root)

    layout.report_path.write_text("\n".join(lines))

    return layout.report_path
