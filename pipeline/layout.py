"""
Where every artifact lives. One tag -> one directory tree under results/.

results/<tag>/
    data/                       generated transitions + graph store (stage: data)
        graphs/, graphs_index.json, transitions_<dm>_<split>.jsonl, metadata.json
    world_model/                checkpoints + training results (stage: train)
        wm_<model>_<dm>.pt, <model>_<dm>.json
    agent/<budget_label>/       one JSON per arm (stage: agent)
        baseline_degree_discount.json, evolve_scored.json, ...
    plots/                      paper figures (stage: plots)
    report.md                   final write-up (stage: report)
    pipeline.json               run manifest: config + per-stage status
"""

from pathlib import Path

results_root = Path("results")

# Written next to the results when an arm could not run; same directory and same
# extension, so every reader must filter on it
skip_marker_suffix = ".skipped.json"


def budget_label(budget_pct: float | None, budget: int) -> str:
    """Directory name for one point of the budget sweep."""
    return f"pct{budget_pct:g}" if budget_pct is not None else f"k{budget}"


class Layout:
    def __init__(self, tag: str, root: Path | str = results_root) -> None:
        self.root = Path(root) / tag
        self.tag = tag

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def world_model_dir(self) -> Path:
        return self.root / "world_model"

    @property
    def agent_dir(self) -> Path:
        return self.root / "agent"

    @property
    def baselines_dir(self) -> Path:
        """External published baselines (condition 7), kept apart from our arms."""
        return self.root / "baselines"

    @property
    def plots_dir(self) -> Path:
        return self.root / "plots"

    @property
    def report_path(self) -> Path:
        return self.root / "report.md"

    @property
    def manifest_path(self) -> Path:
        return self.root / "pipeline.json"

    def data_metadata(self) -> Path:
        return self.data_dir / "metadata.json"

    def wm_results(self, model: str, diffusion_model: str) -> Path:
        return self.world_model_dir / f"{model}_{diffusion_model}.json"

    def wm_checkpoint(self, model: str, diffusion_model: str) -> Path:
        return self.world_model_dir / f"wm_{model}_{diffusion_model}.pt"

    def agent_result(self, label: str, arm: str, external: bool = False) -> Path:
        root = self.baselines_dir if external else self.agent_dir

        return root / label / f"{arm}.json"

    def result_globs(self) -> list[Path]:
        """
        Every result file, ours and external, in one list for plots/report.

        `<arm>.skipped.json` markers live in the same directories and match the
        same glob, but they record why an arm did NOT run and carry none of the
        result fields — so they are excluded here rather than at every reader.
        """
        return [
            path
            for path in sorted(self.agent_dir.glob("*/*.json"))
            + sorted(self.baselines_dir.glob("*/*.json"))
            if not path.name.endswith(skip_marker_suffix)
        ]

    def skip_globs(self) -> list[Path]:
        return sorted(self.baselines_dir.glob(f"*/*{skip_marker_suffix}")) + sorted(
            self.agent_dir.glob(f"*/*{skip_marker_suffix}")
        )
