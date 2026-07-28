"""
Registry of external published IM baselines.

Every external baseline reduces to ONE contract:

    (graph, budget, diffusion_model) -> list[int] seed set

We then score that seed set with **our own** Monte Carlo referee, exactly as we
score our own arms. This is deliberate: a paper's reported spread depends on its
simulator, its edge probabilities, its MC count, and sometimes its own graph
version, so their published numbers are not comparable to ours. Their *seed set*
is. Scoring every method's seeds under one referee is the only apples-to-apples
comparison available.

Each entry declares how to fetch the repo, how to hand it our graph, how to run
it, and how to read its seeds back. `status` records whether it can actually run:

    ready        - installed and runnable
    needs_setup  - registered, fetched by setup_baselines.py, not yet installed
    blocked      - cannot run for a stated structural reason (see `blocker`)
"""

import csv
import json
from functools import partial
import os
import pickle
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
import networkx as nx

# ABSOLUTE, anchored on this file. Every external baseline is launched with
# cwd set to its own repo directory, and POSIX resolves a relative argv[0] or
# interpreter path against that cwd — so a relative root silently becomes
# <repo>/baselines/external/<name>/baselines/external/<name>/... and vanishes.
baselines_root = Path(__file__).resolve().parent
external_root = baselines_root / "external"

classical = "classical"
learned = "learned"

# Cascades simulated per DeepIM .SG training file
deepim_training_samples = 1000


@dataclass
class ExternalBaseline:
    name: str
    kind: str  # classical | learned
    title: str
    venue: str
    repo: str
    paper: str
    entry: str  # what setup/run drives, for the README table
    status: str = "needs_setup"
    blocker: str | None = None
    python: str = "3.10"
    requirements: str | None = "requirements.txt"
    # "git" -> setup_baselines clones it; "manual" -> the authors publish only a
    # tarball (SourceForge), so setup prints instructions instead of guessing
    fetch: str = "git"
    build: list | None = None  # e.g. ["make"] for the C++ baselines
    # Some repos do not put the Makefile / entry point at the top level (OPIM
    # ships OPIM1.0 and OPIM1.1 side by side), so the build and run directory is
    # not always the clone directory
    subdir: str | None = None
    # ...and some ship a zip instead of source (SSA), which must be expanded
    # inside the clone before anything can be built
    unpack: str | None = None
    # (relative_path, old_text, new_text) compatibility edits applied after
    # clone. For repos abandoned against a library version that no longer
    # exists, where the alternative is installing a second multi-GB torch just
    # to pin an old networkx. Idempotent: re-running setup is a no-op.
    patches: list | None = None
    # (graph, work_dir, budget, diffusion_model) -> dict of extra command args
    export: object = None
    # (work_dir, budget, diffusion_model, extras) -> list[str] argv
    command: object = None
    # (work_dir, stdout, budget) -> list[int]
    parse_seeds: object = None
    notes: str = ""
    extra_env: dict = field(default_factory=dict)

    @property
    def root(self) -> Path:
        """Where the repo is cloned, and where a per-baseline venv lives."""
        return external_root / self.name

    @property
    def directory(self) -> Path:
        """Where the build runs and the entry point lives — usually the root."""
        return self.root / self.subdir if self.subdir else self.root

    @property
    def venv_python(self) -> Path:
        return self.root / ".venv" / "bin" / "python"

    def installed(self) -> bool:
        return self.root.exists() and any(self.root.iterdir())

    @property
    def wired(self) -> bool:
        """
        Whether an adapter exists to actually drive this repo.

        A registered repo with no command is documentation, not a baseline: it
        can be cloned but never run, so it is excluded from `all` rather than
        installed and then failing at the first budget.
        """
        return self.command is not None


# Shared helpers -------------------------------------------------------------


def _undirected_view(graph) -> nx.Graph:
    view = nx.Graph()
    view.add_nodes_from(range(graph.num_nodes))
    view.add_edges_from(zip(graph.edge_index[0], graph.edge_index[1], strict=True))
    view.remove_edges_from(nx.selfloop_edges(view))

    return view


def write_edgelist(graph, path: Path) -> None:
    os.makedirs(path.parent, exist_ok=True)
    view = _undirected_view(graph)

    with open(path, "w") as handle:
        for source, target in view.edges():
            handle.write(f"{source} {target}\n")


def parse_seed_integers(text: str, budget: int) -> list[int]:
    """Last resort: pull the final bracketed integer list out of stdout."""
    matches = re.findall(r"\[[\d,\s]+\]", text)
    if not matches:
        raise ValueError(
            f"could not find a seed list in the baseline's output; last 500 chars:\n"
            f"{text[-500:]}"
        )

    seeds = [int(value) for value in re.findall(r"\d+", matches[-1])]

    return seeds[:budget]


# MOEIM ----------------------------------------------------------------------


def _moeim_export(graph, work_dir: Path, budget: int, diffusion_model: str) -> dict:
    """
    MOEIM reads `data/graphs/graphs_cleaned/<name>.txt` (plain edgelist) and
    `data/graphs/graph_communities/<name>.csv`. The name must carry `_un`/`_di`
    because src/load.py switches graph class on that substring.
    """
    spec = external_baselines["moeim"]
    graph_name = "gwm_un"
    repo = spec.directory

    write_edgelist(graph, repo / "data/graphs/graphs_cleaned" / f"{graph_name}.txt")

    # Community assignment: MOEIM's community objective needs one row per node
    view = _undirected_view(graph)
    communities = nx.community.label_propagation_communities(view)
    membership = {
        node: index for index, community in enumerate(communities) for node in community
    }

    # Header must be exactly `node,comm` (influence_maximization.py reads
    # df["comm"]), and the label must be 1-BASED because it indexes with
    # comm_[i]-1. Every node needs a row, isolates included.
    community_path = repo / "data/graphs/graph_communities" / f"{graph_name}.csv"
    os.makedirs(community_path.parent, exist_ok=True)
    with open(community_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node", "comm"])
        for node in range(graph.num_nodes):
            writer.writerow([node, membership.get(node, 0) + 1])

    return {"graph_name": graph_name}


def _moeim_command(
    work_dir: Path, budget: int, diffusion_model: str, extras: dict, graph
) -> list[str]:
    # MOEIM's --k is a FRACTION of the network, not an absolute count
    fraction = max(budget / graph.num_nodes, 1e-6)
    model = "LT" if diffusion_model == "LT" else "WC"

    return [
        "python",
        "influence_maximization.py",
        "--graph",
        extras["graph_name"],
        "--k",
        f"{fraction:.6f}",
        "--model",
        model,
        "--no_simulations",
        "100",
        "--no_runs",
        "1",
        "--max_generations",
        "100",
        "--experimental_setup",
        "setting2",
        # Trailing separator is required: create_folder does
        # '{0}{1}-{2}...'.format(out_dir, graph_name, ...) with no separator of
        # its own, so without it the results land in a SIBLING of work_dir
        "--out_dir",
        f"{work_dir.resolve()}{os.sep}",
    ]


def _moeim_parse(work_dir: Path, stdout: str, budget: int) -> list[int]:
    """
    Read the non-dominated front from `run-N-population_default.csv`.

    src/utils.py::to_csv2 writes columns n_nodes, influence, nodes — where
    `nodes` is str(list) of the seed set. Take the highest-influence row that
    fits the budget; MOEIM is many-objective, so the front trades spread
    against seed count and the largest admissible set is not always the best.
    """
    best, best_influence = [], float("-inf")

    for path in sorted(work_dir.rglob("*_default.csv")):
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                seeds = [int(token) for token in re.findall(r"\d+", row.get("nodes", ""))]
                influence = float(row.get("influence") or 0.0)

                if seeds and len(seeds) <= budget and influence > best_influence:
                    best, best_influence = seeds, influence

    return best or parse_seed_integers(stdout, budget)


# ToupleGDD ------------------------------------------------------------------


relabel_filename = "relabel.json"


def _touplegdd_export(graph, work_dir: Path, budget: int, diffusion_model: str) -> dict:
    """
    ToupleGDD reads a weighted edgelist: `src dst prob` per line.

    Its Graph class builds the node set from the EDGES only, then sets
    num_nodes = len(nodes) while continuing to index tensors by RAW id. Any
    graph with isolated nodes therefore addresses past the end of its own
    embedding table — on GPU that surfaces as a device-side assert rather than
    an IndexError. NetScience has 128 isolates out of 1589.

    So the ids handed over are relabelled to a contiguous 0..M-1 over the nodes
    that actually appear in an edge, and mapped back in _touplegdd_parse.
    Dropping isolates costs nothing: a degree-0 node influences only itself.
    """
    spec = external_baselines["touplegdd"]
    path = spec.directory / "test_data" / "gwm_graph.txt"
    os.makedirs(path.parent, exist_ok=True)
    arcs = graph.edge_index.shape[1]

    endpoints = sorted(
        {int(graph.edge_index[0, column]) for column in range(arcs)}
        | {int(graph.edge_index[1, column]) for column in range(arcs)}
    )
    forward = {node: index for index, node in enumerate(endpoints)}

    with open(path, "w") as handle:
        for column in range(arcs):
            source = forward[int(graph.edge_index[0, column])]
            target = forward[int(graph.edge_index[1, column])]
            probability = float(graph.ic_probs[column])
            handle.write(f"{source} {target} {probability:.6f}\n")

    os.makedirs(work_dir, exist_ok=True)
    (work_dir / relabel_filename).write_text(json.dumps(endpoints))

    return {"graph_file": "test_data/gwm_graph.txt"}


def _touplegdd_command(
    work_dir: Path, budget: int, diffusion_model: str, extras: dict, graph
) -> list[str]:
    # main.py has no --output: runner.test() prints the selected set to stdout
    return [
        "python",
        "main.py",
        "--test",
        "--model_file",
        "tripling.ckpt",
        "--graph",
        extras["graph_file"],
        "--budget",
        str(budget),
    ]


def _touplegdd_parse(work_dir: Path, stdout: str, budget: int) -> list[int]:
    """
    runner.test() prints `Seeds: [3, 17, 42] | Reward: 88.1` per trial, twice
    over: once for the one-shot generation pass and again for the incremental
    one. Take the first line so the arm is deterministic rather than dependent
    on which pass happened to score better.
    """
    match = re.search(r"Seeds:\s*\[([^\]]*)\]", stdout)
    seeds = (
        [int(token) for token in re.findall(r"\d+", match.group(1))] if match else []
    )
    if not seeds:
        seeds = parse_seed_integers(stdout, budget)

    # Undo the contiguous relabelling applied in _touplegdd_export
    relabel_path = work_dir / relabel_filename
    if relabel_path.exists():
        endpoints = json.loads(relabel_path.read_text())
        seeds = [endpoints[node] for node in seeds if node < len(endpoints)]

    return seeds[:budget]


# DeepIM ---------------------------------------------------------------------


def _deepim_export(graph, work_dir: Path, budget: int, diffusion_model: str) -> dict:
    """
    DeepIM trains per (dataset, diffusion model, budget), so the `.SG` file has
    to exist before genim.py runs. baselines/deepim_data.py builds it from our
    own simulator — see that module for why this was previously thought blocked.
    """
    from baselines.deepim_data import build_sg_file, seed_rates

    spec = external_baselines["deepim"]
    dataset = "gwm"
    percentage = round(100.0 * budget / graph.num_nodes)

    # genim.py only accepts the four rates it was written for; snap to the nearest
    rate = min(seed_rates, key=lambda candidate: abs(candidate - percentage))
    target = spec.directory / "data" / f"{dataset}_mean_{diffusion_model}{10 * rate}.SG"

    # genim.py addresses the .SG by a fixed name encoding only dataset, dynamics
    # and rate — NOT the graph. A file left behind by a different dataset is
    # therefore reused silently, and its seeds are indices into the wrong node
    # set; run_baseline drops the out-of-range ones, leaving a handful that
    # score below random. Rebuild whenever the cached graph disagrees.
    if target.exists():
        with open(target, "rb") as handle:
            cached_nodes = pickle.load(handle)["adj"].shape[0]

        if cached_nodes != graph.num_nodes:
            print(
                f"[baseline:deepim] {target.name} holds a {cached_nodes}-node "
                f"graph but this one has {graph.num_nodes} — rebuilding"
            )
            target.unlink()

    if not target.exists():
        build_sg_file(
            data_dir=str(work_dir / "_graphstore"),
            dataset=dataset,
            diffusion_model=diffusion_model,
            seed_rate=rate,
            samples=deepim_training_samples,
            out_dir=spec.directory / "data",
            graph=graph,
        )

    # genim.py sizes its whole model from this file, so a mismatch here is the
    # difference between real seeds and indices into someone else's graph.
    # Stated every run rather than inferred later from out-of-range ids.
    with open(target, "rb") as handle:
        payload = pickle.load(handle)

    nodes, pairs = payload["adj"].shape[0], tuple(payload["inverse_pairs"].shape)
    print(
        f"[baseline:deepim] {target.name}: adj {nodes}x{nodes}, "
        f"inverse_pairs {pairs}, graph {graph.num_nodes} nodes"
    )

    if nodes != graph.num_nodes or pairs[1] != graph.num_nodes:
        raise ValueError(
            f"{target} describes a {nodes}-node graph (pairs {pairs}) but the "
            f"graph being scored has {graph.num_nodes} nodes. Delete it and "
            f"re-run: rm {target}"
        )

    return {"dataset": dataset, "seed_rate": rate}


def _deepim_command(
    work_dir: Path, budget: int, diffusion_model: str, extras: dict, graph
) -> list[str]:
    return [
        "python",
        "genim.py",
        "-d",
        extras["dataset"],
        "-dm",
        diffusion_model,
        "-sp",
        str(extras["seed_rate"]),
    ]


def _deepim_parse(work_dir: Path, stdout: str, budget: int) -> list[int]:
    """
    Read the `Seeds: [...]` line the genim.py patch adds.

    Anchoring on that prefix matters: the run prints 300 lines of
    `Iteration: N \t Total Loss:0.81` beforehand, so a bare integer scrape
    would return iteration numbers and losses.
    """
    match = re.search(r"GWM_SEEDS n=(\d+)\s*\[([^\]]*)\]", stdout)
    if match:
        model_nodes = int(match.group(1))
        seeds = [int(token) for token in re.findall(r"\d+", match.group(2))]
        print(
            f"[baseline:deepim] model reported n={model_nodes}, "
            f"{len(seeds)} seeds, max id {max(seeds) if seeds else -1}"
        )
        if seeds:
            return seeds[:budget]

    return parse_seed_integers(stdout, budget)


# GLIE -----------------------------------------------------------------------


def _glie_export(graph, work_dir: Path, budget: int, diffusion_model: str) -> dict:
    """
    GLIE reads the IMM-style pair: a weighted edgelist `<name>.inf` (src dst prob)
    plus an `attribute.txt` giving n and m.
    """
    spec = external_baselines["glie"]
    data_dir = spec.directory / "data" / "gwm"
    os.makedirs(data_dir, exist_ok=True)
    arcs = graph.edge_index.shape[1]

    with open(data_dir / "graph_ic.inf", "w") as handle:
        for column in range(arcs):
            source = int(graph.edge_index[0, column])
            target = int(graph.edge_index[1, column])
            handle.write(f"{source} {target} {float(graph.ic_probs[column]):.6f}\n")

    (data_dir / "attribute.txt").write_text(f"n={graph.num_nodes}\nm={arcs}\n")

    return {"dataset": "gwm"}


def _glie_command(
    work_dir: Path, budget: int, diffusion_model: str, extras: dict, graph
) -> list[str]:
    # celf_glie.py runs IM off the STORED GLIE model — no training required
    return [
        "python",
        "celf_glie.py",
        "--dataset",
        extras["dataset"],
        "--k",
        str(budget),
    ]


# C++ RIS family -------------------------------------------------------------
#
# SSA, OPIM, SubSIM, IMM and TIM all descend from Tang et al.'s RIS codebase and
# share a two-step shape: format the graph into a binary, then run the selector.
# They differ in flags and — critically — in node indexing.


def _ssa_export(graph, work_dir: Path, budget: int, diffusion_model: str) -> dict:
    """
    SSA's `el2bin` converter wants: first line `n m`, then `src dst weight` per
    line, with node indices starting at ONE. Our nodes are 0-indexed, so every
    id is shifted by +1 here and shifted back in _ssa_parse.

    Not to be confused with the repo's `format` tool, which is the OPTIONAL
    step 0 that derives edge probabilities. We supply our own IC probabilities,
    so that step is skipped entirely.
    """
    spec = external_baselines["ssa"]
    text_path = work_dir / "graph.txt"
    binary_path = work_dir / "graph.bin"
    arcs = graph.edge_index.shape[1]

    with open(text_path, "w") as handle:
        handle.write(f"{graph.num_nodes} {arcs}\n")
        for column in range(arcs):
            source = int(graph.edge_index[0, column]) + 1
            target = int(graph.edge_index[1, column]) + 1
            handle.write(f"{source} {target} {float(graph.ic_probs[column]):.6f}\n")

    converter = spec.directory / "el2bin"
    if not converter.exists():
        raise FileNotFoundError(
            f"SSA's `el2bin` binary is missing at {converter} — run "
            f"`make` in {spec.directory} (setup_baselines does this)"
        )

    # el2bin takes exactly two arguments: text in, binary out. Everything here is
    # resolved because cwd is the repo directory, and a relative path would be
    # re-resolved against it after subprocess chdirs.
    #
    # NOT check=True: el2bin.cpp ends with `return 1`, so a successful
    # conversion exits non-zero — the same quirk OPIM's format step has. The
    # produced .bin is the real signal.
    converted = subprocess.run(
        [
            str(converter.resolve()),
            str(text_path.resolve()),
            str(binary_path.resolve()),
        ],
        cwd=spec.directory,
        capture_output=True,
        text=True,
    )

    if not binary_path.exists() or binary_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"ssa: el2bin did not produce {binary_path} "
            f"(exit {converted.returncode})\n{converted.stdout}{converted.stderr}"
        )

    return {"binary": str(binary_path.resolve())}


def _ssa_command(
    work_dir: Path, budget: int, diffusion_model: str, extras: dict, graph
) -> list[str]:
    return [
        "./SSA",
        "-i",
        extras["binary"],
        "-k",
        str(budget),
        "-epsilon",
        "0.1",
        "-delta",
        "0.01",
        "-m",
        "LT" if diffusion_model == "LT" else "IC",
        "-o",
        str((work_dir / "seeds.txt").resolve()),
    ]


def _ssa_parse(work_dir: Path, stdout: str, budget: int) -> list[int]:
    seed_file = work_dir / "seeds.txt"
    text = seed_file.read_text() if seed_file.exists() else stdout
    # Undo the +1 applied in _ssa_export
    seeds = [int(token) - 1 for token in re.findall(r"\d+", text)]

    return [node for node in seeds if node >= 0][:budget]


def _opim_export(
    name: str, graph, work_dir: Path, budget: int, diffusion_model: str
) -> dict:
    """
    OPIM and SubSIM read `graphInfo/<gname>` as a plain weighted edgelist, then
    build their own binary from it via `-func=0`. Node ids are 0-indexed,
    matching ours, so no shifting. `name` is bound per-baseline with partial so
    each writes into its OWN repo.
    """
    spec = external_baselines[name]
    graph_name = "gwm"
    graph_path = spec.directory / "graphInfo" / graph_name
    os.makedirs(graph_path.parent, exist_ok=True)
    arcs = graph.edge_index.shape[1]

    # graphBase.h reads `numV numE` off the first line and then exactly numE
    # edges. Omitting the header makes it size the adjacency from the first edge
    # and index out of bounds — a segfault, not an error message.
    with open(graph_path, "w") as handle:
        handle.write(f"{graph.num_nodes} {arcs}\n")
        for column in range(arcs):
            source = int(graph.edge_index[0, column])
            target = int(graph.edge_index[1, column])
            weight = float(graph.ic_probs[column])
            handle.write(f"{source} {target} {weight:.6f}\n")

    # -func=0 converts the edgelist into the .vec.rvs.graph binary the selector
    # reads; it must run before -func=1 and is cheap enough to redo each time.
    # -mode=w means "with edge property": read our third column instead of
    # overwriting every weight with OPIM's own 1/in_degree WC setting.
    binary = _opim_binary(spec.directory)
    # NOT check=True: OPIM.cpp ends the format function with `return 1`, so a
    # successful format exits non-zero. The artifact is the real signal.
    formatted = subprocess.run(
        [str(binary), "-func=0", f"-gname={graph_name}", "-mode=w"],
        cwd=spec.directory,
        capture_output=True,
        text=True,
    )
    artifact = graph_path.with_suffix(".vec.rvs.graph")

    if not artifact.exists():
        raise FileNotFoundError(
            f"{name}: -func=0 did not produce {artifact} "
            f"(exit {formatted.returncode})\n{formatted.stdout}{formatted.stderr}"
        )

    return {"graph_name": graph_name, "binary": binary.name}


def _opim_binary(directory: Path) -> Path:
    """
    Find the compiled selector; OPIM and SubSIM name their binaries differently.

    Returned ABSOLUTE: every caller execs it through subprocess with cwd set to
    the repo directory, and POSIX chdirs before exec — so a relative path would
    be resolved a second time against that cwd and vanish.
    """
    for pattern in ("*.o", "OPIM*", "subsim*", "SUBSIM*"):
        for candidate in sorted(directory.glob(pattern)):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate.resolve()

    raise FileNotFoundError(
        f"no compiled binary found in {directory} — run `make` there "
        f"(setup_baselines does this via the spec's build step)"
    )


def _opim_command(
    name: str, work_dir: Path, budget: int, diffusion_model: str, extras: dict, graph
) -> list[str]:
    # -func=1 selects seeds; -pdist=load reads the per-edge probabilities we wrote
    return [
        f"./{extras['binary']}",
        "-func=1",
        f"-gname={extras['graph_name']}",
        "-alg=opim-c",
        "-mode=2",
        f"-seedsize={budget}",
        "-eps=0.01",
        f"-model={'LT' if diffusion_model == 'LT' else 'IC'}",
        "-pdist=load",
    ]


def _subsim_export(graph, work_dir: Path, budget: int, diffusion_model: str) -> dict:
    """
    SubSIM is a fork of OPIM and reads the SAME `numV numE` + edges layout, but
    its CLI diverged: -func takes the strings format|im rather than 0|1, there
    is no -mode, and the weight column is selected with -pdist=weights. It also
    returns 0 from the format step where OPIM returns 1.
    """
    spec = external_baselines["subsim"]
    graph_name = "gwm"
    graph_path = spec.directory / "graphInfo" / graph_name
    os.makedirs(graph_path.parent, exist_ok=True)
    arcs = graph.edge_index.shape[1]

    with open(graph_path, "w") as handle:
        handle.write(f"{graph.num_nodes} {arcs}\n")
        for column in range(arcs):
            source = int(graph.edge_index[0, column])
            target = int(graph.edge_index[1, column])
            weight = float(graph.ic_probs[column])
            handle.write(f"{source} {target} {weight:.6f}\n")

    binary = _opim_binary(spec.directory)
    formatted = subprocess.run(
        [str(binary), "-func=format", f"-gname={graph_name}", "-pdist=weights"],
        cwd=spec.directory,
        capture_output=True,
        text=True,
    )
    artifact = graph_path.with_suffix(".vec.rvs.graph")

    if not artifact.exists():
        raise FileNotFoundError(
            f"subsim: -func=format did not produce {artifact} "
            f"(exit {formatted.returncode})\n{formatted.stdout}{formatted.stderr}"
        )

    return {"graph_name": graph_name, "binary": binary.name}


def _subsim_command(
    work_dir: Path, budget: int, diffusion_model: str, extras: dict, graph
) -> list[str]:
    return [
        f"./{extras['binary']}",
        "-func=im",
        f"-gname={extras['graph_name']}",
        f"-seedsize={budget}",
        "-eps=0.01",
        "-pdist=weights",
    ]


def _opim_parse(name: str, work_dir: Path, stdout: str, budget: int) -> list[int]:
    """
    Read the seed file this budget produced.

    OPIM writes result/seed/seed_<graph>_<alg>_<mode>_k<budget>_<pdist>, and a
    budget sweep leaves every previous run's file in that directory. A plain
    sorted() returns k16 for k=79 and k=318 alike (string order), so the budget
    has to be matched explicitly; mtime is the fallback if the naming changes.
    """
    spec = external_baselines[name]

    def usable(paths):
        return [
            path
            for path in paths
            if path.is_file() and path.stat().st_size > 0
        ]

    candidates = usable(spec.directory.rglob(f"*seed*_k{budget}_*"))
    if not candidates:
        candidates = sorted(
            usable(spec.directory.rglob("*seed*")),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

    for seed_file in candidates:
        numbers = re.findall(r"\d+", seed_file.read_text())
        if numbers:
            return [int(token) for token in numbers][:budget]

    return parse_seed_integers(stdout, budget)


# Registry -------------------------------------------------------------------

external_baselines: dict[str, ExternalBaseline] = {
    "moeim": ExternalBaseline(
        name="moeim",
        kind=learned,
        title="MOEIM — Many-Objective Evolutionary Influence Maximization",
        venue="GECCO 2024",
        repo="https://github.com/eliacunegatti/MOEIM",
        paper="https://arxiv.org/abs/2403.18755",
        entry="influence_maximization.py",
        status="needs_setup",
        patches=[
            # --graph is restricted to the ten datasets the authors shipped, so
            # any other name is rejected by argparse before main() is reached
            (
                "influence_maximization.py",
                "choices=['facebook_combined_un', 'email_di', 'soc-epinions_di', "
                "'gnutella_di', 'wiki-vote_di','CA-HepTh_un', 'lastfm_un',"
                "'power_grid_un', 'jazz_un', 'cora-ml_un'],",
                "",
            ),
            # setting2 discards --k and hardcodes 20% of the graph, which would
            # give the same seed-set size at every budget in our sweep. Reading
            # --k here preserves the published default exactly (it is 0.2).
            (
                "influence_maximization.py",
                'args["k"] = int(0.2 * G.number_of_nodes())',
                'args["k"] = int(args["k"] * G.number_of_nodes())',
            ),
            # ea_global_low_deg_mutation weights genes by 1 - degree/max(degree)
            # so it prefers mutating LOW-degree genes. When every gene in a
            # candidate has the same degree those weights are all zero and
            # random.choices raises. Common on NetScience, where most nodes have
            # degree 1-2. Uniform is the right fallback: if no gene is
            # lower-degree than another, all are equally preferred.
            (
                "src/ea/mutators.py",
                "\tprobs = np.array(probs) / max(probs)\n\tprobs = 1 - probs\n",
                "\tprobs = np.array(probs) / max(probs)\n\tprobs = 1 - probs\n"
                "\tif probs.sum() == 0:\n"
                "\t\tprobs = np.ones(len(probs))\n",
            ),
        ],
        export=_moeim_export,
        command=_moeim_command,
        parse_seeds=_moeim_parse,
        notes=(
            "Pure-Python evolutionary algorithm — no training, no GPU, and its "
            "--k is already a fraction of |V| like our --budget-pcts. The most "
            "directly runnable external baseline, and the current SOTA on "
            "Jazz/Cora-ML (beats DeepIM). Its `setting2` is exactly the "
            "DeepIM-comparison configuration."
        ),
    ),
    "touplegdd": ExternalBaseline(
        name="touplegdd",
        kind=learned,
        title="ToupleGDD — 3 coupled GNNs + Double DQN",
        venue="IEEE TCSS 2023",
        repo="https://github.com/Dtrycode/ToupleGDD",
        paper="https://arxiv.org/abs/2210.07500",
        entry="main.py (inference with the shipped tripling.ckpt)",
        status="needs_setup",
        export=_touplegdd_export,
        command=_touplegdd_command,
        parse_seeds=_touplegdd_parse,
        notes=(
            "Ships pretrained checkpoints (tripling.ckpt, s2vdqn.ckpt), so it "
            "runs inference-only — no training needed. Trained on <300-node "
            "random graphs and designed to generalize, which is exactly the "
            "regime our graphs sit in. Needs torch."
        ),
    ),
    "deepim": ExternalBaseline(
        name="deepim",
        kind=learned,
        title="DeepIM — Deep Graph Representation Learning for IM",
        venue="ICML 2023",
        repo="https://github.com/triplej0079/DeepIM",
        paper="https://arxiv.org/abs/2305.02200",
        entry="genim.py (after building .SG with baselines/deepim_data.py)",
        status="needs_setup",
        # networkx 3.0 removed from_scipy_sparse_matrix in favour of
        # from_scipy_sparse_array; the signature is unchanged. Two call sites,
        # both reached only at the final spread-evaluation step.
        patches=[
            # THE one that mattered: genim.py calls parse_args(args=[]), which
            # parses an EMPTY argv and silently discards -d/-dm/-sp. Every run
            # therefore used the defaults (cora_ml, LT, rate 1) and loaded
            # cora_ml_mean_LT10.SG — a 2810-node graph — no matter which graph
            # we handed it. A notebook leftover, where bare parse_args() would
            # choke on Jupyter's argv.
            (
                "genim.py",
                "args = parser.parse_args(args=[])",
                "args = parser.parse_args()",
            ),
            (
                "main/utils.py",
                "nx.from_scipy_sparse_matrix",
                "nx.from_scipy_sparse_array",
            ),
            # genim.py computes `seed` and then only ever prints the spread it
            # achieves, so the seed set — the one thing we need — never leaves
            # the process. Emit it in a form parse_seed_integers can read.
            # A UNIQUE marker, not "Seeds:" — that token also appears in DeepIM's
            # own output, and re.search would then read whichever list came
            # first. Print the graph size alongside so a size mismatch is
            # self-evident in the log instead of inferred from bad indices.
            (
                "genim.py",
                "influence = diffusion_evaluation(adj, seed",
                "print('GWM_SEEDS n={} {}'.format(adj.shape[0], "
                "[int(node) for node in seed]))\n"
                "influence = diffusion_evaluation(adj, seed",
            ),
            # Upgrade clones carrying the earlier, ambiguous marker
            (
                "genim.py",
                "print('Seeds: {}'.format([int(node) for node in seed]))",
                "print('GWM_SEEDS n={} {}'.format(adj.shape[0], "
                "[int(node) for node in seed]))",
            ),
            # seed_num was read off a STALE x_hat left over from the training
            # loop above, so the number of seeds returned had nothing to do with
            # the requested budget — it varied run to run and produced a
            # non-monotonic spread curve. seed_rate IS the budget percentage.
            (
                "genim.py",
                "seed_num = int(x_hat.sum().item())",
                "seed_num = max(1, round(args.seed_rate * 0.01 * adj.shape[0]))",
            ),
            # Upgrade path for clones patched with the earlier form: x_hat is a
            # leftover from the training loop, so its shape depends on how that
            # loop happened to end. adj is (N, N) and unambiguous.
            (
                "genim.py",
                "seed_num = max(1, round(args.seed_rate * 0.01 * x_hat.shape[-1]))",
                "seed_num = max(1, round(args.seed_rate * 0.01 * adj.shape[0]))",
            ),
        ],
        export=_deepim_export,
        command=_deepim_command,
        parse_seeds=_deepim_parse,
        notes=(
            "UNBLOCKED. The repo's data/ folder ships empty, but genim.py's "
            "input is not exotic: a pickled dict {'adj': scipy sparse, "
            "'inverse_pairs': tensor (n, N, 2)} where [...,0] is the seed "
            "vector and [...,1] the resulting influenced vector. That is "
            "exactly what our simulator produces, so "
            "`python -m baselines.deepim_data` regenerates the missing files "
            "from OUR graphs. Note genim.py then TRAINS a VAE + SpGAT per "
            "(dataset, diffusion model, budget) — it is not an inference-only "
            "baseline, so budget its runtime accordingly. Published tables are "
            "in IM_RESEARCH.md §4 and are directly comparable on Jazz and "
            "Power Grid."
        ),
    ),
    "gcomb": ExternalBaseline(
        name="gcomb",
        kind=learned,
        title="GCOMB — GraphSAGE pruning + Q-learning",
        venue="NeurIPS 2020",
        repo="https://github.com/idea-iitd/GCOMB",
        paper="https://proceedings.neurips.cc/paper/2020/hash/e7532dbeff7ef901f2e70daacb3f452d-Abstract.html",
        entry="IM/ (multi-stage shell pipeline)",
        status="blocked",
        blocker=(
            "Requires BOTH python2.7 and python3 environments "
            "(requirements_python2.7.txt + requirements_python3.txt) plus a "
            "multi-stage supervised-then-RL training pipeline driven by shell "
            "scripts. Python 2.7 is end-of-life and not available in this "
            "toolchain. Unblocking means containerising it."
        ),
        python="2.7+3.6",
        notes="Best scalability among learned methods; billion-edge claims.",
    ),
    "iminfector": ExternalBaseline(
        name="iminfector",
        kind=learned,
        title="IMINFECTOR — multi-task learning on diffusion cascades",
        venue="IEEE TKDE 2020",
        repo="https://github.com/geopanag/IMINFECTOR",
        paper="https://arxiv.org/abs/1904.08804",
        entry="main.py",
        status="blocked",
        blocker=(
            "Structurally inapplicable to our graphs. IMINFECTOR is model-free: "
            "it learns from OBSERVED diffusion cascades and never assumes IC/LT. "
            "Our BA/SBM/ER graphs are synthetic and Jazz/Power Grid/NetScience "
            "ship no cascade logs, so there is nothing for it to train on. It "
            "can only run on Digg/Weibo/MAG, which our pipeline cannot simulate "
            "at scale (see IM_RESEARCH.md §1.1)."
        ),
        notes=(
            "Included in the registry for completeness and to document exactly "
            "why it cannot be a baseline for us — the reason is scientific, not "
            "an engineering gap."
        ),
    ),
    "glie": ExternalBaseline(
        name="glie",
        kind=learned,
        title="GLIE / CELF-GLIE — GNN influence estimator replacing MC",
        venue="ASONAM 2023 (arXiv 2021)",
        repo="https://github.com/geopanag/learn_im",
        paper="https://arxiv.org/abs/2108.04623",
        entry="celf_glie.py (uses the stored GLIE model)",
        status="needs_setup",
        export=_glie_export,
        command=_glie_command,
        parse_seeds=lambda work_dir, stdout, budget: parse_seed_integers(
            stdout, budget
        ),
        notes=(
            "Found at geopanag/learn_im, NOT the geopanag/GLIE URL commonly "
            "cited (that 404s). Ships a stored model so celf_glie.py selects "
            "seeds without training. Closest published relative of our work: it "
            "replaces Monte Carlo with a learned GNN spread estimator, which is "
            "our world model's job — the difference is GLIE predicts a spread "
            "SCALAR while ours predicts the next STATE and is action-conditioned."
        ),
    ),
    "lense": ExternalBaseline(
        name="lense",
        kind=learned,
        title="LeNSE — learn to prune to a solvable subgraph",
        venue="ICML 2022",
        repo="https://github.com/davidireland-iso/LeNSE",
        paper="https://proceedings.mlr.press/v162/ireland22a.html",
        entry="IM/ (train encoder -> prune -> run a heuristic on the subgraph)",
        status="needs_setup",
        notes=(
            "Repo moved: davidireland3/LeNSE -> davidireland-iso/LeNSE. IM/ is "
            "the influence-maximization variant. NOT WIRED, and the reason is "
            "concrete: IM/ is a FOUR-STAGE sequential training pipeline "
            "(embedding_training.py -> guided_exploration_training.py -> "
            "dqn_training.py -> dqn_test.py), there is NO README in the repo, "
            "and no pretrained checkpoints are shipped. Every stage's arguments "
            "would have to be reverse-engineered from source. Writing an adapter "
            "without cloning and running it would be guesswork. TO UNBLOCK: "
            "clone it, read IM/dqn_test.py's args, run the four stages once by "
            "hand, then the adapter is mechanical."
        ),
    ),
    "him": ExternalBaseline(
        name="him",
        kind=learned,
        title="HIM — influence strength estimation in hyperbolic space",
        venue="2025",
        repo="https://github.com/PlaymakerQ/HIM",
        paper="https://arxiv.org/abs/2502.13571",
        entry="run.py",
        status="needs_setup",
        notes=(
            "Found at PlaymakerQ/HIM. Data format IS resolved: DataLoad reads "
            "`data/<name>/<name>.G` (a pickled networkx graph) plus a pickled "
            "cascade file, and run.py computes "
            "seed_num = int(num_node * seed_ratio * 0.01), i.e. seed_ratio is a "
            "percentage exactly like our --budget-pcts. We can write both "
            "pickles. THREE gaps remain: (1) run.py hardcodes "
            "`data_name = datasets[1]` with no CLI flag, so it must be patched "
            "to accept ours; (2) MT.load_model_params(f'HIM_{dm}', data_name) "
            "needs a per-dataset config entry whose SCHEMA is inside "
            "configs.zip, which cannot be read without unpacking the repo; "
            "(3) it trains 200 epochs with geoopt, so budget GPU time. Gap (2) "
            "is the hard blocker — everything else is mechanical."
        ),
    ),
    "opim": ExternalBaseline(
        name="opim",
        kind=classical,
        title="OPIM-C — online processing RIS",
        venue="SIGMOD 2018",
        repo="https://github.com/tangj90/OPIM",
        paper="https://dl.acm.org/doi/10.1145/3183713.3183749",
        entry="C++ binary (make)",
        status="needs_setup",
        requirements=None,
        build=["make"],
        # The repo root holds OPIM1.0 and OPIM1.1 side by side with no top-level
        # Makefile; 1.1 is the current release. Its Makefile writes a binary
        # literally named OPIM1.1.o, which is why _opim_binary globs "*.o".
        subdir="OPIM1.1",
        export=partial(_opim_export, "opim"),
        command=partial(_opim_command, "opim"),
        parse_seeds=partial(_opim_parse, "opim"),
        notes=(
            "C++ reference implementation of a near-optimal RIS method with "
            "anytime guarantees. Appears in DeepIM's comparison table. Our "
            "library has no OPIM — this is the only source for it."
        ),
    ),
    "ssa": ExternalBaseline(
        name="ssa",
        kind=classical,
        title="SSA / D-SSA — stop-and-stare RIS",
        venue="SIGMOD 2016",
        repo="https://github.com/hungnt55/Stop-and-Stare",
        paper="https://arxiv.org/abs/1605.07990",
        entry="C++ binary (make)",
        status="needs_setup",
        requirements=None,
        build=["make"],
        # The repo tracks only release zips, no source. The 2.1 zip expands to a
        # directory named SSA_release_2.0 (the authors' own inconsistency).
        unpack="SSA_release_2.1.zip",
        subdir="SSA_release_2.0/SSA",
        export=_ssa_export,
        command=_ssa_command,
        parse_seeds=_ssa_parse,
        notes=(
            "C++ RIS baseline with the tightest published sample bounds. Our "
            "library's `ssa` is a simplified Python doubling-RIS stand-in; this "
            "is the authors' original."
        ),
    ),
    "subsim": ExternalBaseline(
        name="subsim",
        kind=classical,
        title="SubSIM — sublinear-time RIS",
        venue="SIGMOD 2020",
        repo="https://github.com/qtguo/subsim",
        paper="https://dl.acm.org/doi/10.1145/3318464.3389740",
        entry="C++ binary (make)",
        status="needs_setup",
        requirements=None,
        build=["make"],
        export=_subsim_export,
        command=_subsim_command,
        parse_seeds=partial(_opim_parse, "subsim"),
        notes=(
            "Third of the three RIS methods in DeepIM's table (IMM / OPIM / "
            "SubSIM). Not present in our Python library at all, so this repo is "
            "the only way to run it."
        ),
    ),
    "imm": ExternalBaseline(
        name="imm",
        kind=classical,
        title="IMM — martingale RIS (the standard classical reference)",
        venue="SIGMOD 2015",
        repo="https://sourceforge.net/projects/im-imm/",
        paper="https://dl.acm.org/doi/10.1145/2723372.2723734",
        entry="C++ binary (make)",
        status="needs_setup",
        requirements=None,
        fetch="manual",
        build=["make"],
        notes=(
            "THE classical reference point every IM paper compares against. Our "
            "library's `imm` is a simplified Python reimplementation — good "
            "enough as a condition-1 arm, but this is the authors' original and "
            "the one whose numbers appear in published tables. Tang et al. "
            "release a tarball on SourceForge, not a git repo, so setup cannot "
            "clone it. A third-party git reimplementation exists at "
            "https://github.com/gdelpuente/IMM if a clone is preferred over the "
            "original (not author-endorsed)."
        ),
    ),
    "tim": ExternalBaseline(
        name="tim",
        kind=classical,
        title="TIM / TIM+ — near-optimal-time RIS",
        venue="SIGMOD 2014",
        repo="https://sourceforge.net/projects/timplus/",
        paper="https://arxiv.org/abs/1404.0900",
        entry="C++ binary (make)",
        status="needs_setup",
        requirements=None,
        fetch="manual",
        build=["make"],
        notes=(
            "IMM's predecessor and the paper that made RIS practical. Our "
            "library has a simplified Python `tim`. SourceForge tarball, no git."
        ),
    ),
    "piano": ExternalBaseline(
        name="piano",
        kind=learned,
        title="PIANO — IM meets deep reinforcement learning",
        venue="IEEE TCSS 2022",
        repo="https://ieeexplore.ieee.org/document/9769766",
        paper="https://ieeexplore.ieee.org/document/9769766",
        entry="n/a",
        status="blocked",
        blocker=(
            "No public code release. The paper advertises 'pretrained PIANO "
            "models' but publishes no repository or download link, and no "
            "official implementation was found. ToupleGDD's authors state they "
            "had to REVISE the S2V-DQN code themselves to obtain a PIANO "
            "baseline, which is why PIANO numbers differ between papers. "
            "Unblocking means reimplementing it from the paper."
        ),
        notes=(
            "Appears in DeepIM's comparison table; those published numbers are "
            "transcribed in IM_RESEARCH.md §4."
        ),
    ),
    "oim": ExternalBaseline(
        name="oim",
        kind=classical,
        title="OIM — Online Influence Maximization",
        venue="KDD 2015",
        repo="https://dl.acm.org/doi/10.1145/2783258.2783271",
        paper="https://dl.acm.org/doi/10.1145/2783258.2783271",
        entry="n/a",
        status="blocked",
        blocker=(
            "No public code release found (Lei, Maniu, Mo, Cheng & Senellart, "
            "KDD 2015). Also a different problem setting: OIM assumes repeated "
            "campaigns with feedback that updates edge weights between rounds, "
            "whereas our task is single-shot seeding, so even with code the "
            "comparison would need care."
        ),
        notes=(
            "Appears in DeepIM's IC table (and is out-of-memory on Weibo there); "
            "numbers transcribed in IM_RESEARCH.md §4."
        ),
    ),
}


def available_baselines(kind: str | None = None) -> list[str]:
    """Registered, not blocked, and with an adapter that can drive it."""
    return sorted(
        name
        for name, spec in external_baselines.items()
        if spec.status != "blocked"
        and spec.wired
        and (kind is None or spec.kind == kind)
    )


def runnable_baselines() -> list[str]:
    """...and actually present on disk."""
    return sorted(name for name in available_baselines() if external_baselines[name].installed())


def unwired_baselines() -> list[str]:
    """Cloned-but-undriveable: registered, not blocked, no adapter."""
    return sorted(
        name
        for name, spec in external_baselines.items()
        if spec.status != "blocked" and not spec.wired
    )
