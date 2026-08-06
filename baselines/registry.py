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
import random
import struct
from functools import partial
import os
import pickle
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, replace
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

# AdaptiveIM: its own example uses epsilon 0.5, and one possible world per -time
adaptiveim_epsilon = 0.5
adaptiveim_realization_seed = 0

# RL4IM defaults, taken from its own src/tasks/config/basic_env.yaml rather than
# from its paper, which states an average degree its config does not produce
rl4im_graph_type = "powerlaw"
rl4im_m = 3
rl4im_p = 0.05
rl4im_propagate_p = 0.1

# Collective Influence's ball radius L. 2 is the value Morone & Makse use
# throughout and the one their scaling results are quoted at.
ci_radius = 2

# GND stops once the giant component falls below TARGET_SIZE. 1% of N is the
# upstream default; the floor of 1 is required, not cosmetic — at 0 the
# spectral recursion has no terminating condition and the binary spins.
gnd_target_fraction = 0.01
gnd_min_target = 1
# GND's REMOVE_STRATEGY. 1 is the weighted (cost-aware) rule that is the paper's
# actual contribution and its shipped default; 3 is the unweighted one.
#
# We run **3**, and the reason is measurable rather than a preference. Our budget
# is a CARDINALITY budget (k nodes), and pricing removal by degree optimizes a
# different objective that a node-count metric scores unfairly — research
# §8.2 trap 4. On `crime` (754/2127), driving the giant component to 7:
#
#     strategy 1 (weighted)   needs 290 nodes;  its first 75 leave a GCC of 269
#     strategy 3 (unweighted) needs 110 nodes;  its first 75 leave a GCC of  12
#
# Defaulting to 1 would have put the authors' own code at 269 against our
# reimplementation's 37 and made it look broken. Set it to 1 only alongside a
# cost budget, and say which one any reported number used.
gnd_strategy = 3

# Min-Sum's stage-2 tree-break target and stage-3 reinsertion threshold, as
# fractions of N. The paper's own example uses 100 and 200 at N=78125.
decycler_break_fraction = 0.0013
decycler_reinsert_fraction = 0.0026

# EI's `m`: candidates sampled per un-vaccination step. The paper uses ~10^3
# on million-node graphs and shows the result is insensitive above ~100.
ei_candidates = 1000

# FINDER's shipped ND checkpoint (trained on BA graphs of 30-50 nodes)
finder_checkpoint = "nrange_30_50_iter_78000.ckpt"
# Which member of the survey harness's suite to drive by default
review_default_method = "GDM"


@dataclass
class ExternalBaseline:
    name: str
    kind: str  # classical | learned
    title: str
    venue: str
    repo: str
    paper: str
    entry: str  # what setup/run drives, for the README table
    # Which graph task this baseline solves; a key of pipeline.tasks.tasks.
    # Every entry today is influence_maximization, which is exactly why the
    # field exists — a blocking or CND baseline must not join an IM sweep.
    task: str = "influence_maximization"
    # True when the repo selects seeds in BATCHES and returns them in selection
    # order. run_baseline then hands it the round schedule and the pipeline
    # replays the result per round instead of as one t=0 plan, which is the
    # difference between an adaptive arm and a static one wearing its name.
    rounds_aware: bool = False
    # True when the repo's intervention is an ARC rather than a node, so its
    # parser hands back a flat list of endpoints (two ids per unit of budget).
    # `run_external_baseline` needs to know, or the budget check counts endpoints
    # and rejects a legal k-arc answer as 2k seeds.
    returns_edges: bool = False
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
    # Extra packages to pip-install into the venv, for a library published on
    # PyPI rather than driven from its clone (GraphSL, cosasi). Installed after
    # `requirements`, so a repo can use both.
    pip_packages: tuple = ()
    # Share another entry's clone and venv. One PACKAGE can ship several
    # published METHODS (GraphSL is six), and each deserves its own arm — but
    # six clones of one repo means six multi-GB torch installs for no reason.
    # `installed()` and the setup path both follow this, so the first entry
    # installs and the rest report ready.
    install_name: str | None = None
    notes: str = ""
    # Some repos return non-zero on SUCCESS. EI calls exit() once it reaches
    # its percolation threshold, which is its normal termination; OPIM's
    # format step ends with `return 1`. The artifact is the real signal, so
    # these opt out of the exit-code check rather than the check being dropped.
    allow_nonzero_exit: bool = False
    extra_env: dict = field(default_factory=dict)

    @property
    def root(self) -> Path:
        """Where the repo is cloned, and where a per-baseline venv lives."""
        return external_root / (self.install_name or self.name)

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


# --- critical node detection adapters ---------------------------------------
#
# Two conventions the whole dismantling branch shares, and both bite:
#
#   * **1-indexed node ids.** Every one of these repos reads `id1 id2` and
#     indexes an array at `id - 1`. Our ids are 0-based, so the export shifts up
#     and the parser shifts back down. Getting this wrong does not crash — it
#     silently drops node 0 and off-by-ones every other id.
#   * **They return a removal ORDER, not a set of size k.** Each runs until its
#     own stopping rule (a giant component below some threshold), so the adapter
#     takes the length-k prefix. That is the right reading of a sequential
#     dismantler at a fixed budget, and it is also why a run can come back SHORT:
#     if the algorithm stopped before k, `_pad_removals` tops the set up by
#     degree rather than letting the arm silently under-spend its budget.


def _undirected_pairs(graph) -> list:
    """Deduplicated undirected edges as sorted (u, v) pairs, 0-indexed."""
    seen = set()

    for column in range(graph.edge_index.shape[1]):
        source = int(graph.edge_index[0, column])
        target = int(graph.edge_index[1, column])

        if source != target:
            seen.add((min(source, target), max(source, target)))

    return sorted(seen)


def _write_fallback(graph, work_dir: Path) -> None:
    """
    Stash the degree ranking the parser needs to top up a short removal set.

    `parse_seeds(work_dir, stdout, budget)` does not receive the graph, so the
    only channel from export to parse is the work directory itself.
    """
    degrees = [0] * graph.num_nodes
    for column in range(graph.edge_index.shape[1]):
        degrees[int(graph.edge_index[0, column])] += 1

    order = sorted(range(graph.num_nodes), key=lambda node: -degrees[node])
    (work_dir / "fallback.txt").write_text(
        f"{graph.num_nodes}\n" + " ".join(str(node) for node in order)
    )


def _pad_removals(order: list, work_dir: Path, budget: int) -> list:
    """
    The length-k prefix of a removal order, topped up by degree if it ran short.

    A dismantler that hit its own stopping rule before k returns fewer nodes than
    the budget — CI stops once the giant component is broken, GND once it falls
    under its threshold. Returning that short would make the arm look better per
    node than it is, so the remainder is filled the same way our in-repo library
    fills it, from the ranking `_write_fallback` left behind.
    """
    text = (work_dir / "fallback.txt").read_text().split("\n")
    num_nodes = int(text[0])
    by_degree = [int(node) for node in text[1].split()] if len(text) > 1 else []

    chosen, picked = [], set()

    for node in list(order) + by_degree:
        if len(chosen) >= budget:
            break

        if 0 <= int(node) < num_nodes and int(node) not in picked:
            picked.add(int(node))
            chosen.append(int(node))

    return chosen[:budget]


def _ci_output_name() -> str:
    """CI's `sprintf(fname,"INFLUENCERS_%d_lvl_%d.txt", ntwk, L)`; ntwk is 0 for one graph."""
    return f"INFLUENCERS_0_lvl_{ci_radius}.txt"


def _ci_export(graph, work_dir: Path, budget: int, diffusion_model: str) -> dict:
    """
    Collective Influence reads an ADJACENCY LIST, not an edge list: one line per
    node, `id nbr nbr ...`, 1-indexed. A node with no neighbours still needs its
    line or the ids shift.
    """
    neighbours = [set() for _ in range(graph.num_nodes)]
    for source, target in _undirected_pairs(graph):
        neighbours[source].add(target)
        neighbours[target].add(source)

    network = work_dir / "network.txt"
    with open(network, "w") as handle:
        for node in range(graph.num_nodes):
            listed = " ".join(str(other + 1) for other in sorted(neighbours[node]))
            handle.write(f"{node + 1} {listed}".rstrip() + "\n")

    _write_fallback(graph, work_dir)

    # CI builds its output name with sprintf and opens it relative to the CWD,
    # which run_external_baseline sets to the repo directory. A leftover from an
    # earlier run would then be parsed as this run's result, so it is cleared
    # here and moved into work_dir by the parser.
    stale = external_baselines["collective_influence"].directory / _ci_output_name()
    stale.unlink(missing_ok=True)

    # ABSOLUTE input path for the same cwd reason. CI does not check its fopen,
    # so a miss arrives as a segfault rather than an error message.
    return {"network": str(network.resolve()), "radius": ci_radius}


def _ci_command(
    work_dir: Path, budget: int, diffusion_model: str, extras: dict, graph
) -> list[str]:
    binary = external_baselines["collective_influence"].directory / "CI"

    return [str(binary.resolve()), extras["network"], str(extras["radius"])]


def _ci_parse(work_dir: Path, stdout: str, budget: int) -> list[int]:
    """
    CI writes `INFLUENCERS_<n>_lvl_<L>.txt`, NOT the `Influencers.txt` its own
    source comment advertises — verified by running it. Columns are
    `rank node_id degree component`, tab-separated, already in removal order,
    with `#` comment lines and blanks in the header.
    """
    # Written next to the binary, not in work_dir — see _ci_export. Moved here so
    # the run directory is self-documenting and the next run cannot read this one.
    produced = external_baselines["collective_influence"].directory / _ci_output_name()

    if not produced.exists():
        raise FileNotFoundError(
            f"collective_influence produced no {_ci_output_name()} in "
            f"{produced.parent}; it writes relative to its own directory"
        )

    output = work_dir / produced.name
    produced.replace(output)

    order = []
    for line in output.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) >= 2:
            # column 2 is the node id, 1-indexed
            order.append(int(parts[1]) - 1)

    return _pad_removals(order, work_dir, budget)


def _gnd_export(graph, work_dir: Path, budget: int, diffusion_model: str) -> dict:
    """
    GND reads a plain `id1 id2` edge list, 1-indexed, one undirected edge per line.

    Warning: Upstream hardcodes the node count, the filenames and the stopping threshold
    as compile-time constants (GND.cpp lines 37-41), which would mean recompiling
    once per graph. `setup_baselines` applies a patch making main() read them from
    argv; the patch is verified to reproduce the shipped CrimeNet result byte for
    byte when no arguments are given.
    """
    network = work_dir / "network.txt"
    with open(network, "w") as handle:
        for source, target in _undirected_pairs(graph):
            handle.write(f"{source + 1} {target + 1}\n")

    _write_fallback(graph, work_dir)

    # GND stops when the giant component falls below TARGET_SIZE. Its own default
    # is 1% of N; below 1 the spectral recursion cannot terminate and the binary
    # spins, so the floor is not optional.
    # ABSOLUTE, for the same reason as CI: cwd is the repo directory
    return {
        "network": str(network.resolve()),
        "removed": str((work_dir / "removed.txt").resolve()),
        "plot": str((work_dir / "plot.txt").resolve()),
        "target": max(gnd_min_target, int(gnd_target_fraction * graph.num_nodes)),
    }


def _gnd_command(
    work_dir: Path, budget: int, diffusion_model: str, extras: dict, graph
) -> list[str]:
    binary = external_baselines["gnd"].directory / "GND"

    return [
        str(binary.resolve()),
        str(graph.num_nodes),
        extras["network"],
        extras["removed"],
        extras["plot"],
        str(extras["target"]),
        # 1 = the weighted (cost-aware) strategy, which is the paper's contribution
        # and its default. 3 is the unweighted variant.
        str(gnd_strategy),
    ]


def _gnd_parse(work_dir: Path, stdout: str, budget: int) -> list[int]:
    """`removed.txt` is one 1-indexed node id per line, already in removal order."""
    output = work_dir / "removed.txt"

    if not output.exists():
        raise FileNotFoundError(f"gnd produced no removed.txt in {work_dir}\n{stdout}")

    order = [
        int(line) - 1
        for line in output.read_text().split()
        if line.strip().lstrip("-").isdigit()
    ]

    return _pad_removals(order, work_dir, budget)


def _rank_set_by_degree(members, work_dir: Path, budget: int) -> list[int]:
    """
    Choose k from an UNORDERED removal set, by degree.

    Min-Sum and EI both answer "the minimal set that dismantles this graph", not
    "the best k nodes". Min-Sum's `reverse-greedy` writes its final set sorted by
    NODE ID, and EI writes a 0/1 vector — neither carries a removal order, so a
    length-k prefix of either is an arbitrary subset (measured: taking Min-Sum's
    first 75 by id on `crime` left a giant component of 418, worse than random).

    Ranking the set's members by degree is OUR tie-break, not theirs. Report a
    number from these two as "the highest-degree k of <method>'s dismantling
    set", not as the method's own answer at k — and prefer reporting them at
    their natural budget, which is |set|.
    """
    text = (work_dir / "fallback.txt").read_text().split("\n")
    by_degree = [int(node) for node in text[1].split()] if len(text) > 1 else []
    chosen = {int(node) for node in members}

    return _pad_removals([n for n in by_degree if n in chosen], work_dir, budget)


def _decycler_export(graph, work_dir: Path, budget: int, diffusion_model: str) -> dict:
    """
    Min-Sum's three stages are three PROCESSES joined by pipes, so the adapter
    writes a driver script and the command runs that.

    Its own README documents the chain exactly:

        ./decycler -o < graph.txt > seeds.txt
        cat graph.txt seeds.txt | python treebreaker.py <maxcomp> > broken.txt
        cat graph.txt seeds.txt broken.txt | ./reverse-greedy -t <thr> > out.txt

    Graph format is one `D u v` line per undirected edge, 0-indexed. Every stage
    emits `S <node>` lines, and the final removal set is the union of the
    decycling seeds and the tree-breaking picks, minus whatever reverse-greedy
    put back — which is what the last stage's output already is.
    """
    spec = external_baselines["decycler"]

    edges = work_dir / "graph.txt"
    with open(edges, "w") as handle:
        for source, target in _undirected_pairs(graph):
            handle.write(f"D {source} {target}\n")

    _write_fallback(graph, work_dir)

    # Component-size targets. Min-Sum breaks trees to <= maxcomp and then
    # reinserts while components stay <= threshold; both scale with N the way the
    # paper's own examples do (100 / 200 at N=78125).
    maxcomp = max(2, int(decycler_break_fraction * graph.num_nodes))
    threshold = max(maxcomp, int(decycler_reinsert_fraction * graph.num_nodes))

    script = work_dir / "run.sh"
    script.write_text(
        "#!/bin/bash\nset -eo pipefail\n"
        f'cd "{spec.directory.resolve()}"\n'
        f'G="{edges.resolve()}"\n'
        f'W="{work_dir.resolve()}"\n'
        './decycler -o < "$G" > "$W/seeds.txt"\n'
        f'cat "$G" "$W/seeds.txt" | "{sys.executable}" treebreaker.py {maxcomp} '
        '> "$W/broken.txt" || true\n'
        'cat "$G" "$W/seeds.txt" "$W/broken.txt" | '
        f'./reverse-greedy -t {threshold} > "$W/output.txt"\n'
    )
    script.chmod(0o755)

    return {"script": str(script.resolve())}


def _decycler_command(
    work_dir: Path, budget: int, diffusion_model: str, extras: dict, graph
) -> list[str]:
    return ["/bin/bash", extras["script"]]


def _decycler_parse(work_dir: Path, stdout: str, budget: int) -> list[int]:
    """
    `S <node>` lines, 0-indexed. `output.txt` is the post-reinsertion set (the
    paper's stage 3); if reverse-greedy produced nothing usable we fall back to
    the raw decycling seeds, which is still a real Min-Sum answer, just stage 1.
    """
    order = []

    for name in ("output.txt", "broken.txt", "seeds.txt"):
        path = work_dir / name
        if not path.exists():
            continue

        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "S" and parts[1].lstrip("-").isdigit():
                order.append(int(parts[1]))

        if order:
            break

    if not order:
        raise FileNotFoundError(
            f"decycler produced no `S <node>` lines in {work_dir} "
            f"(checked output.txt, broken.txt, seeds.txt)"
        )

    # reverse-greedy writes its set sorted by node id, not removal order
    return _rank_set_by_degree(order, work_dir, budget)


def _ei_export(graph, work_dir: Path, budget: int, diffusion_model: str) -> dict:
    """
    EI reads `N` on the first line then one `i j` edge per line, 0-indexed, and
    writes its outputs into the CWD.
    """
    network = work_dir / "network.txt"
    with open(network, "w") as handle:
        handle.write(f"{graph.num_nodes}\n")
        for source, target in _undirected_pairs(graph):
            handle.write(f"{source} {target}\n")

    _write_fallback(graph, work_dir)

    for stale in ("threshold_conditions.dat", "output_sigma1.dat", "output_sigma2.dat"):
        (external_baselines["explosive_immunization"].directory / stale).unlink(
            missing_ok=True
        )

    # m = candidates sampled per un-vaccination step. The paper uses ~10^3 on
    # million-node graphs and shows the result is insensitive above ~100.
    return {
        "network": str(network.resolve()),
        "candidates": str(min(ei_candidates, max(10, graph.num_nodes // 4))),
    }


def _ei_command(
    work_dir: Path, budget: int, diffusion_model: str, extras: dict, graph
) -> list[str]:
    binary = external_baselines["explosive_immunization"].directory / "exploimmun"

    return [str(binary.resolve()), extras["candidates"], extras["network"]]


def _ei_parse(work_dir: Path, stdout: str, budget: int) -> list[int]:
    """
    Warning: EI answers a different question than a fixed budget asks.

    `threshold_conditions.dat` is `<node> <flag>` for every node: the vaccination
    state at EI's approximate percolation threshold (1/sqrt(N)), with no ordering
    and no way to ask for k.

    Warning: **The flag is inverted relative to the README**, which states "A 1 means
    vaccinated, and a 0 means unvaccinated". It is the other way round, and the
    difference is not subtle — on `crime` the file has 707 ones and 47 zeros, and
    47 (6% of N) is the plausible immunization set while 707 is not. Scoring
    confirms it: taking the top 75 by degree from the ones leaves a giant
    component of **626**, from the zeros **18**. We read ZEROS as vaccinated.

    Choosing k from that set is then OUR degree tie-break, not EI's, because the
    algorithm publishes no per-node score — see `_rank_set_by_degree`.
    """
    produced = (
        external_baselines["explosive_immunization"].directory
        / "threshold_conditions.dat"
    )
    if not produced.exists():
        raise FileNotFoundError(
            f"explosive_immunization produced no threshold_conditions.dat in "
            f"{produced.parent}; it writes relative to its own directory"
        )

    output = work_dir / produced.name
    produced.replace(output)

    vaccinated = []
    for line in output.read_text().splitlines():
        parts = line.split()
        # "0" is vaccinated despite the README saying otherwise; see the docstring
        if len(parts) >= 2 and parts[0].lstrip("-").isdigit() and parts[1] == "0":
            vaccinated.append(int(parts[0]))

    return _rank_set_by_degree(vaccinated, work_dir, budget)


# --- learned CND repos ------------------------------------------------------
#
# Warning: EVERY ADAPTER BELOW IS UNTESTED. They were written from the cloned source —
# entry points, argument names and output paths were read, not guessed — but none
# of these repos' dependency stacks install on the machine this was built on
# (TensorFlow 1.14 for FINDER, torch-geometric 1.7.2 for NIRM, CUDA-pinned conda
# environments for the rest). Each notes field records exactly what remains to
# confirm. Treat a number from one of these as unverified until it has run once.
#
# All of them share the same shape: write our graph in the repo's format, run a
# SMALL GENERATED RUNNER that imports the repo's own modules, and read back one
# node id per line. The runner exists so the removal ORDER is what crosses the
# boundary — several of these repos only persist a curve or a CSV by default,
# and a curve cannot be turned back into a set.


def _write_runner(work_dir: Path, body: str) -> Path:
    """Drop a generated runner script beside the run and return its path."""
    runner = work_dir / "gwm_runner.py"
    runner.write_text(body)

    return runner


def _edgelist_export(graph, work_dir: Path, one_indexed: bool = False) -> Path:
    """`u v` per undirected edge — what every repo here reads."""
    shift = 1 if one_indexed else 0
    network = work_dir / "network.txt"

    with open(network, "w") as handle:
        for source, target in _undirected_pairs(graph):
            handle.write(f"{source + shift} {target + shift}\n")

    return network


def _order_parse(work_dir: Path, stdout: str, budget: int) -> list[int]:
    """
    Read `removal_order.txt` — one 0-indexed node id per line, best first.

    Every generated runner writes this one file, so the learned repos share a
    parser and differ only in how their runner produces it.
    """
    output = work_dir / "removal_order.txt"

    if not output.exists():
        raise FileNotFoundError(
            f"no removal_order.txt in {work_dir}; the runner did not reach its "
            f"write step. Tail of stdout:\n{stdout[-800:]}"
        )

    order = [
        int(line) for line in output.read_text().split() if line.lstrip("-").isdigit()
    ]

    return _pad_removals(order, work_dir, budget)


def _mind_export(graph, work_dir: Path, budget: int, diffusion_model: str) -> dict:
    """
    MIND loads a PICKLED IGRAPH per file from a directory (`utils/common.py::load_g`
    calls `g.simplify(...)`), and its `test.py` only writes an aggregate CSV — so
    the runner drives the policy directly and records the order.
    """
    graphs = work_dir / "graphs"
    os.makedirs(graphs, exist_ok=True)

    pickle_path = graphs / "gwm.pkl"
    with open(pickle_path, "wb") as handle:
        import igraph

        pickle.dump(
            igraph.Graph(n=graph.num_nodes, edges=_undirected_pairs(graph)), handle
        )

    runner = _write_runner(
        work_dir,
        "import os, sys, torch\n"
        f"sys.path.insert(0, {str(external_baselines['mind'].directory.resolve())!r})\n"
        "from utils.common import load_g\n"
        "from env import DismantleEnv\n"
        "from networks.dismantle import SACPolicy\n"
        f"g = load_g({str(pickle_path.resolve())!r}, 'gwm')\n"
        "policy = SACPolicy(num_features=16, num_heads=4, num_mps=6)\n"
        "state = torch.load("
        f"{str((external_baselines['mind'].directory / 'saved' / 'mind.ckpt').resolve())!r},"
        " map_location='cpu', weights_only=True)['policy_state_dict']\n"
        "policy.load_state_dict(state); policy.eval()\n"
        "env = DismantleEnv(graph_data=[g], batch_size=1, is_val=True)\n"
        "order, obs = [], env.reset()\n"
        "with torch.no_grad():\n"
        "    for _ in range(g.vcount()):\n"
        "        action = policy.act(obs) if hasattr(policy, 'act') else policy(obs)\n"
        "        node = int(action if isinstance(action, int) else int(action[0]))\n"
        "        order.append(node)\n"
        "        obs, _, done, _ = env.step(action)\n"
        "        if done:\n"
        "            break\n"
        f"open({str((work_dir / 'removal_order.txt').resolve())!r}, 'w')"
        ".write('\\n'.join(str(n) for n in order))\n",
    )

    _write_fallback(graph, work_dir)

    return {"runner": str(runner.resolve())}


def _runner_command(
    work_dir: Path, budget: int, diffusion_model: str, extras: dict, graph
) -> list[str]:
    return ["python", extras["runner"]]


def _finder_export(graph, work_dir: Path, budget: int, diffusion_model: str) -> dict:
    """
    FINDER's `testReal.py` calls `dqn.EvaluateRealData(model_file, data_test,
    save_dir, stepRatio)` on a plain edge list and ships the trained
    `FINDER_ND/models/nrange_30_50_iter_78000.ckpt`.
    """
    network = _edgelist_export(graph, work_dir)
    directory = external_baselines["finder"].directory
    model = directory / "code" / "FINDER_ND" / "models" / finder_checkpoint

    runner = _write_runner(
        work_dir,
        "import sys, os\n"
        f"sys.path.insert(0, {str((directory / 'code' / 'FINDER_ND').resolve())!r})\n"
        "from FINDER import FINDER\n"
        "dqn = FINDER()\n"
        f"dqn.LoadModel({str(model.resolve())!r})\n"
        f"solution, _ = dqn.EvaluateRealData({str(model.resolve())!r},"
        f" {str(network.resolve())!r}, {str(work_dir.resolve())!r}, 0.01)\n"
        f"open({str((work_dir / 'removal_order.txt').resolve())!r}, 'w')"
        ".write('\\n'.join(str(int(n)) for n in solution))\n",
    )

    _write_fallback(graph, work_dir)

    return {"runner": str(runner.resolve())}


def _nirm_export(graph, work_dir: Path, budget: int, diffusion_model: str) -> dict:
    """
    NIRM's `Test.py` builds a `Dismatnle(model, ...)` and calls
    `onepass_dismantle()`; it ships `checkpoints/NIRM_onepass.pkl`.
    """
    network = _edgelist_export(graph, work_dir)
    directory = external_baselines["nirm"].directory

    runner = _write_runner(
        work_dir,
        "import sys, torch, networkx as nx\n"
        f"sys.path.insert(0, {str(directory.resolve())!r})\n"
        "from Test import Dismatnle, LoadModel\n"
        "from Model import *\n"
        f"model = LoadModel({str((directory / 'checkpoints' / 'NIRM_onepass.pkl').resolve())!r}, None)\n"
        f"runner = Dismatnle(model, ['network'], {str(work_dir.resolve())!r},"
        f" {str(work_dir.resolve())!r})\n"
        "order = runner.onepass_dismantle()\n"
        f"open({str((work_dir / 'removal_order.txt').resolve())!r}, 'w')"
        ".write('\\n'.join(str(int(n)) for n in (order or [])))\n",
    )

    _write_fallback(graph, work_dir)

    return {"runner": str(runner.resolve()), "network": str(network.resolve())}


def _gdm_export(graph, work_dir: Path, budget: int, diffusion_model: str) -> dict:
    """
    GDM scores every node in ONE pass (it is the static member of this family) and
    the removal order is that score, descending.

    Warning: The clone ships no trained weights, so this needs `--weights` pointing at a
    model produced by the repo's own training script before it can run at all.
    """
    network = _edgelist_export(graph, work_dir)
    directory = external_baselines["gdm"].directory

    runner = _write_runner(
        work_dir,
        "import sys, networkx as nx, numpy as np\n"
        f"sys.path.insert(0, {str(directory.resolve())!r})\n"
        "from network_dismantling.GDM.dataset_providers import prepare_graph\n"
        "from network_dismantling.GDM.models import GAT_Model\n"
        f"g = nx.read_edgelist({str(network.resolve())!r}, nodetype=int)\n"
        "data = prepare_graph(g)\n"
        "model = GAT_Model.load_from_checkpoint()\n"
        "scores = model(data).detach().cpu().numpy().ravel()\n"
        "order = list(np.argsort(-scores))\n"
        f"open({str((work_dir / 'removal_order.txt').resolve())!r}, 'w')"
        ".write('\\n'.join(str(int(n)) for n in order))\n",
    )

    _write_fallback(graph, work_dir)

    return {"runner": str(runner.resolve())}


def _review_export(graph, work_dir: Path, budget: int, diffusion_model: str) -> dict:
    """
    The Artime et al. survey harness, which already wires CI, CoreHD, GND, EI,
    MinSum, FINDER and GDM behind one interface — the single highest-value thing
    to get running, because it is seven baselines for one adapter.

    `--method` selects which; it is threaded through from `extras` so one entry
    can serve the whole suite once the harness's own CLI is confirmed.
    """
    network = _edgelist_export(graph, work_dir)
    _write_fallback(graph, work_dir)

    return {
        "network": str(network.resolve()),
        "output": str((work_dir / "removal_order.txt").resolve()),
        "method": review_default_method,
    }


def _review_command(
    work_dir: Path, budget: int, diffusion_model: str, extras: dict, graph
) -> list[str]:
    return [
        "python",
        "-m",
        "network_dismantling.machine_learning.pytorch.dismantler",
        "--input",
        extras["network"],
        "--output",
        extras["output"],
        "--method",
        extras["method"],
    ]


def _dcrs_export(graph, work_dir: Path, budget: int, diffusion_model: str) -> dict:
    """
    DCRS scores nodes by diffusion competence + role significance, one pass.

    Warning: The clone ships NO trained weights — only `train_synthetic.py` /
    `train_realworld.py` — so this cannot run until a model has been trained by
    the repo's own scripts. The runner points at `checkpoints/dcrs.pt` by
    convention; confirm the real filename after training.
    """
    network = _edgelist_export(graph, work_dir)
    directory = external_baselines["dcrs"].directory

    runner = _write_runner(
        work_dir,
        "import sys, torch, networkx as nx, numpy as np\n"
        f"sys.path.insert(0, {str(directory.resolve())!r})\n"
        "from model import *\n"
        "from utils import *\n"
        f"g = nx.read_edgelist({str(network.resolve())!r}, nodetype=int)\n"
        f"model = torch.load({str((directory / 'checkpoints' / 'dcrs.pt').resolve())!r},"
        " map_location='cpu')\n"
        "model.eval()\n"
        "scores = model(g).detach().cpu().numpy().ravel()\n"
        "order = list(np.argsort(-scores))\n"
        f"open({str((work_dir / 'removal_order.txt').resolve())!r}, 'w')"
        ".write('\\n'.join(str(int(n)) for n in order))\n",
    )

    _write_fallback(graph, work_dir)

    return {"runner": str(runner.resolve())}


def _selinda_export(graph, work_dir: Path, budget: int, diffusion_model: str) -> dict:
    """
    Selinda learns an RL attack policy and symbolic-regresses it into a closed-form
    resilience law; its dismantling arm rides on a bundled GDM-slim model.

    Warning: The repo is driven by shell scripts (`app_rl.sh`, `app_sr.sh`, ...) rather
    than a library entry point, and its published output is a fitted LAW, not a
    dismantling set. This runner targets the bundled GDM-slim scorer, which is the
    only part with a per-node output; confirm the module path before trusting it.
    """
    network = _edgelist_export(graph, work_dir)
    directory = external_baselines["selinda"].directory

    runner = _write_runner(
        work_dir,
        "import sys, networkx as nx, numpy as np\n"
        f"sys.path.insert(0, {str((directory / 'thirdparty' / 'GDM-slim').resolve())!r})\n"
        "from network_dismantling.GDM.models import GAT_Model\n"
        f"g = nx.read_edgelist({str(network.resolve())!r}, nodetype=int)\n"
        "model = GAT_Model.load()\n"
        "scores = model(g).detach().cpu().numpy().ravel()\n"
        "order = list(np.argsort(-scores))\n"
        f"open({str((work_dir / 'removal_order.txt').resolve())!r}, 'w')"
        ".write('\\n'.join(str(int(n)) for n in order))\n",
    )

    _write_fallback(graph, work_dir)

    return {"runner": str(runner.resolve())}


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


# Adaptive IM: AdaptiveIM (Han et al.) and RL4IM ------------------------------
#
# Both select seeds in BATCHES, so both are rounds_aware and both read
# round_schedule.json out of work_dir (written by run_baseline).


def _read_batches(work_dir: Path) -> list[int]:
    from baselines.run_baseline import schedule_filename

    return json.loads((work_dir / schedule_filename).read_text())


def _adaptiveim_export(graph, work_dir: Path, budget: int, diffusion_model: str) -> dict:
    """
    Write the three inputs AdaptiveIM reads, in the formats its source defines.

    The dataset argument MUST be exactly `<dir>/<name>/`: load_possible_world()
    recovers the dataset name by slicing between the FIRST TWO slashes of that
    string, so a deeper path makes it look for `<dir>/<name>/<wrong>_0` and
    die in mmap. That is why this writes into the repo rather than into
    work_dir, which is an absolute path several levels deep.
    """
    spec = external_baselines["adaptiveim"]
    name = "gwm"
    data_dir = spec.directory / "gwmdata" / name
    os.makedirs(data_dir, exist_ok=True)

    edges = {
        (int(graph.edge_index[0, edge]), int(graph.edge_index[1, edge])): float(
            graph.ic_probs[edge]
        )
        for edge in range(graph.edge_index.shape[1])
    }

    (data_dir / "attribute.txt").write_text(
        f"n={graph.num_nodes}\nm={len(edges)}\n"
    )

    # graph.h::readGraph mmaps m records of (int src, int dst, double p) and
    # indexes them by dst. 16 bytes each, native endianness.
    binary = "graph_lt.inf" if diffusion_model == "LT" else "graph_ic.inf"
    with open(data_dir / binary, "wb") as handle:
        for (source, destination), probability in edges.items():
            handle.write(struct.pack("iid", source, destination, probability))

    # The "realization" files the README defers to an unlinked Tools repo. They
    # are IC possible worlds: each edge present independently with its own
    # probability. Format read verbatim from load_possible_world(): TEXT, one
    # "src dst" per line, PO[src].push_back(dst), with an "_lt" suffix under LT.
    # Generated from OUR probabilities, so the algorithm adapts against the same
    # dynamics our referee will score it under.
    rng = random.Random(adaptiveim_realization_seed)
    suffix = "_lt" if diffusion_model == "LT" else ""
    live = [
        f"{source} {destination}"
        for (source, destination), probability in edges.items()
        if rng.random() < probability
    ]
    (data_dir / f"{name}_0{suffix}").write_text("\n".join(live) + "\n")

    return {"dataset": f"gwmdata/{name}", "batches": _read_batches(work_dir)}


def _adaptiveim_command(
    work_dir: Path, budget: int, diffusion_model: str, extras: dict, graph
) -> list[str]:
    batches = extras["batches"]
    # -batch is a single size, so a ragged schedule (k not divisible by r) is
    # run at its largest batch and sliced back to the real sizes afterwards
    return [
        "./exp_epic",
        "-dataset",
        extras["dataset"],
        "-model",
        "LT" if diffusion_model == "LT" else "IC",
        "-epsilon",
        str(adaptiveim_epsilon),
        "-k",
        str(sum(batches)),
        "-batch",
        str(max(batches)),
        "-seedfile",
        str((work_dir / "seeds.txt").resolve()),
        "-time",
        "1",
    ]


def _adaptiveim_parse(work_dir: Path, stdout: str, budget: int) -> list[int]:
    """Seeds in SELECTION order, space separated, so the batch structure survives."""
    seed_file = work_dir / "seeds.txt"
    if not seed_file.exists():
        raise ValueError(
            f"adaptiveim wrote no seedfile at {seed_file}; output tail:\n"
            f"{stdout[-500:]}"
        )

    return [int(token) for token in seed_file.read_text().split()][:budget]


def _rl4im_export(graph, work_dir: Path, budget: int, diffusion_model: str) -> dict:
    """
    RL4IM generates its own graphs, so the only thing exported is the config.

    It has no flag for "run on this graph": src/environment/graph.py builds
    powerlaw_cluster graphs from node_train / graph_nbr_train. The honest
    reading is that RL4IM is trained on a DISTRIBUTION and applied zero-shot,
    which is the regime its own paper reports, so the adapter matches our graph
    on size and family and lets it train there. That is stated on the arm rather
    than hidden: it is not the same as running it on our exact graph.
    """
    batches = _read_batches(work_dir)

    return {
        "batches": batches,
        # T is the number of node-selection steps; budget is per main step
        "T": str(sum(batches)),
        "budget": str(max(batches)),
        "nodes": str(graph.num_nodes),
    }


def _rl4im_command(
    work_dir: Path, budget: int, diffusion_model: str, extras: dict, graph
) -> list[str]:
    """
    sacred, not argparse: overrides go through `with key=value`.

    No pretrained checkpoint ships, so this TRAINS before it selects. Its own
    defaults cap training at max_global_t=2000 steps, which is minutes rather
    than hours (research/adaptive_online_im.md §11).
    """
    return [
        "python",
        "main.py",
        "with",
        f"T={extras['T']}",
        f"budget={extras['budget']}",
        f"node_train={extras['nodes']}",
        f"node_test={extras['nodes']}",
        f"graph_type={rl4im_graph_type}",
        f"m={rl4im_m}",
        f"p={rl4im_p}",
        f"propagate_p={rl4im_propagate_p}",
        "mode=train",
        "use_cuda=False",
        f"results_path={work_dir.resolve()}",
    ]


def _rl4im_parse(work_dir: Path, stdout: str, budget: int) -> list[int]:
    """
    Read the invited set out of the test log.

    runners.py prints `invited:  [...]` and `present:  [...]` under verbose. The
    INVITED set is the seed set: `present` is what survived the willingness
    draw (q=0.6), which is RL4IM's contingency model and not something our
    action space has. Scoring `present` would hand it a budget it did not spend.
    """
    match = re.findall(r"invited:\s*\[([^\]]*)\]", stdout)
    if not match:
        raise ValueError(
            f"rl4im printed no `invited:` line; it is emitted only when "
            f"verbose=True, so the command needs that override. Output tail:\n"
            f"{stdout[-800:]}"
        )

    return [int(token) for token in re.findall(r"\d+", match[-1])][:budget]


# Source localization --------------------------------------------------------
#
# The inverse contract, and it differs from every adapter above in shape: the
# repo is handed a GRAPH plus a BATCH of observed diffusion states and returns
# one source set per observation, so `export` takes the instance list where the
# intervention adapters take a budget, and `parse_seeds` returns a dict keyed by
# episode id rather than a flat list. `run_baseline.run_external_baseline`
# dispatches on whether `instances` was passed.

graphsl_methods = ("lpsi", "netsleuth", "ojc", "gcnsi", "ivgd", "slvae")
# Training epochs for GraphSL's three learned methods. Low by the papers'
# standards and deliberately so: this is a default that has to finish on a
# laptop, and it is the one number to raise before quoting a GCNSI/IVGD/SL-VAE
# comparison as anything but a smoke result.
graphsl_epochs = 50


def training_rows(instances: list):
    """
    Which rows a SUPERVISED external baseline may fit on: the selection pool only.

    Both pools cross the boundary concatenated (`select + evaluate`), because the
    arm makes two passes and a repo invoked once has to cover both. The rule is
    "same split as the first row", and it is a split comparison rather than a
    first-occurrence one for a reason worth stating: the two pools come from
    DISJOINT splits, so no episode id ever repeats, and a first-occurrence rule
    marks every row trainable — which silently fits a learned method on the
    evaluation cascades it is then scored against.

    That is the one thing `run_baseline`'s module docstring says must never
    happen: a label may reach a repo's `train`, never its evaluation-split
    prediction.
    """
    import numpy as np

    if not instances:
        return np.zeros(0, dtype=bool)

    first = getattr(instances[0], "split", "")

    return np.array(
        [getattr(instance, "split", "") == first for instance in instances],
        dtype=bool,
    )


def _graphsl_export(graph, work_dir: Path, instances: list, diffusion_model: str) -> dict:
    """
    Our episodes as GraphSL's own two arrays: a CSR adjacency and per-instance
    (seed, observation) columns.

    Both instance POOLS arrive together (selection and evaluation), flagged by
    `is_train`, because GraphSL's methods tune on labelled data and then predict.
    The driver trains on the selection rows only.
    """
    import numpy as np

    os.makedirs(work_dir, exist_ok=True)
    num_nodes = graph.num_nodes

    # Undirected and unweighted: GraphSL builds a normalized Laplacian and a
    # networkx view off this, and every method in it is defined on the symmetric
    # contact graph
    view = _undirected_view(graph)
    adjacency = nx.to_scipy_sparse_array(
        view, nodelist=list(range(num_nodes)), format="csr", dtype=float
    )
    np.savez(
        work_dir / "graph.npz",
        data=adjacency.data,
        indices=adjacency.indices,
        indptr=adjacency.indptr,
        shape=np.array(adjacency.shape),
    )

    seeds = np.zeros((len(instances), num_nodes), dtype=np.float32)
    observations = np.zeros((len(instances), num_nodes), dtype=np.float32)

    for row, instance in enumerate(instances):
        seeds[row, instance.sources] = 1.0
        # Binarized: GraphSL's methods index `influ_mat[:, -1]` as an indicator
        # and OJC compares it against 1 exactly, so a continuous marginal would
        # silently empty its infected set
        observations[row] = (np.asarray(instance.observation) >= 0.5).astype(np.float32)

    np.savez(
        work_dir / "instances.npz",
        seeds=seeds,
        observations=observations,
        episode_ids=np.array([instance.episode_id for instance in instances]),
        budgets=np.array([max(1, instance.source_count) for instance in instances]),
        is_train=training_rows(instances),
    )

    driver = baselines_root / "drivers" / "graphsl_driver.py"
    shutil.copy(driver, work_dir / "graphsl_driver.py")

    return {"driver": str(work_dir / "graphsl_driver.py")}


def _graphsl_command(
    work_dir: Path, budget: int, diffusion_model: str, extras: dict, graph
) -> list[str]:
    return ["python", extras["driver"], str(work_dir)]


def _graphsl_parse(work_dir: Path, stdout: str, instances: list) -> dict:
    payload = json.loads((work_dir / "predictions.json").read_text())

    return {episode: nodes for episode, nodes in payload["sources"].items()}


def _localization_export(
    graph, work_dir: Path, instances: list, diffusion_model: str, driver: str
) -> dict:
    """
    The shared inverse export: an edge list, and one row per instance.

    Everything except GraphSL reads a plain graph rather than a CSR triple, so
    this writes the edge list and lets each driver build whatever it needs. The
    per-instance arrays are the same for all of them: seed labels (used only by
    the drivers that train), binarized observations, budgets, horizons, the
    train/evaluate flag, and the trajectory for the ones that condition on
    intermediate snapshots.
    """
    import numpy as np

    os.makedirs(work_dir, exist_ok=True)
    num_nodes = graph.num_nodes
    view = _undirected_view(graph)

    np.savez(
        work_dir / "graph.npz",
        edges=np.array(list(view.edges()), dtype=np.int64).reshape(-1, 2),
        num_nodes=np.array(num_nodes),
    )

    seeds = np.zeros((len(instances), num_nodes), dtype=np.float32)
    observations = np.zeros((len(instances), num_nodes), dtype=np.float32)

    for row, instance in enumerate(instances):
        seeds[row, instance.sources] = 1.0
        # Binarized: every one of these repos treats the observation as an
        # indicator, and several compare it against 1 exactly
        observations[row] = (np.asarray(instance.observation) >= 0.5).astype(np.float32)

    # Longest recorded path, so every instance's trajectory can be padded to one
    # array. Padding holds the last step, which is exact for a monotone cascade.
    depth = max(
        (
            instance.trajectory.shape[0]
            for instance in instances
            if instance.trajectory is not None
        ),
        default=1,
    )
    trajectories = np.zeros((len(instances), depth, num_nodes), dtype=np.float32)
    for row, instance in enumerate(instances):
        path = (
            instance.trajectory
            if instance.trajectory is not None
            else observations[row][None, :]
        )
        trajectories[row, : path.shape[0]] = path
        trajectories[row, path.shape[0] :] = path[-1]

    np.savez(
        work_dir / "instances.npz",
        seeds=seeds,
        observations=observations,
        trajectories=trajectories,
        episode_ids=np.array([instance.episode_id for instance in instances]),
        budgets=np.array([max(1, instance.source_count) for instance in instances]),
        horizons=np.array([instance.horizon for instance in instances]),
        is_train=training_rows(instances),
    )

    shutil.copy(baselines_root / "drivers" / driver, work_dir / driver)

    return {"driver": str(work_dir / driver)}


def _localization_command(
    work_dir: Path, budget: int, diffusion_model: str, extras: dict, graph
) -> list[str]:
    return ["python", extras["driver"], str(work_dir)]


def _localization_parse(work_dir: Path, stdout: str, instances: list) -> dict:
    return json.loads((work_dir / "predictions.json").read_text())["sources"]


# Cascade reconstruction -----------------------------------------------------
#
# The inverse export one level up: a decoder is handed whole MASKED HISTORIES
# rather than endpoints, and hands back one time assignment per cascade. The tree
# is then built by our own shared `finalize` rule in `run_baseline`, deliberately:
# only DIPT outputs explicit who-infected-whom edges and it has no public code, so
# every runnable repo here produces per-step node STATES (DITTO's `y_pred`) and a
# Path Precision comparison between two of them would otherwise be a comparison of
# two different tree-building tricks rather than of two decoders.

# DITTO's own hyperparameters, read from `scripts/ditto-ba-si.sh` rather than from
# the paper. Its MCMC is `t_steps` Hastings rounds over `t_samples` parallel
# chains, and `q_steps` is the proposal network's training budget — which is the
# expensive half and the first thing to cut for a smoke run.
ditto_defaults = {
    "b_pI0": "1e-6",
    "b_pR0": "1e-6",
    "b_steps": "500",
    "b_lr": "0.003",
    "q_steps": "500",
    "q_lr": "0.001",
    "q_hid": "16",
    "q_gnn": "3",
    "q_mlp": "2",
    "q_samples": "10",
    "q_zlim": "16",
    "p_coef": "1.0",
    "t_samples": "100",
    "t_steps": "10",
    "t_keep": "0.5",
}


def _reconstruction_export(
    graph, work_dir: Path, instances: list, diffusion_model: str, driver: str,
    supervised: bool = False
) -> dict:
    """
    Our masked cascades as a graph file plus one row per instance.

    Everything a decoder needs and nothing it may not see. The SOURCE SET is
    deliberately absent: only the source COUNT crosses, because DITTO reads `I0`
    off `y[:, 0]` in its own pipeline and our own protocol hands every inverse
    method its `k` (research/source_localization.md §2.4.1).

    `reported` / `reported_times` are the mask, `final_state` the terminal
    snapshot, and `visible` the hidden-node mask. A repo that only consumes the
    snapshot (DITTO) ignores the first two; one that consumes reports ignores the
    third.

    `supervised` adds the two arrays a TRAINED imputer needs — GRIN, SPIN and
    Deep Demixing all fit a model on labelled histories before predicting, which
    is a setting none of our own arms have and which every row produced this way
    has to be labelled with. Two invariants make that safe rather than a leak, and
    both are ASSERTED here rather than trusted:

      * `is_train` is True for the SELECTION pool only, decided by `training_rows`
        on the instance's own split. Both pools cross together (the arm makes two
        passes and a repo invoked once has to cover both) and their episode ids are
        disjoint, which is exactly why the rule is a split comparison rather than a
        first-occurrence one.
      * `trajectories` is the true per-step history for training rows and EXACTLY
        ZERO for evaluation rows. A driver bug can then only ever read zeros for a
        row it must not see the answer to, instead of reading the answer.

    That is the same rule `run_baseline`'s module docstring states for the
    inverse batch: labels may reach a repo's `train`, never its evaluation-split
    prediction.
    """
    import numpy as np

    os.makedirs(work_dir, exist_ok=True)
    num_nodes = graph.num_nodes
    view = _undirected_view(graph)

    np.savez(
        work_dir / "graph.npz",
        edges=np.array(list(view.edges()), dtype=np.int64).reshape(-1, 2),
        num_nodes=np.array(num_nodes),
    )

    count = len(instances)
    reported = np.zeros((count, num_nodes), dtype=np.float32)
    # -1 = "reported but the time is withheld", -2 = "not reported at all"
    reported_times = np.full((count, num_nodes), -2, dtype=np.int64)
    final_state = np.zeros((count, num_nodes), dtype=np.float32)
    visible = np.ones((count, num_nodes), dtype=bool)

    for row, instance in enumerate(instances):
        for node, time in instance.observation.reported.items():
            reported[row, int(node)] = 1.0
            reported_times[row, int(node)] = -1 if time is None else int(time)

        final_state[row] = np.asarray(instance.final_state, dtype=np.float32)

        if instance.observation.visible is not None:
            visible[row] = np.asarray(instance.observation.visible, dtype=bool)

    payload = dict(
        reported=reported,
        reported_times=reported_times,
        final_state=final_state,
        visible=visible,
        episode_ids=np.array([instance.episode_id for instance in instances]),
        horizons=np.array([instance.horizon for instance in instances]),
        # The number of sources, never which ones: DITTO's own pipeline reads this
        # off y[:, 0] and every inverse arm here is given its k
        source_counts=np.array([len(instance.sources) for instance in instances]),
        settings=np.array(
            [instance.observation.setting for instance in instances]
        ),
    )

    if supervised:
        is_train = training_rows(instances)

        # Padded to one array; holding the last step is exact for a monotone
        # cascade, and a training row's own horizon is in `horizons` anyway
        depth = max(instance.horizon for instance in instances) + 1
        trajectories = np.zeros((count, depth, num_nodes), dtype=np.float32)

        for row, instance in enumerate(instances):
            if not is_train[row]:
                continue

            for node, time in instance.true_times.items():
                trajectories[row, min(int(time), depth - 1) :, int(node)] = 1.0

        if trajectories[~is_train].any():
            raise ValueError(
                "an EVALUATION row carries a ground-truth trajectory; that is the "
                "one thing this export must never do. A label may reach a repo's "
                "train() and never its evaluation-split prediction."
            )

        payload |= dict(trajectories=trajectories, is_train=is_train)
        # The three supervised drivers share their loading, training and decoding,
        # so the helper travels with them. It is copied NEXT TO the driver rather
        # than imported from our tree because the subprocess runs in the repo's own
        # venv and cannot see ours; Python puts the script's own directory on
        # `sys.path[0]`, so `import imputation_common` resolves from work_dir while
        # each driver's `sys.path.insert(0, os.getcwd())` reaches the repo itself.
        shutil.copy(
            baselines_root / "drivers" / "imputation_common.py",
            work_dir / "imputation_common.py",
        )

    np.savez(work_dir / "instances.npz", **payload)
    shutil.copy(baselines_root / "drivers" / driver, work_dir / driver)

    return {"driver": str(work_dir / driver)}


def _reconstruction_command(
    work_dir: Path, budget: int, diffusion_model: str, extras: dict, graph
) -> list[str]:
    return ["python", extras["driver"], str(work_dir)]


def _reconstruction_parse(work_dir: Path, stdout: str, instances: list) -> dict:
    """
    `{episode_id: {node: [time, parent or null]}}` from the driver's own artifact.

    A `null` parent at `t > 0` is not an error here: it means the repo produced a
    time assignment and no tree, which is what every runnable one in this file
    does. `run_baseline._collect_trajectories` then builds the tree with the same
    `finalize` rule the library decoders use, so a Path Precision difference
    between rows is a difference in TIMES rather than in tree construction.
    """
    payload = json.loads((Path(work_dir) / "predictions.json").read_text())

    return payload["trajectories"]


# Cascade prediction ---------------------------------------------------------
#
# The inverse export turned around: a predictor is handed the observed PREFIX of a
# real cascade and hands back one NUMBER per cascade. Everything here is a
# SUPERVISED regressor — that is what the whole §4.1 line is — so the split flag is
# the load-bearing field, exactly as it is for the three imputers on the
# reconstruction side: a label may reach a repo's `fit`, never its evaluation-row
# prediction.
#
# The exported shape is CasFlow's own canonical five-field line format
# (`research/cascade_prediction.md` §6.3), because §8.5's finding is that there is
# no benchmark for this task and that format IS the de-facto standard artefact —
# every repo in §4.1 either reads it or reads something one rename away from it.


def _prediction_export(
    graph, work_dir: Path, instances: list, diffusion_model: str, driver: str
) -> dict:
    """
    Our logged cascades as the canonical line format, one row per cascade.

    `paths` is the field every repo here consumes, in CasFlow's own shape: a list of
    `(node chain, elapsed)` pairs where the chain's last id is the adopter and the
    one before it is who they took it from. Only the OBSERVED prefix crosses — a
    predictor that could see past `t_o` would be reading the answer.

    `labels` is the INCREMENT (`P(t_p) - P(t_o)`), which is CasFlow's own label and
    what every repo here regresses; the drivers convert back to a total on the way
    out. `splits` carries our protocol so a repo cannot re-split at random, which is
    §8.3's whole warning.

    Two invariants are ASSERTED rather than trusted, the same pair
    `_reconstruction_export` asserts for its supervised drivers: no evaluation row
    carries a label, and no row's `paths` extends past the observation window.
    """
    import numpy as np

    os.makedirs(work_dir, exist_ok=True)
    num_nodes = graph.num_nodes
    view = _undirected_view(graph)

    np.savez(
        work_dir / "graph.npz",
        edges=np.array(list(view.edges()), dtype=np.int64).reshape(-1, 2),
        num_nodes=np.array(num_nodes),
    )

    cascade_ids, paths, labels, splits, publish_times, observed = [], [], [], [], [], []

    for instance in instances:
        observation = instance.observation
        window = observation.observed_steps

        # `(chain, elapsed)`, sorted by elapsed then id so the file order is
        # deterministic. The chain is `[root, adopter]` for everything but the root
        # itself, because our replay stores the parent per event and a corpus that
        # logs no parent already attributes to the root.
        rows = sorted(
            (
                ([int(node)] if node == observation.root else [int(observation.root), int(node)]),
                int(step),
            )
            for node, step in observation.adopters.items()
        )

        if any(step > window for _, step in rows):
            raise ValueError(
                "a cascade's exported prefix extends past its observation window; "
                "that would hand the repo part of the answer"
            )

        cascade_ids.append(str(instance.cascade_id))
        paths.append(json.dumps(rows))
        splits.append(str(instance.split))
        publish_times.append(int(observation.publish_time))
        observed.append(int(observation.popularity))
        labels.append(int(instance.increment))

    is_train = training_rows(instances)

    if np.array(labels)[~is_train].size and np.any(np.array(labels)[~is_train] < 0):
        raise ValueError("an evaluation row carries a negative increment label")

    np.savez(
        work_dir / "instances.npz",
        cascade_ids=np.array(cascade_ids),
        paths=np.array(paths),
        labels=np.array(labels, dtype=np.int64),
        splits=np.array(splits),
        publish_times=np.array(publish_times, dtype=np.int64),
        observed=np.array(observed, dtype=np.int64),
        is_train=is_train,
        observation=np.array(
            instances[0].observation.observed_steps if instances else 0
        ),
        horizon=np.array(instances[0].observation.horizon if instances else 0),
    )

    shutil.copy(baselines_root / "drivers" / driver, work_dir / driver)

    return {"driver": str(work_dir / driver)}


def _prediction_command(
    work_dir: Path, budget: int, diffusion_model: str, extras: dict, graph
) -> list[str]:
    return ["python", extras["driver"], str(work_dir)]


def _prediction_parse(work_dir: Path, stdout: str, instances: list) -> dict:
    """`{cascade_id: predicted total popularity}` from the driver's own artifact."""
    payload = json.loads((Path(work_dir) / "predictions.json").read_text())

    return payload["popularities"]


def _casflow_entry(
    name: str,
    title: str,
    venue: str,
    repo: str,
    paper: str,
    entry: str,
    source_subdir: str,
    patches: list | None,
    notes: str,
) -> ExternalBaseline:
    """
    CasFlow or CCGL: same author, same line format, same three-stage pipeline.

    One driver covers both and `GWM_CASFLOW_SRC` picks the clone's source directory,
    which is the repo root for CasFlow and `src/` for CCGL. Both pin an exact
    TensorFlow that no longer resolves on a current Python; the pin is patched away
    rather than honoured, because the code uses plain Keras 2 APIs that every TF 2.x
    provides and installing a 2021 wheel is not possible on most current platforms.
    """
    return ExternalBaseline(
        name=name,
        kind=learned,
        title=title,
        venue=venue,
        repo=repo,
        paper=paper,
        entry=entry,
        task="cascade_prediction",
        status="needs_setup",
        requirements="requirements.txt",
        patches=patches,
        export=partial(_prediction_export, driver="casflow_driver.py"),
        command=_prediction_command,
        parse_seeds=_prediction_parse,
        extra_env={"GWM_CASFLOW_SRC": source_subdir},
        notes=notes,
    )


# Influence blocking ---------------------------------------------------------
#
# S_N is not derivable from (graph, budget), so it crosses the process boundary the
# same way the round schedule does: written into work_dir as JSON by
# run_baseline.run_external_baseline rather than widened into every adapter's
# signature. A blocking adapter reads it; nothing else looks.
negative_seeds_filename = "negative_seeds.json"

# SandIMIN's approximation parameters, from its own Readme's example invocation
sandimin_epsilon = 0.2
sandimin_gamma = 0.1
sandimin_beta = 0.1

# Xie's stdin protocol: influence model 0 = IC, 1 = LT; propagation model 0 = TR
# (trivalency {0.1, 0.01, 0.001}), 1 = WC (1/in-degree). We run WC because it is what
# our own --prob-model weighted generates, so the repo recomputes the SAME
# probabilities we simulate under rather than a different edge model.
joc_weighted_cascade = 1

# DiffIM's own defaults, read from its constants.ipynb
diffim_latent_dim = [128, 128, 128, 128, 128, 128]
diffim_train_samples = 200
diffim_algorithm = "DiffIM+"


def read_negative_seeds(work_dir: Path) -> list[int]:
    """S_N for this run, written beside the export by the pipeline."""
    path = Path(work_dir) / negative_seeds_filename

    if not path.exists():
        raise FileNotFoundError(
            f"no {negative_seeds_filename} in {work_dir}: this baseline solves "
            f"influence blocking and needs the rumour's own seed set, which "
            f"run_external_baseline writes when the task is competitive"
        )

    return [int(node) for node in json.loads(path.read_text())]


def _sandimin_export(
    graph, work_dir: Path, budget: int, diffusion_model: str
) -> dict:
    """
    SandIMIN's three input files, written directly rather than through its `el2bin`.

    `graph_ic.inf` is a packed binary of `(int u, int v, double p)` per arc — read
    from `graph.h::readGraph`, which mmaps the file and steps by
    `2 * sizeof(int) + sizeof(double)` and asserts every id is `< n`. Writing it here
    skips the shipped `el2bin` binary entirely, which matters: that is a prebuilt
    x86 executable with no source in the repo, so on any other architecture it is a
    file that cannot run.
    """
    negative_seeds = read_negative_seeds(work_dir)
    pairs = [
        (int(graph.edge_index[0, edge]), int(graph.edge_index[1, edge]))
        for edge in range(graph.edge_index.shape[1])
    ]

    (work_dir / "attribute.txt").write_text(
        f"n={graph.num_nodes}\nm={len(pairs)}\n"
    )

    with open(work_dir / "graph_ic.inf", "wb") as handle:
        for edge, (source, target) in enumerate(pairs):
            handle.write(
                struct.pack("iid", source, target, float(graph.ic_probs[edge]))
            )

    (work_dir / f"rumorSet_{len(negative_seeds)}.txt").write_text(
        "\n".join(str(node) for node in negative_seeds) + "\n"
    )

    return {"rumor_num": len(negative_seeds)}


def _sandimin_command(
    work_dir: Path, budget: int, diffusion_model: str, extras: dict, graph
) -> list[str]:
    directory = external_baselines["sandimin"].directory
    # `arg.res` opens `results/res_...` relative to cwd and fails silently if the
    # directory is absent; the blocker set we actually read is written into work_dir
    os.makedirs(directory / "results", exist_ok=True)

    return [
        str((directory / "IMIN").resolve()),
        "-dataset", str(Path(work_dir).resolve()),
        "-k", str(budget),
        "-rumorNum", str(extras["rumor_num"]),
        "-algo", os.environ.get("SANDIMIN_ALGO", "SandIMIN"),
        "-epsilon", str(sandimin_epsilon),
        "-gamma", str(sandimin_gamma),
        "-beta", str(sandimin_beta),
    ]


def _sandimin_parse(work_dir: Path, stdout: str, budget: int) -> list[int]:
    output = Path(work_dir) / "gwm_blockers.txt"

    if not output.exists():
        raise FileNotFoundError(
            f"no gwm_blockers.txt in {work_dir}; the patched OutputSeedSetToFile did "
            f"not run. Tail of stdout:\n{stdout[-800:]}"
        )

    return [int(line) for line in output.read_text().split() if line.lstrip("-").isdigit()]


def _joc_export(graph, work_dir: Path, budget: int, diffusion_model: str) -> dict:
    """
    Xie's graph file plus the stdin script its `main()` reads.

    Format from its own README and verified against `data/sample_graph.txt`: first
    line `n m`, then one `u v` per DIRECTED arc, 0-indexed. The file has to live under
    the repo's `data/` because `main()` hardcodes `"../data/" + fileName`, and the
    result lands under `../results/` for the same reason.
    """
    negative_seeds = read_negative_seeds(work_dir)
    directory = external_baselines["imin_joc"].directory
    name = f"gwm_k{budget}.txt"

    os.makedirs(directory.parent / "data", exist_ok=True)
    os.makedirs(directory.parent / "results", exist_ok=True)

    pairs = [
        f"{int(graph.edge_index[0, edge])} {int(graph.edge_index[1, edge])}"
        for edge in range(graph.edge_index.shape[1])
    ]
    (directory.parent / "data" / name).write_text(
        f"{graph.num_nodes} {len(pairs)}\n" + "\n".join(pairs) + "\n"
    )

    influence_model = 1 if diffusion_model == "LT" else 0
    # dataset, influence model, propagation model, |S_N|, budget, then S_N itself —
    # the last of which only exists because of the patch that replaces the repo's
    # own random source draw with a read from stdin
    stdin = "\n".join(
        [
            name,
            str(influence_model),
            str(joc_weighted_cascade),
            str(len(negative_seeds)),
            str(budget),
        ]
        + [str(node) for node in negative_seeds]
    ) + "\n"
    (work_dir / "stdin.txt").write_text(stdin)

    prefix = "AG" if os.environ.get("JOC_ALGO", "AdvancedGreedy") == "AdvancedGreedy" else "GR"
    dynamics = "LT" if influence_model else "IC"

    return {
        "graph_name": name,
        "stdin": str((work_dir / "stdin.txt").resolve()),
        "blockers": str(
            (directory.parent / "results" / f"{prefix}-{dynamics}-WC-{name}.blockers").resolve()
        ),
    }


def _joc_command(
    work_dir: Path, budget: int, diffusion_model: str, extras: dict, graph
) -> list[str]:
    directory = external_baselines["imin_joc"].directory
    binary = directory / os.environ.get("JOC_ALGO", "AdvancedGreedy")
    runner = work_dir / "run.sh"
    # It reads its whole configuration from stdin, so the command is a two-line
    # shell wrapper rather than an argv — cheaper than a pty and fully deterministic
    runner.write_text(f'#!/bin/sh\nexec "{binary.resolve()}" < "{extras["stdin"]}"\n')
    runner.chmod(0o755)
    (work_dir / "blockers_path.txt").write_text(extras["blockers"])

    return ["/bin/sh", str(runner.resolve())]


def _joc_parse(work_dir: Path, stdout: str, budget: int) -> list[int]:
    path = Path((Path(work_dir) / "blockers_path.txt").read_text().strip())

    if not path.exists():
        raise FileNotFoundError(
            f"no {path}; the patched blocker writer did not run. Tail of "
            f"stdout:\n{stdout[-800:]}"
        )

    return [int(line) for line in path.read_text().split() if line.lstrip("-").isdigit()]


def _diffim_export(graph, work_dir: Path, budget: int, diffusion_model: str) -> dict:
    """
    DiffIM's graph and instance files, plus a runner that drives its notebooks directly.

    The repo ships ONLY notebooks — no `.py` anywhere — and its own cross-imports go
    through `import_ipynb`, which needs the algorithms directory to behave like a
    package it is not. The runner therefore execs the code cells of each notebook
    into ONE shared namespace in dependency order, which is both simpler and more
    deterministic than fighting the import machinery, and it runs THEIR code
    unmodified.

    Formats verified from `utils.ipynb::txt2adj` (`n m`, then `u v p` per arc) and
    `algorithm_pipeline.ipynb` (a gzipped pickle of `(is_seed, prob)` pairs, and a
    dataset name whose part before the first `-` is the graph file).
    """
    negative_seeds = read_negative_seeds(work_dir)
    directory = external_baselines["diffim"].directory
    os.makedirs(directory / "graphs", exist_ok=True)
    os.makedirs(directory / "datasets", exist_ok=True)

    name = f"gwm_k{budget}"
    lines = [
        f"{int(graph.edge_index[0, edge])} {int(graph.edge_index[1, edge])} "
        f"{float(graph.ic_probs[edge])}"
        for edge in range(graph.edge_index.shape[1])
    ]
    (directory / "graphs" / f"{name}.txt").write_text(
        f"{graph.num_nodes} {len(lines)}\n" + "\n".join(lines) + "\n"
    )

    # ABSOLUTE, and passed in rather than built inside the runner: the runner
    # chdir's into the repo before it does anything, so a relative path there lands
    # in the clone instead of beside the run. Silent, because the algorithm still
    # succeeds — the mask just is not where the parser looks.
    mask = (work_dir / "gwm_mask.txt").resolve()
    runner = _write_runner(
        work_dir,
        _diffim_runner_source(
            directory,
            name,
            negative_seeds,
            budget,
            graph.num_nodes,
            diffusion_model,
            mask,
        ),
    )

    return {"runner": str(runner.resolve()), "mask": str(mask)}


def _diffim_runner_source(
    directory: Path,
    name: str,
    negative_seeds: list,
    budget: int,
    num_nodes: int,
    diffusion_model: str,
    mask: Path,
) -> str:
    """The generated runner: exec their notebooks, train if needed, then select arcs."""
    algorithm = os.environ.get("DIFFIM_ALG", diffim_algorithm)
    model_name = os.environ.get("DIFFIM_MODEL", "")
    samples = int(os.environ.get("DIFFIM_TRAIN_SAMPLES", diffim_train_samples))

    return f"""\
import json, os, sys
os.chdir({str(directory.resolve())!r})
sys.path.insert(0, {str(directory.resolve())!r})

import numpy as np

# Their notebooks cross-import through `import_ipynb`, which needs `algorithms/` to
# be a package it is not. Exec the code cells into ONE namespace instead, in
# dependency order, skipping only the import lines that namespace already satisfies.
SKIP = ("import import_ipynb", "from constants import", "from utils import",
        "from simulation import", "from gnn import", "from dataset import",
        "from algorithms.", "get_ipython")
namespace = {{"__name__": "diffim_runner"}}


def run_notebook(path):
    cells = json.load(open(path))["cells"]
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        body = "".join(cell["source"])
        body = "\\n".join(
            line for line in body.splitlines()
            if not any(line.strip().startswith(prefix) for prefix in SKIP)
        )
        exec(compile(body, path, "exec"), namespace)


for notebook in ("constants.ipynb", "gnn.ipynb", "simulation.ipynb", "utils.ipynb",
                 "dataset.ipynb", "train.ipynb",
                 "algorithms/centrality.ipynb", "algorithms/greedy.ipynb",
                 "algorithms/BPM.ipynb", "algorithms/KED.ipynb",
                 "algorithms/MDS.ipynb", "algorithms/RIS.ipynb",
                 "algorithms/random.ipynb", "algorithms/DiffIM.ipynb"):
    run_notebook(notebook)

graph_name = {name!r}
n, m, adj_list = namespace["txt2adj"](graph_name)
seed_idx = np.array({list(negative_seeds)!r})
prob = namespace["simul"]({diffusion_model!r} if {diffusion_model!r} in ("IC", "LT") else "IC",
                          adj_list, seed_idx)

model_name = {model_name!r}
algorithm = {algorithm!r}

# DiffIM's own methods need a trained surrogate. The shipped checkpoints were fit on
# THEIR graphs, so unless one is named explicitly we train on ours with their own
# generate_dataset + train, which is the faithful thing to run.
if algorithm.startswith("DiffIM") and not model_name:
    namespace["generate_dataset"](graph_name + ".txt", {samples}, saving_tag="-train")
    namespace["generate_dataset"](graph_name + ".txt", max(10, {samples} // 10), saving_tag="-test")
    model_name = graph_name + ".pt"
    namespace["train"](graph_name + "-train.pkl.gz", graph_name + "-test.pkl.gz",
                       saving_name=model_name,
                       hyper_params={{"gnn_latent_dim": {diffim_latent_dim!r}}},
                       gpu_num="cpu")

kwargs = dict(model_name=model_name, gnn_latent_dim={diffim_latent_dim!r}, gpu_num="cpu")
if algorithm in ("DiffIM+",):
    kwargs.update(namespace["default_hyper_params"])

selector = {{"DiffIM": namespace["DiffIM"], "DiffIM+": namespace["DiffIMp"],
             "DiffIM++": namespace["DiffIMpp"], "BPM": namespace["BPM"],
             "RIS": namespace["RIS"], "MDS": namespace["MDS"],
             "KED": namespace["KED"], "greedy": namespace["greedy_orig"]}}[algorithm]

if algorithm in ("BPM", "KED"):
    mask, _ = selector(adj_list, seed_idx, {budget})
else:
    mask, _ = selector(adj_list, seed_idx, prob, {budget}, **kwargs)

with open({str(mask)!r}, "w") as handle:
    for entry in mask:
        handle.write(f"{{int(entry[0])}} {{int(entry[1])}}\\n")
"""


def _diffim_parse(work_dir: Path, stdout: str, budget: int) -> list[int]:
    """
    DiffIM returns ARCS, and the pipeline's seed path only understands node ids.

    Both endpoints are returned as a flat list so `run_external_baseline`'s range
    check still applies; `_external_script` re-pairs them for the edge lever. Kept
    here rather than widening the shared parser signature, which seven wired IM
    adapters would otherwise have to grow a field for.
    """
    path = Path(work_dir) / "gwm_mask.txt"

    if not path.exists():
        raise FileNotFoundError(
            f"no gwm_mask.txt in {work_dir}; the runner did not reach its write "
            f"step. Tail of stdout:\n{stdout[-1500:]}"
        )

    arcs = [
        (int(parts[0]), int(parts[1]))
        for parts in (line.split() for line in path.read_text().splitlines())
        if len(parts) >= 2
    ]

    return [node for arc in arcs[:budget] for node in arc]


def _ditto_entry(
    name: str, entry: str, function: str, title: str, venue: str, notes: str
) -> ExternalBaseline:
    """
    One arm per entry point in DITTO's clone, all three sharing its single install.

    DITTO ships three drivable methods and two of them are the paper's own MLE
    baselines: `dhrec.py` is its implementation of DHREC-PCDSVC for SI and SIR
    (the original code covers only SEIRS) and `cri.py` is its implementation of
    CRI, whose authors published none. Both also exist in our own pool, so these
    two arms are a CROSS-CHECK on our reimplementations rather than new coverage —
    and a valuable one, because §5.1's Tables 4-5 report DHREC and CRI only as
    DITTO reports them.
    """
    return ExternalBaseline(
        name=name,
        kind=learned if entry == "ditto.py" else classical,
        title=title,
        venue=venue,
        repo="https://github.com/q-rz/KDD23-DITTO",
        paper="https://arxiv.org/abs/2306.00488",
        entry=entry,
        task="cascade_reconstruction",
        status="needs_setup",
        # The repo pins CUDA 11.4 / torch 1.7 and ships no requirements file. The
        # deps are listed unpinned here for the same reason cosasi's are: resolving
        # 2021 wheels on a current Python fails outright. Warning: torch-scatter has
        # no universal wheel and builds from source against the installed torch,
        # which is the slow half of this install and the one that fails first on a
        # machine with no compiler.
        requirements=None,
        pip_packages=(
            "torch",
            "torch-geometric",
            "torch-scatter",
            "ndlib",
            "class-resolver",
            "networkx",
            "numpy",
            "pandas",
            "scikit-learn",
            "matplotlib",
            "seaborn",
            "tqdm",
        ),
        install_name="ditto",
        export=partial(_reconstruction_export, driver="ditto_driver.py"),
        command=_reconstruction_command,
        parse_seeds=_reconstruction_parse,
        extra_env={"GWM_DITTO_ENTRY": entry, "GWM_DITTO_FN": function},
        notes=notes,
    )


def _forecasting_entry(
    name: str, title: str, venue: str, repo: str, paper: str
) -> ExternalBaseline:
    """
    One of §4.3's forecasting GNNs, registered BLOCKED for a reason they all share.

    All three have live, maintained public code — which is exactly why they are
    registered rather than omitted: a reader who sees "Cola-GNN, code available" in
    §4.3 will reasonably ask why it is not a baseline, and the answer has to be on
    the row rather than in someone's memory.

    They predict CASE COUNTS per region on a metapopulation and never intervene.
    §4.3 says so outright — "these predict case counts; none of them intervene" —
    so there is no k-node set, no arc set and no allocation of any kind to score.
    They matter to this file only because they define the benchmark suite the
    2024-2026 GNN-for-epidemics surveys use, and because EpiLearn packages them.
    """
    return ExternalBaseline(
        name=name,
        kind=learned,
        title=title,
        venue=venue,
        repo=repo,
        paper=paper,
        entry="forecasting model (see repo)",
        task="epidemic_control",
        status="blocked",
        blocker=(
            "IT FORECASTS, IT DOES NOT INTERVENE. The code is live [verified, "
            "2026-08-05], but the model maps a history of regional case counts to "
            "future case counts — there is no action space, no budget and no node "
            "set to read back, so there is nothing for our referee to score. "
            "research/epidemic_control.md §4.3 lists this whole family under "
            "'not control, but the source of the datasets'. TO UNBLOCK: nothing; "
            "this is a category difference, not a missing adapter."
        ),
        notes=(
            "Relevant to this repo as a FORWARD model rather than a baseline: it is "
            "a learned transition on a metapopulation, which is the same object "
            "`data/wm_epidemic.py` is on a contact graph. A future comparison of "
            "one-step accuracy, not of allocations."
        ),
    )


def _netimm_export(
    graph, work_dir: Path, budget: int, diffusion_model: str
) -> dict:
    """
    Our graph as a weighted arc list plus the outbreak the solvers condition on.

    Warning: THE OUTBREAK CROSSES THE BOUNDARY, and it is not derivable from
    `(graph, budget)`. Two of the five solvers are DATA-AWARE — `Dom` is DAVA and
    `NetShape` is Khalil's hazard-matrix program, and both are defined as "given the
    OBSERVED infected set, choose k" — so a run without it answers a different
    question. `run_external_baseline` writes `negative_seeds.json` for a blocking
    task through the same channel; this reads the epidemic outbreak from the same
    file, because the harness's `--outbreak-*` machinery produces both.

    Written as a plain `u v w` arc list rather than the repo's own pickled NetworkX
    graph: a pickle written by our networkx and read by theirs is a version gamble
    that buys nothing, and the driver rebuilds the DiGraph in three lines.

    BOTH ORIENTATIONS of an undirected edge are written, because `DomSolver` needs a
    DiGraph (`immediate_dominators` is undefined on anything else) and a one-way
    export would give it a reachability the graph does not have.
    """
    probabilities = {}
    for column in range(graph.edge_index.shape[1]):
        source = int(graph.edge_index[0, column])
        target = int(graph.edge_index[1, column])

        if source != target:
            probabilities[(source, target)] = float(graph.ic_probs[column])

    network = work_dir / "graph.txt"
    with open(network, "w") as handle:
        for (source, target), probability in sorted(probabilities.items()):
            handle.write(f"{source} {target} {probability:.8f}\n")

    outbreak_path = work_dir / negative_seeds_filename
    outbreak = (
        json.loads(outbreak_path.read_text()) if outbreak_path.exists() else []
    )

    if not outbreak:
        # `Solver.__init__` raises outright on an empty seed set, and the two
        # data-aware solvers have nothing to condition on without one. Naming the
        # flag is more useful than the repo's own "Seeds can not be empty".
        raise ValueError(
            "no outbreak was written for this run, but every Network-Immunization "
            "solver takes the observed infected set as an input (Dom/DAVA and "
            "NetShape are DEFINED on it). Run with --outbreak-pct > 0."
        )

    _write_fallback(graph, work_dir)
    (work_dir / "outbreak.json").write_text(
        json.dumps(
            {
                "num_nodes": int(graph.num_nodes),
                "outbreak": [int(node) for node in outbreak],
                # `Solver.__init__` raises when k > |V| - |seeds|, which our budget
                # sweep can reach on a tiny graph
                "budget": int(
                    max(1, min(budget, graph.num_nodes - len(outbreak) - 1))
                ),
            }
        )
    )

    driver = "network_immunization_driver.py"
    shutil.copy(baselines_root / "drivers" / driver, work_dir / driver)

    return {"driver": str(work_dir / driver)}


def _netimm_command(
    work_dir: Path, budget: int, diffusion_model: str, extras: dict, graph
) -> list[str]:
    return ["python", extras["driver"], str(work_dir)]


def _netimm_parse(work_dir: Path, stdout: str, budget: int) -> list[int]:
    """`blocked.json` is the solver's own node set; top it up by degree if short."""
    output = work_dir / "blocked.json"

    if not output.exists():
        raise FileNotFoundError(
            f"network-immunization produced no blocked.json in {work_dir}\n{stdout}"
        )

    return _pad_removals(
        [int(node) for node in json.loads(output.read_text())["blocked"]],
        work_dir,
        budget,
    )


def _netimm_entry(
    name: str, solver: str, title: str, venue: str, notes: str, fast: bool = False
) -> ExternalBaseline:
    """
    One arm per solver in `allogn/Network-Immunization`, all sharing its single install.

    Six arms from one clone, and three of them are methods
    `research/epidemic_control.md` §11 lists as having NO public release:
    **NetShield** (Tong ICDM'10), **DAVA** and **DAVA-fast** (Zhang & Prakash
    SDM'14). §3.2's claim that EpiLearn ships a NetShield is wrong — that repo's
    tree has no shield, immunization or intervention code at all [derived,
    2026-08-05] — so this clone is the only third-party NetShield or DAVA anywhere,
    and each of these arms is the cross-check on our own reimplementation in
    `coding_agent/tools/immunization_algorithms.py`.
    """
    return ExternalBaseline(
        name=name,
        kind=classical,
        title=title,
        venue=venue,
        repo="https://github.com/allogn/Network-Immunization",
        paper="https://faculty.cc.gatech.edu/~badityap/papers/netshield-icdm10.pdf",
        entry=f"run_solver.py ({solver})",
        task="epidemic_control",
        status="needs_setup",
        # The repo ships a Pipfile pinned to python 3.7 and nothing else; the three
        # packages it lists are already ours, so they are named unpinned here
        requirements=None,
        pip_packages=("networkx", "numpy", "scipy"),
        install_name="network_immunization",
        # Three 2016-era APIs that no longer exist. Each `new` carries a /*gwm*/-style
        # marker or is a strict superset of its `old`, so `setup_baselines.patch()`
        # can tell "already applied" from "not applied yet".
        patches=[
            # networkx 3.0 removed to_numpy_matrix outright
            (
                "NetShieldSolver.py",
                "A = nx.to_numpy_matrix(G, nodelist=nodelist, weight=None)",
                "A = np.asarray(nx.to_numpy_array(G, nodelist=nodelist, weight=None))",
            ),
            (
                "NetShapeSolver.py",
                "F = nx.to_numpy_matrix(self.G, nodelist=self.nodelist, weight='weight')",
                "F = np.asarray(nx.to_numpy_array(self.G, nodelist=self.nodelist, weight='weight'))",
            ),
            # scipy 1.12 removed eigh's `eigvals` in favour of `subset_by_index`
            (
                "NetShieldSolver.py",
                "W, V = eigh(A, eigvals=(M-1, M-1), type=1, overwrite_a=True)",
                "W, V = eigh(A, subset_by_index=[M-1, M-1], type=1, overwrite_a=True)",
            ),
            (
                "NetShapeSolver.py",
                "W, V = eigh(M2, eigvals=(N-1, N-1), type=1, overwrite_a=True)",
                "W, V = eigh(M2, subset_by_index=[N-1, N-1], type=1, overwrite_a=True)",
            ),
            # numpy 1.24 removed the np.warnings alias
            (
                "NetShapeSolver.py",
                "np.warnings.filterwarnings('ignore')",
                "import warnings; warnings.filterwarnings('ignore')",
            ),
        ],
        export=_netimm_export,
        command=_netimm_command,
        parse_seeds=_netimm_parse,
        extra_env=(
            {"GWM_NETIMM_SOLVER": solver, "GWM_NETIMM_FAST": "1"}
            if fast
            else {"GWM_NETIMM_SOLVER": solver}
        ),
        notes=notes,
    )


def _shared_dismantler_entry(
    name: str, source: str, title: str, notes: str
) -> ExternalBaseline:
    """
    Re-register one already-wired NODE-REMOVAL repo under `epidemic_control`.

    The registry's `task` field exists so one task's baselines never join another's
    sweep, and re-registering under a second task with a SHARED install is the
    sanctioned way to say "this method answers both". It genuinely does here:
    `research/epidemic_control.md` §2.5 maps vaccination onto `remove_node`, so a
    dismantler's output IS an allocation under the `vaccinate` lever, and §4.2 lists
    FINDER under this task explicitly.

    Everything but `name`, `task` and `notes` is inherited from the critical-node
    entry, so the two arms share a clone, a venv, a build and an adapter, and a fix
    to one is a fix to both.
    """
    origin = external_baselines[source]

    return replace(
        origin,
        name=name,
        task="epidemic_control",
        install_name=origin.install_name or source,
        notes=notes,
    )


def _cosasi_entry(method: str, title: str, venue: str, notes: str) -> ExternalBaseline:
    """One arm per cosasi estimator, all sharing its single install."""
    return ExternalBaseline(
        name=f"cosasi_{method}",
        kind=classical,
        title=title,
        venue=venue,
        repo="https://github.com/lmiconsulting/cosasi",
        paper="https://joss.theoj.org/papers/10.21105/joss.04894",
        entry=f"cosasi package ({method})",
        task="source_localization",
        status="needs_setup",
        # Warning: The published `cosasi` wheel declares NO dependencies, so a bare
        # `pip install cosasi` imports and then fails on `networkx`. Its clone
        # carries a requirements.txt, but every line in it is pinned to 2022
        # (numpy 1.21, scikit-learn 1.1) and resolving those on a current Python
        # fails outright — so the deps are listed here UNPINNED instead, and the
        # one incompatibility that causes is shimmed in the driver rather than
        # frozen around.
        requirements=None,
        pip_packages=(
            "cosasi",
            "networkx",
            "numpy",
            "scipy",
            "scikit-learn",
            "ndlib",
            "six",
            "matplotlib",
        ),
        install_name="cosasi",
        export=partial(_localization_export, driver="cosasi_driver.py"),
        command=_localization_command,
        parse_seeds=_localization_parse,
        extra_env={"GWM_COSASI_METHOD": method},
        notes=notes,
    )


def _graphsl_entry(method: str, title: str, venue: str, notes: str) -> ExternalBaseline:
    """One arm per published method, all six sharing GraphSL's single install."""
    return ExternalBaseline(
        name=f"graphsl_{method}",
        kind=classical if method in ("lpsi", "netsleuth", "ojc") else learned,
        title=title,
        venue=venue,
        repo="https://github.com/xianggebenben/GraphSL",
        paper="https://arxiv.org/abs/2405.03724",
        entry=f"GraphSL package ({method})",
        task="source_localization",
        status="needs_setup",
        # The clone carries no usable requirements file; the PyPI package is the
        # install. `six` and `matplotlib` are undeclared transitive imports of
        # GraphSL's own `utils` module, which fails at import without them.
        requirements=None,
        pip_packages=("GraphSL", "six", "matplotlib"),
        install_name="graphsl",
        export=_graphsl_export,
        command=_graphsl_command,
        parse_seeds=_graphsl_parse,
        extra_env={
            "GWM_GRAPHSL_METHOD": method,
            "GWM_GRAPHSL_EPOCHS": str(graphsl_epochs),
        },
        notes=notes,
    )


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
            # The latent search — the step that actually picks the seeds — never
            # zeroes its gradients, so all 300 iterations accumulate into one
            # running sum and z_hat is driven away from its (good) initial
            # value in a direction dominated by the earliest steps. The symptom
            # is a loss that random-walks instead of descending. zero_grad is
            # present in the VAE loop above, so this is an omission, not intent.
            (
                "genim.py",
                "for i in range(300):\n    \n    x_hat = decoder(z_hat)",
                "for i in range(300):\n    z_optimizer.zero_grad()\n"
                "    x_hat = decoder(z_hat)",
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
            "in research/influence_maximization.md §5.1 and are directly "
                "comparable on Jazz and "
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
            "at scale (see research/influence_maximization.md §6.1)."
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
            "transcribed in research/influence_maximization.md §5.1."
        ),
    ),
    "oim": ExternalBaseline(
        name="oim",
        kind=classical,
        title="OIM — Online Influence Maximization",
        venue="KDD 2015",
        repo="https://github.com/smaniu/oim",
        paper="https://arxiv.org/abs/1506.01188",
        entry="C++ (make)",
        task="adaptive_online_im",
        status="needs_setup",
        requirements=None,
        build=["make"],
        notes=(
            "CORRECTED 2026-08-01, interface read from source 2026-08-03. Code "
            "IS published by a co-author (research/adaptive_online_im.md §3.2) "
            "and the method is online IM, so it belongs to this task.\n\n"
            "  argv    ./oim real <graph> <exploit> <budget> <k> [model] "
            "[samples]  (positional, src/main.cpp)\n"
            "  build   make; needs GCC >= 4.9, C++14, and Boost HEADERS in "
            "/usr/local/include (header-only, no linking)\n\n"
            "NOT WIRED, and the obstacle is the PROBLEM SETTING rather than the "
            "interface. `budget` here is the number of TRIALS, not a seed count: "
            "OIM runs a sequence of campaigns, updates a Beta posterior per edge "
            "from observed activations, and reports the UNION over trials. There "
            "is no single seed set to hand back, and the union grows "
            "monotonically in the trial count, so it is not comparable to a "
            "single-campaign spread (§8.2 trap 2). A faithful arm needs a "
            "repeated-campaign external protocol that our contract does not "
            "have. DeepIM sidestepped this by re-running OIM under its own "
            "single-campaign protocol; those numbers are transcribed in "
            "research/influence_maximization.md §5.1."
        ),
    ),
    # Adaptive / online IM (research/adaptive_online_im.md §3.1, §3.2, §4.2).
    #
    # None of the five below is WIRED, and that is deliberate rather than
    # unfinished. An adapter has to know a repo's input format, its CLI, and
    # where it writes its seeds; every wired entry above was written after
    # cloning and running the thing. Writing one from a README would be the
    # guesswork the `lense` note already calls out. Each entry therefore records
    # the URL, the entry point, and what specifically remains, which is the
    # difference between "not done" and "not known".
    "adaptiveim": ExternalBaseline(
        name="adaptiveim",
        kind=classical,
        title="AdaptGreedy / EPIC: scalable adaptive IM",
        venue="PVLDB 11, 2018",
        repo="https://github.com/kkhuang81/AdaptiveIM",
        paper="http://www.vldb.org/pvldb/vol11/p1029-han.pdf",
        entry="C++ (make -f Makefile_expepic)",
        task="adaptive_online_im",
        status="needs_setup",
        rounds_aware=True,
        requirements=None,
        build=["make", "-f", "Makefile_expepic"],
        patches=[
            # head.h's rdtsc() is x86 inline asm, so the repo does not compile on
            # arm64 at all ("invalid output constraint '=a' in asm"). It feeds
            # only the RUN_TIME diagnostic macro and two commented-out lines, so
            # a steady_clock counter is functionally identical and portable.
            (
                "expepic/head.h",
                'uint64 rdtsc(void)\n{\n    unsigned a, d;\n    //asm("cpuid");\n'
                '    asm volatile("rdtsc" : "=a" (a), "=d" (d));\n'
                "    return (((uint64)a) | (((uint64)d) << 32));\n}",
                "uint64 rdtsc(void)\n{\n    return (uint64)std::chrono::"
                "duration_cast<std::chrono::nanoseconds>(\n        std::chrono::"
                "steady_clock::now().time_since_epoch()).count();\n}",
            ),
            ("expepic/head.h", "#include <cstring>", "#include <cstring>\n#include <chrono>"),
        ],
        export=_adaptiveim_export,
        command=_adaptiveim_command,
        parse_seeds=_adaptiveim_parse,
        allow_nonzero_exit=True,
        notes=(
            "WIRED 2026-08-03, built and run. THE reference implementation for "
            "this task, by the authors. Our "
            "library's `adapt_greedy` and `adapt_epic` are Python "
            "reimplementations of AdaptGreedy and its RIS instantiation, good "
            "enough as condition-1 arms, but this is the original.\n\n"
            "INTERFACE, read from source on 2026-08-03 (expepic/aim.cpp, "
            "graph.h, infgraph.h) so the remaining work is known rather than "
            "guessed:\n"
            "  argv    ./exp_epic -dataset <dir>/<name>/ -model IC|LT "
            "-epsilon 0.5 -k <k> -batch <b> -seedfile <path> -time 1\n"
            "  reads   <dir>/<name>/attribute.txt  ('n=<N>' and 'm=<M>')\n"
            "          <dir>/<name>/graph_ic.inf   BINARY, m records of "
            "(int src, int dst, double p), 16 bytes each, indexed by dst\n"
            "          <dir>/<name>/<name>_<i>     possible world i, TEXT, one "
            "'src dst' pair per line (a live-edge sample). We can GENERATE "
            "these: it is the standard IC possible world, and the format is "
            "read verbatim from load_possible_world().\n"
            "  writes  the seedfile, space-separated node ids, in SELECTION "
            "order, so slicing by <b> recovers the per-batch sets\n"
            "  build   make -f Makefile_expepic\n\n"
            "The possible-world files the README defers to an unlinked Tools "
            "repo are generated by the adapter: they are IC live-edge samples "
            "drawn from OUR probabilities, so it adapts against the dynamics "
            "our referee scores it under.\n\n"
            "READ THE ARM AS: AdaptGreedy's SCHEDULE replayed under our referee. "
            "Its batches were chosen against its own realizations, not against "
            "the state our simulator goes on to produce, so this is not the "
            "algorithm adapting inside our environment. True cross-process "
            "adaptivity would need it to accept an already-active set per round, "
            "which its CLI does not expose."
        ),
    ),
    "mrim": ExternalBaseline(
        name="mrim",
        kind=classical,
        title="MRIM: Multi-Round Influence Maximization",
        venue="KDD 2018",
        repo="https://github.com/lichao-sun/Multi-Round-Influence-Maximization",
        paper="https://arxiv.org/abs/1802.04189",
        entry="n/a",
        task="adaptive_online_im",
        status="blocked",
        blocker=(
            "THE CODE IS NOT IN THE REPOSITORY. Cloned and inspected "
            "2026-08-03: the README says verbatim 'If you wana the code, "
            "please email to james.lichao.sun@gmail.com'. What the repo does "
            "ship is maxinf_v2.1.0, which is Wei Chen's older SINGLE-round "
            "max_influence toolkit (cgreedy, degreediscount_ic, pmia, "
            "general_cascade) plus three Windows .exe binaries. Not one file "
            "under code/ mentions multi-round or MRIM. Wiring it would put Wei "
            "Chen's static IM code in the table under MRIM's name, which is "
            "worse than an empty cell. Unblocking means emailing the author."
        ),
        notes=(
            "Our own --campaigns flag now implements the multi-round SETTING "
            "(r separate diffusions scored on their union, §1.5), so the "
            "protocol side is no longer the obstacle; the algorithm is. §11 "
            "also records that this paper's tables were never extracted, so "
            "there would be no published number to check an adapter against."
        ),
    ),
    "rl4im": ExternalBaseline(
        name="rl4im",
        kind=learned,
        title="RL4IM: contingency-aware IM as a multi-round MDP",
        venue="UAI 2021",
        repo="https://github.com/wmd3i/RL4IM-Contingency",
        paper="https://arxiv.org/abs/2106.07039",
        entry="main.py (sacred; TRAINS before selecting)",
        task="adaptive_online_im",
        status="needs_setup",
        rounds_aware=True,
        # torch 1.7.0 and torch_geometric 1.6.3 are 2020 pins with no wheels for
        # a modern interpreter, so the venv has to be built on 3.9
        python="3.9",
        patches=[
            # ipdb 0.12 builds with use_2to3, which setuptools removed in v58,
            # so the install dies before torch is even reached. It is a DEBUGGER,
            # but src/tasks/task_rl4im.py imports it at module scope, so it must
            # be unpinned rather than dropped.
            # NB the replacement must not be a SUBSTRING of the original, or
            # setup_baselines' idempotency check ("already applied?") matches
            # the unpatched line and skips the patch silently
            ("requirements.txt", "ipdb==0.12", "ipdb>=0.13"),
        ],
        export=_rl4im_export,
        command=_rl4im_command,
        parse_seeds=_rl4im_parse,
        extra_env={"PYTHONUNBUFFERED": "1"},
        notes=(
            "WIRED 2026-08-03, adapter written from source. INSTALL VERIFIED "
            "up to one architecture-specific blocker: `uv pip install -r "
            "requirements.txt` on python 3.9 resolves everything except "
            "`torch==1.7.0 has no wheels with a matching Python ABI tag`, "
            "because PyTorch published no arm64 macOS wheels before 1.12. On "
            "linux x86_64, which is where the sbatch scripts run, torch 1.7.0 "
            "ships cp36-cp39 wheels and the pinned interpreter above resolves "
            "it. So this is untested end to end on THIS machine and expected to "
            "install on the cluster; run "
            "`python -m baselines.setup_baselines --only rl4im` there first.\n\n"
            "It TRAINS before it selects (no checkpoint ships). Budget from its "
            "own config: max_global_t=2000 steps, minutes rather than hours; the "
            "540k IC simulations that bound implies were measured at about half "
            "a minute total (research/adaptive_online_im.md §11).\n\n"
            "TWO HONEST LIMITS on the arm. (1) It generates its own graphs: "
            "there is no flag for 'run on this graph', so the adapter matches "
            "our graph on size and family (powerlaw_cluster, m=3, p=0.05, its "
            "own config values) and lets it train there. That is zero-shot "
            "transfer from a distribution, which is the regime its paper "
            "reports, not a run on our exact graph. (2) It models WILLINGNESS "
            "(q=0.6, a seed may decline). parse_seeds reads `invited`, not "
            "`present`: scoring the survivors would credit it with a budget it "
            "did not spend, since our action space has no notion of a seed "
            "declining.\n\n"
            "The closest published thing to our setting and the ONLY genuinely "
            "multi-round RL baseline in the literature (§4.1: DISCO, PIANO, "
            "GCOMB and ToupleGDD are all RL over seed-set CONSTRUCTION, with no "
            "cascade between actions, so they belong to static IM). Our claim "
            "against it is model-based vs model-free: RL4IM learns a policy "
            "Q(s,a), we learn a transition function and plan against it. "
            "Protocol from §5.5: powerlaw-cluster graphs (now loadable as "
            "--dataset powerlaw_cluster), |V|=200, T=2 rounds, B=4 per round, "
            "IC p=0.1, 100 sims per number. Two mismatches to resolve before "
            "wiring: its `willingness` q=0.6 (a seed may DECLINE, which our "
            "action space has no notion of) and the fact that all its results "
            "are figure-only, so there is no table to validate against.\n\n"
            "Inspected 2026-08-03: main.py is driven by `sacred` "
            "(Experiment/FileStorageObserver) rather than plain argparse, and "
            "NO pretrained checkpoint is shipped, so an arm has to TRAIN a DQN "
            "before it can select anything. That makes it the largest single "
            "piece of the five, and unlike the bandit entries it is a genuine "
            "contract fit: it does produce per-round seed sets. Highest-value "
            "target once the per-round contract exists."
        ),
    ),
    "oim_lt": ExternalBaseline(
        name="oim_lt",
        kind=classical,
        title="Online IM under the Linear Threshold model",
        venue="NeurIPS 2020",
        repo="https://github.com/Ritchiegit/Online_Influence_Maximization_under_Linear_Threshold_Model",
        paper="https://arxiv.org/abs/2011.06378",
        entry="see repo",
        task="adaptive_online_im",
        status="needs_setup",
        notes=(
            "The LT counterpart of IMLinUCB, and the official implementation. "
            "Inspected 2026-08-03: Main.py is a SIMULATION HARNESS, not a seed "
            "selector. It instantiates bandit algorithms (OIM_ETC, "
            "IMLinUCB_LT), runs them for `iterationTime` rounds against its own "
            "datasets, and accumulates reward, loss and regret against an "
            "oracle. There is no entry point that answers 'give me k seeds for "
            "this graph', which is what our contract asks for. Same "
            "repeated-campaign obstacle as `oim`, plus the harness shape. Worth "
            "keeping registered because it is the only online-IM code found "
            "that targets LT at all, and our LT arms otherwise have no "
            "published online baseline."
        ),
    ),
    "timlinucb": ExternalBaseline(
        name="timlinucb",
        kind=classical,
        title="IMLinUCB / TIMLinUCB: linear-generalization bandit IM",
        venue="NeurIPS 2017",
        repo="https://github.com/olety/TIMLinUCB",
        paper="https://arxiv.org/abs/1605.06593",
        entry="see repo",
        task="adaptive_online_im",
        status="needs_setup",
        notes=(
            "THIRD-PARTY, not the authors'. Wen et al. released no code; this is "
            "a temporal port found by the review (§3.2), so any number it "
            "produces is attributable to this repo and not to the paper, and "
            "must be labelled that way if it is ever reported.\n\n"
            "Inspected 2026-08-03: the surface is library functions, not a CLI "
            "(timlinucb.py exposes timlinucb(), oim_node2vec(), "
            "timlinucb_parallel_t()), and it needs node2vec edge features built "
            "first (get_features_nodes / generate_node2vec_fetures, with a "
            "vendored node2vec/). Same repeated-campaign obstacle as the other "
            "two bandit entries: it optimizes cumulative regret over rounds, "
            "not one seed set. Scale note from §5.4: the strongest theory "
            "result in bandit IM was validated on a 327-node Facebook subgraph, "
            "and §6.2 records that the exact subgraph is unpublished, so its own "
            "figure cannot be reproduced regardless."
        ),
    ),
    # -- critical node detection -------------------------------------------
    # Registered, none wired. Every one of these solves the STRUCTURAL variant:
    # it returns a removal set optimizing a connectivity functional, whereas our
    # arms are scored on the diffusion the removal set fails to stop
    # (research/critical_node_detection.md §2.2). That is not a blocker for
    # comparison — their seed set crosses the process boundary and OUR referee
    # scores it, exactly as on the IM side — but it does mean a fair report has to
    # show both columns, which is what the report's structural section is for.
    # Warning: Two shared traps before any of these is wired (§8.2): almost all of them
    # run on the LARGEST CONNECTED COMPONENT of the input silently (trap 7), and
    # several ship a REINSERTION pass that makes `X` and `X+R` different methods
    # cited under one name (trap 2).
    # Influence blocking (research/influence_blocking.md §3.2, §4). Only FOUR repos
    # exist in this entire literature — §11 records that no public code was found for
    # NIE, CMIA-H/CMIA-O, CLDAG, the DRL rumour-minimization line, OCIM or JCCIM
    # after searching GitHub for each — so these are its whole reproducible surface.
    "sandimin": ExternalBaseline(
        name="sandimin",
        kind=classical,
        title="SandIMIN: efficient influence minimization via node blocking",
        venue="PVLDB 17(10), 2024",
        repo="https://github.com/wjh0116/IMIN",
        paper="https://arxiv.org/abs/2405.12871",
        entry="C++ (g++ -O3 Sandwich.cpp sfmt/SFMT.c)",
        task="influence_blocking",
        status="needs_setup",
        requirements=None,
        subdir="SandIMIN_code",
        build=["g++", "-O3", "-o", "IMIN", "Sandwich.cpp", "sfmt/SFMT.c",
               "-DSFMT_MEXP=19937"],
        patches=[
            # `rdtsc` is x86 inline asm and does not assemble on ARM. It is only ever
            # read by the RUN_TIME macro, which the code never invokes — every timing
            # that reaches a result goes through std::chrono — so a portable clock()
            # is behaviour-preserving rather than an approximation.
            (
                "SandIMIN_code/head.h",
                '    asm volatile("rdtsc" : "=a" (a), "=d" (d));\n'
                "    return (((uint64)a) | (((uint64)d) << 32));",
                "    (void)a; (void)d;\n    return (uint64)clock();",
            ),
            # The repo computes its blocker set and then throws it away: it writes
            # only (influence, influence-after, decrease, time) and its own
            # OutputSeedSetToFile call is commented out. We need the SET, because our
            # referee scores the set rather than trusting their spread number.
            (
                "SandIMIN_code/Sandwich.cpp",
                'string seedfile = "results/res_" + arg.dataset;',
                'string seedfile = arg.dataset + "gwm_blockers.txt";',
            ),
            (
                "SandIMIN_code/Sandwich.cpp",
                "ofstream of(seedfile, ios::app);",
                "ofstream of(seedfile);",
            ),
            (
                "SandIMIN_code/Sandwich.cpp",
                "    //OutputSeedSetToFile(g.seedSet, arg);",
                "    OutputSeedSetToFile(g.seedSet, arg);",
            ),
        ],
        export=_sandimin_export,
        command=_sandimin_command,
        parse_seeds=_sandimin_parse,
        notes=(
            "The most comparable published table in this literature: its Table 5 "
            "reports DECREASED SPREAD — literally our prevented-influence metric — "
            "under IC with weighted-cascade p = 1/in-degree at absolute k = 10..50, "
            "on EmailCore and YouTube among others, both of which we load. Sandwich "
            "approximation over a submodular lower bound of the non-submodular IMIN "
            "objective. `SANDIMIN_ALGO=SandIMIN-` selects the cheaper variant. "
            "Warning: the shipped `el2bin` is a prebuilt x86 binary with no source, so "
            "the adapter writes its packed `(int, int, double)` graph_ic.inf "
            "directly instead — verified against `graph.h::readGraph`, which mmaps "
            "the file and steps by 2*sizeof(int)+sizeof(double). Read its own §5.3 "
            "row before reading ours: its trivial LHGA heuristic beats both of its "
            "principled methods in 6 of that table's 30 cells."
        ),
    ),
    "imin_joc": ExternalBaseline(
        name="imin_joc",
        kind=classical,
        title="AdvancedGreedy / GreedyReplace: influence minimization via blocking strategies",
        venue="INFORMS Journal on Computing, 2025 (ICDE 2023 line)",
        repo="https://github.com/INFORMSJoC/2024.0591",
        paper="https://arxiv.org/abs/2312.17488",
        entry="C++ (g++ -std=c++11 -O3), configuration read from stdin",
        task="influence_blocking",
        status="needs_setup",
        requirements=None,
        subdir="src",
        build=["sh", "-c",
               "g++ -o AdvancedGreedy AdvancedGreedy.cpp -std=c++11 -O3 && "
               "g++ -o GreedyReplace GreedyReplace.cpp -std=c++11 -O3"],
        patches=[
            # `bits/stdc++.h` is a libstdc++ convenience header that clang/libc++ does
            # not ship, so the repo does not compile outside GCC at all
            (
                "src/AdvancedGreedy.cpp",
                "#include <bits/stdc++.h>",
                "#include <algorithm>\n#include <cmath>\n#include <cstring>\n"
                "#include <ctime>\n#include <fstream>\n#include <iostream>\n"
                "#include <map>\n#include <queue>\n#include <random>\n"
                "#include <set>\n#include <string>\n#include <utility>\n"
                "#include <vector>",
            ),
            (
                "src/GreedyReplace.cpp",
                "#include <bits/stdc++.h>",
                "#include <algorithm>\n#include <cmath>\n#include <cstring>\n"
                "#include <ctime>\n#include <fstream>\n#include <iostream>\n"
                "#include <map>\n#include <queue>\n#include <random>\n"
                "#include <set>\n#include <string>\n#include <utility>\n"
                "#include <vector>",
            ),
            # Both binaries draw their own random sources from a fixed seed, so
            # without this they would answer a DIFFERENT rumour than every other arm
            # in the sweep and their column would be incomparable
            (
                "src/AdvancedGreedy.cpp",
                "        uniform_int_distribution<int> dist(0, n - 1);\n"
                "        x = dist(rand_num);\n",
                "        cin >> x;\n",
            ),
            (
                "src/GreedyReplace.cpp",
                "        uniform_int_distribution<int> dist(0, n - 1);\n"
                "        x = dist(rand_num);\n",
                "        cin >> x;\n",
            ),
            # ...and both write only (budget, spread, time), never the blocked set
            (
                "src/AdvancedGreedy.cpp",
                '    out << budget << "\\t" << res << "\\t" << totalTime / CLOCKS_PER_SEC << endl;',
                '    ofstream blockers((outName + ".blockers").c_str());\n'
                "    for (int i = 0; i < n; i++)\n"
                "        if (remove_flag[i] && find(sources.begin(), sources.end(), i) == sources.end())\n"
                "            blockers << i << endl;\n"
                "    blockers.close();\n"
                '    out << budget << "\\t" << res << "\\t" << totalTime / CLOCKS_PER_SEC << endl;',
            ),
            (
                "src/GreedyReplace.cpp",
                '    out << budget << "\\t" << res << "\\t" << totalTime / CLOCKS_PER_SEC << endl;',
                '    ofstream blockers((outName + ".blockers").c_str());\n'
                "    for (int i = 0; i < n; i++)\n"
                "        if (remove_flag[i] && find(sources.begin(), sources.end(), i) == sources.end())\n"
                "            blockers << i << endl;\n"
                "    blockers.close();\n"
                '    out << budget << "\\t" << res << "\\t" << totalTime / CLOCKS_PER_SEC << endl;',
            ),
        ],
        export=_joc_export,
        command=_joc_command,
        parse_seeds=_joc_parse,
        notes=(
            "The INFORMS artifact for the ICDE'23 vertex-blocking line, and the "
            "source of the sharpest single fact in this literature: its Tables V-VI "
            "put GreedyReplace within 0.12% of the EXACT optimum at b = 4, in a "
            "third of a second against 22 hours. `JOC_ALGO=GreedyReplace` selects "
            "the replace variant (the default is AdvancedGreedy). Both recompute "
            "edge probabilities themselves; we pass propagation model 1 (WC, "
            "1/in-degree), which is exactly what `--prob-model weighted` generates, "
            "so the repo simulates the same edge model we do. Both are node-blocking "
            "methods, so run them with `--blocking-lever node_block`."
        ),
    ),
    "diffim": ExternalBaseline(
        name="diffim",
        kind=learned,
        title="DiffIM: differentiable influence minimization with surrogate modeling",
        venue="AAAI 2025",
        repo="https://github.com/junghunl/DiffIM",
        paper="https://arxiv.org/abs/2502.01031",
        entry="Jupyter notebooks (PyTorch Geometric)",
        task="influence_blocking",
        status="needs_setup",
        # The repo's own requirements.txt pins torch-scatter==2.1.0+pt112cu113 and
        # torch-sparse==0.6.16+pt112cu113 — CUDA 11.3 wheels that exist only on the
        # PyG wheel index, so installing it verbatim fails on any CPU machine.
        # Verified 2026-08-04. The trimmed set below is what its code actually
        # imports; modern PyG needs neither scatter nor sparse for GCNConv.
        requirements=None,
        # Every third-party name its notebooks import, extracted from the cells
        # rather than guessed: `optuna` is only used by `train.ipynb`'s
        # `hparam_tuning`, which we never call — but the runner execs that
        # notebook's cells to reach `train`, so its top-level import still has to
        # resolve. Verified by running: without it the runner dies on
        # ModuleNotFoundError before reaching a single algorithm.
        pip_packages=(
            "torch",
            "torch-geometric",
            "numpy",
            "networkx",
            "scipy",
            "optuna",
        ),
        returns_edges=True,
        export=_diffim_export,
        command=_runner_command,
        parse_seeds=_diffim_parse,
        notes=(
            "The closest published analogue to what this project builds: a GNN "
            "surrogate for influence plus a CONTINUOUS RELAXATION of the edge "
            "decisions, p~(u,v) = p(u,v) * r~(u,v) with r~ in [0,1], optimized by "
            "gradient descent — which is literally our `set_edge_weight` op. It is "
            "the existence proof that differentiable blocking works, and it is NOT "
            "action-conditioned or rolled forward in time, which is exactly the gap "
            "we target. An EDGE method: run it with `--blocking-lever edge_block` or "
            "`weight_block`. `DIFFIM_ALG` selects the member (DiffIM, DiffIM+, "
            "DiffIM++, and its own BPM / RIS / MDS / KED / greedy baselines); "
            "`DIFFIM_MODEL` names a shipped checkpoint instead of training on our "
            "graph, and `DIFFIM_TRAIN_SAMPLES` sizes that training. Warning: the repo "
            "ships ONLY notebooks and no .py, and its cross-imports need "
            "`algorithms/` to be a package it is not — the adapter execs the code "
            "cells into one namespace rather than fighting `import_ipynb`."
        ),
    ),
    "stratlearner": ExternalBaseline(
        name="stratlearner",
        kind=learned,
        title="StratLearner: learning a strategy for misinformation prevention",
        venue="NeurIPS 2020",
        repo="https://github.com/cdslabamotong/stratLearner",
        paper="https://arxiv.org/abs/2009.14337",
        entry="Python (structured SVM over random-subgraph features)",
        task="influence_blocking",
        status="blocked",
        blocker=(
            "RUNNING IT ON OUR GRAPHS IS NOT AN ADAPTER, IT IS A RE-DERIVATION. Three "
            "things it needs are not in the clone and are not computable at our "
            "sizes, all verified against the code on 2026-08-04. (1) The `data/` "
            "directory is a separate download from udel.edu (its README says so; the "
            "URL 301s), and every path in `train.py` is built from it. (2) Each "
            "feature is a random subgraph PLUS a full pairwise DISTANCE MATRIX — "
            "`DiffusionGraph.__init__` reads `<i>_distance.txt` per feature — so 800 "
            "features on our smallest real graph is O(800 x N^2) numbers on disk. "
            "That is why the paper's own graphs are 512-1024 nodes. (3) Training "
            "needs 2500 labelled attacker/protector PAIRS whose protector is a "
            "best-known approximation, i.e. solving the blocking problem near-"
            "optimally 2500 times is the PREREQUISITE for running the method we "
            "would be benchmarking. Its published Table 1 is still transcribed in "
            "research/influence_blocking.md §5.6 and is usable as a reference "
            "without running it — including the row that matters most to us, plain "
            "proximity at 0.770/0.776 against a GCN at 0.281/0.091."
        ),
        notes=(
            "Kept registered rather than deleted because §5.6 is one of the two most "
            "useful tables in this literature for us: a plain GCN scores 0.091-0.657 "
            "on the SAME task depending only on the graph family, which is the "
            "sharpest published argument for a structured head over an unstructured "
            "backbone, and proximity beats every learned method except StratLearner "
            "on two of its three graphs."
        ),
    ),
    "finder": ExternalBaseline(
        name="finder",
        kind=learned,
        title="FINDER: finding key players via deep RL",
        venue="Nature Machine Intelligence 2, 2020",
        repo="https://github.com/FFrankyy/FINDER",
        paper="https://pmc.ncbi.nlm.nih.gov/articles/PMC8191335/",
        entry="Cython + TensorFlow 1.x (make)",
        task="critical_node_detection",
        status="needs_setup",
        python="3.7",
        build=["make"],
        export=_finder_export,
        command=_runner_command,
        parse_seeds=_order_parse,
        notes=(
            "The most-cited learned dismantler and the one every later paper "
            "compares to. Ships four trained variants (CN / ND, unit and "
            "node-weighted cost); pick the ND unit-cost one to match a cardinality "
            "budget. TO WIRE: clone, build the Cython extensions, read which of "
            "`FINDER_CN` / `FINDER_ND` the entry point drives and what its graph "
            "format is. Warning: TensorFlow 1.x and Cython pin this to Python 3.7, which "
            "is why it is the most expensive entry here to install. Warning: §11 records "
            "that its per-network results are HEATMAP-ONLY (no arXiv version, "
            "paywalled PDF, and the `results/` directory its README advertises does "
            "not exist in master), so there is no published table to validate an "
            "adapter against — you would be generating the number, not checking it."
        ),
    ),
    "gdm": ExternalBaseline(
        name="gdm",
        kind=learned,
        title="GDM: geometric deep learning for network dismantling",
        venue="Nature Communications 12, 2021",
        repo="https://github.com/NetworkScienceLab/GDM",
        paper="https://arxiv.org/abs/2101.02453",
        entry="PyTorch Geometric",
        task="critical_node_detection",
        status="needs_setup",
        export=_gdm_export,
        command=_runner_command,
        parse_seeds=_order_parse,
        notes=(
            "Supervised (not RL) on brute-force-optimal dismantling of small "
            "graphs, and STATIC: one scoring pass with no recomputation, which "
            "makes it the cheapest learned entry to run. `GDM+R` adds reinsertion "
            "and is a different method with different numbers (§8.2 trap 2). TO "
            "WIRE: clone, install PyG, read its graph format and which pretrained "
            "checkpoint corresponds to the ND objective. Relevant to us beyond its "
            "score: MIND measured GDM's dismantling order at Spearman 0.762 against "
            "a PCA of its own input features (§9.5), which is the correlation our "
            "report's `degree_rank_spearman` column measures for our arms."
        ),
    ),
    "mind": ExternalBaseline(
        name="mind",
        kind=learned,
        title="MIND: message-iteration network dismantling",
        venue="AAAI 2026",
        repo="https://github.com/HaozheTian/MIND-ND",
        paper="https://arxiv.org/abs/2508.00706",
        entry="PyTorch",
        task="critical_node_detection",
        status="needs_setup",
        export=_mind_export,
        command=_runner_command,
        parse_seeds=_order_parse,
        notes=(
            "Current SOTA and the ONLY paper with FINDER + GDM + a 2026 method on "
            "one 47-network table, which makes its Table 5 the single most useful "
            "comparison target in this literature. Drops handcrafted structural "
            "features entirely; O(|V|+|E|). TO WIRE: clone, read the graph format "
            "and which of MIND-AM / MIND-MP the entry point drives. Warning: Its table is "
            "published AUC RELATIVE TO ITSELF = 100 (§8.2 trap 5), so it is "
            "internally consistent and externally useless — you cannot combine it "
            "with an absolute number, only re-run it. The rows to read first are "
            "adaptive degree at 119.9 vs FINDER at 115.0 (§9.3) and the "
            "power-grid rows where FINDER scores 161.7 against EI's 80.6 (§9.4)."
        ),
    ),
    "spr": ExternalBaseline(
        name="spr",
        kind=learned,
        title="SPR / HoGNN: higher-order GNN dismantling",
        venue="Communications Physics 9:181, 2026",
        repo="https://github.com/zhouwn/spr",
        paper="https://www.nature.com/articles/s42005-026-02601-y",
        entry="PyTorch",
        task="critical_node_detection",
        status="blocked",
        blocker=(
            "THE REPOSITORY IS EMPTY. `git clone` succeeds and yields a bare .git "
            "with no commits on its default branch ('your current branch main does "
            "not have any commits yet'), so there is no code to wire — verified "
            "2026-08-03. research/critical_node_detection.md §4 lists this URL as "
            "the paper's code link; that link resolves but publishes nothing. "
            "Re-check upstream before spending time here."
        ),
        notes=(
            "The one learned dismantler with an ABSOLUTE per-network table that "
            "beats FINDER, GDM, NIRM and DCRS in one printed comparison (§5.4), so "
            "it is the cheapest published number to check an adapter against. Its "
            "graphs overlap ours: `netscience`, Yeast PPI, Crime and Human PPI "
            "(Vidal / Figeys) are all loadable here. TO WIRE: clone, read the graph "
            "format and the rho-at-Theta reduction it reports."
        ),
    ),
    "nirm": ExternalBaseline(
        name="nirm",
        kind=learned,
        title="NIRM: neural influence ranking for target attack",
        venue="CIKM 2022",
        repo="https://github.com/JiazhengZhang/NIRM",
        paper="https://arxiv.org/abs/2208.07792",
        entry="PyTorch (GAT)",
        task="critical_node_detection",
        status="needs_setup",
        export=_nirm_export,
        command=_runner_command,
        parse_seeds=_order_parse,
        notes=(
            "Supervised on brute-forced optimal removal sets of 20-30-node "
            "synthetic graphs, applied zero-shot — the same train-tiny/apply-large "
            "claim our BA-100 regime makes (§9.5). Its Table 3 is the ONLY numeric "
            "table in this file that separates adaptive from one-pass removal on "
            "the same method and graph (`UsPower`: rho 8.58% vs 16.81%, a 1.96x "
            "gap), which is the measurement §8.2 trap 1 exists for and which our "
            "`degree_removal` vs `adaptive_degree` pair reproduces. TO WIRE: clone, "
            "read its graph format; note it runs on the LCC (its `Ca-GrQc` is "
            "4,158/13,422 where ours is 5,242/14,484)."
        ),
    ),
    "dcrs": ExternalBaseline(
        name="dcrs",
        kind=learned,
        title="DCRS: diffusion competence and role significance",
        venue="WWW 2023",
        repo="https://github.com/JiazhengZhang/DCRS",
        paper="https://arxiv.org/abs/2301.12349",
        entry="PyTorch",
        task="critical_node_detection",
        status="needs_setup",
        export=_dcrs_export,
        command=_runner_command,
        parse_seeds=_order_parse,
        notes=(
            "The direct NIRM follow-up, and the learned entry closest in spirit to "
            "what we do: it encodes a node's DIFFUSION COMPETENCE rather than pure "
            "topology, which is the same observation §2.2 builds our variant on. "
            "Best on 21 of 22 networks in its own table. TO WIRE: clone, read its "
            "graph format; same LCC caveat as NIRM."
        ),
    ),
    "gnd": ExternalBaseline(
        name="gnd",
        kind=classical,
        title="GND / GNDR: generalized network dismantling",
        venue="PNAS 116(14), 2019",
        repo="https://github.com/renxiaolong/Generalized-Network-Dismantling",
        paper="https://arxiv.org/abs/1801.01357",
        entry="C++ (g++ GND.cpp)",
        task="critical_node_detection",
        status="needs_setup",
        requirements=None,
        # NOT `make`: the shipped Makefile also builds `reinsertion`, which links
        # boost_program_options against a hardcoded Windows/cygwin path and fails
        # on any Unix without that exact layout. The GND binary itself needs no
        # boost, so it is built directly and GNDR is left to our own `gndr`.
        build=["g++", "GND.cpp", "-O3", "-o", "GND"],
        # Upstream hardcodes the node count, filenames and stopping threshold as
        # compile-time constants, so one binary serves exactly one graph. This
        # makes main() read them from argv, defaulting to the shipped CrimeNet
        # example when called with none — verified to reproduce its 290-node
        # result byte for byte. Idempotent: re-running setup is a no-op.
        # Warning: Each `new` carries a /*gwm*/ marker so it is NOT a substring of its
        # own `old`. setup_baselines.patch() treats "new already in text" as
        # "already applied", and dropping `const` alone leaves the replacement
        # inside the original — so the de-const edits would be skipped forever
        # and the build would fail on `cannot assign to const-qualified type`.
        patches=[
            (
                "GND.cpp",
                "const int NODE_NUM = 754;",
                "/*gwm*/ int NODE_NUM = 754;",
            ),
            (
                "GND.cpp",
                "const int REMOVE_STRATEGY = 1;",
                "/*gwm*/ int REMOVE_STRATEGY = 1;",
            ),
            ("GND.cpp", "const int PLOT_SIZE = 1;", "/*gwm*/ int PLOT_SIZE = 1;"),
            (
                "GND.cpp",
                "const int TARGET_SIZE = 0.01*NODE_NUM;",
                "/*gwm*/ int TARGET_SIZE = 0.01*754;",
            ),
            (
                "GND.cpp",
                "int main()\n{",
                "int main(int argc, char** argv)\n{\n"
                "\t// argv: <node_num> <in_edgelist> <out_removed_ids> <out_plot> "
                "[target_size] [strategy]\n"
                "\tif (argc >= 5) {\n"
                "\t\tNODE_NUM = atoi(argv[1]);\n"
                "\t\tFILE_NET = argv[2];\n"
                "\t\tFILE_ID = argv[3];\n"
                "\t\tFILE_PLOT = argv[4];\n"
                "\t\tTARGET_SIZE = (argc >= 6) ? atoi(argv[5]) : int(0.01 * NODE_NUM);\n"
                "\t\tREMOVE_STRATEGY = (argc >= 7) ? atoi(argv[6]) : 1;\n"
                "\t}\n",
            ),
        ],
        export=_gnd_export,
        command=_gnd_command,
        parse_seeds=_gnd_parse,
        notes=(
            "The authors' spectral-partitioning + weighted-vertex-cover code, and "
            "the reference for our own `gnd` / `gndr`, which reproduce its two "
            "stages with a greedy cover instead of its LP 2-approximation. Also "
            "the SOURCE of six of our datasets (`crime`, `corruption`, "
            "`hamsterster`, `road_eu`, `intnet1`, `ppi_yeast` all load its "
            "`Datasets_*` files), so the graphs already match byte for byte. "
            "Warning: Its contribution is COST-weighted dismantling, so comparing its "
            "cost-optimal set against a cardinality budget is unfair in both "
            "directions (§8.2 trap 4) — run it with unit costs, or report a cost "
            "budget. TO WIRE: clone, make, read the flag that selects unit vs "
            "degree cost and whether reinsertion is on."
        ),
    ),
    "decycler": ExternalBaseline(
        name="decycler",
        kind=classical,
        title="Min-Sum: decycling and dismantling by message passing",
        venue="PNAS 113(44), 2016",
        repo="https://github.com/abraunst/decycler",
        paper="https://arxiv.org/abs/1603.08883",
        entry="C++ (make) + treebreaker.py",
        task="critical_node_detection",
        status="needs_setup",
        requirements=None,
        # Boost lives under /opt/homebrew on macOS and /usr on most Linux; the
        # shipped Makefile assumes the latter, so the include/lib paths are made
        # explicit and harmless when the directory does not exist.
        build=[
            "make",
            "CXX=g++",
            "FLAGS=-Wall -O3 -g -Wno-deprecated -I. -Wno-unknown-pragmas -I/opt/homebrew/include",
            "LIBS=-L/opt/homebrew/lib -lboost_program_options",
        ],
        # treebreaker.py is Python 2 (four bare `print` statements). Ported here
        # rather than requiring a py2 interpreter; `scomp/2` is made explicitly
        # integer to preserve py2 semantics exactly (equivalent for integer M,
        # but the ambiguity is not worth leaving in).
        patches=[
            (
                "treebreaker.py",
                'print "# the graph is NOT acyclic"',
                'print("# the graph is NOT acyclic")',
            ),
            (
                "treebreaker.py",
                'print "# N:", N, "Ncc:", Ncc, "M:", G.M',
                'print("# N:", N, "Ncc:", Ncc, "M:", G.M)',
            ),
            (
                "treebreaker.py",
                'print "# the graph is acyclic"',
                'print("# the graph is acyclic")',
            ),
            ("treebreaker.py", 'print "S", i, n, scomp', 'print("S", i, n, scomp)'),
            ("treebreaker.py", "if M <= scomp/2:", "if M <= scomp//2:"),
        ],
        export=_decycler_export,
        command=_decycler_command,
        parse_seeds=_decycler_parse,
        notes=(
            "The authors' 1RSB cavity Min-Sum code: near-minimal decycling set, "
            "O(N log N) tree breaking, then reverse-greedy reinsertion. Our "
            "`decycling` reproduces the SHAPE with a greedy first stage and is "
            "explicitly not this; wire it before quoting any Min-Sum comparison. "
            "Warning: §11 records that its real-network table does not exist — the PNAS "
            "paper reports two graphs in prose, and the Hamsterster/PGP/Enron rows "
            "commonly attributed to Min-Sum actually come from CoreHD and BPD."
        ),
    ),
    "collective_influence": ExternalBaseline(
        name="collective_influence",
        kind=classical,
        title="Collective Influence: optimal percolation",
        venue="Nature 524, 2015",
        repo="https://github.com/makselab/Collective-Influence",
        paper="https://arxiv.org/abs/1506.08326",
        entry="C (gcc CI_HEAP.c)",
        task="critical_node_detection",
        status="needs_setup",
        requirements=None,
        # The repo ships a single .c file and NO Makefile; the build line is the
        # one in its own source header
        build=["gcc", "-o", "CI", "CI_HEAP.c", "-lm", "-O3"],
        export=_ci_export,
        command=_ci_command,
        parse_seeds=_ci_parse,
        notes=(
            "The authors' O(N log N) CI, and the reference for our own "
            "`collective_influence_removal`, which reproduces the score and the "
            "adaptive removal but NOT the greedy reinsertion pass the paper runs "
            "afterwards (§8.2 trap 2). Cheap to wire relative to the learned "
            "entries: one C binary, no framework. Third-party alternative at "
            "zhfkt/ComplexCi if the original does not build."
        ),
    ),
    "explosive_immunization": ExternalBaseline(
        name="explosive_immunization",
        kind=classical,
        title="Explosive Immunization",
        venue="Phys. Rev. Lett. 117:208301, 2016",
        repo="https://github.com/pclus/explosive-immunization",
        paper="https://arxiv.org/abs/1604.00073",
        entry="C (make -C Library)",
        task="critical_node_detection",
        status="needs_setup",
        requirements=None,
        # The makefile lives in Library/ and writes ../exploimmun
        build=["make", "-C", "Library"],
        export=_ei_export,
        command=_ei_command,
        parse_seeds=_ei_parse,
        # exploimmun calls exit() when it reaches the percolation threshold,
        # which is its normal, successful termination and returns 1
        allow_nonzero_exit=True,
        notes=(
            "Warning: TWO TRAPS, both verified by running it — read _ei_parse before "
            "reporting a number. (1) Its output flag is INVERTED relative to its "
            "README: 0 means vaccinated, not 1, and reading it the documented way "
            "leaves a giant component of 626 against 18. (2) It emits a SET at its "
            "own percolation threshold, not a removal order, so choosing k from it "
            "is our degree tie-break rather than EI's answer. "
            "The authors' inverse/Achlioptas construction, and the reference for "
            "our `explosive_immunization`, which uses one regime rather than their "
            "two. Worth wiring specifically for §9.4: MIND's Table 5 puts EI at "
            "80.6 on `eu-powergrid` and 23.4 on `roads-california` where FINDER "
            "scores 161.7 and 116.3 — the physics heuristic beating the RL method "
            "by up to 5x on mesh graphs is the limitation we predict and should "
            "confirm rather than discover."
        ),
    ),
    "dismantling_review": ExternalBaseline(
        name="dismantling_review",
        kind=classical,
        title="NetworkDismantling: the Artime et al. runnable baseline suite",
        venue="Nature Reviews Physics 6, 2024",
        repo="https://github.com/NetworkDismantling/review",
        paper="https://arxiv.org/abs/2509.19867",
        entry="Python + C++ (see repo)",
        task="critical_node_detection",
        status="needs_setup",
        export=_review_export,
        command=_review_command,
        parse_seeds=_order_parse,
        notes=(
            "Not one method — the survey's harness, which already wires CI, "
            "CoreHD, GND, EI, MinSum, FINDER and GDM behind one interface. Wiring "
            "THIS instead of the seven repos above is almost certainly the right "
            "trade (§9.2 item 5), and it is also the only practical route to a "
            "FINDER number given §11's note that FINDER publishes no table. TO "
            "WIRE: clone, follow its own install (it builds several C++ "
            "dependencies), and read which of its drivers emits a removal ORDER "
            "rather than a curve."
        ),
    ),
    "selinda": ExternalBaseline(
        name="selinda",
        kind=learned,
        title="Selinda: symbolized RL for network resilience",
        venue="preprint 2025",
        repo="https://github.com/tsinghua-fib-lab/selinda",
        paper="https://arxiv.org/abs/2507.08827",
        entry="PyTorch",
        task="critical_node_detection",
        status="needs_setup",
        export=_selinda_export,
        command=_runner_command,
        parse_seeds=_order_parse,
        notes=(
            "Learns an RL attack policy and then SYMBOLIC-REGRESSES it into a "
            "closed-form resilience law coupling topology and dynamics. Registered "
            "less as a baseline than as the closest published relative of the "
            "coding-agent framing: its output IS a formula, which is what our "
            "generated `score()` is. Its numbers are not directly comparable — it "
            "reports a fitted law's accuracy, not a dismantling set size."
        ),
    ),
    # SOURCE LOCALIZATION (research/source_localization.md §4, §10).
    # Registered, none wired. The contract here differs from every other task's in
    # a way that matters before any adapter is written: these methods do not
    # return a seed set from a graph, they return a SOURCE set from a (graph,
    # observation) pair, so the process boundary has to carry `y` across as well.
    # Our own referee then scores it exactly as it scores our arms.
    # Warning: Two shared traps (§8.4): the SOURCE FRACTION is not standardized (10%
    # uniform-random in SL-VAE, first 5% by infection time in SL-Diff, top 10% by
    # influence time in SIDSL), and NOTHING in this literature evaluates under IC
    # or LT — SL-VAE uses SI/SIR, everything else uses real cascades. §11 calls
    # that the single biggest comparability gap in the file, and it is bigger than
    # any graph-version disagreement.
    # Six published methods behind one package, one arm each, one shared install.
    # This is the entry that makes SL-VAE, IVGD and GCNSI runnable at all: the
    # standalone repos for the first two are worse packaging of the same methods
    # (see their entries below).
    "graphsl_lpsi": _graphsl_entry(
        "lpsi",
        "LPSI: multiple source detection without the propagation model",
        "AAAI 2017",
        "The row that has to be beaten. SIDSL's Table 1 puts this 2017 "
        "label-propagation method at F1 0.544 on Digg against SL-VAE's 0.479 and "
        "DDMSL's 0.517, with no learning in it at all. Cross-checks our own "
        "`lpsi` reimplementation, which differs in the selection rule (we take "
        "local maxima of the field, GraphSL thresholds it).",
    ),
    "graphsl_netsleuth": _graphsl_entry(
        "netsleuth",
        "NETSLEUTH: identifying culprits in epidemics (MDL)",
        "ICDM 2012",
        "The multi-source MDL reference. GraphSL's version tunes `k` on the "
        "training split, so unlike our own `netsleuth` it does exercise the "
        "count-inference half of the paper, then we take its top-k for a matched "
        "comparison. Beats LPSI on Power Grid in SL-VAE's Table 1.",
    ),
    "graphsl_ojc": _graphsl_entry(
        "ojc",
        "OJC: locating multiple sources with partial observations",
        "AAAI 2017",
        "Built for the sparse-observer regime, run here under full observation "
        "because that is what our episodes record, so this row understates it by "
        "construction rather than by accident.",
    ),
    "graphsl_gcnsi": _graphsl_entry(
        "gcnsi",
        "GCNSI: multiple rumor source detection with graph convolutional networks",
        "CIKM 2019",
        "The first GNN for this task and every later paper's WEAKEST learned "
        "baseline: lowest-scoring learned method in §5.1, §5.2, §5.3 and §5.5 "
        "without exception. Its `ACC 0.8840` with `F1 0.0218` on Network Science "
        "is the single clearest demonstration that accuracy is useless alone "
        "here. Trained at `graphsl_epochs`, which is far below the paper's.",
    ),
    "graphsl_ivgd": _graphsl_entry(
        "ivgd",
        "IVGD: invertible validity-aware graph diffusion",
        "WWW 2022",
        "The near-miss on the amortization axis: its inversion is one backward "
        "pass, but the validity-aware projection layers are an unrolled "
        "per-instance optimization, so inference still scales with an inner loop. "
        "Its published recall is 1.0000 on five of six graphs, which means the "
        "projection is tuned to over-predict and let precision carry F1 — worth "
        "knowing before treating its FS ~ 0.97 as a ceiling.",
    ),
    "graphsl_slvae": _graphsl_entry(
        "slvae",
        "SL-VAE: source localization with a learned generative prior",
        "KDD 2022",
        "The seed paper, and the method arm A reimplements against OUR likelihood. "
        "Running the original matters precisely because §2.2's argument is that "
        "swapping the forward model is a no-op the authors already published — "
        "the way to show that rather than assert it is to run both. Its inference "
        "is itself a per-instance gradient loop, which is what puts it in the "
        "non-amortized row of §1's table and what the cost columns should show.",
    ),
    "graphsl": ExternalBaseline(
        name="graphsl",
        kind=classical,
        title="GraphSL: a library for graph source localization",
        venue="JOSS 9(99):6796, 2024",
        repo="https://github.com/xianggebenben/GraphSL",
        paper="https://arxiv.org/abs/2405.03724",
        entry="pip install GraphSL",
        task="source_localization",
        status="blocked",
        fetch="git",
        blocker=(
            "the umbrella entry is not an arm — it ships SIX methods, and one arm "
            "returning one of them would hide which. Use graphsl_lpsi, "
            "graphsl_netsleuth, graphsl_ojc, graphsl_gcnsi, graphsl_ivgd or "
            "graphsl_slvae, all of which share this clone and venv."
        ),
        notes=(
            "THE one to wire first, and by a wide margin. It ships LPSI, NETSLEUTH, "
            "OJC, GCNSI, IVGD and SL-VAE behind ONE API returning accuracy / "
            "precision / recall / F1 / AUC, so a single adapter buys six published "
            "methods including two of the three seed papers. It also packages the "
            "six benchmark graphs (Karate, Dolphins, Jazz, Network Science, "
            "Cora-ML, Power Grid) — five of which we now load — and it packages "
            "OUR version of Network Science (1,589 / 2,742), which §6.4.1 shows is "
            "the version IVGD, SIDSL and Network Repository agree on and SL-VAE "
            "does not. Written by IVGD's first author. TO WIRE: `pip install "
            "GraphSL` into a per-baseline venv, then serialize our adjacency AND "
            "the observation vector across the process boundary and read the "
            "source set back. Warning: An adapter alone is NOT enough: "
            "`run_baseline.run_external_baseline` takes no observation argument, "
            "`seed_script` emits `plan_horizon` rather than `localize`, and the "
            "pipeline invokes an external repo once per arm rather than once per "
            "labelled episode. All three are harness changes and all three are "
            "listed in baselines/README.md. Its own `Metric` object is a "
            "cross-check on our PR/RE/F1/AUC, not a replacement: only our referee "
            "makes arms comparable."
        ),
    ),
    "slvae": ExternalBaseline(
        name="slvae",
        kind=learned,
        title="SL-VAE: source localization with a learned generative prior",
        venue="KDD 2022",
        repo="https://github.com/triplej0079/SLVAE",
        paper="https://arxiv.org/abs/2206.12327",
        entry="PyTorch",
        task="source_localization",
        status="blocked",
        blocker=(
            "superseded by `graphsl_slvae`, which is WIRED and runs the same "
            "method. This standalone repo is strictly worse packaging: slvae.py "
            "prints P/R/F1/AUC and writes no artifact to parse, and its "
            "load_dataset looks for `<name>_25c.SG` while the shipped files carry "
            "no `_25c` suffix. Wiring it means patching the repo to dump "
            "predictions, for a second copy of a number graphsl_slvae already "
            "produces. Unblock it only if a discrepancy against graphsl_slvae "
            "needs adjudicating."
        ),
        notes=(
            "The seed paper, and the method our arm A reimplements against our own "
            "likelihood. Worth wiring the ORIGINAL anyway: §2.2's whole argument is "
            "that swapping the forward model is a no-op the paper already "
            "published, and the way to show that rather than assert it is to run "
            "both. Its Table 1/2 are the (key) comparable tables (§5.1) — five of its "
            "seven graphs are ours — but read §5.1's column-order warning first: "
            "Table 1 is RE·PR·F1·AUC and Table 2 is PR·RE·F1·AUC, and a summarizer "
            "that assumes one ordering transposes precision and recall for a whole "
            "table. Warning: Its Network Science is 1,565 / 13,532, which is NOT ours and "
            "is not publicly downloadable (§6.4.1), so that row is not comparable "
            "in either direction. TO WIRE: clone, read which diffusion estimator "
            "the entry point defaults to (it tries GAT / MONSTOR / DeepIS) and how "
            "its observation tensor is laid out."
        ),
    ),
    "ivgd": ExternalBaseline(
        name="ivgd",
        kind=learned,
        title="IVGD: invertible validity-aware graph diffusion",
        venue="WWW 2022",
        repo="https://github.com/xianggebenben/IVGD",
        paper="https://arxiv.org/abs/2206.09214",
        entry="PyTorch (pretrain.py then main.py)",
        task="source_localization",
        status="blocked",
        blocker=(
            "superseded by `graphsl_ivgd`, which is WIRED and runs the same "
            "method (GraphSL is written by this repo's first author). The "
            "standalone is not drivable as shipped: main.py sets "
            "`dataset = 'karate'` at module scope with no argparse, and the clone "
            "carries a committed virtualenv (Lib/, Scripts/, pyvenv.cfg) that "
            "collides with ours. Wiring it means rewriting its entry point for a "
            "second copy of a number graphsl_ivgd already produces."
        ),
        notes=(
            "The near-miss on §1's amortization axis: its inversion is a single "
            "backward pass, but the validity-aware projection layers are an "
            "unrolled per-instance optimization, so its inference cost still scales "
            "with an inner loop. `pretrain.py` exists purely so `main.py` has "
            "something to invert, which is the cleanest statement in the literature "
            "that the forward model is a component. Warning: Its Table 3 numbers are far "
            "higher than SL-VAE's on the identical graph NAMES and the two are not "
            "comparable (§5.2): different protocol, different unstated seed "
            "fraction, and a different Network Science. Its recall is 1.0000 on "
            "five of six graphs, which means the projection is tuned to "
            "over-predict and let precision carry F1 — worth knowing before "
            "treating FS ~ 0.97 as a ceiling. Also reachable through `graphsl`, "
            "which is the cheaper route."
        ),
    ),
    "cnsl": ExternalBaseline(
        name="cnsl",
        kind=learned,
        title="CNSL: cross-network source localization",
        venue="preprint 2024 (Emory)",
        repo="https://github.com/tanmoysr/CNSL",
        paper="https://arxiv.org/abs/2404.14668",
        entry="PyTorch",
        task="source_localization",
        status="blocked",
        blocker=(
            "the CONTRACT does not fit. CNSL localizes across a PAIR of coupled "
            "networks, with the diffusion originating in one that is never "
            "observed; our instance is one graph and one observation over it. "
            "There is no honest way to hand it our episodes, and an adapter that "
            "fed it the same graph twice would be measuring something else. It "
            "stays registered because it is the closest published relative of our "
            "transfer experiment, not because it is runnable here."
        ),
        notes=(
            "Diffusion crosses from an UNOBSERVED source network into an observed "
            "one, with network-specific propagation learned jointly. Registered as "
            "the closest published relative of our transfer experiment (§8.5.1): it "
            "is the only entry here that evaluates a method on a graph other than "
            "the one it was fit to, though its setting is genuinely different — two "
            "coupled networks in one instance, rather than one artifact reused "
            "across instances. Same lab as SL-VAE and DeepIM."
        ),
    ),
    "pdsl": ExternalBaseline(
        name="pdsl",
        kind=learned,
        title="PDSL: propagation-dynamics-aware source localization",
        venue="arXiv 2026 (TNSE format)",
        repo="https://github.com/MrYansong/PDSL",
        paper="https://arxiv.org/abs/2605.03550",
        entry="PyTorch",
        task="source_localization",
        status="blocked",
        blocker=(
            "its published inference SELECTS ON THE GROUND TRUTH, so it cannot be "
            "run honestly as shipped. `utiles.x_hat_initialization` scores every "
            "optimizer iterate against the true source vector `x` "
            "(`f1_score(x[0], x_pred[0])`, line 36) and appends it to a list; "
            "`PDSL.py` then takes `initial_x[initial_x_prec.index(max(...))]`, "
            "i.e. the iterate closest to the answer. Verified by reading the repo "
            "at HEAD on 2026-08-03, not inferred from the paper. Under our "
            "contract a label may never reach an evaluation-split prediction "
            "(research/source_localization.md §2.3.3), so wiring it as-is would "
            "put an oracle-selected row in the same column as label-free ones. "
            "TO WIRE HONESTLY: replace that selection with a label-free rule (its "
            "own `loss`, which is what it already computes and optimizes), report "
            "the result as a MODIFIED method, and report the as-shipped number "
            "beside it so the size of the leak is visible. That is a worthwhile "
            "experiment and it is not this baseline."
        ),
        notes=(
            "The closest published framing to ours: it infers "
            "`argmax p_psi(Y_T | s*, Y_t, G) * p_phi(s* | z, Y_t, G) * p(z)`, which "
            "is a world-model-shaped factorization with an explicit forward term. "
            "Still per-instance — it optimizes s* rather than producing a reusable "
            "artifact — so it sits in §1's occupied cell, not the empty one. Worth "
            "wiring precisely because a referee will ask how we differ from it."
        ),
    ),
    "gnn_source_detection": ExternalBaseline(
        name="gnn_source_detection",
        kind=learned,
        title="GNN source detection: a review and benchmark study",
        venue="arXiv 2026 (v2)",
        repo="https://github.com/martinSter/gnn-source-detection",
        paper="https://arxiv.org/abs/2512.20657",
        entry="PyTorch",
        task="source_localization",
        status="blocked",
        blocker=(
            "the METRIC does not fit. It is single-source, scored by top-k "
            "accuracy under SIR; our table is multi-source F1 at matched k. "
            "research/source_localization.md §8.2 states the two literatures' "
            "numbers never mix, so a wired adapter would emit a value that looks "
            "comparable in the spread column and is not. Run it as its own study "
            "against a --budgets 1 sweep if the single-source question is wanted; "
            "do not put it in the same table."
        ),
        notes=(
            "Not a method — the only reproducible THIRD-PARTY evaluation in this "
            "literature, and the only one authored by none of the method groups. "
            "SIR, SINGLE-source, top-k accuracy on six contact networks against "
            "Jordan centre, betweenness, SME and MCMF. Warning: Its framing is not ours: "
            "single-source ranking is a different problem from multi-source "
            "classification and the two never mix (§8.2), so its numbers cannot "
            "join our table without a dedicated `--budgets 1` run. Its value is the "
            "CEILING it establishes — the best GNN reaches 72.9% top-5 on a "
            "34-node graph where random already gets 39.4% — which is the honest "
            "picture of how hard this problem is, against §5.2's near-perfect F1. "
            "Its Karate is 34 / 77, one edge fewer than `nx.karate_club_graph()`."
        ),
    ),
    # Four estimators, one arm each, one shared install. Their value is a
    # CROSS-CHECK: `lpsi`, `netsleuth`, `jordan_center` and `rumor_centrality` in
    # our own pool are reimplementations from the papers' prose, and rumor
    # centrality has no first-party code anywhere.
    "cosasi_jordan": _cosasi_entry(
        "jordan",
        "Jordan centrality (multi-source), cosasi implementation",
        "JOSS 2022",
        "Cross-checks our `jordan_center`. cosasi partitions the infected "
        "subgraph by spectral clustering and takes a centre per part; we take one "
        "centre per connected COMPONENT and then rank by eccentricity, so a large "
        "gap between the two rows is a difference in the multi-source "
        "generalization rather than in the underlying estimator.",
    ),
    "cosasi_netsleuth": _cosasi_entry(
        "netsleuth",
        "NETSLEUTH (multi-source), cosasi implementation",
        "JOSS 2022",
        "The third independent NETSLEUTH in the sweep, alongside our own and "
        "GraphSL's. Three implementations of one 2012 paper is unusual and worth "
        "using: they differ in how `k` is fixed, so their spread bounds how much "
        "of a NETSLEUTH number is the method and how much is the harness.",
    ),
    "cosasi_lisn": _cosasi_entry(
        "lisn",
        "LISN: infection-time likelihood source inference",
        "JOSS 2022",
        "The one estimator in this file we do NOT have our own version of, so it "
        "is coverage rather than a cross-check. It conditions on elapsed time, "
        "which our episodes record exactly (`SourceInstance.horizon`) — most "
        "published evaluations have to assume it.",
    ),
    "cosasi_rumor_centrality": _cosasi_entry(
        "rumor_centrality",
        "Rumor centrality (Shah & Zaman), cosasi implementation",
        "JOSS 2022",
        "The highest-value cross-check in the file. Shah & Zaman published no "
        "code, so our `rumor_centrality` is reimplemented from the paper's prose "
        "and cosasi's is an independent reading of the same text. Warning: "
        "SINGLE-SOURCE: under a multi-source protocol at k = 10% of N it scores "
        "near zero by construction (§8.2), so compare it at `--budgets 1`.",
    ),
    # Cascade reconstruction (research/cascade_reconstruction.md §3, §4).
    # Three arms, one install: DITTO's clone is the only public code in the group
    # of §4.2 (the methods that invert a LEARNED forward operator) and it ships the
    # paper's two MLE baselines beside it.
    "ditto": _ditto_entry(
        "ditto",
        "ditto.py",
        "main",
        "DITTO: reconstructing graph diffusion history from a single snapshot",
        "KDD 2023",
        "(key) THE reference point for this task, and the method arm A reimplements "
        "against our own kernel. §2.3's whole argument is that swapping the forward "
        "model is a second-order result inside DITTO's frame, and the way to show "
        "that rather than assert it is to run both. Its Tables 4-5 are the "
        "comparable ones (§5.1) and TWO of its four synthetic rows are graphs we "
        "generate natively (`--dataset ba --ba-m 4` and `--dataset er --er-p 0.008` "
        "at n=1,000). Warning: DITTO always solves the DASH (final-snapshot) problem "
        "whatever --cr-setting the sweep runs, because `ditto.py` conditions on "
        "`data.y[:, -1]` and nothing else — verified by reading every `data.*` "
        "access. Under --cr-setting partial_times it is therefore answering a "
        "strictly harder instance than every other arm and its row must say so. It "
        "outputs per-step node STATES and no propagation tree, so its Path "
        "Precision is measured on a tree derived by our shared `finalize` rule; "
        "§5.8 records that DIPT is the only method in this literature that emits "
        "edges, and DIPT has no public code. §2.11 risk 6 is why this arm matters: "
        "DITTO beats a supervised model trained with the TRUE beta on BA-SIR and "
        "ER-SIR, so it is genuinely hard to beat and a weak version of it makes "
        "6-vs-A meaningless.",
    ),
    "ditto_dhrec": _ditto_entry(
        "ditto_dhrec",
        "dhrec.py",
        "pcdsvc_run",
        "DHREC-PCDSVC, as implemented by DITTO's authors",
        "ICDM 2014 / KAIS 2016",
        "A CROSS-CHECK on our own `dhrec`, not new coverage. Sefer & Kingsford's "
        "original code is specially for SEIRS, so DITTO's authors reimplemented "
        "PCDSVC for SI and SIR — which means this arm and ours are two independent "
        "readings of one paper's prose, and their spread bounds how much of a DHREC "
        "number is the method and how much is the harness. DITTO's Table 5 puts it "
        "at F1 .50-.66, i.e. 10-35% behind the supervised ideal [verified, §5.1], "
        "which is the bar an interesting result has to clear rather than the one it "
        "has to beat.",
    ),
    "ditto_cri": _ditto_entry(
        "ditto_cri",
        "cri.py",
        "cri_run",
        "CRI (clustering + reverse infection), as implemented by DITTO's authors",
        "TNSE 2016",
        "The second cross-check, and the more valuable of the two: the CRI paper "
        "published NO source code at all, so our `cri` and this one are both "
        "reimplementations from prose and neither is authoritative. DITTO's Tables "
        "4-5 report it at F1 .42-.82 [verified]. Same caveat as its sibling — it "
        "conditions on the final snapshot only.",
    ),
    "reconstructing_cascade": ExternalBaseline(
        name="reconstructing_cascade",
        kind=classical,
        title="Reconstructing a cascade from temporal observations (OrderedSteinerTree)",
        venue="SDM 2018",
        repo="https://github.com/xiaohan2012/reconstructing-cascade",
        paper="https://arxiv.org/abs/1801.08586",
        entry="paper_experiment.py",
        task="cascade_reconstruction",
        status="blocked",
        blocker=(
            "hard dependency on `graph_tool`, which is NOT pip-installable — it is "
            "a Boost/C++ extension distributed through conda, apt or a source "
            "build, so it cannot go into a per-baseline venv the way every other "
            "entry here does. Verified by reading the repo at HEAD: `steiner_tree.py` "
            "opens with `from graph_tool import Graph, GraphView` and `gt_utils.py`, "
            "`core.py`, `tbfs.py` and `utils.py` all do the same. All four of its "
            "methods are reimplemented in `coding_agent/tools/"
            "reconstruction_algorithms.py` (`delayed_bfs`, `ordered_steiner_closure`, "
            "`greedy_ordered`, `steiner_tree`), so the coverage is not lost — only "
            "the authors' own code is. TO UNBLOCK: a conda-based install path for "
            "one baseline, which the setup script does not have and which would be "
            "the first of its kind here."
        ),
        notes=(
            "The source of `delayed_bfs`, the row that actually has to be beaten. "
            "Its `closure` gives O(sqrt(k)) and `delayed-bfs` a k-approximation in "
            "O(m + k log k) [verified, §5.6]. Warning: it publishes ZERO result "
            "tables — everything in §5.6 is [figure] — so running it would produce "
            "the per-cell baselines this literature does not have, which is exactly "
            "why the blocker above is worth revisiting. Three of its four graphs "
            "(email-Eu-core, ca-GrQc, facebook) are ones we load."
        ),
    ),
    "cascade_tree_samples": ExternalBaseline(
        name="cascade_tree_samples",
        kind=classical,
        title="Robust cascade reconstruction by Steiner tree sampling",
        venue="ICDM 2018",
        repo="https://github.com/xiaohan2012/cascade-reconstruction-by-tree-samples",
        paper="https://arxiv.org/abs/1809.05812",
        entry="inference.py",
        task="cascade_reconstruction",
        status="blocked",
        blocker=(
            "the same `graph_tool` blocker as `reconstructing_cascade`, PLUS a "
            "second one: it needs the author's separate Cython package "
            "`xiaohan2012/random_steiner_tree` (loop-erased random walk / "
            "cycle-popping), which is not on PyPI and builds against graph_tool's "
            "own headers. `tree_sampling` in our pool implements the method with "
            "randomized-cost shortest-path trees instead of cycle-popping, which is "
            "a documented approximation of the sampler rather than of the method."
        ),
        notes=(
            "The only classical method in §3 that outputs calibrated per-node "
            "PROBABILITIES rather than a binary set. It is also the source of §8.2 "
            "trap 4, the most specific warning in that file: on `grqc` "
            "(assortativity 0.164) a Personalized PageRank baseline BEATS this "
            "method, and loses elsewhere [verified]. We load that graph as "
            "`ca_grqc`, `personalized_pagerank` is in the default pool for exactly "
            "this reason, and the comparison is runnable today without this repo."
        ),
    ),
    "cult": ExternalBaseline(
        name="cult",
        kind=classical,
        title="CulT: reconstructing an epidemic over time",
        venue="KDD 2016",
        repo="https://github.com/polinapolina/reconstructing-an-epidemic-over-time",
        paper="https://www.kdd.org/kdd2016/papers/files/rpp0920-rozenshteinAT3.pdf",
        entry="experiments/demo.py",
        task="cascade_reconstruction",
        status="blocked",
        blocker=(
            "PYTHON 2, and a different input object. Verified by reading "
            "`experiments/demo.py` at HEAD: it uses py2 print STATEMENTS "
            "(`print len(TS), ...`) throughout, as does every module under "
            "`experiments/utils/`, so it does not parse under Python 3 at all. The "
            "deeper problem is the contract rather than the syntax: CulT consumes a "
            "temporal INTERACTION STREAM `TS` (§5.5 — 'works on interaction "
            "streams, not a static G') and our episodes are diffusion states on a "
            "static graph, so `readFile(..., mode='general')` has nothing to read. "
            "TO WIRE: a 2to3 pass over `experiments/` plus a synthetic interaction "
            "stream built from our per-step frontiers, which would be a different "
            "experiment rather than this one. `cult` in our own pool implements the "
            "alpha-TempSteinerTree objective on the static graph instead."
        ),
        notes=(
            "The honest ceiling for 'what can you do without a kernel' (§5.5): the "
            "only method in §3 that assumes NO propagation model at all, which "
            "makes it the right thing for a learned kernel to beat. Warning: it "
            "publishes zero tables — `grep -c \"Table\"` on the KDD'16 text returns "
            "0 [verified] — so its MCC 0.60-0.90 is [figure] and no per-cell "
            "comparison exists in either direction."
        ),
    ),
    "active_cascade_reconstruction": ExternalBaseline(
        name="active_cascade_reconstruction",
        kind=classical,
        title="Active cascade reconstruction (query selection)",
        venue="Xiao et al., follow-up to ICDM 2018",
        repo="https://github.com/xiaohan2012/active-cascade-reconstruction",
        paper="https://arxiv.org/abs/1809.05812",
        entry="query_selection.py",
        task="cascade_reconstruction",
        status="blocked",
        blocker=(
            "the SETTING is one we deliberately do not run. Active reconstruction "
            "lets the decoder CHOOSE which nodes to query, which turns the "
            "observation mask into an action — §2.11 risk 7 records that as the fix "
            "for this task exercising none of the five action ops, and recommends "
            "deferring it for the same reason source localization defers its "
            "analogue. Wiring it before we run the passive setting would put an "
            "arm with a different information budget in the same column. It also "
            "inherits both of `cascade_tree_samples`' dependency blockers."
        ),
        notes=(
            "The one entry here that is registered for a FUTURE experiment rather "
            "than a current one. If the active variant is ever run, this is the "
            "published comparison for it and the query-selection strategies are the "
            "baselines."
        ),
    ),
    "deep_demixing": ExternalBaseline(
        name="deep_demixing",
        kind=learned,
        title="DDMIX / Deep Demixing: a conditional-VAE GNN over aggregated snapshots",
        venue="EUSIPCO 2021 / TSIPN 2023",
        repo="https://github.com/gojkoc54/Deep_demixing",
        paper="https://arxiv.org/abs/2306.07938",
        entry="models.py (CVAE_UNET_Batch)",
        task="cascade_reconstruction",
        status="needs_setup",
        # Its environment.yml is a conda spec pinned to 2021 and does not resolve
        # through pip; the model itself needs only torch + torch-geometric, whose
        # GraphUNet and GCNConv are what `models.py` imports.
        # Warning: `CVAE_UNET_Batch` is built from torch_geometric's `GraphUNet`,
        # whose `augment_adj` does a sparse-CSR @ sparse-CSR matmul that a torch
        # built WITHOUT MKL cannot do on CPU — the stock Apple Silicon wheel is the
        # common case, and the same four lines fail with GraphUNet alone. It runs
        # on Linux (whose torch ships with MKL) and on CUDA; the driver catches
        # that specific RuntimeError and says so rather than surfacing a traceback
        # that points into torch_geometric and reads like a wiring bug.
        requirements=None,
        pip_packages=("torch", "torch-geometric", "networkx", "numpy", "scipy"),
        export=partial(
            _reconstruction_export,
            driver="deep_demixing_driver.py",
            supervised=True,
        ),
        command=_reconstruction_command,
        parse_seeds=_reconstruction_parse,
        extra_env={},
        notes=(
            "The learned method closest to our own output shape: it demixes ONE "
            "aggregated snapshot into node states at ALL T steps, which is exactly "
            "the object `reconstruct()` returns. DIPT reports it at Path Precision "
            "0.062-0.327 and Jaccard 0.031-0.195 across five graphs — the WEAKEST "
            "row of §5.2's table [verified] — and its own paper says accuracy "
            "degrades as T grows because the solution space blows up, so a low "
            "number here is the expected outcome rather than a wiring failure. "
            "Warning: SUPERVISED. It fits on the selection split's labelled "
            "histories, which no other arm on the table does, so its row is not "
            "comparable to an unsupervised decoder's without saying so. Driven "
            "through `CVAE_UNET_Batch` directly rather than through "
            "`scripts/exp0-minimal.py`, which reads the repo's own pickle layout "
            "and trains four models at once: the authors' MODEL, our optimizer. "
            "At prediction time the driver passes ZEROS for `y` — the model takes "
            "its prior path at eval and never uses the posterior, so the output is "
            "unchanged and an evaluation row's history never enters the forward "
            "pass at all."
        ),
    ),
    "grin": ExternalBaseline(
        name="grin",
        kind=learned,
        title="GRIN: graph recurrent imputation network",
        venue="ICLR 2022",
        repo="https://github.com/Graph-Machine-Learning-Group/grin",
        paper="https://arxiv.org/abs/2108.00298",
        entry="lib/nn/models/grin.py (GRINet)",
        task="cascade_reconstruction",
        status="needs_setup",
        # Its requirements.txt pins tensorflow==2.5.0, tensorflow-gpu==2.4.0,
        # pytorch-lightning==1.4 and torch==1.8, none of which resolves on a
        # current Python and none of which GRINet needs: tracing every import in
        # lib/nn/layers/{rits,gril,gcrnn,spatial_conv,spatial_attention}.py gives
        # torch + einops and nothing else.
        requirements=None,
        pip_packages=("torch", "einops", "numpy"),
        export=partial(
            _reconstruction_export, driver="grin_driver.py", supervised=True
        ),
        command=_reconstruction_command,
        parse_seeds=_reconstruction_parse,
        extra_env={},
        notes=(
            "(key) DITTO's STRONGEST supervised baseline, and the one it uses as the "
            "IDEAL upper bound when trained with the true beta [verified, §5.1] — "
            "every `Gap` column in its Tables 4-5 is measured against this row, "
            "which makes it the single most useful reference number in this "
            "literature. Warning: SUPERVISED, and that is the caveat its row must "
            "carry: it fits on the selection split's labelled histories while every "
            "other arm on the table never sees one. §5.1.2 is why that matters "
            "concretely rather than pedantically — the supervised family collapses "
            "from F1 ~ 0.80 on simulated diffusion to F1 ~ 0.32 on real, which is "
            "§2.11 risk 4 in one table. Driven through `GRINet` directly rather "
            "than through `scripts/run_imputation.py`, a Lightning experiment over "
            "four hardcoded traffic and air-quality datasets: the authors' MODEL, "
            "our optimizer, so GWM_IMPUTE_EPOCHS is the first number to raise "
            "before quoting it as parity. `impute_only_holes` is left ON, its own "
            "default, so an observed entry is pinned back exactly as our shared "
            "decoder does."
        ),
    ),
    "spin": ExternalBaseline(
        name="spin",
        kind=learned,
        title="SPIN: sparse spatiotemporal attention for imputation",
        venue="NeurIPS 2022",
        repo="https://github.com/Graph-Machine-Learning-Group/spin",
        paper="https://arxiv.org/abs/2205.13479",
        entry="spin/models/spin.py (SPINModel)",
        task="cascade_reconstruction",
        status="needs_setup",
        # Its conda_env.yml wants conda channels; the model needs torch,
        # torch-geometric and three tsl.nn pieces (StaticGraphEmbedding, MLP, and
        # the repo's own spin/layers/), and `torch-spatiotemporal` is on PyPI.
        #
        # Warning: `torch-scatter` is NOT optional here even though nothing in SPIN
        # imports it directly — `tsl.nn.functional` does, at module load, so
        # `from spin.models import SPINModel` dies with ModuleNotFoundError without
        # it. It has no universal wheel and builds from source against the
        # installed torch, which is the slow half of this install and the one that
        # fails first on a machine with no compiler. Same hazard `ditto` carries.
        requirements=None,
        pip_packages=(
            "torch",
            "torch-geometric",
            "torch-scatter",
            "torch-spatiotemporal",
            "einops",
            "numpy",
        ),
        export=partial(
            _reconstruction_export, driver="spin_driver.py", supervised=True
        ),
        command=_reconstruction_command,
        parse_seeds=_reconstruction_parse,
        extra_env={},
        notes=(
            "GRIN's attention-based successor, and the one method in §5.1.2 that "
            "BEATS DITTO outright on a real-diffusion row (BrFarmers, F1 .8268 "
            "against .8206) while running OUT OF MEMORY on Oregon2, Prost and Pol "
            "[verified]. That OOM pattern is the useful half: it is the clearest "
            "published statement of the scale ceiling on attention-based "
            "reconstruction, and our own graphs sit on both sides of it — `jazz` "
            "and `infectious` well below, `oregon2` and `rt_pol` at or above. **An "
            "OOM here is a REPORTABLE RESULT**, not a failed arm; the driver "
            "catches it and exits with the reason, which `run_baseline` records as "
            "a skip. Same SUPERVISED caveat and same model-not-recipe wiring as "
            "`grin`. Its `u` argument takes time-of-day encodings in the authors' "
            "own experiments; a cascade has no calendar, so the driver passes the "
            "normalized step index, which is the same information in the form its "
            "positional encoder expects."
        ),
    ),
    "brits": ExternalBaseline(
        name="brits",
        kind=learned,
        title="BRITS: bidirectional recurrent imputation for time series",
        venue="NeurIPS 2018",
        repo="https://github.com/caow13/BRITS",
        paper="https://arxiv.org/abs/1805.10572",
        entry="PyTorch",
        task="cascade_reconstruction",
        status="blocked",
        blocker=(
            "GRAPH-AGNOSTIC, which makes it the wrong control here. It imputes a "
            "multivariate time series with no adjacency at all, so it cannot use "
            "the one input this whole task is conditioned on and its row would "
            "measure the difficulty of the imputation rather than of the "
            "reconstruction. DITTO includes it as the weakest supervised baseline "
            "and it OOMs on Pol [verified, §5.1.2]. Registered because it is in "
            "that table, not because it belongs in ours; run it only if a "
            "'does the graph help at all' ablation is wanted, and label it as one."
        ),
        notes=(
            "The floor of DITTO's supervised group: F1 .31-.52 on real diffusion "
            "against GRIN's .54-.80 [verified]. Its value here is as the published "
            "statement of how much the GRAPH is worth, which is a question our own "
            "arm 3 (@native, no kernel) answers differently and better."
        ),
    ),
    "dipt": ExternalBaseline(
        name="dipt",
        kind=learned,
        title="DIPT: deep identification of propagation trees",
        venue="preprint 2025 (Emory)",
        repo="https://arxiv.org/abs/2503.00646",
        paper="https://arxiv.org/abs/2503.00646",
        entry="none published",
        task="cascade_reconstruction",
        status="blocked",
        blocker=(
            "NO PUBLIC CODE. §4.1 records that the anonymous 4open.science link in "
            "the paper is the compartmental-disease SIMULATOR, not the model, and "
            "no other release was found. Registered rather than omitted because it "
            "is the only method in this literature that outputs explicit "
            "who-infected-whom EDGES — i.e. the only published Path Precision "
            "comparison that exists — and because §9 item 12 notes it is an Emory "
            "paper (Memon, Ling, Kong, Seshagiri, Zufle, Liang Zhao) using two of "
            "our graphs, which makes it a collaboration surface rather than only a "
            "citation."
        ),
        notes=(
            "(key) The ONLY published propagation-TREE table (§5.2): Path Precision "
            "0.421-0.680 and Jaccard 0.266-0.515 across five graphs, against DDMSL "
            "at 0.119-0.412 and DDMIX at 0.062-0.327 [verified, Table 1]. Two of "
            "its five graphs are ours (`cora_ml`, `power_grid`) and its protocol is "
            "reproducible from flags we already have — 10% of nodes as sources, SI "
            "to convergence. Its Table 3 is also the argument for why "
            "`--trace-parents` was worth a day of simulator work: 30% tree "
            "supervision alone is worth +0.079 Path Precision over none."
        ),
    ),
    "netrate": ExternalBaseline(
        name="netrate",
        kind=classical,
        title="NETRATE: uncovering the temporal dynamics of diffusion networks",
        venue="ICML 2011",
        repo="https://github.com/Networks-Learning/netrate",
        paper="https://arxiv.org/abs/1105.0697",
        entry="MATLAB + CVX",
        task="cascade_reconstruction",
        status="blocked",
        blocker=(
            "MATLAB and CVX, neither of which this harness can drive, and the wrong "
            "TASK besides: NETRATE recovers the transmission RATES of a latent "
            "network from many cascades, which is network inference "
            "(research/network_inference.md), not trajectory decoding. It is "
            "registered here only because Farajtabar's 'Back to the Past' fits it "
            "FIRST and then inverts the result, so it is a component of a §3 method "
            "rather than a method of its own."
        ),
        notes=(
            "Supplies the `p(y | x, G)` that Farajtabar AISTATS'15 inverts. "
            "Farajtabar itself has no public code and publishes no tables, so that "
            "whole row of §5.7 is [figure] and [verified] prose — its headline "
            "number is a ~1% absolute success probability on MemeTracker, which is "
            "the sober reminder that retrospective reconstruction on real data is "
            "very hard."
        ),
    ),
    "cosasi": ExternalBaseline(
        name="cosasi",
        kind=classical,
        title="cosasi: graph diffusion source inference toolkit",
        venue="JOSS 2022",
        repo="https://github.com/lmiconsulting/cosasi",
        paper="https://joss.theoj.org/papers/10.21105/joss.04894",
        entry="pip install cosasi",
        task="source_localization",
        status="blocked",
        fetch="git",
        blocker=(
            "the umbrella entry is not an arm — it ships several estimators, and "
            "one arm returning one of them would hide which. Use cosasi_jordan, "
            "cosasi_netsleuth, cosasi_lisn or cosasi_rumor_centrality, all of "
            "which share this clone and venv."
        ),
        notes=(
            "The centrality-style classical estimators, including the rumor "
            "centrality reimplementation there is no first-party code for. Overlaps "
            "our own `localization_algorithms` pool by design — wiring it is a "
            "CROSS-CHECK on our implementations rather than new coverage, which is "
            "worth doing once for LPSI and rumor centrality specifically, since "
            "both are reimplemented from prose here. Warning: Several secondary sources "
            "cite a `qwertyjl/cosasi` URL that 404s; the repo above is the one its "
            "JOSS paper names."
        ),
    ),
    # Epidemic control (research/epidemic_control.md §3, §4). SIX arms from ONE
    # clone, and three of them are methods §11 lists as having no public release at
    # all — NetShield, DAVA and DAVA-fast. `allogn/Network-Immunization` is the only
    # third-party implementation of any of them, which makes this clone the single
    # highest-value install in the file: without it every spectral and every
    # data-aware number in our table is our own reimplementation and nobody else's.
    "netimm_netshield": _netimm_entry(
        "netimm_netshield",
        "NetShield",
        "NetShield: node immunization on large graphs",
        "ICDM 2010 / TKDE 2015",
        "(key) THE canonical node baseline of the spectral line, and the reference for "
        "our own `netshield`. Its Shield-value is submodular, so its greedy is "
        "(1 - 1/e)-optimal against it, and the paper's own Table 3 measures that "
        "surrogate against the TRUE eigendrop at 0.977-1.000 across four "
        "co-authorship graphs [verified, §5.3]. Warning: the authors published NO "
        "code, and §3.2's widely-repeated claim that EpiLearn ships a NetShield "
        "implementation is WRONG — that repo's tree has no shield, immunization or "
        "intervention code at all [derived, 2026-08-05]. This clone is therefore "
        "the only third-party NetShield anywhere and the only cross-check our own "
        "reimplementation can have. Its immunization comparison is [figure]-only "
        "(Fig 1), so running it produces the per-cell numbers that figure does not.",
    ),
    "netimm_dava": _netimm_entry(
        "netimm_dava",
        "Dom",
        "DAVA: data-aware vaccine allocation",
        "SDM 2014 / ACM TKDD 2015",
        "(key) THE ROW THIS TASK IS POSITIONED AGAINST (§9.4). NetShield optimizes "
        "lambda_1, needs no simulator and runs in milliseconds — we cannot beat it "
        "on its own metric and should not try. DAVA makes exactly OUR argument, that "
        "conditioning on the observed infection state changes the optimal "
        "allocation, and does it with a dominator-tree heuristic on a single "
        "snapshot; a learned action-conditioned model is the natural generalization. "
        "No public code by the authors (§11), so this is the only third-party "
        "implementation and the cross-check on our own `dava`. Its results are "
        "[figure]-only (expected saved nodes vs budget), so per-cell numbers do not "
        "exist upstream either.",
    ),
    "netimm_dava_fast": _netimm_entry(
        "netimm_dava_fast",
        "Dom",
        "DAVA-fast: one dominator tree, top-k children",
        "SDM 2014",
        "The near-linear variant of the entry above: ONE dominator tree and the top "
        "`k` of its root's children, rather than rebuilding after each dose. A "
        "separate arm rather than a flag for the reason §8.2 trap 2 of the "
        "dismantling review gives about reinsertion variants — `X` and `X-fast` get "
        "cited under one name and are not the same method, and the gap between these "
        "two rows is exactly what the rebuild buys.",
        fast=True,
    ),
    "netimm_netshape": _netimm_entry(
        "netimm_netshape",
        "NetShape",
        "NetShape: convex optimization of a hazard matrix",
        "KDD 2014",
        "Khalil, Dilkina & Song's continuous relaxation: minimize the spectral "
        "radius of the hazard matrix by projected subgradient, then threshold. The "
        "third line of §3 — neither a centrality nor a simulation — and the only "
        "convex method in the pool. Warning: it runs `ceil((R/eps)^2)` iterations, "
        "each a DENSE N x N eigendecomposition, so it is the arm most likely to hit "
        "--baseline-timeout on anything past a few thousand nodes; raise "
        "GWM_NETIMM_EPSILON to trade quality for iterations.",
    ),
    "netimm_degree": _netimm_entry(
        "netimm_degree",
        "Degree",
        "Targeted degree immunization, as implemented in Network-Immunization",
        "PRE 65:036104, 2002",
        "A CROSS-CHECK on our `degree_immunization`, not new coverage — and worth "
        "one arm precisely because it is the row §9.4 says has to be beaten. Two "
        "independent implementations of a five-line heuristic agreeing is the "
        "cheapest possible confirmation that our outbreak, budget and referee are "
        "wired the way this repo's are.",
    ),
    "netimm_random": _netimm_entry(
        "netimm_random",
        "Random",
        "Uniform random immunization, as implemented in Network-Immunization",
        "PRL 86:3200, 2001",
        "The control Pastor-Satorras & Vespignani's whole line exists to beat, and "
        "not a throwaway floor: their 2002 result is that on a power-law graph this "
        "needs an immunized fraction approaching 1 to halt spread, so the GAP "
        "between this row and `netimm_degree` is the finding that founded the field.",
    ),
    # Registered and BLOCKED, each with a verified reason. Every one of these is
    # named in research/epidemic_control.md §4 as a candidate; listing them with the
    # reason is what stops the same investigation being repeated.
    "rlgn": ExternalBaseline(
        name="rlgn",
        kind=learned,
        title="RLGN: controlling graph dynamics with RL and GNNs",
        venue="ICML 2021",
        repo="https://proceedings.mlr.press/v139/meirom21a.html",
        paper="https://arxiv.org/abs/2010.05313",
        entry="none published",
        task="epidemic_control",
        status="blocked",
        blocker=(
            "NO PUBLIC CODE. Checked arXiv, the PMLR proceedings page, the NVIDIA "
            "Research project page and a GitHub search for the method name and the "
            "authors [verified, 2026-08-05, and independently in §4.1 and §11]. "
            "Reproducing it means reimplementing two GNNs, a PPO loop and its "
            "temporal contact-graph environment from the paper's prose. TO UNBLOCK: "
            "reimplement it, or write to the authors."
        ),
        notes=(
            "(key) THE most comparable published table to what this task produces "
            "(§5.1): a per-step test budget of 1% of N over 20 steps, SIR-style "
            "dynamics with a latent period, and % healthy at the horizon. Two of its "
            "five graphs are ours to the digit — `ca_grqc` at 5,242 / 14,496 and "
            "`deezer_ro`, whose row we load at its own true edge count (its Table S4 "
            "pairs Romania's node count with Hungary's edge count). Warning: its "
            "budget convention is PER STEP and ours is a one-shot k, which §8.2 trap "
            "3 records as non-comparable — cite its numbers as context, not as a "
            "run. Its own Table 2 also has Degree and Eigenvector tying to within "
            "0.1 on two of five graphs, which is the heuristic-collapse signature "
            "§9.4 predicts for us."
        ),
    ),
    "durleca": ExternalBaseline(
        name="durleca",
        kind=learned,
        title="DURLECA: dual-objective RL epidemic control agent",
        venue="KDD 2020",
        repo="https://github.com/AnyLeoPeace/DURLECA",
        paper="https://arxiv.org/abs/2008.01257",
        entry="Flow-GNN + RL (see repo)",
        task="epidemic_control",
        status="blocked",
        blocker=(
            "TWO blockers, and the second is the real one. (1) Its Beijing mobility "
            "dataset is withheld for privacy and is available only on request with "
            "the provider's authorization — the README says so [verified]. (2) More "
            "fundamentally, its state is an hourly ORIGIN-DESTINATION FLOW TENSOR "
            "over city regions and its action is a per-edge mobility multiplier on "
            "that tensor; it has no interface that takes a contact graph and an "
            "index case. Feeding it our graph would need a metapopulation model we "
            "do not have, not an adapter."
        ),
        notes=(
            "Its intervention IS our `contact_reduce` lever — a continuous per-edge "
            "multiplier — which is the branch §2.2 says NDlib's compartmental models "
            "cannot express at all and the reason we wrote our own stepper. So the "
            "lever is comparable in KIND even though the instance is not, and that "
            "is worth saying in a table rather than leaving the row absent."
        ),
    ),
    "epilearn": ExternalBaseline(
        name="epilearn",
        kind=learned,
        title="EpiLearn: ML toolkit for epidemic modelling",
        venue="arXiv 2406.06016, 2024",
        repo="https://github.com/Emory-Melody/EpiLearn",
        paper="https://arxiv.org/abs/2406.06016",
        entry="epilearn package",
        task="epidemic_control",
        status="blocked",
        blocker=(
            "IT HAS NO INTERVENTION CODE. Warning: §3.2 and §11 both state that a "
            "`NetShield` implementation ships in EpiLearn, and that is WRONG — "
            "verified by listing the repo's whole tree and grepping every module "
            "[derived, 2026-08-05]. What it ships is FORECASTING (`STGCN`, `EpiGNN`, "
            "`ColaGNN`, `DCRNN`, `STAN`, `CausalGNN`-adjacent models), DETECTION, a "
            "`NetworkSIR` forward simulator, `DMP`, and graph transforms. There is "
            "no shield value, no immunization selector, no intervention API and no "
            "`tasks/intervention.py`. It cannot choose k nodes, so there is nothing "
            "to adapt. TO UNBLOCK: nothing — this is a correction to the review, not "
            "a missing adapter. `external:netimm_netshield` is the runnable "
            "NetShield."
        ),
        notes=(
            "From the Emory Melody lab and the nearest thing this space has to a "
            "shared harness, so the mis-attribution is worth recording rather than "
            "silently dropping: every secondary source that says 'NetShield ships in "
            "EpiLearn' traces back to §3.2 of our own review. Its FORWARD models "
            "(NetworkSIR, DMP) are a legitimate future cross-check on "
            "`data/wm_epidemic.py`, which is a different use than a baseline."
        ),
    ),
    "covasim": ExternalBaseline(
        name="covasim",
        kind=classical,
        title="Covasim: agent-based COVID model with an intervention API",
        venue="PLOS Comp Biol 17(7), 2021",
        repo="https://github.com/InstituteforDiseaseModeling/covasim",
        paper="https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009149",
        entry="covasim package",
        task="epidemic_control",
        status="blocked",
        blocker=(
            "ITS INTERVENTIONS ARE POPULATION-LEVEL, NOT TOPOLOGICAL. It has a rich "
            "intervention API (`test_num`, `contact_tracing`, `vaccinate_num`, "
            "`change_beta`) and it CAN be handed a custom contact layer, so the "
            "input side is not the blocker — the output side is. Its allocations are "
            "by priority, age band or probability, never 'these k node ids', so it "
            "produces no set for our referee to score. Adapting it would mean "
            "writing the selector ourselves, at which point the baseline is ours "
            "rather than theirs."
        ),
        notes=(
            "§4.4 lists it as a software baseline rather than a method, and that is "
            "the right reading: its contribution is the SIMULATOR and its "
            "household/school/work layer structure. Its `change_beta` intervention is "
            "our `contact_reduce` lever by another name."
        ),
    ),
    "pandemic_simulator": ExternalBaseline(
        name="pandemic_simulator",
        kind=learned,
        title="PandemicSimulator: RL over an agent-based pandemic model",
        venue="arXiv 2010.10560 / AAAI-21 workshop",
        repo="https://github.com/SonyResearch/PandemicSimulator",
        paper="https://arxiv.org/abs/2010.10560",
        entry="python_scripts/ (see repo)",
        task="epidemic_control",
        status="blocked",
        blocker=(
            "IT GENERATES ITS OWN POPULATION AND HAS NO GRAPH INPUT. Verified from "
            "its README [2026-08-05]: it builds a synthetic age-distributed "
            "population across homes, schools, stores, bars and restaurants, and its "
            "actions are government REGULATION stages (stay-at-home orders, "
            "closures) rather than per-node choices. There is no place to hand it a "
            "contact graph and no k-node set to read back."
        ),
        notes=(
            "§4.1 records that the simulator is the contribution as much as the "
            "policy. Same category as Covasim: a forward model to compare "
            "`data/wm_epidemic.py` against some day, not a selector to score."
        ),
    ),
    # ...and the rest of §3 and §4, registered with a verified reason rather than
    # left absent. The registry's own norm: a listed reason is what stops the same
    # investigation being repeated, and every one of these is a method a reader of
    # research/epidemic_control.md would reasonably ask about.
    "idrleca": ExternalBaseline(
        name="idrleca",
        kind=learned,
        title="IDRLECA: contact tracing and epidemic intervention via deep RL",
        venue="arXiv 2102.08251 -> ACM TKDD 17(3), 2023",
        repo="https://dl.acm.org/doi/10.1145/3546870",
        paper="https://arxiv.org/abs/2102.08251",
        entry="none published",
        task="epidemic_control",
        status="blocked",
        blocker=(
            "NO PUBLIC CODE. Searched arXiv, the ACM DL entry, the authors' pages "
            "and GitHub by method name and by all four authors [verified, "
            "2026-08-05, and independently in §4.1]. Same Tsinghua group as "
            "DURLECA, whose repo IS public — so the absence here is specific to "
            "this paper rather than a group policy."
        ),
        notes=(
            "Per-node isolate/test actions over an infection graph, which is the "
            "closest ACTION SPACE in §4.1 to our `quarantine` lever — closer than "
            "DURLECA's per-edge mobility multiplier. Warning: it models contact "
            "TRACING as part of the policy, which §2.5 records is not an action in "
            "our formulation at all: tracing changes the OBSERVATION, not the graph "
            "or the state, and belongs in a POMDP observation model we do not have."
        ),
    ),
    "epimodel": ExternalBaseline(
        name="epimodel",
        kind=classical,
        title="EpiModel: stochastic network epidemic models",
        venue="J Stat Software 84(8), 2018",
        repo="https://github.com/EpiModel/EpiModel",
        paper="https://www.jstatsoft.org/v84/i08/",
        entry="R package",
        task="epidemic_control",
        status="blocked",
        blocker=(
            "R, and the wrong object. The repo is live [verified, 2026-08-05] but "
            "it is an R package, and `setup_baselines` builds Python venvs — an R "
            "install path would be the first of its kind here. More decisive: its "
            "networks are ERGM-GENERATED dynamic graphs rather than a supplied "
            "adjacency, and its interventions are population-level rates, so there "
            "is no k-node set to read back even with an R bridge."
        ),
        notes=(
            "§4.4 lists it as a software baseline rather than a method, and that is "
            "the right reading: it is the reference implementation of stochastic "
            "network epidemic models. A future cross-check on `data/wm_epidemic.py`, "
            "not a selector to score."
        ),
    ),
    "colagnn": _forecasting_entry(
        "colagnn",
        "Cola-GNN: cross-location attention for ILI prediction",
        "CIKM 2020",
        "https://github.com/amy-deng/colagnn",
        "https://yue-ning.github.io/docs/CIKM20-colagnn.pdf",
    ),
    "stan": _forecasting_entry(
        "stan",
        "STAN: spatio-temporal attention network with an SIR-consistency loss",
        "JAMIA 28(4), 2021",
        "https://github.com/v1xerunt/STAN",
        "https://arxiv.org/abs/2008.04215",
    ),
    "epignn": _forecasting_entry(
        "epignn",
        "EpiGNN: region-aware transmission graph with a learned adjacency",
        "ECML-PKDD 2022",
        "https://github.com/Xiefeng69/EpiGNN",
        "https://arxiv.org/abs/2208.11517",
    ),
    "netmelt": ExternalBaseline(
        name="netmelt",
        kind=classical,
        title="NetMelt / NetGel (Gelling): edge immunization by eigendrop",
        venue="CIKM 2012 (best paper)",
        repo="https://faculty.cc.gatech.edu/~badityap/papers/netgel-cikm12.pdf",
        paper="https://faculty.cc.gatech.edu/~badityap/papers/netgel-cikm12.pdf",
        entry="none published",
        task="epidemic_control",
        status="blocked",
        blocker=(
            "NO PUBLIC CODE FOUND (§3.2, §5.5). Unlike NetShield and DAVA — whose "
            "third-party implementations turned up in "
            "`allogn/Network-Immunization` (§0.0) — that repo has no EDGE solver at "
            "all, so there is no third-party route either. Our `netmelt` scores an "
            "arc by `u(i) * u(j)` per the paper's own rule and is a "
            "reimplementation from prose."
        ),
        notes=(
            "The canonical EDGE baseline of the spectral line and the row our "
            "`edge_cut` and `contact_reduce` pools lead with. Its results are "
            "[figure]-only (eigendrop vs k curves in both the melting and the "
            "gelling direction), so per-cell numbers do not exist upstream either "
            "and running it would produce the table that paper does not have."
        ),
    ),
    "fractional_immunization": ExternalBaseline(
        name="fractional_immunization",
        kind=classical,
        title="Fractional Immunization in Networks",
        venue="SDM 2013",
        repo="https://faculty.cc.gatech.edu/~badityap/papers/smartalloc-sdm13.pdf",
        paper="https://faculty.cc.gatech.edu/~badityap/papers/smartalloc-sdm13.pdf",
        entry="none published",
        task="epidemic_control",
        status="blocked",
        blocker=(
            "NO PUBLIC CODE FOUND (§3.2, §5.5), and its instance is unobtainable "
            "besides: the paper is motivated by hospital-transfer networks that "
            "were never released. `allogn/Network-Immunization`'s `NetShape` solver "
            "(Khalil KDD'14) is the nearest RUNNABLE convex relaxation and is wired "
            "as `external:netimm_netshape`."
        ),
        notes=(
            "The method our `contact_reduce` lever is closest to in kind: it drops "
            "the all-or-nothing assumption and allocates a CONTINUOUS amount of "
            "resource per node or edge under a budget. §2.2 records that this whole "
            "branch is inexpressible against NDlib's compartmental models, which is "
            "why `data/wm_epidemic.py` exists."
        ),
    ),
    "preciado": ExternalBaseline(
        name="preciado",
        kind=classical,
        title="Optimal resource allocation for network protection (geometric program)",
        venue="IEEE TCNS 1(1), 2014",
        repo="https://arxiv.org/abs/1309.6270",
        paper="https://arxiv.org/abs/1309.6270",
        entry="none published",
        task="epidemic_control",
        status="blocked",
        blocker=(
            "NO PUBLIC CODE FOUND (§3.2), AND ITS INSTANCE CANNOT BE REBUILT. §6.5: "
            "its 56-airport network was never published as a file, and while it is "
            "reconstructable from OpenFlights with a >10 MPPY filter, the PASSENGER "
            "WEIGHTS that make it a weighted digraph are not in OpenFlights. A "
            "cardinality budget also mismatches its formulation, which allocates a "
            "continuous per-node rate reduction."
        ),
        notes=(
            "Convex and therefore globally optimal, which is unusual in this "
            "literature — everything else here is greedy or heuristic. Our "
            "`preciado_allocation` discretizes the GP's own first-order structure "
            "to a top-k and is LABELLED a discretization rather than the method. "
            "Its published results are [figure]-only."
        ),
    ),
    # Cascade prediction ------------------------------------------------------
    #
    # research/cascade_prediction.md 8.5's finding is that this literature has NO
    # benchmark — no OGB or TGB analogue, no leaderboard to enter, no split to
    # inherit — so every entry here is scored on OUR protocol against OUR replayed
    # corpus, and the published numbers in each `notes` are context markers rather
    # than comparisons. 9.9 is emphatic: do not chase the leaderboard, CasFlow is a
    # 2M-parameter model tuned for this one task and we will not beat it.
    #
    # Warning: FOUR of the eight most-cited methods here are PYTHON 2 and two more
    # are TensorFlow 1. That is not neglect on our part — this line of work peaked
    # in 2017-19 and its reference implementations were never ported. Each is
    # registered with the exact evidence, because a reader who sees "DeepHawkes,
    # code available" in 4.1 will reasonably ask why it is not a baseline.
    "casflow": _casflow_entry(
        "casflow",
        "CasFlow: hierarchical structures and propagation uncertainty for cascade prediction",
        "TKDE 2021",
        "https://github.com/Xovee/casflow",
        "https://doi.org/10.1109/TKDE.2021.3126475",
        "casflow.py",
        ".",
        # The pin is `tensorflow==2.9.3`, which has no wheel for current Python or
        # for arm64 macOS at all. The code uses plain Keras 2 APIs that every TF 2.x
        # provides, so unpinning is a smaller lie than not running it.
        [("requirements.txt", "tensorflow==2.9.3", "tensorflow")],
        "(key) THE reference SOTA of 2021-23 and the baseline every later paper "
        "reports. Its own dataset bundle is the de-facto benchmark artefact of this "
        "literature (8.5), which is why `data/datasets/casflow_bundle.py` parses "
        "exactly its line format. Warning: its headline result table (TKDE'21 Table "
        "3) is a RASTER IMAGE — 0 records that `pdftotext -layout` returns the "
        "caption and nothing else, so every CasFlow number in 5 is a RE-RUN by a "
        "later paper and those three disagree with each other. CasFT's re-run puts "
        "it at MSLE 2.3370 on Weibo 0.5h, 4.7799 on Twitter 1d, 1.4370 on APS 3y "
        "[verified, 5.1]; under CasTemp's leak-free split the same method reads "
        "1.685 / 1.329 / 2.438 [verified, 5.3], and 5.3's diagnosis is that "
        "CasFlow 'exhibit[s] low training losses but significantly higher test "
        "losses' because its architecture 'learned dataset-specific shortcuts "
        "enabled by temporal leakage'. Run it under BOTH --cp-split values; the gap "
        "is the experiment.",
    ),
    "ccgl": _casflow_entry(
        "ccgl",
        "CCGL: contrastive cascade graph learning",
        "TKDE 2022",
        "https://github.com/Xovee/ccgl",
        "https://arxiv.org/abs/2107.12576",
        "src/base_model.py",
        "src",
        # Same TF pin problem, plus one genuine py2 leftover: `graphwave/utils/
        # function_utils.py` carries `print "error: argument is negative"`, which
        # does not parse under Python 3 at all. Verified by reading the file at HEAD.
        [
            ("requirements.txt", "tensorflow-gpu==2.3", "tensorflow"),
            ("requirements.txt", "networkx==2.4", "networkx"),
            ("requirements.txt", "scikit-learn==0.21.1", "scikit-learn"),
            ("requirements.txt", "scipy==1.4.1", "scipy"),
            (
                "src/utils/graphwave/utils/function_utils.py",
                'print "error: argument is negative"',
                'print("error: argument is negative")',
            ),
        ],
        "The self-supervised entry: contrastive pretraining on augmented cascade "
        "graphs, then fine-tuning. Same author and same pipeline as `casflow`, so "
        "one driver covers both — what differs is the pretraining stage, which is "
        "the transfer-learning question this literature otherwise never asks. It "
        "reports on the same Weibo-A / Twitter-A / APS-A triple (7), so its row is "
        "directly comparable to `casflow`'s under our protocol.",
    ),
    "ctcp": ExternalBaseline(
        name="ctcp",
        kind=learned,
        title="CTCP: continuous-time graph learning for cascade popularity prediction",
        venue="IJCAI 2023",
        repo="https://github.com/lxd99/CTCP",
        paper="https://arxiv.org/abs/2306.03756",
        entry="main.py",
        task="cascade_prediction",
        status="needs_setup",
        # The repo ships no requirements file; its README pins python 3.7 / torch
        # 1.9.1 / dgl 0.8.2, none of which resolve on a current interpreter. Listed
        # unpinned for the same reason DITTO's and cosasi's are.
        requirements=None,
        pip_packages=("torch", "dgl", "scikit-learn", "numpy", "pandas", "tqdm"),
        export=partial(_prediction_export, driver="ctcp_driver.py"),
        command=_prediction_command,
        parse_seeds=_prediction_parse,
        notes=(
            "(key) The most interesting arm to run BESIDE CasFlow rather than "
            "instead of it. 5.3: under CasTemp's leak-free split CTCP is one of only "
            "two methods whose train-vs-test loss curves stay flat while CasFlow's "
            "and CasDO's diverge [verified, Fig 2] — so it is the published method "
            "least likely to be exploiting the temporal shortcut 8.3 describes, and "
            "the one whose ranking should move LEAST between our two --cp-split "
            "settings. It is also the easiest to wire: its input is a plain (id, "
            "src, dst, cas, time) event table and its split crosses as TIME "
            "BOUNDARIES rather than as flags, so our chronological protocol is "
            "reproduced inside the repo rather than fought. Its own Table 2 reports "
            "MSLE 4.6916 / 2.5929 / 1.6289 on its OWN re-preprocessing of "
            "Twitter/Weibo/APS (19,718 / 39,076 / 48,575 cascades — 6.4's versions "
            "C, B and B), which is 3-10x smaller than CasFlow's and NOT comparable "
            "to it."
        ),
    ),
    "castemp": ExternalBaseline(
        name="castemp",
        kind=learned,
        title="CasTemp / Beyond Leakage: temporal random walks with an inter-cascade competition graph",
        venue="preprint (arXiv 2510.25348)",
        repo="https://github.com/Lucas-PJ/CasTemp-ALGO",
        paper="https://arxiv.org/abs/2510.25348",
        entry="main.py",
        task="cascade_prediction",
        status="blocked",
        blocker=(
            "ITS INPUT IS ITS OWN PREPROCESSED ARTEFACT, NOT A CASCADE FILE. "
            "Verified by reading `process_dataset.py` and `dataset_process/"
            "data_loader.py` at HEAD: `main.py` reads six files from "
            "`./processed_data/{dataset}/` — `{d}_node_feat.npy`, "
            "`{d}_item_feat_emb.npy`, `{d}_item_forward_counts.csv`, "
            "`{d}_item_price_delta.csv` and per-split `{d}_*_forward_relations.csv` "
            "— of which the two `.npy` files are LEARNED EMBEDDINGS produced by a "
            "pipeline the repo does not ship for an arbitrary corpus, and "
            "`item_price_delta` is a Taoke-specific auxiliary signal with no "
            "analogue in Weibo, Twitter or APS. Writing plausible embeddings "
            "ourselves would make the row a measurement of OUR feature choice "
            "rather than of their method. TO UNBLOCK: reimplement "
            "`process_dataset.py`'s embedding stage from its own code, which is "
            "tractable but is a reimplementation rather than an adapter."
        ),
        notes=(
            "(key) The paper 5.3 and 8.3 both lean on, and the source of the single "
            "most transferable finding in that file: the field's standard 70/15/15 "
            "random-over-cascades split LEAKS, and under its 1:1:1 chronological fix "
            "CasFlow and CasDO fall BELOW a plain MLP while the whole field's APS "
            "band moves from 1.19-2.11 to 2.28-4.82 [verified, Table 4]. **We "
            "implement its protocol rather than its model** — `--cp-split "
            "chronological` IS its fix, and it is our default from the first commit "
            "for exactly that reason, so the finding is inherited even though the "
            "code is not. Its `Taoke.zip` IS wired, as `data/datasets/taoke.py`. "
            "Warning: 11 records that this is an arXiv preprint carrying an ACM "
            "template with placeholder conference metadata, not yet independently "
            "replicated, and that its leakage claim deserves a second source before "
            "being treated as settled."
        ),
    ),
    "deephawkes": ExternalBaseline(
        name="deephawkes",
        kind=learned,
        title="DeepHawkes: bridging prediction and understanding of information cascades",
        venue="CIKM 2017",
        repo="https://github.com/CaoQi92/DeepHawkes",
        paper="https://dl.acm.org/doi/10.1145/3132847.3132973",
        entry="deep_learning/run_sparse.py",
        task="cascade_prediction",
        status="blocked",
        blocker=(
            "PYTHON 2 AND TENSORFLOW 1, both verified by reading the repo at HEAD. "
            "`deep_learning/config.py` uses py2 print STATEMENTS throughout "
            "(`print \"observation time\",observation`), so the module does not parse "
            "under Python 3 at all; `deep_learning/model_sparse.py` is built on "
            "`tf.placeholder`, which TF 2 removed. Python 2.7 is EOL and has no "
            "wheels for TF on any current platform, so a per-baseline venv cannot be "
            "created for it — the constraint is the interpreter, not an adapter. TO "
            "UNBLOCK: a full py2->py3 port plus a `tf.compat.v1` rewrite, which is "
            "a reimplementation."
        ),
        notes=(
            "(key) The most-reproduced baseline in this literature and the "
            "interpretability-vs-accuracy bridge: it injects the three Hawkes "
            "ingredients (user influence, self-excitation, time decay) into a GRU "
            "over diffusion PATHS. Its release is also the origin of Weibo-A, the "
            "corpus 6.4 lists first and the one carrying almost every published "
            "number — so its data reaches us through "
            "`data/datasets/casflow_weibo.py` even though its code does not. "
            "CasFT's re-run puts it at MSLE 2.8741 on Weibo 0.5h against CasFlow's "
            "2.3370 [verified, 5.1]. Our `hawkes` and `hawkes_hybrid` implement the "
            "same generative ingredients without the GRU."
        ),
    ),
    "deepcas": ExternalBaseline(
        name="deepcas",
        kind=learned,
        title="DeepCas: an end-to-end predictor of information cascades",
        venue="WWW 2017",
        repo="https://github.com/chengli-um/DeepCas",
        paper="https://arxiv.org/abs/1611.05373",
        entry="tensorflow/run.py",
        task="cascade_prediction",
        status="blocked",
        blocker=(
            "PYTHON 2 AND TENSORFLOW 1, verified at HEAD: `tensorflow/run.py` ends "
            "in py2 print STATEMENTS (`print \"Test Loss:\", best_test_loss`) and "
            "`tensorflow/model.py` is placeholder-based TF 1. The repo's second "
            "implementation is TORCH7 LUA (`torch/main/params.lua`), which needs an "
            "interpreter that has been unmaintained since 2017. Same interpreter-"
            "level blocker as `deephawkes`."
        ),
        notes=(
            "(key) The founding deep model of this literature — random walks over "
            "the cascade graph, bi-GRU with attention, regressing log dP — and the "
            "paper that killed hand-crafted features as the default. CasFT's re-run "
            "puts it at MSLE 4.6460 on Weibo 0.5h [verified, 5.1], which is WORSE "
            "than DeepHawkes and only slightly better than the feature baseline: "
            "3.1's point that feature models remain competitive is visible in that "
            "column."
        ),
    ),
    "cascn": ExternalBaseline(
        name="cascn",
        kind=learned,
        title="CasCN: recurrent cascades convolutional networks",
        venue="ICDE 2019",
        repo="https://github.com/ChenNed/CasCN",
        paper="https://par.nsf.gov/servlets/purl/10122600",
        entry="model/run_graph_sequence.py",
        task="cascade_prediction",
        status="needs_setup",
        requirements=None,
        pip_packages=("tensorflow", "networkx", "scipy", "numpy", "six"),
        # Two patches, and NEITHER is the one an earlier reading of this repo
        # assumed. (1) `preprocessing/utils.py` does not parse under Python 3
        # because of a genuine INDENTATION BUG at line 118-122 — an `if` whose body
        # is over-indented and whose `else` is then mismatched — not because of any
        # py2 construct; every other file in the repo parses cleanly. (2) The TF1
        # API is shimmed to `compat.v1` exactly as `coupledgnn`'s is.
        patches=[
            (
                "preprocessing/utils.py",
                "            if len(observation_path)>100:\n"
                "                    discard_cascade_id[cascadeID] = 1\n"
                "                    continue\n"
                "                else:\n"
                "                    discard_cascade_id[cascadeID]=0\n",
                "            if len(observation_path)>100:\n"
                "                discard_cascade_id[cascadeID] = 1\n"
                "                continue\n"
                "            else:\n"
                "                discard_cascade_id[cascadeID]=0\n",
            ),
            (
                "model/model_sparse_graph_signal.py",
                "import tensorflow as tf",
                "import tensorflow.compat.v1 as tf\ntf.disable_v2_behavior()",
            ),
            (
                "model/run_graph_sequence.py",
                "import tensorflow as tf",
                "import tensorflow.compat.v1 as tf\ntf.disable_v2_behavior()",
            ),
        ],
        export=partial(_prediction_export, driver="cascn_driver.py"),
        command=_prediction_command,
        parse_seeds=_prediction_parse,
        notes=(
            "The first method in this literature to use both structure and time "
            "properly: a cascade as a SEQUENCE of sub-cascade graphs, a GCN per "
            "snapshot and an LSTM across them. **The row worth running rather than "
            "citing**, because 5.3 shows it is where the leak-free split changes a "
            "RANKING rather than a level: under the field's own leaky protocol it "
            "sits mid-pack (MSLE 2.7931 on Weibo 0.5h, behind CasFlow's 2.3370), and "
            "under CasTemp's fix it is the BEST of the six re-run baselines on "
            "Twitter (1.206) and second on APS [verified, 5.3 Table 4]. Run it under "
            "both --cp-split values. Its own preprocessing is reused rather than "
            "reimplemented — it reads exactly the per-split line format "
            "`casflow_driver.py` writes, and `caslaplacian."
            "calculate_scaled_laplacian_dir` is the method's actual input. Warning: "
            "its preprocessing DISCARDS any cascade with more than 100 observed "
            "participants, so --cp-truncate above 100 silently shrinks this arm's "
            "pool relative to every other one; the driver truncates to match and the "
            "row's cascade count says so. Its `num_nodes` flag is a PER-CASCADE "
            "bound of 100, not the global graph — the dense "
            "`[batch, steps, num_nodes, num_nodes]` placeholder is 100x100 per step."
        ),
    ),
    "mucas": ExternalBaseline(
        name="mucas",
        kind=learned,
        title="MUCas: multi-scale graph capsule networks with influence attention",
        venue="IJCAI 2022",
        repo="https://github.com/ChenNed/MUCas",
        paper="https://doi.org/10.24963/ijcai.2022/300",
        entry="model/mucas.py",
        task="cascade_prediction",
        status="blocked",
        blocker=(
            "ELEVEN INPUT ARRAYS, not two. Its TF1 API is the same `compat.v1` shim "
            "`cascn` and `coupledgnn` now use, and its sources parse cleanly under "
            "Python 3, so neither is the blocker — `model/run_model.py`'s "
            "`get_batch(id_list, x, support, adj, pos, y, y_c, time_interval, k, "
            "rnn_index, max_order, order_level, ...)` is. Beyond the cascade "
            "snapshots it needs sinusoidal POSITION encodings, per-cascade Chebyshev "
            "K-ORDER tensors, a randomized-edge-sampled support, and a second "
            "OUTBREAK-CLASSIFICATION label `y_c` that our contract has no analogue "
            "for. Its `utils/gen_model_input.py` builds all of them and IS shipped "
            "(unlike CasTemp's embedding stage), so this is wireable by the same "
            "route `cascn` took — write the per-split line files, run the repo's own "
            "`gen_model_input`, then its model. It is a driver's worth of work "
            "rather than a structural blocker, and it is the next one to do."
        ),
        notes=(
            "The graph-capsule entry, with directional / dynamic / position-aware "
            "cascade encoding. Reports on the Weibo-A / Twitter-A / APS-A triple "
            "(7) so it would be directly comparable to `casflow` if it ran; it "
            "appears in this file only through other papers' comparison tables."
        ),
    ),
    "coupledgnn": ExternalBaseline(
        name="coupledgnn",
        kind=learned,
        title="CoupledGNN: popularity prediction with coupled graph neural networks",
        venue="WSDM 2020",
        repo="https://github.com/CaoQi92/CoupledGNN",
        paper="https://arxiv.org/abs/1906.09032",
        entry="train.py",
        task="cascade_prediction",
        status="needs_setup",
        # No requirements file; the README pins Python 2.7.5 and TF 1.14. Its
        # sources DO parse under Python 3 (the prints are function-style and
        # `utils.load_data` already branches on `sys.version_info`), so the
        # interpreter is not the blocker and only the TF1 API is — which the
        # patches below shim rather than port.
        requirements=None,
        pip_packages=("tensorflow", "networkx", "scipy", "numpy"),
        # `tf.placeholder`, `tf.flags` and `tf.set_random_seed` were all removed in
        # TF 2. `compat.v1` still ships every one of them, so the edit is mechanical
        # and has no behavioural content — the alternative was a Python 3.7 venv
        # with `tensorflow==1.15`, which has no wheel on current platforms.
        patches=[
            (
                "models.py",
                "import tensorflow as tf",
                "import tensorflow.compat.v1 as tf\ntf.disable_v2_behavior()",
            ),
            (
                "layers.py",
                "import tensorflow as tf",
                "import tensorflow.compat.v1 as tf\ntf.disable_v2_behavior()",
            ),
            (
                "train.py",
                "import tensorflow as tf",
                "import tensorflow.compat.v1 as tf\ntf.disable_v2_behavior()",
            ),
        ],
        export=partial(_prediction_export, driver="coupledgnn_driver.py"),
        command=_prediction_command,
        parse_seeds=_prediction_parse,
        notes=(
            "(key) **The closest published method to our own formulation**, which is "
            "why it is wired despite being the hardest of the three. Two COUPLED "
            "GNNs — one propagating node ACTIVATION STATE, one propagating "
            "INFLUENCE, iterated over K layers to imitate the cascading effect on "
            "the global graph — is structurally what our structured head does with "
            "`infected` and `frontier`, so a difference in this row is a statement "
            "about the architecture rather than about a feature pipeline. It is also "
            "the only entry here whose protocol withholds TIMESTAMPS entirely "
            "(research/cascade_prediction.md 5.5): it sees the early adopter SET "
            "plus the global graph and nothing else, which is the protocol our own "
            "`neighborhood_size` and `degree_scaled` rows sit under, and its "
            "published numbers are MRSE / mRSE / MAPE / WroPerc rather than MSLE for "
            "that reason. **Warning: two of its inputs are OURS, not the authors'.** "
            "`load_data` needs a 6-dimensional per-node feature vector and a 32-d "
            "`.emb_32` embedding, and the repo ships a derivation for neither; the "
            "driver computes six standard graph statistics (matching the shape of "
            "the shipped example, which is `[degree, 5 normalized floats]`) and a "
            "spectral embedding in place of the DeepWalk one its own comment names. "
            "Neither is the method's contribution, but a reader comparing this row "
            "against its Table 1 has to know. That table is on its own Weibo-D "
            "subset (23,681 users / 3,228 cascades, sampled down from 1.78M — the "
            "move `--cp-max-nodes` reproduces) and is not comparable to a row "
            "produced here in any case."
        ),
        extra_env={},
    ),
    "seismic": ExternalBaseline(
        name="seismic",
        kind=classical,
        title="SEISMIC: self-exciting point process with time-varying infectiousness",
        venue="KDD 2015",
        repo="https://cran.r-project.org/package=seismic",
        paper="https://arxiv.org/abs/1506.02594",
        entry="R package `seismic`",
        task="cascade_prediction",
        status="blocked",
        blocker=(
            "IT IS AN R PACKAGE. The reference implementation is on CRAN and there "
            "is no Python release; `baselines/setup_baselines.py` creates per-"
            "baseline venvs with `python -m venv`, which cannot install an R "
            "library, and adding an R toolchain for one baseline would be the first "
            "of its kind here. Our own `seismic` in "
            "`coding_agent/tools/prediction_algorithms.py` implements the same "
            "closed-form estimator and the same supercritical DECLINE, with the one "
            "stated deviation that the infectiousness is estimated from binned "
            "waves rather than exact event times."
        ),
        notes=(
            "(key) The generative baseline every paper in 5 prints, and the one "
            "whose FAILURE MODE matters most to us: it produces no prediction at "
            "all for supercritical cascades (507 of ~30K Tweet-1Mo at five minutes, "
            "1,022 of ~20K News) because the branching factor exceeds 1 and the "
            "expected size diverges [verified, 5.4]. 3.2 records that this is the "
            "same runaway our `ens_count_bias` metric was built to catch and the "
            "same reason our structured head gates on `frontier_u`. Its own "
            "breakout coverage is 78 of the top-100 most-reshared tweets identified "
            "within ten minutes [verified]."
        ),
    ),
    "featuredriven_hawkes": ExternalBaseline(
        name="featuredriven_hawkes",
        kind=classical,
        title="Feature-driven and point-process approaches for popularity prediction",
        venue="CIKM 2016",
        repo="https://github.com/s-mishra/featuredriven-hawkes",
        paper="https://arxiv.org/abs/1608.04862",
        entry="code/rscripts/marked_hawkes.R",
        task="cascade_prediction",
        status="blocked",
        blocker=(
            "R, plus a Jupyter notebook. Verified by listing the repo at HEAD: the "
            "only executable code is `code/rscripts/marked_hawkes.R` and "
            "`simulation.R`, with `code/marked_hawkes_point_process.ipynb` as a "
            "tutorial around them — there is no Python entry point at all. Same "
            "R-toolchain blocker as `seismic`. Our `hawkes` and `hawkes_hybrid` "
            "implement its two rows, with the stated deviation that the fit is "
            "moment-matched on binned waves rather than MLE on exact times."
        ),
        notes=(
            "(key) Its Table 2 is the ONLY place in this literature where a "
            "generative model's failure COUNT is published beside its error, which "
            "is why 8.4 singles it out: Hawkes reaches ARE 0.36 against SEISMIC's "
            "2.61 on Tweet-1Mo at five minutes AND fails on 302 cascades against "
            "SEISMIC's 507 — better on both axes [verified, 5.4]. Its hybrid row "
            "(0.17 / 0.15 / 0.11 against pure Hawkes 0.27 / 0.22 / 0.17) is "
            "structurally OUR `structured_residual` head: a learned corrective layer "
            "on a generative core (3.2)."
        ),
    ),
    "hip": ExternalBaseline(
        name="hip",
        kind=classical,
        title="HIP: Hawkes Intensity Process with exogenous promotion",
        venue="WWW 2017",
        repo="https://github.com/andrei-rizoiu/hip-popularity",
        paper="https://arxiv.org/abs/1602.06033",
        entry="pyhip.py",
        task="cascade_prediction",
        status="blocked",
        blocker=(
            "IT TAKES AN EXOGENOUS STIMULUS SERIES OUR CASCADES DO NOT CARRY. "
            "Verified by reading `pyhip.py` and `pyhip_example.py` at HEAD: "
            "`HIP.initial(daily_share, daily_view, num_train, num_test, "
            "num_initialization)` needs TWO aligned series — an external promotion "
            "signal and the response it drives — and that is the whole contribution "
            "of the method. None of Weibo, Twitter, APS, Digg, MemeTracker or Taoke "
            "publishes a separate stimulus series, so a run here would have to "
            "fabricate one and would be measuring our fabrication. (`pyhip_example.py` "
            "is separately py2 — `import cPickle` — but that is the smaller "
            "problem.) TO UNBLOCK: a corpus with a paired exogenous signal; HIP's "
            "own is YouTube share counts against view counts."
        ),
        notes=(
            "The one method in 3.2 that models EXOGENOUS arrivals — search, front "
            "pages, off-platform sharing — which 2.2 names as one of three specific "
            "mechanisms by which real cascades violate our structured head's "
            "composition rule (a node adopts with no infected in-neighbour at all). "
            "That makes it the most diagnostically interesting row in the pool and "
            "the one we can least honestly run. Our `hip` implements its power-law "
            "memory kernel WITHOUT the exogenous term and says so."
        ),
    ),
    "topolstm": ExternalBaseline(
        name="topolstm",
        kind=learned,
        title="Topo-LSTM: topological recurrent network for diffusion prediction",
        venue="ICDM 2017",
        repo="https://github.com/vwz/topolstm",
        paper="https://arxiv.org/abs/1711.10162",
        entry="code/main.py",
        task="cascade_prediction",
        status="blocked",
        blocker=(
            "PYTHON 2. Verified at HEAD: `code/data_utils.py` uses py2 print "
            "STATEMENTS (`print 'pickle exists.'`), so the module does not parse "
            "under Python 3. It is also MICROSCOPIC rather than macroscopic — it "
            "ranks the NEXT ADOPTER and is scored with Hits@k / MAP@k (1.2), not a "
            "popularity — so even ported it would answer a different question than "
            "`predict()` asks and would need its own contract and its own metric "
            "layer."
        ),
        notes=(
            "The source of 6.2's Digg / Twitter-URL / Memes rows, and therefore of "
            "the counts `data/datasets/digg_cascades.py` and "
            "`data/datasets/memetracker.py` are measured against. Its Table II is "
            "where Digg 2009's 279,632 / 2,617,993 / 3,553 comes from [verified]. "
            "Its structural idea — an LSTM whose gates are wired to the cascade's "
            "dynamic DAG rather than to a linear sequence — seeded CasCN."
        ),
    ),
    "deepinf": ExternalBaseline(
        name="deepinf",
        kind=learned,
        title="DeepInf: social influence prediction with deep learning",
        venue="KDD 2018",
        repo="https://github.com/xptree/DeepInf",
        paper="https://arxiv.org/abs/1807.05560",
        entry="src/deepinf.py",
        task="cascade_prediction",
        status="blocked",
        blocker=(
            "IT ANSWERS A DIFFERENT QUESTION. Verified by reading `src/` at HEAD: it "
            "is a BINARY classifier over (user, cascade) pairs — 'will `v` adopt', "
            "scored with AUC and F1 (1.2) — built on each user's r-hop ego network "
            "plus its active-neighbour states. There is no popularity to read back, "
            "so nothing our `predict()` contract or `popularity_metrics` can score. "
            "This is a category difference, not a missing adapter. Its code is "
            "PyTorch and Python 3 and would run fine; the output type is the blocker."
        ),
        notes=(
            "(key) Registered rather than omitted because it is the purest "
            "NODE-LEVEL analogue of our own head anywhere in this folder: given a "
            "node's neighbourhood and which neighbours are active, predict whether "
            "it activates. That is exactly `p_new(v)`, and a one-step accuracy "
            "comparison against our structured head on real cascades would be a "
            "genuinely interesting experiment that neither this task's contract nor "
            "its metric layer currently supports. Worth its own microscopic task "
            "entry rather than a forced fit into this one (1.2)."
        ),
    ),
    "forest": ExternalBaseline(
        name="forest",
        kind=learned,
        title="FOREST: multi-scale reinforced diffusion prediction",
        venue="IJCAI 2019",
        repo="https://github.com/albertyang33/FOREST",
        paper="https://www.ijcai.org/proceedings/2019/560",
        entry="train.py",
        task="cascade_prediction",
        status="blocked",
        blocker=(
            "MICROSCOPIC, same category difference as `topolstm` and `deepinf`: it "
            "decodes the NEXT ADOPTER sequentially and is scored with Hits@k / "
            "MAP@k. Its macroscopic size reward is an RL SIGNAL inside training, not "
            "an output — verified by reading `train.py` and `model.py` at HEAD — so "
            "there is still no per-cascade popularity to read back. The code is "
            "PyTorch and Python 3; the output type is the blocker."
        ),
        notes=(
            "The one microscopic method that also optimizes a macroscopic objective, "
            "which makes it the natural first entry if a next-adopter task is ever "
            "added. Reports on 6.2's Digg / Twitter-URL / Memes triple (7), all "
            "three of which we now load."
        ),
    ),
    "ms_hgat": ExternalBaseline(
        name="ms_hgat",
        kind=learned,
        title="MS-HGAT: memory-enhanced sequential hypergraph attention",
        venue="AAAI 2022",
        repo="https://github.com/slingling/MS-HGAT",
        paper="https://ojs.aaai.org/index.php/AAAI/article/view/20334",
        entry="run.py",
        task="cascade_prediction",
        status="blocked",
        blocker=(
            "MICROSCOPIC, same as `topolstm` and `forest`. Additionally, CTCP's own "
            "Table 2 records it as OOM on Weibo and APS at that paper's scale "
            "[verified, 5.2] — so even with a contract it would not complete on two "
            "of the three standard corpora."
        ),
        notes=(
            "The hypergraph entry: user-cascade interaction as a hyperedge, with a "
            "memory of past cascades. Present in 5.2's table as the only row with "
            "OOM cells, which is itself the useful datum — it is the scale ceiling "
            "of the microscopic line."
        ),
    ),
    "casseqgcn": ExternalBaseline(
        name="casseqgcn",
        kind=learned,
        title="CasSeqGCN: combining network structure and temporal sequence",
        venue="ESWA 2021",
        repo="https://github.com/MrYansong/CasSeqGCN",
        paper="https://arxiv.org/abs/2110.06836",
        entry="main.py",
        task="cascade_prediction",
        status="blocked",
        blocker=(
            "ITS INPUT IS AN UNDOCUMENTED PER-SNAPSHOT JSON BUNDLE SHIPPED AS A "
            ".RAR. Verified at HEAD: `param_parser.py` takes `--graph-folder` with a "
            "fixed `--number-of-nodes` (200 for its synthetic data, 100 for "
            "Weibo/Digg) and `--sub_size`, and the only example data is "
            "`synthetic_data_V2.rar` — an archive format `zipfile`/`tarfile` cannot "
            "read and which needs a non-stdlib `unrar` binary. The per-graph JSON "
            "schema is not documented anywhere in the repo, so an exporter would be "
            "reverse-engineered from an archive we cannot open. The code itself is "
            "PyTorch and Python 3."
        ),
        notes=(
            "A cheaper CasCN: the node STATE varies across snapshots rather than "
            "the structure, which is a modelling simplification worth comparing "
            "against ours — our own state is exactly two channels varying over a "
            "fixed graph. Not in any of 5's comparison tables."
        ),
    ),
    "ccasgnn": ExternalBaseline(
        name="ccasgnn",
        kind=learned,
        title="CCasGNN: collaborative cascade prediction with graph neural networks",
        venue="CSCWD 2022",
        repo="https://github.com/MrYansong/CCasGNN",
        paper="https://arxiv.org/abs/2112.03644",
        entry="main.py",
        task="cascade_prediction",
        status="blocked",
        blocker=(
            "The same undocumented per-snapshot JSON input as its sibling "
            "`casseqgcn` (same authors, same `param_parser.py` shape), with no "
            "example data shipped at all. Same blocker, one step worse."
        ),
        notes=(
            "GAT and GCN with positional encoding, fused in sequence. Catalogued "
            "from 4.1; not in any published comparison table this review could "
            "extract."
        ),
    ),
    "casft": ExternalBaseline(
        name="casft",
        kind=learned,
        title="CasFT: neural-ODE dynamic cues with a conditional diffusion decoder",
        venue="AAAI 2025",
        repo="no public release located",
        paper="https://arxiv.org/abs/2409.16619",
        entry="none",
        task="cascade_prediction",
        status="blocked",
        blocker=(
            "NO PUBLIC CODE. Searched GitHub and the paper's own text; no release "
            "was located [verified, 2026-08-05]. Registered rather than omitted "
            "because it is the current best method on the standard protocol and a "
            "reader will look for it."
        ),
        notes=(
            "(key) The source of 5.1, which is the widest baseline set in this "
            "literature and the only recent result table that survives "
            "`pdftotext` — every published number our report prints as a context "
            "marker comes from its Table 2. Its own rows are MSLE 2.1728 (Weibo "
            "0.5h) / 3.8546 (Twitter 1d) / 1.2468 (APS 3y) [verified]. Warning: it "
            "regresses TOTAL popularity (its Eq. 26) while CasFlow regresses the "
            "INCREMENT, and 5.1 prints both in one table — 5.7 difference 3, and "
            "the reason `--cp-target` exists."
        ),
    ),
    "casdo": ExternalBaseline(
        name="casdo",
        kind=learned,
        title="CasDO: probabilistic diffusion denoiser with a neural ODE",
        venue="TKDE 2024",
        repo="no public release located",
        paper="https://doi.org/10.1109/TKDE.2024.3465241",
        entry="none",
        task="cascade_prediction",
        status="blocked",
        blocker=(
            "NO PUBLIC CODE that this review could locate, despite being a 2024 "
            "TKDE paper and despite CasTemp having re-run it — 11 records that "
            "CasTemp presumably obtained it privately. GitHub search returned "
            "nothing [verified]."
        ),
        notes=(
            "Registered for one reason, and it is a cautionary one: under CasTemp's "
            "leak-free split CasDO is the WORST method on all four corpora and falls "
            "below a plain MLP on the diagnostic scenarios [verified, 5.3], despite "
            "being the most recent published architecture in that comparison. 8.3 is "
            "the lesson and this row is the evidence."
        ),
    ),
    "greedywalk": ExternalBaseline(
        name="greedywalk",
        kind=classical,
        title="GreedyWalk / PrimalDual: approximation algorithms for spectral-radius minimization",
        venue="SDM 2015",
        repo="http://tinyurl.com/l3lgsq7",
        paper="https://arxiv.org/abs/1501.06614",
        entry="MATLAB (per the authors)",
        task="epidemic_control",
        status="blocked",
        blocker=(
            "MATLAB, AND THE LINK IS A TINYURL PRINTED IN THE PDF. §11 records the "
            "URL as unverified; independently, `allogn/Network-Immunization`'s own "
            "README states that its Walk8 solver 'is not available as the code was "
            "provided by authors of Scalable Approximation Algorithm for Network "
            "Immunization and the algorithm is implemented in MATLAB' [verified, "
            "2026-08-05]. Two sources agree the artifact is MATLAB and not publicly "
            "downloadable. Our `greedy_walk` / `greedy_walk_edge` are "
            "reimplementations from the paper's prose and are labelled as such."
        ),
        notes=(
            "(key) Its Table 2 is the SPECTRAL BENCHMARK DEFINITION (§5.2) and is why "
            "`oregon1`, `oregon2_010331`, `brightkite`, `p2p_gnutella05` and "
            "`p2p_gnutella06` are loaded: it publishes `lambda_1` per graph, so four "
            "of those are checkable against somebody else's arithmetic. All four "
            "reproduce to the decimal [derived, 2026-08-05] — Oregon-1 58.72, "
            "Oregon-2 70.74, Brightkite 101.49, and the `youtube` we already load at "
            "210.4. Its own RESULT cells are [figure]-only."
        ),
    ),
}


# Four already-wired NODE-REMOVAL repos re-registered under `epidemic_control`.
# Applied AFTER the dict literal rather than inside it, because each one COPIES
# its critical-node twin's entry and that twin has to exist first.
# research/epidemic_control.md §2.5 maps vaccination onto `remove_node`, so a
# dismantler's output IS an allocation under the `vaccinate` lever; §4.2 lists
# FINDER under this task explicitly. Each shares its twin's clone, venv, build
# and adapter, so one install serves both tasks and a fix to one is a fix to both.
external_baselines |= {
    "finder_epi": _shared_dismantler_entry(
        "finder_epi",
        "finder",
        "FINDER, run as a vaccination allocation",
        "The strongest LEARNED node-removal baseline, scored here on the simulated "
        "attack rate rather than on the structural objective it was trained for. "
        "That gap is the point: §4.2 lists FINDER under this task while noting its "
        "objective is structural, and §8.2 trap 1 says a method can win the "
        "connectivity metric and lose the epidemic one. Running it under BOTH tasks "
        "and reading the two rows against each other is a comparison neither "
        "literature currently makes.",
    ),
    "collective_influence_epi": _shared_dismantler_entry(
        "collective_influence_epi",
        "collective_influence",
        "Collective Influence, run as a vaccination allocation",
        "Morone & Makse's optimal percolation, which their own paper frames as "
        "IMMUNIZATION rather than as dismantling — the Nature abstract is about "
        "immunizing the minimal set that fragments a network. Registering it here "
        "as well as under critical node detection is closer to the authors' own "
        "framing than either task alone.",
    ),
    "explosive_immunization_epi": _shared_dismantler_entry(
        "explosive_immunization_epi",
        "explosive_immunization",
        "Explosive Immunization, run as a vaccination allocation",
        "Literally an immunization paper (Clusella et al., PRL 117:208301), and the "
        "one physics method §9.4's prediction turns on: MIND's Table 5 puts EI at "
        "80.6 on `eu-powergrid` against FINDER's 161.7, so a hand-built heuristic "
        "beating a learned method by 2x on mesh graphs is the limitation we predict "
        "and should confirm rather than discover. Warning: its output flag is "
        "INVERTED relative to its README and it emits a SET at its own percolation "
        "threshold rather than a removal order — see the critical-node entry.",
    ),
    "gdm_epi": _shared_dismantler_entry(
        "gdm_epi",
        "gdm",
        "Graph Dismantling with Machine learning, run as a vaccination allocation",
        "The second LEARNED allocation beside `finder_epi`, and the one "
        "research/critical_node_detection.md §9.5's self-measurement was written "
        "for: MIND found GDM's removal order correlates at 0.762 with a PCA of its "
        "own handcrafted input features. We feed `log1p(degree)` as a channel, so "
        "`degree_rank_spearman` near that value means a method re-derived the "
        "degree heuristic with extra steps — which is exactly what §9.4 predicts "
        "for this task too. Running it here as well as under critical node "
        "detection is what makes the prediction checkable on both objectives.",
    ),
    "dismantling_review_epi": _shared_dismantler_entry(
        "dismantling_review_epi",
        "dismantling_review",
        "NetworkDismantling review harness, run as a vaccination allocation",
        "The Artime et al. survey harness, which already wires CI, CoreHD, GND, EI, "
        "MinSum, FINDER and GDM behind one interface. Wiring THIS rather than the "
        "seven repos separately is the right trade under either task, and it is also "
        "the only practical route to a FINDER number given §11's note that FINDER "
        "publishes no table.",
    ),
}


def available_baselines(
    kind: str | None = None, task: str | None = None
) -> list[str]:
    """Registered, not blocked, and with an adapter that can drive it."""
    return sorted(
        name
        for name, spec in external_baselines.items()
        if spec.status != "blocked"
        and spec.wired
        and (kind is None or spec.kind == kind)
        and (task is None or spec.task == task)
    )


def runnable_baselines(task: str | None = None) -> list[str]:
    """...and actually present on disk."""
    return sorted(
        name
        for name in available_baselines(task=task)
        if external_baselines[name].installed()
    )


def baselines_for_task(task: str) -> list[str]:
    """Every registered baseline for one task, blocked ones included."""
    return sorted(
        name for name, spec in external_baselines.items() if spec.task == task
    )


def unwired_baselines() -> list[str]:
    """Cloned-but-undriveable: registered, not blocked, no adapter."""
    return sorted(
        name
        for name, spec in external_baselines.items()
        if spec.status != "blocked" and not spec.wired
    )
