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
import sys
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
    # Some repos return non-zero on SUCCESS. EI calls exit() once it reaches
    # its percolation threshold, which is its normal termination; OPIM's
    # format step ends with `return 1`. The artifact is the real signal, so
    # these opt out of the exit-code check rather than the check being dropped.
    allow_nonzero_exit: bool = False
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

    ⚠️ Upstream hardcodes the node count, the filenames and the stopping threshold
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
    ⚠️ EI answers a different question than a fixed budget asks.

    `threshold_conditions.dat` is `<node> <flag>` for every node: the vaccination
    state at EI's approximate percolation threshold (1/sqrt(N)), with no ordering
    and no way to ask for k.

    ⚠️ **The flag is inverted relative to the README**, which states "A 1 means
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
# ⚠️ EVERY ADAPTER BELOW IS UNTESTED. They were written from the cloned source —
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

    ⚠️ The clone ships no trained weights, so this needs `--weights` pointing at a
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

    ⚠️ The clone ships NO trained weights — only `train_synthetic.py` /
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

    ⚠️ The repo is driven by shell scripts (`app_rl.sh`, `app_sr.sh`, ...) rather
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
            "CORRECTED 2026-08-01. This entry previously said 'no public code "
            "release found' and sat under influence_maximization; both were "
            "wrong. Code IS published by a co-author at github.com/smaniu/oim "
            "(research/adaptive_online_im.md §3.2), and the method is an "
            "online/bandit IM algorithm, so it belongs to this task. NOT WIRED: "
            "OIM's own protocol is a SEQUENCE of campaigns reporting the UNION "
            "of activated nodes across trials, which §8.2 trap 2 shows is not "
            "comparable to a single-campaign spread: the number grows "
            "monotonically in the trial count. Driving it as a condition-7 arm "
            "needs the repeated-campaign loop this task does not have (see the "
            "regret entry in §11). DeepIM sidestepped this by re-running OIM "
            "under its own single-campaign protocol; those numbers are "
            "transcribed in research/influence_maximization.md §5.1."
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
        entry="C++ (make)",
        task="adaptive_online_im",
        status="needs_setup",
        requirements=None,
        build=["make"],
        notes=(
            "THE reference implementation for this task, by the authors. Our "
            "library's `adapt_greedy` and `adapt_epic` are Python "
            "reimplementations of AdaptGreedy and its RIS instantiation, good "
            "enough as condition-1 arms, but this is the original and the one "
            "whose figures §5.2 describes. Its two sweeps are the b-setting "
            "(k=500 fixed, b in {1,2,5,10,20,50,500}) and the k-setting (r=50 "
            "fixed), which map onto our --per-round-budget and --rounds "
            "respectively. TO WIRE: clone, make, read its graph format and the "
            "flag that sets b, and confirm where the per-round seed sets are "
            "written. Note that the paper reports AdaptIM-1 running out of "
            "memory at b<5 on LiveJournal and Orkut, so expect the same."
        ),
    ),
    "mrim": ExternalBaseline(
        name="mrim",
        kind=classical,
        title="MRIM: Multi-Round Influence Maximization",
        venue="KDD 2018",
        repo="https://github.com/lichao-sun/Multi-Round-Influence-Maximization",
        paper="https://arxiv.org/abs/1802.04189",
        entry="see repo",
        task="adaptive_online_im",
        status="needs_setup",
        notes=(
            "r separate campaigns of k seeds each, non-adaptive and adaptive "
            "variants, both with approximation guarantees. Distinct from "
            "adaptive IM in our sense: MRIM's rounds are separate DIFFUSIONS "
            "whose union is scored, whereas ours is one diffusion observed in "
            "stages (§1.5). Comparing the two needs the multi-round state "
            "bookkeeping §2.4e describes (reset `frontier` between campaigns, "
            "keep `infected` as the union), which is not built. §11 also records "
            "that this paper's dataset and result tables were never extracted, "
            "so there is no published number to check an adapter against yet."
        ),
    ),
    "rl4im": ExternalBaseline(
        name="rl4im",
        kind=learned,
        title="RL4IM: contingency-aware IM as a multi-round MDP",
        venue="UAI 2021",
        repo="https://github.com/wmd3i/RL4IM-Contingency",
        paper="https://arxiv.org/abs/2106.07039",
        entry="see repo",
        task="adaptive_online_im",
        status="needs_setup",
        notes=(
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
            "are figure-only, so there is no table to validate against."
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
            "Same blocker as `oim`: bandit IM is a repeated-campaign regret "
            "setting, not single-campaign spread, so it needs the loop §11 "
            "records as missing. Worth registering now because it is the only "
            "online-IM code we found that targets LT at all, and our LT arms "
            "otherwise have no published online baseline."
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
            "⚠️ THIRD-PARTY, not the authors'. Wen et al. released no code; this "
            "is a temporal port found by the review (§3.2), so any number it "
            "produces is attributable to this repo and not to the paper, so say so "
            "if it is ever reported. Same repeated-campaign blocker as the other "
            "bandit entries. Scale note from §5.4: the strongest theory result in "
            "bandit IM was validated on a 327-node Facebook subgraph, and §6.2 "
            "records that the exact subgraph is unpublished, so its own figure "
            "cannot be reproduced regardless."
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
    # ⚠️ Two shared traps before any of these is wired (§8.2): almost all of them
    # run on the LARGEST CONNECTED COMPONENT of the input silently (trap 7), and
    # several ship a REINSERTION pass that makes `X` and `X+R` different methods
    # cited under one name (trap 2).
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
            "format is. ⚠️ TensorFlow 1.x and Cython pin this to Python 3.7, which "
            "is why it is the most expensive entry here to install. ⚠️ §11 records "
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
            "and which of MIND-AM / MIND-MP the entry point drives. ⚠️ Its table is "
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
        # ⚠️ Each `new` carries a /*gwm*/ marker so it is NOT a substring of its
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
            "⚠️ Its contribution is COST-weighted dismantling, so comparing its "
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
            "⚠️ §11 records that its real-network table does not exist — the PNAS "
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
            "⚠️ TWO TRAPS, both verified by running it — read _ei_parse before "
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
    # ⚠️ Two shared traps (§8.4): the SOURCE FRACTION is not standardized (10%
    # uniform-random in SL-VAE, first 5% by infection time in SL-Diff, top 10% by
    # influence time in SIDSL), and NOTHING in this literature evaluates under IC
    # or LT — SL-VAE uses SI/SIR, everything else uses real cascades. §11 calls
    # that the single biggest comparability gap in the file, and it is bigger than
    # any graph-version disagreement.
    "graphsl": ExternalBaseline(
        name="graphsl",
        kind=classical,
        title="GraphSL: a library for graph source localization",
        venue="JOSS 9(99):6796, 2024",
        repo="https://github.com/xianggebenben/GraphSL",
        paper="https://arxiv.org/abs/2405.03724",
        entry="pip install GraphSL",
        task="source_localization",
        status="needs_setup",
        fetch="git",
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
            "source set back. ⚠️ An adapter alone is NOT enough: "
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
        status="needs_setup",
        notes=(
            "The seed paper, and the method our arm A reimplements against our own "
            "likelihood. Worth wiring the ORIGINAL anyway: §2.2's whole argument is "
            "that swapping the forward model is a no-op the paper already "
            "published, and the way to show that rather than assert it is to run "
            "both. Its Table 1/2 are the ⭐ comparable tables (§5.1) — five of its "
            "seven graphs are ours — but read §5.1's column-order warning first: "
            "Table 1 is RE·PR·F1·AUC and Table 2 is PR·RE·F1·AUC, and a summarizer "
            "that assumes one ordering transposes precision and recall for a whole "
            "table. ⚠️ Its Network Science is 1,565 / 13,532, which is NOT ours and "
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
        status="needs_setup",
        notes=(
            "The near-miss on §1's amortization axis: its inversion is a single "
            "backward pass, but the validity-aware projection layers are an "
            "unrolled per-instance optimization, so its inference cost still scales "
            "with an inner loop. `pretrain.py` exists purely so `main.py` has "
            "something to invert, which is the cleanest statement in the literature "
            "that the forward model is a component. ⚠️ Its Table 3 numbers are far "
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
        status="needs_setup",
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
        status="needs_setup",
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
        status="needs_setup",
        notes=(
            "Not a method — the only reproducible THIRD-PARTY evaluation in this "
            "literature, and the only one authored by none of the method groups. "
            "SIR, SINGLE-source, top-k accuracy on six contact networks against "
            "Jordan centre, betweenness, SME and MCMF. ⚠️ Its framing is not ours: "
            "single-source ranking is a different problem from multi-source "
            "classification and the two never mix (§8.2), so its numbers cannot "
            "join our table without a dedicated `--budgets 1` run. Its value is the "
            "CEILING it establishes — the best GNN reaches 72.9% top-5 on a "
            "34-node graph where random already gets 39.4% — which is the honest "
            "picture of how hard this problem is, against §5.2's near-perfect F1. "
            "Its Karate is 34 / 77, one edge fewer than `nx.karate_club_graph()`."
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
        status="needs_setup",
        fetch="git",
        notes=(
            "The centrality-style classical estimators, including the rumor "
            "centrality reimplementation there is no first-party code for. Overlaps "
            "our own `localization_algorithms` pool by design — wiring it is a "
            "CROSS-CHECK on our implementations rather than new coverage, which is "
            "worth doing once for LPSI and rumor centrality specifically, since "
            "both are reimplemented from prose here. ⚠️ Several secondary sources "
            "cite a `qwertyjl/cosasi` URL that 404s; the repo above is the one its "
            "JOSS paper names."
        ),
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
