"""
Condition 9: published LLM algorithm-discovery systems as external baselines.

Every framework here (OpenEvolve, CodeEvolve, LLaMEA, EoH, ReEvo, MCTS-AHD,
LLM4AD's FunSearch and HillClimb, DeepEvolve) runs its OWN loop, its own prompts
and its own defaults inside its own venv. What we supply is the problem: a task
statement, an initial program, and a fitness function. The fitness function is
one subprocess, `baselines/score_program.py`, run under OUR interpreter, which
wraps the candidate program as a canned Strategy and scores it on the plain Monte
Carlo simulator through the identical `run_experiment` path every other arm uses.
Their venvs never import this package, so the world model is unreachable by
construction rather than by policy.

The artefact that crosses the boundary is a PROGRAM rather than a seed set, and
`program_script` is what turns it back into a Strategy the pipeline can replay on
the ground-truth referee. One function contract per task, and the same statement
and initial program are handed to every framework, so a difference between two
rows is a difference between two search loops and nothing else.
"""

import json
import os
import sys
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from urllib.parse import urlparse

discovery = "discovery"
# `task` value of a registry entry that serves every task
any_task = "*"
context_filename = "discovery_context.json"
scoring_filename = "discovery_scoring.py"
launcher_filename = "launch.py"
best_program_filename = "best_program.py"
external_root = Path(__file__).resolve().parent / "external"
repo_root = Path(__file__).resolve().parents[1]

# Seconds one candidate scoring subprocess may take: the executor's own 300 s cap
# on a plan_horizon() call plus the Monte Carlo rollouts behind it
candidate_timeout_seconds = 900
# Failure fitness. Every task's fitness is non-negative (see `oriented_fitness`),
# so zero is the floor and a crashed or timed-out candidate can never outrank a
# valid one under a minimizing task
failure_fitness = 0.0
base_url_env = "GATEWAY_BASE_URL"
# Frameworks that truncate at their own default token budget lose long programs
max_completion_tokens = 16000

allowed_imports_text = (
    "numpy, networkx, scipy, math, random, statistics, heapq, bisect, "
    "collections, itertools, functools"
)


@dataclass()
class Contract:
    function: str
    signature: str
    description: str
    initial_program: str
    reward_name: str


contracts = {
    "influence_maximization": Contract(
        function="select_seeds",
        signature="def select_seeds(graph, k):",
        description=(
            "Influence maximization. Choose k seed nodes to activate at t=0 so that "
            "the expected number of nodes the cascade eventually activates is as "
            "LARGE as possible. Return a list of k distinct node ids."
        ),
        reward_name="expected final spread (higher is better)",
        initial_program='''\
import networkx as nx


def select_seeds(graph, k):
    """Pick the k highest-degree nodes."""
    ranked = sorted(graph.nodes(), key=lambda node: graph.degree(node), reverse=True)
    return [int(node) for node in ranked[:k]]
''',
    ),
    "critical_node_detection": Contract(
        function="select_removals",
        signature="def select_removals(graph, k, outbreak):",
        description=(
            "Critical node detection (reactive containment). An outbreak has "
            "already started from the nodes in `outbreak`. Choose k nodes to DELETE "
            "from the graph (they can neither transmit nor be infected) so that the "
            "expected number of nodes the outbreak eventually infects is as SMALL as "
            "possible. Outbreak sources may not be removed. Return a list of k "
            "distinct node ids not in `outbreak`."
        ),
        reward_name="expected final infected count (lower is better)",
        initial_program='''\
import networkx as nx


def select_removals(graph, k, outbreak):
    """Delete the k highest-degree nodes that are not outbreak sources."""
    banned = set(int(node) for node in outbreak)
    ranked = sorted(
        (node for node in graph.nodes() if node not in banned),
        key=lambda node: graph.degree(node),
        reverse=True,
    )
    return [int(node) for node in ranked[:k]]
''',
    ),
    "influence_blocking": Contract(
        function="select_blockers",
        signature="def select_blockers(graph, k, rumour, lever):",
        description=(
            "Influence blocking. A rumour cascade has already started from the "
            "nodes in `rumour` and is spreading. You have k units of budget to "
            "answer it, and `lever` says what one unit buys: 'counter_seed' seeds a "
            "competing positive cascade at a node, 'node_block' deletes a node, "
            "'edge_block' deletes an arc (u, v), 'weight_block' sets an arc's "
            "transmission probability to zero. Minimize the rumour's expected final "
            "size. Return k distinct node ids for the node levers, or k distinct "
            "(u, v) arc pairs for the edge levers. Rumour nodes may not be chosen."
        ),
        reward_name="rumour's expected final size (lower is better)",
        initial_program='''\
import networkx as nx


def select_blockers(graph, k, rumour, lever):
    """Protect the rumour's highest-degree neighbours, or cut its arcs into them."""
    banned = set(int(node) for node in rumour)
    if lever in ("edge_block", "weight_block"):
        arcs = []
        for source in banned:
            for target in graph.neighbors(source):
                if target not in banned:
                    arcs.append((graph.degree(target), int(source), int(target)))
        arcs.sort(reverse=True)
        return [(source, target) for _, source, target in arcs[:k]]
    ring = {
        target
        for source in banned
        for target in graph.neighbors(source)
        if target not in banned
    }
    ranked = sorted(ring, key=lambda node: graph.degree(node), reverse=True)
    rest = sorted(
        (node for node in graph.nodes() if node not in banned and node not in ring),
        key=lambda node: graph.degree(node),
        reverse=True,
    )
    return [int(node) for node in (ranked + rest)[:k]]
''',
    ),
    "epidemic_control": Contract(
        function="select_doses",
        signature="def select_doses(graph, k, outbreak, lever):",
        description=(
            "Epidemic control. A compartmental epidemic (nodes recover, and under "
            "SIS become susceptible again) has already started from the index cases "
            "in `outbreak`. You have k units of budget and `lever` says what one "
            "unit buys: 'vaccinate' makes a node immune, 'quarantine' cuts every arc "
            "of a node, 'edge_cut' deletes an arc (u, v), 'contact_reduce' lowers an "
            "arc's transmission probability. Minimize the expected number of nodes "
            "ever infected. Return k distinct node ids for the node levers, or k "
            "distinct (u, v) arc pairs for the edge levers. Index cases may not be "
            "chosen."
        ),
        reward_name="expected attack size, nodes ever infected (lower is better)",
        initial_program='''\
import networkx as nx


def select_doses(graph, k, outbreak, lever):
    """Dose the outbreak's highest-degree neighbours, or cut the arcs into them."""
    banned = set(int(node) for node in outbreak)
    if lever in ("edge_cut", "contact_reduce"):
        arcs = []
        for source in banned:
            for target in graph.neighbors(source):
                if target not in banned:
                    arcs.append((graph.degree(target), int(source), int(target)))
        arcs.sort(reverse=True)
        return [(source, target) for _, source, target in arcs[:k]]
    ring = {
        target
        for source in banned
        for target in graph.neighbors(source)
        if target not in banned
    }
    ranked = sorted(ring, key=lambda node: graph.degree(node), reverse=True)
    rest = sorted(
        (node for node in graph.nodes() if node not in banned and node not in ring),
        key=lambda node: graph.degree(node),
        reverse=True,
    )
    return [int(node) for node in (ranked + rest)[:k]]
''',
    ),
    "source_localization": Contract(
        function="localize",
        signature="def localize(graph, observation, k):",
        description=(
            "Source localization. A cascade already happened; `observation` is a "
            "numpy array of length N giving, per node, the observed probability "
            "that it was infected (0 or 1 for a single realization). Recover the k "
            "nodes that STARTED the cascade. Scored by F1 against the true source "
            "set. Return a list of k distinct node ids."
        ),
        reward_name="F1 against the true sources (higher is better)",
        initial_program='''\
import networkx as nx


def localize(graph, observation, k):
    """Rank nodes by observed infection mass times degree."""
    scores = {
        node: float(observation[node]) * (1.0 + graph.degree(node))
        for node in graph.nodes()
    }
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [int(node) for node in ranked[:k]]
''',
    ),
    "cascade_reconstruction": Contract(
        function="reconstruct",
        signature="def reconstruct(graph, observation, horizon):",
        description=(
            "Cascade reconstruction. A cascade already happened and only part of it "
            "was observed. `observation` is a dict with keys 'reported' (observed "
            "node -> activation timestep, or None when the time is unknown), "
            "'infected' (every node the observation says was infected), 'times' "
            "(only the reports with a known time), 'setting', 'final_state' (a "
            "0/1 list over nodes under the final-snapshot setting, else None), "
            "'visible' (a 0/1 list of nodes present in the adjacency, else None), "
            "'num_nodes' and 'horizon'. Recover the whole hidden history: return a "
            "dict mapping every node you believe was infected to a pair "
            "(activation timestep, parent node id), with parent None for a source. "
            "Scored by a tree-weighted mix of path precision (who infected whom) and "
            "event F1 (which nodes, when)."
        ),
        reward_name="tree-weighted reconstruction score (higher is better)",
        initial_program='''\
import networkx as nx


def reconstruct(graph, observation, horizon):
    """Keep every reported node; parent = its earliest-activated in-neighbour."""
    times = {int(node): int(time) for node, time in observation["times"].items()}
    infected = [int(node) for node in observation["infected"]]

    def upstream(node):
        return graph.predecessors(node) if graph.is_directed() else graph.neighbors(node)

    for node in infected:
        if node not in times:
            earlier = [times[other] for other in upstream(node) if other in times]
            times[node] = (min(earlier) + 1) if earlier else 0
    history = {}
    for node in infected:
        candidates = [
            (times[other], other)
            for other in upstream(node)
            if other in times and times[other] < times[node]
        ]
        parent = int(min(candidates)[1]) if candidates else None
        history[node] = (times[node], parent)
    return history
''',
    ),
    "cascade_prediction": Contract(
        function="predict",
        signature="def predict(graph, observation):",
        description=(
            "Cascade popularity prediction on a REAL cascade log. `observation` is "
            "a dict with keys 'cascade_id', 'root', 'adopters' (node -> timestep it "
            "adopted, for every adopter inside the observation window), 'frontier' "
            "(the last observed wave), 'observed_steps', 'horizon' (the prediction "
            "step), 'publish_time', 'popularity' (adopters observed so far) and "
            "'num_nodes'. Predict how many nodes will have adopted by the horizon. "
            "Scored by mean squared log error against the logged truth. Return a "
            "float, or None to decline this cascade."
        ),
        reward_name="MSLE against the logged popularity (lower is better)",
        initial_program='''\
import networkx as nx


def predict(graph, observation):
    """Grow the observed popularity by a constant factor."""
    return float(observation["popularity"]) * 1.5
''',
    ),
}
# Adaptive IM runs the same static contract: the program commits its k seeds at
# t=0 and the harness scores that as a one-shot plan, which is the non-adaptive
# side of the adaptivity gap every adaptive arm is compared against
contracts["adaptive_online_im"] = contracts["influence_maximization"]

edge_levers = ("edge_block", "weight_block", "edge_cut", "contact_reduce")


def token_env(model: str) -> str:
    """Same rule as coding_agent.agent.GatewayProvider: the token follows the model family."""
    return "CLAUDE_GATEWAY_TOKEN" if model.startswith("claude") else "CHATGPT_GATEWAY_TOKEN"


def lever_for(task: str, experiment: dict) -> str | None:
    if task == "influence_blocking":
        return experiment["blocking_lever"]

    if task == "epidemic_control":
        return experiment["epi_lever"]

    return None


def call_arguments(task: str, lever: str | None) -> str:
    """The argument list the wrapper passes to the discovered function."""
    if task in ("critical_node_detection",):
        return "nx_graph, budget, list(self.outbreak)"

    if task in ("influence_blocking", "epidemic_control"):
        return f"nx_graph, budget, list(self.outbreak), {lever!r}"

    return "nx_graph, budget"


def oriented_fitness(task: str, sense: str, reward: float, num_nodes: int) -> float:
    """
    Non-negative, higher-is-better fitness for every task.

    Every framework here either maximizes or asserts a positive objective, and a
    failed candidate has to sit BELOW every valid one. A minimizing spread task
    reports nodes saved and cascade prediction reports 1/(1+MSLE), so zero is the
    floor everywhere and the raw reward travels beside it under its own name.
    """
    if sense == "maximize":
        return max(0.0, float(reward))

    if task == "cascade_prediction":
        return 1.0 / (1.0 + max(0.0, float(reward)))

    return max(0.0, float(num_nodes) - float(reward))


# The wrapper is a plain string rather than an f-string: it is full of braces and
# the only values spliced in are repr()'d, so placeholders keep it readable
_wrapper_template = '''

import networkx as nx
import numpy as np


def _discovery_graph(graph):
    built = nx.DiGraph() if graph.directed else nx.Graph()
    built.add_nodes_from(range(graph.num_nodes))
    sources, targets = graph.edge_index
    for source, target, weight in zip(
        sources.tolist(), targets.tolist(), graph.ic_probs.tolist()
    ):
        built.add_edge(int(source), int(target), weight=float(weight))
    return built


def _discovery_function(name):
    # ReEvo and MCTS-AHD name the evolved function <name>_v2; take the newest
    found = globals().get(name)
    version = -1
    for key, value in list(globals().items()):
        suffix = key[len(name) + 2 :]
        if key.startswith(name + "_v") and suffix.isdigit() and callable(value):
            if int(suffix) > version:
                found, version = value, int(suffix)
    if found is None or not callable(found):
        raise ValueError(
            "discovered program defines no function named " + repr(name)
        )
    return found


def _discovery_nodes(picks, graph, budget, banned):
    chosen = []
    for node in picks:
        node = int(node)
        if node in chosen or node in banned or not (0 <= node < graph.num_nodes):
            continue
        chosen.append(node)
        if len(chosen) == budget:
            break
    return chosen


def _discovery_arcs(picks, graph, budget):
    chosen = []
    for pair in picks:
        source, target = int(pair[0]), int(pair[1])
        arc = (source, target)
        if arc in chosen or source == target:
            continue
        if not (0 <= source < graph.num_nodes and 0 <= target < graph.num_nodes):
            continue
        chosen.append(arc)
        if len(chosen) == budget:
            break
    return chosen


class DiscoveredStrategy(Strategy):
    def _graph(self, graph):
        cached = getattr(self, "_nx_graph", None)
        if cached is None:
            cached = _discovery_graph(graph)
            self._nx_graph = cached
        return cached

    def plan_horizon(self, graph, budget, horizon):
        nx_graph = self._graph(graph)
        picks = _discovery_function(__FUNCTION__)(__ARGUMENTS__)
        if __EDGES__:
            arcs = _discovery_arcs(picks, graph, budget)
            bag = [ActionOp(__BUDGET_OP__, u, v__WEIGHT__) for u, v in arcs]
        else:
            nodes = _discovery_nodes(picks, graph, budget, set(self.outbreak))
            bag = [ActionOp(__BUDGET_OP__, node) for node in nodes]
        return [bag] + [[] for _ in range(horizon)]

    def localize(self, graph, observation, budget):
        nx_graph = self._graph(graph)
        picks = _discovery_function(__FUNCTION__)(
            nx_graph, np.asarray(observation, dtype=float), budget
        )
        return _discovery_nodes(picks, graph, budget, set())

    def reconstruct(self, graph, observation, horizon):
        nx_graph = self._graph(graph)
        plain = {
            "reported": {
                int(node): (None if time is None else int(time))
                for node, time in observation.reported.items()
            },
            "infected": [int(node) for node in observation.infected],
            "times": {int(node): int(time) for node, time in observation.times.items()},
            "setting": observation.setting,
            "final_state": (
                None
                if observation.final_state is None
                else [int(value >= 0.5) for value in np.asarray(observation.final_state)]
            ),
            "visible": (
                None
                if observation.visible is None
                else [int(bool(value)) for value in np.asarray(observation.visible)]
            ),
            "num_nodes": int(observation.num_nodes),
            "horizon": int(observation.horizon),
        }
        history = _discovery_function(__FUNCTION__)(nx_graph, plain, horizon)
        return {
            int(node): (int(entry[0]), None if entry[1] is None else int(entry[1]))
            for node, entry in dict(history).items()
        }

    def predict(self, graph, observation, horizon=None):
        nx_graph = self._graph(graph)
        plain = {
            "cascade_id": observation.cascade_id,
            "root": int(observation.root),
            "adopters": {int(node): int(when) for node, when in observation.adopters.items()},
            "frontier": [int(node) for node in observation.frontier],
            "observed_steps": int(observation.observed_steps),
            "horizon": int(observation.horizon),
            "publish_time": int(observation.publish_time),
            "popularity": int(observation.popularity),
            "num_nodes": int(observation.num_nodes),
        }
        value = _discovery_function(__FUNCTION__)(nx_graph, plain)
        return None if value is None else float(value)
'''


def program_script(
    program: str, task: str, budget_op: str = "add_node", lever: str | None = None
) -> str:
    """
    Wrap a discovered program as a canned Strategy.

    The program is spliced in verbatim and the wrapper below it calls the task's
    contract function, so the executor's import whitelist and timeout apply to the
    discovered code exactly as they apply to a generated one.
    """
    contract = contracts[task]
    edges = budget_op in ("remove_edge", "set_edge_weight", "add_edge")
    # Same convention as run_baseline.edge_script: a weight op blocks as p -> 0
    weight = ", 0.0" if budget_op == "set_edge_weight" else ""

    wrapper = (
        _wrapper_template.replace("__FUNCTION__", repr(contract.function))
        .replace("__ARGUMENTS__", call_arguments(task, lever))
        .replace("__EDGES__", repr(edges))
        .replace("__BUDGET_OP__", repr(budget_op))
        .replace("__WEIGHT__", weight)
    )

    return program.rstrip() + "\n" + wrapper


# Context: what the scorer and every glue file need ------------------------------


def build_context(
    experiment: object,
    task: str,
    budget_op: str,
    work_dir: Path,
    graph: object,
    timeout: int = candidate_timeout_seconds,
) -> dict:
    """
    Everything a framework's glue needs, minus secrets.

    `experiment` is the pipeline's ExperimentConfig for this arm: the scorer
    rebuilds it verbatim so the candidate is scored under the same task settings
    (outbreak, lever, splits, mc_runs) as the arm being compared against. Tokens
    never land here: the glue reads them from the environment by NAME.
    """
    experiment_dict = asdict(experiment)
    # The pipeline hands a percentage point over with a placeholder `budget` and
    # lets run_experiment resolve k; the statement the framework reads has to
    # carry the SAME k the scorer and the referee will use (same rule as
    # run_experiment: pct of N overrides the absolute k)
    if experiment_dict["budget_pct"] is not None:
        experiment_dict["budget"] = max(
            1, round(graph.num_nodes * experiment_dict["budget_pct"] / 100)
        )
    lever = lever_for(task, experiment_dict)
    model = experiment_dict["model"]

    return {
        "experiment": experiment_dict,
        "task": task,
        "budget_op": budget_op,
        "lever": lever,
        "python": sys.executable,
        "repo_root": str(repo_root),
        "work_dir": str(Path(work_dir).resolve()),
        "model": model,
        "base_url_env": base_url_env,
        "token_env": token_env(model),
        "eval_timeout": int(timeout),
        "seed": int(experiment_dict["seed"]),
        "graph": {
            "num_nodes": int(graph.num_nodes),
            "num_edges": int(graph.edge_index.shape[1]),
            "directed": bool(graph.directed),
        },
        "contract": asdict(contracts[task]),
        # A smoke run shrinks every framework's budget to a handful of samples so
        # the adapter can be exercised end to end without a day of LLM calls
        "smoke": os.environ.get("DISCOVERY_SMOKE") == "1",
    }


def write_context(work_dir: Path, context: dict) -> Path:
    os.makedirs(work_dir, exist_ok=True)
    path = Path(work_dir) / context_filename
    path.write_text(json.dumps(context, indent=2))

    return path


def read_context(work_dir: Path) -> dict:
    path = Path(work_dir) / context_filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing: a discovery baseline needs the pipeline to write "
            f"its context before export() runs (run_external_baseline(context=...))"
        )

    return json.loads(path.read_text())


def problem_statement(context: dict) -> str:
    """The task text every framework receives: identical across the nine rows."""
    contract = context["contract"]
    experiment = context["experiment"]
    graph = context["graph"]
    task = context["task"]
    kind = "directed graph (arcs)" if graph["directed"] else "undirected graph"
    lines = [
        contract["description"],
        "",
        f"Instance: one {kind} with N={graph['num_nodes']} nodes (ids 0..N-1) and "
        f"{graph['num_edges']} arcs, handed to you as a networkx "
        f"{'DiGraph' if graph['directed'] else 'Graph'} whose edge attribute "
        f"'weight' is the arc's transmission probability. Diffusion model: "
        f"{experiment['diffusion_model']}. Budget k={experiment['budget']}. "
        f"Horizon: {experiment['horizon']} timesteps.",
    ]

    if context["lever"] is not None:
        lines.append(f"Lever in this run: '{context['lever']}'.")

    if task == "source_localization":
        lines.append(
            f"Observation kind: {experiment['sl_observation']}. The source count k "
            f"is the instance's own."
        )
    elif task == "cascade_reconstruction":
        lines.append(
            f"Observation setting: {experiment['cr_setting']}; reported fraction "
            f"{experiment['cr_observation_rate']}; tree weight "
            f"{experiment['cr_tree_weight']} on path precision."
        )
    elif task == "cascade_prediction":
        lines.append(
            f"Metric: {experiment['cp_metric']} on the {experiment['cp_target']} target."
        )

    lines += [
        "",
        f"Fitness reported to you is non-negative and higher is better: the raw "
        f"quantity is the {contract['reward_name']}, and for a minimizing task the "
        f"fitness is the number of nodes saved (or 1/(1+error) for prediction). "
        f"Every candidate is scored by a fixed Monte Carlo simulator; it never "
        f"changes between candidates.",
        "",
        f"Rules: implement exactly `{contract['signature'][4:-1]}` in Python. Only "
        f"these modules may be imported: {allowed_imports_text}. No file, network "
        f"or subprocess access; do not use open(), eval() or exec(). Any helper "
        f"must be defined ABOVE the contract function, which must be the LAST "
        f"top-level function in the file. The call must finish within "
        f"{experiment['strategy_timeout']:.0f} seconds. Use only the arguments given; "
        f"there is no simulator you can query.",
    ]

    return "\n".join(lines)


def gateway_host(context: dict) -> str:
    """
    Host (and port) of the gateway, for frameworks whose HTTP client takes a bare host.

    EoH and LLM4AD's HttpsApi hard-code the path `/v1/chat/completions` and
    HTTPS, so they can only be pointed at a gateway whose base URL is exactly
    `https://<host>/v1`. Anything else is refused here rather than failing on the
    first request with a bare 404.
    """
    base_url = os.environ[context["base_url_env"]]
    parsed = urlparse(base_url)

    if parsed.scheme != "https" or parsed.path.rstrip("/") != "/v1":
        raise ValueError(
            f"{context['base_url_env']}={base_url!r} is not of the form "
            f"https://<host>/v1, which is the only shape this framework's HTTP "
            f"client can address (fixed path /v1/chat/completions over HTTPS)"
        )

    return parsed.netloc


# Files every framework gets ------------------------------------------------------

_scoring_template = '''\
"""Written by baselines/discovery.py: score one candidate program on our simulator."""

import json
import os
import subprocess
import time
import uuid

context_path = __CONTEXT__


def load_context():
    with open(context_path) as handle:
        return json.load(handle)


def score_program(program=None, program_path=None, tag=None):
    """
    Run baselines/score_program.py under the harness's own interpreter.

    Returns a dict with `ok`, `fitness` (non-negative, higher is better),
    `reward` (the task's raw quantity), `reward_name`, `sense`, `seconds`
    and `error`. Never raises: a broken candidate is a fitness of zero with the
    error text attached, and each framework's glue decides how to show that.
    """
    context = load_context()
    candidates = os.path.join(context["work_dir"], "candidates")
    os.makedirs(candidates, exist_ok=True)

    if program_path is None:
        stem = tag or uuid.uuid4().hex
        program_path = os.path.join(candidates, f"candidate_{stem}.py")
        with open(program_path, "w") as handle:
            handle.write(program)
    else:
        # The framework may overwrite its file before we finish (ReEvo and
        # MCTS-AHD rewrite gpt.py per candidate), so score a private copy
        with open(program_path) as handle:
            source = handle.read()
        copied = os.path.join(candidates, f"candidate_{tag or uuid.uuid4().hex}.py")
        with open(copied, "w") as handle:
            handle.write(source)
        program_path = copied

    out_json = program_path[:-3] + ".json"
    argv = [
        context["python"],
        "-m",
        "baselines.score_program",
        "--context",
        context_path,
        "--program",
        program_path,
        "--json-out",
        out_json,
    ]
    start = time.time()

    try:
        completed = subprocess.run(
            argv,
            cwd=context["repo_root"],
            capture_output=True,
            text=True,
            timeout=context["eval_timeout"],
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "fitness": 0.0,
            "reward": None,
            "reward_name": context["contract"]["reward_name"],
            "sense": None,
            "seconds": time.time() - start,
            "error": f"scoring exceeded {context['eval_timeout']} seconds",
        }

    seconds = time.time() - start

    if completed.returncode != 0 or not os.path.exists(out_json):
        tail = (completed.stderr or completed.stdout or "")[-3000:]
        return {
            "ok": False,
            "fitness": 0.0,
            "reward": None,
            "reward_name": context["contract"]["reward_name"],
            "sense": None,
            "seconds": seconds,
            "error": tail.strip() or f"scorer exited {completed.returncode}",
        }

    with open(out_json) as handle:
        data = json.load(handle)
    data["ok"] = True
    data["seconds"] = seconds
    data["error"] = None

    return data
'''


def write_scoring_glue(directory: Path, context_path: Path) -> Path:
    os.makedirs(directory, exist_ok=True)
    path = Path(directory) / scoring_filename
    path.write_text(_scoring_template.replace("__CONTEXT__", repr(str(context_path))))

    return path


_launcher_template = '''\
"""Written by baselines/discovery.py: set the framework's environment and exec it."""

import os
import sys

env_map = __ENV_MAP__
static_env = __STATIC_ENV__
path_prepend = __PATH_PREPEND__
program = __PROGRAM__
argv = __ARGV__
cwd = __CWD__

for target, source in env_map.items():
    if source not in os.environ:
        raise KeyError(
            f"{source} is not set: the gateway credentials come from the .env file "
            f"and this baseline maps {source} onto {target}"
        )
    os.environ[target] = os.environ[source]

os.environ.update(static_env)
os.environ["PATH"] = os.pathsep.join(path_prepend + [os.environ.get("PATH", "")])
os.chdir(cwd)
sys.stdout.flush()
os.execv(program or sys.executable, [program or sys.executable] + argv)
'''


def write_launcher(
    work_dir: Path,
    argv: list[str],
    cwd: Path,
    env_map: dict | None = None,
    static_env: dict | None = None,
    venv_bin: Path | None = None,
    program: str | None = None,
) -> Path:
    """
    The one command every discovery entry runs: `python launch.py`.

    Secrets stay out of argv (which run_external_baseline prints) because the
    launcher copies them between environment variables by name, and `python` is
    the framework's own venv interpreter because run_external_baseline swaps it in.
    """
    path = Path(work_dir) / launcher_filename
    path.write_text(
        _launcher_template.replace("__ENV_MAP__", repr(env_map or {}))
        .replace("__STATIC_ENV__", repr(static_env or {}))
        .replace("__PATH_PREPEND__", repr([str(venv_bin)] if venv_bin else []))
        .replace("__PROGRAM__", repr(program))
        .replace("__ARGV__", repr([str(item) for item in argv]))
        .replace("__CWD__", repr(str(cwd)))
    )

    return path


def framework_root(name: str) -> Path:
    return external_root / name


def venv_bin(name: str) -> Path:
    return framework_root(name) / ".venv" / "bin"


def run_uid(work_dir: Path) -> str:
    """
    A filesystem- and hydra-safe name unique to this (task, dataset, run, budget).

    ReEvo and MCTS-AHD write the problem INTO their checkout and overwrite
    `problems/<name>/gpt.py` on every candidate, so two concurrent jobs sharing
    one name would score each other's programs.
    """
    parts = Path(work_dir).resolve().parts
    # .../<task>/<dataset>/<run>/baselines/_runs/<framework>/<label>
    tail = parts[-7:-4] + parts[-1:] if len(parts) >= 7 else parts[-2:]
    raw = "_".join(tail)

    return "d_" + "".join(char if char.isalnum() else "_" for char in raw).lower()


def launch_command(work_dir: Path, *_args) -> list[str]:
    """Registry `command`: the launcher the export wrote."""
    return ["python", str(Path(work_dir) / launcher_filename)]


def _smoke(context: dict) -> bool:
    return bool(context.get("smoke"))


def _read_best(path: Path, name: str) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"{name} finished but wrote no best program at {path}; read its logs "
            f"in {path.parent}"
        )

    program = path.read_text()
    if not program.strip():
        raise ValueError(f"{name} wrote an empty best program at {path}")

    return program


# OpenEvolve ---------------------------------------------------------------------

_openevolve_evaluator = '''\
"""Written by baselines/discovery.py for OpenEvolve."""

from openevolve.evaluation_result import EvaluationResult

from discovery_scoring import score_program


def evaluate(program_path):
    scored = score_program(program_path=program_path)
    metrics = {
        "combined_score": float(scored["fitness"]),
        "raw_reward": float(scored["reward"]) if scored["reward"] is not None else 0.0,
        "eval_seconds": float(scored["seconds"]),
    }
    artifacts = {}
    if not scored["ok"]:
        artifacts["error"] = scored["error"]
    else:
        artifacts["reward"] = f"{scored['reward_name']}: {scored['reward']}"
    return EvaluationResult(metrics=metrics, artifacts=artifacts)
'''


def _openevolve_export(graph, work_dir: Path, budget: int, diffusion_model: str) -> dict:
    import yaml

    context = read_context(work_dir)
    context_path = Path(work_dir) / context_filename
    contract = context["contract"]
    root = framework_root("openevolve")

    initial = (
        "# EVOLVE-BLOCK-START\n" + contract["initial_program"].rstrip() + "\n# EVOLVE-BLOCK-END\n"
    )
    (Path(work_dir) / "initial_program.py").write_text(initial)
    (Path(work_dir) / "evaluator.py").write_text(_openevolve_evaluator)
    write_scoring_glue(work_dir, context_path)

    # Their shipped defaults, with only the wiring keys replaced: the endpoint,
    # the model, the task text, our evaluator's own timeout, and a code-length cap
    # that would otherwise discard any child longer than 10,000 characters
    with open(root / "configs" / "default_config.yaml") as handle:
        config = yaml.safe_load(handle)

    model = {"name": context["model"], "weight": 1.0}
    config["llm"]["models"] = [model]
    config["llm"]["evaluator_models"] = [dict(model)]
    config["llm"]["api_base"] = os.environ[context["base_url_env"]]
    config["llm"]["api_key"] = "${" + context["token_env"] + "}"
    config["llm"]["max_tokens"] = max_completion_tokens
    config["llm"]["timeout"] = 300
    config["prompt"]["system_message"] = (
        "You are an expert algorithm designer improving a Python program through "
        "evolution.\n\n" + problem_statement(context) + "\n\nOnly the code between the "
        "EVOLVE-BLOCK markers may change; keep the function name and signature."
    )
    config["evaluator"]["timeout"] = context["eval_timeout"] + 30
    config["evaluator"]["cascade_evaluation"] = False
    config["max_code_length"] = 50000
    config["random_seed"] = context["seed"]

    if _smoke(context):
        config["max_iterations"] = 2
        config["evaluator"]["parallel_evaluations"] = 1

    with open(Path(work_dir) / "config.yaml", "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    out_dir = Path(work_dir) / "out"
    write_launcher(
        work_dir,
        argv=[
            str(root / "openevolve-run.py"),
            str(Path(work_dir) / "initial_program.py"),
            str(Path(work_dir) / "evaluator.py"),
            "--config",
            str(Path(work_dir) / "config.yaml"),
            "--output",
            str(out_dir),
        ],
        cwd=work_dir,
        venv_bin=venv_bin("openevolve"),
    )

    return {"out_dir": str(out_dir)}


def _openevolve_parse(work_dir: Path, stdout: str) -> tuple[str, dict]:
    best = Path(work_dir) / "out" / "best" / "best_program.py"
    program = _read_best(best, "openevolve")
    info_path = best.with_name("best_program_info.json")
    info = json.loads(info_path.read_text()) if info_path.exists() else {}

    return program, {
        "best_iteration": info.get("iteration"),
        "metrics": info.get("metrics"),
    }


# CodeEvolve ---------------------------------------------------------------------

_codeevolve_evaluator = '''\
"""Written by baselines/discovery.py for CodeEvolve."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from discovery_scoring import score_program  # noqa: E402


if __name__ == "__main__":
    program_path, results_path = sys.argv[1], sys.argv[2]
    scored = score_program(program_path=program_path)
    with open(results_path, "w") as handle:
        json.dump(
            {
                "fitness": float(scored["fitness"]),
                "raw_reward": scored["reward"],
                "eval_time": float(scored["seconds"]),
            },
            handle,
            indent=2,
        )
    if not scored["ok"]:
        print(scored["error"], file=sys.stderr)
'''


def _codeevolve_export(graph, work_dir: Path, budget: int, diffusion_model: str) -> dict:
    import yaml

    context = read_context(work_dir)
    context_path = Path(work_dir) / context_filename
    contract = context["contract"]
    root = framework_root("codeevolve")
    inpt = Path(work_dir) / "input"
    os.makedirs(inpt / "src", exist_ok=True)

    initial = (
        "# EVOLVE-BLOCK-START\n" + contract["initial_program"].rstrip() + "\n# EVOLVE-BLOCK-END\n"
    )
    (inpt / "src" / "init_program.py").write_text(initial)
    (inpt / "evaluate.py").write_text(_codeevolve_evaluator)
    # inpt_dir is copied per evaluation, so the glue has to live inside it
    write_scoring_glue(inpt, context_path)

    # Their shipped template (configs/templates/config_mock.yaml) with the model,
    # the task text and the evaluation budget swapped in
    with open(root / "configs" / "templates" / "config_mock.yaml") as handle:
        config = yaml.safe_load(handle)

    config["SEED"] = context["seed"]
    config["SYS_MSG"] = (
        "# PROMPT-BLOCK-START\n\nSETTING:\nYou are an expert algorithm designer "
        "iteratively improving a Python program to maximize its fitness.\n\n"
        + problem_statement(context)
        + "\n\nPERFORMANCE METRICS:\n1. **fitness**: the score (higher is better)\n"
        "2. **raw_reward**: the task's own quantity\n3. **eval_time**: seconds the "
        "evaluation took\n\n# PROMPT-BLOCK-END\n"
    )
    timeout = context["eval_timeout"]
    config["BUDGET_CONFIG"] = {
        "eval_timeout": timeout,
        "max_mem_bytes": 4_000_000_000,
        "resource_check_interval_s": 0.1,
    }
    config["EVOLVE_CONFIG"]["fitness_key"] = "fitness"
    config["MAP_ELITES"] = {
        "elite_map_type": "grid",
        "features": [
            {"name": "eval_time", "min_val": 0, "max_val": timeout, "num_bins": 20}
        ],
    }
    ensemble = [
        {
            "model_name": context["model"],
            "temp": 0.7,
            "top_p": 0.95,
            "retries": 3,
            "weight": 1,
            "max_tok": max_completion_tokens,
        }
    ]
    config["EXPLORATION_ENSEMBLE"] = [dict(ensemble[0], temp=0.9)]
    config["EXPLOITATION_ENSEMBLE"] = ensemble
    config["SAMPLER_AUX_LM"] = dict(ensemble[0])

    if _smoke(context):
        config["EVOLVE_CONFIG"]["num_epochs"] = 2
        config["EVOLVE_CONFIG"]["num_islands"] = 1
        config["EVOLVE_CONFIG"]["init_pop"] = 1

    with open(Path(work_dir) / "config.yaml", "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    out_dir = Path(work_dir) / "out"
    write_launcher(
        work_dir,
        argv=[
            "--inpt_dir",
            str(inpt),
            "--cfg_path",
            str(Path(work_dir) / "config.yaml"),
            "--out_dir",
            str(out_dir),
            "--load_ckpt",
            "0",
            "--y",
        ],
        cwd=work_dir,
        env_map={"API_BASE": context["base_url_env"], "API_KEY": context["token_env"]},
        # The CPU-time cap is summed over the process tree, so a multi-threaded
        # BLAS inside the scorer would trip it well before the wall clock
        static_env={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        venv_bin=venv_bin("codeevolve"),
        program=str(venv_bin("codeevolve") / "codeevolve"),
    )

    return {"out_dir": str(out_dir)}


def _codeevolve_parse(work_dir: Path, stdout: str) -> tuple[str, dict]:
    out_dir = Path(work_dir) / "out"
    metadata_path = out_dir / "run_metadata.json"
    island = None
    info = {}

    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        entries = list(metadata.values()) if isinstance(metadata, dict) else metadata
        if entries:
            last = entries[-1]
            best = last.get("best_sol") if isinstance(last, dict) else None
            if isinstance(best, dict):
                island = best.get("island_found")
                info = {
                    "fitness": best.get("fitness"),
                    "iteration_found": best.get("iteration_found"),
                    "island_found": island,
                }

    if island is None:
        # No usable metadata: the best across islands is the one whose own
        # metadata carries the highest fitness, and with a single island it is
        # simply that island's file
        candidates = sorted(out_dir.glob("island_*/best_sol.py"))
        if not candidates:
            raise FileNotFoundError(
                f"codeevolve wrote no island_*/best_sol.py under {out_dir}"
            )
        if len(candidates) > 1:
            raise ValueError(
                f"codeevolve left {len(candidates)} islands and no readable "
                f"run_metadata.json to pick between them under {out_dir}"
            )
        best_path = candidates[0]
    else:
        best_path = out_dir / f"island_{island}" / "best_sol.py"

    return _read_best(best_path, "codeevolve"), info


# LLaMEA ------------------------------------------------------------------------

_llamea_driver = '''\
"""Written by baselines/discovery.py for LLaMEA."""

import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from discovery_scoring import load_context, score_program  # noqa: E402

context = load_context()
os.environ["OPENAI_API_KEY"] = os.environ[context["token_env"]]
os.environ["OPENAI_BASE_URL"] = os.environ[context["base_url_env"]]

from llamea import LLaMEA, OpenAI_LLM  # noqa: E402

task_prompt = __TASK_PROMPT__
example_prompt = __EXAMPLE_PROMPT__
budget = __BUDGET__
work_dir = __WORK_DIR__


def evaluate(solution, logger=None):
    scored = score_program(program=solution.code, tag=str(solution.id))
    if scored["ok"]:
        feedback = (
            f"Fitness {scored['fitness']:.4f} (higher is better); "
            f"{scored['reward_name']} = {scored['reward']}."
        )
    else:
        feedback = f"The program failed to score: {scored['error']}"
    solution.set_scores(float(scored["fitness"]), feedback)
    return solution


if __name__ == "__main__":
    random.seed(context["seed"])
    np.random.seed(context["seed"])
    llm = OpenAI_LLM(api_key=os.environ["OPENAI_API_KEY"], model=context["model"])
    os.chdir(work_dir)
    search = LLaMEA(
        evaluate,
        llm=llm,
        task_prompt=task_prompt,
        example_prompt=example_prompt,
        experiment_name="discovery",
        budget=budget,
        minimization=False,
        parallel_backend="threading",
    )
    best = search.run()
    with open(os.path.join(work_dir, "best_program.py"), "w") as handle:
        handle.write(best.code)
    with open(os.path.join(work_dir, "best_info.json"), "w") as handle:
        json.dump(
            {
                "fitness": best.fitness,
                "name": best.name,
                "description": best.description,
                "generation": best.generation,
                "budget": budget,
            },
            handle,
            indent=2,
            default=str,
        )
'''


def _llamea_export(graph, work_dir: Path, budget: int, diffusion_model: str) -> dict:
    context = read_context(work_dir)
    context_path = Path(work_dir) / context_filename
    contract = context["contract"]
    write_scoring_glue(work_dir, context_path)

    task_prompt = (
        problem_statement(context)
        + "\n\nGive an excellent and novel algorithm for this task and also give it a "
        "one-line description with the main idea."
    )
    example_prompt = (
        "An example of an acceptable program (the function signature is fixed):\n"
        "```python\n" + contract["initial_program"].rstrip() + "\n```"
    )
    # 100 evaluations is the constructor default and the paper's setting
    search_budget = 3 if _smoke(context) else 100

    driver = (
        _llamea_driver.replace("__TASK_PROMPT__", repr(task_prompt))
        .replace("__EXAMPLE_PROMPT__", repr(example_prompt))
        .replace("__BUDGET__", repr(search_budget))
        .replace("__WORK_DIR__", repr(str(Path(work_dir).resolve())))
    )
    (Path(work_dir) / "run_llamea.py").write_text(driver)
    write_launcher(
        work_dir,
        argv=[str(Path(work_dir) / "run_llamea.py")],
        cwd=work_dir,
        venv_bin=venv_bin("llamea"),
    )

    return {}


def _driver_parse(name: str, work_dir: Path, stdout: str) -> tuple[str, dict]:
    program = _read_best(Path(work_dir) / best_program_filename, name)
    info_path = Path(work_dir) / "best_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}

    return program, info


# EoH ---------------------------------------------------------------------------

_eoh_driver = '''\
"""Written by baselines/discovery.py for EoH (the v0.2 EoH/LLMConfig/BaseProblem API)."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from discovery_scoring import load_context, score_program  # noqa: E402
from eoh import BaseProblem, EoH, LLMConfig  # noqa: E402

context = load_context()
work_dir = __WORK_DIR__


class DiscoveryProblem(BaseProblem):
    template_program = __TEMPLATE__
    task_description = __TASK__

    def evaluate_program(self, program_str, callable_func):
        scored = score_program(program=program_str)
        if not scored["ok"]:
            return None
        # EoH minimizes
        return -float(scored["fitness"])


if __name__ == "__main__":
    llm = LLMConfig(
        api_endpoint=__HOST__,
        api_key=os.environ[context["token_env"]],
        model=context["model"],
        timeout=300,
    )
    problem = DiscoveryProblem(timeout=context["eval_timeout"] + 30)
    search = EoH(llm=llm, problem=problem, output_dir=work_dir, **__OVERRIDES__)
    search.run()

    best_path = os.path.join(work_dir, "results", "samples", "samples_best.json")
    with open(best_path) as handle:
        best = json.load(handle)
    if isinstance(best, list):
        best = best[-1]
    with open(os.path.join(work_dir, "best_program.py"), "w") as handle:
        handle.write(best["code"])
    with open(os.path.join(work_dir, "best_info.json"), "w") as handle:
        json.dump(
            {
                "objective": best.get("objective"),
                "algorithm": best.get("algorithm"),
                "sample_order": best.get("sample_order"),
            },
            handle,
            indent=2,
            default=str,
        )
'''


def _eoh_export(graph, work_dir: Path, budget: int, diffusion_model: str) -> dict:
    context = read_context(work_dir)
    context_path = Path(work_dir) / context_filename
    contract = context["contract"]
    write_scoring_glue(work_dir, context_path)

    # pop_size 5, n_pop 20 and the four operators are EoH's own defaults
    overrides = {"pop_size": 2, "n_pop": 1} if _smoke(context) else {}
    driver = (
        _eoh_driver.replace("__WORK_DIR__", repr(str(Path(work_dir).resolve())))
        .replace("__TEMPLATE__", repr(contract["initial_program"]))
        .replace("__TASK__", repr(problem_statement(context)))
        .replace("__HOST__", repr(gateway_host(context)))
        .replace("__OVERRIDES__", repr(overrides))
    )
    (Path(work_dir) / "run_eoh.py").write_text(driver)
    write_launcher(
        work_dir,
        argv=[str(Path(work_dir) / "run_eoh.py")],
        cwd=work_dir,
        venv_bin=venv_bin("eoh"),
    )

    return {}


# ReEvo and MCTS-AHD (shared file layout) --------------------------------------------

_hydra_eval = '''\
"""Written by baselines/discovery.py: ReEvo / MCTS-AHD eval.py, scores gpt.py."""

import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)

# gpt.py is rewritten per candidate as soon as this process reports it is
# running, so the source is read BEFORE the first print
with open(os.path.join(here, "gpt.py")) as handle:
    source = handle.read()

print("[*] Running ...")
sys.stdout.flush()

from discovery_scoring import score_program  # noqa: E402

if __name__ == "__main__":
    scored = score_program(program=source)
    if not scored["ok"]:
        raise RuntimeError(f"candidate failed to score: {scored['error']}")
    print(f"[*] {scored['reward_name']}: {scored['reward']}")
    # The objective is parsed off the LAST line and must be positive
    print(max(float(scored["fitness"]), 1e-9))
'''


def _hydra_problem_files(
    name: str, context: dict, work_dir: Path, signature: str, description: str
) -> str:
    """Write the problem into the clone under a run-unique name; return that name."""
    import yaml

    root = framework_root(name)
    uid = run_uid(work_dir)
    contract = context["contract"]
    context_path = Path(work_dir) / context_filename

    problem_dir = root / "problems" / uid
    prompts_dir = root / "prompts" / uid
    os.makedirs(problem_dir, exist_ok=True)
    os.makedirs(prompts_dir, exist_ok=True)
    os.makedirs(root / "cfg" / "problem", exist_ok=True)

    (problem_dir / "eval.py").write_text(_hydra_eval)
    write_scoring_glue(problem_dir, context_path)

    seed_program = contract["initial_program"].replace(
        f"def {contract['function']}(", f"def {contract['function']}_v1("
    )
    (problem_dir / "gpt.py").write_text(seed_program)
    (prompts_dir / "seed_func.txt").write_text(seed_program)
    (prompts_dir / "func_signature.txt").write_text(signature + "\n")
    (prompts_dir / "func_desc.txt").write_text(description + "\n")

    problem_config = {
        "problem_name": uid,
        "problem_type": "constructive",
        # Our fitness is already oriented; both frameworks negate a max objective
        "obj_type": "max",
        "problem_size": int(context["experiment"]["budget"]),
        "func_name": contract["function"],
        "description": problem_statement(context),
    }
    with open(root / "cfg" / "problem" / f"{uid}.yaml", "w") as handle:
        yaml.safe_dump(problem_config, handle, sort_keys=False)

    return uid


def _reevo_export(graph, work_dir: Path, budget: int, diffusion_model: str) -> dict:
    context = read_context(work_dir)
    contract = context["contract"]
    signature = contract["signature"].replace(
        f"def {contract['function']}(", f"def {contract['function']}_v{{version}}("
    )
    description = (
        f"The {contract['function']} function is the whole algorithm: one Python "
        f"function, importing networkx as nx inside the function body if it needs "
        f"it. " + contract["description"]
    )
    uid = _hydra_problem_files("reevo", context, work_dir, signature, description)
    root = framework_root("reevo")

    argv = [
        str(root / "main.py"),
        f"problem={uid}",
        "llm_client=openai",
        f"llm_client.model={context['model']}",
        # cfg/llm_client/openai.yaml carries no base_url key, so it is APPENDED
        "+llm_client.base_url=${oc.env:" + context["base_url_env"] + "}",
        "llm_client.api_key=${oc.env:" + context["token_env"] + "}",
        f"timeout={context['eval_timeout'] + 30}",
        f"hydra.run.dir={Path(work_dir).resolve() / 'hydra'}",
    ]
    if _smoke(context):
        argv += ["max_fe=4", "pop_size=2", "init_pop_size=2"]

    write_launcher(
        work_dir,
        argv=argv,
        cwd=root,
        venv_bin=venv_bin("reevo"),
    )

    return {"uid": uid}


def _hydra_best_from_log(work_dir: Path) -> Path | None:
    log = Path(work_dir) / "hydra" / "main.log"
    if not log.exists():
        return None

    for line in reversed(log.read_text().splitlines()):
        if "Best Code Path Overall" in line:
            candidate = Path(line.split("Best Code Path Overall:", 1)[1].strip())
            if not candidate.is_absolute():
                candidate = Path(work_dir) / "hydra" / candidate
            if candidate.exists():
                return candidate

    return None


def _reevo_parse(work_dir: Path, stdout: str) -> tuple[str, dict]:
    best = _hydra_best_from_log(work_dir)
    if best is None:
        # main.py also writes the winner back into the checkout at the end
        best = framework_root("reevo") / "problems" / run_uid(work_dir) / "gpt.py"

    return _read_best(best, "reevo"), {"best_path": str(best)}


def _mcts_ahd_export(graph, work_dir: Path, budget: int, diffusion_model: str) -> dict:
    context = read_context(work_dir)
    contract = context["contract"]
    # The adapter's regex needs `def name(args) -> ret:` on one line, and its code
    # extractor keeps `import ... return` and appends " result", so the program
    # must open with an import and end with `return result`
    signature = contract["signature"][:-1] + " -> list:"
    description = (
        f"The {contract['function']} function is the whole algorithm. Start the "
        f"code with `import networkx as nx`, define any helper above the function, "
        f"assign the answer to a variable named result and end with `return result`. "
        + contract["description"]
    )
    uid = _hydra_problem_files("mcts_ahd", context, work_dir, signature, description)
    root = framework_root("mcts_ahd")

    argv = [
        str(root / "main.py"),
        f"problem={uid}",
        f"llm_client.model={context['model']}",
        "+llm_client.base_url=${oc.env:" + context["base_url_env"] + "}",
        "llm_client.api_key=${oc.env:" + context["token_env"] + "}",
        f"timeout={context['eval_timeout'] + 30}",
        f"hydra.run.dir={Path(work_dir).resolve() / 'hydra'}",
    ]
    if _smoke(context):
        argv += ["max_fe=6", "pop_size=2", "init_pop_size=2"]

    write_launcher(
        work_dir,
        argv=argv,
        cwd=root,
        venv_bin=venv_bin("mcts_ahd"),
    )

    return {"uid": uid}


def _mcts_ahd_parse(work_dir: Path, stdout: str) -> tuple[str, dict]:
    hydra_dir = Path(work_dir) / "hydra"
    populations = sorted(
        hydra_dir.glob("best_population_generation_*.json"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    if populations:
        entries = json.loads(populations[-1].read_text())
        entry = entries[0] if isinstance(entries, list) else entries
        program = entry.get("code") or ""
        if program.strip():
            return program, {
                "objective": entry.get("objective"),
                "algorithm": entry.get("algorithm"),
                "evaluations": int(populations[-1].stem.rsplit("_", 1)[1]),
            }

    best = _hydra_best_from_log(work_dir)
    if best is None:
        best = framework_root("mcts_ahd") / "problems" / run_uid(work_dir) / "gpt.py"

    return _read_best(best, "mcts_ahd"), {"best_path": str(best)}


# LLM4AD: FunSearch and HillClimb ------------------------------------------------------

_llm4ad_driver = '''\
"""Written by baselines/discovery.py for LLM4AD (__METHOD__)."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, __CLONE__)

from discovery_scoring import load_context, score_program  # noqa: E402
from llm4ad.base import Evaluation  # noqa: E402
from llm4ad.tools.llm.llm_api_openai import OpenAIAPI  # noqa: E402
from llm4ad.tools.profiler import ProfilerBase  # noqa: E402

context = load_context()
work_dir = __WORK_DIR__


class DiscoveryEvaluation(Evaluation):
    def __init__(self):
        super().__init__(
            template_program=__TEMPLATE__,
            task_description=__TASK__,
            exec_code=False,
            timeout_seconds=context["eval_timeout"] + 30,
        )

    def evaluate_program(self, program_str, callable_func, **kwargs):
        scored = score_program(program=program_str)
        if not scored["ok"]:
            return None
        return float(scored["fitness"])


if __name__ == "__main__":
    llm = OpenAIAPI(
        base_url=os.environ[context["base_url_env"]],
        api_key=os.environ[context["token_env"]],
        model=context["model"],
        timeout=300,
    )
    profiler = ProfilerBase(
        log_dir=os.path.join(work_dir, "logs"),
        log_style="simple",
        create_random_path=False,
    )
    if __METHOD__ == "funsearch":
        from llm4ad.method.funsearch import FunSearch

        method = FunSearch(
            llm=llm, evaluation=DiscoveryEvaluation(), profiler=profiler, **__OVERRIDES__
        )
    else:
        from llm4ad.method.hillclimb import HillClimb

        method = HillClimb(
            llm=llm, evaluation=DiscoveryEvaluation(), profiler=profiler, **__OVERRIDES__
        )
    method.run()

    # samples_best.json carries only the FUNCTION (no template imports) with an
    # empty `program`; the per-sample logs carry the whole program, so the best
    # is taken from those and the bare function is the fallback
    samples_dir = os.path.join(work_dir, "logs", "samples")
    best = None
    for name in sorted(os.listdir(samples_dir)):
        if not name.startswith("samples_") or name == "samples_best.json":
            continue
        with open(os.path.join(samples_dir, name)) as handle:
            for entry in json.load(handle):
                if entry.get("score") is None or not (entry.get("program") or "").strip():
                    continue
                if best is None or float(entry["score"]) > float(best["score"]):
                    best = entry
    if best is None:
        with open(os.path.join(samples_dir, "samples_best.json")) as handle:
            best = json.load(handle)
        if isinstance(best, list):
            best = best[-1]
        imports = chr(10).join(
            line for line in __TEMPLATE__.splitlines()
            if line.startswith("import ") or line.startswith("from ")
        )
        best["program"] = imports + chr(10) * 2 + (best.get("program") or best.get("function") or "")
    with open(os.path.join(work_dir, "best_program.py"), "w") as handle:
        handle.write(best["program"])
    with open(os.path.join(work_dir, "best_info.json"), "w") as handle:
        json.dump(
            {"score": best.get("score"), "sample_order": best.get("sample_order")},
            handle,
            indent=2,
            default=str,
        )
'''


def _llm4ad_export(
    method: str, graph, work_dir: Path, budget: int, diffusion_model: str
) -> dict:
    context = read_context(work_dir)
    context_path = Path(work_dir) / context_filename
    contract = context["contract"]
    write_scoring_glue(work_dir, context_path)

    # max_sample_nums 20 with four samplers and four evaluators are the
    # constructors' own defaults; a smoke run just needs the loop to turn once
    overrides = (
        {"max_sample_nums": 3, "num_samplers": 1, "num_evaluators": 1}
        if _smoke(context)
        else {}
    )
    driver = (
        _llm4ad_driver.replace("__METHOD__", repr(method))
        .replace("__CLONE__", repr(str(framework_root("llm4ad"))))
        .replace("__WORK_DIR__", repr(str(Path(work_dir).resolve())))
        .replace("__TEMPLATE__", repr(contract["initial_program"]))
        .replace("__TASK__", repr(problem_statement(context)))
        .replace("__OVERRIDES__", repr(overrides))
    )
    (Path(work_dir) / "run_llm4ad.py").write_text(driver)
    write_launcher(
        work_dir,
        argv=[str(Path(work_dir) / "run_llm4ad.py")],
        cwd=work_dir,
        static_env={"PYTHONPATH": str(framework_root("llm4ad"))},
        venv_bin=venv_bin("llm4ad"),
    )

    return {}


# DeepEvolve -------------------------------------------------------------------

_deepevolve_interface = '''\
"""Scores main.py on the benchmark simulator. Please keep this file as is."""

import importlib.util
import os
import signal

scoring_path = __SCORING__


def _load_scoring():
    spec = importlib.util.spec_from_file_location("discovery_scoring", scoring_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deepevolve_interface():
    scoring = _load_scoring()
    here = os.path.dirname(os.path.abspath(__file__))
    program_path = os.path.join(here, "main.py")

    def on_alarm(signum, frame):
        raise TimeoutError("scoring timed out")

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(__TIMEOUT__)
    try:
        scored = scoring.score_program(program_path=program_path)
    except TimeoutError as error:
        return False, str(error)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)

    if not scored["ok"]:
        return False, scored["error"]

    return True, {
        "combined_score": float(scored["fitness"]),
        "raw_reward": float(scored["reward"]),
        "reward_name": scored["reward_name"],
        "runtime_seconds": float(scored["seconds"]),
    }
'''


def _deepevolve_export(graph, work_dir: Path, budget: int, diffusion_model: str) -> dict:
    context = read_context(work_dir)
    context_path = Path(work_dir) / context_filename
    contract = context["contract"]
    root = framework_root("deepevolve")
    uid = run_uid(work_dir)
    workspace = Path(work_dir) / "problems"
    problem_dir = workspace / uid
    code_dir = problem_dir / "initial_code"
    os.makedirs(code_dir, exist_ok=True)

    scoring_path = write_scoring_glue(work_dir, context_path)
    (code_dir / "main.py").write_text(contract["initial_program"])
    (code_dir / "deepevolve_interface.py").write_text(
        _deepevolve_interface.replace("__SCORING__", repr(str(scoring_path))).replace(
            "__TIMEOUT__", repr(int(context["eval_timeout"]) + 60)
        )
    )

    statement = problem_statement(context)
    (problem_dir / "info.json").write_text(
        json.dumps(
            {
                "problem": {
                    "name": uid,
                    # `metric` is never read by the loop, so it lives in the text
                    "description": statement
                    + "\n\nThe algorithm lives in main.py; deepevolve_interface.py "
                    "scores it and must not be changed.",
                    "metric": contract["reward_name"],
                    "interface": "deepevolve_interface.py",
                },
                # Read unconditionally at start-up; the cached initial_idea.json
                # below is what actually seeds the loop
                "initial_idea": {
                    "title": "Degree heuristic",
                    "content": (
                        f"Rank nodes by degree and take the top k as the answer to "
                        f"`{contract['function']}`."
                    ),
                    "supplement": "",
                },
            },
            indent=2,
        )
    )
    # Hand-written so the first start does not spend a web-search call deriving it
    (problem_dir / "initial_idea.json").write_text(
        json.dumps(
            {
                "description": (
                    f"A degree-based heuristic: rank nodes by degree and take the "
                    f"top k as the answer to `{contract['function']}`."
                ),
                "motivation": (
                    "High-degree nodes touch the most arcs, so under independent "
                    "per-arc transmission they are the cheapest first guess."
                ),
                "implementation_notes": (
                    "Pure networkx: sort graph.nodes() by graph.degree and slice."
                ),
                "pseudocode": "rank nodes by degree; return the first k",
                "originality": {
                    "score": 1,
                    "positive": "A standard baseline every paper reports.",
                    "negative": "Ignores overlap between the chosen nodes' reach.",
                },
                "future_potential": {
                    "score": 3,
                    "positive": "Easy to extend with reach-aware discounting.",
                    "negative": "No use of the transmission weights.",
                },
                "code_difficulty": {
                    "score": 1,
                    "positive": "A few lines of networkx.",
                    "negative": "None.",
                },
            },
            indent=2,
        )
    )

    model = context["model"]
    query = (
        f"Design a better algorithm for {context['task'].replace('_', ' ')} on a "
        f"graph, scored by a fixed simulator."
    )
    argv = [
        str(root / "deepevolve.py"),
        f"query='{query}'",
        f"problem={uid}",
        f"workspace={workspace.resolve()}",
        "checkpoint=ckpt",
        f"researcher.planner={model}",
        f"researcher.searcher={model}",
        f"researcher.writer={model}",
        f"coder.developer={model}",
        f"coder.debugger={model}",
        f"database.random_seed={context['seed']}",
    ]
    if _smoke(context):
        argv += ["max_iterations=1", "max_research_reflect=0", "max_coding_reflect=0"]

    write_launcher(
        work_dir,
        argv=argv,
        cwd=root,
        env_map={
            "OPENAI_API_KEY": context["token_env"],
            "OPENAI_BASE_URL": context["base_url_env"],
        },
        # Spans would otherwise be posted to OpenAI's tracing backend with the
        # gateway token and fail on every call
        static_env={"OPENAI_AGENTS_DISABLE_TRACING": "1"},
        venv_bin=venv_bin("deepevolve"),
    )

    return {"uid": uid, "workspace": str(workspace)}


def _deepevolve_parse(work_dir: Path, stdout: str) -> tuple[str, dict]:
    best_dir = Path(work_dir) / "problems" / run_uid(work_dir) / "ckpt" / "best"
    # The evolved interface is discarded on purpose: only main.py is the algorithm
    program = _read_best(best_dir / "main.py", "deepevolve")
    info_path = best_dir / "best_program_info.json"
    info = {}
    if info_path.exists():
        raw = json.loads(info_path.read_text())
        info = {
            "generation": raw.get("generation"),
            "iteration": raw.get("iteration"),
            "metrics": raw.get("metrics"),
        }

    return program, info


# What the registry binds ------------------------------------------------------------

adapters = {
    "openevolve": (_openevolve_export, _openevolve_parse),
    "codeevolve": (_codeevolve_export, _codeevolve_parse),
    "llamea": (_llamea_export, partial(_driver_parse, "llamea")),
    "eoh": (_eoh_export, partial(_driver_parse, "eoh")),
    "reevo": (_reevo_export, _reevo_parse),
    "mcts_ahd": (_mcts_ahd_export, _mcts_ahd_parse),
    "llm4ad_funsearch": (
        partial(_llm4ad_export, "funsearch"),
        partial(_driver_parse, "llm4ad_funsearch"),
    ),
    "llm4ad_hillclimb": (
        partial(_llm4ad_export, "hillclimb"),
        partial(_driver_parse, "llm4ad_hillclimb"),
    ),
    "deepevolve": (_deepevolve_export, _deepevolve_parse),
}
