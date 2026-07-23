from coding_agent.types import GraphInfo, TaskSpec
from coding_agent.executor import scored_blocked_primitives
from coding_agent.tools.library_api import (
    build_algorithm_menu,
    build_api_reference,
    build_primitives_reference,
)

# Shared system-prompt preamble containing common rules for the coding agent
common_rules = """\
You are designing an Influence Maximization algorithm as an executable Python script.

OUTPUT FORMAT: reply with exactly ONE fenced ```python block and nothing else —
no prose before or after. The block MUST define a class subclassing `Strategy`
and contain nothing outside that class: no imports, no module-level code, no
example usage.

AVAILABLE NAMES (already imported into your script's namespace — do NOT import them):
- `ActionOp(op, target, destination=None, weight=None)` : a graph action. Ops:
    add_node, remove_node, add_edge, remove_edge, set_edge_weight.
- `State` : has .infected (list[int]) and .frontier (list[int]).
- `GraphInfo` : .num_nodes, .out_neighbors(node), .in_neighbors(node), .degree(node), .edge_index, .ic_probs.
- `algorithms` and `primitives` modules (API below).

ACTION RULES:
- A seed is ActionOp("add_node", node). Emit at most `budget` add_node actions in total.
- Node ids must be in [0, num_nodes).
- You may also use remove_node / add_edge / remove_edge / set_edge_weight to steer the cascade.
"""

# System prompts dict-keyed by method name (one_shot, per_step, windowed)
system_prompts = {
    "one_shot": common_rules
    + """\

METHOD: ONE-SHOT SUPER-ALGORITHM.
Implement `plan_horizon(self, graph, budget, horizon) -> list[list[ActionOp]]`.
Return a list of length (horizon+1): element t is the action bag applied at timestep t.
This is your whole multi-timestep plan, decided up front. Classical algorithms only
fill element 0 (the seed set) and leave the rest empty — go beyond that: schedule
interventions across t0..tT to maximize final spread.

REPLY SHAPE (adapt the logic, keep the structure):
```python
class MyStrategy(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        seeds = algorithms.high_degree(graph, budget, "IC")
        plan = [[ActionOp("add_node", node) for node in seeds]]
        plan += [[] for _ in range(horizon)]
        return plan
```
Do not return this baseline unchanged — improve on it.
""",
    "per_step": common_rules
    + """\

METHOD: PER-STEP POLICY.
Implement `act(self, state, graph, timestep) -> list[ActionOp]`.
You are called once per timestep with the CURRENT state; return that step's action bag.
React to which nodes are infected/frontier right now.

REPLY SHAPE (adapt the logic, keep the structure; budget/horizon are in the task):
```python
class MyStrategy(Strategy):
    def act(self, state, graph, timestep):
        if timestep == 0:
            seeds = algorithms.high_degree(graph, 5, "IC")  # 5 = task budget
            return [ActionOp("add_node", node) for node in seeds]
        return []
```
""",
    "windowed": common_rules
    + """\

METHOD: WINDOWED ONLINE ALGORITHM.
Implement `act(self, state, graph, timestep) -> list[ActionOp]`.
You are called once per time WINDOW with the current state and the window index.
Treat each call as solving a fresh IM sub-problem on the current state; you may reuse a
classical algorithm (e.g. `algorithms.celf`) within each window.
Note: for this windowed method the budget applies PER window call (you are invoked once per window).

REPLY SHAPE (adapt the logic, keep the structure; budget/horizon are in the task):
```python
class MyStrategy(Strategy):
    def act(self, state, graph, timestep):
        seeds = algorithms.celf(graph, 5, "IC")  # 5 = per-window budget
        return [ActionOp("add_node", node) for node in seeds]
```
""",
}


def build_user_prompt(
    method: str, task: TaskSpec, graph: GraphInfo, strategy_mode: str = "free"
) -> str:
    if strategy_mode == "scored":
        # Library source is inspiration, not callable — ideas must be written
        # out inside score()/schedule(), where they can be mutated
        reference = (
            "PRIMITIVES API (available as `primitives`; spread-simulation "
            "functions are NOT available):\n"
            f"{build_primitives_reference(exclude=scored_blocked_primitives)}\n\n"
            "ALGORITHM IDEAS (NOT callable — steal the ideas into your score()):\n"
            f"{build_algorithm_menu()}"
        )
        final_line = "Write the ScoredStrategy subclass now."
    else:
        reference = f"LIBRARY API:\n{build_api_reference()}"
        final_line = f"Write the Strategy now (method = {method})."

    return f"""\
TASK: {task.task} — {task.objective}
diffusion_model = {task.diffusion_model}
budget = {task.budget}   (max seeds total)
horizon = {task.horizon} (timesteps)
allowed_ops = {", ".join(task.allowed_ops)}   (any other op is REJECTED)

GRAPH:
num_nodes = {graph.num_nodes}
num_edges = {graph.edge_index.shape[1]}
directed = {graph.directed}

{reference}

{final_line}"""


# Scored mode: the agent edits an algorithm's internals (score/schedule hooks),
# never whole programs and never compositions over the library
scored_system = """\
You are designing the SCORING RULE of an Influence Maximization algorithm — not a whole program.

A fixed harness (ScoredStrategy.plan_horizon) greedily picks the highest-score node
until the budget is spent, then calls schedule() to place the picked seeds across
timesteps. You may override ONLY these two methods:

- score(self, node, selected, graph) -> float
    Called for every candidate node at every pick; `selected` is the tuple of
    already-chosen seeds. Higher score = picked sooner. This is where your
    algorithm lives: combine structural signals, penalize redundancy, adapt to
    what the diagnostics reveal.
- schedule(self, seeds, graph, horizon) -> list[list[ActionOp]]   (optional)
    Element t is the action bag applied at timestep t. Default: all seeds at t=0.

RULES:
- Overriding plan_horizon is REJECTED by the executor.
- `algorithms.*` does NOT exist here. Write your own scoring logic from graph
  structure and `primitives` (spread-simulation primitives are unavailable).
- Reply with exactly ONE fenced ```python block containing ONE class subclassing
  ScoredStrategy. No imports, no module-level code, no prose.
- `ActionOp(op, target, destination=None, weight=None)`; `GraphInfo` has
  .num_nodes, .out_neighbors(node), .in_neighbors(node), .degree(node),
  .edge_index, .ic_probs.

REPLY SHAPE (adapt the logic — improve on it, do not return it unchanged):
```python
class MyScorer(ScoredStrategy):
    def score(self, node, selected, graph):
        overlap = len(set(graph.out_neighbors(node)) & set(selected))
        return graph.degree(node) - 2.0 * overlap
```
"""


def build_system_prompt(method: str, strategy_mode: str = "free") -> str:
    if strategy_mode == "scored":
        if method in ("one_shot", "evolve"):
            return scored_system

        raise ValueError(
            f"strategy_mode='scored' is not supported for method {method!r}; "
            f"use one_shot or evolve"
        )

    # evolve generates plan_horizon strategies under the same contract as one_shot
    return system_prompts["one_shot" if method == "evolve" else method]


# Evolve method: each generation is an EDIT of a parent from the population
evolve_operator_instructions = {
    "refine": (
        "Make a SMALL, targeted improvement to the PARENT: tune a weight, add or "
        "adjust one term, fix one weakness the diagnostics expose. Keep its "
        "overall approach."
    ),
    "restructure": (
        "REDESIGN the approach: keep the same contract but change the core idea — "
        "different structural signals, different selection logic. Do not just "
        "re-tune the parent."
    ),
}


def build_evolve_prompt(
    operator: str,
    parent: dict,
    inspirations: list[dict],
    error: str | None = None,
) -> str:
    inspiration_text = "".join(
        f"\nALTERNATIVE from the population (reward={record['reward']:.2f}):\n"
        f"```python\n{record['script']}\n```\n"
        for record in inspirations
    )
    error_text = (
        f"\nYour previous attempt failed with:\n{error}\n" if error else ""
    )

    return f"""
You are evolving a population of strategies. Produce a NEW candidate by modifying the PARENT.

PARENT (reward={parent["reward"]:.2f}):
```python
{parent["script"]}
```
Parent rollout diagnostics:
{parent["summary"]}
{inspiration_text}{error_text}
OPERATION — {operator.upper()}: {evolve_operator_instructions[operator]}
Reply with one ```python block."""


# GA-routing baseline: the LLM selects from the pool but never synthesizes code
routing_system = """\
You are an algorithm-selection router for Influence Maximization.
You will be given a task, a graph description, and a menu of classical library
algorithms. Pick the single most promising algorithm for this graph and task.

Reply with EXACTLY ONE algorithm name from the menu — no code, no punctuation,
no explanation."""


def build_routing_prompt(task: TaskSpec, graph: GraphInfo) -> str:
    return f"""\
TASK: {task.task} — {task.objective}
diffusion_model = {task.diffusion_model}
budget = {task.budget}   (max seeds total)
horizon = {task.horizon} (timesteps)

GRAPH:
num_nodes = {graph.num_nodes}
num_edges = {graph.edge_index.shape[1]}
directed = {graph.directed}

ALGORITHM MENU:
{build_algorithm_menu()}

Reply with exactly one name from the menu."""


def build_feedback_prompt(
    reward: float,
    summary: str,
    error: str | None = None,
    credit_report: str | None = None,
) -> str:
    error_text = f"\nThe previous script raised an error:\n{error}\n" if error else ""
    credit_text = f"\n{credit_report}\n" if credit_report else ""
    return f"""\
Your previous strategy achieved final spread (reward) = {reward}.
Trajectory summary: {summary}{error_text}{credit_text}
Revise the Strategy to increase final spread. Reply with one ```python block."""
