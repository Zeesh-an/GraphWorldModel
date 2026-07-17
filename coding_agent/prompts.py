from coding_agent.types import GraphInfo, TaskSpec
from coding_agent.tools.library_api import build_api_reference

common_rules = """\
You are designing an Influence Maximization algorithm as an executable Python script.

OUTPUT FORMAT: reply with exactly ONE fenced ```python block and nothing else —
no prose before or after. The block MUST define a class subclassing `Strategy`
and contain nothing outside that class: no imports, no module-level code, no
example usage.

AVAILABLE NAMES (already imported into your script's namespace — do NOT import them):
- `Action(op, target, destination=None, weight=None)` : a graph action. Ops:
    add_node, remove_node, add_edge, remove_edge, set_edge_weight.
- `State` : has .infected (list[int]) and .frontier (list[int]).
- `GraphInfo` : .num_nodes, .out_neighbors(node), .in_neighbors(node), .degree(node), .edge_index, .ic_probs.
- `algorithms` and `primitives` modules (API below).

ACTION RULES:
- A seed is Action("add_node", node). Emit at most `budget` add_node actions in total.
- Node ids must be in [0, num_nodes).
- You may also use remove_node / add_edge / remove_edge / set_edge_weight to steer the cascade.
"""

system_prompts = {
    "one_shot": common_rules
    + """\

METHOD: ONE-SHOT SUPER-ALGORITHM.
Implement `plan_horizon(self, graph, budget, horizon) -> list[list[Action]]`.
Return a list of length (horizon+1): element t is the action bag applied at timestep t.
This is your whole multi-timestep plan, decided up front. Classical algorithms only
fill element 0 (the seed set) and leave the rest empty — go beyond that: schedule
interventions across t0..tT to maximize final spread.

REPLY SHAPE (adapt the logic, keep the structure):
```python
class MyStrategy(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        seeds = algorithms.high_degree(graph, budget, "IC")
        plan = [[Action("add_node", node) for node in seeds]]
        plan += [[] for _ in range(horizon)]
        return plan
```
Do not return this baseline unchanged — improve on it.
""",
    "per_step": common_rules
    + """\

METHOD: PER-STEP POLICY.
Implement `act(self, state, graph, timestep) -> list[Action]`.
You are called once per timestep with the CURRENT state; return that step's action bag.
React to which nodes are infected/frontier right now.

REPLY SHAPE (adapt the logic, keep the structure; budget/horizon are in the task):
```python
class MyStrategy(Strategy):
    def act(self, state, graph, timestep):
        if timestep == 0:
            seeds = algorithms.high_degree(graph, 5, "IC")  # 5 = task budget
            return [Action("add_node", node) for node in seeds]
        return []
```
""",
    "windowed": common_rules
    + """\

METHOD: WINDOWED ONLINE ALGORITHM.
Implement `act(self, state, graph, timestep) -> list[Action]`.
You are called once per time WINDOW with the current state and the window index.
Treat each call as solving a fresh IM sub-problem on the current state; you may reuse a
classical algorithm (e.g. `algorithms.celf`) within each window.
Note: for this windowed method the budget applies PER window call (you are invoked once per window).

REPLY SHAPE (adapt the logic, keep the structure; budget/horizon are in the task):
```python
class MyStrategy(Strategy):
    def act(self, state, graph, timestep):
        seeds = algorithms.celf(graph, 5, "IC")  # 5 = per-window budget
        return [Action("add_node", node) for node in seeds]
```
""",
}


def build_user_prompt(method: str, task: TaskSpec, graph: GraphInfo) -> str:
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

LIBRARY API:
{build_api_reference()}

Write the Strategy now (method = {method})."""


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
