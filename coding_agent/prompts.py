from coding_agent.rounds import round_batches, round_schedule
from coding_agent.types import GraphInfo, TaskSpec, myopic
from data.wm_simulator import blocked, spent
from coding_agent.executor import (
    allowed_imports,
    mc_blocked_algorithms,
    scored_blocked_primitives,
)
from coding_agent.tools.graph_profile import build_graph_profile
from coding_agent.tools.library_api import (
    build_adaptive_reference,
    build_algorithm_menu,
    build_algorithm_sources,
    build_api_reference,
    build_primitives_reference,
)

edge_ops = ("add_edge", "remove_edge", "set_edge_weight")

# Shared system-prompt preamble containing common rules for the coding agent
common_rules = f"""\
You are designing an Influence Maximization algorithm as an executable Python script.

OUTPUT FORMAT: reply with exactly ONE fenced ```python block and nothing else —
no prose before or after. The block contains import lines (if you need any) and
then exactly ONE class subclassing `Strategy`. Nothing else at module level: no
example usage, no test code.

IMPORTS: you MAY import any of {", ".join(allowed_imports)}. Use numpy for
anything you would otherwise write as a Python loop over all nodes — vectorized
scoring is what lets you afford large sample sizes inside the time limit.
Importing anything else is rejected.

AVAILABLE NAMES (already in your script's namespace — do NOT import these):
- `ActionOp(op, target, destination=None, weight=None)` : a graph action. Ops:
    add_node, remove_node, add_edge, remove_edge, set_edge_weight.
- `State` : has .infected (list[int]) and .frontier (list[int]).
- `GraphInfo` : .num_nodes, .out_neighbors(node), .in_neighbors(node), .degree(node), .edge_index, .ic_probs.
- `algorithms`, `adaptive_algorithms` and `primitives` modules (API below).
  `adaptive_algorithms` members are per-ROUND policies and are only callable
  from act() on an adaptive task; the reference below lists them when so.

ACTION RULES:
- A seed is ActionOp("add_node", node). Emit at most `budget` add_node actions in total.
- Seeding the same node twice is REJECTED: it spends two units of budget on one node.
- Node ids must be in [0, num_nodes).
- add_edge / remove_edge / set_edge_weight rewire the graph the cascade runs on;
  under LT edge weights are ignored and only the structural change applies.
- what remove_node does is stated below; read it before using the op.
"""

# What remove_node means is a property of the dataset, so it cannot be baked into
# the module-level preamble. build_system_prompt appends the matching one.
remove_semantics_notes = {
    spent: """\

REMOVE_NODE (this task: spent): under IC it sets the node to Removed, so it STAYS
counted as infected and simply stops spreading; it can only ever lower your score.
Under LT it returns the node to Susceptible, which it may re-cross later. Neither
deletes the node or its edges. For influence MAXIMIZATION this is almost never the
right action: do not spend budget on it without a specific reason.
""",
    blocked: """\

REMOVE_NODE (this task: blocked): it DELETES the node from the graph. The node
stops counting toward the spread, cannot transmit, and cannot be infected or
re-infected, under both IC and LT. Its incident edges go with it, so its
neighbours lose that path entirely. This is the containment action: use it to cut
the routes a cascade would otherwise take, and remember that deleting a node the
cascade already passed through only removes its onward reach, not the infections
it already caused.
""",
}

# Two worked strategies at the quality the search should START from. The trivial
# high_degree stub they replaced set the anchor far too low: the model would
# submit a one-line variation of it and spend its iterations tuning a constant.
one_shot_exemplars = """\
EXAMPLES — two strategies at the level you should START from, not finish at.

Example 1, community-aware discount (spends budget across communities, then
suppresses neighbours of picks so seeds do not overlap):
```python
class CommunityDiscount(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        labels = primitives.detect_communities(graph)
        allocation = primitives.allocate_budget(labels, budget)
        degrees = primitives.compute_degree(graph)

        members = {}
        for node, community_id in labels.items():
            members.setdefault(community_id, []).append(node)

        seeds = []
        for community_id, quota in allocation.items():
            discounted = {node: float(degrees[node]) for node in members[community_id]}
            for _ in range(quota):
                if not discounted:
                    break
                pick = max(discounted, key=discounted.get)
                seeds.append(pick)
                del discounted[pick]
                for neighbor in graph.out_neighbors(pick):
                    if neighbor in discounted:
                        discounted[neighbor] *= 0.5

        return [[ActionOp("add_node", int(node)) for node in seeds]] + [
            [] for _ in range(horizon)
        ]
```

Example 2, lazy-greedy max coverage over reverse-reachable sets (the RIS family;
the heap re-scores only the node it pops, which is what makes a large theta
affordable):
```python
import heapq

class RisCoverage(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        theta = min(200000, 20 * graph.num_nodes)
        covers = {}
        for index, rr_set in enumerate(
            primitives.batch_reverse_sample(graph, theta=theta, seed=0)
        ):
            for node in rr_set:
                covers.setdefault(int(node), set()).add(index)

        heap = [(-len(cover), node, 0) for node, cover in covers.items()]
        heapq.heapify(heap)
        seeds, covered = [], set()

        while len(seeds) < budget and heap:
            _, node, stamp = heapq.heappop(heap)
            if stamp == len(seeds):
                seeds.append(node)
                covered |= covers[node]
            else:
                heapq.heappush(heap, (-len(covers[node] - covered), node, len(seeds)))

        # Coverage can run out before the budget does
        for node in primitives.get_top_degree_nodes(graph, graph.num_nodes):
            if len(seeds) >= budget:
                break
            if node not in seeds:
                seeds.append(node)

        return [[ActionOp("add_node", int(node)) for node in seeds]] + [
            [] for _ in range(horizon)
        ]
```
Beat both. Combining their ideas, or replacing them, are both fair game.
"""

# Only shown when the task actually allows edge ops. Under add_node-only IC the
# cascade is progressive and monotone, so delaying a seed is weakly worse and
# every schedule is dominated by "all seeds at t=0" — telling the model to
# schedule across time there just burns iterations on a flat direction.
temporal_scheduling_note = """\

USING THE HORIZON: this task allows edge operations, so the timing of actions
matters. Rewiring after the cascade has started can open a path the seeds could
not otherwise reach; rewiring before it starts changes where the cascade goes
from the first step. Use the frontier trajectory in the feedback to decide when
each intervention lands.
"""

seed_timing_note = """\

USING THE HORIZON: this task is add_node-only under a progressive cascade, so a
seed placed at t>0 has strictly fewer steps to spread than the same seed at t=0.
Put every seed in element 0 and leave the rest of the plan empty. All of your
effort belongs in WHICH nodes you pick, not when.
"""

# The adaptive counterpart. seed_timing_note ("put every seed at t=0") is exactly
# wrong here: the schedule is fixed by the harness and the late rounds are the
# whole point, so the model needs the opposite instruction.
adaptive_timing_note = """\

USING THE HORIZON: you do NOT choose when your seeds land. The round schedule
does, and it is fixed. A late round is not a wasted one: it has strictly less
time left to spread, and in exchange you get to see where the cascade actually
went. That trade is the entire task. Judge each round by what is still
susceptible and still reachable, not by what looked best at t=0.
"""

# System prompts dict-keyed by method name (one_shot, per_step, windowed)
system_prompts = {
    "one_shot": common_rules
    + """\

METHOD: ONE-SHOT SUPER-ALGORITHM.
Implement `plan_horizon(self, graph, budget, horizon) -> list[list[ActionOp]]`.
Return a list of length (horizon+1): element t is the action bag applied at timestep t.
This is your whole multi-timestep plan, decided up front.

"""
    + one_shot_exemplars,
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
    "adaptive": common_rules
    + """\

METHOD: ADAPTIVE MULTI-ROUND POLICY.
Implement `act(self, state, graph, timestep) -> list[ActionOp]`.

Your budget is NOT spent all at once. It is split into rounds, and you are called
once per round with the state the PREVIOUS round's diffusion actually produced.
Between rounds the cascade runs on its own and you emit nothing. The round
schedule (how many rounds, how many seeds each, at which timesteps) is stated in
the task block below: read it, it is the shape of your problem.

WHAT WINS HERE, AND WHAT DOES NOT:
- Picking a good static seed set and dealing it out in equal slices is the
  NON-ADAPTIVE strategy wearing a costume. It is the arm you are being compared
  against, and theory says it is hard to beat on spread alone.
- The one thing you have that it does not is the realized state. Spend each round
  on what the cascade did NOT reach: a node whose neighbourhood the previous
  round already covered is now worth far less than it looked at t=0.
- Over-committing a round is REJECTED. Each call may return at most that round's
  own batch size.
- Seeding an already-active node wastes the slot. Under full_adoption feedback it
  is REJECTED outright, because you were handed the infected set and could have
  filtered it. Under myopic feedback you were not handed it, so the seed is
  silently DROPPED and the slot is spent for nothing, which is the price of the
  weaker observation. Either way, filter what you can see.

REPLY SHAPE (adapt the logic, keep the structure; the schedule is in the task):
```python
class MyStrategy(Strategy):
    def act(self, state, graph, timestep):
        batch = 2                       # this round's seed count, from the task block
        active = set(state.infected) | set(state.frontier)
        # Discount candidates whose neighbourhood the cascade already owns
        scored = []
        for node in range(graph.num_nodes):
            if node in active:
                continue
            fresh = sum(1 for other in graph.out_neighbors(node) if other not in active)
            scored.append((fresh, node))
        scored.sort(reverse=True)
        return [ActionOp("add_node", node) for _, node in scored[:batch]]
```
""",
}


def build_round_block(task: TaskSpec) -> str:
    """The round schedule, spelled out per call, or nothing for a static task."""
    if not task.adaptive:
        return ""

    batches = round_batches(task.budget, task.rounds, task.per_round_budget)
    schedule = round_schedule(batches, task.round_gap, task.horizon)
    observes = (
        "state.frontier ONLY; state.infected is BLANKED for you, so you can see "
        "the wave that just activated but not the accumulated total"
        if task.feedback_model == myopic
        else "the full realized state: both state.infected and state.frontier"
    )
    lines = [
        "",
        f"ROUND SCHEDULE ({len(batches)} rounds, feedback = {task.feedback_model}). "
        f"act() is called at exactly these timesteps and nowhere else:",
    ]
    lines += [
        f"  t={timestep:<3} return at most {schedule[timestep]} add_node op(s)"
        for timestep in sorted(schedule)
    ]
    lines += [
        f"  totals {sum(batches)} seeds = the budget above; between rounds the "
        f"cascade runs with no input from you",
        f"  at each call you observe {observes}",
        "",
    ]

    return "\n".join(lines)


def build_user_prompt(
    method: str,
    task: TaskSpec,
    graph: GraphInfo,
    strategy_mode: str = "free",
    allow_mc_algorithms: bool = False,
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
        blocked = () if allow_mc_algorithms else mc_blocked_algorithms
        # Signatures alone do not teach the idiom — the model reproduces the
        # library's shape much more reliably once it has read a few of them
        # The adaptive policies are the published baselines for this task, so an
        # adaptive prompt that hides them asks the model to reinvent AdaptGreedy
        adaptive_reference = (
            f"\n\n{build_adaptive_reference()}" if task.adaptive else ""
        )
        reference = (
            f"LIBRARY API:\n{build_api_reference(exclude=blocked)}"
            f"{adaptive_reference}\n\n"
            f"{build_algorithm_sources()}"
        )
        final_line = f"Write the Strategy now (method = {method})."

    return f"""\
TASK: {task.task} — {task.objective}
diffusion_model = {task.diffusion_model}
budget = {task.budget}   ({100.0 * task.budget / graph.num_nodes:.1f}% of nodes, max seeds total)
horizon = {task.horizon} (timesteps)
allowed_ops = {", ".join(task.allowed_ops)}   (any other op is REJECTED)
{build_round_block(task)}
{build_graph_profile(graph)}

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


def build_system_prompt(
    method: str, strategy_mode: str = "free", task: TaskSpec | None = None
) -> str:
    horizon_note = ""
    remove_note = remove_semantics_notes[spent]

    if task is not None:
        if task.adaptive:
            horizon_note = adaptive_timing_note
        elif any(op in task.allowed_ops for op in edge_ops):
            horizon_note = temporal_scheduling_note
        else:
            horizon_note = seed_timing_note

        remove_note = remove_semantics_notes[task.remove_semantics]

    # Only worth stating when the strategy may actually emit the op
    if "remove_node" not in (task.allowed_ops if task is not None else ()):
        remove_note = ""

    if strategy_mode == "scored":
        if method in ("one_shot", "evolve"):
            return scored_system + remove_note + horizon_note

        raise ValueError(
            f"strategy_mode='scored' is not supported for method {method!r}; "
            f"use one_shot or evolve"
        )

    # evolve generates plan_horizon strategies under the same contract as one_shot
    base = system_prompts["one_shot" if method == "evolve" else method]

    if method in ("one_shot", "evolve", "adaptive"):
        return base + remove_note + horizon_note

    return base + remove_note


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
    last_result: str | None = None,
) -> str:
    """
    `last_result` is the previous generation's paired delta.

    The parent's own summary carries the delta it was created with, but a
    candidate that did not become the parent never surfaces one — so without
    this the model's most recent edit gets no verdict at all.
    """
    inspiration_text = "".join(
        f"\nALTERNATIVE from the population (reward={record['reward']:.2f}):\n"
        f"```python\n{record['script']}\n```\n"
        for record in inspirations
    )
    error_text = (
        f"\nYour previous attempt failed with:\n{error}\n" if error else ""
    )
    last_text = f"\nYOUR LAST EDIT: {last_result}\n" if last_result else ""

    return f"""
You are evolving a population of strategies. Produce a NEW candidate by modifying the PARENT.
{last_text}
PARENT — the best in the population, which is NOT necessarily your last attempt
(reward={parent["reward"]:.2f}):
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
budget = {task.budget}   ({100.0 * task.budget / graph.num_nodes:.1f}% of nodes, max seeds total)
horizon = {task.horizon} (timesteps)

{build_graph_profile(graph)}

ALGORITHM MENU:
{build_algorithm_menu()}

Reply with exactly one name from the menu."""


def build_explanation_prompt(
    script: str, reward: float, history: list[dict], summary: str
) -> str:
    """
    Closing turn: describe the search and the winner in plain English.

    Sent on the generation thread, so the model can see the scripts it actually
    wrote rather than reconstructing them. `history` is passed anyway because a
    long run trims the middle of the thread while every reward survives here,
    and `script` because the winner is the max over iterations, not the last
    turn — without the echo the model would narrate the wrong algorithm.
    """
    iteration_lines = "\n".join(
        f"  iteration {record['iteration']}: "
        + (
            f"FAILED — {record['error'].splitlines()[0]}"
            if record.get("error")
            else f"reward={record['reward']:.2f} (best so far {record['best']:.2f})"
        )
        + (f" [operator={record['operator']}]" if record.get("operator") else "")
        for record in history
    )

    return f"""\
The search is over. Write the final report in plain English — no code blocks.

THE WINNING SCRIPT (reward={reward:.2f}) — this is the one to describe, and it
is NOT necessarily your last attempt:
```python
{script}
```
Its rollout diagnostics:
{summary}

Reward per iteration:
{iteration_lines or "  (none recorded)"}

Reply with GitHub-flavoured markdown using EXACTLY these headings, in this
order, and nothing before the first one:

## Summary
Two or three sentences: what the final algorithm is, and what it scored.

## Iteration log
One `### Iteration N — reward X` subsection per iteration above. For each: what
you were trying to fix, what you actually changed in the code, and whether it
worked. Be specific about the change ("raised the redundancy penalty from 1.0 to
2.5", not "tuned parameters"). If an iteration is no longer visible in this
conversation, say so for that iteration rather than inventing what you did.

## How the final algorithm works
A numbered step-by-step walkthrough of the winning script, in execution order.
Each step: what it computes, and why. Name the variables and functions as they
appear in the code so a reader can follow along with the source.

## Why it beats the baseline
What structural property of this graph the algorithm exploits, and which part of
the diagnostics above shows it working.

## Limitations
Where this algorithm would do badly, and what you would try next with more
iterations.

Write for someone who has the script in front of them but has not read this
conversation. Do not invent results that are not in the numbers above."""


def build_feedback_prompt(
    reward: float,
    summary: str,
    error: str | None = None,
    credit_report: str | None = None,
    reference_report: str | None = None,
    script: str | None = None,
    incumbent_script: str | None = None,
    incumbent_reward: float | None = None,
    delta_report: str | None = None,
) -> str:
    """
    `script` is the code that produced this reward; `incumbent_script` is the
    best-scoring code so far, shown only when the two differ.

    Both are also assistant turns in the conversation, so the echo is
    deliberately redundant — but it is what makes "EDIT this one" unambiguous,
    and it survives history trimming and any gateway that mangles multi-turn
    threads. The incumbent is what the next edit must be applied to: editing the
    latest attempt instead turns the loop into a random walk, because a
    regression then becomes the base for everything after it.
    """
    error_text = f"\nThe previous script raised an error:\n{error}\n" if error else ""
    credit_text = f"\n{credit_report}\n" if credit_report else ""
    reference_text = f"\n{reference_report}\n" if reference_report else ""
    delta_text = f"{delta_report}\n" if delta_report else ""
    script_text = (
        f"\nTHE SCRIPT THAT PRODUCED THIS RESULT:\n```python\n{script}\n```\n"
        if script
        else ""
    )

    if incumbent_script is not None:
        target_text = (
            f"\nYOUR BEST SCRIPT SO FAR (reward {incumbent_reward:.2f}) — THIS is "
            f"the one to edit, NOT the attempt above, which scored worse:\n"
            f"```python\n{incumbent_script}\n```\n"
        )
        instruction = (
            "EDIT THE BEST SCRIPT ABOVE to increase final spread. The attempt that "
            "just ran is shown so you can see what did not work — do not build on "
            "it. Change what the diagnostics say is weak and keep what is working."
        )
    else:
        target_text = ""
        instruction = (
            "That attempt is your best so far. EDIT it to increase final spread — "
            "change what the diagnostics say is weak and keep what is working, "
            "rather than starting a new design from scratch."
        )

    return f"""\
Your previous strategy achieved final spread (reward) = {reward}.
{delta_text}Trajectory summary: {summary}{reference_text}{error_text}{credit_text}{script_text}{target_text}
{instruction}
Reply with one ```python block containing the complete updated script."""
