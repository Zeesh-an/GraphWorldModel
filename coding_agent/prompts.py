import numpy as np

from coding_agent.rounds import round_batches, round_schedule
from coding_agent.types import GraphInfo, TaskSpec, myopic
from data.wm_simulator import blocked, spent
from coding_agent.executor import (
    allowed_imports,
    mc_blocked_algorithms,
    mc_blocked_dismantling,
    mc_blocked_localization,
    scored_blocked_primitives,
)
from coding_agent.tools.graph_profile import build_graph_profile
from coding_agent.tools.library_api import (
    build_adaptive_reference,
    build_algorithm_menu,
    build_algorithm_sources,
    build_api_reference,
    build_dismantling_menu,
    build_dismantling_reference,
    build_localization_menu,
    build_localization_reference,
    build_primitives_reference,
)

edge_ops = ("add_edge", "remove_edge", "set_edge_weight")

# The three problem families the prompt preamble has to distinguish. They are not
# rephrasings of each other: the objective sign, what a unit of budget buys, and
# whether the planner starts the cascade all differ, and a model told the wrong
# one optimizes the wrong direction with a perfectly valid program. The inverse
# family differs more than the other two do from each other — it emits no action
# at all.
seeding_brief = """\
You are designing an Influence Maximization algorithm as an executable Python script.

YOUR GOAL: MAXIMIZE the number of infected nodes at the end of the cascade. You
choose which nodes to seed; the cascade starts from your seeds and nowhere else.
"""

containment_brief = """\
You are designing a Critical Node Detection algorithm as an executable Python script.

YOUR GOAL: MINIMIZE the number of infected nodes at the end of the cascade. LOWER
IS BETTER, and every score you are shown reads that way.

You do NOT start the cascade. An OUTBREAK is already seeded at fixed source nodes
you did not choose and cannot change — they are listed in the task block below.
Your budget buys DELETIONS: each `remove_node` takes that node out of the graph
along with all its edges, so it can never be infected, never transmit, and never
counts toward the final total. You are cutting the routes the outbreak would
otherwise travel.

⚠️ REMOVING AN OUTBREAK SOURCE IS REJECTED. Deleting a source ends the outbreak
rather than containing it, which is a different problem. Filter the sources out of
your candidate set before you rank anything.

WHAT ACTUALLY WORKS HERE, AND WHAT DOES NOT:
- A blocker the cascade never reaches does nothing. Distance from the outbreak
  sources matters as much as centrality does.
- Blocking a hub is the obvious move and often the right one, but it is exactly
  what the `adaptive_degree` baseline already does. To beat it you have to find
  the nodes that carry the cascade BETWEEN regions, not merely the ones with the
  most neighbours.
- Structural dismantling (shrink the giant component) and containment (shrink the
  cascade) rank nodes DIFFERENTLY, sometimes in opposite orders. You are scored on
  the cascade.
"""


localization_brief = """\
You are designing a Source Localization algorithm as an executable Python script.

YOUR GOAL: given a graph and an OBSERVED diffusion state, recover the SEED SET
that produced it. You are not intervening in anything — you are inferring a hidden
cause. You are scored on F1 against the true source set, averaged over many
labelled cascades. HIGHER IS BETTER.

You will be given `observation`, a numpy float array of length num_nodes. Entry v
is P(node v was infected) at the end of the cascade, in [0, 1]. You return the
`budget` node ids you believe the cascade STARTED from.

WHAT MAKES THIS HARD, AND WHAT ACTUALLY WORKS:
- The problem is ILL-POSED. Diffusion is many-to-one: different seed sets produce
  the same final state, and a cascade that saturated retains almost no trace of
  where it began. Perfect F1 is not achievable and chasing it is not the goal —
  beating the reference table is.
- Sources are a TINY MINORITY of nodes, so accuracy is worthless as a signal.
  A rule that names nothing scores 90%+ accuracy and 0 F1.
- The strongest classical method is LPSI: label the infected +1 and the
  uninfected -1, propagate to convergence, and take the LOCAL MAXIMA of the
  converged field. It has no learning in it and it beats several deep generative
  methods on real cascades. Assume you have to beat it, not merely match it.
- High degree is a trap. A hub the cascade passed THROUGH looks exactly like a hub
  the cascade started from, and the reference diagnostics will tell you when you
  have fallen for it.
- Sources rarely cluster. Two adjacent nodes are usually one source and one of its
  first infections, so a separation constraint (one per community, one per k-hop
  ball, local maxima only) is usually worth more than a better score function.
"""


def _localization_rules(task: TaskSpec) -> str:
    """The inverse-task preamble: no actions, one method, and the forward oracle."""
    if task.forward_model:
        oracle_block = """\

THE FORWARD ORACLE — `self.predict_marginals(seeds)`:
    Returns a numpy array of length num_nodes: P(node infected at the end) if the
    cascade had STARTED from `seeds`. This is the simulator the observation came
    from, and it is the one thing a purely structural rule does not have.

    Use it to TEST a hypothesis: re-simulate a candidate source set and compare
    the prediction against what you observed.

        predicted = self.predict_marginals(candidate)
        error = float(((predicted - observation) ** 2).sum())

    It is not free. Every call is a full rollout and the calls are counted, so a
    scan over all nodes inside a per-pick loop will not finish. Narrow to a short
    candidate list with a cheap structural rule FIRST, then spend calls ranking it.
    You are also free to never call it at all — if structure alone wins, that is a
    result."""
    else:
        oracle_block = """\

NO FORWARD ORACLE IN THIS CONDITION. `self.predict_marginals` raises if you call
it. This arm exists to measure what pure structure achieves, so your algorithm
must be a structural inference rule over the graph and the observation alone."""

    return f"""\
{localization_brief}
OUTPUT FORMAT: reply with exactly ONE fenced ```python block and nothing else —
no prose before or after. The block contains import lines (if you need any) and
then exactly ONE class subclassing `Strategy`. Nothing else at module level: no
example usage, no test code.

IMPORTS: you MAY import any of {", ".join(allowed_imports)}. Use numpy for
anything you would otherwise write as a Python loop over all nodes.
Importing anything else is rejected.

AVAILABLE NAMES (already in your script's namespace — do NOT import these):
- `GraphInfo` : .num_nodes, .out_neighbors(node), .in_neighbors(node),
  .degree(node), .edge_index (2, E), .ic_probs (E,).
- `localization_algorithms` and `localization_scorers` : the published baselines
  and their per-node score vectors (API below).
- `primitives` : structural helpers (API below).
- `ActionOp` and `State` exist but you will not need them — this task emits no
  actions.

WHAT YOU IMPLEMENT:
    def localize(self, graph, observation, budget) -> list[int]
        The node ids you believe started the cascade. AT MOST `budget` of them,
        no duplicates, every id in [0, num_nodes). Returning more than `budget` is
        REJECTED — extra names would buy recall for free.

    def source_scores(self, graph, observation) -> np.ndarray     (OPTIONAL)
        One float per node, higher meaning more likely to be a source. Implement
        it and your AUC is measured on that real ranking; omit it and AUC falls
        back to the ORDER of the list localize() returned, which ties every node
        you did not name. It costs a few lines and it is a reported column.
{oracle_block}
"""


def _common_rules(task: TaskSpec | None) -> str:
    """The preamble, with the problem family and the budgeted op filled in."""
    if task is not None and task.recovers:
        return _localization_rules(task)

    contains = task is not None and task.contains
    brief = containment_brief if contains else seeding_brief

    if contains:
        budget_rules = """\
- A blocker is ActionOp("remove_node", node). Emit at most `budget` remove_node
  actions in total. You do NOT need to emit the incident remove_edge ops — the
  harness expands each removal into a full node deletion for you.
- Removing the same node twice is REJECTED: it spends two units of budget on one node.
- Emitting add_node is REJECTED. You are not seeding this cascade."""
    else:
        budget_rules = """\
- A seed is ActionOp("add_node", node). Emit at most `budget` add_node actions in total.
- Seeding the same node twice is REJECTED: it spends two units of budget on one node."""

    library_line = (
        "- `algorithms`, `adaptive_algorithms`, `dismantling_algorithms` and\n"
        "  `primitives` modules (API below). `dismantling_algorithms` members\n"
        "  return node-REMOVAL sets and are the published baselines for this task."
        if contains
        else "- `algorithms`, `adaptive_algorithms` and `primitives` modules (API below).\n"
        "  `adaptive_algorithms` members are per-ROUND policies and are only callable\n"
        "  from act() on an adaptive task; the reference below lists them when so."
    )

    return f"""\
{brief}
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
{library_line}

ACTION RULES:
{budget_rules}
- Node ids must be in [0, num_nodes).
- add_edge / remove_edge / set_edge_weight rewire the graph the cascade runs on;
  under LT edge weights are ignored and only the structural change applies.
- what remove_node does is stated below; read it before using the op.
"""


# The influence-maximization preamble, kept as a module-level name because the
# system_prompts table below is built from it. build_system_prompt swaps in the
# containment variant per task.
common_rules = _common_rules(None)

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

# The containment counterpart. `one_shot_exemplars` is worse than useless here:
# both of its programs emit add_node, which this task REJECTS, so the model would
# spend its first iteration repairing a syntax-valid but disallowed plan.
containment_exemplars = """\
EXAMPLES — two strategies at the level you should START from, not finish at.

Example 1, outbreak-weighted cut scoring (rank by how much a node carries the
cascade AWAY from the sources, rather than by raw degree):
```python
class OutbreakCut(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        outbreak = self.outbreak          # the source nodes, injected by the harness
        # Reach probability by one-hop propagation from the sources: a cheap proxy
        # for "will the cascade ever get here"
        exposure = {node: 0.0 for node in range(graph.num_nodes)}
        wave = {int(node): 1.0 for node in outbreak}
        for _ in range(4):
            nxt = {}
            for node, mass in wave.items():
                exposure[node] += mass
                for neighbor in graph.out_neighbors(node):
                    nxt[neighbor] = nxt.get(neighbor, 0.0) + mass * 0.5
            wave = nxt

        # Score = exposure * onward reach, discounted for neighbours already cut
        removed, blocked_nbrs = [], set()
        for _ in range(budget):
            best, best_score = -1, float("-inf")
            for node in range(graph.num_nodes):
                if node in removed or node in outbreak:
                    continue
                onward = sum(
                    1 for other in graph.out_neighbors(node) if other not in removed
                )
                penalty = 0.5 if node in blocked_nbrs else 1.0
                score = exposure[node] * onward * penalty
                if score > best_score:
                    best, best_score = node, score
            if best < 0:
                break
            removed.append(best)
            blocked_nbrs.update(graph.out_neighbors(best))

        return [[ActionOp("remove_node", int(node)) for node in removed]] + [
            [] for _ in range(horizon)
        ]
```

Example 2, adaptive degree restricted to the outbreak's reachable set (the
published HDA baseline, narrowed — nodes the cascade cannot reach are free to
ignore, which spends the whole budget where it can matter):
```python
class ReachableHDA(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        reach, wave = set(self.outbreak), set(self.outbreak)
        for _ in range(horizon):
            wave = {
                other for node in wave for other in graph.out_neighbors(node)
            } - reach
            if not wave:
                break
            reach |= wave

        candidates = sorted(reach - set(self.outbreak))
        degree = {node: graph.degree(node) for node in candidates}
        removed = []
        for _ in range(min(budget, len(candidates))):
            pick = max((node for node in candidates if node not in removed),
                       key=lambda node: degree[node], default=None)
            if pick is None:
                break
            removed.append(pick)
            for neighbor in graph.out_neighbors(pick) + graph.in_neighbors(pick):
                if neighbor in degree:
                    degree[neighbor] -= 1

        return [[ActionOp("remove_node", int(node)) for node in removed]] + [
            [] for _ in range(horizon)
        ]
```
`self.outbreak` is set on your Strategy instance before plan_horizon is called;
it is the tuple of source node ids, and it is also printed in the task block.
Beat both. Combining their ideas, or replacing them, are both fair game.
"""

# The inverse-task counterpart. Neither of the other two exemplar sets works here:
# both emit action bags from plan_horizon, and this task calls localize and
# rejects actions outright, so showing them would spend the first iteration on a
# repair turn for a contract the model was never asked to implement.
localization_exemplars = """\
EXAMPLES — two algorithms at the level you should START from, not finish at.

Example 1, LPSI with a community separation constraint (pure structure, no
forward model — this is roughly the classical bar):
```python
import numpy as np

class FieldMaxima(Strategy):
    def source_scores(self, graph, observation):
        # Label propagation over the observed state: +1 infected, -1 uninfected
        labels = np.where(observation >= 0.5, 1.0, -1.0)
        sources, targets = graph.edge_index[0], graph.edge_index[1]
        degree = np.zeros(graph.num_nodes)
        np.add.at(degree, sources, 1.0)
        np.add.at(degree, targets, 1.0)
        scale = np.where(degree > 0, degree, 1.0) ** -0.5
        weight = scale[sources] * scale[targets]

        field = labels.copy()
        for _ in range(60):
            spread = np.zeros(graph.num_nodes)
            np.add.at(spread, targets, weight * field[sources])
            np.add.at(spread, sources, weight * field[targets])
            field = 0.5 * spread + 0.5 * labels

        # A node the observation never saw infected cannot be a source
        return np.where(observation >= 0.5, field, field.min() - 1.0)

    def localize(self, graph, observation, budget):
        field = self.source_scores(graph, observation)
        communities = primitives.detect_communities(graph)

        # One candidate per community, and only local maxima of the field:
        # adjacent high-field nodes are one source plus its first infection
        best_per_community = {}
        for node in range(graph.num_nodes):
            if observation[node] < 0.5:
                continue
            neighbours = graph.out_neighbors(node) + graph.in_neighbors(node)
            if any(field[other] > field[node] for other in neighbours):
                continue
            group = communities.get(node, -1)
            if group not in best_per_community or field[node] > field[best_per_community[group]]:
                best_per_community[group] = node

        ranked = sorted(best_per_community.values(), key=lambda v: -field[v])
        # Top up if there were fewer communities than the budget
        for node in np.argsort(-field):
            if len(ranked) >= budget:
                break
            if int(node) not in ranked:
                ranked.append(int(node))

        return [int(node) for node in ranked[:budget]]
```

Example 2, propose-then-test: a cheap structural shortlist, then the forward
oracle ranks it by how well re-simulating it reproduces the observation:
```python
import numpy as np

class ResimulationGreedy(Strategy):
    def source_scores(self, graph, observation):
        return localization_scorers.lpsi(graph, observation)

    def localize(self, graph, observation, budget):
        field = self.source_scores(graph, observation)
        field = np.where(observation >= 0.5, field, field.min() - 1.0)
        # Shortlist FIRST: a forward call per node per pick does not finish
        shortlist = [int(node) for node in np.argsort(-field)[:25]]

        selected = []
        for _ in range(budget):
            best, best_error = None, float("inf")
            for candidate in shortlist:
                if candidate in selected:
                    continue
                predicted = self.predict_marginals(selected + [candidate])
                error = float(((predicted - observation) ** 2).sum())
                if error < best_error:
                    best, best_error = candidate, error
            if best is None:
                break
            selected.append(best)

        return selected
```
Beat both. Combining their ideas, or replacing them, are both fair game — and if
the forward oracle turns out not to help, say so with a program that does not use it.
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

# The inverse-task counterpart. There is no horizon to schedule across at all:
# nothing is emitted, nothing is timed, and the model needs to be told that
# explicitly or it will try to return an action plan it was never asked for.
localization_timing_note = """\

THERE IS NO HORIZON TO PLAN ACROSS. You emit no actions and nothing you return is
scheduled. `horizon` appears in the task block only because it is how long the
cascade you are inverting ran for — a longer cascade means a more saturated
observation and therefore LESS information about where it started, which is worth
knowing when you decide how much to trust the observed state.
"""

# The containment counterpart of seed_timing_note, and its mirror image: the
# reason to act at t=0 is not "more time to spread" but "the node is still
# uninfected". Blocking a node the cascade already passed through removes its
# onward reach and nothing else, so a late removal is strictly weaker.
containment_timing_note = """\

USING THE HORIZON: put every removal in element 0 and leave the rest of the plan
empty. Under a progressive cascade a node blocked at t>0 may already be infected,
and blocking it then only cuts its ONWARD reach — the infections it already caused
stand. Pre-emptive removal is strictly stronger, so all of your effort belongs in
WHICH nodes you cut, not when.
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

# METHOD BODIES, dict-keyed by method name. The shared preamble is prepended by
# build_system_prompt rather than baked in here, because it differs by problem
# family (seed to maximize vs remove to minimize) and the bodies do not.
method_bodies = {
    "one_shot": """\

METHOD: ONE-SHOT SUPER-ALGORITHM.
Implement `plan_horizon(self, graph, budget, horizon) -> list[list[ActionOp]]`.
Return a list of length (horizon+1): element t is the action bag applied at timestep t.
This is your whole multi-timestep plan, decided up front.

""",
    "per_step": """\

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
    "windowed": """\

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
    "adaptive": """\

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
    "localize": """\

METHOD: SOURCE-SET INFERENCE.
Implement `localize(self, graph, observation, budget) -> list[int]`, and
optionally `source_scores(self, graph, observation) -> np.ndarray`.

You are called ONCE PER LABELLED CASCADE, with that cascade's own observation and
its own source count as `budget`. Your score is the mean F1 across all of them, so
a rule that nails one episode and collapses on the rest loses to a rule that is
uniformly decent. Write an ALGORITHM, not a fit to one graph: hardcoded node ids
will score zero on every other episode.

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


max_listed_outbreak = 60


def build_outbreak_block(task: TaskSpec) -> str:
    """
    The source nodes, spelled out, or nothing for a task that seeds its own cascade.

    Listed in full rather than summarized: the whole containment problem is
    "where is it coming from", and a model that has to infer the sources from the
    graph profile is solving a different, harder problem than the one posed.
    """
    if not task.outbreak:
        return ""

    sources = list(task.outbreak)
    listed = ", ".join(str(node) for node in sources[:max_listed_outbreak])
    overflow = (
        f", … and {len(sources) - max_listed_outbreak} more (all of them are on "
        f"`self.outbreak`)"
        if len(sources) > max_listed_outbreak
        else ""
    )

    return (
        f"\nOUTBREAK SOURCES ({len(sources)} nodes, fixed, NOT yours to choose; also\n"
        f"available inside your Strategy as `self.outbreak`): {listed}{overflow}\n"
        f"The cascade starts here at t=0 and spreads for {task.horizon} timesteps.\n"
    )


max_listed_episodes = 6


def build_observation_block(task: TaskSpec) -> str:
    """
    What `y` actually is, and how many episodes the score averages over.

    Spelled out because the observation mode changes the problem: an MC marginal
    is a continuous, averaged view of the last wave, while a binarized draw is one
    realization. Ours is strictly MORE informative than the published protocol's
    (research/source_localization.md §2.9 risk 5), so a model told the wrong one
    calibrates its threshold against a distribution it will not see.
    """
    if not task.recovers:
        return ""

    instances = list(task.instances)
    if not instances:
        return ""

    counts = [instance.source_count for instance in instances]
    infected = [instance.infected_count for instance in instances]
    binary = all(set(np.unique(instance.observation)) <= {0.0, 1.0} for instance in instances)

    lines = [
        "",
        f"THE EPISODES YOU ARE SCORED ON ({len(instances)} labelled cascades, "
        f"mean F1 across all of them):",
        f"  sources per episode: {min(counts)}-{max(counts)} nodes "
        f"({100.0 * np.mean(counts) / instances[0].num_nodes:.1f}% of N on average) "
        f"— you are handed the exact count as `budget`",
        f"  observed infected per episode: {min(infected)}-{max(infected)} nodes "
        f"({100.0 * np.mean(infected) / instances[0].num_nodes:.1f}% of N on average)",
    ]

    if binary:
        lines.append(
            "  observation is a BINARIZED single draw: every entry is exactly 0.0 "
            "or 1.0, so `observation >= 0.5` is the infected set and there is no "
            "extra signal in the magnitudes"
        )
    else:
        lines.append(
            "  observation is a CONTINUOUS Monte-Carlo marginal: entries strictly "
            "between 0 and 1 are nodes the last wave reached only sometimes, so "
            "the fractional values carry real information about the cascade's "
            "FRONTIER — a node at 0.3 was reached late, a node at 1.0 early"
        )

    lines.append(
        f"  the cascades ran for up to {max(instance.horizon for instance in instances)} "
        f"timesteps; a longer cascade is a more saturated observation and therefore "
        f"a harder inversion"
    )
    lines.append("")

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
        # out inside score()/schedule()/source_score(), where they can be mutated
        if task.recovers:
            menu = build_localization_menu()
        elif task.contains:
            menu = build_dismantling_menu()
        else:
            menu = build_algorithm_menu()

        reference = (
            "PRIMITIVES API (available as `primitives`; spread-simulation "
            "functions are NOT available):\n"
            f"{build_primitives_reference(exclude=scored_blocked_primitives)}\n\n"
            "ALGORITHM IDEAS (NOT callable — steal the ideas into your "
            f"{'source_score()' if task.recovers else 'score()'}):\n"
            f"{menu}"
        )
        final_line = "Write the ScoredStrategy subclass now."
    elif task.recovers:
        # An inverse task's library is the localization pool. The IM algorithms
        # return seed sets to maximize with and the dismantlers return deletions;
        # neither is an inference rule, and showing them invites a program that
        # confuses "where would a cascade start best" with "where did this one".
        reference = (
            build_localization_reference(
                exclude=() if allow_mc_algorithms else mc_blocked_localization
            )
            + "\n\nPRIMITIVES  (from coding_agent.tools.primitives, imported as "
            "`primitives`)\n"
            + build_primitives_reference(exclude=scored_blocked_primitives)
        )
        final_line = f"Write the Strategy now (method = {method})."
    else:
        blocked = () if allow_mc_algorithms else mc_blocked_algorithms
        # Signatures alone do not teach the idiom — the model reproduces the
        # library's shape much more reliably once it has read a few of them
        # The adaptive policies are the published baselines for this task, so an
        # adaptive prompt that hides them asks the model to reinvent AdaptGreedy
        adaptive_reference = (
            f"\n\n{build_adaptive_reference()}" if task.adaptive else ""
        )
        # ...and the dismantling library is the published baseline set for
        # containment, for exactly the same reason
        dismantling_reference = (
            "\n\n"
            + build_dismantling_reference(
                exclude=() if allow_mc_algorithms else mc_blocked_dismantling
            )
            if task.contains
            else ""
        )
        reference = (
            f"LIBRARY API:\n{build_api_reference(exclude=blocked)}"
            f"{adaptive_reference}{dismantling_reference}\n\n"
            f"{build_algorithm_sources()}"
        )
        final_line = f"Write the Strategy now (method = {method})."

    if task.recovers:
        budget_unit = "max sources to name per episode"
        objective_line = (
            "recover the seed set that produced the observation — MAXIMIZE F1 "
            "against the true sources (higher is better)"
        )
        ops_line = "allowed_ops = none — this task emits no actions\n"
    else:
        budget_unit = "max removals total" if task.contains else "max seeds total"
        objective_line = (
            "MINIMIZE the final infected count (lower is better)"
            if task.contains
            else task.objective
        )
        ops_line = (
            f"allowed_ops = {', '.join(task.allowed_ops)}   (any other op is REJECTED)\n"
        )

    return f"""\
TASK: {task.task} — {objective_line}
diffusion_model = {task.diffusion_model}
budget = {task.budget}   ({100.0 * task.budget / graph.num_nodes:.1f}% of nodes, {budget_unit})
horizon = {task.horizon} (timesteps)
{ops_line}{build_outbreak_block(task)}{build_round_block(task)}{build_observation_block(task)}
{build_graph_profile(graph)}

{reference}

{final_line}"""


# Scored mode: the agent edits an algorithm's internals (score/schedule hooks),
# never whole programs and never compositions over the library
def _scored_localization_system(task: TaskSpec) -> str:
    """
    Scored mode for the inverse task (research/source_localization.md §2.4.2).

    The same trick as the intervention half — a fixed harness the agent cannot
    override, with one hook it can — and it is a better fit here than anywhere
    else: LPSI, the Comin-Costa centralities and rumor centrality are ALL exactly
    node-scoring functions over the observed state, so the constrained search space
    is directly comparable to the classical methods rather than a subset of them.
    """
    oracle_line = (
        "- `self.predict_marginals(seeds)` -> np.ndarray of P(infected at the end) "
        "if the\n  cascade had started from `seeds`. Every call is a full rollout "
        "and calls are\n  counted, so use it sparingly inside score() — it runs "
        "once per candidate per pick."
        if task.forward_model
        else "- `self.predict_marginals` RAISES in this condition: this arm has no "
        "forward\n  model by design. Score from structure and the observation alone."
    )

    return f"""\
You are designing the SCORING RULE of a Source Localization algorithm — not a
whole program.

Given a graph and an OBSERVED diffusion state, the task is to recover the SEED SET
that produced it. You are scored on F1 against the true sources, averaged over
many labelled cascades. HIGHER IS BETTER.

A fixed harness (ScoredStrategy.localize) greedily names the highest-scoring node
until the budget is spent. You may override ONLY:

- source_score(self, node, graph, observation, selected) -> float
    Called for every candidate node at every pick. `observation[v]` is
    P(node v was infected) at the end of the cascade, in [0, 1]. `selected` is the
    tuple of sources already named — use it to penalize a candidate whose
    neighbourhood an earlier pick already explains, because sources rarely cluster.
    Higher score = named sooner. Return float("-inf") to rule a node out.
- source_scores(self, graph, observation) -> np.ndarray            (optional)
    One float per node, for the AUC column. Omit it and AUC falls back to the
    order the harness happened to name nodes in.

RULES:
- Overriding localize (or plan_horizon) is REJECTED by the executor.
- `localization_algorithms.*` does NOT exist here. Write your own scoring logic
  from graph structure, the observation, and `primitives`.
- Reply with exactly ONE fenced ```python block containing ONE class subclassing
  ScoredStrategy. No imports, no module-level code, no prose.
- `GraphInfo` has .num_nodes, .out_neighbors(node), .in_neighbors(node),
  .degree(node), .edge_index, .ic_probs.
{oracle_line}

REPLY SHAPE (adapt the logic — improve on it, do not return it unchanged):
```python
class MyLocalizer(ScoredStrategy):
    def source_score(self, node, graph, observation, selected):
        if observation[node] < 0.5:
            return float("-inf")            # never infected, cannot be a source
        neighbours = graph.out_neighbors(node) + graph.in_neighbors(node)
        infected_around = sum(1 for other in neighbours if observation[other] >= 0.5)
        # Penalize a candidate an already-named source sits next to
        overlap = sum(1 for other in neighbours if other in selected)
        return infected_around - 3.0 * overlap
```
"""


def _scored_system(task: TaskSpec | None) -> str:
    if task is not None and task.recovers:
        return _scored_localization_system(task)

    contains = task is not None and task.contains
    problem = (
        "a Critical Node Detection algorithm" if contains else "an Influence Maximization algorithm"
    )
    picks = "removes" if contains else "seeds"
    goal = (
        "Higher score = removed sooner. You are MINIMIZING the final infected "
        "count, so a high score should mean 'cutting this node hurts the cascade "
        "most'"
        if contains
        else "Higher score = picked sooner"
    )
    example = (
        """\
```python
class MyScorer(ScoredStrategy):
    def score(self, node, selected, graph):
        overlap = len(set(graph.out_neighbors(node)) & set(selected))
        return graph.degree(node) - 2.0 * overlap
```"""
    )

    return f"""\
You are designing the SCORING RULE of {problem} — not a whole program.

A fixed harness (ScoredStrategy.plan_horizon) greedily picks the highest-score node
until the budget is spent, then calls schedule() to place the picked {picks} across
timesteps. You may override ONLY these two methods:

- score(self, node, selected, graph) -> float
    Called for every candidate node at every pick; `selected` is the tuple of
    already-chosen nodes. {goal}. This is where your
    algorithm lives: combine structural signals, penalize redundancy, adapt to
    what the diagnostics reveal.
- schedule(self, seeds, graph, horizon) -> list[list[ActionOp]]   (optional)
    Element t is the action bag applied at timestep t. Default: all picks at t=0.

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
{example}
"""


scored_system = _scored_system(None)


def build_system_prompt(
    method: str, strategy_mode: str = "free", task: TaskSpec | None = None
) -> str:
    horizon_note = ""
    remove_note = remove_semantics_notes[spent]
    contains = task is not None and task.contains
    recovers = task is not None and task.recovers

    if task is not None:
        if recovers:
            horizon_note = localization_timing_note
        elif task.adaptive:
            horizon_note = adaptive_timing_note
        elif contains:
            horizon_note = containment_timing_note
        elif any(op in task.allowed_ops for op in edge_ops):
            horizon_note = temporal_scheduling_note
        else:
            horizon_note = seed_timing_note

        remove_note = remove_semantics_notes[task.remove_semantics]

    # Only worth stating when the strategy may actually emit the op, which an
    # inverse task never does
    if recovers or "remove_node" not in (task.allowed_ops if task is not None else ()):
        remove_note = ""

    if strategy_mode == "scored":
        if method in ("one_shot", "evolve"):
            return _scored_system(task) + remove_note + horizon_note

        raise ValueError(
            f"strategy_mode='scored' is not supported for method {method!r}; "
            f"use one_shot or evolve"
        )

    # The CONTRACT is decided by the task, not by the method: `evolve` is a
    # population search over programs, and what those programs implement is
    # plan_horizon on an intervention task and localize on an inverse one. `method`
    # only picks which body describes the outer loop.
    if recovers:
        resolved = "localize"
    else:
        # evolve generates plan_horizon strategies under the same contract as one_shot
        resolved = "one_shot" if method == "evolve" else method

    base = _common_rules(task) + method_bodies[resolved]

    # Worked programs, matched to the family: the IM exemplars emit add_node,
    # which a containment task REJECTS and an inverse task has no use for at all,
    # so showing the wrong set would cost the search its first iteration on a
    # repair turn for a contract nobody asked for
    if resolved == "localize":
        base += localization_exemplars
    elif resolved == "one_shot":
        base += containment_exemplars if contains else one_shot_exemplars

    if resolved == "localize" or method in ("one_shot", "evolve", "adaptive"):
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

containment_routing_system = """\
You are an algorithm-selection router for Critical Node Detection.
You will be given a task, a graph description, and a menu of classical network
DISMANTLING algorithms. Each returns a set of nodes to DELETE from the graph. Pick
the single one most likely to MINIMIZE the final infected count on this graph.

Reply with EXACTLY ONE algorithm name from the menu — no code, no punctuation,
no explanation."""


localization_routing_system = """\
You are an algorithm-selection router for Source Localization.
You will be given a task, a graph description, a summary of the cascades to invert,
and a menu of classical source-localization algorithms. Each takes the observed
diffusion state and returns the nodes it believes STARTED the cascade. Pick the
single one most likely to MAXIMIZE F1 against the true sources on these instances.

Reply with EXACTLY ONE algorithm name from the menu — no code, no punctuation,
no explanation."""


def build_routing_system(task: TaskSpec | None = None) -> str:
    if task is not None and task.recovers:
        return localization_routing_system

    return (
        containment_routing_system
        if task is not None and task.contains
        else routing_system
    )


def build_routing_prompt(task: TaskSpec, graph: GraphInfo) -> str:
    if task.recovers:
        budget_unit = "max sources to name per episode"
        objective_line = "MAXIMIZE F1 against the true source set (higher is better)"
        menu = build_localization_menu()
    else:
        budget_unit = "max removals total" if task.contains else "max seeds total"
        objective_line = (
            "MINIMIZE the final infected count (lower is better)"
            if task.contains
            else task.objective
        )
        menu = build_dismantling_menu() if task.contains else build_algorithm_menu()

    return f"""\
TASK: {task.task} — {objective_line}
diffusion_model = {task.diffusion_model}
budget = {task.budget}   ({100.0 * task.budget / graph.num_nodes:.1f}% of nodes, {budget_unit})
horizon = {task.horizon} (timesteps)
{build_outbreak_block(task)}{build_observation_block(task)}
{build_graph_profile(graph)}

ALGORITHM MENU:
{menu}

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
    objective: str = "final spread",
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
            f"EDIT THE BEST SCRIPT ABOVE to improve {objective}. The attempt that "
            f"just ran is shown so you can see what did not work — do not build on "
            f"it. Change what the diagnostics say is weak and keep what is working."
        )
    else:
        target_text = ""
        instruction = (
            f"That attempt is your best so far. EDIT it to improve {objective} — "
            f"change what the diagnostics say is weak and keep what is working, "
            f"rather than starting a new design from scratch."
        )

    return f"""\
Your previous strategy achieved {objective} (reward) = {reward}.
{delta_text}Trajectory summary: {summary}{reference_text}{error_text}{credit_text}{script_text}{target_text}
{instruction}
Reply with one ```python block containing the complete updated script."""
