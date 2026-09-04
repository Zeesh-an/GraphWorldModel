import difflib

import numpy as np

from coding_agent.probes import max_probes_per_generation
from coding_agent.rounds import round_batches, round_schedule
from coding_agent.types import GraphInfo, TaskSpec, myopic
from data.wm_simulator import blocked, spent
from coding_agent.executor import (
    allowed_imports,
    mc_blocked_algorithms,
    mc_blocked_blocking,
    mc_blocked_dismantling,
    mc_blocked_immunization,
    mc_blocked_localization,
    mc_blocked_prediction,
    mc_blocked_reconstruction,
    scored_blocked_primitives,
)
from coding_agent.tools.graph_profile import build_graph_profile
from coding_agent.tools.library_api import (
    build_immunization_menu,
    build_immunization_reference,
    build_adaptive_reference,
    build_algorithm_menu,
    build_algorithm_sources,
    build_api_reference,
    build_blocking_menu,
    build_blocking_reference,
    build_dismantling_menu,
    build_dismantling_reference,
    build_localization_menu,
    build_localization_reference,
    build_prediction_menu,
    build_prediction_reference,
    build_primitives_reference,
    build_reconstruction_menu,
    build_reconstruction_reference,
)

edge_ops = ("add_edge", "remove_edge", "set_edge_weight")

# The three problem families the prompt preamble has to distinguish. They are not
# rephrasings of each other: the objective sign, what a unit of budget buys, and
# whether the planner starts the cascade all differ, and a model told the wrong
# one optimizes the wrong direction with a perfectly valid program. The inverse
# family differs more than the other two do from each other: it emits no action
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
you did not choose and cannot change: they are listed in the task block below.
Your budget buys DELETIONS: each `remove_node` takes that node out of the graph
along with all its edges, so it can never be infected, never transmit, and never
counts toward the final total. You are cutting the routes the outbreak would
otherwise travel.

Warning: REMOVING AN OUTBREAK SOURCE IS REJECTED. Deleting a source ends the outbreak
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


blocking_briefs = {
    "counter_seed": """\
You are designing an Influence Blocking algorithm as an executable Python script.

YOUR GOAL: MINIMIZE how far a RUMOUR spreads. LOWER IS BETTER, and every score you
are shown reads that way.

There are TWO cascades on this graph. The rumour was seeded first, at fixed source
nodes you did not choose and cannot change (listed below), and it is ALREADY
spreading at t=0. Your budget buys POSITIVE seeds: nodes you activate with a
counter-cascade that spreads by the same rules and competes for the same nodes. A
node taken by one cascade is closed to the other forever.

WHAT ACTUALLY WORKS HERE, AND WHAT DOES NOT:
- ARRIVING FIRST IS THE ENTIRE GAME. A node you reach after the rumour does is worth
  nothing: it is already lost. Read the tie-break rule stated below: it decides who
  wins a node you both reach on the SAME step, and it is the difference between a
  seed being worth something and worth nothing.
- Your score counts only nodes the rumour WOULD have infected. Protecting a node it
  was never going to reach scores exactly zero, however central that node is.
- HIGH DEGREE IS A TRAP HERE, and this is the opposite of influence maximization.
  The published finding is blunt: the degree heuristic "cannot be used for influence
  blocking maximization at all". A hub far from the rumour is useless.
- PROXIMITY is the strong cheap baseline: seed the rumour's own out-neighbours and
  you intercept it before it gets moving. It is also unstable across graphs, which
  is where a better algorithm has room to win.
""",
    "node_block": """\
You are designing an Influence Minimization algorithm as an executable Python script.

YOUR GOAL: MINIMIZE how far a RUMOUR spreads. LOWER IS BETTER, and every score you
are shown reads that way.

The rumour was seeded first, at fixed source nodes you did not choose and cannot
change (listed below), and it is ALREADY spreading at t=0. Your budget buys
DELETIONS: each `remove_node` takes that node out of the graph along with all its
edges, so the rumour can never pass through it.

WHAT ACTUALLY WORKS HERE, AND WHAT DOES NOT:
- Your score counts only nodes the rumour WOULD have infected. Deleting a node it
  was never going to reach scores exactly zero.
- The nodes that matter are the ones the rumour has to pass THROUGH: cut points on
  its routes out of the sources, not the highest-degree nodes in the graph.
- The published trivial heuristic here is "rank the sources' out-neighbours by
  degree", and it beats two principled VLDB algorithms in a fifth of their own
  table's cells. Assume you have to beat it.
""",
    "edge_block": """\
You are designing a Link Blocking algorithm as an executable Python script.

YOUR GOAL: MINIMIZE how far a RUMOUR spreads. LOWER IS BETTER, and every score you
are shown reads that way.

The rumour was seeded first, at fixed source nodes you did not choose and cannot
change (listed below), and it is ALREADY spreading at t=0. Your budget buys EDGE
CUTS: each `remove_edge` deletes one arc `u -> v`, so the rumour can no longer
traverse it. Nodes stay in the graph; only the route goes.

WHAT ACTUALLY WORKS HERE, AND WHAT DOES NOT:
- An arc is worth cutting in proportion to how much traffic it CARRIES: how likely
  the rumour is to reach its tail, times how much of the graph only hangs off its
  head. An arc into a node with another way in buys almost nothing.
- Your score counts only nodes the rumour WOULD have infected, so an arc outside its
  reachable region scores exactly zero.
- High-probability arcs are not automatically the right ones. A p = 0.9 arc into a
  well-connected node is redundant; a p = 0.2 arc that is the only way into a whole
  region is not.
""",
    "weight_block": """\
You are designing a Link-Weight Reduction algorithm as an executable Python script.

YOUR GOAL: MINIMIZE how far a RUMOUR spreads. LOWER IS BETTER, and every score you
are shown reads that way.

The rumour was seeded first, at fixed source nodes you did not choose and cannot
change (listed below), and it is ALREADY spreading at t=0. Your budget buys WEIGHT
REDUCTIONS: `ActionOp("set_edge_weight", u, v, w)` sets the arc's transmission
probability to `w`. You may only LOWER an arc: a weight above its current
probability is REJECTED, and `w = 0.0` is the same thing as cutting it.

WHAT ACTUALLY WORKS HERE, AND WHAT DOES NOT:
- Choosing `w = 0.0` on the right arcs is a strictly stronger move than any
  intermediate value, so spend your effort on WHICH arcs, not on how far to turn
  each one down. Intermediate values only matter if you have a reason to prefer
  damping many arcs over cutting few.
- An arc is worth reducing in proportion to how much traffic it carries: how likely
  the rumour reaches its tail, times how much only hangs off its head.
- Your score counts only nodes the rumour WOULD have infected.
""",
}


localization_brief = """\
You are designing a Source Localization algorithm as an executable Python script.

YOUR GOAL: given a graph and an OBSERVED diffusion state, recover the SEED SET
that produced it. You are not intervening in anything: you are inferring a hidden
cause. You are scored on CONSISTENCY: the set you name is rolled forward through
the diffusion model and the reward is minus the mean squared error between that
re-simulated P(infected) and the observed state, averaged over many cascades.
HIGHER IS BETTER and 0 is perfect. The true sources are never shown to you and
never used to score you: your algorithm has to explain what was observed.

You will be given `observation`, a numpy float array of length num_nodes. Entry v
is P(node v was infected) at the end of the cascade, in [0, 1]. You return the
`budget` node ids you believe the cascade STARTED from.

WHAT MAKES THIS HARD, AND WHAT ACTUALLY WORKS:
- The problem is ILL-POSED. Diffusion is many-to-one: different seed sets produce
  the same final state, and a cascade that saturated retains almost no trace of
  where it began. A perfect consistency is not achievable and chasing it is not
  the goal, beating the reference table is.
- A set that OVERSHOOTS (its cascade reaches nodes the observation says stayed
  clean) and one that UNDERSHOOTS (parts of the observed region stay unexplained)
  both lose; the feedback names the nodes on each side, so read it.
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


reconstruction_brief = """\
You are designing a Cascade Reconstruction algorithm as an executable Python script.

YOUR GOAL: a diffusion already happened on this graph and you only saw part of it.
Recover the WHOLE HISTORY, which nodes were infected, at which timestep each one
activated, and WHO INFECTED WHOM. You are not intervening in anything; you are
reconstructing the past. HIGHER IS BETTER.

YOUR SCORE IS THE LIKELIHOOD OF YOUR HISTORY, NOT ITS AGREEMENT WITH A KEY:

    reward = [log p(sources) + log p(transitions | kernel)] / N  -  (fraction of the observation you contradict)

The history you return implies a sequence of states; the diffusion kernel assigns
each asserted transition a probability (a node you activate at t must have an
active in-neighbour at t-1 whose arc transmits; nodes you leave silent must be
ones the kernel did not expect to activate), and every node you declare a SOURCE
is an exogenous event of prior probability 1/N, so each extra source costs about
log N. Those log-probabilities, summed and divided by the node count, are the
first term. The second term is the share of reported nodes you dropped, reported
times you moved, or nodes you named that a final snapshot says stayed clean.
HIGHER IS BETTER; the true history is never shown to you and never used to score
you. Thirteen years of published work says the node set is nearly free and the
who-infected-whom EDGES are the hard half: an implausible parent, or a source
that is really somebody's child, is exactly what the likelihood punishes. **Spend
your effort on the parents and the times.**

WHAT MAKES THIS HARD, AND WHAT ACTUALLY WORKS:
- The observation is a SUBSET. Nodes the cascade infected but nobody reported are
  yours to infer, and a report set that looks disconnected is usually one cascade
  with the connecting nodes missing: filling them in is most of the recall.
- TIME IS THE STRUCTURE. If you were given activation times, they constrain the
  tree completely: an edge `u -> v` is only possible when `t(u) < t(v)`. If you
  were not, you have to infer an ordering before you can infer any parent.
- The classical bar is an ORDERED STEINER TREE: connect the reported nodes with
  the cheapest tree whose rooted paths respect the observed order, treating an arc
  as costing `-log p(u -> v)` so the cheapest tree is the most likely one. That is
  `delayed_bfs`, it runs in near-linear time, and it has no learning in it.
- A PLAIN RANDOM WALKER IS COMPETITIVE and sometimes wins. On an assortative graph
  the infected region is densely interconnected and personalized PageRank from the
  reports beats principled tree sampling. Check what the graph profile says about
  assortativity before assuming a clever method is better.
- Do not predict the whole graph. Recall is cheap and precision is not; a decoder
  that names every reachable node scores near zero on the tree half because almost
  none of its edges are real.
"""


prediction_brief = """\
You are designing a Cascade Popularity Prediction algorithm as an executable Python
script.

YOUR GOAL: a REAL cascade is spreading on this graph. You are shown its first `t_o`
timesteps (who adopted and when) and you must predict how many nodes will have
adopted by the horizon. You are not intervening in anything and you are not
recovering the past; you are forecasting. **LOWER IS BETTER**: your score is a
prediction ERROR, unlike every other task in this system.

THE SCORE IS MSLE: MEAN SQUARED **LOG** ERROR:

    MSLE = mean( (log2(predicted) - log2(actual))^2 )

Read that twice, because it changes what a good prediction is:
- Being off by 10 on a cascade of 20 costs the same as being off by 1000 on a
  cascade of 2000. RELATIVE accuracy is everything; absolute accuracy is nothing.
- It is SYMMETRIC in log space, so over-predicting by 2x and under-predicting by 2x
  cost exactly the same. A systematic multiplicative bias is therefore the cheapest
  possible thing to fix and the first thing to check.
- Predicting a CONSTANT is much stronger than it sounds. The geometric mean of the
  training sizes is the best instance-blind prediction, and it is a real bar.

WHAT MAKES THIS HARD, AND WHAT ACTUALLY WORKS:
- THE DYNAMICS ARE NOT INDEPENDENT CASCADE. These are real logged adoptions, not a
  simulation. Adoption is not memoryless (a node's second exposure matters), a node
  is exposed repeatedly rather than once per neighbour, and some adopters arrive
  from outside the graph entirely: search, front pages, off-platform sharing. Any
  model that assumes one-shot independent transmission will be wrong in a
  systematic direction.
- TEMPORAL FEATURES DOMINATE. The single most predictive quantity in this
  literature is the adoption RATE IN THE SECOND HALF of the observation window,
  it beats every structural feature by a wide margin. A cascade still accelerating
  at `t_o` is a different object from one that has flattened, and the observed
  wave series tells you which.
- THE CLASSICAL BAR IS ONE LINE. `log P(horizon) = alpha * log P(t_o) + beta`: a
  2008 result, one feature, and it is still printed as a baseline in every paper
  published since. Beat it or explain why you did not.
- BRANCHING RATIO IS THE MECHANISM. If each adopter produces R more and R < 1, the
  remaining total is a geometric sum. If R >= 1 the expected size DIVERGES and the
  honest answer is to decline (see below) or to fall back on a feature estimate.
- DO NOT PREDICT THE REACHABLE SET. Most cascades die far short of what the graph
  would allow. An estimate near "everything within k hops" is the classic
  saturation failure and it scores terribly in log space.

DECLINING IS ALLOWED AND IS NOT AN ERROR: return None when your model genuinely has
no estimate (a divergent generative fit, an empty observation). Declines are COUNTED
in a separate column, never scored as a wrong answer. But a predictor that declines
EVERYTHING is rejected: it has not answered the question.
"""


def _prediction_rules(task: TaskSpec) -> str:
    """The forecasting preamble: no actions, one method, and the forward model."""
    if task.forward_model:
        oracle_block = """\

THE FORWARD MODEL: `self.forecast_marginals(adopters, frontier, steps)`:
    Returns a numpy array of length num_nodes: P(node has adopted `steps`
    timesteps after the end of the observation window), given that `adopters` is
    everyone who has adopted so far and `frontier` is the wave that adopted most
    recently. `self.expected_popularity(adopters, frontier, steps)` is the same
    call summed into a single number, which is usually all you want.

    Pass `observation.frontier` as the frontier and not the whole adopter set:
    someone who adopted five steps ago has already had their chance to spread, and
    seeding them again would over-predict badly.

    This is the ONLY thing that differs between the experimental conditions here.
    The action space of this task is empty, so what is being measured is whether a
    forward model in the prediction loop is worth anything at all against a
    feature-driven estimate. The published evidence is genuinely mixed.

    It is not free. One call unrolls the kernel `steps` times across several
    sampled realizations, and the calls are counted. Get an estimate from features
    first, THEN spend calls refining it. A loop that calls it per node will not
    finish.

    IT IS ALSO NOT GROUND TRUTH. It is an Independent-Cascade-shaped model of a
    process that is not Independent Cascade. Treat its output as one input among
    several: blending it with a feature estimate, or using it only to rank rather
    than to size, is a legitimate and often better use of it than trusting it."""
    else:
        oracle_block = """\

NO FORWARD MODEL IN THIS CONDITION. `self.forecast_marginals` raises if you call
it. This arm exists to measure what pure features and point-process fits achieve,
and the literature suggests that may be a great deal: feature-driven regression is
reported to beat deep models on some corpora. Work from the observed adoption
history, the wave series and the graph structure alone."""

    return f"""\
{prediction_brief}
OUTPUT FORMAT: reply with exactly ONE fenced ```python block and nothing else,
no prose before or after. The block contains import lines (if you need any) and
then exactly ONE class subclassing `Strategy`. Nothing else at module level: no
example usage, no test code.

IMPORTS: you MAY import any of {", ".join(allowed_imports)}. Use numpy for
anything you would otherwise write as a Python loop over all nodes.
Importing anything else is rejected.

AVAILABLE NAMES (already in your script's namespace: do NOT import these):
- `GraphInfo` : .num_nodes, .out_neighbors(node), .in_neighbors(node),
  .degree(node), .edge_index (2, E), .ic_probs (E,).
- `prediction_algorithms` : the published predictors (API below).
- `cascade_features(graph, observation)` : the standard feature dict (API below).
- `primitives` : structural helpers (API below).
- `ActionOp` and `State` exist but you will not need them: this task emits no
  actions at all.

THE OBSERVATION OBJECT you are handed:
- `observation.adopters` : `{{node: timestep it adopted}}` for everyone who adopted
  INSIDE the observation window. This is the whole observed history, not a count.
- `observation.popularity` : `len(observation.adopters)`, i.e. P(t_o). Your
  prediction is bounded below by this: adoption is progressive and nobody
  un-adopts.
- `observation.frontier` : the node ids that adopted in the LAST observed step.
- `observation.wave(t)`  : the node ids that adopted at step `t`.
- `observation.root`     : who started it.
- `observation.observed_steps` : how many steps you were shown.
- `observation.horizon`  : the step your prediction is for. `horizon -
  observed_steps` is how much future there is.
- `observation.num_nodes` : |V|; your prediction cannot exceed it.

- `self.fit_examples` : a list of LABELLED cascades from the selection split, each
  with `.observation` and `.actual` (its true final popularity). Use them to fit a
  constant, a regression, or a calibration: that is exactly what the classical
  baselines do and it is not cheating. These are never the cascades you are scored
  on.

WHAT YOU IMPLEMENT:
    def predict(self, graph, observation, horizon) -> float | None
        The popularity the cascade will have reached by `horizon`, as a TOTAL
        count of distinct adopters (not an increment). Return None to DECLINE.

        RULES, all enforced:
        - the value must be >= `observation.popularity`. A cascade cannot shrink.
        - the value must be <= `observation.num_nodes`. It counts distinct nodes.
        - None or a non-finite value means DECLINE, which is counted separately
          and never scored as an error.
        - declining EVERY cascade is rejected.
{oracle_block}
"""


def _reconstruction_rules(task: TaskSpec) -> str:
    """The decoding preamble: no actions, one method, and the transition kernel."""
    if task.forward_model:
        oracle_block = """\

THE TRANSITION KERNEL: `self.step_marginals(infected, frontier)`:
    Returns a numpy array of length num_nodes: P(node activates on the NEXT step)
    given that `infected` is the set active now and `frontier` is the wave that
    just activated. This is the one-step dynamics the cascade you are inverting
    actually ran under, and it is the thing a purely structural decoder does not
    have.

    Use it to SCORE a hypothesis. Given a proposed history you can unroll it:

        p = self.step_marginals(infected_so_far, wave_at_t)
        logp = self.transition_logprob(p, infected_so_far, wave_at_t_plus_1)

    `transition_logprob(marginal, infected, next_frontier)` is already bound for
    you and derives from the same call, so summing it over the steps of a proposed
    trajectory gives that trajectory's log-likelihood: a score you can compute
    WITHOUT any labels, and the natural objective for a local search.

    It is not free. Every call is one evaluation of the kernel and the calls are
    counted. Get a decode first with a cheap structural method, THEN spend calls
    improving it: a search that calls the kernel inside a loop over all nodes
    will not finish. You are also free to never call it at all; if structure alone
    wins, that is a result."""
    else:
        oracle_block = """\

NO TRANSITION KERNEL IN THIS CONDITION. `self.step_marginals` raises if you call
it. This arm exists to measure what pure structure and timing achieve, so your
decoder must work from the graph, the reports and their times alone: Steiner
trees, BFS/shortest-path orderings, centralities on the observed subgraph."""

    return f"""\
{reconstruction_brief}
OUTPUT FORMAT: reply with exactly ONE fenced ```python block and nothing else,
no prose before or after. The block contains import lines (if you need any) and
then exactly ONE class subclassing `Strategy`. Nothing else at module level: no
example usage, no test code.

IMPORTS: you MAY import any of {", ".join(allowed_imports)}. Use numpy for
anything you would otherwise write as a Python loop over all nodes.
Importing anything else is rejected.

AVAILABLE NAMES (already in your script's namespace: do NOT import these):
- `GraphInfo` : .num_nodes, .out_neighbors(node), .in_neighbors(node),
  .degree(node), .edge_index (2, E), .ic_probs (E,).
- `reconstruction_algorithms` : the published decoders (API below).
- `primitives` : structural helpers (API below).
- `ActionOp` and `State` exist but you will not need them: this task emits no
  actions.

THE OBSERVATION OBJECT you are handed:
- `observation.reported` : `{{node: activation timestep or None}}`. A node in here
  WAS infected: that is ground truth. `None` means "infected, time unknown".
- `observation.times`    : just the entries whose time is known, as `{{node: t}}`.
- `observation.infected` : the observed infected node ids, sorted.
- `observation.final_state` : an (N,) 0/1 array, or None. Under the
  final-snapshot setting this is ALL you get and `reported` is empty.
- `observation.visible`  : an (N,) bool mask, or None. Under the hidden-node
  setting a False entry is a node that is NOT IN THE GRAPH: naming one is
  rejected.

WHAT YOU IMPLEMENT:
    def reconstruct(self, graph, observation, horizon) -> dict[int, tuple[int, int | None]]
        `{{node: (activation timestep, the node that infected it)}}` for every node
        you believe was infected. Nodes you believe were never infected are simply
        ABSENT from the dict.

        RULES, all enforced:
        - `parent = None` means SOURCE, and a source activates at timestep 0.
          A node at t > 0 must name a parent; a node at t = 0 must not.
        - a parent must be an actual in-neighbour: `parent in
          graph.in_neighbors(node)`. A transmission travels along an edge.
        - every timestep is in `[0, horizon]`, every node id in `[0, num_nodes)`.
        - returning an empty dict is rejected. At minimum the nodes the observation
          REPORTS were infected.
{oracle_block}
"""


def _localization_rules(task: TaskSpec) -> str:
    """The inverse-task preamble: no actions, one method, and the forward oracle."""
    if task.forward_model:
        oracle_block = """\

THE FORWARD ORACLE: `self.predict_marginals(seeds)`:
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
    You are also free to never call it at all: if structure alone wins, that is a
    result."""
    else:
        oracle_block = """\

NO FORWARD ORACLE IN THIS CONDITION. `self.predict_marginals` raises if you call
it. This arm exists to measure what pure structure achieves, so your algorithm
must be a structural inference rule over the graph and the observation alone."""

    return f"""\
{localization_brief}
OUTPUT FORMAT: reply with exactly ONE fenced ```python block and nothing else,
no prose before or after. The block contains import lines (if you need any) and
then exactly ONE class subclassing `Strategy`. Nothing else at module level: no
example usage, no test code.

IMPORTS: you MAY import any of {", ".join(allowed_imports)}. Use numpy for
anything you would otherwise write as a Python loop over all nodes.
Importing anything else is rejected.

AVAILABLE NAMES (already in your script's namespace: do NOT import these):
- `GraphInfo` : .num_nodes, .out_neighbors(node), .in_neighbors(node),
  .degree(node), .edge_index (2, E), .ic_probs (E,).
- `localization_algorithms` and `localization_scorers` : the published baselines
  and their per-node score vectors (API below).
- `primitives` : structural helpers (API below).
- `ActionOp` and `State` exist but you will not need them: this task emits no
  actions.

WHAT YOU IMPLEMENT:
    def localize(self, graph, observation, budget) -> list[int]
        The node ids you believe started the cascade. AT MOST `budget` of them,
        no duplicates, every id in [0, num_nodes). Returning more than `budget` is
        REJECTED: extra names would buy recall for free.

    def source_scores(self, graph, observation) -> np.ndarray     (OPTIONAL)
        One float per node, higher meaning more likely to be a source. Implement
        it and your AUC is measured on that real ranking; omit it and AUC falls
        back to the ORDER of the list localize() returned, which ties every node
        you did not name. It costs a few lines and it is a reported column.
{oracle_block}
"""


blocking_budget_rules = {
    "counter_seed": """\
- A blocker is ActionOp("add_node", node), which seeds YOUR cascade: never the
  rumour's. Emit at most `budget` add_node actions in total.
- Seeding the same node twice is REJECTED: it spends two units of budget on one node.
- Seeding a node the rumour already owns does nothing; it is already committed.""",
    "node_block": """\
- A blocker is ActionOp("remove_node", node). Emit at most `budget` remove_node
  actions in total. You do NOT need to emit the incident remove_edge ops: the
  harness expands each removal into a full node deletion for you.
- Removing the same node twice is REJECTED: it spends two units of budget on one node.
- Emitting add_node is REJECTED under this lever. You are not seeding anything.""",
    "edge_block": """\
- A blocker is ActionOp("remove_edge", u, v), one directed arc. Emit at most
  `budget` remove_edge actions in total.
- Cutting the same arc twice is REJECTED. Cutting two different arcs out of the same
  node is FINE: the budget is counted per arc, not per node.
- Emitting add_node or remove_node is REJECTED under this lever.""",
    "weight_block": """\
- A blocker is ActionOp("set_edge_weight", u, v, w), which sets arc u -> v to
  transmission probability w. Emit at most `budget` of them.
- w must be between 0.0 and the arc's CURRENT probability. Raising an arc is
  REJECTED: you are blocking, not boosting.
- Reweighting the same arc twice is REJECTED. Two different arcs out of one node is
  FINE: the budget is per arc.
- Emitting add_node or remove_node is REJECTED under this lever.""",
}



# EPIDEMIC CONTROL, per lever. Separate from the containment briefs because the
# DYNAMICS differ, not only the intervention: nodes RECOVER here, which means the
# outbreak burns out on its own and a dose is only worth what it saves BEFORE that
# happens (research/epidemic_control.md §2.4, §8.3).
epidemic_briefs = {
    "vaccinate": """\
You are designing a network VACCINATION algorithm as an executable Python script.

YOUR GOAL: MINIMIZE the attack rate, the number of nodes EVER infected by the end.
LOWER IS BETTER, and every score you are shown reads that way.

You do NOT start the outbreak. Index cases are already infectious at fixed nodes
you did not choose and cannot change: they are listed in the task block below.
Your budget buys DOSES: each `remove_node` immunizes that node, taking it out of
the graph along with all its edges, so it can never be infected, never transmit,
and never counts toward the attack rate.

Warning: DOSING AN INDEX CASE IS REJECTED. Immunizing a node that is already
infectious ends the outbreak rather than controlling it, which is a different
problem. Filter the sources out of your candidate set before you rank anything.

THE DYNAMICS ARE NOT MONOTONE, and this is what separates the task from node
removal on a one-way cascade:
- An infectious node RECOVERS at a fixed rate and then stops transmitting forever
  (under SIS it becomes susceptible again instead). The outbreak therefore burns
  out on its own, and a dose is worth only what it saves before that happens.
- That means TIMING is built into the structure: a node three hops from the
  sources may never be reached at all, so protecting it buys nothing however
  central it is.
- The peak of the epidemic matters as much as its total. A policy that flattens
  the curve without shrinking the total is a real result in this literature.

WHAT ACTUALLY WORKS HERE, AND WHAT DOES NOT:
- Two published families disagree, and the disagreement is the interesting part.
  The SPECTRAL family (NetShield) minimizes the adjacency's leading eigenvalue,
  which is a model-independent bound on whether an epidemic can take off at all,
  but it does not know where this outbreak IS.
- The DATA-AWARE family (DAVA) conditions on exactly that: it cuts the nodes that
  DOMINATE the paths out of the observed sources. On a small localized outbreak it
  beats the spectral family by a wide margin, and on a large diffuse one it does not.
- Plain top-degree is a 2002 heuristic and it is the row you actually have to
  beat. `acquaintance_immunization` uses no global information at all and is the
  published embarrassment for methods that read the whole graph.
""",
    "quarantine": """\
You are designing a network QUARANTINE algorithm as an executable Python script.

YOUR GOAL: MINIMIZE the attack rate, the number of nodes EVER infected by the end.
LOWER IS BETTER, and every score you are shown reads that way.

You do NOT start the outbreak. Index cases are already infectious at fixed nodes
you did not choose: they are listed in the task block below. Your budget buys
ISOLATION: each `remove_node` cuts every contact of that node while LEAVING THE
NODE IN THE GRAPH. It is not immune. If the outbreak already reached it, it stays
counted in the attack rate; it simply stops passing the infection on.

That difference from vaccination is the whole point of this lever: isolating a
node you were too late to protect saves its neighbours and not the node, while a
vaccine spent on an already-infected node saves nobody at all. Rank candidates by
what they carry ONWARD, not by whether they themselves survive.

The same non-monotone dynamics apply: infectious nodes recover and stop
transmitting, so the outbreak burns out on its own and an isolation is worth only
what it prevents before that happens.
""",
    "edge_cut": """\
You are designing a CONTACT-SEVERING algorithm as an executable Python script.

YOUR GOAL: MINIMIZE the attack rate, the number of nodes EVER infected by the end.
LOWER IS BETTER.

You do NOT start the outbreak. Index cases are already infectious at fixed nodes
listed in the task block. Your budget buys ARC CUTS: each `remove_edge(u, v)`
deletes one directed contact, so the infection can no longer travel that way.
Nobody is immunized: every node stays in the graph and stays infectable through
whatever routes remain.

This is a strictly weaker instrument than vaccination at the same k (one dose
removes deg(v) arcs at once), so a k-arc budget only pays off when the graph has
BRIDGES: a few arcs whose loss disconnects the outbreak from a large region. On a
dense graph it will not, and reporting that honestly is a result.

The published spectral rule scores an arc by u(i) * u(j), the product of the
leading eigenvector's endpoint entries: that is NetMelt, and it does not know
where the outbreak is. Cutting the boundary of the observed infected set does.
""",
}
epidemic_briefs["contact_reduce"] = """\
You are designing a CONTACT-REDUCTION algorithm as an executable Python script.

YOUR GOAL: MINIMIZE the attack rate, the number of nodes EVER infected by the end.
LOWER IS BETTER.

You do NOT start the outbreak. Index cases are already infectious at fixed nodes
listed in the task block. Your budget buys REWEIGHTS: each
`set_edge_weight(u, v, w)` lowers the per-contact transmission probability on that
arc to `w`. This is the graded version of a cut: social distancing rather than
a travel ban, and it is the only lever in this literature that is continuous.

You may only LOWER an arc. Raising one is rejected.

The same reasoning as the cut lever applies: at equal k this buys less than a dose
does, so it pays off exactly where a few arcs carry the outbreak between regions.
"""

epidemic_budget_rules = {
    "vaccinate": """\
- A dose is ActionOp("remove_node", node). Emit at most `budget` remove_node
  actions in total. You do NOT need to emit the incident remove_edge ops: the
  harness expands each dose into a full node deletion for you.
- Dosing the same node twice is REJECTED: it spends two units of budget on one node.
- Dosing an INDEX CASE is REJECTED. Filter `self.outbreak` out of your candidates.
- Emitting add_node is REJECTED. You are not seeding this outbreak.""",
    "quarantine": """\
- An isolation is ActionOp("remove_node", node). Emit at most `budget` of them.
  The harness cuts that node's incident arcs and LEAVES THE NODE in the graph, so
  it stays susceptible and stays counted.
- Isolating the same node twice is REJECTED.
- Isolating an INDEX CASE is REJECTED. Filter `self.outbreak` out of your candidates.
- Emitting add_node is REJECTED.""",
    "edge_cut": """\
- A cut is ActionOp("remove_edge", u, v), one directed arc. Emit at most `budget`
  remove_edge actions in total.
- Cutting the same arc twice is REJECTED. Cutting two different arcs out of the
  same node is FINE: the budget is counted per arc, not per node.
- Emitting add_node or remove_node is REJECTED under this lever.""",
    "contact_reduce": """\
- A reduction is ActionOp("set_edge_weight", u, v, w), which sets arc u -> v to
  transmission probability w. Emit at most `budget` of them.
- w must be between 0.0 and the arc's CURRENT probability. Raising an arc is
  REJECTED: you are reducing contact, not increasing it.
- Reweighting the same arc twice is REJECTED. Two different arcs out of one node
  is FINE: the budget is per arc.
- Emitting add_node or remove_node is REJECTED under this lever.""",
}


def _epidemic_rules(task: TaskSpec) -> tuple[str, str, str]:
    """(brief, budget rules, library line) for one epidemic lever."""
    library_line = (
        "- `immunization_algorithms`, `dismantling_algorithms`, `algorithms` and\n"
        "  `primitives` modules (API below). `immunization_algorithms` members are\n"
        "  the published baselines for THIS task and take `outbreak=`: they are\n"
        "  what you are being compared against."
    )

    return (
        epidemic_briefs[task.epi_lever],
        epidemic_budget_rules[task.epi_lever],
        library_line,
    )


epidemic_timing_note = """\

USING THE HORIZON: put every dose in element 0 and leave the rest of the plan
empty. A node protected at t>0 may already be infected, and protecting it then
saves nothing it has not already passed on. Pre-emptive allocation is strictly
stronger, so all of your effort belongs in WHICH nodes you protect, not when.

Two consequences of the recovery rate that are easy to miss. The outbreak has a
finite lifetime, so a region it will not reach before burning out is a region you
should not spend on. And its PEAK matters: the report grades peak prevalence and
time-to-peak beside the total, so a policy that delays the wave has bought
something real even when the final count barely moves.
"""


epidemic_exemplars = """\
EXAMPLES: two strategies at the level you should START from, not finish at.

Example 1, dominator-style allocation (the DAVA idea, cut the nodes that all paths
out of the outbreak must pass through, weighted by how much sits behind them):
```python
class OutbreakDominators(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        sources = set(self.outbreak)

        # How much of the graph each node still leads to, discounted by distance
        # from the outbreak: a cheap stand-in for "expected nodes saved by cutting"
        reach = {node: 0.0 for node in range(graph.num_nodes)}
        wave, seen, depth = set(sources), set(sources), 0.0
        while wave and depth < horizon:
            nxt = set()
            for node in wave:
                for other in graph.out_neighbors(node):
                    if other not in seen:
                        nxt.add(other)
            for node in nxt:
                # Onward reach, damped so a node the outbreak barely gets to is cheap
                reach[node] = len(set(graph.out_neighbors(node)) - seen) * (0.6 ** depth)
            seen |= nxt
            wave, depth = nxt, depth + 1.0

        chosen, cut = [], set()
        for _ in range(budget):
            best, best_score = -1, float("-inf")
            for node in range(graph.num_nodes):
                if node in sources or node in cut:
                    continue
                # Discount a node whose neighbours are already protected: two cuts
                # on the same route buy one route
                overlap = sum(1 for other in graph.out_neighbors(node) if other in cut)
                score = reach[node] / (1.0 + overlap)
                if score > best_score:
                    best, best_score = node, score
            if best < 0:
                break
            chosen.append(best)
            cut.add(best)

        return [[ActionOp("remove_node", int(node)) for node in chosen]] + [
            [] for _ in range(horizon)
        ]
```

Example 2, NetShield restricted to the outbreak's reachable set (the spectral rule
is the published bar; narrowing it to nodes the epidemic can actually reach is the
cheapest way to beat it, because a dose outside that set scores exactly zero):
```python
class ReachableShield(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        sources = set(self.outbreak)

        reach, wave = set(sources), set(sources)
        for _ in range(horizon):
            nxt = {o for n in wave for o in graph.out_neighbors(n)} - reach
            if not nxt:
                break
            reach |= nxt
            wave = nxt

        shield = immunization_algorithms.netshield(
            graph, min(graph.num_nodes, budget * 6), "SIR", outbreak=tuple(sources)
        )
        picks = [n for n in shield if n in reach and n not in sources][:budget]

        # Top up from the reachable set by degree if the shield ran short there
        if len(picks) < budget:
            rest = sorted(reach - sources - set(picks), key=graph.degree, reverse=True)
            picks += rest[: budget - len(picks)]

        return [[ActionOp("remove_node", int(node)) for node in picks]] + [
            [] for _ in range(horizon)
        ]
```
"""


def _blocking_rules(task: TaskSpec) -> tuple[str, str, str]:
    """(brief, budget rules, library line) for one blocking lever."""
    from coding_agent.blocking import lever_of

    lever = lever_of(task)
    library_line = (
        "- `blocking_algorithms`, `algorithms`, `dismantling_algorithms` and\n"
        "  `primitives` modules (API below). `blocking_algorithms` members are the\n"
        "  published baselines for THIS task and take `negative_seeds=`: they are\n"
        "  what you are being compared against."
    )

    return blocking_briefs[lever], blocking_budget_rules[lever], library_line


def _common_rules(task: TaskSpec | None) -> str:
    """The preamble, with the problem family and the budgeted op filled in."""
    if task is not None and task.forecasts:
        return _prediction_rules(task)

    if task is not None and task.decodes:
        return _reconstruction_rules(task)

    if task is not None and task.recovers:
        return _localization_rules(task)

    blocks = task is not None and task.blocks
    immunizes = task is not None and task.immunizes
    contains = task is not None and task.contains

    if blocks:
        brief, budget_rules, library_line = _blocking_rules(task)
    elif immunizes:
        brief, budget_rules, library_line = _epidemic_rules(task)
    elif contains:
        brief = containment_brief
        budget_rules = """\
- A blocker is ActionOp("remove_node", node). Emit at most `budget` remove_node
  actions in total. You do NOT need to emit the incident remove_edge ops: the
  harness expands each removal into a full node deletion for you.
- Removing the same node twice is REJECTED: it spends two units of budget on one node.
- Emitting add_node is REJECTED. You are not seeding this cascade."""
        library_line = (
            "- `algorithms`, `adaptive_algorithms`, `dismantling_algorithms` and\n"
            "  `primitives` modules (API below). `dismantling_algorithms` members\n"
            "  return node-REMOVAL sets and are the published baselines for this task."
        )
    else:
        brief = seeding_brief
        budget_rules = """\
- A seed is ActionOp("add_node", node). Emit at most `budget` add_node actions in total.
- Seeding the same node twice is REJECTED: it spends two units of budget on one node."""
        library_line = (
            "- `algorithms`, `adaptive_algorithms` and `primitives` modules (API below).\n"
            "  `adaptive_algorithms` members are per-ROUND policies and are only callable\n"
            "  from act() on an adaptive task; the reference below lists them when so."
        )

    return f"""\
{brief}
OUTPUT FORMAT: reply with exactly ONE fenced ```python block and nothing else,
no prose before or after. The block contains import lines (if you need any) and
then exactly ONE class subclassing `Strategy`. Nothing else at module level: no
example usage, no test code.

IMPORTS: you MAY import any of {", ".join(allowed_imports)}. Use numpy for
anything you would otherwise write as a Python loop over all nodes: vectorized
scoring is what lets you afford large sample sizes inside the time limit.
Importing anything else is rejected.

AVAILABLE NAMES (already in your script's namespace: do NOT import these):
- `ActionOp(op, target, destination=None, weight=None)` : a graph action. Ops:
    add_node, remove_node, add_edge, remove_edge, set_edge_weight.
- `State` : has .infected / .frontier (the RUMOUR under a two-cascade task) and
    .pos_infected / .pos_frontier (your own counter-cascade, empty otherwise).
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
EXAMPLES: two strategies at the level you should START from, not finish at.

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
EXAMPLES: two strategies at the level you should START from, not finish at.

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
published HDA baseline, narrowed: nodes the cascade cannot reach are free to
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

# The influence-blocking counterpart, one per lever family. The IM exemplars emit
# add_node without a rumour to answer and the containment ones assume deletion is the
# only move, so both would teach the wrong shape. `self.outbreak` is S_N here.
blocking_exemplars = {
    "counter_seed": """\
EXAMPLES: two strategies at the level you should START from, not finish at.

Example 1, race-to-the-node scoring (rank a candidate by how much of the graph it
reaches BEFORE the rumour does, which is the only thing that scores):
```python
import heapq

class RaceAhead(Strategy):
    def _arrival(self, graph, sources):
        # Best-probability path from `sources`, as (probability, hops) per node.
        # Hops is what decides who wins a node; probability is how much it is worth.
        best = [0.0] * graph.num_nodes
        hops = [float("inf")] * graph.num_nodes
        weight = {}
        for e in range(graph.edge_index.shape[1]):
            weight[(int(graph.edge_index[0, e]), int(graph.edge_index[1, e]))] = float(
                graph.ic_probs[e]
            )

        heap = []
        for node in sources:
            best[node], hops[node] = 1.0, 0
            heapq.heappush(heap, (0.0, 0, int(node)))

        while heap:
            cost, hop, node = heapq.heappop(heap)
            for other in graph.out_neighbors(node):
                probability = best[node] * weight.get((node, other), 0.0)
                if probability > best[other] and probability > 1e-4:
                    best[other], hops[other] = probability, hop + 1
                    heapq.heappush(heap, (-probability, hop + 1, other))

        return best, hops

    def plan_horizon(self, graph, budget, horizon):
        rumour_reach, rumour_hops = self._arrival(graph, self.outbreak)
        # Only nodes the rumour actually threatens can be saved
        candidates = [v for v in range(graph.num_nodes)
                      if rumour_reach[v] > 0 and v not in self.outbreak]

        seeds = []
        for _ in range(budget):
            best_node, best_gain = -1, 0.0
            for node in candidates:
                if node in seeds:
                    continue
                reach, hops = self._arrival(graph, seeds + [node])
                gain = sum(
                    rumour_reach[v] * reach[v]
                    for v in candidates
                    if hops[v] <= rumour_hops[v]      # arriving LATE saves nobody
                )
                if gain > best_gain:
                    best_node, best_gain = node, gain
            if best_node < 0:
                break
            seeds.append(best_node)

        return [[ActionOp("add_node", int(v)) for v in seeds]] + [
            [] for _ in range(horizon)
        ]
```

Example 2, the published proximity baseline, narrowed (seed the rumour's own
out-neighbours, but spend the budget on the ones with the most onward reach):
```python
class ProximityReach(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        sources = set(self.outbreak)
        ring = sorted({v for s in sources for v in graph.out_neighbors(s)} - sources)
        # Onward reach, discounted for neighbours an earlier pick already covers
        chosen, covered = [], set()
        while len(chosen) < budget and ring:
            best_node, best_score = None, -1.0
            for node in ring:
                if node in chosen:
                    continue
                fresh = sum(1 for o in graph.out_neighbors(node) if o not in covered)
                if fresh > best_score:
                    best_node, best_score = node, fresh
            if best_node is None:
                break
            chosen.append(best_node)
            covered.update(graph.out_neighbors(best_node))

        return [[ActionOp("add_node", int(v)) for v in chosen]] + [
            [] for _ in range(horizon)
        ]
```
`self.outbreak` is set on your Strategy instance before plan_horizon is called; it
is the tuple of RUMOUR source ids, and it is also printed in the task block.
Beat both. Combining their ideas, or replacing them, are both fair game.
""",
    "node_block": """\
EXAMPLES: two strategies at the level you should START from, not finish at.

Example 1, cut-point scoring by sampled reachability (delete the nodes the rumour has
no way around):
```python
import numpy as np

class CutPoints(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        rng = np.random.default_rng(0)
        credit = np.zeros(graph.num_nodes)

        for _ in range(50):
            # One live-edge realization of the rumour's cascade
            live = {v: [] for v in range(graph.num_nodes)}
            keep = rng.random(graph.edge_index.shape[1]) < graph.ic_probs
            for e in np.flatnonzero(keep):
                live[int(graph.edge_index[0, e])].append(int(graph.edge_index[1, e]))

            seen, order, queue = set(self.outbreak), [], list(self.outbreak)
            while queue:
                node = queue.pop(0)
                for other in live[node]:
                    if other not in seen:
                        seen.add(other)
                        order.append(other)
                        queue.append(other)

            # A node that was reached through exactly one live in-arc is a
            # bottleneck on this realization
            arrivals = {}
            for node in seen:
                for other in live[node]:
                    arrivals[other] = arrivals.get(other, 0) + 1
            for node in order:
                if arrivals.get(node, 0) == 1:
                    credit[node] += 1.0

        for node in self.outbreak:
            credit[node] = -1.0

        picks = [int(v) for v in np.argsort(-credit)[:budget]]
        return [[ActionOp("remove_node", v) for v in picks]] + [
            [] for _ in range(horizon)
        ]
```

Example 2, the published trivial heuristic you have to beat (the rumour's
out-neighbours, by degree):
```python
class NeighbourDegree(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        sources = set(self.outbreak)
        ring = sorted({v for s in sources for v in graph.out_neighbors(s)} - sources)
        ring.sort(key=graph.degree, reverse=True)
        return [[ActionOp("remove_node", int(v)) for v in ring[:budget]]] + [
            [] for _ in range(horizon)
        ]
```
Beat both. Combining their ideas, or replacing them, are both fair game.
""",
}
blocking_exemplars["edge_block"] = """\
EXAMPLES: one strategy at the level you should START from, not finish at.

Cut the arcs that carry the most traffic out of the rumour (reach probability of the
tail, times how much only hangs off the head):
```python
import numpy as np

class CarriedTraffic(Strategy):
    def plan_horizon(self, graph, budget, horizon):
        # Damped reach from the rumour: "will it ever get to this tail at all"
        exposure = np.zeros(graph.num_nodes)
        wave = {int(v): 1.0 for v in self.outbreak}
        for _ in range(4):
            nxt = {}
            for node, mass in wave.items():
                exposure[node] += mass
                for other in graph.out_neighbors(node):
                    nxt[other] = nxt.get(other, 0.0) + mass * 0.5
            wave = nxt

        scored = []
        for e in range(graph.edge_index.shape[1]):
            u = int(graph.edge_index[0, e])
            v = int(graph.edge_index[1, e])
            # An arc into a node with other ways in is redundant
            alternatives = max(1, len(graph.in_neighbors(v)))
            scored.append(
                (exposure[u] * float(graph.ic_probs[e]) / alternatives, u, v)
            )

        scored.sort(reverse=True)
        return [[ActionOp("remove_edge", u, v) for _, u, v in scored[:budget]]] + [
            [] for _ in range(horizon)
        ]
```
Beat it.
"""
blocking_exemplars["weight_block"] = blocking_exemplars["edge_block"].replace(
    'ActionOp("remove_edge", u, v)', 'ActionOp("set_edge_weight", u, v, 0.0)'
).replace("class CarriedTraffic", "class CarriedTrafficDamping")

# The blocking timing note. `containment_timing_note` is close but not right: a
# counter-seed placed late has strictly fewer steps to RACE with, which is a stronger
# statement than "the node may already be infected".
blocking_timing_note = """\

USING THE HORIZON: put every action in element 0 and leave the rest of the plan
empty. The rumour was seeded at t=0 and is already moving; every step you wait is a
step of head start you hand it, and under either tie-break a cascade that arrives
second saves nobody. If the task block below states a DETECTION DELAY, the harness
drops anything you emit before that step: that is the experiment, not a bug, and it
is what makes the first-mover advantage measurable.
"""


# The inverse-task counterpart. Neither of the other two exemplar sets works here:
# both emit action bags from plan_horizon, and this task calls localize and
# rejects actions outright, so showing them would spend the first iteration on a
# repair turn for a contract the model was never asked to implement.
localization_exemplars = """\
EXAMPLES: two algorithms at the level you should START from, not finish at.

Example 1, LPSI with a community separation constraint (pure structure, no
forward model: this is roughly the classical bar):
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
Beat both. Combining their ideas, or replacing them, are both fair game, and if
the forward oracle turns out not to help, say so with a program that does not use it.
"""

# The decoding counterpart. Neither the intervention exemplars nor the
# localization ones work here: the first two emit action bags and the third
# returns a node list, while this contract returns a whole trajectory and is
# scored on the who-infected-whom edges. Showing the wrong set would cost the
# search its first iteration on a repair turn for a contract nobody asked for.
reconstruction_exemplars = """\
EXAMPLES: two decoders at the level you should START from, not finish at.

Example 1, an ordered Steiner tree by likelihood-weighted BFS (pure structure, no
kernel: this is roughly the classical bar):
```python
import heapq
import math

class LikelihoodTree(Strategy):
    def reconstruct(self, graph, observation, horizon):
        reports = list(observation.reported)
        if not reports:
            return {}

        known = observation.times
        probability = {}
        for edge in range(graph.edge_index.shape[1]):
            u = int(graph.edge_index[0, edge])
            v = int(graph.edge_index[1, edge])
            probability[(u, v)] = float(graph.ic_probs[edge])

        # Root at the earliest report; if no times were given, the highest-degree
        # one is a serviceable stand-in for the centre of the observed region
        root = min(reports, key=lambda n: (known.get(n, 10**9), -graph.degree(n)))

        # Dijkstra over -log p: the cheapest path is the MOST LIKELY chain of
        # transmissions, which is what makes this a likelihood tree and not a BFS
        times = {root: known.get(root, 0)}
        parent = {root: None}
        cost = {root: 0.0}
        queue = [(0.0, root)]

        while queue:
            spent, node = heapq.heappop(queue)
            if spent > cost.get(node, float("inf")):
                continue
            step = times[node] + 1
            if step > horizon:
                continue

            for other in graph.out_neighbors(node):
                seen_at = known.get(other)
                if seen_at is not None and step > seen_at:
                    continue          # cannot arrive after it was observed to fire
                arc = -math.log(max(probability.get((node, other), 1e-9), 1e-9))
                if spent + arc < cost.get(other, float("inf")):
                    cost[other] = spent + arc
                    times[other] = seen_at if seen_at is not None else step
                    parent[other] = node
                    heapq.heappush(queue, (cost[other], other))

        # Keep the reports plus whatever the tree had to pass through to reach them
        keep = set(reports) | {n for n in times if cost.get(n, 1e9) < 8.0}
        decoded = {}
        for node in keep:
            if node not in times or not observation.is_visible(node):
                continue
            step = max(0, min(int(times[node]), horizon))
            up = parent.get(node)
            if step == 0 or up is None or up not in keep:
                decoded[node] = (0, None)
            else:
                decoded[node] = (step, int(up))

        return decoded
```

Example 2, refine a structural decode with the kernel: start from the library's
`delayed_bfs`, then move activation times one step at a time and keep a move only
when the trajectory's log-likelihood under the transition kernel goes up:
```python
class KernelRefined(Strategy):
    def reconstruct(self, graph, observation, horizon):
        decoded = reconstruction_algorithms.delayed_bfs(graph, observation, horizon)
        times = {node: value[0] for node, value in decoded.items()}
        # An OBSERVED time is ground truth; only the inferred ones may move
        movable = [n for n in times if n not in observation.times]

        def score(assignment):
            waves = {}
            for node, step in assignment.items():
                waves.setdefault(step, []).append(node)
            total = 0.0
            infected = list(waves.get(0, []))
            for step in range(1, horizon + 1):
                wave = waves.get(step - 1, [])
                if not wave:
                    break
                marginal = self.step_marginals(infected, wave)
                total += self.transition_logprob(marginal, infected, waves.get(step, []))
                infected = infected + waves.get(step, [])
            return total

        best = score(times)
        for node in movable[:40]:            # bounded: every call costs a kernel evaluation
            for shift in (-1, 1):
                proposed = max(1, min(times[node] + shift, horizon))
                if proposed == times[node]:
                    continue
                original = times[node]
                times[node] = proposed
                candidate = score(times)
                if candidate > best:
                    best = candidate
                else:
                    times[node] = original

        # Re-attach parents against the updated times: a node's parent must now be
        # an in-neighbour that activated strictly earlier
        decoded = {}
        for node, step in times.items():
            if step == 0:
                decoded[node] = (0, None)
                continue
            earlier = [u for u in graph.in_neighbors(node) if times.get(u, 10**9) < step]
            if not earlier:
                decoded[node] = (0, None)
            else:
                decoded[node] = (step, int(max(earlier, key=lambda u: times[u])))

        return decoded
```
Beat both. Note what example 2 does NOT do: it never calls the kernel inside a
loop over all nodes, because that does not finish. Narrow to a short candidate
list first, then spend calls on it.
"""

# Only shown when the task actually allows edge ops. Under add_node-only IC the
# cascade is progressive and monotone, so delaying a seed is weakly worse and
# every schedule is dominated by "all seeds at t=0": telling the model to
# schedule across time there just burns iterations on a flat direction.
prediction_exemplars = """\

TWO WORKED PREDICTORS (different shapes: write your own, do not return these):

```python
import math

class TemporalGrowth(Strategy):
    \"\"\"Fit Szabo-Huberman on the labelled examples, then correct it with the
    single feature this literature says dominates: the adoption rate in the SECOND
    HALF of the observation window. A cascade still accelerating at t_o behaves
    differently from one that has flattened, and one number captures that.\"\"\"

    def _fitted(self):
        # Cached across calls: the fit is over the selection split and does not
        # depend on which cascade is being predicted
        if getattr(self, \"_coefficients\", None) is None:
            examples = list(getattr(self, \"fit_examples\", []))
            if len(examples) < 4:
                self._coefficients = (1.0, math.log(2.0))
            else:
                observed = [
                    math.log(max(e.observation.popularity, 1.0)) for e in examples
                ]
                actual = [math.log(max(e.actual, 1.0)) for e in examples]
                slope, intercept = numpy.polyfit(observed, actual, 1)
                self._coefficients = (float(slope), float(intercept))

        return self._coefficients

    def predict(self, graph, observation, horizon):
        if observation.popularity <= 0:
            return None

        slope, intercept = self._fitted()
        base = math.exp(
            slope * math.log(observation.popularity) + intercept
        )

        features = cascade_features(graph, observation)
        # Positive acceleration means the cascade is still growing; negative means
        # it has already turned over. Bounded so one odd cascade cannot blow up.
        rate_ratio = features[\"rate_second_half\"] / max(features[\"rate\"], 1e-6)
        adjustment = min(max(rate_ratio, 0.4), 2.5)

        estimate = base * adjustment
        return min(
            max(estimate, float(observation.popularity)), float(observation.num_nodes)
        )
```

```python
import numpy

class BlendedForecast(Strategy):
    \"\"\"Blend a branching-process extrapolation with the learned forward model, in
    LOG space because that is where the error is measured. The forward model is an
    IC-shaped view of a process that is not IC, so it is used as one opinion rather
    than as the answer, and the blend weight is fitted on the labelled examples
    rather than guessed.\"\"\"

    def _waves(self, observation):
        counts = numpy.zeros(observation.observed_steps + 1)
        for _, step in observation.adopters.items():
            counts[min(step, len(counts) - 1)] += 1
        return counts

    def _branching(self, observation, horizon):
        counts = self._waves(observation)
        ratios = [
            counts[i] / counts[i - 1] for i in range(1, len(counts)) if counts[i - 1] > 0
        ]
        if not ratios:
            return float(observation.popularity)

        reproduction = float(numpy.mean(ratios[-3:]))
        if reproduction >= 1.0:
            # Supercritical: the geometric sum diverges, so this branch has no
            # estimate. The blend below falls back on the model alone.
            return None

        remaining = max(horizon - observation.observed_steps, 0)
        tail = counts[-1] * reproduction * (1 - reproduction ** remaining)
        return float(observation.popularity) + tail / (1 - reproduction)

    def predict(self, graph, observation, horizon):
        steps = max(horizon - observation.observed_steps, 0)
        if steps <= 0:
            return float(observation.popularity)

        classical = self._branching(observation, horizon)

        # ONE call, on the frontier rather than the whole adopter set: an adopter
        # from five steps ago has already had its chance to transmit
        model = self.expected_popularity(
            list(observation.adopters), list(observation.frontier), steps
        )

        if classical is None:
            estimate = model
        else:
            # Geometric mean = arithmetic mean in log space, which is the space the
            # score is measured in
            estimate = math.sqrt(max(classical, 1.0) * max(model, 1.0))

        return min(
            max(estimate, float(observation.popularity)), float(observation.num_nodes)
        )
```
"""

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
cascade you are inverting ran for: a longer cascade means a more saturated
observation and therefore LESS information about where it started, which is worth
knowing when you decide how much to trust the observed state.
"""

# The decoding counterpart, and the opposite of every other one: `horizon` here is
# not a budget of steps to plan across, it is the LENGTH OF THE OBJECT you are
# recovering, so it appears in the answer rather than constraining it.
reconstruction_timing_note = """\

THE HORIZON IS PART OF YOUR ANSWER. You schedule nothing and emit nothing, but
every node you name carries a timestep in `[0, horizon]`, and those timesteps are
what the likelihood is computed over: a node placed at t is scored by the kernel
given the nodes you placed at t-1, so a wrong time is a wrong transmission. A longer
horizon means a more saturated cascade and therefore a harder inversion, because
a cascade that ran to convergence retains almost no trace of the order it went in.
"""

# The forecasting counterpart, and the third variant of "there is no horizon to
# plan across": here `horizon` is the QUESTION rather than a budget or an answer
# length. It is the timestep the prediction is for, and the gap between it and the
# observation window is the whole difficulty of the instance.
prediction_timing_note = """\

THE HORIZON IS THE QUESTION, NOT A BUDGET. You schedule nothing and emit nothing.
`horizon` is the timestep your prediction is FOR, and `horizon -
observation.observed_steps` is how much future you have to account for. A wider gap
is a strictly harder instance: the classical result in this literature is that
accuracy rises with how much you have SEEN and falls with how far ahead you must
look, so both numbers belong in whatever you fit.
"""

# The containment counterpart of seed_timing_note, and its mirror image: the
# reason to act at t=0 is not "more time to spread" but "the node is still
# uninfected". Blocking a node the cascade already passed through removes its
# onward reach and nothing else, so a late removal is strictly weaker.
containment_timing_note = """\

USING THE HORIZON: put every removal in element 0 and leave the rest of the plan
empty. Under a progressive cascade a node blocked at t>0 may already be infected,
and blocking it then only cuts its ONWARD reach: the infections it already caused
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
classical algorithm (e.g. `algorithms.degree_discount`) within each window.
Note: for this windowed method the budget applies PER window call (you are invoked once per window).

REPLY SHAPE (adapt the logic, keep the structure; budget/horizon are in the task):
```python
class MyStrategy(Strategy):
    def act(self, state, graph, timestep):
        seeds = algorithms.degree_discount(graph, 5, "IC")  # 5 = per-window budget
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
    "predict": """\

METHOD: POPULARITY FORECASTING.
Implement `predict(self, graph, observation, horizon) -> float | None`.

You are called ONCE PER LOGGED CASCADE, with that cascade's own observed prefix.
Your score is the mean squared LOG error across all of them, so a predictor that
nails the biggest cascade and collapses on the small ones LOSES: every cascade
weighs the same regardless of size, which is the whole reason the metric is in log
space. Write an ALGORITHM, not a fit to one cascade: hardcoded ids or per-cascade
constants score nothing on any other episode.

Two things about this task that no other one in this system has:

1. THE ERROR RUNS DOWNWARD. Lower is better. A change that makes the number go up
   is a regression, whatever it looked like in the code.
2. THE PROCESS IS REAL. These adoptions were logged, not simulated. No transition
   rule you can write is the true one, and the forward model (where you have one)
   is an approximation whose bias is systematic rather than random. Check the
   DIRECTION of your residual in the feedback before changing the model: a
   constant multiplicative bias is one line to fix and is usually most of the gap.

Fit whatever you need on `self.fit_examples`: the labelled selection cascades.
Fitting there is what every classical baseline in this literature does and it is
expected, not a loophole. Those cascades are never the ones you are scored on.

""",
    "reconstruct": """\

METHOD: TRAJECTORY DECODING.
Implement `reconstruct(self, graph, observation, horizon) -> dict`.

You are called ONCE PER MASKED CASCADE, with that cascade's own partial
observation. Your score is the mean across all of them, so a decoder that nails
one episode and collapses on the rest loses to one that is uniformly decent. Write
an ALGORITHM, not a fit to one cascade: hardcoded node ids will score zero on
every other episode.

The masking protocol is stated in the task block below and it changes the problem.
Read it: with activation times you are solving an ordering-constrained tree
problem, without them you are solving the ordering too, and from a final snapshot
alone you are solving both plus the source set.

""",
    "localize": """\

METHOD: SOURCE-SET INFERENCE.
Implement `localize(self, graph, observation, budget) -> list[int]`, and
optionally `source_scores(self, graph, observation) -> np.ndarray`.

You are called ONCE PER CASCADE, with that cascade's own observation and its own
source count as `budget`. Your score is the mean consistency across all of them,
so a rule that nails one episode and collapses on the rest loses to a rule that
is uniformly decent. Write an ALGORITHM, not a fit to one graph: hardcoded node
ids will explain nothing on every other episode.

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


def build_plan_oracle_block(task: TaskSpec) -> str:
    """
    `self.score_plan`, for the three tasks whose cascade is exogenous.

    Advertised only where it is bound: a seeding task's oracle is
    `predict_marginals`, and a task without a forward model gets the raiser text
    so the condition is stated rather than discovered through a traceback.
    """
    if not task.contains:
        return ""

    if not task.forward_model:
        return """\
NO FORWARD MODEL IN THIS CONDITION. `self.score_plan` raises if you call it:
this arm measures what structure and the outbreak's position achieve alone. Do
not re-implement simulation with numpy either; that is the condition you are in.
"""

    unit = {
        "remove_node": "removals",
        "add_node": "counter-seeds",
        "remove_edge": "edge cuts",
        "set_edge_weight": "reweights",
    }.get(task.budget_op, "actions")

    return f"""\
THE PLAN ORACLE: `self.score_plan(plan)` -> float.
    Takes a FULL candidate plan (the same list-of-bags shape plan_horizon
    returns), applies the fixed outbreak and every expansion the real evaluation
    applies, rolls it out on THIS arm's evaluator, and returns the objective
    (LOWER is better here). Use it to compare a few candidate plans of {unit}
    before committing one:

        candidate = [[ActionOp("{task.budget_op}", node) for node in picks]] + [
            [] for _ in range(horizon)
        ]
        reward = self.score_plan(candidate)

    It is not free: every call is a full metered rollout, so narrow to a SHORT
    list of candidate plans with cheap structural reasoning first, then spend
    calls ranking them. Do NOT hand-roll a Monte Carlo simulator with numpy
    instead: it burns your wall-clock budget re-deriving dynamics this call
    already has exactly, and an internal test against the wrong tie-break or
    lever optimizes the wrong problem. An illegal plan (over budget, wrong op,
    targeting a protected source) raises here with the same message the
    evaluation would give.
"""


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
    label = (
        "RUMOUR SEEDS S_N"
        if task.blocks
        else "INDEX CASES"
        if task.immunizes
        else "OUTBREAK SOURCES"
    )
    delay = (
        f"You are DETECTED LATE: anything you emit before t={task.detection_delay} "
        f"is held by the harness and applied at t={task.detection_delay}.\n"
        if task.blocks and task.detection_delay
        else ""
    )
    # §8.4 is emphatic that the tie-break is a reported hyperparameter rather than an
    # implementation detail, and it is the single fact that decides whether a
    # same-step arrival is worth anything, so it goes in the task block, not a
    # footnote
    rule = (
        {
            "negative": "the RUMOUR wins, you must arrive STRICTLY EARLIER to save "
            "a node (this is the competitive-LT convention)",
            "positive": "YOU win, arriving at the same step as the rumour is enough "
            "to save a node (this is Budak's convention)",
            "fixed": "a fixed per-node priority decides, and you cannot see it, "
            "treat a same-step arrival as worth roughly half a node",
        }.get(task.tie_break, task.tie_break)
        if task.blocks
        else ""
    )
    tie = f"TIE-BREAK: if you and the rumour reach a node on the SAME step, {rule}.\n" if rule else ""

    # §8.2 trap 2: beta and gamma are free parameters nobody standardizes, so a
    # model told the wrong ones plans against dynamics it will not get. Stated in
    # the task block rather than a footnote for the same reason the tie-break is.
    rates = (
        f"DYNAMICS: {task.diffusion_model}. Per-contact transmission is "
        f"{task.epi_beta:g} x the arc's own probability; an infectious node leaves "
        f"I with probability {task.epi_gamma:g} per step"
        + (
            f" and an exposed node becomes infectious with probability "
            f"{task.epi_alpha:g} per step"
            if task.diffusion_model == "SEIR"
            else ""
        )
        + (
            ", returning to SUSCEPTIBLE: it can be infected again, and there is no "
            "terminal state.\n"
            if task.diffusion_model == "SIS"
            else ", and is then RECOVERED: immune, non-transmitting, but still "
            "counted in the attack rate.\n"
        )
        + f"Mean infectious period is about {1.0 / max(task.epi_gamma, 1e-9):.1f} "
        f"steps, so the outbreak burns out on its own: a dose is worth only what "
        f"it saves before then.\n"
        if task.immunizes
        else ""
    )

    return (
        f"\n{label} ({len(sources)} nodes, fixed, NOT yours to choose; also\n"
        f"available inside your Strategy as `self.outbreak`): {listed}{overflow}\n"
        f"The cascade starts here at t=0 and spreads for {task.horizon} timesteps.\n"
        f"{rates}{tie}{delay}"
    )


max_listed_episodes = 6


setting_notes = {
    "partial_times": (
        "PARTIAL TIMESTAMPS. Each infected node was reported independently with "
        "probability {rate}, and every report carries the exact timestep it "
        "activated. Those times are ground truth and constrain the tree "
        "completely: an edge u -> v is only possible when t(u) < t(v). This is the "
        "setting the ordered-Steiner literature is defined on."
    ),
    "partial_nodes": (
        "PARTIAL NODES, NO TIMES. Each infected node was reported independently "
        "with probability {rate}, but every report's timestep is withheld: "
        "`observation.reported[v]` is None and `observation.times` is empty. You "
        "have to infer the ORDER before you can infer any parent."
    ),
    "final_snapshot": (
        "FINAL SNAPSHOT ONLY. There are no reports at all: `observation.reported` "
        "is empty and `observation.final_state` is a 0/1 vector of who was "
        "infected when the cascade ended. No times, no sources, no parameters. "
        "This is the hardest published formulation of this problem and the one "
        "where a transition kernel should be worth the most."
    ),
    "hidden_nodes": (
        "HIDDEN NODES. Reports arrive with times at rate {rate}, AND a fraction of "
        "the graph's nodes are absent entirely: `observation.visible` is False "
        "for them. A hidden node is not merely unobserved: it is not in the graph, "
        "naming one is rejected, and the cascade appears to jump across the gap it "
        "leaves."
    ),
}


def build_mask_block(task: TaskSpec) -> str:
    """
    Which of the four settings is running, and what the episodes look like.

    Spelled out because the setting is a PROTOCOL rather than a knob
    (research/cascade_reconstruction.md §8.3): a decoder selected under
    `final_snapshot` is solving a different problem from one selected under
    `partial_times`, and a model told the wrong one writes for an input it will not
    get. The masking DIRECTION is stated explicitly for the reason §8.2 trap 1
    gives: two papers in this literature use the symbol sigma for opposite
    quantities, so "reported with probability q" is written out rather than named.
    """
    if not task.decodes:
        return ""

    instances = list(task.instances)
    if not instances:
        return ""

    observed = [instance.observed_count for instance in instances]
    infected = [instance.infected_count for instance in instances]
    sources = [len(instance.sources) for instance in instances]
    num_nodes = instances[0].num_nodes
    rate = (
        f"{np.mean([o / max(i, 1) for o, i in zip(observed, infected, strict=True)]):.0%}"
    )
    setting = instances[0].observation.setting

    lines = [
        "",
        f"OBSERVATION PROTOCOL: {setting_notes.get(setting, setting).format(rate=rate)}",
        "",
        f"THE CASCADES YOU ARE SCORED ON ({len(instances)} masked episodes, mean "
        f"score across all of them):",
        f"  actually infected: {min(infected)}-{max(infected)} nodes "
        f"({100.0 * np.mean(infected) / num_nodes:.1f}% of N on average)",
        f"  you are shown: {min(observed)}-{max(observed)} of them "
        f"({rate} on average): the rest are yours to infer",
        f"  true sources per cascade: {min(sources)}-{max(sources)}, all committed "
        f"at timestep 0",
        f"  cascades ran for up to "
        f"{max(instance.horizon for instance in instances)} timesteps",
        "  REWARD = log-likelihood of your history per node (a 1/N prior per "
        "declared source, plus the kernel's probability of every transmission), "
        "minus the fraction of the observation you contradict",
        "",
    ]

    return "\n".join(lines)


def build_observation_block(task: TaskSpec) -> str:
    """
    What `y` actually is, and how many episodes the score averages over.

    Spelled out because the observation mode changes the problem: an MC marginal
    is a continuous, averaged view of the last wave, while a binarized draw is one
    realization. Ours is strictly MORE informative than the published protocol's
    (research/source_localization.md §2.9 risk 5), so a model told the wrong one
    calibrates its threshold against a distribution it will not see.
    """
    if not task.recovers or task.decodes:
        return ""

    instances = list(task.instances)
    if not instances:
        return ""

    counts = [instance.source_count for instance in instances]
    infected = [instance.infected_count for instance in instances]
    binary = all(set(np.unique(instance.observation)) <= {0.0, 1.0} for instance in instances)

    lines = [
        "",
        f"THE EPISODES YOU ARE SCORED ON ({len(instances)} cascades, mean "
        f"consistency across all of them):",
        f"  sources per episode: {min(counts)}-{max(counts)} nodes "
        f"({100.0 * np.mean(counts) / instances[0].num_nodes:.1f}% of N on average) "
        f", you are handed the exact count as `budget`",
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
            "FRONTIER: a node at 0.3 was reached late, a node at 1.0 early"
        )

    lines.append(
        f"  the cascades ran for up to {max(instance.horizon for instance in instances)} "
        f"timesteps; a longer cascade is a more saturated observation and therefore "
        f"a harder inversion"
    )
    lines.append("")

    return "\n".join(lines)


def build_cascade_block(task: TaskSpec) -> str:
    """
    What the logged cascades actually look like, and the two floors to clear.

    Spelled out because the SIZE DISTRIBUTION is the whole difficulty here: MSLE is
    an error in log space, so the geometric mean of the training sizes is the best
    instance-blind prediction, and a model told nothing about the distribution will
    spend its first iterations rediscovering it. Printing both floors up front turns
    that into a starting point instead of a wasted turn.
    """
    if not task.forecasts:
        return ""

    instances = list(task.instances)
    if not instances:
        return ""

    observed = np.array([entry.observation.popularity for entry in instances], dtype=float)
    actual = np.array([entry.actual for entry in instances], dtype=float)
    ratio = actual / np.maximum(observed, 1.0)
    geometric = float(np.exp(np.mean(np.log(np.maximum(actual, 1.0)))))

    lines = [
        "",
        f"THE CASCADES YOU ARE SCORED ON ({len(instances)} REAL logged cascades, "
        f"mean {task.prediction_metric.upper()} across all of them):",
        f"  observed by t_o: {observed.min():.0f}-{observed.max():.0f} adopters "
        f"(median {np.median(observed):.0f})",
        f"  actual at the horizon: {actual.min():.0f}-{actual.max():.0f} adopters "
        f"(median {np.median(actual):.0f})",
        f"  growth ratio actual/observed: median {np.median(ratio):.2f}x, "
        f"90th percentile {np.percentile(ratio, 90):.2f}x, "
        + (
            "most of these cascades are essentially OVER by the observation window, "
            "so the hard part is spotting the few that are not"
            if float(np.median(ratio)) < 1.5
            else "these cascades still have most of their growth ahead of them"
        ),
        f"  TWO FLOORS TO CLEAR: predicting the constant {geometric:.1f} (the "
        f"geometric mean, which is the best instance-BLIND answer under a log-space "
        f"error), and predicting `observation.popularity` unchanged (which is right "
        f"whenever a cascade is finished). Both are computed for you in the results; "
        f"an algorithm that ties with either has not used the instance.",
        f"  you see {instances[0].observation.observed_steps} of "
        f"{instances[0].observation.horizon} timesteps.",
        "",
    ]

    return "\n".join(lines)


def _budget_unit(task: TaskSpec) -> str:
    """What one unit of budget buys, in words, for the task block."""
    if task.immunizes:
        return {
            "vaccinate": "max doses total",
            "quarantine": "max isolations total",
            "edge_cut": "max arc cuts total",
            "contact_reduce": "max arc reductions total",
        }[task.epi_lever]

    if task.blocks:
        return {
            "add_node": "max counter-seeds total",
            "remove_node": "max node deletions total",
            "remove_edge": "max arc cuts total",
            "set_edge_weight": "max arc reweights total",
        }[task.budget_op]

    return "max removals total" if task.contains else "max seeds total"


def build_user_prompt(
    method: str,
    task: TaskSpec,
    graph: GraphInfo,
    strategy_mode: str = "free",
    allow_mc_algorithms: bool = False,
) -> str:
    if strategy_mode == "scored":
        # Library source is inspiration, not callable: ideas must be written
        # out inside score()/schedule()/source_score(), where they can be mutated
        if task.forecasts:
            menu = build_prediction_menu()
        elif task.decodes:
            menu = build_reconstruction_menu()
        elif task.recovers:
            menu = build_localization_menu()
        elif task.blocks:
            menu = build_blocking_menu(task.budget_op)
        elif task.immunizes:
            menu = build_immunization_menu(task.epi_lever)
        elif task.contains:
            menu = build_dismantling_menu()
        else:
            menu = build_algorithm_menu()

        hook = (
            "growth_factor()"
            if task.forecasts
            else "edge_cost()"
            if task.decodes
            else "source_score()"
            if task.recovers
            else "score()"
        )
        reference = (
            "PRIMITIVES API (available as `primitives`; spread-simulation "
            "functions are NOT available):\n"
            f"{build_primitives_reference(exclude=scored_blocked_primitives)}\n\n"
            "ALGORITHM IDEAS (NOT callable: steal the ideas into your "
            f"{hook}):\n"
            f"{menu}"
        )
        final_line = "Write the ScoredStrategy subclass now."
    elif task.forecasts:
        # A forecasting task's library is the POPULARITY-predictor pool. Every other
        # pool returns a NODE SET of some kind; this one returns a number, so showing
        # any of the others invites a program that answers a different question
        # entirely.
        reference = (
            build_prediction_reference(
                exclude=() if allow_mc_algorithms else mc_blocked_prediction
            )
            + "\n\nPRIMITIVES  (from coding_agent.tools.primitives, imported as "
            "`primitives`)\n"
            + build_primitives_reference(exclude=scored_blocked_primitives)
        )
        final_line = f"Write the Strategy now (method = {method})."
    elif task.decodes:
        # A decoding task's library is the TRAJECTORY-decoder pool. A seed set, a
        # removal set and a source set are all node lists; a trajectory is not, so
        # showing any of the other three invites a program that returns the wrong
        # object entirely.
        reference = (
            build_reconstruction_reference(
                exclude=() if allow_mc_algorithms else mc_blocked_reconstruction
            )
            + "\n\nPRIMITIVES  (from coding_agent.tools.primitives, imported as "
            "`primitives`)\n"
            + build_primitives_reference(exclude=scored_blocked_primitives)
        )
        final_line = f"Write the Strategy now (method = {method})."
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
    elif task.immunizes:
        # An epidemic task's library is the IMMUNIZATION pool. The dismantlers rank
        # by connectivity damage and know nothing about where the outbreak is or
        # that nodes recover, and the IM algorithms return seed sets: neither
        # answers the question this task asks. The generic primitives ride along
        # because an immunizer still needs centralities.
        reference = (
            build_immunization_reference(
                task.epi_lever,
                exclude=() if allow_mc_algorithms else mc_blocked_immunization,
            )
            + "\n\nPRIMITIVES  (from coding_agent.tools.primitives, imported as "
            "`primitives`)\n"
            + build_primitives_reference()
        )
        final_line = f"Write the Strategy now (method = {method})."
    elif task.blocks:
        # A blocking task's library is the BLOCKING pool: the IM algorithms return
        # seed sets to maximize with and the dismantlers know nothing about a second
        # cascade, so neither answers the question this task asks. The generic
        # primitives ride along because a blocker still needs centralities and RIS.
        reference = (
            build_blocking_reference(
                task.budget_op,
                exclude=() if allow_mc_algorithms else mc_blocked_blocking,
            )
            + "\n\nPRIMITIVES  (from coding_agent.tools.primitives, imported as "
            "`primitives`)\n"
            + build_primitives_reference()
        )
        final_line = f"Write the Strategy now (method = {method})."
    else:
        blocked = () if allow_mc_algorithms else mc_blocked_algorithms
        # Signatures alone do not teach the idiom: the model reproduces the
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

    if task.forecasts:
        budget_unit = "UNUSED: a predictor spends no budget on anything"
        objective_line = (
            f"predict how large each REAL cascade grows: MINIMIZE "
            f"{task.prediction_metric.upper()} against the logged popularity "
            f"(**LOWER IS BETTER**, unlike every other task here)"
        )
        ops_line = "allowed_ops = none: this task emits no actions\n"
    elif task.decodes:
        budget_unit = "UNUSED: a decoder spends no budget on anything"
        objective_line = (
            "recover the hidden trajectory that produced the observation: "
            "MAXIMIZE the kernel log-likelihood of your history per node minus the "
            "fraction of the observation you contradict (higher is better)"
        )
        ops_line = "allowed_ops = none: this task emits no actions\n"
    elif task.recovers:
        budget_unit = "max sources to name per episode"
        objective_line = (
            "recover the seed set that produced the observation: MAXIMIZE "
            "consistency, minus the re-simulation MSE of your sources against the "
            "observation (higher is better, 0 is perfect)"
        )
        ops_line = "allowed_ops = none: this task emits no actions\n"
    else:
        budget_unit = _budget_unit(task)
        objective_line = (
            "MINIMIZE how far the RUMOUR spreads (lower is better)"
            if task.blocks
            else "MINIMIZE the ATTACK RATE: how many nodes are EVER infected "
            "(lower is better)"
            if task.immunizes
            else "MINIMIZE the final infected count (lower is better)"
            if task.contains
            else task.objective
        )
        ops_line = (
            f"allowed_ops = {', '.join(task.allowed_ops)}   (any other op is REJECTED)\n"
        )

    return f"""\
TASK: {task.task}, {objective_line}
diffusion_model = {task.diffusion_model}
budget = {task.budget}   ({100.0 * task.budget / graph.num_nodes:.1f}% of nodes, {budget_unit})
horizon = {task.horizon} (timesteps)
{ops_line}{build_outbreak_block(task)}{build_plan_oracle_block(task)}{build_round_block(task)}{build_observation_block(task)}{build_mask_block(task)}{build_cascade_block(task)}
{build_graph_profile(graph)}

{reference}

{final_line}"""


# Scored mode: the agent edits an algorithm's internals (score/schedule hooks),
# never whole programs and never compositions over the library
def _scored_localization_system(task: TaskSpec) -> str:
    """
    Scored mode for the inverse task (research/source_localization.md §2.4.2).

    The same trick as the intervention half: a fixed harness the agent cannot
    override, with one hook it can, and it is a better fit here than anywhere
    else: LPSI, the Comin-Costa centralities and rumor centrality are ALL exactly
    node-scoring functions over the observed state, so the constrained search space
    is directly comparable to the classical methods rather than a subset of them.
    """
    oracle_line = (
        "- `self.predict_marginals(seeds)` -> np.ndarray of P(infected at the end) "
        "if the\n  cascade had started from `seeds`. Every call is a full rollout "
        "and calls are\n  counted, so use it sparingly inside score(): it runs "
        "once per candidate per pick."
        if task.forward_model
        else "- `self.predict_marginals` RAISES in this condition: this arm has no "
        "forward\n  model by design. Score from structure and the observation alone."
    )

    return f"""\
You are designing the SCORING RULE of a Source Localization algorithm, not a
whole program.

Given a graph and an OBSERVED diffusion state, the task is to recover the SEED SET
that produced it. You are scored on consistency (minus the re-simulation error
of the sources you name against the observation), averaged over
many labelled cascades. HIGHER IS BETTER.

A fixed harness (ScoredStrategy.localize) greedily names the highest-scoring node
until the budget is spent. You may override ONLY:

- source_score(self, node, graph, observation, selected) -> float
    Called for every candidate node at every pick. `observation[v]` is
    P(node v was infected) at the end of the cascade, in [0, 1]. `selected` is the
    tuple of sources already named: use it to penalize a candidate whose
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

REPLY SHAPE (adapt the logic: improve on it, do not return it unchanged):
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


def _scored_reconstruction_system(task: TaskSpec) -> str:
    """
    Scored mode for the decoding task, and the tightest of the three fits.

    Every ordered-Steiner method in research/cascade_reconstruction.md §3 IS
    exactly a shortest-path computation under an arc cost: `delayed-bfs`,
    `closure`, `greedy`, WPCT and CulT differ in their constraints and their
    attachment order, not in the shape of the object they optimize. Fixing the
    harness and exposing only the cost therefore leaves a search space that
    CONTAINS the classical methods rather than a subset of them, which is more
    than can be said for the scored harness on either intervention task.
    """
    return """\
You are designing the ARC-COST FUNCTION of a Cascade Reconstruction algorithm,
not a whole program.

A diffusion already happened on this graph and you only saw part of it. The task
is to recover the whole history: which nodes were infected, when each activated,
and who infected whom. Your score is the kernel log-likelihood of the recovered
history per node minus the fraction of the observation it contradicts, averaged
over many masked cascades. HIGHER IS BETTER, and an implausible parent is exactly
what the likelihood punishes: recovering the node set is nearly free.

A fixed harness (ScoredStrategy.reconstruct) grows a tree out of the observed
region, cheapest arc first under your cost, respecting every observed activation
time, and then attaches parents. You may override ONLY:

- edge_cost(self, source, target, probability, graph, observation) -> float
    How IMPLAUSIBLE the transmission `source -> target` is. LOWER means the
    harness prefers it, so a most-likely path is a cheapest path. `probability` is
    the arc's own transmission probability p(source -> target); the default is
    -log(p), which is the likelihood metric every published method here uses.
    `observation.reported` is the report set and `observation.times` the known
    activation times, so a rule can make an arc into an already-reported node
    cheap, or penalize one that leaves the observed region.

RULES:
- Overriding reconstruct (or plan_horizon, or localize) is REJECTED by the executor.
- `reconstruction_algorithms.*` does NOT exist here. Write the cost from graph
  structure, the arc probability, the observation and `primitives`.
- Reply with exactly ONE fenced ```python block containing ONE class subclassing
  ScoredStrategy. No imports, no module-level code, no prose.
- `GraphInfo` has .num_nodes, .out_neighbors(node), .in_neighbors(node),
  .degree(node), .edge_index, .ic_probs.

REPLY SHAPE (adapt the logic: improve on it, do not return it unchanged):
```python
class MyDecoder(ScoredStrategy):
    def edge_cost(self, source, target, probability, graph, observation):
        # -log p, without importing math: the harness only needs a monotone cost
        likelihood = max(float(probability), 1e-9)
        cost = 1.0 / likelihood
        # An arc into a node we actually SAW is far more likely to be real than one
        # into a node we are only guessing at
        if target in observation.reported:
            cost *= 0.5
        # A hub absorbs every path through it; discount arcs that leave one
        return cost * (1.0 + 0.01 * graph.degree(target))
```
"""


def _scored_prediction_system(task: TaskSpec | None) -> str:
    """
    Scored mode for cascade prediction: the agent writes a GROWTH MULTIPLIER.

    The tightest fit of the four scored harnesses, and for a published reason.
    Szabo & Huberman's founding result is that `log P(t_p)` is near-linear in
    `log P(t_o)` (that the whole problem is a multiplier) so a search over
    multipliers is a search over exactly the space the feature line of this
    literature occupies rather than a subset of it.
    """
    metric = (task.prediction_metric if task is not None else "msle").upper()

    return f"""\
You are designing the GROWTH RULE of a Cascade Popularity Prediction algorithm,
not a whole program.

A fixed harness (ScoredStrategy.predict) multiplies the OBSERVED popularity by
whatever your rule returns and clamps the result from below at the observed count.
You may override ONLY this method:

- growth_factor(self, features, graph, observation) -> float
    A multiplier on `observation.popularity`. 1.0 means "this cascade is over";
    3.0 means "it will triple". Your score is {metric}, an ERROR in LOG space, so
    what you are really choosing is an ADDITIVE offset on `log(popularity)`: being
    off by a factor of 2 costs the same whether the cascade is 20 or 2000.

`features` is `cascade_features(graph, observation)`, already computed:
  observed, log_observed, observed_steps, remaining_steps,
  rate, rate_first_half, rate_second_half, acceleration, time_to_half,
  last_wave, peak_wave, root_degree, log_root_degree,
  mean_degree, max_degree, frontier_size, frontier_mean_degree,
  exposed, exposure_ratio, spread_breadth, graph_fraction

The single most predictive one in this literature is `rate_second_half`: the
adoption rate in the SECOND HALF of the observation window. It beats every
structural feature by a wide margin. `acceleration` is its signed form. Start
there, and use `remaining_steps` to scale: a multiplier that ignores how much
future is left will be right for one horizon and wrong for every other.

`self.fit_examples` is the labelled selection split (each with `.observation` and
`.actual`), so you may fit a constant or a small regression rather than guessing.

RULES:
- Overriding predict() is REJECTED by the executor.
- `prediction_algorithms.*` does NOT exist here. Write your own rule.
- Reply with exactly ONE fenced ```python block containing ONE class subclassing
  ScoredStrategy. No imports, no module-level code, no prose.
- `GraphInfo` has .num_nodes, .out_neighbors(node), .in_neighbors(node),
  .degree(node), .edge_index, .ic_probs.

REPLY SHAPE (adapt the logic: improve on it, do not return it unchanged):
```python
class MyGrowth(ScoredStrategy):
    def growth_factor(self, features, graph, observation):
        # Still accelerating -> more growth left; flattened -> nearly none
        momentum = features["rate_second_half"] / max(features["rate"], 1e-6)
        remaining = features["remaining_steps"]
        return 1.0 + min(momentum, 3.0) * remaining / max(remaining + 2.0, 1.0)
```
"""


def _scored_system(task: TaskSpec | None) -> str:
    if task is not None and task.forecasts:
        return _scored_prediction_system(task)

    if task is not None and task.decodes:
        return _scored_reconstruction_system(task)

    if task is not None and task.recovers:
        return _scored_localization_system(task)

    blocks = task is not None and task.blocks
    contains = task is not None and task.contains

    if blocks:
        problem = "an Influence Blocking algorithm"
        picks = {
            "add_node": "counter-seeds",
            "remove_node": "deletions",
            "remove_edge": "arc cuts",
            "set_edge_weight": "arc reweights",
        }[task.budget_op]
        goal = (
            "Higher score = spent on sooner. You are MINIMIZING how far the RUMOUR "
            "spreads, and the rumour's own seeds are on `self.outbreak`, so a high "
            "score should mean 'this is where the rumour is going and I can get "
            "there first'"
        )
    elif contains:
        problem = "a Critical Node Detection algorithm"
        picks = "removes"
        goal = (
            "Higher score = removed sooner. You are MINIMIZING the final infected "
            "count, so a high score should mean 'cutting this node hurts the cascade "
            "most'"
        )
    else:
        problem = "an Influence Maximization algorithm"
        picks = "seeds"
        goal = "Higher score = picked sooner"
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
You are designing the SCORING RULE of {problem}, not a whole program.

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

REPLY SHAPE (adapt the logic: improve on it, do not return it unchanged):
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
    decodes = task is not None and task.decodes
    forecasts = task is not None and task.forecasts

    blocks = task is not None and task.blocks
    immunizes = task is not None and task.immunizes

    if task is not None:
        if forecasts:
            horizon_note = prediction_timing_note
        elif decodes:
            horizon_note = reconstruction_timing_note
        elif recovers:
            horizon_note = localization_timing_note
        elif task.adaptive:
            horizon_note = adaptive_timing_note
        elif blocks:
            horizon_note = blocking_timing_note
        elif immunizes:
            horizon_note = epidemic_timing_note
        elif contains:
            horizon_note = containment_timing_note
        elif any(op in task.allowed_ops for op in edge_ops):
            horizon_note = temporal_scheduling_note
        else:
            horizon_note = seed_timing_note

        remove_note = remove_semantics_notes[task.remove_semantics]

    # Only worth stating when the strategy may actually emit the op, which an
    # inverse task never does
    if (
        recovers
        or forecasts
        or "remove_node" not in (task.allowed_ops if task is not None else ())
    ):
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
    if forecasts:
        resolved = "predict"
    elif decodes:
        resolved = "reconstruct"
    elif recovers:
        resolved = "localize"
    else:
        # evolve generates plan_horizon strategies under the same contract as one_shot
        resolved = "one_shot" if method == "evolve" else method

    base = _common_rules(task) + method_bodies[resolved]

    # Worked programs, matched to the family: the IM exemplars emit add_node,
    # which a containment task REJECTS and an inverse task has no use for at all,
    # so showing the wrong set would cost the search its first iteration on a
    # repair turn for a contract nobody asked for
    if resolved == "predict":
        base += prediction_exemplars
    elif resolved == "reconstruct":
        base += reconstruction_exemplars
    elif resolved == "localize":
        base += localization_exemplars
    elif resolved == "one_shot":
        if blocks:
            from coding_agent.blocking import lever_of

            base += blocking_exemplars[lever_of(task)]
        elif immunizes:
            # Only the node levers have exemplars: the edge levers' contract is one
            # line different from the blocking edge lever's, and the two node ones
            # are where every published method in this literature lives
            base += (
                epidemic_exemplars
                if task.epi_lever in ("vaccinate", "quarantine")
                else blocking_exemplars["edge_block"]
            )
        elif contains:
            base += containment_exemplars
        else:
            base += one_shot_exemplars

    if resolved in ("localize", "reconstruct", "predict") or method in (
        "one_shot",
        "evolve",
        "adaptive",
    ):
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
        "REDESIGN the approach: keep the same contract but change the core idea, "
        "different structural signals, different selection logic. Do not just "
        "re-tune the parent."
    ),
}


# Appended to the evolve/adaptive opening turn on arms whose evaluator can
# answer probes; the native arm never sees it and refuses a hallucinated block
probe_contract = (
    "\n\nPROBES. Beside your ```python block you may add ONE fenced block\n"
    "```probes\n{\"probes\": [...]}\n```\n"
    "asking the evaluator what-if questions about the plan your script produces "
    "this iteration; the answers arrive with the NEXT iteration's feedback, and "
    "every probe is metered evaluator work like any rollout. At most "
    f"{max_probes_per_generation} per iteration. Ops: "
    '{"op": "drop", "node": N} = plan reward with node N\'s actions removed '
    "(N's marginal contribution); "
    '{"op": "swap", "a": N, "b": M} = reward if M replaces N in the plan; '
    '{"op": "region", "nodes": [...]} = expected probability mass the plan '
    "captures inside that node set."
)


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
    candidate that did not become the parent never surfaces one, so without
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
PARENT: the best in the population, which is NOT necessarily your last attempt
(reward={parent["reward"]:.2f}):
```python
{parent["script"]}
```
Parent rollout diagnostics:
{parent["summary"]}
{inspiration_text}{error_text}
OPERATION: {operator.upper()}: {evolve_operator_instructions[operator]}
Reply with one ```python block."""


# GA-routing baseline: the LLM selects from the pool but never synthesizes code
routing_system = """\
You are an algorithm-selection router for Influence Maximization.
You will be given a task, a graph description, and a menu of classical library
algorithms. Pick the single most promising algorithm for this graph and task.

Reply with EXACTLY ONE algorithm name from the menu: no code, no punctuation,
no explanation."""

containment_routing_system = """\
You are an algorithm-selection router for Critical Node Detection.
You will be given a task, a graph description, and a menu of classical network
DISMANTLING algorithms. Each returns a set of nodes to DELETE from the graph. Pick
the single one most likely to MINIMIZE the final infected count on this graph.

Reply with EXACTLY ONE algorithm name from the menu: no code, no punctuation,
no explanation."""


localization_routing_system = """\
You are an algorithm-selection router for Source Localization.
You will be given a task, a graph description, a summary of the cascades to invert,
and a menu of classical source-localization algorithms. Each takes the observed
diffusion state and returns the nodes it believes STARTED the cascade. Pick the
single one whose recovered sets are most CONSISTENT with the observations on these
instances: re-simulating them should reproduce the observed state.

Reply with EXACTLY ONE algorithm name from the menu: no code, no punctuation,
no explanation."""


blocking_routing_system = """\
You are an algorithm-selection router for Influence Blocking.
You will be given a task, a graph description, the rumour's own seed nodes, and a
menu of classical influence-blocking algorithms. Each returns the intervention this
task's lever buys: counter-seeds, node deletions, or arcs. Pick the single one most
likely to MINIMIZE how far the rumour spreads on this graph.

Reply with EXACTLY ONE algorithm name from the menu: no code, no punctuation,
no explanation."""


reconstruction_routing_system = """\
You are an algorithm-selection router for Cascade Reconstruction.
You will be given a task, a graph description, a summary of the masked cascades to
recover, and a menu of classical trajectory decoders. Each takes the partial
observation and returns the whole history, which nodes were infected, when, and
who infected whom. Pick the single one most likely to MAXIMIZE the tree-weighted
score on these instances.

Reply with EXACTLY ONE algorithm name from the menu: no code, no punctuation,
no explanation."""


prediction_routing_system = """\
You are an algorithm-selection router for Cascade Popularity Prediction.
You will be given a task, a graph description, a summary of the REAL logged
cascades to forecast, and a menu of classical popularity predictors. Each takes a
cascade's observed prefix and returns the popularity it will reach by the horizon.
Pick the single one most likely to MINIMIZE the prediction error on these
cascades: LOWER is better here.

Reply with EXACTLY ONE algorithm name from the menu: no code, no punctuation,
no explanation."""


epidemic_routing_system = """\
You are an algorithm-selection router for Epidemic Control.
You will be given a task, a graph description, the outbreak's index cases, and a
menu of classical immunization algorithms. Each returns the intervention this
task's lever buys: nodes to dose or isolate, or arcs to cut or reduce. Pick the
single one most likely to MINIMIZE the attack rate on this graph.

Reply with EXACTLY ONE algorithm name from the menu: no code, no punctuation,
no explanation."""


def build_routing_system(task: TaskSpec | None = None) -> str:
    if task is not None and task.forecasts:
        return prediction_routing_system

    if task is not None and task.decodes:
        return reconstruction_routing_system

    if task is not None and task.recovers:
        return localization_routing_system

    if task is not None and task.blocks:
        return blocking_routing_system

    if task is not None and task.immunizes:
        return epidemic_routing_system

    return (
        containment_routing_system
        if task is not None and task.contains
        else routing_system
    )


def build_routing_prompt(task: TaskSpec, graph: GraphInfo) -> str:
    if task.forecasts:
        # The menu MUST be the predictor pool: run.py parses the reply against
        # prediction_names, so an IM menu here is a guaranteed parse failure
        budget_unit = "UNUSED: a predictor spends no budget"
        objective_line = (
            f"MINIMIZE {task.prediction_metric.upper()} against the logged "
            f"popularity (LOWER is better)"
        )
        menu = build_prediction_menu()
    elif task.decodes:
        budget_unit = "UNUSED: a decoder spends no budget"
        objective_line = (
            "MAXIMIZE the kernel log-likelihood of the recovered history per node "
            "minus the fraction of the observation it contradicts (higher is better)"
        )
        menu = build_reconstruction_menu()
    elif task.recovers:
        budget_unit = "max sources to name per episode"
        objective_line = (
            "MAXIMIZE consistency: minus the re-simulation MSE of the recovered "
            "sources against the observation (higher is better)"
        )
        menu = build_localization_menu()
    else:
        budget_unit = _budget_unit(task)
        objective_line = (
            "MINIMIZE how far the RUMOUR spreads (lower is better)"
            if task.blocks
            else "MINIMIZE the ATTACK RATE: how many nodes are EVER infected "
            "(lower is better)"
            if task.immunizes
            else "MINIMIZE the final infected count (lower is better)"
            if task.contains
            else task.objective
        )
        if task.blocks:
            menu = build_blocking_menu(task.budget_op)
        elif task.immunizes:
            menu = build_immunization_menu(task.epi_lever)
        elif task.contains:
            menu = build_dismantling_menu()
        else:
            menu = build_algorithm_menu()

    return f"""\
TASK: {task.task}, {objective_line}
diffusion_model = {task.diffusion_model}
budget = {task.budget}   ({100.0 * task.budget / graph.num_nodes:.1f}% of nodes, {budget_unit})
horizon = {task.horizon} (timesteps)
{build_outbreak_block(task)}{build_observation_block(task)}{build_mask_block(task)}{build_cascade_block(task)}
{build_graph_profile(graph)}

ALGORITHM MENU:
{menu}

Reply with exactly one name from the menu."""


# Per-iteration diff cap in the write-up prompt: a restructure can rewrite the
# whole script, and the model only needs enough of it to say what changed
max_diff_lines = 120


def _iteration_change(record: dict, by_iteration: dict, winner: str) -> str:
    """The exact edit one iteration made, as a unified diff against its target."""
    script = record.get("script")
    if script is None:
        return "  (no script recorded for this iteration)"

    parent = by_iteration.get(record.get("parent_iteration"))
    if parent is None or parent.get("script") is None:
        if record.get("operator") in ("refine", "restructure"):
            return "  (edited the population best; the target was not recorded)"
        if script == winner:
            return "  first script, from scratch: identical to the winning script above"
        return f"  first script, from scratch:\n```python\n{script}\n```"

    diff = list(
        difflib.unified_diff(
            parent["script"].splitlines(),
            script.splitlines(),
            fromfile=f"iteration {parent['iteration']}",
            tofile=f"iteration {record['iteration']}",
            lineterm="",
            n=2,
        )
    )
    if not diff:
        return f"  resubmitted iteration {parent['iteration']} unchanged"
    if len(diff) > max_diff_lines:
        diff = diff[:max_diff_lines] + [f"... {len(diff) - max_diff_lines} more diff lines"]

    return "```diff\n" + "\n".join(diff) + "\n```"


def build_explanation_prompt(
    script: str, reward: float, history: list[dict], summary: str
) -> str:
    """
    Closing turn: describe the search and the winner in plain English.

    Sent on the generation thread, but the thread is trimmed to the last few
    exchanges, so the model cannot see most of what it wrote. `history` carries
    every iteration's script and the iteration it edited, and each one is shown
    here as a diff against its target: that is what makes the iteration log
    describable at all. `script` is echoed because the winner is the max over
    iterations, not the last turn, without it the model narrates the wrong
    algorithm.
    """
    by_iteration = {record["iteration"]: record for record in history}
    iteration_blocks = "\n\n".join(
        f"### iteration {record['iteration']}: "
        + (
            f"FAILED: {record['error'].splitlines()[0]}"
            if record.get("error")
            else f"reward={record['reward']:.2f} (best so far {record['best']:.2f})"
        )
        + (f" [operator={record['operator']}]" if record.get("operator") else "")
        + (
            f", edited iteration {record['parent_iteration']}"
            if record.get("parent_iteration") is not None
            else ""
        )
        + "\n"
        + _iteration_change(record, by_iteration, script)
        for record in history
    )
    scored = [record for record in history if record.get("reward") is not None]
    trajectory_line = (
        f"first scored attempt: iteration {scored[0]['iteration']} at "
        f"{scored[0]['reward']:.2f}; winner: {reward:.2f}"
        if scored
        else "no scored attempt recorded"
    )

    return f"""\
The search is over. Write the final report in plain English: no code blocks.

THE WINNING SCRIPT (reward={reward:.2f}), this is the one to describe, and it
is NOT necessarily your last attempt:
```python
{script}
```
Its rollout diagnostics:
{summary}

Every iteration, with the exact change it made to the script it was editing (a
unified diff against that script; a script written from scratch is given in
full). Each iteration is scored on a fresh random realization and "best so far"
is the incumbent re-scored on that same realization, so judge an iteration
against the best-so-far beside it, never against another iteration's number.
Trajectory: {trajectory_line}.
{iteration_blocks or "  (none recorded)"}

Reply with GitHub-flavoured markdown using EXACTLY these headings, in this
order, and nothing before the first one:

## Summary
Two or three sentences: what the final algorithm is, and what it scored.

## Iteration log
One `### Iteration N: reward X` subsection per iteration above. For each: what
you were trying to fix, what you changed in the ALGORITHM, and whether it
worked. Read the change off the diff given for that iteration: it is the ground
truth even where this conversation no longer shows the attempt. Be specific
("raised the redundancy penalty from 1.0 to 2.5", not "tuned parameters") but
describe it at the level of the method, not the code: no libraries, no variable
names, no data-structure details. Only an iteration with no diff recorded may be
described as unknown; never invent a change.

## From first attempt to winner
Three short paragraphs, all read off the diffs above. (1) The starting
algorithm: what the first script did, in method terms. (2) The intermediate
step that mattered most: the one iteration whose change moved the reward the
most, and what that change was. (3) Start to finish: what is different between
the first script and the winner, and how much the reward moved.

## Closest classical algorithm
Name the published or library algorithm the winner is closest to (the baseline
table in the opening turn of this conversation lists the ones that were run
here, with their rewards), say what the winner shares with it, and state
exactly how it differs: added stages, a changed scoring rule, a different
candidate set. If the winner is a composition of several known algorithms, name
each and say which one supplies which stage.

## How the final algorithm works
A short numbered walkthrough of the winning ALGORITHM, in the order it acts,
written the way a paper's method section would describe it. One or two sentences
per step: what that stage does to the graph or the cascade, and what it buys the
result. Rules:
- Name every library algorithm the script calls (`rps`, `degree_removal`, ...)
  and what it is used for.
- Start each step that is your own idea rather than a library call with
  **[new]**, and say what it does that the library algorithms do not. Point out
  anything else interesting about the design in the same way.
- Skip everything that is not the algorithm: reading inputs, clamping the
  budget, empty-graph guards, deduplication, cleaning, random seeds, bookkeeping,
  validation, and how the plan is packaged into action bags.
- No libraries (no NumPy, no arrays, no bit masks), no quoted code, no variable
  or function names. A formula is fine when it IS the idea, such as the scoring
  rule; a number is fine only when the method depends on it.

## What is new and why it wins
For each step marked [new] above: what the agent figured out, why the closest
classical algorithm cannot produce it (what it does not see, or does not do),
and how much of the gain over that classical algorithm it accounts for, pointing
at the iteration that introduced it and the reward move it caused. Then the
structural property of this graph the winner exploits, and which part of the
diagnostics above shows it working. If nothing is new and the winner is a known
algorithm with tuned parameters, say so plainly.

## Limitations
Where this algorithm would do badly, and what you would try next with more
iterations.

Write for someone who has not read the script or this conversation and wants to
know what the algorithm does, not how the code is laid out. Do not invent
results that are not in the numbers above."""


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
    deliberately redundant, but it is what makes "EDIT this one" unambiguous,
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
            f"\nYOUR BEST SCRIPT SO FAR (reward {incumbent_reward:.2f}), THIS is "
            f"the one to edit, NOT the attempt above, which scored worse:\n"
            f"```python\n{incumbent_script}\n```\n"
        )
        instruction = (
            f"EDIT THE BEST SCRIPT ABOVE to improve {objective}. The attempt that "
            f"just ran is shown so you can see what did not work: do not build on "
            f"it. Change what the diagnostics say is weak and keep what is working."
        )
    else:
        target_text = ""
        instruction = (
            f"That attempt is your best so far. EDIT it to improve {objective}: "
            f"change what the diagnostics say is weak and keep what is working, "
            f"rather than starting a new design from scratch."
        )

    return f"""\
Your previous strategy achieved {objective} (reward) = {reward}.
{delta_text}Trajectory summary: {summary}{reference_text}{error_text}{credit_text}{script_text}{target_text}
{instruction}
Reply with one ```python block containing the complete updated script."""
