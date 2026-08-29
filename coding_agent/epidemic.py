"""
Epidemic control: the four intervention levers, the outbreak, and prevented infections.

The harness piece `epidemic_control` needs and the other tasks do not, in the same
shape `coding_agent/blocking.py` has for influence blocking. Three things live here
and each is a decision `research/epidemic_control.md` §2.5 forces rather than a
convenience:

  * **The four levers.** §2.5's table maps the intervention vocabulary of this
    literature onto our action ops, and the map is not one-to-one: two different
    interventions both spend `remove_node`:

      - `vaccinate`: the node is IMMUNE. It leaves the graph, cannot be infected,
        and is never counted in the attack set. This is NetShield's, DAVA's,
        Pastor-Satorras & Vespignani's and Cohen et al.'s intervention, and under
        `blocked` semantics it is exactly what `remove_node` already means.
      - `quarantine`: the node is ISOLATED, not immune. Its incident arcs are cut
        but the node stays in the graph, stays susceptible, and stays counted. §8.2
        trap 7 is why the distinction is worth a lever rather than a footnote:
        "Recovered is not removed", and papers that report nodes-saved against
        "no intervention" versus against "random vaccination" differ by a large
        constant for precisely this reason. A quarantined node also cannot be
        infected in practice (nothing reaches it), so the two levers differ in the
        DOSE ACCOUNTING rather than in the epidemic, which is the honest version of
        the difference and is measurable.
      - `edge_cut`: `remove_edge`, the Van Mieghem / NetMelt / Kimura lever.
      - `contact_reduce`: `set_edge_weight`, and the reason §2.2 says to write our
        own stepper: NDlib's SIR/SIS/SEIR carry no per-edge parameter, so this
        entire branch of the literature (social distancing, DURLECA's mobility
        multiplier, Fractional Immunization's continuous allocation) is
        inexpressible against them. `--contact-reduction r` scales the arc to
        `r * beta_uv`, so `r = 0` is a cut and `r = 0.5` is halved contact.

    `--epi-lever` picks one and sets `budget_op` and `allowed_ops` together, so an
    arm can never be budgeted for one op and permitted another.

  * **The outbreak.** Exogenous, injected at t=0 as `add_node` index cases the
    planner neither chooses nor pays for: identical to critical node detection, and
    it reuses `containment.Outbreak` for the seeding half.

  * **Prevented infections (§8.3).** `|R(inf)| unprotected - |R(inf)| with doses`.
    The reward the search optimizes is the remaining attack rate (lower is better,
    so no sign work is needed anywhere); prevented infections is that subtracted
    from the unprotected reference, which is what an immunization table reports.

Warning: WHAT THIS DELIBERATELY DOES NOT DO. §2.5 lists two more rows and both are
out of scope with a stated reason. **Contact tracing is not an action**: it changes
the OBSERVATION, not the graph or the state, and belongs in a POMDP observation
model we do not have. **Quarantine with a DURATION** is not expressible either: a
release timer is hidden state, the compartment head is Markov in (state, action),
and a head that cannot see the timer would be fit against a transition it cannot
explain. The `quarantine` lever above is therefore isolation for the rest of the
episode, which is what every static immunization baseline in §3 does anyway.
"""

from dataclasses import dataclass

from coding_agent.containment import expand_removals
from coding_agent.types import ActionFn, ActionOp, GraphInfo, State, TaskSpec
from data.wm_epidemic import default_burn_in
from world_model.wm_metrics import epidemic_curve_metrics, immunization_metrics

# §2.5's table, as the (budget_op, allowed_ops) pair each row implies
vaccinate = "vaccinate"
quarantine = "quarantine"
edge_cut = "edge_cut"
contact_reduce = "contact_reduce"

epidemic_levers = {
    vaccinate: ("remove_node", ("remove_node",)),
    quarantine: ("remove_node", ("remove_node",)),
    edge_cut: ("remove_edge", ("remove_edge",)),
    contact_reduce: ("set_edge_weight", ("set_edge_weight",)),
}

lever_papers = {
    vaccinate: "Pastor-Satorras & Vespignani PRE'02, Cohen PRL'03, NetShield ICDM'10, DAVA SDM'14",
    quarantine: "Holme PRE'02 (recalculated removal), RLGN ICML'21 (test-and-isolate)",
    edge_cut: "Kimura TKDD'09, Van Mieghem PRE'11, NetMelt CIKM'12, GreedyWalk SDM'15",
    contact_reduce: "Fractional Immunization SDM'13, Preciado TCNS'14, DURLECA KDD'20",
}

# What one unit of budget buys, in the units the report prints
lever_shape = {
    vaccinate: "node",
    quarantine: "node",
    edge_cut: "arc",
    contact_reduce: "arc",
}

valid_levers = tuple(epidemic_levers)

# `set_edge_weight` multiplier under `contact_reduce`. 0.0 is a full cut through
# the weight channel, which makes it directly comparable to `edge_cut` at the same
# k and isolates "graded vs all-or-nothing" as its own axis.
default_contact_reduction = 0.0


def resolve_lever(lever: str) -> tuple[str, tuple]:
    """(budget_op, allowed_ops) for one lever."""
    if lever not in epidemic_levers:
        raise ValueError(
            f"unknown epidemic lever {lever!r}; choose one of {valid_levers}"
        )

    return epidemic_levers[lever]


def isolate_node_ops(graph: GraphInfo, node: int) -> list[ActionOp]:
    """
    Every incident arc of `node`, and NOT the node itself: quarantine as a bag.

    The one-line difference from `containment.delete_node_ops`, and the whole
    content of the vaccinate/quarantine distinction: the node stays in the graph,
    so it stays susceptible and stays counted in the attack rate if the outbreak
    had already reached it. Both orientations are emitted because the simulator's
    edge store is keyed by arc.
    """
    node = int(node)
    ops = []

    for neighbour in set(graph.out_neighbors(node)) | set(graph.in_neighbors(node)):
        ops.append(ActionOp("remove_edge", node, int(neighbour)))
        ops.append(ActionOp("remove_edge", int(neighbour), node))

    return ops


def expand_immunization(
    bag: list[ActionOp], graph: GraphInfo, lever: str
) -> list[ActionOp]:
    """
    Rewrite every bare `remove_node` in `bag` as the bag its LEVER means.

    Under `vaccinate` that is `containment.expand_removals` unchanged: the node
    plus its incident arcs, because both structured heads document that a blocked
    node's edges are gone from `edge_index` and rely on it. Under `quarantine` it is
    the incident arcs ALONE. The edge levers pass through untouched.

    Idempotent, like `expand_removals`: a strategy that already emitted the incident
    ops gets them deduplicated rather than doubled.
    """
    if lever != quarantine:
        return expand_removals(bag, graph)

    expanded = []
    seen = set()

    for action in bag:
        candidates = (
            isolate_node_ops(graph, action.target)
            if action.op == "remove_node"
            else [action]
        )

        for op in candidates:
            key = (op.op, int(op.target), op.destination)
            if key in seen:
                continue

            seen.add(key)
            expanded.append(op)

    return expanded


@dataclass()
class Immunization:
    """The exogenous outbreak an epidemic-control policy is trying to stop."""

    sources: tuple
    graph: GraphInfo
    lever: str = vaccinate

    @property
    def bag(self) -> list[ActionOp]:
        return [ActionOp("add_node", int(node)) for node in self.sources]

    def wrap(self, action_fn: ActionFn) -> ActionFn:
        """
        Seed the outbreak at t=0 and expand the policy's doses at every step.

        Applied AFTER validation for both halves, exactly as the containment
        wrapper is: the outbreak's `add_node` ops are exogenous and would be
        rejected under `--allowed-ops remove_node`, and a dose's incident
        `remove_edge` ops are the mechanics of one dose rather than a second
        intervention to be charged for.

        Order matters and this is the right one: the index cases go in FIRST, so a
        policy that doses a source node at t=0 still wins: `apply_actions` walks
        the bag in order, so the later `remove_node` overwrites the earlier
        `add_node` and the node ends up immunized rather than infectious. Whether
        that is ALLOWED is a separate question the validator answers, and the
        answer is no (see `executor.validate_actions`): dosing patient zero ends
        the outbreak rather than containing it.
        """

        def immunized(state: State, timestep: int) -> list[ActionOp]:
            policy_bag = expand_immunization(
                list(action_fn(state, timestep)), self.graph, self.lever
            )

            return (self.bag if timestep == 0 else []) + policy_bag

        return immunized


def build_immunization(graph: GraphInfo, task: TaskSpec) -> Immunization | None:
    """
    None for every task that is not compartmental.

    Returned even with an EMPTY outbreak, for the same reason `build_outbreak` is:
    the wrapper does two jobs, and the dose expansion is not optional: both the
    compartment head and the simulator assume a vaccinated node's arcs are gone
    from `edge_index`.
    """
    if not task.immunizes:
        return None

    return Immunization(
        sources=tuple(int(node) for node in task.outbreak),
        graph=graph,
        lever=task.epi_lever,
    )


def immunization_plan(
    picks,
    graph: GraphInfo,
    budget: int,
    lever: str,
    horizon: int = 0,
    protected: tuple = (),
    contact_reduction: float = default_contact_reduction,
) -> list[list[ActionOp]]:
    """
    A library selector's output as a horizon-shaped plan, whatever shape it returns.

    The node levers hand back node ids and the edge levers hand back `(u, v)` arcs,
    so this is the single place the two meet the one plan format: the same job
    `blocking.blocking_plan` does, and for the same reason: the published algorithms
    are not rewritten to know what a plan is.

    `protected` drops the outbreak's own sources. That filter is where a library
    immunizer meets the no-dosing-patient-zero rule, and it tops the set back up
    from the highest-degree survivors so a filtered allocation still spends its
    whole budget rather than silently running short.
    """
    budget_op, _ = resolve_lever(lever)
    guarded = {int(node) for node in protected}
    bag = []
    seen = set()

    for pick in picks:
        if len(bag) >= budget:
            break

        if budget_op == "remove_node":
            node = int(pick)
            if node in seen or node in guarded:
                continue

            seen.add(node)
            bag.append(ActionOp(budget_op, node))
        else:
            source, target = int(pick[0]), int(pick[1])
            if (source, target) in seen:
                continue

            seen.add((source, target))
            bag.append(
                ActionOp(budget_op, source, target, float(contact_reduction))
                if budget_op == "set_edge_weight"
                else ActionOp(budget_op, source, target)
            )

    if budget_op == "remove_node" and len(bag) < budget:
        for node in sorted(range(graph.num_nodes), key=graph.degree, reverse=True):
            if len(bag) >= budget:
                break

            if node not in seen and node not in guarded:
                seen.add(node)
                bag.append(ActionOp(budget_op, int(node)))

    return [bag] + [[] for _ in range(horizon)]


def dose_set(actions: list[list[ActionOp]], lever: str) -> list:
    """
    What a finished trajectory actually spent its budget on, in the order it spent it.

    Read back from the executed bags rather than re-planning, for the same reason
    `--compare` replays them: a randomized allocation returns a different set on a
    second call, and every reported number has to describe the set that earned the
    reward.
    """
    budget_op, _ = resolve_lever(lever)
    ordered = []
    seen = set()

    for bag in actions:
        for action in bag:
            if action.op != budget_op:
                continue

            key = (
                int(action.target)
                if action.destination is None
                else (int(action.target), int(action.destination))
            )
            if key in seen:
                continue

            seen.add(key)
            ordered.append(key)

    return ordered


def epidemic_metrics(
    reward: float,
    unprotected: float,
    task: TaskSpec,
    graph: GraphInfo,
    actions: list[list[ActionOp]],
    prevalence: list[float] | None = None,
    burn_in: float = default_burn_in,
) -> dict:
    """
    The block an immunization table reports (§8.3), in one place.

    Four groups, and the order is the argument this task makes:

      1. **The attack rate and prevented infections**: the objective, directly, on
         the same evaluator that produced `reward`. A ratio of two different rulers
         means nothing, which is why `unprotected` is measured on this arm's own
         evaluator and `--compare` re-measures both on the shared referee.
      2. **The outbreak SHAPE**: peak prevalence, time to peak, AUC, endemic
         prevalence. §8.2 trap 4 is the reason these are not optional: a good policy
         flattens rather than eliminates, so a terminal-state number alone can rank
         two policies backwards.
      3. **The eigendrop**: `lambda_1(A) - lambda_1(A - S)`, which is what the
         spectral line optimizes and therefore the only column our table and theirs
         share. Reported as CONTEXT beside the simulated number, never as the score:
         §8.2 trap 1 records that a method can win here and lose on final size, and
         grading ourselves on the surrogate the classical methods were built for is
         a comparison we cannot win and that does not test the world model.
      4. **What was spent**, so an under-spent budget stays visible.

    The eigendrop is only defined for a NODE lever; under `edge_cut` and
    `contact_reduce` the doses are arcs and `immunization_metrics` would be handed
    something it cannot delete, so those rows carry the arc list and no spectral
    column rather than a wrong one.
    """
    lever = task.epi_lever
    spent = dose_set(actions, lever)
    prevented = float(unprotected - reward)

    metrics = {
        "lever": lever,
        "lever_papers": lever_papers[lever],
        "lever_unit": lever_shape[lever],
        "outbreak": [int(node) for node in task.outbreak],
        "n_outbreak": len(task.outbreak),
        "attack_rate": float(reward),
        "attack_rate_pct": 100.0 * reward / max(graph.num_nodes, 1),
        "unprotected_attack_rate": float(unprotected),
        "prevented_infections": prevented,
        "prevented_pct_of_unprotected": (
            100.0 * prevented / unprotected if unprotected else 0.0
        ),
        "prevented_pct_of_nodes": 100.0 * prevented / max(graph.num_nodes, 1),
        "spent": [list(entry) if isinstance(entry, tuple) else entry for entry in spent],
        "n_spent": len(spent),
    }

    if prevalence:
        metrics |= epidemic_curve_metrics(prevalence, graph.num_nodes, burn_in)

    if lever_shape[lever] == "node" and spent:
        metrics["spectral"] = immunization_metrics(
            graph.edge_index, graph.num_nodes, [int(node) for node in spent]
        )

    return metrics


def unprotected_reference(
    environment: object, task: TaskSpec, graph: GraphInfo, horizon: int, budget: int
) -> tuple[float, list[float] | None]:
    """
    `(|R(inf)|, prevalence curve)` with nobody dosed, on this arm's own evaluator.

    One rollout, charged to the arm like every other, because computing it on a
    private simulator would make prevented infections a comparison between two
    different measurement processes. It doubles as the floor any allocation has to
    beat and as the denominator §8.3 reports against.

    Warning: THE EMPTY PLAN IS WRAPPED, and it has to be. Influence blocking's
    counterpart can hand the environment a bare `lambda: []` because `S_N` is
    committed by the simulator's own `reset`; here the outbreak is INJECTED by the
    wrapper as `add_node` index cases at t=0, so an unwrapped empty plan seeds
    nothing, the epidemic never starts, and the reference comes back 0, which
    silently turns every prevented-infections number negative.
    """
    immunization = build_immunization(graph, task)
    action_fn = (
        immunization.wrap(lambda state, timestep: [])
        if immunization is not None
        else (lambda state, timestep: [])
    )
    trajectory = environment.rollout(action_fn, horizon, budget)

    return float(trajectory.reward), trajectory.prevalence_curve
