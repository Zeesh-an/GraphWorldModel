"""
Which graph tasks exist, what each one needs, and which are runnable today.

`--task` validates against this registry, and `Layout` uses the name as the first
level of the results tree, so `results/<task>/<dataset>/<run>/` and
`research/<task>.md` always agree on spelling.

Every planned entry names the one thing that blocks it. "What would it take to
add X" is answered here; why it is worth adding is answered in `research/<X>.md`.
"""

from dataclasses import dataclass

from data.wm_simulator import blocked, spent, valid_action_ops, valid_remove_semantics

implemented = "implemented"
planned = "planned"
out_of_scope = "out_of_scope"

# Objective sense the planner optimizes. `recover` tasks invert the forward model
# instead of choosing an intervention, so they have no action ops.
maximize = "maximize"
minimize = "minimize"
recover = "recover"
forecast = "forecast"

default_run = "default"


@dataclass()
class Task:
    name: str
    title: str
    status: str
    objective: str | None
    dynamics: tuple
    action_ops: tuple
    summary: str
    # What remove_node means for this task. `spent` (the node is a used-up
    # spreader, stays counted) is right for maximization; every containment task
    # needs `blocked` (the node is deleted from the graph), because `spent`
    # counts each immunized node as infected and biases the spread by +k.
    remove_semantics: str = spent
    # Arms this task sweeps by default. None = pipeline.conditions.default_arms
    # (the six-condition ladder). A task overrides this only when its question
    # needs a second method held alongside the ladder, the way adaptive IM needs
    # a matched non-adaptive control to divide by.
    default_arms: tuple | None = None
    # Condition-1 pool. None = pipeline.conditions.default_baselines (the static
    # IM classics). A task whose published baselines are a different KIND of
    # algorithm overrides it: adaptive IM's are per-round policies, and running
    # only static ones would leave its own literature off the table.
    default_baselines: tuple | None = None
    # Which ops a generated strategy may EMIT, and which the data generator
    # injects. Both default to `action_ops`, which is right whenever the task's
    # intervention vocabulary is the same at planning time and generation time.
    # CND narrows the planner to `remove_node` (edge removals ride along inside
    # the deletion bag, and are not the planner's to spend budget on).
    default_allowed_ops: tuple | None = None
    default_gen_action_ops: tuple | None = None
    # The op one unit of budget buys. `add_node` for a seeding task, `remove_node`
    # for a containment one; the executor counts it and the prompt states it.
    budget_op: str = "add_node"
    # Fraction of N the exogenous outbreak seeds, for a task whose cascade the
    # planner does not start. 0 = the planner seeds it (every maximize task).
    outbreak_pct: float = 0.0
    # World-family discriminators, ported from the 8-task branch. They decide the
    # STATE LAYOUT, which is what makes a checkpoint loadable or not:
    #
    #   competitive  two campaigns, 8 input / 4 output channels
    #   epidemic     S/E/I/R compartments, 9 input / 5 output channels
    #   neither      the single-cascade IC/LT world, 6 input / 2 output
    #
    # `reconstructs` is not a layout flag: the task recovers a hidden trajectory
    # rather than choosing an intervention, which changes the READOUT, not the
    # state. Kept separate for exactly that reason.
    competitive: bool = False
    epidemic: bool = False
    reconstructs: bool = False
    blocker: str | None = None

    @property
    def research_doc(self) -> str:
        return f"research/{self.name}.md"

    @property
    def runnable(self) -> bool:
        return self.status == implemented

    @property
    def allowed_ops(self) -> tuple:
        # `is None` rather than a falsy test: an EMPTY tuple is a real value
        # meaning "this task emits no interventions", and it is exactly what the
        # inverse and forecasting tasks declare. Falling back to `action_ops`
        # there would give source localization a planner action vocabulary it
        # never exercises, which is the difference between FORWARD_DYNAMICS and
        # EXACT_CHECKPOINT in registry.task_families.
        if self.default_allowed_ops is None:
            return self.action_ops

        return self.default_allowed_ops

    @property
    def gen_action_ops(self) -> tuple:
        if self.default_gen_action_ops is None:
            return self.action_ops

        return self.default_gen_action_ops

    @property
    def contains(self) -> bool:
        """
        True when the planner fights a cascade it did not start.

        The one structural difference between this task family and the seeding
        one: an exogenous outbreak has to be injected, the budget buys removals
        rather than seeds, and every "is this better" comparison flips sign.
        """
        return self.objective == minimize


tasks = {
    "influence_maximization": Task(
        name="influence_maximization",
        title="Influence Maximization",
        status=implemented,
        objective=maximize,
        # The INTERVENTION this task is defined by. The two `default_*_ops` below
        # are the wider set the pipeline has always shipped, and they are pinned
        # rather than derived: `remove_node` under `spent` is a legal (if almost
        # always bad) move, and the model is trained on both node ops so the
        # action channels are exercised. Every checkpoint in `results/` was
        # produced under these, so narrowing them to `action_ops` would silently
        # change what a rerun means.
        dynamics=("IC", "LT"),
        action_ops=("add_node",),
        default_allowed_ops=("add_node", "remove_node"),
        default_gen_action_ops=("add_node", "remove_node"),
        summary="Choose k seeds maximizing expected spread sigma(S).",
    ),
    "influence_blocking": Task(
        name="influence_blocking",
        title="Influence Blocking",
        status=planned,
        objective=minimize,
        dynamics=("IC", "LT"),
        action_ops=valid_action_ops,
        remove_semantics=blocked,
        summary="Two competing cascades; place blockers to minimize the negative one.",
        blocker="NDlib ships no competitive model (`Blocked: -1` is a static "
        "non-adopter set, and CompositeModel expresses only one global "
        "tie-break). Needs a CompetitiveSimulator, an 8-channel state, and a "
        "CompetitiveICHead. Phase 0 (single-cascade node-blocking IMIN) needs "
        "none of that and is runnable once the objective sign flips.",
    ),
    "critical_node_detection": Task(
        name="critical_node_detection",
        title="Critical Node Detection",
        status=implemented,
        objective=minimize,
        dynamics=("IC", "LT"),
        action_ops=("remove_node", "remove_edge"),
        remove_semantics=blocked,
        summary="Remove k nodes to minimize the eventual spread of an outbreak.",
        # The DIFFUSION variant (research/critical_node_detection.md §2.2), not
        # the structural one. §2.1 argues at length that structural CNDP is a bad
        # fit for a world model — its T_endo is nothing, and its ground truth
        # (`nx.connected_components`, O(N+E)) is cheaper than one forward pass, so
        # a learned surrogate has nothing to amortize. The connectivity
        # functionals are still computed and reported per arm as descriptive
        # context (§8.3), just never as the learned target.
        #
        # A planner emits `remove_node` only. Edge removals are the deletion bag
        # `containment.expand_removals` builds around each one, so charging the
        # planner for them would make k mean deg(v) different things per node.
        default_allowed_ops=("remove_node",),
        default_gen_action_ops=("remove_node",),
        budget_op="remove_node",
        # The cascade is exogenous: 1% of N is the standard immunization setup
        # and matches the smallest point of our own --budget-pcts sweep, so the
        # k=1% row is "one blocker per source"
        outbreak_pct=1.0,
        # §8.3 and §9.3, in ascending order of danger. `adaptive_degree` is the
        # one that hurts: MIND's Table 5 puts plain HDA at 119.9 against FINDER's
        # 115.0, so a learned dismantler that does not clearly beat it has
        # demonstrated nothing. `iterative_betweenness` is Wandelt's best-in-70-80%
        # method that almost no learned paper reports, and omitting it would
        # reproduce the exact methodological gap that survey calls out.
        # One representative per family, and the reinserting variant wherever the
        # published method HAS one — §8.2 trap 2 is that `X` and `X+R` are cited
        # under one name and are not the same method. `bpd` and `min_sum` are the
        # message-passing pair; `corehd` and `decycling` are the greedy versions of
        # the same two stages, so the four together isolate what inference buys.
        default_baselines=(
            "adaptive_degree",
            "iterative_betweenness",
            "collective_influence_r",
            "corehd",
            "bpd_r",
            "decycling",
            "explosive_immunization",
            "gnd",
            "netshield",
            "degree_removal",
            "random_removal",
        ),
        blocker=None,
    ),
    "epidemic_control": Task(
        name="epidemic_control",
        title="Epidemic Control",
        status=planned,
        objective=minimize,
        dynamics=("SIR", "SIS", "SEIR"),
        action_ops=("remove_node", "remove_edge", "set_edge_weight"),
        # Vaccination: immune, non-infectious, never counted in the outbreak.
        # Note this collides with SIR's own Removed compartment, where a
        # RECOVERED node must stay counted, so the compartment head has to carry
        # that distinction, `blocked` only covers the intervention.
        remove_semantics=blocked,
        summary="Vaccinate, quarantine, or reduce contact to minimize an outbreak.",
        blocker="NDlib already ships SIR/SIS/SEIR, so the simulator is ~60 "
        "lines — but ICTransmissionHead composes "
        "`y_inf = infected + (1-infected) * p_new`, which is monotone by "
        "construction and cannot represent recovery or re-infection.",
    ),
    "source_localization": Task(
        name="source_localization",
        title="Source Localization",
        status=planned,
        objective=recover,
        dynamics=("IC", "LT"),
        action_ops=(),
        summary="Recover the seed set s_0 from an observed final state s_T.",
        blocker=None,
    ),
    "influence_estimation": Task(
        name="influence_estimation",
        title="Influence Spread Estimation",
        status=planned,
        objective=forecast,
        dynamics=("IC", "LT"),
        action_ops=(),
        summary="Predict sigma(S) without Monte Carlo.",
        blocker=None,
    ),
    "cascade_reconstruction": Task(
        name="cascade_reconstruction",
        title="Cascade Reconstruction",
        status=planned,
        objective=recover,
        dynamics=("IC", "LT"),
        action_ops=(),
        summary="Recover the hidden trajectory from partial observations.",
        blocker="Node-level reconstruction needs only a masked-episode harness. "
        "Tree-level metrics need the transmission edge, which NDlib never "
        "emits — IndependentCascadesModel.iteration flips v without recording "
        "which u caused it.",
    ),
    "adaptive_online_im": Task(
        name="adaptive_online_im",
        title="Adaptive and Online Influence Maximization",
        status=implemented,
        objective=maximize,
        dynamics=("IC", "LT"),
        action_ops=("add_node",),
        # Same reasoning as influence_maximization: the pipeline's shipped pair,
        # pinned so the checkpoints already on disk keep meaning what they meant
        default_allowed_ops=("add_node", "remove_node"),
        default_gen_action_ops=("add_node", "remove_node"),
        summary="Choose seeds over rounds, observing realized activations between them.",
        # The published adaptive algorithms (AdaptGreedy, EPIC) plus the per-round
        # heuristic floor and `static_split`, the same-machinery control that
        # isolates the timing penalty from the adaptivity benefit. `celf_pp` and
        # `imm` ride along as the strongest STATIC seed sets, which is what the
        # adaptivity gap divides by (research/adaptive_online_im.md §8.1).
        default_baselines=(
            "adapt_greedy",
            "adapt_epic",
            "adapt_degree_discount",
            "adapt_random",
            "static_split",
            "celf_pp",
            "imm",
        ),
        # Every adaptive arm needs the non-adaptive arm at the same k to divide
        # by: the adaptivity gap is a RATIO, and a multi-round spread number on
        # its own says nothing (research/adaptive_online_im.md 5.1, 9.3 item 3).
        # Same evaluator on both sides of each pair, so the gap is not confounded
        # by evaluator fidelity.
        default_arms=(
            "routing",
            "evolve_free@native",
            "adaptive_free@native",
            "evolve_free@monte_carlo",
            "adaptive_free@monte_carlo",
            "evolve_free@oracle",
            "adaptive_free@oracle",
            "evolve_free@world_model",
            "adaptive_free@world_model",
        ),
        blocker=None,
    ),
    "network_inference": Task(
        name="network_inference",
        title="Diffusion Network Inference",
        status=planned,
        objective=recover,
        dynamics=("IC",),
        action_ops=("add_edge", "remove_edge", "set_edge_weight"),
        summary="Recover the latent influence network from cascade traces alone.",
        blocker="Edge weights are already differentiable via the "
        "structured_residual head, but edge *existence* is not: edge_index is "
        "integer, so recovery needs a dense N x N scorer (feasible only at "
        "N <= ~2K) and a survival likelihood incompatible with the "
        "teacher-forced loop.",
    ),
    "cascade_prediction": Task(
        name="cascade_prediction",
        title="Cascade / Popularity Prediction",
        status=planned,
        objective=forecast,
        dynamics=("IC", "LT"),
        action_ops=(),
        summary="Predict a real cascade's final size from its early window.",
        blocker="Needs real observed cascade corpora (Weibo, Twitter, APS), "
        "which no loader emits — our transitions come from NDlib.",
    ),
    "cascading_failure": Task(
        name="cascading_failure",
        title="Cascading Failure in Infrastructure Networks",
        status=planned,
        objective=minimize,
        dynamics=("motter_lai", "dc_power_flow"),
        action_ops=("remove_node", "remove_edge", "add_edge", "set_edge_weight"),
        # A bus outage takes the substation and its lines out of the network
        remove_semantics=blocked,
        summary="Line trips redistribute load and trigger further failures.",
        blocker="T_endo is global load redistribution, not local edge-wise "
        "propagation, so a k-hop encoder structurally cannot see the next "
        "failure. Motter-Lai (betweenness load + tolerance alpha) is the "
        "entry point that needs no electrical data.",
    ),
    "graph_completion": Task(
        name="graph_completion",
        title="Graph Completion under Incompleteness",
        status=out_of_scope,
        objective=None,
        dynamics=(),
        action_ops=(),
        summary="Impute missing node features or edges.",
        blocker="No state evolves and none of the five ops apply — `add_edge` "
        "here means infer, not intervene. Our model never reads node features "
        "at all. Useful only as a robustness condition; see the research doc.",
    ),
    "temporal_forecasting": Task(
        name="temporal_forecasting",
        title="Traffic and Temporal Graph Forecasting",
        status=out_of_scope,
        objective=None,
        dynamics=(),
        action_ops=(),
        summary="Forecast continuous node signals over time on a fixed graph.",
        blocker="No interventions, so T_exo is unused and action-conditioning "
        "— the contribution — goes untested. No simulator, no counterfactual "
        "forks, no dataset overlap. Kept for its rollout-training techniques.",
    ),
}

# ---------------------------------------------------------------------------
# Task Family Extension
# ---------------------------------------------------------------------------
#
# Semantics ported from the 8-task branch, applied as an overlay rather than by
# rewriting the entries above. Two reasons:
#
#   * the overlay IS the delta. Reading it tells you exactly what the 8-task
#     branch settled that this file predated, with no diffing.
#   * `status` is deliberately NOT overlaid. A task is `implemented` here only
#     if THIS repository can build its head and step its dynamics; the 8-task
#     branch marks all eight implemented because it carries CompetitiveICHead,
#     CompetitiveLTHead and CompartmentTransitionHead, which have not been
#     ported. Claiming their status without their code would make
#     `runnable_task_names()` lie.
#
# `registry.task_families` derives transfer compatibility from these fields.
task_family_extension = {
    "source_localization": {
        # The recovered source set is replayed as an add_node seed bag, so the
        # action VOCABULARY is shared with IM -- but generation injects nothing
        # mid-cascade, which is what keeps this out of EXACT_CHECKPOINT.
        "action_ops": ("add_node",),
        "default_allowed_ops": ("add_node",),
        "default_gen_action_ops": (),
    },
    "cascade_reconstruction": {
        "action_ops": ("add_node",),
        "default_allowed_ops": ("add_node",),
        "default_gen_action_ops": (),
        "reconstructs": True,
    },
    "cascade_prediction": {
        "action_ops": (),
        "default_allowed_ops": (),
        "default_gen_action_ops": (),
    },
    "influence_blocking": {
        "action_ops": ("add_node", "remove_node", "remove_edge", "set_edge_weight"),
        "default_allowed_ops": ("add_node",),
        "default_gen_action_ops": (
            "add_node", "remove_node", "remove_edge", "set_edge_weight",
        ),
        "budget_op": "add_node",
        "outbreak_pct": 1.0,
        "competitive": True,
    },
    "epidemic_control": {
        "action_ops": ("remove_node", "remove_edge", "set_edge_weight"),
        "default_allowed_ops": ("remove_node",),
        "default_gen_action_ops": ("remove_node", "remove_edge", "set_edge_weight"),
        "budget_op": "remove_node",
        "outbreak_pct": 1.0,
        "epidemic": True,
    },
    "critical_node_detection": {
        "outbreak_pct": 1.0,
    },
}


def _apply_task_family_extension() -> None:
    for name, overrides in task_family_extension.items():
        task = tasks.get(name)

        if task is None:
            raise ValueError(
                f"task_family_extension names {name!r}, which is not registered"
            )

        for field, value in overrides.items():
            setattr(task, field, value)


_apply_task_family_extension()


# A typo in an action op above would silently produce a task whose ops the
# simulator rejects only at generation time
for _task in tasks.values():
    _unknown = set(_task.action_ops) - set(valid_action_ops)
    if _unknown:
        raise ValueError(
            f"task {_task.name!r} declares unknown action ops {sorted(_unknown)}; "
            f"valid ops are {sorted(valid_action_ops)}"
        )

    if _task.remove_semantics not in valid_remove_semantics:
        raise ValueError(
            f"task {_task.name!r} declares unknown remove_semantics "
            f"{_task.remove_semantics!r}; valid values are {valid_remove_semantics}"
        )

    # A budget op the planner may not emit is a task whose every candidate is
    # rejected at validation — cheaper to catch here than one budget in. Only the
    # runnable tasks are held to it: a planned entry's ops are documentation of
    # what it WOULD need, and `budget_op` is not settled until it ships.
    if _task.runnable and _task.budget_op not in _task.allowed_ops:
        raise ValueError(
            f"task {_task.name!r} spends budget on {_task.budget_op!r} but does "
            f"not allow the planner to emit it (allowed_ops={_task.allowed_ops})"
        )

    # `spent` counts an immunized node as infected, which biases any containment
    # objective by exactly +k (research/critical_node_detection.md §2.3)
    if _task.contains and _task.remove_semantics != blocked:
        raise ValueError(
            f"task {_task.name!r} minimizes spread but uses remove_semantics="
            f"{_task.remove_semantics!r}; a containment task needs {blocked!r}"
        )


def task_names() -> list[str]:
    return sorted(tasks)


def runnable_task_names() -> list[str]:
    return sorted(name for name, task in tasks.items() if task.runnable)


def get_task(name: str) -> Task:
    if name not in tasks:
        raise ValueError(f"unknown task {name!r}; choose one of {task_names()}")

    return tasks[name]


def require_runnable(name: str) -> Task:
    task = get_task(name)

    if not task.runnable:
        raise ValueError(
            f"task {name!r} is {task.status}, not runnable by this pipeline. "
            f"{task.blocker or ''} "
            f"See {task.research_doc} for the full analysis. "
            f"Runnable today: {runnable_task_names()}"
        )

    return task
