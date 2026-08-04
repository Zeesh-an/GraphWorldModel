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
    # Budget sweep this task defaults to, as percentages of N. None = the
    # pipeline's 1/5/10/20 ladder. A task overrides it when its k is a property of
    # the INSTANCE rather than a choice: source localization is handed the source
    # count it has to recover, so a four-point budget sweep would be four runs of
    # the same experiment.
    default_budget_pcts: tuple | None = None
    # Fraction of N the exogenous outbreak seeds, for a task whose cascade the
    # planner does not start. 0 = the planner seeds it (every maximize task).
    outbreak_pct: float = 0.0
    blocker: str | None = None

    @property
    def research_doc(self) -> str:
        return f"research/{self.name}.md"

    @property
    def runnable(self) -> bool:
        return self.status == implemented

    # `is None` rather than `or`: an EMPTY override is meaningful. Source
    # localization generates diffusion-only episodes on purpose (no injected
    # actions at all, so the transition degenerates to f(G, s_t) -> s_{t+1}), and
    # `or` would read that empty tuple as "unset" and inject node ops anyway.
    @property
    def allowed_ops(self) -> tuple:
        return self.action_ops if self.default_allowed_ops is None else self.default_allowed_ops

    @property
    def gen_action_ops(self) -> tuple:
        return (
            self.action_ops
            if self.default_gen_action_ops is None
            else self.default_gen_action_ops
        )

    @property
    def contains(self) -> bool:
        """
        True when the planner fights a cascade it did not start.

        The one structural difference between this task family and the seeding
        one: an exogenous outbreak has to be injected, the budget buys removals
        rather than seeds, and every "is this better" comparison flips sign.
        """
        return self.objective == minimize

    @property
    def recovers(self) -> bool:
        """
        True when the program INVERTS the transition instead of steering it.

        The structural difference is bigger than the containment one: there is no
        rollout, no action bag and no intervention at all. The program is handed a
        graph and an observed diffusion state and returns the seed set that
        produced it, scored on F1 against ground truth. The world model stops being
        the thing being optimized against and becomes a subroutine the generated
        program calls (research/source_localization.md §2.5).
        """
        return self.objective == recover


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
        status=implemented,
        objective=recover,
        dynamics=("IC", "LT"),
        # The recovered source set IS an `add_node` bag: the environment converts
        # x_hat into exactly the seed commit the generator writes at t=0 for any
        # re-simulation it needs (research/source_localization.md §2.4.1). Nothing
        # is ever emitted as an intervention — a localize() program returns node
        # ids — but naming the op keeps `predict_marginals` and the --compare
        # referee speaking the same action vocabulary as every other task.
        action_ops=("add_node",),
        summary="Recover the seed set s_0 from an observed diffusion state s_T.",
        # k is a property of the INSTANCE here, not a choice: the localizer is told
        # how many sources to name. A four-point budget sweep would be four runs of
        # one experiment, so the default is SL-VAE's single 10%-of-N convention.
        # Pass --budget-pcts 5 10 20 with --sl-budget sweep for §8.5.1's
        # source-fraction axis.
        default_budget_pcts=(10.0,),
        # Diffusion-only episodes: no injected actions at any t > 0, so the
        # transition degenerates to f(G, s_t) -> s_{t+1} and every episode's whole
        # observable history is caused by its t=0 seed commit alone. That is what
        # makes the (x, y) label pair well defined — an episode with a mid-cascade
        # injection has an observation its seed set did not produce (§2.1).
        default_gen_action_ops=(),
        # §2.6 arm 1, one representative per family of §3, ordered by how dangerous
        # each is. `lpsi` heads the list because it is the row that actually has to
        # be beaten: SIDSL's Table 1 puts a 2017 label-propagation method with no
        # learning at F1 0.544 on Digg against SL-VAE's 0.479 and DDMSL's 0.517
        # [verified, §5.5], and LPSI sits INSIDE the agent's expressible space, so
        # "the search rediscovers LPSI" is the realistic floor (§2.9 risk 1).
        # `rumor_centrality` is single-source and scores near zero under a
        # multi-source protocol by construction (§8.2) — it is here because it
        # founded the field and because a --budgets 1 run makes it admissible.
        # `resim_greedy` is deliberately absent for the same reason `celf` and
        # `greedy_blocking` are: it re-simulates every candidate on its own private
        # simulator, which would dominate startup for a table that exists to set a
        # bar. Run it as its own --baselines arm when you want its number.
        default_baselines=(
            "lpsi",
            "netsleuth",
            "ojc",
            "jordan_center",
            "dmp_localize",
            "dynamic_age",
            "effective_distance",
            "infected_degree",
            "rumor_centrality",
            "random_sources",
        ),
        # The six-condition ladder plus arm A (§2.6): §2.2's framing, in which the
        # world model is FROZEN and a relaxed source vector is gradient-descended
        # against it. That is SL-VAE with our likelihood plugged in, which the seed
        # paper has already declared a no-op component swap — so it is the CONTROL
        # program search is measured against, not the method. 6 vs A is the
        # methodological claim this task exists to make.
        default_arms=(
            "routing",
            "evolve_free@native",
            "evolve_free@monte_carlo",
            "evolve_free@oracle",
            "evolve_free@world_model",
            "gradient_free@world_model",
        ),
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

    # An inverse task labels each episode with its t=0 seed commit, so an episode
    # carrying a mid-cascade injection has an observation its seed set did not
    # produce and its (x, y) pair is a lie (research/source_localization.md §2.1)
    if _task.runnable and _task.recovers and _task.gen_action_ops:
        raise ValueError(
            f"task {_task.name!r} inverts the transition but generates with "
            f"action ops {_task.gen_action_ops}; a recover task needs "
            f"default_gen_action_ops=() so every episode's observation is caused "
            f"by its t=0 seed set alone"
        )

    # ...and it cannot also fight an exogenous cascade: the sources ARE the unknown
    if _task.recovers and _task.outbreak_pct:
        raise ValueError(
            f"task {_task.name!r} inverts the transition but declares an outbreak; "
            f"the source set is what it is solving for, not something handed to it"
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
