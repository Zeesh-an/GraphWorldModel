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
    blocker: str | None = None

    @property
    def research_doc(self) -> str:
        return f"research/{self.name}.md"

    @property
    def runnable(self) -> bool:
        return self.status == implemented


tasks = {
    "influence_maximization": Task(
        name="influence_maximization",
        title="Influence Maximization",
        status=implemented,
        objective=maximize,
        dynamics=("IC", "LT"),
        action_ops=("add_node",),
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
        status=planned,
        objective=minimize,
        dynamics=("IC", "LT"),
        action_ops=("remove_node", "remove_edge"),
        remove_semantics=blocked,
        summary="Remove k nodes to minimize eventual spread or connectivity.",
        blocker="Needs a minimize-mode planner objective. `remove_semantics="
        "blocked` now deletes the node and its edges and stops counting it, so "
        "the +k spread bias is gone; what is left is the objective sign and the "
        "containment metrics (see research/critical_node_detection.md 9.2).",
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
        status=planned,
        objective=maximize,
        dynamics=("IC", "LT"),
        action_ops=("add_node",),
        summary="Choose seeds over rounds, observing realized activations between them.",
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
