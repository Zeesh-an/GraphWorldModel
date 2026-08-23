"""
Which tasks can share which parts of a world model, derived rather than declared.

The question this answers: given a checkpoint trained on task S, what can task T
do with it? "They're both influence tasks" is not an answer — the thing that
decides whether a `state_dict` loads is the tensor layout, and the thing that
decides whether the numbers MEAN anything is the semantics behind it.

So nothing here is keyed on a task name. Every field is computed from properties
`pipeline.tasks` already owns:

    dynamics           IC/LT vs SIR/SIS/SEIR
    competitive        two campaigns -> 8 input / 4 output channels
    epidemic           compartments  -> 9 input / 5 output channels
    remove_semantics   spent vs blocked -- the T_exo branch inside the head
    action_ops         the intervention vocabulary
    gen_action_ops     what the DATA GENERATOR injects mid-cascade. This is the
                       field that separates "reuses the forward model" from
                       "reuses the action-conditioned model", and it is empty for
                       every inverse and forecasting task.

The four transfer levels are defined in `docs/task_family_generalization.md`.
Two things this module is careful NOT to do:

  * classify on tensor shape alone. Two tasks can share a 6/2 layout and still be
    MECHANISM_ONLY because their T_exo differs (IM's `spent` seeding vs CND's
    `blocked` removal). Shape is necessary, not sufficient.
  * confuse "the checkpoint loads" with "the task is runnable here". A task can
    be classified EXACT_CHECKPOINT and still be unrunnable in this repository
    because its head was never ported; `runnable` reports that separately.
"""

from dataclasses import asdict, dataclass, field
from itertools import permutations

from data.wm_simulator import blocked as blocked_semantics
from pipeline.tasks import Task, tasks

# ---------------------------------------------------------------------------
# World families -- the state/output layout a task's head must have
# ---------------------------------------------------------------------------

single_cascade = "W1_single_cascade"
competitive_cascade = "W2_competitive_cascade"
compartmental = "W3_compartmental"

#: family -> (input channels, output channels). The numbers are the layouts the
#: 8-task branch implements; they are recorded here because the compatibility
#: question is about them, and `world_model.wm_data` in this repository defines
#: only the W1 pair (the other two heads have not been ported).
family_layout = {
    single_cascade: (6, 2),
    competitive_cascade: (8, 4),
    compartmental: (9, 5),
}

# Transfer levels, strongest first.
exact_checkpoint = "EXACT_CHECKPOINT"
forward_dynamics = "FORWARD_DYNAMICS"
mechanism_only = "MECHANISM_ONLY"
incompatible = "INCOMPATIBLE"

transfer_levels = (
    exact_checkpoint,
    forward_dynamics,
    mechanism_only,
    incompatible,
)


def world_family(task: Task) -> str:
    """Which state/dynamics world a task lives in."""
    if task.epidemic:
        return compartmental

    if task.competitive:
        return competitive_cascade

    return single_cascade


def state_layout(task: Task) -> tuple[int, int]:
    return family_layout[world_family(task)]


def requires_actions(task: Task) -> bool:
    """
    Does the task's DATA carry mid-cascade interventions?

    Not `action_ops`, which is the vocabulary a solution may name. Source
    localization declares `add_node` because a recovered source set is replayed
    as a seed bag, but its generator injects nothing after t=0 — so a model
    transferred into it is never asked an action-conditioned question, and
    calling that action-conditioned transfer would overstate it.
    """
    return bool(task.gen_action_ops)


def observational(task: Task) -> bool:
    """No interventions at all: the task observes a process it cannot steer."""
    return not task.action_ops and not task.gen_action_ops


# ---------------------------------------------------------------------------
# Pairwise compatibility
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Compatibility:
    source: str
    target: str
    level: str
    reason: str
    same_state_layout: bool
    shared_dynamics: tuple
    same_remove_semantics: bool
    shared_action_ops: tuple
    requires_actions: bool
    competitive_compatible: bool
    epidemic_compatible: bool
    observational_shift: bool
    #: What a transfer run may reuse and what it must rebuild. Empty for
    #: INCOMPATIBLE. Stated per pair because MECHANISM_ONLY is meaningless
    #: without it.
    reusable_components: tuple = ()
    rebuilt_components: tuple = ()

    def to_dict(self) -> dict:
        return asdict(self)


def compatibility(source: Task, target: Task) -> Compatibility:
    """Classify what a source-trained checkpoint can do for a target task."""
    source_family, target_family = world_family(source), world_family(target)
    same_layout = state_layout(source) == state_layout(target)
    shared_dyn = tuple(sorted(set(source.dynamics) & set(target.dynamics)))
    same_remove = source.remove_semantics == target.remove_semantics
    shared_ops = tuple(sorted(set(source.action_ops) & set(target.action_ops)))
    target_needs_actions = requires_actions(target)
    obs_shift = observational(target) and not observational(source)

    common = {
        "source": source.name,
        "target": target.name,
        "same_state_layout": same_layout,
        "shared_dynamics": shared_dyn,
        "same_remove_semantics": same_remove,
        "shared_action_ops": shared_ops,
        "requires_actions": target_needs_actions,
        "competitive_compatible": source.competitive == target.competitive,
        "epidemic_compatible": source.epidemic == target.epidemic,
        "observational_shift": obs_shift,
    }

    # --- No shared dynamics ----------------------------------------------------
    # Nothing about the learned transition operator can carry over: it is a model
    # OF a process, and there is no shared process. But the ACTION semantics can
    # still be shared — two containment tasks both spend budget on `blocked`
    # removals to suppress a spreading process, whatever that process is. That is
    # the §7 containment overlay, and it is a genuinely weaker claim than
    # anything above, so it gets its own reason string rather than being folded
    # in silently.
    if not shared_dyn:
        both_contain = (
            source.remove_semantics == target.remove_semantics == blocked_semantics
            and source.objective == target.objective == "minimize"
        )

        if both_contain:
            return Compatibility(
                level=mechanism_only,
                reason=(
                    f"no shared dynamics ({list(source.dynamics)} vs "
                    f"{list(target.dynamics)}), so no learned transition transfers. "
                    f"Both are containment tasks spending budget on `blocked` "
                    f"removals, so what transfers is the ACTION SEMANTICS and the "
                    f"architecture — not any parameter fitted to a cascade."
                ),
                reusable_components=("action_semantics", "architecture"),
                rebuilt_components=("graph_backbone", "edge_propensity", "T_endo",
                                    "transition_head", "readout"),
                **common,
            )

        return Compatibility(
            level=incompatible,
            reason=(
                f"no shared dynamics: {source.name} runs "
                f"{list(source.dynamics)}, {target.name} runs "
                f"{list(target.dynamics)}, and they do not even share containment "
                f"action semantics. The learned transition operator is a model OF "
                f"a process; there is no shared process here."
            ),
            **common,
        )

    # --- Different world family: layout rules out loading the checkpoint --------
    if not same_layout:
        source_in, source_out = state_layout(source)
        target_in, target_out = state_layout(target)

        return Compatibility(
            level=mechanism_only,
            reason=(
                f"different world family ({source_family} -> {target_family}): "
                f"{source_in}/{source_out} channels vs {target_in}/{target_out}. "
                f"The state_dict cannot load. What can carry over is the "
                f"mechanism — the edge-propensity model and the graph backbone — "
                f"and any experiment must say which parameter groups it reused."
            ),
            reusable_components=("graph_backbone", "edge_propensity"),
            rebuilt_components=("input_projection", "transition_head", "readout"),
            **common,
        )

    # --- Same layout from here on ---------------------------------------------
    # T_exo is a branch INSIDE the head keyed on remove_semantics, so a spent
    # checkpoint evaluated under blocked applies the wrong deterministic
    # semantics with no shape error and no warning. Shape is not sufficient.
    if not same_remove:
        return Compatibility(
            level=mechanism_only,
            reason=(
                f"same {source_family} layout, but remove_semantics differs "
                f"({source.remove_semantics} -> {target.remove_semantics}). T_exo "
                f"is a branch inside the head on exactly this value, so the "
                f"checkpoint would load and apply the wrong deterministic "
                f"semantics silently. T_endo is what transfers."
            ),
            reusable_components=("graph_backbone", "edge_propensity", "T_endo"),
            rebuilt_components=("T_exo_semantics",),
            **common,
        )

    if not target_needs_actions:
        return Compatibility(
            level=forward_dynamics,
            reason=(
                f"{target.name} injects no mid-cascade interventions "
                f"(gen_action_ops is empty), so the frozen model is used as a "
                f"forward oracle and its action-conditioning is never exercised. "
                f"Frozen forward-dynamics reuse, not action-conditioned transfer."
                + (
                    " Target data is observational, adding a domain shift on top."
                    if obs_shift
                    else ""
                )
            ),
            reusable_components=("graph_backbone", "edge_propensity", "T_endo",
                                 "T_exo_semantics"),
            rebuilt_components=("readout", "search_procedure"),
            **common,
        )

    # Target does exercise interventions: the vocabularies must actually overlap.
    if not shared_ops:
        return Compatibility(
            level=mechanism_only,
            reason=(
                f"same layout and semantics, but no shared action ops "
                f"({list(source.action_ops)} vs {list(target.action_ops)}): the "
                f"transferred model was never asked the target's questions."
            ),
            reusable_components=("graph_backbone", "edge_propensity", "T_endo"),
            rebuilt_components=("action_encoder", "T_exo_semantics"),
            **common,
        )

    if set(target.action_ops) - set(source.action_ops):
        missing = sorted(set(target.action_ops) - set(source.action_ops))

        return Compatibility(
            level=mechanism_only,
            reason=(
                f"target emits {missing}, which the source never trained on. The "
                f"checkpoint loads, but those ops' action channels were never "
                f"exercised, so their effects are unmodelled rather than learned."
            ),
            reusable_components=("graph_backbone", "edge_propensity", "T_endo",
                                 "T_exo_semantics"),
            rebuilt_components=("action_encoder_coverage",),
            **common,
        )

    return Compatibility(
        level=exact_checkpoint,
        reason=(
            f"identical world: {source_family}, dynamics {list(shared_dyn)}, "
            f"remove_semantics {source.remove_semantics}, and the target's "
            f"action vocabulary {list(target.action_ops)} is covered by the "
            f"source's. The frozen checkpoint is usable with no override."
        ),
        reusable_components=("graph_backbone", "edge_propensity", "T_endo",
                             "T_exo_semantics", "prediction_head"),
        rebuilt_components=(),
        **common,
    )


def transfer_matrix(task_names: list[str] | None = None) -> list[Compatibility]:
    """Every ordered source -> target pair. Directional: transfer need not be symmetric."""
    names = sorted(task_names or [n for n, t in tasks.items() if t.objective])

    return [
        compatibility(tasks[source], tasks[target])
        for source, target in permutations(names, 2)
    ]


def families() -> dict[str, list[str]]:
    """World family -> the tasks living in it."""
    grouped: dict[str, list[str]] = {}

    for name, task in sorted(tasks.items()):
        if task.objective is None:
            continue

        grouped.setdefault(world_family(task), []).append(name)

    return grouped


def containment_overlay() -> list[str]:
    """
    Tasks that suppress an undesirable process, ACROSS world families.

    Deliberately non-exclusive and deliberately not a compatibility class: these
    share `blocked` removal semantics and a minimize objective while living in
    three different state layouts. The overlay exists so the shared idea is
    visible without anyone reading it as checkpoint compatibility.
    """
    return sorted(
        name
        for name, task in tasks.items()
        if task.objective == "minimize" and task.remove_semantics == "blocked"
    )
