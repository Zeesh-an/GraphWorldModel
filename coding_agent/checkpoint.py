"""
Mid-search state for a refinement loop, so a killed sweep resumes where it
stopped instead of paying for every LLM call again.

Written after each evaluation and deleted on success: a checkpoint on disk
means "this arm did not finish". Everything needed to continue is JSON: the
population, the conversation thread, the counters, and the anchor rollouts (so a
resume does not re-pay for the baseline leaderboard).

A checkpoint is only reused when its fingerprint matches the run about to start.
Resuming a search under a different budget, model, or evaluator would silently
mix two experiments, which is worse than losing the work.
"""

import json
from pathlib import Path

from coding_agent.types import ActionOp, State, Trajectory


def fingerprint(config: object, method: str, graph: object) -> dict:
    """Everything that must match for a resume to be the same experiment."""
    return {
        "method": method,
        "strategy_mode": config.strategy_mode,
        "evaluator": config.evaluator,
        "model": config.model,
        "temperature": config.temperature,
        "diffusion_model": config.diffusion_model,
        "budget": config.budget,
        "horizon": config.horizon,
        "outer_iters": config.outer_iters,
        "allowed_ops": list(config.allowed_ops),
        "allow_mc_algorithms": config.allow_mc_algorithms,
        "mc_runs": config.mc_runs,
        "n_samples": config.n_samples,
        "seed": config.seed,
        "num_nodes": graph.num_nodes,
        "num_edges": int(graph.edge_index.shape[1]),
        # Source localization. Every one of these changes WHAT THE REWARD MEANS,
        # so a resume across any of them would carry a population whose recorded
        # scores were measured on a different problem: a different episode pool, a
        # different observation, a different k, or (for `native_arm`) with or
        # without a forward model in the search loop at all. `native_arm` is not
        # recoverable from `evaluator`, which reads monte_carlo for both.
        "native_arm": getattr(config, "native_arm", False),
        "sl_select_split": getattr(config, "sl_select_split", None),
        "sl_instances": getattr(config, "sl_instances", None),
        "sl_observation": getattr(config, "sl_observation", None),
        "sl_budget_mode": getattr(config, "sl_budget_mode", None),
        "sl_source_tolerance": getattr(config, "sl_source_tolerance", None),
        # Everything else that changes what the reward MEASURES: the task and
        # its outbreak, the lever, the compartmental rates, the round schedule,
        # the stream, the union, the decoder and forecast pools, and which
        # checkpoint the evaluator is. None of these are in the results path.
        **{
            key: getattr(config, key, None)
            for key in (
                "task",
                "remove_semantics",
                "outbreak_pct",
                "outbreak_selector",
                "blocking_lever",
                "tie_break",
                "positive_prob",
                "detection_delay",
                "epi_lever",
                "epi_beta",
                "epi_gamma",
                "epi_alpha",
                "contact_reduction",
                "rounds",
                "per_round_budget",
                "round_gap",
                "feedback_model",
                "edit_rate",
                "campaigns",
                "cr_setting",
                "cr_observation_rate",
                "cr_hidden_rate",
                "cr_instances",
                "cr_select_split",
                "cr_tree_weight",
                "cp_select_split",
                "cp_instances",
                "cp_observation_steps",
                "cp_metric",
                "cp_target",
                "cp_forecast_samples",
                "wm_results_json",
                "credit",
            )
        },
    }


# The State fields that survive a checkpoint; `sample` is never serialized
state_fields = (
    "infected",
    "frontier",
    "pos_infected",
    "pos_frontier",
    "exposed",
    "recovered",
)


def trajectory_to_dict(trajectory: Trajectory) -> dict:
    return {
        "states": [
            {key: list(getattr(state, key)) for key in state_fields}
            for state in trajectory.states
        ],
        "actions": [
            [action.to_dict() for action in bag] for bag in trajectory.actions
        ],
        "reward": trajectory.reward,
        "infected_counts": trajectory.infected_counts,
        "cost": trajectory.cost,
        "final_marginals": trajectory.final_marginals,
        "spread_curve": trajectory.spread_curve,
        "prevalence_curve": trajectory.prevalence_curve,
    }


def trajectory_from_dict(record: dict) -> Trajectory:
    # `.get` on the newer keys so a checkpoint written before they existed still
    # resumes, with those fields simply absent as they were then
    return Trajectory(
        states=[
            State(**{key: state.get(key, []) for key in state_fields})
            for state in record["states"]
        ],
        actions=[
            [ActionOp(**action) for action in bag] for bag in record["actions"]
        ],
        reward=record["reward"],
        infected_counts=record["infected_counts"],
        cost=record["cost"],
        final_marginals=record["final_marginals"],
        spread_curve=record.get("spread_curve"),
        prevalence_curve=record.get("prevalence_curve"),
    )


def save(path: str | Path | None, payload: dict) -> None:
    if path is None:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write-then-rename: a kill mid-write must not leave a truncated file that
    # the next resume would reject (or worse, half-read)
    staging = path.with_suffix(".tmp")
    staging.write_text(json.dumps(payload, default=str))
    staging.replace(path)


def load(path: str | Path | None, expected: dict) -> dict | None:
    """Return the checkpoint, or None if it is missing, stale, or unreadable."""
    if path is None or not Path(path).exists():
        return None

    try:
        payload = json.loads(Path(path).read_text())
    except (json.JSONDecodeError, OSError) as error:
        print(f"[checkpoint] ignoring unreadable checkpoint {path}: {error}")
        return None

    if payload.get("fingerprint") != expected:
        differing = sorted(
            key
            for key in set(expected) | set(payload.get("fingerprint") or {})
            if (payload.get("fingerprint") or {}).get(key) != expected.get(key)
        )
        print(
            f"[checkpoint] ignoring checkpoint {path}: it was written for a "
            f"different run (differs on {differing})"
        )
        return None

    return payload


def clear(path: str | Path | None) -> None:
    if path is None:
        return

    Path(path).unlink(missing_ok=True)
    Path(path).with_suffix(".tmp").unlink(missing_ok=True)
