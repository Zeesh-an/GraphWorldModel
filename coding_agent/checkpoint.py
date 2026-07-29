"""
Mid-search state for a refinement loop, so a killed sweep resumes where it
stopped instead of paying for every LLM call again.

Written after each evaluation and deleted on success — a checkpoint on disk
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
    }


def trajectory_to_dict(trajectory: Trajectory) -> dict:
    return {
        "states": [
            {"infected": list(state.infected), "frontier": list(state.frontier)}
            for state in trajectory.states
        ],
        "actions": [
            [action.to_dict() for action in bag] for bag in trajectory.actions
        ],
        "reward": trajectory.reward,
        "infected_counts": trajectory.infected_counts,
        "cost": trajectory.cost,
        "final_marginals": trajectory.final_marginals,
    }


def trajectory_from_dict(record: dict) -> Trajectory:
    return Trajectory(
        states=[
            State(infected=state["infected"], frontier=state["frontier"])
            for state in record["states"]
        ],
        actions=[
            [ActionOp(**action) for action in bag] for bag in record["actions"]
        ],
        reward=record["reward"],
        infected_counts=record["infected_counts"],
        cost=record["cost"],
        final_marginals=record["final_marginals"],
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
