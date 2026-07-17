"""
Counterfactual credit assignment: per-action reward attribution via ablation.

For each action in a per-timestep plan, re-roll the SAME environment with that
single action removed; delta = base_reward - ablated_reward is the final spread
that action is responsible for (~0 = wasted budget). All rollouts share one seed
so the comparison is paired. For state-dependent strategies (per_step / windowed)
the recorded trajectory bags are replayed as a fixed plan, so credit is an
approximation there — the live policy would have reacted to the ablated cascade.
"""

from functools import partial

from coding_agent.types import Action, State


def planned_action(
    plan: list[list[Action]], state: State, timestep: int
) -> list[Action]:
    """Dapts a static plan into an ActionFn."""
    return plan[timestep] if timestep < len(plan) else []


def counterfactual_credit(
    environment: object,
    plan: list[list[Action]],
    horizon: int,
    budget: int,
    seed: int = 0,
) -> tuple[float, list[dict]]:
    """Return (base_reward, one credit entry per action in the plan)."""
    # Base rollout with the full plan
    base_reward = environment.rollout(
        partial(planned_action, plan), horizon, budget, seed=seed
    ).reward

    entries = []
    for timestep, bag in enumerate(plan[: horizon + 1]):
        for action_index, action in enumerate(bag):
            # Same plan minus exactly this one action (bags shallow-copied so the original plan is untouched)
            ablated = [list(action_bag) for action_bag in plan]
            del ablated[timestep][action_index]

            # Run the rollout without the ablated action
            ablated_reward = environment.rollout(
                partial(planned_action, ablated), horizon, budget, seed=seed
            ).reward

            entry = {
                "t": int(timestep),
                "op": action.op,
                "target": int(action.target),
                "delta": round(base_reward - ablated_reward, 3),
            }

            if action.destination is not None:
                entry["destination"] = int(action.destination)

            entries.append(entry)

    return base_reward, entries


def format_credit_report(base_reward: float, entries: list[dict]) -> str:
    if not entries:
        return "No actions in the plan to credit."

    lines = [
        f"Per-action counterfactual credit (base spread {base_reward:.2f}; "
        "delta = spread lost if that single action is removed; ~0 = wasted budget):"
    ]
    for entry in entries:
        destination = f"->{entry['destination']}" if "destination" in entry else ""
        lines.append(
            f"  t={entry['t']}: {entry['op']}({entry['target']}{destination})  "
            f"delta={entry['delta']:+.2f}"
        )

    return "\n".join(lines)
