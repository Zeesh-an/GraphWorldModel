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


def _planned_action(plan: list[list[Action]], state: State, t: int) -> list[Action]:
    return plan[t] if t < len(plan) else []


def counterfactual_credit(
    env: object,
    plan: list[list[Action]],
    horizon: int,
    budget: int,
    seed: int = 0,
) -> tuple[float, list[dict]]:
    """Return (base_reward, one credit entry per action in the plan)."""
    base = env.rollout(
        partial(_planned_action, plan), horizon, budget, seed=seed
    ).reward

    entries = []
    for t, bag in enumerate(plan[: horizon + 1]):
        for i, action in enumerate(bag):
            # Same plan minus exactly this one action (bags shallow-copied so the
            # original plan is untouched).
            ablated = [list(b) for b in plan]
            del ablated[t][i]
            r = env.rollout(
                partial(_planned_action, ablated), horizon, budget, seed=seed
            ).reward

            entry = {
                "t": int(t),
                "op": action.op,
                "target": int(action.target),
                "delta": round(base - r, 3),
            }
            if action.destination is not None:
                entry["destination"] = int(action.destination)
            entries.append(entry)

    return base, entries


def format_credit_report(base_reward: float, entries: list[dict]) -> str:
    if not entries:
        return "No actions in the plan to credit."

    lines = [
        f"Per-action counterfactual credit (base spread {base_reward:.2f}; "
        "delta = spread lost if that single action is removed; ~0 = wasted budget):"
    ]
    for e in entries:
        dest = f"->{e['destination']}" if "destination" in e else ""
        lines.append(
            f"  t={e['t']}: {e['op']}({e['target']}{dest})  delta={e['delta']:+.2f}"
        )

    return "\n".join(lines)
