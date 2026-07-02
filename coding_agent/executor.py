"""
Execute a generated strategy script in a controlled namespace and return a
validated Strategy. All failures raise StrategyError carrying a message suitable
for feeding back to the agent as a repair turn.

NOTE: exec in a restricted namespace is a research scaffold, NOT a security
sandbox. Do not run untrusted scripts from outside the agent loop.
"""

import traceback

from coding_agent.types import Action, GraphInfo, State, Strategy, VALID_ACTION_OPS
from coding_agent.tools import algorithms, primitives


class StrategyError(RuntimeError):
    """A generated script failed to parse, execute, or expose a valid Strategy."""


def _namespace() -> dict:
    # The exact callable surface the prompts advertise.
    return {
        "Action": Action,
        "State": State,
        "GraphInfo": GraphInfo,
        "Strategy": Strategy,
        "algorithms": algorithms,
        "primitives": primitives,
        "__builtins__": __builtins__,
    }


def build_strategy(script: str) -> Strategy:
    ns = _namespace()
    try:
        compiled = compile(script, "<agent_strategy>", "exec")
        exec(compiled, ns)  # noqa: S102 — scaffold; see module docstring
    except SyntaxError as exc:
        raise StrategyError(f"SyntaxError in generated script: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 — surface any exec error to the agent
        raise StrategyError(
            f"Error executing generated script: {exc}\n{traceback.format_exc()}"
        ) from exc

    candidates = [
        obj
        for name, obj in ns.items()
        if isinstance(obj, type)
        and name != "Strategy"
        and (hasattr(obj, "plan_horizon") or hasattr(obj, "act"))
    ]
    if not candidates:
        raise StrategyError(
            "No Strategy subclass with plan_horizon()/act() found in the script."
        )
    strategy_cls = candidates[-1]
    try:
        return strategy_cls()  # type: ignore[call-arg]
    except Exception as exc:  # noqa: BLE001
        raise StrategyError(f"Could not instantiate Strategy: {exc}") from exc


def validate_actions(bag: list, num_nodes: int, budget: int) -> None:
    """Raise StrategyError if an action bag references invalid nodes, uses an unknown op, or exceeds budget."""
    n_add = 0
    for a in bag:
        if a.op not in VALID_ACTION_OPS:
            raise StrategyError(
                f"action op '{a.op}' is not valid; must be one of {VALID_ACTION_OPS}."
            )
        if not (0 <= int(a.target) < num_nodes):
            raise StrategyError(
                f"action targets node {a.target} out of range [0,{num_nodes})."
            )
        if a.op == "add_node":
            n_add += 1
    if n_add > budget:
        raise StrategyError(f"action bag adds {n_add} seeds, exceeds budget {budget}.")
