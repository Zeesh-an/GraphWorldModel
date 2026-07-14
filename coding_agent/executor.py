"""
Execute a generated strategy script in a controlled namespace and return a
validated Strategy. All failures raise StrategyError carrying a message suitable
for feeding back to the agent as a repair turn.

NOTE: exec in a restricted namespace is a research scaffold, NOT a security
sandbox. Do not run untrusted scripts from outside the agent loop.
"""

import traceback

from coding_agent.tools import algorithms, primitives
from coding_agent.types import Action, GraphInfo, State, Strategy
from data.wm_simulator import valid_action_ops


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
    namespace = _namespace()
    try:
        compiled = compile(script, "<agent_strategy>", "exec")
        exec(compiled, namespace)  # noqa: S102 — scaffold; see module docstring
    except SyntaxError as error:
        raise StrategyError(f"SyntaxError in generated script: {error}") from error
    except Exception as error:  # noqa: BLE001 — surface any exec error to the agent
        raise StrategyError(
            f"Error executing generated script: {error}\n{traceback.format_exc()}"
        ) from error

    candidates = [
        value
        for name, value in namespace.items()
        if isinstance(value, type)
        and name != "Strategy"
        and (hasattr(value, "plan_horizon") or hasattr(value, "act"))
    ]
    if not candidates:
        raise StrategyError(
            "No Strategy subclass with plan_horizon()/act() found in the script."
        )

    strategy_class = candidates[-1]
    try:
        return strategy_class()  # type: ignore[call-arg]
    except Exception as error:  # noqa: BLE001
        raise StrategyError(f"Could not instantiate Strategy: {error}") from error


def validate_actions(bag: list, num_nodes: int, budget: int) -> None:
    """Raise StrategyError if an action bag references invalid nodes, uses an unknown op, or exceeds budget."""
    num_adds = 0
    for action in bag:
        if action.op not in valid_action_ops:
            raise StrategyError(
                f"action op '{action.op}' is not valid; "
                f"must be one of {valid_action_ops}."
            )
        if not (0 <= int(action.target) < num_nodes):
            raise StrategyError(
                f"action targets node {action.target} out of range [0,{num_nodes})."
            )
        if action.op == "add_node":
            num_adds += 1

    if num_adds > budget:
        raise StrategyError(
            f"action bag adds {num_adds} seeds, exceeds budget {budget}."
        )
