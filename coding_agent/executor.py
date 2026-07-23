"""
Execute a generated strategy script in a controlled namespace and return a
validated Strategy. All failures raise StrategyError carrying a message suitable
for feeding back to the agent as a repair turn.
"""

import inspect
import traceback
from types import SimpleNamespace
from typing import Callable

from coding_agent.tools import algorithms, primitives
from coding_agent.types import ActionOp, GraphInfo, ScoredStrategy, State, Strategy
from data.wm_simulator import valid_action_ops

# Hidden from scored mode: simulation-based selection would rebuild CELF (and
# bypass the metered evaluator) instead of inventing structural scoring logic
scored_blocked_primitives = (
    "mc_simulate_spread",
    "compute_marginal_gain",
    "build_simulator",
)


class StrategyError(RuntimeError):
    """A generated script failed to parse, execute, or expose a valid Strategy."""


def _namespace(strategy_mode: str = "free") -> dict:
    if strategy_mode == "scored":
        # No algorithms module: the agent must write its own scoring logic
        # instead of composing whole library algorithms
        allowed = {
            name: function
            for name, function in vars(primitives).items()
            if inspect.isfunction(function)
            and function.__module__ == primitives.__name__
            and name not in scored_blocked_primitives
        }

        return {
            "ActionOp": ActionOp,
            "State": State,
            "GraphInfo": GraphInfo,
            "ScoredStrategy": ScoredStrategy,
            "primitives": SimpleNamespace(**allowed),
            "__builtins__": __builtins__,
        }

    # The exact callable surface the prompts advertise
    return {
        "ActionOp": ActionOp,
        "State": State,
        "GraphInfo": GraphInfo,
        "Strategy": Strategy,
        "algorithms": algorithms,
        "primitives": primitives,
        "__builtins__": __builtins__,
    }


def build_strategy(script: str, strategy_mode: str = "free") -> Strategy:
    # Converts a generated script string into a live object
    namespace = _namespace(strategy_mode)

    try:
        # Convert script string into an executable Strategy object for the agent
        compiled = compile(script, "<agent_strategy>", "exec")
        exec(compiled, namespace)
    except SyntaxError as error:
        raise StrategyError(f"SyntaxError in generated script: {error}") from error
    except Exception as error:
        raise StrategyError(
            f"Error executing generated script: {error}\n{traceback.format_exc()}"
        ) from error

    if strategy_mode == "scored":
        candidates = [
            value
            for name, value in namespace.items()
            if isinstance(value, type)
            and value is not ScoredStrategy
            and issubclass(value, ScoredStrategy)
        ]

        if not candidates:
            raise StrategyError(
                "No ScoredStrategy subclass found in the script; define one class "
                "subclassing ScoredStrategy that overrides score() (and optionally schedule())."
            )
    else:
        candidates = [
            value
            for name, value in namespace.items()
            # Identity check, not name: a script may legally name its class "Strategy"
            if isinstance(value, type)
            and value is not Strategy
            and (hasattr(value, "plan_horizon") or hasattr(value, "act"))
        ]

        if not candidates:
            raise StrategyError(
                "No Strategy subclass with plan_horizon()/act() found in the script."
            )

    strategy_class = candidates[-1]

    if (
        strategy_mode == "scored"
        and strategy_class.plan_horizon is not ScoredStrategy.plan_horizon
    ):
        raise StrategyError(
            "scored mode: plan_horizon is the fixed harness and may not be "
            "overridden — override only score() and/or schedule()."
        )
    try:
        strategy = strategy_class()
    except Exception as error:
        raise StrategyError(f"Could not instantiate Strategy: {error}") from error

    # Retained so results JSONs archive the exact code that produced the reward
    strategy.source_script = script
    return strategy


def call_strategy(method: Callable, *args) -> object:
    """Invoke generated-strategy code, converting any runtime failure into a StrategyError repair turn."""
    try:
        return method(*args)
    except StrategyError:
        raise
    except Exception as error:
        raise StrategyError(
            f"strategy raised {type(error).__name__}: {error}\n"
            f"{traceback.format_exc()}"
        ) from error


def validate_actions(
    bag: list, num_nodes: int, budget: int, allowed_ops: tuple = valid_action_ops
) -> None:
    """Raise StrategyError if an action bag references invalid nodes, uses a disallowed op, or exceeds budget."""
    num_adds = 0

    for action in bag:
        if action.op not in allowed_ops:
            raise StrategyError(
                f"action op '{action.op}' is not allowed for this task; "
                f"must be one of {allowed_ops}."
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
