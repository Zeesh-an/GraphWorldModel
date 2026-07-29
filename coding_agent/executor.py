"""
Execute a generated strategy script in a controlled namespace and return a
validated Strategy. All failures raise StrategyError carrying a message suitable
for feeding back to the agent as a repair turn.
"""

import ast
import inspect
import signal
import threading
import traceback
from contextlib import contextmanager
from functools import partial
from types import SimpleNamespace
from typing import Callable

from coding_agent.tools import algorithms, primitives
from coding_agent.types import ActionOp, GraphInfo, ScoredStrategy, State, Strategy
from data.wm_simulator import valid_action_ops

# Wall-clock cap on one plan_horizon()/act() call. A generated O(N^2) scan would
# otherwise hang the whole sweep with no diagnosis; the cap turns it into a
# repair turn that names the limit. Overridden per run by run.py from
# --strategy-timeout; 0 disables. The signal only lands between bytecodes, so a
# call stuck inside a long numpy/networkx C call overruns until it returns.
strategy_timeout_seconds = 300.0

# Third-party modules a generated script may import. numpy and networkx are the
# point: without them the model writes pure-Python loops over the adjacency and
# cannot afford the sample sizes that make RIS-style selection work.
allowed_imports = (
    "numpy",
    "networkx",
    "scipy",
    "math",
    "random",
    "statistics",
    "heapq",
    "bisect",
    "collections",
    "itertools",
    "functools",
)

# These defeat the import whitelist or reach the filesystem, so the AST check
# rejects them by name rather than pretending the whitelist is enforceable
forbidden_builtins = ("__import__", "eval", "exec", "compile", "open", "input")

# Hidden from scored mode: simulation-based selection would rebuild CELF (and
# bypass the metered evaluator) instead of inventing structural scoring logic
scored_blocked_primitives = (
    "mc_simulate_spread",
    "compute_marginal_gain",
    "build_simulator",
)

# Hidden from free mode unless --allow-mc-algorithms. Every one of these
# estimates spread by simulating the cascade once per candidate node per pick,
# which costs on two axes:
#
#   time      measured on BA-1589 at k=79 (5% of N): all nine below exceed 60s
#             per call, vanilla_greedy exceeds 120s. Everything still available
#             runs in under 2.7s — imm 0.55s, betweenness_seeds 2.68s, the rest
#             under 0.6s. No library algorithm sits in between.
#
#   honesty   their episodes run on a private NDlib Simulator, so
#             MonteCarloEnvironment.episodes_used never sees them. One one_shot
#             iteration calling celf(mc_runs=20) simulated 7,462 episodes and
#             reported real_env_episodes=2 — and that field is the
#             sample-efficiency axis the whole condition ladder is read on.
#
# genetic_algorithm and simulated_annealing are bounded (~17s, a fixed
# population x generations budget rather than a full-N scan) but still call
# mc_simulate_spread, so they leak episodes too and are blocked on that ground.
# static_greedy calls no MC primitive at all — it is here because its per-pick
# snapshot reachability scan is the same shape and the same cost.
mc_blocked_algorithms = (
    "vanilla_greedy",
    "celf",
    "celf_pp",
    "celf_local_search",
    "community_celf",
    "pagerank_greedy",
    "adaptive_greedy",
    "hill_climbing",
    "static_greedy",
    "genetic_algorithm",
    "simulated_annealing",
)

# A rename in algorithms.py would otherwise silently un-block one of these
_unknown_blocked = set(mc_blocked_algorithms) - set(algorithms.algorithms)
if _unknown_blocked:
    raise ValueError(
        f"mc_blocked_algorithms names {sorted(_unknown_blocked)}, which are not "
        f"in algorithms.algorithms; fix the list or the rename"
    )


class StrategyError(RuntimeError):
    """A generated script failed to parse, execute, or expose a valid Strategy."""


class _StrategyTimeout(Exception):
    """Raised by the SIGALRM handler; converted to StrategyError by call_strategy."""


def _raise_timeout(_signum: int, _frame: object) -> None:
    raise _StrategyTimeout()


@contextmanager
def _time_limit(seconds: float):
    # SIGALRM is main-thread and Unix only; elsewhere the call runs uncapped
    if (
        seconds <= 0
        or not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)

    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def check_script_imports(script: str) -> None:
    """Raise StrategyError if the script imports outside the whitelist."""
    try:
        tree = ast.parse(script)
    except SyntaxError as error:
        raise StrategyError(f"SyntaxError in generated script: {error}") from error

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in forbidden_builtins:
            raise StrategyError(
                f"generated script uses {node.id!r}, which is not available. "
                f"Import what you need directly from: {', '.join(allowed_imports)}."
            )

        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            continue

        for module in modules:
            if module.split(".")[0] not in allowed_imports:
                raise StrategyError(
                    f"generated script imports {module!r}, which is not available. "
                    f"Importable modules: {', '.join(allowed_imports)}. "
                    f"ActionOp/State/GraphInfo/Strategy/algorithms/primitives are "
                    f"already in the namespace and must not be imported."
                )


def _blocked_algorithm(name: str, *_args, **_kwargs) -> None:
    """
    Stand-in for a blocked algorithm.

    Raising here rather than letting the attribute be missing turns a wasted
    refinement iteration into a repair turn that says what to do instead.
    """
    raise StrategyError(
        f"algorithms.{name} is not available: it estimates spread by simulating "
        f"the cascade for every candidate node, which bypasses the metered "
        f"evaluator and dominates wall clock. Blocked: "
        f"{', '.join(mc_blocked_algorithms)}. Use a structural or RIS-based "
        f"algorithm (e.g. degree_discount, imm, tim, voterank, collective_influence) "
        f"or write your own selection logic."
    )


def _namespace(strategy_mode: str = "free", allow_mc_algorithms: bool = False) -> dict:
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

    # The exact callable surface the prompts advertise. Blocked names stay
    # present as raisers rather than vanishing, so calling one is a legible
    # repair turn instead of an AttributeError traceback
    callable_algorithms = {
        name: (
            function
            if allow_mc_algorithms or name not in mc_blocked_algorithms
            else partial(_blocked_algorithm, name)
        )
        for name, function in algorithms.algorithms.items()
    }

    return {
        "ActionOp": ActionOp,
        "State": State,
        "GraphInfo": GraphInfo,
        "Strategy": Strategy,
        "algorithms": SimpleNamespace(**callable_algorithms),
        "primitives": primitives,
        "__builtins__": __builtins__,
    }


def build_strategy(
    script: str, strategy_mode: str = "free", allow_mc_algorithms: bool = False
) -> Strategy:
    # Converts a generated script string into a live object
    check_script_imports(script)
    namespace = _namespace(strategy_mode, allow_mc_algorithms)

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
        with _time_limit(strategy_timeout_seconds):
            return method(*args)
    except StrategyError:
        raise
    except _StrategyTimeout:
        raise StrategyError(
            f"strategy exceeded the {strategy_timeout_seconds:.0f}s wall-clock limit "
            f"for one call. Something in it scales badly — a scan over all nodes "
            f"inside a per-pick loop, or a sample size far larger than the graph "
            f"needs. Rewrite it to run in seconds: vectorize with numpy, precompute "
            f"scores once outside the selection loop, or shrink the sample count."
        ) from None
    except Exception as error:
        raise StrategyError(
            f"strategy raised {type(error).__name__}: {error}\n"
            f"{traceback.format_exc()}"
        ) from error


def validate_actions(
    bag: list, num_nodes: int, budget: int, allowed_ops: tuple = valid_action_ops
) -> None:
    """Raise StrategyError if an action bag references invalid nodes, uses a disallowed op, repeats a seed, or exceeds budget."""
    num_adds = 0
    seeded = set()

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

            # Seeding a node twice spends two of k on one node and would
            # otherwise pass silently as a budget the strategy never used
            if int(action.target) in seeded:
                raise StrategyError(
                    f"action bag seeds node {action.target} more than once; a "
                    f"duplicate add_node spends budget without adding a node. "
                    f"Deduplicate the seed set before returning it."
                )

            seeded.add(int(action.target))

    if num_adds > budget:
        raise StrategyError(
            f"action bag adds {num_adds} seeds, exceeds budget {budget}."
        )
