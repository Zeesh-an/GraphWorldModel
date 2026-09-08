"""
Execute a generated strategy script in a controlled namespace and return a
validated Strategy. All failures raise StrategyError carrying a message suitable
for feeding back to the agent as a repair turn.
"""

import ast
import importlib
import inspect
import signal
import threading
import traceback
from contextlib import contextmanager
from functools import partial
from types import SimpleNamespace
from typing import Callable

from coding_agent.tools import (
    adaptive_algorithms,
    algorithms,
    blocking_algorithms,
    dismantling_algorithms,
    immunization_algorithms,
    localization_algorithms,
    prediction_algorithms,
    primitives,
    reconstruction_algorithms,
)
from coding_agent.types import ActionOp, GraphInfo, ScoredStrategy, State, Strategy
from data.wm_simulator import valid_action_ops

# Wall-clock cap on one plan_horizon()/act() call. A generated O(N^2) scan would
# otherwise hang the whole sweep with no diagnosis; the cap turns it into a
# repair turn that names the limit. Overridden per run by run.py from
# --strategy-timeout; 0 disables. The signal only lands between bytecodes, so a
# call stuck inside a long numpy/networkx C call overruns until it returns.
strategy_timeout_seconds = 300.0
# Outbreak sources shown in a rejection message; the prompt's own outbreak block
# uses the same cap, and a 10% outbreak on a 117K-node graph is 11K ids
max_listed_sources = 60

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
#             runs in under 2.7s: imm 0.55s, betweenness_seeds 2.68s, the rest
#             under 0.6s. No library algorithm sits in between.
#
#   honesty   their episodes run on a private NDlib Simulator, so
#             MonteCarloEnvironment.episodes_used never sees them. One one_shot
#             iteration calling celf(mc_runs=20) simulated 7,462 episodes and
#             reported real_env_episodes=2, and that field is the
#             sample-efficiency axis the whole condition ladder is read on.
#
# genetic_algorithm and simulated_annealing are bounded (~17s, a fixed
# population x generations budget rather than a full-N scan) but still call
# mc_simulate_spread, so they leak episodes too and are blocked on that ground.
# static_greedy calls no MC primitive at all: it is here because its per-pick
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

# Same rule on the dismantling side: `greedy_blocking` simulates the contained
# cascade once per candidate per pick, so it leaks episodes past
# MonteCarloEnvironment.episodes_used exactly as `celf` does
mc_blocked_dismantling = dismantling_algorithms.mc_dismantling_algorithms

_unknown_blocked = set(mc_blocked_dismantling) - set(
    dismantling_algorithms.dismantling_algorithms
)
if _unknown_blocked:
    raise ValueError(
        f"mc_dismantling_algorithms names {sorted(_unknown_blocked)}, which are "
        f"not in dismantling_algorithms; fix the list or the rename"
    )

# Same rule on the source-localization side. `resim_greedy` falls back to a
# PRIVATE NDlib estimator when it is not handed a forward oracle, so a generated
# script calling it would bypass `real_env_episodes` exactly as `celf` does, and
# a generated script is offline by construction, so nothing is lost by
# blocking it.
mc_blocked_localization = localization_algorithms.mc_localization_algorithms

_unknown_blocked = set(mc_blocked_localization) - set(
    localization_algorithms.localization_algorithms
)
if _unknown_blocked:
    raise ValueError(
        f"mc_localization_algorithms names {sorted(_unknown_blocked)}, which are "
        f"not in localization_algorithms; fix the list or the rename"
    )

# ...and on the influence-blocking side. `greedy_prevention` simulates the whole
# COMPETITIVE cascade once per candidate per pick, which is the most expensive thing
# in this repo and, like `celf`, runs on a private simulator that
# MonteCarloEnvironment.episodes_used never sees.
mc_blocked_blocking = blocking_algorithms.mc_blocking_algorithms

_unknown_blocked = set(mc_blocked_blocking) - set(
    blocking_algorithms.all_blocking_algorithms
)
if _unknown_blocked:
    raise ValueError(
        f"mc_blocking_algorithms names {sorted(_unknown_blocked)}, which are not "
        f"in blocking_algorithms; fix the list or the rename"
    )

# ...and on the epidemic-control side. `mc_greedy_immunization` simulates the whole
# compartmental outbreak once per candidate per dose, which is the same leak
# `greedy_blocking` has and for the same reason.
mc_blocked_immunization = immunization_algorithms.mc_immunization_algorithms

_unknown_blocked = set(mc_blocked_immunization) - set(
    immunization_algorithms.immunization_algorithms
)
if _unknown_blocked:
    raise ValueError(
        f"mc_immunization_algorithms names {sorted(_unknown_blocked)}, which are "
        f"not in immunization_algorithms; fix the list or the rename"
    )

# ...and on the cascade-reconstruction side. `mcmc_decode` and `forward_backward`
# evaluate the transition kernel `proposals x horizon` times per instance, and a
# generated program is offline by construction (the kernel is bound only to
# canned baselines), so blocking them keeps a generated decoder honest at no cost.
mc_blocked_reconstruction = reconstruction_algorithms.mc_reconstruction_algorithms

_unknown_blocked = set(mc_blocked_reconstruction) - set(
    reconstruction_algorithms.reconstruction_algorithms
)
if _unknown_blocked:
    raise ValueError(
        f"mc_reconstruction_algorithms names {sorted(_unknown_blocked)}, which are "
        f"not in reconstruction_algorithms; fix the list or the rename"
    )


# Kernel-heavy POPULARITY predictors, on the same terms as every other pool's:
# `mc_forward` unrolls the forward model `steps * forecast_samples` times per
# cascade; a generated predictor is offline by construction (the forward model
# is bound only to canned baselines), so blocking it costs nothing.
mc_blocked_prediction = prediction_algorithms.mc_prediction_algorithms

_unknown_blocked = set(mc_blocked_prediction) - set(
    prediction_algorithms.prediction_algorithms
)
if _unknown_blocked:
    raise ValueError(
        f"mc_prediction_algorithms names {sorted(_unknown_blocked)}, which are "
        f"not in prediction_algorithms; fix the list or the rename"
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


def _blocked_algorithm(name: str, *_args: object, **_kwargs: object) -> None:
    """
    Stand-in for a blocked algorithm.

    Raising here rather than letting the attribute be missing turns a wasted
    refinement iteration into a repair turn that says what to do instead.
    """
    blocked = (
        adaptive_algorithms.mc_adaptive_algorithms
        if name in adaptive_algorithms.mc_adaptive_algorithms
        else mc_blocked_algorithms
    )

    raise StrategyError(
        f"{name} is not available: it estimates spread by simulating "
        f"the cascade for every candidate node, which bypasses the metered "
        f"evaluator and dominates wall clock. Blocked: "
        f"{', '.join(blocked)}. Use a structural or RIS-based "
        f"algorithm (e.g. degree_discount, imm, tim, voterank, collective_influence) "
        f"or write your own selection logic."
    )


def _blocked_dismantler(name: str, *_args: object, **_kwargs: object) -> None:
    raise StrategyError(
        f"dismantling_algorithms.{name} is not available: it simulates the "
        f"contained cascade once per candidate per pick, which bypasses the "
        f"metered evaluator and dominates wall clock. Blocked: "
        f"{', '.join(mc_blocked_dismantling)}. Use a structural dismantler "
        f"(adaptive_degree, corehd, collective_influence_removal, "
        f"explosive_immunization, netshield) or write your own selection logic."
    )


def _blocked_blocker(name: str, *_args: object, **_kwargs: object) -> None:
    raise StrategyError(
        f"blocking_algorithms.{name} is not available: it simulates the whole "
        f"competitive cascade once per candidate per pick, which bypasses the "
        f"metered evaluator and dominates wall clock. Blocked: "
        f"{', '.join(mc_blocked_blocking)}. Use a structural or RIS-based blocker "
        f"(proximity, rps, reverse_blocking, cmia_o, imin_lhga) or write your own "
        f"selection logic."
    )


def _blocked_immunizer(name: str, *_args: object, **_kwargs: object) -> None:
    raise StrategyError(
        f"immunization_algorithms.{name} is not available: it simulates the whole "
        f"compartmental outbreak once per candidate per dose, which bypasses the "
        f"metered evaluator and dominates wall clock. Blocked: "
        f"{', '.join(mc_blocked_immunization)}. Use a structural or data-aware "
        f"immunizer (degree_immunization, netshield, dava, frontier_immunization, "
        f"acquaintance_immunization) or write your own selection logic."
    )


def _blocked_localizer(name: str, *_args: object, **_kwargs: object) -> None:
    raise StrategyError(
        f"localization_algorithms.{name} is not available: it re-simulates every "
        f"candidate on its own private simulator, which bypasses the metered "
        f"evaluator. Blocked: {', '.join(mc_blocked_localization)}. Your program is "
        f"OFFLINE: infer the sources from the graph, the edge probabilities and "
        f"the observation alone (an analytic re-simulation over `graph.ic_probs` "
        f"is fine; a simulator call is not)."
    )


def _blocked_decoder(name: str, *_args: object, **_kwargs: object) -> None:
    raise StrategyError(
        f"reconstruction_algorithms.{name} is not available: it evaluates the "
        f"transition kernel proposals x horizon times per instance, which "
        f"dominates wall clock. Blocked: {', '.join(mc_blocked_reconstruction)}. "
        f"Your program is OFFLINE: decode from the graph, the edge probabilities, "
        f"the reports and their times alone (an analytic one-step IC kernel over "
        f"`graph.ic_probs` is fine; a simulator call is not)."
    )


def _blocked_predictor(name: str, *_args: object, **_kwargs: object) -> None:
    raise StrategyError(
        f"prediction_algorithms.{name} is not available: it unrolls the forward "
        f"model steps x forecast_samples times per cascade, which dominates wall "
        f"clock. Blocked: {', '.join(mc_blocked_prediction)}. Your program is "
        f"OFFLINE: predict from the observed adoption history, the wave series and "
        f"the graph structure alone."
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
            # The one library object scored mode DOES get, because the fixed
            # `predict()` harness already calls it: a growth_factor() rule that
            # could not read the features would be searching a space of constants.
            "cascade_features": prediction_algorithms.cascade_features,
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

    # Same treatment for the per-round policies: adapt_greedy re-simulates every
    # candidate on a PRIVATE simulator, so a generated act() calling it would
    # spend thousands of episodes that episodes_used never sees. Exactly the
    # honesty problem documented for celf above, one task over.
    callable_adaptive = {
        name: (
            function
            if allow_mc_algorithms
            or name not in adaptive_algorithms.mc_adaptive_algorithms
            else partial(_blocked_algorithm, name)
        )
        for name, function in adaptive_algorithms.adaptive_algorithms.items()
    }

    return {
        "ActionOp": ActionOp,
        "State": State,
        "GraphInfo": GraphInfo,
        "Strategy": Strategy,
        "algorithms": SimpleNamespace(**callable_algorithms),
        # Per-round policies for adaptive IM. Present under every task: a
        # non-adaptive arm simply never has a round to call one from, and hiding
        # them per task would mean the namespace no longer matches the prompt.
        "adaptive_algorithms": SimpleNamespace(**callable_adaptive),
        # Node-REMOVAL selectors for critical node detection, on the same terms:
        # present under every task, and the MC-heavy member blocked unless
        # --allow-mc-algorithms, exactly like `celf` on the seeding side
        "dismantling_algorithms": SimpleNamespace(
            **{
                name: (
                    function
                    if allow_mc_algorithms or name not in mc_blocked_dismantling
                    else partial(_blocked_dismantler, name)
                )
                for name, function in dismantling_algorithms.dismantling_algorithms.items()
            }
        ),
        # Counter-seed / node / edge blockers for influence blocking, on the same
        # terms: present under every task, and the MC-heavy member blocked unless
        # --allow-mc-algorithms. `blocking_levers` rides along because a generated
        # script that composes one of these has to know which op it may emit.
        "blocking_algorithms": SimpleNamespace(
            **{
                name: (
                    function
                    if allow_mc_algorithms or name not in mc_blocked_blocking
                    else partial(_blocked_blocker, name)
                )
                for name, function in blocking_algorithms.all_blocking_algorithms.items()
            },
            blocking_levers=blocking_algorithms.blocking_levers,
        ),
        # Vaccination / quarantine / edge selectors for epidemic control, on the
        # same terms: present under every task, and the MC-heavy member blocked
        # unless --allow-mc-algorithms. `immunization_levers` rides along because a
        # generated script that composes one has to know which op it may emit,
        # `netmelt` returns arcs and `netshield` returns nodes.
        "immunization_algorithms": SimpleNamespace(
            **{
                name: (
                    function
                    if allow_mc_algorithms or name not in mc_blocked_immunization
                    else partial(_blocked_immunizer, name)
                )
                for name, function in immunization_algorithms.immunization_algorithms.items()
            },
            immunization_levers=immunization_algorithms.immunization_levers,
        ),
        # SOURCE-SET inference for source localization, on the same terms again.
        # These are the published methods a localize() program is being compared
        # against, so hiding them would ask the model to reinvent LPSI.
        "localization_algorithms": SimpleNamespace(
            **{
                name: (
                    function
                    if allow_mc_algorithms or name not in mc_blocked_localization
                    else partial(_blocked_localizer, name)
                )
                for name, function in localization_algorithms.localization_algorithms.items()
            }
        ),
        # ...and their per-node score vectors, which is what an AUC-aware program
        # returns from source_scores() and what a hybrid rule combines
        "localization_scorers": SimpleNamespace(
            **localization_algorithms.localization_scorers
        ),
        # TRAJECTORY decoders for cascade reconstruction, on the same terms again.
        # These are the published methods a reconstruct() program is compared
        # against, so hiding them would ask the model to reinvent delayed-BFS.
        "reconstruction_algorithms": SimpleNamespace(
            **{
                name: (
                    function
                    if allow_mc_algorithms or name not in mc_blocked_reconstruction
                    else partial(_blocked_decoder, name)
                )
                for name, function in reconstruction_algorithms.reconstruction_algorithms.items()
            }
        ),
        # POPULARITY predictors for cascade prediction, on the same terms again.
        # `cascade_features` rides along because it is the shared feature extractor
        # every §3.1 method here is defined over: Cheng et al.'s five classes, minus
        # content, and the scored-mode harness takes its output directly, so hiding
        # it would ask the model to reinvent the one thing that paper actually found.
        "prediction_algorithms": SimpleNamespace(
            **{
                name: (
                    function
                    if allow_mc_algorithms or name not in mc_blocked_prediction
                    else partial(_blocked_predictor, name)
                )
                for name, function in prediction_algorithms.prediction_algorithms.items()
            }
        ),
        "cascade_features": prediction_algorithms.cascade_features,
        "primitives": primitives,
        # Containment helpers a canned dismantling baseline needs (removal_plan
        # filters the outbreak's sources out of a published algorithm's output).
        # Imported here rather than at module scope: containment imports the tools
        # package, which imports this module.
        "containment": importlib.import_module("coding_agent.containment"),
        # ...and the blocking helpers, on the same terms. `blocking_plan` is what
        # turns a library selector's output: node ids from three levers, arcs from
        # the fourth: into the one plan shape the executor validates.
        "blocking": importlib.import_module("coding_agent.blocking"),
        # ...and the epidemic helpers. `immunization_plan` is what turns an
        # immunizer's output: node ids from two levers, arcs from the other two,
        # into the one plan shape the executor validates, and it also drops any
        # index case the algorithm happened to pick.
        "epidemic": importlib.import_module("coding_agent.epidemic"),
        "__builtins__": __builtins__,
    }


def build_strategy(
    script: str,
    strategy_mode: str = "free",
    allow_mc_algorithms: bool = False,
    canned: bool = False,
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
            and (
                hasattr(value, "plan_horizon")
                or hasattr(value, "act")
                or hasattr(value, "localize")
                or hasattr(value, "reconstruct")
                or hasattr(value, "predict")
            )
        ]

        if not candidates:
            raise StrategyError(
                "No Strategy subclass with plan_horizon()/act()/localize()/"
                "reconstruct()/predict() found in the script."
            )

    strategy_class = candidates[-1]

    # All four harnesses are fixed in scored mode: plan_horizon for the
    # intervention tasks, localize for source localization, reconstruct for
    # cascade reconstruction, predict for cascade prediction. Overriding one
    # turns scored mode back into free mode, which is the one thing the
    # condition exists to prevent.
    for fixed in ("plan_horizon", "localize", "reconstruct", "predict"):
        if strategy_mode == "scored" and getattr(strategy_class, fixed) is not getattr(
            ScoredStrategy, fixed
        ):
            raise StrategyError(
                f"scored mode: {fixed} is the fixed harness and may not be "
                f"overridden: override only score(), schedule(), source_score(), "
                f"edge_cost() or growth_factor()."
            )
    try:
        strategy = strategy_class()
    except Exception as error:
        raise StrategyError(f"Could not instantiate Strategy: {error}") from error

    # Retained so results JSONs archive the exact code that produced the reward
    strategy.source_script = script
    # True only for a library, routed or external baseline built by the harness.
    # The evaluator bindings (`predict_marginals`, `step_marginals`,
    # `forecast_marginals`) exist for the kernel-using members of those pools and
    # are attached to canned strategies only: a GENERATED program is offline by
    # construction and never gets a handle on the arm's evaluator.
    strategy.canned = canned
    return strategy


def call_strategy(method: Callable, *args: object, allow_none: bool = False) -> object:
    """
    Invoke generated-strategy code, converting any runtime failure into a
    StrategyError repair turn.

    A `None` return is caught here rather than downstream: `plan_horizon`, `act`,
    `localize`, `reconstruct` and the scored hooks all must return a value, and the
    commonest way to get None is a model that implemented the wrong method for the
    task's own harness: `plan_horizon` when the method wanted `act`, or an `act`
    that falls off the end without returning. Left alone it surfaces as
    `TypeError: 'NoneType' object is not iterable` from inside validation, which
    is an opaque traceback instead of a turn the model can act on.

    `allow_none` is the one exception, and it exists for exactly one contract:
    `predict()` returns None to DECLINE scoring a cascade, which is a legitimate
    answer rather than a failure (research/cascade_prediction.md §8.4: a
    generative model whose fit diverges on a supercritical cascade produces no
    estimate, and reporting the mean over scoreable cascades alone "silently
    favours the model that gives up more often"). Declines are COUNTED in
    `n_failed`, never scored as errors.
    """
    try:
        with _time_limit(strategy_timeout_seconds):
            result = method(*args)

        if result is None and not allow_none:
            raise StrategyError(
                f"{getattr(method, '__name__', 'the strategy method')}() returned "
                f"None; it must return a list of ActionOp bags. The usual cause is "
                f"implementing the wrong entry point for this method, or an early "
                f"code path that falls off the end without a return statement."
            )

        return result
    except StrategyError:
        raise
    except _StrategyTimeout:
        raise StrategyError(
            f"strategy exceeded the {strategy_timeout_seconds:.0f}s wall-clock limit "
            f"for one call. Something in it scales badly: a scan over all nodes "
            f"inside a per-pick loop, or a sample size far larger than the graph "
            f"needs. Rewrite it to run in seconds: vectorize with numpy, precompute "
            f"scores once outside the selection loop, or shrink the sample count."
        ) from None
    except Exception as error:
        raise StrategyError(
            f"strategy raised {type(error).__name__}: {error}\n"
            f"{traceback.format_exc()}"
        ) from error


def budget_key(action) -> object:
    """
    What "the same unit of budget" means for one action.

    A node op is identified by its target; an EDGE op by the whole arc, because two
    arcs out of the same `u` are two different interventions. Keying edge ops on
    `target` alone (which is what a node-shaped check does) would reject a legal
    plan that cuts two of a hub's out-edges as a duplicate, and that is exactly the
    plan an edge-blocking lever is supposed to produce.
    """
    if action.destination is None:
        return int(action.target)

    return (int(action.target), int(action.destination))


def _listed_sources(guarded: set[int] | frozenset[int]) -> str:
    sources = sorted(guarded)
    listed = ", ".join(str(node) for node in sources[:max_listed_sources])
    if len(sources) > max_listed_sources:
        listed += f", … and {len(sources) - max_listed_sources} more (self.outbreak has them all)"
    return f"[{listed}]"


def validate_actions(
    bag: list,
    num_nodes: int,
    budget: int,
    allowed_ops: tuple = valid_action_ops,
    budget_op: str = "add_node",
    protected: tuple = (),
    edge_weight_caps: dict | None = None,
) -> None:
    """
    Raise StrategyError if an action bag references invalid nodes, uses a
    disallowed op, repeats a budgeted target, touches a protected node, or exceeds
    budget.

    `budget_op` is what one unit of budget buys: `add_node` for a seeding task,
    `remove_node` for a containment one. Only that op is counted: a containment
    plan's `remove_edge` ops are the mechanics of a node deletion
    (`containment.delete_node_ops`), so charging them would make k mean deg(v)
    different things per node.

    `protected` is the outbreak's source set on a containment task. Deleting a
    source ENDS the outbreak instead of containing it, which collapses the problem
    to "find the sources": a different task (`source_localization`) with a
    different objective, and one where a uniform random removal set beats every
    dismantler by luck. The published immunization protocol vaccinates first and
    then infects a NON-immunized node, so this rule is that protocol rather than a
    house restriction.

    `edge_weight_caps` is `{(u, v): p}` on the weight-reduction lever of influence
    blocking. DiffIM's relaxation is `p~ = p * r~` with `r~ in [0, 1]`, so a blocker
    may only LOWER an edge: raising one would be an unbudgeted boost to its own
    counter-cascade wearing a blocking action's name.
    """
    spent_units = 0
    targeted = set()
    guarded = {int(node) for node in protected}
    noun = {
        "add_node": "seeds",
        "remove_node": "removes",
        "remove_edge": "cuts",
        "set_edge_weight": "reweights",
    }.get(budget_op, "spends")

    for action in bag:
        if action.op not in allowed_ops:
            # The likeliest way a containment strategy hits this is by writing out
            # the incident edge removals itself, so name the reason rather than
            # only the rule: those ops are free, and a plan that buys them is
            # spending an intervention the budget never charged for
            hint = (
                " A node deletion is emitted as a BARE remove_node: the harness "
                "expands it into the incident remove_edge ops for you, and letting "
                "you emit them would be an unbudgeted second intervention."
                if action.op in ("add_edge", "remove_edge", "set_edge_weight")
                and budget_op == "remove_node"
                else ""
            )
            raise StrategyError(
                f"action op '{action.op}' is not allowed for this task; "
                f"must be one of {allowed_ops}.{hint}"
            )

        if not (0 <= int(action.target) < num_nodes):
            raise StrategyError(
                f"action targets node {action.target} out of range [0,{num_nodes})."
            )

        if action.op == "remove_node" and int(action.target) in guarded:
            raise StrategyError(
                f"node {action.target} is an OUTBREAK SOURCE and cannot be removed. "
                f"Deleting a source ends the outbreak rather than containing it, "
                f"which is not the problem you are being scored on. The sources are "
                f"{_listed_sources(guarded)}: filter them out of your candidates and "
                f"spend the budget on the routes out of them instead."
            )

        if action.op in ("add_edge", "remove_edge", "set_edge_weight"):
            if action.destination is None:
                raise StrategyError(
                    f"action op '{action.op}' needs a destination: emit "
                    f"ActionOp('{action.op}', u, v) for the arc u -> v."
                )

            if not (0 <= int(action.destination) < num_nodes):
                raise StrategyError(
                    f"action targets node {action.destination} out of range "
                    f"[0,{num_nodes})."
                )

        if action.op == "set_edge_weight" and edge_weight_caps is not None:
            arc = (int(action.target), int(action.destination))
            cap = edge_weight_caps.get(arc)

            if cap is None:
                raise StrategyError(
                    f"set_edge_weight targets arc {arc}, which does not exist in the "
                    f"graph. Reweighting a missing arc buys nothing and spends a unit "
                    f"of budget; pick from the arcs in graph.edge_index."
                )

            weight = 0.0 if action.weight is None else float(action.weight)
            if weight > cap + 1e-9:
                raise StrategyError(
                    f"set_edge_weight raises arc {arc} from {cap:.6f} to "
                    f"{weight:.6f}. A blocker may only REDUCE an edge, the "
                    f"published relaxation is p~(u,v) = p(u,v) * r with r in [0, 1]: "
                    f"so a weight above the arc's own probability would be an "
                    f"unbudgeted boost rather than a block."
                )

        if action.op == budget_op:
            spent_units += 1

            # Targeting the same unit twice spends two of k on one intervention and
            # would otherwise pass silently as a budget the strategy never used
            key = budget_key(action)
            if key in targeted:
                raise StrategyError(
                    f"action bag {noun} {key} more than once; a duplicate "
                    f"{budget_op} spends budget without changing the graph. "
                    f"Deduplicate the set before returning it."
                )

            targeted.add(key)

    if spent_units > budget:
        raise StrategyError(
            f"action bag {noun} {spent_units} nodes, exceeds budget {budget}."
        )
