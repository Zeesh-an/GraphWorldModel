"""
Adaptive IM: the round schedule, the feedback model, and the ActionFn that turns
a generated policy into something an environment can roll out.

Adaptive IM picks `k` seeds in `r` batches instead of all at once, choosing each
batch AFTER observing the diffusion the previous one produced
(research/adaptive_online_im.md §1.1). Our transition function already accepts a
bag of add_node ops at any timestep, so the whole difference from static IM lives
in this file: when the batches land, how big each one is, and what the policy is
allowed to see when it picks them.

Two properties this is built around, both consequences of how the environments
are shaped rather than of the task:

  * The budget is enforced STRUCTURALLY, by capping each round's bag at its own
    batch size. Sum of batches == k by construction, so no cross-call counter is
    needed — which matters because MonteCarloEnvironment loops (episode, then
    timestep) while WorldModelEnvironment loops (timestep, then sample), so any
    cumulative state carried between action_fn calls would mean different things
    under the two.
  * Re-seeding is checked against the TRUE state at the call, not against a
    remembered set, for the same reason. Under full_adoption that is an error the
    policy could have avoided; under myopic it could not, so the seed is dropped
    and the slot is spent instead. See prepare_round_bag.
"""

import math

from coding_agent.executor import StrategyError, call_strategy, validate_actions
from coding_agent.types import (
    ActionFn,
    ActionOp,
    GraphInfo,
    State,
    Strategy,
    TaskSpec,
    myopic,
    valid_feedback_models,
)


def round_batches(
    budget: int, rounds: int | None, per_round_budget: int | None = None
) -> list[int]:
    """
    Seeds per round, summing to exactly `budget`.

    Two conventions, matching Han et al.'s two sweeps (§8.2 trap 3):
    fix `b` and derive `r = ceil(k / b)`, or fix `r` and split `k` as evenly as
    possible. `per_round_budget` wins when both are given.
    """
    if budget < 1:
        raise ValueError(f"adaptive rounds need a positive budget, got {budget}")

    if per_round_budget is not None:
        if per_round_budget < 1:
            raise ValueError(
                f"--per-round-budget must be >= 1, got {per_round_budget}"
            )

        count = math.ceil(budget / per_round_budget)
        batches = [per_round_budget] * count
        # Trim the last batch rather than overspending k
        batches[-1] -= sum(batches) - budget

        return batches

    if rounds is None or rounds < 1:
        raise ValueError(f"--rounds must be >= 1, got {rounds}")

    # More rounds than seeds would mean empty rounds, which spend a decision
    # point on nothing; cap instead so r is always the number of real batches
    rounds = min(rounds, budget)
    base, extra = divmod(budget, rounds)

    return [base + 1] * extra + [base] * (rounds - extra)


def round_schedule(batches: list[int], round_gap: int, horizon: int) -> dict[int, int]:
    """{timestep: batch size} for each round, validated against the horizon."""
    if round_gap < 1:
        raise ValueError(f"--round-gap must be >= 1, got {round_gap}")

    last = (len(batches) - 1) * round_gap
    if last > horizon:
        raise ValueError(
            f"{len(batches)} rounds at --round-gap {round_gap} put the last batch "
            f"at t={last}, past --horizon {horizon}. Seeds scheduled past the "
            f"horizon are silently never spent, so this is rejected rather than "
            f"clipped: raise --horizon to >= {last}, lower --rounds, or lower "
            f"--round-gap."
        )

    return {index * round_gap: size for index, size in enumerate(batches)}


def observed_state(state: State, feedback_model: str) -> State:
    """
    What the policy is allowed to read at a round boundary.

    full_adoption hands over the realized state as-is. myopic hides `infected`,
    leaving only the wave activated since the last step — the information state
    Peng & Chen analyse, and the one where adaptive submodularity fails. The
    harness still checks proposals against the true state, so hiding it costs
    the policy information without letting it spend budget on an active node.
    """
    if feedback_model not in valid_feedback_models:
        raise ValueError(
            f"unknown feedback_model {feedback_model!r}; "
            f"choose one of {valid_feedback_models}"
        )

    if feedback_model == myopic:
        return State(infected=[], frontier=list(state.frontier))

    return state


def prepare_round_bag(
    bag: list[ActionOp],
    state: State,
    graph: GraphInfo,
    task: TaskSpec,
    batch_size: int,
    timestep: int,
) -> list[ActionOp]:
    """
    Ops, ids, this round's own batch cap, and the re-seeding rule.

    Re-seeding an already-active node is handled differently under the two
    feedback models, and the asymmetry is forced rather than chosen:

      * full_adoption — the policy was handed the infected set, so proposing a
        node from it is a bug in the policy. Raise, and let the repair loop see it.
      * myopic — the policy was NOT handed the infected set and cannot check.
        Erroring would make the arm unrunnable; letting the op through would be
        worse, because add_node writes status 1 over NDlib's status 2 and hands a
        spent IC spreader a second round of transmission. So the seed is DROPPED:
        the slot is spent, nothing is activated, and blind re-seeding costs the
        policy budget. That cost is the price of the weaker observation, which is
        exactly what the feedback model is supposed to charge for.
    """
    validate_actions(
        bag,
        graph.num_nodes,
        batch_size,
        task.allowed_ops,
        task.budget_op,
        task.outbreak,
    )

    active = set(state.infected)
    reseeds = [
        action
        for action in bag
        if action.op == "add_node" and int(action.target) in active
    ]

    if not reseeds:
        return bag

    if task.feedback_model != myopic:
        raise StrategyError(
            f"round at t={timestep} seeds node {reseeds[0].target}, which is "
            f"ALREADY ACTIVE in the realized state you were handed. Spending a "
            f"seed on an activated node buys nothing, and reacting to what the "
            f"cascade has already reached is the entire point of seeding in "
            f"rounds. Filter state.infected and state.frontier out of your "
            f"candidates."
        )

    dropped = {id(action) for action in reseeds}

    return [action for action in bag if id(action) not in dropped]


def adaptive_action_fn(
    strategy: Strategy,
    task: TaskSpec,
    graph: GraphInfo,
    batches: list[int],
    stream: object = None,
) -> ActionFn:
    """
    Wrap a generated policy's act() as an ActionFn that fires only on rounds.

    Between rounds the bag is empty and the cascade just runs, which is what
    gives the next round something new to observe.

    Under a graph-edit stream the policy is handed the CURRENT graph rather than
    the base one. That is the whole reason an adaptive arm can answer branch (d)
    and a static plan cannot: being re-queried each round is only useful if what
    you are re-queried about has actually changed.
    """
    schedule = round_schedule(batches, task.round_gap, task.horizon)

    def action_fn(state: State, timestep: int) -> list[ActionOp]:
        batch_size = schedule.get(timestep)
        if batch_size is None:
            return []

        live_graph = graph if stream is None else stream.graph_at(timestep)
        bag = call_strategy(
            strategy.act,
            observed_state(state, task.feedback_model),
            live_graph,
            timestep,
        )

        return prepare_round_bag(bag, state, live_graph, task, batch_size, timestep)

    return action_fn


def describe_schedule(task: TaskSpec, batches: list[int]) -> str:
    """The round structure, in the words the prompt and the results JSON use."""
    timesteps = sorted(round_schedule(batches, task.round_gap, task.horizon))

    return (
        f"{len(batches)} rounds of {batches} seeds at timesteps {timesteps} "
        f"(k={sum(batches)}, gap={task.round_gap}, "
        f"feedback={task.feedback_model})"
    )


def round_spreads(
    infected_counts: list[float], batches: list[int], round_gap: int
) -> list[float]:
    """
    Realized spread at each round boundary, for the per-round cost/benefit curve.

    infected_counts[t] is the count AFTER the step at t-1, so a round firing at
    timestep `t` is read at index `t + 1`; a cascade that terminated early leaves
    fewer entries than rounds, and those rounds simply have no reading.
    """
    return [
        float(infected_counts[index * round_gap + 1])
        for index in range(len(batches))
        if index * round_gap + 1 < len(infected_counts)
    ]
