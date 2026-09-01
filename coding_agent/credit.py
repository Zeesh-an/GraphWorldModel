"""
Counterfactual credit assignment: per-action reward attribution via ablation.

For each action in a per-timestep plan, re-roll the SAME environment with that
single action removed; delta = base_reward - ablated_reward is the final spread
that action is responsible for (~0 = wasted budget). On the world-model
environment every ablation, and a solo cascade per action, is scored through the
batched one-policy-per-sample rollout, which is what makes credit affordable
inside a search loop; other evaluators take the sequential seed-paired path.
For state-dependent strategies (per_step / windowed) the recorded trajectory
bags are replayed as a fixed plan, so credit is an approximation there: the live
policy would have reacted to the ablated cascade.
"""

from functools import partial

from coding_agent.types import ActionOp, State


def planned_action(
    plan: list[list[ActionOp]], state: State, timestep: int
) -> list[ActionOp]:
    """Adapts a static plan into an ActionFn."""
    return plan[timestep] if timestep < len(plan) else []


def counterfactual_credit(
    environment: object,
    plan: list[list[ActionOp]],
    horizon: int,
    budget: int,
    seed: int | None = None,
    creditable_ops: tuple | None = None,
) -> tuple[float, list[dict]]:
    """
    Return (base_reward, one credit entry per creditable action in the plan).

    `creditable_ops` restricts WHICH actions get ablated, not which are replayed.
    Under a graph-edit stream the executed bags also carry exogenous edge ops that
    the policy never chose (coding_agent/stream.py), and ablating one of those
    would report the graph's own churn as something the strategy is responsible
    for. Passing the task's allowed_ops keeps the report to the policy's own
    decisions while the stream stays in the base rollout, where it belongs.
    """
    ablated_plans, entries = [], []
    for timestep, bag in enumerate(plan[: horizon + 1]):
        for action_index, action in enumerate(bag):
            if creditable_ops is not None and action.op not in creditable_ops:
                continue

            # Same plan minus exactly this one action (bags shallow-copied so the original plan is untouched)
            ablated = [list(action_bag) for action_bag in plan]
            del ablated[timestep][action_index]
            ablated_plans.append(ablated)

            entry = {
                "t": int(timestep),
                "op": action.op,
                "target": int(action.target),
            }
            if action.destination is not None:
                entry["destination"] = int(action.destination)
            entries.append(entry)

    if _batched(environment):
        rewards, _ = _batched_plan_rewards(
            environment, [plan] + ablated_plans, horizon, budget, seed
        )
        base_reward = rewards[0]
        for entry, ablated_reward in zip(entries, rewards[1:], strict=True):
            entry["delta"] = round(base_reward - ablated_reward, 3)

        return base_reward, entries

    # Sequential seed-paired path for evaluators without the batched contract
    base_reward = environment.rollout(
        partial(planned_action, plan), horizon, budget, seed=seed
    ).reward

    for entry, ablated in zip(entries, ablated_plans, strict=True):
        ablated_reward = environment.rollout(
            partial(planned_action, ablated), horizon, budget, seed=seed
        ).reward
        entry["delta"] = round(base_reward - ablated_reward, 3)

    return base_reward, entries


# Chunk cap on sample blocks per batched call, so a pct20 credit pass does not
# put (2k+1) * n_samples block-diagonal graphs on the device at once
max_blocks_per_call = 512

# Mirrors methods.base's feedback thresholds; credit cannot import methods.base
# (base imports planned_action from here), so the pair is restated
reached_p = 0.50
unreached_p = 0.10


def _batched(environment: object) -> bool:
    # The one-policy-per-sample rollout contract: the world-model env (and its
    # oracle variant), which also exposes per-sample count curves
    return hasattr(environment, "last_sample_curves") and bool(
        getattr(environment, "n_samples", 0)
    )


def _batched_plan_rewards(
    environment: object,
    plans: list,
    horizon: int,
    budget: int,
    seed: int | None,
) -> tuple[list[float], list[list[float]]]:
    """
    (mean reward, mean count curve) per plan via one-policy-per-sample batching.

    Samples are striped across plans inside one call, so unlike the sequential
    path two plans do NOT share realization streams; at the environment's own
    n_samples per plan the residual noise is the ensemble SE the loop already
    lives with.
    """
    samples_per_plan = environment.n_samples
    chunk = max(1, max_blocks_per_call // samples_per_plan)
    rewards, curves = [], []

    for start in range(0, len(plans), chunk):
        group = plans[start : start + chunk]
        fns = [
            partial(planned_action, group_plan)
            for group_plan in group
            for _ in range(samples_per_plan)
        ]
        environment.rollout(fns, horizon, budget, seed=seed, num_samples=len(fns))

        for index in range(len(group)):
            block = slice(index * samples_per_plan, (index + 1) * samples_per_plan)
            counts = environment.last_sample_counts[block]
            rewards.append(sum(counts) / len(counts))

            block_curves = environment.last_sample_curves[block]
            steps = len(block_curves[0])
            curves.append(
                [
                    sum(curve[step] for curve in block_curves) / len(block_curves)
                    for step in range(steps)
                ]
            )

    return rewards, curves


def augment_solo(
    environment: object,
    plan: list[list[ActionOp]],
    entries: list[dict],
    horizon: int,
    budget: int,
    seed: int | None = None,
    creditable_ops: tuple | None = None,
) -> None:
    """
    Add each credited action's SOLO cascade to its entry: `solo` (its reward
    alone) and `solo_flat_by` (the step its solo cascade stops moving, the
    per-seed stagnation time). Batched environments only; elsewhere a no-op,
    since k more sequential rollouts is exactly the cost credit is escaping.
    """
    if not entries or not _batched(environment):
        return

    # Recover each entry's action object by replaying counterfactual_credit's
    # iteration order, so weights and destinations survive exactly
    actions, bags = [], []
    for timestep, bag in enumerate(plan[: horizon + 1]):
        for action in bag:
            if creditable_ops is not None and action.op not in creditable_ops:
                continue
            actions.append(action)
            bags.append(bag)

    solo_plans = []
    for action, bag in zip(actions, bags, strict=True):
        solo = [action]
        if action.op == "remove_node":
            # A blocked removal travels as a deletion bag; its incident arcs are
            # in the same recorded bag, and a solo removal without them would be
            # a bare block the structured head documents as unreliable
            solo += [
                edge_op
                for edge_op in bag
                if edge_op.op == "remove_edge"
                and (
                    int(edge_op.target) == int(action.target)
                    or (
                        edge_op.destination is not None
                        and int(edge_op.destination) == int(action.target)
                    )
                )
            ]
        solo_plans.append([solo])

    rewards, curves = _batched_plan_rewards(
        environment, solo_plans, horizon, budget, seed
    )
    for entry, reward, curve in zip(entries, rewards, curves, strict=True):
        entry["solo"] = round(reward, 2)
        moved = [
            step
            for step in range(1, len(curve))
            if abs(curve[step] - curve[step - 1]) >= 0.5
        ]
        entry["solo_flat_by"] = int(moved[-1] + 1) if moved else 1


def frontier_bottlenecks(
    environment: object, trajectory, graph, task, top_n: int = 8
) -> str | None:
    """
    Unreached nodes the learned kernel says one more wave would claim.

    Puts the whole reached region on the frontier and asks the model's own
    transition for next-step P(infected): a high-P node the real rollout never
    touched marks a boundary the cascade dies before crossing (or, under a
    containment objective, the front the intervention is about to leak through).
    Only environments exposing the transition kernel can answer (world model and
    oracle); a sampler cannot, and the line is simply absent there.
    """
    marginals = getattr(trajectory, "final_marginals", None)
    if marginals is None or not hasattr(environment, "step_marginals"):
        return None

    reached = [node for node, p in enumerate(marginals) if p >= reached_p]
    unreached = [node for node, p in enumerate(marginals) if p < unreached_p]
    if not reached or not unreached:
        return None

    probs = environment.step_marginals(State(sorted(reached), sorted(reached)))
    ranked = sorted(unreached, key=lambda node: -float(probs[node]))[:top_n]
    kept = [node for node in ranked if float(probs[node]) >= 0.05]
    if not kept:
        return None

    listed = ", ".join(
        f"{node}(d={graph.degree(node)}, P={float(probs[node]):.2f})" for node in kept
    )
    reading = (
        "the front your intervention is closest to leaking through"
        if task.contains
        else "a frontier your cascade dies before crossing"
    )

    return (
        f"KERNEL BOTTLENECKS (the model's own transmission: P(node caught in one "
        f"more wave if the whole reached region were spreading); high P but never "
        f"reached = {reading}: {listed}"
    )


def credit_feedback(environment: object, trajectory, task, graph) -> str | None:
    """
    The full per-action feedback block every method shares: marginal credit
    (leave-one-out), solo cascades with stagnation timing, kernel bottlenecks.
    """
    if task.recovers or task.forecasts or not trajectory.actions:
        return None

    creditable = tuple(task.allowed_ops)
    base_reward, entries = counterfactual_credit(
        environment,
        trajectory.actions,
        task.horizon,
        task.budget,
        creditable_ops=creditable,
    )
    augment_solo(
        environment,
        trajectory.actions,
        entries,
        task.horizon,
        task.budget,
        creditable_ops=creditable,
    )

    report = format_credit_report(base_reward, entries)
    bottleneck = frontier_bottlenecks(environment, trajectory, graph, task)

    return report + (f"\n{bottleneck}" if bottleneck else "")


def format_credit_report(base_reward: float, entries: list[dict]) -> str:
    if not entries:
        return "No actions in the plan to credit."

    lines = [
        f"Per-action counterfactual credit (base spread {base_reward:.2f}; "
        "delta = spread lost if that single action is removed; ~0 = wasted budget):"
    ]
    for entry in entries:
        destination = f"->{entry['destination']}" if "destination" in entry else ""
        solo = (
            f"  solo={entry['solo']:.1f}, flat by t={entry['solo_flat_by']}"
            if "solo" in entry
            else ""
        )
        lines.append(
            f"  t={entry['t']}: {entry['op']}({entry['target']}{destination})  "
            f"delta={entry['delta']:+.2f}{solo}"
        )

    return "\n".join(lines)
