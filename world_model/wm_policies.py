"""
Action policies for the off-policy (OOD) rollout test.

Data generation injects actions UNIFORMLY AT RANDOM over the enabled ops. A coding
agent does not: it seeds hubs, it blocks cut vertices, it concentrates its budget.
So "the rollout is faithful" measured under the recorded action sequence is a
statement about the training action distribution, not about the distribution the
model will be queried at when it is used as the agent's inner loop.

Each policy maps (store_entry, episode_records, rng) -> one action bag per step,
depending only on the graph and the step index. State-independence is deliberate:
both the true simulator and the model must replay the SAME sequence for the
comparison to isolate model error, and a state-conditioned policy would diverge
between the two as soon as their states did.

Node ops only — `rollout_ensemble` reconstructs the adjacency from the recorded
edge ops, so a policy emitting different ones would desynchronise it.
"""

import numpy as np

# Registry key -> factory. `recorded` is represented by None (rollout_ensemble's
# default) rather than a policy, so the untouched path stays byte-identical.
recorded = "recorded"


def _degrees(store_entry: dict) -> np.ndarray:
    edge_index = store_entry["edge_index"]
    degrees = np.zeros(store_entry["num_nodes"], dtype=np.float64)

    if edge_index.size:
        np.add.at(degrees, edge_index[0], 1.0)
        np.add.at(degrees, edge_index[1], 1.0)

    return degrees


def null_policy(store_entry: dict, episode_records: list[dict], rng) -> list[list]:
    """No actions at all: pure diffusion from the episode's own starting state.

    The floor case. If fidelity collapses here, the model is leaning on the action
    channels to compensate for a mis-learned diffusion term.
    """
    return [[] for _ in episode_records]


def degree_seed_policy(
    store_entry: dict, episode_records: list[dict], rng
) -> list[list]:
    """Seed the t-th highest-degree node at every step — the agent-like extreme.

    Maximally off-distribution against uniform-random injection, and close to what
    a generated `plan_horizon()` actually emits, since degree is the first
    heuristic every LLM-written IM strategy reaches for.
    """
    order = np.argsort(-_degrees(store_entry))

    return [
        [{"op": "add_node", "target": int(order[step % order.size])}]
        for step in range(len(episode_records))
    ]


def random_seed_policy(
    store_entry: dict, episode_records: list[dict], rng
) -> list[list]:
    """Seed a uniformly random node every step.

    The control for `degree_seed_policy`: same op, same one-per-step rate, no
    structural targeting. A gap between the two isolates fidelity loss caused by
    WHERE the actions land rather than by how many there are.
    """
    num_nodes = store_entry["num_nodes"]

    return [
        [{"op": "add_node", "target": int(rng.integers(num_nodes))}]
        for _ in episode_records
    ]


def block_hubs_policy(
    store_entry: dict, episode_records: list[dict], rng
) -> list[list]:
    """Remove the t-th highest-degree node at every step (containment's extreme).

    Emits a bare `remove_node`, not a full deletion bag: the incident `remove_edge`
    ops would need replaying onto the adjacency, which this path does not do. Under
    `blocked` semantics the simulator's `_enforce_blocked` still holds the node out
    of the dynamics, so the comparison stays valid — the graph simply keeps the
    stripped arcs on both sides, identically.
    """
    order = np.argsort(-_degrees(store_entry))

    return [
        [{"op": "remove_node", "target": int(order[step % order.size])}]
        for step in range(len(episode_records))
    ]


policies = {
    "null": null_policy,
    "degree_seed": degree_seed_policy,
    "random_seed": random_seed_policy,
    "block_hubs": block_hubs_policy,
}


def resolve(names: list[str]) -> dict:
    """Look up policies by name, failing loudly on a typo rather than silently skipping."""
    unknown = [name for name in names if name not in policies]

    if unknown:
        raise ValueError(
            f"unknown action policy {unknown}; choose from {sorted(policies)}"
        )

    return {name: policies[name] for name in names}
