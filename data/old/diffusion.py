"""
Diffusion Simulations
=====================

Independent Cascade (IC) and Linear Threshold (LT) diffusion models.
Used for data generation and evaluation across IM, source localization, and cascade reconstruction tasks.
"""

import numpy as np


def simulate_IC(seed_set: list[int], out_adj: dict, N: int, max_steps: int = 50):
    """
    Independent Cascade simulation.

    At each step, every newly activated node tries to activate each
    inactive out-neighbor independently with probability p(u→v).

    Returns
    -------
    cascade_trace: list of sets — nodes newly activated at each timestep
                    cascade_trace[0] == seed_set (t=0)
    final_spread: int — total number of activated nodes
    """
    # Initialize with the seed set
    active = set(seed_set)
    newly_active = set(seed_set)
    cascade_trace = [set(seed_set)]

    for _ in range(max_steps):
        next_wave = set()

        # For every newly active node u, attempt to activate each neighbor node v with probability p
        for u in newly_active:
            for v, p in out_adj.get(u, []):
                if v not in active and np.random.random() < p:
                    next_wave.add(v)

        if not next_wave:
            break

        # Merge the next wave into the active set and record it in the cascade trace
        active |= next_wave
        newly_active = next_wave
        cascade_trace.append(next_wave)

    return cascade_trace, len(active)  # len(active) = total spread count


def simulate_LT(seed_set: list[int], in_adj: dict, N: int, max_steps: int = 50):
    """
    Linear Threshold simulation.

    Each node v has threshold θ_v ~ Uniform[0, 1].
    v activates when Σ_{active u ∈ N_in(v)} w(u,v) ≥ θ_v.

    Returns
    -------
    cascade_trace: list of sets
    final_spread: int
    """
    # Get a random threshold for each node
    thresholds = np.random.uniform(0, 1, N)

    # Initialize with the seed set
    active = set(seed_set)
    newly_active = set(seed_set)
    cascade_trace = [set(seed_set)]

    # Track accumulated influence per node
    influence_sum = np.zeros(N, dtype=np.float32)

    # For each seed node, add its edge weight to each neighbor's influence score
    for u in seed_set:
        for v, w in in_adj.get(u, []):
            if v not in active:
                influence_sum[v] += w

    for _ in range(max_steps):
        next_wave = set()

        # Find all inactive nodes whose accumulated influence meets their threshold for wave
        for v in range(N):
            if v not in active and influence_sum[v] >= thresholds[v]:
                next_wave.add(v)

        if not next_wave:
            break

        # Merge wave into the active set and record it in the cascade trace
        active |= next_wave
        newly_active = next_wave
        cascade_trace.append(next_wave)

        # Propagate the influence from newly active nodes to their inactive neighbors
        for u in newly_active:
            for v, w in in_adj.get(u, []):
                if v not in active:
                    influence_sum[v] += w

    return cascade_trace, len(active)  # len(active) = total spread count


def estimate_spread(
    seed_set: list[int],
    out_adj: dict,
    in_adj: dict,
    N: int,
    model: str = "IC",
    mc_runs: int = 100,
):
    """
    Monte Carlo estimate of influence spread.
    Returns mean and std over mc_runs simulations.
    """
    spreads = []
    all_traces = []

    sim_fn = simulate_IC if model == "IC" else simulate_LT
    kwargs = dict(out_adj=out_adj, N=N) if model == "IC" else dict(in_adj=in_adj, N=N)

    for _ in range(mc_runs):
        # Run each simulation, collect spread values, and record traces
        trace, spread = sim_fn(seed_set, **kwargs)
        spreads.append(spread)
        all_traces.append(trace)

    return np.mean(spreads), np.std(spreads), all_traces
