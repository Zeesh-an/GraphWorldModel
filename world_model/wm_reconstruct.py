"""
Arm A: DITTO's decoder, with our learned kernel substituted for its mean-field `beta`.

`research/cascade_reconstruction.md` §2.3 names the tempting move and §2.9 makes it
the CONTROL rather than the method: DITTO estimates a single scalar `beta_hat` by
mean field from one snapshot, we have a per-edge `q(u -> v)` supervised against MC
marginals at every step, so swapping ours in and reporting the delta is a
second-order result inside someone else's frame. 6-vs-A is the methodological
claim, and §2.11 risk 6 is why this file is built carefully rather than
conveniently: DITTO beats a supervised model trained with the TRUE `beta` on BA-SIR
and ER-SIR, so a weak arm A makes the comparison meaningless.

Three things are reproduced faithfully and one is deliberately not:

  * **Metropolis-Hastings over histories** [verified, §4.1]: the sampler, at
    `O(T(n log n + m))` per sample.
  * **The barycenter output**: DITTO's central design decision is that plain MLE
    over histories is unstable because the likelihood is nearly flat in `beta_hat`,
    which is what motivates estimating posterior EXPECTED hitting times instead of
    the single most likely history. So this returns the posterior mean hitting
    time per node, rounded, not the MAP sample.
  * **The DASH setting**: neither the diffusion parameters nor the sources are
    given. Both fall out of the sampler: a node's hitting time of 0 IS the claim
    that it is a source.
  * **NOT the learned proposal `Q_theta`.** DITTO trains an unsupervised GNN to
    propose; ours is hand-written (shift one hitting time, toggle one node). That
    is stated here rather than buried, because it is the one respect in which this
    control is weaker than the paper, and §2.3 notes the premise behind DITTO's
    objective may not even hold for a per-edge kernel, which is exactly the
    question decoder SEARCH is supposed to answer and this arm is not.

The kernel is reached through whatever `predict(infected, frontier) -> (N,)` the
caller binds, which under `decode_free@world_model` is the metered
`WorldModelEnvironment.step_marginals`. That is what puts arm A's cost in the same
`kernel_calls` column as every other arm's (§8.5.3).
"""

import math

import numpy as np

from coding_agent.tools.reconstruction_algorithms import (
    _merge_observed,
    delayed_bfs,
    finalize,
)

# MH samples drawn per instance. §2.4.2 puts a real decoder at ~10^4 kernel
# evaluations per instance; this is `proposals x horizon` of them, so 400 is a
# smoke value and the first number to raise before quoting arm A as DITTO-parity.
default_proposals = 400
# Fraction of the chain discarded before the barycenter starts accumulating
default_burn_in = 0.3
# Probability a proposal TOGGLES a node in or out of the infected set rather than
# shifting a hitting time. Both moves are needed under DASH: the first infers
# WHEN, the second infers WHO.
toggle_probability = 0.3
# Candidates a toggle may reach: the current decode's boundary rather than the
# whole graph, which keeps the chain mixing on a large sparse graph
max_toggle_candidates = 400

log_floor = 1e-12


def _history_logprob(
    times: dict[int, int], horizon: int, predict, num_nodes: int
) -> float:
    """
    `sum_t log p(s_{t+1} | s_t, G)` for one proposed history, under the bound kernel.

    The quantity §8.1 marks "**nobody**: this is ours to add", because it needs a
    kernel to evaluate under and no method in that literature has one. It is also
    what makes this sampler a sampler rather than a heuristic: every accept/reject
    decision is a likelihood ratio, not a score comparison.
    """
    waves = {}
    for node, step in times.items():
        waves.setdefault(int(step), []).append(int(node))

    total = 0.0
    infected = list(waves.get(0, []))

    for step in range(1, horizon + 1):
        frontier = waves.get(step - 1, [])
        if not frontier:
            break

        marginal = np.clip(
            np.asarray(predict(infected, frontier), dtype=np.float64),
            log_floor,
            1.0 - log_floor,
        )

        active = np.zeros(num_nodes, dtype=bool)
        active[infected] = True
        gained = np.zeros(num_nodes, dtype=bool)
        gained[waves.get(step, [])] = True

        susceptible = ~active
        total += float(
            np.log(marginal[susceptible & gained]).sum()
            + np.log1p(-marginal[susceptible & ~gained]).sum()
        )
        infected = infected + waves.get(step, [])

    return total


def _toggle_candidates(graph, times: dict[int, int], observation) -> list[int]:
    """
    Nodes on the boundary of the current decode: the only ones worth toggling.

    An interior node is already explained and a far-away one cannot be reached in
    one move, so proposing either wastes a kernel evaluation. The boundary is the
    frontier of the current infected set plus the non-reported members of it.
    """
    boundary = set()

    for node in times:
        for neighbour in graph.out_neighbors(node):
            if neighbour not in times and observation.is_visible(neighbour):
                boundary.add(int(neighbour))

    # ...plus the inferred (not reported) members, which the chain may drop again
    boundary |= {
        int(node) for node in times if node not in observation.reported and times[node] > 0
    }

    return sorted(boundary)[:max_toggle_candidates]


def barycenter_decode(
    graph,
    observation,
    horizon: int,
    predict,
    proposals: int = default_proposals,
    burn_in: float = default_burn_in,
    seed: int = 0,
) -> tuple[dict[int, tuple[int, int | None]], dict]:
    """
    DITTO's reconstruction: MH over histories, then the POSTERIOR MEAN hitting time.

    Returns `(decoded trajectory, diagnostics)`. The chain starts from
    `delayed_bfs`: a sampler needs a feasible point and the classical decoder is
    the cheapest one, and every accepted state contributes its hitting times to a
    running mean, which is the barycenter DITTO reports instead of an MLE.

    The reported nodes are never moved: a report is ground truth about that node,
    so the chain explores only the part of the history that is actually unknown.
    """
    rng = np.random.default_rng(seed)
    initial = delayed_bfs(graph, observation, horizon)

    if not initial:
        return {}, {"proposals": 0, "accepted": 0, "kernel_calls": 0}

    times = {node: int(step) for node, (step, _) in initial.items()}
    anchored = {int(node) for node in observation.reported}
    observed_times = observation.times

    current = _history_logprob(times, horizon, predict, graph.num_nodes)
    burn = int(max(1, proposals) * burn_in)

    # Running posterior mean over hitting times, and the posterior probability a
    # node was infected at all: both are what a barycenter estimate reports
    total_time = np.zeros(graph.num_nodes, dtype=np.float64)
    total_hits = np.zeros(graph.num_nodes, dtype=np.float64)
    samples = 0
    accepted = 0

    for step in range(max(1, proposals)):
        proposal = dict(times)

        if rng.random() < toggle_probability:
            candidates = _toggle_candidates(graph, proposal, observation)
            if not candidates:
                continue

            node = int(candidates[int(rng.integers(len(candidates)))])
            if node in proposal:
                # A reported node may never be dropped
                if node in anchored:
                    continue

                proposal.pop(node)
            else:
                earlier = [
                    proposal[other]
                    for other in graph.in_neighbors(node)
                    if other in proposal
                ]
                if not earlier:
                    continue

                proposal[node] = max(0, min(min(earlier) + 1, horizon))
        else:
            movable = [node for node in proposal if node not in observed_times]
            if not movable:
                break

            node = int(movable[int(rng.integers(len(movable)))])
            shift = 1 if rng.random() < 0.5 else -1
            moved = max(0, min(proposal[node] + shift, horizon))

            if moved == proposal[node]:
                continue

            proposal[node] = moved

        candidate = _history_logprob(proposal, horizon, predict, graph.num_nodes)

        # Symmetric proposal, so the Hastings ratio is the likelihood ratio alone
        if math.log(max(rng.random(), log_floor)) < candidate - current:
            times, current = proposal, candidate
            accepted += 1

        if step >= burn:
            samples += 1
            for node, value in times.items():
                total_time[node] += value
                total_hits[node] += 1.0

    if not samples:
        return initial, {
            "proposals": proposals,
            "accepted": accepted,
            "samples": 0,
        }

    # The barycenter: a node is in the reconstruction when the posterior says it
    # was infected more often than not, at its posterior MEAN hitting time
    kept = {
        int(node)
        for node in np.flatnonzero(total_hits >= 0.5 * samples)
        if observation.is_visible(int(node))
    } | anchored

    mean_times = {
        node: int(round(total_time[node] / max(total_hits[node], 1.0)))
        for node in kept
    }

    decoded = finalize(
        graph, _merge_observed(mean_times, observation, horizon), observation, horizon
    )

    return decoded, {
        "proposals": proposals,
        "accepted": accepted,
        "samples": samples,
        "acceptance_rate": round(accepted / max(1, proposals), 4),
    }


class DittoDecoder:
    """
    The `reconstruct()`-shaped object arm A is scored as.

    Presented as a Strategy so nothing downstream special-cases it: it carries a
    `source_script` (the procedure, as source, so the report shows what actually
    ran) and the harness binds `self.step_marginals` onto it exactly as it does for
    a generated program, which is what makes arm A's kernel-call count directly
    comparable to arm 6's.
    """

    def __init__(
        self,
        proposals: int = default_proposals,
        burn_in: float = default_burn_in,
        seed: int = 0,
    ) -> None:
        self.proposals = proposals
        self.burn_in = burn_in
        self.seed = seed
        self.kernel_calls = 0
        self.accepted = 0
        self.instances = 0
        self.source_script = (
            f"# Arm A: research/cascade_reconstruction.md §2.9, §2.3\n"
            f"# DITTO (KDD 2023) with our learned per-edge kernel in place of its\n"
            f"# mean-field scalar beta-hat:\n"
            f"#   1. initialize from delayed-BFS (a feasible history)\n"
            f"#   2. Metropolis-Hastings over histories, {proposals} proposals,\n"
            f"#      accepting on the ratio of trajectory log-likelihoods under\n"
            f"#      f_theta rather than on a score\n"
            f"#   3. report the POSTERIOR MEAN hitting time per node (DITTO's\n"
            f"#      barycenter), not the MAP sample\n"
            f"# Proposals: shift one hitting time, or toggle one boundary node.\n"
            f"# NOT reproduced: DITTO's unsupervised GNN proposal Q_theta.\n"
            f"# This is the component swap §2.3 warns against: hence a CONTROL.\n"
        )

    def reconstruct(self, graph, observation, horizon: int) -> dict:
        def predict(infected, frontier):
            self.kernel_calls += 1

            return self.step_marginals(infected, frontier)

        decoded, info = barycenter_decode(
            graph,
            observation,
            horizon,
            predict,
            proposals=self.proposals,
            burn_in=self.burn_in,
            seed=self.seed + self.instances,
        )
        self.accepted += info.get("accepted", 0)
        self.instances += 1

        return decoded
