"""
Multi-round IM: `r` separate campaigns of `k` seeds, scored on their union.

Branch (e) of research/adaptive_online_im.md §1.5, and the one genuinely distinct
from adaptive IM: adaptive IM is ONE diffusion observed in stages, multi-round is
`r` separate DIFFUSIONS whose activated sets are unioned. A node activated in
campaign i counts once in the union and may still be re-seeded later.

§2.4e costs this at "reset `frontier` between campaigns, keep `infected` as the
union. A `State` bookkeeping change." That is the right idea and slightly
optimistic about where the bookkeeping goes. Surgically clearing the frontier
mid-episode is not equivalent to a fresh diffusion under either dynamics: under
IC it means rewriting every status-1 node to status 2, and under LT there is no
spent state at all, so an active node keeps contributing to its neighbours'
thresholds forever and the campaigns never actually separate.

So a campaign here is a genuinely fresh inner rollout. That gives real separation
under both dynamics (LT even re-draws its hidden thresholds, which is correct:
they are per-episode), needs no change to either environment, and matches §1.5's
"separate diffusions" exactly. The wrapper only has to carry the union across
them, which is the bookkeeping §2.4e names.

WHY THE UNION IS COMPUTED FROM MARGINALS. Campaigns are independent diffusions
given their seed sets, so for a fixed schedule

    P(v never activated) = prod_i (1 - p_i(v))

with `p_i(v)` the per-node marginal the inner environment already returns. That
identity is EXACT, not an approximation, and it costs nothing extra. It does
condition on the seed sets: a policy that re-plans against the realized union
makes the later sets random, and the estimate then conditions on the realized
path rather than integrating over it. The policy is handed the representative
sample's union for exactly that reason: it is one realization, and it is
labelled as such rather than being passed off as the expectation.
"""

import time
import numpy as np

from coding_agent.types import ActionFn, State, Trajectory


class MultiRoundEnvironment:
    """Wraps any inner environment into `campaigns` separate diffusions."""

    def __init__(self, inner: object, campaigns: int = 3, base_seed: int = 0) -> None:
        if campaigns < 1:
            raise ValueError(f"--campaigns must be >= 1, got {campaigns}")

        self.inner = inner
        self.campaigns = campaigns
        self.base_seed = base_seed
        self.graph = inner.graph
        self.diffusion_model = inner.diffusion_model

    # The cost counters belong to the inner environment: this wrapper runs no
    # episodes of its own, it just calls the inner one `campaigns` times, and
    # forwarding rather than shadowing keeps the arm table honest about that.
    @property
    def episodes_used(self) -> int:
        return getattr(self.inner, "episodes_used", 0)

    @property
    def rollout_calls(self) -> int:
        return getattr(self.inner, "rollout_calls", 0)

    @property
    def evaluator_seconds(self) -> float:
        return getattr(self.inner, "evaluator_seconds", 0.0)

    @property
    def forward_passes(self) -> int:
        return getattr(self.inner, "forward_passes", 0)

    def rollout(
        self, action_fn: ActionFn, horizon: int, budget: int, seed: int | None = None
    ) -> Trajectory:
        start = time.perf_counter()
        seed = self.base_seed if seed is None else seed

        num_nodes = self.graph.num_nodes
        # P(v never activated in any campaign so far)
        never = np.ones(num_nodes, dtype=np.float64)
        realized_union = set()
        campaign_rewards = []
        first = None

        for campaign in range(self.campaigns):
            union_so_far = sorted(realized_union)

            def campaign_action_fn(
                state: State, timestep: int, union: list = union_so_far
            ) -> list:
                # `infected` carries the union, `frontier` stays this campaign's
                # own wave: §2.4e's bookkeeping, and the reason a policy can tell
                # campaign 3 from campaign 1 without a round counter. Default-arg
                # binding, not closure capture, so the loop variable cannot leak.
                merged = State(
                    infected=sorted(set(state.infected) | set(union)),
                    frontier=list(state.frontier),
                )

                return action_fn(merged, timestep)

            # A distinct seed per campaign: reusing one would make every campaign
            # flip identical coins, and the union would collapse to a single
            # campaign's reach no matter how many were run
            trajectory = self.inner.rollout(
                campaign_action_fn, horizon, budget, seed=seed + campaign
            )
            campaign_rewards.append(trajectory.reward)

            if trajectory.final_marginals is not None:
                never *= 1.0 - np.asarray(trajectory.final_marginals, dtype=np.float64)

            realized_union |= set(trajectory.states[-1].infected)

            if first is None:
                first = trajectory

        union_marginals = 1.0 - never
        reward = float(union_marginals.sum())
        elapsed = time.perf_counter() - start

        return Trajectory(
            # The first campaign's trajectory is the representative one; every
            # aggregate below spans all of them
            states=first.states,
            actions=first.actions,
            reward=reward,
            infected_counts=first.infected_counts,
            cost={
                **first.cost,
                "env": f"multi_round[{first.cost.get('env', '?')}]",
                "campaigns": self.campaigns,
                "campaign_rewards": [round(value, 4) for value in campaign_rewards],
                # SE of the union is not the SE of one campaign, and this wrapper
                # runs one union per call, so there is no spread to take one over.
                # Reported as 0 rather than a per-campaign number that would be
                # read as the union's noise band by paired_delta.
                "reward_se": 0.0,
                "rollout_seconds": elapsed,
            },
            final_marginals=union_marginals.round(3).tolist(),
            spread_curve=first.spread_curve,
        )
