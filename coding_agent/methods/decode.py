"""
Arm A for cascade reconstruction: DITTO's decoder over our kernel, no LLM anywhere.

The control that program search is measured against
(`research/cascade_reconstruction.md` §2.9 arm A, §2.3). It implements the same
`OuterLoopMethod` contract as `one_shot` and `evolve` so it lands in the same
results table, the same summary row and the same figures, but it has no
conversation, no population and no refinement: the "program" it returns is a fixed
Metropolis-Hastings sampler, and its only iteration is inside `barycenter_decode`.

Two properties keep the comparison honest and both are worth stating:

  * **It needs a transition KERNEL, not merely an evaluator.** Every accept/reject
    decision is a ratio of trajectory log-likelihoods, so the `@native` binding,
    which has no kernel by design: cannot run it, and requesting that raises
    rather than silently substituting something else. `@monte_carlo` can run it and
    is the honest sampling comparison; §2.11 records that it may not COMPLETE at a
    useful proposal count, and that finding needs stating as one rather than as a
    missing row.
  * **It is built to be strong.** §2.11 risk 6: DITTO beats a supervised model
    trained with the true `beta` on two of eight rows, so a weak arm A makes 6-vs-A
    meaningless. The one respect in which this is weaker than the paper: the
    hand-written proposal rather than DITTO's learned `Q_theta`: is documented in
    `world_model/wm_reconstruct.py` rather than buried.
"""

import time

from tqdm import tqdm

from coding_agent.agent import CodingAgent
from coding_agent.executor import StrategyError
from coding_agent.methods.base import OuterLoopMethod, attach_context
from coding_agent.reconstruction import CascadeInstance, evaluate_reconstructor
from coding_agent.types import GraphInfo, Strategy, TaskSpec, Trajectory
from world_model.wm_reconstruct import DittoDecoder, default_burn_in, default_proposals

# The one evaluator with no transition kernel at all, by design
kernel_free_evaluators = ("native",)


class BarycenterDecoding(OuterLoopMethod):
    def __init__(
        self,
        instances: list[CascadeInstance],
        proposals: int = default_proposals,
        burn_in: float = default_burn_in,
        tree_weight: float = 0.6,
        seed: int = 42,
    ) -> None:
        self.instances = instances
        self.proposals = proposals
        self.burn_in = burn_in
        self.tree_weight = tree_weight
        self.seed = seed
        # Same shape as every other method's, so run.py's convergence plot and
        # summary row need no branch. One entry: this method does not iterate.
        self.history = []
        self.effective_budget = None

    def optimize(
        self, agent: CodingAgent, environment: object, task: TaskSpec, graph: GraphInfo
    ) -> tuple[Strategy, Trajectory]:
        if not task.decodes:
            raise ValueError(
                f"the decode method reconstructs a hidden TRAJECTORY and only "
                f"applies to a reconstruction task; {task.task!r} is not one"
            )

        if not task.forward_model:
            raise ValueError(
                "arm A is Metropolis-Hastings on a likelihood RATIO, so it needs a "
                "transition kernel; the @native condition has none by design. Use "
                "decode_free@world_model (or @oracle for the analytic IC form); "
                "arm 3 is an agent condition, not this one."
            )

        self.effective_budget = task.budget
        start = time.perf_counter()

        decoder = DittoDecoder(
            proposals=self.proposals, burn_in=self.burn_in, seed=self.seed
        )
        # Binds `self.step_marginals` onto the decoder exactly as it does for a
        # generated program, which is what makes arm A's kernel-call count
        # comparable to arm 6's rather than measured on a private path
        attach_context(decoder, task, environment)

        tqdm.write(
            f"[decode] MH over {len(self.instances)} cascades: {self.proposals} "
            f"proposals each, burn-in {self.burn_in:.0%}, barycenter output"
        )
        trajectory, plan_seconds = evaluate_reconstructor(
            decoder, environment, task, graph, self.instances, self.tree_weight
        )

        # Recorded on the trajectory so the cost block reports what THIS arm pays
        # per instance. `kernel_calls` is already counted by the bound oracle; this
        # is the sampler's own accounting beside it.
        trajectory.cost["mcmc_proposals"] = self.proposals
        trajectory.cost["mcmc_proposals_per_instance"] = self.proposals
        trajectory.cost["mcmc_accepted"] = decoder.accepted
        trajectory.cost["mcmc_acceptance_rate"] = round(
            decoder.accepted / max(1, self.proposals * max(1, decoder.instances)), 4
        )

        self.history.append(
            {
                "iteration": 1,
                "reward": trajectory.reward,
                "best": trajectory.reward,
                "operator": "decode",
                "plan_seconds": round(plan_seconds, 3),
                "rollout_seconds": round(time.perf_counter() - start, 3),
            }
        )
        tqdm.write(
            f"[decode] score={trajectory.reward:.4f} "
            f"(acceptance {trajectory.cost['mcmc_acceptance_rate']:.1%}, "
            f"{trajectory.cost['kernel_calls']} kernel calls)"
        )

        if trajectory.reward is None:
            raise StrategyError("barycenter decoding produced no scored instance")

        return decoder, trajectory
