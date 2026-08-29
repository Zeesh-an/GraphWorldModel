"""
Arm A: per-instance gradient descent on a relaxed source vector, no LLM anywhere.

The control that program search is measured against
(`research/source_localization.md` §2.6 arm A, §9.3 step 3). It implements the
same `OuterLoopMethod` contract as `one_shot` and `evolve` so it lands in the same
results table, the same summary row and the same figures, but it has no
conversation, no population and no refinement: the "program" it returns is a fixed
numerical procedure, and its only iteration is Adam inside `wm_sl.invert`.

Two properties keep the comparison honest and both are worth stating:

  * **It only makes sense against a differentiable forward model.** The loss is
    `||y - f_theta(x~)||^2`, so it needs gradients through `f_theta`, which the
    `@monte_carlo` and `@native` evaluators cannot supply. Requesting those raises
    rather than silently substituting something else, because "arm A under a
    sampling oracle" is a method nobody has and a row nobody should read.
  * **The prior is fit on the TRAINING split only.** SL-VAE learns `p(x)` from
    observed source sets, so withholding it would strawman the control; fitting it
    on the evaluation split would leak the answer. Both mistakes flip the sign of
    the headline claim.
"""

import time
from tqdm import tqdm

from coding_agent.agent import CodingAgent
from coding_agent.executor import StrategyError
from coding_agent.localization import (
    SourceInstance,
    evaluate_localizer,
    episode_budget,
)
from coding_agent.methods.base import OuterLoopMethod
from coding_agent.types import GraphInfo, Strategy, TaskSpec, Trajectory

# Evaluators whose rollout is not differentiable in the source vector
non_differentiable_evaluators = ("monte_carlo", "native")


class GradientLocalizer:
    """
    The `localize()`-shaped object the sweep scores, wrapping `wm_sl.invert`.

    Presented as a Strategy so nothing downstream special-cases arm A: it carries a
    `source_script` (the procedure, as source, so the report shows what actually
    ran) and exposes `source_scores`, which for this arm is a genuine continuous
    ranking rather than the rank-derived stand-in a set-only method falls back to.
    """

    def __init__(
        self,
        model: object,
        graph_input: object,
        degree_channel: object,
        horizon: int,
        steps: int,
        lr: float,
        cardinality_weight: float,
        prior: object,
        prior_weight: float,
        prior_kind: str,
    ) -> None:
        self.model = model
        self.graph_input = graph_input
        self.degree_channel = degree_channel
        self.horizon = horizon
        self.steps = steps
        self.lr = lr
        self.cardinality_weight = cardinality_weight
        self.prior = prior
        self.prior_weight = prior_weight
        self.prior_kind = prior_kind
        self.gradient_steps = 0
        # Cached per observation so `source_scores` reports the vector that
        # produced the set rather than re-solving with a different random path
        self._scores = None
        self.source_script = (
            f"# Arm A: research/source_localization.md §2.6, §2.2\n"
            f"# Freeze f_theta, relax x to x~ in [0,1]^N, run Adam on\n"
            f"#     ||y - f_theta(x~, G)||^2 + {cardinality_weight} * (sum(x~) - k)^2"
            + (f" - {prior_weight} * log p(x~)\n" if prior is not None else "\n")
            + f"# steps={steps}, lr={lr}, prior={prior_kind}, unroll={horizon}\n"
            f"# Then take the top-k entries of the converged x~.\n"
            f"# This is SL-VAE's procedure with our world model as its likelihood,\n"
            f"# which the seed paper reports is a no-op swap: hence a CONTROL.\n"
        )

    def localize(self, graph: GraphInfo, observation, budget: int) -> list[int]:
        from world_model.wm_sl import invert

        sources, scores, info = invert(
            self.model,
            self.graph_input,
            self.degree_channel,
            observation,
            budget=budget,
            steps=self.steps,
            horizon=self.horizon,
            lr=self.lr,
            cardinality_weight=self.cardinality_weight,
            prior=self.prior,
            prior_weight=self.prior_weight,
        )
        self._scores = scores
        self.gradient_steps += info["gradient_steps"]

        return sources

    def source_scores(self, graph: GraphInfo, observation):
        if self._scores is None:
            self.localize(graph, observation, 1)

        return self._scores


class GradientInversion(OuterLoopMethod):
    def __init__(
        self,
        wm_results_json: str | None,
        instances: list[SourceInstance],
        budget_mode: str = episode_budget,
        steps: int = 200,
        lr: float = 0.1,
        cardinality_weight: float = 0.05,
        prior_kind: str = "vae",
        prior_weight: float = 1.0,
        prior_epochs: int = 300,
        training_sources: list[list[int]] | None = None,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.wm_results_json = wm_results_json
        self.instances = instances
        self.budget_mode = budget_mode
        self.steps = steps
        self.lr = lr
        self.cardinality_weight = cardinality_weight
        self.prior_kind = prior_kind
        self.prior_weight = prior_weight
        self.prior_epochs = prior_epochs
        self.training_sources = training_sources or []
        self.device = device
        self.seed = seed
        # Same shape as every other method's, so run.py's convergence plot and
        # summary row need no branch. One entry: this method does not iterate.
        self.history = []
        self.effective_budget = None

    def optimize(
        self, agent: CodingAgent, environment: object, task: TaskSpec, graph: GraphInfo
    ) -> tuple[Strategy, Trajectory]:
        import torch

        from world_model.wm_sl import (
            build_graph_tensors,
            load_world_model,
            no_prior,
            train_source_prior,
            vae_prior,
        )

        if not task.recovers:
            raise ValueError(
                f"the gradient method inverts a forward model and only applies to a "
                f"recover task; {task.task!r} is not one"
            )

        if self.wm_results_json is None:
            raise ValueError(
                "arm A needs a trained world model to invert: pass "
                "--wm-results-json, or run the train stage. It is not defined "
                "against a sampling evaluator: the loss is "
                "||y - f_theta(x~)||^2 and needs gradients through f_theta."
            )

        self.effective_budget = task.budget
        start = time.perf_counter()

        model, wm_config = load_world_model(self.wm_results_json, self.device)
        # A w-hidden checkpoint must be inverted against ones, not the true
        # p(u->v): handed the true weights it is not the model that was trained
        graph_input, degree_channel = build_graph_tensors(
            graph,
            task.diffusion_model,
            torch.device(self.device),
            hide_edge_weights=bool(wm_config.get("hide_edge_weights", False)),
        )

        prior = None
        if self.prior_kind == vae_prior:
            if not self.training_sources:
                raise ValueError(
                    "the VAE prior is fit on the TRAINING split's source sets and "
                    "none were supplied; pass --sl-prior none for the SL-VAE (a) "
                    "ablation, or generate a train split"
                )

            tqdm.write(
                f"[gradient] fitting the source prior on "
                f"{len(self.training_sources)} training episodes"
            )
            prior = train_source_prior(
                self.training_sources,
                graph.num_nodes,
                epochs=self.prior_epochs,
                device=self.device,
                seed=self.seed,
            )
        elif self.prior_kind != no_prior:
            raise ValueError(
                f"unknown prior {self.prior_kind!r}; choose {no_prior} (SL-VAE (a): "
                f"forward model + descent) or {vae_prior} (the full method)"
            )

        localizer = GradientLocalizer(
            model,
            graph_input,
            degree_channel,
            horizon=task.horizon,
            steps=self.steps,
            lr=self.lr,
            cardinality_weight=self.cardinality_weight,
            prior=prior,
            prior_weight=self.prior_weight,
            prior_kind=self.prior_kind,
        )

        tqdm.write(
            f"[gradient] inverting {len(self.instances)} episodes: {self.steps} Adam "
            f"steps each, prior={self.prior_kind}"
        )
        trajectory, plan_seconds = evaluate_localizer(
            localizer, environment, task, graph, self.instances, self.budget_mode
        )

        # Recorded on the trajectory so the cost block reports what THIS arm pays
        # per instance: gradient steps, not forward-oracle calls (§8.5.3)
        trajectory.cost["gradient_steps"] = localizer.gradient_steps
        trajectory.cost["gradient_steps_per_instance"] = round(
            localizer.gradient_steps / max(1, len(self.instances)), 3
        )
        trajectory.cost["sl_prior"] = self.prior_kind

        self.history.append(
            {
                "iteration": 1,
                "reward": trajectory.reward,
                "best": trajectory.reward,
                "operator": "gradient",
                "plan_seconds": round(plan_seconds, 3),
                "rollout_seconds": round(time.perf_counter() - start, 3),
            }
        )
        tqdm.write(f"[gradient] F1={trajectory.reward:.4f}")

        if trajectory.reward is None:
            raise StrategyError("gradient inversion produced no scored instance")

        return localizer, trajectory
